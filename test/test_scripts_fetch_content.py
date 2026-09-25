import gzip
import hashlib
import http.server
import io
import json
import lzma
import os
import pathlib
import shutil
import stat
import sys
import tarfile
import threading
import urllib.error
import urllib.request
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from unittest.mock import MagicMock

import pytest

import taskgraph


@pytest.fixture(scope="module")
def fetch_content_mod():
    spec = spec_from_loader(
        "fetch-content",
        SourceFileLoader(
            "fetch-content",
            os.path.join(
                os.path.dirname(taskgraph.__file__), "run-task", "fetch-content"
            ),
        ),
    )
    assert spec
    assert spec.loader
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RangeServer(http.server.ThreadingHTTPServer):
    """Serves a single blob, optionally honouring range requests.

    ``faults`` maps a Range header value to a list of faults to inject, one
    per request for that range: "error" answers 500, "truncate" sends half of
    the body and closes the connection, and "slow" pauses half way through.
    """

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self, content, ranges=True, accept_ranges="bytes", gzip=False, log=None
    ):
        super().__init__(("127.0.0.1", 0), RangeHandler)
        self.content = content
        self.ranges = ranges
        self.accept_ranges = accept_ranges
        self.gzip = gzip
        self.faults = {}
        self.requests = []
        self.log = log
        self.lock = threading.Lock()

    @property
    def url(self):
        return "http://{}:{}/blob".format(*self.server_address)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_GET(self):
        content = self.server.content
        requested = self.headers.get("Range")
        with self.server.lock:
            self.server.requests.append(requested)
            if self.server.log is not None:
                self.server.log.append((self.server, requested))
            faults = self.server.faults.get(requested)
            fault = faults.pop(0) if faults else None

        if fault == "error":
            self.send_response(500)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        start, end = 0, len(content) - 1
        partial = False
        if requested and self.server.ranges:
            start, _, last = requested.partition("=")[2].partition("-")
            start, end = int(start), int(last)
            if start >= len(content):
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{len(content)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            end = min(end, len(content) - 1)
            partial = True

        body = content[start : end + 1]
        self.send_response(206 if partial else 200)
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(content)}")
        elif self.server.gzip and "gzip" in self.headers.get("Accept-Encoding", ""):
            body = gzip.compress(body)
            self.send_header("Content-Encoding", "gzip")
        if self.server.accept_ranges:
            self.send_header("Accept-Ranges", self.server.accept_ranges)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if fault == "truncate":
            self.wfile.write(body[: len(body) // 2])
            self.close_connection = True
            return
        if fault == "slow":
            self.wfile.write(body[: len(body) // 2])
            self.wfile.flush()
            threading.Event().wait(1)
            self.wfile.write(body[len(body) // 2 :])
            return
        self.wfile.write(body)


@pytest.fixture
def serve():
    servers = []

    def inner(content, **kwargs):
        server = RangeServer(content, **kwargs)
        threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        ).start()
        servers.append(server)
        return server

    yield inner

    for server in servers:
        server.shutdown()
        server.server_close()


@pytest.fixture
def sliced(monkeypatch, fetch_content_mod):
    """Enable slicing with sizes small enough to exercise in a test."""

    def inner(min_bytes=1024, piece_bytes=1000, connections=4):
        monkeypatch.setenv("TASKGRAPH_FETCH_SLICES", "8")
        monkeypatch.setenv("TASKGRAPH_FETCH_SLICE_MIN_BYTES", str(min_bytes))
        monkeypatch.setenv("TASKGRAPH_FETCH_PIECE_BYTES", str(piece_bytes))
        monkeypatch.setenv("TASKGRAPH_FETCH_CONNECTIONS", str(connections))
        monkeypatch.setattr(fetch_content_mod, "_connection_pool", None)
        monkeypatch.setattr(fetch_content_mod.time, "sleep", lambda s: None)
        return fetch_content_mod.get_connection_pool()

    return inner


def test_sliced_download(tmp_path, fetch_content_mod, serve, sliced):
    content = os.urandom(4096)
    server = serve(content)
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(
        server.url,
        dest,
        sha256=hashlib.sha256(content).hexdigest(),
        size=len(content),
    )
    assert dest.read_bytes() == content
    assert not dest.with_name("blob.tmp").exists()
    # A plain GET that carries on with the first piece, then one range request
    # for each of the others.
    assert server.requests[0] is None
    assert sorted(server.requests[1:]) == [
        "bytes=1000-1999",
        "bytes=2000-2999",
        "bytes=3000-3999",
        "bytes=4000-4095",
    ]


def test_sliced_download_reassembles_out_of_order(
    tmp_path, fetch_content_mod, serve, sliced
):
    """Pieces land at the right offsets no matter what order they finish in."""
    content = bytes(i % 251 for i in range(100000))
    server = serve(content)
    sliced(piece_bytes=999, connections=8)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert len(server.requests) == 101


def test_sliced_download_exact_multiple(tmp_path, fetch_content_mod, serve, sliced):
    content = os.urandom(4000)
    server = serve(content)
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest, size=len(content))
    assert dest.read_bytes() == content
    assert sorted(server.requests[1:]) == [
        "bytes=1000-1999",
        "bytes=2000-2999",
        "bytes=3000-3999",
    ]


def test_sliced_download_first_piece_covers_everything(
    tmp_path, fetch_content_mod, serve, sliced
):
    content = os.urandom(2000)
    server = serve(content)
    sliced(min_bytes=1024, piece_bytes=4096)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest()
    )
    assert dest.read_bytes() == content
    assert server.requests == [None]


@pytest.mark.parametrize("size", (None, 512), ids=("unknown size", "known size"))
def test_sliced_download_small_file_single_request(
    tmp_path, fetch_content_mod, serve, sliced, size
):
    """Files under the threshold cost exactly one request, with no Range."""
    content = os.urandom(512)
    server = serve(content)
    sliced(min_bytes=1024)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest(), size=size
    )
    assert dest.read_bytes() == content
    assert server.requests == [None]


def test_sliced_download_at_threshold(tmp_path, fetch_content_mod, serve, sliced):
    content = os.urandom(1024)
    server = serve(content)
    sliced(min_bytes=1024, piece_bytes=512)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert server.requests == [None, "bytes=512-1023"]


@pytest.mark.parametrize(
    "value,expected",
    (
        pytest.param(None, False, id="unset"),
        pytest.param("", False, id="empty"),
        pytest.param("0", False, id="0"),
        pytest.param("1", False, id="1"),
        pytest.param("2", True, id="2"),
        pytest.param("8", True, id="8"),
        pytest.param("lots", False, id="garbage"),
    ),
)
def test_slicing_enabled(fetch_content_mod, monkeypatch, value, expected):
    monkeypatch.delenv("TASKGRAPH_FETCH_SLICES", raising=False)
    if value is not None:
        monkeypatch.setenv("TASKGRAPH_FETCH_SLICES", value)
    assert fetch_content_mod.slicing_enabled() is expected


def test_download_to_path_slicing_disabled(
    tmp_path, fetch_content_mod, serve, sliced, monkeypatch
):
    content = os.urandom(4096)
    server = serve(content)
    sliced()
    monkeypatch.setenv("TASKGRAPH_FETCH_SLICES", "1")
    monkeypatch.setattr(
        fetch_content_mod,
        "get_connection_pool",
        lambda: pytest.fail("the connection pool must not be used"),
    )
    dest = tmp_path / "blob"

    fetch_content_mod.download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert server.requests == [None]


def test_download_to_path_sliced(tmp_path, fetch_content_mod, serve, sliced):
    content = os.urandom(4096)
    server = serve(content)
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    fetch_content_mod.download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest()
    )
    assert dest.read_bytes() == content
    assert len(server.requests) == 5


def test_sliced_download_no_range_support(tmp_path, fetch_content_mod, serve, sliced):
    """A server that ignores Range headers is detected on the first piece."""
    server = serve(os.urandom(4096), ranges=False, accept_ranges=None)
    sliced()
    dest = tmp_path / "blob"

    with pytest.raises(fetch_content_mod.RangeNotSupported):
        fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert not dest.exists()
    assert not dest.with_name("blob.tmp").exists()


def test_download_to_path_no_range_support_falls_back(
    tmp_path, fetch_content_mod, serve, sliced
):
    content = os.urandom(4096)
    server = serve(content, ranges=False, accept_ranges=None)
    sliced()
    dest = tmp_path / "blob"

    fetch_content_mod.download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest()
    )
    assert dest.read_bytes() == content
    # The single stream fallback is the last request.
    assert server.requests[-1] is None


def test_sliced_download_accept_ranges_none(tmp_path, fetch_content_mod, serve, sliced):
    content = os.urandom(4096)
    server = serve(content, accept_ranges="none")
    sliced()
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert server.requests == [None]


def test_sliced_download_gzip_encoded(tmp_path, fetch_content_mod, serve, sliced):
    """Byte ranges of a gzip encoded response can't be spliced together."""
    content = b"compressible " * 1000
    server = serve(content, gzip=True)
    sliced(min_bytes=16)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest()
    )
    assert dest.read_bytes() == content
    assert server.requests == [None]


def test_sliced_download_empty_object(tmp_path, fetch_content_mod, serve, sliced):
    server = serve(b"")
    sliced()
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == b""
    assert server.requests == [None]


@pytest.mark.parametrize("fault", ("error", "truncate"))
def test_sliced_download_piece_retry(tmp_path, fetch_content_mod, serve, sliced, fault):
    content = os.urandom(4096)
    server = serve(content)
    server.faults["bytes=1000-1999"] = [fault]
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(
        server.url, dest, sha256=hashlib.sha256(content).hexdigest()
    )
    assert dest.read_bytes() == content
    # A truncated piece resumes from where it stopped.
    retry = "bytes=1000-1999" if fault == "error" else "bytes=1500-1999"
    assert retry in server.requests[server.requests.index("bytes=1000-1999") + 1 :]


def test_sliced_download_first_piece_retry(tmp_path, fetch_content_mod, serve, sliced):
    """The first piece is re-requested with a range if the GET breaks off."""
    content = os.urandom(4000)
    server = serve(content)
    server.faults[None] = ["truncate"]
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    # Half of the 4000 bytes body went out before the connection closed, so
    # the first piece was complete.
    assert "bytes=0-999" not in server.requests

    server.requests.clear()
    server.faults[None] = ["error"]
    with pytest.raises(urllib.error.HTTPError):
        fetch_content_mod.sliced_download_to_path(server.url, dest)


def test_sliced_download_first_piece_resumes(
    tmp_path, fetch_content_mod, serve, sliced
):
    content = os.urandom(4000)
    server = serve(content)
    server.faults[None] = ["truncate"]
    sliced(piece_bytes=3000)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert "bytes=2000-2999" in server.requests


def test_sliced_download_piece_fails(tmp_path, fetch_content_mod, serve, sliced):
    server = serve(os.urandom(4096))
    server.faults["bytes=2000-2999"] = ["error"] * 3
    sliced(piece_bytes=1000)
    dest = tmp_path / "blob"

    with pytest.raises(urllib.error.HTTPError):
        fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert server.requests.count("bytes=2000-2999") == 3
    assert not dest.exists()
    assert not dest.with_name("blob.tmp").exists()


def test_sliced_download_straggler(
    tmp_path, fetch_content_mod, serve, sliced, monkeypatch
):
    """A piece that is much slower than the others is re-requested."""
    content = os.urandom(4000)
    server = serve(content)
    server.faults["bytes=1000-1999"] = ["slow"]
    pool = sliced(piece_bytes=1000, connections=1)
    monkeypatch.setattr(fetch_content_mod, "CHUNK_SIZE", 100)
    monkeypatch.setattr(fetch_content_mod, "STRAGGLER_MIN_SECONDS", 0.2)
    for _ in range(4):
        pool.record(1000, 0.001)
    dest = tmp_path / "blob"

    fetch_content_mod.sliced_download_to_path(server.url, dest)
    assert dest.read_bytes() == content
    assert "bytes=1600-1999" in server.requests


def test_sliced_download_bad_sha256(tmp_path, fetch_content_mod, serve, sliced):
    server = serve(os.urandom(4096))
    sliced()
    dest = tmp_path / "blob"

    with pytest.raises(fetch_content_mod.IntegrityError, match="sha256 mismatch"):
        fetch_content_mod.sliced_download_to_path(server.url, dest, sha256="0" * 64)

    assert not dest.exists()
    assert not dest.with_name(f"{dest.name}.tmp").exists()


@pytest.mark.parametrize("length", (4096, 512), ids=("sliced", "single"))
def test_sliced_download_bad_size(tmp_path, fetch_content_mod, serve, sliced, length):
    server = serve(os.urandom(length))
    sliced()
    dest = tmp_path / "blob"

    with pytest.raises(fetch_content_mod.IntegrityError, match="size mismatch"):
        fetch_content_mod.sliced_download_to_path(server.url, dest, size=9999)

    assert server.requests == [None]
    assert not dest.exists()
    assert not dest.with_name(f"{dest.name}.tmp").exists()


def test_sliced_download_bad_sha256_single(tmp_path, fetch_content_mod, serve, sliced):
    server = serve(os.urandom(512))
    sliced()
    dest = tmp_path / "blob"

    with pytest.raises(fetch_content_mod.IntegrityError, match="sha256 mismatch"):
        fetch_content_mod.sliced_download_to_path(server.url, dest, sha256="0" * 64)
    assert not dest.with_name(f"{dest.name}.tmp").exists()


def test_sliced_download_forwards_headers(tmp_path, fetch_content_mod, serve, sliced):
    """Caller supplied headers reach every request, not just the first."""
    seen = []

    class Recording(RangeHandler):
        def do_GET(self):
            seen.append(self.headers.get("X-Taskcluster-Skip-Cdn"))
            super().do_GET()

    server = serve(os.urandom(4096))
    server.RequestHandlerClass = Recording
    sliced(piece_bytes=1000)

    fetch_content_mod.sliced_download_to_path(
        server.url, tmp_path / "blob", headers=["x-taskcluster-skip-cdn: true"]
    )
    assert seen == ["true"] * 5


def test_connection_pool_priority(fetch_content_mod):
    pool = fetch_content_mod.ConnectionPool(1)
    gate = threading.Event()
    done = threading.Event()
    order = []
    pool.submit(0, gate.wait)
    for priority, name in ((5, "a"), (1, "b"), (5, "c"), (float("-inf"), "d")):
        pool.submit(priority, order.append, name)
    pool.submit(10, done.set)
    gate.set()
    assert done.wait(5)
    assert order == ["d", "b", "a", "c"]


def test_sliced_download_largest_first(tmp_path, fetch_content_mod, serve, sliced):
    """Pieces of the larger file are all fetched before those of the smaller."""
    log = []
    small = serve(os.urandom(3000), log=log)
    large = serve(os.urandom(8000), log=log)
    pool = sliced(piece_bytes=1000, connections=1)

    # Hold the only connection until both downloads are queued.
    gate = threading.Event()
    pool.submit(float("-inf"), gate.wait)
    threads = [
        threading.Thread(
            target=fetch_content_mod.sliced_download_to_path,
            args=(server.url, tmp_path / name),
        )
        for server, name in ((small, "small"), (large, "large"))
    ]
    for thread in threads:
        thread.start()
    while len(pool._queue) < 2:
        threading.Event().wait(0.01)
    gate.set()
    for thread in threads:
        thread.join(10)

    assert (tmp_path / "small").read_bytes() == small.content
    assert (tmp_path / "large").read_bytes() == large.content
    # Both GETs go first, as they are how sizes are discovered.
    assert [server for server, _ in log[:2]] == [small, large]
    assert [server for server, _ in log[2:]] == [large] * 7 + [small] * 2


@pytest.mark.parametrize(
    "start,total,piece_bytes,expected",
    (
        pytest.param(
            0, 400, 100, [(0, 99), (100, 199), (200, 299), (300, 399)], id="exact"
        ),
        pytest.param(0, 250, 100, [(0, 99), (100, 199), (200, 249)], id="remainder"),
        pytest.param(0, 50, 100, [(0, 49)], id="smaller than a piece"),
        pytest.param(64, 128, 32, [(64, 95), (96, 127)], id="offset start"),
        pytest.param(100, 100, 32, [], id="nothing left"),
        pytest.param(200, 100, 32, [], id="past the end"),
    ),
)
def test_split_pieces(fetch_content_mod, start, total, piece_bytes, expected):
    assert fetch_content_mod.split_pieces(start, total, piece_bytes) == expected
    if expected:
        # The ranges must tile the region exactly, with no gaps or overlaps.
        assert expected[0][0] == start
        assert expected[-1][1] == total - 1
        for (_, prev_end), (next_start, _) in zip(expected, expected[1:]):
            assert next_start == prev_end + 1


@pytest.mark.parametrize(
    "headers,expected",
    (
        pytest.param(["Foo: bar"], {"Foo": "bar"}, id="simple"),
        pytest.param([], {}, id="empty"),
        pytest.param(None, {}, id="none"),
        pytest.param(
            ["Location: https://example.com:443/x"],
            {"Location": "https://example.com:443/x"},
            id="colon in value",
        ),
    ),
)
def test_parse_headers(fetch_content_mod, headers, expected):
    assert fetch_content_mod.parse_headers(headers) == expected


@pytest.mark.parametrize(
    "value,expected",
    (
        pytest.param(None, None, id="unset"),
        pytest.param("1", ["x-taskcluster-skip-cdn: true"], id="1"),
        pytest.param("true", ["x-taskcluster-skip-cdn: true"], id="true"),
        pytest.param("0", None, id="0"),
    ),
)
def test_command_task_artifacts_skip_cdn(
    monkeypatch, tmp_path, fetch_content_mod, value, expected
):
    fetches = [{"task": "abc123", "artifact": "public/foo.zip", "extract": False}]
    monkeypatch.setenv("MOZ_FETCHES", json.dumps(fetches))
    monkeypatch.setenv("TASKCLUSTER_ROOT_URL", "https://tc.example.com")
    monkeypatch.delenv("TASKGRAPH_SKIP_CDN", raising=False)
    if value is not None:
        monkeypatch.setenv("TASKGRAPH_SKIP_CDN", value)

    captured = []
    monkeypatch.setattr(fetch_content_mod, "fetch_urls", captured.extend)

    args = MagicMock()
    args.dest = str(tmp_path)
    fetch_content_mod.command_task_artifacts(args)

    assert [download[5] for download in captured] == [expected]


@pytest.mark.parametrize(
    "url,sha256,size,headers,raises",
    (
        pytest.param(
            "https://example.com",
            "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2",
            6,
            ["User-Agent: foobar"],
            False,
            id="valid",
        ),
        pytest.param(
            "https://example.com",
            "abcdef",
            6,
            ["User-Agent: foobar"],
            True,
            id="invalid sha256",
        ),
        pytest.param(
            "https://example.com",
            "c3ab8ff13720e8ad9047dd39466b3c8974e592c2fa383d4a3960714caef0c4f2",
            123,
            ["User-Agent: foobar"],
            True,
            id="invalid size",
        ),
    ),
)
def test_stream_download(
    monkeypatch, fetch_content_mod, url, sha256, size, headers, raises
):
    def mock_urlopen(req, timeout=None, *, context=None):
        assert req._full_url == url
        assert timeout is not None
        if headers:
            # stream_download adds accept-encoding
            assert len(req.headers) == len(headers) + 1
            for header in headers:
                k, v = header.split(":")
                k = k.lower().capitalize().strip()
                assert k in req.headers
                assert req.headers[k] == v.strip()

        # create a mock context manager
        cm = MagicMock()
        cm.getcode.return_value = 200

        def getheader(field):
            if field.lower() == "content-length":
                return size

        # simulates chunking
        cm.getheader = getheader
        cm.read.side_effect = [b"foo", b"bar", None]
        cm.__enter__.return_value = cm
        return cm

    monkeypatch.setattr(urllib.request, "urlopen", mock_urlopen)

    result = b""
    try:
        for chunk in fetch_content_mod.stream_download(url, sha256, size, headers):
            result += chunk
        assert result == b"foobar"
    except fetch_content_mod.IntegrityError:
        if not raises:
            raise


@pytest.mark.parametrize(
    "artifact,expected_url_suffix",
    (
        pytest.param(
            "public/foo.apworld",
            "task/abc123/artifacts/public/foo.apworld",
            id="simple artifact name",
        ),
        pytest.param(
            "public/Twilight Princess-0.2.3.apworld",
            "task/abc123/artifacts/public/Twilight%20Princess-0.2.3.apworld",
            id="artifact name with space",
        ),
    ),
)
def test_command_task_artifacts_url_encoding(
    monkeypatch,
    tmp_path,
    fetch_content_mod,
    artifact,
    expected_url_suffix,
):
    fetches = [{"task": "abc123", "artifact": artifact, "extract": False}]
    monkeypatch.setenv("MOZ_FETCHES", json.dumps(fetches))
    monkeypatch.setenv("TASKCLUSTER_ROOT_URL", "https://tc.example.com")

    captured_urls = []

    def mock_fetch_urls(downloads):
        for url, dest_dir, extract, sha256, size, headers in downloads:
            captured_urls.append(url)

    monkeypatch.setattr(fetch_content_mod, "fetch_urls", mock_fetch_urls)

    args = MagicMock()
    args.dest = str(tmp_path)
    fetch_content_mod.command_task_artifacts(args)

    assert len(captured_urls) == 1
    url = captured_urls[0]
    assert url == f"https://tc.example.com/api/queue/v1/{expected_url_suffix}"


@pytest.mark.parametrize(
    "url,expected_dest_filename",
    (
        pytest.param(
            "https://tc.example.com/api/queue/v1/task/abc/artifacts/public/foo.apworld",
            "foo.apworld",
            id="simple",
        ),
        pytest.param(
            "https://tc.example.com/api/queue/v1/task/abc/artifacts/public/Twilight%20Princess-0.2.3.apworld",
            "Twilight Princess-0.2.3.apworld",
            id="url-encoded space",
        ),
    ),
)
def test_fetch_and_extract_dest_filename(
    monkeypatch,
    tmp_path,
    fetch_content_mod,
    url,
    expected_dest_filename,
):
    downloaded_to = []

    def mock_download_to_path(url, path, sha256=None, size=None, headers=None):
        downloaded_to.append(path)
        path.touch()

    monkeypatch.setattr(fetch_content_mod, "download_to_path", mock_download_to_path)

    fetch_content_mod.fetch_and_extract(url, tmp_path, extract=False)

    assert len(downloaded_to) == 1
    assert downloaded_to[0].name == expected_dest_filename


@pytest.mark.parametrize(
    "expected,orig,dest,strip_components,add_prefix",
    [
        # Archives to repack
        (True, pathlib.Path("archive"), pathlib.Path("archive.tar.zst"), 0, ""),
        (True, pathlib.Path("archive.tar"), pathlib.Path("archive.tar.zst"), 0, ""),
        (True, pathlib.Path("archive.tgz"), pathlib.Path("archive.tar.zst"), 0, ""),
        (True, pathlib.Path("archive.zip"), pathlib.Path("archive.tar.zst"), 0, ""),
        (True, pathlib.Path("archive.tar.xz"), pathlib.Path("archive.tar.zst"), 0, ""),
        (True, pathlib.Path("archive.zst"), pathlib.Path("archive.tar.zst"), 0, ""),
        # Path is exactly the same
        (False, pathlib.Path("archive"), pathlib.Path("archive"), 0, ""),
        (False, pathlib.Path("file.txt"), pathlib.Path("file.txt"), 0, ""),
        (False, pathlib.Path("archive.tar"), pathlib.Path("archive.tar"), 0, ""),
        (False, pathlib.Path("archive.tgz"), pathlib.Path("archive.tgz"), 0, ""),
        (False, pathlib.Path("archive.zip"), pathlib.Path("archive.zip"), 0, ""),
        (
            False,
            pathlib.Path("archive.tar.zst"),
            pathlib.Path("archive.tar.zst"),
            0,
            "",
        ),
        (
            False,
            pathlib.Path("archive-before.tar.zst"),
            pathlib.Path("archive-after.tar.zst"),
            0,
            "",
        ),
        (
            False,
            pathlib.Path("before.foo.bar.baz"),
            pathlib.Path("after.foo.bar.baz"),
            0,
            "",
        ),
        # Non-default values for strip_components and add_prefix parameters
        (True, pathlib.Path("archive.tar.zst"), pathlib.Path("archive.tar.zst"), 1, ""),
        (
            True,
            pathlib.Path("archive.tar.zst"),
            pathlib.Path("archive.tar.zst"),
            0,
            "prefix",
        ),
        (
            True,
            pathlib.Path("archive.tar.zst"),
            pathlib.Path("archive.tar.zst"),
            1,
            "prefix",
        ),
        # Real edge cases that should not be repacks
        (
            False,
            pathlib.Path("python-3.8.10-amd64.exe"),
            pathlib.Path("python.exe"),
            0,
            "",
        ),
        (
            False,
            pathlib.Path("9ee26e91-9b52-44ba-8d30-c0230dd587b2.bin"),
            pathlib.Path("model.esen.intgemm.alphas.bin"),
            0,
            "",
        ),
    ],
)
def test_should_repack_archive(
    fetch_content_mod, orig, dest, expected, strip_components, add_prefix
):
    assert (
        fetch_content_mod.should_repack_archive(
            orig, dest, strip_components, add_prefix
        )
        == expected
    ), (
        f"Failed for orig: {orig}, dest: {dest}, strip_components: {strip_components}, add_prefix: {add_prefix}, expected {expected} but received {not expected}"
    )


def _make_tar(path, files):
    with tarfile.open(path, "w") as tar:
        for name, content in files.items():
            data = content.encode("utf-8")
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))


def test_fetch_urls_merges_staged_extractions(monkeypatch, tmp_path, fetch_content_mod):
    archives = tmp_path / "archives"
    archives.mkdir()
    _make_tar(
        archives / "common.tar",
        {"tests/common/a.txt": "a", "tests/shared.txt": "first"},
    )
    _make_tar(
        archives / "suite.tar",
        {"tests/suite/b.txt": "b", "tests/shared.txt": "second"},
    )
    dest = tmp_path / "fetches"
    dest.mkdir()

    def mock_download_to_path(url, path, sha256=None, size=None, headers=None):
        shutil.copy(archives / path.name, path)

    monkeypatch.setattr(fetch_content_mod, "download_to_path", mock_download_to_path)

    fetch_content_mod.fetch_urls(
        [
            ("https://example.com/common.tar", dest, True, None),
            ("https://example.com/suite.tar", dest, True, None),
        ]
    )

    assert sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*")) == [
        "tests",
        "tests/common",
        "tests/common/a.txt",
        "tests/shared.txt",
        "tests/suite",
        "tests/suite/b.txt",
    ]
    assert (dest / "tests" / "shared.txt").read_text() == "second"


def test_fetch_urls_places_unextracted_files(monkeypatch, tmp_path, fetch_content_mod):
    archives = tmp_path / "archives"
    archives.mkdir()
    _make_tar(archives / "tool.tar", {"tool/bin/tool": "t"})
    dest = tmp_path / "fetches"
    dest.mkdir()

    def mock_download_to_path(url, path, sha256=None, size=None, headers=None):
        if path.name.endswith(".tar"):
            shutil.copy(archives / path.name, path)
        else:
            path.write_text("plain")

    monkeypatch.setattr(fetch_content_mod, "download_to_path", mock_download_to_path)

    fetch_content_mod.fetch_urls(
        [
            ("https://example.com/tool.tar", dest, True, None),
            ("https://example.com/plain.txt", dest, False, None),
            ("https://example.com/notatar.txt", dest, True, None),
        ]
    )

    assert sorted(p.name for p in dest.iterdir()) == [
        "notatar.txt",
        "plain.txt",
        "tool",
    ]
    assert (dest / "tool" / "bin" / "tool").read_text() == "t"
    assert (dest / "notatar.txt").read_text() == "plain"


def test_merge_tree_replaces_conflicting_entries(tmp_path, fetch_content_mod):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    (src / "dir").mkdir(parents=True)
    (src / "dir" / "new.txt").write_text("new")
    (src / "file").write_text("file")
    (dest / "dir" / "kept").mkdir(parents=True)
    (dest / "dir" / "new.txt").write_text("old")
    (dest / "file").mkdir()

    fetch_content_mod.merge_tree(src, dest)

    assert not src.exists()
    assert (dest / "dir" / "new.txt").read_text() == "new"
    assert (dest / "dir" / "kept").is_dir()
    assert (dest / "file").is_file()


@pytest.mark.skipif(
    sys.platform == "win32" or os.getuid() == 0, reason="needs POSIX directory modes"
)
def test_merge_tree_readonly_dir_from_later_fetch(tmp_path, fetch_content_mod):
    src = tmp_path / "src"
    dest = tmp_path / "dest"
    (src / "tests").mkdir(parents=True)
    (src / "tests" / "b.txt").write_text("b")
    (dest / "tests").mkdir(parents=True)
    (dest / "tests" / "a.txt").write_text("a")
    (src / "tests").chmod(0o555)

    fetch_content_mod.merge_tree(src, dest)

    assert (dest / "tests" / "a.txt").read_text() == "a"
    assert (dest / "tests" / "b.txt").read_text() == "b"
    assert stat.S_IMODE((dest / "tests").stat().st_mode) == 0o555


@pytest.fixture
def popen_calls(monkeypatch, fetch_content_mod):
    """Record the arguments of every subprocess.Popen call, letting them run."""
    calls = []
    real_popen = fetch_content_mod.subprocess.Popen

    def recording_popen(args, *a, **kw):
        calls.append(args)
        return real_popen(args, *a, **kw)

    monkeypatch.setattr(fetch_content_mod.subprocess, "Popen", recording_popen)
    return calls


def _tar_bytes(tmp_path, files):
    _make_tar(tmp_path / "raw.tar", files)
    return (tmp_path / "raw.tar").read_bytes()


def _make_tar_zst(path, files):
    zstandard = pytest.importorskip("zstandard")
    path.write_bytes(
        zstandard.ZstdCompressor().compress(_tar_bytes(path.parent, files))
    )


@pytest.mark.skipif(sys.platform == "win32", reason="Windows extracts with tarfile")
@pytest.mark.skipif(not shutil.which("zstd"), reason="needs the zstd program")
def test_extract_archive_zstd_with_tar(tmp_path, fetch_content_mod, popen_calls):
    archive = tmp_path / "archive.tar.zst"
    _make_tar_zst(archive, {"dir/a.txt": "a", "b.txt": "b"})
    dest = tmp_path / "dest"
    dest.mkdir()

    fetch_content_mod.extract_archive(archive, dest)

    assert popen_calls == [
        ["tar", "--use-compress-program=zstd", "-xf", str(archive.resolve())]
    ]
    assert (dest / "dir" / "a.txt").read_text() == "a"
    assert (dest / "b.txt").read_text() == "b"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows extracts with tarfile")
def test_extract_archive_zstd_without_zstd_program(
    tmp_path, fetch_content_mod, popen_calls, monkeypatch
):
    """Without a zstd program, decompress in Python and pipe to tar."""
    archive = tmp_path / "archive.tar.zst"
    _make_tar_zst(archive, {"dir/a.txt": "a"})
    dest = tmp_path / "dest"
    dest.mkdir()
    monkeypatch.setattr(fetch_content_mod.shutil, "which", lambda name: None)

    fetch_content_mod.extract_archive(archive, dest)

    assert popen_calls == [["tar", "xf", "-"]]
    assert (dest / "dir" / "a.txt").read_text() == "a"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows extracts with tarfile")
def test_extract_archive_other_compression_pipes_to_tar(
    tmp_path, fetch_content_mod, popen_calls
):
    """Only zstd is handed to tar; other formats still go through the pipe."""
    archive = tmp_path / "archive.tar.xz"
    archive.write_bytes(lzma.compress(_tar_bytes(tmp_path, {"dir/a.txt": "a"})))
    dest = tmp_path / "dest"
    dest.mkdir()

    fetch_content_mod.extract_archive(archive, dest)

    assert popen_calls == [["tar", "xf", "-"]]
    assert (dest / "dir" / "a.txt").read_text() == "a"


@pytest.mark.skipif(sys.platform == "win32", reason="Windows extracts with tarfile")
@pytest.mark.skipif(not shutil.which("zstd"), reason="needs the zstd program")
def test_extract_archive_zstd_tar_failure(tmp_path, fetch_content_mod):
    """A corrupt archive makes tar exit non-zero, which must be reported."""
    archive = tmp_path / "archive.tar.zst"
    # Incompressible, so that the tar header survives the truncation.
    _make_tar_zst(archive, {"a.txt": os.urandom(1000000).hex()})
    data = archive.read_bytes()
    archive.write_bytes(data[: len(data) // 2])
    dest = tmp_path / "dest"
    dest.mkdir()

    with pytest.raises(Exception, match="exited"):
        fetch_content_mod.extract_archive(archive, dest)
