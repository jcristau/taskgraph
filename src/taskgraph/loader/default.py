# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import logging

from .transform import loader as transform_loader

logger = logging.getLogger(__name__)


# These are defined here, but applied by the Kind class in generator.py
DEFAULT_TRANSFORMS = [
    "taskgraph.transforms.job:transforms",
    "taskgraph.transforms.task:transforms",
]


def loader(*args, **kwargs):
    """
    This default loader builds on the `transform` loader by providing sensible
    default transforms that the majority of simple tasks will need.
    Specifically, `job` and `task` transforms will be appended to the end of the
    list of transforms in the kind being loaded.
    """

    yield from transform_loader(*args, **kwargs)
