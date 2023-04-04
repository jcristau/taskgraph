# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.


import logging

from ..util.python_path import find_object
from .transform import loader as transform_loader
from ..transforms.base import TransformSequence

logger = logging.getLogger(__name__)


# These are defined here, but applied by the Kind class in generator.py
DEFAULT_TRANSFORMS = [
    "taskgraph.transforms.job:transforms",
    "taskgraph.transforms.task:transforms",
]


def loader(kind, path, config, params, loaded_tasks):
    """
    This default loader builds on the `transform` loader by providing sensible
    default transforms that the majority of simple tasks will need.
    Specifically, `job` and `task` transforms will be appended to the end of the
    list of transforms in the kind being loaded.
    """
    transform_refs = config.get("transforms", [])
    for t in DEFAULT_TRANSFORMS:
        if t in config.get("transforms", ()):
            raise KeyError(f"Transform {t} is already added by the default transform; it must not be defined in the kind")
    transform_refs.extend(DEFAULT_TRANSFORMS)
    transforms = TransformSequence()
    for xform_path in transform_refs:
        transform = find_object(xform_path)
        transforms.add(transform)
    return transforms, transform_loader(kind, path, config, params, loaded_tasks)
