#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Image and particle velocimetry."""

from .._model import register_artifact_value
from . import imaging, io, piv, synthetic, tracking


for _namespace in (imaging, io, piv, synthetic, tracking):
    for _artifact_name in _namespace.__all__:
        _artifact_value = getattr(_namespace, _artifact_name)
        if isinstance(_artifact_value, type) and _artifact_value.__module__.startswith(
            f"{_namespace.__name__}."
        ):
            register_artifact_value(
                f"{_namespace.__name__}:{_artifact_name}",
                _artifact_value,
            )

del _namespace, _artifact_name, _artifact_value


__all__ = [
    "imaging",
    "io",
    "piv",
    "synthetic",
    "tracking",
]
