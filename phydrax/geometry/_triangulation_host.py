#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host conversion and immutable storage for triangulation preparation."""

from __future__ import annotations

import numpy as np

from ..typing import ConvertibleToArray


def host_point_array(
    points: ConvertibleToArray, name: str, dimensions: tuple[int, ...], /
) -> np.ndarray:
    """Convert finite real point rows to an immutable float64 bank."""
    array = np.asarray(points)
    if np.issubdtype(array.dtype, np.bool_) or not (
        np.issubdtype(array.dtype, np.floating) or np.issubdtype(array.dtype, np.integer)
    ):
        raise TypeError(
            f"{name} must be a real floating or integer array; got dtype {array.dtype}."
        )
    if array.ndim != 2 or array.shape[1] not in dimensions:
        shapes = " or ".join(f"(n, {dimension})" for dimension in dimensions)
        raise ValueError(f"{name} must have shape {shapes}; got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite; got nonfinite coordinates.")
    converted = np.array(array, dtype=np.float64, copy=True)
    converted.setflags(write=False)
    return converted


def frozen_array(array: np.ndarray, /) -> np.ndarray:
    """Retain contiguous host storage with writes disabled."""
    result = np.ascontiguousarray(array)
    result.setflags(write=False)
    return result
