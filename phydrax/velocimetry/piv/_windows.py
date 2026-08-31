#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..imaging import bilinear_sample
from ._types import WindowGrid2D


def _pair(value: int | Sequence[int], /, *, name: str, minimum: int) -> tuple[int, int]:
    if isinstance(value, int):
        result = (int(value), int(value))
    else:
        result = tuple(int(item) for item in value)
    if len(result) != 2 or any(item < minimum for item in result):
        raise ValueError(f"{name} must contain two integers >= {minimum}.")
    return result


class WindowBatch2D(StrictModule):
    """Flattened fixed-grid windows and their per-sample support masks."""

    values: Array
    mask: Array
    centers_rc: Array
    grid_shape: tuple[int, int] = eqx.field(static=True)


def prepare_window_grid(
    image_shape: Sequence[int],
    window_size: int | Sequence[int],
    overlap: int | Sequence[int],
    search_margin: int | Sequence[int],
    /,
) -> WindowGrid2D:
    """Prepare deterministic row-major interrogation centers."""
    shape = _pair(image_shape, name="image_shape", minimum=1)
    window = _pair(window_size, name="window_size", minimum=2)
    overlap_ = _pair(overlap, name="overlap", minimum=0)
    margin = _pair(search_margin, name="search_margin", minimum=0)
    if any(overlap_[axis] >= window[axis] for axis in range(2)):
        raise ValueError("overlap must be smaller than window_size.")
    if any(window[axis] > shape[axis] for axis in range(2)):
        raise ValueError("window_size cannot exceed image_shape.")
    spacing = tuple(window[axis] - overlap_[axis] for axis in range(2))
    starts = tuple(
        jnp.arange(0, shape[axis] - window[axis] + 1, spacing[axis], dtype=float)
        for axis in range(2)
    )
    row, column = jnp.meshgrid(
        starts[0] + 0.5 * (window[0] - 1),
        starts[1] + 0.5 * (window[1] - 1),
        indexing="ij",
    )
    centers = jnp.stack((row, column), axis=-1)
    grid_shape = (int(centers.shape[0]), int(centers.shape[1]))
    active = jnp.ones(grid_shape, dtype=bool)
    grid_id = canonical_fingerprint(
        {
            "kind": "piv-window-grid-2d",
            "image_shape": list(shape),
            "window_size": list(window),
            "search_margin": list(margin),
            "spacing": list(spacing),
            "grid_shape": list(grid_shape),
        }
    )
    return WindowGrid2D(centers, active, grid_shape, window, margin, spacing, grid_id)


def window_sample_coordinates(
    grid: WindowGrid2D,
    /,
    *,
    extended: bool = False,
    center_shift_rc: Array | None = None,
) -> Array:
    """Return flattened sample coordinates for every prepared window."""
    if not isinstance(grid, WindowGrid2D):
        raise TypeError("grid must be a WindowGrid2D.")
    size = tuple(
        grid.window_size[axis] + (2 * grid.search_margin[axis] if extended else 0)
        for axis in range(2)
    )
    rows, columns = jnp.meshgrid(
        jnp.arange(size[0], dtype=float) - 0.5 * (size[0] - 1),
        jnp.arange(size[1], dtype=float) - 0.5 * (size[1] - 1),
        indexing="ij",
    )
    offsets = jnp.stack((rows, columns), axis=-1)
    centers = grid.centers_rc.reshape((-1, 2))
    if center_shift_rc is not None:
        shift = jnp.asarray(center_shift_rc, dtype=float)
        if shift.shape == grid.grid_shape + (2,):
            shift = shift.reshape((-1, 2))
        if shift.shape != centers.shape:
            raise ValueError("center_shift_rc must match the prepared grid.")
        centers = centers + shift
    return centers[:, None, None, :] + offsets[None, :, :, :]


def extract_windows(
    image: Array,
    mask: Array,
    grid: WindowGrid2D,
    /,
    *,
    extended: bool = False,
    center_shift_rc: Array | None = None,
) -> WindowBatch2D:
    """Extract padded/deformed windows through the shared bilinear sampler."""
    coordinates = window_sample_coordinates(
        grid, extended=extended, center_shift_rc=center_shift_rc
    )
    sampled = bilinear_sample(image, coordinates, valid_mask=mask, fill_value=0.0)
    active = grid.active.reshape((-1, 1, 1))
    sample_mask = sampled.valid & active
    return WindowBatch2D(
        jnp.where(sample_mask, sampled.values, 0.0),
        sample_mask,
        grid.centers_rc.reshape((-1, 2)),
        grid.grid_shape,
    )


__all__ = [
    "WindowBatch2D",
    "extract_windows",
    "prepare_window_grid",
    "window_sample_coordinates",
]
