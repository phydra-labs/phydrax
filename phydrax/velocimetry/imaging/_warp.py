#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array

from ..._interpolation import apply_gather_stencil, rectilinear_stencil
from ._types import DenseDisplacementField2D, ImageGeometry2D, ImageSample2D


def image_coordinates(geometry_or_shape: ImageGeometry2D | Sequence[int], /) -> Array:
    """Return pixel-index coordinates with components ``(row_down, column_right)``."""
    if isinstance(geometry_or_shape, ImageGeometry2D):
        shape = geometry_or_shape.image_shape
    else:
        shape = tuple(int(item) for item in geometry_or_shape)
        if len(shape) != 2 or any(item < 1 for item in shape):
            raise ValueError("Image shape must contain two positive dimensions.")
    rows, columns = jnp.meshgrid(
        jnp.arange(shape[0], dtype=float),
        jnp.arange(shape[1], dtype=float),
        indexing="ij",
    )
    return jnp.stack((rows, columns), axis=-1)


def bilinear_sample(
    image: Array,
    coordinates_rc: Array,
    /,
    *,
    valid_mask: Array | None = None,
    fill_value: Array | float = 0.0,
) -> ImageSample2D:
    """Bilinearly sample through the native strict, nonperiodic rectilinear map."""
    values = jnp.asarray(image)
    coordinates = jnp.asarray(coordinates_rc, dtype=float)
    if values.ndim < 2:
        raise ValueError("image must have at least two dimensions.")
    if coordinates.ndim < 1 or coordinates.shape[-1] != 2:
        raise ValueError("coordinates_rc must have shape (..., 2).")
    source_mask = (
        jnp.ones(values.shape[:2], dtype=bool)
        if valid_mask is None
        else jnp.asarray(valid_mask, dtype=bool)
    )
    if source_mask.shape != values.shape[:2]:
        raise ValueError("valid_mask must match the first two image dimensions.")
    source_shape = values.shape[:2]
    finite_coordinates = jnp.all(jnp.isfinite(coordinates), axis=-1)
    safe_coordinates = jnp.where(finite_coordinates[..., None], coordinates, 0.0)
    working_values = values
    working_mask = source_mask
    if source_shape[0] == 1:
        working_values = jnp.repeat(working_values, 2, axis=0)
        working_mask = jnp.repeat(working_mask, 2, axis=0)
    if source_shape[1] == 1:
        working_values = jnp.repeat(working_values, 2, axis=1)
        working_mask = jnp.repeat(working_mask, 2, axis=1)
    nodes = (
        jnp.arange(working_values.shape[0], dtype=coordinates.dtype),
        jnp.arange(working_values.shape[1], dtype=coordinates.dtype),
    )
    stencil = rectilinear_stencil(
        nodes,
        safe_coordinates,
        boundary=("constant", "constant"),
    )
    payload_shape = values.shape[2:]
    channels = 1
    for size in payload_shape:
        channels *= int(size)
    flat_values = working_values.reshape(
        (working_values.shape[0] * working_values.shape[1], channels)
    )
    interpolation = apply_gather_stencil(
        flat_values,
        stencil,
        source_mask=working_mask.reshape((-1,)),
        mask_mode="strict",
    )
    support = (
        interpolation.support
        & finite_coordinates
        & (coordinates[..., 0] >= 0.0)
        & (coordinates[..., 0] <= source_shape[0] - 1)
        & (coordinates[..., 1] >= 0.0)
        & (coordinates[..., 1] <= source_shape[1] - 1)
    )
    sampled = interpolation.values.reshape(coordinates.shape[:-1] + payload_shape)
    expanded_support = support.reshape(support.shape + (1,) * len(payload_shape))
    output = jnp.where(
        expanded_support,
        sampled,
        jnp.asarray(fill_value, dtype=sampled.dtype),
    )
    return ImageSample2D(output, support)


def backward_warp(
    image: Array,
    displacement_rc: Array,
    /,
    *,
    valid_mask: Array | None = None,
    fill_value: Array | float = 0.0,
) -> ImageSample2D:
    """Backward warp so output[r,c] samples input[r-dr,c-dc]."""
    displacement = jnp.asarray(displacement_rc, dtype=float)
    if displacement.ndim != 3 or displacement.shape[-1] != 2:
        raise ValueError("displacement_rc must have shape (rows, columns, 2).")
    if tuple(displacement.shape[:2]) != tuple(jnp.shape(image)[:2]):
        raise ValueError("displacement_rc must match the image spatial shape.")
    coordinates = image_coordinates(displacement.shape[:2]) - displacement
    return bilinear_sample(
        image,
        coordinates,
        valid_mask=valid_mask,
        fill_value=fill_value,
    )


def sample_rectilinear_field(
    field: DenseDisplacementField2D,
    coordinates_rc: Array,
    /,
    *,
    fill_value: Array | float = 0.0,
) -> ImageSample2D:
    """Sample a two-dimensional rectilinear displacement grid at pixel positions."""
    if not isinstance(field, DenseDisplacementField2D):
        raise TypeError("field must be a DenseDisplacementField2D.")
    if field.positions_rc.ndim != 3:
        raise ValueError("field must have a two-dimensional grid.")
    coordinates = jnp.asarray(coordinates_rc, dtype=float)
    if coordinates.shape[-1] != 2:
        raise ValueError("coordinates_rc must have shape (..., 2).")
    rows = field.positions_rc[:, 0, 0]
    columns = field.positions_rc[0, :, 1]
    row_indices = jnp.interp(
        coordinates[..., 0], rows, jnp.arange(rows.shape[0], dtype=float)
    )
    column_indices = jnp.interp(
        coordinates[..., 1], columns, jnp.arange(columns.shape[0], dtype=float)
    )
    grid_coordinates = jnp.stack((row_indices, column_indices), axis=-1)
    sampled = bilinear_sample(
        field.displacement_rc,
        grid_coordinates,
        valid_mask=field.valid,
        fill_value=fill_value,
    )
    inside = (
        (coordinates[..., 0] >= rows[0])
        & (coordinates[..., 0] <= rows[-1])
        & (coordinates[..., 1] >= columns[0])
        & (coordinates[..., 1] <= columns[-1])
    )
    valid = sampled.valid & inside
    return ImageSample2D(jnp.where(valid[..., None], sampled.values, fill_value), valid)


__all__ = [
    "backward_warp",
    "bilinear_sample",
    "image_coordinates",
    "sample_rectilinear_field",
]
