#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from phydrax._interpolation import linear_interpolate

from ...imaging import bilinear_sample, ImageSample2D
from ._types import DenseDisplacementField2D


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
    row_indices = linear_interpolate(
        rows, jnp.arange(rows.shape[0], dtype=float), coordinates[..., 0]
    ).values
    column_indices = linear_interpolate(
        columns, jnp.arange(columns.shape[0], dtype=float), coordinates[..., 1]
    ).values
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


__all__ = ["sample_rectilinear_field"]
