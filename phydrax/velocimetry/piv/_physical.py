#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...units import TIME, UnitDefinition
from ..imaging import DenseDisplacementField2D
from ._types import PhysicalPIVResult2D, PIVResult


class AffinePixelMap2D(StrictModule, NonTrainableState):
    """Affine map from pixel ``(column, row, 1)`` to physical ``(x, y)``."""

    matrix: Array
    transform_id: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)

    def __init__(
        self,
        matrix: Array,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        transform_id: str | None = None,
    ):
        matrix_ = jnp.asarray(matrix, dtype=jnp.float64)
        if matrix_.shape != (2, 3):
            raise ValueError("Affine matrix must have shape (2, 3).")
        if not bool(jnp.all(jnp.isfinite(matrix_))):
            raise ValueError("Affine matrix must be finite.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        homogeneous = jnp.concatenate((matrix_, jnp.asarray([[0.0, 0.0, 1.0]])), axis=0)
        resolved_id = transform_id or canonical_fingerprint(
            {
                "kind": "affine-pixel-map-2d",
                "matrix": array_tree_fingerprint(matrix_),
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("transform_id must be a non-empty string.")
        self.matrix = homogeneous
        self.transform_id = resolved_id
        self.coordinate_contract = coordinate_contract


class HomographyPixelMap2D(StrictModule, NonTrainableState):
    """Projective map from pixel ``(column, row, 1)`` to physical ``(x, y)``."""

    matrix: Array
    transform_id: str = eqx.field(static=True)
    coordinate_contract: SpatialCoordinateContract = eqx.field(static=True)

    def __init__(
        self,
        matrix: Array,
        coordinate_contract: SpatialCoordinateContract,
        /,
        *,
        transform_id: str | None = None,
    ):
        matrix_ = jnp.asarray(matrix, dtype=jnp.float64)
        if matrix_.shape != (3, 3):
            raise ValueError("Homography matrix must have shape (3, 3).")
        if not bool(jnp.all(jnp.isfinite(matrix_))):
            raise ValueError("Homography matrix must be finite.")
        if not isinstance(coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        resolved_id = transform_id or canonical_fingerprint(
            {
                "kind": "homography-pixel-map-2d",
                "matrix": array_tree_fingerprint(matrix_),
                "coordinate_contract": coordinate_contract.spatial_id,
            }
        )
        if not isinstance(resolved_id, str) or not resolved_id:
            raise ValueError("transform_id must be a non-empty string.")
        self.matrix = matrix_
        self.transform_id = resolved_id
        self.coordinate_contract = coordinate_contract


def map_pixels_to_physical(
    transform: AffinePixelMap2D | HomographyPixelMap2D,
    positions_rc: Array,
    /,
) -> tuple[Array, Array]:
    """Map row/column positions to right-handed named coordinates ``(x, y)``."""
    if not isinstance(transform, (AffinePixelMap2D, HomographyPixelMap2D)):
        raise TypeError("transform must be an AffinePixelMap2D or HomographyPixelMap2D.")
    positions = jnp.asarray(positions_rc, dtype=jnp.float64)
    if positions.shape[-1] != 2:
        raise ValueError("positions_rc must have shape (..., 2).")
    homogeneous = jnp.stack(
        (
            positions[..., 1],
            positions[..., 0],
            jnp.ones(positions.shape[:-1], dtype=positions.dtype),
        ),
        axis=-1,
    )
    mapped = contract("ij,...j->...i", transform.matrix, homogeneous)
    denominator = mapped[..., 2]
    epsilon = jnp.finfo(mapped.dtype).tiny
    valid = jnp.isfinite(denominator) & (jnp.abs(denominator) > epsilon)
    xy = mapped[..., :2] / jnp.where(valid, denominator, 1.0)[..., None]
    valid = valid & jnp.all(jnp.isfinite(xy), axis=-1)
    return jnp.where(valid[..., None], xy, 0.0), valid


def convert_to_physical(
    source: DenseDisplacementField2D | PIVResult,
    transform: AffinePixelMap2D | HomographyPixelMap2D,
    /,
    *,
    delta_t: Array | float,
    time_unit: UnitDefinition,
    stage: str = "replaced",
) -> PhysicalPIVResult2D:
    """Convert finite pixel endpoints, preserving nonlinear homography displacement."""
    if isinstance(source, PIVResult):
        if stage == "raw":
            field = source.raw
        elif stage == "validated":
            field = source.validated
        elif stage == "replaced":
            field = source.replaced
        else:
            raise ValueError("stage must be raw, validated, or replaced.")
    elif isinstance(source, DenseDisplacementField2D):
        field = source
    else:
        raise TypeError("source must be a DenseDisplacementField2D or PIVResult.")
    if not isinstance(time_unit, UnitDefinition) or time_unit.dimension != TIME:
        raise ValueError("time_unit must be a time UnitDefinition.")
    elapsed = jnp.asarray(delta_t, dtype=jnp.float64)
    if elapsed.shape != ():
        raise ValueError("delta_t must be scalar.")
    start_xy, start_valid = map_pixels_to_physical(transform, field.positions_rc)
    end_xy, end_valid = map_pixels_to_physical(
        transform, field.positions_rc + field.displacement_rc
    )
    displacement_xy = end_xy - start_xy
    time_valid = jnp.isfinite(elapsed) & (elapsed > 0.0)
    valid = (
        field.valid
        & start_valid
        & end_valid
        & time_valid
        & jnp.all(jnp.isfinite(displacement_xy), axis=-1)
    )
    velocity_xy = displacement_xy / jnp.where(time_valid, elapsed, 1.0)
    return PhysicalPIVResult2D(
        start_xy,
        jnp.where(valid[..., None], displacement_xy, 0.0),
        jnp.where(valid[..., None], velocity_xy, 0.0),
        valid,
        field.field_id,
        transform.transform_id,
        transform.coordinate_contract.length_unit,
        time_unit,
        transform.coordinate_contract.reference_frame,
    )


__all__ = [
    "AffinePixelMap2D",
    "HomographyPixelMap2D",
    "convert_to_physical",
    "map_pixels_to_physical",
]
