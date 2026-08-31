#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array

from ..imaging import DenseDisplacementField2D
from ._types import PIVQuality2D, ValidationEvidence2D


def _neighborhood(
    values: Array,
    valid: Array,
    /,
    *,
    radius: int,
) -> tuple[Array, Array]:
    rows, columns = valid.shape
    padded_values = jnp.pad(values, ((radius, radius), (radius, radius), (0, 0)))
    padded_valid = jnp.pad(valid, ((radius, radius), (radius, radius)))
    samples = []
    masks = []
    for row_offset in range(2 * radius + 1):
        for column_offset in range(2 * radius + 1):
            if row_offset == radius and column_offset == radius:
                continue
            samples.append(
                padded_values[
                    row_offset : row_offset + rows,
                    column_offset : column_offset + columns,
                ]
            )
            masks.append(
                padded_valid[
                    row_offset : row_offset + rows,
                    column_offset : column_offset + columns,
                ]
            )
    return jnp.stack(samples, axis=-2), jnp.stack(masks, axis=-1)


def _masked_component_median(samples: Array, mask: Array) -> Array:
    safe = jnp.where(mask[..., None], samples, jnp.inf)
    ordered = jnp.sort(safe, axis=-2)
    count = jnp.sum(mask, axis=-1, dtype=jnp.int32)
    index = jnp.maximum((count - 1) // 2, 0)
    gather_index = jnp.broadcast_to(index[..., None, None], index.shape + (1, 2))
    return jnp.take_along_axis(ordered, gather_index, axis=-2)[..., 0, :]


def validate_field(
    field: DenseDisplacementField2D,
    quality: PIVQuality2D,
    /,
    *,
    maximum_displacement: tuple[float, float],
    minimum_correlation: float,
    minimum_peak_ratio: float,
    radius: int,
    minimum_neighbors: int,
    median_threshold: float,
    median_epsilon: float,
) -> tuple[DenseDisplacementField2D, ValidationEvidence2D]:
    """Validate vectors while retaining every independent decision component."""
    if not isinstance(field, DenseDisplacementField2D):
        raise TypeError("field must be a DenseDisplacementField2D.")
    if field.valid.ndim != 2:
        raise ValueError("PIV validation requires a two-dimensional vector grid.")
    vectors = field.displacement_rc
    finite = jnp.all(jnp.isfinite(vectors), axis=-1)
    within_limit = (jnp.abs(vectors[..., 0]) <= maximum_displacement[0]) & (
        jnp.abs(vectors[..., 1]) <= maximum_displacement[1]
    )
    correlation_accepted = jnp.isfinite(quality.primary_peak) & (
        quality.primary_peak >= minimum_correlation
    )
    peak_ratio_accepted = jnp.isfinite(quality.peak_ratio) & (
        quality.peak_ratio >= minimum_peak_ratio
    )
    neighbor_values, neighbor_mask = _neighborhood(
        vectors, field.valid & finite, radius=radius
    )
    neighbor_count = jnp.sum(neighbor_mask, axis=-1, dtype=jnp.int32)
    local_median = _masked_component_median(neighbor_values, neighbor_mask)
    residual = jnp.sqrt(jnp.sum((vectors - local_median) ** 2, axis=-1))
    neighbor_residual = jnp.sqrt(
        jnp.sum((neighbor_values - local_median[..., None, :]) ** 2, axis=-1)
    )
    safe_residual = jnp.where(neighbor_mask, neighbor_residual, jnp.inf)
    ordered_residual = jnp.sort(safe_residual, axis=-1)
    median_index = jnp.maximum((neighbor_count - 1) // 2, 0)
    scale = jnp.take_along_axis(ordered_residual, median_index[..., None], axis=-1)[
        ..., 0
    ]
    threshold = median_threshold * (
        jnp.where(jnp.isfinite(scale), scale, 0.0) + median_epsilon
    )
    enough_neighbors = neighbor_count >= minimum_neighbors
    local_accepted = (residual <= threshold) & enough_neighbors
    if minimum_neighbors == 0:
        local_accepted = jnp.ones_like(local_accepted)
    valid = (
        field.valid
        & finite
        & within_limit
        & correlation_accepted
        & peak_ratio_accepted
        & local_accepted
    )
    evidence = ValidationEvidence2D(
        finite,
        within_limit,
        correlation_accepted,
        peak_ratio_accepted,
        local_accepted,
        neighbor_count,
        local_median,
        residual,
        threshold,
        valid,
    )
    validated = DenseDisplacementField2D(
        field.positions_rc,
        field.displacement_rc,
        valid,
        geometry_id=field.geometry_id,
        provenance=field.provenance + ("validated",),
    )
    return validated, evidence


__all__ = ["validate_field"]
