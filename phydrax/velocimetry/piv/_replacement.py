#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp

from ..imaging import DenseDisplacementField2D
from ._types import ReplacementEvidence2D
from ._validation import _masked_component_median, _neighborhood


def replace_invalid_vectors(
    field: DenseDisplacementField2D,
    /,
    *,
    radius: int = 1,
    iterations: int = 2,
    minimum_neighbors: int = 1,
) -> tuple[DenseDisplacementField2D, ReplacementEvidence2D]:
    """Return a new field filled by deterministic local medians where supported."""
    if not isinstance(field, DenseDisplacementField2D):
        raise TypeError("field must be a DenseDisplacementField2D.")
    if field.valid.ndim != 2:
        raise ValueError("Replacement requires a two-dimensional vector grid.")
    radius_ = int(radius)
    iterations_ = int(iterations)
    minimum_ = int(minimum_neighbors)
    if radius_ < 1 or iterations_ < 0 or minimum_ < 1:
        raise ValueError("Replacement radius/iterations/neighbors are invalid.")
    original_valid = field.valid
    replacement_iteration = jnp.full(field.valid.shape, -1, dtype=jnp.int32)
    contributing = jnp.zeros(field.valid.shape, dtype=jnp.int32)

    def body(
        iteration: int,
        state: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        vectors, valid, iteration_map, neighbor_evidence = state
        samples, masks = _neighborhood(vectors, valid, radius=radius_)
        count = jnp.sum(masks, axis=-1, dtype=jnp.int32)
        median = _masked_component_median(samples, masks)
        selected = (~valid) & (count >= minimum_) & jnp.all(jnp.isfinite(median), axis=-1)
        next_vectors = jnp.where(selected[..., None], median, vectors)
        next_valid = valid | selected
        next_iteration = jnp.where(selected, iteration + 1, iteration_map)
        next_evidence = jnp.where(selected, count, neighbor_evidence)
        return next_vectors, next_valid, next_iteration, next_evidence

    vectors, valid, replacement_iteration, contributing = jax.lax.fori_loop(
        0,
        iterations_,
        body,
        (field.displacement_rc, field.valid, replacement_iteration, contributing),
    )
    replaced_mask = (~original_valid) & valid
    evidence = ReplacementEvidence2D(
        original_valid,
        replaced_mask,
        replacement_iteration,
        contributing,
        ~valid,
    )
    replaced = DenseDisplacementField2D(
        field.positions_rc,
        vectors,
        valid,
        geometry_id=field.geometry_id,
        provenance=field.provenance + ("neighborhood-median-replacement",),
    )
    return replaced, evidence


__all__ = ["replace_invalid_vectors"]
