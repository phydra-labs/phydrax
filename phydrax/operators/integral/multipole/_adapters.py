#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Layer-potential and QBX adapters for prepared spherical multipoles."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._laplace3d import (
    LaplaceMultipoleEvaluation3D,
    MultipoleFarLocal3D,
    PreparedLaplaceMultipole3D,
)


def _weighted_strengths(
    source_density: ArrayLike,
    source_weights: ArrayLike,
    source_count: int,
    /,
):
    density = jnp.asarray(source_density)
    weights = jnp.asarray(source_weights, dtype=jnp.real(density).dtype)
    if density.ndim < 1 or density.shape[0] != source_count:
        raise ValueError("source_density must begin with source_count.")
    if weights.shape != (source_count,):
        raise ValueError("source_weights must have shape (source_count,).")
    return density * weights.reshape((source_count,) + (1,) * (density.ndim - 1))


def evaluate_laplace_layer_multipole_3d(
    prepared: PreparedLaplaceMultipole3D,
    source_positions: ArrayLike,
    source_density: ArrayLike,
    source_weights: ArrayLike,
    target_positions: ArrayLike | None = None,
    /,
    *,
    source_normals: ArrayLike | None = None,
    active_mask: ArrayLike | None = None,
    target_source_indices: ArrayLike | None = None,
) -> LaplaceMultipoleEvaluation3D:
    """Evaluate weighted single- or source-normal double-layer point data."""
    if not isinstance(prepared, PreparedLaplaceMultipole3D):
        raise TypeError("prepared must be PreparedLaplaceMultipole3D.")
    strengths = _weighted_strengths(
        source_density,
        source_weights,
        prepared.plan.source_capacity,
    )
    return prepared.evaluate(
        source_positions,
        strengths,
        target_positions,
        source_normals=source_normals,
        active_mask=active_mask,
        target_source_indices=target_source_indices,
    )


def prepare_laplace_qbx_far_local_3d(
    prepared: PreparedLaplaceMultipole3D,
    source_positions: ArrayLike,
    source_density: ArrayLike,
    source_weights: ArrayLike,
    expansion_centers: ArrayLike,
    /,
    *,
    source_normals: ArrayLike | None = None,
    active_mask: ArrayLike | None = None,
) -> MultipoleFarLocal3D:
    """Return far-only local coefficients at prepared three-dimensional QBX centers."""
    if not isinstance(prepared, PreparedLaplaceMultipole3D):
        raise TypeError("prepared must be PreparedLaplaceMultipole3D.")
    strengths = _weighted_strengths(
        source_density,
        source_weights,
        prepared.plan.source_capacity,
    )
    far = prepared.far_local(
        source_positions,
        strengths,
        expansion_centers,
        source_normals=source_normals,
        active_mask=active_mask,
    )
    centers = jnp.asarray(expansion_centers)
    if centers.shape != (prepared.plan.target_capacity, 3):
        raise ValueError("expansion_centers must have shape (target_capacity, 3).")
    hierarchy = prepared.topology.hierarchy
    target_leaf = hierarchy.logical_point_leaf_slots[
        prepared.plan.source_capacity : prepared.plan.source_capacity
        + prepared.plan.target_capacity
    ]
    leaf_centers = hierarchy.node_centers[jnp.maximum(target_leaf, 0)]
    shifted = jax.vmap(prepared.l2l)(far.coefficients, leaf_centers, centers)
    return MultipoleFarLocal3D(
        coefficients=shifted,
        truncation=far.truncation,
        capacity=far.capacity,
        m2m_count=far.m2m_count,
        m2l_count=far.m2l_count,
        l2l_count=far.l2l_count + prepared.plan.target_capacity,
        successful=far.successful,
        prepared_id=far.prepared_id,
    )


__all__ = [
    "evaluate_laplace_layer_multipole_3d",
    "prepare_laplace_qbx_far_local_3d",
]
