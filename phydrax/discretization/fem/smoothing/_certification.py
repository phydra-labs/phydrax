#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import ArrayLike

from ._common import (
    SmoothingEnergyEvidence,
    SmoothingEvidence,
    SmoothingPatchGeometry,
    SmoothingPatchLayout,
)
from ._moments import boundary_moment


def certify_smoothing_operator(
    layout: SmoothingPatchLayout,
    geometry: SmoothingPatchGeometry,
    stiffness: ArrayLike,
    constrained_dofs: ArrayLike,
    expected_total_measure: ArrayLike,
    expected_rigid_modes: int,
    /,
    *,
    eigen_tolerance: float = 1.0e-10,
    energy_evidence: SmoothingEnergyEvidence = "none",
) -> SmoothingEvidence:
    matrix = jnp.asarray(stiffness)
    constrained = jnp.asarray(constrained_dofs, dtype=bool)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Smoothing stiffness must be one square matrix.")
    if constrained.shape != (matrix.shape[0],):
        raise ValueError("Constraint mask must match smoothing stiffness size.")
    free = jnp.flatnonzero(~constrained, size=matrix.shape[0], fill_value=0)
    free_count = int(jnp.sum(~constrained))
    reduced = matrix[free[:free_count, None], free[None, :free_count]]
    eigenvalues = jnp.linalg.eigvalsh(0.5 * (reduced + reduced.T))
    scale = jnp.maximum(jnp.max(jnp.abs(eigenvalues)), 1.0)
    near_zero = int(jnp.sum(jnp.abs(eigenvalues) <= eigen_tolerance * scale))
    closure = jnp.sum(
        geometry.boundary_lengths[..., None] * geometry.boundary_normals,
        axis=1,
    )
    affine = jnp.sum(boundary_moment(layout, geometry), axis=1)
    total_measure = jnp.sum(geometry.area)
    partition_defect = jnp.abs(total_measure - jnp.asarray(expected_total_measure))
    return SmoothingEvidence(
        geometry.valid,
        jnp.sqrt(jnp.sum(closure**2, axis=-1)),
        partition_defect,
        jnp.sqrt(jnp.sum(affine**2, axis=-1)),
        expected_rigid_modes,
        max(near_zero - expected_rigid_modes, 0),
        jnp.min(eigenvalues),
        energy_evidence=energy_evidence,
    )


__all__ = ["certify_smoothing_operator"]
