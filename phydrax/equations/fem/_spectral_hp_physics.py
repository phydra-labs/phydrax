#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule
from ...discretization.fem._mortar import (
    FiniteElementMortarMetricData,
)
from ._conservation import certify_dgsem_mortar_compatibility


class EntropyStableWallEvidence(StrictModule):
    mass_flux: Array
    entropy_flux: Array
    passed: Array


def entropy_stable_wall_evidence(
    state: ArrayLike,
    flux: ArrayLike,
    entropy_variables: ArrayLike,
    normal: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
) -> EntropyStableWallEvidence:
    state_ = jnp.asarray(state)
    flux_ = jnp.asarray(flux)
    entropy = jnp.asarray(entropy_variables)
    normal_ = jnp.asarray(normal)
    mass_flux = flux_[..., 0]
    entropy_flux = jnp.sum(entropy * flux_, axis=-1)
    momentum = state_[..., 1 : 1 + normal_.shape[-1]]
    wall_velocity = jnp.sum(momentum * normal_, axis=-1)
    passed = (
        (jnp.max(jnp.abs(mass_flux)) <= tolerance)
        & (jnp.max(jnp.abs(wall_velocity)) <= tolerance)
        & (jnp.max(entropy_flux) <= tolerance)
    )
    return EntropyStableWallEvidence(mass_flux, entropy_flux, passed)


def derived_mortar_entropy_defect(
    left_state: ArrayLike,
    right_state: ArrayLike,
    left_entropy_variables: ArrayLike,
    right_entropy_variables: ArrayLike,
    numerical_flux: ArrayLike,
    left_entropy_potential: ArrayLike,
    right_entropy_potential: ArrayLike,
    /,
) -> Array:
    left = jnp.asarray(left_state)
    right = jnp.asarray(right_state)
    left_variables = jnp.asarray(left_entropy_variables)
    right_variables = jnp.asarray(right_entropy_variables)
    flux = jnp.asarray(numerical_flux)
    left_potential = jnp.asarray(left_entropy_potential)
    right_potential = jnp.asarray(right_entropy_potential)
    if (
        left.shape != right.shape
        or left.shape != left_variables.shape
        or left.shape != right_variables.shape
        or left.shape != flux.shape
    ):
        raise ValueError("Entropy mortar states, variables, and flux must match.")
    return jnp.sum((right_variables - left_variables) * flux, axis=-1) - (
        right_potential - left_potential
    )


def certify_derived_dgsem_mortar(
    mortar,
    metric: FiniteElementMortarMetricData,
    left_state: ArrayLike,
    right_state: ArrayLike,
    left_entropy_variables: ArrayLike,
    right_entropy_variables: ArrayLike,
    numerical_flux: ArrayLike,
    left_entropy_potential: ArrayLike,
    right_entropy_potential: ArrayLike,
    /,
    *,
    tolerance: float = 1.0e-10,
):
    defect = derived_mortar_entropy_defect(
        left_state,
        right_state,
        left_entropy_variables,
        right_entropy_variables,
        numerical_flux,
        left_entropy_potential,
        right_entropy_potential,
    )
    return certify_dgsem_mortar_compatibility(
        mortar,
        metric,
        entropy_error=defect,
        tolerance=tolerance,
    )


__all__ = [
    "certify_derived_dgsem_mortar",
    "derived_mortar_entropy_defect",
    "EntropyStableWallEvidence",
    "entropy_stable_wall_evidence",
]
