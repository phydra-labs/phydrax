#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared semi-infinite periodic principal-layer lead embeddings."""

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import (
    DenseLinearOperator,
    DenseLU,
    DensePropertyVerificationPolicy,
    LinearSolvePolicy,
    LinearSystem,
    solve,
    verify_dense_properties,
)


class PeriodicPrincipalLayerLeadPlan(StrictModule, NonTrainableState):
    onsite: Array
    coupling: Array
    tolerance: float = eqx.field(static=True)
    fixed_point_tolerance: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        onsite: ArrayLike,
        coupling: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-12,
        fixed_point_tolerance: float = 1.0e-6,
        maximum_iterations: int = 100,
    ):
        onsite_ = np.asarray(onsite)
        coupling_ = np.asarray(coupling)
        tolerance_ = float(tolerance)
        fixed_tolerance = float(fixed_point_tolerance)
        iterations = int(maximum_iterations)
        if (
            onsite_.ndim != 2
            or onsite_.shape[0] != onsite_.shape[1]
            or coupling_.shape != onsite_.shape
            or np.any(~np.isfinite(onsite_))
            or np.any(~np.isfinite(coupling_))
            or not np.allclose(onsite_, np.conj(onsite_.T))
            or not isfinite(tolerance_)
            or tolerance_ <= 0.0
            or not isfinite(fixed_tolerance)
            or fixed_tolerance <= 0.0
            or iterations < 1
        ):
            raise ValueError("Principal-layer lead matrices or controls are invalid.")
        self.onsite = jnp.asarray(onsite_, dtype=jnp.complex128)
        self.coupling = jnp.asarray(coupling_, dtype=jnp.complex128)
        self.tolerance = tolerance_
        self.fixed_point_tolerance = fixed_tolerance
        self.maximum_iterations = iterations
        self.plan_id = canonical_fingerprint(
            {
                "kind": "periodic-principal-layer-lead-plan",
                "arrays": array_tree_fingerprint(
                    {"onsite": onsite_, "coupling": coupling_}
                ),
                "tolerance": tolerance_,
                "fixed_point_tolerance": fixed_tolerance,
                "maximum_iterations": iterations,
            }
        )


class PeriodicLeadEmbeddingResult(StrictModule, NonTrainableState):
    surface_green: Array
    self_energy: Array
    broadening: Array
    fixed_point_residual: Array
    decimation_residual: Array
    iterations: Array
    causal: Array
    broadening_positive_semidefinite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _dense_solve(matrix: Array, right: Array, /) -> Array:
    return solve(
        LinearSystem(DenseLinearOperator(matrix)),
        right,
        policy=LinearSolvePolicy(DenseLU()),
    ).value


def prepare_periodic_lead_embedding(
    plan: PeriodicPrincipalLayerLeadPlan,
    energy: ArrayLike,
    /,
    *,
    broadening: float = 1.0e-8,
) -> PeriodicLeadEmbeddingResult:
    if not isinstance(plan, PeriodicPrincipalLayerLeadPlan):
        raise TypeError("plan must be PeriodicPrincipalLayerLeadPlan.")
    energy_ = jnp.asarray(energy)
    eta = float(broadening)
    if energy_.shape != () or not isfinite(eta) or eta <= 0.0:
        raise ValueError("Lead energy must be scalar and broadening positive finite.")
    dimension = plan.onsite.shape[0]
    identity = jnp.eye(dimension, dtype=jnp.complex128)
    spectral = energy_.astype(jnp.complex128) + 1.0j * eta

    def body(index, state):
        onsite_bulk, onsite_surface, forward, backward, first_converged = state
        green = _dense_solve(spectral * identity - onsite_bulk, identity)
        forward_green = forward @ green
        backward_green = backward @ green
        surface_update = forward_green @ backward
        bulk_update = surface_update + backward_green @ forward
        next_surface = onsite_surface + surface_update
        next_bulk = onsite_bulk + bulk_update
        next_forward = forward_green @ forward
        next_backward = backward_green @ backward
        residual = jnp.sqrt(
            jnp.sum(jnp.abs(next_forward) ** 2) + jnp.sum(jnp.abs(next_backward) ** 2)
        )
        first_converged = jnp.where(
            (first_converged == plan.maximum_iterations) & (residual <= plan.tolerance),
            index + 1,
            first_converged,
        )
        return next_bulk, next_surface, next_forward, next_backward, first_converged

    initial = (
        plan.onsite,
        plan.onsite,
        plan.coupling,
        jnp.conj(plan.coupling.T),
        jnp.asarray(plan.maximum_iterations, dtype=jnp.int32),
    )
    bulk, surface, forward, backward, iterations = jax.lax.fori_loop(
        0,
        plan.maximum_iterations,
        body,
        initial,
    )
    del bulk
    surface_green = _dense_solve(spectral * identity - surface, identity)
    self_energy = jnp.conj(plan.coupling.T) @ surface_green @ plan.coupling
    gamma = 1.0j * (self_energy - jnp.conj(self_energy.T))
    fixed_point = (
        spectral * identity
        - plan.onsite
        - jnp.conj(plan.coupling.T) @ surface_green @ plan.coupling
    ) @ surface_green - identity
    fixed_residual = jnp.sqrt(jnp.sum(jnp.abs(fixed_point) ** 2))
    decimation_residual = jnp.sqrt(
        jnp.sum(jnp.abs(forward) ** 2) + jnp.sum(jnp.abs(backward) ** 2)
    )
    broadening_evidence = verify_dense_properties(
        gamma,
        policy=DensePropertyVerificationPolicy(
            require_hermitian=True,
            require_positive_semidefinite=True,
        ),
    )
    causal = jnp.all(jnp.imag(jnp.diag(surface_green)) <= plan.fixed_point_tolerance)
    successful = (
        jnp.isfinite(fixed_residual)
        & (fixed_residual <= plan.fixed_point_tolerance)
        & (decimation_residual <= plan.tolerance)
        & causal
        & broadening_evidence.successful
    )
    result_id = canonical_fingerprint(
        {
            "kind": "periodic-lead-embedding-result",
            "plan": plan.plan_id,
            "energy": array_tree_fingerprint(np.asarray(energy_)),
            "broadening": eta,
        }
    )
    return PeriodicLeadEmbeddingResult(
        surface_green,
        self_energy,
        gamma,
        fixed_residual,
        decimation_residual,
        iterations,
        causal,
        broadening_evidence.positive_semidefinite,
        successful,
        plan.plan_id,
        result_id,
    )


__all__ = [
    "PeriodicLeadEmbeddingResult",
    "PeriodicPrincipalLayerLeadPlan",
    "prepare_periodic_lead_embedding",
]
