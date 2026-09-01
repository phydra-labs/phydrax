#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    prepare,
    solve,
    TolerancePolicy,
)


class ElectrostaticConductorState(StrictModule):
    collected_charge: Array
    potential: Array


class ConductorCircuitSolveResult(StrictModule):
    potential: Array
    conductor_potential: Array
    conductor_charge: Array
    multiplier: Array
    poisson_residual: Array
    constraint_residual: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ElectrostaticConductorCoupling(StrictModule, NonTrainableState):
    """Fixed-size equipotential/conductor-charge KKT coupling."""

    stiffness: Array
    constraint: Array
    prepared_linear: object
    tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        stiffness: ArrayLike,
        constraint_matrix: ArrayLike,
        /,
        *,
        tolerance: float = 1.0e-10,
        maximum_iterations: int = 500,
    ):
        matrix = np.asarray(stiffness, dtype=float)
        constraint = np.asarray(constraint_matrix, dtype=float)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("stiffness must be square.")
        if constraint.ndim != 2 or constraint.shape[1] != matrix.shape[0]:
            raise ValueError("constraint_matrix must act on potential DOFs.")
        if (
            constraint.shape[0] == 0
            or np.any(~np.isfinite(matrix))
            or np.any(~np.isfinite(constraint))
        ):
            raise ValueError("Conductor KKT inputs are invalid.")
        zero = np.zeros((constraint.shape[0], constraint.shape[0]), dtype=float)
        kkt = np.block([[matrix, constraint.T], [constraint, zero]])
        operator = DenseLinearOperator(
            jnp.asarray(kkt),
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=False,
                evidence={"self_adjoint": "construction"},
            ),
            operator_id=canonical_fingerprint(
                {
                    "kind": "electrostatic-conductor-kkt",
                    "matrix": array_tree_fingerprint(kkt),
                }
            ),
        )
        policy = LinearSolvePolicy(
            GMRES(restart=min(64, kkt.shape[0])),
            tolerance=TolerancePolicy(
                relative=float(tolerance),
                absolute=float(tolerance),
                max_steps=int(maximum_iterations),
            ),
        )
        prepared = prepare(LinearSystem(operator), policy)
        self.stiffness = jnp.asarray(matrix)
        self.constraint = jnp.asarray(constraint)
        self.prepared_linear = prepared
        self.tolerance = float(tolerance)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "electrostatic-conductor-coupling",
                "stiffness": array_tree_fingerprint(matrix),
                "constraint": array_tree_fingerprint(constraint),
                "linear": prepared.plan.plan_id,
            }
        )

    def initialize(self) -> ElectrostaticConductorState:
        return ElectrostaticConductorState(
            jnp.zeros((self.constraint.shape[0],), dtype=self.stiffness.dtype),
            jnp.zeros((self.constraint.shape[0],), dtype=self.stiffness.dtype),
        )

    def solve(
        self,
        nodal_charge: ArrayLike,
        prescribed_potential: ArrayLike,
        state: ElectrostaticConductorState,
        /,
        *,
        collected_charge: ArrayLike | None = None,
    ) -> tuple[ElectrostaticConductorState, ConductorCircuitSolveResult]:
        charge = jnp.asarray(nodal_charge, dtype=self.stiffness.dtype)
        prescribed = jnp.asarray(prescribed_potential, dtype=self.stiffness.dtype)
        if charge.shape != (self.stiffness.shape[0],) or prescribed.shape != (
            self.constraint.shape[0],
        ):
            raise ValueError("Conductor solve vectors have incompatible shapes.")
        collected = (
            jnp.zeros_like(state.collected_charge)
            if collected_charge is None
            else jnp.asarray(collected_charge, dtype=self.stiffness.dtype)
        )
        conductor_charge = state.collected_charge + collected
        rhs = jnp.concatenate((charge, prescribed + conductor_charge))
        linear = solve(self.prepared_linear, rhs)
        potential = linear.value[: self.stiffness.shape[0]]
        multiplier = linear.value[self.stiffness.shape[0] :]
        poisson = self.stiffness @ potential + self.constraint.T @ multiplier - charge
        constraints = self.constraint @ potential - prescribed - conductor_charge
        poisson_norm = jnp.sqrt(jnp.sum(poisson**2))
        constraint_norm = jnp.sqrt(jnp.sum(constraints**2))
        finite = jnp.all(jnp.isfinite(linear.value)) & jnp.isfinite(
            poisson_norm + constraint_norm
        )
        successful = (
            linear.successful
            & finite
            & (
                poisson_norm
                <= self.tolerance * jnp.maximum(1.0, jnp.sqrt(jnp.sum(charge**2)))
            )
            & (
                constraint_norm
                <= self.tolerance * jnp.maximum(1.0, jnp.sqrt(jnp.sum(prescribed**2)))
            )
        )
        candidate_state = ElectrostaticConductorState(
            conductor_charge, self.constraint @ potential
        )
        accepted = ElectrostaticConductorState(
            jnp.where(
                successful, candidate_state.collected_charge, state.collected_charge
            ),
            jnp.where(successful, candidate_state.potential, state.potential),
        )
        return accepted, ConductorCircuitSolveResult(
            potential,
            self.constraint @ potential,
            conductor_charge,
            multiplier,
            poisson_norm,
            constraint_norm,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "ConductorCircuitSolveResult",
    "ElectrostaticConductorCoupling",
    "ElectrostaticConductorState",
]
