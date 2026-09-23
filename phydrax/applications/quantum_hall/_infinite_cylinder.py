#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Thermodynamic-length quantum Hall cylinders through native uniform VUMPS."""

from __future__ import annotations

from math import gcd, isfinite

import equinox as eqx
import jax.numpy as jnp

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...solver import (
    solve_uniform_vumps,
    UniformVUMPSPolicy,
    UniformVUMPSProblem,
    UniformVUMPSResult,
)
from ...tensor_network import (
    uniform_transfer_fixed_points,
    UniformAbelianMatrixProductOperator,
    UniformAbelianMatrixProductState,
    UniformTransferPolicy,
)


class InfiniteHallCylinderPlan(StrictModule, NonTrainableState):
    filling_numerator: int = eqx.field(static=True)
    filling_denominator: int = eqx.field(static=True)
    circumference_in_magnetic_lengths: float = eqx.field(static=True)
    initial_state: UniformAbelianMatrixProductState
    hamiltonian: UniformAbelianMatrixProductOperator
    vumps_policy: UniformVUMPSPolicy
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        filling_numerator: int,
        filling_denominator: int,
        circumference_in_magnetic_lengths: float,
        initial_state: UniformAbelianMatrixProductState,
        hamiltonian: UniformAbelianMatrixProductOperator,
        vumps_policy: UniformVUMPSPolicy,
        /,
    ):
        numerator = int(filling_numerator)
        denominator = int(filling_denominator)
        circumference = float(circumference_in_magnetic_lengths)
        if not isinstance(
            initial_state, UniformAbelianMatrixProductState
        ) or not isinstance(hamiltonian, UniformAbelianMatrixProductOperator):
            raise TypeError(
                "Infinite Hall cylinders require uniform Abelian MPS/MPO values."
            )
        if not isinstance(vumps_policy, UniformVUMPSPolicy):
            raise TypeError("vumps_policy must be UniformVUMPSPolicy.")
        if (
            numerator < 1
            or denominator < 1
            or numerator > denominator
            or gcd(numerator, denominator) != 1
            or not isfinite(circumference)
            or circumference <= 0.0
            or initial_state.state.unit_cell_size != denominator
            or hamiltonian.operator.unit_cell_size != denominator
        ):
            raise ValueError(
                "Infinite Hall filling, circumference, or unit cell is invalid."
            )
        particle_axis = 0
        if initial_state.unit_cell_charge[particle_axis] != numerator:
            raise ValueError(
                "Uniform state unit-cell particle charge does not match filling p/q."
            )
        self.filling_numerator = numerator
        self.filling_denominator = denominator
        self.circumference_in_magnetic_lengths = circumference
        self.initial_state = initial_state
        self.hamiltonian = hamiltonian
        self.vumps_policy = vumps_policy
        self.plan_id = canonical_fingerprint(
            {
                "kind": "infinite-hall-cylinder-plan",
                "filling": (numerator, denominator),
                "circumference": circumference,
                "state": initial_state.state_id,
                "hamiltonian": hamiltonian.operator_id,
                "vumps_policy": vumps_policy.policy_id,
            }
        )


class InfiniteHallCylinderResult(StrictModule):
    vumps: UniformVUMPSResult
    transfer_eigenvalues: jnp.ndarray
    injectivity_gap: jnp.ndarray
    charge_exact: jnp.ndarray
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def solve_infinite_hall_cylinder(
    plan: InfiniteHallCylinderPlan,
    /,
) -> InfiniteHallCylinderResult:
    if not isinstance(plan, InfiniteHallCylinderPlan):
        raise TypeError("plan must be InfiniteHallCylinderPlan.")
    solved = solve_uniform_vumps(
        UniformVUMPSProblem(
            plan.initial_state,
            plan.hamiltonian,
            problem_id=plan.plan_id,
        ),
        plan.vumps_policy,
    )
    fixed = uniform_transfer_fixed_points(
        solved.state,
        UniformTransferPolicy(maximum_modes=8),
    )
    charge_exact = jnp.asarray(
        plan.initial_state.unit_cell_charge[0] == plan.filling_numerator
    )
    successful = solved.successful & fixed.successful & charge_exact
    return InfiniteHallCylinderResult(
        solved,
        fixed.eigenvalues,
        fixed.injectivity_gap,
        charge_exact,
        successful,
        plan.plan_id,
        canonical_fingerprint(
            {
                "kind": "infinite-hall-cylinder-result",
                "plan": plan.plan_id,
                "prepared": solved.prepared_id,
            }
        ),
    )


__all__ = [
    "InfiniteHallCylinderPlan",
    "InfiniteHallCylinderResult",
    "solve_infinite_hall_cylinder",
]
