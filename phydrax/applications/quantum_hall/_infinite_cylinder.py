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
    particle_charge_axis: int = eqx.field(static=True)
    charge_tolerance: float = eqx.field(static=True)
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
        *,
        charge_tolerance: float = 1.0e-10,
    ):
        numerator = int(filling_numerator)
        denominator = int(filling_denominator)
        circumference = float(circumference_in_magnetic_lengths)
        tolerance = float(charge_tolerance)
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
            or not isfinite(tolerance)
            or tolerance <= 0.0
            or any(
                tensor.shape[0] != 1 or tensor.shape[2] != 1
                for tensor in initial_state.state.tensors
            )
        ):
            raise ValueError(
                "Infinite Hall filling, circumference, or unit cell is invalid."
            )
        if (
            initial_state.group.group_id != hamiltonian.group.group_id
            or initial_state.charge_labels != hamiltonian.charge_labels
            or initial_state.physical_charges != hamiltonian.physical_charges
        ):
            raise ValueError(
                "Uniform Hall state and Hamiltonian charge identities differ."
            )
        if "particle-number" not in initial_state.charge_labels:
            raise ValueError("Uniform Hall charge labels must include particle-number.")
        particle_axis = initial_state.charge_labels.index("particle-number")
        if initial_state.group.components[particle_axis] is not None:
            raise ValueError("particle-number must be an integral U(1) charge.")
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
        self.particle_charge_axis = particle_axis
        self.charge_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "infinite-hall-cylinder-plan",
                "filling": (numerator, denominator),
                "circumference": circumference,
                "state": initial_state.state_id,
                "hamiltonian": hamiltonian.operator_id,
                "vumps_policy": vumps_policy.policy_id,
                "particle_charge_axis": particle_axis,
                "charge_tolerance": tolerance,
            }
        )


class InfiniteHallCylinderResult(StrictModule):
    vumps: UniformVUMPSResult
    transfer_eigenvalues: jnp.ndarray
    injectivity_gap: jnp.ndarray
    particle_charge_expectation: jnp.ndarray
    particle_charge_variance: jnp.ndarray
    charge_exact: jnp.ndarray
    successful: jnp.ndarray
    plan_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)


def _product_cell_charge_evidence(
    state,
    physical_charges,
    particle_axis: int,
    /,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    expectation = jnp.asarray(0.0, dtype=state.tensors[0].real.dtype)
    variance = jnp.asarray(0.0, dtype=state.tensors[0].real.dtype)
    for tensor, charges in zip(state.tensors, physical_charges, strict=True):
        amplitudes = tensor[0, :, 0]
        weights = jnp.abs(amplitudes) ** 2
        probabilities = weights / jnp.maximum(
            jnp.sum(weights), jnp.finfo(weights.dtype).tiny
        )
        values = jnp.asarray(
            tuple(value[particle_axis] for value in charges),
            dtype=probabilities.dtype,
        )
        local_mean = jnp.sum(probabilities * values)
        expectation = expectation + local_mean
        variance = variance + jnp.sum(probabilities * (values - local_mean) ** 2)
    return expectation, variance


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
    charge_expectation, charge_variance = _product_cell_charge_evidence(
        solved.state,
        plan.initial_state.physical_charges,
        plan.particle_charge_axis,
    )
    charge_exact = (
        jnp.isfinite(charge_expectation)
        & jnp.isfinite(charge_variance)
        & (jnp.abs(charge_expectation - plan.filling_numerator) <= plan.charge_tolerance)
        & (charge_variance <= plan.charge_tolerance)
    )
    successful = solved.successful & fixed.successful & charge_exact
    return InfiniteHallCylinderResult(
        solved,
        fixed.eigenvalues,
        fixed.injectivity_gap,
        charge_expectation,
        charge_variance,
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
