#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact/fixed-shot measurements and bounded mid-circuit classical control."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key
from opt_einsum import contract

from .._doc import DOC_KEY0
from .._sampling import derive_key, SampleAddress
from .._strict import StrictModule
from ..linalg import HermitianSpectrum
from ._quantum_program import (
    DenseQuantumProgramResult,
    execute_dense_quantum_program,
    PreparedDenseQuantumProgram,
)


_SHOT_ADDRESS = SampleAddress(
    "quantum", "program-measurement", target="outcome", role="shot"
)


class QuantumMeasurementPlan(StrictModule):
    effects: Array
    physicality_residuals: Array
    completeness_residual: Array
    valid: Array
    shots: int = eqx.field(static=True)
    outcome_count: int = eqx.field(static=True)
    measurement_id: str = eqx.field(static=True)

    def __init__(
        self,
        effects: ArrayLike,
        /,
        *,
        shots: int = 0,
        tolerance: float = 1e-8,
        measurement_id: str,
    ):
        values = jnp.asarray(effects)
        if values.ndim != 3 or values.shape[0] < 1 or values.shape[1] != values.shape[2]:
            raise ValueError(
                "POVM effects require shape (outcomes, dimension, dimension)."
            )
        if not jnp.issubdtype(values.dtype, jnp.complexfloating):
            raise TypeError("POVM effects must use complex floating coordinates.")
        count = int(shots)
        if count < 0 or not isinstance(measurement_id, str) or not measurement_id:
            raise ValueError("shots/measurement_id are invalid.")
        spectra = tuple(
            HermitianSpectrum(effect, tolerance=tolerance) for effect in values
        )
        residuals = jnp.stack([spectrum.hermiticity_residual for spectrum in spectra])
        positive = jnp.stack(
            [spectrum.minimum_eigenvalue >= -tolerance for spectrum in spectra]
        )
        completeness = jnp.max(
            jnp.abs(
                jnp.sum(values, axis=0) - jnp.eye(values.shape[1], dtype=values.dtype)
            )
        )
        self.effects = values
        self.physicality_residuals = residuals
        self.completeness_residual = completeness
        self.valid = (
            jnp.all(positive)
            & jnp.all(jnp.stack([value.valid for value in spectra]))
            & (completeness <= tolerance)
        )
        self.shots = count
        self.outcome_count = int(values.shape[0])
        self.measurement_id = measurement_id


class QuantumMeasurementResult(StrictModule):
    probabilities: Array
    counts: Array
    sampled_outcomes: Array
    probability_sum_residual: Array
    negative_probability_residual: Array
    valid: Array
    root_key: Array
    measurement_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def measure_dense_quantum_program(
    prepared: PreparedDenseQuantumProgram,
    result: DenseQuantumProgramResult,
    measurement: QuantumMeasurementPlan,
    /,
    *,
    key: Key[Array, ""] = DOC_KEY0,
) -> QuantumMeasurementResult:
    if (
        not isinstance(prepared, PreparedDenseQuantumProgram)
        or not isinstance(result, DenseQuantumProgramResult)
        or not isinstance(measurement, QuantumMeasurementPlan)
    ):
        raise TypeError("prepared/result/measurement types are invalid.")
    if measurement.effects.shape[-1] != prepared.plan.cost.total_dimension:
        raise ValueError("POVM dimension does not match the prepared Hilbert layout.")
    state = result.final_state
    if state.ndim != (1 if prepared.plan.state_kind == "state-vector" else 2):
        raise ValueError("Circuit measurement currently requires one unbatched state.")
    if prepared.plan.state_kind == "state-vector":
        probabilities = jnp.real(
            jax.vmap(lambda effect: jnp.vdot(state, effect @ state))(measurement.effects)
        )
    else:
        probabilities = jnp.real(
            jax.vmap(lambda effect: jnp.trace(effect @ state))(measurement.effects)
        )
    negative = jnp.maximum(-jnp.min(probabilities), 0.0)
    sum_residual = jnp.abs(jnp.sum(probabilities) - 1.0)
    safe_probabilities = jnp.where(probabilities >= 0.0, probabilities, 0.0)
    safe_probabilities = safe_probabilities / jnp.sum(safe_probabilities)
    if measurement.shots:
        shot_indices = jnp.arange(measurement.shots, dtype=jnp.uint32)
        keys = jax.vmap(lambda shot: derive_key(key, _SHOT_ADDRESS, shot, 0))(
            shot_indices
        )
        outcomes = jax.vmap(
            lambda shot_key: jr.choice(
                shot_key, measurement.outcome_count, p=safe_probabilities
            )
        )(keys)
        counts = jnp.bincount(outcomes, length=measurement.outcome_count)
    else:
        outcomes = jnp.empty((0,), dtype=jnp.int32)
        counts = jnp.zeros((measurement.outcome_count,), dtype=jnp.int32)
    valid = (
        measurement.valid
        & result.diagnostics.successful
        & jnp.all(jnp.isfinite(probabilities))
        & (negative <= prepared.plan.policy.positivity_tolerance)
        & (sum_residual <= prepared.plan.policy.trace_tolerance)
    )
    return QuantumMeasurementResult(
        probabilities=probabilities,
        counts=counts,
        sampled_outcomes=outcomes,
        probability_sum_residual=sum_residual,
        negative_probability_residual=negative,
        valid=valid,
        root_key=jnp.asarray(key),
        measurement_id=measurement.measurement_id,
        claim="finite-povm-exact-probabilities-and-semantic-fixed-shots",
    )


class MidCircuitQuantumPlan(StrictModule):
    prefix: PreparedDenseQuantumProgram
    measurement_kraus: Array
    branches: tuple[PreparedDenseQuantumProgram, ...]
    completeness_residual: Array
    valid: Array
    outcome_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)

    def __init__(
        self,
        prefix: PreparedDenseQuantumProgram,
        measurement_kraus: ArrayLike,
        branches: Sequence[PreparedDenseQuantumProgram],
        /,
        *,
        tolerance: float = 1e-8,
        plan_id: str,
    ):
        if not isinstance(prefix, PreparedDenseQuantumProgram):
            raise TypeError("prefix must be PreparedDenseQuantumProgram.")
        operators = jnp.asarray(measurement_kraus)
        selected = tuple(branches)
        dimension = prefix.plan.cost.total_dimension
        if (
            operators.ndim != 3
            or operators.shape[1:] != (dimension, dimension)
            or operators.shape[0] < 1
        ):
            raise ValueError("measurement_kraus must have global shape (outcomes,d,d).")
        if len(selected) != operators.shape[0] or any(
            not isinstance(branch, PreparedDenseQuantumProgram) for branch in selected
        ):
            raise ValueError(
                "One prepared density-matrix branch is required per outcome."
            )
        if any(
            branch.plan.state_kind != "density-matrix"
            or branch.plan.layout.layout_id != prefix.plan.layout.layout_id
            for branch in selected
        ):
            raise ValueError(
                "Mid-circuit branches must be density programs on the same Hilbert layout."
            )
        completeness = contract("kai,kaj->ij", jnp.conj(operators), operators)
        residual = jnp.max(
            jnp.abs(completeness - jnp.eye(dimension, dtype=operators.dtype))
        )
        self.prefix = prefix
        self.measurement_kraus = operators
        self.branches = selected
        self.completeness_residual = residual
        self.valid = (
            prefix.operations_valid
            & jnp.all(jnp.isfinite(operators))
            & jnp.isfinite(residual)
            & (residual <= tolerance)
        )
        self.outcome_count = int(operators.shape[0])
        self.plan_id = plan_id
        self.claim = "finite-outcome-mid-circuit-measurement-with-explicit-branches"


class MidCircuitQuantumResult(StrictModule):
    prefix_result: DenseQuantumProgramResult
    branch_densities: Array
    outcome_probabilities: Array
    branch_status: Array
    valid_branches: Array
    valid: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def execute_mid_circuit_quantum_plan(
    plan: MidCircuitQuantumPlan,
    initial_state: ArrayLike,
    /,
) -> MidCircuitQuantumResult:
    """Execute every bounded classical branch; no hidden outcome/decomposition choice."""
    if not isinstance(plan, MidCircuitQuantumPlan):
        raise TypeError("plan must be MidCircuitQuantumPlan.")
    prefix = execute_dense_quantum_program(plan.prefix, initial_state)
    if plan.prefix.plan.state_kind == "state-vector":
        vector = prefix.final_state
        density = vector[:, None] * jnp.conj(vector[None, :])
    else:
        density = prefix.final_state
    branch_densities = []
    probabilities = []
    statuses = []
    branch_execution_valid = []
    for operator, branch in zip(plan.measurement_kraus, plan.branches, strict=True):
        unnormalized = operator @ density @ jnp.conj(operator.T)
        probability = jnp.real(jnp.trace(unnormalized))
        normalized = unnormalized / jnp.where(probability > 0.0, probability, 1.0)
        result = execute_dense_quantum_program(branch, normalized)
        branch_densities.append(result.final_state)
        probabilities.append(probability)
        statuses.append(result.diagnostics.status)
        branch_execution_valid.append(
            result.diagnostics.successful & jnp.all(jnp.isfinite(result.final_state))
        )
    probabilities_ = jnp.stack(probabilities)
    branch_execution_valid_ = jnp.stack(branch_execution_valid)
    active = jnp.isfinite(probabilities_) & (
        probabilities_ > plan.prefix.plan.policy.positivity_tolerance
    )
    valid_ = active & branch_execution_valid_
    return MidCircuitQuantumResult(
        prefix_result=prefix,
        branch_densities=jnp.stack(branch_densities),
        outcome_probabilities=probabilities_,
        branch_status=jnp.stack(statuses),
        valid_branches=valid_,
        valid=plan.valid
        & prefix.diagnostics.successful
        & jnp.all(jnp.isfinite(probabilities_))
        & jnp.all(probabilities_ >= -plan.prefix.plan.policy.positivity_tolerance)
        & jnp.all((~active) | branch_execution_valid_)
        & (
            jnp.abs(jnp.sum(probabilities_) - 1.0)
            <= plan.prefix.plan.policy.trace_tolerance
        ),
        plan_id=plan.plan_id,
        claim="bounded-exact-branch-ensemble-classical-control",
    )


__all__ = [
    "MidCircuitQuantumPlan",
    "MidCircuitQuantumResult",
    "QuantumMeasurementPlan",
    "QuantumMeasurementResult",
    "execute_mid_circuit_quantum_plan",
    "measure_dense_quantum_program",
]
