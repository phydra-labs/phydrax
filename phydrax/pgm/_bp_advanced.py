#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from ..linalg import GMRES, LinearSolvePolicy, TolerancePolicy
from ..nonlinear import (
    AndersonAcceleration,
    FixedPointIteration,
    FixedPointProblem,
    implicit_root_result,
    ImplicitRootDerivativePolicy,
    NewtonKrylov,
    NonlinearResult,
    NonlinearTermination,
)
from ._belief_propagation import (
    _bethe_log_normalizer,
    _bp_step,
    _factor_probabilities,
    _graph_numeric_tables,
    _variable_log_beliefs,
    BeliefPropagationSchedulePolicy,
    BeliefPropagationState,
    PreparedBeliefPropagation,
    SumProductBeliefPropagation,
    SumProductBeliefPropagationResult,
)
from ._model import VariableStateValues
from ._types import (
    BeliefPropagationDiagnostics,
    BeliefPropagationStatus,
    FactorGraphProvenance,
)


class AdvancedBeliefPropagationResult(StrictModule):
    """Sum-product beliefs with nonlinear solver and precision evidence."""

    inference: SumProductBeliefPropagationResult
    nonlinear: NonlinearResult
    precision_evidence: PrecisionEvidenceEnvelope
    schedule: BeliefPropagationSchedulePolicy
    implicit_derivative: bool = eqx.field(static=True)


def _sum_product_result(
    prepared: PreparedBeliefPropagation,
    state: BeliefPropagationState,
    nonlinear: NonlinearResult,
    /,
    *,
    method_id: str,
) -> SumProductBeliefPropagationResult:
    variable_values = _variable_log_beliefs(prepared, state)
    factor_probabilities = _factor_probabilities(prepared, state)
    log_normalizer = _bethe_log_normalizer(
        prepared,
        state,
        variable_values,
        factor_probabilities,
    )
    successful = nonlinear.successful
    status = jnp.where(
        successful,
        int(BeliefPropagationStatus.SUCCESS),
        int(BeliefPropagationStatus.MAXIMUM_STEPS_REACHED),
    ).astype(jnp.int32)
    diagnostics = BeliefPropagationDiagnostics(
        initial_residual=nonlinear.diagnostics.initial_residual_norm,
        final_residual=nonlinear.diagnostics.final_residual_norm,
        iterations=nonlinear.diagnostics.iterations,
        support_changes=jnp.asarray(0, dtype=jnp.int32),
        factor_evaluations=nonlinear.diagnostics.residual_evaluations
        * prepared.graph.num_factors,
    )
    return SumProductBeliefPropagationResult(
        variable_log_probabilities=VariableStateValues(
            prepared.precision.output(variable_values),
            structure_id=prepared.graph.structure_id,
        ),
        factor_probabilities=tuple(
            prepared.precision.output(probabilities)
            for probabilities in factor_probabilities
        ),
        factor_tables=_graph_numeric_tables(prepared.graph),
        log_normalizer=prepared.precision.output(log_normalizer),
        state=state,
        status=status,
        valid=successful,
        converged=successful,
        diagnostics=diagnostics,
        provenance=FactorGraphProvenance(
            structure_id=prepared.graph.structure_id,
            plan_id=prepared.plan_id,
            method_id=method_id,
            implementation="nonlinear-fixed-point",
            exact=prepared.forest,
            configuration=(
                ("evaluation_dtype", prepared.precision.evaluation_dtype),
                ("accumulation_dtype", prepared.precision.accumulation_dtype),
                ("decision_dtype", prepared.precision.decision_dtype),
                ("output_dtype", prepared.precision.output_dtype),
            ),
        ),
        marginals_exact=prepared.forest,
        log_normalizer_exact=prepared.forest,
        log_normalizer_kind="exact" if prepared.forest else "bethe",
    )


def _mapping(
    prepared: PreparedBeliefPropagation, evidence: Array, messages: Array
) -> Array:
    return _bp_step(prepared, messages, evidence, force_full=True)[0]


def _probability_mapping(
    prepared: PreparedBeliefPropagation,
    evidence: Array,
    probabilities: Array,
    /,
) -> Array:
    tiny = jnp.finfo(probabilities.dtype).tiny
    messages = jnp.log(jnp.maximum(probabilities, tiny))
    return jnp.exp(_mapping(prepared, evidence, messages))


def _termination(
    prepared: PreparedBeliefPropagation,
    value: NonlinearTermination | None,
    /,
) -> NonlinearTermination:
    if value is not None:
        if not isinstance(value, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        return value
    return NonlinearTermination(
        absolute_residual=prepared.method.absolute_tolerance,
        relative_residual=prepared.method.relative_tolerance,
        maximum_steps=prepared.method.maximum_steps,
    )


def _validate_state(
    prepared: PreparedBeliefPropagation,
    state: BeliefPropagationState,
    /,
) -> None:
    if not isinstance(state, BeliefPropagationState):
        raise TypeError("state must be BeliefPropagationState.")
    if state.messages.shape != (prepared.message_count,):
        raise ValueError("State message shape does not match the prepared plan.")
    if state.evidence.structure_id != prepared.graph.structure_id:
        raise ValueError("State evidence does not match the prepared graph.")
    expected_evidence_shape = (prepared.state_variable_indices.shape[0],)
    if state.evidence.values.shape != expected_evidence_shape:
        raise ValueError(
            f"State evidence must have shape {expected_evidence_shape}; "
            f"got {state.evidence.values.shape}."
        )


def _invalid_log_values(value: Array, /) -> Array:
    return jnp.any(jnp.isnan(value) | jnp.isposinf(value))


def _infeasible_support(
    prepared: PreparedBeliefPropagation,
    evidence: Array,
    messages: Array,
    /,
) -> Array:
    infeasible = jnp.asarray(False)
    offsets = np.asarray(prepared.graph.variable_state_offsets)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        infeasible = infeasible | jnp.all(jnp.isneginf(evidence[int(start) : int(stop)]))
    for table in prepared.factor_tables:
        axes = tuple(range(1, table.ndim))
        infeasible = infeasible | jnp.any(jnp.all(jnp.isneginf(table), axis=axes))
    for layout in prepared.message_layout:
        for start, stop, count, cardinality in layout:
            rows = messages[start:stop].reshape((count, cardinality))
            infeasible = infeasible | jnp.any(jnp.all(jnp.isneginf(rows), axis=-1))
    return infeasible


def run_accelerated_belief_propagation(
    prepared: PreparedBeliefPropagation,
    state: BeliefPropagationState,
    /,
    *,
    acceleration: AndersonAcceleration | None = None,
    termination: NonlinearTermination | None = None,
    schedule: BeliefPropagationSchedulePolicy | None = None,
) -> AdvancedBeliefPropagationResult:
    """Solve the normalized sum-product fixed point with safeguarded Anderson acceleration."""
    if not isinstance(prepared.method, SumProductBeliefPropagation):
        raise TypeError("Accelerated belief propagation requires sum-product.")
    _validate_state(prepared, state)
    selected_schedule = (
        BeliefPropagationSchedulePolicy() if schedule is None else schedule
    )
    if not isinstance(selected_schedule, BeliefPropagationSchedulePolicy):
        raise TypeError("schedule must be BeliefPropagationSchedulePolicy or None.")
    if selected_schedule.kind != "synchronous":
        raise ValueError("Nonlinear acceleration supports only synchronous BP.")
    policy = prepared.precision
    initial = policy.accumulation(state.messages)
    problem = FixedPointProblem(
        lambda messages, evidence: _mapping(prepared, evidence, messages),
        problem_id=f"bp:{prepared.plan_id}",
    )
    method = FixedPointIteration(
        damping=prepared.method.relaxation,
        acceleration=acceleration,
    )
    nonlinear = method.solve(
        problem,
        initial,
        termination=_termination(prepared, termination),
        args=policy.evaluation(state.evidence.values),
    )
    final_state = BeliefPropagationState(
        policy.accumulation(nonlinear.state),
        state.evidence,
        step_index=state.step_index + nonlinear.diagnostics.iterations,
    )
    inference = _sum_product_result(
        prepared,
        final_state,
        nonlinear,
        method_id=f"sum-product-{selected_schedule.kind}-anderson",
    )
    return AdvancedBeliefPropagationResult(
        inference=inference,
        nonlinear=nonlinear,
        precision_evidence=policy.evidence(),
        schedule=selected_schedule,
        implicit_derivative=False,
    )


def run_implicit_belief_propagation(
    prepared: PreparedBeliefPropagation,
    state: BeliefPropagationState,
    /,
    *,
    termination: NonlinearTermination | None = None,
    derivative_policy: ImplicitRootDerivativePolicy | None = None,
) -> AdvancedBeliefPropagationResult:
    """Return a fixed-support root; nonfinite runtime inputs yield failed evidence.

    Preparation owns topology and support validation. Numeric factors and evidence
    may vary under JIT/autodiff; implicit derivative failures remain certified by
    the native root solver and never fall back to stale messages.
    """
    if not isinstance(prepared.method, SumProductBeliefPropagation):
        raise TypeError("Implicit belief propagation requires sum-product.")
    _validate_state(prepared, state)
    if any(evidence.capabilities.sparse_support for evidence in prepared.factor_evidence):
        raise ValueError("Implicit BP does not support sparse structural support.")
    policy = prepared.precision
    initial = jnp.exp(policy.accumulation(state.messages))
    evidence = policy.evaluation(state.evidence.values)
    problem = FixedPointProblem(
        lambda probabilities, args: _probability_mapping(
            prepared,
            args,
            probabilities,
        ),
        problem_id=f"implicit-bp:{prepared.plan_id}",
    ).as_nonlinear_problem()
    # Inexact Newton already scales each correction by its residual via forcing.
    # A fixed absolute linear floor can accept a zero correction before the
    # requested nonlinear residual tolerance is met.
    method = NewtonKrylov(
        linear_policy=LinearSolvePolicy(
            GMRES(restart=16),
            tolerance=TolerancePolicy(relative=1e-6, absolute=0.0, max_steps=64),
        )
    )
    nonlinear = implicit_root_result(
        problem,
        initial,
        method=method,
        termination=_termination(prepared, termination),
        derivative_policy=derivative_policy,
        args=evidence,
    )
    final_probabilities = _probability_mapping(
        prepared,
        evidence,
        nonlinear.state,
    )
    final_messages = jnp.where(
        final_probabilities > 0.0,
        jnp.log(final_probabilities),
        -jnp.inf,
    )
    final_state = BeliefPropagationState(
        policy.accumulation(final_messages),
        state.evidence,
        step_index=state.step_index + nonlinear.diagnostics.iterations,
    )
    inference = _sum_product_result(
        prepared,
        final_state,
        nonlinear,
        method_id="sum-product-implicit-root",
    )
    invalid_input = _invalid_log_values(state.evidence.values) | _invalid_log_values(
        state.messages
    )
    for table in prepared.factor_tables:
        invalid_input = invalid_input | _invalid_log_values(table)
    invalid_output = _invalid_log_values(inference.log_normalizer) | _invalid_log_values(
        final_state.messages
    )
    infeasible = (
        _infeasible_support(prepared, state.evidence.values, state.messages)
        | _infeasible_support(
            prepared,
            state.evidence.values,
            final_state.messages,
        )
        | jnp.isneginf(inference.log_normalizer)
    )
    accepted = inference.successful & ~invalid_input & ~invalid_output & ~infeasible
    status = jnp.where(
        invalid_input,
        int(BeliefPropagationStatus.NONFINITE_INPUT),
        jnp.where(
            infeasible,
            int(BeliefPropagationStatus.INFEASIBLE),
            jnp.where(
                invalid_output,
                int(BeliefPropagationStatus.NONFINITE_MESSAGE),
                inference.status,
            ),
        ),
    ).astype(jnp.int32)
    inference = eqx.tree_at(
        lambda value: (value.valid, value.converged, value.status),
        inference,
        (accepted, accepted, status),
    )
    return AdvancedBeliefPropagationResult(
        inference=inference,
        nonlinear=nonlinear,
        precision_evidence=policy.evidence(),
        schedule=BeliefPropagationSchedulePolicy("synchronous"),
        implicit_derivative=True,
    )


__all__ = [
    "AdvancedBeliefPropagationResult",
    "BeliefPropagationSchedulePolicy",
    "run_accelerated_belief_propagation",
    "run_implicit_belief_propagation",
]
