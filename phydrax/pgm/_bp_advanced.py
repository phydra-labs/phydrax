#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
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
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._belief_propagation import (
    _bethe_log_normalizer,
    _bp_step,
    _factor_probabilities,
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
    initial = policy.accumulation(state.messages)
    evidence = policy.evaluation(state.evidence.values)
    problem = NonlinearSystemProblem(
        lambda messages, args: messages - _mapping(prepared, args, messages),
        problem_id=f"implicit-bp:{prepared.plan_id}",
    )
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
    final_state = BeliefPropagationState(
        policy.accumulation(nonlinear.state),
        state.evidence,
        step_index=state.step_index + nonlinear.diagnostics.iterations,
    )
    inference = _sum_product_result(
        prepared,
        final_state,
        nonlinear,
        method_id="sum-product-implicit-root",
    )
    finite_input = jnp.all(jnp.isfinite(state.evidence.values)) & jnp.all(
        jnp.isfinite(state.messages)
    )
    for table in prepared.factor_tables:
        finite_input = finite_input & jnp.all(jnp.isfinite(table))
    finite_output = jnp.isfinite(inference.log_normalizer) & jnp.all(
        jnp.isfinite(final_state.messages)
    )
    accepted = inference.successful & finite_input & finite_output
    status = jnp.where(
        ~finite_input,
        int(BeliefPropagationStatus.NONFINITE_INPUT),
        jnp.where(
            ~finite_output,
            int(BeliefPropagationStatus.NONFINITE_MESSAGE),
            inference.status,
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
