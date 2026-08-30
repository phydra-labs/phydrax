#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array, ArrayLike, Key

from .._sampling import sample_markov
from .._strict import StrictModule
from ..linalg import (
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    NullspacePolicy,
    solve,
)
from ..operators.quantum import (
    AbstractDiscreteQuantumOperator,
    evaluate_local_operator,
)
from ._variational_monte_carlo import (
    _model_log_target,
    _score_geometry,
    _validate_model_coordinates,
    _validate_state_compatibility,
    VariationalMonteCarloProblem,
    VariationalMonteCarloState,
)
from ._variational_tdvp import _tdvp_force


class NeuralRateEvidence(StrictModule):
    rates: Array
    standard_errors: Array
    effective_sample_size: Array
    valid: Array

    def __init__(
        self,
        rates: ArrayLike,
        standard_errors: ArrayLike,
        effective_sample_size: ArrayLike,
        /,
        *,
        relative_error_tolerance: float,
    ):
        rates_ = jnp.asarray(rates)
        errors = jnp.asarray(standard_errors)
        if rates_.shape != errors.shape:
            raise ValueError("Neural rate values and errors must share shape.")
        scale = jnp.maximum(rates_, 1e-12)
        ess = jnp.asarray(effective_sample_size)
        if ess.shape not in ((), rates_.shape):
            raise ValueError(
                "Neural effective sample size must be scalar or match the rate shape."
            )
        self.rates = rates_
        self.standard_errors = errors
        self.effective_sample_size = ess
        self.valid = (
            jnp.all(jnp.isfinite(rates_) & (rates_ >= 0.0))
            & jnp.all(jnp.isfinite(errors) & (errors >= 0.0))
            & jnp.all(errors / scale <= relative_error_tolerance)
            & jnp.all(jnp.isfinite(ess) & (ess > 1.0))
        )


class ConnectedVMCNeuralTrajectoryProblem(StrictModule):
    """Connected-operator neural trajectory over a persistent VMC state."""

    vmc_problem: VariationalMonteCarloProblem
    collapse_operators: tuple[AbstractDiscreteQuantumOperator, ...]
    jump_projection: Callable[[int, Any, Array], Array]
    projection_residual: Callable[[int, Any, Any], Array]
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        vmc_problem: VariationalMonteCarloProblem,
        collapse_operators: Sequence[AbstractDiscreteQuantumOperator],
        jump_projection: Callable[[int, Any, Array], Array],
        projection_residual: Callable[[int, Any, Any], Array],
        /,
        *,
        problem_id: str | None = None,
    ):
        if not isinstance(vmc_problem, VariationalMonteCarloProblem):
            raise TypeError("vmc_problem must be a VariationalMonteCarloProblem.")
        operators = tuple(collapse_operators)
        if not operators or any(
            not isinstance(operator, AbstractDiscreteQuantumOperator)
            for operator in operators
        ):
            raise TypeError(
                "collapse_operators must contain connected discrete operators."
            )
        shape = vmc_problem.operator.configuration_shape
        if any(operator.configuration_shape != shape for operator in operators):
            raise ValueError(
                "The no-jump and collapse operators must share configuration shape."
            )
        if not callable(jump_projection) or not callable(projection_residual):
            raise TypeError("jump_projection and projection_residual must be callable.")
        identifier = (
            f"{vmc_problem.problem_id}:connected-neural-trajectory"
            if problem_id is None
            else str(problem_id)
        )
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        self.vmc_problem = vmc_problem
        self.collapse_operators = operators
        self.jump_projection = jump_projection
        self.projection_residual = projection_residual
        self.problem_id = identifier


class ConnectedVMCJumpProjectionAudit(StrictModule):
    channel: int
    projected_coordinates: Array
    residual: Array
    valid: Array

    def __init__(
        self,
        channel: int,
        projected_coordinates: ArrayLike,
        residual: ArrayLike,
        /,
        *,
        tolerance: float,
    ):
        self.channel = int(channel)
        self.projected_coordinates = jnp.asarray(projected_coordinates)
        self.residual = jnp.asarray(residual).reshape(())
        self.valid = (
            jnp.all(jnp.isfinite(self.projected_coordinates))
            & jnp.isfinite(self.residual)
            & (self.residual >= 0.0)
            & (self.residual <= float(tolerance))
        )


def audit_connected_vmc_jump_projection(
    problem: ConnectedVMCNeuralTrajectoryProblem,
    channel: int,
    /,
    *,
    coordinates: ArrayLike | None = None,
    tolerance: float = 1e-3,
) -> ConnectedVMCJumpProjectionAudit:
    """Apply one selected jump and independently evaluate its closure residual."""
    channel_ = int(channel)
    if not 0 <= channel_ < len(problem.collapse_operators):
        raise ValueError("Projection-audit channel is outside the problem.")
    source_coordinates = (
        problem.vmc_problem.initial_coordinates
        if coordinates is None
        else jnp.asarray(coordinates)
    )
    source_model = problem.vmc_problem.model_from_coordinates(source_coordinates)
    projected_coordinates = jnp.asarray(
        problem.jump_projection(channel_, source_model, source_coordinates)
    )
    if projected_coordinates.shape != source_coordinates.shape:
        raise ValueError("Jump projection must preserve coordinate shape.")
    projected_model = problem.vmc_problem.model_from_coordinates(projected_coordinates)
    residual = problem.projection_residual(channel_, source_model, projected_model)
    return ConnectedVMCJumpProjectionAudit(
        channel_,
        projected_coordinates,
        residual,
        tolerance=tolerance,
    )


class ConnectedVMCNeuralTrajectoryPolicy(StrictModule):
    """Fixed-step sampling, TDVP, rate, and jump-projection release policy."""

    step_size: float = eqx.field(static=True)
    steps: int = eqx.field(static=True)
    draws_per_step: int = eqx.field(static=True)
    transitions_per_draw: int = eqx.field(static=True)
    warmup_steps: int = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    rate_relative_error_tolerance: float = eqx.field(static=True)
    maximum_jump_probability: float = eqx.field(static=True)
    projection_residual_tolerance: float = eqx.field(static=True)
    maximum_velocity_norm: float | None = eqx.field(static=True)
    require_projected_jump: bool = eqx.field(static=True)
    linear_policy: LinearSolvePolicy | None
    nullspace_policy: NullspacePolicy | None

    def __init__(
        self,
        *,
        step_size: float,
        steps: int,
        draws_per_step: int,
        transitions_per_draw: int = 1,
        warmup_steps: int = 0,
        damping: float = 1e-3,
        rate_relative_error_tolerance: float = 0.25,
        maximum_jump_probability: float = 0.1,
        projection_residual_tolerance: float = 1e-3,
        maximum_velocity_norm: float | None = None,
        require_projected_jump: bool = False,
        linear_policy: LinearSolvePolicy | None = None,
        nullspace_policy: NullspacePolicy | None = None,
    ):
        size = float(step_size)
        step_count = int(steps)
        draws = int(draws_per_step)
        transitions = int(transitions_per_draw)
        warmup = int(warmup_steps)
        damping_ = float(damping)
        rate_tolerance = float(rate_relative_error_tolerance)
        jump_limit = float(maximum_jump_probability)
        projection_tolerance = float(projection_residual_tolerance)
        if not isfinite(size) or size <= 0.0:
            raise ValueError("step_size must be finite and positive.")
        if step_count <= 0:
            raise ValueError("steps must be positive.")
        if draws < 4:
            raise ValueError("draws_per_step must be at least four for rate ESS.")
        if transitions <= 0 or warmup < 0:
            raise ValueError(
                "transitions_per_draw must be positive and warmup_steps non-negative."
            )
        if not isfinite(damping_) or damping_ < 0.0:
            raise ValueError("damping must be finite and non-negative.")
        if not isfinite(rate_tolerance) or rate_tolerance <= 0.0:
            raise ValueError("rate_relative_error_tolerance must be finite and positive.")
        if not isfinite(jump_limit) or not 0.0 < jump_limit <= 0.1:
            raise ValueError(
                "maximum_jump_probability must be finite and lie in (0, 0.1]."
            )
        if not isfinite(projection_tolerance) or projection_tolerance < 0.0:
            raise ValueError(
                "projection_residual_tolerance must be finite and non-negative."
            )
        if maximum_velocity_norm is None:
            velocity_limit = None
        else:
            velocity_limit = float(maximum_velocity_norm)
            if not isfinite(velocity_limit) or velocity_limit <= 0.0:
                raise ValueError("maximum_velocity_norm must be finite and positive.")
        if linear_policy is not None and not isinstance(linear_policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if nullspace_policy is not None and not isinstance(
            nullspace_policy, NullspacePolicy
        ):
            raise TypeError("nullspace_policy must be a NullspacePolicy or None.")
        self.step_size = size
        self.steps = step_count
        self.draws_per_step = draws
        self.transitions_per_draw = transitions
        self.warmup_steps = warmup
        self.damping = damping_
        self.rate_relative_error_tolerance = rate_tolerance
        self.maximum_jump_probability = jump_limit
        self.projection_residual_tolerance = projection_tolerance
        self.maximum_velocity_norm = velocity_limit
        self.require_projected_jump = bool(require_projected_jump)
        self.linear_policy = linear_policy
        self.nullspace_policy = nullspace_policy


class ConnectedVMCNeuralTrajectoryResult(StrictModule):
    """Persistent VMC trajectory plus rate, projection, and replay evidence."""

    final_state: VariationalMonteCarloState
    parameter_history: Array
    rate_history: Array
    rate_standard_error_history: Array
    effective_sample_size_history: Array
    jump_probability_history: Array
    jump_history: Array
    channel_history: Array
    decision_uniform_history: Array
    channel_uniform_history: Array
    projection_residual_history: Array
    rate_evidence_valid_history: Array
    status_history: Array
    linear_results: tuple[LinearSolveResult, ...]
    projected_jump_observed: Array
    valid: Array
    completed_steps: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        final_state: VariationalMonteCarloState,
        parameter_history: ArrayLike,
        rate_history: ArrayLike,
        rate_standard_error_history: ArrayLike,
        effective_sample_size_history: ArrayLike,
        jump_probability_history: ArrayLike,
        jump_history: ArrayLike,
        channel_history: ArrayLike,
        decision_uniform_history: ArrayLike,
        channel_uniform_history: ArrayLike,
        projection_residual_history: ArrayLike,
        rate_evidence_valid_history: ArrayLike,
        status_history: ArrayLike,
        linear_results: Sequence[LinearSolveResult],
        /,
        *,
        completed_steps: int,
        planned_steps: int,
        require_projected_jump: bool,
        problem_id: str,
    ):
        self.final_state = final_state
        self.parameter_history = jnp.asarray(parameter_history)
        self.rate_history = jnp.asarray(rate_history)
        self.rate_standard_error_history = jnp.asarray(rate_standard_error_history)
        self.effective_sample_size_history = jnp.asarray(effective_sample_size_history)
        self.jump_probability_history = jnp.asarray(jump_probability_history)
        self.jump_history = jnp.asarray(jump_history, dtype=bool)
        self.channel_history = jnp.asarray(channel_history, dtype=jnp.int32)
        self.decision_uniform_history = jnp.asarray(decision_uniform_history)
        self.channel_uniform_history = jnp.asarray(channel_uniform_history)
        self.projection_residual_history = jnp.asarray(projection_residual_history)
        self.rate_evidence_valid_history = jnp.asarray(
            rate_evidence_valid_history, dtype=bool
        )
        self.status_history = jnp.asarray(status_history, dtype=bool)
        self.linear_results = tuple(linear_results)
        self.completed_steps = int(completed_steps)
        self.problem_id = str(problem_id)
        self.projected_jump_observed = jnp.any(
            self.jump_history & (self.channel_history >= 0)
        )
        self.valid = (
            (self.completed_steps == int(planned_steps))
            & jnp.all(self.status_history)
            & jnp.all(self.rate_evidence_valid_history)
            & jnp.all(jnp.isfinite(self.parameter_history))
            & jnp.all(jnp.isfinite(self.rate_history))
            & jnp.all(jnp.isfinite(self.rate_standard_error_history))
            & jnp.all(jnp.isfinite(self.effective_sample_size_history))
            & jnp.all(jnp.isfinite(self.projection_residual_history))
        )
        if require_projected_jump:
            self.valid = self.valid & self.projected_jump_observed


def _connected_rate_statistics(
    problem: ConnectedVMCNeuralTrajectoryProblem,
    model: Any,
    samples,
    /,
) -> tuple[Array, Array, Array, Array]:
    local_rates = []
    local_valid = []
    for operator in problem.collapse_operators:
        estimate = evaluate_local_operator(model, operator, samples.samples)
        local_rates.append(jnp.abs(estimate.value) ** 2)
        local_valid.append(estimate.successful)
    values = jnp.stack(local_rates, axis=-1)
    valid = jnp.all(jnp.stack(local_valid, axis=-1))
    from ..uq._diagnostics import mcmc_diagnostics

    diagnostics = mcmc_diagnostics(
        {"jump-rates": values},
        acceptance_rate=samples.acceptance_rate,
        divergent=jnp.zeros(samples.log_target.shape, dtype=bool),
    )
    rates = jnp.mean(values, axis=(0, 1))
    variances = jnp.var(values, axis=(0, 1), ddof=1)
    nominal = jnp.asarray(values.shape[0] * values.shape[1], dtype=float)
    measured_ess = diagnostics.bulk_ess["jump-rates"]
    effective_sample_size = jnp.where(variances == 0.0, nominal, measured_ess)
    standard_errors = jnp.sqrt(
        jnp.maximum(variances, 0.0) / jnp.maximum(effective_sample_size, 1.0)
    )
    valid = (
        valid
        & jnp.all(jnp.isfinite(values))
        & jnp.all(jnp.isfinite(effective_sample_size))
    )
    return rates, standard_errors, effective_sample_size, valid


def solve_connected_vmc_neural_trajectory(
    problem: ConnectedVMCNeuralTrajectoryProblem,
    policy: ConnectedVMCNeuralTrajectoryPolicy,
    key: Key[Array, ""],
    /,
    *,
    state: VariationalMonteCarloState | None = None,
) -> ConnectedVMCNeuralTrajectoryResult:
    """Evolve one sampled neural trajectory from connected VMC operators."""
    if not isinstance(problem, ConnectedVMCNeuralTrajectoryProblem):
        raise TypeError("problem must be a ConnectedVMCNeuralTrajectoryProblem.")
    if not isinstance(policy, ConnectedVMCNeuralTrajectoryPolicy):
        raise TypeError("policy must be a ConnectedVMCNeuralTrajectoryPolicy.")
    if problem.vmc_problem.initial_configurations.shape[0] < 2:
        raise ValueError("Connected VMC rate evidence requires at least two chains.")
    current = problem.vmc_problem.initial_state(key=key) if state is None else state
    if not isinstance(current, VariationalMonteCarloState):
        raise TypeError("state must be a VariationalMonteCarloState or None.")
    if not jnp.array_equal(jr.key_data(key), jr.key_data(current.root_key)):
        raise ValueError("Resume key does not match the VMC state root key.")
    _validate_state_compatibility(problem.vmc_problem, current)
    _validate_model_coordinates(problem.vmc_problem, current)

    parameter_history = [current.parameter_coordinates]
    rate_history = []
    rate_error_history = []
    ess_history = []
    jump_probability_history = []
    jump_history = []
    channel_history = []
    decision_history = []
    channel_uniform_history = []
    projection_history = []
    rate_valid_history = []
    status_history = []
    linear_results = []
    completed = 0

    for _ in range(policy.steps):
        iteration = int(current.iteration)
        step_key = jr.fold_in(key, iteration)
        refreshed = problem.vmc_problem.kernel.refresh(
            _model_log_target(current.model), current.markov_state
        )
        samples = sample_markov(
            _model_log_target(current.model),
            problem.vmc_problem.kernel,
            refreshed,
            key=jr.fold_in(step_key, 0x51A7),
            num_draws=policy.draws_per_step,
            steps_per_draw=policy.transitions_per_draw,
            warmup_steps=policy.warmup_steps if iteration == 0 else 0,
        )
        rates, errors, ess, local_rates_valid = _connected_rate_statistics(
            problem, current.model, samples
        )
        evidence = NeuralRateEvidence(
            rates,
            errors,
            ess,
            relative_error_tolerance=policy.rate_relative_error_tolerance,
        )
        rate_valid = evidence.valid & local_rates_valid
        total_rate = jnp.sum(rates)
        jump_probability = policy.step_size * total_rate
        decision = jr.uniform(jr.fold_in(step_key, 0xDEC1))
        channel_uniform = jr.uniform(jr.fold_in(step_key, 0xC4A7))
        rate_history.append(rates)
        rate_error_history.append(errors)
        ess_history.append(ess)
        jump_probability_history.append(jump_probability)
        decision_history.append(decision)
        channel_uniform_history.append(channel_uniform)
        rate_valid_history.append(rate_valid)
        probability_valid = (
            jnp.isfinite(jump_probability)
            & (jump_probability >= 0.0)
            & (jump_probability <= policy.maximum_jump_probability)
        )
        if not bool(rate_valid & probability_valid):
            jump_history.append(jnp.asarray(False))
            channel_history.append(jnp.asarray(-1, dtype=jnp.int32))
            projection_history.append(jnp.asarray(jnp.nan))
            status_history.append(jnp.asarray(False))
            break

        jumped = bool(decision < jump_probability)
        channel = -1
        projection_residual = jnp.asarray(0.0)
        coordinates = current.parameter_coordinates
        model = current.model
        step_valid = jnp.asarray(True)
        if jumped:
            cumulative = jnp.cumsum(rates / total_rate)
            channel = min(
                int(jnp.searchsorted(cumulative, channel_uniform, side="right")),
                len(problem.collapse_operators) - 1,
            )
            source_model = current.model
            coordinates = jnp.asarray(
                problem.jump_projection(
                    channel, source_model, current.parameter_coordinates
                )
            )
            if coordinates.shape != current.parameter_coordinates.shape:
                raise ValueError(
                    "jump_projection must preserve the parameter coordinate shape."
                )
            model = problem.vmc_problem.model_from_coordinates(coordinates)
            projection_residual = jnp.asarray(
                problem.projection_residual(channel, source_model, model)
            ).reshape(())
            step_valid = (
                jnp.all(jnp.isfinite(coordinates))
                & jnp.isfinite(projection_residual)
                & (projection_residual >= 0.0)
                & (projection_residual <= policy.projection_residual_tolerance)
            )
        else:
            local_generator = evaluate_local_operator(
                current.model,
                problem.vmc_problem.operator,
                samples.samples,
            )
            generator_valid = jnp.all(local_generator.successful) & jnp.all(
                jnp.isfinite(local_generator.value)
            )
            score, metric = _score_geometry(
                problem.vmc_problem,
                current.parameter_coordinates,
                samples.samples,
                damping=policy.damping,
            )
            mean_generator = jnp.mean(local_generator.value)
            force = _tdvp_force(
                score,
                local_generator.value,
                mean_generator,
                problem.vmc_problem.complex_parameter_mode,
                "real-time",
            )
            linear = solve(
                LinearSystem(metric, nullspace_policy=policy.nullspace_policy),
                force,
                policy=policy.linear_policy,
            )
            linear_results.append(linear)
            velocity = jnp.asarray(linear.value)
            if policy.maximum_velocity_norm is not None:
                norm = jnp.linalg.norm(velocity)
                scale = jnp.minimum(
                    1.0,
                    policy.maximum_velocity_norm / jnp.maximum(norm, 1e-30),
                )
                velocity = scale * velocity
            coordinates = current.parameter_coordinates + policy.step_size * velocity
            model = problem.vmc_problem.model_from_coordinates(coordinates)
            step_valid = (
                generator_valid
                & jnp.all(linear.successful)
                & jnp.all(jnp.isfinite(velocity))
                & jnp.all(jnp.isfinite(coordinates))
            )

        jump_history.append(jnp.asarray(jumped))
        channel_history.append(jnp.asarray(channel, dtype=jnp.int32))
        projection_history.append(projection_residual)
        status_history.append(step_valid)
        if not bool(step_valid):
            break
        current = VariationalMonteCarloState(
            model=model,
            parameter_coordinates=coordinates,
            markov_state=samples.final_state,
            iteration=iteration + 1,
            root_key=key,
        )
        parameter_history.append(coordinates)
        completed += 1

    channels = len(problem.collapse_operators)
    return ConnectedVMCNeuralTrajectoryResult(
        current,
        jnp.stack(parameter_history),
        jnp.stack(rate_history) if rate_history else jnp.empty((0, channels)),
        jnp.stack(rate_error_history) if rate_error_history else jnp.empty((0, channels)),
        jnp.stack(ess_history) if ess_history else jnp.empty((0, channels)),
        jnp.stack(jump_probability_history)
        if jump_probability_history
        else jnp.empty((0,)),
        jnp.stack(jump_history) if jump_history else jnp.empty((0,), dtype=bool),
        jnp.stack(channel_history)
        if channel_history
        else jnp.empty((0,), dtype=jnp.int32),
        jnp.stack(decision_history) if decision_history else jnp.empty((0,)),
        jnp.stack(channel_uniform_history)
        if channel_uniform_history
        else jnp.empty((0,)),
        jnp.stack(projection_history) if projection_history else jnp.empty((0,)),
        jnp.stack(rate_valid_history)
        if rate_valid_history
        else jnp.empty((0,), dtype=bool),
        jnp.stack(status_history) if status_history else jnp.empty((0,), dtype=bool),
        tuple(linear_results),
        completed_steps=completed,
        planned_steps=policy.steps,
        require_projected_jump=policy.require_projected_jump,
        problem_id=problem.problem_id,
    )


__all__ = [
    "ConnectedVMCJumpProjectionAudit",
    "ConnectedVMCNeuralTrajectoryPolicy",
    "ConnectedVMCNeuralTrajectoryProblem",
    "ConnectedVMCNeuralTrajectoryResult",
    "NeuralRateEvidence",
    "audit_connected_vmc_jump_projection",
    "solve_connected_vmc_neural_trajectory",
]
