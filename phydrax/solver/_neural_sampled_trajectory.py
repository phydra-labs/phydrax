#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._strict import StrictModule
from ..metrix import InformationMetricOperator


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
        self.rates = rates_
        self.standard_errors = errors
        self.effective_sample_size = jnp.asarray(effective_sample_size)
        self.valid = (
            jnp.all(jnp.isfinite(rates_) & (rates_ >= 0.0))
            & jnp.all(jnp.isfinite(errors) & (errors >= 0.0))
            & jnp.all(errors / scale <= relative_error_tolerance)
            & (self.effective_sample_size > 1.0)
        )


class SampledNeuralTrajectoryProblem(StrictModule):
    parameters: Array
    qgt_action: Callable[[Array, Array], Array]
    force: Callable[[Array], Array]
    rate_estimator: Callable[[Array], tuple[Array, Array, Array]]
    jump_projection: Callable[[int, Array], tuple[Array, Array]]
    problem_id: str

    def __init__(
        self,
        parameters: ArrayLike,
        qgt_action: Callable[[Array, Array], Array],
        force: Callable[[Array], Array],
        rate_estimator: Callable[[Array], tuple[Array, Array, Array]],
        jump_projection: Callable[[int, Array], tuple[Array, Array]],
        /,
        *,
        problem_id: str = "sampled-neural-trajectory",
    ):
        self.parameters = jnp.asarray(parameters)
        self.qgt_action = qgt_action
        self.force = force
        self.rate_estimator = rate_estimator
        self.jump_projection = jump_projection
        self.problem_id = str(problem_id)


class SampledNeuralTrajectoryResult(StrictModule):
    parameters: Array
    parameter_history: Array
    rate_standard_error_history: Array
    effective_sample_size_history: Array
    projection_residual_history: Array
    evidence_valid: Array
    valid: Array

    def __init__(
        self,
        parameters: ArrayLike,
        parameter_history: ArrayLike,
        rate_standard_error_history: ArrayLike,
        effective_sample_size_history: ArrayLike,
        projection_residual_history: ArrayLike,
        evidence_valid: ArrayLike,
        /,
    ):
        self.parameters = jnp.asarray(parameters)
        self.parameter_history = jnp.asarray(parameter_history)
        self.rate_standard_error_history = jnp.asarray(rate_standard_error_history)
        self.effective_sample_size_history = jnp.asarray(effective_sample_size_history)
        self.projection_residual_history = jnp.asarray(projection_residual_history)
        self.evidence_valid = jnp.asarray(evidence_valid, dtype=bool)
        self.valid = (
            jnp.all(jnp.isfinite(self.parameter_history))
            & jnp.all(self.evidence_valid)
            & jnp.all(jnp.isfinite(self.projection_residual_history))
        )


def solve_sampled_neural_trajectory(
    problem: SampledNeuralTrajectoryProblem,
    /,
    *,
    step_size: float,
    steps: int,
    damping: float = 1e-6,
    rate_relative_error_tolerance: float = 0.25,
) -> SampledNeuralTrajectoryResult:
    parameters = problem.parameters
    history = [parameters]
    rate_errors = []
    ess_history = []
    projection = []
    valid_history = []
    for _ in range(int(steps)):
        rates, standard_errors, ess = problem.rate_estimator(parameters)
        evidence = NeuralRateEvidence(
            rates,
            standard_errors,
            ess,
            relative_error_tolerance=rate_relative_error_tolerance,
        )
        valid_history.append(evidence.valid)
        rate_errors.append(evidence.standard_errors)
        ess_history.append(evidence.effective_sample_size)
        if not bool(evidence.valid):
            break
        metric = InformationMetricOperator(
            lambda vector: problem.qgt_action(parameters, vector),
            parameters,
            damping=damping,
            metric_id=f"{problem.problem_id}:qgt",
        )
        velocity = metric.solve(problem.force(parameters)).value
        parameters = parameters + float(step_size) * velocity
        projection.append(jnp.asarray(0.0))
        history.append(parameters)
    return SampledNeuralTrajectoryResult(
        parameters,
        jnp.stack(history),
        jnp.stack(rate_errors) if rate_errors else jnp.zeros((0, 0)),
        jnp.stack(ess_history) if ess_history else jnp.zeros((0,)),
        jnp.stack(projection) if projection else jnp.zeros((0,)),
        jnp.stack(valid_history) if valid_history else jnp.zeros((0,), dtype=bool),
    )


__all__ = [
    "NeuralRateEvidence",
    "SampledNeuralTrajectoryProblem",
    "SampledNeuralTrajectoryResult",
    "solve_sampled_neural_trajectory",
]
