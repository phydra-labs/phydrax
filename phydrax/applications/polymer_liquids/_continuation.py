#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...continuation import (
    ContinuationResult,
    continue_branch,
    NaturalParameterContinuation,
    ParameterContinuationProblem,
)
from ...linalg import ArraySpace
from ...nonlinear import NewtonKrylov, NonlinearTermination
from ._prism import PreparedPRISM, PRISMEvaluation, PRISMResult


class PRISMDensityContinuationPlan(StrictModule, NonTrainableState):
    minimum_density_scale: float = eqx.field(static=True)
    maximum_density_scale: float = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_density_scale: float,
        maximum_density_scale: float,
        initial_step: float = 0.05,
        minimum_step: float = 1.0e-4,
        maximum_step: float = 0.25,
        maximum_steps: int = 64,
        maximum_retries: int = 8,
    ):
        lower = float(minimum_density_scale)
        upper = float(maximum_density_scale)
        initial = float(initial_step)
        minimum = float(minimum_step)
        maximum = float(maximum_step)
        steps = int(maximum_steps)
        retries = int(maximum_retries)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 < lower < upper
            or not math.isfinite(initial)
            or not math.isfinite(minimum)
            or not math.isfinite(maximum)
            or not 0.0 < minimum <= initial <= maximum
            or steps <= 0
            or retries < 0
        ):
            raise ValueError("PRISM density-continuation controls are invalid.")
        self.minimum_density_scale = lower
        self.maximum_density_scale = upper
        self.initial_step = initial
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.maximum_steps = steps
        self.maximum_retries = retries
        self.plan_id = canonical_fingerprint(
            {
                "kind": "prism-density-continuation-plan",
                "minimum_density_scale": lower,
                "maximum_density_scale": upper,
                "initial_step": initial,
                "minimum_step": minimum,
                "maximum_step": maximum,
                "maximum_steps": steps,
                "maximum_retries": retries,
            }
        )


class PRISMDensityContinuationResult(StrictModule):
    continuation: ContinuationResult
    final_evaluation: PRISMEvaluation
    successful: Array
    prepared_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def continue_prism_density(
    prepared: PreparedPRISM,
    initial_result: PRISMResult,
    initial_density_scale: float,
    target_density_scale: float,
    plan: PRISMDensityContinuationPlan,
    /,
) -> PRISMDensityContinuationResult:
    if not isinstance(prepared, PreparedPRISM):
        raise TypeError("prepared must be PreparedPRISM.")
    if not isinstance(initial_result, PRISMResult):
        raise TypeError("initial_result must be PRISMResult.")
    if not isinstance(plan, PRISMDensityContinuationPlan):
        raise TypeError("plan must be PRISMDensityContinuationPlan.")
    if initial_result.prepared_id != prepared.prepared_id:
        raise ValueError("Initial PRISM result belongs to another prepared problem.")
    if not bool(initial_result.successful):
        raise ValueError("Density continuation requires a successful initial PRISM root.")
    initial_scale = float(initial_density_scale)
    target_scale = float(target_density_scale)
    if (
        not plan.minimum_density_scale <= initial_scale <= plan.maximum_density_scale
        or not plan.minimum_density_scale <= target_scale <= plan.maximum_density_scale
        or initial_scale == target_scale
    ):
        raise ValueError("Initial and target density scales are invalid or identical.")
    shape = initial_result.nonlinear.state.shape
    space = ArraySpace(
        shape,
        dtype=initial_result.nonlinear.state.dtype,
        space_id=f"{prepared.prepared_id}:density-continuation-space",
    )

    def residual(gamma, density_scale, _):
        evaluation = prepared.evaluate(
            gamma,
            number_densities=prepared.mixture.number_densities * density_scale,
        )
        iteration_successful = (
            evaluation.closure.successful
            & evaluation.oz.successful
            & jnp.all(jnp.isfinite(evaluation.residual))
        )
        return jnp.where(iteration_successful, evaluation.residual, jnp.nan)

    problem = ParameterContinuationProblem(
        residual,
        parameter_lower=plan.minimum_density_scale,
        parameter_upper=plan.maximum_density_scale,
        state_space=space,
        residual_space=space,
        problem_id=f"{prepared.prepared_id}:density-continuation",
    )
    direction = 1 if target_scale > initial_scale else -1
    termination = NonlinearTermination(
        absolute_residual=prepared.plan.absolute_tolerance,
        relative_residual=prepared.plan.relative_tolerance,
        maximum_steps=prepared.plan.maximum_iterations,
    )
    continuation = continue_branch(
        problem,
        initial_result.nonlinear.state,
        jnp.asarray(initial_scale, dtype=initial_result.nonlinear.state.dtype),
        num_steps=plan.maximum_steps,
        method=NaturalParameterContinuation(
            corrector=NewtonKrylov(),
            termination=termination,
            initial_step=plan.initial_step,
            minimum_step=plan.minimum_step,
            maximum_step=plan.maximum_step,
            maximum_retries=plan.maximum_retries,
            direction=direction,
            predictor="tangent",
            predictor_failure="constant",
        ),
        terminal_coordinate=target_scale,
        branch_id=f"prism-density:{plan.plan_id}",
    )
    if continuation.points:
        final_point = continuation.points[-1]
        final_evaluation = prepared.evaluate(
            final_point.state,
            number_densities=(prepared.mixture.number_densities * final_point.coordinate),
        )
    else:
        final_evaluation = initial_result.evaluation
    successful = continuation.successful & final_evaluation.successful
    return PRISMDensityContinuationResult(
        continuation,
        final_evaluation,
        successful,
        prepared.prepared_id,
        plan.plan_id,
    )


__all__ = [
    "PRISMDensityContinuationPlan",
    "PRISMDensityContinuationResult",
    "continue_prism_density",
]
