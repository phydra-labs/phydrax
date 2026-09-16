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
from ._scft import PreparedSCFT, SCFTEvaluation, SCFTResult


class SCFTInteractionContinuationPlan(StrictModule, NonTrainableState):
    minimum_scale: float = eqx.field(static=True)
    maximum_scale: float = eqx.field(static=True)
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_retries: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        minimum_scale: float,
        maximum_scale: float,
        initial_step: float = 0.05,
        minimum_step: float = 1.0e-4,
        maximum_step: float = 0.25,
        maximum_steps: int = 64,
        maximum_retries: int = 8,
    ):
        lower = float(minimum_scale)
        upper = float(maximum_scale)
        initial = float(initial_step)
        minimum = float(minimum_step)
        maximum = float(maximum_step)
        steps = int(maximum_steps)
        retries = int(maximum_retries)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or not 0.0 <= lower < upper
            or not 0.0 < minimum <= initial <= maximum
            or steps <= 0
            or retries < 0
        ):
            raise ValueError("SCFT interaction-continuation controls are invalid.")
        self.minimum_scale = lower
        self.maximum_scale = upper
        self.initial_step = initial
        self.minimum_step = minimum
        self.maximum_step = maximum
        self.maximum_steps = steps
        self.maximum_retries = retries
        self.plan_id = canonical_fingerprint(
            {
                "kind": "scft-interaction-continuation-plan",
                "minimum_scale": lower,
                "maximum_scale": upper,
                "initial_step": initial,
                "minimum_step": minimum,
                "maximum_step": maximum,
                "maximum_steps": steps,
                "maximum_retries": retries,
            }
        )


class SCFTInteractionContinuationResult(StrictModule):
    continuation: ContinuationResult
    final_evaluation: SCFTEvaluation
    successful: Array
    prepared_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def continue_scft_interactions(
    prepared: PreparedSCFT,
    initial_result: SCFTResult,
    initial_scale: float,
    target_scale: float,
    plan: SCFTInteractionContinuationPlan,
    /,
) -> SCFTInteractionContinuationResult:
    if not isinstance(prepared, PreparedSCFT):
        raise TypeError("prepared must be PreparedSCFT.")
    if not isinstance(initial_result, SCFTResult):
        raise TypeError("initial_result must be SCFTResult.")
    if not isinstance(plan, SCFTInteractionContinuationPlan):
        raise TypeError("plan must be SCFTInteractionContinuationPlan.")
    if initial_result.prepared_id != prepared.prepared_id:
        raise ValueError("Initial SCFT result belongs to another prepared problem.")
    if not bool(initial_result.successful):
        raise ValueError("SCFT continuation requires a successful initial root.")
    initial = float(initial_scale)
    target = float(target_scale)
    if (
        not plan.minimum_scale <= initial <= plan.maximum_scale
        or not plan.minimum_scale <= target <= plan.maximum_scale
        or initial == target
    ):
        raise ValueError(
            "Initial and target interaction scales are invalid or identical."
        )
    state = initial_result.nonlinear.state
    space = ArraySpace(
        state.shape,
        dtype=state.dtype,
        space_id=f"{prepared.prepared_id}:interaction-continuation-space",
    )

    def residual(fields, scale, _):
        evaluation = prepared.evaluate(fields, chi_n=prepared.plan.model.chi_n * scale)
        return jnp.where(evaluation.successful, evaluation.residual, jnp.nan)

    problem = ParameterContinuationProblem(
        residual,
        parameter_lower=plan.minimum_scale,
        parameter_upper=plan.maximum_scale,
        state_space=space,
        residual_space=space,
        problem_id=f"{prepared.prepared_id}:interaction-continuation",
    )
    termination = NonlinearTermination(
        absolute_residual=prepared.plan.absolute_tolerance,
        relative_residual=prepared.plan.relative_tolerance,
        maximum_steps=prepared.plan.maximum_iterations,
    )
    continuation = continue_branch(
        problem,
        state,
        jnp.asarray(initial, dtype=state.dtype),
        num_steps=plan.maximum_steps,
        method=NaturalParameterContinuation(
            corrector=NewtonKrylov(),
            termination=termination,
            initial_step=plan.initial_step,
            minimum_step=plan.minimum_step,
            maximum_step=plan.maximum_step,
            maximum_retries=plan.maximum_retries,
            direction=1 if target > initial else -1,
            predictor="tangent",
            predictor_failure="constant",
        ),
        terminal_coordinate=target,
        branch_id=f"scft-interaction:{plan.plan_id}",
    )
    if continuation.points:
        final_point = continuation.points[-1]
        final_evaluation = prepared.evaluate(
            final_point.state,
            chi_n=prepared.plan.model.chi_n * final_point.coordinate,
        )
    else:
        final_evaluation = initial_result.evaluation
    successful = continuation.successful & final_evaluation.successful
    return SCFTInteractionContinuationResult(
        continuation,
        final_evaluation,
        successful,
        prepared.prepared_id,
        plan.plan_id,
    )


__all__ = [
    "SCFTInteractionContinuationPlan",
    "SCFTInteractionContinuationResult",
    "continue_scft_interactions",
]
