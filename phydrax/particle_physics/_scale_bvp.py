#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded log-scale RGE integration with low/high boundary residual correction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..linalg import inverse


class ScaleBVPStatus(IntEnum):
    SUCCESS = 0
    MAXIMUM_STEPS_REACHED = 1
    SINGULAR_JACOBIAN = 2
    LINE_SEARCH_FAILED = 3
    NONFINITE = 4
    INVALID_DOMAIN = 5


class ScaleBVPPlan(StrictModule):
    """One finite low-to-high log-scale RGE and square boundary system."""

    beta_function: Callable[[Array, Array], Array]
    low_constraint: Callable[[Array], Array]
    high_constraint: Callable[[Array], Array]
    parameter_labels: tuple[str, ...] = eqx.field(static=True)
    low_residual_count: int = eqx.field(static=True)
    lower_scale: float = eqx.field(static=True)
    upper_scale: float = eqx.field(static=True)
    integration_steps: int = eqx.field(static=True)
    maximum_newton_steps: int = eqx.field(static=True)
    maximum_backtracks: int = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    source_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        parameter_labels: Sequence[str],
        beta_function: Callable[[Array, Array], Array],
        low_constraint: Callable[[Array], Array],
        high_constraint: Callable[[Array], Array],
        /,
        *,
        low_residual_count: int,
        lower_scale: float,
        upper_scale: float,
        integration_steps: int,
        maximum_newton_steps: int = 32,
        maximum_backtracks: int = 12,
        residual_tolerance: float = 1e-10,
        source_ids: Sequence[str],
    ):
        labels = tuple(str(value).strip() for value in parameter_labels)
        sources = tuple(sorted(str(value).strip() for value in source_ids))
        low_count = int(low_residual_count)
        lower = float(lower_scale)
        upper = float(upper_scale)
        integration = int(integration_steps)
        newton = int(maximum_newton_steps)
        backtracks = int(maximum_backtracks)
        tolerance = float(residual_tolerance)
        if (
            not labels
            or any(not value for value in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("parameter_labels must be unique and non-empty.")
        if (
            not callable(beta_function)
            or not callable(low_constraint)
            or not callable(high_constraint)
        ):
            raise TypeError("Scale BVP beta and constraint functions must be callable.")
        if not 0 <= low_count <= len(labels):
            raise ValueError("low_residual_count is outside the parameter dimension.")
        if not np.isfinite(lower) or not np.isfinite(upper) or not 0.0 < lower < upper:
            raise ValueError("Scale BVP scales must be finite, positive, and ordered.")
        if integration < 1 or newton < 1 or backtracks < 1:
            raise ValueError("Scale BVP work limits must be positive.")
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("Scale BVP residual tolerance is invalid.")
        if (
            not sources
            or any(not value for value in sources)
            or len(set(sources)) != len(sources)
        ):
            raise ValueError("Scale BVP source IDs must be unique and non-empty.")
        self.beta_function = beta_function
        self.low_constraint = low_constraint
        self.high_constraint = high_constraint
        self.parameter_labels = labels
        self.low_residual_count = low_count
        self.lower_scale = lower
        self.upper_scale = upper
        self.integration_steps = integration
        self.maximum_newton_steps = newton
        self.maximum_backtracks = backtracks
        self.residual_tolerance = tolerance
        self.source_ids = sources
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-spectrum-scale-bvp-plan",
                "parameter_labels": labels,
                "low_residual_count": low_count,
                "lower_scale": lower,
                "upper_scale": upper,
                "integration_steps": integration,
                "maximum_newton_steps": newton,
                "maximum_backtracks": backtracks,
                "residual_tolerance": tolerance,
                "source_ids": sources,
            }
        )

    @property
    def parameter_count(self) -> int:
        return len(self.parameter_labels)


class ScaleBVPEvaluation(StrictModule):
    log_scales: Array
    trajectory: Array
    residual: Array
    residual_norm: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class ScaleBVPResult(StrictModule):
    initial_parameters: Array
    final_parameters: Array
    log_scales: Array
    trajectory: Array
    residual: Array
    residual_history: Array
    accepted_steps: Array
    status: Array
    converged: Array
    finite: Array
    plan_id: str = eqx.field(static=True)
    claim: str = eqx.field(static=True)


def evaluate_scale_bvp(
    plan: ScaleBVPPlan,
    initial_parameters: ArrayLike,
    /,
) -> ScaleBVPEvaluation:
    if not isinstance(plan, ScaleBVPPlan):
        raise TypeError("plan must be ScaleBVPPlan.")
    initial = jnp.asarray(initial_parameters)
    if initial.shape != (plan.parameter_count,) or not jnp.issubdtype(
        initial.dtype, jnp.inexact
    ):
        raise ValueError("initial_parameters must match the inexact parameter vector.")
    lower_log = jnp.log(jnp.asarray(plan.lower_scale, dtype=initial.dtype))
    upper_log = jnp.log(jnp.asarray(plan.upper_scale, dtype=initial.dtype))
    step = (upper_log - lower_log) / plan.integration_steps
    values = [initial]
    current = initial
    time = lower_log
    for _ in range(plan.integration_steps):
        k1 = jnp.asarray(plan.beta_function(time, current))
        k2 = jnp.asarray(plan.beta_function(time + 0.5 * step, current + 0.5 * step * k1))
        k3 = jnp.asarray(plan.beta_function(time + 0.5 * step, current + 0.5 * step * k2))
        k4 = jnp.asarray(plan.beta_function(time + step, current + step * k3))
        for value in (k1, k2, k3, k4):
            if value.shape != current.shape:
                raise ValueError("beta_function must preserve the parameter shape.")
        current = current + step / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        time = time + step
        values.append(current)
    trajectory = jnp.stack(values)
    log_scales = jnp.linspace(lower_log, upper_log, plan.integration_steps + 1)
    low = jnp.asarray(plan.low_constraint(initial))
    high = jnp.asarray(plan.high_constraint(current))
    if low.shape != (plan.low_residual_count,) or high.shape != (
        plan.parameter_count - plan.low_residual_count,
    ):
        raise ValueError("Scale boundary residual dimensions contradict the plan.")
    residual = jnp.concatenate((low, high))
    norm = jnp.linalg.norm(residual)
    finite = jnp.all(jnp.isfinite(trajectory)) & jnp.all(jnp.isfinite(residual))
    return ScaleBVPEvaluation(
        log_scales=log_scales,
        trajectory=trajectory,
        residual=residual,
        residual_norm=norm,
        finite=finite,
        plan_id=plan.plan_id,
    )


def solve_scale_bvp(
    plan: ScaleBVPPlan,
    initial_guess: ArrayLike,
    /,
) -> ScaleBVPResult:
    if not isinstance(plan, ScaleBVPPlan):
        raise TypeError("plan must be ScaleBVPPlan.")
    parameters = jnp.asarray(initial_guess)
    if parameters.shape != (plan.parameter_count,):
        raise ValueError("initial_guess has the wrong parameter shape.")
    histories = []
    accepted = []
    status = ScaleBVPStatus.MAXIMUM_STEPS_REACHED
    converged = False
    evaluation = evaluate_scale_bvp(plan, parameters)

    def residual_function(value):
        return evaluate_scale_bvp(plan, value).residual

    for _ in range(plan.maximum_newton_steps):
        histories.append(evaluation.residual_norm)
        if not bool(evaluation.finite):
            status = ScaleBVPStatus.NONFINITE
            break
        if float(evaluation.residual_norm) <= plan.residual_tolerance:
            status = ScaleBVPStatus.SUCCESS
            converged = True
            break
        jacobian = jax.jacfwd(residual_function)(parameters)
        inverse_result = inverse(jacobian)
        if not bool(jnp.all(inverse_result.successful)):
            status = ScaleBVPStatus.SINGULAR_JACOBIAN
            accepted.append(False)
            break
        direction = -(inverse_result.value @ evaluation.residual)
        step = 1.0
        did_accept = False
        candidate = parameters
        candidate_evaluation = evaluation
        for _ in range(plan.maximum_backtracks):
            trial = parameters + step * direction
            trial_evaluation = evaluate_scale_bvp(plan, trial)
            if bool(
                trial_evaluation.finite
                & (trial_evaluation.residual_norm < evaluation.residual_norm)
            ):
                candidate = trial
                candidate_evaluation = trial_evaluation
                did_accept = True
                break
            step *= 0.5
        accepted.append(did_accept)
        if not did_accept:
            status = ScaleBVPStatus.LINE_SEARCH_FAILED
            break
        parameters = candidate
        evaluation = candidate_evaluation
    if (
        not converged
        and bool(evaluation.finite)
        and float(evaluation.residual_norm) <= plan.residual_tolerance
    ):
        status = ScaleBVPStatus.SUCCESS
        converged = True
    if not histories or float(histories[-1]) != float(evaluation.residual_norm):
        histories.append(evaluation.residual_norm)
    finite = evaluation.finite & jnp.all(jnp.isfinite(jnp.asarray(histories)))
    return ScaleBVPResult(
        initial_parameters=jnp.asarray(initial_guess),
        final_parameters=parameters,
        log_scales=evaluation.log_scales,
        trajectory=evaluation.trajectory,
        residual=evaluation.residual,
        residual_history=jnp.asarray(histories),
        accepted_steps=jnp.asarray(accepted, dtype=jnp.bool_),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        converged=jnp.asarray(converged),
        finite=finite,
        plan_id=plan.plan_id,
        claim="finite-native-log-scale-boundary-value-workflow-no-model-loop-correction-claim",
    )


__all__ = [
    "ScaleBVPEvaluation",
    "ScaleBVPPlan",
    "ScaleBVPResult",
    "ScaleBVPStatus",
    "evaluate_scale_bvp",
    "solve_scale_bvp",
]
