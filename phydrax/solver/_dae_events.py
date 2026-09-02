#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import DenseLinearOperator, FactorizationPolicy, factorize
from ..metrix import AbstractStateGeometry
from ._dae_initialization import DAEInitializationResult, DAEInitializationSpec
from ._differential_algebraic import (
    DAESolvePolicy,
    DifferentialAlgebraicProblem,
    initialize_dae,
)
from ._hybrid_event import HybridEventSensitivityResult, HybridEventTape
from ._hybrid_schedule import HybridSchedulePlan


class DAEConsistencyPolicy(StrictModule, NonTrainableState):
    """Explicit admissibility limits for a computed DAE consistency candidate."""

    maximum_state_correction: float = eqx.field(static=True)
    maximum_rate_correction: float = eqx.field(static=True)
    weighted_correction_tolerance: float = eqx.field(static=True)
    failure: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_state_correction: float,
        maximum_rate_correction: float,
        weighted_correction_tolerance: float,
        /,
        *,
        failure: int = -1,
    ):
        values = tuple(
            float(value)
            for value in (
                maximum_state_correction,
                maximum_rate_correction,
                weighted_correction_tolerance,
            )
        )
        if any(not np.isfinite(value) or value < 0.0 for value in values):
            raise ValueError(
                "DAE consistency correction limits must be finite and nonnegative."
            )
        if not isinstance(failure, int) or isinstance(failure, bool):
            raise TypeError("failure must be an integer status.")
        self.maximum_state_correction = values[0]
        self.maximum_rate_correction = values[1]
        self.weighted_correction_tolerance = values[2]
        self.failure = failure
        self.policy_id = canonical_fingerprint(
            {
                "kind": "dae-consistency-policy",
                "maximum_state_correction": values[0],
                "maximum_rate_correction": values[1],
                "weighted_correction_tolerance": values[2],
                "failure": failure,
            }
        )


class DAEConsistencyCandidate(StrictModule, NonTrainableState):
    """A consistency result that never mutates or silently repairs its source."""

    initialization: DAEInitializationResult
    original_state: Array
    original_rate: Array
    state_correction_norm: Array
    rate_correction_norm: Array
    weighted_correction_norm: Array
    admissible: Array
    status: Array
    policy_id: str = eqx.field(static=True)

    def apply(
        self, problem: DifferentialAlgebraicProblem, /
    ) -> DifferentialAlgebraicProblem:
        """Construct a new problem only when the candidate passes every limit."""

        if not isinstance(problem, DifferentialAlgebraicProblem):
            raise TypeError("problem must be a DifferentialAlgebraicProblem.")
        admissible = bool(np.asarray(jax.device_get(self.admissible)))
        if not admissible:
            raise ValueError(
                "An inadmissible DAE consistency candidate cannot be applied."
            )
        return DifferentialAlgebraicProblem(
            problem.system,
            self.initialization.state,
            initial_state_rate=self.initialization.state_rate,
            args=problem.args,
            input_policy=problem.input_policy,
            initialization=problem.initialization,
            discretization_bundle=problem.discretization_bundle,
            problem_id=f"consistent:{problem.problem_id}",
        )


def dae_consistency_candidate(
    problem: DifferentialAlgebraicProblem,
    time: ArrayLike,
    policy: DAEConsistencyPolicy,
    /,
    *,
    solve_policy: DAESolvePolicy | None = None,
    args: Any = None,
    state_guess: ArrayLike | None = None,
    rate_guess: ArrayLike | None = None,
) -> DAEConsistencyCandidate:
    """Compute and bound one candidate; adoption remains a separate explicit call."""

    if not isinstance(problem, DifferentialAlgebraicProblem):
        raise TypeError("problem must be a DifferentialAlgebraicProblem.")
    if not isinstance(policy, DAEConsistencyPolicy):
        raise TypeError("policy must be a DAEConsistencyPolicy.")
    original_state = (
        problem.initial_state if state_guess is None else jnp.asarray(state_guess)
    )
    original_rate = (
        problem.initial_state_rate if rate_guess is None else jnp.asarray(rate_guess)
    )
    result = initialize_dae(
        problem,
        time,
        policy=solve_policy,
        args=problem.args if args is None else args,
        initial_state=original_state,
        initial_state_rate=original_rate,
    )
    state_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.state_correction))))
    rate_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(result.rate_correction))))
    state_scale = problem.system.state_scale
    rate_scale = problem.system.state_rate_scale
    weighted = jnp.sqrt(
        0.5
        * (
            jnp.mean(jnp.square(jnp.abs(result.state_correction / state_scale)))
            + jnp.mean(jnp.square(jnp.abs(result.rate_correction / rate_scale)))
        )
    )
    admissible = (
        result.valid
        & jnp.isfinite(state_norm)
        & jnp.isfinite(rate_norm)
        & jnp.isfinite(weighted)
        & (state_norm <= policy.maximum_state_correction)
        & (rate_norm <= policy.maximum_rate_correction)
        & (weighted <= policy.weighted_correction_tolerance)
    )
    return DAEConsistencyCandidate(
        result,
        original_state,
        original_rate,
        state_norm,
        rate_norm,
        weighted,
        admissible,
        jnp.where(admissible, result.status, policy.failure).astype(jnp.int32),
        policy.policy_id,
    )


class DAEResetMap(StrictModule, NonTrainableState):
    """A reset returning explicit post-event state/rate guesses and mask contract."""

    reset: Callable[[Array, Array, Array, Any], tuple[Array, Array]]
    initialization: DAEInitializationSpec
    reset_id: str = eqx.field(static=True)

    def __init__(
        self,
        reset: Callable[[Array, Array, Array, Any], tuple[Array, Array]],
        initialization: DAEInitializationSpec,
        /,
        *,
        reset_id: str,
    ):
        if not callable(reset):
            raise TypeError("DAEResetMap reset must be callable.")
        if not isinstance(initialization, DAEInitializationSpec):
            raise TypeError("initialization must be a DAEInitializationSpec.")
        if not isinstance(reset_id, str) or not reset_id:
            raise ValueError("reset_id must be non-empty.")
        self.reset = reset
        self.initialization = initialization
        self.reset_id = canonical_fingerprint(
            {
                "kind": "dae-reset-map",
                "user_id": reset_id,
                "initialization": initialization.initialization_id,
            }
        )


class DAEEventPlan(StrictModule, NonTrainableState):
    """Bind a canonical event schedule to consistent DAE restart maps."""

    schedule: HybridSchedulePlan
    reset_maps: tuple[DAEResetMap, ...]
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        schedule: HybridSchedulePlan,
        reset_maps: Sequence[DAEResetMap],
        /,
    ):
        resets = tuple(reset_maps)
        if not isinstance(schedule, HybridSchedulePlan):
            raise TypeError("schedule must be a HybridSchedulePlan.")
        if len(resets) != len(schedule.events) or any(
            not isinstance(value, DAEResetMap) for value in resets
        ):
            raise ValueError(
                "DAE event schedules require exactly one reset map per event."
            )
        self.schedule = schedule
        self.reset_maps = resets
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dae-event-plan",
                "schedule": schedule.plan_id,
                "resets": [value.reset_id for value in resets],
            }
        )


class DAEEventEvidence(StrictModule, NonTrainableState):
    event_index: Array
    localized: HybridEventSensitivityResult
    consistency: DAEConsistencyCandidate
    pre_residual_norm: Array
    post_residual_norm: Array
    bdf_restart_order: Array
    tape: HybridEventTape
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


def apply_dae_event(
    plan: DAEEventPlan,
    problem: DifferentialAlgebraicProblem,
    event_index: int,
    localized: HybridEventSensitivityResult,
    tape: HybridEventTape,
    consistency_policy: DAEConsistencyPolicy,
    /,
    *,
    time: ArrayLike | None = None,
    args: Any = None,
    solve_policy: DAESolvePolicy | None = None,
) -> tuple[DifferentialAlgebraicProblem | None, DAEEventEvidence]:
    """Apply one localized reset through an explicit consistent restart candidate."""

    if not isinstance(plan, DAEEventPlan):
        raise TypeError("plan must be a DAEEventPlan.")
    if event_index < 0 or event_index >= len(plan.reset_maps):
        raise ValueError("event_index is outside the prepared DAE event table.")
    if not isinstance(localized, HybridEventSensitivityResult):
        raise TypeError("localized must be HybridEventSensitivityResult.")
    event_time = localized.event_time if time is None else jnp.asarray(time)
    reset = plan.reset_maps[event_index]
    runtime_args = problem.args if args is None else args
    state_guess, rate_guess = reset.reset(
        event_time,
        localized.state_before,
        problem.initial_state_rate,
        runtime_args,
    )
    reset_problem = DifferentialAlgebraicProblem(
        problem.system,
        state_guess,
        initial_state_rate=rate_guess,
        args=runtime_args,
        input_policy=problem.input_policy,
        initialization=reset.initialization,
        discretization_bundle=problem.discretization_bundle,
        problem_id=f"reset-candidate:{problem.problem_id}:{reset.reset_id}",
    )
    consistency = dae_consistency_candidate(
        reset_problem,
        event_time,
        consistency_policy,
        solve_policy=solve_policy,
        args=runtime_args,
    )
    pre = problem.system.evaluate(
        event_time,
        localized.state_before,
        problem.initial_state_rate,
        runtime_args,
    )
    post = problem.system.evaluate(
        event_time,
        consistency.initialization.state,
        consistency.initialization.state_rate,
        runtime_args,
    )
    pre_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(pre))))
    post_norm = jnp.sqrt(jnp.mean(jnp.square(jnp.abs(post))))
    valid = localized.successful & consistency.admissible & (~tape.capacity_exceeded)
    evidence = DAEEventEvidence(
        jnp.asarray(event_index, dtype=jnp.int32),
        localized,
        consistency,
        pre_norm,
        post_norm,
        jnp.asarray(1, dtype=jnp.int32),
        tape,
        valid,
        jnp.where(valid, consistency.status, consistency_policy.failure).astype(
            jnp.int32
        ),
        plan.plan_id,
    )
    adopted = (
        consistency.apply(reset_problem)
        if bool(np.asarray(jax.device_get(valid)))
        else None
    )
    return adopted, evidence


class DAERegularityDomain(StrictModule, NonTrainableState):
    """Finite coordinate cells over which one operator enclosure is claimed."""

    lower: Array
    upper: Array
    domain_id: str = eqx.field(static=True)

    def __init__(self, lower: ArrayLike, upper: ArrayLike, /, *, domain_id: str):
        lower_ = jnp.asarray(lower)
        upper_ = jnp.asarray(upper, dtype=lower_.dtype)
        if lower_.ndim != 2 or lower_.shape != upper_.shape or lower_.shape[0] == 0:
            raise ValueError(
                "DAE regularity cells require matching nonempty shape (cells,coordinates)."
            )
        if (
            np.any(~np.isfinite(np.asarray(lower_)))
            or np.any(~np.isfinite(np.asarray(upper_)))
            or np.any(np.asarray(lower_) >= np.asarray(upper_))
        ):
            raise ValueError(
                "DAE regularity cells must be finite and have positive width."
            )
        if not isinstance(domain_id, str) or not domain_id:
            raise ValueError("domain_id must be non-empty.")
        self.lower = lower_
        self.upper = upper_
        self.domain_id = canonical_fingerprint(
            {
                "kind": "dae-regularity-domain",
                "user_id": domain_id,
                "cells": int(lower_.shape[0]),
                "coordinates": int(lower_.shape[1]),
            }
        )


class DAERegularityCertificatePlan(StrictModule, NonTrainableState):
    """Typed center operator and certified variation bound provider."""

    domain: DAERegularityDomain
    enclosure: Callable[[Array, Any], tuple[Array, Array]]
    operator_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        domain: DAERegularityDomain,
        enclosure: Callable[[Array, Any], tuple[Array, Array]],
        /,
        *,
        operator_id: str,
    ):
        if not isinstance(domain, DAERegularityDomain):
            raise TypeError("domain must be a DAERegularityDomain.")
        if not callable(enclosure):
            raise TypeError("enclosure must be callable.")
        if not isinstance(operator_id, str) or not operator_id:
            raise ValueError("operator_id must be non-empty.")
        self.domain = domain
        self.enclosure = enclosure
        self.operator_id = operator_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dae-regularity-certificate-plan",
                "domain": domain.domain_id,
                "operator": operator_id,
            }
        )


class DAERegularityCertificate(StrictModule, NonTrainableState):
    lower_singular_value_bounds: Array
    center_singular_values: Array
    variation_bounds: Array
    covered: Array
    uncovered_cells: Array
    certified: Array
    status: Array
    hypotheses: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def certify_dae_regularity(
    plan: DAERegularityCertificatePlan,
    /,
    *,
    args: Any = None,
) -> DAERegularityCertificate:
    """Certify only cells with sigma_min(A(center)) minus variation strictly positive."""

    if not isinstance(plan, DAERegularityCertificatePlan):
        raise TypeError("plan must be a DAERegularityCertificatePlan.")
    centers = 0.5 * (plan.domain.lower + plan.domain.upper)
    minimum = []
    variation = []
    for cell in range(centers.shape[0]):
        matrix, bound = plan.enclosure(centers[cell], args)
        matrix_ = jnp.asarray(matrix)
        bound_ = jnp.asarray(bound, dtype=matrix_.real.dtype).reshape(())
        if matrix_.ndim != 2 or matrix_.shape[0] != matrix_.shape[1]:
            raise ValueError(
                "DAE regularity enclosure operators must be square matrices."
            )
        factorization = factorize(
            DenseLinearOperator(matrix_, operator_id=f"{plan.operator_id}:cell:{cell}"),
            FactorizationPolicy("svd"),
        )
        singular_values = factorization.singular_values()
        minimum.append(jnp.min(singular_values))
        variation.append(bound_)
    center_values = jnp.stack(tuple(minimum))
    variation_values = jnp.stack(tuple(variation))
    lower_bounds = center_values - variation_values
    covered = (
        jnp.isfinite(center_values)
        & jnp.isfinite(variation_values)
        & (variation_values >= 0)
        & (lower_bounds > 0)
    )
    uncovered = jnp.nonzero(~covered, size=covered.shape[0], fill_value=-1)[0]
    certified = jnp.all(covered)
    return DAERegularityCertificate(
        lower_bounds,
        center_values,
        variation_values,
        covered,
        uncovered,
        certified,
        jnp.where(certified, 0, -1).astype(jnp.int32),
        "finite declared cells; center singular value minus certified operator variation",
        plan.plan_id,
    )


class ManifoldBDFMethod(StrictModule, NonTrainableState):
    """Prepared local-coordinate BDF1/BDF2 method; higher orders fail closed."""

    order: int = eqx.field(static=True)
    coefficients: tuple[float, ...] = eqx.field(static=True)
    method_id: str = eqx.field(static=True)

    def __init__(self, order: int = 1, /):
        if order not in (1, 2):
            raise ValueError("Manifold BDF currently supports only orders one and two.")
        coefficients = (1.0, -1.0) if order == 1 else (1.5, -2.0, 0.5)
        self.order = order
        self.coefficients = coefficients
        self.method_id = f"manifold-bdf:{order}"


class ManifoldBDFStage(StrictModule, NonTrainableState):
    state: Array
    state_rate: Array
    local_coordinate: Array
    local_rate: Array
    contained: Array
    chart_valid: Array
    method_id: str = eqx.field(static=True)


def manifold_bdf_stage(
    method: ManifoldBDFMethod,
    geometry: AbstractStateGeometry,
    base_state: ArrayLike,
    history: Sequence[ArrayLike],
    candidate_local: ArrayLike,
    step_size: ArrayLike,
    /,
) -> ManifoldBDFStage:
    """Construct one fixed-chart manifold BDF endpoint and physical tangent."""

    if not isinstance(method, ManifoldBDFMethod):
        raise TypeError("method must be a ManifoldBDFMethod.")
    if not isinstance(geometry, AbstractStateGeometry):
        raise TypeError("geometry must be an AbstractStateGeometry.")
    states = tuple(jnp.asarray(value) for value in history)
    if len(states) != method.order:
        raise ValueError("history length must equal the prepared manifold BDF order.")
    base = jnp.asarray(base_state)
    local = jnp.asarray(candidate_local)
    dt = jnp.asarray(step_size, dtype=base.real.dtype).reshape(())
    history_local = tuple(
        jnp.asarray(geometry.inverse_retract(base, value)) for value in states
    )
    local_rate = method.coefficients[0] * local
    for coefficient, value in zip(method.coefficients[1:], history_local, strict=True):
        local_rate = local_rate + coefficient * value
    local_rate = local_rate / dt
    state, state_rate = jax.jvp(
        lambda tangent: geometry.retract(base, tangent),
        (local,),
        (local_rate,),
    )
    contained = jnp.asarray(geometry.contains(state), dtype=bool)
    chart_valid = (
        contained
        & jnp.all(jnp.isfinite(state))
        & jnp.all(jnp.isfinite(state_rate))
        & (dt > 0)
    )
    return ManifoldBDFStage(
        state,
        jnp.where(chart_valid, state_rate, jnp.nan),
        local,
        local_rate,
        contained,
        chart_valid,
        method.method_id,
    )


__all__ = [
    "DAEConsistencyCandidate",
    "DAEConsistencyPolicy",
    "DAEEventEvidence",
    "DAEEventPlan",
    "DAERegularityCertificate",
    "DAERegularityCertificatePlan",
    "DAERegularityDomain",
    "DAEResetMap",
    "ManifoldBDFMethod",
    "ManifoldBDFStage",
    "apply_dae_event",
    "certify_dae_regularity",
    "dae_consistency_candidate",
    "manifold_bdf_stage",
]
