#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled,
    tree_allfinite,
    tree_inner,
    tree_norm,
    tree_scale,
)
from ..linalg import (
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    PyTreeSpace,
    TolerancePolicy,
)
from ._bordered import (
    BorderedLinearSystem,
    plan_bordered_solve,
    prepare_bordered_solve,
    refresh_bordered_solve,
    solve_bordered,
)
from ._core import (
    BranchPoint,
    ContinuationBranch,
    ContinuationCurveProblem,
    ContinuationStatus,
    EventBracket,
)


EventIndicator = Callable[
    [ContinuationCurveProblem, PyTree[Any], Array, Any],
    Any,
]


class EventLocalizationStatus(IntEnum):
    """Portable terminal status for heuristic event localization."""

    SUCCESS = 0
    INVALID_BRACKET = 1
    CORRECTOR_FAILED = 2
    NONFINITE = 3
    MAXIMUM_STEPS_REACHED = 4


class EventLocalizationPolicy(StrictModule):
    """Safeguarded interpolation and bordered Newton correction policy."""

    linear_policy: LinearSolvePolicy
    bracket_tolerance: float = eqx.field(static=True)
    indicator_tolerance: float = eqx.field(static=True)
    residual_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    maximum_corrector_steps: int = eqx.field(static=True)
    secant_safeguard: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        bracket_tolerance: float = 1e-8,
        indicator_tolerance: float = 1e-8,
        residual_tolerance: float = 1e-9,
        maximum_steps: int = 40,
        maximum_corrector_steps: int = 12,
        secant_safeguard: float = 0.1,
        policy_id: str | None = None,
    ):
        linear_policy_ = (
            LinearSolvePolicy(
                GMRES(),
                tolerance=TolerancePolicy(relative=1e-8, absolute=1e-11),
            )
            if linear_policy is None
            else linear_policy
        )
        if not isinstance(linear_policy_, LinearSolvePolicy):
            raise TypeError("linear_policy must be a LinearSolvePolicy or None.")
        if linear_policy_.failure.mode != "status":
            raise ValueError("Event localization requires linear failure mode 'status'.")
        tolerances = tuple(
            float(value)
            for value in (
                bracket_tolerance,
                indicator_tolerance,
                residual_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Event localization tolerances must be finite and non-negative."
            )
        steps = int(maximum_steps)
        corrector_steps = int(maximum_corrector_steps)
        safeguard = float(secant_safeguard)
        if steps < 1 or corrector_steps < 1:
            raise ValueError("Event localization step limits must be positive.")
        if not isfinite(safeguard) or not 0.0 < safeguard < 0.5:
            raise ValueError("secant_safeguard must lie strictly between zero and 0.5.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "event-localization-policy",
                    "linear_method": linear_policy_.method.name,
                    "preconditioner": (
                        "none"
                        if linear_policy_.preconditioning is None
                        else type(linear_policy_.preconditioning.builder).__qualname__
                    ),
                    "bracket_tolerance": tolerances[0],
                    "indicator_tolerance": tolerances[1],
                    "residual_tolerance": tolerances[2],
                    "maximum_steps": steps,
                    "maximum_corrector_steps": corrector_steps,
                    "secant_safeguard": safeguard,
                }
            )
            if policy_id is None
            else str(policy_id)
        )
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.linear_policy = linear_policy_
        (
            self.bracket_tolerance,
            self.indicator_tolerance,
            self.residual_tolerance,
        ) = tolerances
        self.maximum_steps = steps
        self.maximum_corrector_steps = corrector_steps
        self.secant_safeguard = safeguard
        self.policy_id = identifier


class EventLocalizationDiagnostics(StrictModule):
    """Final sign bracket, residual, bordered work, and cache-reuse evidence."""

    iterations: Array
    corrector_iterations: Array
    bordered_preparations: Array
    bordered_numeric_refreshes: Array
    cached_bordered_column_reuses: Array
    bracket_width: Array
    left_indicator: Array
    right_indicator: Array
    localized_indicator: Array
    residual_norm: Array
    linear_solve_status: Array

    def __init__(
        self,
        *,
        iterations: Any,
        corrector_iterations: Any,
        bordered_preparations: Any,
        bordered_numeric_refreshes: Any,
        cached_bordered_column_reuses: Any,
        bracket_width: Any,
        left_indicator: Any,
        right_indicator: Any,
        localized_indicator: Any,
        residual_norm: Any,
        linear_solve_status: Any,
    ):
        integer_values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                iterations,
                corrector_iterations,
                bordered_preparations,
                bordered_numeric_refreshes,
                cached_bordered_column_reuses,
                linear_solve_status,
            )
        )
        (
            self.iterations,
            self.corrector_iterations,
            self.bordered_preparations,
            self.bordered_numeric_refreshes,
            self.cached_bordered_column_reuses,
            self.linear_solve_status,
        ) = integer_values
        self.bracket_width = jnp.asarray(bracket_width)
        self.left_indicator = jnp.asarray(left_indicator)
        self.right_indicator = jnp.asarray(right_indicator)
        self.localized_indicator = jnp.asarray(localized_indicator)
        self.residual_norm = jnp.asarray(residual_norm)


class EventLocalizationProvenance(StrictModule):
    """Stable problem, bracket, policy, bordered, and indicator identities."""

    bordered_numeric_version: Array
    problem_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    bracket_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    linear_method: str = eqx.field(static=True)
    bordered_plan_id: str = eqx.field(static=True)
    bordered_prepared_id: str = eqx.field(static=True)
    indicator_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        branch_id: str,
        bracket_id: str,
        policy_id: str,
        linear_method: str,
        bordered_plan_id: str,
        bordered_prepared_id: str,
        bordered_numeric_version: Any,
        indicator_id: str,
    ):
        values = tuple(
            str(value)
            for value in (
                problem_id,
                branch_id,
                bracket_id,
                policy_id,
                linear_method,
                bordered_plan_id,
                bordered_prepared_id,
                indicator_id,
            )
        )
        if any(not value for value in values):
            raise ValueError(
                "Event localization provenance identities must be non-empty."
            )
        (
            self.problem_id,
            self.branch_id,
            self.bracket_id,
            self.policy_id,
            self.linear_method,
            self.bordered_plan_id,
            self.bordered_prepared_id,
            self.indicator_id,
        ) = values
        version = jnp.asarray(bordered_numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("bordered_numeric_version must be scalar.")
        self.bordered_numeric_version = version


class EventLocalizationResult(StrictModule):
    """Localized indicator point or explicit terminal failure evidence."""

    point: BranchPoint | None
    status: Array
    diagnostics: EventLocalizationDiagnostics
    provenance: EventLocalizationProvenance

    def __init__(
        self,
        point: BranchPoint | None,
        status: Any,
        diagnostics: EventLocalizationDiagnostics,
        provenance: EventLocalizationProvenance,
        /,
    ):
        if point is not None and not isinstance(point, BranchPoint):
            raise TypeError("point must be a BranchPoint or None.")
        if not isinstance(diagnostics, EventLocalizationDiagnostics):
            raise TypeError("diagnostics must be EventLocalizationDiagnostics.")
        if not isinstance(provenance, EventLocalizationProvenance):
            raise TypeError("provenance must be EventLocalizationProvenance.")
        self.point = point
        self.status = jnp.asarray(status, dtype=jnp.int32)
        self.diagnostics = diagnostics
        self.provenance = provenance

    @property
    def successful(self) -> Array:
        return self.status == int(EventLocalizationStatus.SUCCESS)


def _indicator_value(
    indicator: EventIndicator,
    problem: ContinuationCurveProblem,
    point: BranchPoint,
    args: Any,
    /,
) -> Array:
    value = jnp.asarray(indicator(problem, point.state, point.coordinate, args))
    if value.shape != () or not jnp.issubdtype(value.dtype, jnp.floating):
        raise TypeError(
            "event indicator must return one real floating-point scalar array."
        )
    return value


def _branch_difference(
    left: BranchPoint,
    right: BranchPoint,
    /,
) -> tuple[PyTree[Array], Array]:
    return (
        tree_add_scaled(right.state, left.state, -1.0),
        right.coordinate - left.coordinate,
    )


def _branch_width(left: BranchPoint, right: BranchPoint, /) -> Array:
    state_difference, coordinate_difference = _branch_difference(left, right)
    return jnp.sqrt(
        tree_inner(state_difference, state_difference) + coordinate_difference**2
    )


def _normalized_chord(
    left: BranchPoint,
    right: BranchPoint,
    /,
) -> tuple[PyTree[Array], Array]:
    state_difference, coordinate_difference = _branch_difference(left, right)
    norm = _branch_width(left, right)
    return tree_scale(1.0 / norm, state_difference), coordinate_difference / norm


def _interpolate_tree(
    left: PyTree[Any],
    right: PyTree[Any],
    weight: Any,
    /,
) -> PyTree[Array]:
    return tree_add_scaled(left, tree_add_scaled(right, left, -1.0), weight)


def _provenance(
    problem: ContinuationCurveProblem,
    branch: ContinuationBranch,
    bracket: EventBracket,
    policy: EventLocalizationPolicy,
    indicator_id: str,
    /,
    *,
    bordered_plan_id: str = "not-prepared",
    bordered_prepared_id: str = "not-prepared",
    bordered_numeric_version: Any = -1,
) -> EventLocalizationProvenance:
    return EventLocalizationProvenance(
        problem_id=problem.problem_id,
        branch_id=branch.branch_id,
        bracket_id=bracket.bracket_id,
        policy_id=policy.policy_id,
        linear_method=policy.linear_policy.method.name,
        bordered_plan_id=bordered_plan_id,
        bordered_prepared_id=bordered_prepared_id,
        bordered_numeric_version=bordered_numeric_version,
        indicator_id=indicator_id,
    )


def _result(
    *,
    point: BranchPoint | None,
    status: EventLocalizationStatus,
    iterations: int,
    corrector_iterations: int,
    bordered_preparations: int,
    bordered_numeric_refreshes: int,
    cached_bordered_column_reuses: int,
    bracket_width: Any,
    left_indicator: Any,
    right_indicator: Any,
    localized_indicator: Any,
    residual_norm: Any,
    linear_solve_status: Any,
    provenance: EventLocalizationProvenance,
) -> EventLocalizationResult:
    return EventLocalizationResult(
        point,
        status,
        EventLocalizationDiagnostics(
            iterations=iterations,
            corrector_iterations=corrector_iterations,
            bordered_preparations=bordered_preparations,
            bordered_numeric_refreshes=bordered_numeric_refreshes,
            cached_bordered_column_reuses=cached_bordered_column_reuses,
            bracket_width=bracket_width,
            left_indicator=left_indicator,
            right_indicator=right_indicator,
            localized_indicator=localized_indicator,
            residual_norm=residual_norm,
            linear_solve_status=linear_solve_status,
        ),
        provenance,
    )


def _invalid_result(
    provenance: EventLocalizationProvenance,
    /,
    *,
    bracket_width: Any = jnp.inf,
    left_indicator: Any = jnp.nan,
    right_indicator: Any = jnp.nan,
) -> EventLocalizationResult:
    return _result(
        point=None,
        status=EventLocalizationStatus.INVALID_BRACKET,
        iterations=0,
        corrector_iterations=0,
        bordered_preparations=0,
        bordered_numeric_refreshes=0,
        cached_bordered_column_reuses=0,
        bracket_width=bracket_width,
        left_indicator=left_indicator,
        right_indicator=right_indicator,
        localized_indicator=jnp.nan,
        residual_norm=jnp.inf,
        linear_solve_status=-1,
        provenance=provenance,
    )


def _correct_candidate(
    problem: ContinuationCurveProblem,
    left: BranchPoint,
    right: BranchPoint,
    weight: float,
    policy: EventLocalizationPolicy,
    prepared: Any,
    args: Any,
    /,
):
    state_tangent, coordinate_tangent = _normalized_chord(left, right)
    predicted_state = _interpolate_tree(left.state, right.state, weight)
    predicted_coordinate = left.coordinate + weight * (right.coordinate - left.coordinate)
    state = predicted_state
    coordinate = predicted_coordinate
    bordered_preparations = 0
    bordered_refreshes = 0
    cached_reuses = 0
    last_status = jnp.asarray(-1, dtype=jnp.int32)
    residual_norm = jnp.asarray(jnp.inf, dtype=predicted_coordinate.dtype)
    system_id = f"{problem.problem_id}/event-localization-bordered-system"
    plan_id = f"{problem.problem_id}/event-localization-bordered-plan"
    for iteration in range(policy.maximum_corrector_steps + 1):
        residual, state_linearization = jax.linearize(
            lambda candidate: problem.residual(candidate, coordinate, args),
            state,
        )
        _, coordinate_action = jax.jvp(
            lambda candidate: problem.residual(state, candidate, args),
            (coordinate,),
            (jnp.ones_like(coordinate),),
        )
        displacement = tree_add_scaled(state, predicted_state, -1.0)
        hyperplane = (
            tree_inner(displacement, state_tangent)
            + (coordinate - predicted_coordinate) * coordinate_tangent
        )
        residual_norm = jnp.sqrt(
            tree_inner(residual, residual) + jnp.abs(hyperplane) ** 2
        )
        finite = (
            tree_allfinite(state) & jnp.isfinite(coordinate) & jnp.isfinite(residual_norm)
        )
        if not bool(finite):
            return (
                state,
                coordinate,
                state_tangent,
                coordinate_tangent,
                residual_norm,
                last_status,
                iteration,
                prepared,
                bordered_preparations,
                bordered_refreshes,
                cached_reuses,
                False,
                False,
            )
        if float(residual_norm) <= policy.residual_tolerance:
            return (
                state,
                coordinate,
                state_tangent,
                coordinate_tangent,
                tree_norm(residual),
                last_status,
                iteration,
                prepared,
                bordered_preparations,
                bordered_refreshes,
                cached_reuses,
                True,
                True,
            )
        if iteration == policy.maximum_corrector_steps:
            break
        space = PyTreeSpace(
            state,
            space_id=f"{problem.problem_id}/event-localization-space",
        )
        operator = FunctionLinearOperator(
            state_linearization,
            source=space,
            target=space,
            operator_id=f"{problem.problem_id}/event-localization-jacobian",
        )
        system = BorderedLinearSystem(
            operator,
            coordinate_action,
            state_tangent,
            coordinate_tangent,
            system_id=system_id,
        )
        if prepared is None:
            plan = plan_bordered_solve(
                system,
                policy.linear_policy,
                plan_id=plan_id,
            )
            prepared = prepare_bordered_solve(system, plan)
            bordered_preparations += 1
        else:
            prepared = refresh_bordered_solve(prepared, system)
            bordered_refreshes += 1
        correction = solve_bordered(
            prepared,
            tree_scale(-1.0, residual),
            -hyperplane,
        )
        last_status = correction.status
        cached_reuses += int(bool(correction.diagnostics.cached_column_solve_reused))
        if not bool(correction.successful):
            return (
                state,
                coordinate,
                state_tangent,
                coordinate_tangent,
                tree_norm(residual),
                last_status,
                iteration + 1,
                prepared,
                bordered_preparations,
                bordered_refreshes,
                cached_reuses,
                False,
                True,
            )
        state = tree_add_scaled(state, correction.value.primal, 1.0)
        coordinate = coordinate + correction.value.scalar
    return (
        state,
        coordinate,
        state_tangent,
        coordinate_tangent,
        residual_norm,
        last_status,
        policy.maximum_corrector_steps,
        prepared,
        bordered_preparations,
        bordered_refreshes,
        cached_reuses,
        False,
        True,
    )


def localize_event(
    problem: ContinuationCurveProblem,
    branch: ContinuationBranch,
    bracket: EventBracket,
    indicator: EventIndicator,
    /,
    *,
    indicator_id: str,
    policy: EventLocalizationPolicy | None = None,
    args: Any = None,
) -> EventLocalizationResult:
    """Refine a heuristic sign bracket along the corrected solution branch.

    A successful result localizes only the supplied numerical indicator. It is
    not a mathematical bifurcation certificate.
    """
    if not isinstance(problem, ContinuationCurveProblem):
        raise TypeError("problem must be a ContinuationCurveProblem.")
    if not isinstance(branch, ContinuationBranch):
        raise TypeError("branch must be a ContinuationBranch.")
    if not isinstance(bracket, EventBracket):
        raise TypeError("bracket must be an EventBracket.")
    if not callable(indicator):
        raise TypeError("indicator must be callable.")
    policy_ = EventLocalizationPolicy() if policy is None else policy
    if not isinstance(policy_, EventLocalizationPolicy):
        raise TypeError("policy must be an EventLocalizationPolicy or None.")
    indicator_identifier = str(indicator_id)
    if not indicator_identifier:
        raise ValueError("indicator_id must be non-empty.")
    provenance = _provenance(
        problem,
        branch,
        bracket,
        policy_,
        indicator_identifier,
    )
    if branch.problem_id != problem.problem_id:
        return _invalid_result(provenance)
    points = {point.point_id: point for point in branch.points}
    if bracket.left_point_id not in points or bracket.right_point_id not in points:
        return _invalid_result(provenance)
    left = points[bracket.left_point_id]
    right = points[bracket.right_point_id]
    width = _branch_width(left, right)
    if not bool(jnp.isfinite(width) & (width > 0.0)):
        return _invalid_result(provenance, bracket_width=width)
    left_value = _indicator_value(indicator, problem, left, args)
    right_value = _indicator_value(indicator, problem, right, args)
    if not bool(jnp.isfinite(left_value) & jnp.isfinite(right_value)):
        return _result(
            point=None,
            status=EventLocalizationStatus.NONFINITE,
            iterations=0,
            corrector_iterations=0,
            bordered_preparations=0,
            bordered_numeric_refreshes=0,
            cached_bordered_column_reuses=0,
            bracket_width=width,
            left_indicator=left_value,
            right_indicator=right_value,
            localized_indicator=jnp.nan,
            residual_norm=jnp.inf,
            linear_solve_status=-1,
            provenance=provenance,
        )
    if not bool(
        (left_value == 0.0) | (right_value == 0.0) | (left_value * right_value < 0.0)
    ):
        return _invalid_result(
            provenance,
            bracket_width=width,
            left_indicator=left_value,
            right_indicator=right_value,
        )
    for endpoint, value in ((left, left_value), (right, right_value)):
        if abs(float(value)) <= policy_.indicator_tolerance:
            return _result(
                point=endpoint,
                status=EventLocalizationStatus.SUCCESS,
                iterations=0,
                corrector_iterations=0,
                bordered_preparations=0,
                bordered_numeric_refreshes=0,
                cached_bordered_column_reuses=0,
                bracket_width=width,
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=value,
                residual_norm=endpoint.residual_norm,
                linear_solve_status=-1,
                provenance=provenance,
            )
    total_corrector_iterations = 0
    total_preparations = 0
    total_refreshes = 0
    total_cached_reuses = 0
    last_point: BranchPoint | None = None
    last_indicator = jnp.asarray(jnp.nan, dtype=left_value.dtype)
    last_residual_norm = jnp.asarray(jnp.inf, dtype=left_value.dtype)
    last_linear_status = jnp.asarray(-1, dtype=jnp.int32)
    bordered_prepared = None
    for iteration in range(1, policy_.maximum_steps + 1):
        denominator = float(right_value - left_value)
        secant_weight = (
            -float(left_value) / denominator
            if denominator != 0.0 and isfinite(denominator)
            else 0.5
        )
        safeguarded_weight = min(
            1.0 - policy_.secant_safeguard,
            max(
                policy_.secant_safeguard,
                secant_weight if isfinite(secant_weight) else 0.5,
            ),
        )
        if abs(float(left_value)) <= abs(float(right_value)):
            weight = (1.0 - policy_.secant_safeguard) * safeguarded_weight
        else:
            weight = (
                policy_.secant_safeguard
                + (1.0 - policy_.secant_safeguard) * safeguarded_weight
            )
        (
            corrected_state,
            corrected_coordinate,
            state_tangent,
            coordinate_tangent,
            last_residual_norm,
            candidate_linear_status,
            corrector_iterations,
            bordered_prepared,
            preparations,
            refreshes,
            cached_reuses,
            corrected_successfully,
            finite_candidate,
        ) = _correct_candidate(
            problem,
            left,
            right,
            weight,
            policy_,
            bordered_prepared,
            args,
        )
        total_corrector_iterations += corrector_iterations
        if int(candidate_linear_status) >= 0:
            last_linear_status = candidate_linear_status
        total_preparations += preparations
        total_refreshes += refreshes
        total_cached_reuses += cached_reuses
        if bordered_prepared is not None:
            provenance = _provenance(
                problem,
                branch,
                bracket,
                policy_,
                indicator_identifier,
                bordered_plan_id=bordered_prepared.plan.plan_id,
                bordered_prepared_id=bordered_prepared.prepared_id,
                bordered_numeric_version=bordered_prepared.numeric_version,
            )
        if not finite_candidate:
            return _result(
                point=None,
                status=EventLocalizationStatus.NONFINITE,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                bordered_preparations=total_preparations,
                bordered_numeric_refreshes=total_refreshes,
                cached_bordered_column_reuses=total_cached_reuses,
                bracket_width=_branch_width(left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=jnp.nan,
                residual_norm=last_residual_norm,
                linear_solve_status=last_linear_status,
                provenance=provenance,
            )
        if not corrected_successfully or not bool(
            problem.contains_coordinate(corrected_coordinate)
        ):
            return _result(
                point=None,
                status=EventLocalizationStatus.CORRECTOR_FAILED,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                bordered_preparations=total_preparations,
                bordered_numeric_refreshes=total_refreshes,
                cached_bordered_column_reuses=total_cached_reuses,
                bracket_width=_branch_width(left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=jnp.nan,
                residual_norm=last_residual_norm,
                linear_solve_status=last_linear_status,
                provenance=provenance,
            )
        physical_parameters, physical_tangent = problem.parameters_jvp(
            corrected_coordinate,
            coordinate_tangent,
            args,
        )
        last_point = BranchPoint(
            state=corrected_state,
            coordinate=corrected_coordinate,
            parameters=physical_parameters,
            tangent_state=state_tangent,
            tangent_coordinate=coordinate_tangent,
            tangent_parameters=physical_tangent,
            residual_norm=last_residual_norm,
            step_size=_branch_width(left, right),
            corrector_iterations=corrector_iterations,
            corrector_retries=0,
            status=ContinuationStatus.SUCCESS,
            tangent_status=LinearSolveStatus.SUCCESS,
            fold_candidate=bracket.kind == "fold-candidate",
            point_id=f"{branch.branch_id}/localized/{bracket.bracket_id}",
            parent_point_id=left.point_id,
        )
        last_indicator = _indicator_value(indicator, problem, last_point, args)
        if not bool(jnp.isfinite(last_indicator)):
            return _result(
                point=None,
                status=EventLocalizationStatus.NONFINITE,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                bordered_preparations=total_preparations,
                bordered_numeric_refreshes=total_refreshes,
                cached_bordered_column_reuses=total_cached_reuses,
                bracket_width=_branch_width(left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=last_indicator,
                residual_norm=last_residual_norm,
                linear_solve_status=last_linear_status,
                provenance=provenance,
            )
        indicator_converged = abs(float(last_indicator)) <= policy_.indicator_tolerance
        if bool((left_value == 0.0) | (left_value * last_indicator < 0.0)):
            right, right_value = last_point, last_indicator
        else:
            left, left_value = last_point, last_indicator
        width = _branch_width(left, right)
        if indicator_converged or float(width) <= policy_.bracket_tolerance:
            return _result(
                point=last_point,
                status=EventLocalizationStatus.SUCCESS,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                bordered_preparations=total_preparations,
                bordered_numeric_refreshes=total_refreshes,
                cached_bordered_column_reuses=total_cached_reuses,
                bracket_width=width,
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=last_indicator,
                residual_norm=last_residual_norm,
                linear_solve_status=last_linear_status,
                provenance=provenance,
            )
    return _result(
        point=last_point,
        status=EventLocalizationStatus.MAXIMUM_STEPS_REACHED,
        iterations=policy_.maximum_steps,
        corrector_iterations=total_corrector_iterations,
        bordered_preparations=total_preparations,
        bordered_numeric_refreshes=total_refreshes,
        cached_bordered_column_reuses=total_cached_reuses,
        bracket_width=_branch_width(left, right),
        left_indicator=left_value,
        right_indicator=right_value,
        localized_indicator=last_indicator,
        residual_norm=last_residual_norm,
        linear_solve_status=last_linear_status,
        provenance=provenance,
    )


__all__ = [
    "EventIndicator",
    "EventLocalizationDiagnostics",
    "EventLocalizationPolicy",
    "EventLocalizationProvenance",
    "EventLocalizationResult",
    "EventLocalizationStatus",
    "localize_event",
]
