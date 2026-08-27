#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from math import isfinite
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled,
    tree_allfinite,
    tree_scale,
)
from ..linalg import (
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    TolerancePolicy,
)
from ..nonlinear import (
    AbstractNonlinearMethod,
    NewtonKrylov,
    NonlinearTermination,
    PreparedNonlinearSolve,
)
from ._core import (
    _execution_residual,
    _run_nonlinear_corrector,
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
    """Safeguarded interpolation and full augmented correction policy."""

    corrector: AbstractNonlinearMethod
    termination: NonlinearTermination
    bracket_tolerance: float = eqx.field(static=True)
    indicator_tolerance: float = eqx.field(static=True)
    maximum_steps: int = eqx.field(static=True)
    secant_safeguard: float = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        corrector: AbstractNonlinearMethod | None = None,
        termination: NonlinearTermination | None = None,
        bracket_tolerance: float = 1e-8,
        indicator_tolerance: float = 1e-8,
        maximum_steps: int = 40,
        secant_safeguard: float = 0.1,
        policy_id: str | None = None,
    ):
        corrector_ = (
            NewtonKrylov(
                linear_policy=LinearSolvePolicy(
                    GMRES(),
                    tolerance=TolerancePolicy(relative=1e-8, absolute=1e-11),
                )
            )
            if corrector is None
            else corrector
        )
        termination_ = (
            NonlinearTermination(
                absolute_residual=1e-9,
                relative_residual=0.0,
                absolute_step=0.0,
                relative_step=0.0,
                maximum_steps=12,
            )
            if termination is None
            else termination
        )
        if not isinstance(corrector_, AbstractNonlinearMethod):
            raise TypeError("corrector must be an AbstractNonlinearMethod or None.")
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination or None.")
        tolerances = tuple(
            float(value)
            for value in (
                bracket_tolerance,
                indicator_tolerance,
            )
        )
        if any(not isfinite(value) or value < 0.0 for value in tolerances):
            raise ValueError(
                "Event localization tolerances must be finite and non-negative."
            )
        steps = int(maximum_steps)
        safeguard = float(secant_safeguard)
        if steps < 1:
            raise ValueError("Event localization maximum_steps must be positive.")
        if not isfinite(safeguard) or not 0.0 < safeguard < 0.5:
            raise ValueError("secant_safeguard must lie strictly between zero and 0.5.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "event-localization-policy-v2",
                    "corrector": corrector_.method_id,
                    "absolute_residual": termination_.absolute_residual,
                    "relative_residual": termination_.relative_residual,
                    "maximum_corrector_steps": termination_.maximum_steps,
                    "bracket_tolerance": tolerances[0],
                    "indicator_tolerance": tolerances[1],
                    "maximum_steps": steps,
                    "secant_safeguard": safeguard,
                }
            )
            if policy_id is None
            else str(policy_id)
        )
        if not identifier:
            raise ValueError("policy_id must be non-empty.")
        self.corrector = corrector_
        self.termination = termination_
        self.bracket_tolerance, self.indicator_tolerance = tolerances
        self.maximum_steps = steps
        self.secant_safeguard = safeguard
        self.policy_id = identifier


class EventLocalizationDiagnostics(StrictModule):
    """Final sign bracket, residual, nonlinear work, and refresh evidence."""

    iterations: Array
    corrector_iterations: Array
    jacobian_preparations: Array
    numeric_refreshes: Array
    linear_solves: Array
    bracket_width: Array
    left_indicator: Array
    right_indicator: Array
    localized_indicator: Array
    residual_norm: Array
    corrector_status: Array

    def __init__(
        self,
        *,
        iterations: Any,
        corrector_iterations: Any,
        jacobian_preparations: Any,
        numeric_refreshes: Any,
        linear_solves: Any,
        bracket_width: Any,
        left_indicator: Any,
        right_indicator: Any,
        localized_indicator: Any,
        residual_norm: Any,
        corrector_status: Any,
    ):
        integer_values = tuple(
            jnp.asarray(value, dtype=jnp.int32)
            for value in (
                iterations,
                corrector_iterations,
                jacobian_preparations,
                numeric_refreshes,
                linear_solves,
                corrector_status,
            )
        )
        (
            self.iterations,
            self.corrector_iterations,
            self.jacobian_preparations,
            self.numeric_refreshes,
            self.linear_solves,
            self.corrector_status,
        ) = integer_values
        self.bracket_width = jnp.asarray(bracket_width)
        self.left_indicator = jnp.asarray(left_indicator)
        self.right_indicator = jnp.asarray(right_indicator)
        self.localized_indicator = jnp.asarray(localized_indicator)
        self.residual_norm = jnp.asarray(residual_norm)


class EventLocalizationProvenance(StrictModule):
    """Stable problem, bracket, policy, corrector, and indicator identities."""

    corrector_numeric_version: Array
    problem_id: str = eqx.field(static=True)
    branch_id: str = eqx.field(static=True)
    bracket_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    corrector_method: str = eqx.field(static=True)
    corrector_plan_id: str = eqx.field(static=True)
    corrector_prepared_id: str = eqx.field(static=True)
    indicator_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        problem_id: str,
        branch_id: str,
        bracket_id: str,
        policy_id: str,
        corrector_method: str,
        corrector_plan_id: str,
        corrector_prepared_id: str,
        corrector_numeric_version: Any,
        indicator_id: str,
    ):
        values = tuple(
            str(value)
            for value in (
                problem_id,
                branch_id,
                bracket_id,
                policy_id,
                corrector_method,
                corrector_plan_id,
                corrector_prepared_id,
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
            self.corrector_method,
            self.corrector_plan_id,
            self.corrector_prepared_id,
            self.indicator_id,
        ) = values
        version = jnp.asarray(corrector_numeric_version, dtype=jnp.int32)
        if version.shape != ():
            raise ValueError("corrector_numeric_version must be scalar.")
        self.corrector_numeric_version = version


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
    branch: ContinuationBranch,
    left: BranchPoint,
    right: BranchPoint,
    /,
) -> tuple[PyTree[Array], Array]:
    left_state = branch.geometry.state_to_execution(left.state)
    right_state = branch.geometry.state_to_execution(right.state)
    return (
        tree_add_scaled(right_state, left_state, -1.0),
        right.coordinate - left.coordinate,
    )


def _branch_width(
    branch: ContinuationBranch,
    left: BranchPoint,
    right: BranchPoint,
    /,
) -> Array:
    state_difference, coordinate_difference = _branch_difference(
        branch,
        left,
        right,
    )
    return branch.geometry.augmented_norm(
        state_difference,
        coordinate_difference,
    )


def _normalized_chord(
    branch: ContinuationBranch,
    left: BranchPoint,
    right: BranchPoint,
    /,
) -> tuple[PyTree[Array], Array]:
    state_difference, coordinate_difference = _branch_difference(
        branch,
        left,
        right,
    )
    norm = _branch_width(branch, left, right)
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
    corrector_plan_id: str = "not-prepared",
    corrector_prepared_id: str = "not-prepared",
    corrector_numeric_version: Any = -1,
) -> EventLocalizationProvenance:
    return EventLocalizationProvenance(
        problem_id=problem.problem_id,
        branch_id=branch.branch_id,
        bracket_id=bracket.bracket_id,
        policy_id=policy.policy_id,
        corrector_method=policy.corrector.method_id,
        corrector_plan_id=corrector_plan_id,
        corrector_prepared_id=corrector_prepared_id,
        corrector_numeric_version=corrector_numeric_version,
        indicator_id=indicator_id,
    )


def _result(
    *,
    point: BranchPoint | None,
    status: EventLocalizationStatus,
    iterations: int,
    corrector_iterations: int,
    jacobian_preparations: int,
    numeric_refreshes: int,
    linear_solves: int,
    bracket_width: Any,
    left_indicator: Any,
    right_indicator: Any,
    localized_indicator: Any,
    residual_norm: Any,
    corrector_status: Any,
    provenance: EventLocalizationProvenance,
) -> EventLocalizationResult:
    return EventLocalizationResult(
        point,
        status,
        EventLocalizationDiagnostics(
            iterations=iterations,
            corrector_iterations=corrector_iterations,
            jacobian_preparations=jacobian_preparations,
            numeric_refreshes=numeric_refreshes,
            linear_solves=linear_solves,
            bracket_width=bracket_width,
            left_indicator=left_indicator,
            right_indicator=right_indicator,
            localized_indicator=localized_indicator,
            residual_norm=residual_norm,
            corrector_status=corrector_status,
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
        jacobian_preparations=0,
        numeric_refreshes=0,
        linear_solves=0,
        bracket_width=bracket_width,
        left_indicator=left_indicator,
        right_indicator=right_indicator,
        localized_indicator=jnp.nan,
        residual_norm=jnp.inf,
        corrector_status=-1,
        provenance=provenance,
    )


def _correct_candidate(
    problem: ContinuationCurveProblem,
    branch: ContinuationBranch,
    left: BranchPoint,
    right: BranchPoint,
    weight: float,
    policy: EventLocalizationPolicy,
    prepared: PreparedNonlinearSolve | None,
    args: Any,
    /,
):
    geometry = branch.geometry
    state_tangent, coordinate_tangent = _normalized_chord(branch, left, right)
    left_state = geometry.state_to_execution(left.state)
    right_state = geometry.state_to_execution(right.state)
    predicted_state = _interpolate_tree(left_state, right_state, weight)
    predicted_coordinate = left.coordinate + weight * (right.coordinate - left.coordinate)

    def augmented_residual(variables):
        state, coordinate = variables
        residual = _execution_residual(
            problem,
            geometry,
            state,
            coordinate,
            args,
        )
        displacement = tree_add_scaled(state, predicted_state, -1.0)
        hyperplane = geometry.augmented_inner(
            displacement,
            coordinate - predicted_coordinate,
            state_tangent,
            coordinate_tangent,
        )
        return residual, hyperplane

    result, retained = _run_nonlinear_corrector(
        augmented_residual,
        (predicted_state, predicted_coordinate),
        policy.corrector,
        policy.termination,
        prepared,
        identity=f"{problem.problem_id}/event-localization-corrector",
    )
    state, coordinate = result.state
    residual = _execution_residual(
        problem,
        geometry,
        state,
        coordinate,
        args,
    )
    residual_norm = geometry.residual_norm(residual)
    finite = bool(
        tree_allfinite(state) & jnp.isfinite(coordinate) & jnp.isfinite(residual_norm)
    )
    threshold = policy.termination.residual_threshold(
        result.diagnostics.initial_residual_norm
    )
    corrected_successfully = bool(
        result.successful & finite & (residual_norm <= threshold)
    )
    return (
        state,
        coordinate,
        state_tangent,
        coordinate_tangent,
        residual_norm,
        result.status,
        int(result.diagnostics.iterations),
        retained,
        int(result.diagnostics.jacobian_preparations),
        int(result.diagnostics.numeric_refreshes),
        int(result.diagnostics.linear_solves),
        corrected_successfully,
        finite,
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
    width = _branch_width(branch, left, right)
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
            jacobian_preparations=0,
            numeric_refreshes=0,
            linear_solves=0,
            bracket_width=width,
            left_indicator=left_value,
            right_indicator=right_value,
            localized_indicator=jnp.nan,
            residual_norm=jnp.inf,
            corrector_status=-1,
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
                jacobian_preparations=0,
                numeric_refreshes=0,
                linear_solves=0,
                bracket_width=width,
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=value,
                residual_norm=endpoint.residual_norm,
                corrector_status=-1,
                provenance=provenance,
            )
    total_corrector_iterations = 0
    total_preparations = 0
    total_refreshes = 0
    total_linear_solves = 0
    last_point: BranchPoint | None = None
    last_indicator = jnp.asarray(jnp.nan, dtype=left_value.dtype)
    last_residual_norm = jnp.asarray(jnp.inf, dtype=left_value.dtype)
    last_corrector_status = jnp.asarray(-1, dtype=jnp.int32)
    corrector_prepared = None
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
            candidate_corrector_status,
            corrector_iterations,
            corrector_prepared,
            preparations,
            refreshes,
            linear_solve_count,
            corrected_successfully,
            finite_candidate,
        ) = _correct_candidate(
            problem,
            branch,
            left,
            right,
            weight,
            policy_,
            corrector_prepared,
            args,
        )
        total_corrector_iterations += corrector_iterations
        if int(candidate_corrector_status) >= 0:
            last_corrector_status = candidate_corrector_status
        total_preparations += preparations
        total_refreshes += refreshes
        total_linear_solves += linear_solve_count
        if corrector_prepared is not None:
            provenance = _provenance(
                problem,
                branch,
                bracket,
                policy_,
                indicator_identifier,
                corrector_plan_id=corrector_prepared.linear_plan_id,
                corrector_prepared_id=f"{policy_.policy_id}/prepared",
                corrector_numeric_version=corrector_prepared.numeric_version,
            )
        if not finite_candidate:
            return _result(
                point=None,
                status=EventLocalizationStatus.NONFINITE,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                jacobian_preparations=total_preparations,
                numeric_refreshes=total_refreshes,
                linear_solves=total_linear_solves,
                bracket_width=_branch_width(branch, left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=jnp.nan,
                residual_norm=last_residual_norm,
                corrector_status=last_corrector_status,
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
                jacobian_preparations=total_preparations,
                numeric_refreshes=total_refreshes,
                linear_solves=total_linear_solves,
                bracket_width=_branch_width(branch, left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=jnp.nan,
                residual_norm=last_residual_norm,
                corrector_status=last_corrector_status,
                provenance=provenance,
            )
        public_state = branch.geometry.state_from_execution(corrected_state)
        public_state_tangent = branch.geometry.state_tangent_from_execution(
            corrected_state,
            state_tangent,
        )
        physical_parameters, physical_tangent = problem.parameters_jvp(
            corrected_coordinate,
            coordinate_tangent,
            args,
        )
        last_point = BranchPoint(
            state=public_state,
            coordinate=corrected_coordinate,
            parameters=physical_parameters,
            tangent_state=public_state_tangent,
            tangent_coordinate=coordinate_tangent,
            tangent_parameters=physical_tangent,
            residual_norm=last_residual_norm,
            step_size=_branch_width(branch, left, right),
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
                jacobian_preparations=total_preparations,
                numeric_refreshes=total_refreshes,
                linear_solves=total_linear_solves,
                bracket_width=_branch_width(branch, left, right),
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=last_indicator,
                residual_norm=last_residual_norm,
                corrector_status=last_corrector_status,
                provenance=provenance,
            )
        indicator_converged = abs(float(last_indicator)) <= policy_.indicator_tolerance
        if bool((left_value == 0.0) | (left_value * last_indicator < 0.0)):
            right, right_value = last_point, last_indicator
        else:
            left, left_value = last_point, last_indicator
        width = _branch_width(branch, left, right)
        if indicator_converged or float(width) <= policy_.bracket_tolerance:
            return _result(
                point=last_point,
                status=EventLocalizationStatus.SUCCESS,
                iterations=iteration,
                corrector_iterations=total_corrector_iterations,
                jacobian_preparations=total_preparations,
                numeric_refreshes=total_refreshes,
                linear_solves=total_linear_solves,
                bracket_width=width,
                left_indicator=left_value,
                right_indicator=right_value,
                localized_indicator=last_indicator,
                residual_norm=last_residual_norm,
                corrector_status=last_corrector_status,
                provenance=provenance,
            )
    return _result(
        point=last_point,
        status=EventLocalizationStatus.MAXIMUM_STEPS_REACHED,
        iterations=policy_.maximum_steps,
        corrector_iterations=total_corrector_iterations,
        jacobian_preparations=total_preparations,
        numeric_refreshes=total_refreshes,
        linear_solves=total_linear_solves,
        bracket_width=_branch_width(branch, left, right),
        left_indicator=left_value,
        right_indicator=right_value,
        localized_indicator=last_indicator,
        residual_norm=last_residual_norm,
        corrector_status=last_corrector_status,
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
