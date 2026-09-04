#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-population continuation of mean-field-game candidates."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import IntEnum
from math import isfinite
from typing import Any, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import EmpiricalMeanField
from ._mean_field_fixed_point import (
    MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE,
    MeanFieldGameFixedPointResult,
    MeanFieldGameFixedPointStatus,
)


FINITE_POPULATION_EPSILON_NASH_EVIDENCE = "FINITE_POPULATION_EPSILON_NASH_EVIDENCE"
FINITE_POPULATION_CONTINUATION_EVALUATION = "FINITE_POPULATION_CONTINUATION_EVALUATION"
CoverageMethod: TypeAlias = Literal["exact", "asymptotic-normal", "hoeffding", "none"]


class FinitePopulationContinuationStatus(IntEnum):
    """Stable outcomes for finite-population continuation."""

    SUCCESS = 0
    INVALID_MFG_FIXED_POINT = 1
    INVALID_JOINT_PROFILE = 2
    INVALID_FINITE_LAW = 3
    MISSING_BEST_RESPONSE = 4
    FAILED_BEST_RESPONSE = 5
    INCOMPLETE_BOUNDS = 6
    INVALID_LAW_DISTANCE = 7
    LAW_MISMATCH = 8
    EPSILON_EXCEEDED = 9


class FinitePopulationJointPolicyEvaluation(StrictModule):
    """Typed pathwise costs and dependence provenance for one finite-N profile.

    Player costs have shape ``(population_size, num_paths)``.  Paths sharing a
    cluster label may be dependent; different labels are the declared independent
    units.  Player values are therefore averages of cluster averages rather than
    path-weighted averages.  ``statistically_exact`` is reserved for deterministic
    enumeration or an otherwise exact expectation, not a zero sample variance.
    """

    player_costs: Array
    path_valid: Array
    cluster_labels: Array
    population_size: int = eqx.field(static=True)
    num_paths: int = eqx.field(static=True)
    path_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str = eqx.field(static=True)
    policy_ids: tuple[str, ...] = eqx.field(static=True)
    evaluation_id: str = eqx.field(static=True)
    statistically_exact: bool = eqx.field(static=True)

    def __init__(
        self,
        player_costs: ArrayLike,
        /,
        *,
        path_ids: Sequence[str],
        cluster_labels: ArrayLike,
        coupling_id: str,
        policy_ids: Sequence[str],
        evaluation_id: str,
        path_valid: ArrayLike | None = None,
        statistically_exact: bool = False,
    ):
        costs = _real_matrix(player_costs, "player_costs")
        population_size, num_paths = map(int, costs.shape)
        if population_size < 1 or num_paths < 1:
            raise ValueError("player_costs must contain players and paths.")
        identifiers = _identifiers(path_ids, "path_ids")
        if len(identifiers) != num_paths or len(set(identifiers)) != num_paths:
            raise ValueError("path_ids must uniquely identify every path.")
        policies = _identifiers(policy_ids, "policy_ids")
        if len(policies) != population_size:
            raise ValueError("policy_ids must contain one ID per player.")
        labels = _integer_labels(cluster_labels, num_paths)
        validity = (
            jnp.ones((num_paths,), dtype=bool)
            if path_valid is None
            else jnp.asarray(path_valid, dtype=bool)
        )
        if validity.shape != (num_paths,):
            raise ValueError("path_valid must contain one flag per path.")
        validity = validity & jnp.all(jnp.isfinite(costs), axis=0)
        self.player_costs = costs
        self.path_valid = validity
        self.cluster_labels = labels
        self.population_size = population_size
        self.num_paths = num_paths
        self.path_ids = identifiers
        self.coupling_id = _identifier(coupling_id, "coupling_id")
        self.policy_ids = policies
        self.evaluation_id = _identifier(evaluation_id, "evaluation_id")
        self.statistically_exact = bool(statistically_exact)

    @property
    def independence_labels(self) -> Array:
        return self.cluster_labels

    @property
    def costs(self) -> Array:
        return self.player_costs


class FinitePopulationBestResponseEvidence(StrictModule):
    """Certified lower player cost and its additive uncertainty bounds.

    ``best_response_value`` is the lower cost found over the declared feasible
    unilateral deviations.  For a minimizer, numerical and statistical errors
    enlarge the exploitability upper bound.  They are never subtracted.  The
    statistical bound must already be simultaneous across all players; this
    explicit declaration prevents pointwise intervals from being relabeled as a
    simultaneous Nash statement.
    """

    best_response_value: Array
    numerical_error_bound: Array
    statistical_error_bound: Array
    valid: Array
    certified: Array
    independent_cluster_count: Array
    player_index: int = eqx.field(static=True)
    best_response_id: str = eqx.field(static=True)
    feasible_deviation_id: str = eqx.field(static=True)
    deviation_policy_id: str = eqx.field(static=True)
    coverage_method: CoverageMethod = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    simultaneous: bool = eqx.field(static=True)
    failure_reason: str | None = eqx.field(static=True)

    def __init__(
        self,
        best_response_value: ArrayLike,
        /,
        *,
        player_index: int,
        numerical_error_bound: ArrayLike | None,
        statistical_error_bound: ArrayLike | None,
        best_response_id: str,
        feasible_deviation_id: str,
        deviation_policy_id: str,
        coverage_method: CoverageMethod,
        confidence: float,
        independent_cluster_count: ArrayLike,
        simultaneous: bool,
        valid: ArrayLike = True,
        certified: ArrayLike = True,
        failure_reason: str | None = None,
    ):
        if (
            isinstance(player_index, bool)
            or not isinstance(player_index, int)
            or player_index < 0
        ):
            raise ValueError("player_index must be a nonnegative integer.")
        value = _real_scalar(best_response_value, "best_response_value")
        numerical = _optional_bound(numerical_error_bound, value.dtype)
        statistical = _optional_bound(statistical_error_bound, value.dtype)
        validity = jnp.asarray(valid, dtype=bool)
        certification = jnp.asarray(certified, dtype=bool)
        clusters = jnp.asarray(independent_cluster_count)
        if validity.shape != () or certification.shape != () or clusters.shape != ():
            raise ValueError(
                "valid, certified, and independent_cluster_count must be scalar."
            )
        if not jnp.issubdtype(clusters.dtype, jnp.integer) or int(clusters) < 0:
            raise ValueError("independent_cluster_count must be a nonnegative integer.")
        level = float(confidence)
        if not isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("confidence must be finite and lie in (0, 1).")
        if failure_reason is not None and (
            not isinstance(failure_reason, str) or not failure_reason
        ):
            raise ValueError("failure_reason must be non-empty when supplied.")
        self.best_response_value = value
        self.numerical_error_bound = numerical
        self.statistical_error_bound = statistical
        self.valid = validity
        self.certified = certification
        self.independent_cluster_count = clusters.astype(jnp.int32)
        self.player_index = player_index
        self.best_response_id = _identifier(best_response_id, "best_response_id")
        self.feasible_deviation_id = _identifier(
            feasible_deviation_id, "feasible_deviation_id"
        )
        self.deviation_policy_id = _identifier(deviation_policy_id, "deviation_policy_id")
        self.coverage_method = _coverage_method(coverage_method)
        self.confidence = level
        self.simultaneous = bool(simultaneous)
        self.failure_reason = failure_reason

    @property
    def lower_value(self) -> Array:
        return self.best_response_value

    @property
    def lower_player_cost(self) -> Array:
        return self.best_response_value

    @property
    def numerical_bound_available(self) -> Array:
        return jnp.isfinite(self.numerical_error_bound)

    @property
    def statistical_bound_available(self) -> Array:
        return jnp.isfinite(self.statistical_error_bound)


class FinitePopulationGameProblem(StrictModule):
    """Finite-N callbacks and provenance anchored to one MFG fixed point."""

    fixed_point_result: MeanFieldGameFixedPointResult
    population_size: int = eqx.field(static=True)
    policy_ids: tuple[str, ...] = eqx.field(static=True)
    joint_profile_evaluator: Callable[[MeanFieldGameFixedPointResult, Any], Any] = (
        eqx.field(static=True)
    )
    best_response_callbacks: tuple[
        Callable[[FinitePopulationJointPolicyEvaluation, Any], Any] | None, ...
    ] = eqx.field(static=True)
    finite_law_builder: Callable[[FinitePopulationJointPolicyEvaluation, Any], Any] = (
        eqx.field(static=True)
    )
    law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any], ArrayLike] = (
        eqx.field(static=True)
    )
    joint_profile_evaluator_id: str = eqx.field(static=True)
    best_response_ids: tuple[str, ...] = eqx.field(static=True)
    feasible_deviation_ids: tuple[str, ...] = eqx.field(static=True)
    finite_law_builder_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        fixed_point_result: MeanFieldGameFixedPointResult,
        population_size: int,
        policy_ids: Sequence[str],
        joint_profile_evaluator: Callable[[MeanFieldGameFixedPointResult, Any], Any],
        best_response_callbacks: Sequence[
            Callable[[FinitePopulationJointPolicyEvaluation, Any], Any] | None
        ],
        finite_law_builder: Callable[[FinitePopulationJointPolicyEvaluation, Any], Any],
        law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any], ArrayLike],
        /,
        *,
        joint_profile_evaluator_id: str,
        best_response_ids: Sequence[str],
        feasible_deviation_ids: Sequence[str],
        finite_law_builder_id: str,
        law_distance_id: str,
        problem_id: str,
    ):
        if not isinstance(fixed_point_result, MeanFieldGameFixedPointResult):
            raise TypeError("fixed_point_result must be a MeanFieldGameFixedPointResult.")
        if (
            isinstance(population_size, bool)
            or not isinstance(population_size, int)
            or population_size <= 0
        ):
            raise ValueError("population_size must be a positive integer.")
        policies = _identifiers(policy_ids, "policy_ids")
        callbacks = tuple(best_response_callbacks)
        response_ids = _identifiers(best_response_ids, "best_response_ids")
        deviation_ids = _identifiers(feasible_deviation_ids, "feasible_deviation_ids")
        for owner, sequence in (
            ("policy_ids", policies),
            ("best_response_callbacks", callbacks),
            ("best_response_ids", response_ids),
            ("feasible_deviation_ids", deviation_ids),
        ):
            if len(sequence) != population_size:
                raise ValueError(f"{owner} must contain one entry per player.")
        if any(callback is not None and not callable(callback) for callback in callbacks):
            raise TypeError("best_response_callbacks entries must be callable or None.")
        for owner, callback in (
            ("joint_profile_evaluator", joint_profile_evaluator),
            ("finite_law_builder", finite_law_builder),
            ("law_distance", law_distance),
        ):
            if not callable(callback):
                raise TypeError(f"{owner} must be callable.")
        self.fixed_point_result = fixed_point_result
        self.population_size = population_size
        self.policy_ids = policies
        self.joint_profile_evaluator = joint_profile_evaluator
        self.best_response_callbacks = callbacks
        self.finite_law_builder = finite_law_builder
        self.law_distance = law_distance
        self.joint_profile_evaluator_id = _identifier(
            joint_profile_evaluator_id, "joint_profile_evaluator_id"
        )
        self.best_response_ids = response_ids
        self.feasible_deviation_ids = deviation_ids
        self.finite_law_builder_id = _identifier(
            finite_law_builder_id, "finite_law_builder_id"
        )
        self.law_distance_id = _identifier(law_distance_id, "law_distance_id")
        self.problem_id = _identifier(problem_id, "problem_id")

    @property
    def mfg_result(self) -> MeanFieldGameFixedPointResult:
        return self.fixed_point_result

    @property
    def num_players(self) -> int:
        return self.population_size


class FinitePopulationContinuationPlan(StrictModule):
    """Numerical/statistical acceptance policy, separate from the finite game."""

    epsilon: float = eqx.field(static=True)
    law_tolerance: float = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    coverage_method: CoverageMethod = eqx.field(static=True)
    minimum_clusters: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        epsilon: float,
        law_tolerance: float,
        confidence: float,
        coverage_method: CoverageMethod,
        minimum_clusters: int,
        problem_id: str,
    ):
        epsilon_value = _nonnegative_finite(epsilon, "epsilon")
        tolerance = _nonnegative_finite(law_tolerance, "law_tolerance")
        level = float(confidence)
        if not isfinite(level) or not 0.0 < level < 1.0:
            raise ValueError("confidence must be finite and lie in (0, 1).")
        if (
            isinstance(minimum_clusters, bool)
            or not isinstance(minimum_clusters, int)
            or minimum_clusters <= 0
        ):
            raise ValueError("minimum_clusters must be a positive integer.")
        method = _coverage_method(coverage_method)
        identifier = _identifier(problem_id, "problem_id")
        self.epsilon = epsilon_value
        self.law_tolerance = tolerance
        self.confidence = level
        self.coverage_method = method
        self.minimum_clusters = minimum_clusters
        self.problem_id = identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-population-continuation-plan",
                "problem": identifier,
                "epsilon": epsilon_value,
                "law_tolerance": tolerance,
                "confidence": level,
                "coverage_method": method,
                "minimum_clusters": minimum_clusters,
            }
        )


class FinitePopulationContinuationResult(StrictModule):
    """Finite-N deviation bounds and MFG-law comparison with full provenance."""

    profile_values: Array
    best_response_values: Array
    raw_improvements: Array
    exploitabilities: Array
    numerical_error_bounds: Array
    statistical_error_bounds: Array
    exploitability_upper_bounds: Array
    epsilon_upper_bound: Array
    best_response_valid: Array
    bound_available: Array
    independent_cluster_counts: Array
    law_distance: Array
    mfg_valid: Array
    joint_profile_valid: Array
    finite_law_valid: Array
    all_best_responses_valid: Array
    all_bounds_available: Array
    law_matches: Array
    epsilon_satisfied: Array
    valid: Array
    status: Array
    joint_evaluation: FinitePopulationJointPolicyEvaluation | None
    finite_law: EmpiricalMeanField | None
    population_size: int = eqx.field(static=True)
    epsilon: float = eqx.field(static=True)
    confidence: float = eqx.field(static=True)
    coverage_method: CoverageMethod = eqx.field(static=True)
    minimum_clusters: int = eqx.field(static=True)
    path_ids: tuple[str, ...] = eqx.field(static=True)
    coupling_id: str | None = eqx.field(static=True)
    policy_ids: tuple[str, ...] = eqx.field(static=True)
    deviation_policy_ids: tuple[str | None, ...] = eqx.field(static=True)
    mfg_law_id: str = eqx.field(static=True)
    finite_law_id: str | None = eqx.field(static=True)
    mfg_source_path_id: str | None = eqx.field(static=True)
    finite_source_path_id: str | None = eqx.field(static=True)
    mfg_problem_id: str = eqx.field(static=True)
    mfg_plan_id: str = eqx.field(static=True)
    joint_profile_evaluator_id: str = eqx.field(static=True)
    best_response_ids: tuple[str, ...] = eqx.field(static=True)
    feasible_deviation_ids: tuple[str, ...] = eqx.field(static=True)
    finite_law_builder_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)
    epsilon_nash_claimed: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    common_noise_equilibrium_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def exploitability(self) -> Array:
        return self.epsilon_upper_bound

    @property
    def finite_population_epsilon_nash_evidence(self) -> bool:
        return self.epsilon_nash_claimed

    @property
    def exploitability_bound(self) -> Array:
        return self.epsilon_upper_bound

    @property
    def finite_population_law_id(self) -> str | None:
        return self.finite_law_id

    @property
    def finite_population_source_path_id(self) -> str | None:
        return self.finite_source_path_id

    @property
    def num_players(self) -> int:
        return self.population_size


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _identifiers(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(values)
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError(f"{owner} must contain non-empty strings.")
    return result


def _nonnegative_finite(value: float, owner: str, /) -> float:
    result = float(value)
    if not isfinite(result) or result < 0.0:
        raise ValueError(f"{owner} must be finite and nonnegative.")
    return result


def _real_scalar(value: ArrayLike, owner: str, /) -> Array:
    result = jnp.asarray(value)
    if (
        result.shape != ()
        or not jnp.issubdtype(result.dtype, jnp.number)
        or jnp.issubdtype(result.dtype, jnp.complexfloating)
    ):
        raise ValueError(f"{owner} must be a real numeric scalar.")
    return result if jnp.issubdtype(result.dtype, jnp.inexact) else result.astype(float)


def _real_matrix(value: ArrayLike, owner: str, /) -> Array:
    result = jnp.asarray(value)
    if (
        result.ndim != 2
        or not jnp.issubdtype(result.dtype, jnp.number)
        or jnp.issubdtype(result.dtype, jnp.complexfloating)
    ):
        raise ValueError(f"{owner} must be a two-dimensional real numeric array.")
    return result if jnp.issubdtype(result.dtype, jnp.inexact) else result.astype(float)


def _optional_bound(value: ArrayLike | None, dtype: Any, /) -> Array:
    if value is None:
        return jnp.asarray(jnp.nan, dtype=dtype)
    result = jnp.asarray(value, dtype=dtype)
    if result.shape != ():
        raise ValueError("error bounds must be scalar.")
    if bool(jnp.isfinite(result)) and float(result) < 0.0:
        raise ValueError("error bounds must be nonnegative.")
    return result


def _coverage_method(value: str, /) -> CoverageMethod:
    if value in ("exact", "asymptotic-normal", "hoeffding", "none"):
        return value  # type: ignore[return-value]
    raise ValueError(
        "coverage_method must be 'exact', 'asymptotic-normal', 'hoeffding', or 'none'."
    )


def _integer_labels(value: ArrayLike, count: int, /) -> Array:
    raw = np.asarray(value)
    if raw.shape != (count,):
        raise ValueError("cluster_labels must contain one label per path.")
    integer_by_label: dict[Any, int] = {}
    labels = []
    for raw_label in raw.tolist():
        label = raw_label.item() if isinstance(raw_label, np.generic) else raw_label
        if label not in integer_by_label:
            integer_by_label[label] = len(integer_by_label)
        labels.append(integer_by_label[label])
    return jnp.asarray(labels, dtype=jnp.int32)


def _validated_mfg(result: MeanFieldGameFixedPointResult, /) -> bool:
    response = result.best_response_result
    return (
        bool(result.valid)
        and bool(result.converged)
        and int(result.status) == int(MeanFieldGameFixedPointStatus.SUCCESS)
        and result.certificate_label == MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE
        and bool(result.law_consistency_evaluated)
        and response is not None
        and bool(response.valid)
        and isinstance(result.flow, EmpiricalMeanField)
    )


def _cluster_means(evaluation: FinitePopulationJointPolicyEvaluation, /) -> Array:
    labels = np.asarray(evaluation.cluster_labels)
    return jnp.stack(
        [
            jnp.mean(
                evaluation.player_costs[
                    :, jnp.asarray(np.nonzero(labels == label)[0], dtype=jnp.int32)
                ],
                axis=1,
            )
            for label in np.unique(labels)
        ],
        axis=1,
    )


def _compatible_laws(left: EmpiricalMeanField, right: EmpiricalMeanField, /) -> bool:
    return (
        left.state_shape == right.state_shape
        and left.times.shape == right.times.shape
        and bool(jnp.all(left.times == right.times))
    )


def evaluate_finite_population_continuation(
    problem: FinitePopulationGameProblem,
    plan: FinitePopulationContinuationPlan,
    /,
    *,
    args: Any = None,
) -> FinitePopulationContinuationResult:
    """Build conservative finite-N exploitability evidence.

    The minimizer sign is ``max(J(profile) - J(best response), 0)``.  The
    numerical and simultaneous statistical errors are then added per player.
    The epsilon-Nash label is emitted only if all such bounds, the finite-law
    comparison, and every provenance check are complete.
    """

    if not isinstance(problem, FinitePopulationGameProblem):
        raise TypeError("problem must be a FinitePopulationGameProblem.")
    if not isinstance(plan, FinitePopulationContinuationPlan):
        raise TypeError("plan must be a FinitePopulationContinuationPlan.")
    if plan.problem_id != problem.problem_id:
        raise ValueError("plan and problem IDs must match.")

    size = problem.population_size
    dtype = jnp.result_type(problem.fixed_point_result.flow.particles, float)
    nan_players = jnp.full((size,), jnp.nan, dtype=dtype)
    false_players = jnp.zeros((size,), dtype=bool)
    zero_counts = jnp.zeros((size,), dtype=jnp.int32)
    mfg_valid = _validated_mfg(problem.fixed_point_result)
    joint: FinitePopulationJointPolicyEvaluation | None = None
    if mfg_valid:
        candidate = problem.joint_profile_evaluator(problem.fixed_point_result, args)
        if isinstance(candidate, FinitePopulationJointPolicyEvaluation):
            if (
                candidate.population_size == size
                and candidate.policy_ids == problem.policy_ids
                and candidate.evaluation_id == problem.joint_profile_evaluator_id
                and bool(jnp.all(candidate.path_valid))
            ):
                joint = candidate
    if joint is None:
        status = (
            FinitePopulationContinuationStatus.INVALID_JOINT_PROFILE
            if mfg_valid
            else FinitePopulationContinuationStatus.INVALID_MFG_FIXED_POINT
        )
        return _make_result(
            problem,
            plan,
            joint=None,
            finite_law=None,
            profile_values=nan_players,
            best_response_values=nan_players,
            raw_improvements=nan_players,
            exploitabilities=nan_players,
            numerical_errors=nan_players,
            statistical_errors=nan_players,
            upper_bounds=nan_players,
            epsilon_upper=jnp.asarray(jnp.nan, dtype=dtype),
            response_valid=false_players,
            bound_available=false_players,
            cluster_counts=zero_counts,
            law_distance=jnp.asarray(jnp.nan, dtype=dtype),
            mfg_valid=mfg_valid,
            joint_valid=False,
            finite_law_valid=False,
            all_responses=False,
            all_bounds=False,
            law_matches=False,
            epsilon_satisfied=False,
            status=status,
            deviation_policy_ids=(None,) * size,
        )

    finite_candidate = problem.finite_law_builder(joint, args)
    finite_law = (
        finite_candidate if isinstance(finite_candidate, EmpiricalMeanField) else None
    )
    mfg_law = problem.fixed_point_result.flow
    finite_law_valid = (
        finite_law is not None
        and _compatible_laws(finite_law, mfg_law)
        and finite_law.mean_field_id != mfg_law.mean_field_id
        and finite_law.source_path_id is not None
        and finite_law.source_path_id != mfg_law.source_path_id
    )
    if not finite_law_valid:
        return _make_result(
            problem,
            plan,
            joint=joint,
            finite_law=finite_law,
            profile_values=jnp.mean(_cluster_means(joint), axis=1),
            best_response_values=nan_players,
            raw_improvements=nan_players,
            exploitabilities=nan_players,
            numerical_errors=nan_players,
            statistical_errors=nan_players,
            upper_bounds=nan_players,
            epsilon_upper=jnp.asarray(jnp.nan, dtype=dtype),
            response_valid=false_players,
            bound_available=false_players,
            cluster_counts=zero_counts,
            law_distance=jnp.asarray(jnp.nan, dtype=dtype),
            mfg_valid=True,
            joint_valid=True,
            finite_law_valid=False,
            all_responses=False,
            all_bounds=False,
            law_matches=False,
            epsilon_satisfied=False,
            status=FinitePopulationContinuationStatus.INVALID_FINITE_LAW,
            deviation_policy_ids=(None,) * size,
        )

    profile_values = jnp.mean(_cluster_means(joint), axis=1)
    best_values = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    raw_improvements = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    exploitabilities = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    numerical_errors = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    statistical_errors = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    upper_bounds = jnp.full((size,), jnp.nan, dtype=profile_values.dtype)
    response_valid = jnp.zeros((size,), dtype=bool)
    bound_available = jnp.zeros((size,), dtype=bool)
    cluster_counts = jnp.zeros((size,), dtype=jnp.int32)
    deviation_policy_ids: list[str | None] = [None] * size
    missing = False
    failed = False
    for player, callback in enumerate(problem.best_response_callbacks):
        if callback is None:
            missing = True
            continue
        evidence = callback(joint, args)
        if evidence is None:
            missing = True
            continue
        if not isinstance(evidence, FinitePopulationBestResponseEvidence):
            failed = True
            continue
        identity_valid = (
            evidence.player_index == player
            and evidence.best_response_id == problem.best_response_ids[player]
            and evidence.feasible_deviation_id == problem.feasible_deviation_ids[player]
        )
        response_is_valid = (
            bool(evidence.valid)
            and bool(evidence.certified)
            and identity_valid
            and bool(jnp.isfinite(evidence.best_response_value))
        )
        response_valid = response_valid.at[player].set(response_is_valid)
        cluster_counts = cluster_counts.at[player].set(evidence.independent_cluster_count)
        deviation_policy_ids[player] = evidence.deviation_policy_id
        if not response_is_valid:
            failed = True
            continue
        numerical_available = bool(jnp.isfinite(evidence.numerical_error_bound))
        statistical_available = (
            bool(jnp.isfinite(evidence.statistical_error_bound))
            and evidence.coverage_method == plan.coverage_method
            and evidence.confidence >= plan.confidence
            and int(evidence.independent_cluster_count) >= plan.minimum_clusters
            and evidence.simultaneous
            and (plan.coverage_method != "exact" or joint.statistically_exact)
            and plan.coverage_method != "none"
        )
        available = numerical_available and statistical_available
        raw = profile_values[player] - evidence.best_response_value
        exploitability = jnp.maximum(raw, 0.0)
        upper = (
            exploitability
            + evidence.numerical_error_bound
            + evidence.statistical_error_bound
        )
        best_values = best_values.at[player].set(evidence.best_response_value)
        raw_improvements = raw_improvements.at[player].set(raw)
        exploitabilities = exploitabilities.at[player].set(exploitability)
        numerical_errors = numerical_errors.at[player].set(evidence.numerical_error_bound)
        statistical_errors = statistical_errors.at[player].set(
            evidence.statistical_error_bound
        )
        bound_available = bound_available.at[player].set(available)
        upper_bounds = upper_bounds.at[player].set(
            jnp.where(available, upper, jnp.asarray(jnp.nan, dtype=upper.dtype))
        )

    all_responses = bool(jnp.all(response_valid))
    all_bounds = all_responses and bool(jnp.all(bound_available))
    law_value = jnp.asarray(jnp.nan, dtype=profile_values.dtype)
    law_distance_valid = False
    candidate_distance = jnp.asarray(
        problem.law_distance(finite_law, mfg_law, args), dtype=profile_values.dtype
    )
    if (
        candidate_distance.shape == ()
        and bool(jnp.isfinite(candidate_distance))
        and float(candidate_distance) >= 0.0
    ):
        law_value = candidate_distance
        law_distance_valid = True
    law_matches = law_distance_valid and float(law_value) <= plan.law_tolerance
    epsilon_upper = (
        jnp.max(upper_bounds)
        if all_bounds
        else jnp.asarray(jnp.nan, dtype=profile_values.dtype)
    )
    epsilon_satisfied = all_bounds and float(epsilon_upper) <= plan.epsilon
    if missing:
        status = FinitePopulationContinuationStatus.MISSING_BEST_RESPONSE
    elif failed or not all_responses:
        status = FinitePopulationContinuationStatus.FAILED_BEST_RESPONSE
    elif not all_bounds:
        status = FinitePopulationContinuationStatus.INCOMPLETE_BOUNDS
    elif not law_distance_valid:
        status = FinitePopulationContinuationStatus.INVALID_LAW_DISTANCE
    elif not law_matches:
        status = FinitePopulationContinuationStatus.LAW_MISMATCH
    elif not epsilon_satisfied:
        status = FinitePopulationContinuationStatus.EPSILON_EXCEEDED
    else:
        status = FinitePopulationContinuationStatus.SUCCESS
    return _make_result(
        problem,
        plan,
        joint=joint,
        finite_law=finite_law,
        profile_values=profile_values,
        best_response_values=best_values,
        raw_improvements=raw_improvements,
        exploitabilities=exploitabilities,
        numerical_errors=numerical_errors,
        statistical_errors=statistical_errors,
        upper_bounds=upper_bounds,
        epsilon_upper=epsilon_upper,
        response_valid=response_valid,
        bound_available=bound_available,
        cluster_counts=cluster_counts,
        law_distance=law_value,
        mfg_valid=True,
        joint_valid=True,
        finite_law_valid=True,
        all_responses=all_responses,
        all_bounds=all_bounds,
        law_matches=law_matches,
        epsilon_satisfied=epsilon_satisfied,
        status=status,
        deviation_policy_ids=tuple(deviation_policy_ids),
    )


def _make_result(
    problem: FinitePopulationGameProblem,
    plan: FinitePopulationContinuationPlan,
    /,
    *,
    joint: FinitePopulationJointPolicyEvaluation | None,
    finite_law: EmpiricalMeanField | None,
    profile_values: Array,
    best_response_values: Array,
    raw_improvements: Array,
    exploitabilities: Array,
    numerical_errors: Array,
    statistical_errors: Array,
    upper_bounds: Array,
    epsilon_upper: Array,
    response_valid: Array,
    bound_available: Array,
    cluster_counts: Array,
    law_distance: Array,
    mfg_valid: bool,
    joint_valid: bool,
    finite_law_valid: bool,
    all_responses: bool,
    all_bounds: bool,
    law_matches: bool,
    epsilon_satisfied: bool,
    status: FinitePopulationContinuationStatus,
    deviation_policy_ids: tuple[str | None, ...],
) -> FinitePopulationContinuationResult:
    successful = status == FinitePopulationContinuationStatus.SUCCESS
    mfg_law = problem.fixed_point_result.flow
    return FinitePopulationContinuationResult(
        profile_values=profile_values,
        best_response_values=best_response_values,
        raw_improvements=raw_improvements,
        exploitabilities=exploitabilities,
        numerical_error_bounds=numerical_errors,
        statistical_error_bounds=statistical_errors,
        exploitability_upper_bounds=upper_bounds,
        epsilon_upper_bound=epsilon_upper,
        best_response_valid=response_valid,
        bound_available=bound_available,
        independent_cluster_counts=cluster_counts,
        law_distance=law_distance,
        mfg_valid=jnp.asarray(mfg_valid),
        joint_profile_valid=jnp.asarray(joint_valid),
        finite_law_valid=jnp.asarray(finite_law_valid),
        all_best_responses_valid=jnp.asarray(all_responses),
        all_bounds_available=jnp.asarray(all_bounds),
        law_matches=jnp.asarray(law_matches),
        epsilon_satisfied=jnp.asarray(epsilon_satisfied),
        valid=jnp.asarray(successful),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        joint_evaluation=joint,
        finite_law=finite_law,
        population_size=problem.population_size,
        epsilon=plan.epsilon,
        confidence=plan.confidence,
        coverage_method=plan.coverage_method,
        minimum_clusters=plan.minimum_clusters,
        path_ids=() if joint is None else joint.path_ids,
        coupling_id=None if joint is None else joint.coupling_id,
        policy_ids=problem.policy_ids,
        deviation_policy_ids=deviation_policy_ids,
        mfg_law_id=mfg_law.mean_field_id,
        finite_law_id=None if finite_law is None else finite_law.mean_field_id,
        mfg_source_path_id=mfg_law.source_path_id,
        finite_source_path_id=(None if finite_law is None else finite_law.source_path_id),
        mfg_problem_id=problem.fixed_point_result.problem_id,
        mfg_plan_id=problem.fixed_point_result.plan_id,
        joint_profile_evaluator_id=problem.joint_profile_evaluator_id,
        best_response_ids=problem.best_response_ids,
        feasible_deviation_ids=problem.feasible_deviation_ids,
        finite_law_builder_id=problem.finite_law_builder_id,
        law_distance_id=problem.law_distance_id,
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        certificate_label=(
            FINITE_POPULATION_EPSILON_NASH_EVIDENCE
            if successful
            else FINITE_POPULATION_CONTINUATION_EVALUATION
        ),
        finite_population_game_claimed=successful,
        epsilon_nash_claimed=successful,
        mean_field_game_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        common_noise_equilibrium_claimed=False,
    )


__all__ = [
    "FINITE_POPULATION_CONTINUATION_EVALUATION",
    "FINITE_POPULATION_EPSILON_NASH_EVIDENCE",
    "CoverageMethod",
    "FinitePopulationBestResponseEvidence",
    "FinitePopulationContinuationPlan",
    "FinitePopulationContinuationResult",
    "FinitePopulationContinuationStatus",
    "FinitePopulationGameProblem",
    "FinitePopulationJointPolicyEvaluation",
    "evaluate_finite_population_continuation",
]
