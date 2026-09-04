#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Finite-scenario conditional-law iteration for common-noise mean-field games."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite, prod
from typing import Any, Callable, Hashable, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import EmpiricalMeanField
from ._mean_field import (
    FROZEN_LAW_BEST_RESPONSE,
    FrozenLawBestResponseResult,
    FrozenLawBestResponseStatus,
)


COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE = "COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE"


class CommonNoiseMeanFieldStatus(IntEnum):
    """Stable termination codes for conditional-law fixed-point evaluation."""

    SUCCESS = 0
    INVALID_BEST_RESPONSE = 1
    INVALID_INDUCED_LAW = 2
    INSUFFICIENT_INDEPENDENT_CLUSTERS = 3
    LOW_EFFECTIVE_SAMPLE_SIZE = 4
    NONFINITE_LAW_DISTANCE = 5
    INVALID_LAW_DISTANCE = 6
    MAX_ITERATIONS = 7
    ZERO_PROBABILITY_SCENARIO = 8
    INVALID_LAW_MIXTURE = 9


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


def _identifiers(values: Sequence[str], owner: str, /) -> tuple[str, ...]:
    result = tuple(_identifier(value, owner) for value in values)
    if not result:
        raise ValueError(f"{owner} must not be empty.")
    if len(set(result)) != len(result):
        raise ValueError(f"{owner} must be unique.")
    return result


def _compatible_flows(left: EmpiricalMeanField, right: EmpiricalMeanField) -> bool:
    return (
        left.sample_shape == right.sample_shape
        and left.state_shape == right.state_shape
        and left.particles.shape == right.particles.shape
        and left.weights.shape == right.weights.shape
        and left.support == right.support
        and bool(jnp.array_equal(left.times, right.times))
    )


class CommonNoiseMeanFieldProblem(StrictModule):
    """A finite common-noise support with one conditional empirical law per atom.

    Callbacks receive, in order, the conditional object being evaluated, the fixed
    public common-noise history for that scenario, and ``args``.  In particular,
    ``best_response(flow, history, args)`` is followed by
    ``induced_flow(response, history, args)``.  ``law_distance`` receives
    ``(flow, induced, history, args)``.  For damping below one, ``law_mixture``
    must return the exact convex-law mixture on the union of the two conditional
    particle supports and receives
    ``(current, induced, damping, iteration, history, args)``.  No callback is
    given an unconditional mixture in place of the conditional law.

    ``independent_cluster_labels`` assigns every flattened idiosyncratic particle
    to an independent cluster.  Repeated labels are allowed, but effective sample
    size is computed after particle weights have been aggregated by cluster.
    """

    initial_conditional_flows: tuple[EmpiricalMeanField, ...]
    common_histories: tuple[Any, ...]
    scenario_probabilities: Array
    best_response: Callable[
        [EmpiricalMeanField, Any, Any], FrozenLawBestResponseResult
    ] = eqx.field(static=True)
    induced_flow: Callable[
        [FrozenLawBestResponseResult, Any, Any], EmpiricalMeanField
    ] = eqx.field(static=True)
    law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any, Any], Array] = (
        eqx.field(static=True)
    )
    law_mixture: (
        Callable[
            [EmpiricalMeanField, EmpiricalMeanField, float, int, Any, Any],
            EmpiricalMeanField,
        ]
        | None
    ) = eqx.field(static=True)
    scenario_ids: tuple[str, ...] = eqx.field(static=True)
    common_history_ids: tuple[str, ...] = eqx.field(static=True)
    independent_cluster_labels: tuple[tuple[Hashable, ...], ...] = eqx.field(static=True)
    best_response_id: str = eqx.field(static=True)
    induced_flow_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    law_mixture_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_conditional_flows: Sequence[EmpiricalMeanField],
        common_histories: Sequence[Any],
        scenario_probabilities: ArrayLike,
        independent_cluster_labels: Sequence[Sequence[Hashable]],
        best_response: Callable[
            [EmpiricalMeanField, Any, Any], FrozenLawBestResponseResult
        ],
        induced_flow: Callable[
            [FrozenLawBestResponseResult, Any, Any], EmpiricalMeanField
        ],
        law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any, Any], Array],
        /,
        *,
        law_mixture: (
            Callable[
                [EmpiricalMeanField, EmpiricalMeanField, float, int, Any, Any],
                EmpiricalMeanField,
            ]
            | None
        ) = None,
        law_mixture_id: str | None = None,
        scenario_ids: Sequence[str],
        common_history_ids: Sequence[str],
        best_response_id: str,
        induced_flow_id: str,
        law_distance_id: str,
        problem_id: str,
    ):
        flows = tuple(initial_conditional_flows)
        if not flows or any(not isinstance(flow, EmpiricalMeanField) for flow in flows):
            raise TypeError(
                "initial_conditional_flows must contain EmpiricalMeanField values."
            )
        count = len(flows)
        scenarios = _identifiers(scenario_ids, "scenario_ids")
        history_ids = _identifiers(common_history_ids, "common_history_ids")
        histories = tuple(common_histories)
        labels = tuple(tuple(grouping) for grouping in independent_cluster_labels)
        if not (
            len(scenarios) == len(history_ids) == len(histories) == len(labels) == count
        ):
            raise ValueError(
                "Flows, histories, probabilities, labels, and scenario IDs must "
                "have one entry per common-noise scenario."
            )
        probabilities = jnp.asarray(scenario_probabilities, dtype=float)
        if probabilities.shape != (count,):
            raise ValueError("scenario_probabilities must have shape (num_scenarios,).")
        if not bool(jnp.all(jnp.isfinite(probabilities))) or bool(
            jnp.any(probabilities < 0.0)
        ):
            raise ValueError("scenario_probabilities must be finite and nonnegative.")
        probability_sum = float(jnp.sum(probabilities))
        if not isfinite(probability_sum) or not jnp.isclose(
            probability_sum, 1.0, rtol=1.0e-6, atol=1.0e-8
        ):
            raise ValueError("scenario_probabilities must sum to one.")
        probabilities = probabilities / probability_sum
        for owner, callback in (
            ("best_response", best_response),
            ("induced_flow", induced_flow),
            ("law_distance", law_distance),
        ):
            if not callable(callback):
                raise TypeError(f"{owner} must be callable.")
        if law_mixture is not None and not callable(law_mixture):
            raise TypeError("law_mixture must be callable.")
        if (law_mixture is None) != (law_mixture_id is None):
            raise ValueError("law_mixture and law_mixture_id must be supplied together.")
        reference = flows[0]
        if any(
            flow.state_shape != reference.state_shape
            or flow.support != reference.support
            or not bool(jnp.array_equal(flow.times, reference.times))
            for flow in flows[1:]
        ):
            raise ValueError(
                "All conditional flows must share state shape and time support."
            )
        flow_ids = tuple(flow.mean_field_id for flow in flows)
        if len(set(flow_ids)) != count:
            raise ValueError("Initial conditional flow IDs must be unique by scenario.")
        source_ids = tuple(flow.source_path_id for flow in flows)
        positive_sources = tuple(value for value in source_ids if value is not None)
        if len(set(positive_sources)) != len(positive_sources):
            raise ValueError("Initial conditional source path IDs must be distinct.")
        for index, (flow, grouping) in enumerate(zip(flows, labels, strict=True)):
            particle_count = prod(flow.sample_shape)
            if len(grouping) != particle_count:
                raise ValueError(
                    "independent_cluster_labels must label every flattened "
                    f"particle in scenario {scenarios[index]!r}."
                )
            unique_count = len(set(grouping))
            if unique_count == 0:
                raise ValueError("Each scenario must contain a cluster label.")
        self.initial_conditional_flows = flows
        self.common_histories = histories
        self.scenario_probabilities = probabilities
        self.best_response = best_response
        self.induced_flow = induced_flow
        self.law_distance = law_distance
        self.law_mixture = law_mixture
        self.scenario_ids = scenarios
        self.common_history_ids = history_ids
        self.independent_cluster_labels = labels
        self.best_response_id = _identifier(best_response_id, "best_response_id")
        self.induced_flow_id = _identifier(induced_flow_id, "induced_flow_id")
        self.law_distance_id = _identifier(law_distance_id, "law_distance_id")
        self.law_mixture_id = (
            None
            if law_mixture_id is None
            else _identifier(law_mixture_id, "law_mixture_id")
        )
        self.problem_id = _identifier(problem_id, "problem_id")

    @property
    def initial_flows(self) -> tuple[EmpiricalMeanField, ...]:
        return self.initial_conditional_flows


class CommonNoiseMeanFieldPlan(StrictModule):
    """Fixed capacity and conditional acceptance thresholds."""

    maximum_iterations: int = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    minimum_independent_clusters: int = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int,
        consistency_tolerance: float,
        damping: float = 1.0,
        minimum_effective_sample_size: float = 2.0,
        minimum_independent_clusters: int = 2,
        problem_id: str,
    ):
        if not isinstance(maximum_iterations, int) or maximum_iterations <= 0:
            raise ValueError("maximum_iterations must be a positive integer.")
        tolerance = float(consistency_tolerance)
        damping_value = float(damping)
        minimum_ess = float(minimum_effective_sample_size)
        if not isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("consistency_tolerance must be finite and nonnegative.")
        if not isfinite(damping_value) or not 0.0 < damping_value <= 1.0:
            raise ValueError("damping must be finite and in (0, 1].")
        if not isfinite(minimum_ess) or minimum_ess <= 0.0:
            raise ValueError("minimum_effective_sample_size must be finite and positive.")
        if (
            not isinstance(minimum_independent_clusters, int)
            or minimum_independent_clusters <= 0
        ):
            raise ValueError("minimum_independent_clusters must be a positive integer.")
        problem_identifier = _identifier(problem_id, "problem_id")
        self.maximum_iterations = maximum_iterations
        self.consistency_tolerance = tolerance
        self.damping = damping_value
        self.minimum_effective_sample_size = minimum_ess
        self.minimum_independent_clusters = minimum_independent_clusters
        self.problem_id = problem_identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "common-noise-mean-field-game-fixed-point-plan",
                "problem": problem_identifier,
                "maximum_iterations": maximum_iterations,
                "consistency_tolerance": tolerance,
                "damping": damping_value,
                "minimum_effective_sample_size": minimum_ess,
                "minimum_independent_clusters": minimum_independent_clusters,
            }
        )


class CommonNoiseMeanFieldResult(StrictModule):
    """Conditional evidence for one finite common-noise fixed-point run."""

    problem: CommonNoiseMeanFieldProblem
    plan: CommonNoiseMeanFieldPlan
    conditional_flows: tuple[EmpiricalMeanField, ...]
    induced_conditional_flows: tuple[EmpiricalMeanField | None, ...]
    best_response_results: tuple[FrozenLawBestResponseResult | None, ...]
    distance_history: Array
    aggregate_distance_history: Array
    maximum_conditional_distance_history: Array
    current_effective_sample_size_history: Array
    induced_effective_sample_size_history: Array
    independent_cluster_count_history: Array
    best_response_validity_history: Array
    induced_flow_validity_history: Array
    consistency_validity_history: Array
    scenario_iteration_validity_history: Array
    iteration_validity_history: Array
    scenario_status_history: Array
    scenario_statuses: Array
    final_distances: Array
    scenario_converged: Array
    scenario_valid: Array
    supported_scenarios: Array
    iterations: Array
    accepted_iterations: Array
    accepted_iteration: Array
    converged: Array
    valid: Array
    status: Array
    scenario_probabilities: Array
    common_histories: tuple[Any, ...]
    scenario_ids: tuple[str, ...] = eqx.field(static=True)
    common_history_ids: tuple[str, ...] = eqx.field(static=True)
    current_flow_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    induced_flow_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    best_response_flow_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    best_response_common_history_ids: tuple[tuple[str | None, ...], ...] = eqx.field(
        static=True
    )
    current_source_path_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    induced_source_path_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    best_response_path_ids: tuple[tuple[str | None, ...], ...] = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    best_response_id: str = eqx.field(static=True)
    induced_flow_builder_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    law_mixture_id: str | None = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    conditional_law_consistency_evaluated: bool = eqx.field(static=True)
    unconditional_law_consistency_evaluated: bool = eqx.field(static=True)
    best_response_optimality_evaluated: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    common_noise_equilibrium_claimed: bool = eqx.field(static=True)
    unconditional_mean_field_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def flows(self) -> tuple[EmpiricalMeanField, ...]:
        return self.conditional_flows


def _cluster_evidence(
    flow: EmpiricalMeanField, labels: tuple[Hashable, ...]
) -> tuple[Array, Array, int]:
    snapshots = jax.vmap(flow.snapshot)(flow.times)
    finite = (
        jnp.all(jnp.isfinite(flow.particles))
        & jnp.all(jnp.isfinite(flow.weights))
        & jnp.all(jnp.isfinite(snapshots.effective_sample_size))
    )
    valid = jnp.all(flow.valid) & jnp.all(snapshots.valid) & finite
    flattened_weights = flow.weights.reshape((prod(flow.sample_shape), flow.times.size))
    unique_labels = tuple(dict.fromkeys(labels))
    cluster_weights = jnp.stack(
        [
            jnp.sum(
                flattened_weights[
                    jnp.asarray(
                        [index for index, value in enumerate(labels) if value == label],
                        dtype=jnp.int32,
                    )
                ],
                axis=0,
            )
            for label in unique_labels
        ]
    )
    total = jnp.sum(cluster_weights, axis=0)
    denominator = jnp.sum(jnp.square(cluster_weights), axis=0)
    cluster_ess = jnp.where(denominator > 0.0, jnp.square(total) / denominator, 0.0)
    effective_sample_size = jnp.minimum(
        jnp.min(snapshots.effective_sample_size), jnp.min(cluster_ess)
    )
    return valid, effective_sample_size, len(unique_labels)


def _best_response_matches_flow(
    response: FrozenLawBestResponseResult, current: EmpiricalMeanField
) -> bool:
    return (
        response.problem.mean_field is current
        and response.mean_field is current
        and response.flow_id == current.mean_field_id
        and response.support == current.support
        and response.source_path_id == current.source_path_id
        and response.certificate_label == FROZEN_LAW_BEST_RESPONSE
        and not response.law_consistency_evaluated
    )


def _induced_identity_is_new(
    current: EmpiricalMeanField,
    induced: EmpiricalMeanField,
    response: FrozenLawBestResponseResult,
    used_flow_ids: set[str],
    used_source_path_ids: set[str],
) -> bool:
    source = induced.source_path_id
    return (
        induced is not current
        and induced.mean_field_id != current.mean_field_id
        and induced.mean_field_id not in used_flow_ids
        and source is not None
        and source != current.source_path_id
        and source != response.paths.path_id
        and source not in used_source_path_ids
    )


def _normalised_weights(flow: EmpiricalMeanField) -> Array:
    weights = flow.weights.reshape((flow.num_particles, flow.times.size))
    valid = flow.valid.reshape((flow.num_particles, flow.times.size))
    valid_weights = jnp.where(valid, weights, 0.0)
    return valid_weights / jnp.sum(valid_weights, axis=0, keepdims=True)


def _canonical_particle_order(particles: Array, valid: Array, weights: Array) -> Array:
    particle_count = particles.shape[0]
    features = jnp.concatenate(
        (
            particles.reshape((particle_count, -1)),
            valid.reshape((particle_count, -1)),
            weights.reshape((particle_count, -1)),
        ),
        axis=1,
    )
    keys = tuple(features[:, column] for column in range(features.shape[1] - 1, -1, -1))
    return jnp.lexsort(keys)


def _law_mixture_evidence(
    current: EmpiricalMeanField,
    induced: EmpiricalMeanField,
    mixture: EmpiricalMeanField,
    damping: float,
    used_flow_ids: set[str],
) -> tuple[bool, Array | None, Array | None]:
    """Validate a union-support mixture and return its particle alignment."""

    if (
        mixture is current
        or mixture is induced
        or mixture.mean_field_id in used_flow_ids
        or mixture.mean_field_id in (current.mean_field_id, induced.mean_field_id)
        or mixture.source_path_id is not None
        or mixture.state_shape != current.state_shape
        or mixture.support != current.support
        or not bool(jnp.array_equal(mixture.times, current.times))
        or mixture.num_particles != current.num_particles + induced.num_particles
    ):
        return False, None, None
    time_count = current.times.size
    expected_particles = jnp.concatenate(
        (
            current.particles.reshape(
                (current.num_particles, time_count) + current.state_shape
            ),
            induced.particles.reshape(
                (induced.num_particles, time_count) + induced.state_shape
            ),
        ),
        axis=0,
    )
    expected_valid = jnp.concatenate(
        (
            current.valid.reshape((current.num_particles, time_count)),
            induced.valid.reshape((induced.num_particles, time_count)),
        ),
        axis=0,
    )
    expected_weights = jnp.concatenate(
        (
            (1.0 - damping) * _normalised_weights(current),
            damping * _normalised_weights(induced),
        ),
        axis=0,
    )
    mixture_particles = mixture.particles.reshape(
        (mixture.num_particles, time_count) + mixture.state_shape
    )
    mixture_valid = mixture.valid.reshape((mixture.num_particles, time_count))
    mixture_weights = _normalised_weights(mixture)
    expected_order = _canonical_particle_order(
        expected_particles, expected_valid, expected_weights
    )
    mixture_order = _canonical_particle_order(
        mixture_particles, mixture_valid, mixture_weights
    )
    valid = (
        bool(
            jnp.array_equal(
                mixture_particles[mixture_order], expected_particles[expected_order]
            )
        )
        and bool(
            jnp.array_equal(mixture_valid[mixture_order], expected_valid[expected_order])
        )
        and bool(
            jnp.allclose(
                mixture_weights[mixture_order],
                expected_weights[expected_order],
                rtol=1.0e-6,
                atol=1.0e-8,
            )
        )
    )
    return valid, expected_order, mixture_order


def _aligned_mixture_cluster_labels(
    current: EmpiricalMeanField,
    induced: EmpiricalMeanField,
    labels: tuple[Hashable, ...],
    expected_order: Array,
    mixture_order: Array,
) -> tuple[Hashable, ...]:
    """Align source-cluster labels with a validated, arbitrarily ordered union."""

    expected_labels = tuple((current.mean_field_id, label) for label in labels) + tuple(
        (induced.mean_field_id, label) for label in labels
    )
    aligned_labels = [expected_labels[0]] * len(expected_labels)
    for rank in range(len(expected_labels)):
        aligned_labels[int(mixture_order[rank])] = expected_labels[
            int(expected_order[rank])
        ]
    return tuple(aligned_labels)


def _global_failure_status(
    statuses: Sequence[CommonNoiseMeanFieldStatus],
) -> CommonNoiseMeanFieldStatus:
    priority = (
        CommonNoiseMeanFieldStatus.INVALID_BEST_RESPONSE,
        CommonNoiseMeanFieldStatus.INVALID_INDUCED_LAW,
        CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE,
        CommonNoiseMeanFieldStatus.INSUFFICIENT_INDEPENDENT_CLUSTERS,
        CommonNoiseMeanFieldStatus.LOW_EFFECTIVE_SAMPLE_SIZE,
        CommonNoiseMeanFieldStatus.NONFINITE_LAW_DISTANCE,
        CommonNoiseMeanFieldStatus.INVALID_LAW_DISTANCE,
    )
    return next(status for status in priority if status in statuses)


def solve_common_noise_mean_field_fixed_point(
    problem: CommonNoiseMeanFieldProblem,
    plan: CommonNoiseMeanFieldPlan,
    /,
    *,
    args: Any = None,
) -> CommonNoiseMeanFieldResult:
    """Evaluate conditional fixed points without first mixing common-noise atoms.

    Success requires every positive-probability scenario to have a valid frozen-law
    response, an independently sourced induced conditional law, sufficient
    independent-cluster ESS, and a conditional distance within tolerance in the
    same outer iteration.  The label is candidate evidence, not an equilibrium,
    mean-field-control, finite-population, or unconditional-law claim.
    """

    if not isinstance(problem, CommonNoiseMeanFieldProblem):
        raise TypeError("problem must be a CommonNoiseMeanFieldProblem.")
    if not isinstance(plan, CommonNoiseMeanFieldPlan):
        raise TypeError("plan must be a CommonNoiseMeanFieldPlan.")
    if plan.problem_id != problem.problem_id:
        raise ValueError("plan and problem IDs must match.")
    if plan.damping < 1.0 and problem.law_mixture is None:
        raise ValueError("law_mixture and law_mixture_id are required when damping < 1.")

    capacity = plan.maximum_iterations
    scenario_count = len(problem.scenario_ids)
    dtype = jnp.result_type(
        *(flow.particles for flow in problem.initial_conditional_flows), float
    )
    shape = (capacity, scenario_count)
    distance_history = jnp.full(shape, jnp.nan, dtype=dtype)
    aggregate_distance_history = jnp.full((capacity,), jnp.nan, dtype=dtype)
    maximum_distance_history = jnp.full((capacity,), jnp.nan, dtype=dtype)
    current_ess_history = jnp.full(shape, jnp.nan, dtype=dtype)
    induced_ess_history = jnp.full(shape, jnp.nan, dtype=dtype)
    cluster_count_history = jnp.zeros(shape, dtype=jnp.int32)
    best_response_validity = jnp.zeros(shape, dtype=bool)
    induced_flow_validity = jnp.zeros(shape, dtype=bool)
    consistency_validity = jnp.zeros(shape, dtype=bool)
    scenario_iteration_validity = jnp.zeros(shape, dtype=bool)
    iteration_validity = jnp.zeros((capacity,), dtype=bool)
    scenario_status_history = jnp.full(
        shape, int(CommonNoiseMeanFieldStatus.ZERO_PROBABILITY_SCENARIO), dtype=jnp.int32
    )
    current_flow_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    induced_flow_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    current_source_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    induced_source_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    response_path_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    response_flow_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]
    response_history_ids: list[list[str | None]] = [
        [None] * scenario_count for _ in range(capacity)
    ]

    supported = problem.scenario_probabilities > 0.0
    current = list(problem.initial_conditional_flows)
    current_labels = list(problem.independent_cluster_labels)
    final_induced: list[EmpiricalMeanField | None] = [None] * scenario_count
    final_responses: list[FrozenLawBestResponseResult | None] = [None] * scenario_count
    final_distances = jnp.full((scenario_count,), jnp.nan, dtype=dtype)
    final_scenario_statuses = [
        CommonNoiseMeanFieldStatus.ZERO_PROBABILITY_SCENARIO
    ] * scenario_count
    used_flow_ids = {flow.mean_field_id for flow in current}
    used_source_path_ids = {
        flow.source_path_id for flow in current if flow.source_path_id is not None
    }
    iterations = 0
    accepted_iterations = 0
    accepted_iteration = -1
    status = CommonNoiseMeanFieldStatus.MAX_ITERATIONS

    for iteration in range(capacity):
        iterations = iteration + 1
        iteration_statuses = [
            CommonNoiseMeanFieldStatus.ZERO_PROBABILITY_SCENARIO
        ] * scenario_count
        induced_this_iteration: list[EmpiricalMeanField | None] = [None] * scenario_count
        response_this_iteration: list[FrozenLawBestResponseResult | None] = [
            None
        ] * scenario_count

        for scenario_index in range(scenario_count):
            flow = current[scenario_index]
            current_flow_ids[iteration][scenario_index] = flow.mean_field_id
            current_source_ids[iteration][scenario_index] = flow.source_path_id
            if not bool(supported[scenario_index]):
                continue
            history = problem.common_histories[scenario_index]
            labels = current_labels[scenario_index]
            current_valid, current_ess, cluster_count = _cluster_evidence(flow, labels)
            current_ess_history = current_ess_history.at[iteration, scenario_index].set(
                current_ess
            )
            cluster_count_history = cluster_count_history.at[
                iteration, scenario_index
            ].set(cluster_count)
            if cluster_count < plan.minimum_independent_clusters:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INSUFFICIENT_INDEPENDENT_CLUSTERS
                )
                continue
            if not bool(current_valid):
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_INDUCED_LAW
                )
                continue
            if float(current_ess) < plan.minimum_effective_sample_size:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.LOW_EFFECTIVE_SAMPLE_SIZE
                )
                continue

            response = problem.best_response(flow, history, args)
            if not isinstance(response, FrozenLawBestResponseResult):
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_BEST_RESPONSE
                )
                continue
            response_this_iteration[scenario_index] = response
            final_responses[scenario_index] = response
            response_path_ids[iteration][scenario_index] = response.paths.path_id
            response_flow_ids[iteration][scenario_index] = response.flow_id
            response_history_ids[iteration][scenario_index] = problem.common_history_ids[
                scenario_index
            ]
            response_valid = (
                bool(response.valid)
                and int(response.status) == int(FrozenLawBestResponseStatus.SUCCESS)
                and bool(response.law_evidence_valid)
                and bool(response.effective_sample_size_sufficient)
                and _best_response_matches_flow(response, flow)
            )
            best_response_validity = best_response_validity.at[
                iteration, scenario_index
            ].set(response_valid)
            if not response_valid:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_BEST_RESPONSE
                )
                continue

            induced = problem.induced_flow(response, history, args)
            if not isinstance(induced, EmpiricalMeanField):
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_INDUCED_LAW
                )
                continue
            induced_this_iteration[scenario_index] = induced
            final_induced[scenario_index] = induced
            induced_flow_ids[iteration][scenario_index] = induced.mean_field_id
            induced_source_ids[iteration][scenario_index] = induced.source_path_id
            induced_valid, induced_ess, induced_cluster_count = _cluster_evidence(
                induced, labels
            )
            induced_ess_history = induced_ess_history.at[iteration, scenario_index].set(
                induced_ess
            )
            identity_is_new = _induced_identity_is_new(
                flow,
                induced,
                response,
                used_flow_ids,
                used_source_path_ids,
            )
            law_valid = (
                bool(induced_valid)
                and induced_cluster_count == cluster_count
                and _compatible_flows(flow, induced)
                and identity_is_new
            )
            induced_flow_validity = induced_flow_validity.at[
                iteration, scenario_index
            ].set(law_valid)
            if not law_valid:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_INDUCED_LAW
                )
                continue
            used_flow_ids.add(induced.mean_field_id)
            if induced.source_path_id is not None:
                used_source_path_ids.add(induced.source_path_id)
            if float(induced_ess) < plan.minimum_effective_sample_size:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.LOW_EFFECTIVE_SAMPLE_SIZE
                )
                continue

            distance = jnp.asarray(
                problem.law_distance(flow, induced, history, args), dtype=dtype
            )
            if distance.shape != ():
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_LAW_DISTANCE
                )
                continue
            distance_history = distance_history.at[iteration, scenario_index].set(
                distance
            )
            final_distances = final_distances.at[scenario_index].set(distance)
            if not bool(jnp.isfinite(distance)):
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.NONFINITE_LAW_DISTANCE
                )
                continue
            if float(distance) < 0.0:
                iteration_statuses[scenario_index] = (
                    CommonNoiseMeanFieldStatus.INVALID_LAW_DISTANCE
                )
                continue
            consistency_validity = consistency_validity.at[iteration, scenario_index].set(
                True
            )
            scenario_iteration_validity = scenario_iteration_validity.at[
                iteration, scenario_index
            ].set(True)
            iteration_statuses[scenario_index] = (
                CommonNoiseMeanFieldStatus.SUCCESS
                if float(distance) <= plan.consistency_tolerance
                else CommonNoiseMeanFieldStatus.MAX_ITERATIONS
            )

        scenario_status_history = scenario_status_history.at[iteration].set(
            jnp.asarray([int(value) for value in iteration_statuses], dtype=jnp.int32)
        )
        final_scenario_statuses = iteration_statuses
        supported_statuses = [
            iteration_statuses[index]
            for index in range(scenario_count)
            if bool(supported[index])
        ]
        failures = [
            value
            for value in supported_statuses
            if value
            not in (
                CommonNoiseMeanFieldStatus.SUCCESS,
                CommonNoiseMeanFieldStatus.MAX_ITERATIONS,
            )
        ]
        if failures:
            status = _global_failure_status(failures)
            break

        all_converged = all(
            value == CommonNoiseMeanFieldStatus.SUCCESS for value in supported_statuses
        )
        next_current = list(current)
        next_labels = list(current_labels)
        if not all_converged:
            for scenario_index in range(scenario_count):
                if not bool(supported[scenario_index]):
                    continue
                induced = induced_this_iteration[scenario_index]
                response = response_this_iteration[scenario_index]
                if induced is None or response is None:  # guarded by law evidence
                    raise RuntimeError(
                        "Missing conditional evidence after a valid iteration."
                    )
                if plan.damping == 1.0:
                    next_current[scenario_index] = induced
                    continue

                mixture_builder = problem.law_mixture
                if mixture_builder is None:  # guarded by the preflight requirement
                    raise RuntimeError("Missing law_mixture after validation.")
                flow = current[scenario_index]
                history = problem.common_histories[scenario_index]
                mixture = mixture_builder(
                    flow, induced, plan.damping, iteration, history, args
                )
                if not isinstance(mixture, EmpiricalMeanField):
                    scenario_iteration_validity = scenario_iteration_validity.at[
                        iteration, scenario_index
                    ].set(False)
                    iteration_statuses[scenario_index] = (
                        CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE
                    )
                    continue
                labels = current_labels[scenario_index]
                (
                    law_mixture_valid,
                    expected_order,
                    mixture_order,
                ) = _law_mixture_evidence(
                    flow, induced, mixture, plan.damping, used_flow_ids
                )
                if (
                    not law_mixture_valid
                    or expected_order is None
                    or mixture_order is None
                ):
                    scenario_iteration_validity = scenario_iteration_validity.at[
                        iteration, scenario_index
                    ].set(False)
                    iteration_statuses[scenario_index] = (
                        CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE
                    )
                    continue
                mixture_labels = _aligned_mixture_cluster_labels(
                    flow, induced, labels, expected_order, mixture_order
                )
                mixture_valid, mixture_ess, mixture_cluster_count = _cluster_evidence(
                    mixture, mixture_labels
                )
                if not bool(mixture_valid):
                    scenario_iteration_validity = scenario_iteration_validity.at[
                        iteration, scenario_index
                    ].set(False)
                    iteration_statuses[scenario_index] = (
                        CommonNoiseMeanFieldStatus.INVALID_LAW_MIXTURE
                    )
                    continue
                if mixture_cluster_count < plan.minimum_independent_clusters:
                    scenario_iteration_validity = scenario_iteration_validity.at[
                        iteration, scenario_index
                    ].set(False)
                    iteration_statuses[scenario_index] = (
                        CommonNoiseMeanFieldStatus.INSUFFICIENT_INDEPENDENT_CLUSTERS
                    )
                    continue
                if float(mixture_ess) < plan.minimum_effective_sample_size:
                    scenario_iteration_validity = scenario_iteration_validity.at[
                        iteration, scenario_index
                    ].set(False)
                    iteration_statuses[scenario_index] = (
                        CommonNoiseMeanFieldStatus.LOW_EFFECTIVE_SAMPLE_SIZE
                    )
                    continue
                next_current[scenario_index] = mixture
                next_labels[scenario_index] = mixture_labels
                used_flow_ids.add(mixture.mean_field_id)

            mixture_failures = [
                iteration_statuses[index]
                for index in range(scenario_count)
                if bool(supported[index])
                and iteration_statuses[index]
                not in (
                    CommonNoiseMeanFieldStatus.SUCCESS,
                    CommonNoiseMeanFieldStatus.MAX_ITERATIONS,
                )
            ]
            if mixture_failures:
                scenario_status_history = scenario_status_history.at[iteration].set(
                    jnp.asarray(
                        [int(value) for value in iteration_statuses], dtype=jnp.int32
                    )
                )
                final_scenario_statuses = iteration_statuses
                status = _global_failure_status(mixture_failures)
                break

        iteration_validity = iteration_validity.at[iteration].set(True)
        accepted_iterations += 1
        accepted_iteration = iteration
        conditional_distances = distance_history[iteration]
        aggregate_distance_history = aggregate_distance_history.at[iteration].set(
            jnp.sum(
                jnp.where(
                    supported,
                    problem.scenario_probabilities * conditional_distances,
                    0.0,
                )
            )
        )
        maximum_distance_history = maximum_distance_history.at[iteration].set(
            jnp.max(jnp.where(supported, conditional_distances, -jnp.inf))
        )
        if all_converged:
            status = CommonNoiseMeanFieldStatus.SUCCESS
            final_induced = induced_this_iteration
            final_responses = response_this_iteration
            break

        current = next_current
        current_labels = next_labels
        final_induced = induced_this_iteration
        final_responses = response_this_iteration

    converged = status == CommonNoiseMeanFieldStatus.SUCCESS
    scenario_converged = jnp.asarray(
        [
            bool(supported[index])
            and final_scenario_statuses[index] == CommonNoiseMeanFieldStatus.SUCCESS
            for index in range(scenario_count)
        ]
    )
    scenario_valid = scenario_converged & jnp.isfinite(final_distances)
    valid = bool(converged) and bool(jnp.all(jnp.where(supported, scenario_valid, True)))
    return CommonNoiseMeanFieldResult(
        problem=problem,
        plan=plan,
        conditional_flows=tuple(current),
        induced_conditional_flows=tuple(final_induced),
        best_response_results=tuple(final_responses),
        distance_history=distance_history,
        aggregate_distance_history=aggregate_distance_history,
        maximum_conditional_distance_history=maximum_distance_history,
        current_effective_sample_size_history=current_ess_history,
        induced_effective_sample_size_history=induced_ess_history,
        independent_cluster_count_history=cluster_count_history,
        best_response_validity_history=best_response_validity,
        induced_flow_validity_history=induced_flow_validity,
        consistency_validity_history=consistency_validity,
        scenario_iteration_validity_history=scenario_iteration_validity,
        iteration_validity_history=iteration_validity,
        scenario_status_history=scenario_status_history,
        scenario_statuses=jnp.asarray(
            [int(value) for value in final_scenario_statuses], dtype=jnp.int32
        ),
        final_distances=final_distances,
        scenario_converged=scenario_converged,
        scenario_valid=scenario_valid,
        supported_scenarios=supported,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        accepted_iterations=jnp.asarray(accepted_iterations, dtype=jnp.int32),
        accepted_iteration=jnp.asarray(accepted_iteration, dtype=jnp.int32),
        converged=jnp.asarray(converged),
        valid=jnp.asarray(valid),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        scenario_probabilities=problem.scenario_probabilities,
        common_histories=problem.common_histories,
        scenario_ids=problem.scenario_ids,
        common_history_ids=problem.common_history_ids,
        current_flow_ids=tuple(tuple(row) for row in current_flow_ids),
        induced_flow_ids=tuple(tuple(row) for row in induced_flow_ids),
        current_source_path_ids=tuple(tuple(row) for row in current_source_ids),
        induced_source_path_ids=tuple(tuple(row) for row in induced_source_ids),
        best_response_path_ids=tuple(tuple(row) for row in response_path_ids),
        best_response_flow_ids=tuple(tuple(row) for row in response_flow_ids),
        best_response_common_history_ids=tuple(
            tuple(row) for row in response_history_ids
        ),
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        best_response_id=problem.best_response_id,
        induced_flow_builder_id=problem.induced_flow_id,
        law_distance_id=problem.law_distance_id,
        law_mixture_id=problem.law_mixture_id,
        certificate_label=COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE,
        candidate_evaluation_only=True,
        conditional_law_consistency_evaluated=True,
        unconditional_law_consistency_evaluated=False,
        best_response_optimality_evaluated=False,
        mean_field_game_equilibrium_claimed=False,
        common_noise_equilibrium_claimed=False,
        unconditional_mean_field_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        finite_population_game_claimed=False,
    )


__all__ = [
    "COMMON_NOISE_MFG_FIXED_POINT_CANDIDATE",
    "CommonNoiseMeanFieldPlan",
    "CommonNoiseMeanFieldProblem",
    "CommonNoiseMeanFieldResult",
    "CommonNoiseMeanFieldStatus",
    "solve_common_noise_mean_field_fixed_point",
]
