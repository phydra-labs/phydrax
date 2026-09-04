#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed-capacity induced-law iteration for mean-field-game candidates."""

from __future__ import annotations

from enum import IntEnum
from math import isfinite
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...stochastic import EmpiricalMeanField
from ._mean_field import (
    FROZEN_LAW_BEST_RESPONSE,
    FrozenLawBestResponseResult,
    FrozenLawBestResponseStatus,
)


MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE = "MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE"


class MeanFieldGameFixedPointStatus(IntEnum):
    """Stable termination codes for induced-law fixed-point evaluation."""

    SUCCESS = 0
    INVALID_BEST_RESPONSE = 1
    INVALID_INDUCED_LAW = 2
    LOW_EFFECTIVE_SAMPLE_SIZE = 3
    NONFINITE_LAW_DISTANCE = 4
    INVALID_LAW_DISTANCE = 5
    MAX_ITERATIONS = 6
    INVALID_LAW_MIXTURE = 7


def _identifier(value: str, owner: str, /) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{owner} must be a non-empty string.")
    return value


class MeanFieldGameFixedPointProblem(StrictModule):
    """Callbacks and provenance for one induced-law consistency problem.

    ``best_response`` must evaluate a candidate against the exact flow supplied
    to it. ``induced_flow`` must then generate a new empirical flow from that
    response; reusing either the frozen input flow or its source paths is
    rejected. ``law_distance`` compares the frozen and newly induced laws before
    damping is applied.  For damping below one, ``law_mixture`` must return the
    exact empirical convex-law mixture on the union of their particle supports;
    it receives ``(current, induced, damping, iteration, args)``.
    """

    initial_flow: EmpiricalMeanField
    best_response: Callable[[EmpiricalMeanField, Any], FrozenLawBestResponseResult] = (
        eqx.field(static=True)
    )
    induced_flow: Callable[[FrozenLawBestResponseResult, Any], EmpiricalMeanField] = (
        eqx.field(static=True)
    )
    law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any], Array] = (
        eqx.field(static=True)
    )
    law_mixture: (
        Callable[
            [EmpiricalMeanField, EmpiricalMeanField, float, int, Any],
            EmpiricalMeanField,
        ]
        | None
    ) = eqx.field(static=True)
    best_response_id: str = eqx.field(static=True)
    induced_flow_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    law_mixture_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)

    def __init__(
        self,
        initial_flow: EmpiricalMeanField,
        best_response: Callable[[EmpiricalMeanField, Any], FrozenLawBestResponseResult],
        induced_flow: Callable[[FrozenLawBestResponseResult, Any], EmpiricalMeanField],
        law_distance: Callable[[EmpiricalMeanField, EmpiricalMeanField, Any], Array],
        /,
        *,
        law_mixture: (
            Callable[
                [EmpiricalMeanField, EmpiricalMeanField, float, int, Any],
                EmpiricalMeanField,
            ]
            | None
        ) = None,
        law_mixture_id: str | None = None,
        best_response_id: str,
        induced_flow_id: str,
        law_distance_id: str,
        problem_id: str,
    ):
        if not isinstance(initial_flow, EmpiricalMeanField):
            raise TypeError("initial_flow must be an EmpiricalMeanField.")
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
        self.initial_flow = initial_flow
        self.best_response = best_response
        self.induced_flow = induced_flow
        self.law_distance = law_distance
        self.law_mixture = law_mixture
        self.best_response_id = _identifier(best_response_id, "best_response_id")
        self.induced_flow_id = _identifier(induced_flow_id, "induced_flow_id")
        self.law_distance_id = _identifier(law_distance_id, "law_distance_id")
        self.law_mixture_id = (
            None
            if law_mixture_id is None
            else _identifier(law_mixture_id, "law_mixture_id")
        )
        self.problem_id = _identifier(problem_id, "problem_id")


class MeanFieldGameFixedPointPlan(StrictModule):
    """Static capacity and acceptance thresholds for induced-law iteration."""

    maximum_iterations: int = eqx.field(static=True)
    consistency_tolerance: float = eqx.field(static=True)
    damping: float = eqx.field(static=True)
    minimum_effective_sample_size: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_iterations: int,
        consistency_tolerance: float,
        damping: float = 1.0,
        minimum_effective_sample_size: float = 2.0,
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
        problem_identifier = _identifier(problem_id, "problem_id")
        self.maximum_iterations = maximum_iterations
        self.consistency_tolerance = tolerance
        self.damping = damping_value
        self.minimum_effective_sample_size = minimum_ess
        self.problem_id = problem_identifier
        self.plan_id = canonical_fingerprint(
            {
                "kind": "mean-field-game-fixed-point-plan",
                "problem": problem_identifier,
                "maximum_iterations": maximum_iterations,
                "consistency_tolerance": tolerance,
                "damping": damping_value,
                "minimum_effective_sample_size": minimum_ess,
            }
        )


class MeanFieldGameFixedPointResult(StrictModule):
    """Best-response and induced-law consistency evidence from a fixed-capacity run."""

    problem: MeanFieldGameFixedPointProblem
    plan: MeanFieldGameFixedPointPlan
    flow: EmpiricalMeanField
    induced_flow: EmpiricalMeanField | None
    best_response_result: FrozenLawBestResponseResult | None
    distance_history: Array
    current_effective_sample_size_history: Array
    induced_effective_sample_size_history: Array
    best_response_validity_history: Array
    induced_flow_validity_history: Array
    consistency_validity_history: Array
    iteration_validity_history: Array
    iterations: Array
    accepted_iterations: Array
    accepted_iteration: Array
    final_distance: Array
    converged: Array
    valid: Array
    status: Array
    current_flow_ids: tuple[str | None, ...] = eqx.field(static=True)
    induced_flow_ids: tuple[str | None, ...] = eqx.field(static=True)
    current_flow_id: str = eqx.field(static=True)
    induced_flow_id: str | None = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    best_response_id: str = eqx.field(static=True)
    induced_flow_builder_id: str = eqx.field(static=True)
    law_distance_id: str = eqx.field(static=True)
    law_mixture_id: str | None = eqx.field(static=True)
    certificate_label: str = eqx.field(static=True)
    candidate_evaluation_only: bool = eqx.field(static=True)
    law_consistency_evaluated: bool = eqx.field(static=True)
    best_response_optimality_evaluated: bool = eqx.field(static=True)
    mean_field_game_equilibrium_claimed: bool = eqx.field(static=True)
    mean_field_control_optimum_claimed: bool = eqx.field(static=True)
    finite_population_game_claimed: bool = eqx.field(static=True)
    common_noise_equilibrium_claimed: bool = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.valid

    @property
    def mean_field(self) -> EmpiricalMeanField:
        return self.flow

    @property
    def final_best_response(self) -> FrozenLawBestResponseResult | None:
        return self.best_response_result


def _flow_evidence(flow: EmpiricalMeanField) -> tuple[Array, Array]:
    snapshots = jax.vmap(flow.snapshot)(flow.times)
    effective_sample_size = jnp.min(snapshots.effective_sample_size)
    finite = (
        jnp.all(jnp.isfinite(flow.particles))
        & jnp.all(jnp.isfinite(flow.weights))
        & jnp.all(jnp.isfinite(snapshots.effective_sample_size))
    )
    valid = jnp.all(flow.valid) & jnp.all(snapshots.valid) & finite
    return valid, effective_sample_size


def _compatible_flows(current: EmpiricalMeanField, induced: EmpiricalMeanField) -> bool:
    return (
        current.sample_shape == induced.sample_shape
        and current.state_shape == induced.state_shape
        and current.particles.shape == induced.particles.shape
        and current.weights.shape == induced.weights.shape
        and current.support == induced.support
        and bool(jnp.array_equal(current.times, induced.times))
    )


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


def _law_mixture_is_valid(
    current: EmpiricalMeanField,
    induced: EmpiricalMeanField,
    mixture: EmpiricalMeanField,
    damping: float,
    used_flow_ids: set[str],
) -> bool:
    """Check an exact union-support representation of the convex law mixture."""

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
        return False
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
    return (
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


def solve_mean_field_game_fixed_point(
    problem: MeanFieldGameFixedPointProblem,
    plan: MeanFieldGameFixedPointPlan,
    /,
    *,
    args: Any = None,
) -> MeanFieldGameFixedPointResult:
    """Evaluate best responses and independently induced laws to fixed capacity.

    A successful result certifies only that the final frozen-law response was valid
    and its independently induced law was within ``consistency_tolerance``. It does
    not establish finite-population, common-noise, or mean-field-control claims.
    """

    if not isinstance(problem, MeanFieldGameFixedPointProblem):
        raise TypeError("problem must be a MeanFieldGameFixedPointProblem.")
    if not isinstance(plan, MeanFieldGameFixedPointPlan):
        raise TypeError("plan must be a MeanFieldGameFixedPointPlan.")
    if plan.problem_id != problem.problem_id:
        raise ValueError("plan and problem IDs must match.")
    if plan.damping < 1.0 and problem.law_mixture is None:
        raise ValueError("law_mixture and law_mixture_id are required when damping < 1.")

    capacity = plan.maximum_iterations
    dtype = jnp.result_type(problem.initial_flow.particles, float)
    distance_history = jnp.full((capacity,), jnp.nan, dtype=dtype)
    current_ess_history = jnp.full((capacity,), jnp.nan, dtype=dtype)
    induced_ess_history = jnp.full((capacity,), jnp.nan, dtype=dtype)
    best_response_validity = jnp.zeros((capacity,), dtype=bool)
    induced_flow_validity = jnp.zeros((capacity,), dtype=bool)
    consistency_validity = jnp.zeros((capacity,), dtype=bool)
    iteration_validity = jnp.zeros((capacity,), dtype=bool)
    current_flow_ids: list[str | None] = [None] * capacity
    induced_flow_ids: list[str | None] = [None] * capacity

    current = problem.initial_flow
    final_induced: EmpiricalMeanField | None = None
    final_response: FrozenLawBestResponseResult | None = None
    used_flow_ids = {current.mean_field_id}
    used_source_path_ids = (
        set() if current.source_path_id is None else {current.source_path_id}
    )
    iterations = 0
    accepted_iterations = 0
    accepted_iteration = -1
    final_distance = jnp.asarray(jnp.nan, dtype=dtype)
    status = MeanFieldGameFixedPointStatus.MAX_ITERATIONS

    for index in range(capacity):
        iterations = index + 1
        current_flow_ids[index] = current.mean_field_id
        current_valid, current_ess = _flow_evidence(current)
        current_ess_history = current_ess_history.at[index].set(current_ess)

        response = problem.best_response(current, args)
        if not isinstance(response, FrozenLawBestResponseResult):
            status = MeanFieldGameFixedPointStatus.INVALID_BEST_RESPONSE
            break
        final_response = response
        response_valid = (
            bool(response.valid)
            and int(response.status) == int(FrozenLawBestResponseStatus.SUCCESS)
            and bool(response.law_evidence_valid)
            and bool(response.effective_sample_size_sufficient)
            and bool(current_valid)
            and _best_response_matches_flow(response, current)
        )
        best_response_validity = best_response_validity.at[index].set(response_valid)
        if not response_valid:
            status = MeanFieldGameFixedPointStatus.INVALID_BEST_RESPONSE
            break

        induced = problem.induced_flow(response, args)
        if not isinstance(induced, EmpiricalMeanField):
            status = MeanFieldGameFixedPointStatus.INVALID_INDUCED_LAW
            break
        final_induced = induced
        induced_flow_ids[index] = induced.mean_field_id
        compatible = _compatible_flows(current, induced)
        identity_is_new = _induced_identity_is_new(
            current,
            induced,
            response,
            used_flow_ids,
            used_source_path_ids,
        )
        induced_valid, induced_ess = _flow_evidence(induced)
        induced_ess_history = induced_ess_history.at[index].set(induced_ess)
        law_valid = bool(induced_valid) and compatible and identity_is_new
        induced_flow_validity = induced_flow_validity.at[index].set(law_valid)
        if not law_valid:
            status = MeanFieldGameFixedPointStatus.INVALID_INDUCED_LAW
            break
        if (
            float(current_ess) < plan.minimum_effective_sample_size
            or float(induced_ess) < plan.minimum_effective_sample_size
        ):
            status = MeanFieldGameFixedPointStatus.LOW_EFFECTIVE_SAMPLE_SIZE
            break

        distance = jnp.asarray(problem.law_distance(current, induced, args), dtype=dtype)
        if distance.shape != ():
            status = MeanFieldGameFixedPointStatus.INVALID_LAW_DISTANCE
            break
        distance_history = distance_history.at[index].set(distance)
        final_distance = distance
        if not bool(jnp.isfinite(distance)):
            status = MeanFieldGameFixedPointStatus.NONFINITE_LAW_DISTANCE
            break
        if float(distance) < 0.0:
            status = MeanFieldGameFixedPointStatus.INVALID_LAW_DISTANCE
            break

        consistency_validity = consistency_validity.at[index].set(True)
        if float(distance) <= plan.consistency_tolerance:
            iteration_validity = iteration_validity.at[index].set(True)
            accepted_iterations += 1
            accepted_iteration = index
            status = MeanFieldGameFixedPointStatus.SUCCESS
            break

        used_flow_ids.add(induced.mean_field_id)
        used_source_path_ids.add(induced.source_path_id)
        if plan.damping == 1.0:
            current = induced
            iteration_validity = iteration_validity.at[index].set(True)
            accepted_iterations += 1
            accepted_iteration = index
            continue

        mixture_builder = problem.law_mixture
        if mixture_builder is None:  # guarded by the preflight requirement
            raise RuntimeError("Missing law_mixture after validation.")
        mixture = mixture_builder(current, induced, plan.damping, index, args)
        if not isinstance(mixture, EmpiricalMeanField):
            status = MeanFieldGameFixedPointStatus.INVALID_LAW_MIXTURE
            break
        mixture_valid, mixture_ess = _flow_evidence(mixture)
        if not bool(mixture_valid) or not _law_mixture_is_valid(
            current, induced, mixture, plan.damping, used_flow_ids
        ):
            status = MeanFieldGameFixedPointStatus.INVALID_LAW_MIXTURE
            break
        if float(mixture_ess) < plan.minimum_effective_sample_size:
            status = MeanFieldGameFixedPointStatus.LOW_EFFECTIVE_SAMPLE_SIZE
            break
        current = mixture
        used_flow_ids.add(current.mean_field_id)
        iteration_validity = iteration_validity.at[index].set(True)
        accepted_iterations += 1
        accepted_iteration = index

    converged = status == MeanFieldGameFixedPointStatus.SUCCESS
    valid = (
        converged
        and final_response is not None
        and bool(final_response.valid)
        and _best_response_matches_flow(final_response, current)
        and bool(jnp.isfinite(final_distance))
        and float(final_distance) <= plan.consistency_tolerance
    )
    return MeanFieldGameFixedPointResult(
        problem=problem,
        plan=plan,
        flow=current,
        induced_flow=final_induced,
        best_response_result=final_response,
        distance_history=distance_history,
        current_effective_sample_size_history=current_ess_history,
        induced_effective_sample_size_history=induced_ess_history,
        best_response_validity_history=best_response_validity,
        induced_flow_validity_history=induced_flow_validity,
        consistency_validity_history=consistency_validity,
        iteration_validity_history=iteration_validity,
        iterations=jnp.asarray(iterations, dtype=jnp.int32),
        accepted_iterations=jnp.asarray(accepted_iterations, dtype=jnp.int32),
        accepted_iteration=jnp.asarray(accepted_iteration, dtype=jnp.int32),
        final_distance=final_distance,
        converged=jnp.asarray(converged),
        valid=jnp.asarray(valid),
        status=jnp.asarray(int(status), dtype=jnp.int32),
        current_flow_ids=tuple(current_flow_ids),
        induced_flow_ids=tuple(induced_flow_ids),
        current_flow_id=current.mean_field_id,
        induced_flow_id=(None if final_induced is None else final_induced.mean_field_id),
        problem_id=problem.problem_id,
        plan_id=plan.plan_id,
        best_response_id=problem.best_response_id,
        induced_flow_builder_id=problem.induced_flow_id,
        law_distance_id=problem.law_distance_id,
        law_mixture_id=problem.law_mixture_id,
        certificate_label=MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE,
        candidate_evaluation_only=True,
        law_consistency_evaluated=True,
        best_response_optimality_evaluated=False,
        mean_field_game_equilibrium_claimed=False,
        mean_field_control_optimum_claimed=False,
        finite_population_game_claimed=False,
        common_noise_equilibrium_claimed=False,
    )


__all__ = [
    "MEAN_FIELD_GAME_FIXED_POINT_CANDIDATE",
    "MeanFieldGameFixedPointPlan",
    "MeanFieldGameFixedPointProblem",
    "MeanFieldGameFixedPointResult",
    "MeanFieldGameFixedPointStatus",
    "solve_mean_field_game_fixed_point",
]
