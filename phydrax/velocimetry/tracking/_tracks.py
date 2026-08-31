#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from numbers import Integral
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...combinatorial import (
    BipartiteAssignmentSpace,
    CapacitatedFlowSpace,
    CombinatorialCertification,
    CycleCancelingMinCostFlow,
    HungarianAssignment,
    LinearCombinatorialProblem,
    solve_combinatorial,
)
from ...dynamics import StateLayout, TrajectoryData
from ...linalg import SmallLinearSolvePlan, solve_small_linear
from ...sparse import EdgeRelation
from ._reconstruction import ParticleReconstructionResult
from ._types import (
    TrackResult,
    TrackRuntimeState,
    TrackStatus,
    TrackStepEvidence,
    TrackStepResult,
)


class TrackLinkPlan(StrictModule, NonTrainableState):
    """Fixed-capacity constant-velocity association and lifecycle policy."""

    small_solve_plan: SmallLinearSolvePlan
    maximum_tracks: int = eqx.field(static=True)
    maximum_missed: int = eqx.field(static=True)
    mahalanobis_gate: float = eqx.field(static=True)
    unmatched_cost: float = eqx.field(static=True)
    process_acceleration_variance: float = eqx.field(static=True)
    initial_velocity_variance: float = eqx.field(static=True)
    ambiguity_margin: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        maximum_tracks: int,
        /,
        *,
        maximum_missed: int = 2,
        mahalanobis_gate: float = 16.0,
        unmatched_cost: float = 20.0,
        process_acceleration_variance: float = 1e-3,
        initial_velocity_variance: float = 1.0,
        ambiguity_margin: float = 0.25,
        small_solve_plan: SmallLinearSolvePlan | None = None,
    ):
        for name, value in (
            ("maximum_tracks", maximum_tracks),
            ("maximum_missed", maximum_missed),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if maximum_tracks <= 0 or maximum_missed < 0:
            raise ValueError(
                "Track capacity must be positive and maximum_missed nonnegative."
            )
        values = jnp.asarray(
            (
                mahalanobis_gate,
                unmatched_cost,
                process_acceleration_variance,
                initial_velocity_variance,
                ambiguity_margin,
            )
        )
        if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError(
                "Track-link costs and variances must be finite and nonnegative."
            )
        if mahalanobis_gate <= 0.0 or unmatched_cost <= 0.0:
            raise ValueError("mahalanobis_gate and unmatched_cost must be positive.")
        resolved_solve = (
            SmallLinearSolvePlan(3) if small_solve_plan is None else small_solve_plan
        )
        if not isinstance(resolved_solve, SmallLinearSolvePlan):
            raise TypeError("small_solve_plan must be a SmallLinearSolvePlan or None.")
        if resolved_solve.dimension != 3:
            raise ValueError("small_solve_plan must solve dimension three.")
        self.small_solve_plan = resolved_solve
        self.maximum_tracks = int(maximum_tracks)
        self.maximum_missed = int(maximum_missed)
        self.mahalanobis_gate = float(mahalanobis_gate)
        self.unmatched_cost = float(unmatched_cost)
        self.process_acceleration_variance = float(process_acceleration_variance)
        self.initial_velocity_variance = float(initial_velocity_variance)
        self.ambiguity_margin = float(ambiguity_margin)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "streaming-constant-velocity-tracks",
                "maximum_tracks": self.maximum_tracks,
                "maximum_missed": self.maximum_missed,
                "mahalanobis_gate": self.mahalanobis_gate,
                "unmatched_cost": self.unmatched_cost,
                "process_acceleration_variance": self.process_acceleration_variance,
                "initial_velocity_variance": self.initial_velocity_variance,
                "ambiguity_margin": self.ambiguity_margin,
                "small_solve_plan": resolved_solve.plan_id,
            }
        )


def initialize_tracks(
    plan: TrackLinkPlan,
    /,
    *,
    dtype: Any = float,
    initial_time: float | None = None,
    first_track_id: int = 0,
) -> TrackRuntimeState:
    """Create an empty fixed-capacity runtime without consuming any track IDs."""
    if not isinstance(plan, TrackLinkPlan):
        raise TypeError("plan must be a TrackLinkPlan.")
    if isinstance(first_track_id, bool) or not isinstance(first_track_id, Integral):
        raise TypeError("first_track_id must be a nonnegative integer.")
    if first_track_id < 0:
        raise ValueError("first_track_id must be nonnegative.")
    if initial_time is not None and not jnp.isfinite(initial_time):
        raise ValueError("initial_time must be finite or None.")
    capacity = plan.maximum_tracks
    resolved_dtype = jnp.dtype(dtype)
    if not jnp.issubdtype(resolved_dtype, jnp.inexact):
        raise TypeError("Track state dtype must be inexact.")
    initialized = initial_time is not None
    time = 0.0 if initial_time is None else float(initial_time)
    return TrackRuntimeState(
        track_ids=jnp.full((capacity,), -1, dtype=jnp.int64),
        active=jnp.zeros((capacity,), dtype=bool),
        age=jnp.zeros((capacity,), dtype=jnp.int32),
        missed=jnp.zeros((capacity,), dtype=jnp.int32),
        states=jnp.zeros((capacity, 6), dtype=resolved_dtype),
        covariance=jnp.zeros((capacity, 6, 6), dtype=resolved_dtype),
        last_observation_time=jnp.full((capacity,), time, dtype=resolved_dtype),
        time=jnp.asarray(time, dtype=resolved_dtype),
        initialized=jnp.asarray(initialized),
        next_track_id=jnp.asarray(first_track_id, dtype=jnp.int64),
        step_index=jnp.asarray(0, dtype=jnp.int32),
        capacity=capacity,
        plan_id=plan.plan_id,
    )


def _motion_matrices(dt: Array, acceleration_variance: float, dtype, /):
    identity = jnp.eye(3, dtype=dtype)
    transition = jnp.eye(6, dtype=dtype).at[:3, 3:].set(dt * identity)
    process = jnp.zeros((6, 6), dtype=dtype)
    process = process.at[:3, :3].set(0.25 * dt**4 * identity)
    process = process.at[:3, 3:].set(0.5 * dt**3 * identity)
    process = process.at[3:, :3].set(0.5 * dt**3 * identity)
    process = process.at[3:, 3:].set(dt**2 * identity)
    return transition, acceleration_variance * process


def _validate_measurements(positions, covariance, valid, /):
    if positions.ndim != 2 or positions.shape[-1] != 3:
        raise ValueError("positions_xyz must have shape (observation, 3).")
    capacity = int(positions.shape[0])
    if covariance.shape != (capacity, 3, 3):
        raise ValueError("covariance_xyz must have shape (observation, 3, 3).")
    if valid.shape != (capacity,):
        raise ValueError("valid must have shape (observation,).")
    finite = jnp.all(jnp.isfinite(positions), axis=-1) & jnp.all(
        jnp.isfinite(covariance), axis=(-2, -1)
    )
    positions = eqx.error_if(
        positions,
        jnp.any(valid & ~finite),
        "Every valid track observation must have finite position and covariance.",
    )
    symmetric = jnp.all(
        jnp.abs(covariance - jnp.swapaxes(covariance, -1, -2)) <= 1e-6,
        axis=(-2, -1),
    )
    positions = eqx.error_if(
        positions,
        jnp.any(valid & ~symmetric),
        "Every valid track observation covariance must be symmetric.",
    )
    return (
        jnp.where(valid[:, None], positions, 0.0),
        jnp.where(valid[:, None, None], covariance, 0.0),
        valid,
    )


def link_tracks_step(
    state: TrackRuntimeState,
    positions_xyz: ArrayLike,
    covariance_xyz: ArrayLike,
    valid: ArrayLike,
    time: ArrayLike,
    plan: TrackLinkPlan,
    /,
) -> TrackStepResult:
    """Predict, gate, assign, update, birth, miss, and terminate one frame."""
    if not isinstance(state, TrackRuntimeState):
        raise TypeError("state must be a TrackRuntimeState.")
    if not isinstance(plan, TrackLinkPlan):
        raise TypeError("plan must be a TrackLinkPlan.")
    if state.plan_id != plan.plan_id or state.capacity != plan.maximum_tracks:
        raise ValueError("Track runtime state and plan do not match.")
    positions = jnp.asarray(positions_xyz, dtype=state.states.dtype)
    covariance = jnp.asarray(covariance_xyz, dtype=state.states.dtype)
    observation_valid = jnp.asarray(valid, dtype=bool)
    positions, covariance, observation_valid = _validate_measurements(
        positions, covariance, observation_valid
    )
    observation_capacity = int(positions.shape[0])
    current_time = jnp.asarray(time, dtype=state.states.dtype)
    current_time = eqx.error_if(
        current_time,
        ~jnp.isfinite(current_time),
        "Track time must be finite.",
    )
    current_time = eqx.error_if(
        current_time,
        state.initialized & (current_time <= state.time),
        "Track times must be strictly increasing.",
    )
    dt = jnp.where(state.initialized, current_time - state.time, 0.0)
    transition, process_covariance = _motion_matrices(
        dt, plan.process_acceleration_variance, state.states.dtype
    )
    predicted_states = contract("ij,kj->ki", transition, state.states)
    predicted_covariance = (
        contract("ij,kjl,ml->kim", transition, state.covariance, transition)
        + process_covariance[None, :, :]
    )
    predicted_states = jnp.where(state.active[:, None], predicted_states, 0.0)
    predicted_covariance = jnp.where(
        state.active[:, None, None], predicted_covariance, 0.0
    )

    innovation = positions[None, :, :] - predicted_states[:, None, :3]
    innovation_covariance = (
        predicted_covariance[:, None, :3, :3] + covariance[None, :, :, :]
    )
    solve = solve_small_linear(
        plan.small_solve_plan,
        innovation_covariance,
        innovation[..., None],
    )
    solved_innovation = solve.value[..., 0]
    mahalanobis = contract("kmi,kmi->km", innovation, solved_innovation)
    pair_valid = (
        state.active[:, None]
        & observation_valid[None, :]
        & solve.successful
        & jnp.isfinite(mahalanobis)
        & (mahalanobis <= plan.mahalanobis_gate)
    )
    pair_cost = jnp.where(pair_valid, mahalanobis, 0.0)

    track_capacity = state.capacity
    dimension = track_capacity + observation_capacity
    costs = jnp.zeros((dimension, dimension), dtype=state.states.dtype)
    allowed = jnp.zeros((dimension, dimension), dtype=bool)
    costs = costs.at[:track_capacity, :observation_capacity].set(pair_cost)
    allowed = allowed.at[:track_capacity, :observation_capacity].set(pair_valid)
    track_index = jnp.arange(track_capacity, dtype=jnp.int32)
    observation_index = jnp.arange(observation_capacity, dtype=jnp.int32)
    costs = costs.at[track_index, observation_capacity + track_index].set(
        jnp.where(state.active, plan.unmatched_cost, 0.0)
    )
    costs = costs.at[track_capacity + observation_index, observation_index].set(
        jnp.where(observation_valid, plan.unmatched_cost, 0.0)
    )
    allowed = allowed.at[track_index, observation_capacity + track_index].set(True)
    allowed = allowed.at[track_capacity + observation_index, observation_index].set(True)
    allowed = allowed.at[track_capacity:, observation_capacity:].set(True)
    problem = LinearCombinatorialProblem(
        BipartiteAssignmentSpace(dimension, dimension, valid=allowed),
        costs,
        problem_id=f"track-step:{plan.plan_id}:{state.step_index}",
    )
    method = HungarianAssignment(maximum_dimension=dimension)
    assignment = method.solve(problem, method.plan(problem, CombinatorialCertification()))
    assigned = assignment.decision.columns[:track_capacity]
    matched = (
        assignment.valid
        & state.active
        & (assigned >= 0)
        & (assigned < observation_capacity)
    )
    track_observation = jnp.where(matched, assigned, -1).astype(jnp.int32)
    safe_observation = jnp.clip(track_observation, 0, observation_capacity - 1)
    matched_measurement = positions[safe_observation]
    matched_covariance = covariance[safe_observation]

    matched_innovation = matched_measurement - predicted_states[:, :3]
    matched_innovation_covariance = predicted_covariance[:, :3, :3] + matched_covariance
    inverse_result = solve_small_linear(
        plan.small_solve_plan,
        matched_innovation_covariance,
        jnp.broadcast_to(
            jnp.eye(3, dtype=state.states.dtype),
            (track_capacity, 3, 3),
        ),
    )
    gain = contract("kij,kjl->kil", predicted_covariance[:, :, :3], inverse_result.value)
    proposed_states = predicted_states + contract("kij,kj->ki", gain, matched_innovation)
    observation_matrix = (
        jnp.zeros((3, 6), dtype=state.states.dtype)
        .at[:, :3]
        .set(jnp.eye(3, dtype=state.states.dtype))
    )
    update_operator = jnp.eye(6, dtype=state.states.dtype)[None, :, :] - contract(
        "kij,jl->kil", gain, observation_matrix
    )
    proposed_covariance = contract(
        "kij,kjl,kml->kim",
        update_operator,
        predicted_covariance,
        update_operator,
    ) + contract("kij,kjl,kml->kim", gain, matched_covariance, gain)
    proposed_covariance = 0.5 * (
        proposed_covariance + jnp.swapaxes(proposed_covariance, -1, -2)
    )
    matched = matched & inverse_result.successful
    next_states = jnp.where(matched[:, None], proposed_states, predicted_states)
    next_covariance = jnp.where(
        matched[:, None, None], proposed_covariance, predicted_covariance
    )
    next_missed = jnp.where(
        state.active, jnp.where(matched, 0, state.missed + 1), 0
    ).astype(jnp.int32)
    deaths = state.active & ~matched & (next_missed > plan.maximum_missed)
    surviving = state.active & ~deaths
    next_active = surviving
    next_ids = jnp.where(surviving, state.track_ids, -1)
    next_age = jnp.where(surviving, state.age + 1, 0).astype(jnp.int32)
    next_missed = jnp.where(surviving, next_missed, 0)
    next_last_observation_time = jnp.where(
        matched, current_time, state.last_observation_time
    )
    next_states = jnp.where(surviving[:, None], next_states, 0.0)
    next_covariance = jnp.where(surviving[:, None, None], next_covariance, 0.0)

    match_matrix = (
        jax.nn.one_hot(
            jnp.clip(track_observation, 0, observation_capacity - 1),
            observation_capacity,
            dtype=bool,
        )
        & matched[:, None]
    )
    observation_matched = jnp.any(match_matrix, axis=0)
    unmatched_observations = observation_valid & ~observation_matched
    free = ~next_active
    birth_order = jnp.cumsum(unmatched_observations.astype(jnp.int32)) - 1
    free_order = jnp.cumsum(free.astype(jnp.int32)) - 1
    birth_matrix = (
        free[:, None]
        & unmatched_observations[None, :]
        & (free_order[:, None] == birth_order[None, :])
    )
    births = jnp.any(birth_matrix, axis=-1)
    birth_observation = jnp.argmax(birth_matrix, axis=-1).astype(jnp.int32)
    safe_birth_observation = jnp.clip(birth_observation, 0, observation_capacity - 1)
    birth_rank = jnp.where(births, free_order, 0)
    birth_ids = state.next_track_id + birth_rank.astype(jnp.int64)
    birth_state = jnp.concatenate(
        (
            positions[safe_birth_observation],
            jnp.zeros((track_capacity, 3), dtype=state.states.dtype),
        ),
        axis=-1,
    )
    birth_covariance = jnp.zeros((track_capacity, 6, 6), dtype=state.states.dtype)
    birth_covariance = birth_covariance.at[:, :3, :3].set(
        covariance[safe_birth_observation]
    )
    birth_covariance = birth_covariance.at[:, 3:, 3:].set(
        plan.initial_velocity_variance * jnp.eye(3, dtype=state.states.dtype)
    )
    next_states = jnp.where(births[:, None], birth_state, next_states)
    next_covariance = jnp.where(births[:, None, None], birth_covariance, next_covariance)
    next_active = next_active | births
    next_ids = jnp.where(births, birth_ids, next_ids)
    next_age = jnp.where(births, 1, next_age)
    next_missed = jnp.where(births, 0, next_missed)
    next_last_observation_time = jnp.where(
        births, current_time, next_last_observation_time
    )
    accepted_birth_count = jnp.sum(births, dtype=jnp.int32)
    birth_requested_count = jnp.sum(unmatched_observations, dtype=jnp.int32)
    overflow_count = jnp.maximum(birth_requested_count - accepted_birth_count, 0)
    next_track_observation = jnp.where(
        births, birth_observation, track_observation
    ).astype(jnp.int32)

    observation_track_slots = jnp.full((observation_capacity,), -1, dtype=jnp.int32)
    observation_track_ids = jnp.full((observation_capacity,), -1, dtype=jnp.int64)
    ownership = match_matrix | birth_matrix
    owner_valid = jnp.any(ownership, axis=0)
    owner_slot = jnp.argmax(ownership, axis=0).astype(jnp.int32)
    observation_track_slots = jnp.where(owner_valid, owner_slot, -1)
    observation_track_ids = jnp.where(
        owner_valid, next_ids[jnp.clip(owner_slot, 0, track_capacity - 1)], -1
    )
    ordered_cost = jnp.sort(jnp.where(pair_valid, mahalanobis, jnp.inf), axis=-1)
    ambiguous = (
        matched
        & jnp.isfinite(ordered_cost[:, 1])
        & ((ordered_cost[:, 1] - ordered_cost[:, 0]) <= plan.ambiguity_margin)
        if observation_capacity > 1
        else jnp.zeros((track_capacity,), dtype=bool)
    )
    status = jnp.where(
        ~assignment.valid,
        int(TrackStatus.ASSIGNMENT_FAILED),
        jnp.where(
            overflow_count > 0,
            int(TrackStatus.CAPACITY_EXHAUSTED),
            int(TrackStatus.SUCCESS),
        ),
    ).astype(jnp.int32)
    next_state = TrackRuntimeState(
        next_ids,
        next_active,
        next_age,
        next_missed,
        next_states,
        next_covariance,
        next_last_observation_time,
        current_time,
        jnp.asarray(True),
        state.next_track_id + accepted_birth_count.astype(jnp.int64),
        state.step_index + 1,
        state.capacity,
        state.plan_id,
    )
    evidence = TrackStepEvidence(
        matched_count=jnp.sum(matched, dtype=jnp.int32),
        missed_count=jnp.sum(state.active & ~matched, dtype=jnp.int32),
        birth_count=accepted_birth_count,
        death_count=jnp.sum(deaths, dtype=jnp.int32),
        unmatched_observation_count=birth_requested_count,
        overflow_count=overflow_count,
        ambiguous_match_count=jnp.sum(ambiguous, dtype=jnp.int32),
        assignment_status=assignment.status,
        plan_id=plan.plan_id,
    )
    return TrackStepResult(
        next_state,
        observation_track_ids,
        observation_track_slots,
        next_track_observation,
        matched,
        births,
        deaths,
        unmatched_observations,
        ambiguous,
        positions,
        covariance,
        observation_valid,
        status,
        evidence,
    )


def link_tracks(
    reconstructions: Sequence[ParticleReconstructionResult],
    times: ArrayLike,
    plan: TrackLinkPlan,
    /,
    *,
    initial_state: TrackRuntimeState | None = None,
) -> TrackResult:
    """Link a finite reconstruction sequence while retaining every streaming event."""
    sequence = tuple(reconstructions)
    if not sequence:
        raise ValueError("At least one reconstruction frame is required.")
    if any(not isinstance(item, ParticleReconstructionResult) for item in sequence):
        raise TypeError(
            "reconstructions must contain ParticleReconstructionResult values."
        )
    time_values = jnp.asarray(times, dtype=float)
    if time_values.shape != (len(sequence),):
        raise ValueError("times must contain one scalar per reconstruction frame.")
    observation_capacity = int(sequence[0].positions_xyz.shape[0])
    if any(item.positions_xyz.shape != (observation_capacity, 3) for item in sequence):
        raise ValueError("Every reconstruction must use one observation capacity.")
    runtime = (
        initialize_tracks(plan, dtype=sequence[0].positions_xyz.dtype)
        if initial_state is None
        else initial_state
    )
    steps = []
    for index, reconstruction in enumerate(sequence):
        step = link_tracks_step(
            runtime,
            reconstruction.positions_xyz,
            reconstruction.covariance_xyz,
            reconstruction.valid,
            time_values[index],
            plan,
        )
        steps.append(step)
        runtime = step.state
    states = jnp.stack(tuple(step.state.states for step in steps), axis=1)
    covariances = jnp.stack(tuple(step.state.covariance for step in steps), axis=1)
    active = jnp.stack(tuple(step.state.active for step in steps), axis=1)
    observed = jnp.stack(
        tuple(step.matched_tracks | step.births for step in steps), axis=1
    )
    track_ids = jnp.stack(tuple(step.state.track_ids for step in steps), axis=1)
    observation_indices = jnp.stack(
        tuple(step.track_observation_indices for step in steps), axis=1
    )
    safe_observation = jnp.clip(observation_indices, 0, observation_capacity - 1)
    observations_by_time = jnp.stack(tuple(step.observations for step in steps), axis=0)
    covariance_by_time = jnp.stack(
        tuple(step.observation_covariance for step in steps), axis=0
    )
    time_index = jnp.arange(len(steps), dtype=jnp.int32)[None, :]
    observations = observations_by_time[time_index, safe_observation]
    observation_covariances = covariance_by_time[time_index, safe_observation]
    observations = jnp.where(observed[:, :, None], observations, 0.0)
    observation_covariances = jnp.where(
        observed[:, :, None, None], observation_covariances, 0.0
    )
    births = jnp.stack(tuple(step.births for step in steps), axis=1)
    deaths = jnp.stack(tuple(step.deaths for step in steps), axis=1)
    if len(sequence) > 1:
        resets = (
            (track_ids[:, 1:] != track_ids[:, :-1]) | ~observed[:, 1:] | ~observed[:, :-1]
        )
    else:
        resets = jnp.zeros((plan.maximum_tracks, 0), dtype=bool)
    observation_track_ids = jnp.stack(
        tuple(step.observation_track_ids for step in steps), axis=0
    )
    observation_valid = jnp.stack(tuple(step.observation_valid for step in steps), axis=0)
    step_status = jnp.stack(tuple(step.status for step in steps), axis=0)
    overflow_count = jnp.stack(
        tuple(step.evidence.overflow_count for step in steps), axis=0
    )
    source_ids = tuple(item.reconstruction_id for item in sequence)
    result_id = "tracks:" + canonical_fingerprint(
        {
            "sources": source_ids,
            "times": tuple(float(value) for value in time_values),
            "plan": plan.plan_id,
        }
    )
    return TrackResult(
        time_values,
        states,
        covariances,
        active,
        observed,
        track_ids,
        observation_indices,
        observations,
        observation_covariances,
        births,
        deaths,
        resets,
        observation_track_ids,
        observation_valid,
        step_status,
        overflow_count,
        runtime,
        plan.maximum_tracks,
        observation_capacity,
        result_id,
        source_ids,
    )


class OfflineTrackRefinementPlan(StrictModule, NonTrainableState):
    """Time-expanded native min-cost-flow policy for frozen 3-D observations."""

    maximum_gap: int = eqx.field(static=True)
    maximum_displacement: float = eqx.field(static=True)
    birth_cost: float = eqx.field(static=True)
    death_cost: float = eqx.field(static=True)
    gap_penalty: float = eqx.field(static=True)
    maximum_iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_gap: int = 2,
        maximum_displacement: float,
        birth_cost: float = 1.0,
        death_cost: float = 1.0,
        gap_penalty: float = 0.25,
        maximum_iterations: int = 10_000,
    ):
        for name, value in (
            ("maximum_gap", maximum_gap),
            ("maximum_iterations", maximum_iterations),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer.")
        if maximum_gap <= 0 or maximum_iterations <= 0:
            raise ValueError("Flow gap and iteration limits must be positive.")
        values = jnp.asarray((maximum_displacement, birth_cost, death_cost, gap_penalty))
        if not bool(jnp.all(jnp.isfinite(values))) or bool(jnp.any(values < 0.0)):
            raise ValueError(
                "Flow costs and displacement must be finite and nonnegative."
            )
        if maximum_displacement <= 0.0:
            raise ValueError("maximum_displacement must be positive.")
        self.maximum_gap = int(maximum_gap)
        self.maximum_displacement = float(maximum_displacement)
        self.birth_cost = float(birth_cost)
        self.death_cost = float(death_cost)
        self.gap_penalty = float(gap_penalty)
        self.maximum_iterations = int(maximum_iterations)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "offline-particle-track-min-cost-flow",
                "maximum_gap": self.maximum_gap,
                "maximum_displacement": self.maximum_displacement,
                "birth_cost": self.birth_cost,
                "death_cost": self.death_cost,
                "gap_penalty": self.gap_penalty,
                "maximum_iterations": self.maximum_iterations,
            }
        )


class OfflineTrackRefinementResult(StrictModule):
    """Selected time-expanded links in the original frame/observation layout."""

    predecessor: Array
    successor: Array
    selected_observations: Array
    status: Array
    valid: Array
    solver_result: object
    refinement_id: str = eqx.field(static=True)


def refine_tracks_min_cost_flow(
    reconstructions: Sequence[ParticleReconstructionResult],
    times: ArrayLike,
    plan: OfflineTrackRefinementPlan,
    /,
) -> OfflineTrackRefinementResult:
    """Globally refine observation links with native capacitated min-cost flow."""
    sequence = tuple(reconstructions)
    if len(sequence) < 2:
        raise ValueError("Offline flow refinement requires at least two frames.")
    if any(not isinstance(item, ParticleReconstructionResult) for item in sequence):
        raise TypeError(
            "reconstructions must contain ParticleReconstructionResult values."
        )
    time_values = jnp.asarray(times, dtype=float)
    if time_values.shape != (len(sequence),):
        raise ValueError("times must contain one scalar per reconstruction frame.")
    if bool(jnp.any(~jnp.isfinite(time_values))) or bool(
        jnp.any(time_values[1:] <= time_values[:-1])
    ):
        raise ValueError(
            "Offline refinement times must be finite and strictly increasing."
        )
    observation_capacity = int(sequence[0].positions_xyz.shape[0])
    if any(item.positions_xyz.shape != (observation_capacity, 3) for item in sequence):
        raise ValueError("Every reconstruction must have the same observation capacity.")
    positions = jnp.stack(tuple(item.positions_xyz for item in sequence), axis=0)
    observation_valid = jnp.stack(tuple(item.valid for item in sequence), axis=0)
    frame_count = len(sequence)
    observation_count = frame_count * observation_capacity
    source_vertex = 2 * observation_count
    sink_vertex = source_vertex + 1
    vertex_count = sink_vertex + 1
    source_indices: list[int] = [sink_vertex]
    target_indices: list[int] = [source_vertex]
    edge_valid_values = [jnp.asarray(True)]
    edge_capacities = [jnp.asarray(observation_count, dtype=jnp.int32)]
    edge_costs = [jnp.asarray(0.0, dtype=positions.dtype)]
    birth_edge = []
    detection_edge = []
    death_edge = []
    detection_reward = -(plan.birth_cost + plan.death_cost + 1.0)
    for observation in range(observation_count):
        frame = observation // observation_capacity
        slot = observation % observation_capacity
        active = observation_valid[frame, slot]
        input_vertex = observation
        output_vertex = observation_count + observation
        birth_edge.append(len(source_indices))
        source_indices.append(source_vertex)
        target_indices.append(input_vertex)
        edge_valid_values.append(active)
        edge_capacities.append(active.astype(jnp.int32))
        edge_costs.append(jnp.asarray(plan.birth_cost, dtype=positions.dtype))
        detection_edge.append(len(source_indices))
        source_indices.append(input_vertex)
        target_indices.append(output_vertex)
        edge_valid_values.append(active)
        edge_capacities.append(active.astype(jnp.int32))
        edge_costs.append(jnp.asarray(detection_reward, dtype=positions.dtype))
        death_edge.append(len(source_indices))
        source_indices.append(output_vertex)
        target_indices.append(sink_vertex)
        edge_valid_values.append(active)
        edge_capacities.append(active.astype(jnp.int32))
        edge_costs.append(jnp.asarray(plan.death_cost, dtype=positions.dtype))
    link_edges: list[tuple[int, int, int]] = []
    for source_frame in range(frame_count - 1):
        for gap in range(1, min(plan.maximum_gap, frame_count - 1 - source_frame) + 1):
            target_frame = source_frame + gap
            delta_time = time_values[target_frame] - time_values[source_frame]
            displacement = (
                positions[source_frame, :, None, :] - positions[target_frame, None, :, :]
            )
            distance = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
            gate = plan.maximum_displacement * delta_time
            for source_slot in range(observation_capacity):
                for target_slot in range(observation_capacity):
                    source_observation = source_frame * observation_capacity + source_slot
                    target_observation = target_frame * observation_capacity + target_slot
                    active = (
                        observation_valid[source_frame, source_slot]
                        & observation_valid[target_frame, target_slot]
                        & (distance[source_slot, target_slot] <= gate)
                    )
                    edge = len(source_indices)
                    source_indices.append(observation_count + source_observation)
                    target_indices.append(target_observation)
                    edge_valid_values.append(active)
                    edge_capacities.append(active.astype(jnp.int32))
                    normalized = distance[source_slot, target_slot] / jnp.maximum(
                        gate, jnp.finfo(positions.dtype).tiny
                    )
                    edge_costs.append(normalized**2 + plan.gap_penalty * float(gap - 1))
                    link_edges.append((edge, source_observation, target_observation))
    edge_valid = jnp.stack(edge_valid_values)
    capacities = jnp.stack(edge_capacities)
    costs = jnp.stack(edge_costs)
    relation = EdgeRelation(
        jnp.asarray(source_indices, dtype=jnp.int32),
        jnp.asarray(target_indices, dtype=jnp.int32),
        source_size=vertex_count,
        target_size=vertex_count,
        valid=edge_valid,
    )
    space = CapacitatedFlowSpace(
        relation,
        jnp.zeros((vertex_count,), dtype=jnp.int32),
        capacities,
    )
    problem = LinearCombinatorialProblem(
        space,
        costs,
        problem_id=f"offline-track-flow:{plan.plan_id}:"
        + canonical_fingerprint(tuple(item.reconstruction_id for item in sequence)),
    )
    method = CycleCancelingMinCostFlow(
        maximum_iterations=plan.maximum_iterations,
        maximum_vertices=vertex_count,
        maximum_edges=len(source_indices),
    )
    solved = solve_combinatorial(problem, method)
    flow = solved.decision.flow
    predecessor = jnp.full((observation_count,), -1, dtype=jnp.int32)
    successor = jnp.full((observation_count,), -1, dtype=jnp.int32)
    for edge, source_observation, target_observation in link_edges:
        predecessor = jax.lax.cond(
            flow[edge] > 0,
            lambda value, target=target_observation, source=source_observation: value.at[
                target
            ].set(source),
            lambda value: value,
            predecessor,
        )
        successor = jax.lax.cond(
            flow[edge] > 0,
            lambda value, source=source_observation, target=target_observation: value.at[
                source
            ].set(target),
            lambda value: value,
            successor,
        )
    selected_observations = flow[jnp.asarray(detection_edge, dtype=jnp.int32)] > 0
    refinement_id = "offline-tracks:" + canonical_fingerprint(
        {
            "sources": tuple(item.reconstruction_id for item in sequence),
            "plan": plan.plan_id,
        }
    )
    return OfflineTrackRefinementResult(
        predecessor.reshape((frame_count, observation_capacity)),
        successor.reshape((frame_count, observation_capacity)),
        selected_observations.reshape((frame_count, observation_capacity)),
        solved.status,
        solved.valid,
        solved,
        refinement_id,
    )


def to_trajectory_data(
    result: TrackResult,
    /,
    *,
    state_layout: StateLayout | None = None,
) -> TrajectoryData:
    """Convert track slots to padded trajectories without erasing gaps or ID resets."""
    if not isinstance(result, TrackResult):
        raise TypeError("result must be a TrackResult.")
    if result.times.shape[0] < 2:
        raise ValueError("TrajectoryData conversion requires at least two track times.")
    layout = (
        StateLayout(
            (6,),
            component_names=("x", "y", "z", "velocity_x", "velocity_y", "velocity_z"),
            layout_id="particle-track-xyz-velocity",
        )
        if state_layout is None
        else state_layout
    )
    if not isinstance(layout, StateLayout) or layout.shape != (6,):
        raise ValueError("state_layout must be a six-component StateLayout.")
    coordinates = jnp.broadcast_to(
        result.times[None, :], (result.track_capacity, result.times.shape[0])
    )
    transition_valid = result.observed[:, :-1] & result.observed[:, 1:] & ~result.resets
    position_variance = jnp.trace(result.covariances[:, :, :3, :3], axis1=-2, axis2=-1)
    weights = jnp.where(
        result.observed,
        1.0 / jnp.maximum(position_variance, 1e-12),
        0.0,
    )
    return TrajectoryData(
        coordinates,
        result.states,
        state_layout=layout,
        sample_valid=result.observed,
        transition_valid=transition_valid,
        reset_mask=result.resets,
        weights=weights,
        case_axes=("track",),
        case_axis_roles=("process",),
        coordinate_id="time",
        source_id=result.result_id,
        dataset_id=f"trajectory:{result.result_id}",
    )


__all__ = [
    "OfflineTrackRefinementPlan",
    "OfflineTrackRefinementResult",
    "TrackLinkPlan",
    "initialize_tracks",
    "link_tracks",
    "link_tracks_step",
    "refine_tracks_min_cost_flow",
    "to_trajectory_data",
]
