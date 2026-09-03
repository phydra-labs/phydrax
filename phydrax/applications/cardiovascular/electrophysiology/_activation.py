#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Online activation-time and directed chord-velocity observations."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntFlag
from math import isfinite

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


class ActivationObservationStatus(IntFlag):
    """Fail-closed status for one online activation observation update."""

    SUCCESS = 0
    NONFINITE_SAMPLE = 1
    NONFINITE_TIME = 2
    NONMONOTONE_TIME = 4
    INVALID_INTERPOLATION = 8


class ActivationObservationPlan(StrictModule, NonTrainableState):
    """Selected nodes and upward threshold for branchwise online LAT detection."""

    node_ids: Array
    node_id_tuple: tuple[int, ...] = eqx.field(static=True)
    node_count: int = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        node_count: int,
        node_ids: Sequence[int],
        /,
        *,
        threshold: float,
    ):
        if isinstance(node_count, bool) or not isinstance(node_count, int):
            raise TypeError("node_count must be an integer.")
        if node_count <= 0:
            raise ValueError("node_count must be positive.")
        ids = tuple(int(value) for value in node_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("node_ids must be nonempty and unique.")
        if any(value < 0 or value >= node_count for value in ids):
            raise ValueError("node_ids must index the declared node count.")
        threshold_ = float(threshold)
        if not isfinite(threshold_):
            raise ValueError("threshold must be finite.")
        self.node_ids = jnp.asarray(ids, dtype=jnp.int32)
        self.node_id_tuple = ids
        self.node_count = node_count
        self.threshold = threshold_
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-online-activation-plan",
                "node_count": node_count,
                "node_ids": list(ids),
                "threshold": threshold_,
                "crossing": "first-upward-linear-interpolation",
                "unactivated_representation": "nan-and-false",
                "time_unit": "ms",
            }
        )


class OnlineActivationState(StrictModule):
    """Fixed-shape committed observer state, separate from integrator arithmetic."""

    previous_activation: Array
    previous_time_ms: Array
    activation_times_ms: Array
    activated: Array
    update_count: Array
    observer_id: str = eqx.field(static=True)


class ActivationObservationEvidence(StrictModule):
    """Evidence for a candidate online LAT update."""

    newly_activated: Array
    interpolation_fraction: Array
    sample_finite: Array
    time_finite: Array
    time_monotone: Array
    interpolation_valid: Array
    status: Array
    successful: Array


class ActivationObservationCandidate(StrictModule):
    """Uncommitted observer update with source identity and evidence."""

    source: OnlineActivationState
    proposed: OnlineActivationState
    evidence: ActivationObservationEvidence


class ActivationObservationResult(StrictModule):
    """Current selected-node LATs; unactivated entries are censored as NaN."""

    node_ids: Array
    activation_times_ms: Array
    activated: Array
    status: Array
    successful: Array
    observer_id: str = eqx.field(static=True)


def initialize_activation_observation(
    plan: ActivationObservationPlan,
    activation: ArrayLike,
    /,
    *,
    time_ms: float = 0.0,
) -> OnlineActivationState:
    """Initialize online LAT state, counting initially supra-threshold nodes at t0."""

    if not isinstance(plan, ActivationObservationPlan):
        raise TypeError("plan must be an ActivationObservationPlan.")
    values = jnp.asarray(activation)
    if values.shape != (plan.node_count,):
        raise ValueError(f"activation must have shape ({plan.node_count},).")
    if not jnp.issubdtype(values.dtype, jnp.inexact):
        values = values.astype(float)
    if not bool(np.all(np.isfinite(np.asarray(values)))):
        raise ValueError("Initial activation must be finite.")
    time = float(time_ms)
    if not isfinite(time):
        raise ValueError("time_ms must be finite.")
    selected = values[plan.node_ids]
    activated = selected >= plan.threshold
    lat = jnp.where(activated, jnp.asarray(time, dtype=values.dtype), jnp.nan)
    return OnlineActivationState(
        selected,
        jnp.asarray(time, dtype=values.dtype),
        lat,
        activated,
        jnp.asarray(0, dtype=jnp.int32),
        plan.plan_id,
    )


def evaluate_activation_observation(
    plan: ActivationObservationPlan,
    state: OnlineActivationState,
    activation: ArrayLike,
    time_ms: ArrayLike,
    /,
) -> ActivationObservationCandidate:
    """Propose first upward crossings by branchwise linear interpolation."""

    if not isinstance(plan, ActivationObservationPlan):
        raise TypeError("plan must be an ActivationObservationPlan.")
    if not isinstance(state, OnlineActivationState):
        raise TypeError("state must be an OnlineActivationState.")
    if state.observer_id != plan.plan_id:
        raise ValueError("Online activation state does not match its plan.")
    values = jnp.asarray(activation)
    if values.shape != (plan.node_count,):
        raise ValueError(f"activation must have shape ({plan.node_count},).")
    selected = values[plan.node_ids].astype(state.previous_activation.dtype)
    time = jnp.asarray(time_ms, dtype=state.previous_time_ms.dtype)
    if time.shape != ():
        raise ValueError("time_ms must be scalar.")
    delta_time = time - state.previous_time_ms
    delta_activation = selected - state.previous_activation
    crossing = (
        (~state.activated)
        & (state.previous_activation < plan.threshold)
        & (selected >= plan.threshold)
    )
    safe_delta = jnp.where(crossing, delta_activation, jnp.ones_like(delta_activation))
    fraction = jnp.where(
        crossing,
        (plan.threshold - state.previous_activation) / safe_delta,
        jnp.zeros_like(delta_activation),
    )
    interpolation_valid = jnp.all(
        (~crossing)
        | (
            jnp.isfinite(fraction)
            & (fraction >= 0.0)
            & (fraction <= 1.0)
            & (delta_activation > 0.0)
        )
    )
    crossing_times = state.previous_time_ms + fraction * delta_time
    proposed_times = jnp.where(crossing, crossing_times, state.activation_times_ms)
    proposed_activated = state.activated | crossing
    sample_finite = jnp.all(jnp.isfinite(selected))
    time_finite = jnp.isfinite(time)
    time_monotone = time > state.previous_time_ms
    status = jnp.asarray(int(ActivationObservationStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        sample_finite,
        status,
        jnp.bitwise_or(status, int(ActivationObservationStatus.NONFINITE_SAMPLE)),
    )
    status = jnp.where(
        time_finite,
        status,
        jnp.bitwise_or(status, int(ActivationObservationStatus.NONFINITE_TIME)),
    )
    status = jnp.where(
        time_monotone,
        status,
        jnp.bitwise_or(status, int(ActivationObservationStatus.NONMONOTONE_TIME)),
    )
    status = jnp.where(
        interpolation_valid,
        status,
        jnp.bitwise_or(status, int(ActivationObservationStatus.INVALID_INTERPOLATION)),
    )
    successful = status == int(ActivationObservationStatus.SUCCESS)
    proposed = OnlineActivationState(
        selected,
        time,
        proposed_times,
        proposed_activated,
        state.update_count + jnp.asarray(1, dtype=jnp.int32),
        state.observer_id,
    )
    return ActivationObservationCandidate(
        state,
        proposed,
        ActivationObservationEvidence(
            crossing,
            fraction,
            sample_finite,
            time_finite,
            time_monotone,
            interpolation_valid,
            status,
            successful,
        ),
    )


def commit_activation_observation(
    candidate: ActivationObservationCandidate,
    current: OnlineActivationState,
    /,
) -> OnlineActivationState:
    """Commit a valid candidate from exactly ``current`` or preserve ``current``."""

    if not isinstance(candidate, ActivationObservationCandidate):
        raise TypeError("candidate must be an ActivationObservationCandidate.")
    if not isinstance(current, OnlineActivationState):
        raise TypeError("current must be an OnlineActivationState.")
    source_matches = (
        (candidate.source.observer_id == current.observer_id)
        & jnp.array_equal(
            candidate.source.previous_activation, current.previous_activation
        )
        & jnp.array_equal(
            candidate.source.activation_times_ms,
            current.activation_times_ms,
            equal_nan=True,
        )
        & jnp.array_equal(candidate.source.activated, current.activated)
        & (candidate.source.previous_time_ms == current.previous_time_ms)
        & (candidate.source.update_count == current.update_count)
    )
    return jax.lax.cond(
        candidate.evidence.successful & source_matches,
        lambda _: candidate.proposed,
        lambda _: current,
        operand=None,
    )


def activation_observation_result(
    plan: ActivationObservationPlan,
    state: OnlineActivationState,
    /,
) -> ActivationObservationResult:
    """Freeze the current online state into its public observation result."""

    if (
        state.observer_id != plan.plan_id
        or state.activation_times_ms.shape != plan.node_ids.shape
    ):
        raise ValueError("Online activation state does not match its plan.")
    status = jnp.asarray(int(ActivationObservationStatus.SUCCESS), dtype=jnp.int32)
    finite_activated = jnp.all(
        jnp.where(state.activated, jnp.isfinite(state.activation_times_ms), True)
    )
    status = jnp.where(
        finite_activated,
        status,
        jnp.bitwise_or(status, int(ActivationObservationStatus.INVALID_INTERPOLATION)),
    )
    return ActivationObservationResult(
        plan.node_ids,
        jnp.where(state.activated, state.activation_times_ms, jnp.nan),
        state.activated,
        status,
        status == int(ActivationObservationStatus.SUCCESS),
        plan.plan_id,
    )


class ChordConductionVelocityStatus(IntFlag):
    """Status for one directed chord conduction-velocity observation."""

    SUCCESS = 0
    ACTIVATION_OBSERVATION_FAILURE = 1
    ENDPOINT_UNACTIVATED = 2
    NONPOSITIVE_TRANSIT_TIME = 4
    NONFINITE_RESULT = 8


class ChordConductionVelocityPlan(StrictModule, NonTrainableState):
    """Directed source-to-target chord bound to an activation observation plan."""

    source_node_id: int = eqx.field(static=True)
    target_node_id: int = eqx.field(static=True)
    source_index: int = eqx.field(static=True)
    target_index: int = eqx.field(static=True)
    distance_mm: float = eqx.field(static=True)
    activation_plan_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        activation_plan: ActivationObservationPlan,
        source_node_id: int,
        target_node_id: int,
        distance_mm: float,
        /,
    ):
        if not isinstance(activation_plan, ActivationObservationPlan):
            raise TypeError("activation_plan must be an ActivationObservationPlan.")
        source = int(source_node_id)
        target = int(target_node_id)
        if source == target:
            raise ValueError("Chord endpoints must be distinct.")
        if (
            source not in activation_plan.node_id_tuple
            or target not in activation_plan.node_id_tuple
        ):
            raise ValueError("Chord endpoints must be selected by the activation plan.")
        distance = float(distance_mm)
        if not isfinite(distance) or distance <= 0.0:
            raise ValueError("distance_mm must be finite and positive.")
        self.source_node_id = source
        self.target_node_id = target
        self.source_index = activation_plan.node_id_tuple.index(source)
        self.target_index = activation_plan.node_id_tuple.index(target)
        self.distance_mm = distance
        self.activation_plan_id = activation_plan.plan_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-directed-chord-cv-plan",
                "activation_plan": activation_plan.plan_id,
                "source_node_id": source,
                "target_node_id": target,
                "distance_mm": distance,
                "velocity_unit": "mm/ms",
            }
        )

    @classmethod
    def from_coordinates(
        cls,
        activation_plan: ActivationObservationPlan,
        coordinates_mm: ArrayLike,
        source_node_id: int,
        target_node_id: int,
        /,
    ) -> ChordConductionVelocityPlan:
        coordinates = np.asarray(coordinates_mm, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[0] != activation_plan.node_count:
            raise ValueError("coordinates_mm must provide one point per mesh node.")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates_mm must be finite.")
        displacement = coordinates[int(target_node_id)] - coordinates[int(source_node_id)]
        distance = float(np.sqrt(np.sum(displacement * displacement)))
        return cls(activation_plan, source_node_id, target_node_id, distance)


class ChordConductionVelocityResult(StrictModule):
    """Directed chord transit time and speed in kernel units."""

    transit_time_ms: Array
    velocity_mm_per_ms: Array
    endpoints_activated: Array
    status: Array
    successful: Array
    chord_id: str = eqx.field(static=True)


def evaluate_chord_conduction_velocity(
    plan: ChordConductionVelocityPlan,
    activation: ActivationObservationResult,
    /,
) -> ChordConductionVelocityResult:
    """Evaluate chord velocity after online LAT observation, never inside SSPRK."""

    if not isinstance(plan, ChordConductionVelocityPlan):
        raise TypeError("plan must be a ChordConductionVelocityPlan.")
    if not isinstance(activation, ActivationObservationResult):
        raise TypeError("activation must be an ActivationObservationResult.")
    if activation.observer_id != plan.activation_plan_id:
        raise ValueError("Activation observation does not match the chord plan.")
    source_activated = activation.activated[plan.source_index]
    target_activated = activation.activated[plan.target_index]
    endpoints = source_activated & target_activated
    transit = (
        activation.activation_times_ms[plan.target_index]
        - activation.activation_times_ms[plan.source_index]
    )
    positive = transit > 0.0
    safe_transit = jnp.where(endpoints & positive, transit, jnp.ones_like(transit))
    velocity = plan.distance_mm / safe_transit
    finite = jnp.isfinite(transit) & jnp.isfinite(velocity)
    status = jnp.asarray(int(ChordConductionVelocityStatus.SUCCESS), dtype=jnp.int32)
    status = jnp.where(
        activation.successful,
        status,
        jnp.bitwise_or(
            status, int(ChordConductionVelocityStatus.ACTIVATION_OBSERVATION_FAILURE)
        ),
    )
    status = jnp.where(
        endpoints,
        status,
        jnp.bitwise_or(status, int(ChordConductionVelocityStatus.ENDPOINT_UNACTIVATED)),
    )
    status = jnp.where(
        (~endpoints) | positive,
        status,
        jnp.bitwise_or(
            status, int(ChordConductionVelocityStatus.NONPOSITIVE_TRANSIT_TIME)
        ),
    )
    status = jnp.where(
        (~endpoints) | finite,
        status,
        jnp.bitwise_or(status, int(ChordConductionVelocityStatus.NONFINITE_RESULT)),
    )
    successful = status == int(ChordConductionVelocityStatus.SUCCESS)
    return ChordConductionVelocityResult(
        jnp.where(successful, transit, jnp.nan),
        jnp.where(successful, velocity, jnp.nan),
        endpoints,
        status,
        successful,
        plan.plan_id,
    )


__all__ = [
    "ActivationObservationCandidate",
    "ActivationObservationEvidence",
    "ActivationObservationPlan",
    "ActivationObservationResult",
    "ActivationObservationStatus",
    "ChordConductionVelocityPlan",
    "ChordConductionVelocityResult",
    "ChordConductionVelocityStatus",
    "OnlineActivationState",
    "activation_observation_result",
    "commit_activation_observation",
    "evaluate_activation_observation",
    "evaluate_chord_conduction_velocity",
    "initialize_activation_observation",
]
