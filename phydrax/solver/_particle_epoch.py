#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization.particle._dem import (
    DEMResolvedLoad,
    DEMRuntimeState,
    DEMStepEvaluation,
)
from ..discretization.particle._dem_contact_state import (
    DEMContactHistory,
    DEMRotationalHistory,
    DEMTangentialHistory,
)
from ..discretization.particle._pair_state import (
    match_particle_pair_keys,
    remap_particle_pair_values,
)
from ..discretization.particle._particle_epoch import (
    grow_particle_execution_epoch,
    ParticleCapacityGrowthPolicy,
    ParticleCapacityRequest,
    ParticleEpochTransition,
    ParticleExecutionEpoch,
)
from ..discretization.particle._particle_morphology import ParticleDynamicBodyProperties
from ..discretization.particle._rigid_sphere import RigidSphereKinematics, RigidSphereLoad


class ParticleEpochSegmentRecord(StrictModule, NonTrainableState):
    epoch_id: str = eqx.field(static=True)
    start_step: int = eqx.field(static=True)
    end_step: int = eqx.field(static=True)
    accepted: Array
    route_digest: Array
    successful: Array
    record_id: str = eqx.field(static=True)


class ParticleEpochTrajectory(StrictModule):
    final_epoch: ParticleExecutionEpoch
    segments: tuple[ParticleEpochSegmentRecord, ...]
    transitions: tuple[ParticleEpochTransition, ...]
    successful: Array
    trajectory_id: str = eqx.field(static=True)


def _route_digest(state: DEMRuntimeState, /) -> Array:
    identity = jnp.where(
        state.particle_history.valid[:, None], state.particle_history.pair_keys + 1, 0
    )
    weights = jnp.asarray([3, 5, 7, 11, 13], dtype=jnp.int64)
    pair_digest = jnp.sum(identity * weights[None, :])
    active_digest = jnp.sum(
        state.body_properties.active.astype(jnp.int64)
        * (jnp.arange(state.body_properties.active.shape[0], dtype=jnp.int64) + 17)
    )
    return pair_digest + 31 * active_digest


def advance_particle_epoch_segments(
    initial_epoch: ParticleExecutionEpoch,
    step_function: Callable[[ParticleExecutionEpoch, int], DEMStepEvaluation],
    segment_steps: Sequence[int],
    /,
    *,
    growth_policy: ParticleCapacityGrowthPolicy | None = None,
    growth_requests: Sequence[ParticleCapacityRequest | None] = (),
    transition_time: Array | None = None,
) -> ParticleEpochTrajectory:
    if not isinstance(initial_epoch, ParticleExecutionEpoch):
        raise TypeError("initial_epoch must be ParticleExecutionEpoch.")
    if not callable(step_function):
        raise TypeError("step_function must be callable.")
    counts = tuple(int(value) for value in segment_steps)
    if not counts or any(value < 0 for value in counts):
        raise ValueError("segment_steps must contain nonnegative counts.")
    requests = tuple(growth_requests)
    if requests and len(requests) != len(counts) - 1:
        raise ValueError("growth_requests must describe each segment boundary.")
    if requests and growth_policy is None:
        raise ValueError("growth_policy is required when growth requests are supplied.")
    transition_time_ = (
        jnp.asarray(0.0) if transition_time is None else jnp.asarray(transition_time)
    )
    epoch = initial_epoch
    segments = []
    transitions = []
    successful = jnp.asarray(True)
    global_step = 0
    for segment_index, count in enumerate(counts):
        accepted = jnp.zeros((count,), dtype=bool)
        segment_successful = jnp.asarray(True)
        for local_step in range(count):
            result = step_function(epoch, global_step)
            if not isinstance(result, DEMStepEvaluation):
                raise TypeError("step_function must return DEMStepEvaluation.")
            epoch = ParticleExecutionEpoch(
                epoch.dynamics,
                result.accepted_state,
                epoch.ever_occupied,
                epoch.retired,
                epoch.epoch_index,
                epoch.epoch_id,
            )
            accepted = accepted.at[local_step].set(result.successful)
            segment_successful = segment_successful & result.successful
            global_step += 1
        digest = _route_digest(epoch.state)
        record_id = canonical_fingerprint(
            {
                "kind": "particle-epoch-segment-record",
                "epoch": epoch.epoch_id,
                "start": global_step - count,
                "end": global_step,
            }
        )
        segments.append(
            ParticleEpochSegmentRecord(
                epoch.epoch_id,
                global_step - count,
                global_step,
                accepted,
                digest,
                segment_successful,
                record_id,
            )
        )
        successful = successful & segment_successful
        if segment_index < len(counts) - 1 and requests:
            request = requests[segment_index]
            if request is not None:
                transition = grow_particle_execution_epoch(
                    epoch,
                    growth_policy,
                    request,
                    transition_time_,
                )
                transitions.append(transition)
                epoch = transition.accepted_epoch
                successful = successful & transition.successful
    trajectory_id = canonical_fingerprint(
        {
            "kind": "particle-epoch-trajectory",
            "initial": initial_epoch.epoch_id,
            "segments": [value.record_id for value in segments],
            "transitions": [value.transition_id for value in transitions],
        }
    )
    return ParticleEpochTrajectory(
        epoch,
        tuple(segments),
        tuple(transitions),
        successful,
        trajectory_id,
    )


def _pullback_history(
    source: DEMContactHistory,
    target: DEMContactHistory,
    cotangent: DEMContactHistory,
    /,
) -> DEMContactHistory:
    remap = match_particle_pair_keys(
        target.pair_keys,
        target.valid,
        source.pair_keys,
        source.valid,
    )
    active, normal, cohesion, tangential, rotational = remap_particle_pair_values(
        remap, cotangent.values
    )
    return DEMContactHistory(
        source.pair_keys,
        source.valid,
        active,
        normal,
        cohesion,
        DEMTangentialHistory(
            tangential.sliding,
            tangential.previous_normal,
            tangential.displacement,
        ),
        DEMRotationalHistory(
            rotational.rolling_displacement,
            rotational.torsional_displacement,
            rotational.previous_normal,
            rotational.rolling_yielded,
            rotational.torsional_yielded,
        ),
    )


def _slice_load(load: RigidSphereLoad, capacity: int, /) -> RigidSphereLoad:
    return RigidSphereLoad(load.force[:capacity], load.torque[:capacity])


def pullback_particle_epoch_transition(
    transition: ParticleEpochTransition,
    cotangent: DEMRuntimeState,
    /,
) -> DEMRuntimeState:
    if not isinstance(transition, ParticleEpochTransition):
        raise TypeError("transition must be ParticleEpochTransition.")
    if not isinstance(cotangent, DEMRuntimeState):
        raise TypeError("cotangent must be DEMRuntimeState.")
    source = transition.source_epoch.state
    target = transition.candidate_epoch.state
    capacity = transition.old_to_new.shape[0]
    kinematics = RigidSphereKinematics(
        cotangent.kinematics.position[:capacity],
        cotangent.kinematics.velocity[:capacity],
        cotangent.kinematics.angular_velocity[:capacity],
    )
    properties = ParticleDynamicBodyProperties(
        cotangent.body_properties.masses[:capacity],
        cotangent.body_properties.inverse_masses[:capacity],
        cotangent.body_properties.radii[:capacity],
        cotangent.body_properties.inertias[:capacity],
        cotangent.body_properties.inverse_inertias[:capacity],
        cotangent.body_properties.active[:capacity],
    )
    particle_history = _pullback_history(
        source.particle_history,
        target.particle_history,
        cotangent.particle_history,
    )
    boundary_histories = tuple(
        _pullback_history(source_history, target_history, cotangent_history)
        for source_history, target_history, cotangent_history in zip(
            source.boundary_histories,
            target.boundary_histories,
            cotangent.boundary_histories,
            strict=True,
        )
    )
    loads = DEMResolvedLoad(
        _slice_load(cotangent.loads.particle_contact, capacity),
        tuple(_slice_load(value, capacity) for value in cotangent.loads.boundaries),
        _slice_load(cotangent.loads.gravity, capacity),
        _slice_load(cotangent.loads.external, capacity),
        _slice_load(cotangent.loads.total, capacity),
    )
    return DEMRuntimeState(
        kinematics,
        properties,
        particle_history,
        boundary_histories,
        None,
        loads,
        cotangent.energy,
    )


def segmented_particle_epoch_vjp(
    transition_pullbacks: Sequence[Callable[[Any], Any]],
    transitions: Sequence[ParticleEpochTransition],
    terminal_cotangent: DEMRuntimeState,
    /,
):
    pullbacks = tuple(transition_pullbacks)
    transition_values = tuple(transitions)
    if len(pullbacks) != len(transition_values) + 1:
        raise ValueError("One segment pullback is required around every transition.")
    cotangent = pullbacks[-1](terminal_cotangent)
    for index in range(len(transition_values) - 1, -1, -1):
        cotangent = pullback_particle_epoch_transition(
            transition_values[index], cotangent
        )
        cotangent = pullbacks[index](cotangent)
    return cotangent


__all__ = [
    "ParticleEpochSegmentRecord",
    "ParticleEpochTrajectory",
    "advance_particle_epoch_segments",
    "pullback_particle_epoch_transition",
    "segmented_particle_epoch_vjp",
]
