#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..camera._rig import CameraRig
from ..imaging._photometry import CameraStackRenderResult, ParticleImageFormation
from ..imaging._types import ImageGeometry2D
from ._ipr import (
    IPR_CAPACITY_EXHAUSTED,
    IPR_NONFINITE,
    IPR_SUBSET_REJECTED,
    IPRPlan,
    IPRResult,
    iterative_particle_reconstruction,
)
from ._shake import shake_particles, ShakePlan, ShakeResult
from ._tracks import (
    initialize_tracks,
    link_tracks_step,
    TrackLinkPlan,
)
from ._types import TrackRuntimeState, TrackStepResult


STB_SUCCESS = 0
STB_SUBSET_REJECTED = 1
STB_CAPACITY_EXHAUSTED = 2
STB_NONFINITE = 3


class STBPlan(StrictModule, NonTrainableState):
    """Fixed-capacity streaming Shake-the-Box lifecycle policy."""

    ipr: IPRPlan
    shake: ShakePlan
    track_link: TrackLinkPlan
    promotion_steps: int = eqx.field(static=True)
    minimum_active_amplitude: float = eqx.field(static=True)
    observation_variance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ipr: IPRPlan,
        shake: ShakePlan,
        track_link: TrackLinkPlan,
        *,
        promotion_steps: int = 2,
        minimum_active_amplitude: float = 0.0,
        observation_variance: float = 1.0e-4,
    ):
        if not isinstance(ipr, IPRPlan):
            raise TypeError("ipr must be IPRPlan.")
        if not isinstance(shake, ShakePlan):
            raise TypeError("shake must be ShakePlan.")
        if not isinstance(track_link, TrackLinkPlan):
            raise TypeError("track_link must be TrackLinkPlan.")
        if track_link.maximum_tracks != ipr.particle_capacity:
            raise ValueError("Track and IPR particle capacities must match.")
        promotion = int(promotion_steps)
        minimum_amplitude = float(minimum_active_amplitude)
        variance = float(observation_variance)
        if promotion <= 0:
            raise ValueError("promotion_steps must be positive.")
        if not isfinite(minimum_amplitude) or minimum_amplitude < 0.0:
            raise ValueError("minimum_active_amplitude must be finite and non-negative.")
        if not isfinite(variance) or variance <= 0.0:
            raise ValueError("observation_variance must be finite and positive.")
        self.ipr = ipr
        self.shake = shake
        self.track_link = track_link
        self.promotion_steps = promotion
        self.minimum_active_amplitude = minimum_amplitude
        self.observation_variance = variance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fixed-capacity-shake-the-box",
                "ipr_plan_id": ipr.plan_id,
                "shake_plan_id": shake.plan_id,
                "track_link_plan_id": track_link.plan_id,
                "promotion_steps": promotion,
                "minimum_active_amplitude": minimum_amplitude,
                "observation_variance": variance,
            }
        )


class PreparedSTB(StrictModule, NonTrainableState):
    """STB policy bound to one rig, image geometry, and fixed PSF widths."""

    plan: STBPlan
    formation: ParticleImageFormation
    rig: CameraRig
    geometry: ImageGeometry2D
    sigma: Array
    prepared_id: str = eqx.field(static=True)


class STBState(StrictModule):
    """Fixed particle slots plus a separate identity-bearing tracking state."""

    positions_xyz: Array
    velocities_xyz: Array
    amplitude: Array
    active: Array
    slot_ids: Array
    candidate_track_ids: Array
    track_ids: Array
    hit_count: Array
    missed_count: Array
    tracking: TrackRuntimeState
    time: Array
    frame_index: Array
    plan_id: str = eqx.field(static=True)


class STBStepEvidence(StrictModule):
    """Streaming birth, promotion, termination, and capacity evidence."""

    born: Array
    promoted: Array
    terminated: Array
    matched: Array
    born_count: Array
    promoted_count: Array
    terminated_count: Array
    active_count: Array
    capacity_rejected_count: Array
    subset_rejected_count: Array
    status: Array


class STBStepResult(StrictModule):
    """One prediction-render-Shake-IPR-lifecycle transaction."""

    state: STBState
    predicted_render: CameraStackRenderResult
    shake: ShakeResult
    ipr: IPRResult
    tracking: TrackStepResult
    residual: Array
    evidence: STBStepEvidence
    successful: Array


class STBResult(StrictModule):
    """Audited fixed-capacity streaming history."""

    final_state: STBState
    steps: tuple[STBStepResult, ...]
    positions_xyz: Array
    velocities_xyz: Array
    amplitude: Array
    active: Array
    track_ids: Array
    residual_loss: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def prepare_stb(
    plan: STBPlan,
    formation: ParticleImageFormation,
    rig: CameraRig,
    geometry: ImageGeometry2D,
    sigma: ArrayLike,
    /,
) -> PreparedSTB:
    """Bind STB to calibrated image formation and fixed particle support width."""
    if not isinstance(plan, STBPlan):
        raise TypeError("plan must be STBPlan.")
    if not isinstance(formation, ParticleImageFormation):
        raise TypeError("formation must be ParticleImageFormation.")
    if formation.response.stochastic:
        raise ValueError("STB inference requires deterministic image formation.")
    if not isinstance(rig, CameraRig):
        raise TypeError("rig must be CameraRig.")
    if not isinstance(geometry, ImageGeometry2D):
        raise TypeError("geometry must be ImageGeometry2D.")
    sigma_ = jnp.asarray(sigma)
    capacity = plan.ipr.particle_capacity
    if sigma_.ndim == 0:
        sigma_ = jnp.broadcast_to(sigma_, (capacity,))
    elif sigma_.shape not in ((capacity,), (capacity, 2)):
        raise ValueError(
            "sigma must be scalar, (particle_capacity,), or (particle_capacity, 2)."
        )
    prepared_id = canonical_fingerprint(
        {
            "kind": "prepared-shake-the-box",
            "plan_id": plan.plan_id,
            "formation_id": formation.formation_id,
            "rig_id": rig.rig_id,
            "geometry_id": geometry.geometry_id,
            "sigma_shape": tuple(sigma_.shape),
        }
    )
    return PreparedSTB(plan, formation, rig, geometry, sigma_, prepared_id)


def initialize_stb(
    prepared: PreparedSTB,
    positions_xyz: ArrayLike,
    amplitude: ArrayLike,
    active: ArrayLike,
    *,
    velocities_xyz: ArrayLike | None = None,
    first_track_id: int = 0,
    initial_time: float = 0.0,
) -> STBState:
    """Initialize particle slots and the independent tracking identity state."""
    if not isinstance(prepared, PreparedSTB):
        raise TypeError("prepared must be PreparedSTB.")
    capacity = prepared.plan.ipr.particle_capacity
    positions = jnp.asarray(positions_xyz)
    if positions.shape != (capacity, 3):
        raise ValueError("positions_xyz must have shape (particle_capacity, 3).")
    if not jnp.issubdtype(positions.dtype, jnp.inexact):
        positions = positions.astype(float)
    amplitudes = jnp.asarray(amplitude, dtype=positions.dtype)
    active_ = jnp.asarray(active, dtype=bool)
    if amplitudes.shape != (capacity,) or active_.shape != (capacity,):
        raise ValueError("amplitude and active must match particle_capacity.")
    velocities = (
        jnp.zeros_like(positions)
        if velocities_xyz is None
        else jnp.asarray(velocities_xyz, dtype=positions.dtype)
    )
    if velocities.shape != positions.shape:
        raise ValueError("velocities_xyz must have the positions shape.")
    invalid_active = active_ & (
        ~jnp.all(jnp.isfinite(positions), axis=-1)
        | ~jnp.all(jnp.isfinite(velocities), axis=-1)
        | ~jnp.isfinite(amplitudes)
        | (amplitudes < 0.0)
    )
    if bool(jnp.any(invalid_active)):
        raise ValueError(
            "Active STB particles must have finite state and non-negative amplitude."
        )
    initial_time_ = float(initial_time)
    if not isfinite(initial_time_):
        raise ValueError("initial_time must be finite.")
    tracking = initialize_tracks(
        prepared.plan.track_link,
        dtype=positions.dtype,
        first_track_id=first_track_id,
    )
    covariance = prepared.plan.observation_variance * jnp.broadcast_to(
        jnp.eye(3, dtype=positions.dtype), (capacity, 3, 3)
    )
    tracking_step = link_tracks_step(
        tracking,
        positions,
        covariance,
        active_,
        jnp.asarray(initial_time_, dtype=positions.dtype),
        prepared.plan.track_link,
    )
    identities = tracking_step.observation_track_ids
    return STBState(
        jnp.where(active_[:, None], positions, 0.0),
        jnp.where(active_[:, None], velocities, 0.0),
        jnp.where(active_, amplitudes, 0.0),
        active_,
        jnp.arange(capacity, dtype=jnp.int32),
        identities,
        identities,
        jnp.where(active_, prepared.plan.promotion_steps, 0).astype(jnp.int32),
        jnp.zeros((capacity,), dtype=jnp.int32),
        tracking_step.state,
        jnp.asarray(initial_time_, dtype=positions.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        prepared.plan.plan_id,
    )


def stb_step(
    prepared: PreparedSTB,
    state: STBState,
    observed_images: ArrayLike,
    time: ArrayLike,
    *,
    valid_mask: ArrayLike | None = None,
) -> STBStepResult:
    """Advance one point-tracer STB frame with deterministic slot lifecycle."""
    if not isinstance(prepared, PreparedSTB):
        raise TypeError("prepared must be PreparedSTB.")
    if not isinstance(state, STBState):
        raise TypeError("state must be STBState.")
    if state.plan_id != prepared.plan.plan_id:
        raise ValueError("state belongs to a different STB plan.")
    time_ = jnp.asarray(time, dtype=state.time.dtype)
    delta_time = time_ - state.time
    if time_.shape != ():
        raise ValueError("time must be scalar.")
    predicted_positions = state.positions_xyz + delta_time * state.velocities_xyz
    predicted_positions = jnp.where(
        state.active[:, None], predicted_positions, state.positions_xyz
    )
    shake = shake_particles(
        prepared.plan.shake,
        prepared.formation,
        prepared.rig,
        prepared.geometry,
        observed_images,
        predicted_positions,
        state.amplitude,
        prepared.sigma,
        state.active,
        valid_mask=valid_mask,
    )
    ipr = iterative_particle_reconstruction(
        prepared.plan.ipr,
        prepared.formation,
        prepared.rig,
        prepared.geometry,
        observed_images,
        shake.positions_xyz,
        shake.amplitude,
        prepared.sigma,
        shake.active,
        valid_mask=valid_mask,
    )
    observation_valid = ipr.active & (
        ipr.amplitude >= prepared.plan.minimum_active_amplitude
    )
    observation_covariance = prepared.plan.observation_variance * jnp.broadcast_to(
        jnp.eye(3, dtype=ipr.positions_xyz.dtype),
        (prepared.plan.ipr.particle_capacity, 3, 3),
    )
    tracking = link_tracks_step(
        state.tracking,
        ipr.positions_xyz,
        observation_covariance,
        observation_valid,
        time_,
        prepared.plan.track_link,
    )
    linked_ids = tracking.observation_track_ids
    continued = (linked_ids >= 0) & (linked_ids == state.candidate_track_ids)
    candidate_track_ids = jnp.where(
        linked_ids >= 0, linked_ids, state.candidate_track_ids
    )
    track_identity_match = (
        candidate_track_ids[:, None] == tracking.state.track_ids[None, :]
    ) & tracking.state.active[None, :]
    previous_survives = state.active & jnp.any(track_identity_match, axis=1)
    final_active = observation_valid | previous_survives
    born = observation_valid & jnp.any(
        (linked_ids[:, None] == tracking.state.track_ids[None, :])
        & tracking.births[None, :],
        axis=1,
    )
    hit_count = jnp.where(
        linked_ids >= 0,
        jnp.where(continued, state.hit_count + 1, 1),
        state.hit_count,
    ).astype(jnp.int32)
    missed_count = jnp.sum(
        jnp.where(
            track_identity_match,
            tracking.state.missed[None, :],
            0,
        ),
        axis=1,
        dtype=jnp.int32,
    )
    terminated = state.active & ~final_active
    retained_identity = continued | ((linked_ids < 0) & previous_survives)
    track_ids = jnp.where(retained_identity, state.track_ids, -1)
    promote = (
        final_active
        & (track_ids < 0)
        & (candidate_track_ids >= 0)
        & (hit_count >= prepared.plan.promotion_steps)
    )
    track_ids = jnp.where(promote, candidate_track_ids, track_ids)
    track_ids = jnp.where(terminated, -1, track_ids)
    candidate_track_ids = jnp.where(final_active, candidate_track_ids, -1)
    track_position = jnp.sum(
        jnp.where(
            track_identity_match[:, :, None],
            tracking.state.states[None, :, :3],
            0.0,
        ),
        axis=1,
    )
    track_velocity = jnp.sum(
        jnp.where(
            track_identity_match[:, :, None],
            tracking.state.states[None, :, 3:],
            0.0,
        ),
        axis=1,
    )
    positions = jnp.where(observation_valid[:, None], ipr.positions_xyz, track_position)
    positions = jnp.where(final_active[:, None], positions, 0.0)
    velocities = jnp.where(final_active[:, None], track_velocity, 0.0)
    amplitudes = jnp.where(observation_valid, ipr.amplitude, state.amplitude)
    amplitudes = jnp.where(final_active, amplitudes, 0.0)
    hit_count = jnp.where(final_active, hit_count, 0)
    missed_count = jnp.where(final_active, missed_count, 0)
    tracking_capacity_count = tracking.evidence.overflow_count
    capacity_count = ipr.capacity_rejected_count + tracking_capacity_count
    subset_count = ipr.subset_rejected_count
    finite = (
        jnp.isfinite(time_)
        & (delta_time > 0.0)
        & jnp.all(jnp.isfinite(positions))
        & jnp.all(jnp.isfinite(amplitudes))
        & (ipr.status != IPR_NONFINITE)
    )
    status = jnp.where(
        ~finite,
        STB_NONFINITE,
        jnp.where(
            (ipr.status == IPR_CAPACITY_EXHAUSTED) | (capacity_count > 0),
            STB_CAPACITY_EXHAUSTED,
            jnp.where(
                (ipr.status == IPR_SUBSET_REJECTED) | (subset_count > 0),
                STB_SUBSET_REJECTED,
                STB_SUCCESS,
            ),
        ),
    ).astype(jnp.int32)
    next_state = STBState(
        positions,
        velocities,
        amplitudes,
        final_active,
        state.slot_ids,
        candidate_track_ids,
        track_ids,
        hit_count,
        missed_count,
        tracking.state,
        time_,
        state.frame_index + 1,
        state.plan_id,
    )
    evidence = STBStepEvidence(
        born,
        promote,
        terminated,
        observation_valid,
        jnp.sum(born, dtype=jnp.int32),
        jnp.sum(promote, dtype=jnp.int32),
        jnp.sum(terminated, dtype=jnp.int32),
        jnp.sum(final_active, dtype=jnp.int32),
        capacity_count,
        subset_count,
        status,
    )
    return STBStepResult(
        next_state,
        shake.initial_render,
        shake,
        ipr,
        tracking,
        ipr.residual,
        evidence,
        finite & ipr.successful & (tracking_capacity_count == 0),
    )


def run_stb(
    prepared: PreparedSTB,
    initial_state: STBState,
    observed_images: ArrayLike,
    times: ArrayLike,
    *,
    valid_masks: ArrayLike | None = None,
) -> STBResult:
    """Run the streaming transaction over a fixed frame sequence."""
    if not isinstance(prepared, PreparedSTB):
        raise TypeError("prepared must be PreparedSTB.")
    if not isinstance(initial_state, STBState):
        raise TypeError("initial_state must be STBState.")
    if initial_state.plan_id != prepared.plan.plan_id:
        raise ValueError("initial_state belongs to a different STB plan.")
    images = jnp.asarray(observed_images)
    times_ = jnp.asarray(times)
    if (
        images.ndim != 4
        or images.shape[1:] != (prepared.rig.capacity,) + prepared.geometry.image_shape
    ):
        raise ValueError("observed_images must have shape (time, camera, rows, columns).")
    if images.shape[0] == 0:
        raise ValueError("STB requires at least one image frame.")
    if times_.shape != (images.shape[0],):
        raise ValueError("times must have one entry per image frame.")
    masks = (
        jnp.ones(images.shape, dtype=bool)
        if valid_masks is None
        else jnp.asarray(valid_masks, dtype=bool)
    )
    if masks.shape != images.shape:
        raise ValueError("valid_masks must have the observed image sequence shape.")
    state = initial_state
    results: list[STBStepResult] = []
    for index in range(images.shape[0]):
        result = stb_step(
            prepared,
            state,
            images[index],
            times_[index],
            valid_mask=masks[index],
        )
        results.append(result)
        state = result.state
    positions = jnp.stack(tuple(result.state.positions_xyz for result in results), axis=0)
    velocities = jnp.stack(
        tuple(result.state.velocities_xyz for result in results), axis=0
    )
    amplitudes = jnp.stack(tuple(result.state.amplitude for result in results), axis=0)
    active = jnp.stack(tuple(result.state.active for result in results), axis=0)
    track_ids = jnp.stack(tuple(result.state.track_ids for result in results), axis=0)
    losses = jnp.stack(tuple(result.ipr.final_loss for result in results), axis=0)
    successful = jnp.stack(tuple(result.successful for result in results), axis=0)
    return STBResult(
        state,
        tuple(results),
        positions,
        velocities,
        amplitudes,
        active,
        track_ids,
        losses,
        jnp.all(successful),
        prepared.plan.plan_id,
    )


__all__ = [
    "PreparedSTB",
    "STB_CAPACITY_EXHAUSTED",
    "STB_NONFINITE",
    "STB_SUBSET_REJECTED",
    "STB_SUCCESS",
    "STBPlan",
    "STBResult",
    "STBState",
    "STBStepEvidence",
    "STBStepResult",
    "initialize_stb",
    "prepare_stb",
    "run_stb",
    "stb_step",
]
