#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry import CompiledGeometry
from ...solver import (
    DifferentialProblem,
    HybridEventPlan,
    HybridGuardPlan,
    HybridSchedulePlan,
    JumpDifferentialProblem,
    ScheduledHybridGuard,
)
from ...stochastic import (
    AbstractJumpProcess,
    JUMP_INVALID_INTENSITY,
    JUMP_MAX_EVENTS,
    JUMP_SUCCESS,
)
from ._profiles import (
    LayeredTerrestrialProfile,
    RadialBodyProfile,
    SmoothStellarRadialProfile,
)
from ._scattering import (
    elastic_scatter_velocity,
    ElasticCollisionResult,
    ElasticScatteringTable,
)


class TransportOutcome(IntEnum):
    """Exclusive physical disposition, independent of numerical success."""

    UNRESOLVED = 0
    TRANSMITTED = 1
    REFLECTED = 2
    CAPTURED = 3
    ESCAPED = 4


class TransportNumericalStatus(IntEnum):
    """Fail-closed numerical disposition, never overloaded as physics."""

    SUCCESS = 0
    INVALID_INITIAL_STATE = 1
    INVALID_RATE = 2
    JUMP_EVENT_CAPACITY = 3
    GUARD_EVENT_CAPACITY = 4
    DIFFERENTIAL_SOLVER_FAILURE = 5
    NONFINITE_RESULT = 6
    COLLISION_INVARIANT_FAILURE = 7


class BodyFrameTransportState(StrictModule, NonTrainableState):
    """One Cartesian dark-matter state in explicit SI/body-frame coordinates."""

    position_m: Array
    velocity_m_s: Array
    frame_id: str = eqx.field(static=True)

    def __init__(
        self,
        position_m: ArrayLike,
        velocity_m_s: ArrayLike,
        /,
        *,
        frame_id: str,
    ):
        position = jnp.asarray(position_m, dtype=float)
        velocity = jnp.asarray(velocity_m_s, dtype=position.dtype)
        frame = str(frame_id)
        if position.shape != (3,) or velocity.shape != (3,):
            raise ValueError("Transport state vectors must have shape (3,).")
        if not frame or frame != frame.strip():
            raise ValueError("Transport state frame_id must be canonical and non-empty.")
        position = eqx.error_if(
            position,
            jnp.any(~jnp.isfinite(position)) | jnp.any(~jnp.isfinite(velocity)),
            "Transport state vectors must be finite.",
        )
        self.position_m = position
        self.velocity_m_s = velocity
        self.frame_id = frame

    def packed(self) -> Array:
        return jnp.concatenate((self.position_m, self.velocity_m_s))

    @classmethod
    def from_packed(
        cls, values: ArrayLike, /, *, frame_id: str
    ) -> BodyFrameTransportState:
        packed = jnp.asarray(values, dtype=float)
        if packed.shape != (6,):
            raise ValueError("Packed transport state must have shape (6,).")
        return cls(packed[:3], packed[3:], frame_id=frame_id)


class RadialSignedDistanceGuard(StrictModule, NonTrainableState):
    """Hybrid guard backed by a compiled analytic sphere signed distance."""

    geometry: CompiledGeometry
    frame_id: str = eqx.field(static=True)

    def __init__(self, geometry: CompiledGeometry, /, *, frame_id: str):
        if not isinstance(geometry, CompiledGeometry):
            raise TypeError("Radial guard geometry must be a CompiledGeometry.")
        if (
            geometry.ambient_dimension != 3
            or not geometry.field_certificate.is_signed_distance
        ):
            raise ValueError(
                "Radial guard requires an exact three-dimensional signed distance."
            )
        frame = str(frame_id)
        if not frame:
            raise ValueError("Radial guard frame_id must be non-empty.")
        self.geometry = geometry
        self.frame_id = frame

    def __call__(self, time: Array, state: Array, args=None, /) -> Array:
        del time, args
        return self.geometry.signed_distance(jnp.asarray(state[:3])[None, :])[0]


class RadialGravityDrift(StrictModule):
    """Body-frame Cartesian drift under one radial profile's enclosed gravity."""

    profile: RadialBodyProfile

    def __init__(self, profile: RadialBodyProfile, /):
        if not isinstance(
            profile, (LayeredTerrestrialProfile, SmoothStellarRadialProfile)
        ):
            raise TypeError(
                "RadialGravityDrift requires a supported radial body profile."
            )
        self.profile = profile

    def __call__(self, time: Array, state: Array, args=None, /) -> Array:
        del time, args
        position = state[:3]
        velocity = state[3:]
        evaluation = self.profile.evaluate(position)
        radius = evaluation.radius_m
        safe_radius = jnp.maximum(radius, jnp.finfo(position.dtype).tiny)
        acceleration = (
            -evaluation.gravitational_acceleration_m_s2 * position / safe_radius
        )
        acceleration = jnp.where(radius > 0.0, acceleration, jnp.zeros_like(position))
        return jnp.concatenate((velocity, acceleration))


class ProfiledElasticJumpProcess(AbstractJumpProcess):
    """Profiled marked jump process for elastic target scattering.

    Channels are targets. Marks contain a Maxwellian target velocity followed by
    an isotropic outgoing center-of-mass relative direction. Random event histories
    are replayable through the guarded jump solver; no pathwise-gradient claim is
    made through discrete event times, channels, or marks.
    """

    profile: RadialBodyProfile
    scattering: ElasticScatteringTable
    projectile_mass_kg: Array
    state_shape: tuple[int, ...] = eqx.field(static=True)
    num_channels: int = eqx.field(static=True)
    mark_shape: tuple[int, ...] = eqx.field(static=True)
    process_id: str = eqx.field(static=True)

    def __init__(
        self,
        profile: RadialBodyProfile,
        scattering: ElasticScatteringTable,
        projectile_mass_kg: ArrayLike,
        /,
    ):
        if not isinstance(
            profile, (LayeredTerrestrialProfile, SmoothStellarRadialProfile)
        ):
            raise TypeError("Profiled scattering requires a supported radial profile.")
        if not isinstance(scattering, ElasticScatteringTable):
            raise TypeError("scattering must be an ElasticScatteringTable.")
        if profile.target_ids != scattering.target_ids:
            raise ValueError("Profile and scattering target axes must match exactly.")
        mass = np.asarray(projectile_mass_kg, dtype=float)
        if mass.shape != () or not np.isfinite(mass) or mass <= 0.0:
            raise ValueError("projectile_mass_kg must be finite and positive.")
        self.profile = profile
        self.scattering = scattering
        self.projectile_mass_kg = jnp.asarray(mass)
        self.state_shape = (6,)
        self.num_channels = scattering.num_targets
        self.mark_shape = (6,)
        self.process_id = canonical_fingerprint(
            {
                "kind": "profiled-elastic-jump-process",
                "profile": profile.profile_id,
                "scattering": scattering.table_id,
                "projectile_mass_kg": float(mass),
            }
        )

    def intensities(self, t: ArrayLike, state: ArrayLike, args=None, /) -> Array:
        del t, args
        packed = jnp.asarray(state)
        evaluation = self.profile.evaluate(packed[:3])
        rates = self.scattering.partial_rates(
            evaluation.target_number_densities_m3,
            packed[3:],
            evaluation.temperature_K,
        )
        return jnp.where(evaluation.inside & evaluation.finite, rates, 0.0)

    def sample_mark(
        self,
        key: Array,
        t: ArrayLike,
        state: ArrayLike,
        channel: ArrayLike,
        args=None,
        /,
    ) -> Array:
        del t, args
        packed = jnp.asarray(state)
        evaluation = self.profile.evaluate(packed[:3])
        return self.scattering.sample_mark(
            key, channel, packed[3:], evaluation.temperature_K
        )

    def collision_evidence(
        self,
        state: ArrayLike,
        channel: ArrayLike,
        mark: ArrayLike,
        /,
    ) -> ElasticCollisionResult:
        """Evaluate one marked collision through the shared particle primitive."""
        packed = jnp.asarray(state)
        mark_ = jnp.asarray(mark)
        channel_ = jnp.asarray(channel, dtype=jnp.int32)
        return elastic_scatter_velocity(
            packed[3:],
            mark_[:3],
            mark_[3:],
            self.projectile_mass_kg,
            self.scattering.target_masses_kg[channel_],
        )

    def jump(
        self,
        state: ArrayLike,
        channel: ArrayLike,
        mark: ArrayLike,
        args=None,
        /,
    ) -> Array:
        del args
        packed = jnp.asarray(state)
        result = self.collision_evidence(packed, channel, mark)
        return packed.at[3:].set(result.projectile_velocity_m_s)


class TransportCollisionEvidence(StrictModule):
    """Fixed-capacity per-event elastic invariant and failure evidence."""

    valid: Array
    scattered: Array
    momentum_residual_kg_m_s: Array
    energy_residual_J: Array
    mass_valid: Array
    direction_valid: Array
    conservative: Array
    finite: Array
    successful: Array


def transport_collision_evidence(
    process: ProfiledElasticJumpProcess,
    pre_states: ArrayLike,
    channels: ArrayLike,
    marks: ArrayLike,
    valid: ArrayLike,
    /,
) -> TransportCollisionEvidence:
    """Re-evaluate recorded marks without changing the stored event history."""
    if not isinstance(process, ProfiledElasticJumpProcess):
        raise TypeError("process must be a ProfiledElasticJumpProcess.")
    states = jnp.asarray(pre_states)
    channels_ = jnp.asarray(channels, dtype=jnp.int32)
    marks_ = jnp.asarray(marks)
    valid_ = jnp.asarray(valid, dtype=bool)
    if (
        states.shape[-1:] != (6,)
        or marks_.shape != states.shape
        or channels_.shape != states.shape[:-1]
        or valid_.shape != states.shape[:-1]
    ):
        raise ValueError("Recorded collision arrays have incompatible fixed shapes.")
    flat_states = states.reshape((-1, 6))
    flat_channels = channels_.reshape((-1,))
    flat_marks = marks_.reshape((-1, 6))
    flat_valid = valid_.reshape((-1,))
    identity_mark = jnp.asarray((0.0, 0.0, 0.0, 1.0, 0.0, 0.0), dtype=marks_.dtype)
    safe_channels = jnp.where(flat_valid, flat_channels, 0)
    safe_marks = jnp.where(flat_valid[:, None], flat_marks, identity_mark)
    evaluated = jax.vmap(process.collision_evidence)(
        flat_states, safe_channels, safe_marks
    )
    prefix = valid_.shape
    residual = jnp.where(
        flat_valid[:, None], evaluated.momentum_residual_kg_m_s, 0.0
    ).reshape(prefix + (3,))
    energy = jnp.where(flat_valid, evaluated.energy_residual_J, 0.0).reshape(prefix)
    return TransportCollisionEvidence(
        valid_,
        (flat_valid & evaluated.scattered).reshape(prefix),
        residual,
        energy,
        ((~flat_valid) | evaluated.mass_valid).reshape(prefix),
        ((~flat_valid) | evaluated.direction_valid).reshape(prefix),
        ((~flat_valid) | evaluated.conservative).reshape(prefix),
        ((~flat_valid) | evaluated.finite).reshape(prefix),
        ((~flat_valid) | evaluated.successful).reshape(prefix),
    )


class TransportPathEvidence(StrictModule):
    """Per-path numerical evidence kept separate from physical outcomes."""

    jump_status: Array
    deterministic_event_count: Array
    deterministic_capacity_exceeded: Array
    initial_valid: Array
    solver_successful: Array
    collision_invariants_valid: Array
    finite: Array
    successful: Array
    numerical_status: Array


def transport_path_evidence(
    jump_status: ArrayLike,
    deterministic_event_count: ArrayLike,
    deterministic_capacity_exceeded: ArrayLike,
    finite: ArrayLike,
    /,
    *,
    initial_valid: ArrayLike | None = None,
    solver_successful: ArrayLike | None = None,
    collision_invariants_valid: ArrayLike | None = None,
) -> TransportPathEvidence:
    jump = jnp.asarray(jump_status, dtype=jnp.int32)
    event_count = jnp.asarray(deterministic_event_count, dtype=jnp.int32)
    capacity = jnp.asarray(deterministic_capacity_exceeded, dtype=bool)
    finite_ = jnp.asarray(finite, dtype=bool)
    initial = (
        jnp.ones_like(finite_, dtype=bool)
        if initial_valid is None
        else jnp.broadcast_to(jnp.asarray(initial_valid, dtype=bool), finite_.shape)
    )
    solver_ok = (
        jnp.ones_like(finite_, dtype=bool)
        if solver_successful is None
        else jnp.broadcast_to(jnp.asarray(solver_successful, dtype=bool), finite_.shape)
    )
    collision_ok = (
        jnp.ones_like(finite_, dtype=bool)
        if collision_invariants_valid is None
        else jnp.broadcast_to(
            jnp.asarray(collision_invariants_valid, dtype=bool), finite_.shape
        )
    )
    status = jnp.where(
        ~initial,
        int(TransportNumericalStatus.INVALID_INITIAL_STATE),
        jnp.where(
            ~finite_,
            int(TransportNumericalStatus.NONFINITE_RESULT),
            jnp.where(
                capacity,
                int(TransportNumericalStatus.GUARD_EVENT_CAPACITY),
                jnp.where(
                    jump == JUMP_INVALID_INTENSITY,
                    int(TransportNumericalStatus.INVALID_RATE),
                    jnp.where(
                        jump == JUMP_MAX_EVENTS,
                        int(TransportNumericalStatus.JUMP_EVENT_CAPACITY),
                        jnp.where(
                            ~collision_ok,
                            int(TransportNumericalStatus.COLLISION_INVARIANT_FAILURE),
                            jnp.where(
                                (jump == JUMP_SUCCESS) & solver_ok,
                                int(TransportNumericalStatus.SUCCESS),
                                int(TransportNumericalStatus.DIFFERENTIAL_SOLVER_FAILURE),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ).astype(jnp.int32)
    successful = status == int(TransportNumericalStatus.SUCCESS)
    return TransportPathEvidence(
        jump,
        event_count,
        capacity,
        initial,
        solver_ok,
        collision_ok,
        finite_,
        successful,
        status,
    )


def radial_guard_schedule(
    specifications: tuple[tuple[str, CompiledGeometry, int, bool, int], ...],
    vector_field: RadialGravityDrift,
    /,
    *,
    frame_id: str,
    maximum_events: int,
) -> HybridSchedulePlan:
    """Compose exact sphere signed distances into one guarded jump schedule."""
    scheduled = []
    for name, geometry, direction, terminal, priority in specifications:
        guard = HybridGuardPlan(
            RadialSignedDistanceGuard(geometry, frame_id=frame_id),
            direction=direction,
            terminal=terminal,
            priority=priority,
            guard_id=f"{frame_id}:{name}",
        )
        event = HybridEventPlan(
            guard,
            lambda time, state, args: state,
            vector_field,
            vector_field,
            plan_id=f"{frame_id}:{name}:identity-reset",
        )
        scheduled.append(ScheduledHybridGuard(guard, event=event))
    return HybridSchedulePlan(tuple(scheduled), maximum_events=maximum_events)


def jump_differential_problem(
    process: ProfiledElasticJumpProcess,
    initial_state: ArrayLike,
    /,
    *,
    t0_s: float,
    t1_s: float,
) -> JumpDifferentialProblem:
    """Build the continuous radial-gravity component for guarded jump transport."""
    state = jnp.asarray(initial_state, dtype=float)
    if state.shape != (6,):
        raise ValueError("initial_state must have packed shape (6,).")
    differential = DifferentialProblem(
        RadialGravityDrift(process.profile),
        state,
        t0=t0_s,
        t1=t1_s,
    )
    return JumpDifferentialProblem(differential, process)


def transparent_state(state: ArrayLike, elapsed_s: ArrayLike, /) -> Array:
    """Exact force-free propagation, used for transparent-path qualification."""
    packed = jnp.asarray(state, dtype=float)
    elapsed = jnp.asarray(elapsed_s, dtype=packed.dtype).reshape(())
    if packed.shape != (6,):
        raise ValueError("Transparent propagation requires packed shape (6,).")
    position = packed[:3] + elapsed * packed[3:]
    return jnp.concatenate((position, packed[3:]))


__all__ = [
    "TransportCollisionEvidence",
    "BodyFrameTransportState",
    "ProfiledElasticJumpProcess",
    "RadialGravityDrift",
    "RadialSignedDistanceGuard",
    "TransportNumericalStatus",
    "TransportOutcome",
    "TransportPathEvidence",
    "transport_collision_evidence",
    "jump_differential_problem",
    "radial_guard_schedule",
    "transparent_state",
    "transport_path_evidence",
]
