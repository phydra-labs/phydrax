#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._fingerprint import canonical_fingerprint
from ..._physical import ElectromagneticScaleContract
from ..._sampling import derive_key, SampleAddress
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.pic import (
    PIC_CODE_RELATIVITY,
    RadiationReactionPlan,
    RadiationReactionResult,
    RelativisticPushPlan,
)
from ...electromagnetics._trajectory_radiation import ChargedTrajectory
from ...typing import PRNGKey
from ._core import DetectorConditions, TransportTrackBank


# (position, proper velocity, alive) per flattened track.
_PropagationCarry: TypeAlias = tuple[Array, Array, Array]
_PropagationHistory: TypeAlias = tuple[Array, Array, Array, Array, Array, Array]


class ChargedPropagationPlan(StrictModule, NonTrainableState):
    """Bounded constant-field propagation; no shower or material-interaction claim.

    ``radiation_reaction`` optionally applies a `RadiationReactionPlan` after
    each push (its scale must match the pusher's units and speed of light, and
    every propagated track must be its species); the constant field has exactly
    zero gradients and time derivatives. The stochastic Fokker–Planck model
    requires ``radiation_key``; its Wiener increments are addressed by step and
    track identity ``(event_id, track_id)``.
    """

    __strict_contract__ = True

    conditions: DetectorConditions
    pusher: RelativisticPushPlan
    step_size: Array
    radiation_reaction: RadiationReactionPlan | None
    radiation_key: PRNGKey | None
    step_count: int = eqx.field(static=True)
    mean_energy_loss_per_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        conditions: DetectorConditions,
        /,
        *,
        step_size: float,
        step_count: int,
        pusher: RelativisticPushPlan | None = None,
        mean_energy_loss_per_length: float = 0.0,
        radiation_reaction: RadiationReactionPlan | None = None,
        radiation_key: PRNGKey | None = None,
    ) -> None:
        if not isinstance(conditions, DetectorConditions):
            raise TypeError("conditions must be DetectorConditions.")
        step = float(step_size)
        if isinstance(step_count, bool) or not isinstance(step_count, int):
            raise TypeError("step_count must be an integer.")
        count = step_count
        loss = float(mean_energy_loss_per_length)
        if not math.isfinite(step) or step <= 0.0 or count < 1:
            raise ValueError("step_size and step_count must be positive.")
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError(
                "mean_energy_loss_per_length must be finite and nonnegative."
            )
        pusher_ = (
            RelativisticPushPlan(PIC_CODE_RELATIVITY, method="boris")
            if pusher is None
            else pusher
        )
        if not isinstance(pusher_, RelativisticPushPlan):
            raise TypeError("pusher must be RelativisticPushPlan or None.")
        if radiation_reaction is not None:
            if not isinstance(radiation_reaction, RadiationReactionPlan):
                raise TypeError("radiation_reaction must be RadiationReactionPlan.")
            radiation_reaction.validate_relativity(pusher_.relativity)
        stochastic = radiation_reaction is not None and radiation_reaction.stochastic
        if stochastic != (radiation_key is not None):
            raise ValueError(
                "radiation_key is required by, and only by, stochastic radiation "
                "reaction."
            )
        self.conditions = conditions
        self.pusher = pusher_
        self.step_size = jnp.asarray(step)
        self.radiation_reaction = radiation_reaction
        self.radiation_key = radiation_key
        self.step_count = count
        self.mean_energy_loss_per_length = loss
        self.plan_id = canonical_fingerprint(
            {
                "kind": "detector-constant-field-propagation",
                "conditions": conditions.conditions_id,
                "step_size": step,
                "step_count": count,
                "pusher": pusher_.plan_id,
                "mean_energy_loss_per_length": loss,
                "radiation_reaction": None
                if radiation_reaction is None
                else radiation_reaction.plan_id,
            }
        )


class ChargedPropagationResult(StrictModule, NonTrainableState):
    """Propagated tracks with per-step histories ``[step, event, track, ...]``.

    Entry ``s`` of each history is the state after step ``s + 1``.
    ``active_history[s]`` marks that step ``s + 1`` committed a finite state of an
    existing track; a track brought to rest by the declared energy loss is active
    at the committing step and inactive afterwards.
    ``radiated_energy_history[s]`` is the energy each track radiated by radiation
    reaction in committed step ``s + 1`` (zero without radiation reaction), and
    ``radiation_flags_history[s]`` its `RadiationReactionFlag` bits; a step with
    an unsupported reaction is neither finite in ``finite_steps`` nor committed,
    so the track is not ``accepted``.
    """

    tracks: TransportTrackBank
    position_history: Array
    momentum_history: Array
    finite_steps: Array
    active_history: Array
    accepted: Array
    derivative_valid: Array
    radiated_energy_history: Array
    radiation_flags_history: Array
    plan_id: str = eqx.field(static=True)


def _require_radiating_species(
    reaction: RadiationReactionPlan,
    charges: Array,
    masses: Array,
    active: Array,
    /,
) -> None:
    """Refuse active tracks whose charge or mass is not the reaction species."""
    mismatched = np.asarray(active & ~reaction.matches_species(charges, masses))
    if np.any(mismatched):
        raise ValueError(
            f"{int(np.sum(mismatched))} active tracks differ in charge or mass from "
            "the radiation-reaction species."
        )


def _reaction_step(
    plan: ChargedPropagationPlan,
    reaction: RadiationReactionPlan,
    proper_velocity: Array,
    electric: Array,
    magnetic: Array,
    alive: Array,
    identities: tuple[Array, Array],
    step_index: Array,
    /,
) -> RadiationReactionResult:
    """Radiation reaction in the constant field after one push."""
    wiener = None
    if reaction.stochastic:
        if plan.radiation_key is None:
            raise ValueError("Stochastic radiation reaction lost its key.")
        key = derive_key(
            plan.radiation_key,
            SampleAddress("phydrax.detector", "radiation-reaction", role="step"),
            step_index,
        )
        wiener = reaction.wiener_increments(key, *identities)
    gradient = rate = None
    if reaction.requires_field_derivatives:
        # A constant field has exactly zero gradients and time derivatives.
        gradient = jnp.zeros(proper_velocity.shape + (3,), dtype=jnp.float64)
        rate = jnp.zeros(proper_velocity.shape, dtype=jnp.float64)
    return reaction.apply(
        proper_velocity,
        electric,
        magnetic,
        plan.step_size,
        alive,
        electric_gradient=gradient,
        magnetic_gradient=gradient,
        electric_rate=rate,
        magnetic_rate=rate,
        wiener=wiener,
    )


def propagate_charged_tracks(
    plan: ChargedPropagationPlan,
    tracks: TransportTrackBank,
    /,
) -> ChargedPropagationResult:
    """Propagate massive charged tracks through one constant field."""
    if not isinstance(plan, ChargedPropagationPlan):
        raise TypeError("plan must be ChargedPropagationPlan.")
    if not isinstance(tracks, TransportTrackBank):
        raise TypeError("tracks must be TransportTrackBank.")
    if tracks.conditions_id != plan.conditions.conditions_id:
        raise ValueError("Tracks and propagation conditions do not match.")
    shape = tracks.active.shape
    count = shape[0] * shape[1]
    positions = tracks.positions.reshape((count, 3))
    momenta = tracks.momenta.reshape((count, 3))
    rest_energies = tracks.rest_energies.reshape((count,))
    charges = tracks.charges.reshape((count,))
    active = tracks.active.reshape((count,)) & tracks.valid.reshape((count,))
    light = plan.pusher.speed_of_light
    masses = rest_energies / light**2
    proper = momenta / masses[:, None]
    specific_charge = charges / masses
    electric = jnp.broadcast_to(plan.conditions.electric_field, (count, 3))
    magnetic = jnp.broadcast_to(plan.conditions.magnetic_field, (count, 3))
    reaction = plan.radiation_reaction
    identities = (
        _identity_words(tracks)
        if reaction is not None and reaction.stochastic
        else (jnp.zeros((count,), jnp.uint32), jnp.zeros((count,), jnp.uint32))
    )
    if reaction is not None:
        _require_radiating_species(reaction, charges, masses, active)

    def step(
        carry: _PropagationCarry, step_index: Array
    ) -> tuple[_PropagationCarry, _PropagationHistory]:
        position, proper_velocity, alive = carry
        pushed = plan.pusher.push(
            proper_velocity,
            electric,
            magnetic,
            specific_charge,
            alive,
            plan.step_size,
        )
        after_push = pushed.proper_velocity
        velocity = pushed.velocity
        radiated = jnp.zeros((count,), dtype=after_push.dtype)
        flags = jnp.zeros((count,), dtype=jnp.int32)
        supported = jnp.ones((count,), dtype=jnp.bool_)
        if reaction is not None:
            reacted = _reaction_step(
                plan,
                reaction,
                after_push,
                electric,
                magnetic,
                alive,
                identities,
                step_index,
            )
            after_push = reacted.proper_velocity.astype(after_push.dtype)
            velocity = plan.pusher.velocity(after_push)
            radiated = reacted.radiated_energy.astype(after_push.dtype)
            flags = reacted.flags
            supported = reacted.supported
        displacement = velocity * plan.step_size
        candidate_position = position + displacement
        distance = jnp.linalg.norm(displacement, axis=-1)
        momentum_magnitude = jnp.linalg.norm(after_push * masses[:, None], axis=-1)
        total_energy = jnp.sqrt((light * momentum_magnitude) ** 2 + rest_energies**2)
        remaining_energy = jnp.maximum(
            total_energy - plan.mean_energy_loss_per_length * distance,
            rest_energies,
        )
        reduced_magnitude = (
            jnp.sqrt(jnp.maximum(remaining_energy**2 - rest_energies**2, 0.0)) / light
        )
        direction = after_push / jnp.maximum(
            jnp.linalg.norm(after_push, axis=-1, keepdims=True),
            jnp.finfo(after_push.dtype).tiny,
        )
        candidate_proper = direction * (reduced_magnitude / masses)[:, None]
        finite = (
            jnp.all(jnp.isfinite(candidate_position), axis=-1)
            & jnp.all(jnp.isfinite(candidate_proper), axis=-1)
            & pushed.successful
            & supported
        )
        commit = alive & finite
        next_alive = commit & (reduced_magnitude > 0.0)
        next_position = jnp.where(commit[:, None], candidate_position, position)
        next_proper = jnp.where(commit[:, None], candidate_proper, proper_velocity)
        return (next_position, next_proper, next_alive), (
            next_position,
            next_proper * masses[:, None],
            finite,
            commit,
            jnp.where(commit, radiated, 0.0),
            jnp.where(alive, flags, 0),
        )

    (final_position, final_proper, final_active), history = jax.lax.scan(
        step,
        (positions, proper, active),
        xs=jnp.arange(plan.step_count, dtype=jnp.int32),
    )
    (
        position_history,
        momentum_history,
        finite_steps,
        active_history,
        radiated_history,
        flags_history,
    ) = history
    final_momentum = final_proper * masses[:, None]
    result_tracks = TransportTrackBank(
        event_ids=tracks.event_ids,
        track_ids=tracks.track_ids,
        parent_track_ids=tracks.parent_track_ids,
        pdg_ids=tracks.pdg_ids,
        positions=final_position.reshape(shape + (3,)),
        momenta=final_momentum.reshape(shape + (3,)),
        rest_energies=tracks.rest_energies,
        charges=tracks.charges,
        active=final_active.reshape(shape),
        conditions_id=tracks.conditions_id,
    )
    accepted = jnp.all(jnp.where(active[None, :], finite_steps, True), axis=0).reshape(
        shape
    )
    return ChargedPropagationResult(
        result_tracks,
        position_history.reshape((plan.step_count,) + shape + (3,)),
        momentum_history.reshape((plan.step_count,) + shape + (3,)),
        finite_steps.reshape((plan.step_count,) + shape),
        active_history.reshape((plan.step_count,) + shape),
        accepted,
        accepted & tracks.active,
        radiated_history.reshape((plan.step_count,) + shape),
        flags_history.reshape((plan.step_count,) + shape),
        plan.plan_id,
    )


def _require_float64(value: Array, name: str, /) -> None:
    if value.dtype != jnp.float64:
        raise TypeError(
            f"{name} must be float64; radiation phases are always float64 and "
            f"{value.dtype} detector kinematics are refused."
        )


def _identity_words(tracks: TransportTrackBank, /) -> tuple[Array, Array]:
    """Return lane identity words ``(event_id, track_id)`` in row-major lane order.

    Event IDs are validated on the host at this boundary: they must be integers in
    ``[0, 2³²)``. Track IDs are int32 and reinterpreted bit-exactly as uint32, so
    empty slots keep their negative sentinel as a distinct word.
    """
    event_ids = np.asarray(tracks.event_ids)
    if not np.issubdtype(event_ids.dtype, np.integer):
        raise TypeError("Track-bank event IDs must be integers to form identities.")
    if np.any(event_ids < 0) or np.any(event_ids > np.iinfo(np.uint32).max):
        raise ValueError("Track-bank event IDs must lie in [0, 2**32) for identities.")
    capacity = tracks.track_capacity
    high = jnp.asarray(np.repeat(event_ids.astype(np.uint32), capacity))
    low = jax.lax.bitcast_convert_type(tracks.track_ids.reshape(-1), jnp.uint32)
    return high, low


def charged_trajectory(
    plan: ChargedPropagationPlan,
    tracks: TransportTrackBank,
    result: ChargedPropagationResult,
    scale: ElectromagneticScaleContract,
    /,
) -> ChargedTrajectory:
    """Return the radiating lanes of ``result = propagate_charged_tracks(plan, tracks)``.

    Lane ``e * track_capacity + k`` is track ``k`` of event ``e`` with identity
    words ``(event_id, track_id)``. Sample 0 is the initial state in ``tracks`` at
    time 0 and sample ``s`` the state after ``s`` steps at ``s * step_size``, on
    every lane. A sample is active when it is a committed state of an existing,
    valid track. Inactive samples keep the last committed position (propagation
    freezes them, so every lane stays causal) and carry zero proper velocity.

    The pusher's relativity scale must have the same dimensional scale and exact
    speed of light as ``scale``, so positions, times, and velocities are already
    in ``scale`` units. Track charges are the per-particle charges the pusher
    uses, read in ``scale.charge_unit``; each lane has multiplicity one. Proper
    velocities are ``u = p c² / E₀`` from momenta and rest energies. Detector
    kinematics must already be float64; other dtypes are refused.
    """
    if not isinstance(plan, ChargedPropagationPlan):
        raise TypeError("plan must be ChargedPropagationPlan.")
    if not isinstance(tracks, TransportTrackBank):
        raise TypeError("tracks must be TransportTrackBank.")
    if not isinstance(result, ChargedPropagationResult):
        raise TypeError("result must be ChargedPropagationResult.")
    if not isinstance(scale, ElectromagneticScaleContract):
        raise TypeError("scale must be ElectromagneticScaleContract.")
    if tracks.conditions_id != plan.conditions.conditions_id:
        raise ValueError("Tracks and propagation conditions do not match.")
    if result.plan_id != plan.plan_id:
        raise ValueError("result was not produced by plan.")
    shape = tracks.active.shape
    if result.position_history.shape != (plan.step_count,) + shape + (3,):
        raise ValueError("result histories do not match the track-bank capacity.")
    relativity = plan.pusher.relativity
    if (
        relativity.dimensional_scale.scale_id
        != scale.relativity.dimensional_scale.scale_id
    ):
        raise ValueError(
            "The pusher relativity scale and the electromagnetic scale use "
            "different length, mass, or time units."
        )
    if relativity.speed_of_light != scale.speed_of_light:
        raise ValueError(
            "The pusher speed of light "
            f"{relativity.speed_of_light} does not match the electromagnetic scale "
            f"speed of light {scale.speed_of_light}."
        )
    for value, name in (
        (tracks.positions, "Track positions"),
        (tracks.momenta, "Track momenta"),
        (tracks.rest_energies, "Track rest energies"),
        (tracks.charges, "Track charges"),
        (result.position_history, "Position history"),
        (result.momentum_history, "Momentum history"),
    ):
        _require_float64(value, name)
    count = shape[0] * shape[1]
    active = jnp.concatenate(
        (
            (tracks.active & tracks.valid).reshape((1, count)),
            result.active_history.reshape((plan.step_count, count)),
        )
    )
    positions = jnp.concatenate(
        (
            tracks.positions.reshape((1, count, 3)),
            result.position_history.reshape((plan.step_count, count, 3)),
        )
    )
    momenta = jnp.concatenate(
        (
            tracks.momenta.reshape((1, count, 3)),
            result.momentum_history.reshape((plan.step_count, count, 3)),
        )
    )
    light_sq = float(scale.speed_of_light) ** 2
    rest_energies = tracks.rest_energies.reshape((count,))
    # Empty slots may hold nonpositive rest energies; a unit placeholder mass keeps
    # their masked kinematics and derivatives finite.
    masses = jnp.where(rest_energies > 0.0, rest_energies, light_sq) / light_sq
    # Active samples are finite by validity and commit; only positions of empty
    # slots can be non-finite, and they are constant along their lane.
    positions = jnp.where(jnp.isfinite(positions), positions, 0.0)
    times = jnp.arange(plan.step_count + 1, dtype=jnp.float64) * jnp.asarray(
        plan.step_size, dtype=jnp.float64
    )
    return ChargedTrajectory(
        times,
        positions,
        jnp.where(active[..., None], momenta / masses[None, :, None], 0.0),
        tracks.charges.reshape((count,)),
        jnp.ones((count,), dtype=jnp.float64),
        active,
        _identity_words(tracks),
    )


__all__ = [
    "ChargedPropagationPlan",
    "ChargedPropagationResult",
    "charged_trajectory",
    "propagate_charged_tracks",
]
