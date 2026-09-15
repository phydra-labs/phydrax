#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...discretization.pic import RelativisticBorisPlan
from ._core import DetectorConditions, TransportTrackBank


class ChargedPropagationPlan(StrictModule, NonTrainableState):
    """Bounded constant-field propagation; no shower or material-interaction claim."""

    conditions: DetectorConditions
    pusher: RelativisticBorisPlan
    step_size: Array
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
        speed_of_light: float = 1.0,
        mean_energy_loss_per_length: float = 0.0,
    ):
        if not isinstance(conditions, DetectorConditions):
            raise TypeError("conditions must be DetectorConditions.")
        step = float(step_size)
        count = int(step_count)
        loss = float(mean_energy_loss_per_length)
        if not math.isfinite(step) or step <= 0.0 or count < 1:
            raise ValueError("step_size and step_count must be positive.")
        if not math.isfinite(loss) or loss < 0.0:
            raise ValueError(
                "mean_energy_loss_per_length must be finite and nonnegative."
            )
        self.conditions = conditions
        self.pusher = RelativisticBorisPlan(speed_of_light)
        self.step_size = jnp.asarray(step)
        self.step_count = count
        self.mean_energy_loss_per_length = loss
        self.plan_id = canonical_fingerprint(
            {
                "kind": "detector-constant-field-propagation",
                "conditions": conditions.conditions_id,
                "step_size": step,
                "step_count": count,
                "speed_of_light": speed_of_light,
                "mean_energy_loss_per_length": loss,
            }
        )


class ChargedPropagationResult(StrictModule, NonTrainableState):
    tracks: TransportTrackBank
    position_history: Array
    momentum_history: Array
    finite_steps: Array
    accepted: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


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
    masses = tracks.rest_energies.reshape((count,))
    charges = tracks.charges.reshape((count,))
    active = tracks.active.reshape((count,)) & tracks.valid.reshape((count,))
    proper = momenta / masses[:, None]
    specific_charge = charges / masses
    electric = jnp.broadcast_to(plan.conditions.electric_field, (count, 3))
    magnetic = jnp.broadcast_to(plan.conditions.magnetic_field, (count, 3))

    def step(carry, _):
        position, proper_velocity, alive = carry
        pushed = plan.pusher.push(
            proper_velocity,
            electric,
            magnetic,
            specific_charge,
            alive,
            plan.step_size,
        )
        displacement = pushed.velocity * plan.step_size
        candidate_position = position + displacement
        distance = jnp.linalg.norm(displacement, axis=-1)
        momentum_magnitude = jnp.linalg.norm(
            pushed.proper_velocity * masses[:, None], axis=-1
        )
        reduced_magnitude = jnp.maximum(
            momentum_magnitude - plan.mean_energy_loss_per_length * distance,
            0.0,
        )
        direction = pushed.proper_velocity / jnp.maximum(
            jnp.linalg.norm(pushed.proper_velocity, axis=-1, keepdims=True),
            jnp.finfo(pushed.proper_velocity.dtype).tiny,
        )
        candidate_proper = direction * (reduced_magnitude / masses)[:, None]
        finite = (
            jnp.all(jnp.isfinite(candidate_position), axis=-1)
            & jnp.all(jnp.isfinite(candidate_proper), axis=-1)
            & pushed.accepted
        )
        next_alive = alive & finite & (reduced_magnitude > 0.0)
        next_position = jnp.where(next_alive[:, None], candidate_position, position)
        next_proper = jnp.where(next_alive[:, None], candidate_proper, proper_velocity)
        return (next_position, next_proper, next_alive), (
            next_position,
            next_proper * masses[:, None],
            finite,
        )

    (final_position, final_proper, final_active), history = jax.lax.scan(
        step,
        (positions, proper, active),
        xs=None,
        length=plan.step_count,
    )
    position_history, momentum_history, finite_steps = history
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
        accepted,
        accepted & tracks.active,
        plan.plan_id,
    )


__all__ = [
    "ChargedPropagationPlan",
    "ChargedPropagationResult",
    "propagate_charged_tracks",
]
