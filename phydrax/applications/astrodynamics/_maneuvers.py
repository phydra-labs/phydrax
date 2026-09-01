#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._state import CartesianOrbitState
from ._status import AstrodynamicsStatus


@dataclass(frozen=True)
class ImpulseManeuver:
    epoch_seconds: float
    delta_velocity: tuple[float, float, float]
    maneuver_id: str


@dataclass(frozen=True)
class FiniteBurnSegment:
    start_seconds: float
    stop_seconds: float
    acceleration: tuple[float, float, float]
    mass_flow_rate: float
    maneuver_id: str


class ManeuverEvaluation(StrictModule):
    acceleration: Array
    mass_flow_rate: Array
    active_segment: Array
    valid: Array
    status: Array
    schedule_id: str = eqx.field(static=True)


class ManeuverSchedule(StrictModule, NonTrainableState):
    impulses: tuple[ImpulseManeuver, ...] = eqx.field(static=True)
    finite_burns: tuple[FiniteBurnSegment, ...] = eqx.field(static=True)
    schedule_id: str = eqx.field(static=True)

    def __init__(
        self,
        impulses: tuple[ImpulseManeuver, ...] = (),
        finite_burns: tuple[FiniteBurnSegment, ...] = (),
        /,
    ):
        impulses_ = tuple(sorted(impulses, key=lambda value: value.epoch_seconds))
        burns = tuple(sorted(finite_burns, key=lambda value: value.start_seconds))
        if any(not np.isfinite(value.epoch_seconds) for value in impulses_):
            raise ValueError("Impulse epochs must be finite.")
        if any(
            not np.isfinite(value.start_seconds)
            or not np.isfinite(value.stop_seconds)
            or value.stop_seconds <= value.start_seconds
            or value.mass_flow_rate < 0.0
            for value in burns
        ):
            raise ValueError("Finite-burn segments are invalid.")
        for left, right in zip(burns[:-1], burns[1:], strict=True):
            if right.start_seconds < left.stop_seconds:
                raise ValueError("Finite-burn segments may not overlap.")
        self.impulses = impulses_
        self.finite_burns = burns
        self.schedule_id = canonical_fingerprint(
            {
                "kind": "maneuver-schedule",
                "impulses": [value.__dict__ for value in impulses_],
                "finite_burns": [value.__dict__ for value in burns],
            }
        )

    def evaluate(self, time: ArrayLike, /) -> ManeuverEvaluation:
        query = jnp.asarray(time).reshape(())
        acceleration = jnp.zeros((3,), dtype=query.dtype)
        mass_flow = jnp.asarray(0.0, dtype=query.dtype)
        active = jnp.asarray(-1, dtype=jnp.int32)
        for index, segment in enumerate(self.finite_burns):
            selected = (query >= segment.start_seconds) & (query < segment.stop_seconds)
            acceleration = jnp.where(
                selected,
                jnp.asarray(segment.acceleration, dtype=query.dtype),
                acceleration,
            )
            mass_flow = jnp.where(selected, segment.mass_flow_rate, mass_flow)
            active = jnp.where(selected, index, active)
        valid = jnp.isfinite(query) & jnp.all(jnp.isfinite(acceleration))
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.NONFINITE_INPUT),
        ).astype(jnp.int32)
        return ManeuverEvaluation(
            acceleration, mass_flow, active, valid, status, self.schedule_id
        )

    def apply_impulse(
        self, state: CartesianOrbitState, impulse_index: int, /
    ) -> CartesianOrbitState:
        index = int(impulse_index)
        if not 0 <= index < len(self.impulses):
            raise ValueError("impulse_index is outside schedule capacity.")
        return CartesianOrbitState(
            state.position,
            state.velocity + jnp.asarray(self.impulses[index].delta_velocity),
            state.context,
        )


__all__ = [
    "FiniteBurnSegment",
    "ImpulseManeuver",
    "ManeuverEvaluation",
    "ManeuverSchedule",
]
