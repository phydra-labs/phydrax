#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Manufacturing paths, process schedules, moving sources, and material activation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..qualification import CapabilityProfile, SupportTuple


ProcessEventKind: TypeAlias = Literal[
    "move", "deposit", "remove", "dwell", "heat", "fixture", "transfer"
]


@dataclass(frozen=True, slots=True)
class ToolpathEvent:
    event_id: str
    kind: ProcessEventKind
    start_time_s: float
    end_time_s: float
    frame_id: str
    start: tuple[float, ...]
    end: tuple[float, ...]
    power_w: float = 0.0
    mass_rate_kg_s: float = 0.0

    def __post_init__(self) -> None:
        if not self.event_id or not self.frame_id or self.end_time_s < self.start_time_s:
            raise ValueError("Toolpath event identity, frame, and times are invalid.")
        if len(self.start) != len(self.end) or not self.start:
            raise ValueError("Toolpath event endpoints must share a nonzero dimension.")
        values = (*self.start, *self.end, self.power_w, self.mass_rate_kg_s)
        if (
            any(not np.isfinite(value) for value in values)
            or self.power_w < 0.0
            or self.mass_rate_kg_s < 0.0
        ):
            raise ValueError(
                "Toolpath event data must be finite and non-negative where required."
            )

    @property
    def event_fingerprint(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "toolpath-event",
                "event_id": self.event_id,
                "event_kind": self.kind,
                "start_time_s": self.start_time_s,
                "end_time_s": self.end_time_s,
                "frame_id": self.frame_id,
                "start": self.start,
                "end": self.end,
                "power_w": self.power_w,
                "mass_rate_kg_s": self.mass_rate_kg_s,
            }
        )

    def position(self, time_s: ArrayLike, /) -> Array:
        time = jnp.asarray(time_s)
        duration = max(self.end_time_s - self.start_time_s, np.finfo(float).eps)
        fraction = jnp.clip((time - self.start_time_s) / duration, 0.0, 1.0)
        return jnp.asarray(self.start) + fraction * (
            jnp.asarray(self.end) - jnp.asarray(self.start)
        )


@dataclass(frozen=True, slots=True)
class ProcessSchedule:
    events: tuple[ToolpathEvent, ...]

    @classmethod
    def create(cls, events):
        return cls(
            tuple(sorted(events, key=lambda item: (item.start_time_s, item.event_id)))
        )

    def __post_init__(self) -> None:
        if not self.events or any(
            not isinstance(value, ToolpathEvent) for value in self.events
        ):
            raise TypeError("A process schedule requires ToolpathEvent values.")
        ids = tuple(value.event_id for value in self.events)
        if len(set(ids)) != len(ids):
            raise ValueError("Toolpath event IDs must be unique.")
        for left, right in zip(self.events, self.events[1:], strict=False):
            if right.start_time_s < left.end_time_s:
                raise ValueError("This bounded schedule requires nonoverlapping events.")

    @property
    def schedule_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "process-schedule",
                "events": [value.event_fingerprint for value in self.events],
            }
        )


class MaterialActivationState(StrictModule, NonTrainableState):
    active: Array
    activation_time_s: Array

    def __init__(self, active: ArrayLike, activation_time_s: ArrayLike, /):
        active_ = jnp.asarray(active, dtype=bool)
        times = jnp.asarray(activation_time_s)
        if active_.shape != times.shape:
            raise ValueError("Activation mask and times must align.")
        self.active = active_
        self.activation_time_s = times

    def activate(
        self, selection: ArrayLike, time_s: ArrayLike, /
    ) -> MaterialActivationState:
        selection_ = jnp.asarray(selection, dtype=bool)
        if selection_.shape != self.active.shape:
            raise ValueError("Activation selection has the wrong shape.")
        newly = selection_ & ~self.active
        return MaterialActivationState(
            self.active | selection_,
            jnp.where(newly, jnp.asarray(time_s), self.activation_time_s),
        )


class GaussianMovingSource(StrictModule, NonTrainableState):
    power_w: Array
    absorptivity: Array
    radius_m: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, power_w: float, absorptivity: float, radius_m: float, /):
        if power_w < 0.0 or not 0.0 <= absorptivity <= 1.0 or radius_m <= 0.0:
            raise ValueError(
                "Gaussian source parameters are outside their physical bounds."
            )
        self.power_w = jnp.asarray(power_w)
        self.absorptivity = jnp.asarray(absorptivity)
        self.radius_m = jnp.asarray(radius_m)
        self.source_id = canonical_fingerprint(
            {
                "kind": "gaussian-moving-source",
                "power_w": power_w,
                "absorptivity": absorptivity,
                "radius_m": radius_m,
            }
        )

    def evaluate(self, coordinates: ArrayLike, center: ArrayLike, /) -> Array:
        delta = jnp.asarray(coordinates) - jnp.asarray(center)
        radius_squared = jnp.sum(delta * delta, axis=-1)
        normalization = (
            2.0 * self.absorptivity * self.power_w / (jnp.pi * self.radius_m**2)
        )
        return normalization * jnp.exp(-2.0 * radius_squared / self.radius_m**2)


class ProcessHistory(StrictModule, NonTrainableState):
    times_s: Array
    deposited_mass_kg: Array
    supplied_energy_j: Array
    active_measure: Array
    history_id: str = eqx.field(static=True)

    def __init__(self, times_s, deposited_mass_kg, supplied_energy_j, active_measure, /):
        times = np.asarray(times_s, dtype=float)
        mass = np.asarray(deposited_mass_kg, dtype=float)
        energy = np.asarray(supplied_energy_j, dtype=float)
        measure = np.asarray(active_measure, dtype=float)
        if times.ndim != 1 or any(
            value.shape != times.shape for value in (mass, energy, measure)
        ):
            raise ValueError("Process-history arrays must be aligned vectors.")
        if (
            np.any(np.diff(times) <= 0.0)
            or np.any(np.diff(mass) < 0.0)
            or np.any(np.diff(energy) < 0.0)
            or np.any(measure < 0.0)
        ):
            raise ValueError("Process history violates monotonicity or positivity.")
        self.times_s = jnp.asarray(times)
        self.deposited_mass_kg = jnp.asarray(mass)
        self.supplied_energy_j = jnp.asarray(energy)
        self.active_measure = jnp.asarray(measure)
        self.history_id = canonical_fingerprint(
            {
                "kind": "process-history",
                "times_s": times.tolist(),
                "deposited_mass_kg": mass.tolist(),
                "supplied_energy_j": energy.tolist(),
                "active_measure": measure.tolist(),
            }
        )


def manufacturing_candidate_profiles() -> tuple[CapabilityProfile, ...]:
    specs = (
        (
            "manufacturing.process-schedule",
            {"events": "move-deposit-remove-dwell-heat-fixture-transfer"},
        ),
        ("manufacturing.material-activation", {"topology": "fixed-capacity-activation"}),
        ("manufacturing.moving-gaussian-source", {"source": "surface-gaussian"}),
        ("manufacturing.process-history", {"ledgers": "mass-energy-active-measure"}),
    )
    return tuple(
        CapabilityProfile(
            f"{name}.profile",
            "phydrax",
            "candidate",
            (SupportTuple(name, attrs),),
            required_gates=("identity", "conservation", "restart", "public-workflow"),
        )
        for name, attrs in specs
    )


__all__ = [
    "GaussianMovingSource",
    "MaterialActivationState",
    "ProcessEventKind",
    "ProcessHistory",
    "ProcessSchedule",
    "ToolpathEvent",
    "manufacturing_candidate_profiles",
]
