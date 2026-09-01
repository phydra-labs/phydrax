#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AerodynamicLoadHistory(StrictModule):
    times: Array
    source_position: Array
    source_velocity: Array
    force: Array
    normal: Array
    area: Array
    history_id: str = eqx.field(static=True)

    def __init__(
        self,
        times: ArrayLike,
        source_position: ArrayLike,
        source_velocity: ArrayLike,
        force: ArrayLike,
        normal: ArrayLike,
        area: ArrayLike,
        /,
        *,
        history_id: str | None = None,
    ):
        time = jnp.asarray(times)
        position, velocity, force_ = (
            jnp.asarray(source_position),
            jnp.asarray(source_velocity),
            jnp.asarray(force),
        )
        normal_, area_ = jnp.asarray(normal), jnp.asarray(area)
        if (
            time.ndim != 1
            or position.shape[:2] != (time.size, position.shape[1])
            or position.shape[-1] != 3
            or velocity.shape != position.shape
            or force_.shape != position.shape
            or normal_.shape != position.shape
            or area_.shape != position.shape[:2]
            or jnp.any(jnp.diff(time) <= 0.0)
        ):
            raise ValueError("Aerodynamic load history arrays are incompatible.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "aerodynamic-load-history",
                    "time_count": int(time.size),
                    "source_count": int(position.shape[1]),
                }
            )
            if history_id is None
            else str(history_id)
        )
        if not identifier:
            raise ValueError("history_id must be nonempty.")
        self.times, self.source_position, self.source_velocity, self.force = (
            time,
            position,
            velocity,
            force_,
        )
        self.normal, self.area, self.history_id = normal_, area_, identifier


class FWHObserverResult(StrictModule):
    observer_position: Array
    pressure: Array
    retarded_time: Array
    thickness_pressure: Array
    loading_pressure: Array
    finite: Array
    acoustics_id: str = eqx.field(static=True)


class FWHTonalAcousticsPlan(StrictModule, NonTrainableState):
    sound_speed: float = eqx.field(static=True)
    ambient_density: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, sound_speed: float, ambient_density: float, /):
        if float(sound_speed) <= 0.0 or float(ambient_density) <= 0.0:
            raise ValueError("FW-H sound speed/density must be positive.")
        self.sound_speed, self.ambient_density = (
            float(sound_speed),
            float(ambient_density),
        )
        self.plan_id = canonical_fingerprint(
            {
                "kind": "fwh-tonal-acoustics",
                "sound_speed": self.sound_speed,
                "ambient_density": self.ambient_density,
            }
        )

    def evaluate(
        self, history: AerodynamicLoadHistory, observers: ArrayLike, /
    ) -> FWHObserverResult:
        if not isinstance(history, AerodynamicLoadHistory):
            raise TypeError("history must be AerodynamicLoadHistory.")
        observer = jnp.asarray(observers, dtype=history.source_position.dtype)
        if observer.ndim != 2 or observer.shape[1] != 3:
            raise ValueError("FW-H observers require shape (count,3).")
        displacement = observer[:, None, None, :] - history.source_position[None, :, :, :]
        distance = jnp.linalg.norm(displacement, axis=-1)
        direction = displacement / jnp.maximum(
            distance[..., None], jnp.finfo(distance.dtype).tiny
        )
        retarded = history.times[None, :, None] - distance / self.sound_speed
        radial_force = jnp.sum(history.force[None, :, :, :] * direction, axis=-1)
        radial_velocity = jnp.sum(
            history.source_velocity[None, :, :, :] * history.normal[None, :, :, :],
            axis=-1,
        )
        dt = jnp.diff(history.times)
        force_rate = jnp.concatenate(
            (
                jnp.zeros_like(radial_force[:, :1]),
                jnp.diff(radial_force, axis=1) / dt[None, :, None],
            ),
            axis=1,
        )
        velocity_rate = jnp.concatenate(
            (
                jnp.zeros_like(radial_velocity[:, :1]),
                jnp.diff(radial_velocity, axis=1) / dt[None, :, None],
            ),
            axis=1,
        )
        loading = jnp.sum(
            force_rate
            / (
                4.0
                * jnp.pi
                * self.sound_speed
                * jnp.maximum(distance, jnp.finfo(distance.dtype).tiny)
            ),
            axis=-1,
        )
        thickness = self.ambient_density * jnp.sum(
            history.area[None, :, :]
            * velocity_rate
            / (4.0 * jnp.pi * jnp.maximum(distance, jnp.finfo(distance.dtype).tiny)),
            axis=-1,
        )
        pressure = loading + thickness
        finite = jnp.all(jnp.isfinite(pressure))
        return FWHObserverResult(
            observer, pressure, retarded, thickness, loading, finite, self.plan_id
        )


class BroadbandSectionNoiseResult(StrictModule):
    frequency: Array
    power_spectral_density: Array
    integrated_power: Array
    finite: Array
    model_id: str = eqx.field(static=True)


class BroadbandSectionNoisePlan(StrictModule, NonTrainableState):
    model_id: str = eqx.field(static=True)

    def __init__(self):
        self.model_id = canonical_fingerprint({"kind": "broadband-section-noise"})

    def evaluate(
        self,
        frequency: ArrayLike,
        relative_speed: ArrayLike,
        chord: ArrayLike,
        turbulence_intensity: ArrayLike,
        observer_distance: ArrayLike,
        /,
    ) -> BroadbandSectionNoiseResult:
        frequency_, speed, chord_, turbulence, distance = (
            jnp.asarray(value)
            for value in (
                frequency,
                relative_speed,
                chord,
                turbulence_intensity,
                observer_distance,
            )
        )
        if (
            frequency_.ndim != 1
            or jnp.any(frequency_ <= 0.0)
            or jnp.any(speed <= 0.0)
            or jnp.any(chord_ <= 0.0)
            or jnp.any(turbulence < 0.0)
            or jnp.any(distance <= 0.0)
        ):
            raise ValueError("Broadband section noise inputs are invalid.")
        strouhal = frequency_[:, None] * chord_.reshape((1, -1)) / speed.reshape((1, -1))
        shape = strouhal**2 * jnp.exp(-2.0 * strouhal)
        spectrum = (
            turbulence.reshape((1, -1)) ** 2
            * speed.reshape((1, -1)) ** 5
            * chord_.reshape((1, -1)) ** 2
            * shape
            / (distance.reshape((1, -1)) ** 2)
        )
        integrated = jnp.trapezoid(jnp.sum(spectrum, axis=-1), frequency_)
        finite = jnp.all(jnp.isfinite(spectrum))
        return BroadbandSectionNoiseResult(
            frequency_, spectrum, integrated, finite, self.model_id
        )


__all__ = [
    "AerodynamicLoadHistory",
    "BroadbandSectionNoisePlan",
    "BroadbandSectionNoiseResult",
    "FWHObserverResult",
    "FWHTonalAcousticsPlan",
]
