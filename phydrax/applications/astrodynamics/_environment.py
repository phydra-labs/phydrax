#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._context import AstrodynamicsContext
from ._data import AstrodynamicsDataProvenance
from ._forces import AbstractAstrodynamicsForce, AstrodynamicsForceEvaluation
from ._status import AstrodynamicsStatus


def _norm(value: Array, /) -> Array:
    return jnp.sqrt(jnp.sum(value * value))


class SpaceWeatherTable(StrictModule, NonTrainableState):
    times: Array
    f107: Array
    f107_average: Array
    ap: Array
    provenance: AstrodynamicsDataProvenance
    product_id: str = eqx.field(static=True)

    def __init__(self, times, f107, f107_average, ap, provenance, /):
        values = tuple(
            np.asarray(value, dtype=float) for value in (times, f107, f107_average, ap)
        )
        if (
            values[0].ndim != 1
            or values[0].size < 2
            or any(v.shape != values[0].shape for v in values[1:])
            or any(np.any(~np.isfinite(v)) for v in values)
            or np.any(np.diff(values[0]) <= 0.0)
        ):
            raise ValueError("Space-weather arrays are invalid.")
        self.times, self.f107, self.f107_average, self.ap = tuple(
            jnp.asarray(v) for v in values
        )
        self.provenance = provenance
        self.product_id = canonical_fingerprint(
            {
                "kind": "space-weather-table",
                "nodes": values[0].tolist(),
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(self, time: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        query = jnp.asarray(time)
        support = (query >= self.times[0]) & (query <= self.times[-1])
        return (
            jnp.interp(query, self.times, self.f107),
            jnp.interp(query, self.times, self.f107_average),
            jnp.interp(query, self.times, self.ap),
            support,
        )


class ExponentialAtmosphere(StrictModule, NonTrainableState):
    reference_radius: Array
    reference_density: Array
    reference_altitude: Array
    scale_height: Array
    atmosphere_id: str = eqx.field(static=True)

    def __init__(
        self, reference_radius, reference_density, reference_altitude, scale_height, /
    ):
        self.reference_radius = jnp.asarray(reference_radius).reshape(())
        self.reference_density = jnp.asarray(reference_density).reshape(())
        self.reference_altitude = jnp.asarray(reference_altitude).reshape(())
        self.scale_height = jnp.asarray(scale_height).reshape(())
        if any(
            float(value) <= 0.0
            for value in (
                self.reference_radius,
                self.reference_density,
                self.scale_height,
            )
        ):
            raise ValueError("Atmosphere scales must be positive.")
        self.atmosphere_id = canonical_fingerprint(
            {
                "kind": "exponential-atmosphere",
                "radius": float(self.reference_radius),
                "density": float(self.reference_density),
                "altitude": float(self.reference_altitude),
                "height": float(self.scale_height),
            }
        )

    def density(self, position: Array, /) -> Array:
        altitude = _norm(position) - self.reference_radius
        return self.reference_density * jnp.exp(
            -(altitude - self.reference_altitude) / self.scale_height
        )


class AtmosphericDrag(AbstractAstrodynamicsForce):
    atmosphere: ExponentialAtmosphere
    context: AstrodynamicsContext
    drag_coefficient: Array
    area_to_mass: Array
    angular_velocity: Array
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        atmosphere,
        context,
        /,
        *,
        drag_coefficient,
        area_to_mass,
        angular_velocity=(0.0, 0.0, 7.292115146706979e-5),
    ):
        self.atmosphere = atmosphere
        self.context = context
        self.drag_coefficient = jnp.asarray(drag_coefficient).reshape(())
        self.area_to_mass = jnp.asarray(area_to_mass).reshape(())
        self.angular_velocity = jnp.asarray(angular_velocity)
        self.force_id = canonical_fingerprint(
            {
                "kind": "atmospheric-drag",
                "atmosphere": atmosphere.atmosphere_id,
                "context": context.context_id,
            }
        )

    def evaluate(self, time, state, args: Any = None, /):
        del time, args
        packed = jnp.asarray(state)
        position, velocity = packed[:3], packed[3:]
        density = self.atmosphere.density(position)
        relative = velocity - jnp.cross(self.angular_velocity, position)
        speed = _norm(relative)
        acceleration = (
            -0.5 * density * self.drag_coefficient * self.area_to_mass * speed * relative
        )
        valid = (
            jnp.all(jnp.isfinite(packed))
            & (self.drag_coefficient >= 0.0)
            & (self.area_to_mass >= 0.0)
            & jnp.isfinite(density)
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, 0.0),
            jnp.asarray(jnp.nan),
            status[None],
            valid,
            status,
            self.force_id,
        )


class EclipseGeometry(StrictModule, NonTrainableState):
    occulting_radius: Array
    source_radius: Array

    def illumination(
        self, spacecraft: Array, source: Array, occulting_center: Array, /
    ) -> Array:
        to_source = source - spacecraft
        to_occulter = occulting_center - spacecraft
        source_distance = _norm(to_source)
        occulting_distance = _norm(to_occulter)
        source_angle = jnp.arcsin(
            jnp.clip(self.source_radius / source_distance, 0.0, 1.0)
        )
        occulting_angle = jnp.arcsin(
            jnp.clip(self.occulting_radius / occulting_distance, 0.0, 1.0)
        )
        separation = jnp.arccos(
            jnp.clip(
                jnp.sum(to_source * to_occulter) / (source_distance * occulting_distance),
                -1.0,
                1.0,
            )
        )
        full = separation + source_angle <= occulting_angle
        none = separation >= source_angle + occulting_angle
        partial = jnp.clip(
            (separation - (occulting_angle - source_angle))
            / jnp.maximum(2.0 * source_angle, 1.0e-30),
            0.0,
            1.0,
        )
        return jnp.where(full, 0.0, jnp.where(none, 1.0, partial))


class SolarRadiationPressure(AbstractAstrodynamicsForce):
    source_position: Callable
    occulting_position: Callable
    eclipse: EclipseGeometry
    context: AstrodynamicsContext
    reference_pressure: Array
    reference_distance: Array
    reflectivity: Array
    area_to_mass: Array
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_position,
        occulting_position,
        eclipse,
        context,
        /,
        *,
        reference_pressure=4.56e-6,
        reference_distance=149597870700.0,
        reflectivity=1.0,
        area_to_mass=0.01,
        force_id="solar-radiation-pressure",
    ):
        if not callable(source_position) or not callable(occulting_position):
            raise TypeError("Radiation ephemeris providers must be callable.")
        self.source_position = source_position
        self.occulting_position = occulting_position
        self.eclipse = eclipse
        self.context = context
        self.reference_pressure = jnp.asarray(reference_pressure).reshape(())
        self.reference_distance = jnp.asarray(reference_distance).reshape(())
        self.reflectivity = jnp.asarray(reflectivity).reshape(())
        self.area_to_mass = jnp.asarray(area_to_mass).reshape(())
        self.force_id = str(force_id)

    def evaluate(self, time, state, args=None, /):
        packed = jnp.asarray(state)
        source = jnp.asarray(self.source_position(time, args))
        occulter = jnp.asarray(self.occulting_position(time, args))
        relative = packed[:3] - source
        distance = _norm(relative)
        illumination = self.eclipse.illumination(packed[:3], source, occulter)
        pressure = self.reference_pressure * (self.reference_distance / distance) ** 2
        acceleration = (
            illumination
            * pressure
            * self.reflectivity
            * self.area_to_mass
            * relative
            / distance
        )
        valid = (
            jnp.all(jnp.isfinite(acceleration))
            & (distance > 0.0)
            & (self.area_to_mass >= 0.0)
        )
        status = jnp.where(
            valid,
            int(AstrodynamicsStatus.SUCCESS),
            int(AstrodynamicsStatus.INVALID_DOMAIN),
        ).astype(jnp.int32)
        return AstrodynamicsForceEvaluation(
            jnp.where(valid, acceleration, 0.0),
            jnp.asarray(jnp.nan),
            status[None],
            valid,
            status,
            self.force_id,
        )


class ThermalRadiationPressure(AbstractAstrodynamicsForce):
    """Planetary thermal/albedo pressure through an explicit radiation geometry."""

    radiation: SolarRadiationPressure
    context: AstrodynamicsContext
    force_id: str = eqx.field(static=True)

    def __init__(
        self,
        radiation: SolarRadiationPressure,
        /,
        *,
        force_id: str = "thermal-radiation-pressure",
    ):
        if not isinstance(radiation, SolarRadiationPressure):
            raise TypeError("radiation must be a SolarRadiationPressure.")
        self.radiation = radiation
        self.context = radiation.context
        self.force_id = str(force_id)

    def evaluate(self, time, state, args=None, /):
        result = self.radiation.evaluate(time, state, args)
        return AstrodynamicsForceEvaluation(
            result.acceleration,
            result.potential,
            result.component_status,
            result.valid,
            result.status,
            self.force_id,
        )


__all__ = [
    "AtmosphericDrag",
    "EclipseGeometry",
    "ExponentialAtmosphere",
    "SolarRadiationPressure",
    "SpaceWeatherTable",
    "ThermalRadiationPressure",
]
