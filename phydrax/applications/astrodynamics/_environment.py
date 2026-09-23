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

from phydrax._interpolation import linear_interpolate

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
            np.asarray(value, dtype=np.float64)
            for value in (times, f107, f107_average, ap)
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
                "values": values,
                "provenance": provenance.provenance_id,
            }
        )

    def evaluate(self, time: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
        query = jnp.asarray(time)
        support = (query >= self.times[0]) & (query <= self.times[-1])
        return (
            linear_interpolate(self.times, self.f107, query).values,
            linear_interpolate(self.times, self.f107_average, query).values,
            linear_interpolate(self.times, self.ap, query).values,
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
    source_parameters_id: str = eqx.field(static=True)

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
        if not isinstance(atmosphere, ExponentialAtmosphere):
            raise TypeError("atmosphere must be an ExponentialAtmosphere.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        coefficient = np.asarray(drag_coefficient, dtype=np.float64)
        area = np.asarray(area_to_mass, dtype=np.float64)
        angular = np.asarray(angular_velocity, dtype=np.float64)
        if coefficient.shape != () or area.shape != () or angular.shape != (3,):
            raise ValueError(
                "Drag coefficients must be scalars and angular_velocity a three-vector."
            )
        if (
            not np.isfinite(coefficient)
            or coefficient < 0.0
            or not np.isfinite(area)
            or area < 0.0
            or np.any(~np.isfinite(angular))
        ):
            raise ValueError(
                "Atmospheric drag parameters must be finite and nonnegative."
            )
        self.atmosphere = atmosphere
        self.context = context
        self.drag_coefficient = jnp.asarray(coefficient)
        self.area_to_mass = jnp.asarray(area)
        self.angular_velocity = jnp.asarray(angular)
        self.source_parameters_id = canonical_fingerprint(
            {
                "drag_coefficient": coefficient,
                "area_to_mass": area,
                "angular_velocity": angular,
            }
        )
        self.force_id = canonical_fingerprint(
            {
                "kind": "atmospheric-drag",
                "atmosphere": atmosphere.atmosphere_id,
                "context": context.context_id,
                "parameters": self.source_parameters_id,
            }
        )

    def evaluate(self, time, state, args: Any = None, /):
        del time, args
        packed = jnp.asarray(state)
        if packed.shape != (6,):
            raise ValueError("Atmospheric drag state must have shape (6,).")
        position, velocity = packed[:3], packed[3:]
        density = self.atmosphere.density(position)
        relative = velocity - jnp.cross(self.angular_velocity, position)
        speed = _norm(relative)
        acceleration = (
            -0.5 * density * self.drag_coefficient * self.area_to_mass * speed * relative
        )
        finite = (
            jnp.all(jnp.isfinite(packed))
            & jnp.all(jnp.isfinite(self.angular_velocity))
            & jnp.isfinite(self.drag_coefficient)
            & jnp.isfinite(self.area_to_mass)
            & jnp.isfinite(density)
            & jnp.all(jnp.isfinite(acceleration))
        )
        valid = finite & (self.drag_coefficient >= 0.0) & (self.area_to_mass >= 0.0)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
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
    source_provider_id: str = eqx.field(static=True)
    occulting_provider_id: str = eqx.field(static=True)

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
        source_provider_id,
        occulting_provider_id,
    ):
        if not callable(source_position) or not callable(occulting_position):
            raise TypeError("Radiation ephemeris providers must be callable.")
        if not isinstance(eclipse, EclipseGeometry):
            raise TypeError("eclipse must be an EclipseGeometry.")
        if not isinstance(context, AstrodynamicsContext):
            raise TypeError("context must be an AstrodynamicsContext.")
        source_id = str(source_provider_id).strip()
        occulting_id = str(occulting_provider_id).strip()
        declared_id = str(force_id).strip()
        values = np.asarray(
            (reference_pressure, reference_distance, reflectivity, area_to_mass),
            dtype=np.float64,
        )
        radii = np.asarray(
            (eclipse.occulting_radius, eclipse.source_radius), dtype=np.float64
        )
        if not source_id or not occulting_id or not declared_id:
            raise ValueError("Radiation force and provider identities must be non-empty.")
        if (
            np.any(~np.isfinite(values))
            or values[0] <= 0.0
            or values[1] <= 0.0
            or values[2] < 0.0
            or values[3] < 0.0
            or np.any(~np.isfinite(radii))
            or np.any(radii <= 0.0)
        ):
            raise ValueError("Solar-radiation parameters must be finite and physical.")
        self.source_position = source_position
        self.occulting_position = occulting_position
        self.eclipse = eclipse
        self.context = context
        self.reference_pressure = jnp.asarray(values[0])
        self.reference_distance = jnp.asarray(values[1])
        self.reflectivity = jnp.asarray(values[2])
        self.area_to_mass = jnp.asarray(values[3])
        self.source_provider_id = source_id
        self.occulting_provider_id = occulting_id
        self.force_id = canonical_fingerprint(
            {
                "kind": "solar-radiation-pressure",
                "declared_id": declared_id,
                "context": context.context_id,
                "source_provider": source_id,
                "occulting_provider": occulting_id,
                "eclipse_radii": radii,
                "parameters": values,
            }
        )

    def evaluate(self, time, state, args=None, /):
        packed = jnp.asarray(state)
        source = jnp.asarray(self.source_position(time, args))
        occulter = jnp.asarray(self.occulting_position(time, args))
        if packed.shape != (6,) or source.shape != (3,) or occulter.shape != (3,):
            raise ValueError(
                "Radiation state and ephemeris positions have invalid shapes."
            )
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
        finite = (
            jnp.all(jnp.isfinite(packed))
            & jnp.all(jnp.isfinite(source))
            & jnp.all(jnp.isfinite(occulter))
            & jnp.all(jnp.isfinite(acceleration))
        )
        valid = finite & (distance > 0.0)
        status = jnp.where(
            ~finite,
            int(AstrodynamicsStatus.NONFINITE_INPUT),
            jnp.where(
                valid,
                int(AstrodynamicsStatus.SUCCESS),
                int(AstrodynamicsStatus.INVALID_DOMAIN),
            ),
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
