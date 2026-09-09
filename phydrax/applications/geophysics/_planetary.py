#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.linalg as la

from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...interchange import ReferenceBodyContract


class RadialBodyModel(StrictModule, NonTrainableState):
    radius_m: Array
    density_kg_m3: Array
    p_velocity_m_s: Array
    s_velocity_m_s: Array
    temperature_K: Array
    heat_capacity_J_kg_K: Array
    thermal_conductivity_W_m_K: Array
    body: ReferenceBodyContract

    def __init__(
        self,
        radius_m: ArrayLike,
        density_kg_m3: ArrayLike,
        p_velocity_m_s: ArrayLike,
        s_velocity_m_s: ArrayLike,
        temperature_K: ArrayLike,
        heat_capacity_J_kg_K: ArrayLike,
        thermal_conductivity_W_m_K: ArrayLike,
        body: ReferenceBodyContract,
        /,
    ):
        radius = jnp.asarray(radius_m)
        values = tuple(
            jnp.asarray(value)
            for value in (
                density_kg_m3,
                p_velocity_m_s,
                s_velocity_m_s,
                temperature_K,
                heat_capacity_J_kg_K,
                thermal_conductivity_W_m_K,
            )
        )
        if not isinstance(body, ReferenceBodyContract):
            raise TypeError("Radial model requires ReferenceBodyContract.")
        if (
            radius.ndim != 1
            or radius.size < 3
            or any(value.shape != radius.shape for value in values)
        ):
            raise ValueError(
                "Radial body properties must match three or more radius nodes."
            )
        radius = eqx.error_if(
            radius,
            jnp.any(~jnp.isfinite(radius))
            | jnp.any(jnp.diff(radius) <= 0)
            | (radius[0] < 0)
            | (
                jnp.abs(radius[-1] - body.semi_major_axis_m)
                > 1e-8 * body.semi_major_axis_m
            )
            | any(jnp.any(~jnp.isfinite(value)) for value in values)
            | jnp.any(values[0] <= 0)
            | jnp.any(values[1] <= 0)
            | jnp.any(values[2] < 0)
            | jnp.any(values[1] < values[2])
            | jnp.any(values[3] <= 0)
            | jnp.any(values[4] <= 0)
            | jnp.any(values[5] <= 0),
            "Radial body radii/properties must be finite and physical.",
        )
        self.radius_m = radius
        (
            self.density_kg_m3,
            self.p_velocity_m_s,
            self.s_velocity_m_s,
            self.temperature_K,
            self.heat_capacity_J_kg_K,
            self.thermal_conductivity_W_m_K,
        ) = values
        self.body = body


class SphericalRayResult(StrictModule):
    ray_parameter_s: Array
    epicentral_distance_radians: Array
    travel_time_s: Array
    turning_radius_m: Array
    successful: Array
    derivative_available: Array


class SphericalRayPlan(StrictModule, NonTrainableState):
    model: RadialBodyModel
    phase: str = eqx.field(static=True)

    def __init__(self, model: RadialBodyModel, phase: str, /):
        if not isinstance(model, RadialBodyModel) or phase not in ("P", "S"):
            raise ValueError("Spherical ray phase must be P or S on a radial body model.")
        if phase == "S" and bool(jnp.any(model.s_velocity_m_s <= 0)):
            raise ValueError("S rays cannot traverse zero-shear fluid nodes.")
        self.model, self.phase = model, phase

    def evaluate(self, ray_parameter_s: ArrayLike, /) -> SphericalRayResult:
        parameter = jnp.asarray(ray_parameter_s)
        if parameter.shape != ():
            raise ValueError("Spherical ray parameter must be scalar seconds.")
        parameter = eqx.error_if(
            parameter,
            ~jnp.isfinite(parameter) | (parameter < 0),
            "Spherical ray parameter must be finite and nonnegative.",
        )
        velocity = (
            self.model.p_velocity_m_s if self.phase == "P" else self.model.s_velocity_m_s
        )
        radius = self.model.radius_m
        argument = 1.0 / velocity**2 - parameter**2 / jnp.where(
            radius > 0, radius**2, 1.0
        )
        propagating = argument > 0
        transitions = ~propagating[:-1] & propagating[1:]
        index = jnp.argmax(transitions) + 1
        central = parameter == 0
        has_turn = jnp.any(transitions) | central
        start = jnp.where(central, 0, jnp.where(has_turn, index, 1))
        q = jnp.sqrt(jnp.maximum(argument, jnp.finfo(argument.dtype).tiny))
        radial_mask = jnp.arange(radius.size) >= start
        distance_integrand = parameter / (jnp.where(radius > 0, radius**2, 1.0) * q)
        time_integrand = 1.0 / (velocity**2 * q)
        dr = jnp.diff(radius)
        interval_mask = radial_mask[:-1] & radial_mask[1:]
        distance = 2.0 * jnp.sum(
            jnp.where(
                interval_mask,
                0.5 * dr * (distance_integrand[:-1] + distance_integrand[1:]),
                0.0,
            )
        )
        travel = 2.0 * jnp.sum(
            jnp.where(
                interval_mask, 0.5 * dr * (time_integrand[:-1] + time_integrand[1:]), 0.0
            )
        )
        successful = has_turn & jnp.isfinite(distance) & jnp.isfinite(travel)
        margin = jnp.min(jnp.abs(argument[1:]))
        return SphericalRayResult(
            parameter,
            distance,
            travel,
            radius[start],
            successful,
            successful & ~central & (margin > 1e-10 / jnp.max(velocity) ** 2),
        )


class PlanetaryPotentialPlan(StrictModule, NonTrainableState):
    body: ReferenceBodyContract

    def potential(self, body_fixed_position_m: ArrayLike, /) -> Array:
        position = jnp.asarray(body_fixed_position_m)
        if position.shape[-1] != 3:
            raise ValueError("Planetary potential position needs trailing XYZ.")
        radius = jnp.sqrt(jnp.sum(position**2, axis=-1))
        radius = eqx.error_if(
            radius,
            jnp.any(~jnp.isfinite(radius)) | jnp.any(radius <= 0),
            "Planetary potential position must be finite and nonzero.",
        )
        gravitational = self.body.gravitational_parameter_m3_s2 / radius
        centrifugal = (
            0.5
            * self.body.rotation_rate_rad_s**2
            * jnp.sum(position[..., :2] ** 2, axis=-1)
        )
        return gravitational + centrifugal


class RadialThermalStepResult(StrictModule):
    temperature_K: Array
    energy_residual_J: Array
    successful: Array


class RadialThermalConductionPlan(StrictModule, NonTrainableState):
    model: RadialBodyModel
    cell_volume_m3: Array
    interface_area_m2: Array

    def __init__(self, model: RadialBodyModel, /):
        if not isinstance(model, RadialBodyModel):
            raise TypeError("Radial thermal conduction requires RadialBodyModel.")
        radius = np.asarray(model.radius_m)
        boundaries = np.empty(radius.size + 1)
        boundaries[1:-1] = 0.5 * (radius[:-1] + radius[1:])
        boundaries[0] = max(0.0, radius[0] - 0.5 * (radius[1] - radius[0]))
        boundaries[-1] = radius[-1] + 0.5 * (radius[-1] - radius[-2])
        self.model = model
        self.cell_volume_m3 = jnp.asarray(
            4.0 * np.pi / 3.0 * (boundaries[1:] ** 3 - boundaries[:-1] ** 3)
        )
        self.interface_area_m2 = jnp.asarray(
            4.0 * np.pi * (0.5 * (radius[:-1] + radius[1:])) ** 2
        )

    def step(
        self,
        temperature_K: ArrayLike,
        dt_s: ArrayLike,
        /,
        *,
        volumetric_heating_W_m3: ArrayLike = 0.0,
        surface_temperature_K: ArrayLike,
    ) -> RadialThermalStepResult:
        temperature = jnp.asarray(temperature_K)
        dt = jnp.asarray(dt_s)
        heating = jnp.broadcast_to(
            jnp.asarray(volumetric_heating_W_m3), temperature.shape
        )
        surface = jnp.asarray(surface_temperature_K)
        if (
            temperature.shape != self.model.radius_m.shape
            or dt.shape != ()
            or surface.shape != ()
        ):
            raise ValueError("Radial thermal state/timestep/surface shapes are invalid.")
        capacity = (
            self.model.density_kg_m3
            * self.model.heat_capacity_J_kg_K
            * self.cell_volume_m3
        )
        radius = self.model.radius_m
        conductivity = self.model.thermal_conductivity_W_m_K
        interface = (
            2.0
            * conductivity[:-1]
            * conductivity[1:]
            / (conductivity[:-1] + conductivity[1:])
        )
        conductance = interface * self.interface_area_m2 / jnp.diff(radius)
        count = temperature.size
        matrix = jnp.diag(capacity / dt)
        matrix = matrix.at[jnp.arange(count - 1), jnp.arange(count - 1)].add(conductance)
        matrix = matrix.at[jnp.arange(1, count), jnp.arange(1, count)].add(conductance)
        matrix = matrix.at[jnp.arange(count - 1), jnp.arange(1, count)].add(-conductance)
        matrix = matrix.at[jnp.arange(1, count), jnp.arange(count - 1)].add(-conductance)
        surface_conductance = (
            conductivity[-1] * self.interface_area_m2[-1] / (radius[-1] - radius[-2])
        )
        matrix = matrix.at[-1, -1].add(surface_conductance)
        rhs = capacity / dt * temperature + heating * self.cell_volume_m3
        rhs = rhs.at[-1].add(surface_conductance * surface)
        space = la.ArraySpace((count,), dtype=temperature.dtype)
        operator = la.DenseLinearOperator(
            matrix,
            source=space,
            target=space,
            properties=la.OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={
                    "self_adjoint": "construction",
                    "positive_definite": "construction",
                },
            ),
        )
        result = la.solve(
            la.LinearSystem(operator),
            rhs,
            policy=la.LinearSolvePolicy(
                la.DenseCholesky(), failure=la.FailurePolicy("status")
            ),
        )
        next_temperature = result.value
        stored = jnp.sum(capacity * (next_temperature - temperature))
        boundary = dt * surface_conductance * (next_temperature[-1] - surface)
        generated = dt * jnp.sum(heating * self.cell_volume_m3)
        residual = stored + boundary - generated
        successful = (
            result.successful
            & jnp.all(jnp.isfinite(next_temperature))
            & jnp.all(next_temperature > 0)
        )
        return RadialThermalStepResult(next_temperature, residual, successful)


__all__ = [
    "PlanetaryPotentialPlan",
    "RadialBodyModel",
    "RadialThermalConductionPlan",
    "RadialThermalStepResult",
    "SphericalRayPlan",
    "SphericalRayResult",
]
