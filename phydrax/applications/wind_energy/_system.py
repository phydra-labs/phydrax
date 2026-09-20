#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Linear aero–hydro–servo–elastic wind-turbine dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ...structural_dynamics import LinearStructuralSystem, StructuralDynamicState


@dataclass(frozen=True, slots=True)
class WindTurbineState:
    structural: StructuralDynamicState
    rotor_azimuth_rad: Array


@dataclass(frozen=True, slots=True)
class WindTurbineStep:
    state: WindTurbineState
    aerodynamic_force_n: Array
    wave_force_n: Array
    generator_power_w: Array
    available_wind_power_w: Array
    mechanical_residual_norm: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class AeroHydroServoElasticSystem:
    dynamics: LinearStructuralSystem
    aerodynamic_force_shape: Array
    wave_force_shape_n_m: Array
    rotor_coordinate: int
    rotor_swept_area_m2: float
    air_density_kg_m3: float
    thrust_coefficient: float

    @classmethod
    def create(
        cls,
        mass: ArrayLike,
        structural_damping: ArrayLike,
        stiffness: ArrayLike,
        aerodynamic_damping: ArrayLike,
        hydrodynamic_added_mass: ArrayLike,
        hydrodynamic_damping: ArrayLike,
        hydrostatic_mooring_stiffness: ArrayLike,
        aerodynamic_force_shape: ArrayLike,
        wave_force_shape_n_m: ArrayLike,
        rotor_coordinate: int,
        /,
        *,
        rotor_swept_area_m2: float,
        air_density_kg_m3: float = 1.225,
        thrust_coefficient: float = 0.8,
    ) -> AeroHydroServoElasticSystem:
        mass_ = np.asarray(mass, dtype=np.float64) + np.asarray(
            hydrodynamic_added_mass, dtype=np.float64
        )
        damping = (
            np.asarray(structural_damping, dtype=np.float64)
            + np.asarray(aerodynamic_damping, dtype=np.float64)
            + np.asarray(hydrodynamic_damping, dtype=np.float64)
        )
        stiffness_ = np.asarray(stiffness, dtype=np.float64) + np.asarray(
            hydrostatic_mooring_stiffness, dtype=np.float64
        )
        dynamics = LinearStructuralSystem.create(mass_, damping, stiffness_)
        aero_shape = np.asarray(aerodynamic_force_shape, dtype=np.float64)
        wave_shape = np.asarray(wave_force_shape_n_m, dtype=np.float64)
        size = dynamics.size
        if aero_shape.shape != (size,) or wave_shape.shape != (size,):
            raise ValueError("Wind and wave generalized-force shapes must align.")
        if not 0 <= rotor_coordinate < size:
            raise ValueError("Wind-turbine rotor coordinate is outside the state basis.")
        if rotor_swept_area_m2 <= 0 or air_density_kg_m3 <= 0 or thrust_coefficient < 0:
            raise ValueError("Wind-turbine aerodynamic parameters are invalid.")
        return cls(
            dynamics,
            jnp.asarray(aero_shape),
            jnp.asarray(wave_shape),
            int(rotor_coordinate),
            float(rotor_swept_area_m2),
            float(air_density_kg_m3),
            float(thrust_coefficient),
        )

    def advance(
        self,
        state: WindTurbineState,
        wind_speed_m_s: float,
        wave_elevation_m: float,
        generator_torque_n_m: float,
        step_size_s: float,
        /,
    ) -> WindTurbineStep:
        if wind_speed_m_s < 0 or generator_torque_n_m < 0:
            raise ValueError(
                "Wind speed and opposing generator torque must be non-negative."
            )
        thrust = (
            0.5
            * self.air_density_kg_m3
            * self.rotor_swept_area_m2
            * self.thrust_coefficient
            * float(wind_speed_m_s) ** 2
        )
        aerodynamic_force = thrust * self.aerodynamic_force_shape
        wave_force = float(wave_elevation_m) * self.wave_force_shape_n_m
        generator_force = (
            jnp.zeros_like(aerodynamic_force)
            .at[self.rotor_coordinate]
            .set(-float(generator_torque_n_m))
        )
        structural = self.dynamics.newmark_step(
            state.structural,
            aerodynamic_force + wave_force + generator_force,
            step_size_s,
        )
        rotor_speed = structural.state.velocity[self.rotor_coordinate]
        azimuth = state.rotor_azimuth_rad + float(step_size_s) * rotor_speed
        generator_power = float(generator_torque_n_m) * rotor_speed
        available_power = (
            0.5
            * self.air_density_kg_m3
            * self.rotor_swept_area_m2
            * float(wind_speed_m_s) ** 3
        )
        finite = jnp.all(
            jnp.isfinite(jnp.asarray((azimuth, generator_power, available_power)))
        )
        return WindTurbineStep(
            WindTurbineState(structural.state, azimuth),
            aerodynamic_force,
            wave_force,
            generator_power,
            available_power,
            structural.equilibrium_residual_norm,
            structural.successful & finite,
        )


__all__ = [
    "AeroHydroServoElasticSystem",
    "WindTurbineState",
    "WindTurbineStep",
]
