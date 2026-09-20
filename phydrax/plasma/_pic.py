#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Periodic one-dimensional electrostatic particle-in-cell workflow."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class ElectrostaticPICState:
    position_m: Array
    velocity_m_s: Array
    time_s: Array


@dataclass(frozen=True, slots=True)
class ElectrostaticPICStep:
    state: ElectrostaticPICState
    charge_density_c_m: Array
    electric_field_v_m: Array
    potential_v: Array
    physical_particle_charge_c: Array
    neutralized_charge_residual_c: Array
    gauss_residual_norm_c_m: Array
    kinetic_energy_j: Array
    electric_energy_j: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class ElectrostaticPIC1D:
    domain_length_m: float
    grid_point_count: int
    permittivity_f_m: float

    def __post_init__(self):
        if (
            self.domain_length_m <= 0
            or self.grid_point_count < 4
            or self.permittivity_f_m <= 0
        ):
            raise ValueError("Electrostatic PIC grid and permittivity must be positive.")

    @property
    def spacing_m(self) -> float:
        return self.domain_length_m / self.grid_point_count

    def deposit_charge(
        self,
        position_m: ArrayLike,
        particle_charge_c: ArrayLike,
        macro_weight: ArrayLike,
        /,
    ) -> tuple[Array, Array]:
        position = jnp.mod(jnp.asarray(position_m), self.domain_length_m)
        charge = jnp.broadcast_to(jnp.asarray(particle_charge_c), position.shape)
        weight = jnp.broadcast_to(jnp.asarray(macro_weight), position.shape)
        coordinate = position / self.spacing_m
        left = jnp.floor(coordinate).astype(jnp.int32) % self.grid_point_count
        fraction = coordinate - jnp.floor(coordinate)
        right = (left + 1) % self.grid_point_count
        density = jnp.zeros(
            (self.grid_point_count,), dtype=jnp.result_type(charge, weight)
        )
        density = density.at[left].add(charge * weight * (1 - fraction) / self.spacing_m)
        density = density.at[right].add(charge * weight * fraction / self.spacing_m)
        physical_charge = jnp.sum(charge * weight)
        neutralized = density - jnp.mean(density)
        return neutralized, physical_charge

    def solve_field(self, charge_density_c_m: ArrayLike, /) -> tuple[Array, Array, Array]:
        density = jnp.asarray(charge_density_c_m)
        if density.shape != (self.grid_point_count,):
            raise ValueError("PIC charge density does not match the periodic grid.")
        wave_number = (
            2 * jnp.pi * jnp.fft.fftfreq(self.grid_point_count, d=self.spacing_m)
        )
        transformed = jnp.fft.fft(density)
        nonzero = wave_number != 0
        potential_hat = jnp.where(
            nonzero,
            transformed / (self.permittivity_f_m * jnp.where(nonzero, wave_number**2, 1)),
            0,
        )
        electric_hat = -1j * wave_number * potential_hat
        potential = jnp.fft.ifft(potential_hat).real
        electric = jnp.fft.ifft(electric_hat).real
        divergence = jnp.fft.ifft(1j * wave_number * electric_hat).real
        gauss_residual = divergence - density / self.permittivity_f_m
        gauss_norm = self.permittivity_f_m * jnp.sqrt(
            self.spacing_m * jnp.sum(gauss_residual**2)
        )
        return potential, electric, gauss_norm

    def interpolate_field(
        self, position_m: ArrayLike, electric_field_v_m: ArrayLike, /
    ) -> Array:
        position = jnp.mod(jnp.asarray(position_m), self.domain_length_m)
        field = jnp.asarray(electric_field_v_m)
        coordinate = position / self.spacing_m
        left = jnp.floor(coordinate).astype(jnp.int32) % self.grid_point_count
        fraction = coordinate - jnp.floor(coordinate)
        right = (left + 1) % self.grid_point_count
        return (1 - fraction) * field[left] + fraction * field[right]

    def advance(
        self,
        state: ElectrostaticPICState,
        particle_charge_c: ArrayLike,
        particle_mass_kg: ArrayLike,
        macro_weight: ArrayLike,
        step_size_s: float,
        /,
    ) -> ElectrostaticPICStep:
        position = jnp.asarray(state.position_m)
        velocity = jnp.asarray(state.velocity_m_s)
        charge = jnp.broadcast_to(jnp.asarray(particle_charge_c), position.shape)
        mass = jnp.broadcast_to(jnp.asarray(particle_mass_kg), position.shape)
        weight = jnp.broadcast_to(jnp.asarray(macro_weight), position.shape)
        if position.ndim != 1 or velocity.shape != position.shape:
            raise ValueError(
                "PIC particle position and velocity must be aligned vectors."
            )
        if bool(jnp.any(mass <= 0) | jnp.any(weight <= 0)) or step_size_s <= 0:
            raise ValueError("PIC masses, macro weights, and step size must be positive.")
        density, physical_charge = self.deposit_charge(position, charge, weight)
        _, electric, _ = self.solve_field(density)
        acceleration = charge / mass * self.interpolate_field(position, electric)
        half_velocity = velocity + 0.5 * float(step_size_s) * acceleration
        next_position = jnp.mod(
            position + float(step_size_s) * half_velocity, self.domain_length_m
        )
        next_density, next_physical_charge = self.deposit_charge(
            next_position, charge, weight
        )
        potential, next_electric, gauss_norm = self.solve_field(next_density)
        next_acceleration = (
            charge / mass * self.interpolate_field(next_position, next_electric)
        )
        next_velocity = half_velocity + 0.5 * float(step_size_s) * next_acceleration
        kinetic = 0.5 * jnp.sum(weight * mass * next_velocity**2)
        electric_energy = (
            0.5 * self.permittivity_f_m * self.spacing_m * jnp.sum(next_electric**2)
        )
        neutralized_residual = self.spacing_m * jnp.sum(next_density)
        successful = (
            jnp.all(jnp.isfinite(next_position))
            & jnp.all(jnp.isfinite(next_velocity))
            & jnp.isclose(next_physical_charge, physical_charge)
        )
        return ElectrostaticPICStep(
            ElectrostaticPICState(
                next_position,
                next_velocity,
                jnp.asarray(state.time_s) + float(step_size_s),
            ),
            next_density,
            next_electric,
            potential,
            next_physical_charge,
            neutralized_residual,
            gauss_norm,
            kinetic,
            electric_energy,
            successful,
        )


__all__ = ["ElectrostaticPIC1D", "ElectrostaticPICState", "ElectrostaticPICStep"]
