#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Many-particle phoretic drift with Stokes hydrodynamic interactions."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..ein import contract


@dataclass(frozen=True, slots=True)
class PhoreticCloudStep:
    position_m: Array
    candidate_position_m: Array
    velocity_m_s: Array
    hydrodynamic_mobility: Array
    force_power_w: Array
    minimum_surface_separation_m: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class HydrodynamicPhoreticSolver:
    dynamic_viscosity_pa_s: float
    particle_radii_m: Array
    spatial_dimension: int

    @classmethod
    def create(
        cls,
        dynamic_viscosity_pa_s: float,
        particle_radii_m: ArrayLike,
        spatial_dimension: int,
        /,
    ) -> HydrodynamicPhoreticSolver:
        radii = np.asarray(particle_radii_m, dtype=np.float64)
        if not isfinite(dynamic_viscosity_pa_s) or dynamic_viscosity_pa_s <= 0:
            raise ValueError("Phoresis dynamic viscosity must be finite and positive.")
        if (
            radii.ndim != 1
            or radii.size == 0
            or not np.all(np.isfinite(radii))
            or np.any(radii <= 0)
        ):
            raise ValueError("Phoretic particle radii must be a finite positive vector.")
        if spatial_dimension != 3:
            raise ValueError("Oseen hydrodynamic phoresis requires three dimensions.")
        return cls(float(dynamic_viscosity_pa_s), jnp.asarray(radii), spatial_dimension)

    def _minimum_surface_gap(self, positions: Array, /) -> Array:
        minimum = jnp.asarray(jnp.inf, dtype=positions.dtype)
        count = self.particle_radii_m.size
        for left in range(count):
            for right in range(left + 1, count):
                distance = jnp.linalg.norm(positions[right] - positions[left])
                gap = (
                    distance - self.particle_radii_m[left] - self.particle_radii_m[right]
                )
                minimum = jnp.minimum(minimum, gap)
        return minimum

    def mobility(self, position_m: ArrayLike, /) -> tuple[Array, Array]:
        positions = jnp.asarray(position_m)
        count = self.particle_radii_m.size
        dimension = self.spatial_dimension
        if positions.shape != (count, dimension):
            raise ValueError("Phoretic positions do not match particle topology.")
        positions = eqx.error_if(
            positions,
            jnp.any(~jnp.isfinite(positions)),
            "Phoretic positions must be finite.",
        )
        minimum_gap = self._minimum_surface_gap(positions)
        positions = eqx.error_if(
            positions,
            minimum_gap <= 0,
            "Oseen mobility requires strictly separated particle surfaces.",
        )
        identity = jnp.eye(dimension, dtype=positions.dtype)
        matrix = jnp.zeros((count * dimension, count * dimension), dtype=positions.dtype)
        minimum_gap = self._minimum_surface_gap(positions)
        for left in range(count):
            diagonal = identity / (
                6 * jnp.pi * self.dynamic_viscosity_pa_s * self.particle_radii_m[left]
            )
            row = slice(left * dimension, (left + 1) * dimension)
            matrix = matrix.at[row, row].set(diagonal)
            for right in range(left + 1, count):
                delta = positions[right] - positions[left]
                distance = jnp.sqrt(contract("d,d->", delta, delta))
                gap = (
                    distance - self.particle_radii_m[left] - self.particle_radii_m[right]
                )
                minimum_gap = jnp.minimum(minimum_gap, gap)
                direction = delta / distance
                pair = (identity + direction[:, None] * direction[None, :]) / (
                    8 * jnp.pi * self.dynamic_viscosity_pa_s * distance
                )
                other = slice(right * dimension, (right + 1) * dimension)
                matrix = matrix.at[row, other].set(pair)
                matrix = matrix.at[other, row].set(pair.T)
        if count == 1:
            minimum_gap = jnp.asarray(jnp.inf, dtype=positions.dtype)
        return matrix, minimum_gap

    def advance(
        self,
        position_m: ArrayLike,
        fluid_velocity_m_s: ArrayLike,
        phoretic_slip_velocity_m_s: ArrayLike,
        external_force_n: ArrayLike,
        step_size_s: float,
        /,
    ) -> PhoreticCloudStep:
        positions = jnp.asarray(position_m)
        fluid = jnp.asarray(fluid_velocity_m_s)
        slip = jnp.asarray(phoretic_slip_velocity_m_s)
        force = jnp.asarray(external_force_n)
        if any(value.shape != positions.shape for value in (fluid, slip, force)):
            raise ValueError("Phoretic velocity and force fields must match positions.")
        if not isfinite(step_size_s) or step_size_s <= 0:
            raise ValueError("Phoretic step size must be finite and positive.")
        positions = eqx.error_if(
            positions,
            jnp.any(
                ~jnp.isfinite(positions)
                | ~jnp.isfinite(fluid)
                | ~jnp.isfinite(slip)
                | ~jnp.isfinite(force)
            ),
            "Phoretic positions, velocities, and forces must be finite.",
        )
        mobility, minimum_gap = self.mobility(positions)
        induced = (mobility @ force.reshape((-1,))).reshape(positions.shape)
        velocity = fluid + slip + induced
        candidate = positions + float(step_size_s) * velocity
        candidate_minimum_gap = self._minimum_surface_gap(candidate)
        force_power = contract("qd,qd->", force, velocity - fluid - slip)
        successful = (
            jnp.all(jnp.isfinite(candidate))
            & jnp.all(jnp.isfinite(velocity))
            & jnp.isfinite(force_power)
            & (force_power >= -1e-12)
            & (candidate_minimum_gap > 0)
        )
        accepted = jnp.where(successful, candidate, positions)
        return PhoreticCloudStep(
            accepted,
            candidate,
            velocity,
            mobility,
            force_power,
            candidate_minimum_gap,
            successful,
        )


__all__ = ["HydrodynamicPhoreticSolver", "PhoreticCloudStep"]
