#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class LatticeBoltzmannScaling(StrictModule, NonTrainableState):
    """Physical-to-lattice conversion bound to one grid spacing and time step."""

    cell_size: Array
    time_step: Array
    reference_density: Array
    sound_speed_squared: Array
    scaling_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_size: float,
        time_step: float,
        reference_density: float,
        /,
        *,
        sound_speed_squared: float = 1.0 / 3.0,
    ):
        values = tuple(
            float(value)
            for value in (
                cell_size,
                time_step,
                reference_density,
                sound_speed_squared,
            )
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("LBM scaling values must be finite and positive.")
        dx, dt, rho0, cs2 = values
        self.cell_size = jnp.asarray(dx, dtype=jnp.float64)
        self.time_step = jnp.asarray(dt, dtype=jnp.float64)
        self.reference_density = jnp.asarray(rho0, dtype=jnp.float64)
        self.sound_speed_squared = jnp.asarray(cs2, dtype=jnp.float64)
        self.scaling_id = canonical_fingerprint(
            {
                "kind": "lattice-boltzmann-scaling",
                "cell_size": dx,
                "time_step": dt,
                "reference_density": rho0,
                "sound_speed_squared": cs2,
            }
        )

    def lattice_velocity(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity)
        return (
            value
            * self.time_step.astype(value.dtype)
            / self.cell_size.astype(value.dtype)
        )

    def physical_velocity(self, velocity: ArrayLike, /) -> Array:
        value = jnp.asarray(velocity)
        return (
            value
            * self.cell_size.astype(value.dtype)
            / self.time_step.astype(value.dtype)
        )

    def lattice_viscosity(self, viscosity: ArrayLike, /) -> Array:
        value = jnp.asarray(viscosity)
        return (
            value
            * self.time_step.astype(value.dtype)
            / self.cell_size.astype(value.dtype) ** 2
        )

    def physical_viscosity(self, viscosity: ArrayLike, /) -> Array:
        value = jnp.asarray(viscosity)
        return (
            value
            * self.cell_size.astype(value.dtype) ** 2
            / self.time_step.astype(value.dtype)
        )

    def lattice_acceleration(self, acceleration: ArrayLike, /) -> Array:
        value = jnp.asarray(acceleration)
        return (
            value
            * self.time_step.astype(value.dtype) ** 2
            / self.cell_size.astype(value.dtype)
        )

    def physical_acceleration(self, acceleration: ArrayLike, /) -> Array:
        value = jnp.asarray(acceleration)
        return (
            value
            * self.cell_size.astype(value.dtype)
            / self.time_step.astype(value.dtype) ** 2
        )

    def lattice_density(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        return value / self.reference_density.astype(value.dtype)

    def physical_density(self, density: ArrayLike, /) -> Array:
        value = jnp.asarray(density)
        return value * self.reference_density.astype(value.dtype)

    def relaxation_rate(self, kinematic_viscosity: ArrayLike, /) -> Array:
        physical = jnp.asarray(kinematic_viscosity)
        lattice = self.lattice_viscosity(physical)
        cs2 = self.sound_speed_squared.astype(lattice.dtype)
        rate = 1.0 / (0.5 + lattice / cs2)
        invalid = (
            (physical.shape != ())
            | ~jnp.isfinite(physical)
            | (physical <= 0.0)
            | ~jnp.isfinite(rate)
            | (rate <= 0.0)
            | (rate >= 2.0)
        )
        return eqx.error_if(
            rate,
            invalid,
            "Kinematic viscosity produces an invalid LBM relaxation rate.",
        )

    def physical_pressure(self, lattice_density: ArrayLike, /) -> Array:
        density = jnp.asarray(lattice_density)
        gauge_lattice = self.sound_speed_squared.astype(density.dtype) * (density - 1.0)
        velocity_scale = self.cell_size.astype(density.dtype) / self.time_step.astype(
            density.dtype
        )
        return (
            self.reference_density.astype(density.dtype)
            * velocity_scale**2
            * gauge_lattice
        )


__all__ = ["LatticeBoltzmannScaling"]
