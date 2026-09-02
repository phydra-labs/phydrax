#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from ..equations._kinetic_gas import MonatomicBGKCollisionPlan


class KineticSyntheticResidual(StrictModule):
    stress_residual: Array
    heat_flux_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class KineticSyntheticCorrection(StrictModule):
    population: Array
    target_moments: Array
    moment_defect: Array
    positivity_damping: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class KineticSyntheticAccelerationPlan(StrictModule):
    collision: MonatomicBGKCollisionPlan
    positivity_floor: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        collision: MonatomicBGKCollisionPlan,
        /,
        *,
        positivity_floor: float = 1.0e-14,
    ) -> None:
        floor = float(positivity_floor)
        if not isinstance(collision, MonatomicBGKCollisionPlan):
            raise TypeError("collision must be MonatomicBGKCollisionPlan.")
        if not np.isfinite(floor) or floor <= 0.0:
            raise ValueError("positivity_floor must be finite and positive.")
        self.collision = collision
        self.positivity_floor = floor
        self.plan_id = canonical_fingerprint(
            {
                "kind": "kinetic-synthetic-acceleration",
                "collision": collision.plan_id,
                "positivity_floor": floor,
            }
        )

    def residual(
        self,
        population: ArrayLike,
        navier_stokes_stress: ArrayLike,
        navier_stokes_heat_flux: ArrayLike,
        /,
    ) -> KineticSyntheticResidual:
        value = jnp.asarray(population)
        moments = self.collision.quadrature.moments(value)
        density = moments[0]
        velocity = moments[1:4] / density
        peculiar = self.collision.quadrature.velocities.astype(value.dtype) - velocity
        speed_squared = jnp.sum(peculiar**2, axis=-1)
        weight = self.collision.quadrature.weights.astype(value.dtype)
        pressure_tensor = contract(
            "q,q,qi,qj->ij", weight, value, peculiar, peculiar, backend="jax"
        )
        pressure = jnp.trace(pressure_tensor) / 3.0
        kinetic_stress = pressure_tensor - pressure * jnp.eye(3, dtype=value.dtype)
        kinetic_heat_flux = contract(
            "q,q,q,qi->i",
            weight,
            value,
            0.5 * speed_squared,
            peculiar,
            backend="jax",
        )
        reference_stress = jnp.asarray(navier_stokes_stress, dtype=value.dtype)
        reference_heat = jnp.asarray(navier_stokes_heat_flux, dtype=value.dtype)
        if reference_stress.shape != (3, 3) or reference_heat.shape != (3,):
            raise ValueError("Synthetic closure references have incompatible shapes.")
        successful = (
            jnp.all(jnp.isfinite(value))
            & jnp.all(value >= 0.0)
            & jnp.all(jnp.isfinite(reference_stress))
            & jnp.all(jnp.isfinite(reference_heat))
        )
        return KineticSyntheticResidual(
            kinetic_stress - reference_stress,
            kinetic_heat_flux - reference_heat,
            successful,
            self.plan_id,
        )

    def correct(
        self,
        population: ArrayLike,
        target_moments: ArrayLike,
        /,
    ) -> KineticSyntheticCorrection:
        value = jnp.asarray(population)
        target = jnp.asarray(target_moments, dtype=value.dtype)
        if value.shape != (self.collision.quadrature.velocity_count,) or target.shape != (
            5,
        ):
            raise ValueError("Synthetic correction shapes are incompatible.")
        current_moments = self.collision.quadrature.moments(value)
        current_equilibrium = self.collision.maxwellian.solve(current_moments)
        target_equilibrium = self.collision.maxwellian.solve(target)
        micro = value - current_equilibrium.population
        raw = target_equilibrium.population + micro
        delta = raw - target_equilibrium.population
        limiting = jnp.where(
            delta < 0.0,
            (target_equilibrium.population - self.positivity_floor) / (-delta),
            jnp.inf,
        )
        damping = jnp.minimum(1.0, jnp.min(limiting))
        damping = jnp.maximum(damping, 0.0)
        corrected = target_equilibrium.population + damping * micro
        defect = self.collision.quadrature.moments(corrected) - target
        successful = (
            current_equilibrium.successful
            & target_equilibrium.successful
            & jnp.all(corrected >= self.positivity_floor)
            & (jnp.max(jnp.abs(defect)) <= 100.0 * self.collision.maxwellian.tolerance)
        )
        return KineticSyntheticCorrection(
            corrected,
            target,
            defect,
            damping,
            successful,
            self.plan_id,
        )


__all__ = [
    "KineticSyntheticAccelerationPlan",
    "KineticSyntheticCorrection",
    "KineticSyntheticResidual",
]
