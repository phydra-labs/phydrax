#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ..particle import ParticlePopulationPlan, ParticlePopulationState
from ._types import PICParticleState


class PICBoundaryKind(IntEnum):
    ABSORB = 0
    REFLECT = 1


class PICBoundarySurfaceState(StrictModule):
    collected_charge: Array
    collected_mass: Array
    collected_kinetic_energy: Array


class PICBoundaryResult(StrictModule):
    candidate_particles: PICParticleState
    accepted_particles: PICParticleState
    candidate_population: ParticlePopulationState
    accepted_population: ParticlePopulationState
    candidate_surface: PICBoundarySurfaceState
    accepted_surface: PICBoundarySurfaceState
    hit_mask: Array
    hit_axis: Array
    hit_side: Array
    hit_fraction: Array
    hit_position: Array
    boundary_charge_flux: Array
    boundary_mass_flux: Array
    boundary_energy_flux: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class PICOpenBoundaryPlan(StrictModule, NonTrainableState):
    lower: Array
    upper: Array
    kinds: tuple[PICBoundaryKind, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        lower: ArrayLike,
        upper: ArrayLike,
        /,
        *,
        kinds: tuple[PICBoundaryKind, ...],
    ):
        lo = np.asarray(lower, dtype=float)
        hi = np.asarray(upper, dtype=float)
        if lo.ndim != 1 or hi.shape != lo.shape or lo.size not in (1, 2, 3):
            raise ValueError("PIC boundary bounds are invalid.")
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
            raise ValueError("PIC boundary bounds must be finite and ordered.")
        values = tuple(PICBoundaryKind(value) for value in kinds)
        if len(values) != 2 * lo.size:
            raise ValueError("One boundary kind is required for each lower/upper face.")
        self.lower = jnp.asarray(lo)
        self.upper = jnp.asarray(hi)
        self.kinds = values
        self.plan_id = canonical_fingerprint(
            {
                "kind": "pic-open-boundary",
                "lower": lo.tolist(),
                "upper": hi.tolist(),
                "kinds": [int(value) for value in values],
            }
        )

    def initialize_surface(self, dtype=float) -> PICBoundarySurfaceState:
        shape = (len(self.kinds),)
        return PICBoundarySurfaceState(
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape, dtype=dtype),
        )

    def apply(
        self,
        population_plan: ParticlePopulationPlan,
        population: ParticlePopulationState,
        particles: PICParticleState,
        proposed_position: ArrayLike,
        macrocharge: ArrayLike,
        surface: PICBoundarySurfaceState,
        /,
    ) -> PICBoundaryResult:
        start = particles.position
        end = jnp.asarray(proposed_position, dtype=start.dtype)
        charge = jnp.asarray(macrocharge, dtype=start.dtype)
        dimension = start.shape[1]
        if (
            dimension != self.lower.size
            or end.shape != start.shape
            or charge.shape != population.active.shape
        ):
            raise ValueError("PIC boundary state shapes are incompatible.")
        delta = end - start
        capacity = start.shape[0]
        candidate_t = jnp.full((capacity, 2 * dimension), jnp.inf, dtype=start.dtype)
        for axis in range(dimension):
            safe_delta = jnp.where(jnp.abs(delta[:, axis]) > 0.0, delta[:, axis], 1.0)
            lower_t = (self.lower[axis] - start[:, axis]) / safe_delta
            upper_t = (self.upper[axis] - start[:, axis]) / safe_delta
            candidate_t = candidate_t.at[:, 2 * axis].set(
                jnp.where(
                    (delta[:, axis] < 0.0) & (lower_t >= 0.0) & (lower_t <= 1.0),
                    lower_t,
                    jnp.inf,
                )
            )
            candidate_t = candidate_t.at[:, 2 * axis + 1].set(
                jnp.where(
                    (delta[:, axis] > 0.0) & (upper_t >= 0.0) & (upper_t <= 1.0),
                    upper_t,
                    jnp.inf,
                )
            )
        face = jnp.argmin(candidate_t, axis=1).astype(jnp.int32)
        fraction = jnp.min(candidate_t, axis=1)
        hit = population.active & jnp.isfinite(fraction) & (fraction < 1.0)
        safe_fraction = jnp.where(hit, fraction, 1.0)
        hit_position = start + safe_fraction[:, None] * delta
        kind_values = jnp.asarray(
            tuple(int(value) for value in self.kinds), dtype=jnp.int32
        )
        kind = kind_values[face]
        absorb = hit & (kind == int(PICBoundaryKind.ABSORB))
        reflect = hit & (kind == int(PICBoundaryKind.REFLECT))
        velocity = particles.proper_velocity
        normal_axis = face // 2
        reflected = velocity
        for axis in range(dimension):
            selected = reflect & (normal_axis == axis)
            reflected = reflected.at[:, axis].set(
                jnp.where(selected, -reflected[:, axis], reflected[:, axis])
            )
        remaining = (1.0 - safe_fraction)[:, None] * delta
        reflected_position = hit_position + remaining * jnp.where(
            jnp.arange(dimension)[None, :] == normal_axis[:, None], -1.0, 1.0
        )
        final_position = jnp.where(
            absorb[:, None],
            hit_position,
            jnp.where(reflect[:, None], reflected_position, end),
        )
        deactivation = population_plan.deactivate(population, absorb)
        kinetic = 0.5 * population.mass * jnp.sum(velocity**2, axis=-1)
        charge_flux = (
            jnp.zeros((2 * dimension,), dtype=start.dtype)
            .at[face]
            .add(jnp.where(absorb, charge, 0.0))
        )
        mass_flux = (
            jnp.zeros_like(charge_flux)
            .at[face]
            .add(jnp.where(absorb, population.mass, 0.0))
        )
        energy_flux = (
            jnp.zeros_like(charge_flux).at[face].add(jnp.where(absorb, kinetic, 0.0))
        )
        candidate_surface = PICBoundarySurfaceState(
            surface.collected_charge + charge_flux,
            surface.collected_mass + mass_flux,
            surface.collected_kinetic_energy + energy_flux,
        )
        candidate_particles = PICParticleState(
            jnp.where(deactivation.candidate_state.active[:, None], final_position, 0.0),
            jnp.where(deactivation.candidate_state.active[:, None], reflected, 0.0),
        )
        finite = jnp.all(jnp.isfinite(candidate_particles.position)) & jnp.all(
            jnp.isfinite(candidate_particles.proper_velocity)
        )
        successful = deactivation.successful & finite
        accepted_particles = PICParticleState(
            jnp.where(successful, candidate_particles.position, particles.position),
            jnp.where(
                successful, candidate_particles.proper_velocity, particles.proper_velocity
            ),
        )
        accepted_surface = PICBoundarySurfaceState(
            jnp.where(
                successful, candidate_surface.collected_charge, surface.collected_charge
            ),
            jnp.where(
                successful, candidate_surface.collected_mass, surface.collected_mass
            ),
            jnp.where(
                successful,
                candidate_surface.collected_kinetic_energy,
                surface.collected_kinetic_energy,
            ),
        )
        return PICBoundaryResult(
            candidate_particles,
            accepted_particles,
            deactivation.candidate_state,
            deactivation.accepted_state,
            candidate_surface,
            accepted_surface,
            hit,
            normal_axis,
            face % 2,
            safe_fraction,
            hit_position,
            charge_flux,
            mass_flux,
            energy_flux,
            finite,
            successful,
            self.plan_id,
        )


__all__ = [
    "PICBoundaryKind",
    "PICBoundaryResult",
    "PICBoundarySurfaceState",
    "PICOpenBoundaryPlan",
]
