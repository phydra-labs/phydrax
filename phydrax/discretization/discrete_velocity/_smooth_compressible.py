#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._materials import IdealGasMaterial
from ...equations._transport_closures import AbstractTransportClosure, TransportProperties
from ..lattice_boltzmann._program import (
    KineticProgramManifest,
    smooth_compressible_dvm_manifest,
)
from ._quadrature import (
    CertifiedDiscreteVelocityQuadrature,
    d2v17_quadrature,
    d2v37_off_lattice_quadrature,
)


class SmoothCompressibleKineticState(StrictModule):
    """Coupled mass/momentum and total-energy population fields."""

    particle_populations: Array
    total_energy_populations: Array

    def __init__(
        self,
        particle_populations: ArrayLike,
        total_energy_populations: ArrayLike,
        /,
    ):
        particles = jnp.asarray(particle_populations)
        energy = jnp.asarray(total_energy_populations)
        if particles.ndim == 0 or particles.shape != energy.shape:
            raise ValueError(
                "Particle and total-energy populations must have equal trailing-Q shapes."
            )
        if not jnp.issubdtype(particles.dtype, jnp.inexact) or not jnp.issubdtype(
            energy.dtype, jnp.inexact
        ):
            raise TypeError("Smooth-compressible populations must have inexact dtypes.")
        if particles.dtype != energy.dtype:
            raise TypeError("Coupled kinetic population fields must use one dtype.")
        self.particle_populations = particles
        self.total_energy_populations = energy


class SmoothCompressibleMoments(StrictModule):
    """Mass, momentum, and total energy recovered from the kinetic state."""

    density: Array
    momentum: Array
    total_energy: Array
    velocity: Array
    pressure: Array
    temperature: Array
    specific_internal_energy: Array
    conserved: Array


class SmoothCompressibleRealizabilityEvidence(StrictModule):
    """Macroscopic and population-level admissibility without hidden clipping."""

    density: Array
    pressure: Array
    temperature: Array
    minimum_particle_population: Array
    minimum_total_energy_population: Array
    finite: Array
    macroscopic_admissible: Array
    populations_nonnegative: Array
    local_realizable: Array
    realizable: Array


class SmoothCompressibleEquilibriumEvidence(StrictModule):
    """Exact equilibrium moment and energy-flux residuals."""

    target_conserved: Array
    recovered_conserved: Array
    conserved_residual: Array
    target_total_energy_flux: Array
    recovered_total_energy_flux: Array
    total_energy_flux_residual: Array
    realizability: SmoothCompressibleRealizabilityEvidence


class SmoothCompressibleCollisionEvidence(StrictModule):
    """Before/after conservation evidence for a coupled collision."""

    particle_relaxation_rate: Array
    total_energy_relaxation_rate: Array
    pre_collision_conserved: Array
    post_collision_conserved: Array
    conservation_residual: Array
    maximum_absolute_residual: Array
    post_collision_realizability: SmoothCompressibleRealizabilityEvidence


class SmoothCompressibleD2VKineticMethod(StrictModule, NonTrainableState):
    """Two-population D2V17/D2V37 compressible-energy research method.

    This class is deliberately not named as an on-lattice streaming method.
    D2V37 remains explicitly off-lattice and requires finite-volume or prepared
    semi-Lagrangian transport. Total energy is a conserved kinetic population,
    and pressure/temperature are derived from it rather than passively advected.
    """

    quadrature: CertifiedDiscreteVelocityQuadrature
    material: IdealGasMaterial
    transport: AbstractTransportClosure
    particle_moment_matrix: Array
    particle_moment_lift: Array
    particle_nullspace_projector: Array
    energy_moment_lift: Array
    program_manifest: KineticProgramManifest
    method_id: str = eqx.field(static=True)

    def __init__(
        self,
        quadrature: CertifiedDiscreteVelocityQuadrature,
        material: IdealGasMaterial,
        transport: AbstractTransportClosure,
        /,
    ):
        if not isinstance(quadrature, CertifiedDiscreteVelocityQuadrature):
            raise TypeError("quadrature must be a CertifiedDiscreteVelocityQuadrature.")
        if quadrature.dimension != 2 or quadrature.population_count not in (17, 37):
            raise ValueError(
                "Smooth compressible research methods require D2V17 or D2V37."
            )
        if quadrature.certification.maximum_degree < 4:
            raise ValueError(
                "Smooth compressible D2V methods require fourth-degree certification."
            )
        if not isinstance(material, IdealGasMaterial):
            raise TypeError(
                "material must implement the certified IdealGasMaterial interface."
            )
        if not isinstance(transport, AbstractTransportClosure):
            raise TypeError("transport must implement AbstractTransportClosure.")
        velocities = np.asarray(quadrature.velocities)
        matrix = np.concatenate(
            (np.ones((1, quadrature.population_count)), velocities.T), axis=0
        )
        gram = matrix @ matrix.T
        lift = matrix.T @ np.linalg.solve(gram, np.eye(matrix.shape[0]))
        projector = np.eye(quadrature.population_count) - lift @ matrix
        program_manifest = smooth_compressible_dvm_manifest(
            quadrature.quadrature_id,
            f"dtype:{quadrature.velocities.dtype}",
            quadrature.population_count,
            quadrature.dimension,
        )
        self.quadrature = quadrature
        self.material = material
        self.transport = transport
        self.program_manifest = program_manifest
        self.particle_moment_matrix = jnp.asarray(
            matrix, dtype=quadrature.velocities.dtype
        )
        self.particle_moment_lift = jnp.asarray(lift, dtype=quadrature.velocities.dtype)
        self.particle_nullspace_projector = jnp.asarray(
            projector, dtype=quadrature.velocities.dtype
        )
        self.energy_moment_lift = quadrature.weights / jnp.sum(quadrature.weights)
        self.method_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v-kinetic-method-v1",
                "quadrature": quadrature.quadrature_id,
                "material": material.material_id,
                "transport": transport.closure_id,
                "program_manifest": program_manifest.manifest_id,
                "energy_layout": "separate-total-energy-populations",
            }
        )

    def validate_state(self, state: SmoothCompressibleKineticState, /) -> None:
        if not isinstance(state, SmoothCompressibleKineticState):
            raise TypeError("state must be SmoothCompressibleKineticState.")
        self.quadrature.validate_populations(state.particle_populations)
        self.quadrature.validate_populations(state.total_energy_populations)
        if state.particle_populations.shape != state.total_energy_populations.shape:
            raise ValueError(
                "Smooth-compressible population fields must have equal shapes."
            )

    def _raw_moments(
        self, state: SmoothCompressibleKineticState, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        self.validate_state(state)
        particles = state.particle_populations
        energy_populations = state.total_energy_populations
        density = jnp.sum(particles, axis=-1)
        momentum = oe.contract("...q,qd->...d", particles, self.quadrature.velocities)
        total_energy = jnp.sum(energy_populations, axis=-1)
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        kinetic_energy = 0.5 * oe.contract("...d,...d->...", momentum, velocity)
        specific_internal_energy = (total_energy - kinetic_energy) / safe_density
        pressure = self.material.pressure(density, specific_internal_energy)
        return (
            density,
            momentum,
            total_energy,
            velocity,
            specific_internal_energy,
            pressure,
        )

    def moments(
        self, state: SmoothCompressibleKineticState, /
    ) -> SmoothCompressibleMoments:
        density, momentum, total_energy, velocity, internal_energy, pressure = (
            self._raw_moments(state)
        )
        temperature = self.material.temperature(density, pressure)
        conserved = jnp.concatenate(
            (density[..., None], momentum, total_energy[..., None]), axis=-1
        )
        return SmoothCompressibleMoments(
            density=density,
            momentum=momentum,
            total_energy=total_energy,
            velocity=velocity,
            pressure=pressure,
            temperature=temperature,
            specific_internal_energy=internal_energy,
            conserved=conserved,
        )

    def realizability(
        self,
        state: SmoothCompressibleKineticState,
        /,
        *,
        population_floor: float = 0.0,
    ) -> SmoothCompressibleRealizabilityEvidence:
        floor = float(population_floor)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("population_floor must be finite and non-negative.")
        density, _, _, _, _, pressure = self._raw_moments(state)
        safe_density = jnp.where(density > 0.0, density, 1.0)
        temperature = self.material.temperature(safe_density, pressure)
        minimum_particles = jnp.min(state.particle_populations, axis=-1)
        minimum_energy = jnp.min(state.total_energy_populations, axis=-1)
        finite = (
            jnp.isfinite(density)
            & jnp.isfinite(pressure)
            & jnp.isfinite(temperature)
            & jnp.all(jnp.isfinite(state.particle_populations), axis=-1)
            & jnp.all(jnp.isfinite(state.total_energy_populations), axis=-1)
        )
        macro = self.material.admissible(density, pressure) & (temperature > 0.0)
        populations = (minimum_particles >= floor) & (minimum_energy >= floor)
        local = finite & macro & populations
        return SmoothCompressibleRealizabilityEvidence(
            density=density,
            pressure=pressure,
            temperature=temperature,
            minimum_particle_population=minimum_particles,
            minimum_total_energy_population=minimum_energy,
            finite=finite,
            macroscopic_admissible=macro,
            populations_nonnegative=populations,
            local_realizable=local,
            realizable=jnp.all(local),
        )

    def equilibrium(self, conserved: ArrayLike, /) -> SmoothCompressibleKineticState:
        values = jnp.asarray(conserved)
        if values.ndim == 0 or values.shape[-1] != self.quadrature.dimension + 2:
            raise ValueError(
                "Compressible conserved state must have trailing shape (D + 2,)."
            )
        density = values[..., 0]
        momentum = values[..., 1:-1]
        total_energy = values[..., -1]
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        kinetic_energy = 0.5 * oe.contract("...d,...d->...", momentum, velocity)
        internal_energy = (total_energy - kinetic_energy) / safe_density
        pressure = self.material.pressure(density, internal_energy)
        valid = self.material.admissible(density, pressure) & jnp.all(
            jnp.isfinite(values), axis=-1
        )
        values = eqx.error_if(
            values, jnp.any(~valid), "Compressible equilibrium state is inadmissible."
        )
        density = values[..., 0]
        momentum = values[..., 1:-1]
        total_energy = values[..., -1]
        velocity = momentum / density[..., None]
        kinetic_energy = 0.5 * oe.contract("...d,...d->...", momentum, velocity)
        internal_energy = (total_energy - kinetic_energy) / density
        pressure = self.material.pressure(density, internal_energy)
        temperature = self.quadrature.reference_temperature
        velocity_square = oe.contract("...d,...d->...", velocity, velocity)
        projected_velocity = oe.contract(
            "...d,qd->...q", velocity, self.quadrature.velocities
        )
        particle_raw = (
            self.quadrature.weights
            * density[..., None]
            * (
                1.0
                + projected_velocity / temperature
                + 0.5
                * (
                    projected_velocity**2 / temperature**2
                    - velocity_square[..., None] / temperature
                )
            )
        )
        target_particle_moments = jnp.concatenate((density[..., None], momentum), axis=-1)
        recovered_particle_moments = oe.contract(
            "mq,...q->...m", self.particle_moment_matrix, particle_raw
        )
        particle_equilibrium = particle_raw + oe.contract(
            "qm,...m->...q",
            self.particle_moment_lift,
            target_particle_moments - recovered_particle_moments,
        )
        enthalpy_density = total_energy + pressure
        energy_raw = self.quadrature.weights * (
            total_energy[..., None]
            + enthalpy_density[..., None] * projected_velocity / temperature
        )
        energy_equilibrium = (
            energy_raw
            + self.energy_moment_lift
            * (total_energy - jnp.sum(energy_raw, axis=-1))[..., None]
        )
        return SmoothCompressibleKineticState(particle_equilibrium, energy_equilibrium)

    def equilibrium_from_state(
        self, state: SmoothCompressibleKineticState, /
    ) -> SmoothCompressibleKineticState:
        return self.equilibrium(self.moments(state).conserved)

    def equilibrium_with_evidence(
        self, conserved: ArrayLike, /
    ) -> tuple[SmoothCompressibleKineticState, SmoothCompressibleEquilibriumEvidence]:
        values = jnp.asarray(conserved)
        equilibrium = self.equilibrium(values)
        recovered = self.moments(equilibrium).conserved
        density = values[..., 0]
        momentum = values[..., 1:-1]
        total_energy = values[..., -1]
        velocity = momentum / density[..., None]
        kinetic_energy = 0.5 * oe.contract("...d,...d->...", momentum, velocity)
        pressure = self.material.pressure(
            density, (total_energy - kinetic_energy) / density
        )
        target_flux = (total_energy + pressure)[..., None] * velocity
        energy_velocity_flux = oe.contract(
            "...q,qd->...d",
            equilibrium.total_energy_populations,
            self.quadrature.velocities,
        )
        return equilibrium, SmoothCompressibleEquilibriumEvidence(
            target_conserved=values,
            recovered_conserved=recovered,
            conserved_residual=recovered - values,
            target_total_energy_flux=target_flux,
            recovered_total_energy_flux=energy_velocity_flux,
            total_energy_flux_residual=energy_velocity_flux - target_flux,
            realizability=self.realizability(equilibrium),
        )

    def transport_properties(
        self, state: SmoothCompressibleKineticState, args: Any = None, /
    ) -> TransportProperties:
        moments = self.moments(state)
        return self.transport.properties(moments.temperature, moments.conserved, args)

    def collide_with_evidence(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[SmoothCompressibleKineticState, SmoothCompressibleCollisionEvidence]:
        self.validate_state(state)
        step = jnp.asarray(time_step, dtype=state.particle_populations.dtype)
        step = eqx.error_if(
            step,
            jnp.any(~jnp.isfinite(step) | (step <= 0.0)),
            "Smooth-compressible collision requires a finite positive time step.",
        )
        moments = self.moments(state)
        valid = self.material.admissible(moments.density, moments.pressure)
        step = eqx.error_if(
            step, jnp.any(~valid), "Collision state is macroscopically inadmissible."
        )
        properties = self.transport.properties(
            moments.temperature, moments.conserved, args
        )
        step = eqx.error_if(
            step,
            jnp.any(properties.bulk_viscosity != 0.0),
            "Smooth-compressible single-rate collision does not support nonzero bulk viscosity.",
        )
        cp = self.material.specific_heat_cp(moments.density, moments.pressure)
        particle_time = properties.dynamic_viscosity / moments.pressure
        energy_time = properties.thermal_conductivity / (cp * moments.pressure)
        particle_rate = step / (step + particle_time)
        energy_rate = step / (step + energy_time)
        rates_valid = (
            jnp.isfinite(particle_rate)
            & jnp.isfinite(energy_rate)
            & (particle_rate >= 0.0)
            & (particle_rate <= 1.0)
            & (energy_rate >= 0.0)
            & (energy_rate <= 1.0)
        )
        particle_rate = eqx.error_if(
            particle_rate,
            jnp.any(~rates_valid),
            "Transport closure produced invalid kinetic relaxation rates.",
        )
        equilibrium = self.equilibrium(moments.conserved)
        raw_particle_increment = particle_rate[..., None] * (
            equilibrium.particle_populations - state.particle_populations
        )
        particle_increment = oe.contract(
            "pq,...q->...p", self.particle_nullspace_projector, raw_particle_increment
        )
        raw_energy_increment = energy_rate[..., None] * (
            equilibrium.total_energy_populations - state.total_energy_populations
        )
        energy_increment = (
            raw_energy_increment
            - self.energy_moment_lift * jnp.sum(raw_energy_increment, axis=-1)[..., None]
        )
        collided = SmoothCompressibleKineticState(
            state.particle_populations + particle_increment,
            state.total_energy_populations + energy_increment,
        )
        post = self.moments(collided).conserved
        residual = post - moments.conserved
        evidence = SmoothCompressibleCollisionEvidence(
            particle_relaxation_rate=particle_rate,
            total_energy_relaxation_rate=energy_rate,
            pre_collision_conserved=moments.conserved,
            post_collision_conserved=post,
            conservation_residual=residual,
            maximum_absolute_residual=jnp.max(jnp.abs(residual)),
            post_collision_realizability=self.realizability(collided),
        )
        return collided, evidence

    def collide(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        args: Any = None,
        /,
    ) -> SmoothCompressibleKineticState:
        return self.collide_with_evidence(state, time_step, args)[0]


def smooth_compressible_d2v17_method(
    material: IdealGasMaterial,
    transport: AbstractTransportClosure,
    /,
    *,
    dtype: DTypeLike = jnp.float64,
) -> SmoothCompressibleD2VKineticMethod:
    return SmoothCompressibleD2VKineticMethod(
        d2v17_quadrature(dtype=dtype), material, transport
    )


def smooth_compressible_d2v37_off_lattice_method(
    material: IdealGasMaterial,
    transport: AbstractTransportClosure,
    /,
    *,
    dtype: DTypeLike = jnp.float64,
) -> SmoothCompressibleD2VKineticMethod:
    return SmoothCompressibleD2VKineticMethod(
        d2v37_off_lattice_quadrature(dtype=dtype), material, transport
    )


__all__ = [
    "SmoothCompressibleCollisionEvidence",
    "SmoothCompressibleD2VKineticMethod",
    "SmoothCompressibleEquilibriumEvidence",
    "SmoothCompressibleKineticState",
    "SmoothCompressibleMoments",
    "SmoothCompressibleRealizabilityEvidence",
    "smooth_compressible_d2v17_method",
    "smooth_compressible_d2v37_off_lattice_method",
]
