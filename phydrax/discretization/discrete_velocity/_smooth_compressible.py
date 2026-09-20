#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax.typing import DTypeLike
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...equations._materials import IdealGasMaterial
from ...equations._transport_closures import AbstractTransportClosure, TransportProperties
from ..lattice_boltzmann._program import (
    KineticProgramManifest,
    smooth_compressible_dvm_manifest,
)
from ._energy_equilibrium import EnergyEquilibriumEvidence, PositiveEnergyEquilibriumPlan
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
    """Exact equilibrium conserved-moment and flux residuals."""

    target_conserved: Array
    recovered_conserved: Array
    conserved_residual: Array
    target_total_energy_flux: Array
    recovered_total_energy_flux: Array
    total_energy_flux_residual: Array
    target_particle_momentum_flux: Array
    recovered_particle_momentum_flux: Array
    particle_momentum_flux_residual: Array
    maximum_absolute_particle_momentum_flux_residual: Array
    minimum_particle_equilibrium_population: Array
    realizability: SmoothCompressibleRealizabilityEvidence


class SmoothCompressibleCollisionEvidence(StrictModule):
    """Before/after conservation evidence for a coupled collision."""

    particle_relaxation_rate: Array
    total_energy_relaxation_rate: Array
    pre_collision_conserved: Array
    post_collision_conserved: Array
    conservation_residual: Array
    maximum_absolute_residual: Array
    valid: Array
    post_collision_realizability: SmoothCompressibleRealizabilityEvidence


class SmoothCompressibleLearnedEquilibriumEvidence(StrictModule):
    """Particle/energy equilibrium evidence for an explicit learned closure."""

    energy: EnergyEquilibriumEvidence
    target_conserved: Array
    recovered_conserved: Array
    conservation_residual: Array
    maximum_absolute_residual: Array
    target_particle_momentum_flux: Array
    recovered_particle_momentum_flux: Array
    particle_momentum_flux_residual: Array
    maximum_absolute_particle_momentum_flux_residual: Array
    minimum_particle_equilibrium_population: Array
    realizability: SmoothCompressibleRealizabilityEvidence
    successful: Array


class SmoothCompressibleLearnedCollisionResult(StrictModule):
    """Transactional local collision driven by a learned total-energy equilibrium."""

    candidate_state: SmoothCompressibleKineticState
    accepted_state: SmoothCompressibleKineticState
    equilibrium_evidence: SmoothCompressibleLearnedEquilibriumEvidence
    collision_evidence: SmoothCompressibleCollisionEvidence
    successful: Array
    rollback_applied: Array


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
    particle_equilibrium_moment_matrix: Array
    particle_equilibrium_moment_lift: Array
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
        particle_moment_matrix = np.concatenate(
            (np.ones((1, quadrature.population_count)), velocities.T), axis=0
        )
        particle_moment_gram = particle_moment_matrix @ particle_moment_matrix.T
        particle_moment_lift = np.linalg.solve(
            particle_moment_gram, particle_moment_matrix
        ).T
        projector = (
            np.eye(quadrature.population_count)
            - particle_moment_lift @ particle_moment_matrix
        )
        cx = velocities[:, 0]
        cy = velocities[:, 1]
        particle_equilibrium_moment_matrix = np.stack(
            (
                np.ones((quadrature.population_count,)),
                cx,
                cy,
                cx**2,
                cx * cy,
                cy**2,
            ),
            axis=0,
        )
        particle_equilibrium_gram = (
            particle_equilibrium_moment_matrix @ particle_equilibrium_moment_matrix.T
        )
        particle_equilibrium_moment_lift = np.linalg.solve(
            particle_equilibrium_gram, particle_equilibrium_moment_matrix
        ).T
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
            particle_moment_matrix, dtype=quadrature.velocities.dtype
        )
        self.particle_moment_lift = jnp.asarray(
            particle_moment_lift, dtype=quadrature.velocities.dtype
        )
        self.particle_equilibrium_moment_matrix = jnp.asarray(
            particle_equilibrium_moment_matrix, dtype=quadrature.velocities.dtype
        )
        self.particle_equilibrium_moment_lift = jnp.asarray(
            particle_equilibrium_moment_lift, dtype=quadrature.velocities.dtype
        )
        self.particle_nullspace_projector = jnp.asarray(
            projector, dtype=quadrature.velocities.dtype
        )
        self.energy_moment_lift = quadrature.weights / jnp.sum(quadrature.weights)
        self.method_id = canonical_fingerprint(
            {
                "kind": "smooth-compressible-d2v-kinetic-method",
                "quadrature": quadrature.quadrature_id,
                "material": material.material_id,
                "transport": transport.closure_id,
                "program_manifest": program_manifest.manifest_id,
                "energy_layout": "separate-total-energy-populations",
                "particle_equilibrium": (
                    "variable-temperature-second-order-raw-hermite-six-moment-lift"
                ),
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
        momentum = ein.contract("...q,qd->...d", particles, self.quadrature.velocities)
        total_energy = jnp.sum(energy_populations, axis=-1)
        safe_density = jnp.where(density > 0.0, density, 1.0)
        velocity = momentum / safe_density[..., None]
        kinetic_energy = 0.5 * ein.contract("...d,...d->...", momentum, velocity)
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

    def _safe_equilibrium_fields(
        self, conserved: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array, Array, Array]:
        values = jnp.asarray(conserved)
        if values.ndim == 0 or values.shape[-1] != self.quadrature.dimension + 2:
            raise ValueError(
                "Compressible conserved state must have trailing shape (D + 2,)."
            )
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            raise TypeError("Compressible conserved state must use an inexact dtype.")
        density = values[..., 0]
        momentum = values[..., 1:-1]
        total_energy = values[..., -1]
        safe_density = jnp.where(jnp.isfinite(density) & (density > 0.0), density, 1.0)
        safe_momentum = jnp.where(jnp.isfinite(momentum), momentum, 0.0)
        safe_energy = jnp.where(jnp.isfinite(total_energy), total_energy, 1.0)
        velocity = safe_momentum / safe_density[..., None]
        kinetic_energy = 0.5 * ein.contract("...d,...d->...", safe_momentum, velocity)
        internal_energy = (safe_energy - kinetic_energy) / safe_density
        pressure = self.material.pressure(safe_density, internal_energy)
        valid = jnp.all(jnp.isfinite(values), axis=-1) & self.material.admissible(
            density, pressure
        )
        reference_energy = (
            self.material.gas_constant
            * self.quadrature.reference_temperature
            / (self.material.gamma - 1.0)
        )
        reference = jnp.asarray((1.0, 0.0, 0.0, reference_energy), dtype=values.dtype)
        safe_values = jnp.where(valid[..., None], values, reference)
        density = safe_values[..., 0]
        momentum = safe_values[..., 1:-1]
        total_energy = safe_values[..., -1]
        velocity = momentum / density[..., None]
        kinetic_energy = 0.5 * ein.contract("...d,...d->...", momentum, velocity)
        internal_energy = (total_energy - kinetic_energy) / density
        pressure = self.material.pressure(density, internal_energy)
        return (
            safe_values,
            density,
            momentum,
            total_energy,
            velocity,
            pressure,
            valid,
        )

    def _equilibrium_fields(
        self, conserved: ArrayLike, /
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        (
            values,
            density,
            momentum,
            total_energy,
            velocity,
            pressure,
            valid,
        ) = self._safe_equilibrium_fields(conserved)
        values = eqx.error_if(
            values, jnp.any(~valid), "Compressible equilibrium state is inadmissible."
        )
        return values, density, momentum, total_energy, velocity, pressure

    @staticmethod
    def _target_particle_momentum_flux(
        density: Array, momentum: Array, pressure: Array, /
    ) -> Array:
        return ein.contract("...d,...e->...de", momentum, momentum) / density[
            ..., None, None
        ] + pressure[..., None, None] * jnp.eye(momentum.shape[-1], dtype=momentum.dtype)

    def _particle_momentum_flux(self, populations: Array, /) -> Array:
        return ein.contract(
            "...q,qd,qe->...de",
            populations,
            self.quadrature.velocities,
            self.quadrature.velocities,
        )

    def _particle_equilibrium(
        self,
        density: Array,
        momentum: Array,
        velocity: Array,
        pressure: Array,
        /,
    ) -> Array:
        reference_temperature = self.quadrature.reference_temperature
        theta = pressure / density
        velocity_square = ein.contract("...d,...d->...", velocity, velocity)
        particle_velocity_square = ein.contract(
            "qd,qd->q", self.quadrature.velocities, self.quadrature.velocities
        )
        projected_velocity = ein.contract(
            "...d,qd->...q", velocity, self.quadrature.velocities
        )
        second_order = (
            projected_velocity**2
            - reference_temperature * velocity_square[..., None]
            + (theta - reference_temperature)[..., None]
            * (
                particle_velocity_square
                - self.quadrature.dimension * reference_temperature
            )
        ) / (2.0 * reference_temperature**2)
        particle_raw = (
            self.quadrature.weights
            * density[..., None]
            * (1.0 + projected_velocity / reference_temperature + second_order)
        )
        target_momentum_flux = self._target_particle_momentum_flux(
            density, momentum, pressure
        )
        target_moments = jnp.concatenate(
            (
                density[..., None],
                momentum,
                target_momentum_flux[..., 0, 0, None],
                target_momentum_flux[..., 0, 1, None],
                target_momentum_flux[..., 1, 1, None],
            ),
            axis=-1,
        )
        recovered_moments = ein.contract(
            "mq,...q->...m", self.particle_equilibrium_moment_matrix, particle_raw
        )
        return particle_raw + ein.contract(
            "qm,...m->...q",
            self.particle_equilibrium_moment_lift,
            target_moments - recovered_moments,
        )

    def _analytic_energy_equilibrium(
        self,
        total_energy: Array,
        pressure: Array,
        velocity: Array,
        /,
    ) -> Array:
        projected_velocity = ein.contract(
            "...d,qd->...q", velocity, self.quadrature.velocities
        )
        enthalpy_density = total_energy + pressure
        energy_raw = self.quadrature.weights * (
            total_energy[..., None]
            + enthalpy_density[..., None]
            * projected_velocity
            / self.quadrature.reference_temperature
        )
        return (
            energy_raw
            + self.energy_moment_lift
            * (total_energy - jnp.sum(energy_raw, axis=-1))[..., None]
        )

    @staticmethod
    def _target_energy_flux(
        total_energy: Array, pressure: Array, velocity: Array, /
    ) -> Array:
        return (total_energy + pressure)[..., None] * velocity

    def equilibrium(self, conserved: ArrayLike, /) -> SmoothCompressibleKineticState:
        _, density, momentum, total_energy, velocity, pressure = self._equilibrium_fields(
            conserved
        )
        return SmoothCompressibleKineticState(
            self._particle_equilibrium(density, momentum, velocity, pressure),
            self._analytic_energy_equilibrium(total_energy, pressure, velocity),
        )

    def equilibrium_from_energy_dual_with_evidence(
        self,
        conserved: ArrayLike,
        dual: ArrayLike,
        plan: PositiveEnergyEquilibriumPlan,
        /,
    ) -> tuple[
        SmoothCompressibleKineticState,
        SmoothCompressibleLearnedEquilibriumEvidence,
    ]:
        if not isinstance(plan, PositiveEnergyEquilibriumPlan):
            raise TypeError("plan must be a PositiveEnergyEquilibriumPlan.")
        if plan.quadrature.quadrature_id != self.quadrature.quadrature_id:
            raise ValueError(
                "Energy-equilibrium plan and kinetic quadrature do not match."
            )
        (
            values,
            density,
            momentum,
            total_energy,
            velocity,
            pressure,
            input_valid,
        ) = self._safe_equilibrium_fields(conserved)
        target_flux = self._target_energy_flux(total_energy, pressure, velocity)
        target_particle_momentum_flux = self._target_particle_momentum_flux(
            density, momentum, pressure
        )
        energy = plan.evaluate(total_energy, target_flux, dual)
        particle_equilibrium = self._particle_equilibrium(
            density, momentum, velocity, pressure
        )
        equilibrium = SmoothCompressibleKineticState(
            particle_equilibrium,
            energy.populations,
        )
        recovered = self.moments(equilibrium).conserved
        residual = recovered - values
        recovered_particle_momentum_flux = self._particle_momentum_flux(
            particle_equilibrium
        )
        particle_momentum_flux_residual = (
            recovered_particle_momentum_flux - target_particle_momentum_flux
        )
        maximum_particle_momentum_flux_residual = jnp.max(
            jnp.abs(particle_momentum_flux_residual)
        )
        realizability = self.realizability(equilibrium)
        scale = jnp.maximum(jnp.max(jnp.abs(values)), 1.0)
        tolerance = 256.0 * jnp.finfo(values.dtype).eps * scale + jnp.asarray(
            plan.residual_tolerance, dtype=values.dtype
        )
        particle_flux_scale = jnp.maximum(
            jnp.max(jnp.abs(target_particle_momentum_flux)), 1.0
        )
        particle_flux_tolerance = (
            256.0 * jnp.finfo(values.dtype).eps * particle_flux_scale
        )
        successful = (
            jnp.all(input_valid)
            & jnp.all(energy.evidence.successful)
            & realizability.realizable
            & (jnp.max(jnp.abs(residual)) <= tolerance)
            & (maximum_particle_momentum_flux_residual <= particle_flux_tolerance)
        )
        return equilibrium, SmoothCompressibleLearnedEquilibriumEvidence(
            energy=energy.evidence,
            target_conserved=values,
            recovered_conserved=recovered,
            conservation_residual=residual,
            maximum_absolute_residual=jnp.max(jnp.abs(residual)),
            target_particle_momentum_flux=target_particle_momentum_flux,
            recovered_particle_momentum_flux=recovered_particle_momentum_flux,
            particle_momentum_flux_residual=particle_momentum_flux_residual,
            maximum_absolute_particle_momentum_flux_residual=(
                maximum_particle_momentum_flux_residual
            ),
            minimum_particle_equilibrium_population=jnp.min(
                particle_equilibrium, axis=-1
            ),
            realizability=realizability,
            successful=successful,
        )

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
        _, density, momentum, total_energy, velocity, pressure = self._equilibrium_fields(
            values
        )
        target_flux = self._target_energy_flux(total_energy, pressure, velocity)
        target_particle_momentum_flux = self._target_particle_momentum_flux(
            density, momentum, pressure
        )
        energy_velocity_flux = ein.contract(
            "...q,qd->...d",
            equilibrium.total_energy_populations,
            self.quadrature.velocities,
        )
        recovered_particle_momentum_flux = self._particle_momentum_flux(
            equilibrium.particle_populations
        )
        particle_momentum_flux_residual = (
            recovered_particle_momentum_flux - target_particle_momentum_flux
        )
        return equilibrium, SmoothCompressibleEquilibriumEvidence(
            target_conserved=values,
            recovered_conserved=recovered,
            conserved_residual=recovered - values,
            target_total_energy_flux=target_flux,
            recovered_total_energy_flux=energy_velocity_flux,
            total_energy_flux_residual=energy_velocity_flux - target_flux,
            target_particle_momentum_flux=target_particle_momentum_flux,
            recovered_particle_momentum_flux=recovered_particle_momentum_flux,
            particle_momentum_flux_residual=particle_momentum_flux_residual,
            maximum_absolute_particle_momentum_flux_residual=jnp.max(
                jnp.abs(particle_momentum_flux_residual)
            ),
            minimum_particle_equilibrium_population=jnp.min(
                equilibrium.particle_populations, axis=-1
            ),
            realizability=self.realizability(equilibrium),
        )

    def transport_properties(
        self, state: SmoothCompressibleKineticState, args: Any = None, /
    ) -> TransportProperties:
        moments = self.moments(state)
        return self.transport.properties(moments.temperature, moments.conserved, args)

    def _collide_toward_equilibrium_with_evidence(
        self,
        state: SmoothCompressibleKineticState,
        equilibrium: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        args: Any = None,
        /,
        *,
        strict: bool = True,
    ) -> tuple[SmoothCompressibleKineticState, SmoothCompressibleCollisionEvidence]:
        self.validate_state(state)
        self.validate_state(equilibrium)
        step = jnp.asarray(time_step, dtype=state.particle_populations.dtype)
        step_valid = jnp.all(jnp.isfinite(step) & (step > 0.0))
        moments = self.moments(state)
        moments_valid = jnp.all(
            self.material.admissible(moments.density, moments.pressure)
        )
        properties = self.transport.properties(
            moments.temperature, moments.conserved, args
        )
        bulk_valid = jnp.all(properties.bulk_viscosity == 0.0)
        if strict:
            step = eqx.error_if(
                step,
                ~step_valid,
                "Smooth-compressible collision requires a finite positive time step.",
            )
            step = eqx.error_if(
                step,
                ~moments_valid,
                "Collision state is macroscopically inadmissible.",
            )
            step = eqx.error_if(
                step,
                ~bulk_valid,
                "Smooth-compressible single-rate collision does not support nonzero bulk viscosity.",
            )
        else:
            step = jnp.where(step_valid, step, 1.0)
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
        if strict:
            particle_rate = eqx.error_if(
                particle_rate,
                jnp.any(~rates_valid),
                "Transport closure produced invalid kinetic relaxation rates.",
            )
        else:
            particle_rate = jnp.where(rates_valid, particle_rate, 0.0)
            energy_rate = jnp.where(rates_valid, energy_rate, 0.0)
        valid = step_valid & moments_valid & bulk_valid & jnp.all(rates_valid)
        raw_particle_increment = particle_rate[..., None] * (
            equilibrium.particle_populations - state.particle_populations
        )
        particle_increment = ein.contract(
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
            valid=valid,
            post_collision_realizability=self.realizability(collided),
        )
        return collided, evidence

    def collide_with_evidence(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        args: Any = None,
        /,
    ) -> tuple[SmoothCompressibleKineticState, SmoothCompressibleCollisionEvidence]:
        self.validate_state(state)
        equilibrium = self.equilibrium(self.moments(state).conserved)
        return self._collide_toward_equilibrium_with_evidence(
            state, equilibrium, time_step, args
        )

    def collide_with_energy_dual_with_evidence(
        self,
        state: SmoothCompressibleKineticState,
        time_step: ArrayLike,
        dual: ArrayLike,
        plan: PositiveEnergyEquilibriumPlan,
        args: Any = None,
        /,
    ) -> SmoothCompressibleLearnedCollisionResult:
        self.validate_state(state)
        equilibrium, equilibrium_evidence = (
            self.equilibrium_from_energy_dual_with_evidence(
                self.moments(state).conserved, dual, plan
            )
        )
        candidate, collision_evidence = self._collide_toward_equilibrium_with_evidence(
            state, equilibrium, time_step, args, strict=False
        )
        scale = jnp.maximum(
            jnp.max(jnp.abs(collision_evidence.pre_collision_conserved)), 1.0
        )
        tolerance = 256.0 * jnp.finfo(state.particle_populations.dtype).eps * scale
        successful = (
            equilibrium_evidence.successful
            & collision_evidence.valid
            & collision_evidence.post_collision_realizability.realizable
            & (collision_evidence.maximum_absolute_residual <= tolerance)
        )
        accepted = SmoothCompressibleKineticState(
            jnp.where(
                successful,
                candidate.particle_populations,
                state.particle_populations,
            ),
            jnp.where(
                successful,
                candidate.total_energy_populations,
                state.total_energy_populations,
            ),
        )
        return SmoothCompressibleLearnedCollisionResult(
            candidate_state=candidate,
            accepted_state=accepted,
            equilibrium_evidence=equilibrium_evidence,
            collision_evidence=collision_evidence,
            successful=successful,
            rollback_applied=~successful,
        )

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
    "SmoothCompressibleLearnedCollisionResult",
    "SmoothCompressibleLearnedEquilibriumEvidence",
    "SmoothCompressibleMoments",
    "SmoothCompressibleRealizabilityEvidence",
    "smooth_compressible_d2v17_method",
    "smooth_compressible_d2v37_off_lattice_method",
]
