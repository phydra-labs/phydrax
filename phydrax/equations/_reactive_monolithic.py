#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_allfinite
from ..discretization.particle import ParticleConversionState
from ._particle_conversion import PreparedParticleConversionDynamics
from ._particle_thermochemistry import ParticleTransportBoundary
from ._reactive_cfd_dem import ParticleContinuumExchangePlan


class ReactiveFluidImplicitState(StrictModule):
    velocity: Array
    temperature: Array
    species_concentration: Array


class CellwiseReactiveFluidImplicitPlan(StrictModule, NonTrainableState):
    cell_mass: Array
    cell_heat_capacity: Array
    species_storage: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_mass: ArrayLike,
        cell_heat_capacity: ArrayLike,
        species_storage: ArrayLike,
        /,
    ):
        mass = np.asarray(cell_mass, dtype=float)
        heat = np.asarray(cell_heat_capacity, dtype=float)
        species = np.asarray(species_storage, dtype=float)
        if (
            mass.ndim != 1
            or mass.size == 0
            or heat.shape != mass.shape
            or species.ndim != 2
            or species.shape[0] != mass.size
            or species.shape[1] == 0
            or np.any(~np.isfinite(mass))
            or np.any(mass <= 0.0)
            or np.any(~np.isfinite(heat))
            or np.any(heat <= 0.0)
            or np.any(~np.isfinite(species))
            or np.any(species <= 0.0)
        ):
            raise ValueError("Reactive fluid storage coefficients are invalid.")
        self.cell_mass = jnp.asarray(mass)
        self.cell_heat_capacity = jnp.asarray(heat)
        self.species_storage = jnp.asarray(species)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cellwise-reactive-fluid-implicit-plan",
                "mass": array_tree_fingerprint(mass),
                "heat": array_tree_fingerprint(heat),
                "species": array_tree_fingerprint(species),
            }
        )

    @property
    def cell_count(self) -> int:
        return int(self.cell_mass.shape[0])

    @property
    def species_count(self) -> int:
        return int(self.species_storage.shape[1])

    def validate_state(self, state: ReactiveFluidImplicitState, /) -> Array:
        if not isinstance(state, ReactiveFluidImplicitState):
            raise TypeError("state must be ReactiveFluidImplicitState.")
        valid_shape = (
            state.velocity.ndim == 2
            and state.velocity.shape[0] == self.cell_count
            and state.temperature.shape == (self.cell_count,)
            and state.species_concentration.shape == (self.cell_count, self.species_count)
        )
        if not valid_shape:
            raise ValueError("Reactive fluid state shapes do not match the plan.")
        return (
            tree_allfinite(state)
            & jnp.all(state.temperature > 0.0)
            & jnp.all(state.species_concentration >= 0.0)
        )


class ReactiveMonolithicUnknown(StrictModule):
    fluid_velocity: Array
    fluid_temperature: Array
    fluid_species_concentration: Array
    particle_velocity: Array
    batch_internal_energy: tuple[Array, ...]
    batch_species_amount: tuple[Array, ...]


class ReactiveMonolithicRouteCertificate(StrictModule):
    particle_active: Array
    conversion_active: tuple[Array, ...]
    transfer_route_digest: Array
    minimum_species_margin: Array
    minimum_temperature_margin: Array
    unchanged: Array
    successful: Array
    certificate_id: str = eqx.field(static=True)


class ReactiveMonolithicStage(StrictModule):
    previous_fluid: ReactiveFluidImplicitState
    previous_conversion: ParticleConversionState
    previous_particle_velocity: Array
    particle_position: Array
    particle_mass: Array
    particle_active: Array
    conversion_boundaries: tuple[ParticleTransportBoundary, ...]
    contact_energy_rate: tuple[Array, ...]
    radiative_energy_rate: tuple[Array, ...]
    external_fluid_momentum_rate: Array
    external_fluid_energy_rate: Array
    external_fluid_species_rate: Array
    time: Array
    step_size: Array
    stage_id: str = eqx.field(static=True)


class ReactiveMonolithicResidualEvaluation(StrictModule):
    residual: ReactiveMonolithicUnknown
    conversion_state: ParticleConversionState
    exchange: object
    conversion: object
    particle_force: Array
    fluid_momentum_source: Array
    route: ReactiveMonolithicRouteCertificate
    energy_residual: Array
    species_residual: Array
    momentum_residual: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReactiveMonolithicCouplingPlan(StrictModule, NonTrainableState):
    fluid: CellwiseReactiveFluidImplicitPlan
    conversion: PreparedParticleConversionDynamics
    continuum_exchange: ParticleContinuumExchangePlan
    drag_coefficient: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        fluid: CellwiseReactiveFluidImplicitPlan,
        conversion: PreparedParticleConversionDynamics,
        continuum_exchange: ParticleContinuumExchangePlan,
        drag_coefficient: ArrayLike,
        /,
    ):
        if not isinstance(fluid, CellwiseReactiveFluidImplicitPlan):
            raise TypeError("fluid must be CellwiseReactiveFluidImplicitPlan.")
        if not isinstance(conversion, PreparedParticleConversionDynamics):
            raise TypeError("conversion must be PreparedParticleConversionDynamics.")
        if not isinstance(continuum_exchange, ParticleContinuumExchangePlan):
            raise TypeError("continuum_exchange must be ParticleContinuumExchangePlan.")
        drag = np.asarray(drag_coefficient, dtype=float)
        capacity = continuum_exchange.transfer.particles.capacity
        if drag.shape != (capacity,) or np.any(~np.isfinite(drag)) or np.any(drag < 0.0):
            raise ValueError("drag_coefficient must be nonnegative particle data.")
        if fluid.cell_count != continuum_exchange.transfer.cell_count:
            raise ValueError("Fluid and continuum exchange cell counts differ.")
        if fluid.species_count != continuum_exchange.species_count:
            raise ValueError("Fluid and exchange species counts differ.")
        self.fluid = fluid
        self.conversion = conversion
        self.continuum_exchange = continuum_exchange
        self.drag_coefficient = jnp.asarray(drag)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-monolithic-coupling-plan",
                "fluid": fluid.plan_id,
                "conversion": conversion.dynamics_id,
                "exchange": continuum_exchange.plan_id,
                "drag": array_tree_fingerprint(drag),
            }
        )

    def initial_unknown(self, stage: ReactiveMonolithicStage, /):
        return ReactiveMonolithicUnknown(
            stage.previous_fluid.velocity,
            stage.previous_fluid.temperature,
            stage.previous_fluid.species_concentration,
            stage.previous_particle_velocity,
            tuple(value.internal_energy for value in stage.previous_conversion.batches),
            tuple(value.species_amount for value in stage.previous_conversion.batches),
        )

    def conversion_state(
        self, unknown: ReactiveMonolithicUnknown, stage: ReactiveMonolithicStage, /
    ) -> ParticleConversionState:
        batches = tuple(
            eqx.tree_at(
                lambda value: (value.internal_energy, value.species_amount),
                previous,
                (energy, species),
            )
            for previous, energy, species in zip(
                stage.previous_conversion.batches,
                unknown.batch_internal_energy,
                unknown.batch_species_amount,
                strict=True,
            )
        )
        return ParticleConversionState(
            batches,
            stage.previous_conversion.ledger,
            stage.previous_conversion.state_id,
        )

    def evaluate(
        self, unknown: ReactiveMonolithicUnknown, stage: ReactiveMonolithicStage, /
    ) -> ReactiveMonolithicResidualEvaluation:
        if not isinstance(unknown, ReactiveMonolithicUnknown):
            raise TypeError("unknown must be ReactiveMonolithicUnknown.")
        if not isinstance(stage, ReactiveMonolithicStage):
            raise TypeError("stage must be ReactiveMonolithicStage.")
        dt = stage.step_size
        candidate_conversion = self.conversion_state(unknown, stage)
        thermodynamics = tuple(
            value.thermodynamics for value in self.conversion.problem.materials
        )
        exchange = self.continuum_exchange.evaluate(
            stage.particle_position,
            self.conversion.batches,
            candidate_conversion,
            thermodynamics,
            unknown.fluid_temperature,
            unknown.fluid_species_concentration,
            active_mask=stage.particle_active,
        )
        conversion_evaluation = self.conversion.evaluate(
            candidate_conversion, stage.conversion_boundaries
        )
        batch_energy_residual = []
        batch_species_residual = []
        for index, (previous, candidate, evaluation) in enumerate(
            zip(
                stage.previous_conversion.batches,
                candidate_conversion.batches,
                conversion_evaluation.batches,
                strict=True,
            )
        ):
            energy_rate = (
                evaluation.internal_energy_rate
                + exchange.batch_internal_energy_rate[index]
                + stage.contact_energy_rate[index]
                + stage.radiative_energy_rate[index]
            )
            species_rate = (
                evaluation.species_amount_rate + exchange.batch_species_amount_rate[index]
            )
            batch_energy_residual.append(
                candidate.internal_energy - previous.internal_energy - dt * energy_rate
            )
            batch_species_residual.append(
                candidate.species_amount - previous.species_amount - dt * species_rate
            )
        relation = self.continuum_exchange.transfer.relation(
            stage.particle_position, active_mask=stage.particle_active
        )
        sampled_fluid_velocity = self.continuum_exchange.transfer.gather(
            relation, unknown.fluid_velocity
        )
        particle_force = self.drag_coefficient[:, None] * (
            sampled_fluid_velocity - unknown.particle_velocity
        )
        particle_force = jnp.where(stage.particle_active[:, None], particle_force, 0.0)
        fluid_momentum_source = self.continuum_exchange.transfer.deposit_particle_content(
            relation, -particle_force
        )
        particle_momentum_residual = (
            stage.particle_mass[:, None]
            * (unknown.particle_velocity - stage.previous_particle_velocity)
            - dt * particle_force
        )
        particle_momentum_residual = jnp.where(
            stage.particle_active[:, None], particle_momentum_residual, 0.0
        )
        fluid_momentum_residual = self.fluid.cell_mass[:, None] * (
            unknown.fluid_velocity - stage.previous_fluid.velocity
        ) - dt * (fluid_momentum_source + stage.external_fluid_momentum_rate)
        fluid_energy_residual = self.fluid.cell_heat_capacity * (
            unknown.fluid_temperature - stage.previous_fluid.temperature
        ) - dt * (exchange.fluid_energy_source_rate + stage.external_fluid_energy_rate)
        fluid_species_residual = self.fluid.species_storage * (
            unknown.fluid_species_concentration
            - stage.previous_fluid.species_concentration
        ) - dt * (exchange.fluid_species_source_rate + stage.external_fluid_species_rate)
        residual = ReactiveMonolithicUnknown(
            fluid_momentum_residual,
            fluid_energy_residual,
            fluid_species_residual,
            particle_momentum_residual,
            tuple(batch_energy_residual),
            tuple(batch_species_residual),
        )
        transfer_digest = jnp.sum(
            relation.cell_indices.astype(jnp.int64)
            * (jnp.arange(relation.cell_indices.shape[1], dtype=jnp.int64)[None, :] + 1)
            * relation.valid.astype(jnp.int64)
        )
        species_margin = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(
                        jnp.where(
                            value.active[:, None, None],
                            value.species_amount,
                            jnp.inf,
                        )
                    )
                    for value in candidate_conversion.batches
                )
            )
        )
        temperature_margin = jnp.min(
            jnp.stack(
                tuple(
                    jnp.min(value.transport.thermodynamic_state.temperature_margin)
                    for value in conversion_evaluation.batches
                )
            )
        )
        route = ReactiveMonolithicRouteCertificate(
            stage.particle_active,
            tuple(value.active for value in candidate_conversion.batches),
            transfer_digest,
            species_margin,
            temperature_margin,
            jnp.asarray(True),
            jnp.asarray(True),
            canonical_fingerprint(
                {
                    "kind": "reactive-monolithic-route-certificate",
                    "plan": self.plan_id,
                    "stage": stage.stage_id,
                }
            ),
        )
        momentum_residual = jnp.sum(fluid_momentum_source, axis=0) + jnp.sum(
            particle_force, axis=0
        )
        energy_residual = exchange.energy_residual
        species_residual = exchange.species_residual
        fluid_state = ReactiveFluidImplicitState(
            unknown.fluid_velocity,
            unknown.fluid_temperature,
            unknown.fluid_species_concentration,
        )
        successful = (
            self.fluid.validate_state(fluid_state)
            & conversion_evaluation.successful
            & exchange.successful
            & tree_allfinite(residual)
            & jnp.all(stage.particle_mass >= 0.0)
            & jnp.all(~stage.particle_active | (stage.particle_mass > 0.0))
        )
        route = eqx.tree_at(
            lambda value: (value.unchanged, value.successful),
            route,
            (successful, successful),
        )
        return ReactiveMonolithicResidualEvaluation(
            residual,
            candidate_conversion,
            exchange,
            conversion_evaluation,
            particle_force,
            fluid_momentum_source,
            route,
            energy_residual,
            species_residual,
            momentum_residual,
            successful,
            self.plan_id,
        )


__all__ = [
    "CellwiseReactiveFluidImplicitPlan",
    "ReactiveFluidImplicitState",
    "ReactiveMonolithicCouplingPlan",
    "ReactiveMonolithicResidualEvaluation",
    "ReactiveMonolithicRouteCertificate",
    "ReactiveMonolithicStage",
    "ReactiveMonolithicUnknown",
]
