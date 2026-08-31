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
from ..discretization.particle import (
    DensityPorosityMorphologyPlan,
    ParticleContactExchangePlan,
    ParticleConversionState,
    PreparedParticleGridTransfer,
    PreparedParticleInternalBatch,
    PreparedSoftSphereDEMDynamics,
    ReciprocalPairRadiationPlan,
)
from ..discretization.particle._particle_internal_unstructured import (
    PreparedUnstructuredParticleInternalMesh,
)
from ._cfd_dem import UnresolvedCFDEMCouplingPlan
from ._particle_conversion import PreparedParticleConversionDynamics
from ._particle_thermochemistry import ParticleThermodynamicMaterialPlan


class ParticleContinuumExchangeEvaluation(StrictModule):
    batch_internal_energy_rate: tuple[Array, ...]
    batch_species_amount_rate: tuple[Array, ...]
    owner_heat_rate: Array
    owner_species_rate: Array
    fluid_energy_source_rate: Array
    fluid_species_source_rate: Array
    energy_residual: Array
    species_residual: Array
    entropy_production: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ParticleContinuumExchangePlan(StrictModule, NonTrainableState):
    transfer: PreparedParticleGridTransfer
    heat_transfer_coefficient: Array
    mass_transfer_coefficient: Array
    schema_id: str = eqx.field(static=True)
    species_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        transfer: PreparedParticleGridTransfer,
        heat_transfer_coefficient: ArrayLike,
        mass_transfer_coefficient: ArrayLike,
        /,
        *,
        schema_id: str,
    ):
        if not isinstance(transfer, PreparedParticleGridTransfer):
            raise TypeError("transfer must be a PreparedParticleGridTransfer.")
        heat = np.asarray(heat_transfer_coefficient, dtype=float)
        mass = np.asarray(mass_transfer_coefficient, dtype=float)
        if heat.shape != (transfer.particles.capacity,):
            raise ValueError("heat_transfer_coefficient must have particle capacity.")
        if (
            mass.ndim != 2
            or mass.shape[0] != transfer.particles.capacity
            or mass.shape[1] == 0
        ):
            raise ValueError(
                "mass_transfer_coefficient must have particle-species shape."
            )
        if (
            np.any(~np.isfinite(heat))
            or np.any(heat < 0.0)
            or np.any(~np.isfinite(mass))
            or np.any(mass < 0.0)
        ):
            raise ValueError(
                "Continuum transfer coefficients must be finite and nonnegative."
            )
        schema = str(schema_id)
        if not schema:
            raise ValueError("schema_id must be nonempty.")
        self.transfer = transfer
        self.heat_transfer_coefficient = jnp.asarray(heat)
        self.mass_transfer_coefficient = jnp.asarray(mass)
        self.schema_id = schema
        self.species_count = int(mass.shape[1])
        self.plan_id = canonical_fingerprint(
            {
                "kind": "particle-continuum-exchange-plan",
                "transfer": transfer.prepared_id,
                "heat": array_tree_fingerprint(heat),
                "mass": array_tree_fingerprint(mass),
                "schema": schema,
            }
        )

    def evaluate(
        self,
        positions: ArrayLike,
        conversion_batches: tuple[PreparedParticleInternalBatch, ...],
        conversion_state: ParticleConversionState,
        thermodynamics: tuple[ParticleThermodynamicMaterialPlan, ...],
        fluid_temperature: ArrayLike,
        fluid_species_concentration: ArrayLike,
        /,
        *,
        active_mask: ArrayLike | None = None,
    ) -> ParticleContinuumExchangeEvaluation:
        batches = tuple(conversion_batches)
        materials = tuple(thermodynamics)
        if len(batches) != len(conversion_state.batches) or len(batches) != len(
            materials
        ):
            raise ValueError("Conversion batches, state, and materials must match.")
        if any(
            value.schema.schema_id != self.schema_id
            or value.schema.species_count != self.species_count
            for value in materials
        ):
            raise ValueError("Continuum exchange requires one common species schema.")
        relation = self.transfer.relation(positions, active_mask=active_mask)
        fluid_temperature_ = jnp.asarray(fluid_temperature)
        fluid_species = jnp.asarray(
            fluid_species_concentration, dtype=fluid_temperature_.dtype
        )
        if fluid_temperature_.shape != (
            self.transfer.cell_count,
        ) or fluid_species.shape != (
            self.transfer.cell_count,
            self.species_count,
        ):
            raise ValueError("Fluid temperature/species arrays have invalid shapes.")
        sampled_temperature = self.transfer.gather(relation, fluid_temperature_)
        sampled_species = self.transfer.gather(relation, fluid_species)
        capacity = self.transfer.particles.capacity
        owner_temperature = jnp.zeros((capacity,), dtype=fluid_temperature_.dtype)
        owner_species = jnp.zeros(
            (capacity, self.species_count), dtype=fluid_temperature_.dtype
        )
        owner_surface_measure = jnp.zeros((capacity,), dtype=fluid_temperature_.dtype)
        owner_coverage = jnp.zeros((capacity,), dtype=jnp.int32)
        surface_routes = []
        for prepared, state, material in zip(
            batches, conversion_state.batches, materials, strict=True
        ):
            metrics = prepared.mesh.metrics(state.outer_scale)
            thermo = material.state(
                state.internal_energy,
                state.species_amount,
                metrics.cell_measures,
                state.porosity,
            )
            if isinstance(prepared.mesh, PreparedUnstructuredParticleInternalMesh):
                boundary_mask = metrics.boundary_faces[None, :] & metrics.active_faces
                owner_cells = metrics.owner_cells
                face_weight = jnp.where(
                    boundary_mask,
                    metrics.face_measures
                    / jnp.maximum(metrics.surface_measure[:, None], 1.0e-30),
                    0.0,
                )
                surface_temperature = jnp.sum(
                    face_weight * thermo.temperature[:, owner_cells], axis=1
                )
                concentration = state.species_amount / metrics.cell_measures[:, :, None]
                surface_species = jnp.sum(
                    face_weight[:, :, None] * concentration[:, owner_cells, :],
                    axis=1,
                )
                surface_routes.append((owner_cells, face_weight))
            else:
                surface_temperature = thermo.temperature[:, -1]
                surface_species = (
                    state.species_amount[:, -1, :] / metrics.cell_measures[:, -1, None]
                )
                surface_routes.append(None)
            owner_temperature = owner_temperature.at[prepared.owner_indices].set(
                surface_temperature
            )
            owner_species = owner_species.at[prepared.owner_indices].set(surface_species)
            owner_surface_measure = owner_surface_measure.at[prepared.owner_indices].set(
                metrics.surface_measure
            )
            owner_coverage = owner_coverage.at[prepared.owner_indices].add(
                state.active.astype(jnp.int32)
            )
        active = relation.active
        heat_rate = (
            self.heat_transfer_coefficient
            * owner_surface_measure
            * (sampled_temperature - owner_temperature)
        )
        species_rate = (
            self.mass_transfer_coefficient
            * owner_surface_measure[:, None]
            * (sampled_species - owner_species)
        )
        heat_rate = jnp.where(active, heat_rate, 0.0)
        species_rate = jnp.where(active[:, None], species_rate, 0.0)
        fluid_energy = self.transfer.deposit_particle_content(relation, -heat_rate)
        fluid_species_source = self.transfer.deposit_particle_content(
            relation, -species_rate
        )
        batch_energy = []
        batch_species = []
        for prepared, state, route in zip(
            batches, conversion_state.batches, surface_routes, strict=True
        ):
            owner_heat = heat_rate[prepared.owner_indices]
            owner_species_exchange = species_rate[prepared.owner_indices]
            if route is None:
                energy = jnp.zeros_like(state.internal_energy).at[:, -1].set(owner_heat)
                species = (
                    jnp.zeros_like(state.species_amount)
                    .at[:, -1, :]
                    .set(owner_species_exchange)
                )
            else:
                owner_cells, face_weight = route
                energy = (
                    jnp.zeros_like(state.internal_energy)
                    .at[:, owner_cells]
                    .add(owner_heat[:, None] * face_weight)
                )
                species = (
                    jnp.zeros_like(state.species_amount)
                    .at[:, owner_cells, :]
                    .add(owner_species_exchange[:, None, :] * face_weight[:, :, None])
                )
            batch_energy.append(energy)
            batch_species.append(species)
        energy_residual = jnp.sum(heat_rate) + jnp.sum(fluid_energy)
        species_residual = jnp.sum(species_rate, axis=0) + jnp.sum(
            fluid_species_source, axis=0
        )
        entropy = jnp.sum(
            jnp.where(
                active,
                self.heat_transfer_coefficient
                * owner_surface_measure
                * (sampled_temperature - owner_temperature) ** 2
                / jnp.maximum(sampled_temperature * owner_temperature, 1.0e-30),
                0.0,
            )
        )
        tolerance = 128.0 * jnp.finfo(fluid_temperature_.dtype).eps
        successful = (
            relation.successful
            & jnp.all(~active | (owner_coverage == 1))
            & jnp.all(jnp.isfinite(heat_rate))
            & jnp.all(jnp.isfinite(species_rate))
            & (
                jnp.abs(energy_residual)
                <= tolerance * jnp.maximum(jnp.sum(jnp.abs(heat_rate)), 1.0)
            )
            & jnp.all(
                jnp.abs(species_residual)
                <= tolerance * jnp.maximum(jnp.sum(jnp.abs(species_rate), axis=0), 1.0)
            )
            & jnp.isfinite(entropy)
            & (entropy >= 0.0)
        )
        return ParticleContinuumExchangeEvaluation(
            tuple(batch_energy),
            tuple(batch_species),
            heat_rate,
            species_rate,
            fluid_energy,
            fluid_species_source,
            energy_residual,
            species_residual,
            entropy,
            successful,
            self.plan_id,
        )


class ReactiveCFDDEMCouplingPlan(StrictModule, NonTrainableState):
    dem: PreparedSoftSphereDEMDynamics
    conversion: PreparedParticleConversionDynamics
    continuum_exchange: ParticleContinuumExchangePlan
    contact_exchange: ParticleContactExchangePlan | None
    hydrodynamics: UnresolvedCFDEMCouplingPlan | None
    morphology: DensityPorosityMorphologyPlan | None
    radiation: ReciprocalPairRadiationPlan | None
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dem,
        conversion,
        continuum_exchange,
        /,
        *,
        contact_exchange=None,
        hydrodynamics=None,
        morphology=None,
        radiation=None,
    ):
        if not isinstance(dem, PreparedSoftSphereDEMDynamics):
            raise TypeError("dem must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(conversion, PreparedParticleConversionDynamics):
            raise TypeError("conversion must be PreparedParticleConversionDynamics.")
        if not isinstance(continuum_exchange, ParticleContinuumExchangePlan):
            raise TypeError("continuum_exchange must be ParticleContinuumExchangePlan.")
        if contact_exchange is not None and not isinstance(
            contact_exchange, ParticleContactExchangePlan
        ):
            raise TypeError(
                "contact_exchange must be ParticleContactExchangePlan or None."
            )
        if hydrodynamics is not None and not isinstance(
            hydrodynamics, UnresolvedCFDEMCouplingPlan
        ):
            raise TypeError("hydrodynamics must be UnresolvedCFDEMCouplingPlan or None.")
        if morphology is not None and not isinstance(
            morphology, DensityPorosityMorphologyPlan
        ):
            raise TypeError("morphology must be DensityPorosityMorphologyPlan or None.")
        if radiation is not None and not isinstance(
            radiation, ReciprocalPairRadiationPlan
        ):
            raise TypeError("radiation must be ReciprocalPairRadiationPlan or None.")
        if (
            dem.bodies.particles.prepared_id
            != continuum_exchange.transfer.particles.prepared_id
        ):
            raise ValueError("DEM and continuum transfer populations must match.")
        self.dem = dem
        self.conversion = conversion
        self.continuum_exchange = continuum_exchange
        self.contact_exchange = contact_exchange
        self.hydrodynamics = hydrodynamics
        self.morphology = morphology
        self.radiation = radiation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "reactive-cfd-dem-coupling-plan",
                "dem": dem.prepared_id,
                "conversion": conversion.dynamics_id,
                "continuum_exchange": continuum_exchange.plan_id,
                "contact_exchange": None
                if contact_exchange is None
                else contact_exchange.plan_id,
                "hydrodynamics": None if hydrodynamics is None else hydrodynamics.plan_id,
                "morphology": None if morphology is None else morphology.plan_id,
                "radiation": None if radiation is None else radiation.plan_id,
            }
        )


__all__ = [
    "ParticleContinuumExchangeEvaluation",
    "ParticleContinuumExchangePlan",
    "ReactiveCFDDEMCouplingPlan",
]
