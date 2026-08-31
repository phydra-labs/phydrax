#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import StrEnum

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMEvaluation, PreparedSoftSphereDEMDynamics
from ._pairwise import scatter_pair_exchange, scatter_pair_sum
from ._particle_internal_mesh import PreparedParticleInternalBatch
from ._particle_internal_state import ParticleConversionState


class ContactAreaMode(StrEnum):
    CONSTANT = "constant"
    OVERLAP = "overlap"
    PROJECTION = "projection"


class ParticleContactExchangeEvaluation(StrictModule):
    batch_internal_energy_rate: tuple[Array, ...]
    owner_internal_energy_rate: Array
    pair_heat_to_left: Array
    boundary_heat_to_particles: tuple[Array, ...]
    mechanical_heat_rate: Array
    energy_residual: Array
    entropy_production: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ParticleContactExchangePlan(StrictModule, NonTrainableState):
    conductance: Array
    area_mode: ContactAreaMode = eqx.field(static=True)
    constant_area: float = eqx.field(static=True)
    original_young_modulus: Array | None
    area_correction_exponent: float = eqx.field(static=True)
    normal_viscous_fraction: float = eqx.field(static=True)
    tangential_fraction: float = eqx.field(static=True)
    rotational_fraction: float = eqx.field(static=True)
    plastic_fraction: float = eqx.field(static=True)
    cohesion_fraction: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        conductance: ArrayLike,
        /,
        *,
        area_mode: ContactAreaMode = ContactAreaMode.CONSTANT,
        constant_area: float = 1.0,
        original_young_modulus: ArrayLike | None = None,
        area_correction_exponent: float = 1.0,
        normal_viscous_fraction: float = 0.0,
        tangential_fraction: float = 0.0,
        rotational_fraction: float = 0.0,
        plastic_fraction: float = 0.0,
        cohesion_fraction: float = 0.0,
        plan_id: str | None = None,
    ):
        values = np.asarray(conductance)
        if (
            values.ndim != 2
            or values.shape[0] != values.shape[1]
            or not np.array_equal(values, values.T)
            or np.any(~np.isfinite(values))
            or np.any(values < 0.0)
        ):
            raise ValueError("conductance must be a symmetric nonnegative pair table.")
        if not isinstance(area_mode, ContactAreaMode):
            raise TypeError("area_mode must be a ContactAreaMode.")
        area = float(constant_area)
        exponent = float(area_correction_exponent)
        fractions = tuple(
            float(value)
            for value in (
                normal_viscous_fraction,
                tangential_fraction,
                rotational_fraction,
                plastic_fraction,
                cohesion_fraction,
            )
        )
        if (
            not np.isfinite(area)
            or area <= 0.0
            or not np.isfinite(exponent)
            or exponent < 0.0
            or any(
                not np.isfinite(value) or value < 0.0 or value > 1.0
                for value in fractions
            )
        ):
            raise ValueError("Contact exchange area/fractions are invalid.")
        original = (
            None
            if original_young_modulus is None
            else np.asarray(original_young_modulus, dtype=float)
        )
        if original is not None and (
            original.shape != (values.shape[0],)
            or np.any(~np.isfinite(original))
            or np.any(original <= 0.0)
        ):
            raise ValueError("original_young_modulus must have material shape.")
        generated = canonical_fingerprint(
            {
                "kind": "particle-contact-exchange-plan",
                "conductance": array_tree_fingerprint(values),
                "area_mode": area_mode.value,
                "constant_area": area,
                "original_young_modulus": None
                if original is None
                else array_tree_fingerprint(original),
                "area_correction_exponent": exponent,
                "fractions": fractions,
            }
        )
        self.conductance = jnp.asarray(values)
        self.area_mode = area_mode
        self.constant_area = area
        self.original_young_modulus = None if original is None else jnp.asarray(original)
        self.area_correction_exponent = exponent
        (
            self.normal_viscous_fraction,
            self.tangential_fraction,
            self.rotational_fraction,
            self.plastic_fraction,
            self.cohesion_fraction,
        ) = fractions
        self.plan_id = generated if plan_id is None else str(plan_id)
        if not self.plan_id:
            raise ValueError("plan_id must be nonempty.")

    def evaluate(
        self,
        dynamics: PreparedSoftSphereDEMDynamics,
        evaluation: DEMEvaluation,
        conversion_batches: tuple[PreparedParticleInternalBatch, ...],
        conversion_state: ParticleConversionState,
        thermodynamic_materials,
        step_size: Array,
        /,
        *,
        boundary_temperatures: ArrayLike = (),
    ) -> ParticleContactExchangeEvaluation:
        if not isinstance(dynamics, PreparedSoftSphereDEMDynamics):
            raise TypeError("dynamics must be PreparedSoftSphereDEMDynamics.")
        if not isinstance(evaluation, DEMEvaluation):
            raise TypeError("evaluation must be a DEMEvaluation.")
        batches = tuple(conversion_batches)
        materials = tuple(thermodynamic_materials)
        if len(batches) != len(conversion_state.batches) or len(batches) != len(
            materials
        ):
            raise ValueError("Conversion batches, state, and thermodynamics must match.")
        owner_temperature = jnp.zeros(
            (dynamics.bodies.capacity,), dtype=dynamics.bodies.radii.dtype
        )
        owner_coverage = jnp.zeros((dynamics.bodies.capacity,), dtype=jnp.int32)
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
            owner_temperature = owner_temperature.at[prepared.owner_indices].set(
                thermo.temperature[:, -1]
            )
            owner_coverage = owner_coverage.at[prepared.owner_indices].add(
                state.active.astype(jnp.int32)
            )
        pairs = evaluation.neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        material_left = dynamics.bodies.material_ids[left]
        material_right = dynamics.bodies.material_ids[right]
        coefficient = self.conductance[material_left, material_right]
        overlap = evaluation.particle_contact.next_history.normal.previous_overlap
        radius_left = dynamics.bodies.radii[left]
        radius_right = dynamics.bodies.radii[right]
        effective_radius = (
            radius_left * radius_right / jnp.maximum(radius_left + radius_right, 1.0e-30)
        )
        if self.area_mode is ContactAreaMode.CONSTANT:
            area = jnp.full_like(coefficient, self.constant_area)
        elif self.area_mode is ContactAreaMode.OVERLAP:
            area = jnp.pi * effective_radius * overlap
        else:
            area = jnp.pi * jnp.minimum(radius_left, radius_right) ** 2
        if self.original_young_modulus is not None:
            simulated = dynamics.materials.effective_young_modulus(
                material_left, material_right
            )
            original_inverse = (
                1.0 - dynamics.materials.poisson_ratio[material_left] ** 2
            ) / self.original_young_modulus[material_left] + (
                1.0 - dynamics.materials.poisson_ratio[material_right] ** 2
            ) / self.original_young_modulus[material_right]
            original_effective = 1.0 / original_inverse
            area = (
                area * (simulated / original_effective) ** self.area_correction_exponent
            )
        conductance = coefficient * area
        contact_active = evaluation.particle_contact.active & pairs.valid
        pair_heat_left = conductance * (
            owner_temperature[right] - owner_temperature[left]
        )
        pair_heat_left = jnp.where(contact_active, pair_heat_left, 0.0)
        owner_heat = scatter_pair_exchange(
            pairs,
            pair_heat_left,
            size=dynamics.bodies.capacity,
            accumulation="deterministic",
            valid=contact_active,
        )
        dt = jnp.asarray(step_size, dtype=owner_heat.dtype)
        mechanical_work = (
            self.normal_viscous_fraction
            * evaluation.particle_contact.normal_viscous_endpoint_loss
            + self.tangential_fraction
            * evaluation.particle_contact.tangential_constitutive_loss_estimate
            + self.rotational_fraction
            * evaluation.particle_contact.rotational_dissipated_work
            + self.plastic_fraction
            * evaluation.particle_contact.normal_plastic_dissipated_work
            + self.cohesion_fraction
            * evaluation.particle_contact.cohesion_dissipated_work
        )
        mechanical_rate = jnp.where(contact_active, mechanical_work / dt, 0.0)
        owner_mechanical = 0.5 * scatter_pair_sum(
            pairs,
            mechanical_rate,
            mechanical_rate,
            size=dynamics.bodies.capacity,
            accumulation="deterministic",
            valid=contact_active,
        )
        owner_heat = owner_heat + owner_mechanical
        wall_temperatures = jnp.asarray(boundary_temperatures, dtype=owner_heat.dtype)
        if wall_temperatures.shape != (len(evaluation.boundaries),):
            raise ValueError("boundary_temperatures must match DEM barriers.")
        boundary_heat = []
        for index, response in enumerate(evaluation.boundaries):
            wall_material = jnp.full(
                (dynamics.bodies.capacity,), response.material_id, dtype=jnp.int32
            )
            wall_coefficient = self.conductance[
                dynamics.bodies.material_ids, wall_material
            ]
            wall_active = response.contact.active
            wall_heat = (
                wall_coefficient
                * self.constant_area
                * (wall_temperatures[index] - owner_temperature)
            )
            wall_heat = jnp.where(wall_active, wall_heat, 0.0)
            wall_mechanical = (
                self.normal_viscous_fraction
                * response.contact.normal_viscous_endpoint_loss
                + self.tangential_fraction
                * response.contact.tangential_constitutive_loss_estimate
                + self.rotational_fraction * response.contact.rotational_dissipated_work
                + self.plastic_fraction * response.contact.normal_plastic_dissipated_work
                + self.cohesion_fraction * response.contact.cohesion_dissipated_work
            ) / dt
            wall_heat = wall_heat + jnp.where(wall_active, wall_mechanical, 0.0)
            owner_heat = owner_heat + wall_heat
            boundary_heat.append(wall_heat)
        batch_rates = []
        assigned_total = jnp.zeros((), dtype=owner_heat.dtype)
        for prepared, state in zip(batches, conversion_state.batches, strict=True):
            owner_rate = owner_heat[prepared.owner_indices]
            batch_rate = jnp.zeros_like(state.internal_energy).at[:, -1].set(owner_rate)
            batch_rates.append(batch_rate)
            assigned_total = assigned_total + jnp.sum(owner_rate)
        active_contact_count = jnp.zeros_like(owner_coverage)
        active_contact_count = active_contact_count.at[left].add(
            contact_active.astype(jnp.int32)
        )
        active_contact_count = active_contact_count.at[right].add(
            contact_active.astype(jnp.int32)
        )
        active_contact_owner = active_contact_count > 0
        coverage_valid = jnp.all(~active_contact_owner | (owner_coverage == 1))
        residual = jnp.sum(owner_heat) - assigned_total
        entropy = jnp.sum(
            jnp.where(
                contact_active,
                conductance
                * (owner_temperature[right] - owner_temperature[left]) ** 2
                / jnp.maximum(
                    owner_temperature[left] * owner_temperature[right], 1.0e-30
                ),
                0.0,
            )
        )
        successful = (
            evaluation.successful
            & coverage_valid
            & jnp.all(jnp.isfinite(owner_heat))
            & jnp.isfinite(residual)
            & (
                jnp.abs(residual)
                <= 128.0
                * jnp.finfo(owner_heat.dtype).eps
                * jnp.maximum(jnp.sum(jnp.abs(owner_heat)), 1.0)
            )
            & jnp.isfinite(entropy)
            & (entropy >= 0.0)
            & (dt > 0.0)
        )
        return ParticleContactExchangeEvaluation(
            tuple(batch_rates),
            owner_heat,
            pair_heat_left,
            tuple(boundary_heat),
            owner_mechanical,
            residual,
            entropy,
            successful,
            self.plan_id,
        )


__all__ = [
    "ContactAreaMode",
    "ParticleContactExchangeEvaluation",
    "ParticleContactExchangePlan",
]
