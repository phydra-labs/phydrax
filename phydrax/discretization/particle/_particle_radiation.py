#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._dem import DEMEvaluation, PreparedSoftSphereDEMDynamics
from ._pairwise import scatter_pair_exchange
from ._particle_internal_mesh import PreparedParticleInternalBatch
from ._particle_internal_state import ParticleConversionState


_STEFAN_BOLTZMANN = 5.670374419e-8


class ParticleRadiationEvaluation(StrictModule):
    batch_internal_energy_rate: tuple[Array, ...]
    owner_internal_energy_rate: Array
    pair_heat_to_left: Array
    boundary_heat_to_particles: tuple[Array, ...]
    boundary_heat_source_rate: Array
    energy_residual: Array
    entropy_production: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class ReciprocalPairRadiationPlan(StrictModule, NonTrainableState):
    particle_emissivity: Array
    pair_view_factor: Array
    wall_emissivity: Array
    wall_view_factor: Array
    maximum_range: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        particle_emissivity: ArrayLike,
        pair_view_factor: ArrayLike,
        /,
        *,
        wall_emissivity: ArrayLike = (),
        wall_view_factor: ArrayLike = (),
        maximum_range: float,
        plan_id: str | None = None,
    ):
        emissivity = np.asarray(particle_emissivity, dtype=float)
        view = np.asarray(pair_view_factor, dtype=float)
        wall = np.asarray(wall_emissivity, dtype=float)
        wall_view = np.asarray(wall_view_factor, dtype=float)
        cutoff = float(maximum_range)
        if (
            emissivity.ndim != 1
            or emissivity.size == 0
            or np.any(~np.isfinite(emissivity))
            or np.any((emissivity <= 0.0) | (emissivity > 1.0))
            or view.shape != (emissivity.size, emissivity.size)
            or not np.array_equal(view, view.T)
            or np.any(~np.isfinite(view))
            or np.any((view < 0.0) | (view > 1.0))
            or wall.ndim != 1
            or wall_view.shape != wall.shape
            or np.any(~np.isfinite(wall))
            or np.any((wall <= 0.0) | (wall > 1.0))
            or np.any(~np.isfinite(wall_view))
            or np.any((wall_view < 0.0) | (wall_view > 1.0))
            or not np.isfinite(cutoff)
            or cutoff <= 0.0
        ):
            raise ValueError("Radiation emissivity/view-factor inputs are invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "reciprocal-pair-radiation",
                "particle_emissivity": array_tree_fingerprint(emissivity),
                "pair_view_factor": array_tree_fingerprint(view),
                "wall_emissivity": array_tree_fingerprint(wall),
                "wall_view_factor": array_tree_fingerprint(wall_view),
                "maximum_range": cutoff,
            }
        )
        self.particle_emissivity = jnp.asarray(emissivity)
        self.pair_view_factor = jnp.asarray(view)
        self.wall_emissivity = jnp.asarray(wall)
        self.wall_view_factor = jnp.asarray(wall_view)
        self.maximum_range = cutoff
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
        /,
        *,
        wall_temperatures: ArrayLike = (),
    ) -> ParticleRadiationEvaluation:
        batches = tuple(conversion_batches)
        materials = tuple(thermodynamic_materials)
        if len(batches) != len(conversion_state.batches) or len(batches) != len(
            materials
        ):
            raise ValueError("Radiation conversion batches/state/materials must match.")
        capacity = dynamics.bodies.capacity
        dtype = conversion_state.batches[0].internal_energy.dtype
        owner_temperature = jnp.zeros((capacity,), dtype=dtype)
        owner_area = jnp.zeros((capacity,), dtype=dtype)
        coverage = jnp.zeros((capacity,), dtype=jnp.int32)
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
            owner_area = owner_area.at[prepared.owner_indices].set(
                metrics.surface_measure
            )
            coverage = coverage.at[prepared.owner_indices].add(
                state.active.astype(jnp.int32)
            )
        pairs = evaluation.neighborhood.pair_relation
        left = pairs.left_indices
        right = pairs.right_indices
        left_material = dynamics.bodies.material_ids[left]
        right_material = dynamics.bodies.material_ids[right]
        emissivity = _effective_emissivity(
            self.particle_emissivity[left_material],
            self.particle_emissivity[right_material],
        )
        reciprocal_area = self.pair_view_factor[
            left_material, right_material
        ] * jnp.minimum(owner_area[left], owner_area[right])
        conductance = _STEFAN_BOLTZMANN * emissivity * reciprocal_area
        pair_valid = pairs.valid & (coverage[left] == 1) & (coverage[right] == 1)
        pair_heat = conductance * (
            owner_temperature[right] ** 4 - owner_temperature[left] ** 4
        )
        pair_heat = jnp.where(pair_valid, pair_heat, 0.0)
        owner_heat = scatter_pair_exchange(
            pairs,
            pair_heat,
            size=capacity,
            accumulation="deterministic",
            valid=pair_valid,
        )
        wall_temperature = jnp.asarray(wall_temperatures, dtype=dtype)
        if (
            wall_temperature.shape != self.wall_emissivity.shape
            or wall_temperature.shape != (len(evaluation.boundaries),)
        ):
            raise ValueError("Wall radiation arrays must match DEM barriers.")
        boundary_heat = []
        wall_source = jnp.zeros_like(wall_temperature)
        for index, response in enumerate(evaluation.boundaries):
            particle_emissivity = self.particle_emissivity[dynamics.bodies.material_ids]
            effective = _effective_emissivity(
                particle_emissivity,
                self.wall_emissivity[index],
            )
            coefficient = (
                _STEFAN_BOLTZMANN * effective * self.wall_view_factor[index] * owner_area
            )
            active = response.contact.active & (coverage == 1)
            heat = coefficient * (wall_temperature[index] ** 4 - owner_temperature**4)
            heat = jnp.where(active, heat, 0.0)
            owner_heat = owner_heat + heat
            boundary_heat.append(heat)
            wall_source = wall_source.at[index].set(-jnp.sum(heat))
        batch_rates = []
        assigned = jnp.zeros((), dtype=dtype)
        for prepared, state in zip(batches, conversion_state.batches, strict=True):
            values = owner_heat[prepared.owner_indices]
            batch_rates.append(
                jnp.zeros_like(state.internal_energy).at[:, -1].set(values)
            )
            assigned = assigned + jnp.sum(values)
        residual = assigned + jnp.sum(wall_source)
        pair_entropy = jnp.sum(
            pair_heat
            * (
                1.0 / jnp.maximum(owner_temperature[left], 1.0e-30)
                - 1.0 / jnp.maximum(owner_temperature[right], 1.0e-30)
            )
        )
        wall_entropy = (
            jnp.sum(
                jnp.stack(
                    tuple(
                        jnp.sum(
                            heat
                            * (
                                1.0 / jnp.maximum(owner_temperature, 1.0e-30)
                                - 1.0 / wall_temperature[index]
                            )
                        )
                        for index, heat in enumerate(boundary_heat)
                    )
                )
            )
            if boundary_heat
            else jnp.zeros((), dtype=dtype)
        )
        entropy = pair_entropy + wall_entropy
        tolerance = 128.0 * jnp.finfo(dtype).eps
        successful = (
            evaluation.successful
            & jnp.all(jnp.isfinite(owner_heat))
            & jnp.all(jnp.isfinite(pair_heat))
            & jnp.isfinite(residual)
            & (
                jnp.abs(residual)
                <= tolerance * jnp.maximum(jnp.sum(jnp.abs(owner_heat)), 1.0)
            )
            & jnp.isfinite(entropy)
            & (entropy >= -tolerance)
        )
        return ParticleRadiationEvaluation(
            tuple(batch_rates),
            owner_heat,
            pair_heat,
            tuple(boundary_heat),
            wall_source,
            residual,
            entropy,
            successful,
            self.plan_id,
        )


def _effective_emissivity(left, right):
    return 1.0 / (1.0 / left + 1.0 / right - 1.0)


__all__ = ["ParticleRadiationEvaluation", "ReciprocalPairRadiationPlan"]
