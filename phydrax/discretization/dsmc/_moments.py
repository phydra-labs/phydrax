#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan, DSMCStructuredCellPlan


class DSMCMomentReason(IntFlag):
    INSUFFICIENT_PARTICLES = 1 << 8
    INSUFFICIENT_BLOCKS = 1 << 9
    RELATIVE_ERROR_EXCEEDED = 1 << 10


class DSMCMomentEvaluation(StrictModule):
    species_number_density: Array
    number_density: Array
    mass_density: Array
    velocity: Array
    pressure_tensor: Array
    translational_temperature: Array
    rotational_energy_density: Array
    vibrational_energy_density: Array
    total_energy_density: Array
    particle_count: Array
    sample_weight: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCMomentAccumulatorState(StrictModule):
    block_sum: Array
    steps_in_block: Array
    block_mean: Array
    block_m2: Array
    completed_blocks: Array
    plan_id: str = eqx.field(static=True)


class DSMCStatisticalEvidence(StrictModule):
    mean: Array
    covariance: Array
    relative_standard_error: Array
    completed_blocks: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCMomentPlan(StrictModule, NonTrainableState):
    """Weighted cell moments and fixed-block uncertainty evidence."""

    species: DSMCSpeciesPlan
    cells: DSMCStructuredCellPlan
    minimum_particles_per_cell: int = eqx.field(static=True)
    block_size: int = eqx.field(static=True)
    minimum_blocks: int = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        species: DSMCSpeciesPlan,
        cells: DSMCStructuredCellPlan,
        /,
        *,
        minimum_particles_per_cell: int = 2,
        block_size: int = 1,
        minimum_blocks: int = 2,
        maximum_relative_standard_error: float = 0.25,
    ) -> None:
        minimum_particles = int(minimum_particles_per_cell)
        block = int(block_size)
        blocks = int(minimum_blocks)
        maximum_error = float(maximum_relative_standard_error)
        if (
            not isinstance(species, DSMCSpeciesPlan)
            or not isinstance(cells, DSMCStructuredCellPlan)
            or minimum_particles < 1
            or block < 1
            or blocks < 2
            or not np.isfinite(maximum_error)
            or maximum_error <= 0.0
        ):
            raise ValueError("DSMC moment statistics controls are invalid.")
        self.species = species
        self.cells = cells
        self.minimum_particles_per_cell = minimum_particles
        self.block_size = block
        self.minimum_blocks = blocks
        self.maximum_relative_standard_error = maximum_error
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-moments",
                "species": species.plan_id,
                "cells": cells.plan_id,
                "minimum_particles_per_cell": minimum_particles,
                "block_size": block,
                "minimum_blocks": blocks,
                "maximum_relative_standard_error": maximum_error,
            }
        )

    @property
    def observable_count(self) -> int:
        return 3 + self.cells.dimension

    def evaluate(self, particles: DSMCParticleState, /) -> DSMCMomentEvaluation:
        cell_count = self.cells.cell_count
        species_count = self.species.species_count
        dimension = particles.velocity.shape[-1]
        safe_cell = jnp.clip(particles.cell_id, 0, cell_count - 1)
        safe_species = jnp.clip(particles.species_index, 0, species_count - 1)
        finite_slot = (
            jnp.all(jnp.isfinite(particles.position), axis=-1)
            & jnp.all(jnp.isfinite(particles.velocity), axis=-1)
            & jnp.isfinite(particles.rotational_energy)
            & jnp.isfinite(particles.vibrational_energy)
            & jnp.isfinite(particles.statistical_weight)
        )
        valid_index = (
            (particles.cell_id >= 0)
            & (particles.cell_id < cell_count)
            & (particles.species_index >= 0)
            & (particles.species_index < species_count)
        )
        active = (
            particles.active
            & valid_index
            & finite_slot
            & (particles.statistical_weight > 0.0)
        )
        weight = jnp.where(active, particles.statistical_weight, 0.0)
        particle_count = (
            jnp.zeros((cell_count,), dtype=jnp.int32)
            .at[safe_cell]
            .add(active.astype(jnp.int32))
        )
        species_weight = (
            jnp.zeros((cell_count, species_count), dtype=weight.dtype)
            .at[safe_cell, safe_species]
            .add(weight)
        )
        cell_weight = jnp.sum(species_weight, axis=-1)
        molecular_mass = self.species.molecular_masses[safe_species].astype(weight.dtype)
        represented_mass = weight * molecular_mass
        cell_mass = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[safe_cell]
            .add(represented_mass)
        )
        momentum = (
            jnp.zeros((cell_count, dimension), dtype=particles.velocity.dtype)
            .at[safe_cell]
            .add(represented_mass[:, None] * particles.velocity)
        )
        tiny = jnp.finfo(weight.dtype).tiny
        velocity = jnp.where(
            cell_mass[:, None] > 0.0,
            momentum / jnp.maximum(cell_mass[:, None], tiny),
            0.0,
        )
        peculiar = particles.velocity - velocity[safe_cell]
        pressure_numerator = (
            jnp.zeros((cell_count, dimension, dimension), dtype=particles.velocity.dtype)
            .at[safe_cell]
            .add(
                represented_mass[:, None, None]
                * contract("pi,pj->pij", peculiar, peculiar, backend="jax")
            )
        )
        translational_energy = 0.5 * represented_mass * jnp.sum(peculiar**2, axis=-1)
        cell_translational = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[safe_cell]
            .add(translational_energy)
        )
        rotational = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[safe_cell]
            .add(jnp.where(active, weight * particles.rotational_energy, 0.0))
        )
        vibrational = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[safe_cell]
            .add(jnp.where(active, weight * particles.vibrational_energy, 0.0))
        )
        absolute_kinetic = (
            0.5 * represented_mass * jnp.sum(particles.velocity**2, axis=-1)
        )
        absolute_energy = (
            jnp.zeros((cell_count,), dtype=weight.dtype)
            .at[safe_cell]
            .add(
                absolute_kinetic
                + jnp.where(
                    active,
                    weight * (particles.rotational_energy + particles.vibrational_energy),
                    0.0,
                )
            )
        )
        volume = self.cells.cell_volumes.astype(weight.dtype)
        boltzmann = jnp.asarray(1.380649e-23, dtype=weight.dtype)
        temperature = jnp.where(
            cell_weight > 0.0,
            2.0
            * cell_translational
            / jnp.maximum(dimension * boltzmann * cell_weight, tiny),
            0.0,
        )
        invalid_active = jnp.any(particles.active & ~active)
        finite = (
            ~invalid_active
            & jnp.all(jnp.isfinite(velocity))
            & jnp.all(jnp.isfinite(temperature))
            & jnp.all(jnp.isfinite(pressure_numerator))
        )
        enough = particle_count >= self.minimum_particles_per_cell
        reasons = jnp.zeros((cell_count,), dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            enough,
            reasons,
            reasons
            | jnp.asarray(int(DSMCMomentReason.INSUFFICIENT_PARTICLES), jnp.uint32),
        )
        margin = (particle_count - self.minimum_particles_per_cell).astype(weight.dtype)
        margin = jnp.where(finite, margin, -jnp.inf)
        header = AdmissibilityHeader(
            margin,
            reasons,
            self.plan_id,
            canonical_fingerprint({"kind": "dsmc-moment-evidence", "plan": self.plan_id}),
        )
        return DSMCMomentEvaluation(
            species_weight / volume[:, None],
            cell_weight / volume,
            cell_mass / volume,
            velocity,
            pressure_numerator / volume[:, None, None],
            temperature,
            rotational / volume,
            vibrational / volume,
            absolute_energy / volume,
            particle_count,
            cell_weight,
            header,
            self.plan_id,
        )

    def observable(self, moments: DSMCMomentEvaluation, /) -> Array:
        if (
            not isinstance(moments, DSMCMomentEvaluation)
            or moments.plan_id != self.plan_id
        ):
            raise ValueError("DSMC moments do not belong to this statistics plan.")
        return jnp.concatenate(
            (
                moments.number_density[:, None],
                moments.mass_density[:, None],
                moments.velocity,
                moments.translational_temperature[:, None],
            ),
            axis=-1,
        )

    def initialize_accumulator(self, dtype=jnp.float64, /) -> DSMCMomentAccumulatorState:
        shape = (self.cells.cell_count, self.observable_count)
        return DSMCMomentAccumulatorState(
            jnp.zeros(shape, dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.zeros(shape, dtype=dtype),
            jnp.zeros(shape + (self.observable_count,), dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def accumulate(
        self,
        state: DSMCMomentAccumulatorState,
        moments: DSMCMomentEvaluation,
        /,
    ) -> DSMCMomentAccumulatorState:
        if (
            not isinstance(state, DSMCMomentAccumulatorState)
            or state.plan_id != self.plan_id
        ):
            raise ValueError("DSMC moment accumulator does not belong to this plan.")
        value = self.observable(moments).astype(state.block_sum.dtype)
        block_sum = state.block_sum + value
        steps = state.steps_in_block + 1
        complete = steps == self.block_size
        block_value = block_sum / self.block_size
        completed = state.completed_blocks + complete.astype(jnp.int32)
        denominator = jnp.maximum(completed.astype(value.dtype), 1.0)
        delta = block_value - state.block_mean
        mean = state.block_mean + jnp.where(
            complete, delta / denominator, jnp.zeros_like(delta)
        )
        second_delta = block_value - mean
        outer = contract("ci,cj->cij", delta, second_delta, backend="jax")
        m2 = state.block_m2 + jnp.where(complete, outer, jnp.zeros_like(outer))
        return DSMCMomentAccumulatorState(
            jnp.where(complete, jnp.zeros_like(block_sum), block_sum),
            jnp.where(complete, jnp.asarray(0, jnp.int32), steps),
            mean,
            m2,
            completed,
            self.plan_id,
        )

    def statistical_evidence(
        self, state: DSMCMomentAccumulatorState, /
    ) -> DSMCStatisticalEvidence:
        if (
            not isinstance(state, DSMCMomentAccumulatorState)
            or state.plan_id != self.plan_id
        ):
            raise ValueError("DSMC moment accumulator does not belong to this plan.")
        dtype = state.block_mean.dtype
        completed = state.completed_blocks
        covariance = state.block_m2 / jnp.maximum(completed - 1, 1).astype(dtype)
        variance = jnp.maximum(jnp.diagonal(covariance, axis1=-2, axis2=-1), 0.0)
        standard_error = jnp.sqrt(variance / jnp.maximum(completed, 1).astype(dtype))
        relative = standard_error / jnp.maximum(
            jnp.abs(state.block_mean), jnp.sqrt(jnp.finfo(dtype).eps)
        )
        enough = completed >= self.minimum_blocks
        resolved = jnp.all(relative <= self.maximum_relative_standard_error, axis=-1)
        finite = jnp.all(jnp.isfinite(relative), axis=-1)
        reasons = jnp.zeros((self.cells.cell_count,), dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            enough,
            reasons,
            reasons | jnp.asarray(int(DSMCMomentReason.INSUFFICIENT_BLOCKS), jnp.uint32),
        )
        reasons = jnp.where(
            resolved | ~enough,
            reasons,
            reasons
            | jnp.asarray(int(DSMCMomentReason.RELATIVE_ERROR_EXCEEDED), jnp.uint32),
        )
        block_margin = (completed - self.minimum_blocks).astype(dtype)
        error_margin = self.maximum_relative_standard_error - jnp.max(relative, axis=-1)
        margin = jnp.minimum(block_margin, error_margin)
        header = AdmissibilityHeader(
            margin,
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dsmc-statistical-evidence", "plan": self.plan_id}
            ),
        )
        return DSMCStatisticalEvidence(
            state.block_mean,
            covariance,
            relative,
            completed,
            header,
            self.plan_id,
        )


__all__ = [
    "DSMCMomentAccumulatorState",
    "DSMCMomentEvaluation",
    "DSMCMomentPlan",
    "DSMCMomentReason",
    "DSMCStatisticalEvidence",
]
