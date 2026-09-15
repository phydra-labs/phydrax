#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntFlag
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike, PRNGKeyArray

from ..._admissibility import AdmissibilityHeader, AdmissibilityReason
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DSMCParticleState, DSMCSpeciesPlan, DSMCStructuredCellPlan
from ._surface import DSMCSurfaceInteractionPlan


class DSMCBoundaryReason(IntFlag):
    INJECTION_CAPACITY_EXCEEDED = 1 << 8
    PARTICLE_CAPACITY_EXCEEDED = 1 << 9
    INVALID_RESERVOIR = 1 << 10


class DSMCBoundaryFacePlan(StrictModule, NonTrainableState):
    cells: DSMCStructuredCellPlan
    axis: int = eqx.field(static=True)
    side: Literal["lower", "upper"] = eqx.field(static=True)
    kind: Literal["periodic", "specular", "surface", "open", "reservoir"] = eqx.field(
        static=True
    )
    outward_normal: Array
    measure: float = eqx.field(static=True)
    face_id: str = eqx.field(static=True)

    def __init__(
        self,
        cells: DSMCStructuredCellPlan,
        axis: int,
        side: Literal["lower", "upper"],
        kind: Literal["periodic", "specular", "surface", "open", "reservoir"],
        /,
    ) -> None:
        axis_ = int(axis)
        if (
            not isinstance(cells, DSMCStructuredCellPlan)
            or axis_ < 0
            or axis_ >= cells.dimension
            or side not in ("lower", "upper")
            or kind not in ("periodic", "specular", "surface", "open", "reservoir")
        ):
            raise ValueError("DSMC boundary face declaration is invalid.")
        normal = np.zeros((cells.dimension,), dtype=float)
        normal[axis_] = -1.0 if side == "lower" else 1.0
        lengths = np.asarray(cells.upper - cells.lower)
        measure = (
            float(np.prod(np.delete(lengths, axis_))) if cells.dimension > 1 else 1.0
        )
        self.cells = cells
        self.axis = axis_
        self.side = side
        self.kind = kind
        self.outward_normal = jnp.asarray(normal)
        self.measure = measure
        self.face_id = canonical_fingerprint(
            {
                "kind": "dsmc-boundary-face",
                "cells": cells.plan_id,
                "axis": axis_,
                "side": side,
                "condition": kind,
                "measure": measure,
            }
        )


class DSMCSurfaceBoundaryPlan(StrictModule, NonTrainableState):
    """Bind one physical wall kernel to an axis-aligned streaming face."""

    face: DSMCBoundaryFacePlan
    interaction: DSMCSurfaceInteractionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        face: DSMCBoundaryFacePlan,
        interaction: DSMCSurfaceInteractionPlan,
        /,
    ) -> None:
        if (
            not isinstance(face, DSMCBoundaryFacePlan)
            or face.kind != "surface"
            or not isinstance(interaction, DSMCSurfaceInteractionPlan)
            or interaction.wall_velocity.shape != (face.cells.dimension,)
        ):
            raise ValueError("DSMC surface boundary plans are incompatible.")
        self.face = face
        self.interaction = interaction
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-surface-boundary",
                "face": face.face_id,
                "interaction": interaction.plan_id,
            }
        )


class DSMCReservoirState(StrictModule):
    fractional_remainder: Array
    epoch: Array
    plan_id: str = eqx.field(static=True)


class DSMCReservoirResult(StrictModule):
    state: DSMCParticleState
    candidate_reservoir_state: DSMCReservoirState
    injected_count: Array
    injected_mass: Array
    injected_momentum: Array
    injected_energy: Array
    header: AdmissibilityHeader
    plan_id: str = eqx.field(static=True)


class DSMCReservoirFacePlan(StrictModule, NonTrainableState):
    """Equilibrium half-range reservoir injection on one complete box face."""

    face: DSMCBoundaryFacePlan
    species: DSMCSpeciesPlan
    number_densities: Array
    temperature: float = eqx.field(static=True)
    tangential_bulk_velocity: Array
    simulator_weights: Array
    maximum_injections_per_species: int = eqx.field(static=True)
    boltzmann_constant: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        face: DSMCBoundaryFacePlan,
        species: DSMCSpeciesPlan,
        number_densities: ArrayLike,
        tangential_bulk_velocity: ArrayLike,
        simulator_weights: ArrayLike,
        /,
        *,
        temperature: float,
        maximum_injections_per_species: int,
        boltzmann_constant: float = 1.380649e-23,
    ) -> None:
        density = np.asarray(number_densities, dtype=float)
        velocity = np.asarray(tangential_bulk_velocity, dtype=float)
        weights = np.asarray(simulator_weights, dtype=float)
        temperature_ = float(temperature)
        capacity = int(maximum_injections_per_species)
        boltzmann = float(boltzmann_constant)
        if (
            not isinstance(face, DSMCBoundaryFacePlan)
            or face.kind != "reservoir"
            or not isinstance(species, DSMCSpeciesPlan)
            or density.shape != (species.species_count,)
            or weights.shape != density.shape
            or velocity.shape != (face.cells.dimension,)
            or np.any(~np.isfinite(density))
            or np.any(density < 0.0)
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or np.any(~np.isfinite(velocity))
            or not np.isclose(
                float(np.dot(velocity, np.asarray(face.outward_normal))), 0.0
            )
            or not np.isfinite(temperature_)
            or temperature_ <= 0.0
            or capacity <= 0
            or not np.isfinite(boltzmann)
            or boltzmann <= 0.0
        ):
            raise ValueError("DSMC reservoir face data are invalid.")
        self.face = face
        self.species = species
        self.number_densities = jnp.asarray(density)
        self.temperature = temperature_
        self.tangential_bulk_velocity = jnp.asarray(velocity)
        self.simulator_weights = jnp.asarray(weights)
        self.maximum_injections_per_species = capacity
        self.boltzmann_constant = boltzmann
        self.plan_id = canonical_fingerprint(
            {
                "kind": "dsmc-reservoir-face",
                "face": face.face_id,
                "species": species.plan_id,
                "number_densities": array_tree_fingerprint(density),
                "temperature": temperature_,
                "tangential_bulk_velocity": array_tree_fingerprint(velocity),
                "simulator_weights": array_tree_fingerprint(weights),
                "maximum_injections_per_species": capacity,
                "boltzmann_constant": boltzmann,
            }
        )

    @property
    def request_capacity(self) -> int:
        return self.species.species_count * self.maximum_injections_per_species

    def initialize(self, dtype=jnp.float64, /) -> DSMCReservoirState:
        return DSMCReservoirState(
            jnp.zeros((self.species.species_count,), dtype=dtype),
            jnp.asarray(0, dtype=jnp.int32),
            self.plan_id,
        )

    def inject(
        self,
        particles: DSMCParticleState,
        state: DSMCReservoirState,
        step_size: ArrayLike,
        key: PRNGKeyArray,
        /,
    ) -> DSMCReservoirResult:
        if not isinstance(state, DSMCReservoirState) or state.plan_id != self.plan_id:
            raise ValueError("DSMC reservoir state does not belong to this face plan.")
        if jax.random.key_data(key).shape != (2,):
            raise ValueError("DSMC reservoir injection requires one PRNG key.")
        dtype = particles.position.dtype
        step = jnp.asarray(step_size, dtype=dtype)
        if step.shape != ():
            raise ValueError("DSMC reservoir step size must be scalar.")
        masses = self.species.molecular_masses.astype(dtype)
        thermal_standard_deviation = jnp.sqrt(
            self.boltzmann_constant * self.temperature / masses
        )
        incoming_flux = (
            self.number_densities.astype(dtype)
            * thermal_standard_deviation
            / jnp.sqrt(2.0 * jnp.pi)
        )
        expected = (
            incoming_flux
            * self.face.measure
            * step
            / self.simulator_weights.astype(dtype)
            + state.fractional_remainder.astype(dtype)
        )
        finite = jnp.all(jnp.isfinite(expected)) & jnp.isfinite(step) & (step > 0.0)
        counts = jnp.floor(jnp.where(finite, expected, 0.0)).astype(jnp.int32)
        injection_capacity_ok = jnp.all(counts <= self.maximum_injections_per_species)
        total_requested = jnp.sum(counts)
        free_slots = jnp.nonzero(
            ~particles.active,
            size=particles.capacity,
            fill_value=-1,
        )[0]
        free_count = jnp.sum(~particles.active)
        particle_capacity_ok = total_requested <= free_count
        reasons = jnp.asarray(0, dtype=jnp.uint32)
        reasons = jnp.where(
            finite,
            reasons,
            reasons | jnp.asarray(int(AdmissibilityReason.NONFINITE), jnp.uint32),
        )
        reasons = jnp.where(
            injection_capacity_ok,
            reasons,
            reasons
            | jnp.asarray(
                int(DSMCBoundaryReason.INJECTION_CAPACITY_EXCEEDED), jnp.uint32
            ),
        )
        reasons = jnp.where(
            particle_capacity_ok,
            reasons,
            reasons
            | jnp.asarray(int(DSMCBoundaryReason.PARTICLE_CAPACITY_EXCEEDED), jnp.uint32),
        )
        margin = jnp.minimum(
            (self.maximum_injections_per_species - jnp.max(counts)).astype(dtype),
            (free_count - total_requested).astype(dtype),
        )
        header = AdmissibilityHeader(
            jnp.where(finite, margin, -jnp.inf),
            reasons,
            self.plan_id,
            canonical_fingerprint(
                {"kind": "dsmc-reservoir-evidence", "plan": self.plan_id}
            ),
        )
        local = jnp.tile(
            jnp.arange(self.maximum_injections_per_species, dtype=jnp.int32),
            self.species.species_count,
        )
        request_species = jnp.repeat(
            jnp.arange(self.species.species_count, dtype=jnp.int32),
            self.maximum_injections_per_species,
        )
        requested = local < counts[request_species]
        request_rank = jnp.cumsum(requested.astype(jnp.int32)) - 1
        request_slot = free_slots[jnp.clip(request_rank, 0, particles.capacity - 1)]
        requested = requested & header.globally_eligible & (request_slot >= 0)

        lower = self.face.cells.lower.astype(dtype)
        upper = self.face.cells.upper.astype(dtype)
        normal = self.face.outward_normal.astype(dtype)
        epsilon = 8.0 * jnp.finfo(dtype).eps * jnp.max(upper - lower)

        def body(index, carry):
            current, mass_total, momentum_total, energy_total = carry
            species_index = request_species[index]
            slot = jnp.clip(request_slot[index], 0, particles.capacity - 1)
            event_key = jax.random.fold_in(key, index)
            uniform_position = jax.random.uniform(
                jax.random.fold_in(event_key, 0),
                (self.face.cells.dimension,),
                dtype=dtype,
            )
            position = lower + uniform_position * (upper - lower)
            boundary_coordinate = (
                lower[self.face.axis] + epsilon
                if self.face.side == "lower"
                else upper[self.face.axis] - epsilon
            )
            position = position.at[self.face.axis].set(boundary_coordinate)
            normal_uniform = jax.random.uniform(
                jax.random.fold_in(event_key, 1),
                (),
                minval=jnp.finfo(dtype).tiny,
                maxval=1.0,
                dtype=dtype,
            )
            sigma = thermal_standard_deviation[species_index]
            inward_speed = sigma * jnp.sqrt(-2.0 * jnp.log(normal_uniform))
            gaussian = jax.random.normal(
                jax.random.fold_in(event_key, 2),
                (self.face.cells.dimension,),
                dtype=dtype,
            )
            tangential = gaussian - jnp.sum(gaussian * normal) * normal
            velocity = (
                self.tangential_bulk_velocity.astype(dtype)
                - inward_speed * normal
                + sigma * tangential
            )
            rotational_dof = self.species.rotational_degrees[species_index]
            rotational = (
                jax.random.gamma(
                    jax.random.fold_in(event_key, 3),
                    jnp.maximum(0.5 * rotational_dof, 0.5),
                    dtype=dtype,
                )
                * self.boltzmann_constant
                * self.temperature
            )
            rotational = jnp.where(rotational_dof > 0.0, rotational, 0.0)
            weight = self.simulator_weights[species_index].astype(dtype)
            molecular_mass = masses[species_index]
            cell_id, inside = self.face.cells.locate(position[None, :])
            apply = requested[index] & inside[0]
            updated = DSMCParticleState(
                current.position.at[slot].set(
                    jnp.where(apply, position, current.position[slot])
                ),
                current.velocity.at[slot].set(
                    jnp.where(apply, velocity, current.velocity[slot])
                ),
                current.species_index.at[slot].set(
                    jnp.where(apply, species_index, current.species_index[slot])
                ),
                current.rotational_energy.at[slot].set(
                    jnp.where(apply, rotational, current.rotational_energy[slot])
                ),
                current.vibrational_energy.at[slot].set(
                    jnp.where(apply, 0.0, current.vibrational_energy[slot])
                ),
                current.statistical_weight.at[slot].set(
                    jnp.where(apply, weight, current.statistical_weight[slot])
                ),
                current.cell_id.at[slot].set(
                    jnp.where(apply, cell_id[0], current.cell_id[slot])
                ),
                current.active.at[slot].set(jnp.where(apply, True, current.active[slot])),
                current.incarnation.at[slot].set(
                    jnp.where(
                        apply,
                        current.incarnation[slot] + jnp.asarray(1, dtype=jnp.int32),
                        current.incarnation[slot],
                    )
                ),
            )
            represented_mass = weight * molecular_mass
            represented_momentum = represented_mass * velocity
            represented_energy = weight * (
                0.5 * molecular_mass * jnp.sum(velocity**2) + rotational
            )
            return (
                updated,
                mass_total + jnp.where(apply, represented_mass, 0.0),
                momentum_total + jnp.where(apply, represented_momentum, 0.0),
                energy_total + jnp.where(apply, represented_energy, 0.0),
            )

        updated, mass_total, momentum_total, energy_total = jax.lax.fori_loop(
            0,
            self.request_capacity,
            body,
            (
                particles,
                jnp.asarray(0.0, dtype=dtype),
                jnp.zeros((self.face.cells.dimension,), dtype=dtype),
                jnp.asarray(0.0, dtype=dtype),
            ),
        )
        candidate_state = DSMCReservoirState(
            jnp.where(
                header.globally_eligible,
                expected - counts.astype(dtype),
                state.fractional_remainder,
            ),
            state.epoch,
            self.plan_id,
        )
        return DSMCReservoirResult(
            updated,
            candidate_state,
            jnp.sum(requested),
            mass_total,
            momentum_total,
            energy_total,
            header,
            self.plan_id,
        )


__all__ = [
    "DSMCBoundaryFacePlan",
    "DSMCBoundaryReason",
    "DSMCReservoirFacePlan",
    "DSMCReservoirResult",
    "DSMCReservoirState",
    "DSMCSurfaceBoundaryPlan",
]
