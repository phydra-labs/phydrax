#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike
from opt_einsum import contract

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._pairwise import particle_pair_geometry
from ._periodic_cell import ParticleCell


class DEMPeriodicCellState(StrictModule):
    vectors: Array
    cumulative_work: Array
    last_work: Array
    successful: Array


class DEMBulkStress(StrictModule):
    contact_stress: Array
    kinetic_stress: Array
    total_stress: Array
    pressure: Array
    volume: Array
    successful: Array


class DEMPeriodicCellUpdate(StrictModule):
    position: Array
    velocity: Array
    state: DEMPeriodicCellState
    strain_rate: Array
    strain_increment: Array
    work: Array
    conditioning_margin: Array
    unique_image_margin: Array
    successful: Array


class DEMPeriodicCellControlPlan(StrictModule, NonTrainableState):
    prescribed_strain_rate: Array
    strain_rate_mask: Array
    target_stress: Array
    stress_mask: Array
    stress_compliance: Array
    maximum_strain_increment: float = eqx.field(static=True)
    maximum_condition_number: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        prescribed_strain_rate: ArrayLike,
        /,
        *,
        strain_rate_mask: ArrayLike | None = None,
        target_stress: ArrayLike | None = None,
        stress_mask: ArrayLike | None = None,
        stress_compliance: ArrayLike | float = 0.0,
        maximum_strain_increment: float = 0.02,
        maximum_condition_number: float = 1.5,
        plan_id: str | None = None,
    ):
        rate = np.asarray(prescribed_strain_rate, dtype=float)
        if (
            rate.ndim != 2
            or rate.shape[0] != rate.shape[1]
            or rate.shape[0] not in (2, 3)
        ):
            raise ValueError("prescribed_strain_rate must be a 2x2 or 3x3 tensor.")
        if np.any(~np.isfinite(rate)):
            raise ValueError("prescribed_strain_rate must be finite.")
        rate_mask = (
            np.ones(rate.shape, dtype=bool)
            if strain_rate_mask is None
            else np.asarray(strain_rate_mask, dtype=bool)
        )
        controlled_mask = (
            np.zeros(rate.shape, dtype=bool)
            if stress_mask is None
            else np.asarray(stress_mask, dtype=bool)
        )
        target = (
            np.zeros(rate.shape)
            if target_stress is None
            else np.asarray(target_stress, dtype=float)
        )
        compliance = np.asarray(stress_compliance, dtype=float)
        if compliance.ndim == 0:
            compliance = np.full(rate.shape, float(compliance))
        if (
            rate_mask.shape != rate.shape
            or controlled_mask.shape != rate.shape
            or target.shape != rate.shape
            or compliance.shape != rate.shape
        ):
            raise ValueError("Cell-control tensors and masks must have matching shapes.")
        if np.any(rate_mask & controlled_mask):
            raise ValueError("Strain-rate and stress control masks must be disjoint.")
        if not np.any(rate_mask | controlled_mask):
            raise ValueError("At least one cell tensor component must be controlled.")
        if not np.array_equal(controlled_mask, controlled_mask.T):
            raise ValueError("stress_mask must be symmetric.")
        if not np.allclose(target, target.T):
            raise ValueError("target_stress must be symmetric.")
        if np.any(~np.isfinite(target)) or np.any(~np.isfinite(compliance)):
            raise ValueError("Stress targets and compliance must be finite.")
        if np.any(controlled_mask & (compliance <= 0.0)):
            raise ValueError("Stress-controlled compliance must be positive.")
        maximum_increment = float(maximum_strain_increment)
        maximum_condition = float(maximum_condition_number)
        if (
            not np.isfinite(maximum_increment)
            or maximum_increment <= 0.0
            or not np.isfinite(maximum_condition)
            or maximum_condition <= 1.0
        ):
            raise ValueError("Cell increment or conditioning limit is invalid.")
        generated = canonical_fingerprint(
            {
                "kind": "dem-periodic-cell-control",
                "rate": array_tree_fingerprint(rate),
                "rate_mask": array_tree_fingerprint(rate_mask),
                "target": array_tree_fingerprint(target),
                "stress_mask": array_tree_fingerprint(controlled_mask),
                "compliance": array_tree_fingerprint(compliance),
                "maximum_increment": maximum_increment,
                "maximum_condition": maximum_condition,
            }
        )
        identifier = generated if plan_id is None else str(plan_id)
        if not identifier:
            raise ValueError("plan_id must be nonempty.")
        self.prescribed_strain_rate = jnp.asarray(rate)
        self.strain_rate_mask = jnp.asarray(rate_mask)
        self.target_stress = jnp.asarray(target)
        self.stress_mask = jnp.asarray(controlled_mask)
        self.stress_compliance = jnp.asarray(compliance)
        self.maximum_strain_increment = maximum_increment
        self.maximum_condition_number = maximum_condition
        self.plan_id = identifier

    @property
    def ambient_dimension(self) -> int:
        return int(self.prescribed_strain_rate.shape[0])

    def initialize(self, cell: ParticleCell, dtype, /) -> DEMPeriodicCellState:
        if not isinstance(cell, ParticleCell) or not cell.fully_periodic:
            raise ValueError(
                "Periodic DEM cell control requires a fully periodic ParticleCell."
            )
        if cell.ambient_dimension != self.ambient_dimension:
            raise ValueError("Cell and control dimensions differ.")
        zero = jnp.zeros((), dtype=dtype)
        return DEMPeriodicCellState(
            cell.vectors.astype(dtype), zero, zero, jnp.asarray(True)
        )

    def update(
        self,
        cell: ParticleCell,
        state: DEMPeriodicCellState,
        position: ArrayLike,
        velocity: ArrayLike,
        observed_stress: ArrayLike,
        step_size: ArrayLike,
        maximum_interaction_radius: float,
        /,
    ) -> DEMPeriodicCellUpdate:
        if not isinstance(cell, ParticleCell):
            raise TypeError("cell must be a ParticleCell.")
        if not isinstance(state, DEMPeriodicCellState):
            raise TypeError("state must be DEMPeriodicCellState.")
        position_ = jnp.asarray(position)
        velocity_ = jnp.asarray(velocity, dtype=position_.dtype)
        observed = jnp.asarray(observed_stress, dtype=position_.dtype)
        dt = jnp.asarray(step_size, dtype=position_.dtype)
        if position_.ndim != 2 or position_.shape != velocity_.shape:
            raise ValueError(
                "Periodic-cell position and velocity must be matching matrices."
            )
        expected = (self.ambient_dimension, self.ambient_dimension)
        if observed.shape != expected or state.vectors.shape != expected:
            raise ValueError("Periodic-cell stress and vectors have invalid shapes.")
        prescribed = jnp.where(
            self.strain_rate_mask,
            self.prescribed_strain_rate.astype(position_.dtype),
            0.0,
        )
        feedback = self.stress_compliance.astype(position_.dtype) * (
            self.target_stress.astype(position_.dtype) - observed
        )
        feedback = 0.5 * (feedback + feedback.T)
        strain_rate = prescribed + jnp.where(self.stress_mask, feedback, 0.0)
        increment = dt * strain_rate
        candidate_vectors = state.vectors + contract(
            "ij,kj->ki", increment, state.vectors
        )
        fractional = cell.fractional_with_vectors(position_, state.vectors)
        candidate_position = cell.cartesian_with_vectors(fractional, candidate_vectors)
        affine_velocity = contract(
            "ni,ij->nj", fractional, (candidate_vectors - state.vectors) / dt
        )
        candidate_velocity = velocity_ + affine_velocity
        wrapped_position, _ = cell.wrap_with_vectors(
            candidate_position, candidate_vectors
        )
        determinant = _cell_determinant(candidate_vectors)
        inverse = cell.inverse_for_vectors(candidate_vectors)
        condition_bound = (
            jnp.sqrt(jnp.sum(candidate_vectors**2))
            * jnp.sqrt(jnp.sum(inverse**2))
            / self.ambient_dimension
        )
        shifted = contract(
            "si,ij->sj", cell.image_shifts.astype(position_.dtype), candidate_vectors
        )
        nonzero_shift = jnp.any(cell.image_shifts != 0, axis=-1)
        shortest = jnp.min(
            jnp.where(
                nonzero_shift,
                jnp.sqrt(jnp.sum(shifted**2, axis=-1)),
                jnp.inf,
            )
        )
        unique_margin = 0.5 * shortest - jnp.asarray(
            maximum_interaction_radius, dtype=position_.dtype
        )
        conditioning_margin = self.maximum_condition_number - condition_bound
        work = (
            jnp.abs(_cell_determinant(state.vectors))
            * dt
            * jnp.sum(observed * strain_rate)
        )
        increment_norm = jnp.max(jnp.abs(increment))
        successful = (
            state.successful
            & (dt > 0.0)
            & jnp.all(jnp.isfinite(candidate_vectors))
            & jnp.all(jnp.isfinite(candidate_position))
            & jnp.all(jnp.isfinite(candidate_velocity))
            & jnp.isfinite(determinant)
            & (determinant > 0.0)
            & (increment_norm <= self.maximum_strain_increment)
            & (conditioning_margin >= 0.0)
            & (unique_margin > 0.0)
        )
        candidate_state = DEMPeriodicCellState(
            candidate_vectors,
            state.cumulative_work + work,
            work,
            successful,
        )
        return DEMPeriodicCellUpdate(
            wrapped_position,
            candidate_velocity,
            candidate_state,
            strain_rate,
            increment,
            work,
            conditioning_margin,
            unique_margin,
            successful,
        )


def dem_bulk_stress(dynamics, state, evaluation, /) -> DEMBulkStress:
    cell_state = state.periodic_cell
    if cell_state is None:
        raise ValueError("DEM state has no deforming periodic cell.")
    cell = dynamics.neighborhood.box
    if not isinstance(cell, ParticleCell):
        raise TypeError("DEM bulk stress requires a ParticleCell neighborhood.")
    pairs = evaluation.neighborhood.pair_relation
    geometry = particle_pair_geometry(
        state.kinematics.position,
        pairs,
        box=cell,
        cell_vectors=cell_state.vectors,
    )
    pair_active = evaluation.particle_contact.active & pairs.valid
    pair_force = jnp.where(
        pair_active[:, None], evaluation.particle_contact.pair_force, 0.0
    )
    displacement = jnp.where(pair_active[:, None], geometry.displacement, 0.0)
    contact_virial = -jnp.sum(contract("pi,pj->pij", pair_force, displacement), axis=0)
    active = state.body_properties.active
    masses = jnp.where(active, state.body_properties.masses, 0.0)
    total_mass = jnp.sum(masses)
    mean_velocity = jnp.sum(
        masses[:, None] * state.kinematics.velocity, axis=0
    ) / jnp.where(total_mass > 0.0, total_mass, 1.0)
    peculiar = jnp.where(active[:, None], state.kinematics.velocity - mean_velocity, 0.0)
    kinetic_virial = -jnp.sum(
        masses[:, None, None] * contract("pi,pj->pij", peculiar, peculiar),
        axis=0,
    )
    volume = jnp.abs(_cell_determinant(cell_state.vectors))
    contact_stress = contact_virial / volume
    kinetic_stress = kinetic_virial / volume
    total_stress = contact_stress + kinetic_stress
    pressure = -jnp.trace(total_stress) / total_stress.shape[0]
    successful = (
        evaluation.successful
        & cell_state.successful
        & jnp.isfinite(volume)
        & (volume > 0.0)
        & jnp.all(jnp.isfinite(total_stress))
    )
    return DEMBulkStress(
        contact_stress,
        kinetic_stress,
        total_stress,
        pressure,
        volume,
        successful,
    )


def _cell_determinant(vectors: Array, /) -> Array:
    if vectors.shape == (2, 2):
        return vectors[0, 0] * vectors[1, 1] - vectors[0, 1] * vectors[1, 0]
    if vectors.shape == (3, 3):
        return jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2]))
    raise ValueError("DEM periodic cells require dimension two or three.")


__all__ = [
    "DEMBulkStress",
    "DEMPeriodicCellControlPlan",
    "DEMPeriodicCellState",
    "DEMPeriodicCellUpdate",
    "dem_bulk_stress",
]
