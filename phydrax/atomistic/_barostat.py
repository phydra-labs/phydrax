#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ._dynamics import (
    AtomisticDynamicsState,
    AtomisticEnergyLedgerState,
    AtomisticKinematics,
    PreparedAtomisticDynamics,
)


class IsotropicMonteCarloBarostatPlan(StrictModule, NonTrainableState):
    pressure: float = eqx.field(static=True)
    temperature: float = eqx.field(static=True)
    maximum_log_volume_change: float = eqx.field(static=True)
    scaled_entity_count: int | None = eqx.field(static=True)
    realization_id: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        pressure: float,
        temperature: float,
        maximum_log_volume_change: float,
        /,
        *,
        scaled_entity_count: int | None = None,
        realization_id: int = 0,
    ):
        pressure_ = float(pressure)
        temperature_ = float(temperature)
        maximum = float(maximum_log_volume_change)
        entities = None if scaled_entity_count is None else int(scaled_entity_count)
        realization = int(realization_id)
        if (
            not math.isfinite(pressure_)
            or not math.isfinite(temperature_)
            or temperature_ <= 0.0
            or not math.isfinite(maximum)
            or maximum <= 0.0
            or (entities is not None and entities <= 0)
            or realization < 0
        ):
            raise ValueError("Monte Carlo barostat parameters are invalid.")
        self.pressure = pressure_
        self.temperature = temperature_
        self.maximum_log_volume_change = maximum
        self.scaled_entity_count = entities
        self.realization_id = realization
        self.plan_id = canonical_fingerprint(
            {
                "kind": "isotropic-monte-carlo-barostat",
                "pressure": pressure_,
                "temperature": temperature_,
                "maximum_log_volume_change": maximum,
                "scaled_entity_count": entities,
                "realization_id": realization,
            }
        )


class AtomisticBarostatEvaluation(StrictModule):
    candidate_state: AtomisticDynamicsState
    accepted_state: AtomisticDynamicsState
    accepted: Array
    log_acceptance_probability: Array
    volume_before: Array
    volume_after: Array
    energy_before: Array
    energy_after: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def _volume(vectors: Array, /) -> Array:
    return jnp.abs(jnp.sum(vectors[0] * jnp.cross(vectors[1], vectors[2])))


def apply_isotropic_monte_carlo_barostat(
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    plan: IsotropicMonteCarloBarostatPlan,
    move_index: ArrayLike,
    /,
) -> AtomisticBarostatEvaluation:
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if not isinstance(plan, IsotropicMonteCarloBarostatPlan):
        raise TypeError("plan must be IsotropicMonteCarloBarostatPlan.")
    if dynamics.system.cell is None or not dynamics.system.cell.fully_periodic:
        raise ValueError("Isotropic barostat requires a fully periodic cell.")
    if dynamics.potential.plan.requirements.directed_graph:
        raise ValueError("Dynamic-cell barostat does not support learned graph terms.")
    if dynamics.neighborhood.backend != "dense_pairs":
        raise ValueError("Dynamic-cell barostat currently requires dense pair authority.")
    if dynamics.constraints is not None:
        indices = np.asarray(dynamics.system.topology.constraint_indices)
        molecules = np.asarray(dynamics.system.plan.molecule_ids)
        if indices.size and np.any(molecules[indices[:, 0]] != molecules[indices[:, 1]]):
            raise ValueError(
                "Barostat constraints must remain within one scaled molecule."
            )
    key = jr.wrap_key_data(state.random_key)
    key = jr.fold_in(key, jnp.asarray(plan.realization_id, dtype=jnp.uint32))
    key = jr.fold_in(key, jnp.asarray(move_index, dtype=jnp.uint32))
    proposal_key = jr.fold_in(key, jnp.uint32(0))
    acceptance_key = jr.fold_in(key, jnp.uint32(1))
    dtype = state.kinematics.positions.dtype
    log_volume_change = jr.uniform(
        proposal_key,
        (),
        minval=-plan.maximum_log_volume_change,
        maxval=plan.maximum_log_volume_change,
        dtype=dtype,
    )
    linear_scale = jnp.exp(log_volume_change / 3.0)
    old_vectors = state.cell_vectors
    new_vectors = old_vectors * linear_scale
    cell = dynamics.system.cell
    old_unwrapped = dynamics._unwrapped(state.kinematics, old_vectors)
    proposed_unwrapped = old_unwrapped
    masses = dynamics.system.plan.masses.astype(dtype)
    for molecule_label in dynamics.system.molecule_labels:
        mask = dynamics.system.active_mask & (
            dynamics.system.plan.molecule_ids == molecule_label
        )
        weight = jnp.where(mask, masses, 0.0)
        center = jnp.sum(weight[:, None] * old_unwrapped, axis=0) / jnp.sum(weight)
        center_fractional = cell.fractional_with_vectors(center, old_vectors)
        proposed_center = cell.cartesian_with_vectors(center_fractional, new_vectors)
        proposed_unwrapped = jnp.where(
            mask[:, None],
            old_unwrapped + (proposed_center - center),
            proposed_unwrapped,
        )
    proposed_positions, proposed_images = cell.wrap_with_vectors(
        proposed_unwrapped, new_vectors
    )
    fractional = cell.fractional_with_vectors(proposed_positions, new_vectors)
    neighborhood, cache = dynamics._build_neighborhood(
        proposed_positions, None, new_vectors
    )
    evaluation = dynamics.potential.evaluate(
        proposed_positions,
        neighborhood,
        unwrapped_positions=proposed_unwrapped,
        species=state.species,
        cell=cell,
        fractional_positions=fractional,
        cell_vectors=new_vectors,
    )
    volume_before = _volume(old_vectors)
    volume_after = _volume(new_vectors)
    entity_count = (
        len(dynamics.system.molecule_labels)
        if plan.scaled_entity_count is None
        else plan.scaled_entity_count
    )
    beta = 1.0 / (dynamics.system.plan.units.boltzmann_constant * plan.temperature)
    energy_change = evaluation.energy - state.force.potential_energy
    pressure_work = plan.pressure * (volume_after - volume_before)
    log_acceptance = -beta * (energy_change + pressure_work) + entity_count * jnp.log(
        volume_after / volume_before
    )
    log_uniform = jnp.log(jr.uniform(acceptance_key, (), dtype=dtype))
    cutoff = dynamics.potential.plan.requirements.cutoff
    image_valid = (
        jnp.asarray(True)
        if cutoff is None
        else linear_scale * cell.unique_image_radius > cutoff
    )
    accepted = (
        evaluation.successful
        & neighborhood.successful
        & image_valid
        & jnp.isfinite(log_acceptance)
        & (log_uniform < jnp.minimum(log_acceptance, 0.0))
    )
    force = dynamics._force_state(evaluation, cache, state.step_index)
    kinetic = dynamics.kinetic_energy(state.kinematics.momenta)
    total = kinetic + evaluation.energy
    candidate_energy = AtomisticEnergyLedgerState(
        initial_kinetic_energy=state.energy.initial_kinetic_energy,
        initial_potential_energy=state.energy.initial_potential_energy,
        kinetic_energy=kinetic,
        potential_energy=evaluation.energy,
        total_energy=total,
        thermostat_heat=state.energy.thermostat_heat,
        barostat_work=state.energy.barostat_work + pressure_work,
        external_work=state.energy.external_work,
        constraint_work=state.energy.constraint_work,
        cumulative_balance_residual=state.energy.cumulative_balance_residual,
        last_relative_energy_change=jnp.abs(energy_change)
        / jnp.maximum(jnp.abs(state.energy.total_energy), 1.0e-30),
        accepted_steps=state.energy.accepted_steps,
    )
    candidate = AtomisticDynamicsState(
        time=state.time,
        step_index=state.step_index,
        kinematics=AtomisticKinematics(
            proposed_positions,
            state.kinematics.momenta,
            proposed_images,
        ),
        species=state.species,
        cell_vectors=new_vectors,
        neighborhood=neighborhood,
        neighborhood_cache=cache,
        force=force,
        constraint_lagrange=state.constraint_lagrange,
        constraint_position_residual=state.constraint_position_residual,
        constraint_velocity_residual=state.constraint_velocity_residual,
        thermostat_state=state.thermostat_state,
        barostat_state=state.barostat_state,
        random_key=state.random_key,
        energy=candidate_energy,
        last_status=state.last_status,
        last_rejection_reasons=state.last_rejection_reasons,
        prepared_dynamics_id=state.prepared_dynamics_id,
    )
    successor = tree_where(accepted, candidate, state)
    return AtomisticBarostatEvaluation(
        candidate,
        successor,
        accepted,
        log_acceptance,
        volume_before,
        volume_after,
        state.force.potential_energy,
        evaluation.energy,
        evaluation.successful & neighborhood.successful & image_valid,
        plan.plan_id,
    )


__all__ = [
    "AtomisticBarostatEvaluation",
    "IsotropicMonteCarloBarostatPlan",
    "apply_isotropic_monte_carlo_barostat",
]
