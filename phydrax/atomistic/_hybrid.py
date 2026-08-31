#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import abc
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import AbstractAttribute, StrictModule
from .._trainable import NonTrainableState
from .._tree_math import tree_where
from ..discretization import ParticleNeighborhoodState
from ._dynamics import (
    AtomisticDynamicsState,
    AtomisticEnergyLedgerState,
    AtomisticKinematics,
    PreparedAtomisticDynamics,
)
from ._potential import AtomisticPotentialCapabilities, AtomisticPotentialRequirements
from ._potential_program import (
    AbstractAtomisticEnergyTerm,
    AbstractPreparedAtomisticEnergyTerm,
    AtomisticPotentialContext,
    AtomisticTermEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._rollout import AtomisticRolloutPlan, AtomisticRolloutResult
from ._system import PreparedAtomisticSystem


class AlchemicalScaledPotential(AbstractAtomisticEnergyTerm):
    term: AbstractAtomisticEnergyTerm
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        term: AbstractAtomisticEnergyTerm,
        /,
        *,
        name: str | None = None,
        force_group: int | None = None,
    ):
        if not isinstance(term, AbstractAtomisticEnergyTerm):
            raise TypeError("term must be AbstractAtomisticEnergyTerm.")
        identifier = f"alchemical-{term.name}" if name is None else str(name).strip()
        group = term.force_group if force_group is None else int(force_group)
        if not identifier or group < 0:
            raise ValueError("Alchemical term name or force group is invalid.")
        self.term = term
        self.name = identifier
        self.force_group = group
        self.capabilities = term.capabilities
        self.requirements = term.requirements
        self.term_id = canonical_fingerprint(
            {
                "kind": "alchemical-scaled-potential",
                "term": term.term_id,
                "name": identifier,
                "force_group": group,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedAlchemicalScaledPotential":
        return PreparedAlchemicalScaledPotential(self, self.term.prepare(system))


class PreparedAlchemicalScaledPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: AlchemicalScaledPotential
    term: AbstractPreparedAtomisticEnergyTerm
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: AlchemicalScaledPotential,
        term: AbstractPreparedAtomisticEnergyTerm,
        /,
    ):
        self.plan = plan
        self.term = term
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-alchemical-potential",
                "plan": plan.term_id,
                "term": term.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        value = self.term.energy(context)
        scale = context.alchemical_lambda.astype(value.energy.dtype)
        return AtomisticTermEvaluation(
            scale * value.energy,
            scale * value.atom_energy,
            value.successful,
        )


class ForceGroupEvaluation(StrictModule):
    energy: Array
    forces: Array
    term_energies: Array
    successful: Array
    group: int = eqx.field(static=True)


def evaluate_force_group(
    potential: PreparedAtomisticPotentialProgram,
    group: int,
    positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState,
    /,
    **context_kwargs: Any,
) -> ForceGroupEvaluation:
    selected = tuple(
        index
        for index, term in enumerate(potential.terms)
        if term.force_group == int(group)
    )
    if not selected:
        raise ValueError(f"Potential program has no force group {group}.")

    def energy_closure(value):
        context = potential.context(value, neighborhood, **context_kwargs)
        evaluations = tuple(potential.terms[index].energy(context) for index in selected)
        terms = jnp.stack(tuple(item.energy for item in evaluations))
        coefficients = potential.plan.coefficients[jnp.asarray(selected)]
        energy = jnp.sum(coefficients.astype(terms.dtype) * terms)
        successful = context.neighborhood_successful & jnp.all(
            jnp.stack(tuple(item.successful for item in evaluations))
        )
        return energy, (terms, successful)

    (energy, auxiliary), gradient = jax.value_and_grad(energy_closure, has_aux=True)(
        jnp.asarray(positions)
    )
    terms, successful = auxiliary
    finite = jnp.isfinite(energy) & jnp.all(jnp.isfinite(gradient))
    return ForceGroupEvaluation(energy, -gradient, terms, successful & finite, int(group))


class RegionMaskedPotential(AbstractAtomisticEnergyTerm):
    term: AbstractAtomisticEnergyTerm
    region_id: int = eqx.field(static=True)
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        term: AbstractAtomisticEnergyTerm,
        region_id: int,
        /,
        *,
        name: str | None = None,
    ):
        if not isinstance(term, AbstractAtomisticEnergyTerm):
            raise TypeError("term must be AbstractAtomisticEnergyTerm.")
        if not term.capabilities.local_energy:
            raise ValueError("Region masking requires a local atom-energy decomposition.")
        identifier = (
            f"region-{int(region_id)}-{term.name}" if name is None else str(name).strip()
        )
        if not identifier:
            raise ValueError("Masked potential name must be non-empty.")
        self.term = term
        self.region_id = int(region_id)
        self.name = identifier
        self.force_group = term.force_group
        self.capabilities = term.capabilities
        self.requirements = term.requirements
        self.term_id = canonical_fingerprint(
            {
                "kind": "region-masked-potential",
                "term": term.term_id,
                "region_id": int(region_id),
                "name": identifier,
            }
        )

    def prepare(
        self, system: PreparedAtomisticSystem, /
    ) -> "PreparedRegionMaskedPotential":
        mask = system.plan.region_ids == self.region_id
        if not bool(jnp.any(mask & system.active_mask)):
            raise ValueError("Region mask selects no active atoms.")
        return PreparedRegionMaskedPotential(self, self.term.prepare(system), mask)


class PreparedRegionMaskedPotential(AbstractPreparedAtomisticEnergyTerm):
    plan: RegionMaskedPotential
    term: AbstractPreparedAtomisticEnergyTerm
    mask: Array
    name: str = eqx.field(static=True)
    force_group: int = eqx.field(static=True)
    term_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    capabilities: AtomisticPotentialCapabilities
    requirements: AtomisticPotentialRequirements

    def __init__(
        self,
        plan: RegionMaskedPotential,
        term: AbstractPreparedAtomisticEnergyTerm,
        mask: Array,
        /,
    ):
        self.plan = plan
        self.term = term
        self.mask = jnp.asarray(mask, dtype=bool)
        self.name = plan.name
        self.force_group = plan.force_group
        self.term_id = plan.term_id
        self.capabilities = plan.capabilities
        self.requirements = plan.requirements
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-region-masked-potential",
                "plan": plan.term_id,
                "term": term.prepared_id,
            }
        )

    def energy(self, context: AtomisticPotentialContext, /) -> AtomisticTermEvaluation:
        value = self.term.energy(context)
        atom_energy = jnp.where(self.mask, value.atom_energy, 0.0)
        return AtomisticTermEvaluation(
            jnp.sum(atom_energy), atom_energy, value.successful
        )


class RESPAPlan(StrictModule, NonTrainableState):
    outer_step_size: float = eqx.field(static=True)
    inner_steps: int = eqx.field(static=True)
    fast_group: int = eqx.field(static=True)
    slow_group: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        outer_step_size: float,
        inner_steps: int,
        /,
        *,
        fast_group: int = 0,
        slow_group: int = 1,
    ):
        step = float(outer_step_size)
        inner = int(inner_steps)
        fast = int(fast_group)
        slow = int(slow_group)
        if (
            not np.isfinite(step)
            or step <= 0.0
            or inner <= 0
            or fast < 0
            or slow < 0
            or fast == slow
        ):
            raise ValueError("RESPA step, groups, or inner_steps are invalid.")
        self.outer_step_size = step
        self.inner_steps = inner
        self.fast_group = fast
        self.slow_group = slow
        self.plan_id = canonical_fingerprint(
            {
                "kind": "atomistic-respa",
                "outer_step_size": step,
                "inner_steps": inner,
                "fast_group": fast,
                "slow_group": slow,
            }
        )


class RESPAStepEvaluation(StrictModule):
    state: AtomisticDynamicsState
    successful: Array
    fast_work: Array
    slow_work: Array
    plan_id: str = eqx.field(static=True)


def respa_step(
    dynamics: PreparedAtomisticDynamics,
    state: AtomisticDynamicsState,
    plan: RESPAPlan,
    /,
) -> RESPAStepEvaluation:
    if not isinstance(dynamics, PreparedAtomisticDynamics):
        raise TypeError("dynamics must be PreparedAtomisticDynamics.")
    if not isinstance(state, AtomisticDynamicsState):
        raise TypeError("state must be AtomisticDynamicsState.")
    if not isinstance(plan, RESPAPlan):
        raise TypeError("plan must be RESPAPlan.")
    if dynamics.constraints is not None:
        raise ValueError("RESPA constraints require a separately qualified splitting.")
    outer = jnp.asarray(plan.outer_step_size, dtype=state.kinematics.positions.dtype)
    inner = outer / plan.inner_steps
    force_scale = dynamics.system.plan.units.force_to_momentum_rate
    inverse_mass = dynamics.system.inverse_masses[:, None]
    mobile = dynamics.system.mobile_mask[:, None]
    position = state.kinematics.positions
    momentum = state.kinematics.momenta
    images = state.kinematics.image_counts
    neighborhood = state.neighborhood
    unwrapped = dynamics._unwrapped(state.kinematics, state.cell_vectors)

    def kwargs(current_unwrapped):
        values: dict[str, Any] = {
            "unwrapped_positions": current_unwrapped,
            "species": state.species,
            "cell": dynamics.system.cell,
        }
        if (
            dynamics.system.cell is not None
            and not dynamics.potential.plan.requirements.directed_graph
        ):
            values["fractional_positions"] = dynamics.system.cell.fractional_with_vectors(
                position, state.cell_vectors
            )
            values["cell_vectors"] = state.cell_vectors
        return values

    slow = evaluate_force_group(
        dynamics.potential,
        plan.slow_group,
        position,
        neighborhood,
        **kwargs(unwrapped),
    )
    momentum = momentum + 0.5 * outer * force_scale * slow.forces
    momentum = jnp.where(mobile, momentum, 0.0)
    fast_work = jnp.zeros((), dtype=jnp.int32)
    successful = slow.successful
    cache = state.neighborhood_cache
    for _ in range(plan.inner_steps):
        fast = evaluate_force_group(
            dynamics.potential,
            plan.fast_group,
            position,
            neighborhood,
            **kwargs(unwrapped),
        )
        momentum = momentum + 0.5 * inner * force_scale * fast.forces
        unwrapped = unwrapped + inner * momentum * inverse_mass
        if dynamics.system.cell is None:
            position = unwrapped
        else:
            position, images = dynamics.system.cell.wrap_with_vectors(
                unwrapped, state.cell_vectors
            )
        neighborhood, cache = dynamics._build_neighborhood(
            position, cache, state.cell_vectors
        )
        fast = evaluate_force_group(
            dynamics.potential,
            plan.fast_group,
            position,
            neighborhood,
            **kwargs(unwrapped),
        )
        momentum = momentum + 0.5 * inner * force_scale * fast.forces
        momentum = jnp.where(mobile, momentum, 0.0)
        successful = successful & fast.successful & neighborhood.successful
        fast_work = fast_work + neighborhood.candidate_pair_count
    slow = evaluate_force_group(
        dynamics.potential,
        plan.slow_group,
        position,
        neighborhood,
        **kwargs(unwrapped),
    )
    momentum = momentum + 0.5 * outer * force_scale * slow.forces
    momentum = jnp.where(mobile, momentum, 0.0)
    total = dynamics.potential.evaluate(
        position,
        neighborhood,
        **kwargs(unwrapped),
    )
    kinetic = dynamics.kinetic_energy(momentum)
    total_energy = kinetic + total.energy
    delta = total_energy - state.energy.total_energy
    ledger = AtomisticEnergyLedgerState(
        state.energy.initial_kinetic_energy,
        state.energy.initial_potential_energy,
        kinetic,
        total.energy,
        total_energy,
        state.energy.thermostat_heat,
        state.energy.barostat_work,
        state.energy.external_work,
        state.energy.constraint_work,
        state.energy.cumulative_balance_residual + delta,
        jnp.abs(delta) / jnp.maximum(jnp.abs(state.energy.total_energy), 1.0e-30),
        state.energy.accepted_steps + 1,
    )
    force = dynamics._force_state(total, cache, state.step_index + 1)
    candidate = AtomisticDynamicsState(
        state.time + outer,
        state.step_index + 1,
        AtomisticKinematics(position, momentum, images),
        state.species,
        state.cell_vectors,
        neighborhood,
        cache,
        force,
        state.constraint_lagrange,
        state.constraint_position_residual,
        state.constraint_velocity_residual,
        state.thermostat_state,
        state.barostat_state,
        state.random_key,
        ledger,
        state.last_status,
        state.last_rejection_reasons,
        state.prepared_dynamics_id,
    )
    successful = successful & slow.successful & total.successful
    return RESPAStepEvaluation(
        tree_where(successful, candidate, state),
        successful,
        fast_work,
        neighborhood.candidate_pair_count,
        plan.plan_id,
    )


class SubtractivePotentialEvaluation(StrictModule):
    energy: Array
    full_forces: Array
    region_forces: Array
    successful: Array


def evaluate_subtractive_potential(
    low_all: PreparedAtomisticPotentialProgram,
    low_region: PreparedAtomisticPotentialProgram,
    high_region: PreparedAtomisticPotentialProgram,
    full_positions: ArrayLike,
    region_positions: ArrayLike,
    full_neighborhood: ParticleNeighborhoodState,
    region_neighborhood: ParticleNeighborhoodState,
    region_indices: ArrayLike,
    /,
) -> SubtractivePotentialEvaluation:
    indices = jnp.asarray(region_indices, dtype=jnp.int32)
    low_all_evaluation = low_all.evaluate(full_positions, full_neighborhood)
    low_region_evaluation = low_region.evaluate(region_positions, region_neighborhood)
    high_region_evaluation = high_region.evaluate(region_positions, region_neighborhood)
    region_force = high_region_evaluation.forces - low_region_evaluation.forces
    full_force = low_all_evaluation.forces.at[indices].add(region_force)
    energy = (
        low_all_evaluation.energy
        + high_region_evaluation.energy
        - low_region_evaluation.energy
    )
    successful = (
        low_all_evaluation.successful
        & low_region_evaluation.successful
        & high_region_evaluation.successful
        & jnp.all(jnp.isfinite(full_force))
    )
    return SubtractivePotentialEvaluation(energy, full_force, region_force, successful)


def local_species_energy_delta(
    potential: PreparedAtomisticPotentialProgram,
    positions: ArrayLike,
    neighborhood: ParticleNeighborhoodState,
    species: ArrayLike,
    particle_index: ArrayLike,
    proposed_species: ArrayLike,
    /,
) -> Array:
    if not potential.plan.capabilities.dynamic_species:
        raise ValueError("Potential program does not support dynamic species.")
    current = jnp.asarray(species, dtype=jnp.int32)
    index = jnp.asarray(particle_index, dtype=jnp.int32).reshape(())
    proposed = current.at[index].set(jnp.asarray(proposed_species, dtype=jnp.int32))
    current_energy = potential.energy(positions, neighborhood, species=current)[0]
    proposed_energy = potential.energy(positions, neighborhood, species=proposed)[0]
    return proposed_energy - current_energy


class ExternalAtomisticEvaluation(StrictModule):
    energy: Array
    forces: Array
    stress: Array | None
    successful: Array
    provider_id: str = eqx.field(static=True)


class AbstractExternalAtomisticProvider(StrictModule, NonTrainableState):
    provider_id: AbstractAttribute[str]
    conservative: AbstractAttribute[bool]
    differentiable: AbstractAttribute[bool]

    @abc.abstractmethod
    def evaluate(
        self,
        system: PreparedAtomisticSystem,
        positions: ArrayLike,
        cell_vectors: ArrayLike | None,
        /,
    ) -> ExternalAtomisticEvaluation:
        raise NotImplementedError


class AtomisticSegmentedResult(StrictModule):
    final_state: AtomisticDynamicsState
    segments: tuple[AtomisticRolloutResult, ...]
    successful: Array
    protocol_id: str = eqx.field(static=True)


def run_atomistic_segments(
    rollout: AtomisticRolloutPlan,
    initial_state: AtomisticDynamicsState,
    segment_count: int,
    /,
) -> AtomisticSegmentedResult:
    if not isinstance(rollout, AtomisticRolloutPlan):
        raise TypeError("rollout must be AtomisticRolloutPlan.")
    count = int(segment_count)
    if count <= 0:
        raise ValueError("segment_count must be positive.")
    state = initial_state
    segments: list[AtomisticRolloutResult] = []
    successful = jnp.asarray(True)
    for _ in range(count):
        result = rollout.rollout(state)
        segments.append(result)
        state = result.final_state
        successful = successful & result.successful
        if not bool(result.successful):
            break
    protocol_id = canonical_fingerprint(
        {
            "kind": "atomistic-segmented-protocol",
            "rollout": rollout.rollout_id,
            "segments": count,
        }
    )
    return AtomisticSegmentedResult(state, tuple(segments), successful, protocol_id)


__all__ = [
    "AbstractExternalAtomisticProvider",
    "AlchemicalScaledPotential",
    "AtomisticSegmentedResult",
    "ExternalAtomisticEvaluation",
    "ForceGroupEvaluation",
    "RegionMaskedPotential",
    "RESPAPlan",
    "RESPAStepEvaluation",
    "SubtractivePotentialEvaluation",
    "evaluate_force_group",
    "evaluate_subtractive_potential",
    "local_species_energy_delta",
    "respa_step",
    "run_atomistic_segments",
]
