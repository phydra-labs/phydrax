#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization import (
    halo_sum,
    halo_update,
    migrate_particle_halos,
    ParticleDomainDecompositionPlan,
    ParticleHaloState,
    prepare_particle_halos,
)
from ._constraints import PreparedDistanceConstraints
from ._potential_program import (
    AtomisticPotentialEvaluation,
    PreparedAtomisticPotentialProgram,
)
from ._system import PreparedAtomisticSystem


class DistributedAtomisticPlan(StrictModule, NonTrainableState):
    system: PreparedAtomisticSystem
    decomposition: ParticleDomainDecompositionPlan
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        system: PreparedAtomisticSystem,
        decomposition: ParticleDomainDecompositionPlan,
        /,
    ):
        if not isinstance(system, PreparedAtomisticSystem) or not isinstance(
            decomposition, ParticleDomainDecompositionPlan
        ):
            raise TypeError(
                "Distributed atomistics requires prepared system and decomposition."
            )
        if decomposition.halo_radius <= 0.0:
            raise ValueError("Distributed atomistic halo radius must be positive.")
        self.system, self.decomposition = system, decomposition
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic",
                "system": system.prepared_id,
                "decomposition": decomposition.plan_id,
            }
        )

    def prepare(self, positions: ArrayLike, /) -> "DistributedAtomisticState":
        coordinate = jnp.asarray(positions)
        if coordinate.shape != (self.system.capacity, 3):
            raise ValueError("Distributed positions must match atomistic capacity.")
        halo = prepare_particle_halos(
            self.decomposition, coordinate, self.system.active_mask
        )
        return DistributedAtomisticState(
            coordinate,
            halo,
            jnp.zeros((self.decomposition.partitions, 3)),
            jnp.zeros((self.decomposition.partitions,)),
            halo.successful,
            self.plan_id,
        )


class DistributedAtomisticState(StrictModule):
    positions: Array
    halos: ParticleHaloState
    partition_momentum: Array
    partition_energy: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def migrate_distributed_atomistic(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    positions: ArrayLike,
    /,
):
    if state.plan_id != plan.plan_id:
        raise ValueError("Distributed state belongs to another plan.")
    coordinate = jnp.asarray(positions)
    if coordinate.shape != state.positions.shape:
        raise ValueError("Migrated distributed positions changed shape.")
    halo = migrate_particle_halos(
        plan.decomposition, state.halos, coordinate, plan.system.active_mask
    )
    return DistributedAtomisticState(
        coordinate,
        halo,
        state.partition_momentum,
        state.partition_energy,
        state.successful & halo.successful,
        state.plan_id,
    )


def halo_short_range_evaluate(
    plan: DistributedAtomisticPlan,
    state: DistributedAtomisticState,
    potential: PreparedAtomisticPotentialProgram,
    neighborhood,
    /,
):
    if (
        state.plan_id != plan.plan_id
        or potential.system.prepared_id != plan.system.prepared_id
    ):
        raise ValueError("Distributed state or potential belongs to another plan.")
    cutoff = potential.plan.requirements.cutoff
    if cutoff is not None and cutoff > plan.decomposition.halo_radius:
        raise ValueError("Distributed halo radius is smaller than potential cutoff.")
    if potential.plan.requirements.reciprocal_grid:
        raise ValueError(
            "Reciprocal terms require distributed_particle_mesh_electrostatics."
        )
    evaluation = potential.evaluate(state.positions, neighborhood)
    local_force = halo_update(evaluation.forces, state.halos)
    owned = state.halos.owned_mask[..., None]
    local_force = jnp.where(owned, local_force, 0.0)
    reverse_force = halo_sum(local_force, state.halos)
    local_atom_energy = halo_update(evaluation.atom_energy, state.halos)
    partition_energy = jnp.sum(
        jnp.where(state.halos.owned_mask, local_atom_energy, 0.0), axis=1
    )
    distributed = AtomisticPotentialEvaluation(
        evaluation.energy,
        evaluation.term_energies,
        evaluation.atom_energy,
        reverse_force,
        evaluation.virial,
        evaluation.successful & state.halos.successful,
        evaluation.neighborhood_successful,
        evaluation.graph_overflow,
        evaluation.program_id,
    )
    return distributed, partition_energy


def distributed_constraint_projection(
    constraints: PreparedDistanceConstraints,
    previous_positions: ArrayLike,
    proposed_positions: ArrayLike,
    momenta: ArrayLike,
    /,
):
    return constraints.project_positions(previous_positions, proposed_positions, momenta)


def distributed_thermodynamic_reduction(
    local_energy: ArrayLike, local_momentum: ArrayLike, /
):
    return jnp.sum(jnp.asarray(local_energy)), jnp.sum(
        jnp.asarray(local_momentum), axis=0
    )


def distributed_particle_mesh_electrostatics(
    evaluation: AtomisticPotentialEvaluation, halo: ParticleHaloState, /
):
    local_force = halo_update(evaluation.forces, halo)
    local_force = jnp.where(halo.owned_mask[..., None], local_force, 0.0)
    return halo_sum(local_force, halo), evaluation.energy


class DistributedAtomisticCheckpointIdentity(StrictModule, NonTrainableState):
    plan_id: str = eqx.field(static=True)
    owner_digest: str = eqx.field(static=True)
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, state: DistributedAtomisticState, /):
        owner_digest = canonical_fingerprint(
            {"kind": "halo-owner", "owner": jnp.asarray(state.halos.owner).tolist()}
        )
        self.plan_id, self.owner_digest = state.plan_id, owner_digest
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "distributed-atomistic-checkpoint",
                "plan": state.plan_id,
                "owner": owner_digest,
            }
        )


__all__ = [
    "DistributedAtomisticCheckpointIdentity",
    "DistributedAtomisticPlan",
    "DistributedAtomisticState",
    "distributed_constraint_projection",
    "distributed_particle_mesh_electrostatics",
    "distributed_thermodynamic_reduction",
    "halo_short_range_evaluate",
    "migrate_distributed_atomistic",
]
