#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-only, transactional material insertion between fixed-topology MD segments.

An epoch is a *new prepared runtime*, not an in-place mask change. All numeric
stepping and checkpoint persistence remain owned by the atomistic executors.
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from ._dynamics import AtomisticDynamicsState, PreparedAtomisticDynamics
from ._system import AtomisticSystemPlan, PreparedAtomisticSystem
from ._topology import MolecularTopologyPlan


def prepare_dormant_system(
    material: PreparedAtomisticSystem,
    active_particle_ids: tuple[int, ...],
    /,
    *,
    mobile_particle_ids: tuple[int, ...] | None = None,
) -> PreparedAtomisticSystem:
    """Restrict a fully parameterized identity-coordinate system to present material.

    Stable IDs and coordinate capacity remain fixed. Every bonded table,
    nonbonded exception, constraint, interaction-site map and particle/pair
    identity is rebuilt from the selected support. Future material parameters
    remain in ``material``, never in the dormant system's active support.
    Virtual-site activation needs an explicit per-epoch coordinate-map compiler
    and is refused here rather than dropping virtual-site physics.
    """
    plan = material.plan
    ids = np.asarray(plan.particle_ids)
    available = set(int(x) for x in ids[np.asarray(material.active_mask)])
    selected = tuple(active_particle_ids)
    if (
        not selected
        or len(set(selected)) != len(selected)
        or not set(selected) <= available
    ):
        raise ValueError("Active IDs must be a nonempty unique subset of material IDs.")
    active = np.isin(ids, selected)
    mobile_ids = (
        tuple(int(x) for x in ids[active & np.asarray(material.mobile_mask)])
        if mobile_particle_ids is None
        else tuple(mobile_particle_ids)
    )
    if len(set(mobile_ids)) != len(mobile_ids) or not set(mobile_ids) <= set(selected):
        raise ValueError("Mobile IDs must be a unique subset of active IDs.")
    mapping = plan.coordinate_map
    expected_slots = np.where(np.asarray(material.active_mask), np.arange(ids.size), -1)
    if (
        mapping.virtual_rules
        or not np.array_equal(np.asarray(mapping.sites.site_ids), ids)
        or not np.array_equal(np.asarray(mapping.physical_dof_indices), expected_slots)
    ):
        raise ValueError(
            "Dormant-system compilation requires identity interaction sites."
        )
    topology = plan.topology

    def selected_rows(table: Array) -> np.ndarray:
        return np.all(np.isin(np.asarray(table), selected), axis=1)

    bond = selected_rows(topology.bonds)
    angle = selected_rows(topology.angles)
    torsion = selected_rows(topology.torsions)
    improper = selected_rows(topology.impropers)
    constraint = selected_rows(topology.constraints)
    exception = selected_rows(topology.pair_exceptions)
    restricted = MolecularTopologyPlan(
        bonds=np.asarray(topology.bonds)[bond],
        bond_type_ids=np.asarray(topology.bond_type_ids)[bond],
        angles=np.asarray(topology.angles)[angle],
        angle_type_ids=np.asarray(topology.angle_type_ids)[angle],
        torsions=np.asarray(topology.torsions)[torsion],
        torsion_type_ids=np.asarray(topology.torsion_type_ids)[torsion],
        impropers=np.asarray(topology.impropers)[improper],
        improper_type_ids=np.asarray(topology.improper_type_ids)[improper],
        constraints=np.asarray(topology.constraints)[constraint],
        constraint_distances=np.asarray(topology.constraint_distances)[constraint],
        pair_exceptions=np.asarray(topology.pair_exceptions)[exception],
        lennard_jones_scales=np.asarray(topology.lennard_jones_scales)[exception],
        electrostatic_scales=np.asarray(topology.electrostatic_scales)[exception],
    )
    return AtomisticSystemPlan(
        ids,
        np.where(active, np.asarray(plan.atomic_numbers), 0),
        plan.masses,
        plan.units,
        atom_type_ids=plan.atom_type_ids,
        element_mask=active & np.asarray(plan.element_mask),
        charges=plan.charges,
        active_mask=active,
        mobile_mask=np.isin(ids, mobile_ids),
        molecule_ids=plan.molecule_ids,
        region_ids=plan.region_ids,
        topology=restricted,
        cell=plan.cell,
        name=plan.name,
        coordinate_dtype=plan.coordinate_dtype,
    ).prepare(numeric_version=material.numeric_version)


@dataclass(frozen=True)
class TopologyEpochTransition:
    """A source-declared replacement of the complete prepared runtime.

    Existing material identity, units, masses and chemistry cannot change.
    Potentials, constraints and mobility may change: their energy and impulse
    changes are explicitly charged to the boundary source. This implementation
    admits nonperiodic Cartesian insertion; periodic insertion/image conventions
    must be separately provided rather than inferred.
    """

    before: PreparedAtomisticDynamics
    after: PreparedAtomisticDynamics
    source_id: str
    maximum_absolute_work: float | None = None

    def __post_init__(self) -> None:
        if not self.source_id or self.source_id != self.source_id.strip():
            raise ValueError(
                "An explicit canonical insertion/protocol source ID is required."
            )
        old, new = self.before.system, self.after.system
        if old.cell is not None or new.cell is not None:
            raise ValueError("Topology insertion currently requires nonperiodic systems.")
        if not np.array_equal(
            np.asarray(old.plan.particle_ids), np.asarray(new.plan.particle_ids)
        ):
            raise ValueError(
                "Topology epochs must preserve stable particle IDs and capacity."
            )
        if old.plan.units.unit_system_id != new.plan.units.unit_system_id:
            raise ValueError("Topology epochs must use identical complete units.")
        active = np.asarray(old.active_mask)
        if np.any(active & ~np.asarray(new.active_mask)):
            raise ValueError("An insertion epoch cannot remove existing material.")
        old_fields = (
            old.plan.masses,
            old.plan.atomic_numbers,
            old.plan.element_mask,
            old.plan.atom_type_ids,
            old.plan.charges,
            old.plan.molecule_ids,
            old.plan.region_ids,
        )
        new_fields = (
            new.plan.masses,
            new.plan.atomic_numbers,
            new.plan.element_mask,
            new.plan.atom_type_ids,
            new.plan.charges,
            new.plan.molecule_ids,
            new.plan.region_ids,
        )
        if any(
            not np.array_equal(np.asarray(a)[active], np.asarray(b)[active])
            for a, b in zip(old_fields, new_fields, strict=True)
        ):
            raise ValueError(
                "Insertion cannot silently change existing material parameters."
            )
        for runtime in (self.before, self.after):
            if runtime.system.topology.constraint_count and runtime.constraints is None:
                raise ValueError(
                    "Every epoch containing constraints requires their executor."
                )
        if self.maximum_absolute_work is not None and (
            not np.isfinite(self.maximum_absolute_work) or self.maximum_absolute_work < 0
        ):
            raise ValueError("The insertion work bound must be finite and nonnegative.")

    @property
    def inserted_particle_ids(self) -> tuple[int, ...]:
        old, new = self.before.system, self.after.system
        inserted = np.asarray(new.active_mask) & ~np.asarray(old.active_mask)
        return tuple(int(x) for x in np.asarray(new.plan.particle_ids)[inserted])

    @property
    def transition_id(self) -> str:
        return canonical_fingerprint(
            {
                "kind": "atomistic-topology-epoch-transition",
                "before": self.before.prepared_id,
                "after": self.after.prepared_id,
                "source": self.source_id,
                "maximum_absolute_work": self.maximum_absolute_work,
            }
        )


@dataclass(frozen=True)
class InsertionLedger:
    """Observed boundary sources in the runtime's mass, momentum and energy units.

    ``external_work`` includes carried kinetic energy. ``protocol_work`` removes
    that advected energy; ``boundary_impulse`` removes the prescribed inserted
    momentum. Constraint projection and changing mobility are therefore visible,
    not mislabeled as conservation failure or hidden by a ledger reset.
    """

    mass_source: Array
    momentum_source: Array
    inserted_momentum: Array
    boundary_impulse: Array
    carried_kinetic_energy: Array
    potential_change: Array
    external_work: Array
    protocol_work: Array
    transition_id: str
    source_id: str


@dataclass(frozen=True)
class TopologyEpochResult:
    runtime: PreparedAtomisticDynamics
    state: AtomisticDynamicsState
    ledger: InsertionLedger | None
    successful: bool
    refusal: str | None


def activate_topology_epoch(
    transition: TopologyEpochTransition,
    state: AtomisticDynamicsState,
    inserted_positions: ArrayLike,
    inserted_momenta: ArrayLike,
    /,
) -> TopologyEpochResult:
    """Build, evaluate and commit a complete new epoch, or return the old state.

    Input rows follow ``transition.inserted_particle_ids`` (stable capacity
    order). Initialization reprojects constraints and rebuilds all neighborhoods
    and force caches. No old runtime, cache, state or random stream is mutated.
    This discrete host operation has no pathwise derivative. Within either
    accepted epoch, native JIT/gradient contracts are unchanged.
    """
    before, after = transition.before, transition.after
    if state.prepared_dynamics_id != before.prepared_id:
        raise ValueError("Activation state belongs to another source epoch.")
    if not bool(state.force.successful) or int(state.force.position_epoch) != int(
        state.step_index
    ):
        raise ValueError("Activation requires a current, successful source force state.")
    slots = np.flatnonzero(
        np.asarray(after.system.active_mask) & ~np.asarray(before.system.active_mask)
    )
    position = np.asarray(inserted_positions)
    momentum = np.asarray(inserted_momenta)
    expected = (len(slots), 3)
    if position.shape != expected or momentum.shape != expected:
        raise ValueError(f"Insertion positions and momenta must have shape {expected}.")

    def rejected(reason: str) -> TopologyEpochResult:
        return TopologyEpochResult(before, state, None, False, reason)

    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(momentum)):
        return rejected("nonfinite-insertion-state")
    dtype = state.kinematics.positions.dtype
    positions = state.kinematics.positions.at[slots].set(
        jnp.asarray(position, dtype=dtype)
    )
    momenta = state.kinematics.momenta.at[slots].set(jnp.asarray(momentum, dtype=dtype))
    species = jnp.where(
        before.system.active_mask, state.species, after.system.plan.atom_type_ids
    )
    # Initialization raises for failed constraint, potential or capacity checks.
    # Catch only its numerical/admission errors: this is the transactional rollback
    # boundary, not a fallback force or a partially accepted physical state.
    try:
        candidate = after.initialize_state(
            positions,
            momentum=momenta,
            time=state.time,
            key=state.random_key,
            species=species,
        )
        candidate.kinematics.positions.block_until_ready()
    except (ValueError, RuntimeError) as error:
        return rejected(f"candidate-initialization-failed:{error}")
    dormant = ~np.asarray(after.system.active_mask)
    if np.any(np.asarray(candidate.force.forces)[dormant] != 0):
        return rejected("potential-exerts-force-on-dormant-material")
    work = candidate.energy.total_energy - state.energy.total_energy
    if not np.isfinite(float(work)):
        return rejected("nonfinite-insertion-work")
    if (
        transition.maximum_absolute_work is not None
        and abs(float(work)) > transition.maximum_absolute_work
    ):
        return rejected("insertion-work-bound-exceeded")
    inserted_p = jnp.sum(jnp.asarray(momentum, dtype=dtype), axis=0)
    delta_p = jnp.sum(candidate.kinematics.momenta - state.kinematics.momenta, axis=0)
    inserted_mass = after.system.plan.masses[slots]
    carried = (
        0.5
        * after.system.plan.units.kinetic_to_energy
        * jnp.sum(jnp.asarray(momentum, dtype=dtype) ** 2 / inserted_mass[:, None])
    )
    ledger = InsertionLedger(
        jnp.sum(inserted_mass),
        delta_p,
        inserted_p,
        delta_p - inserted_p,
        carried,
        candidate.energy.potential_energy - state.energy.potential_energy,
        work,
        work - carried,
        transition.transition_id,
        transition.source_id,
    )
    energy = eqx.tree_at(
        lambda x: (
            x.initial_kinetic_energy,
            x.initial_potential_energy,
            x.thermostat_heat,
            x.barostat_work,
            x.external_work,
            x.constraint_work,
            x.cumulative_balance_residual,
            x.accepted_steps,
        ),
        candidate.energy,
        (
            state.energy.initial_kinetic_energy,
            state.energy.initial_potential_energy,
            state.energy.thermostat_heat,
            state.energy.barostat_work,
            state.energy.external_work + work,
            state.energy.constraint_work,
            state.energy.cumulative_balance_residual,
            state.energy.accepted_steps,
        ),
    )
    candidate = eqx.tree_at(
        lambda x: (x.step_index, x.force.position_epoch, x.energy),
        candidate,
        (state.step_index, state.step_index, energy),
    )
    return TopologyEpochResult(after, candidate, ledger, True, None)


__all__ = [
    "prepare_dormant_system",
    "TopologyEpochTransition",
    "InsertionLedger",
    "TopologyEpochResult",
    "activate_topology_epoch",
]
