#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from itertools import pairwise

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ....atomistic import (
    AtomisticDynamicsState,
    PolymerChainLayoutPlan,
    PreparedAtomisticDynamics,
)


class PrimitivePathSnapshotPlan(StrictModule, NonTrainableState):
    maximum_particles: int = eqx.field(static=True)
    maximum_chains: int = eqx.field(static=True)
    maximum_beads_per_chain: int = eqx.field(static=True)
    maximum_bond_length: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_particles: int,
        maximum_chains: int,
        maximum_beads_per_chain: int,
        maximum_bond_length: float,
    ):
        particles = int(maximum_particles)
        chains = int(maximum_chains)
        beads = int(maximum_beads_per_chain)
        bond = float(maximum_bond_length)
        if (
            particles <= 0
            or chains <= 0
            or beads < 2
            or not math.isfinite(bond)
            or bond <= 0.0
        ):
            raise ValueError("Primitive-path snapshot capacities are invalid.")
        self.maximum_particles = particles
        self.maximum_chains = chains
        self.maximum_beads_per_chain = beads
        self.maximum_bond_length = bond
        self.plan_id = canonical_fingerprint(
            {
                "kind": "primitive-path-snapshot-plan",
                "maximum_particles": particles,
                "maximum_chains": chains,
                "maximum_beads_per_chain": beads,
                "maximum_bond_length": bond,
                "architecture": "linear-open-unbranched",
                "coordinate_representation": "unwrapped",
            }
        )

    def prepare(
        self,
        dynamics: PreparedAtomisticDynamics,
        layout: PolymerChainLayoutPlan,
        chain_ids: tuple[str, ...],
        /,
    ) -> PreparedPrimitivePathSnapshot:
        return PreparedPrimitivePathSnapshot(self, dynamics, layout, chain_ids)


class PrimitivePathSnapshot(StrictModule):
    unwrapped_positions: Array
    chain_indices: Array
    chain_mask: Array
    stable_particle_ids: Array
    end_to_end_squared: Array
    image_counts: Array
    cell_vectors: Array
    time: Array
    step_index: Array
    source_state_id: str = eqx.field(static=True)
    snapshot_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)


class PrimitivePathSnapshotEvidence(StrictModule):
    maximum_bond_length: Array
    chain_winding: Array
    active_particles: Array
    finite: Array
    topology_matches: Array
    continuity_valid: Array
    successful: Array
    prepared_id: str = eqx.field(static=True)


class PreparedPrimitivePathSnapshot(StrictModule, NonTrainableState):
    plan: PrimitivePathSnapshotPlan
    dynamics: PreparedAtomisticDynamics
    layout: PolymerChainLayoutPlan
    chain_ids: tuple[str, ...] = eqx.field(static=True)
    expected_bonds: Array
    prepared_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: PrimitivePathSnapshotPlan,
        dynamics: PreparedAtomisticDynamics,
        layout: PolymerChainLayoutPlan,
        chain_ids: tuple[str, ...],
        /,
    ):
        if not isinstance(plan, PrimitivePathSnapshotPlan):
            raise TypeError("plan must be PrimitivePathSnapshotPlan.")
        if not isinstance(dynamics, PreparedAtomisticDynamics):
            raise TypeError("dynamics must be PreparedAtomisticDynamics.")
        if not isinstance(layout, PolymerChainLayoutPlan):
            raise TypeError("layout must be PolymerChainLayoutPlan.")
        identifiers = tuple(str(value).strip() for value in chain_ids)
        indices = np.asarray(layout.particle_indices, dtype=np.int32)
        mask = np.asarray(layout.chain_mask, dtype=np.bool_)
        selected = indices[mask]
        active = np.asarray(dynamics.system.active_mask, dtype=np.bool_)
        if (
            len(identifiers) != indices.shape[0]
            or any(not value for value in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or indices.shape[0] > plan.maximum_chains
            or indices.shape[1] > plan.maximum_beads_per_chain
            or selected.size > plan.maximum_particles
            or np.unique(selected).size != selected.size
            or selected.size != int(np.count_nonzero(active))
            or np.any(selected < 0)
            or np.any(selected >= dynamics.system.capacity)
            or not np.all(active[selected])
            or np.any(np.sum(mask, axis=1) < 2)
        ):
            raise ValueError(
                "Primitive-path layout must partition all active particles into linear chains."
            )
        expected: list[tuple[int, int]] = []
        for row, active_row in zip(indices, mask, strict=True):
            chain = row[active_row]
            expected.extend(
                tuple(sorted((int(left), int(right)))) for left, right in pairwise(chain)
            )
        actual = {
            tuple(sorted((int(left), int(right))))
            for left, right in np.asarray(dynamics.system.topology.bond_indices)
        }
        if actual != set(expected):
            raise ValueError(
                "Primitive-path canonical requires exactly adjacent open-chain backbone bonds."
            )
        if dynamics.system.topology.angle_indices.shape[0] and any(
            len(set(row)) != 3
            for row in np.asarray(dynamics.system.topology.angle_indices)
        ):
            raise ValueError("Primitive-path source contains malformed angles.")
        self.plan = plan
        self.dynamics = dynamics
        self.layout = layout
        self.chain_ids = identifiers
        self.expected_bonds = jnp.asarray(expected, dtype=jnp.int32)
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "prepared-primitive-path-snapshot",
                "plan": plan.plan_id,
                "dynamics": dynamics.prepared_id,
                "layout": layout.plan_id,
                "chain_ids": list(identifiers),
                "topology": dynamics.system.topology.topology_id,
            }
        )

    def capture(
        self, state: AtomisticDynamicsState, /
    ) -> tuple[PrimitivePathSnapshot, PrimitivePathSnapshotEvidence]:
        if not isinstance(state, AtomisticDynamicsState):
            raise TypeError("state must be AtomisticDynamicsState.")
        if state.prepared_dynamics_id != self.dynamics.prepared_id:
            raise ValueError("Source state belongs to another dynamics runtime.")
        unwrapped = self.dynamics._unwrapped(state.kinematics, state.cell_vectors)
        indices = self.layout.particle_indices
        mask = self.layout.chain_mask
        chain_positions = unwrapped[indices]
        displacement = chain_positions[:, 1:, :] - chain_positions[:, :-1, :]
        bond_mask = mask[:, 1:] & mask[:, :-1]
        lengths = jnp.sqrt(jnp.sum(displacement * displacement, axis=-1))
        maximum_bond = jnp.max(jnp.where(bond_mask, lengths, 0.0))
        first = unwrapped[self.layout.first_indices]
        last = unwrapped[self.layout.last_indices]
        end_squared = jnp.sum((last - first) ** 2, axis=-1)
        cell = self.dynamics.system.cell
        if cell is None:
            vectors = jnp.zeros((0, 3), dtype=unwrapped.dtype)
            winding = jnp.zeros((len(self.chain_ids), 0), dtype=jnp.int32)
        else:
            vectors = state.cell_vectors
            first_fractional = cell.fractional_with_vectors(first, vectors)
            last_fractional = cell.fractional_with_vectors(last, vectors)
            winding = jnp.rint(last_fractional - first_fractional).astype(jnp.int32)
        finite = (
            jnp.all(jnp.isfinite(unwrapped))
            & jnp.all(jnp.isfinite(lengths))
            & jnp.all(jnp.isfinite(end_squared))
        )
        continuity = jnp.all(~bond_mask | (lengths <= self.plan.maximum_bond_length))
        topology_matches = jnp.asarray(True)
        successful = state.force.successful & finite & continuity & topology_matches
        source_state_id = canonical_fingerprint(
            {
                "kind": "primitive-path-source-state",
                "prepared": self.prepared_id,
                "step": int(state.step_index),
                "time": float(state.time),
                "positions": array_tree_fingerprint(np.asarray(unwrapped)),
                "images": array_tree_fingerprint(
                    np.asarray(state.kinematics.image_counts)
                ),
                "cell_vectors": array_tree_fingerprint(np.asarray(vectors)),
            }
        )
        snapshot_id = canonical_fingerprint(
            {
                "kind": "primitive-path-snapshot",
                "source_state": source_state_id,
                "coordinate_representation": "unwrapped",
            }
        )
        snapshot = PrimitivePathSnapshot(
            unwrapped,
            indices,
            mask,
            self.dynamics.system.plan.particle_ids,
            end_squared,
            state.kinematics.image_counts,
            vectors,
            state.time,
            state.step_index,
            source_state_id,
            snapshot_id,
            self.prepared_id,
        )
        evidence = PrimitivePathSnapshotEvidence(
            maximum_bond,
            winding,
            jnp.sum(self.dynamics.system.active_mask, dtype=jnp.int32),
            finite,
            topology_matches,
            continuity,
            successful,
            self.prepared_id,
        )
        return snapshot, evidence


__all__ = [
    "PreparedPrimitivePathSnapshot",
    "PrimitivePathSnapshot",
    "PrimitivePathSnapshotEvidence",
    "PrimitivePathSnapshotPlan",
]
