#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState


class DistributedConservationLedger(StrictModule):
    mass_defect: Array
    element_defect: Array
    charge_defect: Array
    momentum_defect: Array
    energy_defect: Array
    surface_site_defect: Array
    finite: Array
    successful: Array


class DistributedOwnershipEvidence(StrictModule):
    cell_counts: Array
    particle_counts: Array
    capacity_margin: Array
    unique_face_owners: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class DistributedAerothermodynamicPlan(StrictModule, NonTrainableState):
    """Static cell/face ownership and particle capacities for coupled execution."""

    cell_owner: Array
    face_owner: Array
    particle_capacity_per_shard: Array
    shard_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_owner: ArrayLike,
        face_owner: ArrayLike,
        particle_capacity_per_shard: ArrayLike,
        /,
    ):
        cells = np.asarray(cell_owner, dtype=np.int32)
        faces = np.asarray(face_owner, dtype=np.int32)
        capacities = np.asarray(particle_capacity_per_shard, dtype=np.int32)
        shard_count = capacities.size
        if (
            cells.ndim != 1
            or cells.size == 0
            or faces.ndim != 1
            or capacities.ndim != 1
            or shard_count == 0
            or np.any((cells < 0) | (cells >= shard_count))
            or np.any((faces < 0) | (faces >= shard_count))
            or np.any(capacities < 0)
        ):
            raise ValueError("Distributed ownership or particle capacities are invalid.")
        self.cell_owner = jnp.asarray(cells)
        self.face_owner = jnp.asarray(faces)
        self.particle_capacity_per_shard = jnp.asarray(capacities)
        self.shard_count = shard_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-aerothermodynamics",
                "cell_owner": array_tree_fingerprint(self.cell_owner),
                "face_owner": array_tree_fingerprint(self.face_owner),
                "particle_capacities": array_tree_fingerprint(
                    self.particle_capacity_per_shard
                ),
            }
        )

    def ownership_evidence(
        self, particle_cell_ids: ArrayLike, active: ArrayLike, /
    ) -> DistributedOwnershipEvidence:
        cell_id = jnp.asarray(particle_cell_ids, dtype=jnp.int32)
        active_ = jnp.asarray(active, dtype=bool)
        if cell_id.shape != active_.shape:
            raise ValueError("Particle cell IDs and activity must match.")
        valid_id = (cell_id >= 0) & (cell_id < self.cell_owner.size)
        particle_owner = self.cell_owner[jnp.clip(cell_id, 0, self.cell_owner.size - 1)]
        particle_counts = (
            jnp.zeros((self.shard_count,), dtype=jnp.int32)
            .at[particle_owner]
            .add(jnp.where(active_ & valid_id, 1, 0))
        )
        cell_counts = jnp.bincount(self.cell_owner, length=self.shard_count)
        capacity_margin = self.particle_capacity_per_shard - particle_counts
        face_counts = jnp.bincount(self.face_owner, length=self.shard_count)
        finite = jnp.all(jnp.isfinite(capacity_margin))
        successful = (
            finite
            & jnp.all(capacity_margin >= 0)
            & jnp.all(face_counts >= 0)
            & jnp.all(~active_ | valid_id)
        )
        return DistributedOwnershipEvidence(
            cell_counts,
            particle_counts,
            capacity_margin,
            jnp.sum(face_counts) == self.face_owner.size,
            finite,
            successful,
            self.plan_id,
        )

    def particle_permutation(
        self, particle_cell_ids: ArrayLike, stable_ids: ArrayLike, /
    ) -> Array:
        cells = jnp.asarray(particle_cell_ids, dtype=jnp.int32)
        identifiers = jnp.asarray(stable_ids, dtype=jnp.int64)
        if cells.shape != identifiers.shape:
            raise ValueError("Particle cells and stable IDs must match.")
        owners = self.cell_owner[jnp.clip(cells, 0, self.cell_owner.size - 1)]
        key = owners.astype(jnp.int64) * (jnp.max(identifiers) + 1) + identifiers
        return jnp.argsort(key, stable=True)

    def reconcile_ledgers(
        self,
        ledgers: tuple[DistributedConservationLedger, ...],
        /,
        *,
        tolerance: float = 1.0e-10,
    ) -> DistributedConservationLedger:
        if len(ledgers) != self.shard_count or any(
            not isinstance(value, DistributedConservationLedger) for value in ledgers
        ):
            raise ValueError("Distributed ledgers must contain one value per shard.")
        mass = jnp.sum(jnp.stack(tuple(value.mass_defect for value in ledgers)), axis=0)
        elements = jnp.sum(
            jnp.stack(tuple(value.element_defect for value in ledgers)), axis=0
        )
        charge = jnp.sum(
            jnp.stack(tuple(value.charge_defect for value in ledgers)), axis=0
        )
        momentum = jnp.sum(
            jnp.stack(tuple(value.momentum_defect for value in ledgers)), axis=0
        )
        energy = jnp.sum(
            jnp.stack(tuple(value.energy_defect for value in ledgers)), axis=0
        )
        sites = jnp.sum(
            jnp.stack(tuple(value.surface_site_defect for value in ledgers)), axis=0
        )
        values = (mass, elements, charge, momentum, energy, sites)
        finite = jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in values))
        )
        maximum = jnp.max(jnp.stack(tuple(jnp.max(jnp.abs(value)) for value in values)))
        return DistributedConservationLedger(
            *values, finite, finite & (maximum <= tolerance)
        )


__all__ = [
    "DistributedAerothermodynamicPlan",
    "DistributedConservationLedger",
    "DistributedOwnershipEvidence",
]
