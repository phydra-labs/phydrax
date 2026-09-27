#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Space-filling-curve partitions, ghost layers, and migration for forest AMR.

Leaves are owned by weighted contiguous ranges of the canonical Morton leaf order,
the same locality rule as fixed-block AMR partitions.  Ghost layers and colored
peer exchange reuse :class:`DistributedHaloPlan`; distributed execution uses its
``jax.shard_map`` + ``lax.ppermute`` phases, and migration after adaptation is one
sparse route from source to target packed layouts.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...sparse import EdgeRelation, linear_apply
from .._distributed_field import DistributedHaloPlan
from ._distributed import _active_costs, _contiguous_owners
from ._forest import (
    _locate_cells,
    _shifted_cells,
    AMRBalanceStencil,
    balance_directions,
    ForestHierarchyTopology,
)
from ._forest_transfer import ForestFieldTransition


def _ghost_adjacency(
    topology: ForestHierarchyTopology,
    stencil: AMRBalanceStencil,
    /,
) -> np.ndarray:
    """Symmetric-complete touching leaf pairs over one ghost stencil.

    Every touching pair is found from its finer (or equal) member, whose
    same-level neighbor cell lies inside the coarser member.
    """
    plan = topology.plan
    levels = topology.leaf_levels()
    coordinates = topology.leaf_coordinates()
    keys = topology.leaf_keys()
    directions = balance_directions(plan.dimension, stencil)
    count = topology.leaf_count
    sources = np.repeat(np.arange(count, dtype=np.int64), directions.shape[0])
    source_levels = levels[sources]
    cells, inside = _shifted_cells(
        plan,
        source_levels,
        coordinates[sources],
        np.tile(directions, (count, 1)),
    )
    located = _locate_cells(plan, keys, source_levels[inside], cells[inside])
    pairs = np.stack((sources[inside], located), axis=1)
    pairs = pairs[
        (levels[located] <= source_levels[inside]) & (pairs[:, 0] != pairs[:, 1])
    ]
    return np.unique(np.sort(pairs, axis=1), axis=0)


class ForestPartitionEvidence(StrictModule, NonTrainableState):
    """Per-part leaf counts, costs, ghost counts, imbalance, and cut faces."""

    part_leaf_counts: tuple[int, ...] = eqx.field(static=True)
    part_costs: tuple[float, ...] = eqx.field(static=True)
    ghost_counts: tuple[int, ...] = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    cut_faces: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_leaf_counts: Any,
        part_costs: Any,
        ghost_counts: Any,
        cut_faces: int,
        /,
    ) -> None:
        counts = tuple(int(value) for value in part_leaf_counts)
        costs = tuple(float(value) for value in part_costs)
        ghosts = tuple(int(value) for value in ghost_counts)
        if (
            not counts
            or len(costs) != len(counts)
            or len(ghosts) != len(counts)
            or min(counts) <= 0
            or any(not np.isfinite(value) or value <= 0.0 for value in costs)
            or min(ghosts) < 0
            or int(cut_faces) < 0
        ):
            raise ValueError("Forest partition evidence is invalid.")
        mean = sum(costs) / len(costs)
        self.part_leaf_counts = counts
        self.part_costs = costs
        self.ghost_counts = ghosts
        self.maximum_imbalance = max(costs) / mean - 1.0
        self.cut_faces = int(cut_faces)
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "forest-partition-evidence",
                "counts": counts,
                "costs": costs,
                "ghosts": ghosts,
                "cut_faces": int(cut_faces),
            }
        )


class ForestPartitionPlan(StrictModule, NonTrainableState):
    """Weighted Morton-contiguous leaf ownership with one ghost layer."""

    part_count: int = eqx.field(static=True)
    ghost_stencil: AMRBalanceStencil = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_count: int,
        /,
        *,
        ghost_stencil: AMRBalanceStencil = AMRBalanceStencil.FACE,
        axis_name: str = "forest_parts",
    ) -> None:
        parts = int(part_count)
        name = str(axis_name).strip()
        if parts <= 0 or not name:
            raise ValueError("Forest partitions require positive parts and an axis name.")
        stencil = AMRBalanceStencil(ghost_stencil)
        self.part_count = parts
        self.ghost_stencil = stencil
        self.axis_name = name
        self.plan_id = canonical_fingerprint(
            {
                "kind": "forest-partition-plan",
                "part_count": parts,
                "ghost_stencil": stencil.value,
                "axis_name": name,
                "locality": "morton-contiguous-weighted",
            }
        )

    def prepare(
        self,
        topology: ForestHierarchyTopology,
        /,
        *,
        weights: ArrayLike | None = None,
    ) -> PreparedForestPartition:
        return PreparedForestPartition(self, topology, weights=weights)


class PreparedForestPartition(StrictModule, NonTrainableState):
    """Owned/ghost packed layout, colored exchange, and part-local face routes.

    ``pack`` returns ``(P, L, *T)`` owned values; ``exchange_ghosts`` fills ghost
    rows from their owners across parts with ``jax.shard_map`` phases.  Local
    faces ``local_face_minus/plus`` index the packed local layout of each part and
    cover every face touching an owned leaf, so part-local flux kernels need only
    ghost-exchanged state.
    """

    plan: ForestPartitionPlan
    topology: ForestHierarchyTopology
    owners: Array
    local_slots: Array
    halo: DistributedHaloPlan
    local_face_ids: Array
    local_face_minus: Array
    local_face_plus: Array
    local_face_valid: Array
    evidence: ForestPartitionEvidence
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: ForestPartitionPlan,
        topology: ForestHierarchyTopology,
        /,
        *,
        weights: ArrayLike | None = None,
    ) -> None:
        if not isinstance(plan, ForestPartitionPlan) or not isinstance(
            topology, ForestHierarchyTopology
        ):
            raise TypeError("Forest partitions require a plan and a forest topology.")
        count = topology.leaf_count
        parts = plan.part_count
        if count < parts:
            raise ValueError("Forest partitions require at least one leaf per part.")
        costs = _active_costs(weights, count, topology.signature.leaf_capacity)
        owners = _contiguous_owners(costs, parts)
        pairs = _ghost_adjacency(topology, plan.ghost_stencil)
        halo = DistributedHaloPlan(
            owners,
            pairs if pairs.size else np.zeros((0, 2), dtype=np.int64),
            parts,
        )
        global_ids = np.asarray(halo.local_global_ids, dtype=np.int64)
        local_valid = np.asarray(halo.local_valid, dtype=np.bool_)
        local_owned = np.asarray(halo.local_owned, dtype=np.bool_)
        local_slots = np.full((topology.signature.leaf_capacity,), -1, dtype=np.int32)
        owned_parts, owned_rows = np.nonzero(local_owned)
        local_slots[global_ids[owned_parts, owned_rows]] = owned_rows
        workset = topology.workset
        face_valid = np.asarray(workset.face_valid, dtype=np.bool_)
        face_minus = np.asarray(workset.face_minus, dtype=np.int64)[face_valid]
        face_plus = np.asarray(workset.face_plus, dtype=np.int64)[face_valid]
        face_ids: list[np.ndarray] = []
        face_local: list[tuple[np.ndarray, np.ndarray]] = []
        for part in range(parts):
            to_local = np.full((count,), -1, dtype=np.int64)
            valid_rows = np.flatnonzero(local_valid[part])
            to_local[global_ids[part, valid_rows]] = valid_rows
            touching = np.flatnonzero(
                (owners[face_minus] == part) | (owners[face_plus] == part)
            )
            minus_local = to_local[face_minus[touching]]
            plus_local = to_local[face_plus[touching]]
            if np.any(minus_local < 0) or np.any(plus_local < 0):
                raise RuntimeError("Forest ghost layer does not close owned faces.")
            face_ids.append(touching)
            face_local.append((minus_local, plus_local))
        face_capacity = max(max(ids.size for ids in face_ids), 1)
        local_face_ids = np.zeros((parts, face_capacity), dtype=np.int32)
        local_face_minus = np.zeros((parts, face_capacity), dtype=np.int32)
        local_face_plus = np.zeros((parts, face_capacity), dtype=np.int32)
        local_face_valid = np.zeros((parts, face_capacity), dtype=np.bool_)
        for part, (ids, (minus_local, plus_local)) in enumerate(
            zip(face_ids, face_local, strict=True)
        ):
            local_face_ids[part, : ids.size] = ids
            local_face_minus[part, : ids.size] = minus_local
            local_face_plus[part, : ids.size] = plus_local
            local_face_valid[part, : ids.size] = True
        cut_faces = int(np.count_nonzero(owners[face_minus] != owners[face_plus]))
        padded_owners = np.full((topology.signature.leaf_capacity,), -1, dtype=np.int32)
        padded_owners[:count] = owners
        evidence = ForestPartitionEvidence(
            np.bincount(owners, minlength=parts),
            np.bincount(owners, weights=costs, minlength=parts),
            np.count_nonzero(local_valid & ~local_owned, axis=1),
            cut_faces,
        )
        self.plan = plan
        self.topology = topology
        self.owners = jnp.asarray(padded_owners)
        self.local_slots = jnp.asarray(local_slots)
        self.halo = halo
        self.local_face_ids = jnp.asarray(local_face_ids)
        self.local_face_minus = jnp.asarray(local_face_minus)
        self.local_face_plus = jnp.asarray(local_face_plus)
        self.local_face_valid = jnp.asarray(local_face_valid)
        self.evidence = evidence
        self.partition_id = canonical_fingerprint(
            {
                "kind": "forest-partition",
                "plan": plan.plan_id,
                "topology": topology.topology_id,
                "owners": array_tree_fingerprint(owners),
                "halo": halo.plan_id,
            }
        )

    @property
    def local_capacity(self) -> int:
        return self.halo.local_capacity

    def _compact(self, values: ArrayLike, /) -> Array:
        array = jnp.asarray(values)
        if array.ndim == 0 or array.shape[0] != self.topology.signature.leaf_capacity:
            raise ValueError("Forest partition values must match the leaf capacity.")
        return array[: self.topology.leaf_count]

    def pack(self, values: ArrayLike, /) -> Array:
        """Owned packed layout ``(P, L, *T)`` of padded leaf values ``(C, *T)``."""
        return self.halo.pack_owned(self._compact(values))

    def pack_with_ghosts(self, values: ArrayLike, /) -> Array:
        """Owned plus ghost packed layout ``(P, L, *T)`` of padded leaf values."""
        return self.halo.pack_reference(self._compact(values))

    def unpack(self, local_values: ArrayLike, /) -> Array:
        """Padded leaf values ``(C, *T)`` from owned packed rows."""
        owned = self.halo.unpack_owned(local_values)
        padding = jnp.zeros(
            (self.topology.signature.leaf_capacity - self.topology.leaf_count,)
            + owned.shape[1:],
            dtype=owned.dtype,
        )
        return jnp.concatenate((owned, padding), axis=0)

    def exchange_ghosts(
        self,
        local_values: ArrayLike,
        /,
        *,
        execution_group: ExecutionGroup | None = None,
    ) -> Array:
        """Fill ghost rows from their owners with one device per part."""
        values = jnp.asarray(local_values)
        if values.shape[:2] != (self.halo.part_count, self.halo.local_capacity):
            raise ValueError("Packed forest values do not match the partition layout.")
        devices = (
            tuple(jax.devices())[: self.halo.part_count]
            if execution_group is None
            else execution_group.devices
        )
        if len(devices) != self.halo.part_count:
            raise ValueError("Ghost exchange requires one JAX device per partition.")
        axis_name = self.plan.axis_name
        mesh = Mesh(np.asarray(devices, dtype=object), (axis_name,))
        packed_spec = PartitionSpec(axis_name, *(None for _ in range(values.ndim - 1)))
        part_spec = PartitionSpec(axis_name)
        values = jax.device_put(values, NamedSharding(mesh, packed_spec))
        parts = jax.device_put(
            jnp.arange(self.halo.part_count, dtype=jnp.int32),
            NamedSharding(mesh, part_spec),
        )
        halo = self.halo

        def exchange(local: Any, part: Any) -> Any:
            return halo.exchange(local[0], part[0], axis_name=axis_name)[None, ...]

        return jax.shard_map(
            exchange,
            mesh=mesh,
            in_specs=(packed_spec, part_spec),
            out_specs=packed_spec,
            check_vma=False,
        )(values, parts)

    def migration_to(
        self,
        target: PreparedForestPartition,
        /,
        *,
        transition: ForestFieldTransition | None = None,
    ) -> ForestMigrationPlan:
        return ForestMigrationPlan(self, target, transition=transition)


class ForestMigrationPlan(StrictModule, NonTrainableState):
    """Owned-layout migration between two forest partitions, across adaptation.

    Without a transition both partitions share one topology and migration is the
    permutation between owners.  With a :class:`ForestFieldTransition` the
    conservative leaf routes are composed with both packed layouts, so data moves
    and transfers in one sparse action.
    """

    source: PreparedForestPartition
    target: PreparedForestPartition
    relation: EdgeRelation
    coefficients: Array
    moved_leaves: int = eqx.field(static=True)
    migration_id: str = eqx.field(static=True)

    def __init__(
        self,
        source: PreparedForestPartition,
        target: PreparedForestPartition,
        /,
        *,
        transition: ForestFieldTransition | None = None,
    ) -> None:
        if not isinstance(source, PreparedForestPartition) or not isinstance(
            target, PreparedForestPartition
        ):
            raise TypeError("Forest migration requires prepared partitions.")
        if source.plan.part_count != target.plan.part_count:
            raise ValueError("Forest migration requires one shared part count.")
        if transition is None:
            if source.topology.topology_id != target.topology.topology_id:
                raise ValueError("Topology-changing migration requires a transition.")
            count = source.topology.leaf_count
            route_sources = np.arange(count, dtype=np.int64)
            route_targets = route_sources
            weights = np.ones((count,), dtype=np.float64)
        else:
            if not isinstance(transition, ForestFieldTransition):
                raise TypeError("Forest migration transitions must be field transitions.")
            if (
                transition.source.epoch.epoch_id != source.topology.epoch.epoch_id
                or transition.target.epoch.epoch_id != target.topology.epoch.epoch_id
            ):
                raise ValueError("Forest migration transition epochs do not match.")
            valid = np.asarray(transition.routes.relation.valid, dtype=np.bool_)
            route_sources = np.asarray(
                transition.routes.relation.source_indices, dtype=np.int64
            )[valid]
            route_targets = np.asarray(
                transition.routes.relation.target_indices, dtype=np.int64
            )[valid]
            weights = np.asarray(transition.routes.coefficients, dtype=np.float64)[valid]
        source_owners = np.asarray(source.owners, dtype=np.int64)
        target_owners = np.asarray(target.owners, dtype=np.int64)
        source_rows = (
            source_owners[route_sources] * source.local_capacity
            + np.asarray(source.local_slots, dtype=np.int64)[route_sources]
        )
        target_rows = (
            target_owners[route_targets] * target.local_capacity
            + np.asarray(target.local_slots, dtype=np.int64)[route_targets]
        )
        moved = np.unique(
            route_targets[source_owners[route_sources] != target_owners[route_targets]]
        )
        parts = source.plan.part_count
        self.source = source
        self.target = target
        self.relation = EdgeRelation(
            source_rows,
            target_rows,
            source_size=parts * source.local_capacity,
            target_size=parts * target.local_capacity,
        )
        self.coefficients = jnp.asarray(weights)
        self.moved_leaves = int(moved.size)
        self.migration_id = canonical_fingerprint(
            {
                "kind": "forest-migration",
                "source": source.partition_id,
                "target": target.partition_id,
                "routes": array_tree_fingerprint((source_rows, target_rows, weights)),
            }
        )

    def migrate(self, local_values: ArrayLike, /) -> Array:
        """Map source owned rows ``(P, Ls, *T)`` to target owned rows ``(P, Lt, *T)``."""
        values = jnp.asarray(local_values)
        parts = self.source.plan.part_count
        if values.shape[:2] != (parts, self.source.local_capacity):
            raise ValueError("Migration input does not match the source layout.")
        flat = values.reshape((parts * self.source.local_capacity,) + values.shape[2:])
        moved = linear_apply(self.relation, self.coefficients, flat)
        return moved.reshape((parts, self.target.local_capacity) + values.shape[2:])


__all__ = [
    "ForestMigrationPlan",
    "ForestPartitionEvidence",
    "ForestPartitionPlan",
    "PreparedForestPartition",
]
