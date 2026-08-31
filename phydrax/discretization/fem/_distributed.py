#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator, ConstraintMap
from .._cell_mesh import CellMesh
from ._generic import FiniteElementDofMap
from ._mortar import FiniteElementMortarPlan


class PartitionedFiniteElementDofMap(StrictModule, NonTrainableState):
    """Owned/halo view of one FE coordinate map with stable global IDs."""

    dof_map: FiniteElementDofMap
    global_ids: Array
    owned_mask: Array
    halo_mask: Array
    multiplicity: Array
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        dof_map: FiniteElementDofMap,
        global_ids: ArrayLike,
        owned_mask: ArrayLike,
        /,
        *,
        multiplicity: ArrayLike | None = None,
        partition_id: str | None = None,
    ):
        if not isinstance(dof_map, FiniteElementDofMap):
            raise TypeError("dof_map must be FiniteElementDofMap.")
        identifiers = np.asarray(global_ids, dtype=np.int64)
        owned = np.asarray(owned_mask, dtype=bool)
        if (
            identifiers.shape != (dof_map.global_dof_count,)
            or owned.shape != identifiers.shape
        ):
            raise ValueError("Distributed DOF IDs and ownership must match global DOFs.")
        if np.any(identifiers < 0) or np.unique(identifiers).size != identifiers.size:
            raise ValueError(
                "Distributed DOF global IDs must be unique and non-negative."
            )
        weights = (
            np.ones(identifiers.shape, dtype=float)
            if multiplicity is None
            else np.asarray(multiplicity, dtype=float)
        )
        if (
            weights.shape != identifiers.shape
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError("DOF multiplicity must be positive and finite.")
        identifier = (
            canonical_fingerprint(
                {
                    "kind": "partitioned-finite-element-dof-map",
                    "dof_map": dof_map.dof_map_id,
                    "global_ids": array_tree_fingerprint(identifiers),
                    "owned": array_tree_fingerprint(owned),
                    "multiplicity": array_tree_fingerprint(weights),
                }
            )
            if partition_id is None
            else str(partition_id)
        )
        if not identifier:
            raise ValueError("partition_id must be non-empty.")
        self.dof_map = dof_map
        self.global_ids = jnp.asarray(identifiers)
        self.owned_mask = jnp.asarray(owned)
        self.halo_mask = jnp.asarray(~owned)
        self.multiplicity = jnp.asarray(weights)
        self.partition_id = identifier

    def global_inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return this partition's exactly-once contribution to the global pairing."""

        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape or left_.shape[0] != self.dof_map.global_dof_count:
            raise ValueError("Distributed inner-product arrays have invalid shape.")
        owned = self.owned_mask.reshape(self.owned_mask.shape + (1,) * (left_.ndim - 1))
        return jnp.sum(jnp.where(owned, jnp.conj(left_) * right_, 0.0))

    def replica_inner(self, left: ArrayLike, right: ArrayLike, /) -> Array:
        """Return a multiplicity-weighted contribution from every local replica."""

        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape or left_.shape[0] != self.dof_map.global_dof_count:
            raise ValueError("Distributed inner-product arrays have invalid shape.")
        weights = self.multiplicity.reshape(
            self.multiplicity.shape + (1,) * (left_.ndim - 1)
        )
        return jnp.sum(jnp.conj(left_) * right_ / weights)

    def pullback_global(
        self,
        local_dual: ArrayLike,
        /,
        *,
        halo_plan: FiniteElementHaloPlan | None = None,
    ) -> Array:
        """Sum replica duals, when supplied, then retain each global DOF once."""

        value = jnp.asarray(local_dual)
        if value.shape[0] != self.dof_map.global_dof_count:
            raise ValueError("Distributed dual array has invalid shape.")
        if halo_plan is not None:
            if not isinstance(halo_plan, FiniteElementHaloPlan):
                raise TypeError("halo_plan must be FiniteElementHaloPlan or None.")
            value = halo_plan.sum_contributions(value)
        owned = self.owned_mask.reshape(self.owned_mask.shape + (1,) * (value.ndim - 1))
        return jnp.where(owned, value, jnp.zeros((), dtype=value.dtype))


class FiniteElementHaloPlan(StrictModule, NonTrainableState):
    """Replica routes with fixed-order update, sum, average, and pullbacks."""

    replica_groups: Array
    valid: Array
    owner_columns: Array
    replica_count: int = eqx.field(static=True)
    reduction_semantics: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        replica_groups: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
        owner_columns: ArrayLike | None = None,
    ):
        groups = np.asarray(replica_groups, dtype=np.int32)
        if groups.ndim != 2 or groups.shape[0] == 0 or groups.shape[1] < 2:
            raise ValueError("replica_groups must have shape (groups, width >= 2).")
        valid_ = (
            np.ones(groups.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if (
            valid_.shape != groups.shape
            or np.any(groups[valid_] < 0)
            or np.any(np.sum(valid_, axis=1) < 2)
        ):
            raise ValueError("Halo routes or validity mask are invalid.")
        active = groups[valid_]
        if np.unique(active).size != active.size:
            raise ValueError("A replica index may occur in only one halo group.")
        owners = (
            np.argmax(valid_, axis=1).astype(np.int32)
            if owner_columns is None
            else np.asarray(owner_columns, dtype=np.int32)
        )
        if (
            owners.shape != (groups.shape[0],)
            or np.any(owners < 0)
            or np.any(owners >= groups.shape[1])
            or np.any(~valid_[np.arange(groups.shape[0]), owners])
        ):
            raise ValueError("Every halo group requires one valid owner column.")
        groups = np.where(valid_, groups, -1)
        self.replica_groups = jnp.asarray(groups)
        self.valid = jnp.asarray(valid_)
        self.owner_columns = jnp.asarray(owners)
        self.replica_count = int(np.max(active)) + 1
        self.reduction_semantics = "replica-columns-left-to-right"
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-halo-plan",
                "groups": array_tree_fingerprint(groups),
                "valid": array_tree_fingerprint(valid_),
                "owners": array_tree_fingerprint(owners),
                "reduction": self.reduction_semantics,
            }
        )

    def _validate_values(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] < self.replica_count:
            raise ValueError("Halo values do not contain every planned replica.")
        return value

    def _ordered_total(self, value: Array, /) -> Array:
        safe = jnp.where(self.valid, self.replica_groups, 0)
        total = jnp.zeros(
            (self.replica_groups.shape[0],) + value.shape[1:],
            dtype=value.dtype,
        )
        for column in range(self.replica_groups.shape[1]):
            mask = self.valid[:, column].reshape(
                self.valid[:, column].shape + (1,) * (value.ndim - 1)
            )
            total = total + jnp.where(mask, value[safe[:, column]], 0.0)
        return total

    def _replace_group_values(self, value: Array, group_values: Array, /) -> Array:
        safe = jnp.where(self.valid, self.replica_groups, 0)
        result = value
        for column in range(self.replica_groups.shape[1]):
            indices = safe[:, column]
            mask = self.valid[:, column].reshape(
                self.valid[:, column].shape + (1,) * (value.ndim - 1)
            )
            delta = jnp.where(mask, group_values - result[indices], 0.0)
            result = result.at[indices].add(delta)
        return result

    def sum_contributions(self, values: ArrayLike, /) -> Array:
        value = self._validate_values(values)
        return self._replace_group_values(value, self._ordered_total(value))

    def average_replicas(self, values: ArrayLike, /) -> Array:
        value = self._validate_values(values)
        count = jnp.sum(self.valid, axis=1).reshape(
            (self.valid.shape[0],) + (1,) * (value.ndim - 1)
        )
        return self._replace_group_values(value, self._ordered_total(value) / count)

    def update_replicas(
        self, values: ArrayLike, owner_column: int | None = None, /
    ) -> Array:
        value = self._validate_values(values)
        safe = jnp.where(self.valid, self.replica_groups, 0)
        owners = self.owner_columns
        if owner_column is not None:
            owner = int(owner_column)
            if owner < 0 or owner >= self.replica_groups.shape[1]:
                raise ValueError("owner_column is out of bounds.")
            owners = jnp.full(owners.shape, owner, dtype=owners.dtype)
            owners = eqx.error_if(
                owners,
                ~jnp.all(self.valid[:, owner]),
                "owner_column must be valid for every halo group.",
            )
        owner_values = value[safe[jnp.arange(safe.shape[0]), owners]]
        return self._replace_group_values(value, owner_values)

    def update_pullback(
        self, cotangent: ArrayLike, owner_column: int | None = None, /
    ) -> Array:
        """Apply the raw dual pullback of owner-to-replica halo update."""

        value = self._validate_values(cotangent)
        safe = jnp.where(self.valid, self.replica_groups, 0)
        owners = self.owner_columns
        if owner_column is not None:
            owner = int(owner_column)
            if owner < 0 or owner >= self.replica_groups.shape[1]:
                raise ValueError("owner_column is out of bounds.")
            owners = jnp.full(owners.shape, owner, dtype=owners.dtype)
            owners = eqx.error_if(
                owners,
                ~jnp.all(self.valid[:, owner]),
                "owner_column must be valid for every halo group.",
            )
        total = self._ordered_total(value)
        result = value
        for column in range(self.replica_groups.shape[1]):
            indices = safe[:, column]
            mask = self.valid[:, column].reshape(
                self.valid[:, column].shape + (1,) * (value.ndim - 1)
            )
            result = result.at[indices].add(jnp.where(mask, -result[indices], 0.0))
        owner_indices = safe[jnp.arange(safe.shape[0]), owners]
        return result.at[owner_indices].add(total)

    def sum_pullback(self, cotangent: ArrayLike, /) -> Array:
        return self.sum_contributions(cotangent)

    def average_pullback(self, cotangent: ArrayLike, /) -> Array:
        return self.average_replicas(cotangent)


class DistributedFiniteElementConstraint(StrictModule, NonTrainableState):
    constraint: ConstraintMap
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        constraint: ConstraintMap,
        partition: PartitionedFiniteElementDofMap,
        /,
    ):
        if not isinstance(constraint, ConstraintMap):
            raise TypeError("constraint must be ConstraintMap.")
        if constraint.full_space.size != partition.dof_map.global_dof_count:
            raise ValueError("Constraint and distributed DOF dimensions do not match.")
        self.constraint = constraint
        self.partition_id = partition.partition_id


class FiniteElementPartition(StrictModule, NonTrainableState):
    """Contiguous host partition of top-dimensional cells."""

    cell_owner: Array
    part_count: int = eqx.field(static=True)
    partition_id: str = eqx.field(static=True)

    def __init__(self, cell_owner: ArrayLike, part_count: int, /):
        owner = np.asarray(cell_owner, dtype=np.int32)
        count = int(part_count)
        if owner.ndim != 1 or count <= 0 or np.any(owner < 0) or np.any(owner >= count):
            raise ValueError("Cell ownership or part_count is invalid.")
        if set(owner.tolist()) != set(range(count)):
            raise ValueError("Every partition must own at least one cell.")
        self.cell_owner = jnp.asarray(owner)
        self.part_count = count
        self.partition_id = canonical_fingerprint(
            {
                "kind": "finite-element-partition",
                "cell_owner": array_tree_fingerprint(owner),
                "part_count": count,
            }
        )


def partition_cells_contiguous(
    mesh: CellMesh,
    part_count: int,
    /,
) -> FiniteElementPartition:
    if not isinstance(mesh, CellMesh):
        raise TypeError("mesh must be CellMesh.")
    count = sum(block.cell_count for block in mesh.blocks)
    parts = int(part_count)
    if parts <= 0 or parts > count:
        raise ValueError("part_count must lie between one and the cell count.")
    owner = np.minimum(
        np.arange(count, dtype=np.int64) * parts // count,
        parts - 1,
    ).astype(np.int32)
    return FiniteElementPartition(owner, parts)


class FiniteElementPartitionWorksetPlan(StrictModule, NonTrainableState):
    """Compiler-facing owned/halo cell worksets and dependency completions."""

    owned_cells: Array
    owned_valid: Array
    halo_cells: Array
    halo_valid: Array
    dependencies: Array
    completions: Array
    partition_id: str = eqx.field(static=True)
    completion_ids: tuple[str, ...] = eqx.field(static=True)
    dependency_ids: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: FiniteElementPartition,
        owned_cells: ArrayLike,
        owned_valid: ArrayLike,
        halo_cells: ArrayLike,
        halo_valid: ArrayLike,
        dependencies: ArrayLike,
        completions: ArrayLike,
        /,
    ):
        if not isinstance(partition, FiniteElementPartition):
            raise TypeError("partition must be FiniteElementPartition.")
        owned = np.asarray(owned_cells, dtype=np.int32)
        owned_valid_ = np.asarray(owned_valid, dtype=bool)
        halo = np.asarray(halo_cells, dtype=np.int32)
        halo_valid_ = np.asarray(halo_valid, dtype=bool)
        dependency = np.asarray(dependencies, dtype=bool)
        completion = np.asarray(completions, dtype=bool)
        cell_count = np.asarray(partition.cell_owner).size
        shape = (partition.part_count, cell_count)
        if (
            owned.shape != shape
            or owned_valid_.shape != shape
            or halo.shape != shape
            or halo_valid_.shape != shape
            or dependency.shape != (partition.part_count, partition.part_count)
            or completion.shape != dependency.shape
            or not np.array_equal(completion, dependency.T)
            or np.any(np.diag(dependency))
        ):
            raise ValueError("Partition workset/dependency arrays are incompatible.")
        if (
            np.any(owned[owned_valid_] < 0)
            or np.any(owned[owned_valid_] >= cell_count)
            or np.any(halo[halo_valid_] < 0)
            or np.any(halo[halo_valid_] >= cell_count)
            or np.any(owned[~owned_valid_] != -1)
            or np.any(halo[~halo_valid_] != -1)
        ):
            raise ValueError("Partition workset routes or sentinels are invalid.")
        owners = np.asarray(partition.cell_owner)
        seen_owned = []
        for part in range(partition.part_count):
            local_owned = owned[part, owned_valid_[part]]
            local_halo = halo[part, halo_valid_[part]]
            if (
                np.unique(local_owned).size != local_owned.size
                or np.unique(local_halo).size != local_halo.size
                or np.intersect1d(local_owned, local_halo).size
                or np.any(owners[local_owned] != part)
                or np.any(owners[local_halo] == part)
            ):
                raise ValueError("Owned/halo workset membership is inconsistent.")
            required = np.zeros((partition.part_count,), dtype=bool)
            required[np.unique(owners[local_halo])] = True
            if not np.array_equal(required, dependency[part]):
                raise ValueError("Halo worksets and dependency data disagree.")
            seen_owned.extend(local_owned.tolist())
        if not np.array_equal(np.sort(np.asarray(seen_owned)), np.arange(cell_count)):
            raise ValueError("Every cell must occur in exactly one owned workset.")
        completion_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "finite-element-partition-completion",
                    "partition": partition.partition_id,
                    "producer": part,
                }
            )
            for part in range(partition.part_count)
        )
        dependency_ids = tuple(
            tuple(
                completion_ids[producer]
                for producer in range(partition.part_count)
                if dependency[consumer, producer]
            )
            for consumer in range(partition.part_count)
        )
        self.owned_cells = jnp.asarray(owned)
        self.owned_valid = jnp.asarray(owned_valid_)
        self.halo_cells = jnp.asarray(halo)
        self.halo_valid = jnp.asarray(halo_valid_)
        self.dependencies = jnp.asarray(dependency)
        self.completions = jnp.asarray(completion)
        self.partition_id = partition.partition_id
        self.completion_ids = completion_ids
        self.dependency_ids = dependency_ids
        self.part_count = partition.part_count
        self.cell_count = cell_count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-partition-worksets",
                "partition": partition.partition_id,
                "owned": array_tree_fingerprint(owned),
                "owned_valid": array_tree_fingerprint(owned_valid_),
                "halo": array_tree_fingerprint(halo),
                "halo_valid": array_tree_fingerprint(halo_valid_),
                "dependencies": array_tree_fingerprint(dependency),
                "completions": array_tree_fingerprint(completion),
                "completion_ids": list(completion_ids),
            }
        )

    def gather_owned(self, part: int, cell_values: ArrayLike, /) -> tuple[Array, Array]:
        index = int(part)
        values = jnp.asarray(cell_values)
        if index < 0 or index >= self.part_count or values.shape[0] != self.cell_count:
            raise ValueError("Owned workset partition or cell values are invalid.")
        valid = self.owned_valid[index]
        safe = jnp.where(valid, self.owned_cells[index], 0)
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values[safe], 0.0), valid

    def gather_halo(self, part: int, cell_values: ArrayLike, /) -> tuple[Array, Array]:
        index = int(part)
        values = jnp.asarray(cell_values)
        if index < 0 or index >= self.part_count or values.shape[0] != self.cell_count:
            raise ValueError("Halo workset partition or cell values are invalid.")
        valid = self.halo_valid[index]
        safe = jnp.where(valid, self.halo_cells[index], 0)
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, values[safe], 0.0), valid


def finite_element_partition_workset_plan(
    partition: FiniteElementPartition,
    facet_cells: ArrayLike,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> FiniteElementPartitionWorksetPlan:
    if not isinstance(partition, FiniteElementPartition):
        raise TypeError("partition must be FiniteElementPartition.")
    owner = np.asarray(partition.cell_owner)
    facets = np.asarray(facet_cells, dtype=np.int32)
    cell_count = owner.size
    identifiers = (
        np.arange(cell_count, dtype=np.int64)
        if cell_global_ids is None
        else np.asarray(cell_global_ids, dtype=np.int64)
    )
    if (
        facets.ndim != 2
        or facets.shape[1] != 2
        or np.any(facets < 0)
        or np.any(facets >= cell_count)
        or np.any(facets[:, 0] == facets[:, 1])
        or identifiers.shape != (cell_count,)
        or np.any(identifiers < 0)
        or np.unique(identifiers).size != cell_count
    ):
        raise ValueError("Facet adjacency or cell global IDs are invalid.")
    shape = (partition.part_count, cell_count)
    owned = np.full(shape, -1, dtype=np.int32)
    owned_valid = np.zeros(shape, dtype=bool)
    halo = np.full(shape, -1, dtype=np.int32)
    halo_valid = np.zeros(shape, dtype=bool)
    dependency = np.zeros((partition.part_count, partition.part_count), dtype=bool)
    for part in range(partition.part_count):
        local_owned = np.flatnonzero(owner == part)
        local_owned = local_owned[np.argsort(identifiers[local_owned], kind="stable")]
        halo_set: set[int] = set()
        for left, right in facets:
            if owner[left] == part and owner[right] != part:
                halo_set.add(int(right))
            if owner[right] == part and owner[left] != part:
                halo_set.add(int(left))
        local_halo = np.asarray(
            sorted(halo_set, key=lambda cell: int(identifiers[cell])),
            dtype=np.int32,
        )
        owned[part, : local_owned.size] = local_owned
        owned_valid[part, : local_owned.size] = True
        halo[part, : local_halo.size] = local_halo
        halo_valid[part, : local_halo.size] = True
        dependency[part, np.unique(owner[local_halo])] = True
    return FiniteElementPartitionWorksetPlan(
        partition,
        owned,
        owned_valid,
        halo,
        halo_valid,
        dependency,
        dependency.T,
    )


class FiniteElementFacetOwnershipPlan(StrictModule, NonTrainableState):
    """Deterministic exactly-once ownership for conforming interior facets."""

    facet_cells: Array
    facet_global_ids: Array
    facet_owner: Array
    evaluation_mask: Array
    reduction_order: Array
    partition_id: str = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    cell_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: FiniteElementPartition,
        facet_cells: ArrayLike,
        /,
        *,
        cell_global_ids: ArrayLike | None = None,
        facet_global_ids: ArrayLike | None = None,
    ):
        if not isinstance(partition, FiniteElementPartition):
            raise TypeError("partition must be FiniteElementPartition.")
        owner = np.asarray(partition.cell_owner)
        facets = np.asarray(facet_cells, dtype=np.int32)
        cell_ids = (
            np.arange(owner.size, dtype=np.int64)
            if cell_global_ids is None
            else np.asarray(cell_global_ids, dtype=np.int64)
        )
        facet_ids = (
            np.arange(facets.shape[0], dtype=np.int64)
            if facet_global_ids is None and facets.ndim == 2
            else np.asarray(facet_global_ids, dtype=np.int64)
        )
        if (
            facets.ndim != 2
            or facets.shape[0] == 0
            or facets.shape[1] != 2
            or np.any(facets < 0)
            or np.any(facets >= owner.size)
            or np.any(facets[:, 0] == facets[:, 1])
            or cell_ids.shape != owner.shape
            or np.any(cell_ids < 0)
            or np.unique(cell_ids).size != cell_ids.size
            or facet_ids.shape != (facets.shape[0],)
            or np.any(facet_ids < 0)
            or np.unique(facet_ids).size != facet_ids.size
        ):
            raise ValueError("Conforming facet ownership inputs are invalid.")
        canonical_side = np.argmin(cell_ids[facets], axis=1)
        chosen_cells = facets[np.arange(facets.shape[0]), canonical_side]
        facet_owner_ = owner[chosen_cells]
        evaluation = np.arange(partition.part_count)[:, None] == facet_owner_[None, :]
        if np.any(np.sum(evaluation, axis=0) != 1):
            raise ValueError("Every conforming facet must have exactly one evaluator.")
        order = np.argsort(facet_ids, kind="stable").astype(np.int32)
        self.facet_cells = jnp.asarray(facets)
        self.facet_global_ids = jnp.asarray(facet_ids)
        self.facet_owner = jnp.asarray(facet_owner_)
        self.evaluation_mask = jnp.asarray(evaluation)
        self.reduction_order = jnp.asarray(order)
        self.partition_id = partition.partition_id
        self.part_count = partition.part_count
        self.cell_count = owner.size
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-facet-ownership",
                "partition": partition.partition_id,
                "facets": array_tree_fingerprint(facets),
                "facet_ids": array_tree_fingerprint(facet_ids),
                "owners": array_tree_fingerprint(facet_owner_),
                "order": array_tree_fingerprint(order),
            }
        )

    def owned_by(self, part: int, /) -> Array:
        index = int(part)
        if index < 0 or index >= self.part_count:
            raise ValueError("Facet partition is out of bounds.")
        return self.evaluation_mask[index]

    def route_equal_opposite(self, facet_values: ArrayLike, /) -> Array:
        values = jnp.asarray(facet_values)
        if values.ndim == 0 or values.shape[0] != self.facet_cells.shape[0]:
            raise ValueError("Facet values do not match the ownership plan.")
        result = jnp.zeros(
            (self.cell_count,) + values.shape[1:],
            dtype=values.dtype,
        )
        for offset in range(self.facet_cells.shape[0]):
            facet = self.reduction_order[offset]
            left = self.facet_cells[facet, 0]
            right = self.facet_cells[facet, 1]
            result = result.at[left].add(values[facet])
            result = result.at[right].add(-values[facet])
        return result

    def route_partition(self, part: int, facet_values: ArrayLike, /) -> Array:
        values = jnp.asarray(facet_values)
        mask = self.owned_by(part).reshape(
            self.owned_by(part).shape + (1,) * (values.ndim - 1)
        )
        return self.route_equal_opposite(jnp.where(mask, values, 0.0))


class DistributedFiniteElementMortarPlan(StrictModule, NonTrainableState):
    """Exactly-once distributed ownership layered over serial mortar patches."""

    ownership: FiniteElementFacetOwnershipPlan
    mortars: tuple[FiniteElementMortarPlan, ...]
    facet_indices: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        ownership: FiniteElementFacetOwnershipPlan,
        mortars: tuple[FiniteElementMortarPlan, ...],
        facet_indices: ArrayLike,
        /,
    ):
        mortar_plans = tuple(mortars)
        indices = np.asarray(facet_indices, dtype=np.int32)
        if not isinstance(ownership, FiniteElementFacetOwnershipPlan):
            raise TypeError("ownership must be FiniteElementFacetOwnershipPlan.")
        if (
            not mortar_plans
            or any(not isinstance(plan, FiniteElementMortarPlan) for plan in mortar_plans)
            or indices.shape != (len(mortar_plans),)
            or np.any(indices < 0)
            or np.any(indices >= ownership.facet_cells.shape[0])
            or np.unique(indices).size != indices.size
            or len({plan.plan_id for plan in mortar_plans}) != len(mortar_plans)
        ):
            raise ValueError("Distributed mortar composition is incomplete or ambiguous.")
        self.ownership = ownership
        self.mortars = mortar_plans
        self.facet_indices = jnp.asarray(indices)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-finite-element-mortar",
                "ownership": ownership.plan_id,
                "mortars": [plan.plan_id for plan in mortar_plans],
                "facets": array_tree_fingerprint(indices),
            }
        )

    def evaluated_by(self, part: int, /) -> Array:
        return self.ownership.owned_by(part)[self.facet_indices]

    def conservative_flux_contributions(
        self,
        fluxes: tuple[ArrayLike, ...],
        /,
        *,
        part: int | None = None,
    ) -> tuple[tuple[Array, Array], ...]:
        if len(fluxes) != len(self.mortars):
            raise ValueError("Distributed mortar fluxes do not match serial patches.")
        active = (
            jnp.ones((len(self.mortars),), dtype=bool)
            if part is None
            else self.evaluated_by(part)
        )
        contributions = []
        for index, (mortar, flux) in enumerate(zip(self.mortars, fluxes, strict=True)):
            left, right = mortar.conservative_flux_contributions(flux)
            contributions.append(
                (
                    jnp.where(active[index], left, 0.0),
                    jnp.where(active[index], right, 0.0),
                )
            )
        return tuple(contributions)

    def conservation_residuals(
        self,
        fluxes: tuple[ArrayLike, ...],
        /,
        *,
        part: int | None = None,
    ) -> tuple[Array, ...]:
        return tuple(
            jnp.sum(left, axis=0) + jnp.sum(right, axis=0)
            for left, right in self.conservative_flux_contributions(fluxes, part=part)
        )


def distributed_finite_element_mortar_plan(
    ownership: FiniteElementFacetOwnershipPlan,
    mortars: tuple[FiniteElementMortarPlan, ...],
    facet_indices: ArrayLike,
    /,
) -> DistributedFiniteElementMortarPlan:
    return DistributedFiniteElementMortarPlan(ownership, mortars, facet_indices)


class JaxCollectiveBackend(StrictModule, NonTrainableState):
    """Real JAX named-axis collective reduction for pmap/shard-map execution."""

    axis_name: str = eqx.field(static=True)

    def __init__(self, axis_name: str, /):
        name = str(axis_name)
        if not name:
            raise ValueError("axis_name must be non-empty.")
        self.axis_name = name

    def sum(self, value: ArrayLike, /) -> Array:
        return jax.lax.psum(jnp.asarray(value), self.axis_name)

    def mean(self, value: ArrayLike, /) -> Array:
        return jax.lax.pmean(jnp.asarray(value), self.axis_name)


class DistributedFiniteElementOperator(StrictModule, NonTrainableState):
    """Local FE action followed by a real named-axis contribution sum."""

    local_operator: AbstractLinearOperator
    collective: JaxCollectiveBackend
    operator_id: str = eqx.field(static=True)

    def __init__(
        self,
        local_operator: AbstractLinearOperator,
        collective: JaxCollectiveBackend,
        /,
    ):
        if not isinstance(local_operator, AbstractLinearOperator) or not isinstance(
            collective, JaxCollectiveBackend
        ):
            raise TypeError(
                "Distributed operator requires local operator and JAX collective."
            )
        self.local_operator = local_operator
        self.collective = collective
        self.operator_id = canonical_fingerprint(
            {
                "kind": "distributed-finite-element-operator",
                "local_operator": local_operator.operator_id,
                "axis_name": collective.axis_name,
            }
        )

    def mv(self, value: ArrayLike, /) -> Array:
        return self.collective.sum(self.local_operator.mv(value))


__all__ = [
    "DistributedFiniteElementConstraint",
    "DistributedFiniteElementMortarPlan",
    "DistributedFiniteElementOperator",
    "FiniteElementFacetOwnershipPlan",
    "FiniteElementHaloPlan",
    "FiniteElementPartition",
    "FiniteElementPartitionWorksetPlan",
    "JaxCollectiveBackend",
    "PartitionedFiniteElementDofMap",
    "distributed_finite_element_mortar_plan",
    "finite_element_partition_workset_plan",
    "partition_cells_contiguous",
]
