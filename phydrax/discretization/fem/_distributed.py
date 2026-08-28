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
        left_ = jnp.asarray(left)
        right_ = jnp.asarray(right)
        if left_.shape != right_.shape or left_.shape[0] != self.dof_map.global_dof_count:
            raise ValueError("Distributed inner-product arrays have invalid shape.")
        weights = self.multiplicity.reshape(
            self.multiplicity.shape + (1,) * (left_.ndim - 1)
        )
        owned = self.owned_mask.reshape(self.owned_mask.shape + (1,) * (left_.ndim - 1))
        return jnp.sum(jnp.where(owned, jnp.conj(left_) * right_ / weights, 0.0))


class FiniteElementHaloPlan(StrictModule, NonTrainableState):
    """Replica groups implementing update, sum, and average halo semantics."""

    replica_groups: Array
    valid: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        replica_groups: ArrayLike,
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        groups = np.asarray(replica_groups, dtype=np.int32)
        if groups.ndim != 2 or groups.shape[1] < 2:
            raise ValueError("replica_groups must have shape (groups, width >= 2).")
        valid_ = (
            np.ones(groups.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if valid_.shape != groups.shape or np.any(groups[valid_] < 0):
            raise ValueError("Halo routes or validity mask are invalid.")
        self.replica_groups = jnp.asarray(groups)
        self.valid = jnp.asarray(valid_)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-halo-plan",
                "groups": array_tree_fingerprint(groups),
                "valid": array_tree_fingerprint(valid_),
            }
        )

    def sum_contributions(self, values: ArrayLike, /) -> Array:
        value = jnp.asarray(values)
        safe = jnp.where(self.valid, self.replica_groups, 0)
        gathered = value[safe]
        mask = self.valid.reshape(self.valid.shape + (1,) * (value.ndim - 1))
        total = jnp.sum(jnp.where(mask, gathered, 0.0), axis=1)
        result = value
        for column in range(self.replica_groups.shape[1]):
            indices = safe[:, column]
            result = result.at[indices].set(
                jnp.where(
                    self.valid[:, column].reshape(
                        self.valid[:, column].shape + (1,) * (value.ndim - 1)
                    ),
                    total,
                    result[indices],
                )
            )
        return result

    def average_replicas(self, values: ArrayLike, /) -> Array:
        summed = self.sum_contributions(values)
        safe = jnp.where(self.valid, self.replica_groups, 0)
        count = jnp.sum(self.valid, axis=1)
        result = summed
        for column in range(self.replica_groups.shape[1]):
            indices = safe[:, column]
            divisor = count.reshape(count.shape + (1,) * (summed.ndim - 1))
            result = result.at[indices].set(
                jnp.where(
                    self.valid[:, column].reshape(
                        self.valid[:, column].shape + (1,) * (summed.ndim - 1)
                    ),
                    summed[indices] / divisor,
                    result[indices],
                )
            )
        return result

    def update_replicas(self, values: ArrayLike, owner_column: int = 0, /) -> Array:
        value = jnp.asarray(values)
        owner = int(owner_column)
        if owner < 0 or owner >= self.replica_groups.shape[1]:
            raise ValueError("owner_column is out of bounds.")
        safe = jnp.where(self.valid, self.replica_groups, 0)
        owner_values = value[safe[:, owner]]
        result = value
        for column in range(self.replica_groups.shape[1]):
            indices = safe[:, column]
            result = result.at[indices].set(
                jnp.where(
                    self.valid[:, column].reshape(
                        self.valid[:, column].shape + (1,) * (value.ndim - 1)
                    ),
                    owner_values,
                    result[indices],
                )
            )
        return result


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
    "FiniteElementPartition",
    "DistributedFiniteElementOperator",
    "JaxCollectiveBackend",
    "FiniteElementHaloPlan",
    "PartitionedFiniteElementDofMap",
    "partition_cells_contiguous",
]
