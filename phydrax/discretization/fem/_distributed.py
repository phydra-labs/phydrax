#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, PartitionSpec
from jax.typing import ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...linalg import AbstractLinearOperator, ConstraintMap
from ...linalg._distributed import (
    DistributedKrylovPolicy,
    DistributedKrylovResult,
    DistributedLinearOperator,
    DistributedPairing,
    solve_distributed_pcg,
)
from ...linalg._results import LinearSolveStatus
from ...linalg.krylov._results import KrylovBreakdownStatus
from ...sparse import EdgeRelation, RowRelation, SparseLinearMap
from ...typing import checked
from .._cell_geometry_validity import cell_geometry_id
from .._distributed_field import DistributedHaloPlan, make_owner_local_field_array
from .._partition import (
    CellAdjacency,
    CellPartition,
    CellPartitionHalo,
    inherit_cell_owners,
    padded_part_table,
)
from .._periodic_topology import _identification_id
from .._spaces import DiscreteFieldSpace
from ._generic import (
    _assemble_local_operator,
    _build_finite_element_dof_routes,
    _finite_element_field_space,
    _lifted_dof_entities,
    _lifted_finite_element_dof_layout,
    _local_mass_tensor,
    _local_stiffness_tensor,
    _quotient_dof_layout,
    _validate_restored_fem_state,
    FiniteElementDiscretization,
    FiniteElementDofMap,
    FiniteElementDofSourceProjection,
    FiniteElementPlan,
)
from ._hp import FiniteElementHPLineage
from ._hp_runtime import FiniteElementHPEpoch
from ._mortar import FiniteElementMortarPlan
from ._restoration import authenticate_restored_node
from ._topology_transfer import FiniteElementFieldTransfer


if TYPE_CHECKING:
    from ...meshing._measurements import NativeExecutionRecord


class PartitionedFiniteElementDofMap(StrictModule, NonTrainableState):
    """Owned/halo view of one FE coordinate map with stable global IDs."""

    dof_map: FiniteElementDofMap
    global_ids: Array
    owned_mask: Array
    halo_mask: Array
    multiplicity: Array
    partition_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        dof_map: FiniteElementDofMap,
        global_ids: ArrayLike,
        owned_mask: ArrayLike,
        /,
        *,
        multiplicity: ArrayLike | None = None,
        partition_id: str | None = None,
    ) -> None:
        identifiers = np.asarray(global_ids, dtype=np.int64)
        owned = np.asarray(owned_mask, dtype=np.bool_)
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
            np.ones(identifiers.shape, dtype=np.float64)
            if multiplicity is None
            else np.asarray(multiplicity, dtype=np.float64)
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
    ) -> None:
        groups = np.asarray(replica_groups, dtype=np.int32)
        if groups.ndim != 2 or groups.shape[0] == 0 or groups.shape[1] < 2:
            raise ValueError("replica_groups must have shape (groups, width >= 2).")
        valid_ = (
            np.ones(groups.shape, dtype=np.bool_)
            if valid is None
            else np.asarray(valid, dtype=np.bool_)
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

    @checked
    def __init__(
        self,
        constraint: ConstraintMap,
        partition: PartitionedFiniteElementDofMap,
        /,
    ) -> None:
        if constraint.full_space.size != partition.dof_map.global_dof_count:
            raise ValueError("Constraint and distributed DOF dimensions do not match.")
        self.constraint = constraint
        self.partition_id = partition.partition_id


class FiniteElementPartitionCostEvidence(StrictModule, NonTrainableState):
    cell_costs: Array
    part_costs: Array
    imbalance_ratio: Array
    edge_cut: Array
    evidence_id: str = eqx.field(static=True)


class CostAwareFiniteElementPartition(StrictModule, NonTrainableState):
    partition: CellPartition
    evidence: FiniteElementPartitionCostEvidence
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        partition: CellPartition,
        evidence: FiniteElementPartitionCostEvidence,
        /,
    ) -> None:
        if not isinstance(partition, CellPartition) or not isinstance(
            evidence, FiniteElementPartitionCostEvidence
        ):
            raise TypeError("Cost-aware partition requires partition and evidence.")
        self.partition = partition
        self.evidence = evidence
        self.plan_id = canonical_fingerprint(
            {
                "kind": "cost-aware-finite-element-partition",
                "partition": partition.partition_id,
                "evidence": evidence.evidence_id,
            }
        )


def partition_cells_cost_aware(
    discretization: FiniteElementDiscretization,
    part_count: int,
    /,
    *,
    physics_weight: float = 1.0,
    cut_penalty: float = 0.25,
) -> CostAwareFiniteElementPartition:
    """Greedily balance shape/order work while preferring adjacent ownership."""
    if not isinstance(discretization, FiniteElementDiscretization):
        raise TypeError("discretization must be FiniteElementDiscretization.")
    parts = int(part_count)
    cell_count = sum(block.cell_count for block in discretization.mesh.blocks)
    if parts <= 0 or parts > cell_count:
        raise ValueError("part_count must lie between one and the cell count.")
    weight = float(physics_weight)
    penalty = float(cut_penalty)
    if (
        not np.isfinite(weight)
        or weight <= 0.0
        or not np.isfinite(penalty)
        or penalty < 0.0
    ):
        raise ValueError("Partition physics weight and cut penalty are invalid.")
    costs = []
    for block_index, block in enumerate(discretization.mesh.blocks):
        local_width = max(
            element.local_dof_count
            for elements in discretization.elements
            for element in (elements[block_index],)
        )
        degree = max(elements[block_index].degree for elements in discretization.elements)
        shape_factor = {
            "triangle": 1.0,
            "quadrilateral": 0.8,
            "tetrahedron": 1.5,
            "hexahedron": 1.0,
            "prism": 1.35,
            "pyramid": 1.6,
        }.get(block.cell_kind, 1.0)
        cost = weight * shape_factor * local_width * max(degree + 1, 1)
        costs.extend((cost,) * block.cell_count)
    costs_array = np.asarray(costs, dtype=np.float64)
    domain = discretization.interior_facet_domain
    left_cells = np.asarray(domain.owner_cells, dtype=np.int32)
    right_cells = np.asarray(domain.neighbor_cells, dtype=np.int32)
    adjacency = CellAdjacency(np.stack((left_cells, right_cells), axis=1), cell_count)
    offsets = np.asarray(adjacency.offsets)
    neighbors = np.asarray(adjacency.neighbors)
    owner = np.full((cell_count,), -1, dtype=np.int32)
    part_costs = np.zeros((parts,), dtype=np.float64)
    order = np.argsort(-costs_array, kind="stable")
    for part, cell in enumerate(order[:parts]):
        owner[cell] = part
        part_costs[part] += costs_array[cell]
    for cell in order[parts:]:
        adjacent = owner[neighbors[offsets[cell] : offsets[cell + 1]]]
        adjacent = adjacent[adjacent >= 0]
        locality = np.bincount(adjacent, minlength=parts)
        cut = adjacent.size - locality
        scores = (
            part_costs
            + costs_array[cell]
            + penalty * costs_array[cell] * (cut - locality)
        )
        selected = int(np.argmin(scores))
        owner[cell] = selected
        part_costs[selected] += costs_array[cell]
    partition = CellPartition(owner, parts)
    edge_cut = np.count_nonzero(owner[left_cells] != owner[right_cells])
    mean_cost = np.mean(part_costs)
    imbalance = np.max(part_costs) / mean_cost if mean_cost > 0.0 else 1.0
    evidence_id = canonical_fingerprint(
        {
            "kind": "finite-element-partition-cost-evidence",
            "cell_costs": array_tree_fingerprint(costs_array),
            "part_costs": array_tree_fingerprint(part_costs),
            "edge_cut": int(edge_cut),
            "imbalance": float(imbalance),
        }
    )
    evidence = FiniteElementPartitionCostEvidence(
        jnp.asarray(costs_array),
        jnp.asarray(part_costs),
        jnp.asarray(imbalance),
        jnp.asarray(edge_cut, dtype=jnp.int32),
        evidence_id,
    )
    return CostAwareFiniteElementPartition(partition, evidence)


class FiniteElementPartitionWorksetPlan(StrictModule, NonTrainableState):
    """Compiler-facing owned/halo cell worksets and dependency completions.

    Worksets are fixed-capacity ``(parts, capacity)`` tables padded with -1,
    where each capacity is the largest per-part owned or halo count.
    """

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

    @checked
    def __init__(
        self,
        partition: CellPartition,
        owned_cells: ArrayLike,
        owned_valid: ArrayLike,
        halo_cells: ArrayLike,
        halo_valid: ArrayLike,
        dependencies: ArrayLike,
        completions: ArrayLike,
        /,
    ) -> None:
        owned = np.asarray(owned_cells, dtype=np.int32)
        owned_valid_ = np.asarray(owned_valid, dtype=np.bool_)
        halo = np.asarray(halo_cells, dtype=np.int32)
        halo_valid_ = np.asarray(halo_valid, dtype=np.bool_)
        dependency = np.asarray(dependencies, dtype=np.bool_)
        completion = np.asarray(completions, dtype=np.bool_)
        cell_count = np.asarray(partition.cell_owner).size
        parts = partition.part_count
        if (
            owned.ndim != 2
            or owned.shape[0] != parts
            or owned_valid_.shape != owned.shape
            or halo.ndim != 2
            or halo.shape[0] != parts
            or halo_valid_.shape != halo.shape
            or dependency.shape != (parts, parts)
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
        owned_parts = np.nonzero(owned_valid_)[0]
        halo_parts = np.nonzero(halo_valid_)[0]
        local_owned = owned[owned_valid_]
        local_halo = halo[halo_valid_]
        halo_keys = _workset_keys(halo, halo_valid_, cell_count)
        if (
            np.any(owners[local_owned] != owned_parts)
            or np.any(owners[local_halo] == halo_parts)
            or np.unique(halo_keys).size != halo_keys.size
        ):
            raise ValueError("Owned/halo workset membership is inconsistent.")
        required = np.zeros((parts, parts), dtype=np.bool_)
        required[halo_parts, owners[local_halo]] = True
        if not np.array_equal(required, dependency):
            raise ValueError("Halo worksets and dependency data disagree.")
        if not np.array_equal(np.sort(local_owned), np.arange(cell_count)):
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


def _workset_keys(cells: ArrayLike, valid: ArrayLike, cell_count: int, /) -> np.ndarray:
    """Encode valid ``(part, cell)`` workset entries as unique integer keys."""
    table = np.asarray(cells, dtype=np.int64)
    mask = np.asarray(valid, dtype=np.bool_)
    return np.nonzero(mask)[0] * cell_count + table[mask]


def finite_element_partition_workset_plan(
    partition: CellPartition,
    facet_cells: ArrayLike,
    /,
    *,
    cell_global_ids: ArrayLike | None = None,
) -> FiniteElementPartitionWorksetPlan:
    if not isinstance(partition, CellPartition):
        raise TypeError("partition must be CellPartition.")
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
    halo = CellPartitionHalo(
        partition,
        CellAdjacency(facets, cell_count),
        layers=1,
        cell_global_ids=identifiers,
    )
    owned = padded_part_table(halo.owned_offsets, halo.owned_cells)
    halo_table = padded_part_table(halo.halo_offsets, halo.halo_cells)
    return FiniteElementPartitionWorksetPlan(
        partition,
        owned,
        owned >= 0,
        halo_table,
        halo_table >= 0,
        halo.dependencies,
        halo.dependencies.T,
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

    @checked
    def __init__(
        self,
        partition: CellPartition,
        facet_cells: ArrayLike,
        /,
        *,
        cell_global_ids: ArrayLike | None = None,
        facet_global_ids: ArrayLike | None = None,
    ) -> None:
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
        if self.facet_cells.shape[0] == 0:
            # No interior facets: indexing the empty ownership arrays cannot be traced.
            return result

        def route(offset: Array, current: Array) -> Array:
            facet = self.reduction_order[offset]
            left = self.facet_cells[facet, 0]
            right = self.facet_cells[facet, 1]
            current = current.at[left].add(values[facet])
            return current.at[right].add(-values[facet])

        return jax.lax.fori_loop(0, self.facet_cells.shape[0], route, result)

    def route_partition(self, part: int, facet_values: ArrayLike, /) -> Array:
        values = jnp.asarray(facet_values)
        mask = self.owned_by(part).reshape(
            self.owned_by(part).shape + (1,) * (values.ndim - 1)
        )
        return self.route_equal_opposite(jnp.where(mask, values, 0.0))


class FiniteElementDistributedPhasePlan(StrictModule, NonTrainableState):
    """Owned-local, halo, and exactly-once interface execution phases."""

    partition: CellPartition
    worksets: FiniteElementPartitionWorksetPlan
    facet_ownership: FiniteElementFacetOwnershipPlan
    phase_names: tuple[str, ...] = eqx.field(static=True)
    phase_ids: tuple[str, ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    mesh_id: str = eqx.field(static=True)

    def __init__(
        self,
        discretization: FiniteElementDiscretization,
        partition: CellPartition,
        /,
        *,
        worksets: FiniteElementPartitionWorksetPlan | None = None,
    ) -> None:
        if not isinstance(discretization, FiniteElementDiscretization) or not isinstance(
            partition, CellPartition
        ):
            raise TypeError("Distributed phases require FE discretization and partition.")
        if partition.cell_owner.size != sum(
            block.cell_count for block in discretization.mesh.blocks
        ):
            raise ValueError("Partition ownership must cover the exact FE cell count.")
        domain = discretization.interior_facet_domain
        facets = np.stack(
            (
                np.asarray(domain.owner_cells, dtype=np.int32),
                np.asarray(domain.neighbor_cells, dtype=np.int32),
            ),
            axis=-1,
        )
        required_worksets = finite_element_partition_workset_plan(
            partition,
            facets,
            cell_global_ids=np.concatenate(
                tuple(
                    np.asarray(block.global_ids) for block in discretization.mesh.blocks
                )
            ),
        )
        if worksets is None:
            worksets = required_worksets
        else:
            if (
                not isinstance(worksets, FiniteElementPartitionWorksetPlan)
                or worksets.partition_id != partition.partition_id
            ):
                raise ValueError("Supplied FE worksets must match the partition.")
            cell_count = partition.cell_owner.size
            if not np.all(
                np.isin(
                    _workset_keys(
                        required_worksets.halo_cells,
                        required_worksets.halo_valid,
                        cell_count,
                    ),
                    _workset_keys(worksets.halo_cells, worksets.halo_valid, cell_count),
                )
            ):
                raise ValueError("Supplied FE halos omit an adjacent remote cell.")
        ownership = FiniteElementFacetOwnershipPlan(
            partition,
            facets,
            cell_global_ids=np.concatenate(
                tuple(
                    np.asarray(block.global_ids) for block in discretization.mesh.blocks
                )
            ),
            facet_global_ids=np.asarray(domain.entity_indices, dtype=np.int64),
        )
        names = (
            "owned-local",
            "halo-update",
            "interface",
            "contribution-sum",
        )
        phase_ids = tuple(
            canonical_fingerprint(
                {
                    "kind": "finite-element-distributed-phase",
                    "name": name,
                    "partition": partition.partition_id,
                    "worksets": worksets.plan_id,
                }
            )
            for name in names
        )
        self.partition = partition
        self.mesh_id = discretization.mesh.mesh_id
        self.worksets = worksets
        self.facet_ownership = ownership
        self.phase_names = names
        self.phase_ids = phase_ids
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-distributed-phase-plan",
                "mesh": discretization.mesh.mesh_id,
                "geometry": array_tree_fingerprint(
                    discretization.default_runtime.coordinates
                ),
                "partition": partition.partition_id,
                "worksets": worksets.plan_id,
                "facet_ownership": ownership.plan_id,
                "phases": phase_ids,
            }
        )

    def local_contribution(self, part: int, cell_values: ArrayLike, /) -> Array:
        values, valid = self.worksets.gather_owned(part, cell_values)
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - 1))
        return jnp.sum(jnp.where(mask, values, 0.0), axis=0)

    def interface_mask(self, part: int, /) -> Array:
        return self.facet_ownership.owned_by(part)


def lower_distributed_finite_element_phases(
    discretization: FiniteElementDiscretization,
    partition: CellPartition | CostAwareFiniteElementPartition,
    /,
) -> FiniteElementDistributedPhasePlan:
    selected = (
        partition.partition
        if isinstance(partition, CostAwareFiniteElementPartition)
        else partition
    )
    return FiniteElementDistributedPhasePlan(discretization, selected)


class DistributedFiniteElementMortarPlan(StrictModule, NonTrainableState):
    """Exactly-once distributed ownership layered over serial mortar patches."""

    ownership: FiniteElementFacetOwnershipPlan
    mortars: tuple[FiniteElementMortarPlan, ...]
    facet_indices: Array
    plan_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        ownership: FiniteElementFacetOwnershipPlan,
        mortars: tuple[FiniteElementMortarPlan, ...],
        facet_indices: ArrayLike,
        /,
    ) -> None:
        mortar_plans = tuple(mortars)
        indices = np.asarray(facet_indices, dtype=np.int32)
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
            jnp.ones((len(self.mortars),), dtype=jnp.bool_)
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

    def __init__(self, axis_name: str, /) -> None:
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
    ) -> None:
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


def finite_element_dof_identity_keys(
    plan: FiniteElementPlan, dof_map: FiniteElementDofMap, /
) -> tuple[str, ...]:
    """Identify actual quotient entities and authored node/moment coordinates.

    Distributed producers must project authoritative representative charts,
    including nonresident representatives, before preparing the local DOF map.
    Closure-local representative selection is not a global-frame witness.
    """
    if dof_map.scientific_keys:
        return dof_map.scientific_keys
    mesh = plan.mesh
    if len(plan.fields) != 1:
        raise ValueError("Scientific DOF identity requires one field.")
    field = plan.fields[0]
    elements = field.resolve(mesh)
    lifted = _lifted_finite_element_dof_layout(mesh, elements, field.component_shape)
    degrees, entities, positions, _ = _lifted_dof_entities(mesh, lifted)
    periodic = mesh.periodic_topology
    rows = np.arange(lifted.global_count, dtype=np.int64)
    if periodic is not None:
        quotient = _quotient_dof_layout(
            mesh, lifted, np.asarray(dof_map.dof_coordinates)
        ).quotient
        if quotient is None:
            raise ValueError("Periodic DOF identity requires an actual quotient witness.")
        rows = quotient.representatives
    if rows.size != dof_map.global_dof_count:
        raise ValueError("Scientific DOF identity differs from the prepared layout.")
    nodal_keys, nodal_families = lifted.nodal_orbit_keys, lifted.nodal_orbit_families
    cell_node_keys = {}
    if lifted.conformity == "H1":
        from ._nodal_orbits import _labels, nodal_orbit_keys

        if nodal_keys is None and lifted.association == "entity":
            nodal_keys, nodal_families = nodal_orbit_keys(mesh, elements)
        cell_offset = 0
        for block, element in zip(mesh.blocks, elements, strict=True):
            dofs = element.entity_dofs[mesh.topological_dimension][0]
            if dofs:
                labels = _labels(element)
                reference_keys = tuple(labels[dof] for dof in dofs)
                for cell in range(block.cell_count):
                    cell_node_keys[cell_offset + cell] = (
                        reference_keys,
                        element.element_id,
                    )
            cell_offset += block.cell_count
    keys = []
    for row in rows:
        degree, entity, position = (
            int(degrees[row]),
            int(entities[row]),
            int(positions[row]),
        )
        orbit_key = None
        if periodic is None:
            identifier = int(
                np.asarray(mesh.topology.entities(degree).entity_ids)[entity]
            )
        else:
            orbit = int(np.asarray(periodic.orbits(degree)[0])[entity])
            identifier = int(
                np.asarray(periodic.quotient.entities(degree).entity_ids)[orbit]
            )
            orbit_key = periodic.entity_keys(degree)[orbit]
        reference = None
        basis = None
        if nodal_keys is not None and (degree, entity) in nodal_keys:
            reference = tuple(
                (value.numerator, value.denominator)
                for value in nodal_keys[degree, entity][position]
            )
            if nodal_families is None:
                raise ValueError("Nodal identity lacks its authored reference family.")
            basis = nodal_families[degree, entity]
        elif degree == mesh.topological_dimension and entity in cell_node_keys:
            labels, basis = cell_node_keys[entity]
            reference = tuple(
                (value.numerator, value.denominator) for value in labels[position]
            )
        elif lifted.canonical_bases is not None:
            trace = lifted.canonical_bases[degree][entity]
            basis = None if trace is None else trace.basis_id
        keys.append(
            canonical_fingerprint(
                {
                    "kind": "scientific-finite-element-dof",
                    "degree": degree,
                    "entity": identifier,
                    "orbit": orbit_key,
                    "reference": reference,
                    "reference_owner": basis,
                    "moment_position": position if reference is None else None,
                }
            )
        )
    if len(set(keys)) != len(keys):
        raise ValueError("Scientific DOF keys are not unique.")
    return tuple(keys)


class FiniteElementDofOwnershipPlan(StrictModule, NonTrainableState):
    """Projection of an authoritative scientific DOF identity/owner table.

    Keys name the canonical quotient entity, authored reference node or moment,
    and its canonical frame. They are not coordinates or closure-local row IDs.
    The producer must first project that frame into ``dof_map.cell_transforms``.
    Only the resident projection is retained; sparse halo exchange and its
    transpose operate in those canonical coefficient coordinates.
    """

    global_ids: Array
    owners: Array
    owned_mask: Array
    global_dof_count: int = eqx.field(static=True)
    dof_map_id: str = eqx.field(static=True)
    identity_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    construction_plan: FiniteElementPlan
    construction_dof_map: FiniteElementDofMap
    source_authority: FiniteElementGlobalDofOwnership
    halo: DistributedHaloPlan

    def __init__(
        self,
        plan: FiniteElementPlan,
        dof_map: FiniteElementDofMap,
        authority: FiniteElementGlobalDofOwnership,
        halo: DistributedHaloPlan,
        /,
    ) -> None:
        if type(authority) is not FiniteElementGlobalDofOwnership:
            raise TypeError(
                "DOF ownership requires its actual canonical whole-source preparation."
            )
        if (
            not dof_map.scientific_keys
            or dof_map.source_projection_id != authority.plan_id
        ):
            raise ValueError(
                "DOF ownership requires an actual whole-source projected coefficient frame."
            )
        if (
            plan.mesh.storage is None
            or plan.mesh.storage.logical_topology_id != authority.source_topology_id
            or halo.part_count != authority.part_count
        ):
            raise ValueError(
                "DOF ownership must bind the actual source lifecycle and placement."
            )
        keys = authority.scientific_keys
        resident = finite_element_dof_identity_keys(plan, dof_map)
        identifiers = np.asarray(authority.global_ids, dtype=np.int64)
        owners = np.asarray(authority.owners, dtype=np.int32)
        if (
            not halo.owner_local
            or identifiers.shape != (len(keys),)
            or owners.shape != identifiers.shape
            or len(resident) != dof_map.global_dof_count
            or len(set(keys)) != len(keys)
            or len(set(resident)) != len(resident)
            or any(not isinstance(key, str) or not key for key in (*keys, *resident))
            or np.any(identifiers < 0)
            or np.unique(identifiers).size != identifiers.size
            or np.any(owners < 0)
            or np.any(owners >= halo.part_count)
        ):
            raise ValueError("Scientific DOF identity and ownership tables are invalid.")
        rows = {key: row for row, key in enumerate(keys)}
        if any(key not in rows for key in resident):
            raise ValueError("A resident DOF lacks an authoritative scientific identity.")
        projection = np.asarray([rows[key] for key in resident], dtype=np.int64)
        local_ids = identifiers[projection]
        local_owners = owners[projection]
        owned = local_owners == int(np.asarray(halo.partition_index))
        if (
            halo.entity_count != len(keys)
            or not np.array_equal(np.asarray(halo.local_global_ids), local_ids)
            or not np.array_equal(np.asarray(halo.local_owned), owned)
            or not np.all(np.asarray(halo.local_valid))
        ):
            raise ValueError(
                "DOF halo does not match the authoritative owner projection."
            )
        for phase, permutation in enumerate(halo.permutations):
            for source, target in permutation:
                if target == int(np.asarray(halo.partition_index)):
                    received = np.asarray(halo.phase_receive_indices[phase])[
                        np.asarray(halo.phase_receive_valid[phase])
                    ]
                    if np.any(local_owners[received] != source):
                        raise ValueError("DOF halo receives from a non-owner.")
        identity = canonical_fingerprint(
            {
                "kind": "scientific-finite-element-dof-identity",
                "keys": keys,
                "ids": array_tree_fingerprint(identifiers),
            }
        )
        self.global_ids = jnp.asarray(local_ids)
        self.owners = jnp.asarray(local_owners)
        self.owned_mask = jnp.asarray(owned)
        self.global_dof_count = len(keys)
        self.dof_map_id = dof_map.dof_map_id
        self.identity_id = identity
        self.evidence_id = halo.evidence_id
        self.construction_plan = plan
        self.construction_dof_map = dof_map
        self.source_authority = authority
        self.halo = halo
        self.plan_id = canonical_fingerprint(
            {
                "identity": identity,
                "dof_map": dof_map.dof_map_id,
                "owners": array_tree_fingerprint(owners),
                "projection": array_tree_fingerprint(projection),
                "halo": halo.plan_id,
            }
        )

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        self.source_authority.validate_restored()
        self.construction_dof_map.validate_restored()
        rebuilt = FiniteElementDofOwnershipPlan(
            self.construction_plan,
            self.construction_dof_map,
            self.source_authority,
            self.halo,
        )
        _validate_restored_fem_state(self, rebuilt)


class FiniteElementGlobalDofOwnership(StrictModule, NonTrainableState):
    """Canonical prepartition DOF numbering and incident-cell ownership.

    Prepared on the actual immutable whole source, before closure extraction.
    No numerical operator or second solver is prepared. IDs enumerate the
    complete scientific key table, independently of partition or closure order.
    """

    global_ids: Array
    owners: Array
    scientific_keys: tuple[str, ...] = eqx.field(static=True)
    part_count: int = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    source_geometry_id: str = eqx.field(static=True)
    field_spec_id: str = eqx.field(static=True)
    source_cell_ids: Array
    source_plan: FiniteElementPlan
    source_partition: CellPartition
    source_field: DiscreteFieldSpace
    source_dof_map: FiniteElementDofMap
    source_to_global: Array
    global_to_source: Array
    source_block_vertex_ids: tuple[Array, ...]
    source_lifted_routes: tuple[Array, ...]
    source_lifted_frames: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: FiniteElementPlan,
        partition: CellPartition,
        /,
        *,
        source_discretization: FiniteElementDiscretization | None = None,
    ) -> None:
        mesh = plan.mesh
        if mesh.storage is not None or partition.storage_id is not None:
            raise ValueError(
                "Global DOF ownership must precede owner-local closure extraction."
            )
        if len(plan.fields) != 1:
            raise ValueError("Global DOF ownership requires one actual field.")
        cells = np.concatenate([np.asarray(block.global_ids) for block in mesh.blocks])
        cell_owners = np.asarray(partition.cell_owner, dtype=np.int32)
        if cell_owners.shape != cells.shape or np.unique(cells).size != cells.size:
            raise ValueError(
                "Global DOF ownership requires exact whole-source cell ownership."
            )
        field = plan.fields[0]
        if source_discretization is None:
            dof_map = FiniteElementDofMap(
                mesh,
                field.resolve(mesh),
                component_shape=field.component_shape,
                coordinate_spec=plan.coordinate_spec,
            )
        else:
            if (
                type(source_discretization) is not FiniteElementDiscretization
                or source_discretization.plan_id != plan.plan_id
                or source_discretization.mesh.topology_id != mesh.topology_id
                or len(source_discretization.dof_maps) != 1
            ):
                raise ValueError(
                    "Global DOF ownership can reuse only the actual immutable whole-source preparation."
                )
            dof_map = source_discretization.dof_maps[0]
        keys = finite_element_dof_identity_keys(plan, dof_map)
        owners = np.full((len(keys),), partition.part_count, dtype=np.int32)
        offset = 0
        for routes in dof_map.cell_dofs:
            rows = np.asarray(routes, dtype=np.int64)
            incident = np.broadcast_to(
                cell_owners[offset : offset + rows.shape[0], None], rows.shape
            )
            np.minimum.at(owners, rows.reshape(-1), incident.reshape(-1))
            offset += rows.shape[0]
        if offset != cells.size or np.any(owners == partition.part_count):
            raise ValueError("A scientific DOF lacks an incident owned source cell.")
        order = np.asarray(sorted(range(len(keys)), key=keys.__getitem__), dtype=np.int64)
        self.scientific_keys = tuple(keys[int(row)] for row in order)
        self.global_ids = jnp.arange(len(keys), dtype=jnp.int64)
        self.owners = jnp.asarray(owners[order])
        self.part_count = partition.part_count
        self.source_topology_id = mesh.topology_id
        self.source_geometry_id = cell_geometry_id(plan.coordinate_spec)
        self.field_spec_id = field.field_spec_id
        self.source_cell_ids = jnp.asarray(cells)
        self.source_plan = plan
        self.source_partition = partition
        self.source_field = (
            _finite_element_field_space(
                mesh,
                field,
                field.resolve(mesh),
                dof_map,
                plan.coefficient_dtype,
            )
            if source_discretization is None
            else source_discretization.field_spaces[0]
        )
        self.source_dof_map = dof_map
        self.source_to_global = jnp.asarray(np.argsort(order))
        self.global_to_source = jnp.asarray(order)
        self.source_block_vertex_ids = tuple(
            mesh.vertex_global_ids[block.vertices] for block in mesh.blocks
        )
        lifted = _lifted_finite_element_dof_layout(
            mesh, field.resolve(mesh), field.component_shape
        )
        lifted_routes, _, _ = _build_finite_element_dof_routes(
            mesh, field.resolve(mesh), lifted
        )
        self.source_lifted_routes = lifted_routes
        degrees, entities, _, _ = _lifted_dof_entities(mesh, lifted)
        frames = np.broadcast_to(
            np.eye(mesh.ambient_dimension + 1),
            (lifted.global_count, mesh.ambient_dimension + 1, mesh.ambient_dimension + 1),
        ).copy()
        if mesh.periodic_topology is not None:
            for degree in range(mesh.topological_dimension + 1):
                selected = degrees == degree
                maps = mesh.periodic_topology.orbit_isometries(degree)[entities[selected]]
                rotations = np.swapaxes(maps[:, :-1, :-1], -1, -2)
                frames[selected, :-1, :-1] = rotations
                frames[selected, :-1, -1] = -np.einsum(
                    "nij,nj->ni", rotations, maps[:, :-1, -1]
                )
        self.source_lifted_frames = jnp.asarray(frames)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "canonical-global-finite-element-dof-ownership",
                "topology": mesh.topology_id,
                "field": field.field_spec_id,
                "geometry": self.source_geometry_id,
                "field_space": self.source_field.field_space_id,
                "keys": self.scientific_keys,
                "source_dof_map": dof_map.dof_map_id,
                "source_frames": array_tree_fingerprint(frames),
                "owners": array_tree_fingerprint(owners[order]),
                "source_cells": array_tree_fingerprint(cells),
                "partition": partition.partition_id,
            }
        )

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        self.source_dof_map.validate_restored()
        partition = CellPartition(
            self.source_partition.cell_owner,
            self.source_partition.part_count,
        )
        _validate_restored_fem_state(self.source_partition, partition)
        rebuilt = FiniteElementGlobalDofOwnership(self.source_plan, partition)
        _validate_restored_fem_state(self, rebuilt)

    def supporting_cells(self, global_dof_ids: ArrayLike, /) -> Array:
        """Resolve requested scientific DOFs to actual incident source cell IDs."""
        ids = np.asarray(global_dof_ids)
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= len(self.scientific_keys))
        ):
            raise ValueError(
                "Source support requests must name actual scientific DOF IDs."
            )
        source_rows = np.asarray(self.global_to_source)[ids]
        selected = []
        offset = 0
        cells = np.asarray(self.source_cell_ids)
        for routes in self.source_dof_map.cell_dofs:
            active = np.any(np.isin(np.asarray(routes), source_rows), axis=1)
            selected.append(cells[offset : offset + routes.shape[0]][active])
            offset += routes.shape[0]
        return jnp.asarray(np.concatenate(selected), dtype=jnp.int64)

    def project_dof_map(self, plan: FiniteElementPlan, /) -> FiniteElementDofMap:
        """Project whole-source scientific routes and frames by actual cell IDs."""
        mesh = plan.mesh
        if (
            mesh.storage is None
            or mesh.storage.logical_topology_id != self.source_topology_id
        ):
            raise ValueError(
                "Source DOF projection requires source-bound canonical closure storage."
            )
        if len(plan.fields) != 1 or plan.fields[0].field_spec_id != self.field_spec_id:
            raise ValueError(
                "A closure changed the actual whole-source field reference owner."
            )
        periodic = mesh.periodic_topology
        source_periodic = self.source_plan.mesh.periodic_topology
        if (periodic is None) != (source_periodic is None):
            raise ValueError("A closure changed the actual source periodic identity.")
        if periodic is not None and source_periodic is not None:
            if _identification_id(periodic.cell) != _identification_id(
                source_periodic.cell
            ):
                raise ValueError(
                    "A closure changed the actual authored periodic isometry group."
                )
            for degree in range(mesh.topological_dimension + 1):
                source_keys = dict(
                    zip(
                        np.asarray(
                            source_periodic.quotient.entities(degree).entity_ids
                        ).tolist(),
                        source_periodic.entity_keys(degree),
                        strict=True,
                    )
                )
                for identifier, key in zip(
                    np.asarray(periodic.quotient.entities(degree).entity_ids).tolist(),
                    periodic.entity_keys(degree),
                    strict=True,
                ):
                    if source_keys.get(identifier) != key:
                        raise ValueError(
                            "A closure changed an actual persistent quotient entity or winding key."
                        )
        field = plan.fields[0]
        source = self.source_dof_map
        source_ids = np.asarray(self.source_cell_ids)
        block_offsets = np.cumsum([0] + [rows.shape[0] for rows in source.cell_dofs])
        block_indices = {name: index for index, name in enumerate(source.block_names)}
        projected_indices, projected_rows, projected_frames = [], [], []
        for block in mesh.blocks:
            index = block_indices.get(block.name)
            if index is None:
                raise ValueError("A closure block lacks its immutable source block.")
            lookup = {
                int(identifier): row
                for row, identifier in enumerate(
                    source_ids[block_offsets[index] : block_offsets[index + 1]]
                )
            }
            if any(
                int(identifier) not in lookup
                for identifier in np.asarray(block.global_ids)
            ):
                raise ValueError("A closure cell lacks its immutable source cell ID.")
            rows = np.asarray(
                [lookup[int(identifier)] for identifier in np.asarray(block.global_ids)],
                dtype=np.int64,
            )
            if not np.array_equal(
                np.asarray(mesh.vertex_global_ids)[np.asarray(block.vertices)],
                np.asarray(self.source_block_vertex_ids[index])[rows],
            ):
                raise ValueError(
                    "Closure cell reference corners differ from the actual source frame."
                )
            projected_indices.append(index)
            projected_rows.append(rows)
            lifted_rows = np.asarray(self.source_lifted_routes[index])[rows]
            projected_frames.append(self.source_lifted_frames[lifted_rows])
        source_keys = tuple(
            self.scientific_keys[int(identifier)]
            for identifier in np.asarray(self.source_to_global)
        )
        projection = FiniteElementDofSourceProjection(
            source,
            projected_rows,
            np.asarray(projected_indices),
            projected_frames,
            source_keys,
            self.plan_id,
            self,
        )
        return FiniteElementDofMap(
            mesh,
            field.resolve(mesh),
            component_shape=field.component_shape,
            coordinate_spec=plan.coordinate_spec,
            source_projection=projection,
        )

    def prepare_closures(
        self,
        plans: Sequence[FiniteElementPlan],
        /,
        *,
        message_capacity: int,
        numeric_version: str = "0",
    ) -> tuple[
        tuple[
            FiniteElementDiscretization,
            FiniteElementDofOwnershipPlan,
            DistributedHaloPlan,
        ],
        ...,
    ]:
        """Produce bounded sparse packets before distributing actual closures.

        This is the whole-source lifecycle route, not an all-gather of resident
        meshes. The storage producer supplies each canonical extracted closure.
        """
        local_plans = tuple(plans)
        if len(local_plans) != self.part_count:
            raise ValueError("Prepartition DOF projection requires every actual closure.")
        capacity = int(message_capacity)
        if capacity < 1:
            raise ValueError(
                "DOF communication requires an explicit positive message bound."
            )
        projected_maps = tuple(self.project_dof_map(plan) for plan in local_plans)
        lookup = {key: row for row, key in enumerate(self.scientific_keys)}
        global_owners = np.asarray(self.owners)
        identifiers, owners, storages = [], [], []
        for rank, (plan, dof_map) in enumerate(
            zip(local_plans, projected_maps, strict=True)
        ):
            storage = plan.mesh.storage
            if (
                storage is None
                or storage.partition_index != rank
                or storage.partition_count != self.part_count
                or storage.logical_topology_id != self.source_topology_id
            ):
                raise ValueError(
                    "DOF projection requires actual source-bound closure storage."
                )
            keys = finite_element_dof_identity_keys(plan, dof_map)
            if any(key not in lookup for key in keys):
                raise ValueError(
                    "A closure changed its authoritative quotient entity or reference frame."
                )
            rows = np.asarray([lookup[key] for key in keys], dtype=np.int64)
            identifiers.append(rows)
            owners.append(global_owners[rows])
            storages.append(storage)
        evidence = storages[0].evidence_id
        if any(storage.evidence_id != evidence for storage in storages):
            raise ValueError(
                "DOF closure storage must share actual distribution evidence."
            )
        owner_rows = [
            {
                int(identifier): row
                for row, identifier in enumerate(ids)
                if own[row] == rank
            }
            for rank, (ids, own) in enumerate(zip(identifiers, owners, strict=True))
        ]
        if sum(len(rows) for rows in owner_rows) != len(self.scientific_keys):
            raise ValueError(
                "Extracted closures do not retain every canonical DOF owner exactly once."
            )
        requests = {}
        required_width = 0
        for source in range(self.part_count):
            for target in range(self.part_count):
                if source == target:
                    continue
                rows = np.flatnonzero(owners[target] == source)
                rows = rows[np.argsort(identifiers[target][rows], kind="stable")]
                requests[source, target] = rows
                required_width = max(required_width, rows.size)
        if required_width > capacity:
            raise ValueError(
                "Actual quotient DOF halo exceeds its authored message bound."
            )
        width = max(required_width, 1)
        permutations = tuple(
            tuple(
                (rank, (rank + offset) % self.part_count)
                for rank in range(self.part_count)
            )
            for offset in range(1, self.part_count)
        )
        sends = [
            np.zeros((len(permutations), width), dtype=np.int32) for _ in local_plans
        ]
        receives = [np.zeros_like(value) for value in sends]
        send_valid = [np.zeros(value.shape, dtype=np.bool_) for value in sends]
        receive_valid = [np.zeros(value.shape, dtype=np.bool_) for value in sends]
        for phase, pairs in enumerate(permutations):
            for source, target in pairs:
                rows = requests[source, target]
                for column, row in enumerate(rows):
                    owner_row = owner_rows[source].get(int(identifiers[target][row]))
                    if owner_row is None:
                        raise ValueError(
                            "A quotient ghost has no resident canonical owner."
                        )
                    sends[source][phase, column] = owner_row
                    receives[target][phase, column] = row
                    send_valid[source][phase, column] = True
                    receive_valid[target][phase, column] = True
        prepared = tuple(
            FiniteElementDiscretization(
                plan,
                numeric_version=numeric_version,
                dof_maps=(dof_map,),
            )
            for plan, dof_map in zip(local_plans, projected_maps, strict=True)
        )
        result = []
        for rank, (plan, local) in enumerate(zip(local_plans, prepared, strict=True)):
            halo = DistributedHaloPlan(
                None,
                None,
                self.part_count,
                local_global_ids=identifiers[rank],
                local_owned=owners[rank] == rank,
                global_entity_count=len(self.scientific_keys),
                partition_index=rank,
                phase_send_indices=sends[rank],
                phase_receive_indices=receives[rank],
                phase_send_valid=send_valid[rank],
                phase_receive_valid=receive_valid[rank],
                permutations=permutations,
                evidence_id=evidence,
            )
            result.append(
                (
                    local,
                    FiniteElementDofOwnershipPlan(
                        plan,
                        local.dof_maps[0],
                        self,
                        halo,
                    ),
                    halo,
                )
            )
        return tuple(result)

    def prepare_discretizations(
        self,
        plans: Sequence[FiniteElementPlan],
        /,
        *,
        axis_name: str,
        message_capacity: int,
        numeric_version: str = "0",
    ) -> tuple[OwnerLocalFiniteElementDiscretization, ...]:
        """Publish source-derived resident FE consumers without external tables."""
        local_plans = tuple(plans)
        packets = self.prepare_closures(
            local_plans,
            message_capacity=message_capacity,
            numeric_version=numeric_version,
        )
        return tuple(
            OwnerLocalFiniteElementDiscretization(
                plan,
                halo,
                axis_name=axis_name,
                numeric_version=numeric_version,
                ownership=ownership,
                local_discretization=local,
            )
            for plan, (local, ownership, halo) in zip(local_plans, packets, strict=True)
        )


class FiniteElementExecutionLimits(StrictModule, NonTrainableState):
    """Authored upper bounds for genuine masked execution buckets, not DOF IDs."""

    maximum_dofs: int = eqx.field(static=True)
    maximum_cells: int = eqx.field(static=True)
    maximum_routes: int = eqx.field(static=True)
    maximum_message_entries: int = eqx.field(static=True)
    maximum_storage_bytes: int = eqx.field(static=True)

    def __init__(
        self,
        maximum_dofs: int,
        maximum_cells: int,
        maximum_routes: int,
        maximum_message_entries: int,
        maximum_storage_bytes: int,
        /,
    ) -> None:
        values = (
            maximum_dofs,
            maximum_cells,
            maximum_routes,
            maximum_message_entries,
            maximum_storage_bytes,
        )
        if any(type(value) is not int or value < 1 for value in values):
            raise ValueError(
                "Execution bucket bounds must be explicit positive integers."
            )
        self.maximum_dofs = maximum_dofs
        self.maximum_cells = maximum_cells
        self.maximum_routes = maximum_routes
        self.maximum_message_entries = maximum_message_entries
        self.maximum_storage_bytes = maximum_storage_bytes

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        rebuilt = FiniteElementExecutionLimits(
            self.maximum_dofs,
            self.maximum_cells,
            self.maximum_routes,
            self.maximum_message_entries,
            self.maximum_storage_bytes,
        )
        _validate_restored_fem_state(self, rebuilt)


def _execution_sparse_map(
    original: SparseLinearMap, source_size: int, target_size: int, capacity: int, /
) -> SparseLinearMap:
    relation = original.relation
    edges = relation.as_edge_relation() if isinstance(relation, RowRelation) else relation
    count = edges.capacity
    if count > capacity:
        raise ValueError("Actual sparse execution routes exceed the declared bucket.")
    sources = np.zeros((capacity,), dtype=np.int32)
    targets = np.zeros((capacity,), dtype=np.int32)
    valid = np.zeros((capacity,), dtype=np.bool_)
    coefficients = np.zeros((capacity,), dtype=np.asarray(original.coefficients).dtype)
    sources[:count] = np.asarray(edges.source_indices)
    targets[:count] = np.asarray(edges.target_indices)
    valid[:count] = np.asarray(edges.valid)
    coefficients[:count] = np.asarray(original.coefficients).reshape(-1)
    return SparseLinearMap(
        EdgeRelation(
            sources,
            targets,
            source_size=source_size,
            target_size=target_size,
            valid=valid,
        ),
        coefficients,
        properties=original.properties,
        operator_id=original.operator_id,
    )


def _execution_halo(
    original: DistributedHaloPlan, dof_capacity: int, message_capacity: int, /
) -> DistributedHaloPlan:
    count = original.local_capacity
    if count > dof_capacity or original.message_capacity > message_capacity:
        raise ValueError("Actual halo routes exceed the declared execution bucket.")
    padding = dof_capacity - count
    identifiers = np.concatenate(
        (
            np.asarray(original.local_global_ids),
            np.arange(-padding, 0, dtype=np.int64),
        )
    )
    valid = np.arange(dof_capacity) < count
    owned = np.zeros((dof_capacity,), dtype=np.bool_)
    owned[:count] = np.asarray(original.local_owned)
    width_padding = message_capacity - original.message_capacity
    pads = ((0, 0), (0, width_padding))
    return DistributedHaloPlan(
        None,
        None,
        original.part_count,
        local_global_ids=identifiers,
        local_owned=owned,
        local_valid=valid,
        global_entity_count=original.entity_count,
        partition_index=int(np.asarray(original.partition_index)),
        phase_send_indices=np.pad(np.asarray(original.phase_send_indices), pads),
        phase_receive_indices=np.pad(np.asarray(original.phase_receive_indices), pads),
        phase_send_valid=np.pad(np.asarray(original.phase_send_valid), pads),
        phase_receive_valid=np.pad(np.asarray(original.phase_receive_valid), pads),
        permutations=original.permutations,
        evidence_id=original.evidence_id,
    )


class FiniteElementClosurePreparation(StrictModule, NonTrainableState):
    """Actual scientific closure consumers plus separately ended preparation work."""

    authority: FiniteElementGlobalDofOwnership
    programs: tuple[OwnerLocalFiniteElementDiscretization, ...]
    execution_evidence: NativeExecutionRecord | None

    def __init__(
        self,
        authority: FiniteElementGlobalDofOwnership,
        programs: Sequence[OwnerLocalFiniteElementDiscretization],
        /,
        *,
        execution_evidence: NativeExecutionRecord | None = None,
    ) -> None:
        from ...meshing._measurements import NativeExecutionRecord

        consumers = tuple(programs)
        if type(authority) is not FiniteElementGlobalDofOwnership or not consumers:
            raise TypeError(
                "Closure preparation requires its actual global source authority and consumers."
            )
        ranks = []
        for program in consumers:
            if (
                type(program) is not OwnerLocalFiniteElementDiscretization
                or program.local_discretization.dof_maps[0].source_projection_id
                != authority.plan_id
                or program.halo.part_count != authority.part_count
            ):
                raise ValueError(
                    "Closure consumers must retain their actual source projections and placement."
                )
            ranks.append(int(np.asarray(program.halo.partition_index)))
        if len(set(ranks)) != len(ranks):
            raise ValueError(
                "A closure preparation may retain each resident partition only once."
            )
        if execution_evidence is not None:
            if type(execution_evidence) is not NativeExecutionRecord:
                raise TypeError(
                    "Preparation evidence must be its actual ended native record."
                )
            execution_evidence.require_valid()
            if execution_evidence.owner_id != authority.plan_id:
                raise ValueError(
                    "Preparation work must bind the actual source ownership operation."
                )
        self.authority = authority
        self.programs = consumers
        self.execution_evidence = execution_evidence

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        self.authority.validate_restored()
        for program in self.programs:
            program.validate_restored()
        rebuilt = FiniteElementClosurePreparation(
            self.authority,
            self.programs,
            execution_evidence=self.execution_evidence,
        )
        _validate_restored_fem_state(self, rebuilt)


def _owner_local_fe_certificate(
    halo: DistributedHaloPlan,
    cell_owned: Array,
    global_cell_count: int,
    metadata: Array,
    axis_name: str,
    /,
) -> Array:
    routes = halo.collective_certificate(axis_name=axis_name)
    cells = jax.lax.psum(jnp.sum(cell_owned, dtype=jnp.int64), axis_name)
    agreed = jnp.all(
        jax.lax.pmin(metadata, axis_name) == jax.lax.pmax(metadata, axis_name)
    )
    return routes & (cells == global_cell_count) & agreed


class _OwnerLocalFiniteElementAction(StrictModule, NonTrainableState):
    """Sparse owner-to-ghost input and ghost-to-owner assembled dual action."""

    local_operator: SparseLinearMap
    halo: DistributedHaloPlan
    cell_owned: Array
    collective_metadata: Array
    global_cell_count: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)

    def __call__(self, value: Array, /) -> Array:
        certified = _owner_local_fe_certificate(
            self.halo,
            self.cell_owned,
            self.global_cell_count,
            self.collective_metadata,
            self.axis_name,
        )
        values = eqx.error_if(
            value,
            ~certified,
            "Owner-local finite-element routes failed collective certification.",
        )
        part = jnp.asarray(self.halo.partition_index, dtype=jnp.int32)
        exchanged = self.halo.exchange(values, part, axis_name=self.axis_name)
        contributions = self.local_operator.mv(exchanged.reshape(-1)).reshape(
            exchanged.shape
        )
        accumulated = self.halo.accumulate_halo(
            contributions, part, axis_name=self.axis_name
        )
        owned = self.halo.local_owned.reshape((-1,) + (1,) * (values.ndim - 1))
        return jnp.where(owned, accumulated, 0)


class OwnerLocalFiniteElementSolveResult(StrictModule):
    """Numerical solve evidence, not a global mesh-geometry acceptance verdict."""

    solve: DistributedKrylovResult
    collective_certified: Array
    accepted: Array
    global_space_id: str = eqx.field(static=True)
    local_space_id: str = eqx.field(static=True)
    distribution_evidence_id: str = eqx.field(static=True)


class OwnerLocalFiniteElementDiscretization(StrictModule, NonTrainableState):
    """Conforming nodal or canonical moment FE on a CellMesh closure.

    P1 nonperiodic fields reuse vertex storage. Periodic and higher-order fields
    require an explicit scientific DOF ownership projection and canonical-frame
    halo. Cell tensors contribute only on owned cells; the assembled sparse
    action uses the DOF map's authored node permutations and moment transforms.
    """

    local_discretization: FiniteElementDiscretization
    halo: DistributedHaloPlan
    mass: SparseLinearMap
    stiffness: SparseLinearMap
    pairing: DistributedPairing
    global_ids: Array
    owned_mask: Array
    cell_owned: Array
    collective_metadata: Array
    local_dof_count: int = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)
    global_dof_count: int = eqx.field(static=True)
    global_cell_count: int = eqx.field(static=True)
    global_space_id: str = eqx.field(static=True)
    local_space_id: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    distribution_evidence_id: str = eqx.field(static=True)
    construction_plan: FiniteElementPlan
    construction_ownership: FiniteElementDofOwnershipPlan | None
    execution_source: OwnerLocalFiniteElementDiscretization | None
    execution_capacities: tuple[int, int, int, int, int] | None = eqx.field(static=True)

    def __init__(
        self,
        plan: FiniteElementPlan,
        halo: DistributedHaloPlan,
        /,
        *,
        axis_name: str,
        numeric_version: str = "0",
        ownership: FiniteElementDofOwnershipPlan | None = None,
        local_discretization: FiniteElementDiscretization | None = None,
        _execution_source: OwnerLocalFiniteElementDiscretization | None = None,
        _execution_capacities: tuple[int, int, int, int, int] | None = None,
    ) -> None:
        if _execution_source is not None:
            if (
                type(_execution_source) is not OwnerLocalFiniteElementDiscretization
                or _execution_capacities is None
                or plan.plan_id != _execution_source.construction_plan.plan_id
                or halo.plan_id != _execution_source.halo.plan_id
            ):
                raise ValueError(
                    "Execution views require their complete actual owning FE source."
                )
            self._initialize_execution_view(_execution_source, _execution_capacities)
            return
        if _execution_capacities is not None:
            raise ValueError(
                "Execution capacities require their actual owning FE source."
            )
        if not isinstance(plan, FiniteElementPlan) or not isinstance(
            halo, DistributedHaloPlan
        ):
            raise TypeError(
                "Owner-local FE requires FiniteElementPlan and DistributedHaloPlan."
            )
        mesh = plan.mesh
        storage = mesh.storage
        if storage is None or not halo.owner_local:
            raise ValueError(
                "Owner-local FE requires explicit canonical storage and local halo routes."
            )
        axis = str(axis_name).strip()
        if not axis:
            raise ValueError("axis_name must be non-empty.")
        if len(plan.fields) != 1:
            raise NotImplementedError("Owner-local FE admits one scalar field.")
        if mesh.periodic_topology is not None and ownership is None:
            raise ValueError("Periodic owner-local FE requires quotient DOF ownership.")
        field = plan.fields[0]
        elements = field.resolve(mesh)
        if any(
            block.cell_kind not in ("interval", "triangle", "tetrahedron")
            or not (
                (element.conformity == "H1" and element.mapping == "identity")
                or (
                    ownership is not None
                    and element.form_basis is not None
                    and element.conformity in ("Hdiv", "Hcurl", "HLambda")
                )
            )
            or (
                ownership is None
                and (element.degree != 1 or element.local_dof_count != block.arity)
            )
            for block, element in zip(mesh.blocks, elements, strict=True)
        ):
            raise NotImplementedError(
                "Owner-local FE requires simplex H1 or canonical moment fields."
            )
        if ownership is not None and not isinstance(
            ownership, FiniteElementDofOwnershipPlan
        ):
            raise TypeError("ownership must be FiniteElementDofOwnershipPlan or None.")
        ids = np.asarray(
            storage.entity_global_ids[0] if ownership is None else ownership.global_ids,
            dtype=np.int64,
        )
        owned = np.asarray(
            storage.entity_owned[0] if ownership is None else ownership.owned_mask,
            dtype=np.bool_,
        )
        global_count = (
            storage.global_entity_counts[0]
            if ownership is None
            else ownership.global_dof_count
        )
        if (
            halo.partition_index != storage.partition_index
            or halo.part_count != storage.partition_count
            or halo.entity_count != global_count
            or not np.array_equal(np.asarray(halo.local_global_ids), ids)
            or not np.array_equal(np.asarray(halo.local_owned), owned)
            or halo.evidence_id != storage.evidence_id
        ):
            raise ValueError(
                "FE vector routes must match canonical DOF storage and evidence exactly."
            )
        vertex_owners = np.asarray(
            storage.entity_owner[0] if ownership is None else ownership.owners,
            dtype=np.int32,
        )
        for phase, permutation in enumerate(halo.permutations):
            for source, target in permutation:
                if target == storage.partition_index:
                    rows = np.asarray(halo.phase_receive_indices[phase])[
                        np.asarray(halo.phase_receive_valid[phase])
                    ]
                    if np.any(vertex_owners[rows] != source):
                        raise ValueError(
                            "FE ghost routes disagree with canonical DOF owners."
                        )
        local = local_discretization
        if local is None:
            local = plan.prepare(numeric_version=numeric_version)
        elif not isinstance(local, FiniteElementDiscretization):
            raise TypeError(
                "local_discretization must be FiniteElementDiscretization or None."
            )
        elif local.plan_id != plan.plan_id or local.numeric_version != numeric_version:
            raise ValueError(
                "Resident preparation must bind the actual plan and numeric version."
            )
        dof_map = local.dof_maps[0]
        if dof_map.global_dof_count != ids.size:
            raise ValueError("Closure DOFs must coincide with canonical local DOF rows.")
        if ownership is None and dof_map.association != "vertex":
            raise ValueError("Nonvertex DOFs require an explicit ownership projection.")
        if ownership is not None and (
            ownership.dof_map_id != dof_map.dof_map_id
            or ownership.evidence_id != storage.evidence_id
        ):
            raise ValueError(
                "DOF ownership must bind the actual prepared canonical frame and storage."
            )
        cell_owned = storage.entity_owned[mesh.topological_dimension]
        masses = []
        stiffnesses = []
        offset = 0
        for block, geometry in zip(mesh.blocks, local.block_geometries[0], strict=True):
            flags = cell_owned[offset : offset + block.cell_count, None, None]
            masses.append(jnp.where(flags, _local_mass_tensor(geometry), 0))
            stiffnesses.append(jnp.where(flags, _local_stiffness_tensor(geometry), 0))
            offset += block.cell_count
        self.local_discretization = local
        self.halo = halo
        self.mass = _assemble_local_operator(
            dof_map,
            tuple(masses),
            "owner-local-finite-element-mass",
            positive_definite=False,
            coefficient_dtype=plan.coefficient_dtype,
        )
        self.stiffness = _assemble_local_operator(
            dof_map,
            tuple(stiffnesses),
            "owner-local-finite-element-stiffness",
            positive_definite=False,
            coefficient_dtype=plan.coefficient_dtype,
        )
        self.pairing = DistributedPairing(owned, axis_name=axis)
        self.global_ids = jnp.asarray(ids, dtype=jnp.int64)
        self.owned_mask = jnp.asarray(owned, dtype=jnp.bool_)
        self.cell_owned = jnp.asarray(cell_owned, dtype=jnp.bool_)
        self.local_dof_count = ids.size
        self.component_shape = field.component_shape
        self.global_dof_count = global_count
        self.global_cell_count = storage.global_entity_counts[mesh.topological_dimension]
        self.axis_name = axis
        self.distribution_evidence_id = storage.evidence_id
        self.global_space_id = canonical_fingerprint(
            {
                "kind": "logical-owner-local-finite-element-space",
                "topology": storage.logical_topology_id,
                "field": field.name,
                "elements": sorted({element.element_id for element in elements}),
                "field_spec": field.field_spec_id,
                "coefficient_dtype": str(plan.coefficient_dtype),
                "global_dof_count": self.global_dof_count,
                "dof_identity": None if ownership is None else ownership.identity_id,
            }
        )
        scientific_id = canonical_fingerprint(
            {
                "space": self.global_space_id,
                "geometry": storage.logical_coordinate_geometry_id,
                "global_cell_count": self.global_cell_count,
                "precision": plan.precision_policy.policy_id,
                "numeric_version": numeric_version,
            }
        )
        self.collective_metadata = jnp.asarray(
            [int(scientific_id[offset : offset + 7], 16) for offset in range(0, 64, 7)],
            dtype=jnp.int32,
        )
        self.local_space_id = canonical_fingerprint(
            {
                "kind": "local-finite-element-closure-space",
                "global": self.global_space_id,
                "ids": array_tree_fingerprint(ids),
                "owned": array_tree_fingerprint(owned),
                "local_dof_map": dof_map.dof_map_id,
                "partition": storage.partition_index,
            }
        )
        self.prepared_id = canonical_fingerprint(
            {
                "kind": "owner-local-finite-element-prepared",
                "local": local.prepared_id,
                "local_space": self.local_space_id,
                "halo": halo.plan_id,
                "axis": axis,
                "distribution": storage.evidence_id,
            }
        )
        self.construction_plan = plan
        self.construction_ownership = ownership
        self.execution_source = None
        self.execution_capacities = None

    def _initialize_execution_view(
        self,
        source: OwnerLocalFiniteElementDiscretization,
        capacities: tuple[int, int, int, int, int],
        /,
    ) -> None:
        dofs, cells, mass_routes, stiffness_routes, messages = capacities
        if (
            dofs < source.local_dof_count
            or cells < source.cell_owned.size
            or any(type(value) is not int or value < 1 for value in capacities)
        ):
            raise ValueError("Execution buckets must contain the actual source worksets.")
        scalar_count = dofs * int(np.prod(source.component_shape, dtype=np.int64))
        self.mass = _execution_sparse_map(
            source.mass, scalar_count, scalar_count, mass_routes
        )
        self.stiffness = _execution_sparse_map(
            source.stiffness, scalar_count, scalar_count, stiffness_routes
        )
        self.halo = _execution_halo(source.halo, dofs, messages)
        self.local_discretization = source.local_discretization
        self.pairing = DistributedPairing(
            self.halo.local_owned, axis_name=source.axis_name
        )
        self.global_ids = self.halo.local_global_ids
        self.owned_mask = self.halo.local_owned
        self.cell_owned = jnp.pad(source.cell_owned, (0, cells - source.cell_owned.size))
        self.collective_metadata = source.collective_metadata
        self.local_dof_count = source.local_dof_count
        self.component_shape = source.component_shape
        self.global_dof_count = source.global_dof_count
        self.global_cell_count = source.global_cell_count
        self.global_space_id = source.global_space_id
        self.local_space_id = source.local_space_id
        self.prepared_id = source.prepared_id
        self.axis_name = source.axis_name
        self.distribution_evidence_id = source.distribution_evidence_id
        self.construction_plan = source.construction_plan
        self.construction_ownership = source.construction_ownership
        self.execution_source = source
        self.execution_capacities = capacities

    @property
    def execution_vector_shape(self) -> tuple[int, ...]:
        count = (
            self.local_dof_count
            if self.execution_capacities is None
            else self.execution_capacities[0]
        )
        return (count,) + self.component_shape

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        if self.execution_source is not None:
            self.execution_source.validate_restored()
            rebuilt = OwnerLocalFiniteElementDiscretization(
                self.construction_plan,
                self.execution_source.halo,
                axis_name=self.axis_name,
                _execution_source=self.execution_source,
                _execution_capacities=self.execution_capacities,
            )
            _validate_restored_fem_state(self, rebuilt)
            return
        if self.construction_ownership is not None:
            self.construction_ownership.validate_restored()
        self.local_discretization.validate_restored()
        rebuilt = OwnerLocalFiniteElementDiscretization(
            self.construction_plan,
            self.halo,
            axis_name=self.axis_name,
            numeric_version=self.local_discretization.numeric_version,
            ownership=self.construction_ownership,
            local_discretization=self.local_discretization,
        )
        _validate_restored_fem_state(self, rebuilt)

    def collective_certificate(self, /) -> Array:
        return _owner_local_fe_certificate(
            self.halo,
            self.cell_owned,
            self.global_cell_count,
            self.collective_metadata,
            self.axis_name,
        )

    def operator(self, kind: str, /) -> DistributedLinearOperator:
        """Return a collective operator acting on owned local closure vectors."""
        if kind not in ("mass", "stiffness"):
            raise ValueError("Owner-local FE operator kind must be mass or stiffness.")
        selected = self.mass if kind == "mass" else self.stiffness
        action = _OwnerLocalFiniteElementAction(
            selected,
            self.halo,
            self.cell_owned,
            self.collective_metadata,
            self.global_cell_count,
            self.axis_name,
        )
        return DistributedLinearOperator(
            action,
            action,
            self.execution_vector_shape,
            self.execution_vector_shape,
            operator_id=canonical_fingerprint(
                {
                    "kind": kind,
                    "prepared": self.prepared_id,
                    "global_space": self.global_space_id,
                }
            ),
        )

    def solve_mass(
        self,
        right_hand_side: ArrayLike,
        policy: DistributedKrylovPolicy,
        /,
        *,
        initial: ArrayLike | None = None,
    ) -> OwnerLocalFiniteElementSolveResult:
        """Solve the SPD logical mass problem with exactly-once global pairings."""
        rhs = jnp.asarray(right_hand_side)
        if rhs.shape != self.execution_vector_shape:
            raise ValueError(
                "Owner-local FE right-hand side must match local closure DOFs."
            )
        certificate = self.collective_certificate()
        rhs = eqx.error_if(
            rhs, ~certificate, "Owner-local FE collective coverage failed."
        )
        owned = self.owned_mask.reshape((-1,) + (1,) * len(self.component_shape))
        rhs = jnp.where(owned, rhs, 0)
        initial_ = None if initial is None else jnp.asarray(initial)
        if initial_ is not None:
            if initial_.shape != rhs.shape:
                raise ValueError(
                    "Owner-local FE initial value must match local closure DOFs."
                )
            initial_ = jnp.where(owned, initial_, 0)
        result = solve_distributed_pcg(
            self.operator("mass"), rhs, self.pairing, policy, initial=initial_
        )
        accepted = self.pairing.global_all(
            certificate
            & result.converged
            & (
                (result.breakdown == int(KrylovBreakdownStatus.NONE))
                | (result.breakdown == int(KrylovBreakdownStatus.HAPPY))
            )
            & jnp.all(jnp.isfinite(result.value))
        )
        return OwnerLocalFiniteElementSolveResult(
            result,
            certificate,
            accepted,
            self.global_space_id,
            self.local_space_id,
            self.distribution_evidence_id,
        )


def prepare_owner_local_finite_element(
    plan: FiniteElementPlan,
    halo: DistributedHaloPlan,
    /,
    *,
    axis_name: str,
    numeric_version: str = "0",
    ownership: FiniteElementDofOwnershipPlan | None = None,
    local_discretization: FiniteElementDiscretization | None = None,
) -> OwnerLocalFiniteElementDiscretization:
    """Prepare a closure consumer with authoritative DOF ownership and sparse routes."""
    return OwnerLocalFiniteElementDiscretization(
        plan,
        halo,
        axis_name=axis_name,
        numeric_version=numeric_version,
        ownership=ownership,
        local_discretization=local_discretization,
    )


def _global_transfer_edges(
    transfer: FiniteElementFieldTransfer,
    source: FiniteElementGlobalDofOwnership,
    target: FiniteElementGlobalDofOwnership,
    /,
) -> tuple[EdgeRelation, np.ndarray]:
    if type(transfer) is not FiniteElementFieldTransfer:
        raise TypeError(
            "Distributed transfer requires the actual owning finite-element field artifact."
        )
    native = transfer.transfer
    if (
        native.source_topology_id != source.source_topology_id
        or native.target_topology_id != target.source_topology_id
        or transfer.geometry.source_geometry_id != source.source_geometry_id
        or transfer.geometry.target_geometry_id != target.source_geometry_id
        or transfer.source_field.field_space_id != source.source_field.field_space_id
        or transfer.target_field.field_space_id != target.source_field.field_space_id
        or not transfer.evidence.passed
        or source.part_count != target.part_count
    ):
        raise ValueError(
            "Distributed transfer must bind its actual whole-source fields and passed evidence."
        )
    if not isinstance(native.primal, SparseLinearMap):
        raise NotImplementedError(
            "Owner-row transfer projection requires the actual canonical sparse primal."
        )
    relation = native.primal.relation
    edges = relation.as_edge_relation() if isinstance(relation, RowRelation) else relation
    if (
        native.primal.batch_shape
        or edges.source_size != source.source_dof_map.global_dof_count
        or edges.target_size != target.source_dof_map.global_dof_count
    ):
        raise ValueError(
            "Transfer routes do not realize the authoritative source and target DOFs."
        )
    return edges, np.asarray(native.primal.coefficients).reshape(-1)


def owner_local_finite_element_transfer_support(
    transfer: FiniteElementFieldTransfer,
    source: FiniteElementGlobalDofOwnership,
    target: FiniteElementGlobalDofOwnership,
    /,
) -> tuple[Array, ...]:
    """Request actual source DOF support before extracting target-owner closures.

    The storage producer resolves these IDs through ``source.supporting_cells``
    under its original closure/resource bounds, before preparing routes once.
    """
    edges, _ = _global_transfer_edges(transfer, source, target)
    valid = np.asarray(edges.valid)
    source_ids = np.asarray(source.source_to_global)[
        np.asarray(edges.source_indices)[valid]
    ]
    target_ids = np.asarray(target.source_to_global)[
        np.asarray(edges.target_indices)[valid]
    ]
    target_owners = np.asarray(target.owners)[target_ids]
    return tuple(
        jnp.asarray(np.unique(source_ids[target_owners == rank]), dtype=jnp.int64)
        for rank in range(target.part_count)
    )


class OwnerLocalFiniteElementTransfer(StrictModule, NonTrainableState):
    """Exactly-once owner-row primal with sparse raw/Hermitian pullbacks.

    This is the coefficient pairing, not an invented physical Hilbert adjoint.
    No conservation claim is made independently of the canonical global transfer
    certificate and both executable ownership witnesses.
    """

    local_map: SparseLinearMap
    source_halo: DistributedHaloPlan
    target_halo: DistributedHaloPlan
    source_cells: Array
    target_cells: Array
    source_metadata: Array
    target_metadata: Array
    transfer_metadata: Array
    source_cell_count: int = eqx.field(static=True)
    target_cell_count: int = eqx.field(static=True)
    axis_name: str = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)
    field_transfer: FiniteElementFieldTransfer
    source_authority: FiniteElementGlobalDofOwnership
    target_authority: FiniteElementGlobalDofOwnership
    source_program: OwnerLocalFiniteElementDiscretization
    target_program: OwnerLocalFiniteElementDiscretization
    execution_source: OwnerLocalFiniteElementTransfer | None = None
    execution_capacities: tuple[int, int, int, int, int, int] | None = eqx.field(
        static=True, default=None
    )

    @authenticate_restored_node
    def validate_restored(self, /) -> None:
        if self.execution_source is not None:
            self.execution_source.validate_restored()
            if self.execution_capacities is None:
                raise ValueError(
                    "Execution transfer lacks its actual bucket declaration."
                )
            rebuilt = _execution_transfer(
                self.execution_source, self.execution_capacities
            )
            _validate_restored_fem_state(self, rebuilt)
            return
        self.source_program.validate_restored()
        self.target_program.validate_restored()
        self.source_authority.validate_restored()
        self.target_authority.validate_restored()
        rank = int(np.asarray(self.source_halo.partition_index))
        rebuilt = _prepare_owner_local_transfer_parts(
            self.field_transfer,
            self.source_authority,
            self.target_authority,
            (self.source_program,),
            (self.target_program,),
            (rank,),
        )[0]
        _validate_restored_fem_state(self, rebuilt)

    def collective_certificate(self, /) -> Array:
        return (
            _owner_local_fe_certificate(
                self.source_halo,
                self.source_cells,
                self.source_cell_count,
                self.source_metadata,
                self.axis_name,
            )
            & _owner_local_fe_certificate(
                self.target_halo,
                self.target_cells,
                self.target_cell_count,
                self.target_metadata,
                self.axis_name,
            )
            & jnp.all(
                jax.lax.pmin(self.transfer_metadata, self.axis_name)
                == jax.lax.pmax(self.transfer_metadata, self.axis_name)
            )
        )

    def _values(self, values: ArrayLike, count: int, /) -> Array:
        value = jnp.asarray(values)
        if value.ndim == 0 or value.shape[0] != count:
            raise ValueError(
                "Owner-local transfer values must match resident coefficient rows."
            )
        return eqx.error_if(
            value,
            ~self.collective_certificate(),
            "Owner-local transfer failed its source/target ownership certificates.",
        )

    def apply(self, values: ArrayLike, /) -> Array:
        value = self._values(values, self.local_map.source.size)
        exchanged = self.source_halo.exchange(
            value,
            self.source_halo.partition_index,
            axis_name=self.axis_name,
        )
        result = self.local_map.mv_block(exchanged.reshape((value.shape[0], -1)))
        shape = (result.shape[0],) + value.shape[1:]
        owned = self.target_halo.local_owned.reshape((-1,) + (1,) * (value.ndim - 1))
        return jnp.where(owned, result.reshape(shape), 0)

    def _pullback(self, values: ArrayLike, /, *, hermitian: bool) -> Array:
        value = self._values(values, self.local_map.target.size)
        owned = self.target_halo.local_owned.reshape((-1,) + (1,) * (value.ndim - 1))
        columns = jnp.where(owned, value, 0).reshape((value.shape[0], -1))
        contributions = (
            self.local_map.adjoint_mv_block(columns)
            if hermitian
            else self.local_map.transpose_mv_block(columns)
        ).reshape((self.local_map.source.size,) + value.shape[1:])
        accumulated = self.source_halo.accumulate_halo(
            contributions,
            self.source_halo.partition_index,
            axis_name=self.axis_name,
        )
        source_owned = self.source_halo.local_owned.reshape(
            (-1,) + (1,) * (value.ndim - 1)
        )
        return jnp.where(source_owned, accumulated, 0)

    def pullback(self, values: ArrayLike, /) -> Array:
        """Apply the actual algebraic transpose, reducing every replica dual once."""
        return self._pullback(values, hermitian=False)

    def adjoint(self, values: ArrayLike, /) -> Array:
        """Apply the conjugate transpose under the exactly-once coefficient pairing."""
        return self._pullback(values, hermitian=True)


def _prepare_owner_local_transfer_parts(
    transfer: FiniteElementFieldTransfer,
    source_authority: FiniteElementGlobalDofOwnership,
    target_authority: FiniteElementGlobalDofOwnership,
    source_programs: Sequence[OwnerLocalFiniteElementDiscretization],
    target_programs: Sequence[OwnerLocalFiniteElementDiscretization],
    partition_indices: Sequence[int],
    /,
) -> tuple[OwnerLocalFiniteElementTransfer, ...]:
    """Replay source-derived owner-row stencils for actual resident partitions."""
    edges, coefficients = _global_transfer_edges(
        transfer, source_authority, target_authority
    )
    sources, targets = tuple(source_programs), tuple(target_programs)
    if len(sources) != len(targets) or len(sources) != len(partition_indices):
        raise ValueError("Transfer source/target closure declarations are incomplete.")
    valid = np.asarray(edges.valid)
    source_ids = np.asarray(source_authority.source_to_global)[
        np.asarray(edges.source_indices)[valid]
    ]
    target_ids = np.asarray(target_authority.source_to_global)[
        np.asarray(edges.target_indices)[valid]
    ]
    values = coefficients[valid]
    owners = np.asarray(target_authority.owners)[target_ids]
    edge_capacity = max(
        int(np.count_nonzero(owners == rank))
        for rank in range(target_authority.part_count)
    )
    identity = canonical_fingerprint(
        {
            "transfer": transfer.transfer_id,
            "source": source_authority.plan_id,
            "target": target_authority.plan_id,
        }
    )
    metadata = jnp.asarray(
        [int(identity[offset : offset + 7], 16) for offset in range(0, 64, 7)],
        dtype=jnp.int32,
    )
    result = []
    for rank, source, target in zip(partition_indices, sources, targets, strict=True):
        if (
            source.local_discretization.dof_maps[0].source_projection_id
            != source_authority.plan_id
            or target.local_discretization.dof_maps[0].source_projection_id
            != target_authority.plan_id
            or int(np.asarray(source.halo.partition_index)) != rank
            or int(np.asarray(target.halo.partition_index)) != rank
            or source.axis_name != target.axis_name
        ):
            raise ValueError(
                "Transfer closures changed their actual scientific source frames or placement."
            )
        source_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(source.global_ids))
        }
        target_rows = {
            int(identifier): row
            for row, identifier in enumerate(np.asarray(target.global_ids))
        }
        selected = owners == rank
        required = source_ids[selected]
        if any(int(identifier) not in source_rows for identifier in required):
            raise ValueError(
                "Transfer source closure lacks precomputed authoritative DOF support."
            )
        count = required.size
        local_sources = np.zeros((edge_capacity,), dtype=np.int32)
        local_targets = np.zeros((edge_capacity,), dtype=np.int32)
        local_values = np.zeros((edge_capacity,), dtype=values.dtype)
        route_valid = np.arange(edge_capacity) < count
        local_sources[:count] = [source_rows[int(identifier)] for identifier in required]
        local_targets[:count] = [
            target_rows[int(identifier)] for identifier in target_ids[selected]
        ]
        local_values[:count] = values[selected]
        relation = EdgeRelation(
            local_sources,
            local_targets,
            valid=route_valid,
            source_size=source.local_dof_count,
            target_size=target.local_dof_count,
        )
        local_map = SparseLinearMap(
            relation,
            local_values,
            operator_id=canonical_fingerprint(
                {
                    "kind": "source-projected-owner-row-transfer",
                    "transfer": transfer.transfer_id,
                    "source": source.local_space_id,
                    "target": target.local_space_id,
                    "rank": rank,
                }
            ),
        )
        result.append(
            OwnerLocalFiniteElementTransfer(
                local_map,
                source.halo,
                target.halo,
                source.cell_owned,
                target.cell_owned,
                source.collective_metadata,
                target.collective_metadata,
                metadata,
                source.global_cell_count,
                target.global_cell_count,
                source.axis_name,
                canonical_fingerprint(
                    {
                        "global_transfer": transfer.transfer_id,
                        "local_map": local_map.operator_id,
                        "source_halo": source.halo.plan_id,
                        "target_halo": target.halo.plan_id,
                    }
                ),
                transfer,
                source_authority,
                target_authority,
                source,
                target,
            )
        )
    return tuple(result)


def prepare_owner_local_finite_element_transfer(
    transfer: FiniteElementFieldTransfer,
    source_authority: FiniteElementGlobalDofOwnership,
    target_authority: FiniteElementGlobalDofOwnership,
    source_programs: Sequence[OwnerLocalFiniteElementDiscretization],
    target_programs: Sequence[OwnerLocalFiniteElementDiscretization],
    /,
) -> tuple[OwnerLocalFiniteElementTransfer, ...]:
    """Project an actual global field artifact onto every declared owner partition."""
    sources, targets = tuple(source_programs), tuple(target_programs)
    if (
        len(sources) != source_authority.part_count
        or len(targets) != target_authority.part_count
    ):
        raise ValueError(
            "Transfer projection requires actual source and target owner closures."
        )
    return _prepare_owner_local_transfer_parts(
        transfer,
        source_authority,
        target_authority,
        sources,
        targets,
        tuple(range(source_authority.part_count)),
    )


def execute_owner_local_finite_element_transfer(
    transfers: Sequence[OwnerLocalFiniteElementTransfer],
    local_values: Sequence[ArrayLike],
    /,
    *,
    devices: Sequence[jax.Device],
    direction: str = "forward",
    execution_limits: FiniteElementExecutionLimits | None = None,
) -> Array:
    """Execute source-derived owner-row transfers on actual named-axis devices."""
    if direction not in ("forward", "transpose", "adjoint"):
        raise ValueError("Transfer direction must be forward, transpose, or adjoint.")

    def action(
        local: OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer,
        values: Array,
    ) -> tuple[Array, ...]:
        if not isinstance(local, OwnerLocalFiniteElementTransfer):
            raise TypeError(
                "Transfer execution requires the registered owner-row transfer."
            )
        if direction == "forward":
            return (local.apply(values),)
        if direction == "transpose":
            return (local.pullback(values),)
        return (local.adjoint(values),)

    return _map_owner_local_finite_element(
        transfers,
        (local_values,),
        action,
        devices,
        execution_limits=execution_limits,
        input_sides=("source" if direction == "forward" else "target",),
    )[0]


class OwnerLocalFiniteElementExecutionResult(StrictModule):
    """Globally sharded execution buckets, with process-local scientific identities."""

    value: Array
    status: Array
    iterations: Array
    residual_norm: Array
    breakdown: Array
    collective_certified: Array
    accepted: Array
    condition_estimate: Array | None
    global_space_id: str = eqx.field(static=True)
    local_space_ids: tuple[tuple[int, str], ...] = eqx.field(static=True)
    distribution_evidence_id: str = eqx.field(static=True)


def _execution_transfer(
    original: OwnerLocalFiniteElementTransfer,
    capacities: tuple[int, int, int, int, int, int],
    /,
) -> OwnerLocalFiniteElementTransfer:
    source_dofs, target_dofs, source_cells, target_cells, routes, messages = capacities
    if (
        source_cells < original.source_cells.size
        or target_cells < original.target_cells.size
    ):
        raise ValueError("Transfer execution buckets omit actual source/target cells.")
    return OwnerLocalFiniteElementTransfer(
        _execution_sparse_map(original.local_map, source_dofs, target_dofs, routes),
        _execution_halo(original.source_halo, source_dofs, messages),
        _execution_halo(original.target_halo, target_dofs, messages),
        jnp.pad(original.source_cells, (0, source_cells - original.source_cells.size)),
        jnp.pad(original.target_cells, (0, target_cells - original.target_cells.size)),
        original.source_metadata,
        original.target_metadata,
        original.transfer_metadata,
        original.source_cell_count,
        original.target_cell_count,
        original.axis_name,
        original.transfer_id,
        original.field_transfer,
        original.source_authority,
        original.target_authority,
        original.source_program,
        original.target_program,
        original,
        capacities,
    )


def _owner_local_execution_fields(
    program: OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer,
    /,
) -> tuple[StrictModule | Array, ...]:
    if isinstance(program, OwnerLocalFiniteElementTransfer):
        return (
            program.local_map,
            program.source_halo,
            program.target_halo,
            program.source_cells,
            program.target_cells,
            program.source_metadata,
            program.target_metadata,
            program.transfer_metadata,
        )
    return (
        program.mass,
        program.stiffness,
        program.pairing,
        program.halo,
        program.cell_owned,
        program.collective_metadata,
        program.global_ids,
        program.owned_mask,
    )


def _execution_descriptor(
    program: OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer, /
) -> Array:
    if isinstance(program, OwnerLocalFiniteElementDiscretization):
        mass = program.mass.relation
        stiffness = program.stiffness.relation
        mass_edges = mass.as_edge_relation() if isinstance(mass, RowRelation) else mass
        stiffness_edges = (
            stiffness.as_edge_relation()
            if isinstance(stiffness, RowRelation)
            else stiffness
        )
        return jnp.asarray(
            (
                program.local_dof_count,
                program.local_dof_count,
                program.cell_owned.size,
                program.cell_owned.size,
                mass_edges.capacity,
                stiffness_edges.capacity,
                program.halo.message_capacity,
            ),
            dtype=jnp.int64,
        )
    return jnp.asarray(
        (
            program.local_map.source.size,
            program.local_map.target.size,
            program.source_cells.size,
            program.target_cells.size,
            program.local_map.relation.capacity,
            program.local_map.relation.capacity,
            max(
                program.source_halo.message_capacity, program.target_halo.message_capacity
            ),
        ),
        dtype=jnp.int64,
    )


def _execution_packet_bytes(
    program: OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer,
    sizes: tuple[int, ...],
    /,
) -> int:
    source, target, source_cells, target_cells, first_routes, second_routes, messages = (
        sizes
    )
    if isinstance(program, OwnerLocalFiniteElementDiscretization):
        phases = len(program.halo.permutations)
        return (
            first_routes * (9 + np.dtype(program.mass.coefficients.dtype).itemsize)
            + second_routes
            * (9 + np.dtype(program.stiffness.coefficients.dtype).itemsize)
            + 20 * source
            + 10 * phases * messages
            + 4
            + source_cells
            + program.collective_metadata.nbytes
        )
    phases = len(program.source_halo.permutations)
    return (
        first_routes * (9 + np.dtype(program.local_map.coefficients.dtype).itemsize)
        + 10 * (source + target)
        + 20 * phases * messages
        + 8
        + source_cells
        + target_cells
        + program.source_metadata.nbytes
        + program.target_metadata.nbytes
        + program.transfer_metadata.nbytes
    )


def _prepare_execution_buckets(
    programs: tuple[
        OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer, ...
    ],
    inputs: Sequence[Sequence[ArrayLike]],
    ranks: tuple[int, ...],
    devices: tuple[jax.Device, ...],
    mesh: Mesh,
    axis: str,
    limits: FiniteElementExecutionLimits | None,
    input_sides: tuple[str, ...] | None,
    /,
) -> tuple[
    tuple[OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer, ...],
    tuple[tuple[Array, ...], ...],
]:
    descriptors = make_owner_local_field_array(
        tuple(_execution_descriptor(program) for program in programs),
        ranks,
        devices,
        axis_name=axis,
    )
    spec = PartitionSpec(axis)
    low, high = jax.shard_map(
        lambda value: (jax.lax.pmin(value[0], axis), jax.lax.pmax(value[0], axis)),
        mesh=mesh,
        in_specs=spec,
        out_specs=(PartitionSpec(), PartitionSpec()),
        check_vma=False,
    )(descriptors)
    sizes = tuple(int(value) for value in np.asarray(high))
    if limits is None:
        if not np.array_equal(np.asarray(low), np.asarray(high)):
            raise ValueError(
                "Unequal authentic execution worksets require explicit execution_limits."
            )
        return programs, tuple(
            tuple(jnp.asarray(value) for value in values) for values in inputs
        )
    limits.validate_restored()
    declared = (
        limits.maximum_dofs,
        limits.maximum_cells,
        limits.maximum_routes,
        limits.maximum_message_entries,
        limits.maximum_storage_bytes,
    )
    declaration = make_owner_local_field_array(
        tuple(jnp.asarray(declared, dtype=jnp.int64) for _ in programs),
        ranks,
        devices,
        axis_name=axis,
    )
    agreed = jax.shard_map(
        lambda value: jnp.all(
            jax.lax.pmin(value[0], axis) == jax.lax.pmax(value[0], axis)
        ),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(declaration)
    if (
        not bool(np.asarray(agreed))
        or max(sizes[:2]) > limits.maximum_dofs
        or max(sizes[2:4]) > limits.maximum_cells
        or max(sizes[4:6]) > limits.maximum_routes
        or sizes[6] > limits.maximum_message_entries
    ):
        raise ValueError(
            "Actual execution worksets exceed their original authored bounds."
        )
    sides = tuple("source" for _ in inputs) if input_sides is None else input_sides
    if len(sides) != len(inputs) or any(
        side not in ("source", "target") for side in sides
    ):
        raise ValueError(
            "Execution inputs require their actual source/target coefficient axes."
        )
    input_bytes = []
    normalized = []
    for side, values in zip(sides, inputs, strict=True):
        width = sizes[0 if side == "source" else 1]
        bucket = []
        for program, value in zip(programs, values, strict=True):
            array = jnp.asarray(value)
            actual = (
                program.local_dof_count
                if isinstance(program, OwnerLocalFiniteElementDiscretization)
                else program.local_map.source.size
                if side == "source"
                else program.local_map.target.size
            )
            if array.ndim == 0 or array.shape[0] != actual:
                raise ValueError(
                    "Execution values must match actual scientific resident rows."
                )
            input_bytes.append(
                width
                * int(np.prod(array.shape[1:], dtype=np.int64))
                * array.dtype.itemsize
            )
            bucket.append(array)
        normalized.append(tuple(bucket))
    byte_counts = make_owner_local_field_array(
        tuple(
            jnp.asarray(
                _execution_packet_bytes(program, sizes)
                + sum(
                    input_bytes[index * len(programs) + part]
                    for index in range(len(inputs))
                ),
                dtype=jnp.int64,
            )
            for part, program in enumerate(programs)
        ),
        ranks,
        devices,
        axis_name=axis,
    )
    total = jax.shard_map(
        lambda value: jax.lax.psum(value[0], axis),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(byte_counts)
    # Reserve a conservative host-staging/execution/launch coexistence envelope;
    # this is an admission bound, never a measured allocation/work receipt.
    if 3 * int(np.asarray(total)) > limits.maximum_storage_bytes:
        raise ValueError(
            "Actual coexisting execution packets exceed the authored storage bound."
        )
    prepared = []
    for program in programs:
        if isinstance(program, OwnerLocalFiniteElementDiscretization):
            prepared.append(
                OwnerLocalFiniteElementDiscretization(
                    program.construction_plan,
                    program.halo,
                    axis_name=axis,
                    _execution_source=program,
                    _execution_capacities=(
                        sizes[0],
                        sizes[2],
                        sizes[4],
                        sizes[5],
                        sizes[6],
                    ),
                )
            )
        else:
            prepared.append(
                _execution_transfer(
                    program,
                    (sizes[0], sizes[1], sizes[2], sizes[3], sizes[4], sizes[6]),
                )
            )
    padded_inputs = tuple(
        tuple(
            jnp.pad(
                array,
                ((0, sizes[0 if side == "source" else 1] - array.shape[0]),)
                + ((0, 0),) * (array.ndim - 1),
            )
            for array in values
        )
        for side, values in zip(sides, normalized, strict=True)
    )
    return tuple(prepared), padded_inputs


def _map_owner_local_finite_element(
    programs: Sequence[
        OwnerLocalFiniteElementDiscretization | OwnerLocalFiniteElementTransfer
    ],
    inputs: Sequence[Sequence[ArrayLike]],
    action: Callable[..., tuple[Array, ...]],
    devices: Sequence[jax.Device],
    /,
    *,
    execution_limits: FiniteElementExecutionLimits | None = None,
    input_sides: tuple[str, ...] | None = None,
) -> tuple[Array, ...]:
    """Launch the canonical local kernels, never canonical logical mesh payloads."""
    local_programs = tuple(programs)
    assigned = tuple(devices)
    if not local_programs or any(
        type(program)
        not in (OwnerLocalFiniteElementDiscretization, OwnerLocalFiniteElementTransfer)
        or type(program) is not type(local_programs[0])
        for program in local_programs
    ):
        raise TypeError(
            "Execution requires prepared owner-local finite-element programs."
        )
    prototype = local_programs[0]
    axis = prototype.axis_name
    halos = tuple(
        program.halo
        if isinstance(program, OwnerLocalFiniteElementDiscretization)
        else program.source_halo
        for program in local_programs
    )
    ranks = tuple(int(np.asarray(halo.partition_index)) for halo in halos)
    if any(
        program.axis_name != axis or halo.part_count != len(assigned)
        for program, halo in zip(local_programs, halos, strict=True)
    ):
        raise ValueError(
            "FE programs must match the declared real device placement and named axis."
        )
    if any(len(values) != len(local_programs) for values in inputs):
        raise ValueError(
            "Execution inputs must provide one array per process-addressable partition."
        )
    mesh = Mesh(np.asarray(assigned, dtype=object), (axis,))
    spec = PartitionSpec(axis)
    local_programs, inputs = _prepare_execution_buckets(
        local_programs,
        inputs,
        ranks,
        assigned,
        mesh,
        axis,
        execution_limits,
        input_sides,
    )
    prototype = local_programs[0]
    buckets = []
    numeric_leaves = []
    numerical, metadata = eqx.partition(
        _owner_local_execution_fields(prototype), eqx.is_array
    )
    _, structure = jax.tree_util.tree_flatten(numerical)
    for program in local_programs:
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(_owner_local_execution_fields(program), eqx.is_array)
        )
        identity = canonical_fingerprint(
            {
                "kind": "owner-local-finite-element-execution-bucket",
                "leaves": [(leaf.shape, str(leaf.dtype)) for leaf in leaves],
                "inputs": [
                    (
                        jnp.asarray(values[len(numeric_leaves)]).shape,
                        str(jnp.asarray(values[len(numeric_leaves)]).dtype),
                    )
                    for values in inputs
                ],
            }
        )
        buckets.append(
            jnp.asarray(
                [int(identity[offset : offset + 7], 16) for offset in range(0, 64, 7)],
                dtype=jnp.int32,
            )
        )
        numeric_leaves.append(leaves)
    bucket_array = make_owner_local_field_array(buckets, ranks, assigned, axis_name=axis)
    agreed = jax.shard_map(
        lambda value: jnp.all(
            jax.lax.pmin(value[0], axis) == jax.lax.pmax(value[0], axis)
        ),
        mesh=mesh,
        in_specs=spec,
        out_specs=PartitionSpec(),
        check_vma=False,
    )(bucket_array)
    if not bool(np.asarray(agreed)):
        raise ValueError(
            "FE closure/operator execution buckets differ; implicit padding is not admitted."
        )
    packed = tuple(
        make_owner_local_field_array(values, ranks, assigned, axis_name=axis)
        for values in zip(*numeric_leaves, strict=True)
    )
    input_arrays = tuple(
        make_owner_local_field_array(values, ranks, assigned, axis_name=axis)
        for values in inputs
    )

    def execute(
        local_leaves: tuple[Array, ...], *local_inputs: Array
    ) -> tuple[Array, ...]:
        fields = eqx.combine(
            jax.tree_util.tree_unflatten(
                structure, tuple(value[0] for value in local_leaves)
            ),
            metadata,
        )
        local = eqx.tree_at(_owner_local_execution_fields, prototype, fields)
        outputs = action(local, *(value[0] for value in local_inputs))
        return tuple(value[None, ...] for value in outputs)

    return jax.shard_map(
        execute,
        mesh=mesh,
        in_specs=(spec,) + (spec,) * len(input_arrays),
        out_specs=spec,
        check_vma=False,
    )(packed, *input_arrays)


def execute_owner_local_finite_element_operator(
    programs: Sequence[OwnerLocalFiniteElementDiscretization],
    local_values: Sequence[ArrayLike],
    kind: str,
    /,
    *,
    devices: Sequence[jax.Device],
    execution_limits: FiniteElementExecutionLimits | None = None,
) -> Array:
    """Apply mass/stiffness on actual process-local inputs and return sharded buckets."""
    (result,) = _map_owner_local_finite_element(
        programs,
        (local_values,),
        lambda program, value: (program.operator(kind).mv(value),),
        devices,
        execution_limits=execution_limits,
    )
    return result


def execute_owner_local_finite_element_mass(
    programs: Sequence[OwnerLocalFiniteElementDiscretization],
    right_hand_sides: Sequence[ArrayLike],
    policy: DistributedKrylovPolicy,
    /,
    *,
    devices: Sequence[jax.Device],
    initial: Sequence[ArrayLike] | None = None,
    execution_limits: FiniteElementExecutionLimits | None = None,
) -> OwnerLocalFiniteElementExecutionResult:
    """Continue a mass solve from supplied carried/checkpoint-restored local fields.

    Inputs are never reconstructed from an analytic oracle. A condition estimate
    is unavailable from the selected PCG kernel and is explicitly ``None``.
    Declared component axes and compatible moment families retain their actual
    source coefficient pairing; unused execution lanes contribute exactly zero.
    """
    local_programs = tuple(programs)
    if not isinstance(policy, DistributedKrylovPolicy):
        raise TypeError("Owner-local FE execution requires DistributedKrylovPolicy.")
    guesses = (
        tuple(jnp.zeros_like(jnp.asarray(value)) for value in right_hand_sides)
        if initial is None
        else tuple(initial)
    )

    def solve(
        program: OwnerLocalFiniteElementDiscretization, rhs: Array, guess: Array
    ) -> tuple[Array, ...]:
        result = program.solve_mass(rhs, policy, initial=guess)
        breakdown = result.solve.breakdown
        blocking = (breakdown != int(KrylovBreakdownStatus.NONE)) & (
            breakdown != int(KrylovBreakdownStatus.HAPPY)
        )
        status = jnp.where(
            result.accepted,
            int(LinearSolveStatus.SUCCESS),
            jnp.where(
                blocking,
                int(LinearSolveStatus.BREAKDOWN),
                int(LinearSolveStatus.MAXIMUM_STEPS_REACHED),
            ),
        ).astype(jnp.int32)
        status = jnp.where(
            breakdown == int(KrylovBreakdownStatus.STAGNATION),
            int(LinearSolveStatus.STAGNATION),
            status,
        ).astype(jnp.int32)
        status = jnp.where(
            breakdown == int(KrylovBreakdownStatus.NONFINITE_ACTION),
            int(LinearSolveStatus.NONFINITE_OUTPUT),
            status,
        ).astype(jnp.int32)
        return (
            result.solve.value,
            status,
            result.solve.iterations,
            result.solve.residual_norm,
            breakdown,
            result.collective_certified,
            result.accepted,
        )

    values = _map_owner_local_finite_element(
        local_programs,
        (right_hand_sides, guesses),
        solve,
        devices,
        execution_limits=execution_limits,
    )
    prototype = local_programs[0]
    return OwnerLocalFiniteElementExecutionResult(
        value=values[0],
        status=values[1],
        iterations=values[2],
        residual_norm=values[3],
        breakdown=values[4],
        collective_certified=values[5],
        accepted=values[6],
        condition_estimate=None,
        global_space_id=prototype.global_space_id,
        local_space_ids=tuple(
            (int(np.asarray(program.halo.partition_index)), program.local_space_id)
            for program in local_programs
        ),
        distribution_evidence_id=prototype.distribution_evidence_id,
    )


class FiniteElementHPPartitionPlan(StrictModule, NonTrainableState):
    """Inherited hp cell ownership, adaptive halos, and mortar dependencies."""

    cell_owner_by_slot: Array
    partition: CellPartition
    worksets: FiniteElementPartitionWorksetPlan
    interface_owner_part: Array
    mortar_dependencies: Array
    epoch_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        epoch: FiniteElementHPEpoch,
        cell_owner_by_slot: ArrayLike,
        part_count: int,
        /,
    ) -> None:
        owners = np.asarray(cell_owner_by_slot, dtype=np.int32)
        active = np.asarray(epoch.topology.active)
        parts = int(part_count)
        if (
            owners.shape != (epoch.topology.capacity,)
            or parts <= 0
            or np.any(owners[active] < 0)
            or np.any(owners[active] >= parts)
            or np.any(owners[~active] != -1)
        ):
            raise ValueError("Adaptive hp ownership or inactive sentinels are invalid.")
        active_slots = np.asarray(epoch.active_cell_slots, dtype=np.int32)
        active_owners = owners[active_slots]
        partition = CellPartition(active_owners, parts)
        slot_to_cell = np.full((epoch.topology.capacity,), -1, dtype=np.int32)
        slot_to_cell[active_slots] = np.arange(active_slots.size, dtype=np.int32)
        valid = np.asarray(epoch.interfaces.valid)
        interface_owners = np.asarray(epoch.interfaces.owner_slots)[valid]
        interface_neighbors = np.asarray(epoch.interfaces.neighbor_slots)[valid]
        interior = interface_neighbors >= 0
        facets = np.stack(
            (
                slot_to_cell[interface_owners[interior]],
                slot_to_cell[interface_neighbors[interior]],
            ),
            axis=1,
        )
        worksets = finite_element_partition_workset_plan(
            partition,
            facets,
            cell_global_ids=np.asarray(epoch.topology.cell_global_ids)[active_slots],
        )
        interface_owner_part = owners[interface_owners]
        paired_parts = np.where(
            interior,
            owners[np.maximum(interface_neighbors, 0)],
            interface_owner_part,
        )
        interface_owner_part = np.minimum(interface_owner_part, paired_parts)
        dependencies = np.zeros((parts, parts), dtype=np.bool_)
        left = owners[interface_owners[interior]]
        right = owners[interface_neighbors[interior]]
        crossing = left != right
        dependencies[left[crossing], right[crossing]] = True
        dependencies[right[crossing], left[crossing]] = True
        self.cell_owner_by_slot = jnp.asarray(owners)
        self.partition = partition
        self.worksets = worksets
        self.interface_owner_part = jnp.asarray(interface_owner_part)
        self.mortar_dependencies = jnp.asarray(dependencies)
        self.epoch_id = epoch.epoch_id
        self.topology_id = epoch.topology.topology_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-partition",
                "epoch": epoch.epoch_id,
                "topology": epoch.topology.topology_id,
                "owners": array_tree_fingerprint(owners),
                "worksets": worksets.plan_id,
                "interface_owner_part": array_tree_fingerprint(interface_owner_part),
                "mortar_dependencies": array_tree_fingerprint(dependencies),
            }
        )


def inherit_finite_element_hp_ownership(
    source: FiniteElementHPPartitionPlan,
    target: FiniteElementHPEpoch,
    lineage: FiniteElementHPLineage,
    /,
) -> FiniteElementHPPartitionPlan:
    """Inherit child ownership and require coarsening siblings to share one owner."""

    if (
        lineage.source_topology_id != source.topology_id
        or lineage.target_topology_id != target.topology.topology_id
        or lineage.source_capacity != source.cell_owner_by_slot.shape[0]
        or lineage.target_capacity != target.topology.capacity
    ):
        raise ValueError(
            "hp ownership lineage does not match source and target topologies."
        )
    source_owners = np.asarray(source.cell_owner_by_slot)
    valid = np.asarray(lineage.valid, dtype=np.bool_)
    source_slots = np.asarray(lineage.source_slots, dtype=np.int64)[valid]
    target_slots = np.asarray(lineage.target_slots, dtype=np.int64)[valid]
    unowned = source_owners[source_slots] < 0
    inherited, distinct = inherit_cell_owners(
        source_owners,
        source_slots[~unowned],
        target_slots[~unowned],
        target.topology.capacity,
    )
    reads_unowned = np.zeros((target.topology.capacity,), dtype=np.bool_)
    reads_unowned[target_slots[unowned]] = True
    active = np.asarray(target.topology.active, dtype=np.bool_)
    if np.any(active & ((distinct != 1) | reads_unowned)):
        raise ValueError(
            "Every active hp target cell must inherit exactly one valid owner."
        )
    return FiniteElementHPPartitionPlan(
        target,
        np.where(active, inherited, -1),
        source.partition.part_count,
    )


__all__ = [
    "CostAwareFiniteElementPartition",
    "DistributedFiniteElementConstraint",
    "DistributedFiniteElementMortarPlan",
    "DistributedFiniteElementOperator",
    "FiniteElementDistributedPhasePlan",
    "FiniteElementFacetOwnershipPlan",
    "FiniteElementHaloPlan",
    "FiniteElementHPPartitionPlan",
    "FiniteElementPartitionCostEvidence",
    "FiniteElementPartitionWorksetPlan",
    "JaxCollectiveBackend",
    "PartitionedFiniteElementDofMap",
    "OwnerLocalFiniteElementDiscretization",
    "OwnerLocalFiniteElementSolveResult",
    "OwnerLocalFiniteElementExecutionResult",
    "distributed_finite_element_mortar_plan",
    "execute_owner_local_finite_element_mass",
    "execute_owner_local_finite_element_operator",
    "inherit_finite_element_hp_ownership",
    "lower_distributed_finite_element_phases",
    "finite_element_partition_workset_plan",
    "partition_cells_cost_aware",
    "prepare_owner_local_finite_element",
]
