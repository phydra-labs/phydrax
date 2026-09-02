#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

import phydrax.ein as ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FiniteElementHPCellKind = Literal["quadrilateral", "hexahedron"]
FiniteElementHPLineageKind = Literal["unchanged", "refinement", "coarsening"]
FiniteElementHPTransferKind = Literal["p", "h-refinement", "h-coarsening"]

_LINEAGE_CODES = {"unchanged": 0, "refinement": 1, "coarsening": 2}


class FiniteElementHPTopology(StrictModule, NonTrainableState):
    """Fixed-capacity quad/hex refinement forest with anisotropic leaf degrees."""

    cell_global_ids: Array
    allocated: Array
    active: Array
    cell_degrees: Array
    root_cell_ids: Array
    path_codes: Array
    levels: Array
    parent_slots: Array
    child_slots: Array
    child_valid: Array
    cell_kind: FiniteElementHPCellKind = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    child_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: FiniteElementHPCellKind,
        topology_id: str,
        cell_global_ids: ArrayLike,
        allocated: ArrayLike,
        active: ArrayLike,
        cell_degrees: ArrayLike,
        /,
        *,
        root_cell_ids: ArrayLike | None = None,
        path_codes: ArrayLike | None = None,
        levels: ArrayLike | None = None,
        parent_slots: ArrayLike | None = None,
        child_slots: ArrayLike | None = None,
        child_valid: ArrayLike | None = None,
    ):
        kind = cell_kind
        identifier = str(topology_id)
        identifiers = np.asarray(cell_global_ids, dtype=np.int64)
        allocated_ = np.asarray(allocated, dtype=bool)
        active_ = np.asarray(active, dtype=bool)
        degrees = np.asarray(cell_degrees, dtype=np.int32)
        dimension = 2 if kind == "quadrilateral" else 3 if kind == "hexahedron" else 0
        children_per_parent = 2**dimension if dimension else 0
        if not identifier or dimension == 0:
            raise ValueError(
                "hp topology requires a quad/hex kind and non-empty identity."
            )
        if (
            identifiers.ndim != 1
            or identifiers.size == 0
            or allocated_.shape != identifiers.shape
            or active_.shape != identifiers.shape
            or degrees.shape != (identifiers.size, dimension)
            or np.any(active_ & ~allocated_)
            or not np.any(active_)
        ):
            raise ValueError("hp topology arrays have incompatible fixed capacities.")
        capacity = identifiers.size
        roots = (
            np.where(allocated_, identifiers, -1)
            if root_cell_ids is None
            else np.asarray(root_cell_ids, dtype=np.int64)
        )
        paths = (
            np.where(allocated_, 1, -1)
            if path_codes is None
            else np.asarray(path_codes, dtype=np.int64)
        )
        levels_ = (
            np.where(allocated_, 0, -1).astype(np.int32)
            if levels is None
            else np.asarray(levels, dtype=np.int32)
        )
        parents = (
            np.full((capacity,), -1, dtype=np.int32)
            if parent_slots is None
            else np.asarray(parent_slots, dtype=np.int32)
        )
        children = (
            np.full((capacity, children_per_parent), -1, dtype=np.int32)
            if child_slots is None
            else np.asarray(child_slots, dtype=np.int32)
        )
        children_valid = (
            np.zeros((capacity, children_per_parent), dtype=bool)
            if child_valid is None
            else np.asarray(child_valid, dtype=bool)
        )
        if (
            roots.shape != identifiers.shape
            or paths.shape != identifiers.shape
            or levels_.shape != identifiers.shape
            or parents.shape != identifiers.shape
            or children.shape != (capacity, children_per_parent)
            or children_valid.shape != children.shape
        ):
            raise ValueError("hp refinement-forest arrays have incompatible shapes.")
        if (
            np.any(identifiers[allocated_] < 0)
            or np.unique(identifiers[allocated_]).size != np.count_nonzero(allocated_)
            or np.any(identifiers[~allocated_] != -1)
            or np.any(degrees[active_] < 1)
            or np.any(degrees[~active_] != 0)
            or np.any(roots[allocated_] < 0)
            or np.any(paths[allocated_] < 1)
            or np.any(levels_[allocated_] < 0)
            or np.any(roots[~allocated_] != -1)
            or np.any(paths[~allocated_] != -1)
            or np.any(levels_[~allocated_] != -1)
            or np.any(parents[~allocated_] != -1)
            or np.any(children[~children_valid] != -1)
            or np.any(children[children_valid] < 0)
            or np.any(children[children_valid] >= capacity)
        ):
            raise ValueError("hp forest identities, degrees, or sentinels are invalid.")
        tree_pairs = np.stack((roots[allocated_], paths[allocated_]), axis=1)
        if np.unique(tree_pairs, axis=0).shape[0] != tree_pairs.shape[0]:
            raise ValueError("Allocated hp cells require unique stable tree identities.")
        for slot in np.flatnonzero(allocated_):
            parent = int(parents[slot])
            if parent >= 0:
                if (
                    parent >= capacity
                    or not allocated_[parent]
                    or levels_[slot] != levels_[parent] + 1
                    or roots[slot] != roots[parent]
                ):
                    raise ValueError("hp parent routes or tree identities are invalid.")
            local_children = children[slot, children_valid[slot]]
            if local_children.size:
                if (active_[slot] and np.any(active_[local_children])) or np.unique(
                    local_children
                ).size != local_children.size:
                    raise ValueError(
                        "An active hp leaf cannot own active allocated children."
                    )
                if np.any(parents[local_children] != slot):
                    raise ValueError("hp parent and child routes disagree.")
                ordinals = np.flatnonzero(children_valid[slot])
                expected_paths = paths[slot] * children_per_parent + ordinals + 1
                if np.any(paths[local_children] != expected_paths):
                    raise ValueError("hp child path codes are not canonical.")
        self.cell_global_ids = jnp.asarray(identifiers)
        self.allocated = jnp.asarray(allocated_)
        self.active = jnp.asarray(active_)
        self.cell_degrees = jnp.asarray(degrees)
        self.root_cell_ids = jnp.asarray(roots)
        self.path_codes = jnp.asarray(paths)
        self.levels = jnp.asarray(levels_)
        self.parent_slots = jnp.asarray(parents)
        self.child_slots = jnp.asarray(children)
        self.child_valid = jnp.asarray(children_valid)
        self.cell_kind = kind
        self.topology_id = identifier
        self.capacity = capacity
        self.dimension = dimension
        self.child_capacity = children_per_parent
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-topology",
                "cell_kind": kind,
                "topology": identifier,
                "cell_ids": array_tree_fingerprint(identifiers),
                "allocated": array_tree_fingerprint(allocated_),
                "active": array_tree_fingerprint(active_),
                "degrees": array_tree_fingerprint(degrees),
                "roots": array_tree_fingerprint(roots),
                "paths": array_tree_fingerprint(paths),
                "levels": array_tree_fingerprint(levels_),
                "parents": array_tree_fingerprint(parents),
                "children": array_tree_fingerprint(children),
                "child_valid": array_tree_fingerprint(children_valid),
            }
        )

    @property
    def active_count(self) -> int:
        return int(np.count_nonzero(np.asarray(self.active)))

    @property
    def allocated_count(self) -> int:
        return int(np.count_nonzero(np.asarray(self.allocated)))

    def stable_tree_ids(self, /) -> Array:
        return jnp.stack((self.root_cell_ids, self.path_codes), axis=1)


class FiniteElementHPLineage(StrictModule, NonTrainableState):
    """Fixed-capacity unchanged/refinement/coarsening edges between topologies."""

    source_slots: Array
    target_slots: Array
    relation_codes: Array
    valid: Array
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    lineage_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        source_capacity: int,
        target_capacity: int,
        source_slots: ArrayLike,
        target_slots: ArrayLike,
        relations: tuple[FiniteElementHPLineageKind, ...],
        /,
        *,
        valid: ArrayLike | None = None,
    ):
        source_id = str(source_topology_id)
        target_id = str(target_topology_id)
        source_count = int(source_capacity)
        target_count = int(target_capacity)
        source = np.asarray(source_slots, dtype=np.int32)
        target = np.asarray(target_slots, dtype=np.int32)
        relation_names = tuple(str(value) for value in relations)
        valid_ = (
            np.ones(source.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if (
            not source_id
            or not target_id
            or source_count <= 0
            or target_count <= 0
            or source.ndim != 1
            or target.shape != source.shape
            or valid_.shape != source.shape
            or len(relation_names) != source.size
            or any(value not in _LINEAGE_CODES for value in relation_names)
        ):
            raise ValueError(
                "hp lineage identity, capacity, or relation data is invalid."
            )
        codes = np.asarray(
            [_LINEAGE_CODES[value] for value in relation_names], dtype=np.int8
        )
        if source_id == target_id and np.any(
            codes[valid_] != _LINEAGE_CODES["unchanged"]
        ):
            raise ValueError(
                "Refinement/coarsening lineage requires a new topology identity."
            )
        if (
            np.any(source[valid_] < 0)
            or np.any(source[valid_] >= source_count)
            or np.any(target[valid_] < 0)
            or np.any(target[valid_] >= target_count)
            or np.any(source[~valid_] != -1)
            or np.any(target[~valid_] != -1)
        ):
            raise ValueError("hp lineage routes or inactive sentinels are invalid.")
        active_edges = set(
            zip(source[valid_].tolist(), target[valid_].tolist(), strict=True)
        )
        if len(active_edges) != np.count_nonzero(valid_):
            raise ValueError("hp lineage must not contain duplicate source/target edges.")
        self.source_slots = jnp.asarray(source)
        self.target_slots = jnp.asarray(target)
        self.relation_codes = jnp.asarray(codes)
        self.valid = jnp.asarray(valid_)
        self.source_topology_id = source_id
        self.target_topology_id = target_id
        self.source_capacity = source_count
        self.target_capacity = target_count
        self.lineage_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-lineage",
                "source": source_id,
                "target": target_id,
                "source_capacity": source_count,
                "target_capacity": target_count,
                "source_slots": array_tree_fingerprint(source),
                "target_slots": array_tree_fingerprint(target),
                "relation_codes": array_tree_fingerprint(codes),
                "valid": array_tree_fingerprint(valid_),
            }
        )

    def relation_mask(self, relation: FiniteElementHPLineageKind, /) -> Array:
        name = str(relation)
        if name not in _LINEAGE_CODES:
            raise ValueError("Unknown hp lineage relation.")
        return self.valid & (self.relation_codes == _LINEAGE_CODES[name])


class FiniteElementHPWorksetPlan(StrictModule, NonTrainableState):
    """Deterministic fixed-shape workset buckets keyed by degree tuples."""

    bucket_degrees: Array
    bucket_valid: Array
    cell_slots: Array
    cell_valid: Array
    cell_bucket: Array
    topology_id: str = eqx.field(static=True)
    topology_plan_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology_id: str,
        topology_plan_id: str,
        bucket_degrees: ArrayLike,
        bucket_valid: ArrayLike,
        cell_slots: ArrayLike,
        cell_valid: ArrayLike,
        cell_bucket: ArrayLike,
        /,
    ):
        identifier = str(topology_id)
        topology_plan = str(topology_plan_id)
        degrees = np.asarray(bucket_degrees, dtype=np.int32)
        buckets = np.asarray(bucket_valid, dtype=bool)
        slots = np.asarray(cell_slots, dtype=np.int32)
        valid = np.asarray(cell_valid, dtype=bool)
        reverse = np.asarray(cell_bucket, dtype=np.int32)
        if (
            not identifier
            or not topology_plan
            or degrees.ndim != 2
            or degrees.shape[0] == 0
            or buckets.shape != (degrees.shape[0],)
            or slots.shape != (degrees.shape[0], degrees.shape[0])
            or valid.shape != slots.shape
            or reverse.shape != (degrees.shape[0],)
        ):
            raise ValueError("hp workset arrays must use one fixed cell capacity.")
        capacity = degrees.shape[0]
        if (
            np.any(degrees[buckets] < 1)
            or np.any(degrees[~buckets] != 0)
            or np.any(slots[valid] < 0)
            or np.any(slots[valid] >= capacity)
            or np.any(slots[~valid] != -1)
            or np.any(reverse < -1)
            or np.any(reverse >= capacity)
        ):
            raise ValueError("hp workset bucket contents are invalid.")
        active_slots = slots[valid]
        if np.unique(active_slots).size != active_slots.size:
            raise ValueError("Every active hp cell must occur in exactly one bucket.")
        for bucket in range(capacity):
            if np.any(valid[bucket]) != bool(buckets[bucket]):
                raise ValueError("hp bucket validity and membership disagree.")
            if np.any(valid[bucket]) and np.any(
                reverse[slots[bucket, valid[bucket]]] != bucket
            ):
                raise ValueError("hp reverse bucket routes are inconsistent.")
        self.bucket_degrees = jnp.asarray(degrees)
        self.bucket_valid = jnp.asarray(buckets)
        self.cell_slots = jnp.asarray(slots)
        self.cell_valid = jnp.asarray(valid)
        self.cell_bucket = jnp.asarray(reverse)
        self.topology_id = identifier
        self.topology_plan_id = topology_plan
        self.capacity = capacity
        self.dimension = degrees.shape[1]
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-worksets",
                "topology": identifier,
                "topology_plan": topology_plan,
                "degrees": array_tree_fingerprint(degrees),
                "bucket_valid": array_tree_fingerprint(buckets),
                "slots": array_tree_fingerprint(slots),
                "cell_valid": array_tree_fingerprint(valid),
                "cell_bucket": array_tree_fingerprint(reverse),
            }
        )

    def gather(self, cell_values: ArrayLike, /) -> Array:
        values = jnp.asarray(cell_values)
        if values.shape[0] != self.capacity:
            raise ValueError("hp cell values do not match workset capacity.")
        safe = jnp.where(self.cell_valid, self.cell_slots, 0)
        gathered = values[safe]
        mask = self.cell_valid.reshape(self.cell_valid.shape + (1,) * (values.ndim - 1))
        return jnp.where(mask, gathered, jnp.zeros((), dtype=values.dtype))


def finite_element_hp_workset_plan(
    topology: FiniteElementHPTopology, /
) -> FiniteElementHPWorksetPlan:
    if not isinstance(topology, FiniteElementHPTopology):
        raise TypeError("topology must be FiniteElementHPTopology.")
    identifiers = np.asarray(topology.cell_global_ids)
    active = np.asarray(topology.active)
    degrees = np.asarray(topology.cell_degrees)
    capacity = topology.capacity
    unique_degrees = sorted(
        {tuple(int(value) for value in degree) for degree in degrees[active]}
    )
    bucket_degrees = np.zeros((capacity, topology.dimension), dtype=np.int32)
    bucket_valid = np.zeros((capacity,), dtype=bool)
    slots = np.full((capacity, capacity), -1, dtype=np.int32)
    valid = np.zeros((capacity, capacity), dtype=bool)
    reverse = np.full((capacity,), -1, dtype=np.int32)
    for bucket, degree in enumerate(unique_degrees):
        members = np.flatnonzero(active & np.all(degrees == degree, axis=1))
        members = members[np.argsort(identifiers[members], kind="stable")]
        bucket_degrees[bucket] = degree
        bucket_valid[bucket] = True
        slots[bucket, : members.size] = members
        valid[bucket, : members.size] = True
        reverse[members] = bucket
    return FiniteElementHPWorksetPlan(
        topology.topology_id,
        topology.plan_id,
        bucket_degrees,
        bucket_valid,
        slots,
        valid,
        reverse,
    )


class FiniteElementHPTransferPlan(StrictModule, NonTrainableState):
    """Padded p/h routes with distinct primal, dual, adjoint, and projection maps."""

    source_slots: Array
    target_slots: Array
    valid: Array
    source_dof_count: Array
    target_dof_count: Array
    primal: Array
    raw_dual_pullback: Array
    pairing_adjoint: Array | None
    mass_projection: Array | None
    source_plan_id: str = eqx.field(static=True)
    target_plan_id: str = eqx.field(static=True)
    source_topology_id: str = eqx.field(static=True)
    target_topology_id: str = eqx.field(static=True)
    transfer_kind: FiniteElementHPTransferKind = eqx.field(static=True)
    source_capacity: int = eqx.field(static=True)
    target_capacity: int = eqx.field(static=True)
    has_pairing_adjoint: bool = eqx.field(static=True)
    has_mass_projection: bool = eqx.field(static=True)
    transfer_id: str = eqx.field(static=True)

    def __init__(
        self,
        source_topology_id: str,
        target_topology_id: str,
        transfer_kind: FiniteElementHPTransferKind,
        source_capacity: int,
        target_capacity: int,
        source_slots: ArrayLike,
        target_slots: ArrayLike,
        source_dof_count: ArrayLike,
        target_dof_count: ArrayLike,
        primal: ArrayLike,
        /,
        *,
        source_plan_id: str,
        target_plan_id: str,
        valid: ArrayLike | None = None,
        pairing_adjoint: ArrayLike | None = None,
        mass_projection: ArrayLike | None = None,
    ):
        source_id = str(source_topology_id)
        target_id = str(target_topology_id)
        source_plan = str(source_plan_id)
        target_plan = str(target_plan_id)
        kind = str(transfer_kind)
        source_capacity_ = int(source_capacity)
        target_capacity_ = int(target_capacity)
        source = np.asarray(source_slots, dtype=np.int32)
        target = np.asarray(target_slots, dtype=np.int32)
        source_count = np.asarray(source_dof_count, dtype=np.int32)
        target_count = np.asarray(target_dof_count, dtype=np.int32)
        primal_ = np.asarray(primal)
        valid_ = (
            np.ones(source.shape, dtype=bool)
            if valid is None
            else np.asarray(valid, dtype=bool)
        )
        if (
            not source_id
            or not target_id
            or (source_id == target_id and kind != "p")
            or not source_plan
            or not target_plan
            or kind not in ("p", "h-refinement", "h-coarsening")
            or source_capacity_ <= 0
            or target_capacity_ <= 0
            or source.ndim != 1
            or target.shape != source.shape
            or source_count.shape != source.shape
            or target_count.shape != source.shape
            or valid_.shape != source.shape
            or primal_.ndim != 3
            or primal_.shape[0] != source.size
            or not np.issubdtype(primal_.dtype, np.inexact)
            or np.any(~np.isfinite(primal_))
        ):
            raise ValueError(
                "hp transfer identity, routes, or primal matrices are invalid."
            )
        target_width, source_width = primal_.shape[1:]
        if (
            np.any(source[valid_] < 0)
            or np.any(source[valid_] >= source_capacity_)
            or np.any(target[valid_] < 0)
            or np.any(target[valid_] >= target_capacity_)
            or np.any(source[~valid_] != -1)
            or np.any(target[~valid_] != -1)
            or np.any(source_count[valid_] < 1)
            or np.any(source_count[valid_] > source_width)
            or np.any(target_count[valid_] < 1)
            or np.any(target_count[valid_] > target_width)
            or np.any(source_count[~valid_] != 0)
            or np.any(target_count[~valid_] != 0)
        ):
            raise ValueError(
                "hp transfer capacities, counts, or inactive sentinels are invalid."
            )
        support = (
            np.arange(target_width)[None, :, None] < target_count[:, None, None]
        ) & (np.arange(source_width)[None, None, :] < source_count[:, None, None])
        if np.any(primal_[~support] != 0.0):
            raise ValueError(
                "hp primal transfer must be zero outside active padded blocks."
            )
        adjoint = None if pairing_adjoint is None else np.asarray(pairing_adjoint)
        projection = None if mass_projection is None else np.asarray(mass_projection)
        if adjoint is not None and (
            adjoint.shape != (source.size, source_width, target_width)
            or not np.issubdtype(adjoint.dtype, np.inexact)
            or np.any(~np.isfinite(adjoint))
            or np.any(adjoint[~np.swapaxes(support, 1, 2)] != 0.0)
        ):
            raise ValueError("hp pairing adjoint is invalid.")
        if projection is not None and (
            projection.shape != primal_.shape
            or not np.issubdtype(projection.dtype, np.inexact)
            or np.any(~np.isfinite(projection))
            or np.any(projection[~support] != 0.0)
        ):
            raise ValueError("hp physical mass projection is invalid.")
        self.source_slots = jnp.asarray(source)
        self.target_slots = jnp.asarray(target)
        self.valid = jnp.asarray(valid_)
        self.source_dof_count = jnp.asarray(source_count)
        self.target_dof_count = jnp.asarray(target_count)
        self.primal = jnp.asarray(primal_)
        self.raw_dual_pullback = jnp.swapaxes(self.primal, 1, 2)
        self.pairing_adjoint = None if adjoint is None else jnp.asarray(adjoint)
        self.mass_projection = None if projection is None else jnp.asarray(projection)
        self.source_topology_id = source_id
        self.target_topology_id = target_id
        self.transfer_kind = kind
        self.source_plan_id = source_plan
        self.target_plan_id = target_plan
        self.source_capacity = source_capacity_
        self.target_capacity = target_capacity_
        self.has_pairing_adjoint = adjoint is not None
        self.has_mass_projection = projection is not None
        self.transfer_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-transfer",
                "source": source_id,
                "target": target_id,
                "transfer_kind": kind,
                "source_capacity": source_capacity_,
                "target_capacity": target_capacity_,
                "source_plan": source_plan,
                "target_plan": target_plan,
                "source_slots": array_tree_fingerprint(source),
                "target_slots": array_tree_fingerprint(target),
                "source_count": array_tree_fingerprint(source_count),
                "target_count": array_tree_fingerprint(target_count),
                "valid": array_tree_fingerprint(valid_),
                "primal": array_tree_fingerprint(primal_),
                "pairing_adjoint": (
                    None if adjoint is None else array_tree_fingerprint(adjoint)
                ),
                "mass_projection": (
                    None if projection is None else array_tree_fingerprint(projection)
                ),
            }
        )

    def _forward(self, matrices: Array, source_values: ArrayLike, /) -> Array:
        values = jnp.asarray(source_values)
        if values.ndim < 2 or values.shape[:2] != (
            self.source_capacity,
            matrices.shape[2],
        ):
            raise ValueError("hp source values do not match transfer capacity/width.")
        safe_source = jnp.where(self.valid, self.source_slots, 0)
        safe_target = jnp.where(self.valid, self.target_slots, 0)
        local = values[safe_source]
        source_mask = (
            jnp.arange(matrices.shape[2])[None, :] < self.source_dof_count[:, None]
        ).reshape((matrices.shape[0], matrices.shape[2]) + (1,) * (values.ndim - 2))
        local = jnp.where(source_mask, local, 0.0)
        mapped = ein.contract("rts,rs...->rt...", matrices, local)
        target_mask = (
            jnp.arange(matrices.shape[1])[None, :] < self.target_dof_count[:, None]
        ).reshape((matrices.shape[0], matrices.shape[1]) + (1,) * (values.ndim - 2))
        route_mask = self.valid.reshape(self.valid.shape + (1,) * (mapped.ndim - 1))
        mapped = jnp.where(route_mask & target_mask, mapped, 0.0)
        result = jnp.zeros(
            (self.target_capacity, matrices.shape[1]) + values.shape[2:],
            dtype=mapped.dtype,
        )
        for route in range(matrices.shape[0]):
            result = result.at[safe_target[route]].add(mapped[route])
        return result

    def _reverse(self, matrices: Array, target_values: ArrayLike, /) -> Array:
        values = jnp.asarray(target_values)
        if values.ndim < 2 or values.shape[:2] != (
            self.target_capacity,
            matrices.shape[2],
        ):
            raise ValueError("hp target values do not match transfer capacity/width.")
        safe_source = jnp.where(self.valid, self.source_slots, 0)
        safe_target = jnp.where(self.valid, self.target_slots, 0)
        local = values[safe_target]
        target_mask = (
            jnp.arange(matrices.shape[2])[None, :] < self.target_dof_count[:, None]
        ).reshape((matrices.shape[0], matrices.shape[2]) + (1,) * (values.ndim - 2))
        local = jnp.where(target_mask, local, 0.0)
        mapped = ein.contract("rst,rt...->rs...", matrices, local)
        source_mask = (
            jnp.arange(matrices.shape[1])[None, :] < self.source_dof_count[:, None]
        ).reshape((matrices.shape[0], matrices.shape[1]) + (1,) * (values.ndim - 2))
        route_mask = self.valid.reshape(self.valid.shape + (1,) * (mapped.ndim - 1))
        mapped = jnp.where(route_mask & source_mask, mapped, 0.0)
        result = jnp.zeros(
            (self.source_capacity, matrices.shape[1]) + values.shape[2:],
            dtype=mapped.dtype,
        )
        for route in range(matrices.shape[0]):
            result = result.at[safe_source[route]].add(mapped[route])
        return result

    def apply_primal(self, source_values: ArrayLike, /) -> Array:
        return self._forward(self.primal, source_values)

    def apply_mass_projection(self, source_values: ArrayLike, /) -> Array:
        if self.mass_projection is None:
            raise ValueError("This hp transfer has no physical mass projection.")
        return self._forward(self.mass_projection, source_values)

    def pullback_raw(self, target_dual: ArrayLike, /) -> Array:
        return self._reverse(self.raw_dual_pullback, target_dual)

    def apply_pairing_adjoint(self, target_values: ArrayLike, /) -> Array:
        if self.pairing_adjoint is None:
            raise ValueError("This hp transfer has no declared pairing adjoint.")
        return self._reverse(self.pairing_adjoint, target_values)


__all__ = [
    "FiniteElementHPCellKind",
    "FiniteElementHPLineage",
    "FiniteElementHPLineageKind",
    "FiniteElementHPTopology",
    "FiniteElementHPTransferKind",
    "FiniteElementHPTransferPlan",
    "FiniteElementHPWorksetPlan",
    "finite_element_hp_workset_plan",
]
