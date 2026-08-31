#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


FiniteElementHPCellKind = Literal["quadrilateral", "hexahedron"]
FiniteElementHPLineageKind = Literal["unchanged", "refinement", "coarsening"]
FiniteElementHPTransferKind = Literal["p", "h-refinement", "h-coarsening"]

_LINEAGE_CODES = {"unchanged": 0, "refinement": 1, "coarsening": 2}


class FiniteElementHPTopology(StrictModule, NonTrainableState):
    """Fixed-capacity quad/hex topology identity and anisotropic cell degrees."""

    cell_global_ids: Array
    active: Array
    cell_degrees: Array
    cell_kind: FiniteElementHPCellKind = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    capacity: int = eqx.field(static=True)
    dimension: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        cell_kind: FiniteElementHPCellKind,
        topology_id: str,
        cell_global_ids: ArrayLike,
        active: ArrayLike,
        cell_degrees: ArrayLike,
        /,
    ):
        kind = cell_kind
        identifier = str(topology_id)
        identifiers = np.asarray(cell_global_ids, dtype=np.int64)
        active_ = np.asarray(active, dtype=bool)
        degrees = np.asarray(cell_degrees, dtype=np.int32)
        dimension = 2 if kind == "quadrilateral" else 3 if kind == "hexahedron" else 0
        if not identifier or dimension == 0:
            raise ValueError(
                "hp topology requires a quad/hex kind and non-empty identity."
            )
        if (
            identifiers.ndim != 1
            or identifiers.size == 0
            or active_.shape != identifiers.shape
            or degrees.shape != (identifiers.size, dimension)
        ):
            raise ValueError("hp topology arrays have incompatible fixed capacities.")
        if not np.any(active_):
            raise ValueError("hp topology must contain at least one active cell.")
        if (
            np.any(identifiers[active_] < 0)
            or np.unique(identifiers[active_]).size != np.count_nonzero(active_)
            or np.any(identifiers[~active_] != -1)
            or np.any(degrees[active_] < 1)
            or np.any(degrees[~active_] != 0)
        ):
            raise ValueError("hp active IDs/degrees or inactive sentinels are invalid.")
        self.cell_global_ids = jnp.asarray(identifiers)
        self.active = jnp.asarray(active_)
        self.cell_degrees = jnp.asarray(degrees)
        self.cell_kind = kind
        self.topology_id = identifier
        self.capacity = identifiers.size
        self.dimension = dimension
        self.plan_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-topology",
                "cell_kind": kind,
                "topology": identifier,
                "cell_ids": array_tree_fingerprint(identifiers),
                "active": array_tree_fingerprint(active_),
                "degrees": array_tree_fingerprint(degrees),
            }
        )


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
        mapped = oe.contract("rts,rs...->rt...", matrices, local)
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
        mapped = oe.contract("rst,rt...->rs...", matrices, local)
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


class FiniteElementHPAcceptedPlan(StrictModule, NonTrainableState):
    """One accepted or candidate topology paired with deterministic worksets."""

    topology: FiniteElementHPTopology
    worksets: FiniteElementHPWorksetPlan
    accepted_id: str = eqx.field(static=True)

    def __init__(
        self,
        topology: FiniteElementHPTopology,
        worksets: FiniteElementHPWorksetPlan | None = None,
        /,
    ):
        if not isinstance(topology, FiniteElementHPTopology):
            raise TypeError("topology must be FiniteElementHPTopology.")
        selected = (
            finite_element_hp_workset_plan(topology) if worksets is None else worksets
        )
        if (
            not isinstance(selected, FiniteElementHPWorksetPlan)
            or selected.topology_id != topology.topology_id
            or selected.topology_plan_id != topology.plan_id
            or selected.capacity != topology.capacity
            or selected.dimension != topology.dimension
        ):
            raise ValueError("hp worksets do not belong to the supplied topology.")
        self.topology = topology
        self.worksets = selected
        self.accepted_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-accepted-plan",
                "topology": topology.plan_id,
                "worksets": selected.plan_id,
            }
        )


class FiniteElementHPTransaction(StrictModule, NonTrainableState):
    """Rollback-safe host promotion of one immutable fixed-capacity candidate."""

    accepted: FiniteElementHPAcceptedPlan
    candidate: FiniteElementHPAcceptedPlan
    lineage: FiniteElementHPLineage
    p_transfers: tuple[FiniteElementHPTransferPlan, ...]
    h_transfers: tuple[FiniteElementHPTransferPlan, ...]
    transaction_id: str = eqx.field(static=True)

    def __init__(
        self,
        accepted: FiniteElementHPAcceptedPlan,
        candidate: FiniteElementHPAcceptedPlan,
        lineage: FiniteElementHPLineage,
        /,
        *,
        p_transfers: tuple[FiniteElementHPTransferPlan, ...] = (),
        h_transfers: tuple[FiniteElementHPTransferPlan, ...] = (),
    ):
        if (
            not isinstance(accepted, FiniteElementHPAcceptedPlan)
            or not isinstance(candidate, FiniteElementHPAcceptedPlan)
            or not isinstance(lineage, FiniteElementHPLineage)
        ):
            raise TypeError(
                "hp transaction requires accepted/candidate plans and lineage."
            )
        source = accepted.topology
        target = candidate.topology
        if (
            source.cell_kind != target.cell_kind
            or source.capacity != target.capacity
            or lineage.source_topology_id != source.topology_id
            or lineage.target_topology_id != target.topology_id
            or lineage.source_capacity != source.capacity
            or lineage.target_capacity != target.capacity
        ):
            raise ValueError("hp candidate topology and lineage are incompatible.")
        p_transfers_ = tuple(p_transfers)
        h_transfers_ = tuple(h_transfers)
        if any(
            not isinstance(transfer, FiniteElementHPTransferPlan)
            for transfer in p_transfers_ + h_transfers_
        ):
            raise TypeError(
                "hp transaction transfers must be FiniteElementHPTransferPlan."
            )
        if any(transfer.transfer_kind != "p" for transfer in p_transfers_) or any(
            transfer.transfer_kind not in ("h-refinement", "h-coarsening")
            for transfer in h_transfers_
        ):
            raise ValueError("hp transaction transfer kinds are assigned incorrectly.")
        for transfer in p_transfers_ + h_transfers_:
            if (
                transfer.source_topology_id != source.topology_id
                or transfer.target_topology_id != target.topology_id
                or transfer.source_plan_id != source.plan_id
                or transfer.target_plan_id != target.plan_id
                or transfer.source_capacity != source.capacity
                or transfer.target_capacity != target.capacity
            ):
                raise ValueError("hp transaction transfer topology identities disagree.")
        source_slots = np.asarray(lineage.source_slots)
        target_slots = np.asarray(lineage.target_slots)
        valid = np.asarray(lineage.valid)
        if np.any(~np.asarray(source.active)[source_slots[valid]]) or np.any(
            ~np.asarray(target.active)[target_slots[valid]]
        ):
            raise ValueError("hp lineage edges must connect active cells.")
        active_source = np.flatnonzero(np.asarray(source.active))
        active_target = np.flatnonzero(np.asarray(target.active))
        routed_source = source_slots[valid]
        routed_target = target_slots[valid]
        relation_codes = np.asarray(lineage.relation_codes)[valid]
        if not np.array_equal(
            np.unique(routed_source), active_source
        ) or not np.array_equal(np.unique(routed_target), active_target):
            raise ValueError("hp lineage must cover every active source and target cell.")
        child_capacity = 2**source.dimension
        for slot in active_source:
            local_codes = np.unique(relation_codes[routed_source == slot])
            if local_codes.size != 1:
                raise ValueError("One hp source cell cannot mix lineage relations.")
            if (
                local_codes[0] == _LINEAGE_CODES["refinement"]
                and np.count_nonzero(routed_source == slot) > child_capacity
            ):
                raise ValueError("hp refinement exceeds quad/hex child capacity.")
        for slot in active_target:
            local_codes = np.unique(relation_codes[routed_target == slot])
            if local_codes.size != 1:
                raise ValueError("One hp target cell cannot mix lineage relations.")
            if (
                local_codes[0] == _LINEAGE_CODES["coarsening"]
                and np.count_nonzero(routed_target == slot) > child_capacity
            ):
                raise ValueError("hp coarsening exceeds quad/hex child capacity.")
        unchanged = relation_codes == _LINEAGE_CODES["unchanged"]
        refinement = relation_codes == _LINEAGE_CODES["refinement"]
        coarsening = relation_codes == _LINEAGE_CODES["coarsening"]
        if (
            np.unique(routed_source[unchanged]).size != np.count_nonzero(unchanged)
            or np.unique(routed_target[unchanged]).size != np.count_nonzero(unchanged)
            or np.unique(routed_target[refinement]).size != np.count_nonzero(refinement)
            or np.unique(routed_source[coarsening]).size != np.count_nonzero(coarsening)
        ):
            raise ValueError("hp lineage refinement/coarsening directions are ambiguous.")
        self.accepted = accepted
        self.candidate = candidate
        self.lineage = lineage
        self.p_transfers = p_transfers_
        self.h_transfers = h_transfers_
        self.transaction_id = canonical_fingerprint(
            {
                "kind": "finite-element-hp-transaction",
                "accepted": accepted.accepted_id,
                "candidate": candidate.accepted_id,
                "lineage": lineage.lineage_id,
                "p_transfers": [value.transfer_id for value in p_transfers_],
                "h_transfers": [value.transfer_id for value in h_transfers_],
            }
        )

    def rollback(self, /) -> FiniteElementHPAcceptedPlan:
        return self.accepted

    def promote(self, candidate_accepted: bool, /) -> FiniteElementHPAcceptedPlan:
        if not isinstance(candidate_accepted, (bool, np.bool_)):
            raise TypeError("hp candidate promotion is an explicit host decision.")
        return self.candidate if bool(candidate_accepted) else self.accepted


__all__ = [
    "FiniteElementHPAcceptedPlan",
    "FiniteElementHPCellKind",
    "FiniteElementHPLineage",
    "FiniteElementHPLineageKind",
    "FiniteElementHPTopology",
    "FiniteElementHPTransaction",
    "FiniteElementHPTransferKind",
    "FiniteElementHPTransferPlan",
    "FiniteElementHPWorksetPlan",
    "finite_element_hp_workset_plan",
]
