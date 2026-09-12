#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic variable-patch placement, packing, and explicit repartition."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._topology_epoch import TopologyEpoch
from ._variable import VariablePatchFieldState, VariablePatchHierarchyTopology
from ._variable_runtime import VariablePatchHierarchyState


class VariablePatchPartitionEvidence(StrictModule, NonTrainableState):
    patch_counts: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    part_costs: tuple[tuple[float, ...], ...] = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class VariablePatchPartitionPlan(StrictModule, NonTrainableState):
    """Canonical box-ordered placement into fixed per-bucket local lane capacity."""

    part_count: int = eqx.field(static=True)
    local_lane_capacities: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        part_count: int,
        local_lane_capacities: Sequence[Sequence[int]],
        /,
    ):
        parts = int(part_count)
        capacities = tuple(
            tuple(int(value) for value in row) for row in local_lane_capacities
        )
        if (
            parts <= 0
            or not capacities
            or any(not row or any(value <= 0 for value in row) for row in capacities)
        ):
            raise ValueError("Variable patch partition capacities must be positive.")
        self.part_count = parts
        self.local_lane_capacities = capacities
        self.plan_id = canonical_fingerprint(
            {
                "kind": "variable-patch-partition-plan",
                "part_count": parts,
                "local_lane_capacities": capacities,
            }
        )

    def prepare(
        self,
        topology: VariablePatchHierarchyTopology,
        /,
        *,
        costs: Sequence[Sequence[ArrayLike | None]] | None = None,
    ) -> "PreparedVariablePatchPartition":
        return PreparedVariablePatchPartition(self, topology, costs=costs)


class PreparedVariablePatchPartition(StrictModule, NonTrainableState):
    """Placement-independent stable-box partition with canonical packed routes."""

    plan: VariablePatchPartitionPlan
    topology: VariablePatchHierarchyTopology
    owners: tuple[tuple[Array, ...], ...]
    local_slots: tuple[tuple[Array, ...], ...]
    evidence: VariablePatchPartitionEvidence
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        plan: VariablePatchPartitionPlan,
        topology: VariablePatchHierarchyTopology,
        /,
        *,
        costs: Sequence[Sequence[ArrayLike | None]] | None = None,
    ):
        if not isinstance(plan, VariablePatchPartitionPlan) or not isinstance(
            topology, VariablePatchHierarchyTopology
        ):
            raise TypeError("Variable patch partition requires a plan and topology.")
        if len(plan.local_lane_capacities) != len(topology.levels) or any(
            len(row) != len(level.plan.buckets)
            for row, level in zip(
                plan.local_lane_capacities,
                topology.levels,
                strict=True,
            )
        ):
            raise ValueError(
                "Variable patch partition capacities do not match topology buckets."
            )
        cost_values = (
            tuple(tuple(None for _ in level.plan.buckets) for level in topology.levels)
            if costs is None
            else tuple(tuple(value for value in row) for row in costs)
        )
        if len(cost_values) != len(topology.levels):
            raise ValueError("Variable patch costs require one row per level.")
        owners_by_level = []
        slots_by_level = []
        counts_by_level = []
        part_costs_by_level = []
        for level, (metadata, capacities, level_costs) in enumerate(
            zip(
                topology.levels,
                plan.local_lane_capacities,
                cost_values,
                strict=True,
            )
        ):
            if len(level_costs) != len(metadata.plan.buckets):
                raise ValueError("Variable patch costs require one vector per bucket.")
            level_owners = []
            level_slots = []
            level_counts = []
            accumulated = np.zeros((plan.part_count,), dtype=float)
            used = [np.zeros((plan.part_count,), dtype=np.int32) for _ in capacities]
            for bucket_index, (bucket, active, supplied_cost) in enumerate(
                zip(metadata.plan.buckets, metadata.active, level_costs, strict=True)
            ):
                active_host = np.asarray(active, dtype=bool)
                cost = (
                    np.ones((bucket.lane_capacity,), dtype=float)
                    if supplied_cost is None
                    else np.asarray(supplied_cost, dtype=float)
                )
                if cost.shape != (bucket.lane_capacity,) or np.any(
                    active_host & (~np.isfinite(cost) | (cost <= 0.0))
                ):
                    raise ValueError(
                        "Variable patch active costs must be positive and finite."
                    )
                owner = np.full((bucket.lane_capacity,), -1, dtype=np.int32)
                local = np.full((bucket.lane_capacity,), -1, dtype=np.int32)
                for lane in np.flatnonzero(active_host):
                    candidates = tuple(
                        part
                        for part in range(plan.part_count)
                        if used[bucket_index][part] < capacities[bucket_index]
                    )
                    if not candidates:
                        raise ValueError(
                            "Variable patch local partition capacity is exceeded."
                        )
                    part = min(candidates, key=lambda value: (accumulated[value], value))
                    owner[lane] = part
                    local[lane] = used[bucket_index][part]
                    used[bucket_index][part] += 1
                    accumulated[part] += cost[lane]
                level_owners.append(jnp.asarray(owner))
                level_slots.append(jnp.asarray(local))
                level_counts.append(int(np.count_nonzero(active_host)))
            owners_by_level.append(tuple(level_owners))
            slots_by_level.append(tuple(level_slots))
            counts_by_level.append(tuple(level_counts))
            part_costs_by_level.append(tuple(float(value) for value in accumulated))
        all_costs = np.concatenate(
            tuple(np.asarray(value) for value in part_costs_by_level)
        )
        positive = all_costs[all_costs > 0.0]
        imbalance = (
            0.0
            if positive.size == 0
            else float(np.max(positive) / np.mean(positive) - 1.0)
        )
        evidence = VariablePatchPartitionEvidence(
            tuple(counts_by_level),
            tuple(part_costs_by_level),
            imbalance,
            canonical_fingerprint(
                {
                    "kind": "variable-patch-partition-evidence",
                    "topology": topology.topology_id,
                    "plan": plan.plan_id,
                    "owners": [
                        [array_tree_fingerprint(value) for value in level]
                        for level in owners_by_level
                    ],
                }
            ),
        )
        partition_id = canonical_fingerprint(
            {
                "kind": "prepared-variable-patch-partition",
                "topology": topology.topology_id,
                "plan": plan.plan_id,
                "owners": [
                    [array_tree_fingerprint(value) for value in level]
                    for level in owners_by_level
                ],
                "local_slots": [
                    [array_tree_fingerprint(value) for value in level]
                    for level in slots_by_level
                ],
            }
        )
        self.plan = plan
        self.topology = topology
        self.owners = tuple(owners_by_level)
        self.local_slots = tuple(slots_by_level)
        self.evidence = evidence
        self.partition_id = partition_id

    def pack(
        self, state: VariablePatchHierarchyState, /
    ) -> tuple[tuple[Array, ...], ...]:
        if (
            not isinstance(state, VariablePatchHierarchyState)
            or state.topology.epoch.epoch_id != self.topology.epoch.epoch_id
        ):
            raise ValueError("Variable patch partition state has a stale topology epoch.")
        output = []
        for level_index, (metadata, field, owners, local_slots, capacities) in enumerate(
            zip(
                self.topology.levels,
                state.levels,
                self.owners,
                self.local_slots,
                self.plan.local_lane_capacities,
                strict=True,
            )
        ):
            level = []
            for bucket_index, (
                bucket,
                values,
                active,
                owner,
                local,
                capacity,
            ) in enumerate(
                zip(
                    metadata.plan.buckets,
                    field.safe_values(),
                    metadata.active,
                    owners,
                    local_slots,
                    capacities,
                    strict=True,
                )
            ):
                packed = jnp.zeros(
                    (self.plan.part_count, capacity) + values.shape[1:],
                    dtype=values.dtype,
                )
                for lane in np.flatnonzero(np.asarray(active, dtype=bool)):
                    packed = packed.at[int(owner[lane]), int(local[lane])].set(
                        values[lane]
                    )
                level.append(packed)
            output.append(tuple(level))
        return tuple(output)

    def unpack(
        self,
        packed_values: Sequence[Sequence[ArrayLike]],
        /,
    ) -> VariablePatchHierarchyState:
        packed = tuple(
            tuple(jnp.asarray(value) for value in level) for level in packed_values
        )
        if len(packed) != len(self.topology.levels):
            raise ValueError("Packed variable patch state requires every level.")
        levels = []
        for metadata, values, owners, local_slots in zip(
            self.topology.levels,
            packed,
            self.owners,
            self.local_slots,
            strict=True,
        ):
            if len(values) != len(metadata.plan.buckets):
                raise ValueError("Packed variable patch state requires every bucket.")
            restored = []
            for bucket, packed_bucket, active, owner, local in zip(
                metadata.plan.buckets,
                values,
                metadata.active,
                owners,
                local_slots,
                strict=True,
            ):
                expected_prefix = (
                    self.plan.part_count,
                    self.plan.local_lane_capacities[metadata.plan.level][len(restored)],
                    *bucket.signature.envelope_shape,
                )
                if packed_bucket.shape[: len(expected_prefix)] != expected_prefix:
                    raise ValueError(
                        "Packed variable patch bucket shape is incompatible."
                    )
                canonical = jnp.zeros(
                    (bucket.lane_capacity,) + packed_bucket.shape[2:],
                    dtype=packed_bucket.dtype,
                )
                for lane in np.flatnonzero(np.asarray(active, dtype=bool)):
                    canonical = canonical.at[lane].set(
                        packed_bucket[int(owner[lane]), int(local[lane])]
                    )
                restored.append(canonical)
            levels.append(VariablePatchFieldState(metadata, tuple(restored)))
        return VariablePatchHierarchyState(self.topology, tuple(levels))

    def repartition_epoch(self, source: TopologyEpoch, /) -> TopologyEpoch:
        if source.epoch_id != self.topology.epoch.epoch_id:
            raise ValueError("Repartition source epoch is stale.")
        return TopologyEpoch(
            source.index + 1,
            source.geometry_id,
            source.topology_id,
            self.partition_id,
        )


__all__ = [
    "PreparedVariablePatchPartition",
    "VariablePatchPartitionEvidence",
    "VariablePatchPartitionPlan",
]
