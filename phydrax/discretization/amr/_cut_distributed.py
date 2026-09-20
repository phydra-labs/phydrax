#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device- and process-sharded multivalued cut-component execution."""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import PartitionSpec
from jaxtyping import Array, ArrayLike

from ..._execution_runtime import ExecutionGroup
from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._cut_complex import MultivaluedCutCellComplex


class DistributedCutCellEvidence(StrictModule, NonTrainableState):
    """Deterministic load and cross-part interface evidence."""

    part_costs: tuple[float, ...] = eqx.field(static=True)
    part_counts: tuple[int, ...] = eqx.field(static=True)
    maximum_imbalance: float = eqx.field(static=True)
    cross_part_face_count: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


class DistributedCutCellState(StrictModule):
    """Global JAX array sharded by owner part and fixed local component slot."""

    values: Array
    partition_id: str = eqx.field(static=True)
    component_shape: tuple[int, ...] = eqx.field(static=True)


class PreparedDistributedCutCellComplex(StrictModule, NonTrainableState):
    """Stable component ownership and cross-part face routes on an execution mesh."""

    complex: MultivaluedCutCellComplex
    group: ExecutionGroup = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    component_owners: Array
    component_local_slots: Array
    face_owner_parts: Array
    face_neighbor_parts: Array
    cross_part_faces: Array
    evidence: DistributedCutCellEvidence
    partition_id: str = eqx.field(static=True)

    def __init__(
        self,
        complex_: MultivaluedCutCellComplex,
        group: ExecutionGroup,
        local_capacity: int,
        /,
        *,
        costs: ArrayLike | None = None,
    ):
        if not isinstance(complex_, MultivaluedCutCellComplex) or not isinstance(
            group, ExecutionGroup
        ):
            raise TypeError(
                "Distributed cut cells require a complex and live ExecutionGroup."
            )
        capacity = int(local_capacity)
        part_count = len(group.devices)
        cell_count = complex_.component_count
        if capacity <= 0 or part_count <= 0 or capacity * part_count < cell_count:
            raise ValueError("Distributed cut-cell local capacity is insufficient.")
        cost = (
            np.ones((cell_count,), dtype=np.float64)
            if costs is None
            else np.asarray(costs, dtype=np.float64)
        )
        if cost.shape != (cell_count,) or np.any(~np.isfinite(cost) | (cost <= 0.0)):
            raise ValueError("Distributed cut-cell costs must be positive per component.")
        levels = np.asarray(complex_.component_levels, dtype=np.int32)[:cell_count]
        coordinates = np.asarray(complex_.component_cell_coordinates, dtype=np.int32)[
            :cell_count
        ]
        slots = np.asarray(complex_.component_slots, dtype=np.int32)[:cell_count]
        stable_order = sorted(
            range(cell_count),
            key=lambda index: (
                int(levels[index]),
                tuple(coordinates[index]),
                int(slots[index]),
            ),
        )
        owners = np.full((complex_.component_capacity,), -1, dtype=np.int32)
        local_slots = np.full((complex_.component_capacity,), -1, dtype=np.int32)
        used = np.zeros((part_count,), dtype=np.int32)
        accumulated = np.zeros((part_count,), dtype=np.float64)
        for component in stable_order:
            candidates = tuple(
                part for part in range(part_count) if used[part] < capacity
            )
            if not candidates:
                raise ValueError("Distributed cut-cell owner capacity is exceeded.")
            owner = min(candidates, key=lambda part: (accumulated[part], part))
            owners[component] = owner
            local_slots[component] = used[owner]
            used[owner] += 1
            accumulated[owner] += cost[component]
        active_faces = np.asarray(complex_.face_active, dtype=np.bool_)
        face_owner = np.asarray(complex_.face_owner_components, dtype=np.int32)
        face_neighbor = np.asarray(complex_.face_neighbor_components, dtype=np.int32)
        face_owner_parts = np.full((complex_.face_capacity,), -1, dtype=np.int32)
        face_neighbor_parts = np.full((complex_.face_capacity,), -1, dtype=np.int32)
        face_owner_parts[active_faces] = owners[face_owner[active_faces]]
        internal = active_faces & (face_neighbor >= 0)
        face_neighbor_parts[internal] = owners[face_neighbor[internal]]
        cross = internal & (face_owner_parts != face_neighbor_parts)
        positive_costs = accumulated[accumulated > 0.0]
        imbalance = (
            0.0
            if positive_costs.size == 0
            else float(np.max(positive_costs) / np.mean(positive_costs) - 1.0)
        )
        evidence_id = canonical_fingerprint(
            {
                "kind": "distributed-cut-cell-evidence",
                "complex": complex_.topology_id,
                "group": group.spec.group_id,
                "owners": array_tree_fingerprint(owners),
                "local_slots": array_tree_fingerprint(local_slots),
                "part_costs": accumulated.tolist(),
                "cross_part_faces": array_tree_fingerprint(cross),
            }
        )
        evidence = DistributedCutCellEvidence(
            part_costs=tuple(float(value) for value in accumulated),
            part_counts=tuple(used),
            maximum_imbalance=imbalance,
            cross_part_face_count=int(np.count_nonzero(cross)),
            evidence_id=evidence_id,
        )
        self.complex = complex_
        self.group = group
        self.local_capacity = capacity
        self.component_owners = jnp.asarray(owners)
        self.component_local_slots = jnp.asarray(local_slots)
        self.face_owner_parts = jnp.asarray(face_owner_parts)
        self.face_neighbor_parts = jnp.asarray(face_neighbor_parts)
        self.cross_part_faces = jnp.asarray(cross)
        self.evidence = evidence
        self.partition_id = canonical_fingerprint(
            {
                "kind": "prepared-distributed-cut-cell-complex",
                "evidence": evidence_id,
                "local_capacity": capacity,
            }
        )

    @property
    def part_count(self) -> int:
        return len(self.group.devices)

    @property
    def mesh_axis(self) -> str:
        return self.group.spec.mesh_axes[0][0]

    def pack(self, canonical_values: ArrayLike, /) -> DistributedCutCellState:
        value = jnp.asarray(canonical_values)
        if value.ndim == 0 or value.shape[0] != self.complex.component_capacity:
            raise ValueError(
                "Distributed cut-cell pack requires padded canonical component values."
            )
        trailing = value.shape[1:]
        inactive = ~self.complex.component_active
        value = eqx.error_if(
            value,
            jnp.any(
                inactive.reshape(inactive.shape + (1,) * len(trailing)) & (value != 0.0)
            ),
            "Distributed cut-cell padding must be exactly zero.",
        )
        packed = jnp.zeros(
            (self.part_count, self.local_capacity) + trailing,
            dtype=value.dtype,
        )
        count = self.complex.component_count
        packed = packed.at[
            self.component_owners[:count], self.component_local_slots[:count]
        ].set(value[:count])
        sharding = self.group.named_sharding(
            PartitionSpec(self.mesh_axis, None, *((None,) * len(trailing)))
        )
        return DistributedCutCellState(
            values=jax.device_put(packed, sharding),
            partition_id=self.partition_id,
            component_shape=trailing,
        )

    def pack_process_local(
        self,
        component_indices: Sequence[int],
        local_values: ArrayLike,
        /,
    ) -> DistributedCutCellState:
        """Construct a global array from only this process's owned components."""

        from ..._data_plane import make_global_array_from_process_local_data

        indices = tuple(component_indices)
        expected = self.local_component_indices()
        if indices != expected:
            raise ValueError(
                "Process-local component indices must equal canonical local ownership."
            )
        values = jnp.asarray(local_values)
        if values.ndim == 0 or values.shape[0] != len(indices):
            raise ValueError(
                "Process-local values require one leading entry per local component."
            )
        process = jax.process_index()
        local_parts = tuple(
            part
            for part, key in enumerate(self.group.spec.device_keys)
            if key[0] == process
        )
        if len(local_parts) != jax.local_device_count():
            raise ValueError(
                "Execution group process membership must cover every local device."
            )
        part_to_local = {part: local for local, part in enumerate(local_parts)}
        trailing = values.shape[1:]
        local_data = jnp.zeros(
            (len(local_parts), self.local_capacity) + trailing,
            dtype=values.dtype,
        )
        owners = np.asarray(self.component_owners, dtype=np.int32)
        slots = np.asarray(self.component_local_slots, dtype=np.int32)
        for row, component in enumerate(indices):
            owner = int(owners[component])
            if owner not in part_to_local:
                raise ValueError("Process-local component is owned by another process.")
            local_data = local_data.at[part_to_local[owner], int(slots[component])].set(
                values[row]
            )
        sharding = self.group.named_sharding(
            PartitionSpec(self.mesh_axis, None, *((None,) * len(trailing)))
        )
        global_array = make_global_array_from_process_local_data(
            sharding,
            local_data,
            global_shape=(self.part_count, self.local_capacity) + trailing,
        )
        return DistributedCutCellState(
            values=global_array,
            partition_id=self.partition_id,
            component_shape=trailing,
        )

    def unpack_process_local(self, state: DistributedCutCellState, /) -> Array:
        """Return canonical local-owner values without materializing remote cells."""

        if (
            not isinstance(state, DistributedCutCellState)
            or state.partition_id != self.partition_id
        ):
            raise ValueError("Distributed cut-cell state belongs to another partition.")
        indices = self.local_component_indices()
        owners = self.component_owners[jnp.asarray(indices, dtype=jnp.int32)]
        slots = self.component_local_slots[jnp.asarray(indices, dtype=jnp.int32)]
        return state.values[owners, slots]

    def unpack(self, state: DistributedCutCellState, /) -> Array:
        if (
            not isinstance(state, DistributedCutCellState)
            or state.partition_id != self.partition_id
        ):
            raise ValueError("Distributed cut-cell state belongs to another partition.")
        flat = state.values.reshape(
            (self.part_count * self.local_capacity,) + state.component_shape
        )
        count = self.complex.component_count
        routes = (
            self.component_owners[:count] * self.local_capacity
            + self.component_local_slots[:count]
        )
        active_values = flat[routes]
        canonical = jnp.zeros(
            (self.complex.component_capacity,) + state.component_shape,
            dtype=state.values.dtype,
        )
        return canonical.at[:count].set(active_values)

    def face_states(
        self,
        state: DistributedCutCellState,
        /,
    ) -> tuple[Array, Array, Array]:
        """Gather owner/neighbor states; XLA inserts required cross-host collectives."""

        canonical = self.unpack(state)
        active = self.complex.face_active
        owner = self.complex.face_owner_components
        neighbor = self.complex.face_neighbor_components
        safe_neighbor = jnp.maximum(neighbor, 0)
        left = canonical[owner]
        right = canonical[safe_neighbor]
        trailing = (1,) * len(state.component_shape)
        return (
            jnp.where(
                active.reshape(active.shape + trailing),
                left,
                jnp.zeros((), dtype=left.dtype),
            ),
            jnp.where(
                (active & (neighbor >= 0)).reshape(active.shape + trailing),
                right,
                left,
            ),
            active,
        )

    @staticmethod
    def collective_accept(local_accept: ArrayLike, /) -> bool:
        """Process-global acceptance; every host observes the same decision."""

        local = np.asarray(local_accept, dtype=np.bool_)
        if local.shape != ():
            raise ValueError("Collective acceptance input must be scalar Boolean.")
        gathered = np.asarray(multihost_utils.process_allgather(local))
        return bool(np.all(gathered))

    def local_component_indices(self, /) -> tuple[int, ...]:
        process = jax.process_index()
        local_parts = {
            part
            for part, key in enumerate(self.group.spec.device_keys)
            if key[0] == process
        }
        owners = np.asarray(self.component_owners)[: self.complex.component_count]
        return tuple(
            int(index) for index, owner in enumerate(owners) if int(owner) in local_parts
        )


class DistributedCutCellPartitionPlan(StrictModule, NonTrainableState):
    """Bind component ownership to a real JAX execution group."""

    group: ExecutionGroup = eqx.field(static=True)
    local_capacity: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, group: ExecutionGroup, local_capacity: int, /):
        capacity = int(local_capacity)
        if not isinstance(group, ExecutionGroup) or capacity <= 0:
            raise ValueError("Distributed cut-cell partition plan is invalid.")
        self.group = group
        self.local_capacity = capacity
        self.plan_id = canonical_fingerprint(
            {
                "kind": "distributed-cut-cell-partition-plan",
                "group": group.spec.group_id,
                "local_capacity": capacity,
            }
        )

    def prepare(
        self,
        complex_: MultivaluedCutCellComplex,
        /,
        *,
        costs: ArrayLike | None = None,
    ) -> PreparedDistributedCutCellComplex:
        return PreparedDistributedCutCellComplex(
            complex_, self.group, self.local_capacity, costs=costs
        )


__all__ = [
    "DistributedCutCellEvidence",
    "DistributedCutCellPartitionPlan",
    "DistributedCutCellState",
    "PreparedDistributedCutCellComplex",
]
