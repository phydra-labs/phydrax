#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Owner-local distributed refinement for stable-ID weighted CSR graphs.

Each process supplies complete adjacency rows for its owned vertices. Neighbor
IDs and their owner slots are explicit; remote labels are fetched only from the
owners referenced by those rows. The exchange carries sparse owner-routed rows,
never a replicated graph or owner table.
"""

from __future__ import annotations

from fractions import Fraction
from math import gcd
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from jax.typing import ArrayLike

from .._execution_runtime import ExecutionGroup
from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..typing import checked
from ._partition import (
    _INT64_MAX,
    _integer_vector,
    _NATIVE_WEIGHT_TOTAL_MAX,
    _targets_and_capacities,
    GraphPartitionPlan,
)


_UINT64_MASK = (1 << 64) - 1
_DIGEST_MASK = (1 << 256) - 1


def _digest_integer_limbs(value: int, /) -> tuple[int, int, int, int]:
    return (
        value & _UINT64_MASK,
        (value >> 64) & _UINT64_MASK,
        (value >> 128) & _UINT64_MASK,
        (value >> 192) & _UINT64_MASK,
    )


def _integer_from_limbs(values: np.ndarray, /) -> int:
    return sum(int(value) << (64 * limb) for limb, value in enumerate(values))


@final
class DistributedWeightedCSRGraph(StrictModule, NonTrainableState):
    """Owned rows of a distributed weighted CSR graph.

    ``entity_ids`` and ``owner_ranks`` identify the local owned rows. The
    flattened ``neighbors`` are stable global IDs, not local row indices;
    ``neighbor_owner_ranks`` names the execution-group slot that owns each
    neighbor. Both owner arrays use slots in
    ``execution_group.spec.process_indices`` order. Every owned row must provide
    its complete adjacency. Remote neighbor labels are resolved from those
    owners during partitioning, so a ghost's onward adjacency is neither
    required nor gathered.

    Rows and each row's neighbors are stored in stable-ID order. All ranks must
    construct the graph and call :func:`partition_graph_distributed` in the
    same collective sequence.
    """

    offsets: Array
    neighbors: Array
    neighbor_owner_ranks: Array
    edge_weights: Array
    vertex_weights: Array
    entity_ids: Array
    owner_ranks: Array
    vertex_count: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        offsets: ArrayLike,
        neighbors: ArrayLike,
        /,
        *,
        entity_ids: ArrayLike,
        owner_ranks: ArrayLike,
        neighbor_owner_ranks: ArrayLike,
        edge_weights: ArrayLike | None = None,
        vertex_weights: ArrayLike | None = None,
    ) -> None:
        ids = _integer_vector(entity_ids, "entity_ids")
        owners = _integer_vector(owner_ranks, "owner_ranks")
        row_offsets = _integer_vector(offsets, "offsets")
        adjacent = _integer_vector(neighbors, "neighbors")
        adjacent_owners = _integer_vector(neighbor_owner_ranks, "neighbor_owner_ranks")
        count = ids.size
        if owners.shape != (count,) or np.any(owners < 0):
            raise ValueError(
                "owner_ranks must contain one nonnegative group slot per owned vertex."
            )
        if ids.size and (np.any(ids < 0) or np.unique(ids).size != ids.size):
            raise ValueError(
                "entity_ids must be unique nonnegative stable IDs on each rank."
            )
        if row_offsets.shape != (count + 1,) or row_offsets[0] != 0:
            raise ValueError(
                "offsets must contain one CSR row per owned vertex and start at zero."
            )
        if np.any(np.diff(row_offsets) < 0) or row_offsets[-1] != adjacent.size:
            raise ValueError("offsets must be monotone and end at the adjacency length.")
        if adjacent_owners.shape != adjacent.shape or np.any(adjacent_owners < 0):
            raise ValueError(
                "neighbor_owner_ranks must align with neighbors and be nonnegative."
            )
        if adjacent.size and np.any(adjacent < 0):
            raise ValueError("neighbors must contain nonnegative stable vertex IDs.")
        if vertex_weights is None:
            vertex_values = np.ones((count,), dtype=np.int64)
        else:
            vertex_values = np.asarray(vertex_weights)
            if vertex_values.shape != (count,) or vertex_values.dtype.kind not in "iuf":
                raise ValueError(
                    "vertex_weights must contain one finite positive weight per owned vertex."
                )
            if vertex_values.dtype.kind == "f":
                if np.any(~np.isfinite(vertex_values)) or np.any(vertex_values <= 0):
                    raise ValueError(
                        "vertex_weights must contain one finite positive weight per owned vertex."
                    )
            elif np.any(vertex_values <= 0):
                raise ValueError(
                    "vertex_weights must contain one finite positive weight per owned vertex."
                )
        edge_values = (
            np.ones((adjacent.size,), dtype=np.int64)
            if edge_weights is None
            else _integer_vector(edge_weights, "edge_weights")
        )
        if edge_values.shape != adjacent.shape or np.any(edge_values < 0):
            raise ValueError(
                "edge_weights must be nonnegative integers aligned with neighbors."
            )

        source_rows = np.repeat(np.arange(count, dtype=np.int64), np.diff(row_offsets))
        if np.any(ids[source_rows] == adjacent):
            raise ValueError("Distributed CSR graph cannot contain self loops.")
        row_order = np.argsort(ids, kind="stable")
        sorted_ids = ids[row_order]
        sorted_owners = owners[row_order]
        sorted_weights = vertex_values[row_order]
        canonical_offsets = np.zeros((count + 1,), dtype=np.int64)
        neighbor_rows: list[np.ndarray] = []
        neighbor_owner_rows: list[np.ndarray] = []
        edge_rows: list[np.ndarray] = []
        for canonical_row, source_row in enumerate(row_order):
            start, stop = map(int, row_offsets[source_row : source_row + 2])
            row_neighbors = adjacent[start:stop]
            row_owners = adjacent_owners[start:stop]
            row_edges = edge_values[start:stop]
            local_order = np.argsort(row_neighbors, kind="stable")
            row_neighbors = row_neighbors[local_order]
            row_owners = row_owners[local_order]
            row_edges = row_edges[local_order]
            if row_neighbors.size > 1 and np.any(row_neighbors[1:] == row_neighbors[:-1]):
                raise ValueError(
                    "A distributed CSR row cannot repeat an adjacency entry."
                )
            neighbor_rows.append(row_neighbors)
            neighbor_owner_rows.append(row_owners)
            edge_rows.append(row_edges)
            canonical_offsets[canonical_row + 1] = (
                canonical_offsets[canonical_row] + row_neighbors.size
            )
        canonical_neighbors = (
            np.concatenate(neighbor_rows)
            if neighbor_rows
            else np.zeros((0,), dtype=np.int64)
        )
        canonical_neighbor_owners = (
            np.concatenate(neighbor_owner_rows)
            if neighbor_owner_rows
            else np.zeros((0,), dtype=np.int64)
        )
        canonical_edges = (
            np.concatenate(edge_rows) if edge_rows else np.zeros((0,), dtype=np.int64)
        )
        self.offsets = jnp.asarray(canonical_offsets, dtype=jnp.int64)
        self.neighbors = jnp.asarray(canonical_neighbors, dtype=jnp.int64)
        self.neighbor_owner_ranks = jnp.asarray(
            canonical_neighbor_owners, dtype=jnp.int64
        )
        self.edge_weights = jnp.asarray(canonical_edges, dtype=jnp.int64)
        self.vertex_weights = jnp.asarray(sorted_weights)
        self.entity_ids = jnp.asarray(sorted_ids, dtype=jnp.int64)
        self.owner_ranks = jnp.asarray(sorted_owners, dtype=jnp.int64)
        self.vertex_count = int(count)


@final
class DistributedGraphPartitionResult(StrictModule, NonTrainableState):
    """Local labels with collectively measured global partition evidence.

    ``owners`` aligns with local ``entity_ids`` and contains execution-group
    slots. Load and migration weights use the globally gcd-reduced exact integer
    units of the input vertex weights.
    """

    entity_ids: Array
    owners: Array
    part_weights: Array
    target_weights: Array
    capacity_weights: Array
    edge_cut: int = eqx.field(static=True)
    imbalance: float = eqx.field(static=True)
    feasible: bool = eqx.field(static=True)
    feasibility_status: str = eqx.field(static=True)
    migration_count: int = eqx.field(static=True)
    migration_weight: int = eqx.field(static=True)
    provenance: str = eqx.field(static=True)
    graph_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)
    migration_id: str = eqx.field(static=True)
    result_id: str = eqx.field(static=True)

    def __init__(
        self,
        entity_ids: np.ndarray,
        owners: np.ndarray,
        part_weights: np.ndarray,
        target_weights: np.ndarray,
        capacity_weights: np.ndarray,
        edge_cut: int,
        imbalance: float,
        feasible: bool,
        migration_count: int,
        migration_weight: int,
        graph_id: str,
        plan_id: str,
        migration_id: str,
        owner_identity: str,
        /,
    ) -> None:
        self.entity_ids = jnp.asarray(entity_ids, dtype=jnp.int64)
        self.owners = jnp.asarray(owners, dtype=jnp.int32)
        self.part_weights = jnp.asarray(part_weights, dtype=jnp.int64)
        self.target_weights = jnp.asarray(target_weights, dtype=jnp.int64)
        self.capacity_weights = jnp.asarray(capacity_weights, dtype=jnp.int64)
        self.edge_cut = int(edge_cut)
        self.imbalance = float(imbalance)
        self.feasible = bool(feasible)
        self.feasibility_status = "feasible" if feasible else "balance_not_reached"
        self.migration_count = int(migration_count)
        self.migration_weight = int(migration_weight)
        self.provenance = "phydrax-owner-local-graph-refinement-v1"
        self.graph_id = graph_id
        self.plan_id = plan_id
        self.migration_id = migration_id
        self.result_id = canonical_fingerprint(
            {
                "kind": "distributed-graph-partition-result",
                "graph": graph_id,
                "plan": plan_id,
                "owners": owner_identity,
                "part_weights": array_tree_fingerprint(part_weights),
                "target_weights": array_tree_fingerprint(target_weights),
                "capacity_weights": array_tree_fingerprint(capacity_weights),
                "edge_cut": int(edge_cut),
                "feasible": bool(feasible),
                "migration_id": migration_id,
            }
        )


def _group_mesh(
    execution_group: ExecutionGroup, axis_name: str, /
) -> tuple[Mesh, tuple[int, ...], int]:

    if not isinstance(execution_group, ExecutionGroup):
        raise TypeError("execution_group must be an ExecutionGroup.")
    processes = execution_group.spec.process_indices
    devices = execution_group.devices
    if (
        not processes
        or len(devices) != len(processes)
        or tuple(device.process_index for device in devices) != processes
        or len(set(processes)) != len(processes)
    ):
        raise ValueError(
            "Distributed graph partitioning requires one group device per process."
        )
    if not isinstance(axis_name, str) or not axis_name.strip():
        raise ValueError("axis_name must be a nonempty string.")
    process = int(jax.process_index())
    if process not in processes:
        raise ValueError("The current process is not a member of execution_group.")
    mesh = Mesh(np.asarray(devices, dtype=object), (axis_name.strip(),))
    return mesh, processes, processes.index(process)


def _process_local_array(values: np.ndarray, mesh: Mesh, axis_name: str, /) -> Array:
    parts = mesh.devices.size
    return jax.make_array_from_process_local_data(
        NamedSharding(mesh, PartitionSpec(axis_name)),
        values[None, ...],
        global_shape=(parts, *values.shape),
    )


def _group_allgather(values: ArrayLike, mesh: Mesh, axis_name: str, /) -> np.ndarray:
    from ..backends.distributed import JaxCollectiveProvider

    local = np.asarray(values)
    if local.ndim < 1 or local.dtype.hasobject:
        raise TypeError(
            "Distributed graph summaries must be typed arrays with a leading extent."
        )
    distributed = _process_local_array(local, mesh, axis_name)

    def gather(value: Array, /) -> Array:
        return JaxCollectiveProvider(axis_name).all_gather(value[0], axis=0, tiled=False)

    gathered = jax.shard_map(
        gather,
        mesh=mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(),
        check_vma=False,
    )(distributed)
    return np.asarray(jax.device_get(gathered))


def _exchange_by_owner(
    outgoing_by_owner: tuple[np.ndarray, ...], mesh: Mesh, axis_name: str, rank: int, /
) -> dict[int, np.ndarray]:
    """Exchange typed variable-length row banks without gathering their payloads."""
    from ..backends.distributed import JaxCollectiveProvider

    parts = mesh.devices.size
    if len(outgoing_by_owner) != parts:
        raise ValueError("One outgoing graph message bank is required per owner.")
    banks = tuple(np.asarray(value) for value in outgoing_by_owner)
    width = banks[0].shape[1] if banks[0].ndim == 2 else -1
    dtype = banks[0].dtype
    if (
        width < 1
        or dtype.kind not in "iu"
        or dtype.hasobject
        or any(
            value.ndim != 2 or value.shape[1] != width or value.dtype != dtype
            for value in banks
        )
    ):
        raise ValueError("Graph message banks must share a typed row layout.")
    local_counts = np.asarray([bank.shape[0] for bank in banks], dtype=np.int64)
    count_matrix = np.asarray(
        _group_allgather(local_counts, mesh, axis_name), dtype=np.int64
    )
    if (
        count_matrix.shape != (parts, parts)
        or np.any(count_matrix < 0)
        or not np.array_equal(count_matrix[rank], local_counts)
    ):
        raise ValueError("Collective graph message counts are inconsistent.")
    capacity = int(np.max(count_matrix)) if count_matrix.size else 0
    if capacity == 0:
        return {}
    packed = np.zeros((parts, capacity, width), dtype=dtype)
    for receiver, rows in enumerate(banks):
        packed[receiver, : rows.shape[0]] = rows
    distributed = _process_local_array(packed, mesh, axis_name)

    def exchange(value: Array, /) -> Array:
        # Each shard block is (1, receivers, capacity, width); exchange the
        # receiver axis of the single local block.
        incoming = JaxCollectiveProvider(axis_name).all_to_all(
            value[0],
            split_axis=0,
            concat_axis=0,
        )
        return incoming[None, ...]

    exchanged = jax.shard_map(
        exchange,
        mesh=mesh,
        in_specs=PartitionSpec(axis_name),
        out_specs=PartitionSpec(axis_name),
        check_vma=False,
    )(distributed)
    received = np.asarray(jax.device_get(exchanged.addressable_data(0)))[0]
    return {
        sender: received[sender, : int(count_matrix[sender, rank])].copy()
        for sender in range(parts)
        if count_matrix[sender, rank] > 0
    }


def _agree_failure(failed: bool, detail: str, mesh: Mesh, axis_name: str, /) -> None:
    flags = _group_allgather(np.asarray([int(failed)], dtype=np.int32), mesh, axis_name)
    if np.any(flags):
        if failed:
            raise ValueError(detail)
        raise ValueError(
            "Distributed graph partitioning failed on another execution-group rank."
        )


def _global_integer_sum(
    values: np.ndarray, name: str, mesh: Mesh, axis_name: str, /
) -> int:
    local_total = sum(map(int, np.asarray(values).reshape(-1)))
    invalid = local_total > _NATIVE_WEIGHT_TOTAL_MAX
    record = np.asarray([int(invalid), 0 if invalid else local_total], dtype=np.int64)
    gathered = _group_allgather(record, mesh, axis_name)
    if np.any(gathered[:, 0]):
        raise ValueError(
            f"{name} exceed the exact integer range of the graph partition kernel."
        )
    total = sum(map(int, gathered[:, 1]))
    if total > _NATIVE_WEIGHT_TOTAL_MAX:
        raise ValueError(
            f"{name} exceed the exact integer range of the graph partition kernel."
        )
    return total


def _exact_global_vertex_weights(
    values: ArrayLike,
    mesh: Mesh,
    axis_name: str,
    /,
) -> tuple[np.ndarray, int]:
    """Scale exact represented binary weights using a global power-of-two gcd."""
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(
            "vertex_weights must contain finite positive numbers in one dimension."
        )
    ratios: list[Fraction] = []
    local_error: str | None = None
    try:
        for item in raw:
            if raw.dtype.kind == "f":
                number = float(item)
                if not np.isfinite(number) or number <= 0:
                    raise ValueError("vertex_weights must be finite and positive.")
                ratios.append(Fraction.from_float(number))
            else:
                number = int(item)
                if number <= 0:
                    raise ValueError("vertex_weights must be finite and positive.")
                ratios.append(Fraction(number))
    except (OverflowError, TypeError, ValueError) as error:
        local_error = str(error)
    local_exponent = max(
        (ratio.denominator.bit_length() - 1 for ratio in ratios), default=0
    )
    control = _group_allgather(
        np.asarray((local_exponent, int(local_error is not None)), dtype=np.int64),
        mesh,
        axis_name,
    )
    if np.any(control[:, 1]):
        if local_error is not None:
            raise ValueError(local_error)
        raise ValueError("A rank supplied invalid distributed vertex weights.")
    common_exponent = int(np.max(control[:, 0]))
    numerators = [
        ratio.numerator << (common_exponent - (ratio.denominator.bit_length() - 1))
        for ratio in ratios
    ]
    local_divisor = 0
    for numerator in numerators:
        local_divisor = gcd(local_divisor, numerator)
    encoded = local_divisor.to_bytes(
        max(1, (local_divisor.bit_length() + 7) // 8), "little"
    )
    lengths = _group_allgather(
        np.asarray((len(encoded),), dtype=np.int64), mesh, axis_name
    ).reshape(-1)
    byte_width = int(np.max(lengths)) if lengths.size else 1
    local_bytes = np.zeros((byte_width,), dtype=np.uint8)
    local_bytes[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
    divisor_rows = _group_allgather(local_bytes, mesh, axis_name)
    common_divisor = 0
    for row, length in zip(divisor_rows, lengths, strict=True):
        common_divisor = gcd(
            common_divisor,
            int.from_bytes(
                np.asarray(row[: int(length)], dtype=np.uint8).tobytes(), "little"
            ),
        )
    if common_divisor <= 0:
        raise ValueError("A distributed graph must contain at least one vertex.")
    reduced = [value // common_divisor for value in numerators]
    local_invalid = any(value > _INT64_MAX for value in reduced)
    local_total = sum(reduced)
    local_invalid = local_invalid or local_total > _NATIVE_WEIGHT_TOTAL_MAX
    summary = _group_allgather(
        np.asarray(
            [int(local_invalid), 0 if local_invalid else local_total], dtype=np.int64
        ),
        mesh,
        axis_name,
    )
    if np.any(summary[:, 0]):
        raise ValueError(
            "vertex_weights cannot be represented exactly by the native integer-weight partition kernel."
        )
    total = sum(map(int, summary[:, 1]))
    if total > _NATIVE_WEIGHT_TOTAL_MAX:
        raise ValueError(
            "vertex_weights cannot be represented exactly by the native integer-weight partition kernel."
        )
    return np.asarray(reduced, dtype=np.int64), total


def _global_vertex_count(count: int, mesh: Mesh, axis_name: str, /) -> int:
    counts = _group_allgather(np.asarray([count], dtype=np.int64), mesh, axis_name)
    return sum(map(int, counts.reshape(-1)))


def _global_part_summary(
    labels: np.ndarray,
    weights: np.ndarray,
    part_count: int,
    mesh: Mesh,
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    local_loads = np.zeros((part_count,), dtype=np.int64)
    local_counts = np.zeros((part_count,), dtype=np.int64)
    for label, weight in zip(labels, weights, strict=True):
        part = int(label)
        local_loads[part] += int(weight)
        local_counts[part] += 1
    gathered = _group_allgather(
        np.concatenate((local_loads, local_counts)), mesh, axis_name
    )
    loads = np.asarray(
        [sum(int(value) for value in gathered[:, part]) for part in range(part_count)],
        dtype=np.int64,
    )
    counts = np.asarray(
        [
            sum(int(value) for value in gathered[:, part_count + part])
            for part in range(part_count)
        ],
        dtype=np.int64,
    )
    return loads, counts


def _part_minimum_ids(
    labels: np.ndarray,
    entity_ids: np.ndarray,
    part_count: int,
    mesh: Mesh,
    axis_name: str,
    /,
) -> tuple[np.ndarray, np.ndarray]:
    local = np.zeros((part_count, 2), dtype=np.int64)
    for part in range(part_count):
        rows = np.flatnonzero(labels == part)
        if rows.size:
            local[part] = (1, int(np.min(entity_ids[rows])))
    gathered = _group_allgather(local, mesh, axis_name)
    found = np.any(gathered[:, :, 0] != 0, axis=0)
    minima = np.full((part_count,), _INT64_MAX, dtype=np.int64)
    for part in np.flatnonzero(found):
        minima[part] = min(
            int(value) for value in gathered[:, int(part), 1] if int(value) >= 0
        )
    return found, minima


def _minimum_candidate(
    eligible: np.ndarray,
    entity_ids: np.ndarray,
    mesh: Mesh,
    axis_name: str,
    /,
) -> tuple[int, int] | None:
    present = bool(eligible.size)
    local_id = int(np.min(entity_ids[eligible])) if present else _INT64_MAX
    records = _group_allgather(
        np.asarray((int(present), local_id), dtype=np.int64), mesh, axis_name
    )
    if not np.any(records[:, 0]):
        return None
    chosen = min(int(record[1]) for record in records if record[0])
    chosen_slots = np.flatnonzero((records[:, 0] != 0) & (records[:, 1] == chosen))
    return chosen, int(chosen_slots[0])


def _repair_nonempty(
    labels: np.ndarray,
    entity_ids: np.ndarray,
    weights: np.ndarray,
    plan: GraphPartitionPlan,
    mesh: Mesh,
    axis_name: str,
    rank: int,
    /,
) -> np.ndarray:
    if not plan.require_nonempty:
        return labels
    current = labels.copy()
    _, part_counts = _global_part_summary(
        current, weights, plan.part_count, mesh, axis_name
    )
    missing = np.flatnonzero(part_counts == 0)
    for target in missing:
        has_rows, protected = _part_minimum_ids(
            current, entity_ids, plan.part_count, mesh, axis_name
        )
        movable = np.asarray(
            [
                row
                for row, label in enumerate(current)
                if part_counts[int(label)] > 1
                and (
                    not has_rows[int(label)]
                    or int(entity_ids[row]) != int(protected[int(label)])
                )
            ],
            dtype=np.int64,
        )
        chosen = _minimum_candidate(movable, entity_ids, mesh, axis_name)
        if chosen is None:
            raise RuntimeError(
                "Distributed nonempty-part repair found no movable vertex."
            )
        chosen_id, source_rank = chosen
        if rank == source_rank:
            row = int(np.searchsorted(entity_ids, chosen_id))
            if row >= entity_ids.size or int(entity_ids[row]) != chosen_id:
                raise RuntimeError(
                    "The selected stable vertex ID is absent from its source owner."
                )
            current[row] = int(target)
        _, part_counts = _global_part_summary(
            current, weights, plan.part_count, mesh, axis_name
        )
    return current


def _exchange_unique_ids(
    entity_ids: np.ndarray,
    mesh: Mesh,
    axis_name: str,
    rank: int,
    /,
) -> None:
    parts = mesh.devices.size
    buckets: list[list[int]] = [[] for _ in range(parts)]
    local_seen = {
        int(identifier) for identifier in entity_ids if int(identifier) % parts == rank
    }
    for identifier in entity_ids:
        bucket = int(identifier) % parts
        if bucket != rank:
            buckets[bucket].append(int(identifier))
    outgoing = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 1)) for rows in buckets
    )
    incoming = _exchange_by_owner(outgoing, mesh, axis_name, rank)
    failed = False
    for records in incoming.values():
        for identifier in records[:, 0]:
            key = int(identifier)
            if key in local_seen:
                failed = True
            local_seen.add(key)
    _agree_failure(
        failed,
        "Distributed entity_ids must be globally unique across owned rows.",
        mesh,
        axis_name,
    )


def _local_rows_by_id(
    offsets: np.ndarray,
    neighbors: np.ndarray,
    edge_weights: np.ndarray,
    entity_ids: np.ndarray,
    /,
) -> dict[int, dict[int, int]]:
    result: dict[int, dict[int, int]] = {}
    for row, identifier in enumerate(entity_ids):
        start, stop = map(int, offsets[row : row + 2])
        result[int(identifier)] = {
            int(neighbor): int(weight)
            for neighbor, weight in zip(
                neighbors[start:stop], edge_weights[start:stop], strict=True
            )
        }
    return result


def _validate_symmetric_adjacency(
    graph: DistributedWeightedCSRGraph,
    local_rows: dict[int, dict[int, int]],
    mesh: Mesh,
    axis_name: str,
    rank: int,
    /,
) -> None:
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    ids = np.asarray(graph.entity_ids, dtype=np.int64)
    neighbors = np.asarray(graph.neighbors, dtype=np.int64)
    neighbor_owners = np.asarray(graph.neighbor_owner_ranks, dtype=np.int64)
    edges = np.asarray(graph.edge_weights, dtype=np.int64)
    outgoing_rows: list[list[tuple[int, int, int]]] = [
        [] for _ in range(mesh.devices.size)
    ]
    failed = False
    for row, source_id in enumerate(ids):
        start, stop = map(int, offsets[row : row + 2])
        for neighbor_id, owner, weight in zip(
            neighbors[start:stop],
            neighbor_owners[start:stop],
            edges[start:stop],
            strict=True,
        ):
            target = int(neighbor_id)
            owner_rank = int(owner)
            if owner_rank == rank:
                reverse = local_rows.get(target)
                if reverse is None or reverse.get(int(source_id)) != int(weight):
                    failed = True
            else:
                outgoing_rows[owner_rank].append((int(source_id), target, int(weight)))
    outgoing = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 3)) for rows in outgoing_rows
    )
    incoming = _exchange_by_owner(outgoing, mesh, axis_name, rank)
    for records in incoming.values():
        for source_id, target_id, weight in records:
            reverse = local_rows.get(int(target_id))
            if reverse is None or reverse.get(int(source_id)) != int(weight):
                failed = True
    _agree_failure(
        failed,
        "Distributed CSR adjacency must be symmetric with equal weights on both directions.",
        mesh,
        axis_name,
    )


def _neighbor_labels(
    graph: DistributedWeightedCSRGraph,
    labels: np.ndarray,
    local_rows_by_id: dict[int, int],
    mesh: Mesh,
    axis_name: str,
    rank: int,
    /,
) -> np.ndarray:
    neighbors = np.asarray(graph.neighbors, dtype=np.int64)
    neighbor_owners = np.asarray(graph.neighbor_owner_ranks, dtype=np.int64)
    owner_for_id: dict[int, int] = {}
    failed = False
    for identifier, owner in zip(neighbors, neighbor_owners, strict=True):
        key, value = int(identifier), int(owner)
        previous = owner_for_id.setdefault(key, value)
        if previous != value:
            failed = True
    _agree_failure(
        failed,
        "A stable neighbor ID has conflicting owner slots in local adjacency.",
        mesh,
        axis_name,
    )
    requests: list[list[int]] = [[] for _ in range(mesh.devices.size)]
    for identifier, owner in sorted(owner_for_id.items()):
        if owner != rank:
            requests[owner].append(identifier)
        elif identifier not in local_rows_by_id:
            failed = True
    _agree_failure(
        failed,
        "A locally owned neighbor ID is absent from its owned CSR rows.",
        mesh,
        axis_name,
    )
    incoming_requests = _exchange_by_owner(
        tuple(np.asarray(rows, dtype=np.int64).reshape((-1, 1)) for rows in requests),
        mesh,
        axis_name,
        rank,
    )
    responses: list[list[tuple[int, int, int]]] = [[] for _ in range(mesh.devices.size)]
    for sender, request_rows in incoming_requests.items():
        for request in request_rows[:, 0]:
            identifier = int(request)
            row = local_rows_by_id.get(identifier)
            found = row is not None
            responses[sender].append(
                (identifier, int(labels[row]) if found else -1, int(found))
            )
    incoming_responses = _exchange_by_owner(
        tuple(np.asarray(rows, dtype=np.int64).reshape((-1, 3)) for rows in responses),
        mesh,
        axis_name,
        rank,
    )
    remote: dict[int, int] = {}
    expected = {identifier for identifier, owner in owner_for_id.items() if owner != rank}
    for sender, records in incoming_responses.items():
        for identifier, label, found in records:
            key = int(identifier)
            if not int(found) or key in remote or owner_for_id.get(key) != sender:
                failed = True
            else:
                remote[key] = int(label)
    if set(remote) != expected:
        failed = True
    if not failed:
        for identifier, owner in owner_for_id.items():
            if owner != rank and identifier not in remote:
                failed = True
    _agree_failure(
        failed,
        "Required ghost support did not resolve every stable neighbor ID at its declared owner.",
        mesh,
        axis_name,
    )
    edge_labels = np.empty(neighbors.shape, dtype=np.int32)
    for edge, (identifier, owner) in enumerate(
        zip(neighbors, neighbor_owners, strict=True)
    ):
        key = int(identifier)
        edge_labels[edge] = (
            int(labels[local_rows_by_id[key]]) if int(owner) == rank else remote[key]
        )
    return edge_labels


def _digest_summary(records: list[dict[str, object]], /) -> tuple[int, int, int]:
    total = 0
    exclusive = 0
    for record in records:
        value = int(canonical_fingerprint(record), 16)
        total = (total + value) & _DIGEST_MASK
        exclusive ^= value
    return len(records), total, exclusive


def _global_digest_identity(
    local_summary: tuple[int, int, int],
    mesh: Mesh,
    axis_name: str,
    kind: str,
    /,
) -> str:
    count, total, exclusive = local_summary
    summary = np.asarray(
        [count, *_digest_integer_limbs(total), *_digest_integer_limbs(exclusive)],
        dtype=np.uint64,
    )
    gathered = _group_allgather(summary, mesh, axis_name)
    global_count = sum(int(value) for value in gathered[:, 0])
    global_total = sum(_integer_from_limbs(row[1:5]) for row in gathered) & _DIGEST_MASK
    global_exclusive = 0
    for row in gathered:
        global_exclusive ^= _integer_from_limbs(row[5:9])
    return canonical_fingerprint(
        {
            "kind": kind,
            "record_count": global_count,
            "digest_sum": f"{global_total:064x}",
            "digest_xor": f"{global_exclusive:064x}",
        }
    )


def _distributed_graph_identity(
    graph: DistributedWeightedCSRGraph,
    vertex_weights: np.ndarray,
    /,
    mesh: Mesh,
    axis_name: str,
) -> str:
    ids = np.asarray(graph.entity_ids, dtype=np.int64)
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    neighbors = np.asarray(graph.neighbors, dtype=np.int64)
    edges = np.asarray(graph.edge_weights, dtype=np.int64)
    records: list[dict[str, object]] = []
    for row, identifier in enumerate(ids):
        start, stop = map(int, offsets[row : row + 2])
        records.append(
            {
                "entity_id": int(identifier),
                "vertex_weight": int(vertex_weights[row]),
                "neighbors": [
                    [int(neighbor), int(weight)]
                    for neighbor, weight in zip(
                        neighbors[start:stop], edges[start:stop], strict=True
                    )
                ],
            }
        )
    return _global_digest_identity(
        _digest_summary(records), mesh, axis_name, "distributed-weighted-csr-graph"
    )


def _global_ids_and_labels_identity(
    entity_ids: np.ndarray,
    labels: np.ndarray,
    mesh: Mesh,
    axis_name: str,
    /,
) -> str:
    records: list[dict[str, object]] = [
        {"entity_id": int(identifier), "owner": int(owner)}
        for identifier, owner in zip(entity_ids, labels, strict=True)
    ]
    return _global_digest_identity(
        _digest_summary(records), mesh, axis_name, "distributed-partition-owners"
    )


def _global_migration_identity(
    entity_ids: np.ndarray,
    previous: np.ndarray,
    owners: np.ndarray,
    weights: np.ndarray,
    mesh: Mesh,
    axis_name: str,
    /,
    graph_id: str,
    plan_id: str,
) -> tuple[str, int, int]:
    moved = np.flatnonzero(previous != owners)
    records: list[dict[str, object]] = [
        {
            "entity_id": int(entity_ids[row]),
            "source_owner": int(previous[row]),
            "target_owner": int(owners[row]),
            "vertex_weight": int(weights[row]),
        }
        for row in moved
    ]
    migration_identity = _global_digest_identity(
        _digest_summary(records), mesh, axis_name, "distributed-graph-migration"
    )
    local_summary = np.asarray(
        [moved.size, sum(int(weights[row]) for row in moved)], dtype=np.int64
    )
    gathered = _group_allgather(local_summary, mesh, axis_name)
    moved_count = sum(int(value) for value in gathered[:, 0])
    moved_weight = sum(int(value) for value in gathered[:, 1])
    return (
        canonical_fingerprint(
            {
                "kind": "distributed-graph-migration-identity",
                "graph": graph_id,
                "plan": plan_id,
                "records": migration_identity,
            }
        ),
        moved_count,
        moved_weight,
    )


def _candidate_proposals(
    graph: DistributedWeightedCSRGraph,
    labels: np.ndarray,
    edge_labels: np.ndarray,
    weights: np.ndarray,
    loads: np.ndarray,
    targets: np.ndarray,
    capacities: np.ndarray,
    protected_ids: np.ndarray,
    /,
) -> tuple[np.ndarray, ...]:
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    ids = np.asarray(graph.entity_ids, dtype=np.int64)
    edge_weights = np.asarray(graph.edge_weights, dtype=np.int64)
    proposals: list[list[tuple[int, int, int, int, int]]] = [
        [] for _ in range(len(loads))
    ]
    for row, identifier in enumerate(ids):
        source = int(labels[row])
        if protected_ids[source] == int(identifier):
            continue
        start, stop = map(int, offsets[row : row + 2])
        support = np.zeros((len(loads),), dtype=np.int64)
        for owner, weight in zip(
            edge_labels[start:stop], edge_weights[start:stop], strict=True
        ):
            support[int(owner)] += int(weight)
        source_support = int(support[source])
        source_ratio = Fraction(int(loads[source]), max(int(targets[source]), 1))
        best: tuple[int, Fraction, int] | None = None
        for target in range(len(loads)):
            if target == source or loads[target] >= capacities[target]:
                continue
            target_target = int(targets[target])
            if target_target <= 0:
                continue
            gain = int(support[target]) - source_support
            target_ratio = Fraction(int(loads[target]), target_target)
            balance_gain = source_ratio - target_ratio
            if gain <= 0 and balance_gain <= 0:
                continue
            if gain <= 0 and loads[target] >= target_target:
                continue
            candidate = (gain, balance_gain, -target)
            if best is None or candidate > best:
                best = candidate
        if best is None:
            continue
        target = -best[2]
        proposals[target].append(
            (
                int(identifier),
                source,
                target,
                int(weights[row]),
                best[0],
            )
        )
    return tuple(np.asarray(rows, dtype=np.int64).reshape((-1, 5)) for rows in proposals)


def _refinement_pass(
    graph: DistributedWeightedCSRGraph,
    labels: np.ndarray,
    weights: np.ndarray,
    targets: np.ndarray,
    capacities: np.ndarray,
    require_nonempty: bool,
    mesh: Mesh,
    axis_name: str,
    rank: int,
    /,
) -> np.ndarray:
    part_count = len(targets)
    entity_ids = np.asarray(graph.entity_ids, dtype=np.int64)
    local_rows_by_id = {int(identifier): row for row, identifier in enumerate(entity_ids)}
    edge_labels = _neighbor_labels(graph, labels, local_rows_by_id, mesh, axis_name, rank)
    loads, _ = _global_part_summary(labels, weights, part_count, mesh, axis_name)
    protected = np.full((part_count,), -1, dtype=np.int64)
    if require_nonempty:
        found, minima = _part_minimum_ids(labels, entity_ids, part_count, mesh, axis_name)
        protected[found] = minima[found]
    outgoing = _candidate_proposals(
        graph,
        labels,
        edge_labels,
        weights,
        loads,
        targets,
        capacities,
        protected,
    )
    incoming = _exchange_by_owner(outgoing, mesh, axis_name, rank)
    proposals: list[tuple[int, int, int, int, int, int]] = []
    for sender, rows in incoming.items():
        proposals.extend(
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4]), sender)
            for row in rows
        )
    proposals.sort(key=lambda item: (-item[4], item[0], item[1], item[2]))
    available = int(capacities[rank]) - int(loads[rank])
    target_limit = int(targets[rank]) - int(loads[rank])
    accepted: list[list[tuple[int, int]]] = [[] for _ in range(mesh.devices.size)]
    for identifier, source, target, weight, gain, sender in proposals:
        limit = available if gain > 0 else min(available, target_limit)
        if weight <= limit:
            accepted[sender].append((identifier, target))
            available -= weight
            target_limit -= weight
    replies = tuple(
        np.asarray(rows, dtype=np.int64).reshape((-1, 2)) for rows in accepted
    )
    received = _exchange_by_owner(replies, mesh, axis_name, rank)
    updated = labels.copy()
    failed = False
    for _, rows in received.items():
        for identifier, target in rows:
            row = local_rows_by_id.get(int(identifier))
            if row is None or int(updated[row]) == int(target):
                failed = True
            else:
                updated[row] = int(target)
    _agree_failure(
        failed,
        "A graph migration response did not match one locally owned proposal.",
        mesh,
        axis_name,
    )
    return updated


@checked
def partition_graph_distributed(
    graph: DistributedWeightedCSRGraph,
    plan: GraphPartitionPlan,
    execution_group: ExecutionGroup,
    /,
    *,
    axis_name: str = "graph",
) -> DistributedGraphPartitionResult:
    """Refine owned rows through sparse owner-routed collectives.

    The group must contain one device per process and ``plan.part_count`` must
    equal that process count. Local labels align with ``entity_ids`` and are
    target slots in ``execution_group.spec.process_indices`` order; load,
    target, capacity, cut, feasibility, and migration evidence are global.
    The graph is never gathered, and there is no serial or METIS fallback.
    """
    if not isinstance(graph, DistributedWeightedCSRGraph) or not isinstance(
        plan, GraphPartitionPlan
    ):
        raise TypeError(
            "partition_graph_distributed requires DistributedWeightedCSRGraph and GraphPartitionPlan values."
        )
    axis_name = axis_name.strip() if isinstance(axis_name, str) else axis_name
    mesh, processes, rank = _group_mesh(execution_group, axis_name)
    plan_ids = _group_allgather(
        np.frombuffer(bytes.fromhex(plan.plan_id), dtype=np.uint8),
        mesh,
        axis_name,
    )
    if not np.all(plan_ids == plan_ids[0]):
        raise ValueError(
            "Every process in execution_group must use the same GraphPartitionPlan."
        )
    parts = len(processes)
    if plan.part_count != parts:
        raise ValueError("plan.part_count must equal the execution group process count.")
    owner_rows_invalid = bool(
        np.any(np.asarray(graph.owner_ranks, dtype=np.int64) != rank)
        or np.any(np.asarray(graph.neighbor_owner_ranks, dtype=np.int64) >= parts)
        or np.any(np.asarray(graph.owner_ranks, dtype=np.int64) >= parts)
    )
    _agree_failure(
        owner_rows_invalid,
        "Distributed graph owner slots must match locally owned rows and execution_group.",
        mesh,
        axis_name,
    )
    global_vertices = _global_vertex_count(graph.vertex_count, mesh, axis_name)
    if global_vertices == 0:
        raise ValueError(
            "Distributed graph partitioning requires at least one owned vertex."
        )
    if plan.require_nonempty and plan.part_count > global_vertices:
        raise ValueError("part_count exceeds the distributed graph vertex count.")

    entity_ids = np.asarray(graph.entity_ids, dtype=np.int64)
    offsets = np.asarray(graph.offsets, dtype=np.int64)
    neighbors = np.asarray(graph.neighbors, dtype=np.int64)
    edge_weights = np.asarray(graph.edge_weights, dtype=np.int64)
    _exchange_unique_ids(entity_ids, mesh, axis_name, rank)
    _global_integer_sum(edge_weights, "edge_weights", mesh, axis_name)
    vertex_weights, total_weight = _exact_global_vertex_weights(
        graph.vertex_weights, mesh, axis_name
    )
    if total_weight <= 0:
        raise ValueError(
            "Distributed graph vertex weights must have positive total load."
        )
    targets, capacities = _targets_and_capacities(
        np.asarray((total_weight,), dtype=np.int64), plan
    )
    local_rows = _local_rows_by_id(offsets, neighbors, edge_weights, entity_ids)
    _validate_symmetric_adjacency(graph, local_rows, mesh, axis_name, rank)
    graph_id = _distributed_graph_identity(graph, vertex_weights, mesh, axis_name)

    labels = np.asarray(graph.owner_ranks, dtype=np.int32).copy()
    labels = _repair_nonempty(
        labels, entity_ids, vertex_weights, plan, mesh, axis_name, rank
    )
    for _ in range(plan.refinement_passes):
        labels = _refinement_pass(
            graph,
            labels,
            vertex_weights,
            targets,
            capacities,
            plan.require_nonempty,
            mesh,
            axis_name,
            rank,
        )
    final_edge_labels = _neighbor_labels(
        graph,
        labels,
        {int(identifier): row for row, identifier in enumerate(entity_ids)},
        mesh,
        axis_name,
        rank,
    )
    local_cut = 0
    for row, identifier in enumerate(entity_ids):
        start, stop = map(int, offsets[row : row + 2])
        for edge in range(start, stop):
            if int(identifier) < int(neighbors[edge]) and int(labels[row]) != int(
                final_edge_labels[edge]
            ):
                local_cut += 1
    cut_records = _group_allgather(
        np.asarray((local_cut,), dtype=np.int64), mesh, axis_name
    )
    edge_cut = sum(map(int, cut_records.reshape(-1)))
    part_weights, part_counts = _global_part_summary(
        labels, vertex_weights, plan.part_count, mesh, axis_name
    )
    if plan.require_nonempty and np.any(part_counts == 0):
        raise RuntimeError(
            "Distributed graph refinement returned an empty requested part."
        )
    positive = targets > 0
    imbalance = (
        float(
            max(
                Fraction(int(part_weights[index_]), int(targets[index_]))
                for index_ in np.flatnonzero(positive)
            )
        )
        if np.any(positive)
        else float("inf")
    )
    feasible = bool(np.all(part_weights <= capacities))
    owner_identity = _global_ids_and_labels_identity(entity_ids, labels, mesh, axis_name)
    migration_id, migration_count, migration_weight = _global_migration_identity(
        entity_ids,
        np.asarray(graph.owner_ranks, dtype=np.int32),
        labels,
        vertex_weights,
        mesh,
        axis_name,
        graph_id,
        plan.plan_id,
    )
    return DistributedGraphPartitionResult(
        entity_ids,
        labels,
        part_weights,
        targets,
        capacities,
        edge_cut,
        imbalance,
        feasible,
        migration_count,
        migration_weight,
        graph_id,
        plan.plan_id,
        migration_id,
        owner_identity,
    )


__all__ = [
    "DistributedGraphPartitionResult",
    "DistributedWeightedCSRGraph",
    "partition_graph_distributed",
]
