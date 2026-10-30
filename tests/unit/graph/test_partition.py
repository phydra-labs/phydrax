from __future__ import annotations

import jax
import numpy as np
import pytest
from jax.experimental import multihost_utils

from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.graph import (
    DistributedWeightedCSRGraph,
    GraphPartitionPlan,
    partition_graph,
    partition_graph_distributed,
    WeightedCSRGraph,
)


def _graph(ids: np.ndarray | None = None) -> WeightedCSRGraph:
    # Two disconnected paths force stable-ID ordering to be independent of the
    # storage permutation without prescribing a particular optimal partition.
    return WeightedCSRGraph.from_edges(
        6,
        np.asarray(((0, 1), (1, 2), (3, 4), (4, 5)), dtype=np.int64),
        vertex_weights=np.asarray((1, 2, 1, 1, 2, 1), dtype=np.int64),
        entity_ids=ids,
    )


def _rank_symmetric_group() -> ExecutionGroup:
    processes = tuple(range(jax.process_count()))
    first_device: dict[int, jax.Device] = {}
    for device in sorted(
        jax.devices(), key=lambda value: (value.process_index, value.id)
    ):
        first_device.setdefault(int(device.process_index), device)
    devices = tuple(first_device[process] for process in processes)
    keys = tuple((int(device.process_index), int(device.id)) for device in devices)
    spec = ExecutionGroupSpec(
        f"graph-partition-test-{len(processes)}",
        processes,
        keys,
        mesh_axes=(("graph_test", len(devices)),),
    )
    return ExecutionGroup(spec, devices)


def test_native_partition_reports_measured_labels_and_cut() -> None:
    graph = _graph()
    result = partition_graph(graph, GraphPartitionPlan(2, refinement_passes=2))

    owners = np.asarray(result.owners)
    assert owners.shape == (graph.vertex_count,)
    assert np.array_equal(np.unique(owners), np.asarray((0, 1)))
    assert np.array_equal(
        np.asarray(result.part_weights).sum(), np.asarray(graph.vertex_weights).sum()
    )
    assert result.edge_cut >= 0
    assert result.provenance.startswith("phydrax-meshcore ")


def test_native_partition_is_stable_under_canonical_id_row_permutation() -> None:
    ids = np.asarray((30, 10, 20, 60, 40, 50), dtype=np.int64)
    weights = np.asarray((1, 2, 1, 1, 2, 1), dtype=np.int64)
    edges = np.asarray(((0, 1), (1, 2), (3, 4), (4, 5)), dtype=np.int64)
    first = WeightedCSRGraph.from_edges(6, edges, vertex_weights=weights, entity_ids=ids)
    permutation = np.asarray((4, 0, 5, 1, 3, 2), dtype=np.int64)
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size)
    permuted = WeightedCSRGraph.from_edges(
        6,
        inverse[edges],
        vertex_weights=weights[permutation],
        entity_ids=ids[permutation],
    )
    plan = GraphPartitionPlan(2)
    expected = np.asarray(partition_graph(first, plan).owners)
    actual = np.asarray(partition_graph(permuted, plan).owners)
    assert np.array_equal(actual[inverse], expected)


def test_weighted_graph_rejects_asymmetric_csr() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        WeightedCSRGraph(
            np.asarray((0, 1, 1), dtype=np.int64),
            np.asarray((1,), dtype=np.int32),
        )


def test_distributed_partition_one_worker_matches_serial_and_oracle() -> None:
    if jax.process_count() != 1:
        pytest.skip("The direct graph partition smoke requires one JAX process.")
    group = _rank_symmetric_group()
    graph = DistributedWeightedCSRGraph(
        np.asarray((0, 1, 2, 4), dtype=np.int64),
        np.asarray((20, 20, 10, 30), dtype=np.int64),
        entity_ids=np.asarray((30, 10, 20), dtype=np.int64),
        owner_ranks=np.zeros((3,), dtype=np.int64),
        neighbor_owner_ranks=np.zeros((4,), dtype=np.int64),
        edge_weights=np.asarray((2, 4, 4, 2), dtype=np.int64),
        vertex_weights=np.asarray((1.5, 1.0, 0.5), dtype=np.float64),
    )
    plan = GraphPartitionPlan(1, maximum_imbalance=1.0)
    result = partition_graph_distributed(graph, plan, group)

    ids = np.asarray(result.entity_ids, dtype=np.int64)
    owners = np.asarray(result.owners, dtype=np.int64)
    edge_pairs = ((10, 20), (20, 30))
    weight_by_id = {10: 2, 20: 1, 30: 3}
    oracle_load = sum(
        weight_by_id[int(identifier)]
        for identifier, owner in zip(ids, owners, strict=True)
        if owner == 0
    )
    oracle_loads = np.asarray((oracle_load,), dtype=np.int64)
    owner_by_id = dict(zip(ids.tolist(), owners.tolist(), strict=True))
    oracle_cut = sum(
        owner_by_id[left] != owner_by_id[right] for left, right in edge_pairs
    )
    oracle_feasible = bool(np.all(oracle_loads <= np.asarray(result.capacity_weights)))
    assert np.array_equal(ids, np.asarray((10, 20, 30)))
    assert np.array_equal(owners, np.asarray((0, 0, 0)))
    assert np.array_equal(np.asarray(result.part_weights), oracle_loads)
    assert np.array_equal(np.asarray(result.target_weights), np.asarray((6,)))
    assert np.array_equal(np.asarray(result.capacity_weights), np.asarray((6,)))
    assert result.edge_cut == oracle_cut == 0
    assert result.feasible and oracle_feasible
    assert result.migration_count == 0
    assert result.migration_weight == 0

    serial = WeightedCSRGraph.from_edges(
        3,
        np.asarray(((0, 1), (1, 2)), dtype=np.int64),
        edge_weights=np.asarray((4, 2), dtype=np.int64),
        vertex_weights=np.asarray((1.0, 0.5, 1.5), dtype=np.float64),
        entity_ids=np.asarray((10, 20, 30), dtype=np.int64),
    )
    serial_result = partition_graph(serial, plan)
    assert np.array_equal(owners, np.asarray(serial_result.owners))
    assert np.array_equal(
        np.asarray(result.part_weights), np.asarray(serial_result.part_weights)
    )
    assert result.edge_cut == serial_result.edge_cut
    assert result.feasible is serial_result.feasible

    equivalent_weights = DistributedWeightedCSRGraph(
        np.asarray((0, 1, 2, 4), dtype=np.int64),
        np.asarray((20, 20, 10, 30), dtype=np.int64),
        entity_ids=np.asarray((30, 10, 20), dtype=np.int64),
        owner_ranks=np.zeros((3,), dtype=np.int64),
        neighbor_owner_ranks=np.zeros((4,), dtype=np.int64),
        edge_weights=np.asarray((2, 4, 4, 2), dtype=np.int64),
        vertex_weights=np.asarray((3.0, 2.0, 1.0), dtype=np.float64),
    )
    repeated = partition_graph_distributed(equivalent_weights, plan, group)
    assert repeated.graph_id == result.graph_id
    assert repeated.migration_id == result.migration_id
    assert repeated.result_id == result.result_id


def test_distributed_partition_two_process_sparse_migration() -> None:
    if jax.process_count() != 2:
        pytest.skip("The owner-local migration case requires two JAX processes.")
    group = _rank_symmetric_group()
    rank = group.spec.process_indices.index(int(jax.process_index()))
    all_ids = np.asarray((10, 20, 30, 40, 50, 60), dtype=np.int64)
    owner_by_id = {
        int(identifier): int(index >= 4) for index, identifier in enumerate(all_ids)
    }
    adjacency = {
        10: (20,),
        20: (10, 30),
        30: (20, 40),
        40: (30, 50),
        50: (40, 60),
        60: (50,),
    }
    owned_ids = all_ids[:4] if rank == 0 else all_ids[4:]
    all_vertex_weights = np.asarray((1.0, 1.0, 1.0, 1.0, 1.5, 0.5), dtype=np.float64)
    owned_weights = all_vertex_weights[:4] if rank == 0 else all_vertex_weights[4:]
    neighbors: list[int] = []
    neighbor_owners: list[int] = []
    edge_weights: list[int] = []
    offsets = [0]
    for identifier in owned_ids:
        for neighbor in adjacency[int(identifier)]:
            neighbors.append(neighbor)
            neighbor_owners.append(owner_by_id[neighbor])
            edge_weights.append(1)
        offsets.append(len(neighbors))
    graph = DistributedWeightedCSRGraph(
        np.asarray(offsets, dtype=np.int64),
        np.asarray(neighbors, dtype=np.int64),
        entity_ids=owned_ids,
        owner_ranks=np.full((owned_ids.size,), rank, dtype=np.int64),
        neighbor_owner_ranks=np.asarray(neighbor_owners, dtype=np.int64),
        edge_weights=np.asarray(edge_weights, dtype=np.int64),
        vertex_weights=owned_weights,
    )
    plan = GraphPartitionPlan(2, maximum_imbalance=1.0)
    result = partition_graph_distributed(graph, plan, group)

    ids = np.asarray(result.entity_ids, dtype=np.int64)
    labels = np.asarray(result.owners, dtype=np.int64)
    assert np.array_equal(
        labels, np.asarray([int(identifier >= 40) for identifier in ids])
    )
    assert result.migration_count == 1
    assert result.migration_weight == 2
    assert np.array_equal(np.asarray(result.part_weights), np.asarray((6, 6)))
    assert np.array_equal(np.asarray(result.target_weights), np.asarray((6, 6)))
    assert np.array_equal(np.asarray(result.capacity_weights), np.asarray((6, 6)))
    assert result.feasible
    repeated = partition_graph_distributed(graph, plan, group)
    weighted_plan = GraphPartitionPlan(
        2,
        target_shares=np.asarray((2.0, 1.0), dtype=np.float64),
        maximum_imbalance=1.0,
    )
    weighted = partition_graph_distributed(graph, weighted_plan, group)
    assert np.array_equal(np.asarray(weighted.part_weights), np.asarray((8, 4)))
    assert np.array_equal(np.asarray(weighted.target_weights), np.asarray((8, 4)))
    assert np.array_equal(np.asarray(weighted.capacity_weights), np.asarray((8, 4)))
    assert weighted.migration_count == 0
    assert weighted.feasible
    assert repeated.migration_id == result.migration_id
    assert repeated.result_id == result.result_id

    local_labels = np.full((4, 2), -1, dtype=np.int64)
    local_labels[: ids.size, 0] = ids
    local_labels[: ids.size, 1] = labels
    gathered_labels = np.asarray(
        multihost_utils.process_allgather(local_labels, tiled=False)
    ).reshape((2, 4, 2))
    owner_by_id = {
        int(identifier): int(owner)
        for identifier, owner in gathered_labels.reshape((-1, 2))
        if identifier >= 0
    }
    edges = ((10, 20), (20, 30), (30, 40), (40, 50), (50, 60))
    weight_by_id = {10: 2, 20: 2, 30: 2, 40: 2, 50: 3, 60: 1}
    oracle_loads = np.zeros((2,), dtype=np.int64)
    for identifier, owner in owner_by_id.items():
        oracle_loads[owner] += weight_by_id[identifier]
    oracle_cut = sum(owner_by_id[left] != owner_by_id[right] for left, right in edges)
    oracle_feasible = bool(np.all(oracle_loads <= np.asarray(result.capacity_weights)))
    assert set(owner_by_id) == set(all_ids.tolist())
    assert np.array_equal(np.asarray(result.part_weights), oracle_loads)
    assert result.edge_cut == oracle_cut
    assert result.feasible is oracle_feasible

    serial = WeightedCSRGraph.from_edges(
        all_ids.size,
        np.asarray(tuple((index, index + 1) for index in range(all_ids.size - 1))),
        entity_ids=all_ids,
        vertex_weights=all_vertex_weights,
    )
    serial_result = partition_graph(serial, plan)
    assert np.array_equal(
        np.sort(np.asarray(result.part_weights)),
        np.sort(np.asarray(serial_result.part_weights)),
    )
    assert result.edge_cut == serial_result.edge_cut
    assert result.feasible is serial_result.feasible
    serial_weighted_result = partition_graph(serial, weighted_plan)
    assert np.array_equal(
        np.sort(np.asarray(weighted.part_weights)),
        np.sort(np.asarray(serial_weighted_result.part_weights)),
    )
    assert weighted.edge_cut == serial_weighted_result.edge_cut
    assert weighted.feasible is serial_weighted_result.feasible
    identities = np.frombuffer(bytes.fromhex(result.result_id), dtype=np.uint8)
    assert np.all(
        np.asarray(multihost_utils.process_allgather(identities, tiled=False))
        == identities
    )
