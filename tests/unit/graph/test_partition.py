import numpy as np
import pytest

from phydrax._meshcore import meshcore_available, MeshcoreError, MeshcoreStatus
from phydrax.graph import (
    GraphPartitionPlan,
    partition_graph,
    WeightedCSRGraph,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="phydrax-meshcore unavailable"
)


def _csr(
    count: int, edges: list[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Canonical CSR arrays of an undirected weighted edge list."""
    pairs = np.asarray([(u, v) for u, v, _ in edges], dtype=np.int64).reshape(-1, 2)
    weights = np.asarray([w for _, _, w in edges], dtype=np.int64)
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    both = np.concatenate((weights, weights))
    order = np.lexsort((columns, rows))
    offsets = np.zeros((count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=count), out=offsets[1:])
    return offsets, columns[order], both[order]


def _grid_edges(nx: int, ny: int, offset: int = 0) -> list[tuple[int, int, int]]:
    edges = []
    for j in range(ny):
        for i in range(nx):
            vertex = offset + j * nx + i
            if i + 1 < nx:
                edges.append((vertex, vertex + 1, 1))
            if j + 1 < ny:
                edges.append((vertex, vertex + nx, 1))
    return edges


def _grid(nx: int, ny: int) -> WeightedCSRGraph:
    offsets, neighbors, weights = _csr(nx * ny, _grid_edges(nx, ny))
    return WeightedCSRGraph(offsets, neighbors, edge_weights=weights)


def _cut(graph: WeightedCSRGraph, parts: np.ndarray) -> int:
    rows = np.repeat(np.arange(graph.vertex_count), np.diff(graph.offsets))
    crossing = parts[rows] != parts[graph.neighbors]
    return int(np.sum(graph.edge_weights[crossing])) // 2


def test_grid_bisection_finds_the_optimal_strip_cut() -> None:
    graph = _grid(32, 32)
    result = partition_graph(graph, GraphPartitionPlan(2, maximum_imbalance=1.03))
    evidence = result.evidence
    # A balanced bisection of a 32 x 32 grid cuts at least one 32-edge line.
    assert 32 <= evidence.edge_cut <= 34
    assert evidence.edge_cut == _cut(graph, np.asarray(result.parts))
    assert evidence.status == "balanced"
    assert np.all(evidence.part_weights <= evidence.part_capacities)
    assert evidence.imbalance <= 1.03
    assert evidence.boundary_vertex_count >= 2 * 32
    assert evidence.work.coarsening_levels >= 1
    assert evidence.work.adjacency_visits > 0


def test_grid_four_way_partition_approaches_quadrants() -> None:
    graph = _grid(40, 40)
    evidence = partition_graph(graph, GraphPartitionPlan(4)).evidence
    # Quadrants cut 80 edges; four strips cut 120.
    assert evidence.edge_cut <= 90
    assert evidence.status == "balanced"
    np.testing.assert_array_equal(evidence.part_vertex_counts >= 1, True)


def test_unequal_part_shares_set_the_part_weights() -> None:
    graph = _grid(30, 30)
    evidence = partition_graph(
        graph, GraphPartitionPlan(2, maximum_imbalance=1.02, part_shares=(1.0, 2.0))
    ).evidence
    np.testing.assert_allclose(evidence.part_targets, (300.0, 600.0))
    assert evidence.status == "balanced"
    assert np.all(evidence.part_weights <= np.asarray((306, 612), dtype=np.int64))
    assert np.sum(evidence.part_weights) == 900


def test_disconnected_components_are_kept_whole() -> None:
    edges = [edge for block in range(4) for edge in _grid_edges(6, 6, 36 * block)]
    graph = WeightedCSRGraph(*_csr(4 * 36, edges)[:2])
    result = partition_graph(graph, GraphPartitionPlan(4))
    assert result.evidence.edge_cut == 0
    assert result.evidence.boundary_vertex_count == 0
    np.testing.assert_array_equal(result.evidence.part_weights, 36)
    components = np.asarray(result.parts).reshape(4, 36)
    assert all(np.unique(row).size == 1 for row in components)


def test_isolated_vertices_are_balanced() -> None:
    graph = WeightedCSRGraph(np.zeros((101,), dtype=np.int64), np.zeros((0,), np.int64))
    evidence = partition_graph(graph, GraphPartitionPlan(3)).evidence
    assert evidence.edge_cut == 0
    assert evidence.status == "balanced"
    assert np.max(evidence.part_weights) <= 35


def test_edge_weights_steer_the_cut_to_the_light_seam() -> None:
    edges = [
        (u, v, 1 if (u % 16, v % 16) == (7, 8) else 50) for u, v, _ in _grid_edges(16, 8)
    ]
    offsets, neighbors, weights = _csr(128, edges)
    graph = WeightedCSRGraph(offsets, neighbors, edge_weights=weights)
    evidence = partition_graph(
        graph, GraphPartitionPlan(2, maximum_imbalance=1.0)
    ).evidence
    assert evidence.edge_cut == 8
    assert evidence.cut_edge_count == 8
    np.testing.assert_array_equal(evidence.part_weights, (64, 64))


def test_indivisible_heavy_vertex_is_reported_not_split() -> None:
    offsets, neighbors, _ = _csr(12, [(v, v + 1, 1) for v in range(11)])
    weights = np.ones((12,), dtype=np.int64)
    weights[5] = 40
    graph = WeightedCSRGraph(offsets, neighbors, vertex_weights=weights)
    result = partition_graph(graph, GraphPartitionPlan(2, maximum_imbalance=1.05))
    evidence = result.evidence
    assert evidence.status == "indivisible_vertex_overload"
    np.testing.assert_array_equal(evidence.indivisible_vertices, (5,))
    assert evidence.imbalance_lower_bound == pytest.approx(40.0 / 25.5)
    assert evidence.imbalance >= evidence.imbalance_lower_bound
    assert np.all(evidence.part_vertex_counts >= 1)
    assert evidence.part_weights[np.asarray(result.parts)[5]] >= 40


def test_partition_is_deterministic_and_independent_of_row_order() -> None:
    offsets, neighbors, weights = _csr(24 * 20, _grid_edges(24, 20))
    shuffled = neighbors.copy()
    for row in range(offsets.size - 1):
        shuffled[offsets[row] : offsets[row + 1]] = neighbors[
            offsets[row] : offsets[row + 1]
        ][::-1]
    canonical = WeightedCSRGraph(offsets, neighbors, edge_weights=weights)
    reordered = WeightedCSRGraph(offsets, shuffled, edge_weights=weights)
    assert reordered.graph_id == canonical.graph_id
    plan = GraphPartitionPlan(3)
    first = partition_graph(canonical, plan)
    second = partition_graph(reordered, plan)
    np.testing.assert_array_equal(first.parts, second.parts)
    assert first.result_id == second.result_id
    assert first.evidence.work == second.evidence.work


def test_nonempty_policy_gives_every_part_a_vertex() -> None:
    offsets, neighbors, _ = _csr(5, [(v, v + 1, 1) for v in range(4)])
    graph = WeightedCSRGraph(offsets, neighbors, vertex_weights=(1, 0, 0, 0, 0))
    result = partition_graph(graph, GraphPartitionPlan(5))
    np.testing.assert_array_equal(np.sort(np.asarray(result.parts)), np.arange(5))
    assert result.evidence.empty_part_count == 0
    with pytest.raises(ValueError, match="exceeds the vertex count"):
        partition_graph(graph, GraphPartitionPlan(6))
    permitted = partition_graph(graph, GraphPartitionPlan(6, empty_parts="permit_empty"))
    assert permitted.evidence.empty_part_count >= 1


def test_work_limit_is_a_resource_refusal() -> None:
    with pytest.raises(MeshcoreError, match="work limit 50") as refusal:
        partition_graph(_grid(20, 20), GraphPartitionPlan(2, work_limit=50))
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_isolated_vertex_attempts_consume_the_work_limit() -> None:
    graph = WeightedCSRGraph(np.zeros((101,), dtype=np.int64), np.empty((0,), np.int32))
    with pytest.raises(MeshcoreError) as refusal:
        partition_graph(graph, GraphPartitionPlan(2, work_limit=5))
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_weighted_disconnected_packing_meets_exact_capacity() -> None:
    graph = WeightedCSRGraph(
        np.zeros((6,), dtype=np.int64),
        np.empty((0,), np.int32),
        vertex_weights=(8, 7, 6, 5, 4),
    )
    result = partition_graph(graph, GraphPartitionPlan(2, maximum_imbalance=1.0))
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(result.evidence.part_weights, (15, 15))
    assert result.evidence.edge_cut == 0
    assert np.all(result.evidence.part_vertex_counts > 0)
    assert result.evidence.work.adjacency_visits == 0
    assert result.evidence.work.candidate_evaluations > 0


def test_multilevel_projection_keeps_exact_fine_capacities() -> None:
    result = partition_graph(_grid(32, 32), GraphPartitionPlan(8, maximum_imbalance=1.0))
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(
        result.evidence.part_weights, np.full((8,), 128, np.int64)
    )
    np.testing.assert_array_equal(
        result.evidence.part_capacities, np.full((8,), 128, np.int64)
    )


def test_impossible_packing_is_not_reported_as_balanced() -> None:
    graph = WeightedCSRGraph(
        np.zeros((4,), dtype=np.int64),
        np.empty((0,), np.int32),
        vertex_weights=(2, 2, 2),
    )
    result = partition_graph(graph, GraphPartitionPlan(2, maximum_imbalance=1.0))
    assert result.evidence.status == "balance_not_reached"
    assert result.evidence.indivisible_vertices.size == 0
    np.testing.assert_array_equal(result.evidence.part_capacities, (3, 3))
    assert np.max(result.evidence.part_weights) == 4
    assert np.all(result.evidence.part_vertex_counts > 0)


def test_integer_weight_total_above_exact_limit_is_refused() -> None:
    with pytest.raises(ValueError, match="2\\*\\*53"):
        WeightedCSRGraph(
            np.zeros((3,), dtype=np.int64),
            np.empty((0,), np.int32),
            vertex_weights=(2**53, 1),
        )


def test_disconnected_kway_matches_independently_constructed_grid_cut() -> None:
    edges = _grid_edges(16, 16) + _grid_edges(16, 16, 256)
    graph = WeightedCSRGraph(*_csr(512, edges)[:2])
    result = partition_graph(graph, GraphPartitionPlan(32, maximum_imbalance=1.0))
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(
        result.evidence.part_weights, np.full((32,), 16, np.int64)
    )
    # A 4x4 block tiling of each disconnected 16x16 component cuts 96 edges.
    assert result.evidence.edge_cut <= 192
    assert set(np.asarray(result.parts)[:256]).isdisjoint(
        set(np.asarray(result.parts)[256:])
    )


def test_disconnected_spatial_kway_matches_independent_grid_cut() -> None:
    edges = []
    for block in range(2):
        for vertex in range(512):
            source = vertex + 512 * block
            if vertex % 8 < 7:
                edges.append((source, source + 1, 1))
            if (vertex // 8) % 8 < 7:
                edges.append((source, source + 8, 1))
            if vertex // 64 < 7:
                edges.append((source, source + 64, 1))
    graph = WeightedCSRGraph(*_csr(1024, edges)[:2])
    result = partition_graph(graph, GraphPartitionPlan(32, maximum_imbalance=1.0))
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(
        result.evidence.part_weights, np.full((32,), 32, np.int64)
    )
    # Independent 4x2x2 block arrangements cut 320 edges per component.
    assert result.evidence.edge_cut <= 640


def test_component_weights_match_unequal_capacity_parts() -> None:
    offsets, neighbors, _ = _csr(4, [(0, 1, 1), (2, 3, 1)])
    graph = WeightedCSRGraph(offsets, neighbors, vertex_weights=(4, 4, 1, 1))
    result = partition_graph(
        graph, GraphPartitionPlan(2, maximum_imbalance=1.0, part_shares=(1.0, 4.0))
    )
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(result.evidence.part_weights, (2, 8))
    assert result.evidence.edge_cut == 0


_PATH = _csr(4, [(0, 1, 1), (1, 2, 1), (2, 3, 1)])


@pytest.mark.parametrize(
    ("offsets", "neighbors", "options", "error", "message"),
    [
        pytest.param(
            _PATH[0],
            _PATH[1],
            {"edge_weights": (2, 1, 1, 1, 1, 1)},
            ValueError,
            "symmetric",
            id="asymmetric-weight",
        ),
        pytest.param((0, 1, 1), (1,), {}, ValueError, "symmetric", id="missing-mirror"),
        pytest.param((0, 1, 2), (0, 0), {}, ValueError, "self loops", id="self-loop"),
        pytest.param(
            (0, 2, 4), (1, 1, 0, 0), {}, ValueError, "repeat", id="repeated-neighbor"
        ),
        pytest.param(
            (0, 1, 2), (1, 2), {}, ValueError, "vertex indices", id="out-of-range"
        ),
        pytest.param((1, 2, 2), (1, 0), {}, ValueError, "offsets", id="bad-offsets"),
        pytest.param(
            _PATH[0],
            _PATH[1],
            {"vertex_weights": (1.0, 1.0, 1.0, 1.0)},
            TypeError,
            "integer",
            id="real-weights",
        ),
        pytest.param(
            _PATH[0],
            _PATH[1],
            {"vertex_weights": (1, -1, 1, 1)},
            ValueError,
            "nonnegative",
            id="negative-weight",
        ),
        pytest.param(
            _PATH[0],
            _PATH[1],
            {"vertex_weights": (0, 0, 0, 0)},
            ValueError,
            "positive total",
            id="zero-total",
        ),
        pytest.param(
            _PATH[0],
            _PATH[1],
            {"edge_weights": (2**53,) * 6},
            ValueError,
            "2\\*\\*53",
            id="weight-overflow",
        ),
    ],
)
def test_invalid_csr_is_refused(
    offsets: object,
    neighbors: object,
    options: dict[str, object],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        WeightedCSRGraph(offsets, neighbors, **options)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    ("arguments", "options", "error"),
    [
        pytest.param(
            (2,), {"maximum_imbalance": 0.9}, ValueError, id="imbalance-below-one"
        ),
        pytest.param((2,), {"empty_parts": "allow"}, ValueError, id="unknown-policy"),
        pytest.param((2,), {"part_shares": (1.0,)}, ValueError, id="share-count"),
        pytest.param((2,), {"part_shares": (1.0, 0.0)}, ValueError, id="zero-share"),
        pytest.param((0,), {}, ValueError, id="no-parts"),
        pytest.param((True,), {}, TypeError, id="boolean-parts"),
    ],
)
def test_invalid_plans_are_refused(
    arguments: tuple[object, ...], options: dict[str, object], error: type[Exception]
) -> None:
    with pytest.raises(error):
        GraphPartitionPlan(*arguments, **options)  # ty: ignore[invalid-argument-type]


def _distributed_graph(
    weights: np.ndarray,
    edges: list[tuple[int, int, int]],
    *,
    active_count: int | None = None,
    corrupt: bool = False,
    ghost: bool = False,
    corrupt_ghost: bool = False,
) -> tuple[WeightedCSRGraph, np.ndarray, np.ndarray]:
    import jax
    from jax.sharding import Mesh, NamedSharding, PartitionSpec

    if jax.device_count() < 2:
        pytest.skip("Requires two or more real CPU devices.")
    ranks = jax.device_count()
    capacity = len(weights) // ranks
    weights = np.asarray(weights, np.int64).copy()
    ids = (np.arange(len(weights), dtype=np.int64) * 17 + 9).reshape(ranks, capacity)
    count = len(weights) if active_count is None else active_count
    offsets = np.zeros((ranks, capacity + 1), np.int64)
    neighbors = np.full((ranks, max(1, 2 * len(edges))), 2**63 - 1, np.int64)
    edge_weights = np.zeros_like(neighbors)
    for rank in range(ranks):
        position = 0
        for row in range(capacity):
            vertex = rank * capacity + row
            for left, right, weight in edges:
                other = right if left == vertex else left if right == vertex else None
                if other is not None:
                    neighbors[rank, position] = ids.flat[other]
                    edge_weights[rank, position] = weight
                    position += 1
            offsets[rank, row + 1] = position
    if corrupt:
        edge_weights[0, 0] += 1
    owners = np.broadcast_to(np.arange(ranks, dtype=np.int32)[:, None], ids.shape).copy()
    valid = np.arange(len(weights)).reshape(ids.shape) < count
    if ghost:
        ids[-1, -1] = ids[0, 0]
        owners[-1, -1] = 0
        valid[-1, -1] = True
        weights[-1] = weights[0] + int(corrupt_ghost)
    mesh = Mesh(np.asarray(jax.devices()), ("parts",))
    sharding = NamedSharding(mesh, PartitionSpec("parts"))
    put = lambda value: jax.device_put(value, sharding)
    graph = WeightedCSRGraph.owner_local(
        put(offsets),
        put(neighbors),
        put(ids),
        put(owners),
        put(valid),
        mesh=mesh,
        axis_name="parts",
        edge_weights=put(edge_weights),
        vertex_weights=put(np.asarray(weights, np.int64).reshape(ids.shape)),
    )
    return graph, owners, ids


def test_distributed_partition_balances_and_measures_actual_cut() -> None:
    import jax

    count = jax.device_count() * 4
    edges = [(vertex, vertex + 1, 3) for vertex in range(count - 1)]
    graph, owners, _ = _distributed_graph(np.ones(count, np.int64), edges)
    result = partition_graph(
        graph, GraphPartitionPlan(jax.device_count(), refinement_passes=2)
    )
    parts = np.asarray(result.parts).ravel()
    assert result.evidence.accepted
    assert result.evidence.status == "balanced"
    np.testing.assert_array_equal(result.evidence.part_weights, 4)
    assert result.evidence.edge_cut == sum(
        weight for left, right, weight in edges if parts[left] != parts[right]
    )
    assert result.evidence.migration_vertex_count == np.count_nonzero(
        result.parts != owners
    )
    assert result.evidence.work.matched_pairs > 0
    assert result.evidence.work.coarsening_levels > 0
    assert result.evidence.ghost_entry_count == 2 * result.evidence.cut_edge_count


def test_distributed_disconnected_unequal_weights_and_indivisible_vertex() -> None:
    import jax

    ranks = jax.device_count()
    weights = np.tile((1, 2, 1, 2), ranks)
    graph, _, _ = _distributed_graph(weights, [])
    shares = tuple(range(1, ranks + 1))
    result = partition_graph(
        graph,
        GraphPartitionPlan(
            ranks, part_shares=shares, maximum_imbalance=1.5, refinement_passes=2
        ),
    )
    assert result.evidence.accepted and result.evidence.status == "balanced"
    assert result.evidence.edge_cut == 0
    np.testing.assert_allclose(
        result.evidence.part_targets, sum(weights) * np.asarray(shares) / sum(shares)
    )
    assert np.all(result.evidence.part_weights <= result.evidence.part_capacities)
    weights[0] = 100
    heavy, _, ids = _distributed_graph(weights, [])
    overloaded = partition_graph(heavy, GraphPartitionPlan(ranks, refinement_passes=1))
    assert overloaded.evidence.accepted
    assert overloaded.evidence.status == "indivisible_vertex_overload"
    assert np.asarray(overloaded.evidence.indivisible_vertices)[0, 0] == ids[0, 0]
    assert np.max(overloaded.evidence.part_weights) >= 100


def test_distributed_collective_refusals_preserve_original_owners() -> None:
    import jax

    ranks = jax.device_count()
    weights = np.ones(ranks * 4, np.int64)
    edges = [(vertex, vertex + 1, 1) for vertex in range(weights.size - 1)]
    graph, owners, _ = _distributed_graph(weights, edges)
    refused = partition_graph(graph, GraphPartitionPlan(ranks, work_limit=1))
    assert not refused.evidence.accepted
    assert refused.evidence.resource_status & 2
    np.testing.assert_array_equal(refused.parts, owners)
    invalid, owners, _ = _distributed_graph(weights, edges, corrupt=True)
    refused = partition_graph(invalid, GraphPartitionPlan(ranks, refinement_passes=1))
    assert not refused.evidence.accepted
    assert refused.evidence.resource_status & 1
    np.testing.assert_array_equal(refused.parts, owners)


def test_distributed_empty_policy_is_collective() -> None:
    import jax

    ranks = jax.device_count()
    graph, owners, _ = _distributed_graph(
        np.ones(ranks * 2, np.int64), [], active_count=1
    )
    required = partition_graph(graph, GraphPartitionPlan(ranks, refinement_passes=0))
    assert not required.evidence.accepted and required.evidence.resource_status & 1
    np.testing.assert_array_equal(required.parts, owners)
    permitted = partition_graph(
        graph, GraphPartitionPlan(ranks, empty_parts="permit_empty", refinement_passes=0)
    )
    assert permitted.evidence.accepted
    assert permitted.evidence.empty_part_count == ranks - 1
    assert permitted.evidence.status == "balanced"


def test_distributed_ghost_copy_tracks_accepted_owner_and_refuses_stale_weight() -> None:
    import jax

    ranks = jax.device_count()
    weights = np.ones(ranks * 3, np.int64)
    weights[0] = 2
    graph, _, _ = _distributed_graph(weights, [], ghost=True)
    result = partition_graph(graph, GraphPartitionPlan(ranks, refinement_passes=1))
    assert result.evidence.accepted
    assert np.asarray(result.parts)[-1, -1] == np.asarray(result.parts)[0, 0]
    assert np.sum(result.evidence.part_weights) == np.sum(weights[:-1])
    invalid, owners, _ = _distributed_graph(weights, [], ghost=True, corrupt_ghost=True)
    refused = partition_graph(invalid, GraphPartitionPlan(ranks, refinement_passes=1))
    assert not refused.evidence.accepted and refused.evidence.resource_status & 1
    np.testing.assert_array_equal(refused.parts, owners)


def test_distributed_nonempty_singleton_parts_cannot_be_refined_away() -> None:
    import jax

    ranks = jax.device_count()
    graph, _, _ = _distributed_graph(
        np.ones(ranks * 2, np.int64),
        [(vertex, vertex + 1, 3) for vertex in range(ranks - 1)],
        active_count=ranks,
    )
    result = partition_graph(
        graph, GraphPartitionPlan(ranks, maximum_imbalance=10.0, refinement_passes=2)
    )
    assert result.evidence.accepted
    np.testing.assert_array_equal(result.evidence.part_vertex_counts, 1)


def test_distributed_semantic_priorities_ignore_local_slot_order() -> None:
    import jax

    count = jax.device_count() * 4
    graph, _, _ = _distributed_graph(
        np.ones(count, np.int64), [(vertex, vertex + 1, 5) for vertex in range(count - 1)]
    )
    assert graph.vertex_ids is not None
    assert graph.mesh is not None
    assert graph.axis_name is not None
    plan = GraphPartitionPlan(jax.device_count(), refinement_passes=1)
    original = partition_graph(graph, plan)
    permutation = np.asarray((2, 0, 3, 1))
    offsets = np.asarray(graph.offsets)
    neighbors = np.asarray(graph.neighbors)
    weights = np.asarray(graph.edge_weights)
    reordered_offsets = np.zeros_like(offsets)
    reordered_neighbors = np.full_like(neighbors, 2**63 - 1)
    reordered_weights = np.zeros_like(weights)
    for rank in range(jax.device_count()):
        position = 0
        for slot, old in enumerate(permutation):
            first, last = offsets[rank, old : old + 2]
            width = last - first
            reordered_neighbors[rank, position : position + width] = neighbors[
                rank, first:last
            ]
            reordered_weights[rank, position : position + width] = weights[
                rank, first:last
            ]
            position += width
            reordered_offsets[rank, slot + 1] = position
    placement = graph.vertex_ids.sharding
    put = lambda value: jax.device_put(value, placement)
    reordered = WeightedCSRGraph.owner_local(
        put(reordered_offsets),
        put(reordered_neighbors),
        put(np.asarray(graph.vertex_ids)[:, permutation]),
        put(np.asarray(graph.vertex_owners)[:, permutation]),
        put(np.asarray(graph.vertex_valid)[:, permutation]),
        mesh=graph.mesh,
        axis_name=graph.axis_name,
        edge_weights=put(reordered_weights),
        vertex_weights=put(np.asarray(graph.vertex_weights)[:, permutation]),
    )
    changed = partition_graph(reordered, plan)
    assert original.evidence.accepted and changed.evidence.accepted
    np.testing.assert_array_equal(
        np.asarray(changed.parts)[:, np.argsort(permutation)], original.parts
    )
    assert changed.evidence.edge_cut == original.evidence.edge_cut


def test_distributed_graph_refuses_implicit_whole_graph_resharding() -> None:
    import jax

    count = jax.device_count() * 2
    graph, _, _ = _distributed_graph(np.ones(count, np.int64), [])
    assert isinstance(graph.offsets, jax.Array)
    assert isinstance(graph.edge_weights, jax.Array)
    assert isinstance(graph.vertex_weights, jax.Array)
    assert graph.vertex_ids is not None
    assert graph.vertex_owners is not None
    assert graph.vertex_valid is not None
    assert graph.mesh is not None
    assert graph.axis_name is not None
    replicated_neighbors = jax.device_put(np.asarray(graph.neighbors), jax.devices()[0])
    with pytest.raises(ValueError, match="rank-row sharding"):
        WeightedCSRGraph.owner_local(
            graph.offsets,
            replicated_neighbors,
            graph.vertex_ids,
            graph.vertex_owners,
            graph.vertex_valid,
            mesh=graph.mesh,
            axis_name=graph.axis_name,
            vertex_weights=graph.vertex_weights,
            edge_weights=graph.edge_weights,
        )
