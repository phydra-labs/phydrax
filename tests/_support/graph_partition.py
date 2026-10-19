"""Host CSR and physical owner-row builders for partition contracts."""

import jax
import numpy as np
import pytest
from jax import Array
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax.graph import WeightedCSRGraph


def csr(
    count: int, edges: list[tuple[int, int, int]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sorted CSR arrays from an undirected weighted edge list."""
    pairs = np.asarray([(u, v) for u, v, _ in edges], dtype=np.int64).reshape(-1, 2)
    weights = np.asarray([w for _, _, w in edges], dtype=np.int64)
    rows = np.concatenate((pairs[:, 0], pairs[:, 1]))
    columns = np.concatenate((pairs[:, 1], pairs[:, 0]))
    both = np.concatenate((weights, weights))
    order = np.lexsort((columns, rows))
    offsets = np.zeros((count + 1,), dtype=np.int64)
    np.cumsum(np.bincount(rows, minlength=count), out=offsets[1:])
    return offsets, columns[order], both[order]


def grid_edges(nx: int, ny: int, offset: int = 0) -> list[tuple[int, int, int]]:
    edges = []
    for j in range(ny):
        for i in range(nx):
            vertex = offset + j * nx + i
            if i + 1 < nx:
                edges.append((vertex, vertex + 1, 1))
            if j + 1 < ny:
                edges.append((vertex, vertex + nx, 1))
    return edges


def grid(nx: int, ny: int) -> WeightedCSRGraph:
    offsets, neighbors, weights = csr(nx * ny, grid_edges(nx, ny))
    return WeightedCSRGraph(offsets, neighbors, edge_weights=weights)


def distributed_graph(
    weights: np.ndarray,
    edges: list[tuple[int, int, int]],
    *,
    active_count: int | None = None,
    corrupt: bool = False,
    ghost: bool = False,
    corrupt_ghost: bool = False,
) -> tuple[WeightedCSRGraph, np.ndarray, np.ndarray]:
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

    def put(value: np.ndarray) -> Array:
        return jax.device_put(value, sharding)

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
