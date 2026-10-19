# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Forced-device receipts for forest routing and raw vertex record ownership."""

from __future__ import annotations

import jax
import numpy as np
import pytest
from jax import Array
from jax.sharding import NamedSharding, PartitionSpec

from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    AdaptiveSimplexState,
)
from phydrax.meshing._distribution_forest import (
    forest_vertex_record_owners,
    vertex_record_owners,
)
from phydrax.meshing._distribution_migration import (
    forest_migration_routes,
    prepare_migration_routes,
)
from tools.compile_counter import CompileCounter


def _forest(
    vertex_rows: np.ndarray, parts: AdaptiveSimplexParts, protected_capacity: int
) -> AdaptiveSimplexState:
    layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=4,
        cell_capacity=1,
        protected_edge_capacity=protected_capacity,
        maximum_closure_iterations=1,
        maximum_coarsening_passes=1,
    )
    cells = np.asarray(((0, 1, 2),), dtype=np.int32)

    def piece(rank: int) -> AdaptiveSimplexState:
        ids = vertex_rows[rank, :3]
        coordinates = np.stack((ids, ids * ids), axis=1).astype(np.float64)
        return adaptive_simplex_state(
            layout,
            coordinates=coordinates,
            vertex_ids=ids,
            vertex_active=np.ones(3, dtype=np.bool_),
            vertex_parents=np.full((3, 2), -1, dtype=np.int32),
            vertex_protected=np.zeros(3, dtype=np.bool_),
            cells=cells,
            tuples=cells,
            tags=np.full(1, 2, dtype=np.int32),
            blocks=np.zeros(1, dtype=np.int32),
            generations=np.zeros(1, dtype=np.int32),
            parents=np.full(1, -1, dtype=np.int32),
            children=np.full((1, 2), -1, dtype=np.int32),
            bisection_vertices=np.full(1, -1, dtype=np.int32),
            cell_ids=np.asarray((rank + 40,), dtype=np.int64),
            cell_active=np.ones(1, dtype=np.bool_),
            cell_classes=np.zeros(1, dtype=np.int32),
            facet_classes=np.zeros((1, 3), dtype=np.int32),
            protected_edges=np.empty((0, 2), dtype=np.int32),
            next_vertex_id=32,
            next_cell_id=43,
        )

    pieces = tuple(piece(rank) for rank in range(parts.part_count))
    sharding = NamedSharding(parts.mesh, PartitionSpec(parts.axis_name))
    return jax.tree_util.tree_map(
        lambda *values: jax.device_put(np.stack(values), sharding), *pieces
    )


def _expected(vertex_rows: np.ndarray, operation: str) -> np.ndarray:
    inventories = tuple(set(row[row >= 0].tolist()) for row in vertex_rows)
    if operation == "routes":
        return np.asarray(
            [
                [
                    source != target and bool(left & right)
                    for target, right in enumerate(inventories)
                ]
                for source, left in enumerate(inventories)
            ],
            dtype=np.bool_,
        )
    return np.asarray(
        [
            [
                min(rank for rank, ids in enumerate(inventories) if vertex in ids)
                if vertex >= 0
                else -1
                for vertex in row
            ]
            for row in vertex_rows
        ],
        dtype=np.int32,
    )


def compile_reuse(operation: str) -> dict[str, object]:
    count = len(jax.devices())
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=())
    first_rows = np.asarray(
        ((2, 5, 11, -1), (5, 17, 23, -1), (11, 29, 31, -1)), dtype=np.int64
    )[:count]
    changed_rows = np.asarray(
        ((2, 17, 23, -1), (5, 11, 29, -1), (5, 29, 31, -1)), dtype=np.int64
    )[:count]
    first_state = _forest(first_rows, parts, 2)
    projected_state = _forest(first_rows, parts, 1)
    changed_state = _forest(changed_rows, parts, 1)

    def run(placement: AdaptiveSimplexParts, state: AdaptiveSimplexState) -> Array:
        if operation == "owners":
            return forest_vertex_record_owners(placement, state.mesh.vertex_ids)
        if operation == "routes":
            return forest_migration_routes(placement, state)
        raise ValueError(
            f"forest_compile_reuse.operation must be owners or routes; got {operation!r}"
        )

    jax.clear_caches()
    with CompileCounter(capture_stack=True) as cold:
        first = jax.block_until_ready(run(parts, first_state))
    # Preparation replaces empty routes before restart validation.
    routes = forest_migration_routes(parts, first_state)
    routed_parts = prepare_migration_routes(parts, routes)
    assert routed_parts.neighbor_pairs != parts.neighbor_pairs
    with CompileCounter(capture_stack=True) as warm:
        repeated = jax.block_until_ready(run(routed_parts, first_state))
        projected = jax.block_until_ready(run(routed_parts, projected_state))
    with CompileCounter(capture_stack=True) as changed:
        updated = jax.block_until_ready(run(routed_parts, changed_state))
    np.testing.assert_array_equal(first, _expected(first_rows, operation))
    np.testing.assert_array_equal(repeated, first)
    np.testing.assert_array_equal(projected, first)
    np.testing.assert_array_equal(updated, _expected(changed_rows, operation))
    assert first.dtype == updated.dtype == _expected(first_rows, operation).dtype
    if operation == "routes":
        prepared = prepare_migration_routes(routed_parts, updated)
        expected_pairs = tuple(
            (int(source), int(target)) for source, target in np.argwhere(updated)
        )
        assert prepared.neighbor_pairs == expected_pairs
    return {
        "cold_count": cold.count,
        "warm_count": warm.count,
        "changed_count": changed.count,
        "warm_events": [event.to_record() for event in warm.events],
        "changed_events": [event.to_record() for event in changed.events],
        "first": np.asarray(first).tolist(),
        "repeated": np.asarray(repeated).tolist(),
        "changed": np.asarray(updated).tolist(),
    }


def route_refusal(kind: str) -> dict[str, object]:
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=())
    count = parts.part_count
    if kind == "shape":
        mask = np.zeros((count, count + 1), dtype=np.bool_)
    elif kind == "asymmetric":
        mask = np.zeros((count, count), dtype=np.bool_)
        mask[0, 1] = True
    else:
        raise ValueError(
            f"forest_compile_reuse.refusal must be shape or asymmetric; got {kind!r}"
        )
    device_mask = jax.device_put(mask, NamedSharding(parts.mesh, PartitionSpec()))
    with pytest.raises(
        ValueError,
        match="migration.route_mask must be a reciprocal global rank matrix",
    ):
        prepare_migration_routes(parts, device_mask)
    return {"refused": True}


def owner_refusal() -> dict[str, object]:
    parts = AdaptiveSimplexParts(jax.devices(), neighbor_pairs=())
    vertices = jax.device_put(np.asarray((2, 5, 11, -1), dtype=np.int64))
    with pytest.raises(NameError, match="unbound axis"):
        vertex_record_owners(parts, vertices)
    return {"refused": True}
