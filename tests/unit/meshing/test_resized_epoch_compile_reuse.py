# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Epoch padding preserves scientific bytes and reuses capacity compilation."""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexState,
)
from phydrax.meshing._device_adaptation import (
    _addressable_part_array,
    _addressable_part_state,
    _host_epoch,
    _host_part_array,
    _host_part_epoch,
    _resized_epoch,
)
from tests._support.forced_devices import run_on_forced_devices
from tools.compile_counter import CompileCounter


def _layout(vertices: int, cells: int, protected: int) -> AdaptiveSimplexLayout:
    return AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=vertices,
        cell_capacity=cells,
        protected_edge_capacity=protected,
        maximum_closure_iterations=8,
        maximum_coarsening_passes=2,
    )


def _epoch(offset: int) -> AdaptiveSimplexState:
    state = adaptive_simplex_state(
        _layout(4, 2, 4),
        coordinates=np.asarray(((-0.0, 0.0), (1.0, -0.0), (0.0, 1.0))),
        vertex_ids=np.arange(3, dtype=np.int64) + offset,
        vertex_active=np.ones(3, dtype=np.bool_),
        vertex_parents=np.full((3, 2), -1, dtype=np.int32),
        vertex_protected=np.ones(3, dtype=np.bool_),
        cells=np.asarray(((0, 1, 2),), dtype=np.int32),
        tuples=np.asarray(((0, 1, 2),), dtype=np.int32),
        tags=np.zeros(1, dtype=np.int32),
        blocks=np.zeros(1, dtype=np.int32),
        generations=np.zeros(1, dtype=np.int32),
        parents=np.full(1, -1, dtype=np.int32),
        children=np.full((1, 2), -1, dtype=np.int32),
        bisection_vertices=np.full(1, -1, dtype=np.int32),
        cell_ids=np.asarray((offset,), dtype=np.int64),
        cell_active=np.ones(1, dtype=np.bool_),
        cell_classes=np.ones(1, dtype=np.int32),
        facet_classes=np.ones((1, 3), dtype=np.int32),
        protected_edges=np.asarray(((0, 1), (0, 1), (1, 2)), dtype=np.int32),
        next_vertex_id=offset + 3,
        next_cell_id=offset + 1,
    )
    state = eqx.tree_at(
        lambda value: (
            value.clocks,
            value.counters,
            value.refine_rejected,
            value.coarsen_marked,
        ),
        state,
        (
            jax.device_put(np.asarray((3, 5, 7), dtype=np.int32)),
            state.counters + np.int64(offset),
            ~state.refine_rejected,
            ~state.coarsen_marked,
        ),
    )
    return jax.tree.map(lambda value: value[None], state)


@pytest.mark.parametrize("capacities", ((8, 4, 8), (3, 1, 3), (4, 2, 4)))
def test_resize_preserves_bytes_and_same_bucket_compiles_zero(
    capacities: tuple[int, int, int],
) -> None:
    first, second = _epoch(10), _epoch(20)
    layout = _layout(*capacities)
    alternate_layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=capacities[0],
        cell_capacity=capacities[1],
        protected_edge_capacity=capacities[2],
        maximum_closure_iterations=7,
        maximum_coarsening_passes=1,
    )
    jax.clear_caches()
    with CompileCounter() as cold:
        result = jax.block_until_ready(_resized_epoch(first, layout))
    with CompileCounter() as warm:
        jax.block_until_ready(_resized_epoch(second, alternate_layout))
    assert cold.count == 1
    assert warm.count == 0
    vertex_fields = {
        "vertex_half_facets",
        "vertex_parents",
        "vertex_levels",
        "vertex_removal",
        "vertex_protected",
    }
    negative_fields = {
        "vertex_half_facets",
        "vertex_parents",
        "vertex_removal",
        "parents",
        "children",
        "bisection_vertices",
    }
    for name in first.__dataclass_fields__:
        if name == "mesh":
            continue
        source = np.asarray(getattr(first, name))
        if name == "cursors":
            expected = source
        elif name == "clocks":
            expected = source.copy()
            expected[..., 2] = 0
        elif name == "counters":
            expected = source & np.int64(0)
        else:
            count = capacities[0] if name in vertex_fields else capacities[1]
            fill = -1 if name in negative_fields else 0
            if name == "protected_codes":
                count, fill = capacities[2], np.iinfo(np.int64).max
                source = np.where(
                    source != fill, (source // 4) * capacities[0] + source % 4, fill
                )
            if name in {"refine_rejected", "coarsen_marked"}:
                source = source & False
            expected = np.full((1, count, *source.shape[2:]), fill, dtype=source.dtype)
            kept = min(count, source.shape[1])
            expected[:, :kept] = source[:, :kept]
        assert np.asarray(getattr(result, name)).tobytes() == expected.tobytes(), name
    for name in (
        "coordinates",
        "vertex_ids",
        "vertex_active",
        "cells",
        "cell_ids",
        "cell_active",
        "facet_neighbors",
    ):
        source = np.asarray(getattr(first.mesh, name))
        count = (
            capacities[0]
            if name in {"coordinates", "vertex_ids", "vertex_active"}
            else capacities[1]
        )
        fill = -1 if name in {"vertex_ids", "cell_ids", "facet_neighbors"} else 0
        expected = np.full((1, count, *source.shape[2:]), fill, dtype=source.dtype)
        kept = min(count, source.shape[1])
        expected[:, :kept] = source[:, :kept]
        assert np.asarray(getattr(result.mesh, name)).tobytes() == expected.tobytes(), (
            name
        )


def test_addressable_epoch_projection_fuses_leaves_and_keeps_owner_index_dynamic() -> (
    None
):
    states = jax.tree.map(
        lambda left, right: jax.device_put(
            np.concatenate((np.asarray(left), np.asarray(right)), axis=0)
        ),
        _epoch(10),
        _epoch(20),
    )
    reference = jax.tree.map(lambda value: _addressable_part_array(value, 1), states)
    jax.clear_caches()
    with CompileCounter() as cold:
        jax.block_until_ready(_addressable_part_state(states, 0))
    with CompileCounter() as warm:
        result = jax.block_until_ready(_addressable_part_state(states, 1))
    assert cold.count == 1
    assert warm.count == 0
    assert jax.tree.structure(result) == jax.tree.structure(reference)
    for expected, value in zip(
        jax.tree.leaves(reference), jax.tree.leaves(result), strict=True
    ):
        assert np.asarray(value).tobytes() == np.asarray(expected).tobytes()
        assert value.sharding == expected.sharding
    for missing in (-1, 2):
        with pytest.raises(ValueError, match="process-addressable"):
            _addressable_part_state(states, missing)


def test_host_epoch_projection_compiles_nothing_and_owns_lossless_snapshots() -> None:
    states = jax.tree.map(
        lambda left, right: jax.device_put(
            np.concatenate((np.asarray(left), np.asarray(right)), axis=0)
        ),
        _epoch(10),
        _epoch(20),
    )
    references = tuple(
        _host_epoch(jax.tree.map(lambda value: np.asarray(value)[part], states))
        for part in range(2)
    )
    jax.clear_caches()
    with CompileCounter() as compiler:
        snapshots = tuple(_host_part_epoch(states, part) for part in range(2))
        cursors = _host_part_array(states.cursors, 1)
    assert compiler.count == 0
    np.testing.assert_array_equal(cursors, references[1].cursors)
    for reference, snapshot in zip(references, snapshots, strict=True):
        for expected, actual in zip(reference, snapshot, strict=True):
            assert np.asarray(actual).tobytes() == np.asarray(expected).tobytes()
    snapshots[0].coordinates[0, 0] = 42.0
    assert snapshots[1].coordinates.tobytes() == references[1].coordinates.tobytes()
    assert (
        np.asarray(states.mesh.coordinates)[0].tobytes()
        == references[0].coordinates.tobytes()
    )
    for missing in (-1, 2):
        with pytest.raises(ValueError, match="process-addressable"):
            _host_part_epoch(states, missing)


def two_device_resize_placement() -> dict[str, int]:
    placement = NamedSharding(
        Mesh(np.asarray(jax.local_devices(), dtype=object), ("owner",)),
        PartitionSpec("owner"),
    )

    def placed(offset: int) -> AdaptiveSimplexState:
        return jax.tree.map(
            lambda left, right: jax.device_put(
                np.concatenate((np.asarray(left), np.asarray(right)), axis=0),
                placement,
            ),
            _epoch(offset),
            _epoch(offset + 10),
        )

    first, second = placed(10), placed(30)
    layout = _layout(8, 4, 8)
    jax.clear_caches()
    with CompileCounter() as cold:
        resized = jax.block_until_ready(_resized_epoch(first, layout))
    with CompileCounter() as warm:
        jax.block_until_ready(_resized_epoch(second, layout))
    for source, result in zip(
        jax.tree.leaves(first), jax.tree.leaves(resized), strict=True
    ):
        assert result.sharding == source.sharding
    for name in ("refine_rejected", "coarsen_marked", "counters"):
        np.testing.assert_array_equal(np.asarray(getattr(resized, name)), 0)
    np.testing.assert_array_equal(
        np.asarray(resized.mesh.coordinates)[:, :4].view(np.uint64),
        np.asarray(first.mesh.coordinates).view(np.uint64),
    )
    return {"cold_count": cold.count, "warm_count": warm.count}


def test_resize_preserves_every_leaf_placement_on_two_devices_when_warm() -> None:
    result = run_on_forced_devices(
        "tests.unit.meshing.test_resized_epoch_compile_reuse",
        "two_device_resize_placement",
        (),
        devices=2,
        env={},
    )
    assert result["cold_count"] == 1
    assert result["warm_count"] == 0
