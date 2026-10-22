# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Stable-ID RNG history across actual allocator refine/coarsen events."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import NamedSharding, PartitionSpec

from phydrax.discretization._adaptive_simplex import (
    adaptive_simplex_state,
    AdaptiveSimplexLayout,
    AdaptiveSimplexParts,
    coarsen_adaptive_simplex_parts,
    refine_adaptive_simplex_parts,
)
from tools.meshing_qualification.scenarios.distributed import (
    _distributed_rng_advance,
    _distributed_rng_local,
    _distributed_rng_require_nonreplay,
    _distributed_rng_root_key,
    _distributed_rng_stream,
    _distributed_rng_transport,
    _DistributedRngBank,
)


def test_refined_children_and_resumed_parents_never_replay_stream_addresses() -> None:
    layout = AdaptiveSimplexLayout(
        2,
        2,
        vertex_capacity=16,
        cell_capacity=16,
        protected_edge_capacity=1,
        maximum_closure_iterations=16,
        maximum_coarsening_passes=8,
    )
    root_id = 2**40 + 7
    cells = np.asarray([[0, 1, 2]], dtype=np.int32)
    state = adaptive_simplex_state(
        layout,
        coordinates=np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        vertex_ids=np.arange(3, dtype=np.int64),
        vertex_active=np.ones(3, dtype=np.bool_),
        vertex_parents=np.full((3, 2), -1, dtype=np.int32),
        vertex_protected=np.zeros(3, dtype=np.bool_),
        cells=cells,
        tuples=cells,
        tags=np.asarray([2], dtype=np.int32),
        blocks=np.zeros(1, dtype=np.int32),
        generations=np.zeros(1, dtype=np.int32),
        parents=np.full(1, -1, dtype=np.int32),
        children=np.full((1, 2), -1, dtype=np.int32),
        bisection_vertices=np.full(1, -1, dtype=np.int32),
        cell_ids=np.asarray([root_id], dtype=np.int64),
        cell_active=np.ones(1, dtype=np.bool_),
        cell_classes=np.ones(1, dtype=np.int32),
        facet_classes=np.ones((1, 3), dtype=np.int32),
        protected_edges=np.empty((0, 2), dtype=np.int32),
        next_vertex_id=3,
        next_cell_id=root_id + 1,
    )
    parts = AdaptiveSimplexParts((jax.devices()[0],), axis_name="parts")
    sharding = NamedSharding(parts.mesh, PartitionSpec("parts"))
    raw = jax.tree_util.tree_map(lambda value: jax.device_put(value[None], sharding), state)
    bank = _DistributedRngBank(
        _distributed_rng_root_key(),
        raw,
        jax.device_put(jnp.zeros(raw.mesh.cell_ids.shape, dtype=jnp.uint32), sharding),
    )
    stream = _distributed_rng_stream()

    def draw(current: _DistributedRngBank) -> tuple[np.ndarray, np.ndarray]:
        ids = _distributed_rng_local(current.raw_state.mesh.cell_ids)
        active = _distributed_rng_local(current.raw_state.mesh.cell_active)
        counters = _distributed_rng_local(current.counters)
        keys = stream.entity_keys(current.key_data, ids[active], counters[active])
        return ids[active], np.asarray(jax.vmap(lambda key: jax.random.bits(key, (4,)))(keys))

    parent_ids, first_parent_draw = draw(bank)
    np.testing.assert_array_equal(parent_ids, [root_id])
    bank = _distributed_rng_advance(bank)
    refined = refine_adaptive_simplex_parts(layout, parts, raw, raw.mesh.cell_active)
    np.testing.assert_array_equal(refined.report.status, [0])
    children, lineage = _distributed_rng_transport(bank, raw, refined.state)
    assert lineage["fresh_children"] == 2
    assert _distributed_rng_local(children.counters)[0] == 1
    child_ids, child_draws = draw(children)
    assert child_ids.size == 2
    children = _distributed_rng_advance(children)

    coarsened = coarsen_adaptive_simplex_parts(
        layout, parts, refined.state, refined.state.mesh.cell_active
    )
    np.testing.assert_array_equal(coarsened.report.status, [0])
    resumed, lineage = _distributed_rng_transport(children, refined.state, coarsened.state)
    assert lineage["reactivated_parents"] == lineage["resumed_parents"] == 1
    assert lineage["forged_reset_owners"] == lineage["forged_reset_refusals"] == 1
    resumed_ids, second_parent_draw = draw(resumed)
    np.testing.assert_array_equal(resumed_ids, parent_ids)
    expected = stream.entity_keys(
        bank.key_data, parent_ids, np.ones(1, dtype=np.uint32)
    )
    np.testing.assert_array_equal(second_parent_draw[0], jax.random.bits(expected[0], (4,)))
    assert not np.array_equal(second_parent_draw, first_parent_draw)
    resumed = _distributed_rng_advance(resumed)

    refined_again = refine_adaptive_simplex_parts(
        layout, parts, coarsened.state, coarsened.state.mesh.cell_active
    )
    np.testing.assert_array_equal(refined_again.report.status, [0])
    fresh, lineage = _distributed_rng_transport(resumed, coarsened.state, refined_again.state)
    fresh_ids, fresh_draws = draw(fresh)
    assert lineage["fresh_children"] == 2
    assert np.intersect1d(child_ids, fresh_ids).size == 0
    assert not np.array_equal(child_draws, fresh_draws)
    active = _distributed_rng_local(fresh.raw_state.mesh.cell_active)
    np.testing.assert_array_equal(_distributed_rng_local(fresh.counters)[active], [0, 0])
    np.testing.assert_array_equal(_distributed_rng_local(resumed.counters)[:3], [2, 1, 1])
    with pytest.raises(ValueError, match="must never decrease"):
        _distributed_rng_require_nonreplay(
            np.asarray([2], dtype=np.uint32),
            np.asarray([0], dtype=np.uint32),
            np.asarray([True]),
        )
