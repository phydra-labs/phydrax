"""Algorithm-level contracts for archived logical partition execution."""

import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.graph import GraphPartitionPlan
from phydrax.graph._distributed_partition import replay_logical_partition


@pytest.mark.parametrize(
    ("edge_weights", "work_limit", "expected_status"),
    (
        (((1, 1), (1, 1)), None, 0),
        (((2, 1), (1, 1)), None, 1),
        (((1, 1), (1, 1)), 1, 2),
        (((2, 1), (1, 1)), 1, 3),
    ),
    ids=("accepted", "invalid-mirror", "work-refusal", "combined-refusal"),
)
def test_logical_partition_collects_status_bits_and_preserves_refused_owners(
    edge_weights: tuple[tuple[int, int], tuple[int, int]],
    work_limit: int | None,
    expected_status: int,
) -> None:
    owners = jnp.asarray(((0, 0), (1, 1)), dtype=jnp.int32)
    parts, _, measured = replay_logical_partition(
        jnp.asarray(((0, 1, 2), (0, 1, 2)), dtype=jnp.int64),
        jnp.asarray(((20, 10), (40, 30)), dtype=jnp.int64),
        jnp.asarray(edge_weights, dtype=jnp.int64),
        jnp.ones((2, 2), dtype=jnp.int64),
        jnp.asarray(((10, 20), (30, 40)), dtype=jnp.int64),
        owners,
        jnp.ones((2, 2), dtype=jnp.bool_),
        GraphPartitionPlan(2, refinement_passes=2, work_limit=work_limit),
        axis_name="logical-parts",
    )
    assert measured.resource_status == expected_status
    np.testing.assert_array_equal(parts, owners)
    np.testing.assert_array_equal(measured.part_weights, (2, 2))
    np.testing.assert_array_equal(measured.part_vertex_counts, (2, 2))
    assert measured.edge_cut == 0
    assert measured.migration_vertex_count == 0


def test_logical_partition_publishes_named_literal_work_counters() -> None:
    _, _, measured = replay_logical_partition(
        jnp.asarray(((0, 1, 2), (0, 1, 2)), dtype=jnp.int64),
        jnp.asarray(((20, 10), (40, 30)), dtype=jnp.int64),
        jnp.ones((2, 2), dtype=jnp.int64),
        jnp.ones((2, 2), dtype=jnp.int64),
        jnp.asarray(((10, 20), (30, 40)), dtype=jnp.int64),
        jnp.asarray(((0, 0), (1, 1)), dtype=jnp.int32),
        jnp.ones((2, 2), dtype=jnp.bool_),
        GraphPartitionPlan(2, refinement_passes=2),
        axis_name="logical-parts",
    )
    work = measured.counters
    assert work.coarsening_levels == 1
    assert work.coarsest_vertex_count == 2
    assert work.matched_pairs == 2
    assert work.refinement_passes == 2
    assert work.bisection_trials == 0
    assert work.moves_rolled_back == 0
    assert work.nonempty_repairs == 0
    assert work.moves_committed == 0
    assert work.balance_moves == 0
    # Four directed entries: validation visits 8; two contractions visit 8;
    # two allocations visit 16; two passes over four vertices visit 64.
    assert work.adjacency_visits == 96
    # Validation evaluates 16; contractions 8; allocations 4; refinement 16.
    assert work.candidate_evaluations == 44
