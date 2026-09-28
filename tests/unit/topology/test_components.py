#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import numpy as np
import pytest
from scipy import ndimage

import phydrax as phx


Status = phx.topology.ComponentLabelingStatus
Event = phx.topology.ComponentEventKind


def _reference_roots(
    mask: np.ndarray, periodic: tuple[bool, ...], color: np.ndarray | None = None
) -> np.ndarray:
    """Union-find over face neighbors; each root is its set's minimum flat index."""
    shape = mask.shape
    parent = np.arange(mask.size)
    index = np.arange(mask.size).reshape(shape)
    flat_mask = mask.ravel()
    flat_color = None if color is None else color.ravel()

    def find(entity: int) -> int:
        while parent[entity] != entity:
            parent[entity] = parent[parent[entity]]
            entity = parent[entity]
        return entity

    for axis in range(mask.ndim):
        for cell in np.ndindex(shape):
            neighbour = list(cell)
            neighbour[axis] += 1
            if neighbour[axis] == shape[axis]:
                if not periodic[axis]:
                    continue
                neighbour[axis] = 0
            u = index[cell]
            v = index[tuple(neighbour)]
            if u == v or not (flat_mask[u] and flat_mask[v]):
                continue
            if flat_color is not None and flat_color[u] != flat_color[v]:
                continue
            first, second = find(u), find(v)
            parent[max(first, second)] = min(first, second)
    return np.asarray(
        [find(entity) if flat_mask[entity] else -1 for entity in range(mask.size)]
    )


def _scipy_roots(mask: np.ndarray) -> np.ndarray:
    labels, _ = ndimage.label(mask)
    flat = labels.ravel()
    roots = np.full(mask.size, -1)
    for component in np.unique(flat[flat > 0]):
        members = np.flatnonzero(flat == component)
        roots[members] = members.min()
    return roots


def _assert_matches_roots(
    result: phx.topology.ComponentLabelingResult, reference: np.ndarray
) -> None:
    root = np.asarray(result.root)
    label = np.asarray(result.label)
    np.testing.assert_array_equal(root, reference)
    component_roots = np.unique(reference[reference >= 0])
    assert int(result.count) == component_roots.size
    expected_label = np.where(
        reference >= 0, np.searchsorted(component_roots, reference), -1
    )
    np.testing.assert_array_equal(label, expected_label)
    capacity = result.component_root.shape[0]
    expected_slots = np.full(capacity, -1)
    expected_slots[: component_roots.size] = component_roots
    np.testing.assert_array_equal(np.asarray(result.component_root), expected_slots)
    expected_sizes = np.zeros(capacity, dtype=np.int64)
    np.add.at(expected_sizes, label[label >= 0], 1)
    np.testing.assert_array_equal(np.asarray(result.component_size), expected_sizes)
    assert bool(result.successful)
    assert int(result.status) == Status.CONVERGED


@pytest.mark.parametrize(
    ("shape", "periodic"),
    [
        ((12, 10), (False, False)),
        ((12, 10), (True, False)),
        ((12, 10), (True, True)),
        ((6, 5, 4), (False, False, False)),
        ((6, 5, 4), (True, False, True)),
    ],
)
def test_random_masks_match_host_reference(
    shape: tuple[int, ...], periodic: tuple[bool, ...]
) -> None:
    relation = phx.topology.grid_adjacency_relation(shape, periodic)
    size = int(np.prod(shape))
    plan = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=size, maximum_rounds=size
    )
    for seed in range(3):
        mask = np.random.default_rng(seed).random(shape) < 0.55
        reference = _reference_roots(mask, periodic)
        if not any(periodic):
            np.testing.assert_array_equal(reference, _scipy_roots(mask))
        _assert_matches_roots(plan.label(mask.ravel()), reference)


def test_edge_mask_restricts_connectivity_to_equal_colors() -> None:
    shape = (12, 10)
    relation = phx.topology.grid_adjacency_relation(shape, (False, False))
    plan = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=120, maximum_rounds=120
    )
    color = np.random.default_rng(7).integers(0, 2, size=shape)
    flat_color = color.ravel()
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    result = plan.label(
        np.ones(120, dtype=bool), edge_valid=flat_color[source] == flat_color[target]
    )
    reference = _reference_roots(np.ones(shape, dtype=bool), (False, False), color)
    per_color = np.maximum(_scipy_roots(color == 0), _scipy_roots(color == 1))
    np.testing.assert_array_equal(reference, per_color)
    _assert_matches_roots(result, reference)


def _serpentine_mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=bool)
    mask[::2, :] = True
    mask[1::4, -1] = True
    mask[3::4, 0] = True
    return mask


def test_serpentine_path_converges_only_with_enough_rounds() -> None:
    relation = phx.topology.grid_adjacency_relation((16, 16), (False, False))
    mask = _serpentine_mask().ravel()
    converged = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=4, maximum_rounds=256
    ).label(mask)
    _assert_matches_roots(converged, _reference_roots(_serpentine_mask(), (False, False)))
    assert int(converged.count) == 1

    truncated = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=4, maximum_rounds=1
    ).label(mask)
    assert int(truncated.rounds) == 1
    assert not bool(truncated.converged)
    assert not bool(truncated.successful)
    assert int(truncated.status) == Status.NOT_CONVERGED


def test_capacity_overflow_refuses_slots_beyond_capacity() -> None:
    relation = phx.topology.grid_adjacency_relation((6, 6), (False, False))
    mask = np.zeros((6, 6), dtype=bool)
    mask[::2, ::2] = True
    result = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=4, maximum_rounds=36
    ).label(mask.ravel())
    label = np.asarray(result.label)
    assert int(result.count) == 9
    assert bool(result.overflow)
    assert not bool(result.successful)
    assert int(result.status) == Status.CAPACITY_EXCEEDED
    assert label.max() < 4
    assert np.count_nonzero(label >= 0) == 4
    np.testing.assert_array_equal(np.asarray(result.component_root), [0, 2, 4, 12])
    assert np.asarray(result.component_mask).all()


def test_labeling_is_jittable() -> None:
    shape = (12, 10)
    relation = phx.topology.grid_adjacency_relation(shape, (True, True))
    plan = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=120, maximum_rounds=120
    )
    mask = np.random.default_rng(11).random(shape) < 0.5

    @jax.jit
    def label(
        compiled_plan: phx.topology.ConnectedComponentPlan, active: jax.Array
    ) -> phx.topology.ComponentLabelingResult:
        return compiled_plan.label(active)

    result = label(plan, jax.numpy.asarray(mask.ravel()))
    _assert_matches_roots(result, _reference_roots(mask, (True, True)))


def test_grid_relation_has_no_duplicate_or_self_edges_on_short_periodic_axes() -> None:
    relation = phx.topology.grid_adjacency_relation((1, 2, 3), (True, True, True))
    source = np.asarray(relation.source_indices)
    target = np.asarray(relation.target_indices)
    assert np.all(source < target)
    pairs = set(zip(source.tolist(), target.tolist()))
    assert len(pairs) == source.size
    # Axis 0 (length 1) adds nothing, axis 1 (length 2) one pair per column of
    # the three axis-2 positions, axis 2 (periodic length 3) a 3-cycle per row.
    assert pairs == {
        (0, 3),
        (1, 4),
        (2, 5),
        (0, 1),
        (1, 2),
        (0, 2),
        (3, 4),
        (4, 5),
        (3, 5),
    }


def test_plan_refuses_non_square_relation_and_empty_capacity() -> None:
    relation = phx.sparse.EdgeRelation(
        np.asarray([0]), np.asarray([1]), source_size=2, target_size=3
    )
    with pytest.raises(ValueError, match="onto itself"):
        phx.topology.ConnectedComponentPlan(
            relation, component_capacity=2, maximum_rounds=2
        )
    square = phx.topology.grid_adjacency_relation((2,), (False,))
    with pytest.raises(ValueError, match="component_capacity"):
        phx.topology.ConnectedComponentPlan(
            square, component_capacity=0, maximum_rounds=2
        )


_TRANSITION_SHAPE = (3, 8)


def _mask(cells: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(_TRANSITION_SHAPE, dtype=bool)
    for cell in cells:
        mask[cell] = True
    return mask.ravel()


def _row(row: int, start: int, stop: int) -> list[tuple[int, int]]:
    return [(row, column) for column in range(start, stop)]


def _transition(
    old_cells: list[tuple[int, int]],
    new_cells: list[tuple[int, int]],
    weight: np.ndarray,
    *,
    pair_capacity: int = 6,
) -> phx.topology.ComponentTransitionResult:
    relation = phx.topology.grid_adjacency_relation(_TRANSITION_SHAPE, (False, False))
    labeling = phx.topology.ConnectedComponentPlan(
        relation, component_capacity=4, maximum_rounds=24
    )
    plan = phx.topology.ComponentTransitionPlan(
        24, old_capacity=4, new_capacity=4, pair_capacity=pair_capacity
    )
    return plan.transition(
        labeling.label(_mask(old_cells)), labeling.label(_mask(new_cells)), weight
    )


_WEIGHT = np.random.default_rng(3).random(24)


def _overlap(first: list[tuple[int, int]], second: list[tuple[int, int]]) -> float:
    shared = _mask(first) & _mask(second)
    return float(_WEIGHT[shared].sum())


def test_transition_classifies_merge_with_summed_overlaps() -> None:
    left, right, bridge = _row(1, 0, 3), _row(1, 5, 8), _row(1, 0, 8)
    result = _transition(left + right, bridge, _WEIGHT)
    np.testing.assert_array_equal(np.asarray(result.pair_old)[:3], [0, 1, -1])
    np.testing.assert_array_equal(np.asarray(result.pair_new)[:3], [0, 0, -1])
    np.testing.assert_allclose(
        np.asarray(result.pair_overlap)[:2],
        [_overlap(left, bridge), _overlap(right, bridge)],
    )
    np.testing.assert_array_equal(np.asarray(result.new_parent_count), [2, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(result.old_child_count), [1, 1, 0, 0])
    np.testing.assert_array_equal(
        np.asarray(result.new_event), [Event.MERGE, Event.NONE, Event.NONE, Event.NONE]
    )
    np.testing.assert_array_equal(
        np.asarray(result.old_event), [Event.MERGE, Event.MERGE, Event.NONE, Event.NONE]
    )
    assert int(result.merge_count) == 1
    assert int(result.split_count) == 0
    assert bool(result.successful)


def test_transition_classifies_split() -> None:
    left, right, bridge = _row(1, 0, 3), _row(1, 5, 8), _row(1, 0, 8)
    result = _transition(bridge, left + right, _WEIGHT)
    np.testing.assert_array_equal(np.asarray(result.old_child_count), [2, 0, 0, 0])
    np.testing.assert_array_equal(np.asarray(result.new_parent_count), [1, 1, 0, 0])
    np.testing.assert_array_equal(
        np.asarray(result.old_event), [Event.SPLIT, Event.NONE, Event.NONE, Event.NONE]
    )
    np.testing.assert_array_equal(
        np.asarray(result.new_event), [Event.SPLIT, Event.SPLIT, Event.NONE, Event.NONE]
    )
    np.testing.assert_allclose(
        np.asarray(result.pair_overlap)[:2],
        [_overlap(bridge, left), _overlap(bridge, right)],
    )
    assert int(result.split_count) == 1
    assert int(result.merge_count) == 0


def test_transition_classifies_continue_create_and_vanish() -> None:
    kept, vanished = _row(0, 0, 3), _row(2, 0, 3)
    grown, created = _row(0, 0, 4), _row(2, 5, 8)
    result = _transition(kept + vanished, grown + created, _WEIGHT)
    np.testing.assert_array_equal(
        np.asarray(result.old_event),
        [Event.CONTINUE, Event.VANISH, Event.NONE, Event.NONE],
    )
    np.testing.assert_array_equal(
        np.asarray(result.new_event),
        [Event.CONTINUE, Event.CREATE, Event.NONE, Event.NONE],
    )
    np.testing.assert_array_equal(np.asarray(result.pair_active), [1, 0, 0, 0, 0, 0])
    np.testing.assert_allclose(float(result.pair_overlap[0]), _overlap(kept, grown))
    assert int(result.create_count) == 1
    assert int(result.vanish_count) == 1


def test_transition_classifies_reconnection() -> None:
    upper, lower = _row(0, 0, 4), _row(2, 0, 4)
    detached = _row(0, 0, 2)
    column = [(0, 3), (1, 3), (2, 3)]
    result = _transition(upper + lower, detached + column, _WEIGHT)
    np.testing.assert_array_equal(
        np.asarray(result.old_event),
        [Event.RECONNECT, Event.MERGE, Event.NONE, Event.NONE],
    )
    np.testing.assert_array_equal(
        np.asarray(result.new_event),
        [Event.SPLIT, Event.RECONNECT, Event.NONE, Event.NONE],
    )
    assert int(result.reconnect_count) == 1


def test_transition_reports_pair_capacity_overflow() -> None:
    left, right, bridge = _row(1, 0, 3), _row(1, 5, 8), _row(1, 0, 8)
    result = _transition(left + right, bridge, _WEIGHT, pair_capacity=1)
    assert bool(result.pair_overflow)
    assert not bool(result.successful)
