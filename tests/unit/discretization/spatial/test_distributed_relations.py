"""Distributed point relations on forced CPU devices.

Run in a dedicated process with
``XLA_FLAGS=--xla_force_host_platform_device_count=4``. Forced CPU devices
prove functional distributed parity of the collectives only; they are not
accelerator or multi-host performance evidence. Oracles are brute-force NumPy
minimum-image distances ordered by ``(distance, stable_id)``.
"""

from __future__ import annotations

import equinox as eqx
import jax
import numpy as np
import pytest
from jax.sharding import PartitionSpec as P

from phydrax._execution_runtime import ExecutionGroup, ExecutionRuntime
from phydrax.discretization.spatial import (
    DistributedHaloPlan,
    DistributedNeighborQueryPlan,
    DistributedOwnershipPlan,
    DistributedPointLayout,
    DistributedRadiusQueryPlan,
    DistributedRelationStatus,
    MortonAddressPlan,
)


def _owner_group() -> ExecutionGroup:
    devices = jax.devices()
    if len(devices) < 4 or len(devices) % 4:
        pytest.skip(
            "Run with XLA_FLAGS=--xla_force_host_platform_device_count=4 to "
            "exercise four owners."
        )
    return ExecutionRuntime.current().child_groups(len(devices) // 4)[0]


def _minimum_image_distance(
    targets: np.ndarray, sources: np.ndarray, address: MortonAddressPlan
) -> np.ndarray:
    lengths = np.asarray(address.upper) - np.asarray(address.lower)
    periodic = np.asarray(address.periodic_axes)
    relative = targets[:, None, :] - sources[None, :, :]
    wrapped = relative - np.round(relative / lengths) * lengths
    relative = np.where(periodic, wrapped, relative)
    return np.sum(relative * relative, axis=-1)


def _brute_knn(
    points: np.ndarray, ids: np.ndarray, k: int, address: MortonAddressPlan
) -> np.ndarray:
    distance = _minimum_image_distance(points, points, address)
    order = np.lexsort((np.broadcast_to(ids, distance.shape), distance), axis=1)
    return ids[order[:, :k]]


def _quadrant_owners(points: np.ndarray, axes: tuple[int, int]) -> np.ndarray:
    first, second = axes
    return (points[:, first] >= 0.5).astype(np.int32) + 2 * (
        points[:, second] >= 0.5
    ).astype(np.int32)


def _cloud(
    dimension: int,
    count: int,
    seed: int,
    cluster_center: float,
    axes: tuple[int, int] = (0, 1),
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Uneven ownership: one quadrant is four times denser than the others.
    dense = rng.uniform(0.0, 0.5, (count // 2, dimension))
    sparse = rng.uniform(0.0, 1.0, (count - count // 2, dimension))
    offsets = np.zeros((4, dimension))
    offsets[:, axes[0]] = (-1.0, -1.0, 1.0, 1.0)
    offsets[:, axes[1]] = (-1.0, 1.0, -1.0, 1.0)
    # A cluster straddling one owner corner forces shells across four owners.
    cluster = np.full((4, dimension), 0.3)
    cluster[:, list(axes)] = cluster_center
    cluster = np.mod(cluster + 0.004 * offsets, 1.0)
    return np.concatenate((dense, sparse, cluster), axis=0)


_KNN_CASES = (
    pytest.param(2, (False, False), 0.5, (0, 1), id="2d-open"),
    pytest.param(2, (True, True), 0.0, (0, 1), id="2d-periodic-xy"),
    pytest.param(3, (True, False, True), 0.0, (0, 2), id="3d-periodic-xz"),
)


@pytest.mark.parametrize(("dimension", "periodic", "center", "axes"), _KNN_CASES)
def test_uneven_shell_knn_matches_brute_force(
    dimension: int,
    periodic: tuple[bool, ...],
    center: float,
    axes: tuple[int, int],
) -> None:
    group = _owner_group()
    address = MortonAddressPlan(
        (0.0,) * dimension, (1.0,) * dimension, 10, periodic_axes=periodic
    )
    points = _cloud(dimension, 72, 3 + dimension, center, axes)
    owners = _quadrant_owners(points, axes)
    ids = np.arange(points.shape[0], dtype=np.int64) * 7 + 11
    loads = np.bincount(owners, minlength=4)
    assert loads.max() >= 2 * loads.min()
    plan = DistributedOwnershipPlan(address, group, int(loads.max()) + 3)
    layout = DistributedPointLayout.from_global(plan, points, owners, stable_ids=ids)
    k = 6
    result = DistributedNeighborQueryPlan(plan, plan, k).query(layout, layout)

    assert bool(result.evidence.successful)
    neighbors = np.asarray(layout.collect(result.source_stable_ids))
    np.testing.assert_array_equal(neighbors, _brute_knn(points, ids, k, address))
    distance = np.asarray(layout.collect(result.distance_squared))
    expected = np.take_along_axis(
        _minimum_image_distance(points, points, address),
        np.argsort(ids)[np.searchsorted(np.sort(ids), neighbors)],
        axis=1,
    )
    np.testing.assert_allclose(distance, expected, rtol=0, atol=1e-14)
    np.testing.assert_array_equal(
        np.asarray(layout.collect(result.status)), DistributedRelationStatus.COMPLETE
    )
    row_owners = np.asarray(layout.collect(result.source_owners))
    assert max(len(set(row)) for row in row_owners.tolist()) >= 3
    assert int(result.evidence.maximum_required_owners) >= 2
    # Targets are shipped only to shell owners, never replicated to every owner.
    assert int(result.evidence.communicated_targets) < points.shape[0] * 3


def test_remote_owners_answer_within_the_search_radius_at_default_capacity() -> None:
    # Q16 regression: uneven angular sectors meeting at the center. Remote owners
    # used to answer an unbounded k-nearest selection for targets outside their
    # sector (828 candidates against the old 768 default); bounded by the
    # requester's search radius they stay within the derived default capacity.
    from scipy.spatial import cKDTree
    from scipy.stats import qmc

    group = _owner_group()
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10)
    points = qmc.LatinHypercube(d=2, seed=0).random(2048)
    angle = np.arctan2(points[:, 1] - 0.5, points[:, 0] - 0.5)
    widths = np.arange(1, 5, dtype=np.float64)
    edges = -np.pi + 2.0 * np.pi * np.cumsum(widths)[:-1] / widths.sum()
    owners = np.digitize(angle, edges).astype(np.int32)
    ids = np.arange(points.shape[0], dtype=np.int64) * 7 + 11
    plan = DistributedOwnershipPlan(address, group, int(np.bincount(owners).max()) + 64)
    layout = DistributedPointLayout.from_global(plan, points, owners, stable_ids=ids)
    k = 24
    result = DistributedNeighborQueryPlan(plan, plan, k).query(layout, layout)

    assert bool(result.evidence.successful)
    assert int(result.evidence.required_candidates) <= int(
        result.evidence.candidate_capacity
    )
    _, index = cKDTree(points).query(points, k=k)
    found = np.sort(np.asarray(layout.collect(result.source_stable_ids)), axis=1)
    np.testing.assert_array_equal(found, np.sort(ids[index], axis=1))
    row_owners = np.asarray(layout.collect(result.source_owners))
    assert max(len(set(row)) for row in row_owners.tolist()) >= 3


def _periodic_layout(
    group: ExecutionGroup, count: int = 64
) -> tuple[MortonAddressPlan, np.ndarray, np.ndarray, DistributedPointLayout]:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, True))
    points = _cloud(2, count, 17, 0.0)
    owners = _quadrant_owners(points, (0, 1))
    ids = np.arange(points.shape[0], dtype=np.int64)[::-1].copy() + 100
    plan = DistributedOwnershipPlan(address, group, 40)
    return (
        address,
        points,
        ids,
        DistributedPointLayout.from_global(plan, points, owners, stable_ids=ids),
    )


def test_radius_rows_and_pair_once_ownership() -> None:
    group = _owner_group()
    address, points, ids, layout = _periodic_layout(group)
    radius = 0.12
    plan = DistributedRadiusQueryPlan(layout.plan, layout.plan, radius, 40)
    distance = _minimum_image_distance(points, points, address)
    within = distance <= radius**2

    rows = plan.query(layout, layout, exclude_self=True)
    assert bool(rows.evidence.successful)
    neighbors = np.asarray(layout.collect(rows.source_stable_ids))
    valid = np.asarray(layout.collect(rows.valid))
    for target in range(points.shape[0]):
        expected = {
            int(ids[source])
            for source in np.flatnonzero(within[target])
            if source != target
        }
        assert set(neighbors[target][valid[target]].tolist()) == expected

    once = plan.query(layout, layout, pair_once=True)
    assert bool(once.evidence.successful)
    sources = np.asarray(once.source_stable_ids)[np.asarray(once.valid)]
    targets = np.repeat(np.asarray(layout.stable_ids), np.asarray(once.valid).sum(axis=1))
    pairs = list(zip(targets.tolist(), sources.tolist(), strict=True))
    assert len(pairs) == len(set(pairs))
    first, second = np.nonzero(np.triu(within, k=1))
    expected_pairs = {
        (min(int(ids[a]), int(ids[b])), max(int(ids[a]), int(ids[b])))
        for a, b in zip(first, second, strict=True)
    }
    assert set(pairs) == expected_pairs


def test_capacity_and_owner_refusals_are_per_target() -> None:
    group = _owner_group()
    address, points, ids, layout = _periodic_layout(group)
    expected = _brute_knn(points, ids, 6, address)

    halo = DistributedNeighborQueryPlan(
        layout.plan, layout.plan, 6, halo_capacity=1
    ).query(layout, layout)
    status = np.asarray(layout.collect(halo.status))
    refused = status == DistributedRelationStatus.HALO_OVERFLOW
    assert refused.any() and not bool(halo.evidence.successful)
    assert int(halo.evidence.maximum_halo_load) > 1
    complete = status == DistributedRelationStatus.COMPLETE
    assert complete.any()
    neighbors = np.asarray(layout.collect(halo.source_stable_ids))
    np.testing.assert_array_equal(neighbors[complete], expected[complete])
    assert not np.asarray(layout.collect(halo.valid))[refused].any()

    owners = DistributedNeighborQueryPlan(
        layout.plan, layout.plan, 6, maximum_remote_owners=1
    ).query(layout, layout)
    owner_status = np.asarray(layout.collect(owners.status))
    assert (owner_status == DistributedRelationStatus.OWNER_OVERFLOW).any()
    assert int(owners.evidence.maximum_required_owners) > 1

    rows = DistributedRadiusQueryPlan(layout.plan, layout.plan, 0.2, 2).query(
        layout, layout
    )
    row_status = np.asarray(layout.collect(rows.status))
    assert (row_status == DistributedRelationStatus.ROW_OVERFLOW).any()
    assert not np.asarray(layout.collect(rows.valid))[
        row_status == DistributedRelationStatus.ROW_OVERFLOW
    ].any()


def test_stale_owner_epoch_refuses_every_target_as_missing_owner() -> None:
    group = _owner_group()
    _, _, _, layout = _periodic_layout(group)
    stale = eqx.tree_at(
        lambda value: value.owner_epochs,
        layout,
        layout.owner_epochs.at[2].add(-1),
    )
    result = DistributedNeighborQueryPlan(stale.plan, stale.plan, 4).query(stale, stale)
    status = np.asarray(stale.collect(result.status))
    np.testing.assert_array_equal(status, DistributedRelationStatus.MISSING_OWNER)
    assert not bool(result.evidence.owners_current)
    assert not bool(result.evidence.successful)
    assert not np.asarray(result.valid).any()


def test_x64_disabled_runtime_is_an_explicit_refusal() -> None:
    group = _owner_group()
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 8)
    plan = DistributedOwnershipPlan(address, group, 4)
    points = np.asarray([[0.1, 0.1], [0.6, 0.2], [0.2, 0.7], [0.8, 0.9]])
    layout = DistributedPointLayout.from_global(
        plan, points.astype(np.float32), np.asarray([0, 1, 2, 3])
    )
    query = DistributedNeighborQueryPlan(plan, plan, 2)
    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="jax_enable_x64=True"):
            DistributedOwnershipPlan(address, group, 4)
        with pytest.raises(ValueError, match="jax_enable_x64=True"):
            query.query(layout, layout)
    # Float32 coordinates remain supported with x64 enabled.
    result = query.query(layout, layout)
    assert bool(result.evidence.successful)
    assert result.distance_squared.dtype == np.float32


def test_ingress_refuses_duplicate_ids_and_capacity_overflow() -> None:
    group = _owner_group()
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 8)
    plan = DistributedOwnershipPlan(address, group, 2)
    points = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.8, 0.8], [0.9, 0.1]])
    with pytest.raises(ValueError, match="unique"):
        DistributedPointLayout.from_global(
            plan, points, np.asarray([0, 1, 2, 3]), stable_ids=np.asarray([1, 1, 2, 3])
        )
    with pytest.raises(ValueError, match="local_capacity"):
        DistributedPointLayout.from_global(plan, points, np.asarray([0, 0, 0, 1]))
    with pytest.raises(ValueError, match="owner_count"):
        DistributedPointLayout.from_global(plan, points, np.asarray([0, 1, 2, 4]))


def test_lane_reference_owners_run_the_same_collective_relations() -> None:
    # Four owner lanes on one device: queries, halos, and migration execute the
    # identical collective program as named vmap lanes.
    group = ExecutionRuntime.current().child_groups(len(jax.devices()))[0]
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 10, periodic_axes=(True, True))
    points = _cloud(2, 64, 17, 0.0)
    owners = _quadrant_owners(points, (0, 1))
    ids = np.arange(points.shape[0], dtype=np.int64) * 5 + 3
    plan = DistributedOwnershipPlan(address, group, 40, owner_lanes=4)
    assert plan.lane_reference and plan.owner_count == 4
    layout = DistributedPointLayout.from_global(plan, points, owners, stable_ids=ids)
    assert bool(layout.stable_ids_unique)

    knn = DistributedNeighborQueryPlan(plan, plan, 6).query(layout, layout)
    assert bool(knn.evidence.successful)
    np.testing.assert_array_equal(
        np.asarray(layout.collect(knn.source_stable_ids)),
        _brute_knn(points, ids, 6, address),
    )

    rows = DistributedRadiusQueryPlan(plan, plan, 0.15, 40).query(layout, layout)
    halo = DistributedHaloPlan(
        plan,
        rows.source_owners.reshape((-1,)),
        rows.source_slots.reshape((-1,)),
        rows.valid.reshape((-1,)),
        halo_capacity=plan.local_capacity,
    )
    assert bool(halo.evidence.successful)
    rng = np.random.default_rng(2)
    values = rng.normal(size=(plan.total_capacity,))
    cotangent = rng.normal(size=(plan.owner_count * halo.column_count,))
    np.testing.assert_allclose(
        np.dot(np.asarray(halo.gather(values)), cotangent),
        np.dot(values, np.asarray(halo.transpose(cotangent))),
        rtol=1e-13,
    )

    destination = np.where(
        np.asarray(layout.active), (np.asarray(layout.slot_owners) + 1) % 4, 0
    ).astype(np.int32)
    moved = layout.migrate(destination, packet_capacity=40)
    assert bool(moved.evidence.committed)
    np.testing.assert_array_equal(
        np.asarray(moved.layout.collect(moved.layout.points)), points
    )
    if len(jax.devices()) > 1:
        with pytest.raises(ValueError, match="one-device group"):
            DistributedOwnershipPlan(
                address, ExecutionRuntime.current().root_group, 40, owner_lanes=4
            )


def _two_lane_plan(dimension: int = 2) -> DistributedOwnershipPlan:
    group = ExecutionRuntime.current().child_groups(len(jax.devices()))[0]
    address = MortonAddressPlan((0.0,) * dimension, (1.0,) * dimension, 10)
    return DistributedOwnershipPlan(address, group, 2, owner_lanes=2)


def test_lane_replicated_specs_follow_shard_map_semantics() -> None:
    # P(None) is replicated, not owner-blocked: every owner sees the whole
    # value, exactly as under shard_map on devices.
    plan = _two_lane_plan(1)
    axis = plan.axis_name
    values = jax.numpy.asarray([1.0, 2.0, 3.0, 4.0])
    whole = plan.map(lambda x: jax.numpy.sum(x, keepdims=True), P(None), P(axis))
    np.testing.assert_array_equal(np.asarray(whole(values)), [10.0, 10.0])
    total = plan.map(
        lambda x: jax.lax.psum(jax.numpy.sum(x, keepdims=True), axis), P(axis), P(None)
    )
    np.testing.assert_array_equal(np.asarray(total(values)), [10.0])
    with pytest.raises(ValueError, match="leading dimension"):
        plan.map(lambda x: x, P(None, axis), P(axis))(values.reshape((2, 2)))


def test_explicit_logical_rows_must_be_distinct_and_in_range() -> None:
    plan = _two_lane_plan()
    points = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.6, 0.6], [0.7, 0.7]])
    values = jax.numpy.asarray([10.0, 20.0, 30.0, 40.0])
    for logical in ([0, 0, 2, 3], [0, 1, 2, 4], [-1, 1, 2, 3]):
        with pytest.raises(ValueError, match="logical_indices"):
            DistributedPointLayout.from_blocks(
                plan,
                points,
                stable_ids=np.arange(4),
                logical_indices=np.asarray(logical),
                logical_count=4,
            )
    with pytest.raises(ValueError, match="logical_indices"):
        DistributedPointLayout.from_blocks(plan, points, logical_count=3)
    permuted = DistributedPointLayout.from_blocks(
        plan, points, logical_indices=np.asarray([3, 2, 1, 0]), logical_count=4
    )
    np.testing.assert_array_equal(
        np.asarray(permuted.collect(values)), [40.0, 30.0, 20.0, 10.0]
    )
    # Inactive rows do not claim logical rows.
    sparse = DistributedPointLayout.from_blocks(
        plan,
        points,
        active=np.asarray([True, False, True, True]),
        logical_indices=np.asarray([0, 0, 1, 2]),
        logical_count=3,
    )
    np.testing.assert_array_equal(np.asarray(sparse.collect(values)), [10.0, 30.0, 40.0])


def test_stale_owner_cannot_commit_a_self_migration() -> None:
    plan = _two_lane_plan()
    points = np.asarray([[0.1, 0.1], [0.2, 0.2], [0.6, 0.6], [0.7, 0.7]])
    layout = DistributedPointLayout.from_global(plan, points, np.asarray([0, 0, 1, 1]))
    stale = eqx.tree_at(
        lambda value: value.owner_epochs,
        layout,
        jax.numpy.asarray([0, 1], dtype=jax.numpy.int64),
    )
    # Each owner only keeps its own rows, so no packet ever crosses the stale
    # owner boundary; the common-epoch requirement must still refuse.
    result = stale.migrate(np.asarray([0, 0, 1, 1]), packet_capacity=2)
    assert not bool(result.evidence.committed)
    assert not bool(result.evidence.epochs_consistent)
    np.testing.assert_array_equal(np.asarray(result.layout.owner_epochs), [0, 1])
    current = layout.migrate(np.asarray([1, 1, 0, 0]), packet_capacity=2)
    assert bool(current.evidence.committed)
    np.testing.assert_array_equal(np.asarray(current.layout.owner_epochs), [1, 1])


def test_halo_gather_transpose_duality_and_exactly_once_conservation() -> None:
    group = _owner_group()
    _, points, _, layout = _periodic_layout(group)
    rows = DistributedRadiusQueryPlan(layout.plan, layout.plan, 0.15, 40).query(
        layout, layout
    )
    assert bool(rows.evidence.successful)
    halo = DistributedHaloPlan(
        layout.plan,
        rows.source_owners.reshape((-1,)),
        rows.source_slots.reshape((-1,)),
        rows.valid.reshape((-1,)),
        halo_capacity=layout.plan.local_capacity,
    )
    assert bool(halo.evidence.successful)
    owners = layout.plan.owner_count
    local = layout.plan.local_capacity
    columns = halo.column_count
    rng = np.random.default_rng(5)
    values = rng.normal(size=(owners * local,))

    gathered = np.asarray(halo.gather(values)).reshape((owners, columns))
    route_columns = np.asarray(halo.route_columns).reshape((owners, -1))
    route_valid = np.asarray(halo.route_valid).reshape((owners, -1))
    source_rows = (
        np.asarray(rows.source_owners) * local + np.asarray(rows.source_slots)
    ).reshape((owners, -1))
    for owner in range(owners):
        selected = route_valid[owner]
        np.testing.assert_array_equal(
            gathered[owner, route_columns[owner, selected]],
            values[source_rows[owner, selected]],
        )

    cotangent = rng.normal(size=(owners * columns,))
    transposed = np.asarray(halo.transpose(cotangent))
    np.testing.assert_allclose(
        np.dot(gathered.reshape(-1), cotangent),
        np.dot(values, transposed),
        rtol=1e-13,
    )

    # Mark each referenced column once: every source slot must receive one unit
    # per distinct owner that references it, i.e. contributions arrive once.
    marks = np.zeros((owners, columns))
    for owner in range(owners):
        marks[owner, route_columns[owner, route_valid[owner]]] = 1.0
    received = np.asarray(halo.transpose(marks.reshape(-1)))
    expected = np.zeros(owners * local)
    for owner in range(owners):
        for source in set(source_rows[owner, route_valid[owner]].tolist()):
            expected[source] += 1.0
    np.testing.assert_array_equal(received, expected)
    assert received.sum() == marks.sum()


def test_migration_commits_exactly_once_and_rolls_back_on_overflow() -> None:
    group = _owner_group()
    address, points, ids, layout = _periodic_layout(group)
    rng = np.random.default_rng(9)
    owners = layout.plan.owner_count
    destination = np.where(
        np.asarray(layout.active), rng.integers(0, owners, layout.active.shape), 0
    ).astype(np.int32)
    payload = {
        "id": np.asarray(layout.stable_ids, dtype=np.float64),
        "vector": np.stack(
            (np.asarray(layout.stable_ids), -np.asarray(layout.stable_ids)), axis=1
        ).astype(np.float64),
    }
    moved = layout.migrate(destination, packet_capacity=24, payload=payload)
    assert bool(moved.evidence.committed)
    np.testing.assert_array_equal(
        np.asarray(moved.layout.owner_epochs), np.asarray(layout.owner_epochs) + 1
    )
    new_active = np.asarray(moved.layout.active)
    new_ids = np.asarray(moved.layout.stable_ids)[new_active]
    assert sorted(new_ids.tolist()) == sorted(ids.tolist())
    slot_owner = np.repeat(np.arange(owners), layout.plan.local_capacity)[new_active]
    old_owner_of = dict(
        zip(
            np.asarray(layout.stable_ids)[np.asarray(layout.active)].tolist(),
            destination[np.asarray(layout.active)].tolist(),
            strict=True,
        )
    )
    assert [old_owner_of[i] for i in new_ids.tolist()] == slot_owner.tolist()
    np.testing.assert_array_equal(np.asarray(moved.payload["id"])[new_active], new_ids)
    np.testing.assert_array_equal(
        np.asarray(moved.payload["vector"])[new_active][:, 1], -new_ids
    )
    codes = np.asarray(address.encode(moved.layout.points).codes)
    for owner in range(owners):
        mask = new_active & (
            np.arange(new_active.size) // layout.plan.local_capacity == owner
        )
        assert np.all(np.diff(codes[mask].astype(np.float64)) >= 0)
    np.testing.assert_array_equal(
        np.asarray(moved.layout.collect(moved.layout.points)), points
    )
    requery = DistributedNeighborQueryPlan(layout.plan, layout.plan, 5).query(
        moved.layout, moved.layout
    )
    assert bool(requery.evidence.successful)
    np.testing.assert_array_equal(
        np.asarray(moved.layout.collect(requery.source_stable_ids)),
        _brute_knn(points, ids, 5, address),
    )

    rolled = layout.migrate(destination, packet_capacity=1, payload=payload)
    assert not bool(rolled.evidence.committed)
    assert int(rolled.evidence.maximum_packet) > 1
    for name in ("points", "stable_ids", "active", "logical_indices", "owner_epochs"):
        np.testing.assert_array_equal(
            np.asarray(getattr(rolled.layout, name)), np.asarray(getattr(layout, name))
        )
    np.testing.assert_array_equal(np.asarray(rolled.payload["id"]), payload["id"])

    crowded = layout.migrate(
        np.zeros_like(destination), packet_capacity=layout.plan.local_capacity
    )
    assert not bool(crowded.evidence.committed)
    assert int(crowded.evidence.maximum_received) > layout.plan.local_capacity
    np.testing.assert_array_equal(
        np.asarray(crowded.layout.owner_epochs), np.asarray(layout.owner_epochs)
    )
