#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import itertools

import equinox as eqx
import numpy as np
import pytest

import phydrax.threshold_dynamics as td
from phydrax.discretization.spatial import MortonAddressPlan
from phydrax.interfacial_transport import InterfaceMobilityMatrix, InterfaceTensionMatrix
from phydrax.threshold_dynamics._sparse import _axis_offsets, _covers_period


@eqx.filter_jit
def _run(
    prepared: td.PreparedThresholdDynamics, state: td.LabelFieldState, steps: int
) -> td.ThresholdDynamicsRunResult:
    return prepared.run(state, steps)


def _voronoi(n: int, count: int, seed: int) -> np.ndarray:
    x = (np.arange(n) + 0.5) / n
    points = np.stack(np.meshgrid(x, x, indexing="ij"), axis=-1)
    seeds = np.random.default_rng(seed).random((count, 2))
    offsets = np.abs(points[:, :, None, :] - seeds[None, None])
    offsets = np.minimum(offsets, 1.0 - offsets)
    return np.argmin(np.sum(offsets**2, axis=-1), axis=-1)


def _grid(n: int, depth: int, radius: int) -> td.SparseLabelGrid:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), depth, periodic_axes=(True, True))
    coordinates = np.asarray(tuple(itertools.product(range(n), repeat=2)))
    return td.SparseLabelGrid(
        address,
        coordinates,
        brick_size=8,
        brick_capacity=(n // 8) ** 2,
        stencil_radius=radius,
    )


def _plan(labels: tuple[str, ...], dt: float) -> td.ThresholdDynamicsPlan:
    return td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        dt,
    )


def test_full_period_sparse_route_equals_the_dense_route() -> None:
    n, count, steps = 32, 7, 6
    labels = tuple(f"grain{index}" for index in range(count))
    plan = _plan(labels, (2.0 / n) ** 2)
    field = _voronoi(n, count, 11)
    grid = _grid(n, 5, n)
    sites = np.asarray(grid.site_coordinates)
    dense = plan.prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
    sparse = plan.prepare(grid, candidate_capacity=count)
    dense_result = _run(dense, dense.initial_state(field), steps)
    sparse_result = _run(
        sparse, sparse.initial_state(field[sites[:, 0], sites[:, 1]]), steps
    )
    dense_final = np.asarray(dense_result.state.labels)[sites[:, 0], sites[:, 1]]
    sparse_evidence = sparse_result.evidence.sparse
    assert sparse_evidence is not None

    assert int(np.sum(np.asarray(dense_result.evidence.changed_sites))) > 0
    np.testing.assert_array_equal(np.asarray(sparse_result.state.labels), dense_final)
    np.testing.assert_allclose(
        sparse_result.evidence.energy_after,
        dense_result.evidence.energy_after,
        rtol=1e-10,
    )
    np.testing.assert_allclose(sparse_evidence.truncated_kernel_mass, 0.0, atol=1e-12)
    assert np.all(np.asarray(sparse_evidence.kernel_symbol_minimum) >= -1e-12)


def test_odd_period_exactness_uses_distinct_residue_coverage() -> None:
    offsets = _axis_offsets(5, 2)
    assert len({offset % 5 for offset in offsets}) == 5
    assert _covers_period(5, 2)
    assert not _covers_period(5, 1)


def test_candidate_overflow_rolls_back() -> None:
    n, count = 32, 12
    labels = tuple(f"grain{index}" for index in range(count))
    plan = _plan(labels, (2.0 / n) ** 2)
    grid = _grid(n, 5, 3)
    sites = np.asarray(grid.site_coordinates)
    field = _voronoi(n, count, 5)[sites[:, 0], sites[:, 1]]
    prepared = plan.prepare(grid, candidate_capacity=2)
    state = prepared.initial_state(field)
    result = _run(prepared, state, 2)
    evidence = result.evidence.sparse
    assert evidence is not None

    assert np.all(
        np.asarray(result.evidence.status)
        == int(td.ThresholdDynamicsStatus.CANDIDATE_OVERFLOW)
    )
    assert np.all(np.asarray(evidence.candidate_overflow))
    assert np.all(np.asarray(evidence.required_candidates) > 2)
    assert np.all(np.asarray(evidence.overflowed_sites) > 0)
    np.testing.assert_array_equal(result.state.labels, state.labels)


def test_current_only_candidate_overflow_remains_visible_after_rollback() -> None:
    n = 16
    grid = _grid(n, 4, 2)
    sites = np.asarray(grid.site_coordinates)
    field = (sites[:, 0] >= n // 2).astype(np.int32)
    prepared = _plan(("left", "right"), (2.0 / n) ** 2).prepare(
        grid, candidate_capacity=1
    )
    state = prepared.initial_state(field)
    result = prepared.step(state)
    evidence = result.evidence.sparse

    assert evidence is not None
    assert int(result.status) == int(td.ThresholdDynamicsStatus.CANDIDATE_OVERFLOW)
    assert bool(evidence.candidate_overflow)
    assert int(evidence.required_candidates) == 2
    assert int(evidence.overflowed_sites) > 0
    np.testing.assert_array_equal(result.state.labels, state.labels)


def test_sparse_storage_is_independent_of_the_declared_label_count() -> None:
    n = 64
    labels = tuple(f"grain{index}" for index in range(50_000))
    policy = td.ThresholdDynamicsResourcePolicy(maximum_working_bytes=64 * 1024**2)
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        (1.5 / n) ** 2,
        resource_policy=policy,
    )
    with pytest.raises(MemoryError):
        plan.prepare(td.PeriodicGridHeatKernel((n, n), (1.0, 1.0)))
    grid = _grid(n, 6, 7)
    sites = np.asarray(grid.site_coordinates)
    field = _voronoi(n, 9, 2)[sites[:, 0], sites[:, 1]] * 5_000
    prepared = plan.prepare(grid, candidate_capacity=8)
    result = _run(prepared, prepared.initial_state(field), 3)
    evidence = result.evidence.sparse
    assert evidence is not None

    assert np.all(np.asarray(result.evidence.committed))
    np.testing.assert_array_equal(evidence.active_label_count, [9, 9, 9])
    assert set(np.unique(np.asarray(result.state.labels))) <= set(range(0, 45_000, 5_000))
    assert float(np.max(np.abs(np.asarray(evidence.truncated_kernel_mass)))) < 1e-2
    assert int(np.max(np.asarray(evidence.required_candidates))) <= 8


def test_sparse_volume_constraint_keeps_exact_counts() -> None:
    n, count = 32, 5
    labels = tuple(f"cell{index}" for index in range(count))
    grid = _grid(n, 5, 4)
    sites = np.asarray(grid.site_coordinates)
    field = _voronoi(n, count, 8)[sites[:, 0], sites[:, 1]]
    counts = np.bincount(field, minlength=count)
    plan = td.ThresholdDynamicsPlan(
        InterfaceTensionMatrix(labels, 1.0, structure="uniform"),
        InterfaceMobilityMatrix(labels, 1.0, structure="uniform"),
        (2.0 / n) ** 2,
        volume_constraint=td.LabelVolumeConstraint(counts),
    )
    prepared = plan.prepare(grid, candidate_capacity=count)
    result = _run(prepared, prepared.initial_state(field), 4)

    assert np.all(np.asarray(result.evidence.committed))
    np.testing.assert_array_equal(
        result.evidence.label_counts, np.broadcast_to(counts, (4, count))
    )


def test_sparse_route_requires_periodic_box_and_capacity() -> None:
    address = MortonAddressPlan((0.0, 0.0), (1.0, 1.0), 4)
    with pytest.raises(ValueError, match="periodic"):
        td.SparseLabelGrid(
            address,
            np.zeros((1, 2), dtype=np.int64),
            brick_size=4,
            brick_capacity=1,
            stencil_radius=1,
        )
    grid = _grid(16, 4, 2)
    plan = _plan(("a", "b"), 1e-2)
    with pytest.raises(ValueError, match="candidate_capacity"):
        plan.prepare(grid)
    with pytest.raises(ValueError, match="exceed"):
        plan.prepare(grid, candidate_capacity=3)
