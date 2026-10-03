# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

from typing import get_args

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la
from phydrax.discretization.meshfree import (
    meshfree_multigrid_builder,
    MeshfreeCoarseningPolicy,
    MeshfreeComponentLayout,
    MeshfreeComponentSpace,
    MeshfreeHierarchyPlan,
    MeshfreeNearNullspace,
    PreparedMeshfreeHierarchy,
)
from phydrax.sparse import EdgeRelation, SparseCoordinateOperator


def _line_hierarchy(count: int = 65) -> PreparedMeshfreeHierarchy:
    points = jnp.linspace(0.0, 1.0, count + 2)[1:-1, None]
    space = la.ArraySpace((count,), dtype=points.dtype)
    policy = MeshfreeCoarseningPolicy(
        minimum_coarse_points=4, coarsening_neighbors=3, interpolation_neighbors=4
    )
    return MeshfreeHierarchyPlan(points, policy=policy).prepare(space)


def _line_operator(
    space: la.ArraySpace, *, neumann: bool = False
) -> SparseCoordinateOperator:
    count = space.size
    indices = np.arange(count, dtype=np.int32)
    edge = EdgeRelation(
        jnp.asarray(np.concatenate((indices, indices[:-1], indices[1:]))),
        jnp.asarray(np.concatenate((indices, indices[1:], indices[:-1]))),
        source_size=count,
        target_size=count,
    )
    diagonal = np.full(count, 2.0)
    if neumann:
        diagonal[[0, -1]] = 1.0
    values = jnp.asarray(
        np.concatenate((diagonal, -np.ones(2 * (count - 1)))), dtype=space.dtype
    )
    return SparseCoordinateOperator(edge, values, source=space, target=space)


def test_each_transfer_reproduces_constants_and_affine_fields() -> None:
    hierarchy = _line_hierarchy()
    for level, (restriction, prolongation) in enumerate(hierarchy.transfers):
        coarse = hierarchy.level_points[level + 1]
        fine = hierarchy.level_points[level]
        np.testing.assert_allclose(
            prolongation.mv(jnp.ones(coarse.shape[0])), 1, atol=1e-10
        )
        np.testing.assert_allclose(
            prolongation.mv(2.5 - 1.7 * coarse[:, 0]), 2.5 - 1.7 * fine[:, 0], atol=1e-10
        )
        v = jnp.cos(fine[:, 0])
        u = jnp.sin(coarse[:, 0])
        np.testing.assert_allclose(
            jnp.vdot(v, prolongation.mv(u)), jnp.vdot(restriction.mv(v), u), atol=1e-10
        )


@pytest.mark.parametrize("dimension", [2, 3])
def test_cross_target_transfer_reproduces_every_affine_coordinate(dimension: int) -> None:
    points = jnp.asarray(np.random.default_rng(82).uniform(-1, 1, (48, dimension)))
    hierarchy = MeshfreeHierarchyPlan(
        points,
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=8, coarsening_neighbors=3),
    ).prepare(la.ArraySpace((48,)))
    direction = jnp.arange(1, dimension + 1, dtype=points.dtype)
    for level, (_, prolongation) in enumerate(hierarchy.transfers):
        fine, coarse = hierarchy.level_points[level : level + 2]
        np.testing.assert_allclose(
            prolongation.mv(1.3 + coarse @ direction), 1.3 + fine @ direction, atol=1e-9
        )


def test_stable_id_coarsening_is_independent_of_input_order() -> None:
    points = jnp.linspace(-1.0, 1.0, 33)[:, None]
    ids = jnp.arange(33, dtype=jnp.int32) * 17 + 11
    policy = MeshfreeCoarseningPolicy(minimum_coarse_points=4, interpolation_neighbors=4)
    first = MeshfreeHierarchyPlan(points, stable_ids=ids, policy=policy).prepare(
        la.ArraySpace((33,))
    )
    permutation = np.random.default_rng(43).permutation(33)
    second = MeshfreeHierarchyPlan(
        points[permutation], stable_ids=ids[permutation], policy=policy
    ).prepare(la.ArraySpace((33,)))
    for left, right in zip(first.level_ids[1:], second.level_ids[1:], strict=True):
        np.testing.assert_array_equal(left, right)


def test_all_boundary_cloud_stops_without_repeated_levels() -> None:
    points = jnp.linspace(-1, 1, 24)[:, None]
    hierarchy = MeshfreeHierarchyPlan(
        points,
        boundary=jnp.ones(24, dtype=jnp.bool_),
        policy=MeshfreeCoarseningPolicy(boundary_retention_levels=None),
    ).prepare(la.ArraySpace((24,)))
    assert hierarchy.evidence.level_sizes == (24,)
    assert hierarchy.evidence.stopping_reason == "no-progress-retention"
    operator = _line_operator(hierarchy.spaces[0])
    action = meshfree_multigrid_builder(hierarchy).prepare(
        operator, materialization=la.MaterializationPolicy()
    )
    exact = jnp.sin(points[:, 0])
    np.testing.assert_allclose(action.apply(operator.mv(exact)), exact, atol=1e-10)


def test_boundary_retained_subset_uses_exact_nodal_inclusion() -> None:
    # On late levels these densely retained square edges have nearest supports
    # lying on one line. Their own coarse nodal values need no ambient fit.
    side = 22
    axis = np.linspace(-1.0, 1.0, side)
    points = np.stack(
        tuple(value.reshape(-1) for value in np.meshgrid(axis, axis, indexing="ij")),
        axis=1,
    )
    boundary = np.any(np.isclose(np.abs(points), 1.0), axis=1)
    points[~boundary] += (
        np.random.default_rng(0).uniform(-0.04, 0.04, (np.count_nonzero(~boundary), 2))
        / side
    )
    boundary[np.flatnonzero(~boundary)[0]] = True
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        boundary=jnp.asarray(boundary),
        policy=MeshfreeCoarseningPolicy(boundary_retention_levels=None),
    ).prepare(la.ArraySpace((side * side,)))
    for level, (_, prolongation) in enumerate(hierarchy.transfers):
        fine_ids = np.asarray(hierarchy.level_ids[level])
        coarse_ids = np.asarray(hierarchy.level_ids[level + 1])
        fine_rows = np.asarray(
            [int(np.flatnonzero(fine_ids == identifier)[0]) for identifier in coarse_ids]
        )
        values = jnp.sin(jnp.asarray(coarse_ids, dtype=jnp.float64))
        np.testing.assert_array_equal(prolongation.mv(values)[fine_rows], values)


def test_transfer_capacity_and_mass_coordinate_mismatch_are_refused() -> None:
    points = jnp.linspace(0, 1, 32)[:, None]
    with pytest.raises(la.LinearCapabilityError, match="maximum_transfer_entries"):
        MeshfreeHierarchyPlan(
            points, policy=MeshfreeCoarseningPolicy(maximum_transfer_entries=2)
        ).prepare(la.ArraySpace((32,)))
    with pytest.raises(ValueError, match="stiffness coordinates"):
        MeshfreeHierarchyPlan(points).prepare(
            la.ArraySpace((32,), pairing=la.DiagonalPairing(jnp.ones(32) / 32))
        )


def test_native_multigrid_solves_the_original_fine_system() -> None:
    hierarchy = _line_hierarchy()
    operator = _line_operator(hierarchy.spaces[0])
    exact = jnp.sin(jnp.pi * hierarchy.level_points[0][:, 0])
    rhs = operator.mv(exact)
    result = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=40),
            preconditioning=la.PreconditioningPolicy(
                meshfree_multigrid_builder(hierarchy)
            ),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=200),
        ),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(operator.mv(result.value), rhs, atol=2e-10)
    np.testing.assert_allclose(result.value, exact, atol=1e-8)


def test_declared_constant_nullspace_coarse_solve_preserves_symmetry_and_gauge() -> None:
    hierarchy = _line_hierarchy(33)
    operator = _line_operator(hierarchy.spaces[0], neumann=True)
    kernel = la.LinearSubspace(hierarchy.spaces[0], jnp.ones((33, 1)))
    policy = la.NullspacePolicy(
        right=kernel, left=kernel, compatibility="project", gauge="project"
    )
    builder = meshfree_multigrid_builder(hierarchy, nullspace_policy=policy)
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    exact = jnp.cos(2 * jnp.pi * hierarchy.level_points[0][:, 0])
    exact -= jnp.mean(exact)
    result = la.solve(
        la.LinearSystem(operator, nullspace_policy=policy),
        operator.mv(exact),
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=32),
            preconditioning=la.PreconditioningPolicy(action),
            tolerance=la.TolerancePolicy(relative=1e-9, absolute=1e-12, max_steps=200),
        ),
    )
    assert bool(result.successful)
    np.testing.assert_allclose(operator.mv(result.value), operator.mv(exact), atol=1e-9)
    np.testing.assert_allclose(jnp.mean(result.value), 0.0, atol=1e-10)
    np.testing.assert_allclose(result.value, exact, atol=1e-7)


def test_projected_coarse_source_refuses_undeclared_kernel_and_incompatible_rhs() -> None:
    space = la.ArraySpace((3,))
    kernel = la.LinearSubspace(space, jnp.ones((3, 1)))
    policy = la.NullspacePolicy(right=kernel, left=kernel, compatibility="error")
    builder = la.ProjectedPseudoinversePreconditionerBuilder(policy)
    materialization = la.MaterializationPolicy()
    with pytest.raises(Exception, match="rank or kernel residual"):
        builder.prepare(
            la.DenseLinearOperator(jnp.zeros((3, 3)), source=space, target=space),
            materialization=materialization,
        )
    action = builder.prepare(
        _line_operator(space, neumann=True), materialization=materialization
    )
    with pytest.raises(Exception, match="incompatible"):
        action.apply(jnp.ones(3))
    rhs = jnp.asarray([-1.0, 0.0, 1.0])
    solved = action.apply(rhs)
    np.testing.assert_allclose(
        _line_operator(space, neumann=True).mv(solved), rhs, atol=1e-10
    )
    np.testing.assert_allclose(jnp.mean(solved), 0, atol=1e-10)
    with pytest.raises(la.LinearCapabilityError, match="budget"):
        builder.prepare(
            _line_operator(space, neumann=True),
            materialization=la.MaterializationPolicy(max_entries=9, max_bytes=1024),
        )


def test_nonsymmetric_collocated_poisson_converges_with_default_multilevel() -> None:
    # This irregular cloud has an unstable Jacobi smoothing spectrum. A fine
    # tridiagonal SPD test alone cannot catch that consumer-visible failure.
    from examples.meshfree_multilevel_poisson import run_workflow

    metrics = run_workflow(size=64, dimension=2, seed=0)
    assert metrics["converged"]
    assert metrics["relative_residual"] < 1e-9
    assert metrics["maximum_error"] < 1e-6
    assert metrics["boundary_residual"] < 1e-8


def _dense_sparse_operator(
    matrix: np.ndarray, space: la.ArraySpace
) -> SparseCoordinateOperator:
    rows, columns = np.nonzero(matrix)
    edge = EdgeRelation(
        jnp.asarray(columns, dtype=jnp.int32),
        jnp.asarray(rows, dtype=jnp.int32),
        source_size=matrix.shape[1],
        target_size=matrix.shape[0],
    )
    return SparseCoordinateOperator(
        edge, jnp.asarray(matrix[rows, columns]), source=space, target=space
    )


def _symmetric_knn_adjacency(points: np.ndarray, width: int) -> np.ndarray:
    # Independent brute-force oracle; ``width`` includes the point itself.
    distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=-1)
    nearest = np.argsort(distances, axis=1, kind="stable")[:, :width]
    adjacency = np.zeros(distances.shape, dtype=np.bool_)
    adjacency[np.arange(points.shape[0])[:, None], nearest] = True
    adjacency |= adjacency.T
    np.fill_diagonal(adjacency, False)
    return adjacency


def _coordinates(field: np.ndarray, layout: MeshfreeComponentLayout) -> np.ndarray:
    return field.reshape(-1) if layout == "nodal" else field.T.reshape(-1)


def test_coarse_level_is_a_maximal_independent_set_with_retained_boundary() -> None:
    points = np.random.default_rng(5).uniform(-1.0, 1.0, (120, 2))
    boundary = np.max(np.abs(points), axis=1) > 0.9
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        boundary=jnp.asarray(boundary),
        policy=MeshfreeCoarseningPolicy(
            maximum_levels=2,
            minimum_coarse_points=8,
            coarsening_neighbors=5,
            boundary_retention_levels=None,
        ),
    ).prepare(la.ArraySpace((120,)))
    adjacency = _symmetric_knn_adjacency(points, 5)
    selected = np.zeros(120, dtype=np.bool_)
    selected[np.asarray(hierarchy.retained_indices[0])] = True
    chosen = selected & ~boundary

    assert np.all(selected[boundary])
    assert not np.any(adjacency[np.ix_(chosen, selected)])
    assert np.all(np.any(adjacency[np.ix_(~selected, selected)], axis=1))
    assert hierarchy.evidence.coarsening_rounds[0] >= 1
    # Default stable IDs follow input order, so coarse coordinates do too.
    np.testing.assert_array_equal(np.asarray(hierarchy.level_points[1]), points[selected])


def test_coarsening_round_limit_stops_without_a_partial_level() -> None:
    points = jnp.asarray(np.random.default_rng(11).uniform(-1.0, 1.0, (200, 2)))
    space = la.ArraySpace((200,))
    limited = MeshfreeHierarchyPlan(
        points, policy=MeshfreeCoarseningPolicy(maximum_coarsening_rounds=1)
    ).prepare(space)
    complete = MeshfreeHierarchyPlan(points).prepare(space)

    assert limited.evidence.stopping_reason == "coarsening-round-limit"
    assert limited.evidence.level_sizes == (200,)
    assert not limited.transfers
    assert complete.evidence.coarsening_rounds[0] > 1
    assert len(complete.evidence.level_sizes) > 1


@pytest.mark.parametrize("dimension", [2, 3])
def test_degree_two_transfer_reproduces_every_quadratic(dimension: int) -> None:
    rng = np.random.default_rng(17 + dimension)
    points = jnp.asarray(rng.uniform(-1.0, 1.0, (160, dimension)))
    hierarchy = MeshfreeHierarchyPlan(
        points,
        policy=MeshfreeCoarseningPolicy(
            minimum_coarse_points=30,
            reproduction_degree=2,
            interpolation_neighbors=14 if dimension == 2 else 24,
        ),
    ).prepare(la.ArraySpace((160,)))
    gradient = jnp.asarray(rng.normal(size=dimension))
    curvature = jnp.asarray(rng.normal(size=(dimension, dimension)))

    def quadratic(x: jax.Array) -> jax.Array:
        return 0.3 + x @ gradient + jnp.sum((x @ curvature) * x, axis=1)

    assert hierarchy.transfers
    assert all(len(level) == 3 for level in hierarchy.evidence.reproduction_residuals)
    for level, (_, prolongation) in enumerate(hierarchy.transfers):
        fine, coarse = hierarchy.level_points[level : level + 2]
        np.testing.assert_allclose(
            prolongation.mv(quadratic(coarse)), quadratic(fine), atol=1e-8
        )


@pytest.mark.parametrize(
    ("dimension", "layout"),
    [
        (dimension, layout)
        for dimension in (2, 3)
        for layout in get_args(MeshfreeComponentLayout)
    ],
)
def test_vector_transfers_reproduce_every_rigid_body_mode(
    dimension: int, layout: MeshfreeComponentLayout
) -> None:
    rng = np.random.default_rng(23 + dimension)
    points = rng.uniform(-1.0, 1.0, (120, dimension))
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        components=MeshfreeComponentSpace(dimension, layout=layout),
        near_nullspace=MeshfreeNearNullspace("rigid-body"),
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=20),
    ).prepare(la.ArraySpace((120 * dimension,)))
    rotation = rng.normal(size=(dimension, dimension))
    rotation -= rotation.T
    translation = rng.normal(size=dimension)

    assert hierarchy.transfers
    for level, (restriction, prolongation) in enumerate(hierarchy.transfers):
        fine = np.asarray(hierarchy.level_points[level])
        coarse = np.asarray(hierarchy.level_points[level + 1])
        assert len(hierarchy.evidence.near_nullspace_defects[level]) == (
            3 if dimension == 2 else 6
        )
        assert max(hierarchy.evidence.near_nullspace_defects[level]) < 1e-9
        np.testing.assert_allclose(
            prolongation.mv(
                jnp.asarray(_coordinates(translation + coarse @ rotation.T, layout))
            ),
            _coordinates(translation + fine @ rotation.T, layout),
            atol=1e-9,
        )
        # Components do not couple: a field in one component stays there.
        single = np.zeros((coarse.shape[0], dimension))
        single[:, -1] = np.sin(3.0 * coarse[:, 0])
        image = np.asarray(prolongation.mv(jnp.asarray(_coordinates(single, layout))))
        image = (
            image.reshape(-1, dimension)
            if layout == "nodal"
            else image.reshape(dimension, -1).T
        )
        np.testing.assert_array_equal(image[:, :-1], 0.0)
        v = jnp.asarray(rng.normal(size=fine.shape[0] * dimension))
        u = jnp.asarray(rng.normal(size=coarse.shape[0] * dimension))
        np.testing.assert_allclose(
            jnp.vdot(v, prolongation.mv(u)), jnp.vdot(restriction.mv(v), u), atol=1e-10
        )


def _spring_network(points: np.ndarray, width: int) -> np.ndarray:
    # Linear truss stiffness: its kernel is exactly the rigid-body motions of a
    # generically rigid kNN network, independent of the transfer construction.
    count, dimension = points.shape
    adjacency = _symmetric_knn_adjacency(points, width)
    stiffness = np.zeros((count * dimension, count * dimension))
    for first, second in zip(*np.nonzero(np.triu(adjacency)), strict=True):
        direction = points[second] - points[first]
        block = np.outer(direction, direction) / np.dot(direction, direction)
        rows = slice(first * dimension, (first + 1) * dimension)
        cols = slice(second * dimension, (second + 1) * dimension)
        stiffness[rows, rows] += block
        stiffness[cols, cols] += block
        stiffness[rows, cols] -= block
        stiffness[cols, rows] -= block
    return stiffness


def test_rigid_body_kernels_give_projected_elasticity_coarse_solve() -> None:
    rng = np.random.default_rng(31)
    points = rng.uniform(-1.0, 1.0, (90, 2))
    stiffness = _spring_network(points, 9)
    centered = points - points.mean(axis=0)
    modes = np.stack(
        (
            _coordinates(np.tile([1.0, 0.0], (90, 1)), "nodal"),
            _coordinates(np.tile([0.0, 1.0], (90, 1)), "nodal"),
            _coordinates(np.stack((-centered[:, 1], centered[:, 0]), axis=1), "nodal"),
        ),
        axis=1,
    )
    assert np.linalg.matrix_rank(stiffness) == 180 - 3
    np.testing.assert_allclose(stiffness @ modes, 0.0, atol=1e-12)
    space = la.ArraySpace((180,))
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        components=MeshfreeComponentSpace(2),
        near_nullspace=MeshfreeNearNullspace("rigid-body"),
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=16),
    ).prepare(space)
    kernel = la.LinearSubspace(space, jnp.asarray(modes))
    nullspace = la.NullspacePolicy(
        right=kernel, left=kernel, compatibility="project", gauge="project"
    )
    operator = _dense_sparse_operator(stiffness, space)
    exact = rng.normal(size=180)
    exact -= modes @ np.linalg.lstsq(modes, exact, rcond=None)[0]
    rhs = jnp.asarray(stiffness @ exact)
    result = la.solve(
        la.LinearSystem(operator, nullspace_policy=nullspace),
        rhs,
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=60),
            preconditioning=la.PreconditioningPolicy(
                meshfree_multigrid_builder(hierarchy, nullspace_policy=nullspace)
            ),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=600),
        ),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(stiffness @ np.asarray(result.value), rhs, atol=1e-8)
    np.testing.assert_allclose(modes.T @ np.asarray(result.value), 0.0, atol=1e-8)
    np.testing.assert_allclose(result.value, exact, atol=1e-6)


@pytest.mark.parametrize(
    "smoother",
    ["point-block-jacobi", "symmetric-gauss-seidel", "ilu"],
)
def test_supplied_native_smoothers_precondition_block_elasticity(smoother: str) -> None:
    rng = np.random.default_rng(37)
    points = rng.uniform(-1.0, 1.0, (90, 2))
    # A mass shift removes the rigid-body kernel, so every smoother applies.
    matrix = _spring_network(points, 9) + 0.05 * np.eye(180)
    space = la.ArraySpace((180,))
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        components=MeshfreeComponentSpace(2),
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=16),
    ).prepare(space)
    sources = {
        "point-block-jacobi": la.BlockJacobiPreconditionerBuilder(2, relaxation=0.6),
        "symmetric-gauss-seidel": la.GaussSeidelPreconditionerBuilder(),
        "ilu": la.ILUPreconditionerBuilder(),
    }
    builder = meshfree_multigrid_builder(
        hierarchy, smoothers=(sources[smoother],) * len(hierarchy.transfers)
    )
    exact = rng.normal(size=180)
    rhs = jnp.asarray(matrix @ exact)
    result = la.solve(
        la.LinearSystem(_dense_sparse_operator(matrix, space)),
        rhs,
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=60),
            preconditioning=la.PreconditioningPolicy(builder),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=400),
        ),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(matrix @ np.asarray(result.value), rhs, atol=1e-8)
    np.testing.assert_allclose(result.value, exact, atol=1e-6)


def test_disconnected_components_transfer_two_declared_kernels() -> None:
    segment = np.linspace(0.0, 1.0, 33)
    points = np.concatenate((segment, segment + 3.0))[:, None]
    neumann = 2.0 * np.eye(33) - np.eye(33, k=1) - np.eye(33, k=-1)
    neumann[0, 0] = neumann[-1, -1] = 1.0
    matrix = np.kron(np.eye(2), neumann)
    indicators = np.kron(np.eye(2), np.ones((33, 1)))
    space = la.ArraySpace((66,))
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        near_nullspace=MeshfreeNearNullspace("constant", vectors=jnp.asarray(indicators)),
        policy=MeshfreeCoarseningPolicy(
            minimum_coarse_points=12, coarsening_neighbors=3, interpolation_neighbors=4
        ),
    ).prepare(space)
    kernel = la.LinearSubspace(space, jnp.asarray(indicators))
    nullspace = la.NullspacePolicy(
        right=kernel, left=kernel, compatibility="project", gauge="project"
    )
    exact = np.cos(np.pi * points[:, 0]) + points[:, 0] ** 2
    exact -= indicators @ (indicators.T @ exact / 33)
    operator = _dense_sparse_operator(matrix, space)
    result = la.solve(
        la.LinearSystem(operator, nullspace_policy=nullspace),
        jnp.asarray(matrix @ exact),
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=40),
            preconditioning=la.PreconditioningPolicy(
                meshfree_multigrid_builder(hierarchy, nullspace_policy=nullspace)
            ),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=300),
        ),
    )

    assert len(hierarchy.transfers) >= 2
    assert all(max(level) < 1e-12 for level in hierarchy.evidence.near_nullspace_defects)
    assert bool(result.successful)
    np.testing.assert_allclose(indicators.T @ np.asarray(result.value), 0.0, atol=1e-9)
    np.testing.assert_allclose(result.value, exact, atol=1e-7)


def test_undeclared_transfer_kernel_is_refused() -> None:
    hierarchy = _line_hierarchy(33)
    space = hierarchy.spaces[0]
    kernel = la.LinearSubspace(
        space, jnp.sin(4.0 * hierarchy.level_points[0][:, :1]) + 2.0
    )
    with pytest.raises(ValueError, match="not reproduced"):
        meshfree_multigrid_builder(
            hierarchy,
            nullspace_policy=la.NullspacePolicy(right=kernel, left=kernel),
        )


@pytest.mark.parametrize("cycle", get_args(la.MultigridCycleKind))
def test_every_native_cycle_solves_the_original_fine_system(
    cycle: la.MultigridCycleKind,
) -> None:
    hierarchy = _line_hierarchy()
    operator = _line_operator(hierarchy.spaces[0])
    builder = meshfree_multigrid_builder(hierarchy, cycle=cycle)
    action = builder.prepare(operator, materialization=la.MaterializationPolicy())
    assert isinstance(action, la.MultigridPreconditioner)
    exact = jnp.sin(jnp.pi * hierarchy.level_points[0][:, 0])
    rhs = operator.mv(exact)
    result = la.solve(
        la.LinearSystem(operator),
        rhs,
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=40),
            preconditioning=la.PreconditioningPolicy(builder),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=200),
        ),
    )
    diagnostics = action.hierarchy.diagnostics

    assert action.cycle_policy.kind == cycle
    assert bool(result.successful)
    np.testing.assert_allclose(operator.mv(result.value), rhs, atol=2e-10)
    assert diagnostics.grid_complexity == pytest.approx(
        hierarchy.evidence.grid_complexity
    )
    assert diagnostics.operator_complexity is not None
    assert diagnostics.operator_complexity > 1.0


def test_numeric_refresh_reuses_transfers_and_new_geometry_rebuilds() -> None:
    hierarchy = _line_hierarchy()
    space = hierarchy.spaces[0]
    operator = _line_operator(space)
    scaled = SparseCoordinateOperator(
        operator.relation,
        3.0 * operator.coefficients,
        source=space,
        target=space,
    )
    materialization = la.MaterializationPolicy()
    builder = meshfree_multigrid_builder(hierarchy)
    action = builder.prepare(operator, materialization=materialization)
    assert isinstance(action, la.MultigridPreconditioner)
    refreshed = builder.refresh(action, scaled, materialization=materialization)
    assert isinstance(refreshed, la.MultigridPreconditioner)
    coarse = jnp.linspace(-1.0, 1.0, action.hierarchy.levels[-1].operator.source.size)
    transitions = len(hierarchy.transfers)

    assert all(
        decision.endswith("transfers-reused;sparse-route-reused;coarse-values-refreshed")
        for decision in refreshed.hierarchy.diagnostics.reuse_decisions[:transitions]
    )
    np.testing.assert_allclose(
        refreshed.hierarchy.levels[-1].operator.mv(coarse),
        3.0 * action.hierarchy.levels[-1].operator.mv(coarse),
        rtol=1e-12,
    )
    fine = hierarchy.level_points[0]
    moved = MeshfreeHierarchyPlan(
        fine + 0.002 * jnp.sin(37.0 * fine), policy=hierarchy.plan.policy
    ).prepare(space)
    rebuilt = meshfree_multigrid_builder(moved).refresh(
        action, scaled, materialization=materialization
    )
    assert isinstance(rebuilt, la.MultigridPreconditioner)
    decisions = rebuilt.hierarchy.diagnostics.reuse_decisions
    assert all(
        "reuse-invalidated-transfer-dependency-change" in decision
        for decision in decisions[: len(moved.transfers)]
    )


def test_coarse_projected_solve_refuses_materialization_budget() -> None:
    hierarchy = _line_hierarchy(33)
    operator = _line_operator(hierarchy.spaces[0], neumann=True)
    kernel = la.LinearSubspace(hierarchy.spaces[0], jnp.ones((33, 1)))
    builder = meshfree_multigrid_builder(
        hierarchy, nullspace_policy=la.NullspacePolicy(right=kernel, left=kernel)
    )
    coarse_size = hierarchy.spaces[-1].size
    with pytest.raises(la.LinearCapabilityError, match="budget"):
        builder.prepare(
            operator,
            materialization=la.MaterializationPolicy(
                max_entries=coarse_size * coarse_size - 1, max_bytes=1_000_000
            ),
        )


def test_multigrid_action_is_independent_of_input_point_order() -> None:
    rng = np.random.default_rng(41)
    points = rng.uniform(-1.0, 1.0, (80, 2))
    ids = np.arange(80, dtype=np.int32) * 7 + 3
    laplacian = _symmetric_knn_adjacency(points, 6).astype(np.float64)
    matrix = np.diag(laplacian.sum(axis=1) + 0.1) - laplacian
    residual = rng.normal(size=80)
    permutation = rng.permutation(80)
    policy = MeshfreeCoarseningPolicy(minimum_coarse_points=10)

    def apply(order: np.ndarray) -> np.ndarray:
        space = la.ArraySpace((80,))
        hierarchy = MeshfreeHierarchyPlan(
            jnp.asarray(points[order]), stable_ids=jnp.asarray(ids[order]), policy=policy
        ).prepare(space)
        operator = _dense_sparse_operator(matrix[np.ix_(order, order)], space)
        action = meshfree_multigrid_builder(hierarchy).prepare(
            operator, materialization=la.MaterializationPolicy()
        )
        return np.asarray(action.apply(jnp.asarray(residual[order])))

    natural = apply(np.arange(80))
    np.testing.assert_allclose(apply(permutation), natural[permutation], atol=1e-10)


def _dirichlet_square(side: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(0.0, 1.0, side)
    points = np.stack(
        tuple(value.reshape(-1) for value in np.meshgrid(axis, axis, indexing="ij")),
        axis=1,
    )
    points += np.random.default_rng(3).uniform(-0.2, 0.2, points.shape) / side
    boundary = np.any((points < 0.5 / side) | (points > 1.0 - 0.5 / side), axis=1)
    return points, boundary


def _eliminated_laplacian(
    points: np.ndarray, eliminated: np.ndarray, identity_scale: float
) -> np.ndarray:
    # Graph Laplacian on free rows; eliminated rows/columns are decoupled
    # identity equations, as after Dirichlet lifting.
    adjacency = _symmetric_knn_adjacency(points, 7).astype(np.float64)
    matrix = np.diag(adjacency.sum(axis=1)) - adjacency
    matrix[eliminated, :] = 0.0
    matrix[:, eliminated] = 0.0
    matrix[eliminated, eliminated] = identity_scale
    return matrix


def test_eliminated_identity_rows_never_reach_coarse_operators() -> None:
    points, boundary = _dirichlet_square(16)
    count = points.shape[0]
    space = la.ArraySpace((count,))
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        eliminated=jnp.asarray(boundary),
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=10),
    ).prepare(space)
    coarse_operators = []
    for scale in (1.0, 7.0):
        operator = _dense_sparse_operator(
            _eliminated_laplacian(points, boundary, scale), space
        )
        action = meshfree_multigrid_builder(hierarchy).prepare(
            operator, materialization=la.MaterializationPolicy()
        )
        assert isinstance(action, la.MultigridPreconditioner)
        level = action.hierarchy.levels[1].operator
        coarse_operators.append(
            np.asarray(
                jax.vmap(level.mv, in_axes=1, out_axes=1)(jnp.eye(level.source.size))
            )
        )
    _, prolongation = hierarchy.transfers[0]
    image = np.asarray(prolongation.mv(jnp.ones(hierarchy.spaces[1].size)))
    retained = np.asarray(hierarchy.retained_indices[0])

    assert len(hierarchy.transfers) >= 2
    assert not np.any(boundary[retained])
    np.testing.assert_array_equal(image[boundary], 0.0)
    np.testing.assert_allclose(image[~boundary], 1.0, atol=1e-10)
    # The decoupled identity scale is invisible to every coarse level.
    np.testing.assert_allclose(coarse_operators[0], coarse_operators[1], atol=1e-12)
    assert np.all(np.diag(coarse_operators[0]) > 0.0)


def test_eliminated_dirichlet_system_solves_with_multigrid() -> None:
    points, boundary = _dirichlet_square(16)
    count = points.shape[0]
    space = la.ArraySpace((count,))
    matrix = _eliminated_laplacian(points, boundary, 1.0)
    hierarchy = MeshfreeHierarchyPlan(
        jnp.asarray(points),
        eliminated=jnp.asarray(boundary),
        policy=MeshfreeCoarseningPolicy(minimum_coarse_points=10),
    ).prepare(space)
    exact = np.sin(3.0 * points[:, 0]) * np.cos(2.0 * points[:, 1])
    rhs = jnp.asarray(matrix @ exact)
    result = la.solve(
        la.LinearSystem(_dense_sparse_operator(matrix, space)),
        rhs,
        policy=la.LinearSolvePolicy(
            la.GMRES(restart=40),
            preconditioning=la.PreconditioningPolicy(
                meshfree_multigrid_builder(hierarchy)
            ),
            tolerance=la.TolerancePolicy(relative=1e-10, absolute=1e-12, max_steps=200),
        ),
    )

    assert bool(result.successful)
    np.testing.assert_allclose(matrix @ np.asarray(result.value), rhs, atol=1e-9)
    np.testing.assert_allclose(result.value, exact, atol=1e-8)


def test_boundary_retention_is_limited_to_declared_finest_levels() -> None:
    points, boundary = _dirichlet_square(18)
    space = la.ArraySpace((points.shape[0],))

    def prepare(levels: int | None) -> PreparedMeshfreeHierarchy:
        return MeshfreeHierarchyPlan(
            jnp.asarray(points),
            boundary=jnp.asarray(boundary),
            policy=MeshfreeCoarseningPolicy(
                minimum_coarse_points=8,
                boundary_retention_levels=levels,
                maximum_coarse_fraction=0.95,
            ),
        ).prepare(space)

    finest = prepare(1)
    every = prepare(None)
    boundary_ids = set(np.flatnonzero(boundary).tolist())
    first = set(np.asarray(finest.level_ids[1]).tolist())
    second = set(np.asarray(finest.level_ids[2]).tolist())

    assert boundary_ids <= first
    assert not boundary_ids <= second
    assert all(boundary_ids <= set(np.asarray(ids).tolist()) for ids in every.level_ids)
    assert finest.evidence.grid_complexity < every.evidence.grid_complexity


def test_insufficient_reduction_stops_before_adding_a_level() -> None:
    points = jnp.asarray(np.random.default_rng(13).uniform(-1.0, 1.0, (100, 2)))
    hierarchy = MeshfreeHierarchyPlan(
        points, policy=MeshfreeCoarseningPolicy(maximum_coarse_fraction=0.05)
    ).prepare(la.ArraySpace((100,)))

    assert hierarchy.evidence.stopping_reason == "insufficient-reduction"
    assert hierarchy.evidence.level_sizes == (100,)
    assert not hierarchy.transfers
