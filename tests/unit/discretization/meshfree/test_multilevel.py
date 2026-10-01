# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.linalg as la
from phydrax.discretization.meshfree import (
    meshfree_multigrid_builder,
    MeshfreeCoarseningPolicy,
    MeshfreeHierarchyPlan,
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
        points, boundary=jnp.ones(24, dtype=jnp.bool_)
    ).prepare(la.ArraySpace((24,)))
    assert hierarchy.evidence.level_sizes == (24,)
    assert hierarchy.evidence.stopping_reason == "no-progress-boundary-retention"
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
        jnp.asarray(points), boundary=jnp.asarray(boundary)
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
