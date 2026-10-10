from fractions import Fraction
from itertools import product

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    PeriodicCell,
)
from phydrax.meshing._periodic import (
    _integer_cell_bounds,
    _require_periodic_topology,
    _simplex_entity_corners,
)
from phydrax.meshing.providers._native_periodic import PeriodicAssociationTransfer


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)


@pytest.mark.parametrize("target", (0.25, 0.5))
@pytest.mark.parametrize("margin_fraction,admitted", ((0.75, True), (0.25, False)))
def test_periodic_area_polish_matches_original_publication_premise(
    target: float,
    margin_fraction: float,
    admitted: bool,
) -> None:
    from phydrax.meshing.providers import _native_periodic_size as size

    points = jnp.asarray(
        ((0.0, 0.0), (target, 0.0), (0.0, target * margin_fraction * size._AREA_MARGIN)),
        dtype=np.float64,
    )
    args = size._SizeArgs(
        points,
        jnp.empty(0, dtype=np.int32),
        jnp.empty((0, 2), dtype=np.int32),
        jnp.empty((0, 2), dtype=np.float64),
        jnp.asarray(((0, 1, 2),), dtype=np.int32),
        jnp.zeros((1, 3, 2), dtype=np.float64),
        jnp.asarray(target, dtype=np.float64),
        jnp.empty(0, dtype=np.bool_),
        jnp.asarray(0.0, dtype=np.float64),
    )
    parameters = jnp.empty((0, 2), dtype=np.float64)
    normalized_cross = np.asarray(size._cell_areas(parameters, args))
    assert bool(np.all(normalized_cross > 0.5 * size._AREA_MARGIN)) is admitted
    polish_args = size._PolishArgs(
        args,
        jnp.empty(0, dtype=np.bool_),
        jnp.empty(0, dtype=np.float64),
        jnp.empty(0, dtype=np.int32),
    )
    problem = size._polish_problem(
        np.empty(0, dtype=np.int64),
        np.empty(0, dtype=np.bool_),
        0.0,
    )
    area_constraint = next(
        constraint
        for constraint in problem.constraints
        if constraint.constraint_id == "periodic-positive-cell-areas"
    )
    cross = area_constraint.value(parameters, polish_args)
    lower, upper = area_constraint.bounds(cross)
    assert bool(np.all(np.asarray(cross) >= np.asarray(lower))) is admitted
    assert np.all(np.asarray(cross) <= np.asarray(upper))


def test_exact_integer_embedding_bvh_has_no_false_negative_or_tangent_loss() -> None:
    from phydrax._bvh import bvh_overlap_pair_blocks, BVHBuildPolicy, prepare_bvh

    origin = 1 << 2100
    width = 1 << 2050
    corners = np.asarray(
        (
            ((origin, origin), (origin + width, origin), (origin, origin + width)),
            (
                (origin + width - 1, origin),
                (origin + 2 * width, origin),
                (origin + width - 1, origin + width),
            ),
            (
                (origin + width, origin),
                (origin + 2 * width, origin),
                (origin + width, origin + width),
            ),
            (
                (origin + 4 * width, origin),
                (origin + 5 * width, origin),
                (origin + 4 * width, origin + width),
            ),
        ),
        dtype=object,
    )
    scale_bits = 1700
    lower, upper = _integer_cell_bounds(corners, scale_bits)
    exact_lower, exact_upper = np.min(corners, axis=1), np.max(corners, axis=1)
    for row in range(4):
        for axis in range(2):
            assert Fraction(float(lower[row, axis])) <= Fraction(
                int(exact_lower[row, axis]),
                1 << scale_bits,
            )
            assert Fraction(float(upper[row, axis])) >= Fraction(
                int(exact_upper[row, axis]),
                1 << scale_bits,
            )
    tree = prepare_bvh(
        lower,
        upper,
        policy=BVHBuildPolicy(leaf_size=1),
        dtype=np.float64,
    )
    actual = {
        (int(first), int(second))
        for rows, columns in bvh_overlap_pair_blocks(
            tree,
            tree,
            include_touching=True,
            maximum_block_pairs=2,
        )
        for first, second in zip(rows, columns, strict=True)
    }
    expected = {
        (first, second)
        for first in range(4)
        for second in range(4)
        if np.all(
            np.minimum(exact_upper[first], exact_upper[second])
            >= np.maximum(exact_lower[first], exact_lower[second])
        )
    }
    assert expected <= actual
    assert (0, 1) in actual
    assert (0, 2) in actual
    assert (0, 3) not in actual


@pytest.mark.parametrize("span,admitted", ((1.0, True), (2.0, False)))
def test_affine_embedding_bvh_preserves_nonidentity_self_images_and_scientific_ids(
    span: float,
    admitted: bool,
) -> None:
    from phydrax.discretization import CellMesh, PeriodicMeshTopology
    from phydrax.meshing._periodic import certify_periodic_embedding

    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (span, 0.0), (span, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
        cell_global_ids=np.asarray((41, 73), dtype=np.int64),
    )
    topology = PeriodicMeshTopology(
        mesh,
        PeriodicCell(np.eye(2), periodic_axes=(True, False)),
        np.arange(4, dtype=np.int64),
        np.zeros((4, 2), dtype=np.int64),
    )
    mesh = CellMesh(mesh.coordinates, mesh.blocks, periodic_topology=topology)
    before = np.asarray(mesh.coordinates).copy()
    observed: list[tuple[int, int, int]] = []

    def record(setup: int, tests: int, interiors: int) -> None:
        observed[:] = [(setup, tests, interiors)]

    if admitted:
        evidence = certify_periodic_embedding(mesh, record_work=record)
        assert evidence.image_count == 3
        assert observed[0][1] > evidence.candidate_pair_count
        assert observed[0][2] == evidence.candidate_pair_count
    else:
        with pytest.raises(ValueError, match="scientific IDs"):
            certify_periodic_embedding(mesh, record_work=record)
        assert observed[0][2] > 0
    np.testing.assert_array_equal(mesh.coordinates, before)
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, (41, 73))


def test_affine_embedding_bvh_refuses_original_scratch_before_growth() -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
        NativeExecutionBudget,
        NativeHostStorageWorkspace,
    )
    from phydrax.discretization import CellMesh, PeriodicMeshTopology
    from phydrax.meshing._periodic import certify_periodic_embedding

    mesh = CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )
    topology = PeriodicMeshTopology(
        mesh,
        PeriodicCell(np.eye(2), periodic_axes=(True, False)),
        np.asarray((0, 0, 3, 3)),
        np.asarray(((0, 0), (1, 0), (1, 0), (0, 0))),
    )
    mesh = CellMesh(mesh.coordinates, mesh.blocks, periodic_topology=topology)
    with (
        pytest.raises(MeshcoreError) as ended,
        NativeExecutionBudget(
            max_work=100000,
            max_geometry_queries=100000,
            max_cavity_cells=10,
            max_scratch_bytes=128,
            max_wall_seconds=10.0,
        ) as budget,
    ):
        with NativeHostStorageWorkspace(budget) as workspace:
            with pytest.raises(MeshcoreError) as failed:
                certify_periodic_embedding(mesh)
            assert failed.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert workspace.bound == 0
    assert ended.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    assert budget.evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence.host_storage_live_bytes_upper == 0


def test_periodic_local_lu_congruence_matches_true_saddle_and_source() -> None:
    import equinox as eqx
    import jax

    from phydrax.meshing.providers import _native_periodic_size as size
    from phydrax.optim._nonlinear_constraints import (
        _canonical_constraints,
        _constraint_layout,
    )

    edge_keys = np.asarray(((0, 1, 0, 0), (1, 2, 0, 0), (0, 2, 0, 0)), dtype=np.int64)
    free = np.arange(3, dtype=np.int64)
    kept = np.asarray((True, True, False))
    mutable = np.ones(3, dtype=np.bool_)
    args = size._PolishArgs(
        size._SizeArgs(
            jnp.asarray(((0.0, 0.0), (0.7, 0.1), (0.2, 0.8)), dtype=np.float64),
            jnp.asarray(free, dtype=np.int32),
            jnp.asarray(edge_keys[:, :2], dtype=np.int32),
            jnp.zeros((3, 2), dtype=np.float64),
            jnp.asarray(((0, 1, 2),), dtype=np.int32),
            jnp.zeros((1, 3, 2), dtype=np.float64),
            jnp.asarray(1.0, dtype=np.float64),
            jnp.asarray(mutable),
            jnp.asarray(0.1, dtype=np.float64),
        ),
        jnp.asarray(kept),
        jnp.ones(3, dtype=np.float64),
        jnp.arange(3, dtype=np.int32),
    )
    method, _ = size._prepare_polish_method(
        "authored-periodic-triangle-fixture",
        free,
        edge_keys,
        kept,
        mutable,
        10000000,
        10000000,
        args,
    )
    parameters = jnp.zeros((3, 2), dtype=np.float64)
    problem = size._polish_problem(np.arange(3), kept, 0.1)
    layout = _constraint_layout(problem, parameters, args)

    def constraints(point: jax.Array) -> tuple[jax.Array, jax.Array]:
        return _canonical_constraints(problem, layout, point.reshape((3, 2)), args)

    primal = parameters.reshape(-1)
    equality, inequality = constraints(primal)
    space = phx.linalg.BlockSpace(
        (
            phx.linalg.ArraySpace(primal.shape, dtype=np.float64),
            phx.linalg.ArraySpace(equality.shape, dtype=np.float64),
        )
    )
    setup = method.kkt_setup
    assert setup is not None
    policy = method.linear_policy.preconditioning
    assert policy is not None and policy.builder is not None
    for point, weights in (
        (primal, jnp.ones_like(inequality)),
        (
            primal + jnp.asarray((0.01, -0.02, 0.03, 0.01, -0.01, 0.02)),
            2.0 * jnp.ones_like(inequality),
        ),
    ):
        equality_multipliers = jnp.asarray((0.3, -0.2), dtype=np.float64)
        inequality_multipliers = jnp.linspace(0.1, 0.7, inequality.size, dtype=np.float64)
        derivatives = setup.prepare_derivatives(
            problem,
            layout,
            args,
            point,
            equality_multipliers,
            inequality_multipliers,
        )
        if derivatives is None:
            raise AssertionError("Periodic local KKT setup must prepare its derivatives.")
        current, hessian = derivatives
        local = size._local_constraint_state(current)
        assert local.edge_gradients.shape == (3, 2)
        assert local.edge_hessians.shape == (3, 2, 2)
        assert local.cell_gradients.shape == (1, 3, 2)
        tangent = jnp.asarray((0.2, -0.1, 0.4, 0.3, -0.2, 0.5), dtype=np.float64)
        expected_push = jax.jvp(constraints, (point,), (tangent,))[1]
        actual_push = current.mv(tangent)
        for actual_component, expected_component in zip(
            actual_push, expected_push, strict=True
        ):
            np.testing.assert_allclose(actual_component, expected_component, atol=1.0e-14)
        expected_pull = jax.vjp(constraints, point)[1](
            (equality_multipliers, inequality_multipliers),
        )[0]
        np.testing.assert_allclose(
            current.adjoint_mv((equality_multipliers, inequality_multipliers)),
            expected_pull,
            atol=1.0e-14,
        )

        def lagrangian(candidate: jax.Array) -> jax.Array:
            equal, inequal = constraints(candidate)
            return (
                size._polish_objective(candidate.reshape((3, 2)), args)
                + jnp.vdot(equality_multipliers, equal)
                + jnp.vdot(inequality_multipliers, inequal)
            )

        expected_hessian = jax.jvp(jax.grad(lagrangian), (point,), (tangent,))[1]
        np.testing.assert_allclose(hessian.mv(tangent), expected_hessian, atol=1.0e-14)
        fused = setup.prepare_kkt_operator(
            current,
            hessian,
            weights,
            method.kkt_regularization,
            space,
        )
        if fused is None:
            raise AssertionError("Periodic local KKT setup must prepare its operator.")
        multiplier = jnp.asarray((0.4, -0.3), dtype=np.float64)
        actual_primal, actual_equality = fused.mv((tangent, multiplier))
        expected_curvature = jax.vjp(constraints, point)[1](
            (multiplier, weights * expected_push[1]),
        )[0]
        np.testing.assert_allclose(
            actual_primal,
            expected_hessian + expected_curvature + method.kkt_regularization * tangent,
            atol=1.0e-14,
        )
        np.testing.assert_allclose(actual_equality, expected_push[0], atol=1.0e-14)
        metric = setup.prepare(
            current, weights, method.kkt_regularization, point, equality, space, fused
        )
        # Dense AD is an independent tiny test oracle, never a production path.
        eq_matrix, ineq_matrix = (
            np.asarray(value) for value in jax.jacfwd(constraints)(point)
        )
        primal_matrix = (
            np.asarray(jax.hessian(lagrangian)(point))
            + ineq_matrix.T @ (np.asarray(weights)[:, None] * ineq_matrix)
            + method.kkt_regularization * np.eye(point.size)
        )
        expected_matrix = np.block(
            [
                [primal_matrix, eq_matrix.T],
                [eq_matrix, np.zeros((equality.size, equality.size))],
            ]
        )
        np.testing.assert_allclose(
            space.flatten(metric.operator.mv((tangent, multiplier))),
            expected_matrix @ np.r_[np.asarray(tangent), np.asarray(multiplier)],
            atol=1.0e-14,
        )
        rhs = (jnp.ones(6, dtype=np.float64), jnp.asarray((1.0, -1.0), dtype=np.float64))
        action = policy.builder.prepare(
            metric.operator, materialization=phx.linalg.MaterializationPolicy()
        )
        actual = action.apply(rhs)
        native = phx.linalg.sparse_preconditioner_factorization(action)
        assert native is not None and native.plan.shape == (8, 8)
        assert int(native.status) == int(phx.linalg.SparseFactorizationStatus.SUCCESS)
        lower = np.eye(native.plan.shape[0])
        positions = np.asarray(native.plan.lower_positions)
        row_indices = np.asarray(native.plan.factor_rows)[positions]
        column_indices = np.asarray(native.plan.factor_indices)[positions]
        coefficients = np.asarray(native.factor_values)[positions]
        strict = column_indices < row_indices
        lower[row_indices[strict], column_indices[strict]] = coefficients[strict]
        magnitudes = np.abs(
            np.asarray(native.factor_values)[np.asarray(native.plan.diagonal_positions)]
        )
        permutation = np.asarray(native.plan.permutation)
        expected_ordered = np.linalg.solve(
            lower.T,
            np.linalg.solve(
                lower, np.r_[np.asarray(rhs[0]), np.asarray(rhs[1])][permutation]
            )
            / magnitudes,
        )
        expected_metric_action = np.zeros_like(expected_ordered)
        expected_metric_action[permutation] = expected_ordered
        np.testing.assert_allclose(
            space.flatten(actual), expected_metric_action, rtol=1.0e-12
        )
        assert action.properties.certifies("positive_definite")
        assert int(metric.jvp_evaluations) == 0
        assert int(metric.status) == int(phx.linalg.LinearSolveStatus.SUCCESS)
    changed_args = args._replace(
        size=args.size._replace(origins=args.size.origins.at[0, 0].set(0.001)),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="original scientific binding"):
        setup.prepare_derivatives(
            problem,
            layout,
            changed_args,
            primal,
            equality_multipliers,
            inequality_multipliers,
        )


def test_placement_bank_keeps_scientific_rows_and_exact_predecessor_ties() -> None:
    from phydrax.meshing.providers import _native_periodic_size as size

    # Root 3 acquires a third predecessor when root 2 is placed. Root 4 must
    # then precede it (two predecessors win over three), and root 3 retains
    # the two closest scientific rows with the original edge-index tie break.
    edges = np.asarray(
        (
            (0, 2, 0, 0),
            (1, 2, 0, 0),
            (0, 3, 0, 0),
            (1, 3, 0, 0),
            (2, 3, 0, 0),
            (0, 4, 0, 0),
            (2, 4, 0, 0),
            (4, 3, 1, 0),
        ),
        dtype=np.int64,
    )
    closeness = np.asarray((0.1, 0.2, 0.3, 0.2, 0.05, 0.4, 0.3, 0.05))
    order, kept = size._placement_order(
        np.asarray((4, 3, 2)),
        edges,
        np.ones(8, dtype=np.bool_),
        closeness,
        np.asarray((True, True, False, False, False)),
    )
    assert order == [2, 4, 3]
    np.testing.assert_array_equal(np.flatnonzero(kept), (0, 1, 4, 5, 6, 7))


def _lifted_simplices(mesh: phx.discretization.CellMesh) -> np.ndarray:
    return np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)]


def _circumballs(corners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = corners[:, 1:] - corners[:, :1]
    rhs = 0.5 * np.sum(edges * edges, axis=2)
    offsets = np.linalg.solve(edges, rhs[..., None])[..., 0]
    return corners[:, 0] + offsets, np.linalg.norm(offsets, axis=1)


def _assert_empty_circumballs(
    mesh: phx.discretization.CellMesh, points: np.ndarray, vectors: np.ndarray
) -> None:
    # Independent oracle: every image within two lattice shells of the
    # published lifts lies outside (or on) every circumball.
    dimension = vectors.shape[0]
    shifts = np.asarray(list(product(range(-2, 3), repeat=dimension)), dtype=np.float64)
    images = (points[None, :, :] + (shifts @ vectors)[:, None, :]).reshape(
        (-1, dimension)
    )
    centers, radii = _circumballs(_lifted_simplices(mesh))
    distances = np.linalg.norm(centers[:, None, :] - images[None, :, :], axis=2)
    assert np.all(distances >= radii[:, None] * (1.0 - 1.0e-9))


def _assert_positive_lifts(mesh: phx.discretization.CellMesh) -> None:
    corners = _lifted_simplices(mesh)
    assert np.all(np.linalg.det(corners[:, 1:] - corners[:, :1]) > 0.0)


def _lattice_points(count: int) -> np.ndarray:
    axis = np.arange(count) / count
    return np.stack(np.meshgrid(axis, axis, indexing="ij"), axis=-1).reshape((-1, 2))


@pytest.mark.parametrize(
    ("points", "vectors"),
    [
        (np.random.default_rng(3).random((60, 2)), np.eye(2)),
        (_lattice_points(5), np.eye(2)),
        (
            np.random.default_rng(4).random((50, 2))
            @ np.asarray(((1.0, 0.0), (0.4, 0.9))),
            np.asarray(((1.0, 0.0), (0.4, 0.9))),
        ),
    ],
    ids=["random-unit-torus", "cocircular-lattice", "skew-lattice"],
)
def test_two_torus_points_publish_the_quotient_delaunay_triangulation(
    points: np.ndarray, vectors: np.ndarray
) -> None:
    cell = PeriodicCell(vectors)
    construction = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(points, cell)
    )
    mesh = construction.mesh
    quotient = construction.quotient

    assert quotient.quotient_counts == (
        points.shape[0],
        3 * points.shape[0],
        2 * points.shape[0],
    )
    assert quotient.euler_characteristic == 0
    assert quotient.boundary_facet_count == 0
    assert quotient.all_cells_valid
    np.testing.assert_allclose(
        quotient.total_measure, abs(np.linalg.det(vectors)), rtol=1.0e-12
    )
    assert construction.triangulation.required_margin <= construction.triangulation.margin
    _assert_positive_lifts(mesh)
    _assert_empty_circumballs(mesh, points, vectors)


def test_three_torus_points_publish_the_quotient_delaunay_tetrahedralization() -> None:
    vectors = np.diag([1.0, 1.5, 0.75])
    points = np.random.default_rng(5).random((45, 3)) @ vectors
    construction = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(points, PeriodicCell(vectors))
    )
    quotient = construction.quotient

    assert quotient.quotient_counts[0] == 45
    assert quotient.euler_characteristic == 0
    assert quotient.boundary_facet_count == 0
    # Every quotient face bounds exactly two tetrahedra.
    assert 2 * quotient.quotient_counts[2] == 4 * quotient.quotient_counts[3]
    np.testing.assert_allclose(quotient.total_measure, 1.125, rtol=1.0e-12)
    _assert_positive_lifts(construction.mesh)
    _assert_empty_circumballs(construction.mesh, points, vectors)


def test_single_orbit_torus_keeps_distinct_winding_edges() -> None:
    construction = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(
            np.asarray(((0.3, 0.6),)), PeriodicCell(np.eye(2))
        )
    )
    periodic = construction.mesh.periodic_topology

    assert periodic is not None
    assert construction.quotient.quotient_counts == (1, 3, 2)
    # Three quotient edges join the one representative to itself and differ
    # only by their normalized winding shift.
    keys = periodic.entity_keys(1)
    assert len(set(keys)) == 3
    assert {key[:2] for key in keys} == {(0, 0)}
    reloaded = phx.meshing.certify_cell_mesh(
        construction.mesh, phx.SpatialCoordinateContract.si()
    ).mesh
    assert reloaded.periodic_topology is not None
    assert _require_periodic_topology(reloaded).euler_characteristic == 0


def test_seam_copies_compile_into_one_orbit_and_translations_into_the_lattice() -> None:
    cell = PeriodicCell(np.diag([2.0, 1.0]))
    points = np.asarray(((0.0, 0.25), (2.0, 0.25), (1.0, 0.5), (0.5, 1.75)))
    orbits = phx.meshing.PeriodicPointOrbits(points, cell, tolerance=1.0e-12)

    assert orbits.orbit_count == 3
    np.testing.assert_array_equal(orbits.orbits, (0, 0, 1, 2))
    np.testing.assert_array_equal(orbits.shifts, ((0, 0), (1, 0), (0, 0), (0, 0)))
    with pytest.raises(ValueError, match="lattice orbit"):
        phx.geometry.PeriodicDelaunayTriangulation(points, cell)

    scope = phx.meshing.MeshingScope(
        "geometry",
        "r1",
        phx.meshing.MeshingEntityKind.GEOMETRY,
        1,
        "geometry-1",
        np.asarray((1,), dtype=np.int64),
    )
    translations = []
    for axis, length in enumerate((2.0, 1.0)):
        transform = np.eye(3)
        transform[axis, 2] = length
        translations.append(phx.meshing.PeriodicConstraint(scope, scope, transform))
    lattice = phx.meshing.periodic_cell_from_constraints(translations)
    np.testing.assert_array_equal(np.asarray(lattice.vectors), np.diag([2.0, 1.0]))
    rotation = np.eye(3)
    rotation[:2, :2] = ((0.0, -1.0), (1.0, 0.0))
    with pytest.raises(ValueError, match="not a translation"):
        phx.meshing.periodic_cell_from_constraints(
            (translations[0], phx.meshing.PeriodicConstraint(scope, scope, rotation))
        )


@pytest.mark.parametrize("dimension", [2, 3], ids=["triangles", "tetrahedra"])
def test_orbit_preserving_refinement_keeps_the_quotient_valid(dimension: int) -> None:
    vectors = np.eye(dimension)
    points = np.random.default_rng(6).random((24, dimension))
    construction = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(points, PeriodicCell(vectors))
    )
    counts = construction.quotient.quotient_counts
    uniform = phx.meshing.refine_periodic_mesh(construction.mesh)

    # Every quotient edge gains one midpoint shared by all of its seam copies.
    assert uniform.bisected_edges == counts[1]
    assert uniform.quotient.quotient_counts[0] == counts[0] + counts[1]
    assert uniform.quotient.quotient_counts[-1] == 2**dimension * counts[-1]
    assert uniform.quotient.euler_characteristic == 0
    assert uniform.quotient.boundary_facet_count == 0
    np.testing.assert_allclose(uniform.quotient.total_measure, 1.0, rtol=1.0e-12)
    np.testing.assert_array_equal(
        np.bincount(uniform.parent_cells), np.full(counts[-1], 2**dimension)
    )
    _assert_positive_lifts(uniform.mesh)

    local = phx.meshing.refine_periodic_mesh(construction.mesh, cells=np.asarray((0, 3)))
    assert 1 <= local.bisected_edges <= 2
    assert local.quotient.quotient_counts[0] == counts[0] + local.bisected_edges
    assert local.quotient.euler_characteristic == 0
    assert local.quotient.boundary_facet_count == 0
    np.testing.assert_allclose(local.quotient.total_measure, 1.0, rtol=1.0e-12)


def _periodic_poisson_error(mesh: phx.discretization.CellMesh) -> tuple[float, float]:
    discretization = FiniteElementPlan(
        mesh,
        FiniteElementFieldSpec(
            "u", {mesh.blocks[0].name: lagrange_element("triangle", 1)}
        ),
    ).prepare()
    stiffness = discretization.stiffness.to_scipy()
    mass = discretization.mass.to_scipy()
    nodes = np.asarray(discretization.dof_maps[0].dof_coordinates)
    exact = np.sin(2.0 * np.pi * nodes[:, 0]) * np.sin(2.0 * np.pi * nodes[:, 1])
    load = mass @ (8.0 * np.pi**2 * exact)
    # The periodic operator annihilates constants; a mean-zero multiplier fixes them.
    weights = mass @ np.ones(nodes.shape[0])
    system = sp.bmat(
        [
            [stiffness, sp.csc_matrix(weights[:, None])],
            [sp.csc_matrix(weights[None, :]), None],
        ],
        format="csc",
    )
    solution = spla.spsolve(system, np.append(load, 0.0))[:-1]
    error = solution - exact
    size = float(np.sqrt(2.0 / mesh.blocks[0].cell_count))
    return float(np.sqrt(error @ (mass @ error))), size


def test_periodic_poisson_converges_at_second_order_on_refined_meshes() -> None:
    construction = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(
            np.random.default_rng(8).random((40, 2)), PeriodicCell(np.eye(2))
        )
    )
    mesh = construction.mesh
    errors = []
    for _ in range(3):
        mesh = phx.meshing.refine_periodic_mesh(mesh).mesh
        errors.append(_periodic_poisson_error(mesh))
    rates = [
        np.log(coarse[0] / fine[0]) / np.log(coarse[1] / fine[1])
        for coarse, fine in zip(errors, errors[1:], strict=False)
    ]

    assert errors[-1][0] < 5.0e-3
    assert min(rates) > 1.8


def test_insufficient_image_budget_is_reported_with_evidence() -> None:
    points = np.random.default_rng(9).random((30, 2))
    with pytest.raises(phx.geometry.PeriodicImageBudgetError) as raised:
        phx.meshing.periodic_delaunay_mesh(
            phx.meshing.PeriodicPointOrbits(points, PeriodicCell(np.eye(2))),
            initial_margin=0.05,
            maximum_images=20,
        )
    evidence = raised.value.evidence

    assert evidence.exhausted_limit == "images"
    assert evidence.maximum_images == 20
    assert evidence.image_count > 20
    assert evidence.rounds == 1
    assert evidence.margin == 0.05
    assert evidence.status == "capacity_exceeded"


def _p2_poisson_error(mesh: phx.discretization.CellMesh) -> tuple[float, float]:
    prepared = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 2))
    ).prepare()
    rule = phx.integration.ReferenceTriangleRule(
        phx.integration.GaussLegendreRule(6)
    ).materialize()
    geometry = prepared.evaluate_block_geometry(
        "u", 0, prepared.default_runtime.coordinates, rule.points, rule.weights
    )
    points = np.asarray(geometry.physical_points)
    weights = np.asarray(geometry.physical_weights)
    basis = np.asarray(geometry.basis_values)
    exact = np.sin(2.0 * np.pi * points[..., 0]) * np.sin(2.0 * np.pi * points[..., 1])
    dofs = prepared.dof_maps[0]
    routes = np.asarray(dofs.cell_dofs[0])
    load = np.zeros(dofs.global_dof_count)
    np.add.at(
        load, routes, np.einsum("cq,qi,cq->ci", weights, basis, 8.0 * np.pi**2 * exact)
    )
    stiffness, mass = prepared.stiffness.to_scipy(), prepared.mass.to_scipy()
    averages = mass @ np.ones(dofs.global_dof_count)
    system = sp.bmat(
        [
            [stiffness, sp.csc_matrix(averages[:, None])],
            [sp.csc_matrix(averages[None, :]), None],
        ],
        format="csc",
    )
    solution = spla.spsolve(system, np.append(load, 0.0))[:-1]
    evaluated = np.einsum("qi,ci->cq", basis, solution[routes])
    return float(np.sqrt(np.sum(weights * (evaluated - exact) ** 2))), float(
        np.sqrt(2.0 / mesh.blocks[0].cell_count)
    )


def test_periodic_p2_poisson_has_third_order_l2_convergence() -> None:
    mesh = phx.meshing.periodic_delaunay_mesh(
        phx.meshing.PeriodicPointOrbits(_lattice_points(4), PeriodicCell(np.eye(2)))
    ).mesh
    mesh = phx.meshing.refine_periodic_mesh(mesh).mesh
    errors = [_p2_poisson_error(mesh)]
    for _ in range(2):
        mesh = phx.meshing.refine_periodic_mesh(mesh).mesh
        errors.append(_p2_poisson_error(mesh))
    rates = [
        np.log(coarse[0] / fine[0]) / np.log(coarse[1] / fine[1])
        for coarse, fine in zip(errors, errors[1:], strict=False)
    ]
    assert min(rates) > 2.8
    assert errors[-1][0] < 2.0e-3


def test_native_provider_publishes_periodic_seed_size_and_quotient_evidence() -> None:
    M = phx.meshing
    from phydrax.meshing.providers._native_periodic import NativePeriodicSource

    source = NativePeriodicSource(
        M.PeriodicPointOrbits(_lattice_points(3), PeriodicCell(np.eye(2))), "torus", "r1"
    )
    scope = M.MeshingScope(
        "torus", "r1", M.MeshingEntityKind.GEOMETRY, 2, "regions", np.asarray((0,))
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)
        ),
        size_controls=(
            M.UniformSizeControl(
                scope, 0.25, maximum_size=0.35, strength=M.SizeControlStrength.SOFT
            ),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certification = result.certification
    if certification is None:
        raise AssertionError("Accepted periodic publication requires its certification.")
    assert result.audit.passed and result.compliance.passed and certification.passed
    assert _require_periodic_topology(result.mesh).euler_characteristic == 0
    achieved = dict(result.compliance.achieved)
    np.testing.assert_allclose(achieved["periodic:quotient_measure"], 1.0, atol=1.0e-12)
    assert achieved["periodic:embedding_images"] >= 9
    edge_points = np.asarray(result.mesh.coordinates)[
        _simplex_entity_corners(result.mesh, 1)
    ]
    assert np.max(np.linalg.norm(edge_points[:, 1] - edge_points[:, 0], axis=1)) <= 0.35


def test_skew_periodic_frontal_construction_meets_exact_hard_statistics() -> None:
    M = phx.meshing
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    source = M.NativePeriodicSource(
        M.PeriodicPointOrbits(_lattice_points(4) @ vectors, PeriodicCell(vectors)),
        "skew-torus",
        "r1",
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "regions",
        np.asarray((0,), dtype=np.int64),
    )
    control = M.UniformSizeControl(scope, 0.25, maximum_size=0.375)
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(control,),
        size_compliance=M.SizeCompliancePolicy(
            absolute_tolerance=0.0,
            relative_tolerance=0.0,
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        )
        .execute()
    )
    achieved = dict(result.compliance.achieved)
    prefix = f"size:{control.control_id}"
    assert result.compliance.passed
    assert achieved[f"{prefix}:p50_edge"] == 0.25
    assert achieved[f"{prefix}:p95_edge"] == 0.25
    assert achieved[f"{prefix}:maximum_edge"] <= 0.375
    quotient = M.PeriodicQuotientEvidence(result.mesh)
    assert quotient.euler_characteristic == 0 and quotient.boundary_facet_count == 0
    np.testing.assert_allclose(quotient.total_measure, 1.0, atol=1.0e-12)
    assert len(result.associations) == 3
    for degree, association in enumerate(result.associations):
        association.validate_target(result.mesh.entity_set(degree))
        assert association.complete and association.exact
        assert association.source_id == source.source_id
        assert association.source_revision == source.source_revision
        np.testing.assert_array_equal(
            np.asarray(association.source_indices),
            np.zeros(result.mesh.entity_set(degree).count, dtype=np.int64),
        )
    transfer = PeriodicAssociationTransfer(source, specification)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            result,
            M.MarkedMeshAdaptation(np.asarray(result.mesh.blocks[0].global_ids)),
            policy=M.MeshAdaptationPolicy(
                M.MeshAdaptationRoute.NATIVE_BISECTION,
                limits=specification.limits,
                association_transfer=transfer,
            ),
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    assert len(refined.target.associations) == 3
    transfer.source_associations(refined.target)


def _skew_material_request(
    limits: phx.meshing.MeshingLimits,
) -> tuple[
    phx.meshing.NativePeriodicSource,
    phx.meshing.SurfaceMeshingSpec,
    phx.meshing.UniformSizeControl,
]:
    M = phx.meshing
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    cell = PeriodicCell(vectors)
    count = 4
    lattice = np.stack(
        np.meshgrid(np.arange(count), np.arange(count), indexing="ij"), axis=-1
    ).reshape((-1, 2))
    corners, regions = [], []
    for i, j in product(range(count), range(count)):
        for triangle in (
            ((i, j), (i + 1, j), (i + 1, j + 1)),
            ((i, j), (i + 1, j + 1), (i, j + 1)),
        ):
            corners.append(triangle)
            regions.append(0 if i < count // 2 else 1)
    integer = np.asarray(corners, dtype=np.int64)
    domain = M.publish_periodic_simplices(
        (lattice / count) @ vectors,
        (integer[..., 0] % count) * count + integer[..., 1] % count,
        integer // count,
        cell,
    )
    source = M.NativePeriodicSource(
        domain, "skew-material", "r1", cell_regions=np.asarray(regions)
    )
    edges = _simplex_entity_corners(domain, 1)
    fractional = np.linalg.solve(vectors.T, np.asarray(domain.coordinates).T).T
    x = fractional[edges, 0]
    interface = (x[:, 0] == x[:, 1]) & ((x[:, 0] % 1.0 == 0.0) | (x[:, 0] % 1.0 == 0.5))
    quotient_edges = np.unique(
        np.asarray(_require_periodic_topology(domain).orbits(1)[0])[interface]
    )
    feature = M.ProtectedFeature(
        M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            1,
            "interface-orbits",
            quotient_edges,
        ),
        M.FeatureKind.MATERIAL_INTERFACE,
    )
    scope = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        "regions",
        np.asarray((0, 1), dtype=np.int64),
    )
    control = M.UniformSizeControl(
        scope, 0.25, maximum_size=0.375, strength=M.SizeControlStrength.HARD
    )
    materials = tuple(
        M.RegionControl(
            M.MeshingScope(
                source.source_id,
                source.source_revision,
                M.MeshingEntityKind.GEOMETRY,
                2,
                "regions",
                np.asarray((region,), dtype=np.int64),
            ),
            f"material-{region}",
            f"material-{region}",
            M.RegionRole.USER,
        )
        for region in (0, 1)
    )
    specification = M.SurfaceMeshingSpec(
        M.CellMeshingTarget(2, 2, M.CellFamilyPolicy(required=("triangle",))),
        scope,
        planar_embedding=phx.geometry.PlanarEmbedding(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        size_controls=(control,),
        protected_features=(feature,),
        region_controls=materials,
        limits=limits,
    )
    return source, specification, control


@pytest.mark.parametrize(
    "limits",
    (
        phx.meshing.MeshingLimits(maximum_work_units=1),
        phx.meshing.MeshingLimits(maximum_scratch_bytes=1),
    ),
    ids=("original-work-refusal", "original-scratch-refusal"),
)
def test_skew_material_hard_statistics_respects_original_resource_boundary(
    limits: phx.meshing.MeshingLimits,
) -> None:
    M = phx.meshing
    source, specification, _ = _skew_material_request(limits)
    domain = source.domain
    if not isinstance(domain, phx.discretization.CellMesh):
        raise AssertionError(
            "The material request must retain its represented source mesh."
        )
    before = np.asarray(domain.coordinates).copy()
    binding, mesh_id = source.binding_id, domain.mesh_id
    with pytest.raises(M.MeshingFailure) as refused:
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay")).plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        ).execute()
    assert refused.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert (
        source.domain is domain
        and source.binding_id == binding
        and domain.mesh_id == mesh_id
    )
    np.testing.assert_array_equal(np.asarray(domain.coordinates), before)
