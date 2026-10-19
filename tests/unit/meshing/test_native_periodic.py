from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from fractions import Fraction
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import phydrax as phx
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreStatus,
    NativeExecutionScope,
    NativeParallelExecution,
    NativeResourceFailure,
    periodic_refine_2d,
    PeriodicRefinement,
)
from phydrax.discretization import (
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
    PeriodicCell,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax.meshing import PeriodicAssociationTransfer
from phydrax.meshing._periodic import (
    integer_cell_bounds,
    require_periodic_topology,
    simplex_entity_corners,
)
from phydrax.meshing.providers._native_periodic import periodic_source_associations
from tests._support.periodic_meshing import (
    periodic_lattice_points,
    periodic_seed_request,
    skew_periodic_material_request,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)


# Fixed empirical allowances from test_native_periodic.py
# at a6827a9e5; no meshing acceptance policy is relaxed.
# Quotient cell area/volume sums against lattice determinants, including volume 1.125.
_QUOTIENT_MEASURE_RTOL = 1e-12
# Published periodic quotient measure against the unit fundamental-domain volume.
_UNIT_QUOTIENT_ATOL = 1e-12
# The transfer owner certifies affine actions at 64 coefficient-dtype ulps,
# scaled by the prepared action's condition estimate and coefficient magnitude.
_FIELD_TRANSFER_ROUNDOFF_ULPS = 64.0


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
    lower, upper = integer_cell_bounds(corners, scale_bits)
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
    active = tuple(np.asarray(block.vertex_valid).copy() for block in mesh.blocks)
    observed: list[tuple[int, int, int]] = []

    def record(setup: int, tests: int, interiors: int) -> None:
        observed[:] = [(setup, tests, interiors)]

    if admitted:
        evidence = certify_periodic_embedding(mesh, record_work=record)
        assert evidence.image_count == 3
        assert observed[0][1] > evidence.candidate_pair_count
        assert observed[0][2] == evidence.candidate_pair_count
        assert evidence.coordinate_scope == "exact_dyadic_group_lift"
    else:
        with pytest.raises(ValueError, match="scientific IDs"):
            certify_periodic_embedding(mesh, record_work=record)
        assert observed[0][2] > 0
    np.testing.assert_array_equal(mesh.coordinates, before)
    np.testing.assert_array_equal(mesh.blocks[0].global_ids, (41, 73))
    for block, mask in zip(mesh.blocks, active, strict=True):
        np.testing.assert_array_equal(block.vertex_valid, mask)


@pytest.mark.parametrize("positive", (False, True))
@pytest.mark.parametrize("closed", (False, True))
def test_default_affine_multiblock_route_preserves_full_mapped_contact_decisions(
    closed: bool,
    positive: bool,
) -> None:
    from phydrax.discretization import (
        CellBlock,
        CellGeometrySpec,
        CellMesh,
        PeriodicMeshTopology,
    )
    from phydrax.geometry._embedding_state import MeshCertificateLimits
    from phydrax.meshing._periodic import certify_periodic_embedding

    coordinates = np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)))
    blocks = (
        CellBlock(
            "first",
            "triangle",
            np.asarray(((0, 1, 2) if positive else (0, 2, 1),), dtype=np.int32),
            global_ids=np.asarray((41,), dtype=np.int64),
        ),
        CellBlock(
            "second",
            "triangle",
            np.asarray(((0, 2, 3) if positive else (0, 3, 2),), dtype=np.int32),
            global_ids=np.asarray((73,), dtype=np.int64),
        ),
    )
    carrier = CellMesh(coordinates, blocks)
    topology = PeriodicMeshTopology(
        carrier,
        PeriodicCell(np.eye(2)),
        np.zeros(4, dtype=np.int64) if closed else np.arange(4, dtype=np.int64),
        np.asarray(coordinates, dtype=np.int64)
        if closed
        else np.zeros((4, 2), dtype=np.int64),
    )
    mesh = CellMesh(coordinates, blocks, periodic_topology=topology)
    geometry = CellGeometrySpec.affine(mesh)
    if not positive:
        with pytest.raises(ValueError):
            certify_periodic_embedding(mesh)
        with pytest.raises(ValueError):
            certify_periodic_embedding(mesh, geometry=geometry)
    elif closed:
        affine = certify_periodic_embedding(mesh)
        mapped = certify_periodic_embedding(mesh, geometry=geometry)
        assert affine.coordinate_scope == "exact_dyadic_closed_quotient_lift"
        assert mapped.coordinate_scope == "exact_source_group_images"
        assert mapped.global_embedding is not None
        assert mapped.global_embedding.status == "certified"
        assert affine.image_count == mapped.image_count == 9
        limits = MeshCertificateLimits(maximum_periodic_images=1)
        with pytest.raises(ValueError):
            certify_periodic_embedding(mesh, limits=limits)
        with pytest.raises(ValueError):
            certify_periodic_embedding(mesh, geometry=geometry, limits=limits)
    else:
        # Unjoined seam contacts remain nonincident closed contacts, even though
        # their open cell interiors are disjoint.
        with pytest.raises(ValueError, match="periodic_mapped_cell_overlap"):
            certify_periodic_embedding(mesh)
        with pytest.raises(ValueError, match="periodic_mapped_cell_overlap"):
            certify_periodic_embedding(mesh, geometry=geometry)
    np.testing.assert_array_equal(mesh.coordinates, coordinates)
    for block in mesh.blocks:
        np.testing.assert_array_equal(block.vertex_valid, np.ones((1, 3), dtype=np.bool_))


def test_affine_multiblock_embedding_rejects_cross_block_contact_with_foreign_ids() -> (
    None
):
    from phydrax.discretization import (
        CellBlock,
        CellGeometrySpec,
        CellMesh,
        PeriodicMeshTopology,
    )
    from phydrax.meshing._periodic import certify_periodic_embedding

    coordinates = np.tile(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))), (2, 1)
    )
    blocks = (
        CellBlock(
            "first",
            "triangle",
            np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int32),
            global_ids=np.asarray((41, 42), dtype=np.int64),
        ),
        CellBlock(
            "second",
            "triangle",
            np.asarray(((4, 5, 6), (4, 6, 7)), dtype=np.int32),
            global_ids=np.asarray((73, 74), dtype=np.int64),
        ),
    )
    carrier = CellMesh(coordinates, blocks)
    topology = PeriodicMeshTopology(
        carrier,
        PeriodicCell(np.eye(2)),
        np.asarray((0, 0, 0, 0, 4, 4, 4, 4), dtype=np.int64),
        np.asarray(coordinates, dtype=np.int64),
    )
    mesh = CellMesh(coordinates, blocks, periodic_topology=topology)
    with pytest.raises(ValueError, match="scientific IDs 41, 73"):
        certify_periodic_embedding(mesh)
    with pytest.raises(ValueError, match="overlap"):
        certify_periodic_embedding(mesh, geometry=CellGeometrySpec.affine(mesh))


def test_affine_embedding_bvh_refuses_original_scratch_before_growth() -> None:
    from phydrax._meshcore import (
        MeshcoreError,
        MeshcoreStatus,
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
        NativeExecutionScope(
            phx.meshing.MeshingLimits(
                maximum_work_units=100000,
                maximum_geometry_queries=100000,
                maximum_cavity_cells=10,
                maximum_scratch_bytes=128,
                maximum_wall_seconds=10.0,
            ).native_limits()
        ) as budget,
    ):
        with NativeHostStorageWorkspace(budget) as workspace:
            with pytest.raises(MeshcoreError) as failed:
                certify_periodic_embedding(mesh, scope=budget)
            assert failed.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert workspace.bound == 0
    assert ended.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence is not None
    assert budget.evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert budget.evidence.host_storage_live_bytes_upper == 0


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


@pytest.mark.parametrize(
    ("points", "vectors"),
    [
        (np.random.default_rng(3).random((60, 2)), np.eye(2)),
        (periodic_lattice_points(5), np.eye(2)),
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
        quotient.total_measure, abs(np.linalg.det(vectors)), rtol=_QUOTIENT_MEASURE_RTOL
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
    np.testing.assert_allclose(quotient.total_measure, 1.125, rtol=_QUOTIENT_MEASURE_RTOL)
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
    assert require_periodic_topology(reloaded).euler_characteristic == 0


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
    np.testing.assert_allclose(
        uniform.quotient.total_measure, 1.0, rtol=_QUOTIENT_MEASURE_RTOL
    )
    np.testing.assert_array_equal(
        np.bincount(uniform.parent_cells), np.full(counts[-1], 2**dimension)
    )
    _assert_positive_lifts(uniform.mesh)

    local = phx.meshing.refine_periodic_mesh(construction.mesh, cells=np.asarray((0, 3)))
    assert 1 <= local.bisected_edges <= 2
    assert local.quotient.quotient_counts[0] == counts[0] + local.bisected_edges
    assert local.quotient.euler_characteristic == 0
    assert local.quotient.boundary_facet_count == 0
    np.testing.assert_allclose(
        local.quotient.total_measure, 1.0, rtol=_QUOTIENT_MEASURE_RTOL
    )


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
        phx.meshing.PeriodicPointOrbits(
            periodic_lattice_points(4), PeriodicCell(np.eye(2))
        )
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
    source, specification = periodic_seed_request()
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay"))
        .plan(
            source, specification, coordinate_contract=phx.SpatialCoordinateContract.si()
        )
        .execute()
    )
    certification = result.certification
    if certification is None:
        raise AssertionError(
            "periodic.publication.certification must be present for accepted output; got None"
        )
    assert result.audit.passed and result.compliance.passed and certification.passed
    assert require_periodic_topology(result.mesh).euler_characteristic == 0
    achieved = dict(result.compliance.achieved)
    np.testing.assert_allclose(
        achieved["periodic:quotient_measure"], 1.0, atol=_UNIT_QUOTIENT_ATOL
    )
    assert achieved["periodic:embedding_images"] >= 9
    edge_points = np.asarray(result.mesh.coordinates)[
        simplex_entity_corners(result.mesh, 1)
    ]
    assert np.max(np.linalg.norm(edge_points[:, 1] - edge_points[:, 0], axis=1)) <= 0.35


@pytest.mark.parametrize("count", (4, 8), ids=("skew-res4", "skew-res8"))
def test_skew_periodic_frontal_construction_meets_exact_hard_statistics(
    count: int,
) -> None:
    M = phx.meshing
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    source = M.NativePeriodicSource(
        M.PeriodicPointOrbits(
            periodic_lattice_points(count) @ vectors, PeriodicCell(vectors)
        ),
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
    control = M.UniformSizeControl(scope, 1.0 / count, maximum_size=1.5 / count)
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
        limits=M.MeshingLimits(
            maximum_vertices=20000,
            maximum_cells=160000,
            maximum_edges=160000,
            maximum_faces=160000,
            maximum_connectivity_entries=1280000,
            maximum_data_bytes=20480000,
            maximum_scratch_bytes=81920000,
            maximum_work_units=40960000,
            maximum_cavity_cells=20000,
            maximum_geometry_queries=1280000,
            maximum_wall_seconds=120.0,
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
    assert achieved[f"{prefix}:p50_edge"] == 1.0 / count
    assert achieved[f"{prefix}:p95_edge"] == 1.0 / count
    assert achieved[f"{prefix}:maximum_edge"] <= 1.5 / count
    quotient = M.PeriodicQuotientEvidence(result.mesh)
    assert quotient.euler_characteristic == 0 and quotient.boundary_facet_count == 0
    np.testing.assert_allclose(quotient.total_measure, 1.0, atol=_UNIT_QUOTIENT_ATOL)
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
    with NativeExecutionScope(specification.limits.native_limits()) as native_scope:
        initial = native_scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=specification.limits.maximum_work_units,
            maximum_memory_bytes=specification.limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(
            lambda bound: native_scope.batch(0, bytes_upper=bound)
        ):
            transfer.source_associations(
                refined.target, scope=native_scope, budget=budget
            )


def test_represented_periodic_front_preserves_subdivided_winding_support() -> None:
    from phydrax.meshing.providers import _native_periodic_constraints as constraints
    from phydrax.meshing.providers._native_periodic import (
        _refine_target_size_sites,
        _triangulate_target_size_sites,
    )

    M = phx.meshing
    limits = M.MeshingLimits(
        maximum_work_units=40960000,
        maximum_scratch_bytes=81920000,
    )
    source, specification, control = skew_periodic_material_request(limits)
    domain = source.domain
    assert isinstance(domain, phx.discretization.CellMesh)
    strata = constraints.prepare_periodic_source_strata(
        domain, np.asarray(source.cell_regions), specification.protected_features
    )
    seeds = constraints.periodic_source_seed_points(
        strata, control.target_size, 0.0, limits.maximum_vertices
    )
    policy = M.SizeCompliancePolicy(absolute_tolerance=0.0, relative_tolerance=0.0)
    vectors = np.asarray(strata.cell.vectors, dtype=np.float64)
    started = perf_counter()
    with NativeExecutionScope(limits.native_limits()) as scope:
        state, work = _triangulate_target_size_sites(
            seeds,
            vectors,
            strata.cell,
            limits,
            M.MeshingStageKind.VOLUME_FILL,
            limits.maximum_vertices,
            0,
            scope=scope,
            source_strata=strata,
        )
        support = state.source_support
        assert support is not None
        protected = np.asarray(support.protected_edges, dtype=np.int64)
        assert np.any(protected[:, 2:] != 0)
        assert np.any(protected[:, :2] >= strata.points.shape[0])
        assert set(support.protected_edges) <= {
            tuple(int(value) for value in row) for row in state.keys
        }
        refined, total_work = _refine_target_size_sites(
            state,
            strata.cell,
            [control],
            policy,
            limits,
            started,
            M.MeshingStageKind.VOLUME_FILL,
            limits.maximum_vertices,
            work,
            scope=scope,
        )
        assert total_work > work
        assert refined.points.shape[0] > seeds.shape[0]
        np.testing.assert_array_equal(refined.points[: seeds.shape[0]], seeds)
        np.testing.assert_array_equal(
            refined.protected_vertices[: seeds.shape[0]], support.protected_vertices
        )
        assert not np.any(refined.protected_vertices[seeds.shape[0] :])
        assert refined.source_support is support
        assert refined.source_support.image_count > seeds.shape[0]
        keys, incidences = np.unique(
            refined.keys[refined.simplex_edges.reshape(-1)],
            axis=0,
            return_counts=True,
        )
        counts = {
            tuple(int(value) for value in row): count
            for row, count in zip(keys, incidences, strict=True)
        }
        assert all(counts[edge] == 2 for edge in support.protected_edges)
        regions, ancestry, bound_work = constraints.bind_periodic_source_strata(
            refined.points,
            refined.simplices,
            refined.simplex_shifts,
            strata,
            limits.maximum_work_units - total_work,
            scope=scope,
        )
        assert bound_work > 0 and len(ancestry) == refined.simplices.shape[0]
        assert set(regions) == {0, 1}
        assert all(
            np.all(strata.regions[np.asarray(parents)] == region)
            for parents, region in zip(ancestry, regions, strict=True)
        )


def test_periodic_feature_labels_follow_source_orbits_across_lattice_lifts() -> None:
    from phydrax.meshing._periodic import periodic_edge_size_evidence
    from phydrax.meshing.providers._native_periodic import (
        _periodic_feature_labels,
        _PeriodicCompliance,
    )

    M = phx.meshing
    limits = M.MeshingLimits(maximum_work_units=40960000, maximum_scratch_bytes=81920000)
    source, specification, _ = skew_periodic_material_request(limits)
    domain = source.domain
    assert isinstance(domain, phx.discretization.CellMesh)
    topology = require_periodic_topology(domain)
    assert isinstance(topology.cell, PeriodicCell)
    leaders = np.asarray(topology.orbit_representatives(0))
    vertices = np.asarray(domain.blocks[0].vertices)
    roots = np.asarray(topology.orbits(0)[0])[vertices]
    shifts = np.asarray(topology.vertex_shifts)[vertices]
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(lambda size: scope.batch(0, bytes_upper=size)):
            target = M.publish_periodic_simplices(
                np.asarray(domain.coordinates)[leaders],
                roots,
                shifts + np.asarray((2, -3), dtype=np.int64),
                topology.cell,
                scope=scope,
                budget=budget,
            )
            associations = periodic_source_associations(
                source, specification, target, scope=scope, budget=budget
            )
            lengths, _ = periodic_edge_size_evidence(target)
            compliance = _PeriodicCompliance([], [], [])
            organization = _periodic_feature_labels(
                domain,
                target,
                specification,
                lengths,
                compliance,
                associations,
                record_phase=None,
            )
    assert not compliance.issues
    assert len(organization.labels) == len(specification.protected_features) == 1
    selected = np.isin(
        np.asarray(target.entity_set(1).entity_ids),
        np.asarray(organization.labels[0].scope.entity_ids),
    )
    target_orbits = np.asarray(require_periodic_topology(target).orbits(1)[0])
    assert (
        np.unique(target_orbits[selected]).size
        == np.asarray(specification.protected_features[0].scope.entity_ids).size
    )


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
    source, specification, _ = skew_periodic_material_request(limits)
    domain = source.domain
    if not isinstance(domain, phx.discretization.CellMesh):
        raise AssertionError(
            f"periodic.material_request.domain must be a represented CellMesh; got {type(domain).__name__}"
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


def test_periodic_planar_cavity_refusal_retains_work_and_named_memory() -> None:
    M = phx.meshing
    limits = M.MeshingLimits(maximum_cavity_cells=1)
    source, specification, _ = skew_periodic_material_request(limits)
    domain = source.domain
    assert isinstance(domain, phx.discretization.CellMesh)
    before = np.asarray(domain.coordinates).copy()
    binding = source.binding_id
    with pytest.raises(M.MeshingFailure) as refused:
        M.NativeMeshingProvider(M.NativeMeshingOptions("periodic_delaunay")).plan(
            source,
            specification,
            coordinate_contract=phx.SpatialCoordinateContract.si(),
        ).execute()
    failure = refused.value
    assert failure.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    native = failure.__cause__
    while native is not None and not isinstance(native, NativeResourceFailure):
        native = native.__cause__
    assert isinstance(native, NativeResourceFailure)
    assert native.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert native.work_evidence.shape == (9,)
    assert native.work_evidence.dtype == np.dtype(np.uint64)
    memory = native.memory_counters
    assert memory is not None
    assert 0 < memory.limit_bytes <= limits.maximum_scratch_bytes
    assert 0 <= memory.live_bytes <= memory.peak_bytes <= memory.limit_bytes
    achieved = dict(failure.evidence.achieved)
    assert achieved["native_status"] == native.status
    assert (
        achieved["native_scope:native_memory:limit_bytes"] == limits.maximum_scratch_bytes
    )
    assert {
        key: value
        for key, value in achieved.items()
        if key.startswith("native_execution:")
    } == {
        f"native_execution:{index}": float(value)
        for index, value in enumerate(native.work_evidence)
    }
    assert {
        key: value for key, value in achieved.items() if key.startswith("native_memory:")
    } == {f"native_memory:{name}": float(value) for name, value in memory.items()}
    assert source.domain is domain and source.binding_id == binding
    np.testing.assert_array_equal(domain.coordinates, before)


def test_periodic_source_refusal_retains_earned_coordinate_phase() -> None:
    source, specification = periodic_seed_request()
    domain = source.domain
    assert isinstance(domain, phx.meshing.PeriodicPointOrbits)
    mesh = phx.meshing.periodic_delaunay_mesh(domain).mesh
    coordinates = np.asarray(mesh.coordinates).view(np.uint64).copy()
    binding = source.binding_id
    limits = specification.limits
    with NativeExecutionScope(limits.native_limits()) as scope:
        initial = scope.batch(0)
        budget = CoordinateEnclosureBudget(
            initial,
            maximum_work_units=1,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        initial.settle()
        with budget.retained_storage(lambda bound: scope.batch(0, bytes_upper=bound)):
            with pytest.raises(
                phx.meshing.MeshingFailure, match=r"periodic\.source_support\.resources"
            ) as refusal:
                periodic_source_associations(
                    source, specification, mesh, scope=scope, budget=budget
                )
            assert (
                refusal.value.category
                is phx.meshing.MeshingFailureCategory.RESOURCE_EXHAUSTED
            )
            cause = refusal.value.__cause__
            assert isinstance(cause, CoordinateEnclosureResourceError)
            assert cause.resource == "coefficient_work"
            assert cause.limit == 1 and cause.requested > cause.limit
            phase = cause.phase_evidence
            assert phase is not None
            assert phase.work_units >= 0
            assert phase.native_charged_work_units >= 0
            assert phase.peak_bytes_upper > 0
    assert source.binding_id == binding
    np.testing.assert_array_equal(
        np.asarray(mesh.coordinates).view(np.uint64), coordinates
    )


def _native_skew_front(threads: int) -> PeriodicRefinement:
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    points = periodic_lattice_points(4) @ vectors
    cell = PeriodicCell(vectors)
    limits = phx.meshing.MeshingLimits()
    with NativeExecutionScope(limits.native_limits()):
        initial = phx.geometry.PeriodicDelaunayTriangulation(points, cell)
        with NativeParallelExecution(threads):
            return periodic_refine_2d(
                points,
                vectors,
                np.asarray(cell.inverse_vectors),
                initial.simplices,
                initial.simplex_shifts,
                np.asarray(((0.25, np.nan, 0.375, np.nan, 1.0),), dtype=np.float64),
                np.asarray((0.5, 0.95), dtype=np.float64),
                absolute_tolerance=0.0,
                relative_tolerance=0.0,
                max_vertices=limits.maximum_vertices,
                max_cells=limits.maximum_cells,
                max_images=20000,
                max_work_units=limits.maximum_work_units,
                max_cavity_cells=limits.maximum_cavity_cells,
                max_scratch_bytes=limits.maximum_scratch_bytes,
            )


@pytest.fixture(scope="module")
def native_skew_front() -> PeriodicRefinement:
    return _native_skew_front(1)


@pytest.mark.parametrize("threads", (2, 7, 64))
def test_native_periodic_front_is_bitwise_deterministic_across_threads(
    threads: int, native_skew_front: PeriodicRefinement
) -> None:
    actual = _native_skew_front(threads)
    assert actual.status is native_skew_front.status is MeshcoreStatus.OK
    assert actual.counters == native_skew_front.counters
    assert actual.counters.commits > 0
    assert actual.counters.hard_issues == 0
    np.testing.assert_array_equal(
        actual.points.view(np.uint64), native_skew_front.points.view(np.uint64)
    )
    np.testing.assert_array_equal(actual.vertices, native_skew_front.vertices)
    np.testing.assert_array_equal(actual.shifts, native_skew_front.shifts)


def test_native_periodic_front_meets_unrelaxed_publication_statistics(
    native_skew_front: PeriodicRefinement,
) -> None:
    from phydrax.meshing.providers._native_periodic import _SiteMesh

    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    state = _SiteMesh(native_skew_front.points, vectors, native_skew_front)
    assert np.quantile(state.lengths, 0.5) == 0.25
    assert np.quantile(state.lengths, 0.95) == 0.25
    assert np.max(state.lengths) <= 0.375


@pytest.fixture(scope="module")
def grouped_skew_transfer_cycle() -> tuple[
    phx.meshing.MeshAdaptationResult,
    phx.meshing.MeshAdaptationResult,
    phx.meshing.MeshAdaptationResult,
]:
    D, M = phx.discretization, phx.meshing
    shifts = np.asarray(((0, 0), (1, 0), (1, 1), (0, 1)), dtype=np.int64)
    vectors = np.asarray(((1.0, 0.0), (0.25, 1.0)), dtype=np.float64)
    coordinates = shifts @ vectors
    blocks = tuple(
        D.CellBlock(
            name,
            "triangle",
            np.asarray((vertices,), dtype=np.int32),
            global_ids=np.asarray((identifier,), dtype=np.int64),
        )
        for name, vertices, identifier in (
            ("first-cohort", (0, 2, 3), 7),
            ("last-cohort", (0, 1, 2), 3),
        )
    )
    lifted = D.CellMesh(coordinates, blocks)
    topology = D.PeriodicMeshTopology(
        lifted, PeriodicCell(vectors), np.zeros(4, dtype=np.int64), shifts
    )
    source = M.certify_cell_mesh(
        D.CellMesh(coordinates, blocks, periodic_topology=topology),
        phx.SpatialCoordinateContract.si(),
    )
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(source.mesh.entity_set(2).entity_ids),
            policy=policy,
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    coarsened = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation(
                (),
                refined.target.mesh.entity_set(2).entity_ids,
                hierarchy=refined.hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarsened.status is M.MeshAdaptationStatus.COMPLETE
    assert coarsened.target.mesh.topology_id == source.mesh.topology_id
    repeated = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            coarsened.target,
            M.MarkedMeshAdaptation(
                coarsened.target.mesh.entity_set(2).entity_ids,
                hierarchy=coarsened.hierarchy,
            ),
            policy=policy,
        )
    )
    assert repeated.status is M.MeshAdaptationStatus.COMPLETE
    return refined, coarsened, repeated


@pytest.mark.parametrize("field_name", ("u", "vector", "flux"))
def test_grouped_skew_periodic_field_transfer_refine_coarsen_refine(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
    field_name: str,
) -> None:
    import jax.numpy as jnp

    D = phx.discretization
    refined, coarsened, repeated = grouped_skew_transfer_cycle
    assert len(refined.source.mesh.blocks) == len(coarsened.target.mesh.blocks) == 2
    witnesses = coarsened.coarsening_witnesses
    assert witnesses is not None
    assert witnesses.fine_cell_ids.size > coarsened.source.mesh.blocks[0].cell_count
    np.testing.assert_array_equal(
        witnesses.fine_cell_ids,
        np.concatenate(
            tuple(np.asarray(block.global_ids) for block in coarsened.source.mesh.blocks)
        ),
    )
    source_ids = np.concatenate(
        tuple(np.asarray(block.global_ids) for block in refined.source.mesh.blocks)
    )
    np.testing.assert_array_equal(
        witnesses.coarse_cell_ids, source_ids[refined.parent_cells]
    )
    np.testing.assert_array_equal(
        witnesses.fine_reference_vertices, refined.parent_reference_vertices
    )
    element = {
        "u": D.lagrange_element("triangle", 1),
        "vector": D.form_element("triangle", 1, 1, proxy="circulation"),
        "flux": D.form_element("triangle", 1, 1, twist="twisted", proxy="flux"),
    }[field_name]
    field = FiniteElementFieldSpec(field_name, element)

    def constant(points: jnp.ndarray, context: object) -> jnp.ndarray:
        del context
        if field_name == "u":
            return jnp.full(points.shape[:-1], 3.25, dtype=jnp.float64)
        return jnp.broadcast_to(jnp.asarray((0.3, -0.7)), points.shape)

    source = FiniteElementPlan(
        refined.source.mesh, field, coordinate_spec=refined.source.geometry
    ).prepare()
    values = source.project(field_name, constant)
    for adaptation, coarsening in (
        (refined, False),
        (coarsened, True),
        (repeated, False),
    ):
        target = FiniteElementPlan(
            adaptation.target.mesh, field, coordinate_spec=adaptation.target.geometry
        ).prepare()
        transfer = D.prepare_nested_field_transfer(
            source,
            target,
            None if coarsening else adaptation.parent_cells,
            field_name=field_name,
            parent_reference_vertices=None
            if coarsening
            else adaptation.parent_reference_vertices,
            source_geometry=adaptation.source.geometry,
            target_geometry=adaptation.target.geometry,
            coarsening_witnesses=witnesses if coarsening else None,
        )
        assert transfer.evidence.passed
        if field_name == "u":
            assert transfer.transfer.preserves_constants
            assert transfer.transfer.conservative
        else:
            assert transfer.evidence.defect("commuting") <= transfer.evidence.tolerance
        values = transfer.transfer.apply(values)
        expected = np.asarray(target.project(field_name, constant))
        bound = (
            _FIELD_TRANSFER_ROUNDOFF_ULPS
            * np.finfo(np.float64).eps
            * transfer.transfer.action_condition
            * max(float(np.max(np.abs(expected))), 1.0)
        )
        np.testing.assert_allclose(values, expected, rtol=0.0, atol=bound)
        source = target
    for degree in range(3):
        np.testing.assert_array_equal(
            repeated.target.mesh.entity_set(degree).entity_ids,
            refined.target.mesh.entity_set(degree).entity_ids,
        )


def test_grouped_skew_periodic_coarsening_refuses_tampered_scientific_cell(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
) -> None:
    D = phx.discretization
    _, coarsened, _ = grouped_skew_transfer_cycle
    witnesses = coarsened.coarsening_witnesses
    assert witnesses is not None
    fine_ids = witnesses.fine_cell_ids.copy()
    fine_ids[0] = np.max(fine_ids) + 1
    tampered = D.NestedReferenceWitnesses(
        fine_ids, witnesses.coarse_cell_ids, witnesses.fine_reference_vertices
    )
    field = FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    source = FiniteElementPlan(
        coarsened.source.mesh, field, coordinate_spec=coarsened.source.geometry
    ).prepare()
    target = FiniteElementPlan(
        coarsened.target.mesh, field, coordinate_spec=coarsened.target.geometry
    ).prepare()
    before = np.asarray(coarsened.source.mesh.coordinates).copy()
    with pytest.raises(ValueError, match=r"nested_reference\.coarsening.*absent ID"):
        D.prepare_nested_field_transfer(
            source,
            target,
            None,
            field_name="u",
            source_geometry=coarsened.source.geometry,
            target_geometry=coarsened.target.geometry,
            coarsening_witnesses=tampered,
        )
    np.testing.assert_array_equal(coarsened.source.mesh.coordinates, before)
    retained = coarsened.coarsening_witnesses
    assert retained is not None
    np.testing.assert_array_equal(retained.fine_cell_ids, witnesses.fine_cell_ids)


def test_periodic_history_checkpoint_binds_restore_scope_budget(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax.discretization import _cell_geometry as geometry_owner
    from phydrax.lifecycle import _meshing_sources as source_owner
    from phydrax.meshing._periodic import PeriodicRefinement as PeriodicHistory

    refined, _, _ = grouped_skew_transfer_cycle
    history = refined.hierarchy
    assert isinstance(history, PeriodicHistory)
    receipt = source_owner.write_meshing_source_closure(
        tmp_path / "periodic-history.checkpoint.zip", history
    )
    original_scope = geometry_owner.owned_coordinate_preparation
    original_reconstruct = source_owner.reconstruct_stored_fields
    active: list[CoordinateEnclosureBudget] = []
    observed: list[CoordinateEnclosureBudget] = []

    @contextmanager
    def restore_scope(
        work: int,
        storage: int,
        /,
        *,
        maximum_geometry_queries: int = 0,
    ) -> Iterator[CoordinateEnclosureBudget]:
        with original_scope(
            work, storage, maximum_geometry_queries=maximum_geometry_queries
        ) as budget:
            active.append(budget)
            try:
                yield budget
            finally:
                active.pop()

    def reconstruct[T](value: T, /, *, bindings: Mapping[str, object] | None = None) -> T:
        if isinstance(value, PeriodicHistory):
            assert bindings is not None and active
            assert bindings["budget"] is active[-1]
            observed.append(active[-1])
        return original_reconstruct(value, bindings=bindings)

    monkeypatch.setattr(geometry_owner, "owned_coordinate_preparation", restore_scope)
    monkeypatch.setattr(source_owner, "reconstruct_stored_fields", reconstruct)
    restored = source_owner.read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    assert isinstance(restored, PeriodicHistory)
    assert observed and not active
    assert all(budget.work_units > 0 for budget in observed)
    assert restored.refinement_id == history.refinement_id
    assert source_owner.source_content_equal(
        restored, history, limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    np.testing.assert_array_equal(restored.parent_cells, history.parent_cells)
    np.testing.assert_array_equal(
        restored.parent_reference_vertices.view(np.uint64),
        history.parent_reference_vertices.view(np.uint64),
    )


def test_periodic_history_reconstruction_refuses_missing_budget_binding(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
) -> None:
    from phydrax.lifecycle._meshing_source_records import reconstruct_stored_fields
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
    from phydrax.meshing._periodic import PeriodicRefinement as PeriodicHistory

    refined, _, _ = grouped_skew_transfer_cycle
    history = refined.hierarchy
    assert isinstance(history, PeriodicHistory)
    register_meshing_source_artifacts()
    with pytest.raises(
        TypeError, match=r"PeriodicRefinement\.budget.*explicit binding.*missing"
    ):
        reconstruct_stored_fields(history)


def test_periodic_history_checkpoint_refuses_one_unit_short_restore_budget(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.lifecycle import _meshing_sources as source_owner
    from phydrax.meshing._periodic import PeriodicRefinement as PeriodicHistory

    refined, _, _ = grouped_skew_transfer_cycle
    history = refined.hierarchy
    assert isinstance(history, PeriodicHistory)
    receipt = source_owner.write_meshing_source_closure(
        tmp_path / "periodic-history.checkpoint.zip", history
    )
    original_reconstruct = source_owner.reconstruct_stored_fields
    observed_work: dict[str, int] = {}
    refuse = False
    before = history.parent_cells.copy()

    def reconstruct[T](value: T, /, *, bindings: Mapping[str, object] | None = None) -> T:
        if not isinstance(value, PeriodicHistory):
            return original_reconstruct(value, bindings=bindings)
        assert bindings is not None
        restore_budget = bindings["budget"]
        assert isinstance(restore_budget, CoordinateEnclosureBudget)
        if refuse:
            restore_budget.maximum_work_units = observed_work[value.refinement_id] - 1
        restored = original_reconstruct(value, bindings=bindings)
        observed_work[value.refinement_id] = restore_budget.work_units
        return restored

    monkeypatch.setattr(source_owner, "reconstruct_stored_fields", reconstruct)
    source_owner.read_meshing_source_closure(
        receipt.path, expected_content_id=receipt.content_id
    )
    completed_work = observed_work[history.refinement_id]
    assert completed_work > 0
    refuse = True
    with pytest.raises(CoordinateEnclosureResourceError) as refused:
        source_owner.read_meshing_source_closure(
            receipt.path, expected_content_id=receipt.content_id
        )
    assert refused.value.resource == "coefficient_work"
    assert refused.value.limit == completed_work - 1
    assert refused.value.requested == completed_work
    np.testing.assert_array_equal(history.parent_cells, before)


def test_periodic_accuracy_coarsening_marks_use_physical_error_density() -> None:
    from tools.meshing_qualification.scenarios.periodic import (
        _periodic_qualification_coarsening_marks,
    )

    measures = np.asarray((0.125, 0.375, 0.5), dtype=np.float64)
    squared_errors = np.asarray((0.0001, 0.0008, 0.004), dtype=np.float64)
    parents = np.asarray((2, 0, 1, 0, 2, 1), dtype=np.int64)
    cell_ids = np.asarray((91, 17, 36, 25, 82, 48), dtype=np.int64)
    selected = _periodic_qualification_coarsening_marks(
        squared_errors, measures, parents, cell_ids, 0.05
    )
    np.testing.assert_array_equal(selected, (17, 36, 25, 48))
    # Physical measure units do not alter a normalized L2 error budget.
    np.testing.assert_array_equal(
        _periodic_qualification_coarsening_marks(
            squared_errors, 8.0 * measures, parents, cell_ids, 0.05
        ),
        selected,
    )
    # A coarse solve already below target permits every sibling family.
    np.testing.assert_array_equal(
        _periodic_qualification_coarsening_marks(
            squared_errors / 4.0, measures, parents, cell_ids, 0.05
        ),
        cell_ids,
    )


def test_periodic_coarsening_selection_matches_complete_execution(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
) -> None:
    from phydrax.meshing._periodic import periodic_coarsening_cell_ids

    M = phx.meshing
    refined, _, _ = grouped_skew_transfer_cycle
    ids = np.asarray(refined.target.mesh.entity_set(2).entity_ids, dtype=np.int64)
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    candidate = M.prepare_mesh_adaptation(
        refined.target,
        M.MarkedMeshAdaptation((), ids, hierarchy=refined.hierarchy),
        policy=policy,
    )
    selected = periodic_coarsening_cell_ids(candidate)
    np.testing.assert_array_equal(selected, np.sort(ids))
    accepted = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation((), selected, hierarchy=refined.hierarchy),
            policy=policy,
        )
    )
    assert accepted.status is M.MeshAdaptationStatus.COMPLETE
    assert accepted.target.mesh.topology_id == refined.source.mesh.topology_id
    assert sum(block.cell_count for block in accepted.target.mesh.blocks) < ids.size
    incomplete = M.prepare_mesh_adaptation(
        refined.target,
        M.MarkedMeshAdaptation((), ids[:1], hierarchy=refined.hierarchy),
        policy=policy,
    )
    assert periodic_coarsening_cell_ids(incomplete).size == 0
    rejected = M.execute_mesh_adaptation(incomplete)
    assert rejected.status is M.MeshAdaptationStatus.PARTIAL
    assert isinstance(rejected.evidence, M.BisectionEvidence)
    np.testing.assert_array_equal(rejected.evidence.rejected_coarsening_ids, ids[:1])
    assert rejected.target.mesh.topology_id == refined.target.mesh.topology_id


def test_multilevel_periodic_coarsening_retains_immediate_parent_source_actions(
    grouped_skew_transfer_cycle: tuple[
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
        phx.meshing.MeshAdaptationResult,
    ],
) -> None:
    M = phx.meshing
    first, _, _ = grouped_skew_transfer_cycle
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    second = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            first.target,
            M.MarkedMeshAdaptation(
                first.target.mesh.entity_set(2).entity_ids, hierarchy=first.hierarchy
            ),
            policy=policy,
        )
    )
    assert second.status is M.MeshAdaptationStatus.COMPLETE
    coarse = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            second.target,
            M.MarkedMeshAdaptation(
                (),
                second.target.mesh.entity_set(2).entity_ids,
                hierarchy=second.hierarchy,
            ),
            policy=policy,
        )
    )
    assert coarse.status is M.MeshAdaptationStatus.COMPLETE
    assert coarse.target.mesh.topology_id == first.target.mesh.topology_id
    old_origin = first.target.geometry.restriction_source
    new_origin = coarse.target.geometry.restriction_source
    assert old_origin is not None and new_origin is not None
    assert new_origin.source_geometry_id == old_origin.source_geometry_id
    assert new_origin.source_topology_id == old_origin.source_topology_id
    expected_elements, expected_routes, expected_bank = first.target.geometry.resolve(
        first.target.mesh
    )
    actual_elements, actual_routes, actual_bank = coarse.target.geometry.resolve(
        coarse.target.mesh
    )
    expected = {
        int(identifier): (
            element.element_id,
            np.asarray(expected_bank)[np.asarray(route)[row]],
        )
        for block, element, route in zip(
            first.target.mesh.blocks, expected_elements, expected_routes, strict=True
        )
        for row, identifier in enumerate(block.global_ids)
    }
    for block, element, route in zip(
        coarse.target.mesh.blocks, actual_elements, actual_routes, strict=True
    ):
        for row, identifier in enumerate(block.global_ids):
            element_id, coefficients = expected[int(identifier)]
            assert element.element_id == element_id
            np.testing.assert_array_equal(
                np.asarray(actual_bank)[np.asarray(route)[row]], coefficients
            )


def test_partial_periodic_coarsening_preserves_kept_fine_cell_actions() -> None:
    from phydrax.meshing._periodic import periodic_coarsening_cell_ids

    M = phx.meshing
    construction = M.periodic_delaunay_mesh(
        M.PeriodicPointOrbits(periodic_lattice_points(4), PeriodicCell(np.eye(2)))
    )
    source = M.certify_cell_mesh(construction.mesh, phx.SpatialCoordinateContract.si())
    policy = M.MeshAdaptationPolicy(M.MeshAdaptationRoute.NATIVE_BISECTION)
    refined = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            source,
            M.MarkedMeshAdaptation(source.mesh.entity_set(2).entity_ids),
            policy=policy,
        )
    )
    assert refined.status is M.MeshAdaptationStatus.COMPLETE
    history = refined.hierarchy
    assert isinstance(history, M.PeriodicRefinement)
    cells = np.concatenate(tuple(block.vertices for block in source.mesh.blocks))
    centers = np.mean(np.asarray(source.mesh.coordinates)[cells], axis=1)[:, 0] % 1
    fine_ids = refined.target.mesh.entity_set(2).entity_ids
    marked = np.asarray(fine_ids)[centers[np.asarray(history.parent_cells)] < 0.6]
    candidate = M.prepare_mesh_adaptation(
        refined.target,
        M.MarkedMeshAdaptation((), marked, hierarchy=history),
        policy=policy,
    )
    selected = periodic_coarsening_cell_ids(candidate)
    assert 0 < selected.size < np.asarray(fine_ids).size
    coarse = M.execute_mesh_adaptation(
        M.prepare_mesh_adaptation(
            refined.target,
            M.MarkedMeshAdaptation((), selected, hierarchy=history),
            policy=policy,
        )
    )
    assert coarse.status is M.MeshAdaptationStatus.COMPLETE
    coarse.target.audit.require_passed()
    assert (
        len(source.mesh.entity_set(2).entity_ids)
        < len(coarse.target.mesh.entity_set(2).entity_ids)
        < len(fine_ids)
    )
    expected = {
        int(identifier): element.element_id
        for block, element in zip(
            refined.target.mesh.blocks,
            refined.target.geometry.resolve(refined.target.mesh)[0],
            strict=True,
        )
        for identifier in block.global_ids
    }
    kept = 0
    for block, element in zip(
        coarse.target.mesh.blocks,
        coarse.target.geometry.resolve(coarse.target.mesh)[0],
        strict=True,
    ):
        for identifier in block.global_ids:
            if int(identifier) in expected:
                kept += 1
                assert element.element_id == expected[int(identifier)]
    assert kept > 0
    from phydrax.discretization._cell_geometry_transition_types import (
        NestedReferenceWitnesses,
    )
    from phydrax.meshing._adaptation import _complete_coarsening_witnesses

    witnesses = coarse.coarsening_witnesses
    assert witnesses is not None and coarse.lineage is not None
    fine = np.asarray(witnesses.fine_cell_ids)
    coarse_ids = np.asarray(witnesses.coarse_cell_ids)
    preserved = np.flatnonzero(fine == coarse_ids)
    assert preserved.size > 0
    changed = np.asarray(witnesses.fine_reference_vertices).copy()
    changed[preserved[0], 0, 0] += 0.125
    with pytest.raises(ValueError, match="coarsening.preserved_cell"):
        _complete_coarsening_witnesses(
            refined.target.mesh,
            coarse.target.mesh,
            coarse.lineage,
            NestedReferenceWitnesses(fine, coarse_ids, changed),
        )
