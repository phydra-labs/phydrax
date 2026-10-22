#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


import numpy as np
import pytest

from phydrax._meshcore import (
    MeshcoreError, meshcore_available, MeshcoreStatus, NativeExecutionScope,
)
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax import _geometry_predicates as predicates_owner
from phydrax.geometry import _supermesh as owner
from phydrax.discretization import CellBlock, CellMesh
from phydrax.geometry import (
    CommonRefinementCoverage,
    CommonRefinementPolicy,
    CommonRefinementStatus,
    PredicateMode,
    prepare_common_refinement,
)
from tests._support.common_refinement import (
    assert_complete_moments,
    CELL_CENTROID_RTOL,
    CELL_MEASURE_RTOL,
    cube_lattice,
    hexahedra as hexahedra_mesh,
    kuhn_tetrahedra,
    PARTITION_MEASURE_ATOL,
    perturbed_triangles,
    quad_mesh,
    SINGLE_CELL_RTOL,
    SOURCE_COVERAGE_RTOL,
    square_grid,
)


pytestmark = [
    pytest.mark.meshcore,
    pytest.mark.skipif(
        not meshcore_available(),
        reason="native phydrax-meshcore library is not available",
    ),
]


def test_identical_triangle_refinement_retains_cell_order_and_moments() -> None:
    mesh = perturbed_triangles(6, 3)
    refinement = prepare_common_refinement(mesh, mesh)

    assert refinement.succeeded
    cells = np.arange(mesh.blocks[0].cell_count, dtype=np.int64)
    np.testing.assert_array_equal(
        refinement.target_offsets, np.arange(cells.size + 1, dtype=np.int64)
    )
    np.testing.assert_array_equal(refinement.source_cells, cells)
    np.testing.assert_array_equal(refinement.target_cells, cells)
    np.testing.assert_allclose(
        refinement.volumes, refinement.source_measures, rtol=CELL_MEASURE_RTOL
    )
    np.testing.assert_allclose(
        refinement.first_moments / refinement.volumes[:, None],
        np.mean(np.asarray(mesh.coordinates)[np.asarray(mesh.blocks[0].vertices)], 1),
        rtol=CELL_CENTROID_RTOL,
    )
    np.testing.assert_array_equal(
        refinement.source_cell_global_ids, mesh.blocks[0].global_ids
    )
    # Neighbors sharing a face only touch: exact classification drops them.
    assert refinement.evidence.candidate_pair_count > refinement.entry_count
    assert refinement.source_mesh_id == refinement.target_mesh_id == mesh.mesh_id


def test_mixed_polygon_refinement_retains_ordered_overlap_partitions() -> None:
    points, quads = square_grid(5)
    # Moving vertex (0.4, 0.4) inward turns its lower-left quadrilateral into a
    # nonconvex dart that needs a certified cone from a vertex.
    points[14] = (0.28, 0.28)
    hexagon = np.asarray((0, 6, 7, 8, 2, 1))
    loops = [hexagon]
    loops += [quad for quad in quads[2:10]]
    loops += [quads[10][[0, 1, 2]], quads[10][[0, 2, 3]]]
    loops += [quad for quad in quads[11:]]
    source = CellMesh.from_polygons(points, loops)
    assert {block.cell_kind for block in source.blocks} == {
        "triangle",
        "quadrilateral",
        "polygon",
    }
    target = perturbed_triangles(7, 5)
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    refinement = prepare_common_refinement(source, target, policy=policy)

    assert_complete_moments(refinement, 2)
    simplices = np.asarray(refinement.simplices)
    edges = simplices[:, 1:] - simplices[:, :1]
    areas = 0.5 * (edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0])
    offsets = np.asarray(refinement.simplex_offsets)
    np.testing.assert_allclose(
        np.add.reduceat(areas, offsets[:-1]),
        refinement.volumes,
        atol=PARTITION_MEASURE_ATOL,
    )
    np.testing.assert_allclose(
        np.bincount(
            refinement.source_cells,
            weights=refinement.volumes,
            minlength=refinement.source_cell_count,
        ),
        refinement.source_measures,
        rtol=SOURCE_COVERAGE_RTOL,
    )
    rows = np.asarray(refinement.target_cells)
    assert np.all(np.diff(rows) >= 0)
    ordered = np.lexsort((np.asarray(refinement.source_cells), rows))
    np.testing.assert_array_equal(ordered, np.arange(rows.size, dtype=np.int64))


def test_reflex_polygon_uses_a_certified_cone_apex() -> None:
    # Reflex vertex (1, 1): the cone from vertex 0 folds, the cone from vertex 1
    # is certified.
    dart = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 1.0), (0.0, 2.0), (1.0, 1.0))),
        [np.arange(4, dtype=np.int64)],
    )
    square = CellMesh.from_polygons(
        np.asarray(((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))),
        [np.arange(4, dtype=np.int64)],
    )
    refinement = prepare_common_refinement(
        dart,
        square,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )

    assert refinement.succeeded
    assert float(refinement.source_measures[0]) == pytest.approx(
        1.0, rel=SINGLE_CELL_RTOL
    )
    assert float(refinement.volumes[0]) == pytest.approx(1.0, rel=SINGLE_CELL_RTOL)
    np.testing.assert_allclose(
        refinement.first_moments[0], (1.0, 1.0), rtol=CELL_MEASURE_RTOL
    )


@pytest.mark.parametrize(
    "pair",
    ("tetrahedra_hexahedra", "hexahedra_polyhedra", "prisms_tetrahedra"),
)
def test_volume_family_refinement_integrates_unit_cube_moments(pair: str) -> None:
    tetrahedra = kuhn_tetrahedra(3)
    hexahedra = hexahedra_mesh(2)
    points, corners = cube_lattice(2)
    hexahedron_faces = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4))
    hexahedron_faces += ((1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6))
    polyhedra = CellMesh.from_polyhedra(
        points, [[cell[list(face)] for face in hexahedron_faces] for cell in corners]
    )
    prism_points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 0)),
        dtype=np.float64,
    )
    prism_points = np.concatenate((prism_points, ((1, 1, 1),)))
    prisms = CellMesh(
        prism_points,
        # ty: ignore[invalid-argument-type]
        (CellBlock("prisms", "prism", ((0, 1, 2, 3, 4, 5), (1, 6, 2, 4, 7, 5))),),
    )
    policy = CommonRefinementPolicy(second_moments=True)
    pairs = {
        "tetrahedra_hexahedra": (tetrahedra, hexahedra),
        "hexahedra_polyhedra": (hexahedra, polyhedra),
        "prisms_tetrahedra": (prisms, tetrahedra),
    }
    source, target = pairs[pair]
    assert_complete_moments(prepare_common_refinement(source, target, policy=policy), 3)


def test_pyramid_refinement_retains_its_volume() -> None:
    hexahedra = hexahedra_mesh(2)
    pyramid_points = np.asarray(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0.5, 0.5, 0.5)), dtype=np.float64
    )
    pyramid = CellMesh(
        pyramid_points,
        # ty: ignore[invalid-argument-type]
        (CellBlock("pyramids", "pyramid", ((0, 1, 2, 3, 4),)),),
    )
    refinement = prepare_common_refinement(
        pyramid,
        hexahedra,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )
    assert refinement.succeeded
    assert float(np.sum(refinement.volumes)) == pytest.approx(
        1.0 / 6.0, rel=CELL_MEASURE_RTOL
    )


def test_coverage_policy_distinguishes_source_and_target_gaps() -> None:
    source = quad_mesh(4)
    points, quads = square_grid(2)
    inner = CellMesh(0.5 * points + 0.25, (CellBlock("inner", "quadrilateral", quads),))
    complete = prepare_common_refinement(source, inner)
    covered_target = prepare_common_refinement(
        source,
        inner,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.TARGET),
    )
    covered_source = prepare_common_refinement(
        source,
        inner,
        policy=CommonRefinementPolicy(coverage=CommonRefinementCoverage.SOURCE),
    )

    assert complete.status is CommonRefinementStatus.COVERAGE_GAP
    assert complete.evidence.source_gap_count == 12
    assert complete.evidence.target_gap_count == 0
    assert covered_target.succeeded
    assert covered_source.status is CommonRefinementStatus.COVERAGE_GAP
    np.testing.assert_allclose(np.sum(complete.volumes), 0.25, rtol=CELL_MEASURE_RTOL)


def test_overlapping_source_cells_report_double_target_coverage() -> None:
    # Two cells covering [0.5, 1] x [0, 1] twice.
    overlapping = CellMesh.from_polygons(
        np.asarray(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.5, 1.0),
            )
        ),
        [np.arange(4, dtype=np.int64), np.arange(4, 8, dtype=np.int64)],
    )
    refinement = prepare_common_refinement(overlapping, quad_mesh(2))

    assert refinement.status is CommonRefinementStatus.DOUBLE_COVERAGE
    assert refinement.evidence.target_double_count == 2


def test_filtered_sliver_refuses_while_exact_refinement_succeeds() -> None:
    # Nearly collinear corner: the filter cannot resolve it, exact predicates can.
    sliver = np.asarray(((0.5, 0.5), (12.0, 12.0), (24.0, 24.0 + 2.0**-48)))
    # ty: ignore[invalid-argument-type]
    mesh = CellMesh.from_triangles(sliver, ((0, 1, 2),))
    filtered = prepare_common_refinement(
        mesh, mesh, policy=CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED)
    )
    exact = prepare_common_refinement(mesh, mesh)
    assert filtered.status is CommonRefinementStatus.PREDICATE_UNCERTAIN
    # One unresolved cell in each of the two (identical) meshes.
    assert filtered.evidence.uncertain_predicate_count == 2
    assert filtered.entry_count == 0
    assert exact.succeeded


def test_outside_exact_domain_reports_uncertain_refinement() -> None:
    tiny = CellMesh.from_triangles(
        1.0e-40 * np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        ((0, 1, 2),),
    )
    outside = prepare_common_refinement(tiny, tiny)
    assert outside.status is CommonRefinementStatus.PREDICATE_UNCERTAIN


def test_bowtie_cell_reports_invalid_geometry() -> None:
    bowtie = CellMesh(
        np.asarray(((0.0, 0.0), (1.0, 1.0), (1.0, 0.0), (0.0, 1.0))),
        # ty: ignore[invalid-argument-type]
        (CellBlock("bowtie", "quadrilateral", ((0, 1, 2, 3),)),),
    )
    invalid = prepare_common_refinement(bowtie, quad_mesh(1))

    assert invalid.status is CommonRefinementStatus.INVALID_GEOMETRY
    assert invalid.evidence.invalid_cell_count == 1


@pytest.mark.parametrize(
    "policy",
    (
        pytest.param(CommonRefinementPolicy(maximum_candidate_pairs=10), id="candidates"),
        pytest.param(CommonRefinementPolicy(maximum_accepted_pairs=10), id="accepted"),
    ),
)
def test_resource_refusal_retains_empty_publication(
    policy: CommonRefinementPolicy,
) -> None:
    source = quad_mesh(6)
    target = perturbed_triangles(5, 1)
    baseline = prepare_common_refinement(source, target)
    refused = prepare_common_refinement(source, target, policy=policy)
    assert refused.status is CommonRefinementStatus.RESOURCE_LIMIT
    assert refused.entry_count == 0
    np.testing.assert_array_equal(refused.target_offsets, 0)
    assert baseline.succeeded
    assert baseline.evidence.retained_bytes > 0


def test_memory_refusal_precedes_an_impossible_negative_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = quad_mesh(6), perturbed_triangles(5, 1)

    def unadmitted(*args: object, **kwargs: object) -> None:
        pytest.fail("decomposition allocated before minimum-publication admission")

    monkeypatch.setattr(owner, "_decompose", unadmitted)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        prepare_common_refinement(
            source, target, policy=CommonRefinementPolicy(maximum_memory_bytes=4096)
        )
    assert caught.value.resource == "polynomial_storage"
    assert caught.value.limit == 4096
    assert caught.value.requested == 4332
    assert caught.value.completed == 0


def test_post_decomposition_refusal_keeps_real_columns_when_failure_artifact_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = quad_mesh(2), perturbed_triangles(2, 3)
    baseline = prepare_common_refinement(source, target)

    def refuse_candidates(
        first: owner.Decomposition, second: owner.Decomposition, limit: int,
        storage: owner._RefinementStorage,
    ) -> None:
        # Exercise admission failure after real decompositions, independently of
        # changing BVH frontier sizes. No scientific cell columns are mocked.
        storage.set("bvh", 250_001)
        pytest.fail("a refused BVH bank was admitted")

    monkeypatch.setattr(owner, "_candidate_pairs", refuse_candidates)
    refused = prepare_common_refinement(
        source, target, policy=CommonRefinementPolicy(maximum_memory_bytes=250_000)
    )
    assert refused.status is CommonRefinementStatus.RESOURCE_LIMIT
    assert refused.entry_count == 0
    np.testing.assert_array_equal(refused.source_measures, baseline.source_measures)
    np.testing.assert_array_equal(refused.target_measures, baseline.target_measures)
    np.testing.assert_array_equal(
        refused.source_first_moments, baseline.source_first_moments
    )
    np.testing.assert_array_equal(
        refused.target_first_moments, baseline.target_first_moments
    )


def test_repeated_refinement_retains_content_identity() -> None:
    source = quad_mesh(3)
    target = perturbed_triangles(4, 2)
    first = prepare_common_refinement(source, target)
    second = prepare_common_refinement(source, target)

    assert first.refinement_id == second.refinement_id
    np.testing.assert_array_equal(first.volumes, second.volumes)


def test_refinement_refuses_wrong_mesh_kind() -> None:
    source = quad_mesh(3)
    with pytest.raises(TypeError, match="CellMesh"):
        # ty: ignore[invalid-argument-type]
        prepare_common_refinement(source, object())


def test_refinement_refuses_incompatible_dimensions() -> None:
    source = quad_mesh(3)
    with pytest.raises(ValueError, match="dimension"):
        prepare_common_refinement(source, kuhn_tetrahedra(1))


def test_policy_refuses_device_predicate_mode() -> None:
    with pytest.raises(ValueError, match="FILTERED or EXACT"):
        CommonRefinementPolicy(predicate_mode=PredicateMode.FILTERED_DEVICE)


def _bounded_refinement_pair(kind: str) -> tuple[CellMesh, CellMesh]:
    if kind == "standard":
        return perturbed_triangles(2, 7), quad_mesh(2)
    if kind == "mixed":
        points, quads = square_grid(2)
        points[4] = (0.35, 0.4)
        loops = [
            np.asarray((0, 3, 4, 5, 2, 1)),
            quads[2],
            quads[3][[0, 1, 2]],
            quads[3][[0, 2, 3]],
        ]
        return CellMesh.from_polygons(points, loops), perturbed_triangles(2, 3)
    points, corners = cube_lattice(1)
    routes = ((0, 2, 3, 1), (4, 5, 7, 6), (0, 1, 5, 4))
    routes += ((1, 3, 7, 5), (3, 2, 6, 7), (2, 0, 4, 6))
    polyhedra = CellMesh.from_polyhedra(
        points, [[cell[list(face)] for face in routes] for cell in corners]
    )
    return hexahedra_mesh(1), polyhedra


def _bounded_refinement(
    source: CellMesh, target: CellMesh, memory: int,
) -> tuple[owner.PreparedCommonRefinement, int]:
    policy = CommonRefinementPolicy(
        maximum_memory_bytes=memory, second_moments=True, overlap_simplices=True
    )
    with NativeExecutionScope(policy.native_limits()) as scope:
        with scope.batch(
            policy.maximum_exact_work,
            queries_upper=policy.maximum_exact_work,
            bytes_upper=memory,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.maximum_exact_work,
                maximum_memory_bytes=memory,
            )
            with budget.activate(), budget.temporary_scope():
                result = prepare_common_refinement(
                    source, target, policy=policy, scope=scope, budget=budget
                )
                alias = result.simplices
                retained = budget.retained_basis_bytes
            assert budget.temporary_bytes_upper == 0
            assert retained == budget.retained_basis_bytes > 0
            # Aliases belong to the retained graph, not to expired call scratch.
            assert budget.retained_owner_objects[id(alias)] is alias
            budget.retain_owner(alias)
            assert budget.retained_basis_bytes == retained
            required = max(budget.peak_bytes_upper, ticket.managed_bytes)
    return result, required


@pytest.mark.parametrize("kind", ("standard", "mixed", "polyhedral"))
def test_real_refinement_storage_boundary_and_retained_aliases(kind: str) -> None:
    source, target = _bounded_refinement_pair(kind)
    first, required = _bounded_refinement(source, target, 256 * 1024**2)
    exact, repeated = _bounded_refinement(source, target, required)
    assert repeated == required
    assert_complete_moments(first, source.topological_dimension)
    assert_complete_moments(exact, source.topological_dimension)
    for field in (
        "target_offsets", "target_cells", "source_cells", "volumes",
        "first_moments", "second_moments", "simplex_offsets", "simplices",
    ):
        np.testing.assert_array_equal(getattr(first, field), getattr(exact, field))
    simplices = np.asarray(first.simplices)
    edges = simplices[:, 1:] - simplices[:, :1]
    if source.topological_dimension == 2:
        measures = 0.5 * (
            edges[:, 0, 0] * edges[:, 1, 1] - edges[:, 0, 1] * edges[:, 1, 0]
        )
    else:
        measures = np.einsum(
            "ij,ij->i", edges[:, 0], np.cross(edges[:, 1], edges[:, 2])
        ) / 6.0
    offsets = np.asarray(first.simplex_offsets)[:-1]
    np.testing.assert_allclose(
        np.add.reduceat(measures, offsets), first.volumes, atol=PARTITION_MEASURE_ATOL
    )
    np.testing.assert_allclose(
        np.add.reduceat(measures[:, None] * np.mean(simplices, axis=1), offsets, axis=0),
        first.first_moments, rtol=CELL_MEASURE_RTOL, atol=PARTITION_MEASURE_ATOL,
    )
    try:
        refused, _ = _bounded_refinement(source, target, required - 1)
    except CoordinateEnclosureResourceError as error:
        assert error.resource in ("polynomial_storage", "retained_basis")
    except MeshcoreError as error:
        assert error.status is MeshcoreStatus.CAPACITY_EXCEEDED
    else:
        assert refused.status is CommonRefinementStatus.RESOURCE_LIMIT
        assert refused.entry_count == 0


def test_storage_admission_precedes_first_decomposition_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = quad_mesh(1)

    def unadmitted(*args: object, **kwargs: object) -> None:
        pytest.fail("decomposition allocated before storage admission")

    monkeypatch.setattr(owner, "_decompose", unadmitted)
    with pytest.raises(CoordinateEnclosureResourceError):
        _bounded_refinement(mesh, mesh, 8191)


def test_live_host_budget_keeps_standalone_partial_native_compatibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = quad_mesh(2), perturbed_triangles(2, 3)
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    baseline = prepare_common_refinement(source, target, policy=policy)

    def unsupported_target(*args: object, **kwargs: object) -> None:
        pytest.fail("standalone host budget was promoted to a native ticket")

    monkeypatch.setattr(HostBatchTicket, "native_target", unsupported_target)
    # Zero host source-query allowance was supported before native accounting;
    # standalone partial coverage must not invent a new query allowance.
    with HostBatchTicket(
        policy.maximum_exact_work, bytes_upper=policy.maximum_memory_bytes
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=policy.maximum_exact_work,
            maximum_memory_bytes=policy.maximum_memory_bytes,
        )
        with budget.activate():
            result = prepare_common_refinement(
                source, target, policy=policy, budget=budget
            )
        assert result.succeeded
        assert budget.ticket is ticket
        assert not ticket.native_accounting
        assert ticket.geometry_queries == 0
        assert ticket.work_units > 0
        assert budget.temporary_bytes_upper == 0
        assert budget.retained_owner_objects[id(result.simplices)] is result.simplices
    for field in ("volumes", "first_moments", "second_moments", "simplices"):
        np.testing.assert_array_equal(getattr(result, field), getattr(baseline, field))


@pytest.mark.parametrize("dimension", (2, 3))
def test_decomposition_preserves_uncached_native_predicate_schedule(
    dimension: int, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = CommonRefinementPolicy()
    # Cancellation zeros, not duplicate-point/axis structural zeros: every
    # determinant monomial is nonzero, so the filter must defer to native exact
    # classification. Repeat each row to detect accidental request memoization.
    vertices = (
        ((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
        if dimension == 2 else
        ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 3.0, 4.0), (3.0, 4.0, 5.0))
    )
    points = tuple(np.repeat(np.asarray(vertex)[None, :], 2, axis=0) for vertex in vertices)
    filtered = owner._orientation(points, PredicateMode.FILTERED)
    assert not np.any(filtered.certain)
    name = f"orient{dimension}d"
    native = predicates_owner._EXACT_ROUTES[name]
    rows: list[int] = []

    def record(*values: np.ndarray) -> np.ndarray:
        rows.append(values[0].shape[0])
        return native(*values)

    monkeypatch.setitem(predicates_owner._EXACT_ROUTES, name, record)
    with NativeExecutionScope(policy.native_limits()) as scope:
        for _ in range(2):
            expected = owner._orientation(points, PredicateMode.EXACT)
        with scope.batch(
            policy.maximum_exact_work,
            queries_upper=policy.maximum_exact_work,
            bytes_upper=policy.maximum_memory_bytes,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.maximum_exact_work,
                maximum_memory_bytes=policy.maximum_memory_bytes,
            )
            with budget.activate():
                for _ in range(2):
                    actual = owner._orientation(
                        points, PredicateMode.EXACT, budget=budget
                    )
                    np.testing.assert_array_equal(actual.signs, expected.signs)
                    np.testing.assert_array_equal(actual.certain, expected.certain)
            assert budget.exact_native_sign_cache == {}
            assert budget.temporary_bytes_upper == 0
    assert rows == [2, 2, 2, 2]


def test_source_decomposition_exact_boundary_precedes_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = quad_mesh(1)
    policy = CommonRefinementPolicy()
    with HostBatchTicket(
        policy.maximum_exact_work, bytes_upper=policy.maximum_memory_bytes
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=policy.maximum_exact_work,
            maximum_memory_bytes=policy.maximum_memory_bytes,
        )
        with budget.activate(), budget.live_storage() as lease:
            storage = owner._RefinementStorage(budget, lease)
            required = 8192 + owner._decomposition_bound(mesh, storage)

    class AllocationReached(Exception):
        pass

    def admitted(*args: object, **kwargs: object) -> None:
        raise AllocationReached

    monkeypatch.setattr(owner, "_decompose", admitted)
    with pytest.raises(AllocationReached):
        _bounded_refinement(mesh, mesh, required)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        _bounded_refinement(mesh, mesh, required - 1)
    assert caught.value.requested == required
    assert caught.value.limit == required - 1


def test_repeated_native_request_releases_scratch_but_keeps_prior_aliases() -> None:
    mesh = quad_mesh(1)
    policy = CommonRefinementPolicy(second_moments=True, overlap_simplices=True)
    with NativeExecutionScope(policy.native_limits()) as scope:
        with scope.batch(
            policy.maximum_exact_work,
            queries_upper=policy.maximum_exact_work,
            bytes_upper=policy.maximum_memory_bytes,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.maximum_exact_work,
                maximum_memory_bytes=policy.maximum_memory_bytes,
            )
            with budget.activate(), budget.temporary_scope():
                first = prepare_common_refinement(
                    mesh, mesh, policy=policy, scope=scope, budget=budget
                )
                alias = first.simplices
            first_retained = budget.retained_basis_bytes
            assert budget.temporary_bytes_upper == 0
            with budget.activate(), budget.temporary_scope():
                second = prepare_common_refinement(
                    mesh, mesh, policy=policy, scope=scope, budget=budget
                )
            assert budget.ticket is ticket
            assert budget.temporary_bytes_upper == 0
            assert budget.retained_basis_bytes > first_retained
            retained = budget.retained_basis_bytes
            assert budget.retained_owner_objects[id(alias)] is alias
            budget.retain_owner(alias)
            assert budget.retained_basis_bytes == retained
            np.testing.assert_array_equal(alias, second.simplices)
            assert first.refinement_id == second.refinement_id


def test_native_tightening_borrows_original_target_and_consumed_residuals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mesh = quad_mesh(1)
    policy = CommonRefinementPolicy(maximum_exact_work=10_000)
    prepare_host = owner._prepare_common_refinement_host
    observed: list[bool] = []
    with NativeExecutionScope(policy.native_limits()) as scope:
        with scope.batch(
            policy.maximum_exact_work,
            queries_upper=policy.maximum_exact_work,
            bytes_upper=policy.maximum_memory_bytes,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=policy.maximum_exact_work,
                maximum_memory_bytes=policy.maximum_memory_bytes,
            )
            with budget.activate():
                budget.reserve(200)
                budget.reserve_geometry_queries(7)
                expected_work = ticket.remaining_work_units
                expected_queries = ticket.remaining_geometry_queries

                def borrowed(
                    source: CellMesh, target: CellMesh,
                    selected: CommonRefinementPolicy, storage: owner._RefinementStorage,
                ) -> owner.PreparedCommonRefinement:
                    native = owner.meshcore_owner.current_native_execution_scope()
                    assert native is not None
                    assert native.parent is scope
                    assert native._borrowed_ticket is ticket
                    assert storage.budget is budget
                    assert budget.ticket is ticket
                    assert native.limits.maximum_work_units == expected_work
                    assert native.limits.maximum_geometry_queries == expected_queries
                    observed.append(True)
                    return prepare_host(source, target, selected, storage)

                monkeypatch.setattr(owner, "_prepare_common_refinement_host", borrowed)
                result = prepare_common_refinement(
                    mesh, mesh, policy=policy, scope=scope, budget=budget
                )
            assert result.succeeded
            assert observed == [True]
            assert budget.temporary_bytes_upper == 0


def test_static_dimension_error_precedes_minimum_artifact_refusal() -> None:
    with pytest.raises(ValueError, match="dimension"):
        prepare_common_refinement(
            quad_mesh(1), kuhn_tetrahedra(1),
            policy=CommonRefinementPolicy(maximum_memory_bytes=1),
        )
