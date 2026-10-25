#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Curved-source boundary adaptation and off-edge insertion lineage."""

from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from fractions import Fraction

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import (
    meshcore_available,
    NativeExecutionLimits,
    NativeExecutionScope,
    TetMesh3D,
    TetMeshCurvedSources,
    TetMeshEdgeConstruction,
    TetMeshInsertionPlan,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization import CellBlock, CellMesh
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.geometry._embedding_state import MeshCertificateLimits
from phydrax.geometry._implicit_cavity_atlas import (
    build_implicit_cavity_atlas,
    CavitySnapshot,
    prepare_implicit_cavity_edit,
)
from phydrax.geometry.implicit._analytic_profile import AnalyticImplicitProfile
from phydrax.geometry.implicit._projection_cover import implicit_facet_projection_bounds
from phydrax.meshing._lineage_records import EntityLineageKind, MeshLineage
from phydrax.meshing._tetra_metric import (
    _CurvedAuthority,
    _insertion_support,
    _NativeMetricBudget,
    _reprove_curved_publication,
    _SplitSamples,
    CurvedBoundarySource,
    execute_tetra_metric_adaptation,
    MetricRemeshingStatus,
    TetraMetricOutcome,
    TetraMetricSource,
)
from phydrax.meshing._topology_edit import assemble_topology_edit
from phydrax.units import METER
from tests._support.cell_meshes import kuhn_cube_arrays
from tests._support.coordinate_references import coordinate_reference_budget
from tests._support.metric_meshes import TETRA_GEOMETRY_TOLERANCE


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore library is unavailable"
)

_RADIUS = 0.371
_LIMITS = NativeExecutionLimits(1_000_000_000, 10_000_000, 512, 8_000_000_000, np.inf)
# Source-query allowance of the reference ledgers that issue whole-facet proofs.
_PROOF_QUERIES = 1_000_000
# Whole-facet tolerance of the level-1 cone: its first surface collapse
# produces facets up to about 0.0554 from the sphere.
_CONE_TOLERANCE = 0.06


@contextmanager
def _metric_request(
    *, mesh: TetMesh3D | None = None,
) -> Iterator[tuple[NativeExecutionScope, CoordinateEnclosureBudget]]:
    ticket = HostBatchTicket(0)
    budget = CoordinateEnclosureBudget(
        ticket,
        maximum_work_units=_LIMITS.maximum_work_units,
        maximum_memory_bytes=_LIMITS.maximum_managed_bytes,
    )
    with ticket, budget.activate():
        pass
    with NativeExecutionScope(_LIMITS, mesh=mesh) as scope:
        yield scope, budget


def _icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    golden = (1.0 + 5.0**0.5) / 2.0
    corners = (
        (-1, golden, 0), (1, golden, 0), (-1, -golden, 0), (1, -golden, 0),
        (0, -1, golden), (0, 1, golden), (0, -1, -golden), (0, 1, -golden),
        (golden, 0, -1), (golden, 0, 1), (-golden, 0, -1), (-golden, 0, 1),
    )  # fmt: skip
    faces = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9),
        (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2),
        (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
        (8, 6, 7), (9, 8, 1),
    ]  # fmt: skip
    points = [np.asarray(point, dtype=np.float64) for point in corners]
    points = [point / np.linalg.norm(point) for point in points]
    for _ in range(level):
        middle: dict[tuple[int, int], int] = {}

        def split(first: int, second: int) -> int:
            key = (min(first, second), max(first, second))
            if key not in middle:
                point = points[first] + points[second]
                points.append(point / np.linalg.norm(point))
                middle[key] = len(points) - 1
            return middle[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = split(a, b), split(b, c), split(c, a)
            refined += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = refined
    return np.asarray(points), np.asarray(faces, dtype=np.int32)


def _ball(level: int) -> CellMesh:
    """Two-shell ball carrier whose outer vertices lie on the analytic sphere."""
    shell, faces = _icosphere(level)
    count = shell.shape[0]
    points = np.vstack((_RADIUS * shell, 0.5 * _RADIUS * shell, np.zeros((1, 3))))
    center = 2 * count
    cells = []
    for row in faces:
        i, j, k = sorted(int(value) for value in row)
        for cell in (
            [i, j, k, k + count],
            [i, j, j + count, k + count],
            [i, i + count, j + count, k + count],
            [i + count, j + count, k + count, center],
        ):
            corners = points[cell]
            if np.linalg.det(corners[1:] - corners[:1]) < 0.0:
                cell[1], cell[2] = cell[2], cell[1]
            cells.append(cell)
    return CellMesh(
        points, (CellBlock("ball", "tetrahedron", np.asarray(cells, dtype=np.int32)),)
    )


def _profile() -> AnalyticImplicitProfile:
    geometry = phx.geometry.Sphere((0.0, 0.0, 0.0), _RADIUS).compile(
        coordinate_unit=METER
    )
    return AnalyticImplicitProfile(geometry, phx.SpatialCoordinateContract.si())


def _boundary_rows(mesh: CellMesh) -> np.ndarray:
    connectivity = mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    mask = np.zeros((mesh.entity_set(2).count,), dtype=np.bool_)
    mask[np.asarray(connectivity.boundary_faces)] = True
    return mask


def _outward_boundary(mesh: CellMesh) -> np.ndarray:
    cells = np.asarray(mesh.blocks[0].vertices)
    local = np.asarray(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)), dtype=np.int32)
    faces = cells[:, local].reshape((-1, 3))
    _, inverse, counts = np.unique(
        np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True
    )
    return faces[counts[inverse.reshape(-1)] == 1]


def _cone(level: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shell, faces = _icosphere(level)
    points = np.vstack((_RADIUS * shell, np.zeros((1, 3))))
    center = shell.shape[0]
    cells = []
    for a, b, c in faces.tolist():
        cell = [center, a, b, c]
        corners = points[cell]
        if np.linalg.det(corners[1:] - corners[:1]) < 0.0:
            cell[2], cell[3] = cell[3], cell[2]
        cells.append(cell)
    return points, np.asarray(cells, dtype=np.int32), faces


def _curved_cone(
    tolerance: float, *, source: CurvedBoundarySource | None = None,
) -> tuple[TetMesh3D, CurvedBoundarySource, bytes]:
    points, cells, faces = _cone(1)
    if source is None:
        with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
            source = CurvedBoundarySource(
                _profile(), tolerance, MeshCertificateLimits(), np.ones((1,), dtype=np.bool_),
                budget=budget,
            )
    outward = faces.copy()
    normals = np.cross(
        points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]]
    )
    flip = np.einsum("ij,ij->i", normals, points[faces[:, 0]]) < 0.0
    outward[flip] = outward[flip][:, (0, 2, 1)]
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        deviation, admitted = source.certify(points[outward], budget=budget)
    assert np.all(admitted)
    native = TetMesh3D(
        points,
        cells,
        np.zeros((cells.shape[0],), dtype=np.int32),
        outward,
        np.zeros((faces.shape[0],), dtype=np.int64),
        np.zeros((0, 2), dtype=np.int32),
        np.zeros((0,), dtype=np.int64),
        boundary_policy="conforming",
        curved=TetMeshCurvedSources(
            np.zeros((1,), dtype=np.int64),
            np.asarray((tolerance,), dtype=np.float64),
            np.frombuffer(source.digest, dtype=np.uint8).reshape((1, 32)).copy(),
            deviation,
        ),
    )
    return native, source, source.digest


def test_curved_proof_cache_binds_current_source_and_exact_oriented_facets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, source, _ = _curved_cone(_CONE_TOLERANCE)
    snapshot = native.arrays()
    triangles = snapshot.points[snapshot.faces[:2]]
    authentications = 0
    authenticate = CurvedBoundarySource.validate_source_integrity

    def counted(current: CurvedBoundarySource) -> None:
        nonlocal authentications
        authentications += 1
        authenticate(current)

    monkeypatch.setattr(CurvedBoundarySource, "validate_source_integrity", counted)
    try:
        with _metric_request() as (scope, budget), budget.live_storage() as storage:
            tables = TetraMetricSource(
                snapshot.points,
                snapshot.tetrahedra,
                np.arange(snapshot.points.shape[0], dtype=np.int64),
                np.arange(snapshot.tetrahedra.shape[0], dtype=np.int64),
                budget=budget,
            )
            epoch = _NativeMetricBudget(
                native, _LIMITS.maximum_work_units, 10_000, tables,
                _LIMITS.maximum_geometry_queries, scope, budget,
                curved_storage=storage,
            )
            with epoch.phase():
                first = epoch.certify_curved(source, triangles)
                witnesses = epoch.refute_curved(source, triangles)
                assert first is not None and witnesses is not None
                completed = budget.work_units
                bound = storage.bound
                repeated = epoch.certify_curved(source, triangles)
                repeated_witnesses = epoch.refute_curved(source, triangles)
                assert repeated is not None and repeated_witnesses is not None
                np.testing.assert_array_equal(first[0], repeated[0])
                np.testing.assert_array_equal(first[1], repeated[1])
                np.testing.assert_array_equal(witnesses, repeated_witnesses)
                assert budget.work_units == completed
                assert storage.bound >= bound > 0
                cyclic = triangles[:, (1, 2, 0)]
                cyclic_bound = epoch.certify_curved(source, cyclic)
                cyclic_witnesses = epoch.refute_curved(source, cyclic)
                assert cyclic_bound is not None and cyclic_witnesses is not None
                np.testing.assert_array_equal(first[0], cyclic_bound[0])
                np.testing.assert_array_equal(witnesses, cyclic_witnesses)
                assert budget.work_units == completed
                assert len(epoch.curved_bounds) == 2
                # A changed corner or reversed orientation must issue its own
                # bound; neither can inherit an admitted facet's cached bool.
                changed = triangles.copy()
                changed[0, 0, 0] = np.nextafter(changed[0, 0, 0], np.inf)
                assert epoch.certify_curved(source, changed) is not None
                assert len(epoch.curved_bounds) == 3
                reversed_facets = triangles[:, (0, 2, 1)]
                reversed_claim = epoch.certify_curved(source, reversed_facets)
                assert reversed_claim is not None
                assert not np.any(reversed_claim[1])
                assert len(epoch.curved_bounds) == 5
                assert authentications == 1
                field_bounds = source.profile.field_bounds
                rounding = field_bounds.rounding_model
                object.__setattr__(field_bounds, "rounding_model", "changed")
                with pytest.raises(ValueError, match="changed source claims"):
                    epoch.certify_curved(source, triangles)
                object.__setattr__(field_bounds, "rounding_model", rounding)
                # A different genuine policy gets distinct issuance.
                tighter = CurvedBoundarySource(
                    source.profile, 1.0e-6, source.limits, source.facets, budget=budget
                )
                tightened = epoch.certify_curved(tighter, triangles)
                assert tightened is not None and not np.any(tightened[1])
                assert len(epoch.curved_bounds) == 7
                # Tampering with an existing policy cannot reuse its old key.
                object.__setattr__(source, "tolerance", 0.5)
                with pytest.raises(ValueError, match="changed source claims"):
                    epoch.certify_curved(source, triangles)
                with pytest.raises(ValueError, match="changed source claims"):
                    epoch.refute_curved(source, triangles)
                object.__setattr__(source, "tolerance", _CONE_TOLERANCE)
                radius = source.profile.radius
                object.__setattr__(source.profile, "radius", radius * 2.0)
                with pytest.raises(ValueError, match="changed source claims"):
                    epoch.certify_curved(source, triangles)
                object.__setattr__(source.profile, "radius", radius)
                center = source.profile.center.copy()
                center[0] = np.nextafter(center[0], np.inf)
                object.__setattr__(source.profile, "center", center)
                with pytest.raises(ValueError, match="changed source claims"):
                    epoch.certify_curved(source, triangles)
    finally:
        native.close()
    assert storage.bound == 0


def test_curved_collapse_commits_only_exact_current_whole_facet_claims() -> None:
    native, source, digest = _curved_cone(_CONE_TOLERANCE)
    snapshot = native.arrays()
    remove = int(snapshot.faces[0, 0])
    keep = int(next(v for v in snapshot.faces[0] if v != remove))
    plan = native.inspect_curved_collapse(remove, keep, maximum_faces=8)
    assert plan is not None and plan.accepted_generation == native.accepted_generation()
    triangles = snapshot.points[plan.faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0.0)
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        deviation, admitted = source.certify(triangles, budget=budget)
    assert np.all(admitted) and np.all(deviation <= _CONE_TOLERANCE)
    generation = native.accepted_generation()
    modified = triangles.copy()
    modified[0, 0, 0] = np.nextafter(modified[0, 0, 0], np.inf)
    refusals = (
        (triangles, deviation, bytes(32)),
        (modified, deviation, digest),
        (triangles, np.full_like(deviation, np.nextafter(_CONE_TOLERANCE, 1.0)), digest),
        (triangles, np.full_like(deviation, np.nan), digest),
        (triangles[:-1], deviation[:-1], digest),
    )
    for corners, claims, binding in refusals:
        if corners.shape[0] != plan.faces.shape[0]:
            with pytest.raises(ValueError, match="must align with the plan faces"):
                native.collapse_curved_edge(plan, corners, claims, digest=binding)
            continue
        assert not native.collapse_curved_edge(plan, corners, claims, digest=binding)
    assert native.accepted_generation() == generation
    np.testing.assert_array_equal(native.arrays().tetrahedra, snapshot.tetrahedra)
    initial_claims = native.curved_claims().rows.size
    assert native.collapse_curved_edge(plan, triangles, deviation, digest=digest)
    assert native.accepted_generation() > generation
    # The consumed plan is stale: its generation no longer validates.
    assert not native.collapse_curved_edge(plan, triangles, deviation, digest=digest)
    claims = native.curved_claims()
    assert claims.rows.size == initial_claims + plan.faces.shape[0]
    np.testing.assert_array_equal(claims.coordinates[-plan.faces.shape[0] :], triangles)
    np.testing.assert_array_equal(claims.deviations[-plan.faces.shape[0] :], deviation)
    live = {tuple(np.roll(row, -int(np.argmin(row)))) for row in native.arrays().faces.tolist()}
    assert all(tuple(np.roll(row, -int(np.argmin(row)))) in live for row in plan.faces.tolist())


def test_outward_python_claims_retain_immediate_and_successive_source_commits() -> None:
    retained, source, digest = _curved_cone(_CONE_TOLERANCE)
    fresh, fresh_source, fresh_digest = _curved_cone(_CONE_TOLERANCE, source=source)
    array_fields = (
        "points", "tetrahedra", "tetrahedron_regions", "faces", "face_sources",
        "segments", "segment_sources", "vertex_dimension", "vertex_sizes",
    )
    evidence_fields = (
        "face_ancestor_offsets", "face_ancestor_rows", "segment_ancestors",
        "witness_strata", "witness_entities", "witness_parameters", "witness_deviations",
        "refused_edges", "refused_strata", "refused_entities", "refused_deviations",
        "refused_bounds", "face_ids", "segment_ids",
    )
    claim_fields = ("rows", "source_ids", "coordinates", "deviations", "generations")

    def same_bits(left: object, right: object, names: tuple[str, ...]) -> None:
        for name in names:
            first, second = getattr(left, name), getattr(right, name)
            assert first.dtype == second.dtype and first.shape == second.shape
            assert first.tobytes() == second.tobytes(), name

    excluded: set[int] = set()
    try:
        assert digest == fresh_digest
        assert fresh_source is source
        for _ in range(2):
            snapshot, fresh_snapshot = retained.arrays(), fresh.arrays()
            same_bits(snapshot, fresh_snapshot, array_fields)
            edges = sorted(
                {
                    (int(removed), int(kept))
                    for face in snapshot.faces
                    for removed in face
                    for kept in face
                    if removed != kept and removed not in excluded and kept not in excluded
                }
            )
            committed = False
            for removed, kept in edges:
                inspection_start = retained.work_units()
                plan = retained.inspect_curved_collapse(removed, kept, maximum_faces=8)
                inspection_work = retained.work_units() - inspection_start
                fresh_plan = fresh.inspect_curved_collapse(removed, kept, maximum_faces=8)
                if plan is None:
                    assert fresh_plan is None
                    continue
                assert fresh_plan is not None
                assert plan.source == fresh_plan.source == 0
                assert plan.accepted_generation == fresh_plan.accepted_generation
                np.testing.assert_array_equal(plan.faces, fresh_plan.faces)
                triangles = snapshot.points[plan.faces]
                fresh_triangles = fresh_snapshot.points[fresh_plan.faces]
                normals = np.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                )
                assert np.all(np.einsum("ij,ij->i", normals, triangles.mean(axis=1)) > 0.0)
                with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
                    deviation, admitted = source.certify(triangles, budget=budget)
                with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
                    fresh_deviation, fresh_admitted = fresh_source.certify(
                        fresh_triangles, budget=budget
                    )
                np.testing.assert_array_equal(admitted, fresh_admitted)
                np.testing.assert_array_equal(deviation, fresh_deviation)
                if not np.all(admitted):
                    continue
                initial_claims = retained.curved_claims().rows.size
                generation = retained.accepted_generation()
                # This existing ordinary operation reaches native preparation,
                # fails without editing, and invalidates only the control owner.
                assert not fresh.collapse_edge(kept, kept)
                retained_start = retained.work_units()
                # No invalid claim is committed before this first valid call:
                # it must exercise the actual outward adapter's retained path.
                assert retained.collapse_curved_edge(plan, triangles, deviation, digest=digest)
                retained_work = retained.work_units() - retained_start
                fresh_start = fresh.work_units()
                assert fresh.collapse_curved_edge(
                    fresh_plan, fresh_triangles, fresh_deviation, digest=fresh_digest
                )
                assert fresh.work_units() > fresh_start
                assert retained_work > 0 and inspection_work > 0
                assert retained.accepted_generation() == fresh.accepted_generation() == generation + 1
                same_bits(retained.arrays(), fresh.arrays(), array_fields)
                retained_evidence, fresh_evidence = retained.source_evidence(), fresh.source_evidence()
                same_bits(retained_evidence, fresh_evidence, evidence_fields)
                assert retained_evidence.requested_bound == fresh_evidence.requested_bound
                assert retained_evidence.achieved_bound == fresh_evidence.achieved_bound
                claims, fresh_claims = retained.curved_claims(), fresh.curved_claims()
                same_bits(claims, fresh_claims, claim_fields)
                assert claims.rows.size == initial_claims + plan.faces.shape[0]
                assert claims.coordinates[-plan.faces.shape[0] :].tobytes() == triangles.tobytes()
                # Committed rows record the resulting accepted generation.
                assert np.all(
                    claims.generations[-plan.faces.shape[0] :] == retained.accepted_generation()
                )
                # Keep the next real sphere collapse away from this cavity so
                # its independently earned whole-facet proof stays local.
                support = np.any(np.isin(snapshot.faces, (removed, kept)), axis=1)
                excluded.update(int(value) for value in snapshot.faces[support].ravel())
                committed = True
                break
            assert committed, "Expected two independently certified same-source sphere collapses"
    finally:
        retained.close()
        fresh.close()


def test_curved_collapse_rejects_cavity_merit_before_querying_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax.meshing._tetra_metric as metric_owner

    native, source, _ = _curved_cone(_CONE_TOLERANCE)
    snapshot = native.arrays()
    generation = native.accepted_generation()
    considered: list[int] = []

    def reject_merit(*args: object, **kwargs: object) -> bool:
        considered.append(1)
        return False

    def unexpected_source(*args: object, **kwargs: object) -> None:
        raise AssertionError("A merit-rejected cavity must not query the source")

    monkeypatch.setattr(metric_owner, "_metric_cavity_merit", reject_merit)
    monkeypatch.setattr(metric_owner, "_refute_curved_collapse", unexpected_source)
    try:
        with _metric_request() as (scope, budget):
            tables = TetraMetricSource(
                snapshot.points,
                snapshot.tetrahedra,
                np.arange(snapshot.points.shape[0], dtype=np.int64),
                np.arange(snapshot.tetrahedra.shape[0], dtype=np.int64),
                budget=budget,
            )
            epoch = _NativeMetricBudget(
                native, _LIMITS.maximum_work_units, 10_000, tables,
                _LIMITS.maximum_geometry_queries, scope, budget,
            )
            with epoch.phase():
                result, applied, _, _ = metric_owner._collapse_pass(
                    native,
                    snapshot,
                    np.broadcast_to(np.eye(3), (snapshot.points.shape[0], 3, 3)),
                    np.zeros(snapshot.points.shape[0], dtype=np.bool_),
                    metric_owner._Ancestry({}, 0, []),
                    100,
                    0.0,
                    _LIMITS.maximum_cavity_cells,
                    epoch,
                    np.inf,
                    None,
                    metric_band=(0.0, np.inf),
                    curved=_CurvedAuthority(source, np.zeros(1, dtype=np.int64)),
                )
            assert considered
            assert applied == 0 and result is snapshot
            assert native.accepted_generation() == generation
            np.testing.assert_array_equal(native.arrays().tetrahedra, snapshot.tetrahedra)
    finally:
        native.close()


def test_legacy_meshes_offer_no_curved_collapse() -> None:
    points, cells, faces = _cone(1)
    # Without a curved declaration every face is its own planar source and
    # every face edge a segment.
    edges = np.unique(
        np.sort(faces[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2)), axis=1), axis=0
    )
    native = TetMesh3D(
        points,
        cells,
        np.zeros((cells.shape[0],), dtype=np.int32),
        faces,
        np.arange(faces.shape[0], dtype=np.int64),
        edges.astype(np.int32),
        np.arange(edges.shape[0], dtype=np.int64),
        boundary_policy="conforming",
    )
    snapshot = native.arrays()
    remove = int(snapshot.faces[0, 0])
    keep = int(next(v for v in snapshot.faces[0] if v != remove))
    assert native.inspect_curved_collapse(remove, keep, maximum_faces=8) is None
    assert native.curved_claims().rows.size == 0


def test_publication_refuses_claims_that_do_not_recertify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    native, source, digest = _curved_cone(_CONE_TOLERANCE)
    snapshot = native.arrays()
    remove = int(snapshot.faces[0, 0])
    keep = int(next(v for v in snapshot.faces[0] if v != remove))
    plan = native.inspect_curved_collapse(remove, keep, maximum_faces=8)
    assert plan is not None
    triangles = snapshot.points[plan.faces]
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        deviation, _ = source.certify(triangles, budget=budget)
    assert native.collapse_curved_edge(plan, triangles, deviation, digest=digest)
    labels = np.zeros((1,), dtype=np.int64)
    issued: list[np.ndarray] = []
    certify = CurvedBoundarySource.certify

    def track(
        self: CurvedBoundarySource,
        facets: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[np.ndarray, np.ndarray]:
        issued.append(facets.copy())
        return certify(self, facets, budget=budget)

    monkeypatch.setattr(CurvedBoundarySource, "certify", track)
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        _reprove_curved_publication(
            native, native.arrays(), _CurvedAuthority(source, labels), budget=budget
        )
    current = native.arrays()
    all_facets = np.concatenate(
        (native.curved_claims().coordinates, current.points[current.faces])
    )
    expected = {
        min(facet.tobytes(), facet[(1, 2, 0), :].tobytes(), facet[(2, 0, 1), :].tobytes())
        for facet in all_facets
    }
    assert len(issued) == 1 and issued[0].shape[0] == len(expected)
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        tighter = CurvedBoundarySource(
            source.profile, float(np.min(deviation)) / 2.0, source.limits, source.facets,
            budget=budget,
        )
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget, pytest.raises(
        ValueError, match="must recertify from the original source"
    ):
        _reprove_curved_publication(
            native, native.arrays(), _CurvedAuthority(tighter, labels), budget=budget
        )
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget, pytest.raises(
        ValueError, match="must belong to the declared curved sources"
    ):
        _reprove_curved_publication(
            native,
            native.arrays(),
            _CurvedAuthority(source, np.ones((1,), dtype=np.int64)),
            budget=budget,
        )




def _adapt(
    mesh: CellMesh, size: float, curved: CurvedBoundarySource | None
) -> tuple[CellMesh, MeshLineage, TetraMetricOutcome]:
    metric = np.broadcast_to(np.eye(3) / size**2, (mesh.coordinates.shape[0], 3, 3))
    with _metric_request() as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            mesh,
            metric,
            maximum_passes=8,
            relocation=False,
            curved_boundary=curved,
            scope=scope,
            budget=budget,
        )
    with coordinate_reference_budget() as budget:
        target, lineage, _ = assemble_topology_edit(
            mesh, outcome.edit, numeric_version="curved", budget=budget
        )
    return target, lineage, outcome


def test_curved_metric_coarsens_sphere_boundary_under_whole_facet_proofs() -> None:
    source = _ball(2)
    tolerance = 0.02
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        curved = CurvedBoundarySource(
            _profile(), tolerance, MeshCertificateLimits(), _boundary_rows(source),
            budget=budget,
        )
    frozen, _, frozen_outcome = _adapt(source, 0.25, None)
    target, lineage, outcome = _adapt(source, 0.25, curved)
    assert outcome.evidence.status is not MetricRemeshingStatus.RESOURCE_LIMIT
    before = _outward_boundary(source)
    after = _outward_boundary(target)
    # The planar route keeps every boundary triangle; the curved route
    # genuinely coarsens the smooth boundary.
    assert _outward_boundary(frozen).shape[0] == before.shape[0]
    assert after.shape[0] < before.shape[0] // 2
    assert outcome.evidence.collapses > frozen_outcome.evidence.collapses
    points = np.asarray(target.coordinates)
    deviation, oriented, _ = implicit_facet_projection_bounds(
        points[after], _profile(), tolerance, MeshCertificateLimits()
    )
    assert np.all(oriented) and np.all(deviation <= tolerance)
    # Closed oriented boundary: every directed edge has exactly one opposite.
    directed = after[:, ((0, 1), (1, 2), (2, 0))].reshape((-1, 2))
    forward = {tuple(row) for row in directed.tolist()}
    assert len(forward) == directed.shape[0]
    assert all((second, first) in forward for first, second in forward)
    corners = points[np.asarray(target.blocks[0].vertices)]
    assert np.all(np.linalg.det(corners[:, 1:] - corners[:, :1]) > 0.0)
    record = lineage.entity_lineage(2)
    target_connectivity = target.connectivity
    source_connectivity = source.connectivity
    assert isinstance(target_connectivity, phx.discretization.TetrahedralConnectivity)
    assert isinstance(source_connectivity, phx.discretization.TetrahedralConnectivity)
    ids = np.asarray(target.entity_set(2).entity_ids)[
        np.asarray(target_connectivity.boundary_faces)
    ]
    old_ids = np.asarray(source.entity_set(2).entity_ids)[
        np.asarray(source_connectivity.boundary_faces)
    ]
    relevant = np.isin(np.asarray(record.target_global_ids), ids)
    assert not np.intersect1d(np.asarray(record.created_target_ids), ids).size
    assert np.all(np.isin(np.asarray(record.source_global_ids)[relevant], old_ids))
    # Every new boundary face descends from old boundary faces; the surviving
    # old boundary faces carry only their identity.
    targets = np.asarray(record.target_global_ids)
    preserved = np.asarray(record.relation_kinds) == int(EntityLineageKind.PRESERVED)
    np.testing.assert_array_equal(np.unique(targets[relevant]), np.unique(ids))
    retained = np.isin(targets, old_ids)
    assert not np.any(relevant & preserved & ~retained)
    assert not np.any(relevant & retained & ~preserved)
    np.testing.assert_array_equal(
        np.asarray(record.source_global_ids)[relevant & preserved], targets[relevant & preserved]
    )
    assert int(EntityLineageKind.COLLAPSED_INTO) in np.asarray(record.relation_kinds)[relevant]


def test_uncertified_initial_faces_keep_planar_constraints() -> None:
    source = _ball(1)
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        curved = CurvedBoundarySource(
            _profile(), 1.0e-6, MeshCertificateLimits(), _boundary_rows(source),
            budget=budget,
        )
    target, _, outcome = _adapt(source, 0.25, curved)
    assert _outward_boundary(target).shape[0] == _outward_boundary(source).shape[0]
    np.testing.assert_array_equal(
        np.sort(np.asarray(target.coordinates)[np.unique(_outward_boundary(target))], axis=0),
        np.sort(np.asarray(source.coordinates)[np.unique(_outward_boundary(source))], axis=0),
    )
    assert outcome.evidence.status is not MetricRemeshingStatus.RESOURCE_LIMIT


def _exact(point: np.ndarray) -> tuple[Fraction, ...]:
    return tuple(Fraction(float(value)) for value in point)


def _on_segment(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> bool:
    a, b, p = _exact(first), _exact(second), _exact(point)
    d = [b[i] - a[i] for i in range(3)]
    e = [p[i] - a[i] for i in range(3)]
    cross = (d[1] * e[2] - d[2] * e[1], d[2] * e[0] - d[0] * e[2], d[0] * e[1] - d[1] * e[0])
    along = sum(d[i] * e[i] for i in range(3))
    return all(value == 0 for value in cross) and 0 <= along <= sum(v * v for v in d)


def _oblique_box() -> CellMesh:
    points, cells = kuhn_cube_arrays(2, jitter=0.17)
    rotation = np.linalg.qr(np.asarray(((0.9, 0.3, 0.1), (-0.2, 0.8, 0.5), (0.4, -0.1, 0.7))))[0]
    points = points @ rotation.T + np.asarray((0.1234567, -0.3771, 0.5512))
    corners = points[cells]
    flip = np.linalg.det(corners[:, 1:] - corners[:, :1]) < 0.0
    cells[flip] = cells[flip][:, (0, 2, 1, 3)]
    return CellMesh(points, (CellBlock("box", "tetrahedron", cells),))


def test_off_edge_insertions_claim_only_exact_nested_lineage() -> None:
    source = _oblique_box()
    target, lineage, outcome = _adapt(source, 0.18, None)
    assert outcome.evidence.splits > 0
    old_points = np.asarray(source.coordinates)
    old_ids = {int(v): row for row, v in enumerate(np.asarray(source.vertex_global_ids))}
    new_points = np.asarray(target.coordinates)
    new_ids = {int(v): row for row, v in enumerate(np.asarray(target.vertex_global_ids))}
    source_connectivity = source.connectivity
    assert isinstance(source_connectivity, phx.discretization.TetrahedralConnectivity)
    old_edges = np.asarray(source_connectivity.edges)
    # Some accepted insertion is a rounded carrier off every source edge.
    created = [row for identifier, row in new_ids.items() if identifier not in old_ids]
    assert any(
        not any(
            _on_segment(new_points[row], old_points[a], old_points[b])
            for a, b in old_edges.tolist()
        )
        for row in created
    )
    # Every claimed nested edge refinement is exact: the child lies on its parent.
    record = lineage.entity_lineage(1)
    sources = np.asarray(record.source_global_ids)
    targets = np.asarray(record.target_global_ids)
    kinds = np.asarray(record.relation_kinds)
    source_keys = {
        int(i): tuple(int(v) for v in row)
        for i, row in zip(
            np.asarray(source.entity_set(1).entity_ids),
            np.asarray(source.vertex_global_ids)[old_edges],
            strict=True,
        )
    }
    target_connectivity = target.connectivity
    assert isinstance(target_connectivity, phx.discretization.TetrahedralConnectivity)
    target_edges = np.asarray(target_connectivity.edges)
    target_keys = {
        int(i): tuple(int(v) for v in row)
        for i, row in zip(
            np.asarray(target.entity_set(1).entity_ids),
            np.asarray(target.vertex_global_ids)[target_edges],
            strict=True,
        )
    }
    refined = kinds == int(EntityLineageKind.REFINED_FROM)
    assert np.any(refined)
    for parent, child in zip(sources[refined].tolist(), targets[refined].tolist(), strict=True):
        a, b = (old_points[old_ids[v]] for v in source_keys[parent])
        for vertex in target_keys[child]:
            assert _on_segment(new_points[new_ids[vertex]], a, b)
    # Conservative coverage: the cell union and affine fields are unchanged.
    corners = new_points[np.asarray(target.blocks[0].vertices)]
    volumes = np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    old_corners = old_points[np.asarray(source.blocks[0].vertices)]
    assert np.all(volumes > 0.0)
    assert np.sum(volumes) == pytest.approx(
        np.sum(np.linalg.det(old_corners[:, 1:] - old_corners[:, :1]) / 6.0),
        abs=TETRA_GEOMETRY_TOLERANCE,
    )
    stencil_sources = np.asarray(outcome.edit.stencil_sources)
    weights = np.asarray(outcome.edit.stencil_weights)
    reproduced = np.sum(
        old_points[np.vectorize(old_ids.__getitem__)(stencil_sources)] * weights[..., None],
        axis=1,
    )
    np.testing.assert_allclose(reproduced, new_points, atol=TETRA_GEOMETRY_TOLERANCE)


def test_insertion_support_is_the_replaced_cavity_off_the_edge() -> None:
    from phydrax.meshing._tetra_metric import _Ancestry

    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (1.0, 1.0, 1.0)),
        dtype=np.float64,
    )
    before = np.asarray(((0, 1, 2, 3), (1, 2, 3, 4)), dtype=np.int32)
    after = np.asarray(((0, 1, 2, 5), (0, 1, 5, 3), (1, 2, 3, 4)), dtype=np.int32)
    ancestry = _Ancestry({}, 0, [{10}, {11}, {12}, {13}, {14}])
    on_edge = np.asarray((0.5, 0.0, 0.0), dtype=np.float64)
    assert _insertion_support(ancestry, before, after, points, (0, 1), on_edge) == {10, 11}
    off_edge = np.asarray((0.5, np.nextafter(0.0, 1.0), 0.0), dtype=np.float64)
    assert _insertion_support(ancestry, before, after, points, (0, 1), off_edge) == {
        10, 11, 12, 13,
    }


@pytest.mark.parametrize("dtype", (np.int32, np.int64))
@pytest.mark.parametrize("reverse_rows", (False, True))
def test_inspected_split_cavity_lineage_matches_full_mesh_reference(
    dtype: type[np.int32] | type[np.int64], reverse_rows: bool,
) -> None:
    from phydrax.meshing._tetra_metric import (
        _Ancestry,
        _record,
    )

    before = np.asarray(
        ((0, 1, 2, 3), (1, 2, 3, 4), (0, 2, 3, 4)), dtype=dtype
    )
    after = np.asarray(
        ((3, 4, 1, 2), (0, 1, 2, 5), (0, 1, 5, 3), (0, 2, 5, 4)),
        dtype=dtype,
    )
    if reverse_rows:
        after = after[::-1]
    removed = before[[0, 2]]
    preserved = np.all(np.sort(after, axis=1) == np.asarray((1, 2, 3, 4)), axis=1)
    proposed = after[~preserved]
    np.testing.assert_array_equal(removed, before[[0, 2]])
    assert {tuple(sorted(row)) for row in proposed.tolist()} == {
        tuple(sorted(row)) for row in after.tolist()
        if tuple(sorted(row)) != (1, 2, 3, 4)
    }
    ancestry = _Ancestry(
        {
            tuple(sorted(row)): (10 + i, {100 + i})
            for i, row in enumerate(before.tolist())
        },
        20,
        [{10}, {11}, {12}, {13}, {14}],
        cell_kinds={
            10: int(EntityLineageKind.REFINED_FROM),
            11: int(EntityLineageKind.PRESERVED),
            12: int(EntityLineageKind.COLLAPSED_INTO),
        },
    )
    full, local = deepcopy(ancestry), deepcopy(ancestry)
    _record(full, before, after, EntityLineageKind.REFINED_FROM)
    _record(local, removed, proposed, EntityLineageKind.REFINED_FROM)
    assert local.cells == full.cells
    assert local.next_cell_id == full.next_cell_id
    assert local.cell_kinds == full.cell_kinds
    points = np.asarray(
        ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.), (1., 1., 1.))
    )
    for position in (
        np.asarray((0.5, 0.0, 0.0)),
        np.asarray((0.5, np.nextafter(0.0, 1.0), 0.0)),
    ):
        assert _insertion_support(
            ancestry, removed, proposed, points, (0, 1), position
        ) == _insertion_support(
            ancestry, before, after, points, (0, 1), position
        )


@pytest.mark.parametrize(
    ("half_span", "height", "floor", "accepted"),
    ((0.75, 0.05, 0.0, False), (1.0, 1.0, 0.0, True), (1.0, 1.0, 1.0, False)),
    ids=("band-refusal", "band-descent", "quality-refusal"),
)
def test_ordinary_split_inspects_complete_merit_and_commits_only_its_issued_plan(
    monkeypatch: pytest.MonkeyPatch,
    half_span: float, height: float, floor: float, accepted: bool,
) -> None:
    import phydrax.meshing._tetra_metric as owner

    points = np.asarray(
        ((-half_span, 0., 0.), (half_span, 0., 0.), (0., height, 0.), (0., 0., height))
    )
    cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
    faces = cells[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    segments = owner._edges(cells).astype(np.int32)
    native = TetMesh3D(
        points, cells, np.zeros(1, dtype=np.int32),
        faces, np.arange(4, dtype=np.int64), segments, np.arange(6, dtype=np.int64),
        boundary_policy="conforming",
    )
    inspected, committed, sampled, deltas = [], [], [], []
    inspect = TetMesh3D.inspect_edge_insertion
    commit = TetMesh3D.commit_edge_insertion
    sample = owner._sample_metric_splits
    merit = owner._metric_relocation_merit

    def inspect_once(
        self: TetMesh3D, first: int, second: int,
        split_position: object, final_position: object, /, *,
        maximum_cavity_cells: int, work_limit: int | None = None,
        source_fraction: float | None = None,
    ) -> TetMeshInsertionPlan | None:
        plan = inspect(
            self, first, second, split_position, final_position,
            maximum_cavity_cells=maximum_cavity_cells, work_limit=work_limit,
            source_fraction=source_fraction,
        )
        if plan is not None:
            inspected.append(plan)
        return plan

    def commit_issued(
        self: TetMesh3D, plan: TetMeshInsertionPlan, /, *,
        target_size: float = 0.0, work_limit: int | None = None,
        claim_digest: bytes | None = None, face_coordinates: object | None = None,
        face_deviations: object | None = None,
    ) -> int | None:
        assert any(plan is issued for issued in inspected)
        # An ordinary split carries no projected curved-source claims.
        assert plan.projected_source < 0
        assert claim_digest is None and face_coordinates is None and face_deviations is None
        committed.append(plan)
        return commit(self, plan, target_size=target_size, work_limit=work_limit)

    def sample_once(
        points: np.ndarray, constructions: tuple[TetMeshEdgeConstruction, ...],
        source_tables: TetraMetricSource, source_metric: np.ndarray,
        native_budget: _NativeMetricBudget,
    ) -> _SplitSamples | None:
        sampled.append(points.copy())
        return sample(points, constructions, source_tables, source_metric, native_budget)

    def complete_delta(
        old: np.ndarray, new: np.ndarray, old_quality: float, new_quality: float,
        minimum_quality: float, metric_band: tuple[float, float], /, *,
        refinement: bool = False,
        required_long_split: bool = False,
    ) -> bool:
        deltas.append((old.copy(), new.copy()))
        return merit(
            old, new, old_quality, new_quality, minimum_quality, metric_band,
            refinement=refinement,
            required_long_split=required_long_split,
        )

    monkeypatch.setattr(TetMesh3D, "inspect_edge_insertion", inspect_once)
    monkeypatch.setattr(TetMesh3D, "commit_edge_insertion", commit_issued)
    monkeypatch.setattr(owner, "_sample_metric_splits", sample_once)
    monkeypatch.setattr(owner, "_metric_relocation_merit", complete_delta)
    snapshot = native.arrays()
    generation = native.accepted_generation()
    values = np.broadcast_to(np.eye(3), (4, 3, 3)).copy()
    ancestry = owner._Ancestry({(0, 1, 2, 3): (0, {0})}, 1, [{v} for v in range(4)])
    before_ancestry = deepcopy(ancestry)
    try:
        with _metric_request(mesh=native) as (scope, budget):
            tables = TetraMetricSource(
                points, cells, np.arange(4, dtype=np.int64), np.arange(1, dtype=np.int64),
                budget=budget,
            )
            epoch = _NativeMetricBudget(
                native, _LIMITS.maximum_work_units, 10_000, tables,
                _LIMITS.maximum_geometry_queries, scope, budget,
                maximum_cavity_cells=512,
            )
            with epoch.phase():
                result = owner._split_batch(
                    native, snapshot, values, tables, values, ancestry, set(),
                    4, 100, 100, 512, epoch, np.sqrt(2.0), None,
                    owner._SplitCandidates(
                        np.asarray(((0, 1),), dtype=np.int32),
                        np.asarray((2 * half_span,)), np.asarray((0,)), False,
                    ),
                    minimum_quality=floor,
                    metric_band=(1 / np.sqrt(2.0), np.sqrt(2.0)),
                )
                assert not epoch.exhausted, (
                    epoch.resource_message, epoch.resource_requested, epoch.resource_achieved
                )
            target, metric, changes, attempts, refusals, exhausted, _, _ = result
            assert changes == int(accepted) and attempts == 1
            assert refusals == int(not accepted) and not exhausted
            assert len(sampled) == len(inspected) == 1
            assert len(committed) == int(accepted)
            assert native.accepted_generation() == generation + int(accepted)
            if accepted:
                assert committed[0] is inspected[0]
                assert target.points.shape[0] == metric.shape[0] == 5
                assert np.all(owner._positive(target.points, target.tetrahedra))
                np.testing.assert_array_equal(metric, np.broadcast_to(np.eye(3), (5, 3, 3)))
            else:
                assert target is snapshot and metric is values
                np.testing.assert_array_equal(native.arrays().points, snapshot.points)
                np.testing.assert_array_equal(native.arrays().tetrahedra, snapshot.tetrahedra)
                assert ancestry == before_ancestry
            if floor == 0:
                assert len(deltas) == 1
                # Unchanged boundary edges cancel even if their incidence
                # changes; new edges shared by two children occur only once.
                np.testing.assert_allclose(deltas[0][0], (2 * half_span,))
                np.testing.assert_allclose(
                    np.sort(deltas[0][1]), np.sort((half_span, half_span, height, height))
                )
    finally:
        native.close()


@pytest.mark.parametrize(
    ("old", "new", "old_quality", "new_quality", "floor", "admitted"),
    (
        ((8.66,), (4.33,) * 6, 0.5, 0.5, 0.0, True),
        ((2.1,), (0.25, 1.5, 1.5), 0.5, 0.5, 0.0, False),
        ((4.0, 3.0), (4.0,), 0.5, 0.5, 0.0, True),
        ((4.0,), (4.0, 3.0), 0.5, 0.5, 0.0, False),
        ((4.0, 3.0), (4.0, 2.5), 0.5, 0.5, 0.0, True),
        ((4.0, 2.5), (4.0, 3.0), 0.5, 0.5, 0.0, False),
        ((4.0,), (4.0, 1.5, 1.5), 0.5, 0.5, 0.0, False),
        ((4.0,), (4.0, 1.5), 0.5, 0.5000005, 0.0, False),
        ((4.0,), (4.0, 1.5), 0.5, 0.500002, 0.0, True),
        ((np.inf,), (2.5,), 0.5, 0.5, 0.0, False),
        ((3.0,), (np.nan,), 0.5, 0.5, 0.0, False),
        ((3.0,), (2.5,), np.nan, 0.5, 0.0, False),
        ((3.0,), (2.5,), 0.5, np.nan, 0.0, False),
        ((3.0,), (2.5,), 0.3, 0.4, 0.5, False),
        ((3.0,), (4.0,), 0.3, 0.6, 0.5, True),
    ),
)
def test_split_lexicographic_positive_band_multiset_keeps_multiplicity_and_ties(
    old: tuple[float, ...], new: tuple[float, ...],
    old_quality: float, new_quality: float, floor: float, admitted: bool,
) -> None:
    from phydrax.meshing._tetra_metric import _metric_relocation_merit

    assert bool(_metric_relocation_merit(
        np.asarray(old), np.asarray(new), old_quality, new_quality, floor, (1.0, 2.0),
        refinement=True,
    )) is admitted


@pytest.mark.parametrize(
    ("size", "floor", "complete"),
    ((_RADIUS, 0.05, True), (1.0, 0.05, False), (0.1, 0.05, False),
     (_RADIUS, 1.0, False)),
)
def test_source_owned_lower_bound_keeps_interior_upper_and_quality_obligations(
    size: float, floor: float, complete: bool,
) -> None:
    points, cells, _ = _cone(1)
    mesh = CellMesh(points, (CellBlock("cone", "tetrahedron", cells),))
    with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
        curved = CurvedBoundarySource(
            _profile(), _CONE_TOLERANCE, MeshCertificateLimits(), _boundary_rows(mesh),
            budget=budget,
        )
    with _metric_request() as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            mesh, np.broadcast_to(np.eye(3) / size**2, (len(points), 3, 3)),
            maximum_passes=0, minimum_metric_quality=floor,
            curved_boundary=curved, scope=scope, budget=budget,
        )
    evidence = outcome.evidence
    assert evidence.converged is complete
    assert evidence.out_of_range_edges > 0
    assert evidence.unmet_metric_edges == (
        evidence.out_of_range_edges - len(evidence.source_lower_exemptions)
    )
    if complete:
        assert evidence.unmet_metric_edges == 0
        assert len(evidence.source_lower_exemptions) == 120
        assert evidence.source_lower_binding is not None
        assert evidence.maximum_fidelity_bound <= _CONE_TOLERANCE
    if size == 1.0:
        assert evidence.unmet_metric_edges == 42
    if size == 0.1:
        assert evidence.source_lower_exemptions == ()
        assert evidence.unmet_metric_edges > 0
    evidence.validate_restored()
    changed = deepcopy(evidence)
    object.__setattr__(changed, "source_lower_binding", "0" * 64)
    with pytest.raises(ValueError):
        changed.validate_restored()


@pytest.mark.parametrize(
    "fault", ("none", "missing", "partial", "foreign", "duplicate", "fidelity", "resource")
)
def test_source_lower_exemption_authenticates_current_support(fault: str) -> None:
    import phydrax.meshing._tetra_metric as owner

    native, source, _ = _curved_cone(_CONE_TOLERANCE)
    try:
        snapshot = native.arrays()
        if fault == "duplicate":
            snapshot.faces[0] = snapshot.faces[1]
        if fault == "partial":
            snapshot.face_sources[0] = 1
        if fault == "fidelity":
            with coordinate_reference_budget(queries_upper=_PROOF_QUERIES) as budget:
                source = CurvedBoundarySource(
                    source.profile, 1.0e-6, source.limits, source.facets, budget=budget
                )
        authority = (
            None if fault == "missing" else
            _CurvedAuthority(source, np.asarray((1 if fault == "foreign" else 0,)))
        )
        with _metric_request() as (scope, budget), budget.live_storage() as storage:
            tables = TetraMetricSource(
                snapshot.points, snapshot.tetrahedra,
                np.arange(len(snapshot.points), dtype=np.int64),
                np.arange(len(snapshot.tetrahedra), dtype=np.int64), budget=budget,
            )
            epoch = _NativeMetricBudget(
                native, _LIMITS.maximum_work_units, 10_000, tables,
                _LIMITS.maximum_geometry_queries, scope, budget,
                curved_storage=storage,
            )
            if fault == "resource":
                epoch.maximum_work_units = 0
            edges = owner._edges(snapshot.tetrahedra)
            lengths = owner._lengths(
                snapshot.points,
                np.broadcast_to(np.eye(3) / _RADIUS**2, (len(snapshot.points), 3, 3)),
                edges,
            )
            with epoch.phase():
                rows, binding, fidelity, valid = owner._source_owned_short_edges(
                    snapshot, authority, epoch, edges, lengths, 1.0 / np.sqrt(2.0)
                )
        if fault == "none":
            assert valid and len(rows) == 120 and binding is not None
            assert fidelity <= _CONE_TOLERANCE
        elif fault == "partial":
            assert valid and len(rows) == 117 and binding is not None
        else:
            assert rows == () and binding is None
            assert valid is (fault == "missing")
    finally:
        native.close()


@pytest.mark.parametrize("secondary", (0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5))
def test_required_interior_split_prioritizes_strict_upper_sizing_descent(
    secondary: float,
) -> None:
    from phydrax.meshing._tetra_metric import _metric_relocation_merit

    old = np.asarray((1.417565256615133,))
    new = np.asarray((0.7087826283075665, 0.7087826283075665, secondary))
    band = (1.0 / np.sqrt(2.0), np.sqrt(2.0))
    assert not _metric_relocation_merit(old, new, 0.4, 0.3, 0.05, band, refinement=True)
    assert _metric_relocation_merit(
        old, new, 0.4, 0.3, 0.05, band,
        refinement=True, required_long_split=True,
    )
    assert not _metric_relocation_merit(
        old, new, 0.4, 0.049, 0.05, band,
        refinement=True, required_long_split=True,
    )


@pytest.mark.parametrize(
    ("old", "new", "accepted"),
    (((1.5,), (1.5, 0.8), False), ((1.5,), (1.6,), False),
     ((1.0,), (0.9,), False), ((1.5,), (1.4, 0.2, 0.2), True),
     ((1.5,), (1.4, 0.2), True), ((1.5,), (np.nan,), False)),
)
def test_required_interior_split_needs_primary_improvement_even_with_shape_gain(
    old: tuple[float, ...], new: tuple[float, ...], accepted: bool,
) -> None:
    from phydrax.meshing._tetra_metric import _metric_relocation_merit

    assert _metric_relocation_merit(
        np.asarray(old), np.asarray(new), 0.2, 0.8, 0.05,
        (1.0 / np.sqrt(2.0), np.sqrt(2.0)),
        refinement=True, required_long_split=True,
    ) is accepted


def test_multiple_required_interior_splits_complete_without_changing_operation_order() -> None:
    ring = np.asarray(((0., 0.72, 0.), (0., -0.36, 0.72 * np.sqrt(3.) / 2.),
                       (0., -0.36, -0.72 * np.sqrt(3.) / 2.)))
    first = np.vstack(((-0.71, 0., 0.), (0.71, 0., 0.), ring))
    points = np.vstack((first, first + (3., 0., 0.)))
    cells = np.asarray(((0, 1, 2, 3), (0, 1, 3, 4), (0, 1, 4, 2)), dtype=np.int32)
    cells = np.vstack((cells, cells + 5))
    determinants = np.linalg.det(points[cells[:, 1:]] - points[cells[:, :1]])
    cells[determinants < 0] = cells[determinants < 0][:, (0, 2, 1, 3)]
    mesh = CellMesh(points, (CellBlock("bipyramids", "tetrahedron", cells),))
    with _metric_request() as (scope, budget):
        outcome = execute_tetra_metric_adaptation(
            mesh, np.broadcast_to(np.eye(3), (len(points), 3, 3)),
            maximum_passes=2, relocation=False, scope=scope, budget=budget,
        )
    assert outcome.evidence.status is MetricRemeshingStatus.COMPLETE
    assert outcome.evidence.splits == 2
    assert outcome.evidence.unmet_metric_edges == 0
    assert outcome.evidence.minimum_metric_quality >= 0.05


def _native_material_metric_cavity(
    height: Fraction | None,
) -> tuple[TetMesh3D, CavitySnapshot]:
    """A flat-rim tetrahedral cavity, optionally lifted by one interior apex."""
    points = np.asarray(((0, 0, -1), (0, 0, 0), (1, 0, 0), (0, 1, 0)), dtype=np.float64)
    if height is None:
        cells = np.asarray(((0, 1, 2, 3),), dtype=np.int32)
        patch = ((1, 2, 3),)
    else:
        points = np.vstack((points, (0.25, 0.25, float(height))))
        cells = np.asarray(((0, 1, 2, 4), (0, 2, 3, 4), (0, 3, 1, 4)), dtype=np.int32)
        patch = ((1, 2, 4), (2, 3, 4), (3, 1, 4))
    mesh = CellMesh(points, (CellBlock("material-cavity", "tetrahedron", cells),))
    boundary = _outward_boundary(mesh)
    segments = np.unique(
        np.sort(boundary[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2), axis=1),
        axis=0,
    ).astype(np.int32)
    native = TetMesh3D(
        points, cells, np.zeros(len(cells), dtype=np.int32),
        boundary, np.arange(len(boundary), dtype=np.int64),
        segments, np.arange(len(segments), dtype=np.int64),
        boundary_policy="conforming",
    )
    actual = native.arrays()
    snapshot = CavitySnapshot(
        tuple(
            (Fraction(float(x)), Fraction(float(y)), Fraction(float(z)))
            for x, y, z in actual.points
        ),
        tuple((int(a), int(b), int(c), int(d)) for a, b, c, d in actual.tetrahedra),
        patch, ("material-top",) * len(patch), tuple(range(len(actual.tetrahedra))),
        ((1, 2, "rim-ab"), (2, 3, "rim-bc"), (3, 1, "rim-ca")),
    )
    return native, snapshot


def _log_affine_material_metric(points: np.ndarray) -> np.ndarray:
    """Diagonal SPD field whose log-Euclidean interpolation is exactly affine."""
    result = np.zeros((len(points), 3, 3))
    for axis in range(3):
        result[:, axis, axis] = np.exp(axis + 1.0 + points[:, axis])
    return result


def test_runtime_metric_stencils_follow_live_chronological_cavity_history() -> None:
    """Split and relocation lookup must pull current points back, not extrapolate."""
    import phydrax.meshing._tetra_metric as owner

    source_native, source_cavity = _native_material_metric_cavity(None)
    expanded_native, expanded_cavity = _native_material_metric_cavity(Fraction(1, 2))
    contracted_native, contracted_cavity = _native_material_metric_cavity(Fraction(1, 4))
    try:
        source = source_native.arrays()
        expanded = expanded_native.arrays()
        contracted = contracted_native.arrays()
        source_metric = _log_affine_material_metric(source.points)
        original_metric = source_metric.copy()
        original_points = source.points.copy()
        assert np.all(np.linalg.eigvalsh(source_metric) > 0.0)
        construction = contracted_native.edge_split_point(1, 4)
        assert construction is not None
        split_points = construction.position[None, :]
        np.testing.assert_array_equal(split_points, ((0.125, 0.125, 0.125),))
        # Both sites lie above z=0, outside the immutable original carrier.
        relocation_points = np.asarray(((0.25, 0.25, 0.125),))
        ancestry = owner._Ancestry(
            {(0, 1, 2, 3): (0, {0})}, 1, [{index} for index in range(4)],
        )
        with _metric_request() as (scope, budget):
            tables = TetraMetricSource(
                source.points, source.tetrahedra,
                np.arange(4, dtype=np.int64), np.arange(1, dtype=np.int64),
                budget=budget,
            )
            epoch = _NativeMetricBudget(
                source_native, _LIMITS.maximum_work_units, 10_000, tables,
                _LIMITS.maximum_geometry_queries, scope, budget,
                cavity_history=ancestry.cavity_history,
            )
            with epoch.phase():
                outside = epoch.stencil(relocation_points)
                assert outside is not None
                assert not np.any(outside[2])
                first = build_implicit_cavity_atlas(
                    prepare_implicit_cavity_edit(
                        source_cavity, expanded_cavity,
                        source_id="metric-material-source",
                        discovery_id="metric-material-discovery",
                        chronology_index=0, predecessor_id="metric-material-root",
                        budget=budget,
                    ),
                    chart_axes=(0, 1),
                    anchor=(Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)),
                    budget=budget,
                )
                ancestry.cavity_history.append(owner.CurvedCavityTransition(
                    first, source, expanded,
                    np.arange(1, dtype=np.int64), np.arange(3, dtype=np.int64),
                ))
                epoch.native = expanded_native
                assert epoch.cavity_history is ancestry.cavity_history
                first_bvh = epoch.source_bvh
                for expected_z, split_z in ((-0.125, -1.0 / 24.0), (-0.0625, 0.0)):
                    stencil = epoch.stencil(relocation_points)
                    assert stencil is not None
                    indices, weights, valid = stencil
                    assert np.all(valid)
                    expected_material = np.asarray(((0.25, 0.25, expected_z),))
                    np.testing.assert_allclose(
                        np.einsum("nk,nkj->nj", weights, source.points[indices]),
                        expected_material, rtol=0.0, atol=TETRA_GEOMETRY_TOLERANCE,
                    )
                    np.testing.assert_allclose(
                        owner._interpolate(source_metric[indices], weights),
                        _log_affine_material_metric(expected_material),
                        rtol=0.0, atol=TETRA_GEOMETRY_TOLERANCE,
                    )
                    samples = owner._sample_metric_splits(
                        split_points, (construction,), tables, source_metric, epoch,
                    )
                    assert samples is not None
                    assert np.all(samples.valid)
                    np.testing.assert_allclose(
                        samples.metric,
                        _log_affine_material_metric(np.asarray(((0.125, 0.125, split_z),))),
                        rtol=0.0, atol=TETRA_GEOMETRY_TOLERANCE,
                    )
                    assert epoch.source_bvh is first_bvh
                    if len(ancestry.cavity_history) == 1:
                        second = build_implicit_cavity_atlas(
                            prepare_implicit_cavity_edit(
                                expanded_cavity, contracted_cavity,
                                source_id=first.source_id, discovery_id=first.discovery_id,
                                chronology_index=1, predecessor_id=first.chart_id,
                                budget=budget,
                            ),
                            chart_axes=(0, 1),
                            anchor=(Fraction(1, 4), Fraction(1, 4), Fraction(-1, 4)),
                            budget=budget,
                        )
                        ancestry.cavity_history.append(owner.CurvedCavityTransition(
                            second, expanded, contracted,
                            np.arange(3, dtype=np.int64), np.arange(3, dtype=np.int64),
                        ))
                        epoch.native = contracted_native
                removed = np.asarray(((0.25, 0.25, 0.375),))
                rejected = epoch.stencil(removed)
                assert rejected is not None
                assert not np.any(rejected[2])
                with pytest.raises(phx.meshing.MeshingFailure, match="complete inverse material history"):
                    owner._material_source_stencil(
                        tables, removed, ancestry.cavity_history,
                        _LIMITS.maximum_geometry_queries, budget=budget,
                    )
                np.testing.assert_array_equal(source_metric, original_metric)
                np.testing.assert_array_equal(tables.points, original_points)
                assert len(ancestry.cavity_history) == 2
    finally:
        contracted_native.close()
        expanded_native.close()
        source_native.close()
