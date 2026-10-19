from collections.abc import Sequence

import numpy as np
import pytest

import phydrax as phx
from phydrax import _meshcore
from phydrax._meshcore import exact_orient3d, TetMesh3D, TetMeshBoundaryPolicy
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._tetra_metric import MetricRemeshingCriterion
from phydrax.meshing._volume_generation import generate_plc_volume, NativeVolumeSchedule
from phydrax.meshing.providers._native_route_policy import publication_policy
from tests._support.coordinate_references import native_meshing_request
from tests._support.distributed_meshing.volume_generation import (
    _mixed_refinement_of_nondyadic_plc,
)
from tests._support.forced_devices import run_on_forced_devices
from tests._support.meshing_fixtures import (
    box_corners,
    BOX_LOOPS,
    volume_mesh,
    volume_specification,
)


M = phx.meshing
_SI = phx.SpatialCoordinateContract.si()


# Fixed scalar-reference allowances from test_native_volume_generation.py
# at a6827a9e5; values and relative/absolute roles are unchanged.
# Sum of exterior oriented triangle areas against independently declared PLC boundary
# area.
_PLC_BOUNDARY_AREA_RTOL = 1e-12
# Signed tetrahedron volume sums versus divergence-volume and analytic solid
# inventories 1, 3, 8, 16 and 56.
_PLC_VOLUME_RTOL = 1e-13


def _mesh(
    complex_: M.PiecewiseLinearComplex,
    *,
    size: float = 4.0,
    limits: M.MeshingLimits | None = None,
) -> M.CellMeshingResult:
    source = M.NativePlcSource(complex_, "plc", "r1")
    specification = volume_specification(source, size=size, limits=limits)
    return volume_mesh(source, specification, coordinate_contract=_SI)


@pytest.fixture(scope="module")
def volume_runtime_resource_reference() -> tuple[
    M.PiecewiseLinearComplex, M.CellMeshingResult, int, int
]:
    complex_ = M.PiecewiseLinearComplex(
        box_corners((0, 0, 0), (1, 1, 1)),
        BOX_LOOPS,
        np.zeros(len(BOX_LOOPS), dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        ("solid",),
        boundary="fixed",
    )
    result = _mesh(complex_)
    evidence = result.execution_evidence
    assert evidence is not None
    queries = int(np.asarray(evidence.total_geometry_queries))
    pending: list[NativeExecutionRecord] = [evidence]
    cavity = 0
    while pending:
        current = pending.pop()
        cavity = max(cavity, int(np.asarray(current.work_counters.peak_cavity_cells)))
        if current.preparation_evidence is not None:
            pending.append(current.preparation_evidence)
        if current.consumer_evidence is not None:
            pending.append(current.consumer_evidence)
    assert cavity > 1 and queries > 1
    return complex_, result, cavity, queries


def test_native_volume_declarations_agree_with_published_resource_contract(
    volume_runtime_resource_reference: tuple[
        M.PiecewiseLinearComplex, M.CellMeshingResult, int, int
    ],
) -> None:
    _, reference, _, _ = volume_runtime_resource_reference
    policy = publication_policy("volume")
    assert reference.runtime.enforced_limits == policy.enforced_limits
    assert reference.runtime.unenforced_limits == policy.unenforced_limits
    assert reference.derivative_mode is policy.derivative_mode
    assert reference.derivative_mode is M.MeshingDerivativeMode.NONDIFFERENTIABLE
    assert not set(reference.runtime.enforced_limits) & set(
        reference.runtime.unenforced_limits
    )
    # Existing resource declarations retain their order when a newly proven
    # resource is added; the two tests below prove the added limits at runtime.
    established = (
        "vertices",
        "edges",
        "faces",
        "cells",
        "connectivity_entries",
        "data_bytes",
        "scratch_bytes",
        "work_units",
        "wall_seconds",
    )
    assert (
        tuple(name for name in reference.runtime.enforced_limits if name in established)
        == established
    )


def test_native_volume_cavity_limit_matches_observed_peak_and_refuses_one_less(
    volume_runtime_resource_reference: tuple[
        M.PiecewiseLinearComplex, M.CellMeshingResult, int, int
    ],
) -> None:
    complex_, reference, cavity, _ = volume_runtime_resource_reference
    admitted = _mesh(complex_, limits=M.MeshingLimits(maximum_cavity_cells=cavity))
    assert admitted.audit.passed
    assert "cavity_cells" in admitted.runtime.enforced_limits
    assert "cavity_cells" not in admitted.runtime.unenforced_limits
    assert "cavity_cells" in publication_policy("volume").enforced_limits
    np.testing.assert_array_equal(admitted.mesh.coordinates, reference.mesh.coordinates)
    before = complex_.vertices.copy()
    published: list[M.CellMeshingResult] = []
    with pytest.raises(M.MeshingFailure) as caught:
        published.append(
            _mesh(complex_, limits=M.MeshingLimits(maximum_cavity_cells=cavity - 1))
        )
    assert not published
    np.testing.assert_array_equal(complex_.vertices, before)
    failure = caught.value
    assert failure.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.evidence.requested)["maximum_cavity_cells"] == cavity - 1
    cause: BaseException | None = failure
    recovery: _meshcore.PlcRecoveryFailure | None = None
    while cause is not None:
        if isinstance(cause, _meshcore.PlcRecoveryFailure):
            recovery = cause
        cause = cause.__cause__
    assert recovery is not None
    assert recovery.status is _meshcore.MeshcoreStatus.CAPACITY_EXCEEDED
    assert recovery.reason == "tetrahedron_budget"


def test_native_volume_query_limit_matches_observed_usage_and_refuses_one_less(
    volume_runtime_resource_reference: tuple[
        M.PiecewiseLinearComplex, M.CellMeshingResult, int, int
    ],
) -> None:
    complex_, reference, _, queries = volume_runtime_resource_reference
    admitted = _mesh(complex_, limits=M.MeshingLimits(maximum_geometry_queries=queries))
    assert admitted.audit.passed
    assert "geometry_queries" in admitted.runtime.enforced_limits
    assert "geometry_queries" not in admitted.runtime.unenforced_limits
    assert "geometry_queries" in publication_policy("volume").enforced_limits
    np.testing.assert_array_equal(admitted.mesh.coordinates, reference.mesh.coordinates)
    before = complex_.vertices.copy()
    published: list[M.CellMeshingResult] = []
    with pytest.raises(M.MeshingFailure) as caught:
        published.append(
            _mesh(complex_, limits=M.MeshingLimits(maximum_geometry_queries=queries - 1))
        )
    assert not published
    np.testing.assert_array_equal(complex_.vertices, before)
    failure = caught.value
    assert failure.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(failure.evidence.requested)["maximum_geometry_queries"] == queries - 1


@pytest.mark.parametrize("maximum_vertices", (1, 7))
def test_native_volume_refuses_insufficient_source_vertex_capacity_before_recovery(
    maximum_vertices: int,
) -> None:
    complex_ = M.PiecewiseLinearComplex(
        box_corners((0, 0, 0), (1, 1, 1)),
        BOX_LOOPS,
        np.zeros(len(BOX_LOOPS), dtype=np.int64),
        np.asarray(((-1, 0),), dtype=np.int64),
        ("solid",),
        boundary="fixed",
    )
    before = phx.array_tree_fingerprint(complex_)

    published: list[M.CellMeshingResult] = []
    with pytest.raises(M.MeshingFailure) as caught:
        published.append(
            _mesh(complex_, limits=M.MeshingLimits(maximum_vertices=maximum_vertices))
        )
    assert not published
    assert phx.array_tree_fingerprint(complex_) == before
    failure = caught.value
    assert failure.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert failure.stage == M.MeshingStageKind.SOURCE_INSPECTION.value
    assert failure.evidence.provider_code == "vertex_budget"
    assert dict(failure.evidence.requested)["maximum_vertices"] == maximum_vertices
    achieved = dict(failure.evidence.achieved)
    assert achieved["required_vertices"] == 8
    assert achieved["available_vertices"] == maximum_vertices


def test_native_volume_accepts_source_vertex_capacity_boundary(
    volume_runtime_resource_reference: tuple[
        M.PiecewiseLinearComplex, M.CellMeshingResult, int, int
    ],
) -> None:
    complex_, reference, _, _ = volume_runtime_resource_reference
    before = phx.array_tree_fingerprint(complex_)
    admitted = _mesh(complex_, limits=M.MeshingLimits(maximum_vertices=8))
    assert admitted.audit.passed
    assert admitted.mesh.coordinates.shape[0] == 8
    np.testing.assert_array_equal(admitted.mesh.coordinates, reference.mesh.coordinates)
    assert phx.array_tree_fingerprint(complex_) == before


@pytest.mark.parametrize(
    "strength",
    (M.SizeControlStrength.HARD, M.SizeControlStrength.SOFT),
    ids=("hard-statistical-goal", "soft-refinement-preference"),
)
def test_plc_statistical_metric_epoch_requires_a_hard_size_goal(
    strength: M.SizeControlStrength,
) -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    plc = M.PiecewiseLinearComplex(
        points,
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("simplex",),
    )
    source = M.NativePlcSource(plc, "size-goal-simplex", "size-goal-authored")
    limits = M.MeshingLimits()
    specification = volume_specification(
        source, size=4.0, limits=limits, strength=strength
    )
    with native_meshing_request(limits) as (scope, _, _):
        construction = generate_plc_volume(
            plc,
            specification,
            NativeVolumeSchedule(),
            validity_policy=phx.discretization.CellValidityPolicy(),
            source_id=source.source_id,
            source_revision=source.source_revision,
            input_id=source.binding_id,
            scope=scope,
        )
    evidence = construction.metric_optimization
    if strength is M.SizeControlStrength.SOFT:
        assert evidence is None
    else:
        assert evidence is not None
        assert evidence.criterion is MetricRemeshingCriterion.SIZE_STATISTICS
        control = specification.size_controls[0]
        assert (
            dict(evidence.size_requested)[f"size:{control.control_id}:target_size"] == 4.0
        )
        assert evidence.size_issues


@pytest.mark.parametrize(
    ("maximum_edges", "maximum_faces", "quantity", "expected"),
    ((5, 4, "edges", 6), (6, 3, "faces", 4)),
    ids=("tetrahedral-edge-budget", "tetrahedral-face-budget"),
)
def test_native_tetrahedron_refuses_complete_entity_budget(
    maximum_edges: int,
    maximum_faces: int,
    quantity: str,
    expected: int,
) -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    source = M.PiecewiseLinearComplex(
        points,
        tuple(
            np.asarray(loop, dtype=np.int64)
            for loop in ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
        ),
        np.arange(4, dtype=np.int64),
        np.tile(np.asarray((-1, 0), dtype=np.int64), (4, 1)),
        ("solid",),
        boundary="fixed",
    )
    with pytest.raises(M.MeshingFailure) as refusal:
        _mesh(
            source,
            limits=M.MeshingLimits(
                maximum_edges=maximum_edges,
                maximum_faces=maximum_faces,
            ),
        )
    assert refusal.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert dict(refusal.value.evidence.achieved)[quantity] == expected


def _cells(result: M.CellMeshingResult) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(result.mesh.coordinates, dtype=np.float64)
    return points, np.asarray(result.mesh.blocks[0].vertices, dtype=np.int64)


def _zone_volumes(result: M.CellMeshingResult) -> dict[str, float]:
    """Region volumes from the published zones and coordinates (numpy oracle)."""

    points, cells = _cells(result)
    corners = points[cells]
    volumes = (
        np.linalg.det(corners[:, 1:] - corners[:, :1]) / 6.0
    )  # oriented tetrahedron volume
    identifiers = np.asarray(result.mesh.entity_set(3).entity_ids)
    return {
        zone.name: float(np.sum(volumes[np.isin(identifiers, zone.scope.entity_ids)]))
        for zone in result.zones
    }


def _assert_embedded(result: M.CellMeshingResult, boundary_area: float) -> None:
    """Check overlap-free cover: every cell positively oriented (exact predicate),
    every interior face shared by two cells with opposite orientation, and the
    boundary faces summing to the declared boundary area. By the degree
    argument these imply that the cells cover the domain exactly once."""

    points, cells = _cells(result)
    corners = points[cells]
    signs = exact_orient3d(corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3])
    assert np.all(signs > 0)
    opposite = ((1, 3, 2), (0, 2, 3), (0, 3, 1), (0, 1, 2))
    oriented = np.concatenate([cells[:, list(face)] for face in opposite])
    keys = np.sort(oriented, axis=1)
    unique, inverse, counts = np.unique(
        keys, axis=0, return_inverse=True, return_counts=True
    )
    assert np.all(counts <= 2)
    boundary = counts[inverse] == 1
    triangles = points[oriented[boundary]]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    assert np.sum(areas) == pytest.approx(boundary_area, rel=_PLC_BOUNDARY_AREA_RTOL)
    assert unique.shape[0] == (4 * cells.shape[0] + np.count_nonzero(boundary)) // 2


def _certification_passed(result: M.CellMeshingResult) -> bool:
    return any(
        stage.stage is M.MeshingStageKind.CERTIFICATION
        and stage.status is M.MeshingStageStatus.PASSED
        for stage in result.trace.stages
    )


def _achieved(result: M.CellMeshingResult, name: str) -> float:
    return dict(result.compliance.achieved)[name]


def test_cube_volume_mesh_is_certified_with_exact_volume_and_boundary() -> None:
    complex_ = M.PiecewiseLinearComplex(
        box_corners((0, 0, 0), (1, 1, 1)), BOX_LOOPS, np.zeros(6), [[-1, 0]], ("solid",)
    )
    result = _mesh(complex_, size=0.4)
    assert _certification_passed(result)
    assert _zone_volumes(result) == {"solid": pytest.approx(1.0, rel=_PLC_VOLUME_RTOL)}
    _assert_embedded(result, 6.0)
    assert {patch.name for patch in result.patches} == {"facet:0"}


def test_nonconvex_l_shape_is_meshed_exactly() -> None:
    base = [(0, 0), (2, 0), (2, 1), (1, 1), (1, 2), (0, 2)]
    count = len(base)
    vertices = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    loops = [
        tuple(reversed(range(count))),
        tuple(range(count, 2 * count)),
        *((i, (i + 1) % count, (i + 1) % count + count, i + count) for i in range(count)),
    ]
    complex_ = M.PiecewiseLinearComplex(
        vertices, loops, np.zeros(len(loops)), [[-1, 0]], ("solid",)
    )
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _zone_volumes(result) == {"solid": pytest.approx(3.0, rel=_PLC_VOLUME_RTOL)}
    _assert_embedded(result, 2 * 3.0 + 8.0)


def _schonhardt(boundary: TetMeshBoundaryPolicy) -> M.PiecewiseLinearComplex:
    """Twisted triangular prism whose side diagonals are reflex edges."""

    vertices = np.asarray(
        [
            (1.0, 0.0, 0.0),
            (-0.5, 0.875, 0.0),
            (-0.5, -0.875, 0.0),
            (0.875, 0.5, 1.0),
            (-0.875, 0.5, 1.0),
            (0.0, -1.0, 1.0),
        ]
    )
    triangles = [(0, 2, 1), (3, 4, 5)]
    for i in range(3):
        a, b = i, (i + 1) % 3
        c, d = b + 3, a + 3
        # Split each side quadrilateral along its non-hull (reflex) diagonal.
        others = [k for k in range(6) if k not in (a, b, c)]
        sides = np.sign(
            [
                np.linalg.det(
                    np.stack(
                        (
                            vertices[b] - vertices[a],
                            vertices[c] - vertices[a],
                            vertices[k] - vertices[a],
                        )
                    )
                )
                for k in others
            ]
        )
        hull = bool(np.all(sides == sides[0]))
        triangles += [(a, b, d), (b, c, d)] if hull else [(a, b, c), (a, c, d)]
    return M.PiecewiseLinearComplex(
        vertices, triangles, np.zeros(8), [[-1, 0]], ("solid",), boundary=boundary
    )


def _divergence_volume(complex_: M.PiecewiseLinearComplex) -> float:
    offsets = complex_.polygon_offsets
    total = 0.0
    for polygon in range(offsets.shape[0] - 1):
        loop = complex_.vertices[
            complex_.polygon_vertices[offsets[polygon] : offsets[polygon + 1]]
        ]
        for k in range(1, loop.shape[0] - 1):
            total += float(np.dot(loop[0], np.cross(loop[k], loop[k + 1]))) / 6.0
    return total


def test_schonhardt_polyhedron_is_recovered_with_steiner_points() -> None:
    complex_ = _schonhardt("conforming")
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _achieved(result, "steiner_points") > 0
    assert _zone_volumes(result)["solid"] == pytest.approx(
        _divergence_volume(complex_), rel=_PLC_VOLUME_RTOL
    )


def test_fixed_schonhardt_uses_interior_fill_without_changing_boundary() -> None:
    complex_ = _schonhardt("fixed")
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _zone_volumes(result)["solid"] == pytest.approx(
        _divergence_volume(complex_), rel=_PLC_VOLUME_RTOL
    )
    points, cells = _cells(result)
    faces = np.concatenate(
        [cells[:, list(face)] for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))]
    )
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    published = {
        tuple(sorted(map(tuple, points[row].tolist()))) for row in keys[counts == 1]
    }
    declared = {
        tuple(
            sorted(
                map(
                    tuple,
                    complex_.vertices[complex_.polygon_vertices[start:end]].tolist(),
                )
            )
        )
        for start, end in zip(
            complex_.polygon_offsets[:-1], complex_.polygon_offsets[1:], strict=True
        )
    }
    assert published == declared


def test_internal_cavity_is_removed_from_the_region() -> None:
    outer = box_corners((0, 0, 0), (4, 4, 4))
    inner = box_corners((1, 1, 1), (3, 3, 3))
    loops = [*BOX_LOOPS, *(tuple(v + 8 for v in loop) for loop in BOX_LOOPS)]
    # Inner loops point out of the cavity into the solid: positive side solid.
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((outer, inner)),
        loops,
        [0] * 6 + [1] * 6,
        [[-1, 0], [0, -1]],
        ("solid",),
    )
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _zone_volumes(result) == {"solid": pytest.approx(56.0, rel=_PLC_VOLUME_RTOL)}
    _assert_embedded(result, 6 * 16.0 + 6 * 4.0)


def test_two_material_box_publishes_conforming_interface() -> None:
    vertices = np.asarray(
        [(x, y, z) for z in (0, 1) for y in (0, 1) for x in (0, 1, 2)], dtype=np.float64
    )

    def index(x: int, y: int, z: int) -> int:
        return x + 3 * y + 6 * z

    loops: list[tuple[int, ...]] = []
    facets: list[int] = []
    for x0, facet in ((0, 0), (1, 1)):
        x1 = x0 + 1
        for loop in (
            (index(x0, 0, 0), index(x0, 1, 0), index(x1, 1, 0), index(x1, 0, 0)),
            (index(x0, 0, 1), index(x1, 0, 1), index(x1, 1, 1), index(x0, 1, 1)),
            (index(x0, 0, 0), index(x1, 0, 0), index(x1, 0, 1), index(x0, 0, 1)),
            (index(x0, 1, 0), index(x0, 1, 1), index(x1, 1, 1), index(x1, 1, 0)),
        ):
            loops.append(loop)
            facets.append(facet)
    loops += [
        (index(0, 0, 0), index(0, 0, 1), index(0, 1, 1), index(0, 1, 0)),
        (index(2, 0, 0), index(2, 1, 0), index(2, 1, 1), index(2, 0, 1)),
        (index(1, 0, 0), index(1, 1, 0), index(1, 1, 1), index(1, 0, 1)),
    ]
    facets += [0, 1, 2]
    complex_ = M.PiecewiseLinearComplex(
        vertices, loops, facets, [[-1, 0], [-1, 1], [1, 0]], ("left", "right")
    )
    source = M.NativePlcSource(complex_, "materials", "r1")
    base = volume_specification(source, size=0.5)
    region_scopes = tuple(
        M.MeshingScope(
            source.source_id,
            source.source_revision,
            M.MeshingEntityKind.GEOMETRY,
            3,
            "material-regions",
            np.asarray([region], dtype=np.int64),
        )
        for region in range(2)
    )
    shared = M.MeshingScope(
        source.source_id,
        source.source_revision,
        M.MeshingEntityKind.GEOMETRY,
        2,
        base.boundary_scope.entity_set_id,
        np.asarray([2], dtype=np.int64),
    )
    specification = M.VolumeMeshingSpec(
        base.target,
        base.boundary_scope,
        base.fill_strategy,
        size_controls=base.size_controls,
        region_controls=(
            M.RegionControl(region_scopes[1], "right", "steel", M.RegionRole.SOLID),
            M.RegionControl(region_scopes[0], "left", "water", M.RegionRole.FLUID),
        ),
        patch_controls=(
            M.PatchControl("shared-material-wall", shared, ("left", "right")),
        ),
    )
    result = (
        M.NativeMeshingProvider(M.NativeMeshingOptions("plc_tetrahedral"))
        .plan(source, specification, coordinate_contract=_SI)
        .execute()
    )
    assert _certification_passed(result)
    assert _zone_volumes(result) == {
        "left": pytest.approx(1.0, rel=_PLC_VOLUME_RTOL),
        "right": pytest.approx(1.0, rel=_PLC_VOLUME_RTOL),
    }
    zones = {zone.name: zone for zone in result.zones}
    assert zones["left"].material_id == "water"
    assert zones["left"].region_role is M.RegionRole.FLUID
    assert zones["right"].material_id == "steel"
    assert zones["right"].region_role is M.RegionRole.SOLID
    points, _ = _cells(result)
    interface = next(label for label in result.labels if label.name == "interface")
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    identifiers = np.asarray(result.mesh.entity_set(2).entity_ids)
    rows = np.flatnonzero(np.isin(identifiers, interface.scope.entity_ids))
    corners = points[faces[rows]]
    assert np.all(corners[..., 0] == 1.0)
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    assert np.sum(area) == pytest.approx(1.0, rel=_PLC_VOLUME_RTOL)
    patch = next(
        patch for patch in result.patches if patch.name == "shared-material-wall"
    )
    assert np.array_equal(patch.scope.entity_ids, interface.scope.entity_ids)
    assert set(patch.adjacent_zone_ids) == {
        zones["left"].zone_id,
        zones["right"].zone_id,
    }


def test_embedded_internal_sheet_is_a_conforming_face_set() -> None:
    cube = box_corners((0, 0, 0), (2, 2, 2))
    sheet = np.asarray(
        [(0.5, 0.5, 1.0), (1.5, 0.5, 1.0), (1.5, 1.5, 1.0), (0.5, 1.5, 1.0)]
    )
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((cube, sheet)),
        [*BOX_LOOPS, (8, 9, 10, 11)],
        [0] * 6 + [1],
        [[-1, 0], [0, 0]],
        ("solid",),
    )
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _zone_volumes(result) == {"solid": pytest.approx(8.0, rel=_PLC_VOLUME_RTOL)}
    points, _ = _cells(result)
    label = next(label for label in result.labels if label.name == "internal_sheet")
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    identifiers = np.asarray(result.mesh.entity_set(2).entity_ids)
    corners = points[faces[np.isin(identifiers, label.scope.entity_ids)]]
    assert np.all(corners[..., 2] == 1.0)
    area = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1
    )
    assert np.sum(area) == pytest.approx(1.0, rel=_PLC_VOLUME_RTOL)


def test_opposed_sheets_keep_individual_oriented_source_associations() -> None:
    cube = box_corners((0, 0, 0), (2, 2, 2))
    lower = np.asarray(
        [(0.5, 0.5, 0.75), (1.5, 0.5, 0.75), (1.5, 1.5, 0.75), (0.5, 1.5, 0.75)],
        dtype=np.float64,
    )
    upper = lower.copy()
    upper[:, 2] = 1.25
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((cube, lower, upper)),
        [*BOX_LOOPS, (8, 9, 10, 11), (15, 14, 13, 12)],
        [0] * 6 + [1, 1],
        [[-1, 0], [0, 0]],
        ("solid",),
    )
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _zone_volumes(result)["solid"] == pytest.approx(8.0, rel=_PLC_VOLUME_RTOL)
    points, _ = _cells(result)
    face_set = result.mesh.entity_set(2)
    identifiers = np.asarray(face_set.entity_ids)
    connectivity = result.mesh.connectivity
    assert isinstance(connectivity, phx.discretization.TetrahedralConnectivity)
    faces = np.asarray(connectivity.faces, dtype=np.int64)
    association = next(
        value
        for value in result.associations
        if value.target_entity_set_id == face_set.entity_set_id
    )
    orientation = dict(
        zip(
            np.asarray(association.target_global_ids).tolist(),
            np.asarray(association.orientations).tolist(),
            strict=True,
        )
    )
    sheet = next(label for label in result.labels if label.name == "internal_sheet")
    rows = np.flatnonzero(np.isin(identifiers, sheet.scope.entity_ids))
    corners = points[faces[rows]]
    signs = np.asarray([orientation[int(identifiers[row])] for row in rows])
    source_normal_z = (
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])[:, 2]
        * signs
    )
    assert np.all(source_normal_z[corners[:, 0, 2] == 0.75] > 0)
    assert np.all(source_normal_z[corners[:, 0, 2] == 1.25] < 0)
    areas = 0.5 * np.abs(source_normal_z)
    assert np.sum(areas[corners[:, 0, 2] == 0.75]) == pytest.approx(1.0)
    assert np.sum(areas[corners[:, 0, 2] == 1.25]) == pytest.approx(1.0)


def test_small_angle_wedge_is_protected_and_meshed() -> None:
    base = [(0.0, 0.0), (16.0, -1.0), (16.0, 1.0)]
    vertices = [(x, y, 0.0) for x, y in base] + [(x, y, 1.0) for x, y in base]
    loops = [(0, 2, 1), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)]
    complex_ = M.PiecewiseLinearComplex(
        vertices, loops, np.zeros(5), [[-1, 0]], ("solid",)
    )
    result = _mesh(complex_)
    assert _certification_passed(result)
    assert _achieved(result, "construction:protected_vertices") > 0
    assert _zone_volumes(result) == {"solid": pytest.approx(16.0, rel=_PLC_VOLUME_RTOL)}


def test_fixed_boundary_triangles_are_published_unchanged() -> None:
    corners = box_corners((0, 0, 0), (1, 1, 1))
    triangles = [
        tri
        for a, b, c, d in BOX_LOOPS
        for tri in (((a, b, c), (a, c, d)) if a % 2 == 0 else ((a, b, d), (b, c, d)))
    ]
    complex_ = M.PiecewiseLinearComplex(
        corners, triangles, np.zeros(12), [[-1, 0]], ("solid",), boundary="fixed"
    )
    result = _mesh(complex_, size=0.3)
    assert _certification_passed(result)
    points, cells = _cells(result)
    faces = np.concatenate(
        [cells[:, list(face)] for face in ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))]
    )
    keys, counts = np.unique(np.sort(faces, axis=1), axis=0, return_counts=True)
    published = {
        tuple(sorted(map(tuple, points[row].tolist()))) for row in keys[counts == 1]
    }
    declared = {
        tuple(sorted(map(tuple, corners[list(tri)].tolist()))) for tri in triangles
    }
    assert published == declared


def test_intersecting_facets_are_refused_with_the_polygons() -> None:
    first = box_corners((0, 0, 0), (1, 1, 1))
    second = box_corners((0.5, 0.5, 0.5), (1.5, 1.5, 1.5))
    complex_ = M.PiecewiseLinearComplex(
        np.concatenate((first, second)),
        [*BOX_LOOPS, *(tuple(v + 8 for v in loop) for loop in BOX_LOOPS)],
        [0] * 6 + [1] * 6,
        [[-1, 0], [-1, 1]],
        ("a", "b"),
    )
    with pytest.raises(M.MeshingFailure) as raised:
        _mesh(complex_)
    evidence = raised.value.evidence
    assert evidence.category is M.MeshingFailureCategory.INVALID_SOURCE
    assert evidence.provider_code == "intersecting_constraints"
    assert len(evidence.entity_ids) == 2


def test_open_boundary_is_refused_with_its_region() -> None:
    complex_ = M.PiecewiseLinearComplex(
        box_corners((0, 0, 0), (1, 1, 1)),
        BOX_LOOPS[:5],
        np.zeros(5),
        [[-1, 0]],
        ("solid",),
    )
    with pytest.raises(M.MeshingFailure) as raised:
        _mesh(complex_)
    assert raised.value.evidence.provider_code == "open_boundary"


def test_inverted_orientation_is_refused_as_a_region_leak() -> None:
    complex_ = M.PiecewiseLinearComplex(
        box_corners((0, 0, 0), (1, 1, 1)), BOX_LOOPS, np.zeros(6), [[0, -1]], ("solid",)
    )
    with pytest.raises(M.MeshingFailure) as raised:
        _mesh(complex_)
    assert raised.value.evidence.provider_code == "region_leak"


def test_exhausted_work_budget_is_refused_as_resource_exhaustion() -> None:
    with pytest.raises(M.MeshingFailure) as raised:
        _mesh(_schonhardt("conforming"), limits=M.MeshingLimits(maximum_work_units=50))
    evidence = raised.value.evidence
    assert evidence.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert evidence.provider_code == "work_budget"


def test_mixed_refined_plc_target_certifies_exact_coverage_of_every_source_facet() -> (
    None
):
    limits = M.MeshingLimits(maximum_vertices=100, maximum_cells=1000)
    result, adaptation = _mixed_refinement_of_nondyadic_plc(limits)
    target = adaptation.target
    certificate = target.certification
    assert adaptation.status is M.MeshAdaptationStatus.COMPLETE
    assert certificate is not None and certificate.passed
    coverage, embedding = certificate.coverage, certificate.embedding
    assert coverage is not None and embedding is not None
    assert coverage.status == "certified" and not coverage.findings
    assert coverage.binding.coordinate_scope == "affine"
    assert "exact_plc_source_coordinates" in embedding.evaluated_checks
    assert target.geometry.exact_source is not None
    assert coverage.embedding_certificate_id == embedding.certificate_id
    assert coverage.source_facet_count == 4
    assert coverage.covered_source_facet_count == coverage.source_facet_count
    assert coverage.achieved_region_measures == coverage.requested_region_measures
    source_certificate = result.certification
    assert source_certificate is not None
    source_coverage = source_certificate.coverage
    assert source_coverage is not None
    assert coverage.domain_id == source_coverage.domain_id
    # The target owns its metered proof under the source request limits.
    assert coverage.binding.limits_id == source_coverage.binding.limits_id
    work = coverage.source_expression_work_units
    assert work is not None and 0 < work <= certificate.request.limits.maximum_work_units


def test_mixed_refined_plc_target_coverage_refuses_one_unit_below_its_metered_work() -> (
    None
):
    generous = M.MeshingLimits(maximum_vertices=100, maximum_cells=1000)
    _, adaptation = _mixed_refinement_of_nondyadic_plc(generous)
    certificate = adaptation.target.certification
    assert (
        certificate is not None
        and certificate.coverage is not None
        and certificate.embedding is not None
    )
    coverage_work = certificate.coverage.source_expression_required_work_units
    coverage_charged = certificate.coverage.source_expression_work_units
    embedding_work = certificate.embedding.source_expression_work_units
    assert (
        coverage_work is not None
        and coverage_charged is not None
        and embedding_work is not None
        and coverage_charged <= coverage_work
    )
    # The embedding work fits independently of the coverage phase.
    assert embedding_work < coverage_work
    exact = run_on_forced_devices(
        "tests._support.distributed_meshing.volume_generation",
        "coverage_threshold",
        ("0",),
        devices=1,
        env={},
    )
    assert exact["status"] == "ACCEPT"
    assert exact["phase_required_work"] == coverage_work
    assert exact["phase_charged_work"] <= coverage_work
    assert exact["charged_work"] <= exact["required_work"]
    refused = run_on_forced_devices(
        "tests._support.distributed_meshing.volume_generation",
        "coverage_threshold",
        ("1",),
        devices=1,
        env={},
    )
    assert refused == {
        "status": "RESOURCE",
        "resource": "coefficient_work",
        "limit": exact["required_work"] - 1,
        "requested": exact["required_work"],
    }


# Relative determinant 0.0705 charted from (0, 0, 0) and 3.9e-4 from (10, 10, 1).
_CHART_POINTS = np.asarray(
    ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (10.0, 10.0, 1.0)),
    dtype=np.float64,
)


def test_native_proposal_floor_charts_exactly_like_the_publishing_validity_policy() -> (
    None
):
    policy = phx.discretization.CellValidityPolicy(relative_determinant_floor=0.01)
    statuses = []
    # Positively oriented rows whose first vertex is the near or the far corner.
    for row in ((0, 1, 2, 3), (3, 2, 1, 0)):
        mesh = phx.discretization.CellMesh(
            _CHART_POINTS,
            (phx.discretization.CellBlock("cell", "tetrahedron", np.asarray([row])),),
        )
        validity = phx.discretization.certify_cell_geometry_validity(
            phx.discretization.CellGeometrySpec.affine(mesh),
            mesh=mesh,
            policy=policy,
        )
        statuses.append(int(np.asarray(validity.status)[0]))
    valid, invalid = (
        int(phx.discretization.CellValidityStatus.CERTIFIED_VALID),
        int(phx.discretization.CellValidityStatus.INVALID),
    )
    assert statuses == [valid, invalid]
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    edges = np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32)
    state = TetMesh3D(
        _CHART_POINTS,
        np.asarray([[0, 1, 2, 3]], dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        faces,
        np.arange(4, dtype=np.int32),
        edges,
        np.arange(6, dtype=np.int32),
        boundary_policy="fixed",
        max_vertices=8,
        max_tetrahedra=8,
    )
    try:
        cells = np.asarray([[0, 1, 2, 3]], dtype=np.int32)

        def below(identities: Sequence[int]) -> int:
            return state.proposal_shape(
                _CHART_POINTS,
                cells,
                vertex_ids=np.asarray(identities, dtype=np.int64),
                minimum_relative_determinant=policy.relative_determinant_floor,
            )[2]

        # The publishing chart (smallest identity) decides, not local rows.
        assert below((0, 1, 2, 3)) == 0
        assert below((1, 2, 3, 0)) == 1
        assert below((900, 7, 2**40, 50)) == 0
        assert below((9 * 2**50, 2**40 + 5, 2**40 + 7, 3)) == 1
        with pytest.raises(ValueError, match="nonnegative int64"):
            below((0, -1, 2, 3))
        with pytest.raises(ValueError, match="one identity per point"):
            below((0, 1, 2))
    finally:
        state.close()


def test_postconstruction_host_work_keeps_original_cumulative_allowance() -> None:
    from phydrax.meshing._volume_generation import native_volume_execution_budget

    limits = M.MeshingLimits(maximum_work_units=5, maximum_geometry_queries=3)
    published: list[str] = []
    with pytest.raises(M.MeshingFailure) as caught:
        with native_volume_execution_budget(limits) as execution:
            with execution.batch(3, queries_upper=2) as ticket:
                ticket.consume(3, geometry_queries=2)
            with native_volume_execution_budget(
                M.MeshingLimits(maximum_work_units=100, maximum_geometry_queries=100),
                stage=M.MeshingStageKind.CERTIFICATION,
            ) as certification:
                with certification.batch(3, queries_upper=2) as ticket:
                    ticket.consume(3, geometry_queries=2)
                published.append("inadmissible")
    assert not published
    assert caught.value.category is M.MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert execution.evidence is not None
    assert execution.evidence.externally_charged_work == 3
    assert int(execution.evidence.work_counters.source_queries) == 2


def test_host_phase_cannot_restart_original_operation_clock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _volume_generation as volume
    from phydrax.meshing.providers import _native_route_checks as publication

    elapsed = [0.0]
    monkeypatch.setattr(volume, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(publication, "monotonic", lambda: elapsed[0])
    monkeypatch.setattr(_meshcore.time, "monotonic", lambda: elapsed[0])
    limits = M.MeshingLimits(maximum_wall_seconds=4.0)
    dispatched: list[str] = []
    with pytest.raises(M.MeshingFailure) as caught:
        with volume.native_volume_execution_budget(limits, operation_started=0.0):
            elapsed[0] = 5.0
            with volume.native_volume_execution_budget(
                limits,
                operation_started=5.0,
                stage=M.MeshingStageKind.CERTIFICATION,
            ):
                dispatched.append("inadmissible")
    assert not dispatched
    assert caught.value.category is M.MeshingFailureCategory.TIMED_OUT
    assert caught.value.stage == M.MeshingStageKind.CERTIFICATION.value
    assert dict(caught.value.evidence.achieved)["elapsed_seconds"] == 5.0
