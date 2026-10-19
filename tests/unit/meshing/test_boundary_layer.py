from pathlib import Path

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available
from phydrax.meshing.qualification import qualify_hybrid_reference_cells
from tests._support.boundary_layer_sweeps import (
    cad_request,
    exact_sweep_fixture,
    face_points,
    invalid_cad_request,
    periodic_wall,
    ridge_wall,
)
from tests._support.boundary_layer_walls import (
    box,
    channel,
    control,
    icosphere,
    plate,
    SCHEDULE,
    wall,
    wedge,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)


# Tolerances below preserve d2dfcb941's independent geometric regression oracles.
# Each bound names its source fixture rather than widening acceptance.
PLANAR_RELATIVE_TOLERANCE = 1e-12
SPHERE_RELATIVE_TOLERANCE = 1e-6
RIDGE_RELATIVE_TOLERANCE = 1e-9
NUMPY_REFERENCE_RELATIVE_TOLERANCE = 1e-7
# Preserved absolute slab-coordinate oracle bound from the CAD fixture.
CAD_COORDINATE_ABSOLUTE_TOLERANCE = 1e-14


def _assert_certified(result: phx.meshing.BoundaryLayerMesh) -> None:
    cells = sum(block.cell_count for block in result.mesh.blocks)
    assert result.validity.certified_valid_count == cells
    assert result.evidence.certified_valid_count == cells


@pytest.fixture(scope="module")
def planar_layers() -> tuple[phx.discretization.CellMesh, phx.meshing.BoundaryLayerMesh]:
    points, triangles = plate(5)
    mesh = wall(points, triangles)
    return mesh, phx.meshing.prepare_boundary_layers(mesh, control(mesh))


def test_planar_layers_preserve_schedule_support_heights_and_certification(
    planar_layers: tuple[phx.discretization.CellMesh, phx.meshing.BoundaryLayerMesh],
) -> None:
    source, result = planar_layers
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses,
        SCHEDULE.thicknesses,
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        result.evidence.achieved_growth_rates, 1.2, rtol=PLANAR_RELATIVE_TOLERANCE
    )
    active = np.asarray(result.evidence.layer_active)
    assert active.dtype == np.bool_
    np.testing.assert_array_equal(active, (True, True, True))
    assert dict(result.evidence.cell_counts) == {"prism": 3 * source.blocks[0].cell_count}
    heights = np.unique(np.round(np.asarray(result.mesh.coordinates)[:, 2], 12))
    np.testing.assert_allclose(
        heights,
        (0.0, 0.01, 0.022, 0.0364),
        rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    _assert_certified(result)
    assert not result.closed_cap


def test_layer_evidence_refuses_mismatched_schedule_shape(
    planar_layers: tuple[phx.discretization.CellMesh, phx.meshing.BoundaryLayerMesh],
) -> None:
    _, result = planar_layers
    e = result.evidence
    values = {
        "requested_thicknesses": e.requested_thicknesses,
        "achieved_thicknesses": np.ones((2,), dtype=np.float64),
        "minimum_thicknesses": e.minimum_thicknesses,
        "maximum_thicknesses": e.maximum_thicknesses,
        "achieved_growth_rates": e.achieved_growth_rates,
        "layer_active": e.layer_active,
        "column_thicknesses": e.column_thicknesses,
        "column_active": e.column_active,
        "active_column_counts": e.active_column_counts,
        "column_count": e.column_count,
        "fan_column_count": e.fan_column_count,
        "corner_patch_count": e.corner_patch_count,
        "convex_ridge_count": e.convex_ridge_count,
        "concave_ridge_count": e.concave_ridge_count,
        "rim_vertex_count": e.rim_vertex_count,
        "minimum_visibility": e.minimum_visibility,
        "maximum_stretch": e.maximum_stretch,
        "unconverged_visibility_count": e.unconverged_visibility_count,
        "collision_policy": e.collision_policy,
        "predicted_collision_vertex_count": e.predicted_collision_vertex_count,
        "detected_collision_count": e.detected_collision_count,
        "resolution_iterations": e.resolution_iterations,
        "reduced_vertex_count": e.reduced_vertex_count,
        "terminated_vertex_count": e.terminated_vertex_count,
        "merged_vertex_count": e.merged_vertex_count,
        "minimum_scale": e.minimum_scale,
        "cell_counts": e.cell_counts,
        "certified_valid_count": e.certified_valid_count,
    }
    with pytest.raises(ValueError, match="BoundaryLayerEvidence.measurements"):
        phx.meshing.BoundaryLayerEvidence(**values)


def test_quadrilateral_layers_preserve_triangular_cap_and_transition_interval() -> None:
    values = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(values, values, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange(points.shape[0]).reshape(4, 4)
    quads = np.asarray(
        [
            (index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
            for i in range(3)
            for j in range(3)
        ],
        dtype=np.int64,
    )
    mesh = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "quadrilaterals",
                "quadrilateral",
                quads,
                global_ids=np.arange(9, dtype=np.int64),
            ),
        ),
    )
    result = phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    assert dict(result.evidence.cell_counts) == {"pyramid": 9, "hexahedron": 27}
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses,
        SCHEDULE.thicknesses,
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    assert result.cap is not None
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    layer_index = np.asarray(result.layer_index)
    pyramids = result.mesh.block("pyramids")
    np.testing.assert_array_equal(layer_index[np.asarray(pyramids.global_ids)], 3)
    _assert_certified(result)


def test_opposing_sloped_layers_terminate_locally_with_collapse_cells() -> None:
    lower, lower_triangles = plate(4)
    upper, upper_triangles = plate(4, flip=True)
    upper[:, 2] = 0.03 + 0.17 * upper[:, 0]
    mesh = wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )
    result = phx.meshing.prepare_boundary_layers(
        mesh,
        control(
            mesh, collision=phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY
        ),
    )
    assert 0 < result.evidence.terminated_vertex_count < 50
    counts = dict(result.evidence.cell_counts)
    assert counts["pyramid"] > 0 and counts["tetrahedron"] > 0
    assert result.evidence.achieved_thicknesses[0] == pytest.approx(0.01)
    _assert_certified(result)


def test_spherical_layers_preserve_radial_schedule_and_closed_cap() -> None:
    mesh = wall(*icosphere(0.4, 2))
    result = phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    evidence = result.evidence
    np.testing.assert_allclose(
        evidence.minimum_thicknesses, SCHEDULE.thicknesses, rtol=SPHERE_RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        evidence.maximum_thicknesses, SCHEDULE.thicknesses, rtol=SPHERE_RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        evidence.achieved_growth_rates, 1.2, rtol=SPHERE_RELATIVE_TOLERANCE
    )
    assert evidence.fan_column_count == 0
    assert result.closed_cap
    assert result.cap is not None
    radii = np.linalg.norm(np.asarray(result.cap.coordinates), axis=1)
    assert np.all(radii > 0.4)
    _assert_certified(result)


def test_convex_box_layers_keep_ridges_corners_fans_and_closed_cap() -> None:
    mesh = wall(*box((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), 3, outward=True))
    result = phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    evidence = result.evidence
    assert evidence.convex_ridge_count == 12 * 3
    assert evidence.corner_patch_count == 8
    assert evidence.fan_column_count > 0
    counts = dict(evidence.cell_counts)
    assert (
        counts["hexahedron"] > 0 and counts["tetrahedron"] > 0 and counts["pyramid"] > 0
    )
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=RIDGE_RELATIVE_TOLERANCE
    )
    assert result.closed_cap
    assert result.cap is not None
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    _assert_certified(result)


def test_concave_box_layers_preserve_corner_stretch_without_fans() -> None:
    mesh = wall(*box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3, outward=False))
    result = phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    evidence = result.evidence
    assert evidence.concave_ridge_count == 12 * 3
    assert evidence.fan_column_count == 0
    assert evidence.maximum_stretch == pytest.approx(
        np.sqrt(3.0), rel=SPHERE_RELATIVE_TOLERANCE
    )
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=RIDGE_RELATIVE_TOLERANCE
    )
    _assert_certified(result)
    assert result.cap is not None and result.closed_cap
    assert all(block.cell_kind == "triangle" for block in result.cap.blocks)


def test_sharp_wedge_refusal_identifies_every_crease_vertex() -> None:
    points, triangles = wedge()
    mesh = wall(points, triangles)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
    crease = np.flatnonzero(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 2], 0.0))
    assert set(failure.value.entity_ids) == {int(value) for value in crease}
    np.testing.assert_allclose(
        failure.value.locations,
        points[crease],
        rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
        atol=0.0,
    )


def test_selected_open_box_floor_keeps_rim_columns_on_side_planes() -> None:
    points, triangles = box((0.0, 0.0, 0.0), (1.0, 1.0, 0.5), 4, outward=False)
    heights = points[triangles][:, :, 2]
    open_box = triangles[~np.all(np.isclose(heights, 0.5), axis=1)]
    mesh = wall(points, open_box)
    cells = mesh.entity_set(2)
    floor = np.flatnonzero(np.all(np.isclose(points[open_box][:, :, 2], 0.0), axis=1))
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.asarray(cells.entity_ids)[floor],
    )
    request = phx.meshing.BoundaryLayerControl(
        scope, SCHEDULE, route=phx.meshing.BoundaryLayerRoute.ADVANCING
    )
    result = phx.meshing.prepare_boundary_layers(mesh, request)
    assert result.evidence.rim_vertex_count == 16
    coordinates = np.asarray(result.mesh.coordinates)
    assert np.all((coordinates[:, :2] >= 0.0) & (coordinates[:, :2] <= 1.0))
    assert np.count_nonzero(coordinates[:, 0] == 0.0) == 5 * 4
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses,
        SCHEDULE.thicknesses,
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    assert not result.closed_cap
    _assert_certified(result)


def test_opposing_channel_refusal_keeps_all_collision_locations() -> None:
    mesh = channel(0.05)
    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.prepare_boundary_layers(mesh, control(mesh))
    assert failure.value.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
    assert len(failure.value.entity_ids) == 50
    assert len(failure.value.locations) == 50


def _channel_layers(
    collision: phx.meshing.BoundaryLayerCollisionPolicy,
) -> phx.meshing.BoundaryLayerMesh:
    mesh = channel(0.05)
    result = phx.meshing.prepare_boundary_layers(
        mesh, control(mesh, collision=collision, minimum_thickness_fraction=0.2)
    )
    assert result.evidence.collision_policy is collision
    assert result.evidence.predicted_collision_vertex_count == 50
    _assert_certified(result)
    return result


def test_channel_thickness_reduction_keeps_growth_and_active_support() -> None:
    result = _channel_layers(phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS)
    evidence = result.evidence
    heights = np.asarray(result.mesh.coordinates)[:, 2]
    assert 0.2 <= evidence.minimum_scale < 1.0
    assert evidence.reduced_vertex_count == 50
    assert np.max(heights[heights < 0.025]) < 0.025
    np.testing.assert_allclose(
        evidence.achieved_growth_rates, 1.2, rtol=RIDGE_RELATIVE_TOLERANCE
    )
    assert np.all(np.asarray(evidence.layer_active))


def test_channel_local_termination_keeps_nan_inactive_intervals() -> None:
    result = _channel_layers(phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY)
    evidence = result.evidence
    assert evidence.terminated_vertex_count == 50
    assert dict(evidence.cell_counts) == {"prism": 2 * 32}
    achieved = np.asarray(evidence.achieved_thicknesses)
    active = np.asarray(evidence.layer_active)
    assert achieved[0] == pytest.approx(0.01)
    np.testing.assert_array_equal(active, (True, False, False))
    np.testing.assert_array_equal(np.isnan(achieved), ~active)
    assert np.all(np.isnan(evidence.achieved_growth_rates))


def test_channel_merge_retains_the_shared_midsurface_without_a_free_cap() -> None:
    result = _channel_layers(phx.meshing.BoundaryLayerCollisionPolicy.MERGE)
    evidence = result.evidence
    heights = np.asarray(result.mesh.coordinates)[:, 2]
    assert evidence.merged_vertex_count == 50
    assert evidence.minimum_scale == pytest.approx(0.025 / SCHEDULE.total_thickness)
    assert np.any(np.isclose(heights, 0.025))
    assert result.cap is None


@pytest.mark.parametrize(
    ("thicknesses", "remaining_core"),
    (((0.05, 0.10), True), ((0.25, 0.75), False)),
    ids=("slab-and-core", "entire-solid-slab"),
)
def test_native_cad_extrusion_preserves_physical_wall_cap_and_volume(
    tmp_path: Path,
    thicknesses: tuple[float, ...],
    remaining_core: bool,
) -> None:
    source, request = cad_request(remaining_core)
    np.testing.assert_array_equal(request.schedule.thicknesses, thicknesses)
    destination = tmp_path / "native-extrusion.brep"
    result = phx.meshing.prepare_boundary_layer_extrusion(
        source,
        request,
        destination=destination,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    published, sweep = result.source, result.control
    domain = phx.geometry.MeshingDomain.from_brep(published)
    total = sum(thicknesses)
    assert sweep.route is phx.meshing.BoundaryLayerRoute.EXACT_SWEEP
    np.testing.assert_array_equal(sweep.schedule.thicknesses, thicknesses)
    np.testing.assert_allclose(
        result.slab_volumes, (6.0 * total,), rtol=PLANAR_RELATIVE_TOLERANCE
    )
    # Existing CAD arrangement-volume residual bound in the archived slab oracle.
    assert result.maximum_relative_volume_residual <= RIDGE_RELATIVE_TOLERANCE
    assert len(result.layer_solid_ids) == 1
    assert len(result.core_solid_ids) == int(remaining_core)
    assert sweep.volume_scope is not None and sweep.cap_scope is not None
    volume_rows = domain.resolve_indices(
        3, np.asarray(sweep.volume_scope.entity_ids, dtype=np.int64)
    )
    assert (
        tuple(sorted(domain.region_source_indices[row] for row in volume_rows))
        == result.layer_solid_ids
    )
    assert sweep.wall_scope.source_revision == published.source_revision
    assert sweep.cap_scope.source_revision == published.source_revision
    assert published.source_id == str(destination.resolve())
    wall_rows = domain.resolve_indices(
        2, np.asarray(sweep.wall_scope.entity_ids, dtype=np.int64)
    )
    for face in (domain.source_indices[2][row] for row in wall_rows):
        np.testing.assert_allclose(
            face_points(published, face)[:, 2],
            0.0,
            atol=CAD_COORDINATE_ABSOLUTE_TOLERANCE,
            rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
        )
    cap_rows = domain.resolve_indices(
        2, np.asarray(sweep.cap_scope.entity_ids, dtype=np.int64)
    )
    for face in (domain.source_indices[2][row] for row in cap_rows):
        np.testing.assert_allclose(
            face_points(published, face)[:, 2],
            total,
            atol=CAD_COORDINATE_ABSOLUTE_TOLERANCE,
            rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
        )
        assert len(published.topology.face_solids[face]) == 1 + int(remaining_core)


def test_native_cad_extrusion_keeps_the_remaining_core_bounds(tmp_path: Path) -> None:
    source, request = cad_request(True)
    result = phx.meshing.prepare_boundary_layer_extrusion(
        source,
        request,
        destination=tmp_path / "core.brep",
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    assert len(result.core_solid_ids) == 1
    core_points = np.concatenate(
        [
            face_points(result.source, face)
            for face in result.source.topology.solid_faces[result.core_solid_ids[0]]
        ]
    )
    np.testing.assert_allclose(
        np.min(core_points, axis=0),
        (0.0, 0.0, 0.15),
        atol=CAD_COORDINATE_ABSOLUTE_TOLERANCE,
        rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        np.max(core_points, axis=0),
        (2.0, 3.0, 1.0),
        atol=CAD_COORDINATE_ABSOLUTE_TOLERANCE,
        rtol=NUMPY_REFERENCE_RELATIVE_TOLERANCE,
    )


@pytest.mark.parametrize(
    ("thickness", "opposite_walls", "reason", "field"),
    (
        (1.25, False, "leaves its controlled solid", "BoundaryLayerControl.volume_scope"),
        (
            0.75,
            True,
            "slabs of different walls overlap",
            "BoundaryLayerControl.wall_scope",
        ),
    ),
    ids=("outside-controlled-solid", "overlapping-wall-slabs"),
)
def test_native_cad_extrusion_refuses_invalid_slabs_atomically(
    tmp_path: Path,
    thickness: float,
    opposite_walls: bool,
    reason: str,
    field: str,
) -> None:
    source, request, expected_reason = invalid_cad_request(opposite_walls)
    assert expected_reason == reason
    np.testing.assert_array_equal(request.schedule.thicknesses, (thickness,))
    original_id = source.model_id
    destination = tmp_path / "protected.brep"
    destination.write_bytes(b"accepted artifact")
    with pytest.raises(phx.meshing.MeshingFailure, match=field) as failure:
        phx.meshing.prepare_boundary_layer_extrusion(
            source, request, destination=destination, overwrite=True
        )
    assert reason in str(failure.value)
    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
    assert source.model_id == original_id
    assert destination.read_bytes() == b"accepted artifact"
    assert not tuple(destination.parent.glob(f".{destination.name}.extrusion-*"))


@pytest.mark.parametrize(
    "periodicity",
    ("translation", "partial", "rotation"),
    ids=("translation", "partial", "rotation"),
)
@pytest.mark.parametrize(
    "quadrilateral", (False, True), ids=("triangle", "quadrilateral")
)
def test_periodic_layers_preserve_winding_cap_and_physical_schedule(
    periodicity: str,
    quadrilateral: bool,
) -> None:
    mesh, request, identification, topology, area, transition_height = periodic_wall(
        periodicity, quadrilateral
    )
    result = phx.meshing.prepare_boundary_layers(mesh, request)
    assert result.mesh.periodic_topology is not None
    assert result.cap is not None and result.cap.periodic_topology is not None
    for carrier in (result.mesh, result.cap):
        descriptor = carrier.periodic_topology
        assert descriptor is not None
        assert isinstance(descriptor.cell, phx.discretization.PeriodicIsometryGroup)
        assert descriptor.cell.group_id == identification.group_id
        incidences = [
            incidence.scipy_boundary().toarray()
            for incidence in descriptor.quotient.incidences
        ]
        for lower, upper in zip(incidences, incidences[1:]):
            np.testing.assert_array_equal(lower @ upper, 0.0)
    assert result.closed_cap is (periodicity == "translation")
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses,
        request.schedule.thicknesses,
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    np.testing.assert_allclose(
        phx.discretization.periodic_orbit_measures(result.mesh).total_measure,
        area * (sum(request.schedule.thicknesses) + transition_height / 3.0),
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    assert result.source_wall.periodic_topology is not None
    assert (
        result.source_wall.periodic_topology.periodic_topology_id
        == topology.periodic_topology_id
    )


def test_periodic_layers_preserve_a_convex_ridge_crossing_the_seam() -> None:
    mesh, request = ridge_wall()
    result = phx.meshing.prepare_boundary_layers(mesh, request)
    assert result.evidence.convex_ridge_count == 1
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses,
        request.schedule.thicknesses,
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    assert result.cap is not None and result.cap.periodic_topology is not None
    boundary = result.cap.periodic_topology.quotient.entities(1).subset("boundary")
    assert np.count_nonzero(np.asarray(boundary.mask)) == 2
    assert result.mesh.periodic_topology is not None
    first, second, third = (
        incidence.scipy_boundary().toarray()
        for incidence in result.mesh.periodic_topology.quotient.incidences
    )
    np.testing.assert_array_equal(first @ second, 0.0)
    np.testing.assert_array_equal(second @ third, 0.0)


@pytest.mark.parametrize("periodic", (False, True), ids=("nonperiodic", "periodic"))
def test_exact_sweep_authors_independent_measured_reference_stations(
    periodic: bool,
) -> None:
    fixture = exact_sweep_fixture(periodic)
    reference = phx.meshing.prepare_boundary_layers(
        fixture.wall,
        fixture.control,
        sweep_control=fixture.sweep,
        wall_association=fixture.association,
        source_domain=fixture.domain,
    )
    expected = fixture.points[None] + fixture.stations[:, None, None] * np.asarray(
        (0.0, 0.0, 1.0), dtype=np.float64
    )
    np.testing.assert_array_equal(
        np.asarray(reference.mesh.coordinates).view(np.uint64),
        expected.reshape(-1, 3).view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.asarray(fixture.physical.mesh.coordinates).view(np.uint64),
        fixture.actual_points.view(np.uint64),
    )
    np.testing.assert_array_equal(fixture.physical.column_index, fixture.actual_columns)
    np.testing.assert_array_equal(reference.layer_index, fixture.physical.layer_index)
    np.testing.assert_array_equal(reference.column_index, fixture.physical.column_index)
    np.testing.assert_allclose(
        reference.evidence.column_thicknesses,
        np.broadcast_to(np.diff(fixture.stations)[:, None], (2, 6)),
        rtol=PLANAR_RELATIVE_TOLERANCE,
    )
    assert (
        reference.source_wall is fixture.wall
        and reference.source_domain is fixture.domain
    )
    assert (
        reference.control is fixture.control
        and reference.wall_association is fixture.association
    )
    assert reference.result_id != fixture.physical.result_id
    assert reference.validity.certified_valid_count == 8
    assert reference.cap is not None
    for carrier in (reference.mesh, reference.cap):
        assert (carrier.periodic_topology is not None) == periodic
    qualify_hybrid_reference_cells(reference.mesh)


@pytest.mark.parametrize("periodic", (False, True), ids=("nonperiodic", "periodic"))
def test_exact_sweep_refuses_a_perturbed_independent_cap(periodic: bool) -> None:
    fixture = exact_sweep_fixture(periodic, perturb_cap=True)
    with pytest.raises(ValueError, match="BoundaryLayerControl.wall_scope"):
        phx.meshing.prepare_boundary_layers(
            fixture.wall,
            fixture.control,
            sweep_control=fixture.sweep,
            wall_association=fixture.association,
            source_domain=fixture.domain,
        )
