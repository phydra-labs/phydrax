#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.geometry.surface._contracts import (
    InterfaceSide,
    SurfaceAuditPolicy,
    SurfaceInterface,
    SurfaceMetadata,
    SurfacePreparationError,
    SurfacePreparationStatus,
)
from phydrax.geometry.surface._model import SurfaceModel


def _metadata(*, tags=()):
    return SurfaceMetadata(
        source_id="unit-tetrahedron",
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("unit-test", "native-triangles"),
        cell_tags=tags,
    )


def _tetrahedron():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    outward_faces = np.asarray(
        ((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32
    )
    return points, outward_faces


def test_closed_and_open_surfaces_prepare_with_computed_classification():
    points, faces = _tetrahedron()
    closed = SurfaceModel.from_triangles(
        points,
        faces,
        _metadata(tags=("wall", "wall", "wall", "wall")),
        cell_global_ids=np.asarray((11, 12, 13, 14), dtype=np.int64),
    ).prepare(SurfaceAuditPolicy(require_closed=True, require_outward_orientation=True))

    assert closed.audit.closed
    assert closed.audit.classification == "closed"
    assert closed.audit.component_count == 1
    assert closed.audit.boundary_edge_count == 0
    np.testing.assert_array_equal(
        closed.chart_mapping.chart_ids, np.asarray((0, 1, 2, 3), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        closed.chart_mapping.cell_global_ids,
        np.asarray((11, 12, 13, 14), dtype=np.int64),
    )

    open_model = SurfaceModel.from_triangles(points[:3], faces[:1], _metadata())
    open_surface = open_model.prepare()
    assert open_surface.audit.open
    assert open_surface.audit.classification == "open"
    assert open_surface.audit.boundary_edge_count == 3
    assert open_surface.audit.boundary_loop_count == 1
    with pytest.raises(SurfacePreparationError) as open_error:
        open_model.prepare(SurfaceAuditPolicy(require_closed=True))
    assert open_error.value.status is SurfacePreparationStatus.AUDIT_REJECTED

    component_points = np.vstack((points[:3], points[:3] + np.asarray((3.0, 0.0, 0.0))))
    component_surface = SurfaceModel.from_triangles(
        component_points,
        np.asarray(((0, 2, 1), (3, 5, 4)), dtype=np.int32),
        _metadata(),
    ).prepare()
    assert component_surface.classification == "component"
    assert component_surface.audit.component_count == 2
    assert component_surface.audit.component_classification == ("open", "open")
    assert component_surface.audit.boundary_loop_count == 2


def test_audit_rejects_degenerate_and_capacity_exhausted_surfaces():
    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    model = SurfaceModel.from_triangles(
        points, np.asarray(((0, 1, 2),), dtype=np.int32), _metadata()
    )
    report = model.audit()

    assert not bool(report.metric_valid)
    assert not bool(report.valid)
    with pytest.raises(SurfacePreparationError) as error:
        model.prepare()
    assert error.value.status is SurfacePreparationStatus.AUDIT_REJECTED
    assert (
        error.value.report is report or error.value.report.report_id == report.report_id
    )

    valid_points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    valid = SurfaceModel.from_triangles(
        valid_points, np.asarray(((0, 1, 2),), dtype=np.int32), _metadata()
    )
    with pytest.raises(SurfacePreparationError) as capacity_error:
        valid.prepare(SurfaceAuditPolicy(maximum_vertices=2))
    assert capacity_error.value.status is SurfacePreparationStatus.AUDIT_REJECTED
    assert not bool(capacity_error.value.report.capacity_valid)


def test_malformed_nonfinite_and_unrepaired_orientation_fail_typed():
    points, faces = _tetrahedron()
    nonfinite = points.copy()
    nonfinite[0, 0] = np.nan
    with pytest.raises(SurfacePreparationError) as nonfinite_error:
        SurfaceModel.from_triangles(nonfinite, faces, _metadata())
    assert nonfinite_error.value.status is SurfacePreparationStatus.NONFINITE_GEOMETRY

    with pytest.raises(SurfacePreparationError) as malformed_error:
        SurfaceModel.from_triangles(
            points,
            np.asarray(((0, 0, 1),), dtype=np.int32),
            _metadata(),
        )
    assert malformed_error.value.status is SurfacePreparationStatus.INVALID_INPUT

    nonmanifold_points = np.vstack((points, np.asarray(((0.0, -1.0, 0.0),))))
    with pytest.raises(SurfacePreparationError) as nonmanifold_error:
        SurfaceModel.from_triangles(
            nonmanifold_points,
            np.asarray(((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int32),
            _metadata(),
        )
    assert nonmanifold_error.value.status is SurfacePreparationStatus.NONMANIFOLD_TOPOLOGY

    inconsistent = faces.copy()
    inconsistent[0] = inconsistent[0, [0, 2, 1]]
    with pytest.raises(SurfacePreparationError) as orientation_error:
        SurfaceModel.from_triangles(points, inconsistent, _metadata())
    assert (
        orientation_error.value.status
        is SurfacePreparationStatus.INCONSISTENT_ORIENTATION
    )

    repaired = SurfaceModel.from_triangles(
        points,
        inconsistent,
        _metadata(),
        repair_orientation=True,
        orient_closed_outward=True,
    ).prepare(SurfaceAuditPolicy(require_outward_orientation=True))
    assert repaired.model.orientation_repair is not None
    assert (
        repaired.model.orientation_repair.source_topology_id
        != repaired.model.orientation_repair.repaired_topology_id
    )
    np.testing.assert_array_equal(
        repaired.model.orientation_repair.source_face_indices,
        np.arange(faces.shape[0], dtype=np.int64),
    )
    assert bool(repaired.audit.orientation_consistent)


def test_selections_interfaces_and_refresh_remain_exactly_topology_bound():
    points, faces = _tetrahedron()
    model = SurfaceModel.from_triangles(
        points,
        faces,
        _metadata(tags=("a", "b", "c", "d")),
        vertex_global_ids=np.asarray((101, 102, 103, 104), dtype=np.int64),
        cell_global_ids=np.asarray((11, 12, 13, 14), dtype=np.int64),
    )
    support = model.bind_selection("dielectric", (12, 14), role="interface")
    interface = SurfaceInterface(
        "material-jump",
        support,
        minus_region="inside",
        plus_region="outside",
    )
    model = model.with_selection(support).with_interface(interface)
    realization = model.prepare()

    np.testing.assert_array_equal(
        realization.chart_ids_for(support), np.asarray((1, 3), dtype=np.int32)
    )
    np.testing.assert_array_equal(
        realization.selection_atlas("dielectric").source_entity_ids,
        np.asarray((1, 3), dtype=np.int32),
    )
    np.testing.assert_array_equal(
        realization.interface_atlas("material-jump").source_entity_ids,
        np.asarray((1, 3), dtype=np.int32),
    )
    assert realization.audit.classification == "interface"
    assert interface.region(InterfaceSide.MINUS) == "inside"
    assert interface.region(InterfaceSide.PLUS) == "outside"

    refreshed = realization.refresh(points * 2.0, numeric_version="deformed-1")
    assert refreshed.mesh.topology_id == realization.mesh.topology_id
    assert refreshed.mesh.geometry_id != realization.mesh.geometry_id
    assert refreshed.chart_mapping.mapping_id == realization.chart_mapping.mapping_id
    np.testing.assert_array_equal(
        refreshed.chart_mapping.cell_global_ids,
        realization.chart_mapping.cell_global_ids,
    )
