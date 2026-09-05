#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np

from phydrax import SpatialCoordinateContract
from phydrax.geometry.surface._contracts import SurfaceMetadata
from phydrax.geometry.surface._intersection import PlaneSectionStatus
from phydrax.geometry.surface._model import SurfaceModel


def _realization():
    points = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    )
    faces = np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32)
    metadata = SurfaceMetadata(
        source_id="section-tetrahedron",
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("unit-test",),
    )
    return SurfaceModel.from_triangles(
        points,
        faces,
        metadata,
        vertex_global_ids=np.asarray((101, 102, 103, 104), dtype=np.int64),
        cell_global_ids=np.asarray((11, 12, 13, 14), dtype=np.int64),
    ).prepare()


def test_plane_section_returns_deterministic_loop_and_exact_provenance():
    realization = _realization()

    first = realization.intersect_plane((0.0, 0.0, 0.25), (0.0, 0.0, 1.0))
    second = realization.intersect_plane((8.0, -3.0, 0.25), (0.0, 0.0, -2.0))

    assert first.status is PlaneSectionStatus.RESOLVED
    assert len(first.loops) == 1
    assert first.loops[0].points.shape == (3, 3)
    assert first.loops[0].source_cell_global_ids.shape == (3,)
    assert set(np.asarray(first.loops[0].source_cell_global_ids).tolist()) == {
        12,
        13,
        14,
    }
    assert first.evidence.chart_mapping_id == realization.chart_mapping.mapping_id
    assert first.section_id == second.section_id
    np.testing.assert_allclose(first.loops[0].points, second.loops[0].points)
    np.testing.assert_array_equal(
        first.loops[0].source_edge_vertex_global_ids,
        second.loops[0].source_edge_vertex_global_ids,
    )


def test_plane_section_reports_typed_unresolved_vertex_and_open_chain():
    realization = _realization()
    vertex_contact = realization.intersect_plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))

    assert vertex_contact.status is PlaneSectionStatus.UNRESOLVED_VERTEX_CONTACT
    assert vertex_contact.unresolved
    assert vertex_contact.loops == ()

    points = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    metadata = SurfaceMetadata(
        source_id="open-triangle",
        source_revision="r1",
        coordinate_contract=SpatialCoordinateContract.si(),
        provenance=("unit-test",),
    )
    open_surface = SurfaceModel.from_triangles(
        points,
        np.asarray(((0, 1, 2),), dtype=np.int32),
        metadata,
    ).prepare()
    open_chain = open_surface.intersect_plane((0.25, 0.0, 0.0), (1.0, 0.0, 0.0))

    assert open_chain.status is PlaneSectionStatus.UNRESOLVED_OPEN_CHAIN
    assert open_chain.unresolved
    assert open_chain.evidence.segment_count == 1
