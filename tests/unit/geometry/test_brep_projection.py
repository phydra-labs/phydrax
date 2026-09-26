import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder, BRepPrimAPI_MakeSphere
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

import phydrax as phx


_CONTRACT = phx.SpatialCoordinateContract.si()
_STATUS = phx.geometry.BRepProjectionStatus
_FACE = phx.geometry.BRepEntityDimension.FACE
_EDGE = phx.geometry.BRepEntityDimension.EDGE


def _bound(shape, embedding=None):
    model = phx.geometry.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return model, phx.geometry.prepare_brep_projection(model, shape, embedding=embedding)


def _project(projection, points, dimension, index):
    points = np.asarray(points, dtype=np.float64)
    count = points.shape[0]
    return projection.project(
        points, np.full(count, int(dimension)), np.full(count, int(index))
    )


def _circle_edge(projection, height):
    """Index of the closed circular edge at ``height`` (edges keep OCCT order)."""
    vertices = np.asarray(projection.vertex_points)
    edge_vertices = np.asarray(projection.edge_vertices)
    closed = np.flatnonzero(np.asarray(projection.edge_closed))
    return int(closed[np.isclose(vertices[edge_vertices[closed, 0], 2], height)][0])


def test_cylinder_face_projection_reports_parameters_frames_and_residuals():
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    _, projection = _bound(shape)
    result = _project(projection, [[0.3, 0.4, 1.0], [1.5, 1.5, 0.25]], _FACE, 0)

    np.testing.assert_allclose(result.points, [[0.6, 0.8, 1.0], [2**-0.5, 2**-0.5, 0.25]])
    np.testing.assert_allclose(
        result.parameters, [[np.arctan2(0.8, 0.6), 1.0], [np.pi / 4, 0.25]]
    )
    np.testing.assert_allclose(result.residuals, [0.5, np.hypot(1.5, 1.5) - 1.0])
    assert np.all(np.asarray(result.status) == _STATUS.UNIQUE)
    # The lateral face's oriented normal points out of the solid.
    np.testing.assert_allclose(
        result.normals, [[0.6, 0.8, 0.0], [2**-0.5, 2**-0.5, 0.0]], atol=1e-12
    )
    tangents = np.asarray(result.tangents)
    np.testing.assert_allclose(
        np.einsum("nka,na->nk", tangents, result.normals), 0.0, atol=1e-12
    )
    np.testing.assert_allclose(np.linalg.norm(tangents, axis=-1), 1.0)
    assert result.entity_ids()[0] == f"{projection.source_revision}:face:0"


def test_cylinder_projection_flags_seams_and_continua_of_minima():
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    _, projection = _bound(shape)
    face = _project(projection, [[2.0, 0.0, 0.5], [0.0, 0.0, 1.0]], _FACE, 0)
    rim = _circle_edge(projection, 2.0)
    edge = _project(
        projection, [[0.0, 2.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, 2.0]], _EDGE, rim
    )

    # A point projecting onto the periodic seam, and a point on the axis whose
    # closest points form a whole circle.
    assert np.asarray(face.status).tolist() == [_STATUS.SEAM, _STATUS.AMBIGUOUS]
    np.testing.assert_allclose(face.residuals, [1.0, 1.0])
    assert np.asarray(edge.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.AMBIGUOUS,
    ]
    np.testing.assert_allclose(np.asarray(edge.parameters)[0, 0], np.pi / 2)
    np.testing.assert_allclose(edge.residuals, [1.0, 2.0, 1.0])
    np.testing.assert_allclose(
        np.abs(np.asarray(edge.tangents)[0, 0]), [1.0, 0.0, 0.0], atol=1e-12
    )


def test_sphere_projection_parameters_poles_and_center():
    shape = BRepPrimAPI_MakeSphere(1.0).Shape()
    _, projection = _bound(shape)
    query = np.asarray([[0.3, 0.4, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.0]])
    result = _project(projection, query, _FACE, 0)

    radius = np.linalg.norm(query[0])
    np.testing.assert_allclose(result.points[0], query[0] / radius)
    np.testing.assert_allclose(
        result.parameters[0], [np.arctan2(0.4, 0.3), np.arcsin(1.0 / radius)]
    )
    np.testing.assert_allclose(result.residuals, [radius - 1.0, 1.0, 1.0])
    # The pole is a singular point of the chart; the center is equidistant.
    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.AMBIGUOUS,
    ]


def test_classification_picks_the_lowest_dimensional_entity_within_tolerance():
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    _, projection = _bound(shape)
    points = np.asarray(
        [[0.0, 1.0, 1.0], [0.0, 1.0, 2.0], [1.0, 0.0, 2.0], [0.2, 0.1, 1.0]]
    )
    result = projection.classify(points, tolerance=1e-9)
    located = projection.locate_solids(points)
    rim = _circle_edge(projection, 2.0)

    assert np.asarray(result.dimensions).tolist() == [2, 1, 0, -1]
    assert int(result.indices[1]) == rim
    assert np.asarray(result.status)[-1] == _STATUS.FAILED
    assert np.asarray(located.status).tolist() == [
        _STATUS.AMBIGUOUS,
        _STATUS.AMBIGUOUS,
        _STATUS.AMBIGUOUS,
        _STATUS.UNIQUE,
    ]
    corner = int(np.asarray(projection.edge_vertices)[rim, 0])
    assert np.asarray(
        projection.contains([2, 1, 1], [0, rim, rim], [1, 0, 2], [rim, corner, 0])
    ).tolist() == [True, True, False]


def test_projection_binds_the_exact_source_revision():
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    model = phx.geometry.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )
    remodeled = phx.geometry.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.3,
    )

    # Tessellating the shape does not change its exact-geometry revision.
    assert remodeled.source_revision == model.source_revision
    with pytest.raises(ValueError, match="revision digest"):
        phx.geometry.prepare_brep_projection(
            model, BRepPrimAPI_MakeCylinder(1.0, 3.0).Shape()
        )


def test_planar_embedding_projects_two_dimensional_mesh_coordinates():
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((-2, -2), (2, -2), (2, 2), (-2, 2)):
        polygon.Add(gp_Pnt(x, y, 0))
    polygon.Close()
    square = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    circle = BRepBuilderAPI_MakeEdge(
        gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.0)
    ).Edge()
    disk = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(circle).Wire()).Face()
    plate = BRepAlgoAPI_Cut(square, disk).Shape()
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    _, projection = _bound(plate, embedding)
    hole = _circle_edge(projection, 0.0)
    edge = _project(projection, [[0.5, 0.5]], _EDGE, hole)
    face = _project(projection, [[0.5, 0.5], [1.5, 0.0]], _FACE, 0)

    np.testing.assert_allclose(edge.points, [[2**-0.5, 2**-0.5]])
    np.testing.assert_allclose(edge.residuals, [1.0 - 2**-0.5])
    # Inside the hole the trimmed face's closest point lies on the hole boundary.
    np.testing.assert_allclose(face.points, [[2**-0.5, 2**-0.5], [1.5, 0.0]])
    np.testing.assert_allclose(face.residuals, [1.0 - 2**-0.5, 0.0], atol=1e-12)
    assert np.all(np.isnan(np.asarray(face.normals)))
