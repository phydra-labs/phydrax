from typing import Any

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


def _bound(shape: Any, embedding: Any = None) -> Any:
    model = phx.interchange.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.2,
    )
    return model, phx.interchange.prepare_occt_projection(
        model, shape, embedding=embedding
    )


def _project(projection: Any, points: Any, dimension: Any, index: Any) -> Any:
    points = np.asarray(points, dtype=np.float64)
    count = points.shape[0]
    return projection.project(
        points, np.full(count, int(dimension)), np.full(count, int(index))
    )


def _circle_edge(projection: Any, height: Any) -> Any:
    """Index of the closed circular edge at ``height`` (edges keep OCCT order)."""
    vertices = np.asarray(projection.vertex_points)
    edge_vertices = np.asarray(projection.edge_vertices)
    closed = np.flatnonzero(np.asarray(projection.edge_closed))
    return int(closed[np.isclose(vertices[edge_vertices[closed, 0], 2], height)][0])


def test_brep_projection_scenario_1() -> None:
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


def test_brep_projection_scenario_2() -> None:
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    model = phx.interchange.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )
    remodeled = phx.interchange.model_from_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        linear_deflection=0.05,
        angular_deflection=0.3,
    )

    # Tessellating the shape does not change its exact-geometry revision.
    assert remodeled.source_revision == model.source_revision
    with pytest.raises(ValueError, match="revision digest"):
        phx.interchange.prepare_occt_projection(
            model, BRepPrimAPI_MakeCylinder(1.0, 3.0).Shape()
        )
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


def test_native_projection_agrees_with_occt_on_independent_samples() -> None:
    _, occt = _bound(BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape())
    model = phx.geometry.brep_cylinder(1.0, 2.0, coordinate_contract=_CONTRACT)
    native = phx.geometry.prepare_brep_projection(model)
    points = np.random.default_rng(7).uniform(
        (-2.0, -2.0, -0.5), (2.0, 2.0, 2.5), (48, 3)
    )
    lateral = model.physical_tags.index("cylinder")
    reference = _project(occt, points, _FACE, 0)
    result = _project(native, points, _FACE, lateral)
    reference_unique = np.asarray(reference.status) == _STATUS.UNIQUE
    native_resolved = np.isin(
        np.asarray(result.status),
        (_STATUS.UNIQUE, _STATUS.SEAM),
    )
    assert np.any(reference_unique)
    assert np.all(native_resolved[reference_unique])

    # OCCT is an independent physical oracle where it reports a unique point.
    # Native results additionally preserve the cylinder's nonunique seam chart.
    np.testing.assert_allclose(
        np.asarray(result.residuals)[reference_unique],
        np.asarray(reference.residuals)[reference_unique],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(result.points)[reference_unique],
        np.asarray(reference.points)[reference_unique],
        atol=1e-9,
    )
    np.testing.assert_allclose(
        np.asarray(result.normals)[reference_unique],
        np.asarray(reference.normals)[reference_unique],
        atol=1e-9,
    )
    classified = native.classify(points, tolerance=0.3)
    reference_classified = occt.classify(points, tolerance=0.3)
    np.testing.assert_array_equal(classified.dimensions, reference_classified.dimensions)
    np.testing.assert_allclose(
        classified.residuals, reference_classified.residuals, atol=1e-9
    )
    located = native.locate_solids(points)
    reference_located = occt.locate_solids(points)
    np.testing.assert_array_equal(located.status, reference_located.status)
