import io
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
    BRepBuilderAPI_Transform,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec
from OCP.TopoDS import TopoDS_Compound

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.geometry._source_identity import source_entity_id
from phydrax.geometry.brep._assembly import (
    BRepAffinePlacement,
    BRepAssemblyDefinition,
    BRepAssemblyGraph,
    BRepAssemblyUse,
)
from phydrax.geometry.brep._constructors import (
    brep_box,
    brep_planar_face,
    brep_sphere,
    brep_torus,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileLoop,
    ProfilePlane,
)
from phydrax.geometry.brep._model import BRepGeometry, BRepModel
from phydrax.interchange import import_occt_shape, prepare_occt_projection


_CONTRACT = phx.SpatialCoordinateContract.si()
_STATUS = phx.geometry.BRepProjectionStatus
_FACE = phx.geometry.BRepEntityDimension.FACE
_EDGE = phx.geometry.BRepEntityDimension.EDGE


def _bound(shape: Any, embedding: Any = None) -> Any:
    model = import_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.05,
            angular_deflection=0.2,
        ),
    ).model
    return model, phx.geometry.prepare_brep_projection(model, embedding=embedding)


def _project(projection: Any, points: Any, dimension: Any, index: Any) -> Any:
    points = np.asarray(points, dtype=np.float64)
    count = points.shape[0]
    return projection.project(
        points, np.full(count, int(dimension)), np.full(count, int(index))
    )


def _circle_edge(projection: Any, height: Any) -> Any:
    """Index of the closed circular edge at ``height`` (edges keep OCCT order)."""
    geometry = projection.model.geometry
    vertices = np.asarray(geometry.vertex_points)
    edge_vertices = np.asarray(geometry.edge_vertices)
    closed = np.flatnonzero(edge_vertices[:, 0] == edge_vertices[:, 1])
    return int(closed[np.isclose(vertices[edge_vertices[closed, 0], 2], height)][0])


def test_brep_projection_scenario_1() -> None:
    # The projection is immutable, so one cylinder import serves every query.
    _, cylinder = _bound(BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape())
    projection = cylinder
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
    assert result.entity_ids()[0] == source_entity_id(
        projection.source_id,
        projection.source_revision,
        "face",
        0,
        occurrence_path=result.occurrence_paths[0],
    )
    projection = cylinder
    face = _project(projection, [[2.0, 0.0, 0.5], [0.0, 0.0, 1.0]], _FACE, 0)
    rim = _circle_edge(projection, 2.0)
    edge = _project(
        projection, [[0.0, 2.0, 2.0], [3.0, 0.0, 2.0], [0.0, 0.0, 2.0]], _EDGE, rim
    )

    # A point projecting onto the periodic seam, and a point on the axis whose
    # closest points form a whole circle.
    assert np.asarray(face.status).tolist() == [_STATUS.SEAM, _STATUS.NON_ISOLATED]
    np.testing.assert_allclose(face.residuals, [1.0, 1.0])
    assert np.asarray(edge.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.NON_ISOLATED,
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
        _STATUS.NON_ISOLATED,
    ]
    projection = cylinder
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
    corner = int(projection.model.geometry.edge_vertices[rim][0])
    assert np.asarray(
        projection.contains([2, 1, 1], [0, rim, rim], [1, 0, 2], [rim, corner, 0])
    ).tolist() == [True, True, False]


def test_native_sphere_projects_interior_pole_seam_and_center_points() -> None:
    # The native sphere is a revolved circle profile meeting the axis at its poles.
    projection = phx.geometry.prepare_brep_projection(
        brep_sphere(1.0, coordinate_contract=_CONTRACT)
    )
    query = np.asarray(
        [
            [0.3, 0.4, 0.5],
            [-0.5, 0.0, 0.3],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, -0.5],
            [0.0, 0.6, 0.8],
            [0.0, 0.0, 0.0],
        ]
    )
    result = _project(projection, query, _FACE, 0)
    radii = np.linalg.norm(query[:5], axis=1)
    surface = query[:5] / radii[:, None]

    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.SEAM,
        _STATUS.SEAM,
        _STATUS.NON_ISOLATED,
    ]
    np.testing.assert_allclose(result.points[:5], surface, atol=1e-12)
    np.testing.assert_allclose(result.residuals, [*np.abs(radii - 1.0), 1.0], atol=1e-12)
    # The oriented normal points out of the solid, also at the poles where the
    # chart's tangent cross product vanishes.
    np.testing.assert_allclose(result.normals[:5], surface, atol=1e-12)


def test_native_torus_projects_interior_and_both_seam_points() -> None:
    # A full torus face is one period in each direction of its doubly periodic chart.
    projection = phx.geometry.prepare_brep_projection(
        brep_torus(2.0, 0.5, coordinate_contract=_CONTRACT)
    )
    query = np.asarray(
        [
            [1.5, 0.0, 0.3],
            [-1.0, -2.0, -0.4],
            [0.0, 2.3, 0.2],
            [-3.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
        ]
    )
    result = _project(projection, query, _FACE, 0)
    planar = np.linalg.norm(query[:5, :2], axis=1)
    tube_centers = np.column_stack(
        (2.0 * query[:5, :2] / planar[:, None], np.zeros((5,)))
    )
    offsets = query[:5] - tube_centers
    distances = np.linalg.norm(offsets, axis=1)
    outward = offsets / distances[:, None]

    # The angle seam is the profile's half-plane (+y) and the profile seam is
    # the outer equator; the last surface point lies on both.
    assert np.asarray(result.status).tolist() == [
        _STATUS.UNIQUE,
        _STATUS.UNIQUE,
        _STATUS.SEAM,
        _STATUS.SEAM,
        _STATUS.SEAM,
        _STATUS.NON_ISOLATED,
    ]
    np.testing.assert_allclose(result.parameters[2, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.parameters[3, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(result.parameters[4], [0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(
        result.points[:5], tube_centers + 0.5 * outward, atol=1e-12
    )
    np.testing.assert_allclose(
        result.residuals, [*np.abs(distances - 0.5), 0.5], atol=1e-12
    )
    np.testing.assert_allclose(result.normals[:5], outward, atol=1e-12)


@pytest.mark.parametrize(
    ("model", "inside", "outside", "boundary"),
    [
        (
            brep_sphere(1.0, coordinate_contract=_CONTRACT),
            ((0.3, 0.4, 0.5), (0.0, 0.0, 0.0), (0.0, 0.0, -0.5), (-0.9, 0.0, 0.0)),
            ((0.0, 0.0, 2.0), (0.6, 0.6, 0.6), (0.0, 0.0, -1.5), (1.0, 0.0, 1.0)),
            ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),
        ),
        (
            brep_sphere(
                0.5,
                center=(0.25, -1.0, 2.0),
                axis=(0.0, 1.0, 0.0),
                coordinate_contract=_CONTRACT,
            ),
            ((0.25, -1.0, 2.0), (0.5, -1.25, 2.0), (0.25, -0.625, 2.0)),
            ((0.0, 0.0, 0.0), (0.25, -0.25, 2.0), (0.75, -0.5, 2.0)),
            ((0.25, -0.5, 2.0), (0.25, -1.5, 2.0), (0.75, -1.0, 2.0), (0.25, -1.0, 1.5)),
        ),
        (
            brep_torus(2.0, 0.5, coordinate_contract=_CONTRACT),
            ((2.0, 0.0, 0.0), (0.0, -2.0, 0.25), (1.75, 0.0, -0.125), (-2.25, 0.0, 0.0)),
            ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (2.0, 0.0, 0.75), (1.0, 0.0, 0.0)),
            ((2.5, 0.0, 0.0), (1.5, 0.0, 0.0), (0.0, 2.0, 0.5), (0.0, -2.0, -0.5)),
        ),
    ],
    ids=("sphere", "offset-sphere", "torus"),
)
def test_closed_revolution_containment_is_exact_with_unresolved_boundary(
    model: Any, inside: Any, outside: Any, boundary: Any
) -> None:
    # The meridian image (rho, h) of a point decides containment against the
    # exact profile circle; points on the surface are unresolved, not guessed.
    projection = phx.geometry.prepare_brep_projection(model)

    def contains(points: Any) -> list[bool | None]:
        return [
            projection._contains_solid(np.asarray(point, dtype=np.float64), 0)
            for point in points
        ]

    assert contains(inside) == [True] * len(inside)
    assert contains(outside) == [False] * len(outside)
    assert contains(boundary) == [None] * len(boundary)
    located = projection.locate_solids(np.asarray(inside, dtype=np.float64))
    assert np.asarray(located.status).tolist() == [_STATUS.UNIQUE] * len(inside)
    assert np.asarray(located.dimensions).tolist() == [3] * len(inside)


def test_capped_cylinder_containment_retries_non_generic_rays() -> None:
    # From the axis point (0, 0, 1) the first ray direction meets the cylinder
    # exactly on the top rim (height 1 + sqrt(2) * sqrt(3) / sqrt(6) = 2); that
    # ray is not decisive, the next transversal direction decides containment.
    model = import_occt_shape(
        BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape(),
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.1, angular_deflection=0.3
        ),
    ).model
    projection = phx.geometry.prepare_brep_projection(model)

    def contains(points: Any) -> list[bool | None]:
        return [
            projection._contains_solid(np.asarray(point, dtype=np.float64), 0)
            for point in points
        ]

    inside = ((0.0, 0.0, 1.0), (0.5, 0.0, 1.0), (0.0, 0.0, 0.5), (-0.3, 0.4, 1.9))
    outside = ((0.0, 0.0, 3.0), (0.0, 0.0, -1.0), (2.0, 0.0, 1.0), (0.0, 1.5, 2.0))
    boundary = ((0.0, 0.0, 2.0), (0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (0.5, 0.0, 2.0))
    assert contains(inside) == [True] * len(inside)
    assert contains(outside) == [False] * len(outside)
    assert contains(boundary) == [None] * len(boundary)
    located = projection.locate_solids(np.asarray(inside, dtype=np.float64))
    assert np.asarray(located.status).tolist() == [_STATUS.UNIQUE] * len(inside)


def test_brep_projection_scenario_2() -> None:
    shape = BRepPrimAPI_MakeCylinder(1.0, 2.0).Shape()
    model = import_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.1,
            angular_deflection=0.3,
        ),
    ).model
    remodeled = import_occt_shape(
        shape,
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.05,
            angular_deflection=0.3,
        ),
    ).model

    # Tessellating the shape does not change its exact-geometry revision.
    assert remodeled.source_revision == model.source_revision
    polygon = BRepBuilderAPI_MakePolygon()
    for x, y in ((-2, -2), (2, -2), (2, 2), (-2, 2)):
        polygon.Add(gp_Pnt(x, y, 0))
    polygon.Close()
    square = BRepBuilderAPI_MakeFace(polygon.Wire()).Face()
    circle = BRepBuilderAPI_MakeEdge(
        gp_Circ(gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), 1.0)
    ).Edge()
    disk = BRepBuilderAPI_MakeFace(BRepBuilderAPI_MakeWire(circle).Wire()).Face()
    BRepAlgoAPI = pytest.importorskip("OCP.BRepAlgoAPI")
    plate = BRepAlgoAPI.BRepAlgoAPI_Cut(square, disk).Shape()
    embedding = phx.geometry.PlanarEmbedding((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))
    _, projection = _bound(plate, embedding)
    hole = _circle_edge(projection, 0.0)
    edge = _project(projection, [[0.5, 0.5]], _EDGE, hole)
    face = _project(projection, [[0.5, 0.5], [1.5, 0.0]], _FACE, 0)

    np.testing.assert_allclose(edge.points, [[2**-0.5, 2**-0.5]])
    np.testing.assert_allclose(edge.residuals, [1.0 - 2**-0.5])
    # The planar result keeps the edge's defined tangent; only its second
    # frame slot is undefined.
    edge_tangents = np.asarray(edge.tangents)
    np.testing.assert_allclose(
        np.abs(edge_tangents[0, 0]), [2**-0.5, 2**-0.5], atol=1e-12
    )
    assert np.all(np.isnan(edge_tangents[0, 1]))
    # Inside the hole the trimmed face's closest point lies on the hole boundary.
    np.testing.assert_allclose(face.points, [[2**-0.5, 2**-0.5], [1.5, 0.0]])
    np.testing.assert_allclose(face.residuals, [1.0 - 2**-0.5, 0.0], atol=1e-12)
    assert np.all(np.isnan(np.asarray(face.normals)))


def _triangle_reference(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    """Project on the plane or an edge; independent NumPy constrained reference."""
    origin, second, third = triangle
    basis = np.column_stack((second - origin, third - origin))
    coordinates = np.linalg.lstsq(basis, point - origin, rcond=None)[0]
    barycentric = np.asarray(
        [1.0 - coordinates.sum(), coordinates[0], coordinates[1]], dtype=np.float64
    )
    if np.all(barycentric >= 0.0):
        return barycentric @ triangle
    candidates = []
    for start, stop in ((origin, second), (second, third), (third, origin)):
        edge = stop - start
        fraction = np.clip(np.dot(point - start, edge) / np.dot(edge, edge), 0.0, 1.0)
        candidates.append(start + fraction * edge)
    candidates_ = np.asarray(candidates, dtype=np.float64)
    return candidates_[np.argmin(np.sum((candidates_ - point) ** 2, axis=-1))]


def _mesh_reference(
    points: np.ndarray, vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    projections = np.asarray(
        [
            [_triangle_reference(point, triangle) for triangle in triangles]
            for point in points
        ],
        dtype=np.float64,
    )
    squared = np.sum((projections - points[:, None]) ** 2, axis=-1)
    order = np.argsort(squared, axis=-1, kind="stable")
    rows = np.arange(points.shape[0])
    return (
        projections[rows, order[:, 0]],
        order[:, 0],
        squared[rows, order[:, 0]],
        squared[rows, order[:, 1]],
    )


@pytest.fixture
def query_box_model() -> Any:
    return import_occt_shape(
        BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(),
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.1,
            angular_deflection=0.3,
        ),
    ).model


@pytest.mark.parametrize(
    "differentiable", [False, True], ids=["imported", "fixed-topology"]
)
def test_brep_query_nearest_matches_independent_faces(
    query_box_model: Any, differentiable: bool
) -> None:
    model = query_box_model
    source = (
        phx.geometry.FixedTopologyBRepSource(model)
        if differentiable
        else phx.geometry.BRepSource(model)
    )
    geometry = source.compile()
    # Include face interiors, a two-face tie, a corner and an interior query.
    points = np.asarray(
        [
            [-0.2, 0.3, 0.8],
            [0.2, -0.4, 1.3],
            [-0.2, -0.2, 1.0],
            [-0.1, -0.1, -0.1],
            [0.4, 0.8, 1.4],
        ],
        dtype=np.float64,
    )
    closest, face, first, _ = _mesh_reference(
        points, np.asarray(model.mesh_vertices), np.asarray(model.mesh_faces)
    )
    result = geometry.closest_point(jnp.asarray(points))
    np.testing.assert_allclose(result.closest_point, closest, atol=2e-14)
    # The imported source evaluates native face carriers and reports source
    # faces; the fixed-topology kernel queries its tessellation triangles.
    entities = face if differentiable else np.asarray(model.triangle_face_ids)[face]
    np.testing.assert_array_equal(result.source_entity_id, entities)
    np.testing.assert_allclose(
        np.abs(result.normal_coordinate), np.sqrt(first), atol=2e-14
    )
    triangles = np.asarray(model.mesh_vertices)[np.asarray(model.mesh_faces)[face]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    np.testing.assert_allclose(result.oriented_normal, normals, atol=2e-14)
    assert result.physical_geometry_id == model.source_revision
    if differentiable:
        assert result.represented_geometry_id == f"{model.model_id}:fixed-topology-query"
        assert not result.exact_to_physical
    else:
        assert result.represented_geometry_id == model.source_revision
        assert result.exact_to_physical


def test_brep_two_face_margin_matches_independent_reference(query_box_model: Any) -> None:
    model = query_box_model
    geometry = phx.geometry.FixedTopologyBRepSource(model).compile()
    points = np.asarray([[-0.02, 0.3, 0.8], [-0.2, -0.2, 1.0]], dtype=np.float64)
    closest, face, first, second = _mesh_reference(
        points, np.asarray(model.mesh_vertices), np.asarray(model.mesh_faces)
    )
    triangles = np.asarray(model.mesh_vertices)[np.asarray(model.mesh_faces)[face]]
    margins = []
    for point, projection, triangle, first_sq, second_sq in zip(
        points, closest, triangles, first, second, strict=True
    ):
        coordinates = np.linalg.lstsq(
            np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0])),
            projection - triangle[0],
            rcond=None,
        )[0]
        barycentric = np.asarray(
            [1.0 - coordinates.sum(), coordinates[0], coordinates[1]], dtype=np.float64
        )
        edge_minimum = min(
            np.linalg.norm(triangle[1] - triangle[0]),
            np.linalg.norm(triangle[2] - triangle[1]),
            np.linalg.norm(triangle[0] - triangle[2]),
        )
        feature = barycentric.min() * edge_minimum
        gap = np.sqrt(second_sq) - np.sqrt(first_sq)
        tolerance = 64.0 * np.finfo(np.float64).eps * max(np.abs(point).max(), 1.0)
        margins.append(min(feature, gap) if min(feature, gap) > tolerance else 0.0)
    result = geometry.closest_point(jnp.asarray(points))
    np.testing.assert_allclose(result.margin, margins, atol=2e-14)
    np.testing.assert_array_equal(result.unique, np.asarray(margins) > 0.0)
    np.testing.assert_array_equal(result.regular, result.unique)
    assert float(result.margin[0]) > 0.0
    assert float(result.margin[1]) == 0.0


def test_brep_imported_boundary_tolerance_includes_face_edge_vertex(
    query_box_model: Any,
) -> None:
    geometry = phx.geometry.BRepSource(query_box_model).compile()
    points = jnp.asarray(
        [
            [0.0, 0.7, 1.2],
            [0.0, 0.0, 1.2],
            [0.0, 0.0, 0.0],
            [-np.finfo(np.float64).eps, 0.7, 1.2],
            [-1e-10, 0.7, 1.2],
        ],
        dtype=jnp.float64,
    )
    np.testing.assert_array_equal(
        geometry.contains(points), [True, True, True, True, False]
    )


def test_brep_memory_candidate_nearest_preserves_second_face_and_refit(
    query_box_model: Any,
) -> None:
    from benchmarks.memory_geometry_brep import candidate_dynamic_nearest

    model = query_box_model
    mesh = phx.geometry.TriangleMesh(model.mesh_vertices, model.mesh_faces)
    tree = phx.geometry.TriangleBVH(mesh)
    vertices = mesh.vertices * jnp.asarray([1.4, 0.8, 1.1], dtype=jnp.float64)
    points = np.asarray(
        [[-0.2, 0.3, 0.8], [1.7, 0.4, 0.9], [-0.2, -0.2, 1.0]], dtype=np.float64
    )
    closest, face, _, second = _mesh_reference(
        points, np.asarray(vertices), np.asarray(mesh.faces)
    )
    actual, _, selected, second_sq = candidate_dynamic_nearest(
        tree, vertices, jnp.asarray(points)
    )
    np.testing.assert_allclose(actual, closest, atol=2e-14)
    np.testing.assert_array_equal(selected, face)
    np.testing.assert_allclose(second_sq, second, atol=2e-14)

    # Reverse-mode must not try to transpose the native traversal's while_loop.
    def objective(current: jax.Array) -> jax.Array:
        projected, _, _, second_squared = candidate_dynamic_nearest(
            tree, current, jnp.asarray(points[:1])
        )
        return jnp.sum(projected) + jnp.sum(second_squared)

    gradient = jax.grad(objective)(vertices)
    direction = jnp.full_like(vertices, 0.1)
    epsilon = 1e-5
    finite_difference = (
        objective(vertices + epsilon * direction)
        - objective(vertices - epsilon * direction)
    ) / (2.0 * epsilon)
    np.testing.assert_allclose(
        jnp.sum(gradient * direction), finite_difference, atol=2e-9
    )


def test_fixed_topology_brep_queries_follow_changed_radius_and_derivative() -> None:
    model = import_occt_shape(
        BRepPrimAPI_MakeSphere(1.0).Shape(),
        coordinate_contract=_CONTRACT,
        tessellation=BRepTessellationPolicy(
            linear_deflection=0.2,
            angular_deflection=0.4,
        ),
    ).model
    geometry = phx.geometry.FixedTopologyBRepSource(model).compile()
    kernel: Any = geometry.kernel
    radius_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "radius"
    )
    vertices = np.asarray(model.mesh_vertices)
    faces = np.asarray(model.mesh_faces)
    centroid = vertices[faces[len(faces) // 3]].mean(axis=0)
    points = np.asarray([2.0 * centroid, 1.1 * centroid], dtype=np.float64)
    changed_state = geometry.state.replace_at(
        radius_index, jnp.asarray(1.3, dtype=jnp.float64)
    )
    realization = kernel.realize(changed_state)
    closest, face, _, _ = _mesh_reference(
        points, np.asarray(realization.vertices), np.asarray(realization.faces)
    )
    result = kernel.closest_point(changed_state, jnp.asarray(points))
    np.testing.assert_allclose(result.closest_point, closest, atol=2e-13)
    np.testing.assert_array_equal(result.source_entity_id, face)
    np.testing.assert_array_equal(
        kernel.contains(changed_state, jnp.asarray(points)), [False, True]
    )
    np.testing.assert_array_equal(geometry.contains(jnp.asarray(points)), [False, False])
    np.testing.assert_array_equal(realization.faces, model.mesh_faces)

    def coordinate(radius: jax.Array) -> jax.Array:
        state = geometry.state.replace_at(radius_index, radius)
        return kernel.closest_point(state, jnp.asarray(points[:1])).normal_coordinate[0]

    radius = jnp.asarray(1.3, dtype=jnp.float64)
    epsilon = 1e-5
    derivative = jax.grad(coordinate)(radius)
    finite_difference = (coordinate(radius + epsilon) - coordinate(radius - epsilon)) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(derivative, finite_difference, rtol=1e-8, atol=1e-9)


def test_fixed_topology_brep_prepared_leaves_round_trip(query_box_model: Any) -> None:
    geometry = phx.geometry.FixedTopologyBRepSource(query_box_model).compile()
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, geometry)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, geometry)
    fingerprint = array_tree_fingerprint(geometry)
    assert array_tree_fingerprint(restored) == fingerprint
    assert any(
        record["path"].startswith(".kernel.query_bvh.")
        for record in fingerprint["signature"]
    )
    points = jnp.asarray(((-0.02, 0.3, 0.8), (-0.2, -0.2, 1.0)), dtype=jnp.float64)
    expected = geometry.closest_point(points)
    actual = restored.closest_point(points)
    np.testing.assert_array_equal(actual.closest_point, expected.closest_point)
    np.testing.assert_array_equal(actual.source_entity_id, expected.source_entity_id)
    np.testing.assert_array_equal(actual.margin, expected.margin)
    assert actual.physical_geometry_id == query_box_model.source_revision
    assert actual.represented_geometry_id == expected.represented_geometry_id


def _instanced_box(placement: BRepAffinePlacement) -> BRepModel:
    """Unit box definition used twice: at the identity and under ``placement``."""
    definition = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        source_id="projection-instance-definition",
    )
    source = definition.geometry
    graph = BRepAssemblyGraph(
        roots=(BRepAssemblyUse("pair", definition_id="two-copies"),),
        definitions=(
            BRepAssemblyDefinition(
                "two-copies",
                (
                    BRepAssemblyUse("left", solid_index=0),
                    BRepAssemblyUse("right", placement=placement, solid_index=0),
                ),
            ),
        ),
        solid_count=1,
    )
    geometry = BRepGeometry(
        vertex_points=source.vertex_points,
        vertex_exact_points=source.vertex_exact_points,
        vertex_roots=source.vertex_roots,
        edge_endpoint_roots=source.edge_endpoint_roots,
        coedge_endpoint_roots=source.coedge_endpoint_roots,
        edge_vertex_gap_bounds=source.edge_vertex_gap_bounds,
        edge_face_gap_bounds=source.edge_face_gap_bounds,
        curves=source.curves,
        edge_curves=source.edge_curves,
        edge_ranges=source.edge_ranges,
        edge_vertices=source.edge_vertices,
        pcurves=source.pcurves,
        coedge_edges=source.coedge_edges,
        coedge_senses=source.coedge_senses,
        face_loops=source.face_loops,
        shell_faces=source.shell_faces,
        shell_orientations=source.shell_orientations,
        solid_shells=source.solid_shells,
        assembly_graph=graph,
    )
    return BRepModel(
        geometry=geometry,
        patches=definition.patches,
        parameter_bounds=definition.parameter_bounds,
        orientation=definition.orientation,
        physical_tags=definition.physical_tags,
        coordinate_contract=definition.coordinate_contract,
        report=definition.report,
        trim_domains=definition.trim_domains,
    )


def test_classification_restricts_duplicate_definitions_by_occurrence_path() -> None:
    model = _instanced_box(
        BRepAffinePlacement.from_matrix(
            (
                (1, 0, 0, 10),
                (0, 1, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )
    )
    projection = phx.geometry.prepare_brep_projection(model)
    vertices = np.asarray(model.geometry.vertex_points, dtype=np.float64)
    vertex = int(
        np.flatnonzero(np.all(vertices == np.asarray((0.0, 0.0, 0.0)), axis=1))[0]
    )
    point = vertices[vertex][None, :]

    def classify(path: tuple[str, ...]) -> Any:
        return projection.classify(
            point,
            tolerance=1e-9,
            maximum_dimension=0,
            groups=np.asarray((7,), dtype=np.int64),
            allowed_groups=np.asarray((7,), dtype=np.int64),
            allowed_dimensions=np.asarray((0,), dtype=np.int64),
            allowed_indices=np.asarray((vertex,), dtype=np.int64),
            allowed_occurrence_paths=(path,),
        )

    left = classify(("pair", "left"))
    right = classify(("pair", "right"))
    assert np.asarray(left.status).tolist() == [_STATUS.UNIQUE]
    assert left.occurrence_paths == (("pair", "left"),)
    assert np.asarray(right.status).tolist() == [_STATUS.FAILED]
    projected = projection.project(
        point,
        np.asarray((0,), dtype=np.int64),
        np.asarray((vertex,), dtype=np.int64),
        occurrence_paths=(("pair", "left"),),
    )
    assert projected.occurrence_paths == (("pair", "left"),)
    located = projection.locate_solids(np.asarray(((0.5, 0.5, 0.5),)))
    assert located.occurrence_paths == (("pair", "left"),)


def test_occurrence_projection_rotates_each_defined_edge_tangent() -> None:
    # Edge frames are partial (curve tangent, undefined second slot); the
    # occurrence rotation must still carry the defined tangent into the world.
    model = _instanced_box(
        BRepAffinePlacement.from_matrix(
            (
                (0, -1, 0, 0),
                (1, 0, 0, 0),
                (0, 0, 1, 0),
                (0, 0, 0, 1),
            )
        )
    )
    projection = phx.geometry.prepare_brep_projection(model)
    vertices = np.asarray(model.geometry.vertex_points, dtype=np.float64)
    ends = np.sort(vertices[np.asarray(model.geometry.edge_vertices)], axis=1)
    edge = int(
        np.flatnonzero(
            np.all(ends[:, 0] == 0.0, axis=1)
            & np.all(ends[:, 1] == np.asarray((1.0, 0.0, 0.0)), axis=1)
        )[0]
    )
    projected = projection.project(
        np.asarray(((0.0, 0.5, 0.0),)),
        np.asarray((1,), dtype=np.int64),
        np.asarray((edge,), dtype=np.int64),
        occurrence_paths=(("pair", "right"),),
    )
    assert np.asarray(projected.status).tolist() == [_STATUS.UNIQUE]
    np.testing.assert_allclose(projected.points, [[0.0, 0.5, 0.0]], atol=1e-12)
    tangents = np.asarray(projected.tangents)
    np.testing.assert_allclose(np.abs(tangents[0, 0]), [0.0, 1.0, 0.0], atol=1e-12)
    assert np.all(np.isnan(tangents[0, 1]))


def test_native_projection_exposes_reflexive_topology_incidence() -> None:
    model = brep_box(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        coordinate_contract=_CONTRACT,
        source_id="projection-incidence",
    )
    projection = phx.geometry.prepare_brep_projection(model)
    topology = model.geometry.topology

    owners, member_dims, member_indices = projection.members(
        np.asarray((3,), dtype=np.int64), np.asarray((0,), dtype=np.int64)
    )
    np.testing.assert_array_equal(owners, np.zeros((27,), dtype=np.int64))
    np.testing.assert_array_equal(np.bincount(member_dims, minlength=4), (8, 12, 6, 1))
    np.testing.assert_array_equal(
        projection.entity_codes(member_dims, member_indices),
        np.unique(projection.entity_codes(member_dims, member_indices)),
    )

    edge = int(topology.face_edges[0][0])
    vertex = int(model.geometry.edge_vertices[edge][0])
    containers, container_dims, container_indices = projection.containers(
        np.asarray((0,), dtype=np.int64), np.asarray((vertex,), dtype=np.int64)
    )
    assert containers.shape == container_dims.shape == container_indices.shape
    assert set(container_dims.tolist()) == {0, 1, 2, 3}
    assert np.all(
        projection.contains(
            container_dims,
            container_indices,
            np.full(container_dims.shape, 0, dtype=np.int64),
            np.full(container_dims.shape, vertex, dtype=np.int64),
        )
    )
    unrelated_edge = next(
        candidate
        for candidate in range(model.geometry.topology.num_edges)
        if candidate not in topology.face_edges[0]
    )
    assert not bool(
        projection.contains(
            np.asarray((2,), dtype=np.int64),
            np.asarray((0,), dtype=np.int64),
            np.asarray((1,), dtype=np.int64),
            np.asarray((unrelated_edge,), dtype=np.int64),
        )[0]
    )


def test_definition_projection_returns_explicit_empty_occurrence_paths() -> None:
    source = brep_planar_face(
        PlanarProfile(
            ProfilePlane(),
            ProfileLoop.polygon(((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))),
        ),
        coordinate_contract=_CONTRACT,
        source_id="projection-definition-only",
    )
    projection = phx.geometry.prepare_brep_projection(source)
    result = projection.classify(
        np.asarray(((0.0, 0.0, 0.0),)), tolerance=1e-9, maximum_dimension=0
    )
    assert np.asarray(result.status).tolist() == [_STATUS.UNIQUE]
    assert result.occurrence_paths == ((),)


def test_optional_occt_projection_binds_physical_source_revision() -> None:
    shape = BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape()
    model = import_occt_shape(shape, coordinate_contract=_CONTRACT).model
    projection = prepare_occt_projection(model, shape)
    assert projection.source_revision == model.source_revision
    source_vertex = np.asarray(model.geometry.vertex_points)[0]
    observed = projection.project(
        source_vertex[None, :],
        np.asarray((0,), dtype=np.int64),
        np.asarray((0,), dtype=np.int64),
    )
    occurrence = next(model.geometry.iter_occurrences()).path
    assert observed.occurrence_paths == (occurrence,)
    assert np.asarray(observed.status).tolist() == [_STATUS.UNRESOLVED]
    np.testing.assert_allclose(observed.points, source_vertex[None, :])
    np.testing.assert_array_equal(observed.distance_lower_bounds, (0.0,))
    assert np.isinf(np.asarray(observed.residuals)[0])
    assert observed.frame_id == model.coordinate_contract.spatial_id

    different = import_occt_shape(
        BRepPrimAPI_MakeBox(2.0, 2.0, 3.0).Shape(),
        coordinate_contract=_CONTRACT,
        source_digest=model.report.source_digest,
    ).model
    with pytest.raises(ValueError, match="physical source_revision"):
        prepare_occt_projection(different, shape)


def test_optional_occt_projection_maps_assembly_entities_to_occurrence_paths() -> None:
    source = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape()
    transform = gp_Trsf()
    transform.SetTranslation(gp_Vec(10.0, 0.0, 0.0))
    placed = BRepBuilderAPI_Transform(source, transform, False).Shape()
    compound = TopoDS_Compound()
    builder = BRep_Builder()
    builder.MakeCompound(compound)
    builder.Add(compound, source)
    builder.Add(compound, placed)
    model = import_occt_shape(compound, coordinate_contract=_CONTRACT).model
    projection = prepare_occt_projection(model, compound)
    occurrences = tuple(model.geometry.iter_occurrences())
    assert len(occurrences) == 2
    query = None
    source_vertex = None
    for occurrence in occurrences:
        face = model.topology.solid_faces[occurrence.solid_index][0]
        edge = model.topology.face_edges[face][0]
        vertex = model.geometry.edge_vertices[edge][0]
        point = np.asarray(model.geometry.vertex_points, dtype=np.float64)[vertex]
        placed_point = np.asarray(
            occurrence.placement.apply(tuple(float(value) for value in point)),
            dtype=np.float64,
        )
        result = projection.project(
            placed_point[None, :],
            np.asarray((0,), dtype=np.int64),
            np.asarray((vertex,), dtype=np.int64),
            occurrence_paths=(occurrence.path,),
        )
        assert result.occurrence_paths == (occurrence.path,)
        assert np.asarray(result.status).tolist() == [_STATUS.UNRESOLVED]
        np.testing.assert_allclose(result.points, placed_point[None, :])
        assert result.entity_ids()[0] == source_entity_id(
            projection.source_id,
            projection.source_revision,
            "vertex",
            int(vertex),
            occurrence_path=occurrence.path,
        )
        assert np.isinf(np.asarray(result.residuals)[0])
        query = placed_point
        source_vertex = vertex
    assert query is not None and source_vertex is not None
    if occurrences[0].solid_index == occurrences[1].solid_index:
        ambiguous = projection.project(
            query[None, :],
            np.asarray((0,), dtype=np.int64),
            np.asarray((source_vertex,), dtype=np.int64),
        )
        assert np.asarray(ambiguous.dimensions).tolist() == [-1]
        assert np.asarray(ambiguous.indices).tolist() == [-1]
        assert ambiguous.occurrence_paths == ((),)
        assert np.asarray(ambiguous.status).tolist() == [_STATUS.UNRESOLVED]
