import io
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCP.BRepBuilderAPI import (
    BRepBuilderAPI_MakeEdge,
    BRepBuilderAPI_MakeFace,
    BRepBuilderAPI_MakePolygon,
    BRepBuilderAPI_MakeWire,
)
from OCP.BRepPrimAPI import (
    BRepPrimAPI_MakeBox,
    BRepPrimAPI_MakeCylinder,
    BRepPrimAPI_MakeSphere,
)
from OCP.gp import gp_Ax2, gp_Circ, gp_Dir, gp_Pnt

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax.units import METER
from tests._support.cad_models import native_solid


# Empirical `test_mesh_region_queries_follow_changed_radius_and_derivative`: matched max |delta|=2.00362e-12; joint allowance fraction=0.000184373, margin=99.9816%.
_MESH_REGION_RADIUS_DERIVATIVE_RTOL = 1e-8
# Empirical `test_mesh_region_queries_follow_changed_radius_and_derivative`: matched max |delta|=2.00362e-12; joint allowance fraction=0.000184373, margin=99.9816%.
_MESH_REGION_RADIUS_DERIVATIVE_ATOL = 1e-9
# Empirical `test_fixed_topology_native_brep_queries_follow_changed_radius_and_derivative`: matched max |delta|=2.22045e-16; joint allowance fraction=3.89978e-09, margin=100%.
_BREP_DYNAMIC_RADIUS_CLOSEST_POINT_ATOL = 2e-13


_CONTRACT = phx.SpatialCoordinateContract.si()
_STATUS = phx.geometry.BRepProjectionStatus
_FACE = phx.geometry.BRepEntityDimension.FACE
_EDGE = phx.geometry.BRepEntityDimension.EDGE

# Existing empirical budgets from the cylinder/sphere OCCT closed-form scenarios
# (d2dfcb941 test_brep_projection_scenario_1/2) and the seed-7, 48-sample native
# cylinder comparison below. These oracle tolerances are not certified bounds.
_OCCT_CLOSED_FORM_ATOL = 1e-12
_NATIVE_OCCT_ATOL = 1e-9
# Declared classification distance for the on-surface cylinder membership case.
_CLASSIFICATION_DISTANCE = 1e-9


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
        result.normals,
        [[0.6, 0.8, 0.0], [2**-0.5, 2**-0.5, 0.0]],
        atol=_OCCT_CLOSED_FORM_ATOL,
    )
    tangents = np.asarray(result.tangents)
    np.testing.assert_allclose(
        np.einsum("nka,na->nk", tangents, result.normals),
        0.0,
        atol=_OCCT_CLOSED_FORM_ATOL,
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
        np.abs(np.asarray(edge.tangents)[0, 0]),
        [1.0, 0.0, 0.0],
        atol=_OCCT_CLOSED_FORM_ATOL,
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
    result = projection.classify(points, tolerance=_CLASSIFICATION_DISTANCE)
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

    # Tessellation settings leave the source geometry revision unchanged.
    assert remodeled.source_revision == model.source_revision
    with pytest.raises(ValueError, match=r"prepare_occt_projection\.source"):
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
    np.testing.assert_allclose(
        face.residuals, [1.0 - 2**-0.5, 0.0], atol=_OCCT_CLOSED_FORM_ATOL
    )
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
        atol=_NATIVE_OCCT_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.points)[reference_unique],
        np.asarray(reference.points)[reference_unique],
        atol=_NATIVE_OCCT_ATOL,
    )
    np.testing.assert_allclose(
        np.asarray(result.normals)[reference_unique],
        np.asarray(reference.normals)[reference_unique],
        atol=_NATIVE_OCCT_ATOL,
    )
    classified = native.classify(points, tolerance=0.3)
    reference_classified = occt.classify(points, tolerance=0.3)
    np.testing.assert_array_equal(classified.dimensions, reference_classified.dimensions)
    np.testing.assert_allclose(
        classified.residuals, reference_classified.residuals, atol=_NATIVE_OCCT_ATOL
    )
    located = native.locate_solids(points)
    reference_located = occt.locate_solids(points)
    np.testing.assert_array_equal(located.status, reference_located.status)


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
    return phx.interchange.model_from_occt_shape(
        BRepPrimAPI_MakeBox(1.0, 2.0, 3.0).Shape(),
        coordinate_contract=_CONTRACT,
        linear_deflection=0.1,
        angular_deflection=0.3,
    )


@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "jit"])
def test_mesh_region_query_nearest_matches_independent_faces(
    query_box_model: Any, compiled: bool
) -> None:
    model = query_box_model
    geometry = phx.geometry.MeshRegion(
        model.mesh_vertices, model.mesh_faces, feature_id=model.model_id
    ).compile(coordinate_unit=METER)
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
    query = eqx.filter_jit(geometry.closest_point) if compiled else geometry.closest_point
    result = query(jnp.asarray(points))
    np.testing.assert_allclose(result.closest_point, closest, atol=2e-14)
    np.testing.assert_array_equal(result.source_entity_id, face)
    np.testing.assert_allclose(
        np.abs(result.normal_coordinate), np.sqrt(first), atol=2e-14
    )
    triangles = np.asarray(model.mesh_vertices)[np.asarray(model.mesh_faces)[face]]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    np.testing.assert_allclose(result.oriented_normal, normals, atol=2e-14)
    assert result.physical_geometry_id == model.model_id
    assert result.represented_geometry_id == model.model_id
    assert result.exact_to_physical


def test_mesh_region_two_face_margin_matches_independent_reference(
    query_box_model: Any,
) -> None:
    model = query_box_model
    geometry = phx.geometry.MeshRegion(
        model.mesh_vertices, model.mesh_faces, feature_id=model.model_id
    ).compile(coordinate_unit=METER)
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


def test_native_cad_import_boundary_tolerance_includes_face_edge_vertex(
    tmp_path: Path,
) -> None:
    path = tmp_path / "box.brep"
    phx.interchange.write_brep_text(native_solid("box"), path)
    imported = phx.interchange.read_brep_text(
        path,
        phx.interchange.CadImportPolicy(
            _CONTRACT,
            phx.interchange.ResourceLimits(
                max_bytes=1 << 22,
                max_depth=32,
                max_nodes=20_000,
                max_attributes=2_000_000,
                max_losses=8,
            ),
        ),
        source_length_unit=phx.units.METER,
        trusted_root=tmp_path,
    )
    assert imported.tolerance_evidence is not None
    assert imported.tolerance_evidence.model_id == imported.model.model_id
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
    # Retain the old roundoff-scale boundary acceptance, not the wider default
    # CAD classifier tolerance. The native import carries certified gap evidence.
    tolerance = (
        64.0 * np.finfo(np.float64).eps * max(float(jnp.max(jnp.abs(points))), 1.0)
    )
    query = phx.geometry.prepare_brep_query(
        imported.model,
        projection_policy=phx.geometry.BRepProjectionPolicy(
            classifier_tolerance=tolerance
        ),
    )
    containment = query.contains(points)
    boundary = containment.distance_upper_bounds <= tolerance
    assert not bool(jnp.any(containment.unresolved & ~boundary))
    assert not bool(jnp.any(containment.resource_exhausted))
    np.testing.assert_array_equal(
        containment.inside | boundary, [True, True, True, True, False]
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


def test_mesh_region_queries_follow_changed_radius_and_derivative() -> None:
    model = phx.interchange.model_from_occt_shape(
        BRepPrimAPI_MakeSphere(1.0).Shape(),
        coordinate_contract=_CONTRACT,
        linear_deflection=0.2,
        angular_deflection=0.4,
    )
    source = phx.geometry.MeshRegion(
        model.mesh_vertices, model.mesh_faces, feature_id=model.model_id
    )
    geometry = source.compile(coordinate_unit=METER)
    kernel: Any = geometry.kernel
    vertices_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "vertices"
    )
    vertices = np.asarray(model.mesh_vertices)
    faces = np.asarray(model.mesh_faces)
    centroid = vertices[faces[len(faces) // 3]].mean(axis=0)
    points = np.asarray([2.0 * centroid, 1.1 * centroid], dtype=np.float64)
    changed_vertices = 1.3 * source.vertices
    changed_state = geometry.state.replace_at(vertices_index, changed_vertices)
    closest, face, _, _ = _mesh_reference(
        points, np.asarray(changed_vertices), np.asarray(source.faces)
    )
    result = kernel.closest_point(changed_state, jnp.asarray(points))
    np.testing.assert_allclose(result.closest_point, closest, atol=2e-13)
    np.testing.assert_array_equal(result.source_entity_id, face)
    np.testing.assert_array_equal(
        kernel.contains(changed_state, jnp.asarray(points)), [False, True]
    )
    np.testing.assert_array_equal(geometry.contains(jnp.asarray(points)), [False, False])
    np.testing.assert_array_equal(source.faces, model.mesh_faces)

    def coordinate(radius: jax.Array) -> jax.Array:
        state = geometry.state.replace_at(vertices_index, radius * source.vertices)
        return kernel.closest_point(state, jnp.asarray(points[:1])).normal_coordinate[0]

    radius = jnp.asarray(1.3, dtype=jnp.float64)
    epsilon = 1e-5
    derivative = jax.grad(coordinate)(radius)
    finite_difference = (coordinate(radius + epsilon) - coordinate(radius - epsilon)) / (
        2.0 * epsilon
    )
    np.testing.assert_allclose(
        derivative,
        finite_difference,
        rtol=_MESH_REGION_RADIUS_DERIVATIVE_RTOL,
        atol=_MESH_REGION_RADIUS_DERIVATIVE_ATOL,
    )


def test_fixed_topology_native_brep_queries_follow_changed_radius_and_derivative() -> (
    None
):
    model = native_solid(
        "sphere",
        catalog="geometry",
        tessellation=phx.geometry.BRepTessellationPolicy(
            linear_deflection=0.2, angular_deflection=0.4
        ),
    )
    geometry = phx.geometry.FixedTopologyBRepSource(model).compile(coordinate_unit=METER)
    kernel: Any = geometry.kernel
    radius_index = next(
        index
        for index, spec in enumerate(geometry.schema.specs)
        if spec.parameter_id.name == "radius"
    )
    centroid = np.asarray(model.mesh_vertices)[
        np.asarray(model.mesh_faces)[model.mesh_faces.shape[0] // 3]
    ].mean(axis=0)
    points = np.asarray([2.0 * centroid, 1.1 * centroid], dtype=np.float64)
    changed_state = geometry.state.replace_at(
        radius_index, jnp.asarray(1.3, dtype=jnp.float64)
    )
    realization = kernel.realize(changed_state)
    result = kernel.closest_point(changed_state, jnp.asarray(points))
    expected = 1.3 * points / np.linalg.norm(points, axis=-1, keepdims=True)
    np.testing.assert_allclose(
        result.closest_point, expected, atol=_BREP_DYNAMIC_RADIUS_CLOSEST_POINT_ATOL
    )
    np.testing.assert_array_equal(result.source_entity_id, np.zeros(2, dtype=np.int32))
    np.testing.assert_array_equal(
        kernel.contains(changed_state, jnp.asarray(points)), [False, True]
    )
    np.testing.assert_array_equal(geometry.contains(jnp.asarray(points)), [False, False])
    np.testing.assert_array_equal(realization.faces, model.mesh_faces)
    assert result.physical_geometry_id == model.source_revision
    assert result.represented_geometry_id == f"{model.model_id}:fixed-topology-query"
    assert result.exact_to_physical

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


def test_mesh_region_prepared_leaves_round_trip(query_box_model: Any) -> None:
    geometry = phx.geometry.MeshRegion(
        query_box_model.mesh_vertices,
        query_box_model.mesh_faces,
        feature_id=query_box_model.model_id,
    ).compile(coordinate_unit=METER)
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, geometry)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, geometry)
    fingerprint = array_tree_fingerprint(geometry)
    assert array_tree_fingerprint(restored) == fingerprint
    assert any(
        record["path"].startswith(".kernel.index.") for record in fingerprint["signature"]
    )
    points = jnp.asarray(((-0.02, 0.3, 0.8), (-0.2, -0.2, 1.0)), dtype=jnp.float64)
    expected = geometry.closest_point(points)
    actual = restored.closest_point(points)
    np.testing.assert_array_equal(actual.closest_point, expected.closest_point)
    np.testing.assert_array_equal(actual.source_entity_id, expected.source_entity_id)
    np.testing.assert_array_equal(actual.margin, expected.margin)
    assert actual.physical_geometry_id == query_box_model.model_id
    assert actual.represented_geometry_id == expected.represented_geometry_id
