from pathlib import Path
from typing import Any

import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)

SCHEDULE = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)


def _wall(points: Any, triangles: Any) -> Any:
    return phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                np.asarray(triangles, dtype=np.int64),
                global_ids=np.arange(len(triangles), dtype=np.int64),
            ),
        ),
    )


def _whole(mesh: Any) -> Any:
    cells = mesh.entity_set(2)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )


def _control(mesh: Any, **options: Any) -> Any:
    return phx.meshing.BoundaryLayerControl(
        _whole(mesh),
        SCHEDULE,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        **options,
    )


def test_quad_layer_cap_resolves_narrow_clearance_without_retry_allowance() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (64.0, 0.0, 0.0), (64.0, 64.0, 0.0), (0.0, 64.0, 0.0)),
        dtype=np.float64,
    )
    wall = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "wall", "quadrilateral", np.asarray(((0, 1, 2, 3),), dtype=np.int64)
            ),
        ),
    )
    clearance = 0.1
    obstacle = phx.discretization.CellMesh.from_triangles(
        np.asarray(
            (
                (-64.0, -64.0, clearance),
                (128.0, -64.0, clearance),
                (128.0, 128.0, clearance),
                (-64.0, 128.0, clearance),
            ),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2), (0, 2, 3)), dtype=np.int64),
    )

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall), obstacles=obstacle)

    coordinates = np.asarray(result.mesh.coordinates)
    np.testing.assert_array_equal(coordinates[:4], points)
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-12
    )
    assert SCHEDULE.total_thickness < np.max(coordinates[:, 2]) < clearance
    assert _cell_counts(result) == {"hexahedron": 3, "pyramid": 1}
    _assert_certified(result)


def _plate(count: Any, height: Any = 0.0, *, flip: Any = False) -> Any:
    values = np.linspace(0.0, 1.0, count + 1)
    x, y = np.meshgrid(values, values, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.full(x.size, height)), axis=1)
    index = np.arange(points.shape[0]).reshape(count + 1, count + 1)
    triangles = []
    for i in range(count):
        for j in range(count):
            a, b, c, d = (
                index[i, j],
                index[i + 1, j],
                index[i + 1, j + 1],
                index[i, j + 1],
            )
            triangles.extend(((a, b, c), (a, c, d)))
    triangles = np.asarray(triangles)
    return points, triangles[:, ::-1] if flip else triangles


def _oriented(points: Any, triangles: Any, center: Any, outward: Any) -> Any:
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    away = np.sum(normals * (corners.mean(axis=1) - center), axis=1) > 0.0
    flip = away != outward
    triangles = triangles.copy()
    triangles[flip] = triangles[flip][:, ::-1]
    return triangles


def _icosphere(radius: Any, subdivisions: Any) -> Any:
    t = (1.0 + 5.0**0.5) / 2.0
    points = [
        np.asarray(value, dtype=np.float64)
        for value in (
            (-1, t, 0),
            (1, t, 0),
            (-1, -t, 0),
            (1, -t, 0),
            (0, -1, t),
            (0, 1, t),
            (0, -1, -t),
            (0, 1, -t),
            (t, 0, -1),
            (t, 0, 1),
            (-t, 0, -1),
            (-t, 0, 1),
        )
    ]
    points = [value / np.linalg.norm(value) for value in points]
    faces = [
        (0, 11, 5),
        (0, 5, 1),
        (0, 1, 7),
        (0, 7, 10),
        (0, 10, 11),
        (1, 5, 9),
        (5, 11, 4),
        (11, 10, 2),
        (10, 7, 6),
        (7, 1, 8),
        (3, 9, 4),
        (3, 4, 2),
        (3, 2, 6),
        (3, 6, 8),
        (3, 8, 9),
        (4, 9, 5),
        (2, 4, 11),
        (6, 2, 10),
        (8, 6, 7),
        (9, 8, 1),
    ]
    for _ in range(subdivisions):
        middle = {}

        def split(first: Any, second: Any, *, middle: Any = middle) -> Any:
            key = (min(first, second), max(first, second))
            if key not in middle:
                value = points[first] + points[second]
                points.append(value / np.linalg.norm(value))
                middle[key] = len(points) - 1
            return middle[key]

        refined = []
        for a, b, c in faces:
            ab, bc, ca = split(a, b), split(b, c), split(c, a)
            refined.extend(((a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)))
        faces = refined
    points = radius * np.asarray(points)
    return points, _oriented(points, np.asarray(faces), np.zeros(3), True)


def _box(lower: Any, upper: Any, count: Any, *, outward: Any) -> Any:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    index: dict[tuple[float, ...], int] = {}
    points = []
    triangles = []
    grid = np.linspace(0.0, 1.0, count + 1)

    def vertex(value: Any) -> Any:
        key = tuple(np.round(value, 12))
        if key not in index:
            index[key] = len(points)
            points.append(value)
        return index[key]

    for axis in range(3):
        u, v = (value for value in range(3) if value != axis)
        for bound in (lower[axis], upper[axis]):
            for i in range(count):
                for j in range(count):
                    quad = []
                    for di, dj in ((0, 0), (1, 0), (1, 1), (0, 1)):
                        value = np.zeros(3)
                        value[axis] = bound
                        value[u] = lower[u] + (upper[u] - lower[u]) * grid[i + di]
                        value[v] = lower[v] + (upper[v] - lower[v]) * grid[j + dj]
                        quad.append(vertex(value))
                    a, b, c, d = quad
                    triangles.extend(((a, b, c), (a, c, d)))
    points = np.asarray(points)
    return points, _oriented(
        points, np.asarray(triangles), 0.5 * (lower + upper), outward
    )


def _cell_counts(result: Any) -> Any:
    return dict(result.evidence.cell_counts)


def _assert_certified(result: Any) -> None:
    cells = sum(block.cell_count for block in result.mesh.blocks)
    assert result.validity.certified_valid_count == cells
    assert result.evidence.certified_valid_count == cells


def test_boundary_layer_retains_stable_source_column_ancestry() -> None:
    points, triangles = _plate(2)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    assert result.source_wall is wall
    assert result.wall_association is None
    column_index = np.asarray(result.column_index)
    assert column_index.dtype == np.int64
    cell_ids = np.concatenate(
        [np.asarray(block.global_ids) for block in result.mesh.blocks]
    )
    np.testing.assert_array_equal(np.argsort(cell_ids), np.arange(cell_ids.size))
    np.testing.assert_array_equal(column_index, column_index[np.argsort(cell_ids)])
    row_ids = np.asarray(result.front_row_ids)
    np.testing.assert_array_equal(np.sort(row_ids), np.arange(row_ids.size))
    assert set(column_index) <= set(row_ids)
    assert set(result.front_row_source_entity_kinds) == {"face"}
    assert set(np.asarray(result.front_row_source_global_ids)) <= set(
        np.asarray(wall.entity_set(2).entity_ids)
    )
    assert np.all(np.asarray(result.front_row_orientations) == 1)
    assert len(result.front_row_identity) == row_ids.size
    assert len(set(result.front_row_identity)) == row_ids.size
    column_ids = np.asarray(result.column_vertex_ids)
    np.testing.assert_array_equal(np.sort(column_ids), np.arange(column_ids.size))
    assert len(set(result.column_vertex_identity)) == column_ids.size
    assert set(np.asarray(result.column_vertex_sources)) <= set(
        np.asarray(wall.vertex_global_ids)
    )
    face_ids = np.asarray(result.column_source_face_ids)
    offsets = np.asarray(result.column_source_face_offsets)
    assert offsets.shape == (column_ids.size + 1,)
    np.testing.assert_array_equal(offsets[[0, -1]], (0, face_ids.size))
    assert np.all(np.diff(offsets) > 0)
    assert set(face_ids) <= set(np.asarray(wall.entity_set(2).entity_ids))
    face_weights = np.asarray(result.column_source_face_weights)
    face_orientations = np.asarray(result.column_source_face_orientations)
    assert face_weights.shape == face_orientations.shape == face_ids.shape
    assert np.all(face_weights >= 0.0)
    np.testing.assert_allclose(
        [
            face_weights[start:end].sum()
            for start, end in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        1.0,
    )
    assert np.all(np.isin(face_orientations, (-1, 1)))
    row_kinds = result.front_row_source_entity_kinds
    row_global_ids = np.asarray(result.front_row_source_global_ids)
    row_orientations = np.asarray(result.front_row_orientations)
    assert row_global_ids.shape == row_orientations.shape == (row_ids.size,)
    assert np.all(np.isin(row_orientations, (-1, 1)))
    for kind, dimension in (("face", 2), ("edge", 1), ("vertex", 0)):
        rows = np.asarray([value == kind for value in row_kinds], dtype=np.bool_)
        assert set(row_global_ids[rows]) <= set(
            np.asarray(wall.entity_set(dimension).entity_ids)
        )
    vertex_sources = np.asarray(result.vertex_source_vertex_ids)
    vertex_columns = np.asarray(result.vertex_column_ids)
    assert (
        vertex_sources.shape
        == vertex_columns.shape
        == (result.mesh.coordinates.shape[0],)
    )
    assert set(vertex_sources[vertex_sources >= 0]) <= set(
        np.asarray(wall.vertex_global_ids)
    )
    retained = np.flatnonzero(np.asarray(result.wall_vertices) >= 0)
    retained_rows = np.asarray(result.wall_vertices)[retained]
    assert np.all(vertex_columns[retained_rows] == -1)
    assert np.array_equal(
        np.asarray(result.mesh.coordinates)[retained_rows].view(np.uint64),
        np.asarray(wall.coordinates)[retained].view(np.uint64),
    )
    np.testing.assert_array_equal(
        np.diff(np.asarray(result.vertex_column_parent_offsets))[retained_rows], 0
    )
    generated = np.flatnonzero(vertex_columns >= 0)
    assert generated.size > 0
    parent_offsets = np.asarray(result.vertex_column_parent_offsets)
    parent_weights = np.asarray(result.vertex_column_parent_weights)
    parent_counts = np.diff(parent_offsets)
    assert np.all(parent_counts[generated] > 0)
    np.testing.assert_allclose(
        [
            parent_weights[parent_offsets[row] : parent_offsets[row + 1]].sum()
            for row in generated
        ],
        1.0,
    )


def test_boundary_layer_scenario_1() -> None:
    points, triangles = _plate(5)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-12
    )
    np.testing.assert_allclose(result.evidence.achieved_growth_rates, 1.2, rtol=1e-12)
    active = np.asarray(result.evidence.layer_active)
    assert active.dtype == np.bool_
    np.testing.assert_array_equal(active, (True, True, True))
    evidence_values = {
        name: getattr(result.evidence, name)
        for name in result.evidence.__dataclass_fields__
        if name != "evidence_id"
    }
    with pytest.raises(ValueError, match="invalid shapes"):
        phx.meshing.BoundaryLayerEvidence(
            **{
                **evidence_values,
                "achieved_thicknesses": np.ones((2,), dtype=np.float64),
            }
        )
    assert _cell_counts(result) == {"prism": 3 * triangles.shape[0]}
    heights = np.unique(np.round(np.asarray(result.mesh.coordinates)[:, 2], 12))
    np.testing.assert_allclose(heights, (0.0, 0.01, 0.022, 0.0364))
    _assert_certified(result)
    assert not result.closed_cap
    values = np.linspace(0.0, 1.0, 4)
    x, y = np.meshgrid(values, values, indexing="ij")
    points = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange(points.shape[0]).reshape(4, 4)
    quads = np.asarray(
        [
            (index[i, j], index[i + 1, j], index[i + 1, j + 1], index[i, j + 1])
            for i in range(3)
            for j in range(3)
        ]
    )
    wall = phx.discretization.CellMesh(
        points,
        (
            phx.discretization.CellBlock(
                "quadrilaterals", "quadrilateral", quads, global_ids=np.arange(9)
            ),
        ),
    )

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    assert _cell_counts(result) == {"pyramid": 9, "hexahedron": 27}
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-12
    )
    # ty: ignore[unresolved-attribute]
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    layer_index = np.asarray(result.layer_index)
    pyramids = result.mesh.block("pyramids")
    np.testing.assert_array_equal(layer_index[np.asarray(pyramids.global_ids)], 3)
    _assert_certified(result)
    lower, lower_triangles = _plate(4)
    upper, upper_triangles = _plate(4, flip=True)
    upper[:, 2] = 0.03 + 0.17 * upper[:, 0]
    wall = _wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )

    result = phx.meshing.prepare_boundary_layers(
        wall,
        _control(
            wall, collision=phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY
        ),
    )

    evidence = result.evidence
    assert 0 < evidence.terminated_vertex_count < 50
    counts = _cell_counts(result)
    assert counts["pyramid"] > 0 and counts["tetrahedron"] > 0
    assert evidence.achieved_thicknesses[0] == pytest.approx(0.01)
    _assert_certified(result)


def test_boundary_layer_scenario_2() -> None:
    points, triangles = _icosphere(0.4, 2)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    evidence = result.evidence
    np.testing.assert_allclose(
        evidence.minimum_thicknesses, SCHEDULE.thicknesses, rtol=1e-6
    )
    np.testing.assert_allclose(
        evidence.maximum_thicknesses, SCHEDULE.thicknesses, rtol=1e-6
    )
    np.testing.assert_allclose(evidence.achieved_growth_rates, 1.2, rtol=1e-6)
    assert evidence.fan_column_count == 0
    assert result.closed_cap
    # ty: ignore[unresolved-attribute]
    radii = np.linalg.norm(np.asarray(result.cap.coordinates), axis=1)
    assert np.all(radii > 0.4)
    _assert_certified(result)
    points, triangles = _box((-0.3, -0.3, -0.3), (0.3, 0.3, 0.3), 3, outward=True)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    evidence = result.evidence
    assert evidence.convex_ridge_count == 12 * 3
    assert evidence.corner_patch_count == 8
    assert evidence.fan_column_count > 0
    counts = _cell_counts(result)
    assert (
        counts["hexahedron"] > 0 and counts["tetrahedron"] > 0 and counts["pyramid"] > 0
    )
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-9
    )
    assert result.closed_cap
    # ty: ignore[unresolved-attribute]
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    _assert_certified(result)
    points, triangles = _box((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), 3, outward=False)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    evidence = result.evidence
    assert evidence.concave_ridge_count == 12 * 3
    assert evidence.fan_column_count == 0
    assert evidence.maximum_stretch == pytest.approx(np.sqrt(3.0), rel=1e-6)
    np.testing.assert_allclose(
        evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-9
    )
    _assert_certified(result)


def test_visibility_optimum_is_exact_and_certified_at_sharp_ridges() -> None:
    from phydrax.meshing._boundary_layer import _visibility_optimum

    axes = np.eye(3)
    # Duplicated coplanar normals at a box edge and a box corner, a fold, a flat column.
    columns = np.asarray(
        [
            [axes[0], axes[0], axes[1], axes[1], axes[0], axes[1]],
            [axes[0], axes[1], axes[2], axes[0], axes[1], axes[2]],
            [axes[0], -axes[0], axes[1], axes[1], axes[1], axes[1]],
            [axes[2]] * 6,
        ]
    )
    weights, visibility, gap = _visibility_optimum(columns)
    np.testing.assert_allclose(
        visibility, (np.sqrt(0.5), np.sqrt(1.0 / 3.0), 0.0, 1.0), atol=1e-15
    )
    np.testing.assert_allclose(np.sum(weights, axis=1), 1.0)
    assert np.all(gap[[0, 1, 3]] <= 1.0e-12)
    rng = np.random.default_rng(7)
    normals = rng.normal(size=(64, 7, 3)) + np.asarray((0.0, 0.0, 2.0))
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    weights, visibility, gap = _visibility_optimum(normals)
    visible = visibility > 0.0
    assert np.all(gap[visible] <= 1.0e-9)
    hull = np.einsum("cw,cwd->cd", weights, normals)
    np.testing.assert_allclose(np.linalg.norm(hull, axis=1)[visible], visibility[visible])
    probes = rng.normal(size=(20000, 3))
    probes /= np.linalg.norm(probes, axis=1, keepdims=True)
    sampled = np.max(np.min(np.einsum("cwd,pd->cwp", normals, probes), axis=1), axis=1)
    assert np.all(sampled[visible] <= visibility[visible] + 1.0e-12)
    assert np.all(sampled[~visible] <= 1.0e-12)
    # A collapsed pole fan: hundreds of distinct normals on one cone. The
    # optimum is the cone axis, and the cost stays linear in the fan width.
    tilt = 0.3
    angles = np.linspace(0.0, 2.0 * np.pi, 400, endpoint=False)
    fan = np.stack(
        (
            np.sin(tilt) * np.cos(angles),
            np.sin(tilt) * np.sin(angles),
            np.full_like(angles, np.cos(tilt)),
        ),
        axis=-1,
    )[None]
    weights, visibility, gap = _visibility_optimum(fan)
    assert visibility[0] == pytest.approx(np.cos(tilt), abs=1.0e-12)
    assert gap[0] <= 1.0e-9
    np.testing.assert_allclose(
        np.einsum("cw,cwd->cd", weights, fan)[0], (0.0, 0.0, np.cos(tilt)), atol=1e-12
    )


def test_boundary_layer_scenario_3() -> None:
    count = 3
    values = np.linspace(0.0, 1.0, count + 1)
    angle = np.deg2rad(30.0)
    floor = [(r, w, 0.0) for r in values for w in values]
    slope = [
        (r * np.cos(angle), w, r * np.sin(angle)) for r in values[1:] for w in values
    ]
    points = np.asarray(floor + slope)
    first = np.arange((count + 1) ** 2).reshape(count + 1, count + 1)
    second = np.vstack(
        (first[:1], first.size + np.arange(count * (count + 1)).reshape(count, -1))
    )
    triangles = []
    for grid in (first, second):
        for i in range(count):
            for j in range(count):
                a, b, c, d = (
                    grid[i, j],
                    grid[i + 1, j],
                    grid[i + 1, j + 1],
                    grid[i, j + 1],
                )
                triangles.extend(((a, b, c), (a, c, d)))
    triangles = np.asarray(triangles)
    # Both sheets face into the 30-degree wedge between them.
    inside = np.asarray((0.5 * np.cos(0.5 * angle), 0.5, 0.5 * np.sin(0.5 * angle)))
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.sum(normals * (inside - corners.mean(axis=1)), axis=1) < 0.0
    triangles[flip] = triangles[flip][:, ::-1]
    wall = _wall(points, triangles)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.prepare_boundary_layers(wall, _control(wall))

    assert (
        failure.value.category
        is phx.meshing.MeshingFailureCategory.UNSUPPORTED_COMBINATION
    )
    crease = np.flatnonzero(np.isclose(points[:, 0], 0.0) & np.isclose(points[:, 2], 0.0))
    assert set(failure.value.entity_ids) == {int(value) for value in crease}
    np.testing.assert_allclose(failure.value.locations, points[crease])
    points, triangles = _box((0.0, 0.0, 0.0), (1.0, 1.0, 0.5), 4, outward=False)
    heights = points[triangles][:, :, 2]
    open_box = triangles[~np.all(np.isclose(heights, 0.5), axis=1)]
    wall = _wall(points, open_box)
    cells = wall.entity_set(2)
    floor = np.flatnonzero(np.all(np.isclose(points[open_box][:, :, 2], 0.0), axis=1))
    scope = phx.meshing.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        np.asarray(cells.entity_ids)[floor],
    )
    control = phx.meshing.BoundaryLayerControl(
        scope, SCHEDULE, route=phx.meshing.BoundaryLayerRoute.ADVANCING
    )

    result = phx.meshing.prepare_boundary_layers(wall, control)

    assert result.evidence.rim_vertex_count == 16
    coordinates = np.asarray(result.mesh.coordinates)
    assert np.all((coordinates[:, :2] >= 0.0) & (coordinates[:, :2] <= 1.0))
    # Rim columns stay exactly on their side planes (bitwise), five per side level.
    on_side = np.count_nonzero(coordinates[:, 0] == 0.0)
    assert on_side == 5 * 4
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-12
    )
    assert not result.closed_cap
    _assert_certified(result)
    wall = _channel(0.05)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.prepare_boundary_layers(wall, _control(wall))

    assert failure.value.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
    assert len(failure.value.entity_ids) == 50
    assert len(failure.value.locations) == 50
    for collision in (
        phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        phx.meshing.BoundaryLayerCollisionPolicy.MERGE,
    ):
        gap = 0.05
        wall = _channel(gap)

        result = phx.meshing.prepare_boundary_layers(
            wall, _control(wall, collision=collision, minimum_thickness_fraction=0.2)
        )

        evidence = result.evidence
        assert evidence.collision_policy is collision
        assert evidence.predicted_collision_vertex_count == 50
        heights = np.asarray(result.mesh.coordinates)[:, 2]
        match collision:
            case phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS:
                assert 0.2 <= evidence.minimum_scale < 1.0
                assert evidence.reduced_vertex_count == 50
                assert np.max(heights[heights < 0.5 * gap]) < 0.5 * gap
                np.testing.assert_allclose(evidence.achieved_growth_rates, 1.2, rtol=1e-9)
                assert np.all(np.asarray(evidence.layer_active))
            case phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
                assert evidence.terminated_vertex_count == 50
                assert _cell_counts(result) == {"prism": 2 * 32}
                achieved = np.asarray(evidence.achieved_thicknesses)
                active = np.asarray(evidence.layer_active)
                assert achieved[0] == pytest.approx(0.01)
                np.testing.assert_array_equal(active, (True, False, False))
                np.testing.assert_array_equal(np.isnan(achieved), ~active)
                assert np.all(np.isnan(evidence.achieved_growth_rates))
            case phx.meshing.BoundaryLayerCollisionPolicy.MERGE:
                assert evidence.merged_vertex_count == 50
                assert evidence.minimum_scale == pytest.approx(
                    0.5 * gap / SCHEDULE.total_thickness
                )
                assert np.any(np.isclose(heights, 0.5 * gap))
                # Merged fronts share their midsurface, so no free front remains.
                assert result.cap is None
        _assert_certified(result)


def _channel(gap: Any) -> Any:
    lower, lower_triangles = _plate(4)
    upper, upper_triangles = _plate(4, gap, flip=True)
    return _wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )


@pytest.mark.parametrize("height", (0.5, 0.2))
def test_selected_floor_distinguishes_parallel_sides_from_top_contact(
    height: float,
) -> None:
    points, triangles = _box((0.0, 0.0, 0.0), (1.0, 1.0, height), 8, outward=False)
    wall = _wall(points, triangles)
    floor = np.flatnonzero(np.all(points[triangles][:, :, 2] == 0.0, axis=1))
    scope = phx.meshing.MeshingScope(
        wall.mesh_id,
        wall.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        wall.entity_set(2).entity_set_id,
        np.asarray(wall.entity_set(2).entity_ids)[floor],
    )
    control = phx.meshing.BoundaryLayerControl(
        scope,
        phx.meshing.LayerSchedule.geometric(2, 0.1, growth_rate=1.0),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    )
    if height == 0.2:
        with pytest.raises(phx.meshing.MeshingFailure) as failure:
            phx.meshing.prepare_boundary_layers(wall, control)
        assert (
            failure.value.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
        )
        return
    result = phx.meshing.prepare_boundary_layers(wall, control)
    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, (0.1, 0.1), rtol=1e-12
    )
    _assert_certified(result)


@pytest.mark.parametrize(
    "collision",
    (
        phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        phx.meshing.BoundaryLayerCollisionPolicy.MERGE,
    ),
)
def test_retained_column_state_is_reproved_without_growth(collision: Any) -> None:
    import equinox as eqx

    from phydrax.meshing._boundary_layer import _reprove_retained_boundary_layers

    wall = _channel(0.05)
    control = _control(wall, collision=collision, minimum_thickness_fraction=0.2)
    layers = phx.meshing.prepare_boundary_layers(wall, control)
    assert _reprove_retained_boundary_layers(layers, wall, control) is layers

    coordinates = np.array(layers.column_level_coordinates)
    coordinates[0, 0, 2] += 0.001
    forged = eqx.tree_at(
        lambda value: value.column_level_coordinates, layers, coordinates
    )
    with pytest.raises(ValueError, match="column coordinates|column vertices"):
        _reprove_retained_boundary_layers(forged, wall, control)


def test_retained_quad_cap_and_obstacle_reproof() -> None:
    import equinox as eqx

    from phydrax.meshing._boundary_layer import _reprove_retained_boundary_layers

    points = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    wall = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("wall", "quadrilateral", [[0, 1, 2, 3]]),)
    )
    obstacles = _wall(points + (0, 0, 2), ((0, 1, 2), (0, 2, 3)))
    control = _control(wall)
    layers = phx.meshing.prepare_boundary_layers(wall, control, obstacles=obstacles)
    assert (
        _reprove_retained_boundary_layers(layers, wall, control, obstacles=obstacles)
        is layers
    )
    with pytest.raises(ValueError, match="obstacles"):
        _reprove_retained_boundary_layers(layers, wall, control)
    face_weights = np.array(layers.column_source_face_weights)
    face_weights[0] *= 0.5
    forged = eqx.tree_at(
        lambda value: value.column_source_face_weights, layers, face_weights
    )
    with pytest.raises(ValueError, match="front ancestry"):
        _reprove_retained_boundary_layers(forged, wall, control, obstacles=obstacles)


def test_original_ridge_retained_source_reproof() -> None:
    from phydrax.meshing._boundary_layer import _reprove_retained_boundary_layers
    from tools.meshing_qualification.scenarios.native_hybrid_profiles import (
        _archived_ridge_source,
    )

    source, control, policy = _archived_ridge_source()
    layers = source.layers
    assert (
        _reprove_retained_boundary_layers(
            layers,
            layers.source_wall,
            control,
            obstacles=layers.obstacles,
            wall_association=layers.wall_association,
            policy=policy,
        )
        is layers
    )


def test_cell_certification_rejects_contained_obstacle() -> None:
    from phydrax.meshing._boundary_layer import (
        _CELL_ARITY,
        _certify_cells,
        _Environment,
        _VOLUME_KINDS,
    )

    points = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)), dtype=float)
    wall = phx.discretization.CellMesh(
        points, (phx.discretization.CellBlock("wall", "quadrilateral", [[0, 1, 2, 3]]),)
    )
    policy = phx.meshing.BoundaryLayerPolicy(simplex_cap=False)
    layers = phx.meshing.prepare_boundary_layers(wall, _control(wall), policy=policy)
    cells = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    cells["hexahedron"] = np.asarray(layers.mesh.blocks[0].vertices, dtype=np.int64)[:1]
    obstacle_points = np.asarray(
        ((0.4, 0.4, 0.005), (0.6, 0.4, 0.005), (0.5, 0.6, 0.005))
    )
    environment = _Environment(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0,), dtype=bool),
        obstacle_points,
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    bad, hits, _ = _certify_cells(
        np.asarray(layers.mesh.coordinates), cells, [], environment, policy
    )
    assert bad["hexahedron"][0]
    assert hits > 0


def test_cell_certification_proves_disjoint_warped_face() -> None:
    from phydrax.meshing._boundary_layer import (
        _CELL_ARITY,
        _certify_cells,
        _Environment,
        _VOLUME_KINDS,
    )

    points = np.asarray(
        (
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 0, 1),
            (1, 0, 1),
            (1, 1, 1.4),
            (0, 1, 1),
        ),
        dtype=float,
    )
    cells = {
        name: np.empty((0, _CELL_ARITY[name]), dtype=np.int64) for name in _VOLUME_KINDS
    }
    cells["hexahedron"] = np.asarray(((0, 1, 2, 3, 4, 5, 6, 7),), dtype=np.int64)
    obstacle = np.asarray(((0.1, 0.1, 1.2), (0.2, 0.1, 1.2), (0.1, 0.2, 1.2)))
    environment = _Environment(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0,), dtype=bool),
        obstacle,
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    bad, hits, _ = _certify_cells(
        points, cells, [], environment, phx.meshing.BoundaryLayerPolicy()
    )
    assert not bad["hexahedron"][0]
    assert hits == 0


def test_warped_wall_contact_requires_declared_zero_hull() -> None:
    from phydrax.geometry._global_embedding import (
        _bilinear_quad_triangle_contact_is_declared,
    )

    quad = np.asarray(((0, 0, 0), (1, 0, 0), (1, 1, 1), (0, 1, 0.5)), dtype=float)
    wall = np.asarray(((0, 0, 0), (1, 0, 0), (0, -1, 0)), dtype=float)
    quad_ids = np.asarray((0, 1, 2, 3))
    assert _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((0, 1, 4))
    )
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((5, 6, 7))
    )
    assert _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((5, 6, 7)), sliding_side=True
    )
    quad[3, 2] = -0.5
    wall[2, 1] = 1.0
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((0, 1, 4))
    )


def test_warped_wall_contact_uses_exact_control_hull() -> None:
    from phydrax.geometry._global_embedding import (
        _bilinear_quad_triangle_contact_is_declared,
    )

    # None of the wall-plane or triangle-edge separating planes suffices.
    # The tetrahedral control hull meets the wall triangle only at vertex 0.
    quad = np.asarray(((0, 0, 0), (1, -1, -1), (-1, 1, -1), (1, 1, 1)), dtype=float)
    wall = np.asarray(((0, 0, 0), (-1, 0, 0), (0, -1, 0)), dtype=float)
    quad_ids = np.arange(4)
    assert _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((0, 4, 5))
    )
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((6, 4, 5))
    )

    # At parameters (u,v)=(1/4,1/3), the changed bilinear patch meets
    # the triangle at (-1/6,-1/3,0), outside the shared vertex.
    quad[3] = (-1, -1, 1)
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((0, 4, 5))
    )
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, wall, np.asarray((0, 4, 5)), sliding_side=True
    )


def test_warped_wall_contact_original_sloped_plate() -> None:
    from phydrax.geometry._global_embedding import (
        _bilinear_quad_triangle_contact_is_declared,
    )

    # Original opposing_sloped_plates4, native 730; hexadecimal binary64
    # preserves the source operands, including its nonzero plane residual.
    quad = np.asarray(
        [
            [float.fromhex(v) for v in row]
            for row in (
                ("0x1.0000000000000p+0", "0x1.0000000000000p-2", "0x1.999999999999ap-3"),
                ("0x1.8000000000000p-1", "0x0.0p+0", "0x1.428f5c28f5c29p-3"),
                ("0x1.80dbabb74edcfp-1", "0x0.0p+0", "0x1.2e5ea2d75bcddp-3"),
                ("0x1.006dd5dba76e9p+0", "0x1.0000000000000p-2", "0x1.8568e047ffa27p-3"),
            )
        ]
    )
    triangle = np.asarray(
        [
            [float.fromhex(v) for v in row]
            for row in (
                ("0x1.8000000000000p-1", "0x1.0000000000000p-2", "0x1.428f5c28f5c29p-3"),
                ("0x1.8000000000000p-1", "0x0.0p+0", "0x1.428f5c28f5c29p-3"),
                ("0x1.0000000000000p-1", "0x0.0p+0", "0x1.d70a3d70a3d71p-4"),
            )
        ]
    )
    quad_ids = np.asarray((46, 40, 90, 96))
    assert _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, triangle, np.asarray((41, 40, 35))
    )
    assert not _bilinear_quad_triangle_contact_is_declared(
        quad, quad_ids, triangle, np.asarray((41, 140, 35))
    )


@pytest.fixture
def closed_core_source() -> tuple[Any, Any, Any, Any, Any, Any]:
    from phydrax.meshing.providers import NativeLayerCoreSource

    wall_points, wall_triangles = _box((-0.3,) * 3, (0.3,) * 3, 2, outward=True)
    wall = _wall(wall_points, wall_triangles)
    control = _control(
        wall,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        corner=phx.meshing.BoundaryLayerCornerPolicy.FAN,
        minimum_thickness_fraction=0.5,
    )
    policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(wall, control, policy=policy)
    points, triangles = _box((-1.0,) * 3, (1.0,) * 3, 1, outward=True)
    contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
    outer = phx.geometry.surface.SurfaceModel.from_triangles(
        points,
        triangles,
        phx.geometry.surface.SurfaceMetadata(
            source_id="outer-box",
            source_revision="outer-revision",
            coordinate_contract=contract,
            provenance=("authored-outer",),
        ),
    )
    source = NativeLayerCoreSource.fixed_core(
        layers,
        outer,
        contract,
        "layer-core",
        "revision",
        region_id="fluid",
        region_entity_id=17,
        region_occurrence_path=("fluid",),
        wall_occurrence_path=("wall",),
        outer_occurrence_path=("outer",),
        layer_control=control,
        layer_policy=policy,
    )
    return source, layers, outer, contract, control, policy


def test_fixed_core_source_binds_disconnected_authored_shells(
    closed_core_source: tuple[Any, Any, Any, Any, Any, Any],
) -> None:
    from copy import copy

    from phydrax.geometry._domain_boundary import MeshingDomainBoundarySource
    from phydrax.geometry._source_identity import source_entity_id
    from phydrax.meshing._layer_core_association import _core_source_identity
    from phydrax.meshing._layer_core_boundary_replay import (
        core_cap_source_keys,
        core_source_facet_keys,
    )
    from phydrax.meshing._layer_core_topology import validate_staged_layer_domain
    from phydrax.meshing.providers import NativeLayerCoreSource

    source, layers, outer, contract, _, _ = closed_core_source
    domain = validate_staged_layer_domain(source)
    assert domain.region_ids == ("fluid",)
    assert source.source_domain is not None
    assert isinstance(source.fidelity_source, MeshingDomainBoundarySource)
    assert layers.cap is not None
    assert len(source.cap_polygon_ids) == sum(
        block.cell_count for block in layers.cap.blocks
    )
    assert source.source_boundary_charts is not None
    assert (
        source.source_boundary_charts.coordinate_contract.spatial_id
        == contract.spatial_id
    )
    np.testing.assert_array_equal(
        np.asarray(source.complex.vertices)[np.asarray(source.vertex_layer_ids) < 0],
        np.asarray(outer.mesh.coordinates),
    )
    restored = NativeLayerCoreSource.from_stored_fields(source)
    assert restored.binding_id == source.binding_id
    validate_staged_layer_domain(restored)
    facets = np.asarray(source.complex.polygon_facets, dtype=np.int64)
    for polygon in np.asarray(source.cap_polygon_ids, dtype=np.int64):
        facet = int(facets[int(polygon)])
        identifier = source.complex.facet_ids[facet]
        path = source.complex.facet_occurrence_paths[facet]
        assert _core_source_identity(source, 2, identifier, path) == (
            source_entity_id(
                source.source_id,
                source.source_revision,
                "face",
                identifier,
                occurrence_path=path,
            ),
            path,
        )
    expected_cap_keys = frozenset(
        _core_source_identity(
            source,
            2,
            source.complex.facet_ids[int(facets[int(polygon)])],
            source.complex.facet_occurrence_paths[int(facets[int(polygon)])],
        )
        for polygon in np.asarray(source.cap_polygon_ids, dtype=np.int64)
    )
    assert core_cap_source_keys(restored) == expected_cap_keys
    assert expected_cap_keys <= core_source_facet_keys(restored)
    first_polygon = int(source.cap_polygon_ids[0])
    first_facet = int(facets[first_polygon])
    altered_complex = copy(source.complex)
    altered_points = np.asarray(source.complex.vertices).copy()
    offsets = np.asarray(source.complex.polygon_offsets)
    first_vertex = int(
        np.asarray(source.complex.polygon_vertices)[int(offsets[first_polygon])]
    )
    altered_points[first_vertex, 0] += 0.125
    object.__setattr__(altered_complex, "vertices", altered_points)
    altered_source = copy(source)
    object.__setattr__(altered_source, "complex", altered_complex)
    with pytest.raises(ValueError, match="accepted layer triangle"):
        _core_source_identity(
            altered_source,
            2,
            source.complex.facet_ids[first_facet],
            source.complex.facet_occurrence_paths[first_facet],
        )
    scope = phx.meshing.MeshingScope.whole(source, 2)
    kinds = tuple(
        sorted({block.cell_kind for block in layers.mesh.blocks} - {"tetrahedron"})
    )
    specification = phx.meshing.VolumeMeshingSpec(
        phx.meshing.CellMeshingTarget(
            3,
            3,
            phx.meshing.CellFamilyPolicy(
                required=("tetrahedron",), allowed_transitions=kinds, allow_mixed=True
            ),
        ),
        scope,
        phx.meshing.VolumeFillStrategy.SIMPLEX,
        size_controls=(
            phx.meshing.UniformSizeControl(
                scope, 0.5, strength=phx.meshing.SizeControlStrength.SOFT
            ),
        ),
        maximum_deviation=0.0,
    )
    provider = phx.meshing.NativeMeshingProvider()
    for wrong in (
        phx.SpatialCoordinateContract.si(),
        phx.SpatialCoordinateContract(
            phx.units.MILLIMETER, reference_frame="other-frame"
        ),
    ):
        with pytest.raises(phx.meshing.MeshingFailure, match="coordinate_contract"):
            provider.plan(source, specification, coordinate_contract=wrong)
    forged = copy(contract)
    object.__setattr__(forged, "reference_frame", "forged-frame")
    with pytest.raises(ValueError, match="fingerprint"):
        provider.plan(source, specification, coordinate_contract=forged)
    forged_charts = copy(source.source_boundary_charts)
    object.__setattr__(forged_charts, "coordinate_contract", forged)
    forged_source = copy(source)
    object.__setattr__(forged_source, "source_boundary_charts", forged_charts)
    with pytest.raises(ValueError, match="fingerprint"):
        provider.plan(forged_source, specification, coordinate_contract=contract)

    stale = copy(source)
    object.__setattr__(stale, "source_revision", "stale")
    with pytest.raises(ValueError):
        validate_staged_layer_domain(stale)

    changed = copy(source.complex)
    sides = np.asarray(changed.facet_regions).copy()
    cap_facet = int(np.asarray(changed.polygon_facets)[int(source.cap_polygon_ids[0])])
    sides[cap_facet] = sides[cap_facet, ::-1]
    object.__setattr__(changed, "facet_regions", sides)
    wrong_side = copy(source)
    object.__setattr__(wrong_side, "complex", changed)
    with pytest.raises(ValueError):
        validate_staged_layer_domain(wrong_side)

    query = copy(source.fidelity_source)
    object.__setattr__(query, "patches", query.patches[:-1])
    missing_chart = copy(source)
    object.__setattr__(missing_chart, "fidelity_source", query)
    with pytest.raises(ValueError):
        validate_staged_layer_domain(missing_chart)


def test_fixed_core_source_cold_authoring(
    closed_core_source: tuple[Any, Any, Any, Any, Any, Any],
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle import Composition, CompositionEntry
    from phydrax.meshing._layer_core_topology import validate_staged_layer_domain
    from phydrax.meshing._model_codec import (
        layer_core_authoring_recipe,
        MODEL_ARCHIVE_LIMITS,
        read_model_artifact,
        restore_layer_core_authoring,
        restore_model_archive_value,
        write_model_artifact,
    )

    source, _, _, contract, control, policy = closed_core_source
    author = layer_core_authoring_recipe(source, control, policy)
    model = Composition(
        (
            CompositionEntry(
                author,
                entry_id="source",
                role="topology",
                owner_id="phydrax.meshing",
                structure_id=source.binding_id,
                revision_id=source.binding_id,
                semantics_id="closed-fixed-core",
            ),
        ),
        boundary_id="closed-fixed-core-test",
    )
    archive = write_model_artifact(
        tmp_path / "closed-fixed-core.phx", model, limits=MODEL_ARCHIVE_LIMITS
    )
    decoded = read_model_artifact(
        archive,
        expected_type=Composition,
        restore=restore_model_archive_value,
        limits=MODEL_ARCHIVE_LIMITS,
    )
    cold = restore_layer_core_authoring(decoded.value("source"))
    assert cold.binding_id == source.binding_id
    assert cold.source_boundary_charts is not None
    assert (
        cold.source_boundary_charts.coordinate_contract.spatial_id == contract.spatial_id
    )
    validate_staged_layer_domain(cold)


def test_closed_inward_cap_only_core_retains_original_wall_charts() -> None:
    from copy import copy

    from phydrax.meshing._boundary_layer import BoundaryLayerMesh
    from phydrax.meshing._layer_core_association import _is_exact_closed_cap_facet
    from phydrax.meshing._layer_core_topology import (
        closed_fixed_core_source,
        validate_staged_layer_domain,
    )
    from phydrax.meshing._plc_source import _numeric_entity_ids
    from phydrax.meshing.providers import NativeLayerCoreSource

    points, triangles = _box((-1.0,) * 3, (1.0,) * 3, 2, outward=False)
    wall = _wall(points, triangles)
    control = _control(wall)
    policy = phx.meshing.BoundaryLayerPolicy()
    layers = phx.meshing.prepare_boundary_layers(wall, control, policy=policy)
    contract = phx.SpatialCoordinateContract(phx.units.MILLIMETER)
    region_entity_id = _numeric_entity_ids("plc-region", ("fluid",), ((),))[0]

    def build(
        accepted_layers: BoundaryLayerMesh, outer_path: tuple[str, ...] | None = None
    ) -> NativeLayerCoreSource:
        return closed_fixed_core_source(
            accepted_layers,
            None,
            contract,
            str(wall.mesh_id),
            str(wall.numeric_version),
            region_id="fluid",
            region_entity_id=region_entity_id,
            region_occurrence_path=(),
            wall_occurrence_path=(),
            outer_occurrence_path=outer_path,
            layer_control=control,
            layer_policy=policy,
        )

    source = build(layers)
    assert source.source_wall_charts is not None
    assert source.source_wall_charts.mesh is wall
    assert source.source_boundary_charts is not None
    assert source.maximum_deviation == 0.0
    assert np.all(np.asarray(source.vertex_layer_ids) >= 0)
    assert len(source.complex.facet_ids) == len(source.cap_polygon_ids)
    assert source.complex.facet_ids == tuple(
        f"{wall.mesh_id}:cap-facet:{row}" for row in range(len(source.cap_polygon_ids))
    )
    np.testing.assert_array_equal(
        source.complex.facet_regions, np.tile((0, -1), (len(source.cap_polygon_ids), 1))
    )
    facet = int(np.asarray(source.complex.polygon_facets)[int(source.cap_polygon_ids[0])])
    assert _is_exact_closed_cap_facet(source, facet)
    wrong_sides = copy(source)
    wrong_complex = copy(source.complex)
    sides = np.asarray(source.complex.facet_regions).copy()
    sides[facet] = (-1, 0)
    object.__setattr__(wrong_complex, "facet_regions", sides)
    object.__setattr__(wrong_sides, "complex", wrong_complex)
    with pytest.raises(ValueError, match="identity or material sides"):
        _is_exact_closed_cap_facet(wrong_sides, facet)
    validate_staged_layer_domain(source)
    assert (
        NativeLayerCoreSource.from_stored_fields(source).binding_id == source.binding_id
    )

    with pytest.raises(ValueError, match="no outer occurrence"):
        build(layers, ("outer",))
    wrong_wall = copy(layers)
    object.__setattr__(wrong_wall, "source_wall", None)
    with pytest.raises(ValueError, match="actual closed accepted layer cap"):
        build(wrong_wall)
