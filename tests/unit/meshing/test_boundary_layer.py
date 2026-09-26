import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import meshcore_available


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="exact layer certification requires meshcore"
)

SCHEDULE = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)


def _wall(points, triangles):
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


def _whole(mesh):
    cells = mesh.entity_set(2)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )


def _control(mesh, **options):
    return phx.meshing.BoundaryLayerControl(
        _whole(mesh),
        SCHEDULE,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        **options,
    )


def _plate(count, height=0.0, *, flip=False):
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


def _oriented(points, triangles, center, outward):
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    away = np.sum(normals * (corners.mean(axis=1) - center), axis=1) > 0.0
    flip = away != outward
    triangles = triangles.copy()
    triangles[flip] = triangles[flip][:, ::-1]
    return triangles


def _icosphere(radius, subdivisions):
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

        def split(first, second):
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


def _box(lower, upper, count, *, outward):
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    index: dict[tuple[float, ...], int] = {}
    points = []
    triangles = []
    grid = np.linspace(0.0, 1.0, count + 1)

    def vertex(value):
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


def _cell_counts(result):
    return dict(result.evidence.cell_counts)


def _assert_certified(result):
    cells = sum(block.cell_count for block in result.mesh.blocks)
    assert result.validity.certified_valid_count == cells
    assert result.evidence.certified_valid_count == cells


def test_flat_wall_layers_realize_the_exact_schedule():
    points, triangles = _plate(5)
    wall = _wall(points, triangles)

    result = phx.meshing.prepare_boundary_layers(wall, _control(wall))

    np.testing.assert_allclose(
        result.evidence.achieved_thicknesses, SCHEDULE.thicknesses, rtol=1e-12
    )
    np.testing.assert_allclose(result.evidence.achieved_growth_rates, 1.2, rtol=1e-12)
    assert _cell_counts(result) == {"prism": 3 * triangles.shape[0]}
    heights = np.unique(np.round(np.asarray(result.mesh.coordinates)[:, 2], 12))
    np.testing.assert_allclose(heights, (0.0, 0.01, 0.022, 0.0364))
    _assert_certified(result)
    assert not result.closed_cap


def test_quadrilateral_walls_grow_hexahedra_closed_by_transition_pyramids():
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
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    layer_index = np.asarray(result.layer_index)
    pyramids = result.mesh.block("pyramids")
    np.testing.assert_array_equal(layer_index[np.asarray(pyramids.global_ids)], 3)
    _assert_certified(result)


def test_local_terminations_close_columns_with_pyramids_and_tetrahedra():
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


def test_curved_wall_layers_realize_thickness_and_growth_along_the_wall_distance():
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
    radii = np.linalg.norm(np.asarray(result.cap.coordinates), axis=1)
    assert np.all(radii > 0.4)
    _assert_certified(result)


def test_convex_corners_fan_into_certified_hexahedra_pyramids_and_corner_tetrahedra():
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
    assert {block.cell_kind for block in result.cap.blocks} == {"triangle"}
    _assert_certified(result)


def test_concave_right_angle_corners_are_stretched_and_certified():
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


def test_acute_concave_wedge_is_rejected_with_its_crease_vertices():
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


def test_rim_columns_slide_along_the_adjacent_side_surfaces():
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


def _channel(gap):
    lower, lower_triangles = _plate(4)
    upper, upper_triangles = _plate(4, gap, flip=True)
    return _wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )


def test_opposing_channel_walls_fail_with_colliding_vertex_evidence():
    wall = _channel(0.05)

    with pytest.raises(phx.meshing.MeshingFailure) as failure:
        phx.meshing.prepare_boundary_layers(wall, _control(wall))

    assert failure.value.category is phx.meshing.MeshingFailureCategory.CONTROL_CONFLICT
    assert len(failure.value.entity_ids) == 50
    assert len(failure.value.locations) == 50


@pytest.mark.parametrize(
    "collision",
    (
        phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY,
        phx.meshing.BoundaryLayerCollisionPolicy.MERGE,
    ),
)
def test_opposing_channel_walls_resolve_by_the_explicit_collision_policy(collision):
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
        case phx.meshing.BoundaryLayerCollisionPolicy.TERMINATE_LOCALLY:
            assert evidence.terminated_vertex_count == 50
            assert _cell_counts(result) == {"prism": 2 * 32}
            assert evidence.achieved_thicknesses[0] == pytest.approx(0.01)
            assert np.all(np.isnan(evidence.achieved_thicknesses[1:]))
        case phx.meshing.BoundaryLayerCollisionPolicy.MERGE:
            assert evidence.merged_vertex_count == 50
            assert evidence.minimum_scale == pytest.approx(
                0.5 * gap / SCHEDULE.total_thickness
            )
            assert np.any(np.isclose(heights, 0.5 * gap))
            # Merged fronts share their midsurface, so no free front remains.
            assert result.cap is None
    _assert_certified(result)
