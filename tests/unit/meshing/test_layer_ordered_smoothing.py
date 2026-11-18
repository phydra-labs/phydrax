"""Order-sensitive binary64 regressions at the layer smoothing boundaries."""

import math
from fractions import Fraction

import numpy as np

import phydrax as phx
from phydrax.meshing import _boundary_layer as layer


def _add(a: float, b: float) -> float:
    return float(Fraction(a) + Fraction(b))


def _mul(a: float, b: float) -> float:
    return float(Fraction(a) * Fraction(b))


def _div(a: float, b: float) -> float:
    return float(Fraction(a) / Fraction(b))


def _unit(vector: list[float]) -> list[float]:
    squares = [_mul(value, value) for value in vector]
    length = math.sqrt(_add(_add(squares[0], squares[1]), squares[2]))
    return [_div(value, length) for value in vector]


def _fan(
    heights: list[float] | None = None,
    radii: list[float] | None = None,
) -> tuple[
    phx.discretization.CellMesh,
    layer._Wall,
    layer._Sectors,
    phx.meshing.BoundaryLayerControl,
    list[float],
]:
    count = 16
    # Dyadic ring directions avoid platform libm in this order-sensitive fixture.
    ring = (
        np.asarray(
            [
                (1048576, 0),
                (968758, 401273),
                (741455, 741455),
                (401273, 968758),
                (0, 1048576),
                (-401273, 968758),
                (-741455, 741455),
                (-968758, 401273),
                (-1048576, 0),
                (-968758, -401273),
                (-741455, -741455),
                (-401273, -968758),
                (0, -1048576),
                (401273, -968758),
                (741455, -741455),
                (968758, -401273),
            ],
            dtype=np.float64,
        )
        / 1048576.0
    )
    z = np.asarray(heights if heights is not None else [0.0] * count, dtype=np.float64)
    points = np.vstack(
        (
            np.zeros(3, dtype=np.float64),
            np.column_stack(
                (
                    ring[:, 0]
                    * np.asarray(
                        radii if radii is not None else [1.0] * count, dtype=np.float64
                    ),
                    ring[:, 1]
                    * np.asarray(
                        radii if radii is not None else [1.0] * count, dtype=np.float64
                    ),
                    z,
                )
            ),
        )
    )
    faces = np.asarray(
        [(0, i + 1, (i + 1) % count + 1, -1) for i in range(count)],
        dtype=np.int64,
    )
    wall = layer._analyze_wall(
        points,
        faces,
        np.full(count, 3, dtype=np.int64),
        np.ones(count, dtype=np.bool_),
        np.pi / 6.0,
    )
    sectors = layer._sectors(wall, np.zeros(wall.edges.shape[0], dtype=np.bool_))
    mesh = phx.discretization.CellMesh.from_triangles(points, faces[:, :3])
    entities = mesh.entity_set(2)
    scope = phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        entities.entity_set_id,
        entities.entity_ids,
    )
    control = phx.meshing.BoundaryLayerControl(
        scope,
        phx.meshing.LayerSchedule.geometric(1, 0.001),
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        smoothing_iterations=1,
    )
    # Independent weights for the 16 radial edges, in vertex-index order.
    weights = []
    for point in points[1:]:
        squares = [_mul(float(component), float(component)) for component in point]
        length = math.sqrt(_add(_add(squares[0], squares[1]), squares[2]))
        weights.append(_div(1.0, length))
    return mesh, wall, sectors, control, weights


def _sum(values: list[float]) -> float:
    result = 0.0
    for value in values:
        result = _add(result, value)
    return result


def test_scale_weighted_bincount_is_index_ordered() -> None:
    radii = [
        1.0,
        0.6,
        1.3,
        0.8,
        2.1,
        0.3,
        1.3,
        1.7,
        0.3,
        0.4,
        1.7,
        0.4,
        1.3,
        1.7,
        0.6,
        1.3,
    ]
    samples = [13, 4, 4, 11, 7, 11, 6, 8, 6, 11, 9, 4, 4, 6, 13, 9]
    mesh, wall, _, base_control, weights = _fan(radii=radii)
    control = phx.meshing.BoundaryLayerControl(
        base_control.wall_scope,
        base_control.schedule,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        collision=phx.meshing.BoundaryLayerCollisionPolicy.REDUCE_THICKNESS,
        minimum_thickness_fraction=0.01,
        smoothing_iterations=1,
    )
    obstacle_points = []
    for point, sample in zip(wall.points[1:], samples, strict=True):
        x, y = float(point[0]), float(point[1])
        z = 2 * (sample + 0.5) / 16 * 0.001 / 0.8
        obstacle_points.extend(
            ((x - 0.01, y - 0.01, z), (x + 0.01, y - 0.01, z), (x, y + 0.01, z))
        )
    obstacles = phx.discretization.CellMesh.from_triangles(
        np.asarray(obstacle_points, dtype=np.float64),
        np.arange(48, dtype=np.int64).reshape(-1, 3),
    )
    # The obstacle crosses the medial test between samples k and k+1.
    reach = _div(0.001, 0.8)
    values = [_div(_mul(0.8, _mul(reach, _div(float(k), 16.0))), 0.001) for k in samples]
    total = _sum(weights)
    products = [
        _mul(weight, value) for weight, value in zip(weights, values, strict=True)
    ]

    def expected(products: list[float]) -> float:
        return min(1.0, _mul(0.5, _add(1.0, _div(_sum(products), total))))

    forward = expected(products)
    reverse = expected(products[::-1])
    assert float(forward).hex() != float(reverse).hex()
    result = phx.meshing.prepare_boundary_layers(mesh, control, obstacles=obstacles)
    assert float(np.asarray(result.column_scale)[0]).hex() == float(forward).hex()


def test_direction_add_at_is_index_ordered() -> None:
    heights = [
        0.03969503522277448,
        -0.014105769441679537,
        -0.003684780407193,
        0.013383715716291614,
        0.04272454770532787,
        0.038745024895400465,
        -0.026089900468445615,
        -0.039074164550580354,
        0.04708103054832234,
        -0.005501801308974853,
        0.010767143848749626,
        -0.02890365858882106,
        -0.04144381433535156,
        -0.04031971207318942,
        -0.007190145117348459,
        -0.028140590099320774,
    ]
    mesh, wall, _, control, weights = _fan(heights)
    split = layer._split_ridges(wall, control.corner)
    sectors = layer._sectors(wall, split)
    unsmoothed_control = phx.meshing.BoundaryLayerControl(
        control.wall_scope,
        control.schedule,
        route=control.route,
        smoothing_iterations=0,
    )
    direction = layer._sector_directions(wall, sectors, unsmoothed_control).direction
    total = _sum(weights)
    products = [
        [_mul(weight, float(direction[i + 1, k])) for k in range(3)]
        for i, weight in enumerate(weights)
    ]

    def expected(rows: list[list[float]]) -> list[float]:
        average = [_div(_sum([row[k] for row in rows]), total) for k in range(3)]
        candidate = [
            _add(
                float(direction[0, k]),
                _mul(0.5, _add(average[k], -float(direction[0, k]))),
            )
            for k in range(3)
        ]
        return _unit(_unit(candidate))

    forward = expected(products)
    reverse = expected(products[::-1])
    result = phx.meshing.prepare_boundary_layers(mesh, control)
    stretch = float(np.asarray(result.column_stretch)[0])
    height = _mul(stretch, float(control.schedule.thicknesses[0]))
    forward_coordinates = [_mul(height, value) for value in forward]
    reverse_coordinates = [_mul(height, value) for value in reverse]
    assert [float(value).hex() for value in forward_coordinates] != [
        float(value).hex() for value in reverse_coordinates
    ]
    coordinates = np.asarray(result.column_level_coordinates, dtype=np.float64)[0, 0]
    assert [float(value).hex() for value in coordinates] == [
        float(value).hex() for value in forward_coordinates
    ]
