"""Host-authored wall fixtures retained from the pre-remediation layer contracts."""

import numpy as np
from numpy.typing import ArrayLike

import phydrax as phx


# Authored three-layer wall fixture: first thickness 0.01, geometric ratio 1.2.
SCHEDULE = phx.meshing.LayerSchedule.geometric(3, 0.01, growth_rate=1.2)


def wall(points: ArrayLike, triangles: ArrayLike) -> phx.discretization.CellMesh:
    rows = np.asarray(triangles, dtype=np.int64)
    return phx.discretization.CellMesh(
        np.asarray(points, dtype=np.float64),
        (
            phx.discretization.CellBlock(
                "triangles",
                "triangle",
                rows,
                global_ids=np.arange(rows.shape[0], dtype=np.int64),
            ),
        ),
    )


def whole(mesh: phx.discretization.CellMesh) -> phx.meshing.MeshingScope:
    cells = mesh.entity_set(2)
    return phx.meshing.MeshingScope(
        mesh.mesh_id,
        mesh.numeric_version,
        phx.meshing.MeshingEntityKind.MESH,
        2,
        cells.entity_set_id,
        cells.entity_ids,
    )


def control(
    mesh: phx.discretization.CellMesh,
    *,
    collision: phx.meshing.BoundaryLayerCollisionPolicy = phx.meshing.BoundaryLayerCollisionPolicy.FAIL,
    minimum_thickness_fraction: float = 0.5,
) -> phx.meshing.BoundaryLayerControl:
    return phx.meshing.BoundaryLayerControl(
        whole(mesh),
        SCHEDULE,
        route=phx.meshing.BoundaryLayerRoute.ADVANCING,
        collision=collision,
        minimum_thickness_fraction=minimum_thickness_fraction,
    )


def plate(
    count: int, height: float = 0.0, *, flip: bool = False
) -> tuple[np.ndarray, np.ndarray]:
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
    rows = np.asarray(triangles, dtype=np.int64)
    return points, rows[:, ::-1] if flip else rows


def oriented(
    points: np.ndarray, triangles: np.ndarray, center: np.ndarray, outward: bool
) -> np.ndarray:
    corners = points[triangles]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    away = np.sum(normals * (corners.mean(axis=1) - center), axis=1) > 0.0
    flip = away != outward
    triangles = triangles.copy()
    triangles[flip] = triangles[flip][:, ::-1]
    return triangles


def icosphere(radius: float, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
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
        middle: dict[tuple[int, int], int] = {}

        def split(first: int, second: int) -> int:
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
    vertices = radius * np.asarray(points, dtype=np.float64)
    return vertices, oriented(
        vertices, np.asarray(faces, dtype=np.int64), np.zeros(3), True
    )


def box(
    lower: ArrayLike, upper: ArrayLike, count: int, *, outward: bool
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)
    index: dict[tuple[float, ...], int] = {}
    points = []
    triangles = []
    grid = np.linspace(0.0, 1.0, count + 1)

    def vertex(value: np.ndarray) -> int:
        # Decimal face-grid coordinates merge at 12 places.
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
                        value = np.zeros(3, dtype=np.float64)
                        value[axis] = bound
                        value[u] = lower[u] + (upper[u] - lower[u]) * grid[i + di]
                        value[v] = lower[v] + (upper[v] - lower[v]) * grid[j + dj]
                        quad.append(vertex(value))
                    a, b, c, d = quad
                    triangles.extend(((a, b, c), (a, c, d)))
    vertices = np.asarray(points, dtype=np.float64)
    return vertices, oriented(
        vertices, np.asarray(triangles, dtype=np.int64), 0.5 * (lower + upper), outward
    )


def channel(gap: float) -> phx.discretization.CellMesh:
    lower, lower_triangles = plate(4)
    upper, upper_triangles = plate(4, gap, flip=True)
    return wall(
        np.concatenate((lower, upper)),
        np.concatenate((lower_triangles, upper_triangles + lower.shape[0])),
    )


def wedge() -> tuple[np.ndarray, np.ndarray]:
    count = 3
    values = np.linspace(0.0, 1.0, count + 1)
    angle = np.deg2rad(30.0)
    floor = [(r, w, 0.0) for r in values for w in values]
    slope = [
        (r * np.cos(angle), w, r * np.sin(angle)) for r in values[1:] for w in values
    ]
    points = np.asarray(floor + slope, dtype=np.float64)
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
    rows = np.asarray(triangles, dtype=np.int64)
    inside = np.asarray((0.5 * np.cos(0.5 * angle), 0.5, 0.5 * np.sin(0.5 * angle)))
    corners = points[rows]
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    flip = np.sum(normals * (inside - corners.mean(axis=1)), axis=1) < 0.0
    rows[flip] = rows[flip][:, ::-1]
    return points, rows
