#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""The native prism layers against independent references.

Prism validity is recomputed with exact rational arithmetic: the nine
Bernstein controls of each prism's Jacobian (degree one on the triangle, two
along the column) from the binary64 output coordinates. Depths are true
distances to the closed-form solid (the unit sphere, the inner offset of a
box), and the slab's top levels are compared column against column.
"""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

from phydrax import _meshcore
from tests._support.film_meshes import icosphere


pytestmark = pytest.mark.skipif(
    not _meshcore.meshcore_available(), reason="the native prism layers need meshcore"
)

_HEIGHTS = np.cumsum(0.01 * 1.2 ** np.arange(5, dtype=np.float64))


def _sphere() -> tuple[np.ndarray, np.ndarray]:
    mesh = icosphere(2)
    return (
        np.asarray(mesh.vertices, dtype=np.float64),
        np.asarray(mesh.faces, dtype=np.int64),
    )


def _box(size: tuple[float, float, float], count: int) -> tuple[np.ndarray, np.ndarray]:
    # Outward-oriented structured box surface, `count` x `count` quads a face.
    rows: dict[tuple[int, int, int], int] = {}
    points: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    for axis in range(3):
        u, v = (axis + 1) % 3, (axis + 2) % 3
        for side in (0, count):
            grid = np.empty((count + 1, count + 1), dtype=np.int64)
            for i in range(count + 1):
                for j in range(count + 1):
                    lattice = [0, 0, 0]
                    lattice[axis], lattice[u], lattice[v] = side, i, j
                    key = (lattice[0], lattice[1], lattice[2])
                    if key not in rows:
                        rows[key] = len(points)
                        points.append(
                            (
                                size[0] * key[0] / count,
                                size[1] * key[1] / count,
                                size[2] * key[2] / count,
                            )
                        )
                    grid[i, j] = rows[key]
            for i in range(count):
                for j in range(count):
                    a, b = int(grid[i, j]), int(grid[i + 1, j])
                    c, d = int(grid[i + 1, j + 1]), int(grid[i, j + 1])
                    if side == count:
                        triangles.extend(((a, b, c), (a, c, d)))
                    else:
                        triangles.extend(((a, c, b), (a, d, c)))
    return np.asarray(points, dtype=np.float64), np.asarray(triangles, dtype=np.int64)


def _grow(
    vertices: np.ndarray, outward: np.ndarray, *, minimum_scale: float = 0.05
) -> _meshcore.PrismLayers:
    return _meshcore.prism_layers(
        vertices,
        outward[:, ::-1],
        _HEIGHTS,
        feature_angle_degrees=30.0,
        minimum_scale=minimum_scale,
        prove=True,
    )


def _determinant(a: list[Fraction], b: list[Fraction], c: list[Fraction]) -> Fraction:
    return (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )


def _least_bernstein_control(points: np.ndarray, prism: np.ndarray) -> Fraction:
    q = [[Fraction(float(value)) for value in points[index]] for index in prism]

    def edge(head: int, base: int) -> list[Fraction]:
        return [q[head][k] - q[base][k] for k in range(3)]

    a0, b0, a1, b1 = edge(1, 0), edge(2, 0), edge(4, 3), edge(5, 3)
    controls: list[Fraction] = []
    for vertex in range(3):
        column = edge(vertex + 3, vertex)
        controls.extend(
            (
                _determinant(a0, b0, column),
                _determinant(a1, b1, column),
                _determinant(a0, b1, column) + _determinant(a1, b0, column),
            )
        )
    return min(controls)


@pytest.mark.parametrize("wall", ["sphere", "box"])
def test_every_certified_prism_has_positive_exact_bernstein_controls(wall: str) -> None:
    vertices, faces = _sphere() if wall == "sphere" else _box((1.0, 0.8, 0.6), 3)
    layers = _grow(vertices, faces)

    assert layers.status is _meshcore.MeshcoreStatus.OK
    assert layers.stats[16] == 1.0
    assert layers.prisms.shape == (5 * faces.shape[0], 6)
    assert all(
        _least_bernstein_control(layers.points, prism) > 0 for prism in layers.prisms
    )
    # Level 0 is the wall itself, bitwise.
    assert np.array_equal(layers.points[: vertices.shape[0]], vertices)


def test_sphere_levels_lie_at_their_heights_by_true_distance() -> None:
    vertices, faces = _sphere()
    layers = _grow(vertices, faces)
    count = vertices.shape[0]
    radii = np.linalg.norm(layers.points.reshape((6, count, 3)), axis=-1)
    expected = 1.0 - np.concatenate((np.zeros((1,)), _HEIGHTS))

    assert np.max(np.abs(radii - expected[:, None])) < 1e-3 * _HEIGHTS[-1]
    assert np.all(layers.vertex_kind == 0)
    assert np.all(layers.scale == 1.0)


def test_box_tops_form_the_inner_offset_box() -> None:
    size = (1.0, 0.8, 0.6)
    vertices, faces = _box(size, 4)
    layers = _grow(vertices, faces)
    top = layers.points[-vertices.shape[0] :]
    depth = _HEIGHTS[-1]
    upper = np.asarray(size, dtype=np.float64)

    on_low, on_high = vertices == 0.0, vertices == upper
    assert np.allclose(top[on_low], depth, rtol=0.0, atol=1e-12)
    assert np.allclose((upper - top)[on_high], depth, rtol=0.0, atol=1e-12)
    assert np.all(top >= depth - 1e-12) and np.all(upper - top >= depth - 1e-12)
    # 12 edges with three interior vertices each, and the 8 corners.
    assert np.count_nonzero(layers.vertex_kind == 1) == 36
    assert np.count_nonzero(layers.vertex_kind == 2) == 8


def test_slab_columns_are_thinned_until_the_stacks_separate() -> None:
    vertices, faces = _box((1.0, 1.0, 0.1), 6)
    layers = _grow(vertices, faces)
    top = layers.points[-vertices.shape[0] :]
    lower = np.flatnonzero(vertices[:, 2] == 0.0)
    upper = np.flatnonzero(vertices[:, 2] == np.max(vertices[:, 2]))
    pairs = {tuple(vertices[v, :2]): v for v in upper}

    assert layers.status is _meshcore.MeshcoreStatus.OK
    assert layers.stats[16] == 1.0
    assert np.min(layers.scale) < 1.0
    # Over every wall vertex of the lower face, its stack top stays below the
    # top of the column grown down from the upper face.
    assert all(top[v, 2] < top[pairs[tuple(vertices[v, :2])], 2] for v in lower)
    assert all(
        _least_bernstein_control(layers.points, prism) > 0 for prism in layers.prisms
    )


def test_a_slab_without_room_to_thin_is_reported_unresolved() -> None:
    vertices, faces = _box((1.0, 1.0, 0.1), 6)
    layers = _grow(vertices, faces, minimum_scale=1.0)

    assert layers.status is _meshcore.MeshcoreStatus.UNRESOLVED
    assert layers.stats[16] == 0.0


def test_an_open_wall_is_refused() -> None:
    vertices, faces = _sphere()
    with pytest.raises(ValueError, match="closed"):
        _grow(vertices, faces[:-1])
