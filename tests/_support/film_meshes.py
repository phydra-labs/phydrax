#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic triangle meshes for surface thin-film tests."""

from __future__ import annotations

import numpy as np

from phydrax.geometry.simplicial import TriangleMesh


def planar_grid(
    nx: int, ny: int, length_x: float = 1.0, length_y: float = 1.0
) -> TriangleMesh:
    """Right-triangle grid on ``[0, Lx] x [0, Ly]`` in the ``z = 0`` plane."""
    xs = np.linspace(0.0, length_x, nx + 1)
    ys = np.linspace(0.0, length_y, ny + 1)
    x, y = np.meshgrid(xs, ys, indexing="ij")
    vertices = np.stack((x.ravel(), y.ravel(), np.zeros(x.size)), axis=1)
    index = np.arange((nx + 1) * (ny + 1)).reshape((nx + 1, ny + 1))
    lower = np.stack((index[:-1, :-1], index[1:, :-1], index[1:, 1:]), axis=-1).reshape(
        (-1, 3)
    )
    upper = np.stack((index[:-1, :-1], index[1:, 1:], index[:-1, 1:]), axis=-1).reshape(
        (-1, 3)
    )
    return TriangleMesh(vertices, np.concatenate((lower, upper)).astype(np.int32))


def icosphere(level: int, radius: float = 1.0) -> TriangleMesh:
    """Outward-oriented subdivided icosahedron projected onto a sphere."""
    ratio = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, ratio, 0),
            (1, ratio, 0),
            (-1, -ratio, 0),
            (1, -ratio, 0),
            (0, -1, ratio),
            (0, 1, ratio),
            (0, -1, -ratio),
            (0, 1, -ratio),
            (ratio, 0, -1),
            (ratio, 0, 1),
            (-ratio, 0, -1),
            (-ratio, 0, 1),
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [
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
        ],
        dtype=np.int64,
    )
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
            axis=1,
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        midpoints = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        midpoints /= np.linalg.norm(midpoints, axis=1, keepdims=True)
        offset = vertices.shape[0]
        count = faces.shape[0]
        ab, bc, ca = (
            offset + inverse[:count],
            offset + inverse[count : 2 * count],
            offset + inverse[2 * count :],
        )
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        faces = np.concatenate(
            (
                np.stack((a, ab, ca), axis=1),
                np.stack((b, bc, ab), axis=1),
                np.stack((c, ca, bc), axis=1),
                np.stack((ab, bc, ca), axis=1),
            )
        )
        vertices = np.concatenate((vertices, midpoints))
    return TriangleMesh(radius * vertices, faces.astype(np.int32))


def obtuse_mesh() -> TriangleMesh:
    """Planar quad split along its long diagonal, giving a negative cotangent."""
    vertices = np.asarray(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.2, 0.15, 0.0), (0.2, 0.15, 0.0)]
    )
    faces = np.asarray([(0, 1, 2), (0, 2, 3)], dtype=np.int32)
    return TriangleMesh(vertices, faces)
