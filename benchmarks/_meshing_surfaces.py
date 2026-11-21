"""Generated benchmark surfaces shared by the native meshing rows and their
reference drivers (NumPy only, so reference environments can import it)."""

from __future__ import annotations

import numpy as np


def icosphere(level: int, /) -> tuple[np.ndarray, np.ndarray]:
    # The ledger's unit icosphere: 20 * 4**level outward triangles.
    golden = (1.0 + 5.0**0.5) / 2.0
    vertices = np.asarray(
        [
            [-1, golden, 0], [1, golden, 0], [-1, -golden, 0], [1, -golden, 0],
            [0, -1, golden], [0, 1, golden], [0, -1, -golden], [0, 1, -golden],
            [golden, 0, -1], [golden, 0, 1], [-golden, 0, -1], [-golden, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.asarray(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11], [1, 5, 9],
            [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8], [3, 9, 4], [3, 4, 2],
            [3, 2, 6], [3, 6, 8], [3, 8, 9], [4, 9, 5], [2, 4, 11], [6, 2, 10],
            [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        edges = np.sort(
            np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        middle = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        middle /= np.linalg.norm(middle, axis=1, keepdims=True)
        count = faces.shape[0]
        inverse = inverse.reshape((-1,)) + vertices.shape[0]
        a, b, c = inverse[:count], inverse[count : 2 * count], inverse[2 * count :]
        vertices = np.concatenate((vertices, middle))
        faces = np.concatenate(
            (
                np.stack((faces[:, 0], a, c), axis=1),
                np.stack((faces[:, 1], b, a), axis=1),
                np.stack((faces[:, 2], c, b), axis=1),
                np.stack((a, b, c), axis=1),
            )
        )
    return vertices, faces
