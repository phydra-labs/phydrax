from __future__ import annotations

from itertools import product

import numpy as np


_INTERVAL_CHILDREN = ((0.0, 0.5), (0.5, 0.5))
_TRIANGLE_CHILDREN = (
    ((0.0, 0.0), (0.5, 0.0), (0.0, 0.5)),
    ((0.5, 0.0), (1.0, 0.0), (0.5, 0.5)),
    ((0.0, 0.5), (0.5, 0.5), (0.0, 1.0)),
    ((0.5, 0.5), (0.0, 0.5), (0.5, 0.0)),
)


def _tetrahedron_children() -> tuple[np.ndarray, ...]:
    vertices = np.eye(4, 3, k=-1)
    middle = {
        (first, second): 0.5 * (vertices[first] + vertices[second])
        for first in range(4)
        for second in range(first + 1, 4)
    }
    corners = (
        (vertices[0], middle[0, 1], middle[0, 2], middle[0, 3]),
        (middle[0, 1], vertices[1], middle[1, 2], middle[1, 3]),
        (middle[0, 2], middle[1, 2], vertices[2], middle[2, 3]),
        (middle[0, 3], middle[1, 3], middle[2, 3], vertices[3]),
    )
    # Red refinement: the inner octahedron is split around diagonal m02-m13,
    # whose equatorial cycle is m01, m03, m23, m12.
    cycle = (middle[0, 1], middle[0, 3], middle[2, 3], middle[1, 2])
    inner = tuple(
        (middle[0, 2], middle[1, 3], cycle[index], cycle[(index + 1) % 4])
        for index in range(4)
    )
    return tuple(np.asarray(value) for value in (*corners, *inner))


def simplex_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    if dimension == 2:
        children = tuple(np.asarray(value) for value in _TRIANGLE_CHILDREN)
    elif dimension == 3:
        children = _tetrahedron_children()
    elif dimension > 0:
        vertices = np.eye(dimension + 1, dimension, k=-1, dtype=np.float64)
        midpoint = 0.5 * (vertices[0] + vertices[1])
        first = vertices.copy()
        second = vertices.copy()
        first[1] = midpoint
        second[0] = midpoint
        children = (first, second)
    else:
        raise ValueError(
            f"simplex_child_maps.dimension must be positive; got {dimension!r}"
        )
    origins = np.stack([child[0] for child in children])
    matrices = np.stack([(child[1:] - child[0]).T for child in children])
    return origins, matrices


def box_child_maps(dimension: int, /) -> tuple[np.ndarray, np.ndarray]:
    choices = tuple(product(_INTERVAL_CHILDREN, repeat=dimension))
    origins = np.asarray([[origin for origin, _ in choice] for choice in choices])
    matrices = np.stack(
        [np.diag([scale for _, scale in choice]) for choice in choices]
    ).reshape(-1, dimension, dimension)
    return origins, matrices


def prism_child_maps() -> tuple[np.ndarray, np.ndarray]:
    triangle_origins, triangle_matrices = simplex_child_maps(2)
    origins = []
    matrices = []
    for origin, matrix in zip(triangle_origins, triangle_matrices, strict=True):
        for start, scale in _INTERVAL_CHILDREN:
            origins.append(np.concatenate((origin, (start,))))
            block = np.zeros((3, 3))
            block[:2, :2] = matrix
            block[2, 2] = scale
            matrices.append(block)
    return np.asarray(origins), np.asarray(matrices)
