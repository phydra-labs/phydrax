#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Directed distance bounds from triangles to a triangle surface."""

from __future__ import annotations

import numpy as np

from .._bvh import BVHBuildPolicy, host_tree_arrays, prepare_bvh
from .._bvh_native import triangle_nearest
from .._meshcore import point_triangle_distances


# A piece splits while its bound exceeds the running lower bound by this
# relative margin and it is larger than _FINEST of its original triangle.
_SPLIT_MARGIN = 1.0 / 16.0
_FINEST = 1.0 / 64.0


def _diameters(triangles: np.ndarray, /) -> np.ndarray:
    edges = triangles - np.roll(triangles, 1, axis=1)
    return np.max(np.linalg.norm(edges, axis=-1), axis=-1)


def directed_distance_bounds(
    triangles: np.ndarray, vertices: np.ndarray, faces: np.ndarray, /
) -> tuple[float, float]:
    """Binary64 bounds of ``sup_x dist(x, S)`` over ``triangles`` (n, 3, 3).

    ``S`` is the surface ``vertices[faces]``. The distance to one triangle is
    convex, so over a piece it peaks at a corner: the largest corner distance
    to the triangle of ``S`` nearest the piece's centroid bounds the piece
    from above, and the centroid's distance to ``S`` bounds the supremum from
    below. A piece splits into four while its bound exceeds the running lower
    bound by more than 1/16 and it is larger than 1/64 of its triangle. The
    values are computed in binary64 and carry no rounding allowance.
    """
    pieces = np.asarray(triangles, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64)
    rows = np.asarray(faces, dtype=np.int32)
    surface = points[rows]
    tree = host_tree_arrays(
        prepare_bvh(
            np.min(surface, axis=1),
            np.max(surface, axis=1),
            policy=BVHBuildPolicy(leaf_size=8),
            dtype=np.float64,
        )
    )
    finest = _FINEST * _diameters(pieces)
    lower = 0.0
    upper = 0.0
    while pieces.shape[0]:
        nearest = triangle_nearest(tree, points, rows, np.mean(pieces, axis=1), k=1)
        lower = max(lower, float(np.sqrt(np.max(nearest.distance_squared[:, 0]))))
        corners = point_triangle_distances(
            pieces.reshape((-1, 3)), np.repeat(surface[nearest.faces[:, 0]], 3, axis=0)
        )
        bounds = np.max(corners.reshape((-1, 3)), axis=1)
        done = (bounds <= lower * (1.0 + _SPLIT_MARGIN)) | (_diameters(pieces) <= finest)
        if np.any(done):
            upper = max(upper, float(np.max(bounds[done])))
        rest = pieces[~done]
        first, second, third = rest[:, 0], rest[:, 1], rest[:, 2]
        near, middle, far = (
            0.5 * (first + second),
            0.5 * (second + third),
            0.5 * (third + first),
        )
        pieces = np.concatenate(
            (
                np.stack((first, near, far), axis=1),
                np.stack((near, second, middle), axis=1),
                np.stack((far, middle, third), axis=1),
                np.stack((near, middle, far), axis=1),
            )
        )
        finest = np.tile(finest[~done], 4)
    return lower, upper


__all__ = ["directed_distance_bounds"]
