"""Binding for the bounded implicit contour batch kernel."""

from __future__ import annotations

import numpy as np


def contours(
    corner_signs: np.ndarray,
    corner_values: np.ndarray,
    boxes: np.ndarray,
    edge_roots: np.ndarray,
    vertex_by_root: np.ndarray,
    root_points: np.ndarray,
    root_issues: np.ndarray,
    root_errors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run a native batch.

    Unhandled rows still require the canonical scalar contour path. A native
    triangle is kept only when its orientation is certified against the root
    error bounds by the same predicate as the scalar path; otherwise its row is
    left unhandled, so the scalar path decides it. Arrays passed to the C ABI
    remain owned by this call until it returns.
    """
    from phydrax._meshcore import _raise_for_call, load_meshcore

    from ._adaptive_discovery import (
        _certified_triangle_orientations,
        _CORNER_OFFSETS,
        _widest_corner_offsets,
    )

    function = load_meshcore()["phx_mc_implicit_contours"]
    signs = np.ascontiguousarray(corner_signs, dtype=np.int8)
    values = np.ascontiguousarray(corner_values, dtype=np.float64)
    bounds = np.ascontiguousarray(boxes, dtype=np.float64)
    roots = np.ascontiguousarray(edge_roots, dtype=np.int64)
    vertices = np.ascontiguousarray(vertex_by_root, dtype=np.int64)
    points = np.ascontiguousarray(root_points, dtype=np.float64)
    root_status = np.ascontiguousarray(root_issues, dtype=np.int64)
    root_bounds = np.asarray(root_errors, dtype=np.float64)
    if signs.ndim != 2 or signs.shape[1] != 8:
        raise ValueError("corner_signs must have shape (n, 8)")
    n = len(signs)
    if (
        values.shape != (n, 8)
        or bounds.shape != (n, 2, 3)
        or roots.shape != (n, 12)
        or vertices.ndim != 1
        or points.shape != (len(vertices), 3)
        or root_status.shape != vertices.shape
        or root_bounds.shape != vertices.shape
    ):
        raise ValueError("inconsistent implicit contour batch shapes")
    handled = np.zeros(n, dtype=np.bool_)
    counts = np.zeros(n, dtype=np.int8)
    triangles = np.zeros((n, 10, 3), dtype=np.int64)
    issues = np.zeros(n, dtype=np.int64)
    arrays = (signs, values, bounds, roots, vertices, points, root_status)
    outputs = (handled, counts, triangles, issues)
    status = function(
        n,
        len(vertices),
        *(a.ctypes.data for a in arrays),
        0.0,
        *(a.ctypes.data for a in outputs),
    )
    _raise_for_call(status, "implicit_contours")
    rows = np.flatnonzero(handled)
    if not len(rows):
        return handled, counts, triangles, issues
    # Native triangles name encounter-ordered vertices; recover their roots.
    row_roots = roots[rows]
    row_vertices = np.where(row_roots >= 0, vertices[np.maximum(row_roots, 0)], -1)
    slots = np.arange(10)[None, :] < counts[rows, None].astype(np.int64)
    triangle_vertices = triangles[rows]
    matches = (triangle_vertices[:, :, :, None] == row_vertices[:, None, None, :]) & (
        row_vertices[:, None, None, :] >= 0
    )
    if np.any(np.sum(matches, axis=3)[slots] != 1):
        raise RuntimeError("Native contour triangles must use their own cell roots.")
    triangle_roots = np.take_along_axis(
        row_roots[:, None, None, :], np.argmax(matches, axis=3)[..., None], axis=3
    )[..., 0]
    # The same corner-mean reference gradient as the scalar contour path.
    cell_values = values[rows]
    gradients = np.stack(
        [
            (
                np.mean(cell_values[:, _CORNER_OFFSETS[:, axis] == 1], axis=1)
                - np.mean(cell_values[:, _CORNER_OFFSETS[:, axis] == 0], axis=1)
            )
            / (bounds[rows, 1, axis] - bounds[rows, 0, axis])
            for axis in range(3)
        ],
        axis=1,
    )
    selected = triangle_roots[slots]
    certified = np.ones(slots.shape, dtype=np.bool_)
    certified[slots] = _certified_triangle_orientations(
        points[selected],
        root_bounds[selected],
        np.broadcast_to(gradients[:, None, :], (*slots.shape, 3))[slots],
    )
    # Kept triangles start at their corner opposite the longest edge, as the
    # scalar path's do.
    rotation = (_widest_corner_offsets(points[selected])[:, None] + np.arange(3)) % 3
    row_triangles = triangles[rows]
    row_triangles[slots] = np.take_along_axis(row_triangles[slots], rotation, axis=1)
    triangles[rows] = row_triangles
    rejected = rows[~np.all(certified, axis=1)]
    handled[rejected] = False
    counts[rejected] = 0
    triangles[rejected] = 0
    issues[rejected] = 0
    return handled, counts, triangles, issues
