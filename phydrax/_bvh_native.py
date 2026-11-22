#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native traversals of one host ``PackedBVH``: the meshcore ``BVH`` C ABI.

Python ``prepare_bvh`` remains the only tree builder; native code validates
and traverses its packed arrays (``phx_mc_bvh_tree_view``). The point queries
are batched and single threaded and write caller-owned arrays allocated here.
Generic owners dispatch to them only for concrete float64 inputs while the
library is present (``native_point_queries``); traced inputs and a missing
library keep their JAX routes.
"""

from __future__ import annotations

import ctypes
from typing import NamedTuple

import jax
import numpy as np
from jax.core import Tracer

from ._meshcore import _raise_for_call, load_meshcore, meshcore_available


# (node_min, node_max, left, right, leaf, leaf_items, item_min, item_max):
# float64 bounds and int64 indices on the host.
type HostTree = tuple[np.ndarray, ...]

# Native PHX_MC_WINDING_* routes.
_WINDING_EXACT = 0
_WINDING_FAST_DIPOLE = 1


class _TreeView(ctypes.Structure):
    _fields_ = (
        ("node_count", ctypes.c_int64),
        ("leaf_count", ctypes.c_int64),
        ("leaf_width", ctypes.c_int64),
        ("item_count", ctypes.c_int64),
        ("node_min", ctypes.c_void_p),
        ("node_max", ctypes.c_void_p),
        ("node_size", ctypes.c_void_p),
        ("left", ctypes.c_void_p),
        ("right", ctypes.c_void_p),
        ("leaf", ctypes.c_void_p),
        ("leaf_items", ctypes.c_void_p),
        ("item_min", ctypes.c_void_p),
        ("item_max", ctypes.c_void_p),
    )


def _pointer(array: np.ndarray, /) -> int:
    return array.ctypes.data


def tree_view(tree: HostTree, /) -> tuple[_TreeView, tuple[np.ndarray, ...]]:
    """The native view of a host tree and the contiguous arrays it points into.

    The caller keeps the returned arrays alive while the view is in use.
    """
    if len(tree) != 8:
        raise ValueError("host trees require eight arrays.")
    for index, value in enumerate(tree):
        dtype = np.float64 if index in (0, 1, 6, 7) else np.int64
        if not isinstance(value, np.ndarray) or value.dtype != dtype:
            raise ValueError("host tree bounds must be float64 and indices int64.")
    node_min, node_max, left, right, leaf, leaf_items, item_min, item_max = tree
    if (
        node_min.ndim != 2
        or node_min.shape[0] < 1
        or node_min.shape[1] < 1
        or node_max.shape != node_min.shape
    ):
        raise ValueError("node bounds must share one nonempty (nodes, dimension) shape.")
    node_count, dimension = node_min.shape
    if any(value.shape != (node_count,) for value in (left, right, leaf)):
        raise ValueError("node metadata must have one entry per node.")
    if leaf_items.ndim != 2 or leaf_items.shape[1] < 1:
        raise ValueError("leaf items must have shape (leaves, positive leaf width).")
    if (
        item_min.ndim != 2
        or item_min.shape[1] != dimension
        or item_max.shape != item_min.shape
    ):
        raise ValueError("item bounds must share the node dimension and one shape.")
    if np.any(leaf >= leaf_items.shape[0]):
        raise ValueError("node leaf indices exceed the leaf array.")
    internal = leaf < 0
    if any(
        np.any((value[internal] < 0) | (value[internal] >= node_count))
        for value in (left, right)
    ):
        raise ValueError("child indices exceed the node arrays.")
    if np.any(leaf_items >= item_min.shape[0]):
        raise ValueError("leaf item indices exceed the item bounds.")
    node_min, node_max, left, right, leaf, leaf_items, item_min, item_max = (
        np.ascontiguousarray(value) for value in tree
    )
    # The descent key is the reference's own float64 extent sum.
    node_size = np.ascontiguousarray(np.sum(node_max - node_min, axis=-1))
    arrays = (
        node_min,
        node_max,
        node_size,
        left,
        right,
        leaf,
        leaf_items,
        item_min,
        item_max,
    )
    view = _TreeView(
        int(node_min.shape[0]),
        int(leaf_items.shape[0]),
        int(leaf_items.shape[1]),
        int(item_min.shape[0]),
        *(_pointer(array) for array in arrays),
    )
    return view, arrays


def native_point_queries(*values: object) -> bool:
    """Whether native point queries apply: the library is present and no leaf is traced."""
    if not meshcore_available():
        return False
    return not any(isinstance(leaf, Tracer) for leaf in jax.tree.leaves(values))


def _points(points: np.ndarray, /) -> np.ndarray:
    values = np.ascontiguousarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape (queries, 3).")
    return values


def _simplices(
    vertices: np.ndarray, simplices: np.ndarray, corners: int, /
) -> tuple[np.ndarray, np.ndarray]:
    vertex_values = np.ascontiguousarray(vertices, dtype=np.float64)
    simplex_values = np.ascontiguousarray(simplices, dtype=np.int32)
    if vertex_values.ndim != 2 or vertex_values.shape[1] != 3:
        raise ValueError("vertices must have shape (vertices, 3).")
    if simplex_values.ndim != 2 or simplex_values.shape[1] != corners:
        raise ValueError(f"simplices must have shape (items, {corners}).")
    return vertex_values, simplex_values


class TriangleNearest(NamedTuple):
    """The k nearest faces per point ordered by (squared distance, face index).

    ``faces`` is ``-1``, ``distance_squared`` infinite, ``closest`` NaN and
    ``features`` ``-1`` where fewer than ``k`` faces have finite distances.
    Features: 0 face, 1/2/3 edges (a,b) (b,c) (c,a), 4/5/6 corners a/b/c.
    """

    faces: np.ndarray
    distance_squared: np.ndarray
    closest: np.ndarray
    features: np.ndarray


def triangle_nearest(
    tree: HostTree,
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    /,
    *,
    k: int,
) -> TriangleNearest:
    """The ``k`` nearest triangles of every point, ties to the lower face index."""
    view, arrays = tree_view(tree)
    vertex_values, face_values = _simplices(vertices, faces, 3)
    queries = _points(points)
    count = queries.shape[0]
    found = np.empty((count, k), dtype=np.int32)
    squared = np.empty((count, k), dtype=np.float64)
    closest = np.empty((count, k, 3), dtype=np.float64)
    features = np.empty((count, k), dtype=np.int8)
    status = load_meshcore()["phx_mc_bvh_triangle_nearest"](
        ctypes.addressof(view),
        vertex_values.shape[0],
        _pointer(vertex_values),
        _pointer(face_values),
        count,
        _pointer(queries),
        k,
        _pointer(found),
        _pointer(squared),
        _pointer(closest),
        _pointer(features),
    )
    del arrays
    _raise_for_call(status, "bvh_triangle_nearest")
    return TriangleNearest(found, squared, closest, features)


def triangle_signed_distance(
    tree: HostTree,
    vertices: np.ndarray,
    faces: np.ndarray,
    feature_normals: np.ndarray,
    points: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signed distance, nearest face, closest point)`` by feature pseudonormals."""
    view, arrays = tree_view(tree)
    vertex_values, face_values = _simplices(vertices, faces, 3)
    normals = np.ascontiguousarray(feature_normals, dtype=np.float64)
    if normals.shape != (face_values.shape[0], 7, 3):
        raise ValueError("feature normals must have shape (faces, 7, 3).")
    queries = _points(points)
    count = queries.shape[0]
    distance = np.empty((count,), dtype=np.float64)
    found = np.empty((count,), dtype=np.int32)
    closest = np.empty((count, 3), dtype=np.float64)
    status = load_meshcore()["phx_mc_bvh_triangle_signed_distance"](
        ctypes.addressof(view),
        vertex_values.shape[0],
        _pointer(vertex_values),
        _pointer(face_values),
        _pointer(normals),
        count,
        _pointer(queries),
        _pointer(distance),
        _pointer(found),
        _pointer(closest),
    )
    del arrays
    _raise_for_call(status, "bvh_triangle_signed_distance")
    return distance, found, closest


def _winding(
    tree: HostTree,
    vertices: np.ndarray,
    faces: np.ndarray,
    points: np.ndarray,
    route: int,
    offsets: np.ndarray | None,
    edges: np.ndarray | None,
    dipoles: np.ndarray | None,
    opening_angle: float,
    /,
) -> np.ndarray:
    view, arrays = tree_view(tree)
    vertex_values, face_values = _simplices(vertices, faces, 3)
    queries = _points(points)
    values = np.empty((queries.shape[0],), dtype=np.float64)
    status = load_meshcore()["phx_mc_bvh_triangle_winding"](
        ctypes.addressof(view),
        vertex_values.shape[0],
        _pointer(vertex_values),
        _pointer(face_values),
        route,
        None if offsets is None else _pointer(offsets),
        0 if edges is None else edges.shape[0],
        None if edges is None or edges.shape[0] == 0 else _pointer(edges),
        None if dipoles is None else _pointer(dipoles),
        opening_angle,
        queries.shape[0],
        _pointer(queries),
        _pointer(values),
    )
    del arrays
    _raise_for_call(status, "bvh_triangle_winding")
    return values


def triangle_winding_exact(
    tree: HostTree,
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_offsets: np.ndarray,
    boundary_edges: np.ndarray,
    points: np.ndarray,
    /,
) -> np.ndarray:
    """Exact generalized winding numbers by hierarchical closing fans."""
    offsets = np.ascontiguousarray(boundary_offsets, dtype=np.int64)
    edges = np.ascontiguousarray(boundary_edges, dtype=np.int32)
    if offsets.shape != (tree[0].shape[0] + 1,):
        raise ValueError("boundary offsets must have one entry per node plus one.")
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("boundary edges must have shape (edges, 2).")
    return _winding(
        tree, vertices, faces, points, _WINDING_EXACT, offsets, edges, None, 0.0
    )


def triangle_winding_dipole(
    tree: HostTree,
    vertices: np.ndarray,
    faces: np.ndarray,
    dipoles: np.ndarray,
    opening_angle: float,
    points: np.ndarray,
    /,
) -> np.ndarray:
    """Approximate winding numbers by per-node dipoles (Barill et al. 2018)."""
    table = np.ascontiguousarray(dipoles, dtype=np.float64)
    if table.shape != (tree[0].shape[0], 7):
        raise ValueError("dipoles must have shape (nodes, 7).")
    return _winding(
        tree,
        vertices,
        faces,
        points,
        _WINDING_FAST_DIPOLE,
        None,
        None,
        table,
        opening_angle,
    )


class TetrahedronLocation(NamedTuple):
    """Lowest accepting affine tetrahedron per point and every accepting cell.

    ``status`` holds ``CellLocationStatus`` codes (located, outside,
    degenerate cell, nonfinite); ``candidates`` lists the accepting cells
    ascending, ``-1`` padded, with their reference coordinates.
    """

    cells: np.ndarray
    reference: np.ndarray
    residual: np.ndarray
    condition: np.ndarray
    status: np.ndarray
    containing: np.ndarray
    candidates: np.ndarray
    candidate_reference: np.ndarray


def _locate_call(
    view: _TreeView,
    vertices: np.ndarray,
    cells: np.ndarray,
    mask: np.ndarray | None,
    tolerance: float,
    queries: np.ndarray,
    capacity: int,
    /,
) -> TetrahedronLocation:
    count = queries.shape[0]
    result = TetrahedronLocation(
        np.empty((count,), dtype=np.int32),
        np.empty((count, 3), dtype=np.float64),
        np.empty((count,), dtype=np.float64),
        np.empty((count,), dtype=np.float64),
        np.empty((count,), dtype=np.int32),
        np.empty((count,), dtype=np.int32),
        np.empty((count, capacity), dtype=np.int32),
        np.empty((count, capacity, 3), dtype=np.float64),
    )
    status = load_meshcore()["phx_mc_bvh_tetrahedron_locate"](
        ctypes.addressof(view),
        vertices.shape[0],
        _pointer(vertices),
        _pointer(cells),
        None if mask is None else _pointer(mask),
        tolerance,
        count,
        _pointer(queries),
        capacity,
        *(_pointer(array) for array in result),
    )
    _raise_for_call(status, "bvh_tetrahedron_locate")
    return result


def tetrahedron_locate(
    tree: HostTree,
    vertices: np.ndarray,
    cells: np.ndarray,
    points: np.ndarray,
    /,
    *,
    cell_mask: np.ndarray | None,
    tolerance: float,
) -> TetrahedronLocation:
    """Locate points in positively oriented affine tetrahedra, lowest cell on ties.

    Candidate lists are as wide as the most accepting cells of any point; the
    few points beyond the first pass's width are located again at full width.
    """
    view, arrays = tree_view(tree)
    vertex_values, cell_values = _simplices(vertices, cells, 4)
    mask = None
    if cell_mask is not None:
        mask = np.ascontiguousarray(cell_mask, dtype=np.uint8)
        if mask.shape != (cell_values.shape[0],):
            raise ValueError("cell_mask must have one entry per cell.")
    queries = _points(points)
    capacity = 4
    result = _locate_call(
        view, vertex_values, cell_values, mask, tolerance, queries, capacity
    )
    width = max(1, int(np.max(result.containing, initial=0)))
    if width > capacity:
        wide = np.flatnonzero(result.containing > capacity)
        again = _locate_call(
            view, vertex_values, cell_values, mask, tolerance, queries[wide], width
        )
        candidates = np.full((queries.shape[0], width), -1, dtype=np.int32)
        candidate_reference = np.zeros((queries.shape[0], width, 3), dtype=np.float64)
        candidates[:, :capacity] = result.candidates
        candidate_reference[:, :capacity] = result.candidate_reference
        candidates[wide] = again.candidates
        candidate_reference[wide] = again.candidate_reference
        result = result._replace(
            candidates=candidates, candidate_reference=candidate_reference
        )
    else:
        result = result._replace(
            candidates=result.candidates[:, :width],
            candidate_reference=result.candidate_reference[:, :width],
        )
    del arrays
    return result


__all__ = [
    "HostTree",
    "TetrahedronLocation",
    "TriangleNearest",
    "native_point_queries",
    "tetrahedron_locate",
    "tree_view",
    "triangle_nearest",
    "triangle_signed_distance",
    "triangle_winding_dipole",
    "triangle_winding_exact",
]
