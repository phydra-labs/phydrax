#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


@dataclasses.dataclass(frozen=True)
class PackedBVH:
    """A packed binary AABB BVH with fixed-size leaf payloads."""

    bbox_min: Array  # (nNodes, dim)
    bbox_max: Array  # (nNodes, dim)
    left: Array  # (nNodes,) int32, -1 for leaf
    right: Array  # (nNodes,) int32, -1 for leaf
    leaf_id: Array  # (nNodes,) int32, >=0 for leaf else -1

    leaf_items: Array  # (nLeaves, leaf_size) int32, padded with -1
    leaf_bbox_min: Array  # (nLeaves, dim)
    leaf_bbox_max: Array  # (nLeaves, dim)

    leaf_size: int
    max_depth: int


def build_packed_bvh(
    item_bbox_min: np.ndarray,
    item_bbox_max: np.ndarray,
    centers: np.ndarray | None = None,
    /,
    *,
    leaf_size: int = 16,
    dtype: jnp.dtype = jnp.float32,
) -> PackedBVH:
    """Build a packed BVH over axis-aligned item bounding boxes (NumPy build)."""
    bmin = np.asarray(item_bbox_min, dtype=np.float64)
    bmax = np.asarray(item_bbox_max, dtype=np.float64)
    if bmin.ndim != 2 or bmax.ndim != 2:
        raise ValueError("item_bbox_min/max must be rank-2 arrays.")
    if bmin.shape != bmax.shape:
        raise ValueError("item_bbox_min/max must have matching shapes.")
    if bmin.shape[0] == 0:
        raise ValueError("cannot build BVH over empty item set.")
    if leaf_size <= 0:
        raise ValueError(f"leaf_size must be positive, got {leaf_size}.")

    if centers is None:
        ctr = 0.5 * (bmin + bmax)
    else:
        ctr = np.asarray(centers, dtype=np.float64)
        if ctr.shape != bmin.shape:
            raise ValueError("centers must have shape (nItems, dim).")

    bbox_min_nodes: list[np.ndarray] = []
    bbox_max_nodes: list[np.ndarray] = []
    left_nodes: list[int] = []
    right_nodes: list[int] = []
    leaf_id_nodes: list[int] = []
    leaf_items_list: list[np.ndarray] = []
    leaf_node_for_id: list[int] = []

    def _build(indices: np.ndarray, depth: int) -> tuple[int, int]:
        node = len(left_nodes)
        left_nodes.append(-1)
        right_nodes.append(-1)
        leaf_id_nodes.append(-1)

        bmin_n = bmin[indices].min(axis=0)
        bmax_n = bmax[indices].max(axis=0)
        bbox_min_nodes.append(bmin_n)
        bbox_max_nodes.append(bmax_n)

        max_depth = depth
        if indices.size <= leaf_size:
            lid = len(leaf_items_list)
            leaf = np.full((leaf_size,), -1, dtype=np.int32)
            leaf[: indices.size] = indices.astype(np.int32, copy=False)
            leaf_items_list.append(leaf)
            leaf_id_nodes[node] = lid
            leaf_node_for_id.append(node)
            return node, max_depth

        extent = bmax_n - bmin_n
        axis = int(np.argmax(extent))
        vals = ctr[indices, axis]
        mid = int(indices.size // 2)
        part = np.argpartition(vals, mid)
        left_idx = indices[part[:mid]]
        right_idx = indices[part[mid:]]
        if left_idx.size == 0 or right_idx.size == 0:
            left_idx = indices[:mid]
            right_idx = indices[mid:]

        lnode, ldepth = _build(left_idx, depth + 1)
        rnode, rdepth = _build(right_idx, depth + 1)
        left_nodes[node] = int(lnode)
        right_nodes[node] = int(rnode)
        max_depth = max(max_depth, ldepth, rdepth)
        return node, max_depth

    n_items = int(bmin.shape[0])
    root, max_depth = _build(np.arange(n_items, dtype=np.int32), 0)
    if root != 0:
        raise RuntimeError("BVH build invariant violated: root must be node 0.")

    bbox_min_np = np.stack(bbox_min_nodes, axis=0)
    bbox_max_np = np.stack(bbox_max_nodes, axis=0)
    left_np = np.asarray(left_nodes, dtype=np.int32)
    right_np = np.asarray(right_nodes, dtype=np.int32)
    leaf_id_np = np.asarray(leaf_id_nodes, dtype=np.int32)
    leaf_items_np = np.stack(leaf_items_list, axis=0).astype(np.int32, copy=False)

    leaf_node_for_id_np = np.asarray(leaf_node_for_id, dtype=np.int32)
    leaf_bbox_min_np = bbox_min_np[leaf_node_for_id_np]
    leaf_bbox_max_np = bbox_max_np[leaf_node_for_id_np]

    return PackedBVH(
        bbox_min=jnp.asarray(bbox_min_np, dtype=dtype),
        bbox_max=jnp.asarray(bbox_max_np, dtype=dtype),
        left=jnp.asarray(left_np, dtype=jnp.int32),
        right=jnp.asarray(right_np, dtype=jnp.int32),
        leaf_id=jnp.asarray(leaf_id_np, dtype=jnp.int32),
        leaf_items=jnp.asarray(leaf_items_np, dtype=jnp.int32),
        leaf_bbox_min=jnp.asarray(leaf_bbox_min_np, dtype=dtype),
        leaf_bbox_max=jnp.asarray(leaf_bbox_max_np, dtype=dtype),
        leaf_size=int(leaf_size),
        max_depth=int(max_depth),
    )


def build_point_bvh(
    points: ArrayLike,
    /,
    *,
    leaf_size: int = 32,
    dtype: jnp.dtype = jnp.float32,
) -> PackedBVH:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError("points must have shape (nPoints, dim).")
    return build_packed_bvh(
        pts,
        pts,
        pts,
        leaf_size=leaf_size,
        dtype=dtype,
    )


def aabb_dist2(p: Array, bmin: Array, bmax: Array, /) -> Array:
    z = jnp.asarray(0.0, dtype=p.dtype)
    d = jnp.maximum(z, jnp.maximum(bmin - p, p - bmax))
    return jnp.sum(d * d, axis=-1)


def beam_select_nodes(
    points: Array,
    /,
    *,
    bvh: PackedBVH,
    beam_width: int,
    steps: int,
) -> Array:
    """Beam traverse BVH by AABB lower bounds. Returns node ids with shape (N, B)."""
    if beam_width <= 0:
        raise ValueError(f"beam_width must be positive, got {beam_width}.")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")

    pts = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    if pts.ndim == 1:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError("points must have shape (N, dim) or (dim,).")

    B = int(beam_width)
    inf = jnp.asarray(jnp.inf, dtype=pts.dtype)

    bbox_min = bvh.bbox_min
    bbox_max = bvh.bbox_max
    left = bvh.left
    right = bvh.right
    leaf_id = bvh.leaf_id

    nodes = jnp.full((pts.shape[0], B), jnp.int32(-1))
    nodes = nodes.at[:, 0].set(jnp.int32(0))

    def _step(_, nodes):
        valid = nodes >= 0
        safe = jnp.where(valid, nodes, jnp.int32(0))
        is_leaf = valid & (leaf_id[safe] >= 0)

        lch = jnp.where(valid, left[safe], jnp.int32(-1))
        rch = jnp.where(valid, right[safe], jnp.int32(-1))

        cand0 = jnp.where(is_leaf, safe, lch)
        cand1 = jnp.where(is_leaf, jnp.int32(-1), rch)
        cand_nodes = jnp.concatenate([cand0, cand1], axis=1)  # (N, 2B)

        cand_safe = jnp.where(cand_nodes >= 0, cand_nodes, jnp.int32(0))
        bmin = bbox_min[cand_safe]
        bmax = bbox_max[cand_safe]
        d2 = aabb_dist2(pts[:, None, :], bmin, bmax)
        d2 = jnp.where(cand_nodes >= 0, d2, inf)

        _, idx = jax.lax.top_k(-d2, B)
        return jnp.take_along_axis(cand_nodes, idx, axis=1)

    nodes = jax.lax.fori_loop(0, int(steps), _step, nodes)
    return nodes


def beam_select_leaf_items(
    points: Array,
    /,
    *,
    bvh: PackedBVH,
    beam_width: int,
    steps: int,
) -> tuple[Array, Array]:
    """Return candidate leaf items (and validity mask) using beam BVH traversal."""
    pts = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    is_single = pts.ndim == 1
    if is_single:
        pts = pts.reshape((1, -1))
    if pts.ndim != 2:
        raise ValueError("points must have shape (N, dim) or (dim,).")

    nodes = beam_select_nodes(pts, bvh=bvh, beam_width=beam_width, steps=steps)

    safe_nodes = jnp.where(nodes >= 0, nodes, jnp.int32(0))
    lids = bvh.leaf_id[safe_nodes]
    valid_leaf = (nodes >= 0) & (lids >= 0)
    safe_lids = jnp.where(valid_leaf, lids, jnp.int32(0))

    items = bvh.leaf_items[safe_lids]  # (N, B, leaf_size)
    items = jnp.where(valid_leaf[..., None], items, jnp.int32(-1))
    items = items.reshape((pts.shape[0], int(beam_width * bvh.leaf_size)))

    valid = items >= 0
    safe_items = jnp.where(valid, items, jnp.int32(0))
    if is_single:
        return safe_items.reshape((-1,)), valid.reshape((-1,))
    return safe_items, valid


__all__ = [
    "PackedBVH",
    "aabb_dist2",
    "beam_select_leaf_items",
    "beam_select_nodes",
    "build_packed_bvh",
    "build_point_bvh",
]
