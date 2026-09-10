#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ._strict import StrictModule
from ._trainable import NonTrainableState


class PackedBVH(StrictModule, NonTrainableState):
    """A packed binary AABB BVH with fixed-size leaf payloads."""

    bbox_min: Array
    bbox_max: Array
    left: Array
    right: Array
    leaf_id: Array
    leaf_items: Array
    leaf_node: Array
    leaf_bbox_min: Array
    leaf_bbox_max: Array
    leaf_size: int = eqx.field(static=True)
    max_depth: int = eqx.field(static=True)


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
        leaf_node=jnp.asarray(leaf_node_for_id_np, dtype=jnp.int32),
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


def refit_packed_bvh_bounds(
    item_bbox_min: ArrayLike,
    item_bbox_max: ArrayLike,
    /,
    *,
    left: Array,
    right: Array,
    leaf_id: Array,
    leaf_items: Array,
    leaf_node: Array,
) -> tuple[Array, Array, Array, Array]:
    """Refit fixed BVH topology to differentiable current item bounds."""
    item_min = jnp.asarray(item_bbox_min)
    item_max = jnp.asarray(item_bbox_max, dtype=item_min.dtype)
    if item_min.ndim != 2 or item_max.shape != item_min.shape:
        raise ValueError("item_bbox_min/max must share shape (n_items, dimension).")
    if leaf_items.ndim != 2 or leaf_node.shape != (leaf_items.shape[0],):
        raise ValueError("leaf_items and leaf_node shapes are inconsistent.")
    valid_item = leaf_items >= 0
    safe_item = jnp.where(valid_item, leaf_items, 0)
    gathered_min = item_min[safe_item]
    gathered_max = item_max[safe_item]
    infinity = jnp.asarray(jnp.inf, dtype=item_min.dtype)
    leaf_min = jnp.min(jnp.where(valid_item[..., None], gathered_min, infinity), axis=1)
    leaf_max = jnp.max(jnp.where(valid_item[..., None], gathered_max, -infinity), axis=1)
    node_count = int(left.shape[0])
    bbox_min = jnp.full((node_count, item_min.shape[1]), infinity, dtype=item_min.dtype)
    bbox_max = jnp.full((node_count, item_min.shape[1]), -infinity, dtype=item_min.dtype)
    bbox_min = bbox_min.at[leaf_node].set(leaf_min)
    bbox_max = bbox_max.at[leaf_node].set(leaf_max)

    def body(index, bounds):
        current_min, current_max = bounds
        node = node_count - 1 - index
        internal = leaf_id[node] < 0
        left_node = jnp.maximum(left[node], 0)
        right_node = jnp.maximum(right[node], 0)
        combined_min = jnp.minimum(current_min[left_node], current_min[right_node])
        combined_max = jnp.maximum(current_max[left_node], current_max[right_node])
        current_min = current_min.at[node].set(
            jnp.where(internal, combined_min, current_min[node])
        )
        current_max = current_max.at[node].set(
            jnp.where(internal, combined_max, current_max[node])
        )
        return current_min, current_max

    bbox_min, bbox_max = jax.lax.fori_loop(0, node_count, body, (bbox_min, bbox_max))
    return bbox_min, bbox_max, leaf_min, leaf_max


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


def _bounded_leaf_candidates(
    overlaps,
    bvh: PackedBVH,
    maximum_candidates: int,
    /,
) -> tuple[Array, Array, Array]:
    capacity = int(maximum_candidates)
    if capacity <= 0:
        raise ValueError("maximum_candidates must be positive.")
    node_count = int(bvh.left.shape[0])
    stack = jnp.full((node_count,), -1, dtype=jnp.int32).at[0].set(0)
    candidates = jnp.full((capacity,), -1, dtype=jnp.int32)

    def visit_leaf(leaf_items, state):
        values, count = state

        def append_item(slot, carry):
            current, current_count = carry
            item = leaf_items[slot]
            item_valid = item >= 0
            has_space = current_count < capacity
            target = jnp.minimum(current_count, capacity - 1)
            current = current.at[target].set(
                jnp.where(item_valid & has_space, item, current[target])
            )
            return current, current_count + item_valid.astype(jnp.int32)

        return jax.lax.fori_loop(
            0,
            bvh.leaf_size,
            append_item,
            (values, count),
        )

    def traverse(_, state):
        current_stack, top, values, count = state
        active = top > 0
        popped_top = jnp.maximum(top - 1, 0)
        node = current_stack[popped_top]
        safe_node = jnp.maximum(node, 0)
        hit = (
            active
            & (node >= 0)
            & overlaps(
                bvh.bbox_min[safe_node],
                bvh.bbox_max[safe_node],
            )
        )
        leaf = hit & (bvh.leaf_id[safe_node] >= 0)
        internal = hit & ~leaf
        leaf_index = jnp.maximum(bvh.leaf_id[safe_node], 0)
        values, count = jax.lax.cond(
            leaf,
            lambda carry: visit_leaf(bvh.leaf_items[leaf_index], carry),
            lambda carry: carry,
            (values, count),
        )
        left = jnp.maximum(bvh.left[safe_node], 0)
        right = jnp.maximum(bvh.right[safe_node], 0)
        current_stack = current_stack.at[popped_top].set(
            jnp.where(internal, left, current_stack[popped_top])
        )
        second_slot = jnp.minimum(popped_top + 1, node_count - 1)
        current_stack = current_stack.at[second_slot].set(
            jnp.where(internal, right, current_stack[second_slot])
        )
        next_top = jnp.where(internal, popped_top + 2, popped_top)
        return current_stack, next_top, values, count

    _, _, candidates, count = jax.lax.fori_loop(
        0,
        node_count,
        traverse,
        (
            stack,
            jnp.asarray(1, dtype=jnp.int32),
            candidates,
            jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    retained = jnp.minimum(count, capacity)
    valid = jnp.arange(capacity, dtype=jnp.int32) < retained
    return jnp.maximum(candidates, 0), valid, count <= capacity


def point_select_leaf_items(
    points: ArrayLike,
    /,
    *,
    bvh: PackedBVH,
    maximum_candidates: int,
    tolerance: ArrayLike = 0.0,
) -> tuple[Array, Array, Array]:
    """Return all bounded leaf candidates whose node boxes contain each point."""
    values = jnp.asarray(points, dtype=bvh.bbox_min.dtype)
    single = values.ndim == 1
    if single:
        values = values[None, :]
    if values.ndim != 2 or values.shape[-1] != bvh.bbox_min.shape[-1]:
        raise ValueError("points must have shape (queries, dimension) or (dimension,).")
    padding = jnp.asarray(tolerance, dtype=values.dtype)
    if padding.shape != ():
        raise ValueError("tolerance must be a scalar.")
    if isinstance(tolerance, (int, float)) and tolerance < 0.0:
        raise ValueError("tolerance must be non-negative.")

    def query(point):
        return _bounded_leaf_candidates(
            lambda lower, upper: jnp.all(
                (point >= lower - padding) & (point <= upper + padding)
            ),
            bvh,
            maximum_candidates,
        )

    candidates, valid, complete = jax.vmap(query)(values)
    if single:
        return candidates[0], valid[0], complete[0]
    return candidates, valid, complete


def ray_select_leaf_items(
    origins: ArrayLike,
    directions: ArrayLike,
    /,
    *,
    bvh: PackedBVH,
    maximum_candidates: int,
    minimum_parameter: ArrayLike = 0.0,
    maximum_parameter: ArrayLike = jnp.inf,
) -> tuple[Array, Array, Array]:
    """Return bounded leaf candidates intersected by each ray."""
    ray_origins = jnp.asarray(origins, dtype=bvh.bbox_min.dtype)
    ray_directions = jnp.asarray(directions, dtype=ray_origins.dtype)
    single = ray_origins.ndim == 1
    if single:
        ray_origins = ray_origins[None, :]
        ray_directions = ray_directions[None, :]
    if (
        ray_origins.ndim != 2
        or ray_origins.shape != ray_directions.shape
        or ray_origins.shape[-1] != bvh.bbox_min.shape[-1]
    ):
        raise ValueError(
            "origins and directions must share shape (rays, dimension) or (dimension,)."
        )
    parameter_shape = (ray_origins.shape[0],)
    lower_parameter = jnp.broadcast_to(
        jnp.asarray(minimum_parameter, dtype=ray_origins.dtype),
        parameter_shape,
    )
    upper_parameter = jnp.broadcast_to(
        jnp.asarray(maximum_parameter, dtype=ray_origins.dtype),
        parameter_shape,
    )

    def query(origin, direction, parameter_lower, parameter_upper):
        def overlaps(lower, upper):
            parallel = jnp.abs(direction) <= jnp.finfo(direction.dtype).tiny
            outside = parallel & ((origin < lower) | (origin > upper))
            inverse = jnp.where(parallel, 1.0, 1.0 / direction)
            first = (lower - origin) * inverse
            second = (upper - origin) * inverse
            axis_lower = jnp.where(parallel, -jnp.inf, jnp.minimum(first, second))
            axis_upper = jnp.where(parallel, jnp.inf, jnp.maximum(first, second))
            entry = jnp.maximum(jnp.max(axis_lower), parameter_lower)
            exit = jnp.minimum(jnp.min(axis_upper), parameter_upper)
            return ~jnp.any(outside) & (entry <= exit)

        return _bounded_leaf_candidates(overlaps, bvh, maximum_candidates)

    candidates, valid, complete = jax.vmap(query)(
        ray_origins,
        ray_directions,
        lower_parameter,
        upper_parameter,
    )
    if single:
        return candidates[0], valid[0], complete[0]
    return candidates, valid, complete


__all__ = [
    "PackedBVH",
    "aabb_dist2",
    "beam_select_leaf_items",
    "beam_select_nodes",
    "build_packed_bvh",
    "build_point_bvh",
    "point_select_leaf_items",
    "ray_select_leaf_items",
    "refit_packed_bvh_bounds",
]
