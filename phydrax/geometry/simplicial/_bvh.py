#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from dataclasses import fields
from enum import StrEnum
from typing import Any, assert_never, Literal, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._bvh import (
    bvh_hierarchical_sum,
    bvh_nearest_items,
    BVHBuildPolicy,
    BVHNearestResult,
    PackedBVH,
    prepare_bvh,
    reduce_packed_bvh_nodes,
    refit_packed_bvh_bounds,
)
from ..._fingerprint import tree_content_equal
from ..._strict import StrictModule
from ...typing import checked, Dim, Float64, Int32, parse, validate
from ._mesh import closest_points_on_triangles, MeshQueryResult, TriangleMesh


if TYPE_CHECKING:
    from ._regions import SegmentQueryResult


type SegmentQueryExecution = Literal["auto", "bvh", "dense"]


class _SegmentVertexDim(Dim):
    pass


class _SegmentDim(Dim):
    pass


def _closest_points_on_segments(points: Array, segments: Array) -> Array:
    direction = segments[..., 1, :] - segments[..., 0, :]
    coordinate = jnp.clip(
        jnp.sum((points - segments[..., 0, :]) * direction, axis=-1)
        / jnp.sum(direction * direction, axis=-1),
        0.0,
        1.0,
    )
    return segments[..., 0, :] + coordinate[..., None] * direction


def _segment_ray_parity(points: Array, vertices: Array, edges: Array) -> Array:
    """Horizontal-ray parity in fixed-width point and edge tiles."""
    leading = points.shape[:-1]
    flat = points.reshape((-1, 2))
    count = edges.shape[0]
    width = 64
    lanes = jnp.arange(width, dtype=jnp.int32)

    def query(point: Array) -> Array:
        def tile(block: Array, parity: Array) -> Array:
            index = block * width + lanes
            selected = vertices[edges[jnp.minimum(index, count - 1)]]
            start, end = selected[:, 0], selected[:, 1]
            crossing = (start[:, 1] > point[1]) != (end[:, 1] > point[1])
            intersection = (end[:, 0] - start[:, 0]) * (
                point[1] - start[:, 1]
            ) / jnp.where(end[:, 1] != start[:, 1], end[:, 1] - start[:, 1], 1.0) + start[
                :, 0
            ]
            hits = (index < count) & crossing & (point[0] < intersection)
            return parity ^ (jnp.sum(hits) % 2 == 1)

        return jax.lax.fori_loop(
            0, (count + width - 1) // width, tile, jnp.asarray(False, dtype=jnp.bool_)
        )

    return jax.lax.map(query, flat, batch_size=64).reshape(leading)


class SegmentBVH(StrictModule):
    """Exact planar segment hierarchy with fixed topology and refittable vertices.

    Host preparation canonicalizes finite ``(V, 2)`` vertices to float64 and
    nonempty ``(E, 2)`` topology to int32, rejecting zero-length segments.
    ``policy`` defaults to ``BVHBuildPolicy(leaf_size=8)``. ``execution="auto"``
    selects dense queries only when ``64 * query_count * E`` logical bytes fit
    ``dense_workspace_bytes`` (default 512 KiB); ``"bvh"`` always traverses the
    hierarchy, and ``"dense"`` raises ``ValueError`` above that admission budget.
    The budget excludes prepared arrays and required outputs and is not a
    process-RSS limit. Equal distances are ordered by original segment index.
    Selected distances and projections remain locally differentiable away from
    branch changes.
    """

    __strict_contract__ = True

    vertices: Float64[_SegmentVertexDim, Literal[2]]
    edges: Int32[_SegmentDim, Literal[2]]
    bvh: PackedBVH
    execution: SegmentQueryExecution = eqx.field(static=True)
    dense_workspace_bytes: int = eqx.field(static=True)

    @checked
    def __init__(
        self,
        vertices: ArrayLike,
        edges: ArrayLike,
        *,
        policy: BVHBuildPolicy = BVHBuildPolicy(leaf_size=8),
        execution: SegmentQueryExecution = "auto",
        dense_workspace_bytes: int = 512 * 1024,
    ) -> None:
        from ._topology import SegmentTopology

        execution_ = parse(execution, SegmentQueryExecution, "execution")
        if isinstance(dense_workspace_bytes, bool) or not isinstance(
            dense_workspace_bytes, int
        ):
            raise TypeError("dense_workspace_bytes must be an integer.")
        if dense_workspace_bytes < 1:
            raise ValueError("dense_workspace_bytes must be positive.")

        values = np.asarray(vertices, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 2 or values.shape[0] == 0:
            raise ValueError("vertices must have shape (num_vertices > 0, 2).")
        if not np.all(np.isfinite(values)):
            raise ValueError("vertices must contain only finite values.")
        topology = SegmentTopology(
            jnp.asarray(edges, dtype=jnp.int32), num_vertices=values.shape[0]
        )
        segments = values[np.asarray(topology.edges)]
        if np.any(np.sum((segments[:, 1] - segments[:, 0]) ** 2, axis=-1) == 0.0):
            raise ValueError("SegmentBVH contains a zero-length edge.")
        packed = prepare_bvh(
            np.min(segments, axis=1),
            np.max(segments, axis=1),
            policy=policy,
            dtype=jnp.float64,
        )
        self.vertices = jnp.asarray(values, dtype=jnp.float64)
        self.edges = topology.edges
        self.bvh = packed
        self.execution = execution_
        self.dense_workspace_bytes = dense_workspace_bytes

    @classmethod
    @checked
    def from_stored_fields(cls: type[SegmentBVH], value: SegmentBVH, /) -> SegmentBVH:
        """Restore a validated fixed hierarchy without changing refitted topology."""
        from ..._model._structure import restore_registered_fields
        from ...lifecycle._meshing_source_records import register_meshing_source_types

        register_meshing_source_types()
        if type(value) is not cls:
            raise TypeError(
                f"SegmentBVH.stored_fields must match its registered type; got {type(value).__name__}"
            )
        value.validate_restored()
        return restore_registered_fields(
            cls, {member.name: getattr(value, member.name) for member in fields(value)}
        )

    def validate_restored(self) -> None:
        """Validate segment values, query policy, and every retained hierarchy binding."""
        from ._topology import SegmentTopology

        validate(self)
        parse(self.execution, SegmentQueryExecution, "SegmentBVH.execution")
        if type(self.dense_workspace_bytes) is not int or self.dense_workspace_bytes < 1:
            raise ValueError(
                f"SegmentBVH.dense_workspace_bytes must be a positive integer; got {self.dense_workspace_bytes!r}"
            )
        vertices = np.asarray(self.vertices)
        if vertices.shape[0] == 0 or not np.all(np.isfinite(vertices)):
            raise ValueError(
                "SegmentBVH.vertices must contain finite nonempty planar coordinates; got empty or nonfinite vertices"
            )
        topology = SegmentTopology(self.edges, num_vertices=vertices.shape[0])
        segments = vertices[np.asarray(topology.edges)]
        if np.any(np.sum((segments[:, 1] - segments[:, 0]) ** 2, axis=-1) == 0.0):
            raise ValueError(
                "SegmentBVH.edges must define nonzero segment lengths; got a zero-length edge"
            )
        self.bvh.validate_restored()
        for name, expected in (
            ("item_bbox_min", np.min(segments, axis=1)),
            ("item_bbox_max", np.max(segments, axis=1)),
        ):
            if not tree_content_equal(np.asarray(getattr(self.bvh, name)), expected):
                raise ValueError(
                    f"SegmentBVH.bvh.{name} must bind the stored segment coordinates; got changed item bounds"
                )

    @property
    def segments(self) -> Array:
        return self.vertices[self.edges]

    def refit(self, vertices: ArrayLike, /) -> SegmentBVH:
        """Refit same-shaped vertices without changing topology, order, or policy.

        This JAX-safe update validates shape only. Callers own finiteness and
        nonzero segment lengths of the new embedding.
        """
        values = jnp.asarray(vertices, dtype=self.vertices.dtype)
        if values.shape != self.vertices.shape:
            raise ValueError(f"vertices must have shape {self.vertices.shape}.")
        segments = values[self.edges]
        packed = refit_packed_bvh_bounds(
            self.bvh, jnp.min(segments, axis=1), jnp.max(segments, axis=1)
        )
        return eqx.tree_at(lambda tree: (tree.vertices, tree.bvh), self, (values, packed))

    def _points(self, points: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        values = jnp.asarray(points, dtype=self.vertices.dtype)
        if values.ndim == 0 or values.shape[-1] != 2:
            raise ValueError("points must have trailing dimension 2.")
        return values.reshape((-1, 2)), values.shape[:-1]

    def _dense_items(self, points: Array, k: int) -> Array:
        closest = _closest_points_on_segments(points[:, None, :], self.segments)
        squared = jnp.sum((closest - points[:, None, :]) ** 2, axis=-1)
        first = jnp.argmin(squared, axis=-1).astype(jnp.int32)
        if k == 1:
            return first[:, None]
        if self.edges.shape[0] == 1:
            return jnp.stack((first, jnp.full_like(first, -1)), axis=-1)
        other = jnp.where(
            jnp.arange(self.edges.shape[0], dtype=jnp.int32) == first[:, None],
            jnp.inf,
            squared,
        )
        second = jnp.argmin(other, axis=-1).astype(jnp.int32)
        return jnp.stack((first, second), axis=-1)

    def _dense_admitted(self, query_count: int) -> bool:
        # Conservative logical admission covers coordinates, projections,
        # distances, and candidate buffers, not just the distance table.
        required = 64 * query_count * self.edges.shape[0]
        match self.execution:
            case "auto":
                return required <= self.dense_workspace_bytes
            case "bvh":
                return False
            case "dense":
                if required > self.dense_workspace_bytes:
                    raise ValueError(
                        f"Dense segment query needs {required} workspace bytes, "
                        f"exceeding dense_workspace_bytes={self.dense_workspace_bytes}; "
                        "select execution='bvh'."
                    )
                return True
            case _:
                assert_never(self.execution)

    def nearest_segments(self, points: ArrayLike, /, *, k: int = 1) -> BVHNearestResult:
        """Return exact nearest segment indices and squared distances.

        ``points`` has shape ``(..., 2)``; ``k`` is 1 or 2 and fixes the trailing
        result axis. ``BVHNearestResult.items`` and ``distance_squared`` have
        shape ``(..., k)``, with missing ranks padded by ``-1`` and ``inf``.
        Ties are ordered by original segment index.
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError("k must be an integer.")
        if k not in (1, 2):
            raise ValueError("k must be 1 or 2.")
        flat, leading = self._points(points)

        def distance(point: Array, items: Array) -> Array:
            selected = self.vertices[self.edges[items]]
            closest = _closest_points_on_segments(point, selected)
            return jnp.sum((closest - point) ** 2, axis=-1)

        if self._dense_admitted(flat.shape[0]):
            items = self._dense_items(flat, k)
        else:
            items = bvh_nearest_items(
                self.bvh, flat, k=k, item_distance_squared=distance
            ).items
        # Search chooses a discrete branch. Recompute only its selected distances
        # outside the while-loop so reverse mode follows the same local projection.
        items = jax.lax.stop_gradient(items)
        selected = self.vertices[self.edges[jnp.maximum(items, 0)]]
        closest = _closest_points_on_segments(flat[:, None, :], selected)
        squared = jnp.where(
            items >= 0, jnp.sum((closest - flat[:, None, :]) ** 2, axis=-1), jnp.inf
        )
        return BVHNearestResult(
            items=items.reshape((*leading, k)),
            distance_squared=squared.reshape((*leading, k)),
        )

    def query(self, points: ArrayLike, /) -> SegmentQueryResult:
        """Project ``(..., 2)`` points onto their nearest segments.

        The result preserves leading point axes and reports ``closest_point``,
        Euclidean ``distance``, original ``segment_index``, and unit ``normal``.
        The right-hand normal of direction ``(dx, dy)`` is ``(dy, -dx)``; no
        enclosing-region orientation is inferred.
        """
        from ._regions import SegmentQueryResult

        flat, leading = self._points(points)
        segment = self.nearest_segments(flat).items[:, 0]
        selected = self.vertices[self.edges[segment]]
        closest = _closest_points_on_segments(flat, selected)
        direction = selected[:, 1] - selected[:, 0]
        normal = jnp.stack((direction[:, 1], -direction[:, 0]), axis=-1)
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return SegmentQueryResult(
            closest_point=closest.reshape((*leading, 2)),
            distance=jnp.linalg.norm(closest - flat, axis=-1).reshape(leading),
            segment_index=segment.reshape(leading),
            normal=normal.reshape((*leading, 2)),
        )


# Closing-fan edges are evaluated in fixed-width chunks inside the traversal.
_FAN_CHUNK = 8


class WindingNumberRoute(StrEnum):
    """Evaluation route of generalized winding numbers."""

    EXACT = "exact"
    FAST_DIPOLE = "fast_dipole"


class WindingNumberResult(StrictModule):
    """Generalized winding numbers together with their evaluation route.

    `EXACT` is exact up to floating-point roundoff: it sums the solid angles of
    every triangle in leaves whose box contains the query and replaces every
    other subtree by the closing fan of its boundary (Jacobson et al. 2013).
    `FAST_DIPOLE` is the approximate first-order dipole expansion of Barill et al.
    (2018) with opening parameter `opening_angle` (beta).
    """

    values: Array
    route: WindingNumberRoute = eqx.field(static=True)
    opening_angle: float | None = eqx.field(static=True)

    @property
    def approximate(self) -> bool:
        return self.route is WindingNumberRoute.FAST_DIPOLE


def _solid_angles(point: Array, first: Array, second: Array, third: Array) -> Array:
    """Signed solid angles of triangles seen from `point` (Van Oosterom-Strackee)."""
    a = first - point
    b = second - point
    c = third - point
    length_a = jnp.linalg.norm(a, axis=-1)
    length_b = jnp.linalg.norm(b, axis=-1)
    length_c = jnp.linalg.norm(c, axis=-1)
    numerator = jnp.sum(a * jnp.cross(b, c), axis=-1)
    denominator = (
        length_a * length_b * length_c
        + jnp.sum(a * b, axis=-1) * length_c
        + jnp.sum(b * c, axis=-1) * length_a
        + jnp.sum(c * a, axis=-1) * length_b
    )
    return 2.0 * jnp.arctan2(numerator, denominator)


def _reduce_chains(
    node: np.ndarray, low: np.ndarray, high: np.ndarray, net: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cancel oppositely oriented copies of each edge within each node chain."""
    order = np.lexsort((high, low, node))
    node, low, high, net = node[order], low[order], high[order], net[order]
    first = np.ones(node.shape, dtype=np.bool_)
    first[1:] = (node[1:] != node[:-1]) | (low[1:] != low[:-1]) | (high[1:] != high[:-1])
    starts = np.flatnonzero(first)
    if starts.shape[0] == 0:
        return node, low, high, net
    total = np.add.reduceat(net, starts)
    keep = total != 0
    return node[starts][keep], low[starts][keep], high[starts][keep], total[keep]


def _closing_boundaries(
    faces: np.ndarray, bvh: PackedBVH, /
) -> tuple[np.ndarray, np.ndarray]:
    """Oriented boundary 1-chain of every node patch as CSR `(offsets, edges)`.

    A node's triangles minus the fan joining its boundary to any point of its box
    form a closed chain inside the box, so outside the box the fan reproduces the
    patch's winding number exactly.  Chains are merged bottom-up level by level.
    """
    left = np.asarray(bvh.left, dtype=np.int64)
    right = np.asarray(bvh.right, dtype=np.int64)
    leaf_items = np.asarray(bvh.leaf_items, dtype=np.int64)
    leaf_node = np.asarray(bvh.leaf_node, dtype=np.int64)
    node_count = left.shape[0]
    parent = np.full((node_count,), -1, dtype=np.int64)
    internal = np.flatnonzero(left >= 0)
    parent[left[internal]] = internal
    parent[right[internal]] = internal
    face_node = np.empty((faces.shape[0],), dtype=np.int64)
    valid = leaf_items >= 0
    face_node[leaf_items[valid]] = np.broadcast_to(leaf_node[:, None], leaf_items.shape)[
        valid
    ]
    origin = faces.reshape((-1,)).astype(np.int64)
    destination = faces[:, [1, 2, 0]].reshape((-1,)).astype(np.int64)
    leaf_chain = _reduce_chains(
        np.repeat(face_node, 3),
        np.minimum(origin, destination),
        np.maximum(origin, destination),
        np.where(origin < destination, 1, -1).astype(np.int64),
    )
    offsets = np.asarray(bvh.level_offsets, dtype=np.int64)
    leaf_level = np.searchsorted(offsets, leaf_chain[0], side="right") - 1
    collected = [leaf_chain]
    current = tuple(part[leaf_level == bvh.max_depth] for part in leaf_chain)
    for level in range(bvh.max_depth - 1, -1, -1):
        merged = _reduce_chains(parent[current[0]], *current[1:])
        collected.append(merged)
        current = tuple(
            np.concatenate((merged_part, leaf_part[leaf_level == level]))
            for merged_part, leaf_part in zip(merged, leaf_chain, strict=True)
        )
    node, low, high, net = (
        np.concatenate([chain[index] for chain in collected]) for index in range(4)
    )
    by_node = np.argsort(node, kind="stable")
    order = np.repeat(by_node, np.abs(net[by_node]))
    edges = np.stack(
        (
            np.where(net[order] > 0, low[order], high[order]),
            np.where(net[order] > 0, high[order], low[order]),
        ),
        axis=1,
    )
    counts = np.bincount(node[order], minlength=node_count)
    boundary_offsets = np.concatenate((np.zeros((1,), dtype=np.int64), np.cumsum(counts)))
    return boundary_offsets, edges


class TriangleBVH(StrictModule):
    """Exact stack-traversed AABB hierarchy over one fixed triangle topology.

    `refit` moves the hierarchy to new differentiable vertex positions while
    keeping its topology; every query remains exact after refitting.
    """

    vertices: Array
    faces: Array
    bvh: PackedBVH
    boundary_offsets: Array
    boundary_edges: Array

    @checked
    def __init__(
        self,
        mesh: TriangleMesh,
        *,
        policy: BVHBuildPolicy = BVHBuildPolicy(leaf_size=8),
    ) -> None:
        faces = np.asarray(mesh.faces)
        triangles = np.asarray(mesh.vertices)[faces]
        packed = prepare_bvh(
            np.min(triangles, axis=1),
            np.max(triangles, axis=1),
            policy=policy,
            dtype=mesh.vertices.dtype,
        )
        boundary_offsets, boundary_edges = _closing_boundaries(faces, packed)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.bvh = packed
        self.boundary_offsets = jnp.asarray(boundary_offsets, dtype=jnp.int32)
        self.boundary_edges = jnp.asarray(boundary_edges, dtype=jnp.int32)

    @property
    def triangles(self) -> Array:
        return self.vertices[self.faces]

    def refit(self, vertices: ArrayLike, /) -> TriangleBVH:
        """Return this topology refitted to differentiable current vertices."""
        values = jnp.asarray(vertices, dtype=self.vertices.dtype)
        if values.shape != self.vertices.shape:
            raise ValueError(f"vertices must have shape {self.vertices.shape}.")
        triangles = values[self.faces]
        packed = refit_packed_bvh_bounds(
            self.bvh, jnp.min(triangles, axis=1), jnp.max(triangles, axis=1)
        )
        return eqx.tree_at(lambda tree: (tree.vertices, tree.bvh), self, (values, packed))

    def _points(self, points: ArrayLike, /) -> tuple[Array, tuple[int, ...]]:
        values = jnp.asarray(points, dtype=self.vertices.dtype)
        if values.ndim == 0 or values.shape[-1] != 3:
            raise ValueError("points must have trailing dimension 3.")
        return values.reshape((-1, 3)), values.shape[:-1]

    def nearest_faces(self, points: ArrayLike, /, *, k: int = 1) -> BVHNearestResult:
        """Exact `k` nearest faces per point, ties ordered by face index."""
        flat, leading = self._points(points)

        def distance(point: Array, items: Array) -> Array:
            selected = self.vertices[self.faces[items]]
            closest = closest_points_on_triangles(point, selected)
            return jnp.sum((closest - point) ** 2, axis=-1)

        result = bvh_nearest_items(self.bvh, flat, k=k, item_distance_squared=distance)
        return BVHNearestResult(
            items=result.items.reshape((*leading, k)),
            distance_squared=result.distance_squared.reshape((*leading, k)),
        )

    def query(self, points: ArrayLike, /) -> MeshQueryResult:
        flat, leading = self._points(points)
        face = self.nearest_faces(flat).items[:, 0]
        triangle = self.vertices[self.faces[face]]
        closest = jax.vmap(
            lambda point, vertices: closest_points_on_triangles(point, vertices[None])[0]
        )(flat, triangle)
        normal = jnp.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return MeshQueryResult(
            closest_point=closest.reshape((*leading, 3)),
            distance=jnp.linalg.norm(closest - flat, axis=-1).reshape(leading),
            face_index=face.reshape(leading),
            normal=normal.reshape((*leading, 3)),
        )

    def _leaf_solid_angle(self, point: Array, leaf: Array) -> Array:
        items = self.bvh.leaf_items[leaf]
        triangles = self.triangles[jnp.maximum(items, 0)]
        angles = _solid_angles(point, triangles[:, 0], triangles[:, 1], triangles[:, 2])
        return jnp.sum(jnp.where(items >= 0, angles, 0.0))

    def winding_number(self, points: ArrayLike, /) -> WindingNumberResult:
        """Exact generalized winding numbers by hierarchical closing fans."""
        flat, leading = self._points(points)
        lanes = jnp.arange(_FAN_CHUNK, dtype=jnp.int32)
        edge_count = self.boundary_edges.shape[0]

        def inside(point: Any, node: Any) -> Any:
            return jnp.all(
                (point >= self.bvh.bbox_min[node]) & (point <= self.bvh.bbox_max[node])
            )

        def closing_fan(point: Any, node: Any) -> Any:
            if edge_count == 0:
                return jnp.zeros((), dtype=point.dtype)
            apex = 0.5 * (self.bvh.bbox_min[node] + self.bvh.bbox_max[node])
            start = self.boundary_offsets[node]
            stop = self.boundary_offsets[node + 1]

            def chunk(state: Any) -> Any:
                offset, total = state
                index = offset + lanes
                edges = self.boundary_edges[jnp.clip(index, 0, edge_count - 1)]
                angles = _solid_angles(
                    point,
                    apex,
                    self.vertices[edges[:, 0]],
                    self.vertices[edges[:, 1]],
                )
                return offset + _FAN_CHUNK, total + jnp.sum(
                    jnp.where(index < stop, angles, 0.0)
                )

            _, total = jax.lax.while_loop(
                lambda state: state[0] < stop,
                chunk,
                (start, jnp.zeros((), dtype=point.dtype)),
            )
            return total

        values = bvh_hierarchical_sum(
            self.bvh,
            flat,
            open_node=inside,
            far_value=closing_fan,
            leaf_value=self._leaf_solid_angle,
        )
        return WindingNumberResult(
            values=(values / (4.0 * jnp.pi)).reshape(leading),
            route=WindingNumberRoute.EXACT,
            opening_angle=None,
        )

    def fast_winding_number(
        self, points: ArrayLike, /, *, opening_angle: float = 2.0
    ) -> WindingNumberResult:
        """Approximate winding numbers by per-node dipoles (Barill et al. 2018).

        A node whose area-weighted center lies farther than `opening_angle`
        times its radius from the query contributes its dipole term; closer
        nodes are opened and leaves are summed exactly.
        """
        if isinstance(opening_angle, bool) or not isinstance(opening_angle, (int, float)):
            raise TypeError("opening_angle must be a real number.")
        if not np.isfinite(opening_angle) or opening_angle <= 0.0:
            raise ValueError("opening_angle must be finite and positive.")
        flat, leading = self._points(points)
        triangles = self.triangles
        area_vectors = 0.5 * jnp.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        areas = jnp.linalg.norm(area_vectors, axis=-1)
        moments = reduce_packed_bvh_nodes(
            self.bvh,
            jnp.concatenate(
                (
                    area_vectors,
                    areas[:, None],
                    areas[:, None] * jnp.mean(triangles, axis=1),
                ),
                axis=1,
            ),
            reduction="sum",
        )
        node_normal = moments[:, :3]
        node_center = moments[:, 4:] / moments[:, 3:4]
        reach = jnp.maximum(
            jnp.abs(self.bvh.bbox_max - node_center),
            jnp.abs(node_center - self.bvh.bbox_min),
        )
        node_radius = jnp.linalg.norm(reach, axis=-1)
        beta = jnp.asarray(opening_angle, dtype=flat.dtype)

        def near(point: Any, node: Any) -> Any:
            return jnp.linalg.norm(node_center[node] - point) <= beta * node_radius[node]

        def dipole(point: Any, node: Any) -> Any:
            offset = node_center[node] - point
            distance = jnp.linalg.norm(offset)
            return jnp.dot(node_normal[node], offset) / distance**3

        values = bvh_hierarchical_sum(
            self.bvh,
            flat,
            open_node=near,
            far_value=dipole,
            leaf_value=self._leaf_solid_angle,
        )
        return WindingNumberResult(
            values=(values / (4.0 * jnp.pi)).reshape(leading),
            route=WindingNumberRoute.FAST_DIPOLE,
            opening_angle=float(opening_angle),
        )


__all__ = [
    "SegmentBVH",
    "SegmentQueryExecution",
    "TriangleBVH",
    "WindingNumberResult",
    "WindingNumberRoute",
]
