#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections import Counter

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...geometry.simplicial import TriangleMesh


class CrackSurfaceGeometry3D(StrictModule, NonTrainableState):
    """Oriented manifold crack surface with explicit boundary-front topology."""

    vertices: Array
    triangles: Array
    triangle_normals: Array
    triangle_areas: Array
    front_edges: Array
    front_midpoints: Array
    front_tangents: Array
    front_lengths: Array
    surface_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: ArrayLike,
        triangles: ArrayLike,
        /,
        *,
        surface_id: str | None = None,
    ):
        points = np.asarray(vertices, dtype=np.float64)
        cells = np.asarray(triangles, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError("vertices must have shape (vertex_count >= 3, 3).")
        if cells.ndim != 2 or cells.shape[1] != 3 or cells.shape[0] == 0:
            raise ValueError("triangles must have shape (triangle_count, 3).")
        if (
            np.any(~np.isfinite(points))
            or np.any(cells < 0)
            or np.any(cells >= points.shape[0])
            or np.any(np.diff(np.sort(cells, axis=1), axis=1) == 0)
        ):
            raise ValueError("Crack surface vertices or triangle indices are invalid.")
        first = points[cells[:, 1]] - points[cells[:, 0]]
        second = points[cells[:, 2]] - points[cells[:, 0]]
        cross = np.cross(first, second)
        double_area = np.linalg.norm(cross, axis=1)
        scale = max(float(np.max(np.abs(points))), 1.0)
        tolerance = 256.0 * np.finfo(np.float64).eps * scale**2
        if np.any(double_area <= tolerance):
            raise ValueError("Crack surface contains a degenerate triangle.")
        normals = cross / double_area[:, None]
        areas = 0.5 * double_area

        oriented_edges: list[tuple[int, int]] = []
        undirected: list[tuple[int, int]] = []
        for triangle in cells:
            for start, stop in (
                (int(triangle[0]), int(triangle[1])),
                (int(triangle[1]), int(triangle[2])),
                (int(triangle[2]), int(triangle[0])),
            ):
                oriented_edges.append((start, stop))
                undirected.append((min(start, stop), max(start, stop)))
        counts = Counter(undirected)
        if any(count > 2 for count in counts.values()):
            raise ValueError("Crack surface contains a nonmanifold edge.")
        incidences: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for oriented, key in zip(oriented_edges, undirected, strict=True):
            incidences.setdefault(key, []).append(oriented)
        if any(
            len(edges) == 2 and edges[0] != (edges[1][1], edges[1][0])
            for edges in incidences.values()
        ):
            raise ValueError(
                "Adjacent crack triangles must traverse shared edges oppositely."
            )
        front = np.asarray(
            [
                oriented
                for oriented, key in zip(oriented_edges, undirected, strict=True)
                if counts[key] == 1
            ],
            dtype=np.int32,
        )
        if front.size == 0:
            raise ValueError("A sharp crack surface must expose a boundary front.")
        front = front.reshape((-1, 2))
        front_vectors = points[front[:, 1]] - points[front[:, 0]]
        front_lengths = np.linalg.norm(front_vectors, axis=1)
        if np.any(front_lengths <= tolerance):
            raise ValueError("Crack front contains a degenerate edge.")
        resolved_id = (
            canonical_fingerprint(
                {
                    "kind": "crack-surface-geometry-3d",
                    "vertices": array_tree_fingerprint(points),
                    "triangles": array_tree_fingerprint(cells),
                }
            )
            if surface_id is None
            else str(surface_id)
        )
        if not resolved_id:
            raise ValueError("surface_id must be non-empty.")
        self.vertices = jnp.asarray(points)
        self.triangles = jnp.asarray(cells)
        self.triangle_normals = jnp.asarray(normals)
        self.triangle_areas = jnp.asarray(areas)
        self.front_edges = jnp.asarray(front)
        self.front_midpoints = jnp.asarray(
            0.5 * (points[front[:, 0]] + points[front[:, 1]])
        )
        self.front_tangents = jnp.asarray(front_vectors / front_lengths[:, None])
        self.front_lengths = jnp.asarray(front_lengths)
        self.surface_id = resolved_id

    @property
    def surface_area(self) -> Array:
        return jnp.sum(self.triangle_areas)

    @property
    def front_length(self) -> Array:
        return jnp.sum(self.front_lengths)

    def as_triangle_mesh(self) -> TriangleMesh:
        return TriangleMesh(
            self.vertices,
            self.triangles,
            source_id=self.surface_id,
        )


__all__ = ["CrackSurfaceGeometry3D"]
