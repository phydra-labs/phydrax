#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import uuid4

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from ...discretization._cell_mesh import CellMesh
from ...discretization._support import DiscreteSupport
from .._atlas import BoundaryAtlas, BoundaryMap
from .._cubature import AbstractCubatureMap, CubatureAtlas
from ._topology import TriangleTopology


if TYPE_CHECKING:
    from ._bvh import TriangleBVH


class MeshQueryResult(StrictModule):
    """Closest-feature result preserving every leading point dimension."""

    closest_point: Array
    distance: Array
    face_index: Array
    normal: Array

    def __init__(
        self,
        *,
        closest_point: Array,
        distance: Array,
        face_index: Array,
        normal: Array,
    ):
        self.closest_point = jnp.asarray(closest_point, dtype=float)
        self.distance = jnp.asarray(distance, dtype=float)
        self.face_index = jnp.asarray(face_index, dtype=jnp.int32)
        self.normal = jnp.asarray(normal, dtype=float)


class TriangleMesh(StrictModule):
    """Immutable triangular surface embedding with validated connectivity."""

    vertices: Array
    faces: Array
    topology: TriangleTopology
    source_id: str = eqx.field(static=True)

    def __init__(self, vertices: Array, faces: Array, *, source_id: str | None = None):
        vertices_host = np.asarray(vertices, dtype=float)
        faces_host = np.asarray(faces, dtype=np.int32)
        if vertices_host.ndim != 2 or vertices_host.shape[1] != 3:
            raise ValueError("vertices must have shape (num_vertices, 3).")
        if faces_host.ndim != 2 or faces_host.shape[1] != 3:
            raise ValueError("faces must have shape (num_faces, 3).")
        if vertices_host.shape[0] == 0:
            raise ValueError("TriangleMesh requires at least one vertex.")
        if faces_host.shape[0] == 0:
            raise ValueError("TriangleMesh requires at least one face.")
        if not np.all(np.isfinite(vertices_host)):
            raise ValueError("vertices must contain only finite values.")
        if np.any(faces_host < 0) or np.any(faces_host >= vertices_host.shape[0]):
            raise ValueError("faces contain out-of-range vertex indices.")
        if np.any(
            (faces_host[:, 0] == faces_host[:, 1])
            | (faces_host[:, 1] == faces_host[:, 2])
            | (faces_host[:, 2] == faces_host[:, 0])
        ):
            raise ValueError("Every triangle must reference three distinct vertices.")
        triangles = vertices_host[faces_host]
        doubled_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=-1,
        )
        edge_scale_squared = np.max(
            np.sum(
                np.stack(
                    (
                        triangles[:, 1] - triangles[:, 0],
                        triangles[:, 2] - triangles[:, 1],
                        triangles[:, 0] - triangles[:, 2],
                    ),
                    axis=1,
                )
                ** 2,
                axis=-1,
            ),
            axis=1,
        )
        tolerance = np.finfo(float).eps * edge_scale_squared * 64.0
        if np.any(doubled_area <= tolerance):
            raise ValueError("TriangleMesh contains a degenerate face.")
        if source_id is not None and not source_id:
            raise ValueError("source_id must be non-empty.")
        self.vertices = jnp.asarray(vertices_host, dtype=float)
        self.faces = jnp.asarray(faces_host, dtype=jnp.int32)
        self.topology = TriangleTopology(faces_host, num_vertices=vertices_host.shape[0])
        self.source_id = source_id or f"triangle-mesh-{uuid4().hex}"

    @property
    def triangles(self) -> Array:
        return self.vertices[self.faces]

    def discrete_support(self, /) -> DiscreteSupport:
        """Return canonical topology bound to this immutable embedding."""
        return DiscreteSupport(
            self.topology.cell_complex_topology(),
            3,
            self.source_id,
        )

    def as_cell_mesh(self, /) -> CellMesh:
        """Return the shared computational realization of this surface mesh."""
        return CellMesh.from_triangles(
            self.vertices,
            self.faces,
            vertex_global_ids=jnp.arange(self.vertices.shape[0], dtype=jnp.int64),
            cell_global_ids=jnp.arange(self.faces.shape[0], dtype=jnp.int64),
        )

    @property
    def face_normals(self) -> Array:
        triangles = self.triangles
        normal = jnp.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        return normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)

    @property
    def face_areas(self) -> Array:
        triangles = self.triangles
        return 0.5 * jnp.linalg.norm(
            jnp.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=-1,
        )

    @property
    def measure(self) -> Array:
        return jnp.sum(self.face_areas)

    @property
    def boundary_atlas(self) -> BoundaryAtlas:
        return BoundaryAtlas(
            _TriangleSurfaceMap(self.vertices, self.faces),
            source_entity_ids=jnp.arange(self.faces.shape[0], dtype=jnp.int32),
            source_id=self.source_id,
        )

    @property
    def cubature_atlas(self) -> CubatureAtlas:
        return CubatureAtlas(
            _TriangleCubatureMap(self.vertices, self.faces),
            source_entity_ids=jnp.arange(self.faces.shape[0], dtype=jnp.int32),
            source_id=self.source_id,
            physical_tags=tuple("face" for _ in range(self.faces.shape[0])),
        )

    def query_index(self) -> TriangleMeshQueryIndex:
        return TriangleMeshQueryIndex(self)


class _TriangleSurfaceMap(BoundaryMap):
    vertices: Array
    faces: Array

    def __init__(self, vertices: Array, faces: Array):
        self.vertices = vertices
        self.faces = faces

    @property
    def num_charts(self) -> int:
        return self.faces.shape[0]

    @property
    def reference_dimension(self) -> int:
        return 2

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        triangles = self.vertices[self.faces[chart_indices]]
        first = reference[..., :1]
        second = reference[..., 1:2]
        return (
            triangles[..., 0, :]
            + first * (triangles[..., 1, :] - triangles[..., 0, :])
            + (1.0 - first) * second * (triangles[..., 2, :] - triangles[..., 0, :])
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        triangles = self.vertices[self.faces[chart_indices]]
        doubled_area = jnp.linalg.norm(
            jnp.cross(
                triangles[..., 1, :] - triangles[..., 0, :],
                triangles[..., 2, :] - triangles[..., 0, :],
            ),
            axis=-1,
        )
        return doubled_area * (1.0 - reference[..., 0])


class _TriangleCubatureMap(AbstractCubatureMap):
    vertices: Array
    faces: Array

    def __init__(self, vertices: Array, faces: Array):
        self.vertices = jnp.asarray(vertices, dtype=float)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)

    @property
    def num_charts(self) -> int:
        return int(self.faces.shape[0])

    @property
    def reference_domain(self):
        return "triangle"

    @property
    def ambient_dimension(self) -> int:
        return 3

    def map(self, chart_indices: Array, reference: Array, /) -> Array:
        triangles = self.vertices[self.faces[chart_indices]]
        return (
            triangles[..., 0, :]
            + reference[..., :1] * (triangles[..., 1, :] - triangles[..., 0, :])
            + reference[..., 1:2] * (triangles[..., 2, :] - triangles[..., 0, :])
        )

    def jacobian(self, chart_indices: Array, reference: Array, /) -> Array:
        triangles = self.vertices[self.faces[chart_indices]]
        doubled_area = jnp.linalg.norm(
            jnp.cross(
                triangles[..., 1, :] - triangles[..., 0, :],
                triangles[..., 2, :] - triangles[..., 0, :],
            ),
            axis=-1,
        )
        return jnp.broadcast_to(doubled_area, reference.shape[:-1])

    def reference_mask(self, chart_indices: Array, reference: Array, /) -> Array:
        del reference
        return jnp.ones(jnp.asarray(chart_indices).shape, dtype=bool)

    def evaluate(
        self,
        chart_indices: Array,
        reference: Array,
        /,
    ):
        return super().evaluate(chart_indices, reference)


def _closest_segment(point: Array, start: Array, end: Array) -> Array:
    direction = end - start
    parameter = jnp.sum((point - start) * direction, axis=-1) / jnp.sum(
        direction * direction,
        axis=-1,
    )
    parameter = jnp.clip(parameter, 0.0, 1.0)
    return start + parameter[..., None] * direction


def _closest_points_on_triangles(point: Array, triangles: Array) -> Array:
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]
    ab = b - a
    ac = c - a
    normal = jnp.cross(ab, ac)
    normal_norm_sq = jnp.sum(normal * normal, axis=-1)
    projection = (
        point
        - (jnp.sum((point - a) * normal, axis=-1) / normal_norm_sq)[:, None] * normal
    )

    v0 = ab
    v1 = ac
    v2 = projection - a
    d00 = jnp.sum(v0 * v0, axis=-1)
    d01 = jnp.sum(v0 * v1, axis=-1)
    d11 = jnp.sum(v1 * v1, axis=-1)
    d20 = jnp.sum(v2 * v0, axis=-1)
    d21 = jnp.sum(v2 * v1, axis=-1)
    denominator = d00 * d11 - d01 * d01
    second = (d11 * d20 - d01 * d21) / denominator
    third = (d00 * d21 - d01 * d20) / denominator
    first = 1.0 - second - third
    inside = (first >= 0.0) & (second >= 0.0) & (third >= 0.0)

    on_ab = _closest_segment(point, a, b)
    on_bc = _closest_segment(point, b, c)
    on_ca = _closest_segment(point, c, a)
    candidates = jnp.stack((projection, on_ab, on_bc, on_ca), axis=1)
    distance_sq = jnp.sum((candidates - point) ** 2, axis=-1)
    distance_sq = distance_sq.at[:, 0].set(jnp.where(inside, distance_sq[:, 0], jnp.inf))
    selected = jnp.argmin(distance_sq, axis=1)
    return jnp.take_along_axis(candidates, selected[:, None, None], axis=1)[:, 0]


class TriangleMeshQueryIndex(StrictModule):
    """Canonical exact BVH query preparation for one immutable triangle mesh."""

    mesh: TriangleMesh
    bvh: TriangleBVH

    def __init__(self, mesh: TriangleMesh):
        from ._bvh import TriangleBVH

        if not isinstance(mesh, TriangleMesh):
            raise TypeError("TriangleMeshQueryIndex requires a TriangleMesh.")
        self.mesh = mesh
        self.bvh = TriangleBVH(mesh)

    def query(self, points: Array, /) -> MeshQueryResult:
        return self.bvh.query(points)


__all__ = [
    "MeshQueryResult",
    "TriangleMesh",
    "TriangleMeshQueryIndex",
]
