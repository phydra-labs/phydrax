#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence
from uuid import uuid4

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from .._atlas import AbstractBoundaryMap, BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    exact_signed_distance_certificate,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import (
    bounded_rejection_sample,
    complete_sampling_result,
    RejectionSamplingPlan,
)
from ..design._schema import (
    _ParameterCollector,
    ParameterBinding,
    ParameterId,
)
from ._mesh import (
    _closest_points_on_triangles,
    _TriangleSurfaceMap,
    MeshQueryResult,
    TriangleMesh,
)
from ._topology import SegmentTopology


class SegmentQueryResult(StrictModule):
    closest_point: Array
    distance: Array
    segment_index: Array
    normal: Array

    def __init__(self, *, closest_point, distance, segment_index, normal):
        self.closest_point = jnp.asarray(closest_point, dtype=float)
        self.distance = jnp.asarray(distance, dtype=float)
        self.segment_index = jnp.asarray(segment_index, dtype=jnp.int32)
        self.normal = jnp.asarray(normal, dtype=float)


class SegmentMesh(StrictModule):
    """Immutable planar segment embedding with canonical topology."""

    vertices: Array
    topology: SegmentTopology
    source_id: str = eqx.field(static=True)

    def __init__(self, vertices: Array, edges: Array, *, source_id: str | None = None):
        vertices_host = np.asarray(vertices, dtype=float)
        if (
            vertices_host.ndim != 2
            or vertices_host.shape[1] != 2
            or vertices_host.shape[0] == 0
        ):
            raise ValueError("vertices must have shape (num_vertices > 0, 2).")
        if not np.all(np.isfinite(vertices_host)):
            raise ValueError("vertices must contain only finite values.")
        topology = SegmentTopology(edges, num_vertices=vertices_host.shape[0])
        segments = vertices_host[np.asarray(topology.edges)]
        if np.any(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1) <= 0.0):
            raise ValueError("SegmentMesh contains a zero-length edge.")
        if source_id is not None and not source_id:
            raise ValueError("source_id must be non-empty.")
        self.vertices = jnp.asarray(vertices_host, dtype=float)
        self.topology = topology
        self.source_id = source_id or f"segment-mesh-{uuid4().hex}"

    @property
    def edges(self) -> Array:
        return self.topology.edges

    @property
    def segments(self) -> Array:
        return self.vertices[self.edges]

    @property
    def lengths(self) -> Array:
        return jnp.linalg.norm(self.segments[:, 1] - self.segments[:, 0], axis=-1)

    @property
    def measure(self) -> Array:
        return jnp.sum(self.lengths)

    def query(self, points: Array, /) -> SegmentQueryResult:
        points_ = jnp.asarray(points, dtype=self.vertices.dtype)
        if points_.ndim == 0 or points_.shape[-1] != 2:
            raise ValueError("points must have trailing dimension 2.")
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 2))
        segments = self.segments
        direction = segments[:, 1] - segments[:, 0]
        relative = flat[:, None, :] - segments[None, :, 0, :]
        coordinate = jnp.clip(
            jnp.sum(relative * direction, axis=-1)
            / jnp.sum(direction * direction, axis=-1),
            0.0,
            1.0,
        )
        closest_by_segment = segments[:, 0] + coordinate[..., None] * direction
        distance_sq = jnp.sum((flat[:, None, :] - closest_by_segment) ** 2, axis=-1)
        segment = jnp.argmin(distance_sq, axis=-1).astype(jnp.int32)
        closest = jnp.take_along_axis(closest_by_segment, segment[:, None, None], axis=1)[
            :, 0
        ]
        distance = jnp.sqrt(
            jnp.take_along_axis(distance_sq, segment[:, None], axis=1)[:, 0]
        )
        selected_direction = direction[segment]
        normal = jnp.stack((selected_direction[:, 1], -selected_direction[:, 0]), axis=-1)
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return SegmentQueryResult(
            closest_point=closest.reshape((*leading, 2)),
            distance=distance.reshape(leading),
            segment_index=segment.reshape(leading),
            normal=normal.reshape((*leading, 2)),
        )


class _LoopBoundaryMap(AbstractBoundaryMap):
    vertices: Array
    edges: Array

    def __init__(self, vertices: Array, edges: Array):
        self.vertices, self.edges = vertices, edges

    @property
    def num_charts(self):
        return self.edges.shape[0]

    @property
    def reference_dimension(self):
        return 1

    @property
    def ambient_dimension(self):
        return 2

    def map(self, chart_indices, reference, /):
        segment = self.vertices[self.edges[chart_indices]]
        return segment[..., 0, :] + reference * (segment[..., 1, :] - segment[..., 0, :])

    def jacobian(self, chart_indices, reference, /):
        del reference
        segment = self.vertices[self.edges[chart_indices]]
        return jnp.linalg.norm(segment[..., 1, :] - segment[..., 0, :], axis=-1)


class PlanarMeshRegion(GeometrySource):
    """Planar polygonal region with explicit outer and hole loops."""

    vertices: Array
    edges: Array
    loop_offsets: Array
    feature_id: str = eqx.field(static=True)

    def __init__(
        self,
        vertices: Array,
        loops: Sequence[Sequence[int]],
        *,
        feature_id: str | None = None,
    ):
        vertices_host = np.asarray(vertices, dtype=float)
        if vertices_host.ndim != 2 or vertices_host.shape[1] != 2:
            raise ValueError("vertices must have shape (num_vertices, 2).")
        loops_host = [np.asarray(loop, dtype=np.int32) for loop in loops]
        if not loops_host or any(loop.ndim != 1 or loop.size < 3 for loop in loops_host):
            raise ValueError(
                "loops must contain one or more index cycles of length at least three."
            )
        if any(
            np.any(loop < 0) or np.any(loop >= vertices_host.shape[0])
            for loop in loops_host
        ):
            raise ValueError("loop indices are out of range.")
        edges = np.concatenate(
            [np.stack((loop, np.roll(loop, -1)), axis=1) for loop in loops_host], axis=0
        )
        SegmentTopology(edges, num_vertices=vertices_host.shape[0])
        signed_areas = []
        for loop in loops_host:
            points = vertices_host[loop]
            signed_areas.append(
                0.5
                * np.sum(
                    points[:, 0] * np.roll(points[:, 1], -1)
                    - np.roll(points[:, 0], -1) * points[:, 1]
                )
            )
        if signed_areas[0] <= 0.0 or any(area >= 0.0 for area in signed_areas[1:]):
            raise ValueError(
                "The outer loop must be counter-clockwise and holes clockwise."
            )
        offsets = np.zeros((len(loops_host) + 1,), dtype=np.int32)
        offsets[1:] = np.cumsum([loop.size for loop in loops_host], dtype=np.int32)
        self.vertices = jnp.asarray(vertices_host, dtype=float)
        self.edges = jnp.asarray(edges, dtype=jnp.int32)
        self.loop_offsets = jnp.asarray(offsets, dtype=jnp.int32)
        self.feature_id = feature_id or f"planar-region-{uuid4().hex}"

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        vertices = context.bind(
            ParameterId(self.feature_id, "vertices"),
            self.vertices,
            role="position",
            physical_scale=float(np.max(np.ptp(np.asarray(self.vertices), axis=0))),
        )
        return _PlanarMeshRegionKernel(
            vertices,
            self.edges,
            self.loop_offsets,
            source_id=self.feature_id,
        )


class _PlanarMeshRegionKernel(GeometryKernel):
    vertices: ParameterBinding = eqx.field(static=True)
    edges: Array
    loop_offsets: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, vertices, edges, loop_offsets, *, source_id):
        self.vertices, self.edges, self.loop_offsets, self.source_id = (
            vertices,
            edges,
            loop_offsets,
            source_id,
        )

    @property
    def ambient_dimension(self):
        return 2

    @property
    def intrinsic_dimension(self):
        return 2

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return frozenset(
            {
                GeometryCapability.REGION_QUERY,
                GeometryCapability.SIGNED_DISTANCE,
                GeometryCapability.BOUNDARY_NORMAL,
                GeometryCapability.MEASURE,
                GeometryCapability.INTERIOR_SAMPLING,
                GeometryCapability.BOUNDARY_SAMPLING,
                GeometryCapability.BOUNDARY_ATLAS,
            }
        )

    @property
    def field_certificate(self):
        return exact_signed_distance_certificate(smooth=False)

    def _vertices(self, state):
        return self.vertices.read(state)

    def _query(self, state, points):
        vertices = self._vertices(state)
        points_ = jnp.asarray(points, dtype=vertices.dtype)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 2))
        segments = vertices[self.edges]
        direction = segments[:, 1] - segments[:, 0]
        relative = flat[:, None, :] - segments[None, :, 0, :]
        coordinate = jnp.clip(
            jnp.sum(relative * direction, axis=-1)
            / jnp.sum(direction * direction, axis=-1),
            0.0,
            1.0,
        )
        closest_by_segment = segments[:, 0] + coordinate[..., None] * direction
        distance_sq = jnp.sum((flat[:, None, :] - closest_by_segment) ** 2, axis=-1)
        segment = jnp.argmin(distance_sq, axis=-1).astype(jnp.int32)
        closest = jnp.take_along_axis(closest_by_segment, segment[:, None, None], axis=1)[
            :, 0
        ]
        distance = jnp.sqrt(
            jnp.take_along_axis(distance_sq, segment[:, None], axis=1)[:, 0]
        )
        selected_direction = direction[segment]
        normal = jnp.stack((selected_direction[:, 1], -selected_direction[:, 0]), axis=-1)
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return SegmentQueryResult(
            closest_point=closest.reshape((*leading, 2)),
            distance=distance.reshape(leading),
            segment_index=segment.reshape(leading),
            normal=normal.reshape((*leading, 2)),
        )

    def contains(self, state, points, /):
        vertices = self._vertices(state)
        points_ = jnp.asarray(points, dtype=vertices.dtype)
        start = vertices[self.edges[:, 0]]
        end = vertices[self.edges[:, 1]]
        x = points_[..., 0, None]
        y = points_[..., 1, None]
        crossing = (start[:, 1] > y) != (end[:, 1] > y)
        intersection = (end[:, 0] - start[:, 0]) * (y - start[:, 1]) / jnp.where(
            end[:, 1] != start[:, 1], end[:, 1] - start[:, 1], 1.0
        ) + start[:, 0]
        return jnp.sum(crossing & (x < intersection), axis=-1) % 2 == 1

    def boundary_field(self, state, points, /):
        query = self._query(state, points)
        points_ = jnp.asarray(points, dtype=query.closest_point.dtype)
        difference = points_ - query.closest_point
        squared_distance = jnp.sum(difference * difference, axis=-1)
        away_from_boundary = squared_distance > 0.0
        distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
        signed_distance = jnp.where(
            self.contains(state, points_),
            -distance,
            distance,
        )
        boundary_linearization = jnp.sum(difference * query.normal, axis=-1)
        return jnp.where(
            away_from_boundary,
            signed_distance,
            boundary_linearization,
        )

    def boundary_normal(self, state, points, /):
        return self._query(state, points).normal

    def bounds(self, state, /):
        vertices = self._vertices(state)
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state, /):
        vertices = self._vertices(state)
        start = vertices[self.edges[:, 0]]
        end = vertices[self.edges[:, 1]]
        return 0.5 * jnp.sum(start[:, 0] * end[:, 1] - end[:, 0] * start[:, 1])

    def boundary_measure(self, state, /):
        vertices = self._vertices(state)
        return jnp.sum(
            jnp.linalg.norm(
                vertices[self.edges[:, 1]] - vertices[self.edges[:, 0]], axis=-1
            )
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan
        return bounded_rejection_sample(
            lambda proposal_key, count: jr.uniform(
                proposal_key,
                (count, 2),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            ),
            lambda values: self.contains(state, values),
            num_points=num_points,
            point_dimension=2,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        vertices = self._vertices(state)
        segments = vertices[self.edges]
        length = jnp.linalg.norm(segments[:, 1] - segments[:, 0], axis=-1)
        edge_key, coordinate_key = jr.split(key)
        count = int(num_points)
        indices = jr.choice(
            edge_key, self.edges.shape[0], (count,), p=length / jnp.sum(length)
        )
        coordinate = jr.uniform(coordinate_key, (count, 1), dtype=vertices.dtype)
        return complete_sampling_result(
            segments[indices, 0]
            + coordinate * (segments[indices, 1] - segments[indices, 0])
        )

    def boundary_atlas(self, state, /):
        return BoundaryAtlas(
            _LoopBoundaryMap(self._vertices(state), self.edges),
            source_entity_ids=jnp.arange(self.edges.shape[0], dtype=jnp.int32),
            source_id=self.source_id,
        )


_MESH_CERTIFICATE = FieldCertificate(
    ZeroSetAccuracy.TOLERANCE_BOUND,
    SignReliability.RELIABLE,
    DistanceSemantics.EXACT,
    FieldRegularity.PIECEWISE_SMOOTH,
    1.0,
    "watertight_non_self_intersecting_mesh",
    True,
    ("simplicial", "generalized_winding"),
)


class MeshRegion(GeometrySource):
    """Watertight oriented triangular region with differentiable vertices."""

    vertices: Array
    faces: Array
    feature_id: str = eqx.field(static=True)

    def __init__(self, vertices: Array, faces: Array, *, feature_id: str | None = None):
        mesh = TriangleMesh(vertices, faces)
        if not mesh.topology.watertight:
            raise ValueError("MeshRegion requires a watertight triangle topology.")
        vertices_host = np.asarray(mesh.vertices)
        faces_host = np.asarray(mesh.faces)
        triangles = vertices_host[faces_host]
        signed_volume = (
            np.sum(
                np.einsum(
                    "ij,ij->i",
                    triangles[:, 0],
                    np.cross(triangles[:, 1], triangles[:, 2]),
                )
            )
            / 6.0
        )
        volume_tolerance = (
            np.finfo(float).eps * float(np.max(np.ptp(vertices_host, axis=0))) ** 3 * 64.0
        )
        if abs(signed_volume) <= volume_tolerance:
            raise ValueError("MeshRegion has zero signed volume.")
        if signed_volume < 0.0:
            faces_host = faces_host[:, [0, 2, 1]]
            mesh = TriangleMesh(vertices_host, faces_host)
        self.vertices = mesh.vertices
        self.faces = mesh.faces
        self.feature_id = feature_id or f"mesh-region-{uuid4().hex}"

    def _compile(self, context):
        vertices = context.bind(
            ParameterId(self.feature_id, "vertices"),
            self.vertices,
            role="position",
            physical_scale=float(np.max(np.ptp(np.asarray(self.vertices), axis=0))),
        )
        return _MeshRegionKernel(vertices, self.faces, source_id=self.feature_id)


class _MeshRegionKernel(GeometryKernel):
    vertices: ParameterBinding = eqx.field(static=True)
    faces: Array
    source_id: str = eqx.field(static=True)

    def __init__(self, vertices, faces, *, source_id):
        self.vertices, self.faces, self.source_id = vertices, faces, source_id

    @property
    def ambient_dimension(self):
        return 3

    @property
    def intrinsic_dimension(self):
        return 3

    @property
    def kind(self):
        return GeometryKind.REGION

    @property
    def capabilities(self):
        return frozenset(
            {
                GeometryCapability.REGION_QUERY,
                GeometryCapability.SIGNED_DISTANCE,
                GeometryCapability.BOUNDARY_NORMAL,
                GeometryCapability.MEASURE,
                GeometryCapability.INTERIOR_SAMPLING,
                GeometryCapability.BOUNDARY_SAMPLING,
                GeometryCapability.BOUNDARY_ATLAS,
            }
        )

    @property
    def field_certificate(self):
        return _MESH_CERTIFICATE

    def _vertices(self, state):
        return self.vertices.read(state)

    def _triangles(self, state):
        return self._vertices(state)[self.faces]

    def _query(self, state, points):
        points_ = jnp.asarray(points, dtype=float)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        triangles = self._triangles(state)
        closest_by_face = jax.vmap(_closest_points_on_triangles, in_axes=(0, None))(
            flat, triangles
        )
        distance_sq = jnp.sum((closest_by_face - flat[:, None, :]) ** 2, axis=-1)
        face = jnp.argmin(distance_sq, axis=-1).astype(jnp.int32)
        closest = jnp.take_along_axis(closest_by_face, face[:, None, None], axis=1)[:, 0]
        distance = jnp.sqrt(jnp.take_along_axis(distance_sq, face[:, None], axis=1)[:, 0])
        triangle = triangles[face]
        normal = jnp.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        return MeshQueryResult(
            closest_point=closest.reshape((*leading, 3)),
            distance=distance.reshape(leading),
            face_index=face.reshape(leading),
            normal=normal.reshape((*leading, 3)),
        )

    def contains(self, state, points, /):
        points_ = jnp.asarray(points, dtype=float)
        triangles = self._triangles(state)
        a = triangles[:, 0] - points_[..., None, :]
        b = triangles[:, 1] - points_[..., None, :]
        c = triangles[:, 2] - points_[..., None, :]
        numerator = jnp.sum(a * jnp.cross(b, c), axis=-1)
        denominator = (
            jnp.linalg.norm(a, axis=-1)
            * jnp.linalg.norm(b, axis=-1)
            * jnp.linalg.norm(c, axis=-1)
            + jnp.sum(a * b, axis=-1) * jnp.linalg.norm(c, axis=-1)
            + jnp.sum(b * c, axis=-1) * jnp.linalg.norm(a, axis=-1)
            + jnp.sum(c * a, axis=-1) * jnp.linalg.norm(b, axis=-1)
        )
        winding = jnp.sum(2.0 * jnp.arctan2(numerator, denominator), axis=-1) / (
            4.0 * jnp.pi
        )
        return jnp.abs(winding) > 0.5

    def boundary_field(self, state, points, /):
        query = self._query(state, points)
        points_ = jnp.asarray(points, dtype=query.closest_point.dtype)
        difference = points_ - query.closest_point
        squared_distance = jnp.sum(difference * difference, axis=-1)
        away_from_boundary = squared_distance > 0.0
        distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
        signed_distance = jnp.where(
            self.contains(state, points_),
            -distance,
            distance,
        )
        boundary_linearization = jnp.sum(difference * query.normal, axis=-1)
        return jnp.where(
            away_from_boundary,
            signed_distance,
            boundary_linearization,
        )

    def boundary_normal(self, state, points, /):
        return self._query(state, points).normal

    def bounds(self, state, /):
        vertices = self._vertices(state)
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state, /):
        triangles = self._triangles(state)
        return (
            jnp.sum(
                jnp.sum(
                    triangles[:, 0] * jnp.cross(triangles[:, 1], triangles[:, 2]), axis=-1
                )
            )
            / 6.0
        )

    def boundary_measure(self, state, /):
        triangles = self._triangles(state)
        return jnp.sum(
            0.5
            * jnp.linalg.norm(
                jnp.cross(
                    triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
                ),
                axis=-1,
            )
        )

    def sample_interior(self, state, num_points, /, *, key, plan=None):
        bounds = self.bounds(state)
        plan_ = RejectionSamplingPlan() if plan is None else plan
        return bounded_rejection_sample(
            lambda proposal_key, count: jr.uniform(
                proposal_key,
                (count, 3),
                minval=bounds[0],
                maxval=bounds[1],
                dtype=bounds.dtype,
            ),
            lambda values: self.contains(state, values),
            num_points=num_points,
            point_dimension=3,
            key=key,
            plan=plan_,
            dtype=bounds.dtype,
        )

    def sample_boundary(self, state, num_points, /, *, key):
        triangles = self._triangles(state)
        area = 0.5 * jnp.linalg.norm(
            jnp.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            ),
            axis=-1,
        )
        face_key, coordinate_key = jr.split(key)
        count = int(num_points)
        faces = jr.choice(face_key, triangles.shape[0], (count,), p=area / jnp.sum(area))
        coordinate = jr.uniform(coordinate_key, (count, 2), dtype=triangles.dtype)
        root = jnp.sqrt(coordinate[:, :1])
        barycentric = jnp.concatenate(
            (1.0 - root, root * (1.0 - coordinate[:, 1:]), root * coordinate[:, 1:]),
            axis=-1,
        )
        return complete_sampling_result(
            jnp.sum(barycentric[..., None] * triangles[faces], axis=-2)
        )

    def boundary_atlas(self, state, /):
        return BoundaryAtlas(
            _TriangleSurfaceMap(self._vertices(state), self.faces),
            source_entity_ids=jnp.arange(self.faces.shape[0], dtype=jnp.int32),
            source_id=self.source_id,
        )


class TriangleSurface(StrictModule):
    """Canonical codimension-one triangular surface realization."""

    mesh: TriangleMesh

    def __init__(self, vertices: Array, faces: Array, *, source_id: str | None = None):
        self.mesh = TriangleMesh(vertices, faces, source_id=source_id)

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 2

    @property
    def measure(self) -> Array:
        return self.mesh.measure

    @property
    def atlas(self) -> BoundaryAtlas:
        return self.mesh.boundary_atlas

    def query(self, points: Array, /) -> MeshQueryResult:
        return self.mesh.query_index().query(points)


__all__ = [
    "MeshRegion",
    "PlanarMeshRegion",
    "SegmentMesh",
    "SegmentQueryResult",
    "TriangleSurface",
]
