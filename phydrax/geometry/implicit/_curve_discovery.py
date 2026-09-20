#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ...discretization._tensor_support import PreparedTensorGrid
from .._certificate import FieldRegularity, SignReliability, ZeroSetAccuracy
from .._contracts import CompiledGeometry, GeometryKind
from ..design._schema import DesignState, ParameterSchema
from ..simplicial._regions import SegmentMesh
from ._discovery import _bisect_root
from ._policy import ImplicitSurfacePolicy
from ._projection import (
    _field_and_gradient,
    ImplicitPointProjectionEvidence,
    ImplicitPointProjectionPlan,
)


_DEFAULT_CURVE_POLICY = ImplicitSurfacePolicy()
_CELL_CORNERS = ((0, 0), (1, 0), (1, 1), (0, 1))
_CELL_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


class ImplicitCurveEvidence(StrictModule):
    projection: ImplicitPointProjectionEvidence
    minimum_edge_length: jax.Array
    minimum_orientation_margin: jax.Array
    intersection_free: jax.Array
    finite: jax.Array
    accepted: jax.Array
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class ImplicitCurveRealization(StrictModule):
    proposed_vertices: jax.Array
    vertices: jax.Array
    edges: jax.Array
    evidence: ImplicitCurveEvidence
    source_id: str = eqx.field(static=True)

    @property
    def accepted(self):
        return self.evidence.accepted

    @property
    def refresh_required(self):
        return ~self.evidence.accepted | self.evidence.projection.refresh_required

    def to_segment_mesh(self, /) -> SegmentMesh:
        if not bool(np.asarray(self.accepted)):
            raise ValueError("Only an accepted implicit curve can be materialized.")
        return SegmentMesh(
            np.asarray(self.vertices),
            np.asarray(self.edges),
            source_id=self.source_id,
        )


def _cross_2d(first, second):
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _segment_pair_intersects(first, second, tolerance):
    first_start, first_stop = first
    second_start, second_stop = second
    first_direction = first_stop - first_start
    second_direction = second_stop - second_start
    a = _cross_2d(first_direction, second_start - first_start)
    b = _cross_2d(first_direction, second_stop - first_start)
    c = _cross_2d(second_direction, first_start - second_start)
    d = _cross_2d(second_direction, first_stop - second_start)
    return (a * b < -(tolerance**2)) & (c * d < -(tolerance**2))


class ImplicitCurvePlan(StrictModule):
    """Fixed segment topology with differentiable normal-gauge vertex refresh."""

    projection: ImplicitPointProjectionPlan
    base_vertices: jax.Array
    edges: jax.Array
    base_tangents: jax.Array
    intersection_pairs: jax.Array
    schema: ParameterSchema = eqx.field(static=True)
    policy: ImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CompiledGeometry,
        projection: ImplicitPointProjectionPlan,
        vertices,
        edges,
        intersection_pairs,
        /,
        *,
        policy: ImplicitSurfacePolicy,
        source_id: str,
    ):
        vertices_ = np.asarray(vertices, dtype=np.float64)
        edges_ = np.asarray(edges, dtype=np.int32)
        pairs_ = np.asarray(intersection_pairs, dtype=np.int32).reshape((-1, 2))
        vectors = vertices_[edges_[:, 1]] - vertices_[edges_[:, 0]]
        lengths = np.linalg.norm(vectors, axis=-1)
        if np.any(lengths <= 0.0):
            raise ValueError("Implicit curve contains a degenerate segment.")
        topology_id = canonical_fingerprint(
            {
                "kind": "implicit-curve-topology",
                "source": source_id,
                "edges": array_tree_fingerprint(edges_),
            }
        )
        self.projection = projection
        self.base_vertices = jnp.asarray(vertices_)
        self.edges = jnp.asarray(edges_)
        self.base_tangents = jnp.asarray(vectors / lengths[:, None])
        self.intersection_pairs = jnp.asarray(pairs_)
        self.schema = geometry.schema
        self.policy = policy
        self.source_id = source_id
        self.topology_id = topology_id
        self.plan_id = canonical_fingerprint(
            {
                "kind": "implicit-curve-plan",
                "topology": topology_id,
                "projection": projection.plan_id,
                "policy": repr(policy),
            }
        )

    def realize(self, state: DesignState, /) -> ImplicitCurveRealization:
        if not isinstance(state, DesignState) or state.schema != self.schema:
            raise ValueError("Implicit curve state must use the discovery schema.")
        projection = self.projection.realize(state)
        proposed = projection.proposed_points
        segments = proposed[self.edges]
        vectors = segments[:, 1] - segments[:, 0]
        lengths = jnp.linalg.norm(vectors, axis=-1)
        tangents = vectors / jnp.maximum(
            lengths[:, None],
            jnp.finfo(vectors.dtype).tiny,
        )
        orientation_margin = jnp.min(jnp.sum(tangents * self.base_tangents, axis=-1))
        tolerance = jnp.asarray(
            self.policy.minimum_face_area,
            dtype=lengths.dtype,
        )
        if self.intersection_pairs.shape[0]:
            selected = segments[self.intersection_pairs]
            intersections = jax.vmap(
                lambda pair: _segment_pair_intersects(pair[0], pair[1], tolerance)
            )(selected)
            intersection_free = ~jnp.any(intersections)
        else:
            intersection_free = jnp.asarray(True)
        minimum_length = jnp.min(lengths)
        finite = jnp.all(jnp.isfinite(proposed)) & jnp.all(jnp.isfinite(lengths))
        accepted = (
            projection.accepted
            & finite
            & (minimum_length > tolerance)
            & (orientation_margin > 0.0)
            & intersection_free
        )
        safe = jnp.where(accepted, proposed, self.base_vertices)
        evidence = ImplicitCurveEvidence(
            projection,
            minimum_length,
            orientation_margin,
            intersection_free,
            finite,
            accepted,
            self.topology_id,
            self.plan_id,
        )
        return ImplicitCurveRealization(
            proposed,
            safe,
            self.edges,
            evidence,
            self.source_id,
        )


def _edge_key(cell_i: int, cell_j: int, first: int, second: int):
    first_offset = _CELL_CORNERS[first]
    second_offset = _CELL_CORNERS[second]
    first_index = (cell_i + first_offset[0], cell_j + first_offset[1])
    second_index = (cell_i + second_offset[0], cell_j + second_offset[1])
    return tuple(sorted((first_index, second_index)))


def discover_implicit_curve(
    geometry: CompiledGeometry,
    grid: PreparedTensorGrid,
    /,
    *,
    policy: ImplicitSurfacePolicy = _DEFAULT_CURVE_POLICY,
    source_id: str,
) -> ImplicitCurvePlan:
    """Discover an oriented closed planar zero contour and freeze its segment topology."""
    if not isinstance(geometry, CompiledGeometry):
        raise TypeError("geometry must be CompiledGeometry.")
    if not isinstance(grid, PreparedTensorGrid):
        raise TypeError("grid must be PreparedTensorGrid.")
    if not isinstance(policy, ImplicitSurfacePolicy):
        raise TypeError("policy must be ImplicitSurfacePolicy.")
    if not source_id:
        raise ValueError("source_id must be non-empty.")
    if geometry.ambient_dimension != 2 or geometry.kind is not GeometryKind.REGION:
        raise ValueError("Implicit curve discovery requires a two-dimensional region.")
    if len(grid.structured_axes) != 2:
        raise ValueError("Implicit curve discovery requires a two-dimensional grid.")
    if any(axis.periodic for axis in grid.structured_axes):
        raise ValueError("Implicit curve discovery requires nonperiodic axes.")
    if not bool(np.asarray(geometry.validity().accepted)):
        raise ValueError("Implicit curve discovery geometry must be valid.")
    certificate = geometry.field_certificate
    if certificate.sign_reliability is not SignReliability.RELIABLE:
        raise ValueError("Implicit curve discovery requires reliable field sign.")
    if (
        certificate.zero_set_accuracy is ZeroSetAccuracy.APPROXIMATE
        and not policy.allow_approximate_zero_set
    ):
        raise ValueError("Approximate zero sets require explicit policy approval.")
    if (
        certificate.regularity is FieldRegularity.NONSMOOTH
        and not policy.allow_nonsmooth_field
    ):
        raise ValueError("Nonsmooth fields require explicit selected-branch approval.")

    axes = tuple(
        np.asarray(axis.point_coordinates, dtype=np.float64)
        for axis in grid.structured_axes
    )
    lattice_count = axes[0].size * axes[1].size
    if lattice_count > policy.maximum_lattice_points:
        raise ValueError(
            "Implicit curve grid exceeds maximum_lattice_points: "
            f"required {lattice_count}, allowed {policy.maximum_lattice_points}."
        )
    first, second = np.meshgrid(*axes, indexing="ij")
    lattice_points = np.stack((first, second), axis=-1)
    values = np.asarray(
        geometry.boundary_field(jnp.asarray(lattice_points)),
        dtype=np.float64,
    )
    root_tolerance = float(policy.projection.root_tolerance)
    zero_tolerance = float(policy.lattice_zero_tolerance)
    if np.any(~np.isfinite(values)):
        raise ValueError("Implicit curve lattice values must be finite.")
    if np.any(np.abs(values) <= zero_tolerance):
        raise ValueError(
            "Implicit curve discovery refuses lattice vertices on the zero set."
        )

    vertices: list[np.ndarray] = []
    edge_vertices: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    segments: list[tuple[int, int]] = []
    for cell_i in range(axes[0].size - 1):
        for cell_j in range(axes[1].size - 1):
            corner_indices = tuple(
                (cell_i + offset[0], cell_j + offset[1]) for offset in _CELL_CORNERS
            )
            corner_values = np.asarray(
                [values[index] for index in corner_indices],
                dtype=np.float64,
            )
            crossings = tuple(
                edge
                for edge, (start, stop) in enumerate(_CELL_EDGES)
                if (corner_values[start] < 0.0) != (corner_values[stop] < 0.0)
            )
            if not crossings:
                continue
            if len(crossings) not in (2, 4):
                raise ValueError("Implicit curve cell has an invalid crossing count.")

            crossing_vertices: dict[int, int] = {}
            for edge in crossings:
                start, stop = _CELL_EDGES[edge]
                key = _edge_key(cell_i, cell_j, start, stop)
                vertex = edge_vertices.get(key)
                if vertex is None:
                    first_index, second_index = key
                    root = _bisect_root(
                        geometry,
                        lattice_points[first_index],
                        lattice_points[second_index],
                        values[first_index],
                        values[second_index],
                        root_tolerance,
                    )
                    vertex = len(vertices)
                    vertices.append(root)
                    if len(vertices) > policy.maximum_vertices:
                        raise ValueError(
                            "Implicit curve discovery exceeds maximum_vertices."
                        )
                    if len(edge_vertices) > policy.maximum_crossings:
                        raise ValueError(
                            "Implicit curve discovery exceeds maximum_crossings."
                        )
                    edge_vertices[key] = vertex
                crossing_vertices[edge] = vertex

            if len(crossings) == 2:
                pairings = ((crossings[0], crossings[1]),)
            else:
                center = np.asarray(
                    (
                        0.5 * (axes[0][cell_i] + axes[0][cell_i + 1]),
                        0.5 * (axes[1][cell_j] + axes[1][cell_j + 1]),
                    ),
                    dtype=np.float64,
                )
                center_value = float(
                    np.asarray(geometry.boundary_field(jnp.asarray(center)))
                )
                if abs(center_value) <= zero_tolerance:
                    raise ValueError(
                        "Implicit curve ambiguity decider landed on the zero set."
                    )
                if (center_value < 0.0) == (corner_values[0] < 0.0):
                    pairings = ((0, 3), (1, 2))
                else:
                    pairings = ((0, 1), (2, 3))

            for first_edge, second_edge in pairings:
                start = crossing_vertices[first_edge]
                stop = crossing_vertices[second_edge]
                midpoint = 0.5 * (vertices[start] + vertices[stop])
                tangent = vertices[stop] - vertices[start]
                right_normal = np.asarray((tangent[1], -tangent[0]))
                finite_gradient = np.asarray(
                    _field_and_gradient(
                        geometry.kernel,
                        geometry.state,
                        jnp.asarray(midpoint[None, :]),
                    )[1][0],
                    dtype=np.float64,
                )
                if np.dot(right_normal, finite_gradient) < 0.0:
                    start, stop = stop, start
                segments.append((start, stop))
                if len(segments) > policy.maximum_faces:
                    raise ValueError("Implicit curve discovery exceeds maximum_faces.")

    if not vertices or not segments:
        raise ValueError("Implicit curve discovery found no closed zero contour.")
    vertices_array = np.asarray(vertices, dtype=np.float64)
    segment_array = np.asarray(segments, dtype=np.int32)
    degree = np.bincount(segment_array.reshape((-1,)), minlength=len(vertices))
    if np.any(degree != 2):
        raise ValueError("Implicit curve discovery produced a nonmanifold open topology.")
    intersection_pairs = np.asarray(
        [
            (first_index, second_index)
            for first_index in range(segment_array.shape[0])
            for second_index in range(first_index + 1, segment_array.shape[0])
            if not np.intersect1d(
                segment_array[first_index],
                segment_array[second_index],
            ).size
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    if intersection_pairs.shape[0] > policy.maximum_intersection_pairs:
        raise ValueError("Implicit curve exceeds maximum_intersection_pairs.")
    base_segments = vertices_array[segment_array]
    if intersection_pairs.shape[0]:
        intersections = jax.vmap(
            lambda pair: _segment_pair_intersects(
                pair[0],
                pair[1],
                policy.minimum_face_area,
            )
        )(jnp.asarray(base_segments[intersection_pairs]))
        if bool(np.asarray(jnp.any(intersections))):
            raise ValueError("Implicit curve discovery produced a self-intersection.")
    minimum_spacing = min(float(np.min(np.diff(axis))) for axis in axes)
    projection = ImplicitPointProjectionPlan(
        geometry,
        vertices_array,
        policy.projection.trust_fraction * minimum_spacing,
        policy=policy.projection,
        source_id=source_id,
    )
    return ImplicitCurvePlan(
        geometry,
        projection,
        vertices_array,
        segment_array,
        intersection_pairs,
        policy=policy,
        source_id=source_id,
    )


__all__ = [
    "ImplicitCurveEvidence",
    "ImplicitCurvePlan",
    "ImplicitCurveRealization",
    "discover_implicit_curve",
]
