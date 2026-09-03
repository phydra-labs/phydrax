#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from phydrax.ein import contract

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ...linalg import (
    DenseCholesky,
    DenseLinearOperator,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
    solve,
)
from .._contracts import CompiledGeometry, GeometryKernel, GeometryTolerance
from ..design._schema import DesignState, ParameterSchema
from ..simplicial import TriangleMesh
from ._policy import ImplicitSurfacePolicy, ImplicitSurfaceStatus
from ._projection import (
    ImplicitPointProjectionEvidence,
    ImplicitPointProjectionPlan,
)


_QEF_POLICY = LinearSolvePolicy(
    DenseCholesky(),
    failure=FailurePolicy("status"),
)


def _cross2(first: Array, second: Array) -> Array:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _segment_intersection_2d(
    first: Array,
    second: Array,
    third: Array,
    fourth: Array,
    tolerance: float,
) -> Array:
    first_direction = second - first
    second_direction = fourth - third
    denominator = _cross2(first_direction, second_direction)
    difference = third - first
    usable = jnp.abs(denominator) > tolerance
    first_parameter = _cross2(difference, second_direction) / jnp.where(
        usable, denominator, 1.0
    )
    second_parameter = _cross2(difference, first_direction) / jnp.where(
        usable, denominator, 1.0
    )
    proper = (
        usable
        & (first_parameter >= -tolerance)
        & (first_parameter <= 1.0 + tolerance)
        & (second_parameter >= -tolerance)
        & (second_parameter <= 1.0 + tolerance)
    )
    collinear = (~usable) & (jnp.abs(_cross2(difference, first_direction)) <= tolerance)
    first_min = jnp.minimum(first, second)
    first_max = jnp.maximum(first, second)
    second_min = jnp.minimum(third, fourth)
    second_max = jnp.maximum(third, fourth)
    overlap = jnp.all(
        jnp.maximum(first_min, second_min)
        <= jnp.minimum(first_max, second_max) + tolerance
    )
    return proper | (collinear & overlap)


def _point_in_triangle_2d(point: Array, triangle: Array, tolerance: float) -> Array:
    values = jnp.stack(
        (
            _cross2(triangle[1] - triangle[0], point - triangle[0]),
            _cross2(triangle[2] - triangle[1], point - triangle[1]),
            _cross2(triangle[0] - triangle[2], point - triangle[2]),
        )
    )
    return jnp.all(values >= -tolerance) | jnp.all(values <= tolerance)


def _segment_triangle_intersection(
    first: Array,
    second: Array,
    triangle: Array,
    tolerance: float,
) -> Array:
    direction = second - first
    edge_first = triangle[1] - triangle[0]
    edge_second = triangle[2] - triangle[0]
    h = jnp.cross(direction, edge_second)
    determinant = jnp.dot(edge_first, h)
    usable = jnp.abs(determinant) > tolerance
    inverse = jnp.where(usable, 1.0 / determinant, 0.0)
    s = first - triangle[0]
    u = inverse * jnp.dot(s, h)
    q = jnp.cross(s, edge_first)
    v = inverse * jnp.dot(direction, q)
    t = inverse * jnp.dot(edge_second, q)
    return (
        usable
        & (u >= -tolerance)
        & (v >= -tolerance)
        & (u + v <= 1.0 + tolerance)
        & (t >= -tolerance)
        & (t <= 1.0 + tolerance)
    )


def _triangles_intersect(first: Array, second: Array, tolerance: float) -> Array:
    first_normal = jnp.cross(first[1] - first[0], first[2] - first[0])
    second_normal = jnp.cross(second[1] - second[0], second[2] - second[0])
    scale = jnp.maximum(
        jnp.sqrt(jnp.dot(first_normal, first_normal)),
        jnp.sqrt(jnp.dot(second_normal, second_normal)),
    )
    coplanar = (
        jnp.sqrt(jnp.sum(jnp.cross(first_normal, second_normal) ** 2))
        <= tolerance * jnp.maximum(scale * scale, 1.0)
    ) & (
        jnp.max(jnp.abs((second - first[0]) @ first_normal))
        <= tolerance * jnp.maximum(scale, 1.0)
    )
    edge_hits = jnp.asarray(False)
    for index in range(3):
        edge_hits = edge_hits | _segment_triangle_intersection(
            first[index],
            first[(index + 1) % 3],
            second,
            tolerance,
        )
        edge_hits = edge_hits | _segment_triangle_intersection(
            second[index],
            second[(index + 1) % 3],
            first,
            tolerance,
        )
    axis = jnp.argmax(jnp.abs(first_normal))
    project = (
        lambda value: value[:, 1:],
        lambda value: value[:, jnp.asarray((0, 2))],
        lambda value: value[:, :2],
    )
    first_2d = jax.lax.switch(axis, project, first)
    second_2d = jax.lax.switch(axis, project, second)
    coplanar_hit = _point_in_triangle_2d(first_2d[0], second_2d, tolerance) | (
        _point_in_triangle_2d(second_2d[0], first_2d, tolerance)
    )
    for first_index in range(3):
        for second_index in range(3):
            coplanar_hit = coplanar_hit | _segment_intersection_2d(
                first_2d[first_index],
                first_2d[(first_index + 1) % 3],
                second_2d[second_index],
                second_2d[(second_index + 1) % 3],
                tolerance,
            )
    return jnp.where(coplanar, coplanar_hit, edge_hits)


class ImplicitSurfaceEvidence(StrictModule):
    """JAX-safe evidence for one fixed-topology surface realization."""

    projection: ImplicitPointProjectionEvidence
    sign_pattern_unchanged: Array
    qef_solve_status: Array
    minimum_face_area: Array
    minimum_orientation_margin: Array
    intersection_free: Array
    finite: Array
    status: Array
    plan_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        projection: ImplicitPointProjectionEvidence,
        sign_pattern_unchanged: Any,
        qef_solve_status: Any,
        minimum_face_area: Any,
        minimum_orientation_margin: Any,
        intersection_free: Any,
        finite: Any,
        status: Any,
        plan_id: str,
        topology_id: str,
    ):
        self.projection = projection
        self.sign_pattern_unchanged = jnp.asarray(
            sign_pattern_unchanged, dtype=bool
        ).reshape(())
        self.qef_solve_status = jnp.asarray(qef_solve_status, dtype=jnp.int32)
        self.minimum_face_area = jnp.asarray(minimum_face_area, dtype=float).reshape(())
        self.minimum_orientation_margin = jnp.asarray(
            minimum_orientation_margin, dtype=float
        ).reshape(())
        self.intersection_free = jnp.asarray(intersection_free, dtype=bool).reshape(())
        self.finite = jnp.asarray(finite, dtype=bool).reshape(())
        self.status = jnp.asarray(status, dtype=jnp.int32).reshape(())
        self.plan_id = str(plan_id)
        self.topology_id = str(topology_id)

    @property
    def accepted(self) -> Array:
        return self.status == int(ImplicitSurfaceStatus.SUCCESS)

    @property
    def refresh_required(self) -> Array:
        refresh_bits = int(
            ImplicitSurfaceStatus.INVALID_GEOMETRY
            | ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED
            | ImplicitSurfaceStatus.PROJECTION_FAILED
        )
        return ((self.status & refresh_bits) != 0) | self.projection.refresh_required


class ImplicitSurfaceRealization(StrictModule):
    """Proposed and safe vertices sharing one immutable triangle topology."""

    proposed_vertices: Array
    vertices: Array
    faces: Array
    evidence: ImplicitSurfaceEvidence
    source_id: str = eqx.field(static=True)

    def __init__(
        self,
        proposed_vertices: Array,
        vertices: Array,
        faces: Array,
        evidence: ImplicitSurfaceEvidence,
        /,
        *,
        source_id: str,
    ):
        proposed = jnp.asarray(proposed_vertices, dtype=float)
        safe = jnp.asarray(vertices, dtype=proposed.dtype)
        faces_ = jnp.asarray(faces, dtype=jnp.int32)
        if proposed.ndim != 2 or proposed.shape[1] != 3 or safe.shape != proposed.shape:
            raise ValueError("Implicit surface vertices must have shape (vertices, 3).")
        if faces_.ndim != 2 or faces_.shape[1] != 3:
            raise ValueError("Implicit surface faces must have shape (faces, 3).")
        self.proposed_vertices = proposed
        self.vertices = safe
        self.faces = faces_
        self.evidence = evidence
        self.source_id = str(source_id)

    @property
    def accepted(self) -> Array:
        return self.evidence.accepted

    @property
    def refresh_required(self) -> Array:
        return self.evidence.refresh_required

    def to_triangle_mesh(self, /) -> TriangleMesh:
        if not bool(np.asarray(self.accepted)):
            raise ValueError("Only an accepted implicit realization can be materialized.")
        return TriangleMesh(
            np.asarray(self.vertices),
            np.asarray(self.faces),
            source_id=self.source_id,
        )


class ImplicitSurfacePlan(StrictModule):
    """Fixed-topology dual surface with differentiable coordinate realization."""

    kernel: GeometryKernel
    grid_points: Array
    inside_pattern: Array
    projection: ImplicitPointProjectionPlan
    vertex_anchor_indices: Array
    vertex_anchor_mask: Array
    qef_regularization: Array
    cell_lower: Array
    cell_upper: Array
    base_vertices: Array
    faces: Array
    base_face_normals: Array
    intersection_pairs: Array
    schema: ParameterSchema = eqx.field(static=True)
    tolerance: GeometryTolerance = eqx.field(static=True)
    policy: ImplicitSurfacePolicy = eqx.field(static=True)
    source_id: str = eqx.field(static=True)
    topology_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        geometry: CompiledGeometry,
        grid_points: Any,
        inside_pattern: Any,
        projection: ImplicitPointProjectionPlan,
        vertex_anchor_indices: Any,
        vertex_anchor_mask: Any,
        qef_regularization: Any,
        cell_lower: Any,
        cell_upper: Any,
        base_vertices: Any,
        faces: Any,
        base_face_normals: Any,
        intersection_pairs: Any,
        policy: ImplicitSurfacePolicy,
        source_id: str,
        topology_id: str,
    ):
        if not isinstance(projection, ImplicitPointProjectionPlan):
            raise TypeError("projection must be ImplicitPointProjectionPlan.")
        arrays = {
            "grid_points": np.asarray(grid_points, dtype=float),
            "inside_pattern": np.asarray(inside_pattern, dtype=bool),
            "vertex_anchor_indices": np.asarray(vertex_anchor_indices, dtype=np.int32),
            "vertex_anchor_mask": np.asarray(vertex_anchor_mask, dtype=bool),
            "qef_regularization": np.asarray(qef_regularization, dtype=float),
            "cell_lower": np.asarray(cell_lower, dtype=float),
            "cell_upper": np.asarray(cell_upper, dtype=float),
            "base_vertices": np.asarray(base_vertices, dtype=float),
            "faces": np.asarray(faces, dtype=np.int32),
            "base_face_normals": np.asarray(base_face_normals, dtype=float),
            "intersection_pairs": np.asarray(intersection_pairs, dtype=np.int32),
        }
        vertices = arrays["base_vertices"].shape[0]
        if arrays["base_vertices"].shape != (vertices, 3):
            raise ValueError("base_vertices must have shape (vertices, 3).")
        if arrays["cell_lower"].shape != (vertices, 3) or arrays["cell_upper"].shape != (
            vertices,
            3,
        ):
            raise ValueError("Implicit cell bounds must match base vertices.")
        if (
            arrays["vertex_anchor_indices"].shape != arrays["vertex_anchor_mask"].shape
            or arrays["vertex_anchor_indices"].shape[0] != vertices
        ):
            raise ValueError("Implicit vertex-anchor routes are inconsistent.")
        if arrays["qef_regularization"].shape != (vertices,) or np.any(
            arrays["qef_regularization"] <= 0.0
        ):
            raise ValueError(
                "qef_regularization must contain one positive value per vertex."
            )
        if arrays["faces"].ndim != 2 or arrays["faces"].shape[1] != 3:
            raise ValueError("faces must have shape (faces, 3).")
        if arrays["base_face_normals"].shape != (arrays["faces"].shape[0], 3):
            raise ValueError("base_face_normals must match faces.")
        if (
            arrays["intersection_pairs"].ndim != 2
            or arrays["intersection_pairs"].shape[1] != 2
        ):
            raise ValueError("intersection_pairs must have shape (pairs, 2).")
        identifier = canonical_fingerprint(
            {
                "kind": "implicit-surface-plan",
                "source_id": source_id,
                "topology_id": topology_id,
                "projection_id": projection.plan_id,
                "grid_points": arrays["grid_points"].tolist(),
                "faces": arrays["faces"].tolist(),
                "qef_regularization": arrays["qef_regularization"].tolist(),
                "policy": repr(policy),
            }
        )
        self.kernel = geometry.kernel
        self.grid_points = jnp.asarray(arrays["grid_points"])
        self.inside_pattern = jnp.asarray(arrays["inside_pattern"])
        self.projection = projection
        self.vertex_anchor_indices = jnp.asarray(arrays["vertex_anchor_indices"])
        self.vertex_anchor_mask = jnp.asarray(arrays["vertex_anchor_mask"])
        self.qef_regularization = jnp.asarray(arrays["qef_regularization"])
        self.cell_lower = jnp.asarray(arrays["cell_lower"])
        self.cell_upper = jnp.asarray(arrays["cell_upper"])
        self.base_vertices = jnp.asarray(arrays["base_vertices"])
        self.faces = jnp.asarray(arrays["faces"])
        self.base_face_normals = jnp.asarray(arrays["base_face_normals"])
        self.intersection_pairs = jnp.asarray(arrays["intersection_pairs"])
        self.schema = geometry.schema
        self.tolerance = geometry.tolerance
        self.policy = policy
        self.source_id = str(source_id)
        self.topology_id = str(topology_id)
        self.plan_id = identifier

    def realize(self, state: DesignState, /) -> ImplicitSurfaceRealization:
        if not isinstance(state, DesignState) or state.schema != self.schema:
            raise ValueError("Implicit realization state must use the discovery schema.")
        projection = self.projection.realize(state)
        field = self.kernel.boundary_field(state, self.grid_points)
        sign_pattern = field < 0.0
        sign_unchanged = jnp.all(sign_pattern == self.inside_pattern) & jnp.all(
            jnp.abs(field) > self.policy.lattice_zero_tolerance
        )
        points = projection.proposed_points[self.vertex_anchor_indices]
        normals = projection.normals[self.vertex_anchor_indices]
        mask = self.vertex_anchor_mask.astype(points.dtype)
        count = jnp.sum(mask, axis=1)
        mass = jnp.sum(points * mask[..., None], axis=1) / count[..., None]
        relative = points - mass[:, None, :]
        matrix = contract("vki,vkj,vk->vij", normals, normals, mask)
        matrix = matrix + (
            self.qef_regularization[:, None, None]
            * count[:, None, None]
            * jnp.eye(3, dtype=points.dtype)[None, :, :]
        )
        projection_scalar = contract("vki,vki->vk", normals, relative)
        right_hand_side = contract(
            "vki,vk,vk->vi",
            normals,
            projection_scalar,
            mask,
        )
        operator = DenseLinearOperator(
            matrix,
            properties=OperatorProperties(
                self_adjoint=True,
                positive_definite=True,
                evidence={"positive_definite": "construction"},
            ),
            operator_id=f"{self.plan_id}:qef",
        )
        qef = solve(
            LinearSystem(operator, problem_id=f"{self.plan_id}:qef-system"),
            right_hand_side,
            policy=_QEF_POLICY,
        )
        proposed = mass + qef.value
        finite = (
            jnp.all(jnp.isfinite(field))
            & jnp.all(jnp.isfinite(proposed))
            & projection.evidence.finite
        )
        in_cell = jnp.all(
            (proposed >= self.cell_lower - self.policy.projection.root_tolerance)
            & (proposed <= self.cell_upper + self.policy.projection.root_tolerance)
        )
        triangles = proposed[self.faces]
        face_normals = jnp.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        face_area = 0.5 * jnp.sqrt(jnp.sum(face_normals * face_normals, axis=-1))
        minimum_area = jnp.min(face_area)
        orientation = jnp.sum(face_normals * self.base_face_normals, axis=-1)
        minimum_orientation = jnp.min(orientation)
        if self.intersection_pairs.shape[0]:
            first_triangles = triangles[self.intersection_pairs[:, 0]]
            second_triangles = triangles[self.intersection_pairs[:, 1]]
            intersections = jax.vmap(
                lambda first, second: _triangles_intersect(
                    first,
                    second,
                    self.policy.projection.root_tolerance,
                )
            )(first_triangles, second_triangles)
            intersection_free = ~jnp.any(intersections)
        else:
            intersection_free = jnp.asarray(True)
        qef_success = jnp.all(qef.successful)
        status = jnp.asarray(int(ImplicitSurfaceStatus.SUCCESS), dtype=jnp.int32)
        status = status | jnp.where(
            projection.evidence.geometry.accepted,
            0,
            int(ImplicitSurfaceStatus.INVALID_GEOMETRY),
        ).astype(jnp.int32)
        status = status | jnp.where(
            sign_unchanged,
            0,
            int(ImplicitSurfaceStatus.SIGN_PATTERN_CHANGED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            projection.accepted,
            0,
            int(ImplicitSurfaceStatus.PROJECTION_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            qef_success & finite,
            0,
            int(ImplicitSurfaceStatus.QEF_FAILED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            in_cell,
            0,
            int(ImplicitSurfaceStatus.QEF_OUT_OF_CELL),
        ).astype(jnp.int32)
        status = status | jnp.where(
            minimum_area >= self.policy.minimum_face_area,
            0,
            int(ImplicitSurfaceStatus.DEGENERATE_FACE),
        ).astype(jnp.int32)
        status = status | jnp.where(
            minimum_orientation > 0.0,
            0,
            int(ImplicitSurfaceStatus.ORIENTATION_CHANGED),
        ).astype(jnp.int32)
        status = status | jnp.where(
            intersection_free,
            0,
            int(ImplicitSurfaceStatus.SELF_INTERSECTION),
        ).astype(jnp.int32)
        evidence = ImplicitSurfaceEvidence(
            projection=projection.evidence,
            sign_pattern_unchanged=sign_unchanged,
            qef_solve_status=qef.status,
            minimum_face_area=minimum_area,
            minimum_orientation_margin=minimum_orientation,
            intersection_free=intersection_free,
            finite=finite,
            status=status,
            plan_id=self.plan_id,
            topology_id=self.topology_id,
        )
        safe = jnp.where(evidence.accepted, proposed, self.base_vertices)
        return ImplicitSurfaceRealization(
            proposed,
            safe,
            self.faces,
            evidence,
            source_id=self.source_id,
        )


__all__ = [
    "ImplicitSurfaceEvidence",
    "ImplicitSurfacePlan",
    "ImplicitSurfaceRealization",
]
