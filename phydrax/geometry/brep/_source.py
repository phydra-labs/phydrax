#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import Array

from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._closest_point import represented_mesh_closest_point
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingResult,
)
from ..design._schema import _ParameterCollector, DesignState
from ..simplicial import TriangleMesh, TriangleMeshQueryIndex
from ._model import BRepImportReport, BRepModel
from ._occt import import_brep


@jax.custom_jvp
def _oriented_boundary_field(
    points: Array,
    closest_points: Array,
    outward_normals: Array,
    inside: Array,
) -> Array:
    """Signed distance with the selected outward normal as its point derivative."""
    difference = points - closest_points
    squared_distance = jnp.sum(difference * difference, axis=-1)
    away_from_boundary = squared_distance > 0.0
    distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
    signed_distance = jnp.where(inside, -distance, distance)
    boundary_linearization = jnp.sum(difference * outward_normals, axis=-1)
    return jnp.where(
        away_from_boundary,
        signed_distance,
        boundary_linearization,
    )


@_oriented_boundary_field.defjvp
def _oriented_boundary_field_jvp(primals, tangents):
    points, closest_points, outward_normals, inside = primals
    points_tangent, _, _, _ = tangents
    value = _oriented_boundary_field(
        points,
        closest_points,
        outward_normals,
        inside,
    )
    tangent = jnp.sum(outward_normals * points_tangent, axis=-1)
    return value, tangent


class BRepSource(GeometrySource):
    """Direct CAD source preserving B-Rep topology and parametric face charts."""

    model: BRepModel

    def __init__(self, model: BRepModel):
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        mesh = TriangleMesh(model.mesh_vertices, model.mesh_faces)
        if not mesh.topology.watertight:
            raise ValueError(
                "A solid BRepSource requires a watertight query tessellation."
            )
        self.model = model

    @property
    def report(self) -> BRepImportReport:
        return self.model.report

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        del context
        return _BRepKernel(self.model)


class _BRepKernel(GeometryKernel):
    model: BRepModel
    mesh: TriangleMesh
    query_index: TriangleMeshQueryIndex

    def __init__(self, model: BRepModel):
        self.model = model
        self.mesh = TriangleMesh(
            model.mesh_vertices,
            model.mesh_faces,
            source_id=f"{model.source_id}:query-mesh:{model.source_revision}",
        )
        self.query_index = self.mesh.query_index()

    @property
    def ambient_dimension(self) -> int:
        return 3

    @property
    def intrinsic_dimension(self) -> int:
        return 3

    @property
    def kind(self) -> GeometryKind:
        return GeometryKind.REGION

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return frozenset(
            {
                GeometryCapability.REGION_QUERY,
                GeometryCapability.SIGNED_DISTANCE,
                GeometryCapability.CLOSEST_POINT,
                GeometryCapability.BOUNDARY_NORMAL,
                GeometryCapability.MEASURE,
                GeometryCapability.INTERIOR_SAMPLING,
                GeometryCapability.BOUNDARY_SAMPLING,
                GeometryCapability.BOUNDARY_ATLAS,
            }
        )

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.TOLERANCE_BOUND,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.APPROXIMATE,
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=1.0,
            validity_region=(
                f"watertight query mesh; linear deflection "
                f"{self.model.report.linear_deflection:g}"
            ),
            parameter_differentiable=False,
            provenance=("occt_brep", "reported_query_tessellation"),
        )

    def _triangles(self) -> Array:
        return self.mesh.vertices[self.mesh.faces]

    def _query(self, points: Array):
        return self.query_index.query(points)

    def contains(self, state: DesignState, points: Array, /) -> Array:
        del state
        points_ = jnp.asarray(points, dtype=self.mesh.vertices.dtype)
        triangles = self._triangles()
        first = triangles[:, 0] - points_[..., None, :]
        second = triangles[:, 1] - points_[..., None, :]
        third = triangles[:, 2] - points_[..., None, :]
        numerator = jnp.sum(first * jnp.cross(second, third), axis=-1)
        denominator = (
            jnp.linalg.norm(first, axis=-1)
            * jnp.linalg.norm(second, axis=-1)
            * jnp.linalg.norm(third, axis=-1)
            + jnp.sum(first * second, axis=-1) * jnp.linalg.norm(third, axis=-1)
            + jnp.sum(second * third, axis=-1) * jnp.linalg.norm(first, axis=-1)
            + jnp.sum(third * first, axis=-1) * jnp.linalg.norm(second, axis=-1)
        )
        winding = jnp.sum(2.0 * jnp.arctan2(numerator, denominator), axis=-1)
        return jnp.abs(winding / (4.0 * jnp.pi)) > 0.5

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        points_ = jnp.asarray(points, dtype=float)
        query = self._query(points_)
        return _oriented_boundary_field(
            points_,
            query.closest_point,
            query.normal,
            self.contains(state, points_),
        )

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        del state
        return jax.lax.stop_gradient(self._query(points).normal)

    def closest_point(self, state: DesignState, points: Array, /):
        points_ = jnp.asarray(points, dtype=self.mesh.vertices.dtype)
        query = self._query(points_)
        leading = points_.shape[:-1]
        unavailable = jnp.zeros(leading, dtype=bool)
        return represented_mesh_closest_point(
            points_,
            closest_point=query.closest_point,
            distance=query.distance,
            normal=jax.lax.stop_gradient(query.normal),
            source_entity_id=query.face_index,
            inside=self.contains(state, points_),
            unique=unavailable,
            regular=unavailable,
            margin=jnp.zeros(leading, dtype=points_.dtype),
            represented_geometry_id=self.mesh.source_id,
            physical_geometry_id=self.model.source_id,
            exact_to_physical=False,
        )

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.stack(
            (jnp.min(self.mesh.vertices, axis=0), jnp.max(self.mesh.vertices, axis=0))
        )

    def measure(self, state: DesignState, /) -> Array:
        del state
        triangles = self._triangles()
        return jnp.abs(
            jnp.sum(
                jnp.sum(
                    triangles[:, 0] * jnp.cross(triangles[:, 1], triangles[:, 2]),
                    axis=-1,
                )
            )
            / 6.0
        )

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        triangles = self._triangles()
        return jnp.sum(
            0.5
            * jnp.linalg.norm(
                jnp.cross(
                    triangles[:, 1] - triangles[:, 0],
                    triangles[:, 2] - triangles[:, 0],
                ),
                axis=-1,
            )
        )

    def sample_interior(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Array,
        plan: RejectionSamplingPlan | None = None,
    ) -> SamplingResult:
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

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Array,
    ) -> SamplingResult:
        del state
        return sample_boundary_atlas(self.model.boundary_atlas, num_points, key=key)

    def boundary_atlas(self, state: DesignState, /):
        del state
        return self.model.boundary_atlas


def BRep(
    path: str | Path,
    *,
    linear_deflection: float = 1e-3,
    angular_deflection: float = 0.1,
    trim_samples_per_edge: int = 33,
) -> BRepSource:
    """Construct a direct STEP/IGES/BREP geometry source with explicit report."""

    return BRepSource(
        import_brep(
            path,
            linear_deflection=linear_deflection,
            angular_deflection=angular_deflection,
            trim_samples_per_edge=trim_samples_per_edge,
        )
    )


__all__ = ["BRep", "BRepSource"]
