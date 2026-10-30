#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import fsum

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.core import Tracer

from ..._physical import SpatialCoordinateContract
from ...typing import checked
from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._contracts import (
    ClosestPointResult,
    GeometryKernel,
    GeometryKind,
    GeometrySource,
)
from .._sampling import (
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingReport,
    SamplingResult,
)
from ..design._schema import _ParameterCollector, DesignState
from ._model import BRepImportReport, BRepModel
from ._patches import (
    AbstractSurfacePatch,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)
from ._projection import NativeBRepProjection, prepare_brep_projection
from ._projection_contracts import BRepProjectionStatus


@jax.custom_jvp
def _oriented_boundary_field(
    points: Array,
    closest_points: Array,
    outward_normals: Array,
    inside: Array,
) -> Array:
    difference = points - closest_points
    squared_distance = jnp.sum(difference * difference, axis=-1)
    away_from_boundary = squared_distance > 0.0
    distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
    signed_distance = jnp.where(inside, -distance, distance)
    boundary_linearization = jnp.sum(difference * outward_normals, axis=-1)
    return jnp.where(away_from_boundary, signed_distance, boundary_linearization)


@_oriented_boundary_field.defjvp
def _oriented_boundary_field_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
    points, closest_points, outward_normals, inside = primals
    points_tangent, _, _, _ = tangents
    value = _oriented_boundary_field(points, closest_points, outward_normals, inside)
    derivative = jnp.sum(outward_normals * points_tangent, axis=-1)
    return value, derivative


def _require_closed_solid(model: BRepModel, /) -> None:
    topology = model.topology
    geometry = model.geometry
    if not topology.num_solids:
        raise ValueError("BRepSource.model must contain at least one solid.")
    degenerate = tuple(index < 0 for index in geometry.edge_curves)
    for solid, (faces, face_signs) in enumerate(
        zip(topology.solid_faces, topology.solid_face_orientations, strict=True)
    ):
        balance = [0] * topology.num_edges
        for face, solid_sign in zip(faces, face_signs, strict=True):
            face_sign = solid_sign * int(model.orientation[face])
            for loop in geometry.face_loops[face]:
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    balance[edge] += face_sign * geometry.coedge_senses[coedge]
        invalid = tuple(
            edge
            for edge, value in enumerate(balance)
            if value != 0 and not degenerate[edge]
        )
        if invalid:
            raise ValueError(
                "BRepSource.model must have closed, consistently oriented source shells; "
                f"solid {solid} has unbalanced edges {invalid!r}."
            )


def _host_points(points: Array, /) -> tuple[np.ndarray, tuple[int, ...]]:
    if isinstance(points, Tracer):
        raise TypeError(
            "Native B-Rep source queries are host-evaluated and cannot trace through JAX."
        )
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(
            "BRepSource queries must be finite points with trailing dimension three."
        )
    return values.reshape((-1, 3)), values.shape[:-1]


def _face_polygon_vertices(model: BRepModel, face: int, /) -> tuple[np.ndarray, ...]:
    geometry = model.geometry
    loops: list[np.ndarray] = []
    for loop in geometry.face_loops[face]:
        vertices = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            first, last = geometry.edge_vertices[edge]
            vertex = first if geometry.coedge_senses[coedge] > 0 else last
            vertices.append(np.asarray(geometry.vertex_points[vertex], dtype=np.float64))
        if len(vertices) < 3:
            raise ValueError("Planar B-Rep source loops require at least three vertices.")
        loops.append(np.asarray(vertices, dtype=np.float64))
    return tuple(loops)


def _affine_planar_face(model: BRepModel, face: int, /) -> bool:
    patch = model.patches[face]
    return isinstance(patch, PlanePatch) or (
        isinstance(patch, ExtrusionSurface)
        and isinstance(patch.curve, (LineCurve, ExactLineCurve))
    )


def _face_area(model: BRepModel, face: int, /) -> float:
    loops = _face_polygon_vertices(model, face)
    area_vector = np.zeros((3,), dtype=np.float64)
    for polygon in loops:
        area_vector += 0.5 * np.sum(
            np.cross(polygon, np.roll(polygon, -1, axis=0)), axis=0
        )
    return float(np.linalg.norm(area_vector))


def _polyhedral_measure(model: BRepModel, /) -> float:
    geometry = model.geometry
    topology = model.topology
    volumes = []
    for faces, face_signs in zip(
        topology.solid_faces, topology.solid_face_orientations, strict=True
    ):
        volume_terms = []
        for face, solid_sign in zip(faces, face_signs, strict=True):
            sign = solid_sign * float(model.orientation[face])
            for polygon in _face_polygon_vertices(model, face):
                for index in range(1, polygon.shape[0] - 1):
                    volume_terms.append(
                        sign
                        * float(polygon[0] @ np.cross(polygon[index], polygon[index + 1]))
                        / 6.0
                    )
        volumes.append(abs(fsum(volume_terms)))
    return fsum(volumes)


def _source_bounds(model: BRepModel, /) -> np.ndarray:
    points = np.asarray(model.geometry.vertex_points, dtype=np.float64)
    if points.size == 0:
        raise ValueError("BRepSource bounds require source vertices.")
    lower, upper = np.min(points, axis=0), np.max(points, axis=0)
    for face, patch in enumerate(model.patches):
        bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
        match patch:
            case PlanePatch():
                continue
            case ExtrusionSurface(curve=curve) if isinstance(
                curve, (LineCurve, ExactLineCurve)
            ):
                uv = bounds
                values = np.asarray(
                    [
                        curve.evaluate(jnp.asarray(u)) + v * np.asarray(patch.direction)
                        for u in uv[:, 0]
                        for v in uv[:, 1]
                    ],
                    dtype=np.float64,
                )
                lower = np.minimum(lower, np.min(values, axis=0))
                upper = np.maximum(upper, np.max(values, axis=0))
            case _:
                raise NotImplementedError(
                    f"BRepSource bounds are unresolved for {type(patch).__name__}."
                )
    return np.stack((lower, upper))


class BRepSource(GeometrySource):
    """Spatial source whose queries evaluate native CAD carriers, never triangles."""

    model: BRepModel

    @checked
    def __init__(self, model: BRepModel) -> None:
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        _require_closed_solid(model)
        self.model = model

    @classmethod
    @checked
    def from_stored_fields(cls: type[BRepSource], value: BRepSource, /) -> BRepSource:
        return cls(value.model)

    @property
    def report(self) -> BRepImportReport:
        return self.model.report

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.model.coordinate_contract

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        del context
        return _BRepKernel(self.model, prepare_brep_projection(self.model))


class _BRepKernel(GeometryKernel):
    """Host-query kernel over source patches, coedges, and solid occurrences."""

    model: BRepModel
    projection: NativeBRepProjection

    @checked
    def __init__(self, model: BRepModel, projection: NativeBRepProjection) -> None:
        if projection.source_revision != model.source_revision:
            raise ValueError("The prepared projection must match the source revision.")
        self.model = model
        self.projection = projection

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
        capabilities = {
            GeometryCapability.REGION_QUERY,
            GeometryCapability.SIGNED_DISTANCE,
            GeometryCapability.CLOSEST_POINT,
            GeometryCapability.BOUNDARY_NORMAL,
            GeometryCapability.BOUNDARY_ATLAS,
        }
        if all(
            _affine_planar_face(self.model, face)
            for face in range(len(self.model.patches))
        ):
            capabilities.update(
                {
                    GeometryCapability.INTERIOR_MEASURE,
                    GeometryCapability.BOUNDARY_MEASURE,
                    GeometryCapability.INTERIOR_SAMPLING,
                }
            )
        if self.model.patches and all(
            domain is not None and domain.outer is not None and not domain.curve_loops
            for domain in self.model.trim_domains
        ) and all(
            isinstance(curve, (LineCurve, ExactLineCurve))
            for curve in self.model.geometry.pcurves
        ):
            capabilities.add(GeometryCapability.BOUNDARY_SAMPLING)
        return frozenset(capabilities)

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.APPROXIMATE,
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=None,
            validity_region="closed native B-Rep source; unsupported carrier queries are UNRESOLVED",
            parameter_differentiable=False,
            provenance=("native_brep", "source_evaluating_query"),
        )

    def _inside(self, points: np.ndarray, /) -> np.ndarray:
        result = self.projection.locate_solids(points)
        status = np.asarray(result.status)
        unresolved = status == BRepProjectionStatus.UNRESOLVED
        if np.any(unresolved):
            rows = np.flatnonzero(unresolved).tolist()
            raise ValueError(
                f"Native B-Rep membership is UNRESOLVED for query rows {rows!r}."
            )
        return (np.asarray(result.dimensions) == 3) & np.isin(
            status,
            (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.AMBIGUOUS),
        )

    def _nearest_boundary(
        self, points: np.ndarray, /
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        count = points.shape[0]
        best_upper = np.full((count,), np.inf, dtype=np.float64)
        best_lower = np.zeros((count,), dtype=np.float64)
        second_lower = np.full((count,), np.inf, dtype=np.float64)
        second_upper = np.full((count,), np.inf, dtype=np.float64)
        closest = np.full((count, 3), np.nan, dtype=np.float64)
        normals = np.full((count, 3), np.nan, dtype=np.float64)
        faces = np.full((count,), -1, dtype=np.int32)
        status = np.full((count,), BRepProjectionStatus.FAILED, dtype=np.int8)
        for face in range(len(self.model.patches)):
            result = self.projection.project(
                points,
                np.full((count,), 2, dtype=np.int8),
                np.full((count,), face, dtype=np.int32),
            )
            candidate_state = np.asarray(result.status)
            unresolved = candidate_state == BRepProjectionStatus.UNRESOLVED
            if np.any(unresolved):
                rows = np.flatnonzero(unresolved).tolist()
                raise ValueError(
                    f"Native B-Rep nearest-boundary query is UNRESOLVED for rows {rows!r}."
                )
            nonisolated = candidate_state == BRepProjectionStatus.NON_ISOLATED
            if np.any(nonisolated):
                rows = np.flatnonzero(nonisolated).tolist()
                raise ValueError(
                    f"Native B-Rep nearest-boundary query has non-isolated minima at rows {rows!r}."
                )
            valid = np.isin(
                candidate_state,
                (
                    BRepProjectionStatus.UNIQUE,
                    BRepProjectionStatus.SEAM,
                    BRepProjectionStatus.AMBIGUOUS,
                ),
            )
            candidate_upper = np.asarray(result.residuals)
            candidate_lower = np.asarray(result.distance_lower_bounds)
            better = valid & (candidate_upper < best_upper)
            update_second = valid & ~better & (candidate_upper < second_upper)
            old_best = better & np.isfinite(best_upper)
            second_lower[old_best] = best_lower[old_best]
            second_upper[old_best] = best_upper[old_best]
            second_lower[update_second] = candidate_lower[update_second]
            second_upper[update_second] = candidate_upper[update_second]
            best_upper[better] = candidate_upper[better]
            best_lower[better] = candidate_lower[better]
            closest[better] = np.asarray(result.points)[better]
            normals[better] = np.asarray(result.normals)[better]
            faces[better] = face
            status[better] = candidate_state[better]
        if np.any(~np.isfinite(best_upper)):
            rows = np.flatnonzero(~np.isfinite(best_upper)).tolist()
            raise ValueError(
                f"Native B-Rep source has no resolved boundary minimum for rows {rows!r}."
            )
        tied = np.isfinite(second_upper) & (
            (best_lower <= second_upper + self.projection.policy.ambiguity_tolerance)
            & (second_lower <= best_upper + self.projection.policy.ambiguity_tolerance)
        )
        status[tied] = BRepProjectionStatus.AMBIGUOUS
        return closest, normals, faces, best_lower, best_upper, status

    def contains(self, state: DesignState, points: Array, /) -> Array:
        del state
        query, leading = _host_points(points)
        return jnp.asarray(self._inside(query).reshape(leading), dtype=jnp.bool_)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        del state
        query, leading = _host_points(points)
        closest, normals, _, lower, upper, status = self._nearest_boundary(query)
        width_limit = (
            256.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.max(np.abs(query), axis=1))
        )
        if (
            np.any(status == BRepProjectionStatus.AMBIGUOUS)
            or not np.all(np.isfinite(normals))
            or np.any(upper - lower > width_limit)
        ):
            raise ValueError(
                "Signed B-Rep boundary field is unresolved at an ambiguous or uncertified minimum."
            )
        inside = self._inside(query)
        return _oriented_boundary_field(
            jnp.asarray(query),
            jnp.asarray(closest),
            jnp.asarray(normals),
            jnp.asarray(inside),
        ).reshape(leading)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        del state
        query, leading = _host_points(points)
        _, normals, _, lower, upper, status = self._nearest_boundary(query)
        width_limit = (
            256.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.max(np.abs(query), axis=1))
        )
        if (
            np.any(status == BRepProjectionStatus.AMBIGUOUS)
            or not np.all(np.isfinite(normals))
            or np.any(upper - lower > width_limit)
        ):
            raise ValueError(
                "B-Rep boundary normal is unresolved at an ambiguous or uncertified minimum."
            )
        return jnp.asarray(normals, dtype=jnp.float64).reshape((*leading, 3))

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        del state
        query, leading = _host_points(points)
        closest, normals, faces, lower, upper, status = self._nearest_boundary(query)
        inside = self._inside(query)
        displacement = query - closest
        distances = np.linalg.norm(displacement, axis=-1)
        signed = np.where(inside, -distances, distances)
        unique = np.isin(status, (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.SEAM))
        width_limit = (
            256.0
            * np.finfo(np.float64).eps
            * np.maximum(1.0, np.max(np.abs(query), axis=1))
        )
        valid_normal = (
            unique & np.all(np.isfinite(normals), axis=1) & (upper - lower <= width_limit)
        )
        return ClosestPointResult(
            closest_point=jnp.asarray(closest, dtype=jnp.float64).reshape((*leading, 3)),
            normal_coordinate=jnp.asarray(signed, dtype=jnp.float64).reshape(leading),
            oriented_normal=jnp.asarray(normals, dtype=jnp.float64).reshape(
                (*leading, 3)
            ),
            source_entity_id=jnp.asarray(faces, dtype=jnp.int32).reshape(leading),
            unique=jnp.asarray(unique, dtype=jnp.bool_).reshape(leading),
            regular=jnp.zeros(leading, dtype=jnp.bool_),
            margin=jnp.zeros(leading, dtype=jnp.float64),
            normal_coordinate_valid=jnp.asarray(valid_normal, dtype=jnp.bool_).reshape(
                leading
            ),
            represented_geometry_id=self.model.source_revision,
            physical_geometry_id=self.model.source_revision,
            exact_to_physical=bool(np.all(upper - lower <= width_limit)),
        )

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return jnp.asarray(_source_bounds(self.model), dtype=jnp.float64)

    def measure(self, state: DesignState, /) -> Array:
        del state
        if not all(
            _affine_planar_face(self.model, face)
            for face in range(len(self.model.patches))
        ):
            raise NotImplementedError(
                "Native B-Rep measure is unresolved for curved source faces."
            )
        return jnp.asarray(_polyhedral_measure(self.model), dtype=jnp.float64)

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        if not all(
            _affine_planar_face(self.model, face)
            for face in range(len(self.model.patches))
        ):
            raise NotImplementedError(
                "Native B-Rep boundary measure is unresolved for curved source faces."
            )
        return jnp.asarray(
            fsum(_face_area(self.model, face) for face in range(len(self.model.patches))),
            dtype=jnp.float64,
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
        if isinstance(num_points, bool) or not isinstance(num_points, int):
            raise TypeError("num_points must be an integer.")
        if num_points < 0:
            raise ValueError("num_points must be nonnegative.")
        plan_ = RejectionSamplingPlan() if plan is None else plan
        if not isinstance(plan_, RejectionSamplingPlan):
            raise TypeError("plan must be a RejectionSamplingPlan or None.")
        bounds = np.asarray(self.bounds(state), dtype=np.float64)
        samples = np.zeros((num_points, 3), dtype=np.float64)
        accepted = 0
        proposed = 0
        rounds = 0
        loop_key = key
        while accepted < num_points and rounds < plan_.maximum_rounds:
            loop_key, proposal_key = jr.split(loop_key)
            candidates = np.asarray(
                jr.uniform(
                    proposal_key,
                    (plan_.proposals_per_round, 3),
                    minval=bounds[0],
                    maxval=bounds[1],
                    dtype=jnp.float64,
                ),
                dtype=np.float64,
            )
            accepted_rows = np.flatnonzero(self._inside(candidates))
            take = min(num_points - accepted, accepted_rows.size)
            if take:
                samples[accepted : accepted + take] = candidates[accepted_rows[:take]]
                accepted += take
            proposed += plan_.proposals_per_round
            rounds += 1
        valid = jnp.arange(num_points, dtype=jnp.int32) < accepted
        report = SamplingReport(
            requested=num_points,
            proposed=jnp.asarray(proposed, dtype=jnp.int32),
            accepted=jnp.asarray(accepted, dtype=jnp.int32),
            rounds=jnp.asarray(rounds, dtype=jnp.int32),
        )
        return SamplingResult(jnp.asarray(samples), valid, report)

    def sample_boundary(
        self,
        state: DesignState,
        num_points: int,
        /,
        *,
        key: Array,
    ) -> SamplingResult:
        del state
        if not self.model.patches or any(
            domain is None or domain.outer is None or domain.curve_loops
            for domain in self.model.trim_domains
        ):
            raise NotImplementedError(
                "Native B-Rep boundary sampling requires polygonal source chart trims."
            )
        if any(
            not isinstance(curve, (LineCurve, ExactLineCurve))
            for curve in self.model.geometry.pcurves
        ):
            raise NotImplementedError(
                "Native B-Rep boundary sampling requires line p-curves."
            )
        return sample_boundary_atlas(self.model.boundary_atlas, num_points, key=key)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        return self.model.boundary_atlas


__all__ = ["BRepSource"]
