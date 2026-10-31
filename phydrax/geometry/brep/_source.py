#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from math import fsum

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array
from jax.core import Tracer
from jax.typing import DTypeLike

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
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
)
from ._projection import NativeBRepProjection, prepare_brep_projection
from ._projection_contracts import BRepProjectionPolicy, BRepProjectionStatus


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


def _host_points(points: np.ndarray | Array, /) -> tuple[np.ndarray, tuple[int, ...]]:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3 or not np.all(np.isfinite(values)):
        raise ValueError(
            "BRepSource queries must be finite points with trailing dimension three."
        )
    return values.reshape((-1, 3)), values.shape[:-1]


def _width_limit(query: np.ndarray, /) -> np.ndarray:
    """Admissible certified distance-enclosure width at each query point."""
    return (
        256.0 * np.finfo(np.float64).eps * np.maximum(1.0, np.max(np.abs(query), axis=1))
    )


class BRepSourceDerivativeError(TypeError):
    """A native B-Rep source query output has no certified point derivative."""


@partial(jax.custom_jvp, nondiff_argnums=(0,))
def _refuse_derivative(query: str, points: Array, /) -> Array:
    return points


@_refuse_derivative.defjvp
def _refuse_derivative_jvp(
    query: str, primals: tuple[Array], tangents: tuple[Array], /
) -> tuple[Array, Array]:
    del primals, tangents
    raise BRepSourceDerivativeError(
        f"Native B-Rep source {query} has no certified point derivative; "
        "differentiate boundary_field, or a planar-source boundary_normal."
    )


def _source_query(
    points: Array,
    host: Callable[[np.ndarray], tuple[np.ndarray, ...]],
    outputs: tuple[tuple[tuple[int, ...], DTypeLike], ...],
    /,
    *,
    underived: str | None,
) -> tuple[Array, ...]:
    """Evaluate a host source query at concrete or traced points.

    The query is always evaluated on the host from exact source carriers. Traced
    points reach it through a callback on their primal values. ``underived=None``
    states that the outputs are discrete or locally constant data, or that the
    caller supplies their derivative; otherwise differentiating the outputs raises
    ``BRepSourceDerivativeError`` naming that query instead of yielding zeros.
    """
    if not isinstance(points, Tracer):
        query, leading = _host_points(points)
        return tuple(
            jnp.asarray(np.asarray(value).reshape((*leading, *shape)), dtype=dtype)
            for value, (shape, dtype) in zip(host(query), outputs, strict=True)
        )
    values = jnp.asarray(points, dtype=jnp.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError(
            "BRepSource queries must be finite points with trailing dimension three."
        )
    leading = values.shape[:-1]

    def callback(concrete: np.ndarray, /) -> tuple[np.ndarray, ...]:
        query, batch = _host_points(concrete)
        return tuple(
            np.asarray(value, dtype=dtype).reshape((*batch, *shape))
            for value, (shape, dtype) in zip(host(query), outputs, strict=True)
        )

    arguments = (
        jax.lax.stop_gradient(values)
        if underived is None
        else jax.lax.stop_gradient(_refuse_derivative(underived, values))
    )
    return tuple(
        jax.pure_callback(
            callback,
            tuple(
                jax.ShapeDtypeStruct((*leading, *shape), dtype)
                for shape, dtype in outputs
            ),
            arguments,
            vmap_method="expand_dims",
        )
    )


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
        # Containment is exact: only the binary64 resolution of the source scale
        # counts as boundary, never the projection's default classifier band.
        scale = max(
            1.0,
            float(
                np.max(np.abs(np.asarray(self.model.geometry.vertex_points)), initial=0.0)
            ),
        )
        policy = BRepProjectionPolicy(
            classifier_tolerance=256.0 * float(np.finfo(np.float64).eps) * scale
        )
        return _BRepKernel(self.model, prepare_brep_projection(self.model, policy=policy))


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
        if (
            self.model.patches
            and all(
                domain is not None and domain.outer is not None and not domain.curve_loops
                for domain in self.model.trim_domains
            )
            and all(
                isinstance(curve, (LineCurve, ExactLineCurve))
                for curve in self.model.geometry.pcurves
            )
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

    def _certified_boundary(
        self, query: np.ndarray, role: str, /
    ) -> tuple[np.ndarray, np.ndarray]:
        closest, normals, _, lower, upper, status = self._nearest_boundary(query)
        if (
            np.any(status == BRepProjectionStatus.AMBIGUOUS)
            or not np.all(np.isfinite(normals))
            or np.any(upper - lower > _width_limit(query))
        ):
            raise ValueError(
                f"{role} is unresolved at an ambiguous or uncertified minimum."
            )
        return closest, normals

    def _boundary_frame(self, query: np.ndarray, /) -> tuple[np.ndarray, ...]:
        closest, normals = self._certified_boundary(query, "Signed B-Rep boundary field")
        return closest, normals, self._inside(query)

    def _boundary_normals(self, query: np.ndarray, /) -> tuple[np.ndarray, ...]:
        return (self._certified_boundary(query, "B-Rep boundary normal")[1],)

    def _closest_rows(self, query: np.ndarray, /) -> tuple[np.ndarray, ...]:
        closest, normals, faces, lower, upper, status = self._nearest_boundary(query)
        distances = np.linalg.norm(query - closest, axis=-1)
        signed = np.where(self._inside(query), -distances, distances)
        unique = np.isin(status, (BRepProjectionStatus.UNIQUE, BRepProjectionStatus.SEAM))
        certified = upper - lower <= _width_limit(query)
        valid_normal = unique & np.all(np.isfinite(normals), axis=1) & certified
        return closest, signed, normals, faces, unique, valid_normal, certified

    def contains(self, state: DesignState, points: Array, /) -> Array:
        del state
        (inside,) = _source_query(
            points,
            lambda query: (self._inside(query),),
            (((), jnp.bool_),),
            underived=None,
        )
        return inside

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        del state
        # The signed field's derivative is the oriented normal, supplied by the
        # custom JVP of ``_oriented_boundary_field`` on the traced points.
        closest, normals, inside = _source_query(
            points,
            self._boundary_frame,
            (((3,), jnp.float64), ((3,), jnp.float64), ((), jnp.bool_)),
            underived=None,
        )
        return _oriented_boundary_field(
            jnp.asarray(points, dtype=jnp.float64), closest, normals, inside
        )

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        del state
        # Over affine planar faces the selected face normal is locally constant;
        # curved faces would need their shape operator, which is not certified.
        planar = all(
            _affine_planar_face(self.model, face)
            for face in range(len(self.model.patches))
        )
        (normals,) = _source_query(
            points,
            self._boundary_normals,
            (((3,), jnp.float64),),
            underived=None if planar else "boundary_normal",
        )
        return normals

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        del state
        closest, signed, normals, faces, unique, valid_normal, certified = _source_query(
            points,
            self._closest_rows,
            (
                ((3,), jnp.float64),
                ((), jnp.float64),
                ((3,), jnp.float64),
                ((), jnp.int32),
                ((), jnp.bool_),
                ((), jnp.bool_),
                ((), jnp.bool_),
            ),
            underived="closest_point",
        )
        leading = unique.shape
        return ClosestPointResult(
            closest_point=closest,
            normal_coordinate=signed,
            oriented_normal=normals,
            source_entity_id=faces,
            unique=unique,
            regular=jnp.zeros(leading, dtype=jnp.bool_),
            margin=jnp.zeros(leading, dtype=jnp.float64),
            normal_coordinate_valid=valid_normal,
            represented_geometry_id=self.model.source_revision,
            physical_geometry_id=self.model.source_revision,
            # Exactness is a static claim: traced queries cannot inspect their
            # certificates, so they conservatively claim none.
            exact_to_physical=(
                not isinstance(certified, Tracer) and bool(np.all(np.asarray(certified)))
            ),
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


__all__ = ["BRepSource", "BRepSourceDerivativeError"]
