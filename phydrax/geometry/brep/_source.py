#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax import Array

from ...units import UnitDefinition


if TYPE_CHECKING:
    from ...interchange._cad import CadImportPolicy

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
    bounded_rejection_sample,
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingResult,
)
from ..design._schema import _ParameterCollector, DesignState
from ._model import BRepImportReport, BRepModel
from ._projection_contracts import BRepProjectionStatus
from ._query import prepare_brep_query, PreparedBRepQuery


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
def _oriented_boundary_field_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array],
) -> tuple[Array, Array]:
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


def _require_closed_solid(model: BRepModel) -> None:
    """Require authoritative exact geometry with closed, consistently oriented shells."""
    geometry = model.geometry
    if geometry is None:
        raise ValueError(
            "A native BRepSource requires authoritative exact curves and trims."
        )
    if model.topology.num_solids == 0:
        raise ValueError("A BRepSource requires at least one solid.")
    for solid in range(model.topology.num_solids):
        balance = geometry.edge_use_balance(model.orientation, solid=solid)
        if any(
            value != 0 and not degenerate
            for value, degenerate in zip(balance, geometry.degenerate_edges, strict=True)
        ):
            raise ValueError(
                "A BRepSource requires closed, consistently oriented region shells."
            )


class BRepSource(GeometrySource):
    """Direct CAD source preserving B-Rep topology and parametric face charts.

    Native exact curves, trimmed faces and occurrence placements are authoritative;
    derived tessellation never decides membership or closest points.
    """

    model: BRepModel

    @checked
    def __init__(self, model: BRepModel) -> None:
        _require_closed_solid(model)
        self.model = model

    @property
    def report(self) -> BRepImportReport:
        return self.model.report

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.model.coordinate_contract

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        del context
        return _NativeBRepKernel(self.model, prepare_brep_query(self.model))


_CAPABILITIES = frozenset(
    {
        GeometryCapability.REGION_QUERY,
        GeometryCapability.SIGNED_DISTANCE,
        GeometryCapability.CLOSEST_POINT,
        GeometryCapability.BOUNDARY_NORMAL,
        GeometryCapability.INTERIOR_MEASURE,
        GeometryCapability.BOUNDARY_MEASURE,
        GeometryCapability.INTERIOR_SAMPLING,
        GeometryCapability.BOUNDARY_SAMPLING,
        GeometryCapability.BOUNDARY_ATLAS,
    }
)


class _NativeBRepKernel(GeometryKernel):
    """Exact-geometry kernel over prepared native B-Rep queries."""

    model: BRepModel
    query: PreparedBRepQuery
    face_entity_codes: Array
    atlas: BoundaryAtlas
    boundary_definition_faces: Array

    def __init__(self, model: BRepModel, query: PreparedBRepQuery) -> None:
        geometry = model.geometry
        if geometry is None:
            raise ValueError(
                "A native B-Rep kernel requires authoritative source geometry."
            )
        inventory = {
            entity: index for index, entity in enumerate(model.qualified_face_ids)
        }
        face_identity = {
            (record.container.occurrence_path, record.member.index): record.member
            for record in geometry.qualified_entity_incidence(model.source_revision)
            if record.container.kind == "solid" and record.member.kind == "face"
        }
        codes = np.full(
            (len(geometry.occurrences) + 1, len(model.patches)), -1, dtype=np.int32
        )
        for face, entity in enumerate(model.face_ids):
            if entity in inventory:
                codes[0, face] = inventory[entity]
        for index, occurrence in enumerate(geometry.occurrences):
            for face in model.topology.solid_faces[occurrence.solid]:
                codes[index + 1, face] = inventory[face_identity[(occurrence.path, face)]]
        atlas = model.boundary_atlas
        definition_faces = np.asarray(
            [
                model.qualified_face_ids[index].index
                for index in np.asarray(atlas.source_entity_ids)
            ],
            dtype=np.int32,
        )
        self.model, self.query = model, query
        self.face_entity_codes = jnp.asarray(codes, dtype=jnp.int32)
        self.atlas = atlas
        self.boundary_definition_faces = jnp.asarray(definition_faces, dtype=jnp.int32)

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
        return _CAPABILITIES

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.EXACT,
            sign_reliability=SignReliability.RELIABLE,
            distance_semantics=DistanceSemantics.APPROXIMATE,
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=None,
            validity_region=(
                "exact trimmed B-Rep boundary; conservative source-bound closest "
                "point discovery with explicit unresolved work and pointwise status"
            ),
            parameter_differentiable=False,
            provenance=("native_brep", "exact_boundary_query"),
        )

    def _flat(self, points: Array, /) -> tuple[Array, tuple[int, ...]]:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        return points_.reshape((-1, 3)), points_.shape[:-1]

    def contains(self, state: DesignState, points: Array, /) -> Array:
        del state
        flat, leading = self._flat(points)
        inside = jnp.zeros((flat.shape[0],), dtype=jnp.bool_)
        resolved = jnp.ones((flat.shape[0],), dtype=jnp.bool_)
        geometry = self.model.geometry
        if geometry is None:
            raise RuntimeError("A native B-Rep kernel lost its exact geometry.")
        for occurrence in geometry.occurrences:
            result = self.query.contains(flat, path=occurrence.path)
            boundary = result.distances <= self.query.tolerances.classifier
            resolved &= (result.status == BRepProjectionStatus.UNIQUE) | boundary
            inside |= result.inside
        inside = eqx.error_if(
            inside,
            ~jnp.all(resolved),
            "Native B-Rep containment exhausted its certification budget.",
        )
        return inside.reshape(leading)

    def boundary_field(self, state: DesignState, points: Array, /) -> Array:
        flat, leading = self._flat(points)
        closest = jax.lax.stop_gradient(self.query.closest_point(flat))
        return _oriented_boundary_field(
            flat,
            closest.points,
            closest.normals,
            self.contains(state, flat),
        ).reshape(leading)

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        del state
        flat, leading = self._flat(points)
        normals = jax.lax.stop_gradient(self.query.closest_point(flat).normals)
        return normals.reshape((*leading, 3))

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        flat, leading = self._flat(points)
        closest = jax.lax.stop_gradient(self.query.closest_point(flat))
        inside = self.contains(state, flat)
        on_boundary = closest.distances <= self.query.tolerances.classifier
        coordinate = jnp.where(
            on_boundary,
            jnp.sum((flat - closest.points) * closest.normals, axis=-1),
            jnp.where(inside, -closest.distances, closest.distances),
        )
        single = (closest.status == BRepProjectionStatus.UNIQUE) | (
            closest.status == BRepProjectionStatus.SEAM
        )
        regular = (closest.status == BRepProjectionStatus.UNIQUE) & jnp.all(
            jnp.isfinite(closest.normals), axis=-1
        )
        return ClosestPointResult(
            closest_point=closest.points.reshape((*leading, 3)),
            normal_coordinate=coordinate.reshape(leading),
            oriented_normal=closest.normals.reshape((*leading, 3)),
            source_entity_id=self.face_entity_codes[
                closest.occurrences + 1, closest.faces
            ].reshape(leading),
            unique=single.reshape(leading),
            regular=regular.reshape(leading),
            margin=jnp.zeros(leading, dtype=flat.dtype),
            represented_geometry_id=self.model.model_id,
            physical_geometry_id=self.model.source_revision,
            exact_to_physical=True,
            normal_coordinate_valid=(single & jnp.isfinite(coordinate)).reshape(leading),
        )

    def bounds(self, state: DesignState, /) -> Array:
        del state
        return self.query.bounds

    def measure(self, state: DesignState, /) -> Array:
        del state
        geometry = self.model.geometry
        if geometry is None:
            raise RuntimeError("A native B-Rep kernel lost its exact geometry.")
        solids = jnp.asarray(
            [occurrence.solid for occurrence in geometry.occurrences], dtype=jnp.int32
        )
        return jnp.sum(self.query.measures.solid_volumes[solids])

    def boundary_measure(self, state: DesignState, /) -> Array:
        del state
        return jnp.sum(self.query.measures.face_areas[self.boundary_definition_faces])

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
        return sample_boundary_atlas(self.atlas, num_points, key=key)

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        del state
        return self.atlas


def BRep(
    path: str | Path,
    policy: CadImportPolicy,
    /,
    *,
    trusted_root: str | Path,
    source_length_unit: UnitDefinition | None = None,
) -> BRepSource:
    """Decode an exact native CAD source under explicit units and resource policy."""
    from ...interchange._cad_io import read_cad

    result = read_cad(
        path,
        policy,
        trusted_root=trusted_root,
        source_length_unit=source_length_unit,
    )
    return BRepSource(result.model)


__all__ = ["BRep", "BRepSource"]
