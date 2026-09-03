#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from .._atlas import BoundaryAtlas
from .._capabilities import GeometryCapability
from .._certificate import (
    DistanceSemantics,
    FieldCertificate,
    FieldRegularity,
    SignReliability,
    ZeroSetAccuracy,
)
from .._closest_point import represented_mesh_closest_point, triangle_query_evidence
from .._contracts import GeometryKernel, GeometryKind, GeometrySource
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingResult,
)
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ..simplicial import MeshQueryResult
from ..simplicial._mesh import _closest_points_on_triangles
from ._model import BRepBoundaryMap, BRepModel
from ._patches import AbstractSurfacePatch


@dataclass(frozen=True, slots=True)
class BRepParameterLink:
    """Assign one face field to a shared stable design parameter."""

    face_index: int
    field: str
    parameter_id: ParameterId

    def __post_init__(self):
        if self.face_index < 0:
            raise ValueError("face_index must be non-negative.")
        if not self.field:
            raise ValueError("field must be non-empty.")


class _PatchBinding(StrictModule):
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)
    tree_definition: Any = eqx.field(static=True)

    def __init__(self, bindings, tree_definition):
        self.bindings = tuple(bindings)
        self.tree_definition = tree_definition

    def realize(self, state: DesignState, /) -> AbstractSurfacePatch:
        return jax.tree_util.tree_unflatten(
            self.tree_definition,
            tuple(binding.read(state) for binding in self.bindings),
        )


class FixedTopologyBRepRealization(StrictModule):
    """Differentiable face and welded-mesh realization at one design state."""

    patches: tuple[AbstractSurfacePatch, ...]
    vertices: Array
    faces: Array
    atlas: BoundaryAtlas
    seam_residual: Array

    def __init__(self, *, patches, vertices, faces, atlas, seam_residual):
        self.patches = tuple(patches)
        self.vertices = jnp.asarray(vertices, dtype=float)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.atlas = atlas
        self.seam_residual = jnp.asarray(seam_residual, dtype=float).reshape(())


def _corner_weld_weights(model: BRepModel) -> np.ndarray:
    vertex_indices = np.asarray(model.mesh_faces, dtype=np.int32).reshape((-1,))
    face_indices = np.repeat(np.asarray(model.triangle_face_ids, dtype=np.int32), 3)
    pair_counts: dict[tuple[int, int], int] = {}
    incident_faces: dict[int, set[int]] = {}
    for vertex, face in zip(vertex_indices.tolist(), face_indices.tolist(), strict=True):
        pair = (vertex, face)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
        incident_faces.setdefault(vertex, set()).add(face)
    return np.asarray(
        [
            1.0 / (pair_counts[(vertex, face)] * len(incident_faces[vertex]))
            for vertex, face in zip(
                vertex_indices.tolist(), face_indices.tolist(), strict=True
            )
        ],
        dtype=float,
    )


def _evaluate_corners(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
) -> Array:
    chart_indices = jnp.repeat(model.triangle_face_ids, 3)
    parameters = model.triangle_parameters.reshape((-1, 2))
    bounds = model.parameter_bounds[chart_indices]
    reference = (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    return BRepBoundaryMap(patches, model.parameter_bounds).map(chart_indices, reference)


def evaluate_fixed_topology_mesh(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    /,
    *,
    corner_weights: Array | None = None,
) -> tuple[Array, Array]:
    """Evaluate CAD faces, then weld shared mesh vertices by face-balanced averaging."""

    if len(patches) != len(model.patches):
        raise ValueError("patches must preserve the imported face count and ordering.")
    weights = (
        jnp.asarray(_corner_weld_weights(model), dtype=float)
        if corner_weights is None
        else jnp.asarray(corner_weights, dtype=float).reshape((-1,))
    )
    corners = _evaluate_corners(patches, model)
    vertex_indices = model.mesh_faces.reshape((-1,))
    if weights.shape != (vertex_indices.shape[0],):
        raise ValueError("corner_weights must contain one entry per triangle corner.")
    vertices = (
        jnp.zeros_like(model.mesh_vertices)
        .at[vertex_indices]
        .add(corners * weights[:, None])
    )
    return vertices, model.mesh_faces


def _seam_residual(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
) -> Array:
    corners = _evaluate_corners(patches, model)
    vertex_indices = model.mesh_faces.reshape((-1,))
    reference = jnp.zeros_like(model.mesh_vertices)
    reference = reference.at[vertex_indices].add(corners)
    counts = jnp.zeros((model.mesh_vertices.shape[0],), dtype=corners.dtype)
    counts = counts.at[vertex_indices].add(1.0)
    reference = reference / counts[:, None]
    difference = corners - reference[vertex_indices]
    tiny = jnp.finfo(difference.dtype).tiny
    distances = jnp.sqrt(jnp.sum(difference * difference, axis=-1) + tiny) - jnp.sqrt(
        tiny
    )
    return jnp.max(distances)


class FixedTopologyBRepSource(GeometrySource):
    """Differentiable B-Rep source with immutable incidence and mesh connectivity."""

    model: BRepModel
    parameter_links: tuple[BRepParameterLink, ...] = eqx.field(static=True)
    trainable_fields: frozenset[str] = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        *,
        parameter_links: Sequence[BRepParameterLink] = (),
        trainable_fields: Sequence[str] = (
            "origin",
            "center",
            "radius",
            "reference_radius",
            "semi_angle",
            "major_radius",
            "minor_radius",
            "control_points",
            "weights",
        ),
    ):
        if not isinstance(model, BRepModel):
            raise TypeError("model must be a BRepModel.")
        links = tuple(parameter_links)
        keys = tuple((link.face_index, link.field) for link in links)
        if len(set(keys)) != len(keys):
            raise ValueError("Each face field may have at most one parameter link.")
        if any(link.face_index >= len(model.patches) for link in links):
            raise ValueError("A parameter link references an absent face.")
        self.model = model
        self.parameter_links = links
        self.trainable_fields = frozenset(trainable_fields)

    def _compile(self, context: _ParameterCollector, /) -> GeometryKernel:
        links: Mapping[tuple[int, str], ParameterId] = {
            (link.face_index, link.field): link.parameter_id
            for link in self.parameter_links
        }
        patch_bindings: list[_PatchBinding] = []
        for face_index, patch in enumerate(self.model.patches):
            path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(patch)
            bindings: list[ParameterBinding] = []
            for path, value in path_leaves:
                field = jax.tree_util.keystr(path).removeprefix(".")
                parameter_id = links.get(
                    (face_index, field),
                    ParameterId(f"{self.model.source_revision}:face:{face_index}", field),
                )
                value_host = np.asarray(value)
                scale = float(max(np.max(np.abs(value_host), initial=0.0), 1.0))
                bounds = (
                    (float(np.finfo(float).eps), None)
                    if field == "weights"
                    else (None, None)
                )
                bindings.append(
                    context.bind(
                        parameter_id,
                        value,
                        role=f"brep_face_{field}",
                        physical_scale=scale,
                        bounds=bounds,
                        trainable=field in self.trainable_fields,
                    )
                )
            patch_bindings.append(_PatchBinding(bindings, tree_definition))
        return _FixedTopologyBRepKernel(
            self.model,
            tuple(patch_bindings),
            _corner_weld_weights(self.model),
        )


class _FixedTopologyBRepKernel(GeometryKernel):
    model: BRepModel
    patch_bindings: tuple[_PatchBinding, ...] = eqx.field(static=True)
    corner_weights: Array

    def __init__(self, model, patch_bindings, corner_weights):
        self.model = model
        self.patch_bindings = tuple(patch_bindings)
        self.corner_weights = jnp.asarray(corner_weights, dtype=float)

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
                GeometryCapability.SEAM_DIAGNOSTICS,
            }
        )

    @property
    def field_certificate(self) -> FieldCertificate:
        return FieldCertificate(
            zero_set_accuracy=ZeroSetAccuracy.TOLERANCE_BOUND,
            sign_reliability=SignReliability.LOCAL,
            distance_semantics=DistanceSemantics.APPROXIMATE,
            regularity=FieldRegularity.PIECEWISE_SMOOTH,
            safe_step_factor=1.0,
            validity_region="fixed topology; requires positive Jacobians and compatible seams",
            parameter_differentiable=True,
            provenance=("occt_brep", "fixed_topology_realization"),
        )

    def _patches(self, state: DesignState) -> tuple[AbstractSurfacePatch, ...]:
        return tuple(binding.realize(state) for binding in self.patch_bindings)

    def realize(self, state: DesignState, /) -> FixedTopologyBRepRealization:
        patches = self._patches(state)
        vertices, faces = evaluate_fixed_topology_mesh(
            patches,
            self.model,
            corner_weights=self.corner_weights,
        )
        atlas = BoundaryAtlas(
            BRepBoundaryMap(patches, self.model.parameter_bounds),
            source_entity_ids=jnp.arange(len(patches), dtype=jnp.int32),
            source_id=self.model.source_id,
            physical_tags=self.model.physical_tags,
            orientation=self.model.orientation,
            trim_domains=self.model.trim_domains,
        )
        return FixedTopologyBRepRealization(
            patches=patches,
            vertices=vertices,
            faces=faces,
            atlas=atlas,
            seam_residual=_seam_residual(patches, self.model),
        )

    def seam_residual(self, state: DesignState, /) -> Array:
        return self.realize(state).seam_residual

    def _triangles(self, state: DesignState) -> Array:
        realization = self.realize(state)
        return realization.vertices[realization.faces]

    def _query(self, state: DesignState, points: Array) -> MeshQueryResult:
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

    def contains(self, state: DesignState, points: Array, /) -> Array:
        points_ = jnp.asarray(points, dtype=float)
        triangles = self._triangles(state)
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

    def boundary_normal(self, state: DesignState, points: Array, /) -> Array:
        return self._query(state, points).normal

    def closest_point(self, state: DesignState, points: Array, /):
        points_ = jnp.asarray(points, dtype=float)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        triangles = self._triangles(state)
        closest_by_face = jax.vmap(_closest_points_on_triangles, in_axes=(0, None))(
            flat, triangles
        )
        query = self._query(state, flat)
        unique, regular, margin = triangle_query_evidence(
            flat,
            triangles,
            closest_by_face,
            query.face_index,
        )
        return represented_mesh_closest_point(
            points_,
            closest_point=query.closest_point.reshape((*leading, 3)),
            distance=query.distance.reshape(leading),
            normal=query.normal.reshape((*leading, 3)),
            source_entity_id=query.face_index.reshape(leading),
            inside=self.contains(state, points_),
            unique=unique.reshape(leading),
            regular=regular.reshape(leading),
            margin=margin.reshape(leading),
            represented_geometry_id=(
                f"{self.model.source_id}:fixed-topology-query:"
                f"{self.model.source_revision}"
            ),
            physical_geometry_id=self.model.source_id,
            exact_to_physical=False,
        )

    def bounds(self, state: DesignState, /) -> Array:
        vertices = self.realize(state).vertices
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def measure(self, state: DesignState, /) -> Array:
        triangles = self._triangles(state)
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
        triangles = self._triangles(state)
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
            lambda proposal_key, count: jax.random.uniform(
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
        return sample_boundary_atlas(
            self.realize(state).atlas,
            num_points,
            key=key,
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        return self.realize(state).atlas


__all__ = [
    "BRepParameterLink",
    "FixedTopologyBRepRealization",
    "FixedTopologyBRepSource",
    "evaluate_fixed_topology_mesh",
]
