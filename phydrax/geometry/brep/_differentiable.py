#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.tree_util import PyTreeDef
from jax.typing import ArrayLike

from ..._bvh import refit_packed_bvh_bounds
from ..._physical import SpatialCoordinateContract
from ..._strict import StrictModule
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
from .._closest_point import represented_mesh_closest_point, triangle_query_evidence
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
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._model import BRepBoundaryMap, BRepModel, BRepPCurve, BRepPlacedBoundaryMap
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    EllipseCurve,
    ExtrusionSurface,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
    TorusPatch,
)
from ._placed import PlacedSurface
from ._projection_contracts import BRepProjectionStatus
from ._query import (
    _EdgeData,
    _FaceData,
    _jadd,
    _jvector,
    _trimmed_face_rules,
    BRepQueryPolicy,
    prepare_brep_query,
    PreparedBRepQuery,
)
from ._source import _require_closed_solid


if TYPE_CHECKING:
    from ..simplicial import MeshQueryResult


@dataclass(frozen=True, slots=True)
class BRepParameterLink:
    """Assign one face field to a shared stable design parameter."""

    face_index: int
    field: str
    parameter_id: ParameterId

    def __post_init__(self) -> None:
        if self.face_index < 0:
            raise ValueError("face_index must be non-negative.")
        if not self.field:
            raise ValueError("field must be non-empty.")


class _PatchBinding(StrictModule):
    bindings: tuple[ParameterBinding, ...] = eqx.field(static=True)
    tree_definition: PyTreeDef = eqx.field(static=True)

    def __init__(
        self, bindings: Iterable[ParameterBinding], tree_definition: PyTreeDef
    ) -> None:
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

    def __init__(
        self,
        *,
        patches: Iterable[AbstractSurfacePatch],
        vertices: ArrayLike,
        faces: ArrayLike,
        atlas: BoundaryAtlas,
        seam_residual: ArrayLike,
    ) -> None:
        self.patches = tuple(patches)
        self.vertices = jnp.asarray(vertices, dtype=jnp.float64)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.atlas = atlas
        self.seam_residual = jnp.asarray(seam_residual, dtype=jnp.float64).reshape(())


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
        dtype=np.float64,
    )


def _evaluate_corners(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
) -> Array:
    chart_indices = jnp.repeat(model.triangle_face_ids, 3)
    parameters = model.triangle_parameters.reshape((-1, 2))
    bounds = model.parameter_bounds[chart_indices]
    reference = (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    points = BRepBoundaryMap(patches, model.parameter_bounds).map(
        chart_indices, reference
    )
    geometry = model.geometry
    if geometry is None or not geometry.occurrences:
        return points
    occurrences = jnp.repeat(model.triangle_occurrence_ids, 3)
    rotations = jnp.asarray(
        [occurrence.rotation for occurrence in geometry.occurrences], dtype=jnp.float64
    )
    translations = jnp.asarray(
        [occurrence.translation for occurrence in geometry.occurrences], dtype=jnp.float64
    )
    indices = jnp.maximum(occurrences, 0)
    placed = (rotations[indices] @ points[..., None])[..., 0] + translations[indices]
    return jnp.where((occurrences >= 0)[:, None], placed, points)


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
        jnp.asarray(_corner_weld_weights(model), dtype=jnp.float64)
        if corner_weights is None
        else jnp.asarray(corner_weights, dtype=jnp.float64).reshape((-1,))
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
    return jnp.max(distances, initial=0.0)


class FixedTopologyBRepSource(GeometrySource):
    """Differentiable B-Rep source with immutable incidence and mesh connectivity."""

    model: BRepModel
    parameter_links: tuple[BRepParameterLink, ...] = eqx.field(static=True)
    trainable_fields: frozenset[str] = eqx.field(static=True)

    @checked
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
    ) -> None:
        _require_closed_solid(model)
        links = tuple(parameter_links)
        keys = tuple((link.face_index, link.field) for link in links)
        if len(set(keys)) != len(keys):
            raise ValueError("Each face field may have at most one parameter link.")
        if any(link.face_index >= len(model.patches) for link in links):
            raise ValueError("A parameter link references an absent face.")
        self.model = model
        self.parameter_links = links
        self.trainable_fields = frozenset(trainable_fields)

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.model.coordinate_contract

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
                    (float(np.finfo(np.float64).eps), None)
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
            None
            if self.model.geometry is None
            else _trimmed_face_rules(self.model, BRepQueryPolicy()),
        )


def _interval_product(
    first: tuple[Array, Array], second: tuple[Array, Array], /
) -> tuple[Array, Array]:
    a, b, c, d = (
        first[0] * second[0],
        first[0] * second[1],
        first[1] * second[0],
        first[1] * second[1],
    )
    return jnp.minimum(jnp.minimum(a, b), jnp.minimum(c, d)), jnp.maximum(
        jnp.maximum(a, b), jnp.maximum(c, d)
    )


def _trig_intervals(
    first: Array, last: Array, /
) -> tuple[tuple[Array, Array], tuple[Array, Array]]:
    period = 2.0 * jnp.pi
    slack = 32.0 * jnp.finfo(jnp.float64).eps * (1.0 + jnp.abs(first) + jnp.abs(last))

    def includes(phase: float) -> Array:
        return phase + period * jnp.ceil((first - slack - phase) / period) <= last + slack

    cosine = (
        jnp.where(includes(pi), -1.0, jnp.minimum(jnp.cos(first), jnp.cos(last))),
        jnp.where(includes(0.0), 1.0, jnp.maximum(jnp.cos(first), jnp.cos(last))),
    )
    sine = (
        jnp.where(includes(-0.5 * pi), -1.0, jnp.minimum(jnp.sin(first), jnp.sin(last))),
        jnp.where(includes(0.5 * pi), 1.0, jnp.maximum(jnp.sin(first), jnp.sin(last))),
    )
    return cosine, sine


def _affine_interval_box(
    origin: Array, terms: tuple[tuple[Array, tuple[Array, Array]], ...], /
) -> Array:
    lower, upper = origin, origin
    for vector, interval in terms:
        a, b = vector * interval[0], vector * interval[1]
        lower, upper = lower + jnp.minimum(a, b), upper + jnp.maximum(a, b)
    box = jnp.stack((lower, upper))
    margin = 256.0 * jnp.finfo(jnp.float64).eps * jnp.maximum(1.0, jnp.max(jnp.abs(box)))
    return box + jnp.asarray((-1.0, 1.0), dtype=jnp.float64)[:, None] * margin


def _curve_interval_box(curve: AbstractCurve, interval: Array, /) -> Array:
    first, last = interval[0], interval[1]
    if isinstance(curve, LineCurve):
        return _affine_interval_box(curve.origin, ((curve.direction, (first, last)),))
    if isinstance(curve, (CircleCurve, EllipseCurve)):
        cosine, sine = _trig_intervals(first, last)
        if isinstance(curve, CircleCurve):
            a, b = curve.radius, curve.radius
        else:
            a, b = curve.first_radius, curve.second_radius
        return _affine_interval_box(
            curve.center, ((a * curve.first_axis, cosine), (b * curve.second_axis, sine))
        )
    if isinstance(curve, BSplineCurve):
        lower, upper = (
            jnp.min(curve.control_points, axis=0),
            jnp.max(curve.control_points, axis=0),
        )
        return _affine_interval_box(
            lower, ((upper - lower, (jnp.asarray(0.0), jnp.asarray(1.0))),)
        )
    raise TypeError("This realized curve has no native interval-box capability.")


def _surface_interval_box(patch: AbstractSurfacePatch, box: Array, /) -> Array:
    u, v = (box[0, 0], box[1, 0]), (box[0, 1], box[1, 1])
    if isinstance(patch, PlacedSurface):
        source = _surface_interval_box(patch.definition, box)
        image = _jvector((patch.rotation, patch.rotation), (source[0], source[1]))
        return jnp.stack(_jadd(image, (patch.translation, patch.translation)))
    if isinstance(patch, PlanePatch):
        return _affine_interval_box(
            patch.origin, ((patch.first_axis, u), (patch.second_axis, v))
        )
    cosine_u, sine_u = _trig_intervals(*u)
    if isinstance(patch, (CylinderPatch, ConePatch)):
        if isinstance(patch, CylinderPatch):
            radius = (patch.radius, patch.radius)
        else:
            a, b = (
                patch.reference_radius + v[0] * jnp.tan(patch.semi_angle),
                patch.reference_radius + v[1] * jnp.tan(patch.semi_angle),
            )
            radius = (jnp.minimum(a, b), jnp.maximum(a, b))
        return _affine_interval_box(
            patch.origin,
            (
                (patch.first_axis, _interval_product(radius, cosine_u)),
                (patch.second_axis, _interval_product(radius, sine_u)),
                (patch.axis, v),
            ),
        )
    if isinstance(patch, (SpherePatch, TorusPatch)):
        cosine_v, sine_v = _trig_intervals(*v)
        if isinstance(patch, SpherePatch):
            radius = (patch.radius * cosine_v[0], patch.radius * cosine_v[1])
            height = (patch.radius * sine_v[0], patch.radius * sine_v[1])
        else:
            radial = _interval_product((patch.minor_radius, patch.minor_radius), cosine_v)
            radius = (patch.major_radius + radial[0], patch.major_radius + radial[1])
            height = _interval_product((patch.minor_radius, patch.minor_radius), sine_v)
        return _affine_interval_box(
            patch.center,
            (
                (patch.first_axis, _interval_product(radius, cosine_u)),
                (patch.second_axis, _interval_product(radius, sine_u)),
                (patch.axis, height),
            ),
        )
    if isinstance(patch, BSplineSurfacePatch):
        lower, upper = (
            jnp.min(patch.control_points, axis=(0, 1)),
            jnp.max(patch.control_points, axis=(0, 1)),
        )
        return _affine_interval_box(
            lower, ((upper - lower, (jnp.asarray(0.0), jnp.asarray(1.0))),)
        )
    if isinstance(patch, ExtrusionSurface):
        base = _curve_interval_box(patch.curve, box[:, 0])
        sweep = _affine_interval_box(
            jnp.zeros((3,), dtype=jnp.float64), ((patch.direction, v),)
        )
        return base + sweep
    if isinstance(patch, RevolutionSurface):
        base = _curve_interval_box(patch.curve, box[:, 1]) - patch.axis_origin
        radius = jnp.linalg.norm(jnp.maximum(jnp.abs(base[0]), jnp.abs(base[1])))
        return jnp.stack((patch.axis_origin - radius, patch.axis_origin + radius))
    if isinstance(patch, RuledSurface):
        first, second = (
            _curve_interval_box(patch.first, box[:, 0]),
            _curve_interval_box(patch.second, box[:, 0]),
        )
        left = _interval_product((1.0 - v[1], 1.0 - v[0]), (first[0], first[1]))
        right = _interval_product(v, (second[0], second[1]))
        return jnp.stack((left[0] + right[0], left[1] + right[1]))
    raise TypeError("This realized patch has no native interval-box capability.")


class _RealizedTrimEdge(AbstractCurve):
    patch: AbstractSurfacePatch
    pcurve: BRepPCurve

    @property
    def ambient_dimension(self) -> int:
        return 3

    def evaluate(self, parameters: Array, /) -> Array:
        return self.patch.evaluate(self.pcurve.evaluate(parameters))

    def bounding_box(self, first: float, last: float, /) -> np.ndarray:
        return self.patch.bounding_box(self.pcurve.bounding_box(first, last))


def _realized_face(face: _FaceData, patch: AbstractSurfacePatch, /) -> _FaceData:
    center = jnp.zeros((3,), dtype=jnp.float64)
    projection = jnp.zeros((3, 3), dtype=jnp.float64)
    radii = jnp.zeros((2,), dtype=jnp.float64)
    if isinstance(patch, PlanePatch):
        center = patch.origin
        normal = jnp.cross(patch.first_axis, patch.second_axis)
        normal = normal / jnp.linalg.norm(normal)
        projection = jnp.outer(normal, normal)
    elif isinstance(patch, (SpherePatch, CylinderPatch)):
        center = patch.center if isinstance(patch, SpherePatch) else patch.origin
        if isinstance(patch, SpherePatch):
            projection = jnp.eye(3, dtype=jnp.float64)
            frame = jnp.stack((patch.first_axis, patch.second_axis, patch.axis), axis=1)
        else:
            axis = patch.axis / jnp.linalg.norm(patch.axis)
            projection = jnp.eye(3, dtype=jnp.float64) - jnp.outer(axis, axis)
            frame = projection @ jnp.stack((patch.first_axis, patch.second_axis), axis=1)
        defect = jax.lax.stop_gradient(
            jnp.linalg.norm(frame.T @ frame - jnp.eye(frame.shape[1], dtype=jnp.float64))
        )
        margin = 256.0 * jnp.finfo(jnp.float64).eps
        radii = jnp.abs(patch.radius) * jnp.stack(
            (
                jnp.sqrt(jnp.maximum(0.0, 1.0 - defect - margin)),
                jnp.sqrt(1.0 + defect + margin),
            )
        )

    def bound(box: Array) -> Array:
        return _surface_interval_box(patch, box)

    boxes = jax.vmap(bound)(face.parameter_boxes)

    return eqx.tree_at(
        lambda value: (
            value.patch,
            value.seed_points,
            value.interval_boxes,
            value.pole_points,
            value.carrier_center,
            value.carrier_projection,
            value.carrier_radii,
            value.discovery_bvh,
        ),
        face,
        (
            patch,
            patch.evaluate(face.seed_uv),
            boxes,
            patch.evaluate(face.pole_parameters),
            center,
            projection,
            radii,
            refit_packed_bvh_bounds(face.discovery_bvh, boxes[:, 0], boxes[:, 1]),
        ),
    )


def _realized_placed_patch(
    patch: AbstractSurfacePatch,
    definition: AbstractSurfacePatch,
    /,
) -> PlacedSurface:
    if not isinstance(patch, PlacedSurface):
        raise TypeError(
            "A materialized world face must retain its canonical source pose."
        )
    return eqx.tree_at(lambda value: value.definition, patch, definition)


def _realized_native_query(
    template: PreparedBRepQuery,
    patches: tuple[AbstractSurfacePatch, ...],
    rules: tuple[tuple[Array, Array], ...],
    /,
) -> PreparedBRepQuery:
    """Refresh numeric source leaves without host preparation during tracing."""
    faces = tuple(
        _realized_face(face, patch)
        for face, patch in zip(template.faces, patches, strict=True)
    )
    edges = []
    for edge_index, edge in enumerate(template.edges):
        face_index = template.model.topology.edge_faces[edge_index][0]
        face = faces[face_index]
        local_index = face.coedge_edges.index(edge_index)
        curve = _RealizedTrimEdge(face.patch, face.pcurves[local_index])
        start = curve.evaluate(jnp.asarray(edge.first, dtype=jnp.float64))
        end = curve.evaluate(jnp.asarray(edge.last, dtype=jnp.float64))
        samples = curve.evaluate(edge.seed_parameters)
        boxes = jnp.broadcast_to(
            jnp.stack(
                (
                    jnp.full((3,), -jnp.inf, dtype=jnp.float64),
                    jnp.full((3,), jnp.inf, dtype=jnp.float64),
                )
            ),
            edge.interval_boxes.shape,
        )
        edges.append(
            _EdgeData(
                curve=None if edge.curve is None else curve,
                start_point=start,
                end_point=end,
                seed_parameters=edge.seed_parameters,
                seed_points=samples,
                interval_boxes=boxes,
                discovery_bvh=refit_packed_bvh_bounds(
                    edge.discovery_bvh, boxes[:, 0], boxes[:, 1]
                ),
                first=edge.first,
                last=edge.last,
                closed=edge.closed,
            )
        )
    face_boxes = jnp.stack(
        [
            _surface_interval_box(patch, template.model.parameter_bounds[index])
            for index, patch in enumerate(patches)
        ]
    )
    nodes, vector_areas, areas = [], [], []
    for index, (patch, (parameters, weights)) in enumerate(
        zip(patches, rules, strict=True)
    ):
        differential = jax.vmap(jax.jacfwd(patch.evaluate))(parameters)
        cross = jnp.cross(differential[:, :, 0], differential[:, :, 1])
        nodes.append(patch.evaluate(parameters))
        vector_areas.append(template.model.orientation[index] * weights[:, None] * cross)
        areas.append(weights * jnp.linalg.norm(cross, axis=-1))
    quadrature = eqx.tree_at(
        lambda value: (value.points, value.vector_areas, value.areas),
        template.quadrature,
        (jnp.concatenate(nodes), jnp.concatenate(vector_areas), jnp.concatenate(areas)),
    )
    result = eqx.tree_at(
        lambda value: (
            value.faces,
            value.edges,
            value.face_boxes,
            value.face_bvh,
            value.quadrature,
        ),
        template,
        (
            faces,
            tuple(edges),
            face_boxes,
            refit_packed_bvh_bounds(
                template.face_bvh, face_boxes[:, 0], face_boxes[:, 1]
            ),
            quadrature,
        ),
    )
    # This preparation mode is static metadata, not a replaceable numeric leaf.
    object.__setattr__(result, "certify_on_host", False)
    return result


class _FixedTopologyBRepKernel(GeometryKernel):
    """Fixed-epoch realization; exact models integrate measures over their trims.

    ``face_rules`` holds, per face, fixed parameter nodes and Green-reduced
    weights of the exact trimmed domain. The domain is fixed by the epoch's
    p-curves, so volume and area derivatives follow the realized patches.
    """

    model: BRepModel
    patch_bindings: tuple[_PatchBinding, ...] = eqx.field(static=True)
    corner_weights: Array
    face_rules: tuple[tuple[Array, Array], ...] | None
    native_query: PreparedBRepQuery | None
    reference_atlas: BoundaryAtlas

    def __init__(
        self,
        model: BRepModel,
        patch_bindings: Iterable[_PatchBinding],
        corner_weights: ArrayLike,
        face_rules: tuple[tuple[Array, Array], ...] | None,
    ) -> None:
        self.model = model
        self.patch_bindings = tuple(patch_bindings)
        self.corner_weights = jnp.asarray(corner_weights, dtype=jnp.float64)
        self.face_rules = face_rules
        self.native_query = None if model.geometry is None else prepare_brep_query(model)
        self.reference_atlas = model.boundary_atlas

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
                GeometryCapability.INTERIOR_MEASURE,
                GeometryCapability.BOUNDARY_MEASURE,
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
            provenance=(
                "native_brep" if self.model.geometry is not None else "occt_brep",
                "fixed_topology_realization",
            ),
        )

    def _patches(self, state: DesignState) -> tuple[AbstractSurfacePatch, ...]:
        return tuple(binding.realize(state) for binding in self.patch_bindings)

    def _native_query(
        self, patches: tuple[AbstractSurfacePatch, ...], /
    ) -> PreparedBRepQuery:
        template = self.native_query
        if template is None:
            raise RuntimeError("A native query requires exact native geometry.")
        geometry = self.model.geometry
        if geometry is None or self.face_rules is None:
            raise RuntimeError(
                "Native realization lost its exact geometry or trim quadrature."
            )
        result = _realized_native_query(template, patches, self.face_rules)
        if template.world_query is not None:
            world_patches = tuple(
                _realized_placed_patch(patch, patches[source])
                for patch, source in zip(
                    template.world_query.model.patches,
                    template.world_face_sources,
                    strict=True,
                )
            )
            world_rules = tuple(
                self.face_rules[source] for source in template.world_face_sources
            )
            world = _realized_native_query(
                template.world_query, world_patches, world_rules
            )
            result = eqx.tree_at(lambda value: value.world_query, result, world)
        return result

    def _native_membership(
        self, query: PreparedBRepQuery, points: Array, /
    ) -> tuple[Array, Array]:
        inside, unresolved = [], []
        for occurrence in query.occurrences:
            result = query.contains(points, path=occurrence.path)
            inside.append(result.inside)
            unresolved.append(
                result.unresolved & (result.distances > query.tolerances.classifier)
            )
        states, pending = jnp.stack(inside), jnp.stack(unresolved)
        certified_inside = jnp.any(states & ~pending, axis=0)
        return jnp.any(states, axis=0), jnp.any(pending, axis=0) & ~certified_inside

    def realize(self, state: DesignState, /) -> FixedTopologyBRepRealization:
        patches = self._patches(state)
        vertices, faces = evaluate_fixed_topology_mesh(
            patches,
            self.model,
            corner_weights=self.corner_weights,
        )
        mapping = self.reference_atlas.mapping
        if isinstance(mapping, BRepPlacedBoundaryMap):
            mapping = eqx.tree_at(lambda value: value.base.patches, mapping, patches)
        elif isinstance(mapping, BRepBoundaryMap):
            mapping = eqx.tree_at(lambda value: value.patches, mapping, patches)
        else:
            raise TypeError("A B-Rep realization requires its owning native chart map.")
        atlas = eqx.tree_at(lambda value: value.mapping, self.reference_atlas, mapping)
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
        from ..simplicial import MeshQueryResult
        from ..simplicial._mesh import _closest_points_on_triangles

        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        if self.native_query is not None:
            result = self._native_query(self._patches(state)).closest_point(flat)
            projected = eqx.error_if(
                result.points,
                jnp.any(result.unresolved),
                "Native realized closest-point discovery is unresolved.",
            )
            return MeshQueryResult(
                closest_point=projected.reshape((*leading, 3)),
                distance=result.distances.reshape(leading),
                face_index=result.faces.reshape(leading),
                normal=result.normals.reshape((*leading, 3)),
            )
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
        points_ = jnp.asarray(points, dtype=jnp.float64)
        if self.native_query is not None:
            patches = self._patches(state)
            points_ = eqx.error_if(
                points_,
                _seam_residual(patches, self.model) > 1.0e-8,
                "The realized B-Rep has incompatible source seams.",
            )
            leading = points_.shape[:-1]
            query = self._native_query(patches)
            inside, unresolved = self._native_membership(query, points_.reshape((-1, 3)))
            inside = eqx.error_if(
                inside, jnp.any(unresolved), "Native realized containment is unresolved."
            )
            return inside.reshape(leading)
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

    def closest_point(self, state: DesignState, points: Array, /) -> ClosestPointResult:
        from ..simplicial._mesh import _closest_points_on_triangles

        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        if self.native_query is not None:
            native = self._native_query(self._patches(state))
            hit = native.closest_point(flat)
            inside, unresolved = self._native_membership(native, flat)
            regular = (
                ~hit.unresolved
                & ~unresolved
                & jnp.all(jnp.isfinite(hit.normals), axis=-1)
            )
            unique = regular & (hit.status == int(BRepProjectionStatus.UNIQUE))
            boundary = hit.distances <= native.tolerances.classifier
            normal_coordinate = jnp.where(
                boundary,
                jnp.sum((flat - hit.points) * hit.normals, axis=-1),
                jnp.where(inside, -hit.distances, hit.distances),
            )
            return ClosestPointResult(
                closest_point=hit.points.reshape((*leading, 3)),
                normal_coordinate=normal_coordinate.reshape(leading),
                oriented_normal=hit.normals.reshape((*leading, 3)),
                source_entity_id=hit.faces.reshape(leading),
                unique=unique.reshape(leading),
                regular=regular.reshape(leading),
                margin=jnp.where(unique, hit.distance_lower_bounds, 0.0).reshape(leading),
                represented_geometry_id=f"{self.model.model_id}:fixed-topology-query",
                physical_geometry_id=self.model.source_revision,
                exact_to_physical=True,
                normal_coordinate_valid=regular.reshape(leading),
            )
        triangles = self._triangles(state)
        closest_by_face = jax.vmap(_closest_points_on_triangles, in_axes=(0, None))(
            flat, triangles
        )
        query = self._query(state, flat)
        distance_sq = jnp.sum((closest_by_face - flat[:, None, :]) ** 2, axis=-1)
        second_distance_sq = (
            jnp.sort(distance_sq, axis=-1)[:, 1]
            if triangles.shape[0] > 1
            else jnp.full((flat.shape[0],), jnp.inf, dtype=distance_sq.dtype)
        )
        unique, regular, margin = triangle_query_evidence(
            flat,
            triangles[query.face_index],
            query.closest_point,
            second_distance_sq,
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
            represented_geometry_id=f"{self.model.model_id}:fixed-topology-query",
            physical_geometry_id=self.model.source_revision,
            exact_to_physical=False,
        )

    def bounds(self, state: DesignState, /) -> Array:
        if self.native_query is not None:
            return self._native_query(self._patches(state)).bounds
        vertices = self.realize(state).vertices
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def _face_integrals(self, state: DesignState, /) -> tuple[Array, Array]:
        """Per-face oriented volume flux ``int S . n dA / 3`` and area."""
        if self.face_rules is None:
            raise RuntimeError("Face integrals require exact trimmed face rules.")
        volumes, areas = [], []
        orientation = self.model.orientation
        for face, (patch, (parameters, weights)) in enumerate(
            zip(self._patches(state), self.face_rules, strict=True)
        ):
            points = patch.evaluate(parameters)
            differential = jax.vmap(jax.jacfwd(patch.evaluate))(parameters)
            normal = jnp.cross(differential[:, :, 0], differential[:, :, 1])
            volumes.append(
                orientation[face] * jnp.sum(weights * jnp.sum(points * normal, -1)) / 3.0
            )
            areas.append(jnp.sum(weights * jnp.linalg.norm(normal, axis=-1)))
        return jnp.stack(volumes), jnp.stack(areas)

    def measure(self, state: DesignState, /) -> Array:
        if self.face_rules is not None:
            volumes, _ = self._face_integrals(state)
            topology = self.model.topology
            return sum(
                (
                    sign * volumes[face]
                    for faces, signs in zip(
                        topology.solid_faces,
                        topology.solid_face_orientations,
                        strict=True,
                    )
                    for face, sign in zip(faces, signs, strict=True)
                ),
                jnp.zeros((), dtype=jnp.float64),
            )
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
        if self.face_rules is not None:
            return jnp.sum(self._face_integrals(state)[1])
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
