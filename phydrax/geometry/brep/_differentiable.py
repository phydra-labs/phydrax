#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import pi

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
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from ...units import ONE, RADIAN, UnitDefinition
from .._atlas import BoundaryAtlas, TrimDomain
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
    GeometrySource,
)
from .._interval_enclosure import (
    _device_multiply,
    _device_round_out,
    device_interval_evaluation_error,
)
from .._sampling import (
    bounded_rejection_sample,
    RejectionSamplingPlan,
    sample_boundary_atlas,
    SamplingResult,
)
from ..design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ..simplicial import MeshQueryResult, TriangleBVH, TriangleMesh
from ..simplicial._mesh import closest_points_on_triangles
from ._coordinate_request import coordinate_source_request
from ._deformation import TaperedCylinderPatch
from ._linked_parameters import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
    linked_curve_interval_box,
    linked_triangle_parameters,
    linked_trim_seam_evidence,
    LinkedBRepQuadrature,
    LinkedTrimDomains,
    validate_linked_geometry,
)
from ._measure import trimmed_face_rules
from ._model import (
    BRepBoundaryMap,
    BRepGeometry,
    BRepModel,
    BRepPCurve,
    BRepPlacedBoundaryMap,
)
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BernsteinSurfacePiece,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    device_bspline_surface_box,
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
    BRepQueryPolicy,
    jadd,
    jvector,
    prepare_brep_query,
    PreparedBRepQuery,
)
from ._query_types import EdgeData, FaceData
from ._request_preparations import TrimRequestPreparations
from ._source import BRepGeometryKernel, require_closed_solid


@dataclass(frozen=True, slots=True)
class BRepParameterLink:
    """Assign one face field to a shared stable design parameter."""

    face_index: int
    field: str
    parameter_id: ParameterId

    def __post_init__(self) -> None:
        if self.face_index < 0:
            raise ValueError(
                "BRepParameterLink.face_index must be non-negative; "
                f"got {self.face_index}."
            )
        if not self.field:
            raise ValueError(
                f"BRepParameterLink.field must be non-empty; got {self.field!r}."
            )


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
    """Diagnostic native boundary realization at one fixed-incidence state.

    ``seam_residual <= seam_bound`` with finite values is the F-041 seam
    evaluation evidence. Numerical consumers refuse incompatible linked joins;
    this diagnostic record remains available to inspect a refused state.
    """

    patches: tuple[AbstractSurfacePatch, ...]
    vertices: Array
    faces: Array
    atlas: BoundaryAtlas
    seam_residual: Array
    seam_bound: Array
    geometry: BRepGeometry | None
    parameter_bounds: Array

    @checked
    def __init__(
        self,
        *,
        patches: Iterable[AbstractSurfacePatch],
        vertices: ArrayLike,
        faces: ArrayLike,
        atlas: BoundaryAtlas,
        seam_residual: ArrayLike,
        seam_bound: ArrayLike,
        geometry: BRepGeometry | None = None,
        parameter_bounds: ArrayLike,
    ) -> None:
        self.patches = tuple(patches)
        self.vertices = jnp.asarray(vertices, dtype=jnp.float64)
        self.faces = jnp.asarray(faces, dtype=jnp.int32)
        self.atlas = atlas
        self.seam_residual = jnp.asarray(seam_residual, dtype=jnp.float64).reshape(())
        self.seam_bound = jnp.asarray(seam_bound, dtype=jnp.float64).reshape(())
        self.geometry = geometry
        self.parameter_bounds = jnp.asarray(parameter_bounds, dtype=jnp.float64)


def _corner_weld_weights(model: BRepModel) -> np.ndarray:
    mesh_faces, triangle_face_ids = jax.device_get(
        (model.mesh_faces, model.triangle_face_ids)
    )
    vertex_indices = np.asarray(mesh_faces, dtype=np.int32).reshape((-1,))
    face_indices = np.repeat(np.asarray(triangle_face_ids, dtype=np.int32), 3)
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
    boundary_map: BRepBoundaryMap | None = None,
) -> Array:
    chart_indices = jnp.repeat(model.triangle_face_ids, 3)
    parameters = model.triangle_parameters.reshape((-1, 2))
    bounds = model.parameter_bounds[chart_indices]
    reference = (parameters - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    if boundary_map is None:
        boundary_map = BRepBoundaryMap(patches, model.parameter_bounds)
    points = boundary_map.map(chart_indices, reference)
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


@checked
def evaluate_fixed_topology_mesh(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    /,
    *,
    corner_weights: Array | None = None,
    boundary_map: BRepBoundaryMap | None = None,
) -> tuple[Array, Array]:
    """Evaluate CAD faces, then weld shared mesh vertices by face-balanced averaging."""

    if len(patches) != len(model.patches):
        raise ValueError(
            "evaluate_fixed_topology_mesh.patches must preserve the imported "
            f"face count and ordering; got {len(patches)} faces, expected {len(model.patches)}."
        )
    weights = (
        jnp.asarray(_corner_weld_weights(model), dtype=jnp.float64)
        if corner_weights is None
        else jnp.asarray(corner_weights, dtype=jnp.float64).reshape((-1,))
    )
    corners = _evaluate_corners(patches, model, boundary_map)
    vertex_indices = model.mesh_faces.reshape((-1,))
    if weights.shape != (vertex_indices.shape[0],):
        raise ValueError(
            "evaluate_fixed_topology_mesh.corner_weights must contain one entry "
            f"per triangle corner; got shape {weights.shape}, expected {(vertex_indices.shape[0],)}."
        )
    vertices = (
        jnp.zeros_like(model.mesh_vertices)
        .at[vertex_indices]
        .add(corners * weights[:, None])
    )
    return vertices, model.mesh_faces


def _seam_residual(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    boundary_map: BRepBoundaryMap | None = None,
) -> Array:
    corners = _evaluate_corners(patches, model, boundary_map)
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


class BRepSeamError(RuntimeError):
    """Refuse a realized seam whose residual exceeds certified evaluation error."""

    def __init__(self, residual: float, bound: float, /) -> None:
        self.residual = residual
        self.bound = bound
        super().__init__(
            "FixedTopologyBRepSource.seam_residual must have a finite certified "
            "evaluation-error bound and not exceed it; "
            f"got residual {residual!r}, bound {bound!r}."
        )


class _SeamEvidence(StrictModule):
    """Report the sampled seam residual and its evaluation-error allowance."""

    residual: Array
    bound: Array


class _SeamSampleGroup(StrictModule):
    face: int = eqx.field(static=True)
    occurrence: int = eqx.field(static=True)
    corners: tuple[int, ...] = eqx.field(static=True)
    source_quantized: bool = eqx.field(static=True)


def _seam_sample_groups(model: BRepModel, /) -> tuple[_SeamSampleGroup, ...]:
    faces, occurrences, mesh_faces, restriction_vertices = jax.device_get(
        (
            model.triangle_face_ids,
            model.triangle_occurrence_ids,
            model.mesh_faces,
            model.mesh_chart_restriction_vertices,
        )
    )
    rounded_vertices = frozenset(np.asarray(restriction_vertices).tolist())
    groups: dict[tuple[int, int, bool], list[int]] = {}
    for triangle, (face, occurrence, vertices) in enumerate(
        zip(
            np.asarray(faces).tolist(),
            np.asarray(occurrences).tolist(),
            np.asarray(mesh_faces).tolist(),
            strict=True,
        )
    ):
        for corner, vertex in enumerate(vertices):
            key = (face, occurrence, vertex in rounded_vertices)
            groups.setdefault(key, []).append(3 * triangle + corner)
    return tuple(
        _SeamSampleGroup(
            face=face,
            occurrence=occurrence,
            corners=tuple(corners),
            source_quantized=source_quantized,
        )
        for (face, occurrence, source_quantized), corners in sorted(groups.items())
    )


def _place_seam_source_box(
    box: tuple[Array, Array], rotation: Array, translation: Array, /
) -> tuple[Array, Array]:
    """Transport a source box through the authored affine operation tree."""
    matrix = jax.lax.stop_gradient(rotation)
    shift = jnp.broadcast_to(jax.lax.stop_gradient(translation), box[0].shape)
    lower, upper = shift, shift
    for axis in range(3):
        coefficient = matrix[:, axis].reshape((1,) * (box[0].ndim - 1) + (3,))
        term = _device_multiply(
            (coefficient, coefficient),
            (box[0][..., axis, None], box[1][..., axis, None]),
        )
        lower, upper = _device_round_out(lower + term[0], upper + term[1])
    return lower, upper


def _spline_seam_source_box(
    patch: AbstractSurfacePatch, parameters: Array, source_quantized: bool, /
) -> tuple[Array, Array] | None:
    """Enclose spline source images without interpreting their basis recursion."""
    match patch:
        case BSplineSurfacePatch():
            points = jax.lax.stop_gradient(parameters)
            lower, upper = (
                (jnp.nextafter(points, -jnp.inf), jnp.nextafter(points, jnp.inf))
                if source_quantized
                else (points, points)
            )
            return device_bspline_surface_box(patch, lower, upper)
        case PlacedSurface():
            box = _spline_seam_source_box(patch.definition, parameters, source_quantized)
            if box is None:
                return None
            return _place_seam_source_box(box, patch.rotation, patch.translation)
        case _:
            return None


def _seam_evaluation_errors(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    groups: tuple[_SeamSampleGroup, ...],
    corners: Array,
    /,
) -> Array:
    parameters = model.triangle_parameters.reshape((-1, 2))
    errors = jnp.zeros((parameters.shape[0], 3), dtype=jnp.float64)
    geometry = model.require_geometry()
    for group in groups:
        patch = patches[group.face]
        bounds = model.parameter_bounds[group.face]
        indices = jnp.asarray(group.corners, dtype=jnp.int32)
        source_box = _spline_seam_source_box(
            patch, parameters[indices], group.source_quantized
        )
        if source_box is not None:
            if group.occurrence >= 0:
                occurrence = geometry.occurrences[group.occurrence]
                source_box = _place_seam_source_box(
                    source_box,
                    jnp.asarray(occurrence.rotation, dtype=jnp.float64),
                    jnp.asarray(occurrence.translation, dtype=jnp.float64),
                )
            # The restricted homogeneous box encloses the mathematical source;
            # comparing the already realized corners also covers normalization,
            # surface evaluation, and occurrence-placement rounding.
            actual = jax.lax.stop_gradient(corners[indices])
            difference = _device_round_out(source_box[0] - actual, source_box[1] - actual)
            radius = jnp.maximum(jnp.abs(difference[0]), jnp.abs(difference[1]))
            errors = errors.at[indices].set(jax.lax.stop_gradient(radius))
            continue

        def evaluate(parameter: Array) -> Array:
            reference = (parameter - bounds[0]) / (bounds[1] - bounds[0])
            point = patch.evaluate(bounds[0] + (bounds[1] - bounds[0]) * reference)
            if group.occurrence >= 0:
                occurrence = geometry.occurrences[group.occurrence]
                rotation = jnp.asarray(occurrence.rotation, dtype=jnp.float64)
                translation = jnp.asarray(occurrence.translation, dtype=jnp.float64)
                point = rotation @ point + translation
            return point

        def error(parameter: Array) -> Array:
            parameter = jax.lax.stop_gradient(parameter)
            input_bounds = None
            if group.source_quantized:
                # ExactRationalChartRestriction requires the nearest binary64
                # image of each source UV. Its adjacent floats enclose the
                # rational value without introducing a geometric tolerance.
                input_bounds = (
                    jnp.nextafter(parameter, -jnp.inf),
                    jnp.nextafter(parameter, jnp.inf),
                )
            return device_interval_evaluation_error(
                evaluate, parameter, input_bounds=input_bounds
            )

        errors = errors.at[indices].set(jax.vmap(error)(parameters[indices]))
    return errors


@eqx.filter_jit
def _seam_evidence(
    patches: tuple[AbstractSurfacePatch, ...],
    model: BRepModel,
    groups: tuple[_SeamSampleGroup, ...],
    boundary_map: BRepBoundaryMap,
    /,
) -> _SeamEvidence:
    corners = _evaluate_corners(patches, model, boundary_map)
    vertex_indices = model.mesh_faces.reshape((-1,))
    reference = jnp.zeros_like(model.mesh_vertices).at[vertex_indices].add(corners)
    counts = jnp.zeros((model.mesh_vertices.shape[0],), dtype=jnp.float64)
    counts = counts.at[vertex_indices].add(1.0)
    reference = reference / counts[:, None]
    difference = corners - reference[vertex_indices]
    magnitude = jnp.max(jnp.abs(difference), axis=-1)
    normalized = difference / jnp.where(magnitude > 0.0, magnitude, 1.0)[:, None]
    residual = jnp.max(
        magnitude * jnp.sqrt(jnp.sum(normalized * normalized, axis=-1)), initial=0.0
    )
    errors = _seam_evaluation_errors(patches, model, groups, corners)
    # Directed rounding belongs to refusal evidence, not the represented map.
    # Freeze its inputs before nextafter; freezing its result is too late for AD.
    maximum_error = jax.lax.stop_gradient(jnp.max(errors, axis=0, initial=0.0))
    maximum_coordinate = jax.lax.stop_gradient(
        jnp.max(jnp.abs(corners), axis=0, initial=0.0)
    )
    # Higham's gamma_n bound covers arbitrary scatter-add order, division,
    # subtraction, and the scaled norm. eps (rather than unit roundoff)
    # conservatively encloses construction of gamma itself. The L1 envelope
    # bounds the Euclidean residual without a model-unit scale floor.
    operations = corners.shape[0] + 8
    scaled_epsilon = operations * np.finfo(np.float64).eps
    gamma = np.nextafter(scaled_epsilon / (1.0 - scaled_epsilon), np.float64(np.inf))
    roundoff = jnp.nextafter(gamma * (2.0 * maximum_coordinate), jnp.inf)
    coordinate_bound = jnp.nextafter(2.0 * maximum_error + roundoff, jnp.inf)
    bound = jnp.nextafter(coordinate_bound[0] + coordinate_bound[1], jnp.inf)
    bound = jnp.nextafter(bound + coordinate_bound[2], jnp.inf)
    return _SeamEvidence(residual=residual, bound=bound)


def _raise_seam_error(residual: Array, bound: Array, /) -> None:
    raise BRepSeamError(float(residual), float(bound))


def _require_compatible_seams(points: Array, evidence: _SeamEvidence, /) -> Array:
    def refuse(value: Array) -> Array:
        jax.debug.callback(
            _raise_seam_error, evidence.residual, evidence.bound, ordered=True
        )
        return value

    def accept(value: Array) -> Array:
        return value

    return jax.lax.cond(
        (evidence.residual > evidence.bound)
        | ~jnp.isfinite(evidence.residual)
        | ~jnp.isfinite(evidence.bound),
        refuse,
        accept,
        points,
    )


def _patch_parameter_unit(field: str, coordinate_unit: UnitDefinition) -> UnitDefinition:
    """Resolve the native surface field's scientific unit; unknown fields refuse."""
    name = field.rsplit(".", 1)[-1]
    if name in {
        "origin",
        "center",
        "radius",
        "reference_radius",
        "major_radius",
        "minor_radius",
        "control_points",
        "translation",
        "direction",
    }:
        return coordinate_unit
    if name in {
        "first_axis",
        "second_axis",
        "axis",
        "weights",
        "knots",
        "u_knots",
        "v_knots",
        "rotation",
    }:
        return ONE
    if name == "semi_angle":
        return RADIAN
    raise ValueError(
        f"FixedTopologyBRepSource.parameter_field must have a supported native unit; got {field!r}"
    )


class FixedTopologyBRepSource(GeometrySource):
    """Differentiable B-Rep source with immutable incidence and entity identity.

    By default, face coefficients retain the existing independent/shared
    parameter bindings and fixed trim domains. ``parameter_map`` instead owns
    one correlated design-state map for patches, native trim geometry, bounds,
    and optionally mesh UVs; measures, charts, and native queries all consume
    that same realization. Maps bind parameters with explicit physical units
    through the global collector and cannot change incidence or orientation.
    """

    model: BRepModel
    parameter_links: tuple[BRepParameterLink, ...] = eqx.field(static=True)
    trainable_fields: frozenset[str] = eqx.field(static=True)
    parameter_map: AbstractBRepParameterMap | None

    @checked
    def __init__(
        self,
        model: BRepModel,
        *,
        parameter_links: Sequence[BRepParameterLink] = (),
        parameter_map: AbstractBRepParameterMap | None = None,
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
        require_closed_solid(model)
        links = tuple(parameter_links)
        if parameter_map is not None and links:
            raise ValueError(
                "FixedTopologyBRepSource.parameter_links must be empty when parameter_map owns bindings; got simultaneous coefficient links."
            )
        if parameter_map is not None:
            model.require_geometry()
        keys = tuple((link.face_index, link.field) for link in links)
        if len(set(keys)) != len(keys):
            raise ValueError(
                "FixedTopologyBRepSource.parameter_links must assign at most one "
                f"parameter link per face field; got {keys!r}."
            )
        if any(link.face_index >= len(model.patches) for link in links):
            raise ValueError(
                "FixedTopologyBRepSource.parameter_links must reference present faces; "
                f"got {[link.face_index for link in links]!r} for {len(model.patches)} faces."
            )
        self.model = model
        self.parameter_links = links
        self.trainable_fields = frozenset(trainable_fields)
        self.parameter_map = parameter_map

    @property
    def coordinate_contract(self) -> SpatialCoordinateContract:
        return self.model.coordinate_contract

    def _compile(self, context: ParameterCollector, /) -> GeometryKernel:
        if context.coordinate_unit != self.coordinate_contract.length_unit:
            raise ValueError(
                f"FixedTopologyBRepSource.coordinate_unit must exactly match its coordinate contract; got {context.coordinate_unit.symbol}"
            )
        with coordinate_source_request((self.model,)) as budget:
            if self.parameter_map is not None:
                bound_map = self.parameter_map.compile(context, self.model)
                return _FixedTopologyBRepKernel(
                    self.model,
                    (),
                    _corner_weld_weights(self.model),
                    None,
                    bound_map=bound_map,
                    linked_quadrature=LinkedBRepQuadrature(
                        self.model, coordinate_budget=budget
                    ),
                )
            links: Mapping[tuple[int, str], ParameterId] = {
                (link.face_index, link.field): link.parameter_id
                for link in self.parameter_links
            }
            patch_bindings: list[_PatchBinding] = []
            host_patches = jax.device_get(self.model.patches)
            for face_index, (patch, host_patch) in enumerate(
                zip(self.model.patches, host_patches, strict=True)
            ):
                path_leaves, tree_definition = jax.tree_util.tree_flatten_with_path(patch)
                host_leaves = jax.tree_util.tree_leaves(host_patch)
                bindings: list[ParameterBinding] = []
                for (path, value), host_value in zip(
                    path_leaves, host_leaves, strict=True
                ):
                    field = jax.tree_util.keystr(path).removeprefix(".")
                    parameter_id = links.get(
                        (face_index, field),
                        ParameterId(
                            f"{self.model.source_revision}:face:{face_index}", field
                        ),
                    )
                    value_host = np.asarray(host_value)
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
                            unit=_patch_parameter_unit(field, context.coordinate_unit),
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
                else trimmed_face_rules(
                    self.model, BRepQueryPolicy(), coordinate_budget=budget
                ),
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
    match curve:
        case LineCurve():
            return _affine_interval_box(curve.origin, ((curve.direction, (first, last)),))
        case CircleCurve() | EllipseCurve():
            cosine, sine = _trig_intervals(first, last)
            match curve:
                case CircleCurve():
                    a, b = curve.radius, curve.radius
                case EllipseCurve():
                    a, b = curve.first_radius, curve.second_radius
            return _affine_interval_box(
                curve.center,
                ((a * curve.first_axis, cosine), (b * curve.second_axis, sine)),
            )
        case BSplineCurve():
            lower, upper = (
                jnp.min(curve.control_points, axis=0),
                jnp.max(curve.control_points, axis=0),
            )
            return _affine_interval_box(
                lower, ((upper - lower, (jnp.asarray(0.0), jnp.asarray(1.0))),)
            )
        case _:
            raise TypeError(
                "_curve_interval_box.curve must provide a native interval-box capability; "
                f"got {type(curve).__name__}."
            )


def _surface_interval_box(patch: AbstractSurfacePatch, box: Array, /) -> Array:
    u, v = (box[0, 0], box[1, 0]), (box[0, 1], box[1, 1])
    match patch:
        case PlacedSurface():
            source = _surface_interval_box(patch.definition, box)
            image = jvector((patch.rotation, patch.rotation), (source[0], source[1]))
            return jnp.stack(jadd(image, (patch.translation, patch.translation)))
        case TaperedCylinderPatch():
            return patch.interval_box(box)
        case PlanePatch():
            return _affine_interval_box(
                patch.origin, ((patch.first_axis, u), (patch.second_axis, v))
            )
        case BernsteinSurfacePiece():
            low = eqx.error_if(
                patch.controls_lower,
                jnp.any(patch.controls_lower[..., 3] <= 0.0),
                "Native realized Bernstein denominator positivity is unresolved.",
            )
            high = patch.controls_upper
            coordinates = _interval_product(
                (low[..., :3], high[..., :3]),
                (1.0 / high[..., 3, None], 1.0 / low[..., 3, None]),
            )
            lower = jnp.min(coordinates[0], axis=(0, 1))
            upper = jnp.max(coordinates[1], axis=(0, 1))
            return _affine_interval_box(
                lower,
                ((upper - lower, (jnp.asarray(0.0), jnp.asarray(1.0))),),
            )
        case BSplineSurfacePatch():
            lower, upper = (
                jnp.min(patch.control_points, axis=(0, 1)),
                jnp.max(patch.control_points, axis=(0, 1)),
            )
            return _affine_interval_box(
                lower, ((upper - lower, (jnp.asarray(0.0), jnp.asarray(1.0))),)
            )
        case _:
            return _curved_surface_interval_box(patch, box)


def _curved_surface_interval_box(patch: AbstractSurfacePatch, box: Array, /) -> Array:
    """Extend native analytic and sweep carriers after control-hull dispatch."""
    u, v = (box[0, 0], box[1, 0]), (box[0, 1], box[1, 1])
    cosine_u, sine_u = _trig_intervals(*u)
    match patch:
        case CylinderPatch() | ConePatch():
            match patch:
                case CylinderPatch():
                    radius = (patch.radius, patch.radius)
                case ConePatch():
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
        case SpherePatch() | TorusPatch():
            cosine_v, sine_v = _trig_intervals(*v)
            match patch:
                case SpherePatch():
                    radius = (patch.radius * cosine_v[0], patch.radius * cosine_v[1])
                    height = (patch.radius * sine_v[0], patch.radius * sine_v[1])
                case TorusPatch():
                    radial = _interval_product(
                        (patch.minor_radius, patch.minor_radius), cosine_v
                    )
                    radius = (
                        patch.major_radius + radial[0],
                        patch.major_radius + radial[1],
                    )
                    height = _interval_product(
                        (patch.minor_radius, patch.minor_radius), sine_v
                    )
            return _affine_interval_box(
                patch.center,
                (
                    (patch.first_axis, _interval_product(radius, cosine_u)),
                    (patch.second_axis, _interval_product(radius, sine_u)),
                    (patch.axis, height),
                ),
            )
        case ExtrusionSurface():
            base = _curve_interval_box(patch.curve, box[:, 0])
            sweep = _affine_interval_box(
                jnp.zeros((3,), dtype=jnp.float64), ((patch.direction, v),)
            )
            return base + sweep
        case RevolutionSurface():
            base = _curve_interval_box(patch.curve, box[:, 1]) - patch.axis_origin
            radius = jnp.linalg.norm(jnp.maximum(jnp.abs(base[0]), jnp.abs(base[1])))
            return jnp.stack((patch.axis_origin - radius, patch.axis_origin + radius))
        case RuledSurface():
            first, second = (
                _curve_interval_box(patch.first, box[:, 0]),
                _curve_interval_box(patch.second, box[:, 0]),
            )
            left = _interval_product((1.0 - v[1], 1.0 - v[0]), (first[0], first[1]))
            right = _interval_product(v, (second[0], second[1]))
            return jnp.stack((left[0] + right[0], left[1] + right[1]))
        case _:
            raise TypeError(
                "_curved_surface_interval_box.patch must provide a native interval-box capability; "
                f"got {type(patch).__name__}."
            )


class _RealizedTrimEdge(AbstractCurve):
    patch: AbstractSurfacePatch
    pcurve: BRepPCurve

    @property
    def ambient_dimension(self) -> int:
        return 3

    def evaluate(self, parameters: Array, /) -> Array:
        return self.patch.evaluate(self.pcurve.evaluate(parameters))

    def bounding_box(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> np.ndarray:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.bounding_box(
                    first, last, preparations=preparations, budget=budget
                )
        return self.patch.bounding_box(
            self.pcurve.bounding_box(
                first, last, preparations=preparations, budget=budget
            ),
            preparations=preparations,
            budget=budget,
        )


def _realized_face(face: FaceData, patch: AbstractSurfacePatch, /) -> FaceData:
    center = jnp.zeros((3,), dtype=jnp.float64)
    projection = jnp.zeros((3, 3), dtype=jnp.float64)
    radii = jnp.zeros((2,), dtype=jnp.float64)
    match patch:
        case PlanePatch():
            center = patch.origin
            normal = jnp.cross(patch.first_axis, patch.second_axis)
            normal = normal / jnp.linalg.norm(normal)
            projection = jnp.outer(normal, normal)
        case SpherePatch() | CylinderPatch():
            match patch:
                case SpherePatch():
                    center = patch.center
                    projection = jnp.eye(3, dtype=jnp.float64)
                    frame = jnp.stack(
                        (patch.first_axis, patch.second_axis, patch.axis), axis=1
                    )
                case CylinderPatch():
                    center = patch.origin
                    axis = patch.axis / jnp.linalg.norm(patch.axis)
                    projection = jnp.eye(3, dtype=jnp.float64) - jnp.outer(axis, axis)
                    frame = projection @ jnp.stack(
                        (patch.first_axis, patch.second_axis), axis=1
                    )
            defect = jax.lax.stop_gradient(
                jnp.linalg.norm(
                    frame.T @ frame - jnp.eye(frame.shape[1], dtype=jnp.float64)
                )
            )
            margin = 256.0 * jnp.finfo(jnp.float64).eps
            radii = jnp.abs(patch.radius) * jnp.stack(
                (
                    jnp.sqrt(jnp.maximum(0.0, 1.0 - defect - margin)),
                    jnp.sqrt(1.0 + defect + margin),
                )
            )
        case _:
            pass

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
    match patch:
        case PlacedSurface():
            return eqx.tree_at(lambda value: value.definition, patch, definition)
        case _:
            raise TypeError(
                "_realized_placed_patch.patch must retain its source pose; "
                f"got {type(patch).__name__}."
            )


def _linked_face_template(
    face: FaceData,
    model: BRepModel,
    original: BRepModel,
    index: int,
    domain: TrimDomain,
    /,
) -> FaceData:
    old, new = original.parameter_bounds[index], model.parameter_bounds[index]

    def transport(parameters: Array) -> Array:
        return new[0] + (parameters - old[0]) * (new[1] - new[0]) / (old[1] - old[0])

    geometry = model.require_geometry()
    coedges = tuple(coedge for loop in geometry.face_loops[index] for coedge in loop)
    return FaceData(
        patch=model.patches[index],
        seed_uv=transport(face.seed_uv),
        seed_points=face.seed_points,
        trim_domain=domain,
        interval_boxes=face.interval_boxes,
        discovery_bvh=face.discovery_bvh,
        parameter_boxes=transport(face.parameter_boxes),
        pole_points=face.pole_points,
        pole_parameters=transport(face.pole_parameters),
        pcurves=tuple(geometry.pcurves[coedge] for coedge in coedges),
        carrier_center=face.carrier_center,
        carrier_projection=face.carrier_projection,
        carrier_radii=face.carrier_radii,
        lower=new[0],
        upper=new[1],
        periods=face.periods,
        orientation=face.orientation,
        coedge_edges=face.coedge_edges,
        seam_coedges=face.seam_coedges,
        full_rectangle=False,
    )


def _realized_edge(
    edge: EdgeData, face: FaceData, edge_index: int, limits: Array | None, /
) -> EdgeData:
    curve = _RealizedTrimEdge(
        face.patch, face.pcurves[face.coedge_edges.index(edge_index)]
    )
    first = edge.first if limits is None else limits[0]
    last = edge.last if limits is None else limits[1]
    fractions = (edge.seed_parameters - edge.first) / (edge.last - edge.first)
    parameters = first + fractions * (last - first)
    boxes = jnp.broadcast_to(
        jnp.stack(
            (
                jnp.full((3,), -jnp.inf, dtype=jnp.float64),
                jnp.full((3,), jnp.inf, dtype=jnp.float64),
            )
        ),
        edge.interval_boxes.shape,
    )
    return EdgeData(
        curve=None if edge.curve is None else curve,
        start_point=curve.evaluate(jnp.asarray(first, dtype=jnp.float64)),
        end_point=curve.evaluate(jnp.asarray(last, dtype=jnp.float64)),
        seed_parameters=parameters,
        seed_points=curve.evaluate(parameters),
        interval_boxes=boxes,
        discovery_bvh=refit_packed_bvh_bounds(
            edge.discovery_bvh, boxes[:, 0], boxes[:, 1]
        ),
        first=first,
        last=last,
        closed=edge.closed,
    )


def _linked_world_model(
    template: PreparedBRepQuery,
    model: BRepModel,
    patches: tuple[AbstractSurfacePatch, ...],
    /,
) -> BRepModel:
    world = template.world_query
    if world is None:
        raise RuntimeError(
            "FixedTopologyBRepSource.world_query must be prepared for placed geometry; got None."
        )
    source_geometry = model.require_geometry()
    geometry = world.model.require_geometry()
    pcurves = list(geometry.pcurves)
    for target, source in enumerate(template.world_face_sources):
        target_coedges = tuple(
            coedge for loop in geometry.face_loops[target] for coedge in loop
        )
        source_coedges = tuple(
            coedge for loop in source_geometry.face_loops[source] for coedge in loop
        )
        for target_coedge, source_coedge in zip(
            target_coedges, source_coedges, strict=True
        ):
            pcurves[target_coedge] = source_geometry.pcurves[source_coedge]
    ranges = source_geometry.edge_ranges[
        jnp.asarray(template.world_edge_sources, dtype=jnp.int32)
    ]
    vertices = jnp.zeros_like(geometry.vertex_points)
    curves = list(geometry.curves)
    for edge, (start, end) in enumerate(geometry.edge_vertices):
        face = world.model.topology.edge_faces[edge][0]
        coedge = next(
            coedge
            for loop in geometry.face_loops[face]
            for coedge in loop
            if geometry.coedge_edges[coedge] == edge
        )
        curve = _RealizedTrimEdge(patches[face], pcurves[coedge])
        points = curve.evaluate(ranges[edge])
        vertices = vertices.at[start].set(points[0]).at[end].set(points[1])
        curve_index = geometry.edge_curves[edge]
        if curve_index >= 0:
            curves[curve_index] = curve
    geometry = eqx.tree_at(
        lambda value: (
            value.pcurves,
            value.edge_ranges,
            value.vertex_points,
            value.curves,
        ),
        geometry,
        (tuple(pcurves), ranges, vertices, tuple(curves)),
    )
    return eqx.tree_at(
        lambda value: (value.patches, value.geometry, value.parameter_bounds),
        world.model,
        (
            patches,
            geometry,
            model.parameter_bounds[
                jnp.asarray(template.world_face_sources, dtype=jnp.int32)
            ],
        ),
    )


def _realized_native_query(
    template: PreparedBRepQuery,
    patches: tuple[AbstractSurfacePatch, ...],
    rules: tuple[tuple[Array, Array], ...],
    /,
    *,
    model: BRepModel | None = None,
    domains: tuple[TrimDomain, ...] | None = None,
) -> PreparedBRepQuery:
    """Refresh numeric source leaves without host preparation during tracing."""
    realized_model = template.model if model is None else model
    templates = template.faces
    if domains is not None:
        templates = tuple(
            _linked_face_template(face, realized_model, template.model, index, domain)
            for index, (face, domain) in enumerate(zip(templates, domains, strict=True))
        )
    faces = tuple(
        _realized_face(face, patch)
        for face, patch in zip(templates, patches, strict=True)
    )
    geometry = realized_model.require_geometry()
    edges = tuple(
        _realized_edge(
            edge,
            faces[realized_model.topology.edge_faces[index][0]],
            index,
            None if domains is None else geometry.edge_ranges[index],
        )
        for index, edge in enumerate(template.edges)
    )
    face_boxes = jnp.stack(
        [
            _surface_interval_box(patch, realized_model.parameter_bounds[index])
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
        lambda value: (
            value.points,
            value.vector_areas,
            value.areas,
            value.parameters,
            value.weights,
            value.faces,
        ),
        template.quadrature,
        (
            jnp.concatenate(nodes),
            jnp.concatenate(vector_areas),
            jnp.concatenate(areas),
            jnp.concatenate(tuple(rule[0] for rule in rules)),
            jnp.concatenate(tuple(rule[1] for rule in rules)),
            jnp.concatenate(
                tuple(
                    jnp.full((rule[1].size,), index, dtype=jnp.int32)
                    for index, rule in enumerate(rules)
                )
            ),
        ),
    )
    result = eqx.tree_at(
        lambda value: (
            value.faces,
            value.edges,
            value.face_boxes,
            value.face_bvh,
            value.quadrature,
            value.model,
            value.vertex_points,
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
            realized_model,
            geometry.vertex_points,
        ),
    )
    return result


class _FixedTopologyBRepKernel(BRepGeometryKernel):
    """Realize fixed-epoch patches and integrate measures over their trims.

    Coefficient-only sources retain the epoch's fixed ``face_rules``. Linked
    sources rebuild Green-reduced rules and native source-arc covers from their
    correlated trims and bounds, preserving the fixed numerical layout.
    """

    model: BRepModel
    patch_bindings: tuple[_PatchBinding, ...] = eqx.field(static=True)
    corner_weights: Array
    face_rules: tuple[tuple[Array, Array], ...] | None
    native_query: PreparedBRepQuery | None
    reference_atlas: BoundaryAtlas
    query_bvh: TriangleBVH | None
    seam_groups: tuple[_SeamSampleGroup, ...] = eqx.field(static=True)
    bound_map: AbstractBoundBRepParameterMap | None
    linked_quadrature: LinkedBRepQuadrature | None
    linked_domains: LinkedTrimDomains | None
    atlas_face_sources: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        model: BRepModel,
        patch_bindings: Iterable[_PatchBinding],
        corner_weights: ArrayLike,
        face_rules: tuple[tuple[Array, Array], ...] | None,
        *,
        bound_map: AbstractBoundBRepParameterMap | None = None,
        linked_quadrature: LinkedBRepQuadrature | None = None,
    ) -> None:
        self.model = model
        self.patch_bindings = tuple(patch_bindings)
        self.corner_weights = jnp.asarray(corner_weights, dtype=jnp.float64)
        self.face_rules = face_rules
        self.bound_map = bound_map
        self.linked_quadrature = linked_quadrature
        self.seam_groups = _seam_sample_groups(model)
        self.native_query = (
            None
            if model.geometry is None
            else prepare_brep_query(model, certify_on_host=False)
        )
        self.reference_atlas = model.boundary_atlas
        self.linked_domains = (
            LinkedTrimDomains(
                tuple(face.trim_domain for face in self.native_query.faces),
                model.require_geometry(),
            )
            if bound_map is not None and self.native_query is not None
            else None
        )
        mapping = self.reference_atlas.mapping
        self.atlas_face_sources = (
            tuple(np.asarray(jax.device_get(mapping.definition_faces)).tolist())
            if isinstance(mapping, BRepPlacedBoundaryMap)
            else tuple(range(len(model.patches)))
        )
        self.query_bvh = (
            TriangleBVH(
                TriangleMesh(
                    model.mesh_vertices, model.mesh_faces, source_id=model.model_id
                )
            )
            if model.geometry is None
            else None
        )

    @property
    def capabilities(self) -> frozenset[GeometryCapability]:
        return super().capabilities | {GeometryCapability.SEAM_DIAGNOSTICS}

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

    def _state_model(self, state: DesignState, /) -> BRepModel:
        if self.bound_map is None:
            patches = tuple(binding.realize(state) for binding in self.patch_bindings)
            return eqx.tree_at(lambda model: model.patches, self.model, patches)
        value = validate_linked_geometry(self.bound_map.realize(state), self.model)
        parameters = linked_triangle_parameters(value, self.model)
        return eqx.tree_at(
            lambda model: (
                model.patches,
                model.geometry,
                model.parameter_bounds,
                model.triangle_parameters,
            ),
            self.model,
            (value.patches, value.geometry, value.parameter_bounds, parameters),
        )

    def _state_rules(self, model: BRepModel, /) -> tuple[tuple[Array, Array], ...]:
        if self.linked_quadrature is not None:
            return self.linked_quadrature.realize(
                BRepParameterGeometry(
                    patches=model.patches,
                    geometry=model.require_geometry(),
                    parameter_bounds=model.parameter_bounds,
                )
            )
        if self.face_rules is None:
            raise RuntimeError(
                "FixedTopologyBRepSource.face_rules must contain native trimmed quadrature; got None."
            )
        return self.face_rules

    def _state_evidence(self, model: BRepModel, /) -> _SeamEvidence:
        evidence = _seam_evidence(
            model.patches,
            model,
            self.seam_groups,
            self._corner_map(model.patches, model.parameter_bounds),
        )
        if self.bound_map is not None:
            residual, bound = linked_trim_seam_evidence(
                BRepParameterGeometry(
                    patches=model.patches,
                    geometry=model.require_geometry(),
                    parameter_bounds=model.parameter_bounds,
                ),
                linked_curve_interval_box,
            )
            evidence = _SeamEvidence(
                residual=jnp.maximum(evidence.residual, residual),
                bound=jnp.maximum(evidence.bound, bound),
            )
        return evidence

    def _validated_model(self, state: DesignState, /) -> BRepModel:
        model = self._state_model(state)
        if self.bound_map is not None:
            bounds = _require_compatible_seams(
                model.parameter_bounds, self._state_evidence(model)
            )
            model = eqx.tree_at(lambda value: value.parameter_bounds, model, bounds)
        return model

    def _corner_map(
        self,
        patches: tuple[AbstractSurfacePatch, ...],
        parameter_bounds: Array | None = None,
        /,
    ) -> BRepBoundaryMap:
        mapping = self.reference_atlas.mapping
        match mapping:
            case BRepPlacedBoundaryMap():
                base = mapping.base
            case BRepBoundaryMap():
                base = mapping
            case _:
                raise TypeError(
                    "_FixedTopologyBRepKernel.reference_atlas.mapping must be a native chart map; "
                    f"got {type(mapping).__name__}."
                )
        return eqx.tree_at(
            lambda value: (value.patches, value.parameter_bounds),
            base,
            (
                patches,
                base.parameter_bounds if parameter_bounds is None else parameter_bounds,
            ),
        )

    def _native_query(self, state: DesignState, /) -> PreparedBRepQuery:
        template = self.native_query
        if template is None:
            raise RuntimeError(
                "_FixedTopologyBRepKernel.native_query must contain a native source query; got None."
            )
        model = self._validated_model(state)
        patches = model.patches
        rules = self._state_rules(model)
        domains = None
        if self.linked_domains is not None:
            domains = self.linked_domains.realize(
                BRepParameterGeometry(
                    patches=patches,
                    geometry=model.require_geometry(),
                    parameter_bounds=model.parameter_bounds,
                ),
                linked_curve_interval_box,
            )
        result = _realized_native_query(
            template, patches, rules, model=model, domains=domains
        )
        if template.world_query is not None:
            world_patches = tuple(
                _realized_placed_patch(patch, patches[source])
                for patch, source in zip(
                    template.world_query.model.patches,
                    template.world_face_sources,
                    strict=True,
                )
            )
            world_rules = tuple(rules[source] for source in template.world_face_sources)
            world_model = (
                None
                if domains is None
                else _linked_world_model(template, model, world_patches)
            )
            world_domains = (
                None
                if domains is None
                else tuple(domains[source] for source in template.world_face_sources)
            )
            world = _realized_native_query(
                template.world_query,
                world_patches,
                world_rules,
                model=world_model,
                domains=world_domains,
            )
            result = eqx.tree_at(lambda value: value.world_query, result, world)
        return result

    def _native_membership(
        self,
        query: PreparedBRepQuery,
        points: Array,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> tuple[Array, Array]:
        inside, unresolved = [], []
        for occurrence in query.occurrences:
            result = query.contains(
                points,
                path=occurrence.path,
                preparations=preparations,
                coordinate_budget=budget,
            )
            inside.append(result.inside)
            unresolved.append(
                result.unresolved & (result.distances > query.tolerances.classifier)
            )
        states, pending = jnp.stack(inside), jnp.stack(unresolved)
        certified_inside = jnp.any(states & ~pending, axis=0)
        return jnp.any(states, axis=0), jnp.any(pending, axis=0) & ~certified_inside

    def realize(self, state: DesignState, /) -> FixedTopologyBRepRealization:
        model = self._state_model(state)
        patches = model.patches
        corner_map = self._corner_map(patches, model.parameter_bounds)
        evidence = self._state_evidence(model)
        vertices, faces = evaluate_fixed_topology_mesh(
            patches,
            model,
            corner_weights=self.corner_weights,
            boundary_map=corner_map,
        )
        mapping = self.reference_atlas.mapping
        match mapping:
            case BRepPlacedBoundaryMap():
                mapping = eqx.tree_at(lambda value: value.base, mapping, corner_map)
            case BRepBoundaryMap():
                mapping = corner_map
            case _:
                raise TypeError(
                    "_FixedTopologyBRepKernel.reference_atlas.mapping must be a native chart map; "
                    f"got {type(mapping).__name__}."
                )
        atlas = eqx.tree_at(lambda value: value.mapping, self.reference_atlas, mapping)
        if self.linked_domains is not None:
            domains = self.linked_domains.realize(
                BRepParameterGeometry(
                    patches=patches,
                    geometry=model.require_geometry(),
                    parameter_bounds=model.parameter_bounds,
                ),
                linked_curve_interval_box,
                normalized=True,
            )
            atlas = eqx.tree_at(
                lambda value: value.trim_domains,
                atlas,
                tuple(domains[face] for face in self.atlas_face_sources),
            )
        return FixedTopologyBRepRealization(
            patches=patches,
            vertices=vertices,
            faces=faces,
            atlas=atlas,
            seam_residual=(
                evidence.residual
                if self.bound_map is not None
                else _seam_residual(patches, model, corner_map)
            ),
            seam_bound=evidence.bound,
            geometry=model.geometry,
            parameter_bounds=model.parameter_bounds,
        )

    def seam_residual(self, state: DesignState, /) -> Array:
        return self.realize(state).seam_residual

    def _triangles(self, state: DesignState) -> Array:
        realization = self.realize(state)
        return realization.vertices[realization.faces]

    def _nearest_query(
        self, state: DesignState, points: Array, *, k: int
    ) -> tuple[MeshQueryResult, Array, Array]:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        if self.query_bvh is None:
            raise RuntimeError(
                f"_FixedTopologyBRepKernel.query_bvh must be a prepared triangle BVH; got {self.query_bvh!r}."
            )
        realization = self.realize(state)
        tree = self.query_bvh.refit(realization.vertices)
        selected = jax.lax.stop_gradient(tree.nearest_faces(flat, k=k).items)
        selected_triangles = realization.vertices[realization.faces[selected]]
        closest_by_selected = jax.vmap(closest_points_on_triangles)(
            flat, selected_triangles
        )
        distance_squared = jnp.sum((closest_by_selected - flat[:, None, :]) ** 2, axis=-1)
        closest = closest_by_selected[:, 0]
        triangle = selected_triangles[:, 0]
        normal = jnp.cross(
            triangle[:, 1] - triangle[:, 0], triangle[:, 2] - triangle[:, 0]
        )
        normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
        query = MeshQueryResult(
            closest_point=closest.reshape((*leading, 3)),
            distance=jnp.sqrt(distance_squared[:, 0]).reshape(leading),
            face_index=selected[:, 0].reshape(leading),
            normal=normal.reshape((*leading, 3)),
        )
        # Distances are differentiated only after discrete traversal selected IDs.
        return query, triangle, distance_squared

    def _query(
        self,
        state: DesignState,
        points: Array,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget,
    ) -> MeshQueryResult:
        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        if self.native_query is not None:
            result = self._native_query(state).closest_point(
                flat, preparations=preparations, coordinate_budget=budget
            )
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
        return self._nearest_query(state, points, k=1)[0]

    def contains(
        self,
        state: DesignState,
        points: Array,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> Array:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if coordinate_budget is None:
            with coordinate_source_request(
                (
                    self,
                    state,
                    points,
                )
            ) as coordinate_budget:
                return self.contains(
                    state,
                    points,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )
        points_ = jnp.asarray(points, dtype=jnp.float64)
        if self.native_query is not None:
            if self.bound_map is None:
                model = self._state_model(state)
                points_ = _require_compatible_seams(points_, self._state_evidence(model))
            leading = points_.shape[:-1]
            query = self._native_query(state)
            inside, unresolved = self._native_membership(
                query,
                points_.reshape((-1, 3)),
                preparations=preparations,
                budget=coordinate_budget,
            )
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

    def boundary_field(
        self,
        state: DesignState,
        points: Array,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Array:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    state,
                    points,
                )
            ) as budget:
                return self.boundary_field(
                    state, points, preparations=preparations, budget=budget
                )
        query = self._query(state, points, preparations=preparations, budget=budget)
        points_ = jnp.asarray(points, dtype=query.closest_point.dtype)
        difference = points_ - query.closest_point
        squared_distance = jnp.sum(difference * difference, axis=-1)
        away_from_boundary = squared_distance > 0.0
        distance = jnp.sqrt(jnp.where(away_from_boundary, squared_distance, 1.0))
        signed_distance = jnp.where(
            self.contains(
                state, points_, preparations=preparations, coordinate_budget=budget
            ),
            -distance,
            distance,
        )
        boundary_linearization = jnp.sum(difference * query.normal, axis=-1)
        return jnp.where(
            away_from_boundary,
            signed_distance,
            boundary_linearization,
        )

    def boundary_normal(
        self,
        state: DesignState,
        points: Array,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> Array:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    state,
                    points,
                )
            ) as budget:
                return self.boundary_normal(
                    state, points, preparations=preparations, budget=budget
                )
        return self._query(state, points, preparations=preparations, budget=budget).normal

    def closest_point(
        self,
        state: DesignState,
        points: Array,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        coordinate_budget: CoordinateEnclosureBudget | None = None,
    ) -> ClosestPointResult:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if coordinate_budget is None:
            with coordinate_source_request(
                (
                    self,
                    state,
                    points,
                )
            ) as coordinate_budget:
                return self.closest_point(
                    state,
                    points,
                    preparations=preparations,
                    coordinate_budget=coordinate_budget,
                )

        points_ = jnp.asarray(points, dtype=jnp.float64)
        leading = points_.shape[:-1]
        flat = points_.reshape((-1, 3))
        if self.native_query is not None:
            native = self._native_query(state)
            hit = native.closest_point(
                flat, preparations=preparations, coordinate_budget=coordinate_budget
            )
            inside, unresolved = self._native_membership(
                native, flat, preparations=preparations, budget=coordinate_budget
            )
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
        query, triangle, selected_distance_squared = self._nearest_query(state, flat, k=2)
        unique, regular, margin = triangle_query_evidence(
            flat,
            triangle,
            query.closest_point,
            selected_distance_squared[:, 1],
        )
        return represented_mesh_closest_point(
            points_,
            closest_point=query.closest_point.reshape((*leading, 3)),
            distance=query.distance.reshape(leading),
            normal=query.normal.reshape((*leading, 3)),
            source_entity_id=query.face_index.reshape(leading),
            inside=self.contains(
                state,
                points_,
                preparations=preparations,
                coordinate_budget=coordinate_budget,
            ),
            unique=unique.reshape(leading),
            regular=regular.reshape(leading),
            margin=margin.reshape(leading),
            represented_geometry_id=f"{self.model.model_id}:fixed-topology-query",
            physical_geometry_id=self.model.source_revision,
            exact_to_physical=False,
        )

    def bounds(self, state: DesignState, /) -> Array:
        if self.native_query is not None:
            return self._native_query(state).bounds
        vertices = self.realize(state).vertices
        return jnp.stack((jnp.min(vertices, axis=0), jnp.max(vertices, axis=0)))

    def _face_integrals(self, state: DesignState, /) -> tuple[Array, Array]:
        """Per-face oriented volume flux ``int S . n dA / 3`` and area."""
        model = self._validated_model(state)
        rules = self._state_rules(model)
        volumes, areas = [], []
        orientation = self.model.orientation
        for face, (patch, (parameters, weights)) in enumerate(
            zip(model.patches, rules, strict=True)
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
        if self.face_rules is not None or self.linked_quadrature is not None:
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
        if self.face_rules is not None or self.linked_quadrature is not None:
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
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> SamplingResult:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    state,
                    num_points,
                )
            ) as budget:
                return self.sample_interior(
                    state,
                    num_points,
                    key=key,
                    plan=plan,
                    preparations=preparations,
                    budget=budget,
                )
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
            lambda values: self.contains(
                state, values, preparations=preparations, coordinate_budget=budget
            ),
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
            self.boundary_atlas(state),
            num_points,
            key=key,
        )

    def boundary_atlas(self, state: DesignState, /) -> BoundaryAtlas:
        realization = self.realize(state)
        atlas = realization.atlas
        if self.bound_map is not None:
            mapping = atlas.mapping
            base = mapping.base if isinstance(mapping, BRepPlacedBoundaryMap) else mapping
            if not isinstance(base, BRepBoundaryMap):
                raise TypeError(
                    "FixedTopologyBRepSource.atlas_mapping must be native; got an unsupported chart map."
                )
            bounds = _require_compatible_seams(
                base.parameter_bounds,
                _SeamEvidence(
                    residual=realization.seam_residual, bound=realization.seam_bound
                ),
            )
            base = eqx.tree_at(lambda value: value.parameter_bounds, base, bounds)
            mapping = (
                eqx.tree_at(lambda value: value.base, mapping, base)
                if isinstance(mapping, BRepPlacedBoundaryMap)
                else base
            )
            atlas = eqx.tree_at(lambda value: value.mapping, atlas, mapping)
        return atlas


__all__ = [
    "AbstractBRepParameterMap",
    "AbstractBoundBRepParameterMap",
    "BRepParameterGeometry",
    "BRepParameterLink",
    "BRepSeamError",
    "FixedTopologyBRepRealization",
    "FixedTopologyBRepSource",
    "evaluate_fixed_topology_mesh",
]
