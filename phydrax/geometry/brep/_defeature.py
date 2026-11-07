#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local analytic feature removal with certified boundary and volume changes.

Plane-pair blends collapse onto their rational intersection line; normal
plane/cylinder torus blends collapse onto their intersection circle. Only
affected trims change. Publication recomputes endpoint and face-gap enclosures
from the native carriers before assembling a target model.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from math import isqrt, pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...typing import checked, parse
from .._atlas import TrimDomain
from .._cad_revision import AssociationGraph
from .._interval_enclosure import prepare_interval_function
from .._rational import (
    fraction_down,
    fraction_up,
    rational_cross,
    rational_dot,
    rational_vector,
)
from ..design._naming import RecipeEntityNames
from ._assembly import BRepAffinePlacement
from ._boolean_core import PeriodicPCurve
from ._carrier_pieces import child_coefficient_enclosures
from ._constructors import (
    _Builder as Builder,
    _line_arc_contact,
    _profile_edge,
    assemble_brep_model,
    BRepTessellationPolicy,
    ProfileArc,
    ProfileLine,
)
from ._correspondence import native_edge_pcurve
from ._edge_classification import brep_edge_classification
from ._exact import exact_frame_orientation, square_root_interval
from ._intersection import _exact_cylinder_axis, NativePeriodEndpoint, RootEndpoint
from ._intersection_curve import SurfaceRegion
from ._measure import brep_mass_properties
from ._model import BRepCurve, BRepEntityId, BRepGeometry, BRepModel
from ._offset_result import _association_graph_for_images, _entity_images
from ._offset_rounding import offset_rounding_evidence, vertex_rounding_bounds
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    TorusPatch,
)
from ._planar_arrangement import inside_loop, intersections, on_segment, Point
from ._planar_offset import _dot, _solve, planar_carrier
from ._source_correspondence import _affine_curve_coefficients


BRepDefeatureStatus: TypeAlias = Literal[
    "certified", "unsupported", "unresolved", "collision"
]
BRepDefeatureOperation: TypeAlias = Literal["fillet", "chamfer", "hole"]
BRepDefeatureVolumeKind: TypeAlias = Literal["exact", "enclosed", "absent"]
BRepDefeatureIntersectionKind: TypeAlias = Literal["none", "line", "circle"]
type _Vector3 = tuple[Fraction, Fraction, Fraction]
type _Vector = _Vector3


def _fraction_vector3(value: object, /) -> _Vector3 | None:
    if not isinstance(value, tuple) or len(value) != 3:
        return None
    first, second, third = value
    if not (
        isinstance(first, Fraction)
        and isinstance(second, Fraction)
        and isinstance(third, Fraction)
    ):
        return None
    return first, second, third


def _line_vector3(curve: AbstractCurve, /) -> tuple[_Vector3, _Vector3] | None:
    if isinstance(curve, LineCurve):
        origin = _fraction_vector3(curve.origin_exact)
        direction = _fraction_vector3(curve.direction_exact)
    elif isinstance(curve, ExactLineCurve):
        first = _fraction_vector3(curve.first_exact)
        last = _fraction_vector3(curve.last_exact)
        if first is None or last is None:
            return None
        origin = first
        direction = (
            last[0] - first[0],
            last[1] - first[1],
            last[2] - first[2],
        )
    else:
        return None
    if origin is None or direction is None:
        return None
    return origin, direction


def _vector3_affine(
    origin: _Vector3, parameter: Fraction, direction: _Vector3, /
) -> _Vector3:
    return (
        origin[0] + parameter * direction[0],
        origin[1] + parameter * direction[1],
        origin[2] + parameter * direction[2],
    )


def _vector3_scale(vector: _Vector3, factor: Fraction, /) -> _Vector3:
    return (factor * vector[0], factor * vector[1], factor * vector[2])


def _vector3_remove_projection(
    vector: _Vector3, axis: _Vector3, projection: Fraction, /
) -> _Vector3:
    return (
        vector[0] - projection * axis[0],
        vector[1] - projection * axis[1],
        vector[2] - projection * axis[2],
    )


def _native_endpoint(root: object | None, /) -> NativePeriodEndpoint | None:
    if root is None:
        return None
    if not isinstance(root, NativePeriodEndpoint):
        raise _Refusal(
            "brep_defeature.source must preserve native endpoint period identity; "
            "received an unsupported endpoint root."
        )
    return root


def _native_endpoint_pair(
    roots: tuple[object | None, object | None], /
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    return (_native_endpoint(roots[0]), _native_endpoint(roots[1]))


def _native_roots(
    geometry: BRepGeometry, edge: int, /
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    return _native_endpoint_pair(geometry.edge_endpoint_roots[edge])


type _Images = tuple[tuple[int, int], ...]
type _G1Evidence = tuple[tuple[BRepEntityId, ...], Array]
type _BlendRemoval = tuple[
    Builder,
    _Images,
    _Images,
    tuple[Fraction, Fraction, Fraction] | None,
    tuple[float, float],
    tuple[_Vector, ...],
    _G1Evidence,
]
type _HoleRemoval = tuple[
    Builder,
    _Images,
    _Images,
    tuple[Fraction, Fraction, Fraction] | None,
    tuple[float, float],
]


@dataclass(frozen=True, slots=True)
class _HoleWallData:
    origin: _Vector
    first_axis: _Vector
    second_axis: _Vector
    axis: _Vector
    reference_radius: Fraction
    height_range: tuple[Fraction, Fraction]
    slope: Fraction
    family: Literal["cylinder", "cone", "revolved-cylinder", "revolved-cone"]
    orientation_sign: int


type _Shells = tuple[
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
]


class _Refusal(ValueError):
    """A family lacks an extension or publication certificate."""


class _Collision(_Refusal):
    """A rational native intersection proves an extension collision."""


class _Unresolved(_Refusal):
    """The supported geometry or its continuous proof could not be decided."""


@final
class _TorusIntent(StrictModule):
    plane_face: int = eqx.field(static=True)
    cylinder_face: int = eqx.field(static=True)
    plane_edge: int = eqx.field(static=True)
    cylinder_edge: int = eqx.field(static=True)
    seam_edge: int = eqx.field(static=True)
    plane: PlanePatch
    cylinder: CylinderPatch
    center: _Vector = eqx.field(static=True)
    z: Fraction = eqx.field(static=True)
    radius: Fraction = eqx.field(static=True)
    minor_radius: Fraction = eqx.field(static=True)
    old_radius: Fraction = eqx.field(static=True)
    old_z: Fraction = eqx.field(static=True)
    phase_error: Fraction = eqx.field(static=True)


@final
class BRepDefeatureResult(StrictModule):
    """A complete native feature removal with exact entity correspondence."""

    status: BRepDefeatureStatus = eqx.field(static=True)
    reason: str = eqx.field(static=True)
    operation: BRepDefeatureOperation = eqx.field(static=True)
    volume_kind: BRepDefeatureVolumeKind = eqx.field(static=True)
    source_model_id: str = eqx.field(static=True)
    removed_face: BRepEntityId = eqx.field(static=True)
    volume_rational: Fraction | None = eqx.field(static=True)
    volume_pi: Fraction | None = eqx.field(static=True)
    volume_pi_squared: Fraction | None = eqx.field(static=True)
    face_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    edge_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    vertex_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    solid_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = eqx.field(static=True)
    created_entity_sources: tuple[tuple[BRepEntityId, tuple[BRepEntityId, ...]], ...] = (
        eqx.field(static=True)
    )
    association_graph: AssociationGraph | None = eqx.field(static=True)
    intersection_kind: BRepDefeatureIntersectionKind = eqx.field(static=True)
    carrier_intent_points: tuple[_Vector, ...] = eqx.field(static=True)
    g1_edges: tuple[BRepEntityId, ...] = eqx.field(static=True)
    model: BRepModel | None
    volume_change_enclosure: Array
    hausdorff_enclosure: Array
    intersection_rounding_bounds: Array
    g1_angle_bounds: Array
    recipe_names: RecipeEntityNames | None

    @checked
    def __init__(
        self,
        *,
        status: BRepDefeatureStatus,
        reason: str,
        operation: BRepDefeatureOperation,
        source_model_id: str,
        removed_face: BRepEntityId,
        model: BRepModel | None,
        volume_coefficients: tuple[Fraction, Fraction, Fraction] | None = None,
        volume_enclosure: tuple[float, float] | None = None,
        hausdorff_enclosure: tuple[float, float] = (0.0, 0.0),
        intersection_kind: BRepDefeatureIntersectionKind = "none",
        carrier_intent_points: tuple[_Vector, ...] = (),
        intersection_rounding_bounds: tuple[float, ...] = (),
        g1_angle_bounds: Array | None = None,
        g1_edges: tuple[BRepEntityId, ...] = (),
        face_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        edge_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        vertex_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        solid_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        created_entity_sources: tuple[
            tuple[BRepEntityId, tuple[BRepEntityId, ...]], ...
        ] = (),
        association_graph: AssociationGraph | None = None,
        recipe_names: RecipeEntityNames | None = None,
    ) -> None:
        self.status = parse(status, BRepDefeatureStatus, "BRepDefeatureResult.status")
        self.operation = parse(
            operation, BRepDefeatureOperation, "BRepDefeatureResult.operation"
        )
        if (self.status == "certified") != (
            model is not None
            and association_graph is not None
            and (volume_coefficients is not None or volume_enclosure is not None)
        ):
            raise ValueError(
                "BRepDefeatureResult.evidence must accompany exactly complete certified results."
            )
        if (
            not isinstance(removed_face, BRepEntityId)
            or removed_face.kind != "face"
            or removed_face.source_revision != source_model_id
        ):
            raise ValueError(
                "BRepDefeatureResult.removed_face must be an exact source face ID."
            )
        if association_graph is not None and (
            association_graph.source_revision.revision_id != source_model_id
            or model is None
            or association_graph.target_revision.revision_id != model.source_revision
        ):
            raise ValueError(
                "BRepDefeatureResult.association_graph must bind its exact source and target revisions."
            )
        for kind, pairs in (
            ("face", face_images),
            ("edge", edge_images),
            ("vertex", vertex_images),
            ("solid", solid_images),
        ):
            if any(
                not isinstance(source, BRepEntityId)
                or not isinstance(target, BRepEntityId)
                or source.kind != kind
                or target.kind != kind
                or source.source_revision != source_model_id
                or model is None
                or target.source_revision != model.source_revision
                for source, target in pairs
            ):
                raise ValueError(
                    f"BRepDefeatureResult.{kind}_images must contain exact source/target IDs."
                )
        mapped_targets = {
            target
            for pairs in (face_images, edge_images, vertex_images, solid_images)
            for _, target in pairs
        }
        if any(
            not isinstance(target, BRepEntityId)
            or target.source_revision != ("" if model is None else model.source_revision)
            or not sources
            or any(
                not isinstance(source, BRepEntityId)
                or source.source_revision != source_model_id
                for source in sources
            )
            or target in mapped_targets
            for target, sources in created_entity_sources
        ):
            raise ValueError(
                "BRepDefeatureResult.created_entity_sources must bind each new exact target ID to source IDs."
            )
        if model is None and created_entity_sources:
            raise ValueError(
                "A refused defeature result cannot publish target entity IDs."
            )
        if model is not None:
            target_entities = {
                entity
                for kind in ("solid", "face", "edge", "vertex")
                for entity in getattr(model, f"{kind}_ids")
            }
            created_entities = {target_id for target_id, _ in created_entity_sources}
            if (
                not created_entities <= target_entities
                or (mapped_targets | created_entities) != target_entities
            ):
                raise ValueError(
                    "BRepDefeatureResult must map or source every published target entity."
                )
        self.volume_kind = (
            "exact"
            if volume_coefficients is not None
            else "enclosed"
            if volume_enclosure is not None
            else "absent"
        )
        self.reason = reason
        self.source_model_id = source_model_id
        self.removed_face = removed_face
        self.model = model
        self.volume_rational = (
            None if volume_coefficients is None else volume_coefficients[0]
        )
        self.volume_pi = None if volume_coefficients is None else volume_coefficients[1]
        self.volume_pi_squared = (
            None if volume_coefficients is None else volume_coefficients[2]
        )
        self.face_images = face_images
        self.edge_images = edge_images
        self.vertex_images = vertex_images
        self.solid_images = solid_images
        self.created_entity_sources = created_entity_sources
        self.association_graph = association_graph
        self.recipe_names = recipe_names
        self.intersection_kind = parse(
            intersection_kind,
            BRepDefeatureIntersectionKind,
            "BRepDefeatureResult.intersection_kind",
        )
        self.carrier_intent_points = carrier_intent_points
        bounds = (
            _polynomial_bounds(volume_coefficients)
            if volume_coefficients is not None
            else volume_enclosure
        )
        self.volume_change_enclosure = jnp.asarray(
            () if bounds is None else bounds, dtype=jnp.float64
        )
        self.hausdorff_enclosure = jnp.asarray(
            () if model is None else hausdorff_enclosure, dtype=jnp.float64
        )
        self.intersection_rounding_bounds = jnp.asarray(
            intersection_rounding_bounds, dtype=jnp.float64
        )
        self.g1_angle_bounds = (
            jnp.empty((0, 2), dtype=jnp.float64)
            if g1_angle_bounds is None
            else g1_angle_bounds
        )
        if self.g1_angle_bounds.shape != (len(g1_edges), 2):
            raise ValueError(
                "BRepDefeatureResult.g1_edges must key every angle row; got mismatched evidence rows"
            )
        self.g1_edges = g1_edges

    @property
    def volume_change(self) -> Array:
        """Numerical realization within the certified volume enclosure."""
        if self.model is None:
            raise ValueError(
                "BRepDefeatureResult.volume_change must have certified evidence; got a refused result"
            )
        return jnp.mean(self.volume_change_enclosure)


def _polynomial_bounds(
    coefficients: tuple[Fraction, Fraction, Fraction], /
) -> tuple[float, float]:
    p = (
        Fraction(float(np.nextafter(pi, -np.inf))),
        Fraction(float(np.nextafter(pi, np.inf))),
    )
    lower = upper = coefficients[0]
    for coefficient, interval in zip(
        coefficients[1:], (p, (p[0] ** 2, p[1] ** 2)), strict=True
    ):
        values = tuple((coefficient * value for value in interval))
        lower += min(values)
        upper += max(values)
    return (fraction_down(lower), fraction_up(upper))


@checked
def brep_defeature(
    model: BRepModel,
    face: BRepEntityId,
    /,
    *,
    operation: BRepDefeatureOperation,
    recipe_names: RecipeEntityNames | None = None,
    volume_relative_tolerance: float = 1e-6,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-defeature",
) -> BRepDefeatureResult:
    """Remove a supported local blend or cap a supported through-hole wall.

    Other geometric families remain explicitly unsupported. Exact native
    intersections distinguish collision from unresolved contact proofs.
    ``volume_relative_tolerance`` is the certified relative half-width requested
    from the source and target volume measures.
    """
    operation_ = parse(operation, BRepDefeatureOperation, "brep_defeature.operation")
    if face not in model.face_ids:
        raise ValueError(
            "brep_defeature.face must belong to the source revision; got a foreign face identity"
        )
    if recipe_names is not None and recipe_names.model_id != model.model_id:
        raise ValueError(
            "brep_defeature.recipe_names must bind the source model; got foreign names"
        )
    if (
        isinstance(volume_relative_tolerance, bool)
        or not isinstance(volume_relative_tolerance, (int, float))
        or not np.isfinite(volume_relative_tolerance)
        or volume_relative_tolerance <= 0
    ):
        raise ValueError(
            "brep_defeature.volume_relative_tolerance must be finite and positive"
        )
    geometry = model.geometry
    try:
        _single_solid(model)
        index = model.face_ids.index(face)
        patch = model.patches[index]
        match operation_:
            case "hole":
                builder, faces, edges, coefficients, distance = _cap_hole(model, index)
                intersection_kind: BRepDefeatureIntersectionKind = "none"
                points: tuple[_Vector, ...] = ()
                g1: _G1Evidence = ((), jnp.empty((0, 2), dtype=jnp.float64))
            case "fillet" if isinstance(patch, TorusPatch):
                builder, faces, edges, coefficients, distance, points, g1 = _remove_torus(
                    model, index
                )
                intersection_kind = "circle"
            case "fillet" | "chamfer":
                builder, faces, edges, coefficients, distance, points, g1 = (
                    _remove_plane_blend(model, index, operation_)
                )
                intersection_kind = "line"
            case _:
                raise ValueError(
                    "brep_defeature.operation must select a supported operation; got an invalid selector"
                )
        rounding = vertex_rounding_bounds(points)
        margin = max(rounding, default=0.0)
        (
            staged_geometry,
            staged_patches,
            staged_bounds,
            staged_orientation,
            staged_tags,
            staged_trims,
        ) = _staged_geometry(builder, geometry, _surviving_shell(geometry, faces))
        target = assemble_brep_model(
            staged_geometry,
            staged_patches,
            staged_bounds,
            staged_orientation,
            staged_tags,
            coordinate_contract=model.coordinate_contract,
            source_id=source_id,
            tessellation=tessellation,
            trim_domains=staged_trims,
        )
        source_gap = _gap_upper(geometry)
        target_gap = _gap_upper(target.geometry)
        margin = fraction_up(
            Fraction(margin) + Fraction(source_gap) + Fraction(target_gap)
        )
        enclosure = None
        if coefficients is None or margin:
            coefficients = None
            enclosure = _volume_difference(
                model, target, builder, points, float(volume_relative_tolerance)
            )
        distance = (
            max(0.0, float(np.nextafter(distance[0] - margin, -np.inf))),
            float(np.nextafter(distance[1] + margin, np.inf)),
        )
    except _Refusal as refusal:
        status: BRepDefeatureStatus = (
            "collision"
            if isinstance(refusal, _Collision)
            else "unresolved"
            if isinstance(refusal, _Unresolved)
            else "unsupported"
        )
        return BRepDefeatureResult(
            status=status,
            reason=str(refusal),
            operation=operation_,
            source_model_id=model.source_revision,
            removed_face=face,
            model=None,
        )

    face_images = _entity_images(model, target, "face", dict(faces))
    edge_images = _entity_images(model, target, "edge", dict(edges))
    vertex_indices = _vertex_images(model, target, edges)
    vertex_images = _entity_images(model, target, "vertex", dict(vertex_indices))
    solid_indices = ((0, 0),)
    solid_images = _entity_images(model, target, "solid", dict(solid_indices))
    if operation_ == "hole":
        caps, _ = _hole_boundary(model, index)
        support_faces = (index, *caps)
    elif operation_ == "fillet" and isinstance(patch, TorusPatch):
        plane_face, cylinder_face, *_ = _torus_junction(model, index)
        support_faces = (index, plane_face, cylinder_face)
    else:
        sides, caps, _, _ = _plane_junction(model, index, operation_)
        support_faces = (index, *sides, *caps)
    support_ids = tuple(
        dict.fromkeys(model.face_ids[face_index] for face_index in support_faces)
    )
    mapped_target_ids = {
        target_id
        for pairs in (face_images, edge_images, vertex_images, solid_images)
        for _, target_id in pairs
    }
    created_entity_sources = tuple(
        (target_id, support_ids)
        for kind in ("face", "edge", "vertex", "solid")
        for target_id in getattr(target, f"{kind}_ids")
        if target_id not in mapped_target_ids
    )
    association_graph = _association_graph_for_images(
        model,
        target,
        {
            "solid": dict(solid_indices),
            "face": dict(faces),
            "edge": dict(edges),
            "vertex": dict(vertex_indices),
        },
        f"defeature-{operation_}",
    )
    names = _transport_names(recipe_names, target, faces, edges)
    return BRepDefeatureResult(
        status="certified",
        reason="native local extension and certified shape change",
        operation=operation_,
        source_model_id=model.source_revision,
        removed_face=face,
        model=target,
        volume_coefficients=coefficients,
        volume_enclosure=enclosure,
        hausdorff_enclosure=distance,
        intersection_kind=intersection_kind,
        carrier_intent_points=points,
        intersection_rounding_bounds=rounding,
        g1_angle_bounds=g1[1],
        g1_edges=g1[0],
        face_images=face_images,
        edge_images=edge_images,
        vertex_images=vertex_images,
        solid_images=solid_images,
        created_entity_sources=created_entity_sources,
        association_graph=association_graph,
        recipe_names=names,
    )


def _staged_geometry(
    builder: Builder, source: BRepGeometry, shells: _Shells, /
) -> tuple[
    BRepGeometry,
    tuple[AbstractSurfacePatch, ...],
    np.ndarray,
    np.ndarray,
    tuple[str, ...],
    tuple[TrimDomain | None, ...],
]:
    from ._model import (
        _certify_edge_face_gap_bounds,
        _validate_degenerate_coedge_poles,
    )

    # The staged geometry recertifies every edge/vertex gap from the transported
    # vertex and endpoint roots; no caller-side limit is needed.
    shell_faces, shell_orientations, solid_shells = shells
    try:
        provisional = builder.geometry(
            solid=True,
            shell_faces=shell_faces,
            shell_orientations=shell_orientations,
            solid_shells=solid_shells,
            assembly_graph=source.assembly_graph,
        )
    except ValueError as error:
        raise _Unresolved(
            f"brep_defeature.endpoint_gap must be continuously certified; got {error}"
        ) from error
    provisional_geometry, patches, provisional_bounds, _, _, _ = provisional
    try:
        # Retained poles must replay their exact full-interval collapse proof
        # before their degenerate coedges contribute a zero face gap.
        _validate_degenerate_coedge_poles(provisional_geometry, patches)
        face_gaps = _certify_edge_face_gap_bounds(
            provisional_geometry,
            patches,
            exact_unsupported=True,
            parameter_bounds=provisional_bounds,
            certified_degenerate_coedges=True,
        )
    except ValueError as error:
        raise _Unresolved(
            f"brep_defeature.edge_face_gap must be continuously certified; got {error}"
        ) from error
    return builder.geometry(
        solid=True,
        shell_faces=shell_faces,
        shell_orientations=shell_orientations,
        solid_shells=solid_shells,
        assembly_graph=source.assembly_graph,
        edge_vertex_gap_bounds=provisional_geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=face_gaps,
    )


def _single_solid(model: BRepModel, /) -> None:
    geometry = model.geometry
    occurrences = geometry.iter_occurrences()
    occurrence = next(occurrences, None)
    extra_occurrence = next(occurrences, None)
    if (
        len(geometry.solid_shells) != 1
        or len(geometry.shell_faces) != 1
        or len(geometry.solid_shells[0]) != 1
        or occurrence is None
        or extra_occurrence is not None
        or occurrence.solid_index != 0
        or occurrence.placement != BRepAffinePlacement.identity()
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; "
            "defeaturing requires one unplaced solid occurrence with one shell"
        )


def _surviving_shell(geometry: BRepGeometry, faces: _Images, /) -> _Shells:
    mapping = dict(faces)
    selected = tuple(
        (mapping[face], sense)
        for face, sense in zip(
            geometry.shell_faces[0], geometry.shell_orientations[0], strict=True
        )
        if face in mapping
    )
    return (
        (tuple(face for face, _ in selected),),
        (tuple(sense for _, sense in selected),),
        ((0,),),
    )


def _volume_difference(
    source: BRepModel,
    target: BRepModel,
    builder: Builder,
    points: tuple[_Vector, ...],
    relative_tolerance: float,
    /,
) -> tuple[float, float]:
    """Enclose the volume change from the two certified source measures.

    Each measure already includes the flux of the strips and fans closing its
    model's certified edge-face and vertex gaps.
    """
    source_mass = brep_mass_properties(source, relative_tolerance=relative_tolerance)
    target_mass = brep_mass_properties(target, relative_tolerance=relative_tolerance)
    for label, measured in (("source", source_mass), ("target", target_mass)):
        if not measured.status or measured.status[0] != "certified":
            reason = measured.reasons[0] if measured.reasons else "no solid occurrence"
            raise _Unresolved(
                f"brep_defeature.volume_measure could not certify the {label}: {reason}"
            )
    target_bounds = tuple(float(value) for value in target_mass.volume_enclosures[0])
    intent = _planar_volume_intent(builder, target.geometry, points)
    if intent is not None:
        target_bounds = min(target_bounds[0], intent[0]), max(target_bounds[1], intent[1])
    lower = Fraction(target_bounds[0]) - Fraction(
        float(source_mass.volume_enclosures[0, 1])
    )
    upper = Fraction(target_bounds[1]) - Fraction(
        float(source_mass.volume_enclosures[0, 0])
    )
    return fraction_down(lower), fraction_up(upper)


def _gap_upper(geometry: BRepGeometry, /) -> float:
    endpoint_rows = geometry.edge_vertex_gap_bounds
    face_rows = geometry.edge_face_gap_bounds
    vertex_bound = (
        max((max(row) for row in endpoint_rows), default=0.0)
        if endpoint_rows is not None
        else 0.0
    )
    face_bound = max(face_rows, default=0.0) if face_rows is not None else 0.0
    return fraction_up(Fraction(vertex_bound) + Fraction(face_bound))


def _planar_volume_intent(
    builder: Builder,
    geometry: BRepGeometry,
    points: tuple[_Vector, ...],
    /,
) -> tuple[float, float] | None:
    if len(points) != 2 or not all(
        isinstance(face.patch, PlanePatch) for face in builder.faces
    ):
        return None
    if not all(isinstance(curve, LineCurve) for curve in builder.pcurves):
        return None
    vertices = [
        exact if exact is not None else rational_vector(point)
        for exact, point in zip(
            builder.vertex_exact_points, builder.vertices, strict=True
        )
    ]
    first, last = builder.edge_vertices[-1]
    vertices[first], vertices[last] = points
    normals = []
    for face in builder.faces:
        if not isinstance(face.patch, PlanePatch):
            raise RuntimeError(
                "brep_defeature.publication must retain its native plane carriers."
            )
        normal = _normal(face.patch)
        lower, _ = square_root_interval(_dot(normal, normal))
        normals.append(tuple(value / Fraction(lower) for value in normal))
    evidence, _, _ = offset_rounding_evidence(
        builder,
        geometry,
        tuple(vertices),
        tuple(normals),
        tuple(face.orientation for face in builder.faces),
    )
    enclosure = evidence.volume_enclosure
    if enclosure is None:
        raise RuntimeError(
            "brep_defeature.publication lost its rational rounding enclosure."
        )
    return float(enclosure[0]), float(enclosure[1])


def _transport_names(
    source: RecipeEntityNames | None,
    target: BRepModel,
    faces: _Images,
    edges: _Images,
    /,
) -> RecipeEntityNames | None:
    if source is None:
        return None
    face_names = (
        None
        if source.face_names is None
        else tuple(
            source.face_names[index]
            for index, _ in sorted(faces, key=lambda pair: pair[1])
        )
    )
    edge_names = (
        None
        if source.edge_names is None or len(edges) != len(target.edge_ids)
        else tuple(
            source.edge_names[index]
            for index, _ in sorted(edges, key=lambda pair: pair[1])
        )
    )
    return RecipeEntityNames(target, face_names, edge_names)


def _vertex_images(source: BRepModel, target: BRepModel, edges: _Images, /) -> _Images:
    source_geometry = source.geometry
    target_geometry = target.geometry
    images: dict[int, int] = {}
    for source_edge, target_edge in edges:
        for source_vertex, target_vertex in zip(
            source_geometry.edge_vertices[source_edge],
            target_geometry.edge_vertices[target_edge],
            strict=True,
        ):
            prior = images.get(source_vertex)
            if prior is not None and prior != target_vertex:
                raise _Unresolved(
                    "brep_defeature.correspondence must preserve each retained source vertex; got conflicting exact target images"
                )
            images[source_vertex] = target_vertex
    return tuple(sorted(images.items()))


def _defeature_association_graph(
    source: BRepModel,
    target: BRepModel,
    faces: _Images,
    edges: _Images,
    vertices: _Images,
    solids: _Images,
    operation: BRepDefeatureOperation,
    /,
) -> AssociationGraph:
    return _association_graph_for_images(
        source,
        target,
        {
            "solid": dict(solids),
            "face": dict(faces),
            "edge": dict(edges),
            "vertex": dict(vertices),
        },
        f"defeature-{operation}",
    )


def _plane(model: BRepModel, face: int, /) -> PlanePatch:
    patch = planar_carrier(model.patches[face])
    if patch is not None:
        return patch
    source = model.patches[face]
    if isinstance(source, RevolutionSurface):
        line = _line_vector3(source.curve)
        if line is None:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "revolved planar host coefficients lack rational straight source coefficients"
            )
        curve_origin, direction = line
        axis_origin = _fraction_vector3(rational_vector(np.asarray(source.axis_origin)))
        axis = _fraction_vector3(rational_vector(np.asarray(source.axis_direction)))
        if axis_origin is None or axis is None:
            raise _Refusal("brep_defeature.family must have a rational revolution frame")
        delta = _offset(axis_origin, curve_origin)
        height = _dot(delta, axis)
        if (
            _dot(axis, axis) == 1
            and _dot(direction, axis) == 0
            and any(direction)
            and any(rational_cross(axis, direction))
        ):
            origin = _vector3_affine(axis_origin, height, axis)
            first = direction
            second = rational_cross(axis, first)
            realized_origin = np.asarray(
                (float(origin[0]), float(origin[1]), float(origin[2])),
                dtype=np.float64,
            )
            realized_first = np.asarray(
                (float(first[0]), float(first[1]), float(first[2])),
                dtype=np.float64,
            )
            realized_second = np.asarray(
                (float(second[0]), float(second[1]), float(second[2])),
                dtype=np.float64,
            )
            if (
                rational_vector(realized_origin) == origin
                and rational_vector(realized_first) == first
                and rational_vector(realized_second) == second
            ):
                return PlanePatch(realized_origin, realized_first, realized_second)
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "revolved planar host coefficients lack an exact radial-line frame"
        )
    raise _Refusal(
        "brep_defeature.family must have a supported analytic certificate; got "
        + "plane-pair removal requires analytic planar neighbors and planar end trims"
    )


def _normal(patch: PlanePatch, /) -> _Vector:
    return rational_cross(
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )


def _offset(first: _Vector, second: _Vector, /) -> _Vector:
    return (
        second[0] - first[0],
        second[1] - first[1],
        second[2] - first[2],
    )


def _length(vector: _Vector, /) -> Fraction | None:
    squared = _dot(vector, vector)
    a, b = (isqrt(squared.numerator), isqrt(squared.denominator))
    return (
        Fraction(a, b)
        if a * a == squared.numerator and b * b == squared.denominator
        else None
    )


def _coordinate(patch: PlanePatch, point: _Vector, /) -> Point:
    x = _fraction_vector3(rational_vector(np.asarray(patch.first_axis)))
    y = _fraction_vector3(rational_vector(np.asarray(patch.second_axis)))
    origin = _fraction_vector3(rational_vector(np.asarray(patch.origin)))
    if x is None or y is None or origin is None:
        raise _Refusal(
            "brep_defeature.family must have finite rational plane axes and origin."
        )
    delta = _offset(origin, point)
    result = _solve(
        ((_dot(x, x), _dot(x, y)), (_dot(x, y), _dot(y, y))),
        (_dot(x, delta), _dot(y, delta)),
    )
    return (result[0], result[1])


def _plane_junction(
    model: BRepModel, face: int, operation: BRepDefeatureOperation, /
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    geometry = model.geometry
    patch = model.patches[face]
    if operation == "fillet" and (not isinstance(patch, CylinderPatch)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "sphere blends join more than two neighbors; this fillet is not a cylindrical plane-pair blend"
        )
    if isinstance(patch, CylinderPatch):
        box = np.asarray(model.parameter_bounds[face])
        if Fraction(float(box[1, 0])) - Fraction(float(box[0, 0])) >= Fraction(
            float(np.nextafter(pi, -np.inf))
        ):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "fillet detection requires a minor cylindrical arc, not a major-arc bulge"
            )
    if operation == "chamfer" and planar_carrier(patch) is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "chamfer removal requires a planar face"
        )
    if len(geometry.face_loops[face]) != 1 or len(geometry.face_loops[face][0]) != 4:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "local blend requires two neighbor boundaries and two analytic end boundaries"
        )
    edges = tuple((geometry.coedge_edges[c] for c in geometry.face_loops[face][0]))
    adjacent = _blend_neighbors(model, face, edges)
    if len(set(adjacent)) != 4:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "blend joins more than two neighbors or lacks two distinct end trims"
        )
    for i, j in ((0, 2), (1, 3)):
        contact = (edges[i], edges[j])
        sides = (adjacent[i], adjacent[j])
        if _parallel_contact_lines(model, sides, contact):
            other = tuple((k for k in range(4) if k not in (i, j)))
            return (
                sides,
                (adjacent[other[0]], adjacent[other[1]]),
                contact,
                (edges[other[0]], edges[other[1]]),
            )
    raise _Refusal(
        "brep_defeature.family must have a supported analytic certificate; got blend boundaries outside the rational analytic plane intersection"
    )


def _blend_neighbors(
    model: BRepModel, face: int, edges: tuple[int, ...], /
) -> tuple[int, ...]:
    result = []
    for edge in edges:
        neighbors = set(model.topology.edge_faces[edge]) - {face}
        if len(neighbors) != 1:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "blend requires manifold two-face boundary incidence"
            )
        result.append(next(iter(neighbors)))
    return tuple(result)


def _parallel_contact_lines(
    model: BRepModel, sides: tuple[int, int], edges: tuple[int, int], /
) -> bool:
    geometry = model.geometry
    direction = rational_cross(*tuple(_normal(_plane(model, side)) for side in sides))
    if not any(direction):
        return False
    for edge in edges:
        index = geometry.edge_curves[edge]
        if index < 0:
            return False
        coefficients = _affine_edge(geometry.curves[index])
        if coefficients is None or any(rational_cross(coefficients[1], direction)):
            return False
    return True


def _sharp_endpoints(
    model: BRepModel,
    sides: tuple[int, int],
    caps: tuple[int, int],
    cap_edges: tuple[int, int],
    /,
) -> tuple[tuple[_Vector, _Vector], dict[int, int]]:
    geometry = model.geometry
    normals = tuple((_normal(_plane(model, face)) for face in sides))
    heights = tuple(
        (
            _dot(normal, rational_vector(np.asarray(_plane(model, face).origin)))
            for face, normal in zip(sides, normals, strict=True)
        )
    )
    points = []
    replacements = {}
    for end, (cap, edge) in enumerate(zip(caps, cap_edges, strict=True)):
        patch = _plane(model, cap)
        normal = _normal(patch)
        solved = _solve(
            (*normals, normal),
            (*heights, _dot(normal, rational_vector(np.asarray(patch.origin)))),
        )
        point = _fraction_vector3(solved)
        if point is None:
            raise _Unresolved(
                "brep_defeature.source must yield a rational three-plane intersection."
            )
        points.append(point)
        for vertex in geometry.edge_vertices[edge]:
            replacements[vertex] = end
    if len(replacements) != 4 or points[0] == points[1]:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "blend has collapsed or ambiguous endpoint incidence"
        )
    return ((points[0], points[1]), replacements)


def _literal_face_loops(
    model: BRepModel, face: int, /
) -> tuple[tuple[_Vector, ...], ...]:
    geometry = model.geometry
    result = []
    for loop in geometry.face_loops[face]:
        points = []
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if (
                geometry.edge_curves[edge] < 0
                or _affine_edge(geometry.curves[geometry.edge_curves[edge]]) is None
            ):
                raise _Unresolved(
                    "brep_defeature.certificate must decide the requested operation; "
                    "got an overlapping face outside literal planar polygon support"
                )
            vertices = geometry.edge_vertices[edge]
            endpoint = 0 if geometry.coedge_senses[coedge] == 1 else 1
            points.append(_exact_vertex(geometry, vertices[endpoint]))
        result.append(tuple(points))
    return tuple(result)


def _contains(point: Point, loops: tuple[tuple[Point, ...], ...], /) -> bool:
    boundary = any(
        (
            on_segment(point, (a, b))
            for loop in loops
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True)
        )
    )
    return boundary or (
        inside_loop(point, loops[0])
        and (not any((inside_loop(point, hole) for hole in loops[1:])))
    )


def _plane_cut(
    loops: tuple[tuple[_Vector, ...], ...], normal: _Vector, height: Fraction, /
) -> tuple[_Vector, ...]:
    points = set()
    for loop in loops:
        for first, last in zip(loop, loop[1:] + loop[:1], strict=True):
            a, b = (_dot(normal, first) - height, _dot(normal, last) - height)
            if a == 0:
                points.add(first)
            if a * b < 0:
                ratio = a / (a - b)
                point: _Vector = (
                    first[0] + ratio * (last[0] - first[0]),
                    first[1] + ratio * (last[1] - first[1]),
                    first[2] + ratio * (last[2] - first[2]),
                )
                points.add(point)
    return tuple(points)


def _polygon_contact(
    first_patch: PlanePatch,
    first: tuple[tuple[_Vector, ...], ...],
    second_patch: PlanePatch,
    second: tuple[tuple[_Vector, ...], ...],
    /,
) -> bool:
    n, m = (_normal(first_patch), _normal(second_patch))
    h, k = (
        _dot(n, rational_vector(np.asarray(first_patch.origin))),
        _dot(m, rational_vector(np.asarray(second_patch.origin))),
    )
    direction = rational_cross(n, m)
    a = _project_loops(first_patch, first)
    if not any(direction):
        if any((_dot(n, p) != h for loop in second for p in loop)):
            return False
        return _coplanar_contact(a, _project_loops(first_patch, second))
    candidates = sorted(
        set(_plane_cut(first, m, k)) | set(_plane_cut(second, n, h)),
        key=lambda p: _dot(p, direction),
    )
    b = _project_loops(second_patch, second)
    probes: list[_Vector] = candidates + [
        (
            (p[0] + q[0]) / 2,
            (p[1] + q[1]) / 2,
            (p[2] + q[2]) / 2,
        )
        for p, q in zip(candidates, candidates[1:])
    ]
    return any(
        (
            _contains(_coordinate(first_patch, p), a)
            and _contains(_coordinate(second_patch, p), b)
            for p in probes
            if _dot(n, p) == h and _dot(m, p) == k
        )
    )


def _project_loops(
    patch: PlanePatch, loops: tuple[tuple[_Vector, ...], ...], /
) -> tuple[tuple[Point, ...], ...]:
    return tuple((tuple((_coordinate(patch, point) for point in loop)) for loop in loops))


def _coplanar_contact(
    first: tuple[tuple[Point, ...], ...], second: tuple[tuple[Point, ...], ...], /
) -> bool:
    segments = tuple(
        ((a, b) for loop in first for a, b in zip(loop, loop[1:] + loop[:1], strict=True))
    )
    others = tuple(
        (
            (a, b)
            for loop in second
            for a, b in zip(loop, loop[1:] + loop[:1], strict=True)
        )
    )
    return (
        any((intersections(a, b) for a in segments for b in others))
        or _contains(first[0][0], second)
        or _contains(second[0][0], first)
    )


def _boxes_separate(
    first: tuple[_Vector, ...], lower: np.ndarray, upper: np.ndarray, /
) -> bool:
    return any(
        (
            max((p[axis] for p in first)) < Fraction(float(lower[axis]))
            or min((p[axis] for p in first)) > Fraction(float(upper[axis]))
            for axis in range(3)
        )
    )


def _certify_extensions(
    model: BRepModel,
    removed: int,
    sides: tuple[int, int],
    caps: tuple[int, int],
    contacts: tuple[int, int],
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> None:
    geometry = model.geometry
    strips = []
    for side, edge in zip(sides, contacts, strict=True):
        vertices = sorted(geometry.edge_vertices[edge], key=lambda v: replacements[v])
        old = tuple(_exact_vertex(geometry, vertex) for vertex in vertices)
        strips.append((side, (old[0], old[1], points[1], points[0])))
    for face, patch in enumerate(model.patches):
        if face in (removed, *sides, *caps):
            continue
        bounds = _source_surface_box(model, face)
        for side, strip in strips:
            _certify_strip(model, face, patch, bounds, side, strip, caps[0])


def _certify_strip(
    model: BRepModel,
    face: int,
    patch: AbstractSurfacePatch,
    bounds: np.ndarray,
    side: int,
    strip: tuple[_Vector, ...],
    cap: int,
    /,
) -> None:
    if _boxes_separate(strip, bounds[0], bounds[1]):
        return
    planar = planar_carrier(patch)
    if planar is None:
        if isinstance(
            patch, (CylinderPatch, RevolutionSurface)
        ) and _cylinder_extension_contact(model, face, cap, strip[0], strip[3]):
            raise _Collision(
                "brep_defeature.extension must avoid unrelated faces; got a certified native line/circle intersection"
            )
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got curved face overlap without a local Boolean clipping certificate"
        )
    if _polygon_contact(
        _plane(model, side), (strip,), planar, _literal_face_loops(model, face)
    ):
        raise _Collision(
            "brep_defeature.extension must avoid unrelated faces; got a certified native planar Boolean intersection"
        )


def _cylinder_extension_contact(
    model: BRepModel, face: int, cap: int, first: _Vector, last: _Vector, /
) -> bool:
    """Prove contact of a cap-plane strip edge with a cylindrical wall's cap circle.

    The wall boundary on the cap is either an authored circle p-curve or a
    closed one-coedge loop on the plane normal to an exact circular cylinder,
    which is then the full section circle. A line/circle contact is a point of
    the wall boundary on the extension.
    """
    geometry = model.geometry
    patch = _plane(model, cap)
    a, b = (_coordinate(patch, first), _coordinate(patch, last))
    if any((Fraction(float(value)) != value for point in (a, b) for value in point)):
        return False
    line = _profile_edge(
        (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), ProfileLine()
    )
    section = _cap_section_circle(model, face, patch)
    for loop in geometry.face_loops[cap]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if face not in model.topology.edge_faces[edge]:
                continue
            pcurve = geometry.pcurves[coedge]
            if isinstance(pcurve, CircleCurve):
                _full_turn(geometry, edge)
                center = tuple((float(value) for value in pcurve.center))
                radius = float(pcurve.radius)
            elif (
                section is not None
                and len(loop) == 1
                and geometry.edge_vertices[edge][0] == geometry.edge_vertices[edge][1]
            ):
                center = (float(section[0][0]), float(section[0][1]))
                radius = float(section[1])
            else:
                continue
            arc = _profile_edge(
                (center[0] + radius, center[1]),
                (center[0] + radius, center[1]),
                ProfileArc((center[0], center[1])),
            )
            if _line_arc_contact(line, arc, frozenset()):
                return True
    return False


def _cap_section_circle(
    model: BRepModel, face: int, cap: PlanePatch, /
) -> tuple[Point, Fraction] | None:
    """Return the exact cap-chart section circle of a cylinder normal to the cap.

    Only float-representable centers and radii are returned, so the planar
    contact predicate consumes the exact circle.
    """
    data = _exact_cylinder_axis(
        SurfaceRegion(
            model.patches[face],
            np.asarray(model.parameter_bounds[face], dtype=np.float64),
        )
    )
    first = _fraction_vector3(rational_vector(np.asarray(cap.first_axis)))
    second = _fraction_vector3(rational_vector(np.asarray(cap.second_axis)))
    cap_origin = _fraction_vector3(rational_vector(np.asarray(cap.origin)))
    if data is None or first is None or second is None or cap_origin is None:
        return None
    axis_origin, axis, radius = data
    normal = rational_cross(first, second)
    if (
        _dot(first, first) != 1
        or _dot(second, second) != 1
        or _dot(first, second) != 0
        or any(rational_cross(normal, axis))
    ):
        return None
    height = _dot(normal, _offset(axis_origin, cap_origin)) / _dot(normal, axis)
    center = _coordinate(
        cap,
        (
            axis_origin[0] + height * axis[0],
            axis_origin[1] + height * axis[1],
            axis_origin[2] + height * axis[2],
        ),
    )
    if any(Fraction(float(value)) != value for value in (*center, radius)):
        return None
    return center, radius


def _affine_edge(curve: BRepCurve, /) -> tuple[_Vector, _Vector] | None:
    if isinstance(curve, BSplineCurve):
        if curve.degree != 1 or curve.control_points.shape != (2, 3):
            return None
        weights = rational_vector(np.asarray(curve.weights))
        knots = rational_vector(np.asarray(curve.knots))
        if (
            len(knots) != 4
            or weights[0] <= 0
            or weights[0] != weights[1]
            or knots[0] != knots[1]
            or knots[2] != knots[3]
            or knots[2] <= knots[1]
        ):
            return None
        controls = np.asarray(curve.control_points)
        first = (
            Fraction.from_float(float(controls[0, 0])),
            Fraction.from_float(float(controls[0, 1])),
            Fraction.from_float(float(controls[0, 2])),
        )
        last = (
            Fraction.from_float(float(controls[1, 0])),
            Fraction.from_float(float(controls[1, 1])),
            Fraction.from_float(float(controls[1, 2])),
        )
        span = knots[2] - knots[1]
        slope = (
            (last[0] - first[0]) / span,
            (last[1] - first[1]) / span,
            (last[2] - first[2]) / span,
        )
        origin = (
            first[0] - knots[1] * slope[0],
            first[1] - knots[1] * slope[1],
            first[2] - knots[1] * slope[2],
        )
        return origin, slope
    if not isinstance(curve, AbstractCurve):
        return None
    coefficients = _affine_curve_coefficients(
        curve, (Fraction(0),) * curve.ambient_dimension
    )
    if coefficients is None:
        return None
    origin = _fraction_vector3(coefficients[0])
    direction = _fraction_vector3(coefficients[1])
    if origin is None or direction is None:
        return None
    return origin, direction


def _exact_vertex(geometry: BRepGeometry, vertex: int, /) -> _Vector:
    exact = geometry.vertex_exact_points[vertex]
    point = (
        exact
        if exact is not None
        else rational_vector(np.asarray(geometry.vertex_points[vertex]))
    )
    result = _fraction_vector3(point)
    if result is None:
        raise _Unresolved(
            "brep_defeature.source must preserve rational vertex coordinates."
        )
    return result


def _exact_source_vertex(geometry: BRepGeometry, vertex: int, /) -> _Vector | None:
    exact = geometry.vertex_exact_points[vertex]
    if exact is None:
        return None
    result = _fraction_vector3(exact)
    if result is None:
        raise _Unresolved(
            "brep_defeature.source must preserve rational vertex coordinates."
        )
    return result


def _stage_sharp_edges(
    model: BRepModel,
    removed_edges: set[int],
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> tuple[Builder, dict[int, int], int]:
    geometry = model.geometry
    builder = Builder()
    vertices = {
        vertex: builder.vertex(
            np.asarray(point), exact_point=_exact_source_vertex(geometry, vertex)
        )
        for vertex, point in enumerate(geometry.vertex_points)
        if vertex not in replacements
    }
    sharp_vertices = tuple(
        builder.vertex(
            np.asarray(tuple(float(value) for value in point), dtype=np.float64),
            exact_point=point,
        )
        for point in points
    )
    for old, endpoint in replacements.items():
        vertices[old] = sharp_vertices[endpoint]
    edges = {}
    for source in range(len(geometry.edge_vertices)):
        if source not in removed_edges:
            edges[source] = _stage_retained_edge(
                builder, geometry, source, vertices, points, replacements
            )
    sharp = builder.edge(
        ExactLineCurve(points[0], points[1]),
        0.0,
        1.0,
        sharp_vertices[0],
        sharp_vertices[1],
    )
    return builder, edges, sharp


def _stage_retained_edge(
    builder: Builder,
    geometry: BRepGeometry,
    source: int,
    vertices: dict[int, int],
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> int:
    start, end = geometry.edge_vertices[source]
    curve_index = geometry.edge_curves[source]
    if curve_index < 0:
        raise _Refusal(
            "brep_defeature.family must retain native endpoint support; got an unrelated pole edge"
        )
    first, last = (float(value) for value in geometry.edge_ranges[source])
    changed = start in replacements or end in replacements
    if changed:
        curve = _extended_edge_curve(geometry, source, points, replacements)
        first, last = (0.0, 1.0)
        endpoint_roots = (None, None)
    else:
        curve = geometry.curves[curve_index]
        endpoint_roots = geometry.edge_endpoint_roots[source]
    return builder.edge(
        curve,
        first,
        last,
        vertices[start],
        vertices[end],
        endpoint_roots=endpoint_roots,
    )


def _extended_edge_curve(
    geometry: BRepGeometry,
    edge: int,
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> ExactLineCurve:
    start, end = geometry.edge_vertices[edge]
    source = _affine_edge(geometry.curves[geometry.edge_curves[edge]])
    if source is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; "
            "local removal requires analytic affine endpoint trims"
        )
    a = (
        points[replacements[start]]
        if start in replacements
        else _exact_vertex(geometry, start)
    )
    b = points[replacements[end]] if end in replacements else _exact_vertex(geometry, end)
    origin, direction = source
    for point in (a, b):
        if any(rational_cross(_offset(origin, point), direction)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; "
                "got a retained trim outside the rational analytic intersection"
            )
    if _dot(_offset(a, b), direction) <= 0:
        raise _Collision(
            "brep_defeature.extension must avoid unrelated faces and reversed trims; "
            "got an extension reverses a neighboring edge or makes a face vanish"
        )
    return ExactLineCurve(a, b)


def _plane_pcurve(patch: PlanePatch, curve: BRepCurve, /) -> LineCurve:
    source = _affine_edge(curve)
    if source is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "rebuilt plane p-curve requires a native affine carrier"
        )
    origin, direction = source
    start = _coordinate(patch, origin)
    end = _coordinate(
        patch,
        (
            origin[0] + direction[0],
            origin[1] + direction[1],
            origin[2] + direction[2],
        ),
    )
    return LineCurve(
        tuple((float(v) for v in start)),
        tuple((float(b - a) for a, b in zip(start, end, strict=True))),
    )


def _stage_sharp_faces(
    model: BRepModel,
    removed: int,
    contacts: tuple[int, int],
    cap_edges: tuple[int, int],
    replacements: dict[int, int],
    builder: Builder,
    edges: dict[int, int],
    sharp: int,
    /,
) -> _Images:
    geometry = model.geometry
    images = []
    for source, patch in enumerate(model.patches):
        if source == removed:
            continue
        native_plane = planar_carrier(patch)
        converted = native_plane is not None and (not isinstance(patch, PlanePatch))
        if native_plane is not None:
            patch = native_plane
        loops = []
        box = np.asarray(model.parameter_bounds[source]).copy()
        for loop in geometry.face_loops[source]:
            loops.append(
                _stage_sharp_loop(
                    geometry,
                    loop,
                    patch,
                    converted,
                    cap_edges,
                    contacts,
                    replacements,
                    builder,
                    edges,
                    sharp,
                    box,
                )
            )
        images.append(
            (
                source,
                builder.face(
                    patch,
                    box,
                    int(np.asarray(model.orientation)[source]),
                    model.physical_tags[source],
                    loops,
                ),
            )
        )
    return tuple(images)


def _stage_sharp_loop(
    geometry: BRepGeometry,
    loop: tuple[int, ...],
    patch: AbstractSurfacePatch,
    converted: bool,
    cap_edges: tuple[int, int],
    contacts: tuple[int, int],
    replacements: dict[int, int],
    builder: Builder,
    edges: dict[int, int],
    sharp: int,
    box: np.ndarray,
    /,
) -> list[int]:
    staged = []
    for coedge in loop:
        edge = geometry.coedge_edges[coedge]
        if edge in cap_edges:
            continue
        sense = geometry.coedge_senses[coedge]
        if edge in contacts:
            target, sense, pcurve = _sharp_contact_use(
                geometry, coedge, patch, replacements, builder, sharp
            )
            endpoint_roots = (None, None)
        else:
            target = edges[edge]
            pcurve = _retained_pcurve(
                geometry,
                coedge,
                patch,
                converted,
                replacements,
                builder.curves[builder.edge_curves[target]],
            )
            if any(vertex in replacements for vertex in geometry.edge_vertices[edge]):
                # The edge was extended to a sharp vertex (_stage_retained_edge), so
                # its source endpoint roots name a vertex that no longer exists.
                endpoint_roots = (None, None)
            elif pcurve is geometry.pcurves[coedge]:
                endpoint_roots = geometry.coedge_endpoint_roots[coedge]
            else:
                endpoint_roots = _coedge_roots(pcurve, _native_roots(geometry, edge))
        staged.append(
            builder.coedge(target, sense, pcurve, endpoint_roots=endpoint_roots)
        )
        if isinstance(pcurve, LineCurve):
            _extend_parameter_box(box, pcurve, builder.edge_ranges[target])
    return staged


def _coedge_roots(
    pcurve: BRepCurve,
    roots: tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None],
    /,
    *,
    patch: AbstractSurfacePatch | None = None,
    axis: int | None = None,
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    def bind(root: NativePeriodEndpoint | None, /) -> NativePeriodEndpoint | None:
        if root is None:
            return None
        return NativePeriodEndpoint(
            pcurve,
            root.patch if patch is None else patch,
            root.axis if patch is None else axis,
            rational=root.rational,
            turns=root.turns,
        )

    return bind(roots[0]), bind(roots[1])


def _retained_pcurve(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    converted: bool,
    replacements: dict[int, int],
    curve: BRepCurve,
    /,
) -> BRepCurve:
    edge = geometry.coedge_edges[coedge]
    first, last = geometry.edge_vertices[edge]
    if not converted and first not in replacements and last not in replacements:
        return geometry.pcurves[coedge]
    if not isinstance(patch, PlanePatch):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got rebuilt affine trims require a planar native p-curve"
        )
    return _plane_pcurve(patch, curve)


def _extend_parameter_box(
    box: np.ndarray, pcurve: LineCurve, bounds: tuple[float, float], /
) -> None:
    origin, direction = (
        rational_vector(np.asarray(pcurve.origin)),
        rational_vector(np.asarray(pcurve.direction)),
    )
    for axis in range(2):
        values = tuple(
            (origin[axis] + Fraction(parameter) * direction[axis] for parameter in bounds)
        )
        box[0, axis] = min(box[0, axis], fraction_down(min(values)))
        box[1, axis] = max(box[1, axis], fraction_up(max(values)))


def _sharp_contact_use(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    replacements: dict[int, int],
    builder: Builder,
    sharp: int,
    /,
) -> tuple[int, int, BRepCurve]:
    edge = geometry.coedge_edges[coedge]
    first, last = geometry.edge_vertices[edge]
    sense = geometry.coedge_senses[coedge] * (
        1 if replacements[first] == 0 and replacements[last] == 1 else -1
    )
    curve = builder.curves[builder.edge_curves[sharp]]
    if not isinstance(patch, PlanePatch) or not isinstance(
        curve, (LineCurve, ExactLineCurve)
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; "
            "got sharp intersection p-curve requires analytic plane/line support"
        )
    return (sharp, sense, _plane_pcurve(patch, curve))


def _shape_change(
    model: BRepModel,
    face: int,
    sides: tuple[int, int],
    caps: tuple[int, int],
    cap_edges: tuple[int, int],
    points: tuple[_Vector, _Vector],
    operation: BRepDefeatureOperation,
    /,
) -> tuple[tuple[Fraction, Fraction, Fraction] | None, tuple[float, float]]:
    patch = model.patches[face]
    radius = (
        Fraction(float(patch.radius)) if isinstance(patch, CylinderPatch) else Fraction()
    )
    bounds = _source_surface_box(model, face)
    diagonal = tuple(
        (
            max(Fraction(float(bounds[1, axis])), *(point[axis] for point in points))
            - min(Fraction(float(bounds[0, axis])), *(point[axis] for point in points))
            for axis in range(3)
        )
    )
    bound = square_root_interval(_dot(diagonal, diagonal))[1]
    coefficients = None
    n, m = (_normal(_plane(model, side)) for side in sides)
    vector = _offset(points[0], points[1])
    perpendicular = all(
        (not any(rational_cross(_normal(_plane(model, cap)), vector)) for cap in caps)
    )
    length = _length(vector)
    if (
        _dot(n, m) == 0
        and perpendicular
        and (length is not None)
        and (operation != "fillet" or _plane_pair_tangent(model, face, sides))
    ):
        return _right_angle_change(
            model, caps[0], cap_edges[0], points[0], radius, length, operation
        )
    return (coefficients, (0.0, bound))


def _plane_pair_tangent(model: BRepModel, face: int, sides: tuple[int, int], /) -> bool:
    patch = model.patches[face]
    if not isinstance(patch, CylinderPatch):
        return False
    if exact_frame_orientation(patch.first_axis, patch.second_axis, patch.axis) is None:
        return False
    center = _fraction_vector3(rational_vector(np.asarray(patch.origin)))
    if center is None:
        return False
    radius = Fraction(float(patch.radius))
    for side in sides:
        plane = _plane(model, side)
        normal = _normal(plane)
        plane_origin = _fraction_vector3(rational_vector(np.asarray(plane.origin)))
        if plane_origin is None:
            return False
        distance = _dot(normal, _offset(plane_origin, center))
        if distance**2 != radius**2 * _dot(normal, normal):
            return False
    return True


def _right_angle_change(
    model: BRepModel,
    cap: int,
    cap_edge: int,
    corner: _Vector,
    radius: Fraction,
    length: Fraction,
    operation: BRepDefeatureOperation,
    /,
) -> tuple[tuple[Fraction, Fraction, Fraction] | None, tuple[float, float]]:
    geometry = model.geometry
    first, last = geometry.edge_vertices[cap_edge]
    a, b = (_exact_vertex(geometry, vertex) for vertex in (first, last))
    cross = rational_cross(_offset(a, corner), _offset(a, b))
    coedge = next(
        (
            c
            for loop in geometry.face_loops[cap]
            for c in loop
            if geometry.coedge_edges[c] == cap_edge
        )
    )
    sign = (
        1
        if _dot(cross, _normal(_plane(model, cap))) * geometry.coedge_senses[coedge] > 0
        else -1
    )
    if operation == "fillet":
        r2 = radius**2
        lo, hi = square_root_interval(Fraction(2))
        return (
            (sign * r2 * length, -sign * r2 * length / 4, Fraction()),
            (
                fraction_down(radius * (Fraction(lo) - 1)),
                fraction_up(radius * (Fraction(hi) - 1)),
            ),
        )
    triangle = square_root_interval(_dot(cross, cross))
    coefficients = (
        (sign * Fraction(triangle[0]) * length / 2, Fraction(), Fraction())
        if triangle[0] == triangle[1]
        else None
    )
    squared = _dot(_offset(a, corner), _offset(a, corner)) - _dot(
        _offset(a, corner), _offset(a, b)
    ) ** 2 / _dot(_offset(a, b), _offset(a, b))
    return (coefficients, square_root_interval(squared))


def _remove_plane_blend(
    model: BRepModel, face: int, operation: BRepDefeatureOperation, /
) -> _BlendRemoval:
    sides, caps, contacts, cap_edges = _plane_junction(model, face, operation)
    points, replacements = _sharp_endpoints(model, sides, caps, cap_edges)
    _certify_extensions(model, face, sides, caps, contacts, points, replacements)
    g1: _G1Evidence = ((), jnp.empty((0, 2), dtype=jnp.float64))
    if operation == "fillet":
        g1 = _g1_evidence(model, contacts)
        if not np.all(np.asarray(g1[1])[:, 0] <= 0) or not np.all(
            np.asarray(g1[1])[:, 1] >= 0
        ):
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got "
                + "fillet boundaries do not have a BX-11 whole-edge G1 bound containing tangency"
            )
    builder, edges, sharp = _stage_sharp_edges(
        model, set((*contacts, *cap_edges)), points, replacements
    )
    faces = _stage_sharp_faces(
        model, face, contacts, cap_edges, replacements, builder, edges, sharp
    )
    coefficients, distance = _shape_change(
        model, face, sides, caps, cap_edges, points, operation
    )
    return (builder, faces, tuple(edges.items()), coefficients, distance, points, g1)


def _g1_evidence(model: BRepModel, edges: tuple[int, int], /) -> _G1Evidence:
    records = brep_edge_classification(model, model.geometry)
    selected = tuple((records[edge] for edge in sorted(edges)))
    return (
        tuple((record.edge_id for record in selected)),
        jnp.stack(tuple((record.dihedral_range for record in selected))),
    )


def _copy_retained(
    model: BRepModel,
    retained: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...],
    /,
) -> tuple[Builder, _Images]:
    geometry = model.geometry
    builder = Builder()
    used = sorted(
        {
            geometry.coedge_edges[coedge]
            for _, loops in retained
            for loop in loops
            for coedge in loop
        }
    )
    vertices = sorted(
        {vertex for edge in used for vertex in geometry.edge_vertices[edge]}
    )
    vertex_map = {
        source: builder.vertex(
            np.asarray(geometry.vertex_points[source]),
            exact_point=geometry.vertex_exact_points[source],
            root=geometry.vertex_roots[source],
        )
        for source in vertices
    }
    edge_map = {}
    for source in used:
        curve_index = geometry.edge_curves[source]
        first, last = (float(value) for value in geometry.edge_ranges[source])
        start, end = geometry.edge_vertices[source]
        if curve_index < 0:
            # An unchanged pole keeps its vertex and native-period chart roots.
            edge_map[source] = builder.degenerate(
                vertex_map[start],
                first,
                last,
                endpoint_roots=geometry.edge_endpoint_roots[source],
            )
            continue
        edge_map[source] = builder.edge(
            geometry.curves[curve_index],
            first,
            last,
            vertex_map[start],
            vertex_map[end],
            endpoint_roots=geometry.edge_endpoint_roots[source],
        )
    for source, loops in retained:
        staged = []
        for loop in loops:
            coedges = []
            for coedge in loop:
                coedges.append(
                    builder.coedge(
                        edge_map[geometry.coedge_edges[coedge]],
                        geometry.coedge_senses[coedge],
                        geometry.pcurves[coedge],
                        endpoint_roots=geometry.coedge_endpoint_roots[coedge],
                    )
                )
            staged.append(coedges)
        builder.face(
            model.patches[source],
            np.asarray(model.parameter_bounds[source]),
            int(np.asarray(model.orientation)[source]),
            model.physical_tags[source],
            staged,
        )
    return builder, tuple(edge_map.items())


def _reparameterize_revolved_cap(
    model: BRepModel,
    wall_face: int,
    cap_face: int,
    target_face: int,
    edge_map: dict[int, int],
    builder: Builder,
    /,
) -> None:
    patch = model.patches[cap_face]
    if not isinstance(patch, RevolutionSurface):
        raise RuntimeError("Only a revolved planar cap can be reparameterized.")
    geometry = model.geometry
    wall_edges = {
        geometry.coedge_edges[coedge]
        for loop in geometry.face_loops[wall_face]
        for coedge in loop
    }
    circular_edges = {
        geometry.coedge_edges[coedge]
        for loop in geometry.face_loops[cap_face]
        for coedge in loop
        if geometry.coedge_edges[coedge] not in wall_edges
        and geometry.edge_curves[geometry.coedge_edges[coedge]] >= 0
        and isinstance(
            geometry.curves[geometry.edge_curves[geometry.coedge_edges[coedge]]],
            CircleCurve,
        )
    }
    if len(circular_edges) != 1:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a revolved planar host without one retained full circular boundary"
        )
    source_edge = next(iter(circular_edges))
    _full_turn(geometry, source_edge)
    circle_index = geometry.edge_curves[source_edge]
    source_circle = geometry.curves[circle_index]
    if not isinstance(source_circle, CircleCurve):
        raise RuntimeError("A revolved cap circular boundary lost its source circle.")
    axis = _fraction_vector3(rational_vector(np.asarray(patch.axis_direction)))
    axis_origin = _fraction_vector3(rational_vector(np.asarray(patch.axis_origin)))
    circle_center = _fraction_vector3(rational_vector(np.asarray(source_circle.center)))
    if axis is None or axis_origin is None or circle_center is None:
        raise _Refusal(
            "brep_defeature.family must have rational three-dimensional revolution coefficients"
        )
    if any(rational_cross(_offset(axis_origin, circle_center), axis)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a revolved host circle off its exact rotation axis"
        )
    circle_first = _fraction_vector3(
        rational_vector(np.asarray(source_circle.first_axis))
    )
    circle_second = _fraction_vector3(
        rational_vector(np.asarray(source_circle.second_axis))
    )
    if circle_first is None or circle_second is None:
        raise _Refusal(
            "brep_defeature.family must have a rational circular boundary frame"
        )
    plane_normal = rational_cross(circle_first, circle_second)
    if exact_frame_orientation(
        source_circle.first_axis,
        source_circle.second_axis,
        patch.axis_direction,
    ) is None or any(rational_cross(plane_normal, axis)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a revolved host circle without an exact plane frame"
        )
    source_plane = _plane(model, cap_face)
    source_normal = _normal(source_plane)
    if any(rational_cross(source_normal, plane_normal)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "revolved host charts that do not share the circular boundary plane"
        )
    coefficients = _line_vector3(patch.curve)
    if coefficients is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a revolved planar host without rational straight source coefficients"
        )
    curve_origin, curve_direction = coefficients
    parameter_bounds = np.asarray(model.parameter_bounds[cap_face], dtype=np.float64)
    parameter = (
        sum(
            (Fraction.from_float(float(value)) for value in parameter_bounds[:, 1]),
            Fraction(0),
        )
        / 2
    )
    point = _vector3_affine(curve_origin, parameter, curve_direction)
    delta = _offset(axis_origin, point)
    height = _dot(delta, axis)
    radial = _vector3_remove_projection(delta, axis, height)
    parameter_normal = rational_cross(rational_cross(axis, radial), curve_direction)
    alignment = rational_dot(parameter_normal, plane_normal)
    if alignment == 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a degenerate revolved host parameter frame"
        )
    # Coedge senses follow the chart orientation, so a chart whose normal is
    # opposite the revolved parameter normal reverses both the face
    # orientation and the retained circle use.
    chart_sign = 1 if alignment > 0 else -1
    orientation = int(np.asarray(model.orientation)[cap_face]) * chart_sign
    plane = PlanePatch(
        source_circle.center,
        source_circle.first_axis,
        source_circle.second_axis,
    )
    target_edge = edge_map[source_edge]
    coedges = tuple(
        coedge for loop in builder.faces[target_face].loops for coedge in loop
    )
    if len(coedges) != 1 or builder.coedge_edges[coedges[0]] != target_edge:
        raise _Refusal(
            "brep_defeature.family must retain one complete revolved host circle after removing its hole boundary"
        )
    target_curve = builder.curves[builder.edge_curves[target_edge]]
    if not isinstance(target_curve, CircleCurve):
        raise RuntimeError("A revolved cap target lost its circle carrier.")
    pcurve = CircleCurve(
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 1.0),
        float(np.asarray(target_curve.radius)),
        radius_exact=target_curve.radius_exact,
    )
    roots = builder.edge_endpoint_roots[target_edge]

    def bind(curve: AbstractCurve, root: object, /) -> NativePeriodEndpoint:
        if not isinstance(root, NativePeriodEndpoint):
            raise _Refusal(
                "brep_defeature.family must have native full-turn roots on each retained host circle"
            )
        return NativePeriodEndpoint(curve, rational=root.rational, turns=root.turns)

    first_root, last_root = roots
    edge_roots = (
        bind(target_curve, first_root),
        bind(target_curve, last_root),
    )
    coedge_roots = (bind(pcurve, first_root), bind(pcurve, last_root))
    coedge = coedges[0]
    builder.pcurves[coedge] = pcurve
    builder.coedge_senses[coedge] *= chart_sign
    builder.edge_endpoint_roots[target_edge] = edge_roots
    builder.coedge_endpoint_roots[coedge] = coedge_roots
    first, last = builder.edge_ranges[target_edge]
    builder.faces[target_face].patch = plane
    builder.faces[target_face].bounds = pcurve.bounding_box(first, last)
    builder.faces[target_face].orientation = orientation


def _full_turn(geometry: BRepGeometry, edge: int, /) -> None:
    first, last = geometry.edge_endpoint_roots[edge]
    if (
        not isinstance(first, NativePeriodEndpoint)
        or not isinstance(last, NativePeriodEndpoint)
        or last.turns - first.turns != 1
        or (first.rational != last.rational)
        or (geometry.edge_vertices[edge][0] != geometry.edge_vertices[edge][1])
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got circular endpoints without native full-turn evidence"
        )


def _hole_boundary(
    model: BRepModel, face: int, /
) -> tuple[tuple[int, int], tuple[int, int]]:
    geometry = model.geometry
    wall_edges = {
        geometry.coedge_edges[coedge]
        for loop in geometry.face_loops[face]
        for coedge in loop
    }
    cap_edges: dict[int, list[int]] = {}
    for edge in wall_edges:
        neighbors = set(model.topology.edge_faces[edge]) - {face}
        if not neighbors:
            continue
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0 or not isinstance(geometry.curves[curve_index], CircleCurve):
            continue
        _full_turn(geometry, edge)
        if len(neighbors) != 1:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "a circular wall edge shared by multiple host faces"
            )
        cap = next(iter(neighbors))
        cap_edges.setdefault(cap, []).append(edge)
    if len(cap_edges) != 2:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires one wall between exactly two planar hosts"
        )
    caps = tuple(sorted(cap_edges))
    edges = []
    for cap in caps:
        _plane(model, cap)
        circular_edges = cap_edges[cap]
        if len(circular_edges) != 1:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hosts without exactly one full circular wall boundary each"
            )
        edges.append(circular_edges[0])
    return ((caps[0], caps[1]), (edges[0], edges[1]))


def _frame(
    patch: CylinderPatch | ConePatch | TorusPatch, /
) -> tuple[_Vector, _Vector, _Vector]:
    if isinstance(patch, (CylinderPatch, TorusPatch)):
        x = _fraction_vector3(patch.first_axis_exact)
        y = _fraction_vector3(patch.second_axis_exact)
        axis = _fraction_vector3(patch.axis_exact)
        if x is None or y is None or axis is None:
            raise _Refusal(
                "brep_defeature.family must have a rational three-dimensional frame"
            )
        if (
            rational_dot(x, x) != 1
            or rational_dot(y, y) != 1
            or rational_dot(axis, axis) != 1
            or rational_dot(x, y) != 0
            or rational_dot(x, axis) != 0
            or rational_dot(y, axis) != 0
            or abs(rational_dot(rational_cross(x, y), axis)) != 1
        ):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "circular removal requires an orthonormal rational frame"
            )
        return x, y, axis
    x = _fraction_vector3(rational_vector(np.asarray(patch.first_axis)))
    y = _fraction_vector3(rational_vector(np.asarray(patch.second_axis)))
    axis = _fraction_vector3(rational_vector(np.asarray(patch.axis)))
    if x is None or y is None or axis is None:
        raise _Refusal(
            "brep_defeature.family must have a rational three-dimensional frame"
        )
    if exact_frame_orientation(patch.first_axis, patch.second_axis, patch.axis) is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "circular removal requires an orthonormal rational frame"
        )
    return (x, y, axis)


def _fraction_square_root(value: Fraction, /) -> Fraction | None:
    if value < 0:
        return None
    numerator, denominator = isqrt(value.numerator), isqrt(value.denominator)
    if (
        numerator * numerator != value.numerator
        or denominator * denominator != value.denominator
    ):
        return None
    return Fraction(numerator, denominator)


def _revolved_hole_wall(model: BRepModel, face: int, /) -> _HoleWallData | None:
    patch = model.patches[face]
    if not isinstance(patch, RevolutionSurface):
        return None
    if patch.native_period_symbols()[0] != "two_pi":
        return None
    bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    period = patch.periods[0]
    if period is None or bounds[1, 0] - bounds[0, 0] != float(period):
        return None
    coefficients = _line_vector3(patch.curve)
    if coefficients is None:
        return None
    curve_origin, curve_direction = coefficients
    axis = _fraction_vector3(rational_vector(np.asarray(patch.axis_direction)))
    axis_origin = _fraction_vector3(rational_vector(np.asarray(patch.axis_origin)))
    if axis is None or axis_origin is None:
        return None
    if rational_dot(axis, axis) != 1:
        return None
    first_parameter = Fraction.from_float(float(bounds[0, 1]))
    last_parameter = Fraction.from_float(float(bounds[1, 1]))
    first_point = _vector3_affine(curve_origin, first_parameter, curve_direction)
    delta = _offset(axis_origin, first_point)
    height = rational_dot(delta, axis)
    radial = _vector3_remove_projection(delta, axis, height)
    radius = _fraction_square_root(rational_dot(radial, radial))
    axial_rate = rational_dot(curve_direction, axis)
    if radius is None or radius == 0 or axial_rate == 0:
        return None
    radial_rate = _vector3_remove_projection(curve_direction, axis, axial_rate)
    if any(rational_cross(radial, radial_rate)):
        return None
    slope = rational_dot(radial, radial_rate) / (radius * axial_rate)
    end_radius = radius + slope * axial_rate * (last_parameter - first_parameter)
    if end_radius <= 0:
        return None
    first_axis = _vector3_scale(radial, Fraction(1, 1) / radius)
    second_axis = rational_cross(axis, first_axis)
    realized_first = np.asarray(
        (float(first_axis[0]), float(first_axis[1]), float(first_axis[2])),
        dtype=np.float64,
    )
    realized_second = np.asarray(
        (float(second_axis[0]), float(second_axis[1]), float(second_axis[2])),
        dtype=np.float64,
    )
    realized_axis = np.asarray(
        (float(axis[0]), float(axis[1]), float(axis[2])), dtype=np.float64
    )
    if (
        rational_vector(realized_first) != first_axis
        or rational_vector(realized_second) != second_axis
        or rational_vector(realized_axis) != axis
        or exact_frame_orientation(realized_first, realized_second, realized_axis) is None
    ):
        return None
    wall_origin = _vector3_affine(axis_origin, height, axis)
    last_height = axial_rate * (last_parameter - first_parameter)
    return _HoleWallData(
        wall_origin,
        first_axis,
        second_axis,
        axis,
        radius,
        (
            min(Fraction(0), last_height),
            max(Fraction(0), last_height),
        ),
        slope,
        "revolved-cylinder" if slope == 0 else "revolved-cone",
        1 if axial_rate > 0 else -1,
    )


def _hole_wall_data(model: BRepModel, face: int, /) -> _HoleWallData | None:
    patch = model.patches[face]
    if isinstance(patch, RevolutionSurface):
        return _revolved_hole_wall(model, face)
    if not isinstance(patch, (CylinderPatch, ConePatch)):
        return None
    first_axis, second_axis, axis = _frame(patch)
    if isinstance(patch, CylinderPatch):
        radius = Fraction(float(patch.radius))
        family: Literal["cylinder", "cone", "revolved-cylinder", "revolved-cone"] = (
            "cylinder"
        )
        slope = Fraction(0)
    else:
        radius = Fraction(float(patch.reference_radius))
        family = "cone"
        slope = Fraction(0)
    parameter_bounds = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    first_height = Fraction.from_float(float(parameter_bounds[0, 1]))
    last_height = Fraction.from_float(float(parameter_bounds[1, 1]))
    height_range = (min(first_height, last_height), max(first_height, last_height))
    origin = _fraction_vector3(rational_vector(np.asarray(patch.origin)))
    if origin is None:
        raise _Refusal(
            "brep_defeature.family must have a rational three-dimensional origin"
        )
    return _HoleWallData(
        origin,
        first_axis,
        second_axis,
        axis,
        radius,
        height_range,
        slope,
        family,
        1,
    )


def _cap_hole(model: BRepModel, face: int, /) -> _HoleRemoval:
    wall = _hole_wall_data(model, face)
    if wall is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires a source-bound cylinder or straight-generator revolution"
        )
    caps, edges = _hole_boundary(model, face)
    geometry = model.geometry
    if (
        int(np.asarray(model.orientation)[face])
        * wall.orientation_sign
        * _dot(rational_cross(wall.first_axis, wall.second_axis), wall.axis)
        >= 0
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires an inward material boundary, not an exterior boss"
        )
    heights, radii = _hole_sections(model, wall, caps, edges)
    coefficients = None
    height = abs(heights[1] - heights[0])
    if height == 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires distinct host planes"
        )
    if wall.family in ("cylinder", "revolved-cylinder"):
        if any(radius != wall.reference_radius for radius in radii):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "cylindrical cap radii must agree exactly with the wall"
            )
        coefficients = (Fraction(), wall.reference_radius**2 * height, Fraction())
    elif wall.family == "revolved-cone":
        expected = tuple(wall.reference_radius + wall.slope * value for value in heights)
        if radii != expected:
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got "
                + "host circle radii do not match the exact revolved cone generator"
            )
    container_bound = _hole_clearance(model, face, caps, wall)
    retained = _retained_hole_faces(geometry, face, caps)
    builder, edge_images = _copy_retained(model, retained)
    face_map = {source: target for target, (source, _) in enumerate(retained)}
    edge_map = dict(edge_images)
    for cap in caps:
        if isinstance(model.patches[cap], RevolutionSurface):
            _reparameterize_revolved_cap(
                model, face, cap, face_map[cap], edge_map, builder
            )
    cylindrical = wall.family in ("cylinder", "revolved-cylinder")
    return (
        builder,
        tuple(((source, target) for target, (source, _) in enumerate(retained))),
        edge_images,
        coefficients,
        (0.0, fraction_up(max(*radii, height / 2)))
        if cylindrical
        else (0.0, container_bound),
    )


def _retained_hole_faces(
    geometry: BRepGeometry, face: int, caps: tuple[int, int], /
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    wall_edges = {
        geometry.coedge_edges[coedge]
        for loop in geometry.face_loops[face]
        for coedge in loop
    }
    cap_seams: set[tuple[int, int]] = set()
    for cap in caps:
        uses: dict[int, list[int]] = {}
        for loop in geometry.face_loops[cap]:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                uses.setdefault(edge, []).append(coedge)
        for edge, coedges in uses.items():
            if (
                len(coedges) == 2
                and sum(geometry.coedge_senses[coedge] for coedge in coedges) == 0
                and set(geometry.topology.edge_faces[edge]) == {cap}
            ):
                cap_seams.add((cap, edge))

    retained: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    for source, loops in enumerate(geometry.face_loops):
        if source == face:
            continue
        retained_loops = []
        for original in loops:
            remaining = tuple(
                coedge
                for coedge in original
                if geometry.coedge_edges[coedge] not in wall_edges
                and (source, geometry.coedge_edges[coedge]) not in cap_seams
            )
            if remaining:
                retained_loops.append(remaining)
        if not retained_loops:
            raise _Refusal(
                "brep_defeature.family must retain every non-wall face boundary; got a face erased by hole capping"
            )
        retained.append((source, tuple(retained_loops)))
    return tuple(retained)


def _source_surface_box(model: BRepModel, face: int, /) -> np.ndarray:
    patch = model.patches[face]
    parameter_box = np.asarray(model.parameter_bounds[face], dtype=np.float64)
    try:
        prepared = prepare_interval_function(
            patch.evaluate,
            2,
            batch_capacity=1,
            constant_bounds=child_coefficient_enclosures(patch),
        )
        lower, upper = prepared.evaluate(
            parameter_box[0][None, :], parameter_box[1][None, :]
        )
    except (NotImplementedError, ValueError) as error:
        raise _Unresolved(
            f"brep_defeature.certificate must enclose source face {face}; got {error}"
        ) from error
    bounds = np.stack((lower[0], upper[0]))
    if (
        bounds.shape != (2, 3)
        or not np.all(np.isfinite(bounds))
        or np.any(bounds[0] > bounds[1])
    ):
        raise _Unresolved(
            f"brep_defeature.certificate must produce finite source bounds for face {face}"
        )
    return bounds


def _hole_clearance(
    model: BRepModel, face: int, caps: tuple[int, int], wall: _HoleWallData, /
) -> float:
    patch = model.patches[face]
    origin, axis = wall.origin, wall.axis
    box = _source_surface_box(model, face)
    corners = _box_corners(box)
    radial_upper = _box_radial_upper(corners, origin, axis)
    axial = tuple((_dot(_offset(origin, point), axis) for point in corners))
    heights = (min(axial), max(axial))
    for other, other_patch in enumerate(model.patches):
        if other in (face, *caps) or _axial_plane_separate(
            other_patch, axis, origin, heights
        ):
            continue
        bounds = _source_surface_box(model, other)
        if _boxes_separate(corners, bounds[0], bounds[1]):
            continue
        other_wall = _hole_wall_data(model, other)
        if _coaxial_cylinder_separate(
            other_patch, origin, axis, radial_upper, heights, other_wall
        ):
            continue
        candidate = _box_corners(bounds)
        if _box_inside_cylindrical_fill(patch, candidate, origin, axis, heights):
            raise _Collision(
                "brep_defeature.capping must avoid unrelated faces; got a face enclosed by the cylindrical fill"
            )
        raise _Unresolved(
            "brep_defeature.capping must certify a collision-free fill; got an overlapping face without native Boolean support"
        )
    diagonal = tuple(
        Fraction(float(b)) - Fraction(float(a))
        for a, b in zip(box[0], box[1], strict=True)
    )
    return square_root_interval(_dot(diagonal, diagonal))[1]


def _box_inside_cylindrical_fill(
    wall: AbstractSurfacePatch,
    corners: tuple[_Vector, ...],
    origin: _Vector,
    axis: _Vector,
    heights: tuple[Fraction, Fraction],
    /,
) -> bool:
    if not isinstance(wall, CylinderPatch):
        return False
    if _box_radial_upper(corners, origin, axis) >= Fraction(float(wall.radius)):
        return False
    projection = tuple((_dot(_offset(origin, point), axis) for point in corners))
    return min(projection) >= heights[0] and max(projection) <= heights[1]


def _box_corners(box: np.ndarray, /) -> tuple[_Vector, ...]:
    return tuple(
        (
            (
                Fraction(float(box[sides[0], 0])),
                Fraction(float(box[sides[1], 1])),
                Fraction(float(box[sides[2], 2])),
            )
            for sides in product((0, 1), repeat=3)
        )
    )


def _box_radial_upper(
    corners: tuple[_Vector, ...], origin: _Vector, axis: _Vector, /
) -> Fraction:
    squared = []
    for point in corners:
        delta = _offset(origin, point)
        radial = tuple(
            (
                value - _dot(delta, axis) * direction
                for value, direction in zip(delta, axis, strict=True)
            )
        )
        squared.append(_dot(radial, radial))
    return Fraction(square_root_interval(max(squared))[1])


def _coaxial_cylinder_separate(
    patch: AbstractSurfacePatch,
    origin: _Vector,
    axis: _Vector,
    radius: Fraction,
    heights: tuple[Fraction, Fraction],
    wall: _HoleWallData | None,
    /,
) -> bool:
    if isinstance(patch, CylinderPatch):
        try:
            _, _, patch_axis = _frame(patch)
        except _Refusal:
            return False
        if any(rational_cross(patch_axis, axis)):
            return False
        patch_origin = _fraction_vector3(patch.origin_exact)
        patch_radius = patch.radius_exact
        if patch_origin is None or not isinstance(patch_radius, Fraction):
            return False
        if any(rational_cross(_offset(origin, patch_origin), axis)):
            return False
        return patch_radius > radius
    if wall is None or wall.family not in ("revolved-cylinder", "revolved-cone"):
        return False
    if any(rational_cross(wall.axis, axis)):
        return False
    delta = _offset(origin, wall.origin)
    if any(rational_cross(delta, axis)):
        return False
    alignment = _dot(wall.axis, axis)
    if alignment not in (-1, 1):
        return False
    center_height = _dot(delta, axis)
    candidate_heights = tuple(
        center_height + alignment * value for value in wall.height_range
    )
    overlap = (
        max(heights[0], min(candidate_heights)),
        min(heights[1], max(candidate_heights)),
    )
    if overlap[0] > overlap[1]:
        return True
    local_heights = tuple(alignment * (value - center_height) for value in overlap)
    radii = tuple(wall.reference_radius + wall.slope * value for value in local_heights)
    return min(radii) > radius


def _hole_sections(
    model: BRepModel,
    wall: _HoleWallData,
    caps: tuple[int, int],
    edges: tuple[int, int],
    /,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    geometry = model.geometry
    origin, axis = wall.origin, wall.axis
    heights, radii = ([], [])
    for cap, edge in zip(caps, edges, strict=True):
        plane = _plane(model, cap)
        normal = _normal(plane)
        if any(rational_cross(normal, axis)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole capping requires host planes normal to the wall axis"
            )
        plane_origin = _fraction_vector3(rational_vector(np.asarray(plane.origin)))
        if plane_origin is None:
            raise _Refusal("brep_defeature.family must have a rational host-plane origin")
        z = _dot(normal, _offset(origin, plane_origin)) / _dot(normal, axis)
        heights.append(z)
        curve = geometry.curves[geometry.edge_curves[edge]]
        if not isinstance(curve, CircleCurve):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole capping requires native circle carriers on its hosts"
            )
        axis_array = np.asarray(tuple(float(value) for value in axis))
        if (
            exact_frame_orientation(curve.first_axis, curve.second_axis, axis_array)
            is None
        ):
            raise _Refusal(
                "brep_defeature.hole must have circular host loops; got a non-orthonormal circle frame"
            )
        center = _vector3_affine(origin, z, axis)
        if center != rational_vector(np.asarray(curve.center)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole cap is not concentric with its analytic wall"
            )
        radii.append(Fraction(float(curve.radius)))
    return (tuple(heights), tuple(radii))


def _torus_junction(model: BRepModel, face: int, /) -> tuple[int, int, int, int, int]:
    geometry = model.geometry
    edges = {geometry.coedge_edges[c] for loop in geometry.face_loops[face] for c in loop}
    neighbors: dict[int, list[int]] = {}
    for edge in sorted(edges):
        for other in set(model.topology.edge_faces[edge]) - {face}:
            neighbors.setdefault(other, []).append(edge)
    if len(neighbors) != 2:
        raise _Refusal(
            "brep_defeature.neighbors must contain two analytic faces; got a multi-neighbor torus"
        )
    first, second = sorted(neighbors)
    match (model.patches[first], model.patches[second]):
        case [PlanePatch(), CylinderPatch()]:
            plane, cylinder = (first, second)
        case [CylinderPatch(), PlanePatch()]:
            plane, cylinder = (second, first)
        case [RevolutionSurface(), RevolutionSurface()]:
            plane, cylinder = _revolved_torus_neighbors(model, first, second)
        case _:
            raise _Refusal(
                "brep_defeature.neighbors must be a plane and a cylinder; got an unsupported pair, including cylinder/cylinder quartics"
            )
    plane_edges, cylinder_edges = (neighbors[plane], neighbors[cylinder])
    if len(plane_edges) != 1 or len(cylinder_edges) != 1:
        raise _Refusal(
            "brep_defeature.boundaries must have one circle per neighbor; got a split circular boundary"
        )
    seam = _closed_cylinder_seam(geometry, cylinder)
    _full_turn(geometry, plane_edges[0])
    _full_turn(geometry, cylinder_edges[0])
    return (plane, cylinder, plane_edges[0], cylinder_edges[0], seam)


def _closed_cylinder_seam(geometry: BRepGeometry, face: int, /) -> int:
    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus cylinder neighbor requires one native rectangular chart"
        )
    edges = tuple((geometry.coedge_edges[c] for c in loops[0]))
    seams = tuple((edge for edge in set(edges) if edges.count(edge) == 2))
    if len(seams) != 1:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus cylinder neighbor requires one identity-shared seam"
        )
    return seams[0]


def _revolved_torus_neighbors(
    model: BRepModel, first: int, second: int, /
) -> tuple[int, int]:
    """Order a revolved planar host and a revolved straight-generator wall."""
    for cap, wall in ((first, second), (second, first)):
        data = _revolved_hole_wall(model, wall)
        if data is None or data.family != "revolved-cylinder":
            continue
        try:
            _plane(model, cap)
        except _Refusal:
            continue
        return cap, wall
    raise _Refusal(
        "brep_defeature.neighbors must be a plane and a cylinder; got an unsupported pair, including cylinder/cylinder quartics"
    )


def _circular_clearance(
    model: BRepModel,
    removed: int,
    neighbors: tuple[int, int],
    axis: _Vector,
    center: _Vector,
    radius: Fraction,
    old_radius: Fraction,
    height_range: tuple[Fraction, Fraction],
    /,
) -> None:
    extent = max(radius, old_radius)
    low, high = (min(height_range), max(height_range))
    points = tuple(
        (
            (
                center[0] + z * axis[0] + s * extent,
                center[1] + z * axis[1] + s * extent,
                center[2] + z * axis[2] + s * extent,
            )
            for z in (low, high)
            for s in (-1, 1)
        )
    )
    for face, patch in enumerate(model.patches):
        if face == removed or face in neighbors:
            continue
        if _axial_plane_separate(patch, axis, center, (low, high)):
            continue
        bounds = _source_surface_box(model, face)
        if not _boxes_separate(points, bounds[0], bounds[1]):
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got circular face overlap without a native Boolean clipping certificate"
            )


def _axial_plane_separate(
    patch: AbstractSurfacePatch,
    axis: _Vector,
    center: _Vector,
    heights: tuple[Fraction, Fraction],
    /,
) -> bool:
    plane = planar_carrier(patch)
    if plane is None:
        return False
    normal = _normal(plane)
    if any(rational_cross(normal, axis)):
        return False
    plane_origin = _fraction_vector3(rational_vector(np.asarray(plane.origin)))
    if plane_origin is None:
        return False
    z = _dot(normal, _offset(center, plane_origin)) / _dot(normal, axis)
    return z < heights[0] or z > heights[1]


def _plane_circle_clearance(
    model: BRepModel,
    face: int,
    center: _Vector,
    radius: Fraction,
    old_radius: Fraction,
    /,
) -> None:
    geometry = model.geometry
    for loop in geometry.face_loops[face]:
        if len(loop) != 1:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "torus plane neighbor requires circular analytic trim loops"
            )
        edge = geometry.coedge_edges[loop[0]]
        curve = geometry.curves[geometry.edge_curves[edge]]
        if not isinstance(curve, CircleCurve):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "torus plane neighbor requires native circle carriers"
            )
        curve_center = _fraction_vector3(curve.center_exact)
        if curve_center is None:
            raise _Refusal(
                "brep_defeature.family must have rational neighboring circle sources"
            )
        if curve_center != center:
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got a nonconcentric trim without a native circular Boolean certificate"
            )
        r = curve.radius_exact
        if not isinstance(r, Fraction):
            raise _Refusal(
                "brep_defeature.family must have rational neighboring circle radii"
            )
        if r != old_radius and min(radius, old_radius) <= r <= max(radius, old_radius):
            raise _Collision(
                "brep_defeature.extension must avoid unrelated faces and reversed trims; got "
                + "circular extension intersects another native circular plane trim"
            )


def _prepare_torus(model: BRepModel, face: int, /) -> _TorusIntent:
    patch = model.patches[face]
    if not isinstance(patch, TorusPatch):
        raise RuntimeError("brep_defeature.certificate lost the selected torus carrier.")
    plane_face, cylinder_face, plane_edge, cylinder_edge, seam_edge = _torus_junction(
        model, face
    )
    cylinder = model.patches[cylinder_face]
    if not isinstance(cylinder, CylinderPatch):
        raise RuntimeError("brep_defeature.certificate lost the torus cylinder carrier.")
    x, y, axis = _frame(cylinder)
    tx, ty, ta = _frame(patch)
    plane = _plane(model, plane_face)
    if axis != ta or x != tx or y != ty or any(rational_cross(_normal(plane), axis)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus removal requires a plane normal to the shared cylinder/torus frame"
        )
    origin = _fraction_vector3(cylinder.origin_exact)
    plane_origin = _fraction_vector3(rational_vector(np.asarray(plane.origin)))
    if origin is None or plane_origin is None:
        raise _Refusal(
            "brep_defeature.family must have rational cylinder and plane origins"
        )
    z = _dot(_normal(plane), _offset(origin, plane_origin)) / _dot(_normal(plane), axis)
    center = _vector3_affine(origin, z, axis)
    r = patch.minor_radius_exact
    radius = cylinder.radius_exact
    major = patch.major_radius_exact
    if (
        not isinstance(r, Fraction)
        or not isinstance(radius, Fraction)
        or not isinstance(major, Fraction)
    ):
        raise _Refusal(
            "brep_defeature.family must have rational torus and cylinder radii"
        )
    torus_center = _fraction_vector3(patch.center_exact)
    if torus_center is None:
        raise _Refusal("brep_defeature.family must have a rational torus center.")
    delta = _offset(center, torus_center)
    if (
        any(rational_cross(delta, axis))
        or abs(_dot(delta, axis)) != r
        or abs(radius - major) != r
    ):
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got torus coefficients inconsistent with the signed-distance G1 equations"
        )
    geometry = model.geometry
    plane_curve = geometry.curves[geometry.edge_curves[plane_edge]]
    if not isinstance(plane_curve, CircleCurve):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus plane boundary requires its native circle carrier"
        )
    old_radius = plane_curve.radius_exact
    if not isinstance(old_radius, Fraction):
        raise _Refusal(
            "brep_defeature.family must have a rational plane contact circle radius"
        )
    old_z = _circular_boundary_level(geometry, cylinder_face, cylinder_edge)
    if old_radius != major or old_z != z + _dot(delta, axis):
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got "
            + "torus contact circles do not agree with the tangent meridian"
        )
    phase_error = _torus_phase_error(model, face, r)
    return _TorusIntent(
        plane_face=plane_face,
        cylinder_face=cylinder_face,
        plane_edge=plane_edge,
        cylinder_edge=cylinder_edge,
        seam_edge=seam_edge,
        plane=plane,
        cylinder=cylinder,
        center=center,
        z=z,
        radius=radius,
        minor_radius=r,
        old_radius=old_radius,
        old_z=old_z,
        phase_error=phase_error,
    )


def _circular_boundary_level(geometry: BRepGeometry, face: int, edge: int, /) -> Fraction:
    coedge = next(
        (
            c
            for loop in geometry.face_loops[face]
            for c in loop
            if geometry.coedge_edges[c] == edge
        )
    )
    pcurve = geometry.pcurves[coedge]
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, LineCurve) or source.direction_exact != (
        Fraction(1),
        Fraction(0),
    ):
        raise _Refusal(
            "brep_defeature.circular_boundary must use a native angular isoline; got another p-curve family or direction"
        )
    level = source.origin_exact[1]
    if not isinstance(level, Fraction):
        raise _Refusal(
            "brep_defeature.circular_boundary must have a rational angular level"
        )
    return level


def _remove_torus(model: BRepModel, face: int, /) -> _BlendRemoval:
    _, cylinder_face, *_ = _torus_junction(model, face)
    if isinstance(model.patches[cylinder_face], RevolutionSurface):
        return _remove_revolved_torus(model, face)
    intent = _prepare_torus(model, face)
    plane_face, cylinder_face = (intent.plane_face, intent.cylinder_face)
    plane_edge, cylinder_edge, seam_edge = (
        intent.plane_edge,
        intent.cylinder_edge,
        intent.seam_edge,
    )
    plane, cylinder = (intent.plane, intent.cylinder)
    center, z, radius, r = (intent.center, intent.z, intent.radius, intent.minor_radius)
    old_radius, old_z, phase_error = (intent.old_radius, intent.old_z, intent.phase_error)
    geometry = model.geometry
    _plane_circle_clearance(model, plane_face, center, radius, old_radius)
    _circular_clearance(
        model,
        face,
        (plane_face, cylinder_face),
        _frame(cylinder)[2],
        center,
        radius,
        old_radius,
        (Fraction(), old_z - z),
    )
    retained = _retained_without(geometry, face)
    builder, edge_images = _copy_retained(model, retained)
    face_map = {source: target for target, (source, _) in enumerate(retained)}
    edge_map = dict(edge_images)
    circle = CircleCurve(
        tuple(float(v) for v in center),
        cylinder.first_axis,
        cylinder.second_axis,
        float(radius),
    )
    circle_center = _fraction_vector3(rational_vector(np.asarray(circle.center)))
    circle_axis = _fraction_vector3(rational_vector(np.asarray(circle.first_axis)))
    if circle_center is None or circle_axis is None:
        raise _Refusal(
            "brep_defeature.family must have a rational circular intersection frame"
        )
    circle_radius = Fraction.from_float(float(radius))
    exact_point = _vector3_affine(circle_center, circle_radius, circle_axis)
    vertex = builder.vertex(
        np.asarray(tuple(float(value) for value in exact_point)),
        exact_point=exact_point,
    )
    edge_roots = (
        NativePeriodEndpoint(circle, rational=Fraction(), turns=Fraction()),
        NativePeriodEndpoint(circle, rational=Fraction(), turns=Fraction(1)),
    )
    new_edge = builder.edge(
        circle, 0.0, 2 * pi, vertex, vertex, endpoint_roots=edge_roots
    )
    region = SurfaceRegion(plane, np.asarray(model.parameter_bounds[plane_face]))
    plane_pcurve = native_edge_pcurve(circle, region.patch)
    if not isinstance(plane_pcurve, CircleCurve):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got a plane/cylinder intersection circle without a native plane p-curve"
        )
    plane_bounds = plane_pcurve.bounding_box(0.0, 2 * pi)
    builder.faces[face_map[plane_face]].bounds = np.stack(
        (
            np.minimum(builder.faces[face_map[plane_face]].bounds[0], plane_bounds[0]),
            np.maximum(builder.faces[face_map[plane_face]].bounds[1], plane_bounds[1]),
        )
    )
    _restore_circular_loops(
        model,
        plane_face,
        cylinder_face,
        plane_edge,
        cylinder_edge,
        builder,
        edge_map,
        face_map,
        new_edge,
        plane_pcurve,
        float(z),
    )
    _extend_cylinder_seam(
        cylinder_face, seam_edge, edge_map, builder, face_map, old_z, z, vertex
    )
    _compact_staging(builder)
    distance = _torus_distance(r, phase_error)
    return (
        builder,
        tuple(face_map.items()),
        edge_images,
        None,
        distance,
        (center,),
        _g1_evidence(model, (plane_edge, cylinder_edge)),
    )


def _retained_without(
    geometry: BRepGeometry, face: int, /
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    removed_edges = {
        geometry.coedge_edges[c] for loop in geometry.face_loops[face] for c in loop
    }
    return tuple(
        (
            source,
            tuple(
                tuple(c for c in loop if geometry.coedge_edges[c] not in removed_edges)
                for loop in loops
            ),
        )
        for source, loops in enumerate(geometry.face_loops)
        if source != face
    )


def _torus_distance(minor: Fraction, phase_error: Fraction, /) -> tuple[float, float]:
    lo, hi = square_root_interval(Fraction(2))
    return (
        max(0.0, fraction_down(minor * (Fraction(lo) - 1) - phase_error)),
        fraction_up(minor * (Fraction(hi) - 1) + phase_error),
    )


def _line_parameter(
    line: tuple[_Vector3, _Vector3], point: _Vector3, /
) -> Fraction | None:
    origin, direction = line
    axis = next((index for index in range(3) if direction[index] != 0), None)
    if axis is None:
        return None
    parameter = (point[axis] - origin[axis]) / direction[axis]
    return parameter if _vector3_affine(origin, parameter, direction) == point else None


def _remove_revolved_torus(model: BRepModel, face: int, /) -> _BlendRemoval:
    """Remove a torus blend between a revolved planar host and a revolved wall.

    Both neighbors are revolutions of straight profile lines about the torus
    axis. Their profile lines meet at an exact rational corner; the restored
    sharp circle, its corner vertex and the extended profile seams are bound
    to that corner's conic root, mirroring the revolution constructor.
    """
    from ._intersection import ConicParameterEndpoint
    from ._patches import ConicParameterRoot
    from ._root_bindings import BRepConicVertex, BRepVertexRoot

    torus = model.patches[face]
    if not isinstance(torus, TorusPatch):
        raise RuntimeError("brep_defeature.certificate lost the selected torus carrier.")
    geometry = model.geometry
    cap_face, wall_face, cap_ring, wall_ring, wall_seam = _torus_junction(model, face)
    cap_seam = _closed_cylinder_seam(geometry, cap_face)
    cap, wall = model.patches[cap_face], model.patches[wall_face]
    wall_data = _revolved_hole_wall(model, wall_face)
    if (
        not isinstance(cap, RevolutionSurface)
        or not isinstance(wall, RevolutionSurface)
        or wall_data is None
    ):
        raise RuntimeError(
            "brep_defeature.certificate lost the revolved torus neighbors."
        )
    axis = wall_data.axis
    axis_origin = _fraction_vector3(rational_vector(np.asarray(wall.axis_origin)))
    torus_axis = _fraction_vector3(torus.axis_exact)
    torus_center = _fraction_vector3(torus.center_exact)
    if axis_origin is None or torus_axis is None or torus_center is None:
        raise _Refusal("brep_defeature.family must have a rational revolution frame")
    cap_axis = _fraction_vector3(rational_vector(np.asarray(cap.axis_direction)))
    cap_origin = _fraction_vector3(rational_vector(np.asarray(cap.axis_origin)))
    if (
        cap_axis != axis
        or cap_origin is None
        or any(rational_cross(_offset(axis_origin, cap_origin), axis))
        or any(rational_cross(torus_axis, axis))
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "revolved torus neighbors without one shared exact rotation axis"
        )
    plane = _plane(model, cap_face)
    normal = _normal(plane)
    plane_origin = _fraction_vector3(rational_vector(np.asarray(plane.origin)))
    wall_line = _line_vector3(wall.curve)
    cap_line = _line_vector3(cap.curve)
    if plane_origin is None or wall_line is None or cap_line is None:
        raise _Refusal(
            "brep_defeature.family must have rational straight revolution generators"
        )
    if any(rational_cross(normal, axis)) or _dot(normal, wall_line[1]) == 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus removal requires a plane normal to the shared cylinder/torus frame"
        )
    wall_corner = _dot(normal, _offset(wall_line[0], plane_origin)) / _dot(
        normal, wall_line[1]
    )
    corner = _vector3_affine(wall_line[0], wall_corner, wall_line[1])
    cap_corner = _line_parameter(cap_line, corner)
    if cap_corner is None:
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got "
            + "revolved torus generators without an exact common corner"
        )
    height = _dot(axis, _offset(axis_origin, corner))
    center = _vector3_affine(axis_origin, height, axis)
    rings = tuple(
        geometry.curves[geometry.edge_curves[edge]] for edge in (cap_ring, wall_ring)
    )
    cap_circle, wall_circle = rings
    if not isinstance(cap_circle, CircleCurve) or not isinstance(
        wall_circle, CircleCurve
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus contact boundaries without native circle carriers"
        )
    first_axis = _fraction_vector3(wall_circle.first_axis_exact)
    second_axis = _fraction_vector3(wall_circle.second_axis_exact)
    wall_center = _fraction_vector3(wall_circle.center_exact)
    if (
        first_axis is None
        or second_axis is None
        or wall_center is None
        or cap_circle.first_axis_exact != wall_circle.first_axis_exact
        or cap_circle.second_axis_exact != wall_circle.second_axis_exact
        or _fraction_vector3(cap_circle.center_exact) != center
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus contact circles without one shared revolution frame"
        )
    radial = _offset(center, corner)
    radius = _dot(radial, first_axis)
    if radius <= 0 or _vector3_scale(first_axis, radius) != radial:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a restored corner off the revolution seam half-plane"
        )
    r, major = torus.minor_radius_exact, torus.major_radius_exact
    old_radius = cap_circle.radius_exact
    if (
        not isinstance(r, Fraction)
        or not isinstance(major, Fraction)
        or not isinstance(old_radius, Fraction)
    ):
        raise _Refusal("brep_defeature.family must have rational torus and circle radii")
    delta = _offset(center, torus_center)
    old_height = _dot(axis, _offset(center, wall_center))
    if (
        any(rational_cross(delta, axis))
        or abs(_dot(delta, axis)) != r
        or abs(radius - major) != r
        or old_radius != major
        or wall_circle.radius_exact != radius
        or old_height != _dot(delta, axis)
    ):
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got torus coefficients inconsistent with the signed-distance G1 equations"
        )
    phase_error = _torus_phase_error(model, face, r)
    _circular_clearance(
        model,
        face,
        (cap_face, wall_face),
        axis,
        center,
        radius,
        old_radius,
        (Fraction(), old_height),
    )
    retained = _retained_without(geometry, face)
    builder, edge_images = _copy_retained(model, retained)
    face_map = {source: target for target, (source, _) in enumerate(retained)}
    edge_map = dict(edge_images)
    circle = CircleCurve(
        tuple(float(value) for value in center),
        wall_circle.first_axis,
        wall_circle.second_axis,
        float(radius),
        center_exact=center,
        first_axis_exact=wall_circle.first_axis_exact,
        second_axis_exact=wall_circle.second_axis_exact,
        radius_exact=radius,
    )
    corner_root = ConicParameterRoot(circle, Fraction(1), Fraction(0))
    vertex = builder.vertex(
        np.asarray(tuple(float(value) for value in corner)),
        exact_point=corner,
        root=BRepVertexRoot(BRepConicVertex(corner_root)),
    )
    period_roots = (
        NativePeriodEndpoint(circle, rational=Fraction(), turns=Fraction()),
        NativePeriodEndpoint(circle, rational=Fraction(), turns=Fraction(1)),
    )
    new_edge = builder.edge(
        circle, 0.0, 2 * pi, vertex, vertex, endpoint_roots=period_roots
    )
    for source, patch, line, seam, ring, level in (
        (cap_face, cap, cap_line, cap_seam, cap_ring, cap_corner),
        (wall_face, wall, wall_line, wall_seam, wall_ring, wall_corner),
    ):
        ring_vertex = geometry.edge_vertices[ring][0]
        start, end = geometry.edge_vertices[seam]
        if start == end or ring_vertex not in (start, end):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "a revolved torus neighbor seam that does not end on its contact circle"
            )
        moving = 0 if start == ring_vertex else 1
        old_roots = geometry.edge_endpoint_roots[seam]
        fixed_root = old_roots[1 - moving]
        if fixed_root is not None and not isinstance(fixed_root, ConicParameterEndpoint):
            raise _Refusal(
                "brep_defeature.source must preserve native endpoint period identity; "
                "received an unsupported endpoint root."
            )
        fixed_point = (
            _fraction_vector3(fixed_root.point_exact)
            if fixed_root is not None
            else _exact_vertex(geometry, (start, end)[1 - moving])
        )
        fixed_level = None if fixed_point is None else _line_parameter(line, fixed_point)
        if fixed_point is None or fixed_level is None:
            raise _Unresolved(
                "brep_defeature.source must preserve rational vertex coordinates."
            )
        points = (corner, fixed_point) if moving == 0 else (fixed_point, corner)
        levels = (level, fixed_level) if moving == 0 else (fixed_level, level)
        span = _offset(points[0], points[1])
        seam_line = LineCurve(
            tuple(float(value) for value in points[0]),
            tuple(float(value) for value in span),
            origin_exact=points[0],
            direction_exact=span,
        )
        seam_roots: list[ConicParameterEndpoint | None] = [None, None]
        seam_roots[moving] = ConicParameterEndpoint(
            corner_root, seam_line, Fraction(moving), corner
        )
        if fixed_root is not None:
            seam_roots[1 - moving] = ConicParameterEndpoint(
                fixed_root.root,
                seam_line,
                Fraction(1 - moving),
                fixed_root.point_exact,
            )
        target_seam = edge_map[seam]
        builder.curves[builder.edge_curves[target_seam]] = seam_line
        builder.edge_ranges[target_seam] = (0.0, 1.0)
        seam_vertices = list(builder.edge_vertices[target_seam])
        seam_vertices[moving] = vertex
        builder.edge_vertices[target_seam] = (seam_vertices[0], seam_vertices[1])
        builder.edge_endpoint_roots[target_seam] = (seam_roots[0], seam_roots[1])
        ring_pcurve = LineCurve(
            (0.0, float(level)),
            (1.0, 0.0),
            origin_exact=(Fraction(0), level),
            direction_exact=(Fraction(1), Fraction(0)),
        )
        old_level = None
        loops = []
        for loop in geometry.face_loops[source]:
            staged = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                pcurve = geometry.pcurves[coedge]
                if edge == ring:
                    old_level = _isoline_level(pcurve)
                    target, pcurve = new_edge, ring_pcurve
                    endpoint_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (
                        _coedge_roots(ring_pcurve, period_roots, patch=patch, axis=0)
                    )
                elif edge == seam:
                    target = target_seam
                    pcurve = _seam_pcurve(pcurve, patch, levels)
                    first_root, last_root = (
                        None
                        if root is None
                        else ConicParameterEndpoint(
                            root.root, pcurve, root.parameter_exact, root.point_exact
                        )
                        for root in seam_roots
                    )
                    endpoint_roots = (first_root, last_root)
                elif edge in (cap_ring, wall_ring):
                    raise _Refusal(
                        "brep_defeature.family must have a supported analytic certificate; got "
                        + "a torus contact circle shared by both revolved neighbors"
                    )
                else:
                    target = edge_map[edge]
                    endpoint_roots = geometry.coedge_endpoint_roots[coedge]
                staged.append(
                    builder.coedge(
                        target,
                        geometry.coedge_senses[coedge],
                        pcurve,
                        endpoint_roots=endpoint_roots,
                    )
                )
            loops.append(staged)
        target_face = builder.faces[face_map[source]]
        target_face.loops = loops
        box = np.asarray(target_face.bounds, dtype=np.float64).copy()
        # Chart bounds are outward-rounded; the contact isoline owns the nearer
        # profile bound and the opposite ring keeps the other one.
        distances = (
            None
            if old_level is None
            else tuple(abs(float(box[row, 1]) - old_level) for row in (0, 1))
        )
        if old_level is None or distances is None or distances[0] == distances[1]:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "a revolved torus neighbor chart not bounded by its contact isoline"
            )
        row = 0 if distances[0] < distances[1] else 1
        other = Fraction(float(box[1 - row, 1]))
        if (level - other) * (Fraction(old_level) - other) <= 0:
            raise _Collision(
                "brep_defeature.extension must avoid unrelated faces and reversed trims; got "
                + "circular extension makes a revolved neighbor vanish"
            )
        bound = float(level)
        if (row == 1 and Fraction(bound) < level) or (
            row == 0 and Fraction(bound) > level
        ):
            bound = float(np.nextafter(bound, np.inf if row == 1 else -np.inf))
        box[row, 1] = bound
        target_face.bounds = box
    _compact_staging(builder)
    return (
        builder,
        tuple(face_map.items()),
        edge_images,
        None,
        _torus_distance(r, phase_error),
        (center,),
        _g1_evidence(model, (cap_ring, wall_ring)),
    )


def _isoline_level(pcurve: BRepCurve, /) -> float | None:
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, LineCurve) or source.direction_exact[1] != 0:
        return None
    return float(np.asarray(source.origin)[1])


def _seam_pcurve(
    pcurve: BRepCurve,
    patch: AbstractSurfacePatch,
    levels: tuple[Fraction, Fraction],
    /,
) -> BRepCurve:
    """Re-span a profile seam p-curve over the extended seam edge parameter."""
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, LineCurve) or source.direction_exact[0] != 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "a revolved neighbor seam without a native profile isoline"
        )
    angle = source.origin_exact[0]
    span = levels[1] - levels[0]
    line = LineCurve(
        (float(np.asarray(source.origin)[0]), float(levels[0])),
        (0.0, float(span)),
        origin_exact=(angle, levels[0]),
        direction_exact=(Fraction(0), span),
    )
    if isinstance(pcurve, PeriodicPCurve):
        return PeriodicPCurve(line, patch, pcurve.period_shifts)
    return line


def _torus_phase_error(model: BRepModel, face: int, radius: Fraction, /) -> Fraction:
    low, high = (
        Fraction(float(np.nextafter(pi, -np.inf))),
        Fraction(float(np.nextafter(pi, np.inf))),
    )
    phases = []
    errors = []
    for value in np.asarray(model.parameter_bounds[face])[:, 1]:
        coefficient = Fraction(float(value))
        candidates = [
            (k, tuple(sorted((k * low / 2, k * high / 2)))) for k in range(-4, 5)
        ]
        # Chart bounds may be outward-rounded root enclosures; any coefficient
        # within pi/8 of one enclosure identifies its quarter turn uniquely, and
        # the distance to both enclosure ends bounds the phase error rigorously.
        admitted = [
            (k, interval)
            for k, interval in candidates
            if interval[0] - low / 8 <= coefficient <= interval[1] + low / 8
        ]
        if len(admitted) != 1:
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got "
                + "torus meridian endpoints lack native quarter-turn enclosures"
            )
        k, interval = admitted[0]
        phases.append(k)
        errors.append(max(abs(coefficient - interval[0]), abs(coefficient - interval[1])))
    if phases[1] - phases[0] != 1:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus removal requires a quarter-circle meridian, not a major arc"
        )
    return radius * max(errors)


def _restore_circular_loops(
    model: BRepModel,
    plane_face: int,
    cylinder_face: int,
    plane_edge: int,
    cylinder_edge: int,
    builder: Builder,
    edges: dict[int, int],
    faces: dict[int, int],
    circle: int,
    plane_pcurve: BRepCurve,
    z: float,
    /,
) -> None:
    geometry = model.geometry
    cylinder = model.patches[cylinder_face]
    if not isinstance(cylinder, CylinderPatch):
        raise _Refusal("brep_defeature.family must retain its analytic cylinder carrier.")
    period_roots = _native_endpoint_pair(builder.edge_endpoint_roots[circle])
    for source in (plane_face, cylinder_face):
        loops = []
        for loop in geometry.face_loops[source]:
            staged = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if edge in (plane_edge, cylinder_edge):
                    target = circle
                    if source == plane_face:
                        pcurve = plane_pcurve
                        endpoint_roots = _coedge_roots(pcurve, period_roots)
                    else:
                        pcurve = LineCurve((0.0, z), (1.0, 0.0))
                        endpoint_roots = _coedge_roots(
                            pcurve, period_roots, patch=cylinder, axis=0
                        )
                else:
                    target = edges[edge]
                    pcurve = geometry.pcurves[coedge]
                    endpoint_roots = geometry.coedge_endpoint_roots[coedge]
                staged.append(
                    builder.coedge(
                        target,
                        geometry.coedge_senses[coedge],
                        pcurve,
                        endpoint_roots=endpoint_roots,
                    )
                )
            loops.append(staged)
        builder.faces[faces[source]].loops = loops


def _compact_staging(builder: Builder, /) -> None:
    used = [c for face in builder.faces for loop in face.loops for c in loop]
    mapping = {old: new for new, old in enumerate(used)}
    builder.coedge_edges = [builder.coedge_edges[c] for c in used]
    builder.coedge_senses = [builder.coedge_senses[c] for c in used]
    builder.pcurves = [builder.pcurves[c] for c in used]
    builder.coedge_endpoint_roots = [builder.coedge_endpoint_roots[c] for c in used]
    for face in builder.faces:
        face.loops = [[mapping[c] for c in loop] for loop in face.loops]
    _compact_vertices(builder)


def _compact_vertices(builder: Builder, /) -> None:
    vertices = sorted({v for edge in builder.edge_vertices for v in edge})
    vertex_map = {old: new for new, old in enumerate(vertices)}
    builder.vertices = [builder.vertices[v] for v in vertices]
    builder.vertex_exact_points = [builder.vertex_exact_points[v] for v in vertices]
    builder.vertex_roots = [builder.vertex_roots[v] for v in vertices]
    builder.edge_vertices = [
        (vertex_map[a], vertex_map[b]) for a, b in builder.edge_vertices
    ]


def _extend_cylinder_seam(
    face: int,
    seam: int,
    edges: dict[int, int],
    builder: Builder,
    faces: dict[int, int],
    old_z: Fraction,
    z: Fraction,
    vertex: int,
    /,
) -> None:
    target = edges[seam]
    bounds = builder.edge_ranges[target]
    endpoint = 0 if Fraction(bounds[0]) == old_z else 1
    if Fraction(bounds[endpoint]) != old_z:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus cylinder seam does not use the native axial parameter"
        )
    updated = list(bounds)
    updated[endpoint] = float(z)
    if updated[0] >= updated[1]:
        raise _Collision(
            "brep_defeature.extension must avoid unrelated faces and reversed trims; got "
            + "circular extension makes the cylinder neighbor vanish"
        )
    builder.edge_ranges[target] = (updated[0], updated[1])
    first, last = builder.edge_vertices[target]
    builder.edge_vertices[target] = (vertex, last) if endpoint == 0 else (first, vertex)
    builder.edge_endpoint_roots[target] = (None, None)
    box = builder.faces[faces[face]].bounds.copy()
    box[endpoint, 1] = float(z)
    builder.faces[faces[face]].bounds = box
    for loop in builder.faces[faces[face]].loops:
        for coedge in loop:
            if builder.coedge_edges[coedge] == target:
                builder.coedge_endpoint_roots[coedge] = (None, None)
