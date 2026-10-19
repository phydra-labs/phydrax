#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Local analytic feature removal with certified boundary and volume changes.

Plane-pair blends are collapsed locally onto their rational intersection line;
normal plane/cylinder torus blends collapse onto their intersection circle.
Only affected trims change. Publication uses the native endpoint-gap convention
when an exact rational intersection has no binary64 realization.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import product
from math import isqrt, pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked, parse
from .._rational import fraction_down, fraction_up, rational_cross, rational_vector
from ..design._naming import RecipeEntityNames
from ._constructors import (
    _line_arc_contact,
    _profile_edge,
    BRepTessellationPolicy,
    Builder,
    finish,
    ProfileArc,
    ProfileLine,
)
from ._coordinate_request import coordinate_source_request
from ._edge_classification import brep_edge_classification
from ._exact import square_root_interval
from ._intersection import native_edge_pcurve, NativePeriodEndpoint
from ._intersection_curve import affine_curve_coefficients, PeriodicPCurve, SurfaceRegion
from ._measure import brep_mass_properties
from ._model import BRepCurve, BRepEntityId, BRepGeometry, BRepModel, BRepPCurve
from ._offset_rounding import offset_rounding_evidence, vertex_rounding_bounds
from ._patches import (
    AbstractCurve,
    AbstractSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    exact_frame_orientation,
    LineCurve,
    PlanePatch,
    TorusPatch,
)
from ._planar_arrangement import inside_loop, intersections, on_segment, Point
from ._planar_offset import _dot, _solve, planar_carrier


BRepDefeatureStatus: TypeAlias = Literal[
    "certified", "unsupported", "unresolved", "collision"
]
BRepDefeatureOperation: TypeAlias = Literal["fillet", "chamfer", "hole"]
BRepDefeatureVolumeKind: TypeAlias = Literal["exact", "enclosed", "absent"]
BRepDefeatureIntersectionKind: TypeAlias = Literal["none", "line", "circle"]
type _Vector = tuple[Fraction, ...]
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
type _Shells = tuple[
    tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]
]


class _Refusal(ValueError):
    """A family lacks an extension or publication certificate."""


class _Collision(_Refusal):
    """A rational native intersection proves an extension collision."""


class _Unresolved(_Refusal):
    """The declared work or source family cannot decide a collision."""


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
    """A represented solid, rational intent, and certified shape-change evidence.

    Symbolic volumes use ``volume_rational + pi*volume_pi + pi**2*volume_pi_squared``.
    Other admitted families report the difference of independent native BX-01
    enclosures, never an estimated volume. Positive change means material added.
    Recipe keys of surviving faces are unchanged, including extended hosts.
    Edge names are supported only with exhaustive surviving target-edge ancestry.

    ``intersection_kind`` identifies the carrier data: two line endpoints,
    one circle center, or no intersection data for capping. The rational
    ``carrier_intent_points`` are defining data, not revision-scoped vertices.
    """

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
    intersection_kind: BRepDefeatureIntersectionKind = eqx.field(static=True)
    carrier_intent_points: tuple[_Vector, ...] = eqx.field(static=True)
    g1_edges: tuple[BRepEntityId, ...] = eqx.field(static=True)
    model: BRepModel | None
    volume_change_enclosure: Array
    volume_gap_error_bound: Array
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
        volume_gap_error_bound: float = 0.0,
        hausdorff_enclosure: tuple[float, float] = (0.0, 0.0),
        intersection_kind: BRepDefeatureIntersectionKind = "none",
        carrier_intent_points: tuple[_Vector, ...] = (),
        intersection_rounding_bounds: tuple[float, ...] = (),
        g1_angle_bounds: Array | None = None,
        g1_edges: tuple[BRepEntityId, ...] = (),
        face_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        edge_images: tuple[tuple[BRepEntityId, BRepEntityId], ...] = (),
        recipe_names: RecipeEntityNames | None = None,
    ) -> None:
        self.status = parse(status, BRepDefeatureStatus, "BRepDefeatureResult.status")
        self.operation = parse(
            operation, BRepDefeatureOperation, "BRepDefeatureResult.operation"
        )
        if (self.status == "certified") != (
            model is not None
            and (volume_coefficients is not None or volume_enclosure is not None)
        ):
            raise ValueError(
                "BRepDefeatureResult.evidence must accompany exactly certified results; got inconsistent evidence"
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
        self.volume_gap_error_bound = jnp.asarray(
            () if model is None else volume_gap_error_bound, dtype=jnp.float64
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
        values = tuple(coefficient * value for value in interval)
        lower += min(values)
        upper += max(values)
    return fraction_down(lower), fraction_up(upper)


@checked
def brep_defeature(
    model: BRepModel,
    face: BRepEntityId,
    /,
    *,
    operation: BRepDefeatureOperation,
    recipe_names: RecipeEntityNames | None = None,
    maximum_cells: int = 1,
    volume_subdivisions: int = 8,
    tessellation: BRepTessellationPolicy | None = None,
    source_id: str = "native-defeature",
    budget: CoordinateEnclosureBudget | None = None,
) -> BRepDefeatureResult:
    """Remove a local plane-pair blend, normal circular fillet, or through hole.

    The caller declares work limits, not geometric tolerances. Sphere corner
    blends, cylinder/cylinder torus blends, non-normal circular blends, and
    non-analytic neighbors refuse. Collision decisions use exact predicates
    from the native planar owner; unsupported overlapping curved faces are unresolved.
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
    for value in (maximum_cells, volume_subdivisions):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(
                "brep_defeature.work must declare positive integer limits; got an invalid work limit"
            )
    geometry = model.require_geometry()
    if budget is None:
        with coordinate_source_request(
            (model, face, maximum_cells, volume_subdivisions)
        ) as request:
            return brep_defeature(
                model,
                face,
                operation=operation_,
                recipe_names=recipe_names,
                maximum_cells=maximum_cells,
                volume_subdivisions=volume_subdivisions,
                tessellation=tessellation,
                source_id=source_id,
                budget=request,
            )
    budget.reserve(
        len(geometry.coedge_edges) ** 2 + 1, 512 * (len(geometry.coedge_edges) + 1)
    )
    try:
        _single_solid(model)
        index = model.face_ids.index(face)
        patch = model.patches[index]
        match operation_:
            case "hole":
                builder, faces, edges, coefficients, distance = _cap_hole(
                    model, index, budget
                )
                intersection_kind: BRepDefeatureIntersectionKind = "none"
                points: tuple[_Vector, ...] = ()
                g1: _G1Evidence = (), jnp.empty((0, 2), dtype=jnp.float64)
            case "fillet" if isinstance(patch, TorusPatch):
                builder, faces, edges, coefficients, distance, points, g1 = _remove_torus(
                    model, index, budget
                )
                intersection_kind = "circle"
            case "fillet" | "chamfer":
                builder, faces, edges, coefficients, distance, points, g1 = (
                    _remove_plane_blend(model, index, operation_, maximum_cells, budget)
                )
                intersection_kind = "line"
            case _:
                raise ValueError(
                    "brep_defeature.operation must select a supported operation; got an invalid selector"
                )
        rounding = vertex_rounding_bounds(points, budget=budget)
        margin = max(rounding, default=0.0)
        target = finish(
            builder,
            solid=True,
            coordinate_contract=model.coordinate_contract,
            source_id=source_id,
            tessellation=tessellation,
            budget=budget,
            occurrences=geometry.occurrences,
            shells=_surviving_shell(geometry, faces),
        )
        source_gap = _gap_upper(geometry)
        target_gap = _gap_upper(target.require_geometry())
        margin = fraction_up(
            Fraction(margin) + Fraction(source_gap) + Fraction(target_gap)
        )
        volume_gap = 0.0
        enclosure = None
        if coefficients is None or margin:
            coefficients = None
            enclosure, volume_gap = _volume_difference(
                model, target, builder, points, volume_subdivisions, budget
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
            source_model_id=model.model_id,
            removed_face=face,
            model=None,
        )
    face_images = tuple((model.face_ids[a], target.face_ids[b]) for a, b in faces)
    edge_images = tuple((model.edge_ids[a], target.edge_ids[b]) for a, b in edges)
    names = _transport_names(recipe_names, target, faces, edges)
    return BRepDefeatureResult(
        status="certified",
        reason="native local extension and certified shape change",
        operation=operation_,
        source_model_id=model.model_id,
        removed_face=face,
        model=target,
        volume_coefficients=coefficients,
        volume_enclosure=enclosure,
        volume_gap_error_bound=volume_gap,
        hausdorff_enclosure=distance,
        intersection_kind=intersection_kind,
        carrier_intent_points=points,
        intersection_rounding_bounds=rounding,
        g1_angle_bounds=g1[1],
        g1_edges=g1[0],
        face_images=face_images,
        edge_images=edge_images,
        recipe_names=names,
    )


def _single_solid(model: BRepModel, /) -> None:
    geometry = model.require_geometry()
    if (
        len(geometry.solid_shells) != 1
        or len(geometry.shell_faces) != 1
        or len(geometry.solid_shells[0]) != 1
        or len(geometry.occurrences) != 1
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "defeaturing requires one solid occurrence with one shell"
        )
    if any(root is not None for root in geometry.vertex_roots):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "source-root transport needs a separate extension certificate"
        )
    if any(
        not np.array_equal(occurrence.rotation, np.eye(3))
        or not np.array_equal(occurrence.translation, np.zeros(3))
        for occurrence in geometry.occurrences
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "defeaturing requires materialized occurrence placements"
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
    subdivisions: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[float, float], float]:
    a = brep_mass_properties(
        source,
        source.require_geometry(),
        subdivisions=subdivisions,
        coordinate_budget=budget,
    )
    b = brep_mass_properties(
        target,
        target.require_geometry(),
        subdivisions=subdivisions,
        coordinate_budget=budget,
    )
    target_bounds = tuple(float(value) for value in b.volume_enclosures[0])
    intent = _planar_volume_intent(builder, target.require_geometry(), points, budget)
    if intent is not None:
        target_bounds = min(target_bounds[0], intent[0]), max(target_bounds[1], intent[1])
    gap = _gap_flux_bound(source, budget) + _gap_flux_bound(target, budget)
    lo = Fraction(target_bounds[0]) - Fraction(float(a.volume_enclosures[0, 1])) - gap
    hi = Fraction(target_bounds[1]) - Fraction(float(a.volume_enclosures[0, 0])) + gap
    return (fraction_down(lo), fraction_up(hi)), fraction_up(gap)


def _gap_upper(geometry: BRepGeometry, /) -> float:
    endpoints = geometry.edge_vertex_gap_bounds
    faces = geometry.edge_face_gap_bounds
    vertex_bound = (
        max((max(row) for row in endpoints), default=0.0)
        if endpoints is not None
        else 0.0
    )
    face_bound = max(faces, default=0.0) if faces is not None else 0.0
    return fraction_up(Fraction(vertex_bound) + Fraction(face_bound))


def _planar_volume_intent(
    builder: Builder,
    geometry: BRepGeometry,
    points: tuple[_Vector, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[float, float] | None:
    if len(points) != 2 or not all(
        isinstance(face.patch, PlanePatch) for face in builder.faces
    ):
        return None
    if not all(isinstance(curve, LineCurve) for curve in builder.pcurves):
        return None
    vertices = [rational_vector(point) for point in builder.vertices]
    a, b = builder.edge_vertices[-1]
    vertices[a], vertices[b] = points
    normals = []
    for face in builder.faces:
        if not isinstance(face.patch, PlanePatch):
            raise RuntimeError(
                "brep_defeature.publication must retain admitted native evidence; got "
                + "planar volume intent lost its admitted plane carrier"
            )
        normal = _normal(face.patch)
        lower, _ = square_root_interval(_dot(normal, normal))
        # Scaling by the directed lower norm makes the face-distance diagnostic
        # conservative even when the unit normal is not rational.
        normals.append(tuple(value / Fraction(lower) for value in normal))
    evidence, _ = offset_rounding_evidence(
        builder,
        geometry,
        tuple(vertices),
        tuple(normals),
        tuple(face.orientation for face in builder.faces),
        budget=budget,
    )
    enclosure = evidence.volume_enclosure
    if enclosure is None:
        raise RuntimeError(
            "brep_defeature.publication must retain admitted native evidence; got "
            + "brep_defeature.volume must retain native rounding evidence; got an absent enclosure"
        )
    return float(enclosure[0]), float(enclosure[1])


def _interval_norm_upper(lower: np.ndarray, upper: np.ndarray, /) -> Fraction:
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got "
            + "native first-jet bounds do not certify a finite closing-gap flux"
        )
    magnitude = tuple(
        max(abs(Fraction(float(a))), abs(Fraction(float(b))))
        for a, b in zip(lower.reshape(-1), upper.reshape(-1), strict=True)
    )
    return Fraction(square_root_interval(_dot(magnitude, magnitude))[1])


def _gap_flux_bound(model: BRepModel, budget: CoordinateEnclosureBudget, /) -> Fraction:
    """Bound closing-gap flux by ruled bridges, not an invented solid volume.

    For F=x/3, a same-parameter bridge of width d has area at most
    d*(length(source)+length(lift)). Its flux is at most R*area/3.
    Endpoint closing triangles contribute at most 2*d**2 per coedge.
    Native whole-range first jets enclose both lengths independently.
    """
    geometry = model.require_geometry()
    delta = Fraction(_gap_upper(geometry))
    if delta == 0:
        return Fraction()
    budget.reserve(2048 * len(geometry.coedge_edges), 8192 * len(geometry.coedge_edges))
    total = Fraction()
    for face, loops in enumerate(geometry.face_loops):
        active = tuple(
            (coedge, gap)
            for loop in loops
            for coedge in loop
            if (gap := _coedge_gap(geometry, coedge)) > 0
        )
        if not active:
            continue
        patch = model.patches[face]
        box = np.asarray(model.parameter_bounds[face])
        positions = patch.bounding_box(box, budget=budget)
        coordinate = tuple(
            max(abs(Fraction(float(a))), abs(Fraction(float(b)))) + delta
            for a, b in zip(positions[0], positions[1], strict=True)
        )
        radius = Fraction(square_root_interval(_dot(coordinate, coordinate))[1])
        jacobian = patch.derivative_bounds(box, budget=budget)
        for coedge, gap in active:
            length = _gap_bridge_length(geometry, coedge, jacobian, budget)
            total += radius * (gap * length + 2 * gap**2) / 3
    return total


def _coedge_gap(geometry: BRepGeometry, coedge: int, /) -> Fraction:
    edge = geometry.coedge_edges[coedge]
    endpoints, faces = geometry.edge_vertex_gap_bounds, geometry.edge_face_gap_bounds
    vertex = Fraction(max(endpoints[edge])) if endpoints is not None else Fraction()
    face = Fraction(faces[coedge]) if faces is not None else Fraction()
    return vertex + face


def _gap_bridge_length(
    geometry: BRepGeometry,
    coedge: int,
    jacobian: tuple[np.ndarray, np.ndarray],
    budget: CoordinateEnclosureBudget,
    /,
) -> Fraction:
    edge = geometry.coedge_edges[coedge]
    index = geometry.edge_curves[edge]
    if index < 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "closing positive pole-edge gaps requires a separate flux certificate"
        )
    bounds = geometry.edge_parameter_enclosure(edge, budget=budget)
    first, last = float(bounds[0, 0]), float(bounds[1, 1])
    curve = geometry.curves[index]
    pcurve = geometry.pcurves[coedge]
    curve_speed = _interval_norm_upper(
        *curve.derivative_bounds(first, last, budget=budget)
    )
    du, dv = pcurve.derivative_bounds(first, last, budget=budget)
    parameter = tuple(
        max(abs(Fraction(float(a))), abs(Fraction(float(b))))
        for a, b in zip(du, dv, strict=True)
    )
    velocity = tuple(
        sum(
            (
                max(
                    abs(Fraction(float(jacobian[0][row, axis]))),
                    abs(Fraction(float(jacobian[1][row, axis]))),
                )
                * parameter[axis]
                for axis in range(2)
            ),
            Fraction(),
        )
        for row in range(3)
    )
    lift_speed = Fraction(square_root_interval(_dot(velocity, velocity))[1])
    return (Fraction(last) - Fraction(first)) * (curve_speed + lift_speed)


def _transport_names(
    source: RecipeEntityNames | None, target: BRepModel, faces: _Images, edges: _Images, /
) -> RecipeEntityNames | None:
    if source is None:
        return None
    face_names = (
        None
        if source.face_names is None
        else tuple(
            source.face_names[a] for a, _ in sorted(faces, key=lambda pair: pair[1])
        )
    )
    edge_names = (
        None
        if source.edge_names is None or len(edges) != len(target.edge_ids)
        else tuple(
            source.edge_names[a] for a, _ in sorted(edges, key=lambda pair: pair[1])
        )
    )
    return RecipeEntityNames(target, face_names, edge_names)


def _plane(model: BRepModel, face: int, /) -> PlanePatch:
    patch = planar_carrier(model.patches[face])
    if patch is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "plane-pair removal requires analytic planar neighbors and planar end trims"
        )
    return patch


def _normal(patch: PlanePatch, /) -> _Vector:
    return rational_cross(
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )


def _offset(first: _Vector, second: _Vector, /) -> _Vector:
    return tuple(b - a for a, b in zip(first, second, strict=True))


def _length(vector: _Vector, /) -> Fraction | None:
    squared = _dot(vector, vector)
    a, b = isqrt(squared.numerator), isqrt(squared.denominator)
    return (
        Fraction(a, b)
        if a * a == squared.numerator and b * b == squared.denominator
        else None
    )


def _coordinate(
    patch: PlanePatch, point: _Vector, budget: CoordinateEnclosureBudget, /
) -> Point:
    x, y = (
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )
    delta = _offset(rational_vector(np.asarray(patch.origin)), point)
    result = _solve(
        ((_dot(x, x), _dot(x, y)), (_dot(x, y), _dot(y, y))),
        (_dot(x, delta), _dot(y, delta)),
        budget=budget,
    )
    return result[0], result[1]


def _plane_junction(
    model: BRepModel, face: int, operation: BRepDefeatureOperation, /
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    geometry = model.require_geometry()
    patch = model.patches[face]
    if operation == "fillet" and not isinstance(patch, CylinderPatch):
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
    edges = tuple(geometry.coedge_edges[c] for c in geometry.face_loops[face][0])
    adjacent = _blend_neighbors(model, face, edges)
    if len(set(adjacent)) != 4:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "blend joins more than two neighbors or lacks two distinct end trims"
        )
    for i, j in ((0, 2), (1, 3)):
        contact = edges[i], edges[j]
        sides = adjacent[i], adjacent[j]
        if _parallel_contact_lines(model, sides, contact):
            other = tuple(k for k in range(4) if k not in (i, j))
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
    geometry = model.require_geometry()
    direction = rational_cross(*tuple(_normal(_plane(model, side)) for side in sides))
    if not any(direction):
        return False
    for edge in edges:
        index = geometry.edge_curves[edge]
        if index < 0:
            return False
        curve = geometry.curves[index]
        if not isinstance(curve, LineCurve):
            return False
        if any(rational_cross(rational_vector(np.asarray(curve.direction)), direction)):
            return False
    return True


def _sharp_endpoints(
    model: BRepModel,
    sides: tuple[int, int],
    caps: tuple[int, int],
    cap_edges: tuple[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[_Vector, _Vector], dict[int, int]]:
    geometry = model.require_geometry()
    normals = tuple(_normal(_plane(model, face)) for face in sides)
    heights = tuple(
        _dot(normal, rational_vector(np.asarray(_plane(model, face).origin)))
        for face, normal in zip(sides, normals, strict=True)
    )
    points = []
    replacements = {}
    for end, (cap, edge) in enumerate(zip(caps, cap_edges, strict=True)):
        patch = _plane(model, cap)
        normal = _normal(patch)
        point = _solve(
            (*normals, normal),
            (*heights, _dot(normal, rational_vector(np.asarray(patch.origin)))),
            budget=budget,
        )
        points.append(point)
        for vertex in geometry.edge_vertices[edge]:
            replacements[vertex] = end
    if len(replacements) != 4 or points[0] == points[1]:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "blend has collapsed or ambiguous endpoint incidence"
        )
    return (points[0], points[1]), replacements


def _literal_face_loops(
    model: BRepModel, face: int, /
) -> tuple[tuple[_Vector, ...], ...]:
    geometry = model.require_geometry()
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
                    "brep_defeature.certificate must decide the requested operation; got "
                    + "extension overlaps a face without literal planar polygon collision support"
                )
            vertices = geometry.edge_vertices[edge]
            points.append(
                rational_vector(
                    np.asarray(
                        geometry.vertex_points[
                            vertices[0 if geometry.coedge_senses[coedge] == 1 else 1]
                        ]
                    )
                )
            )
        result.append(tuple(points))
    return tuple(result)


def _contains(point: Point, loops: tuple[tuple[Point, ...], ...], /) -> bool:
    boundary = any(
        on_segment(point, (a, b))
        for loop in loops
        for a, b in zip(loop, loop[1:] + loop[:1], strict=True)
    )
    return boundary or (
        inside_loop(point, loops[0])
        and not any(inside_loop(point, hole) for hole in loops[1:])
    )


def _plane_cut(
    loops: tuple[tuple[_Vector, ...], ...], normal: _Vector, height: Fraction, /
) -> tuple[_Vector, ...]:
    points = set()
    for loop in loops:
        for first, last in zip(loop, loop[1:] + loop[:1], strict=True):
            a, b = _dot(normal, first) - height, _dot(normal, last) - height
            if a == 0:
                points.add(first)
            if a * b < 0:
                points.add(
                    tuple(
                        x + a / (a - b) * (y - x)
                        for x, y in zip(first, last, strict=True)
                    )
                )
    return tuple(points)


def _polygon_contact(
    first_patch: PlanePatch,
    first: tuple[tuple[_Vector, ...], ...],
    second_patch: PlanePatch,
    second: tuple[tuple[_Vector, ...], ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    n, m = _normal(first_patch), _normal(second_patch)
    h, k = (
        _dot(n, rational_vector(np.asarray(first_patch.origin))),
        _dot(m, rational_vector(np.asarray(second_patch.origin))),
    )
    direction = rational_cross(n, m)
    a = _project_loops(first_patch, first, budget)
    if not any(direction):
        if any(_dot(n, p) != h for loop in second for p in loop):
            return False
        return _coplanar_contact(a, _project_loops(first_patch, second, budget))
    candidates = sorted(
        set(_plane_cut(first, m, k)) | set(_plane_cut(second, n, h)),
        key=lambda p: _dot(p, direction),
    )
    b = _project_loops(second_patch, second, budget)
    probes = candidates + [
        tuple((x + y) / 2 for x, y in zip(p, q, strict=True))
        for p, q in zip(candidates, candidates[1:])
    ]
    return any(
        _contains(_coordinate(first_patch, p, budget), a)
        and _contains(_coordinate(second_patch, p, budget), b)
        for p in probes
        if _dot(n, p) == h and _dot(m, p) == k
    )


def _project_loops(
    patch: PlanePatch,
    loops: tuple[tuple[_Vector, ...], ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Point, ...], ...]:
    return tuple(
        tuple(_coordinate(patch, point, budget) for point in loop) for loop in loops
    )


def _coplanar_contact(
    first: tuple[tuple[Point, ...], ...], second: tuple[tuple[Point, ...], ...], /
) -> bool:
    segments = tuple(
        (a, b) for loop in first for a, b in zip(loop, loop[1:] + loop[:1], strict=True)
    )
    others = tuple(
        (a, b) for loop in second for a, b in zip(loop, loop[1:] + loop[:1], strict=True)
    )
    return (
        any(intersections(a, b) for a in segments for b in others)
        or _contains(first[0][0], second)
        or _contains(second[0][0], first)
    )


def _boxes_separate(
    first: tuple[_Vector, ...], lower: np.ndarray, upper: np.ndarray, /
) -> bool:
    return any(
        max(p[axis] for p in first) < Fraction(float(lower[axis]))
        or min(p[axis] for p in first) > Fraction(float(upper[axis]))
        for axis in range(3)
    )


def _certify_extensions(
    model: BRepModel,
    removed: int,
    sides: tuple[int, int],
    caps: tuple[int, int],
    contacts: tuple[int, int],
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    geometry = model.require_geometry()
    strips = []
    for side, edge in zip(sides, contacts, strict=True):
        vertices = sorted(geometry.edge_vertices[edge], key=lambda v: replacements[v])
        old = tuple(
            rational_vector(np.asarray(geometry.vertex_points[v])) for v in vertices
        )
        strips.append((side, (old[0], old[1], points[1], points[0])))
    for face, patch in enumerate(model.patches):
        if face in (removed, *sides, *caps):
            continue
        bounds = patch.bounding_box(
            np.asarray(model.parameter_bounds[face]), budget=budget
        )
        for side, strip in strips:
            _certify_strip(model, face, patch, bounds, side, strip, caps[0], budget)


def _certify_strip(
    model: BRepModel,
    face: int,
    patch: AbstractSurfacePatch,
    bounds: np.ndarray,
    side: int,
    strip: tuple[_Vector, ...],
    cap: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    if _boxes_separate(strip, bounds[0], bounds[1]):
        return
    planar = planar_carrier(patch)
    if planar is None:
        if isinstance(patch, CylinderPatch) and _cylinder_extension_contact(
            model, face, cap, strip[0], strip[3], budget
        ):
            raise _Collision(
                "brep_defeature.extension must avoid unrelated faces; got a certified native line/circle intersection"
            )
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got curved face overlap without a local Boolean clipping certificate"
        )
    if _polygon_contact(
        _plane(model, side), (strip,), planar, _literal_face_loops(model, face), budget
    ):
        raise _Collision(
            "brep_defeature.extension must avoid unrelated faces; got a certified native planar Boolean intersection"
        )


def _cylinder_extension_contact(
    model: BRepModel,
    face: int,
    cap: int,
    first: _Vector,
    last: _Vector,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    geometry = model.require_geometry()
    patch = _plane(model, cap)
    a, b = _coordinate(patch, first, budget), _coordinate(patch, last, budget)
    if any(Fraction(float(value)) != value for point in (a, b) for value in point):
        return False
    line = _profile_edge(
        (float(a[0]), float(a[1])), (float(b[0]), float(b[1])), ProfileLine()
    )
    for loop in geometry.face_loops[cap]:
        for coedge in loop:
            edge = geometry.coedge_edges[coedge]
            if face not in model.topology.edge_faces[edge]:
                continue
            pcurve = geometry.pcurves[coedge]
            if not isinstance(pcurve, CircleCurve):
                continue
            _full_turn(geometry, edge)
            center = tuple(float(value) for value in pcurve.center)
            arc = _profile_edge(
                (center[0] + float(pcurve.radius), center[1]),
                (center[0] + float(pcurve.radius), center[1]),
                ProfileArc((center[0], center[1])),
            )
            if _line_arc_contact(line, arc, frozenset()):
                return True
    return False


def _affine_edge(curve: BRepCurve, /) -> tuple[_Vector, _Vector] | None:
    if not isinstance(curve, AbstractCurve):
        return None
    return affine_curve_coefficients(
        curve, np.zeros((curve.ambient_dimension,), dtype=np.float64)
    )


def _stage_sharp_edges(
    model: BRepModel,
    removed_edges: set[int],
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> tuple[Builder, dict[int, int], int]:
    geometry = model.require_geometry()
    builder = Builder()
    vertices = {
        v: builder.vertex(np.asarray(point))
        for v, point in enumerate(geometry.vertex_points)
        if v not in replacements
    }
    sharp_vertices = tuple(
        builder.vertex(np.asarray(tuple(float(value) for value in p), dtype=np.float64))
        for p in points
    )
    for old, end in replacements.items():
        vertices[old] = sharp_vertices[end]
    edges = {}
    for source in range(len(geometry.edge_vertices)):
        if source not in removed_edges:
            edges[source] = _stage_retained_edge(
                builder, geometry, source, vertices, points, replacements
            )
    direction = _offset(points[0], points[1])
    sharp = builder.edge(
        LineCurve(tuple(float(v) for v in points[0]), tuple(float(v) for v in direction)),
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
        first, last = 0.0, 1.0
    else:
        curve = geometry.curves[curve_index]
    target = builder.edge(curve, first, last, vertices[start], vertices[end])
    builder.edge_endpoint_roots[target] = (
        (None, None) if changed else _native_roots(geometry, source)
    )
    return target


def _extended_edge_curve(
    geometry: BRepGeometry,
    edge: int,
    points: tuple[_Vector, _Vector],
    replacements: dict[int, int],
    /,
) -> LineCurve:
    start, end = geometry.edge_vertices[edge]
    source = _affine_edge(geometry.curves[geometry.edge_curves[edge]])
    if source is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "local removal requires analytic affine endpoint trims"
        )
    a = (
        points[replacements[start]]
        if start in replacements
        else rational_vector(np.asarray(geometry.vertex_points[start]))
    )
    b = (
        points[replacements[end]]
        if end in replacements
        else rational_vector(np.asarray(geometry.vertex_points[end]))
    )
    origin, direction = source
    for point in (a, b):
        if any(rational_cross(_offset(origin, point), direction)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got a retained trim outside the rational analytic intersection"
            )
    if _dot(_offset(a, b), direction) <= 0:
        raise _Collision(
            "brep_defeature.extension must avoid unrelated faces and reversed trims; got "
            + "extension reverses a neighboring edge or makes a face vanish"
        )
    return LineCurve(tuple(float(v) for v in a), tuple(float(v) for v in _offset(a, b)))


def _plane_pcurve(
    patch: PlanePatch, curve: BRepCurve, budget: CoordinateEnclosureBudget, /
) -> LineCurve:
    source = _affine_edge(curve)
    if source is None:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "rebuilt plane p-curve requires a native affine carrier"
        )
    origin, direction = source
    start = _coordinate(patch, origin, budget)
    end = _coordinate(
        patch,
        tuple(a + b for a, b in zip(origin, direction, strict=True)),
        budget,
    )
    return LineCurve(
        tuple(float(v) for v in start),
        tuple(float(b - a) for a, b in zip(start, end, strict=True)),
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
    budget: CoordinateEnclosureBudget,
    /,
) -> _Images:
    geometry = model.require_geometry()
    images = []
    for source, patch in enumerate(model.patches):
        if source == removed:
            continue
        native_plane = planar_carrier(patch)
        converted = native_plane is not None and not isinstance(patch, PlanePatch)
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
                    budget,
                )
            )
        images.append(
            (
                source,
                builder.face(
                    patch,
                    box,
                    loops,
                    model.physical_tags[source],
                    None,
                    int(np.asarray(model.orientation)[source]),
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
    budget: CoordinateEnclosureBudget,
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
                geometry, coedge, patch, replacements, builder, sharp, budget
            )
        else:
            target = edges[edge]
            pcurve = _retained_pcurve(
                geometry,
                coedge,
                patch,
                converted,
                replacements,
                builder.curves[builder.edge_curves[target]],
                budget,
            )
        staged.append(builder.coedge(target, sense, pcurve, budget=budget))
        if isinstance(pcurve, LineCurve):
            _extend_parameter_box(box, pcurve, builder.edge_ranges[target])
    return staged


def _retained_pcurve(
    geometry: BRepGeometry,
    coedge: int,
    patch: AbstractSurfacePatch,
    converted: bool,
    replacements: dict[int, int],
    curve: BRepCurve,
    budget: CoordinateEnclosureBudget,
    /,
) -> BRepPCurve:
    edge = geometry.coedge_edges[coedge]
    first, last = geometry.edge_vertices[edge]
    if not converted and first not in replacements and last not in replacements:
        return geometry.pcurves[coedge]
    if not isinstance(patch, PlanePatch):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "rebuilt affine trims require a planar native p-curve"
        )
    return _plane_pcurve(patch, curve, budget)


def _extend_parameter_box(
    box: np.ndarray, pcurve: LineCurve, bounds: tuple[float, float], /
) -> None:
    origin, direction = (
        rational_vector(np.asarray(pcurve.origin)),
        rational_vector(np.asarray(pcurve.direction)),
    )
    for axis in range(2):
        values = tuple(
            origin[axis] + Fraction(parameter) * direction[axis] for parameter in bounds
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
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[int, int, BRepPCurve]:
    edge = geometry.coedge_edges[coedge]
    first, last = geometry.edge_vertices[edge]
    sense = geometry.coedge_senses[coedge] * (
        1 if replacements[first] == 0 and replacements[last] == 1 else -1
    )
    curve = builder.curves[builder.edge_curves[sharp]]
    if not isinstance(patch, PlanePatch) or not isinstance(curve, LineCurve):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "sharp intersection p-curve requires analytic plane/line support"
        )
    return sharp, sense, _plane_pcurve(patch, curve, budget)


def _shape_change(
    model: BRepModel,
    face: int,
    sides: tuple[int, int],
    caps: tuple[int, int],
    cap_edges: tuple[int, int],
    points: tuple[_Vector, _Vector],
    operation: BRepDefeatureOperation,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Fraction, Fraction, Fraction] | None, tuple[float, float]]:
    patch = model.patches[face]
    radius = (
        Fraction(float(patch.radius)) if isinstance(patch, CylinderPatch) else Fraction()
    )
    bounds = patch.bounding_box(np.asarray(model.parameter_bounds[face]), budget=budget)
    diagonal = tuple(
        max(Fraction(float(bounds[1, axis])), *(point[axis] for point in points))
        - min(Fraction(float(bounds[0, axis])), *(point[axis] for point in points))
        for axis in range(3)
    )
    # All changed face regions lie in this local container; either boundary
    # supplies a point in it, so its diameter bounds their Hausdorff distance.
    bound = square_root_interval(_dot(diagonal, diagonal))[1]
    coefficients = None
    n, m = (_normal(_plane(model, side)) for side in sides)
    vector = _offset(points[0], points[1])
    perpendicular = all(
        not any(rational_cross(_normal(_plane(model, cap)), vector)) for cap in caps
    )
    length = _length(vector)
    if (
        _dot(n, m) == 0
        and perpendicular
        and length is not None
        and (operation != "fillet" or _plane_pair_tangent(model, face, sides, budget))
    ):
        return _right_angle_change(
            model, caps[0], cap_edges[0], points[0], radius, length, operation
        )
    return coefficients, (0.0, bound)


def _plane_pair_tangent(
    model: BRepModel,
    face: int,
    sides: tuple[int, int],
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    patch = model.patches[face]
    if not isinstance(patch, CylinderPatch):
        return False
    if (
        exact_frame_orientation(
            patch.first_axis, patch.second_axis, patch.axis, budget=budget
        )
        is None
    ):
        return False
    center = rational_vector(np.asarray(patch.origin))
    radius = Fraction(float(patch.radius))
    for side in sides:
        plane = _plane(model, side)
        normal = _normal(plane)
        distance = _dot(
            normal, _offset(rational_vector(np.asarray(plane.origin)), center)
        )
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
    geometry = model.require_geometry()
    first, last = geometry.edge_vertices[cap_edge]
    a, b = (rational_vector(np.asarray(geometry.vertex_points[v])) for v in (first, last))
    cross = rational_cross(_offset(a, corner), _offset(a, b))
    coedge = next(
        c
        for loop in geometry.face_loops[cap]
        for c in loop
        if geometry.coedge_edges[c] == cap_edge
    )
    sign = (
        1
        if _dot(cross, _normal(_plane(model, cap))) * geometry.coedge_senses[coedge] > 0
        else -1
    )
    if operation == "fillet":
        r2 = radius**2
        lo, hi = square_root_interval(Fraction(2))
        return (sign * r2 * length, -sign * r2 * length / 4, Fraction()), (
            fraction_down(radius * (Fraction(lo) - 1)),
            fraction_up(radius * (Fraction(hi) - 1)),
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
    return coefficients, square_root_interval(squared)


def _remove_plane_blend(
    model: BRepModel,
    face: int,
    operation: BRepDefeatureOperation,
    cells: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> _BlendRemoval:
    sides, caps, contacts, cap_edges = _plane_junction(model, face, operation)
    points, replacements = _sharp_endpoints(model, sides, caps, cap_edges, budget)
    _certify_extensions(model, face, sides, caps, contacts, points, replacements, budget)
    g1: _G1Evidence = (), jnp.empty((0, 2), dtype=jnp.float64)
    if operation == "fillet":
        g1 = _g1_evidence(model, contacts, cells, budget)
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
        model, face, contacts, cap_edges, replacements, builder, edges, sharp, budget
    )
    builder.recompute_endpoint_gaps = True
    coefficients, distance = _shape_change(
        model, face, sides, caps, cap_edges, points, operation, budget
    )
    return builder, faces, tuple(edges.items()), coefficients, distance, points, g1


def _g1_evidence(
    model: BRepModel,
    edges: tuple[int, int],
    cells: int,
    budget: CoordinateEnclosureBudget,
    /,
) -> _G1Evidence:
    records = brep_edge_classification(
        model, model.require_geometry(), maximum_cells=cells, coordinate_budget=budget
    )
    selected = tuple(records[edge] for edge in sorted(edges))
    return tuple(record.edge_id for record in selected), jnp.stack(
        tuple(record.dihedral_range for record in selected)
    )


def _copy_retained(
    model: BRepModel, retained: tuple[tuple[int, tuple[tuple[int, ...], ...]], ...], /
) -> tuple[Builder, _Images]:
    geometry = model.require_geometry()
    builder = Builder()
    used = sorted(
        {
            geometry.coedge_edges[c]
            for _, loops in retained
            for loop in loops
            for c in loop
        }
    )
    vertices = sorted({v for edge in used for v in geometry.edge_vertices[edge]})
    vertex_map = {
        source: builder.vertex(np.asarray(geometry.vertex_points[source]))
        for source in vertices
    }
    edge_map = {}
    for source in used:
        curve_index = geometry.edge_curves[source]
        if curve_index < 0:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "unrelated pole-edge transport is unsupported"
            )
        first, last = (float(value) for value in geometry.edge_ranges[source])
        start, end = geometry.edge_vertices[source]
        edge_map[source] = builder.edge(
            geometry.curves[curve_index], first, last, vertex_map[start], vertex_map[end]
        )
        builder.edge_endpoint_roots[-1] = _native_roots(geometry, source)
    for source, loops in retained:
        staged = []
        for loop in loops:
            coedges = []
            for coedge in loop:
                target = len(builder.coedge_edges)
                builder.coedge_edges.append(edge_map[geometry.coedge_edges[coedge]])
                builder.coedge_senses.append(geometry.coedge_senses[coedge])
                builder.pcurves.append(geometry.pcurves[coedge])
                builder.coedge_endpoint_roots.append(
                    geometry.coedge_endpoint_roots[coedge]
                )
                coedges.append(target)
            staged.append(coedges)
        builder.face(
            model.patches[source],
            np.asarray(model.parameter_bounds[source]),
            staged,
            model.physical_tags[source],
            None,
            int(np.asarray(model.orientation)[source]),
        )
    return builder, tuple(edge_map.items())


def _native_roots(
    geometry: BRepGeometry, edge: int, /
) -> tuple[NativePeriodEndpoint | None, NativePeriodEndpoint | None]:
    first, last = geometry.edge_endpoint_roots[edge]
    if (first is not None and not isinstance(first, NativePeriodEndpoint)) or (
        last is not None and not isinstance(last, NativePeriodEndpoint)
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "unrelated nonnative endpoint-root transport is unsupported"
        )
    return first, last


def _full_turn(geometry: BRepGeometry, edge: int, /) -> None:
    first, last = geometry.edge_endpoint_roots[edge]
    if (
        not isinstance(first, NativePeriodEndpoint)
        or not isinstance(last, NativePeriodEndpoint)
        or last.turns - first.turns != 1
        or first.rational != last.rational
        or geometry.edge_vertices[edge][0] != geometry.edge_vertices[edge][1]
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got circular endpoints without native full-turn evidence"
        )


def _hole_boundary(
    model: BRepModel, face: int, /
) -> tuple[tuple[int, int], tuple[int, int]]:
    geometry = model.require_geometry()
    wall_edges = {
        geometry.coedge_edges[c] for loop in geometry.face_loops[face] for c in loop
    }
    caps = sorted(
        {other for edge in wall_edges for other in model.topology.edge_faces[edge]}
        - {face}
    )
    if len(caps) != 2:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires one wall between exactly two planar hosts"
        )
    edges = []
    for cap in caps:
        _plane(model, cap)
        inner = [
            loop
            for loop in geometry.face_loops[cap][1:]
            if any(geometry.coedge_edges[c] in wall_edges for c in loop)
        ]
        if len(inner) != 1 or len(inner[0]) != 1:
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got hosts without one full circular inner loop each"
            )
        edge = geometry.coedge_edges[inner[0][0]]
        _full_turn(geometry, edge)
        edges.append(edge)
    return (caps[0], caps[1]), (edges[0], edges[1])


def _frame(
    patch: CylinderPatch | ConePatch | TorusPatch, budget: CoordinateEnclosureBudget, /
) -> tuple[_Vector, _Vector, _Vector]:
    x, y, axis = (
        rational_vector(np.asarray(value))
        for value in (patch.first_axis, patch.second_axis, patch.axis)
    )
    if (
        exact_frame_orientation(
            patch.first_axis, patch.second_axis, patch.axis, budget=budget
        )
        is None
    ):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "circular removal requires an orthonormal rational frame"
        )
    return x, y, axis


def _cap_hole(
    model: BRepModel, face: int, budget: CoordinateEnclosureBudget, /
) -> _HoleRemoval:
    patch = model.patches[face]
    if not isinstance(patch, (CylinderPatch, ConePatch)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires one cylindrical or conical wall"
        )
    x, y, axis = _frame(patch, budget)
    caps, edges = _hole_boundary(model, face)
    geometry = model.require_geometry()
    if int(np.asarray(model.orientation)[face]) * _dot(rational_cross(x, y), axis) >= 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires an inward material boundary, not an exterior boss"
        )
    heights, radii = _hole_sections(model, patch, caps, edges, axis, budget)
    coefficients = None
    height = abs(heights[1] - heights[0])
    if height == 0:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "hole capping requires distinct host planes"
        )
    if isinstance(patch, CylinderPatch):
        radius = Fraction(float(patch.radius))
        if any(r != radius for r in radii):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "cylindrical cap radii must agree exactly with the wall"
            )
        coefficients = Fraction(), radius**2 * height, Fraction()
    container_bound = _hole_clearance(model, face, caps, axis, budget)
    retained = _retained_hole_faces(geometry, face)
    builder, edge_images = _copy_retained(model, retained)
    return (
        builder,
        tuple((source, target) for target, (source, _) in enumerate(retained)),
        edge_images,
        coefficients,
        (0.0, fraction_up(max(*radii, height / 2)))
        if isinstance(patch, CylinderPatch)
        else (0.0, container_bound),
    )


def _retained_hole_faces(
    geometry: BRepGeometry, face: int, /
) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    wall_edges = {
        geometry.coedge_edges[c] for loop in geometry.face_loops[face] for c in loop
    }
    return tuple(
        (
            source,
            tuple(
                loop
                for loop in loops
                if not any(geometry.coedge_edges[c] in wall_edges for c in loop)
            ),
        )
        for source, loops in enumerate(geometry.face_loops)
        if source != face
    )


def _hole_clearance(
    model: BRepModel,
    face: int,
    caps: tuple[int, int],
    axis: _Vector,
    budget: CoordinateEnclosureBudget,
    /,
) -> float:
    wall = model.patches[face]
    if not isinstance(wall, (CylinderPatch, ConePatch)):
        raise RuntimeError(
            "brep_defeature.hole must retain its admitted wall; got another carrier family"
        )
    origin = rational_vector(np.asarray(wall.origin))
    box = wall.bounding_box(np.asarray(model.parameter_bounds[face]), budget=budget)
    corners = _box_corners(box)
    radial_upper = _box_radial_upper(corners, origin, axis)
    axial = tuple(_dot(_offset(origin, point), axis) for point in corners)
    heights = min(axial), max(axial)
    for other, patch in enumerate(model.patches):
        if other in (face, *caps) or _axial_plane_separate(patch, axis, origin, heights):
            continue
        bounds = patch.bounding_box(
            np.asarray(model.parameter_bounds[other]), budget=budget
        )
        if _boxes_separate(corners, bounds[0], bounds[1]):
            continue
        if _coaxial_cylinder_separate(patch, origin, axis, radial_upper, budget):
            continue
        candidate = _box_corners(bounds)
        if _box_inside_cylindrical_fill(wall, candidate, origin, axis, heights):
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
    wall: CylinderPatch | ConePatch,
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
    projection = tuple(_dot(_offset(origin, point), axis) for point in corners)
    return min(projection) >= heights[0] and max(projection) <= heights[1]


def _box_corners(box: np.ndarray, /) -> tuple[_Vector, ...]:
    return tuple(
        tuple(Fraction(float(box[side, axis])) for axis, side in enumerate(sides))
        for sides in product((0, 1), repeat=3)
    )


def _box_radial_upper(
    corners: tuple[_Vector, ...], origin: _Vector, axis: _Vector, /
) -> Fraction:
    # The squared norm of a linear projection is convex; its box maximum
    # occurs at a corner. These are support extrema, not geometry samples.
    squared = []
    for point in corners:
        delta = _offset(origin, point)
        radial = tuple(
            value - _dot(delta, axis) * direction
            for value, direction in zip(delta, axis, strict=True)
        )
        squared.append(_dot(radial, radial))
    return Fraction(square_root_interval(max(squared))[1])


def _coaxial_cylinder_separate(
    patch: AbstractSurfacePatch,
    origin: _Vector,
    axis: _Vector,
    radius: Fraction,
    budget: CoordinateEnclosureBudget,
    /,
) -> bool:
    if not isinstance(patch, CylinderPatch):
        return False
    if (
        exact_frame_orientation(
            patch.first_axis, patch.second_axis, patch.axis, budget=budget
        )
        is None
    ):
        return False
    if any(rational_cross(rational_vector(np.asarray(patch.axis)), axis)):
        return False
    if any(
        rational_cross(_offset(origin, rational_vector(np.asarray(patch.origin))), axis)
    ):
        return False
    return Fraction(float(patch.radius)) > radius


def _hole_sections(
    model: BRepModel,
    wall: CylinderPatch | ConePatch,
    caps: tuple[int, int],
    edges: tuple[int, int],
    axis: _Vector,
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[tuple[Fraction, ...], tuple[Fraction, ...]]:
    geometry = model.require_geometry()
    origin = rational_vector(np.asarray(wall.origin))
    heights, radii = [], []
    for cap, edge in zip(caps, edges, strict=True):
        plane = _plane(model, cap)
        normal = _normal(plane)
        if any(rational_cross(normal, axis)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole capping requires host planes normal to the wall axis"
            )
        z = _dot(
            normal, _offset(origin, rational_vector(np.asarray(plane.origin)))
        ) / _dot(normal, axis)
        heights.append(z)
        curve = geometry.curves[geometry.edge_curves[edge]]
        if not isinstance(curve, CircleCurve):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole capping requires native circle carriers on its hosts"
            )
        if (
            exact_frame_orientation(
                curve.first_axis, curve.second_axis, wall.axis, budget=budget
            )
            is None
        ):
            raise _Refusal(
                "brep_defeature.hole must have circular host loops; got a non-orthonormal circle frame"
            )
        center = tuple(a + z * b for a, b in zip(origin, axis, strict=True))
        if center != rational_vector(np.asarray(curve.center)):
            raise _Refusal(
                "brep_defeature.family must have a supported analytic certificate; got "
                + "hole cap is not concentric with its analytic wall"
            )
        radii.append(Fraction(float(curve.radius)))
    return tuple(heights), tuple(radii)


def _torus_junction(model: BRepModel, face: int, /) -> tuple[int, int, int, int, int]:
    geometry = model.require_geometry()
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
    match model.patches[first], model.patches[second]:
        case PlanePatch(), CylinderPatch():
            plane, cylinder = first, second
        case CylinderPatch(), PlanePatch():
            plane, cylinder = second, first
        case _:
            raise _Refusal(
                "brep_defeature.neighbors must be a plane and a cylinder; got an unsupported pair, including cylinder/cylinder quartics"
            )
    plane_edges, cylinder_edges = neighbors[plane], neighbors[cylinder]
    if len(plane_edges) != 1 or len(cylinder_edges) != 1:
        raise _Refusal(
            "brep_defeature.boundaries must have one circle per neighbor; got a split circular boundary"
        )
    seam = _closed_cylinder_seam(geometry, cylinder)
    _full_turn(geometry, plane_edges[0])
    _full_turn(geometry, cylinder_edges[0])
    return plane, cylinder, plane_edges[0], cylinder_edges[0], seam


def _closed_cylinder_seam(geometry: BRepGeometry, face: int, /) -> int:
    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus cylinder neighbor requires one native rectangular chart"
        )
    edges = tuple(geometry.coedge_edges[c] for c in loops[0])
    seams = tuple(edge for edge in set(edges) if edges.count(edge) == 2)
    if len(seams) != 1:
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus cylinder neighbor requires one identity-shared seam"
        )
    return seams[0]


def _circular_clearance(
    model: BRepModel,
    removed: int,
    plane_face: int,
    cylinder_face: int,
    center: _Vector,
    radius: Fraction,
    old_radius: Fraction,
    height_range: tuple[Fraction, Fraction],
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    cylinder = model.patches[cylinder_face]
    if not isinstance(cylinder, CylinderPatch):
        raise RuntimeError(
            "brep_defeature.publication must retain admitted native evidence; got "
            + "defeaturing cylinder certificate lost its owner"
        )
    axis = rational_vector(np.asarray(cylinder.axis))
    extent = max(radius, old_radius)
    low, high = min(height_range), max(height_range)
    points = tuple(
        tuple(c + z * k + s * extent for c, k in zip(center, axis, strict=True))
        for z in (low, high)
        for s in (-1, 1)
    )
    _plane_circle_clearance(model, plane_face, center, radius, old_radius)
    for face, patch in enumerate(model.patches):
        if face in (removed, plane_face, cylinder_face):
            continue
        if _axial_plane_separate(patch, axis, center, (low, high)):
            continue
        bounds = patch.bounding_box(
            np.asarray(model.parameter_bounds[face]), budget=budget
        )
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
    z = _dot(normal, _offset(center, rational_vector(np.asarray(plane.origin)))) / _dot(
        normal, axis
    )
    return z < heights[0] or z > heights[1]


def _plane_circle_clearance(
    model: BRepModel,
    face: int,
    center: _Vector,
    radius: Fraction,
    old_radius: Fraction,
    /,
) -> None:
    geometry = model.require_geometry()
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
        if rational_vector(np.asarray(curve.center)) != center:
            raise _Unresolved(
                "brep_defeature.certificate must decide the requested operation; got a nonconcentric trim without a native circular Boolean certificate"
            )
        r = Fraction(float(curve.radius))
        if r != old_radius and min(radius, old_radius) <= r <= max(radius, old_radius):
            raise _Collision(
                "brep_defeature.extension must avoid unrelated faces and reversed trims; got "
                + "circular extension intersects another native circular plane trim"
            )


def _prepare_torus(
    model: BRepModel, face: int, budget: CoordinateEnclosureBudget, /
) -> _TorusIntent:
    patch = model.patches[face]
    if not isinstance(patch, TorusPatch):
        raise RuntimeError(
            "brep_defeature.publication must retain admitted native evidence; got "
            + "torus removal lost its selected carrier"
        )
    plane_face, cylinder_face, plane_edge, cylinder_edge, seam_edge = _torus_junction(
        model, face
    )
    cylinder = model.patches[cylinder_face]
    if not isinstance(cylinder, CylinderPatch):
        raise RuntimeError(
            "brep_defeature.publication must retain admitted native evidence; got "
            + "torus removal lost its cylinder carrier"
        )
    x, y, axis = _frame(cylinder, budget)
    tx, ty, ta = _frame(patch, budget)
    plane = _plane(model, plane_face)
    if axis != ta or x != tx or y != ty or any(rational_cross(_normal(plane), axis)):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus removal requires a plane normal to the shared cylinder/torus frame"
        )
    origin = rational_vector(np.asarray(cylinder.origin))
    z = _dot(
        _normal(plane), _offset(origin, rational_vector(np.asarray(plane.origin)))
    ) / _dot(_normal(plane), axis)
    center = tuple(a + z * b for a, b in zip(origin, axis, strict=True))
    r = Fraction(float(patch.minor_radius))
    radius = Fraction(float(cylinder.radius))
    major = Fraction(float(patch.major_radius))
    torus_center = rational_vector(np.asarray(patch.center))
    delta = _offset(center, torus_center)
    if (
        any(rational_cross(delta, axis))
        or abs(_dot(delta, axis)) != r
        or abs(radius - major) != r
    ):
        raise _Unresolved(
            "brep_defeature.certificate must decide the requested operation; got torus coefficients inconsistent with the signed-distance G1 equations"
        )
    geometry = model.require_geometry()
    plane_curve = geometry.curves[geometry.edge_curves[plane_edge]]
    if not isinstance(plane_curve, CircleCurve):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got "
            + "torus plane boundary requires its native circle carrier"
        )
    old_radius = Fraction(float(plane_curve.radius))
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
        c
        for loop in geometry.face_loops[face]
        for c in loop
        if geometry.coedge_edges[c] == edge
    )
    pcurve = geometry.pcurves[coedge]
    source = pcurve.source_curve if isinstance(pcurve, PeriodicPCurve) else pcurve
    if not isinstance(source, LineCurve) or not np.array_equal(
        np.asarray(source.direction), (1.0, 0.0)
    ):
        raise _Refusal(
            "brep_defeature.circular_boundary must use a native angular isoline; got another p-curve family or direction"
        )
    return Fraction(float(source.origin[1]))


def _remove_torus(
    model: BRepModel, face: int, budget: CoordinateEnclosureBudget, /
) -> _BlendRemoval:
    intent = _prepare_torus(model, face, budget)
    plane_face, cylinder_face = intent.plane_face, intent.cylinder_face
    plane_edge, cylinder_edge, seam_edge = (
        intent.plane_edge,
        intent.cylinder_edge,
        intent.seam_edge,
    )
    plane, cylinder = intent.plane, intent.cylinder
    center, z, radius, r = intent.center, intent.z, intent.radius, intent.minor_radius
    old_radius, old_z, phase_error = intent.old_radius, intent.old_z, intent.phase_error
    geometry = model.require_geometry()
    _circular_clearance(
        model,
        face,
        plane_face,
        cylinder_face,
        center,
        radius,
        old_radius,
        (Fraction(), old_z - z),
        budget,
    )
    removed_edges = {
        geometry.coedge_edges[c] for loop in geometry.face_loops[face] for c in loop
    }
    retained = tuple(
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
    builder, edge_images = _copy_retained(model, retained)
    face_map = {source: target for target, (source, _) in enumerate(retained)}
    edge_map = dict(edge_images)
    circle = CircleCurve(
        tuple(float(v) for v in center),
        cylinder.first_axis,
        cylinder.second_axis,
        float(radius),
    )
    vertex = builder.vertex(
        np.asarray(circle.center) + float(radius) * np.asarray(circle.first_axis)
    )
    new_edge = builder.edge(circle, 0.0, 2 * pi, vertex, vertex)
    region = SurfaceRegion(
        plane, np.asarray(model.parameter_bounds[plane_face]), budget=budget
    )
    plane_pcurve = native_edge_pcurve(circle, region)
    if not isinstance(plane_pcurve, CircleCurve):
        raise _Refusal(
            "brep_defeature.family must have a supported analytic certificate; got a plane/cylinder intersection circle without a native plane p-curve"
        )
    plane_bounds = plane_pcurve.bounding_box(0.0, 2 * pi, budget=budget)
    builder.faces[face_map[plane_face]].box = np.stack(
        (
            np.minimum(builder.faces[face_map[plane_face]].box[0], plane_bounds[0]),
            np.maximum(builder.faces[face_map[plane_face]].box[1], plane_bounds[1]),
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
        budget,
    )
    _extend_cylinder_seam(
        cylinder_face, seam_edge, edge_map, builder, face_map, old_z, z, vertex
    )
    _compact_staging(builder)
    builder.native_phase(new_edge, cylinder, 0, budget=budget)
    lo, hi = square_root_interval(Fraction(2))
    distance = (
        max(0.0, fraction_down(r * (Fraction(lo) - 1) - phase_error)),
        fraction_up(r * (Fraction(hi) - 1) + phase_error),
    )
    return (
        builder,
        tuple(face_map.items()),
        edge_images,
        None,
        distance,
        (center,),
        _g1_evidence(model, (plane_edge, cylinder_edge), 1, budget),
    )


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
        admitted = [
            (k, interval)
            for k, interval in candidates
            if interval[0] <= coefficient <= interval[1]
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
    plane_pcurve: BRepPCurve,
    z: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    geometry = model.require_geometry()
    for source in (plane_face, cylinder_face):
        loops = []
        for loop in geometry.face_loops[source]:
            staged = []
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if edge in (plane_edge, cylinder_edge):
                    target = circle
                    pcurve = (
                        plane_pcurve
                        if source == plane_face
                        else LineCurve((0.0, z), (1.0, 0.0))
                    )
                else:
                    target = edges[edge]
                    pcurve = geometry.pcurves[coedge]
                staged.append(
                    builder.coedge(
                        target, geometry.coedge_senses[coedge], pcurve, budget=budget
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
    builder.edge_ranges[target] = updated[0], updated[1]
    first, last = builder.edge_vertices[target]
    builder.edge_vertices[target] = (vertex, last) if endpoint == 0 else (first, vertex)
    builder.edge_endpoint_roots[target] = (None, None)
    box = builder.faces[faces[face]].box.copy()
    box[endpoint, 1] = float(z)
    builder.faces[faces[face]].box = box
    for loop in builder.faces[faces[face]].loops:
        for coedge in loop:
            if builder.coedge_edges[coedge] == target:
                builder.coedge_endpoint_roots[coedge] = (None, None)
