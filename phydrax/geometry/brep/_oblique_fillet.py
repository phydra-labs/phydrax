#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Identity-sewn plane-pair fillets with native certified planar end trims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import atan2, isqrt, pi

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._planar_primitives import reserve_fraction_work
from .._rational import fraction_down, fraction_up, rational_cross, rational_vector
from ._constructors import BRepTessellationPolicy, Builder, finish
from ._fillet import BRepBlendUnsupported
from ._intersection import (
    exact_source_lift,
    intersect_surface_regions,
    native_edge_pcurve,
    NativePeriodEndpoint,
    ParametricIntersectionPolicy,
)
from ._intersection_curve import IntersectionCurve, SurfaceRegion
from ._model import BRepGeometry, BRepModel, BRepPCurve
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import CylinderPatch, LineCurve, PlanePatch
from ._planar_offset import _dot, _solve
from ._request_preparations import TrimRequestPreparations


type _Vector = tuple[Fraction, ...]


@dataclass(frozen=True, slots=True)
class _Junction:
    geometry: BRepGeometry
    sides: tuple[int, int]
    caps: tuple[int, int]
    endpoints: tuple[int, int]
    cap_edges: tuple[tuple[int, int], tuple[int, int]]
    points: tuple[_Vector, ...]
    normals: tuple[_Vector, ...]
    axis: _Vector


@dataclass(frozen=True, slots=True)
class _Blend:
    patch: CylinderPatch
    angle: float
    orientation: int
    contacts: tuple[tuple[_Vector, _Vector], tuple[_Vector, _Vector]]
    heights: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    height_bounds: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]


@dataclass(frozen=True, slots=True)
class _Cap:
    curve: IntersectionCurve
    start_side: int


@dataclass(frozen=True, slots=True)
class _StagedEdges:
    builder: Builder
    edge_map: dict[int, int]
    contact_edges: tuple[int, int]
    contact_pcurves: tuple[LineCurve, LineCurve]
    cap_edges: tuple[int, int]


def _unit(vector: _Vector, /) -> _Vector:
    squared = _dot(vector, vector)
    numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
    if (
        numerator == 0
        or numerator * numerator != squared.numerator
        or denominator * denominator != squared.denominator
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact rational unit side/edge frame unavailable"
        )
    length = Fraction(numerator, denominator)
    return tuple(value / length for value in vector)


def _binary(value: Fraction, /) -> float:
    if abs(value) > Fraction(float(np.finfo(np.float64).max)):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: native line coefficient exceeds binary64 range"
        )
    result = float(value)
    if not np.isfinite(result) or Fraction(result) != value:
        raise BRepBlendUnsupported(
            "BRepBlend.geometry must retain its admitted carrier and incidence; got unsupported plane-pair fillet: native line coefficient lacks an rational binary realization"
        )
    return result


def _array(vector: _Vector, /) -> np.ndarray:
    return np.asarray(tuple(_binary(value) for value in vector), dtype=np.float64)


def _plane(model: BRepModel, face: int, /) -> PlanePatch:
    patch = model.patches[face]
    if not isinstance(patch, PlanePatch):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: source and unblended end faces must be planar"
        )
    return patch


def _source_geometry(
    model: BRepModel, budget: CoordinateEnclosureBudget, /
) -> tuple[BRepGeometry, tuple[_Vector, ...], tuple[_Vector, ...]]:
    """Certify the literal convex polyhedron before changing its incidence."""
    geometry = model.require_geometry()
    if (
        len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.solid_shells != ((0,),)
        or any(sense != 1 for sense in geometry.shell_orientations[0])
        or any(root is not None for root in geometry.vertex_roots)
        or any(root is not None for pair in geometry.edge_endpoint_roots for root in pair)
        or any(
            root is not None for pair in geometry.coedge_endpoint_roots for root in pair
        )
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: requires one literal convex planar solid shell"
        )
    points = tuple(rational_vector(np.asarray(point)) for point in geometry.vertex_points)
    normals: list[_Vector] = []
    for face, loops in enumerate(geometry.face_loops):
        patch = _plane(model, face)
        if len(loops) != 1:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: source faces must have one polygonal loop"
            )
        raw = rational_cross(
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )
        normal = tuple(int(np.asarray(model.orientation)[face]) * value for value in raw)
        if not _dot(normal, normal):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: singular source plane"
            )
        normals.append(normal)
    reserve_fraction_work(
        (*points, *normals),
        48 * len(points) * len(normals),
        16 * len(points) * len(normals),
        12,
        budget=budget,
    )
    for face, normal in enumerate(normals):
        origin = rational_vector(np.asarray(_plane(model, face).origin))
        offset = _dot(normal, origin)
        incident = {
            vertex
            for coedge in geometry.face_loops[face][0]
            for vertex in geometry.edge_vertices[geometry.coedge_edges[coedge]]
        }
        if any(_dot(normal, point) > offset for point in points) or any(
            _dot(normal, points[vertex]) != offset for vertex in incident
        ):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: source planes lack a convex support certificate"
            )
    for index, curve_index in enumerate(geometry.edge_curves):
        if curve_index < 0 or not isinstance(geometry.curves[curve_index], LineCurve):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: source polygon edges must be native rational lines"
            )
        curve = geometry.curves[curve_index]
        if not isinstance(curve, LineCurve):
            raise RuntimeError(
                "BRepBlend.geometry must retain its admitted carrier and incidence; got plane-pair fillet source-line narrowing failed"
            )
        origin = rational_vector(np.asarray(curve.origin))
        direction = rational_vector(np.asarray(curve.direction))
        for parameter, vertex in zip(
            np.asarray(geometry.edge_ranges[index]),
            geometry.edge_vertices[index],
            strict=True,
        ):
            endpoint = tuple(
                a + Fraction(float(parameter)) * b
                for a, b in zip(origin, direction, strict=True)
            )
            if endpoint != points[vertex]:
                raise BRepBlendUnsupported(
                    "BRepBlend.geometry must retain its admitted carrier and incidence; got unsupported plane-pair fillet: source line endpoints lack rational incidence"
                )
    return geometry, points, tuple(normals)


def _junction(
    model: BRepModel, edge: int, budget: CoordinateEnclosureBudget, /
) -> _Junction:
    geometry, points, normals = _source_geometry(model, budget)
    if not 0 <= edge < len(geometry.edge_curves):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: edge index out of range"
        )
    faces = model.topology.edge_faces[edge]
    if len(faces) != 2:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: selected edge must have two plane uses"
        )
    sides = (faces[0], faces[1])
    endpoints = geometry.edge_vertices[edge]
    axis = _unit(
        tuple(
            b - a for a, b in zip(points[endpoints[0]], points[endpoints[1]], strict=True)
        )
    )
    if any(_dot(normals[side], axis) != 0 for side in sides):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: side planes must contain the selected edge"
        )
    if not _dot(rational_cross(normals[sides[0]], normals[sides[1]]), axis):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: side planes are parallel"
        )
    caps: list[int] = []
    cap_edges: list[list[int]] = [[], []]
    for endpoint in endpoints:
        incident = {
            index for index, pair in enumerate(geometry.edge_vertices) if endpoint in pair
        }
        if len(incident) != 3:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: endpoint requires a multi-edge corner patch"
            )
        other = incident - {edge}
        cap_faces = set.intersection(
            *(set(model.topology.edge_faces[index]) for index in other)
        ) - set(sides)
        if len(cap_faces) != 1:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: endpoint edges must share one unblended cap plane"
            )
        cap = next(iter(cap_faces))
        if not _dot(normals[cap], axis):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: cap plane is parallel to the blend axis"
            )
        caps.append(cap)
        for side_index, side in enumerate(sides):
            candidates = [
                index for index in other if side in model.topology.edge_faces[index]
            ]
            if len(candidates) != 1:
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: ambiguous side/cap source incidence"
                )
            cap_edges[side_index].append(candidates[0])
    if caps[0] == caps[1]:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: coincident endpoint cap"
        )
    return _Junction(
        geometry,
        sides,
        (caps[0], caps[1]),
        endpoints,
        (
            (cap_edges[0][0], cap_edges[0][1]),
            (cap_edges[1][0], cap_edges[1][1]),
        ),
        points,
        normals,
        axis,
    )


def _blend(
    model: BRepModel,
    junction: _Junction,
    radius: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Blend:
    normals = tuple(_unit(junction.normals[side]) for side in junction.sides)
    anchor = junction.points[junction.endpoints[0]]
    r = Fraction(radius)
    center = _solve(
        (normals[0], normals[1], junction.axis),
        (
            _dot(normals[0], anchor) - r,
            _dot(normals[1], anchor) - r,
            _dot(junction.axis, anchor),
        ),
        budget=budget,
    )
    second = rational_cross(junction.axis, normals[0])
    handed = 1 if _dot(second, normals[1]) > 0 else -1
    second = tuple(handed * value for value in second)
    angle = atan2(float(_dot(second, normals[1])), float(_dot(normals[0], normals[1])))
    if not 0.0 < angle < pi:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: degenerate convex dihedral"
        )
    patch = CylinderPatch(
        _array(center),
        _array(normals[0]),
        _array(second),
        _array(junction.axis),
        radius,
    )
    heights: list[tuple[Fraction, Fraction]] = []
    contacts: list[tuple[_Vector, _Vector]] = []
    for normal in normals:
        origin = tuple(a + r * b for a, b in zip(center, normal, strict=True))
        values: list[Fraction] = []
        points: list[_Vector] = []
        for cap in junction.caps:
            cap_origin = rational_vector(np.asarray(_plane(model, cap).origin))
            raw = junction.normals[cap]
            height = (_dot(raw, cap_origin) - _dot(raw, origin)) / _dot(
                raw, junction.axis
            )
            values.append(height)
            points.append(
                tuple(a + height * b for a, b in zip(origin, junction.axis, strict=True))
            )
        if values[0] >= values[1]:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: contact edge vanishes"
            )
        heights.append((values[0], values[1]))
        contacts.append((points[0], points[1]))
    height_bounds: list[tuple[Fraction, Fraction]] = []
    for cap in junction.caps:
        raw = junction.normals[cap]
        cap_origin = rational_vector(np.asarray(_plane(model, cap).origin))
        denominator = _dot(raw, junction.axis)
        middle = (_dot(raw, cap_origin) - _dot(raw, center)) / denominator
        spread = r * (
            abs(_dot(raw, normals[0]) / denominator)
            + abs(_dot(raw, second) / denominator)
        )
        height_bounds.append((middle - spread, middle + spread))
    if height_bounds[0][1] >= height_bounds[1][0]:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: cap separation cannot certify a nonvanishing cylindrical trim"
        )
    return _Blend(
        patch,
        angle,
        handed,
        (contacts[0], contacts[1]),
        (heights[0], heights[1]),
        (height_bounds[0], height_bounds[1]),
    )


def _cap(
    model: BRepModel,
    junction: _Junction,
    blend: _Blend,
    end: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> _Cap:
    face = junction.caps[end]
    box = np.asarray(model.parameter_bounds[face])
    width = box[1] - box[0]
    # Only cylinder angular boundaries may terminate discovery. The retained
    # source plane is unchanged; the padded rectangle is a generating region,
    # not a replacement for the source polygon's certified trim.
    plane = SurfaceRegion(
        _plane(model, face),
        np.stack((box[0] - width, box[1] + width)),
        preparations=preparations,
        budget=budget,
    )
    lower, upper = blend.height_bounds[end]
    padding = upper - lower + 1
    cylinder = SurfaceRegion(
        blend.patch,
        (
            (0.0, fraction_down(lower - padding)),
            (blend.angle, fraction_up(upper + padding)),
        ),
        preparations=preparations,
        budget=budget,
    )
    result = intersect_surface_regions(
        plane,
        cylinder,
        policy=ParametricIntersectionPolicy(maximum_boxes=20_000),
        preparations=preparations,
        coordinate_budget=budget,
    )
    if (
        not result.complete
        or len(result.curves) != 1
        or result.points
        or result.coincident
    ):
        raise BRepBlendUnsupported(
            "BRepBlend.geometry must retain its admitted carrier and incidence; got unsupported plane-pair fillet: native plane/cylinder cap atlas did not certify one whole branch"
        )
    curve = result.curves[0]
    if (
        curve.closed
        or curve.start_kind != "boundary"
        or curve.end_kind != "boundary"
        or int(curve.node_axes[0]) != 2
        or int(curve.node_axes[-1]) != 2
        or (float(curve.node_values[0]), float(curve.node_values[-1]))
        not in ((0.0, blend.angle), (blend.angle, 0.0))
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: cap atlas endpoints do not retain the two contact-angle identities"
        )
    return _Cap(curve, 0 if float(curve.node_values[0]) == 0.0 else 1)


def _trim_parameter(geometry: BRepGeometry, edge: int, point: _Vector, /) -> Fraction:
    curve = geometry.curves[geometry.edge_curves[edge]]
    if not isinstance(curve, LineCurve):
        raise RuntimeError(
            "BRepBlend.geometry must retain its admitted carrier and incidence; got plane-pair fillet cap boundary lost its source line"
        )
    origin = rational_vector(np.asarray(curve.origin))
    direction = rational_vector(np.asarray(curve.direction))
    offset = tuple(b - a for a, b in zip(origin, point, strict=True))
    parameter = _dot(offset, direction) / _dot(direction, direction)
    first, last = (
        Fraction(float(value)) for value in np.asarray(geometry.edge_ranges[edge])
    )
    if not first < parameter < last or any(
        a != parameter * b for a, b in zip(offset, direction, strict=True)
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: contact reaches a neighboring edge or face vanishes"
        )
    return parameter


def _contact_pcurve(
    model: BRepModel,
    face: int,
    curve: LineCurve,
    first: float,
    last: float,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> LineCurve:
    region = SurfaceRegion(
        _plane(model, face),
        np.asarray(model.parameter_bounds[face]),
        preparations=preparations,
        budget=budget,
    )
    pcurve = native_edge_pcurve(curve, region)
    if not isinstance(pcurve, LineCurve) or not exact_source_lift(
        curve, pcurve, region, first, last
    ):
        raise BRepBlendUnsupported(
            "BRepBlend.geometry must retain its admitted carrier and incidence; got unsupported plane-pair fillet: contact line lacks an rational native source-plane p-curve"
        )
    return pcurve


def _certify_clearance(
    model: BRepModel,
    junction: _Junction,
    blend: _Blend,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    box = blend.patch.bounding_box(
        (
            (0.0, fraction_down(blend.height_bounds[0][0])),
            (blend.angle, fraction_up(blend.height_bounds[1][1])),
        ),
        preparations=preparations,
        budget=budget,
    )
    for face, normal in enumerate(junction.normals):
        if face in (*junction.sides, *junction.caps):
            continue
        origin = rational_vector(np.asarray(_plane(model, face).origin))
        upper = sum(
            (
                value * Fraction(float(box[1 if value > 0 else 0, axis]))
                for axis, value in enumerate(normal)
            ),
            Fraction(),
        )
        if upper >= _dot(normal, origin):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: cylinder clearance from a nonincident source face is uncertified"
            )


def _stage_edges(
    model: BRepModel,
    selected: int,
    junction: _Junction,
    blend: _Blend,
    caps: tuple[_Cap, _Cap],
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> _StagedEdges:
    geometry, builder = junction.geometry, Builder()
    vertex_map = {
        index: builder.vertex(_array(point))
        for index, point in enumerate(junction.points)
        if index not in junction.endpoints
    }
    contacts = tuple(
        tuple(builder.vertex(_array(point)) for point in side) for side in blend.contacts
    )
    changes = {
        edge: (
            contacts[side][end],
            _trim_parameter(geometry, edge, blend.contacts[side][end]),
        )
        for side, edges in enumerate(junction.cap_edges)
        for end, edge in enumerate(edges)
    }
    edge_map: dict[int, int] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if edge == selected:
            continue
        bounds = [float(value) for value in np.asarray(geometry.edge_ranges[edge])]
        vertices: list[int] = []
        for endpoint, vertex in enumerate(geometry.edge_vertices[edge]):
            if vertex in junction.endpoints:
                replacement, parameter = changes[edge]
                vertices.append(replacement)
                bounds[endpoint] = float(parameter)
            else:
                vertices.append(vertex_map[vertex])
        edge_map[edge] = builder.edge(
            geometry.curves[curve_index],
            bounds[0],
            bounds[1],
            vertices[0],
            vertices[1],
        )
        if edge in changes:
            # Q is the zero-turn case of the native Q + Q*(2*pi) scalar
            # algebra. Keep the source line and p-curves, retaining its
            # rational trim scalar rather than requiring a binary fraction.
            root = NativePeriodEndpoint(
                geometry.curves[curve_index],
                blend.patch,
                0,
                rational=changes[edge][1],
                turns=0,
                preparations=preparations,
                budget=budget,
            )
            start, end = geometry.edge_vertices[edge]
            builder.edge_endpoint_roots[edge_map[edge]] = (
                root if start in junction.endpoints else None,
                root if end in junction.endpoints else None,
            )
    contact_edges: list[int] = []
    contact_pcurves: list[LineCurve] = []
    for side, face in enumerate(junction.sides):
        h0, h1 = blend.heights[side]
        origin = tuple(
            a - h0 * b
            for a, b in zip(blend.contacts[side][0], junction.axis, strict=True)
        )
        curve = LineCurve(_array(origin), _array(junction.axis))
        first, last = _binary(h0), _binary(h1)
        contact_edges.append(
            builder.edge(curve, first, last, contacts[side][0], contacts[side][1])
        )
        contact_pcurves.append(
            _contact_pcurve(model, face, curve, first, last, preparations, budget)
        )
    cap_edges = tuple(
        builder.edge(
            cap.curve,
            0.0,
            float(cap.curve.num_charts),
            contacts[cap.start_side][end],
            contacts[1 - cap.start_side][end],
        )
        for end, cap in enumerate(caps)
    )
    return _StagedEdges(
        builder,
        edge_map,
        (contact_edges[0], contact_edges[1]),
        (contact_pcurves[0], contact_pcurves[1]),
        (cap_edges[0], cap_edges[1]),
    )


def _stage_source_faces(
    model: BRepModel,
    selected: int,
    junction: _Junction,
    caps: tuple[_Cap, _Cap],
    edges: _StagedEdges,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> dict[int, tuple[int, int]]:
    geometry, builder = junction.geometry, edges.builder
    boundary_uses: dict[int, tuple[int, int]] = {}
    pcurve: BRepPCurve
    for face, source_loops in enumerate(geometry.face_loops):
        loop: list[int] = []
        source_loop = source_loops[0]
        orientation = int(np.asarray(model.orientation)[face])
        for position, coedge in enumerate(source_loop):
            edge = geometry.coedge_edges[coedge]
            sense = geometry.coedge_senses[coedge]
            if edge == selected:
                side = junction.sides.index(face)
                target_edge = edges.contact_edges[side]
                pcurve = edges.contact_pcurves[side]
                boundary_uses[target_edge] = (sense, orientation)
            else:
                target_edge = edges.edge_map[edge]
                pcurve = geometry.pcurves[coedge]
            loop.append(
                builder.coedge(
                    target_edge,
                    sense,
                    pcurve,
                    preparations=preparations,
                    budget=budget,
                )
            )
            tail = geometry.edge_vertices[edge][1 if sense == 1 else 0]
            if face not in junction.caps or tail not in junction.endpoints:
                continue
            end = junction.caps.index(face)
            start_side = next(
                side for side in range(2) if edge == junction.cap_edges[side][end]
            )
            following = geometry.coedge_edges[
                source_loop[(position + 1) % len(source_loop)]
            ]
            if following != junction.cap_edges[1 - start_side][end]:
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: cap corner incidence is not consecutive"
                )
            cap_sense = 1 if caps[end].start_side == start_side else -1
            cap_edge = edges.cap_edges[end]
            loop.append(
                builder.coedge(
                    cap_edge,
                    cap_sense,
                    caps[end].curve.p_curve("first"),
                    preparations=preparations,
                    budget=budget,
                )
            )
            boundary_uses[cap_edge] = (cap_sense, orientation)
        builder.face(
            model.patches[face],
            np.asarray(model.parameter_bounds[face]),
            [loop],
            model.physical_tags[face],
            None,
            orientation=orientation,
        )
    return boundary_uses


def _stage_blend_face(
    blend: _Blend,
    caps: tuple[_Cap, _Cap],
    edges: _StagedEdges,
    boundary_uses: dict[int, tuple[int, int]],
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
    /,
) -> None:
    builder = edges.builder
    # Order by the shared vertex graph, not geometric proximity. The effective
    # oriented use must oppose the source face on every identity-shared edge.
    blend_uses: dict[int, tuple[int, int]] = {}
    outgoing: dict[int, int] = {}
    for edge in (*edges.contact_edges, *edges.cap_edges):
        owner_sense, owner_orientation = boundary_uses[edge]
        sense = -owner_sense * owner_orientation * blend.orientation
        first, last = builder.edge_vertices[edge]
        head, tail = (first, last) if sense == 1 else (last, first)
        outgoing[head] = edge
        blend_uses[edge] = (sense, tail)
    if len(outgoing) != 4:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: blend boundary is not a directed cycle"
        )
    ordered: list[int] = []
    edge = edges.cap_edges[0]
    while edge not in ordered:
        ordered.append(edge)
        tail = blend_uses[edge][1]
        if tail not in outgoing:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: blend boundary does not form one directed loop"
            )
        edge = outgoing[tail]
    if len(ordered) != 4 or edge != ordered[0]:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: blend boundary loop is incomplete"
        )
    loop: list[int] = []
    pcurve: BRepPCurve
    for edge in ordered:
        if edge in edges.cap_edges:
            pcurve = caps[edges.cap_edges.index(edge)].curve.p_curve("second")
        else:
            side = edges.contact_edges.index(edge)
            pcurve = LineCurve((0.0 if side == 0 else blend.angle, 0.0), (0.0, 1.0))
        loop.append(
            builder.coedge(
                edge,
                blend_uses[edge][0],
                pcurve,
                preparations=preparations,
                budget=budget,
            )
        )
    boxes = np.stack(
        [
            builder.pcurves[coedge].bounding_box(
                *builder.edge_ranges[builder.coedge_edges[coedge]],
                preparations=preparations,
                budget=budget,
            )
            for coedge in loop
        ]
    )
    box = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    builder.face(
        blend.patch,
        box,
        [loop],
        "cylinder",
        None,
        orientation=blend.orientation,
    )


def _contact_radius_limits(
    junction: _Junction,
    delta: _Vector,
    normals: tuple[_Vector, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, ...]:
    """Solve the affine tangent contact motion on each retained source cap edge."""
    budget.reserve(256, 4096)
    limits: list[Fraction] = []
    for side, normal in enumerate(normals):
        transverse = tuple(a + b for a, b in zip(delta, normal, strict=True))
        for end, cap in enumerate(junction.caps):
            raw = junction.normals[cap]
            height = -_dot(raw, transverse) / _dot(raw, junction.axis)
            motion = tuple(
                a + height * b for a, b in zip(transverse, junction.axis, strict=True)
            )
            edge = junction.cap_edges[side][end]
            endpoint = junction.endpoints[end]
            other = next(
                vertex
                for vertex in junction.geometry.edge_vertices[edge]
                if vertex != endpoint
            )
            direction = tuple(
                b - a
                for a, b in zip(
                    junction.points[endpoint], junction.points[other], strict=True
                )
            )
            slope = _dot(motion, direction) / _dot(direction, direction)
            if slope <= 0 or motion != tuple(slope * value for value in direction):
                raise BRepBlendUnsupported(
                    "oblique fillet range: inward affine contact motion is unproved"
                )
            limits.append(1 / slope)
    return tuple(limits)


def _clearance_radius_limits(
    model: BRepModel,
    junction: _Junction,
    delta: _Vector,
    normals: tuple[_Vector, ...],
    budget: CoordinateEnclosureBudget,
    /,
) -> tuple[Fraction, ...]:
    """Bound all cylindrical arcs by rational full-circle support and cap envelopes."""
    budget.reserve(128 * len(junction.normals), 4096)
    second = rational_cross(junction.axis, normals[0])
    anchor = junction.points[junction.endpoints[0]]
    caps: list[tuple[Fraction, Fraction, Fraction]] = []
    for cap in junction.caps:
        raw = junction.normals[cap]
        denominator = _dot(raw, junction.axis)
        origin = rational_vector(np.asarray(_plane(model, cap).origin))
        base = (_dot(raw, origin) - _dot(raw, anchor)) / denominator
        slope = -_dot(raw, delta) / denominator
        spread = abs(_dot(raw, normals[0]) / denominator) + abs(
            _dot(raw, second) / denominator
        )
        caps.append((base, slope - spread, slope + spread))
    gap = caps[1][0] - caps[0][0]
    growth = caps[0][2] - caps[1][1]
    if gap <= 0:
        raise BRepBlendUnsupported(
            "oblique fillet range: positive cap separation is unproved"
        )
    limits = [gap / growth] if growth > 0 else []
    for face, normal in enumerate(junction.normals):
        if face in (*junction.sides, *junction.caps):
            continue
        axial = _dot(normal, junction.axis)
        base, lower, upper = caps[1 if axial >= 0 else 0]
        origin = rational_vector(np.asarray(_plane(model, face).origin))
        margin = _dot(normal, origin) - _dot(normal, anchor) - axial * base
        growth = _dot(normal, delta) + axial * (upper if axial >= 0 else lower)
        growth += abs(_dot(normal, normals[0])) + abs(_dot(normal, second))
        if margin <= 0:
            raise BRepBlendUnsupported(
                "oblique fillet range: strict nonincident-face separation is unproved"
            )
        if growth > 0:
            limits.append(margin / growth)
    return tuple(limits)


def oblique_fillet_parameter_range(
    model: BRepModel,
    edge_index: int,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Certify affine contact limits and a conservative whole-cylinder clearance."""
    try:
        junction = _junction(model, edge_index, budget)
        normals = tuple(_unit(junction.normals[side]) for side in junction.sides)
        delta = _solve(
            (normals[0], normals[1], junction.axis),
            (Fraction(-1), Fraction(-1), Fraction(0)),
            budget=budget,
        )
        contacts = _contact_radius_limits(junction, delta, normals, budget)
        clearance = _clearance_radius_limits(model, junction, delta, normals, budget)
        limit = min((*contacts, *clearance))
        upper = fraction_down(min(limit, Fraction(float(np.finfo(np.float64).max))))
        if upper <= 0:
            raise BRepBlendUnsupported(
                "oblique fillet range: no positive representable certified interval"
            )
    except BRepBlendUnsupported as error:
        return unknown_parameter_range("radius", str(error))
    exact = limit == min(contacts) and Fraction(upper) == limit
    return CertifiedParameterRange(
        "radius",
        (0.0, upper),
        end_status=(
            "attained_exactly",
            "attained_exactly" if exact else "certified_inner_bound",
        ),
        constraints=(
            "parameter_domain",
            "face_vanishes" if limit == min(contacts) else "trim_vanishes",
        ),
        entity_ids=((model.edge_ids[edge_index],), tuple(model.face_ids)),
        evidence=(
            "Exact rational convex source planes and inward affine tangent contact motion.",
            "Retained source edges stay positive; cap envelopes stay separated; full-circle support stays strictly inside every nonincident source plane.",
            f"Exact rational limiting proof value {limit}; clearance envelopes may be conservative.",
        ),
    )


def oblique_fillet(
    model: BRepModel,
    edge_index: int,
    radius: float,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Replace a convex plane junction, retaining native cap-atlas certificates.

    Exact rational source incidence determines every retained face, shortened cap
    line, and contact endpoint. Both source-plane and cylinder cap p-curves come
    from the same native intersection atlas; no fitted conic or local root solve
    is introduced. Radius sensitivity of these coupled atlas trims is owned by
    the operation's separate sensitivity status, never a primal refusal.
    """
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"oblique_fillet.radius must be positive and finite; got {radius!r}"
        )
    junction = _junction(model, edge_index, budget)
    blend = _blend(model, junction, radius, budget)
    # Refuse disappearing source edges before spending native discovery work.
    for side, edges in enumerate(junction.cap_edges):
        for end, edge in enumerate(edges):
            _trim_parameter(junction.geometry, edge, blend.contacts[side][end])
    _certify_clearance(model, junction, blend, preparations, budget)
    caps = (
        _cap(model, junction, blend, 0, preparations, budget),
        _cap(model, junction, blend, 1, preparations, budget),
    )
    edges = _stage_edges(model, edge_index, junction, blend, caps, preparations, budget)
    boundary_uses = _stage_source_faces(
        model, edge_index, junction, caps, edges, preparations, budget
    )
    _stage_blend_face(blend, caps, edges, boundary_uses, preparations, budget)
    geometry = junction.geometry
    return finish(
        edges.builder,
        solid=True,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        occurrences=geometry.occurrences,
        shells=(
            ((*geometry.shell_faces[0], len(model.patches)),),
            ((*geometry.shell_orientations[0], 1),),
            geometry.solid_shells,
        ),
        preparations=preparations,
        budget=budget,
    )
