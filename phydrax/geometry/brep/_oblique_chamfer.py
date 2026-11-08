#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational setback chamfers of rational polyhedra, including oblique end planes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
)
from ...linalg._exact import solve_exact
from .._rational import rational_cross, rational_vector
from ._assembly import BRepAffinePlacement
from ._constructors import _Builder, assemble_brep_model, BRepTessellationPolicy
from ._fillet import BRepBlendUnsupported
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import BRepCurve, ExactLineCurve, LineCurve, PlanePatch


type _Vector = tuple[ExactScalar, ...]
type _Plane = tuple[_Vector, ExactScalar]


@dataclass(frozen=True, slots=True)
class _Corner:
    endpoints: tuple[int, int]
    sides: tuple[int, int]
    caps: tuple[int, int]
    neighbors: tuple[tuple[int, int], tuple[int, int]]


def _difference(first: _Vector, last: _Vector, /) -> _Vector:
    return tuple(b - a for a, b in zip(first, last, strict=True))


def _dot(first: _Vector, second: _Vector, /) -> ExactScalar:
    return sum(
        (a * b for a, b in zip(first, second, strict=True)),
        Fraction(0),
    )


def _binary(values: _Vector, /) -> np.ndarray:
    """Publish rounded coordinates for exact rational-radical sources."""
    try:
        result = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    except OverflowError as error:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact coordinate exceeds binary64 range"
        ) from error
    if not np.all(np.isfinite(result)):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact coordinate exceeds binary64 range"
        )
    return result


def _line_curve(first: _Vector, last: _Vector, /) -> BRepCurve:
    if all(isinstance(value, Fraction) for value in (*first, *last)):
        first_rational = tuple(cast(Fraction, value) for value in first)
        last_rational = tuple(cast(Fraction, value) for value in last)
        return ExactLineCurve(first_rational, last_rational)
    direction = _difference(first, last)
    return LineCurve(
        _binary(first),
        _binary(direction),
        origin_exact=first,
        direction_exact=direction,
    )


def _actions(
    matrix: tuple[_Vector, ...],
    right: tuple[_Vector, ...],
    /,
) -> tuple[_Vector, ...]:
    solution = solve_exact(
        matrix,
        right,
        singular_error=BRepBlendUnsupported,
        singular_message="planar chamfer unsupported: incident plane or chart system is singular",
    )
    return tuple(tuple(row) for row in solution)


def _unit(direction: _Vector, /) -> _Vector:
    squared = _dot(direction, direction)
    if squared <= 0:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact side setback direction is singular"
        )
    if isinstance(squared, Fraction):
        radicand = squared
    elif (
        isinstance(squared, RadicalElement)
        and len(squared.terms) == 1
        and squared.terms[0][0] == 1
    ):
        radicand = squared.terms[0][1]
    else:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact direction norm is outside the rational radical base"
        )
    length = RadicalElement.sqrt(radicand)
    return tuple(value / length for value in direction)


def _corner(geometry: BRepGeometry, selected: int, /) -> _Corner:
    if not 0 <= selected < len(geometry.edge_vertices):
        raise ValueError(
            "oblique_chamfer.edge_index must identify a source edge; got an out-of-range index"
        )
    face_edges = tuple(
        frozenset(geometry.coedge_edges[coedge] for coedge in loops[0])
        for loops in geometry.face_loops
    )
    sides = tuple(face for face, edges in enumerate(face_edges) if selected in edges)
    if len(sides) != 2:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: selected edge must have two incident side faces"
        )
    endpoints = geometry.edge_vertices[selected]
    caps = []
    neighbors = [[], []]
    for endpoint in endpoints:
        incident = frozenset(
            edge
            for edge, vertices in enumerate(geometry.edge_vertices)
            if endpoint in vertices
        )
        cap = tuple(
            face
            for face, edges in enumerate(face_edges)
            if face not in sides and edges & incident
        )
        if len(incident) != 3 or len(cap) != 1:
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: endpoints require unique unblended planar caps"
            )
        caps.append(cap[0])
        for side_index, side in enumerate(sides):
            neighbor = (face_edges[side] & face_edges[cap[0]] & incident) - {selected}
            if len(neighbor) != 1:
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: source cap-side incidence is ambiguous"
                )
            neighbors[side_index].append(next(iter(neighbor)))
    if caps[0] == caps[1]:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: end caps must be distinct source faces"
        )
    return _Corner(
        endpoints,
        (sides[0], sides[1]),
        (caps[0], caps[1]),
        ((neighbors[0][0], neighbors[0][1]), (neighbors[1][0], neighbors[1][1])),
    )


def _source_planes(
    model: BRepModel,
    vertices: tuple[_Vector, ...],
    /,
) -> tuple[_Plane, ...]:
    geometry = model.geometry
    assembly = geometry.assembly_graph
    if (
        geometry.solid_shells != ((0,),)
        or geometry.shell_faces != (tuple(range(len(model.patches))),)
        or geometry.shell_orientations != ((1,) * len(model.patches),)
        or assembly.solid_count != 1
        or assembly.definitions
        or len(assembly.roots) != 1
        or assembly.roots[0].solid_index != 0
        or assembly.roots[0].placement != BRepAffinePlacement.identity()
        or any(root is not None for root in geometry.vertex_roots)
        or any(
            root is not None
            for endpoints in geometry.edge_endpoint_roots
            for root in endpoints
        )
        or any(
            root is not None
            for endpoints in geometry.coedge_endpoint_roots
            for root in endpoints
        )
    ):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: source requires one unplaced, outward literal solid shell"
        )
    planes = []
    for face, patch in enumerate(model.patches):
        if not isinstance(patch, PlanePatch) or len(geometry.face_loops[face]) != 1:
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: source must have single-loop plane faces"
            )
        normal = rational_cross(
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )
        normal = tuple(
            int(np.asarray(model.orientation)[face]) * value for value in normal
        )
        if not any(normal):
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: source plane is singular"
            )
        planes.append((normal, _dot(normal, rational_vector(np.asarray(patch.origin)))))
    for face, (normal, height) in enumerate(planes):
        for coedge in geometry.face_loops[face][0]:
            edge = geometry.coedge_edges[coedge]
            if any(
                _dot(normal, vertices[index]) != height
                for index in geometry.edge_vertices[edge]
            ):
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: source vertices do not exactly retain plane incidence"
                )
    edge_ranges = np.asarray(geometry.edge_ranges)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: degenerate source edges have no line carrier"
            )
        curve = geometry.curves[curve_index]
        if isinstance(curve, ExactLineCurve):
            if curve.first_exact != vertices[first] or curve.last_exact != vertices[last]:
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: source exact line endpoints do not match their vertices"
                )
            if tuple(edge_ranges[edge]) != (0.0, 1.0):
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: exact source lines require their authored unit interval"
                )
            continue
        if not isinstance(curve, LineCurve):
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: source edges require literal line carriers"
            )
        origin, direction = curve.origin_exact, curve.direction_exact
        for parameter, vertex in zip(edge_ranges[edge], (first, last), strict=True):
            point = tuple(
                a + Fraction(float(parameter)) * b
                for a, b in zip(origin, direction, strict=True)
            )
            if point != vertices[vertex]:
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: source line endpoints have nonzero incidence gaps"
                )
    return tuple(planes)


def _contacts(
    geometry: BRepGeometry,
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    planes: tuple[_Plane, ...],
    setbacks: tuple[float, float],
    /,
) -> tuple[tuple[_Vector, _Vector], tuple[_Vector, _Vector]]:
    direction = _difference(vertices[corner.endpoints[0]], vertices[corner.endpoints[1]])
    rows = []
    for side_index, (side, distance) in enumerate(
        zip(corner.sides, setbacks, strict=True)
    ):
        normal, height = planes[side]
        transverse = _unit(rational_cross(direction, normal))
        sign = _dot(planes[corner.sides[1 - side_index]][0], transverse)
        if sign == 0:
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: side planes are parallel"
            )
        if sign > 0:
            transverse = tuple(-value for value in transverse)
        points = []
        for end_index, (endpoint, cap) in enumerate(
            zip(corner.endpoints, corner.caps, strict=True)
        ):
            cap_normal, cap_height = planes[cap]
            solution = _actions(
                (normal, transverse, cap_normal),
                (
                    (height,),
                    (_dot(transverse, vertices[endpoint]) + Fraction(distance),),
                    (cap_height,),
                ),
            )
            point = tuple(row[0] for row in solution)
            neighbor = corner.neighbors[side_index][end_index]
            first, last = (vertices[index] for index in geometry.edge_vertices[neighbor])
            edge_direction, offset = _difference(first, last), _difference(first, point)
            parameter = _dot(offset, edge_direction) / _dot(
                edge_direction, edge_direction
            )
            if not 0 < parameter < 1 or offset != tuple(
                parameter * value for value in edge_direction
            ):
                raise BRepBlendUnsupported(
                    "planar chamfer refuses: setback reaches a neighboring edge or face vanishes"
                )
            points.append(point)
        if _dot(_difference(points[0], points[1]), direction) <= 0:
            raise BRepBlendUnsupported("planar chamfer refuses: oblique end trims cross")
        rows.append((points[0], points[1]))
    return rows[0], rows[1]


def _chamfer_plane(
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    contacts: tuple[tuple[_Vector, _Vector], tuple[_Vector, _Vector]],
    /,
) -> tuple[PlanePatch, int]:
    origin = contacts[0][0]
    first_axis = _difference(origin, contacts[1][0])
    second_axis = _difference(
        vertices[corner.endpoints[0]], vertices[corner.endpoints[1]]
    )
    normal = rational_cross(first_axis, second_axis)
    height = _dot(normal, origin)
    orientation = 1 if _dot(normal, vertices[corner.endpoints[0]]) > height else -1
    if any(
        orientation * (_dot(normal, point) - height) >= 0
        for index, point in enumerate(vertices)
        if index not in corner.endpoints
    ):
        raise BRepBlendUnsupported(
            "planar chamfer refuses: chamfer plane reaches nonincident source geometry"
        )
    if any(_dot(normal, point) != height for row in contacts for point in row):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: contact lines are not coplanar"
        )
    return PlanePatch(
        _binary(origin), _binary(first_axis), _binary(second_axis)
    ), orientation


def _stage_edges(
    geometry: BRepGeometry,
    selected: int,
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    contacts: tuple[tuple[_Vector, _Vector], tuple[_Vector, _Vector]],
    /,
) -> tuple[_Builder, dict[int, int], tuple[tuple[int, int], tuple[int, int]], set[int]]:
    builder = _Builder()
    exact_points: dict[int, _Vector] = {}

    def add_vertex(point: _Vector, /) -> int:
        exact_point: tuple[Fraction, Fraction, Fraction] | None = None
        if all(isinstance(value, Fraction) for value in point):
            exact_point = (
                cast(Fraction, point[0]),
                cast(Fraction, point[1]),
                cast(Fraction, point[2]),
            )
        index = builder.vertex(_binary(point), exact_point=exact_point)
        exact_points[index] = point
        return index

    old_vertices = {
        index: add_vertex(point)
        for index, point in enumerate(vertices)
        if index not in corner.endpoints
    }
    new_vertices = tuple(tuple(add_vertex(point) for point in row) for row in contacts)
    replacements = {
        (corner.neighbors[side][end], corner.endpoints[end]): new_vertices[side][end]
        for side in range(2)
        for end in range(2)
    }
    changed = {edge for row in corner.neighbors for edge in row}
    edge_map: dict[int, int] = {}
    edge_ranges = np.asarray(geometry.edge_ranges)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        if edge == selected:
            continue
        start = (
            replacements[(edge, first)]
            if first in corner.endpoints
            else old_vertices[first]
        )
        end = (
            replacements[(edge, last)] if last in corner.endpoints else old_vertices[last]
        )
        if edge in changed:
            curve = _line_curve(exact_points[start], exact_points[end])
            edge_map[edge] = builder.edge(curve, 0.0, 1.0, start, end)
        else:
            curve_index = geometry.edge_curves[edge]
            if curve_index < 0:
                raise BRepBlendUnsupported(
                    "planar chamfer unsupported: retained source edge is degenerate"
                )
            edge_map[edge] = builder.edge(
                geometry.curves[curve_index],
                edge_ranges[edge, 0],
                edge_ranges[edge, 1],
                start,
                end,
                endpoint_roots=geometry.edge_endpoint_roots[edge],
            )
    longitudinal: list[int] = []
    caps: list[int] = []
    for row in new_vertices:
        longitudinal.append(
            builder.edge(
                _line_curve(exact_points[row[0]], exact_points[row[1]]),
                0.0,
                1.0,
                row[0],
                row[1],
            )
        )
    for end in range(2):
        first, last = new_vertices[0][end], new_vertices[1][end]
        caps.append(
            builder.edge(
                _line_curve(exact_points[first], exact_points[last]),
                0.0,
                1.0,
                first,
                last,
            )
        )
    return (
        builder,
        edge_map,
        ((longitudinal[0], longitudinal[1]), (caps[0], caps[1])),
        changed,
    )


def _stage_face(
    builder: _Builder,
    patch: PlanePatch,
    uses: list[tuple[int, int, BRepCurve | None]],
    orientation: int,
    tag: str,
    /,
) -> None:
    indices = tuple(
        sorted({vertex for edge, _, _ in uses for vertex in builder.edge_vertices[edge]})
    )
    origin, first_axis, second_axis = (
        rational_vector(np.asarray(value))
        for value in (patch.origin, patch.first_axis, patch.second_axis)
    )
    displacements = tuple(
        _difference(origin, rational_vector(builder.vertices[index])) for index in indices
    )
    gram = (
        (_dot(first_axis, first_axis), _dot(first_axis, second_axis)),
        (_dot(second_axis, first_axis), _dot(second_axis, second_axis)),
    )
    solution = _actions(
        gram,
        (
            tuple(_dot(first_axis, point) for point in displacements),
            tuple(_dot(second_axis, point) for point in displacements),
        ),
    )
    coordinates = {
        index: (solution[0][column], solution[1][column])
        for column, index in enumerate(indices)
    }
    loop: list[int] = []
    for edge, sense, pcurve in uses:
        if pcurve is None:
            first, last = builder.edge_vertices[edge]
            pcurve = _line_curve(coordinates[first], coordinates[last])
        loop.append(builder.coedge(edge, sense, pcurve))
    points = np.stack([_binary(coordinates[index]) for index in indices])
    builder.face(
        patch,
        np.stack((points.min(axis=0), points.max(axis=0))),
        orientation,
        tag,
        [loop],
    )


def _stage_source_faces(
    builder: _Builder,
    model: BRepModel,
    selected: int,
    corner: _Corner,
    edge_map: dict[int, int],
    new_edges: tuple[tuple[int, int], tuple[int, int]],
    changed: set[int],
    /,
) -> None:
    geometry = model.geometry
    longitudinal, cap_edges = new_edges
    for face, patch in enumerate(model.patches):
        if not isinstance(patch, PlanePatch):
            raise RuntimeError(
                "oblique_chamfer.source must retain its certified plane carriers; got another patch family"
            )
        uses: list[tuple[int, int, BRepCurve | None]] = []
        for coedge in geometry.face_loops[face][0]:
            source_edge, sense = (
                geometry.coedge_edges[coedge],
                geometry.coedge_senses[coedge],
            )
            if source_edge == selected:
                uses.append((longitudinal[corner.sides.index(face)], sense, None))
            else:
                uses.append(
                    (
                        edge_map[source_edge],
                        sense,
                        None if source_edge in changed else geometry.pcurves[coedge],
                    )
                )
            end = geometry.edge_vertices[source_edge][1 if sense == 1 else 0]
            if (
                end in corner.endpoints
                and face == corner.caps[corner.endpoints.index(end)]
            ):
                endpoint = corner.endpoints.index(end)
                incoming_side = 0 if source_edge == corner.neighbors[0][endpoint] else 1
                uses.append((cap_edges[endpoint], 1 if incoming_side == 0 else -1, None))
        _stage_face(
            builder,
            patch,
            uses,
            int(np.asarray(model.orientation)[face]),
            model.physical_tags[face],
        )


def oblique_chamfer(
    model: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float],
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
) -> BRepModel:
    """Chamfer a convex line junction while retaining exact plane incidence."""
    if len(setbacks) != 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in setbacks
    ):
        raise ValueError(
            "oblique_chamfer.setbacks must contain two positive finite distances; got an invalid specification"
        )
    geometry = model.geometry
    vertices = tuple(
        exact if exact is not None else rational_vector(np.asarray(point))
        for exact, point in zip(
            geometry.vertex_exact_points, geometry.vertex_points, strict=True
        )
    )
    planes = _source_planes(model, vertices)
    corner = _corner(geometry, edge_index)
    contacts = _contacts(geometry, corner, vertices, planes, setbacks)
    patch, orientation = _chamfer_plane(corner, vertices, contacts)
    builder, edge_map, new_edges, changed = _stage_edges(
        geometry, edge_index, corner, vertices, contacts
    )
    _stage_source_faces(
        builder,
        model,
        edge_index,
        corner,
        edge_map,
        new_edges,
        changed,
    )
    longitudinal, caps = new_edges
    _stage_face(
        builder,
        patch,
        [
            (caps[0], 1, None),
            (longitudinal[1], 1, None),
            (caps[1], -1, None),
            (longitudinal[0], -1, None),
        ],
        orientation,
        "chamfer",
    )
    from ._model import _certify_edge_face_gap_bounds

    provisional = builder.geometry(
        solid=True,
        assembly_graph=geometry.assembly_graph,
    )
    provisional_geometry, provisional_patches, provisional_bounds, _, _, _ = provisional
    try:
        face_gaps = _certify_edge_face_gap_bounds(
            provisional_geometry,
            provisional_patches,
            parameter_bounds=provisional_bounds,
            exact_unsupported=True,
        )
    except ValueError as error:
        raise BRepBlendUnsupported(
            "unsupported oblique chamfer: source coedge face gaps lack an exact proof"
        ) from error
    if face_gaps is None:
        raise BRepBlendUnsupported(
            "unsupported oblique chamfer: source coedge face gaps lack an exact proof"
        )
    geometry_, patches, bounds, orientations, tags, trims = builder.geometry(
        solid=True,
        assembly_graph=geometry.assembly_graph,
        edge_face_gap_bounds=face_gaps,
    )
    return assemble_brep_model(
        geometry_,
        patches,
        bounds,
        orientations,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        curve_surface_scale=model.report.curve_surface_scale,
    )


def _chamfer_side_range_motions(
    geometry: BRepGeometry,
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    planes: tuple[_Plane, ...],
    side_index: int,
    distance: float,
    direction: _Vector,
    /,
) -> tuple[tuple[_Vector, _Vector], tuple[ExactScalar, ExactScalar]]:
    """Prove affine inward motion and the opposite-edge collision on one side."""
    normal, height = planes[corner.sides[side_index]]
    transverse = _unit(rational_cross(direction, normal))
    sign = _dot(planes[corner.sides[1 - side_index]][0], transverse)
    if sign == 0:
        raise BRepBlendUnsupported("oblique chamfer range: side planes are parallel")
    if sign > 0:
        transverse = tuple(-value for value in transverse)
    row: list[_Vector] = []
    limits: list[ExactScalar] = []
    for end, (endpoint, cap) in enumerate(
        zip(corner.endpoints, corner.caps, strict=True)
    ):
        cap_normal, cap_height = planes[cap]
        solution = _actions(
            (normal, transverse, cap_normal),
            (
                (height,),
                (_dot(transverse, vertices[endpoint]) + Fraction(distance),),
                (cap_height,),
            ),
        )
        motion = _difference(vertices[endpoint], tuple(value[0] for value in solution))
        neighbor = corner.neighbors[side_index][end]
        other = next(
            vertex for vertex in geometry.edge_vertices[neighbor] if vertex != endpoint
        )
        edge_direction = _difference(vertices[endpoint], vertices[other])
        slope = _dot(motion, edge_direction) / _dot(edge_direction, edge_direction)
        if slope <= 0 or motion != tuple(slope * value for value in edge_direction):
            raise BRepBlendUnsupported(
                "oblique chamfer range: inward contact motion is unproved"
            )
        limits.append(1 / slope)
        row.append(motion)
    return (row[0], row[1]), (limits[0], limits[1])


def _cutting_plane_scale_limits(
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    motions: list[tuple[_Vector, _Vector]],
    direction: _Vector,
    /,
) -> tuple[ExactScalar, ...]:
    """Prove affine cutting-plane incidence and strict retained-vertex separation."""
    normal = rational_cross(_difference(motions[0][0], motions[1][0]), direction)
    shift = _dot(normal, motions[0][0])
    if shift == 0 or any(
        _dot(normal, motion) != shift for row in motions for motion in row
    ):
        raise BRepBlendUnsupported(
            "oblique chamfer range: affine cutting-plane incidence is unproved"
        )
    anchor = vertices[corner.endpoints[0]]
    limits: list[ExactScalar] = []
    for index, point in enumerate(vertices):
        if index not in corner.endpoints:
            limit = _dot(normal, _difference(anchor, point)) / shift
            if limit <= 0:
                raise BRepBlendUnsupported(
                    "oblique chamfer range: nonincident source separation is unproved"
                )
            limits.append(limit)
    return tuple(limits)


def oblique_chamfer_parameter_range(
    model: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float],
    /,
) -> CertifiedParameterRange:
    """Solve exact affine contact, trim-order, and cutting-plane scale limits."""
    from .._rational import fraction_down

    geometry = model.geometry
    vertices = tuple(
        exact if exact is not None else rational_vector(np.asarray(point))
        for exact, point in zip(
            geometry.vertex_exact_points, geometry.vertex_points, strict=True
        )
    )
    try:
        planes = _source_planes(model, vertices)
        corner = _corner(geometry, edge_index)
        direction = _difference(
            vertices[corner.endpoints[0]], vertices[corner.endpoints[1]]
        )
        motions: list[tuple[_Vector, _Vector]] = []
        limits: list[ExactScalar] = []
        for side_index, (_, distance) in enumerate(
            zip(corner.sides, setbacks, strict=True)
        ):
            row, side_limits = _chamfer_side_range_motions(
                geometry,
                corner,
                vertices,
                planes,
                side_index,
                distance,
                direction,
            )
            limits.extend(side_limits)
            motions.append(row)
            growth = _dot(_difference(row[0], row[1]), direction)
            if growth < 0:
                limits.append(-_dot(direction, direction) / growth)
        limits.extend(_cutting_plane_scale_limits(corner, vertices, motions, direction))
        limit = min(limits)
        limit_lower, _ = coefficient_interval(limit, bits=128)
        upper = fraction_down(min(limit_lower, Fraction(float(np.finfo(np.float64).max))))
        if upper <= 0:
            raise BRepBlendUnsupported(
                "oblique chamfer range: no positive representable certified interval"
            )
    except BRepBlendUnsupported as error:
        return unknown_parameter_range("setback_scale", str(error))
    return CertifiedParameterRange(
        "setback_scale",
        (0.0, upper),
        end_status=(
            "attained_exactly",
            "attained_exactly"
            if isinstance(limit, Fraction) and Fraction(upper) == limit
            else "certified_inner_bound",
        ),
        constraints=("parameter_domain", "trim_vanishes"),
        entity_ids=((model.edge_ids[edge_index],), tuple(model.face_ids)),
        evidence=(
            f"Scale multiplies both authored setbacks {setbacks!r}, preserving their ratio.",
            "Exact rational affine contacts stay inside source cap edges, longitudinal trims remain ordered, and the cutting plane strictly separates every retained source vertex.",
            f"First contact, trim-order, or cutting-plane collision occurs at exact algebraic scale {limit}.",
        ),
    )
