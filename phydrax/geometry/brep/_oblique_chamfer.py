#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational setback chamfers of rational polyhedra, including oblique end planes."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import isqrt

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...linalg._small_batched import prepare_exact_small_linear_actions
from .._planar_primitives import reserve_fraction_work
from .._rational import rational_cross, rational_vector
from ._constructors import BRepTessellationPolicy, Builder, finish
from ._fillet import BRepBlendUnsupported
from ._model import BRepGeometry, BRepModel, BRepPCurve
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import LineCurve, PlanePatch
from ._planar_offset import _dot
from ._request_preparations import TrimRequestPreparations


type _Vector = tuple[Fraction, ...]
type _Plane = tuple[_Vector, Fraction]


@dataclass(frozen=True, slots=True)
class _Corner:
    endpoints: tuple[int, int]
    sides: tuple[int, int]
    caps: tuple[int, int]
    neighbors: tuple[tuple[int, int], tuple[int, int]]


def _difference(first: _Vector, last: _Vector, /) -> _Vector:
    return tuple(b - a for a, b in zip(first, last, strict=True))


def _binary(values: _Vector, /) -> np.ndarray:
    """Publish only rational coefficients supported by the native literal line model."""
    if any(Fraction(float(value)) != value for value in values):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact rational intersection needs nonliteral line coefficients"
        )
    return np.asarray(tuple(float(value) for value in values), dtype=np.float64)


def _actions(
    matrix: tuple[_Vector, ...],
    right: tuple[_Vector, ...],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_Vector, ...]:
    proof = prepare_exact_small_linear_actions(matrix, right, budget=budget)
    if proof.actions is None:
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: incident plane or chart system is singular"
        )
    return proof.actions


def _unit(direction: _Vector, /) -> _Vector:
    squared = _dot(direction, direction)
    numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
    if (
        numerator == 0
        or numerator * numerator != squared.numerator
        or denominator * denominator != squared.denominator
    ):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: side setback direction has no exact rational unit vector"
        )
    length = Fraction(numerator, denominator)
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
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[_Plane, ...]:
    geometry = model.require_geometry()
    if (
        geometry.solid_shells != ((0,),)
        or geometry.shell_faces != (tuple(range(len(model.patches))),)
        or geometry.shell_orientations != ((1,) * len(model.patches),)
    ):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: source requires one outward solid shell"
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
    reserve_fraction_work(
        (*vertices, *(normal for normal, _ in planes)),
        24 * len(vertices) * len(planes),
        12 * len(vertices) * len(planes),
        12,
        budget=budget,
    )
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
    for edge, indices in enumerate(geometry.edge_vertices):
        if geometry.edge_curves[edge] < 0:
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: degenerate source edges have no line carrier"
            )
        curve = geometry.curves[geometry.edge_curves[edge]]
        if not isinstance(curve, LineCurve) or any(geometry.edge_endpoint_roots[edge]):
            raise BRepBlendUnsupported(
                "planar chamfer unsupported: source edges require literal line carriers"
            )
        origin, direction = (
            rational_vector(np.asarray(curve.origin)),
            rational_vector(np.asarray(curve.direction)),
        )
        for parameter, vertex in zip(
            np.asarray(geometry.edge_ranges)[edge], indices, strict=True
        ):
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
    *,
    budget: CoordinateEnclosureBudget,
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
                budget=budget,
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
) -> tuple[Builder, dict[int, int], tuple[tuple[int, int], tuple[int, int]], set[int]]:
    builder = Builder()
    old_vertices = {
        index: builder.vertex(_binary(point))
        for index, point in enumerate(vertices)
        if index not in corner.endpoints
    }
    new_vertices = tuple(
        tuple(builder.vertex(_binary(point)) for point in row) for row in contacts
    )
    replacements = {
        (corner.neighbors[side][end], corner.endpoints[end]): new_vertices[side][end]
        for side in range(2)
        for end in range(2)
    }
    changed = {edge for row in corner.neighbors for edge in row}
    edge_map = {}
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
            origin = rational_vector(builder.vertices[start])
            direction = _difference(origin, rational_vector(builder.vertices[end]))
            curve = LineCurve(_binary(origin), _binary(direction))
            edge_map[edge] = builder.edge(curve, 0.0, 1.0, start, end)
        else:
            first_parameter, last_parameter = np.asarray(geometry.edge_ranges)[edge]
            edge_map[edge] = builder.edge(
                geometry.curves[geometry.edge_curves[edge]],
                first_parameter,
                last_parameter,
                start,
                end,
            )
    longitudinal = []
    caps = []
    for indices in new_vertices:
        origin = rational_vector(builder.vertices[indices[0]])
        direction = _difference(origin, rational_vector(builder.vertices[indices[1]]))
        longitudinal.append(
            builder.edge(
                LineCurve(_binary(origin), _binary(direction)),
                0.0,
                1.0,
                indices[0],
                indices[1],
            )
        )
    for end in range(2):
        first, last = new_vertices[0][end], new_vertices[1][end]
        origin = rational_vector(builder.vertices[first])
        direction = _difference(origin, rational_vector(builder.vertices[last]))
        caps.append(
            builder.edge(
                LineCurve(_binary(origin), _binary(direction)), 0.0, 1.0, first, last
            )
        )
    return (
        builder,
        edge_map,
        ((longitudinal[0], longitudinal[1]), (caps[0], caps[1])),
        changed,
    )


def _stage_face(
    builder: Builder,
    patch: PlanePatch,
    uses: list[tuple[int, int, BRepPCurve | None]],
    orientation: int,
    tag: str,
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
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
        budget=budget,
    )
    coordinates = {
        index: (solution[0][column], solution[1][column])
        for column, index in enumerate(indices)
    }
    loop = []
    for edge, sense, pcurve in uses:
        if pcurve is None:
            first, last = builder.edge_vertices[edge]
            uv = coordinates[first]
            pcurve = LineCurve(_binary(uv), _binary(_difference(uv, coordinates[last])))
        loop.append(
            builder.coedge(edge, sense, pcurve, preparations=preparations, budget=budget)
        )
    points = np.stack([_binary(coordinates[index]) for index in indices])
    builder.face(
        patch,
        np.stack((points.min(axis=0), points.max(axis=0))),
        [loop],
        tag,
        None,
        orientation=orientation,
    )


def _stage_source_faces(
    builder: Builder,
    model: BRepModel,
    selected: int,
    corner: _Corner,
    edge_map: dict[int, int],
    new_edges: tuple[tuple[int, int], tuple[int, int]],
    changed: set[int],
    /,
    *,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> None:
    geometry = model.require_geometry()
    longitudinal, cap_edges = new_edges
    for face, patch in enumerate(model.patches):
        if not isinstance(patch, PlanePatch):
            raise RuntimeError(
                "oblique_chamfer.source must retain its certified plane carriers; got another patch family"
            )
        uses: list[tuple[int, int, BRepPCurve | None]] = []
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
            preparations=preparations,
            budget=budget,
        )


def oblique_chamfer(
    model: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float],
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Chamfer a convex line junction while retaining its source plane incidence.

    Setbacks follow ascending incident source-face index. Each is measured
    within that side plane perpendicular to the selected edge, toward material.
    Cap equations use raw rational normals, never normalized cap normals.
    Literal carriers and both literal p-curves are certified exactly; geometry
    needing a nonliteral rational coefficient explicitly refuses publication.
    """
    if len(setbacks) != 2 or any(
        not np.isfinite(value) or value <= 0.0 for value in setbacks
    ):
        raise ValueError(
            "oblique_chamfer.setbacks must contain two positive finite distances; got an invalid specification"
        )
    geometry = model.require_geometry()
    vertices = tuple(
        rational_vector(np.asarray(point)) for point in geometry.vertex_points
    )
    planes = _source_planes(model, vertices, budget=budget)
    corner = _corner(geometry, edge_index)
    contacts = _contacts(geometry, corner, vertices, planes, setbacks, budget=budget)
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
        preparations=preparations,
        budget=budget,
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
        preparations=preparations,
        budget=budget,
    )
    return finish(
        builder,
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


def _chamfer_side_range_motions(
    geometry: BRepGeometry,
    corner: _Corner,
    vertices: tuple[_Vector, ...],
    planes: tuple[_Plane, ...],
    side_index: int,
    distance: float,
    direction: _Vector,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[_Vector, _Vector], tuple[Fraction, Fraction]]:
    """Prove affine inward motion and the opposite-edge collision on one side."""
    normal, height = planes[corner.sides[side_index]]
    transverse = _unit(rational_cross(direction, normal))
    sign = _dot(planes[corner.sides[1 - side_index]][0], transverse)
    if sign == 0:
        raise BRepBlendUnsupported("oblique chamfer range: side planes are parallel")
    if sign > 0:
        transverse = tuple(-value for value in transverse)
    row: list[_Vector] = []
    limits: list[Fraction] = []
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
            budget=budget,
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
) -> tuple[Fraction, ...]:
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
    limits: list[Fraction] = []
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
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Solve rational affine contact, trim-order, and cutting-plane scale limits."""
    from .._rational import fraction_down

    geometry = model.require_geometry()
    vertices = tuple(
        rational_vector(np.asarray(point)) for point in geometry.vertex_points
    )
    try:
        planes = _source_planes(model, vertices, budget=budget)
        corner = _corner(geometry, edge_index)
        budget.reserve(1024, 16384)
        direction = _difference(
            vertices[corner.endpoints[0]], vertices[corner.endpoints[1]]
        )
        motions: list[tuple[_Vector, _Vector]] = []
        limits: list[Fraction] = []
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
                budget=budget,
            )
            limits.extend(side_limits)
            motions.append(row)
            growth = _dot(_difference(row[0], row[1]), direction)
            if growth < 0:
                limits.append(-_dot(direction, direction) / growth)
        limits.extend(_cutting_plane_scale_limits(corner, vertices, motions, direction))
        limit = min(limits)
        upper = fraction_down(min(limit, Fraction(float(np.finfo(np.float64).max))))
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
            "attained_exactly" if Fraction(upper) == limit else "certified_inner_bound",
        ),
        constraints=("parameter_domain", "trim_vanishes"),
        entity_ids=((model.edge_ids[edge_index],), tuple(model.face_ids)),
        evidence=(
            f"Scale multiplies both authored setbacks {setbacks!r}, preserving their ratio.",
            "Exact rational affine contacts stay inside source cap edges, longitudinal trims remain ordered, and the cutting plane strictly separates every retained source vertex.",
            f"First contact, trim-order, or cutting-plane collision occurs at exact rational scale {limit}.",
        ),
    )
