#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational edge blends for perpendicular-ended rational planar prisms."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from .._rational import fraction_down, rational_vector
from ._conic_gaps import endpoint_gap_bounds
from ._constructors import (
    _loop_edges,
    brep_extrusion,
    BRepTessellationPolicy,
    PlanarProfile,
    ProfileArc,
    ProfileCurveUse,
    ProfileLine,
    ProfileLoop,
    ProfilePlane,
    ProfileSegment,
)
from ._fillet import BRepBlendUnsupported
from ._model import BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import BSplineCurve, CircleCurve, LineCurve, PlanePatch
from ._planar_offset import _dot, _normal, _solve
from ._request_preparations import TrimRequestPreparations


def planar_blend(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Replace one extruded corner with its rational tangent circle or setback line."""
    patch, indices, original, vector, uses, x_axis, y_axis = _prism_section(
        model, edge_index
    )
    first = model.require_geometry().edge_vertices[edge_index][0]
    origin = rational_vector(np.asarray(patch.origin))
    points = tuple(
        tuple(value - shift for value, shift in zip(original[index], origin, strict=True))
        for index in indices
    )
    coordinates = tuple((_dot(point, x_axis), _dot(point, y_axis)) for point in points)
    corner = indices.index(first)
    previous, current, following = (
        coordinates[corner - 1],
        coordinates[corner],
        coordinates[(corner + 1) % len(indices)],
    )
    normals = []
    for face in uses:
        side = model.patches[face]
        if not isinstance(side, PlanePatch):
            raise RuntimeError(
                "brep_fillet.side must retain its plane carrier; got another patch family"
            )
        normal = _normal(side, int(np.asarray(model.orientation)[face]))
        if _dot(normal, vector) != 0:
            raise BRepBlendUnsupported(
                "planar blend unsupported: side is not parallel to blend edge"
            )
        normals.append((_dot(normal, x_axis), _dot(normal, y_axis)))
    incoming = tuple(
        value - start for value, start in zip(previous, current, strict=True)
    )
    if _dot(normals[0], incoming) == 0:
        before, after = normals
    else:
        after, before = normals
    outgoing = tuple(b - a for a, b in zip(current, following, strict=True))
    turn = -incoming[0] * outgoing[1] + incoming[1] * outgoing[0]
    side = 1 if turn * _polygon_area(coordinates) > 0 else -1
    if radius is not None:
        r = Fraction(radius)
        center = _solve(
            (before, after),
            (_dot(before, current) - side * r, _dot(after, current) - side * r),
            budget=budget,
        )
        start = tuple(a + side * r * b for a, b in zip(center, before, strict=True))
        end = tuple(a + side * r * b for a, b in zip(center, after, strict=True))
        segment = ProfileArc(
            (float(center[0]), float(center[1])),
            counterclockwise=turn > 0,
        )
    elif setbacks is not None:
        a, b = (Fraction(value) for value in setbacks)
        start = _setback(current, previous, a)
        end = _setback(current, following, b)
        segment = ProfileLine()
    else:
        raise ValueError(
            "brep_fillet.specification must declare a radius or two setbacks; got neither"
        )
    _strict_segment(start, previous, current)
    _strict_segment(end, current, following)
    vertices: list[tuple[Fraction, ...]] = list(coordinates)
    vertices[corner : corner + 1] = [start, end]
    segments: list[ProfileSegment] = [ProfileLine() for _ in vertices]
    segments[corner] = segment
    loop = ProfileLoop(
        tuple((float(point[0]), float(point[1])) for point in vertices),
        tuple(segments),
    )
    loop, gap_rows = _certified_profile_junctions(loop, budget=budget)
    profile = PlanarProfile(
        ProfilePlane(
            (float(origin[0]), float(origin[1]), float(origin[2])),
            (float(x_axis[0]), float(x_axis[1]), float(x_axis[2])),
            (float(y_axis[0]), float(y_axis[1]), float(y_axis[2])),
        ),
        loop,
        endpoint_gap_bounds=gap_rows,
    )
    return brep_extrusion(
        profile,
        tuple(float(value) for value in vector),
        coordinate_contract=model.coordinate_contract,
        tessellation=tessellation,
        source_id=source_id,
        preparations=preparations,
        budget=budget,
    )


def _rectangular_section_directions(
    model: BRepModel,
    indices: tuple[int, ...],
    original: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Prove a four-edge orthogonal section of an eight-vertex prism."""
    geometry = model.require_geometry()
    if (
        len(indices) != 4
        or len(model.patches) != 6
        or len(geometry.vertex_points) != 8
        or len(geometry.edge_vertices) != 12
    ):
        return None
    points = tuple(original[index] for index in indices)
    directions = tuple(
        tuple(b - a for a, b in zip(points[i], points[(i + 1) % 4], strict=True))
        for i in range(4)
    )
    if any(
        _dot(directions[i], directions[(i + 1) % 4]) != 0
        or directions[i] != tuple(-value for value in directions[(i + 2) % 4])
        for i in range(4)
    ):
        return None
    return directions


def _adjacent_section_lengths(
    directions: tuple[tuple[Fraction, ...], ...],
    corner: int,
    /,
) -> tuple[Fraction, Fraction] | None:
    """Admit only exact rational lengths of the two adjacent section edges."""
    from math import isqrt

    lengths: list[Fraction] = []
    for direction in (directions[(corner - 1) % 4], directions[corner]):
        squared = _dot(direction, direction)
        numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
        if (
            numerator * numerator != squared.numerator
            or denominator * denominator != squared.denominator
        ):
            return None
        lengths.append(Fraction(numerator, denominator))
    return lengths[0], lengths[1]


def planar_blend_parameter_range(
    model: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float] | None,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Certify a rectangular prism or reuse convex-polyhedron blend proofs.

    ``setback_scale`` multiplies both supplied setbacks, preserving their ratio;
    the supplied construction is at scale one. Nonrectangular sections require
    the whole-source convexity and clearance proof, not merely adjacent lengths.
    """
    parameter = "radius" if setbacks is None else "setback_scale"
    budget.reserve(256, 4096)
    try:
        _, indices, original, _, uses, _, _ = _prism_section(model, edge_index)
    except BRepBlendUnsupported as error:
        return unknown_parameter_range(parameter, str(error))
    geometry = model.require_geometry()
    directions = _rectangular_section_directions(model, indices, original)
    if directions is None:
        return _convex_section_parameter_range(
            model, edge_index, setbacks, indices, uses, budget=budget
        )
    corner = indices.index(geometry.edge_vertices[edge_index][0])
    lengths = _adjacent_section_lengths(directions, corner)
    if lengths is None:
        return unknown_parameter_range(
            parameter, "adjacent trim extent lacks an exact rational length"
        )
    limits = tuple(
        length if setbacks is None else length / Fraction(setbacks[i])
        for i, length in enumerate(lengths)
    )
    limit = min(limits)
    upper = fraction_down(min(limit, Fraction(float(np.finfo(np.float64).max))))
    if upper <= 0:
        return unknown_parameter_range(
            parameter, "no positive representable certified interval"
        )
    exact = Fraction(upper) == limit
    neighbors = (
        frozenset((indices[(corner - 1) % 4], indices[corner])),
        frozenset((indices[corner], indices[(corner + 1) % 4])),
    )
    limiting_edges = tuple(
        model.edge_ids[index]
        for index, vertices in enumerate(geometry.edge_vertices)
        if any(
            value == limit and frozenset(vertices) == neighbors[i]
            for i, value in enumerate(limits)
        )
    )
    return CertifiedParameterRange(
        parameter,
        (0.0, upper),
        end_status=(
            "attained_exactly",
            "attained_exactly" if exact else "certified_inner_bound",
        ),
        constraints=("parameter_domain", "face_vanishes"),
        entity_ids=((model.edge_ids[edge_index],), limiting_edges),
        evidence=(
            "Exact rectangular-prism incidence and rational orthogonal section edges.",
            "A radius consumes its own length on each adjacent trim; both retained lengths must be positive."
            if setbacks is None
            else f"Scale multiplies authored setbacks {setbacks!r}; both adjacent retained trim lengths must be positive.",
            f"First adjacent opposite-trim collision is the exact rational value {limit}.",
        ),
    )


def _convex_section_parameter_range(
    model: BRepModel,
    edge_index: int,
    setbacks: tuple[float, float] | None,
    indices: tuple[int, ...],
    uses: tuple[int, int],
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    if setbacks is None:
        from ._oblique_fillet import oblique_fillet_parameter_range

        return oblique_fillet_parameter_range(model, edge_index, budget=budget)
    from ._oblique_chamfer import oblique_chamfer_parameter_range

    geometry = model.require_geometry()
    corner = indices.index(geometry.edge_vertices[edge_index][0])
    previous = frozenset((indices[(corner - 1) % len(indices)], indices[corner]))
    previous_edge = next(
        index
        for index, vertices in enumerate(geometry.edge_vertices)
        if frozenset(vertices) == previous
    )
    # Planar construction orders setbacks by profile predecessor/successor;
    # polyhedron construction orders them by incident source-face index.
    ordered = (
        setbacks
        if uses[0] in model.topology.edge_faces[previous_edge]
        else (setbacks[1], setbacks[0])
    )
    return oblique_chamfer_parameter_range(model, edge_index, ordered, budget=budget)


def _polygon_area(points: tuple[tuple[Fraction, Fraction], ...], /) -> Fraction:
    return sum(
        (
            a[0] * b[1] - a[1] * b[0]
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        ),
        Fraction(),
    )


def _strict_segment(
    point: tuple[Fraction, ...],
    first: tuple[Fraction, ...],
    last: tuple[Fraction, ...],
    /,
) -> None:
    direction = tuple(b - a for a, b in zip(first, last, strict=True))
    offset = tuple(b - a for a, b in zip(first, point, strict=True))
    parameter = _dot(offset, direction) / _dot(direction, direction)
    if (
        parameter <= 0
        or parameter >= 1
        or any(a != parameter * b for a, b in zip(offset, direction, strict=True))
    ):
        raise BRepBlendUnsupported(
            "planar blend refuses: setback reaches a neighboring edge or face vanishes"
        )


def _setback(
    corner: tuple[Fraction, ...], endpoint: tuple[Fraction, ...], distance: Fraction, /
) -> tuple[Fraction, ...]:
    from math import isqrt

    direction = tuple(b - a for a, b in zip(corner, endpoint, strict=True))
    squared = _dot(direction, direction)
    numerator, denominator = isqrt(squared.numerator), isqrt(squared.denominator)
    if (
        numerator * numerator != squared.numerator
        or denominator * denominator != squared.denominator
    ):
        raise BRepBlendUnsupported(
            "planar chamfer unsupported: exact rational unit direction unavailable"
        )
    length = Fraction(numerator, denominator)
    return tuple(
        a + distance * b / length for a, b in zip(corner, direction, strict=True)
    )


def _prism_section(
    model: BRepModel,
    edge_index: int,
    /,
) -> tuple[
    PlanePatch,
    tuple[int, ...],
    tuple[tuple[Fraction, ...], ...],
    tuple[Fraction, ...],
    tuple[int, int],
    tuple[Fraction, ...],
    tuple[Fraction, ...],
]:
    """Validate the source prism and its perpendicular endpoint section."""
    geometry = model.require_geometry()
    first, last = geometry.edge_vertices[edge_index]
    original = tuple(
        rational_vector(np.asarray(point)) for point in geometry.vertex_points
    )
    vector = tuple(b - a for a, b in zip(original[first], original[last], strict=True))
    uses = []
    cap = None
    for face, loops in enumerate(geometry.face_loops):
        if len(loops) != 1 or not isinstance(model.patches[face], PlanePatch):
            raise BRepBlendUnsupported(
                "planar blend unsupported: all faces must be single-loop planes"
            )
        edges = tuple(geometry.coedge_edges[coedge] for coedge in loops[0])
        if edge_index in edges:
            uses.append(face)
        vertices = {vertex for edge in edges for vertex in geometry.edge_vertices[edge]}
        if first in vertices and last not in vertices:
            cap = face
    if len(uses) != 2 or cap is None:
        raise BRepBlendUnsupported(
            "planar blend unsupported: edge requires two side faces and rational planar end trims"
        )
    patch = model.patches[cap]
    if not isinstance(patch, PlanePatch):
        raise RuntimeError(
            "brep_fillet.cap must retain its plane carrier; got another patch family"
        )
    x_axis, y_axis = (
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )
    if (
        _dot(x_axis, x_axis) != 1
        or _dot(y_axis, y_axis) != 1
        or _dot(x_axis, y_axis) != 0
        or _dot(vector, x_axis) != 0
        or _dot(vector, y_axis) != 0
    ):
        raise BRepBlendUnsupported(
            "planar blend unsupported: end trim requires a noncircular analytic conic"
        )
    indices = []
    for coedge in geometry.face_loops[cap][0]:
        edge = geometry.coedge_edges[coedge]
        if not isinstance(geometry.curves[geometry.edge_curves[edge]], LineCurve):
            raise BRepBlendUnsupported("planar blend unsupported: nonline prism profile")
        a, b = geometry.edge_vertices[edge]
        indices.append(a if geometry.coedge_senses[coedge] == 1 else b)
    vertex_set = set(original)
    if any(
        tuple(a + b for a, b in zip(original[index], vector, strict=True))
        not in vertex_set
        for index in indices
    ):
        raise BRepBlendUnsupported(
            "planar blend unsupported: solid is not an identity-paired prism"
        )
    return patch, tuple(indices), original, vector, (uses[0], uses[1]), x_axis, y_axis


def _certified_profile_junctions(
    loop: ProfileLoop,
    /,
    *,
    budget: CoordinateEnclosureBudget,
) -> tuple[ProfileLoop, tuple[tuple[tuple[float, float], ...], ...] | None]:
    """Retain native conics and certify mixed affine-spline junction images."""
    edges = _loop_edges(loop)
    if not (
        any(isinstance(edge.pcurve, BSplineCurve) for edge in edges)
        and any(isinstance(edge.pcurve, CircleCurve) for edge in edges)
    ):
        return loop, None
    segments = tuple(
        ProfileCurveUse(edge.pcurve, edge.first, edge.last)
        if isinstance(edge.pcurve, CircleCurve)
        else segment
        for edge, segment in zip(edges, loop.segments, strict=True)
    )
    certified_loop = ProfileLoop(loop.vertices, segments)
    rows = tuple(
        endpoint_gap_bounds(
            edge.pcurve,
            (edge.first, edge.last),
            np.asarray((edge.start, edge.end), dtype=np.float64),
            budget=budget,
        )
        for edge in _loop_edges(certified_loop)
    )
    return certified_loop, (rows,)
