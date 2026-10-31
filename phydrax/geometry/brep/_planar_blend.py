#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Rational edge blends for perpendicular-ended rational planar prisms."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from .._rational import fraction_down, rational_vector
from ._assembly import BRepAffinePlacement
from ._conic_gaps import endpoint_gap_bounds, native_face_gap_squared
from ._constructors import (
    _Builder,
    _loop_edges,
    assemble_brep_model,
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
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    BSplineCurve,
    CircleCurve,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
)
from ._planar_offset import _dot, _normal, _solve


def planar_blend(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    /,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
) -> BRepModel:
    """Replace one extruded corner with its rational tangent circle or setback line."""
    patch, indices, original, vector, uses, x_axis, y_axis, source_segments = (
        _prism_section(model, edge_index)
    )
    first = model.geometry.edge_vertices[edge_index][0]
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
    segments: list[ProfileSegment] = [
        *source_segments[:corner],
        segment,
        *source_segments[corner:],
    ]
    loop = ProfileLoop(
        tuple((float(point[0]), float(point[1])) for point in vertices),
        tuple(segments),
    )
    loop, gap_rows = _certified_profile_junctions(loop)
    profile = PlanarProfile(
        ProfilePlane(
            (float(origin[0]), float(origin[1]), float(origin[2])),
            (float(x_axis[0]), float(x_axis[1]), float(x_axis[2])),
            (float(y_axis[0]), float(y_axis[1]), float(y_axis[2])),
        ),
        loop,
        endpoint_gap_bounds=gap_rows,
    )
    prism = brep_extrusion(
        profile,
        tuple(float(value) for value in vector),
        coordinate_contract=model.coordinate_contract,
        tessellation=None,
        source_id=source_id,
    )
    source_geometry = model.geometry
    prism_geometry = prism.geometry
    geometry = BRepGeometry(
        vertex_points=prism_geometry.vertex_points,
        vertex_exact_points=prism_geometry.vertex_exact_points,
        vertex_roots=prism_geometry.vertex_roots,
        edge_endpoint_roots=prism_geometry.edge_endpoint_roots,
        coedge_endpoint_roots=prism_geometry.coedge_endpoint_roots,
        edge_vertex_gap_bounds=prism_geometry.edge_vertex_gap_bounds,
        edge_face_gap_bounds=prism_geometry.edge_face_gap_bounds,
        curves=prism_geometry.curves,
        edge_curves=prism_geometry.edge_curves,
        edge_ranges=prism_geometry.edge_ranges,
        edge_vertices=prism_geometry.edge_vertices,
        pcurves=prism_geometry.pcurves,
        coedge_edges=prism_geometry.coedge_edges,
        coedge_senses=prism_geometry.coedge_senses,
        face_loops=prism_geometry.face_loops,
        shell_faces=prism_geometry.shell_faces,
        shell_orientations=prism_geometry.shell_orientations,
        solid_shells=prism_geometry.solid_shells,
        assembly_graph=source_geometry.assembly_graph,
    )
    return assemble_brep_model(
        geometry,
        prism.patches,
        prism.parameter_bounds,
        prism.orientation,
        prism.physical_tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=prism.trim_domains,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        curve_surface_scale=model.report.curve_surface_scale,
    )


def _rectangular_section_directions(
    model: BRepModel,
    indices: tuple[int, ...],
    original: tuple[tuple[Fraction, ...], ...],
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    """Prove a four-edge orthogonal section of an eight-vertex prism."""
    geometry = model.geometry
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
) -> CertifiedParameterRange:
    """Certify a rectangular prism or reuse convex-polyhedron blend proofs.

    setback_scale multiplies both supplied setbacks, preserving their ratio;
    the supplied construction is at scale one. Nonrectangular sections require
    the whole-source convexity and clearance proof, not merely adjacent lengths.
    """
    parameter = "radius" if setbacks is None else "setback_scale"
    try:
        _, indices, original, _, uses, _, _, _ = _prism_section(model, edge_index)
    except BRepBlendUnsupported as error:
        return unknown_parameter_range(parameter, str(error))
    geometry = model.geometry
    directions = _rectangular_section_directions(model, indices, original)
    if directions is None:
        return _convex_section_parameter_range(model, edge_index, setbacks, indices, uses)
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
) -> CertifiedParameterRange:
    if setbacks is None:
        from ._oblique_fillet import oblique_fillet_parameter_range

        return oblique_fillet_parameter_range(model, edge_index)
    from ._oblique_chamfer import oblique_chamfer_parameter_range

    geometry = model.geometry
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
    return oblique_chamfer_parameter_range(model, edge_index, ordered)


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
    tuple[ProfileSegment, ...],
]:
    """Validate the source prism and its perpendicular endpoint section."""
    geometry = model.geometry
    assembly = geometry.assembly_graph
    if (
        geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.shell_orientations != ((1,) * len(model.patches),)
        or assembly.solid_count != 1
        or assembly.definitions
        or len(assembly.roots) != 1
        or assembly.roots[0].solid_index != 0
        or assembly.roots[0].placement != BRepAffinePlacement.identity()
        or any(root is not None for root in geometry.vertex_roots)
        or any(root is not None for pair in geometry.edge_endpoint_roots for root in pair)
        or any(
            root is not None for pair in geometry.coedge_endpoint_roots for root in pair
        )
    ):
        raise BRepBlendUnsupported(
            "planar blend unsupported: source requires one unplaced root-free solid shell"
        )
    first, last = geometry.edge_vertices[edge_index]
    original = tuple(
        exact if exact is not None else rational_vector(np.asarray(point))
        for exact, point in zip(
            geometry.vertex_exact_points, geometry.vertex_points, strict=True
        )
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
    from ._intersection_curve import SurfaceRegion

    cap_loop, cap_box = _merged_cap_loop(model, cap)
    source_region = SurfaceRegion(patch, cap_box)
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
    indices: list[int] = []
    source_segments: list[ProfileSegment] = []
    edge_ranges = np.asarray(geometry.edge_ranges)
    for coedge in cap_loop:
        edge = geometry.coedge_edges[coedge]
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            raise BRepBlendUnsupported("planar blend unsupported: nonline prism profile")
        source_curve = geometry.curves[curve_index]
        pcurve = geometry.pcurves[coedge]
        if isinstance(source_curve, BSplineCurve):
            first_parameter, last_parameter = source_curve.parameter_domain
            source_first, source_last = geometry.edge_vertices[edge]
            if (
                not isinstance(pcurve, BSplineCurve)
                or (first_parameter, last_parameter)
                != tuple(float(value) for value in edge_ranges[edge])
                or not np.array_equal(
                    np.asarray(source_curve.control_points)[0],
                    np.asarray(geometry.vertex_points[source_first]),
                )
                or not np.array_equal(
                    np.asarray(source_curve.control_points)[-1],
                    np.asarray(geometry.vertex_points[source_last]),
                )
                or native_face_gap_squared(
                    source_curve,
                    pcurve,
                    source_region,
                    first_parameter,
                    last_parameter,
                )
                != Fraction(0)
            ):
                raise BRepBlendUnsupported(
                    "planar blend unsupported: spline profile edge lacks an exact full-domain plane lift"
                )
        elif not isinstance(source_curve, (ExactLineCurve, LineCurve)):
            raise BRepBlendUnsupported("planar blend unsupported: nonline prism profile")
        sense = geometry.coedge_senses[coedge]
        first_vertex, last_vertex = geometry.edge_vertices[edge]
        indices.append(first_vertex if sense == 1 else last_vertex)
        if isinstance(source_curve, BSplineCurve):
            if not isinstance(pcurve, BSplineCurve):
                raise BRepBlendUnsupported(
                    "planar blend unsupported: spline edge requires a spline cap p-curve"
                )
            first_parameter, last_parameter = pcurve.parameter_domain
            if (first_parameter, last_parameter) != tuple(
                float(value) for value in edge_ranges[edge]
            ):
                raise BRepBlendUnsupported(
                    "planar blend unsupported: spline cap p-curve must span the complete edge parameter domain"
                )
            if sense == 1:
                source_segments.append(pcurve)
            else:
                knots = np.asarray(pcurve.knots)
                source_segments.append(
                    BSplineCurve(
                        np.asarray(pcurve.control_points)[::-1],
                        np.asarray(pcurve.weights)[::-1],
                        first_parameter + last_parameter - knots[::-1],
                        pcurve.degree,
                    )
                )
        elif isinstance(source_curve, (ExactLineCurve, LineCurve)) and isinstance(
            pcurve, (ExactLineCurve, LineCurve)
        ):
            source_segments.append(ProfileLine())
        else:
            raise BRepBlendUnsupported(
                "planar blend unsupported: cap p-curve has no source-preserving profile segment"
            )
    vertex_set = set(original)
    if any(
        tuple(a + b for a, b in zip(original[index], vector, strict=True))
        not in vertex_set
        for index in indices
    ):
        raise BRepBlendUnsupported(
            "planar blend unsupported: solid is not an identity-paired prism"
        )
    return (
        patch,
        tuple(indices),
        original,
        vector,
        (uses[0], uses[1]),
        x_axis,
        y_axis,
        tuple(source_segments),
    )


def _merged_cap_loop(model: BRepModel, cap: int, /) -> tuple[tuple[int, ...], np.ndarray]:
    """Merge coplanar cap fragments that share one plane chart into one loop.

    Boolean arrangements split a source cap into coplanar cells. Fragments with
    the identical plane carrier and orientation, connected through shared
    edges, form the section; edges used by two fragments are interior and drop.
    """
    geometry = model.geometry
    orientation = np.asarray(model.orientation)
    patch = model.patches[cap]
    if not isinstance(patch, PlanePatch):
        raise RuntimeError("brep_fillet.cap must retain its plane carrier.")

    def same_chart(face: int, /) -> bool:
        other = model.patches[face]
        return (
            isinstance(other, PlanePatch)
            and len(geometry.face_loops[face]) == 1
            and int(orientation[face]) == int(orientation[cap])
            and all(
                np.array_equal(np.asarray(getattr(other, name)), np.asarray(getattr(patch, name)))
                for name in ("origin", "first_axis", "second_axis")
            )
        )

    fragments = {cap}
    pending = [cap]
    while pending:
        face = pending.pop()
        for coedge in geometry.face_loops[face][0]:
            for other in model.topology.edge_faces[geometry.coedge_edges[coedge]]:
                if other not in fragments and same_chart(other):
                    fragments.add(other)
                    pending.append(other)
    if len(fragments) == 1:
        return tuple(geometry.face_loops[cap][0]), np.asarray(
            model.parameter_bounds[cap], dtype=np.float64
        )
    uses: dict[int, list[int]] = {}
    for face in fragments:
        for coedge in geometry.face_loops[face][0]:
            uses.setdefault(geometry.coedge_edges[coedge], []).append(coedge)
    boundary: dict[int, int] = {}
    for coedges in uses.values():
        if len(coedges) == 2:
            first, second = coedges
            if geometry.coedge_senses[first] == geometry.coedge_senses[second]:
                raise BRepBlendUnsupported(
                    "planar blend unsupported: coplanar cap fragments disagree in orientation"
                )
            continue
        if len(coedges) != 1:
            raise BRepBlendUnsupported(
                "planar blend unsupported: nonmanifold coplanar cap fragments"
            )
        coedge = coedges[0]
        start, end = geometry.edge_vertices[geometry.coedge_edges[coedge]]
        if geometry.coedge_senses[coedge] < 0:
            start, end = end, start
        if start in boundary:
            raise BRepBlendUnsupported(
                "planar blend unsupported: merged cap fragments do not form one simple loop"
            )
        boundary[start] = coedge
    outer = set(boundary.values())
    first_coedge = next(
        (coedge for coedge in geometry.face_loops[cap][0] if coedge in outer), None
    )
    if first_coedge is None:
        raise BRepBlendUnsupported(
            "planar blend unsupported: merged cap fragments do not form one simple loop"
        )
    loop = [first_coedge]
    while True:
        start, end = geometry.edge_vertices[geometry.coedge_edges[loop[-1]]]
        tail = end if geometry.coedge_senses[loop[-1]] > 0 else start
        following = boundary.get(tail)
        if following is None:
            raise BRepBlendUnsupported(
                "planar blend unsupported: merged cap fragments do not form one simple loop"
            )
        if following == loop[0]:
            break
        loop.append(following)
        if len(loop) > len(boundary):
            break
    if len(loop) != len(boundary):
        raise BRepBlendUnsupported(
            "planar blend unsupported: merged cap fragments do not form one simple loop"
        )
    boxes = np.stack(
        [np.asarray(model.parameter_bounds[face], dtype=np.float64) for face in fragments]
    )
    return tuple(loop), np.stack((boxes[:, 0].min(axis=0), boxes[:, 1].max(axis=0)))


def _certified_profile_junctions(
    loop: ProfileLoop, /
) -> tuple[ProfileLoop, tuple[tuple[tuple[float, float], ...], ...] | None]:
    """Retain native conics and certify mixed affine-spline junction images."""
    edges = _loop_edges(_Builder(), loop, ProfilePlane())
    if not (
        any(isinstance(edge[3], BSplineCurve) for edge in edges)
        and any(isinstance(edge[3], CircleCurve) for edge in edges)
    ):
        return loop, None
    segments = tuple(
        ProfileCurveUse(pcurve, first, last)
        if isinstance(pcurve, CircleCurve)
        else segment
        for (_, first, last, pcurve), segment in zip(
            edges, loop.segments, strict=True
        )
    )
    certified_loop = ProfileLoop(loop.vertices, segments)
    rows = tuple(
        endpoint_gap_bounds(
            pcurve,
            (first, last),
            np.asarray(
                (
                    loop.vertices[index],
                    loop.vertices[(index + 1) % len(loop.vertices)],
                ),
                dtype=np.float64,
            ),
        )
        for index, (_, first, last, pcurve) in enumerate(edges)
    )
    return certified_loop, (rows,)
