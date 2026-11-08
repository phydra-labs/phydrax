#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Identity-sewn plane-pair fillets with native certified planar end trims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import pi

import numpy as np

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
)
from ...linalg._exact import factor_exact
from .._rational import fraction_down, fraction_up, rational_cross, rational_vector
from ._assembly import BRepAffinePlacement
from ._constructors import _Builder, assemble_brep_model, BRepTessellationPolicy
from ._correspondence import exact_source_lift, native_edge_pcurve
from ._fillet import BRepBlendUnresolved, BRepBlendUnsupported
from ._intersection import (
    ConicParameterEndpoint,
    exact_plane_cylinder_conic_section,
    intersect_surface_regions,
    ParametricIntersectionPolicy,
    PlaneCylinderConicSection,
    RootEndpoint,
)
from ._intersection_curve import IntersectionCurve, SurfaceRegion
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    BRepCurve,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
)
from ._root_bindings import BRepConicVertex, BRepVertexRoot


type _Vector = tuple[ExactScalar, ...]
type _Point3 = tuple[ExactScalar, ExactScalar, ExactScalar]


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
    angle_roots: tuple[ConicParameterRoot, ConicParameterRoot]
    orientation: int
    contacts: tuple[tuple[_Vector, _Vector], tuple[_Vector, _Vector]]
    heights: tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]
    height_bounds: tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]


@dataclass(frozen=True, slots=True)
class _Cap:
    curve: IntersectionCurve
    section: PlaneCylinderConicSection
    start_side: int


@dataclass(frozen=True, slots=True)
class _StagedEdges:
    builder: _Builder
    edge_map: dict[int, int]
    coedge_pcurves: dict[int, BRepCurve]
    coedge_root_overrides: dict[int, tuple[RootEndpoint | None, RootEndpoint | None]]
    contact_edges: tuple[int, int]
    contact_pcurves: tuple[BRepCurve, BRepCurve]
    contact_blend_pcurves: tuple[BRepCurve, BRepCurve]
    contact_side_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    contact_blend_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    cap_edges: tuple[int, int]
    cap_plane_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    cap_cylinder_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]


def _dot(first: _Vector, second: _Vector, /) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)),
        Fraction(0),
    )


def _cross(first: _Vector, second: _Vector, /) -> _Vector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _solve(matrix: tuple[_Vector, ...], right: _Vector, /) -> _Vector:
    try:
        factor = factor_exact(matrix)
        solution = factor.solve(tuple((value,) for value in right))
    except ValueError as error:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact tangent frame is singular"
        ) from error
    return tuple(row[0] for row in solution)


def _unit(vector: _Vector, /) -> _Vector:
    squared = _dot(vector, vector)
    if squared <= 0:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact unit frame is singular"
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
            "unsupported plane-pair fillet: exact unit frame norm is outside the rational radical base"
        )
    length = RadicalElement.sqrt(radicand)
    return tuple(value / length for value in vector)


def _lower_float(value: ExactScalar, /) -> float:
    lower, _ = coefficient_interval(value, bits=128)
    return fraction_down(lower)


def _point3(point: _Vector, /) -> _Point3:
    if len(point) != 3:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: expected a 3D exact point"
        )
    return point[0], point[1], point[2]


def _upper_float(value: ExactScalar, /) -> float:
    _, upper = coefficient_interval(value, bits=128)
    return fraction_up(upper)


def _binary(value: ExactScalar, /) -> float:
    """Return the finite binary64 realization of an exact rational coefficient."""
    try:
        result = float(value)
    except OverflowError as error:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact coordinate exceeds binary64 range"
        ) from error
    if not np.isfinite(result):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact coordinate exceeds binary64 range"
        )
    return result


def _fraction_coordinate(value: ExactScalar, /) -> Fraction:
    if not isinstance(value, Fraction):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: source vertex coordinate is not rational"
        )
    return value


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
    model: BRepModel, /
) -> tuple[BRepGeometry, tuple[_Vector, ...], tuple[_Vector, ...]]:
    """Certify the literal convex polyhedron before changing its incidence."""
    geometry = model.geometry
    assembly = geometry.assembly_graph
    if (
        len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.solid_shells != ((0,),)
        or any(sense != 1 for sense in geometry.shell_orientations[0])
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
            "unsupported plane-pair fillet: requires one unplaced literal convex planar shell"
        )
    points = tuple(
        exact if exact is not None else rational_vector(np.asarray(point))
        for exact, point in zip(
            geometry.vertex_exact_points, geometry.vertex_points, strict=True
        )
    )
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
    edge_ranges = np.asarray(geometry.edge_ranges)
    for edge, (first, last) in enumerate(geometry.edge_vertices):
        curve_index = geometry.edge_curves[edge]
        if curve_index < 0:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: source polygon edges must be native rational lines"
            )
        curve = geometry.curves[curve_index]
        if isinstance(curve, ExactLineCurve):
            if (
                curve.first_exact != points[first]
                or curve.last_exact != points[last]
                or tuple(edge_ranges[edge]) != (0.0, 1.0)
            ):
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: exact source line endpoints lack rational incidence"
                )
            continue
        if not isinstance(curve, LineCurve):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: source polygon edges must be native rational lines"
            )
        origin = curve.origin_exact
        direction = curve.direction_exact
        for parameter, vertex in zip(edge_ranges[edge], (first, last), strict=True):
            endpoint = tuple(
                a + Fraction(float(parameter)) * b
                for a, b in zip(origin, direction, strict=True)
            )
            if endpoint != points[vertex]:
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: source line endpoints lack rational incidence"
                )
    return geometry, points, tuple(normals)


def _junction(model: BRepModel, edge: int, /) -> _Junction:
    geometry, points, normals = _source_geometry(model)
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
    if not _dot(_cross(normals[sides[0]], normals[sides[1]]), axis):
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
    )
    second = _cross(junction.axis, normals[0])
    handed = 1 if _dot(second, normals[1]) > 0 else -1
    second = tuple(handed * value for value in second)
    circle = CircleCurve(
        np.zeros((3,), dtype=np.float64),
        _array(normals[0]),
        _array(second),
        1.0,
        center_exact=(Fraction(0), Fraction(0), Fraction(0)),
        first_axis_exact=normals[0],
        second_axis_exact=second,
        radius_exact=Fraction(1),
    )
    angle_roots = (
        ConicParameterRoot(circle, Fraction(1), Fraction(0)),
        ConicParameterRoot(
            circle, _dot(normals[0], normals[1]), _dot(second, normals[1])
        ),
    )
    angle = angle_roots[1].parameter
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
        origin_exact=center,
        first_axis_exact=normals[0],
        second_axis_exact=second,
        axis_exact=junction.axis,
        radius_exact=r,
    )
    heights: list[tuple[ExactScalar, ExactScalar]] = []
    contacts: list[tuple[_Vector, _Vector]] = []
    for normal in normals:
        origin = tuple(a + r * b for a, b in zip(center, normal, strict=True))
        values: list[ExactScalar] = []
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
    height_bounds: list[tuple[ExactScalar, ExactScalar]] = []
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
        angle_roots,
        handed,
        (contacts[0], contacts[1]),
        (heights[0], heights[1]),
        (height_bounds[0], height_bounds[1]),
    )


def _cap(model: BRepModel, junction: _Junction, blend: _Blend, end: int, /) -> _Cap:
    face = junction.caps[end]
    box = np.asarray(model.parameter_bounds[face])
    width = box[1] - box[0]
    # Only the cylinder's angular boundaries may terminate branch discovery;
    # the retained polygonal plane remains the authority for the cap trim.
    plane = SurfaceRegion(
        _plane(model, face),
        np.stack((box[0] - width, box[1] + width)),
    )
    lower, upper = blend.height_bounds[end]
    padding = upper - lower + 1
    cylinder = SurfaceRegion(
        blend.patch,
        np.asarray(
            (
                (0.0, _lower_float(lower - padding)),
                (blend.angle, _upper_float(upper + padding)),
            ),
            dtype=np.float64,
        ),
    )
    result = intersect_surface_regions(
        plane, cylinder, policy=ParametricIntersectionPolicy()
    )
    if result.unresolved or not result.complete:
        raise BRepBlendUnresolved(
            "plane-pair fillet unresolved: native plane/cylinder branch isolation reached its declared mathematical resolution"
        )
    if len(result.curves) != 1 or result.points or result.coincident:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: native plane/cylinder cap atlas did not certify one whole branch"
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
    angle_circle = CircleCurve(
        np.zeros((3,), dtype=np.float64),
        _array(blend.patch.first_axis_exact),
        _array(blend.patch.second_axis_exact),
        1.0,
        center_exact=(Fraction(0), Fraction(0), Fraction(0)),
        first_axis_exact=blend.patch.first_axis_exact,
        second_axis_exact=blend.patch.second_axis_exact,
        radius_exact=Fraction(1),
    )
    cylinder_angle_roots = (
        ConicParameterRoot(angle_circle, Fraction(1), Fraction(0)),
        ConicParameterRoot(
            angle_circle,
            blend.angle_roots[1].cosine_exact,
            blend.angle_roots[1].sine_exact,
        ),
    )
    section = exact_plane_cylinder_conic_section(plane, cylinder, cylinder_angle_roots)
    if section is None:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact plane/cylinder cap section is unavailable"
        )
    section_first, section_last = section.parameter_range
    if not exact_source_lift(
        section.edge_curve,
        section.plane_pcurve,
        plane,
        section_first,
        section_last,
    ) or not exact_source_lift(
        section.edge_curve,
        section.cylinder_pcurve,
        cylinder,
        section_first,
        section_last,
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact conic cap section lacks complete source lifts"
        )
    expected_points = (
        _point3(blend.contacts[0][end]),
        _point3(blend.contacts[1][end]),
    )
    if tuple(root.exact_point() for root in section.endpoint_roots) != expected_points:
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: exact cap conic roots do not match the source contacts"
        )
    return _Cap(curve, section, 0 if float(curve.node_values[0]) == 0.0 else 1)


def _trim_parameter(geometry: BRepGeometry, edge: int, point: _Vector, /) -> ExactScalar:
    curve_index = geometry.edge_curves[edge]
    if curve_index < 0:
        raise RuntimeError("plane-pair fillet cap boundary lost its admitted source line")
    curve = geometry.curves[curve_index]
    if isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    elif isinstance(curve, LineCurve):
        origin = curve.origin_exact
        direction = curve.direction_exact
    else:
        raise RuntimeError("plane-pair fillet cap boundary lost its admitted source line")
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
    /,
) -> BRepCurve:
    region = SurfaceRegion(
        _plane(model, face),
        np.asarray(model.parameter_bounds[face]),
    )
    pcurve = native_edge_pcurve(curve, region.patch)
    if not isinstance(pcurve, (ExactLineCurve, LineCurve)) or not exact_source_lift(
        curve, pcurve, region, first, last
    ):
        raise BRepBlendUnsupported(
            "unsupported plane-pair fillet: contact line lacks an exact native source-plane p-curve"
        )
    return pcurve


def _line_curve(first: _Vector, last: _Vector, /) -> LineCurve:
    direction = tuple(b - a for a, b in zip(first, last, strict=True))
    return LineCurve(
        _array(first),
        _array(direction),
        origin_exact=first,
        direction_exact=direction,
    )


def _certify_clearance(
    model: BRepModel,
    junction: _Junction,
    blend: _Blend,
    /,
) -> None:
    box = blend.patch.bounding_box(
        (
            (0.0, _lower_float(blend.height_bounds[0][0])),
            (blend.angle, _upper_float(blend.height_bounds[1][1])),
        )
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
    /,
) -> _StagedEdges:
    geometry, builder = junction.geometry, _Builder()

    def add_vertex(point: _Vector, vertex_root: BRepVertexRoot | None = None, /) -> int:
        if vertex_root is not None:
            return builder.vertex(_array(point), root=vertex_root)
        if any(not isinstance(value, Fraction) for value in point):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: nonrational source vertex lacks an exact root"
            )
        exact_point = (
            _fraction_coordinate(point[0]),
            _fraction_coordinate(point[1]),
            _fraction_coordinate(point[2]),
        )
        return builder.vertex(_array(point), exact_point=exact_point)

    def endpoint(
        root: ConicParameterRoot,
        carrier: BRepCurve,
        parameter: Fraction | None,
        point_exact: tuple[ExactScalar, ExactScalar, ExactScalar] | None = None,
        /,
    ) -> ConicParameterEndpoint:
        return ConicParameterEndpoint(
            root,
            carrier,
            parameter,
            _point3(root.exact_point()) if point_exact is None else point_exact,
        )

    contact_conics = tuple(
        tuple(caps[end].section.endpoint_roots[side] for end in range(2))
        for side in range(2)
    )
    for side, points in enumerate(blend.contacts):
        for end, point in enumerate(points):
            if contact_conics[side][end].exact_point() != _point3(point):
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: exact cap-section root does not match the source contact"
                )

    cap_line_curves: dict[tuple[int, int], LineCurve] = {}
    cap_line_endpoints: dict[tuple[int, int], int] = {}
    cap_line_pcurves: dict[tuple[int, int, int], BRepCurve] = {}
    for side, source_edges in enumerate(junction.cap_edges):
        for end, source_edge in enumerate(source_edges):
            edge_vertices = geometry.edge_vertices[source_edge]
            try:
                contact_endpoint = edge_vertices.index(junction.endpoints[end])
            except ValueError as error:
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: cap source edge lost its endpoint incidence"
                ) from error
            points: list[_Vector] = []
            for vertex in edge_vertices:
                if vertex == junction.endpoints[end]:
                    points.append(blend.contacts[side][end])
                    continue
                exact = geometry.vertex_exact_points[vertex]
                points.append(
                    exact
                    if exact is not None
                    else rational_vector(np.asarray(geometry.vertex_points[vertex]))
                )
            line = _line_curve(points[0], points[1])
            cap_line_curves[(side, end)] = line
            cap_line_endpoints[(side, end)] = contact_endpoint
            for face in (junction.caps[end], junction.sides[side]):
                cap_line_pcurves[(face, side, end)] = _contact_pcurve(
                    model, face, line, 0.0, 1.0
                )

    contact_line_curves = tuple(
        _line_curve(points[0], points[1]) for points in blend.contacts
    )
    contact_side_pcurves = tuple(
        _contact_pcurve(model, face, contact_line_curves[side], 0.0, 1.0)
        for side, face in enumerate(junction.sides)
    )
    contact_circle = CircleCurve(
        blend.patch.origin,
        blend.patch.first_axis,
        blend.patch.second_axis,
        blend.patch.radius,
        center_exact=blend.patch.origin_exact,
        first_axis_exact=blend.patch.first_axis_exact,
        second_axis_exact=blend.patch.second_axis_exact,
        radius_exact=blend.patch.radius_exact,
    )
    contact_blend_angle_roots = tuple(
        ConicParameterRoot(
            contact_circle,
            root.cosine_exact,
            root.sine_exact,
            turns=root.turns,
        )
        for root in blend.angle_roots
    )
    contact_blend_pcurves: tuple[BRepCurve, BRepCurve] = (
        ConicRootLinePCurve(
            contact_blend_angle_roots[0],
            blend.heights[0][0],
            blend.heights[0][1] - blend.heights[0][0],
            parameter_range=(0.0, 1.0),
        ),
        ConicRootLinePCurve(
            contact_blend_angle_roots[1],
            blend.heights[1][0],
            blend.heights[1][1] - blend.heights[1][0],
            parameter_range=(0.0, 1.0),
        ),
    )
    blend_region = SurfaceRegion(
        blend.patch,
        np.asarray(
            (
                (
                    0.0,
                    _lower_float(min(bounds[0] for bounds in blend.height_bounds)),
                ),
                (
                    blend.angle,
                    _upper_float(max(bounds[1] for bounds in blend.height_bounds)),
                ),
            ),
            dtype=np.float64,
        ),
    )
    for side in range(2):
        if not exact_source_lift(
            contact_line_curves[side],
            contact_blend_pcurves[side],
            blend_region,
            0.0,
            1.0,
        ):
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: exact contact line lacks its source cylinder p-curve"
            )
    vertex_roots = {
        (side, end): BRepVertexRoot(BRepConicVertex(contact_conics[side][end]))
        for side in range(2)
        for end in range(2)
    }
    contacts = tuple(
        tuple(
            add_vertex(blend.contacts[side][end], vertex_roots[(side, end)])
            for end in range(2)
        )
        for side in range(2)
    )
    vertex_map = {
        index: add_vertex(point)
        for index, point in enumerate(junction.points)
        if index not in junction.endpoints
    }
    changes: dict[int, tuple[int, _Vector, int, int]] = {
        edge: (contacts[side][end], blend.contacts[side][end], side, end)
        for side, edges in enumerate(junction.cap_edges)
        for end, edge in enumerate(edges)
    }
    replacement_curves = {
        geometry_edge: cap_line_curves[(side, end)]
        for side, source_edges in enumerate(junction.cap_edges)
        for end, geometry_edge in enumerate(source_edges)
    }
    edge_map: dict[int, int] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if edge == selected:
            continue
        if curve_index < 0:
            raise BRepBlendUnsupported(
                "unsupported plane-pair fillet: retained source edge is degenerate"
            )
        vertices: list[int] = []
        endpoint_roots: list[RootEndpoint | None] = []
        for endpoint_index, vertex in enumerate(geometry.edge_vertices[edge]):
            if edge in changes and vertex in junction.endpoints:
                replacement, _, side, end = changes[edge]
                vertices.append(replacement)
                endpoint_roots.append(
                    endpoint(
                        contact_conics[side][end],
                        replacement_curves[edge],
                        Fraction(endpoint_index),
                    )
                )
            else:
                vertices.append(vertex_map[vertex])
                endpoint_roots.append(None)
        if edge in changes:
            edge_map[edge] = builder.edge(
                replacement_curves[edge],
                0.0,
                1.0,
                vertices[0],
                vertices[1],
                endpoint_roots=(endpoint_roots[0], endpoint_roots[1]),
            )
        else:
            first, last = np.asarray(geometry.edge_ranges[edge])
            edge_map[edge] = builder.edge(
                geometry.curves[curve_index],
                float(first),
                float(last),
                vertices[0],
                vertices[1],
            )

    coedge_pcurves: dict[int, BRepCurve] = {}
    coedge_root_overrides: dict[int, tuple[RootEndpoint | None, RootEndpoint | None]] = {}
    edge_owner = {
        edge: (side, end)
        for side, source_edges in enumerate(junction.cap_edges)
        for end, edge in enumerate(source_edges)
    }
    for face, loops in enumerate(geometry.face_loops):
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                if edge not in edge_owner:
                    continue
                side, end = edge_owner[edge]
                pcurve = cap_line_pcurves[(face, side, end)]
                coedge_pcurves[coedge] = pcurve
                line_endpoint = cap_line_endpoints[(side, end)]
                roots: list[RootEndpoint | None] = [None, None]
                roots[line_endpoint] = endpoint(
                    contact_conics[side][end],
                    pcurve,
                    Fraction(line_endpoint),
                )
                coedge_root_overrides[coedge] = (roots[0], roots[1])

    contact_edges: list[int] = []
    contact_pcurves: list[BRepCurve] = []
    contact_side_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    contact_blend_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    for side, face in enumerate(junction.sides):
        for end in range(2):
            root_point = contact_blend_angle_roots[side].exact_point()
            lifted_point = tuple(
                root_point[axis] + blend.heights[side][end] * blend.patch.axis_exact[axis]
                for axis in range(3)
            )
            if lifted_point != contact_conics[side][end].exact_point():
                raise BRepBlendUnsupported(
                    "unsupported plane-pair fillet: exact cylinder generator misses its conic cap root"
                )
        contact_edge_roots = (
            endpoint(
                contact_conics[side][0],
                contact_line_curves[side],
                Fraction(0),
            ),
            endpoint(
                contact_conics[side][1],
                contact_line_curves[side],
                Fraction(1),
            ),
        )
        contact_edges.append(
            builder.edge(
                contact_line_curves[side],
                0.0,
                1.0,
                contacts[side][0],
                contacts[side][1],
                endpoint_roots=contact_edge_roots,
            )
        )
        contact_pcurves.append(contact_side_pcurves[side])
        contact_side_roots.append(
            (
                endpoint(
                    contact_conics[side][0],
                    contact_side_pcurves[side],
                    Fraction(0),
                ),
                endpoint(
                    contact_conics[side][1],
                    contact_side_pcurves[side],
                    Fraction(1),
                ),
            )
        )
        contact_blend_roots.append(
            (
                endpoint(
                    contact_blend_angle_roots[side],
                    contact_blend_pcurves[side],
                    Fraction(0),
                    _point3(contact_conics[side][0].exact_point()),
                ),
                endpoint(
                    contact_blend_angle_roots[side],
                    contact_blend_pcurves[side],
                    Fraction(1),
                    _point3(contact_conics[side][1].exact_point()),
                ),
            )
        )

    cap_edges: list[int] = []
    cap_plane_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    cap_cylinder_roots: list[tuple[RootEndpoint | None, RootEndpoint | None]] = []
    for end, cap in enumerate(caps):
        section = cap.section
        section_first, section_last = section.parameter_range
        cap_edges.append(
            builder.edge(
                section.edge_curve,
                section_first,
                section_last,
                contacts[0][end],
                contacts[1][end],
                endpoint_roots=(
                    endpoint(section.endpoint_roots[0], section.edge_curve, None),
                    endpoint(section.endpoint_roots[1], section.edge_curve, None),
                ),
            )
        )
        cap_plane_roots.append(
            (
                endpoint(section.endpoint_roots[0], section.plane_pcurve, None),
                endpoint(section.endpoint_roots[1], section.plane_pcurve, None),
            )
        )
        cap_cylinder_roots.append(
            (
                endpoint(section.endpoint_roots[0], section.cylinder_pcurve, None),
                endpoint(section.endpoint_roots[1], section.cylinder_pcurve, None),
            )
        )

    return _StagedEdges(
        builder,
        edge_map,
        coedge_pcurves,
        coedge_root_overrides,
        (contact_edges[0], contact_edges[1]),
        (contact_pcurves[0], contact_pcurves[1]),
        contact_blend_pcurves,
        (contact_side_roots[0], contact_side_roots[1]),
        (contact_blend_roots[0], contact_blend_roots[1]),
        (cap_edges[0], cap_edges[1]),
        (cap_plane_roots[0], cap_plane_roots[1]),
        (cap_cylinder_roots[0], cap_cylinder_roots[1]),
    )


def _stage_source_faces(
    model: BRepModel,
    selected: int,
    junction: _Junction,
    caps: tuple[_Cap, _Cap],
    edges: _StagedEdges,
    /,
) -> dict[int, tuple[int, int]]:
    geometry, builder = junction.geometry, edges.builder
    boundary_uses: dict[int, tuple[int, int]] = {}
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
                endpoint_roots = edges.contact_side_roots[side]
                boundary_uses[target_edge] = (sense, orientation)
            else:
                target_edge = edges.edge_map[edge]
                pcurve = edges.coedge_pcurves.get(coedge, geometry.pcurves[coedge])
                endpoint_roots = edges.coedge_root_overrides.get(
                    coedge, builder.edge_endpoint_roots[target_edge]
                )
            loop.append(
                builder.coedge(
                    target_edge,
                    sense,
                    pcurve,
                    endpoint_roots=endpoint_roots,
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
                    caps[end].section.plane_pcurve,
                    endpoint_roots=edges.cap_plane_roots[end],
                )
            )
            boundary_uses[cap_edge] = (cap_sense, orientation)
        builder.face(
            model.patches[face],
            np.asarray(model.parameter_bounds[face]),
            orientation,
            model.physical_tags[face],
            [loop],
        )
    return boundary_uses


def _stage_blend_face(
    blend: _Blend,
    caps: tuple[_Cap, _Cap],
    edges: _StagedEdges,
    boundary_uses: dict[int, tuple[int, int]],
    /,
) -> None:
    builder = edges.builder
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
    for edge in ordered:
        if edge in edges.cap_edges:
            cap_index = edges.cap_edges.index(edge)
            pcurve = caps[cap_index].section.cylinder_pcurve
            endpoint_roots = edges.cap_cylinder_roots[cap_index]
        else:
            side = edges.contact_edges.index(edge)
            pcurve = edges.contact_blend_pcurves[side]
            endpoint_roots = edges.contact_blend_roots[side]
        loop.append(
            builder.coedge(
                edge,
                blend_uses[edge][0],
                pcurve,
                endpoint_roots=endpoint_roots,
            )
        )
    boxes = np.stack(
        [
            builder.pcurves[coedge].bounding_box(
                *builder.edge_ranges[builder.coedge_edges[coedge]]
            )
            for coedge in loop
        ]
    )
    box = np.stack((np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0)))
    builder.face(blend.patch, box, blend.orientation, "cylinder", [loop])


def _contact_radius_limits(
    junction: _Junction,
    delta: _Vector,
    normals: tuple[_Vector, ...],
    /,
) -> tuple[ExactScalar, ...]:
    """Solve the affine tangent contact motion on each retained source cap edge."""
    limits: list[ExactScalar] = []
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
    /,
) -> tuple[ExactScalar, ...]:
    """Bound exact cylindrical arcs by full-circle support and cap envelopes."""
    second = _cross(junction.axis, normals[0])
    anchor = junction.points[junction.endpoints[0]]
    caps: list[tuple[ExactScalar, ExactScalar, ExactScalar]] = []
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
    limits: list[ExactScalar] = [gap / growth] if growth > 0 else []
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
) -> CertifiedParameterRange:
    """Certify affine contact limits and a conservative whole-cylinder clearance."""
    try:
        junction = _junction(model, edge_index)
        normals = tuple(_unit(junction.normals[side]) for side in junction.sides)
        delta = _solve(
            (normals[0], normals[1], junction.axis),
            (Fraction(-1), Fraction(-1), Fraction(0)),
        )
        contacts = _contact_radius_limits(junction, delta, normals)
        clearance = _clearance_radius_limits(model, junction, delta, normals)
        limit = min((*contacts, *clearance))
        limit_lower, _ = coefficient_interval(limit, bits=128)
        upper = fraction_down(min(limit_lower, Fraction(float(np.finfo(np.float64).max))))
        if upper <= 0:
            raise BRepBlendUnsupported(
                "oblique fillet range: no positive representable certified interval"
            )
    except BRepBlendUnsupported as error:
        return unknown_parameter_range("radius", str(error))
    exact = (
        limit == min(contacts)
        and isinstance(limit, Fraction)
        and Fraction(upper) == limit
    )
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
            "Exact algebraic convex source planes and inward affine tangent contact motion.",
            "Retained source edges stay positive; cap envelopes stay separated; full-circle support stays strictly inside every nonincident source plane.",
            f"Exact algebraic limiting proof value {limit}; clearance envelopes may be conservative.",
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
) -> BRepModel:
    """Replace a convex plane junction with an exact cylindrical fillet."""
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError(
            f"oblique_fillet.radius must be positive and finite; got {radius!r}"
        )
    junction = _junction(model, edge_index)
    blend = _blend(model, junction, radius)
    for side, source_edges in enumerate(junction.cap_edges):
        for end, edge in enumerate(source_edges):
            _trim_parameter(junction.geometry, edge, blend.contacts[side][end])
    _certify_clearance(model, junction, blend)
    caps = (
        _cap(model, junction, blend, 0),
        _cap(model, junction, blend, 1),
    )
    edges = _stage_edges(model, edge_index, junction, blend, caps)
    boundary_uses = _stage_source_faces(model, edge_index, junction, caps, edges)
    _stage_blend_face(blend, caps, edges, boundary_uses)
    geometry, patches, bounds, orientations, tags, trims = edges.builder.geometry(
        solid=True,
        assembly_graph=junction.geometry.assembly_graph,
    )
    return assemble_brep_model(
        geometry,
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
