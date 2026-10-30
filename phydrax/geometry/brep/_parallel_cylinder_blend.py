#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified local blends of parallel cylindrical extrusion faces.

The admitted source is a single straight junction, two rectangular cylindrical
charts, and perpendicular planar end caps.  A circle-offset construction selects
one branch by exact rational radical signs; topology is rebuilt with shared edge
indices, never by proximity sewing.  Oblique, root-authored, and nonrectangular
end trims are outside this leaf's certified family.
"""

from __future__ import annotations

from fractions import Fraction
from math import atan2, pi

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from .._interval_enclosure import (
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from .._rational import fraction_down, fraction_up, rational_vector
from ..design._schema import (
    _ParameterCollector,
    DesignState,
    ParameterBinding,
    ParameterId,
)
from ._assembly import BRepAffinePlacement
from ._constructors import _Builder, assemble_brep_model, BRepTessellationPolicy
from ._differentiable import (
    _RealizedConicCurve,
    _RealizedCylinderPatch,
    _RealizedLineCurve,
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._exact import fraction_norm_upper
from ._fillet import BRepBlendUnsupported
from ._intersection import (
    _ruled_plane_cap_section,
    BranchRootEndpoint,
    ConicParameterEndpoint,
    ConicRootLineEndpointWitness,
    CurveRange,
    CurveSurfaceIntersectionRoot,
    intersect_trim_curves,
    IntersectionCurvePointRoot,
    ParametricIntersectionPolicy,
    RootEndpoint,
    TrimIntersectionRoot,
    TrimRootEndpoint,
)
from ._intersection_curve import IntersectionPCurve, SurfaceRegion
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    AbstractCurve,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RuledSurface,
    SurfaceMappedCurve,
)
from ._root_bindings import (
    BRepConicVertex,
    BRepCurveSurfaceLift,
    BRepRootSupport,
    BRepVertexRoot,
)


class _Side(StrictModule):
    face: int
    patch: CylinderPatch
    coedge: int
    angle: float
    retained_angle: float
    retained_angle_bounds: tuple[float, float]
    sector_directions: tuple[
        tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]
    ]
    cap_edges: tuple[int, int]
    heights: tuple[float, float]
    heights_exact: tuple[Fraction, Fraction]
    radial_orientation: int
    source_root: ConicParameterRoot
    source_phase_shift: Fraction


class _Junction(StrictModule):
    geometry: BRepGeometry
    sides: tuple[_Side, _Side]
    cap_faces: tuple[int, int]
    vertices: tuple[int, int]
    origin: np.ndarray
    origin_exact: _ExactPoint3
    first: np.ndarray
    first_exact: _Vector3
    second: np.ndarray
    second_exact: _Vector3
    axis: np.ndarray
    axis_exact: _Vector3
    centers: np.ndarray
    centers_exact: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    height: float
    height_exact: Fraction


class _Blend(StrictModule):
    points: np.ndarray
    angles: tuple[float, float]
    center: np.ndarray
    radius: float
    sweep: float
    orientation: int
    contact_unit_bases: tuple[_Vector, _Vector] | None = None
    contact_unit_radicals: tuple[_Vector, _Vector] | None = None
    contact_radicand: Fraction | None = None
    center_base: _Vector | None = None
    center_radical: _Vector | None = None
    contact_roots: tuple[ConicParameterRoot, ConicParameterRoot] | None = None
    phase_shifts_exact: tuple[Fraction, Fraction] | None = None


type _Vector = tuple[Fraction, ...]
type _ExactVector = tuple[ExactScalar, ...]
type _Vector3 = tuple[Fraction, Fraction, Fraction]
type _ExactPoint3 = tuple[ExactScalar, ExactScalar, ExactScalar]
type _ContactPointPair = tuple[_ExactPoint3, _ExactPoint3]
type _ContactPoints = tuple[_ContactPointPair, _ContactPointPair]


def _offset_exact_point(point: _ExactVector, offset: _Vector3, /) -> _ExactPoint3:
    return (
        point[0] + offset[0],
        point[1] + offset[1],
        point[2] + offset[2],
    )




def _fraction_exact(value: ExactScalar, /) -> Fraction:
    if not isinstance(value, Fraction):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: this analytic branch requires rational frame coefficients"
        )
    return value


def _rational_exact_vector(values: _ExactVector, /) -> _Vector:
    return tuple(_fraction_exact(value) for value in values)
def _rational_exact_vector3(values: _ExactVector, /) -> _Vector3:
    return (
        _fraction_exact(values[0]),
        _fraction_exact(values[1]),
        _fraction_exact(values[2]),
    )


def _rational_vector3(values: np.ndarray, /) -> _Vector3:
    vector = rational_vector(values)
    return vector[0], vector[1], vector[2]




def _rational_rows(values: np.ndarray, /) -> tuple[_Vector, ...]:
    rows = np.asarray(values, dtype=np.float64)
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise BRepBlendUnsupported(
            "parallel-cylinder blend requires finite represented source coefficients"
        )
    return tuple(rational_vector(row) for row in rows)


def _exact_frame(patch: CylinderPatch | PlanePatch, /) -> None:
    if isinstance(patch, CylinderPatch):
        axes = (
            _rational_exact_vector3(patch.first_axis_exact),
            _rational_exact_vector3(patch.second_axis_exact),
        )
        axial = _rational_exact_vector3(patch.axis_exact)
    else:
        axes = (
            _rational_vector3(np.asarray(patch.first_axis)),
            _rational_vector3(np.asarray(patch.second_axis)),
        )
        axial = None
    for i, left in enumerate(axes):
        for j, right in enumerate(axes):
            if sum((a * b for a, b in zip(left, right, strict=True)), Fraction()) != (
                i == j
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: radial frame lacks a rational orthonormal certificate"
                )
    if axial is not None and (
        not any(axial)
        or any(
            sum((a * b for a, b in zip(axial, row, strict=True)), Fraction()) != 0
            for row in axes
        )
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: axial chart is not exactly normal to its radial frame"
        )


def _unit_axis(side: _Side, /) -> _Vector3:
    first = _rational_exact_vector3(side.patch.first_axis_exact)
    second = _rational_exact_vector3(side.patch.second_axis_exact)
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _axis_scale(side: _Side, /) -> Fraction:
    unit = _unit_axis(side)
    axial = _rational_exact_vector3(side.patch.axis_exact)
    squared = sum((value * value for value in unit), Fraction())
    scale = sum((a * b for a, b in zip(axial, unit, strict=True)), Fraction()) / squared
    if scale == 0 or axial != tuple(scale * value for value in unit):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: axes are not exactly parallel"
        )
    return scale


def _contact_direction(
    blend: _Blend, side: int, /, *, sign: int = 1
) -> tuple[ExactScalar, ExactScalar]:
    if (
        blend.contact_unit_bases is None
        or blend.contact_unit_radicals is None
        or blend.contact_radicand is None
    ):
        raise BRepBlendUnsupported(
            "parallel-cylinder fillet contact lacks an exact quadratic-root definition"
        )
    square_root = RadicalElement.sqrt(blend.contact_radicand)
    bases = blend.contact_unit_bases[side]
    radicals = blend.contact_unit_radicals[side]
    return (
        Fraction(bases[0]) + sign * square_root * Fraction(radicals[0]),
        Fraction(bases[1]) + sign * square_root * Fraction(radicals[1]),
    )


def _root_turns(cosine: ExactScalar, sine: ExactScalar, hint: float, /) -> int:
    principal = atan2(float(sine), float(cosine))
    if principal < 0.0:
        principal += 2.0 * pi
    return int(round((hint - principal) / (2.0 * pi)))


def _conic_root(
    conic: CircleCurve,
    direction: tuple[ExactScalar, ExactScalar],
    angle_hint: float,
    /,
) -> ConicParameterRoot:
    return ConicParameterRoot(
        conic,
        direction[0],
        direction[1],
        turns=_root_turns(direction[0], direction[1], angle_hint),
    )


def _blend_center_exact(
    junction: _Junction, blend: _Blend, /
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    center_base = blend.center_base
    center_radical = blend.center_radical
    radicand = blend.contact_radicand
    if center_base is None or center_radical is None or radicand is None:
        raise BRepBlendUnsupported(
            "parallel-cylinder fillet center lacks its exact quadratic branch"
        )
    square_root = RadicalElement.sqrt(radicand)
    def coordinate(index: int, /) -> ExactScalar:
        return (
            junction.origin_exact[index]
            + center_base[0] * junction.first_exact[index]
            + center_base[1] * junction.second_exact[index]
            + square_root
            * (
                center_radical[0] * junction.first_exact[index]
                + center_radical[1] * junction.second_exact[index]
            )
        )

    return coordinate(0), coordinate(1), coordinate(2)


def _source_contact_root(
    junction: _Junction, blend: _Blend, side_index: int, /
) -> ConicParameterRoot:
    side = junction.sides[side_index]
    patch = side.patch
    conic = CircleCurve(
        patch.origin,
        patch.first_axis,
        patch.second_axis,
        patch.radius,
        center_exact=patch.origin_exact,
        first_axis_exact=patch.first_axis_exact,
        second_axis_exact=patch.second_axis_exact,
        radius_exact=patch.radius_exact,
    )
    return _conic_root(
        conic, _contact_direction(blend, side_index), blend.angles[side_index]
    )


def _blend_contact_root(
    junction: _Junction,
    blend: _Blend,
    side_index: int,
    center_exact: tuple[ExactScalar, ExactScalar, ExactScalar],
    center_numeric: np.ndarray,
    /,
) -> ConicParameterRoot:
    side = junction.sides[side_index]
    sign = blend.orientation * side.radial_orientation
    direction = _contact_direction(blend, side_index, sign=sign)
    hint = blend.angles[side_index] + (0.0 if sign > 0 else pi)
    radius_exact = Fraction.from_float(float(blend.radius))
    conic = CircleCurve(
        center_numeric,
        junction.first,
        junction.second,
        blend.radius,
        center_exact=center_exact,
        first_axis_exact=junction.first_exact,
        second_axis_exact=junction.second_exact,
        radius_exact=radius_exact,
    )
    return _conic_root(conic, direction, hint)


def _side(model: BRepModel, geometry: BRepGeometry, face: int, edge: int) -> _Side:
    patch = model.patches[face]
    if not isinstance(patch, CylinderPatch):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: both adjacent faces must be cylinders"
        )
    loops = geometry.face_loops[face]
    if len(loops) != 1 or len(loops[0]) != 4:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: cylindrical chart is not a single rectangle"
        )
    selected = [c for c in loops[0] if geometry.coedge_edges[c] == edge]
    if len(selected) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: seam or repeated junction use"
        )
    coedge = selected[0]
    pcurve = geometry.pcurves[coedge]
    if not isinstance(pcurve, ConicRootLinePCurve) or pcurve.height_direction_exact == 0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: junction lacks an exact rooted angular isoline"
        )
    angle = float(pcurve.angle)
    selected_direction = (pcurve.root.cosine_exact, pcurve.root.sine_exact)
    if any(
        not isinstance(value, (Fraction, RadicalElement)) for value in selected_direction
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: source angular root lacks an exact algebraic sector ray"
        )
    caps: list[tuple[float, int, Fraction]] = []
    other_angles: list[tuple[float, float, float, tuple[ExactScalar, ExactScalar]]] = []
    for c in loops[0]:
        pc = geometry.pcurves[c]
        if isinstance(pc, LineCurve):
            origin, direction = pc.origin_exact, pc.direction_exact
            if direction == (Fraction(1), Fraction(0)):
                height_exact = _fraction_exact(origin[1])
                caps.append((float(height_exact), geometry.coedge_edges[c], height_exact))
            elif direction[0] == 0 and direction[1] != 0:
                if c != coedge:
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder blend: opposite generator lacks an exact conic root"
                    )
            else:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: noncanonical rectangular chart"
                )
        elif isinstance(pc, ConicRootLinePCurve):
            if pc.height_direction_exact == 0:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: nonaxial conic root p-curve"
                )
            direction = (pc.root.cosine_exact, pc.root.sine_exact)
            if c != coedge:
                if any(
                    not isinstance(value, (Fraction, RadicalElement))
                    for value in direction
                ):
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder blend: retained angular root lacks an exact algebraic sector ray"
                    )
                other_angles.append(
                    (
                        float(pc.angle),
                        pc.root.parameter_lower,
                        pc.root.parameter_upper,
                        direction,
                    )
                )
        else:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: rectangle requires native line p-curves"
            )
    if len(caps) != 2 or len(other_angles) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: missing circle end boundaries"
        )
    caps.sort()
    retained = other_angles[0][0]
    retained_bounds = (other_angles[0][1], other_angles[0][2])
    sector_directions = (selected_direction, other_angles[0][3])
    if not 0.0 < abs(retained - angle) < 2.0 * pi:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: vanishing or multiply wound source trim"
        )
    for _, cap, _ in caps:
        first, last = np.asarray(geometry.edge_ranges[cap])
        if (float(first), float(last)) != tuple(sorted((angle, retained))):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: cap range does not share cylinder angle"
            )
    handed = float(
        np.cross(np.asarray(patch.first_axis), np.asarray(patch.second_axis))
        @ np.asarray(patch.axis)
    )
    radial = int(float(model.orientation[face])) * (1 if handed > 0.0 else -1)
    return _Side(
        face,
        patch,
        coedge,
        angle,
        retained,
        retained_bounds,
        sector_directions,
        (caps[0][1], caps[1][1]),
        (caps[0][0], caps[1][0]),
        (caps[0][2], caps[1][2]),
        radial,
        pcurve.root,
        pcurve.phase_shift_exact,
    )


def _end_caps(
    model: BRepModel,
    sides: tuple[_Side, _Side],
    axis: np.ndarray,
    /,
) -> tuple[int, int]:
    cap_faces: list[int] = []
    for end in range(2):
        uses: list[set[int]] = []
        for side in sides:
            same = _axis_scale(side) > 0
            cap_edge = side.cap_edges[end if same else 1 - end]
            uses.append(set(model.topology.edge_faces[cap_edge]) - {side.face})
        candidate = uses[0] & uses[1]
        if len(candidate) != 1:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: end circles must share one cap plane"
            )
        cap_face = next(iter(candidate))
        patch = model.patches[cap_face]
        if not isinstance(patch, PlanePatch):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: end trim is not planar"
            )
        _exact_frame(patch)
        rows = (
            _unit_axis(sides[0]),
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )
        if any(
            sum((a * b for a, b in zip(rows[0], row, strict=True)), Fraction()) != 0
            for row in rows[1:]
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: oblique cap has no certified circular trim"
            )
        cap_faces.append(cap_face)
    if cap_faces[0] == cap_faces[1]:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: coincident end caps"
        )
    return cap_faces[0], cap_faces[1]


def _require_simple_vertices(
    geometry: BRepGeometry,
    sides: tuple[_Side, _Side],
    vertices: tuple[int, int],
    edge: int,
) -> None:
    for vertex in vertices:
        incident = {i for i, pair in enumerate(geometry.edge_vertices) if vertex in pair}
        expected = {
            edge,
            *(
                cap
                for side in sides
                for cap in side.cap_edges
                if vertex in geometry.edge_vertices[cap]
            ),
        }
        if incident != expected or len(incident) != 3:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: junction endpoint requires a corner patch"
            )


def _conic_vertex(root: object | None, /) -> bool:
    return isinstance(root, BRepVertexRoot) and isinstance(root.primary, BRepConicVertex)


def _junction(model: BRepModel, edge: int, /) -> _Junction:
    geometry = model.geometry
    if not 0 <= edge < len(geometry.edge_curves):
        raise BRepBlendUnsupported("parallel-cylinder blend edge index is out of range")
    assembly = geometry.assembly_graph
    if (
        assembly.solid_count != 1
        or geometry.solid_shells != ((0,),)
        or len(geometry.shell_faces) != 1
        or set(geometry.shell_faces[0]) != set(range(len(model.patches)))
        or geometry.shell_orientations != ((1,) * len(model.patches),)
        or assembly.definitions
        or len(assembly.roots) != 1
        or assembly.roots[0].solid_index != 0
        or assembly.roots[0].placement != BRepAffinePlacement.identity()
        or any(
            root is not None and not _conic_vertex(root) for root in geometry.vertex_roots
        )
        or any(
            root is not None and not isinstance(root, ConicParameterEndpoint)
            for pair in geometry.edge_endpoint_roots
            for root in pair
        )
        or any(
            root is not None and not isinstance(root, ConicParameterEndpoint)
            for pair in geometry.coedge_endpoint_roots
            for root in pair
        )
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: source must be one identity occurrence with only checked conic endpoint roots and one solid shell"
        )

    faces = model.topology.edge_faces[edge]
    if len(faces) != 2:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: junction must have two face uses"
        )
    sides = (
        _side(model, geometry, faces[0], edge),
        _side(model, geometry, faces[1], edge),
    )
    for side in sides:
        _exact_frame(side.patch)
    first, second = (
        np.asarray(sides[0].patch.first_axis),
        np.asarray(sides[0].patch.second_axis),
    )
    axis = np.asarray(tuple(float(value) for value in _unit_axis(sides[0])))
    for side in sides:
        _axis_scale(side)
    if not (
        np.array_equal(np.asarray(sides[1].patch.first_axis), first)
        and np.array_equal(np.asarray(sides[1].patch.second_axis), second)
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: exact source lift requires a shared radial frame"
        )
    endpoint_ids = geometry.edge_vertices[edge]
    edge_roots = geometry.edge_endpoint_roots[edge]
    for endpoint, vertex in enumerate(endpoint_ids):
        vertex_root = geometry.vertex_roots[vertex]
        edge_root = edge_roots[endpoint]
        if (
            not isinstance(vertex_root, BRepVertexRoot)
            or not isinstance(vertex_root.primary, BRepConicVertex)
            or not isinstance(edge_root, ConicParameterEndpoint)
            or edge_root.point_exact != vertex_root.primary.point_exact
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: source junction endpoint lacks a conic line witness"
            )
        for side in sides:
            root = geometry.coedge_endpoint_roots[side.coedge][endpoint]
            if (
                not isinstance(root, ConicParameterEndpoint)
                or root.point_exact != vertex_root.primary.point_exact
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: cylinder generator p-curve lacks a conic endpoint witness"
                )
    def exact_vertex(vertex: int, point: np.ndarray, /) -> _ExactPoint3:
        exact = geometry.vertex_exact_points[vertex]
        if exact is not None:
            return exact
        root = geometry.vertex_roots[vertex]
        if isinstance(root, BRepVertexRoot) and isinstance(
            root.primary, BRepConicVertex
        ):
            exact_point = root.primary.point_exact
            return exact_point[0], exact_point[1], exact_point[2]
        values = rational_vector(point)
        return values[0], values[1], values[2]
    endpoints = np.asarray(geometry.vertex_points)[list(endpoint_ids)]
    order = np.argsort(endpoints @ axis)
    vertices = (endpoint_ids[int(order[0])], endpoint_ids[int(order[1])])
    origin = endpoints[int(order[0])]
    height = float((endpoints[int(order[1])] - origin) @ axis)

    origin_exact = exact_vertex(vertices[0], endpoints[int(order[0])])
    endpoint_exact = exact_vertex(vertices[1], endpoints[int(order[1])])
    first_exact = _rational_exact_vector3(sides[0].patch.first_axis_exact)
    second_exact = _rational_exact_vector3(sides[0].patch.second_axis_exact)
    axis_exact = _unit_axis(sides[0])
    height_value = (
        (endpoint_exact[0] - origin_exact[0]) * axis_exact[0]
        + (endpoint_exact[1] - origin_exact[1]) * axis_exact[1]
        + (endpoint_exact[2] - origin_exact[2]) * axis_exact[2]
    )
    height_exact = _fraction_exact(height_value)
    if height <= 0.0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: vanishing junction length"
        )
    cap_faces = _end_caps(model, sides, axis)
    _require_simple_vertices(geometry, sides, vertices, edge)
    centers_raw = tuple(
        tuple(
            sum(
                (
                    (value - origin_exact[axis]) * frame[axis]
                    for axis, value in enumerate(side.patch.origin_exact)
                ),
                Fraction(),
            )
            for frame in (first_exact, second_exact)
        )
        for side in sides
    )
    centers_exact = (
        (_fraction_exact(centers_raw[0][0]), _fraction_exact(centers_raw[0][1])),
        (_fraction_exact(centers_raw[1][0]), _fraction_exact(centers_raw[1][1])),
    )
    centers = np.asarray(
        tuple(tuple(float(value) for value in row) for row in centers_exact),
        dtype=np.float64,
    )
    junction = _Junction(
        geometry,
        sides,
        (cap_faces[0], cap_faces[1]),
        vertices,
        origin,
        origin_exact,
        first,
        first_exact,
        second,
        second_exact,
        axis,
        axis_exact,
        centers,
        centers_exact,
        height,
        height_exact,
    )
    _certify_section_coordinates(junction)
    return junction


def _certify_section_coordinates(junction: _Junction, /) -> None:
    """Prove the planar reduction preserves the authored rational coefficients."""
    rows = (
        junction.origin_exact,
        junction.sides[0].patch.origin_exact,
        junction.sides[1].patch.origin_exact,
        junction.first_exact,
        junction.second_exact,
        junction.axis_exact,
        tuple(
            junction.origin_exact[index]
            + junction.height_exact * junction.axis_exact[index]
            for index in range(3)
        ),
    )
    for side in range(2):
        displacement = tuple(a - b for a, b in zip(rows[side + 1], rows[0], strict=True))
        for coordinate in range(2):
            projected = sum(
                (a * b for a, b in zip(displacement, rows[coordinate + 3], strict=True)),
                Fraction(),
            )
            if projected != junction.centers_exact[side][coordinate]:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: planar reduction needs an exact coefficient carrier"
                )
    axial = tuple(a - b for a, b in zip(rows[6], rows[0], strict=True))
    for frame in rows[3:5]:
        if sum((a * b for a, b in zip(axial, frame, strict=True)), Fraction()) != 0:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: junction is not a straight axial segment"
            )
    if (
        sum((a * b for a, b in zip(axial, rows[5], strict=True)), Fraction())
        != junction.height_exact
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: axial range needs an exact coefficient carrier"
        )


def _angle(side: _Side, point: np.ndarray, junction: _Junction) -> float:
    world = junction.origin + point[0] * junction.first + point[1] * junction.second
    radial = world - np.asarray(side.patch.origin)
    angle = atan2(
        float(radial @ np.asarray(side.patch.second_axis)),
        float(radial @ np.asarray(side.patch.first_axis)),
    )
    angle += 2.0 * pi * round((side.angle - angle) / (2.0 * pi))
    return angle


def _sector_rays(
    side: _Side,
    junction: _Junction,
    /,
) -> tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]]:
    first_axis = side.patch.first_axis_exact
    second_axis = side.patch.second_axis_exact
    target_axes = (junction.first_exact, junction.second_exact)

    def project(
        direction: tuple[ExactScalar, ExactScalar], /
    ) -> tuple[ExactScalar, ExactScalar]:
        radial = tuple(
            direction[0] * first_axis[coordinate] + direction[1] * second_axis[coordinate]
            for coordinate in range(3)
        )
        return (
            sum(
                (radial[coordinate] * target_axes[0][coordinate] for coordinate in range(3)),
                Fraction(),
            ),
            sum(
                (radial[coordinate] * target_axes[1][coordinate] for coordinate in range(3)),
                Fraction(),
            ),
        )

    return (
        project(side.sector_directions[0]),
        project(side.sector_directions[1]),
    )


def _exact_scalar_sign(value: ExactScalar, /) -> int:
    if isinstance(value, RadicalElement):
        return value.sign()
    return 1 if value > 0 else -1 if value < 0 else 0


def _inside_sector(
    base: tuple[Fraction, Fraction],
    radical: tuple[Fraction, Fraction],
    q: Fraction,
    start: tuple[ExactScalar, ExactScalar],
    end: tuple[ExactScalar, ExactScalar],
    direction: int,
    major: bool,
) -> bool:
    root = RadicalElement.sqrt(q)
    first_sign = _exact_scalar_sign(
        direction
        * (
            start[0] * (base[1] + radical[1] * root)
            - start[1] * (base[0] + radical[0] * root)
        )
    )
    last_sign = _exact_scalar_sign(
        direction
        * (
            (base[0] + radical[0] * root) * end[1]
            - (base[1] + radical[1] * root) * end[0]
        )
    )
    if first_sign == 0 or last_sign == 0:
        return False
    return first_sign > 0 or last_sign > 0 if major else first_sign > 0 and last_sign > 0




def _contact_angle(
    junction: _Junction,
    side: _Side,
    point: np.ndarray,
    base: tuple[Fraction, Fraction],
    radical: tuple[Fraction, Fraction],
    q: Fraction,
    rays: tuple[tuple[ExactScalar, ExactScalar], tuple[ExactScalar, ExactScalar]],
) -> float | None:
    target = _angle(side, point, junction)
    if (
        not min(side.angle, side.retained_angle)
        < target
        < max(side.angle, side.retained_angle)
    ):
        return None
    handed = float(
        np.cross(np.asarray(side.patch.first_axis), np.asarray(side.patch.second_axis))
        @ junction.axis
    )
    direction = (1 if side.retained_angle > side.angle else -1) * (
        1 if handed > 0.0 else -1
    )
    if not _inside_sector(
        base,
        radical,
        q,
        rays[0],
        rays[1],
        direction,
        abs(side.retained_angle - side.angle) > pi,
    ):
        return None
    return target


def _fillet(junction: _Junction, radius: float, /) -> _Blend:
    c0, c1 = junction.centers_exact
    corner = (Fraction(0), Fraction(0))
    blend_radius_exact = Fraction.from_float(float(radius))
    rr = (blend_radius_exact, blend_radius_exact)
    delta = (c1[0] - c0[0], c1[1] - c0[1])
    d2 = delta[0] ** 2 + delta[1] ** 2
    if d2 == 0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: coaxial cylinders have no straight intersection junction"
        )
    branch_value = delta[0] * (corner[1] - c0[1]) - delta[1] * (corner[0] - c0[0])
    if branch_value == 0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: tangent or singular source circle contact"
        )
    branch = 1 if branch_value > 0 else -1
    candidates: list[_Blend] = []
    source_radii = (
        _fraction_exact(junction.sides[0].patch.radius_exact),
        _fraction_exact(junction.sides[1].patch.radius_exact),
    )
    if not (
        d2 < (source_radii[0] + source_radii[1]) ** 2
        and d2 > (source_radii[0] - source_radii[1]) ** 2
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: source circles do not intersect transversely"
        )
    rays = (
        _sector_rays(junction.sides[0], junction),
        _sector_rays(junction.sides[1], junction),
    )
    for orientation in (1, -1):
        rho = (
            source_radii[0] - orientation * junction.sides[0].radial_orientation * rr[0],
            source_radii[1] - orientation * junction.sides[1].radial_orientation * rr[0],
        )
        if min(rho) <= 0:
            continue
        lam = (d2 + rho[0] ** 2 - rho[1] ** 2) / (2 * d2)
        q = rho[0] ** 2 / d2 - lam**2
        if q <= 0:
            continue
        base = (c0[0] + lam * delta[0], c0[1] + lam * delta[1])
        radical = (-branch * delta[1], branch * delta[0])
        contact_unit_bases = (
            (
                (base[0] - c0[0]) / rho[0],
                (base[1] - c0[1]) / rho[0],
            ),
            (
                (base[0] - c1[0]) / rho[1],
                (base[1] - c1[1]) / rho[1],
            ),
        )
        contact_unit_radicals = (
            (radical[0] / rho[0], radical[1] / rho[0]),
            (radical[0] / rho[1], radical[1] / rho[1]),
        )
        square_root = RadicalElement.sqrt(q)
        center_exact = (
            Fraction(base[0]) + square_root * Fraction(radical[0]),
            Fraction(base[1]) + square_root * Fraction(radical[1]),
        )
        center = np.asarray(tuple(float(value) for value in center_exact))
        contact_units_exact = (
            (
                Fraction(contact_unit_bases[0][0])
                + square_root * Fraction(contact_unit_radicals[0][0]),
                Fraction(contact_unit_bases[0][1])
                + square_root * Fraction(contact_unit_radicals[0][1]),
            ),
            (
                Fraction(contact_unit_bases[1][0])
                + square_root * Fraction(contact_unit_radicals[1][0]),
                Fraction(contact_unit_bases[1][1])
                + square_root * Fraction(contact_unit_radicals[1][1]),
            ),
        )
        points = np.asarray(
            (
                (
                    float(c0[0] + source_radii[0] * contact_units_exact[0][0]),
                    float(c0[1] + source_radii[0] * contact_units_exact[0][1]),
                ),
                (
                    float(c1[0] + source_radii[1] * contact_units_exact[1][0]),
                    float(c1[1] + source_radii[1] * contact_units_exact[1][1]),
                ),
            ),
            dtype=np.float64,
        )
        angles: list[float | None] = []
        for i, side in enumerate(junction.sides):
            scaled_base = (
                (base[0] - (c0, c1)[i][0]) * source_radii[i] / rho[i],
                (base[1] - (c0, c1)[i][1]) * source_radii[i] / rho[i],
            )
            scaled_radical = (
                radical[0] * source_radii[i] / rho[i],
                radical[1] * source_radii[i] / rho[i],
            )
            angles.append(
                _contact_angle(
                    junction, side, points[i], scaled_base, scaled_radical, q, rays[i]
                )
            )
        angle0, angle1 = angles
        if angle0 is not None and angle1 is not None:
            vectors = (points - center) / radius
            sweep = atan2(
                float(vectors[0, 0] * vectors[1, 1] - vectors[0, 1] * vectors[1, 0]),
                float(vectors[0] @ vectors[1]),
            )
            if sweep == 0.0 or abs(sweep) >= pi:
                continue
            candidates.append(
                _Blend(
                    points,
                    (angle0, angle1),
                    center,
                    radius,
                    sweep,
                    orientation,
                    contact_unit_bases,
                    contact_unit_radicals,
                    q,
                    base,
                    radical,
                )
            )
    if len(candidates) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: offset branch has no unique nonvanishing retained-side certificate"
        )
    return candidates[0]


def _chamfer(junction: _Junction, setbacks: tuple[float, float]) -> _Blend:
    angles: list[float] = []
    phase_shifts: list[Fraction] = []
    points: list[np.ndarray] = []
    for side, setback in zip(junction.sides, setbacks, strict=True):
        exact_distance = Fraction(setback) / Fraction(float(side.patch.radius))
        if (
            not 0
            < exact_distance
            < abs(Fraction(side.retained_angle) - Fraction(side.angle))
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: setback vanishes or consumes a source face"
            )
        signed_distance = (
            exact_distance
            if side.retained_angle > side.angle
            else -exact_distance
        )
        phase_shift = side.source_phase_shift + signed_distance
        angle = side.angle + float(signed_distance)
        world = (
            np.asarray(
                side.patch.evaluate(
                    jnp.asarray((angle, side.heights[0]), dtype=jnp.float64)
                )
            )
            - junction.origin
        )
        points.append(np.asarray((world @ junction.first, world @ junction.second)))
        angles.append(angle)
        phase_shifts.append(phase_shift)
    if np.array_equal(points[0], points[1]):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: boundary collapses"
        )
    return _Blend(
        np.stack(points),
        (angles[0], angles[1]),
        np.zeros(2, dtype=np.float64),
        0.0,
        0.0,
        1,
        contact_roots=(junction.sides[0].source_root, junction.sides[1].source_root),
        phase_shifts_exact=(phase_shifts[0], phase_shifts[1]),
    )


def _cap_pcurve(
    patch: PlanePatch, curve: CircleCurve | LineCurve
) -> CircleCurve | LineCurve:
    origin_exact = rational_vector(np.asarray(patch.origin))
    axes_exact = (
        rational_vector(np.asarray(patch.first_axis)),
        rational_vector(np.asarray(patch.second_axis)),
    )

    def project(vector: _ExactVector, /) -> tuple[ExactScalar, ExactScalar]:
        return (
            sum(
                (a * b for a, b in zip(vector, axes_exact[0], strict=True)),
                Fraction(),
            ),
            sum(
                (a * b for a, b in zip(vector, axes_exact[1], strict=True)),
                Fraction(),
            ),
        )

    if isinstance(curve, CircleCurve):
        center = project(
            tuple(a - b for a, b in zip(curve.center_exact, origin_exact, strict=True))
        )
        first = project(curve.first_axis_exact)
        second = project(curve.second_axis_exact)
        return CircleCurve(
            np.asarray(tuple(float(value) for value in center)),
            np.asarray(tuple(float(value) for value in first)),
            np.asarray(tuple(float(value) for value in second)),
            curve.radius,
            center_exact=center,
            first_axis_exact=first,
            second_axis_exact=second,
            radius_exact=curve.radius_exact,
        )
    offset = tuple(a - b for a, b in zip(curve.origin_exact, origin_exact, strict=True))
    uv_origin = project(offset)
    uv_direction = project(curve.direction_exact)
    return LineCurve(
        np.asarray(tuple(float(value) for value in uv_origin)),
        np.asarray(tuple(float(value) for value in uv_direction)),
        origin_exact=uv_origin,
        direction_exact=uv_direction,
    )


class _StagedEdges(StrictModule):
    builder: _Builder
    new_vertices: list[tuple[int, int]]
    cap_changes: dict[int, tuple[int, int, float]]
    edge_map: dict[int, int]
    vertical_edges: list[int]
    source_contact_pcurves: tuple[AbstractCurve, AbstractCurve]
    blend_contact_pcurves: tuple[AbstractCurve, AbstractCurve]
    source_endpoint_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    source_coedge_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    blend_endpoint_roots: tuple[
        tuple[RootEndpoint | None, RootEndpoint | None],
        tuple[RootEndpoint | None, RootEndpoint | None],
    ]
    contact_points_exact: _ContactPoints | None
    source_cap_endpoint_roots: dict[tuple[int, ...], RootEndpoint]
    chamfer_patch: RuledSurface | None
    chamfer_cap_curves: tuple[AbstractCurve, AbstractCurve] | None = None
    chamfer_cap_face_pcurves: tuple[IntersectionPCurve, IntersectionPCurve] | None = None
    chamfer_cap_blend_pcurves: tuple[IntersectionPCurve, IntersectionPCurve] | None = None
    chamfer_cap_face_roots: (
        tuple[
            tuple[RootEndpoint, RootEndpoint],
            tuple[RootEndpoint, RootEndpoint],
        ]
        | None
    ) = None
    chamfer_cap_blend_roots: (
        tuple[
            tuple[RootEndpoint, RootEndpoint],
            tuple[RootEndpoint, RootEndpoint],
        ]
        | None
    ) = None


class _StagedBlend(StrictModule):
    patch: CylinderPatch | PlanePatch | RuledSurface
    cap_edges: list[int]
    cap_curves: list[AbstractCurve]
    low: float
    high: float
    endpoints: tuple[int, int]
    cap_roots: (
        tuple[
            tuple[RootEndpoint, RootEndpoint],
            tuple[RootEndpoint, RootEndpoint],
        ]
        | None
    )
    face_bounds: tuple[float, float]
    cap_face_pcurves: tuple[AbstractCurve, AbstractCurve] | None = None
    blend_cap_pcurves: tuple[AbstractCurve, AbstractCurve] | None = None
    blend_cap_roots: (
        tuple[
            tuple[RootEndpoint, RootEndpoint],
            tuple[RootEndpoint, RootEndpoint],
        ]
        | None
    ) = None


def _unique_trim_root(
    first: CurveRange | AbstractCurve,
    second: CurveRange | AbstractCurve,
    /,
    *,
    first_parameter: Fraction | float | None = None,
    second_parameter: Fraction | float | None = None,
    endpoint_witness: ConicRootLineEndpointWitness | None = None,
) -> TrimIntersectionRoot:
    first_curve = first.curve if isinstance(first, CurveRange) else first
    second_curve = second.curve if isinstance(second, CurveRange) else second
    result = intersect_trim_curves(first, second)
    if not result.complete or result.coincident:
        if (
            endpoint_witness is not None
            and not result.coincident
            and not result.points
            and len(result.unresolved) == 1
            and result.unresolved[0].reason == "resolution"
            and endpoint_witness.binds_curves(first_curve, second_curve)
            and (
                second_parameter is None
                or Fraction.from_float(float(second_parameter))
                == endpoint_witness.parameter_exact
            )
        ):
            unresolved = result.unresolved[0]
            unresolved_lower = np.asarray(unresolved.lower, dtype=np.float64)
            unresolved_upper = np.asarray(unresolved.upper, dtype=np.float64)
            endpoint_box = endpoint_witness.parameter_box()
            if (
                unresolved_lower.shape == (2,)
                and unresolved_upper.shape == (2,)
                and np.all(unresolved_lower <= endpoint_box[0])
                and np.all(unresolved_upper >= endpoint_box[1])
                and (
                    first_parameter is None
                    or endpoint_box[0, 0]
                    <= float(first_parameter)
                    <= endpoint_box[1, 0]
                )
            ):
                lower = np.minimum(unresolved_lower, endpoint_box[0])
                upper = np.maximum(unresolved_upper, endpoint_box[1])
                margin = 32.0 * np.finfo(np.float64).eps * np.maximum(
                    1.0, np.maximum(np.abs(lower), np.abs(upper))
                )
                lower = np.nextafter(lower - margin, -np.inf)
                upper = np.nextafter(upper + margin, np.inf)
                try:
                    return TrimIntersectionRoot(
                        first_curve,
                        second_curve,
                        parameter_lower=lower,
                        parameter_upper=upper,
                        endpoint_witness=endpoint_witness,
                    )
                except (ValueError, np.linalg.LinAlgError):
                    pass
        unresolved = tuple(
            (
                region.reason,
                tuple(np.asarray(region.lower, dtype=np.float64)),
                tuple(np.asarray(region.upper, dtype=np.float64)),
            )
            for region in result.unresolved
        )
        ranges = (
            (first.first, first.last)
            if isinstance(first, CurveRange)
            else first.parameter_domain,
            (second.first, second.last)
            if isinstance(second, CurveRange)
            else second.parameter_domain,
        )
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: trim endpoint roots are "
            f"incomplete; curves=({type(first_curve).__name__}, "
            f"{type(second_curve).__name__}), ranges={ranges!r}, "
            f"unresolved={unresolved!r}, coincident={len(result.coincident)}"
        )
    selected = []
    for point in result.points:
        lower = np.asarray(point.parameter_lower, dtype=np.float64)
        upper = np.asarray(point.parameter_upper, dtype=np.float64)
        if first_parameter is not None and not (
            lower[0] <= float(first_parameter) <= upper[0]
        ):
            continue
        if second_parameter is not None and not (
            lower[1] <= float(second_parameter) <= upper[1]
        ):
            continue
        selected.append(
            TrimIntersectionRoot(
                first_curve,
                second_curve,
                parameter_lower=point.parameter_lower,
                parameter_upper=point.parameter_upper,
                endpoint_witness=endpoint_witness,
            )
        )
    if len(selected) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: a cap trim root is not unique"
        )
    return selected[0]

def _constructive_cap_root(
    curve: CircleCurve,
    region: SurfaceRegion,
    witness: ConicRootLineEndpointWitness,
    /,
) -> CurveSurfaceIntersectionRoot:
    point_box = witness.curve_surface_parameter_box(curve, region)
    margin = 64.0 * np.finfo(np.float64).eps * np.maximum(
        1.0, np.maximum(np.abs(point_box[0]), np.abs(point_box[1]))
    )
    domain = np.asarray(region.parameter_box, dtype=np.float64)
    lower = np.maximum(
        np.nextafter(point_box[0] - margin, -np.inf),
        np.asarray((float(witness.trim_interval[0]), *domain[0]), dtype=np.float64),
    )
    upper = np.minimum(
        np.nextafter(point_box[1] + margin, np.inf),
        np.asarray((float(witness.trim_interval[1]), *domain[1]), dtype=np.float64),
    )
    return CurveSurfaceIntersectionRoot(
        curve, region, parameter_lower=lower, parameter_upper=upper,
        endpoint_witness=witness,
    )


def _surface_region(model: BRepModel, face: int, /) -> SurfaceRegion:
    return SurfaceRegion(
        model.patches[face], np.asarray(model.parameter_bounds[face], dtype=np.float64)
    )


def _face_edge_pcurve(
    geometry: BRepGeometry, face: int, edge: int, /
) -> AbstractCurve:
    coedges = tuple(
        coedge
        for loop in geometry.face_loops[face]
        for coedge in loop
        if geometry.coedge_edges[coedge] == edge
    )
    if len(coedges) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: source edge has no unique face p-curve"
        )
    pcurve = geometry.pcurves[coedges[0]]
    if not isinstance(pcurve, AbstractCurve) or pcurve.ambient_dimension != 2:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: source face trim is not a planar p-curve"
        )
    return pcurve


def _stage_chamfer_edges(
    junction: _Junction, blend: _Blend, selected: int, model: BRepModel, /
) -> _StagedEdges:
    geometry, builder = junction.geometry, _Builder()
    if blend.contact_roots is None or blend.phase_shifts_exact is None:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: exact source phase roots are unavailable"
        )
    removed_vertices = set(junction.vertices)
    vertex_map = {
        index: builder.vertex(
            np.asarray(geometry.vertex_points[index]),
            exact_point=geometry.vertex_exact_points[index],
            root=geometry.vertex_roots[index],
        )
        for index in range(len(geometry.vertex_points))
        if index not in removed_vertices
    }
    source_pcurves: list[ConicRootLinePCurve] = []
    source_curves: list[SurfaceMappedCurve] = []
    source_regions: list[SurfaceRegion] = []
    for side_index, side in enumerate(junction.sides):
        same = _axis_scale(side) > 0
        height_origin = side.heights_exact[0] if same else side.heights_exact[1]
        height_direction = Fraction(1) if same else Fraction(-1)
        pcurve = ConicRootLinePCurve(
            blend.contact_roots[side_index],
            height_origin,
            height_direction,
            parameter_range=(0.0, 1.0),
            phase_shift_exact=blend.phase_shifts_exact[side_index],
        )
        source_pcurves.append(pcurve)
        source_curves.append(SurfaceMappedCurve(side.patch, pcurve))
        source_regions.append(_surface_region(model, side.face))
    blend_patch = RuledSurface(source_curves[0], source_curves[1])
    blend_region = SurfaceRegion(
        blend_patch, np.asarray(((0.0, 0.0), (1.0, 1.0)), dtype=np.float64)
    )
    blend_pcurves = tuple(
        LineCurve((0.0, float(side_index)), (1.0, 0.0))
        for side_index in range(2)
    )
    new_vertices: list[list[int]] = [[-1, -1], [-1, -1]]
    source_endpoint_roots: list[list[RootEndpoint | None]] = [
        [None, None],
        [None, None],
    ]
    source_coedge_roots: list[list[RootEndpoint | None]] = [
        [None, None],
        [None, None],
    ]
    blend_endpoint_roots: list[list[RootEndpoint | None]] = [
        [None, None],
        [None, None],
    ]
    source_cap_endpoint_roots: dict[tuple[int, ...], RootEndpoint] = {}
    cap_changes: dict[int, tuple[int, int, float]] = {}
    cap_edge_curves: list[AbstractCurve] = []
    cap_face_pcurves: list[IntersectionPCurve] = []
    cap_blend_pcurves: list[IntersectionPCurve] = []
    cap_face_roots: list[tuple[RootEndpoint, RootEndpoint]] = []
    cap_blend_roots: list[tuple[RootEndpoint, RootEndpoint]] = []
    for end, cap_face in enumerate(junction.cap_faces):
        cap_patch = model.patches[cap_face]
        if not isinstance(cap_patch, PlanePatch):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: source cap is not analytic planar"
            )
        cap_region = _surface_region(model, cap_face)
        section = _ruled_plane_cap_section(
            blend_region, cap_region, ParametricIntersectionPolicy()
        )
        if section is None or not section.complete or len(section.curves) != 1:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: ruled source cap section is incomplete"
            )
        branch = section.curves[0]
        if not branch.fully_certified or branch.closed:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: cap section is not one certified open branch"
            )
        cap_face_pcurve = IntersectionPCurve(branch, "second")
        cap_blend_pcurve = IntersectionPCurve(branch, "first")
        cap_edge_curve = branch
        cap_edge_curves.append(cap_edge_curve)
        cap_face_pcurves.append(cap_face_pcurve)
        cap_blend_pcurves.append(cap_blend_pcurve)
        face_roots: list[RootEndpoint | None] = [None, None]
        blend_roots: list[RootEndpoint | None] = [None, None]
        for side_index, side in enumerate(junction.sides):
            same = _axis_scale(side) > 0
            cap_edge = side.cap_edges[end if same else 1 - end]
            curve_index = geometry.edge_curves[cap_edge]
            cap_circle = (
                None if curve_index < 0 else geometry.curves[curve_index]
            )
            if not isinstance(cap_circle, CircleCurve):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder chamfer: cap endpoint has no exact circular carrier"
                )
            edge_first, edge_last = (
                float(value) for value in np.asarray(geometry.edge_ranges[cap_edge])
            )
            side_cap_pcurve = _face_edge_pcurve(geometry, side.face, cap_edge)
            cap_circle_pcurve = _face_edge_pcurve(geometry, cap_face, cap_edge)
            branch_parameter = (
                0.0 if side_index == 0 else float(branch.num_charts)
            )
            endpoint_witness = ConicRootLineEndpointWitness(
                source_curves[side_index],
                source_pcurves[side_index],
                source_regions[side_index],
                Fraction(end),
                (Fraction(0), Fraction(1)),
                side_cap_pcurve,
                (Fraction.from_float(edge_first), Fraction.from_float(edge_last)),
            )
            side_root = _unique_trim_root(
                CurveRange(side_cap_pcurve, edge_first, edge_last),
                CurveRange(source_pcurves[side_index], 0.0, 1.0),
                second_parameter=float(end),
                endpoint_witness=endpoint_witness,
            )
            cap_root = _constructive_cap_root(cap_circle, blend_region, endpoint_witness)
            side_region = source_regions[side_index]
            source_curve = source_curves[side_index]
            source_pcurve = source_pcurves[side_index]
            source_blend_pcurve = blend_pcurves[side_index]
            try:
                lifts = (
                    BRepCurveSurfaceLift(
                        cap_circle,
                        side.patch,
                        side_cap_pcurve,
                        edge_first,
                        edge_last,
                        source_region=side_region,
                    ),
                    BRepCurveSurfaceLift(
                        cap_circle,
                        cap_patch,
                        cap_circle_pcurve,
                        edge_first,
                        edge_last,
                        source_region=cap_region,
                    ),
                    BRepCurveSurfaceLift(
                        source_curve,
                        side.patch,
                        source_pcurve,
                        0.0,
                        1.0,
                        source_region=side_region,
                    ),
                    BRepCurveSurfaceLift(
                        source_curve,
                        blend_patch,
                        source_blend_pcurve,
                        0.0,
                        1.0,
                        source_region=blend_region,
                    ),
                    BRepCurveSurfaceLift(
                        cap_edge_curve,
                        cap_patch,
                        cap_face_pcurve,
                        0.0,
                        float(branch.num_charts),
                        source_region=cap_region,
                    ),
                    BRepCurveSurfaceLift(
                        cap_edge_curve,
                        blend_patch,
                        cap_blend_pcurve,
                        0.0,
                        float(branch.num_charts),
                        source_region=blend_region,
                    ),
                )
                vertex_root = BRepVertexRoot(
                    cap_root,
                    aliases=(BRepRootSupport(side.patch, side_root),),
                    spatial_root=cap_root,
                    source_edge_lifts=lifts,
                )
            except (TypeError, ValueError, np.linalg.LinAlgError) as error:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder chamfer: contact roots lack exact source lifts"
                ) from error
            point = np.asarray(
                cap_circle.evaluate(jnp.asarray(cap_root.parameters[0], dtype=jnp.float64)),
                dtype=np.float64,
            )
            new_vertices[side_index][end] = builder.vertex(
                point, root=vertex_root
            )
            cap_endpoint = TrimRootEndpoint(
                cap_root,
                "first",
                source_pcurve=cap_circle_pcurve,
                source_surface=cap_region,
            )
            side_endpoint = TrimRootEndpoint(
                side_root,
                "first",
                source_pcurve=side_cap_pcurve,
                source_surface=side_region,
            )
            rail_endpoint = TrimRootEndpoint(
                side_root,
                "second",
                source_pcurve=source_pcurve,
                source_surface=side_region,
            )
            blend_rail_endpoint = TrimRootEndpoint(
                side_root,
                "second",
                source_pcurve=source_blend_pcurve,
                source_surface=blend_region,
            )
            branch_endpoint = BranchRootEndpoint(
                cap_root,
                branch,
                0 if side_index == 0 else branch.num_charts - 1,
                source_pcurve=cap_circle_pcurve,
                source_side="second",
                source_first=edge_first,
                source_last=edge_last,
            )
            if not branch_endpoint.certifies_point_root(
                IntersectionCurvePointRoot(branch, branch_parameter)
            ):
                raise BRepBlendUnsupported(
                    "The constructive cap endpoint is not the certified branch node."
                )
            cap_branch_endpoint = blend_branch_endpoint = branch_endpoint
            source_endpoint_roots[side_index][end] = rail_endpoint
            source_coedge_roots[side_index][end] = rail_endpoint
            blend_endpoint_roots[side_index][end] = blend_rail_endpoint
            source_endpoint_index = geometry.edge_vertices[cap_edge].index(
                junction.vertices[end]
            )
            source_cap_endpoint_roots[(cap_edge, source_endpoint_index)] = (
                cap_endpoint
            )
            source_cap_endpoint_roots[
                (cap_edge, source_endpoint_index, cap_face)
            ] = cap_endpoint
            source_cap_endpoint_roots[
                (cap_edge, source_endpoint_index, side.face)
            ] = side_endpoint
            cap_parameter = float(cap_root.parameters[0])
            cap_changes[cap_edge] = (side_index, end, cap_parameter)
            face_roots[side_index] = cap_branch_endpoint
            blend_roots[side_index] = blend_branch_endpoint
        cap_first_root, cap_second_root = face_roots
        blend_first_root, blend_second_root = blend_roots
        if (
            cap_first_root is None
            or cap_second_root is None
            or blend_first_root is None
            or blend_second_root is None
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: cap branch endpoints are incomplete"
            )
        cap_face_roots.append((cap_first_root, cap_second_root))
        cap_blend_roots.append((blend_first_root, blend_second_root))
    edge_map: dict[int, int] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if edge == selected:
            continue
        ends = geometry.edge_vertices[edge]
        first, last = map(float, np.asarray(geometry.edge_ranges[edge]))
        change = cap_changes.get(edge)
        vertices: list[int] = []
        endpoint_roots = list(geometry.edge_endpoint_roots[edge])
        for endpoint_index, vertex in enumerate(ends):
            if change is not None and vertex in removed_vertices:
                side_index, end, contact_parameter = change
                vertices.append(new_vertices[side_index][end])
                root = source_cap_endpoint_roots.get((edge, endpoint_index))
                if root is None:
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder chamfer: exact cap endpoint root is unavailable"
                    )
                endpoint_roots[endpoint_index] = root
                if endpoint_index == 0:
                    first = contact_parameter
                else:
                    last = contact_parameter
            else:
                vertices.append(vertex_map[vertex])
        endpoint_root_pair = (endpoint_roots[0], endpoint_roots[1])
        if curve_index < 0:
            edge_map[edge] = builder.degenerate(
                vertices[0], first, last, endpoint_roots=endpoint_root_pair
            )
        else:
            edge_map[edge] = builder.edge(
                geometry.curves[curve_index],
                first,
                last,
                vertices[0],
                vertices[1],
                endpoint_roots=endpoint_root_pair,
            )
    vertical_edges = [
        builder.edge(
            source_curves[side_index],
            0.0,
            1.0,
            *new_vertices[side_index],
            endpoint_roots=(
                source_endpoint_roots[side_index][0],
                source_endpoint_roots[side_index][1],
            ),
        )
        for side_index in range(2)
    ]
    return _StagedEdges(
        builder,
        [
            (new_vertices[0][0], new_vertices[0][1]),
            (new_vertices[1][0], new_vertices[1][1]),
        ],
        cap_changes,
        edge_map,
        vertical_edges,
        (source_pcurves[0], source_pcurves[1]),
        (blend_pcurves[0], blend_pcurves[1]),
        (
            (source_endpoint_roots[0][0], source_endpoint_roots[0][1]),
            (source_endpoint_roots[1][0], source_endpoint_roots[1][1]),
        ),
        (
            (source_coedge_roots[0][0], source_coedge_roots[0][1]),
            (source_coedge_roots[1][0], source_coedge_roots[1][1]),
        ),
        (
            (blend_endpoint_roots[0][0], blend_endpoint_roots[0][1]),
            (blend_endpoint_roots[1][0], blend_endpoint_roots[1][1]),
        ),
        None,
        source_cap_endpoint_roots,
        blend_patch,
        (cap_edge_curves[0], cap_edge_curves[1]),
        (cap_face_pcurves[0], cap_face_pcurves[1]),
        (cap_blend_pcurves[0], cap_blend_pcurves[1]),
        (cap_face_roots[0], cap_face_roots[1]),
        (cap_blend_roots[0], cap_blend_roots[1]),
    )

def _stage_edges(
    junction: _Junction, blend: _Blend, selected: int, model: BRepModel, /
) -> _StagedEdges:
    if blend.radius == 0.0:
        return _stage_chamfer_edges(junction, blend, selected, model)
    geometry = junction.geometry
    builder = _Builder()
    removed_vertices = set(junction.vertices)
    vertex_map = {
        index: builder.vertex(
            np.asarray(geometry.vertex_points[index]),
            exact_point=geometry.vertex_exact_points[index],
            root=geometry.vertex_roots[index],
        )
        for index in range(len(geometry.vertex_points))
        if index not in removed_vertices
    }
    new_vertices: list[tuple[int, int]] = []
    source_pcurves: list[AbstractCurve] = []
    blend_pcurves: list[AbstractCurve] = []
    source_endpoint_roots: list[
        tuple[ConicParameterEndpoint | None, ConicParameterEndpoint | None]
    ] = []
    source_coedge_roots: list[
        tuple[ConicParameterEndpoint | None, ConicParameterEndpoint | None]
    ] = []
    blend_endpoint_roots: list[
        tuple[ConicParameterEndpoint | None, ConicParameterEndpoint | None]
    ] = []
    contact_points_exact: _ContactPoints | None = None
    source_cap_endpoint_roots: dict[tuple[int, ...], RootEndpoint] = {}
    if blend.radius > 0.0:
        center_exact = _blend_center_exact(junction, blend)
        center_numeric = np.asarray(tuple(float(value) for value in center_exact))
        exact_points: list[_ContactPointPair] = []
        for index, side in enumerate(junction.sides):
            source_root = _source_contact_root(junction, blend, index)
            target_root = _blend_contact_root(
                junction, blend, index, center_exact, center_numeric
            )
            same = _axis_scale(side) > 0
            height_origin = side.heights_exact[0] if same else side.heights_exact[1]
            height_direction = Fraction(1) if same else Fraction(-1)
            source_axis = _rational_exact_vector3(side.patch.axis_exact)
            source_offsets = (
                (
                    height_origin * source_axis[0],
                    height_origin * source_axis[1],
                    height_origin * source_axis[2],
                ),
                (
                    (height_origin + height_direction) * source_axis[0],
                    (height_origin + height_direction) * source_axis[1],
                    (height_origin + height_direction) * source_axis[2],
                ),
            )
            target_offsets = (
                (Fraction(0), Fraction(0), Fraction(0)),
                (
                    junction.height_exact * junction.axis_exact[0],
                    junction.height_exact * junction.axis_exact[1],
                    junction.height_exact * junction.axis_exact[2],
                ),
            )
            source_points = (
                _offset_exact_point(source_root.exact_point(), source_offsets[0]),
                _offset_exact_point(source_root.exact_point(), source_offsets[1]),
            )
            target_points = (
                _offset_exact_point(target_root.exact_point(), target_offsets[0]),
                _offset_exact_point(target_root.exact_point(), target_offsets[1]),
            )
            if source_points != target_points:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: exact quadratic source and blend contact roots disagree"
                )
            same = _axis_scale(side) > 0
            for cap_end in range(2):
                cap_edge = side.cap_edges[cap_end if same else 1 - cap_end]
                source_vertex = junction.vertices[cap_end]
                endpoint = geometry.edge_vertices[cap_edge].index(source_vertex)
                cap_curve_index = geometry.edge_curves[cap_edge]
                cap_curve = (
                    None if cap_curve_index < 0 else geometry.curves[cap_curve_index]
                )
                if not isinstance(cap_curve, CircleCurve):
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder blend: source cap endpoint lacks its exact circular carrier"
                    )
                cap_root = _conic_root(
                    cap_curve,
                    (source_root.cosine_exact, source_root.sine_exact),
                    source_root.parameter,
                )
                point_exact = source_points[cap_end]
                if cap_root.exact_point() != point_exact:
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder blend: source cap conic root misses its contact endpoint"
                    )
                source_cap_endpoint_roots[(cap_edge, endpoint)] = ConicParameterEndpoint(
                    cap_root, cap_curve, None, point_exact
                )
            source_pcurve = ConicRootLinePCurve(
                source_root, height_origin, height_direction, parameter_range=(0.0, 1.0)
            )
            blend_pcurve = ConicRootLinePCurve(
                target_root,
                Fraction(0),
                junction.height_exact,
                parameter_range=(0.0, 1.0),
            )
            source_curve = SurfaceMappedCurve(side.patch, source_pcurve)
            source_roots = (
                ConicParameterEndpoint(
                    source_root, source_curve, Fraction(0), source_points[0]
                ),
                ConicParameterEndpoint(
                    source_root, source_curve, Fraction(1), source_points[1]
                ),
            )
            source_coedge_roots.append(
                (
                    ConicParameterEndpoint(
                        source_root, source_pcurve, Fraction(0), source_points[0]
                    ),
                    ConicParameterEndpoint(
                        source_root, source_pcurve, Fraction(1), source_points[1]
                    ),
                )
            )
            blend_roots = (
                ConicParameterEndpoint(
                    target_root, blend_pcurve, Fraction(0), target_points[0]
                ),
                ConicParameterEndpoint(
                    target_root, blend_pcurve, Fraction(1), target_points[1]
                ),
            )
            bottom = np.asarray(tuple(float(value) for value in source_points[0]))
            top = np.asarray(tuple(float(value) for value in source_points[1]))
            new_vertices.append(
                (
                    builder.vertex(
                        bottom,
                        root=BRepVertexRoot(
                            BRepConicVertex(source_root, offset_exact=source_offsets[0])
                        ),
                    ),
                    builder.vertex(
                        top,
                        root=BRepVertexRoot(
                            BRepConicVertex(source_root, offset_exact=source_offsets[1])
                        ),
                    ),
                )
            )
            source_pcurves.append(source_pcurve)
            blend_pcurves.append(blend_pcurve)
            source_endpoint_roots.append(source_roots)
            blend_endpoint_roots.append(blend_roots)
            exact_points.append(source_points)
        contact_points_exact = (exact_points[0], exact_points[1])
    cap_changes: dict[int, tuple[int, int, float]] = {}
    for index, side in enumerate(junction.sides):
        same = _axis_scale(side) > 0
        source_pcurve = source_pcurves[index]
        if not isinstance(source_pcurve, ConicRootLinePCurve):
            raise BRepBlendUnsupported(
                "parallel-cylinder fillet requires exact source angular roots"
            )
        contact_angle = float(source_pcurve.angle)
        for end in range(2):
            cap_changes[side.cap_edges[end if same else 1 - end]] = (
                index,
                end,
                contact_angle,
            )
    edge_map: dict[int, int] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if edge == selected:
            continue
        ends = geometry.edge_vertices[edge]
        first, last = map(float, np.asarray(geometry.edge_ranges[edge]))
        change = cap_changes.get(edge)
        vertices: list[int] = []
        endpoint_roots = list(geometry.edge_endpoint_roots[edge])
        for endpoint, vertex in enumerate(ends):
            if change is not None and vertex in removed_vertices:
                vertices.append(new_vertices[change[0]][change[1]])
                root = source_cap_endpoint_roots.get((edge, endpoint))
                if root is None:
                    raise BRepBlendUnsupported(
                        "parallel-cylinder fillet cannot rebind a moved cap endpoint root"
                    )
                endpoint_roots[endpoint] = root
                if endpoint == 0:
                    first = change[2]
                else:
                    last = change[2]
            else:
                vertices.append(vertex_map[vertex])
        endpoint_root_pair = (endpoint_roots[0], endpoint_roots[1])
        if curve_index < 0:
            edge_map[edge] = builder.degenerate(
                vertices[0], first, last, endpoint_roots=endpoint_root_pair
            )
        else:
            edge_map[edge] = builder.edge(
                geometry.curves[curve_index],
                first,
                last,
                vertices[0],
                vertices[1],
                endpoint_roots=endpoint_root_pair,
            )
    vertical_edges: list[int] = []
    for index, side in enumerate(junction.sides):
        source_curve = SurfaceMappedCurve(side.patch, source_pcurves[index])
        vertical_edges.append(
            builder.edge(
                source_curve,
                0.0,
                1.0,
                *new_vertices[index],
                endpoint_roots=source_endpoint_roots[index],
            )
        )
    return _StagedEdges(
        builder,
        new_vertices,
        cap_changes,
        edge_map,
        vertical_edges,
        (source_pcurves[0], source_pcurves[1]),
        (blend_pcurves[0], blend_pcurves[1]),
        (source_endpoint_roots[0], source_endpoint_roots[1]),
        (source_coedge_roots[0], source_coedge_roots[1]),
        (blend_endpoint_roots[0], blend_endpoint_roots[1]),
        contact_points_exact,
        source_cap_endpoint_roots,
        None,
    )


def _stage_caps(
    junction: _Junction, blend: _Blend, edges: _StagedEdges
) -> _StagedBlend:
    builder, new_vertices = edges.builder, edges.new_vertices
    cap_edges: list[int] = []
    cap_curves: list[AbstractCurve] = []
    cap_roots: list[tuple[RootEndpoint, RootEndpoint]] = []
    if blend.radius > 0.0:
        first_pcurve, second_pcurve = edges.blend_contact_pcurves
        if (
            edges.contact_points_exact is None
            or not isinstance(first_pcurve, ConicRootLinePCurve)
            or not isinstance(second_pcurve, ConicRootLinePCurve)
        ):
            raise BRepBlendUnsupported(
                "parallel-cylinder fillet caps require exact contact-root carriers"
            )
        target_roots = (first_pcurve.root, second_pcurve.root)
        beta0 = float(target_roots[0].parameter)
        beta1 = float(target_roots[1].parameter)
        sweep = beta1 - beta0
        if sweep == 0.0 or abs(sweep) >= pi or sweep * blend.sweep <= 0.0:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: exact contact roots do not retain one arc branch"
            )
        low, high = sorted((beta0, beta1))
        face_bounds = (
            min(low, *(root.parameter_lower for root in target_roots)),
            max(high, *(root.parameter_upper for root in target_roots)),
        )
        center_exact = _blend_center_exact(junction, blend)
        blend_origin = np.asarray(tuple(float(value) for value in center_exact))
        blend_patch: CylinderPatch | PlanePatch | RuledSurface = CylinderPatch(
            blend_origin,
            junction.first,
            junction.second,
            junction.axis,
            blend.radius,
            origin_exact=center_exact,
            first_axis_exact=junction.first_exact,
            second_axis_exact=junction.second_exact,
            axis_exact=junction.axis_exact,
            radius_exact=Fraction.from_float(float(blend.radius)),
        )
        endpoints = (0, 1) if sweep > 0.0 else (1, 0)
        for end in range(2):
            axial_offset = (
                end * junction.height_exact * junction.axis_exact[0],
                end * junction.height_exact * junction.axis_exact[1],
                end * junction.height_exact * junction.axis_exact[2],
            )
            cap_center_exact = (
                center_exact[0] + axial_offset[0],
                center_exact[1] + axial_offset[1],
                center_exact[2] + axial_offset[2],
            )
            cap_center = np.asarray(tuple(float(value) for value in cap_center_exact))
            curve = CircleCurve(
                cap_center,
                junction.first,
                junction.second,
                blend.radius,
                center_exact=cap_center_exact,
                first_axis_exact=junction.first_exact,
                second_axis_exact=junction.second_exact,
                radius_exact=Fraction.from_float(float(blend.radius)),
            )
            edge_sides = (endpoints[0], endpoints[1])
            endpoint_roots: list[ConicParameterEndpoint] = []
            for side_index in edge_sides:
                target_root = target_roots[side_index]
                cap_root = _conic_root(
                    curve,
                    (target_root.cosine_exact, target_root.sine_exact),
                    target_root.parameter,
                )
                point_exact = edges.contact_points_exact[side_index][end]
                if cap_root.exact_point() != point_exact:
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder blend: exact cap conic root misses its contact vertex"
                    )
                endpoint_roots.append(
                    ConicParameterEndpoint(cap_root, curve, None, point_exact)
                )
            cap_curves.append(curve)
            cap_roots.append((endpoint_roots[0], endpoint_roots[1]))
            cap_edges.append(
                builder.edge(
                    curve,
                    low,
                    high,
                    new_vertices[edge_sides[0]][end],
                    new_vertices[edge_sides[1]][end],
                    endpoint_roots=(endpoint_roots[0], endpoint_roots[1]),
                )
            )
        return _StagedBlend(
            blend_patch,
            cap_edges,
            cap_curves,
            low,
            high,
            endpoints,
            (cap_roots[0], cap_roots[1]),
            face_bounds,
        )
    if (
        edges.chamfer_patch is None
        or edges.chamfer_cap_curves is None
        or edges.chamfer_cap_face_pcurves is None
        or edges.chamfer_cap_blend_pcurves is None
        or edges.chamfer_cap_face_roots is None
        or edges.chamfer_cap_blend_roots is None
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder chamfer: ruled source cap roots are unavailable"
        )
    blend_patch = edges.chamfer_patch
    endpoints = (0, 1)
    low, high = 0.0, 1.0
    for end, curve in enumerate(edges.chamfer_cap_curves):
        cap_pcurve = edges.chamfer_cap_face_pcurves[end]
        cap_roots_ = edges.chamfer_cap_face_roots[end]
        branch_last = cap_pcurve.last
        cap_curves.append(curve)
        cap_roots.append(cap_roots_)
        cap_edges.append(
            builder.edge(
                curve,
                0.0,
                branch_last,
                new_vertices[0][end],
                new_vertices[1][end],
                endpoint_roots=cap_roots_,
            )
        )
    return _StagedBlend(
        blend_patch,
        cap_edges,
        cap_curves,
        low,
        high,
        endpoints,
        (cap_roots[0], cap_roots[1]),
        (low, high),
        edges.chamfer_cap_face_pcurves,
        edges.chamfer_cap_blend_pcurves,
        edges.chamfer_cap_blend_roots,
    )
def _vertical_use(
    junction: _Junction,
    blend: _Blend,
    side_index: int,
    side: _Side,
    sense: int,
    pcurve: AbstractCurve | IntersectionPCurve,
) -> tuple[int, LineCurve]:
    if isinstance(pcurve, ConicRootLinePCurve):
        height_direction = pcurve.height_direction_exact
    elif isinstance(pcurve, LineCurve):
        if pcurve.direction_exact[0] != 0:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: nonaffine vertical parameter"
            )
        height_direction = pcurve.direction_exact[1]
    else:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: nonaffine vertical parameter"
        )
    same = _axis_scale(side) > 0
    height = side.heights[0] if same else side.heights[1]
    derivative = Fraction(1) if same else Fraction(-1)
    sense *= 1 if height_direction * derivative > 0 else -1
    return sense, LineCurve((blend.angles[side_index], height), (0.0, float(derivative)))


def _stage_source_loop(
    junction: _Junction,
    blend: _Blend,
    model: BRepModel,
    selected: int,
    edges: _StagedEdges,
    boundary: _StagedBlend,
    face: int,
    loop: tuple[int, ...],
    /,
) -> list[int]:
    builder, geometry = edges.builder, junction.geometry
    staged: list[int] = []
    for position, coedge in enumerate(loop):
        edge = geometry.coedge_edges[coedge]
        pcurve = geometry.pcurves[coedge]
        sense = geometry.coedge_senses[coedge]
        if edge == selected:
            matches = [
                (i, side) for i, side in enumerate(junction.sides) if side.face == face
            ]
            if len(matches) != 1:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: unexpected junction face"
                )
            i, side = matches[0]
            sense, _ = _vertical_use(junction, blend, i, side, sense, pcurve)
            pcurve = edges.source_contact_pcurves[i]
            target = edges.vertical_edges[i]
        else:
            target = edges.edge_map[edge]
        if edge == selected:
            coedge_roots = list(edges.source_coedge_roots[i])
        elif edge in edges.cap_changes:
            coedge_roots = list(geometry.coedge_endpoint_roots[coedge])
            for endpoint, vertex in enumerate(geometry.edge_vertices[edge]):
                if vertex not in junction.vertices:
                    continue
                key = (
                    (edge, endpoint, face)
                    if blend.radius == 0.0
                    else (edge, endpoint)
                )
                root = edges.source_cap_endpoint_roots.get(key)
                if blend.radius == 0.0:
                    if root is None:
                        raise BRepBlendUnsupported(
                            "unsupported parallel-cylinder chamfer: exact source cap p-curve root is unavailable"
                        )
                    coedge_roots[endpoint] = root
                else:
                    if not isinstance(root, ConicParameterEndpoint):
                        raise BRepBlendUnsupported(
                            "unsupported parallel-cylinder fillet: exact source cap root is unavailable"
                        )
                    coedge_roots[endpoint] = ConicParameterEndpoint(
                        root.root, pcurve, None, root.point_exact
                    )
        else:
            coedge_roots = list(geometry.coedge_endpoint_roots[coedge])
        coedge_root_pair = (coedge_roots[0], coedge_roots[1])
        staged.append(
            builder.coedge(target, sense, pcurve, endpoint_roots=coedge_root_pair)
        )
        if face not in junction.cap_faces or edge not in edges.cap_changes:
            continue
        following = geometry.coedge_edges[loop[(position + 1) % len(loop)]]
        tail = geometry.edge_vertices[edge][1 if sense > 0 else 0]
        if tail not in junction.vertices or following not in edges.cap_changes:
            continue
        end = junction.cap_faces.index(face)
        start_side = edges.cap_changes[edge][0]
        cap_sense = 1 if start_side == boundary.endpoints[0] else -1
        patch = model.patches[face]
        if not isinstance(patch, PlanePatch):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: nonplanar cap"
            )
        if boundary.cap_roots is None:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: cap endpoint roots are unavailable"
            )
        if blend.radius > 0.0:
            cap_curve = boundary.cap_curves[end]
            if not isinstance(cap_curve, (CircleCurve, LineCurve)):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder fillet: cap curve is not circular or linear"
                )
            cap_pcurve = _cap_pcurve(patch, cap_curve)
            cap_pair = boundary.cap_roots[end]
            cap_first = cap_pair[0]
            cap_second = cap_pair[1]
            if not isinstance(cap_first, ConicParameterEndpoint) or not isinstance(
                cap_second, ConicParameterEndpoint
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder fillet: cap conic roots are unavailable"
                )
            cap_roots = (
                ConicParameterEndpoint(
                    cap_first.root,
                    cap_pcurve,
                    Fraction(0),
                    cap_first.point_exact,
                ),
                ConicParameterEndpoint(
                    cap_second.root,
                    cap_pcurve,
                    Fraction(1),
                    cap_second.point_exact,
                ),
            )
        else:
            if boundary.cap_face_pcurves is None:
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder chamfer: cap face p-curve is unavailable"
                )
            cap_pcurve = boundary.cap_face_pcurves[end]
            cap_roots = boundary.cap_roots[end]
        staged.append(
            builder.coedge(
                boundary.cap_edges[end],
                cap_sense,
                cap_pcurve,
                endpoint_roots=cap_roots,
            )
        )
    return staged


def _stage_faces(
    junction: _Junction,
    blend: _Blend,
    model: BRepModel,
    selected: int,
    edges: _StagedEdges,
    boundary: _StagedBlend,
    /,
) -> None:
    builder, geometry = edges.builder, junction.geometry
    face_sides = {side.face: (i, side) for i, side in enumerate(junction.sides)}
    cap_indices = {face: end for end, face in enumerate(junction.cap_faces)}
    for face, loops in enumerate(geometry.face_loops):
        staged_loops = [
            _stage_source_loop(
                junction,
                blend,
                model,
                selected,
                edges,
                boundary,
                face,
                loop,
            )
            for loop in loops
        ]
        if face in cap_indices:
            cap_edge = boundary.cap_edges[cap_indices[face]]
            if (
                sum(
                    builder.coedge_edges[coedge] == cap_edge
                    for loop in staged_loops
                    for coedge in loop
                )
                != 1
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: cap does not contain one adjacent junction"
                )
        box = np.asarray(model.parameter_bounds[face]).copy()
        if face in face_sides:
            i, side = face_sides[face]
            contact_pcurve = edges.source_contact_pcurves[i]
            contact_bounds = (
                (contact_pcurve.root.parameter_lower, contact_pcurve.root.parameter_upper)
                if isinstance(contact_pcurve, ConicRootLinePCurve)
                else (blend.angles[i], blend.angles[i])
            )
            box[:, 0] = (
                min(contact_bounds[0], side.retained_angle_bounds[0]),
                max(contact_bounds[1], side.retained_angle_bounds[1]),
            )
        elif face in cap_indices:
            end = cap_indices[face]
            cap_patch = model.patches[face]
            if not isinstance(cap_patch, PlanePatch):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: nonplanar cap"
                )
            if blend.radius > 0.0:
                cap_curve = boundary.cap_curves[end]
                if not isinstance(cap_curve, (CircleCurve, LineCurve)):
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder fillet: cap curve is not circular or linear"
                    )
                pc = _cap_pcurve(cap_patch, cap_curve)
                bound = pc.bounding_box(*boundary.face_bounds)
            else:
                if boundary.cap_face_pcurves is None:
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder chamfer: cap face p-curve is unavailable"
                    )
                pc = boundary.cap_face_pcurves[end]
                if not isinstance(pc, IntersectionPCurve):
                    raise BRepBlendUnsupported(
                        "unsupported parallel-cylinder chamfer: cap face branch p-curve is unavailable"
                    )
                bound = pc.bounding_box(pc.first, pc.last)
            box[0] = np.minimum(box[0], bound[0])
            box[1] = np.maximum(box[1], bound[1])
        builder.face(
            model.patches[face],
            box,
            int(float(model.orientation[face])),
            model.physical_tags[face],
            staged_loops,
        )


def _stage_blend_face(
    junction: _Junction,
    blend: _Blend,
    edges: _StagedEdges,
    boundary: _StagedBlend,
    /,
) -> None:
    builder, vertical_edges = edges.builder, edges.vertical_edges
    cap_edges, endpoints = boundary.cap_edges, boundary.endpoints
    blend_patch = boundary.patch
    upper_side, lower_side = endpoints[1], endpoints[0]
    if blend.radius > 0.0:
        upper_pcurve = edges.blend_contact_pcurves[upper_side]
        lower_pcurve = edges.blend_contact_pcurves[lower_side]
        upper_roots = edges.blend_endpoint_roots[upper_side]
        lower_roots = edges.blend_endpoint_roots[lower_side]
        cap_lower = LineCurve((0.0, 0.0), (1.0, 0.0))
        cap_upper = LineCurve((0.0, junction.height), (1.0, 0.0))
        if boundary.cap_roots is None:
            cap_lower_roots: tuple[RootEndpoint | None, RootEndpoint | None] = (
                None,
                None,
            )
            cap_upper_roots = cap_lower_roots
        else:
            cap_lower_pair, cap_upper_pair = boundary.cap_roots
            cap_lower_first = cap_lower_pair[0]
            cap_lower_second = cap_lower_pair[1]
            cap_upper_first = cap_upper_pair[0]
            cap_upper_second = cap_upper_pair[1]
            if (
                not isinstance(cap_lower_first, ConicParameterEndpoint)
                or not isinstance(cap_lower_second, ConicParameterEndpoint)
                or not isinstance(cap_upper_first, ConicParameterEndpoint)
                or not isinstance(cap_upper_second, ConicParameterEndpoint)
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder fillet: cap conic roots are unavailable"
                )
            cap_lower_roots = (
                ConicParameterEndpoint(
                    cap_lower_first.root,
                    cap_lower,
                    None,
                    cap_lower_first.point_exact,
                ),
                ConicParameterEndpoint(
                    cap_lower_second.root,
                    cap_lower,
                    None,
                    cap_lower_second.point_exact,
                ),
            )
            cap_upper_roots = (
                ConicParameterEndpoint(
                    cap_upper_first.root,
                    cap_upper,
                    None,
                    cap_upper_first.point_exact,
                ),
                ConicParameterEndpoint(
                    cap_upper_second.root,
                    cap_upper,
                    None,
                    cap_upper_second.point_exact,
                ),
            )
    else:
        if (
            not isinstance(blend_patch, RuledSurface)
            or boundary.blend_cap_pcurves is None
            or boundary.blend_cap_roots is None
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: ruled blend boundary roots are unavailable"
            )
        upper_pcurve = edges.blend_contact_pcurves[upper_side]
        lower_pcurve = edges.blend_contact_pcurves[lower_side]
        upper_roots = edges.blend_endpoint_roots[upper_side]
        lower_roots = edges.blend_endpoint_roots[lower_side]
        cap_lower, cap_upper = boundary.blend_cap_pcurves
        cap_lower_roots, cap_upper_roots = boundary.blend_cap_roots
    blend_loop = [
        builder.coedge(cap_edges[0], 1, cap_lower, endpoint_roots=cap_lower_roots),
        builder.coedge(
            vertical_edges[upper_side],
            1,
            upper_pcurve,
            endpoint_roots=upper_roots,
        ),
        builder.coedge(cap_edges[1], -1, cap_upper, endpoint_roots=cap_upper_roots),
        builder.coedge(
            vertical_edges[lower_side],
            -1,
            lower_pcurve,
            endpoint_roots=lower_roots,
        ),
    ]
    if blend.radius > 0.0:
        orientation = blend.orientation
    else:
        side0 = junction.sides[0]
        old_normal = side0.radial_orientation * (
            blend.points[0] - junction.centers[0]
        )
        tangent = blend.points[1] - blend.points[0]
        normal = np.asarray((tangent[1], -tangent[0]))
        orientation = 1 if float(normal @ old_normal) > 0.0 else -1
    boxes = np.stack(
        [
            builder.pcurves[coedge].bounding_box(
                *builder.edge_ranges[builder.coedge_edges[coedge]]
            )
            for coedge in blend_loop
        ]
    )
    face_bounds = np.stack(
        (np.min(boxes[:, 0], axis=0), np.max(boxes[:, 1], axis=0))
    )
    builder.face(
        blend_patch,
        face_bounds,
        orientation,
        "cylinder" if blend.radius > 0.0 else "plane",
        [blend_loop],
    )


def _stage(
    junction: _Junction,
    blend: _Blend,
    model: BRepModel,
    selected: int,
    /,
) -> _Builder:
    edges = _stage_edges(junction, blend, selected, model)
    boundary = _stage_caps(junction, blend, edges)
    _stage_faces(junction, blend, model, selected, edges, boundary)
    _stage_blend_face(junction, blend, edges, boundary)
    return edges.builder


def blend_parallel_cylinders(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
) -> BRepModel:
    """Replace one parallel-cylinder junction by a certified analytic blend."""
    if (radius is None) == (setbacks is None):
        raise ValueError(
            f"blend_parallel_cylinders.distances must specify one radius or setback pair; got radius={radius!r}, setbacks={setbacks!r}"
        )
    values = (radius,) if radius is not None else setbacks
    if values is None or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            f"blend_parallel_cylinders.distances must be positive and finite; got {values!r}"
        )
    junction = _junction(model, edge_index)
    if radius is not None:
        blend = _fillet(junction, radius)
    elif setbacks is not None:
        blend = _chamfer(junction, setbacks)
    else:
        raise ValueError(
            "blend_parallel_cylinders.distances must contain a specification; got none"
        )
    builder = _stage(junction, blend, model, edge_index)
    geometry = junction.geometry
    containing = [
        i
        for i, faces in enumerate(geometry.shell_faces)
        if junction.sides[0].face in faces
    ]
    if (
        len(containing) != 1
        or junction.sides[1].face not in geometry.shell_faces[containing[0]]
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: junction does not belong to one shell"
        )
    shell = containing[0]
    shell_faces = tuple(
        (*faces, len(model.patches)) if i == shell else faces
        for i, faces in enumerate(geometry.shell_faces)
    )
    shell_orientations = tuple(
        (*orientations, 1) if i == shell else orientations
        for i, orientations in enumerate(geometry.shell_orientations)
    )
    geometry_, patches, bounds, signs, tags, trims = builder.geometry(
        solid=bool(geometry.solid_shells),
        shell_faces=shell_faces,
        shell_orientations=shell_orientations,
        solid_shells=geometry.solid_shells,
        assembly_graph=geometry.assembly_graph,
    )
    return assemble_brep_model(
        geometry_,
        patches,
        bounds,
        signs,
        tags,
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        trim_domains=trims,
        curve_surface_tolerance=model.report.curve_surface_tolerance,
        curve_surface_scale=model.report.curve_surface_scale,
    )


def _normal_difference(parameters: Array) -> Array:
    """Outward unit-normal difference in rationally certified orthonormal frames."""
    source_angle, blend_angle = parameters[0], parameters[1]
    source_first, source_second = parameters[2:5], parameters[5:8]
    blend_first, blend_second = parameters[8:11], parameters[11:14]
    source = parameters[14] * (
        jnp.cos(source_angle) * source_first + jnp.sin(source_angle) * source_second
    )
    blended = parameters[15] * (
        jnp.cos(blend_angle) * blend_first + jnp.sin(blend_angle) * blend_second
    )
    return source - blended


def parallel_cylinder_g1_bounds(
    model: BRepModel,
    result: BRepModel,
    edge_index: int,
    /,
) -> np.ndarray:
    """Enclose represented boundary normal angles in source-face order.

    The algebraic construction has rho_i = R_i - s o_i r and
    P_i = C_i + R_i (F-C_i)/rho_i. Consequently
    s (P_i-F)/r = o_i (P_i-C_i)/R_i: the two outward normals agree
    identically along each axial boundary. Returned intervals account for
    coefficient and angular realization through interval extension.
    """
    junction = _junction(model, edge_index)
    geometry = result.geometry
    blend_face = len(result.patches) - 1
    patch = result.patches[blend_face]
    if not isinstance(patch, CylinderPatch):
        raise ValueError(
            f"parallel_cylinder_g1_bounds.result must contain a published cylindrical fillet; got patch={type(patch).__name__}"
        )
    _exact_frame(patch)
    program = prepare_interval_function(_normal_difference, 16, batch_capacity=2)
    parameters: list[np.ndarray] = []
    for side in junction.sides:
        edges = [
            edge
            for edge in result.topology.face_edges[side.face]
            if blend_face in result.topology.edge_faces[edge]
        ]
        if len(edges) != 1:
            raise ValueError(
                f"parallel_cylinder_g1_bounds.boundary must contain one shared edge per source face; got {len(edges)}"
            )
        angles: list[float] = []
        for face in (side.face, blend_face):
            coedges = [
                coedge
                for loop in geometry.face_loops[face]
                for coedge in loop
                if geometry.coedge_edges[coedge] == edges[0]
            ]
            if len(coedges) != 1:
                raise ValueError(
                    f"parallel_cylinder_g1_bounds.coedges must contain one boundary use; got {len(coedges)}"
                )
            pcurve = geometry.pcurves[coedges[0]]
            if isinstance(pcurve, ExactLineCurve):
                first, last = pcurve.first_exact, pcurve.last_exact
                if first[0] != last[0]:
                    raise ValueError(
                        "parallel_cylinder_g1_bounds.pcurve must be an angular line isoline"
                    )
                angles.append(float(first[0]))
            elif isinstance(pcurve, ConicRootLinePCurve):
                angles.append(float(pcurve.angle))
            elif isinstance(pcurve, LineCurve) and pcurve.direction_exact[0] == 0:
                angles.append(float(pcurve.origin_exact[0]))
            else:
                raise ValueError(
                    f"parallel_cylinder_g1_bounds.pcurve must be an angular line isoline; got {type(pcurve).__name__}"
                )
        parameters.append(
            np.concatenate(
                (
                    np.asarray(angles),
                    np.asarray(side.patch.first_axis),
                    np.asarray(side.patch.second_axis),
                    np.asarray(patch.first_axis),
                    np.asarray(patch.second_axis),
                    np.asarray(
                        (side.radial_orientation, float(result.orientation[blend_face]))
                    ),
                )
            )
        )
    points = np.stack(parameters)
    lower, upper = program.evaluate(points, points)
    magnitude = np.maximum(np.abs(lower), np.abs(upper))
    rows = _rational_rows(magnitude)
    rows = tuple(
        tuple(
            rational_coefficient(value, role="parallel_cylinder.normal_bound")
            for value in row
        )
        for row in rows
    )
    norms = np.asarray(tuple(fraction_norm_upper(row) for row in rows))
    if np.any(~np.isfinite(norms)) or np.any(norms >= 2.0):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: normal-angle enclosure is unresolved"
        )
    angles = 2.0 * np.arcsin(np.nextafter(norms / 2.0, np.inf))
    for _ in range(4):
        angles = np.nextafter(angles, np.inf)
    return np.stack((np.zeros(2, dtype=np.float64), angles), axis=1)


class ParallelCylinderRadiusMap(AbstractBRepParameterMap):
    """One radius parameter realizing correlated carriers, trims, and mesh UVs."""

    source: BRepModel
    reference: BRepModel
    edge_index: int = eqx.field(static=True)
    initial_radius: float = eqx.field(static=True)
    parameter_id: ParameterId = eqx.field(static=True)

    def __init__(
        self,
        source: BRepModel,
        edge_index: int,
        radius: float,
        result: BRepModel,
        *,
        parameter_id: ParameterId,
    ) -> None:
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(
                f"ParallelCylinderRadiusMap.radius must be positive and finite; got {radius!r}"
            )
        if result.tessellation is not None:
            raise BRepBlendUnsupported(
                "parallel-cylinder radius sensitivity cannot retain a tessellation fidelity certificate under parameter motion"
            )
        self.source = source
        self.reference = result
        self.edge_index = edge_index
        self.initial_radius = radius
        self.parameter_id = parameter_id

    def compile(
        self, context: _ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.reference.model_id:
            raise ValueError(
                f"ParallelCylinderRadiusMap.model must match the reference model; got {model.model_id!r}"
            )
        if (
            self.source.coordinate_contract.spatial_id
            != model.coordinate_contract.spatial_id
        ):
            raise ValueError(
                "ParallelCylinderRadiusMap.source and model coordinate contracts must match."
            )
        if model.tessellation is not None:
            raise BRepBlendUnsupported(
                "parallel-cylinder radius sensitivity cannot retain a tessellation fidelity certificate under parameter motion"
            )
        junction = _junction(self.source, self.edge_index)
        blend = _fillet(junction, self.initial_radius)
        radius_bounds = _radius_epoch(junction, blend, self.initial_radius)
        radius = context.bind(
            self.parameter_id,
            jnp.asarray(self.initial_radius, dtype=jnp.float64),
            role="length",
            physical_scale=self.initial_radius,
            bounds=radius_bounds,
        )
        boundary_edges = tuple(
            len(junction.geometry.edge_curves) - 1 + index for index in range(2)
        )
        cap_changes: list[tuple[int, int, int]] = []
        for side_index, side in enumerate(junction.sides):
            for edge in side.cap_edges:
                endpoint = junction.geometry.edge_vertices[edge].index(
                    next(
                        vertex
                        for vertex in junction.geometry.edge_vertices[edge]
                        if vertex in junction.vertices
                    )
                )
                cap_changes.append(
                    (edge - (edge > self.edge_index), side_index, endpoint)
                )
        first_beta = atan2(
            float(blend.points[0, 1] - blend.center[1]),
            float(blend.points[0, 0] - blend.center[0]),
        )
        delta = junction.centers[1] - junction.centers[0]
        branch = (
            1.0
            if delta[0] * -junction.centers[0, 1] - delta[1] * -junction.centers[0, 0]
            > 0.0
            else -1.0
        )
        return _BoundParallelCylinderRadiusMap(
            radius=radius,
            reference=model,
            source_patches=tuple(side.patch for side in junction.sides),
            source_faces=tuple(side.face for side in junction.sides),
            source_bound_endpoints=tuple(
                int(
                    np.argmin(
                        np.abs(
                            np.asarray(model.parameter_bounds[side.face, :, 0])
                            - blend.angles[index]
                        )
                    )
                )
                for index, side in enumerate(junction.sides)
            ),
            cap_faces=junction.cap_faces,
            boundary_edges=boundary_edges,
            cap_changes=tuple(cap_changes),
            branch=branch,
            orientation=float(blend.orientation),
            centers=jnp.asarray(junction.centers),
            origin=jnp.asarray(junction.origin),
            first=jnp.asarray(junction.first),
            second=jnp.asarray(junction.second),
            axis=jnp.asarray(junction.axis),
            height=jnp.asarray(junction.height, dtype=jnp.float64),
            radial_orientations=jnp.asarray(
                tuple(side.radial_orientation for side in junction.sides),
                dtype=jnp.float64,
            ),
            reference_angles=jnp.asarray(blend.angles, dtype=jnp.float64),
            reference_betas=jnp.asarray(
                (first_beta, first_beta + blend.sweep), dtype=jnp.float64
            ),
        )


class _BoundParallelCylinderRadiusMap(AbstractBoundBRepParameterMap):
    radius: ParameterBinding = eqx.field(static=True)
    reference: BRepModel
    source_patches: tuple[CylinderPatch, ...]
    source_faces: tuple[int, ...] = eqx.field(static=True)
    source_bound_endpoints: tuple[int, ...] = eqx.field(static=True)
    cap_faces: tuple[int, int] = eqx.field(static=True)
    boundary_edges: tuple[int, ...] = eqx.field(static=True)
    cap_changes: tuple[tuple[int, int, int], ...] = eqx.field(static=True)
    branch: float = eqx.field(static=True)
    orientation: float = eqx.field(static=True)
    centers: Array
    origin: Array
    first: Array
    second: Array
    axis: Array
    height: Array
    radial_orientations: Array
    reference_angles: Array
    reference_betas: Array

    def _sections(self, radius: Array) -> tuple[Array, Array, Array, Array]:
        radii = jnp.stack(tuple(patch.radius for patch in self.source_patches))
        rho = radii - self.orientation * self.radial_orientations * radius
        delta = self.centers[1] - self.centers[0]
        squared = jnp.sum(delta * delta)
        lam = (squared + rho[0] ** 2 - rho[1] ** 2) / (2.0 * squared)
        q = rho[0] ** 2 / squared - lam**2
        perpendicular = jnp.stack((-delta[1], delta[0]))
        center = self.centers[0] + lam * delta + self.branch * jnp.sqrt(q) * perpendicular
        points = self.centers + (radii / rho)[:, None] * (center - self.centers)
        world = self.origin + points[:, :1] * self.first + points[:, 1:] * self.second
        angles: list[Array] = []
        for i, patch in enumerate(self.source_patches):
            radial = world[i] - patch.origin
            angle = jnp.arctan2(radial @ patch.second_axis, radial @ patch.first_axis)
            angles.append(
                angle
                + 2.0
                * jnp.pi
                * jnp.round((self.reference_angles[i] - angle) / (2.0 * jnp.pi))
            )
        vectors = points - center
        betas = jnp.arctan2(vectors[:, 1], vectors[:, 0])
        betas += 2.0 * jnp.pi * jnp.round((self.reference_betas - betas) / (2.0 * jnp.pi))
        return center, world, jnp.stack(angles), betas

    def realize(self, state: DesignState, /) -> BRepParameterGeometry:
        radius = self.radius.read(state)
        center, world, angles, betas = self._sections(radius)
        spatial_center = self.origin + center[0] * self.first + center[1] * self.second
        reference_geometry = self.reference.geometry
        patch = self.reference.patches[-1]
        if not isinstance(patch, CylinderPatch):
            raise ValueError(
                f"ParallelCylinderRadiusMap.patch must be cylindrical; got {type(patch).__name__}"
            )
        patches = (
            *self.reference.patches[:-1],
            _RealizedCylinderPatch(
                spatial_center,
                patch.first_axis,
                patch.second_axis,
                patch.axis,
                radius,
                periods=patch.periods,
                period_symbols=patch.native_period_symbols(),
            ),
        )
        curves = list(reference_geometry.curves)
        vertices = reference_geometry.vertex_points
        pcurves = self._realize_pcurves(
            tuple(reference_geometry.pcurves), angles, betas, spatial_center, radius
        )
        ranges = reference_geometry.edge_ranges
        for index, edge in enumerate(self.boundary_edges):
            curve_index = reference_geometry.edge_curves[edge]
            curve = curves[curve_index]
            if not isinstance(curve, SurfaceMappedCurve):
                raise ValueError(
                    f"ParallelCylinderRadiusMap.boundary must have a native surface-mapped carrier; got {type(curve).__name__}"
                )
            coedges = [
                coedge
                for loop in reference_geometry.face_loops[self.source_faces[index]]
                for coedge in loop
                if reference_geometry.coedge_edges[coedge] == edge
            ]
            if len(coedges) != 1 or not isinstance(
                pcurves[coedges[0]], _RealizedLineCurve
            ):
                raise ValueError(
                    "ParallelCylinderRadiusMap.boundary must retain one native source p-curve"
                )
            curves[curve_index] = eqx.tree_at(
                lambda value: value.pcurve, curve, pcurves[coedges[0]]
            )
            dynamic_curve = curves[curve_index]
            for end, vertex in enumerate(reference_geometry.edge_vertices[edge]):
                vertices = vertices.at[vertex].set(
                    dynamic_curve.evaluate(jnp.asarray(float(end), dtype=jnp.float64))
                )
        for edge, side, endpoint in self.cap_changes:
            ranges = ranges.at[edge, endpoint].set(angles[side])
        low, high = jnp.min(betas), jnp.max(betas)
        for end in range(2):
            edge = self.boundary_edges[-1] + 1 + end
            curve_index = reference_geometry.edge_curves[edge]
            curve = curves[curve_index]
            if not isinstance(curve, CircleCurve):
                raise ValueError(
                    f"ParallelCylinderRadiusMap.cap must have a native circle carrier; got {type(curve).__name__}"
                )
            curves[curve_index] = _RealizedConicCurve(
                "circle",
                spatial_center + end * self.height * self.axis,
                curve.first_axis,
                curve.second_axis,
                radius,
                parameter_domain=curve.parameter_domain,
                period=curve.period,
                period_symbol=curve.native_period_symbol(),
            )
            ranges = ranges.at[edge].set(jnp.stack((low, high)))
        bounds = self._realize_bounds(angles, betas, spatial_center, radius)
        return BRepParameterGeometry(
            patches=patches,
            source_geometry=reference_geometry,
            parameter_bounds=bounds,
            orientation=self.reference.orientation,
            trim_domains=self.reference.trim_domains,
            triangle_parameters=None,
            source_patches=self.reference.patches,
            vertex_points=vertices,
            curves=tuple(curves),
            pcurves=pcurves,
            edge_ranges=ranges,
        )

    def _realize_pcurves(
        self,
        reference: tuple[AbstractCurve | IntersectionPCurve, ...],
        angles: Array,
        betas: Array,
        center: Array,
        radius: Array,
    ) -> tuple[AbstractCurve | IntersectionPCurve, ...]:
        geometry = self.reference.geometry
        pcurves = list(reference)
        for face, loops in enumerate(geometry.face_loops):
            for loop in loops:
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    pc = pcurves[coedge]
                    if edge in self.boundary_edges:
                        side = self.boundary_edges.index(edge)
                        if not isinstance(pc, ConicRootLinePCurve):
                            raise BRepBlendUnsupported(
                                "parallel-cylinder radius map requires an exact rooted contact-line template"
                            )
                        height_origin = jnp.asarray(
                            float(pc.height_origin_exact), dtype=jnp.float64
                        )
                        height_direction = jnp.asarray(
                            float(pc.height_direction_exact), dtype=jnp.float64
                        )
                        angle = (
                            angles[side]
                            if face != len(self.reference.patches) - 1
                            else betas[side]
                        )
                        pcurves[coedge] = _RealizedLineCurve(
                            jnp.stack((angle, height_origin)),
                            jnp.stack((jnp.zeros_like(angle), height_direction)),
                            parameter_domain=(0.0, 1.0),
                        )
                    elif face in self.cap_faces and edge > self.boundary_edges[-1]:
                        if not isinstance(pc, CircleCurve):
                            raise ValueError(
                                f"ParallelCylinderRadiusMap.pcurve must be a native circle on the cap; got {type(pc).__name__}"
                            )
                        end = self.cap_faces.index(face)
                        patch = self.reference.patches[face]
                        if not isinstance(patch, PlanePatch):
                            raise ValueError(
                                f"ParallelCylinderRadiusMap.cap must be planar; got {type(patch).__name__}"
                            )
                        axes = jnp.stack((patch.first_axis, patch.second_axis))
                        pc_center = axes @ (
                            center + end * self.height * self.axis - patch.origin
                        )
                        pcurves[coedge] = _RealizedConicCurve(
                            "circle",
                            pc_center,
                            pc.first_axis,
                            pc.second_axis,
                            radius,
                            parameter_domain=pc.parameter_domain,
                            period=pc.period,
                            period_symbol=pc.native_period_symbol(),
                        )
        return tuple(pcurves)

    def _realize_bounds(
        self, angles: Array, betas: Array, center: Array, radius: Array
    ) -> Array:
        bounds = self.reference.parameter_bounds
        for i, face in enumerate(self.source_faces):
            endpoint = self.source_bound_endpoints[i]
            bounds = bounds.at[face, endpoint, 0].set(angles[i])
        bounds = bounds.at[-1, :, 0].set(jnp.stack((jnp.min(betas), jnp.max(betas))))
        for face in self.cap_faces:
            patch = self.reference.patches[face]
            if not isinstance(patch, PlanePatch):
                raise ValueError(
                    f"ParallelCylinderRadiusMap.cap must be planar; got {type(patch).__name__}"
                )
            axes = jnp.stack((patch.first_axis, patch.second_axis))
            projected = axes @ (center - patch.origin)
            bounds = bounds.at[face, 0].set(
                jnp.minimum(bounds[face, 0], projected - radius)
            )
            bounds = bounds.at[face, 1].set(
                jnp.maximum(bounds[face, 1], projected + radius)
            )
        return bounds


def _offset_vectors(parameters: Array) -> Array:
    radius = parameters[0]
    radii = parameters[1:3]
    radial_orientations = parameters[3:5]
    orientation = parameters[5]
    centers = parameters[6:10].reshape(2, 2)
    rho = radii - orientation * radial_orientations * radius
    delta = centers[1] - centers[0]
    d2 = jnp.sum(delta * delta)
    lam = (d2 + rho[0] ** 2 - rho[1] ** 2) / (2.0 * d2)
    q = rho[0] ** 2 / d2 - lam**2
    center = (
        centers[0]
        + lam * delta
        + parameters[10] * jnp.sqrt(q) * jnp.stack((-delta[1], delta[0]))
    )
    vectors = (radii / rho)[:, None] * (center - centers)
    return jnp.concatenate((rho, q[None], vectors.reshape(-1)))


def _exact_scalar_interval(value: ExactScalar, /) -> tuple[float, float]:
    if isinstance(value, Fraction):
        return fraction_down(value), fraction_up(value)
    lower, upper = value.interval(bits=128)
    return fraction_down(lower), fraction_up(upper)


def _interval_sector_sign(
    lower: np.ndarray,
    upper: np.ndarray,
    rays: tuple[ExactScalar, ExactScalar],
    direction: int,
    reverse: bool,
) -> tuple[float, float]:
    ray0 = _exact_scalar_interval(rays[0])
    ray1 = _exact_scalar_interval(rays[1])
    first = interval_multiply(
        (np.asarray(ray0[0]), np.asarray(ray0[1])),
        (np.asarray(lower[1]), np.asarray(upper[1])),
    )
    second = interval_multiply(
        (np.asarray(ray1[0]), np.asarray(ray1[1])),
        (np.asarray(lower[0]), np.asarray(upper[0])),
    )
    low, high = interval_subtract(first, second)
    multiplier = -direction if reverse else direction
    return (float(low), float(high)) if multiplier > 0 else (-float(high), -float(low))


def _radius_epoch(
    junction: _Junction,
    blend: _Blend,
    radius: float,
    /,
) -> tuple[float, float]:
    """Admit an open radius epoch with interval-proven branch and trim invariants."""
    program = prepare_interval_function(_offset_vectors, 11, batch_capacity=1)
    delta = junction.centers[1] - junction.centers[0]
    branch = (
        1.0
        if delta[0] * -junction.centers[0, 1] - delta[1] * -junction.centers[0, 0] > 0.0
        else -1.0
    )
    parameters = np.asarray(
        (
            radius,
            *(float(side.patch.radius) for side in junction.sides),
            *(side.radial_orientation for side in junction.sides),
            blend.orientation,
            *junction.centers.reshape(-1),
            branch,
        )
    )
    rays = tuple(_sector_rays(side, junction) for side in junction.sides)
    width = radius / 2.0
    while width > 0.0:
        low, high = parameters.copy(), parameters.copy()
        low[0], high[0] = radius - width, radius + width
        if low[0] == radius or high[0] == radius or not np.isfinite(high[0]):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder radius map: no representable open certified epoch"
            )
        lower, upper = program.evaluate(low[None], high[None])
        valid = bool(
            np.all(np.isfinite(lower))
            and np.all(np.isfinite(upper))
            and np.all(lower[0, :3] > 0.0)
        )
        for i, side in enumerate(junction.sides):
            start = 3 + 2 * i
            handed = float(
                np.cross(
                    np.asarray(side.patch.first_axis), np.asarray(side.patch.second_axis)
                )
                @ junction.axis
            )
            direction = (1 if side.retained_angle > side.angle else -1) * (
                1 if handed > 0.0 else -1
            )
            a = _interval_sector_sign(
                lower[0, start : start + 2],
                upper[0, start : start + 2],
                rays[i][0],
                direction,
                False,
            )
            b = _interval_sector_sign(
                lower[0, start : start + 2],
                upper[0, start : start + 2],
                rays[i][1],
                direction,
                True,
            )
            definite = (a[0] > 0.0 or a[1] < 0.0) and (b[0] > 0.0 or b[1] < 0.0)
            retained = (
                (a[0] > 0.0 or b[0] > 0.0)
                if abs(side.retained_angle - side.angle) > pi
                else (a[0] > 0.0 and b[0] > 0.0)
            )
            valid = valid and definite and retained
        if valid:
            return float(low[0]), float(high[0])
        width /= 2.0
    raise BRepBlendUnsupported(
        "unsupported parallel-cylinder radius map: radius epoch is unresolved"
    )


def parallel_cylinder_parameter_range(
    source: BRepModel,
    edge_index: int,
    radius: float,
    /,
) -> CertifiedParameterRange:
    """Expose the existing interval-proved open contact/branch epoch."""
    try:
        junction = _junction(source, edge_index)
        blend = _fillet(junction, radius)
        lower, upper = _radius_epoch(junction, blend, radius)
    except BRepBlendUnsupported as error:
        return unknown_parameter_range("radius", str(error))
    return CertifiedParameterRange(
        "radius",
        (lower, upper),
        end_status=("certified_inner_bound", "certified_inner_bound"),
        constraints=("trim_vanishes", "trim_vanishes"),
        entity_ids=((source.edge_ids[edge_index],), (source.edge_ids[edge_index],)),
        evidence=(
            "Parallel-cylinder reduction preserves the rational source section.",
            "Whole-epoch interval evaluation proves positive offset radii, discriminant, retained contact sectors, and fixed branch signs.",
        ),
    )
