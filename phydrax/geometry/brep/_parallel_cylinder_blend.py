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
from math import atan2, pi, sqrt

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from ..._strict import StrictModule
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...discretization._exact_algebraic_geometry import rational_coefficient
from .._interval_enclosure import (
    interval_multiply,
    interval_subtract,
    prepare_interval_function,
)
from .._planar_primitives import rational_points, reserve_fraction_work
from ..design._schema import (
    DesignState,
    ParameterBinding,
    ParameterCollector,
    ParameterId,
)
from ._constructors import _radical_sign, BRepTessellationPolicy, Builder, finish
from ._coordinate_request import coordinate_source_request
from ._differentiable import (
    AbstractBoundBRepParameterMap,
    AbstractBRepParameterMap,
    BRepParameterGeometry,
)
from ._fillet import BRepBlendUnsupported
from ._intersection_curve import IntersectionPCurve
from ._model import BRepGeometry, BRepModel
from ._parameter_range import CertifiedParameterRange, unknown_parameter_range
from ._patches import (
    AbstractCurve,
    CircleCurve,
    CylinderPatch,
    LineCurve,
    PlanePatch,
)
from ._request_preparations import TrimRequestPreparations
from ._source_correspondence import fraction_norm_upper


class _Side(StrictModule):
    face: int
    patch: CylinderPatch
    coedge: int
    angle: float
    retained_angle: float
    cap_edges: tuple[int, int]
    heights: tuple[float, float]
    radial_orientation: int


class _Junction(StrictModule):
    geometry: BRepGeometry
    sides: tuple[_Side, _Side]
    cap_faces: tuple[int, int]
    vertices: tuple[int, int]
    origin: np.ndarray
    first: np.ndarray
    second: np.ndarray
    axis: np.ndarray
    centers: np.ndarray
    corner: np.ndarray
    height: float


class _Blend(StrictModule):
    points: np.ndarray
    angles: tuple[float, float]
    center: np.ndarray
    radius: float
    sweep: float
    orientation: int


def _exact_frame(
    patch: CylinderPatch | PlanePatch, budget: CoordinateEnclosureBudget
) -> None:
    axes = [np.asarray(patch.first_axis), np.asarray(patch.second_axis)]
    if isinstance(patch, CylinderPatch):
        axes.append(np.asarray(patch.axis))
    rows = rational_points(np.stack(axes), budget=budget)
    reserve_fraction_work(rows, 81, 9, 8, budget=budget)
    for i, left in enumerate(rows):
        for j, right in enumerate(rows):
            if sum((a * b for a, b in zip(left, right, strict=True)), Fraction()) != (
                i == j
            ):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: frame lacks a rational orthonormal certificate"
                )


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
    if not isinstance(pcurve, LineCurve) or float(pcurve.direction[0]) != 0.0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: junction is not a straight angular isoline"
        )
    angle = float(pcurve.origin[0])
    caps: list[tuple[float, int]] = []
    other_angles: list[float] = []
    for c in loops[0]:
        pc = geometry.pcurves[c]
        if not isinstance(pc, LineCurve):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: rectangle requires native line p-curves"
            )
        origin, direction = np.asarray(pc.origin), np.asarray(pc.direction)
        if np.array_equal(direction, (1.0, 0.0)) and origin[0] == 0.0:
            caps.append((float(origin[1]), geometry.coedge_edges[c]))
        elif direction[0] == 0.0 and direction[1] != 0.0:
            if c != coedge:
                other_angles.append(float(origin[0]))
        else:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: noncanonical rectangular chart"
            )
    if len(caps) != 2 or len(other_angles) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: missing circle end boundaries"
        )
    caps.sort()
    retained = other_angles[0]
    if not 0.0 < abs(retained - angle) < 2.0 * pi:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: vanishing or multiply wound source trim"
        )
    for _, cap in caps:
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
        (caps[0][1], caps[1][1]),
        (caps[0][0], caps[1][0]),
        radial,
    )


def _end_caps(
    model: BRepModel,
    sides: tuple[_Side, _Side],
    axis: np.ndarray,
    budget: CoordinateEnclosureBudget,
) -> tuple[int, int]:
    cap_faces: list[int] = []
    for end in range(2):
        uses: list[set[int]] = []
        for side in sides:
            same = np.array_equal(np.asarray(side.patch.axis), axis)
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
        _exact_frame(patch, budget)
        rows = rational_points(
            np.stack((axis, np.asarray(patch.first_axis), np.asarray(patch.second_axis))),
            budget=budget,
        )
        reserve_fraction_work(rows, 18, 2, 4, budget=budget)
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


def _junction(
    model: BRepModel, edge: int, budget: CoordinateEnclosureBudget
) -> _Junction:
    geometry = model.geometry
    if geometry is None:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: native edge geometry is required"
        )
    if not 0 <= edge < len(geometry.edge_curves):
        raise BRepBlendUnsupported("parallel-cylinder blend edge index is out of range")
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
        _exact_frame(side.patch, budget)
    if any(root is not None for root in geometry.vertex_roots) or any(
        root is not None for pair in geometry.edge_endpoint_roots for root in pair
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: root-authored source endpoints need a root-preserving rebuild"
        )
    first, second, axis = (
        np.asarray(sides[0].patch.first_axis),
        np.asarray(sides[0].patch.second_axis),
        np.cross(
            np.asarray(sides[0].patch.first_axis), np.asarray(sides[0].patch.second_axis)
        ),
    )
    other_axis = np.asarray(sides[1].patch.axis)
    if not (np.array_equal(other_axis, axis) or np.array_equal(other_axis, -axis)):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: axes are not exactly parallel"
        )
    endpoint_ids = geometry.edge_vertices[edge]
    endpoints = np.asarray(geometry.vertex_points)[list(endpoint_ids)]
    order = np.argsort(endpoints @ axis)
    vertices = (endpoint_ids[int(order[0])], endpoint_ids[int(order[1])])
    origin = endpoints[int(order[0])]
    height = float((endpoints[int(order[1])] - origin) @ axis)
    if height <= 0.0:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: vanishing junction length"
        )
    cap_faces = _end_caps(model, sides, axis, budget)
    _require_simple_vertices(geometry, sides, vertices, edge)
    centers = (
        np.stack([np.asarray(s.patch.origin) - origin for s in sides])
        @ np.stack((first, second)).T
    )
    corner = np.zeros(2, dtype=np.float64)
    junction = _Junction(
        geometry,
        sides,
        (cap_faces[0], cap_faces[1]),
        vertices,
        origin,
        first,
        second,
        axis,
        centers,
        corner,
        height,
    )
    _certify_section_coordinates(junction, budget)
    return junction


def _certify_section_coordinates(
    junction: _Junction,
    budget: CoordinateEnclosureBudget,
) -> None:
    """Prove the planar reduction preserves the authored rational coefficients."""
    top = np.asarray(junction.geometry.vertex_points[junction.vertices[1]])
    rows = rational_points(
        np.stack(
            (
                junction.origin,
                np.asarray(junction.sides[0].patch.origin),
                np.asarray(junction.sides[1].patch.origin),
                junction.first,
                junction.second,
                junction.axis,
                top,
            )
        ),
        budget=budget,
    )
    reserve_fraction_work(rows, 96, 9, 8, budget=budget)
    for side in range(2):
        displacement = tuple(a - b for a, b in zip(rows[side + 1], rows[0], strict=True))
        for coordinate in range(2):
            projected = sum(
                (a * b for a, b in zip(displacement, rows[coordinate + 3], strict=True)),
                Fraction(),
            )
            if projected != Fraction(float(junction.centers[side, coordinate])):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: planar reduction needs a nonbinary coefficient carrier"
                )
    axial = tuple(a - b for a, b in zip(rows[6], rows[0], strict=True))
    for frame in rows[3:5]:
        if sum((a * b for a, b in zip(axial, frame, strict=True)), Fraction()) != 0:
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder blend: junction is not a straight axial segment"
            )
    if sum((a * b for a, b in zip(axial, rows[5], strict=True)), Fraction()) != Fraction(
        junction.height
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: axial range needs a nonbinary coefficient carrier"
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


def _circle_ray(parameters: Array) -> Array:
    return (
        jnp.cos(parameters[0]) * parameters[1:3]
        + jnp.sin(parameters[0]) * parameters[3:5]
    )


def _sector_rays(
    side: _Side,
    junction: _Junction,
    budget: CoordinateEnclosureBudget,
) -> tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]]:
    budget.reserve(256, 4096)
    coefficients = (
        np.stack((np.asarray(side.patch.first_axis), np.asarray(side.patch.second_axis)))
        @ np.stack((junction.first, junction.second)).T
    )
    parameters = np.stack(
        [
            np.concatenate((np.asarray((angle,)), coefficients.reshape(-1)))
            for angle in (side.angle, side.retained_angle)
        ]
    )
    program = prepare_interval_function(_circle_ray, 5, batch_capacity=2)
    lower, upper = program.evaluate(parameters, parameters)
    corners = tuple(
        rational_points(
            np.asarray(((lo[0], lo[1]), (lo[0], hi[1]), (hi[0], lo[1]), (hi[0], hi[1]))),
            budget=budget,
        )
        for lo, hi in zip(lower, upper, strict=True)
    )
    return tuple(
        tuple(
            rational_coefficient(value, role="parallel_cylinder.sector_ray")
            for value in row
        )
        for row in corners[0]
    ), tuple(
        tuple(
            rational_coefficient(value, role="parallel_cylinder.sector_ray")
            for value in row
        )
        for row in corners[1]
    )


def _inside_sector(
    base: tuple[Fraction, Fraction],
    radical: tuple[Fraction, Fraction],
    q: Fraction,
    start: tuple[tuple[Fraction, ...], ...],
    end: tuple[tuple[Fraction, ...], ...],
    direction: int,
    major: bool,
) -> bool:
    a = tuple(
        _radical_sign(
            direction * (ray[0] * base[1] - ray[1] * base[0]),
            direction * (ray[0] * radical[1] - ray[1] * radical[0]),
            q,
        )
        for ray in start
    )
    b = tuple(
        _radical_sign(
            direction * (base[0] * ray[1] - base[1] * ray[0]),
            direction * (radical[0] * ray[1] - radical[1] * ray[0]),
            q,
        )
        for ray in end
    )
    if not ((min(a) > 0 or max(a) < 0) and (min(b) > 0 or max(b) < 0)):
        return False
    return (min(a) > 0 or min(b) > 0) if major else (min(a) > 0 and min(b) > 0)


def _contact_angle(
    junction: _Junction,
    side: _Side,
    point: np.ndarray,
    base: tuple[Fraction, Fraction],
    radical: tuple[Fraction, Fraction],
    q: Fraction,
    rays: tuple[tuple[tuple[Fraction, ...], ...], tuple[tuple[Fraction, ...], ...]],
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


def _fillet(
    junction: _Junction, radius: float, budget: CoordinateEnclosureBudget
) -> _Blend:
    rows = rational_points(
        np.vstack((junction.centers, junction.corner, (radius, radius))), budget=budget
    )
    rows = tuple(
        tuple(
            rational_coefficient(value, role="parallel_cylinder.fillet_coefficient")
            for value in row
        )
        for row in rows
    )
    reserve_fraction_work(rows, 1200, 120, 48, budget=budget)
    c0, c1, corner, rr = rows
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
    source_radii = tuple(Fraction(float(side.patch.radius)) for side in junction.sides)
    if not (
        d2 < (source_radii[0] + source_radii[1]) ** 2
        and d2 > (source_radii[0] - source_radii[1]) ** 2
    ):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: source circles do not intersect transversely"
        )
    rays = tuple(_sector_rays(side, junction, budget) for side in junction.sides)
    for orientation in (1, -1):
        rho = tuple(
            r - orientation * side.radial_orientation * rr[0]
            for r, side in zip(source_radii, junction.sides, strict=True)
        )
        if min(rho) <= 0:
            continue
        lam = (d2 + rho[0] ** 2 - rho[1] ** 2) / (2 * d2)
        q = rho[0] ** 2 / d2 - lam**2
        if q <= 0:
            continue
        base = (c0[0] + lam * delta[0], c0[1] + lam * delta[1])
        radical = (-branch * delta[1], branch * delta[0])
        center = np.asarray(
            [
                float(a) + float(b) * sqrt(float(q))
                for a, b in zip(base, radical, strict=True)
            ]
        )
        points = np.stack(
            [
                junction.centers[i]
                + float(source_radii[i] / rho[i]) * (center - junction.centers[i])
                for i in range(2)
            ]
        )
        angles: list[float | None] = []
        for i, side in enumerate(junction.sides):
            scaled_base = (
                (base[0] - rows[i][0]) * source_radii[i] / rho[i],
                (base[1] - rows[i][1]) * source_radii[i] / rho[i],
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
                _Blend(points, (angle0, angle1), center, radius, sweep, orientation)
            )
    if len(candidates) != 1:
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: offset branch has no unique nonvanishing retained-side certificate"
        )
    return candidates[0]


def _chamfer(junction: _Junction, setbacks: tuple[float, float]) -> _Blend:
    angles: list[float] = []
    points: list[np.ndarray] = []
    for side, setback in zip(junction.sides, setbacks, strict=True):
        exact_distance = Fraction(setback) / Fraction(float(side.patch.radius))
        distance = float(exact_distance)
        if (
            not 0
            < exact_distance
            < abs(Fraction(side.retained_angle) - Fraction(side.angle))
        ):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder chamfer: setback vanishes or consumes a source face"
            )
        angle = side.angle + (distance if side.retained_angle > side.angle else -distance)
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
    )


def _cap_pcurve(
    patch: PlanePatch, curve: CircleCurve | LineCurve
) -> CircleCurve | LineCurve:
    axes = np.stack((np.asarray(patch.first_axis), np.asarray(patch.second_axis)))
    origin = np.asarray(patch.origin)
    if isinstance(curve, CircleCurve):
        return CircleCurve(
            axes @ (np.asarray(curve.center) - origin),
            axes @ np.asarray(curve.first_axis),
            axes @ np.asarray(curve.second_axis),
            curve.radius,
        )
    return LineCurve(
        axes @ (np.asarray(curve.origin) - origin), axes @ np.asarray(curve.direction)
    )


class _StagedEdges(StrictModule):
    builder: Builder
    new_vertices: list[tuple[int, int]]
    cap_changes: dict[int, tuple[int, int, float]]
    edge_map: dict[int, int]
    vertical_edges: list[int]


class _StagedBlend(StrictModule):
    patch: CylinderPatch | PlanePatch
    cap_edges: list[int]
    cap_curves: list[CircleCurve | LineCurve]
    low: float
    high: float
    endpoints: tuple[int, int]


def _stage_edges(junction: _Junction, blend: _Blend, selected: int) -> _StagedEdges:
    geometry = junction.geometry
    builder = Builder()
    removed_vertices = set(junction.vertices)
    vertex_map = {
        i: builder.vertex(np.asarray(p))
        for i, p in enumerate(geometry.vertex_points)
        if i not in removed_vertices
    }
    new_vertices: list[tuple[int, int]] = []
    for point in blend.points:
        bottom = junction.origin + point[0] * junction.first + point[1] * junction.second
        new_vertices.append(
            (
                builder.vertex(bottom),
                builder.vertex(bottom + junction.height * junction.axis),
            )
        )
    cap_changes: dict[int, tuple[int, int, float]] = {}
    for i, side in enumerate(junction.sides):
        same = np.array_equal(np.asarray(side.patch.axis), junction.axis)
        for end in range(2):
            cap_changes[side.cap_edges[end if same else 1 - end]] = (
                i,
                end,
                blend.angles[i],
            )
    edge_map: dict[int, int] = {}
    for edge, curve_index in enumerate(geometry.edge_curves):
        if edge == selected:
            continue
        ends = geometry.edge_vertices[edge]
        first, last = map(float, np.asarray(geometry.edge_ranges[edge]))
        change = cap_changes.get(edge)
        vertices: list[int] = []
        for endpoint, vertex in enumerate(ends):
            if change is not None and vertex in removed_vertices:
                vertices.append(new_vertices[change[0]][change[1]])
                if endpoint == 0:
                    first = change[2]
                else:
                    last = change[2]
            else:
                vertices.append(vertex_map[vertex])
        edge_map[edge] = (
            builder.degenerate(vertices[0], first, last)
            if curve_index < 0
            else builder.edge(
                geometry.curves[curve_index], first, last, vertices[0], vertices[1]
            )
        )
    vertical_edges: list[int] = []
    for i, point in enumerate(blend.points):
        bottom = junction.origin + point[0] * junction.first + point[1] * junction.second
        vertical_edges.append(
            builder.edge(
                LineCurve(bottom, junction.axis), 0.0, junction.height, *new_vertices[i]
            )
        )
    return _StagedEdges(builder, new_vertices, cap_changes, edge_map, vertical_edges)


def _stage_caps(junction: _Junction, blend: _Blend, edges: _StagedEdges) -> _StagedBlend:
    builder, new_vertices = edges.builder, edges.new_vertices
    cap_edges: list[int] = []
    cap_curves: list[CircleCurve | LineCurve] = []
    if blend.radius > 0.0:
        first_angle = atan2(
            float(blend.points[0, 1] - blend.center[1]),
            float(blend.points[0, 0] - blend.center[0]),
        )
        low, high = sorted((first_angle, first_angle + blend.sweep))
        blend_origin = (
            junction.origin
            + blend.center[0] * junction.first
            + blend.center[1] * junction.second
        )
        blend_patch: CylinderPatch | PlanePatch = CylinderPatch(
            blend_origin, junction.first, junction.second, junction.axis, blend.radius
        )
        endpoints = (0, 1) if blend.sweep > 0.0 else (1, 0)
        for end in range(2):
            curve = CircleCurve(
                blend_origin + end * junction.height * junction.axis,
                junction.first,
                junction.second,
                blend.radius,
            )
            cap_curves.append(curve)
            cap_edges.append(
                builder.edge(
                    curve,
                    low,
                    high,
                    new_vertices[endpoints[0]][end],
                    new_vertices[endpoints[1]][end],
                )
            )
    else:
        low, high = 0.0, 1.0
        endpoints = (0, 1)
        bottom = np.asarray(builder.vertices[new_vertices[0][0]])
        direction = np.asarray(builder.vertices[new_vertices[1][0]]) - bottom
        blend_patch = PlanePatch(bottom, direction, junction.axis)
        for end in range(2):
            curve = LineCurve(bottom + end * junction.height * junction.axis, direction)
            cap_curves.append(curve)
            cap_edges.append(
                builder.edge(curve, low, high, new_vertices[0][end], new_vertices[1][end])
            )
    return _StagedBlend(blend_patch, cap_edges, cap_curves, low, high, endpoints)


def _vertical_use(
    junction: _Junction,
    blend: _Blend,
    side_index: int,
    side: _Side,
    sense: int,
    pcurve: AbstractCurve | IntersectionPCurve,
) -> tuple[int, LineCurve]:
    if not isinstance(pcurve, LineCurve):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: nonaffine vertical parameter"
        )
    same = np.array_equal(np.asarray(side.patch.axis), junction.axis)
    height = side.heights[0] if same else side.heights[1]
    derivative = 1.0 if same else -1.0
    sense *= 1 if float(pcurve.direction[1]) * derivative > 0.0 else -1
    return sense, LineCurve((blend.angles[side_index], height), (0.0, derivative))


def _stage_source_loop(
    junction: _Junction,
    blend: _Blend,
    model: BRepModel,
    selected: int,
    edges: _StagedEdges,
    boundary: _StagedBlend,
    face: int,
    loop: tuple[int, ...],
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
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
            sense, pcurve = _vertical_use(junction, blend, i, side, sense, pcurve)
            target = edges.vertical_edges[i]
        else:
            target = edges.edge_map[edge]
        staged.append(
            builder.coedge(
                target, sense, pcurve, preparations=preparations, budget=budget
            )
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
        staged.append(
            builder.coedge(
                boundary.cap_edges[end],
                cap_sense,
                _cap_pcurve(patch, boundary.cap_curves[end]),
                preparations=preparations,
                budget=budget,
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
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> None:
    builder, geometry = edges.builder, junction.geometry
    low, high = boundary.low, boundary.high
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
                preparations,
                budget,
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
            box[:, 0] = sorted((blend.angles[i], side.retained_angle))
        elif face in cap_indices:
            end = cap_indices[face]
            cap_patch = model.patches[face]
            if not isinstance(cap_patch, PlanePatch):
                raise BRepBlendUnsupported(
                    "unsupported parallel-cylinder blend: nonplanar cap"
                )
            pc = _cap_pcurve(cap_patch, boundary.cap_curves[end])
            bound = pc.bounding_box(low, high, preparations=preparations, budget=budget)
            box[0] = np.minimum(box[0], bound[0])
            box[1] = np.maximum(box[1], bound[1])
        builder.face(
            model.patches[face],
            box,
            staged_loops,
            model.physical_tags[face],
            None,
            orientation=int(float(model.orientation[face])),
        )


def _stage_blend_face(
    junction: _Junction,
    blend: _Blend,
    edges: _StagedEdges,
    boundary: _StagedBlend,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> None:
    builder, vertical_edges = edges.builder, edges.vertical_edges
    cap_edges, endpoints = boundary.cap_edges, boundary.endpoints
    low, high, blend_patch = boundary.low, boundary.high, boundary.patch
    blend_loop = [
        builder.coedge(
            cap_edges[0],
            1,
            LineCurve((0.0, 0.0), (1.0, 0.0)),
            preparations=preparations,
            budget=budget,
        ),
        builder.coedge(
            vertical_edges[endpoints[1]],
            1,
            LineCurve((high, 0.0), (0.0, 1.0)),
            preparations=preparations,
            budget=budget,
        ),
        builder.coedge(
            cap_edges[1],
            -1,
            LineCurve((0.0, junction.height), (1.0, 0.0)),
            preparations=preparations,
            budget=budget,
        ),
        builder.coedge(
            vertical_edges[endpoints[0]],
            -1,
            LineCurve((low, 0.0), (0.0, 1.0)),
            preparations=preparations,
            budget=budget,
        ),
    ]
    if blend.radius > 0.0:
        orientation = blend.orientation
    else:
        side0 = junction.sides[0]
        old_normal = side0.radial_orientation * (blend.points[0] - junction.centers[0])
        tangent = blend.points[1] - blend.points[0]
        normal = np.asarray((tangent[1], -tangent[0]))
        orientation = 1 if float(normal @ old_normal) > 0.0 else -1
    builder.face(
        blend_patch,
        np.asarray(((low, 0.0), (high, junction.height))),
        [blend_loop],
        "cylinder" if blend.radius > 0.0 else "plane",
        None,
        orientation=orientation,
    )


def _stage(
    junction: _Junction,
    blend: _Blend,
    model: BRepModel,
    selected: int,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> Builder:
    edges = _stage_edges(junction, blend, selected)
    boundary = _stage_caps(junction, blend, edges)
    _stage_faces(junction, blend, model, selected, edges, boundary, preparations, budget)
    _stage_blend_face(junction, blend, edges, boundary, preparations, budget)
    return edges.builder


def blend_parallel_cylinders(
    model: BRepModel,
    edge_index: int,
    radius: float | None,
    setbacks: tuple[float, float] | None,
    *,
    tessellation: BRepTessellationPolicy | None,
    source_id: str,
    preparations: TrimRequestPreparations,
    budget: CoordinateEnclosureBudget,
) -> BRepModel:
    """Replace one parallel-cylinder junction by a certified analytic blend.

    Setbacks are intrinsic arc lengths on the two cylindrical faces, in their
    topology incident-face order. Radius differentiation uses
    :class:`ParallelCylinderRadiusMap` to realize the correlated angular trims,
    cap loops, and blend coefficients in one fixed-topology source.
    """
    if (radius is None) == (setbacks is None):
        raise ValueError(
            f"blend_parallel_cylinders.distances must specify one radius or setback pair; got radius={radius!r}, setbacks={setbacks!r}"
        )
    values = (radius,) if radius is not None else setbacks
    if values is None or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(
            f"blend_parallel_cylinders.distances must be positive and finite; got {values!r}"
        )
    junction = _junction(model, edge_index, budget)
    if radius is not None:
        blend = _fillet(junction, radius, budget)
    elif setbacks is not None:
        blend = _chamfer(junction, setbacks)
    else:
        raise ValueError(
            "blend_parallel_cylinders.distances must contain a specification; got none"
        )
    builder = _stage(junction, blend, model, edge_index, preparations, budget)
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
    return finish(
        builder,
        solid=bool(geometry.solid_shells),
        coordinate_contract=model.coordinate_contract,
        source_id=source_id,
        tessellation=tessellation,
        occurrences=geometry.occurrences,
        shells=(shell_faces, shell_orientations, geometry.solid_shells),
        preparations=preparations,
        budget=budget,
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
    *,
    budget: CoordinateEnclosureBudget,
) -> np.ndarray:
    """Enclose the represented boundary normal angles, in source-face order.

    The algebraic construction has ``rho_i = R_i - s o_i r`` and
    ``P_i = C_i + R_i (F-C_i)/rho_i``. Consequently
    ``s (P_i-F)/r = o_i (P_i-C_i)/R_i``: the two outward
    normals agree identically along each axial boundary. The returned intervals
    additionally account for finite coefficient and angular realization through
    the native interval extension, not a generic epsilon allowance.
    """
    junction = _junction(model, edge_index, budget)
    geometry = result.geometry
    blend_face = len(result.patches) - 1
    patch = result.patches[blend_face]
    if geometry is None or not isinstance(patch, CylinderPatch):
        raise ValueError(
            f"parallel_cylinder_g1_bounds.result must contain a published cylindrical fillet; got patch={type(patch).__name__}, geometry_present={geometry is not None}"
        )
    _exact_frame(patch, budget)
    budget.reserve(1200, 16384)
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
            if not isinstance(pcurve, LineCurve) or float(pcurve.direction[0]) != 0.0:
                raise ValueError(
                    f"parallel_cylinder_g1_bounds.pcurve must be an angular line isoline; got {type(pcurve).__name__}"
                )
            angles.append(float(pcurve.origin[0]))
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
    rows = rational_points(magnitude, budget=budget)
    rows = tuple(
        tuple(
            rational_coefficient(value, role="parallel_cylinder.normal_bound")
            for value in row
        )
        for row in rows
    )
    reserve_fraction_work(rows, 12, 2, 8, budget=budget)
    norms = np.asarray(tuple(fraction_norm_upper(row) for row in rows))
    if np.any(~np.isfinite(norms)) or np.any(norms >= 2.0):
        raise BRepBlendUnsupported(
            "unsupported parallel-cylinder blend: normal-angle enclosure is unresolved"
        )
    angles = 2.0 * np.arcsin(np.nextafter(norms / 2.0, np.inf))
    for _ in range(4):
        angles = np.nextafter(angles, np.inf)
    return np.stack((np.zeros(2, dtype=np.float64), angles), axis=1)


class _CornerGroup(StrictModule):
    coedge: int = eqx.field(static=True)
    rows: tuple[int, ...] = eqx.field(static=True)
    fractions: Array


def _corner_groups(model: BRepModel) -> tuple[_CornerGroup, ...]:
    geometry = model.require_geometry()
    mesh = np.asarray(model.mesh_faces).reshape(-1)
    faces = np.repeat(np.asarray(model.triangle_face_ids), 3)
    dimensions = np.asarray(model.mesh_vertex_source_dimensions)
    indices = np.asarray(model.mesh_vertex_source_indices)
    parameters = np.asarray(model.mesh_vertex_parameters)
    ranges = np.asarray(geometry.edge_ranges)
    groups: list[_CornerGroup] = []
    for face, loops in enumerate(geometry.face_loops):
        assigned: set[int] = set()
        for loop in loops:
            for coedge in loop:
                edge = geometry.coedge_edges[coedge]
                rows: list[int] = []
                fractions: list[float] = []
                for row in np.flatnonzero(faces == face):
                    row_index = int(row)
                    if row_index in assigned:
                        continue
                    vertex = int(mesh[row])
                    if dimensions[vertex] == 1 and indices[vertex] == edge:
                        t = float(parameters[vertex, 0])
                    elif (
                        dimensions[vertex] == 0
                        and indices[vertex] in geometry.edge_vertices[edge]
                    ):
                        endpoint = geometry.edge_vertices[edge].index(
                            int(indices[vertex])
                        )
                        t = float(ranges[edge, endpoint])
                    else:
                        continue
                    first, last = ranges[edge]
                    rows.append(row_index)
                    fractions.append((t - float(first)) / float(last - first))
                    assigned.add(row_index)
                if rows:
                    groups.append(
                        _CornerGroup(
                            coedge, tuple(rows), jnp.asarray(fractions, dtype=jnp.float64)
                        )
                    )
    return tuple(groups)


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
        self.source = source
        self.reference = result
        self.edge_index = edge_index
        self.initial_radius = radius
        self.parameter_id = parameter_id

    def compile(
        self, context: ParameterCollector, model: BRepModel, /
    ) -> AbstractBoundBRepParameterMap:
        if model.model_id != self.reference.model_id:
            raise ValueError(
                f"ParallelCylinderRadiusMap.model must match the reference model; got {model.model_id!r}"
            )
        if context.coordinate_unit != model.coordinate_contract.length_unit:
            raise ValueError(
                f"ParallelCylinderRadiusMap.coordinate_unit must match the source coordinate unit; got {context.coordinate_unit.symbol!r}"
            )
        with coordinate_source_request((self.source, model)) as budget:
            junction = _junction(self.source, self.edge_index, budget)
            blend = _fillet(junction, self.initial_radius, budget)
            radius_bounds = _radius_epoch(junction, blend, self.initial_radius, budget)
            budget.reserve(
                max(model.mesh_faces.size, 1)
                * max(len(model.require_geometry().coedge_edges), 1)
                * 16,
                max(model.mesh_faces.size, 1) * 256,
            )
            groups = _corner_groups(model)
        radius = context.bind(
            self.parameter_id,
            jnp.asarray(self.initial_radius, dtype=jnp.float64),
            role="length",
            unit=context.coordinate_unit,
            physical_scale=self.initial_radius,
            bounds=radius_bounds,
        )
        boundary_edges = tuple(
            len(junction.geometry.edge_curves) - 1 + i for i in range(2)
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
                            - blend.angles[i]
                        )
                    )
                )
                for i, side in enumerate(junction.sides)
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
            groups=groups,
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
    groups: tuple[_CornerGroup, ...]

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
        reference_geometry = self.reference.require_geometry()
        patch = self.reference.patches[-1]
        if not isinstance(patch, CylinderPatch):
            raise ValueError(
                f"ParallelCylinderRadiusMap.patch must be cylindrical; got {type(patch).__name__}"
            )
        patches = (
            *self.reference.patches[:-1],
            eqx.tree_at(
                lambda value: (value.origin, value.radius),
                patch,
                (spatial_center, radius),
            ),
        )
        curves = list(reference_geometry.curves)
        vertices = reference_geometry.vertex_points
        pcurves = list(reference_geometry.pcurves)
        ranges = reference_geometry.edge_ranges
        for i, edge in enumerate(self.boundary_edges):
            curve_index = reference_geometry.edge_curves[edge]
            curve = curves[curve_index]
            if not isinstance(curve, LineCurve):
                raise ValueError(
                    f"ParallelCylinderRadiusMap.boundary must have a native line carrier; got {type(curve).__name__}"
                )
            curves[curve_index] = eqx.tree_at(lambda value: value.origin, curve, world[i])
            for end, vertex in enumerate(reference_geometry.edge_vertices[edge]):
                vertices = vertices.at[vertex].set(
                    world[i] + end * self.height * self.axis
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
            curves[curve_index] = eqx.tree_at(
                lambda value: (value.center, value.radius),
                curve,
                (spatial_center + end * self.height * self.axis, radius),
            )
            ranges = ranges.at[edge].set(jnp.stack((low, high)))
        pcurves = self._realize_pcurves(
            tuple(pcurves), angles, betas, spatial_center, radius
        )
        geometry = eqx.tree_at(
            lambda value: (
                value.vertex_points,
                value.curves,
                value.pcurves,
                value.edge_ranges,
            ),
            reference_geometry,
            (vertices, tuple(curves), pcurves, ranges),
        )
        bounds = self._realize_bounds(angles, betas, spatial_center, radius)
        triangles = self._realize_triangles(geometry, bounds)
        return BRepParameterGeometry(
            patches=patches,
            geometry=geometry,
            parameter_bounds=bounds,
            triangle_parameters=triangles,
        )

    def _realize_pcurves(
        self,
        reference: tuple[AbstractCurve | IntersectionPCurve, ...],
        angles: Array,
        betas: Array,
        center: Array,
        radius: Array,
    ) -> tuple[AbstractCurve | IntersectionPCurve, ...]:
        geometry = self.reference.require_geometry()
        pcurves = list(reference)
        for face, loops in enumerate(geometry.face_loops):
            for loop in loops:
                for coedge in loop:
                    edge = geometry.coedge_edges[coedge]
                    pc = pcurves[coedge]
                    if edge in self.boundary_edges:
                        side = self.boundary_edges.index(edge)
                        if not isinstance(pc, LineCurve):
                            raise ValueError(
                                f"ParallelCylinderRadiusMap.pcurve must be a native line on the axial boundary; got {type(pc).__name__}"
                            )
                        angle = (
                            angles[side]
                            if face != len(self.reference.patches) - 1
                            else betas[side]
                        )
                        pcurves[coedge] = eqx.tree_at(
                            lambda value: value.origin, pc, pc.origin.at[0].set(angle)
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
                        pcurves[coedge] = eqx.tree_at(
                            lambda value: (value.center, value.radius),
                            pc,
                            (pc_center, radius),
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

    def _realize_triangles(self, geometry: BRepGeometry, bounds: Array) -> Array:
        faces = self.reference.triangle_face_ids
        original = self.reference.parameter_bounds[faces]
        realized = bounds[faces]
        normalized = (self.reference.triangle_parameters - original[:, None, 0]) / (
            original[:, None, 1] - original[:, None, 0]
        )
        parameters = realized[:, None, 0] + normalized * (
            realized[:, None, 1] - realized[:, None, 0]
        )
        parameters = parameters.reshape(-1, 2)
        for group in self.groups:
            edge = geometry.coedge_edges[group.coedge]
            first, last = geometry.edge_ranges[edge, 0], geometry.edge_ranges[edge, 1]
            t = first + group.fractions * (last - first)
            uv = geometry.pcurves[group.coedge].evaluate(t)
            parameters = parameters.at[jnp.asarray(group.rows, dtype=jnp.int32)].set(uv)
        return parameters.reshape(self.reference.triangle_parameters.shape)


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


def _interval_sector_sign(
    lower: np.ndarray,
    upper: np.ndarray,
    rays: tuple[tuple[Fraction, ...], ...],
    direction: int,
    reverse: bool,
) -> tuple[float, float]:
    ray_values = np.asarray([[float(value) for value in ray] for ray in rays])
    ray_lower, ray_upper = np.min(ray_values, axis=0), np.max(ray_values, axis=0)
    first = interval_multiply(
        (np.asarray(ray_lower[0]), np.asarray(ray_upper[0])),
        (np.asarray(lower[1]), np.asarray(upper[1])),
    )
    second = interval_multiply(
        (np.asarray(ray_lower[1]), np.asarray(ray_upper[1])),
        (np.asarray(lower[0]), np.asarray(upper[0])),
    )
    low, high = interval_subtract(first, second)
    multiplier = -direction if reverse else direction
    return (float(low), float(high)) if multiplier > 0 else (-float(high), -float(low))


def _radius_epoch(
    junction: _Junction,
    blend: _Blend,
    radius: float,
    budget: CoordinateEnclosureBudget,
) -> tuple[float, float]:
    """Admit an open radius epoch with interval-proven branch and trim invariants."""
    budget.reserve(2048, 16384)
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
    rays = tuple(_sector_rays(side, junction, budget) for side in junction.sides)
    width = radius / 2.0
    while width > 0.0:
        low, high = parameters.copy(), parameters.copy()
        low[0], high[0] = radius - width, radius + width
        if low[0] == radius or high[0] == radius or not np.isfinite(high[0]):
            raise BRepBlendUnsupported(
                "unsupported parallel-cylinder radius map: no representable open certified epoch"
            )
        budget.reserve(2048, 16384)
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
    *,
    budget: CoordinateEnclosureBudget,
) -> CertifiedParameterRange:
    """Expose the existing interval-proved open contact/branch epoch."""
    try:
        junction = _junction(source, edge_index, budget)
        blend = _fillet(junction, radius, budget)
        lower, upper = _radius_epoch(junction, blend, radius, budget)
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
