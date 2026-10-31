#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact source-curve / support-patch correspondence for native trims."""

from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import cast, TYPE_CHECKING

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._interpolation import (
    rational_bspline_curve_spans,
    rational_bspline_surface_spans,
)
from ..._strict import StrictModule
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
    RadicalElement,
    rational_coefficient,
)
from ...linalg._exact import determinant_exact, solve_exact
from .._rational import rational_vector
from ._carrier_pieces import carrier_payload
from ._intersection_curve import (
    exact_curve_parameter_point,
    IntersectionCurve,
    IntersectionPCurve,
    SurfaceRegion,
)
from ._patches import (
    _combine_conic_parameter_roots,
    AbstractCurve,
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    ConicRootLinePCurve,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlaneCylinderSectionPCurve,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from ._placed import PlacedCurve, PlacedSurface
from ._source_correspondence import (
    _affine_curve_coefficients,
    _harmonic_coefficients,
    _plane_harmonics,
)


if TYPE_CHECKING:
    from ._intersection import ConicParameterEndpoint


def _same_source_definition(first: StrictModule, second: StrictModule, /) -> bool:
    if type(first) is not type(second):
        return False
    first_identity = getattr(first, "identity_payload", None)
    second_identity = getattr(second, "identity_payload", None)
    if callable(first_identity) and callable(second_identity):
        return canonical_fingerprint(first_identity()) == canonical_fingerprint(
            second_identity()
        )
    return canonical_fingerprint(carrier_payload(first)) == canonical_fingerprint(
        carrier_payload(second)
    )


def _same_patch(first: StrictModule, second: StrictModule, /) -> bool:
    return _same_source_definition(first, second)


def _same_pose_values(curve: PlacedCurve, surface: PlacedSurface, /) -> bool:
    return np.array_equal(
        np.asarray(curve.rotation), np.asarray(surface.rotation)
    ) and np.array_equal(np.asarray(curve.translation), np.asarray(surface.translation))


type _AffineSurfaceFrame = tuple[
    tuple[ExactScalar, ...], tuple[ExactScalar, ...], tuple[ExactScalar, ...]
]


def _affine_surface_frame(patch: AbstractSurfacePatch, /) -> _AffineSurfaceFrame | None:
    from .._source_interpolation import exact_surface_coordinate_expressions

    expressions = exact_surface_coordinate_expressions(patch)
    if expressions is None or len(expressions) != 3:
        return None
    if any(
        sum(power) > 1 and coefficient != 0
        for expression in expressions
        for power, coefficient in expression.items()
    ):
        return None
    return (
        tuple(expression.get((0, 0), Fraction(0)) for expression in expressions),
        tuple(expression.get((1, 0), Fraction(0)) for expression in expressions),
        tuple(expression.get((0, 1), Fraction(0)) for expression in expressions),
    )


def _exact_vertex_root_source(
    vertex_root: object | None, /
) -> tuple[
    tuple[ExactScalar, ExactScalar, ExactScalar] | None, ConicParameterRoot | None
]:
    from ._root_bindings import BRepConicVertex, BRepPlacedVertex, BRepVertexRoot

    if not isinstance(vertex_root, BRepVertexRoot):
        return None, None
    primary = vertex_root.primary
    if isinstance(primary, BRepConicVertex):
        return primary.point_exact, primary.root
    if isinstance(primary, BRepPlacedVertex):
        if primary.source_root is None:
            source = tuple(
                Fraction.from_float(float(value))
                for value in np.asarray(primary.source_point, dtype=np.float64)
            )
            return primary.affine.apply(source), None
        point, conic_root = _exact_vertex_root_source(primary.source_root)
        if point is None:
            return None, None
        return primary.affine.apply(point), conic_root
    return None, None


def _exact_constant_chart_point(
    curve: AbstractCurve, first: float, last: float, /
) -> tuple[ExactScalar, ExactScalar] | None:
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return None
    affine = _affine_curve_coefficients(curve, (Fraction(0), Fraction(0)))
    if affine is not None:
        origin, direction = affine
        if all(value == 0 for value in direction):
            return origin[0], origin[1]
    if isinstance(curve, BSplineCurve):
        if any(
            point != curve.exact_control_points[0]
            for point in curve.exact_control_points[1:]
        ):
            return None
        point = curve.exact_control_points[0]
        if len(point) == 2:
            return point[0], point[1]
    return None


def _exact_brep_pole_collapse(
    curve: AbstractCurve | None,
    source_patch: AbstractSurfacePatch,
    pcurve: AbstractCurve,
    first: float,
    last: float,
    corner_exact: tuple[ExactScalar, ExactScalar, ExactScalar] | None,
    corner_root: object | None,
    endpoint_roots: tuple[object | None, object | None] = (None, None),
    /,
) -> bool:
    """Prove one degenerate coedge maps over its full source interval to a vertex."""
    if (
        curve is not None
        or not isinstance(source_patch, AbstractSurfacePatch)
        or not isinstance(pcurve, AbstractCurve)
        or pcurve.ambient_dimension != 2
        or not np.isfinite(first)
        or not np.isfinite(last)
        or not first < last
        or not isinstance(endpoint_roots, tuple)
        or len(endpoint_roots) != 2
    ):
        return False
    root_point, conic_root = _exact_vertex_root_source(corner_root)
    if corner_exact is not None and (
        len(corner_exact) != 3
        or any(
            not isinstance(value, (Fraction, RadicalElement)) for value in corner_exact
        )
    ):
        return False
    point = corner_exact if corner_exact is not None else root_point
    if point is None or (root_point is not None and point != root_point):
        return False

    chart_point = _exact_constant_chart_point(pcurve, first, last)
    if chart_point is not None:
        frame = _affine_surface_frame(source_patch)
        if frame is None:
            return False
        origin, first_axis, second_axis = frame
        mapped = tuple(
            origin[axis]
            + chart_point[0] * first_axis[axis]
            + chart_point[1] * second_axis[axis]
            for axis in range(3)
        )
        return mapped == point

    if not isinstance(source_patch, RevolutionSurface):
        return False
    if conic_root is not None and (
        not _same_source_definition(conic_root.conic, source_patch.curve)
        or conic_root.exact_point() != point
    ):
        return False
    from ._intersection import NativePeriodEndpoint

    first_root, last_root = endpoint_roots
    if not isinstance(first_root, NativePeriodEndpoint) or not isinstance(
        last_root, NativePeriodEndpoint
    ):
        return False
    if first_root.patch is None or last_root.patch is None:
        return False
    if (
        first_root.axis != 0
        or last_root.axis != 0
        or first_root.turns != 0
        or last_root.turns != 1
        or first_root.rational != last_root.rational
        or not _same_source_definition(first_root.patch, source_patch)
        or not _same_source_definition(last_root.patch, source_patch)
        or not _same_source_definition(first_root.curve, pcurve)
        or not _same_source_definition(last_root.curve, pcurve)
    ):
        return False
    period = source_patch.periods[0]
    if period is None or last - first != period:
        return False
    affine = _affine_curve_coefficients(pcurve, (Fraction(0), Fraction(0)))
    if affine is None:
        return False
    origin, direction = affine
    chart_start = tuple(
        origin[axis] + Fraction.from_float(float(first)) * direction[axis]
        for axis in range(2)
    )
    if direction != (Fraction(1), Fraction(0)) or chart_start[0] != first_root.rational:
        return False
    if conic_root is not None:
        if chart_start[1] != Fraction.from_float(conic_root.parameter):
            return False
    else:
        profile = _affine_curve_coefficients(
            source_patch.curve, (Fraction(0), Fraction(0), Fraction(0))
        )
        if profile is None:
            return False
        profile_origin, profile_direction = profile
        profile_point = tuple(
            profile_origin[axis] + chart_start[1] * profile_direction[axis]
            for axis in range(3)
        )
        if profile_point != point:
            return False
    axis_origin = rational_vector(np.asarray(source_patch.axis_origin, dtype=np.float64))
    axis_direction = rational_vector(
        np.asarray(source_patch.axis_direction, dtype=np.float64)
    )
    relative = tuple(value - axis_origin[axis] for axis, value in enumerate(point))
    cross = (
        relative[1] * axis_direction[2] - relative[2] * axis_direction[1],
        relative[2] * axis_direction[0] - relative[0] * axis_direction[2],
        relative[0] * axis_direction[1] - relative[1] * axis_direction[0],
    )
    return all(value == 0 for value in cross)


def _plane_parameter_vector(
    frame: _AffineSurfaceFrame, vector: tuple[ExactScalar, ...], /
) -> tuple[ExactScalar, ExactScalar] | None:
    _, first, second = frame

    def dot(
        left: tuple[ExactScalar, ...], right: tuple[ExactScalar, ...], /
    ) -> ExactScalar:
        return sum((a * b for a, b in zip(left, right, strict=True)), Fraction(0))

    g00, g01, g11 = dot(first, first), dot(first, second), dot(second, second)
    gram = ((g00, g01), (g01, g11))
    if determinant_exact(gram) == 0:
        return None
    actions = solve_exact(gram, ((dot(first, vector),), (dot(second, vector),)))
    return actions[0][0], actions[1][0]


def _same_plane_line_lift(
    curve: AbstractCurve, pcurve: AbstractCurve, frame: _AffineSurfaceFrame, /
) -> bool:
    if curve.ambient_dimension != 3 or pcurve.ambient_dimension != 2:
        return False
    origin, first, second = frame
    source = _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * pcurve.ambient_dimension)
    if source is None or chart is None:
        return False
    source_origin, source_direction = source
    uv_origin, uv_direction = chart
    image_origin = tuple(
        origin[i] + first[i] * uv_origin[0] + second[i] * uv_origin[1] for i in range(3)
    )
    image_direction = tuple(
        first[i] * uv_direction[0] + second[i] * uv_direction[1] for i in range(3)
    )
    return source_origin == image_origin and source_direction == image_direction


def _affine_pcurve_carrier(
    source: AbstractCurve,
    origin: tuple[ExactScalar, ExactScalar],
    direction: tuple[ExactScalar, ExactScalar],
    /,
) -> AbstractCurve:
    if isinstance(source, ExactLineCurve):
        try:
            first = tuple(
                rational_coefficient(value, role="p-curve endpoint") for value in origin
            )
            last = tuple(
                rational_coefficient(value + slope, role="p-curve endpoint")
                for value, slope in zip(origin, direction, strict=True)
            )
        except ValueError:
            # An algebraic parameter frame needs the general exact affine carrier.
            pass
        else:
            return ExactLineCurve(first, last)
    return LineCurve(
        tuple(float(value) for value in origin),
        tuple(float(value) for value in direction),
        origin_exact=origin,
        direction_exact=direction,
    )


def native_edge_pcurve(
    curve: AbstractCurve,
    patch: AbstractSurfacePatch,
    /,
) -> AbstractCurve | None:
    """Recover a whole source-map p-curve without inventing finite patch bounds."""
    if isinstance(curve, IntersectionCurve):
        if _same_patch(curve.first.patch, patch):
            return curve.p_curve("first")
        if _same_patch(curve.second.patch, patch):
            return curve.p_curve("second")
        return None
    if isinstance(curve, PlacedCurve) and isinstance(patch, PlacedSurface):
        if not _same_pose_values(curve, patch):
            return None
        return native_edge_pcurve(curve.definition, patch.definition)
    if isinstance(curve, SurfaceIsoparametricCurve) and _same_patch(curve.surface, patch):
        if curve.fixed_axis == 0:
            return LineCurve((curve.fixed_value, 0.0), (0.0, 1.0))
        return LineCurve((0.0, curve.fixed_value), (1.0, 0.0))
    if isinstance(curve, SurfaceMappedCurve) and _same_patch(curve.surface, patch):
        return curve.pcurve
    if isinstance(curve, CircleCurve) and isinstance(patch, CylinderPatch):
        return _cylinder_section_pcurve(curve, patch)
    if isinstance(patch, RevolutionSurface) and isinstance(curve, CircleCurve):
        return _revolution_circle_pcurve_candidate(curve, patch)
    frame = _affine_surface_frame(patch)
    if frame is not None and curve.ambient_dimension == 3:
        plane_origin = frame[0]
        coefficients = _affine_curve_coefficients(curve, (Fraction(0),) * 3)
        if coefficients is None:
            return None
        uv_origin = _plane_parameter_vector(
            frame, tuple(coefficients[0][i] - plane_origin[i] for i in range(3))
        )
        uv_direction = _plane_parameter_vector(frame, coefficients[1])
        if uv_origin is None or uv_direction is None:
            return None
        candidate = _affine_pcurve_carrier(curve, uv_origin, uv_direction)
        return candidate if _same_plane_line_lift(curve, candidate, frame) else None
    return None


def _cylinder_section_pcurve(
    curve: CircleCurve, patch: CylinderPatch, /
) -> AbstractCurve | None:
    """The exact height isoline of a cross-section circle that shares the cylinder phase.

    The circle must have the cylinder's exact radial harmonics and a center on
    the cylinder axis at an exact height ``h``; its p-curve is then ``(t, h)``.
    A reversed or phase-shifted circle has no affine exact chart map here.
    """
    if curve.ambient_dimension != 3:
        return None
    center, cosine, sine = _harmonic_coefficients(curve)
    radius = patch.radius_exact
    if cosine != tuple(
        radius * value for value in patch.first_axis_exact
    ) or sine != tuple(radius * value for value in patch.second_axis_exact):
        return None
    offset = tuple(
        value - origin for value, origin in zip(center, patch.origin_exact, strict=True)
    )
    pivot = next((value for value in patch.axis_exact if value != 0), None)
    if pivot is None:
        return None
    pivot_offset = offset[patch.axis_exact.index(pivot)]
    height = (
        pivot_offset * pivot.reciprocal()
        if isinstance(pivot, RadicalElement)
        else pivot_offset / pivot
    )
    if tuple(height * value for value in patch.axis_exact) != offset:
        return None
    return LineCurve(
        (0.0, float(height)),
        (1.0, 0.0),
        origin_exact=(Fraction(0), height),
        direction_exact=(Fraction(1), Fraction(0)),
    )


def _revolution_circle_pcurve_candidate(
    curve: CircleCurve, patch: RevolutionSurface, /
) -> AbstractCurve | None:
    """Propose the profile-level ring ``(t, v0)`` of a circle on a line revolution.

    The circle's start point must lie on the line profile at an exactly
    representable ``v0``; ``exact_source_lift`` then proves the whole lift.
    """
    if isinstance(patch.curve, LineCurve):
        origin, direction = patch.curve.origin_exact, patch.curve.direction_exact
    elif isinstance(patch.curve, ExactLineCurve):
        origin = patch.curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(
                patch.curve.first_exact, patch.curve.last_exact, strict=True
            )
        )
    else:
        return None
    start = tuple(
        center + curve.radius_exact * axis
        for center, axis in zip(curve.center_exact, curve.first_axis_exact, strict=True)
    )
    pivot = next((index for index, value in enumerate(direction) if value != 0), None)
    if pivot is None:
        return None
    parameter = (start[pivot] - origin[pivot]) / direction[pivot]
    if not isinstance(parameter, Fraction) or any(
        origin[index] + parameter * direction[index] != start[index] for index in range(3)
    ):
        return None
    value = float(parameter)
    if Fraction(value) != parameter:
        return None
    return LineCurve((0.0, value), (1.0, 0.0))


def _cylinder_chart_line(
    curve: AbstractCurve, /
) -> tuple[CylinderPatch, tuple[ExactScalar, ...], tuple[ExactScalar, ...]] | None:
    if isinstance(curve, SurfaceIsoparametricCurve):
        if curve.fixed_axis != 0:
            return None
        source = curve.surface
        if not isinstance(source, CylinderPatch):
            return None
        return (
            source,
            (Fraction.from_float(curve.fixed_value), Fraction(0)),
            (Fraction(0), Fraction(1)),
        )
    if isinstance(curve, SurfaceMappedCurve) and isinstance(curve.surface, CylinderPatch):
        coefficients = _affine_curve_coefficients(
            curve.pcurve, (Fraction(0),) * curve.pcurve.ambient_dimension
        )
        if coefficients is None or len(coefficients[0]) != 2:
            return None
        return curve.surface, coefficients[0], coefficients[1]
    return None


def _same_cylinder_chart_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    surface: CylinderPatch,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove two cylindrical chart lines realize one identical affine curve."""
    source_data = _cylinder_chart_line(curve)
    target_data = _affine_curve_coefficients(
        pcurve, (Fraction(0),) * pcurve.ambient_dimension
    )
    if source_data is None or target_data is None:
        return False
    source, source_origin, source_direction = source_data
    target_origin, target_direction = target_data
    for candidate in (curve, pcurve):
        domain = candidate.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            return False
    if source_direction[0] != 0 or target_direction[0] != 0:
        return False
    source_axes = (source.first_axis_exact, source.second_axis_exact)
    target_axes = (surface.first_axis_exact, surface.second_axis_exact)
    source_axis = source.axis_exact
    target_axis = surface.axis_exact
    source_offset = source.origin_exact
    target_offset = surface.origin_exact
    source_radius = source.radius_exact
    target_radius = surface.radius_exact
    return (
        source_origin[0] == target_origin[0]
        and source_axes == target_axes
        and source_radius == target_radius
        and tuple(
            source_offset[index] + source_origin[1] * source_axis[index]
            for index in range(3)
        )
        == tuple(
            target_offset[index] + target_origin[1] * target_axis[index]
            for index in range(3)
        )
        and tuple(source_direction[1] * source_axis[index] for index in range(3))
        == tuple(target_direction[1] * target_axis[index] for index in range(3))
    )


def _same_mapped_cylinder_plane_lift(
    curve: SurfaceMappedCurve,
    pcurve: AbstractCurve,
    plane: PlanePatch,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove an exact fixed-angle cylinder generator maps to its plane line."""
    data = _cylinder_chart_line(curve)
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * pcurve.ambient_dimension)
    if data is None or chart is None:
        return False
    cylinder, uv_origin, uv_direction = data
    if uv_origin[0] != 0 or uv_direction[0] != 0:
        return False
    source_domain = curve.parameter_domain
    if source_domain is not None and (
        first < source_domain[0] or last > source_domain[1]
    ):
        return False
    chart_domain = pcurve.parameter_domain
    if chart_domain is not None and (first < chart_domain[0] or last > chart_domain[1]):
        return False
    origin = cylinder.origin_exact
    first_axis = cylinder.first_axis_exact
    axis = cylinder.axis_exact
    radius = cylinder.radius_exact
    curve_origin = tuple(
        origin[index] + radius * first_axis[index] + uv_origin[1] * axis[index]
        for index in range(3)
    )
    curve_direction = tuple(uv_direction[1] * value for value in axis)
    plane_origin = rational_vector(np.asarray(plane.origin))
    plane_first = rational_vector(np.asarray(plane.first_axis))
    plane_second = rational_vector(np.asarray(plane.second_axis))
    chart_origin, chart_direction = chart
    lifted_origin = tuple(
        plane_origin[index]
        + chart_origin[0] * plane_first[index]
        + chart_origin[1] * plane_second[index]
        for index in range(3)
    )
    lifted_direction = tuple(
        chart_direction[0] * plane_first[index] + chart_direction[1] * plane_second[index]
        for index in range(3)
    )
    return curve_origin == lifted_origin and curve_direction == lifted_direction


def _same_cylinder_generator_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: CylinderPatch,
    /,
) -> bool:
    """Prove an axial line is the zero-angle generator of its exact cylinder."""
    source = _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * pcurve.ambient_dimension)
    if source is None or chart is None or len(source[0]) != 3 or len(chart[0]) != 2:
        return False
    chart_origin, chart_direction = chart
    if chart_origin[0] != 0 or chart_direction[0] != 0:
        return False
    origin = patch.origin_exact
    first_axis = patch.first_axis_exact
    axis = patch.axis_exact
    radius = patch.radius_exact
    expected_origin = tuple(
        origin[index] + radius * first_axis[index] + chart_origin[1] * axis[index]
        for index in range(3)
    )
    expected_direction = tuple(chart_direction[1] * value for value in axis)
    return source == (expected_origin, expected_direction)


def _same_extrusion_profile_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: ExtrusionSurface,
    /,
) -> bool:
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * pcurve.ambient_dimension)
    if chart is None or chart[0][0] != 0 or chart[1] != (Fraction(1), Fraction(0)):
        return False
    height = chart[0][1]
    translation = tuple(
        height * value for value in rational_vector(np.asarray(patch.direction))
    )
    base = patch.curve
    if isinstance(curve, (LineCurve, ExactLineCurve)) and isinstance(
        base, (LineCurve, ExactLineCurve)
    ):
        source, origin = (
            _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension),
            _affine_curve_coefficients(base, (Fraction(0),) * base.ambient_dimension),
        )
        if source is None or origin is None:
            return False
        expected = tuple(a + b for a, b in zip(origin[0], translation, strict=True))
        return source == (expected, origin[1])
    if isinstance(curve, (CircleCurve, EllipseCurve)) and type(curve) is type(base):
        if isinstance(curve, CircleCurve) and isinstance(base, CircleCurve):
            source_radii = (curve.radius_exact, curve.radius_exact)
            base_radii = (base.radius_exact, base.radius_exact)
        elif isinstance(curve, EllipseCurve) and isinstance(base, EllipseCurve):
            source_radii = (curve.first_radius_exact, curve.second_radius_exact)
            base_radii = (base.first_radius_exact, base.second_radius_exact)
        else:
            return False
        return (
            curve.center_exact
            == tuple(a + b for a, b in zip(base.center_exact, translation, strict=True))
            and (curve.first_axis_exact, curve.second_axis_exact)
            == (base.first_axis_exact, base.second_axis_exact)
            and source_radii == base_radii
        )
    if isinstance(curve, BSplineCurve) and isinstance(base, BSplineCurve):
        if (
            curve.degree != base.degree
            or not np.array_equal(np.asarray(curve.weights), np.asarray(base.weights))
            or not np.array_equal(np.asarray(curve.knots), np.asarray(base.knots))
        ):
            return False
        source_controls = tuple(
            rational_vector(row) for row in np.asarray(curve.control_points)
        )
        base_controls = tuple(
            rational_vector(row) for row in np.asarray(base.control_points)
        )
        return len(source_controls) == len(base_controls) and all(
            source == tuple(a + b for a, b in zip(original, translation, strict=True))
            for source, original in zip(source_controls, base_controls, strict=True)
        )
    return False


def _same_extrusion_generator_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: ExtrusionSurface,
    /,
) -> bool:
    """Prove an affine edge is the extrusion generator through an exact profile point."""
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * pcurve.ambient_dimension)
    if chart is None or chart[1][0] != 0 or chart[1][1] == 0:
        return False
    (profile_parameter, height), (_, rate) = chart
    if not isinstance(profile_parameter, Fraction) or not isinstance(height, Fraction):
        return False
    domain = patch.curve.parameter_domain
    if domain is not None and not (
        Fraction.from_float(domain[0]) <= profile_parameter <= Fraction.from_float(domain[1])
    ):
        return False
    base = exact_curve_parameter_point(patch.curve, profile_parameter)
    source = _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)
    if base is None or source is None or len(base) != 3:
        return False
    direction = rational_vector(np.asarray(patch.direction))
    return source == (
        tuple(a + height * b for a, b in zip(base, direction, strict=True)),
        tuple(rate * value for value in direction),
    )


def _same_conic_root_extrusion_line_lift(
    curve: AbstractCurve,
    pcurve: ConicRootLinePCurve,
    patch: ExtrusionSurface,
    /,
) -> bool:
    """Prove an affine 3D edge is the exact rooted conic extrusion generator."""
    source = _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)
    if source is None or len(source[0]) != 3:
        return False
    if not _same_source_definition(pcurve.root.conic, patch.curve):
        return False
    direction = rational_vector(np.asarray(patch.direction))
    root_point = pcurve.root.exact_point()
    origin = tuple(
        root_point[axis] + pcurve.height_origin_exact * direction[axis]
        for axis in range(3)
    )
    slope = tuple(pcurve.height_direction_exact * direction[axis] for axis in range(3))
    return source == (origin, slope)


def _same_conic_root_cylinder_line_lift(
    curve: AbstractCurve,
    pcurve: ConicRootLinePCurve,
    patch: CylinderPatch,
    /,
) -> bool:
    """Prove an affine edge follows one exact algebraic radial cylinder root."""
    source = _affine_curve_coefficients(curve, (Fraction(0),) * curve.ambient_dimension)
    conic = pcurve.root.conic
    if source is None or len(source[0]) != 3 or not isinstance(conic, CircleCurve):
        return False
    center = patch.origin_exact
    first_axis = patch.first_axis_exact
    second_axis = patch.second_axis_exact
    axis = patch.axis_exact
    radius = patch.radius_exact
    if (
        conic.center_exact != center
        or conic.first_axis_exact != first_axis
        or conic.second_axis_exact != second_axis
        or conic.radius_exact != radius
    ):
        return False
    point = tuple(
        center[index]
        + radius * pcurve.root.cosine_exact * first_axis[index]
        + radius * pcurve.root.sine_exact * second_axis[index]
        for index in range(3)
    )
    if point != pcurve.root.exact_point():
        return False
    origin = tuple(
        point[index] + pcurve.height_origin_exact * axis[index] for index in range(3)
    )
    slope = tuple(pcurve.height_direction_exact * axis[index] for index in range(3))
    return source == (origin, slope)


def _conic_rootline_cylinder_map(
    pcurve: ConicRootLinePCurve,
    patch: CylinderPatch | OffsetSurface,
    /,
) -> tuple[tuple[ExactScalar, ...], tuple[ExactScalar, ...]] | None:
    if pcurve.phase_shift_exact != 0:
        return None
    if isinstance(patch, CylinderPatch):
        base = patch
        radius = patch.radius_exact
    elif isinstance(patch, OffsetSurface) and isinstance(patch.base, CylinderPatch):
        if patch.frame_sign == 0:
            return None
        base = patch.base
        radius = base.radius_exact + patch.frame_sign * patch.distance_exact
    else:
        return None
    if radius <= 0:
        return None
    conic = pcurve.root.conic
    if not isinstance(conic, CircleCurve) or (
        conic.center_exact != base.origin_exact
        or conic.first_axis_exact != base.first_axis_exact
        or conic.second_axis_exact != base.second_axis_exact
        or conic.radius_exact != radius
    ):
        return None
    root_point = pcurve.root.exact_point()
    expected_point = tuple(
        base.origin_exact[index]
        + radius * pcurve.root.cosine_exact * base.first_axis_exact[index]
        + radius * pcurve.root.sine_exact * base.second_axis_exact[index]
        for index in range(3)
    )
    if root_point != expected_point:
        return None
    origin = tuple(
        root_point[index] + pcurve.height_origin_exact * base.axis_exact[index]
        for index in range(3)
    )
    direction = tuple(pcurve.height_direction_exact * value for value in base.axis_exact)
    return origin, direction


def _same_conic_rootline_cylinder_lift(
    curve: SurfaceMappedCurve,
    pcurve: ConicRootLinePCurve,
    target_patch: CylinderPatch | OffsetSurface,
    first: float,
    last: float,
    /,
) -> bool:
    if not isinstance(curve.pcurve, ConicRootLinePCurve) or not isinstance(
        curve.surface, (CylinderPatch, OffsetSurface)
    ):
        return False
    source = _conic_rootline_cylinder_map(curve.pcurve, curve.surface)
    target = _conic_rootline_cylinder_map(pcurve, target_patch)
    if source is None or target is None:
        return False
    for carrier in (curve, pcurve):
        domain = carrier.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            return False
    return source == target


def _revolution_region_box(
    region: SurfaceRegion, /
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None:
    box = np.asarray(region.parameter_box, dtype=np.float64)
    if box.shape != (2, 2) or not np.all(np.isfinite(box)):
        return None
    exact_box = (
        (
            Fraction.from_float(float(box[0, 0])),
            Fraction.from_float(float(box[0, 1])),
        ),
        (
            Fraction.from_float(float(box[1, 0])),
            Fraction.from_float(float(box[1, 1])),
        ),
    )
    if any(exact_box[0][axis] > exact_box[1][axis] for axis in range(2)):
        return None
    return exact_box


def _periodic_pcurve_source(
    curve: AbstractCurve, patch: AbstractSurfacePatch, /
) -> tuple[AbstractCurve, tuple[int, int]] | None:
    if not (hasattr(curve, "period_shifts") and hasattr(curve, "source_curve")):
        return curve, (0, 0)
    from ._boolean_core import PeriodicPCurve

    if not isinstance(curve, PeriodicPCurve):
        return None
    symbols = patch.native_period_symbols()
    shifts = [0, 0]
    source: AbstractCurve = curve
    while isinstance(source, PeriodicPCurve):
        if source.patch is not patch or any(
            shift and symbols[axis] != "two_pi"
            for axis, shift in enumerate(source.period_shifts)
        ):
            return None
        shifts = [
            current + shift
            for current, shift in zip(shifts, source.period_shifts, strict=True)
        ]
        source = source.source_curve
    return source, (shifts[0], shifts[1])


def _periodic_region_admits_shifts(
    patch: AbstractSurfacePatch,
    shifts: tuple[int, int],
    exact_box: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    /,
) -> bool:
    if not any(shifts):
        return True
    # Only the owning patch's declared period denotes a full-period chart.
    # A directed enclosure also contains partial spans and cannot authorize one.
    return all(
        not shift
        or (
            period is not None
            and exact_box[1][axis] - exact_box[0][axis]
            == Fraction.from_float(float(period))
        )
        for axis, (shift, period) in enumerate(zip(shifts, patch.periods, strict=True))
    )


def _surface_chart_coefficients(
    curve: AbstractCurve, region: SurfaceRegion, first: float, last: float, /
) -> (
    tuple[
        tuple[ExactScalar, ExactScalar],
        tuple[ExactScalar, ExactScalar],
        tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]],
    ]
    | None
):
    if curve.ambient_dimension != 2:
        return None
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return None
    periodic_source = _periodic_pcurve_source(curve, region.patch)
    if periodic_source is None:
        return None
    base_curve, shifts = periodic_source
    affine_chart = _affine_curve_coefficients(base_curve, (Fraction(0), Fraction(0)))
    exact_box = _revolution_region_box(region)
    if affine_chart is None or exact_box is None:
        return None
    if any(abs(shift) != 1 for shift in shifts if shift):
        return None
    if not _periodic_region_admits_shifts(region.patch, shifts, exact_box):
        return None
    origin, direction = affine_chart
    first_parameter = Fraction.from_float(float(first))
    last_parameter = Fraction.from_float(float(last))
    for axis in range(2):
        first_value = origin[axis] + first_parameter * direction[axis]
        last_value = origin[axis] + last_parameter * direction[axis]
        if (
            min(first_value, last_value) < exact_box[0][axis]
            or max(first_value, last_value) > exact_box[1][axis]
        ):
            return None
    return (
        (origin[0], origin[1]),
        (direction[0], direction[1]),
        exact_box,
    )


def _revolution_profile_curve_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: RevolutionSurface,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    source_domain = curve.parameter_domain
    if source_domain is not None and (
        first < source_domain[0] or last > source_domain[1]
    ):
        return False
    chart = _surface_chart_coefficients(pcurve, region, first, last)
    if chart is None:
        return False
    uv_origin, uv_direction, _ = chart
    if uv_origin[0] != 0 or uv_direction[0] != 0:
        return False
    if (
        _same_source_definition(curve, patch.curve)
        and uv_origin == (Fraction(0), Fraction(0))
        and uv_direction == (Fraction(0), Fraction(1))
    ):
        domain = patch.curve.parameter_domain
        return domain is None or (domain[0] <= first and last <= domain[1])
    source = _affine_curve_coefficients(curve, (Fraction(0),) * 3)
    profile = _affine_curve_coefficients(patch.curve, (Fraction(0),) * 3)
    if source is None or profile is None:
        return False
    source_origin, source_direction = source
    profile_origin, profile_direction = profile
    lifted_origin = tuple(
        profile_origin[index] + uv_origin[1] * profile_direction[index]
        for index in range(3)
    )
    lifted_direction = tuple(
        uv_direction[1] * profile_direction[index] for index in range(3)
    )
    if source_origin != lifted_origin or source_direction != lifted_direction:
        return False
    domain = patch.curve.parameter_domain
    if domain is not None:
        profile_first = uv_origin[1] + Fraction.from_float(float(first)) * uv_direction[1]
        profile_last = uv_origin[1] + Fraction.from_float(float(last)) * uv_direction[1]
        if min(profile_first, profile_last) < Fraction.from_float(domain[0]) or max(
            profile_first, profile_last
        ) > Fraction.from_float(domain[1]):
            return False
    return True


def _exact_dot(
    first: tuple[ExactScalar, ...], second: tuple[ExactScalar, ...], /
) -> ExactScalar:
    return sum(
        (left * right for left, right in zip(first, second, strict=True)),
        Fraction(0),
    )


def _exact_cross(
    first: tuple[ExactScalar, ExactScalar, ExactScalar],
    second: tuple[ExactScalar, ExactScalar, ExactScalar],
    /,
) -> tuple[ExactScalar, ExactScalar, ExactScalar]:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _revolution_circle_harmonics_lift(
    curve: CircleCurve,
    center: tuple[ExactScalar, ExactScalar, ExactScalar],
    radial: tuple[ExactScalar, ExactScalar, ExactScalar],
    axis: tuple[ExactScalar, ExactScalar, ExactScalar],
    /,
) -> bool:
    curve_cosine = (
        curve.radius_exact * curve.first_axis_exact[0],
        curve.radius_exact * curve.first_axis_exact[1],
        curve.radius_exact * curve.first_axis_exact[2],
    )
    curve_sine = (
        curve.radius_exact * curve.second_axis_exact[0],
        curve.radius_exact * curve.second_axis_exact[1],
        curve.radius_exact * curve.second_axis_exact[2],
    )
    return (
        curve.center_exact == center
        and curve_cosine == radial
        and curve_sine == _exact_cross(axis, radial)
    )


def _conic_parameter_root_in_interval(
    root: ConicParameterRoot,
    lower: Fraction,
    upper: Fraction,
    /,
) -> bool:
    if lower > upper:
        return False
    root_lower = Fraction.from_float(root.parameter_lower)
    root_upper = Fraction.from_float(root.parameter_upper)
    if root_upper < lower or root_lower > upper:
        return False

    def phase_relation(boundary: Fraction, /) -> int | None:
        if root_lower > boundary:
            return 1
        if root_upper < boundary:
            return -1
        if boundary == 0:
            if root.cosine_exact == 1 and root.sine_exact == 0:
                return 0
            sine_bounds = coefficient_interval(root.sine_exact, bits=128)
            if sine_bounds[0] > 0:
                return 1
            if sine_bounds[1] < 0:
                return -1
            return None
        from .._interval_enclosure import interval_cos, interval_sin

        boundary_value = np.asarray(float(boundary), dtype=np.float64)
        cosine_bounds = interval_cos((boundary_value, boundary_value))
        sine_bounds = interval_sin((boundary_value, boundary_value))
        root_cosine = coefficient_interval(root.cosine_exact, bits=128)
        root_sine = coefficient_interval(root.sine_exact, bits=128)

        def multiply(
            first: tuple[Fraction, Fraction],
            second: tuple[Fraction, Fraction],
            /,
        ) -> tuple[Fraction, Fraction]:
            products = tuple(a * b for a in first for b in second)
            return min(products), max(products)

        cosine_interval = (
            Fraction.from_float(float(cosine_bounds[0])),
            Fraction.from_float(float(cosine_bounds[1])),
        )
        sine_interval = (
            Fraction.from_float(float(sine_bounds[0])),
            Fraction.from_float(float(sine_bounds[1])),
        )
        first_product = multiply(root_sine, cosine_interval)
        second_product = multiply(root_cosine, sine_interval)
        difference = (
            first_product[0] - second_product[1],
            first_product[1] - second_product[0],
        )
        if difference[0] > 0:
            return 1
        if difference[1] < 0:
            return -1
        if difference == (0, 0):
            return 0
        return None

    lower_relation = phase_relation(lower)
    upper_relation = phase_relation(upper)
    return (
        lower_relation is not None
        and upper_relation is not None
        and lower_relation >= 0
        and upper_relation <= 0
    )


def _rooted_profile_chart(
    pcurve: AbstractCurve,
    patch: AbstractSurfacePatch,
    profile: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> ConicParameterRoot | None:
    from ._boolean_core import (
        _pcurve_execution_shadows_match,
        AffinePCurve,
    )

    periodic_source = _periodic_pcurve_source(pcurve, patch)
    if periodic_source is None or not _pcurve_execution_shadows_match(pcurve):
        return None
    source, shifts = periodic_source
    if not isinstance(source, AffinePCurve):
        return None
    root_line = source.curve
    if (
        not isinstance(root_line, ConicRootLinePCurve)
        or source.matrix != ((Fraction(0), Fraction(1)), (Fraction(1), Fraction(0)))
        or source.offset != (Fraction(0), Fraction(0))
        or root_line.phase_shift_exact != 0
        or root_line.height_origin_exact != 0
        or root_line.height_direction_exact != 1
        or shifts[0] != 0
    ):
        return None
    if (
        not isinstance(profile, (CircleCurve, EllipseCurve))
        or profile.ambient_dimension != 3
        or not _same_source_definition(root_line.root.conic, profile)
    ):
        return None
    domain = pcurve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return None
    exact_box = _revolution_region_box(region)
    if exact_box is None:
        return None
    first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
    if not (exact_box[0][0] <= first_exact and last_exact <= exact_box[1][0]):
        return None
    if not _conic_parameter_root_in_interval(
        root_line.root, exact_box[0][1], exact_box[1][1]
    ):
        return None
    if shifts[1]:
        if abs(shifts[1]) != 1 or patch.native_period_symbols()[1] != "two_pi":
            return None
        from ._trim_helpers import period_offset, period_scalar_bounds

        period_lower, period_upper = period_scalar_bounds(
            period_offset(Fraction(0), Fraction(1))
        )
        profile_span = exact_box[1][1] - exact_box[0][1]
        if not (
            Fraction.from_float(float(period_lower))
            <= profile_span
            <= Fraction.from_float(float(period_upper))
        ):
            return None
    return root_line.root


def _revolution_circle_profile_point_lift(
    curve: CircleCurve,
    profile_point: tuple[ExactScalar, ExactScalar, ExactScalar],
    patch: RevolutionSurface,
    /,
) -> bool:
    from .._rational import rational_vector

    axis_origin = rational_vector(np.asarray(patch.axis_origin))
    axis = rational_vector(np.asarray(patch.axis_direction))
    if len(axis_origin) != 3 or len(axis) != 3 or _exact_dot(axis, axis) != 1:
        return False
    axis_origin3 = cast(tuple[ExactScalar, ExactScalar, ExactScalar], axis_origin)
    axis3 = cast(tuple[ExactScalar, ExactScalar, ExactScalar], axis)
    displacement = tuple(profile_point[index] - axis_origin3[index] for index in range(3))
    axial = _exact_dot(displacement, axis3)
    center = tuple(axis_origin3[index] + axial * axis3[index] for index in range(3))
    radial = tuple(displacement[index] - axial * axis3[index] for index in range(3))
    return _revolution_circle_harmonics_lift(
        curve,
        cast(tuple[ExactScalar, ExactScalar, ExactScalar], center),
        cast(tuple[ExactScalar, ExactScalar, ExactScalar], radial),
        axis3,
    )


def _revolution_angular_circle_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: RevolutionSurface,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if not isinstance(curve, CircleCurve) or curve.ambient_dimension != 3:
        return False
    if (
        curve.native_period_symbol() != "two_pi"
        or patch.native_period_symbols()[0] != "two_pi"
    ):
        return False
    profile = patch.curve
    rooted_profile = _rooted_profile_chart(pcurve, patch, profile, region, first, last)
    if rooted_profile is not None:
        profile_point = cast(
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple(
                curve.center_exact[index]
                + curve.radius_exact * curve.first_axis_exact[index]
                for index in range(3)
            ),
        )
        replayed_root = ConicParameterRoot.from_exact_point(
            rooted_profile.conic, profile_point, rooted_profile.parameter
        )
        if (
            replayed_root is None
            or replayed_root.exact_point() != profile_point
            or replayed_root.identity_payload() != rooted_profile.identity_payload()
        ):
            return False
        return _revolution_circle_profile_point_lift(curve, profile_point, patch)
    chart = _surface_chart_coefficients(pcurve, region, first, last)
    if chart is None:
        return False
    uv_origin, uv_direction, exact_box = chart
    if uv_origin[0] != 0 or uv_direction != (Fraction(1), Fraction(0)):
        return False
    profile_parameter = uv_origin[1]
    profile_interval = exact_box[0][1], exact_box[1][1]
    if (
        isinstance(profile, (CircleCurve, EllipseCurve))
        and profile.ambient_dimension == 3
        and profile_parameter == 0
    ):
        first_radius = (
            profile.radius_exact
            if isinstance(profile, CircleCurve)
            else profile.first_radius_exact
        )
        profile_point = cast(
            tuple[ExactScalar, ExactScalar, ExactScalar],
            tuple(
                profile.center_exact[index]
                + first_radius * profile.first_axis_exact[index]
                for index in range(3)
            ),
        )
        return _revolution_circle_profile_point_lift(curve, profile_point, patch)

    witness = patch.exact_quadric_witness(profile_interval)
    if (
        witness is None
        or witness.source is not patch
        or witness.profile_parameter_interval != profile_interval
        or not profile_interval[0] <= profile_parameter <= profile_interval[1]
    ):
        return False
    if witness.family in ("plane", "cylinder", "cone"):
        if (
            witness.origin_exact is None
            or witness.axial_origin_exact is None
            or witness.axial_direction_exact is None
            or witness.radial_origin_exact is None
            or witness.radial_direction_exact is None
        ):
            return False
        axis = (
            witness.axis_exact[0],
            witness.axis_exact[1],
            witness.axis_exact[2],
        )
        witness_origin = (
            witness.origin_exact[0],
            witness.origin_exact[1],
            witness.origin_exact[2],
        )
        radial_axis = (
            witness.radial_axis_exact[0],
            witness.radial_axis_exact[1],
            witness.radial_axis_exact[2],
        )
        axial = (
            witness.axial_origin_exact + profile_parameter * witness.axial_direction_exact
        )
        radius = (
            witness.radial_origin_exact
            + profile_parameter * witness.radial_direction_exact
        )
        center = (
            witness_origin[0] + axial * axis[0],
            witness_origin[1] + axial * axis[1],
            witness_origin[2] + axial * axis[2],
        )
        radial = (
            radius * radial_axis[0],
            radius * radial_axis[1],
            radius * radial_axis[2],
        )
        return _revolution_circle_harmonics_lift(curve, center, radial, axis)
    # A rounded nonzero profile angle is not an exact conic-root chart.
    return False


def _torus_rooted_circle_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: TorusPatch,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    from ._boolean_core import AffinePCurve

    source = _periodic_pcurve_source(pcurve, patch)
    if (
        not isinstance(curve, CircleCurve)
        or curve.ambient_dimension != 3
        or source is None
        or not isinstance(source[0], AffinePCurve)
        or not isinstance(source[0].curve, ConicRootLinePCurve)
        or patch.native_period_symbols() != ("two_pi", "two_pi")
    ):
        return False
    profile = source[0].curve.root.conic
    root = _rooted_profile_chart(pcurve, patch, profile, region, first, last)
    if root is None:
        return False
    center = patch.center_exact
    first_axis = patch.first_axis_exact
    second_axis = patch.second_axis_exact
    axis = patch.axis_exact
    major = patch.major_radius_exact
    minor = patch.minor_radius_exact
    meridian = (
        tuple(center[index] + major * first_axis[index] for index in range(3)),
        tuple(minor * value for value in first_axis),
        tuple(minor * value for value in axis),
    )
    if _harmonic_coefficients(profile) != meridian:
        return False
    radius = major + minor * root.cosine_exact
    expected = (
        tuple(
            center[index] + minor * root.sine_exact * axis[index] for index in range(3)
        ),
        tuple(radius * value for value in first_axis),
        tuple(radius * value for value in second_axis),
    )
    return _harmonic_coefficients(curve) == expected


def _torus_coordinate_circle_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    patch: TorusPatch,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(curve, CircleCurve)
        or curve.ambient_dimension != 3
        or curve.native_period_symbol() != "two_pi"
        or patch.native_period_symbols() != ("two_pi", "two_pi")
    ):
        return False
    chart = _surface_chart_coefficients(pcurve, region, first, last)
    if chart is None:
        return False
    uv_origin, uv_direction, _ = chart
    if uv_origin != (Fraction(0), Fraction(0)):
        return False
    center = patch.center_exact
    first_axis = patch.first_axis_exact
    axis = patch.axis_exact
    major_radius = patch.major_radius_exact
    minor_radius = patch.minor_radius_exact
    if uv_direction == (Fraction(0), Fraction(1)):
        expected_center = tuple(
            center[index] + major_radius * first_axis[index] for index in range(3)
        )
        expected_cosine = tuple(minor_radius * value for value in first_axis)
        expected_sine = tuple(minor_radius * value for value in axis)
    elif uv_direction == (Fraction(1), Fraction(0)):
        second_axis = patch.second_axis_exact
        expected_center = center
        expected_cosine = tuple(
            (major_radius + minor_radius) * value for value in first_axis
        )
        expected_sine = tuple(
            (major_radius + minor_radius) * value for value in second_axis
        )
    else:
        return False
    actual_center, actual_cosine, actual_sine = _harmonic_coefficients(curve)
    return (
        actual_center == expected_center
        and actual_cosine == expected_cosine
        and actual_sine == expected_sine
    )


def _plane_cylinder_section_lift(
    curve: AbstractCurve,
    pcurve: PlaneCylinderSectionPCurve,
    patch: PlanePatch | CylinderPatch,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(curve, (CircleCurve, EllipseCurve))
        or curve.ambient_dimension != 3
        or curve.native_period_symbol() != "two_pi"
        or first < pcurve.parameter_range[0]
        or last > pcurve.parameter_range[1]
    ):
        return False
    actual = _harmonic_coefficients(curve)
    if isinstance(patch, PlanePatch):
        if (
            pcurve.linear_exact != (Fraction(0), Fraction(0))
            or pcurve.angle_offset_root is not None
        ):
            return False
        if _affine_surface_frame(patch) is None:
            return False
        expected = _plane_harmonics(
            patch,
            (pcurve.offset_exact, pcurve.cosine_exact, pcurve.sine_exact),
        )
        return actual == expected
    if (
        patch.native_period_symbols()[0] != "two_pi"
        or pcurve.offset_exact[0] != 0
        or pcurve.linear_exact != (Fraction(1), Fraction(0))
        or pcurve.cosine_exact[0] != 0
        or pcurve.sine_exact[0] != 0
    ):
        return False
    phase = pcurve.angle_offset_root
    if phase is None:
        phase_cosine, phase_sine = Fraction(1), Fraction(0)
    else:
        phase_curve = phase.conic
        if (
            not isinstance(phase_curve, CircleCurve)
            or phase_curve.ambient_dimension != 2
            or phase_curve.center_exact != (Fraction(0), Fraction(0))
            or phase_curve.first_axis_exact != (Fraction(1), Fraction(0))
            or phase_curve.second_axis_exact != (Fraction(0), Fraction(1))
            or phase_curve.radius_exact != 1
            or phase_curve.native_period_symbol() != "two_pi"
            or phase.cosine_exact * phase.cosine_exact
            + phase.sine_exact * phase.sine_exact
            != 1
        ):
            return False
        phase_cosine, phase_sine = phase.cosine_exact, phase.sine_exact
    radial = tuple(
        phase_cosine * patch.first_axis_exact[index]
        + phase_sine * patch.second_axis_exact[index]
        for index in range(3)
    )
    tangent = tuple(
        -phase_sine * patch.first_axis_exact[index]
        + phase_cosine * patch.second_axis_exact[index]
        for index in range(3)
    )
    center = tuple(
        patch.origin_exact[index] + pcurve.offset_exact[1] * patch.axis_exact[index]
        for index in range(3)
    )
    cosine = tuple(
        patch.radius_exact * radial[index]
        + pcurve.cosine_exact[1] * patch.axis_exact[index]
        for index in range(3)
    )
    sine = tuple(
        patch.radius_exact * tangent[index]
        + pcurve.sine_exact[1] * patch.axis_exact[index]
        for index in range(3)
    )
    return actual == (center, cosine, sine)


def _exact_plane_cylinder_section_endpoint_chart(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    endpoint: ConicParameterEndpoint,
    patch: PlanePatch | CylinderPatch,
    /,
) -> (
    tuple[ExactScalar, ExactScalar]
    | tuple[ExactScalar, ExactScalar, int, ExactScalar]
    | None
):
    from ._intersection import ConicParameterEndpoint

    if (
        not isinstance(endpoint, ConicParameterEndpoint)
        or not isinstance(curve, (CircleCurve, EllipseCurve))
        or curve.ambient_dimension != 3
        or not _same_source_definition(endpoint.root.conic, curve)
        or endpoint.point_exact != endpoint.root.exact_point()
    ):
        return None
    periodic_source = _periodic_pcurve_source(pcurve, patch)
    if periodic_source is None:
        return None
    source, period_shifts = periodic_source
    if not isinstance(source, PlaneCylinderSectionPCurve):
        return None
    if not any(
        _same_source_definition(endpoint.carrier, carrier)
        for carrier in (curve, pcurve, source)
    ):
        return None
    phi_cosine, phi_sine = endpoint.root.cosine_exact, endpoint.root.sine_exact
    if isinstance(patch, PlanePatch):
        if (
            source.linear_exact != (Fraction(0), Fraction(0))
            or source.angle_offset_root is not None
        ):
            return None
        return (
            source.offset_exact[0]
            + source.cosine_exact[0] * phi_cosine
            + source.sine_exact[0] * phi_sine,
            source.offset_exact[1]
            + source.cosine_exact[1] * phi_cosine
            + source.sine_exact[1] * phi_sine,
        )
    if (
        patch.native_period_symbols()[0] != "two_pi"
        or source.offset_exact[0] != 0
        or source.linear_exact != (Fraction(1), Fraction(0))
        or source.cosine_exact[0] != 0
        or source.sine_exact[0] != 0
        or source.linear_exact[1] != 0
    ):
        return None
    phase = source.angle_offset_root
    if phase is not None:
        phase_curve = phase.conic
        if (
            not isinstance(phase_curve, CircleCurve)
            or phase_curve.ambient_dimension != 2
            or phase_curve.center_exact != (Fraction(0), Fraction(0))
            or phase_curve.first_axis_exact != (Fraction(1), Fraction(0))
            or phase_curve.second_axis_exact != (Fraction(0), Fraction(1))
            or phase_curve.radius_exact != 1
            or phase_curve.native_period_symbol() != "two_pi"
        ):
            return None
        angle_root = _combine_conic_parameter_roots(endpoint.root, phase)
    else:
        angle_root = endpoint.root
    if angle_root is None:
        return None
    angle_cosine = angle_root.cosine_exact
    angle_sine = angle_root.sine_exact
    turns = angle_root.turns + period_shifts[0]
    height = (
        source.offset_exact[1]
        + source.cosine_exact[1] * phi_cosine
        + source.sine_exact[1] * phi_sine
    )
    return angle_cosine, angle_sine, turns, height


def _same_plane_conic_root_generator_lift(
    curve: SurfaceMappedCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(region.patch, PlanePatch)
        or not isinstance(curve.pcurve, ConicRootLinePCurve)
        or not isinstance(curve.surface, (CylinderPatch, OffsetSurface))
    ):
        return False
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return False
    chart = _surface_chart_coefficients(pcurve, region, first, last)
    if chart is None:
        return False
    source = _conic_rootline_cylinder_map(curve.pcurve, curve.surface)
    frame = _affine_surface_frame(region.patch)
    if source is None or frame is None:
        return False
    chart_origin, chart_direction, _ = chart
    origin, first_axis, second_axis = frame
    lifted_origin = tuple(
        origin[index]
        + chart_origin[0] * first_axis[index]
        + chart_origin[1] * second_axis[index]
        for index in range(3)
    )
    lifted_direction = tuple(
        chart_direction[0] * first_axis[index] + chart_direction[1] * second_axis[index]
        for index in range(3)
    )
    return source == (lifted_origin, lifted_direction)


def _same_ruled_rail_surface_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if not isinstance(region.patch, RuledSurface) or not isinstance(
        pcurve, (LineCurve, ExactLineCurve)
    ):
        return False
    if _same_source_definition(curve, region.patch.first):
        v = Fraction(0)
    elif _same_source_definition(curve, region.patch.second):
        v = Fraction(1)
    else:
        return False
    chart = _affine_curve_coefficients(pcurve, (Fraction(0), Fraction(0)))
    if chart is None:
        return False
    origin, direction = chart
    if origin != (Fraction(0), v) or direction != (Fraction(1), Fraction(0)):
        return False
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return False
    box = np.asarray(region.parameter_box, dtype=np.float64)
    lower = tuple(Fraction.from_float(float(value)) for value in box[0])
    upper = tuple(Fraction.from_float(float(value)) for value in box[1])
    return all(
        lower[axis]
        <= origin[axis] + Fraction.from_float(parameter) * direction[axis]
        <= upper[axis]
        for axis in range(2)
        for parameter in (first, last)
    )


def _rational_bernstein_line_matches(
    coefficients: tuple[tuple[ExactScalar, ...], ...],
    degree: int,
    lower: Fraction,
    upper: Fraction,
    source_origin: tuple[ExactScalar, ...],
    source_direction: tuple[ExactScalar, ...],
    /,
) -> bool:
    if (
        degree < 1
        or len(coefficients) != degree + 1
        or any(
            not isinstance(value, Fraction)
            for coefficient in coefficients
            for value in coefficient
        )
        or any(len(coefficient) != len(source_origin) + 1 for coefficient in coefficients)
    ):
        return False
    weights = tuple(coefficient[-1] for coefficient in coefficients)
    if any(weight <= 0 for weight in weights):
        return False
    first_value = tuple(
        origin + lower * direction
        for origin, direction in zip(source_origin, source_direction, strict=True)
    )
    last_value = tuple(
        origin + upper * direction
        for origin, direction in zip(source_origin, source_direction, strict=True)
    )
    for component in range(len(source_origin)):
        numerator = tuple(coefficient[component] for coefficient in coefficients)
        for index in range(degree + 2):
            elevated = Fraction(0)
            if index <= degree:
                elevated += Fraction(degree + 1 - index, degree + 1) * numerator[index]
            if index > 0:
                elevated += Fraction(index, degree + 1) * numerator[index - 1]
            product = Fraction(0)
            if index <= degree:
                product += (
                    Fraction(comb(degree, index), comb(degree + 1, index))
                    * weights[index]
                    * first_value[component]
                )
            if index > 0:
                product += (
                    Fraction(comb(degree, index - 1), comb(degree + 1, index))
                    * weights[index - 1]
                    * last_value[component]
                )
            if elevated != product:
                return False
    return True


def _bspline_boundary_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(region.patch, BSplineSurfacePatch)
        or not isinstance(curve, (LineCurve, ExactLineCurve))
        or not isinstance(pcurve, (LineCurve, ExactLineCurve))
        or curve.ambient_dimension != 3
        or pcurve.ambient_dimension != 2
    ):
        return False
    source = _affine_curve_coefficients(curve, (Fraction(0),) * 3)
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * 2)
    if source is None or chart is None:
        return False
    source_origin, source_direction = source
    chart_origin, chart_direction = chart
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return False
    patch = region.patch
    exact_knots = (patch.exact_u_knots, patch.exact_v_knots)
    degrees = (patch.u_degree, patch.v_degree)
    control_counts = (len(patch.exact_control_points), len(patch.exact_control_points[0]))
    domain_bounds = tuple(
        (knots[degree], knots[control_count])
        for knots, degree, control_count in zip(
            exact_knots, degrees, control_counts, strict=True
        )
    )
    parameter_first = Fraction.from_float(first)
    parameter_last = Fraction.from_float(last)
    box = np.asarray(region.parameter_box, dtype=np.float64)
    region_bounds = tuple(
        tuple(Fraction.from_float(float(value)) for value in box[:, axis])
        for axis in range(2)
    )
    spans = rational_bspline_surface_spans(
        np.asarray(patch.control_points, dtype=np.float64).tolist(),
        np.asarray(patch.weights, dtype=np.float64).tolist(),
        np.asarray(patch.u_knots, dtype=np.float64).tolist(),
        np.asarray(patch.v_knots, dtype=np.float64).tolist(),
        patch.u_degree,
        patch.v_degree,
        exact_control_points=patch.exact_control_points,
        exact_weights=patch.exact_weights,
        exact_u_knots=patch.exact_u_knots,
        exact_v_knots=patch.exact_v_knots,
    )
    for fixed_axis in range(2):
        free_axis = 1 - fixed_axis
        fixed_value = chart_origin[fixed_axis]
        if chart_direction[fixed_axis] != 0:
            continue
        slope = chart_direction[free_axis]
        if (
            not isinstance(slope, Fraction)
            or slope == 0
            or not isinstance(chart_origin[free_axis], Fraction)
        ):
            continue
        trace_first = chart_origin[free_axis] + slope * parameter_first
        trace_last = chart_origin[free_axis] + slope * parameter_last
        trace_lower, trace_upper = (
            min(trace_first, trace_last),
            max(trace_first, trace_last),
        )
        trace_origin = tuple(
            origin - chart_origin[free_axis] * direction / slope
            for origin, direction in zip(source_origin, source_direction, strict=True)
        )
        trace_direction = tuple(direction / slope for direction in source_direction)
        if fixed_value == domain_bounds[fixed_axis][0]:
            fixed_side = 0
        elif fixed_value == domain_bounds[fixed_axis][1]:
            fixed_side = -1
        else:
            continue
        if not (
            region_bounds[fixed_axis][0] <= fixed_value <= region_bounds[fixed_axis][1]
            and region_bounds[free_axis][0]
            <= trace_lower
            <= trace_upper
            <= region_bounds[free_axis][1]
        ):
            continue
        matched = False
        for span in spans:
            fixed_bounds = span.parameter_bounds[fixed_axis]
            if (fixed_side == 0 and fixed_bounds[0] != fixed_value) or (
                fixed_side == -1 and fixed_bounds[1] != fixed_value
            ):
                continue
            lower, upper = span.parameter_bounds[free_axis]
            if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
                return False
            if upper <= trace_lower or lower >= trace_upper:
                continue
            homogeneous = span.homogeneous_bernstein_coefficients
            boundary = (
                homogeneous[fixed_side]
                if fixed_axis == 0
                else tuple(row[fixed_side] for row in homogeneous)
            )
            degree = span.degrees[free_axis]
            if not _rational_bernstein_line_matches(
                boundary,
                degree,
                lower,
                upper,
                trace_origin,
                trace_direction,
            ):
                return False
            matched = True
        if matched:
            return True
    return False


def _tspline_boundary_line_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    from ._tspline import TSplineSurfacePatch

    if (
        not isinstance(region.patch, TSplineSurfacePatch)
        or not isinstance(curve, (LineCurve, ExactLineCurve))
        or not isinstance(pcurve, (LineCurve, ExactLineCurve))
        or curve.ambient_dimension != 3
        or pcurve.ambient_dimension != 2
    ):
        return False
    source = _affine_curve_coefficients(curve, (Fraction(0),) * 3)
    chart = _affine_curve_coefficients(pcurve, (Fraction(0),) * 2)
    if source is None or chart is None:
        return False
    source_origin, source_direction = source
    chart_origin, chart_direction = chart
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return False
    patch = region.patch
    u0, u1, v0, v1 = patch.mesh.parameter_domain
    domain_bounds = ((u0, u1), (v0, v1))
    parameter_first = Fraction.from_float(first)
    parameter_last = Fraction.from_float(last)
    box = np.asarray(region.parameter_box, dtype=np.float64)
    region_bounds = tuple(
        tuple(Fraction.from_float(float(value)) for value in box[:, axis])
        for axis in range(2)
    )
    patch.validate_exact_transfer_chain()
    for fixed_axis in range(2):
        free_axis = 1 - fixed_axis
        fixed_value = chart_origin[fixed_axis]
        if chart_direction[fixed_axis] != 0:
            continue
        slope = chart_direction[free_axis]
        if (
            not isinstance(slope, Fraction)
            or slope == 0
            or not isinstance(chart_origin[free_axis], Fraction)
        ):
            continue
        trace_first = chart_origin[free_axis] + slope * parameter_first
        trace_last = chart_origin[free_axis] + slope * parameter_last
        trace_lower, trace_upper = (
            min(trace_first, trace_last),
            max(trace_first, trace_last),
        )
        trace_origin = tuple(
            origin - chart_origin[free_axis] * direction / slope
            for origin, direction in zip(source_origin, source_direction, strict=True)
        )
        trace_direction = tuple(direction / slope for direction in source_direction)
        if fixed_value == domain_bounds[fixed_axis][0]:
            fixed_side = 0
        elif fixed_value == domain_bounds[fixed_axis][1]:
            fixed_side = -1
        else:
            continue
        if not (
            region_bounds[fixed_axis][0] <= fixed_value <= region_bounds[fixed_axis][1]
            and region_bounds[free_axis][0]
            <= trace_lower
            <= trace_upper
            <= region_bounds[free_axis][1]
        ):
            continue
        matched = False
        for extraction in patch.realization.extractions:
            u_lower, u_upper, v_lower, v_upper = extraction.parameter_bounds
            span_bounds = ((u_lower, u_upper), (v_lower, v_upper))
            fixed_bounds = span_bounds[fixed_axis]
            if (fixed_side == 0 and fixed_bounds[0] != fixed_value) or (
                fixed_side == -1 and fixed_bounds[1] != fixed_value
            ):
                continue
            lower, upper = span_bounds[free_axis]
            if upper <= trace_lower or lower >= trace_upper:
                continue
            target = patch.exact_target_homogeneous
            homogeneous = tuple(
                tuple(
                    sum(
                        (
                            extraction.exact_extraction_operator[row][coefficient]
                            * target[anchor][component]
                            for row, anchor in enumerate(extraction.coefficient_row_ids)
                        ),
                        Fraction(0),
                    )
                    for component in range(4)
                )
                for coefficient in range(16)
            )
            tensor = tuple(
                tuple(homogeneous[u * 4 + v] for v in range(4)) for u in range(4)
            )
            boundary = (
                tensor[fixed_side]
                if fixed_axis == 0
                else tuple(row[fixed_side] for row in tensor)
            )
            if not _rational_bernstein_line_matches(
                boundary, 3, lower, upper, trace_origin, trace_direction
            ):
                return False
            matched = True
        if matched:
            return True
    return False


def _rational_bernstein_value(
    coefficients: tuple[tuple[ExactScalar, ...], ...],
    lower: Fraction,
    upper: Fraction,
    parameter: Fraction,
    /,
) -> tuple[Fraction, ...] | None:
    if lower >= upper or not lower <= parameter <= upper or not coefficients:
        return None
    relative = (parameter - lower) / (upper - lower)
    homogeneous: list[Fraction] = []
    for component in range(len(coefficients[0])):
        values = tuple(coefficient[component] for coefficient in coefficients)
        if any(not isinstance(value, Fraction) for value in values):
            return None
        row = [cast(Fraction, value) for value in values]
        while len(row) > 1:
            row = [
                (1 - relative) * row[index] + relative * row[index + 1]
                for index in range(len(row) - 1)
            ]
        homogeneous.append(row[0])
    if homogeneous[-1] <= 0:
        return None
    return tuple(value / homogeneous[-1] for value in homogeneous[:-1])


def _rational_spans_cover_interval(
    spans: tuple[tuple[Fraction, Fraction, tuple[tuple[ExactScalar, ...], ...]], ...],
    lower: Fraction,
    upper: Fraction,
    /,
) -> bool:
    if lower >= upper or not spans:
        return False
    relevant = tuple(
        sorted(
            (span for span in spans if span[1] > lower and span[0] < upper),
            key=lambda span: (span[0], span[1]),
        )
    )
    cursor = lower
    for index, span in enumerate(relevant):
        span_lower, span_upper, coefficients = span
        clipped_lower, clipped_upper = max(lower, span_lower), min(upper, span_upper)
        if clipped_lower > cursor:
            return False
        if index and relevant[index - 1][1] == span_lower:
            previous = relevant[index - 1]
            left = _rational_bernstein_value(
                previous[2], previous[0], previous[1], span_lower
            )
            right = _rational_bernstein_value(
                coefficients, span_lower, span_upper, span_lower
            )
            if left is None or left != right:
                return False
        cursor = max(cursor, clipped_upper)
    if cursor < upper:
        return False
    for endpoint in (lower, upper):
        incident = tuple(
            _rational_bernstein_value(coefficients, span_lower, span_upper, endpoint)
            for span_lower, span_upper, coefficients in spans
            if span_lower <= endpoint <= span_upper
        )
        if (
            not incident
            or incident[0] is None
            or any(value != incident[0] for value in incident[1:])
        ):
            return False
    return True


def _exact_bspline_curve_spans(
    curve: BSplineCurve, /
) -> tuple[tuple[Fraction, Fraction, tuple[tuple[ExactScalar, ...], ...]], ...] | None:
    if any(
        not isinstance(value, Fraction)
        for row in curve.exact_control_points
        for value in row
    ) or any(
        not isinstance(value, Fraction)
        for value in (*curve.exact_weights, *curve.exact_knots)
    ):
        return None
    spans = rational_bspline_curve_spans(
        np.asarray(curve.control_points, dtype=np.float64).tolist(),
        np.asarray(curve.weights, dtype=np.float64).tolist(),
        np.asarray(curve.knots, dtype=np.float64).tolist(),
        curve.degree,
        exact_control_points=curve.exact_control_points,
        exact_weights=curve.exact_weights,
        exact_knots=curve.exact_knots,
    )
    result = []
    for span in spans:
        lower, upper = span.parameter_bounds
        if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
            return None
        result.append((lower, upper, span.homogeneous_bernstein_coefficients))
    return tuple(result)


def _bernstein_power_in_parameter(
    coefficients: tuple[tuple[ExactScalar, ...], ...],
    lower: Fraction,
    upper: Fraction,
    offset: Fraction,
    slope: Fraction,
    /,
) -> tuple[tuple[Fraction, ...], ...] | None:
    if (
        lower >= upper
        or slope == 0
        or not coefficients
        or any(
            not isinstance(value, Fraction)
            for coefficient in coefficients
            for value in coefficient
        )
    ):
        return None
    degree = len(coefficients) - 1
    width = upper - lower
    alpha = (offset - lower) / width
    beta = slope / width
    component_count = len(coefficients[0])
    if component_count < 2 or any(
        len(coefficient) != component_count for coefficient in coefficients
    ):
        return None
    result: list[tuple[Fraction, ...]] = []
    for component in range(component_count):
        local_power = [Fraction(0) for _ in range(degree + 1)]
        for index, coefficient in enumerate(coefficients):
            for power in range(index, degree + 1):
                local_power[power] += (
                    coefficient[component]
                    * comb(degree, index)
                    * comb(degree - index, power - index)
                    * (-1) ** (power - index)
                )
        result.append(
            tuple(
                sum(
                    (
                        local_power[power]
                        * comb(power, target)
                        * alpha ** (power - target)
                        * beta**target
                        for power in range(target, degree + 1)
                    ),
                    Fraction(0),
                )
                for target in range(degree + 1)
            )
        )
    return tuple(result)


def _polynomial_product(
    first: tuple[Fraction, ...], second: tuple[Fraction, ...], /
) -> tuple[Fraction, ...]:
    result = [Fraction(0) for _ in range(len(first) + len(second) - 1)]
    for first_power, first_value in enumerate(first):
        for second_power, second_value in enumerate(second):
            result[first_power + second_power] += first_value * second_value
    while len(result) > 1 and result[-1] == 0:
        result.pop()
    return tuple(result)


def _rational_power_maps_match(
    first: tuple[tuple[Fraction, ...], ...],
    second: tuple[tuple[Fraction, ...], ...],
    /,
) -> bool:
    if len(first) != len(second) or len(first) < 2:
        return False
    first_weight, second_weight = first[-1], second[-1]
    return all(
        _polynomial_product(first[component], second_weight)
        == _polynomial_product(second[component], first_weight)
        for component in range(len(first) - 1)
    )


def _exact_spline_surface_boundary_spans(
    patch: AbstractSurfacePatch,
    fixed_axis: int,
    fixed_side: int,
    /,
) -> tuple[tuple[Fraction, Fraction, tuple[tuple[ExactScalar, ...], ...]], ...]:
    from ._tspline import TSplineSurfacePatch

    if isinstance(patch, BSplineSurfacePatch):
        spans = rational_bspline_surface_spans(
            np.asarray(patch.control_points, dtype=np.float64).tolist(),
            np.asarray(patch.weights, dtype=np.float64).tolist(),
            np.asarray(patch.u_knots, dtype=np.float64).tolist(),
            np.asarray(patch.v_knots, dtype=np.float64).tolist(),
            patch.u_degree,
            patch.v_degree,
            exact_control_points=patch.exact_control_points,
            exact_weights=patch.exact_weights,
            exact_u_knots=patch.exact_u_knots,
            exact_v_knots=patch.exact_v_knots,
        )
        exact_knots = (patch.exact_u_knots, patch.exact_v_knots)
        degrees = (patch.u_degree, patch.v_degree)
        counts = (len(patch.exact_control_points), len(patch.exact_control_points[0]))
        domain = tuple(
            (knots[degree], knots[count])
            for knots, degree, count in zip(exact_knots, degrees, counts, strict=True)
        )
        fixed_value = domain[fixed_axis][0 if fixed_side == 0 else 1]
        output = []
        for span in spans:
            fixed_bounds = span.parameter_bounds[fixed_axis]
            if (fixed_side == 0 and fixed_bounds[0] != fixed_value) or (
                fixed_side == -1 and fixed_bounds[1] != fixed_value
            ):
                continue
            lower, upper = span.parameter_bounds[1 - fixed_axis]
            if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
                return ()
            homogeneous = span.homogeneous_bernstein_coefficients
            boundary = (
                homogeneous[fixed_side]
                if fixed_axis == 0
                else tuple(row[fixed_side] for row in homogeneous)
            )
            output.append((lower, upper, boundary))
        return tuple(output)
    if not isinstance(patch, TSplineSurfacePatch):
        return ()
    u0, u1, v0, v1 = patch.mesh.parameter_domain
    domain = ((u0, u1), (v0, v1))
    fixed_value = domain[fixed_axis][0 if fixed_side == 0 else 1]
    patch.validate_exact_transfer_chain()
    target = patch.exact_target_homogeneous
    output = []
    for extraction in patch.realization.extractions:
        u_lower, u_upper, v_lower, v_upper = extraction.parameter_bounds
        bounds = ((u_lower, u_upper), (v_lower, v_upper))
        if (fixed_side == 0 and bounds[fixed_axis][0] != fixed_value) or (
            fixed_side == -1 and bounds[fixed_axis][1] != fixed_value
        ):
            continue
        lower, upper = bounds[1 - fixed_axis]
        homogeneous = tuple(
            tuple(
                sum(
                    (
                        extraction.exact_extraction_operator[row][coefficient]
                        * target[anchor][component]
                        for row, anchor in enumerate(extraction.coefficient_row_ids)
                    ),
                    Fraction(0),
                )
                for component in range(4)
            )
            for coefficient in range(16)
        )
        tensor = tuple(tuple(homogeneous[u * 4 + v] for v in range(4)) for u in range(4))
        boundary = (
            tensor[fixed_side]
            if fixed_axis == 0
            else tuple(row[fixed_side] for row in tensor)
        )
        output.append((lower, upper, boundary))
    return tuple(output)


def _bspline_chart_line(
    curve: BSplineCurve,
    first: float,
    last: float,
    /,
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]] | None:
    spans = _exact_bspline_curve_spans(curve)
    if spans is None or not first < last:
        return None
    lower = curve.exact_knots[curve.degree]
    upper = curve.exact_knots[len(curve.exact_control_points)]
    if not isinstance(lower, Fraction) or not isinstance(upper, Fraction):
        return None
    first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
    if not lower <= first_exact < last_exact <= upper:
        return None
    if not _rational_spans_cover_interval(spans, first_exact, last_exact):
        return None

    def evaluate(parameter: Fraction, /) -> tuple[Fraction, Fraction] | None:
        values = tuple(
            _rational_bernstein_value(coefficients, span_lower, span_upper, parameter)
            for span_lower, span_upper, coefficients in spans
            if span_lower <= parameter <= span_upper
        )
        if (
            not values
            or values[0] is None
            or any(value != values[0] for value in values[1:])
        ):
            return None
        value = values[0]
        return (value[0], value[1]) if len(value) == 2 else None

    first_point, last_point = evaluate(first_exact), evaluate(last_exact)
    if first_point is None or last_point is None:
        return None
    direction = tuple(
        (last_value - first_value) / (last_exact - first_exact)
        for first_value, last_value in zip(first_point, last_point, strict=True)
    )
    origin = tuple(
        value - first_exact * slope
        for value, slope in zip(first_point, direction, strict=True)
    )
    if any(
        span_upper > first_exact
        and span_lower < last_exact
        and not _rational_bernstein_line_matches(
            coefficients,
            curve.degree,
            span_lower,
            span_upper,
            origin,
            direction,
        )
        for span_lower, span_upper, coefficients in spans
    ):
        return None
    return (origin[0], origin[1]), (direction[0], direction[1])


def _affine_bspline_pcurve(
    pcurve: AbstractCurve, first: float, last: float, /
) -> AbstractCurve | None:
    """Normalize only a proved common-parameter affine spline for source proofs."""
    if not isinstance(pcurve, BSplineCurve) or pcurve.ambient_dimension != 2:
        return None
    chart = _bspline_chart_line(pcurve, first, last)
    if chart is None or not any(chart[1]):
        return None
    return _affine_pcurve_carrier(pcurve, chart[0], chart[1])


def _affine_transport_line_pcurve(pcurve: AbstractCurve, /) -> AbstractCurve | None:
    """Compose an exact chart transport of a line p-curve into one exact line.

    Coincident-surface transport keeps the source parameter, so the composed
    line is the same p-curve in the receiving chart, never a refit.
    """
    from ._boolean_core import _pcurve_execution_shadows_match, AffinePCurve

    if not isinstance(pcurve, AffinePCurve) or not _pcurve_execution_shadows_match(
        pcurve
    ):
        return None
    inner = pcurve.curve
    if isinstance(inner, AffinePCurve):
        inner = _affine_transport_line_pcurve(inner)
        if inner is None:
            return None
    if not isinstance(inner, (LineCurve, ExactLineCurve)):
        return None
    coefficients = _affine_curve_coefficients(inner, (Fraction(0), Fraction(0)))
    if coefficients is None:
        return None
    origin, direction = coefficients
    matrix, offset = pcurve.matrix, pcurve.offset
    return _affine_pcurve_carrier(
        inner,
        (
            offset[0] + matrix[0][0] * origin[0] + matrix[0][1] * origin[1],
            offset[1] + matrix[1][0] * origin[0] + matrix[1][1] * origin[1],
        ),
        (
            matrix[0][0] * direction[0] + matrix[0][1] * direction[1],
            matrix[1][0] * direction[0] + matrix[1][1] * direction[1],
        ),
    )


def _exact_spline_boundary_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    from ._tspline import TSplineSurfacePatch

    patch = region.patch
    if (
        not isinstance(curve, BSplineCurve)
        or curve.ambient_dimension != 3
        or not isinstance(patch, (BSplineSurfacePatch, TSplineSurfacePatch))
        or not isinstance(pcurve, (LineCurve, ExactLineCurve, BSplineCurve))
        or pcurve.ambient_dimension != 2
        or not np.isfinite(first)
        or not np.isfinite(last)
        or not first < last
    ):
        return False
    source_spans = _exact_bspline_curve_spans(curve)
    if source_spans is None:
        return False
    first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
    source_lower = curve.exact_knots[curve.degree]
    source_upper = curve.exact_knots[len(curve.exact_control_points)]
    if (
        not isinstance(source_lower, Fraction)
        or not isinstance(source_upper, Fraction)
        or not source_lower <= first_exact < last_exact <= source_upper
        or not _rational_spans_cover_interval(source_spans, first_exact, last_exact)
    ):
        return False
    chart = (
        _bspline_chart_line(pcurve, first, last)
        if isinstance(pcurve, BSplineCurve)
        else _affine_curve_coefficients(pcurve, (Fraction(0), Fraction(0)))
    )
    if chart is None:
        return False
    chart_origin, chart_direction = chart
    if any(
        not isinstance(value, Fraction) for value in (*chart_origin, *chart_direction)
    ):
        return False
    chart_origin = cast(tuple[Fraction, Fraction], chart_origin)
    chart_direction = cast(tuple[Fraction, Fraction], chart_direction)
    if isinstance(patch, BSplineSurfacePatch):
        exact_knots = (patch.exact_u_knots, patch.exact_v_knots)
        degrees = (patch.u_degree, patch.v_degree)
        counts = (len(patch.exact_control_points), len(patch.exact_control_points[0]))
        domain = tuple(
            (knots[degree], knots[count])
            for knots, degree, count in zip(exact_knots, degrees, counts, strict=True)
        )
    else:
        u0, u1, v0, v1 = patch.mesh.parameter_domain
        domain = ((u0, u1), (v0, v1))
    box = np.asarray(region.parameter_box, dtype=np.float64)
    region_bounds = tuple(
        tuple(Fraction.from_float(float(value)) for value in box[:, axis])
        for axis in range(2)
    )
    for fixed_axis in range(2):
        free_axis = 1 - fixed_axis
        fixed_value = chart_origin[fixed_axis]
        if chart_direction[fixed_axis] != 0:
            continue
        if fixed_value == domain[fixed_axis][0]:
            fixed_side = 0
        elif fixed_value == domain[fixed_axis][1]:
            fixed_side = -1
        else:
            continue
        offset, slope = chart_origin[free_axis], chart_direction[free_axis]
        if slope == 0:
            continue
        trace_first = offset + slope * first_exact
        trace_last = offset + slope * last_exact
        trace_lower, trace_upper = (
            min(trace_first, trace_last),
            max(trace_first, trace_last),
        )
        if not (
            domain[free_axis][0] <= trace_lower < trace_upper <= domain[free_axis][1]
            and region_bounds[fixed_axis][0]
            <= fixed_value
            <= region_bounds[fixed_axis][1]
            and region_bounds[free_axis][0]
            <= trace_lower
            < trace_upper
            <= region_bounds[free_axis][1]
        ):
            continue
        boundary_spans = _exact_spline_surface_boundary_spans(
            patch, fixed_axis, fixed_side
        )
        if not _rational_spans_cover_interval(boundary_spans, trace_lower, trace_upper):
            continue
        matched = False
        for boundary_lower, boundary_upper, boundary_coefficients in boundary_spans:
            parameter_first = (boundary_lower - offset) / slope
            parameter_last = (boundary_upper - offset) / slope
            mapped_lower, mapped_upper = (
                min(parameter_first, parameter_last),
                max(parameter_first, parameter_last),
            )
            for source_span_lower, source_span_upper, source_coefficients in source_spans:
                overlap_lower = max(first_exact, mapped_lower, source_span_lower)
                overlap_upper = min(last_exact, mapped_upper, source_span_upper)
                if overlap_lower >= overlap_upper:
                    continue
                source_power = _bernstein_power_in_parameter(
                    source_coefficients,
                    source_span_lower,
                    source_span_upper,
                    Fraction(0),
                    Fraction(1),
                )
                boundary_power = _bernstein_power_in_parameter(
                    boundary_coefficients,
                    boundary_lower,
                    boundary_upper,
                    offset,
                    slope,
                )
                if (
                    source_power is None
                    or boundary_power is None
                    or not _rational_power_maps_match(source_power, boundary_power)
                ):
                    return False
                matched = True
        if matched:
            return True
    return False


def _ruled_rail_chart_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    rail_parameter: int,
    /,
) -> bool:
    if (
        not isinstance(region.patch, RuledSurface)
        or not isinstance(pcurve, (LineCurve, ExactLineCurve))
        or region.patch.periods[0] is None
        or region.patch.native_period_symbols()[0] != "two_pi"
    ):
        return False
    source_domain = curve.parameter_domain
    if source_domain is not None and (
        first < source_domain[0] or last > source_domain[1]
    ):
        return False
    patch_domain = region.patch.parameter_domain
    if patch_domain is not None and (
        first < patch_domain[0][0] or last > patch_domain[1][0]
    ):
        return False
    chart = _surface_chart_coefficients(pcurve, region, first, last)
    return chart is not None and chart[:2] == (
        (Fraction(0), Fraction(rail_parameter)),
        (Fraction(1), Fraction(0)),
    )


def _ruled_first_circle_rail_lift(
    curve: CircleCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(region.patch, RuledSurface)
        or curve.period is None
        or curve.native_period_symbol() != "two_pi"
        or region.patch.periods[0] != curve.period
        or not _same_source_definition(curve, region.patch.first)
    ):
        return False
    return _ruled_rail_chart_lift(curve, pcurve, region, first, last, 0)


def _ruled_second_revolution_ring_lift(
    curve: SurfaceIsoparametricCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if (
        not isinstance(region.patch, RuledSurface)
        or not isinstance(curve.surface, RevolutionSurface)
        or curve.fixed_axis != 1
        or curve.period is None
        or curve.native_period_symbol() != "two_pi"
        or region.patch.periods[0] != curve.period
        or not _same_source_definition(curve, region.patch.second)
    ):
        return False
    return _ruled_rail_chart_lift(curve, pcurve, region, first, last, 1)


def _same_mapped_surface_chart_lift(
    curve: SurfaceMappedCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    if not _same_patch(curve.surface, region.patch):
        return False
    domain = curve.parameter_domain
    if domain is not None and (first < domain[0] or last > domain[1]):
        return False
    source_chart = _periodic_pcurve_source(curve.pcurve, curve.surface)
    target_chart = _periodic_pcurve_source(pcurve, region.patch)
    exact_box = _revolution_region_box(region)
    if (
        source_chart is None
        or target_chart is None
        or exact_box is None
        or not _same_source_definition(source_chart[0], target_chart[0])
    ):
        return False
    if any(
        not _periodic_region_admits_shifts(region.patch, shifts, exact_box)
        for shifts in (source_chart[1], target_chart[1])
    ):
        return False
    image = np.asarray(source_chart[0].bounding_box(first, last), dtype=np.float64)
    if (
        image.shape != (2, 2)
        or not np.all(np.isfinite(image))
        or np.any(image[0] > image[1])
    ):
        return False
    return all(
        exact_box[0][axis] <= Fraction.from_float(float(image[0, axis]))
        and Fraction.from_float(float(image[1, axis])) <= exact_box[1][axis]
        for axis in range(2)
    )


def exact_source_lift(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> bool:
    """Prove p-curve identity from source constructions, never sampled residuals.

    This ordered scientific dispatch stays explicit: overlapping carrier and
    support classes have distinct refusal precedence. Substantial proofs live
    in their owning helpers rather than an opaque class-pair registry.
    """
    if not (np.isfinite(first) and np.isfinite(last) and first <= last):
        return False
    affine_pcurve = _affine_bspline_pcurve(pcurve, first, last)
    if affine_pcurve is None:
        affine_pcurve = _affine_transport_line_pcurve(pcurve)
    if affine_pcurve is not None and exact_source_lift(
        curve, affine_pcurve, region, first, last
    ):
        return True
    if isinstance(region.patch, CylinderPatch):
        source_chart = _cylinder_chart_line(curve)
        if source_chart is not None and not _same_patch(source_chart[0], region.patch):
            return _same_cylinder_chart_line_lift(
                curve, pcurve, region.patch, first, last
            )
    if isinstance(region.patch, CylinderPatch) and _same_cylinder_generator_line_lift(
        curve, pcurve, region.patch
    ):
        return True
    if (
        isinstance(curve, SurfaceMappedCurve)
        and isinstance(curve.pcurve, ConicRootLinePCurve)
        and isinstance(pcurve, ConicRootLinePCurve)
        and isinstance(region.patch, (CylinderPatch, OffsetSurface))
    ):
        domain = curve.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            return False
        if _same_patch(curve.surface, region.patch) and _same_source_definition(
            curve.pcurve, pcurve
        ):
            return True
        return _same_conic_rootline_cylinder_lift(
            curve, pcurve, region.patch, first, last
        )
    if isinstance(pcurve, ConicRootLinePCurve) and isinstance(
        region.patch, (ExtrusionSurface, CylinderPatch)
    ):
        domain = pcurve.parameter_domain
        if domain is not None and (first < domain[0] or last > domain[1]):
            return False
        if isinstance(region.patch, ExtrusionSurface):
            return _same_conic_root_extrusion_line_lift(curve, pcurve, region.patch)
        return _same_conic_root_cylinder_line_lift(curve, pcurve, region.patch)
    if (
        isinstance(region.patch, ExtrusionSurface)
        and isinstance(pcurve, (LineCurve, ExactLineCurve))
        and (
            _same_extrusion_profile_line_lift(curve, pcurve, region.patch)
            or _same_extrusion_generator_line_lift(curve, pcurve, region.patch)
        )
    ):
        return True
    if isinstance(curve, IntersectionCurve) and isinstance(pcurve, IntersectionPCurve):
        if (
            curve.branch_id != pcurve.curve.branch_id
            or not curve.fully_certified
            or pcurve.reversed
            or not pcurve.first <= first <= last <= pcurve.last
        ):
            return False
        selected = curve.first if pcurve.side == "first" else curve.second
        return _same_patch(selected.patch, region.patch)
    if isinstance(curve, PlacedCurve) and isinstance(region.patch, PlacedSurface):
        if not _same_pose_values(curve, region.patch):
            return False
        source_patch = region.patch.definition
        if isinstance(source_patch, RevolutionSurface) and not isinstance(
            curve.definition, (BSplineCurve, SurfaceMappedCurve, IntersectionCurve)
        ):
            return False
        source_region = SurfaceRegion(source_patch, region.parameter_box)
        return exact_source_lift(curve.definition, pcurve, source_region, first, last)
    if _bspline_boundary_line_lift(curve, pcurve, region, first, last):
        return True
    if _tspline_boundary_line_lift(curve, pcurve, region, first, last):
        return True
    if _exact_spline_boundary_lift(curve, pcurve, region, first, last):
        return True
    if isinstance(region.patch, RuledSurface) and _same_ruled_rail_surface_lift(
        curve, pcurve, region, first, last
    ):
        return True

    if isinstance(curve, SurfaceMappedCurve):
        if isinstance(region.patch, PlanePatch) and _same_plane_conic_root_generator_lift(
            curve, pcurve, region, first, last
        ):
            return True
        return _same_mapped_surface_chart_lift(curve, pcurve, region, first, last)
    if isinstance(region.patch, RevolutionSurface) and (
        _revolution_profile_curve_lift(curve, pcurve, region.patch, region, first, last)
        or _revolution_angular_circle_lift(
            curve, pcurve, region.patch, region, first, last
        )
    ):
        return True
    if isinstance(region.patch, TorusPatch):
        return _torus_coordinate_circle_lift(
            curve, pcurve, region.patch, region, first, last
        ) or _torus_rooted_circle_lift(curve, pcurve, region.patch, region, first, last)
    if isinstance(region.patch, (PlanePatch, CylinderPatch)):
        section_source = _periodic_pcurve_source(pcurve, region.patch)
        if section_source is None:
            return False
        section_curve, _ = section_source
        if isinstance(section_curve, PlaneCylinderSectionPCurve):
            return _plane_cylinder_section_lift(
                curve, section_curve, region.patch, first, last
            )
    if isinstance(region.patch, RuledSurface):
        if isinstance(curve, SurfaceIsoparametricCurve):
            return _ruled_second_revolution_ring_lift(curve, pcurve, region, first, last)
        if isinstance(curve, CircleCurve):
            return _ruled_first_circle_rail_lift(curve, pcurve, region, first, last)

    frame = _affine_surface_frame(region.patch)
    if frame is not None and _same_plane_line_lift(curve, pcurve, frame):
        return True
    if isinstance(curve, SurfaceIsoparametricCurve) and isinstance(pcurve, LineCurve):
        if not _same_patch(curve.surface, region.patch) or pcurve.ambient_dimension != 2:
            return False
        parameter_domain = curve.parameter_domain
        if parameter_domain is not None and (
            first < parameter_domain[0] or last > parameter_domain[1]
        ):
            return False
        fixed = (
            curve.fixed_value_exact
            if curve.fixed_value_exact is not None
            else Fraction.from_float(curve.fixed_value)
        )
        if curve.fixed_axis == 0:
            expected_origin = (fixed, Fraction(0))
            expected_direction = (Fraction(0), Fraction(1))
        else:
            expected_origin = (Fraction(0), fixed)
            expected_direction = (Fraction(1), Fraction(0))
        if (
            pcurve.origin_exact != expected_origin
            or pcurve.direction_exact != expected_direction
        ):
            return False
        region_box = np.asarray(region.parameter_box, dtype=np.float64)
        first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
        return all(
            Fraction.from_float(float(region_box[0, axis]))
            <= pcurve.origin_exact[axis] + endpoint * pcurve.direction_exact[axis]
            <= Fraction.from_float(float(region_box[1, axis]))
            for axis in range(2)
            for endpoint in (first_exact, last_exact)
        )
    if isinstance(curve, (CircleCurve, EllipseCurve)):
        from ._source_correspondence import circle_correspondence

        bound = circle_correspondence(curve, pcurve, region.patch)
        return bound == 0.0
    return False


__all__ = ["exact_source_lift", "native_edge_pcurve"]
