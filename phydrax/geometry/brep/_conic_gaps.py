#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exact-source endpoint and face-gap bounds for native CAD carriers."""

from __future__ import annotations

from fractions import Fraction

import numpy as np

from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from .._interval_enclosure import interval_cos, interval_sin
from .._rational import fraction_down, fraction_up, rational_vector
from ._carrier_pieces import carrier_payload
from ._exact import square_root_interval
from ._intersection import NativePeriodEndpoint
from ._intersection_curve import exact_curve_parameter_point, SurfaceRegion
from ._patches import (
    AbstractCurve,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConicParameterRoot,
    CylinderPatch,
    EllipseCurve,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
    SurfaceIsoparametricCurve,
    SurfaceMappedCurve,
    TorusPatch,
)
from ._trim_helpers import period_scalar_bounds, PeriodOffset


type _Vector = tuple[ExactScalar, ...]
type _Harmonics = tuple[_Vector, _Vector, _Vector]


def _affine_coefficients(curve: AbstractCurve, /) -> tuple[_Vector, _Vector] | None:
    if isinstance(curve, LineCurve):
        return curve.origin_exact, curve.direction_exact
    if isinstance(curve, ExactLineCurve):
        return curve.first_exact, tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    return None


def _conic_coefficients(curve: CircleCurve | EllipseCurve, /) -> _Harmonics:
    if isinstance(curve, CircleCurve):
        first_radius = second_radius = curve.radius_exact
    else:
        first_radius = curve.first_radius_exact
        second_radius = curve.second_radius_exact
    return (
        curve.center_exact,
        tuple(first_radius * value for value in curve.first_axis_exact),
        tuple(second_radius * value for value in curve.second_axis_exact),
    )


def _vector_difference(first: _Vector, second: _Vector, /) -> _Vector:
    return tuple(a - b for a, b in zip(first, second, strict=True))


def _trig_intervals(
    first: float, last: float, /
) -> tuple[tuple[Fraction, Fraction], ...]:
    lower, upper = min(first, last), max(first, last)
    cosine = interval_cos((np.asarray(lower), np.asarray(upper)))
    sine = interval_sin((np.asarray(lower), np.asarray(upper)))
    return (
        (Fraction(float(cosine[0])), Fraction(float(cosine[1]))),
        (Fraction(float(sine[0])), Fraction(float(sine[1]))),
    )


def _scaled_interval(
    coefficient: ExactScalar, interval: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    coefficient_lower, coefficient_upper = coefficient_interval(coefficient)
    products = (
        coefficient_lower * interval[0],
        coefficient_lower * interval[1],
        coefficient_upper * interval[0],
        coefficient_upper * interval[1],
    )
    return min(products), max(products)


def _squared_gap_upper(
    point: tuple[ExactScalar, ...], target: tuple[Fraction, ...], /
) -> Fraction:
    if len(point) != len(target):
        raise ValueError("Exact source point and target vertex dimensions must agree.")
    squared = Fraction(0)
    for value, expected in zip(point, target, strict=True):
        lower, upper = coefficient_interval(value - expected)
        magnitude = max(abs(lower), abs(upper))
        squared += magnitude * magnitude
    return squared


def _harmonic_squared_upper(
    coefficients: _Harmonics,
    first: float,
    last: float,
    /,
) -> Fraction:
    cosine, sine = _trig_intervals(first, last)
    squared = Fraction(0)
    for constant, first_coefficient, second_coefficient in zip(
        *coefficients, strict=True
    ):
        first_term = _scaled_interval(first_coefficient, cosine)
        second_term = _scaled_interval(second_coefficient, sine)
        constant_lower, constant_upper = coefficient_interval(constant)
        lower = constant_lower + first_term[0] + second_term[0]
        upper = constant_upper + first_term[1] + second_term[1]
        magnitude = max(abs(lower), abs(upper))
        squared += magnitude * magnitude
    return squared


def _periodic_quarter_trig(
    parameter: Fraction | PeriodOffset, /
) -> tuple[Fraction, Fraction] | None:
    if isinstance(parameter, PeriodOffset):
        rational, turns = parameter.rational, parameter.turns
    else:
        rational, turns = parameter, Fraction(0)
    quarter_turns = 4 * turns
    if rational != 0 or quarter_turns.denominator != 1:
        return None
    phase = int(quarter_turns) % 4
    return (
        (Fraction(1), Fraction(0)),
        (Fraction(0), Fraction(1)),
        (Fraction(-1), Fraction(0)),
        (Fraction(0), Fraction(-1)),
    )[phase]


def _revolution_profile_parameter(
    profile: AbstractCurve, fixed_value: float, /
) -> ExactScalar | PeriodOffset | None:
    if isinstance(profile, (CircleCurve, EllipseCurve)):
        period = profile.period
        if profile.native_period_symbol() != "two_pi" or period is None:
            return None
        for turns in (
            Fraction(0),
            Fraction(1, 4),
            Fraction(1, 2),
            Fraction(3, 4),
            Fraction(1),
        ):
            if fixed_value == float(period) * float(turns):
                return Fraction(0) if turns == 0 else PeriodOffset(Fraction(0), turns)
        return None
    if isinstance(profile, BSplineCurve):
        endpoint_parameters = (
            profile.exact_knots[profile.degree],
            profile.exact_knots[len(profile.exact_control_points)],
        )
        matches = tuple(
            parameter
            for parameter in endpoint_parameters
            if float(parameter) == fixed_value
        )
        return matches[0] if len(matches) == 1 else None
    return None


def _offset_revolution_isoparametric_point(
    curve: AbstractCurve,
    root: NativePeriodEndpoint,
    /,
) -> tuple[ExactScalar, ...] | None:
    if not isinstance(curve, SurfaceIsoparametricCurve):
        return None
    surface = curve.surface
    if not isinstance(surface, OffsetSurface) or not isinstance(
        surface.base, RevolutionSurface
    ):
        return None
    revolution = surface.base
    profile = revolution.curve
    variable_axis = 1 - curve.fixed_axis
    if root.patch is None:
        if root.axis is not None or curve.native_period_symbol() != "two_pi":
            return None
    elif root.axis != variable_axis or carrier_payload(root.patch) != carrier_payload(
        surface
    ):
        return None

    if curve.fixed_axis == 0:
        if curve.fixed_value != 0.0 or not isinstance(
            profile, (CircleCurve, EllipseCurve)
        ):
            return None
        phase = _periodic_quarter_trig(root.exact_parameter)
        if phase is None:
            return None
        cosine, sine = phase
        source_point = tuple(
            profile.center_exact[index]
            + (
                profile.radius_exact
                if isinstance(profile, CircleCurve)
                else profile.first_radius_exact
            )
            * cosine
            * profile.first_axis_exact[index]
            + (
                profile.radius_exact
                if isinstance(profile, CircleCurve)
                else profile.second_radius_exact
            )
            * sine
            * profile.second_axis_exact[index]
            for index in range(3)
        )
        sphere_witness = surface.exact_sphere_revolution_witness()
        if sphere_witness is not None:
            center_exact, radius_exact, axis_exact, frame_sign = sphere_witness
            if (
                len(center_exact) != 3
                or len(axis_exact) != 3
                or sum(value * value for value in axis_exact) != 1
                or frame_sign not in (-1, 1)
                or radius_exact <= 0
            ):
                return None
            radial = tuple(
                source_point[index] - center_exact[index] for index in range(3)
            )
            if sum(value * value for value in radial) != radius_exact * radius_exact:
                return None
            offset_radius = radius_exact + frame_sign * surface.distance_exact
            if offset_radius <= 0:
                return None
            return tuple(
                center_exact[index] + offset_radius * radial[index] / radius_exact
                for index in range(3)
            )
        from ._root_bindings import revolution_offset_conic_frame

        frame = revolution_offset_conic_frame(surface, root.exact_parameter)
        return None if frame is None else frame[0]

    if curve.fixed_axis == 1:
        profile_parameter = _revolution_profile_parameter(profile, curve.fixed_value)
        if profile_parameter is None:
            return None
        phase = _periodic_quarter_trig(root.exact_parameter)
        if phase is None:
            return None
        from ._root_bindings import revolution_offset_conic_frame

        frame = revolution_offset_conic_frame(surface, profile_parameter)
        if frame is None:
            return None
        _, center, first_axis, second_axis, radius = frame
        cosine, sine = phase
        return tuple(
            center[index]
            + radius * (cosine * first_axis[index] + sine * second_axis[index])
            for index in range(3)
        )
    return None


def _offset_sphere_revolution_harmonics(curve: AbstractCurve, /) -> _Harmonics | None:
    if (
        not isinstance(curve, SurfaceIsoparametricCurve)
        or curve.fixed_axis != 0
        or curve.fixed_value != 0.0
    ):
        return None
    surface = curve.surface
    if not isinstance(surface, OffsetSurface):
        return None
    revolution = surface.base
    if not isinstance(revolution, RevolutionSurface):
        return None
    profile = revolution.curve
    if not isinstance(profile, CircleCurve):
        return None
    witness = surface.exact_sphere_revolution_witness()
    if witness is None:
        return None
    center, radius, _, frame_sign = witness
    if (
        len(center) != 3
        or radius <= 0
        or frame_sign not in (-1, 1)
        or profile.center_exact != center
        or profile.radius_exact != radius
    ):
        return None
    offset_radius = radius + frame_sign * surface.distance_exact
    if offset_radius <= 0:
        return None
    return (
        center,
        tuple(offset_radius * value for value in profile.first_axis_exact),
        tuple(offset_radius * value for value in profile.second_axis_exact),
    )


def _periodic_endpoint_squared(
    curve: AbstractCurve,
    root: NativePeriodEndpoint,
    parameter: float,
    vertex: np.ndarray,
    /,
) -> Fraction:
    if carrier_payload(root.curve) != carrier_payload(curve):
        raise ValueError("A periodic endpoint root must bind the unchanged source curve.")
    lower, upper = root.parameter_enclosure()
    if not lower <= parameter <= upper:
        raise ValueError("A numeric endpoint must realize its exact native-period root.")
    exact = exact_curve_parameter_point(curve, root.exact_parameter)
    if exact is None:
        exact = _offset_revolution_isoparametric_point(curve, root)
    target = rational_vector(vertex)
    if exact is not None:
        return _squared_gap_upper(exact, target)
    if isinstance(curve, (CircleCurve, EllipseCurve)) and root.turns.denominator == 1:
        cosine, sine = _trig_intervals(
            fraction_down(root.rational), fraction_up(root.rational)
        )
        center, first, second = _conic_coefficients(curve)
        squared = Fraction(0)
        for base, first_coefficient, second_coefficient, expected in zip(
            center, first, second, target, strict=True
        ):
            first_term = _scaled_interval(first_coefficient, cosine)
            second_term = _scaled_interval(second_coefficient, sine)
            base_lower, base_upper = coefficient_interval(base - expected)
            lower = base_lower + first_term[0] + second_term[0]
            upper = base_upper + first_term[1] + second_term[1]
            magnitude = max(abs(lower), abs(upper))
            squared += magnitude * magnitude
        return squared
    raise ValueError("The periodic endpoint has no rationally proved source image.")


def _conic_root_squared(
    curve: AbstractCurve,
    root: ConicParameterRoot,
    parameter: float,
    vertex: np.ndarray,
    /,
) -> Fraction:
    """Squared gap from a vertex to the exact conic point an endpoint realizes."""
    if carrier_payload(root.conic) != carrier_payload(curve):
        raise ValueError("A conic endpoint root must bind the unchanged source curve.")
    if not root.parameter_lower <= parameter <= root.parameter_upper:
        raise ValueError("A numeric endpoint must realize its exact conic root.")
    return _squared_gap_upper(root.exact_point(), rational_vector(vertex))


def _endpoint_squared(
    curve: AbstractCurve,
    parameter: float,
    vertex: np.ndarray,
    /,
    *,
    root: NativePeriodEndpoint | ConicParameterRoot | None,
) -> Fraction:
    if isinstance(root, ConicParameterRoot):
        return _conic_root_squared(curve, root, parameter, vertex)
    if root is not None:
        return _periodic_endpoint_squared(curve, root, parameter, vertex)
    target = rational_vector(vertex)
    exact = exact_curve_parameter_point(curve, Fraction.from_float(float(parameter)))
    if exact is not None:
        return _squared_gap_upper(exact, target)
    harmonics = _offset_sphere_revolution_harmonics(curve)
    if harmonics is not None:
        center, first, second = harmonics
        return _harmonic_squared_upper(
            (_vector_difference(center, target), first, second),
            parameter,
            parameter,
        )
    affine = _affine_coefficients(curve)
    if affine is not None:
        origin, direction = affine
        scalar = Fraction.from_float(float(parameter))
        point = tuple(
            value + scalar * slope for value, slope in zip(origin, direction, strict=True)
        )
        return _squared_gap_upper(point, target)
    if isinstance(curve, (CircleCurve, EllipseCurve)):
        center, cosine, sine = _conic_coefficients(curve)
        trig = _trig_intervals(parameter, parameter)
        squared = Fraction(0)
        for base, first, second, expected in zip(
            center, cosine, sine, target, strict=True
        ):
            first_interval = _scaled_interval(first, trig[0])
            second_interval = _scaled_interval(second, trig[1])
            base_lower, base_upper = coefficient_interval(base - expected)
            lower = base_lower + first_interval[0] + second_interval[0]
            upper = base_upper + first_interval[1] + second_interval[1]
            magnitude = max(abs(lower), abs(upper))
            squared += magnitude * magnitude
        return squared
    if isinstance(curve, BSplineCurve):
        raise ValueError("The B-spline endpoint has no proved rational source image.")
    if isinstance(curve, SurfaceIsoparametricCurve) and isinstance(
        curve.surface, TorusPatch
    ):
        angles = [float(parameter), float(parameter)]
        angles[curve.fixed_axis] = curve.fixed_value
        return _torus_point_squared(curve.surface, angles[0], angles[1], target)
    if isinstance(curve, SurfaceIsoparametricCurve) and isinstance(
        curve.surface, CylinderPatch
    ):
        coordinates = [float(parameter), float(parameter)]
        coordinates[curve.fixed_axis] = curve.fixed_value
        cylinder = curve.surface
        cosine, sine = _trig_intervals(coordinates[0], coordinates[0])
        height = Fraction.from_float(coordinates[1])
        squared = Fraction(0)
        for index in range(3):
            first_term = _scaled_interval(
                cylinder.radius_exact * cylinder.first_axis_exact[index], cosine
            )
            second_term = _scaled_interval(
                cylinder.radius_exact * cylinder.second_axis_exact[index], sine
            )
            base_lower, base_upper = coefficient_interval(
                cylinder.origin_exact[index]
                + height * cylinder.axis_exact[index]
                - target[index]
            )
            lower = base_lower + first_term[0] + second_term[0]
            upper = base_upper + first_term[1] + second_term[1]
            squared += max(abs(lower), abs(upper)) ** 2
        return squared
    raise ValueError(f"No exact endpoint-gap proof supports {type(curve).__name__}.")


def _torus_point_squared(
    patch: TorusPatch, longitude: float, tube: float, target: _Vector, /
) -> Fraction:
    """Outward squared distance from a torus point at binary64 angles to a target."""
    cos_u, sin_u = _trig_intervals(longitude, longitude)
    cos_v, sin_v = _trig_intervals(tube, tube)
    major = coefficient_interval(patch.major_radius_exact)
    minor = coefficient_interval(patch.minor_radius_exact)
    tube_radial = _interval_product(minor, cos_v)
    ring = (major[0] + tube_radial[0], major[1] + tube_radial[1])
    height = _interval_product(minor, sin_v)
    squared = Fraction(0)
    for index in range(3):
        first_term = _scaled_interval(patch.first_axis_exact[index], cos_u)
        second_term = _scaled_interval(patch.second_axis_exact[index], sin_u)
        radial = (first_term[0] + second_term[0], first_term[1] + second_term[1])
        products = _interval_product(ring, radial)
        axial = _scaled_interval(patch.axis_exact[index], height)
        base_lower, base_upper = coefficient_interval(
            patch.center_exact[index] - target[index]
        )
        lower = base_lower + products[0] + axial[0]
        upper = base_upper + products[1] + axial[1]
        magnitude = max(abs(lower), abs(upper))
        squared += magnitude * magnitude
    return squared


def endpoint_gap_bounds(
    curve: AbstractCurve,
    parameters: tuple[float, float],
    vertices: np.ndarray,
    *,
    endpoint_roots: tuple[
        NativePeriodEndpoint | ConicParameterRoot | None,
        NativePeriodEndpoint | ConicParameterRoot | None,
    ] = (None, None),
) -> tuple[float, float]:
    """Return outward Euclidean endpoint-gap bounds without moving source vertices."""
    if len(parameters) != 2 or not all(np.isfinite(value) for value in parameters):
        raise ValueError("Endpoint parameters must be two finite binary64 values.")
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != (2, curve.ambient_dimension) or not np.all(np.isfinite(points)):
        raise ValueError(
            "Endpoint vertices must be finite points matching the source curve."
        )
    if len(endpoint_roots) != 2:
        raise ValueError("Endpoint roots must contain one source binding per endpoint.")
    values: list[float] = []
    for index in range(2):
        parameter = parameters[index]
        vertex: np.ndarray = points[index]
        root = endpoint_roots[index]
        values.append(
            square_root_interval(_endpoint_squared(curve, parameter, vertex, root=root))[
                1
            ]
        )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("Endpoint-gap enclosures must be finite.")
    return values[0], values[1]


def _affine_face_coefficients(
    patch: PlanePatch | ExtrusionSurface | CylinderPatch,
    pcurve: AbstractCurve,
    /,
) -> tuple[_Vector, _Vector] | None:
    uv = _affine_coefficients(pcurve)
    if uv is None or len(uv[0]) != 2:
        return None
    origin_uv, direction_uv = uv
    if isinstance(patch, CylinderPatch):
        # The angle-zero generator lifts exactly: cos 0 = 1 and sin 0 = 0.
        vectors = (patch.origin_exact, patch.first_axis_exact, patch.axis_exact)
        if (
            origin_uv[0] != 0
            or direction_uv[0] != 0
            or not isinstance(patch.radius_exact, Fraction)
            or any(
                not all(isinstance(value, Fraction) for value in vector)
                for vector in vectors
            )
        ):
            return None
        cylinder_origin, first_axis, cylinder_axis = vectors
        return (
            tuple(
                cylinder_origin[axis]
                + patch.radius_exact * first_axis[axis]
                + origin_uv[1] * cylinder_axis[axis]
                for axis in range(3)
            ),
            tuple(direction_uv[1] * cylinder_axis[axis] for axis in range(3)),
        )
    if isinstance(patch, PlanePatch):
        surface_origin = rational_vector(np.asarray(patch.origin))
        first_axis = rational_vector(np.asarray(patch.first_axis))
        second_axis = rational_vector(np.asarray(patch.second_axis))
        return (
            tuple(
                surface_origin[axis]
                + origin_uv[0] * first_axis[axis]
                + origin_uv[1] * second_axis[axis]
                for axis in range(3)
            ),
            tuple(
                direction_uv[0] * first_axis[axis] + direction_uv[1] * second_axis[axis]
                for axis in range(3)
            ),
        )
    base = _affine_coefficients(patch.curve)
    if base is None or len(base[0]) != 3:
        return None
    direction = rational_vector(np.asarray(patch.direction))
    return (
        tuple(
            base[0][axis] + origin_uv[0] * base[1][axis] + origin_uv[1] * direction[axis]
            for axis in range(3)
        ),
        tuple(
            direction_uv[0] * base[1][axis] + direction_uv[1] * direction[axis]
            for axis in range(3)
        ),
    )


def _mapped_cylinder_plane_gap_squared(
    curve: SurfaceMappedCurve,
    pcurve: AbstractCurve,
    plane: PlanePatch,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    source = curve.surface
    uv = _affine_coefficients(curve.pcurve)
    lifted = _affine_face_coefficients(plane, pcurve)
    if not isinstance(source, CylinderPatch) or uv is None or lifted is None:
        return None
    uv_origin, uv_direction = uv
    if len(uv_origin) != 2 or uv_direction[0] != 0:
        return None
    angle_lower, angle_upper = coefficient_interval(uv_origin[0])
    cosine, sine = _trig_intervals(fraction_down(angle_lower), fraction_up(angle_upper))
    radius = source.radius_exact
    origin = source.origin_exact
    first_axis = source.first_axis_exact
    second_axis = source.second_axis_exact
    axis = source.axis_exact
    source_origin: list[tuple[Fraction, Fraction]] = []
    source_slope = tuple(uv_direction[1] * value for value in axis)
    for dimension in range(3):
        first_term = _scaled_interval(radius * first_axis[dimension], cosine)
        second_term = _scaled_interval(radius * second_axis[dimension], sine)
        axial_lower, axial_upper = coefficient_interval(
            origin[dimension] + uv_origin[1] * axis[dimension]
        )
        source_origin.append(
            (
                axial_lower + first_term[0] + second_term[0],
                axial_upper + first_term[1] + second_term[1],
            )
        )
    residual_origin: list[tuple[Fraction, Fraction]] = []
    residual_slope: list[ExactScalar] = []
    for dimension in range(3):
        lifted_lower, lifted_upper = coefficient_interval(lifted[0][dimension])
        residual_origin.append(
            (
                source_origin[dimension][0] - lifted_upper,
                source_origin[dimension][1] - lifted_lower,
            )
        )
        residual_slope.append(source_slope[dimension] - lifted[1][dimension])
    squared = Fraction(0)
    for base, slope in zip(residual_origin, residual_slope, strict=True):
        endpoint_intervals = []
        for parameter in (first, last):
            point_parameter = Fraction.from_float(float(parameter))
            slope_term = _scaled_interval(slope, (point_parameter, point_parameter))
            endpoint_intervals.append((base[0] + slope_term[0], base[1] + slope_term[1]))
        lower = min(interval[0] for interval in endpoint_intervals)
        upper = max(interval[1] for interval in endpoint_intervals)
        magnitude = max(abs(lower), abs(upper))
        squared += magnitude * magnitude
    return squared


def _revolution_circle_gap_squared(
    curve: CircleCurve,
    pcurve: AbstractCurve,
    source_region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    patch = source_region.patch
    profile = patch.curve if isinstance(patch, RevolutionSurface) else None
    if (
        not isinstance(patch, RevolutionSurface)
        or not isinstance(profile, CircleCurve)
        or profile.ambient_dimension != 3
        or not profile.native_period_symbol() == "two_pi"
        or patch.native_period_symbols() != ("two_pi", "two_pi")
        or curve.ambient_dimension != 3
        or curve.native_period_symbol() != "two_pi"
        or not np.isfinite(first)
        or not np.isfinite(last)
        or first > last
    ):
        return None
    source_domain = curve.parameter_domain
    if source_domain is not None and (
        first < source_domain[0] or last > source_domain[1]
    ):
        return None
    from ._correspondence import _surface_chart_coefficients

    chart = _surface_chart_coefficients(pcurve, source_region, first, last)
    if chart is None:
        return None
    origin, direction, _ = chart
    if (
        origin[0] != 0
        or direction != (Fraction(1), Fraction(0))
        or not isinstance(origin[1], Fraction)
    ):
        return None

    profile_parameter = origin[1]
    profile_lower = fraction_down(profile_parameter)
    profile_upper = fraction_up(profile_parameter)
    cosine_raw = interval_cos((np.asarray(profile_lower), np.asarray(profile_upper)))
    sine_raw = interval_sin((np.asarray(profile_lower), np.asarray(profile_upper)))
    cosine = (
        Fraction.from_float(float(cosine_raw[0])),
        Fraction.from_float(float(cosine_raw[1])),
    )
    sine = (
        Fraction.from_float(float(sine_raw[0])),
        Fraction.from_float(float(sine_raw[1])),
    )

    def add(
        left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
    ) -> tuple[Fraction, Fraction]:
        return left[0] + right[0], left[1] + right[1]

    def subtract(
        left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
    ) -> tuple[Fraction, Fraction]:
        return left[0] - right[1], left[1] - right[0]

    axis_origin = rational_vector(np.asarray(patch.axis_origin, dtype=np.float64))
    axis = rational_vector(np.asarray(patch.axis_direction, dtype=np.float64))
    if (
        len(axis_origin) != 3
        or len(axis) != 3
        or sum((value * value for value in axis), Fraction(0)) != 1
    ):
        return None
    profile_point = tuple(
        add(
            add(
                coefficient_interval(profile.center_exact[index]),
                _scaled_interval(
                    profile.radius_exact * profile.first_axis_exact[index],
                    cosine,
                ),
            ),
            _scaled_interval(
                profile.radius_exact * profile.second_axis_exact[index], sine
            ),
        )
        for index in range(3)
    )
    relative = tuple(
        subtract(profile_point[index], (axis_origin[index], axis_origin[index]))
        for index in range(3)
    )
    axial = (Fraction(0), Fraction(0))
    for coefficient, interval in zip(axis, relative, strict=True):
        axial = add(axial, _scaled_interval(coefficient, interval))
    center = tuple(
        add(
            (axis_origin[index], axis_origin[index]),
            _scaled_interval(axis[index], axial),
        )
        for index in range(3)
    )
    radial = tuple(
        subtract(relative[index], _scaled_interval(axis[index], axial))
        for index in range(3)
    )
    angular = (
        subtract(
            _scaled_interval(axis[1], radial[2]),
            _scaled_interval(axis[2], radial[1]),
        ),
        subtract(
            _scaled_interval(axis[2], radial[0]),
            _scaled_interval(axis[0], radial[2]),
        ),
        subtract(
            _scaled_interval(axis[0], radial[1]),
            _scaled_interval(axis[1], radial[0]),
        ),
    )
    source_center, source_cosine, source_sine = _conic_coefficients(curve)
    squared = Fraction(0)
    for index in range(3):
        coefficient_intervals = (
            subtract(coefficient_interval(source_center[index]), center[index]),
            subtract(coefficient_interval(source_cosine[index]), radial[index]),
            subtract(coefficient_interval(source_sine[index]), angular[index]),
        )
        magnitude = sum(
            (
                max(abs(interval[0]), abs(interval[1]))
                for interval in coefficient_intervals
            ),
            Fraction(0),
        )
        squared += magnitude * magnitude
    return squared


def _interval_product(
    left: tuple[Fraction, Fraction], right: tuple[Fraction, Fraction], /
) -> tuple[Fraction, Fraction]:
    products = (
        left[0] * right[0],
        left[0] * right[1],
        left[1] * right[0],
        left[1] * right[1],
    )
    return min(products), max(products)


def _point_trig(
    value: Fraction | PeriodOffset, /
) -> tuple[tuple[Fraction, Fraction], ...]:
    """Cosine/sine intervals at an exact ``Q + Q*(2*pi)`` angle.

    Quarter turns are exact; other angles get outward enclosures.
    """
    quarter = _periodic_quarter_trig(value)
    if quarter is not None:
        cosine, sine = quarter
        return (cosine, cosine), (sine, sine)
    lower, upper = period_scalar_bounds(value)
    return _trig_intervals(float(lower), float(upper))


def _torus_circle_gap_squared(
    curve: CircleCurve,
    pcurve: AbstractCurve,
    source_region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    """Bound a circle's distance to a torus coordinate circle ``(u0 + s t, v0)``
    or ``(u0, v0 + s t)`` with ``s = +-1`` and binary64 angles.

    The torus image of such a line is the exact circle with harmonics in
    ``cos/sin(u0)`` and ``cos/sin(v0)``; outward trigonometric intervals at the
    authored angles bound every harmonic coefficient difference, which bounds
    the gap uniformly in ``t``. An irrational latitude or a quarter-turn phase
    is therefore a certified (not exact) correspondence. A sphere chart is the
    torus of zero major radius whose minor circle is its meridian.
    """
    patch = source_region.patch
    if (
        not isinstance(patch, (TorusPatch, SpherePatch))
        or curve.ambient_dimension != 3
        or curve.native_period_symbol() != "two_pi"
        or not np.isfinite(first)
        or not np.isfinite(last)
        or first > last
    ):
        return None
    source_domain = curve.parameter_domain
    if source_domain is not None and (
        first < source_domain[0] or last > source_domain[1]
    ):
        return None
    from ._correspondence import periodic_affine_line_chart

    chart = periodic_affine_line_chart(pcurve, source_region, first, last)
    if chart is None:
        return None
    (u0, v0), direction = chart
    if direction in ((Fraction(1), Fraction(0)), (Fraction(-1), Fraction(0))):
        latitude, sense = True, direction[0]
    elif direction in ((Fraction(0), Fraction(1)), (Fraction(0), Fraction(-1))):
        latitude, sense = False, direction[1]
    else:
        return None
    cos_u, sin_u = _point_trig(u0)
    cos_v, sin_v = _point_trig(v0)
    if isinstance(patch, SpherePatch):
        major = (Fraction(0), Fraction(0))
        minor = coefficient_interval(Fraction.from_float(float(patch.radius)))
        first_axis, second_axis, axis, patch_center = (
            rational_vector(np.asarray(value, dtype=np.float64))
            for value in (patch.first_axis, patch.second_axis, patch.axis, patch.center)
        )
    else:
        major = coefficient_interval(patch.major_radius_exact)
        minor = coefficient_interval(patch.minor_radius_exact)
        first_axis = patch.first_axis_exact
        second_axis = patch.second_axis_exact
        axis = patch.axis_exact
        patch_center = patch.center_exact

    def add(*terms: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
        return sum((term[0] for term in terms), Fraction(0)), sum(
            (term[1] for term in terms), Fraction(0)
        )

    def negate(value: tuple[Fraction, Fraction], /) -> tuple[Fraction, Fraction]:
        return -value[1], -value[0]

    def axis_combination(
        cosine: tuple[Fraction, Fraction],
        sine: tuple[Fraction, Fraction],
        first_axis: tuple[ExactScalar, ...],
        second_axis: tuple[ExactScalar, ...],
    ) -> tuple[tuple[Fraction, Fraction], ...]:
        return tuple(
            add(
                _scaled_interval(first_axis[index], cosine),
                _scaled_interval(second_axis[index], sine),
            )
            for index in range(3)
        )

    # w = cos(u0) a1 + sin(u0) a2 and its quarter turn w' = -sin(u0) a1 + cos(u0) a2.
    radial = axis_combination(cos_u, sin_u, first_axis, second_axis)
    tangent = axis_combination(negate(sin_u), cos_u, first_axis, second_axis)
    axial = tuple(coefficient_interval(value) for value in axis)
    center = tuple(coefficient_interval(value) for value in patch_center)
    if latitude:
        ring = add(major, _interval_product(minor, cos_v))
        height = _interval_product(minor, sin_v)
        image = (
            tuple(
                add(center[index], _interval_product(height, axial[index]))
                for index in range(3)
            ),
            tuple(_interval_product(ring, radial[index]) for index in range(3)),
            tuple(
                _scaled_interval(sense, _interval_product(ring, tangent[index]))
                for index in range(3)
            ),
        )
    else:
        image = (
            tuple(
                add(center[index], _interval_product(major, radial[index]))
                for index in range(3)
            ),
            tuple(
                _interval_product(
                    minor,
                    add(
                        _interval_product(cos_v, radial[index]),
                        _interval_product(sin_v, axial[index]),
                    ),
                )
                for index in range(3)
            ),
            tuple(
                _scaled_interval(
                    sense,
                    _interval_product(
                        minor,
                        add(
                            _interval_product(negate(sin_v), radial[index]),
                            _interval_product(cos_v, axial[index]),
                        ),
                    ),
                )
                for index in range(3)
            ),
        )
    squared = Fraction(0)
    for index in range(3):
        magnitude = Fraction(0)
        for source, target in zip(_conic_coefficients(curve), image, strict=True):
            lower, upper = coefficient_interval(source[index])
            target_lower, target_upper = target[index]
            magnitude += max(abs(lower - target_upper), abs(upper - target_lower))
        squared += magnitude * magnitude
    return squared


def _rational_spline_line_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    source_region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    source = _affine_coefficients(curve)
    if source is None or len(source[0]) != 3:
        return None
    source_origin, source_direction = source
    from .._meshing_domain import _prepare_surface_spans
    from .._source_interpolation import (
        _exact_rational_span_interpolation_bounds,
        _exact_rational_surface_chart_breakpoints,
        _exact_rational_surface_point_bounds,
    )
    from ._boolean_core import _pcurve_execution_shadows_match, line_chart_coefficients

    if not _pcurve_execution_shadows_match(pcurve):
        return None
    chart = line_chart_coefficients(pcurve)
    if chart is None:
        return None
    chart_origin, chart_direction = chart
    if len(chart_origin) != 2 or len(chart_direction) != 2:
        return None
    origin_u, origin_v = chart_origin
    if not isinstance(origin_u, Fraction) or not isinstance(origin_v, Fraction):
        return None
    exact_origin = (origin_u, origin_v)
    exact_direction = (chart_direction[0], chart_direction[1])
    surface = source_region.patch
    spans = _prepare_surface_spans(surface)
    if spans is None:
        return None
    first_exact, last_exact = Fraction.from_float(first), Fraction.from_float(last)
    source_domain = curve.parameter_domain
    pcurve_domain = pcurve.parameter_domain
    if (
        source_domain is not None
        and (first < source_domain[0] or last > source_domain[1])
    ) or (
        pcurve_domain is not None
        and (first < pcurve_domain[0] or last > pcurve_domain[1])
    ):
        return None
    region_box = np.asarray(source_region.parameter_box, dtype=np.float64)
    if region_box.shape != (2, 2) or not np.all(np.isfinite(region_box)):
        return None
    region_exact = (
        tuple(Fraction.from_float(float(value)) for value in region_box[0]),
        tuple(Fraction.from_float(float(value)) for value in region_box[1]),
    )

    def chart_point(parameter: Fraction, /) -> tuple[Fraction, Fraction]:
        return (
            exact_origin[0] + parameter * exact_direction[0],
            exact_origin[1] + parameter * exact_direction[1],
        )

    def source_point(parameter: Fraction, /) -> tuple[ExactScalar, ...]:
        return tuple(
            source_origin[axis] + parameter * source_direction[axis] for axis in range(3)
        )

    def endpoint_gap_squared(parameter: Fraction, /) -> Fraction | None:
        uv = chart_point(parameter)
        if any(
            uv[axis] < region_exact[0][axis] or uv[axis] > region_exact[1][axis]
            for axis in range(2)
        ):
            return None
        target = _exact_rational_surface_point_bounds(spans, uv)
        if target is None or len(target) != 3:
            return None
        point = source_point(parameter)
        squared = Fraction(0)
        for value, bounds in zip(point, target, strict=True):
            lower, upper = coefficient_interval(value, bits=128)
            magnitude = max(
                abs(lower - bounds[0]),
                abs(lower - bounds[1]),
                abs(upper - bounds[0]),
                abs(upper - bounds[1]),
            )
            squared += magnitude * magnitude
        return squared

    if first_exact == last_exact:
        endpoint = endpoint_gap_squared(first_exact)
        return None if endpoint is None else endpoint
    breakpoints = _exact_rational_surface_chart_breakpoints(
        spans, exact_origin, exact_direction, first_exact, last_exact
    )
    if breakpoints is None:
        return None
    maximum_upper = Fraction(0)
    for lower, upper in zip(breakpoints[:-1], breakpoints[1:], strict=True):
        lower_uv, upper_uv = chart_point(lower), chart_point(upper)
        lower_gap = endpoint_gap_squared(lower)
        upper_gap = endpoint_gap_squared(upper)
        if lower_gap is None or upper_gap is None:
            return None
        lower_distance = Fraction.from_float(square_root_interval(lower_gap)[1])
        upper_distance = Fraction.from_float(square_root_interval(upper_gap)[1])
        chart_triangle = (
            lower_uv,
            upper_uv,
            upper_uv,
        )
        interpolation = _exact_rational_span_interpolation_bounds(
            spans, (chart_triangle,)
        )
        if interpolation.shape != (1,) or not np.isfinite(interpolation[0]):
            return None
        interpolation_upper = Fraction.from_float(float(interpolation[0]))
        maximum_upper = max(
            maximum_upper,
            max(lower_distance, upper_distance) + interpolation_upper,
        )
    return maximum_upper * maximum_upper


def native_face_gap_squared(
    curve: AbstractCurve,
    pcurve: AbstractCurve,
    source_region: SurfaceRegion,
    first: float,
    last: float,
    /,
) -> Fraction | None:
    """Recompute exact source gaps over the authoritative face region.

    ``None`` signals an unsupported source correspondence, never a zero gap.
    """
    if not isinstance(source_region, SurfaceRegion):
        raise TypeError("source_region must be a SurfaceRegion.")
    if not np.isfinite(first) or not np.isfinite(last) or first > last:
        raise ValueError("Face-gap parameters must be finite and ordered.")
    from ._boolean_core import _pcurve_execution_shadows_match
    from ._correspondence import (
        _affine_bspline_pcurve,
        _periodic_pcurve_source,
        exact_source_lift,
    )

    if exact_source_lift(curve, pcurve, source_region, first, last):
        return Fraction(0)
    periodic_source = _periodic_pcurve_source(pcurve, source_region.patch)
    if (
        periodic_source is not None
        and periodic_source[0] is not pcurve
        and _pcurve_execution_shadows_match(pcurve)
    ):
        # Whole native two-pi shifts on the owning patch leave every surface
        # point unchanged, so the unshifted chart has the identical gap.
        periodic_gap = native_face_gap_squared(
            curve, periodic_source[0], source_region, first, last
        )
        if periodic_gap is not None:
            return periodic_gap
    affine_pcurve = _affine_bspline_pcurve(pcurve, first, last)
    if affine_pcurve is not None:
        affine_gap = native_face_gap_squared(
            curve, affine_pcurve, source_region, first, last
        )
        if affine_gap is not None:
            return affine_gap
    patch = source_region.patch
    from ._tspline import TSplineSurfacePatch

    if isinstance(patch, (BSplineSurfacePatch, TSplineSurfacePatch)):
        from ._spline_face_gap import rational_spline_face_gap_squared

        # The exact Bernstein enclosure of the composed difference refines to
        # within a factor of an attained gap (exactly zero for exact identities);
        # the endpoint-plus-interpolation line bound carries binary64 rounding
        # and only serves correspondences the exact enclosure does not support.
        composed_gap = rational_spline_face_gap_squared(
            curve, pcurve, source_region, first, last
        )
        if composed_gap is not None:
            return composed_gap
        spline_gap = _rational_spline_line_gap_squared(
            curve, pcurve, source_region, first, last
        )
        if spline_gap is not None:
            return spline_gap
    revolution_gap = (
        _revolution_circle_gap_squared(curve, pcurve, source_region, first, last)
        if isinstance(curve, CircleCurve)
        else None
    )
    if revolution_gap is not None:
        return revolution_gap
    if isinstance(curve, CircleCurve) and isinstance(patch, (TorusPatch, SpherePatch)):
        torus_gap = _torus_circle_gap_squared(curve, pcurve, source_region, first, last)
        if torus_gap is not None:
            return torus_gap
    if isinstance(patch, (PlanePatch, ExtrusionSurface, CylinderPatch)):
        lifted = _affine_face_coefficients(patch, pcurve)
        source = _affine_coefficients(curve)
        if lifted is not None and source is not None and len(source[0]) == 3:
            residual = tuple(a - b for a, b in zip(source[0], lifted[0], strict=True))
            slope = tuple(a - b for a, b in zip(source[1], lifted[1], strict=True))
            zero = (Fraction(0), Fraction(0), Fraction(0))
            return max(
                _squared_gap_upper(
                    tuple(
                        origin + Fraction.from_float(float(parameter)) * direction
                        for origin, direction in zip(residual, slope, strict=True)
                    ),
                    zero,
                )
                for parameter in (first, last)
            )
    if isinstance(curve, SurfaceMappedCurve) and isinstance(patch, PlanePatch):
        return _mapped_cylinder_plane_gap_squared(curve, pcurve, patch, first, last)
    if (
        isinstance(curve, BSplineCurve)
        and isinstance(pcurve, BSplineCurve)
        and isinstance(patch, PlanePatch)
        and curve.degree == pcurve.degree
        and curve.exact_knots == pcurve.exact_knots
        and curve.exact_weights == pcurve.exact_weights
    ):
        origin = rational_vector(np.asarray(patch.origin))
        axes = (
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )
        source_controls = curve.exact_control_points
        chart_controls = pcurve.exact_control_points
        if len(source_controls) == len(chart_controls) and all(
            tuple(
                origin[axis]
                + chart_controls[index][0] * axes[0][axis]
                + chart_controls[index][1] * axes[1][axis]
                for axis in range(3)
            )
            == source_controls[index]
            for index in range(len(source_controls))
        ):
            return Fraction(0)
    if (
        isinstance(curve, (CircleCurve, EllipseCurve))
        and isinstance(pcurve, (CircleCurve, EllipseCurve))
        and isinstance(patch, PlanePatch)
    ):
        source = _conic_coefficients(curve)
        chart = _conic_coefficients(pcurve)
        origin = rational_vector(np.asarray(patch.origin))
        axes = (
            rational_vector(np.asarray(patch.first_axis)),
            rational_vector(np.asarray(patch.second_axis)),
        )
        lifted: _Harmonics = (
            tuple(
                origin[dimension]
                + sum(
                    (axes[axis][dimension] * chart[0][axis] for axis in range(2)),
                    Fraction(0),
                )
                for dimension in range(3)
            ),
            tuple(
                sum(
                    (axes[axis][dimension] * chart[1][axis] for axis in range(2)),
                    Fraction(0),
                )
                for dimension in range(3)
            ),
            tuple(
                sum(
                    (axes[axis][dimension] * chart[2][axis] for axis in range(2)),
                    Fraction(0),
                )
                for dimension in range(3)
            ),
        )
        difference: _Harmonics = (
            _vector_difference(source[0], lifted[0]),
            _vector_difference(source[1], lifted[1]),
            _vector_difference(source[2], lifted[2]),
        )
        return _harmonic_squared_upper(difference, first, last)
    if isinstance(patch, PlanePatch):
        from ._spline_face_gap import rational_spline_face_gap_squared

        return rational_spline_face_gap_squared(curve, pcurve, source_region, first, last)
    return None


__all__ = ["endpoint_gap_bounds", "native_face_gap_squared"]
