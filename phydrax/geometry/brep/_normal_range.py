#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous outward normal-angle enclosures over native B-Rep face trims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, floor, inf, pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._validation import finite_real_scalar
from ...discretization._exact_algebraic_geometry import (
    coefficient_interval,
    ExactScalar,
)
from .._atlas import TrimDomain, TrimEnclosures
from .._interval_enclosure import (
    Interval,
    interval_add,
    interval_cos,
    interval_divide,
    interval_multiply,
    interval_sin,
    interval_sqrt,
    interval_square,
    interval_subtract,
)
from .._rational import fraction_down, fraction_up, rational_vector
from ._edge_classification import _material_orientations
from ._model import BRepEntityId, BRepModel
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    BSplineSurfacePatch,
    CircleCurve,
    ConePatch,
    CylinderPatch,
    ExactLineCurve,
    ExtrusionSurface,
    LineCurve,
    OffsetSurface,
    PlanePatch,
    RevolutionSurface,
    RuledSurface,
    SpherePatch,
)
from ._source_correspondence import _affine_curve_coefficients


BRepNormalStatus: TypeAlias = Literal[
    "positive", "insufficient", "vertical", "undercut", "unresolved"
]
# Why a returned cell stayed undecided after refinement:
# ``undefined_normal``: no positive lower bound on |S_u x S_v| and no singular
# point certificate at binary64 parameter resolution; ``unverified_isolated_zero``:
# the cell holds the unique zero of two cross-product components but the third
# is not decided there; ``continuity_unestablished``: C1 continuity is not
# established and the carrier declares no break line to split at;
# ``orientation_undefined``: the face has no material orientation;
# ``angle_unresolved``: the angle enclosure still meets a decision value at
# binary64 parameter resolution.
BRepNormalUnresolvedReason: TypeAlias = Literal[
    "undefined_normal",
    "unverified_isolated_zero",
    "continuity_unestablished",
    "orientation_undefined",
    "angle_unresolved",
]
BRepNormalSingularityKind: TypeAlias = Literal[
    "declared_degenerate_edge", "isolated_cross_product_zero"
]

_PI = float(np.nextafter(pi, np.inf))
_HALF_PI = pi / 2.0


@final
@dataclass(frozen=True, slots=True)
class BRepNormalSingularity:
    """Exact evidence that a singular cell contains a point where S_u x S_v = 0.

    ``declared_degenerate_edge``: the cell meets a degenerate edge p-curve whose
    exact pole collapse the model proved (or a collapsed clamped spline boundary
    row); ``parameter_box`` is the exact box of that p-curve inside the cell and
    ``point`` its pole. ``isolated_cross_product_zero``: a Krawczyk box isolates
    the unique zero of two cross-product components and the third vanishes there
    exactly; ``parameter_box`` is that exact point and ``point`` its exact image.
    """

    kind: BRepNormalSingularityKind
    parameter_box: tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
    point: tuple[ExactScalar, ...] | None
    edge_id: BRepEntityId | None


@final
class BRepNormalRange(StrictModule):
    """Conservative normal angle, singular evidence and unresolved cells for one face.

    ``unresolved_cells`` are parameter boxes with one ``unresolved_reasons`` entry
    each. ``singular_cells`` are parameter boxes certified to contain a point with
    ``S_u x S_v = 0`` (one ``singularities`` record each); ``singular_points``
    encloses that point in space. A singular cell contributes ``[0, pi]``.
    """

    face_id: BRepEntityId = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    status: BRepNormalStatus = eqx.field(static=True)
    angle_enclosure: Array
    unresolved_cells: Array
    unresolved_reasons: tuple[BRepNormalUnresolvedReason, ...] = eqx.field(static=True)
    singular_cells: Array
    singular_points: Array
    singularities: tuple[BRepNormalSingularity, ...] = eqx.field(static=True)


def _direction(value: ArrayLike | tuple[float, float, float], /) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError("normal.direction must be a finite three-vector.")
    scale = float(np.max(np.abs(direction)))
    if scale == 0.0:
        raise ValueError("normal.direction must be nonzero.")
    _, exponent = np.frexp(scale)
    scaled = np.ldexp(direction, -int(exponent))
    if not np.array_equal(np.ldexp(scaled, int(exponent)), direction):
        raise ValueError("normal.direction has unsupported binary64 dynamic range.")
    return scaled


def _vector_interval(values: np.ndarray, /) -> Interval:
    host = np.asarray(values, dtype=np.float64)
    return host, host


def _scalar_interval(lower: float, upper: float, /) -> Interval:
    return np.float64(lower), np.float64(upper)


def _negate(value: Interval, /) -> Interval:
    return np.negative(value[1]), np.negative(value[0])


def _component(value: Interval, axis: int, /) -> Interval:
    return np.asarray(value[0])[axis], np.asarray(value[1])[axis]


def _cross(first: Interval, second: Interval, /) -> Interval:
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for axis in range(3):
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        positive = interval_multiply(
            _component(first, next_axis), _component(second, last_axis)
        )
        negative = interval_multiply(
            _component(first, last_axis), _component(second, next_axis)
        )
        component = interval_subtract(positive, negative)
        lower.append(np.asarray(component[0]))
        upper.append(np.asarray(component[1]))
    return np.stack(lower), np.stack(upper)


def _dot(first: Interval, second: Interval, /) -> Interval:
    result = _scalar_interval(0.0, 0.0)
    for axis in range(3):
        result = interval_add(
            result,
            interval_multiply(_component(first, axis), _component(second, axis)),
        )
    return np.asarray(result[0]), np.asarray(result[1])


def _norm(value: Interval, /) -> tuple[float, float]:
    squared = _scalar_interval(0.0, 0.0)
    for axis in range(3):
        squared = interval_add(squared, interval_square(_component(value, axis)))
    length = interval_sqrt(squared)
    return float(length[0]), float(length[1])


type _ExactVector = tuple[Fraction, Fraction, Fraction]
type _ExactBox = tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]
# Exact polynomial in span-relative parameters ``(s, t)``, keyed by exponents.
type _Poly = dict[tuple[int, int], Fraction]


@dataclass(slots=True)
class _Cell:
    """One parameter cell: its angle enclosure and singular-set decision state."""

    box: np.ndarray
    angle: tuple[float, float, bool]
    singular: BRepNormalSingularity | None = None
    # Certified to contain no point with S_u x S_v = 0 (inherited by children).
    free: bool = False
    # Holds the unique zero of two cross-product components; third undecided.
    unverified: bool = False
    terminal: BRepNormalUnresolvedReason | None = None


@dataclass(frozen=True, slots=True)
class _MeridianProfile:
    """Exact data of a revolution profile lying in one plane through its axis.

    With ``N`` the meridian-plane normal and ``m = N x axis``, the unnormalized
    surface normal is ``(rho(v) / |N|^2) R_u(N x c'(v))`` where
    ``rho(v) = (c(v) - origin) . m``. Its direction is regular through the axis.
    """

    axis: _ExactVector
    axis_slack: float
    circular: bool
    radial: tuple[Fraction, Fraction, Fraction]
    tangent: tuple[_ExactVector, _ExactVector, _ExactVector]
    origin: _ExactVector
    points: tuple[_ExactVector, _ExactVector, _ExactVector]


@dataclass(frozen=True, slots=True)
class _NormalEnclosure:
    vector: Interval
    magnitude: tuple[float, float]
    slack: float = 0.0


def _exact_vector(values: object, /) -> _ExactVector | None:
    if not isinstance(values, tuple) or len(values) != 3:
        return None
    if not all(isinstance(value, Fraction) for value in values):
        return None
    return values[0], values[1], values[2]


def _exact_cross(first: _ExactVector, second: _ExactVector, /) -> _ExactVector:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


def _exact_dot(first: _ExactVector, second: _ExactVector, /) -> Fraction:
    return first[0] * second[0] + first[1] * second[1] + first[2] * second[2]


def _exact_scale(scale: Fraction, vector: _ExactVector, /) -> _ExactVector:
    return scale * vector[0], scale * vector[1], scale * vector[2]


def _exact_interval(values: _ExactVector, /) -> Interval:
    return (
        np.asarray([fraction_down(value) for value in values], dtype=np.float64),
        np.asarray([fraction_up(value) for value in values], dtype=np.float64),
    )


_ZERO_VECTOR: _ExactVector = (Fraction(0), Fraction(0), Fraction(0))


def _meridian_profile(patch: RevolutionSurface, /) -> _MeridianProfile | None:
    origin = rational_vector(np.asarray(patch.axis_origin, dtype=np.float64))
    axis = rational_vector(np.asarray(patch.axis_direction, dtype=np.float64))
    exact_axis: _ExactVector = (axis[0], axis[1], axis[2])
    curve = patch.curve
    if isinstance(curve, (LineCurve, ExactLineCurve)):
        if isinstance(curve, LineCurve):
            start = _exact_vector(curve.origin_exact)
            step = _exact_vector(curve.direction_exact)
        else:
            start = _exact_vector(curve.first_exact)
            last = _exact_vector(curve.last_exact)
            step = (
                None
                if start is None or last is None
                else (last[0] - start[0], last[1] - start[1], last[2] - start[2])
            )
        if start is None or step is None or step == _ZERO_VECTOR:
            return None
        relative = (start[0] - origin[0], start[1] - origin[1], start[2] - origin[2])
        if _exact_dot(exact_axis, _exact_cross(relative, step)) != 0:
            return None
        plane = _exact_cross(exact_axis, relative)
        if plane == _ZERO_VECTOR:
            plane = _exact_cross(exact_axis, step)
        if plane == _ZERO_VECTOR:
            return None
        radial = _exact_cross(plane, exact_axis)
        radius = (_exact_dot(relative, radial), _exact_dot(step, radial), Fraction(0))
        tangent = (_exact_cross(plane, step), _ZERO_VECTOR, _ZERO_VECTOR)
        points = (relative, step, _ZERO_VECTOR)
        circular = False
    elif isinstance(curve, CircleCurve):
        center = _exact_vector(curve.center_exact)
        first = _exact_vector(curve.first_axis_exact)
        second = _exact_vector(curve.second_axis_exact)
        scale = curve.radius_exact
        if center is None or first is None or second is None:
            return None
        if not isinstance(scale, Fraction):
            return None
        plane = _exact_cross(first, second)
        relative = (center[0] - origin[0], center[1] - origin[1], center[2] - origin[2])
        if (
            plane == _ZERO_VECTOR
            or _exact_dot(exact_axis, plane) != 0
            or _exact_dot(relative, plane) != 0
        ):
            return None
        radial = _exact_cross(plane, exact_axis)
        radius = (
            _exact_dot(relative, radial),
            scale * _exact_dot(first, radial),
            scale * _exact_dot(second, radial),
        )
        tangent = (
            _ZERO_VECTOR,
            _exact_scale(scale, _exact_cross(plane, second)),
            _exact_scale(-scale, _exact_cross(plane, first)),
        )
        points = (relative, _exact_scale(scale, first), _exact_scale(scale, second))
        circular = True
    else:
        return None
    unit = _exact_dot(exact_axis, exact_axis) == 1
    return _MeridianProfile(
        exact_axis,
        0.0 if unit else 64.0 * float(np.finfo(np.float64).eps),
        circular,
        radius,
        tangent,
        (origin[0], origin[1], origin[2]),
        points,
    )


def _meridian_point_box(profile: _MeridianProfile, box: np.ndarray, /) -> np.ndarray:
    """Enclose the surface points over ``box`` for singular-region reporting."""
    angle = (np.float64(box[0, 0]), np.float64(box[1, 0]))
    parameter = (np.float64(box[0, 1]), np.float64(box[1, 1]))
    cosine, sine = interval_cos(angle), interval_sin(angle)
    first, second = (
        (interval_cos(parameter), interval_sin(parameter))
        if profile.circular
        else (parameter, _scalar_interval(0.0, 0.0))
    )
    points = interval_add(
        interval_add(
            _exact_interval(profile.origin),
            _rotated(profile.points[0], profile.axis, cosine, sine),
        ),
        interval_add(
            interval_multiply(
                _rotated(profile.points[1], profile.axis, cosine, sine), first
            ),
            interval_multiply(
                _rotated(profile.points[2], profile.axis, cosine, sine), second
            ),
        ),
    )
    lower = np.asarray(points[0], dtype=np.float64)
    upper = np.asarray(points[1], dtype=np.float64)
    slack = profile.axis_slack * float(np.max(np.abs(np.stack((lower, upper)))))
    return np.stack((lower - slack, upper + slack))


def _rotated(
    vector: _ExactVector, axis: _ExactVector, cosine: Interval, sine: Interval, /
) -> Interval:
    """Enclose ``R_u vector`` with the axial component kept exact."""
    scale = _exact_dot(axis, vector) / _exact_dot(axis, axis)
    axial = _exact_scale(scale, axis)
    normal = (vector[0] - axial[0], vector[1] - axial[1], vector[2] - axial[2])
    return interval_add(
        interval_add(
            _exact_interval(axial),
            interval_multiply(_exact_interval(normal), cosine),
        ),
        interval_multiply(_exact_interval(_exact_cross(axis, normal)), sine),
    )


def _meridian_normal(
    profile: _MeridianProfile, box: np.ndarray, /
) -> _NormalEnclosure | None:
    parameter = (np.float64(box[0, 1]), np.float64(box[1, 1]))
    angle = (np.float64(box[0, 0]), np.float64(box[1, 0]))
    if profile.circular:
        cosine, sine = interval_cos(parameter), interval_sin(parameter)
        constant, cosine_part, sine_part = profile.radial
        radius = interval_add(
            interval_add(
                _scalar_interval(fraction_down(constant), fraction_up(constant)),
                interval_multiply(
                    _scalar_interval(
                        fraction_down(cosine_part), fraction_up(cosine_part)
                    ),
                    cosine,
                ),
            ),
            interval_multiply(
                _scalar_interval(fraction_down(sine_part), fraction_up(sine_part)),
                sine,
            ),
        )
        lower_radius, upper_radius = float(radius[0]), float(radius[1])
        if lower_radius >= 0.0 and upper_radius > 0.0:
            sign = 1
        elif upper_radius <= 0.0 and lower_radius < 0.0:
            sign = -1
        else:
            return None
        tangent = interval_add(
            interval_multiply(_exact_interval(profile.tangent[1]), cosine),
            interval_multiply(_exact_interval(profile.tangent[2]), sine),
        )
        angle_cosine, angle_sine = interval_cos(angle), interval_sin(angle)
        rotated = interval_add(
            interval_multiply(
                _rotated(profile.tangent[1], profile.axis, angle_cosine, angle_sine),
                cosine,
            ),
            interval_multiply(
                _rotated(profile.tangent[2], profile.axis, angle_cosine, angle_sine),
                sine,
            ),
        )
    else:
        constant, slope, _ = profile.radial
        endpoints = tuple(
            constant + slope * Fraction.from_float(float(value)) for value in parameter
        )
        if all(value >= 0 for value in endpoints) and any(
            value > 0 for value in endpoints
        ):
            sign = 1
        elif all(value <= 0 for value in endpoints) and any(
            value < 0 for value in endpoints
        ):
            sign = -1
        else:
            return None
        tangent = _exact_interval(profile.tangent[0])
        rotated = _rotated(
            profile.tangent[0], profile.axis, interval_cos(angle), interval_sin(angle)
        )
    magnitude = _norm(tangent)
    return _NormalEnclosure(
        rotated if sign > 0 else _negate(rotated), magnitude, profile.axis_slack
    )


def _normal_interval(
    patch: AbstractSurfacePatch, box: np.ndarray, /
) -> _NormalEnclosure | None:
    if isinstance(patch, RevolutionSurface):
        profile = _meridian_profile(patch)
        if profile is not None:
            return _meridian_normal(profile, box)
    vector = _cartesian_normal_interval(patch, box)
    return None if vector is None else _NormalEnclosure(vector, _norm(vector))


def _cartesian_normal_interval(
    patch: AbstractSurfacePatch, box: np.ndarray, /
) -> Interval | None:
    if not patch.is_c1_on(box):
        return None
    if isinstance(patch, PlanePatch):
        derivatives = patch.derivative_bounds(box)
        return _cross(
            (derivatives[0][:, 0], derivatives[1][:, 0]),
            (derivatives[0][:, 1], derivatives[1][:, 1]),
        )
    if isinstance(patch, CylinderPatch):
        radius = float(np.asarray(patch.radius))
        if radius == 0.0:
            return None
        angle = (box[0, 0], box[1, 0])
        radial = interval_add(
            interval_multiply(
                interval_cos(angle),
                _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(angle),
                _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        return radial if radius > 0.0 else _negate(radial)
    if isinstance(patch, ConePatch):
        angle = (box[0, 0], box[1, 0])
        radial = interval_add(
            interval_multiply(
                interval_cos(angle),
                _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(angle),
                _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        semi = float(np.asarray(patch.semi_angle))
        cosine = interval_cos(_scalar_interval(semi, semi))
        sine = interval_sin(_scalar_interval(semi, semi))
        axial = interval_multiply(_negate(sine), _vector_interval(np.asarray(patch.axis)))
        normal = interval_add(interval_multiply(cosine, radial), axial)
        tangent = interval_divide(sine, cosine)
        reference_radius = float(np.asarray(patch.reference_radius))
        radius = interval_add(
            _scalar_interval(reference_radius, reference_radius),
            interval_multiply((box[0, 1], box[1, 1]), tangent),
        )
        lower_radius, upper_radius = float(radius[0]), float(radius[1])
        if lower_radius <= 0.0 <= upper_radius:
            return None
        return normal if lower_radius > 0.0 else _negate(normal)
    if isinstance(patch, SpherePatch):
        radius = float(np.asarray(patch.radius))
        if radius == 0.0:
            return None
        longitude = (box[0, 0], box[1, 0])
        latitude = (box[0, 1], box[1, 1])
        equatorial = interval_add(
            interval_multiply(
                interval_cos(longitude),
                _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(longitude),
                _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        radial = interval_add(
            interval_multiply(interval_cos(latitude), equatorial),
            interval_multiply(
                interval_sin(latitude),
                _vector_interval(np.asarray(patch.axis)),
            ),
        )
        return radial if radius > 0.0 else _negate(radial)
    derivatives = patch.derivative_bounds(box)
    return _cross(
        (derivatives[0][:, 0], derivatives[1][:, 0]),
        (derivatives[0][:, 1], derivatives[1][:, 1]),
    )


def _angle(
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    orientation: float,
    direction: np.ndarray,
    /,
) -> tuple[float, float, bool]:
    """Return the angle enclosure and whether the normal is undefined on ``box``."""
    normal = _normal_interval(patch, box)
    if normal is None or not np.isfinite(orientation):
        return 0.0, _PI, True
    vector = _negate(normal.vector) if orientation < 0.0 else normal.vector
    magnitude = normal.magnitude
    if (
        not np.isfinite(magnitude[0])
        or not np.isfinite(magnitude[1])
        or magnitude[0] <= 0.0
    ):
        return 0.0, _PI, True
    dot = _dot(vector, _vector_interval(direction))
    direction_norm = _norm(_vector_interval(direction))
    denominator = interval_multiply(
        _scalar_interval(*magnitude), _scalar_interval(*direction_norm)
    )
    cosine = interval_divide(dot, denominator)
    lower_cosine = float(np.clip(float(cosine[0]) - normal.slack, -1.0, 1.0))
    upper_cosine = float(np.clip(float(cosine[1]) + normal.slack, -1.0, 1.0))
    lower = float(np.nextafter(np.arccos(upper_cosine), -np.inf))
    upper = float(np.nextafter(np.arccos(lower_cosine), np.inf))
    return max(0.0, lower), min(_PI, upper), False


def _exact_vertical(patch: AbstractSurfacePatch, direction: np.ndarray, /) -> bool:
    exact_direction = tuple(Fraction.from_float(float(value)) for value in direction)
    if isinstance(patch, RevolutionSurface):
        profile = _meridian_profile(patch)
        if profile is None or profile.circular:
            return False
        axis = rational_vector(np.asarray(patch.axis_direction, dtype=np.float64))
        exact_axis: _ExactVector = (axis[0], axis[1], axis[2])
        pull: _ExactVector = (exact_direction[0], exact_direction[1], exact_direction[2])
        # A meridian line parallel to the axis has radial normals only; they are all
        # orthogonal to the pull exactly when the pull is parallel to the axis.
        return profile.radial[1] == 0 and _exact_cross(pull, exact_axis) == _ZERO_VECTOR
    if not isinstance(patch, (PlanePatch, CylinderPatch)):
        return False
    if isinstance(patch, PlanePatch):
        first = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.first_axis)
        )
        second = tuple(
            Fraction.from_float(float(value)) for value in np.asarray(patch.second_axis)
        )
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        return (
            sum(
                (a * b for a, b in zip(normal, exact_direction, strict=True)), Fraction(0)
            )
            == 0
        )
    first = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(patch.first_axis)
    )
    second = tuple(
        Fraction.from_float(float(value)) for value in np.asarray(patch.second_axis)
    )
    return all(
        sum((a * b for a, b in zip(axis, exact_direction, strict=True)), Fraction(0)) == 0
        for axis in (first, second)
    )


def _trim_excludes(
    model: BRepModel,
    face: int,
    box: np.ndarray,
    enclosures: TrimEnclosures | None,
    /,
) -> bool:
    """Whether ``box`` is certified to lie outside the face trim."""
    domain: TrimDomain | None = model.trim_domains[face]
    if domain is None:
        return False
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
    extent = bounds[1] - bounds[0]
    normalized = np.stack(((box[0] - bounds[0]) / extent, (box[1] - bounds[0]) / extent))
    result = domain.classify_boxes(normalized[None, ...], enclosures=enclosures)
    return bool(np.asarray(result.resolved[0])) and not bool(np.asarray(result.inside[0]))


def _exact_box(box: np.ndarray, /) -> _ExactBox:
    return (
        (Fraction.from_float(float(box[0, 0])), Fraction.from_float(float(box[1, 0]))),
        (Fraction.from_float(float(box[0, 1])), Fraction.from_float(float(box[1, 1]))),
    )


def _box_contains_point(box: _ExactBox, point: tuple[Fraction, Fraction], /) -> bool:
    return all(box[axis][0] <= point[axis] <= box[axis][1] for axis in (0, 1))


def _box_contains_box(outer: _ExactBox, inner: _ExactBox, /) -> bool:
    return all(
        outer[axis][0] <= inner[axis][0] and inner[axis][1] <= outer[axis][1]
        for axis in (0, 1)
    )


def _exact_point_box(point: tuple[ExactScalar, ...], /) -> np.ndarray:
    lower: list[float] = []
    upper: list[float] = []
    for value in point:
        first, last = (
            (value, value)
            if isinstance(value, Fraction)
            else coefficient_interval(value, bits=128)
        )
        lower.append(fraction_down(first))
        upper.append(fraction_up(last))
    return np.asarray((lower, upper), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class _DeclaredLine:
    """Parameter segment the source declares to collapse to one exact point."""

    start: tuple[Fraction, Fraction]
    end: tuple[Fraction, Fraction]
    point: tuple[ExactScalar, ...] | None
    edge_id: BRepEntityId | None

    def box_in(self, box: _ExactBox, /) -> _ExactBox | None:
        """Exact bounding box of the segment inside the closed ``box``."""
        first, last = Fraction(0), Fraction(1)
        for axis in (0, 1):
            origin = self.start[axis]
            step = self.end[axis] - origin
            if step == 0:
                if not box[axis][0] <= origin <= box[axis][1]:
                    return None
                continue
            entry, exit_ = sorted(
                ((box[axis][0] - origin) / step, (box[axis][1] - origin) / step)
            )
            first, last = max(first, entry), min(last, exit_)
            if first > last:
                return None
        ends = tuple(
            tuple(
                self.start[axis] + value * (self.end[axis] - self.start[axis])
                for axis in (0, 1)
            )
            for value in (first, last)
        )
        return (
            (min(ends[0][0], ends[1][0]), max(ends[0][0], ends[1][0])),
            (min(ends[0][1], ends[1][1]), max(ends[0][1], ends[1][1])),
        )


def _collapsed_spline_boundaries(
    patch: BSplineSurfacePatch, /
) -> tuple[_DeclaredLine, ...]:
    """Clamped boundary rows whose exact control points all coincide."""
    try:
        points = patch.exact_control_points
        u_knots = patch.exact_u_knots
        v_knots = patch.exact_v_knots
    except (TypeError, ValueError):
        return ()
    degrees = (patch.u_degree, patch.v_degree)
    knots = (u_knots, v_knots)
    clamped = tuple(
        len(set(values[: degree + 1])) == 1 and len(set(values[-degree - 1 :])) == 1
        for values, degree in zip(knots, degrees, strict=True)
    )
    ranges = tuple(
        (values[degree], values[-degree - 1])
        for values, degree in zip(knots, degrees, strict=True)
    )
    lines: list[_DeclaredLine] = []
    if clamped[0]:
        for row, value in ((points[0], u_knots[0]), (points[-1], u_knots[-1])):
            if all(point == row[0] for point in row):
                lines.append(
                    _DeclaredLine(
                        (value, ranges[1][0]), (value, ranges[1][1]), tuple(row[0]), None
                    )
                )
    if clamped[1]:
        for index, value in ((0, v_knots[0]), (-1, v_knots[-1])):
            column = tuple(row[index] for row in points)
            if all(point == column[0] for point in column):
                lines.append(
                    _DeclaredLine(
                        (ranges[0][0], value),
                        (ranges[0][1], value),
                        tuple(column[0]),
                        None,
                    )
                )
    return tuple(lines)


def _declared_lines(
    model: BRepModel, face: int, patch: AbstractSurfacePatch, /
) -> tuple[_DeclaredLine, ...]:
    """Degenerate edge p-curves of ``face`` and collapsed spline boundary rows.

    The model proved at construction that each degenerate coedge maps its full
    source interval to one exact vertex, so ``S_u x S_v`` vanishes along it.
    """
    lines: list[_DeclaredLine] = []
    geometry = model.geometry
    if geometry is not None:
        ranges = np.asarray(geometry.edge_ranges, dtype=np.float64)
        for loop in geometry.face_loops[face]:
            for coedge in loop:
                edge = int(geometry.coedge_edges[coedge])
                if geometry.edge_curves[edge] >= 0:
                    continue
                affine = _affine_curve_coefficients(
                    geometry.pcurves[coedge], (Fraction(0), Fraction(0))
                )
                if affine is None or len(affine[0]) != 2 or len(affine[1]) != 2:
                    continue
                (x0, y0), (dx, dy) = affine
                if not (
                    isinstance(x0, Fraction)
                    and isinstance(y0, Fraction)
                    and isinstance(dx, Fraction)
                    and isinstance(dy, Fraction)
                ):
                    continue
                first, last = (
                    Fraction.from_float(float(value)) for value in ranges[edge]
                )
                start = (x0 + first * dx, y0 + first * dy)
                end = (x0 + last * dx, y0 + last * dy)
                vertex = int(geometry.edge_vertices[edge][0])
                lines.append(
                    _DeclaredLine(
                        start,
                        end,
                        geometry.vertex_exact_points[vertex],
                        model.edge_ids[edge],
                    )
                )
    if isinstance(patch, BSplineSurfacePatch):
        lines.extend(_collapsed_spline_boundaries(patch))
    unique: dict[tuple[tuple[Fraction, Fraction], ...], _DeclaredLine] = {}
    for line in lines:
        unique.setdefault(tuple(sorted((line.start, line.end))), line)
    return tuple(unique.values())


def _curve_breaks(curve: object, /) -> np.ndarray:
    if isinstance(curve, BSplineCurve):
        return np.unique(np.asarray(curve.knots, dtype=np.float64))
    return np.empty((0,), dtype=np.float64)


def _break_values(patch: AbstractSurfacePatch, /) -> tuple[np.ndarray, np.ndarray]:
    """Knot lines per parameter axis; continuity and span changes lie only there."""
    empty = np.empty((0,), dtype=np.float64)
    if isinstance(patch, OffsetSurface):
        return _break_values(patch.base)
    if isinstance(patch, BSplineSurfacePatch):
        return (
            np.unique(np.asarray(patch.u_knots, dtype=np.float64)),
            np.unique(np.asarray(patch.v_knots, dtype=np.float64)),
        )
    if isinstance(patch, ExtrusionSurface):
        return _curve_breaks(patch.curve), empty
    if isinstance(patch, RevolutionSurface):
        return empty, _curve_breaks(patch.curve)
    if isinstance(patch, RuledSurface):
        return np.union1d(_curve_breaks(patch.first), _curve_breaks(patch.second)), empty
    return empty, empty


def _poly_combine(first: _Poly, second: _Poly, sign: int, /) -> _Poly:
    result = dict(first)
    for key, value in second.items():
        result[key] = result.get(key, Fraction(0)) + sign * value
    return {key: value for key, value in result.items() if value != 0}


def _poly_multiply(first: _Poly, second: _Poly, /) -> _Poly:
    result: _Poly = {}
    for (a, b), left in first.items():
        for (c, d), right in second.items():
            key = (a + c, b + d)
            result[key] = result.get(key, Fraction(0)) + left * right
    return {key: value for key, value in result.items() if value != 0}


def _poly_derivative(poly: _Poly, axis: int, /) -> _Poly:
    result: _Poly = {}
    for (a, b), value in poly.items():
        power = a if axis == 0 else b
        if power:
            result[(a - 1, b) if axis == 0 else (a, b - 1)] = power * value
    return result


def _poly_value(poly: _Poly, point: tuple[Fraction, Fraction], /) -> Fraction:
    return sum(
        (value * point[0] ** a * point[1] ** b for (a, b), value in poly.items()),
        Fraction(0),
    )


def _power_from_bernstein(values: list[Fraction], /) -> list[Fraction]:
    degree = len(values) - 1
    return [
        sum(
            (
                values[index]
                * comb(degree, index)
                * comb(degree - index, power - index)
                * (-1) ** (power - index)
                for index in range(power + 1)
            ),
            Fraction(0),
        )
        for power in range(degree + 1)
    ]


def _bernstein_on(
    values: list[Fraction], lower: Fraction, upper: Fraction, /
) -> list[Fraction]:
    """Bernstein coefficients on ``[lower, upper]`` of a power-basis polynomial."""
    degree = len(values) - 1
    width = upper - lower
    shifted = [
        width**power
        * sum(
            (
                comb(index, power) * lower ** (index - power) * values[index]
                for index in range(power, degree + 1)
            ),
            Fraction(0),
        )
        for power in range(degree + 1)
    ]
    return [
        sum(
            (
                Fraction(comb(index, power), comb(degree, power)) * shifted[power]
                for power in range(index + 1)
            ),
            Fraction(0),
        )
        for index in range(degree + 1)
    ]


def _poly_range(poly: _Poly, box: _ExactBox, /) -> tuple[Fraction, Fraction]:
    """Exact range enclosure from the Bernstein coefficients of ``poly`` on ``box``."""
    if not poly:
        return Fraction(0), Fraction(0)
    degree_s = max(a for a, _ in poly)
    degree_t = max(b for _, b in poly)
    rows = [
        _bernstein_on(
            [poly.get((a, b), Fraction(0)) for b in range(degree_t + 1)], *box[1]
        )
        for a in range(degree_s + 1)
    ]
    values = [
        value
        for b in range(degree_t + 1)
        for value in _bernstein_on([rows[a][b] for a in range(degree_s + 1)], *box[0])
    ]
    return min(values), max(values)


@dataclass(frozen=True, slots=True)
class _SpanNormal:
    """Exact ``S_s x S_t`` numerator of one rational span in relative parameters.

    With ``S = P / W`` the numerator ``(P_s W - P W_s) x (P_t W - P W_t)`` equals
    ``S_s x S_t`` times ``W^4``, so both vanish at the same points.
    """

    bounds: _ExactBox
    homogeneous: tuple[_Poly, ...]
    normal: tuple[_Poly, _Poly, _Poly]
    jacobian: tuple[tuple[_Poly, _Poly], ...]

    def relative(self, box: _ExactBox, /) -> _ExactBox:
        (u0, u1), (v0, v1) = self.bounds
        return (
            ((box[0][0] - u0) / (u1 - u0), (box[0][1] - u0) / (u1 - u0)),
            ((box[1][0] - v0) / (v1 - v0), (box[1][1] - v0) / (v1 - v0)),
        )

    def absolute(self, point: tuple[Fraction, Fraction], /) -> tuple[Fraction, Fraction]:
        return (
            self.bounds[0][0] + point[0] * (self.bounds[0][1] - self.bounds[0][0]),
            self.bounds[1][0] + point[1] * (self.bounds[1][1] - self.bounds[1][0]),
        )

    def surface_point(self, point: tuple[Fraction, Fraction], /) -> tuple[Fraction, ...]:
        weight = _poly_value(self.homogeneous[-1], point)
        return tuple(
            _poly_value(component, point) / weight for component in self.homogeneous[:-1]
        )


def _span_normal(span: object, /) -> _SpanNormal | None:
    bounds = getattr(span, "parameter_bounds", None)
    net = getattr(span, "homogeneous_bernstein_coefficients", None)
    if bounds is None or net is None:
        return None
    (u0, u1), (v0, v1) = bounds
    if not all(isinstance(value, Fraction) for value in (u0, u1, v0, v1)):
        return None
    if not (u0 < u1 and v0 < v1) or len(net[0][0]) != 4:
        return None
    if any(
        not isinstance(value, Fraction) for row in net for point in row for value in point
    ):
        return None
    homogeneous: list[_Poly] = []
    for component in range(4):
        rows = [_power_from_bernstein([point[component] for point in row]) for row in net]
        poly: _Poly = {}
        for b in range(len(rows[0])):
            for a, value in enumerate(_power_from_bernstein([row[b] for row in rows])):
                if value:
                    poly[(a, b)] = value
        homogeneous.append(poly)
    weight = homogeneous[3]
    tangents = tuple(
        tuple(
            _poly_combine(
                _poly_multiply(_poly_derivative(component, axis), weight),
                _poly_multiply(component, _poly_derivative(weight, axis)),
                -1,
            )
            for component in homogeneous[:3]
        )
        for axis in (0, 1)
    )
    normal = tuple(
        _poly_combine(
            _poly_multiply(tangents[0][(c + 1) % 3], tangents[1][(c + 2) % 3]),
            _poly_multiply(tangents[0][(c + 2) % 3], tangents[1][(c + 1) % 3]),
            -1,
        )
        for c in range(3)
    )
    return _SpanNormal(
        ((u0, u1), (v0, v1)),
        tuple(homogeneous),
        (normal[0], normal[1], normal[2]),
        tuple((_poly_derivative(row, 0), _poly_derivative(row, 1)) for row in normal),
    )


def _simplest_rational(lower: Fraction, upper: Fraction, /) -> Fraction:
    """Smallest-denominator rational in ``[lower, upper]`` (continued fractions)."""
    whole = floor(lower)
    if whole == lower:
        return Fraction(whole)
    if whole + 1 <= upper:
        return Fraction(whole + 1)
    return whole + 1 / _simplest_rational(1 / (upper - whole), 1 / (lower - whole))


def _outward_binary64(box: _ExactBox, within: _ExactBox, /) -> _ExactBox:
    """Round ``box`` outward to binary64 endpoints, clipped to ``within``."""
    return (
        (
            max(within[0][0], Fraction.from_float(fraction_down(box[0][0]))),
            min(within[0][1], Fraction.from_float(fraction_up(box[0][1]))),
        ),
        (
            max(within[1][0], Fraction.from_float(fraction_down(box[1][0]))),
            min(within[1][1], Fraction.from_float(fraction_up(box[1][1]))),
        ),
    )


def _subsystem_jacobian(
    system: _SpanNormal, pair: tuple[int, int], point: tuple[Fraction, Fraction], /
) -> tuple[tuple[Fraction, Fraction], tuple[Fraction, Fraction]]:
    first, second = pair
    return (
        (
            _poly_value(system.jacobian[first][0], point),
            _poly_value(system.jacobian[first][1], point),
        ),
        (
            _poly_value(system.jacobian[second][0], point),
            _poly_value(system.jacobian[second][1], point),
        ),
    )


def _krawczyk(
    system: _SpanNormal,
    pair: tuple[int, int],
    box: _ExactBox,
    center: tuple[Fraction, Fraction],
    /,
) -> _ExactBox | None:
    """Exact Krawczyk image of ``box`` for the two-component subsystem ``pair``."""
    jacobian = _subsystem_jacobian(system, pair, center)
    determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
    if determinant == 0:
        return None
    inverse = (
        (jacobian[1][1] / determinant, -jacobian[0][1] / determinant),
        (-jacobian[1][0] / determinant, jacobian[0][0] / determinant),
    )
    values = tuple(_poly_value(system.normal[index], center) for index in pair)
    ranges = tuple(
        tuple(_poly_range(system.jacobian[index][axis], box) for axis in (0, 1))
        for index in pair
    )
    radius = tuple(
        max(center[axis] - box[axis][0], box[axis][1] - center[axis]) for axis in (0, 1)
    )
    image: list[tuple[Fraction, Fraction]] = []
    for row in (0, 1):
        newton = center[row] - (inverse[row][0] * values[0] + inverse[row][1] * values[1])
        spread = Fraction(0)
        for column in (0, 1):
            low = high = Fraction(int(row == column))
            for inner in (0, 1):
                coefficient = inverse[row][inner]
                first, last = sorted(
                    (
                        coefficient * ranges[inner][column][0],
                        coefficient * ranges[inner][column][1],
                    )
                )
                low, high = low - last, high - first
            spread += max(abs(low), abs(high)) * radius[column]
        image.append((newton - spread, newton + spread))
    return image[0], image[1]


@dataclass(frozen=True, slots=True)
class _Isolation:
    """Krawczyk box with a unique zero of a two-component cross-product subsystem."""

    box: _ExactBox
    enclosure: _ExactBox
    zero: tuple[Fraction, Fraction] | None
    # True: the third component vanishes at the exact zero; False: S_u x S_v has
    # no zero in ``box``; None: undecided at binary64 resolution.
    singular: bool | None


def _isolate_cross_product_zero(
    system: _SpanNormal, cell: _ExactBox, /
) -> _Isolation | None:
    """Isolate the unique zero of two components of ``S_s x S_t`` near ``cell``.

    A Newton iterate (binary64, residual strictly decreasing) centers a box with
    the cell's widths; the exact Krawczyk image inside its interior proves one
    zero of the subsystem there. The image is contracted to binary64 resolution
    while an exact rational zero is sought (the iterate itself, then the simplest
    rational in the enclosure); the third component is then decided exactly at
    that zero or by its Bernstein range on the enclosure.
    """
    center = ((cell[0][0] + cell[0][1]) / 2, (cell[1][0] + cell[1][1]) / 2)
    pairs = ((0, 1), (0, 2), (1, 2))
    determinants = []
    for pair in pairs:
        jacobian = _subsystem_jacobian(system, pair, center)
        determinants.append(
            abs(jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0])
        )
    best = max(range(3), key=determinants.__getitem__)
    if determinants[best] == 0:
        return None
    pair = pairs[best]
    third = 3 - pair[0] - pair[1]

    def residual(point: tuple[Fraction, Fraction], /) -> Fraction:
        return max(abs(_poly_value(system.normal[index], point)) for index in pair)

    point = (
        Fraction.from_float(float(center[0])),
        Fraction.from_float(float(center[1])),
    )
    current = residual(point)
    while current:
        jacobian = _subsystem_jacobian(system, pair, point)
        determinant = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0]
        if determinant == 0:
            break
        values = tuple(_poly_value(system.normal[index], point) for index in pair)
        step = (
            (jacobian[1][1] * values[0] - jacobian[0][1] * values[1]) / determinant,
            (jacobian[0][0] * values[1] - jacobian[1][0] * values[0]) / determinant,
        )
        candidate = (
            Fraction.from_float(float(point[0] - step[0])),
            Fraction.from_float(float(point[1] - step[1])),
        )
        following = residual(candidate)
        if candidate == point or not following < current:
            break
        point, current = candidate, following
    box: _ExactBox = (
        (point[0] - (cell[0][1] - cell[0][0]), point[0] + (cell[0][1] - cell[0][0])),
        (point[1] - (cell[1][1] - cell[1][0]), point[1] + (cell[1][1] - cell[1][0])),
    )
    image = _krawczyk(system, pair, box, point)
    if image is None or not all(
        box[axis][0] < image[axis][0] and image[axis][1] < box[axis][1] for axis in (0, 1)
    ):
        return None
    enclosure = image
    while True:
        candidates = (
            point,
            (
                _simplest_rational(*enclosure[0]),
                _simplest_rational(*enclosure[1]),
            ),
        )
        for candidate in candidates:
            if _box_contains_point(enclosure, candidate) and all(
                _poly_value(system.normal[index], candidate) == 0 for index in pair
            ):
                return _Isolation(
                    box,
                    enclosure,
                    candidate,
                    _poly_value(system.normal[third], candidate) == 0,
                )
        low, high = _poly_range(system.normal[third], enclosure)
        if low > 0 or high < 0:
            return _Isolation(box, enclosure, None, False)
        middle = (
            Fraction.from_float(float((enclosure[0][0] + enclosure[0][1]) / 2)),
            Fraction.from_float(float((enclosure[1][0] + enclosure[1][1]) / 2)),
        )
        contracted = _krawczyk(system, pair, enclosure, middle)
        if contracted is None:
            return _Isolation(box, enclosure, None, None)
        narrowed = _outward_binary64(
            (
                (
                    max(enclosure[0][0], contracted[0][0]),
                    min(enclosure[0][1], contracted[0][1]),
                ),
                (
                    max(enclosure[1][0], contracted[1][0]),
                    min(enclosure[1][1], contracted[1][1]),
                ),
            ),
            enclosure,
        )
        if narrowed == enclosure or any(low > high for low, high in narrowed):
            return _Isolation(box, enclosure, None, None)
        enclosure = narrowed


def _bisect(box: np.ndarray, axis: int, /) -> tuple[np.ndarray, np.ndarray] | None:
    low, high = box[0, axis], box[1, axis]
    middle = low + 0.5 * (high - low)
    if not low < middle < high:
        middle = float(np.nextafter(low, high))
    if not low < middle < high:
        return None
    left, right = box.copy(), box.copy()
    left[1, axis], right[0, axis] = middle, middle
    return left, right


def _status(
    lower: float, upper: float, draft: float, vertical: bool, unresolved: bool, /
) -> BRepNormalStatus:
    if unresolved:
        return "unresolved"
    if vertical:
        return "vertical"
    limit = _HALF_PI - draft
    if upper < limit:
        return "positive"
    if lower > limit and upper < _HALF_PI:
        return "insufficient"
    if lower > _HALF_PI:
        return "undercut"
    return "unresolved"


def _split_at(
    box: np.ndarray, axis: int, value: float, /
) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = box.copy(), box.copy()
    lower[1, axis], upper[0, axis] = value, value
    return lower, upper


def _flag_axes(
    patch: AbstractSurfacePatch, profile: _MeridianProfile | None, box: np.ndarray, /
) -> tuple[int, ...] | None:
    """Axes whose refinement alone can decide an undefined-normal cell.

    A meridian revolution normal and an extrusion normal depend on one parameter
    only; a cone normal is undefined only where its radius changes sign, which
    depends on the axial parameter alone.
    """
    if profile is not None:
        return (1,)
    if isinstance(patch, ExtrusionSurface):
        return (0,)
    if isinstance(patch, ConePatch) and _cartesian_normal_interval(patch, box) is None:
        return (1,)
    return None


def _face_range(
    model: BRepModel,
    face: int,
    direction: np.ndarray,
    draft: float,
    angular_tolerance: float,
    thresholds: tuple[float, ...],
    orientation: float,
    /,
) -> BRepNormalRange:
    """Refine one face until every cell is decided or at binary64 resolution.

    A cell with an undefined normal is subdivided (at knot lines first) until
    a positive lower bound on ``|S_u x S_v|`` decides it regular, or it is
    certified to contain a singular point: a declared degenerate edge p-curve
    or an exactly verified isolated zero of the cross product. Certified
    singular cells contribute ``[0, pi]`` and are shrunk toward their evidence
    while each split-off part is immediately regular. Once a singular cell fixes
    the face range to ``[0, pi]``, regular cells cannot change the result and
    are not refined further.
    """
    patch = model.patches[face]
    oriented = bool(np.isfinite(orientation))
    vertical = _exact_vertical(patch, direction) and oriented
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
    extent = bounds[1] - bounds[0]
    critical = (pi / 2.0 - draft, pi / 2.0, *thresholds)
    profile = _meridian_profile(patch) if isinstance(patch, RevolutionSurface) else None
    domain = model.trim_domains[face]
    enclosures = None if domain is None else domain.prepare_enclosures()
    declared = _declared_lines(model, face, patch)
    breaks = _break_values(patch)
    spans: tuple[object, ...] | None = None
    systems: dict[int, _SpanNormal | None] = {}
    cells: list[_Cell] = []

    def angle_of(box: np.ndarray, /) -> tuple[float, float, bool]:
        return _angle(patch, box, orientation, direction)

    def excluded(box: np.ndarray, /) -> bool:
        return _trim_excludes(model, face, box, enclosures)

    def interior_breaks(box: np.ndarray, axis: int, /) -> np.ndarray:
        values = breaks[axis]
        return values[(values > box[0, axis]) & (values < box[1, axis])]

    def span_system(box: _ExactBox, /) -> _SpanNormal | None:
        nonlocal spans
        if not isinstance(patch, BSplineSurfacePatch):
            return None
        if spans is None:
            try:
                spans = tuple(patch.rational_spans())
            except (TypeError, ValueError, RuntimeError):
                spans = ()
        for index, span in enumerate(spans):
            (u0, u1), (v0, v1) = getattr(span, "parameter_bounds")
            if (
                u0 <= box[0][0]
                and box[0][1] <= u1
                and v0 <= box[1][0]
                and box[1][1] <= v1
            ):
                if index not in systems:
                    systems[index] = _span_normal(span)
                return systems[index]
        return None

    def certify(cell: _Cell, /) -> None:
        """Decide an undefined-normal cell's singular set where an exact proof exists."""
        box = cell.box
        if not patch.is_c1_on(box):
            if not any(interior_breaks(box, axis).size for axis in (0, 1)):
                cell.terminal = "continuity_unestablished"
            return
        exact = _exact_box(box)
        hits = [
            (line, region)
            for line in declared
            if (region := line.box_in(exact)) is not None
        ]
        if len(hits) == 1:
            line, region = hits[0]
            cell.singular = BRepNormalSingularity(
                "declared_degenerate_edge", region, line.point, line.edge_id
            )
            return
        if hits or cell.free:
            return
        system = span_system(exact)
        if system is None:
            return
        relative = system.relative(exact)
        isolation = _isolate_cross_product_zero(system, relative)
        if isolation is None:
            return
        covered = _box_contains_box(isolation.box, relative)
        zero = isolation.zero
        if isolation.singular is True and zero is not None:
            if _box_contains_point(relative, zero):
                point = system.absolute(zero)
                cell.singular = BRepNormalSingularity(
                    "isolated_cross_product_zero",
                    ((point[0], point[0]), (point[1], point[1])),
                    system.surface_point(zero),
                    None,
                )
            elif covered:
                cell.free = True
        elif isolation.singular is False:
            cell.free = covered
        elif covered:
            enclosure = isolation.enclosure
            disjoint = any(
                enclosure[axis][1] < relative[axis][0]
                or relative[axis][1] < enclosure[axis][0]
                for axis in (0, 1)
            )
            cell.free = disjoint
            cell.unverified = not disjoint

    def separate(cell: _Cell, evidence: BRepNormalSingularity, /) -> None:
        """Shrink a singular cell while each split-off part is immediately regular."""
        target = evidence.parameter_box
        while True:
            box = cell.box
            widths = (box[1] - box[0]) / extent
            plan: tuple[int, float] | None = None
            for axis in (int(value) for value in np.argsort(-widths, kind="stable")):
                low, high = float(box[0, axis]), float(box[1, axis])
                best: tuple[Fraction, float] | None = None
                for share in (0.25, 0.5, 0.75):
                    value = low + share * (high - low)
                    if not low < value < high:
                        continue
                    exact = Fraction.from_float(value)
                    gap = (
                        target[axis][0] - exact
                        if exact < target[axis][0]
                        else exact - target[axis][1]
                        if exact > target[axis][1]
                        else None
                    )
                    if gap is not None and (best is None or gap > best[0]):
                        best = (gap, value)
                if best is not None:
                    plan = (axis, best[1])
                    break
            if plan is None:
                break
            axis, value = plan
            lower, upper = _split_at(box, axis, value)
            near, far = (
                (lower, upper)
                if Fraction.from_float(value) > target[axis][1]
                else (upper, lower)
            )
            if not excluded(far):
                angle = angle_of(far)
                if angle[2]:
                    break
                cells.append(_Cell(far, angle, free=True))
            if excluded(near):
                return
            cell.box = near
        cells.append(cell)

    def admit(box: np.ndarray, angle: tuple[float, float, bool], free: bool, /) -> None:
        cell = _Cell(box, angle, free=free)
        if angle[2]:
            certify(cell)
        if cell.singular is not None:
            separate(cell, cell.singular)
        else:
            cells.append(cell)

    root = bounds.copy()
    if not excluded(root):
        if oriented:
            admit(root, angle_of(root), False)
        else:
            cells.append(_Cell(root, (0.0, _PI, True), terminal="orientation_undefined"))

    def unresolved_cell(
        angle: tuple[float, float, bool], lower: float, upper: float, /
    ) -> bool:
        if vertical or angle[1] - angle[0] <= angular_tolerance:
            return False
        return (
            angle[0] <= lower + angular_tolerance
            or angle[1] >= upper - angular_tolerance
            or any(angle[0] <= value <= angle[1] for value in critical)
        )

    def flagged_split(box: np.ndarray, /) -> tuple[np.ndarray, np.ndarray] | None:
        widths = (box[1] - box[0]) / extent
        axes = _flag_axes(patch, profile, box) or tuple(
            int(axis) for axis in np.argsort(-widths, kind="stable")
        )
        for axis in axes:
            inner = interior_breaks(box, axis)
            if inner.size:
                middle = 0.5 * (box[0, axis] + box[1, axis])
                return _split_at(
                    box, axis, float(inner[np.argmin(np.abs(inner - middle))])
                )
        for axis in axes:
            halves = _bisect(box, axis)
            if halves is not None:
                return halves
        return None

    def regular_axes(box: np.ndarray, /) -> tuple[int, ...]:
        # Widest normalized axis first so ties keep the isotropic order.
        widths = (box[1] - box[0]) / extent
        return tuple(
            int(axis)
            for axis in np.argsort(-widths, kind="stable")
            if _bisect(box, int(axis)) is not None
        )

    def subdivide(
        box: np.ndarray, axes: tuple[int, ...], /
    ) -> tuple[tuple[np.ndarray, tuple[float, float, bool]], ...]:
        # Bisect along the axis whose children have the tightest enclosure: normal
        # angles of revolved and extruded carriers often depend on one axis only.
        best: tuple[tuple[np.ndarray, tuple[float, float, bool]], ...] | None = None
        best_width = inf
        for axis in axes:
            halves = _bisect(box, axis)
            if halves is None:
                continue
            children = tuple((child, angle_of(child)) for child in halves)
            width = max(angle[1] - angle[0] for _, angle in children)
            if best is None or width < best_width:
                best, best_width = children, width
        return () if best is None else best

    while cells:
        pinned = any(cell.singular is not None for cell in cells)
        lower = min(cell.angle[0] for cell in cells)
        upper = max(cell.angle[1] for cell in cells)
        candidates: list[
            tuple[float, int, tuple[np.ndarray, np.ndarray] | None, tuple[int, ...]]
        ] = []
        for index, cell in enumerate(cells):
            if cell.singular is not None or cell.terminal is not None:
                continue
            if cell.angle[2]:
                halves = flagged_split(cell.box)
                if halves is not None:
                    candidates.append((_PI, -index, halves, ()))
            elif not pinned and unresolved_cell(cell.angle, lower, upper):
                axes = regular_axes(cell.box)
                if axes:
                    candidates.append((cell.angle[1] - cell.angle[0], -index, None, axes))
        if not candidates:
            break
        _, negative_index, halves, axes = max(candidates, key=lambda row: row[:2])
        parent = cells.pop(-negative_index)
        if halves is not None:
            for child in reversed(halves):
                if not excluded(child):
                    admit(child, angle_of(child), parent.free)
        else:
            for child, angle in reversed(subdivide(parent.box, axes)):
                if not excluded(child):
                    admit(child, angle, parent.free)
    pinned = any(cell.singular is not None for cell in cells)
    lower = min((cell.angle[0] for cell in cells), default=0.0)
    upper = max((cell.angle[1] for cell in cells), default=_PI)
    unresolved: list[np.ndarray] = []
    reasons: list[BRepNormalUnresolvedReason] = []
    singular_cells: list[np.ndarray] = []
    singular_points: list[np.ndarray] = []
    singularities: list[BRepNormalSingularity] = []
    for cell in cells:
        if cell.singular is not None:
            singular_cells.append(cell.box)
            singularities.append(cell.singular)
            point = cell.singular.point
            singular_points.append(
                _exact_point_box(point)
                if point is not None
                else _meridian_point_box(profile, cell.box)
                if profile is not None
                else np.asarray(patch.bounding_box(cell.box), dtype=np.float64)
            )
        elif cell.angle[2]:
            unresolved.append(cell.box)
            reasons.append(
                cell.terminal
                if cell.terminal is not None
                else "unverified_isolated_zero"
                if cell.unverified
                else "undefined_normal"
            )
        elif not pinned and unresolved_cell(cell.angle, lower, upper):
            unresolved.append(cell.box)
            reasons.append("angle_unresolved")
    status = _status(lower, upper, draft, bool(vertical), bool(unresolved))
    return BRepNormalRange(
        face_id=model.face_ids[face],
        model_id=model.model_id,
        status=status,
        angle_enclosure=jnp.asarray((lower, upper), dtype=jnp.float64),
        unresolved_cells=jnp.asarray(
            np.asarray(unresolved, dtype=np.float64).reshape((-1, 2, 2))
        ),
        unresolved_reasons=tuple(reasons),
        singular_cells=jnp.asarray(
            np.asarray(singular_cells, dtype=np.float64).reshape((-1, 2, 2))
        ),
        singular_points=jnp.asarray(
            np.asarray(singular_points, dtype=np.float64).reshape((-1, 2, 3))
        ),
        singularities=tuple(singularities),
    )


def brep_normal_ranges(
    model: BRepModel,
    direction: ArrayLike | tuple[float, float, float],
    /,
    *,
    draft_angle: float,
    angular_tolerance: float,
    angle_thresholds: tuple[float, ...] = (),
) -> tuple[BRepNormalRange, ...]:
    """Bound each face's oriented angle to a pull direction over its whole trim.

    ``draft_angle`` and all returned angles are radians. Subdivision follows the
    requested angular enclosure accuracy and representable parameter resolution;
    threshold crossings remain unresolved rather than being decided by samples.
    """
    if not x64_enabled():
        raise ValueError("normal.precision requires JAX binary64; got x64 disabled.")
    draft = finite_real_scalar(draft_angle, "normal.draft_angle")
    tolerance = finite_real_scalar(angular_tolerance, "normal.angular_tolerance")
    if not 0.0 <= draft < _HALF_PI:
        raise ValueError("normal.draft_angle must lie in [0, pi/2).")
    if tolerance <= 0.0:
        raise ValueError("normal.angular_tolerance must be finite and positive.")
    thresholds = tuple(
        finite_real_scalar(value, "normal.angle_thresholds") for value in angle_thresholds
    )
    if any(value < 0.0 or value > pi for value in thresholds):
        raise ValueError("normal.angle_thresholds must lie in [0, pi].")
    vector = _direction(direction)
    orientations = _material_orientations(model)
    return tuple(
        _face_range(
            model,
            face,
            vector,
            draft,
            tolerance,
            thresholds,
            float(orientations[face]),
        )
        for face in range(len(model.patches))
    )


__all__ = [
    "BRepNormalRange",
    "BRepNormalSingularity",
    "BRepNormalStatus",
    "BRepNormalUnresolvedReason",
    "brep_normal_ranges",
]
