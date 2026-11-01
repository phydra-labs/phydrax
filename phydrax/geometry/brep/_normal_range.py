#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous outward normal-angle enclosures over native B-Rep face trims."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import inf, pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._validation import finite_real_scalar
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
    CircleCurve,
    ConePatch,
    CylinderPatch,
    ExactLineCurve,
    LineCurve,
    PlanePatch,
    RevolutionSurface,
    SpherePatch,
)


BRepNormalStatus: TypeAlias = Literal[
    "positive", "insufficient", "vertical", "undercut", "unresolved"
]

_PI = float(np.nextafter(pi, np.inf))
_HALF_PI = pi / 2.0


@final
class BRepNormalRange(StrictModule):
    """Conservative normal angle and unresolved source regions for one face."""

    face_id: BRepEntityId = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    status: BRepNormalStatus = eqx.field(static=True)
    angle_enclosure: Array
    unresolved_cells: Array
    singular_points: Array


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
type _Cell = tuple[np.ndarray, tuple[float, float, bool]]


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


# Cells whose normal is undefined (axis folds, collapsed edges, non-C1 seams) are
# subdivided only to this relative parameter width and then reported unresolved.
_SINGULAR_RESOLUTION = 2.0**-8


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
    patch = model.patches[face]
    vertical = _exact_vertical(patch, direction) and np.isfinite(orientation)
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
    extent = bounds[1] - bounds[0]
    critical = (pi / 2.0 - draft, pi / 2.0, *thresholds)
    # A meridian revolution normal is singular only where the profile meets the
    # axis, which depends on the profile parameter alone.
    profile = _meridian_profile(patch) if isinstance(patch, RevolutionSurface) else None
    singular_axis = None if profile is None else 1
    cells: list[_Cell] = []
    root = bounds.copy()
    domain = model.trim_domains[face]
    enclosures = None if domain is None else domain.prepare_enclosures()
    if not _trim_excludes(model, face, root, enclosures):
        cells.append((root, _angle(patch, root, orientation, direction)))

    def unresolved_cell(
        angle: tuple[float, float, bool], lower: float, upper: float, /
    ) -> bool:
        if angle[2]:
            return True
        if vertical or angle[1] - angle[0] <= angular_tolerance:
            return False
        return (
            angle[0] <= lower + angular_tolerance
            or angle[1] >= upper - angular_tolerance
            or any(angle[0] <= value <= angle[1] for value in critical)
        )

    def splittable_axes(box: np.ndarray, singular: bool, /) -> tuple[int, ...]:
        widths = (box[1] - box[0]) / extent
        if singular:
            axes = (
                (int(np.argmax(widths)),) if singular_axis is None else (singular_axis,)
            )
            if all(widths[axis] <= _SINGULAR_RESOLUTION for axis in axes):
                return ()
        else:
            # Widest normalized axis first so ties keep the isotropic order.
            axes = tuple(int(axis) for axis in np.argsort(-widths, kind="stable"))
        return tuple(axis for axis in axes if _bisect(box, axis) is not None)

    def subdivide(box: np.ndarray, axes: tuple[int, ...], /) -> tuple[_Cell, ...]:
        # Bisect along the axis whose children have the tightest enclosure: normal
        # angles of revolved and extruded carriers often depend on one axis only.
        best: tuple[_Cell, ...] | None = None
        best_width = inf
        for axis in axes:
            halves = _bisect(box, axis)
            if halves is None:
                continue
            children = tuple(
                (child, _angle(patch, child, orientation, direction)) for child in halves
            )
            width = max(angle[1] - angle[0] for _, angle in children)
            if best is None or width < best_width:
                best, best_width = children, width
        return () if best is None else best

    while cells:
        lower = min(angle[0] for _, angle in cells)
        upper = max(angle[1] for _, angle in cells)
        candidates: list[tuple[float, int, tuple[int, ...]]] = []
        for index, (box, angle) in enumerate(cells):
            if not unresolved_cell(angle, lower, upper):
                continue
            axes = splittable_axes(box, angle[2])
            if axes:
                candidates.append((angle[1] - angle[0], -index, axes))
        if not candidates:
            break
        _, negative_index, axes = max(candidates, key=lambda row: row[:2])
        box, _ = cells.pop(-negative_index)
        cells.extend(
            (child, angle)
            for child, angle in reversed(subdivide(box, axes))
            if not _trim_excludes(model, face, child, enclosures)
        )
    lower = min((angle[0] for _, angle in cells), default=0.0)
    upper = max((angle[1] for _, angle in cells), default=_PI)
    unresolved = tuple(
        box for box, angle in cells if unresolved_cell(angle, lower, upper)
    )
    singular = tuple(
        np.asarray(patch.bounding_box(box), dtype=np.float64)
        if profile is None
        else _meridian_point_box(profile, box)
        for box, angle in cells
        if angle[2]
    )
    status = _status(lower, upper, draft, bool(vertical), bool(unresolved))
    return BRepNormalRange(
        model.face_ids[face],
        model.model_id,
        status,
        jnp.asarray((lower, upper), dtype=jnp.float64),
        jnp.asarray(np.asarray(unresolved, dtype=np.float64).reshape((-1, 2, 2))),
        jnp.asarray(np.asarray(singular, dtype=np.float64).reshape((-1, 2, 3))),
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


__all__ = ["BRepNormalRange", "BRepNormalStatus", "brep_normal_ranges"]
