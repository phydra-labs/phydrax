#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Continuous outward normal-angle enclosures over native B-Rep face trims."""

from __future__ import annotations

from fractions import Fraction
from math import pi
from typing import final, Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from ..._jax_config import x64_enabled
from ..._strict import StrictModule
from ..._validation import finite_real_scalar
from .._atlas import TrimDomain
from .._interval_enclosure import (
    interval_add,
    interval_cos,
    interval_divide,
    interval_multiply,
    interval_sin,
    interval_sqrt,
    interval_square,
    interval_subtract,
)
from ._edge_classification import _material_orientations
from ._model import BRepEntityId, BRepModel
from ._patches import (
    AbstractSurfacePatch,
    ConePatch,
    CylinderPatch,
    PlanePatch,
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


def _vector_interval(
    values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray]:
    host = np.asarray(values, dtype=np.float64)
    return host, host


def _cross(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    lower: list[np.ndarray] = []
    upper: list[np.ndarray] = []
    for axis in range(3):
        next_axis = (axis + 1) % 3
        last_axis = (axis + 2) % 3
        positive = interval_multiply(
            (first[0][next_axis], first[1][next_axis]),
            (second[0][last_axis], second[1][last_axis]),
        )
        negative = interval_multiply(
            (first[0][last_axis], first[1][last_axis]),
            (second[0][next_axis], second[1][next_axis]),
        )
        component = interval_subtract(positive, negative)
        lower.append(np.asarray(component[0]))
        upper.append(np.asarray(component[1]))
    return np.stack(lower), np.stack(upper)


def _dot(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    result: tuple[np.ndarray, np.ndarray] = (
        np.asarray(0.0, dtype=np.float64), np.asarray(0.0, dtype=np.float64)
    )
    for axis in range(3):
        result = interval_add(
            result,
            interval_multiply(
                (first[0][axis], first[1][axis]),
                (second[0][axis], second[1][axis]),
            ),
        )
    return np.asarray(result[0]), np.asarray(result[1])


def _norm(value: tuple[np.ndarray, np.ndarray], /) -> tuple[float, float]:
    squared: tuple[np.ndarray, np.ndarray] = (
        np.asarray(0.0, dtype=np.float64), np.asarray(0.0, dtype=np.float64)
    )
    for axis in range(3):
        squared = interval_add(
            squared,
            interval_square((value[0][axis], value[1][axis])),
        )
    length = interval_sqrt(squared)
    return float(length[0]), float(length[1])


def _normal_interval(
    patch: AbstractSurfacePatch, box: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray] | None:
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
                interval_cos(angle), _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(angle), _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        return radial if radius > 0.0 else (-radial[1], -radial[0])
    if isinstance(patch, ConePatch):
        angle = (box[0, 0], box[1, 0])
        radial = interval_add(
            interval_multiply(
                interval_cos(angle), _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(angle), _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        semi = float(np.asarray(patch.semi_angle))
        cosine = interval_cos((semi, semi))
        sine = interval_sin((semi, semi))
        axial = interval_multiply(
            (-sine[1], -sine[0]), _vector_interval(np.asarray(patch.axis))
        )
        normal = interval_add(interval_multiply(cosine, radial), axial)
        tangent = interval_divide(sine, interval_cos((semi, semi)))
        reference_radius = float(np.asarray(patch.reference_radius))
        radius = interval_add(
            (np.asarray(reference_radius), np.asarray(reference_radius)),
            interval_multiply((box[0, 1], box[1, 1]), tangent),
        )
        if radius[0] <= 0.0 <= radius[1]:
            return None
        return normal if radius[0] > 0.0 else (-normal[1], -normal[0])
    if isinstance(patch, SpherePatch):
        radius = float(np.asarray(patch.radius))
        if radius == 0.0:
            return None
        longitude = (box[0, 0], box[1, 0])
        latitude = (box[0, 1], box[1, 1])
        equatorial = interval_add(
            interval_multiply(
                interval_cos(longitude), _vector_interval(np.asarray(patch.first_axis)),
            ),
            interval_multiply(
                interval_sin(longitude), _vector_interval(np.asarray(patch.second_axis)),
            ),
        )
        radial = interval_add(
            interval_multiply(interval_cos(latitude), equatorial),
            interval_multiply(
                interval_sin(latitude), _vector_interval(np.asarray(patch.axis)),
            ),
        )
        return radial if radius > 0.0 else (-radial[1], -radial[0])
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
) -> tuple[float, float]:
    normal = _normal_interval(patch, box)
    if normal is None or not np.isfinite(orientation):
        return 0.0, _PI
    if orientation < 0.0:
        normal = -normal[1], -normal[0]
    magnitude = _norm(normal)
    if not np.isfinite(magnitude[0]) or not np.isfinite(magnitude[1]) or magnitude[0] <= 0.0:
        return 0.0, _PI
    dot = _dot(normal, _vector_interval(direction))
    direction_norm = _norm(_vector_interval(direction))
    denominator = interval_multiply(magnitude, direction_norm)
    cosine = interval_divide(dot, denominator)
    lower_cosine = float(np.clip(cosine[0], -1.0, 1.0))
    upper_cosine = float(np.clip(cosine[1], -1.0, 1.0))
    lower = float(np.nextafter(np.arccos(upper_cosine), -np.inf))
    upper = float(np.nextafter(np.arccos(lower_cosine), np.inf))
    return max(0.0, lower), min(_PI, upper)


def _exact_vertical(
    patch: AbstractSurfacePatch, direction: np.ndarray, /
) -> bool:
    if not isinstance(patch, (PlanePatch, CylinderPatch)):
        return False
    exact_direction = tuple(Fraction.from_float(float(value)) for value in direction)
    if isinstance(patch, PlanePatch):
        first = tuple(Fraction.from_float(float(value)) for value in np.asarray(patch.first_axis))
        second = tuple(Fraction.from_float(float(value)) for value in np.asarray(patch.second_axis))
        normal = (
            first[1] * second[2] - first[2] * second[1],
            first[2] * second[0] - first[0] * second[2],
            first[0] * second[1] - first[1] * second[0],
        )
        return sum((a * b for a, b in zip(normal, exact_direction, strict=True)), Fraction(0)) == 0
    first = tuple(Fraction.from_float(float(value)) for value in np.asarray(patch.first_axis))
    second = tuple(Fraction.from_float(float(value)) for value in np.asarray(patch.second_axis))
    return all(
        sum((a * b for a, b in zip(axis, exact_direction, strict=True)), Fraction(0)) == 0
        for axis in (first, second)
    )


def _trim_status(model: BRepModel, face: int, box: np.ndarray, /) -> bool | None:
    domain: TrimDomain | None = model.trim_domains[face]
    if domain is None:
        return True
    bounds = np.asarray(model.parameter_bounds, dtype=np.float64)[face]
    extent = bounds[1] - bounds[0]
    normalized = np.stack(((box[0] - bounds[0]) / extent, (box[1] - bounds[0]) / extent))
    result = domain.classify_boxes(normalized[None, ...])
    if not bool(np.asarray(result.resolved[0])):
        return None
    return bool(np.asarray(result.inside[0]))


def _split(box: np.ndarray, bounds: np.ndarray, /) -> tuple[np.ndarray, np.ndarray] | None:
    widths = (box[1] - box[0]) / (bounds[1] - bounds[0])
    axis = int(np.argmax(widths))
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
    cells: list[tuple[np.ndarray, tuple[float, float], bool]] = []
    critical = (pi / 2.0 - draft, pi / 2.0, *thresholds)
    root = bounds.copy()
    trim = _trim_status(model, face, root)
    if trim is not False:
        cells.append((root, _angle(patch, root, orientation, direction), trim is None))
    while cells:
        lower = min(value[1][0] for value in cells)
        upper = max(value[1][1] for value in cells)
        candidates: list[int] = []
        for index, (box, angle, trim_unknown) in enumerate(cells):
            width = angle[1] - angle[0]
            straddles = not vertical and width > angular_tolerance and any(
                angle[0] <= value <= angle[1] for value in critical
            )
            near_extreme = angle[0] <= lower + angular_tolerance or angle[1] >= upper - angular_tolerance
            needs_tightening = width > angular_tolerance
            if (needs_tightening and near_extreme) or straddles or (trim_unknown and needs_tightening):
                if _split(box, bounds) is not None:
                    candidates.append(index)
        if not candidates:
            break
        selected = max(candidates, key=lambda index: cells[index][1][1] - cells[index][1][0])
        box, _, _ = cells.pop(selected)
        children = _split(box, bounds)
        if children is None:
            cells.append((box, _angle(patch, box, orientation, direction), True))
            continue
        for child in reversed(children):
            child_trim = _trim_status(model, face, child)
            if child_trim is not False:
                cells.append((child, _angle(patch, child, orientation, direction), child_trim is None))
    lower = min((value[1][0] for value in cells), default=0.0)
    upper = max((value[1][1] for value in cells), default=_PI)
    unresolved = tuple(
        value[0]
        for value in cells
        if (
            (value[2] and not vertical)
            or (
                not vertical
                and value[1][1] - value[1][0] > angular_tolerance
                and (
                    value[1][0] <= lower + angular_tolerance
                    or value[1][1] >= upper - angular_tolerance
                    or any(
                        value[1][0] <= critical_value <= value[1][1]
                        for critical_value in critical
                    )
                )
            )
        )
    )
    status = _status(lower, upper, draft, bool(vertical), bool(unresolved))
    return BRepNormalRange(
        model.face_ids[face],
        model.model_id,
        status,
        jnp.asarray((lower, upper), dtype=jnp.float64),
        jnp.asarray(np.asarray(unresolved, dtype=np.float64).reshape((-1, 2, 2))),
        jnp.zeros((0, 2, 3), dtype=jnp.float64),
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
    thresholds = tuple(finite_real_scalar(value, "normal.angle_thresholds") for value in angle_thresholds)
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
