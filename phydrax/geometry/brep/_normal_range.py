#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Whole-trim outward normal enclosures with threshold-directed subdivision."""

from __future__ import annotations

from collections import deque
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
from ..._validation import finite_real_scalar, positive_integer
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._interval_enclosure import (
    interval_divide,
    interval_multiply,
    interval_subtract,
    monotone,
)
from .._rational import fraction_down, fraction_up, rational_cross, rational_vector
from ._edge_classification import _material_orientations
from ._model import BRepEntityId, BRepModel
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    CylinderPatch,
    ExtrusionSurface,
    interval_cross,
    interval_dot,
    interval_norm,
    PlanePatch,
)
from ._query_membership import full_rectangle
from ._request_preparations import TrimRequestPreparations


BRepNormalStatus: TypeAlias = Literal[
    "positive", "insufficient", "vertical", "undercut", "unresolved"
]

_PI_UPPER = float(np.nextafter(pi, np.inf))
_HALF_PI = (np.nextafter(pi / 2, -np.inf), np.nextafter(pi / 2, np.inf))


@final
class BRepNormalRange(StrictModule):
    """Radians against the declared direction; singular points are spatial boxes.

    An unresolved trim cell is retained conservatively, never discarded by a
    point sample. A strict normal bound on that superset still proves a whole
    face decision. Singular charts never acquire an artificial normal.
    """

    face_id: BRepEntityId = eqx.field(static=True)
    model_id: str = eqx.field(static=True)
    status: BRepNormalStatus = eqx.field(static=True)
    angle_enclosure: Array
    unresolved_cells: Array
    singular_points: Array


def _direction(value: ArrayLike | tuple[float, float, float], /) -> np.ndarray:
    direction = np.asarray(value, dtype=np.float64)
    if direction.shape != (3,) or not np.all(np.isfinite(direction)):
        raise ValueError(
            "normal.direction must be a finite three-vector; got invalid shape or values"
        )
    scale = np.max(np.abs(direction))
    if scale == 0:
        raise ValueError("normal.direction must be nonzero; got zero")
    # Power-of-two rescaling preserves components and prevents norm overflow/underflow.
    _, exponent = np.frexp(scale)
    scaled = np.ldexp(direction, -int(exponent))
    if not np.array_equal(np.ldexp(scaled, int(exponent)), direction):
        raise ValueError(
            "normal.direction must retain its components under binary64 rescaling; got unsupported dynamic range"
        )
    return scaled


def _exact_vertical(patch: AbstractSurfacePatch, direction: np.ndarray, /) -> bool:
    if not isinstance(patch, (PlanePatch, CylinderPatch)):
        return False
    a = tuple(Fraction(float(x)) for x in np.asarray(patch.first_axis))
    b = tuple(Fraction(float(x)) for x in np.asarray(patch.second_axis))
    d = tuple(Fraction(float(x)) for x in direction)
    if isinstance(patch, PlanePatch):
        normal = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        return (
            any(normal)
            and sum((x * y for x, y in zip(normal, d, strict=True)), Fraction(0)) == 0
        )
    axis = tuple(Fraction(float(x)) for x in np.asarray(patch.axis))
    determinant = sum(
        (
            a[i]
            * (b[(i + 1) % 3] * axis[(i + 2) % 3] - b[(i + 2) % 3] * axis[(i + 1) % 3])
            for i in range(3)
        ),
        Fraction(0),
    )
    if determinant == 0 or float(patch.radius) == 0:
        return False
    # Cylinder tangents are radial-angular and axial. Both radial cross axial
    # basis vectors must be perpendicular to the query direction.
    return all(
        sum(
            (
                d[i]
                * (
                    v[(i + 1) % 3] * axis[(i + 2) % 3]
                    - v[(i + 2) % 3] * axis[(i + 1) % 3]
                )
                for i in range(3)
            ),
            Fraction(0),
        )
        == 0
        for v in (a, b)
    )


def _linear_extrusion_normal(
    patch: AbstractSurfacePatch, /
) -> tuple[np.ndarray, np.ndarray] | None:
    if not isinstance(patch, ExtrusionSurface) or not isinstance(
        patch.curve, BSplineCurve
    ):
        return None
    curve = patch.curve
    if curve.degree != 1 or curve.control_points.shape[0] != 2:
        return None
    weights = np.asarray(curve.weights)
    if not weights[0] == weights[1] > 0:
        return None
    knots = np.asarray(curve.knots)
    if not knots[0] == knots[1] < knots[2] == knots[3]:
        return None
    controls = np.asarray(curve.control_points)
    first, last = rational_vector(controls[0]), rational_vector(controls[1])
    width = Fraction(float(knots[2])) - Fraction(float(knots[1]))
    derivative = tuple((b - a) / width for a, b in zip(first, last, strict=True))
    normal = rational_cross(derivative, rational_vector(np.asarray(patch.direction)))
    return (
        np.asarray(tuple(fraction_down(value) for value in normal), dtype=np.float64),
        np.asarray(tuple(fraction_up(value) for value in normal), dtype=np.float64),
    )


def _cell_angle(
    patch: AbstractSurfacePatch,
    box: np.ndarray,
    orientation: float,
    direction: np.ndarray,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
    /,
) -> tuple[float, float]:
    if not np.isfinite(orientation) or not patch.is_c1_on(
        box, budget=budget, preparations=preparations
    ):
        return 0.0, _PI_UPPER
    if _exact_vertical(patch, direction):
        return float(_HALF_PI[0]), float(_HALF_PI[1])
    normal = _linear_extrusion_normal(patch)
    if normal is None:
        lower, upper = patch.derivative_bounds(
            box, budget=budget, preparations=preparations
        )
        normal = interval_cross((lower[:, 0], upper[:, 0]), (lower[:, 1], upper[:, 1]))
    if orientation < 0:
        normal = -normal[1], -normal[0]
    norm = interval_norm(normal)
    if norm[0] <= 0:
        return 0.0, _PI_UPPER
    dot = interval_dot(normal, (direction, direction))
    denominator = interval_multiply(norm, interval_norm((direction, direction)))
    cosine = interval_divide(dot, denominator)
    angle = monotone(np.arccos, False, 8)(
        (np.clip(cosine[0], -1, 1), np.clip(cosine[1], -1, 1))
    )
    return max(0.0, float(angle[0])), min(_PI_UPPER, float(angle[1]))


def _status(
    lower: float, upper: float, draft: float, vertical: bool, regular: bool, /
) -> BRepNormalStatus:
    if not regular:
        return "unresolved"
    if vertical:
        return "vertical"
    limit = interval_subtract(_HALF_PI, (np.float64(draft), np.float64(draft)))
    if upper < limit[0]:
        return "positive"
    if lower > limit[1] and upper < _HALF_PI[0]:
        return "insufficient"
    if lower > _HALF_PI[1]:
        return "undercut"
    return "unresolved"


def _trim_cell(
    model: BRepModel,
    face: int,
    box: np.ndarray,
    rectangle: bool,
    maximum: int,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
    /,
) -> bool | None:
    domain = model.trim_domains[face]
    if rectangle or domain is None:
        return True
    bounds = np.asarray(model.parameter_bounds)[face]
    # Model trim domains use the atlas's normalized chart, not carrier UV.
    # The extent is the binary64 coefficient declared by brep_trim_curve.
    extent = bounds[1] - bounds[0]
    reference = interval_divide(
        interval_subtract((box[0], box[1]), (bounds[0], bounds[0])),
        (extent, extent),
    )
    trim = domain.classify_boxes(
        np.stack(reference)[None],
        maximum_refinements=maximum,
        budget=budget,
        preparations=preparations,
    )
    return bool(trim.inside[0]) if trim.resolved[0] else None


def _singular_points(
    model: BRepModel,
    face: int,
    rectangle: bool,
    maximum: int,
    budget: CoordinateEnclosureBudget,
    preparations: TrimRequestPreparations,
    /,
) -> list[np.ndarray]:
    patch = model.patches[face]
    bounds = np.asarray(model.parameter_bounds)[face]
    points: list[np.ndarray] = []
    for axis, value in patch.degenerate_isolines(bounds):
        box = bounds.copy()
        box[:, axis] = value
        if (
            _trim_cell(model, face, box, rectangle, maximum, budget, preparations)
            is False
        ):
            continue
        points.append(patch.bounding_box(box, budget=budget, preparations=preparations))
    return points


def _face_range(
    model: BRepModel,
    face: int,
    direction: np.ndarray,
    draft: float,
    thresholds: tuple[float, ...],
    maximum: int,
    orientation: float,
    budget: CoordinateEnclosureBudget,
    /,
) -> BRepNormalRange:
    patch = model.patches[face]
    bounds = np.asarray(model.parameter_bounds)[face]
    geometry = model.require_geometry()
    preparations = TrimRequestPreparations()
    rectangle = full_rectangle(geometry, face, patch, bounds)
    vertical = _exact_vertical(patch, direction) and np.isfinite(orientation)
    pending = deque((bounds,))
    accepted: list[tuple[float, float]] = []
    unresolved: list[np.ndarray] = []
    evaluated = 0
    while pending:
        box = pending.popleft()
        trim = _trim_cell(model, face, box, rectangle, maximum, budget, preparations)
        if trim is False:
            continue
        angle = _cell_angle(patch, box, orientation, direction, budget, preparations)
        evaluated += 1
        straddles = any(angle[0] <= threshold <= angle[1] for threshold in thresholds)
        # Reserve enough slots for every pending leaf before splitting this one.
        if straddles and not vertical and evaluated + len(pending) + 2 <= maximum:
            axis = int(np.argmax((box[1] - box[0]) / (bounds[1] - bounds[0])))
            middle = box[0, axis] + (box[1, axis] - box[0, axis]) / 2
            if box[0, axis] < middle < box[1, axis]:
                left, right = box.copy(), box.copy()
                left[1, axis], right[0, axis] = middle, middle
                pending.extend((right, left))
                continue
        accepted.append(angle)
        if straddles and not vertical or trim is None:
            unresolved.append(box)
    singular = _singular_points(model, face, rectangle, maximum, budget, preparations)
    lower = min((a for a, _ in accepted), default=0.0)
    upper = max((b for _, b in accepted), default=_PI_UPPER)
    status = _status(
        lower,
        upper,
        draft,
        bool(vertical),
        not singular and bool(np.isfinite(orientation)),
    )
    return BRepNormalRange(
        model.face_ids[face],
        model.model_id,
        status,
        jnp.asarray((lower, upper), dtype=jnp.float64),
        jnp.asarray(np.asarray(unresolved, dtype=np.float64).reshape((-1, 2, 2))),
        jnp.asarray(np.asarray(singular, dtype=np.float64).reshape((-1, 2, 3))),
    )


@checked
def brep_normal_ranges(
    model: BRepModel,
    direction: ArrayLike | tuple[float, float, float],
    /,
    *,
    draft_angle: float,
    maximum_cells: int,
    coordinate_budget: CoordinateEnclosureBudget,
    angle_thresholds: tuple[float, ...] = (),
) -> tuple[BRepNormalRange, ...]:
    """Certify draft/overhang against a pull/build direction in radians.

    Positive means angle < pi/2 - draft_angle; insufficient lies strictly
    between that limit and pi/2; vertical has a zero rational dot; undercut exceeds
    pi/2. Equality at a declared draft limit is unresolved. Extra thresholds
    drive refinement for geometric selection, without an accuracy setting.
    """
    if not x64_enabled():
        raise ValueError("normal.precision must enable JAX binary64; got x64 disabled")
    maximum = positive_integer(maximum_cells, "normal.maximum_cells")
    draft = finite_real_scalar(draft_angle, "normal.draft_angle")
    if not 0 <= draft < pi / 2:
        raise ValueError(f"normal.draft_angle must lie in [0, pi/2); got {draft}")
    thresholds = tuple(
        finite_real_scalar(x, "normal.angle_thresholds") for x in angle_thresholds
    )
    if any(not 0 <= x <= pi for x in thresholds):
        raise ValueError(f"normal.angle_thresholds must lie in [0, pi]; got {thresholds}")
    count = len(model.patches)
    coordinate_budget.reserve(count * maximum * 1024, count * maximum * 8192)
    vector = _direction(direction)
    orientations = _material_orientations(model)
    return tuple(
        _face_range(
            model,
            face,
            vector,
            draft,
            (pi / 2 - draft, pi / 2, *thresholds),
            maximum,
            float(orientations[face]),
            coordinate_budget,
        )
        for face in range(count)
    )
