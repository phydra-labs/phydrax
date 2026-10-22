#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify distance lower bounds of intersection edges on two analytic carriers.

An intersection edge ``E`` lies on the variety ``V = {g1 = 0, g2 = 0}`` of the
implicit equations of its two generating carriers (plane, circular cylinder,
cone, sphere or torus with an exactly orthonormal binary64 frame). For a query
point ``p``, a candidate edge point at distance ``d`` and the ambiguity
tolerance ``t``, let ``K`` be the cube of half-width ``d + t`` about ``p``;
every point outside ``K`` is farther than ``d + t``. A closest point of ``E``
inside ``K`` is either an end of the arc or a stationary point of
``|q - p|^2`` on ``V``; such stationary points and the singular points of ``V`` are zeros of

    F(q) = (g1(q), g2(q), (q - p) . (grad g1(q) x grad g2(q))).

``K`` is covered by boxes, each certified in outward-rounded interval
arithmetic to lie farther than ``d + t``, to miss a carrier, to miss the
stationarity equation, to contain no zero of ``F`` (empty Krawczyk image) or
to contain exactly one zero of ``F`` (Krawczyk image inside the box; then the
interval Jacobian is regular, so ``V`` is smooth there and the root distance
is enclosed). The minimum over those bounds, ``d`` and the arc-end enclosures
is a rigorous lower bound of the edge distance, and every stationary root
within ``d + t`` is enumerated. The argument never uses the arc's chart
parametrization, so another branch or sheet of ``V`` near ``p`` can only lower
the bound or add a root; a second root within ``d + t`` is an undecided tie or
membership question, and singular or non-transversal neighbourhoods exhaust
the box allowance and stay undecided. Undecided rows keep the general
interval route.
"""

from __future__ import annotations

import numpy as np

import phydrax.geometry.brep._analytic_carrier_closest as analytic_carrier_closest_owner

from ..._meshcore import NativeBatchTicket
from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ._intersection_curve import IntersectionCurve
from ._patches import ConePatch, CylinderPatch, PlanePatch, SpherePatch, TorusPatch


type _Interval = tuple[np.ndarray, np.ndarray]
type _Carrier = PlanePatch | CylinderPatch | ConePatch | SpherePatch | TorusPatch

_add = analytic_carrier_closest_owner.interval_add
_subtract = analytic_carrier_closest_owner.interval_subtract
_multiply = analytic_carrier_closest_owner.interval_multiply
_square_interval = analytic_carrier_closest_owner._square
_tan_interval = analytic_carrier_closest_owner._tan

# Interval operations of one F/Jacobian evaluation per box, used as work units.
_EVALUATION_WORK = 4
# Float64 lanes held live per box during one evaluation and Krawczyk step.
_BOX_WORKSPACE_BYTES = 8 * 1024


def _exact(value: float | np.ndarray, /) -> _Interval:
    array = np.asarray(value, dtype=np.float64)
    return array, array


def _column(value: _Interval, /) -> _Interval:
    return np.asarray(value[0])[..., None], np.asarray(value[1])[..., None]


class _Jet:
    """First-order interval jet ``(value, world gradient)`` over row boxes."""

    __slots__ = ("gradient", "value")

    def __init__(self, value: _Interval, gradient: _Interval, /) -> None:
        self.value, self.gradient = value, gradient

    def __add__(self, other: _Jet, /) -> _Jet:
        return _Jet(_add(self.value, other.value), _add(self.gradient, other.gradient))

    def __sub__(self, other: _Jet, /) -> _Jet:
        return _Jet(
            _subtract(self.value, other.value), _subtract(self.gradient, other.gradient)
        )

    def __mul__(self, other: _Jet, /) -> _Jet:
        return _Jet(
            _multiply(self.value, other.value),
            _add(
                _multiply(_column(self.value), other.gradient),
                _multiply(_column(other.value), self.gradient),
            ),
        )

    def scaled(self, factor: _Interval, /) -> _Jet:
        return _Jet(
            _multiply(self.value, factor), _multiply(self.gradient, _column(factor))
        )

    def shifted(self, offset: _Interval, /) -> _Jet:
        return _Jet(_add(self.value, offset), self.gradient)

    def squared(self) -> _Jet:
        twice = _multiply(self.value, _exact(2.0))
        return _Jet(
            _square_interval(self.value), _multiply(_column(twice), self.gradient)
        )


def _constant(value: _Interval, count: int, /) -> _Jet:
    zero = np.zeros((count, 3), dtype=np.float64)
    return _Jet(
        (
            np.broadcast_to(value[0], (count,)).copy(),
            np.broadcast_to(value[1], (count,)).copy(),
        ),
        (zero, zero.copy()),
    )


def _world(lower: np.ndarray, upper: np.ndarray, /) -> tuple[_Jet, _Jet, _Jet]:
    """The three world coordinates of row boxes as jets with unit gradients."""
    count = lower.shape[0]
    jets = []
    for axis in range(3):
        unit = np.zeros((count, 3), dtype=np.float64)
        unit[:, axis] = 1.0
        jets.append(_Jet((lower[:, axis], upper[:, axis]), (unit, unit.copy())))
    return jets[0], jets[1], jets[2]


def _frame_coordinate(
    world: tuple[_Jet, _Jet, _Jet], origin: np.ndarray, axis: np.ndarray, /
) -> _Jet:
    """``(q - origin) . axis`` for one binary64 axis."""
    total: _Jet | None = None
    for k in range(3):
        if axis[k] == 0.0:
            continue
        term = world[k].shifted(_exact(-origin[k])).scaled(_exact(axis[k]))
        total = term if total is None else total + term
    if total is None:
        raise ValueError("Analytic edge carrier frame axes must be nonzero.")
    return total


class _ImplicitCarrier:
    """Interval jets of one carrier's implicit equation and world gradient."""

    def __init__(self, patch: _Carrier, /) -> None:
        self.patch = patch
        match patch:
            case PlanePatch():
                self.origin = np.asarray(patch.origin, dtype=np.float64)
                first = np.asarray(patch.first_axis, dtype=np.float64)
                second = np.asarray(patch.second_axis, dtype=np.float64)
                self.normal = tuple(
                    _subtract(
                        _multiply(
                            _exact(first[(k + 1) % 3]), _exact(second[(k + 2) % 3])
                        ),
                        _multiply(
                            _exact(first[(k + 2) % 3]), _exact(second[(k + 1) % 3])
                        ),
                    )
                    for k in range(3)
                )
            case CylinderPatch() | ConePatch() | SpherePatch() | TorusPatch():
                self.origin = np.asarray(
                    patch.center
                    if isinstance(patch, (SpherePatch, TorusPatch))
                    else patch.origin,
                    dtype=np.float64,
                )
                self.frame = tuple(
                    np.asarray(axis, dtype=np.float64)
                    for axis in (patch.first_axis, patch.second_axis, patch.axis)
                )

    def evaluate(
        self, world: tuple[_Jet, _Jet, _Jet], /
    ) -> tuple[_Jet, tuple[_Jet, _Jet, _Jet]]:
        count = world[0].value[0].shape[0]
        patch = self.patch
        if isinstance(patch, PlanePatch):
            value: _Jet | None = None
            for k in range(3):
                term = world[k].shifted(_exact(-self.origin[k])).scaled(self.normal[k])
                value = term if value is None else value + term
            assert value is not None
            gradient = tuple(_constant(component, count) for component in self.normal)
            return value, (gradient[0], gradient[1], gradient[2])
        x, y, z = (_frame_coordinate(world, self.origin, axis) for axis in self.frame)
        two = _exact(2.0)
        match patch:
            case CylinderPatch():
                radius = _square_interval(_exact(float(patch.radius)))
                value = (x.squared() + y.squared()).shifted(_negate(radius))
                partials = (x.scaled(two), y.scaled(two), None)
            case ConePatch():
                slope = _tan_interval(_exact(float(patch.semi_angle)))
                radius = z.scaled(slope).shifted(_exact(float(patch.reference_radius)))
                value = x.squared() + y.squared() - radius.squared()
                partials = (
                    x.scaled(two),
                    y.scaled(two),
                    radius.scaled(_multiply(slope, _exact(-2.0))),
                )
            case SpherePatch():
                radius = _square_interval(_exact(float(patch.radius)))
                value = (x.squared() + y.squared() + z.squared()).shifted(_negate(radius))
                partials = (x.scaled(two), y.scaled(two), z.scaled(two))
            case TorusPatch():
                major = _square_interval(_exact(float(patch.major_radius)))
                minor = _square_interval(_exact(float(patch.minor_radius)))
                planar = x.squared() + y.squared()
                total = (planar + z.squared()).shifted(_subtract(major, minor))
                value = total.squared() - planar.scaled(_multiply(major, _exact(4.0)))
                four = _exact(4.0)
                radial = total.shifted(_multiply(major, _exact(-2.0)))
                partials = (
                    (x * radial).scaled(four),
                    (y * radial).scaled(four),
                    (z * total).scaled(four),
                )
            case _:
                raise TypeError(
                    f"Analytic edge carrier must be an analytic surface; got {type(patch).__name__}."
                )
        components = []
        for k in range(3):
            component: _Jet | None = None
            for partial, axis in zip(partials, self.frame, strict=True):
                if partial is None or axis[k] == 0.0:
                    continue
                term = partial.scaled(_exact(axis[k]))
                component = term if component is None else component + term
            components.append(
                _constant(_exact(0.0), count) if component is None else component
            )
        return value, (components[0], components[1], components[2])


def _negate(value: _Interval, /) -> _Interval:
    return -value[1], -value[0]


def surface_pair_carriers(
    first: object, second: object, /
) -> tuple[_ImplicitCarrier, _ImplicitCarrier] | None:
    """Implicit carriers of two admitted analytic patches, or ``None``."""
    analytic = (PlanePatch, CylinderPatch, ConePatch, SpherePatch, TorusPatch)
    if not (
        isinstance(first, analytic)
        and isinstance(second, analytic)
        and analytic_carrier_closest_owner.carrier_supported(first)
        and analytic_carrier_closest_owner.carrier_supported(second)
    ):
        return None
    return _ImplicitCarrier(first), _ImplicitCarrier(second)


def edge_carriers(curve: object, /) -> tuple[_ImplicitCarrier, _ImplicitCarrier] | None:
    """Implicit carriers of an intersection curve on two admitted analytic patches."""
    if not isinstance(curve, IntersectionCurve):
        return None
    return surface_pair_carriers(curve.first.patch, curve.second.patch)


def _system(
    carriers: tuple[_ImplicitCarrier, _ImplicitCarrier],
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    /,
) -> tuple[_Interval, _Interval]:
    """Enclose ``F`` (rows, 3) and its Jacobian (rows, 3, 3) over row boxes."""
    world = _world(lower, upper)
    first, first_gradient = carriers[0].evaluate(world)
    second, second_gradient = carriers[1].evaluate(world)
    stationary: _Jet | None = None
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        cross = first_gradient[i] * second_gradient[j] - (
            first_gradient[j] * second_gradient[i]
        )
        term = world[k].shifted(_exact(-points[:, k])) * cross
        stationary = term if stationary is None else stationary + term
    assert stationary is not None
    jets = (first, second, stationary)
    value = (
        np.stack([np.asarray(jet.value[0]) for jet in jets], axis=1),
        np.stack([np.asarray(jet.value[1]) for jet in jets], axis=1),
    )
    jacobian = (
        np.stack([np.asarray(jet.gradient[0]) for jet in jets], axis=1),
        np.stack([np.asarray(jet.gradient[1]) for jet in jets], axis=1),
    )
    return value, jacobian


def _distance_lower(
    points: np.ndarray, lower: np.ndarray, upper: np.ndarray, /
) -> np.ndarray:
    """Outward lower bound of the Euclidean distance from points to row boxes."""
    below = _subtract(_exact(lower), _exact(points))[0]
    above = _subtract(_exact(points), _exact(upper))[0]
    gap = np.maximum(np.maximum(below, above), 0.0)
    squared = _square_interval((gap, gap))[0]
    total = _exact(np.zeros(points.shape[0], dtype=np.float64))
    for k in range(3):
        total = _add(total, (squared[:, k], squared[:, k]))
    root = np.sqrt(np.maximum(total[0], 0.0))
    return np.maximum(np.nextafter(root, -np.inf), 0.0)


def _krawczyk(
    carriers: tuple[_ImplicitCarrier, _ImplicitCarrier],
    points: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    jacobian: _Interval,
    /,
) -> tuple[_Interval, np.ndarray]:
    """Krawczyk image of row boxes about their midpoints and a finite mask."""
    center = 0.5 * lower + 0.5 * upper
    center = np.minimum(np.maximum(center, lower), upper)
    center_value, center_jacobian = _system(carriers, points, center, center)
    middle = 0.5 * center_jacobian[0] + 0.5 * center_jacobian[1]
    finite = np.all(np.isfinite(middle), axis=(1, 2))
    determinant = np.linalg.det(np.where(finite[:, None, None], middle, np.eye(3)))
    scale = np.max(np.abs(np.where(finite[:, None, None], middle, 1.0)), axis=(1, 2))
    regular = finite & (np.abs(determinant) > 1.0e-12 * np.maximum(scale, 1.0e-300) ** 3)
    inverse = np.linalg.inv(np.where(regular[:, None, None], middle, np.eye(3)))
    displacement = (
        _subtract(_exact(lower), _exact(center))[0],
        _subtract(_exact(upper), _exact(center))[1],
    )
    image_lower, image_upper = [], []
    for i in range(3):
        total = _exact(center[:, i])
        for j in range(3):
            factor = _exact(inverse[:, i, j])
            total = _subtract(
                total,
                _multiply(factor, (center_value[0][:, j], center_value[1][:, j])),
            )
        for k in range(3):
            coefficient = _exact(np.full(lower.shape[0], float(i == k)))
            for j in range(3):
                coefficient = _subtract(
                    coefficient,
                    _multiply(
                        _exact(inverse[:, i, j]),
                        (jacobian[0][:, j, k], jacobian[1][:, j, k]),
                    ),
                )
            total = _add(
                total,
                _multiply(coefficient, (displacement[0][:, k], displacement[1][:, k])),
            )
        image_lower.append(total[0])
        image_upper.append(total[1])
    image = np.stack(image_lower, axis=1), np.stack(image_upper, axis=1)
    usable = regular & np.all(np.isfinite(image[0]) & np.isfinite(image[1]), axis=1)
    return image, usable


def _complement(
    cube_lower: np.ndarray,
    cube_upper: np.ndarray,
    hole_lower: np.ndarray,
    hole_upper: np.ndarray,
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Up to six closed slabs covering ``cube \\ hole`` row-wise."""
    low = np.minimum(np.maximum(hole_lower, cube_lower), cube_upper)
    high = np.maximum(np.minimum(hole_upper, cube_upper), cube_lower)
    lowers, uppers, valid = [], [], []
    for axis in range(3):
        for side in (0, 1):
            slab_lower, slab_upper = cube_lower.copy(), cube_upper.copy()
            # Earlier axes are already restricted to the hole's extent.
            slab_lower[:, :axis] = low[:, :axis]
            slab_upper[:, :axis] = high[:, :axis]
            if side == 0:
                slab_upper[:, axis] = low[:, axis]
            else:
                slab_lower[:, axis] = high[:, axis]
            lowers.append(slab_lower)
            uppers.append(slab_upper)
            valid.append(np.all(slab_lower < slab_upper, axis=1))
    return np.stack(lowers, axis=1), np.stack(uppers, axis=1), np.stack(valid, axis=1)


class EdgeLowerBounds:
    """Row-wise certified lower bounds and the rows they decide."""

    __slots__ = ("boxes", "decided", "lower")

    def __init__(
        self, lower: np.ndarray, decided: np.ndarray, boxes: np.ndarray, /
    ) -> None:
        self.lower, self.decided, self.boxes = lower, decided, boxes


def edge_distance_lower_bounds(
    carriers: tuple[_ImplicitCarrier, _ImplicitCarrier],
    points: np.ndarray,
    candidates: np.ndarray,
    distances: np.ndarray,
    tolerance: float,
    /,
    *,
    end_boxes: np.ndarray,
    maximum_boxes: int,
    maximum_depth: int,
    budget: CoordinateEnclosureBudget,
) -> EdgeLowerBounds:
    """Certify ``distance - lower <= tolerance`` for candidate edge points.

    ``candidates`` are points of the edge at binary64 ``distances`` from
    ``points``; ``end_boxes`` (k, 2, 3) enclose the arc ends (empty for a
    full closed branch). A row is decided when its certified lower bound is at
    least ``distance - tolerance`` and no second stationary root of the
    carrier variety competes within the tolerance. ``boxes`` counts the
    certified boxes of each row.
    """
    count = points.shape[0]
    match budget.ticket:
        case NativeBatchTicket() as ticket:
            if count > ticket.remaining_geometry_queries:
                ticket.refuse(0, queries_upper=count)
            ticket.consume(geometry_queries=count)
        case _:
            pass
    lower_bound = np.zeros((count,), dtype=np.float64)
    decided = np.zeros((count,), dtype=np.bool_)
    work = np.zeros((count,), dtype=np.int64)
    finite = np.isfinite(distances) & np.all(np.isfinite(candidates), axis=1)
    trivial = finite & (distances <= tolerance)
    decided[trivial] = True
    active = np.flatnonzero(finite & ~trivial)
    if active.size == 0:
        return EdgeLowerBounds(lower_bound, decided, work)
    target = np.asarray(distances[active] - tolerance, dtype=np.float64)
    query = np.asarray(points[active], dtype=np.float64)
    radius = np.asarray(distances[active], dtype=np.float64)
    # Every point outside the closed cube lies farther than the candidate.
    best = radius.copy()
    if end_boxes.size:
        for box in end_boxes:
            ends = _distance_lower(
                query,
                np.broadcast_to(box[0], query.shape),
                np.broadcast_to(box[1], query.shape),
            )
            best = np.minimum(best, ends)
    reach = _add(_exact(radius), _exact(tolerance))[1][:, None]
    cube_lower = _subtract(_exact(query), _exact(reach))[0]
    cube_upper = _add(_exact(query), _exact(reach))[1]
    hole = 0.5 * radius[:, None]
    seed_lower = _subtract(_exact(candidates[active]), _exact(hole))[0]
    seed_upper = _add(_exact(candidates[active]), _exact(hole))[1]
    slab_lower, slab_upper, slab_valid = _complement(
        cube_lower, cube_upper, seed_lower, seed_upper
    )
    local = np.arange(active.size)
    slab_rows = np.broadcast_to(local[:, None], slab_valid.shape)[slab_valid]
    lower = np.concatenate((seed_lower, slab_lower[slab_valid]))
    upper = np.concatenate((seed_upper, slab_upper[slab_valid]))
    rows = np.concatenate((local, slab_rows))
    depth = np.zeros(rows.shape, dtype=np.int64)
    failed = np.zeros((active.size,), dtype=np.bool_)
    competing = np.zeros((active.size,), dtype=np.int64)
    used = np.zeros((active.size,), dtype=np.int64)
    with budget.temporary_scope():
        while rows.size:
            budget.reserve(
                _EVALUATION_WORK * int(rows.size),
                _BOX_WORKSPACE_BYTES * int(rows.size),
            )
            np.add.at(used, rows, 1)
            distance = _distance_lower(query[rows], lower, upper)
            # Boxes beyond the tie radius hold no competing point; every
            # stationary root at most one tolerance past the candidate is kept.
            far = distance > radius[rows] + tolerance
            np.minimum.at(best, rows[far], distance[far])
            keep = ~far & ~failed[rows]
            rows, lower, upper, depth = rows[keep], lower[keep], upper[keep], depth[keep]
            if not rows.size:
                break
            value, jacobian = _system(carriers, query[rows], lower, upper)
            missed = np.any((value[0] > 0.0) | (value[1] < 0.0), axis=1)
            keep = ~missed
            rows, lower, upper, depth = rows[keep], lower[keep], upper[keep], depth[keep]
            jacobian = (jacobian[0][keep], jacobian[1][keep])
            if not rows.size:
                break
            image, usable = _krawczyk(carriers, query[rows], lower, upper, jacobian)
            empty = usable & np.any((image[0] > upper) | (image[1] < lower), axis=1)
            unique = (
                usable & ~empty & np.all((image[0] > lower) & (image[1] < upper), axis=1)
            )
            if np.any(unique):
                root_lower, root_upper = image[0][unique], image[1][unique]
                root_rows = rows[unique]
                # Contract the certified root box before bounding its distance.
                for _ in range(3):
                    _, root_jacobian = _system(
                        carriers, query[root_rows], root_lower, root_upper
                    )
                    refined, refined_usable = _krawczyk(
                        carriers, query[root_rows], root_lower, root_upper, root_jacobian
                    )
                    root_lower = np.where(
                        refined_usable[:, None],
                        np.maximum(root_lower, refined[0]),
                        root_lower,
                    )
                    root_upper = np.where(
                        refined_usable[:, None],
                        np.minimum(root_upper, refined[1]),
                        root_upper,
                    )
                root_distance = _distance_lower(query[root_rows], root_lower, root_upper)
                np.minimum.at(best, root_rows, root_distance)
                near = root_distance <= radius[root_rows] + tolerance
                np.add.at(competing, root_rows[near], 1)
            keep = ~(empty | unique)
            rows, lower, upper, depth = rows[keep], lower[keep], upper[keep], depth[keep]
            exhausted = (depth >= maximum_depth) | (used[rows] >= maximum_boxes)
            failed[rows[exhausted]] = True
            keep = ~exhausted
            rows, lower, upper, depth = rows[keep], lower[keep], upper[keep], depth[keep]
            if not rows.size:
                break
            width = upper - lower
            axis = np.argmax(width, axis=1)
            selected = np.arange(rows.size)
            middle = 0.5 * lower[selected, axis] + 0.5 * upper[selected, axis]
            split = (middle > lower[selected, axis]) & (middle < upper[selected, axis])
            failed[rows[~split]] = True
            left_upper = upper.copy()
            left_upper[selected, axis] = middle
            right_lower = lower.copy()
            right_lower[selected, axis] = middle
            rows = np.concatenate((rows, rows))
            lower = np.concatenate((lower, right_lower))
            upper = np.concatenate((left_upper, upper))
            depth = np.concatenate((depth, depth)) + 1
            keep = ~failed[rows]
            rows, lower, upper, depth = rows[keep], lower[keep], upper[keep], depth[keep]
    lower_bound[active] = np.minimum(best, radius)
    decided[active] = ~failed & (competing <= 1) & (best >= target)
    work[active] = used
    return EdgeLowerBounds(lower_bound, decided, work)


__all__ = [
    "EdgeLowerBounds",
    "edge_carriers",
    "edge_distance_lower_bounds",
    "surface_pair_carriers",
]
