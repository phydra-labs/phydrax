#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified membership of a complete closed-meridian revolution solid.

A full-turn revolution of a B-spline meridian ``c`` about an exactly unit
axis, or a normal offset of one, is the rotation of one closed curve ``w``:
``c`` itself, or its exact parallel curve ``c + d n``. Its solid is rotation
invariant, so with ``z = (x - o) . a`` and ``rho = |(x - o) x a|`` a point
belongs to it iff ``(rho_p, z_p)`` lies inside the closed half-plane curve
``(rho_w, z_w)``, and its boundary distance is the half-plane distance to
that curve. Parity counts transversal crossings of one half-plane half-line
(radially outward, or along the axis up or down) by one-dimensional
subdivision of the meridian cycle, from the certified meridian jets that
own the source's construction bounds. No surface interval extension of the
rotated normal is formed.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Literal

import numpy as np

from ._patches import (
    _closure_tangent_continuous,
    _coordinate_budget,
    _directed_sum_of_squares,
    _exact_unit,
    _fraction_down,
    _fraction_up,
    _interval_cross,
    _interval_dot,
    _interval_norm,
    _junction_rows,
    _meridian_frame,
    _meridian_jets,
    _meridian_spans,
    _MeridianFrame,
    _MeridianSpans,
    _parallel_meridian_jets,
    _revolution_meridian,
    AbstractSurfacePatch,
    BSplineCurve,
)
from ._sphere_membership import _square_root_interval


# Equal meridian ranges of the initial cycle cover; bisection refines them.
_INITIAL_RANGES = 16
# Cycle phases tried as the cover's start. Roots sit at symmetric (dyadic)
# meridian parameters, so the starts and their equal cuts avoid those; a cover
# with a cut on a root is abandoned for the next start.
_START_PHASES = (0.0, 1.0 / 3.0, 1.0 / 7.0, 3.0 / 11.0, 5.0 / 13.0)

type _Direction = Literal["outward", "up", "down"]


@dataclass(frozen=True, slots=True)
class RevolutionMembership:
    """Parity membership with half-plane boundary distance bounds.

    ``complete`` is False when the point is not separated from the boundary by
    more than the tolerance, or when no half-line was certified within the row
    allowance; ``exhausted`` reports that the allowance ran out.
    """

    inside: bool
    complete: bool
    distance_lower: float
    distance_upper: float
    operations: int
    exhausted: bool


@dataclass(frozen=True, slots=True)
class _RevolutionSource:
    origin: np.ndarray
    axis: np.ndarray
    spans: _MeridianSpans
    frame: _MeridianFrame | None
    distance: float
    first: float
    last: float


def prepare_full_revolution(
    patch: AbstractSurfacePatch, lower: float, upper: float, /
) -> _RevolutionSource | None:
    """Prove an exactly closed meridian over ``[lower, upper]`` about a unit axis.

    The caller establishes one complete turn and closed solid incidence. An
    offset's parallel meridian also needs its closure's end legs exactly
    positively parallel, so both end normals agree.
    """
    resolved = _revolution_meridian(patch)
    if resolved is None:
        return None
    revolution, distance = resolved
    curve = revolution.curve
    axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    if (
        not isinstance(curve, BSplineCurve)
        or curve.parameter_domain != (lower, upper)
        or not _exact_unit(axis)
        or not _closure_tangent_continuous(
            revolution, 1, lower, upper, tangent=distance != 0.0
        )
    ):
        return None
    budget = _coordinate_budget()
    frame = None
    if distance != 0.0:
        frame = _meridian_frame(revolution, budget)
        if frame is None:
            return None
        spans = frame.spans
    else:
        spans = _meridian_spans(curve, budget)
        if spans is None:
            return None
    return _RevolutionSource(
        np.asarray(revolution.axis_origin, dtype=np.float64),
        axis,
        spans,
        frame,
        distance,
        lower,
        upper,
    )


@dataclass(frozen=True, slots=True)
class _Enclosure:
    """Row boxes of ``rho``, ``z``, their derivative signs and validity."""

    rho: tuple[np.ndarray, np.ndarray]
    height: tuple[np.ndarray, np.ndarray]
    radial_rate: tuple[np.ndarray, np.ndarray]
    height_rate: tuple[np.ndarray, np.ndarray]
    valid: np.ndarray


def _enclose(
    source: _RevolutionSource, first: np.ndarray, last: np.ndarray, /
) -> _Enclosure:
    """Half-plane boxes of ``w`` over closed meridian ranges, row-wise.

    ``radial_rate`` encloses ``((w - o) x a) . (w' x a) = rho rho'``, whose
    sign is that of ``rho'`` wherever ``rho > 0``.
    """
    budget = _coordinate_budget()
    origin = source.origin
    if source.frame is None:
        jets = _meridian_jets(source.spans, first, last, 1, budget)
        position = jets.hull(0)
        relative = (
            np.nextafter(position[0] - origin, -np.inf),
            np.nextafter(position[1] - origin, np.inf),
        )
        derivative = jets.hull(1)
        valid = np.ones(first.shape, dtype=np.bool_)
    else:
        jets = _meridian_jets(source.spans, first, last, 2, budget)
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            parallel, regular = _parallel_meridian_jets(
                source.frame, jets.jets, source.distance
            )
        relative, derivative = (
            (
                np.minimum.reduceat(lower, jets.offsets, axis=0),
                np.maximum.reduceat(upper, jets.offsets, axis=0),
            )
            for lower, upper in parallel[:2]
        )
        valid = np.logical_and.reduceat(regular, jets.offsets) & ~_junction_rows(
            source.spans, first, last, source.spans.g1, closed=False
        )
    axis = (source.axis, source.axis)
    with np.errstate(invalid="ignore", over="ignore"):
        radial = _interval_cross(relative, axis)
        rho = _interval_norm(radial)
        height = _interval_dot(relative, axis)
        radial_rate = _interval_dot(radial, _interval_cross(derivative, axis))
        height_rate = _interval_dot(derivative, axis)
    valid &= np.all(np.isfinite(relative[0]) & np.isfinite(relative[1]), axis=1)
    valid &= np.all(np.isfinite(derivative[0]) & np.isfinite(derivative[1]), axis=1)
    return _Enclosure(rho, height, radial_rate, height_rate, valid)


def _signs(
    value: tuple[np.ndarray, np.ndarray], bounds: tuple[float, float], /
) -> np.ndarray:
    """Strict sign of ``value - target`` for a target inside ``bounds``; 0 if not."""
    return np.where(value[0] > bounds[1], 1, np.where(value[1] < bounds[0], -1, 0))


def _strict(value: tuple[np.ndarray, np.ndarray], /) -> np.ndarray:
    return np.where(value[0] > 0, 1, np.where(value[1] < 0, -1, 0))


@dataclass(frozen=True, slots=True)
class _Cycle:
    """Phases ``tau`` in ``[start, start + 1]`` of the closed meridian cycle."""

    source: _RevolutionSource
    start: float

    def parameter(self, phase: float, /) -> float:
        """Meridian parameter of a phase in ``[0, 1]``."""
        span = self.source.last - self.source.first
        return self.source.last if phase == 1.0 else self.source.first + phase * span

    def pieces(self, first: float, last: float, /) -> tuple[tuple[float, float], ...]:
        if last <= 1.0:
            return ((self.parameter(first), self.parameter(last)),)
        if first >= 1.0:
            return ((self.parameter(first - 1.0), self.parameter(last - 1.0)),)
        # The exact closure glues the cycle's two parameter ends.
        return (
            (self.parameter(first), self.source.last),
            (self.source.first, self.parameter(last - 1.0)),
        )

    def endpoint(self, tau: float, /) -> float:
        return self.parameter(tau - 1.0 if tau > 1.0 else tau)


class _Rows:
    """Batched range and endpoint enclosures with a shared row allowance."""

    def __init__(self, source: _RevolutionSource, allowance: int, /) -> None:
        self.source = source
        self.allowance = allowance
        self.used = 0
        self.exhausted = False

    def admit(self, count: int, /) -> bool:
        if self.used + count > self.allowance:
            self.exhausted = True
            return False
        self.used += count
        return True

    def ranges(
        self, cycle: _Cycle, rows: list[tuple[float, float]], /
    ) -> _Enclosure | None:
        pieces = [cycle.pieces(first, last) for first, last in rows]
        offsets = np.cumsum([0, *(len(row) for row in pieces[:-1])])
        flat = np.asarray([piece for row in pieces for piece in row], dtype=np.float64)
        if not self.admit(flat.shape[0]):
            return None
        enclosure = _enclose(self.source, flat[:, 0], flat[:, 1])

        def hull(
            value: tuple[np.ndarray, np.ndarray],
        ) -> tuple[np.ndarray, np.ndarray]:
            return (
                np.minimum.reduceat(value[0], offsets),
                np.maximum.reduceat(value[1], offsets),
            )

        return _Enclosure(
            hull(enclosure.rho),
            hull(enclosure.height),
            hull(enclosure.radial_rate),
            hull(enclosure.height_rate),
            np.logical_and.reduceat(enclosure.valid, offsets),
        )

    def points(self, parameters: np.ndarray, /) -> _Enclosure | None:
        if not self.admit(parameters.shape[0]):
            return None
        return _enclose(self.source, parameters, parameters)


def _distance_bounds(
    enclosure: _Enclosure,
    rho: tuple[float, float],
    height: tuple[float, float],
    /,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise outward half-plane distance bounds from the query to each box."""
    with np.errstate(invalid="ignore", over="ignore"):
        near = np.stack(
            (
                np.maximum(
                    0.0,
                    np.maximum(
                        np.nextafter(enclosure.rho[0] - rho[1], -np.inf),
                        np.nextafter(rho[0] - enclosure.rho[1], -np.inf),
                    ),
                ),
                np.maximum(
                    0.0,
                    np.maximum(
                        np.nextafter(enclosure.height[0] - height[1], -np.inf),
                        np.nextafter(height[0] - enclosure.height[1], -np.inf),
                    ),
                ),
            ),
            axis=-1,
        )
        far = np.stack(
            (
                np.nextafter(
                    np.maximum(enclosure.rho[1] - rho[0], rho[1] - enclosure.rho[0]),
                    np.inf,
                ),
                np.nextafter(
                    np.maximum(
                        enclosure.height[1] - height[0], height[1] - enclosure.height[0]
                    ),
                    np.inf,
                ),
            ),
            axis=-1,
        )
        lower = np.maximum(
            np.nextafter(np.sqrt(_directed_sum_of_squares(near, -np.inf)), -np.inf), 0.0
        )
        upper = np.nextafter(np.sqrt(_directed_sum_of_squares(far, np.inf)), np.inf)
    valid = enclosure.valid & np.isfinite(upper)
    return np.where(valid, lower, 0.0), np.where(valid, upper, np.inf)


def _half_line(
    enclosure: _Enclosure,
    direction: _Direction,
    rho: tuple[float, float],
    height: tuple[float, float],
    /,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crossing function sign, half-line side and monotone rate sign per row."""
    if direction == "outward":
        return (
            _signs(enclosure.height, height),
            _signs(enclosure.rho, rho),
            _strict(enclosure.height_rate),
        )
    side = _signs(enclosure.height, height)
    rate = np.where(enclosure.rho[0] > 0.0, _strict(enclosure.radial_rate), 0)
    return (
        _signs(enclosure.rho, rho),
        side if direction == "up" else -side,
        rate,
    )


def _crossings(
    rows_owner: _Rows,
    cycle: _Cycle,
    direction: _Direction,
    rho: tuple[float, float],
    height: tuple[float, float],
    /,
) -> int | None:
    """Transversal crossings of one half-line, or None when undecided.

    A monotone range counts one crossing iff its end signs differ strictly;
    a cut whose sign is undecided may hold a root, so the cover is abandoned.
    """
    rows = [
        (
            cycle.start + index / _INITIAL_RANGES,
            cycle.start + (index + 1) / _INITIAL_RANGES,
        )
        for index in range(_INITIAL_RANGES)
    ]
    signs: dict[float, int] = {}
    count = 0
    while rows:
        missing = sorted({tau for row in rows for tau in row if tau not in signs})
        if missing:
            points = rows_owner.points(
                np.asarray([cycle.endpoint(tau) for tau in missing], dtype=np.float64)
            )
            if points is None:
                return None
            value, _, _ = _half_line(points, direction, rho, height)
            value = np.where(points.valid, value, 0)
            signs.update(zip(missing, (int(sign) for sign in value), strict=True))
        enclosure = rows_owner.ranges(cycle, rows)
        if enclosure is None:
            return None
        value, side, rate = _half_line(enclosure, direction, rho, height)
        following = []
        for index, (first, last) in enumerate(rows):
            if not enclosure.valid[index]:
                pass
            elif value[index] != 0 or side[index] < 0:
                continue
            elif side[index] > 0 and rate[index] != 0:
                ends = signs[first], signs[last]
                if 0 in ends:
                    return None
                count += ends[0] != ends[1]
                continue
            middle = 0.5 * (first + last)
            if not first < middle < last:
                return None
            following.extend(((first, middle), (middle, last)))
        rows = following
    return count


def classify_full_revolution(
    source: _RevolutionSource,
    point: np.ndarray,
    /,
    *,
    tolerance: float,
    maximum_rows: int,
) -> RevolutionMembership:
    """Certified parity membership and boundary distance bounds of one point.

    Only ranges whose half-plane distance lower bound does not exceed
    ``tolerance`` are refined, so the bounds separate the point from the
    boundary without solving for its closest point; a boundary point keeps
    a lower bound within ``tolerance``.
    """
    origin = tuple(Fraction(float(value)) for value in source.origin)
    axis = tuple(Fraction(float(value)) for value in source.axis)
    delta = tuple(
        Fraction(float(value)) - shift for value, shift in zip(point, origin, strict=True)
    )
    exact_height = sum((a * b for a, b in zip(delta, axis, strict=True)), Fraction())
    squared = sum((value * value for value in delta), Fraction()) - exact_height**2
    height = (_fraction_down(exact_height), _fraction_up(exact_height))
    rho = _square_root_interval(squared)
    rows_owner = _Rows(source, maximum_rows)
    cycle = _Cycle(source, 0.0)
    rows = [
        (index / _INITIAL_RANGES, (index + 1) / _INITIAL_RANGES)
        for index in range(_INITIAL_RANGES)
    ]
    lower, upper = np.inf, np.inf
    while rows:
        enclosure = rows_owner.ranges(cycle, rows)
        if enclosure is None:
            return RevolutionMembership(False, False, 0.0, upper, rows_owner.used, True)
        row_lower, row_upper = _distance_bounds(enclosure, rho, height)
        upper = min(upper, float(np.min(row_upper)))
        if upper <= tolerance:
            # The boundary is within tolerance; the cover's least row bound
            # stays a sound lower bound and no refinement can separate it.
            lower = min(lower, float(np.min(row_lower)))
            break
        following = []
        for index, (first, last) in enumerate(rows):
            middle = 0.5 * (first + last)
            if row_lower[index] > tolerance or not first < middle < last:
                lower = min(lower, float(row_lower[index]))
            else:
                following.extend(((first, middle), (middle, last)))
        rows = following
    if not lower > tolerance:
        # Within the classifier tolerance parity does not establish membership.
        return RevolutionMembership(False, False, lower, upper, rows_owner.used, False)
    for direction in ("outward", "up", "down"):
        for start in _START_PHASES:
            crossings = _crossings(
                rows_owner, _Cycle(source, start), direction, rho, height
            )
            if crossings is not None:
                return RevolutionMembership(
                    bool(crossings % 2), True, lower, upper, rows_owner.used, False
                )
            if rows_owner.exhausted:
                return RevolutionMembership(
                    False, False, lower, upper, rows_owner.used, True
                )
    return RevolutionMembership(False, False, lower, upper, rows_owner.used, False)


__all__ = ["RevolutionMembership", "classify_full_revolution", "prepare_full_revolution"]
