#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certify membership in a full-turn closed-meridian revolution solid.

A full-turn revolution of a B-spline meridian ``c`` about an exactly unit
axis, or a normal offset of one, is the rotation of one closed curve ``w``:
``c`` itself, or its parallel curve ``c + d n``. Its solid is rotation
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
from typing import assert_never, get_args, Literal, TypeAlias

import numpy as np

from ...discretization._coordinate_budget import CoordinateEnclosureBudget
from ...typing import checked
from .._rational import fraction_down, fraction_up, rational_dot, rational_vector
from ._coordinate_request import coordinate_source_request
from ._exact import square_root_interval
from ._patches import (
    AbstractSurfacePatch,
    BSplineCurve,
    closure_tangent_continuous,
    directed_sum_of_squares,
    evaluate_meridian_jets,
    exact_unit,
    interval_cross,
    interval_dot,
    interval_norm,
    junction_rows,
    meridian_frame,
    MeridianFrame,
    MeridianOverlaps,
    MeridianSpans,
    parallel_meridian_jets,
    prepare_meridian_spans,
    revolution_meridian,
)
from ._request_preparations import TrimRequestPreparations


# Equal meridian ranges of the initial cycle cover; bisection refines them.
_INITIAL_RANGES = 16
# Cycle phases tried as the cover's start. Roots sit at symmetric (dyadic)
# meridian parameters, so the starts and their equal cuts avoid those; a cover
# with a cut on a root is abandoned for the next start.
_START_PHASES = (0.0, 1.0 / 3.0, 1.0 / 7.0, 3.0 / 11.0, 5.0 / 13.0)

_Direction: TypeAlias = Literal["outward", "up", "down"]


@dataclass(frozen=True, slots=True)
class RevolutionMembership:
    """Parity membership with half-plane boundary distance bounds.

    The result is unresolved when the point is not separated from the boundary by
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
class RevolutionSource:
    origin: np.ndarray
    axis: np.ndarray
    spans: MeridianSpans
    frame: MeridianFrame | None
    distance: float
    first: float
    last: float


@checked
def prepare_full_revolution(
    patch: AbstractSurfacePatch,
    lower: float,
    upper: float,
    /,
    *,
    budget: CoordinateEnclosureBudget | None = None,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> RevolutionSource | None:
    """Prove an exactly closed meridian over ``[lower, upper]`` about a unit axis.

    The caller establishes one full turn and closed solid incidence. An
    offset's parallel meridian also needs its closure's end legs exactly
    positively parallel, so both end normals agree.
    """
    if budget is None:
        with coordinate_source_request(
            (
                patch,
                lower,
                upper,
            )
        ) as budget:
            return prepare_full_revolution(
                patch, lower, upper, meridian_jets=meridian_jets, budget=budget
            )
    resolved = revolution_meridian(patch)
    if resolved is None:
        return None
    revolution, distance = resolved
    curve = revolution.curve
    axis = np.asarray(revolution.axis_direction, dtype=np.float64)
    if (
        not isinstance(curve, BSplineCurve)
        or curve.parameter_domain != (lower, upper)
        or not exact_unit(axis)
        or not closure_tangent_continuous(
            revolution, 1, lower, upper, tangent=distance != 0.0
        )
    ):
        return None
    frame = None
    if distance != 0.0:
        frame = meridian_frame(revolution, budget=budget)
        if frame is None:
            return None
        spans = frame.spans
    else:
        spans = prepare_meridian_spans(curve, budget=budget)
        if spans is None:
            return None
    return RevolutionSource(
        origin=np.asarray(revolution.axis_origin, dtype=np.float64),
        axis=axis,
        spans=spans,
        frame=frame,
        distance=distance,
        first=lower,
        last=upper,
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
    source: RevolutionSource,
    first: np.ndarray,
    last: np.ndarray,
    /,
    *,
    budget: CoordinateEnclosureBudget,
    meridian_jets: list[MeridianOverlaps] | None = None,
) -> _Enclosure:
    """Half-plane boxes of ``w`` over closed meridian ranges, row-wise.

    ``radial_rate`` encloses ``((w - o) x a) . (w' x a) = rho rho'``, whose
    sign is that of ``rho'`` wherever ``rho > 0``.
    """
    origin = source.origin
    if source.frame is None:
        jets = evaluate_meridian_jets(
            source.spans, first, last, 1, meridian_jets=meridian_jets, budget=budget
        )
        position = jets.hull(0)
        relative = (
            np.nextafter(position[0] - origin, -np.inf),
            np.nextafter(position[1] - origin, np.inf),
        )
        derivative = jets.hull(1)
        valid = np.ones(first.shape, dtype=np.bool_)
    else:
        jets = evaluate_meridian_jets(
            source.spans, first, last, 2, budget=budget, meridian_jets=meridian_jets
        )
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            parallel, regular = parallel_meridian_jets(
                source.frame, jets.jets, source.distance
            )
        relative, derivative = (
            (
                np.minimum.reduceat(lower, jets.offsets, axis=0),
                np.maximum.reduceat(upper, jets.offsets, axis=0),
            )
            for lower, upper in parallel[:2]
        )
        valid = np.logical_and.reduceat(regular, jets.offsets) & ~junction_rows(
            source.spans, first, last, source.spans.g1, closed=False
        )
    axis = (source.axis, source.axis)
    with np.errstate(invalid="ignore", over="ignore"):
        radial = interval_cross(relative, axis)
        rho = interval_norm(radial)
        height = interval_dot(relative, axis)
        radial_rate = interval_dot(radial, interval_cross(derivative, axis))
        height_rate = interval_dot(derivative, axis)
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

    source: RevolutionSource
    start: float

    def parameter(self, phase: float, /) -> float:
        """Meridian parameter of a phase in ``[0, 1]``."""
        span = self.source.last - self.source.first
        return self.source.last if phase == 1.0 else self.source.first + phase * span

    def pieces(
        self,
        first: float,
        last: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> tuple[tuple[float, float], ...]:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    first,
                    last,
                )
            ) as budget:
                return self.pieces(first, last, preparations=preparations, budget=budget)
        if last <= 1.0:
            return ((self.parameter(first), self.parameter(last)),)
        if first >= 1.0:
            return ((self.parameter(first - 1.0), self.parameter(last - 1.0)),)
        # Proved endpoint equality glues the cycle's two parameter ends.
        return (
            (self.parameter(first), self.source.last),
            (self.source.first, self.parameter(last - 1.0)),
        )

    def endpoint(
        self,
        tau: float,
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> float:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    tau,
                )
            ) as budget:
                return self.endpoint(tau, preparations=preparations, budget=budget)
        return self.parameter(tau - 1.0 if tau > 1.0 else tau)


class _Rows:
    """Batched range and endpoint enclosures with a shared row allowance."""

    def __init__(self, source: RevolutionSource, allowance: int, /) -> None:
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
        self,
        cycle: _Cycle,
        rows: list[tuple[float, float]],
        /,
        *,
        preparations: TrimRequestPreparations | None = None,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> _Enclosure | None:
        if preparations is None:
            preparations = TrimRequestPreparations()
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    cycle,
                    rows,
                )
            ) as budget:
                return self.ranges(cycle, rows, preparations=preparations, budget=budget)
        pieces = [
            cycle.pieces(first, last, preparations=preparations, budget=budget)
            for first, last in rows
        ]
        offsets = np.cumsum([0, *(len(row) for row in pieces[:-1])])
        flat = np.asarray([piece for row in pieces for piece in row], dtype=np.float64)
        if not self.admit(flat.shape[0]):
            return None
        enclosure = _enclose(self.source, flat[:, 0], flat[:, 1], budget=budget)

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

    def points(
        self,
        parameters: np.ndarray,
        /,
        *,
        budget: CoordinateEnclosureBudget | None = None,
    ) -> _Enclosure | None:
        if budget is None:
            with coordinate_source_request(
                (
                    self,
                    parameters,
                )
            ) as budget:
                return self.points(parameters, budget=budget)
        if not self.admit(parameters.shape[0]):
            return None
        return _enclose(self.source, parameters, parameters, budget=budget)


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
            np.nextafter(np.sqrt(directed_sum_of_squares(near, -np.inf)), -np.inf), 0.0
        )
        upper = np.nextafter(np.sqrt(directed_sum_of_squares(far, np.inf)), np.inf)
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
    match direction:
        case "outward":
            return (
                _signs(enclosure.height, height),
                _signs(enclosure.rho, rho),
                _strict(enclosure.height_rate),
            )
        case "up" | "down":
            side = _signs(enclosure.height, height)
            rate = np.where(enclosure.rho[0] > 0.0, _strict(enclosure.radial_rate), 0)
            return (
                _signs(enclosure.rho, rho),
                side if direction == "up" else -side,
                rate,
            )
        case _:
            assert_never(direction)


def _crossings(
    rows_owner: _Rows,
    cycle: _Cycle,
    direction: _Direction,
    rho: tuple[float, float],
    height: tuple[float, float],
    /,
    *,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget,
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
                np.asarray(
                    [
                        cycle.endpoint(tau, preparations=preparations, budget=budget)
                        for tau in missing
                    ],
                    dtype=np.float64,
                ),
                budget=budget,
            )
            if points is None:
                return None
            value, _, _ = _half_line(points, direction, rho, height)
            value = np.where(points.valid, value, 0)
            signs.update(zip(missing, (int(sign) for sign in value), strict=True))
        enclosure = rows_owner.ranges(
            cycle, rows, preparations=preparations, budget=budget
        )
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


@checked
def classify_full_revolution(
    source: RevolutionSource,
    point: np.ndarray,
    /,
    *,
    tolerance: float,
    maximum_rows: int,
    preparations: TrimRequestPreparations | None = None,
    budget: CoordinateEnclosureBudget | None = None,
) -> RevolutionMembership:
    """Certified parity membership and boundary distance bounds of one point.

    Only ranges whose half-plane distance lower bound does not exceed
    ``tolerance`` are refined, so the bounds separate the point from the
    boundary without solving for its closest point; a boundary point keeps
    a lower bound within ``tolerance``.
    """
    if preparations is None:
        preparations = TrimRequestPreparations()
    if budget is None:
        with coordinate_source_request(
            (
                source,
                point,
            )
        ) as budget:
            return classify_full_revolution(
                source,
                point,
                tolerance=tolerance,
                maximum_rows=maximum_rows,
                preparations=preparations,
                budget=budget,
            )
    origin = rational_vector(source.origin)
    axis = rational_vector(source.axis)
    delta = tuple(
        Fraction(float(value)) - shift for value, shift in zip(point, origin, strict=True)
    )
    exact_height = rational_dot(delta, axis)
    squared = rational_dot(delta, delta) - exact_height**2
    height = (fraction_down(exact_height), fraction_up(exact_height))
    rho = square_root_interval(squared)
    rows_owner = _Rows(source, maximum_rows)
    cycle = _Cycle(source, 0.0)
    rows = [
        (index / _INITIAL_RANGES, (index + 1) / _INITIAL_RANGES)
        for index in range(_INITIAL_RANGES)
    ]
    lower, upper = np.inf, np.inf
    while rows:
        enclosure = rows_owner.ranges(
            cycle, rows, preparations=preparations, budget=budget
        )
        if enclosure is None:
            return RevolutionMembership(
                inside=False,
                complete=False,
                distance_lower=0.0,
                distance_upper=upper,
                operations=rows_owner.used,
                exhausted=True,
            )
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
        return RevolutionMembership(
            inside=False,
            complete=False,
            distance_lower=lower,
            distance_upper=upper,
            operations=rows_owner.used,
            exhausted=False,
        )
    for direction in get_args(_Direction):
        for start in _START_PHASES:
            crossings = _crossings(
                rows_owner,
                _Cycle(source, start),
                direction,
                rho,
                height,
                preparations=preparations,
                budget=budget,
            )
            if crossings is not None:
                return RevolutionMembership(
                    inside=bool(crossings % 2),
                    complete=True,
                    distance_lower=lower,
                    distance_upper=upper,
                    operations=rows_owner.used,
                    exhausted=False,
                )
            if rows_owner.exhausted:
                return RevolutionMembership(
                    inside=False,
                    complete=False,
                    distance_lower=lower,
                    distance_upper=upper,
                    operations=rows_owner.used,
                    exhausted=True,
                )
    return RevolutionMembership(
        inside=False,
        complete=False,
        distance_lower=lower,
        distance_upper=upper,
        operations=rows_owner.used,
        exhausted=False,
    )


__all__ = [
    "RevolutionMembership",
    "RevolutionSource",
    "classify_full_revolution",
    "prepare_full_revolution",
]
