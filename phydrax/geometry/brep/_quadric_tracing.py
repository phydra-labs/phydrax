"""Trimmed tracing of quadric pair intersections as graphs over a revolution angle.

A straight revolution (cylinder or cone) ``S1`` is ``P(t) + v W(t)`` in its
angle ``t`` and profile parameter ``v``. A partner quadric ``F2(X) = X.M X +
2 b.X + e`` with exact rational coefficients restricts to ``A(t) v^2 + B(t) v
+ C(t)`` on it, so the intersection is the two graphs ``v = (-B +- sqrt(D)) /
(2 A)`` wherever the discriminant ``D`` is positive. An interval sweep of the
angle circle splits each graph into pieces inside both trims, pieces outside
them, and isolated trim crossings: a crossing piece is accepted only when the
whole branch over it lies in one Krawczyk-certified box on the fixed trim
coordinate, which proves the crossing unique. Crossings of the partner's
periodic seam become nodes on the partner angle with exact decks. A turning
point (``D`` reaching zero) inside the trims is refused for this tracer angle;
the pair is then traced over the other surface, or refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, cast

import jax.numpy as jnp
import numpy as np

from .._rational import rational_dot
from ._intersection import (
    _angular_surface_branch,
    _straight_revolution,
    _StraightRevolution,
    _unresolved_surface,
    CurveIntersectionResult,
    CurveRange,
    ParametricIntersectionPolicy,
    SurfaceIntersectionResult,
)
from ._intersection_curve import (
    _NodeSystem,
    _prepare_pair,
    _PreparedNode,
    IntersectionCurve,
    interval_context,
    SurfacePairSystem,
    SurfaceRegion,
)
from ._intersection_kernel import krawczyk


__all__: list[str] = []

_Vector = tuple[Fraction, Fraction, Fraction]
# An angle piece: (lower, upper, state, evaluation data or crossing coordinate).
_Piece = tuple[float, float, str, Any]


@dataclass(frozen=True, slots=True)
class _Quadric:
    """``F(X) = X.M X + 2 b.X + e`` with exact rational coefficients."""

    matrix: tuple[_Vector, _Vector, _Vector]
    linear: _Vector
    constant: Fraction


def _revolution_quadric(frame: _StraightRevolution, /) -> _Quadric:
    """The implicit quadric of a straight revolution.

    With ``Y = X - O`` and height ``h = Y.k``, the point lies on the surface
    exactly when ``|Y|^2 - h^2 = |r|^2 (p + q h)^2`` for the affine scale
    ``s = s0 + slope v = p + q h``; the mirrored nappe (``s < 0``) is excluded
    by the profile trim.
    """
    k = frame.axis
    if frame.axial_rate == 0:
        # A disk lies in the plane (X - O).k = a0.
        zero = (Fraction(0), Fraction(0), Fraction(0))
        return _Quadric(
            (zero, zero, zero),
            (k[0] / 2, k[1] / 2, k[2] / 2),
            -rational_dot(frame.origin, k) - frame.axial_origin,
        )
    squared = rational_dot(frame.radial, frame.radial)
    q = frame.slope / frame.axial_rate
    p = frame.scale_origin - q * frame.axial_origin
    scale = 1 + squared * q * q
    matrix = tuple(
        tuple(
            (Fraction(1) if row == column else Fraction(0)) - scale * k[row] * k[column]
            for column in range(3)
        )
        for row in range(3)
    )
    shifted = tuple(-squared * p * q * k[index] for index in range(3))
    origin = frame.origin
    applied = tuple(
        sum((matrix[row][column] * origin[column] for column in range(3)), Fraction(0))
        for row in range(3)
    )
    linear = tuple(shifted[index] - applied[index] for index in range(3))
    constant = (
        rational_dot(origin, applied)
        - 2 * rational_dot(shifted, origin)
        - squared * p * p
    )
    rows = cast(tuple[_Vector, _Vector, _Vector], tuple(matrix))
    return _Quadric(rows, (linear[0], linear[1], linear[2]), constant)


@dataclass(frozen=True, slots=True)
class _SphereFrame:
    """A sphere as a revolution of a meridian circle centred on its axis.

    ``radial`` spans the meridian half-plane at angle zero, ``tangent`` is
    ``axis x radial``, and the profile parameter is the meridian circle's own
    angle in its axes ``first`` and ``second``.
    """

    center: tuple[Fraction, ...]
    axis: tuple[Fraction, ...]
    radial: tuple[Fraction, ...]
    tangent: tuple[Fraction, ...]
    first: tuple[Fraction, ...]
    second: tuple[Fraction, ...]
    radius: Fraction


def _sphere_frame(patch: object, /) -> _SphereFrame | None:
    from .._rational import rational_cross, rational_vector
    from ._patches import CircleCurve, RevolutionSurface

    if not isinstance(patch, RevolutionSurface) or not isinstance(
        patch.curve, CircleCurve
    ):
        return None
    circle = patch.curve
    values = (*circle.center_exact, *circle.first_axis_exact, *circle.second_axis_exact)
    if not all(isinstance(value, Fraction) for value in (*values, circle.radius_exact)):
        return None
    exact = cast(tuple[Fraction, ...], values)
    center, first, second = exact[0:3], exact[3:6], exact[6:9]
    origin = rational_vector(np.asarray(patch.axis_origin))
    axis = rational_vector(np.asarray(patch.axis_direction))
    if rational_dot(axis, axis) != 1:
        return None
    offset = tuple(center[index] - origin[index] for index in range(3))
    if (
        any(rational_cross(offset, axis))
        or rational_dot(axis, rational_cross(first, second)) != 0
        or rational_dot(first, first) != 1
        or rational_dot(second, second) != 1
        or rational_dot(first, second) != 0
    ):
        return None
    radial = next(
        (
            candidate
            for candidate in (
                tuple(
                    vector[index] - rational_dot(vector, axis) * axis[index]
                    for index in range(3)
                )
                for vector in (first, second)
            )
            if any(candidate)
        ),
        None,
    )
    if radial is None:
        return None
    return _SphereFrame(
        center,
        axis,
        radial,
        rational_cross(axis, radial),
        first,
        second,
        cast(Fraction, circle.radius_exact),
    )


_Frame = _StraightRevolution | _SphereFrame


def _partner_frame(patch: object, /) -> _Frame | None:
    frame = _straight_revolution(patch, allow_disk=True)
    return frame if frame is not None else _sphere_frame(patch)


def _frame_quadric(frame: _Frame, /) -> _Quadric:
    if isinstance(frame, _StraightRevolution):
        return _revolution_quadric(frame)
    identity = tuple(
        tuple(Fraction(1) if row == column else Fraction(0) for column in range(3))
        for row in range(3)
    )
    center = frame.center
    return _Quadric(
        cast(tuple[_Vector, _Vector, _Vector], identity),
        (-center[0], -center[1], -center[2]),
        rational_dot(center, center) - frame.radius**2,
    )


def _frame_coordinates(
    frame: _Frame, point: list[Any], ctx: Any, /
) -> tuple[Any, Any, Any] | None:
    """Enclose ``(along, across, profile)`` of a point on the frame's surface.

    ``along`` and ``across`` are the radial components whose angle is the
    revolution angle; ``profile`` is the profile parameter (None at a pole).
    """

    def scalar(value: Fraction, /) -> Any:
        return ctx.mpf(value.numerator) / ctx.mpf(value.denominator)

    def dot(left: list[Any], right: list[Any], /) -> Any:
        return sum((left[i] * right[i] for i in range(3)), ctx.mpf(0))

    base = frame.origin if isinstance(frame, _StraightRevolution) else frame.center
    relative = [point[i] - scalar(base[i]) for i in range(3)]
    axis = [scalar(value) for value in frame.axis]
    height = dot(relative, axis)
    planar = [relative[i] - height * axis[i] for i in range(3)]
    along = dot(planar, [scalar(value) for value in frame.radial])
    across = dot(planar, [scalar(value) for value in frame.tangent])
    if isinstance(frame, _SphereFrame):
        squared = _nonnegative(ctx, sum((value**2 for value in planar), ctx.mpf(0)))
        radial_norm = ctx.sqrt(scalar(rational_dot(frame.radial, frame.radial)))
        unit = [scalar(value) / radial_norm for value in frame.radial]
        meridian = [ctx.sqrt(squared) * unit[i] + height * axis[i] for i in range(3)]
        cosine = dot(meridian, [scalar(value) for value in frame.first])
        sine = dot(meridian, [scalar(value) for value in frame.second])
        reference = math.atan2(float(sine.mid), float(cosine.mid))
        c, s = ctx.cos(ctx.mpf(reference)), ctx.sin(ctx.mpf(reference))
        rotated = c * cosine + s * sine
        if not rotated.a > 0:
            return None
        profile = reference + ctx.atan2(c * sine - s * cosine, rotated)
        return along, across, profile
    if frame.axial_rate == 0:
        squared = scalar(rational_dot(frame.radial, frame.radial))
        scale = ctx.sqrt(
            _nonnegative(ctx, sum((value**2 for value in planar), ctx.mpf(0)) / squared)
        )
        profile = (scale - scalar(frame.scale_origin)) / scalar(frame.slope)
    else:
        profile = (height - scalar(frame.axial_origin)) / scalar(frame.axial_rate)
    return along, across, profile


def _full_period(region: SurfaceRegion, /) -> bool:
    period = region.patch.periods[0]
    box = np.asarray(region.parameter_box, dtype=np.float64)
    return (
        period is not None
        and region.patch.native_period_symbols()[0] == "two_pi"
        and Fraction.from_float(float(box[1, 0])) - Fraction.from_float(float(box[0, 0]))
        == Fraction.from_float(float(period))
    )


def trace_revolution_quadric_pair(
    first: SurfaceRegion,
    second: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> SurfaceIntersectionResult | None:
    """Trace two straight revolutions over either one's angle, or refuse."""
    frames = (_partner_frame(first.patch), _partner_frame(second.patch))
    if frames[0] is None or frames[1] is None:
        return None
    if not _full_period(first) or not _full_period(second):
        return _unresolved_surface(first, second, "continuation")
    # Both tracer angles are swept; the branches are built over the angle
    # whose inside graph is flattest, which keeps its certified charts wide.
    swept: list[tuple[float, _Tracer, dict[int, list[_Piece]]]] = []
    for tracer_first in (True, False):
        # Only a straight revolution is linear in its profile parameter.
        if not isinstance(frames[0 if tracer_first else 1], _StraightRevolution):
            continue
        tracer = _Tracer(first, second, frames, tracer_first, policy)
        pieces = tracer.sweep_signs()
        if pieces is not None:
            swept.append((tracer.steepness(pieces), tracer, pieces))
    for _, tracer, pieces in sorted(swept, key=lambda item: item[0]):
        traced = tracer.build(pieces)
        if traced is not None:
            return traced
    return _unresolved_surface(first, second, "continuation")


class _Tracer:
    def __init__(
        self,
        first: SurfaceRegion,
        second: SurfaceRegion,
        frames: tuple[_Frame | None, _Frame | None],
        tracer_first: bool,
        policy: ParametricIntersectionPolicy,
    ) -> None:
        self.first, self.second, self.policy = first, second, policy
        tracer, partner = (
            (frames[0], frames[1]) if tracer_first else (frames[1], frames[0])
        )
        assert isinstance(tracer, _StraightRevolution) and partner is not None
        self.tracer, self.partner = tracer, partner
        self.tracer_region = first if tracer_first else second
        self.partner_region = second if tracer_first else first
        self.tracer_offset = 0 if tracer_first else 2
        self.partner_offset = 2 - self.tracer_offset
        self.quadric = _frame_quadric(partner)
        self.planar_partner = (
            isinstance(partner, _StraightRevolution) and partner.axial_rate == 0
        )
        self.context = interval_context(60)
        tracer_box = np.asarray(self.tracer_region.parameter_box, dtype=np.float64)
        partner_box = np.asarray(self.partner_region.parameter_box, dtype=np.float64)
        self.angle_lower, self.angle_upper = (
            float(tracer_box[0, 0]),
            float(tracer_box[1, 0]),
        )
        self.profile_bounds = (float(tracer_box[0, 1]), float(tracer_box[1, 1]))
        self.partner_angle_lower = float(partner_box[0, 0])
        self.partner_bounds = (float(partner_box[0, 1]), float(partner_box[1, 1]))
        self.resolution = (
            64.0
            * np.finfo(np.float64).eps
            * max(1.0, abs(self.angle_lower), abs(self.angle_upper))
        )
        pair = SurfacePairSystem(first.patch, second.patch)
        self.pair = pair
        self.prepared = _prepare_pair(pair, source_regions=(first, second))

    # Exact coefficients as 60-digit intervals.
    def _scalar(self, value: Fraction, /) -> Any:
        return self.context.mpf(value.numerator) / self.context.mpf(value.denominator)

    def _evaluate(self, lower: float, upper: float, sign: int, /) -> dict[str, Any]:
        """Enclose the graph of one sign over an angle interval."""
        ctx = self.context
        angle = ctx.mpf([lower, upper]) if lower < upper else ctx.mpf(lower)
        cosine, sine = ctx.cos(angle), ctx.sin(angle)
        frame = self.tracer
        radial = [
            self._scalar(frame.radial[i]) * cosine + self._scalar(frame.tangent[i]) * sine
            for i in range(3)
        ]
        point = [
            self._scalar(frame.origin[i] + frame.axial_origin * frame.axis[i])
            + self._scalar(frame.scale_origin) * radial[i]
            for i in range(3)
        ]
        rate = [
            self._scalar(frame.axial_rate * frame.axis[i])
            + self._scalar(frame.slope) * radial[i]
            for i in range(3)
        ]
        matrix = [[self._scalar(v) for v in row] for row in self.quadric.matrix]
        linear = [self._scalar(v) for v in self.quadric.linear]

        def form(left: list[Any], right: list[Any], /) -> Any:
            return sum(
                (left[r] * matrix[r][c] * right[c] for r in range(3) for c in range(3)),
                ctx.mpf(0),
            )

        def dot(left: list[Any], right: list[Any], /) -> Any:
            return sum((left[i] * right[i] for i in range(3)), ctx.mpf(0))

        a = form(rate, rate)
        b = 2 * (form(point, rate) + dot(linear, rate))
        c = (
            form(point, point)
            + 2 * dot(linear, point)
            + self._scalar(self.quadric.constant)
        )
        if self.planar_partner:
            # A plane partner leaves the single root of B v + C = 0.
            result: dict[str, Any] = {"discriminant": ctx.mpf(1 if sign > 0 else -1)}
            if sign < 0:
                return result
            if not b.a <= 0 <= b.b:
                profile = -c / b
            else:
                if not c.a <= 0 <= c.b:
                    smallest = min(abs(c.a), abs(c.b))
                    largest = max(abs(b.a), abs(b.b))
                    bound = max(abs(value) for value in self.profile_bounds)
                    if largest == 0 or smallest / largest > bound:
                        result["escapes"] = True
                return result
            return self._partner_values(result, point, rate, profile)
        discriminant = b * b - 4 * a * c
        result = {"discriminant": discriminant, "leading": a}
        if discriminant.b < 0:
            return result
        root = ctx.sqrt(ctx.mpf([max(discriminant.a, 0), discriminant.b]))
        numerator = -b + sign * root
        conjugate = -b - sign * root
        if not a.a <= 0 <= a.b:
            profile = numerator / (2 * a)
        elif not conjugate.a <= 0 <= conjugate.b:
            # The same root as 2 C / (-B -+ sqrt D), finite where A vanishes.
            profile = 2 * c / conjugate
        else:
            # Both roots satisfy |v| >= |C| / (|B| + sqrt D); where that bound
            # exceeds every profile bound the root escapes the trim.
            bound = max(abs(value) for value in self.profile_bounds)
            if not c.a <= 0 <= c.b:
                smallest = min(abs(c.a), abs(c.b))
                largest = max(abs(b.a), abs(b.b)) + root.b
                if largest == 0 or smallest / largest > bound:
                    result["escapes"] = True
            return result
        return self._partner_values(result, point, rate, profile)

    def _partner_values(
        self, result: dict[str, Any], point: list[Any], rate: list[Any], profile: Any, /
    ) -> dict[str, Any]:
        ctx = self.context

        def dot(left: list[Any], right: list[Any], /) -> Any:
            return sum((left[i] * right[i] for i in range(3)), ctx.mpf(0))

        spatial = [point[i] + profile * rate[i] for i in range(3)]
        coordinates = _frame_coordinates(self.partner, spatial, ctx)
        if coordinates is None:
            return result
        along, across, partner_profile = coordinates
        result.update(
            profile=profile,
            partner_profile=partner_profile,
            along=along,
            across=across,
        )
        return result

    def _partner_angle(self, along: Any, across: Any, reference: float, /) -> Any:
        """The partner angle near ``reference``, continuous across its branch cut."""
        ctx = self.context
        cosine, sine = ctx.cos(ctx.mpf(reference)), ctx.sin(ctx.mpf(reference))
        rotated_along = cosine * along + sine * across
        rotated_across = cosine * across - sine * along
        if not rotated_along.a > 0:
            return None
        return reference + ctx.atan2(rotated_across, rotated_along)

    def _classify(self, lower: float, upper: float, sign: int, /) -> tuple[str, Any]:
        """``in``, ``out``, an event ``(axis, value)``, or ``undecided``."""
        values = self._evaluate(lower, upper, sign)
        discriminant = values["discriminant"]
        if discriminant.b < 0 or values.get("escapes", False):
            return "out", values
        if "profile" not in values:
            return "undecided", values
        constraints = (
            (values["profile"], self.profile_bounds, self.tracer_offset + 1),
            (values["partner_profile"], self.partner_bounds, self.partner_offset + 1),
        )
        straddles = []
        for value, (low, high), axis in constraints:
            if value.b < low or value.a > high:
                return "out", values
            for bound in (low, high):
                if value.a <= bound <= value.b:
                    straddles.append((axis, bound))
        if not discriminant.a > 0:
            return "undecided", values
        if not straddles:
            return "in", values
        if len(straddles) == 1:
            return "event", (values, straddles[0])
        return "undecided", values

    def _sweep(self, sign: int, /) -> list[tuple[float, float, str, Any]] | None:
        """Cover the angle window by decided pieces of one graph."""
        count = 64
        width = (self.angle_upper - self.angle_lower) / count
        pending = [
            (
                self.angle_lower + index * width,
                self.angle_upper
                if index == count - 1
                else self.angle_lower + (index + 1) * width,
            )
            for index in reversed(range(count))
        ]
        pieces: list[tuple[float, float, str, Any]] = []
        while pending:
            lower, upper = pending.pop()
            state, data = self._classify(lower, upper, sign)
            if state in ("in", "out"):
                pieces.append((lower, upper, state, data))
                continue
            # A crossing is bisected to a few units in the last place, so the
            # chart ending next to it meets the crossing's own root box.
            small = upper - lower <= 4 * math.ulp(max(abs(lower), abs(upper), 1.0))
            if state == "event" and small:
                values, (axis, value) = data
                if self._certify_event(lower, upper, values, axis, value):
                    pieces.append((lower, upper, "event", (axis, value)))
                    continue
            middle = lower + 0.5 * (upper - lower)
            if not lower < middle < upper or (
                state != "event" and upper - lower <= self.resolution
            ):
                return None
            pending.append((middle, upper))
            pending.append((lower, middle))
        # A crossing exactly at a float shared by an inside and an outside
        # piece is a zero-width event there.
        resolved: list[tuple[float, float, str, Any]] = []
        for index, piece in enumerate(pieces):
            previous = pieces[index - 1] if index else None
            if previous is not None and {previous[2], piece[2]} == {"in", "out"}:
                state, data = self._classify(piece[0], piece[0], sign)
                if state != "event":
                    return None
                values, (axis, value) = data
                if not self._certify_event(piece[0], piece[0], values, axis, value):
                    return None
                resolved.append((piece[0], piece[0], "event", (axis, value)))
            resolved.append(piece)
        # Adjacent crossing pieces of one constraint are one crossing: their
        # hull is certified again as a whole.
        merged: list[tuple[float, float, str, Any]] = []
        for piece in resolved:
            last = merged[-1] if merged else None
            if (
                last is not None
                and last[2] == piece[2] == "event"
                and last[3] == piece[3]
            ):
                lower, upper = last[0], piece[1]
                values = self._evaluate(lower, upper, sign)
                if "profile" not in values or not self._certify_event(
                    lower, upper, values, *piece[3]
                ):
                    return None
                merged[-1] = (lower, upper, "event", piece[3])
                continue
            merged.append(piece)
        return merged

    def _hull(
        self, lower: float, upper: float, values: dict[str, Any], /
    ) -> tuple[np.ndarray, np.ndarray] | None:
        reference = self._point(0.5 * (lower + upper))
        if reference is None:
            return None
        angle = self._partner_angle(
            values["along"], values["across"], float(reference[self.partner_offset])
        )
        if angle is None:
            return None
        profile, partner_profile = values["profile"], values["partner_profile"]
        tracer = ((lower, upper), (float(profile.a), float(profile.b)))
        partner = (
            (float(angle.a), float(angle.b)),
            (float(partner_profile.a), float(partner_profile.b)),
        )
        blocks = (tracer, partner) if self.tracer_offset == 0 else (partner, tracer)
        low = np.asarray([item[0] for block in blocks for item in block])
        high = np.asarray([item[1] for block in blocks for item in block])
        pad = 1e-10 * np.maximum(1.0, np.maximum(np.abs(low), np.abs(high)))
        low, high = np.nextafter(low - pad, -np.inf), np.nextafter(high + pad, np.inf)
        # The tracer angle is the chart parameter itself: its range stays exact.
        low[self.tracer_offset], high[self.tracer_offset] = lower, upper
        return low, high

    def _certify_event(
        self,
        lower: float,
        upper: float,
        values: dict[str, Any],
        axis: int,
        value: float,
        /,
    ) -> bool:
        hull = self._hull(lower, upper, values)
        if hull is None:
            return False
        low, high = hull
        # The node system fixes the coordinate; its box entry only brackets it.
        # The tracer angle is free here: a wider bracket still holds the whole
        # branch over the piece, so uniqueness in it is uniqueness on the piece.
        spread = 1e-12 * max(1.0, abs(value))
        low[axis], high[axis] = value - spread, value + spread
        angle_pad = 1e-10 * max(1.0, abs(lower), abs(upper))
        low[self.tracer_offset] = np.nextafter(lower - angle_pad, -np.inf)
        high[self.tracer_offset] = np.nextafter(upper + angle_pad, np.inf)
        prepared = _PreparedNode(
            _NodeSystem(self.pair, axis, value),
            self.prepared,
            source_regions=(self.first, self.second),
        )
        result = krawczyk(prepared, low[None], high[None])
        return bool(result.certified[0])

    # Float evaluation for chart endpoints.
    def _point(self, angle: float, side: int = 1, /) -> np.ndarray | None:
        values = self._evaluate(angle, angle, self.sign)
        if "profile" not in values:
            return None
        raw = math.atan2(float(values["across"].mid), float(values["along"].mid))
        tracer = (angle, float(values["profile"].mid))
        partner = (
            self._sheet_angle(raw, angle, side),
            float(values["partner_profile"].mid),
        )
        ordered = (*tracer, *partner) if self.tracer_offset == 0 else (*partner, *tracer)
        return np.asarray(ordered, dtype=np.float64)

    def _sheet_angle(self, raw: float, angle: float, side: int, /) -> float:
        """The partner angle in the window sheet of ``angle``.

        Next to a partner seam crossing, the representation continues the
        sheet the query belongs to past the true crossing by rounding.
        """
        bottom = self.partner_angle_lower
        value = raw + math.floor((bottom - raw) / math.tau + 1) * math.tau
        if value >= bottom + math.tau:
            value -= math.tau
        middle = bottom + math.pi
        near = 1e-9 * max(1.0, abs(angle))
        for seam, upward in self.seams:
            if abs(angle - seam) > near:
                continue
            before = angle < seam or (angle == seam and side < 0)
            high_side = upward == before
            if high_side and value < middle:
                value += math.tau
            elif not high_side and value > middle:
                value -= math.tau
        return value

    sign: int = 1
    seams: tuple[tuple[float, bool], ...] = ()

    def _chart_point(self, angle: float, /) -> np.ndarray:
        point = self._point(angle)
        if point is None:
            raise ValueError("The traced graph has no point at a chart angle.")
        return point

    def sweep_signs(self) -> dict[int, list[_Piece]] | None:
        pieces: dict[int, list[_Piece]] = {}
        for sign in (-1, 1):
            self.sign = sign
            swept = self._sweep(sign)
            if swept is None:
                return None
            pieces[sign] = swept
        return pieces

    def steepness(self, pieces: dict[int, list[_Piece]], /) -> float:
        """The largest rate of the other three coordinates per unit tracer angle.

        Charts over a steep graph must be short to certify, so the tracer with
        the flatter inside graph is built first.
        """
        rate = 0.0
        for sign, swept in pieces.items():
            self.sign = sign
            for lower, upper, state, values in swept:
                if state != "in" or not upper > lower:
                    continue
                reference = self._point(0.5 * (lower + upper))
                if reference is None:
                    return math.inf
                angle = self._partner_angle(
                    values["along"],
                    values["across"],
                    float(reference[self.partner_offset]),
                )
                if angle is None:
                    return math.inf
                spread = max(
                    float((angle.b - angle.a).b),
                    float((values["profile"].b - values["profile"].a).b),
                    float((values["partner_profile"].b - values["partner_profile"].a).b),
                )
                rate = max(rate, spread / (upper - lower))
        return rate

    def build(
        self, pieces: dict[int, list[_Piece]], /
    ) -> SurfaceIntersectionResult | None:
        curves = []
        for sign, swept in pieces.items():
            self.sign = sign
            built = self._branches(swept)
            if built is None:
                return None
            curves.extend(built)
        return SurfaceIntersectionResult(curves=tuple(curves))

    def _branches(
        self, pieces: list[tuple[float, float, str, Any]], /
    ) -> list[Any] | None:
        states = [piece[2] for piece in pieces]
        if all(state == "out" for state in states):
            return []
        if any(
            state == "event" and states[index - 1] == states[(index + 1) % len(states)]
            for index, state in enumerate(states)
        ):
            # A crossing must separate an inside run from an outside one.
            return None
        if all(state == "in" for state in states):
            return self._closed_branch(pieces)
        count = len(pieces)
        runs: list[tuple[int, int]] = []
        for index in range(count):
            if states[index] == "in" and states[index - 1] != "in":
                end = index
                while states[(end + 1) % count] == "in":
                    end = (end + 1) % count
                runs.append((index, end))
        curves = []
        for run_start, run_end in runs:
            before = pieces[run_start - 1]
            after = pieces[(run_end + 1) % count]
            if before[2] != "event" or after[2] != "event":
                return None
            curve = self._open_branch(pieces, run_start, run_end, before, after)
            if curve is None:
                return None
            curves.append(curve)
        return curves

    def _seam_angles(
        self, pieces: list[tuple[float, float, str, Any]], indices: list[int], /
    ) -> tuple[tuple[float, bool], ...] | None:
        """Angles where the inside graph crosses the partner's periodic seam.

        Each crossing is bisected to a few units in the last place and keeps
        whether the partner angle increases through it.
        """
        self.seams = ()
        seams: list[tuple[float, bool]] = []
        bottom = self.partner_angle_lower
        for index in indices:
            lower, upper, _, _ = pieces[index]
            pending = [(lower, upper)]
            while pending:
                low, high = pending.pop()
                data = self._evaluate(low, high, self.sign)
                reference = self._point(0.5 * (low + high))
                angle = (
                    None
                    if reference is None or "along" not in data
                    else self._partner_angle(
                        data["along"],
                        data["across"],
                        float(reference[self.partner_offset]),
                    )
                )
                middle = low + 0.5 * (high - low)
                if angle is None:
                    if not low < middle < high:
                        return None
                    pending.extend(((middle, high), (low, middle)))
                    continue
                turns = {
                    math.floor((float(value) - bottom) / math.tau)
                    for value in (angle.a, angle.b)
                }
                if len(turns) == 1:
                    continue
                if high - low <= 4 * math.ulp(max(abs(low), abs(high), 1.0)) or not (
                    low < middle < high
                ):
                    ends = []
                    for value in (low, high):
                        end = self._evaluate(value, value, self.sign)
                        if "along" not in end:
                            return None
                        near = self._partner_angle(end["along"], end["across"], bottom)
                        if near is None:
                            return None
                        ends.append(float(near.mid))
                    seams.append(
                        (middle if low < middle < high else low, ends[1] > ends[0])
                    )
                    continue
                pending.extend(((middle, high), (low, middle)))
        # One crossing on the boundary of two pieces is reported by both.
        distinct: list[tuple[float, bool]] = []
        for seam in sorted(seams):
            if distinct and seam[0] - distinct[-1][0] <= 64 * math.ulp(
                max(abs(seam[0]), 1.0)
            ):
                continue
            distinct.append(seam)
        self.seams = tuple(distinct)
        return self.seams

    def _sided(self, angle: float, side: int, /) -> np.ndarray:
        point = self._point(angle, side)
        if point is None:
            raise ValueError("The traced graph has no point at a chart angle.")
        return point

    def _box_for(self, lower: float, upper: float, /) -> tuple[np.ndarray, np.ndarray]:
        # The union of enclosures over subintervals is much tighter than one
        # enclosure of the whole interval, whose dependency spread would force
        # many small charts.
        steps = 8
        parts = []
        for index in range(steps):
            low = lower + (upper - lower) * index / steps
            high = (
                upper
                if index == steps - 1
                else lower + (upper - lower) * (index + 1) / steps
            )
            if not low < high:
                continue
            hull = self._hull(low, high, self._evaluate(low, high, self.sign))
            if hull is None:
                raise ValueError("The traced graph has no partner-angle enclosure.")
            parts.append(hull)
        if not parts:
            hull = self._hull(lower, upper, self._evaluate(lower, upper, self.sign))
            if hull is None:
                raise ValueError("The traced graph has no partner-angle enclosure.")
            return hull
        box_lower = np.min(np.stack([part[0] for part in parts]), axis=0)
        box_upper = np.max(np.stack([part[1] for part in parts]), axis=0)
        box_lower[self.tracer_offset], box_upper[self.tracer_offset] = lower, upper
        return box_lower, box_upper

    def _open_branch(
        self,
        pieces: list[_Piece],
        run_start: int,
        run_end: int,
        before: _Piece,
        after: _Piece,
    ) -> IntersectionCurve | None:
        count = len(pieces)
        indices = []
        index = run_start
        while True:
            indices.append(index)
            if index == run_end:
                break
            index = (index + 1) % count
        theta_lower = before[1]
        theta_upper = after[0]
        across_seam = run_end < run_start or theta_upper < theta_lower
        seams = self._seam_angles(pieces, indices)
        if seams is None:
            return None
        return self._build(
            self.first,
            self.second,
            self.tracer_offset,
            theta_lower,
            theta_upper,
            self.policy,
            self._chart_point,
            self._box_for,
            closed=False,
            start_kind="boundary",
            end_kind="boundary",
            end_nodes=(before[3], after[3]),
            interior_nodes=tuple(
                (angle, self.partner_offset, self.partner_angle_lower)
                for angle, _ in seams
            ),
            seam=(self.angle_lower, self.angle_upper) if across_seam else None,
            sided_point_at=self._sided,
            end_node_radius=2.0**-36,
        )

    @staticmethod
    def _build(*args: Any, **kwargs: Any) -> Any:
        try:
            return _angular_surface_branch(*args, **kwargs)
        except ValueError:
            return None

    def _closed_branch(self, pieces: list[_Piece]) -> list[IntersectionCurve] | None:
        seams = self._seam_angles(pieces, list(range(len(pieces))))
        if seams is None:
            return None
        curve = self._build(
            self.first,
            self.second,
            self.tracer_offset,
            self.angle_lower,
            self.angle_upper,
            self.policy,
            self._chart_point,
            self._box_for,
            closed=True,
            interior_nodes=tuple(
                (angle, self.partner_offset, self.partner_angle_lower)
                for angle, _ in seams
            ),
            sided_point_at=self._sided,
            end_node_radius=2.0**-36,
        )
        return None if curve is None else [curve]


def circle_revolution_roots(
    source: CurveRange,
    surface: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> CurveIntersectionResult | None:
    """Isolate an exact circle against a straight revolution (or disk) quadric.

    The quadric restricted to the circle is a trigonometric polynomial ``f(t)``
    of degree two. Its roots are isolated in ``t`` alone: a piece where ``f``
    excludes zero has none, and a piece whose derivative excludes zero with a
    sign change at its ends has exactly one. Each root's tight parameter box
    on both carriers is then certified by the curve-surface Krawczyk root.
    """
    from ._intersection import (
        CurveSurfaceIntersectionRoot,
        ParametricIntersectionPoint,
        UnresolvedParameterRegion,
    )
    from ._patches import CircleCurve

    circle = source.curve
    frame = _partner_frame(surface.patch)
    if frame is None or not isinstance(circle, CircleCurve):
        return None
    coefficients = (
        *circle.center_exact,
        *circle.first_axis_exact,
        *circle.second_axis_exact,
        circle.radius_exact,
    )
    if not all(isinstance(value, Fraction) for value in coefficients):
        return None
    if not _full_period(surface):
        return None
    quadric = _frame_quadric(frame)
    ctx = interval_context(60)

    def scalar(value: Fraction, /) -> Any:
        return ctx.mpf(value.numerator) / ctx.mpf(value.denominator)

    exact = cast(tuple[Fraction, ...], coefficients)
    center = [scalar(value) for value in exact[0:3]]
    first_axis = [scalar(value) for value in exact[3:6]]
    second_axis = [scalar(value) for value in exact[6:9]]
    radius = scalar(exact[9])
    matrix = [[scalar(value) for value in row] for row in quadric.matrix]
    linear = [scalar(value) for value in quadric.linear]
    constant = scalar(quadric.constant)

    def evaluate(lower: float, upper: float, /) -> tuple[Any, Any, list[Any]]:
        angle = ctx.mpf([lower, upper]) if lower < upper else ctx.mpf(lower)
        cosine, sine = ctx.cos(angle), ctx.sin(angle)
        point = [
            center[i] + radius * (cosine * first_axis[i] + sine * second_axis[i])
            for i in range(3)
        ]
        rate = [
            radius * (cosine * second_axis[i] - sine * first_axis[i]) for i in range(3)
        ]
        applied = [
            sum((matrix[r][c] * point[c] for c in range(3)), ctx.mpf(0)) + linear[r]
            for r in range(3)
        ]
        value = (
            sum((point[r] * (applied[r] + linear[r]) for r in range(3)), ctx.mpf(0))
            + constant
        )
        slope = 2 * sum((applied[r] * rate[r] for r in range(3)), ctx.mpf(0))
        return value, slope, point

    domain_lower = np.asarray((source.first, *np.asarray(surface.parameter_box[0])))
    domain_upper = np.asarray((source.last, *np.asarray(surface.parameter_box[1])))

    def refused(reason: Any, /) -> CurveIntersectionResult:
        return CurveIntersectionResult(
            unresolved=(UnresolvedParameterRegion(domain_lower, domain_upper, reason),)
        )

    region_box = np.asarray(surface.parameter_box, dtype=np.float64)
    resolution = (
        64.0 * np.finfo(np.float64).eps * max(1.0, abs(source.first), abs(source.last))
    )
    count = 32
    width = (source.last - source.first) / count
    pending = [
        (
            source.first + index * width,
            source.last if index == count - 1 else source.first + (index + 1) * width,
        )
        for index in reversed(range(count))
    ]
    roots: list[tuple[float, float]] = []
    while pending:
        lower, upper = pending.pop()
        value, slope, piece_points = evaluate(lower, upper)
        if value.a > 0 or value.b < 0:
            continue
        # Points off the region's profile range are no contact with this face.
        piece_chart = _revolution_chart(frame, piece_points, float(region_box[0, 0]), ctx)
        if piece_chart is not None and (
            float(piece_chart[1].b) < region_box[0, 1]
            or float(piece_chart[1].a) > region_box[1, 1]
        ):
            continue
        middle = lower + 0.5 * (upper - lower)
        if slope.a > 0 or slope.b < 0:
            low_value = evaluate(lower, lower)[0]
            high_value = evaluate(upper, upper)[0]
            if (low_value.b < 0 < high_value.a) or (high_value.b < 0 < low_value.a):
                # One simple root: bisected to a few units in the last place.
                if upper - lower <= 4 * math.ulp(
                    max(abs(lower), abs(upper), 1.0)
                ) or not (lower < middle < upper):
                    roots.append((lower, upper))
                    continue
                pending.append((middle, upper))
                pending.append((lower, middle))
                continue
            if (low_value.a > 0 and high_value.a > 0) or (
                low_value.b < 0 and high_value.b < 0
            ):
                # Monotone with one sign at both ends: no root.
                continue
        if upper - lower <= resolution or not lower < middle < upper:
            return refused("singular")
        pending.append((middle, upper))
        pending.append((lower, middle))
    points: list[ParametricIntersectionPoint] = []
    box = np.asarray(surface.parameter_box, dtype=np.float64)
    for lower, upper in sorted(roots):
        _, _, point = evaluate(lower, upper)
        chart = _revolution_chart(frame, point, float(box[0, 0]), ctx)
        if chart is None:
            return refused("continuation")
        angle, profile = chart
        if float(profile.b) < box[0, 1] or float(profile.a) > box[1, 1]:
            continue
        if not (box[0, 1] < float(profile.a) and float(profile.b) < box[1, 1]):
            return refused("continuation")
        if lower <= source.first or upper >= source.last:
            return refused("continuation")
        # The tightest certified box keeps the root's chart image next to the
        # branch node it ends; padding grows only until Krawczyk certifies.
        pad = (
            64
            * np.finfo(np.float64).eps
            * max(1.0, abs(float(angle.b)), abs(float(profile.b)))
        )
        root = None
        while root is None:
            low = np.asarray(
                (lower - pad, float(angle.a) - pad, float(profile.a) - pad),
                dtype=np.float64,
            )
            high = np.asarray(
                (upper + pad, float(angle.b) + pad, float(profile.b) + pad),
                dtype=np.float64,
            )
            try:
                root = CurveSurfaceIntersectionRoot(
                    circle, surface, parameter_lower=low, parameter_upper=high
                )
            except ValueError:
                if pad > 1e-9 * max(1.0, abs(lower)):
                    return refused("resolution")
                pad *= 4.0
        parameters = np.asarray(root.parameters, dtype=np.float64)
        source_point = np.asarray(circle.evaluate(jnp.asarray(parameters[0])))
        surface_point = np.asarray(surface.patch.evaluate(jnp.asarray(parameters[1:])))
        points.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="krawczyk",
                parameters=parameters,
                parameter_lower=np.asarray(root.parameter_lower, dtype=np.float64),
                parameter_upper=np.asarray(root.parameter_upper, dtype=np.float64),
                point=source_point,
                gap_bound=float(np.linalg.norm(source_point - surface_point)),
                condition_estimate=1.0,
                root_id=root.root_id,
            )
        )
    del policy
    return CurveIntersectionResult(points=points)


def _nonnegative(ctx: Any, value: Any, /) -> Any:
    """Clip an enclosure of a square to its nonnegative part."""
    return ctx.mpf([max(value.a, 0), max(value.b, 0)])


def _revolution_chart(
    frame: _Frame, point: list[Any], window: float, ctx: Any, /
) -> tuple[Any, Any] | None:
    """Enclose ``(angle, profile)`` of a point on a revolution frame."""
    coordinates = _frame_coordinates(frame, point, ctx)
    if coordinates is None:
        return None
    along, across, profile = coordinates
    reference = math.atan2(float(across.mid), float(along.mid))
    reference += math.ceil((window - reference) / math.tau) * math.tau
    cosine, sine = ctx.cos(ctx.mpf(reference)), ctx.sin(ctx.mpf(reference))
    rotated_along = cosine * along + sine * across
    rotated_across = cosine * across - sine * along
    if not rotated_along.a > 0:
        return None
    return reference + ctx.atan2(rotated_across, rotated_along), profile


def line_sphere_roots(
    source: CurveRange,
    surface: SurfaceRegion,
    policy: ParametricIntersectionPolicy,
    /,
) -> CurveIntersectionResult | None:
    """Isolate a straight edge against a sphere from its exact quadratic.

    ``|L0 + t D - c|^2 = R^2`` has rational coefficients; its radical roots are
    enclosed to a few units in the last place and certified as curve-surface
    roots with tight boxes.
    """
    from ...discretization._exact_algebraic_geometry import (
        coefficient_interval,
        RadicalElement,
    )
    from ._intersection import (
        CurveSurfaceIntersectionRoot,
        ParametricIntersectionPoint,
        UnresolvedParameterRegion,
    )
    from ._patches import ExactLineCurve, LineCurve

    frame = _sphere_frame(surface.patch)
    curve = source.curve
    if frame is None:
        return None
    if isinstance(curve, LineCurve):
        origin, direction = curve.origin_exact, curve.direction_exact
    elif isinstance(curve, ExactLineCurve):
        origin = curve.first_exact
        direction = tuple(
            last - first
            for first, last in zip(curve.first_exact, curve.last_exact, strict=True)
        )
    else:
        return None
    if not all(isinstance(value, Fraction) for value in (*origin, *direction)):
        return None
    base = cast(tuple[Fraction, ...], tuple(origin))
    rate = cast(tuple[Fraction, ...], tuple(direction))
    offset = tuple(base[index] - frame.center[index] for index in range(3))
    quadratic = rational_dot(rate, rate)
    linear = 2 * rational_dot(offset, rate)
    constant = rational_dot(offset, offset) - frame.radius**2
    domain_lower = np.asarray((source.first, *np.asarray(surface.parameter_box[0])))
    domain_upper = np.asarray((source.last, *np.asarray(surface.parameter_box[1])))

    def refused(reason: Any, /) -> CurveIntersectionResult:
        return CurveIntersectionResult(
            unresolved=(UnresolvedParameterRegion(domain_lower, domain_upper, reason),)
        )

    discriminant = linear * linear - 4 * quadratic * constant
    if quadratic == 0 or discriminant < 0:
        return CurveIntersectionResult()
    if discriminant == 0:
        return refused("singular")
    radical = RadicalElement.sqrt(discriminant)
    ctx = interval_context(60)

    def interval(value: Any, /) -> Any:
        lower, upper = coefficient_interval(value, bits=256)
        first = ctx.mpf(lower.numerator) / ctx.mpf(lower.denominator)
        last = ctx.mpf(upper.numerator) / ctx.mpf(upper.denominator)
        return ctx.mpf([first.a, last.b])

    box = np.asarray(surface.parameter_box, dtype=np.float64)
    points: list[ParametricIntersectionPoint] = []
    for side in (-1, 1):
        root_value = (-linear + side * radical) / (2 * quadratic)
        parameter = interval(root_value)
        if float(parameter.b) < source.first or float(parameter.a) > source.last:
            continue
        if not (source.first < float(parameter.a) and float(parameter.b) < source.last):
            return refused("continuation")
        point = [interval(base[i] + rate[i] * root_value) for i in range(3)]
        chart = _revolution_chart(frame, point, float(box[0, 0]), ctx)
        if chart is None:
            return refused("singular")
        angle, profile = chart
        if float(profile.b) < box[0, 1] or float(profile.a) > box[1, 1]:
            continue
        if not (box[0, 1] < float(profile.a) and float(profile.b) < box[1, 1]):
            return refused("continuation")
        pad = (
            64
            * np.finfo(np.float64).eps
            * max(
                1.0, abs(float(parameter.b)), abs(float(angle.b)), abs(float(profile.b))
            )
        )
        root = None
        while root is None:
            low = np.asarray(
                [float(value.a) - pad for value in (parameter, angle, profile)]
            )
            high = np.asarray(
                [float(value.b) + pad for value in (parameter, angle, profile)]
            )
            try:
                root = CurveSurfaceIntersectionRoot(
                    curve, surface, parameter_lower=low, parameter_upper=high
                )
            except ValueError:
                if pad > 1e-9:
                    return refused("resolution")
                pad *= 4.0
        parameters = np.asarray(root.parameters, dtype=np.float64)
        source_point = np.asarray(curve.evaluate(jnp.asarray(parameters[0])))
        surface_point = np.asarray(surface.patch.evaluate(jnp.asarray(parameters[1:])))
        points.append(
            ParametricIntersectionPoint(
                kind="crossing",
                certificate="krawczyk",
                parameters=parameters,
                parameter_lower=np.asarray(root.parameter_lower, dtype=np.float64),
                parameter_upper=np.asarray(root.parameter_upper, dtype=np.float64),
                point=source_point,
                gap_bound=float(np.linalg.norm(source_point - surface_point)),
                condition_estimate=1.0,
                root_id=root.root_id,
            )
        )
    del policy
    return CurveIntersectionResult(points=points)
