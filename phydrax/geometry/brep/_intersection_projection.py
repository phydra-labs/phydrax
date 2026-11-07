#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Certified projection onto the regular orthogonal-cylinder intersection atlas.

In the exact source frame put ``t = tan(theta/2)``, ``D = 1+t²``,
``N/D = n(theta)``, ``Z/D = z(theta)``, and ``H = R²D²-N²``.
The admitted regular family proves ``H > 0``. Ignoring the constant term,
the squared distance is ``L/D² - 2*s*qx*sqrt(H)/D``, where
``L = -2*qn*N*D + (Z-qz*D)²`` and ``s`` is the certified branch sign.
Its derivative with respect to t is ``(A + B/sqrt(H))/D³``, with
``A = L'D - 2LD'`` and ``B = -s*qx*D*(H'D - 2HD')``. Thus every finite
stationary point is a root of ``A²H-B²``; the unsquared sign test rejects
the extraneous roots and the gcd of A and B retains shared roots. Infinity
and interval endpoints are considered separately. Identically zero derivative
is reported as non-isolated, never as an arbitrarily selected minimum.

The polynomial is only a candidate generator. Root isolation, node ordering
and all distance comparisons use exact arithmetic or outward interval
enclosures. Closed-node binding uses the symbolic period, not binary64 tau.
Unsupported source atlases and undecidable source authorities are refused.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from ...algebraic._exact_univariate import _gcd, RealRootIsolator
from ._intersection_curve import (
    _large_seam_frame,
    _orthogonal_cylinder_curve_branch_sign,
    _OrthogonalCylinderFamily,
    IntersectionCurve,
    interval_context,
    orthogonal_cylinder_family,
)
from ._projection_contracts import BRepProjectionPolicy, BRepProjectionStatus
from ._spline_projection import CurveProjection


F = Fraction
P = tuple[Fraction, ...]


def _trim(p: P) -> P:
    while len(p) > 1 and not p[-1]:
        p = p[:-1]
    return p


def _add(a: P, b: P) -> P:
    return _trim(
        tuple(
            (a[i] if i < len(a) else F(0)) + (b[i] if i < len(b) else F(0))
            for i in range(max(len(a), len(b)))
        )
    )


def _scale(a: P, x: Fraction) -> P:
    return _trim(tuple(x * v for v in a))


def _mul(a: P, b: P) -> P:
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return _trim(tuple(result))


def _derivative(a: P) -> P:
    return _trim(tuple(i * a[i] for i in range(1, len(a))) or (F(0),))


def _sub(a: P, b: P) -> P:
    return _add(a, _scale(b, F(-1)))


def _stationary_polynomials(
    family: _OrthogonalCylinderFamily, sign: int, qx: Fraction, qn: Fraction, qz: Fraction
) -> tuple[P, P, P, P]:
    d = (F(1), F(0), F(1))
    a, b = family.small_normal_coefficients
    c, e = family.small_large_axis_coefficients
    r = family.small_radius
    n = _add(_scale(d, family.delta_n), (r * a, 2 * r * b, -r * a))
    z = _add(_scale(d, family.delta_large_axis), (r * c, 2 * r * e, -r * c))
    h = _sub(_scale(_mul(d, d), family.large_radius**2), _mul(n, n))
    w = _sub(z, _scale(d, qz))
    ell = _add(_scale(_mul(n, d), -2 * qn), _mul(w, w))
    aa = _sub(_mul(_derivative(ell), d), _scale(_mul(ell, _derivative(d)), F(2)))
    bb = _scale(
        _mul(d, _sub(_mul(_derivative(h), d), _scale(_mul(h, _derivative(d)), F(2)))),
        -sign * qx,
    )
    return aa, bb, h, _sub(_mul(_mul(aa, aa), h), _mul(bb, bb))


def _scalar(ctx: Any, x: Fraction) -> Any:
    return ctx.mpf(x.numerator) / ctx.mpf(x.denominator)


def _interval(ctx: Any, lo: Fraction, hi: Fraction) -> Any:
    return ctx.mpf((_scalar(ctx, lo), _scalar(ctx, hi)))


def _poly_interval(ctx: Any, p: P, x: Any) -> Any:
    out = _scalar(ctx, p[-1])
    for coefficient in reversed(p[:-1]):
        out = out * x + _scalar(ctx, coefficient)
    return out


def _bounds(value: Any) -> tuple[float, float]:
    return (
        math.nextafter(float(value.a), -math.inf),
        math.nextafter(float(value.b), math.inf),
    )


@dataclass(frozen=True)
class _Candidate:
    lower: float
    upper: float
    parameter: float
    root_id: tuple[object, ...]
    endpoint: bool


def _distance(
    ctx: Any,
    family: _OrthogonalCylinderFamily,
    sign: int,
    theta: Any,
    qx: Fraction,
    qn: Fraction,
    qz: Fraction,
) -> tuple[float, float] | None:
    cosine, sine = ctx.cos(theta), ctx.sin(theta)
    a, b = family.small_normal_coefficients
    c, d = family.small_large_axis_coefficients
    n = _scalar(ctx, family.delta_n) + _scalar(ctx, family.small_radius) * (
        _scalar(ctx, a) * cosine + _scalar(ctx, b) * sine
    )
    z = _scalar(ctx, family.delta_large_axis) + _scalar(ctx, family.small_radius) * (
        _scalar(ctx, c) * cosine + _scalar(ctx, d) * sine
    )
    h = _scalar(ctx, family.large_radius) ** 2 - n**2
    if float(h.a) <= 0:
        return None
    x = sign * ctx.sqrt(h)
    squared = (
        (x - _scalar(ctx, qx)) ** 2
        + (n - _scalar(ctx, qn)) ** 2
        + (z - _scalar(ctx, qz)) ** 2
    )
    lower, upper = _bounds(ctx.sqrt(squared))
    return max(0.0, lower), upper


@dataclass(slots=True)
class _PreparedProjection:
    """Per-query exact source, chart and root state for candidate isolation.

    Each isolator belongs to this query's exact polynomial, so refinements at
    shrinking widths reuse finished exact subdivision without changing results.
    """

    aa: P
    bb: P
    h: P
    root_isolator: RealRootIsolator
    common: P
    common_isolator: RealRootIsolator | None
    axis: int
    candidates: list[_Candidate]
    ctx: Any
    curve: IntersectionCurve
    edge_range: tuple[float, float]
    ends: np.ndarray
    family: _OrthogonalCylinderFamily
    first: float
    last: float
    qn: Fraction
    qx: Fraction
    qz: Fraction
    roots: tuple[tuple[Fraction, Fraction, int], ...]
    seam_gcd: P
    seam_gcd_isolator: RealRootIsolator | None
    seam_polynomial: P | None
    seam_isolator: RealRootIsolator | None
    seam_roots: tuple[tuple[Fraction, Fraction, int], ...]
    sign: int
    starts: np.ndarray
    tolerance: Fraction
    uncertain: bool

    def seam_order(self, 
        root_index: int, seam_index: int
    ) -> tuple[int, Fraction, Fraction, Fraction, Fraction]:
        """Order two isolated algebraic roots, proving equality by their gcd."""
        assert self.seam_polynomial is not None and self.seam_isolator is not None
        width = self.tolerance
        while True:
            rl, rh, _ = self.root_isolator.isolate(tolerance=width)[root_index]
            sl, sh, _ = self.seam_isolator.isolate(tolerance=width)[seam_index]
            if rh < sl:
                return -1, rl, rh, sl, sh
            if sh < rl:
                return 1, rl, rh, sl, sh
            if self.seam_gcd_isolator is not None:
                common_intervals = self.seam_gcd_isolator.isolate(tolerance=width / 64)
                if any(
                    rl <= gl <= gh <= rh and sl <= gl <= gh <= sh
                    for gl, gh, _ in common_intervals
                ):
                    return 0, rl, rh, sl, sh
            width /= 16

    def angle_order(self, 
        root_index: int, turn: int, target: Fraction
    ) -> tuple[int, Fraction, Fraction]:
        """Compare an algebraic half-angle root to an exact rational angle."""
        width = self.tolerance
        digits = 80
        while True:
            rl, rh, _ = self.root_isolator.isolate(tolerance=width)[root_index]
            if rl == rh == 0 and turn == 0 and target == 0:
                return 0, rl, rh
            local = interval_context(digits)
            angle = (
                2 * local.atan2(_interval(local, rl, rh), local.mpf(1))
                + turn * 2 * local.pi
            )
            boundary = _scalar(local, target)
            if angle.b < boundary.a:
                return -1, rl, rh
            if angle.a > boundary.b:
                return 1, rl, rh
            width /= 16
            digits += 20

    def finite_node_order(
        self, index: int, turn: int, node: int, node_lo: float, node_hi: float,
    ) -> tuple[int, Fraction, Fraction, tuple[Fraction, Fraction] | None, Fraction | None] | None:
        """Compare one stationary root to the exact source-node authority."""
        if int(self.curve.node_axes[node]) == self.axis:
            target = F.from_float(float(self.curve.node_values[node]))
            order, lo, hi = self.angle_order(index, turn, target)
            return order, lo, hi, None, target
        possible = []
        for seam_index, (sl, sh, _) in enumerate(self.seam_roots):
            theta = 2 * self.ctx.atan2(_interval(self.ctx, sl, sh), self.ctx.mpf(1)) + turn * 2 * self.ctx.pi
            lower, upper = _bounds(theta)
            if upper >= node_lo and lower <= node_hi:
                possible.append(seam_index)
        if len(possible) != 1:
            return None
        order, lo, hi, sl, sh = self.seam_order(index, possible[0])
        return order, lo, hi, (sl, sh), None

    def finite_adjacent_candidate(
        self, index: int, multiplicity: int, turn: int, node: int,
        node_order: tuple[int, Fraction, Fraction, tuple[Fraction, Fraction] | None, Fraction | None],
    ) -> None:
        """Place a certified stationary root on the proved side of its node."""
        order, refined_lo, refined_hi, seam_interval, target = node_order
        if (order < 0 and self.first >= node) or (order > 0 and self.last <= node):
            return
        chart = node - 1 if order < 0 else node
        cfirst, clast = max(self.first, float(chart)), min(self.last, float(chart + 1))
        s, e = (
            F.from_float(float(self.starts[chart])),
            F.from_float(float(self.ends[chart])),
        )
        left = s + (F.from_float(cfirst) - chart) * (e - s)
        right = s + (F.from_float(clast) - chart) * (e - s)
        refined_angle = (
            2 * self.ctx.atan2(_interval(self.ctx, refined_lo, refined_hi), self.ctx.mpf(1))
            + turn * 2 * self.ctx.pi
        )
        rlo, rhi = _bounds(refined_angle)
        if (cfirst != node and rlo <= float(left)) or (
            clast != node and rhi >= float(right)
        ):
            self.uncertain = True
        else:
            # Compute the representative relative to the proved
            # source node, rather than subtracting its rounded
            # binary64 angle from a nearly equal root angle.
            digits = 80
            while True:
                local = interval_context(digits)
                root_angle = (
                    2
                    * local.atan2(
                        _interval(local, refined_lo, refined_hi), local.mpf(1)
                    )
                    + turn * 2 * local.pi
                )
                if seam_interval is None:
                    assert target is not None
                    node_angle = _scalar(local, target)
                else:
                    node_angle = (
                        2
                        * local.atan2(
                            _interval(local, *seam_interval), local.mpf(1)
                        )
                        + turn * 2 * local.pi
                    )
                delta = root_angle - node_angle
                if (order < 0 and delta.b < 0) or (order > 0 and delta.a > 0):
                    break
                digits += 20
            parameter = node + (float(delta.a) + float(delta.b)) / (
                2 * float(e - s)
            )
            bounds = _distance(self.ctx, self.family, self.sign, refined_angle, self.qx, self.qn, self.qz)
            if bounds is None or not chart <= parameter <= chart + 1:
                self.uncertain = True
            else:
                self.candidates.append(
                    _Candidate(
                        *bounds,
                        parameter,
                        (
                            "stationary",
                            index,
                            refined_lo,
                            refined_hi,
                            multiplicity,
                            turn,
                            chart,
                        ),
                        False,
                    )
                )

    def finite_node_candidates(self, index: int, multiplicity: int, turn: int, tl: float, th: float) -> bool:
        handled = False
        for node in range(1, self.curve.num_charts):
            if self.last < node - 1 or self.first > node + 1:
                continue
            node_lo = float(self.curve.node_lower[node, self.axis])
            node_hi = float(self.curve.node_upper[node, self.axis])
            if th < node_lo or tl > node_hi:
                continue
            node_order = self.finite_node_order(index, turn, node, node_lo, node_hi)
            if node_order is None:
                self.uncertain = True
                handled = True
                break
            order, refined_lo, refined_hi, seam_interval, target = node_order
            if order == 0:
                if self.first <= node <= self.last:
                    refined_angle = (
                        2
                        * self.ctx.atan2(
                            _interval(self.ctx, refined_lo, refined_hi), self.ctx.mpf(1)
                        )
                        + turn * 2 * self.ctx.pi
                    )
                    bounds = _distance(self.ctx, self.family, self.sign, refined_angle, self.qx, self.qn, self.qz)
                    if bounds is None:
                        self.uncertain = True
                    else:
                        self.candidates.append(
                            _Candidate(
                                *bounds,
                                float(node),
                                (
                                    "stationary-node",
                                    node,
                                    refined_lo,
                                    refined_hi,
                                    multiplicity,
                                    turn,
                                ),
                                float(node) in self.edge_range,
                            )
                        )
            else:
                self.finite_adjacent_candidate(index, multiplicity, turn, node, node_order)
            handled = True
            break
        return handled

    def finite_chart_candidates(self, index: int, lo: Fraction, hi: Fraction, multiplicity: int, turn: int, lifted: Any, tl: float, th: float) -> tuple[Fraction, Fraction]:
        for chart in range(self.curve.num_charts):
            cfirst, clast = max(self.first, float(chart)), min(self.last, float(chart + 1))
            if cfirst > clast:
                continue
            s, e = (
                F.from_float(float(self.starts[chart])),
                F.from_float(float(self.ends[chart])),
            )
            left = s + (F.from_float(cfirst) - chart) * (e - s)
            right = s + (F.from_float(clast) - chart) * (e - s)
            lower_proved = upper_proved = False
            if chart == 0 and cfirst == 0:
                start = F.from_float(float(self.starts[0]))
                boundary = _scalar(self.ctx, start)
                if lifted.b < boundary.a:
                    continue
                if lifted.a > boundary.b:
                    lower_proved = True
                else:
                    order, lo, hi = self.angle_order(index, turn, start)
                    if order <= 0:
                        continue
                    lifted = (
                        2 * self.ctx.atan2(_interval(self.ctx, lo, hi), self.ctx.mpf(1))
                        + turn * 2 * self.ctx.pi
                    )
                    tl, th = _bounds(lifted)
                    lower_proved = True
            if chart == self.curve.num_charts - 1 and clast == self.curve.num_charts:
                start = F.from_float(float(self.starts[0]))
                boundary = _scalar(self.ctx, start) + 2 * self.ctx.pi
                if lifted.a > boundary.b:
                    continue
                if lifted.b < boundary.a:
                    upper_proved = True
                else:
                    order, lo, hi = self.angle_order(index, turn - 1, start)
                    if order >= 0:
                        continue
                    lifted = (
                        2 * self.ctx.atan2(_interval(self.ctx, lo, hi), self.ctx.mpf(1))
                        + turn * 2 * self.ctx.pi
                    )
                    tl, th = _bounds(lifted)
                    upper_proved = True
            if (not lower_proved and th < float(left)) or (
                not upper_proved and tl > float(right)
            ):
                continue
            if (not lower_proved and tl <= float(left)) or (
                not upper_proved and th >= float(right)
            ):
                self.uncertain = True
                continue
            if upper_proved:
                displacement = lifted - (
                    _scalar(self.ctx, F.from_float(float(self.starts[0]))) + 2 * self.ctx.pi
                )
                parameter = self.curve.num_charts + (
                    float(displacement.a) + float(displacement.b)
                ) / (2 * float(e - s))
            else:
                parameter = chart + ((tl + th) / 2 - float(s)) / float(e - s)
            bounds = _distance(self.ctx, self.family, self.sign, lifted, self.qx, self.qn, self.qz)
            if bounds is None:
                self.uncertain = True
                continue
            self.candidates.append(
                _Candidate(
                    *bounds,
                    parameter,
                    ("stationary", index, lo, hi, multiplicity, turn),
                    False,
                )
            )
        return lo, hi

    def infinity_node_order(self, node: int, turn: int, angle: Any) -> int | None:
        """Order the omitted half-angle point against one exact source node."""
        if int(self.curve.node_axes[node]) == self.axis:
            target = _scalar(self.ctx, F.from_float(float(self.curve.node_values[node])))
            if angle.b < target.a:
                return -1
            if angle.a > target.b:
                return 1
            return None
        if self.seam_polynomial is None or self.seam_isolator is None:
            return None
        if len(self.seam_polynomial) < 3:
            return 0
        possible = []
        for seam_index, (sl, sh, _) in enumerate(self.seam_roots):
            for seam_turn in (turn, turn + 1):
                seam_theta = 2 * self.ctx.atan2(_interval(self.ctx, sl, sh), self.ctx.mpf(1)) + seam_turn * 2 * self.ctx.pi
                lower, upper = _bounds(seam_theta)
                if upper >= float(self.curve.node_lower[node, self.axis]) and lower <= float(self.curve.node_upper[node, self.axis]):
                    possible.append((seam_index, seam_turn))
        if len(possible) != 1:
            return None
        seam_index, seam_turn = possible[0]
        width, digits = self.tolerance, 80
        while True:
            sl, sh, _ = self.seam_isolator.isolate(tolerance=width)[seam_index]
            local = interval_context(digits)
            seam_theta = 2 * local.atan2(_interval(local, sl, sh), local.mpf(1)) + seam_turn * 2 * local.pi
            exact_angle = (2 * turn + 1) * local.pi
            if exact_angle.b < seam_theta.a:
                return -1
            if exact_angle.a > seam_theta.b:
                return 1
            width /= 16
            digits += 20

    def infinity_node_candidates(self, turn: int, angle: Any, tl: float, th: float) -> bool:
        handled = False
        for node in range(1, self.curve.num_charts):
            if self.last < node - 1 or self.first > node + 1:
                continue
            if th < float(self.curve.node_lower[node, self.axis]) or tl > float(
                self.curve.node_upper[node, self.axis]
            ):
                continue
            order = self.infinity_node_order(node, turn, angle)
            if order is None:
                self.uncertain = True
                handled = True
                break
            if order == 0:
                if self.first <= node <= self.last:
                    bounds = _distance(self.ctx, self.family, self.sign, angle, self.qx, self.qn, self.qz)
                    if bounds is None:
                        self.uncertain = True
                    else:
                        self.candidates.append(
                            _Candidate(
                                *bounds,
                                float(node),
                                ("infinity-node", node, turn),
                                float(node) in self.edge_range,
                            )
                        )
            elif not ((order < 0 and self.first >= node) or (order > 0 and self.last <= node)):
                chart = node - 1 if order < 0 else node
                cfirst, clast = max(self.first, float(chart)), min(self.last, float(chart + 1))
                s, e = (
                    F.from_float(float(self.starts[chart])),
                    F.from_float(float(self.ends[chart])),
                )
                left = s + (F.from_float(cfirst) - chart) * (e - s)
                right = s + (F.from_float(clast) - chart) * (e - s)
                right_bound = (
                    _scalar(self.ctx, F.from_float(float(self.starts[0]))) + 2 * self.ctx.pi
                    if clast == self.curve.num_charts
                    else _scalar(self.ctx, right)
                )
                lower_ok = cfirst == node or angle.a > _scalar(self.ctx, left).b
                upper_ok = clast == node or angle.b < right_bound.a
                if lower_ok and upper_ok:
                    parameter = chart + ((tl + th) / 2 - float(s)) / float(e - s)
                    bounds = _distance(self.ctx, self.family, self.sign, angle, self.qx, self.qn, self.qz)
                    if bounds is None:
                        self.uncertain = True
                    else:
                        self.candidates.append(
                            _Candidate(
                                *bounds, parameter, ("infinity", turn, chart), False
                            )
                        )
                else:
                    self.uncertain = True
            handled = True
            break
        return handled

    def infinity_chart_candidates(self, turn: int, angle: Any, tl: float, th: float) -> None:
        for chart in range(self.curve.num_charts):
            cfirst, clast = max(self.first, float(chart)), min(self.last, float(chart + 1))
            if cfirst > clast:
                continue
            s, e = (
                F.from_float(float(self.starts[chart])),
                F.from_float(float(self.ends[chart])),
            )
            left = s + (F.from_float(cfirst) - chart) * (e - s)
            right = s + (F.from_float(clast) - chart) * (e - s)
            right_bound = (
                _scalar(self.ctx, F.from_float(float(self.starts[0]))) + 2 * self.ctx.pi
                if clast == self.curve.num_charts
                else _scalar(self.ctx, right)
            )
            if angle.b < _scalar(self.ctx, left).a or angle.a > right_bound.b:
                continue
            if angle.a <= _scalar(self.ctx, left).b or angle.b >= right_bound.a:
                self.uncertain = True
                continue
            parameter = chart + ((tl + th) / 2 - float(s)) / float(e - s)
            bounds = _distance(self.ctx, self.family, self.sign, angle, self.qx, self.qn, self.qz)
            if bounds is None:
                self.uncertain = True
            else:
                self.candidates.append(
                    _Candidate(*bounds, parameter, ("infinity", turn, chart), False)
                )

    def collect_finite_candidates(self) -> None:
        for index, (lo, hi, multiplicity) in enumerate(self.roots):
            width = self.tolerance
            digits = 80
            reject = False
            while True:
                shared = (
                    ()
                    if self.common_isolator is None
                    else self.common_isolator.isolate(tolerance=width / 64)
                )
                if any(lo <= cl <= ch <= hi for cl, ch, _ in shared):
                    break
                local = interval_context(digits)
                argument = _interval(local, lo, hi)
                av, bv = (
                    _poly_interval(local, self.aa, argument),
                    _poly_interval(local, self.bb, argument),
                )
                if (av.a > 0 and bv.a > 0) or (av.b < 0 and bv.b < 0):
                    reject = True
                    break
                if (av.a > 0 and bv.b < 0) or (av.b < 0 and bv.a > 0):
                    break
                width /= 16
                digits += 20
                lo, hi, _ = self.root_isolator.isolate(tolerance=width)[index]
            if reject:
                continue
            arg = _interval(self.ctx, lo, hi)
            angle = 2 * self.ctx.atan2(arg, self.ctx.mpf(1))
            alo, ahi = _bounds(angle)
            # Consider each lift intersecting the authored chart range. The interval
            # for pi is retained; binary64 2*pi is never treated as exact closure.
            low_turn = math.floor((float(self.starts[0]) - ahi) / math.tau) - 1
            high_turn = math.ceil((float(self.ends[-1]) - alo) / math.tau) + 1
            for turn in range(low_turn, high_turn + 1):
                lifted = angle + turn * 2 * self.ctx.pi
                tl, th = _bounds(lifted)
                if self.finite_node_candidates(index, multiplicity, turn, tl, th):
                    continue
                lo, hi = self.finite_chart_candidates(
                    index, lo, hi, multiplicity, turn, lifted, tl, th
                )

    def collect_infinity_candidates(self) -> None:
        # The half-angle chart omits theta=pi. It is stationary only if the
        # homogeneous derivative vanishes there; otherwise it is not a candidate.
        ca = self.aa[4] if len(self.aa) > 4 else F(0)
        cb = self.bb[6] if len(self.bb) > 6 else F(0)
        ch = self.h[4] if len(self.h) > 4 else F(0)
        if ca * ca * ch == cb * cb and (ca == cb == 0 or ca * cb < 0):
            for turn in range(
                math.floor(float(self.starts[0]) / math.tau) - 1,
                math.ceil(float(self.ends[-1]) / math.tau) + 1,
            ):
                angle = (2 * turn + 1) * self.ctx.pi
                tl, th = _bounds(angle)
                if self.infinity_node_candidates(turn, angle, tl, th):
                    continue
                self.infinity_chart_candidates(turn, angle, tl, th)


def _append_endpoint_candidate(
    curve: IntersectionCurve,
    family: _OrthogonalCylinderFamily,
    sign: int,
    parameter: float,
    edge_range: tuple[float, float],
    query_frame: tuple[Fraction, Fraction, Fraction],
    starts: np.ndarray,
    ends: np.ndarray,
    ctx: Any,
    candidates: list[_Candidate],
) -> bool:
    """Enclose an endpoint using its source node, including symbolic closure."""
    axis = 2 * family.small_index
    chart = min(int(math.floor(parameter)), curve.num_charts - 1)
    s, e = F.from_float(float(starts[chart])), F.from_float(float(ends[chart]))
    theta = s + (F.from_float(parameter) - chart) * (e - s)
    if parameter == curve.num_charts:
        theta = F.from_float(float(starts[0]))
    elif (
        parameter.is_integer()
        and 0 < parameter < curve.num_charts
        and int(curve.node_axes[int(parameter)]) != axis
    ):
        node = int(parameter)
        lo = F.from_float(float(curve.node_lower[node, axis]))
        hi = F.from_float(float(curve.node_upper[node, axis]))
        bounds = _distance(ctx, family, sign, _interval(ctx, lo, hi), *query_frame)
        if bounds is None:
            return False
        candidates.append(_Candidate(*bounds, parameter, ("node", node), parameter in edge_range))
        return True
    bounds = _distance(ctx, family, sign, _scalar(ctx, theta), *query_frame)
    if bounds is None:
        return False
    candidates.append(_Candidate(*bounds, parameter, ("endpoint", parameter), parameter in edge_range))
    return True


def _select_candidate_projection(
    curve: IntersectionCurve, parameter_range: tuple[float, float], query: np.ndarray,
    policy: BRepProjectionPolicy, candidates: list[_Candidate], ctx: Any,
) -> CurveProjection:
    """Compare certified distance enclosures and preserve distinct roots."""
    first, last = parameter_range
    best = min(candidates, key=lambda item: item.upper)
    if best.upper - best.lower > policy.ambiguity_tolerance:
        return CurveProjection(np.full(3, np.nan), np.nan, 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED))
    others = [
        item
        for item in candidates
        if item is not best
        and item.lower <= best.upper + policy.ambiguity_tolerance
        and item.upper >= best.lower - policy.ambiguity_tolerance
    ]
    others = [
        item
        for item in others
        if not (
            item.parameter == best.parameter
            and {item.root_id[0], best.root_id[0]}
            in ({"node", "stationary-node"}, {"node", "infinity-node"})
        )
    ]
    if curve.closed and first == 0 and last == curve.num_charts:
        others = [
            item
            for item in others
            if not ({item.parameter, best.parameter} == {0.0, float(curve.num_charts)})
        ]
    status = (
        BRepProjectionStatus.AMBIGUOUS
        if others
        else BRepProjectionStatus.SEAM
        if best.endpoint
        else BRepProjectionStatus.UNIQUE
    )
    point = np.asarray(curve.evaluate_coupled(best.parameter).point, dtype=np.float64)
    squared = sum(
        (
            (F.from_float(float(x)) - F.from_float(float(y))) ** 2
            for x, y in zip(point, query, strict=True)
        ),
        F(0),
    )
    upper = max(best.upper, _bounds(ctx.sqrt(_scalar(ctx, squared)))[1])
    return CurveProjection(
        point,
        best.parameter,
        min(item.lower for item in candidates),
        upper,
        int(status),
        ("intersection", curve.branch_id),
        best.root_id,
    )


def _prepare_seam_roots(
    family: _OrthogonalCylinderFamily, sign: int, polynomial: P, tolerance: Fraction,
) -> tuple[
    P | None, RealRootIsolator | None, tuple[tuple[Fraction, Fraction, int], ...], P
]:
    """Isolate the rational large-seam authority and its stationary gcd."""
    _, sine_frame, seam_angle = _large_seam_frame(family)
    if seam_angle != 0 or not sine_frame[1]:
        return None, None, (), (F(1),)
    seam_n = -sign * (1 if sine_frame[1] > 0 else -1) * sine_frame[0] * family.large_radius
    a, b = family.small_normal_coefficients
    seam_polynomial = _trim((
        family.delta_n + family.small_radius * a - seam_n,
        2 * family.small_radius * b,
        family.delta_n - family.small_radius * a - seam_n,
    ))
    isolator = RealRootIsolator(seam_polynomial)
    return (
        seam_polynomial,
        isolator,
        isolator.isolate(tolerance=tolerance),
        _gcd(polynomial, seam_polynomial),
    )


def project_intersection_curve(
    curve: IntersectionCurve,
    parameter_range: tuple[float, float],
    query: np.ndarray,
    policy: BRepProjectionPolicy,
    /,
    *,
    edge_range: tuple[float, float],
) -> CurveProjection:
    """Project onto a certified cylinder branch or return an honest refusal."""
    unresolved = lambda: CurveProjection(
        np.full(3, np.nan), np.nan, 0.0, np.inf, int(BRepProjectionStatus.UNRESOLVED)
    )
    first, last = map(float, parameter_range)
    if not (0 <= first <= last <= curve.num_charts) or not np.all(np.isfinite(query)):
        return unresolved()
    family = orthogonal_cylinder_family(curve.first, curve.second)
    if family is None:
        return unresolved()
    sign = _orthogonal_cylinder_curve_branch_sign(curve, family)
    if sign is None:
        return unresolved()
    origin = family.large.origin_exact
    vectors = (family.small.axis_exact, family.normal, family.large.axis_exact)
    # The family recognizer admits only rational source frames. Recheck that
    # invariant here before entering the rational polynomial solver.
    if not all(isinstance(value, Fraction) for value in origin):
        return unresolved()
    if not all(isinstance(value, Fraction) for vector in vectors for value in vector):
        return unresolved()
    rational_origin = tuple(value for value in origin if isinstance(value, Fraction))
    rational_vectors = tuple(
        tuple(value for value in vector if isinstance(value, Fraction))
        for vector in vectors
    )
    delta = tuple(F.from_float(float(q)) - o for q, o in zip(query, rational_origin, strict=True))
    qx, qn, qz = (
        sum((x * y for x, y in zip(delta, v, strict=True)), F(0))
        for v in rational_vectors
    )
    aa, bb, h, polynomial = _stationary_polynomials(family, sign, qx, qn, qz)
    ctx = interval_context(80)
    axis = 2 * family.small_index
    starts = np.asarray(curve.chart_start, dtype=np.float64)[:, axis]
    ends = np.asarray(curve.chart_end, dtype=np.float64)[:, axis]
    candidates: list[_Candidate] = []
    uncertain = False

    endpoint_args = (curve, family, sign)
    endpoint_context = (edge_range, (qx, qn, qz), starts, ends, ctx, candidates)
    if not _append_endpoint_candidate(*endpoint_args, first, *endpoint_context) or (
        last != first
        and not _append_endpoint_candidate(*endpoint_args, last, *endpoint_context)
    ):
        return unresolved()
    if first == last:
        return _select_candidate_projection(curve, (first, last), query, policy, candidates, ctx)
    if polynomial == (F(0),):
        if aa != (F(0),) or bb != (F(0),):
            return unresolved()
        best = min(candidates, key=lambda item: item.upper)
        return CurveProjection(
            np.full(3, np.nan), np.nan, best.lower, best.upper,
            int(BRepProjectionStatus.NON_ISOLATED), ("intersection", curve.branch_id), best.root_id,
        )

    tolerance = (
        F.from_float(min(policy.parametric_tolerance, policy.ambiguity_tolerance)) / 16
    )
    # This first pass only needs ordered isolating intervals. Node and common
    # root comparisons below use the default rational-recognizing isolation.
    root_isolator = RealRootIsolator(polynomial)
    roots = root_isolator.isolate(tolerance=tolerance, recognize_rational_roots=False)
    common = _gcd(aa, bb)
    seam_polynomial, seam_isolator, seam_roots, seam_gcd = _prepare_seam_roots(
        family, sign, polynomial, tolerance
    )

    prepared = _PreparedProjection(
        aa=aa,
        bb=bb,
        h=h,
        root_isolator=root_isolator,
        common=common,
        common_isolator=RealRootIsolator(common) if len(common) > 1 else None,
        axis=axis,
        candidates=candidates,
        ctx=ctx,
        curve=curve,
        edge_range=edge_range,
        ends=ends,
        family=family,
        first=first,
        last=last,
        qn=qn,
        qx=qx,
        qz=qz,
        roots=roots,
        seam_gcd=seam_gcd,
        seam_gcd_isolator=RealRootIsolator(seam_gcd) if len(seam_gcd) > 1 else None,
        seam_polynomial=seam_polynomial,
        seam_isolator=seam_isolator,
        seam_roots=seam_roots,
        sign=sign,
        starts=starts,
        tolerance=tolerance,
        uncertain=uncertain,
    )
    prepared.collect_finite_candidates()
    prepared.collect_infinity_candidates()
    if prepared.uncertain:
        return unresolved()
    return _select_candidate_projection(curve, (first, last), query, policy, candidates, ctx)
