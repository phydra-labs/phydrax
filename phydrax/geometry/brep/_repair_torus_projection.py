#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Source-bound full-longitude torus strips and certified distance bounds.

Recognition uses the authored coedges and exact endpoint/carrier identities, not
parameter boxes or numerical coincidence.  The distance routine works in the
orthonormal meridian plane; its interval result remains valid in degenerate and
unresolved nearest-point cases.  It does not infer a solid from a single face.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import cast

import numpy as np

from ...discretization._exact_algebraic_geometry import coefficient_interval, ExactScalar
from .._rational import fraction_down, fraction_up
from ._boolean_core import PeriodicPCurve
from ._carrier_pieces import carrier_payload
from ._exact import square_root_interval
from ._intersection import ConicParameterEndpoint, NativePeriodEndpoint
from ._model import BRepCurve, BRepModel
from ._patches import (
    CircleCurve,
    ConicParameterRoot,
    LineCurve,
    TorusPatch,
    TorusRootedCirclePCurve,
)
from ._source_repair_face_completion import _circle_winding, _strip_rings
from ._trim_helpers import period_offset, period_scalar_bounds


Vector = tuple[ExactScalar, ...]
Interval = tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class TorusStrip:
    """Evidence tied to one unchanged model face and its two exact root levels."""

    model: BRepModel
    face: int
    patch: TorusPatch
    lower: ConicParameterRoot
    upper: ConicParameterRoot
    seam: int
    rings: tuple[tuple[int, ...], tuple[int, ...]]


def _dot(a: Vector, b: Vector) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _root_key(root: ConicParameterRoot) -> tuple[ExactScalar, ExactScalar, int]:
    return root.cosine_exact, root.sine_exact, root.turns


def _bound_endpoint(endpoint: object, curve: BRepCurve, edge_root: object) -> bool:
    """Check the coedge parameter root is transported from its exact edge root."""
    if isinstance(endpoint, ConicParameterEndpoint) and isinstance(
        edge_root, ConicParameterEndpoint
    ):
        return (
            carrier_payload(endpoint.carrier) == carrier_payload(curve)
            and endpoint.parameter_exact == edge_root.parameter_exact
            and endpoint.point_exact == edge_root.point_exact
            and endpoint.root.identity_payload() == edge_root.root.identity_payload()
        )
    if isinstance(endpoint, NativePeriodEndpoint) and isinstance(
        edge_root, NativePeriodEndpoint
    ):
        return (
            carrier_payload(endpoint.curve) == carrier_payload(curve)
            and endpoint.rational == edge_root.rational
            and endpoint.turns == edge_root.turns
        )
    return False


def recognize_torus_strip(model: BRepModel, face: int, /) -> TorusStrip | None:
    """Prove a single-loop, exactly seamed, once-wound ring-torus strip.

    Only the canonical meridian source and exact rooted latitudes are accepted.
    Any unsupported carrier, unproved sheet, incidence, or frame is refused.
    """
    try:
        patch = model.patches[face]
        if not isinstance(patch, TorusPatch):
            return None
        major, minor = patch.major_radius_exact, patch.minor_radius_exact
        if (
            not isinstance(major, Fraction)
            or not isinstance(minor, Fraction)
            or not major > minor > 0
        ):
            return None
        frame = (patch.first_axis_exact, patch.second_axis_exact, patch.axis_exact)
        if any(
            _dot(a, b) != int(i == j)
            for i, a in enumerate(frame)
            for j, b in enumerate(frame)
        ):
            return None
        a, b, axis = frame
        cross = (
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        )
        if cross != axis:
            return None
        geometry = model.geometry
        ring1, ring2, seam = _strip_rings(geometry, face)
        uses = [
            c for c in geometry.face_loops[face][0] if geometry.coedge_edges[c] == seam
        ]
        charts = [geometry.pcurves[c] for c in uses]
        if any(not isinstance(c, PeriodicPCurve) for c in charts):
            return None
        charts = cast(list[PeriodicPCurve], charts)
        if any(carrier_payload(c.patch) != carrier_payload(patch) for c in charts):
            return None
        line = charts[0].source_curve
        if (
            not isinstance(line, LineCurve)
            or line.origin_exact != (0, 0)
            or line.direction_exact != (0, 1)
            or carrier_payload(line) != carrier_payload(charts[1].source_curve)
        ):
            return None
        loop = geometry.face_loops[face][0]
        oriented = [
            geometry.edge_vertices[geometry.coedge_edges[c]]
            if geometry.coedge_senses[c] == 1
            else geometry.edge_vertices[geometry.coedge_edges[c]][::-1]
            for c in loop
        ]
        if any(geometry.coedge_senses[c] not in (-1, 1) for c in loop) or any(
            oriented[i][1] != oriented[(i + 1) % len(oriented)][0]
            for i in range(len(oriented))
        ):
            return None
        shifts = [c.period_shifts for c in charts]
        if (
            shifts[0][1] != 0
            or shifts[1][1] != 0
            or abs(shifts[0][0] - shifts[1][0]) != 1
        ):
            return None
        meridian = geometry.curves[geometry.edge_curves[seam]]
        if not isinstance(meridian, CircleCurve):
            return None
        if (
            meridian.center_exact
            != tuple(c + major * x for c, x in zip(patch.center_exact, a, strict=True))
            or meridian.first_axis_exact != a
            or meridian.second_axis_exact != axis
            or meridian.radius_exact != minor
        ):
            return None
        for c in geometry.face_loops[face][0]:
            edge = geometry.coedge_edges[c]
            if not all(
                _bound_endpoint(x, geometry.pcurves[c], y)
                for x, y in zip(
                    geometry.coedge_endpoint_roots[c],
                    geometry.edge_endpoint_roots[edge],
                    strict=True,
                )
            ):
                return None
        # The canonical meridian parameter is exactly the profile angle.
        seam_roots = geometry.edge_endpoint_roots[seam]
        from ._source_repair_completion import _endpoint_interval, _vertex_point

        seam_keys = []
        cardinal_phases = {
            Fraction(0): (1, 0),
            Fraction(1, 4): (0, 1),
            Fraction(1, 2): (-1, 0),
            Fraction(3, 4): (0, -1),
        }
        for root, vertex in zip(seam_roots, geometry.edge_vertices[seam], strict=True):
            if isinstance(root, ConicParameterEndpoint):
                if (
                    root.parameter_exact is not None
                    or carrier_payload(root.carrier) != carrier_payload(meridian)
                    or carrier_payload(root.root.conic) != carrier_payload(meridian)
                    or root.root.exact_point() != root.point_exact
                ):
                    return None
                key, point = _root_key(root.root), root.point_exact
            elif isinstance(root, NativePeriodEndpoint):
                phase = cardinal_phases.get(root.turns % 1)
                if (
                    root.patch is not None
                    or root.rational != 0
                    or phase is None
                    or carrier_payload(root.curve) != carrier_payload(meridian)
                ):
                    return None
                co, si = phase
                key = (co, si, math.floor(root.turns))
                point = tuple(
                    center + minor * (co * x + si * z)
                    for center, x, z in zip(meridian.center_exact, a, axis, strict=True)
                )
            else:
                return None
            if point != _vertex_point(model, vertex):
                return None
            seam_keys.append(key)
        seam_start = _endpoint_interval(model, seam, 0, repaired=True)
        seam_end = _endpoint_interval(model, seam, 1, repaired=True)
        if seam_start[1] >= seam_end[0]:
            return None
        levels = []
        winds = []
        for ring in (ring1, ring2):
            winding = _circle_winding(model, ring)
            level = None
            for c in ring:
                chart = geometry.pcurves[c]
                if not isinstance(chart, TorusRootedCirclePCurve):
                    return None
                root = chart.root_line.root
                if carrier_payload(root.conic) != carrier_payload(meridian):
                    return None
                if level is None:
                    level = root
                elif _root_key(root) != _root_key(level):
                    return None
                edge = geometry.coedge_edges[c]
                circle = geometry.curves[geometry.edge_curves[edge]]
                if not isinstance(circle, CircleCurve):
                    return None
                # A latitude's 3D carrier must equal its exact symbolic action.
                turns = chart.turns % 1
                cardinal = {
                    Fraction(0): (1, 0),
                    Fraction(1, 4): (0, 1),
                    Fraction(1, 2): (-1, 0),
                    Fraction(3, 4): (0, -1),
                }.get(turns)
                if cardinal is None:
                    return None
                co, si = cardinal
                first = tuple(co * x + si * y for x, y in zip(a, b, strict=True))
                tangent = tuple(
                    chart.orientation * (-si * x + co * y)
                    for x, y in zip(a, b, strict=True)
                )
                if (
                    circle.first_axis_exact != first
                    or circle.second_axis_exact != tangent
                    or circle.radius_exact != major + minor * root.cosine_exact
                    or circle.center_exact
                    != tuple(
                        x + minor * root.sine_exact * z
                        for x, z in zip(patch.center_exact, axis, strict=True)
                    )
                ):
                    return None
            levels.append(cast(ConicParameterRoot, level))
            ring_chart = cast(TorusRootedCirclePCurve, geometry.pcurves[ring[0]])
            winds.append(winding * ring_chart.orientation)
        if winds[0] != -winds[1]:
            return None
        if set(seam_keys) != {_root_key(r) for r in levels}:
            return None
        ordered = sorted(
            zip(levels, (ring1, ring2), strict=True),
            key=lambda pair: pair[0].parameter_lower,
        )
        (low, low_ring), (high, high_ring) = ordered
        period_lo, _ = period_scalar_bounds(period_offset(Fraction(0), Fraction(1)))
        if not (
            low.parameter_upper < high.parameter_lower
            and Fraction.from_float(high.parameter_upper)
            - Fraction.from_float(low.parameter_lower)
            < Fraction.from_float(float(period_lo))
        ):
            return None
        return TorusStrip(model, face, patch, low, high, seam, (low_ring, high_ring))
    except (
        AttributeError,
        IndexError,
        TypeError,
        ValueError,
        OverflowError,
        ZeroDivisionError,
    ):
        return None


def _add(a: Interval, b: Interval) -> Interval:
    return a[0] + b[0], a[1] + b[1]


def _neg(a: Interval) -> Interval:
    return -a[1], -a[0]


def _mul(a: Interval, b: Interval) -> Interval:
    values = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(values), max(values)


def _square(a: Interval) -> Interval:
    return (
        Fraction(0) if a[0] <= 0 <= a[1] else min(a[0] ** 2, a[1] ** 2),
        max(a[0] ** 2, a[1] ** 2),
    )


def _sqrt(a: Interval) -> Interval:
    lo = square_root_interval(max(Fraction(0), a[0]))[0]
    hi = square_root_interval(max(Fraction(0), a[1]))[1]
    if not math.isfinite(lo) or not math.isfinite(hi):
        raise OverflowError
    return Fraction.from_float(lo), Fraction.from_float(hi)


def _exact(value: ExactScalar) -> Interval:
    return coefficient_interval(value, bits=128)


def torus_strip_distance_bounds(
    strip: TorusStrip, point: np.ndarray, /
) -> tuple[float, float] | None:
    """Enclose the minimum Euclidean distance to the proven closed strip.

    Circle extrema occur at the radial stationary direction or an endpoint.
    Interval cross products decide whether the stationary direction lies on the
    oriented profile arc.  When that decision is ambiguous, the full-circle
    minimum remains a valid lower bound and the endpoints supply an upper bound.
    """
    try:
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            return None
        patch = strip.patch
        delta = tuple(
            Fraction.from_float(float(x)) - y
            for x, y in zip(p, patch.center_exact, strict=True)
        )
        x, y, z = (
            _exact(_dot(delta, v))
            for v in (patch.first_axis_exact, patch.second_axis_exact, patch.axis_exact)
        )
        rho = _sqrt(_add(_square(x), _square(y)))
        major = cast(Fraction, patch.major_radius_exact)
        u = _add(rho, (-major, -major))
        r = cast(Fraction, patch.minor_radius_exact)
        length = _sqrt(_add(_square(u), _square(z)))
        free = _square(_add(length, (-r, -r)))
        roots = (strip.lower, strip.upper)
        endpoint = []
        for root in roots:
            c, s = _exact(root.cosine_exact), _exact(root.sine_exact)
            endpoint.append(
                _add(
                    _square(_add(u, _neg(_mul((r, r), c)))),
                    _square(_add(z, _neg(_mul((r, r), s)))),
                )
            )
        end_min = (min(t[0] for t in endpoint), min(t[1] for t in endpoint))
        lo, hi = roots
        # Cross(low, v), Cross(v, high).  Their signs select a short
        # counterclockwise arc; for a long arc the complement is the short arc.
        c1, s1 = _exact(lo.cosine_exact), _exact(lo.sine_exact)
        c2, s2 = _exact(hi.cosine_exact), _exact(hi.sine_exact)
        left = _add(_mul(c1, z), _neg(_mul(s1, u)))
        right = _add(_mul(u, s2), _neg(_mul(z, c2)))
        period_lo, period_hi = period_scalar_bounds(
            period_offset(Fraction(0), Fraction(1))
        )
        span_lo = Fraction.from_float(hi.parameter_lower) - Fraction.from_float(
            lo.parameter_upper
        )
        span_hi = Fraction.from_float(hi.parameter_upper) - Fraction.from_float(
            lo.parameter_lower
        )
        half_lo = Fraction.from_float(float(period_lo)) / 2
        half_hi = Fraction.from_float(float(period_hi)) / 2
        inside = outside = False
        if span_hi < half_lo:
            inside = left[0] > 0 and right[0] > 0
            outside = left[1] < 0 or right[1] < 0
        elif span_lo > half_hi:
            inside = left[0] > 0 or right[0] > 0
            outside = left[1] < 0 and right[1] < 0
        if inside:
            squared = free
        elif outside:
            squared = end_min
        else:
            squared = (free[0], end_min[1])
        distance = _sqrt(squared)
        return fraction_down(distance[0]), fraction_up(distance[1])
    except (AttributeError, TypeError, ValueError, OverflowError, ZeroDivisionError):
        return None
