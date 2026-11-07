#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Certified source-circle fan charts for curved triangular material cells.

This owner proves the whole-cell map, not the caller's source occurrence,
coedge correspondence, trim containment, or adjacency. Those must be proved
separately on the current source before composing the chart into a mesh.
"""
from __future__ import annotations

from fractions import Fraction

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..discretization._exact_algebraic_geometry import coefficient_interval, ExactScalar, RadicalElement
from ._interval_enclosure import interval_add, interval_cos, interval_multiply, interval_sin
from ._rational import fraction_down, fraction_up
from .brep._boolean_core import chart_add, chart_scale, ChartScalar
from .brep._patches import _validate_source_shadows, CircleCurve
from .brep._source_embedding import _sheet_sign
from .brep._trim_helpers import period_scalar_bounds, PeriodOffset


def _cross(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> tuple[ExactScalar, ...]:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: tuple[ExactScalar, ...], b: tuple[ExactScalar, ...], /) -> ExactScalar:
    return sum((x * y for x, y in zip(a, b, strict=True)), Fraction(0))


def _coefficient(value: ExactScalar, /) -> tuple[np.ndarray, np.ndarray]:
    lower, upper = coefficient_interval(value, bits=128)
    return np.asarray(fraction_down(lower)), np.asarray(fraction_up(upper))


def _star_bound(
    curve: CircleCurve, apex: tuple[ExactScalar, ExactScalar, ExactScalar],
    first: ChartScalar, last: ChartScalar, /
) -> tuple[int, float]:
    """Enclose n·((C-A)×C') on every exact phase subinterval."""
    a, b, center, r = curve.first_axis_exact, curve.second_axis_exact, curve.center_exact, curve.radius_exact
    normal = _cross(a, b)
    if not any(normal) or _dot(tuple(center[i] - apex[i] for i in range(3)), normal) != 0:
        raise ValueError("The fan apex and source circle must span one exact plane.")
    delta = chart_add(last, chart_scale(first, Fraction(-1)))
    orientation = _sheet_sign(delta)
    if orientation not in (-1, 1):
        raise ValueError("The exact source arc must have nonzero directed span.")
    absolute = chart_scale(delta, Fraction(orientation))
    if _sheet_sign(chart_add(absolute, PeriodOffset(Fraction(0), Fraction(-1)))) != -1:
        raise ValueError("A source fan arc must be strictly shorter than one native turn.")
    displacement = tuple(center[i] - apex[i] for i in range(3))
    constant = r * r * _dot(normal, normal)
    cosine = r * _dot(_cross(displacement, b), normal)
    sine = -r * _dot(_cross(displacement, a), normal)
    intervals = (_coefficient(constant), _coefficient(cosine), _coefficient(sine))
    pieces: list[tuple[Fraction, Fraction, int]] = [(Fraction(0), Fraction(1), 0)]
    signs: set[int] = set()
    minimum = float("inf")
    while pieces:
        lower, upper, depth = pieces.pop()
        p = chart_add(first, chart_scale(delta, lower))
        q = chart_add(first, chart_scale(delta, upper))
        p_bounds, q_bounds = period_scalar_bounds(p), period_scalar_bounds(q)
        phase = (np.minimum(p_bounds[0], q_bounds[0]), np.maximum(p_bounds[1], q_bounds[1]))
        enclosure = interval_add(interval_add(intervals[0], interval_multiply(intervals[1], interval_cos(phase))), interval_multiply(intervals[2], interval_sin(phase)))
        lo, hi = float(enclosure[0]), float(enclosure[1])
        if lo > 0:
            signs.add(1)
            minimum = min(minimum, lo)
        elif hi < 0:
            signs.add(-1)
            minimum = min(minimum, -hi)
        else:
            if depth == 16:
                raise ValueError("The source arc has no proved strict fan Jacobian.")
            middle = (lower + upper) / 2
            pieces.extend(((lower, middle, depth + 1), (middle, upper, depth + 1)))
        if len(signs) > 1:
            raise ValueError("The source arc reverses its fan orientation.")
    return next(iter(signs)) * orientation, minimum


class CircleFanMaterialMap(StrictModule, NonTrainableState):
    """Map a reference triangle to an exact circle arc and its planar fan.

    Reference vertex 0 is the apex; vertices 1 and 2 are the directed arc
    endpoints.  The apex has a continuous value but generally no single
    derivative; the explicit Jacobian is defined on the open triangle and on
    its non-apex edges. The proof supplies a strict Jacobian there and a
    one-to-one closed-cell map.
    """

    curve: CircleCurve
    apex: Array
    first: Array
    span: Array
    first_exact: ChartScalar = eqx.field(static=True)
    last_exact: ChartScalar = eqx.field(static=True)
    orientation: int = eqx.field(static=True)
    star_lower: float = eqx.field(static=True)

    def __init__(
        self, curve: CircleCurve, apex: tuple[ExactScalar, ExactScalar, ExactScalar],
        first: ChartScalar, last: ChartScalar, /
    ) -> None:
        if not isinstance(curve, CircleCurve) or curve.ambient_dimension != 3:
            raise TypeError("A fan chart requires a three-dimensional source CircleCurve.")
        if (not isinstance(apex, tuple) or len(apex) != 3
                or any(not isinstance(value, (Fraction, RadicalElement)) for value in apex)
                or any(
                    not isinstance(value, Fraction)
                    and not (
                        isinstance(value, PeriodOffset)
                        and isinstance(value.rational, Fraction)
                        and isinstance(value.turns, Fraction)
                    )
                    for value in (first, last)
                )):
            raise TypeError("A fan chart requires exact apex and source parameter expressions.")
        _validate_source_shadows(curve, set())
        orientation, lower = _star_bound(curve, apex, first, last)
        first_bounds, last_bounds = period_scalar_bounds(first), period_scalar_bounds(last)
        first_float = (float(first_bounds[0]) + float(first_bounds[1])) / 2
        last_float = (float(last_bounds[0]) + float(last_bounds[1])) / 2
        if not np.isfinite(first_float) or not np.isfinite(last_float - first_float):
            raise ValueError("The exact source arc has no finite executable parameter shadow.")
        self.curve = curve
        self.apex = jnp.asarray(tuple(float(value) for value in apex))
        self.first = jnp.asarray(first_float)
        self.span = jnp.asarray(last_float - first_float)
        self.first_exact = first
        self.last_exact = last
        self.orientation = orientation
        self.star_lower = lower

    def evaluate_points(self, reference_points: ArrayLike, /) -> Array:
        return self._evaluate(reference_points, derivative=False)[0]

    def evaluate(self, reference_points: ArrayLike, /) -> tuple[Array, Array]:
        points, jacobian = self._evaluate(reference_points, derivative=True)
        if jacobian is None:
            raise RuntimeError("Circle fan derivative evaluation did not produce a Jacobian.")
        return points, jacobian

    def _evaluate(self, reference_points: ArrayLike, /, *, derivative: bool) -> tuple[Array, Array | None]:
        points = jnp.asarray(reference_points)
        x, y = points[..., 0], points[..., 1]
        q = x + y
        safe_q = jnp.where(q == 0, 1, q)
        t = y / safe_q
        angle = self.first + self.span * t
        shape = (1,) * (points.ndim - 1) + (3,)
        first_axis = self.curve.first_axis.reshape(shape)
        second_axis = self.curve.second_axis.reshape(shape)
        center = self.curve.center.reshape(shape)
        apex = self.apex.reshape(shape)
        cosine, sine = jnp.cos(angle)[..., None], jnp.sin(angle)[..., None]
        arc = center + self.curve.radius * (cosine * first_axis + sine * second_axis)
        radial = arc - apex
        physical = apex + q[..., None] * radial
        if not derivative:
            return physical, None
        tangent = self.span * self.curve.radius * (
            -sine * first_axis + cosine * second_axis
        )
        dx = radial - t[..., None] * tangent
        dy = radial + (1 - t)[..., None] * tangent
        jacobian = eqx.error_if(
            jnp.stack((dx, dy), axis=-1),
            jnp.any(q == 0),
            "A circle fan's apex has no single reference Jacobian.",
        )
        return physical, jacobian
