#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array

from .._interval_enclosure import _device_round_out


type _Bounds = tuple[Array, Array]
type _SurfaceBounds = tuple[Array, Array, Array, Array]


def _down(value: Array, /) -> Array:
    return _device_round_out(value, value)[0]


def _up(value: Array, /) -> Array:
    return _device_round_out(value, value)[1]


def _multiply(first: _Bounds, second: _Bounds, /) -> _Bounds:
    products = jnp.stack(
        (
            first[0] * second[0],
            first[0] * second[1],
            first[1] * second[0],
            first[1] * second[1],
        )
    )
    return _down(jnp.min(products, axis=0)), _up(jnp.max(products, axis=0))


def _blend(first: _Bounds, second: _Bounds, alpha: _Bounds, /) -> _Bounds:
    complement = (
        jnp.maximum(0.0, _down(1.0 - alpha[1])),
        jnp.minimum(1.0, _up(1.0 - alpha[0])),
    )
    left, right = _multiply(first, complement), _multiply(second, alpha)
    # Each source blend is convex. Intersecting with its input hull retains
    # constant coefficients without accumulating artificial interval width.
    return (
        jnp.maximum(_down(left[0] + right[0]), jnp.minimum(first[0], second[0])),
        jnp.minimum(_up(left[1] + right[1]), jnp.maximum(first[1], second[1])),
    )


def _blossom(
    controls: _Bounds,
    knots: Array,
    span: Array,
    first: Array,
    last: Array,
    degree: int,
    coefficient: int,
    /,
) -> _Bounds:
    lower, upper = controls
    for level in range(1, degree + 1):
        parameter = first if level <= degree - coefficient else last
        indices = jnp.arange(level, degree + 1, dtype=jnp.int32)
        knot_lower = knots[span - degree + indices]
        knot_upper = knots[span + 1 + indices - level]
        numerator = parameter - knot_lower
        denominator = knot_upper - knot_lower
        denominator_lower, denominator_upper = _down(denominator), _up(denominator)
        safe_lower = jnp.where(denominator_lower > 0.0, denominator_lower, 1.0)
        # Positive source spans make the source ratios lie in [0, 1]. A
        # rounded denominator touching zero can use that full convex range.
        alpha_lower = jnp.maximum(0.0, _down(_down(numerator) / denominator_upper))
        alpha_upper = jnp.minimum(
            1.0,
            jnp.where(
                denominator_lower > 0.0,
                _up(_up(numerator) / safe_lower),
                1.0,
            ),
        )
        expansion = (1,) * (lower.ndim - 1)
        alpha = (
            alpha_lower.reshape((-1, *expansion)),
            alpha_upper.reshape((-1, *expansion)),
        )
        blended = _blend(
            (lower[level - 1 : -1], upper[level - 1 : -1]),
            (lower[level:], upper[level:]),
            alpha,
        )
        lower = lower.at[level:].set(blended[0])
        upper = upper.at[level:].set(blended[1])
    return lower[-1], upper[-1]


def _restrict_axis(
    controls: _Bounds,
    knots: Array,
    span: Array,
    first: Array,
    last: Array,
    degree: int,
    axis: int,
    /,
) -> _Bounds:
    front = (
        jnp.moveaxis(controls[0], axis, 0),
        jnp.moveaxis(controls[1], axis, 0),
    )
    coefficients = tuple(
        _blossom(front, knots, span, first, last, degree, coefficient)
        for coefficient in range(degree + 1)
    )
    return (
        jnp.moveaxis(jnp.stack(tuple(value[0] for value in coefficients)), 0, axis),
        jnp.moveaxis(jnp.stack(tuple(value[1] for value in coefficients)), 0, axis),
    )


def _denominator_margin(weights: _Bounds, degrees: tuple[int, int], /) -> Array:
    """Bound the floating basis/weight reduction error on one source span."""
    # Degree p forms 2p endpoint distances and p(p+1)/2 recurrence cells,
    # each with five arithmetic operations. Tensor products and weight products
    # use 2L multiplications, followed by L-1 additions. eps conservatively
    # bounds unit roundoff in the positive basis reduction's gamma_k estimate.
    local_count = (degrees[0] + 1) * (degrees[1] + 1)
    count = (
        sum(2 * degree + 5 * degree * (degree + 1) // 2 for degree in degrees)
        + 2 * local_count
        + local_count
        - 1
    )
    dtype = weights[0].dtype
    product = _up(
        jnp.asarray(count, dtype=dtype) * jnp.asarray(jnp.finfo(dtype).eps, dtype=dtype)
    )
    denominator = _down(1.0 - product)
    safe = (product < 1.0) & (denominator > 0.0)
    gamma = jnp.where(safe, _up(product / jnp.where(safe, denominator, 1.0)), jnp.inf)
    magnitude = jnp.max(jnp.maximum(jnp.abs(weights[0]), jnp.abs(weights[1])))
    return _up(gamma * magnitude)


def _span_bounds(
    controls: Array,
    weights: Array,
    scale: Array,
    u_knots: Array,
    v_knots: Array,
    lower: Array,
    upper: Array,
    u_span: Array,
    v_span: Array,
    u_degree: int,
    v_degree: int,
    /,
) -> _SurfaceBounds:
    start = (
        jnp.asarray(u_span - u_degree, dtype=jnp.int32),
        jnp.asarray(v_span - v_degree, dtype=jnp.int32),
    )
    shape = (u_degree + 1, v_degree + 1)
    local_controls = jax.lax.dynamic_slice(
        controls, (*start, jnp.asarray(0, dtype=jnp.int32)), (*shape, 3)
    )
    local_weights = jax.lax.dynamic_slice(weights, start, shape) / scale
    weight_bounds = (_down(local_weights), _up(local_weights))
    product = _multiply(
        (local_controls, local_controls),
        (weight_bounds[0][..., None], weight_bounds[1][..., None]),
    )
    homogeneous = (
        jnp.concatenate((product[0], weight_bounds[0][..., None]), axis=-1),
        jnp.concatenate((product[1], weight_bounds[1][..., None]), axis=-1),
    )
    for axis, (knots, span, degree) in enumerate(
        ((u_knots, u_span, u_degree), (v_knots, v_span, v_degree))
    ):
        first = jnp.maximum(lower[axis], knots[span])
        last = jnp.minimum(upper[axis], knots[span + 1])
        homogeneous = _restrict_axis(homogeneous, knots, span, first, last, degree, axis)
    denominator_lower = jnp.min(homogeneous[0][..., -1])
    denominator_upper = jnp.max(homogeneous[1][..., -1])
    quotients = jnp.stack(
        tuple(
            numerator[..., :3] / denominator[..., -1, None]
            for numerator in homogeneous
            for denominator in homogeneous
        )
    )
    valid = (denominator_lower > 0.0) & jnp.all(jnp.isfinite(quotients))
    margin = _denominator_margin(weight_bounds, (u_degree, v_degree))
    return (
        jnp.where(valid, _down(jnp.min(quotients, axis=(0, 1, 2))), -jnp.inf),
        jnp.where(valid, _up(jnp.max(quotients, axis=(0, 1, 2))), jnp.inf),
        _down(denominator_lower - margin),
        _up(denominator_upper + margin),
    )


def _surface_bounds_one(
    controls: Array,
    weights_: Array,
    u_knots_: Array,
    v_knots_: Array,
    lower_: Array,
    upper_: Array,
    u_degree: int,
    v_degree: int,
    /,
) -> _SurfaceBounds:
    knots = (u_knots_, v_knots_)
    degrees = (u_degree, v_degree)
    first_spans: list[Array] = []
    last_spans: list[Array] = []
    valid = jnp.all(jnp.isfinite(lower_)) & jnp.all(jnp.isfinite(upper_))
    valid = valid & jnp.all(lower_ <= upper_)
    for axis, (axis_knots, degree) in enumerate(zip(knots, degrees, strict=True)):
        count = controls.shape[axis]
        lower_ = lower_.at[axis].set(
            jnp.clip(lower_[axis], axis_knots[degree], axis_knots[count])
        )
        upper_ = upper_.at[axis].set(
            jnp.clip(upper_[axis], axis_knots[degree], axis_knots[count])
        )
        valid = (
            valid
            & jnp.all(jnp.isfinite(axis_knots))
            & jnp.all(jnp.diff(axis_knots) >= 0.0)
            & (axis_knots[count] > axis_knots[degree])
            & (
                (upper_[axis] < axis_knots[count])
                | (axis_knots[count - 1] < axis_knots[count])
            )
        )
        first_spans.append(
            jnp.clip(
                jnp.searchsorted(axis_knots, lower_[axis], side="right") - 1,
                degree,
                count - 1,
            )
        )
        last_spans.append(
            jnp.clip(
                jnp.searchsorted(axis_knots, upper_[axis], side="right") - 1,
                degree,
                count - 1,
            )
        )
    scale = jnp.max(jnp.abs(weights_))
    valid = valid & jnp.isfinite(scale) & (scale > 0.0)
    initial = (
        jnp.full((3,), jnp.inf, dtype=controls.dtype),
        jnp.full((3,), -jnp.inf, dtype=controls.dtype),
        jnp.asarray(jnp.inf, dtype=controls.dtype),
        jnp.asarray(-jnp.inf, dtype=controls.dtype),
    )

    def u_body(u_span: Array, accumulated: _SurfaceBounds) -> _SurfaceBounds:
        def v_body(v_span: Array, current: _SurfaceBounds) -> _SurfaceBounds:
            positive = (u_knots_[u_span + 1] > u_knots_[u_span]) & (
                v_knots_[v_span + 1] > v_knots_[v_span]
            )

            def enclose(previous: _SurfaceBounds) -> _SurfaceBounds:
                piece = _span_bounds(
                    controls,
                    weights_,
                    scale,
                    u_knots_,
                    v_knots_,
                    lower_,
                    upper_,
                    u_span,
                    v_span,
                    u_degree,
                    v_degree,
                )
                return (
                    jnp.minimum(previous[0], piece[0]),
                    jnp.maximum(previous[1], piece[1]),
                    jnp.minimum(previous[2], piece[2]),
                    jnp.maximum(previous[3], piece[3]),
                )

            return jax.lax.cond(positive, enclose, lambda value: value, current)

        return jax.lax.fori_loop(first_spans[1], last_spans[1] + 1, v_body, accumulated)

    result = jax.lax.fori_loop(first_spans[0], last_spans[0] + 1, u_body, initial)
    valid = valid & jnp.all(jnp.isfinite(result[0])) & jnp.all(jnp.isfinite(result[1]))
    return (
        jnp.where(valid, result[0], -jnp.inf),
        jnp.where(valid, result[1], jnp.inf),
        jnp.where(valid, result[2], -jnp.inf),
        jnp.where(valid, result[3], jnp.inf),
    )


def device_bspline_surface_bounds(
    control_points: Array,
    weights: Array,
    u_knots: Array,
    v_knots: Array,
    lower: Array,
    upper: Array,
    *,
    u_degree: int,
    v_degree: int,
) -> _SurfaceBounds:
    """Enclose rational spline source boxes with outward homogeneous blossoms.

    Query endpoints have shape ``(..., 2)`` and are clipped to the active
    source domain, matching surface evaluation. Each touched positive span
    is restricted to the queried subrectangle, including singleton boxes.
    Returned denominator bounds use normalized weights and include a derived
    floating basis-reduction envelope. All numeric inputs are fixed evidence,
    so differentiation does not pass through rounding or span routing.
    Invalid source premises return unbounded evidence without host callbacks.
    """
    controls, weights_, u_knots_, v_knots_, lower_, upper_ = (
        jax.lax.stop_gradient(value)
        for value in (control_points, weights, u_knots, v_knots, lower, upper)
    )
    if (
        controls.ndim != 3
        or controls.shape[-1] != 3
        or weights_.shape != controls.shape[:2]
        or lower_.ndim == 0
        or lower_.shape[-1] != 2
        or upper_.shape != lower_.shape
        or u_degree < 0
        or v_degree < 0
        or controls.shape[0] <= u_degree
        or controls.shape[1] <= v_degree
        or u_knots_.shape != (controls.shape[0] + u_degree + 1,)
        or v_knots_.shape != (controls.shape[1] + v_degree + 1,)
    ):
        raise ValueError(
            "device_bspline_surface_bounds.inputs must describe a tensor spline and 2D boxes; "
            f"got controls={controls.shape}, weights={weights_.shape}, "
            f"degrees={(u_degree, v_degree)}, lower={lower_.shape}, upper={upper_.shape}."
        )
    if lower_.ndim == 1:
        return _surface_bounds_one(
            controls, weights_, u_knots_, v_knots_, lower_, upper_, u_degree, v_degree
        )

    def enclose(pair: tuple[Array, Array]) -> _SurfaceBounds:
        return _surface_bounds_one(
            controls, weights_, u_knots_, v_knots_, pair[0], pair[1], u_degree, v_degree
        )

    result = jax.lax.map(enclose, (lower_.reshape((-1, 2)), upper_.reshape((-1, 2))))
    query_shape = lower_.shape[:-1]
    return (
        result[0].reshape((*query_shape, 3)),
        result[1].reshape((*query_shape, 3)),
        result[2].reshape(query_shape),
        result[3].reshape(query_shape),
    )
