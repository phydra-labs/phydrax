#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from functools import partial
from math import factorial
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._stencil import apply_gather_stencil, GatherStencil
from ._types import BoundsMode, InterpolationResult


def _safe_ratio(numerator: Array, denominator: Array, /) -> Array:
    nonzero = denominator != 0.0
    safe_denominator = jnp.where(nonzero, denominator, jnp.ones_like(denominator))
    return jnp.where(nonzero, numerator / safe_denominator, jnp.zeros_like(numerator))


def _basis_jet(
    parameter: Array,
    knots: Array,
    span: Array,
    degree: int,
    maximum_order: int,
    /,
) -> Array:
    """Evaluate one span-local B-spline basis jet."""
    table = jnp.zeros((degree + 1, degree + 1), dtype=knots.dtype)
    table = table.at[0, 0].set(1.0)
    left = jnp.zeros((degree + 1,), dtype=knots.dtype)
    right = jnp.zeros((degree + 1,), dtype=knots.dtype)

    for column in range(1, degree + 1):
        left = left.at[column].set(parameter - knots[span + 1 - column])
        right = right.at[column].set(knots[span + column] - parameter)
        saved = jnp.zeros((), dtype=knots.dtype)
        for row in range(column):
            denominator = right[row + 1] + left[column - row]
            table = table.at[column, row].set(denominator)
            temporary = _safe_ratio(table[row, column - 1], denominator)
            table = table.at[row, column].set(saved + right[row + 1] * temporary)
            saved = left[column - row] * temporary
        table = table.at[column, column].set(saved)

    derivatives = jnp.zeros(
        (maximum_order + 1, degree + 1),
        dtype=knots.dtype,
    )
    derivatives = derivatives.at[0].set(table[:, degree])

    for basis_index in range(degree + 1):
        coefficients = jnp.zeros((2, degree + 1), dtype=knots.dtype)
        coefficients = coefficients.at[0, 0].set(1.0)
        previous = 0
        current = 1
        for order in range(1, maximum_order + 1):
            coefficients = coefficients.at[current].set(0.0)
            value = jnp.zeros((), dtype=knots.dtype)
            shifted_index = basis_index - order
            reduced_degree = degree - order

            if basis_index >= order:
                coefficient = _safe_ratio(
                    coefficients[previous, 0],
                    table[reduced_degree + 1, shifted_index],
                )
                coefficients = coefficients.at[current, 0].set(coefficient)
                value = coefficient * table[shifted_index, reduced_degree]

            first = 1 if shifted_index >= -1 else -shifted_index
            last = (
                order - 1 if basis_index - 1 <= reduced_degree else degree - basis_index
            )
            for column in range(first, last + 1):
                coefficient = _safe_ratio(
                    coefficients[previous, column] - coefficients[previous, column - 1],
                    table[reduced_degree + 1, shifted_index + column],
                )
                coefficients = coefficients.at[current, column].set(coefficient)
                value = (
                    value + coefficient * table[shifted_index + column, reduced_degree]
                )

            if basis_index <= reduced_degree:
                coefficient = _safe_ratio(
                    -coefficients[previous, order - 1],
                    table[reduced_degree + 1, basis_index],
                )
                coefficients = coefficients.at[current, order].set(coefficient)
                value = value + coefficient * table[basis_index, reduced_degree]

            derivatives = derivatives.at[order, basis_index].set(value)
            previous, current = current, previous

    for order in range(1, maximum_order + 1):
        scale = factorial(degree) // factorial(degree - order)
        derivatives = derivatives.at[order].multiply(scale)
    return derivatives


def _bspline_weights_raw(
    knots: Array,
    parameters: Array,
    spans: Array,
    degree: int,
    derivative_order: int,
) -> Array:
    flat_parameters = parameters.reshape((-1,))
    flat_spans = spans.reshape((-1,))
    weights = jax.vmap(
        lambda parameter, span: _basis_jet(
            parameter,
            knots,
            span,
            degree,
            derivative_order,
        )[derivative_order]
    )(flat_parameters, flat_spans)
    return weights.reshape(parameters.shape + (degree + 1,))


@partial(jax.custom_jvp, nondiff_argnums=(3, 4))
def _bspline_weights(
    knots: Array,
    parameters: Array,
    spans: Array,
    degree: int,
    derivative_order: int,
) -> Array:
    return _bspline_weights_raw(
        knots,
        parameters,
        spans,
        degree,
        derivative_order,
    )


@_bspline_weights.defjvp
def _bspline_weights_jvp(
    degree: int,
    derivative_order: int,
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array, Array, Array],
) -> tuple[Array, Array]:
    knots, parameters, spans = primals
    knot_tangent, parameter_tangent, _span_tangent = tangents
    weights = _bspline_weights_raw(
        knots,
        parameters,
        spans,
        degree,
        derivative_order,
    )
    if derivative_order == degree:
        derivative_weights = jnp.zeros_like(weights)
    else:
        derivative_weights = _bspline_weights(
            knots,
            parameters,
            spans,
            degree,
            derivative_order + 1,
        )
    _, knot_component = jax.jvp(
        lambda knot_values: _bspline_weights_raw(
            knot_values,
            parameters,
            spans,
            degree,
            derivative_order,
        ),
        (knots,),
        (knot_tangent,),
    )
    tangent = derivative_weights * parameter_tangent[..., None] + knot_component
    return weights, tangent


def bspline_stencil(
    knots: ArrayLike,
    query: ArrayLike,
    /,
    *,
    degree: int,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
    case_shape: tuple[int, ...] = (),
) -> GatherStencil:
    """Build a span-local B-spline map from control coefficients to queries."""
    if isinstance(degree, bool) or not isinstance(degree, Integral):
        raise TypeError("B-spline degree must be an integer.")
    if isinstance(derivative_order, bool) or not isinstance(derivative_order, Integral):
        raise TypeError("B-spline derivative_order must be an integer.")
    degree_ = int(degree)
    order = int(derivative_order)
    if degree_ < 0:
        raise ValueError("B-spline degree must be non-negative.")
    if order < 0 or order > degree_:
        raise ValueError(
            "B-spline derivative_order must lie between zero and the degree."
        )
    if bounds not in ("clip", "error", "extrapolate", "fill"):
        raise ValueError("bounds must be 'clip', 'error', 'extrapolate', or 'fill'.")

    knots_raw = jnp.asarray(knots)
    query_raw = jnp.asarray(query)
    if jnp.issubdtype(knots_raw.dtype, jnp.complexfloating) or jnp.issubdtype(
        query_raw.dtype,
        jnp.complexfloating,
    ):
        raise TypeError("B-spline coordinates must be real-valued.")
    dtype = jnp.result_type(knots_raw, query_raw, float)
    knots_ = knots_raw.astype(dtype)
    query_ = query_raw.astype(dtype)
    if knots_.ndim != 1:
        raise ValueError("B-spline knots must be a rank-one array.")

    control_count = int(knots_.shape[0]) - degree_ - 1
    if control_count <= degree_:
        raise ValueError(
            "B-spline knots must define at least degree + 1 control coefficients."
        )
    if int(query_.size) == 0:
        raise ValueError("B-spline queries must be non-empty.")

    cases = tuple(int(size) for size in case_shape)
    if any(size <= 0 for size in cases):
        raise ValueError("B-spline case dimensions must be positive.")
    if tuple(int(size) for size in query_.shape[: len(cases)]) != cases:
        raise ValueError(
            f"B-spline queries must begin with case_shape {cases}; got {query_.shape}."
        )

    knots_ = eqx.error_if(
        knots_,
        jnp.any(~jnp.isfinite(knots_)) | jnp.any(jnp.diff(knots_) < 0.0),
        "B-spline knots must be finite and nondecreasing.",
    )
    lower = knots_[degree_]
    upper = knots_[control_count]
    query_ = eqx.error_if(
        query_,
        ~(upper > lower),
        "B-spline knots must define a nonempty active parameter interval.",
    )
    query_ = eqx.error_if(
        query_,
        jnp.any(~jnp.isfinite(query_)),
        "B-spline queries must be finite.",
    )
    outside = (query_ < lower) | (query_ > upper)
    if bounds == "error":
        query_ = eqx.error_if(
            query_,
            jnp.any(outside),
            "B-spline query is outside the active parameter interval.",
        )
    query_eval = (
        jnp.where(
            query_ < lower,
            lower,
            jnp.where(query_ > upper, upper, query_),
        )
        if bounds in ("clip", "fill")
        else query_
    )
    support = ~outside if bounds == "fill" else jnp.ones(query_.shape, dtype=bool)

    spans = jnp.searchsorted(knots_, query_eval, side="right") - 1
    spans = jax.lax.stop_gradient(
        jnp.clip(spans, degree_, control_count - 1).astype(jnp.int32)
    )
    flat_spans = spans.reshape((-1,))
    weights = _bspline_weights(
        knots_,
        query_eval,
        spans,
        degree_,
        order,
    ).reshape((-1, degree_ + 1))
    offsets = jnp.arange(degree_ + 1, dtype=jnp.int32) - degree_
    indices = flat_spans[:, None] + offsets[None, :]
    route_shape = query_.shape + (degree_ + 1,)
    return GatherStencil(
        indices=indices.reshape(route_shape),
        weights=weights.reshape(route_shape),
        source_size=control_count,
        support=support,
        case_shape=cases,
    )


def bspline_evaluate(
    knots: ArrayLike,
    coefficients: ArrayLike,
    query: ArrayLike,
    /,
    *,
    degree: int,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
    case_shape: tuple[int, ...] = (),
) -> InterpolationResult:
    """Evaluate B-spline coefficients with an analytic query derivative rule."""
    stencil = bspline_stencil(
        knots,
        query,
        degree=degree,
        derivative_order=derivative_order,
        bounds=bounds,
        case_shape=case_shape,
    )
    return apply_gather_stencil(coefficients, stencil)


def bspline_batched_evaluate(
    knots: ArrayLike,
    coefficients: ArrayLike,
    query: ArrayLike,
    /,
    *,
    degree: int,
    derivative_order: int = 0,
    bounds: BoundsMode = "error",
) -> InterpolationResult:
    """Evaluate homogeneous knot rows aligned with the second case axis."""
    knots_ = jnp.asarray(knots)
    coefficients_ = jnp.asarray(coefficients)
    query_ = jnp.asarray(query)
    if knots_.ndim != 2 or knots_.shape[0] == 0:
        raise ValueError(
            "Batched B-spline knots must have shape (num_grids, knot_count)."
        )
    if coefficients_.ndim < 3:
        raise ValueError(
            "Batched B-spline coefficients must begin with "
            "(output_count, num_grids, coefficient_count)."
        )
    if query_.ndim < 2:
        raise ValueError(
            "Batched B-spline queries must begin with (output_count, num_grids)."
        )
    output_count = int(coefficients_.shape[0])
    num_grids = int(knots_.shape[0])
    control_count = int(knots_.shape[1]) - int(degree) - 1
    if (
        output_count == 0
        or int(coefficients_.shape[1]) != num_grids
        or int(coefficients_.shape[2]) != control_count
    ):
        raise ValueError("Batched B-spline coefficient axes do not match the knot bank.")
    if query_.shape[:2] != coefficients_.shape[:2]:
        raise ValueError(
            "Batched B-spline query case axes must match the coefficient axes."
        )

    grid_coefficients = jnp.moveaxis(coefficients_, 1, 0)
    grid_queries = jnp.moveaxis(query_, 1, 0)

    def evaluate_grid(
        grid_knots: Array,
        grid_coefficients_: Array,
        grid_query: Array,
    ) -> InterpolationResult:
        return bspline_evaluate(
            grid_knots,
            grid_coefficients_,
            grid_query,
            degree=degree,
            derivative_order=derivative_order,
            bounds=bounds,
            case_shape=(output_count,),
        )

    result = jax.vmap(evaluate_grid)(knots_, grid_coefficients, grid_queries)
    return InterpolationResult(
        jnp.moveaxis(result.values, 0, 1),
        jnp.moveaxis(result.support, 0, 1),
    )


__all__ = ["bspline_batched_evaluate", "bspline_evaluate", "bspline_stencil"]
