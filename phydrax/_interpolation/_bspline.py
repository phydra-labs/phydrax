#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import factorial
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._stencil import GatherStencil
from ._types import BoundsMode


def _safe_ratio(numerator: Array, denominator: Array, /) -> Array:
    nonzero = denominator != 0.0
    safe_denominator = jnp.where(nonzero, denominator, jnp.ones_like(denominator))
    return jnp.where(nonzero, numerator / safe_denominator, jnp.zeros_like(numerator))


def _basis_derivative(
    parameter: Array,
    knots: Array,
    span: Array,
    degree: int,
    derivative_order: int,
    /,
) -> Array:
    """Evaluate one span-local B-spline basis derivative."""
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
        (derivative_order + 1, degree + 1),
        dtype=knots.dtype,
    )
    derivatives = derivatives.at[0].set(table[:, degree])

    for basis_index in range(degree + 1):
        coefficients = jnp.zeros((2, degree + 1), dtype=knots.dtype)
        coefficients = coefficients.at[0, 0].set(1.0)
        previous = 0
        current = 1
        for order in range(1, derivative_order + 1):
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

    for order in range(1, derivative_order + 1):
        scale = factorial(degree) // factorial(degree - order)
        derivatives = derivatives.at[order].multiply(scale)
    return derivatives[derivative_order]


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
    spans = jnp.clip(spans, degree_, control_count - 1).astype(jnp.int32)
    flat_parameters = query_eval.reshape((-1,))
    flat_spans = spans.reshape((-1,))
    weights = jax.vmap(
        lambda parameter, span: _basis_derivative(
            parameter,
            knots_,
            span,
            degree_,
            order,
        )
    )(flat_parameters, flat_spans)
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


__all__ = ["bspline_stencil"]
