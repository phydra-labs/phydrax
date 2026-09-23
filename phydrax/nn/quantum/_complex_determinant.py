#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Stable differentiable mixtures of complex determinants."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._ferminet import _polynomial_determinant


def _scaled_complex_determinant(matrix: Array, /) -> tuple[Array, Array]:
    row_magnitude = jnp.max(jnp.abs(matrix), axis=-1)
    safe_row_magnitude = jnp.where(row_magnitude > 0.0, row_magnitude, 1.0)
    safe_row_magnitude = jax.lax.stop_gradient(safe_row_magnitude)
    row_scaled = matrix / safe_row_magnitude[:, None]
    column_magnitude = jnp.max(jnp.abs(row_scaled), axis=-2)
    safe_column_magnitude = jnp.where(
        column_magnitude > 0.0,
        column_magnitude,
        1.0,
    )
    safe_column_magnitude = jax.lax.stop_gradient(safe_column_magnitude)
    scaled = row_scaled / safe_column_magnitude[None, :]
    log_scale = jnp.sum(jnp.log(safe_row_magnitude)) + jnp.sum(
        jnp.log(safe_column_magnitude)
    )
    return _polynomial_determinant(scaled), log_scale


def _stable_complex_product_primal(value: Array, log_scale: Array, /) -> Array:
    magnitude = jnp.abs(value)
    nonzero = magnitude > 0.0
    safe_magnitude = jnp.where(nonzero, magnitude, 1.0)
    phase = value / safe_magnitude
    combined_log = jnp.where(
        nonzero,
        jnp.log(safe_magnitude) + log_scale,
        0.0,
    )
    result = phase * jnp.exp(combined_log)
    return jnp.where(nonzero, result, jnp.zeros_like(result))


def _apply_linear_stable_complex_product(value: Array, log_scale: Array, /) -> Array:
    def inverse_scale(argument):
        return _stable_complex_product_primal(argument, -log_scale)

    def solve(_inverse_scale, right_hand_side):
        return _stable_complex_product_primal(right_hand_side, log_scale)

    return jax.lax.custom_linear_solve(
        inverse_scale,
        value,
        solve,
        symmetric=True,
    )


def _stable_complex_bilinear_primal(
    left: Array,
    right: Array,
    log_scale: Array,
    /,
) -> Array:
    left_magnitude = jnp.abs(left)
    right_magnitude = jnp.abs(right)
    left_nonzero = left_magnitude > 0.0
    right_nonzero = right_magnitude > 0.0
    nonzero = left_nonzero & right_nonzero
    safe_left_magnitude = jnp.where(left_nonzero, left_magnitude, 1.0)
    safe_right_magnitude = jnp.where(right_nonzero, right_magnitude, 1.0)
    phase = (left / safe_left_magnitude) * (right / safe_right_magnitude)
    combined_log = jnp.where(
        nonzero,
        jnp.log(safe_left_magnitude) + jnp.log(safe_right_magnitude) + log_scale,
        0.0,
    )
    value = phase * jnp.exp(combined_log)
    return jnp.where(nonzero, value, jnp.zeros_like(value))


@jax.custom_jvp
def _zero_complex_multiplier_product(
    multiplier: Array,
    value: Array,
    log_scale: Array,
    /,
) -> Array:
    return jnp.zeros_like(value, dtype=jnp.result_type(multiplier, value))


@_zero_complex_multiplier_product.defjvp
def _zero_complex_multiplier_product_jvp(primals, tangents):
    _, value, log_scale = primals
    multiplier_tangent, _, _ = tangents
    primal = jnp.zeros_like(
        value,
        dtype=jnp.result_type(multiplier_tangent, value),
    )
    tangent = _apply_linear_stable_complex_bilinear(
        value,
        multiplier_tangent,
        log_scale,
    )
    return primal, tangent


def _apply_linear_stable_complex_bilinear(
    multiplier: Array,
    value: Array,
    log_scale: Array,
    /,
) -> Array:
    magnitude = jnp.abs(multiplier)
    nonzero = magnitude > 0.0
    safe_magnitude = jnp.where(nonzero, magnitude, 1.0)
    phase = multiplier / safe_magnitude
    nonzero_value = phase * _apply_linear_stable_complex_product(
        value,
        log_scale + jnp.log(safe_magnitude),
    )
    zero_value = _zero_complex_multiplier_product(
        jnp.where(nonzero, jnp.zeros_like(multiplier), multiplier),
        jnp.where(nonzero, jnp.zeros_like(value), value),
        log_scale,
    )
    return jnp.where(nonzero, nonzero_value, zero_value)


@jax.custom_jvp
def _stable_complex_bilinear(
    left: Array,
    right: Array,
    log_scale: Array,
    /,
) -> Array:
    return _stable_complex_bilinear_primal(left, right, log_scale)


@_stable_complex_bilinear.defjvp
def _stable_complex_bilinear_jvp(primals, tangents):
    left, right, log_scale = primals
    left_tangent, right_tangent, _ = tangents
    value = _stable_complex_bilinear(left, right, log_scale)
    tangent = _apply_linear_stable_complex_bilinear(
        right,
        left_tangent,
        log_scale,
    ) + _apply_linear_stable_complex_bilinear(
        left,
        right_tangent,
        log_scale,
    )
    return value, tangent


def complex_determinant_mixture(
    matrices: Array,
    coefficients: Array,
    /,
) -> tuple[Array, Array, Array]:
    scaled_determinants, log_scales = jax.vmap(_scaled_complex_determinant)(matrices)
    determinant_magnitudes = jnp.abs(scaled_determinants)
    coefficient_magnitudes = jnp.abs(coefficients)
    determinant_defined = jnp.isfinite(scaled_determinants) & jnp.isfinite(log_scales)
    coefficient_defined = jnp.isfinite(coefficients)
    active = (
        determinant_defined
        & coefficient_defined
        & (determinant_magnitudes > 0.0)
        & (coefficient_magnitudes > 0.0)
    )
    safe_determinant_magnitudes = jnp.where(
        determinant_magnitudes > 0.0,
        determinant_magnitudes,
        1.0,
    )
    safe_coefficient_magnitudes = jnp.where(
        coefficient_magnitudes > 0.0,
        coefficient_magnitudes,
        1.0,
    )
    active_logs = (
        jnp.log(safe_determinant_magnitudes)
        + jnp.log(safe_coefficient_magnitudes)
        + log_scales
    )
    any_active = jnp.any(active)
    any_defined = jnp.any(determinant_defined & coefficient_defined)
    fallback_shift = jnp.max(
        jnp.where(
            determinant_defined & coefficient_defined,
            log_scales,
            -jnp.inf,
        )
    )
    shift = jax.lax.stop_gradient(
        jnp.where(
            any_active,
            jnp.max(jnp.where(active, active_logs, -jnp.inf)),
            jnp.where(any_defined, fallback_shift, 0.0),
        )
    )
    safe_determinants = jnp.where(
        determinant_defined,
        scaled_determinants,
        0.0,
    )
    safe_coefficients = jnp.where(coefficient_defined, coefficients, 0.0)
    relative_log_scales = jnp.where(
        determinant_defined & coefficient_defined,
        log_scales - shift,
        0.0,
    )
    scaled_terms = _stable_complex_bilinear(
        safe_coefficients,
        safe_determinants,
        relative_log_scales,
    )
    scaled_sum = jnp.sum(scaled_terms)
    magnitude = jnp.abs(scaled_sum)
    nonzero = magnitude > 0.0
    safe_magnitude = jnp.where(nonzero, magnitude, 1.0)
    log_abs = jnp.where(nonzero, shift + jnp.log(safe_magnitude), -jnp.inf)
    phase = jnp.where(nonzero, scaled_sum / safe_magnitude, 1.0 + 0.0j)
    valid = (
        jnp.all(determinant_defined)
        & jnp.all(coefficient_defined)
        & nonzero
        & jnp.isfinite(log_abs)
        & jnp.isfinite(phase)
    )
    return log_abs, phase, valid


__all__ = ["complex_determinant_mixture"]
