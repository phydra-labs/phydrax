#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Real Carlson symmetric elliptic integrals."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike

from ._dtype import promote_real


def _steps(dtype: jnp.dtype) -> int:
    return 7 if dtype == jnp.float32 else 11


def _scale(values: tuple[Array, ...]) -> tuple[Array, tuple[Array, ...]]:
    scale = jax.lax.stop_gradient(jnp.max(jnp.stack(values), axis=0))
    safe_scale = jnp.where(scale > 0.0, scale, jnp.ones_like(scale))
    return safe_scale, tuple(value / safe_scale for value in values)


def _rf_positive(x: Array, y: Array, z: Array) -> Array:
    def body(_: int, state: tuple[Array, Array, Array]) -> tuple[Array, Array, Array]:
        xn, yn, zn = state
        sqrt_x = jnp.sqrt(xn)
        sqrt_y = jnp.sqrt(yn)
        sqrt_z = jnp.sqrt(zn)
        lam = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z
        return 0.25 * (xn + lam), 0.25 * (yn + lam), 0.25 * (zn + lam)

    xn, yn, zn = jax.lax.fori_loop(0, _steps(x.dtype), body, (x, y, z))
    mean = (xn + yn + zn) / 3.0
    dx = (mean - xn) / mean
    dy = (mean - yn) / mean
    dz = (mean - zn) / mean
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    correction = 1.0 + ((e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0)
    return correction / jnp.sqrt(mean)


def _rc_positive(x: Array, y: Array) -> Array:
    def body(_: int, state: tuple[Array, Array]) -> tuple[Array, Array]:
        xn, yn = state
        lam = 2.0 * jnp.sqrt(xn) * jnp.sqrt(yn) + yn
        return 0.25 * (xn + lam), 0.25 * (yn + lam)

    xn, yn = jax.lax.fori_loop(0, _steps(x.dtype), body, (x, y))
    mean = (xn + 2.0 * yn) / 3.0
    s = (yn - mean) / mean
    s2 = s * s
    correction = 1.0 + s2 * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * (9.0 / 22.0))))
    return correction / jnp.sqrt(mean)


def _rd_positive(x: Array, y: Array, z: Array) -> Array:
    def body(
        _: int, state: tuple[Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array]:
        xn, yn, zn, total, factor = state
        sqrt_x = jnp.sqrt(xn)
        sqrt_y = jnp.sqrt(yn)
        sqrt_z = jnp.sqrt(zn)
        lam = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z
        total = total + factor / (sqrt_z * (zn + lam))
        return (
            0.25 * (xn + lam),
            0.25 * (yn + lam),
            0.25 * (zn + lam),
            total,
            0.25 * factor,
        )

    initial = (x, y, z, jnp.zeros_like(x), jnp.ones_like(x))
    xn, yn, zn, total, factor = jax.lax.fori_loop(0, _steps(x.dtype), body, initial)
    mean = (xn + yn + 3.0 * zn) / 5.0
    dx = (mean - xn) / mean
    dy = (mean - yn) / mean
    dz = (mean - zn) / mean
    ea = dx * dy
    eb = dz * dz
    ec = ea - eb
    ed = ea - 6.0 * eb
    ee = ed + 2.0 * ec
    correction = (
        1.0
        + ed * (-3.0 / 14.0 + 9.0 * ed / 88.0 - 9.0 * dz * ee / 52.0)
        + dz * (ee / 6.0 + dz * (-9.0 * ec / 22.0 + 3.0 * dz * ea / 26.0))
    )
    return 3.0 * total + factor * correction / (mean * jnp.sqrt(mean))


def _rj_positive(x: Array, y: Array, z: Array, p: Array) -> Array:
    def body(
        _: int, state: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        xn, yn, zn, pn, total, factor = state
        sqrt_x = jnp.sqrt(xn)
        sqrt_y = jnp.sqrt(yn)
        sqrt_z = jnp.sqrt(zn)
        lam = sqrt_x * (sqrt_y + sqrt_z) + sqrt_y * sqrt_z
        alpha_root = pn * (sqrt_x + sqrt_y + sqrt_z) + sqrt_x * sqrt_y * sqrt_z
        beta_root = jnp.sqrt(pn) * (pn + lam)
        rc_scale = jnp.maximum(alpha_root, beta_root)
        safe_rc_scale = jnp.where(rc_scale > 0.0, rc_scale, jnp.ones_like(rc_scale))
        normalized_alpha = alpha_root / safe_rc_scale
        normalized_beta = beta_root / safe_rc_scale
        rc_value = _rc_positive(
            normalized_alpha * normalized_alpha, normalized_beta * normalized_beta
        )
        total = total + factor * rc_value / safe_rc_scale
        return (
            0.25 * (xn + lam),
            0.25 * (yn + lam),
            0.25 * (zn + lam),
            0.25 * (pn + lam),
            total,
            0.25 * factor,
        )

    initial = (x, y, z, p, jnp.zeros_like(x), jnp.ones_like(x))
    xn, yn, zn, pn, total, factor = jax.lax.fori_loop(0, _steps(x.dtype), body, initial)
    mean = (xn + yn + zn + 2.0 * pn) / 5.0
    dx = (mean - xn) / mean
    dy = (mean - yn) / mean
    dz = (mean - zn) / mean
    dp = (mean - pn) / mean
    ea = dx * (dy + dz) + dy * dz
    eb = dx * dy * dz
    ec = dp * dp
    ed = ea - 3.0 * ec
    ee = eb + 2.0 * dp * (ea - ec)
    correction = (
        1.0
        + ed * (-3.0 / 14.0 + 9.0 * ed / 88.0 - 9.0 * ee / 52.0)
        + eb * (1.0 / 6.0 + dp * (-3.0 / 11.0 + 3.0 * dp / 26.0))
        + dp * ea * (1.0 / 3.0 - 3.0 * dp / 22.0)
        - dp * ec / 3.0
    )
    return 3.0 * total + factor * correction / (mean * jnp.sqrt(mean))


def _elliprf_primal(x: Array, y: Array, z: Array) -> Array:
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0)
    zeros = (
        (x == 0.0).astype(jnp.int32)
        + (y == 0.0).astype(jnp.int32)
        + (z == 0.0).astype(jnp.int32)
    )
    pole = (~invalid) & (zeros >= 2)
    positive_infinity = jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z)
    special = invalid | pole | positive_infinity
    safe_x = jnp.where(special, jnp.ones_like(x), x)
    safe_y = jnp.where(special, jnp.ones_like(y), y)
    safe_z = jnp.where(special, jnp.ones_like(z), z)
    scale, (xn, yn, zn) = _scale((safe_x, safe_y, safe_z))
    value = _rf_positive(xn, yn, zn) / jnp.sqrt(scale)
    value = jnp.where(positive_infinity, jnp.zeros_like(value), value)
    value = jnp.where(pole, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _elliprc_primal(x: Array, y: Array) -> Array:
    invalid = (x < 0.0) | (y <= 0.0)
    positive_infinity = jnp.isposinf(x) | jnp.isposinf(y)
    special = invalid | positive_infinity
    safe_x = jnp.where(special, jnp.ones_like(x), x)
    safe_y = jnp.where(special, jnp.ones_like(y), y)
    scale, (xn, yn) = _scale((safe_x, safe_y))
    value = _rc_positive(xn, yn) / jnp.sqrt(scale)
    value = jnp.where(positive_infinity, jnp.zeros_like(value), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _elliprd_primal(x: Array, y: Array, z: Array) -> Array:
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0)
    pole = (~invalid) & ((z == 0.0) | ((x == 0.0) & (y == 0.0)))
    positive_infinity = jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z)
    special = invalid | pole | positive_infinity
    safe_x = jnp.where(special, jnp.ones_like(x), x)
    safe_y = jnp.where(special, jnp.ones_like(y), y)
    safe_z = jnp.where(special, jnp.ones_like(z), z)
    scale, (xn, yn, zn) = _scale((safe_x, safe_y, safe_z))
    value = _rd_positive(xn, yn, zn)
    value = value / scale
    value = value / jnp.sqrt(scale)
    value = jnp.where(positive_infinity, jnp.zeros_like(value), value)
    value = jnp.where(pole, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


def _elliprj_primal(x: Array, y: Array, z: Array, p: Array) -> Array:
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0) | (p <= 0.0)
    zeros = (
        (x == 0.0).astype(jnp.int32)
        + (y == 0.0).astype(jnp.int32)
        + (z == 0.0).astype(jnp.int32)
    )
    pole = (~invalid) & (zeros >= 2)
    positive_infinity = (
        jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z) | jnp.isposinf(p)
    )
    special = invalid | pole | positive_infinity
    safe_x = jnp.where(special, jnp.ones_like(x), x)
    safe_y = jnp.where(special, jnp.ones_like(y), y)
    safe_z = jnp.where(special, jnp.ones_like(z), z)
    safe_p = jnp.where(special, jnp.ones_like(p), p)
    scale, (xn, yn, zn, pn) = _scale((safe_x, safe_y, safe_z, safe_p))
    value = _rj_positive(xn, yn, zn, pn)
    value = value / scale
    value = value / jnp.sqrt(scale)
    value = jnp.where(positive_infinity, jnp.zeros_like(value), value)
    value = jnp.where(pole, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _elliprf_array(x: Array, y: Array, z: Array) -> Array:
    return _elliprf_primal(x, y, z)


@jax.custom_jvp
def _elliprc_array(x: Array, y: Array) -> Array:
    return _elliprc_primal(x, y)


@jax.custom_jvp
def _elliprd_array(x: Array, y: Array, z: Array) -> Array:
    return _elliprd_primal(x, y, z)


@jax.custom_jvp
def _elliprj_array(x: Array, y: Array, z: Array, p: Array) -> Array:
    return _elliprj_primal(x, y, z, p)


def _apply_tangent(derivative: Array, tangent: Array | SymbolicZero) -> Array:
    if isinstance(tangent, SymbolicZero):
        return jnp.zeros_like(derivative)
    return derivative * tangent


def _rf_partials(x: Array, y: Array, z: Array) -> tuple[Array, Array, Array]:
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0)
    zeros = (
        (x == 0.0).astype(jnp.int32)
        + (y == 0.0).astype(jnp.int32)
        + (z == 0.0).astype(jnp.int32)
    )
    pole = (~invalid) & (zeros >= 2)
    positive_infinity = jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z)
    regular = (~invalid) & (~pole) & (~positive_infinity)
    infinity_limit = positive_infinity & (~invalid) & (~pole)
    safe_x = jnp.where(regular, x, jnp.ones_like(x))
    safe_y = jnp.where(regular, y, jnp.ones_like(y))
    safe_z = jnp.where(regular, z, jnp.ones_like(z))

    def partial(coordinate: Array) -> Array:
        positive = coordinate > 0.0
        safe_p = jnp.where(regular & positive, coordinate, jnp.ones_like(coordinate))
        derivative = -_elliprj_array(safe_x, safe_y, safe_z, safe_p) / 6.0
        derivative = jnp.where(
            coordinate == 0.0, jnp.full_like(derivative, -jnp.inf), derivative
        )
        exceptional = jnp.where(
            infinity_limit,
            jnp.zeros_like(derivative),
            jnp.full_like(derivative, jnp.nan),
        )
        return jnp.where(regular, derivative, exceptional)

    return partial(x), partial(y), partial(z)


def _rj_partials(
    x: Array, y: Array, z: Array, p: Array, value: Array
) -> tuple[Array, Array, Array, Array]:
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0) | (p <= 0.0)
    zeros = (
        (x == 0.0).astype(jnp.int32)
        + (y == 0.0).astype(jnp.int32)
        + (z == 0.0).astype(jnp.int32)
    )
    pole = (~invalid) & (zeros >= 2)
    positive_infinity = (
        jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z) | jnp.isposinf(p)
    )
    regular = (~invalid) & (~pole) & (~positive_infinity)
    infinity_limit = positive_infinity & (~invalid) & (~pole)
    safe_x = jnp.where(regular, x, jnp.ones_like(x))
    safe_y = jnp.where(regular, y, jnp.ones_like(y))
    safe_z = jnp.where(regular, z, jnp.ones_like(z))
    safe_p = jnp.where(regular, p, jnp.ones_like(p))
    safe_value = jnp.where(regular, value, jnp.ones_like(value))

    def unequal_partial(coordinate: Array) -> tuple[Array, Array]:
        equal = coordinate == safe_p
        evaluable = regular & (~equal) & (coordinate > 0.0)
        evaluation_p = jnp.where(evaluable, coordinate, safe_p)
        comparison = _elliprj_array(safe_x, safe_y, safe_z, evaluation_p)
        denominator = jnp.where(
            equal, jnp.ones_like(coordinate), 2.0 * (safe_p - coordinate)
        )
        derivative = (safe_value - comparison) / denominator
        derivative = jnp.where(
            coordinate == 0.0, jnp.full_like(derivative, -jnp.inf), derivative
        )
        return derivative, equal

    dx, equal_x = unequal_partial(safe_x)
    dy, equal_y = unequal_partial(safe_y)
    dz, equal_z = unequal_partial(safe_z)
    weighted_dx = safe_x * jnp.where((safe_x > 0.0) & (~equal_x), dx, jnp.zeros_like(dx))
    weighted_dy = safe_y * jnp.where((safe_y > 0.0) & (~equal_y), dy, jnp.zeros_like(dy))
    weighted_dz = safe_z * jnp.where((safe_z > 0.0) & (~equal_z), dz, jnp.zeros_like(dz))
    weighted_sum = weighted_dx + weighted_dy + weighted_dz
    equal_count = (
        equal_x.astype(safe_value.dtype)
        + equal_y.astype(safe_value.dtype)
        + equal_z.astype(safe_value.dtype)
    )
    equal_derivative = (-1.5 * safe_value - weighted_sum) / ((equal_count + 2.0) * safe_p)
    dx = jnp.where(equal_x, equal_derivative, dx)
    dy = jnp.where(equal_y, equal_derivative, dy)
    dz = jnp.where(equal_z, equal_derivative, dz)
    dp = jnp.where(
        equal_count > 0.0,
        2.0 * equal_derivative,
        (-1.5 * safe_value - weighted_sum) / safe_p,
    )

    def mask(derivative: Array) -> Array:
        exceptional = jnp.where(
            infinity_limit,
            jnp.zeros_like(derivative),
            jnp.full_like(derivative, jnp.nan),
        )
        return jnp.where(regular, derivative, exceptional)

    return mask(dx), mask(dy), mask(dz), mask(dp)


def _elliprf_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    x, y, z = primals
    x_tangent, y_tangent, z_tangent = tangents
    value = _elliprf_array(x, y, z)
    dx, dy, dz = _rf_partials(x, y, z)
    tangent = (
        _apply_tangent(dx, x_tangent)
        + _apply_tangent(dy, y_tangent)
        + _apply_tangent(dz, z_tangent)
    )
    return value, tangent


def _elliprc_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    x, y = primals
    x_tangent, y_tangent = tangents
    value = _elliprc_array(x, y)
    dx, dy_first, dy_second = _rf_partials(x, y, y)
    tangent = _apply_tangent(dx, x_tangent) + _apply_tangent(
        dy_first + dy_second, y_tangent
    )
    return value, tangent


def _elliprd_jvp(
    primals: tuple[Array, Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    x, y, z = primals
    x_tangent, y_tangent, z_tangent = tangents
    value = _elliprd_array(x, y, z)
    dx, dy, dz, dp = _rj_partials(x, y, z, z, value)
    tangent = (
        _apply_tangent(dx, x_tangent)
        + _apply_tangent(dy, y_tangent)
        + _apply_tangent(dz + dp, z_tangent)
    )
    return value, tangent


def _elliprj_jvp(
    primals: tuple[Array, Array, Array, Array],
    tangents: tuple[
        Array | SymbolicZero,
        Array | SymbolicZero,
        Array | SymbolicZero,
        Array | SymbolicZero,
    ],
) -> tuple[Array, Array]:
    x, y, z, p = primals
    x_tangent, y_tangent, z_tangent, p_tangent = tangents
    value = _elliprj_array(x, y, z, p)
    dx, dy, dz, dp = _rj_partials(x, y, z, p, value)
    tangent = (
        _apply_tangent(dx, x_tangent)
        + _apply_tangent(dy, y_tangent)
        + _apply_tangent(dz, z_tangent)
        + _apply_tangent(dp, p_tangent)
    )
    return value, tangent


_elliprf_array.defjvp(_elliprf_jvp, symbolic_zeros=True)
_elliprc_array.defjvp(_elliprc_jvp, symbolic_zeros=True)
_elliprd_array.defjvp(_elliprd_jvp, symbolic_zeros=True)
_elliprj_array.defjvp(_elliprj_jvp, symbolic_zeros=True)


def elliprf(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    """Evaluate principal Carlson ``R_F(x, y, z)``."""
    if jnp.issubdtype(jnp.result_type(x, y, z), jnp.complexfloating):
        from ._continuation import complex_elliprf

        return complex_elliprf(x, y, z)
    x, y, z = promote_real("elliprf", x, y, z)
    return _elliprf_array(*jnp.broadcast_arrays(x, y, z))


def elliprc(x: ArrayLike, y: ArrayLike) -> Array:
    """Evaluate principal Carlson ``R_C(x, y)``."""
    if jnp.issubdtype(jnp.result_type(x, y), jnp.complexfloating):
        from ._continuation import complex_elliprc

        return complex_elliprc(x, y)
    x, y = promote_real("elliprc", x, y)
    return _elliprc_array(*jnp.broadcast_arrays(x, y))


def elliprd(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    """Evaluate principal Carlson ``R_D(x, y, z)``."""
    if jnp.issubdtype(jnp.result_type(x, y, z), jnp.complexfloating):
        from ._continuation import complex_elliprd

        return complex_elliprd(x, y, z)
    x, y, z = promote_real("elliprd", x, y, z)
    return _elliprd_array(*jnp.broadcast_arrays(x, y, z))


def elliprj(x: ArrayLike, y: ArrayLike, z: ArrayLike, p: ArrayLike) -> Array:
    """Evaluate principal Carlson ``R_J(x, y, z, p)``."""
    if jnp.issubdtype(jnp.result_type(x, y, z, p), jnp.complexfloating):
        from ._continuation import complex_elliprj

        return complex_elliprj(x, y, z, p)
    x, y, z, p = promote_real("elliprj", x, y, z, p)
    return _elliprj_array(*jnp.broadcast_arrays(x, y, z, p))


def elliprg(x: ArrayLike, y: ArrayLike, z: ArrayLike) -> Array:
    """Evaluate Carlson's completely symmetric integral ``R_G(x, y, z)``."""
    if jnp.issubdtype(jnp.result_type(x, y, z), jnp.complexfloating):
        from ._continuation import complex_elliprg

        return complex_elliprg(x, y, z)
    x, y, z = promote_real("elliprg", x, y, z)
    x, y, z = jnp.broadcast_arrays(x, y, z)
    invalid = (x < 0.0) | (y < 0.0) | (z < 0.0)
    positive_infinity = jnp.isposinf(x) | jnp.isposinf(y) | jnp.isposinf(z)
    special = invalid | positive_infinity
    safe_x = jnp.where(special, jnp.ones_like(x), x)
    safe_y = jnp.where(special, jnp.ones_like(y), y)
    safe_z = jnp.where(special, jnp.ones_like(z), z)
    scale, (xn, yn, zn) = _scale((safe_x, safe_y, safe_z))

    swap_x = xn > zn
    ax = jnp.where(swap_x, zn, xn)
    az = jnp.where(swap_x, xn, zn)
    swap_y = yn > az
    ay = jnp.where(swap_y, az, yn)
    az = jnp.where(swap_y, yn, az)

    positive = (
        (xn > 0.0).astype(jnp.int32)
        + (yn > 0.0).astype(jnp.int32)
        + (zn > 0.0).astype(jnp.int32)
    )
    degenerate = positive <= 1
    eval_x = jnp.where(degenerate, jnp.ones_like(ax), ax)
    eval_y = jnp.where(degenerate, jnp.ones_like(ay), ay)
    eval_z = jnp.where(degenerate, jnp.ones_like(az), az)
    normalized_value = 0.5 * (
        eval_z * _elliprf_array(eval_x, eval_y, eval_z)
        - (eval_x - eval_z)
        * (eval_y - eval_z)
        * _elliprd_array(eval_x, eval_y, eval_z)
        / 3.0
        + jnp.sqrt(eval_x * eval_y / eval_z)
    )
    normalized_value = jnp.where(degenerate, 0.5 * jnp.sqrt(az), normalized_value)
    value = normalized_value * jnp.sqrt(scale)
    value = jnp.where(positive_infinity, jnp.full_like(value, jnp.inf), value)
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


__all__ = ["elliprc", "elliprd", "elliprf", "elliprg", "elliprj"]
