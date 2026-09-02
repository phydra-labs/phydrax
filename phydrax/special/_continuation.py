#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


_LANCZOS = (
    0.99999999999980993,
    676.5203681218851,
    -1259.1392167224028,
    771.32342877765313,
    -176.61502916214059,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
)


def promote_principal(*values: ArrayLike) -> tuple[Array, ...]:
    arrays = tuple(jnp.asarray(value) for value in values)
    dtype = jnp.result_type(*arrays)
    if dtype in (jnp.float64, jnp.complex128):
        dtype = jnp.complex128
    else:
        dtype = jnp.complex64
    return tuple(jnp.asarray(value, dtype=dtype) for value in arrays)


def principal_log(value: ArrayLike, /) -> Array:
    (z,) = promote_principal(value)
    return jnp.log(z)


def principal_sqrt(value: ArrayLike, /) -> Array:
    (z,) = promote_principal(value)
    return jnp.sqrt(z)


def _loggamma_lanczos(value: Array, /) -> Array:
    z = value
    reflected = jnp.real(z) < 0.5
    safe = jnp.where(reflected, 1.0 - z, z)
    shifted = safe - 1.0
    series = jnp.asarray(_LANCZOS[0], dtype=z.dtype)
    for index, coefficient in enumerate(_LANCZOS[1:], start=1):
        series = series + coefficient / (shifted + index)
    t = shifted + 7.5
    direct = (
        0.5 * math.log(2.0 * math.pi) + (shifted + 0.5) * jnp.log(t) - t + jnp.log(series)
    )
    reflection = math.log(math.pi) - jnp.log(jnp.sin(math.pi * z)) - direct
    return jnp.where(reflected, reflection, direct)


def _jv_series_direct(order: Array, argument: Array, /) -> Array:
    v, z = jnp.broadcast_arrays(order, argument)
    half = 0.5 * z
    term = jnp.exp(v * jnp.log(half) - _loggamma_lanczos(v + 1.0))
    total = term
    for index in range(1, 96):
        term = term * (-(half * half)) / (index * (v + index))
        total = total + term
    return total


def _negative_integer_jv_order_derivative(order: Array, argument: Array, /) -> Array:
    n = jnp.real(order)
    n_integer = n.astype(jnp.int32)
    half = 0.5 * argument
    log_half = jnp.log(half)
    parity = jnp.where(
        jnp.remainder(n, 2.0) == 0.0,
        jnp.ones_like(argument),
        -jnp.ones_like(argument),
    )

    first_early = -parity * jnp.exp(
        -n * log_half + _loggamma_lanczos(jnp.asarray(n, dtype=argument.dtype))
    )

    def accumulate_early(index, state):
        term, total = state
        total = total + term
        has_next = index + 1 < n_integer
        denominator = jnp.asarray(
            (index + 1) * (n_integer - index - 1),
            dtype=argument.real.dtype,
        )
        denominator = jnp.where(has_next, denominator, 1.0)
        term = jnp.where(
            has_next,
            term * (half * half) / denominator,
            jnp.zeros_like(term),
        )
        return term, total

    _, early = jax.lax.fori_loop(
        0,
        n_integer,
        accumulate_early,
        (first_early, jnp.zeros_like(argument)),
    )

    term = parity * jnp.exp(
        n * log_half - _loggamma_lanczos(jnp.asarray(n + 1.0, dtype=argument.dtype))
    )
    late = jnp.zeros_like(argument)
    harmonic = 0.0
    for index in range(96):
        late = late + term * (log_half - harmonic + 0.5772156649015329)
        term = term * (-(half * half)) / ((index + 1) * (n + index + 1.0))
        harmonic += 1.0 / (index + 1)
    return early + late


def _jv_series(order: Array, argument: Array, /) -> Array:
    v, z = jnp.broadcast_arrays(order, argument)
    nearest = jnp.round(jnp.real(v)).astype(v.dtype)
    negative_integer = (jnp.real(v) < 0.0) & (v == nearest)
    positive_integer = jnp.where(negative_integer, -nearest, jnp.ones_like(v))
    safe_order = jnp.where(negative_integer, positive_integer, v)
    direct = _jv_series_direct(safe_order, z)
    parity = jnp.where(
        jnp.remainder(jnp.real(positive_integer), 2.0) == 0.0,
        jnp.ones_like(v),
        -jnp.ones_like(v),
    )
    reflected_value = parity * _jv_series_direct(positive_integer, z)
    reflected_derivative = jax.vmap(_negative_integer_jv_order_derivative)(
        jnp.ravel(positive_integer), jnp.ravel(z)
    ).reshape(v.shape)
    reflected = reflected_value + (v - nearest) * reflected_derivative
    return jnp.where(negative_integer, reflected, direct)


def _iv_series(order: Array, argument: Array, /) -> Array:
    v, z = jnp.broadcast_arrays(order, argument)
    half = 0.5 * z
    term = jnp.exp(v * jnp.log(half) - _loggamma_lanczos(v + 1.0))
    total = term
    for index in range(1, 96):
        term = term * (half * half) / (index * (v + index))
        total = total + term
    return total


def complex_jv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _jv_series(order_, argument_)


def _order_derivative(function, order: Array, argument: Array, /) -> Array:
    return jax.jvp(
        lambda value: function(value, argument), (order,), (jnp.ones_like(order),)
    )[1]


def _yv_connection(order: Array, argument: Array, /) -> Array:
    sine = jnp.sin(math.pi * order)
    return (
        jnp.cos(math.pi * order) * _jv_series(order, argument)
        - _jv_series(-order, argument)
    ) / sine


def complex_yv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    ordinary = _yv_connection(order_, argument_)
    nearest = jnp.round(jnp.real(order_)).astype(order_.dtype)
    delta = jnp.asarray(
        8.0 * jnp.sqrt(jnp.finfo(argument_.real.dtype).eps),
        dtype=order_.real.dtype,
    ).astype(order_.dtype)
    upper = _yv_connection(nearest + delta, argument_)
    lower = _yv_connection(nearest - delta, argument_)
    center = 0.5 * (upper + lower)
    slope = (upper - lower) / (2.0 * delta)
    integer_limit = center + (order_ - nearest) * slope
    return jnp.where(
        jnp.abs(order_ - nearest) <= delta,
        integer_limit,
        ordinary,
    )


def complex_hankel1(order: ArrayLike, argument: ArrayLike, /) -> Array:
    return complex_jv(order, argument) + 1j * complex_yv(order, argument)


def complex_hankel2(order: ArrayLike, argument: ArrayLike, /) -> Array:
    return complex_jv(order, argument) - 1j * complex_yv(order, argument)


def complex_iv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _iv_series(order_, argument_)


def _kv_connection(order: Array, argument: Array, /) -> Array:
    return (
        0.5
        * math.pi
        * (_iv_series(-order, argument) - _iv_series(order, argument))
        / jnp.sin(math.pi * order)
    )


def complex_kv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    ordinary = _kv_connection(order_, argument_)
    nearest = jnp.round(jnp.real(order_)).astype(order_.dtype)
    delta = jnp.asarray(
        8.0 * jnp.sqrt(jnp.finfo(argument_.real.dtype).eps),
        dtype=order_.real.dtype,
    ).astype(order_.dtype)
    upper = _kv_connection(nearest + delta, argument_)
    lower = _kv_connection(nearest - delta, argument_)
    center = 0.5 * (upper + lower)
    slope = (upper - lower) / (2.0 * delta)
    integer_limit = center + (order_ - nearest) * slope
    return jnp.where(
        jnp.abs(order_ - nearest) <= delta,
        integer_limit,
        ordinary,
    )


def jv_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _order_derivative(_jv_series, order_, argument_)


def yv_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _order_derivative(lambda value, z: complex_yv(value, z), order_, argument_)


def iv_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _order_derivative(_iv_series, order_, argument_)


def kv_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _order_derivative(lambda value, z: complex_kv(value, z), order_, argument_)


def ive_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    _, argument_ = promote_principal(order, argument)
    return jnp.exp(-jnp.abs(jnp.real(argument_))) * iv_order_derivative(order, argument_)


def kve_order_derivative(order: ArrayLike, argument: ArrayLike, /) -> Array:
    _, argument_ = promote_principal(order, argument)
    return jnp.exp(argument_) * kv_order_derivative(order, argument_)


_AI0 = 0.3550280538878172392600631860041831764
_AIP0 = -0.2588194037928067984051835601892039635
_BI0 = 0.6149266274460007351509223690936135536
_BIP0 = 0.4482883573538263579148237103988283909


def _airy_series(z: Array, value0: float, derivative0: float, /) -> tuple[Array, Array]:
    coefficients = [value0, derivative0, 0.0]
    for index in range(1, 94):
        coefficients.append(coefficients[index - 1] / ((index + 2) * (index + 1)))
    value = jnp.zeros_like(z)
    derivative = jnp.zeros_like(z)
    power = jnp.ones_like(z)
    previous_power = jnp.ones_like(z)
    for index, coefficient in enumerate(coefficients):
        value = value + coefficient * power
        if index:
            derivative = derivative + index * coefficient * previous_power
        previous_power = power
        power = power * z
    return value, derivative


def complex_airy(argument: ArrayLike, /) -> tuple[Array, Array, Array, Array]:
    (z,) = promote_principal(argument)
    ai, aip = _airy_series(z, _AI0, _AIP0)
    bi, bip = _airy_series(z, _BI0, _BIP0)
    return ai, aip, bi, bip


def _carlson_steps(dtype) -> int:
    return 14 if dtype == jnp.complex64 else 24


def complex_elliprf(x: ArrayLike, y: ArrayLike, z: ArrayLike, /) -> Array:
    x_, y_, z_ = jnp.broadcast_arrays(*promote_principal(x, y, z))
    for _ in range(_carlson_steps(x_.dtype)):
        sx, sy, sz = jnp.sqrt(x_), jnp.sqrt(y_), jnp.sqrt(z_)
        lam = sx * (sy + sz) + sy * sz
        x_, y_, z_ = 0.25 * (x_ + lam), 0.25 * (y_ + lam), 0.25 * (z_ + lam)
    mean = (x_ + y_ + z_) / 3.0
    dx, dy, dz = (mean - x_) / mean, (mean - y_) / mean, (mean - z_) / mean
    e2 = dx * dy - dz * dz
    e3 = dx * dy * dz
    return (1.0 + ((e2 / 24.0 - 0.1 - 3.0 * e3 / 44.0) * e2 + e3 / 14.0)) / jnp.sqrt(mean)


def complex_elliprc(x: ArrayLike, y: ArrayLike, /) -> Array:
    x_, y_ = jnp.broadcast_arrays(*promote_principal(x, y))
    for _ in range(_carlson_steps(x_.dtype)):
        lam = 2.0 * jnp.sqrt(x_) * jnp.sqrt(y_) + y_
        x_, y_ = 0.25 * (x_ + lam), 0.25 * (y_ + lam)
    mean = (x_ + 2.0 * y_) / 3.0
    s = (y_ - mean) / mean
    return (
        1.0 + s * s * (0.3 + s * (1.0 / 7.0 + s * (0.375 + s * 9.0 / 22.0)))
    ) / jnp.sqrt(mean)


def complex_elliprd(x: ArrayLike, y: ArrayLike, z: ArrayLike, /) -> Array:
    x_, y_, z_ = jnp.broadcast_arrays(*promote_principal(x, y, z))
    total = jnp.zeros_like(x_)
    factor = jnp.ones_like(x_)
    for _ in range(_carlson_steps(x_.dtype)):
        sx, sy, sz = jnp.sqrt(x_), jnp.sqrt(y_), jnp.sqrt(z_)
        lam = sx * (sy + sz) + sy * sz
        total = total + factor / (sz * (z_ + lam))
        x_, y_, z_, factor = (
            0.25 * (x_ + lam),
            0.25 * (y_ + lam),
            0.25 * (z_ + lam),
            0.25 * factor,
        )
    mean = (x_ + y_ + 3.0 * z_) / 5.0
    dx, dy, dz = (mean - x_) / mean, (mean - y_) / mean, (mean - z_) / mean
    ea, eb = dx * dy, dz * dz
    ec, ed = ea - eb, ea - 6.0 * eb
    ee = ed + 2.0 * ec
    correction = (
        1.0
        + ed * (-3.0 / 14.0 + 9.0 * ed / 88.0 - 9.0 * dz * ee / 52.0)
        + dz * (ee / 6.0 + dz * (-9.0 * ec / 22.0 + 3.0 * dz * ea / 26.0))
    )
    return 3.0 * total + factor * correction / (mean * jnp.sqrt(mean))


def complex_elliprj(x: ArrayLike, y: ArrayLike, z: ArrayLike, p: ArrayLike, /) -> Array:
    x_, y_, z_, p_ = jnp.broadcast_arrays(*promote_principal(x, y, z, p))
    total = jnp.zeros_like(x_)
    factor = jnp.ones_like(x_)
    for _ in range(_carlson_steps(x_.dtype)):
        sx, sy, sz = jnp.sqrt(x_), jnp.sqrt(y_), jnp.sqrt(z_)
        lam = sx * (sy + sz) + sy * sz
        alpha = p_ * (sx + sy + sz) + sx * sy * sz
        beta = jnp.sqrt(p_) * (p_ + lam)
        total = total + factor * complex_elliprc(alpha * alpha, beta * beta)
        x_, y_, z_, p_, factor = (
            0.25 * (x_ + lam),
            0.25 * (y_ + lam),
            0.25 * (z_ + lam),
            0.25 * (p_ + lam),
            0.25 * factor,
        )
    mean = (x_ + y_ + z_ + 2.0 * p_) / 5.0
    dx, dy, dz, dp = (
        (mean - x_) / mean,
        (mean - y_) / mean,
        (mean - z_) / mean,
        (mean - p_) / mean,
    )
    ea = dx * (dy + dz) + dy * dz
    eb, ec = dx * dy * dz, dp * dp
    ed, ee = ea - 3.0 * ec, eb + 2.0 * dp * (ea - ec)
    correction = (
        1.0
        + ed * (-3.0 / 14.0 + 9.0 * ed / 88.0 - 9.0 * ee / 52.0)
        + eb * (1.0 / 6.0 + dp * (-3.0 / 11.0 + 3.0 * dp / 26.0))
        + dp * ea * (1.0 / 3.0 - 3.0 * dp / 22.0)
        - dp * ec / 3.0
    )
    return 3.0 * total + factor * correction / (mean * jnp.sqrt(mean))


def complex_elliprg(x: ArrayLike, y: ArrayLike, z: ArrayLike, /) -> Array:
    x_, y_, z_ = jnp.broadcast_arrays(*promote_principal(x, y, z))
    return 0.5 * (
        z_ * complex_elliprf(x_, y_, z_)
        - (x_ - z_) * (y_ - z_) * complex_elliprd(x_, y_, z_) / 3.0
        + jnp.sqrt(x_ * y_ / z_)
    )


def complex_dawsn(argument: ArrayLike, /) -> Array:
    from ._faddeeva import wofz

    (z,) = promote_principal(argument)
    return math.sqrt(math.pi) * (wofz(z) - jnp.exp(-(z * z))) / (2j)


def complex_ellipj(
    argument: ArrayLike,
    parameter: ArrayLike,
    /,
) -> tuple[Array, Array, Array, Array]:
    """Principal fixed-capacity complex Jacobi functions via descending AGM."""

    u, m = jnp.broadcast_arrays(*promote_principal(argument, parameter))
    capacity = 16
    tolerance = 8.0 * jnp.finfo(u.real.dtype).eps
    a_values = [jnp.ones_like(u)]
    c_values = []
    active_values = []
    b = jnp.sqrt(1.0 - m)
    b = jnp.where(jnp.abs(1.0 - b) <= jnp.abs(1.0 + b), b, -b)
    running = jnp.ones_like(u, dtype=bool)
    for _ in range(capacity):
        a = a_values[-1]
        c = 0.5 * (a - b)
        next_a = 0.5 * (a + b)
        next_b = jnp.sqrt(a * b)
        next_b = jnp.where(
            jnp.abs(next_a - next_b) <= jnp.abs(next_a + next_b),
            next_b,
            -next_b,
        )
        active = running & (jnp.abs(c) > tolerance * jnp.maximum(jnp.abs(next_a), 1.0))
        c_values.append(jnp.where(active, c, jnp.zeros_like(c)))
        active_values.append(active)
        a_values.append(jnp.where(active, next_a, a))
        b = jnp.where(active, next_b, b)
        running = active

    amplitude = (2.0**capacity) * a_values[-1] * u
    for index in range(capacity - 1, -1, -1):
        active = active_values[index]
        safe_amplitude = jnp.where(active, amplitude, jnp.zeros_like(amplitude))
        safe_a = jnp.where(active, a_values[index + 1], jnp.ones_like(amplitude))
        correction = jnp.where(
            active,
            jnp.arcsin(c_values[index] * jnp.sin(safe_amplitude) / safe_a),
            jnp.zeros_like(amplitude),
        )
        amplitude = 0.5 * (amplitude + correction)
    sn = jnp.sin(amplitude)
    cn = jnp.cos(amplitude)
    dn = jnp.sqrt(1.0 - m * sn * sn)
    endpoint_zero = m == 0.0
    endpoint_one = m == 1.0
    hyperbolic_sn = jnp.tanh(u)
    hyperbolic_cn = 1.0 / jnp.cosh(u)
    sn = jnp.where(endpoint_zero, jnp.sin(u), sn)
    cn = jnp.where(endpoint_zero, jnp.cos(u), cn)
    dn = jnp.where(endpoint_zero, jnp.ones_like(u), dn)
    amplitude = jnp.where(endpoint_zero, u, amplitude)
    sn = jnp.where(endpoint_one, hyperbolic_sn, sn)
    cn = jnp.where(endpoint_one, hyperbolic_cn, cn)
    dn = jnp.where(endpoint_one, hyperbolic_cn, dn)
    amplitude = jnp.where(endpoint_one, jnp.arcsin(hyperbolic_sn), amplitude)
    return sn, cn, dn, amplitude


__all__ = [
    "complex_airy",
    "complex_dawsn",
    "complex_ellipj",
    "complex_elliprc",
    "complex_elliprd",
    "complex_elliprf",
    "complex_elliprg",
    "complex_elliprj",
    "complex_hankel1",
    "complex_hankel2",
    "complex_iv",
    "complex_jv",
    "complex_kv",
    "complex_yv",
    "ive_order_derivative",
    "iv_order_derivative",
    "jv_order_derivative",
    "kv_order_derivative",
    "kve_order_derivative",
    "principal_log",
    "principal_sqrt",
    "yv_order_derivative",
]
