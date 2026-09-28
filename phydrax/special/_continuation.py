#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable

import jax
import jax.numpy as jnp
from jax import Array
from jax.custom_derivatives import SymbolicZero
from jax.typing import ArrayLike, DTypeLike


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

    def accumulate(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        current, total_ = state
        current = current * (-(half * half)) / (index * (v + index))
        return current, total_ + current

    _, total = jax.lax.fori_loop(1, 96, accumulate, (term, term))
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

    def accumulate_early(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
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

    def accumulate_late(
        index: Array, state: tuple[Array, Array, Array]
    ) -> tuple[Array, Array, Array]:
        current, total, harmonic = state
        total = total + current * (log_half - harmonic + 0.5772156649015329)
        current = current * (-(half * half)) / ((index + 1) * (n + index + 1.0))
        return current, total, harmonic + 1.0 / (index + 1)

    _, late, _ = jax.lax.fori_loop(
        0,
        96,
        accumulate_late,
        (term, jnp.zeros_like(argument), jnp.asarray(0.0)),
    )
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

    def accumulate(index: Array, state: tuple[Array, Array]) -> tuple[Array, Array]:
        current, total_ = state
        current = current * (half * half) / (index * (v + index))
        return current, total_ + current

    _, total = jax.lax.fori_loop(1, 96, accumulate, (term, term))
    return total


def complex_jv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    order_, argument_ = promote_principal(order, argument)
    return _jv_series(order_, argument_)


def _order_derivative(
    function: Callable[[Array, Array], Array], order: Array, argument: Array, /
) -> Array:
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


def _kv_power_series(order: Array, argument: Array, /) -> Array:
    ordinary = _kv_connection(order, argument)
    nearest = jnp.round(jnp.real(order)).astype(order.dtype)
    delta = jnp.asarray(
        8.0 * jnp.sqrt(jnp.finfo(argument.real.dtype).eps),
        dtype=order.real.dtype,
    ).astype(order.dtype)
    upper = _kv_connection(nearest + delta, argument)
    lower = _kv_connection(nearest - delta, argument)
    center = 0.5 * (upper + lower)
    slope = (upper - lower) / (2.0 * delta)
    integer_limit = center + (order - nearest) * slope
    return jnp.where(
        jnp.abs(order - nearest) <= delta,
        integer_limit,
        ordinary,
    )


# Taylor coefficients of 1/Gamma(1 + x) about x = 0; thirty terms reach double
# precision for |x| <= 1, covering the Temme reduced orders |mu| <= 1/2 and the
# admitted complex orders with |Im mu| <= _KV_MAX_IMAGINARY_ORDER.
_RECIPROCAL_GAMMA_TAYLOR = (
    1.0,
    0.5772156649015329,
    -0.6558780715202539,
    -0.04200263503409524,
    0.16653861138229148,
    -0.04219773455554433,
    -0.009621971527876973,
    0.0072189432466631,
    -0.0011651675918590652,
    -0.00021524167411495098,
    0.0001280502823881162,
    -2.013485478078824e-05,
    -1.2504934821426706e-06,
    1.133027231981696e-06,
    -2.056338416977607e-07,
    6.116095104481416e-09,
    5.002007644469223e-09,
    -1.18127457048702e-09,
    1.0434267116911005e-10,
    7.782263439905071e-12,
    -3.696805618642206e-12,
    5.100370287454476e-13,
    -2.0583260535665066e-14,
    -5.348122539423018e-15,
    1.2267786282382608e-15,
    -1.1812593016974588e-16,
    1.1866922547516004e-18,
    1.4123806553180319e-18,
    -2.29874568443537e-19,
    1.7144063219273374e-20,
)
_KV_TEMME_RADIUS = 2.0
_KV_TEMME_TERMS = 28
# CF2 needs 135 steps at |z| = 2 on the imaginary axis for real orders and 155
# for |Im mu| = 1 before the series increment drops below float64 epsilon.
_KV_STEED_STEPS = 168
_KV_MAX_RECURRENCE = 128
_KV_MAX_IMAGINARY_ORDER = 1.0


def _temme_gamma_terms(mu: Array, /) -> tuple[Array, Array, Array, Array]:
    """Return Temme's ``gamma1``, ``gamma2``, ``1/Gamma(1+mu)``, ``1/Gamma(1-mu)``."""
    square = mu * mu
    even = jnp.zeros_like(mu)
    odd = jnp.zeros_like(mu)
    for coefficient in _RECIPROCAL_GAMMA_TAYLOR[-2::-2]:
        even = even * square + coefficient
    for coefficient in _RECIPROCAL_GAMMA_TAYLOR[-1::-2]:
        odd = odd * square + coefficient
    return -odd, even, even + mu * odd, even - mu * odd


def _x_over_sin(x: Array, /) -> Array:
    small = jnp.abs(x) < 1e-2
    safe = jnp.where(small, jnp.ones_like(x), x)
    square = x * x
    series = 1.0 + square * (
        1.0 / 6.0 + square * (7.0 / 360.0 + square * (31.0 / 15120.0))
    )
    return jnp.where(small, series, safe / jnp.sin(safe))


def _sinh_over_x(x: Array, /) -> Array:
    small = jnp.abs(x) < 1e-2
    safe = jnp.where(small, jnp.ones_like(x), x)
    square = x * x
    series = 1.0 + square * (1.0 / 6.0 + square * (1.0 / 120.0 + square * (1.0 / 5040.0)))
    return jnp.where(small, series, jnp.sinh(safe) / safe)


def _kv_temme_series(mu: Array, argument: Array, /) -> tuple[Array, Array]:
    """Temme's series for ``K_mu`` and ``K_{mu+1}`` with ``|mu| <= 1/2``, ``|z| <= 2``."""
    log_inverse_half = -jnp.log(0.5 * argument)
    exponent = mu * log_inverse_half
    gamma1, gamma2, gamma_plus, gamma_minus = _temme_gamma_terms(mu)
    f = _x_over_sin(math.pi * mu) * (
        gamma1 * jnp.cosh(exponent) + gamma2 * _sinh_over_x(exponent) * log_inverse_half
    )
    power = jnp.exp(exponent)
    p = 0.5 * power / gamma_plus
    q = 0.5 / (power * gamma_minus)
    quarter_square = 0.25 * argument * argument
    mu_square = mu * mu

    def accumulate(
        index: Array, state: tuple[Array, Array, Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        f_, p_, q_, c, total, total1 = state
        k = index.astype(argument.dtype)
        f_ = (k * f_ + p_ + q_) / (k * k - mu_square)
        c = c * quarter_square / k
        p_ = p_ / (k - mu)
        q_ = q_ / (k + mu)
        return f_, p_, q_, c, total + c * f_, total1 + c * (p_ - k * f_)

    _, _, _, _, total, total1 = jax.lax.fori_loop(
        1,
        _KV_TEMME_TERMS + 1,
        accumulate,
        (f, p, q, jnp.ones_like(argument), f, p),
    )
    return total, 2.0 * total1 / argument


type _SteedState = tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]


def _kve_steed(mu: Array, argument: Array, /) -> tuple[Array, Array]:
    """Scaled ``e^z K_mu`` and ``e^z K_{mu+1}`` by Temme's CF2 in Steed form.

    Complex-argument analogue of the Thompson--Barnett continued fraction used
    for ``x >= 2`` in Numerical Recipes ``bessik``. The factorially growing
    coefficient ``c_k`` and decaying ``Q_k`` are carried only as the products
    ``c_k Q_{k-1}`` and ``c_k Q_k``. Forward recurrence for ``Q_k`` is unstable
    once the sum has converged, so each lane freezes as soon as both the ``h``
    and ``s`` increments fall below machine epsilon.
    """
    eps = jnp.finfo(argument.real.dtype).eps
    a1 = 0.25 - mu * mu
    b = 2.0 * (1.0 + argument)
    d = 1.0 / b
    state = (-a1, b, d, d, d, a1, jnp.zeros_like(argument), a1, 1.0 + a1 * d)

    def iterate(
        index: Array, carry: tuple[_SteedState, Array]
    ) -> tuple[_SteedState, Array]:
        current, done = carry
        a, b_, d_, delta_h, h, q, lower, upper, s = current
        k = index.astype(argument.dtype)
        a = a - 2.0 * (k - 1.0)
        lower, upper = -a * upper / k, (b_ * upper - lower) / k
        q = q + upper
        b_ = b_ + 2.0
        d_ = 1.0 / (b_ + a * d_)
        delta_h = (b_ * d_ - 1.0) * delta_h
        h = h + delta_h
        delta_s = q * delta_h
        s = s + delta_s
        converged = (jnp.abs(delta_s) <= eps * jnp.abs(s)) & (
            jnp.abs(delta_h) <= eps * jnp.abs(h)
        )
        updated = (a, b_, d_, delta_h, h, q, lower, upper, s)
        a, b_, d_, delta_h, h, q, lower, upper, s = (
            jnp.where(done, old, new) for old, new in zip(current, updated, strict=True)
        )
        return (a, b_, d_, delta_h, h, q, lower, upper, s), done | converged

    (_, _, _, _, h, _, _, _, s), _ = jax.lax.fori_loop(
        2,
        _KV_STEED_STEPS + 2,
        iterate,
        (state, jnp.zeros(argument.shape, dtype=jnp.bool_)),
    )
    scaled = jnp.sqrt(0.5 * math.pi / argument) / s
    return scaled, scaled * (mu + argument + 0.5 - a1 * h) / argument


def _kve_right_half_plane_pair(order: Array, argument: Array, /) -> tuple[Array, Array]:
    """Scaled ``e^z K_v`` and ``e^z K_{v+1}`` for ``Re v >= 0`` and ``Re z >= 0``.

    The reduced order ``mu = v - n`` with ``|Re mu| <= 1/2`` is evaluated by the
    Temme series for ``|z| <= 2`` and by CF2 otherwise, followed by the stable
    upward recurrence ``K_{k+1} = (2k/z) K_k + K_{k-1}`` masked to ``n`` steps.
    """
    count = jnp.round(jnp.real(order))
    mu = order - count.astype(order.dtype)
    small = jnp.abs(argument) <= _KV_TEMME_RADIUS
    temme_argument = jnp.where(small, argument, jnp.ones_like(argument))
    steed_argument = jnp.where(small, 2.0 * _KV_TEMME_RADIUS, argument)
    temme_value, temme_next = _kv_temme_series(mu, temme_argument)
    temme_scale = jnp.exp(temme_argument)
    steed_value, steed_next = _kve_steed(mu, steed_argument)
    value = jnp.where(small, temme_scale * temme_value, steed_value)
    following = jnp.where(small, temme_scale * temme_next, steed_next)
    two_over_argument = 2.0 / argument

    def recur(index: Array, pair: tuple[Array, Array]) -> tuple[Array, Array]:
        lower, upper = pair
        active = index.astype(count.dtype) < count
        k = index.astype(argument.dtype) + 1.0
        raised = (mu + k) * two_over_argument * upper + lower
        return jnp.where(active, upper, lower), jnp.where(active, raised, upper)

    return jax.lax.fori_loop(0, _KV_MAX_RECURRENCE, recur, (value, following))


@jax.custom_jvp
def _kve_right_half_plane(order: Array, argument: Array, /) -> Array:
    return _kve_right_half_plane_pair(order, argument)[0]


def _kve_right_half_plane_jvp(
    primals: tuple[Array, Array],
    tangents: tuple[Array | SymbolicZero, Array | SymbolicZero],
) -> tuple[Array, Array]:
    order, argument = primals
    order_tangent, argument_tangent = tangents
    if isinstance(order_tangent, SymbolicZero):
        value, following = _kve_right_half_plane_pair(order, argument)
        tangent = jnp.zeros_like(value)
    else:
        (value, following), (tangent, _) = jax.jvp(
            lambda v: _kve_right_half_plane_pair(v, argument),
            (order,),
            (order_tangent,),
        )
    if not isinstance(argument_tangent, SymbolicZero):
        # d(e^z K_v)/dz = e^z (K_v + K_v') with K_v' = (v/z) K_v - K_{v+1}.
        derivative = value * (1.0 + order / argument) - following
        tangent = tangent + derivative * argument_tangent
    return value, tangent


_kve_right_half_plane.defjvp(_kve_right_half_plane_jvp, symbolic_zeros=True)


def _kv_parts(
    order: ArrayLike, argument: ArrayLike, /
) -> tuple[Array, Array, Array, Array]:
    """Split principal ``K_v(z)`` into the Temme/CF2 domain and the legacy series.

    Returns ``(z, supported, scaled, series)`` with the promoted broadcast
    argument ``z``, ``scaled = e^z K_v(z)`` on the supported lanes
    (``Re z >= 0``, ``|Re v| <= _KV_MAX_RECURRENCE + 1/2``,
    ``|Im v| <= _KV_MAX_IMAGINARY_ORDER``) and ``series`` the unscaled
    power-series connection formula on the remaining lanes (left half plane,
    very large or strongly complex orders), whose behavior is unchanged.
    """
    order_, argument_ = jnp.broadcast_arrays(*promote_principal(order, argument))
    reflected = jnp.where(jnp.real(order_) < 0.0, -order_, order_)
    supported = (
        (jnp.real(argument_) >= 0.0)
        & (jnp.real(reflected) <= _KV_MAX_RECURRENCE + 0.5)
        & (jnp.abs(jnp.imag(reflected)) <= _KV_MAX_IMAGINARY_ORDER)
    )
    scaled = _kve_right_half_plane(
        jnp.where(supported, reflected, jnp.zeros_like(reflected)),
        jnp.where(supported, argument_, jnp.ones_like(argument_)),
    )
    unsupported = ~supported
    series = jax.lax.cond(
        jnp.any(unsupported),
        lambda: _kv_power_series(
            jnp.where(unsupported, order_, jnp.full_like(order_, 0.5)),
            jnp.where(unsupported, argument_, jnp.ones_like(argument_)),
        ),
        lambda: jnp.zeros_like(argument_),
    )
    return argument_, supported, scaled, series


def complex_kv(order: ArrayLike, argument: ArrayLike, /) -> Array:
    argument_, supported, scaled, series = _kv_parts(order, argument)
    return jnp.where(supported, jnp.exp(-argument_) * scaled, series)


def complex_kve(order: ArrayLike, argument: ArrayLike, /) -> Array:
    argument_, supported, scaled, series = _kv_parts(order, argument)
    return jnp.where(supported, scaled, jnp.exp(argument_) * series)


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
    order_, argument_ = promote_principal(order, argument)
    return _order_derivative(lambda value, z: complex_kve(value, z), order_, argument_)


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


def _carlson_steps(dtype: DTypeLike) -> int:
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
    running = jnp.ones_like(u, dtype=jnp.bool_)
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
    "complex_kve",
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
