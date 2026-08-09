#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The Faddeeva and Dawson kernels are adapted from JAX 0.11.0.
# See NOTICE and LICENSES/JAX-APACHE-2.0.txt.
#

"""Faddeeva, Dawson, and Voigt functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike

from ._dtype import promote_complex, promote_real


# Weideman (1994), N=32. Coefficients are ordered for ``jnp.polyval``.
_WOFZ_L = 4.7568284600108841
_WOFZ_C = (
    -1.3034426067909105e-12,
    3.7411838373738471e-12,
    8.030427700497756e-12,
    -2.1543593557490879e-11,
    -5.5442449963237932e-11,
    1.165824698850374e-10,
    4.1537441121999766e-10,
    -5.2310202114920615e-10,
    -3.2080151339721323e-09,
    8.1248864216535907e-10,
    2.3797556530014025e-08,
    2.2930438438915445e-08,
    -1.4813078929137642e-07,
    -4.1840763750512053e-07,
    4.2558331397138446e-07,
    4.4015317312832251e-06,
    6.8210319443575151e-06,
    -2.1409619201999998e-05,
    -1.3075449254579421e-04,
    -2.4532980270038237e-04,
    3.9259136070109705e-04,
    4.5195411053493093e-03,
    1.9006155784845501e-02,
    5.7304403529837282e-02,
    1.4060716226893755e-01,
    2.9544451071508743e-01,
    5.4601397206393376e-01,
    9.019254893648001e-01,
    1.3455441692345449,
    1.8256696296324815,
    2.2635372999002676,
    2.5722534081245696,
)

# Cody, Paciorek, and Thacher (1970), three rational Dawson regimes.
_DAWSN_AN = (
    1.13681498971755972054e-11,
    8.49262267667473811108e-10,
    1.94434204175553054283e-08,
    9.53151741254484363489e-07,
    3.07828309874913200438e-06,
    3.52513368520288738649e-04,
    -8.50149846724410912031e-04,
    4.22618223005546594270e-02,
    -9.17480371773452345351e-02,
    9.99999999999999994612e-01,
)
_DAWSN_AD = (
    2.40372073066762605484e-11,
    1.48864681368493396752e-09,
    5.21265281010541664570e-08,
    1.27258478273186970203e-06,
    2.32490249820789513991e-05,
    3.25524741826057911661e-04,
    3.48805814657162590916e-03,
    2.79448531198828973716e-02,
    1.58874241960120565368e-01,
    5.74918629489320327824e-01,
    1.00000000000000000539,
)
_DAWSN_BN = (
    5.08955156417900903354e-01,
    -2.44754418142697847934e-01,
    9.41512335303534411857e-02,
    -2.18711255142039025206e-02,
    3.66207612329569181322e-03,
    -4.23209114460388756528e-04,
    3.59641304793896631888e-05,
    -2.14640351719968974225e-06,
    9.10010780076391431042e-08,
    -2.40274520828250956942e-09,
    3.59233385440928410398e-11,
)
_DAWSN_BD = (
    1.00000000000000000000,
    -6.31839869873368190192e-01,
    2.36706788228248691528e-01,
    -5.31806367003223277662e-02,
    8.48041718586295374409e-03,
    -9.47996768486665330168e-04,
    7.81025592944552338085e-05,
    -4.55875153252442634831e-06,
    1.89100358111421846170e-07,
    -4.91324691331920606875e-09,
    7.18466403235734541950e-11,
)
_DAWSN_CN = (
    -5.90592860534773254987e-01,
    6.29235242724368800674e-01,
    -1.72858975380388136411e-01,
    1.64837047825189632310e-02,
    -4.86827613020462700845e-04,
)
_DAWSN_CD = (
    1.00000000000000000000,
    -2.69820057197544900361,
    1.73270799045947845857,
    -3.93708582281939493482e-01,
    3.44278924041233391079e-02,
    -9.73655226040941223894e-04,
)


def _constant(reference: Array, value: float, /) -> Array:
    return jnp.asarray(value, dtype=reference.dtype)


def _wofz_upper(z: Array, /) -> Array:
    real = jnp.real(z)
    length = jax.lax.complex(_constant(real, _WOFZ_L), jnp.zeros_like(real))
    imaginary_z = jax.lax.complex(-jnp.imag(z), real)
    denominator = length - imaginary_z
    transformed = (length + imaginary_z) / denominator
    polynomial = jnp.polyval(jnp.asarray(_WOFZ_C, dtype=z.dtype), transformed)
    inverse_sqrt_pi = jax.lax.complex(
        _constant(real, 1.0 / math.sqrt(math.pi)), jnp.zeros_like(real)
    )
    return 2.0 * polynomial / denominator**2 + inverse_sqrt_pi / denominator


@jax.custom_jvp
def _wofz(z: Array, /) -> Array:
    upper_half_plane = jnp.imag(z) >= _constant(jnp.real(z), 0.0)
    upper_argument = jnp.where(upper_half_plane, z, -z)
    upper_argument_infinite = jnp.isinf(jnp.real(upper_argument)) | jnp.isinf(
        jnp.imag(upper_argument)
    )
    safe_upper_argument = jnp.where(
        upper_argument_infinite, jnp.zeros_like(upper_argument), upper_argument
    )
    upper_value = jnp.where(
        upper_argument_infinite,
        jnp.zeros_like(upper_argument),
        _wofz_upper(safe_upper_argument),
    )

    # Guard the unused exponential against overflow in the upper half-plane.
    lower_argument = jnp.where(upper_half_plane, jnp.zeros_like(z), z)
    reflected_value = 2.0 * jnp.exp(-(lower_argument**2)) - upper_value
    value = jnp.where(upper_half_plane, upper_value, reflected_value)
    real = jnp.real(z)
    imaginary = jnp.imag(z)
    non_nan = ~jnp.isnan(real) & ~jnp.isnan(imaginary)
    decaying_infinite = non_nan & (
        jnp.isinf(real) | (jnp.isinf(imaginary) & (imaginary > 0.0))
    )
    negative_imaginary_axis_infinity = (
        non_nan & (real == 0.0) & jnp.isinf(imaginary) & (imaginary < 0.0)
    )
    value = jnp.where(decaying_infinite, jnp.zeros_like(value), value)
    divergent_limit = jax.lax.complex(jnp.full_like(real, jnp.inf), jnp.zeros_like(real))
    return jnp.where(negative_imaginary_axis_infinity, divergent_limit, value)


@_wofz.defjvp
def _wofz_jvp(primals, tangents):
    (z,) = primals
    (z_tangent,) = tangents
    value = _wofz(z)
    derivative = -2.0 * z * value + jnp.asarray(2.0j / math.sqrt(math.pi), dtype=z.dtype)
    real = jnp.real(z)
    imaginary = jnp.imag(z)
    decaying_infinite = (
        ~jnp.isnan(real)
        & ~jnp.isnan(imaginary)
        & (jnp.isinf(real) | (jnp.isinf(imaginary) & (imaginary > 0.0)))
    )
    derivative = jnp.where(decaying_infinite, jnp.zeros_like(derivative), derivative)
    return value, derivative * z_tangent


def wofz(z: ArrayLike, /) -> Array:
    """Evaluate the Faddeeva function ``exp(-z**2) * erfc(-1j*z)``.

    The implementation is JAX-transformable and supports real or complex
    scalar and array inputs. Real inputs return complex values. Mathematical
    overflow in the lower half-plane is retained rather than clipped.
    """
    return _wofz(promote_complex(z))


def _dawsn_impl(x: Array, /) -> Array:
    sign = jnp.sign(x)
    absolute_x = jnp.abs(x)
    absolute_x_squared = jnp.square(absolute_x)
    safe_reciprocal_argument = jnp.where(
        absolute_x > _constant(x, 0.0), absolute_x_squared, jnp.ones_like(x)
    )
    reciprocal_square = _constant(x, 1.0) / safe_reciprocal_argument

    coefficients = tuple(
        jnp.asarray(values, dtype=x.dtype)
        for values in (
            _DAWSN_AN,
            _DAWSN_AD,
            _DAWSN_BN,
            _DAWSN_BD,
            _DAWSN_CN,
            _DAWSN_CD,
        )
    )
    an, ad, bn, bd, cn, cd = coefficients

    first_region = absolute_x < _constant(x, 3.25)
    safe_x_first = jnp.where(first_region, absolute_x, jnp.ones_like(x))
    safe_x_first_squared = jnp.square(safe_x_first)
    value_first = (
        safe_x_first
        * jnp.polyval(an, safe_x_first_squared)
        / jnp.polyval(ad, safe_x_first_squared)
    )

    second_region = (absolute_x >= _constant(x, 3.25)) & (absolute_x < _constant(x, 6.25))
    safe_t_second = jnp.where(second_region, reciprocal_square, jnp.ones_like(x))
    safe_x_second = jnp.where(second_region, absolute_x, jnp.ones_like(x))
    value_second = (_constant(x, 0.5) / safe_x_second) * (
        _constant(x, 1.0)
        + safe_t_second * jnp.polyval(bn, safe_t_second) / jnp.polyval(bd, safe_t_second)
    )

    third_region = absolute_x >= _constant(x, 6.25)
    safe_t_third = jnp.where(third_region, reciprocal_square, jnp.ones_like(x))
    safe_x_third = jnp.where(third_region, absolute_x, jnp.ones_like(x))
    value_third = (_constant(x, 0.5) / safe_x_third) * (
        _constant(x, 1.0)
        + safe_t_third * jnp.polyval(cn, safe_t_third) / jnp.polyval(cd, safe_t_third)
    )

    value = jnp.where(
        first_region,
        value_first,
        jnp.where(second_region, value_second, value_third),
    )
    return sign * value


@jax.custom_jvp
def _dawsn(x: Array, /) -> Array:
    return _dawsn_impl(x)


@_dawsn.defjvp
def _dawsn_jvp(primals, tangents):
    (x,) = primals
    (x_tangent,) = tangents
    value = _dawsn(x)
    derivative = _constant(x, 1.0) - _constant(x, 2.0) * x * value
    derivative = jnp.where(jnp.isinf(x), jnp.zeros_like(derivative), derivative)
    return value, derivative * x_tangent


def dawsn(x: ArrayLike, /) -> Array:
    """Evaluate Dawson's integral for real scalar or array inputs.

    Dawson's integral is ``exp(-x**2) * integral(exp(t**2), t=0..x)``.
    Complex inputs are rejected. Float16 and bfloat16 inputs are evaluated and
    returned as float32; float32 and float64 are preserved.
    """
    (promoted_x,) = promote_real("dawsn", x)
    return _dawsn(promoted_x)


def _voigt_profile_primal(x: Array, sigma: Array, gamma: Array, /) -> Array:
    valid_general = (sigma > 0.0) & (gamma >= 0.0)
    safe_x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    safe_sigma = jnp.where(valid_general, sigma, jnp.ones_like(sigma))
    safe_gamma = jnp.where(valid_general, gamma, jnp.zeros_like(gamma))
    denominator = safe_sigma * _constant(x, math.sqrt(2.0))
    z = jax.lax.complex(safe_x / denominator, safe_gamma / denominator)
    general = jnp.real(_wofz(z)) / (safe_sigma * _constant(x, math.sqrt(2.0 * math.pi)))
    gaussian = jnp.exp(-0.5 * (safe_x / safe_sigma) ** 2) / (
        safe_sigma * _constant(x, math.sqrt(2.0 * math.pi))
    )
    continuous_value = jnp.where(gamma == 0.0, gaussian, general)

    valid_cauchy = (sigma == 0.0) & (gamma > 0.0)
    cauchy_gamma = jnp.where(valid_cauchy, gamma, jnp.ones_like(gamma))
    ratio = safe_x / cauchy_gamma
    cauchy = _constant(x, 1.0) / (_constant(x, math.pi) * cauchy_gamma * (1.0 + ratio**2))
    point_mass = jnp.where(
        x == 0.0,
        jnp.full_like(x, jnp.inf),
        jnp.zeros_like(x),
    )

    value = jnp.where(
        sigma == 0.0,
        jnp.where(gamma == 0.0, point_mass, cauchy),
        continuous_value,
    )
    invalid = (
        (sigma < 0.0) | (gamma < 0.0) | jnp.isnan(x) | jnp.isnan(sigma) | jnp.isnan(gamma)
    )
    return jnp.where(invalid, jnp.full_like(value, jnp.nan), value)


@jax.custom_jvp
def _voigt_profile(x: Array, sigma: Array, gamma: Array, /) -> Array:
    return _voigt_profile_primal(x, sigma, gamma)


@_voigt_profile.defjvp
def _voigt_profile_jvp(primals, tangents):
    x, sigma, gamma = primals
    x_tangent, sigma_tangent, gamma_tangent = tangents
    value = _voigt_profile(x, sigma, gamma)

    interior = (sigma > 0.0) & (gamma >= 0.0)
    safe_x = jnp.where(jnp.isnan(x), jnp.zeros_like(x), x)
    safe_sigma = jnp.where(interior, sigma, jnp.ones_like(sigma))
    safe_gamma = jnp.where(interior, gamma, jnp.zeros_like(gamma))
    denominator = safe_sigma * _constant(x, math.sqrt(2.0))
    z = jax.lax.complex(safe_x / denominator, safe_gamma / denominator)
    faddeeva = _wofz(z)
    faddeeva_derivative = -2.0 * z * faddeeva + jnp.asarray(
        2.0j / math.sqrt(math.pi), dtype=z.dtype
    )
    asymptotic_zero = jnp.isinf(jnp.real(z)) | jnp.isinf(jnp.imag(z))
    faddeeva_derivative = jnp.where(
        asymptotic_zero, jnp.zeros_like(faddeeva_derivative), faddeeva_derivative
    )
    x_derivative = jnp.real(faddeeva_derivative) / (
        _constant(x, 2.0 * math.sqrt(math.pi)) * safe_sigma**2
    )
    gamma_derivative = -jnp.imag(faddeeva_derivative) / (
        _constant(x, 2.0 * math.sqrt(math.pi)) * safe_sigma**2
    )
    z_times_derivative = jnp.where(
        asymptotic_zero, jnp.zeros_like(z), z * faddeeva_derivative
    )
    sigma_derivative = -value / safe_sigma - jnp.real(z_times_derivative) / (
        _constant(x, math.sqrt(2.0 * math.pi)) * safe_sigma**2
    )
    gaussian_boundary = interior & (gamma == 0.0)
    gaussian_asymptotic = gaussian_boundary & jnp.logical_xor(
        jnp.isinf(safe_x), jnp.isinf(safe_sigma)
    )
    gaussian_x_derivative = -safe_x * value / safe_sigma**2
    gaussian_sigma_derivative = value * (safe_x**2 / safe_sigma**3 - 1.0 / safe_sigma)
    x_derivative = jnp.where(
        gaussian_boundary,
        jnp.where(gaussian_asymptotic, jnp.zeros_like(value), gaussian_x_derivative),
        x_derivative,
    )
    sigma_derivative = jnp.where(
        gaussian_boundary,
        jnp.where(
            gaussian_asymptotic,
            jnp.zeros_like(value),
            gaussian_sigma_derivative,
        ),
        sigma_derivative,
    )
    cauchy_region = (sigma == 0.0) & (gamma > 0.0)
    cauchy_gamma = jnp.where(cauchy_region, gamma, jnp.ones_like(gamma))
    ratio = safe_x / cauchy_gamma
    cauchy_denominator = _constant(x, math.pi) * cauchy_gamma**2 * (1.0 + ratio**2) ** 2
    cauchy_asymptotic = jnp.logical_xor(jnp.isinf(safe_x), jnp.isinf(cauchy_gamma))
    cauchy_x_derivative = jnp.where(
        cauchy_asymptotic,
        jnp.zeros_like(value),
        -2.0 * ratio / cauchy_denominator,
    )
    cauchy_gamma_derivative = jnp.where(
        cauchy_asymptotic,
        jnp.zeros_like(value),
        (ratio**2 - 1.0) / cauchy_denominator,
    )

    undefined_derivative = jnp.full_like(value, jnp.nan)
    x_derivative = jnp.where(
        interior,
        x_derivative,
        jnp.where(cauchy_region, cauchy_x_derivative, undefined_derivative),
    )
    sigma_derivative = jnp.where(
        interior,
        sigma_derivative,
        jnp.where(cauchy_region, jnp.zeros_like(value), undefined_derivative),
    )
    gamma_derivative = jnp.where(
        interior,
        gamma_derivative,
        jnp.where(cauchy_region, cauchy_gamma_derivative, undefined_derivative),
    )
    invalid = (
        (sigma < 0.0) | (gamma < 0.0) | jnp.isnan(x) | jnp.isnan(sigma) | jnp.isnan(gamma)
    )
    x_derivative = jnp.where(invalid, undefined_derivative, x_derivative)
    sigma_derivative = jnp.where(invalid, undefined_derivative, sigma_derivative)
    gamma_derivative = jnp.where(invalid, undefined_derivative, gamma_derivative)
    tangent = (
        x_derivative * x_tangent
        + sigma_derivative * sigma_tangent
        + gamma_derivative * gamma_tangent
    )
    return value, tangent


def voigt_profile(
    x: ArrayLike,
    sigma: ArrayLike,
    gamma: ArrayLike,
    /,
) -> Array:
    """Evaluate the normalized Voigt line profile.

    ``sigma`` is the Gaussian standard deviation and ``gamma`` is the Cauchy
    half-width at half-maximum. Inputs broadcast to a common shape. Negative
    scale parameters return NaN. When exactly one scale is zero, the matching
    Gaussian or Cauchy density is returned; both zero give infinity at ``x=0``
    and zero elsewhere.
    """
    promoted_x, promoted_sigma, promoted_gamma = promote_real(
        "voigt_profile", x, sigma, gamma
    )
    broadcast_x, broadcast_sigma, broadcast_gamma = jnp.broadcast_arrays(
        promoted_x, promoted_sigma, promoted_gamma
    )
    return _voigt_profile(broadcast_x, broadcast_sigma, broadcast_gamma)
