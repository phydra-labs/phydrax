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
# The Faddeeva kernel is adapted from JAX 0.11.0.
# See NOTICE and LICENSES/JAX-APACHE-2.0.txt.
#

"""Faddeeva, Dawson, and Voigt functions."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
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
def _wofz_jvp(primals: tuple[Array], tangents: tuple[Array]) -> tuple[Array, Array]:
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


def _dawsn(x: Array, /) -> Array:
    # JAX owns the finite rational kernel. Its derivative rule is NaN at
    # infinity, so the signed-zero limits F(+-inf) = +-0 and F'(+-inf) = 0 are
    # selected here; infinite lanes never reach the kernel.
    infinite = jnp.isinf(x)
    finite_x = jnp.where(infinite, jnp.zeros_like(x), x)
    limit = jnp.copysign(jnp.zeros_like(x), x)
    return jnp.where(infinite, limit, jsp.dawsn(finite_x))


def dawsn(x: ArrayLike, /) -> Array:
    """Evaluate Dawson's integral on its principal complex continuation."""
    if jnp.issubdtype(jnp.asarray(x).dtype, jnp.complexfloating):
        from ._continuation import complex_dawsn

        return complex_dawsn(x)
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
def _voigt_profile_jvp(
    primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
) -> tuple[Array, Array]:
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
