#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def harmonic_balance_residual(
    coefficients: ArrayLike, base_angular_frequency_rad_s: float, rhs, /
):
    coefficients_ = jnp.asarray(coefficients)
    sample_count = coefficients_.shape[0]
    state = jnp.fft.ifft(coefficients_, axis=0)
    times = (
        2
        * jnp.pi
        * jnp.arange(sample_count)
        / (sample_count * float(base_angular_frequency_rad_s))
    )
    rhs_values = jnp.stack(tuple(rhs(times[i], state[i]) for i in range(sample_count)))
    rhs_coefficients = jnp.fft.fft(rhs_values, axis=0)
    harmonic_indices = jnp.fft.fftfreq(sample_count) * sample_count
    derivative = (
        1j
        * float(base_angular_frequency_rad_s)
        * harmonic_indices[(...,) + (None,) * (coefficients_.ndim - 1)]
        * coefficients_
    )
    return derivative - rhs_coefficients


__all__ = ["harmonic_balance_residual"]
