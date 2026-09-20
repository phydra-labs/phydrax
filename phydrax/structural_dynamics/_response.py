#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
import jax.numpy as jnp
from jaxtyping import ArrayLike


def sdof_transfer_function(
    angular_frequency_rad_s: ArrayLike,
    natural_frequency_rad_s: float,
    damping_ratio: float,
    /,
):
    r = jnp.asarray(angular_frequency_rad_s) / float(natural_frequency_rad_s)
    return 1 / (1 - r * r + 2j * float(damping_ratio) * r)


def random_response_variance(
    transfer_function: ArrayLike, input_psd: ArrayLike, frequency_spacing_hz: float, /
):
    return jnp.sum(
        jnp.abs(jnp.asarray(transfer_function)) ** 2 * jnp.asarray(input_psd)
    ) * float(frequency_spacing_hz)


def shock_response_peak(acceleration: ArrayLike, /):
    return jnp.max(jnp.abs(jnp.asarray(acceleration)))


__all__ = ["random_response_variance", "sdof_transfer_function", "shock_response_peak"]
