#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from math import isfinite

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import ArrayLike


def sdof_transfer_function(
    angular_frequency_rad_s: ArrayLike,
    natural_frequency_rad_s: float,
    damping_ratio: float,
    /,
):
    if (
        not isfinite(natural_frequency_rad_s)
        or natural_frequency_rad_s <= 0
        or not isfinite(damping_ratio)
        or damping_ratio < 0
    ):
        raise ValueError(
            "SDOF frequency must be finite/positive and damping finite/nonnegative."
        )
    frequency = jnp.asarray(angular_frequency_rad_s)
    frequency = eqx.error_if(
        frequency,
        jnp.any(~jnp.isfinite(frequency) | (frequency < 0)),
        "Excitation frequency must be finite and nonnegative.",
    )
    r = frequency / float(natural_frequency_rad_s)
    return 1 / (1 - r * r + 2j * float(damping_ratio) * r)


def random_response_variance(
    transfer_function: ArrayLike, input_psd: ArrayLike, frequency_spacing_hz: float, /
):
    if not isfinite(frequency_spacing_hz) or frequency_spacing_hz <= 0:
        raise ValueError("Frequency spacing must be finite and positive.")
    transfer = jnp.asarray(transfer_function)
    psd = jnp.asarray(input_psd)
    if transfer.shape != psd.shape:
        raise ValueError("Transfer function and input PSD must have matching shapes.")
    psd = eqx.error_if(
        psd,
        jnp.any(~jnp.isfinite(transfer) | ~jnp.isfinite(psd) | (psd < 0)),
        "Transfer function and PSD must be finite with nonnegative PSD.",
    )
    return jnp.sum(jnp.abs(transfer) ** 2 * psd) * float(frequency_spacing_hz)


def shock_response_peak(acceleration: ArrayLike, /):
    values = jnp.asarray(acceleration)
    values = eqx.error_if(
        values,
        jnp.any(~jnp.isfinite(values)) | (values.size == 0),
        "Shock acceleration must be finite and nonempty.",
    )
    return jnp.max(jnp.abs(values))


__all__ = ["random_response_variance", "sdof_transfer_function", "shock_response_peak"]
