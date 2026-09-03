#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._axis import _normalize_axis, _promote_signal_and_taps


ConvolutionMode: TypeAlias = Literal["full", "same", "valid"]
ConvolutionMethod: TypeAlias = Literal["direct", "fft"]


def _full_direct(values: Array, taps: Array, /) -> Array:
    sample_count = int(values.shape[-1])
    flattened = values.reshape((-1, sample_count))
    output = jax.vmap(lambda stream: jnp.convolve(stream, taps, mode="full"))(flattened)
    return output.reshape((*values.shape[:-1], output.shape[-1]))


def _full_fft(values: Array, taps: Array, /) -> Array:
    sample_count = int(values.shape[-1])
    tap_count = int(taps.shape[0])
    output_length = sample_count + tap_count - 1
    fft_length = 1 << (output_length - 1).bit_length()
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        transformed_values = jnp.fft.fft(values, n=fft_length, axis=-1)
        transformed_taps = jnp.fft.fft(taps, n=fft_length)
        output = jnp.fft.ifft(
            transformed_values * transformed_taps,
            n=fft_length,
            axis=-1,
        )
    else:
        transformed_values = jnp.fft.rfft(values, n=fft_length, axis=-1)
        transformed_taps = jnp.fft.rfft(taps, n=fft_length)
        output = jnp.fft.irfft(
            transformed_values * transformed_taps,
            n=fft_length,
            axis=-1,
        )
    return output[..., :output_length]


def convolve(
    values: ArrayLike,
    taps: ArrayLike,
    /,
    *,
    axis: int = -1,
    mode: ConvolutionMode = "full",
    method: ConvolutionMethod = "direct",
) -> Array:
    """Convolve independent streams with one shared one-dimensional kernel.

    ``same`` always returns the input signal length and starts at
    ``(tap_count - 1) // 2`` in the full convolution.
    """
    array, coefficients = _promote_signal_and_taps(values, taps)
    resolved_axis = _normalize_axis(axis, array.ndim)
    canonical = jnp.moveaxis(array, resolved_axis, -1)
    sample_count = int(canonical.shape[-1])
    tap_count = int(coefficients.shape[0])
    if sample_count <= 0:
        raise ValueError("The signal axis must contain at least one sample.")
    if mode not in ("full", "same", "valid"):
        raise ValueError("mode must be 'full', 'same', or 'valid'.")
    if method == "direct":
        full = _full_direct(canonical, coefficients)
    elif method == "fft":
        full = _full_fft(canonical, coefficients)
    else:
        raise ValueError("method must be 'direct' or 'fft'.")

    if mode == "full":
        output = full
    elif mode == "same":
        start = (tap_count - 1) // 2
        output = full[..., start : start + sample_count]
    else:
        if sample_count < tap_count:
            raise ValueError("valid convolution requires signal length >= tap count.")
        start = tap_count - 1
        output = full[..., start : start + sample_count - tap_count + 1]
    return jnp.moveaxis(output, -1, resolved_axis)


__all__ = [
    "ConvolutionMethod",
    "ConvolutionMode",
    "convolve",
]
