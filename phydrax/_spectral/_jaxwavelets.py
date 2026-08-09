#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
import jaxwavelets
from jaxtyping import Array


BackendBoundary = Literal["periodization", "symmetric", "constant"]


def load_filter_taps(name: str, /) -> tuple[Array, Array, Array, Array]:
    """Load one standard filter bank without leaking the backend value type."""
    wavelet = jaxwavelets.get_wavelet(str(name))
    return tuple(jnp.asarray(taps) for taps in wavelet)  # type: ignore[return-value]


def _backend_wavelet(taps: tuple[Array, Array, Array, Array], /):
    return jaxwavelets.Wavelet(*taps)


def dwt_axis(
    values: Array,
    taps: tuple[Array, Array, Array, Array],
    boundary: BackendBoundary,
    axis: int,
    /,
) -> tuple[Array, Array]:
    """Apply the backend's one-dimensional DWT along one array axis."""
    moved = jnp.moveaxis(values, axis, -1)
    flattened = moved.reshape((-1, moved.shape[-1]))
    wavelet = _backend_wavelet(taps)
    low, high = jax.vmap(lambda row: jaxwavelets.dwt(row, wavelet, boundary))(
        flattened
    )
    low = low.reshape(moved.shape[:-1] + (low.shape[-1],))
    high = high.reshape(moved.shape[:-1] + (high.shape[-1],))
    return jnp.moveaxis(low, -1, axis), jnp.moveaxis(high, -1, axis)


def idwt_axis(
    low: Array,
    high: Array,
    taps: tuple[Array, Array, Array, Array],
    boundary: BackendBoundary,
    axis: int,
    /,
) -> Array:
    """Apply the backend's one-dimensional inverse DWT along one array axis."""
    moved_low = jnp.moveaxis(low, axis, -1)
    moved_high = jnp.moveaxis(high, axis, -1)
    if moved_low.shape != moved_high.shape:
        raise ValueError(
            "Low- and high-pass coefficient arrays must have matching shapes."
        )
    wavelet = _backend_wavelet(taps)
    reconstructed = jax.vmap(
        lambda low_row, high_row: jaxwavelets.idwt(
            low_row, high_row, wavelet, boundary
        )
    )(
        moved_low.reshape((-1, moved_low.shape[-1])),
        moved_high.reshape((-1, moved_high.shape[-1])),
    )
    output = reconstructed.reshape(
        moved_low.shape[:-1] + (reconstructed.shape[-1],)
    )
    return jnp.moveaxis(output, -1, axis)


__all__ = ["BackendBoundary", "dwt_axis", "idwt_axis", "load_filter_taps"]
