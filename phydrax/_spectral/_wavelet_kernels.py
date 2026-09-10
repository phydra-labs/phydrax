#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._wavelet_catalog import get_wavelet


WaveletBoundary = Literal["periodization", "symmetric", "constant"]

_FILTERS: dict[str, tuple[tuple[float, ...], ...]] = {
    "haar": (
        (0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
    ),
    "db1": (
        (0.7071067811865476, 0.7071067811865476),
        (-0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, 0.7071067811865476),
        (0.7071067811865476, -0.7071067811865476),
    ),
    "db2": (
        (
            -0.12940952255126037,
            0.2241438680420134,
            0.8365163037378079,
            0.48296291314453416,
        ),
        (
            -0.48296291314453416,
            0.8365163037378079,
            -0.2241438680420134,
            -0.12940952255126037,
        ),
        (
            0.48296291314453416,
            0.8365163037378079,
            0.2241438680420134,
            -0.12940952255126037,
        ),
        (
            -0.12940952255126037,
            -0.2241438680420134,
            0.8365163037378079,
            -0.48296291314453416,
        ),
    ),
    "db4": (
        (
            -0.010597401785069032,
            0.0328830116668852,
            0.030841381835560764,
            -0.18703481171909309,
            -0.027983769416859854,
            0.6308807679298589,
            0.7148465705529157,
            0.2303778133088965,
        ),
        (
            -0.2303778133088965,
            0.7148465705529157,
            -0.6308807679298589,
            -0.027983769416859854,
            0.18703481171909309,
            0.030841381835560764,
            -0.0328830116668852,
            -0.010597401785069032,
        ),
        (
            0.2303778133088965,
            0.7148465705529157,
            0.6308807679298589,
            -0.027983769416859854,
            -0.18703481171909309,
            0.030841381835560764,
            0.0328830116668852,
            -0.010597401785069032,
        ),
        (
            -0.010597401785069032,
            -0.0328830116668852,
            0.030841381835560764,
            0.18703481171909309,
            -0.027983769416859854,
            -0.6308807679298589,
            0.7148465705529157,
            -0.2303778133088965,
        ),
    ),
    "sym4": (
        (
            -0.07576571478927333,
            -0.02963552764599851,
            0.49761866763201545,
            0.8037387518059161,
            0.29785779560527736,
            -0.09921954357684722,
            -0.012603967262037833,
            0.0322231006040427,
        ),
        (
            -0.0322231006040427,
            -0.012603967262037833,
            0.09921954357684722,
            0.29785779560527736,
            -0.8037387518059161,
            0.49761866763201545,
            0.02963552764599851,
            -0.07576571478927333,
        ),
        (
            0.0322231006040427,
            -0.012603967262037833,
            -0.09921954357684722,
            0.29785779560527736,
            0.8037387518059161,
            0.49761866763201545,
            -0.02963552764599851,
            -0.07576571478927333,
        ),
        (
            -0.07576571478927333,
            0.02963552764599851,
            0.49761866763201545,
            -0.8037387518059161,
            0.29785779560527736,
            0.09921954357684722,
            -0.012603967262037833,
            -0.0322231006040427,
        ),
    ),
}


def load_filter_taps(name: str, /) -> tuple[Array, Array, Array, Array]:
    """Return one canonical native orthogonal-wavelet filter bank."""
    key = str(name).lower()
    if key not in _FILTERS:
        wavelet = get_wavelet(key)
        return (
            jnp.asarray(wavelet.dec_lo),
            jnp.asarray(wavelet.dec_hi),
            jnp.asarray(wavelet.rec_lo),
            jnp.asarray(wavelet.rec_hi),
        )
    taps = _FILTERS[key]
    return tuple(jnp.asarray(values) for values in taps)  # type: ignore[return-value]


def _dwt_row(
    values: Array,
    low_filter: Array,
    high_filter: Array,
    boundary: WaveletBoundary,
    /,
) -> tuple[Array, Array]:
    filter_length = int(low_filter.shape[0])
    if boundary == "periodization":
        if values.shape[0] % 2:
            values = jnp.concatenate((values, values[-1:]))
        padded = jnp.pad(
            values,
            (filter_length // 2 - 1, filter_length // 2 - 1),
            mode="wrap",
        )
        count = (values.shape[0] + 1) // 2
        low = jnp.convolve(padded, low_filter, mode="valid")[::2][:count]
        high = jnp.convolve(padded, high_filter, mode="valid")[::2][:count]
        return low, high
    padded = jnp.pad(
        values,
        (filter_length - 2, filter_length - 1),
        mode=boundary,
    )
    return (
        jnp.convolve(padded, low_filter, mode="valid")[::2],
        jnp.convolve(padded, high_filter, mode="valid")[::2],
    )


def dwt_axis(
    values: Array,
    taps: tuple[Array, Array, Array, Array],
    boundary: WaveletBoundary,
    axis: int,
    /,
) -> tuple[Array, Array]:
    """Apply a native one-dimensional DWT along one array axis."""
    moved = jnp.moveaxis(values, axis, -1)
    rows = moved.reshape((-1, moved.shape[-1]))
    low, high = jax.vmap(lambda row: _dwt_row(row, taps[0], taps[1], boundary))(rows)
    low = low.reshape(moved.shape[:-1] + (low.shape[-1],))
    high = high.reshape(moved.shape[:-1] + (high.shape[-1],))
    return jnp.moveaxis(low, -1, axis), jnp.moveaxis(high, -1, axis)


def _upsample_convolve(values: Array, filter_: Array, /) -> Array:
    even = jnp.convolve(values, filter_[::2], mode="valid")
    odd = jnp.convolve(values, filter_[1::2], mode="valid")
    return jnp.stack((even, odd), axis=1).reshape((-1,))


def _upsample_convolve_periodic(values: Array, filter_: Array, /) -> Array:
    filter_length = int(filter_.shape[0])
    output_length = 2 * int(values.shape[0])
    upsampled = jnp.zeros((output_length,), dtype=values.dtype).at[::2].set(values)
    convolved = jnp.convolve(
        jnp.pad(upsampled, (filter_length - 1, filter_length - 1), mode="wrap"),
        filter_,
        mode="valid",
    )
    return jnp.roll(convolved[:output_length], -(filter_length // 2 - 1))


def _idwt_row(
    low: Array,
    high: Array,
    reconstruction_low: Array,
    reconstruction_high: Array,
    boundary: WaveletBoundary,
    /,
) -> Array:
    if boundary == "periodization":
        return _upsample_convolve_periodic(
            low, reconstruction_low
        ) + _upsample_convolve_periodic(high, reconstruction_high)
    return _upsample_convolve(low, reconstruction_low) + _upsample_convolve(
        high, reconstruction_high
    )


def idwt_axis(
    low: Array,
    high: Array,
    taps: tuple[Array, Array, Array, Array],
    boundary: WaveletBoundary,
    axis: int,
    /,
) -> Array:
    """Apply a native one-dimensional inverse DWT along one array axis."""
    moved_low = jnp.moveaxis(low, axis, -1)
    moved_high = jnp.moveaxis(high, axis, -1)
    if moved_low.shape != moved_high.shape:
        raise ValueError("Low- and high-pass coefficient arrays must match.")
    low_rows = moved_low.reshape((-1, moved_low.shape[-1]))
    high_rows = moved_high.reshape((-1, moved_high.shape[-1]))
    reconstructed = jax.vmap(
        lambda low_row, high_row: _idwt_row(
            low_row,
            high_row,
            taps[2],
            taps[3],
            boundary,
        )
    )(low_rows, high_rows)
    output = reconstructed.reshape(moved_low.shape[:-1] + (reconstructed.shape[-1],))
    return jnp.moveaxis(output, -1, axis)


__all__ = ["WaveletBoundary", "dwt_axis", "idwt_axis", "load_filter_taps"]
