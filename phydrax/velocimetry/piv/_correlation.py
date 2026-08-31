#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._types import CorrelationBatch
from ._windows import _pair, WindowBatch2D


def _cross_correlation(a: Array, b: Array, fft_shape: tuple[int, int]) -> Array:
    transform_a = jnp.fft.rfftn(a, s=fft_shape, axes=(-2, -1))
    transform_b = jnp.fft.rfftn(b, s=fft_shape, axes=(-2, -1))
    return jnp.fft.irfftn(
        jnp.conj(transform_a) * transform_b,
        s=fft_shape,
        axes=(-2, -1),
    )


def _lag_grid(margin: tuple[int, int]) -> Array:
    rows, columns = jnp.meshgrid(
        jnp.arange(-margin[0], margin[0] + 1, dtype=jnp.int32),
        jnp.arange(-margin[1], margin[1] + 1, dtype=jnp.int32),
        indexing="ij",
    )
    return jnp.stack((rows, columns), axis=-1)


def _gather_lags(
    surface: Array,
    lags_rc: Array,
    /,
    *,
    base_lag: tuple[int, int],
) -> Array:
    row = jnp.mod(lags_rc[..., 0] + base_lag[0], surface.shape[-2])
    column = jnp.mod(lags_rc[..., 1] + base_lag[1], surface.shape[-1])
    return surface[..., row, column]


def correlate_windows(
    first: WindowBatch2D,
    second: WindowBatch2D,
    /,
    *,
    mode: str,
    search_margin: int | Sequence[int],
    chunk_size: int,
    minimum_valid_fraction: float,
    normalized: bool = True,
) -> CorrelationBatch:
    """Compute chunked mask-aware FFT correlation on a fixed lag rectangle."""
    if not isinstance(first, WindowBatch2D) or not isinstance(second, WindowBatch2D):
        raise TypeError("first and second must be WindowBatch2D instances.")
    if first.values.shape[0] != second.values.shape[0]:
        raise ValueError("Window batches must contain equal window counts.")
    mode_ = str(mode)
    if mode_ not in ("linear", "circular", "extended"):
        raise ValueError("mode must be linear, circular, or extended.")
    margin = _pair(search_margin, name="search_margin", minimum=0)
    chunk = int(chunk_size)
    if chunk < 1:
        raise ValueError("chunk_size must be positive.")
    fraction = float(minimum_valid_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1].")
    first_shape = tuple(int(value) for value in first.values.shape[-2:])
    second_shape = tuple(int(value) for value in second.values.shape[-2:])
    if mode_ == "circular" and first_shape != second_shape:
        raise ValueError("Circular correlation requires equal window shapes.")
    if mode_ == "extended":
        expected = tuple(first_shape[axis] + 2 * margin[axis] for axis in range(2))
        if second_shape != expected:
            raise ValueError("Extended second windows must include both search margins.")
        base_lag = margin
    else:
        if first_shape != second_shape:
            raise ValueError("Linear/circular correlation requires equal window shapes.")
        base_lag = (0, 0)
    fft_shape = (
        first_shape
        if mode_ == "circular"
        else tuple(first_shape[axis] + second_shape[axis] - 1 for axis in range(2))
    )
    lags = _lag_grid(margin)
    count = first.values.shape[0]
    padded_count = ((count + chunk - 1) // chunk) * chunk
    padding = padded_count - count

    def pad(values: Array, value: float | bool) -> Array:
        return jnp.pad(values, ((0, padding), (0, 0), (0, 0)), constant_values=value)

    dtype = jnp.result_type(first.values.dtype, second.values.dtype, jnp.float32)
    first_values = pad(jnp.asarray(first.values, dtype=dtype), 0.0).reshape(
        (-1, chunk) + first_shape
    )
    second_values = pad(jnp.asarray(second.values, dtype=dtype), 0.0).reshape(
        (-1, chunk) + second_shape
    )
    first_mask = pad(jnp.asarray(first.mask, dtype=bool), False).reshape(
        (-1, chunk) + first_shape
    )
    second_mask = pad(jnp.asarray(second.mask, dtype=bool), False).reshape(
        (-1, chunk) + second_shape
    )

    def correlate_chunk(
        carry: None,
        inputs: tuple[Array, Array, Array, Array],
    ) -> tuple[None, tuple[Array, Array, Array]]:
        first_chunk, second_chunk, first_mask_chunk, second_mask_chunk = inputs
        mask_a = first_mask_chunk.astype(first_chunk.dtype)
        mask_b = second_mask_chunk.astype(second_chunk.dtype)
        a = jnp.where(first_mask_chunk, first_chunk, 0.0)
        b = jnp.where(second_mask_chunk, second_chunk, 0.0)
        overlap_full = _cross_correlation(mask_a, mask_b, fft_shape)
        sum_ab_full = _cross_correlation(a, b, fft_shape)
        overlap = jnp.rint(
            jnp.maximum(_gather_lags(overlap_full, lags, base_lag=base_lag), 0.0)
        )
        sum_ab = _gather_lags(sum_ab_full, lags, base_lag=base_lag)
        enough = overlap >= fraction * float(first_shape[0] * first_shape[1])
        if normalized:
            sum_a = _gather_lags(
                _cross_correlation(a, mask_b, fft_shape), lags, base_lag=base_lag
            )
            sum_b = _gather_lags(
                _cross_correlation(mask_a, b, fft_shape), lags, base_lag=base_lag
            )
            sum_a2 = _gather_lags(
                _cross_correlation(a * a, mask_b, fft_shape), lags, base_lag=base_lag
            )
            sum_b2 = _gather_lags(
                _cross_correlation(mask_a, b * b, fft_shape), lags, base_lag=base_lag
            )
            safe_overlap = jnp.maximum(overlap, 1.0)
            numerator = sum_ab - (sum_a * sum_b) / safe_overlap
            variance_a = jnp.maximum(sum_a2 - (sum_a * sum_a) / safe_overlap, 0.0)
            variance_b = jnp.maximum(sum_b2 - (sum_b * sum_b) / safe_overlap, 0.0)
            denominator = jnp.sqrt(variance_a * variance_b)
            epsilon = jnp.finfo(first_chunk.dtype).eps * safe_overlap
            valid = enough & jnp.isfinite(numerator) & (denominator > epsilon)
            correlation = numerator / jnp.where(valid, denominator, 1.0)
        else:
            valid = enough & jnp.isfinite(sum_ab)
            correlation = sum_ab
        return None, (jnp.where(valid, correlation, -jnp.inf), overlap, valid)

    _, chunks = jax.lax.scan(
        correlate_chunk,
        None,
        (first_values, second_values, first_mask, second_mask),
    )
    values = chunks[0].reshape((padded_count,) + tuple(lags.shape[:2]))[:count]
    overlap = chunks[1].reshape((padded_count,) + tuple(lags.shape[:2]))[:count]
    valid = chunks[2].reshape((padded_count,) + tuple(lags.shape[:2]))[:count]
    return CorrelationBatch(values, overlap, valid, lags, mode_)


__all__ = ["correlate_windows"]
