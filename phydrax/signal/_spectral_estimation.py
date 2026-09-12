#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._framing import frame
from ._windows import hann_window, tukey_window


class WelchSpectrumResult(StrictModule):
    frequencies: Array
    power_spectral_density: Array
    segment_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


def _median_bias(count: int, dtype, /) -> Array:
    pairs = 2.0 * np.arange(1, (int(count) - 1) // 2 + 1, dtype=float)
    bias = 1.0 + np.sum(1.0 / (pairs + 1.0) - 1.0 / pairs)
    return jnp.asarray(bias, dtype=dtype)


class WelchSpectrumPlan(StrictModule, NonTrainableState):
    """One-sided Welch density estimate with explicit normalization."""

    sample_interval: float = eqx.field(static=True)
    segment_length: int = eqx.field(static=True)
    overlap: int = eqx.field(static=True)
    detrend: Literal["none", "mean"] = eqx.field(static=True)
    window: Literal["hann", "tukey"] = eqx.field(static=True)
    average: Literal["mean", "median"] = eqx.field(static=True)
    tukey_alpha: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        sample_interval: float,
        segment_length: int,
        /,
        *,
        overlap: int | None = None,
        detrend: Literal["none", "mean"] = "mean",
        window: Literal["hann", "tukey"] = "hann",
        average: Literal["mean", "median"] = "mean",
        tukey_alpha: float = 0.5,
    ):
        interval = float(sample_interval)
        length = int(segment_length)
        overlap_ = length // 2 if overlap is None else int(overlap)
        alpha = float(tukey_alpha)
        if (
            not np.isfinite(interval)
            or interval <= 0.0
            or length < 2
            or not 0 <= overlap_ < length
            or detrend not in ("none", "mean")
            or window not in ("hann", "tukey")
            or average not in ("mean", "median")
            or not np.isfinite(alpha)
            or not 0.0 <= alpha <= 1.0
        ):
            raise ValueError(
                "Welch sampling, segment, overlap, detrend, window, or averaging is invalid."
            )
        self.sample_interval = interval
        self.segment_length = length
        self.overlap = overlap_
        self.detrend = detrend
        self.window = window
        self.average = average
        self.tukey_alpha = alpha
        self.plan_id = canonical_fingerprint(
            {
                "kind": "welch-spectrum",
                "sample_interval": interval,
                "segment_length": length,
                "overlap": overlap_,
                "detrend": detrend,
                "window": {
                    "kind": window,
                    "periodic": True,
                    "tukey_alpha": alpha if window == "tukey" else None,
                },
                "average": average,
                "scaling": "one-sided-density",
            }
        )

    def evaluate(self, values: ArrayLike, /, *, axis: int = -1) -> WelchSpectrumResult:
        signal = jnp.asarray(values)
        if not jnp.issubdtype(signal.dtype, jnp.inexact) or jnp.iscomplexobj(signal):
            raise TypeError("Welch spectrum requires a real inexact signal.")
        if signal.ndim == 0:
            raise ValueError("Welch spectrum requires at least one signal axis.")
        resolved_axis = int(axis) % signal.ndim
        canonical = jnp.moveaxis(signal, resolved_axis, -1)
        framed = frame(
            canonical,
            self.segment_length,
            self.segment_length - self.overlap,
            axis=-1,
        )
        if self.detrend == "mean":
            framed = framed - jnp.mean(framed, axis=-1, keepdims=True)
        window = (
            hann_window(self.segment_length, periodic=True, dtype=signal.dtype)
            if self.window == "hann"
            else tukey_window(
                self.segment_length,
                self.tukey_alpha,
                periodic=True,
                dtype=signal.dtype,
            )
        )
        transformed = jnp.fft.rfft(framed * window, axis=-1)
        sample_frequency = 1.0 / self.sample_interval
        density = jnp.abs(transformed) ** 2 / (
            sample_frequency * jnp.sum(window * window)
        )
        one_sided_factor = (
            jnp.full((density.shape[-1],), 2.0, dtype=density.dtype).at[0].set(1.0)
        )
        if self.segment_length % 2 == 0:
            one_sided_factor = one_sided_factor.at[-1].set(1.0)
        density = density * one_sided_factor
        averaged = (
            jnp.mean(density, axis=-2)
            if self.average == "mean"
            else jnp.median(density, axis=-2)
            / _median_bias(int(framed.shape[-2]), density.dtype)
        )
        frequencies = jnp.fft.rfftfreq(
            self.segment_length, d=self.sample_interval
        ).astype(signal.dtype)
        return WelchSpectrumResult(
            frequencies,
            averaged,
            framed.shape[-2],
            self.plan_id,
        )


__all__ = ["WelchSpectrumPlan", "WelchSpectrumResult"]
