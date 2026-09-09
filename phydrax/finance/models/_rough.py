#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Measure-neutral rough-volatility parameter records and fixed-grid kernels."""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


def _scalar(value: ArrayLike, name: str, /, *, positive: bool = False) -> Array:
    array = jnp.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be scalar.")
    invalid = ~jnp.isfinite(array) | ((array <= 0.0) if positive else False)
    return eqx.error_if(
        array, invalid, f"{name} must be finite" + (" and positive." if positive else ".")
    )


def _rho(value: ArrayLike) -> Array:
    rho = _scalar(value, "correlation")
    return eqx.error_if(
        rho, jnp.abs(rho) >= 1.0, "correlation must lie strictly in (-1, 1)."
    )


def _curve(times: ArrayLike, values: ArrayLike, /) -> tuple[Array, Array]:
    times_ = jnp.asarray(times, dtype=float)
    values_ = jnp.asarray(values, dtype=float)
    if times_.ndim != 1 or times_.size < 2 or values_.shape != times_.shape:
        raise ValueError(
            "forward variance times and values must be aligned vectors with at least two nodes."
        )
    times_ = eqx.error_if(
        times_,
        jnp.any(~jnp.isfinite(times_))
        | jnp.any(times_ < 0.0)
        | jnp.any(jnp.diff(times_) <= 0.0),
        "forward variance times must be finite, non-negative and strictly increasing.",
    )
    values_ = eqx.error_if(
        values_,
        jnp.any(~jnp.isfinite(values_)) | jnp.any(values_ <= 0.0),
        "forward variance values must be finite and strictly positive.",
    )
    return times_, values_


def fractional_kernel_weights(
    exponent: ArrayLike, step_size: ArrayLike, num_steps: int, /
) -> Array:
    """Cell-average weights for the kernel ``t ** (exponent - 1)``."""

    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
        raise ValueError("num_steps must be a positive integer.")
    alpha = _scalar(exponent, "exponent")
    alpha = eqx.error_if(
        alpha, (alpha <= 0.0) | (alpha >= 1.0), "exponent must lie in (0, 1)."
    )
    dt = _scalar(step_size, "step_size", positive=True)
    upper = jnp.arange(1, num_steps + 1, dtype=dt.dtype)
    lower = upper - 1.0
    return dt ** (alpha - 1.0) * (upper**alpha - lower**alpha) / alpha


class RoughBergomiModel(StrictModule):
    """Rough Bergomi forward-variance model with explicit initial curve."""

    hurst: Array
    volatility_of_variance: Array
    correlation: Array
    forward_variance_times: Array
    forward_variance_values: Array
    curve_size: int = eqx.field(static=True)

    def __init__(
        self,
        hurst: ArrayLike,
        volatility_of_variance: ArrayLike,
        correlation: ArrayLike,
        forward_variance_times: ArrayLike,
        forward_variance_values: ArrayLike,
        /,
    ):
        hurst_ = _scalar(hurst, "hurst")
        hurst_ = eqx.error_if(
            hurst_,
            (hurst_ <= 0.0) | (hurst_ >= 0.5),
            "rough Bergomi hurst must lie in (0, 0.5).",
        )
        times, values = _curve(forward_variance_times, forward_variance_values)
        self.hurst = hurst_
        self.volatility_of_variance = _scalar(
            volatility_of_variance, "volatility_of_variance", positive=True
        )
        self.correlation = _rho(correlation)
        self.forward_variance_times = times
        self.forward_variance_values = values
        self.curve_size = int(times.size)

    def forward_variance(self, time: ArrayLike, /) -> Array:
        time_ = jnp.asarray(time, dtype=float)
        time_ = eqx.error_if(
            time_,
            jnp.any(~jnp.isfinite(time_)) | jnp.any(time_ < 0.0),
            "time must be finite and non-negative.",
        )
        return jnp.interp(
            time_, self.forward_variance_times, self.forward_variance_values
        )

    def kernel_weights(self, step_size: ArrayLike, num_steps: int, /) -> Array:
        return fractional_kernel_weights(self.hurst + 0.5, step_size, num_steps)


class RoughHestonModel(StrictModule):
    """Fractional-Volterra Heston parameters with alpha in (1/2, 1)."""

    fractional_order: Array
    mean_reversion: Array
    long_run_variance: Array
    volatility_of_variance: Array
    correlation: Array
    initial_variance: Array

    def __init__(
        self,
        fractional_order: ArrayLike,
        mean_reversion: ArrayLike,
        long_run_variance: ArrayLike,
        volatility_of_variance: ArrayLike,
        correlation: ArrayLike,
        initial_variance: ArrayLike,
        /,
    ):
        order = _scalar(fractional_order, "fractional_order")
        order = eqx.error_if(
            order,
            (order <= 0.5) | (order >= 1.0),
            "fractional_order must lie in (0.5, 1).",
        )
        self.fractional_order = order
        self.mean_reversion = _scalar(mean_reversion, "mean_reversion", positive=True)
        self.long_run_variance = _scalar(
            long_run_variance, "long_run_variance", positive=True
        )
        self.volatility_of_variance = _scalar(
            volatility_of_variance, "volatility_of_variance", positive=True
        )
        self.correlation = _rho(correlation)
        self.initial_variance = _scalar(
            initial_variance, "initial_variance", positive=True
        )

    @property
    def hurst(self) -> Array:
        return self.fractional_order - 0.5

    def kernel_weights(self, step_size: ArrayLike, num_steps: int, /) -> Array:
        return fractional_kernel_weights(self.fractional_order, step_size, num_steps)


__all__ = ["RoughBergomiModel", "RoughHestonModel", "fractional_kernel_weights"]
