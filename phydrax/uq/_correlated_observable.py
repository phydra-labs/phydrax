#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from .._fingerprint import canonical_fingerprint
from .._strict import StrictModule


CORRELATED_OBSERVABLE_SUCCESS = 0
CORRELATED_OBSERVABLE_INSUFFICIENT_DRAWS = 1
CORRELATED_OBSERVABLE_NONFINITE = 2
CORRELATED_OBSERVABLE_ZERO_VARIANCE = 3

CorrelatedObservableStatus = Literal[
    "success",
    "insufficient_draws",
    "nonfinite",
    "zero_variance",
]


def correlated_observable_status_name(value: int, /) -> CorrelatedObservableStatus:
    """Return the stable name of a correlated-observable status code."""
    names: tuple[CorrelatedObservableStatus, ...] = (
        "success",
        "insufficient_draws",
        "nonfinite",
        "zero_variance",
    )
    code = int(value)
    if code < 0 or code >= len(names):
        raise ValueError(f"Unknown correlated-observable status code {code}.")
    return names[code]


class CorrelatedObservablePolicy(StrictModule):
    """Static autocorrelation policy for chain-by-draw observable samples."""

    max_lag: int | None = eqx.field(static=True)
    minimum_draws: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        max_lag: int | None = None,
        minimum_draws: int = 8,
    ):
        if max_lag is not None and int(max_lag) < 1:
            raise ValueError("max_lag must be positive when provided.")
        minimum = int(minimum_draws)
        if minimum < 4:
            raise ValueError("minimum_draws must be at least four.")
        maximum = None if max_lag is None else int(max_lag)
        self.max_lag = maximum
        self.minimum_draws = minimum
        self.policy_id = canonical_fingerprint(
            {
                "kind": "correlated-observable-policy",
                "max_lag": maximum,
                "minimum_draws": minimum,
                "window": "geyer-initial-monotone-sequence",
                "tau_convention": "one-plus-two-rho-sum",
            }
        )


class CorrelatedObservableDiagnostics(StrictModule):
    """Raw-scale autocorrelation diagnostics for real observable samples."""

    mean: Array
    variance: Array
    autocorrelation: Array
    integrated_autocorrelation_time: Array
    effective_sample_size: Array
    standard_error: Array
    window_lag: Array
    valid: Array
    status: Array
    policy: CorrelatedObservablePolicy
    num_chains: int = eqx.field(static=True)
    num_draws: int = eqx.field(static=True)
    resolved_max_lag: int = eqx.field(static=True)

    @property
    def successful(self) -> Array:
        return self.status == CORRELATED_OBSERVABLE_SUCCESS


def correlated_observable_diagnostics(
    samples: ArrayLike,
    /,
    *,
    policy: CorrelatedObservablePolicy | None = None,
) -> CorrelatedObservableDiagnostics:
    """Estimate raw observable autocorrelation with Geyer's monotone window.

    ``samples`` has leading ``(chain, draw)`` axes. Trailing axes identify
    independent real observables and are preserved in every scalar diagnostic.
    The returned autocorrelation has one additional leading lag axis.
    """
    policy_ = CorrelatedObservablePolicy() if policy is None else policy
    if not isinstance(policy_, CorrelatedObservablePolicy):
        raise TypeError("policy must be CorrelatedObservablePolicy.")
    values = jnp.asarray(samples)
    if values.ndim < 2:
        raise ValueError("samples must have leading chain and draw axes.")
    if jnp.issubdtype(values.dtype, jnp.complexfloating):
        raise TypeError(
            "Correlated-observable diagnostics require real samples; diagnose real and imaginary components separately."
        )
    if not jnp.issubdtype(values.dtype, jnp.floating):
        values = values.astype(jnp.float64)

    num_chains = values.shape[0]
    num_draws = values.shape[1]
    if num_chains < 1:
        raise ValueError("samples must contain at least one chain.")
    resolved_max_lag = min(
        num_draws - 1,
        num_draws - 1 if policy_.max_lag is None else policy_.max_lag,
    )
    resolved_max_lag = max(resolved_max_lag, 1)

    output_shape = tuple(values.shape[2:])
    output_size = 1
    for size in output_shape:
        output_size *= size
    flat = jnp.reshape(values, (num_chains, num_draws, output_size))
    finite = jnp.all(jnp.isfinite(flat), axis=(0, 1))
    safe = jnp.where(jnp.isfinite(flat), flat, 0.0)
    chain_mean = jnp.mean(safe, axis=1, keepdims=True)
    centered = safe - chain_mean

    fft_size = 1 << max(1, (2 * num_draws - 1).bit_length())
    transformed = jnp.fft.rfft(centered, n=fft_size, axis=1)
    power = transformed * jnp.conj(transformed)
    covariance = jnp.fft.irfft(power, n=fft_size, axis=1)
    covariance = covariance[:, : resolved_max_lag + 1, :]
    divisor = jnp.arange(
        num_draws,
        num_draws - resolved_max_lag - 1,
        -1,
        dtype=safe.dtype,
    )
    covariance = covariance / divisor[None, :, None]
    pooled_covariance = jnp.mean(covariance, axis=0)
    variance = pooled_covariance[0]
    positive_variance = variance > 0.0
    safe_variance = jnp.where(positive_variance, variance, 1.0)
    autocorrelation = pooled_covariance / safe_variance[None, :]
    autocorrelation = autocorrelation.at[0].set(jnp.where(positive_variance, 1.0, 0.0))

    pair_count = (resolved_max_lag + 1) // 2
    paired = (
        autocorrelation[: 2 * pair_count : 2] + autocorrelation[1 : 2 * pair_count : 2]
    )
    monotone_pairs = jax.lax.associative_scan(jnp.minimum, paired, axis=0)
    positive_prefix = jnp.cumprod(monotone_pairs > 0.0, axis=0, dtype=jnp.int32)
    included_pairs = jnp.where(positive_prefix.astype("bool"), monotone_pairs, 0.0)
    tau = -1.0 + 2.0 * jnp.sum(included_pairs, axis=0)
    tau = jnp.maximum(tau, jnp.finfo(safe.dtype).eps)
    selected_pairs = jnp.sum(positive_prefix, axis=0, dtype=jnp.int32)
    window_lag = jnp.maximum(2 * selected_pairs - 1, 0)

    enough_draws = num_draws >= policy_.minimum_draws
    status = jnp.full((output_size,), CORRELATED_OBSERVABLE_SUCCESS, dtype=jnp.int32)
    status = jnp.where(
        enough_draws,
        status,
        jnp.asarray(CORRELATED_OBSERVABLE_INSUFFICIENT_DRAWS, dtype=jnp.int32),
    )
    status = jnp.where(
        finite,
        status,
        jnp.asarray(CORRELATED_OBSERVABLE_NONFINITE, dtype=jnp.int32),
    )
    status = jnp.where(
        positive_variance | ~finite,
        status,
        jnp.asarray(CORRELATED_OBSERVABLE_ZERO_VARIANCE, dtype=jnp.int32),
    )
    valid = status == CORRELATED_OBSERVABLE_SUCCESS
    sample_count = float(num_chains * num_draws)
    effective_sample_size = sample_count / tau
    standard_error = jnp.sqrt(jnp.maximum(variance, 0.0) * tau / sample_count)

    def restore(array: Array) -> Array:
        return jnp.reshape(array, output_shape)

    return CorrelatedObservableDiagnostics(
        mean=restore(jnp.mean(safe, axis=(0, 1))),
        variance=restore(variance),
        autocorrelation=jnp.reshape(
            autocorrelation,
            (resolved_max_lag + 1,) + output_shape,
        ),
        integrated_autocorrelation_time=restore(tau),
        effective_sample_size=restore(effective_sample_size),
        standard_error=restore(standard_error),
        window_lag=restore(window_lag),
        valid=restore(valid),
        status=restore(status),
        policy=policy_,
        num_chains=num_chains,
        num_draws=num_draws,
        resolved_max_lag=resolved_max_lag,
    )


__all__ = [
    "CORRELATED_OBSERVABLE_INSUFFICIENT_DRAWS",
    "CORRELATED_OBSERVABLE_NONFINITE",
    "CORRELATED_OBSERVABLE_SUCCESS",
    "CORRELATED_OBSERVABLE_ZERO_VARIANCE",
    "CorrelatedObservableDiagnostics",
    "CorrelatedObservablePolicy",
    "CorrelatedObservableStatus",
    "correlated_observable_diagnostics",
    "correlated_observable_status_name",
]
