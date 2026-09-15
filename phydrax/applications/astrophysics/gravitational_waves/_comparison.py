#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from enum import IntEnum

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from ._data import OneSidedPowerSpectralDensity


class WaveformComparisonStatus(IntEnum):
    SUCCESS = 0
    NONFINITE_WAVEFORM = 1
    ZERO_NORM_WAVEFORM = 2
    NORMALIZATION_FAILURE = 3


class WaveformMatchPlan(StrictModule, NonTrainableState):
    """PSD-bound one-sided overlap and discrete time/phase match plan."""

    psd: OneSidedPowerSpectralDensity
    normalization_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        psd: OneSidedPowerSpectralDensity,
        /,
        *,
        normalization_tolerance: float = 1.0e-10,
    ):
        if not isinstance(psd, OneSidedPowerSpectralDensity):
            raise TypeError("psd must be OneSidedPowerSpectralDensity.")
        tolerance = float(normalization_tolerance)
        if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > 1.0e-6:
            raise ValueError(
                "normalization_tolerance must be finite and lie in [0, 1e-6]."
            )
        self.psd = psd
        self.normalization_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "gravitational-wave-match-plan-v1",
                "psd": psd.psd_id,
                "normalization_tolerance": tolerance,
                "inner_product": "one-sided-positive-frequency",
                "time_search": "canonical-grid-discrete-ifft",
            }
        )

    def overlap(
        self,
        first: ArrayLike,
        second: ArrayLike,
        /,
        *,
        time_shift_seconds: ArrayLike = 0.0,
    ) -> WaveformOverlapResult:
        return phase_maximized_overlap(
            self,
            first,
            second,
            time_shift_seconds=time_shift_seconds,
        )

    def match(self, first: ArrayLike, second: ArrayLike, /) -> WaveformMatchResult:
        return maximize_waveform_match(self, first, second)


class WaveformOverlapResult(StrictModule):
    overlap: Array
    mismatch: Array
    first_norm_squared: Array
    second_norm_squared: Array
    time_shift_seconds: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


class WaveformMatchResult(StrictModule):
    match: Array
    mismatch: Array
    lag_seconds: Array
    lag_index: Array
    first_norm_squared: Array
    second_norm_squared: Array
    valid: Array
    status: Array
    plan_id: str = eqx.field(static=True)


def _prepared_strains(
    plan: WaveformMatchPlan, first: ArrayLike, second: ArrayLike, /
) -> tuple[Array, Array, Array, Array, Array]:
    first_ = jnp.asarray(first)
    second_ = jnp.asarray(second)
    expected = tuple(plan.psd.frequency.shape)
    if first_.shape != expected or second_.shape != expected:
        raise ValueError(f"Waveforms must both have the PSD shape {expected}.")
    dtype = jnp.result_type(first_.dtype, second_.dtype, 1.0j)
    first_ = first_.astype(dtype)
    second_ = second_.astype(dtype)
    active = plan.psd.active
    first_safe = jnp.where(active, first_, jnp.zeros((), dtype=dtype))
    second_safe = jnp.where(active, second_, jnp.zeros((), dtype=dtype))
    frequency_step = 1.0 / plan.psd.duration
    inner_product_weight = jnp.where(active, 4.0 * frequency_step / plan.psd.values, 0.0)
    first_norm = jnp.sum(jnp.abs(first_safe) ** 2 * inner_product_weight)
    second_norm = jnp.sum(jnp.abs(second_safe) ** 2 * inner_product_weight)
    return first_safe, second_safe, inner_product_weight, first_norm, second_norm


def _comparison_status(
    finite: Array,
    first_norm: Array,
    second_norm: Array,
    normalized: Array,
    tolerance: float,
    /,
) -> tuple[Array, Array]:
    nonzero = (first_norm > 0.0) & (second_norm > 0.0)
    normalized_finite = jnp.isfinite(normalized)
    normalization_valid = normalized <= 1.0 + tolerance
    valid = finite & nonzero & normalized_finite & normalization_valid
    status = jnp.where(
        valid,
        int(WaveformComparisonStatus.SUCCESS),
        jnp.where(
            ~finite | ~normalized_finite,
            int(WaveformComparisonStatus.NONFINITE_WAVEFORM),
            jnp.where(
                ~nonzero,
                int(WaveformComparisonStatus.ZERO_NORM_WAVEFORM),
                int(WaveformComparisonStatus.NORMALIZATION_FAILURE),
            ),
        ),
    ).astype(jnp.int32)
    return valid, status


def phase_maximized_overlap(
    plan: WaveformMatchPlan,
    first: ArrayLike,
    second: ArrayLike,
    /,
    *,
    time_shift_seconds: ArrayLike = 0.0,
) -> WaveformOverlapResult:
    """Evaluate normalized overlap, maximizing analytically over constant phase."""

    if not isinstance(plan, WaveformMatchPlan):
        raise TypeError("plan must be WaveformMatchPlan.")
    first_, second_, inverse_psd, first_norm, second_norm = _prepared_strains(
        plan, first, second
    )
    shift_raw = jnp.asarray(time_shift_seconds)
    if (
        shift_raw.shape != ()
        or jnp.iscomplexobj(shift_raw)
        or jnp.issubdtype(shift_raw.dtype, jnp.bool_)
    ):
        raise TypeError("time_shift_seconds must be one real numeric scalar.")
    shift = shift_raw.astype(plan.psd.frequency.dtype)
    finite_shift = jnp.isfinite(shift)
    safe_shift = jnp.where(
        finite_shift,
        jnp.remainder(shift, plan.psd.duration),
        0.0,
    )
    phase = jnp.exp(2.0j * jnp.pi * plan.psd.frequency * safe_shift)
    numerator = jnp.abs(jnp.sum(jnp.conj(first_) * second_ * phase * inverse_psd))
    denominator = jnp.sqrt(
        jnp.where(first_norm * second_norm > 0.0, first_norm * second_norm, 1.0)
    )
    overlap = numerator / denominator
    finite = finite_shift & jnp.all(jnp.isfinite(first_)) & jnp.all(jnp.isfinite(second_))
    valid, status = _comparison_status(
        finite,
        first_norm,
        second_norm,
        overlap,
        plan.normalization_tolerance,
    )
    invalid = jnp.asarray(jnp.nan, dtype=overlap.dtype)
    overlap_out = jnp.where(valid, overlap, invalid)
    return WaveformOverlapResult(
        overlap_out,
        jnp.where(valid, 1.0 - overlap, invalid),
        first_norm,
        second_norm,
        shift,
        valid,
        status,
        plan.plan_id,
    )


def maximize_waveform_match(
    plan: WaveformMatchPlan,
    first: ArrayLike,
    second: ArrayLike,
    /,
) -> WaveformMatchResult:
    """Maximize normalized overlap over phase and canonical-grid time shifts."""

    if not isinstance(plan, WaveformMatchPlan):
        raise TypeError("plan must be WaveformMatchPlan.")
    first_, second_, inverse_psd, first_norm, second_norm = _prepared_strains(
        plan, first, second
    )
    cross_spectrum = jnp.conj(first_) * second_ * inverse_psd
    correlation = plan.psd.sample_count * jnp.fft.ifft(
        cross_spectrum, n=plan.psd.sample_count
    )
    denominator = jnp.sqrt(
        jnp.where(first_norm * second_norm > 0.0, first_norm * second_norm, 1.0)
    )
    normalized = jnp.abs(correlation) / denominator
    lag_index = jnp.argmax(normalized).astype(jnp.int32)
    match = normalized[lag_index]
    half = plan.psd.sample_count // 2
    signed_index = jnp.where(
        lag_index <= half,
        lag_index,
        lag_index - plan.psd.sample_count,
    )
    lag_seconds = signed_index.astype(plan.psd.frequency.dtype) * plan.psd.sample_interval
    finite = jnp.all(jnp.isfinite(first_)) & jnp.all(jnp.isfinite(second_))
    valid, status = _comparison_status(
        finite,
        first_norm,
        second_norm,
        match,
        plan.normalization_tolerance,
    )
    invalid = jnp.asarray(jnp.nan, dtype=match.dtype)
    return WaveformMatchResult(
        jnp.where(valid, match, invalid),
        jnp.where(valid, 1.0 - match, invalid),
        jnp.where(valid, lag_seconds, invalid),
        jnp.where(valid, lag_index, jnp.asarray(-1, dtype=jnp.int32)),
        first_norm,
        second_norm,
        valid,
        status,
        plan.plan_id,
    )


__all__ = [
    "WaveformComparisonStatus",
    "WaveformMatchPlan",
    "WaveformMatchResult",
    "WaveformOverlapResult",
    "maximize_waveform_match",
    "phase_maximized_overlap",
]
