#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class NeutralMesonMixingParameters(StrictModule, NonTrainableState):
    decay_width: Array
    mass_difference: Array
    width_difference: Array
    q_over_p: Array
    convention_id: str = eqx.field(static=True)
    parameter_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        decay_width: float,
        mass_difference: float,
        width_difference: float,
        q_over_p: complex,
        convention_id: str,
    ):
        width = float(decay_width)
        mass = float(mass_difference)
        difference = float(width_difference)
        ratio = complex(q_over_p)
        convention = str(convention_id).strip()
        if (
            not all(
                math.isfinite(value)
                for value in (width, mass, difference, ratio.real, ratio.imag)
            )
            or width <= 0.0
            or abs(difference) >= 2.0 * width
            or abs(ratio) <= 0.0
            or not convention
        ):
            raise ValueError("Neutral-meson mixing parameters or convention are invalid.")
        self.decay_width = jnp.asarray(width)
        self.mass_difference = jnp.asarray(mass)
        self.width_difference = jnp.asarray(difference)
        self.q_over_p = jnp.asarray(ratio)
        self.convention_id = convention
        self.parameter_id = canonical_fingerprint(
            {
                "kind": "neutral-meson-mixing-parameters",
                "values": [width, mass, difference, ratio.real, ratio.imag],
                "convention": convention,
            }
        )


class TaggingCalibration(StrictModule, NonTrainableState):
    intercept: Array
    slope: Array
    mean_raw_mistag: Array
    calibration_id: str = eqx.field(static=True)

    def __init__(
        self, intercept: float, slope: float, mean_raw_mistag: float, /, *, source_id: str
    ):
        values = tuple(map(float, (intercept, slope, mean_raw_mistag)))
        source = str(source_id).strip()
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.0 <= values[2] <= 0.5
            or not source
        ):
            raise ValueError("Tagging calibration values and source are invalid.")
        self.intercept, self.slope, self.mean_raw_mistag = (
            jnp.asarray(value) for value in values
        )
        self.calibration_id = canonical_fingerprint(
            {
                "kind": "flavor-tagging-calibration",
                "values": list(values),
                "source": source,
            }
        )

    def calibrate(self, raw_mistag: ArrayLike, /) -> Array:
        raw = jnp.asarray(raw_mistag)
        return jnp.clip(
            self.intercept + self.slope * (raw - self.mean_raw_mistag), 0.0, 0.5
        )


class TimeDependentDecayResult(StrictModule, NonTrainableState):
    rate: Array
    calibrated_mistag: Array
    finite: Array
    valid: Array
    parameter_id: str = eqx.field(static=True)
    calibration_id: str = eqx.field(static=True)


def time_dependent_decay_rate(
    parameters: NeutralMesonMixingParameters,
    calibration: TaggingCalibration,
    decay_times: ArrayLike,
    flavor_tags: ArrayLike,
    raw_mistag: ArrayLike,
    amplitude: ArrayLike,
    conjugate_amplitude: ArrayLike,
    /,
    *,
    time_resolution: ArrayLike = 0.0,
) -> TimeDependentDecayResult:
    """Evaluate tagged neutral-meson mixing with Gaussian time-resolution damping."""
    if not isinstance(parameters, NeutralMesonMixingParameters) or not isinstance(
        calibration, TaggingCalibration
    ):
        raise TypeError("parameters and calibration must use flavor types.")
    time = jnp.asarray(decay_times)
    tags = jnp.asarray(flavor_tags, dtype=jnp.int32)
    mistag = jnp.asarray(raw_mistag, dtype=time.dtype)
    amplitude_ = jnp.asarray(amplitude)
    conjugate = jnp.asarray(conjugate_amplitude, dtype=amplitude_.dtype)
    resolution = jnp.asarray(time_resolution, dtype=time.dtype)
    time, mistag, resolution = jnp.broadcast_arrays(time, mistag, resolution)
    if (
        tags.shape != time.shape
        or amplitude_.shape != time.shape
        or conjugate.shape != time.shape
    ):
        raise ValueError(
            "Flavor time, tag, mistag, resolution, and amplitudes must align."
        )
    calibrated = calibration.calibrate(mistag)
    exponential = jnp.exp(-0.5 * parameters.decay_width * time)
    cosh = jnp.cosh(0.25 * parameters.width_difference * time)
    sinh = jnp.sinh(0.25 * parameters.width_difference * time)
    cosine = jnp.cos(0.5 * parameters.mass_difference * time)
    sine = jnp.sin(0.5 * parameters.mass_difference * time)
    g_plus = exponential * (cosh * cosine - 1j * sinh * sine)
    g_minus = exponential * (-sinh * cosine + 1j * cosh * sine)
    meson_amplitude = g_plus * amplitude_ + parameters.q_over_p * g_minus * conjugate
    antimeson_amplitude = (
        g_plus * conjugate + (1.0 / parameters.q_over_p) * g_minus * amplitude_
    )
    meson_rate = jnp.real(meson_amplitude * jnp.conj(meson_amplitude))
    antimeson_rate = jnp.real(antimeson_amplitude * jnp.conj(antimeson_amplitude))
    dilution = 1.0 - 2.0 * calibrated
    tagged_rate = 0.5 * (
        (meson_rate + antimeson_rate) + tags * dilution * (meson_rate - antimeson_rate)
    )
    damping = jnp.exp(-0.5 * (parameters.mass_difference * resolution) ** 2)
    average_rate = 0.5 * (meson_rate + antimeson_rate)
    rate = average_rate + damping * (tagged_rate - average_rate)
    finite = jnp.isfinite(rate)
    valid = (
        finite
        & (time >= 0.0)
        & (resolution >= 0.0)
        & ((tags == -1) | (tags == 0) | (tags == 1))
        & (rate >= 0.0)
    )
    return TimeDependentDecayResult(
        jnp.where(valid, rate, jnp.nan),
        calibrated,
        finite,
        valid,
        parameters.parameter_id,
        calibration.calibration_id,
    )


__all__ = [
    "NeutralMesonMixingParameters",
    "TaggingCalibration",
    "TimeDependentDecayResult",
    "time_dependent_decay_rate",
]
