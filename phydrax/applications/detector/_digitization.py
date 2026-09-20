#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jaxtyping import Array, ArrayLike, Key

from phydrax import ein

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._core import DigitBank, SensitiveHitBank, TruthStepBank


class SensitiveHitPlan(StrictModule, NonTrainableState):
    element_to_channel: Array
    conditions_id: str = eqx.field(static=True)
    element_count: int = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        element_to_channel: ArrayLike,
        /,
        *,
        channel_count: int,
        conditions_id: str,
    ):
        mapping = np.asarray(element_to_channel)
        channels = int(channel_count)
        conditions = str(conditions_id).strip()
        if (
            mapping.ndim != 1
            or mapping.size < 1
            or not np.issubdtype(mapping.dtype, np.integer)
        ):
            raise ValueError("element_to_channel must be a non-empty integer array.")
        if channels < 1 or np.any(mapping < -1) or np.any(mapping >= channels):
            raise ValueError(
                "Element-to-channel mapping exceeds the declared channel support."
            )
        if not conditions:
            raise ValueError("conditions_id must be non-empty.")
        self.element_to_channel = jnp.asarray(mapping, dtype=jnp.int32)
        self.conditions_id = conditions
        self.element_count = mapping.size
        self.channel_count = channels
        self.plan_id = canonical_fingerprint(
            {
                "kind": "sensitive-hit-plan",
                "mapping": array_tree_fingerprint(mapping),
                "channel_count": channels,
                "conditions": conditions,
            }
        )


def form_sensitive_hits(
    plan: SensitiveHitPlan, steps: TruthStepBank, /
) -> SensitiveHitBank:
    """Convert valid sensitive truth steps to one contribution-preserving hit each."""
    if not isinstance(plan, SensitiveHitPlan) or not isinstance(steps, TruthStepBank):
        raise TypeError("plan and steps must use detector hit types.")
    if steps.conditions_id != plan.conditions_id:
        raise ValueError("Truth steps and sensitive-hit plan use different conditions.")
    elements = steps.detector_element_ids
    in_range = (elements >= 0) & (elements < plan.element_count)
    safe_elements = jnp.clip(elements, 0, plan.element_count - 1)
    channels = plan.element_to_channel[safe_elements]
    active = steps.active & steps.valid & in_range & (channels >= 0)
    capacity = steps.step_capacity
    hit_ids = jnp.broadcast_to(jnp.arange(capacity, dtype=jnp.int32), elements.shape)
    source_indices = hit_ids
    return SensitiveHitBank(
        event_ids=steps.event_ids,
        hit_ids=hit_ids,
        detector_element_ids=elements,
        channel_ids=jnp.where(active, channels, 0),
        source_step_indices=source_indices,
        positions=0.5 * (steps.start_positions + steps.end_positions),
        times=0.5 * (steps.start_times + steps.end_times),
        energies=steps.deposited_energy,
        active=active,
        conditions_id=steps.conditions_id,
    )


class DigitizationPlan(StrictModule, NonTrainableState):
    calibration: Array
    noise_standard_deviation: Array
    crosstalk: Array
    adc_lsb: Array
    threshold: Array
    maximum_adc: int = eqx.field(static=True)
    conditions_id: str = eqx.field(static=True)
    channel_count: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        calibration: ArrayLike,
        noise_standard_deviation: ArrayLike,
        crosstalk: ArrayLike,
        /,
        *,
        adc_lsb: float,
        threshold: float,
        maximum_adc: int,
        conditions_id: str,
    ):
        calibration_ = np.asarray(calibration, dtype=np.float64)
        noise = np.asarray(noise_standard_deviation, dtype=np.float64)
        crosstalk_ = np.asarray(crosstalk, dtype=np.float64)
        if (
            calibration_.ndim != 1
            or calibration_.size < 1
            or noise.shape != calibration_.shape
        ):
            raise ValueError("Calibration and noise require one value per channel.")
        count = calibration_.size
        if crosstalk_.shape != (count, count):
            raise ValueError("crosstalk must have shape (channel_count, channel_count).")
        if (
            np.any(~np.isfinite(calibration_))
            or np.any(calibration_ < 0.0)
            or np.any(~np.isfinite(noise))
            or np.any(noise < 0.0)
        ):
            raise ValueError("Calibration and noise must be finite and nonnegative.")
        if np.any(~np.isfinite(crosstalk_)) or np.any(crosstalk_ < 0.0):
            raise ValueError("crosstalk must be finite and nonnegative.")
        lsb = float(adc_lsb)
        threshold_ = float(threshold)
        maximum = int(maximum_adc)
        conditions = str(conditions_id).strip()
        if (
            not math.isfinite(lsb)
            or lsb <= 0.0
            or not math.isfinite(threshold_)
            or threshold_ < 0.0
            or maximum < 1
            or not conditions
        ):
            raise ValueError(
                "ADC, threshold, maximum, and conditions declarations are invalid."
            )
        self.calibration = jnp.asarray(calibration_)
        self.noise_standard_deviation = jnp.asarray(noise)
        self.crosstalk = jnp.asarray(crosstalk_)
        self.adc_lsb = jnp.asarray(lsb)
        self.threshold = jnp.asarray(threshold_)
        self.maximum_adc = maximum
        self.conditions_id = conditions
        self.channel_count = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "detector-digitization-plan",
                "calibration": array_tree_fingerprint(calibration_),
                "noise": array_tree_fingerprint(noise),
                "crosstalk": array_tree_fingerprint(crosstalk_),
                "adc_lsb": lsb,
                "threshold": threshold_,
                "maximum_adc": maximum,
                "conditions": conditions,
            }
        )


class DigitizationResult(StrictModule, NonTrainableState):
    digits: DigitBank
    deposited_signal: Array
    analog_signal: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


def digitize_sensitive_hits(
    plan: DigitizationPlan,
    hits: SensitiveHitBank,
    key: Key[Array, ""],
    /,
) -> DigitizationResult:
    """Aggregate, smear, couple, quantize, saturate, and suppress channel signals."""
    if not isinstance(plan, DigitizationPlan) or not isinstance(hits, SensitiveHitBank):
        raise TypeError("plan and hits must use detector digitization types.")
    if hits.conditions_id != plan.conditions_id:
        raise ValueError("Hits and digitization conditions do not match.")
    valid_hits = hits.active & hits.valid & (hits.channel_ids < plan.channel_count)
    membership = jax.nn.one_hot(
        jnp.clip(hits.channel_ids, 0, plan.channel_count - 1),
        plan.channel_count,
        dtype=hits.energies.dtype,
    )
    deposited = ein.contract(
        "eh,ehc->ec",
        jnp.where(valid_hits, hits.energies, 0.0),
        membership,
    )
    calibrated = deposited * plan.calibration[None, :]

    def event_noise(event_id):
        event_key = jr.fold_in(key, jnp.asarray(event_id, dtype=jnp.uint32))
        return jr.normal(event_key, (plan.channel_count,), dtype=calibrated.dtype)

    noise = jax.vmap(event_noise)(hits.event_ids) * plan.noise_standard_deviation[None, :]
    coupled = ein.contract("ij,ej->ei", plan.crosstalk, calibrated)
    analog = calibrated + coupled + noise
    raw_adc = jnp.rint(analog / plan.adc_lsb)
    saturated = raw_adc > plan.maximum_adc
    adc = jnp.clip(raw_adc, 0.0, float(plan.maximum_adc))
    active = adc * plan.adc_lsb >= plan.threshold
    times = jnp.zeros_like(adc)
    digit_ids = jnp.broadcast_to(
        jnp.arange(plan.channel_count, dtype=jnp.int32), adc.shape
    )
    channels = digit_ids
    digits = DigitBank(
        event_ids=hits.event_ids,
        digit_ids=digit_ids,
        channel_ids=channels,
        signals=adc,
        times=times,
        active=active,
        saturated=saturated,
        conditions_id=hits.conditions_id,
    )
    derivative_valid = jnp.zeros_like(active)
    return DigitizationResult(digits, deposited, analog, derivative_valid, plan.plan_id)


__all__ = [
    "DigitizationPlan",
    "DigitizationResult",
    "SensitiveHitPlan",
    "digitize_sensitive_hits",
    "form_sensitive_hits",
]
