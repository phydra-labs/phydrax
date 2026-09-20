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

from ...._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState
from .._core import DigitBank
from ._geometry import CalorimeterGeometry
from ._truth import CalorimeterTruth


class CalorimeterResponsePlan(StrictModule, NonTrainableState):
    geometry: CalorimeterGeometry
    gain: Array
    noise_standard_deviation: Array
    crosstalk: Array
    adc_lsb: Array
    threshold: Array
    maximum_adc: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        geometry: CalorimeterGeometry,
        /,
        *,
        gain: ArrayLike,
        noise_standard_deviation: ArrayLike,
        crosstalk: ArrayLike,
        adc_lsb: float,
        threshold: float,
        maximum_adc: int,
    ):
        if not isinstance(geometry, CalorimeterGeometry):
            raise TypeError("geometry must be CalorimeterGeometry.")
        gain_ = np.asarray(gain, dtype=np.float64)
        noise = np.asarray(noise_standard_deviation, dtype=np.float64)
        crosstalk_ = np.asarray(crosstalk, dtype=np.float64)
        count = geometry.cell_count
        if (
            gain_.shape != (count,)
            or noise.shape != (count,)
            or crosstalk_.shape != (count, count)
        ):
            raise ValueError("Calorimeter response arrays must align with cell support.")
        if (
            np.any(~np.isfinite(gain_))
            or np.any(gain_ < 0.0)
            or np.any(~np.isfinite(noise))
            or np.any(noise < 0.0)
            or np.any(~np.isfinite(crosstalk_))
            or np.any(crosstalk_ < 0.0)
        ):
            raise ValueError(
                "Calorimeter response arrays must be finite and nonnegative."
            )
        lsb = float(adc_lsb)
        threshold_ = float(threshold)
        maximum = int(maximum_adc)
        if (
            not math.isfinite(lsb)
            or lsb <= 0.0
            or not math.isfinite(threshold_)
            or threshold_ < 0.0
            or maximum < 1
        ):
            raise ValueError("Calorimeter ADC declaration is invalid.")
        self.geometry = geometry
        self.gain = jnp.asarray(gain_)
        self.noise_standard_deviation = jnp.asarray(noise)
        self.crosstalk = jnp.asarray(crosstalk_)
        self.adc_lsb = jnp.asarray(lsb)
        self.threshold = jnp.asarray(threshold_)
        self.maximum_adc = maximum
        self.plan_id = canonical_fingerprint(
            {
                "kind": "calorimeter-response-plan",
                "geometry": geometry.geometry_id,
                "gain": array_tree_fingerprint(gain_),
                "noise": array_tree_fingerprint(noise),
                "crosstalk": array_tree_fingerprint(crosstalk_),
                "adc_lsb": lsb,
                "threshold": threshold_,
                "maximum_adc": maximum,
            }
        )


class CalorimeterResponse(StrictModule, NonTrainableState):
    digits: DigitBank
    analog_signal: Array
    reconstructed_cell_energy: Array
    derivative_valid: Array
    plan_id: str = eqx.field(static=True)


def apply_calorimeter_response(
    plan: CalorimeterResponsePlan,
    truth: CalorimeterTruth,
    key: Key[Array, ""],
    /,
) -> CalorimeterResponse:
    if not isinstance(plan, CalorimeterResponsePlan) or not isinstance(
        truth, CalorimeterTruth
    ):
        raise TypeError("plan and truth must use calorimeter types.")
    if truth.geometry_id != plan.geometry.geometry_id:
        raise ValueError("Calorimeter truth and response geometry differ.")

    def event_noise(event_id):
        event_key = jr.fold_in(key, jnp.asarray(event_id, dtype=jnp.uint32))
        return jr.normal(
            event_key, (plan.geometry.cell_count,), dtype=truth.cell_energies.dtype
        )

    calibrated = truth.cell_energies * plan.gain[None, :]
    coupled = ein.contract("ij,ej->ei", plan.crosstalk, calibrated)
    noise = (
        jax.vmap(event_noise)(truth.event_ids) * plan.noise_standard_deviation[None, :]
    )
    analog = calibrated + coupled + noise
    raw_adc = jnp.rint(analog / plan.adc_lsb)
    saturated = raw_adc > plan.maximum_adc
    adc = jnp.clip(raw_adc, 0.0, float(plan.maximum_adc))
    active = (
        (adc * plan.adc_lsb >= plan.threshold)
        & plan.geometry.active[None, :]
        & ~plan.geometry.dead[None, :]
    )
    reconstructed = jnp.where(
        active,
        adc * plan.adc_lsb / jnp.maximum(plan.gain[None, :], jnp.finfo(adc.dtype).tiny),
        0.0,
    )
    shape = adc.shape
    digit_ids = jnp.broadcast_to(jnp.arange(shape[1], dtype=jnp.int32), shape)
    digits = DigitBank(
        event_ids=truth.event_ids,
        digit_ids=digit_ids,
        channel_ids=jnp.broadcast_to(plan.geometry.channel_ids, shape),
        signals=adc,
        times=jnp.zeros_like(adc),
        active=active,
        saturated=saturated,
        conditions_id=plan.geometry.conditions_id,
    )
    return CalorimeterResponse(
        digits,
        analog,
        reconstructed,
        jnp.zeros_like(active),
        plan.plan_id,
    )


__all__ = [
    "CalorimeterResponse",
    "CalorimeterResponsePlan",
    "apply_calorimeter_response",
]
