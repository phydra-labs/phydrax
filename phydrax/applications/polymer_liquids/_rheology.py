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


class PolymerStressCorrelationPlan(StrictModule, NonTrainableState):
    maximum_frames: int = eqx.field(static=True)
    maximum_lag: int = eqx.field(static=True)
    block_count: int = eqx.field(static=True)
    minimum_origins: int = eqx.field(static=True)
    maximum_relative_standard_error: float = eqx.field(static=True)
    maximum_stationarity_drift: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        maximum_frames: int,
        maximum_lag: int,
        block_count: int = 4,
        minimum_origins: int = 8,
        maximum_relative_standard_error: float = 0.5,
        maximum_stationarity_drift: float = 0.25,
    ):
        frames = int(maximum_frames)
        lag = int(maximum_lag)
        blocks = int(block_count)
        origins = int(minimum_origins)
        relative = float(maximum_relative_standard_error)
        drift = float(maximum_stationarity_drift)
        if (
            frames <= 1
            or lag <= 0
            or lag >= frames
            or blocks < 2
            or origins <= 0
            or not math.isfinite(relative)
            or relative < 0.0
            or not math.isfinite(drift)
            or drift < 0.0
        ):
            raise ValueError("Polymer stress-correlation controls are invalid.")
        self.maximum_frames = frames
        self.maximum_lag = lag
        self.block_count = blocks
        self.minimum_origins = origins
        self.maximum_relative_standard_error = relative
        self.maximum_stationarity_drift = drift
        self.plan_id = canonical_fingerprint(
            {
                "kind": "polymer-stress-correlation-plan",
                "maximum_frames": frames,
                "maximum_lag": lag,
                "block_count": blocks,
                "minimum_origins": origins,
                "maximum_relative_standard_error": relative,
                "maximum_stationarity_drift": drift,
            }
        )


class PolymerStressCorrelationResult(StrictModule):
    lag_times: Array
    shear_correlation: Array
    viscosity: Array
    block_viscosities: Array
    standard_error: Array
    relative_standard_error: Array
    stationarity_drift: Array
    origin_counts: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


def polymer_green_kubo_viscosity(
    plan: PolymerStressCorrelationPlan,
    stress_tensors: ArrayLike,
    time_step: float,
    volume: float,
    temperature: float,
    boltzmann_constant: float,
    /,
) -> PolymerStressCorrelationResult:
    if not isinstance(plan, PolymerStressCorrelationPlan):
        raise TypeError("plan must be PolymerStressCorrelationPlan.")
    stress = jnp.asarray(stress_tensors)
    step = float(time_step)
    volume_ = float(volume)
    thermal = float(temperature)
    boltzmann = float(boltzmann_constant)
    if (
        stress.ndim != 3
        or stress.shape[1:] != (3, 3)
        or stress.shape[0] > plan.maximum_frames
        or stress.shape[0] <= plan.maximum_lag
        or not math.isfinite(step)
        or step <= 0.0
        or not math.isfinite(volume_)
        or volume_ <= 0.0
        or not math.isfinite(thermal)
        or thermal <= 0.0
        or not math.isfinite(boltzmann)
        or boltzmann <= 0.0
    ):
        raise ValueError("Stress trajectory or thermodynamic normalization is invalid.")
    shear = jnp.stack(
        (
            0.5 * (stress[:, 0, 1] + stress[:, 1, 0]),
            0.5 * (stress[:, 0, 2] + stress[:, 2, 0]),
            0.5 * (stress[:, 1, 2] + stress[:, 2, 1]),
        ),
        axis=-1,
    )
    shear = shear - jnp.mean(shear, axis=0, keepdims=True)
    correlations = []
    counts = []
    for lag in range(plan.maximum_lag + 1):
        left = shear[: shear.shape[0] - lag]
        right = shear[lag:]
        correlations.append(jnp.mean(left * right))
        counts.append(left.shape[0])
    correlation = jnp.stack(correlations)
    origin_counts = jnp.asarray(counts, dtype=jnp.int32)
    prefactor = volume_ / (boltzmann * thermal)
    integral = step * (
        0.5 * correlation[0] + jnp.sum(correlation[1:-1]) + 0.5 * correlation[-1]
    )
    viscosity = prefactor * integral
    block_size = stress.shape[0] // plan.block_count
    block_viscosities = []
    for block_index in range(plan.block_count):
        start = block_index * block_size
        stop = (
            stress.shape[0]
            if block_index == plan.block_count - 1
            else (block_index + 1) * block_size
        )
        block = shear[start:stop]
        block_lag = min(plan.maximum_lag, int(block.shape[0]) - 1)
        block_correlation = []
        for lag in range(block_lag + 1):
            block_correlation.append(
                jnp.mean(block[: block.shape[0] - lag] * block[lag:])
            )
        values = jnp.stack(block_correlation)
        block_integral = step * (
            0.5 * values[0] + jnp.sum(values[1:-1]) + 0.5 * values[-1]
        )
        block_viscosities.append(prefactor * block_integral)
    blocks = jnp.stack(block_viscosities)
    standard_error = jnp.std(blocks, ddof=1) / jnp.sqrt(plan.block_count)
    relative_error = standard_error / jnp.maximum(
        jnp.abs(viscosity), jnp.finfo(stress.dtype).tiny
    )
    midpoint = stress.shape[0] // 2
    first_mean = jnp.mean(shear[:midpoint], axis=0)
    second_mean = jnp.mean(shear[midpoint:], axis=0)
    scale = jnp.maximum(jnp.std(shear, axis=0), jnp.finfo(stress.dtype).tiny)
    stationarity = jnp.max(jnp.abs(second_mean - first_mean) / scale)
    successful = (
        jnp.all(jnp.isfinite(stress))
        & jnp.all(jnp.isfinite(correlation))
        & jnp.isfinite(viscosity)
        & jnp.isfinite(standard_error)
        & (origin_counts[-1] >= plan.minimum_origins)
        & (relative_error <= plan.maximum_relative_standard_error)
        & (stationarity <= plan.maximum_stationarity_drift)
    )
    return PolymerStressCorrelationResult(
        jnp.arange(plan.maximum_lag + 1, dtype=stress.dtype) * step,
        correlation,
        viscosity,
        blocks,
        standard_error,
        relative_error,
        stationarity,
        origin_counts,
        successful,
        plan.plan_id,
    )


__all__ = [
    "PolymerStressCorrelationPlan",
    "PolymerStressCorrelationResult",
    "polymer_green_kubo_viscosity",
]
