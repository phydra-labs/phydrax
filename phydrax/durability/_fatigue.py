#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""ASTM-style rainflow counting and mean-stress-corrected fatigue damage."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class RainflowCycles:
    stress_range_pa: Array
    mean_stress_pa: Array
    cycle_count: Array


@dataclass(frozen=True, slots=True)
class SNCurve:
    fatigue_strength_coefficient_pa: float
    fatigue_strength_exponent: float

    def __post_init__(self):
        if self.fatigue_strength_coefficient_pa <= 0:
            raise ValueError("Fatigue strength coefficient must be positive.")
        if self.fatigue_strength_exponent >= 0:
            raise ValueError("Basquin fatigue exponent must be negative.")

    def cycles_to_failure(self, stress_amplitude_pa: ArrayLike, /) -> Array:
        amplitude = jnp.asarray(stress_amplitude_pa)
        if bool(jnp.any(amplitude <= 0)):
            raise ValueError("Fatigue stress amplitudes must be positive.")
        reversals = (amplitude / self.fatigue_strength_coefficient_pa) ** (
            1.0 / self.fatigue_strength_exponent
        )
        return 0.5 * reversals


@dataclass(frozen=True, slots=True)
class FatigueResult:
    cycles: RainflowCycles
    corrected_amplitude_pa: Array
    cycles_to_failure: Array
    miner_damage: Array
    repeat_blocks_to_failure: Array
    successful: Array


def _reversals(history: np.ndarray) -> list[float]:
    distinct = [float(history[0])]
    for value in history[1:]:
        if value != distinct[-1]:
            distinct.append(float(value))
    if len(distinct) <= 2:
        return distinct
    turning = [distinct[0]]
    for left, center, right in zip(distinct, distinct[1:], distinct[2:], strict=False):
        if (center - left) * (right - center) <= 0:
            turning.append(center)
    turning.append(distinct[-1])
    return turning


def rainflow_cycles(stress_history_pa: ArrayLike, /) -> RainflowCycles:
    history = np.asarray(stress_history_pa, dtype=float)
    if history.ndim != 1 or history.size < 2 or not np.all(np.isfinite(history)):
        raise ValueError(
            "Rainflow history must be a finite vector with at least two samples."
        )
    stack: list[float] = []
    ranges: list[float] = []
    means: list[float] = []
    counts: list[float] = []
    for point in _reversals(history):
        stack.append(point)
        while len(stack) >= 3:
            previous_range = abs(stack[-2] - stack[-3])
            current_range = abs(stack[-1] - stack[-2])
            if previous_range > current_range:
                break
            ranges.append(previous_range)
            means.append(0.5 * (stack[-3] + stack[-2]))
            if len(stack) == 3:
                counts.append(0.5)
                stack.pop(-3)
            else:
                counts.append(1.0)
                del stack[-3:-1]
    for left, right in pairwise(stack):
        ranges.append(abs(right - left))
        means.append(0.5 * (left + right))
        counts.append(0.5)
    nonzero = np.asarray(ranges) > 0
    return RainflowCycles(
        jnp.asarray(np.asarray(ranges)[nonzero]),
        jnp.asarray(np.asarray(means)[nonzero]),
        jnp.asarray(np.asarray(counts)[nonzero]),
    )


@dataclass(frozen=True, slots=True)
class FatigueAssessment:
    sn_curve: SNCurve
    ultimate_strength_pa: float

    def __post_init__(self):
        if self.ultimate_strength_pa <= 0:
            raise ValueError("Fatigue ultimate strength must be positive.")

    def evaluate(self, stress_history_pa: ArrayLike, /) -> FatigueResult:
        cycles = rainflow_cycles(stress_history_pa)
        amplitude = 0.5 * cycles.stress_range_pa
        denominator = (
            1.0 - jnp.maximum(cycles.mean_stress_pa, 0) / self.ultimate_strength_pa
        )
        admissible = jnp.all(denominator > 0)
        corrected = amplitude / jnp.maximum(denominator, jnp.finfo(amplitude.dtype).tiny)
        life = self.sn_curve.cycles_to_failure(corrected)
        damage = jnp.sum(cycles.cycle_count / life)
        blocks = jnp.where(damage > 0, 1 / damage, jnp.inf)
        successful = (
            admissible
            & jnp.isfinite(damage)
            & jnp.all(life > 0)
            & jnp.all(jnp.isfinite(corrected))
        )
        return FatigueResult(cycles, corrected, life, damage, blocks, successful)


__all__ = [
    "FatigueAssessment",
    "FatigueResult",
    "RainflowCycles",
    "SNCurve",
    "rainflow_cycles",
]
