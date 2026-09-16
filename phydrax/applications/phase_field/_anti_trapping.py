#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from phydrax import ein

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class AntiTrappingCurrentEvaluation(StrictModule):
    current: Array
    interface_normal: Array
    amplitude: Array
    current_magnitude: Array
    active_interface: Array
    regularized_fraction: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class AntiTrappingCurrentPlan(StrictModule, NonTrainableState):
    """Quantitative alloy anti-trapping current with declared calibration."""

    calibration_coefficient: Array
    interface_width: Array
    partition_coefficient: Array
    gradient_tolerance: float = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        calibration_coefficient: ArrayLike,
        interface_width: ArrayLike,
        partition_coefficient: ArrayLike,
        /,
        *,
        gradient_tolerance: float = 1.0e-12,
        calibration_id: str,
    ):
        values = tuple(
            np.asarray(value)
            for value in (
                calibration_coefficient,
                interface_width,
                partition_coefficient,
            )
        )
        tolerance = float(gradient_tolerance)
        identifier = str(calibration_id)
        if (
            any(value.shape != () or not np.isfinite(value) for value in values)
            or values[0] < 0.0
            or values[1] <= 0.0
            or values[2] <= 0.0
            or values[2] > 1.0
            or not np.isfinite(tolerance)
            or tolerance <= 0.0
            or not identifier
        ):
            raise ValueError("Anti-trapping calibration is invalid.")
        self.calibration_coefficient = jnp.asarray(values[0])
        self.interface_width = jnp.asarray(values[1])
        self.partition_coefficient = jnp.asarray(values[2])
        self.gradient_tolerance = tolerance
        self.plan_id = canonical_fingerprint(
            {
                "kind": "anti-trapping-current-plan",
                "calibration_id": identifier,
                "calibration_coefficient": float(values[0]),
                "interface_width": float(values[1]),
                "partition_coefficient": float(values[2]),
                "gradient_tolerance": tolerance,
            }
        )

    def evaluate(
        self,
        phase_rate: ArrayLike,
        phase_gradient: ArrayLike,
        supersaturation: ArrayLike,
        /,
    ) -> AntiTrappingCurrentEvaluation:
        rate = jnp.asarray(phase_rate)
        gradient = jnp.asarray(phase_gradient, dtype=rate.dtype)
        solute = jnp.asarray(supersaturation, dtype=rate.dtype)
        if gradient.shape[:-1] != rate.shape or solute.shape != rate.shape:
            raise ValueError("Anti-trapping fields have incompatible shapes.")
        gradient_norm = jnp.sqrt(
            jnp.maximum(ein.contract("...d,...d->...", gradient, gradient), 0.0)
        )
        active = gradient_norm > self.gradient_tolerance
        safe_norm = jnp.where(active, gradient_norm, 1.0)
        normal = jnp.where(active[..., None], gradient / safe_norm[..., None], 0.0)
        amplitude = (
            self.calibration_coefficient.astype(rate.dtype)
            * self.interface_width.astype(rate.dtype)
            * (1.0 + (1.0 - self.partition_coefficient.astype(rate.dtype)) * solute)
            * rate
        )
        current = -amplitude[..., None] * normal
        magnitude = jnp.sqrt(
            jnp.maximum(ein.contract("...d,...d->...", current, current), 0.0)
        )
        regularized_fraction = 1.0 - jnp.mean(active.astype(rate.dtype))
        finite = (
            jnp.all(jnp.isfinite(current))
            & jnp.all(jnp.isfinite(amplitude))
            & jnp.isfinite(regularized_fraction)
        )
        admissible = jnp.all(
            1.0 + (1.0 - self.partition_coefficient.astype(rate.dtype)) * solute >= 0.0
        )
        return AntiTrappingCurrentEvaluation(
            current,
            normal,
            amplitude,
            magnitude,
            active,
            regularized_fraction,
            finite,
            finite & admissible,
            self.plan_id,
        )


__all__ = ["AntiTrappingCurrentEvaluation", "AntiTrappingCurrentPlan"]
