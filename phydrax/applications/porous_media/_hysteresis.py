#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._strict import StrictModule


class BrooksCoreyRetention(StrictModule):
    entry_pressure_Pa: Array
    pore_size_index: Array
    residual_wetting_saturation: Array
    residual_nonwetting_saturation: Array

    def __init__(
        self,
        entry_pressure_Pa: ArrayLike,
        pore_size_index: ArrayLike,
        residual_wetting_saturation: ArrayLike = 0.0,
        residual_nonwetting_saturation: ArrayLike = 0.0,
        /,
    ):
        entry, index, wetting, nonwetting = jnp.broadcast_arrays(
            jnp.asarray(entry_pressure_Pa),
            jnp.asarray(pore_size_index),
            jnp.asarray(residual_wetting_saturation),
            jnp.asarray(residual_nonwetting_saturation),
        )
        invalid = (
            jnp.any(~jnp.isfinite(entry))
            | jnp.any(entry <= 0)
            | jnp.any(~jnp.isfinite(index))
            | jnp.any(index <= 0)
            | jnp.any(~jnp.isfinite(wetting))
            | jnp.any(wetting < 0)
            | jnp.any(~jnp.isfinite(nonwetting))
            | jnp.any(nonwetting < 0)
            | jnp.any(wetting + nonwetting >= 1)
        )
        self.entry_pressure_Pa = eqx.error_if(
            entry, invalid, "Brooks-Corey parameters are outside their physical domain."
        )
        self.pore_size_index, self.residual_wetting_saturation = index, wetting
        self.residual_nonwetting_saturation = nonwetting

    def effective_saturation(self, capillary_pressure_Pa: ArrayLike, /) -> Array:
        pressure = jnp.asarray(capillary_pressure_Pa)
        pressure = eqx.error_if(
            pressure,
            jnp.any(~jnp.isfinite(pressure)) | jnp.any(pressure < 0),
            "Capillary pressure must be finite and nonnegative.",
        )
        return jnp.where(
            pressure <= self.entry_pressure_Pa,
            1.0,
            (self.entry_pressure_Pa / pressure) ** self.pore_size_index,
        )

    def saturation(self, capillary_pressure_Pa: ArrayLike, /) -> Array:
        effective = self.effective_saturation(capillary_pressure_Pa)
        span = (
            1.0 - self.residual_wetting_saturation - self.residual_nonwetting_saturation
        )
        return self.residual_wetting_saturation + span * effective

    def relative_permeability(
        self, capillary_pressure_Pa: ArrayLike, /
    ) -> tuple[Array, Array]:
        effective = self.effective_saturation(capillary_pressure_Pa)
        wetting = effective ** (3.0 + 2.0 / self.pore_size_index)
        nonwetting = (1.0 - effective) ** 2 * (
            1.0 - effective ** (1.0 + 2.0 / self.pore_size_index)
        )
        return wetting, nonwetting


class HysteresisState(StrictModule):
    capillary_pressure_Pa: Array
    saturation: Array
    reversal_pressure_Pa: Array
    reversal_saturation: Array
    branch: Array
    derivative_available: Array


class HystereticRetentionPlan(StrictModule):
    drainage: BrooksCoreyRetention
    imbibition: BrooksCoreyRetention

    def __init__(
        self, drainage: BrooksCoreyRetention, imbibition: BrooksCoreyRetention, /
    ):
        if not isinstance(drainage, BrooksCoreyRetention) or not isinstance(
            imbibition, BrooksCoreyRetention
        ):
            raise TypeError(
                "Hysteresis requires drainage and imbibition Brooks-Corey curves."
            )
        self.drainage, self.imbibition = drainage, imbibition

    def initialize(
        self,
        capillary_pressure_Pa: ArrayLike,
        /,
        *,
        branch: Literal["drainage", "imbibition"],
    ) -> HysteresisState:
        if branch not in ("drainage", "imbibition"):
            raise ValueError("Hysteresis branch must be drainage or imbibition.")
        pressure = jnp.asarray(capillary_pressure_Pa)
        curve = self.drainage if branch == "drainage" else self.imbibition
        saturation = curve.saturation(pressure)
        code = jnp.asarray(1 if branch == "drainage" else -1, dtype=jnp.int32)
        return HysteresisState(
            pressure,
            saturation,
            pressure,
            saturation,
            jnp.broadcast_to(code, pressure.shape),
            jnp.asarray(True),
        )

    def update(
        self, state: HysteresisState, capillary_pressure_Pa: ArrayLike, /
    ) -> HysteresisState:
        pressure = jnp.asarray(capillary_pressure_Pa)
        if pressure.shape != state.capillary_pressure_Pa.shape:
            raise ValueError("Hysteresis pressure shape changed.")
        direction = jnp.sign(pressure - state.capillary_pressure_Pa).astype(jnp.int32)
        direction = jnp.where(direction == 0, state.branch, direction)
        reversed_ = direction != state.branch
        reversal_pressure = jnp.where(
            reversed_, state.capillary_pressure_Pa, state.reversal_pressure_Pa
        )
        reversal_saturation = jnp.where(
            reversed_, state.saturation, state.reversal_saturation
        )
        drainage_value = self.drainage.saturation(pressure)
        imbibition_value = self.imbibition.saturation(pressure)
        drainage_reversal = self.drainage.saturation(reversal_pressure)
        imbibition_reversal = self.imbibition.saturation(reversal_pressure)
        drainage_span = jnp.maximum(drainage_reversal, 1e-12)
        imbibition_span = jnp.maximum(1.0 - imbibition_reversal, 1e-12)
        scanning_drainage = reversal_saturation * drainage_value / drainage_span
        scanning_imbibition = (
            1.0 - (1.0 - reversal_saturation) * (1.0 - imbibition_value) / imbibition_span
        )
        saturation = jnp.where(direction > 0, scanning_drainage, scanning_imbibition)
        tolerance = 100 * jnp.finfo(saturation.dtype).eps
        saturation = eqx.error_if(
            saturation,
            jnp.any((saturation < -tolerance) | (saturation > 1.0 + tolerance)),
            "Hysteretic scanning curve left its physical saturation envelope.",
        )
        saturation = jnp.minimum(jnp.maximum(saturation, 0.0), 1.0)
        return HysteresisState(
            pressure,
            saturation,
            reversal_pressure,
            reversal_saturation,
            direction,
            ~jnp.any(reversed_),
        )


class DynamicCapillaryPressure(StrictModule):
    relaxation_Pa_s: Array

    def __init__(self, relaxation_Pa_s: ArrayLike, /):
        value = jnp.asarray(relaxation_Pa_s)
        self.relaxation_Pa_s = eqx.error_if(
            value,
            jnp.any(~jnp.isfinite(value)) | jnp.any(value < 0),
            "Dynamic capillary coefficient must be finite and nonnegative.",
        )

    def evaluate(
        self,
        equilibrium_pressure_Pa: ArrayLike,
        saturation_rate_s_inverse: ArrayLike,
        /,
    ) -> Array:
        equilibrium, rate = jnp.broadcast_arrays(
            jnp.asarray(equilibrium_pressure_Pa),
            jnp.asarray(saturation_rate_s_inverse),
        )
        equilibrium = eqx.error_if(
            equilibrium,
            jnp.any(~jnp.isfinite(equilibrium)) | jnp.any(~jnp.isfinite(rate)),
            "Dynamic capillary pressure inputs must be finite.",
        )
        return equilibrium - self.relaxation_Pa_s * rate


__all__ = [
    "BrooksCoreyRetention",
    "DynamicCapillaryPressure",
    "HysteresisState",
    "HystereticRetentionPlan",
]
