#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal, TypeAlias

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._interpolation._piecewise import linear_interpolate
from .._strict import StrictModule
from .._trainable import NonTrainableState


RadiativeCoolingBoundsPolicy: TypeAlias = Literal["error", "power_law_extrapolate"]


class TabulatedCoolingEvaluation(StrictModule):
    rate: Array
    supported: Array
    log_temperature: Array
    log_rate: Array


class TabulatedCoolingCurve(StrictModule, NonTrainableState):
    """Positive cooling coefficient tabulated in base-10 log coordinates."""

    log_temperature_nodes: Array
    log_rate_values: Array
    temperature_scale: float = eqx.field(static=True)
    rate_scale: float = eqx.field(static=True)
    bounds_policy: RadiativeCoolingBoundsPolicy = eqx.field(static=True)
    curve_id: str = eqx.field(static=True)

    def __init__(
        self,
        log_temperature_nodes: ArrayLike,
        log_rate_values: ArrayLike,
        /,
        *,
        temperature_scale: float = 1.0,
        rate_scale: float = 1.0,
        bounds_policy: RadiativeCoolingBoundsPolicy = "error",
    ):
        nodes = np.asarray(log_temperature_nodes, dtype=float)
        values = np.asarray(log_rate_values, dtype=float)
        temperature_scale_ = float(temperature_scale)
        rate_scale_ = float(rate_scale)
        if (
            nodes.ndim != 1
            or values.shape != nodes.shape
            or nodes.size < 2
            or np.any(~np.isfinite(nodes))
            or np.any(~np.isfinite(values))
            or np.any(np.diff(nodes) <= 0.0)
            or not np.isfinite(temperature_scale_)
            or temperature_scale_ <= 0.0
            or not np.isfinite(rate_scale_)
            or rate_scale_ <= 0.0
        ):
            raise ValueError("Cooling table and code-unit scales are invalid.")
        if bounds_policy not in ("error", "power_law_extrapolate"):
            raise ValueError("Unknown radiative cooling bounds policy.")
        self.log_temperature_nodes = jnp.asarray(nodes)
        self.log_rate_values = jnp.asarray(values)
        self.temperature_scale = temperature_scale_
        self.rate_scale = rate_scale_
        self.bounds_policy = bounds_policy
        self.curve_id = canonical_fingerprint(
            {
                "kind": "tabulated-radiative-cooling",
                "nodes": array_tree_fingerprint(nodes),
                "values": array_tree_fingerprint(values),
                "temperature_scale": temperature_scale_,
                "rate_scale": rate_scale_,
                "bounds_policy": bounds_policy,
            }
        )

    def evaluate(self, temperature: ArrayLike, /) -> TabulatedCoolingEvaluation:
        value = jnp.asarray(temperature)
        positive = jnp.isfinite(value) & (value > 0.0)
        safe = jnp.where(positive, value, 1.0)
        log_temperature = jnp.log10(safe * self.temperature_scale)
        bounds = "fill" if self.bounds_policy == "error" else "extrapolate"
        interpolated = linear_interpolate(
            self.log_temperature_nodes,
            self.log_rate_values,
            log_temperature,
            bounds=bounds,
            fill_value=0.0,
        )
        supported = positive & interpolated.support
        log_rate = interpolated.values
        rate = self.rate_scale * 10.0**log_rate
        rate = jnp.where(supported, rate, 0.0)
        return TabulatedCoolingEvaluation(rate, supported, log_temperature, log_rate)


__all__ = [
    "RadiativeCoolingBoundsPolicy",
    "TabulatedCoolingCurve",
    "TabulatedCoolingEvaluation",
]
