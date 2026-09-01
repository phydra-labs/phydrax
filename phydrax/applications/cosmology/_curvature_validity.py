#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ._background import FLRWBackground


class LocalCurvatureValidityResult(StrictModule):
    curvature_radius: Array
    support_ratio: Array
    transverse_distance_indicator: Array
    volume_indicator: Array
    maximum_indicator: Array
    within_budget: Array
    finite: Array
    successful: Array
    support_kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)


class LocalCurvatureValidityPlan(StrictModule, NonTrainableState):
    """Quantify local-flat geometry indicators without enabling curved PM."""

    light_speed: float = eqx.field(static=True)
    geometry_error_budget: float = eqx.field(static=True)
    support_kind: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        light_speed: float,
        geometry_error_budget: float,
        support_kind: str,
    ):
        speed = float(light_speed)
        budget = float(geometry_error_budget)
        kind = str(support_kind).strip()
        if (
            not np.isfinite(speed)
            or speed <= 0.0
            or not np.isfinite(budget)
            or budget <= 0.0
            or not kind
        ):
            raise ValueError("Local-curvature validity policy is invalid.")
        self.light_speed = speed
        self.geometry_error_budget = budget
        self.support_kind = kind
        self.plan_id = canonical_fingerprint(
            {
                "kind": "local-curvature-validity",
                "light_speed": speed,
                "geometry_error_budget": budget,
                "support_kind": kind,
            }
        )

    def evaluate(
        self, background: FLRWBackground, support_length: ArrayLike, /
    ) -> LocalCurvatureValidityResult:
        length = jnp.asarray(support_length, dtype=background.hubble_constant.dtype)
        if length.shape != ():
            raise ValueError("Curvature support length must be scalar.")
        length = eqx.error_if(
            length,
            ~jnp.isfinite(length) | (length <= 0.0),
            "Curvature support length must be finite and positive.",
        )
        absolute_curvature = jnp.abs(background.curvature_density)
        flat = absolute_curvature == 0.0
        radius = jnp.where(
            flat,
            jnp.asarray(jnp.inf, dtype=length.dtype),
            self.light_speed
            / (
                background.hubble_constant
                * jnp.sqrt(jnp.maximum(absolute_curvature, jnp.finfo(length.dtype).tiny))
            ),
        )
        ratio = jnp.where(flat, 0.0, length / radius)
        distance_indicator = ratio**2 / 6.0
        volume_indicator = ratio**2 / 5.0
        maximum = jnp.maximum(distance_indicator, volume_indicator)
        finite = jnp.isfinite(maximum)
        within = maximum <= self.geometry_error_budget
        return LocalCurvatureValidityResult(
            radius,
            ratio,
            jnp.sign(background.curvature_density) * distance_indicator,
            jnp.sign(background.curvature_density) * volume_indicator,
            maximum,
            within,
            finite,
            within & finite,
            self.support_kind,
            self.plan_id,
        )


__all__ = ["LocalCurvatureValidityPlan", "LocalCurvatureValidityResult"]
