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


class ResolvedLubricationResult(StrictModule):
    force: Array
    resistance: Array
    asymptotic_resistance: Array
    resolved_resistance: Array
    dissipation_rate: Array
    active: Array
    finite: Array
    plan_id: str = eqx.field(static=True)


class ResolvedLubricationCorrectionPlan(StrictModule, NonTrainableState):
    """Near-gap normal resistance minus the resistance resolved by the grid."""

    dynamic_viscosity: Array
    cutoff: Array
    minimum_gap: Array
    plan_id: str = eqx.field(static=True)

    def __init__(
        self,
        dynamic_viscosity: ArrayLike,
        cutoff: ArrayLike,
        minimum_gap: ArrayLike,
        /,
    ):
        viscosity = np.asarray(dynamic_viscosity)
        cutoff_ = np.asarray(cutoff)
        minimum = np.asarray(minimum_gap)
        if viscosity.shape != cutoff_.shape or cutoff_.shape != minimum.shape:
            raise ValueError("Lubrication parameters must have one shared shape.")
        if any(np.any(~np.isfinite(value)) for value in (viscosity, cutoff_, minimum)):
            raise ValueError("Lubrication parameters must be finite.")
        if (
            np.any(viscosity <= 0.0)
            or np.any(minimum <= 0.0)
            or np.any(cutoff_ <= minimum)
        ):
            raise ValueError(
                "Lubrication requires positive viscosity and 0 < minimum_gap < cutoff."
            )
        self.dynamic_viscosity = jnp.asarray(viscosity)
        self.cutoff = jnp.asarray(cutoff_)
        self.minimum_gap = jnp.asarray(minimum)
        self.plan_id = canonical_fingerprint(
            {
                "kind": "resolved-lubrication-correction",
                "shape": viscosity.shape,
                "dynamic_viscosity": viscosity.tolist(),
                "cutoff": cutoff_.tolist(),
                "minimum_gap": minimum.tolist(),
            }
        )

    def evaluate(
        self,
        gap: ArrayLike,
        normal: ArrayLike,
        normal_velocity: ArrayLike,
        effective_radius: ArrayLike,
        /,
        *,
        resolved_resistance: ArrayLike = 0.0,
        valid: ArrayLike | None = None,
    ) -> ResolvedLubricationResult:
        gap_ = jnp.asarray(gap)
        normal_ = jnp.asarray(normal, dtype=gap_.dtype)
        velocity = jnp.asarray(normal_velocity, dtype=gap_.dtype)
        radius = jnp.asarray(effective_radius, dtype=gap_.dtype)
        resolved = jnp.asarray(resolved_resistance, dtype=gap_.dtype)
        if normal_.shape != gap_.shape + (normal_.shape[-1],):
            raise ValueError("normal must have one vector per gap.")
        if velocity.shape != gap_.shape or radius.shape != gap_.shape:
            raise ValueError("Gap, velocity, and effective radius shapes differ.")
        active_mask = (
            jnp.ones(gap_.shape, dtype=bool)
            if valid is None
            else jnp.asarray(valid, dtype=bool)
        )
        minimum = jnp.broadcast_to(self.minimum_gap, gap_.shape)
        cutoff = jnp.broadcast_to(self.cutoff, gap_.shape)
        viscosity = jnp.broadcast_to(self.dynamic_viscosity, gap_.shape)
        safe_gap = jnp.maximum(gap_, minimum)
        coordinate = jnp.clip((safe_gap - minimum) / (cutoff - minimum), 0.0, 1.0)
        switch = (1.0 - coordinate) ** 2 * (1.0 + 2.0 * coordinate)
        asymptotic = (
            6.0
            * jnp.pi
            * viscosity
            * radius**2
            * jnp.maximum(1.0 / safe_gap - 1.0 / cutoff, 0.0)
            * switch
        )
        resistance = jnp.maximum(asymptotic - resolved, 0.0)
        active = active_mask & (gap_ > 0.0) & (gap_ < cutoff)
        resistance = jnp.where(active, resistance, 0.0)
        force_scalar = -resistance * velocity
        force = force_scalar[..., None] * normal_
        dissipation = resistance * velocity**2
        finite = jnp.all(
            jnp.isfinite(force)
            & jnp.isfinite(resistance)[..., None]
            & jnp.isfinite(dissipation)[..., None]
        )
        return ResolvedLubricationResult(
            force,
            resistance,
            asymptotic,
            resolved,
            dissipation,
            active,
            finite,
            self.plan_id,
        )


__all__ = [
    "ResolvedLubricationCorrectionPlan",
    "ResolvedLubricationResult",
]
