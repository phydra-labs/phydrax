#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Component-conservative staged mineral separation with internal recycle."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike


@dataclass(frozen=True, slots=True)
class MineralCircuitResult:
    stage_feed_kg_s: Array
    stage_concentrate_kg_s: Array
    stage_tailings_kg_s: Array
    stage_recycle_kg_s: Array
    product_concentrate_kg_s: Array
    final_tailings_kg_s: Array
    component_balance_residual_kg_s: Array
    successful: Array


@dataclass(frozen=True, slots=True)
class RecycleSeparationCircuit:
    component_recovery: Array
    tailings_recycle_fraction: Array

    @classmethod
    def create(
        cls,
        component_recovery: ArrayLike,
        tailings_recycle_fraction: ArrayLike,
        /,
    ) -> RecycleSeparationCircuit:
        recovery = np.asarray(component_recovery, dtype=float)
        recycle = np.asarray(tailings_recycle_fraction, dtype=float)
        if recovery.ndim != 2 or recovery.shape[0] == 0 or recovery.shape[1] == 0:
            raise ValueError("Mineral recovery requires stage-by-component shape.")
        if recycle.ndim == 1:
            recycle = np.broadcast_to(recycle[:, None], recovery.shape)
        if recycle.shape != recovery.shape:
            raise ValueError(
                "Mineral recycle fractions must align with stage recoveries."
            )
        if np.any((recovery < 0) | (recovery > 1)) or np.any(
            (recycle < 0) | (recycle >= 1)
        ):
            raise ValueError("Mineral recovery must lie in [0,1] and recycle in [0,1).")
        denominator = 1 - recycle * (1 - recovery)
        if np.any(denominator <= 0):
            raise ValueError("Mineral internal recycle has no finite fixed point.")
        return cls(jnp.asarray(recovery), jnp.asarray(recycle))

    def solve(self, fresh_feed_kg_s: ArrayLike, /) -> MineralCircuitResult:
        fresh = jnp.asarray(fresh_feed_kg_s)
        components = self.component_recovery.shape[1]
        if fresh.shape != (components,) or bool(jnp.any(fresh < 0)):
            raise ValueError(
                "Mineral fresh feed must be a non-negative component vector."
            )
        incoming = fresh
        stage_feed = []
        concentrate = []
        tailings = []
        recycle = []
        product = jnp.zeros_like(fresh)
        for stage in range(self.component_recovery.shape[0]):
            recovery = self.component_recovery[stage]
            recycle_fraction = self.tailings_recycle_fraction[stage]
            total_feed = incoming / (1 - recycle_fraction * (1 - recovery))
            stage_concentrate = recovery * total_feed
            gross_tailings = (1 - recovery) * total_feed
            stage_recycle = recycle_fraction * gross_tailings
            outgoing_tailings = gross_tailings - stage_recycle
            stage_feed.append(total_feed)
            concentrate.append(stage_concentrate)
            tailings.append(outgoing_tailings)
            recycle.append(stage_recycle)
            product = product + stage_concentrate
            incoming = outgoing_tailings
        residual = product + incoming - fresh
        return MineralCircuitResult(
            jnp.stack(stage_feed),
            jnp.stack(concentrate),
            jnp.stack(tailings),
            jnp.stack(recycle),
            product,
            incoming,
            residual,
            jnp.all(jnp.isfinite(product)) & jnp.all(product >= 0),
        )


__all__ = ["MineralCircuitResult", "RecycleSeparationCircuit"]
