#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from .._hyperbolic_systems import AbstractAdmissibleSystem


class MulticomponentFilterResult(StrictModule):
    state: Array
    cell_mean: Array
    limiting_fraction: Array
    mean_defect: Array
    admissible: Array
    finite: Array
    successful: Array
    plan_id: str = eqx.field(static=True)


class MulticomponentAdmissibilityFilterPlan(StrictModule, NonTrainableState):
    """Mean-preserving convex filter for arbitrary admissible conserved states."""

    node_weights: Array
    iterations: int = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, node_weights: ArrayLike, /, *, iterations: int = 48):
        weights = np.asarray(node_weights, dtype=float)
        count = int(iterations)
        if (
            weights.ndim != 1
            or weights.size < 2
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
            or count <= 0
        ):
            raise ValueError(
                "Admissibility filter weights or iteration count are invalid."
            )
        weights = weights / np.sum(weights)
        self.node_weights = jnp.asarray(weights)
        self.iterations = count
        self.plan_id = canonical_fingerprint(
            {
                "kind": "multicomponent-admissibility-filter",
                "node_weights": array_tree_fingerprint(self.node_weights),
                "iterations": count,
            }
        )

    def apply(
        self, system: AbstractAdmissibleSystem, state: ArrayLike, /
    ) -> MulticomponentFilterResult:
        if not isinstance(system, AbstractAdmissibleSystem):
            raise TypeError("Filter requires AbstractAdmissibleSystem.")
        value = jnp.asarray(state)
        if (
            value.ndim < 3
            or value.shape[-2] != self.node_weights.size
            or value.shape[-1] != system.component_count
        ):
            raise ValueError("DG state must end in node and conserved-component axes.")
        mean = jnp.sum(self.node_weights[None, :, None] * value, axis=-2)
        mean_admissible = system.admissible(mean)
        lower = jnp.zeros(mean.shape[:-1], dtype=value.dtype)
        upper = jnp.ones_like(lower)

        def body(_, bounds):
            low, high = bounds
            midpoint = 0.5 * (low + high)
            candidate = mean[..., None, :] + midpoint[..., None, None] * (
                value - mean[..., None, :]
            )
            admissible = jnp.all(system.admissible(candidate), axis=-1)
            return jnp.where(admissible, midpoint, low), jnp.where(
                admissible, high, midpoint
            )

        lower, _ = jax.lax.fori_loop(0, self.iterations, body, (lower, upper))
        filtered = mean[..., None, :] + lower[..., None, None] * (
            value - mean[..., None, :]
        )
        filtered_mean = jnp.sum(self.node_weights[None, :, None] * filtered, axis=-2)
        defect = filtered_mean - mean
        admissible = jnp.all(system.admissible(filtered), axis=-1)
        finite = jnp.all(jnp.isfinite(filtered), axis=(-2, -1)) & jnp.all(
            jnp.isfinite(defect), axis=-1
        )
        scale = jnp.maximum(jnp.max(jnp.abs(mean), axis=-1), 1.0)
        successful = (
            finite
            & mean_admissible
            & admissible
            & jnp.all(
                jnp.abs(defect) <= 512.0 * jnp.finfo(value.dtype).eps * scale[..., None],
                axis=-1,
            )
        )
        return MulticomponentFilterResult(
            filtered, mean, lower, defect, admissible, finite, successful, self.plan_id
        )


__all__ = ["MulticomponentAdmissibilityFilterPlan", "MulticomponentFilterResult"]
