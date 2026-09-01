#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ...._fingerprint import canonical_fingerprint
from ...._strict import StrictModule
from ...._trainable import NonTrainableState


MechanicsCaseReductionKind = Literal["weighted_mean", "mean", "cvar", "max"]


class MechanicsCaseReductionResult(StrictModule):
    """One outer risk evaluation over already-reduced physical cases."""

    value: Array
    case_values: Array
    normalized_weights: Array
    effective_sample_size: Array
    threshold: Array
    tail_mass: Array
    threshold_tie_fraction: Array
    kind: MechanicsCaseReductionKind = eqx.field(static=True)
    case_count: int = eqx.field(static=True)
    batch_dependent: bool = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)


class MechanicsCaseReduction(StrictModule, NonTrainableState):
    """Reduce complete physical-case scalars without changing their support.

    Spatial integration belongs to the mechanics problem. This reducer accepts
    exactly one scalar per physical case and refuses nonfinite or explicitly
    invalid cases even when a supplied probability weight would be zero.
    """

    kind: MechanicsCaseReductionKind = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    reduction_id: str = eqx.field(static=True)

    def __init__(
        self,
        kind: MechanicsCaseReductionKind,
        /,
        *,
        alpha: float = 0.95,
    ):
        if kind not in ("weighted_mean", "mean", "cvar", "max"):
            raise ValueError(
                "Mechanics case reduction must be 'weighted_mean', 'mean', "
                "'cvar', or 'max'."
            )
        level = float(alpha)
        if not math.isfinite(level) or level < 0.0 or level >= 1.0:
            raise ValueError("CVaR alpha must be finite and lie in [0, 1).")
        self.kind = kind
        self.alpha = level
        self.reduction_id = canonical_fingerprint(
            {
                "kind": "mechanics-case-reduction",
                "reduction": kind,
                **(
                    {"alpha": level, "tie_rule": "proportional-threshold-mass"}
                    if kind == "cvar"
                    else {"scope": "batch"}
                    if kind == "max"
                    else {}
                ),
            }
        )

    def evaluate(
        self,
        case_values: ArrayLike,
        /,
        *,
        probability_weights: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ) -> MechanicsCaseReductionResult:
        """Return risk value and support diagnostics for a fixed physical batch."""
        values = jnp.asarray(case_values)
        if values.ndim != 1 or int(values.shape[0]) == 0:
            raise ValueError("case_values must be a non-empty rank-one array.")
        if jnp.iscomplexobj(values):
            raise TypeError("case_values must be real.")
        if not jnp.issubdtype(values.dtype, jnp.inexact):
            values = values.astype(float)
        values = eqx.error_if(
            values,
            jnp.any(~jnp.isfinite(values)),
            "Every physical case must have a finite reduced scalar.",
        )
        if valid is not None:
            support = jnp.asarray(valid, dtype=bool)
            if support.shape != values.shape:
                raise ValueError("valid must have exactly the case_values shape.")
            values = eqx.error_if(
                values,
                jnp.any(~support),
                "Invalid physical cases cannot be dropped from outer risk reduction.",
            )

        count = int(values.shape[0])
        if probability_weights is None:
            if self.kind in ("weighted_mean", "cvar"):
                raise ValueError(
                    f"{self.kind} requires explicit probability or importance weights."
                )
            weights = jnp.full(values.shape, 1.0 / count, dtype=values.dtype)
        else:
            weights = jnp.asarray(probability_weights, dtype=values.dtype)
            if weights.shape != values.shape:
                raise ValueError(
                    "probability_weights must have exactly the case_values shape."
                )
            weights = eqx.error_if(
                weights,
                jnp.any(~jnp.isfinite(weights)) | jnp.any(weights <= 0.0),
                "Case probability weights must be finite and strictly positive.",
            )
            total = jnp.sum(weights)
            weights = eqx.error_if(
                weights,
                ~jnp.isfinite(total) | (total <= 0.0),
                "Case probability weight mass must be finite and positive.",
            )
            weights = weights / total
            if self.kind == "mean":
                uniform = jnp.full_like(weights, 1.0 / count)
                weights = eqx.error_if(
                    weights,
                    jnp.any(weights != uniform),
                    "Plain mean is only defined for exactly equal-mass cases.",
                )

        ess = 1.0 / jnp.sum(weights * weights)
        nan = jnp.asarray(jnp.nan, dtype=values.dtype)
        one = jnp.asarray(1.0, dtype=values.dtype)

        if self.kind in ("mean", "weighted_mean"):
            value = jnp.sum(weights * values)
            threshold = nan
            tail_mass = one
            tie_fraction = one
        elif self.kind == "max":
            value = jnp.max(values)
            threshold = value
            maximum = values == value
            tail_mass = jnp.sum(jnp.where(maximum, weights, 0.0))
            tie_fraction = one
        else:
            tail_target = jnp.asarray(1.0 - self.alpha, dtype=values.dtype)
            order = jnp.argsort(values)
            sorted_values = values[order]
            sorted_weights = weights[order]
            cumulative = jnp.cumsum(sorted_weights)
            threshold_index = jnp.argmax(cumulative >= self.alpha)
            threshold = sorted_values[threshold_index]
            above = values > threshold
            tied = values == threshold
            above_mass = jnp.sum(jnp.where(above, weights, 0.0))
            tied_mass = jnp.sum(jnp.where(tied, weights, 0.0))
            required_tied_mass = jnp.clip(
                tail_target - above_mass,
                min=0.0,
                max=tied_mass,
            )
            tie_fraction = required_tied_mass / tied_mass
            tail_mass = above_mass + required_tied_mass
            tail_sum = jnp.sum(jnp.where(above, weights * values, 0.0))
            tail_sum = tail_sum + required_tied_mass * threshold
            value = tail_sum / tail_target

        return MechanicsCaseReductionResult(
            value=value,
            case_values=values,
            normalized_weights=weights,
            effective_sample_size=ess,
            threshold=threshold,
            tail_mass=tail_mass,
            threshold_tie_fraction=tie_fraction,
            kind=self.kind,
            case_count=count,
            batch_dependent=self.kind == "max",
            reduction_id=self.reduction_id,
        )

    def __call__(
        self,
        case_values: ArrayLike,
        /,
        *,
        probability_weights: ArrayLike | None = None,
        valid: ArrayLike | None = None,
    ) -> Array:
        return self.evaluate(
            case_values,
            probability_weights=probability_weights,
            valid=valid,
        ).value


__all__ = [
    "MechanicsCaseReduction",
    "MechanicsCaseReductionKind",
    "MechanicsCaseReductionResult",
]
