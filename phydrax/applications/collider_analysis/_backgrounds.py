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


class ABCDBackgroundPlan(StrictModule, NonTrainableState):
    closure_uncertainty: float = eqx.field(static=True)
    correlation_id: str = eqx.field(static=True)
    plan_id: str = eqx.field(static=True)

    def __init__(self, closure_uncertainty: float, /, *, correlation_id: str):
        uncertainty = float(closure_uncertainty)
        correlation = str(correlation_id).strip()
        if not math.isfinite(uncertainty) or uncertainty < 0.0 or not correlation:
            raise ValueError(
                "ABCD closure uncertainty and correlation identity are invalid."
            )
        self.closure_uncertainty = uncertainty
        self.correlation_id = correlation
        self.plan_id = canonical_fingerprint(
            {
                "kind": "abcd-background-plan",
                "closure_uncertainty": uncertainty,
                "correlation": correlation,
            }
        )


class ABCDBackgroundResult(StrictModule, NonTrainableState):
    prediction: Array
    variance: Array
    transfer_factor: Array
    finite: Array
    valid: Array
    plan_id: str = eqx.field(static=True)


def estimate_abcd_background(
    plan: ABCDBackgroundPlan,
    region_b: ArrayLike,
    region_c: ArrayLike,
    region_d: ArrayLike,
    /,
    *,
    variance_b: ArrayLike | None = None,
    variance_c: ArrayLike | None = None,
    variance_d: ArrayLike | None = None,
) -> ABCDBackgroundResult:
    """Estimate A = B*C/D with delta-method and closure covariance."""
    if not isinstance(plan, ABCDBackgroundPlan):
        raise TypeError("plan must be ABCDBackgroundPlan.")
    b = jnp.asarray(region_b)
    c = jnp.asarray(region_c, dtype=b.dtype)
    d = jnp.asarray(region_d, dtype=b.dtype)
    if b.shape != c.shape or b.shape != d.shape:
        raise ValueError("ABCD region yields must align.")
    vb = (
        jnp.maximum(b, 0.0)
        if variance_b is None
        else jnp.asarray(variance_b, dtype=b.dtype)
    )
    vc = (
        jnp.maximum(c, 0.0)
        if variance_c is None
        else jnp.asarray(variance_c, dtype=b.dtype)
    )
    vd = (
        jnp.maximum(d, 0.0)
        if variance_d is None
        else jnp.asarray(variance_d, dtype=b.dtype)
    )
    if vb.shape != b.shape or vc.shape != b.shape or vd.shape != b.shape:
        raise ValueError("ABCD variances must align with yields.")
    denominator_valid = d > 0.0
    safe_d = jnp.maximum(d, jnp.finfo(d.dtype).tiny)
    transfer = c / safe_d
    prediction = b * transfer
    variance = (
        transfer * transfer * vb
        + (b / safe_d) ** 2 * vc
        + (b * c / safe_d**2) ** 2 * vd
        + (plan.closure_uncertainty * prediction) ** 2
    )
    finite = jnp.isfinite(prediction) & jnp.isfinite(variance)
    valid = (
        denominator_valid
        & (b >= 0.0)
        & (c >= 0.0)
        & (vb >= 0.0)
        & (vc >= 0.0)
        & (vd >= 0.0)
        & finite
    )
    return ABCDBackgroundResult(
        jnp.where(valid, prediction, jnp.nan),
        jnp.where(valid, variance, jnp.nan),
        jnp.where(valid, transfer, jnp.nan),
        finite,
        valid,
        plan.plan_id,
    )


__all__ = ["ABCDBackgroundPlan", "ABCDBackgroundResult", "estimate_abcd_background"]
