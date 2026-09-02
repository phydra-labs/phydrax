#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


SensitivityDecisionPolicy = Literal[
    "smooth_discrete", "frozen_branch", "smooth_surrogate", "unsupported"
]


class ConservationSensitivityEvidence(StrictModule, NonTrainableState):
    jvp_defect: Array
    vjp_duality_defect: Array
    first_order_taylor_defect: Array
    second_order_taylor_defect: Array
    valid: Array
    decision_policy: SensitivityDecisionPolicy = eqx.field(static=True)
    decision_id: str = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)


def certify_conservation_sensitivity(
    dynamics,
    time: ArrayLike,
    state: ArrayLike,
    direction: ArrayLike,
    cotangent: ArrayLike,
    /,
    *,
    args=None,
    epsilon: float = 1.0e-5,
    decision_policy: SensitivityDecisionPolicy = "smooth_discrete",
    decision_id: str = "smooth",
    tolerance: float = 2.0e-4,
) -> ConservationSensitivityEvidence:
    if decision_policy not in (
        "smooth_discrete",
        "frozen_branch",
        "smooth_surrogate",
        "unsupported",
    ):
        raise ValueError("Unknown conservation sensitivity policy.")
    epsilon_ = float(epsilon)
    tolerance_ = float(tolerance)
    if not math.isfinite(epsilon_) or epsilon_ <= 0.0 or tolerance_ <= 0.0:
        raise ValueError("Sensitivity epsilon and tolerance must be positive.")
    value = jnp.asarray(state)
    tangent = jnp.asarray(direction)
    dual = jnp.asarray(cotangent)
    if tangent.shape != value.shape or dual.shape != value.shape:
        raise ValueError("Sensitivity direction/cotangent shapes must match state.")
    function = lambda candidate: dynamics(jnp.asarray(time), candidate, args)
    primal, pushforward = jax.linearize(function, value)
    jvp = pushforward(tangent)
    _, pullback = jax.vjp(function, value)
    vjp = pullback(dual)[0]
    plus = function(value + epsilon_ * tangent)
    minus = function(value - epsilon_ * tangent)
    finite_jvp = (plus - minus) / (2.0 * epsilon_)
    jvp_defect = jnp.max(jnp.abs(jvp - finite_jvp))
    duality = jnp.abs(jnp.vdot(dual, jvp) - jnp.vdot(vjp, tangent))
    first_taylor = jnp.max(jnp.abs(plus - primal))
    second_taylor = jnp.max(jnp.abs(plus - primal - epsilon_ * jvp))
    scale = jnp.maximum(1.0, jnp.max(jnp.abs(primal)))
    valid = (
        (decision_policy != "unsupported")
        & jnp.all(jnp.isfinite(primal))
        & (jvp_defect <= tolerance_ * scale)
        & (duality <= tolerance_ * scale)
    )
    evidence_id = canonical_fingerprint(
        {
            "kind": "conservation-sensitivity-evidence",
            "dynamics": dynamics.dynamics_id,
            "decision_policy": decision_policy,
            "decision_id": str(decision_id),
            "epsilon": epsilon_,
            "tolerance": tolerance_,
        }
    )
    return ConservationSensitivityEvidence(
        jvp_defect,
        duality,
        first_taylor,
        second_taylor,
        valid,
        decision_policy,
        str(decision_id),
        evidence_id,
    )


__all__ = [
    "ConservationSensitivityEvidence",
    "SensitivityDecisionPolicy",
    "certify_conservation_sensitivity",
]
