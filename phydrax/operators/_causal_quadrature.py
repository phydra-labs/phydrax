#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math

import jax.numpy as jnp
from jaxtyping import Array

from ..integration._rules import (
    GaussLegendreRule,
    interval_rule_data,
    IntervalRule,
)


def causal_reference_rule(
    rule: IntervalRule,
    /,
    *,
    cluster_exponent: float = 1.0,
) -> tuple[Array, Array]:
    """Map an interval rule to ``[0, 1]`` with optional start clustering."""
    exponent = float(cluster_exponent)
    if not math.isfinite(exponent) or exponent < 1.0:
        raise ValueError("cluster_exponent must be finite and at least one.")
    if exponent != 1.0 and not isinstance(rule, GaussLegendreRule):
        raise ValueError(
            "Nontrivial causal clustering currently requires GaussLegendreRule."
        )
    data = interval_rule_data(rule)
    unit_nodes = (jnp.asarray(data.nodes, dtype=float) + 1.0) / 2.0
    unit_weights = jnp.asarray(data.weights, dtype=float) / 2.0
    if exponent == 1.0:
        return unit_nodes, unit_weights
    mapped_nodes = jnp.power(unit_nodes, exponent)
    jacobian = exponent * jnp.power(unit_nodes, exponent - 1.0)
    return mapped_nodes, unit_weights * jacobian


__all__ = ["causal_reference_rule"]
