#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..linalg import SmallLinearSolvePlan, solve_small_linear


@dataclass(frozen=True, slots=True)
class TwoNodeQuadrature:
    nodes: Array
    weights: Array
    realizable: Array


def qmom_two_node(moments: ArrayLike, /) -> TwoNodeQuadrature:
    m = jnp.asarray(moments)
    if m.shape != (4,):
        raise ValueError("Two-node QMOM requires moments m0 through m3.")
    finite_moments = jnp.all(jnp.isfinite(m))
    nonnegative_moments = jnp.all(m >= 0)
    positive_mass = m[0] > 0
    hankel_valid = m[0] * m[2] >= m[1] ** 2
    matrix = jnp.asarray(((m[1], -m[0]), (m[2], -m[1])))
    result = solve_small_linear(
        SmallLinearSolvePlan(2), matrix, jnp.asarray((m[2], m[3]))
    )
    s1, s0 = result.value
    discriminant = s1 * s1 - 4.0 * s0
    root = jnp.sqrt(jnp.maximum(discriminant, 0.0))
    nodes = jnp.asarray(((s1 - root) / 2.0, (s1 + root) / 2.0))
    separation = nodes[1] - nodes[0]
    safe_separation = jnp.where(separation > 0, separation, 1.0)
    weight0 = (m[0] * nodes[1] - m[1]) / safe_separation
    weights = jnp.asarray((weight0, m[0] - weight0))
    valid = (
        result.successful
        & finite_moments
        & nonnegative_moments
        & positive_mass
        & hankel_valid
        & (discriminant > 0)
        & jnp.all(jnp.isfinite(nodes))
        & jnp.all(nodes >= 0)
        & jnp.all(jnp.isfinite(weights))
        & jnp.all(weights >= 0)
    )
    return TwoNodeQuadrature(nodes, weights, valid)


__all__ = ["TwoNodeQuadrature", "qmom_two_node"]
