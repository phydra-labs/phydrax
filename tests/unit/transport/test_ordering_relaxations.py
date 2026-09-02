#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

from phydrax.ml._soft_discrete import relaxed_bernoulli, relaxed_top_k
from phydrax.transport._fast_order import (
    fast_weighted_soft_rank,
    fast_weighted_soft_sort,
)
from phydrax.transport._ordering import (
    HardOrdering,
    ordered_values,
    PAVOrdering,
    straight_through_sort,
)


def test_weighted_pav_preserves_weighted_mean_and_has_weight_jvp():
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5])
    weights = jnp.asarray([0.5, 2.0, 1.0, 3.0])
    ordered = fast_weighted_soft_sort(values, weights, temperature=0.4)
    assert jnp.isclose(
        jnp.sum(weights[jnp.argsort(values)] * ordered), jnp.sum(weights * values)
    )
    ranks, tangent = jax.jvp(
        lambda value, mass: fast_weighted_soft_rank(value, mass, temperature=0.4),
        (values, weights),
        (jnp.ones_like(values), 0.1 * jnp.ones_like(weights)),
    )
    assert jnp.all(jnp.isfinite(ranks))
    assert jnp.all(jnp.isfinite(tangent))


def test_straight_through_sort_separates_hard_forward_and_soft_gradient():
    values = jnp.asarray([2.0, 1.0, 1.0])
    result = straight_through_sort(values, PAVOrdering(0.5))
    assert jnp.array_equal(result, ordered_values(values, HardOrdering()))
    gradient = jax.grad(
        lambda value: jnp.sum(straight_through_sort(value, PAVOrdering(0.5)) ** 2)
    )(values)
    assert jnp.all(jnp.isfinite(gradient))


def test_relaxed_discrete_samples_are_replayable_and_hard_top_k_is_exact():
    logits = jnp.asarray([-0.5, 0.2, 1.5, 0.7])
    first = relaxed_bernoulli(logits, key=jr.key(1), hard=True)
    replay = relaxed_bernoulli(logits, key=jr.key(1), hard=True)
    assert jnp.array_equal(first.hard, replay.hard)
    top = relaxed_top_k(logits, 2, key=jr.key(2), hard=True)
    assert jnp.sum(top.hard) == 2
    assert top.estimator == "gumbel-top-k-straight-through"
