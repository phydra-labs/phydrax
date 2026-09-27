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
    ordered_ranks,
    ordered_values,
    PAVOrdering,
    SinkhornOrdering,
    straight_through_sort,
)


def test_ordering_relaxations_scenario_1() -> None:
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5])
    weights = jnp.asarray([0.5, 2.0, 1.0, 3.0])
    ordered = fast_weighted_soft_sort(values, weights, temperature=0.4)
    assert jnp.isclose(
        # ty: ignore[invalid-argument-type]
        jnp.sum(weights[jnp.argsort(values)] * ordered),
        jnp.sum(weights * values),
    )
    ranks, tangent = jax.jvp(
        lambda value, mass: fast_weighted_soft_rank(value, mass, temperature=0.4),
        (values, weights),
        (jnp.ones_like(values), 0.1 * jnp.ones_like(weights)),
    )
    assert jnp.all(jnp.isfinite(ranks))
    assert jnp.all(jnp.isfinite(tangent))
    values = jnp.asarray([2.0, 1.0, 1.0])
    result = straight_through_sort(values, PAVOrdering(0.5))
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(result, ordered_values(values, HardOrdering()))
    gradient = jax.grad(
        # ty: ignore[invalid-argument-type]
        lambda value: jnp.sum(straight_through_sort(value, PAVOrdering(0.5)) ** 2)
    )(values)
    assert jnp.all(jnp.isfinite(gradient))
    values = jnp.asarray([3.0, -1.0, 2.0, 0.5])
    method = SinkhornOrdering(0.1)
    ascending = ordered_values(values, method)
    descending = ordered_values(values, method, descending=True)
    # ty: ignore[invalid-argument-type]
    assert jnp.allclose(descending, jnp.flip(ascending))
    # ty: ignore[invalid-argument-type]
    assert jnp.all(jnp.diff(ascending) > 0.0)
    ranks = ordered_ranks(values, method)
    descending_ranks = ordered_ranks(values, method, descending=True)
    assert jnp.allclose(descending_ranks, values.shape[0] - 1 - ranks)
    hard_ranks = ordered_ranks(values, HardOrdering())
    assert jnp.array_equal(jnp.argsort(ranks), jnp.argsort(hard_ranks))


def test_relaxed_discrete_samples_are_replayable_and_hard_top_k_is_exact() -> None:
    logits = jnp.asarray([-0.5, 0.2, 1.5, 0.7])
    first = relaxed_bernoulli(logits, key=jr.key(1), hard=True)
    replay = relaxed_bernoulli(logits, key=jr.key(1), hard=True)
    # ty: ignore[invalid-argument-type]
    assert jnp.array_equal(first.hard, replay.hard)
    top = relaxed_top_k(logits, 2, key=jr.key(2), hard=True)
    # ty: ignore[invalid-argument-type]
    assert jnp.sum(top.hard) == 2
    assert top.estimator == "gumbel-top-k-straight-through"
