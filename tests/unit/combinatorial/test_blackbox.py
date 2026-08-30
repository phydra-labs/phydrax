#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _catalog():
    return phx.combinatorial.ExplicitDecisionSpace(
        jnp.asarray([0, 1, 2], dtype=jnp.int32),
        jnp.asarray([[0.0], [1.0], [2.0]]),
    )


def test_blackbox_pullback_matches_explicit_loss_interpolation_formula():
    space = _catalog()
    method = phx.combinatorial.ExhaustiveLinearOracle()
    policy = phx.combinatorial.BlackboxInterpolation(1.0)
    problem = phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray([1.0]))
    cotangent = jnp.asarray([-2.0])
    pullback = phx.combinatorial.estimate_blackbox_pullback(
        problem,
        method,
        cotangent,
        policy=policy,
    )

    expected = (pullback.perturbed.features - pullback.forward.features) / policy.lambda_
    np.testing.assert_allclose(pullback.gradient, expected)
    np.testing.assert_allclose(pullback.gradient, jnp.asarray([2.0]))
    assert pullback.valid
    assert pullback.exact_theory_applicable
    assert not pullback.zero_gradient
    np.testing.assert_allclose(pullback.feature_change_norm, 2.0)


def test_blackbox_custom_vjp_matches_explicit_pullback_under_jit():
    space = _catalog()
    method = phx.combinatorial.ExhaustiveLinearOracle()
    policy = phx.combinatorial.BlackboxInterpolation(1.0)

    def loss(cost):
        problem = phx.combinatorial.LinearCombinatorialProblem(space, cost)
        features = phx.combinatorial.blackbox_solution(
            problem,
            method,
            policy=policy,
        )
        return jnp.sum(-2.0 * features)

    gradient = jax.jit(jax.grad(loss))(jnp.asarray([1.0]))
    explicit = phx.combinatorial.estimate_blackbox_pullback(
        phx.combinatorial.LinearCombinatorialProblem(space, jnp.asarray([1.0])),
        method,
        jnp.asarray([-2.0]),
        policy=policy,
    )

    np.testing.assert_allclose(gradient, explicit.gradient)
    np.testing.assert_allclose(gradient, jnp.asarray([2.0]))

    with pytest.raises(TypeError, match="forward-mode autodiff"):
        jax.jvp(loss, (jnp.asarray([1.0]),), (jnp.asarray([1.0]),))


def test_blackbox_zero_gradient_and_batched_cardinality_pullback():
    catalog = _catalog()
    method = phx.combinatorial.ExhaustiveLinearOracle()
    unchanged = phx.combinatorial.estimate_blackbox_pullback(
        phx.combinatorial.LinearCombinatorialProblem(catalog, jnp.asarray([2.0])),
        method,
        jnp.asarray([0.01]),
        policy=phx.combinatorial.BlackboxInterpolation(0.1),
    )
    np.testing.assert_array_equal(unchanged.gradient, jnp.zeros((1,)))
    assert unchanged.zero_gradient

    space = phx.combinatorial.CardinalitySpace(3, 1)
    cardinality = phx.combinatorial.StableCardinalityOracle()
    problem = phx.combinatorial.LinearCombinatorialProblem(
        space,
        jnp.asarray([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]),
    )
    cotangent = jnp.asarray([[3.0, -3.0, 0.0], [0.0, -3.0, 3.0]])
    pullback = phx.combinatorial.estimate_blackbox_pullback(
        problem,
        cardinality,
        cotangent,
        policy=phx.combinatorial.BlackboxInterpolation(1.0),
    )
    assert pullback.gradient.shape == (2, 3)
    assert bool(jnp.all(pullback.valid))
    np.testing.assert_allclose(jnp.sum(pullback.gradient, axis=-1), jnp.zeros((2,)))


def test_blackbox_policy_rejects_nonpositive_or_nonscalar_lambda():
    with pytest.raises(ValueError, match="finite and positive"):
        phx.combinatorial.BlackboxInterpolation(0.0)
    with pytest.raises(ValueError, match="scalar"):
        phx.combinatorial.BlackboxInterpolation(jnp.ones((2,)))
