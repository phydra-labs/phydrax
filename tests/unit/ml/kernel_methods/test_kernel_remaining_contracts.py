#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp

from phydrax.kernels import SquaredExponentialKernel
from phydrax.ml import MLBatch
from phydrax.ml.kernel_methods import (
    CategoricalGaussianProcessClassifierRecipe,
    KernelRidgeRecipe,
)
from phydrax.uq import GaussianProcessLikelihoodState


def _assert_finite(values):
    assert all(jnp.all(jnp.isfinite(value)) for value in values)


def _assert_prediction_parameter_gradient(model, query):
    gradient = eqx.filter_grad(
        lambda current: jnp.sum(jnp.square(jnp.real(current(query))))
    )(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_kernel_ridge_direct_contract_covers_features_targets_weights_and_kernel():
    features = jnp.array(
        [[-1.2, -0.4], [-0.5, 0.8], [0.2, -0.6], [0.9, 0.5], [1.6, -0.1]]
    )
    targets = jnp.stack(
        (0.7 * features[:, 0] - 0.2 * features[:, 1], features[:, 0] + features[:, 1]),
        axis=-1,
    )
    weights = jnp.array([0.8, 1.1, 0.9, 1.2, 0.7])
    query = jnp.array([[-0.25, 0.1], [1.1, 0.15]])
    base = KernelRidgeRecipe(SquaredExponentialKernel(length_scale=0.9), alpha=0.2)

    def fit_loss(x, y, sample_weight, alpha, length_scale):
        recipe = eqx.tree_at(
            lambda current: (current.alpha, current.kernel.length_scale),
            base,
            (alpha, length_scale),
        )
        prediction = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()(query)
        return jnp.sum(jnp.square(prediction))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3, 4))(
        features,
        targets,
        weights,
        base.alpha,
        base.kernel.length_scale,
    )
    _assert_finite(gradients)
    result = base.fit_batch(MLBatch(features, targets, sample_weight=weights))
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_targets,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == ("conditional", "conditional", "conditional", "conditional")
    assert result.as_trainable()(query).shape == (2, 2)
    _assert_prediction_parameter_gradient(result.as_trainable(), query)


def test_categorical_gp_branch_covers_declared_fit_and_prediction_gradients():
    features = jnp.array(
        [[-1.2, -0.4], [-0.5, 0.8], [0.2, -0.6], [0.9, 0.5], [1.6, -0.1]]
    )
    labels = jnp.array([0, 1, 2, 0, 2], dtype=jnp.int32)
    weights = jnp.array([0.8, 1.1, 0.9, 1.2, 0.7])
    query = jnp.array([[-0.25, 0.1], [1.1, 0.15]])
    state = GaussianProcessLikelihoodState(
        kernel=SquaredExponentialKernel(length_scale=0.9),
        noise_scale=0.05,
        jitter=1e-4,
    )
    base = CategoricalGaussianProcessClassifierRecipe(
        state, class_count=3, iterations=2, curvature_floor=1e-5
    )

    def fit_loss(x, sample_weight, length_scale, noise_scale, jitter, floor):
        recipe = eqx.tree_at(
            lambda current: (
                current.recipe.state.kernel.length_scale,
                current.recipe.state.noise_scale,
                current.recipe.state.jitter,
                current.recipe.curvature_floor,
            ),
            base,
            (length_scale, noise_scale, jitter, floor),
        )
        probability = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()(query)
        return jnp.sum(jnp.square(probability))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3, 4, 5))(
        features,
        weights,
        base.recipe.state.kernel.length_scale,
        base.recipe.state.noise_scale,
        base.recipe.state.jitter,
        base.recipe.curvature_floor,
    )
    _assert_finite(gradients)
    result = base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_targets,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == ("conditional", "none", "conditional", "conditional")
    assert result.as_trainable()(query).shape == (2, 3)
    _assert_prediction_parameter_gradient(result.as_trainable(), query)
