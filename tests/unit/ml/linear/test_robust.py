#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_NONCONVERGED, MLBatch, SparseFeatures
from phydrax.ml.linear import (
    HuberModel,
    HuberRegressorRecipe,
    QuantileModel,
    QuantileRegressorRecipe,
    RANSACModel,
    RANSACRegressorRecipe,
    RobustDiagnostics,
    TheilSenModel,
    TheilSenRegressorRecipe,
)


def _outlier_data():
    features = jnp.array(
        [
            [-2.0, 0.0],
            [-1.5, 1.0],
            [-1.0, -1.0],
            [-0.5, 0.5],
            [0.0, -0.5],
            [0.5, 1.0],
            [1.0, -1.0],
            [1.5, 0.5],
            [2.0, 0.0],
            [2.5, -0.5],
        ]
    )
    clean = features @ jnp.array([[1.2, -0.4], [0.3, 0.8]]) + jnp.array([0.1, -0.2])
    targets = clean.at[3].add(jnp.array([8.0, -6.0])).at[8].add(jnp.array([-7.0, 5.0]))
    return features, targets


def _sparse(features):
    columns = jnp.broadcast_to(jnp.arange(features.shape[-1]), features.shape)
    return SparseFeatures(features, columns, feature_count=features.shape[-1])


def _assert_model_gradients(model, point):
    input_gradient = jax.grad(lambda value: jnp.sum(jnp.square(model(value))))(point)
    coefficient_gradient = jax.grad(
        lambda value: jnp.sum(
            jnp.square(eqx.tree_at(lambda item: item.coefficients, model, value)(point))
        )
    )(model.coefficients)
    intercept_gradient = jax.grad(
        lambda value: jnp.sum(
            jnp.square(eqx.tree_at(lambda item: item.intercept, model, value)(point))
        )
    )(model.intercept)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(coefficient_gradient))
    assert jnp.all(jnp.isfinite(intercept_gradient))


def test_one_step_relaxed_robust_updates_match_weighted_subgradients():
    batch = MLBatch(
        jnp.array([[2.0]]),
        jnp.array([3.0]),
        sample_weight=jnp.array([2.0]),
    )
    huber = (
        HuberRegressorRecipe(
            delta=1.0,
            fit_intercept=False,
            learning_rate=0.1,
            max_iterations=1,
            tolerance=1e6,
        )
        .fit_batch(batch)
        .as_trainable()
    )
    assert jnp.allclose(huber.coefficients, jnp.array([0.4]))

    quantile = (
        QuantileRegressorRecipe(
            0.25,
            solver="fixed-subgradient",
            fit_intercept=False,
            learning_rate=0.1,
            max_iterations=1,
            tolerance=1e6,
        )
        .fit_batch(batch)
        .as_trainable()
    )
    assert jnp.allclose(quantile.coefficients, jnp.array([0.1]))


def test_huber_relaxed_robust_loss_masks_weights_sparse_jit_vmap_and_gradients():
    features, targets = _outlier_data()
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    recipe = HuberRegressorRecipe(
        delta=1.0, l2_strength=0.1, max_iterations=4, tolerance=1e6
    )
    result = recipe.fit_batch(
        MLBatch(
            features,
            targets,
            target_mask=jnp.ones_like(targets, dtype=bool).at[1, 1].set(False),
            sample_weight=weights,
        )
    )
    model = result.as_trainable()
    assert isinstance(model, HuberModel)
    assert model(features).shape == targets.shape
    assert jax.jit(model)(features).shape == targets.shape
    assert jax.vmap(model)(features).shape == targets.shape
    assert result.gradient_contract.fit_mode == "unrolled"
    _assert_model_gradients(model, features[0])

    sparse_model = recipe.fit_batch(
        MLBatch(_sparse(features), targets, sample_weight=weights)
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == targets.shape

    base = HuberRegressorRecipe(
        delta=1.0, l2_strength=0.1, max_iterations=3, tolerance=1e6
    )

    def fit_loss(x, y, sample_weight, delta):
        fitted = (
            eqx.tree_at(lambda item: item.delta, base, delta)
            .fit_batch(MLBatch(x, y, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, targets, weights, base.delta
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_quantile_fixed_sparse_and_native_qp_have_explicit_gradient_policies():
    features, targets = _outlier_data()
    scalar_targets = targets[:, 0]
    weights = jnp.linspace(0.8, 1.2, features.shape[0])
    fixed = QuantileRegressorRecipe(
        0.4,
        solver="fixed-subgradient",
        l2_strength=0.1,
        max_iterations=3,
        tolerance=1e6,
    )
    fixed_result = fixed.fit_batch(
        MLBatch(_sparse(features), scalar_targets, sample_weight=weights)
    )
    fixed_model = fixed_result.as_trainable()
    assert isinstance(fixed_model, QuantileModel)
    assert fixed_model(_sparse(features)).shape == scalar_targets.shape
    assert fixed_result.gradient_contract.fit_mode == "unrolled"
    _assert_model_gradients(fixed_model, features[0])

    def fixed_loss(x, y, sample_weight, quantile):
        recipe = eqx.tree_at(lambda item: item.quantile, fixed, quantile)
        fitted = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(features[:2])))

    fixed_gradients = jax.grad(fixed_loss, argnums=(0, 1, 2, 3))(
        features, scalar_targets, weights, fixed.quantile
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in fixed_gradients)

    qp = QuantileRegressorRecipe(
        0.5,
        solver="dense-qp",
        l2_strength=0.1,
        max_iterations=40,
        tolerance=1e-4,
    )
    qp_result = qp.fit_batch(
        MLBatch(
            features,
            scalar_targets,
            target_mask=jnp.ones_like(scalar_targets, dtype=bool).at[0].set(False),
            sample_weight=weights,
        )
    )
    qp_model = qp_result.as_trainable()
    assert isinstance(qp_model, QuantileModel)
    assert qp_model(features).shape == scalar_targets.shape
    assert qp_result.gradient_contract.fit_mode == "implicit"
    assert jax.jit(qp_model)(features).shape == scalar_targets.shape
    _assert_model_gradients(qp_model, features[0])

    def qp_loss(x, y, sample_weight, quantile):
        recipe = eqx.tree_at(lambda item: item.quantile, qp, quantile)
        fitted = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(features[:2])))

    qp_gradients = jax.grad(qp_loss, argnums=(0, 1, 2, 3))(
        features, scalar_targets, weights, qp.quantile
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in qp_gradients)
    with pytest.raises(TypeError, match="requires dense features"):
        qp.fit_batch(MLBatch(_sparse(features), scalar_targets))
    with pytest.raises(ValueError, match="exceeds max_dense_dimension=72"):
        QuantileRegressorRecipe(solver="dense-qp", max_dense_dimension=72).fit_batch(
            MLBatch(features, scalar_targets)
        )
    with pytest.raises(ValueError, match="max_dense_dimension must be positive"):
        QuantileRegressorRecipe(solver="dense-qp", max_dense_dimension=0)


def test_ransac_requires_key_is_deterministic_and_stops_subset_fit_gradients():
    features, targets = _outlier_data()
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    recipe = RANSACRegressorRecipe(residual_threshold=0.5, min_samples=4, num_trials=8)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features, targets))
    first = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights), key=jax.random.key(7)
    )
    second = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights), key=jax.random.key(7)
    )
    model = first.as_trainable()
    assert isinstance(model, RANSACModel)
    assert isinstance(first.diagnostics, RobustDiagnostics)
    assert jnp.allclose(model.coefficients, second.as_trainable().coefficients)
    assert jnp.array_equal(first.diagnostics.inlier_mask, second.diagnostics.inlier_mask)
    assert first.diagnostics.selected_subset.shape == (features.shape[0],)
    assert model(features).shape == targets.shape
    assert jax.jit(model)(features).shape == targets.shape
    assert jax.vmap(model)(features).shape == targets.shape
    _assert_model_gradients(model, features[0])

    query = features[:2]

    def stopped_fit_loss(x, y, sample_weight):
        fitted = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight), key=jax.random.key(7)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(query)))

    gradients = jax.grad(stopped_fit_loss, argnums=(0, 1, 2))(features, targets, weights)
    assert all(jnp.all(value == 0.0) for value in gradients)
    sparse_result = recipe.fit_batch(
        MLBatch(_sparse(features), targets), key=jax.random.key(8)
    )
    assert sparse_result.as_trainable()(_sparse(features)).shape == targets.shape
    with pytest.raises(ValueError, match="min_samples"):
        RANSACRegressorRecipe(min_samples=features.shape[0] + 1).fit_batch(
            MLBatch(features, targets), key=jax.random.key(0)
        )


def test_theil_sen_requires_key_is_deterministic_sparse_and_capacity_bounded():
    features, targets = _outlier_data()
    recipe = TheilSenRegressorRecipe(subset_size=4, num_subsets=8)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features, targets))
    first = recipe.fit_batch(MLBatch(features, targets), key=jax.random.key(4))
    second = recipe.fit_batch(MLBatch(features, targets), key=jax.random.key(4))
    model = first.as_trainable()
    assert isinstance(model, TheilSenModel)
    assert isinstance(first.diagnostics, RobustDiagnostics)
    assert jnp.allclose(model.coefficients, second.as_trainable().coefficients)
    assert jnp.array_equal(
        first.diagnostics.selected_subset, second.diagnostics.selected_subset
    )
    assert first.diagnostics.selected_subset.shape == (
        recipe.num_subsets,
        features.shape[0],
    )
    assert model(features).shape == targets.shape
    _assert_model_gradients(model, features[0])
    sparse_model = recipe.fit_batch(
        MLBatch(_sparse(features), targets), key=jax.random.key(5)
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == targets.shape

    query = features[:2]

    def stopped_fit_loss(x, y, sample_weight):
        fitted = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight), key=jax.random.key(4)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(query)))

    gradients = jax.grad(stopped_fit_loss, argnums=(0, 1, 2))(
        features, targets, jnp.ones((features.shape[0],))
    )
    assert all(jnp.all(value == 0.0) for value in gradients)
    with pytest.raises(ValueError, match="subset_size"):
        TheilSenRegressorRecipe(
            subset_size=features.shape[0] + 1, num_subsets=2
        ).fit_batch(MLBatch(features, targets), key=jax.random.key(0))


def test_relaxed_robust_losses_ignore_zero_weight_target_changes():
    features, targets = _outlier_data()
    weights = jnp.ones((features.shape[0],)).at[1].set(0.0)
    changed = targets.at[1].set(jnp.array([1e4, -1e4]))
    for recipe in (
        HuberRegressorRecipe(max_iterations=3, tolerance=1e6),
        QuantileRegressorRecipe(
            solver="fixed-subgradient", max_iterations=3, tolerance=1e6
        ),
    ):
        first = recipe.fit_batch(
            MLBatch(features, targets, sample_weight=weights)
        ).as_trainable()
        second = recipe.fit_batch(
            MLBatch(features, changed, sample_weight=weights)
        ).as_trainable()
        assert jnp.allclose(first.coefficients, second.coefficients)
        assert jnp.allclose(first.intercept, second.intercept)
    no_intercept = (
        HuberRegressorRecipe(fit_intercept=False, max_iterations=2, tolerance=1e6)
        .fit_batch(MLBatch(features, targets))
        .as_trainable()
    )
    assert jnp.all(no_intercept.intercept == 0.0)


def test_relaxed_robust_nonconvergence_is_not_reported_as_valid():
    features, targets = _outlier_data()
    result = HuberRegressorRecipe(max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(features, targets)
    )
    assert result.status == ML_NONCONVERGED
    with pytest.raises(TypeError, match="real-valued features"):
        HuberRegressorRecipe(max_iterations=2).fit_batch(
            MLBatch(features.astype(jnp.complex64), targets)
        )
    assert not result.valid
