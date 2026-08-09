#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_INSUFFICIENT_DATA, MLBatch, SparseFeatures
from phydrax.ml.outliers import (
    IsolationForestModel,
    IsolationForestRecipe,
    SmoothIsolationForestModel,
)


def _features():
    return jnp.array(
        [
            [-2.0, -0.3],
            [-1.2, 0.8],
            [-0.5, -0.7],
            [0.2, 0.1],
            [0.9, 0.7],
            [1.7, -0.4],
            [4.5, 4.0],
        ]
    )


def test_isolation_forest_requires_key_has_exact_tree_capacity_and_is_deterministic():
    features = _features()
    recipe = IsolationForestRecipe(n_estimators=5, max_depth=3, contamination=0.2)

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features))

    key = jax.random.key(5)
    first = recipe.fit_batch(MLBatch(features), key=key)
    repeated = recipe.fit_batch(MLBatch(features), key=key)
    changed = recipe.fit_batch(MLBatch(features), key=jax.random.key(6))
    model = first.as_trainable()
    repeated_model = repeated.as_trainable()
    changed_model = changed.as_trainable()
    node_capacity = 2 ** (recipe.max_depth + 1) - 1

    assert isinstance(model, IsolationForestModel)
    assert model.feature_indices.shape == (5, node_capacity)
    assert model.thresholds.shape == (5, node_capacity)
    assert model.splittable.shape == (5, node_capacity)
    assert model.leaf_mass.shape == (5, node_capacity)
    assert jnp.array_equal(model.feature_indices, repeated_model.feature_indices)
    assert jnp.allclose(model.thresholds, repeated_model.thresholds)
    assert jnp.array_equal(model.splittable, repeated_model.splittable)
    assert jnp.allclose(model.leaf_mass, repeated_model.leaf_mass)
    same_changed_topology = jnp.array_equal(
        model.feature_indices, changed_model.feature_indices
    ) & jnp.allclose(model.thresholds, changed_model.thresholds)
    assert not same_changed_topology
    assert first.diagnostics.iterations == recipe.max_depth
    assert first.diagnostics.method == "isolation-forest"
    assert first.gradient_contract.prediction_inputs == "none"
    assert first.gradient_contract.prediction_parameters == "none"
    assert first.gradient_contract.fit_features == "none"
    assert first.gradient_contract.fit_targets == "none"
    assert first.gradient_contract.fit_weights == "none"
    assert first.gradient_contract.fit_hyperparameters == "none"
    assert first.gradient_contract.fit_mode == "stopped"
    assert "exactly 2^(max_depth+1)-1" in " ".join(first.gradient_contract.conditions)


def test_hard_isolation_forest_is_exactly_stopped_and_relaxed_model_is_smooth():
    features = _features()
    hard = (
        IsolationForestRecipe(n_estimators=7, max_depth=3, contamination=0.2)
        .fit_batch(MLBatch(features), key=jax.random.key(10))
        .as_trainable()
    )
    smooth = hard.relaxed(temperature=0.5)
    points = jnp.array([[-0.2, 0.1], [0.8, -0.3], [4.0, 3.5]])
    hard_scores = hard(points)
    smooth_scores = smooth(points)

    assert isinstance(hard, IsolationForestModel)
    assert isinstance(smooth, SmoothIsolationForestModel)
    assert hard_scores.shape == (3,)
    assert smooth_scores.shape == (3,)
    assert jnp.all((hard_scores >= 0.0) & (hard_scores <= 1.0))
    assert jnp.all((smooth_scores >= 0.0) & (smooth_scores <= 1.0))
    assert jnp.array_equal(hard.predict(points), hard_scores > hard.threshold)
    assert not jnp.allclose(hard_scores, smooth_scores)

    hard_input_gradient = jax.grad(hard)(points[0])
    smooth_input_gradient = jax.grad(smooth)(points[0])
    assert jnp.all(hard_input_gradient == 0.0)
    assert jnp.all(jnp.isfinite(smooth_input_gradient))
    assert jnp.any(jnp.abs(smooth_input_gradient) > 1e-8)

    hard_parameter_gradient = eqx.filter_grad(lambda candidate: candidate(points[0]))(
        hard
    )
    hard_parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(hard_parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert hard_parameter_leaves
    assert all(jnp.all(leaf == 0.0) for leaf in hard_parameter_leaves)

    smooth_parameter_gradient = eqx.filter_grad(lambda candidate: candidate(points[0]))(
        smooth
    )
    smooth_parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(smooth_parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert smooth_parameter_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in smooth_parameter_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 1e-8) for leaf in smooth_parameter_leaves)

    membership = smooth.smooth_membership(points, temperature=0.3)
    assert membership.shape == (3,)
    assert jnp.all((membership >= 0.0) & (membership <= 1.0))
    assert jax.jit(hard)(points).shape == (3,)
    assert jax.vmap(hard)(points).shape == (3,)
    assert jax.jit(smooth)(points).shape == (3,)
    assert jax.vmap(smooth)(points).shape == (3,)


def test_isolation_forest_preserves_case_axes_masks_weights_and_frozen_execution():
    base = _features()
    features = jnp.stack((base, base * jnp.array([1.1, 0.9])), axis=0)
    targets = jnp.stack(
        (jnp.sum(features, axis=-1), jnp.prod(features, axis=-1)), axis=-1
    )
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 2, 1].set(False)
    sample_mask = jnp.array([True, True, True, True, True, True, False])
    weights = jnp.array([1.0, 1.5, 7.0, 0.8, 1.3, 1.1, 9.0])
    recipe = IsolationForestRecipe(n_estimators=3, max_depth=2, contamination=0.2)
    result = recipe.fit_batch(
        MLBatch(
            features,
            targets,
            feature_mask=feature_mask,
            sample_mask=sample_mask,
            sample_weight=weights,
            measure_weight=50.0,
        ),
        key=jax.random.key(21),
    )
    model = result.as_trainable()
    queries = features[:, :3] + 0.05

    assert model.case_shape == (2,)
    assert model.feature_indices.shape == (2, 3, 7)
    assert model(queries).shape == (2, 3)
    assert model(jnp.array([0.1, -0.2])).shape == (2,)
    assert result.model(queries).shape == (2, 3)
    assert jnp.array_equal(result.diagnostics.effective_samples, jnp.array([5, 5]))


def test_isolation_forest_rejects_complex_ordering_and_invalid_capacity():
    features = _features()
    complex_features = features.astype(jnp.complex64) + 0.1j
    recipe = IsolationForestRecipe(n_estimators=2, max_depth=2, contamination=0.2)

    with pytest.raises(TypeError, match="undefined for complex"):
        recipe.fit_batch(MLBatch(complex_features), key=jax.random.key(30))
    with pytest.raises(ValueError, match="max_depth"):
        IsolationForestRecipe(max_depth=13)

    model = recipe.fit_batch(MLBatch(features), key=jax.random.key(31)).as_trainable()
    with pytest.raises(TypeError, match="undefined for complex"):
        model(jnp.array([0.2 + 0.1j, -0.3 + 0.2j]))
    with pytest.raises(TypeError, match="requires real"):
        model.relaxed()(jnp.array([0.2 + 0.1j, -0.3 + 0.2j]))


def test_isolation_forest_rejects_sparse_features_explicitly():
    features = _features()
    sparse = SparseFeatures(
        features,
        jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), features.shape),
        feature_count=2,
    )

    with pytest.raises(TypeError, match="requires dense features"):
        IsolationForestRecipe(n_estimators=2, max_depth=2).fit_batch(
            MLBatch(sparse), key=jax.random.key(32)
        )


def test_isolation_forest_insufficient_data_is_an_invalid_status_value():
    mask = jnp.array([True, False, False, False, False, False, False])
    result = IsolationForestRecipe(
        n_estimators=2, max_depth=2, contamination=0.2
    ).fit_batch(MLBatch(_features(), sample_mask=mask), key=jax.random.key(40))

    assert not result.valid
    assert result.status == ML_INSUFFICIENT_DATA
    assert result.diagnostics.effective_samples == 1
