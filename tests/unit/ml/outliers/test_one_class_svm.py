#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.kernels import Matern32Kernel, SquaredExponentialKernel
from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    MLBatch,
    SparseFeatures,
)
from phydrax.ml.outliers import OneClassSVMModel, OneClassSVMRecipe


def _features():
    return jnp.array(
        [
            [-2.0, -0.3],
            [-1.2, 0.8],
            [-0.5, -0.7],
            [0.2, 0.1],
            [0.9, 0.7],
            [1.7, -0.4],
            [2.5, 0.5],
            [4.5, 4.0],
        ]
    )


def test_one_class_svm_uses_native_kernel_score_and_dual_invariants():
    features = _features()
    weights = jnp.array([1.0, 1.3, 0.8, 1.5, 1.1, 0.9, 1.4, 1.0])
    recipe = OneClassSVMRecipe(
        Matern32Kernel(length_scale=jnp.array([1.2, 0.8])),
        nu=0.25,
        iterations=5,
        learning_rate=0.1,
        tolerance=1e6,
    )
    result = recipe.fit_batch(MLBatch(features, sample_weight=weights))
    model = result.as_trainable()
    points = jnp.array([[-0.2, 0.1], [5.0, 4.5]])
    scores = model(points)
    predictions = model.predict(points)
    membership = model.smooth_membership(points, temperature=0.4)

    assert isinstance(model, OneClassSVMModel)
    assert model.training_features.shape == (8, 2)
    assert model.dual_coefficients.shape == (8,)
    assert jnp.all(model.dual_coefficients >= 0.0)
    assert jnp.isclose(jnp.sum(model.dual_coefficients), 1.0, atol=2e-5)
    assert model.kernel.kernel_id == "Matern32Kernel"
    assert scores.shape == (2,)
    assert predictions.dtype == jnp.bool_
    assert jnp.array_equal(predictions, scores > 0.0)
    assert jnp.array_equal(membership > 0.5, predictions)
    assert jnp.allclose(result.model(points), scores)
    assert result.diagnostics.threshold == 0.0
    assert result.diagnostics.score_maximum >= result.diagnostics.score_minimum
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.prediction_parameters == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_targets == "none"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"
    assert result.gradient_contract.fit_mode == "unrolled"


def test_one_class_svm_case_axes_masks_and_inactive_dual_capacity():
    base = _features()
    features = jnp.stack((base, base * jnp.array([1.1, 0.9])), axis=0)
    targets = jnp.stack(
        (jnp.sum(features, axis=-1), jnp.prod(features, axis=-1)), axis=-1
    )
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 2, 1].set(False)
    sample_mask = jnp.array([True, True, True, True, True, True, True, False])
    recipe = OneClassSVMRecipe(nu=0.3, iterations=4, learning_rate=0.1, tolerance=1e6)
    result = recipe.fit_batch(
        MLBatch(
            features,
            targets,
            feature_mask=feature_mask,
            target_mask=jnp.ones_like(targets, dtype=bool).at[:, 0, 0].set(False),
            sample_mask=sample_mask,
            sample_weight=jnp.array([1.0, 1.2, 8.0, 1.1, 0.9, 1.4, 1.0, 9.0]),
            measure_weight=20.0,
        )
    )
    model = result.as_trainable()

    assert model.case_shape == (2,)
    assert model.dual_coefficients.shape == (2, 8)
    assert model(features[:, :3]).shape == (2, 3)
    assert model(jnp.array([0.1, -0.2])).shape == (2,)
    assert jnp.array_equal(result.diagnostics.effective_samples, jnp.array([6, 6]))
    assert jnp.all(model.dual_coefficients[:, 2] == 0.0)
    assert jnp.all(model.dual_coefficients[:, 7] == 0.0)
    assert jnp.allclose(jnp.sum(model.dual_coefficients, axis=-1), 1.0, atol=2e-5)


def test_one_class_svm_jit_vmap_prediction_parameter_and_fit_gradients():
    features = _features()
    weights = jnp.array([1.0, 1.3, 0.8, 1.5, 1.1, 0.9, 1.4, 1.05])
    point = jnp.array([0.35, -0.25])
    recipe = OneClassSVMRecipe(
        SquaredExponentialKernel(length_scale=1.1),
        nu=0.25,
        iterations=4,
        learning_rate=0.1,
        tolerance=1e6,
    )
    model = recipe.fit_batch(MLBatch(features, sample_weight=weights)).as_trainable()
    points = jnp.array([point, point + jnp.array([0.4, 0.2])])

    assert jax.jit(model)(points).shape == (2,)
    assert jax.vmap(model)(points).shape == (2,)
    input_gradient = jax.grad(model)(point)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.any(jnp.abs(input_gradient) > 1e-8)

    parameter_gradient = eqx.filter_grad(lambda candidate: candidate(point))(model)
    parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]
    assert parameter_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in parameter_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 1e-8) for leaf in parameter_leaves)

    def feature_loss(value):
        fitted = recipe.fit_batch(MLBatch(value, sample_weight=weights)).as_trainable()
        return fitted(point)

    def weight_loss(value):
        fitted = recipe.fit_batch(MLBatch(features, sample_weight=value)).as_trainable()
        return fitted(point)

    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(weights)
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_one_class_svm_native_kernel_hyperparameter_gradient_is_finite():
    features = _features()
    point = jnp.array([0.3, -0.2])

    def loss(length_scale):
        recipe = OneClassSVMRecipe(
            SquaredExponentialKernel(length_scale=length_scale),
            nu=0.25,
            iterations=4,
            learning_rate=0.1,
            tolerance=1e6,
        )
        return recipe.fit_batch(MLBatch(features)).as_trainable()(point)

    derivative = jax.grad(loss)(jnp.asarray(1.1))
    assert jnp.isfinite(derivative)
    assert jnp.abs(derivative) > 1e-8


def test_one_class_svm_rejects_precomputed_and_complex_kernel_geometry():
    features = _features()

    with pytest.raises(TypeError, match="native AbstractPositiveDefiniteKernel"):
        OneClassSVMRecipe(jnp.eye(features.shape[0]))

    recipe = OneClassSVMRecipe(nu=0.25, iterations=3, learning_rate=0.1, tolerance=1e6)
    with pytest.raises(TypeError, match="require real features"):
        recipe.fit_batch(MLBatch(features.astype(jnp.complex64) + 0.1j))

    model = recipe.fit_batch(MLBatch(features)).as_trainable()
    with pytest.raises(TypeError, match="require real features"):
        model(jnp.array([0.2 + 0.1j, -0.3 + 0.2j]))


def test_one_class_svm_rejects_sparse_features_explicitly():
    features = _features()
    sparse = SparseFeatures(
        features,
        jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), features.shape),
        feature_count=2,
    )

    with pytest.raises(TypeError, match="requires dense features"):
        OneClassSVMRecipe().fit_batch(MLBatch(sparse))


def test_one_class_svm_invalid_statuses_distinguish_insufficient_and_nonconverged():
    features = _features()
    insufficient_mask = jnp.array([True, False, False, False, False, False, False, False])
    insufficient = OneClassSVMRecipe(nu=0.25, iterations=2, tolerance=1e6).fit_batch(
        MLBatch(features, sample_mask=insufficient_mask)
    )
    nonconverged = OneClassSVMRecipe(
        nu=0.25, iterations=1, learning_rate=0.1, tolerance=1e-30
    ).fit_batch(MLBatch(features))

    assert not insufficient.valid
    assert insufficient.status == ML_INSUFFICIENT_DATA
    assert not nonconverged.valid
    assert nonconverged.status == ML_NONCONVERGED
