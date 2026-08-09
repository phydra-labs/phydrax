#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml._contracts import ML_CAPACITY_EXHAUSTED
from phydrax.ml.neighbors import (
    KernelNeighborsClassifierRecipe,
    KernelNeighborsRegressorRecipe,
    KNeighborsClassifierRecipe,
    KNeighborsRegressorRecipe,
    RadiusNeighborsClassifierRecipe,
    RadiusNeighborsRegressorRecipe,
)


def _data():
    features = jnp.array(
        [[-1.5, 0.0], [-0.8, 0.5], [-0.1, -0.2], [0.6, 0.4], [1.4, -0.3], [2.0, 0.8]]
    )
    targets = 0.5 * features[:, 0] - features[:, 1]
    labels = (targets > 0).astype(jnp.int32)
    return features, targets, labels


def test_exact_regression_and_classification_dense_chunked_jit_vmap_parity():
    features, targets, labels = _data()
    weights = jnp.array([1.0, 2.0, 1.0, 0.5, 1.5, 1.0])
    query = jnp.array([[-1.0, 0.1], [0.2, 0.0], [1.1, -0.1], [1.8, 0.7]])
    reg_result = KNeighborsRegressorRecipe(3, metric="squared-euclidean").fit_batch(
        MLBatch(features, targets, sample_weight=weights)
    )
    cls_result = KNeighborsClassifierRecipe(
        3, class_count=2, metric="squared-euclidean"
    ).fit_batch(MLBatch(features, labels, sample_weight=weights))
    reg_model = reg_result.as_trainable()
    cls_model = cls_result.as_trainable()

    assert jnp.allclose(reg_model(query), reg_model.predict_chunked(query, chunk_size=2))
    assert jnp.allclose(cls_model(query), cls_model.predict_chunked(query, chunk_size=2))
    assert jax.jit(reg_model)(query).shape == (4,)
    assert jax.vmap(reg_model)(query).shape == (4,)
    assert cls_model.probabilities(query).shape == (4, 2)
    assert cls_model.predict(query).dtype == jnp.int32
    assert "neighbor_indices" in reg_result.gradient_contract.nondifferentiable_outputs
    assert "predict" in cls_result.gradient_contract.nondifferentiable_outputs


def test_neighbor_cases_masks_outputs_and_fixed_capacity_status_are_explicit():
    features, targets, labels = _data()
    case_features = jnp.stack((features, features + jnp.array([0.2, -0.1])), axis=0)
    case_targets = jnp.stack((targets, 2.0 * targets), axis=0)
    case_labels = jnp.stack((labels, 1 - labels), axis=0)
    sample_mask = jnp.array([True, True, False, True, True, True])

    reg = KNeighborsRegressorRecipe(2).fit_batch(
        MLBatch(case_features, case_targets, sample_mask=sample_mask)
    )
    cls = KNeighborsClassifierRecipe(2, class_count=2).fit_batch(
        MLBatch(case_features, case_labels, target_mask=sample_mask)
    )
    assert reg.as_trainable()(case_features[:, :2]).shape == (2, 2)
    assert cls.as_trainable()(case_features[:, :2]).shape == (2, 2, 2)
    assert reg.diagnostics.effective_samples.shape == (2,)
    assert cls.diagnostics.effective_samples.shape == (2,)

    exhausted = KNeighborsRegressorRecipe(2, capacity=3).fit_batch(
        MLBatch(features, targets)
    )
    assert not exhausted.valid
    assert exhausted.status == ML_CAPACITY_EXHAUSTED
    assert exhausted.as_trainable().support.shape == (3, 2)


def test_neighbor_recipes_require_explicit_sparse_materialization():
    dense = jnp.array(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 0.5], [2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
    )
    sparse = SparseFeatures(
        jnp.array([[1.0, 2.0], [-1.0, 0.5], [2.0, 1.0], [-1.0, 1.0]]),
        jnp.array([[0, 2], [1, 2], [0, 1], [0, 2]]),
        feature_count=3,
    )
    targets = jnp.array([1.0, -0.2, 2.0, 0.4])
    recipe = KNeighborsRegressorRecipe(2, metric="manhattan")
    dense_model = recipe.fit_batch(MLBatch(dense, targets)).as_trainable()
    with pytest.raises(TypeError, match="requires dense features"):
        recipe.fit_batch(MLBatch(sparse, targets))
    explicit_model = recipe.fit_batch(MLBatch(sparse.to_dense(), targets)).as_trainable()
    assert jnp.allclose(dense_model(dense), explicit_model(dense))


def test_soft_kernel_neighbors_are_smooth_and_distinct_from_hard_top_k():
    features, targets, labels = _data()
    query = jnp.array([[0.05, 0.1], [0.9, -0.2]])
    soft_reg_result = KernelNeighborsRegressorRecipe(temperature=0.4).fit_batch(
        MLBatch(features, targets)
    )
    soft_cls_result = KernelNeighborsClassifierRecipe(
        class_count=2, temperature=0.4
    ).fit_batch(MLBatch(features, labels))
    hard_model = (
        KNeighborsRegressorRecipe(1).fit_batch(MLBatch(features, targets)).as_trainable()
    )
    soft_reg = soft_reg_result.as_trainable()
    soft_cls = soft_cls_result.as_trainable()

    assert not jnp.allclose(soft_reg(query), hard_model(query))
    assert soft_reg.weights(query).shape == (2, features.shape[0])
    assert jnp.allclose(jnp.sum(soft_reg.weights(query), axis=-1), 1.0)
    assert jnp.allclose(jnp.sum(soft_cls.probabilities(query), axis=-1), 1.0)
    assert soft_reg_result.gradient_contract.fit_mode == "relaxed"
    assert soft_cls_result.gradient_contract.fit_mode == "relaxed"
    assert "predict" in soft_cls_result.gradient_contract.nondifferentiable_outputs

    input_gradient = jax.grad(lambda point: soft_reg(point) ** 2)(query[0])
    target_gradient = jax.grad(
        lambda value: (
            KernelNeighborsRegressorRecipe(temperature=0.4)
            .fit_batch(MLBatch(features, value))
            .as_trainable()(query[0])
        )
    )(targets)
    weight_gradient = jax.grad(
        lambda weight: (
            KernelNeighborsRegressorRecipe(temperature=0.4)
            .fit_batch(MLBatch(features, targets, sample_weight=weight))
            .as_trainable()(query[0])
        )
    )(jnp.ones((features.shape[0],)))
    temperature_gradient = jax.grad(
        lambda temperature: (
            KernelNeighborsRegressorRecipe(temperature=temperature)
            .fit_batch(MLBatch(features, targets))
            .as_trainable()(query[0])
        )
    )(jnp.array(0.4))
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(target_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jnp.isfinite(temperature_gradient)


def test_radius_regression_and_classification_expose_empty_and_hard_semantics():
    features, targets, labels = _data()
    reg_result = RadiusNeighborsRegressorRecipe(0.35).fit_batch(
        MLBatch(features, targets)
    )
    cls_result = RadiusNeighborsClassifierRecipe(0.35, class_count=2).fit_batch(
        MLBatch(features, labels)
    )
    reg_model = reg_result.as_trainable()
    cls_model = cls_result.as_trainable()
    far = jnp.array([[20.0, 20.0]])

    assert jnp.isnan(reg_model(far)[0])
    assert jnp.all(cls_model.probabilities(far) == 0.0)
    assert cls_model.predict(far)[0] == -1
    assert "radius_membership" in reg_result.gradient_contract.nondifferentiable_outputs
    assert "radius_membership" in cls_result.gradient_contract.nondifferentiable_outputs
    assert jnp.allclose(
        reg_model(features[:4]),
        reg_model.predict_chunked(features[:4], chunk_size=2),
        equal_nan=True,
    )
