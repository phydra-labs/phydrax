#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_INSUFFICIENT_DATA, ML_NONCONVERGED, MLBatch, SparseFeatures
from phydrax.ml.outliers import (
    CovarianceOutlierModel,
    CovarianceOutlierRecipe,
    EllipticEnvelopeModel,
    EllipticEnvelopeRecipe,
    KernelDensityOutlierModel,
    KernelDensityOutlierRecipe,
    RobustNoveltyModel,
    RobustNoveltyRecipe,
)


def _features():
    return jnp.array(
        [
            [-1.8, -0.4],
            [-1.0, 0.7],
            [-0.5, -0.8],
            [0.1, 0.2],
            [0.7, 0.9],
            [1.3, -0.5],
            [2.0, 0.4],
            [4.5, 4.0],
        ]
    )


def _recipes_and_models():
    return [
        (
            CovarianceOutlierRecipe(contamination=0.25, shrinkage=0.1),
            CovarianceOutlierModel,
            "conditional",
        ),
        (
            EllipticEnvelopeRecipe(contamination=0.25, iterations=3, tolerance=1e6),
            EllipticEnvelopeModel,
            "conditional",
        ),
        (
            KernelDensityOutlierRecipe(bandwidth=0.8, contamination=0.25),
            KernelDensityOutlierModel,
            "smooth",
        ),
        (
            RobustNoveltyRecipe(contamination=0.25, iterations=3, tolerance=1e6),
            RobustNoveltyModel,
            "conditional",
        ),
    ]


@pytest.mark.parametrize("recipe,model_type,hyper_gradient", _recipes_and_models())
def test_continuous_outlier_models_have_score_prediction_membership_and_frozen_execution_contracts(
    recipe, model_type, hyper_gradient
):
    features = _features()
    result = recipe.fit_batch(MLBatch(features))
    model = result.as_trainable()
    points = jnp.array([[-0.2, 0.1], [5.0, 4.5]])
    scores = model(points)
    frozen_scores = result.model(points)
    predictions = model.predict(points)
    memberships = model.smooth_membership(points, temperature=0.4)

    assert isinstance(model, model_type)
    assert scores.shape == (2,)
    assert predictions.shape == (2,)
    assert predictions.dtype == jnp.bool_
    assert memberships.shape == (2,)
    assert jnp.all(jnp.isfinite(scores))
    assert jnp.all((memberships >= 0.0) & (memberships <= 1.0))
    assert jnp.array_equal(memberships > 0.5, predictions)
    assert jnp.allclose(frozen_scores, scores)
    assert result.diagnostics.threshold.shape == ()
    assert jnp.allclose(result.diagnostics.threshold, model.threshold)
    assert result.diagnostics.score_minimum <= result.diagnostics.threshold
    assert result.diagnostics.threshold <= result.diagnostics.score_maximum
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.prediction_parameters == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_targets == "none"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == hyper_gradient
    assert "predict" in result.gradient_contract.nondifferentiable_outputs


def test_outlier_case_sample_feature_target_axes_masks_and_statistical_weight_policy():
    base = _features()
    features = jnp.stack((base, base * jnp.array([1.1, 0.9])), axis=0)
    targets = jnp.stack(
        (jnp.arange(16.0).reshape(2, 8), -jnp.arange(16.0).reshape(2, 8)), axis=-1
    )
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 2, 1].set(False)
    sample_mask = jnp.array([True, True, True, True, True, True, True, False])
    sample_weight = jnp.array([1.0, 2.0, 9.0, 1.0, 3.0, 1.0, 2.0, 8.0])
    common = dict(
        feature_mask=feature_mask,
        target_mask=jnp.ones_like(targets, dtype=bool).at[:, 0, 1].set(False),
        sample_mask=sample_mask,
        sample_weight=sample_weight,
    )
    recipe = CovarianceOutlierRecipe(contamination=0.25, shrinkage=0.1)

    first = recipe.fit_batch(MLBatch(features, targets, measure_weight=1.0, **common))
    second = recipe.fit_batch(MLBatch(features, targets, measure_weight=100.0, **common))
    model = first.as_trainable()
    active = sample_mask.at[2].set(False)
    effective_weights = jnp.where(active, sample_weight, 0.0)
    expected_location = jnp.sum(
        effective_weights[None, :, None] * features, axis=1
    ) / jnp.sum(effective_weights)

    assert model.case_shape == (2,)
    assert model.location.shape == (2, 2)
    assert model.precision.shape == (2, 2, 2)
    assert model(features[:, :3]).shape == (2, 3)
    assert model(jnp.array([0.2, -0.1])).shape == (2,)
    assert first.diagnostics.effective_samples.shape == (2,)
    assert jnp.array_equal(first.diagnostics.effective_samples, jnp.array([6, 6]))
    assert jnp.allclose(model.location, expected_location)
    assert jnp.allclose(model.location, second.as_trainable().location)
    assert jnp.allclose(model.precision, second.as_trainable().precision)


@pytest.mark.parametrize("recipe,model_type,hyper_gradient", _recipes_and_models())
def test_continuous_outlier_prediction_parameter_fit_feature_and_fit_weight_gradients(
    recipe, model_type, hyper_gradient
):
    del model_type, hyper_gradient
    features = _features()
    weights = jnp.array([1.0, 1.3, 0.8, 1.5, 1.1, 0.9, 1.4, 1.05])
    point = jnp.array([0.35, -0.25])
    model = recipe.fit_batch(MLBatch(features, sample_weight=weights)).as_trainable()

    input_gradient = jax.grad(model)(point)

    parameter_gradient = eqx.filter_grad(lambda candidate: candidate(point))(model)
    parameter_leaves = [
        leaf
        for leaf in jax.tree_util.tree_leaves(parameter_gradient)
        if eqx.is_inexact_array(leaf)
    ]

    def feature_loss(value):
        fitted = recipe.fit_batch(MLBatch(value, sample_weight=weights)).as_trainable()
        return fitted(point)

    def weight_loss(value):
        fitted = recipe.fit_batch(MLBatch(features, sample_weight=value)).as_trainable()
        return fitted(point)

    feature_gradient = jax.grad(feature_loss)(features)
    weight_gradient = jax.grad(weight_loss)(weights)

    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.any(jnp.abs(input_gradient) > 1e-8)
    assert parameter_leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in parameter_leaves)
    assert any(jnp.any(jnp.abs(leaf) > 1e-8) for leaf in parameter_leaves)
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_continuous_outlier_scores_are_jittable_and_vmappable():
    features = _features()
    points = jnp.array([[-0.2, 0.1], [0.8, -0.4], [3.0, 2.5]])

    for recipe, _model_type, _hyper_gradient in _recipes_and_models():
        model = recipe.fit_batch(MLBatch(features)).as_trainable()
        assert jax.jit(model)(points).shape == (3,)
        assert jax.vmap(model)(points).shape == (3,)
        assert jnp.all(jnp.isfinite(jax.jit(model)(points)))


def test_continuous_outliers_support_complex_geometry_with_real_scores():
    base = _features()
    complex_features = base + 1j * jnp.flip(base, axis=-1) * 0.2
    point = jnp.array([0.2 + 0.1j, -0.3 + 0.05j])

    for recipe, _model_type, _hyper_gradient in _recipes_and_models():
        model = recipe.fit_batch(MLBatch(complex_features)).as_trainable()
        score = model(point)
        assert score.shape == ()
        assert jnp.isrealobj(score)
        assert jnp.isfinite(score)


def test_sparse_outlier_features_are_explicitly_rejected():
    dense = _features()
    sparse = SparseFeatures(
        dense,
        jnp.broadcast_to(jnp.arange(2, dtype=jnp.int32), dense.shape),
        feature_count=2,
    )

    for recipe, _model_type, _hyper_gradient in _recipes_and_models():
        with pytest.raises(TypeError, match="requires dense features"):
            recipe.fit_batch(MLBatch(sparse))


def test_continuous_outlier_invalid_statuses_are_declared_values():
    features = _features()
    insufficient_mask = jnp.array([True, True, False, False, False, False, False, False])
    insufficient = CovarianceOutlierRecipe(contamination=0.25).fit_batch(
        MLBatch(features, sample_mask=insufficient_mask)
    )
    nonconverged = RobustNoveltyRecipe(
        contamination=0.25, iterations=1, tolerance=1e-30
    ).fit_batch(MLBatch(features))

    assert not insufficient.valid
    assert insufficient.status == ML_INSUFFICIENT_DATA
    assert not nonconverged.valid
    assert nonconverged.status == ML_NONCONVERGED
