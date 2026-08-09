#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_INFEASIBLE, MLBatch, SparseFeatures
from phydrax.ml.linear import (
    PassiveAggressiveClassifierModel,
    PassiveAggressiveClassifierRecipe,
    PassiveAggressiveRegressorModel,
    PassiveAggressiveRegressorRecipe,
    PerceptronModel,
    PerceptronRecipe,
    SGDClassifierModel,
    SGDClassifierRecipe,
    SGDRegressorModel,
    SGDRegressorRecipe,
)


def _online_data():
    features = jnp.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.4, 1.0, 0.0],
            [-0.8, 0.0, -1.0],
            [-0.2, -1.0, 0.0],
            [0.4, 0.0, 1.2],
            [0.9, 1.1, 0.0],
            [1.5, 0.0, -0.8],
            [2.1, -0.7, 0.0],
        ]
    )
    regression = features @ jnp.array([[1.0, -0.2], [0.4, 0.8], [-0.3, 0.5]])
    classification = (features[:, 0] + 0.2 * features[:, 1] > 0.0).astype(jnp.int32)
    return features, regression, classification


def _sparse(features):
    columns = jnp.argsort(jnp.where(features != 0.0, 0, 1), axis=-1)[:, :2]
    values = jnp.take_along_axis(features, columns, axis=-1)
    valid = jnp.take_along_axis(features != 0.0, columns, axis=-1)
    return SparseFeatures(values, columns, feature_count=features.shape[-1], valid=valid)


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


def test_sgd_regression_multioutput_masks_weights_sparse_determinism_jit_and_gradients():
    features, targets, _ = _online_data()
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    recipe = SGDRegressorRecipe(
        learning_rate=1e-2,
        l1_strength=1e-3,
        l2_strength=1e-3,
        passes=3,
    )
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features, targets))
    batch = MLBatch(
        features,
        targets,
        target_mask=jnp.ones_like(targets, dtype=bool).at[2, 1].set(False),
        sample_mask=jnp.ones((features.shape[0],), dtype=bool).at[5].set(False),
        sample_weight=weights,
    )
    first = recipe.fit_batch(batch, key=jax.random.key(1))
    second = recipe.fit_batch(batch, key=jax.random.key(1))
    model = first.as_trainable()
    assert isinstance(model, SGDRegressorModel)
    assert jnp.allclose(model.coefficients, second.as_trainable().coefficients)
    assert model(features).shape == targets.shape
    assert jax.jit(model)(features).shape == targets.shape
    assert jax.vmap(model)(features).shape == targets.shape
    _assert_model_gradients(model, features[0])
    sparse_model = recipe.fit_batch(
        MLBatch(_sparse(features), targets, sample_weight=weights),
        key=jax.random.key(2),
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == targets.shape

    base = SGDRegressorRecipe(learning_rate=1e-2, passes=2)

    def fit_loss(x, y, sample_weight, learning_rate):
        fitted = (
            eqx.tree_at(lambda item: item.learning_rate, base, learning_rate)
            .fit_batch(MLBatch(x, y, sample_weight=sample_weight), key=jax.random.key(3))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, targets, weights, base.learning_rate
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_sgd_classifier_losses_case_multilabel_probabilities_sparse_and_gradients():
    features, _, targets = _online_data()
    weights = jnp.linspace(0.7, 1.3, features.shape[0])
    for loss in ("logistic", "hinge"):
        recipe = SGDClassifierRecipe(loss=loss, learning_rate=1e-2, passes=3)
        result = recipe.fit_batch(
            MLBatch(features, targets, sample_weight=weights), key=jax.random.key(4)
        )
        model = result.as_trainable()
        assert isinstance(model, SGDClassifierModel)
        assert model(features).shape == targets.shape
        assert model.predict(features).shape == targets.shape
        assert jax.jit(model)(features).shape == targets.shape
        assert jax.vmap(model)(features).shape == targets.shape
        assert result.gradient_contract.fit_targets == "none"
        if loss == "logistic":
            assert model.predict_proba(features).shape == (features.shape[0], 2)
        else:
            with pytest.raises(ValueError, match="does not define calibrated"):
                model.predict_proba(features)
        _assert_model_gradients(model, features[0])
        sparse_model = recipe.fit_batch(
            MLBatch(_sparse(features), targets, sample_weight=weights),
            key=jax.random.key(5),
        ).as_trainable()
        assert sparse_model(_sparse(features)).shape == targets.shape

    multilabel = jnp.stack((targets, 1 - targets), axis=-1)
    cases = jnp.stack((features, 0.8 * features), axis=0)
    case_targets = jnp.stack((multilabel, jnp.flip(multilabel, axis=0)), axis=0)
    case_result = SGDClassifierRecipe(passes=2).fit_batch(
        MLBatch(cases, case_targets), key=jax.random.key(6)
    )
    assert case_result.as_trainable()(cases).shape == case_targets.shape

    base = SGDClassifierRecipe(loss="logistic", learning_rate=1e-2, passes=2)

    def fit_loss(x, sample_weight, learning_rate):
        fitted = (
            eqx.tree_at(lambda item: item.learning_rate, base, learning_rate)
            .fit_batch(
                MLBatch(x, targets, sample_weight=sample_weight), key=jax.random.key(7)
            )
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(
        features, weights, base.learning_rate
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_perceptron_key_determinism_sparse_hard_outputs_and_unrolled_gradients():
    features, _, targets = _online_data()
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    recipe = PerceptronRecipe(learning_rate=0.5, passes=3)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(MLBatch(features, targets))
    first = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights), key=jax.random.key(8)
    )
    second = recipe.fit_batch(
        MLBatch(features, targets, sample_weight=weights), key=jax.random.key(8)
    )
    model = first.as_trainable()
    assert isinstance(model, PerceptronModel)
    assert jnp.allclose(model.coefficients, second.as_trainable().coefficients)
    assert model(features).shape == targets.shape
    assert model.predict(features).shape == targets.shape
    assert first.gradient_contract.fit_targets == "none"
    assert "mistake_updates" in first.gradient_contract.nondifferentiable_outputs
    assert jax.jit(model)(features).shape == targets.shape
    _assert_model_gradients(model, features[0])
    sparse_model = recipe.fit_batch(
        MLBatch(_sparse(features), targets), key=jax.random.key(9)
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == targets.shape

    base = PerceptronRecipe(learning_rate=0.5, passes=2)

    def fit_loss(x, sample_weight, learning_rate):
        fitted = (
            eqx.tree_at(lambda item: item.learning_rate, base, learning_rate)
            .fit_batch(
                MLBatch(x, targets, sample_weight=sample_weight), key=jax.random.key(10)
            )
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(
        features, weights, base.learning_rate
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


@pytest.mark.parametrize("variant", ("pa1", "pa2"))
def test_passive_aggressive_regression_and_classification_variants_sparse_and_gradients(
    variant,
):
    features, regression, classification = _online_data()
    scalar_targets = regression[:, 0]
    weights = jnp.linspace(0.8, 1.2, features.shape[0])
    reg_recipe = PassiveAggressiveRegressorRecipe(
        aggressiveness=0.5, epsilon=0.05, variant=variant, passes=2
    )
    cls_recipe = PassiveAggressiveClassifierRecipe(
        aggressiveness=0.5, variant=variant, passes=2
    )
    reg_result = reg_recipe.fit_batch(
        MLBatch(features, scalar_targets, sample_weight=weights),
        key=jax.random.key(11),
    )
    cls_result = cls_recipe.fit_batch(
        MLBatch(features, classification, sample_weight=weights),
        key=jax.random.key(12),
    )
    reg_model = reg_result.as_trainable()
    cls_model = cls_result.as_trainable()
    assert isinstance(reg_model, PassiveAggressiveRegressorModel)
    assert isinstance(cls_model, PassiveAggressiveClassifierModel)
    assert reg_model(features).shape == scalar_targets.shape
    assert cls_model(features).shape == classification.shape
    assert cls_model.predict(features).shape == classification.shape
    assert cls_result.gradient_contract.fit_targets == "none"
    assert jax.jit(reg_model)(features).shape == scalar_targets.shape
    assert jax.vmap(cls_model)(features).shape == classification.shape
    _assert_model_gradients(reg_model, features[0])
    _assert_model_gradients(cls_model, features[0])
    sparse_reg = reg_recipe.fit_batch(
        MLBatch(_sparse(features), scalar_targets), key=jax.random.key(13)
    ).as_trainable()
    sparse_cls = cls_recipe.fit_batch(
        MLBatch(_sparse(features), classification), key=jax.random.key(14)
    ).as_trainable()
    assert sparse_reg(_sparse(features)).shape == scalar_targets.shape
    assert sparse_cls(_sparse(features)).shape == classification.shape

    def regression_loss(x, y, sample_weight, aggressiveness):
        fitted = (
            eqx.tree_at(lambda item: item.aggressiveness, reg_recipe, aggressiveness)
            .fit_batch(MLBatch(x, y, sample_weight=sample_weight), key=jax.random.key(15))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    regression_gradients = jax.grad(regression_loss, argnums=(0, 1, 2, 3))(
        features, scalar_targets, weights, reg_recipe.aggressiveness
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in regression_gradients)

    def classification_loss(x, sample_weight, aggressiveness):
        fitted = (
            eqx.tree_at(lambda item: item.aggressiveness, cls_recipe, aggressiveness)
            .fit_batch(
                MLBatch(x, classification, sample_weight=sample_weight),
                key=jax.random.key(16),
            )
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    classification_gradients = jax.grad(classification_loss, argnums=(0, 1, 2))(
        features, weights, cls_recipe.aggressiveness
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in classification_gradients)


def test_one_step_online_updates_match_weighted_equations():
    features = jnp.array([[2.0]])
    weight = jnp.array([2.0])
    regression = (
        SGDRegressorRecipe(learning_rate=0.1, passes=1, shuffle=False)
        .fit_batch(MLBatch(features, jnp.array([3.0]), sample_weight=weight))
        .as_trainable()
    )
    assert jnp.allclose(regression.coefficients, jnp.array([1.2]))
    assert jnp.allclose(regression.intercept, 0.6)

    logistic = (
        SGDClassifierRecipe(loss="logistic", learning_rate=0.1, passes=1, shuffle=False)
        .fit_batch(MLBatch(features, jnp.array([1]), sample_weight=weight))
        .as_trainable()
    )
    assert jnp.allclose(logistic.coefficients, jnp.array([0.2]))
    assert jnp.allclose(logistic.intercept, 0.1)

    perceptron = (
        PerceptronRecipe(learning_rate=0.5, passes=1, shuffle=False)
        .fit_batch(MLBatch(features, jnp.array([1]), sample_weight=weight))
        .as_trainable()
    )
    assert jnp.allclose(perceptron.coefficients, jnp.array([2.0]))
    assert jnp.allclose(perceptron.intercept, 1.0)


def test_online_updates_ignore_zero_weight_samples_exactly():
    features, regression, classification = _online_data()
    weights = jnp.ones((features.shape[0],)).at[2].set(0.0)
    changed_regression = regression.at[2].set(jnp.array([1e4, -1e4]))
    reg_recipe = SGDRegressorRecipe(passes=2, shuffle=False)
    first_regression = reg_recipe.fit_batch(
        MLBatch(features, regression, sample_weight=weights)
    ).as_trainable()
    second_regression = reg_recipe.fit_batch(
        MLBatch(features, changed_regression, sample_weight=weights)
    ).as_trainable()
    assert jnp.allclose(first_regression.coefficients, second_regression.coefficients)
    assert jnp.allclose(first_regression.intercept, second_regression.intercept)

    changed_classification = classification.at[2].set(1 - classification[2])
    classifier_recipe = PerceptronRecipe(passes=2, shuffle=False)
    first_classifier = classifier_recipe.fit_batch(
        MLBatch(features, classification, sample_weight=weights)
    ).as_trainable()
    second_classifier = classifier_recipe.fit_batch(
        MLBatch(features, changed_classification, sample_weight=weights)
    ).as_trainable()
    assert jnp.allclose(first_classifier.coefficients, second_classifier.coefficients)
    assert jnp.allclose(first_classifier.intercept, second_classifier.intercept)


def test_online_capacity_and_deterministic_no_shuffle_policy():
    features, regression, classification = _online_data()
    deterministic = SGDRegressorRecipe(passes=2, shuffle=False, fit_intercept=False)
    first = deterministic.fit_batch(MLBatch(features, regression)).as_trainable()
    second = deterministic.fit_batch(MLBatch(features, regression)).as_trainable()
    assert jnp.allclose(first.coefficients, second.coefficients)
    assert jnp.all(first.intercept == 0.0)
    with pytest.raises(ValueError, match="passes must be positive"):
        SGDRegressorRecipe(passes=0)
    with pytest.raises(ValueError, match="variant"):
        PassiveAggressiveClassifierRecipe(variant="invalid")
    invalid_targets = jnp.full((features.shape[0],), 2)
    invalid = SGDClassifierRecipe(passes=1, shuffle=False).fit_batch(
        MLBatch(features, invalid_targets)
    )
    assert invalid.status == ML_INFEASIBLE
    for recipe, targets in (
        (SGDClassifierRecipe(passes=1), classification),
        (PassiveAggressiveRegressorRecipe(passes=1), regression[:, 0]),
        (PassiveAggressiveClassifierRecipe(passes=1), classification),
    ):
        with pytest.raises(ValueError, match="explicit JAX key"):
            recipe.fit_batch(MLBatch(features, targets))
