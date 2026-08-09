#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INFEASIBLE,
    ML_NONCONVERGED,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.linear import (
    GammaModel,
    GammaRegressorRecipe,
    LogisticClassifierModel,
    LogisticRegressionRecipe,
    MultinomialLogisticModel,
    MultinomialLogisticRegressionRecipe,
    PoissonModel,
    PoissonRegressorRecipe,
    TweedieModel,
    TweedieRegressorRecipe,
)


def _features():
    return jnp.array(
        [
            [-2.0, 0.0, 1.0],
            [-1.4, 1.0, 0.0],
            [-0.8, 0.0, -1.0],
            [-0.2, -1.0, 0.0],
            [0.4, 0.0, 1.2],
            [0.9, 1.1, 0.0],
            [1.5, 0.0, -0.8],
            [2.1, -0.7, 0.0],
            [2.7, 0.0, 0.6],
        ]
    )


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


def test_one_step_glm_updates_match_weighted_score_equations():
    features = jnp.array([[2.0]])
    weight = jnp.array([2.0])
    logistic = (
        LogisticRegressionRecipe(
            learning_rate=0.1,
            fit_intercept=False,
            max_iterations=1,
            tolerance=1e6,
        )
        .fit_batch(MLBatch(features, jnp.array([1]), sample_weight=weight))
        .as_trainable()
    )
    assert jnp.allclose(logistic.coefficients, jnp.array([0.2]))

    poisson = (
        PoissonRegressorRecipe(
            learning_rate=0.1,
            fit_intercept=False,
            max_iterations=1,
            tolerance=1e6,
        )
        .fit_batch(MLBatch(features, jnp.array([3.0]), sample_weight=weight))
        .as_trainable()
    )
    assert jnp.allclose(poisson.coefficients, jnp.array([0.8]))


def test_binary_logistic_probabilities_labels_sparse_jit_vmap_and_declared_gradients():
    features = _features()
    targets = (features[:, 0] - 0.3 * features[:, 1] > 0.0).astype(jnp.int32)
    weights = jnp.linspace(0.5, 1.5, features.shape[0])
    schema = TargetSchema("binary", class_labels=(-1, 1))
    labeled_targets = jnp.where(targets == 1, 1, -1)
    recipe = LogisticRegressionRecipe(l2_strength=0.1, max_iterations=4, tolerance=1e6)
    result = recipe.fit_batch(
        MLBatch(
            features,
            labeled_targets,
            target_mask=jnp.ones_like(targets, dtype=bool).at[2].set(False),
            sample_weight=weights,
            target_schema=schema,
        )
    )
    model = result.as_trainable()
    assert isinstance(model, LogisticClassifierModel)
    assert model(features).shape == targets.shape
    assert model.predict_proba(features).shape == (features.shape[0], 2)
    assert model.predict(features).shape == targets.shape
    assert jax.jit(model)(features).shape == targets.shape
    assert jax.vmap(model)(features).shape == targets.shape
    assert result.gradient_contract.fit_targets == "none"
    assert {"predict", "predict_indices"}.issubset(
        result.gradient_contract.nondifferentiable_outputs
    )
    _assert_model_gradients(model, features[0])

    sparse_model = recipe.fit_batch(
        MLBatch(
            _sparse(features),
            labeled_targets,
            sample_weight=weights,
            target_schema=schema,
        )
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == targets.shape

    base = LogisticRegressionRecipe(l2_strength=0.1, max_iterations=3, tolerance=1e6)

    def fit_loss(x, sample_weight, strength):
        fitted_recipe = eqx.tree_at(lambda item: item.l2_strength, base, strength)
        fitted = fitted_recipe.fit_batch(
            MLBatch(x, targets, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(features, weights, base.l2_strength)
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_multinomial_logistic_classes_case_axes_sparse_and_gradients():
    features = _features()
    targets = jnp.mod(jnp.arange(features.shape[0]), 3).astype(jnp.int32)
    recipe = MultinomialLogisticRegressionRecipe(
        3, l2_strength=0.1, max_iterations=4, tolerance=1e6
    )
    result = recipe.fit_batch(MLBatch(features, targets))
    model = result.as_trainable()
    assert isinstance(model, MultinomialLogisticModel)
    assert model(features).shape == (features.shape[0], 3)
    assert model.predict(features).shape == targets.shape
    assert jnp.allclose(jnp.sum(model.predict_proba(features), axis=-1), 1.0)
    assert jax.jit(model)(features).shape == (features.shape[0], 3)
    assert jax.vmap(model)(features).shape == (features.shape[0], 3)
    assert result.gradient_contract.fit_targets == "none"
    _assert_model_gradients(model, features[0])

    sparse_result = recipe.fit_batch(MLBatch(_sparse(features), targets))
    assert sparse_result.as_trainable()(_sparse(features)).shape == (
        features.shape[0],
        3,
    )

    cases = jnp.stack((features, 0.5 * features), axis=0)
    case_targets = jnp.stack((targets, jnp.roll(targets, 1)), axis=0)
    case_result = recipe.fit_batch(MLBatch(cases, case_targets))
    assert case_result.as_trainable()(cases).shape == (2, features.shape[0], 3)
    assert case_result.diagnostics.rank.shape == (2,)

    base = MultinomialLogisticRegressionRecipe(
        3, l2_strength=0.1, max_iterations=3, tolerance=1e6
    )

    def fit_loss(x, sample_weight, strength):
        fitted = (
            eqx.tree_at(lambda item: item.l2_strength, base, strength)
            .fit_batch(MLBatch(x, targets, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(
        features, jnp.ones((features.shape[0],)), base.l2_strength
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


@pytest.mark.parametrize(
    ("recipe", "model_type", "targets"),
    (
        (
            PoissonRegressorRecipe(
                l2_strength=0.1, learning_rate=1e-3, max_iterations=3, tolerance=1e6
            ),
            PoissonModel,
            jnp.array([1.0, 2.0, 1.0, 3.0, 2.0, 4.0, 3.0, 5.0, 4.0]),
        ),
        (
            GammaRegressorRecipe(
                l2_strength=0.1, learning_rate=1e-3, max_iterations=3, tolerance=1e6
            ),
            GammaModel,
            jnp.linspace(0.5, 2.5, 9),
        ),
        (
            TweedieRegressorRecipe(
                1.5,
                l2_strength=0.1,
                learning_rate=1e-3,
                max_iterations=3,
                tolerance=1e6,
            ),
            TweedieModel,
            jnp.linspace(0.0, 3.0, 9),
        ),
    ),
)
def test_log_link_glm_families_multioutput_sparse_prediction_and_all_fit_gradients(
    recipe, model_type, targets
):
    features = _features()
    multioutput = jnp.stack((targets, targets + 0.5), axis=-1)
    weights = jnp.linspace(0.7, 1.4, features.shape[0])
    result = recipe.fit_batch(
        MLBatch(
            features,
            multioutput,
            target_mask=jnp.ones_like(multioutput, dtype=bool).at[1, 1].set(False),
            sample_weight=weights,
        )
    )
    model = result.as_trainable()
    assert isinstance(model, model_type)
    assert model(features).shape == multioutput.shape
    assert jnp.all(model(features) > 0.0)
    assert jax.jit(model)(features).shape == multioutput.shape
    assert jax.vmap(model)(features).shape == multioutput.shape
    _assert_model_gradients(model, features[0])

    sparse_model = recipe.fit_batch(
        MLBatch(_sparse(features), multioutput, sample_weight=weights)
    ).as_trainable()
    assert sparse_model(_sparse(features)).shape == multioutput.shape

    def fit_loss(x, y, sample_weight, strength):
        fitted_recipe = eqx.tree_at(lambda item: item.l2_strength, recipe, strength)
        fitted = fitted_recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(fitted(features[:2])))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, multioutput, weights, recipe.l2_strength
    )
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)


def test_zero_weight_samples_do_not_change_weighted_logistic_or_poisson_objectives():
    features = _features()
    weights = jnp.ones((features.shape[0],)).at[0].set(0.0)
    binary = (features[:, 0] > 0.0).astype(jnp.int32)
    flipped = binary.at[0].set(1 - binary[0])
    logistic = LogisticRegressionRecipe(max_iterations=3, tolerance=1e6)
    first_logistic = logistic.fit_batch(
        MLBatch(features, binary, sample_weight=weights)
    ).as_trainable()
    second_logistic = logistic.fit_batch(
        MLBatch(features, flipped, sample_weight=weights)
    ).as_trainable()
    assert jnp.allclose(first_logistic.coefficients, second_logistic.coefficients)
    assert jnp.allclose(first_logistic.intercept, second_logistic.intercept)

    counts = jnp.arange(1.0, features.shape[0] + 1.0)
    changed_counts = counts.at[0].set(1e4)
    poisson = PoissonRegressorRecipe(max_iterations=3, tolerance=1e6)
    first_poisson = poisson.fit_batch(
        MLBatch(features, counts, sample_weight=weights)
    ).as_trainable()
    second_poisson = poisson.fit_batch(
        MLBatch(features, changed_counts, sample_weight=weights)
    ).as_trainable()
    assert jnp.allclose(first_poisson.coefficients, second_poisson.coefficients)
    assert jnp.allclose(first_poisson.intercept, second_poisson.intercept)


def test_glm_domain_failures_missing_classes_and_nonconvergence_are_honest():
    features = _features()
    bad_poisson = PoissonRegressorRecipe(max_iterations=2).fit_batch(
        MLBatch(features, -jnp.ones((features.shape[0],)))
    )
    assert bad_poisson.status == ML_INFEASIBLE
    bad_gamma = GammaRegressorRecipe(max_iterations=2).fit_batch(
        MLBatch(features, jnp.zeros((features.shape[0],)))
    )
    assert bad_gamma.status == ML_INFEASIBLE
    missing_class = MultinomialLogisticRegressionRecipe(
        3, max_iterations=2, tolerance=1e6
    ).fit_batch(MLBatch(features, jnp.mod(jnp.arange(features.shape[0]), 2)))
    assert missing_class.status == ML_INFEASIBLE
    nonconverged = LogisticRegressionRecipe(max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(features, (features[:, 0] > 0.0).astype(jnp.int32))
    )
    assert nonconverged.status == ML_NONCONVERGED
    with pytest.raises(ValueError, match="powers in"):
        TweedieRegressorRecipe(0.5)
    with pytest.raises(TypeError, match="real-valued features"):
        PoissonRegressorRecipe(max_iterations=2).fit_batch(
            MLBatch(
                features.astype(jnp.complex64),
                jnp.ones((features.shape[0],)),
            )
        )
    no_intercept = (
        GammaRegressorRecipe(fit_intercept=False, max_iterations=2, tolerance=1e6)
        .fit_batch(MLBatch(features, jnp.linspace(0.5, 2.0, features.shape[0])))
        .as_trainable()
    )
    assert jnp.all(no_intercept.intercept == 0.0)
