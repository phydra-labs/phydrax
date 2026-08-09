#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INFEASIBLE,
    ML_INSUFFICIENT_DATA,
    ML_NONFINITE,
    MLBatch,
    SparseFeatures,
    TargetSchema,
)
from phydrax.ml.naive_bayes import (
    BernoulliNaiveBayesModel,
    BernoulliNaiveBayesRecipe,
    CategoricalNaiveBayesModel,
    CategoricalNaiveBayesRecipe,
    ComplementNaiveBayesRecipe,
    GaussianNaiveBayesModel,
    GaussianNaiveBayesRecipe,
    MultinomialNaiveBayesModel,
    MultinomialNaiveBayesRecipe,
)


_TARGETS = jnp.repeat(jnp.arange(2, dtype=jnp.int32), 4)
_SCHEMA = TargetSchema("binary", names=("outcome",), class_labels=("control", "treated"))
_GAUSSIAN = jnp.array(
    [
        [-2.0, -1.0],
        [-1.5, -0.4],
        [-1.8, -1.4],
        [-1.1, -0.7],
        [1.0, 0.8],
        [1.5, 1.3],
        [1.9, 0.5],
        [2.2, 1.6],
    ]
)
_BERNOULLI = jnp.array(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
)
_COUNTS = jnp.array(
    [
        [5.0, 1.0, 0.0],
        [4.0, 2.0, 0.0],
        [6.0, 0.0, 1.0],
        [3.0, 1.0, 1.0],
        [0.0, 1.0, 5.0],
        [1.0, 0.0, 4.0],
        [0.0, 2.0, 6.0],
        [1.0, 1.0, 3.0],
    ]
)
_CATEGORIES = jnp.array(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [1.0, 1.0],
        [2.0, 1.0],
    ]
)


def _sparse(values):
    columns = jnp.broadcast_to(jnp.arange(values.shape[-1]), values.shape)
    return SparseFeatures(values, columns, feature_count=values.shape[-1])


@pytest.mark.parametrize(
    ("recipe", "features", "model_type", "method"),
    [
        (GaussianNaiveBayesRecipe(), _GAUSSIAN, GaussianNaiveBayesModel, "gaussian-nb"),
        (
            BernoulliNaiveBayesRecipe(threshold=0.5),
            _BERNOULLI,
            BernoulliNaiveBayesModel,
            "bernoulli-nb",
        ),
        (
            MultinomialNaiveBayesRecipe(alpha=0.5),
            _COUNTS,
            MultinomialNaiveBayesModel,
            "multinomial-nb",
        ),
        (
            ComplementNaiveBayesRecipe(alpha=0.5),
            _COUNTS,
            MultinomialNaiveBayesModel,
            "complement-nb",
        ),
        (
            CategoricalNaiveBayesRecipe((3, 2), alpha=0.5),
            _CATEGORIES,
            CategoricalNaiveBayesModel,
            "categorical-nb",
        ),
    ],
)
def test_every_naive_bayes_family_has_normalized_schema_aware_jit_vmap_behavior(
    recipe, features, model_type, method
):
    result = recipe.fit_batch(MLBatch(features, _TARGETS, target_schema=_SCHEMA))
    model = result.as_trainable()
    probabilities = result.model(features)

    assert isinstance(model, model_type)
    assert bool(result.valid)
    assert result.method == method
    assert result.diagnostics.method == method
    assert probabilities.shape == (8, 2)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.exp(model.predict_log_proba(features)), probabilities)
    assert model.predict(features).shape == (8,)
    assert jnp.array_equal(model.labels, jnp.arange(2))
    assert model.target_schema.class_labels == _SCHEMA.class_labels
    assert jax.jit(model)(features[:2]).shape == (2, 2)
    assert jax.vmap(model)(features[:2]).shape == (2, 2)
    if method == "complement-nb":
        assert model.complement
    expected_input_level = (
        "almost-everywhere" if method in {"bernoulli-nb", "categorical-nb"} else "smooth"
    )
    assert result.gradient_contract.prediction_inputs == expected_input_level
    assert result.gradient_contract.prediction_parameters == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert result.gradient_contract.fit_hyperparameters == "conditional"


def test_gaussian_nb_preserves_case_masks_product_weights_and_string_vocabulary():
    features = jnp.stack((_GAUSSIAN, 1.5 * _GAUSSIAN + 0.2))
    targets = jnp.stack((_TARGETS, _TARGETS))
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 2, 1].set(False)
    target_mask = jnp.ones_like(targets, dtype=bool).at[:, 3].set(False)
    sample_mask = jnp.arange(8) != 7
    sample_weight = jnp.linspace(0.5, 1.2, 8)
    measure_weight = jnp.linspace(1.4, 0.7, 8)
    batch = MLBatch(
        features,
        targets,
        feature_mask=feature_mask,
        target_mask=target_mask,
        sample_mask=sample_mask,
        sample_weight=sample_weight,
        measure_weight=measure_weight,
        target_schema=_SCHEMA,
    )
    result = GaussianNaiveBayesRecipe(weight_policy="product").fit_batch(batch)
    model = result.as_trainable()
    active = sample_mask & target_mask[0]
    effective = jnp.where(active, sample_weight * measure_weight, 0.0)
    expected_mass = jnp.sum(effective[:, None] * jax.nn.one_hot(_TARGETS, 2), axis=0)

    assert result.valid.shape == (2,)
    assert jnp.all(result.valid)
    assert result.diagnostics.class_mass.shape == (2, 2)
    assert jnp.allclose(result.diagnostics.class_mass[0], expected_mass)
    assert model(features).shape == (2, 8, 2)
    assert model.target_schema.class_labels == _SCHEMA.class_labels


def test_naive_bayes_dense_complex_and_out_of_sample_domains_reject_sparse_input():
    gaussian_recipe = GaussianNaiveBayesRecipe(num_classes=2)
    dense = gaussian_recipe.fit_batch(MLBatch(_GAUSSIAN, _TARGETS)).as_trainable()
    with pytest.raises(TypeError, match="requires dense features"):
        gaussian_recipe.fit_batch(MLBatch(_sparse(_GAUSSIAN), _TARGETS))
    complex_features = _GAUSSIAN.astype(jnp.complex64) * (1.0 + 0.3j)
    complex_model = gaussian_recipe.fit_batch(
        MLBatch(complex_features, _TARGETS)
    ).as_trainable()
    multinomial = (
        MultinomialNaiveBayesRecipe(num_classes=2)
        .fit_batch(MLBatch(_COUNTS, _TARGETS))
        .as_trainable()
    )
    with pytest.raises(TypeError, match="requires dense features"):
        MultinomialNaiveBayesRecipe(num_classes=2).fit_batch(
            MLBatch(_sparse(_COUNTS), _TARGETS)
        )

    assert jnp.all(jnp.isfinite(complex_model(complex_features[:3])))
    assert jnp.all(jnp.isfinite(multinomial(_COUNTS[:3])))
    assert jnp.all(jnp.isnan(multinomial(jnp.array([[-1.0, 0.0, 1.0]]))))
    categorical = (
        CategoricalNaiveBayesRecipe((3, 2), num_classes=2)
        .fit_batch(MLBatch(_CATEGORIES, _TARGETS))
        .as_trainable()
    )
    assert jnp.all(jnp.isnan(categorical(jnp.array([[3.0, 0.0]]))))
    assert jnp.all(jnp.isnan(categorical(jnp.array([[1.5, 0.0]]))))
    with pytest.raises(ValueError, match="Expected 2 features"):
        dense(jnp.ones((2, 3)))
    with pytest.raises(TypeError, match="real-valued"):
        BernoulliNaiveBayesRecipe(num_classes=2).fit_batch(
            MLBatch(_BERNOULLI.astype(jnp.complex64), _TARGETS)
        )
    with pytest.raises(TypeError, match="real count"):
        MultinomialNaiveBayesRecipe(num_classes=2).fit_batch(
            MLBatch(_COUNTS.astype(jnp.complex64), _TARGETS)
        )
    with pytest.raises(TypeError, match="real category"):
        CategoricalNaiveBayesRecipe((3, 2), num_classes=2).fit_batch(
            MLBatch(_CATEGORIES.astype(jnp.complex64), _TARGETS)
        )


@pytest.mark.parametrize(
    ("recipe", "features", "probe", "replace_hyperparameter", "initial"),
    [
        (
            GaussianNaiveBayesRecipe(num_classes=2, var_smoothing=0.02),
            _GAUSSIAN,
            jnp.array([0.2, -0.1]),
            lambda item, value: eqx.tree_at(
                lambda fitted: fitted.var_smoothing, item, value
            ),
            jnp.array(0.02),
        ),
        (
            BernoulliNaiveBayesRecipe(num_classes=2, alpha=0.7, threshold=0.5),
            _BERNOULLI,
            jnp.array([1.0, 0.0, 1.0]),
            lambda item, value: eqx.tree_at(lambda fitted: fitted.alpha, item, value),
            jnp.array(0.7),
        ),
        (
            MultinomialNaiveBayesRecipe(num_classes=2, alpha=0.7),
            _COUNTS,
            jnp.array([1.0, 2.0, 1.0]),
            lambda item, value: eqx.tree_at(lambda fitted: fitted.alpha, item, value),
            jnp.array(0.7),
        ),
        (
            ComplementNaiveBayesRecipe(num_classes=2, alpha=0.7),
            _COUNTS,
            jnp.array([1.0, 2.0, 1.0]),
            lambda item, value: eqx.tree_at(lambda fitted: fitted.alpha, item, value),
            jnp.array(0.7),
        ),
        (
            CategoricalNaiveBayesRecipe((3, 2), num_classes=2, alpha=0.7),
            _CATEGORIES,
            jnp.array([1.0, 0.0]),
            lambda item, value: eqx.tree_at(lambda fitted: fitted.alpha, item, value),
            jnp.array(0.7),
        ),
    ],
)
def test_every_naive_bayes_family_has_its_declared_fit_and_prediction_gradients(
    recipe, features, probe, replace_hyperparameter, initial
):
    weights = jnp.linspace(0.8, 1.3, 8)

    def fit_loss(values, sample_weight, hyperparameter):
        configured = replace_hyperparameter(recipe, hyperparameter)
        model = configured.fit_batch(
            MLBatch(values, _TARGETS, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model.decision_function(probe)))

    fit_gradients = jax.grad(fit_loss, argnums=(0, 1, 2))(features, weights, initial)
    model = recipe.fit_batch(
        MLBatch(features, _TARGETS, sample_weight=weights)
    ).as_trainable()
    input_gradient = jax.grad(
        lambda point: jnp.sum(jnp.square(model.decision_function(point)))
    )(probe)
    if isinstance(model, GaussianNaiveBayesModel):
        parameter_gradient = jax.grad(
            lambda parameter: jnp.sum(
                jnp.square(
                    eqx.tree_at(
                        lambda item: item.means, model, parameter
                    ).decision_function(probe)
                )
            )
        )(model.means)
    else:
        parameter_gradient = jax.grad(
            lambda parameter: jnp.sum(
                jnp.square(
                    eqx.tree_at(
                        lambda item: item.feature_log_prob, model, parameter
                    ).decision_function(probe)
                )
            )
        )(model.feature_log_prob)

    assert all(jnp.all(jnp.isfinite(gradient)) for gradient in fit_gradients)
    assert jnp.all(jnp.isfinite(input_gradient))
    assert jnp.all(jnp.isfinite(parameter_gradient))


def test_naive_bayes_failures_report_empty_single_class_nonfinite_weight_and_domains():
    empty = GaussianNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(
            _GAUSSIAN,
            _TARGETS,
            sample_mask=jnp.zeros(8, dtype=bool),
        )
    )
    single_class = GaussianNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_GAUSSIAN, jnp.zeros(8, dtype=jnp.int32))
    )
    nonfinite = GaussianNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_GAUSSIAN.at[0, 0].set(jnp.nan), _TARGETS)
    )
    negative_weight = GaussianNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_GAUSSIAN, _TARGETS, sample_weight=jnp.ones(8).at[2].set(-1.0))
    )
    negative_count = MultinomialNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_COUNTS.at[1, 0].set(-1.0), _TARGETS)
    )
    nonfinite_binary = BernoulliNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_BERNOULLI.at[0, 0].set(jnp.nan), _TARGETS)
    )
    negative_complement = ComplementNaiveBayesRecipe(num_classes=2).fit_batch(
        MLBatch(_COUNTS.at[1, 0].set(-1.0), _TARGETS)
    )
    invalid_category = CategoricalNaiveBayesRecipe((3, 2), num_classes=2).fit_batch(
        MLBatch(_CATEGORIES.at[1, 0].set(3.0), _TARGETS)
    )

    assert int(empty.status) == ML_INSUFFICIENT_DATA
    assert int(single_class.status) == ML_INSUFFICIENT_DATA
    assert int(nonfinite.status) == ML_INFEASIBLE
    assert int(negative_weight.status) == ML_NONFINITE
    assert int(negative_count.status) == ML_INFEASIBLE
    assert int(nonfinite_binary.status) == ML_INFEASIBLE
    assert int(negative_complement.status) == ML_INFEASIBLE
    assert int(invalid_category.status) == ML_INFEASIBLE
    assert not bool(negative_count.diagnostics.domain_valid)
    assert not bool(invalid_category.diagnostics.domain_valid)


def test_naive_bayes_rejects_invalid_capacity_and_target_axes():
    with pytest.raises(ValueError, match="at least two categories"):
        CategoricalNaiveBayesRecipe((3, 1))
    with pytest.raises(ValueError, match="align with the feature axis"):
        CategoricalNaiveBayesRecipe((3, 2, 2), num_classes=2).fit_batch(
            MLBatch(_CATEGORIES, _TARGETS)
        )
    with pytest.raises(ValueError, match="scalar class label"):
        GaussianNaiveBayesRecipe(num_classes=2).fit_batch(
            MLBatch(_GAUSSIAN, jax.nn.one_hot(_TARGETS, 2))
        )
