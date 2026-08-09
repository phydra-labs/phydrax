#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax._model import AbstractArrayModel
from phydrax.ml import AbstractRecipe, linear, MLBatch


EXPECTED_EXPORTS = {
    "ElasticNetRecipe",
    "ElasticNetModel",
    "GammaModel",
    "GammaRegressorRecipe",
    "GeneralizedLinearModel",
    "GroupLassoRecipe",
    "GroupLassoModel",
    "HuberModel",
    "HuberRegressorRecipe",
    "LassoRecipe",
    "LassoModel",
    "LinearRegressorModel",
    "LinearScoreClassifierModel",
    "LogisticClassifierModel",
    "LogisticRegressionRecipe",
    "MultinomialLogisticModel",
    "MultinomialLogisticRegressionRecipe",
    "OLSRecipe",
    "OLSModel",
    "OnlineClassifierModel",
    "PassiveAggressiveClassifierRecipe",
    "PassiveAggressiveClassifierModel",
    "PassiveAggressiveRegressorRecipe",
    "PassiveAggressiveRegressorModel",
    "PerceptronRecipe",
    "PerceptronModel",
    "PoissonModel",
    "PoissonRegressorRecipe",
    "QuantileModel",
    "QuantileRegressorRecipe",
    "RANSACModel",
    "RANSACRegressorRecipe",
    "RidgeRecipe",
    "RidgeModel",
    "RobustDiagnostics",
    "SGDClassifierRecipe",
    "SGDClassifierModel",
    "SGDRegressorRecipe",
    "SGDRegressorModel",
    "SparseGroupLassoRecipe",
    "SparseGroupLassoModel",
    "TheilSenModel",
    "TheilSenRegressorRecipe",
    "TikhonovRecipe",
    "TikhonovModel",
    "TweedieModel",
    "TweedieRegressorRecipe",
}


def test_linear_package_exports_complete_intentional_surface():
    assert set(linear.__all__) == EXPECTED_EXPORTS
    for name in EXPECTED_EXPORTS:
        assert getattr(linear, name).__module__.startswith("phydrax.ml.linear")
    for name in EXPECTED_EXPORTS:
        value = getattr(linear, name)
        if name.endswith("Recipe"):
            assert issubclass(value, AbstractRecipe)
        elif name.endswith("Model"):
            assert issubclass(value, AbstractArrayModel)


def test_recipes_and_fitted_models_are_immutable_modules():
    recipe = linear.OLSRecipe()
    with pytest.raises(AttributeError):
        recipe.fit_intercept = False
    fitted = recipe.fit_batch(
        MLBatch(
            jnp.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
            jnp.array([0.0, 1.0, -1.0, 0.0]),
        )
    ).as_trainable()
    with pytest.raises(AttributeError):
        fitted.intercept = jnp.asarray(0.0)


def test_generic_public_model_types_execute_their_declared_semantics():
    coefficients = jnp.array([[1.0], [-0.5]])
    intercept = jnp.array([0.2])
    inputs = jnp.array([[2.0, 1.0], [-1.0, 0.5]])
    regression = linear.LinearRegressorModel(
        coefficients, intercept, case_shape=(), target_shape=()
    )
    score = linear.LinearScoreClassifierModel(
        coefficients,
        intercept,
        jnp.array([-2, 3]),
        case_shape=(),
        target_shape=(),
    )
    glm = linear.GeneralizedLinearModel(
        coefficients,
        intercept,
        case_shape=(),
        target_shape=(),
        inverse_link="exp",
    )
    probabilistic = linear.OnlineClassifierModel(
        coefficients,
        intercept,
        jnp.array([0, 1]),
        case_shape=(),
        target_shape=(),
        probabilistic=True,
    )
    margin = linear.OnlineClassifierModel(
        coefficients,
        intercept,
        jnp.array([0, 1]),
        case_shape=(),
        target_shape=(),
    )
    expected_score = inputs @ coefficients[:, 0] + intercept[0]
    assert jnp.allclose(regression(inputs), expected_score)
    assert jnp.allclose(score(inputs), expected_score)
    assert jnp.array_equal(score.predict(inputs), jnp.array([3, -2]))
    assert jnp.allclose(glm(inputs), jnp.exp(expected_score))
    assert jnp.allclose(probabilistic(inputs), jax.nn.sigmoid(expected_score))
    assert probabilistic.predict_proba(inputs).shape == (inputs.shape[0], 2)
    assert jnp.allclose(margin(inputs), expected_score)
    with pytest.raises(ValueError, match="does not define calibrated"):
        margin.predict_proba(inputs)
    for model in (regression, score, glm, probabilistic, margin):
        assert model.in_size == 2
        assert model.input_binding().batch_mode == "pointwise"
