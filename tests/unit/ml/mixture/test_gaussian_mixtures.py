#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import (
    ML_INSUFFICIENT_DATA,
    ML_NONCONVERGED,
    ML_NONFINITE,
    ML_SUCCESS,
    MLBatch,
)
from phydrax.ml.mixture import (
    BayesianGaussianMixture,
    BayesianGaussianMixtureModel,
    GaussianMixture,
    GaussianMixtureModel,
)


_DATA = jnp.array(
    [
        [-3.0, -1.0],
        [3.0, 1.0],
        [-2.5, -0.4],
        [2.5, 0.4],
        [-2.0, -1.4],
        [2.0, 1.4],
    ]
)


def _gaussian(covariance_type="full"):
    return GaussianMixture(
        2,
        covariance_type=covariance_type,
        initialization="first",
        max_iterations=12,
        tolerance=1e6,
        regularization=1e-3,
    )


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diagonal", "spherical"])
def test_gaussian_mixture_covariance_modes_are_spd_and_structurally_exact(
    covariance_type,
):
    result = _gaussian(covariance_type).fit_batch(MLBatch(_DATA))
    model = result.as_trainable()
    covariance = model.covariance
    precision = model.precision
    identity = jnp.eye(_DATA.shape[-1])

    assert result.status == ML_SUCCESS
    assert isinstance(model, GaussianMixtureModel)
    assert covariance.shape == (2, 2, 2)
    assert jnp.allclose(
        covariance,
        jnp.conj(jnp.swapaxes(covariance, -1, -2)),
        atol=1e-6,
    )
    assert jnp.all(jnp.linalg.eigvalsh(covariance) > 0.0)
    assert jnp.allclose(precision @ covariance, identity, atol=2e-4)

    if covariance_type == "tied":
        assert jnp.allclose(covariance[0], covariance[1], atol=1e-7)
    if covariance_type in ("diagonal", "spherical"):
        diagonal = jnp.diagonal(covariance, axis1=-2, axis2=-1)
        assert jnp.allclose(covariance, diagonal[..., :, None] * identity, atol=1e-7)
    if covariance_type == "spherical":
        diagonal = jnp.diagonal(covariance, axis1=-2, axis2=-1)
        assert jnp.allclose(diagonal[..., 0], diagonal[..., 1], atol=1e-7)


def test_gaussian_mixture_preserves_cases_and_ignores_target_axes():
    cases = jnp.stack((_DATA, 2.0 * _DATA + jnp.array([10.0, -4.0])))
    targets = jnp.arange(2 * 6 * 3.0).reshape(2, 6, 3)
    recipe = _gaussian("full")
    result = recipe.fit_batch(MLBatch(cases, targets))
    model = result.as_trainable()
    probabilities = result.model(cases)

    assert jnp.all(result.status == ML_SUCCESS)
    assert result.status.shape == (2,)
    assert model.case_shape == (2,)
    assert model.means.shape == (2, 2, 2)
    assert probabilities.shape == (2, 6, 2)
    assert jnp.allclose(jnp.sum(probabilities, axis=-1), 1.0, atol=1e-6)
    assert jnp.array_equal(model.predict(cases), jnp.argmax(probabilities, axis=-1))
    assert jnp.all(jnp.isfinite(model.log_prob(cases)))
    assert recipe.component_count == 2


def test_mixture_product_weights_and_masks_determine_mean_and_mass():
    features = jnp.array([[0.0], [10.0], [100.0], [999.0]])
    batch = MLBatch(
        features,
        jnp.arange(4.0),
        feature_mask=jnp.array([[True], [True], [True], [False]]),
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 3.0, 99.0, 7.0]),
        measure_weight=jnp.array([2.0, 1.0, 5.0, 11.0]),
    )
    product = GaussianMixture(
        1,
        initialization="first",
        weight_policy="product",
        max_iterations=3,
        tolerance=1e6,
    ).fit_batch(batch)
    statistical = GaussianMixture(
        1,
        initialization="first",
        weight_policy="statistical",
        max_iterations=3,
        tolerance=1e6,
    ).fit_batch(batch)

    assert product.status == ML_SUCCESS
    assert jnp.allclose(product.as_trainable().means[0, 0], 6.0, atol=1e-5)
    assert jnp.allclose(statistical.as_trainable().means[0, 0], 7.5, atol=1e-5)
    assert jnp.allclose(product.diagnostics.component_mass, jnp.array([5.0]))
    assert jnp.allclose(product.diagnostics.effective_samples, 25.0 / 13.0)


def test_random_mixture_initialization_requires_and_replays_explicit_key():
    recipe = GaussianMixture(
        2,
        initialization="random",
        max_iterations=3,
        tolerance=1e6,
    )
    batch = MLBatch(_DATA)

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    key = jax.random.key(29)
    first = recipe.fit_batch(batch, key=key)
    second = recipe.fit_batch(batch, key=key)

    assert first.status == ML_SUCCESS
    assert jnp.array_equal(first.as_trainable().means, second.as_trainable().means)
    assert jnp.array_equal(
        first.as_trainable().covariance, second.as_trainable().covariance
    )
    assert jnp.array_equal(first.model(_DATA), second.model(_DATA))


def test_gaussian_model_ties_choose_lowest_component_but_probabilities_stay_soft():
    model = GaussianMixtureModel(
        jnp.array([0.5, 0.5]),
        jnp.array([[0.0], [0.0]]),
        jnp.ones((2, 1, 1)),
        jnp.ones((2, 1, 1)),
        jnp.zeros(2),
        covariance_type="full",
    )
    points = jnp.array([[-1.0], [0.0], [1.0]])

    assert jnp.allclose(model(points), jnp.full((3, 2), 0.5))
    assert jnp.array_equal(model.predict(points), jnp.zeros(3, dtype=jnp.int32))
    assert jax.jit(model)(points).shape == (3, 2)
    assert jax.vmap(model)(points).shape == (3, 2)
    input_gradient = jax.grad(lambda point: model.log_prob(point))(jnp.array([0.2]))
    assert jnp.all(jnp.isfinite(input_gradient))


@pytest.mark.parametrize("covariance_type", ["full", "tied", "diagonal", "spherical"])
def test_bayesian_mixture_exposes_each_covariance_mode_and_posterior_concentration(
    covariance_type,
):
    prior = 0.7
    result = BayesianGaussianMixture(
        2,
        covariance_type=covariance_type,
        concentration=prior,
        mean_precision=0.5,
        initialization="first",
        max_iterations=12,
        tolerance=1e6,
        regularization=1e-3,
    ).fit_batch(MLBatch(_DATA))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert isinstance(model, BayesianGaussianMixtureModel)
    assert jnp.all(jnp.linalg.eigvalsh(model.covariance) > 0.0)
    if covariance_type == "tied":
        assert jnp.allclose(model.covariance[0], model.covariance[1], atol=1e-7)
    if covariance_type in ("diagonal", "spherical"):
        diagonal = jnp.diagonal(model.covariance, axis1=-2, axis2=-1)
        assert jnp.allclose(
            model.covariance, diagonal[..., :, None] * jnp.eye(2), atol=1e-7
        )
    if covariance_type == "spherical":
        diagonal = jnp.diagonal(model.covariance, axis1=-2, axis2=-1)
        assert jnp.allclose(diagonal[..., 0], diagonal[..., 1], atol=1e-7)
    assert jnp.allclose(
        model.concentration, result.diagnostics.component_mass + prior, atol=1e-6
    )
    assert jnp.all(model.concentration > prior)
    assert jnp.allclose(jnp.sum(model.mixing_weights), 1.0, atol=1e-6)
    assert jnp.allclose(jnp.sum(model(_DATA), axis=-1), 1.0, atol=1e-6)
    assert jnp.all(jnp.isfinite(model.log_prob(_DATA)))
    assert result.method == "bayesian-gaussian-mixture"
    assert result.gradient_contract.fit_mode == "unrolled"
    assert "positive variational prior" in result.gradient_contract.conditions


@pytest.mark.parametrize(
    "recipe",
    [
        GaussianMixture(
            2,
            initialization="first",
            max_iterations=4,
            tolerance=1e6,
            regularization=1e-3,
        ),
        BayesianGaussianMixture(
            2,
            initialization="first",
            max_iterations=4,
            tolerance=1e6,
            regularization=1e-3,
        ),
    ],
)
def test_each_mixture_family_exercises_declared_fit_feature_and_weight_gradients(recipe):
    weights = jnp.array([1.0, 1.1, 0.9, 1.2, 0.8, 1.3])
    point = jnp.array([0.25, -0.1])
    feature_gradient = jax.grad(
        lambda values: (
            recipe.fit_batch(MLBatch(values, sample_weight=weights))
            .as_trainable()
            .log_prob(point)
        )
    )(_DATA)
    weight_gradient = jax.grad(
        lambda value: (
            recipe.fit_batch(MLBatch(_DATA, sample_weight=value))
            .as_trainable()
            .log_prob(point)
        )
    )(weights)
    contract = recipe.fit_batch(MLBatch(_DATA, sample_weight=weights)).gradient_contract

    assert contract.prediction_inputs == "smooth"
    assert contract.prediction_parameters == "smooth"
    assert contract.fit_features == "conditional"
    assert contract.fit_weights == "conditional"
    assert contract.fit_hyperparameters == "conditional"
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


@pytest.mark.parametrize(
    "factory",
    [
        lambda regularization: GaussianMixture(
            2,
            initialization="first",
            max_iterations=4,
            tolerance=1e6,
            regularization=regularization,
        ),
        lambda regularization: BayesianGaussianMixture(
            2,
            initialization="first",
            max_iterations=4,
            tolerance=1e6,
            regularization=regularization,
        ),
    ],
)
def test_each_mixture_family_exercises_declared_fit_hyperparameter_gradient(factory):
    gradient = jax.grad(
        lambda regularization: (
            factory(regularization)
            .fit_batch(MLBatch(_DATA))
            .as_trainable()
            .log_prob(jnp.array([0.25, -0.1]))
        )
    )(jnp.asarray(1e-3))
    assert jnp.isfinite(gradient)


def test_mixture_prediction_parameter_gradient_is_finite():
    model = _gaussian("full").fit_batch(MLBatch(_DATA)).as_trainable()
    point = jnp.array([0.25, -0.1])
    gradient = jax.grad(
        lambda means: GaussianMixtureModel(
            model.mixing_weights,
            means,
            model.covariance,
            model.precision,
            model.log_determinant,
            covariance_type=model.covariance_type,
        )(point)[0]
    )(model.means)

    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.any(gradient != 0.0)


def test_mixture_supports_complex_features_with_hermitian_spd_geometry():
    features = jnp.array(
        [
            [1.0 + 1.0j, 0.0 + 0.5j],
            [2.0 + 0.5j, 1.0 - 0.5j],
            [0.5 + 1.5j, -1.0 + 0.2j],
        ]
    )
    result = GaussianMixture(
        1,
        initialization="first",
        max_iterations=4,
        tolerance=1e6,
        regularization=1e-3,
    ).fit_batch(MLBatch(features))
    model = result.as_trainable()

    assert result.status == ML_SUCCESS
    assert jnp.issubdtype(model.means.dtype, jnp.complexfloating)
    assert jnp.allclose(
        model.covariance,
        jnp.conj(jnp.swapaxes(model.covariance, -1, -2)),
        atol=1e-6,
    )
    assert jnp.all(jnp.linalg.eigvalsh(model.covariance) > 0.0)
    assert jnp.all(jnp.isreal(model.log_prob(features)))
    assert jnp.all(jnp.isfinite(model.log_prob(features)))


def test_mixture_reports_empty_singleton_constant_nonfinite_and_nonconvergence():
    empty = GaussianMixture(
        2,
        initialization="first",
        max_iterations=2,
        tolerance=1e6,
    ).fit_batch(MLBatch(_DATA[:3], sample_mask=jnp.zeros(3, dtype=bool)))
    singleton = GaussianMixture(
        1,
        initialization="first",
        max_iterations=2,
        tolerance=1e6,
    ).fit_batch(MLBatch(jnp.array([[4.0, -2.0]])))
    constant = GaussianMixture(
        2,
        initialization="first",
        max_iterations=2,
        tolerance=1e6,
    ).fit_batch(MLBatch(jnp.ones((4, 2))))
    nonfinite = GaussianMixture(
        1,
        initialization="first",
        max_iterations=2,
        tolerance=1e6,
    ).fit_batch(MLBatch(_DATA.at[3, 0].set(jnp.nan)))
    nonconverged = GaussianMixture(
        2,
        initialization="first",
        max_iterations=1,
        tolerance=0.0,
    ).fit_batch(MLBatch(_DATA))

    assert empty.status == ML_INSUFFICIENT_DATA
    assert empty.diagnostics.empty_components_seen
    assert singleton.status == ML_SUCCESS
    assert singleton.diagnostics.singular_components_seen
    assert constant.status == ML_SUCCESS
    assert constant.diagnostics.singular_components_seen
    assert nonfinite.status == ML_NONFINITE
    assert nonconverged.status == ML_NONCONVERGED
    assert not nonconverged.diagnostics.converged


def test_empty_component_error_and_capacity_failures_are_explicit():
    features = jnp.array([[1000.0], [-1.0], [1.0]])
    empty_component = GaussianMixture(
        2,
        initialization="first",
        empty_policy="error",
        max_iterations=2,
        tolerance=1e6,
        regularization=1e-3,
    ).fit_batch(MLBatch(features, sample_mask=jnp.array([False, True, True])))

    assert empty_component.status == ML_INSUFFICIENT_DATA
    assert empty_component.diagnostics.empty_components_seen

    with pytest.raises(ValueError, match="sample capacity"):
        GaussianMixture(4, initialization="first").fit_batch(MLBatch(jnp.ones((3, 1))))


def test_case_bound_mixture_rejects_wrong_case_and_feature_shapes():
    cases = jnp.stack((_DATA, _DATA + jnp.array([10.0, -2.0])))
    model = _gaussian().fit_batch(MLBatch(cases)).as_trainable()

    with pytest.raises(ValueError, match="case"):
        model(jnp.array([0.0, 1.0]))
    with pytest.raises(ValueError, match="feature"):
        model(jnp.zeros((2, 3)))
