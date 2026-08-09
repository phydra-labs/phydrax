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
    MLBatch,
)
from phydrax.ml.covariance import (
    CovarianceModel,
    DiagonalCovariance,
    EmpiricalCovariance,
    FactorCovariance,
    GraphicalLasso,
    LedoitWolfCovariance,
    OASCovariance,
    RobustCovariance,
    StreamingGaussianMoments,
    WeightedCovariance,
)


_DATA = jnp.array(
    [
        [-2.0, 0.2, 1.0],
        [-1.0, 1.1, -0.4],
        [0.2, -0.7, 0.8],
        [1.1, 0.4, -1.2],
        [2.2, -1.0, 0.3],
        [3.0, 1.3, 1.7],
    ]
)


@pytest.mark.parametrize(
    "recipe",
    [
        EmpiricalCovariance(regularization=1e-4),
        WeightedCovariance(regularization=1e-4),
        DiagonalCovariance(regularization=1e-4),
        FactorCovariance(2, regularization=1e-4),
        LedoitWolfCovariance(regularization=1e-4),
        OASCovariance(regularization=1e-4),
        RobustCovariance(max_iterations=8, tolerance=1.0, regularization=1e-4),
        GraphicalLasso(max_iterations=8, tolerance=1.0, regularization=1e-4),
    ],
)
def test_covariance_recipe_families_produce_immutable_spd_geometry(recipe):
    result = recipe.fit_batch(MLBatch(_DATA, sample_weight=jnp.arange(1.0, 7.0)))
    model = result.as_trainable()
    eigenvalues = jnp.linalg.eigvalsh(model.covariance)

    assert isinstance(model, CovarianceModel)
    assert model.mean.shape == (3,)
    assert model.covariance.shape == (3, 3)
    assert model.precision.shape == (3, 3)
    assert jnp.all(eigenvalues > 0.0)
    assert jnp.allclose(model.covariance, model.covariance.T, atol=2e-5)
    assert jnp.isfinite(model.log_density(jnp.array([0.1, -0.2, 0.3])))
    assert result.diagnostics.effective_samples > 0.0
    assert result.gradient_contract.fit_mode in ("direct", "unrolled")
    assert result.gradient_contract.prediction_inputs == "smooth"


def test_weighted_covariance_uses_effective_denominator_masks_and_case_axes():
    features = jnp.stack((_DATA, 2.0 * _DATA), axis=0)
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 2, 1].set(False)
    sample_weight = jnp.array([1.0, 2.0, 50.0, 2.0, 1.0, 1.0])
    result = WeightedCovariance(correction=1.0, regularization=0.0).fit_batch(
        MLBatch(
            features,
            feature_mask=feature_mask,
            sample_mask=jnp.array([True, True, True, True, True, False]),
            sample_weight=sample_weight,
        )
    )
    model = result.as_trainable()
    retained = _DATA[jnp.array([0, 1, 3, 4])]
    weights = jnp.array([1.0, 2.0, 2.0, 1.0])
    expected_mean = jnp.sum(weights[:, None] * retained, axis=0) / jnp.sum(weights)

    assert model.mean.shape == (2, 3)
    assert jnp.allclose(model.mean[0], expected_mean)
    assert jnp.allclose(model.mean[1], 2.0 * expected_mean)
    assert model(features).shape == (2, 6)
    assert result.diagnostics.effective_samples.shape == (2,)


def test_covariance_parameterizations_are_exactly_diagonal_low_rank_and_sparse():
    diagonal = (
        DiagonalCovariance(regularization=1e-5).fit_batch(MLBatch(_DATA)).as_trainable()
    )
    factor = (
        FactorCovariance(1, regularization=1e-5).fit_batch(MLBatch(_DATA)).as_trainable()
    )
    graphical = (
        GraphicalLasso(
            penalty=10.0, max_iterations=12, tolerance=1.0, regularization=1e-4
        )
        .fit_batch(MLBatch(_DATA))
        .as_trainable()
    )

    assert jnp.allclose(diagonal.covariance, jnp.diag(jnp.diag(diagonal.covariance)))
    assert factor.factor_loadings.shape == (3, 1)
    reconstructed = factor.factor_loadings @ factor.factor_loadings.T + jnp.diag(
        factor.diagonal
    )
    regularization_residual = factor.covariance - reconstructed
    assert jnp.allclose(
        regularization_residual,
        jnp.diag(jnp.diag(regularization_residual)),
        atol=3e-5,
    )
    assert jnp.all(jnp.diag(regularization_residual) > 0.0)
    graphical_off_diagonal = graphical.precision - jnp.diag(jnp.diag(graphical.precision))
    assert jnp.count_nonzero(jnp.abs(graphical_off_diagonal) < 1e-7) > 3


def test_complex_covariance_is_hermitian_and_uses_proper_complex_likelihood():
    complex_data = _DATA.astype(jnp.complex64) + 1j * jnp.flip(_DATA, axis=-1)
    model = (
        EmpiricalCovariance(regularization=1e-4)
        .fit_batch(MLBatch(complex_data))
        .as_trainable()
    )
    point = complex_data[0]
    expected = -(model(point) + 3 * jnp.log(jnp.pi) + model.log_determinant)

    assert jnp.allclose(model.covariance, jnp.conj(model.covariance.T), atol=2e-5)
    assert jnp.allclose(model.log_density(point), expected)
    assert jnp.all(jnp.isfinite(model.whiten(complex_data)))


def test_covariance_prediction_and_declared_fit_gradients_are_finite_and_jittable():
    point = jnp.array([0.3, -0.2, 0.7])
    recipe = EmpiricalCovariance(regularization=1e-3)
    model = recipe.fit_batch(MLBatch(_DATA)).as_trainable()

    prediction_gradient = jax.grad(model.log_density)(point)
    feature_gradient = jax.grad(
        lambda values: recipe.fit_batch(MLBatch(values)).as_trainable().log_density(point)
    )(_DATA)
    weight_gradient = jax.grad(
        lambda weights: (
            recipe.fit_batch(MLBatch(_DATA, sample_weight=weights))
            .as_trainable()
            .log_density(point)
        )
    )(jnp.arange(1.0, 7.0))

    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jax.jit(model.log_density)(point).shape == ()

    parameter_gradient = jax.grad(
        lambda mean: CovarianceModel(
            mean,
            model.covariance,
            model.precision,
            model.log_determinant,
            method="gradient-probe",
        ).log_density(point)
    )(model.mean)
    assert jnp.all(jnp.isfinite(parameter_gradient))
    assert jax.vmap(model.log_density)(_DATA).shape == (6,)


@pytest.mark.parametrize(
    "recipe",
    [
        DiagonalCovariance(regularization=1e-3),
        WeightedCovariance(regularization=1e-3),
        FactorCovariance(2, regularization=1e-3),
        LedoitWolfCovariance(regularization=1e-3),
        OASCovariance(regularization=1e-3),
        RobustCovariance(max_iterations=3, tolerance=1.0, regularization=1e-3),
        GraphicalLasso(max_iterations=3, tolerance=1.0, regularization=1e-3),
    ],
)
def test_each_declared_covariance_fit_gradient_is_finite(recipe):
    point = jnp.array([0.1, 0.2, -0.3])
    feature_gradient = jax.grad(
        lambda values: recipe.fit_batch(MLBatch(values)).as_trainable().log_density(point)
    )(_DATA)
    weight_gradient = jax.grad(
        lambda weights: (
            recipe.fit_batch(MLBatch(_DATA, sample_weight=weights))
            .as_trainable()
            .log_density(point)
        )
    )(jnp.arange(1.0, 7.0))
    contract = recipe.fit_batch(MLBatch(_DATA)).gradient_contract

    assert contract.fit_features == "conditional"
    assert contract.fit_weights == "conditional"
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


@pytest.mark.parametrize(
    "factory",
    [
        lambda value: EmpiricalCovariance(regularization=value),
        lambda value: WeightedCovariance(regularization=value),
        lambda value: DiagonalCovariance(regularization=value),
        lambda value: FactorCovariance(2, regularization=value),
        lambda value: LedoitWolfCovariance(regularization=value),
        lambda value: OASCovariance(regularization=value),
        lambda value: RobustCovariance(
            max_iterations=3, tolerance=1.0, regularization=value
        ),
        lambda value: GraphicalLasso(
            max_iterations=3, tolerance=1.0, regularization=value
        ),
    ],
)
def test_each_declared_covariance_fit_hyperparameter_gradient_is_finite(factory):
    point = jnp.array([0.1, -0.3, 0.2])
    gradient = jax.grad(
        lambda regularization: (
            factory(regularization)
            .fit_batch(MLBatch(_DATA))
            .as_trainable()
            .log_density(point)
        )
    )(jnp.asarray(1e-3))
    assert jnp.isfinite(gradient)


def test_covariance_reports_nonfinite_underfull_and_constant_degeneracy():
    nonfinite = _DATA.at[1, 0].set(jnp.nan)
    nonfinite_result = EmpiricalCovariance().fit_batch(MLBatch(nonfinite))
    underfull = WeightedCovariance(correction=1.0).fit_batch(
        MLBatch(_DATA, sample_mask=jnp.array([True, False, False, False, False, False]))
    )
    constant = EmpiricalCovariance(regularization=1e-5).fit_batch(
        MLBatch(jnp.ones((4, 3)))
    )

    assert nonfinite_result.status == ML_NONFINITE
    assert underfull.status == ML_INSUFFICIENT_DATA
    robust_nonconverged = RobustCovariance(
        max_iterations=1, tolerance=0.0, regularization=1e-4
    ).fit_batch(MLBatch(_DATA))
    graphical_nonconverged = GraphicalLasso(
        max_iterations=1, tolerance=0.0, regularization=1e-4
    ).fit_batch(MLBatch(_DATA))
    assert robust_nonconverged.status == ML_NONCONVERGED
    assert graphical_nonconverged.status == ML_NONCONVERGED
    assert constant.diagnostics.rank < 3
    assert jnp.all(jnp.linalg.eigvalsh(constant.as_trainable().covariance) > 0.0)


def test_streaming_gaussian_moments_update_merge_and_model_match_batch_fit():
    empty = StreamingGaussianMoments.initialize(3)
    first = empty.update(_DATA[:3], weights=jnp.array([1.0, 2.0, 3.0]))
    second = StreamingGaussianMoments.initialize(3).update(
        _DATA[3:], weights=jnp.array([4.0, 5.0, 6.0])
    )
    merged = first.merge(second)
    batch = (
        WeightedCovariance(correction=1.0, regularization=1e-4)
        .fit_batch(MLBatch(_DATA, sample_weight=jnp.arange(1.0, 7.0)))
        .as_trainable()
    )
    streamed = merged.model(correction=1.0, regularization=1e-4)

    assert empty.mass == 0.0
    assert first.mass == 6.0
    assert merged.updates == 2
    assert jnp.allclose(merged.mean, batch.mean, atol=2e-6)
    assert jnp.allclose(streamed.covariance, batch.covariance, atol=3e-5)
    assert jnp.allclose(
        first.mean,
        StreamingGaussianMoments.initialize(3)
        .update(_DATA[:3], weights=jnp.array([1.0, 2.0, 3.0]))
        .mean,
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
def test_covariance_preserves_real_precision_and_is_key_deterministic(dtype):
    features = _DATA.astype(dtype)
    recipe = EmpiricalCovariance(regularization=1e-4)
    first = recipe.fit_batch(MLBatch(features), key=jax.random.key(1)).as_trainable()
    second = recipe.fit_batch(MLBatch(features), key=jax.random.key(2)).as_trainable()

    assert first.mean.dtype == features.dtype
    assert first.covariance.dtype == features.dtype
    assert jnp.array_equal(first.mean, second.mean)
    assert jnp.array_equal(first.covariance, second.covariance)


def test_covariance_fit_is_jittable_and_vmappable_over_independent_datasets():
    recipe = EmpiricalCovariance(regularization=1e-3)
    covariance = lambda values: (
        recipe.fit_batch(MLBatch(values)).as_trainable().covariance
    )
    cases = jnp.stack((_DATA, 2.0 * _DATA))

    assert jax.jit(covariance)(_DATA).shape == (3, 3)
    assert jax.vmap(covariance)(cases).shape == (2, 3, 3)


def test_covariance_configuration_fails_closed():
    with pytest.raises(ValueError):
        FactorCovariance(0)
    with pytest.raises(ValueError):
        WeightedCovariance(weight_policy="none")
    with pytest.raises(ValueError):
        GraphicalLasso(regularization=0.0)
