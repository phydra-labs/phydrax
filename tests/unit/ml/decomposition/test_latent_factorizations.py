#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_NONCONVERGED, MLBatch
from phydrax.ml.decomposition import (
    CCA,
    DictionaryLearning,
    FactorAnalysis,
    ICA,
    NMF,
    PLS,
    SparseCoding,
)


def _supervised_data():
    x = jnp.array(
        [
            [-2.0, 0.1, 1.0],
            [-1.4, 1.2, 0.4],
            [-0.8, -1.1, 0.8],
            [-0.1, 0.4, -0.5],
            [0.5, -0.7, 1.4],
            [1.0, 1.3, -1.0],
            [1.7, -0.2, 0.1],
            [2.4, 0.8, -0.9],
        ]
    )
    mapping = jnp.array([[1.5, -0.3], [-0.4, 1.2], [0.7, 0.5]])
    y = x @ mapping + jnp.array([0.2, -0.1])
    return x, y


def test_cca_weighted_masked_scores_inverse_complex_policy_and_gradients():
    x, y = _supervised_data()
    feature_mask = jnp.ones_like(x, dtype=bool).at[2, 2].set(False)
    target_mask = jnp.ones_like(y, dtype=bool).at[4, 1].set(False)
    weights = jnp.arange(1.0, 9.0)
    result = CCA(2, regularization=1e-4).fit_batch(
        MLBatch(
            x,
            y,
            feature_mask=feature_mask,
            target_mask=target_mask,
            sample_weight=weights,
        )
    )
    model = result.as_trainable()

    assert model.transform(x).shape == (8, 2)
    assert model.transform_targets(y).shape == (8, 2)
    assert model.inverse_transform(model.transform(x)).shape == x.shape
    assert model.predict_targets(x).shape == y.shape
    assert jnp.all(result.diagnostics.singular_values >= 0.0)
    assert result.diagnostics.minimum_eigengap.shape == ()
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_targets == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    assert jax.jit(model.transform)(x).shape == (8, 2)
    assert jax.vmap(model.transform)(x).shape == (8, 2)

    point_gradient = jax.grad(lambda value: jnp.sum(jnp.square(model(value))))(x[0])

    def fit_loss(features, targets, sample_weight):
        fitted = (
            CCA(2, regularization=1e-3)
            .fit_batch(MLBatch(features, targets, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted.transform(features[:2])))

    feature_gradient, target_gradient, weight_gradient = jax.grad(
        fit_loss, argnums=(0, 1, 2)
    )(x, y, weights)
    assert jnp.all(jnp.isfinite(point_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(target_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))

    complex_result = CCA(1, regularization=1e-3).fit_batch(
        MLBatch(x.astype(jnp.complex64) * (1.0 + 0.2j), y.astype(jnp.complex64))
    )
    assert jnp.iscomplexobj(complex_result.as_trainable().x_rotations)


def test_pls_prediction_inverse_jit_and_declared_fit_gradients():
    x, y = _supervised_data()
    weights = jnp.linspace(0.5, 2.0, x.shape[0])
    result = PLS(2).fit_batch(MLBatch(x, y, sample_weight=weights))
    model = result.as_trainable()

    assert model.transform(x).shape == (8, 2)
    assert model.inverse_transform(model.transform(x)).shape == x.shape
    assert model.predict(x).shape == y.shape
    assert jax.jit(model.predict)(x).shape == y.shape
    assert jnp.all(jnp.isfinite(model.predict(x)))
    complex_pls = PLS(1).fit_batch(
        MLBatch(
            x.astype(jnp.complex64) * (1.0 + 0.1j),
            y.astype(jnp.complex64) * (1.0 - 0.2j),
        )
    )
    assert jnp.iscomplexobj(complex_pls.as_trainable().x_weights)

    def loss(features, targets, sample_weight):
        fitted = (
            PLS(2)
            .fit_batch(MLBatch(features, targets, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted.predict(features[:2])))

    gradients = jax.grad(loss, argnums=(0, 1, 2))(x, y, weights)
    assert all(jnp.all(jnp.isfinite(value)) for value in gradients)
    parameter_gradient = jax.grad(
        lambda decoder: jnp.sum(
            PLS(2).fit_batch(MLBatch(x, y)).as_trainable().transform(x[:2]) @ decoder
        )
    )(model.y_loadings)
    assert jnp.all(jnp.isfinite(parameter_gradient))


def test_factor_analysis_affine_inverse_case_axes_and_unrolled_fit_gradients():
    x, _ = _supervised_data()
    cases = jnp.stack((x, 1.5 * x + 0.2), axis=0)
    result = FactorAnalysis(
        2, max_iterations=32, tolerance=1e3, min_noise=1e-5
    ).fit_batch(MLBatch(cases, sample_weight=jnp.linspace(1.0, 2.0, x.shape[0])))
    model = result.as_trainable()

    scores = model.transform(cases)
    assert scores.shape == (2, 8, 2)
    assert model.inverse_transform(scores).shape == cases.shape
    assert result.diagnostics.iterations.shape == (2,)
    assert result.diagnostics.converged.shape == (2,)
    assert result.diagnostics.weighted_orthogonality_error.shape == (2,)
    assert jax.jit(model.transform)(cases).shape == scores.shape

    prediction_gradient = jax.grad(
        lambda value: jnp.sum(jnp.square(model.transform(value)))
    )(cases)

    def fit_loss(features, sample_weight):
        fitted = (
            FactorAnalysis(2, max_iterations=4, tolerance=1e3, min_noise=1e-4)
            .fit_batch(MLBatch(features, sample_weight=sample_weight))
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted.transform(features[:2])))

    feature_gradient, weight_gradient = jax.grad(fit_loss, argnums=(0, 1))(
        x, jnp.ones((x.shape[0],))
    )
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    complex_factor = FactorAnalysis(1, max_iterations=3, tolerance=1e3).fit_batch(
        MLBatch(x.astype(jnp.complex64) * (1.0 + 0.2j))
    )
    assert jnp.iscomplexobj(complex_factor.as_trainable().loadings)


def test_ica_requires_key_rejects_complex_reports_nonconvergence_and_differentiates():
    x, _ = _supervised_data()
    nonlinear = jnp.stack((x[:, 0], x[:, 1] ** 3, jnp.tanh(x[:, 2])), axis=-1)
    with pytest.raises(ValueError, match="explicit JAX key"):
        ICA(2).fit_batch(MLBatch(nonlinear))
    with pytest.raises(TypeError, match="complex ICA"):
        ICA(2).fit_batch(MLBatch(nonlinear.astype(jnp.complex64)), key=jax.random.key(0))

    nonconverged = ICA(2, max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(nonlinear), key=jax.random.key(0)
    )
    assert nonconverged.status == ML_NONCONVERGED
    result = ICA(2, max_iterations=16, tolerance=1e3).fit_batch(
        MLBatch(nonlinear), key=jax.random.key(1)
    )
    model = result.as_trainable()
    assert model.transform(nonlinear).shape == (8, 2)
    assert model.inverse_transform(model.transform(nonlinear)).shape == nonlinear.shape
    assert jax.jit(model.transform)(nonlinear).shape == (8, 2)

    prediction_gradient = jax.grad(
        lambda point: jnp.sum(jnp.square(model.transform(point)))
    )(nonlinear[0])

    def fit_loss(features, sample_weight):
        fitted = (
            ICA(2, max_iterations=3, tolerance=1e3)
            .fit_batch(
                MLBatch(features, sample_weight=sample_weight), key=jax.random.key(2)
            )
            .as_trainable()
        )
        return jnp.sum(jnp.square(fitted.transform(features[:2])))

    feature_gradient, weight_gradient = jax.grad(fit_loss, argnums=(0, 1))(
        nonlinear, jnp.ones((nonlinear.shape[0],))
    )
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def _nonnegative_data():
    codes = jnp.array(
        [[1.0, 0.1], [0.8, 0.3], [0.4, 1.0], [0.2, 1.2], [1.1, 0.5], [0.6, 0.9]]
    )
    atoms = jnp.array([[1.0, 0.2, 0.5], [0.1, 1.0, 0.4]])
    return codes @ atoms


def test_nmf_key_mask_nonconvergence_inverse_jit_and_unrolled_gradients():
    values = _nonnegative_data()
    mask = jnp.ones_like(values, dtype=bool).at[1, 2].set(False)
    with pytest.raises(ValueError, match="explicit JAX key"):
        NMF(2).fit_batch(MLBatch(values))
    with pytest.raises(TypeError, match="real nonnegative"):
        NMF(2).fit_batch(MLBatch(values.astype(jnp.complex64)), key=jax.random.key(0))

    nonconverged = NMF(2, max_iterations=1, tolerance=0.0).fit_batch(
        MLBatch(values), key=jax.random.key(0)
    )
    assert nonconverged.status == ML_NONCONVERGED
    result = NMF(2, max_iterations=12, transform_iterations=8, tolerance=1e6).fit_batch(
        MLBatch(values, feature_mask=mask, sample_weight=jnp.arange(1.0, 7.0)),
        key=jax.random.key(1),
    )
    model = result.as_trainable()
    scores = model.transform(values)
    assert scores.shape == (6, 2)
    assert jnp.all(scores >= 0.0)
    assert model.inverse_transform(scores).shape == values.shape
    assert jax.jit(model.transform)(values).shape == scores.shape
    assert result.diagnostics.mask_provenance == "zero-extension"

    prediction_gradient = jax.grad(lambda point: jnp.sum(model.transform(point)))(
        values[0]
    )

    def fit_loss(features, sample_weight):
        fitted = (
            NMF(2, max_iterations=3, transform_iterations=2, tolerance=1e6)
            .fit_batch(
                MLBatch(features, sample_weight=sample_weight), key=jax.random.key(3)
            )
            .as_trainable()
        )
        return jnp.sum(fitted.transform(features[:2]))

    feature_gradient, weight_gradient = jax.grad(fit_loss, argnums=(0, 1))(
        values, jnp.ones((values.shape[0],))
    )
    assert jnp.all(jnp.isfinite(prediction_gradient))
    assert jnp.all(jnp.isfinite(feature_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))


def test_sparse_coding_and_dictionary_learning_masks_complex_keys_and_gradients():
    values = _nonnegative_data()
    dictionary = jnp.array([[1.0, 0.2, 0.5], [0.1, 1.0, 0.4]])
    mask = jnp.ones_like(values, dtype=bool).at[0, 1].set(False)
    sparse = SparseCoding(
        dictionary, regularization=1e-3, transform_iterations=12
    ).fit_batch(MLBatch(values, feature_mask=mask))
    sparse_model = sparse.as_trainable()
    scores = sparse_model.transform(values)
    assert scores.shape == (6, 2)
    assert sparse_model.inverse_transform(scores).shape == values.shape
    assert jax.jit(sparse_model.transform)(values).shape == scores.shape
    assert "active_set" in sparse.gradient_contract.nondifferentiable_outputs

    complex_dictionary = dictionary.astype(jnp.complex64) * (1.0 + 0.3j)
    complex_values = values.astype(jnp.complex64) * (1.0 - 0.2j)
    complex_sparse = SparseCoding(complex_dictionary, transform_iterations=4).fit_batch(
        MLBatch(complex_values)
    )
    assert jnp.iscomplexobj(complex_sparse.as_trainable().dictionary)

    with pytest.raises(ValueError, match="explicit JAX key"):
        DictionaryLearning(2).fit_batch(MLBatch(values))
    nonconverged = DictionaryLearning(
        2, max_iterations=1, code_iterations=1, transform_iterations=1, tolerance=0.0
    ).fit_batch(MLBatch(values), key=jax.random.key(0))
    assert nonconverged.status == ML_NONCONVERGED
    learned = DictionaryLearning(
        2,
        max_iterations=3,
        code_iterations=2,
        transform_iterations=3,
        tolerance=1e6,
    ).fit_batch(
        MLBatch(values, feature_mask=mask, sample_weight=jnp.arange(1.0, 7.0)),
        key=jax.random.key(1),
    )
    learned_model = learned.as_trainable()
    assert learned_model.transform(values).shape == (6, 2)
    assert jnp.allclose(
        jnp.linalg.norm(learned_model.dictionary, axis=-1), 1.0, atol=1e-5
    )
    complex_learned = DictionaryLearning(
        2,
        max_iterations=2,
        code_iterations=2,
        transform_iterations=2,
        tolerance=1e6,
    ).fit_batch(MLBatch(complex_values), key=jax.random.key(5))
    assert jnp.iscomplexobj(complex_learned.as_trainable().dictionary)

    sparse_input_gradient = jax.grad(
        lambda point: jnp.real(jnp.sum(sparse_model.transform(point)))
    )(values[0])

    def sparse_fit_loss(features):
        fitted = (
            SparseCoding(dictionary, transform_iterations=3)
            .fit_batch(MLBatch(features))
            .as_trainable()
        )
        return jnp.real(jnp.sum(fitted.transform(features[:2])))

    def dictionary_fit_loss(features, sample_weight):
        fitted = (
            DictionaryLearning(
                2,
                max_iterations=2,
                code_iterations=2,
                transform_iterations=2,
                tolerance=1e6,
            )
            .fit_batch(
                MLBatch(features, sample_weight=sample_weight), key=jax.random.key(4)
            )
            .as_trainable()
        )
        return jnp.real(jnp.sum(fitted.transform(features[:2])))

    sparse_fit_gradient = jax.grad(sparse_fit_loss)(values)
    dictionary_feature_gradient, dictionary_weight_gradient = jax.grad(
        dictionary_fit_loss, argnums=(0, 1)
    )(values, jnp.ones((values.shape[0],)))
    assert jnp.all(jnp.isfinite(sparse_input_gradient))
    assert jnp.all(jnp.isfinite(sparse_fit_gradient))
    assert jnp.all(jnp.isfinite(dictionary_feature_gradient))
    assert jnp.all(jnp.isfinite(dictionary_weight_gradient))
