#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.kernels import FiniteFeatureKernel, SquaredExponentialKernel
from phydrax.ml import MLBatch
from phydrax.ml.kernel_methods import (
    BernoulliGaussianProcessClassifierRecipe,
    CategoricalGaussianProcessClassifierRecipe,
    KernelPCARecipe,
    KernelRidgeRecipe,
    LeastSquaresSVMRecipe,
    NystromRecipe,
    OneClassSVMRecipe,
    RandomFourierFeaturesRecipe,
    SupportVectorClassifierRecipe,
    SupportVectorRegressorRecipe,
)
from phydrax.uq import (
    ExactGaussianProcessFactor,
    FiniteFeatureGaussianProcessFactor,
    GaussianProcessLikelihoodState,
)


def _regression_data():
    x = jnp.array([[-1.5, 0.0], [-0.5, 1.0], [0.2, -0.4], [1.0, 0.5], [1.8, -0.8]])
    y = 0.7 * x[:, 0] - 0.3 * x[:, 1]
    return x, y


def test_kernel_ridge_preserves_cases_masks_weights_complex_outputs_and_gradients():
    x, y = _regression_data()
    features = jnp.stack((x, 1.2 * x), axis=0)
    targets = jnp.stack((y + 0.2j * y, 2.0 * y - 0.1j * y), axis=0)
    batch = MLBatch(
        features,
        targets,
        sample_mask=jnp.array([True, True, False, True, True]),
        sample_weight=jnp.array([1.0, 2.0, 4.0, 0.5, 1.0]),
    )
    recipe = KernelRidgeRecipe(SquaredExponentialKernel(length_scale=0.8), alpha=0.2)
    result = recipe.fit_batch(batch)
    model = result.as_trainable()
    query = features[:, :3]

    prediction = model(query)
    assert prediction.shape == (2, 3)
    assert jnp.iscomplexobj(prediction)
    assert result.diagnostics.effective_samples.shape == (2,)
    assert result.gradient_contract.fit_mode == "direct"
    assert jax.jit(model)(query).shape == prediction.shape

    point_gradient = jax.grad(
        lambda point: jnp.real(jnp.sum(jnp.abs(model(point)) ** 2))
    )(features[:, 0])
    target_gradient = jax.grad(
        lambda value: jnp.real(
            jnp.sum(
                jnp.abs(
                    KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.3)
                    .fit_batch(MLBatch(x, value))
                    .as_trainable()(x[:2])
                )
                ** 2
            )
        )
    )(y)
    weight_gradient = jax.grad(
        lambda weight: jnp.sum(
            KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.3)
            .fit_batch(MLBatch(x, y, sample_weight=weight))
            .as_trainable()(x[:2])
        )
    )(jnp.ones((x.shape[0],)))
    alpha_gradient = jax.grad(
        lambda alpha: jnp.sum(
            KernelRidgeRecipe(SquaredExponentialKernel(), alpha=alpha)
            .fit_batch(MLBatch(x, y))
            .as_trainable()(x[:2])
        )
    )(jnp.array(0.3))
    assert jnp.all(jnp.isfinite(point_gradient))
    assert jnp.all(jnp.isfinite(target_gradient))
    assert jnp.all(jnp.isfinite(weight_gradient))
    assert jnp.isfinite(alpha_gradient)


def test_callable_kernel_supports_complex_coordinates_but_native_kernel_fails_closed():
    x = jnp.array([[1.0 + 0.2j, 0.0], [0.5 - 0.1j, 1.0], [-0.2 + 0.3j, 0.4]])
    y = jnp.array([1.0 + 0.5j, -0.2j, 0.7 - 0.1j])
    hermitian_kernel = lambda left, right: jnp.vdot(left, right) + 1.0

    result = KernelRidgeRecipe(hermitian_kernel, alpha=0.1).fit_batch(MLBatch(x, y))
    assert jnp.all(jnp.isfinite(result.as_trainable()(x)))
    with pytest.raises(TypeError, match="real coordinates"):
        KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.1).fit_batch(MLBatch(x, y))


def test_ls_svm_svc_svr_and_one_class_expose_smooth_and_hard_contracts():
    x, y = _regression_data()
    labels = (y > 0).astype(jnp.int32)
    kernel = SquaredExponentialKernel(length_scale=1.1)

    ls_result = LeastSquaresSVMRecipe(kernel, alpha=0.2).fit_batch(MLBatch(x, labels))
    svc_result = SupportVectorClassifierRecipe(
        kernel, iterations=8, learning_rate=0.03
    ).fit_batch(MLBatch(x, labels))
    svr_result = SupportVectorRegressorRecipe(
        kernel, iterations=8, learning_rate=0.01
    ).fit_batch(MLBatch(x, y))
    one_result = OneClassSVMRecipe(kernel, iterations=8).fit_batch(MLBatch(x))

    for result in (ls_result, svc_result):
        model = result.as_trainable()
        assert model(x[:2]).shape == (2,)
        assert model.probabilities(x[:2]).shape == (2, 2)
        assert model.predict(x[:2]).dtype == jnp.int32
        assert "predict" in result.gradient_contract.nondifferentiable_outputs
    assert svr_result.as_trainable()(x[:2]).shape == (2,)
    assert one_result.as_trainable().inlier_probability(x[:2]).shape == (2,)
    assert one_result.as_trainable().predict(x[:2]).shape == (2,)
    assert svc_result.diagnostics.iterations == 8
    assert svr_result.diagnostics.iterations == 8
    assert one_result.diagnostics.iterations == 8

    svc_input_gradient = jax.grad(
        lambda point: jnp.sum(svc_result.as_trainable().decision_function(point))
    )(x[0])
    svr_target_gradient = jax.grad(
        lambda target: jnp.sum(
            SupportVectorRegressorRecipe(kernel, iterations=3)
            .fit_batch(MLBatch(x, target))
            .as_trainable()(x[:1])
        )
    )(y)
    assert jnp.all(jnp.isfinite(svc_input_gradient))
    assert jnp.all(jnp.isfinite(svr_target_gradient))


def test_kernel_pca_nystrom_and_random_features_have_key_geometry_and_jit_contracts():
    x, _ = _regression_data()
    kernel = SquaredExponentialKernel(length_scale=jnp.array([0.8, 1.3]))
    batch = MLBatch(x, sample_mask=jnp.array([True, True, True, False, True]))

    kpca = KernelPCARecipe(kernel, n_components=2).fit_batch(batch)
    kpca_model = kpca.as_trainable()
    assert kpca_model(x[:3]).shape == (3, 2)
    assert kpca.diagnostics.rank.shape == ()
    assert kpca.gradient_contract.fit_mode == "spectral"
    assert jnp.all(
        jnp.isfinite(jax.grad(lambda point: jnp.sum(kpca_model(point) ** 2))(x[0]))
    )

    with pytest.raises(ValueError, match="explicit JAX key"):
        NystromRecipe(kernel, n_components=3, selection="random").fit_batch(batch)
    nystrom_a = NystromRecipe(kernel, n_components=3, selection="random").fit_batch(
        batch, key=jax.random.key(3)
    )
    nystrom_b = NystromRecipe(kernel, n_components=3, selection="random").fit_batch(
        batch, key=jax.random.key(3)
    )
    assert jnp.allclose(
        nystrom_a.as_trainable().landmarks, nystrom_b.as_trainable().landmarks
    )
    assert nystrom_a.as_trainable()(x[:2]).shape == (2, 3)
    assert nystrom_a.as_trainable().as_kernel().matrix(x[:2], x[:2]).shape == (2, 2)

    with pytest.raises(ValueError, match="explicit JAX key"):
        RandomFourierFeaturesRecipe(kernel, n_components=8).fit_batch(batch)
    rff_a = RandomFourierFeaturesRecipe(kernel, n_components=8).fit_batch(
        batch, key=jax.random.key(4)
    )
    rff_b = RandomFourierFeaturesRecipe(kernel, n_components=8).fit_batch(
        batch, key=jax.random.key(4)
    )
    rff_model = rff_a.as_trainable()
    assert jnp.allclose(rff_model.frequencies, rff_b.as_trainable().frequencies)
    assert jax.jit(rff_model)(x[:2]).shape == (2, 8)
    assert jax.vmap(rff_model)(x[:2]).shape == (2, 8)
    assert rff_model.as_kernel().matrix(x[:2], x[:2]).shape == (2, 2)


def test_gp_classification_reuses_exact_and_finite_uq_factor_geometry():
    x, y = _regression_data()
    labels = (y > 0).astype(jnp.int32)
    exact_state = GaussianProcessLikelihoodState(
        kernel=SquaredExponentialKernel(length_scale=0.9), noise_scale=0.0, jitter=1e-5
    )
    exact_result = BernoulliGaussianProcessClassifierRecipe(
        exact_state, iterations=4
    ).fit_batch(
        MLBatch(x, labels, sample_mask=jnp.array([True, True, False, True, True]))
    )
    exact_model = exact_result.as_trainable()
    exact_posterior = exact_model.posteriors[0]
    assert isinstance(exact_posterior.factor, ExactGaussianProcessFactor)
    assert exact_posterior.factor.cholesky.shape == (x.shape[0], x.shape[0])
    probability = exact_model.probabilities(x[:3])
    assert probability.shape == (3, 2)
    assert jnp.allclose(jnp.sum(probability, axis=-1), 1.0, atol=1e-5)
    assert jnp.all(jnp.isfinite(jax.grad(lambda point: exact_model(point)[1])(x[0])))

    finite_kernel = FiniteFeatureKernel(
        lambda point: jnp.array([1.0, point[0], point[1]]),
        jnp.eye(3),
        feature_map_id="classification-test",
        max_derivative_order=None,
    )
    finite_state = GaussianProcessLikelihoodState(
        kernel=finite_kernel, noise_scale=0.0, jitter=1e-5
    )
    finite_result = BernoulliGaussianProcessClassifierRecipe(
        finite_state, iterations=4
    ).fit_batch(MLBatch(x, labels))
    finite_posterior = finite_result.as_trainable().posteriors[0]
    assert isinstance(finite_posterior.factor, FiniteFeatureGaussianProcessFactor)
    assert finite_posterior.factor.correction_cholesky.shape == (3, 3)

    three_labels = jnp.array([0, 1, 2, 1, 0], dtype=jnp.int32)
    categorical = CategoricalGaussianProcessClassifierRecipe(
        exact_state, class_count=3, iterations=3
    ).fit_batch(MLBatch(x, three_labels))
    categorical_probability = categorical.as_trainable()(x[:2])
    assert categorical_probability.shape == (2, 3)
    assert jnp.allclose(jnp.sum(categorical_probability, axis=-1), 1.0, atol=1e-5)
    assert categorical.as_trainable().predict(x[:2]).dtype == jnp.int32
    assert categorical.gradient_contract.fit_mode == "unrolled"
