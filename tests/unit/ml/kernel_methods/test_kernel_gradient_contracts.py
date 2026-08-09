#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax.kernels import SquaredExponentialKernel
from phydrax.ml import ML_NONFINITE, MLBatch
from phydrax.ml.kernel_methods import (
    GaussianProcessClassifierRecipe,
    KernelPCARecipe,
    LeastSquaresSVMRecipe,
    NystromRecipe,
    OneClassSVMRecipe,
    RandomFourierFeaturesRecipe,
    SupportVectorClassifierRecipe,
    SupportVectorRegressorRecipe,
)
from phydrax.uq import GaussianProcessLikelihoodState


def _data():
    features = jnp.array(
        [[-1.4, -0.3], [-0.7, 0.8], [0.1, -0.6], [0.8, 0.5], [1.6, -0.2]]
    )
    targets = jnp.array([-1.1, -0.4, 0.2, 0.9, 1.4])
    labels = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
    weights = jnp.array([0.8, 1.2, 0.9, 1.1, 0.7])
    query = jnp.array([[-0.35, 0.15], [1.15, 0.1]])
    return features, targets, labels, weights, query


def _assert_finite(values):
    if isinstance(values, tuple):
        assert all(jnp.all(jnp.isfinite(value)) for value in values)
    else:
        assert jnp.all(jnp.isfinite(values))


def _assert_prediction_parameter_gradient(model, query):
    gradient = eqx.filter_grad(
        lambda current: jnp.sum(jnp.square(jnp.real(current(query))))
    )(model)
    leaves = jax.tree_util.tree_leaves(eqx.filter(gradient, eqx.is_inexact_array))
    assert leaves
    assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)


def test_ls_svm_exercises_every_declared_fit_and_prediction_gradient():
    features, _, labels, weights, query = _data()
    base = LeastSquaresSVMRecipe(SquaredExponentialKernel(length_scale=0.9), alpha=0.25)

    def fit_loss(x, sample_weight, alpha, length_scale):
        recipe = eqx.tree_at(
            lambda current: (current.alpha, current.kernel.length_scale),
            base,
            (alpha, length_scale),
        )
        model = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3))(
        features, weights, base.alpha, base.kernel.length_scale
    )
    _assert_finite(gradients)
    result = base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_targets,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == ("conditional", "none", "conditional", "conditional")
    _assert_finite(jax.grad(lambda point: result.as_trainable()(point) ** 2)(query[0]))
    _assert_prediction_parameter_gradient(result.as_trainable(), query)


def test_projected_svc_and_one_class_exercise_conditional_fit_gradients():
    features, _, labels, weights, query = _data()
    svc_base = SupportVectorClassifierRecipe(
        SquaredExponentialKernel(length_scale=1.1),
        c=0.7,
        iterations=4,
        learning_rate=0.025,
    )

    def svc_loss(x, sample_weight, c, learning_rate, length_scale):
        recipe = eqx.tree_at(
            lambda current: (
                current.c,
                current.learning_rate,
                current.kernel.length_scale,
            ),
            svc_base,
            (c, learning_rate, length_scale),
        )
        model = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    svc_gradients = jax.grad(svc_loss, argnums=(0, 1, 2, 3, 4))(
        features,
        weights,
        svc_base.c,
        svc_base.learning_rate,
        svc_base.kernel.length_scale,
    )
    _assert_finite(svc_gradients)
    svc_result = svc_base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    assert svc_result.gradient_contract.fit_features == "conditional"
    assert svc_result.gradient_contract.fit_weights == "conditional"
    assert svc_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(svc_result.as_trainable(), query)

    one_base = OneClassSVMRecipe(
        SquaredExponentialKernel(length_scale=1.1),
        nu=0.45,
        iterations=4,
        learning_rate=0.025,
    )

    def one_class_loss(x, sample_weight, nu, learning_rate, length_scale):
        recipe = eqx.tree_at(
            lambda current: (
                current.nu,
                current.learning_rate,
                current.kernel.length_scale,
            ),
            one_base,
            (nu, learning_rate, length_scale),
        )
        model = recipe.fit_batch(MLBatch(x, sample_weight=sample_weight)).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    one_gradients = jax.grad(one_class_loss, argnums=(0, 1, 2, 3, 4))(
        features,
        weights,
        one_base.nu,
        one_base.learning_rate,
        one_base.kernel.length_scale,
    )
    _assert_finite(one_gradients)
    one_result = one_base.fit_batch(MLBatch(features, sample_weight=weights))
    assert one_result.gradient_contract.fit_features == "conditional"
    assert one_result.gradient_contract.fit_weights == "conditional"
    assert one_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(one_result.as_trainable(), query)


def test_svr_exercises_almost_everywhere_fit_and_prediction_gradients():
    features, targets, _, weights, query = _data()
    base = SupportVectorRegressorRecipe(
        SquaredExponentialKernel(length_scale=0.85),
        c=0.6,
        epsilon=0.08,
        iterations=4,
        learning_rate=0.01,
    )

    def fit_loss(x, y, sample_weight, c, epsilon, learning_rate, length_scale):
        recipe = eqx.tree_at(
            lambda current: (
                current.c,
                current.epsilon,
                current.learning_rate,
                current.kernel.length_scale,
            ),
            base,
            (c, epsilon, learning_rate, length_scale),
        )
        model = recipe.fit_batch(
            MLBatch(x, y, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3, 4, 5, 6))(
        features,
        targets,
        weights,
        base.c,
        base.epsilon,
        base.learning_rate,
        base.kernel.length_scale,
    )
    _assert_finite(gradients)
    result = base.fit_batch(MLBatch(features, targets, sample_weight=weights))
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_targets,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == (
        "conditional",
        "almost-everywhere",
        "almost-everywhere",
        "almost-everywhere",
    )
    _assert_prediction_parameter_gradient(result.as_trainable(), query)


def test_spectral_and_random_maps_exercise_declared_fit_gradients():
    features, _, _, weights, query = _data()
    kpca_base = KernelPCARecipe(
        SquaredExponentialKernel(length_scale=0.95),
        n_components=2,
        eigenvalue_floor=1e-7,
    )

    def kpca_loss(x, sample_weight, floor, length_scale):
        recipe = eqx.tree_at(
            lambda current: (
                current.eigenvalue_floor,
                current.kernel.length_scale,
            ),
            kpca_base,
            (floor, length_scale),
        )
        model = recipe.fit_batch(MLBatch(x, sample_weight=sample_weight)).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    kpca_gradients = jax.grad(kpca_loss, argnums=(0, 1, 2, 3))(
        features,
        weights,
        kpca_base.eigenvalue_floor,
        kpca_base.kernel.length_scale,
    )
    _assert_finite(kpca_gradients)
    kpca_result = kpca_base.fit_batch(MLBatch(features, sample_weight=weights))
    assert kpca_result.gradient_contract.fit_features == "conditional"
    assert kpca_result.gradient_contract.fit_weights == "conditional"
    assert kpca_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(kpca_result.as_trainable(), query)

    nystrom_base = NystromRecipe(
        SquaredExponentialKernel(length_scale=0.95),
        n_components=3,
        selection="even",
        eigenvalue_floor=1e-7,
    )

    def nystrom_loss(x, floor, length_scale):
        recipe = eqx.tree_at(
            lambda current: (
                current.eigenvalue_floor,
                current.kernel.length_scale,
            ),
            nystrom_base,
            (floor, length_scale),
        )
        model = recipe.fit_batch(MLBatch(x)).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    nystrom_gradients = jax.grad(nystrom_loss, argnums=(0, 1, 2))(
        features,
        nystrom_base.eigenvalue_floor,
        nystrom_base.kernel.length_scale,
    )
    _assert_finite(nystrom_gradients)
    nystrom_result = nystrom_base.fit_batch(MLBatch(features))
    assert nystrom_result.gradient_contract.fit_features == "conditional"
    assert nystrom_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(nystrom_result.as_trainable(), query)

    key = jax.random.key(17)

    def rff_loss(length_scale):
        model = (
            RandomFourierFeaturesRecipe(
                SquaredExponentialKernel(length_scale=length_scale), n_components=12
            )
            .fit_batch(MLBatch(features), key=key)
            .as_trainable()
        )
        return jnp.sum(jnp.square(model(query)))

    assert jnp.isfinite(jax.grad(rff_loss)(jnp.asarray(0.95)))
    rff_result = RandomFourierFeaturesRecipe(
        SquaredExponentialKernel(length_scale=0.95), n_components=12
    ).fit_batch(MLBatch(features), key=key)
    assert rff_result.gradient_contract.fit_hyperparameters == "conditional"
    _assert_prediction_parameter_gradient(rff_result.as_trainable(), query)


def test_generic_gp_classifier_fit_gradients_status_and_complex_contract():
    features, _, labels, weights, query = _data()
    state = GaussianProcessLikelihoodState(
        kernel=SquaredExponentialKernel(length_scale=0.9),
        noise_scale=0.05,
        jitter=1e-4,
    )
    base = GaussianProcessClassifierRecipe(
        state, class_count=2, iterations=2, curvature_floor=1e-5
    )

    def fit_loss(x, sample_weight, length_scale, noise_scale, jitter, floor):
        recipe = eqx.tree_at(
            lambda current: (
                current.state.kernel.length_scale,
                current.state.noise_scale,
                current.state.jitter,
                current.curvature_floor,
            ),
            base,
            (length_scale, noise_scale, jitter, floor),
        )
        model = recipe.fit_batch(
            MLBatch(x, labels, sample_weight=sample_weight)
        ).as_trainable()
        return jnp.sum(jnp.square(model(query)))

    gradients = jax.grad(fit_loss, argnums=(0, 1, 2, 3, 4, 5))(
        features,
        weights,
        base.state.kernel.length_scale,
        base.state.noise_scale,
        base.state.jitter,
        base.curvature_floor,
    )
    _assert_finite(gradients)
    result = base.fit_batch(MLBatch(features, labels, sample_weight=weights))
    contract = result.gradient_contract
    assert (
        contract.fit_features,
        contract.fit_targets,
        contract.fit_weights,
        contract.fit_hyperparameters,
    ) == ("conditional", "none", "conditional", "conditional")
    _assert_prediction_parameter_gradient(result.as_trainable(), query)

    complex_features = features.astype(jnp.complex64) * (1.0 + 0.2j)
    with pytest.raises(TypeError, match="real coordinates"):
        base.fit_batch(MLBatch(complex_features, labels))

    nan_kernel = lambda left, right: jnp.asarray(jnp.nan)
    nonfinite = SupportVectorClassifierRecipe(nan_kernel, iterations=2).fit_batch(
        MLBatch(features, labels)
    )
    assert not nonfinite.valid
    assert nonfinite.status == ML_NONFINITE


def test_real_svm_families_reject_complex_kernel_expansions():
    features, targets, labels, _, _ = _data()
    complex_kernel = lambda left, right: jnp.asarray(
        jnp.vdot(left, right) + 1.0, dtype=jnp.complex64
    )
    recipes_and_targets = (
        (LeastSquaresSVMRecipe(complex_kernel), labels),
        (SupportVectorClassifierRecipe(complex_kernel, iterations=2), labels),
        (SupportVectorRegressorRecipe(complex_kernel, iterations=2), targets),
    )
    for recipe, target in recipes_and_targets:
        with pytest.raises(TypeError, match="real"):
            recipe.fit_batch(MLBatch(features, target))
    with pytest.raises(TypeError, match="real"):
        OneClassSVMRecipe(complex_kernel, iterations=2).fit_batch(MLBatch(features))


def test_case_batched_nystrom_rejects_one_global_out_of_sample_kernel():
    features, _, _, _, _ = _data()
    cases = jnp.stack((features, features + 0.2), axis=0)
    model = (
        NystromRecipe(SquaredExponentialKernel(), n_components=2)
        .fit_batch(MLBatch(cases))
        .as_trainable()
    )

    with pytest.raises(ValueError, match="does not define one global kernel"):
        model.as_kernel()
