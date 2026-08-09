#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.kernels import FiniteFeatureKernel, SquaredExponentialKernel
from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml.kernel_methods import (
    BernoulliGaussianProcessClassifierRecipe,
    KernelPCARecipe,
    KernelRidgeRecipe,
    LeastSquaresSVMRecipe,
    NystromRecipe,
    OneClassSVMRecipe,
    RandomFourierFeaturesRecipe,
    SupportVectorClassifierRecipe,
    SupportVectorRegressorRecipe,
)
from phydrax.uq import GaussianProcessLikelihoodState


def _sparse_data():
    dense = jnp.array(
        [
            [-1.4, -0.2],
            [-0.8, 0.7],
            [-0.1, -0.5],
            [0.7, 0.4],
            [1.5, -0.3],
        ]
    )
    sparse = SparseFeatures(
        dense,
        jnp.broadcast_to(jnp.array([0, 1]), dense.shape),
        feature_count=2,
    )
    targets = jnp.array([-1.1, -0.5, 0.1, 0.8, 1.3])
    labels = jnp.array([0, 0, 0, 1, 1], dtype=jnp.int32)
    return sparse, targets, labels


def _assert_sparse_rejected(recipe, batch, *, key=None):
    with pytest.raises(TypeError, match="SparseFeatures|sparse|dense"):
        recipe.fit_batch(batch, key=key)


def test_dense_only_kernel_machines_reject_sparse_features_explicitly():
    sparse, targets, labels = _sparse_data()
    kernel = SquaredExponentialKernel(length_scale=0.9)
    recipes_and_targets = (
        (KernelRidgeRecipe(kernel, alpha=0.2), targets),
        (LeastSquaresSVMRecipe(kernel, alpha=0.2), labels),
        (
            SupportVectorClassifierRecipe(kernel, iterations=4, learning_rate=0.025),
            labels,
        ),
        (
            SupportVectorRegressorRecipe(kernel, iterations=4, learning_rate=0.01),
            targets,
        ),
    )

    for recipe, target in recipes_and_targets:
        _assert_sparse_rejected(recipe, MLBatch(sparse, target))
    _assert_sparse_rejected(OneClassSVMRecipe(kernel, iterations=4), MLBatch(sparse))


def test_dense_only_spectral_maps_reject_sparse_features_explicitly():
    sparse, _, _ = _sparse_data()
    kernel = SquaredExponentialKernel(length_scale=0.9)

    _assert_sparse_rejected(KernelPCARecipe(kernel, n_components=2), MLBatch(sparse))
    _assert_sparse_rejected(NystromRecipe(kernel, n_components=3), MLBatch(sparse))
    _assert_sparse_rejected(
        RandomFourierFeaturesRecipe(kernel, n_components=10),
        MLBatch(sparse),
        key=jax.random.key(29),
    )


def test_dense_only_exact_and_finite_gp_factors_reject_sparse_features():
    sparse, _, labels = _sparse_data()
    exact_state = GaussianProcessLikelihoodState(
        kernel=SquaredExponentialKernel(length_scale=0.9),
        noise_scale=0.0,
        jitter=1e-5,
    )
    _assert_sparse_rejected(
        BernoulliGaussianProcessClassifierRecipe(exact_state, iterations=3),
        MLBatch(sparse, labels),
    )

    finite_kernel = FiniteFeatureKernel(
        lambda point: jnp.stack((jnp.asarray(1.0), point[0], point[1])),
        jnp.eye(3),
        feature_map_id="sparse-gp-contract",
        max_derivative_order=None,
    )
    finite_state = GaussianProcessLikelihoodState(
        kernel=finite_kernel,
        noise_scale=0.0,
        jitter=1e-5,
    )
    _assert_sparse_rejected(
        BernoulliGaussianProcessClassifierRecipe(finite_state, iterations=3),
        MLBatch(sparse, labels),
    )
