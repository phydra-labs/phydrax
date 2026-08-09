#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.kernels import Matern32Kernel, SquaredExponentialKernel
from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml._contracts import ML_INSUFFICIENT_DATA
from phydrax.ml.kernel_methods import (
    BernoulliGaussianProcessClassifierRecipe,
    KernelRidgeRecipe,
    NystromRecipe,
    RandomFourierFeaturesRecipe,
)
from phydrax.uq import GaussianProcessLikelihoodState


def test_kernel_ridge_requires_explicit_sparse_materialization():
    dense = jnp.array(
        [[1.0, 0.0, 2.0], [0.0, -1.0, 0.5], [2.0, 1.0, 0.0], [-1.0, 0.0, 1.0]]
    )
    values = jnp.array([[1.0, 2.0], [-1.0, 0.5], [2.0, 1.0], [-1.0, 1.0]])
    columns = jnp.array([[0, 2], [1, 2], [0, 1], [0, 2]])
    sparse = SparseFeatures(values, columns, feature_count=3)
    targets = jnp.array([1.0, -0.5, 2.0, 0.2])
    recipe = KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.2)

    dense_model = recipe.fit_batch(MLBatch(dense, targets)).as_trainable()
    with pytest.raises(TypeError, match="requires dense features"):
        recipe.fit_batch(MLBatch(sparse, targets))
    explicit_model = recipe.fit_batch(MLBatch(sparse.to_dense(), targets)).as_trainable()
    assert jnp.allclose(dense_model(dense), explicit_model(dense), atol=2e-5)
    assert jax.vmap(dense_model)(dense).shape == (4,)


def test_masked_nonfinite_rows_are_excluded_and_negative_weights_fail_closed():
    features = jnp.array([[0.0], [jnp.nan], [1.0], [2.0]])
    targets = jnp.array([0.0, jnp.nan, 1.0, 2.0])
    result = KernelRidgeRecipe(SquaredExponentialKernel(), alpha=0.1).fit_batch(
        MLBatch(features, targets, sample_mask=jnp.array([True, False, True, True]))
    )
    assert result.valid
    assert result.diagnostics.effective_samples == 3
    assert jnp.all(jnp.isfinite(result.as_trainable()(jnp.array([[0.5], [1.5]]))))

    with pytest.raises(Exception, match="nonnegative"):
        KernelRidgeRecipe(SquaredExponentialKernel()).fit_batch(
            MLBatch(
                jnp.nan_to_num(features),
                jnp.nan_to_num(targets),
                sample_weight=jnp.array([1.0, -1.0, 1.0, 1.0]),
            )
        )


def test_approximation_capacity_and_kernel_support_fail_closed():
    x = jnp.arange(10.0).reshape((5, 2))
    insufficient = NystromRecipe(SquaredExponentialKernel(), n_components=3).fit_batch(
        MLBatch(x, sample_mask=jnp.array([True, False, False, False, True]))
    )
    assert not insufficient.valid
    assert insufficient.status == ML_INSUFFICIENT_DATA

    with pytest.raises(TypeError, match="SquaredExponentialKernel"):
        RandomFourierFeaturesRecipe(Matern32Kernel(), n_components=4)


def test_gp_classification_preserves_case_axes_masks_and_determinism():
    base = jnp.array([[-1.0], [-0.2], [0.4], [1.2]])
    features = jnp.stack((base, base + 0.1), axis=0)
    labels = jnp.array([[0, 0, 1, 1], [0, 1, 1, 1]], dtype=jnp.int32)
    state = GaussianProcessLikelihoodState(
        kernel=SquaredExponentialKernel(), noise_scale=0.0, jitter=1e-5
    )
    recipe = BernoulliGaussianProcessClassifierRecipe(state, iterations=3)
    batch = MLBatch(
        features,
        labels,
        target_mask=jnp.array([True, False, True, True]),
        sample_weight=jnp.array([1.0, 2.0, 1.0, 0.5]),
    )
    first = recipe.fit_batch(batch)
    second = recipe.fit_batch(batch)
    query = features[:, :2]

    assert first.as_trainable()(query).shape == (2, 2, 2)
    assert jnp.allclose(first.as_trainable()(query), second.as_trainable()(query))
    assert first.diagnostics.effective_samples.shape == (2,)
    assert first.as_trainable().input_binding().batch_mode == "blockwise"
