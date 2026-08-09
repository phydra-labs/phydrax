#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml.mixture import BayesianGaussianMixture, GaussianMixture


def _sparse_batch():
    features = SparseFeatures(
        jnp.array([[1.0], [2.0], [3.0]]),
        jnp.array([[0], [1], [0]]),
        feature_count=2,
    )
    return MLBatch(features)


@pytest.mark.parametrize(
    "recipe",
    [
        GaussianMixture(1, initialization="first"),
        BayesianGaussianMixture(1, initialization="first"),
    ],
)
def test_gaussian_mixture_families_reject_implicit_sparse_materialization(recipe):
    with pytest.raises(TypeError, match="requires dense features"):
        recipe.fit_batch(_sparse_batch())
