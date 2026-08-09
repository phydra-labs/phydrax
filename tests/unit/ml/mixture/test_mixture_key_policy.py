#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import ML_SUCCESS, MLBatch
from phydrax.ml.mixture import BayesianGaussianMixture


def test_bayesian_random_initialization_requires_and_replays_explicit_key():
    features = jnp.array([[-3.0], [3.0], [-2.0], [2.0], [-1.0], [1.0]])
    recipe = BayesianGaussianMixture(
        2, initialization="random", max_iterations=4, tolerance=1e6
    )
    batch = MLBatch(features)

    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    key = jax.random.key(43)
    first = recipe.fit_batch(batch, key=key)
    second = recipe.fit_batch(batch, key=key)
    assert first.status == ML_SUCCESS
    assert jnp.array_equal(first.as_trainable().means, second.as_trainable().means)
    assert jnp.array_equal(first.model(features), second.model(features))
