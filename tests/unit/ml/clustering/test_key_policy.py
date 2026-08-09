#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch
from phydrax.ml.clustering import KMeans, KMedoids, SoftKMeans


_FEATURES = jnp.array([[-3.0], [3.0], [-2.0], [2.0], [-1.0], [1.0]])


@pytest.mark.parametrize(
    "recipe",
    [
        KMeans(2, initialization="random"),
        SoftKMeans(2, initialization="k-means++", tolerance=1e6),
        KMedoids(2, initialization="random"),
    ],
)
def test_randomized_cluster_initializers_require_and_replay_explicit_keys(recipe):
    batch = MLBatch(_FEATURES)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    key = jax.random.key(41)
    first = recipe.fit_batch(batch, key=key).as_trainable()
    second = recipe.fit_batch(batch, key=key).as_trainable()
    assert jnp.array_equal(first.centers, second.centers)
    assert jnp.array_equal(first(_FEATURES), second(_FEATURES))
