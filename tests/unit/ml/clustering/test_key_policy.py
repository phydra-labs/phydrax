#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch
from phydrax.ml.clustering import KMeans, KMedoids, MiniBatchKMeans, SoftKMeans


_FEATURES = jnp.array([[-3.0], [3.0], [-2.0], [2.0], [-1.0], [1.0]])


@pytest.mark.parametrize(
    "recipe",
    [
        KMeans(2, initialization="random"),
        SoftKMeans(2, initialization="k-means++", tolerance=1e6),
        KMedoids(2, initialization="random"),
        MiniBatchKMeans(2, accumulation="deterministic", tolerance=1.0),
    ],
)
def test_randomized_cluster_initializers_require_and_replay_explicit_keys(
    recipe: Any,
) -> None:
    batch = MLBatch(_FEATURES)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    key = jax.random.key(41)
    first = recipe.fit_batch(batch, key=key).as_trainable()
    second = recipe.fit_batch(batch, key=key).as_trainable()
    assert jnp.array_equal(first.centers, second.centers)
    assert jnp.array_equal(first(_FEATURES), second(_FEATURES))


@pytest.mark.parametrize("initialization", ["random", "k-means++"])
def test_legacy_initialization_keys_are_refused(initialization: Any) -> None:
    with pytest.raises((TypeError, ValueError)):
        KMeans(2, initialization=initialization).fit_batch(
            MLBatch(_FEATURES), key=jax.random.PRNGKey(0)
        )


def test_random_initialization_draws_distinct_rows() -> None:
    result = KMeans(
        6, initialization="random", max_iterations=1, empty_policy="error"
    ).fit_batch(MLBatch(_FEATURES), key=jax.random.key(7))
    assert not result.diagnostics.empty_clusters_seen


def test_plus_plus_initialization_is_scale_invariant() -> None:
    recipe = KMeans(2, initialization="k-means++", max_iterations=1)
    a = recipe.fit_batch(MLBatch(_FEATURES), key=jax.random.key(9))
    b = recipe.fit_batch(MLBatch(_FEATURES * 2**-10), key=jax.random.key(9))
    assert jnp.array_equal(a.as_trainable().centers, b.as_trainable().centers * 2**10)  # ty: ignore[unresolved-attribute]
