#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml.clustering import DBSCAN, KMeans, SpectralBiclustering


def _sparse_batch() -> Any:
    features = SparseFeatures(
        jnp.array([[1.0], [2.0], [3.0]]),
        jnp.array([[0], [1], [0]]),
        feature_count=2,
    )
    return MLBatch(features)


def test_dense_clustering_families_reject_implicit_sparse_materialization() -> None:
    for recipe in [
        KMeans(1, initialization="first"),
        DBSCAN(1, 3, minimum_samples=1.0),
        SpectralBiclustering(1, 1),
    ]:
        with pytest.raises(TypeError, match="requires dense features"):
            recipe.fit_batch(_sparse_batch())
