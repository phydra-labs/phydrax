#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch, SparseFeatures
from phydrax.ml.covariance import EmpiricalCovariance, RobustCovariance


def _sparse_batch() -> Any:
    features = SparseFeatures(
        jnp.array([[1.0], [2.0], [3.0]]),
        jnp.array([[0], [1], [0]]),
        feature_count=2,
    )
    return MLBatch(features)


@pytest.mark.parametrize(
    "recipe",
    [EmpiricalCovariance(), RobustCovariance(max_iterations=2, tolerance=1.0)],
)
def test_covariance_families_reject_implicit_sparse_materialization(recipe: Any) -> None:
    with pytest.raises(TypeError, match="requires dense features"):
        recipe.fit_batch(_sparse_batch())
