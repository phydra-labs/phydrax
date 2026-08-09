#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch
from phydrax.ml.clustering import KMeans


def test_case_bound_cluster_model_rejects_a_case_axis_omitted_as_feature_axis():
    features = jnp.array(
        [
            [[-2.0, -1.0], [2.0, 1.0], [-1.0, -0.5], [1.0, 0.5]],
            [[8.0, 4.0], [12.0, 6.0], [9.0, 4.5], [11.0, 5.5]],
        ]
    )
    model = KMeans(2, initialization="first").fit_batch(MLBatch(features)).as_trainable()

    with pytest.raises(ValueError, match="case"):
        model(jnp.array([0.0, 0.0]))
