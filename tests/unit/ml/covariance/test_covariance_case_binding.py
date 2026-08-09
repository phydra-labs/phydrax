#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.ml import MLBatch
from phydrax.ml.covariance import EmpiricalCovariance


def test_case_bound_covariance_rejects_a_case_axis_omitted_as_feature_axis():
    base = jnp.array([[-2.0, -1.0], [2.0, 1.0], [-1.0, 0.5], [1.0, -0.5]])
    features = jnp.stack((base, 2.0 * base + jnp.array([10.0, -3.0])))
    model = (
        EmpiricalCovariance(regularization=1e-3)
        .fit_batch(MLBatch(features))
        .as_trainable()
    )

    with pytest.raises(ValueError, match="case"):
        model(jnp.array([0.0, 0.0]))
    with pytest.raises(ValueError, match="case"):
        model.whiten(jnp.array([0.0, 0.0]))
