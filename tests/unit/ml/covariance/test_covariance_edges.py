#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.ml import ML_INSUFFICIENT_DATA, ML_RANK_DEFICIENT, MLBatch
from phydrax.ml.covariance import EmpiricalCovariance, WeightedCovariance


def test_covariance_distinguishes_empty_and_singleton_batches():
    features = jnp.array([[2.0, -1.0], [4.0, 3.0]])
    empty = EmpiricalCovariance(regularization=1e-4).fit_batch(
        MLBatch(features, sample_mask=jnp.zeros(2, dtype=bool))
    )
    singleton = EmpiricalCovariance(regularization=1e-4).fit_batch(MLBatch(features[:1]))
    corrected_singleton = WeightedCovariance(
        correction=1.0, regularization=1e-4
    ).fit_batch(MLBatch(features[:1]))

    assert empty.status == ML_INSUFFICIENT_DATA
    assert not empty.valid
    assert empty.diagnostics.effective_samples == 0.0
    assert singleton.status == ML_RANK_DEFICIENT
    assert singleton.valid
    assert singleton.diagnostics.effective_samples == 1.0
    assert jnp.allclose(singleton.as_trainable().mean, features[0])
    assert jnp.all(jnp.linalg.eigvalsh(singleton.as_trainable().covariance) > 0.0)
    assert corrected_singleton.status == ML_INSUFFICIENT_DATA
    assert not corrected_singleton.valid
