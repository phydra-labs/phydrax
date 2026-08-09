#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax._model import AbstractArrayModel, FrozenModel


class _ScaleModel(AbstractArrayModel):
    scale: jax.Array
    in_size: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(self, scale):
        self.scale = jnp.asarray(scale)
        self.in_size = 1
        self.out_size = 1

    def __call__(self, x, /, *, key=None):
        del key
        return self.scale * x


class _ScaleRecipe(phx.ml.AbstractRecipe):
    def fit_batch(self, batch, /, *, key=None):
        del key
        targets = batch.require_targets()
        scale = jnp.sum(batch.effective_weight() * targets) / jnp.sum(
            batch.effective_weight()
        )
        diagnostics = phx.ml.FitDiagnostics(
            valid=True,
            status=phx.ml.ML_SUCCESS,
            effective_samples=jnp.sum(batch.effective_weight() > 0),
            method="weighted-mean-scale",
        )
        return phx.ml.FitResult(
            _ScaleModel(scale),
            diagnostics,
            valid=True,
            status=phx.ml.ML_SUCCESS,
            method="weighted-mean-scale",
            gradient_contract=phx.ml.GradientContract.direct(),
        )


def test_batch_preserves_case_sample_output_and_weight_semantics():
    features = jnp.arange(24.0).reshape(2, 4, 3)
    targets = jnp.arange(16.0).reshape(2, 4, 2)
    batch = phx.ml.MLBatch(
        features,
        targets,
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 2.0, 7.0, 3.0]),
        measure_weight=jnp.array([0.5, 0.5, 0.5, 0.5]),
    )

    assert batch.case_shape == (2,)
    assert batch.target_shape == (2,)
    assert batch.feature_schema.names == ("feature_0", "feature_1", "feature_2")
    assert jnp.allclose(
        batch.effective_weight("product"),
        jnp.array([[0.5, 1.0, 0.0, 1.5], [0.5, 1.0, 0.0, 1.5]]),
    )
    selected = batch.take_samples(jnp.array([3, 0]))
    assert selected.features.shape == (2, 2, 3)
    assert jnp.allclose(selected.targets, targets[:, jnp.array([3, 0])])


def test_sparse_features_are_explicit_and_preserve_duplicate_entries():
    sparse = phx.ml.SparseFeatures(
        jnp.array([[1.0, 2.0], [3.0, 4.0]]),
        jnp.array([[0, 2], [1, 1]]),
        feature_count=3,
    )
    dense = sparse.to_dense()

    assert jnp.allclose(dense, jnp.array([[1.0, 0.0, 2.0], [0.0, 7.0, 0.0]]))
    assert jnp.allclose(
        sparse.right_matmul(jnp.arange(6.0).reshape(3, 2)),
        dense @ jnp.arange(6.0).reshape(3, 2),
    )
    with pytest.raises(ValueError, match="feature_mask is unsupported"):
        phx.ml.MLBatch(sparse, feature_mask=jnp.ones((2, 3), dtype=bool))


def test_fit_is_pure_frozen_and_remains_differentiable_when_called():
    recipe = _ScaleRecipe()
    features = jnp.ones((3, 1))
    targets = jnp.array([1.0, 2.0, 5.0])
    result = phx.ml.fit(
        recipe,
        features,
        targets,
        sample_weight=jnp.array([1.0, 1.0, 0.0]),
    )

    assert isinstance(result.model, FrozenModel)
    assert jnp.allclose(result.model(jnp.array([3.0])), jnp.array([4.5]))
    assert result.as_trainable() is result.model.model
    gradient = jax.grad(lambda value: jnp.sum(result.model(value)))(jnp.array([2.0]))
    assert jnp.allclose(gradient, jnp.array([1.5]))
    assert result.gradient_contract.fit_mode == "direct"


def test_existing_batch_rejects_duplicate_metadata():
    batch = phx.ml.MLBatch(jnp.ones((3, 1)), jnp.ones((3,)))
    with pytest.raises(ValueError, match="cannot accompany"):
        phx.ml.fit(_ScaleRecipe(), batch, sample_weight=jnp.ones((3,)))
