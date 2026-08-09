#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.ml.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    NormScaler,
    RobustScaler,
    SimpleImputer,
    StandardScaler,
)


def _case_batch():
    features = jnp.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 6.0], [5.0, 8.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
        ]
    )
    feature_mask = jnp.ones_like(features, dtype=bool).at[:, 1, 1].set(False)
    return phx.ml.MLBatch(
        features,
        feature_mask=feature_mask,
        sample_mask=jnp.array([True, True, False, True]),
        sample_weight=jnp.array([1.0, 2.0, 9.0, 1.0]),
    )


def test_standard_scaler_preserves_case_axes_masks_weights_schema_and_gradients():
    batch = _case_batch()
    result = StandardScaler().fit_batch(batch)
    model = result.as_trainable()

    assert result.valid.shape == (2,)
    assert jnp.all(result.valid)
    assert model.input_schema is batch.feature_schema
    assert model.output_schema is batch.feature_schema
    assert result.diagnostics.input_shape == (2, 4, 2)
    assert result.diagnostics.output_shape == (2, 4, 2)
    assert jnp.allclose(
        result.diagnostics.observed_weight, jnp.array([[4.0, 2.0], [4.0, 2.0]])
    )
    assert jnp.allclose(model.center, jnp.array([[3.0, 5.0], [4.5, 4.0]]))

    transformed = jax.jit(model)(batch.features)
    restored = model.inverse_transform(transformed)
    assert transformed.shape == batch.features.shape
    assert jnp.allclose(restored, batch.features)
    gradient = jax.grad(lambda value: jnp.sum(model(value)))(jnp.array([2.0, 3.0]))
    assert jnp.all(jnp.isfinite(gradient))
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_targets == "none"


@pytest.mark.parametrize("recipe", [MinMaxScaler(), MaxAbsScaler(), RobustScaler()])
def test_affine_scalers_are_vmap_compatible_and_inverse_on_nonconstant_data(recipe):
    features = jnp.array([[-3.0, 1.0], [-1.0, 2.0], [2.0, 4.0], [5.0, 8.0], [9.0, 16.0]])
    result = recipe.fit_batch(
        phx.ml.MLBatch(features, sample_weight=jnp.arange(1.0, 6.0))
    )
    model = result.as_trainable()
    probe = jnp.array([[-2.0, 1.5], [4.0, 7.0]])

    transformed = jax.jit(jax.vmap(model))(probe)
    assert transformed.shape == probe.shape
    assert jnp.allclose(model.inverse_transform(transformed), probe, rtol=2e-5, atol=2e-5)
    assert model.output_schema.names == ("feature_0", "feature_1")
    assert result.diagnostics.constant_features.shape == (2,)


def test_minmax_clip_and_norm_scaling_explicitly_reject_nonbijective_inverse():
    batch = phx.ml.MLBatch(jnp.array([[0.0, -2.0], [2.0, 4.0], [4.0, 8.0]]))
    clipped = MinMaxScaler(clip=True).fit_batch(batch).as_trainable()
    normalized = NormScaler("l2").fit_batch(batch).as_trainable()

    assert jnp.allclose(clipped(jnp.array([8.0, -6.0])), jnp.array([1.0, 0.0]))
    with pytest.raises(NotImplementedError, match="not bijective"):
        clipped.inverse_transform(jnp.array([0.5, 0.5]))
    assert jnp.allclose(normalized(jnp.array([3.0, 4.0])), jnp.array([0.6, 0.8]))
    assert jnp.allclose(normalized(jnp.zeros(2)), jnp.zeros(2))
    with pytest.raises(NotImplementedError, match="not bijective"):
        normalized.inverse_transform(jnp.ones(2))
    assert normalized.input_schema.names == normalized.output_schema.names


def test_scaler_constant_empty_and_invalid_weight_diagnostics_are_finite_and_exact():
    constant = StandardScaler().fit_batch(phx.ml.MLBatch(jnp.ones((3, 2))))
    empty = RobustScaler().fit_batch(
        phx.ml.MLBatch(jnp.ones((3, 2)), sample_mask=jnp.zeros(3, dtype=bool))
    )
    negative = StandardScaler().fit_batch(
        phx.ml.MLBatch(jnp.ones((2, 1)), sample_weight=jnp.array([1.0, -1.0]))
    )
    nonfinite = StandardScaler().fit_batch(
        phx.ml.MLBatch(jnp.ones((2, 1)), sample_weight=jnp.array([1.0, jnp.nan]))
    )

    assert jnp.all(constant.diagnostics.constant_features)
    assert jnp.all(jnp.isfinite(constant.as_trainable().scale))
    assert not bool(empty.valid)
    assert int(empty.status) == phx.ml.ML_INSUFFICIENT_DATA
    assert jnp.all(jnp.isfinite(empty.as_trainable().center))
    assert int(negative.status) == phx.ml.ML_INFEASIBLE
    assert int(nonfinite.status) == phx.ml.ML_NONFINITE


def test_simple_imputer_weighted_masked_strategies_indicators_and_inverse_rejection():
    features = jnp.array([[1.0, jnp.nan], [3.0, 4.0], [100.0, 8.0]])
    batch = phx.ml.MLBatch(
        features,
        sample_mask=jnp.array([True, True, False]),
        sample_weight=jnp.array([1.0, 3.0, 10.0]),
    )
    result = SimpleImputer(strategy="mean", add_indicator=True).fit_batch(batch)
    model = result.as_trainable()

    transformed = jax.jit(model)(jnp.array([[jnp.nan, 2.0], [5.0, jnp.nan]]))
    assert jnp.allclose(
        transformed, jnp.array([[2.5, 2.0, 1.0, 0.0], [5.0, 4.0, 0.0, 1.0]])
    )
    assert model.output_schema.names == (
        "feature_0",
        "feature_1",
        "feature_0_missing",
        "feature_1_missing",
    )
    assert model.output_schema.kinds[-2:] == ("boolean", "boolean")
    assert result.gradient_contract.fit_features == "conditional"
    assert result.gradient_contract.fit_targets == "none"
    with pytest.raises(NotImplementedError, match="not bijective"):
        model.inverse_transform(transformed)

    median = SimpleImputer(strategy="median").fit_batch(batch).as_trainable()
    mode = (
        SimpleImputer(strategy="most_frequent")
        .fit_batch(
            phx.ml.MLBatch(
                jnp.array([[1.0], [2.0], [2.0], [3.0]]),
                sample_weight=jnp.array([5.0, 1.0, 2.0, 1.0]),
            )
        )
        .as_trainable()
    )
    constant = SimpleImputer(strategy="constant", fill_value=-7.0).fit_batch(
        phx.ml.MLBatch(jnp.full((2, 1), jnp.nan))
    )
    assert jnp.allclose(median(jnp.array([jnp.nan, jnp.nan])), jnp.array([3.0, 4.0]))
    assert jnp.allclose(mode(jnp.array([jnp.nan])), jnp.array([1.0]))
    assert bool(constant.valid)
    assert jnp.allclose(constant.as_trainable()(jnp.array([jnp.nan])), jnp.array([-7.0]))
    assert median.input_schema.names == median.output_schema.names
