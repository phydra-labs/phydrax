#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.ml.preprocessing import (
    CategoricalSchema,
    OneHotEncoder,
    OrdinalEncoder,
    TargetEncoder,
)


def _schema():
    return CategoricalSchema(((0, 1), (10, 20, 30)), names=("color", "size"))


def test_categorical_schema_is_explicit_finite_unique_and_fixed_capacity():
    schema = _schema()
    assert schema.feature_count == 2
    assert schema.category_counts == (2, 3)
    assert schema.names == ("color", "size")

    with pytest.raises(ValueError, match="unique"):
        CategoricalSchema(((0, 0),))
    with pytest.raises(ValueError, match="finite"):
        CategoricalSchema(((0.0, jnp.nan),))
    with pytest.raises(TypeError, match="numeric"):
        CategoricalSchema((("red", "blue"),))


def test_ordinal_encoder_fail_and_indicator_policies_names_inverse_jit_and_vmap():
    schema = _schema()
    training = phx.ml.MLBatch(
        jnp.array([[0, 10], [1, 20], [0, 30]]),
        feature_schema=phx.ml.FeatureSchema(
            ("color", "size"), kinds=("categorical", "categorical")
        ),
    )
    failed_fit = OrdinalEncoder(schema, unknown_policy="fail").fit_batch(
        phx.ml.MLBatch(jnp.array([[0, 10], [7, 20], [1, 30]]))
    )
    assert not bool(failed_fit.valid)
    assert int(failed_fit.status) == phx.ml.ML_INFEASIBLE
    assert jnp.array_equal(failed_fit.diagnostics.unknown_count, jnp.array([1, 0]))

    fail_model = OrdinalEncoder(schema).fit_batch(training).as_trainable()
    with pytest.raises(eqx.EquinoxRuntimeError, match="unknown category"):
        fail_model(jnp.array([7, 20]))

    result = OrdinalEncoder(schema, unknown_policy="indicator").fit_batch(training)
    model = result.as_trainable()
    encoded = jax.jit(jax.vmap(model))(jnp.array([[1, 20], [7, 30]]))
    assert jnp.array_equal(encoded, jnp.array([[1, 1, 0, 0], [-1, 2, 1, 0]]))
    assert model.output_schema.names == ("color", "size", "color_unknown", "size_unknown")
    assert model.output_schema.kinds == ("ordinal", "ordinal", "boolean", "boolean")
    assert jnp.array_equal(model.inverse_transform(encoded[:1]), jnp.array([[1, 20]]))
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="not invertible"):
        model.inverse_transform(encoded[1:])
    assert result.gradient_contract.prediction_inputs == "none"
    assert result.gradient_contract.prediction_parameters == "none"
    assert "ordinal_codes" in result.gradient_contract.nondifferentiable_outputs


def test_onehot_encoder_schema_unknown_indicator_inverse_and_hard_contract():
    schema = _schema()
    batch = phx.ml.MLBatch(
        jnp.array([[0, 10], [1, 20], [0, 30]]),
        sample_weight=jnp.array([1.0, 2.0, 3.0]),
    )
    result = OneHotEncoder(schema, unknown_policy="indicator").fit_batch(batch)
    model = result.as_trainable()
    encoded = jax.jit(jax.vmap(model))(jnp.array([[1, 20], [7, 30]]))

    assert encoded.shape == (2, 7)
    assert jnp.array_equal(
        encoded,
        jnp.array(
            [
                [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            ]
        ),
    )
    assert model.output_schema.names == (
        "color=0",
        "color=1",
        "size=10",
        "size=20",
        "size=30",
        "color_unknown",
        "size_unknown",
    )
    assert jnp.array_equal(model.inverse_transform(encoded[:1]), jnp.array([[1, 20]]))
    with pytest.raises((eqx.EquinoxRuntimeError, ValueError), match="not invertible"):
        model.inverse_transform(encoded[1:])
    assert result.diagnostics.category_weight.shape == (2, 3)
    assert result.diagnostics.input_shape == (3, 2)
    assert result.diagnostics.output_shape == (3, 7)
    assert result.gradient_contract.prediction_inputs == "none"
    assert "one_hot_codes" in result.gradient_contract.nondifferentiable_outputs


def test_target_encoder_preserves_case_sample_axes_masks_weights_and_target_gradients():
    schema = CategoricalSchema(((0, 1),), names=("group",))
    features = jnp.array([[[0], [1], [0], [1]], [[0], [1], [0], [1]]])
    targets = jnp.array([[0.0, 10.0, 2.0, 14.0], [2.0, 20.0, 6.0, 24.0]])
    batch = phx.ml.MLBatch(
        features,
        targets,
        sample_mask=jnp.array([True, True, True, False]),
        sample_weight=jnp.array([1.0, 1.0, 3.0, 5.0]),
        feature_schema=phx.ml.FeatureSchema(("group",), kinds=("categorical",)),
    )
    result = TargetEncoder(schema, smoothing=0.0, unknown_policy="indicator").fit_batch(
        batch
    )
    model = result.as_trainable()
    probe = jnp.array([[[0], [1]], [[1], [0]]])
    transformed = jax.jit(model)(probe)

    assert transformed.shape == (2, 2, 2)
    assert jnp.allclose(
        transformed,
        jnp.array([[[1.5, 0.0], [10.0, 0.0]], [[20.0, 0.0], [5.0, 0.0]]]),
    )
    unknown = model(jnp.array([[[7]], [[7]]]))
    assert jnp.array_equal(unknown[..., 1], jnp.ones((2, 1)))
    assert model.output_schema.names == ("group_target", "group_unknown")
    assert result.diagnostics.category_weight.shape == (2, 1, 2)
    assert result.gradient_contract.prediction_inputs == "none"
    assert result.gradient_contract.fit_features == "none"
    assert result.gradient_contract.fit_targets == "conditional"
    assert result.gradient_contract.fit_weights == "conditional"
    with pytest.raises(NotImplementedError, match="not invertible"):
        model.inverse_transform(transformed)

    target_gradient = jax.grad(
        lambda y: jnp.sum(
            TargetEncoder(schema, smoothing=1.0)
            .fit_batch(phx.ml.MLBatch(features[0], y))
            .as_trainable()
            .encodings
        )
    )(targets[0])
    assert target_gradient.shape == targets[0].shape
    assert jnp.all(jnp.isfinite(target_gradient))
