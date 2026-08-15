#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import coordax as cx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_predictive_field_preserves_named_dims_and_decomposes_variance():
    samples = cx.Field(
        jnp.asarray([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0]]),
        dims=("draw", "x"),
    )
    conditional = cx.Field(jnp.full((3,), 4.0), dims=("x",))
    prediction = phx.uq.PredictiveField(
        samples,
        (phx.uq.SampleAxis("draw", "epistemic"),),
        conditional_variance=conditional,
    )

    assert prediction.mean().dims == ("x",)
    assert jnp.allclose(jnp.asarray(prediction.mean().data), jnp.asarray([1.0, 3.0, 5.0]))
    assert jnp.allclose(jnp.asarray(prediction.epistemic_variance().data), 1.0)
    assert jnp.allclose(jnp.asarray(prediction.total_variance().data), 5.0)
    assert prediction.interval(0.1, 0.9).nominal_coverage == pytest.approx(0.8)


def test_predictive_field_valid_mask_excludes_failed_realizations():
    prediction = phx.uq.PredictiveField(
        cx.Field(jnp.asarray([[1.0, 2.0], [100.0, 200.0]]), dims=("draw", "x")),
        (phx.uq.SampleAxis("draw", "input"),),
        valid=cx.Field(jnp.asarray([True, False]), dims=("draw",)),
    )

    assert jnp.allclose(jnp.asarray(prediction.mean().data), jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(jnp.asarray(prediction.input_variance().data), 0.0)


def test_predictive_conditional_variance_broadcasts_over_valid_sample_axis():
    samples = cx.Field(jnp.asarray([[0.0, 1.0], [2.0, 3.0]]), dims=("draw", "x"))
    conditional = cx.Field(jnp.asarray([4.0, 9.0]), dims=("x",))
    valid = cx.Field(jnp.asarray([True, False]), dims=("draw",))
    prediction = phx.uq.PredictiveField(
        samples,
        (phx.uq.SampleAxis("draw", "epistemic"),),
        conditional_variance=conditional,
        valid=valid,
    )

    assert jnp.array_equal(
        jnp.asarray(prediction.observation_variance().data),
        jnp.asarray(conditional.data),
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        phx.uq.PredictiveField(
            samples,
            (phx.uq.SampleAxis("draw", "observation"),),
            conditional_variance=conditional,
        )


def test_feature_dropout_is_function_locked_and_requires_key():
    layer = phx.nn.layers.Dropout(32, p=0.5, mode="feature")
    values = jnp.ones((7, 32))

    with pytest.raises(ValueError, match="explicit evaluation key"):
        layer(values)

    draw = layer(values, key=jr.key(0))
    assert jnp.all(draw == draw[0])
    assert jnp.array_equal(draw, layer(values, key=jr.key(0)))
    assert not jnp.array_equal(draw, layer(values, key=jr.key(1)))


def test_mlp_dropout_scan_matches_unrolled_and_inference_is_deterministic():
    kwargs = dict(
        in_size=2,
        out_size=1,
        width_size=8,
        depth=3,
        dropout=0.25,
        key=jr.key(2),
    )
    unrolled = phx.nn.models.MLP(**kwargs)
    scanned = phx.nn.models.MLP(**kwargs, scan=True)
    x = jnp.asarray([0.2, -0.3])

    assert jnp.allclose(unrolled(x, key=jr.key(3)), scanned(x, key=jr.key(3)))
    deterministic = phx.nn.layers.inference_mode(unrolled)
    assert jnp.array_equal(deterministic(x), deterministic(x))
