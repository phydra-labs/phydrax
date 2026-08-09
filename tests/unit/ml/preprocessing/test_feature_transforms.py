#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.ml.preprocessing import (
    FeatureHasher,
    FourierFeatures,
    GaussianRandomProjection,
    PolynomialFeatures,
    PowerTransformer,
    QuantileTransformer,
    RandomFourierFeatures,
    SparseRandomProjection,
    SplineTransformer,
)
from phydrax.sparse import SparseLinearMap


def _named_batch():
    return phx.ml.MLBatch(
        jnp.array(
            [
                [-2.0, 1.0, 0.5],
                [-1.0, 2.0, 1.0],
                [0.0, 3.0, 1.5],
                [1.0, 4.0, 2.0],
                [2.0, 5.0, 2.5],
            ]
        ),
        feature_schema=phx.ml.FeatureSchema(("a", "b", "c")),
    )


def test_polynomial_and_interaction_features_names_inverse_complex_and_vmap():
    batch = phx.ml.MLBatch(
        jnp.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]),
        feature_schema=phx.ml.FeatureSchema(("x", "y")),
    )
    result = PolynomialFeatures(degree=2, include_bias=False).fit_batch(batch)
    model = result.as_trainable()
    probe = jnp.array([[2.0, 3.0], [4.0, 5.0]])
    transformed = jax.jit(jax.vmap(model))(probe)

    assert model.output_schema.names == ("x", "y", "x^2", "x*y", "y^2")
    assert jnp.allclose(transformed[0], jnp.array([2.0, 3.0, 4.0, 6.0, 9.0]))
    assert jnp.allclose(model.inverse_transform(transformed), probe)
    assert result.gradient_contract.prediction_inputs == "smooth"
    assert result.gradient_contract.prediction_parameters == "none"

    interactions = PolynomialFeatures(
        degree=3, interaction_only=True, include_bias=False
    ).fit_batch(batch)
    assert interactions.as_trainable().output_schema.names == ("x", "y", "x*y")
    complex_output = model(jnp.array([1.0 + 2.0j, 2.0 - 1.0j]))
    assert jnp.issubdtype(complex_output.dtype, jnp.complexfloating)


def test_spline_transformer_uses_fixed_basis_schema_partition_of_unity_and_hard_fit_contract():
    batch = phx.ml.MLBatch(
        jnp.stack((jnp.linspace(-1.0, 1.0, 7), jnp.linspace(0.0, 3.0, 7)), axis=-1),
        sample_weight=jnp.arange(1.0, 8.0),
        feature_schema=phx.ml.FeatureSchema(("position", "time")),
    )
    result = SplineTransformer(
        n_knots=4, degree=2, knots="quantile", bounds="clip"
    ).fit_batch(batch)
    model = result.as_trainable()
    probe = jnp.array([[-0.25, 1.25], [0.75, 2.75]])
    transformed = jax.jit(jax.vmap(model))(probe)

    assert transformed.shape == (2, 10)
    assert jnp.allclose(
        transformed.reshape(2, 2, 5).sum(axis=-1), jnp.ones((2, 2)), atol=2e-5
    )
    assert model.output_schema.names[0] == "position_spline_0"
    assert model.output_schema.names[-1] == "time_spline_4"
    assert result.diagnostics.output_shape == (7, 10)
    assert result.gradient_contract.fit_mode == "stopped"
    assert result.gradient_contract.prediction_inputs == "almost-everywhere"
    assert "knot_spans" in result.gradient_contract.nondifferentiable_outputs
    gradient = jax.grad(lambda value: jnp.sum(model(value) ** 2))(
        jnp.array([0.125, 1.375])
    )
    assert jnp.all(jnp.isfinite(gradient))
    with pytest.raises(NotImplementedError, match="no single-valued inverse"):
        model.inverse_transform(transformed)


def test_deterministic_fourier_features_explicit_schema_inverse_and_fit_contract():
    batch = phx.ml.MLBatch(
        jnp.array([[0.0, 0.5], [0.5, 1.0], [1.0, 1.5]]),
        feature_schema=phx.ml.FeatureSchema(("x", "t")),
    )
    recipe = FourierFeatures(
        2,
        period=(2.0, 4.0),
        origin=0.0,
        include_bias=True,
        include_original=True,
    )
    result = recipe.fit_batch(batch)
    model = result.as_trainable()
    probe = jnp.array([[0.25, 0.75], [0.75, 1.25]])

    first = jax.jit(jax.vmap(model))(probe)
    second = jax.jit(jax.vmap(model))(probe)
    assert first.shape == (2, 11)
    assert jnp.array_equal(first, second)
    assert jnp.allclose(model.inverse_transform(first), probe)
    assert model.output_schema.names[:3] == ("x", "t", "fourier_bias")
    assert result.gradient_contract.fit_features == "none"
    assert result.gradient_contract.prediction_inputs == "smooth"

    periodic = FourierFeatures(1, period=2.0, origin=0.0).fit_batch(batch).as_trainable()
    with pytest.raises(NotImplementedError, match="not injective"):
        periodic.inverse_transform(periodic(probe[0]))


def test_random_fourier_features_require_keys_are_deterministic_and_differentiable_on_apply():
    batch = _named_batch()
    recipe = RandomFourierFeatures(12, gamma=0.5)
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)

    key = jax.random.key(7)
    first = recipe.fit_batch(batch, key=key)
    second = recipe.fit_batch(batch, key=key)
    other = recipe.fit_batch(batch, key=jax.random.key(8))
    model = first.as_trainable()
    probe = batch.features[:2]

    assert jnp.array_equal(model.frequencies, second.as_trainable().frequencies)
    assert not jnp.array_equal(model.frequencies, other.as_trainable().frequencies)
    assert jax.jit(jax.vmap(model))(probe).shape == (2, 12)
    gradient = jax.grad(lambda value: jnp.sum(model(value)))(probe[0])
    assert jnp.all(jnp.isfinite(gradient))
    assert first.gradient_contract.fit_features == "none"
    assert "random_frequencies" in first.gradient_contract.nondifferentiable_outputs
    with pytest.raises(NotImplementedError, match="not invertible"):
        model.inverse_transform(model(probe[0]))


def test_feature_hasher_is_name_deterministic_sparse_in_action_and_noninvertible():
    batch = _named_batch()
    first = FeatureHasher(4).fit_batch(batch)
    second = FeatureHasher(4).fit_batch(batch)
    model = first.as_trainable()
    probe = jnp.array([[2.0, -1.0, 3.0], [1.0, 4.0, -2.0]])
    transformed = jax.jit(jax.vmap(model))(probe)
    expected = jax.vmap(
        lambda row: jnp.zeros(4).at[model.buckets].add(row * model.signs)
    )(probe)

    assert jnp.array_equal(model.buckets, second.as_trainable().buckets)
    assert jnp.allclose(transformed, expected)
    assert model.output_schema.names == ("hash_0", "hash_1", "hash_2", "hash_3")
    assert first.gradient_contract.prediction_parameters == "none"
    assert "hash_routes" in first.gradient_contract.nondifferentiable_outputs
    with pytest.raises(NotImplementedError, match="not invertible"):
        model.inverse_transform(transformed)


@pytest.mark.parametrize(
    "recipe", [GaussianRandomProjection(2), SparseRandomProjection(2, density=0.5)]
)
def test_random_projections_require_explicit_keys_are_deterministic_jittable_and_reject_inverse(
    recipe,
):
    batch = _named_batch()
    with pytest.raises(ValueError, match="explicit JAX key"):
        recipe.fit_batch(batch)
    key = jax.random.key(11)
    first = recipe.fit_batch(batch, key=key)
    second = recipe.fit_batch(batch, key=key)
    model = first.as_trainable()
    probe = batch.features[:2]

    output = jax.jit(jax.vmap(model))(probe)
    assert output.shape == (2, 2)
    if isinstance(model.projection, SparseLinearMap):
        assert isinstance(second.as_trainable().projection, SparseLinearMap)
        assert jnp.array_equal(
            model.projection.relation.valid,
            second.as_trainable().projection.relation.valid,
        )
        assert jnp.allclose(output, probe @ model.projection.as_dense().T)
    else:
        assert jnp.array_equal(model.projection, second.as_trainable().projection)
        assert jnp.allclose(output, probe @ model.projection)
    assert first.gradient_contract.fit_features == "none"
    assert first.gradient_contract.prediction_inputs == "smooth"
    assert first.gradient_contract.nondifferentiable_outputs
    with pytest.raises(NotImplementedError, match="not invertible"):
        model.inverse_transform(output)


def test_power_transformer_hard_fit_smooth_apply_inverse_domains_and_schema():
    batch = phx.ml.MLBatch(
        jnp.array([[-2.0, 0.25], [-1.0, 0.5], [0.0, 1.0], [1.0, 2.0], [3.0, 4.0]]),
        feature_schema=phx.ml.FeatureSchema(("signed", "positive")),
    )
    result = PowerTransformer("yeo-johnson", n_lambdas=17).fit_batch(batch)
    model = result.as_trainable()
    probe = jnp.array([[-1.5, 0.75], [2.0, 3.0]])
    transformed = jax.jit(jax.vmap(model))(probe)

    assert jnp.allclose(model.inverse_transform(transformed), probe, rtol=2e-5, atol=2e-5)
    assert model.output_schema.names == ("signed", "positive")
    assert result.gradient_contract.fit_mode == "stopped"
    assert result.gradient_contract.prediction_inputs == "almost-everywhere"
    assert "selected_lambda" in result.gradient_contract.nondifferentiable_outputs
    gradient = jax.grad(lambda value: jnp.sum(model(value)))(probe[0])
    assert jnp.all(jnp.isfinite(gradient))

    box = (
        PowerTransformer("box-cox", n_lambdas=9)
        .fit_batch(phx.ml.MLBatch(jnp.array([[0.5], [1.0], [2.0], [4.0]])))
        .as_trainable()
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="strictly positive"):
        box(jnp.array([-1.0]))


def test_quantile_transform_uniform_normal_inverse_ties_empty_status_and_grad_contract():
    features = jnp.stack((jnp.linspace(-2.0, 2.0, 9), jnp.linspace(1.0, 5.0, 9)), axis=-1)
    batch = phx.ml.MLBatch(features, feature_schema=phx.ml.FeatureSchema(("x", "y")))
    uniform_result = QuantileTransformer(9).fit_batch(batch)
    uniform = uniform_result.as_trainable()
    probe = jnp.array([[-1.5, 1.5], [0.5, 4.5]])
    transformed = jax.jit(jax.vmap(uniform))(probe)

    assert jnp.allclose(
        uniform.inverse_transform(transformed), probe, rtol=2e-5, atol=2e-5
    )
    assert uniform.output_schema.names == ("x", "y")
    assert uniform_result.gradient_contract.fit_mode == "stopped"
    assert uniform_result.gradient_contract.prediction_inputs == "almost-everywhere"
    assert (
        "weighted_order_statistics"
        in uniform_result.gradient_contract.nondifferentiable_outputs
    )
    gradient = jax.grad(lambda value: jnp.sum(uniform(value)))(probe[0])
    assert jnp.all(jnp.isfinite(gradient))

    normal = (
        QuantileTransformer(9, output_distribution="normal")
        .fit_batch(batch)
        .as_trainable()
    )
    assert jnp.all(jnp.isfinite(normal(jnp.array([0.0, 3.0]))))
    tied = QuantileTransformer(5).fit_batch(phx.ml.MLBatch(jnp.ones((5, 1))))
    with pytest.raises(eqx.EquinoxRuntimeError, match="not bijective"):
        tied.as_trainable().inverse_transform(jnp.array([0.5]))
    empty = QuantileTransformer(5).fit_batch(
        phx.ml.MLBatch(jnp.ones((3, 1)), sample_mask=jnp.zeros(3, dtype=bool))
    )
    assert int(empty.status) == phx.ml.ML_INSUFFICIENT_DATA
    assert jnp.all(jnp.isfinite(empty.as_trainable().quantiles))
