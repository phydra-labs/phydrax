import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.operators.differential._dimension_estimators import (
    coordinate_second_derivative_samples,
    dimension_sum_samples,
    DimensionSamplingPolicy,
)
from phydrax.operators.differential._stochastic_estimators import (
    estimate_stochastic_trace,
    stochastic_divergence_samples,
    stochastic_trace_samples,
    StochasticTracePolicy,
)


def test_raw_trace_samples_reduce_to_the_existing_estimate_and_replay():
    state = jnp.asarray([0.2, -0.3, 0.7])
    matrix = jnp.asarray([[1.0, 0.4, -0.2], [0.4, 2.0, 0.3], [-0.2, 0.3, 0.8]])
    function = lambda value: value[0] ** 4 + value[1] ** 2 + 3.0 * value[2] ** 2
    policy = StochasticTracePolicy(512, distribution="normal")

    samples = stochastic_trace_samples(
        function,
        state,
        lambda value, probe: matrix @ probe,
        jr.key(4),
        policy=policy,
    )
    estimate = estimate_stochastic_trace(
        function,
        state,
        lambda value, probe: matrix @ probe,
        jr.key(4),
        policy=policy,
    )

    assert samples.values.shape == (512,)
    assert jnp.array_equal(samples.mean, estimate.value)
    assert jnp.array_equal(samples.standard_error, estimate.standard_error)
    assert jnp.array_equal(samples.estimate().value, estimate.value)
    assert jnp.array_equal(samples.dependence_ids, jnp.arange(512))


def test_divergence_samples_use_jvps_and_match_linear_trace_in_expectation():
    dimension = 100
    diagonal = jnp.linspace(0.2, 1.2, dimension)
    state = jnp.linspace(-1.0, 1.0, dimension)
    samples = stochastic_divergence_samples(
        lambda value: diagonal * value,
        state,
        jr.key(7),
        policy=StochasticTracePolicy(64),
    )

    assert samples.values.shape == (64,)
    assert jnp.allclose(samples.mean, jnp.sum(diagonal))
    assert jnp.allclose(samples.standard_error, 0.0)


def test_probe_standard_error_and_parameter_gradient_have_expected_behavior():
    state = jnp.asarray([0.3, -0.5, 0.7, 0.1])
    field = lambda scale, value: scale * jnp.asarray(
        [value[1], value[0], value[3], value[2]]
    )

    def estimate(scale, count):
        samples = stochastic_divergence_samples(
            lambda value: field(scale, value),
            state,
            jr.key(11),
            policy=StochasticTracePolicy(count, distribution="normal"),
        )
        return samples.mean, samples.standard_error

    small_mean, small_error = estimate(jnp.asarray(0.8), 32)
    large_mean, large_error = estimate(jnp.asarray(0.8), 2048)
    gradient = jax.grad(lambda scale: estimate(scale, 2048)[0])(jnp.asarray(0.8))

    assert jnp.abs(large_mean) <= 5.0 * large_error
    assert large_error < small_error
    assert jnp.isfinite(gradient)


def test_raw_sample_contract_rejects_vector_field_shape_mismatch():
    with pytest.raises(ValueError, match="preserve"):
        stochastic_divergence_samples(
            lambda value: jnp.sum(value),
            jnp.ones((3,)),
            jr.key(1),
        )


def test_uniform_dimension_sum_is_unbiased_and_full_subset_is_exact():
    contributions = jnp.linspace(0.1, 2.0, 40)
    policy = DimensionSamplingPolicy(40, 12)
    samples = dimension_sum_samples(
        lambda index: contributions[index],
        jr.key(21),
        policy,
    )
    exact = jnp.sum(contributions)

    assert jnp.abs(samples.mean - exact) <= 5.0 * samples.standard_error
    assert samples.indices.shape == (12,)
    assert jnp.unique(samples.indices).shape == (12,)

    full = dimension_sum_samples(
        lambda index: contributions[index],
        jr.key(22),
        DimensionSamplingPolicy(40, 40),
    )
    assert jnp.allclose(full.mean, exact)
    assert jnp.allclose(full.standard_error, 0.0)


def test_importance_dimension_sampling_uses_inverse_probability_weights():
    contributions = jnp.asarray([1.0, 2.0, 4.0, 8.0])
    probabilities = contributions / jnp.sum(contributions)
    samples = dimension_sum_samples(
        lambda index: contributions[index],
        jr.key(23),
        DimensionSamplingPolicy(
            4,
            32,
            sampling="importance",
            replace=True,
            probabilities=probabilities,
        ),
    )

    assert jnp.allclose(samples.values, jnp.sum(contributions))
    assert jnp.allclose(samples.mean, jnp.sum(contributions))
    assert jnp.allclose(samples.standard_error, 0.0)


def test_coordinate_laplacian_samples_scale_to_dimension_1000_without_dense_hessian():
    dimension = 1000
    state = jnp.linspace(-1.0, 1.0, dimension)
    coefficients = jnp.linspace(0.5, 1.5, dimension)
    policy = DimensionSamplingPolicy(dimension, 16)
    samples = coordinate_second_derivative_samples(
        lambda value: jnp.sum(coefficients * value**2),
        state,
        jr.key(24),
        policy,
    )
    exact = 2.0 * jnp.sum(coefficients)

    assert samples.values.shape == (16,)
    assert samples.indices.shape == (16,)
    assert jnp.abs(samples.mean - exact) <= 5.0 * samples.standard_error
    compiled = jax.jit(
        lambda scale: coordinate_second_derivative_samples(
            lambda value: scale * jnp.sum(coefficients * value**2),
            state,
            jr.key(24),
            policy,
        ).mean
    )
    assert jnp.isfinite(compiled(jnp.asarray(1.0)))
