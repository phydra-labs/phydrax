#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import pytest

import phydrax as phx


def _linear_case():
    mean = jnp.asarray([0.3, -0.8])
    root = jnp.asarray([[1.1, 0.0], [0.25, 0.7]])
    matrix = jnp.asarray([[1.5, -0.4], [0.2, 0.9], [-0.7, 0.3]])
    offset = jnp.asarray([0.1, -0.5, 1.2])
    return mean, phx.uq.GaussianFactor(root), matrix, offset


def test_all_transforms_recover_affine_moments_and_cross_covariance():
    mean, factor, matrix, offset = _linear_case()
    covariance = factor.covariance
    expected_mean = matrix @ mean + offset
    expected_covariance = matrix @ covariance @ matrix.T
    expected_cross = covariance @ matrix.T
    function = lambda value: matrix @ value + offset

    results = (
        phx.uq.spherical_radial_cubature(function, mean, factor),
        phx.uq.scaled_unscented_transform(function, mean, factor),
        phx.uq.gauss_hermite_transform(function, mean, factor, order=3),
        phx.uq.first_order_gaussian_transform(function, mean, factor),
    )

    for result in results:
        assert result.successful
        assert result.status == phx.uq.NONLINEAR_GAUSSIAN_SUCCESS
        assert jnp.allclose(result.mean, expected_mean, atol=2e-6)
        assert jnp.allclose(result.factor.covariance, expected_covariance, atol=2e-6)
        assert jnp.allclose(result.cross_covariance, expected_cross, atol=2e-6)
        assert result.input_dimension == 2
        assert result.output_dimension == 3


def test_first_order_transform_uses_complex_real_linear_factor_directions():
    mean = jnp.asarray([0.2 + 0.3j, -0.4 + 0.1j])
    root = jnp.asarray([[0.8 + 0.2j, -0.1 + 0.3j], [0.25 - 0.4j, 0.6 + 0.1j]])

    def real_linear(value):
        return jnp.stack(
            (
                jnp.conj(value[0]),
                2.0 * value[0] - 0.5j * jnp.conj(value[1]),
                value[1] + jnp.conj(value[0]),
            )
        )

    output_directions = jnp.stack(
        (
            jnp.conj(root[0]),
            2.0 * root[0] - 0.5j * jnp.conj(root[1]),
            root[1] + jnp.conj(root[0]),
        )
    )
    expected_cross = root @ jnp.conj(output_directions.T)

    def transformed_cross(scale):
        factor = phx.uq.GaussianFactor(scale * root)
        return phx.uq.first_order_gaussian_transform(
            real_linear, mean, factor
        ).cross_covariance

    compiled_cross = jax.jit(transformed_cross)
    result = phx.uq.first_order_gaussian_transform(
        real_linear, mean, phx.uq.GaussianFactor(root)
    )
    scale = jnp.asarray(0.7)
    cross_gradient = jax.grad(
        lambda value: jnp.real(jnp.vdot(compiled_cross(value), compiled_cross(value)))
    )(scale)

    assert result.cross_covariance.shape == (2, 3)
    assert jnp.allclose(result.mean, real_linear(mean))
    assert jnp.allclose(
        result.factor.covariance,
        output_directions @ jnp.conj(output_directions.T),
    )
    assert jnp.allclose(result.cross_covariance, expected_cross)
    assert jnp.allclose(compiled_cross(scale), scale**2 * expected_cross)
    assert jnp.allclose(
        cross_gradient,
        4.0 * scale**3 * jnp.real(jnp.vdot(expected_cross, expected_cross)),
    )


def test_first_order_high_output_trace_has_no_quadratic_basis():
    output_size = 50_000

    def transformed_cross(root):
        factor = phx.uq.GaussianFactor(root.reshape((1, 1)))
        return phx.uq.first_order_gaussian_transform(
            lambda value: jnp.broadcast_to(value, (output_size,)),
            jnp.asarray(0.0),
            factor,
        ).cross_covariance

    traced = jax.make_jaxpr(transformed_cross)(jnp.asarray(0.5))
    intermediate_shapes = tuple(
        variable.aval.shape
        for equation in traced.jaxpr.eqns
        for variable in equation.outvars
        if hasattr(variable.aval, "shape")
    )

    assert (output_size, output_size) not in intermediate_shapes
    assert jax.jit(transformed_cross)(jnp.asarray(0.5)).shape == (1, output_size)


def test_quadratic_moments_distinguish_exact_and_first_order_rules():
    mean = jnp.asarray(0.3)
    variance = 0.7
    factor = phx.uq.GaussianFactor(jnp.asarray([[jnp.sqrt(variance)]]))
    function = lambda value: value**2
    exact_mean = mean**2 + variance
    exact_variance = 2.0 * variance**2 + 4.0 * mean**2 * variance
    exact_cross = 2.0 * mean * variance

    unscented = phx.uq.scaled_unscented_transform(function, mean, factor)
    hermite = phx.uq.gauss_hermite_transform(function, mean, factor, order=3)
    cubature = phx.uq.spherical_radial_cubature(function, mean, factor)
    first_order = phx.uq.first_order_gaussian_transform(function, mean, factor)

    for result in (unscented, hermite):
        assert jnp.allclose(result.mean, exact_mean, atol=2e-6)
        assert jnp.allclose(result.factor.covariance[0, 0], exact_variance, atol=2e-6)
        assert jnp.allclose(result.cross_covariance[0, 0], exact_cross, atol=2e-6)
    assert jnp.allclose(cubature.mean, exact_mean, atol=2e-6)
    assert jnp.allclose(cubature.cross_covariance[0, 0], exact_cross, atol=2e-6)
    assert jnp.allclose(first_order.mean, mean**2)
    assert jnp.allclose(first_order.factor.covariance[0, 0], 4.0 * mean**2 * variance)
    assert jnp.allclose(first_order.cross_covariance[0, 0], exact_cross, atol=2e-6)


def test_high_order_hermite_matches_a_deterministic_particle_reference():
    mean = jnp.asarray(-0.2)
    scale = 0.65
    factor = phx.uq.GaussianFactor(jnp.asarray([[scale]]))

    def function(value):
        return jnp.asarray([jnp.sin(value), jnp.exp(0.2 * value)])

    result = phx.uq.gauss_hermite_transform(function, mean, factor, order=9)
    particle_count = 100_001
    probabilities = (jnp.arange(particle_count) + 0.5) / particle_count
    particles = mean + scale * jsp.special.ndtri(probabilities)
    outputs = jax.vmap(function)(particles)
    reference_mean = jnp.mean(outputs, axis=0)
    centered_inputs = particles - jnp.mean(particles)
    centered_outputs = outputs - reference_mean
    reference_covariance = centered_outputs.T @ centered_outputs / particle_count
    reference_cross = centered_inputs @ centered_outputs / particle_count

    assert jnp.allclose(result.mean, reference_mean, atol=3e-5)
    assert jnp.allclose(result.factor.covariance, reference_covariance, atol=4e-5)
    assert jnp.allclose(result.cross_covariance[0], reference_cross, atol=4e-5)


def test_singular_and_zero_rank_factors_remain_observable_and_valid():
    singular = phx.uq.GaussianFactor(jnp.asarray([[1.0], [2.0]]))
    matrix = jnp.asarray([[0.5, -0.25], [1.0, 0.5]])
    transformed = phx.uq.spherical_radial_cubature(
        lambda value: matrix @ value,
        jnp.asarray([0.2, -0.1]),
        singular,
    )
    expected = matrix @ singular.covariance @ matrix.T

    assert transformed.valid
    assert transformed.factor.numerical_rank == 1
    assert jnp.allclose(transformed.factor.covariance, expected)

    zero = phx.uq.GaussianFactor(jnp.zeros((2, 0)))
    deterministic = phx.uq.spherical_radial_cubature(
        lambda value: jnp.asarray([value[0] - value[1]]),
        jnp.asarray([2.0, 0.5]),
        zero,
    )
    assert deterministic.valid
    assert deterministic.point_count == 1
    assert deterministic.factor.numerical_rank == 0
    assert jnp.array_equal(deterministic.factor.covariance, jnp.zeros((1, 1)))
    assert jnp.array_equal(deterministic.cross_covariance, jnp.zeros((2, 1)))


def test_dimension_and_tensor_point_guards_are_explicit():
    factor = phx.uq.GaussianFactor(jnp.eye(6))
    with pytest.raises(ValueError, match="got 6, cap 5"):
        phx.uq.gauss_hermite_transform(lambda value: value, jnp.zeros(6), factor)
    with pytest.raises(ValueError, match="requires 729, cap 700"):
        phx.uq.gauss_hermite_transform(
            lambda value: value,
            jnp.zeros(6),
            factor,
            max_dimension=6,
            order=3,
            max_points=700,
        )
    scalar_factor = phx.uq.GaussianFactor(jnp.ones((1, 1)))
    with pytest.raises(ValueError, match="got 257, cap 256"):
        phx.uq.scaled_unscented_transform(
            lambda value: jnp.full((257,), value**2),
            jnp.asarray(0.0),
            scalar_factor,
            beta=-2.0,
        )


def test_invalid_unscented_covariance_is_reported_without_repair():
    factor = phx.uq.GaussianFactor(jnp.ones((1, 1)))
    result = phx.uq.scaled_unscented_transform(
        lambda value: value**2,
        jnp.asarray(0.0),
        factor,
        beta=-2.0,
    )

    assert not result.valid
    assert result.status == phx.uq.NONLINEAR_GAUSSIAN_OUTPUT_FACTOR_INVALID
    assert not result.factor.valid
    assert jnp.any(~jnp.isfinite(result.factor.factor))


def test_regularization_is_applied_and_recorded_explicitly():
    factor = phx.uq.GaussianFactor(jnp.ones((1, 1)))
    result = phx.uq.spherical_radial_cubature(
        lambda value: jnp.asarray([2.0, -1.0]),
        jnp.asarray(0.0),
        factor,
        regularization=0.125,
    )

    assert result.regularization == 0.125
    assert result.factor.regularization == 0.125
    assert jnp.allclose(result.factor.covariance, 0.125 * jnp.eye(2))
    assert jnp.array_equal(result.cross_covariance, jnp.zeros((1, 2)))


def test_method_and_parameter_provenance_are_stable():
    factor = phx.uq.GaussianFactor(jnp.eye(2))
    mean = jnp.zeros(2)
    cubature = phx.uq.spherical_radial_cubature(lambda value: value, mean, factor)
    unscented = phx.uq.scaled_unscented_transform(
        lambda value: value,
        mean,
        factor,
        alpha=0.8,
        beta=2.5,
        kappa=0.25,
    )
    hermite = phx.uq.gauss_hermite_transform(
        lambda value: value,
        mean,
        factor,
        order=4,
    )
    first_order = phx.uq.first_order_gaussian_transform(
        lambda value: value,
        mean,
        factor,
    )

    assert (cubature.method_id, cubature.point_count) == (
        "spherical-radial-cubature",
        4,
    )
    assert (unscented.method_id, unscented.point_count) == ("scaled-unscented", 5)
    assert unscented.method_parameters == (
        ("alpha", 0.8),
        ("beta", 2.5),
        ("kappa", 0.25),
        ("max_output_dimension", 256.0),
    )
    assert (hermite.method_id, hermite.point_count) == ("gauss-hermite", 16)
    assert hermite.method_parameters[0] == ("order", 4.0)
    assert (first_order.method_id, first_order.point_count) == (
        "first-order-jvp-vjp",
        1,
    )


def test_event_pytrees_jit_vmap_and_gradients_preserve_contracts():
    pytree_mean = {"forcing": jnp.asarray([0.2, -0.5]), "parameter": jnp.asarray(0.7)}
    pytree_factor = phx.uq.GaussianFactor(jnp.eye(3))

    def pytree_function(value):
        return {
            "field": jnp.asarray(
                [value["forcing"][0] + value["parameter"], value["forcing"][1]]
            ),
            "total": jnp.sum(value["forcing"]) - value["parameter"],
        }

    pytree_result = phx.uq.spherical_radial_cubature(
        pytree_function,
        pytree_mean,
        pytree_factor,
    )
    assert pytree_result.mean["field"].shape == (2,)
    assert pytree_result.mean["total"].shape == ()
    assert pytree_result.factor.event_size == 3
    assert pytree_result.cross_covariance.shape == (3, 3)

    factor = phx.uq.GaussianFactor(jnp.asarray([[0.4]]))
    compiled = jax.jit(
        lambda center: phx.uq.scaled_unscented_transform(
            lambda value: value**2,
            center,
            factor,
        )
    )
    compiled_result = compiled(jnp.asarray(0.3))
    centers = jnp.asarray([-0.5, 0.0, 0.7])
    mapped_means = jax.vmap(
        lambda center: (
            phx.uq.spherical_radial_cubature(
                lambda value: 2.0 * value + 1.0,
                center,
                factor,
            ).mean
        )
    )(centers)
    mean_gradient = jax.grad(lambda center: compiled(center).mean)(jnp.asarray(0.3))

    def covariance_from_root(root):
        local_factor = phx.uq.GaussianFactor(root.reshape((1, 1)))
        result = phx.uq.first_order_gaussian_transform(
            lambda value: value**2,
            jnp.asarray(0.7),
            local_factor,
        )
        return result.factor.covariance[0, 0]

    root = jnp.asarray(0.4)
    root_gradient = jax.grad(covariance_from_root)(root)
    rule_gradients = (
        jax.grad(
            lambda center: (
                phx.uq.spherical_radial_cubature(
                    lambda value: value**2, center, factor
                ).mean
            )
        )(jnp.asarray(0.3)),
        jax.grad(
            lambda center: (
                phx.uq.gauss_hermite_transform(
                    lambda value: value**2, center, factor, order=3
                ).mean
            )
        )(jnp.asarray(0.3)),
        jax.grad(
            lambda center: (
                phx.uq.first_order_gaussian_transform(
                    lambda value: value**2, center, factor
                ).mean
            )
        )(jnp.asarray(0.3)),
    )

    assert compiled_result.valid
    assert jnp.allclose(mapped_means, 2.0 * centers + 1.0)
    assert jnp.allclose(mean_gradient, 0.6)
    assert jnp.allclose(jnp.asarray(rule_gradients), jnp.full((3,), 0.6))
    assert jnp.allclose(root_gradient, 8.0 * 0.7**2 * root)


def test_gaussian_expectation_reuses_rules_without_materializing_output_covariance():
    mean, factor, matrix, offset = _linear_case()

    def function(value):
        transformed = matrix @ value + offset
        return {"field": transformed[:2], "total": jnp.sum(transformed)}

    expected = function(mean)
    for method in ("cubature", "unscented", "gauss-hermite"):
        result = phx.uq.gaussian_expectation(
            function,
            mean,
            factor,
            method=method,
            order=3,
        )
        assert result.successful
        assert jnp.allclose(result.value["field"], expected["field"], atol=2e-6)
        assert jnp.allclose(result.value["total"], expected["total"], atol=2e-6)
        assert result.input_dimension == 2
        assert result.output_dimension == 3
    assert (
        phx.uq.gaussian_expectation(function, mean, factor, method="cubature").point_count
        == 4
    )


def test_gaussian_expectation_polynomial_values_and_gradients_are_exact():
    mean = jnp.asarray(0.35)
    scale = jnp.asarray(0.6)

    def expectation(center, root):
        factor = phx.uq.GaussianFactor(root.reshape((1, 1)))
        return phx.uq.gaussian_expectation(
            lambda value: jnp.asarray([value**2, value**3]),
            center,
            factor,
            method="gauss-hermite",
            order=4,
        ).value

    value = jax.jit(expectation)(mean, scale)
    mean_gradient, scale_gradient = jax.grad(
        lambda center, root: expectation(center, root)[0],
        argnums=(0, 1),
    )(mean, scale)
    expected = jnp.asarray(
        [
            mean**2 + scale**2,
            mean**3 + 3.0 * mean * scale**2,
        ]
    )
    assert jnp.allclose(value, expected, atol=2e-6)
    assert jnp.allclose(mean_gradient, 2.0 * mean, atol=2e-6)
    assert jnp.allclose(scale_gradient, 2.0 * scale, atol=2e-6)


def test_gaussian_expectation_monte_carlo_is_keyed_and_zero_rank_is_exact():
    factor = phx.uq.GaussianFactor(jnp.asarray([[0.7]]))
    first = phx.uq.gaussian_expectation(
        lambda value: value**2,
        jnp.asarray(0.2),
        factor,
        method="monte-carlo",
        key=jr.key(71),
        num_samples=4_096,
    )
    second = phx.uq.gaussian_expectation(
        lambda value: value**2,
        jnp.asarray(0.2),
        factor,
        method="monte-carlo",
        key=jr.key(71),
        num_samples=4_096,
    )
    assert first.method_id == "fixed-sample-monte-carlo"
    assert first.point_count == 4_096
    assert jnp.array_equal(first.value, second.value)
    assert jnp.allclose(first.value, 0.2**2 + 0.7**2, atol=2.5e-2)
    with pytest.raises(ValueError, match="key is required"):
        phx.uq.gaussian_expectation(
            lambda value: value,
            jnp.asarray(0.0),
            factor,
            method="monte-carlo",
        )

    deterministic = phx.uq.gaussian_expectation(
        lambda value: {"value": 3.0 * value - 1.0},
        jnp.asarray([0.25, -0.5]),
        phx.uq.GaussianFactor(jnp.zeros((2, 0))),
        method="monte-carlo",
        key=jr.key(4),
        num_samples=128,
    )
    assert deterministic.point_count == 1
    assert jnp.array_equal(deterministic.value["value"], jnp.asarray([-0.25, -2.5]))


def test_gaussian_expectation_preserves_guards_and_nonfinite_status():
    factor = phx.uq.GaussianFactor(jnp.eye(2))
    with pytest.raises(ValueError, match="exceeds max_dimension"):
        phx.uq.gaussian_expectation(
            lambda value: value,
            jnp.zeros(2),
            factor,
            method="gauss-hermite",
            max_dimension=1,
        )
    with pytest.raises(ValueError, match="exceeds max_points"):
        phx.uq.gaussian_expectation(
            lambda value: value,
            jnp.zeros(2),
            factor,
            method="gauss-hermite",
            order=5,
            max_points=24,
        )
    nonfinite = phx.uq.gaussian_expectation(
        lambda value: jnp.asarray(jnp.nan),
        jnp.zeros(2),
        factor,
    )
    assert not nonfinite.valid
    assert nonfinite.status == phx.uq.NONLINEAR_GAUSSIAN_NONFINITE
