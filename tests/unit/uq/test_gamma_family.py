#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np

import phydrax as phx


def test_gamma_density_duality_kl_and_fisher_are_exact():
    family = phx.uq.GammaFamily()
    natural = family.natural_from_shape_rate(2.7, 1.4)
    mean = family.mean_from_natural(natural)
    gradient = jax.grad(lambda values: family.log_normalizer(family.natural(values)))(
        natural.values
    )
    direction = jnp.asarray([0.4, -0.3])
    hessian = jax.hessian(lambda values: family.log_normalizer(family.natural(values)))(
        natural.values
    )
    conversion = family.natural_from_mean(mean)
    grid = jnp.linspace(1e-6, 30.0, 100_000)
    expected = jsp.stats.gamma.logpdf(grid, 2.7, scale=1.0 / 1.4)
    other_shape = 1.4
    other_rate = 0.7
    other = family.natural_from_shape_rate(other_shape, other_rate)
    expected_kl = (
        2.7 * jnp.log(1.4)
        - other_shape * jnp.log(other_rate)
        - jsp.special.gammaln(2.7)
        + jsp.special.gammaln(other_shape)
        + (2.7 - other_shape) * (jsp.special.digamma(2.7) - jnp.log(1.4))
        + (other_rate - 1.4) * 2.7 / 1.4
    )

    np.testing.assert_allclose(family.log_prob(natural, grid), expected, atol=3e-14)
    np.testing.assert_allclose(
        np.trapezoid(np.exp(family.log_prob(natural, grid)), grid), 1.0, atol=2e-7
    )
    np.testing.assert_allclose(mean.values, gradient, atol=3e-14)
    np.testing.assert_allclose(conversion.natural.values, natural.values, atol=5e-10)
    np.testing.assert_allclose(
        family.fisher_action(natural, direction), hessian @ direction, atol=3e-14
    )
    np.testing.assert_allclose(family.kl_divergence(natural, natural), 0.0, atol=3e-14)
    np.testing.assert_allclose(
        family.kl_divergence(natural, other), expected_kl, atol=3e-14
    )
    assert bool(conversion.valid)
    assert conversion.method_id == "gamma-safeguarded-newton"
    assert int(conversion.iterations) > 0


def test_gamma_domains_distinguish_numerical_failure_boundary_and_exterior():
    family = phx.uq.GammaFamily()
    boundary = family.mean_domain(family.mean(jnp.asarray([jnp.log(2.0), 2.0])))
    exterior = family.mean_domain(family.mean(jnp.asarray([1.0, 2.0])))
    boundary_natural = family.natural_domain(family.natural(jnp.asarray([-1.0, -1.0])))
    exterior_natural = family.natural_domain(family.natural(jnp.asarray([-1.1, -1.0])))
    nonconvergent = phx.uq.GammaFamily(atol=1e-14, rtol=0.0, max_iterations=1)
    source = family.mean_from_natural(family.natural_from_shape_rate(0.03, 2.0))
    failed = nonconvergent.natural_from_mean(source)

    assert int(boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(exterior.status) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN
    assert bool(boundary_natural.boundary)
    assert not bool(exterior_natural.boundary)
    assert int(boundary_natural.status) == (
        phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN
    )
    assert int(exterior_natural.status) == (
        phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN
    )
    assert int(failed.status) == phx.uq.EXPONENTIAL_FAMILY_NONCONVERGED
    assert phx.uq.exponential_family_status_name(int(failed.status)) == "nonconverged"
    assert not bool(failed.valid)
    assert int(failed.iterations) == 1
    assert jnp.isfinite(failed.residual)
    assert not bool(family.sufficient_statistics(0.0).valid)
    assert not bool(family.sufficient_statistics(-1.0).valid)


def test_gamma_batched_inverse_is_jittable_and_batch_local():
    family = phx.uq.GammaFamily()
    shapes = jnp.asarray([0.08, 0.5, 2.0, 25.0], dtype=jnp.float32)
    rates = jnp.asarray([3.0, 0.7, 1.2, 4.0], dtype=jnp.float32)
    natural = family.natural_from_shape_rate(shapes, rates)
    means = family.mean_from_natural(natural)
    converted = jax.jit(lambda values: family.natural_from_mean(family.mean(values)))(
        means.values
    )

    np.testing.assert_allclose(
        converted.natural.values, natural.values, rtol=8e-5, atol=5e-5
    )
    assert converted.valid.shape == shapes.shape
    assert jnp.all(converted.valid)
    assert jnp.all(converted.iterations >= 0)
    assert jnp.any(converted.iterations > 0)


def test_gamma_inverse_has_implicit_reverse_derivative_and_supports_empty_batches():
    family = phx.uq.GammaFamily()
    natural = family.natural_from_shape_rate(2.0, 1.3)
    mean = family.mean_from_natural(natural)
    converted = family.natural_from_mean(mean)
    inverse_jacobian = jax.jacrev(
        lambda value: family.natural_from_mean(family.mean(value)).natural.values
    )(mean.values)
    forward_jacobian = jax.jacfwd(
        lambda value: family.natural_from_mean(family.mean(value)).natural.values
    )(mean.values)
    fisher = jax.jacfwd(
        lambda value: family.mean_from_natural(family.natural(value)).values
    )(converted.natural.values)
    empty = family.natural_from_mean(family.mean(jnp.empty((0, 2))))

    np.testing.assert_allclose(
        inverse_jacobian @ fisher, jnp.eye(2), rtol=2e-10, atol=2e-10
    )
    np.testing.assert_allclose(inverse_jacobian, forward_jacobian, rtol=2e-12, atol=2e-12)
    assert empty.natural.values.shape == (0, 2)
    assert empty.valid.shape == (0,)


def test_gamma_sampling_and_weighted_projection_recover_sufficient_statistics():
    family = phx.uq.GammaFamily()
    law = family.law_from_shape_rate(3.5, 1.8)
    samples = law.sample(jr.key(10), sample_shape=(40_000,))
    expected_mean = 3.5 / 1.8
    expected_log = jsp.special.digamma(3.5) - jnp.log(1.8)
    observations = jnp.asarray([0.4, 0.8, 1.3, 2.1, 3.4, 5.5])
    log_weights = jnp.asarray([-0.4, 0.2, 0.8, -0.1, 0.4, -0.7])
    one_shot = phx.uq.project_exponential_family(
        family, observations, log_weights=log_weights
    )
    left = phx.uq.ExponentialFamilyProjectionAccumulator.from_log_weights(
        family, observations[:3], log_weights[:3]
    )
    right = phx.uq.ExponentialFamilyProjectionAccumulator.from_log_weights(
        family, observations[3:], log_weights[3:]
    )
    merged = left.merge(right).finalize()
    normalized = jax.nn.softmax(log_weights)
    expected_projection = jnp.asarray(
        [jnp.sum(normalized * jnp.log(observations)), jnp.sum(normalized * observations)]
    )

    np.testing.assert_allclose(jnp.mean(samples), expected_mean, rtol=0.02)
    np.testing.assert_allclose(jnp.mean(jnp.log(samples)), expected_log, atol=0.015)
    np.testing.assert_allclose(
        one_shot.mean_coordinates.values, expected_projection, atol=2e-14
    )
    np.testing.assert_allclose(
        merged.law.natural.values, one_shot.law.natural.values, atol=2e-10
    )
    assert samples.shape == (40_000,)
    assert bool(one_shot.valid)
    assert bool(merged.valid)


def test_gamma_law_is_a_positive_posterior_prior():
    prior = phx.uq.GammaFamily().law_from_shape_rate(2.5, 1.2)
    initial = jnp.log(jnp.asarray(1.7))
    space = phx.uq.ParameterSpace(
        initial,
        priors=prior,
        bijectors=phx.uq.ExpBijector(),
    )
    physical = space.constrain(initial)
    expected = prior.log_prob(physical) + initial
    samples = space.sample_prior(jr.key(14), num_samples=8, constrained=False)

    np.testing.assert_allclose(
        space.log_prior(physical) + space.log_abs_det_jacobian(initial),
        expected,
        atol=2e-14,
    )
    np.testing.assert_allclose(space.constrain(samples), jnp.exp(samples), atol=2e-14)
    assert samples.shape == (8,)
    assert jnp.all(jnp.isfinite(samples))
