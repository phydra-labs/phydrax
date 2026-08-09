#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import pytest

import phydrax as phx


def test_simplex_bijector_round_trip_shape_and_hausdorff_jacobian():
    bijector = phx.uq.SimplexBijector(4)
    raw = jnp.asarray([0.4, -0.7, 1.1])
    physical = bijector.forward(raw)
    jacobian = jax.jacobian(bijector.forward)(raw)
    expected_log_volume = 0.5 * jnp.linalg.slogdet(jacobian.T @ jacobian)[1]

    np.testing.assert_allclose(bijector.inverse(physical), raw, atol=3e-15)
    np.testing.assert_allclose(
        bijector.forward_log_det_jacobian(raw), expected_log_volume, atol=3e-15
    )
    np.testing.assert_allclose(jnp.sum(physical), 1.0, atol=2e-15)
    assert jnp.all(physical > 0.0)
    assert bijector.forward_shape((2, 3)) == (2, 4)
    assert bijector.inverse_shape((2, 4)) == (2, 3)


def test_dirichlet_density_uses_hausdorff_measure_and_normalizes():
    family = phx.uq.DirichletFamily(2)
    concentration = jnp.asarray([2.3, 1.7])
    natural = family.natural_from_concentration(concentration)
    coordinate = jnp.linspace(1e-6, 1.0 - 1e-6, 100_000)
    points = jnp.stack((coordinate, 1.0 - coordinate), axis=-1)
    standard = (
        jsp.special.gammaln(jnp.sum(concentration))
        - jnp.sum(jsp.special.gammaln(concentration))
        + jnp.sum((concentration - 1.0) * jnp.log(points), axis=-1)
    )
    hausdorff = family.log_prob(natural, points)

    np.testing.assert_allclose(hausdorff, standard - 0.5 * jnp.log(2.0), atol=3e-14)
    np.testing.assert_allclose(
        np.trapezoid(np.exp(hausdorff) * jnp.sqrt(2.0), coordinate),
        1.0,
        atol=2e-8,
    )
    assert family.signature.density_measure_kind == "hausdorff"


def test_dirichlet_duality_inverse_kl_and_fisher():
    family = phx.uq.DirichletFamily(4)
    concentration = jnp.asarray([0.4, 1.2, 2.5, 4.0])
    natural = family.natural_from_concentration(concentration)
    mean = family.mean_from_natural(natural)
    conversion = family.natural_from_mean(mean)
    gradient = jax.grad(lambda value: family.log_normalizer(family.natural(value)))(
        natural.values
    )
    direction = jnp.asarray([0.2, -0.4, 0.5, -0.1])
    hessian = jax.hessian(lambda value: family.log_normalizer(family.natural(value)))(
        natural.values
    )
    other_concentration = jnp.asarray([1.5, 0.8, 3.2, 2.0])
    other = family.natural_from_concentration(other_concentration)

    def log_beta(value):
        return jnp.sum(jsp.special.gammaln(value)) - jsp.special.gammaln(jnp.sum(value))

    expected_kl = (
        log_beta(other_concentration)
        - log_beta(concentration)
        + jnp.sum((concentration - other_concentration) * mean.values)
    )

    np.testing.assert_allclose(mean.values, gradient, atol=4e-14)
    np.testing.assert_allclose(
        conversion.natural.values, natural.values, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(
        family.fisher_action(natural, direction), hessian @ direction, atol=5e-14
    )
    np.testing.assert_allclose(family.kl_divergence(natural, natural), 0.0, atol=4e-14)
    np.testing.assert_allclose(
        family.kl_divergence(natural, other), expected_kl, atol=5e-14
    )
    assert bool(conversion.valid)
    assert conversion.method_id == "dirichlet-total-concentration-bisection"
    assert int(conversion.iterations) > 0


def test_dirichlet_inverse_has_implicit_reverse_derivative_and_supports_empty_batches():
    family = phx.uq.DirichletFamily(3)
    natural = family.natural_from_concentration(jnp.asarray([0.7, 1.8, 3.1]))
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
    empty = family.natural_from_mean(family.mean(jnp.empty((0, 3))))

    np.testing.assert_allclose(
        inverse_jacobian @ fisher, jnp.eye(3), rtol=3e-9, atol=3e-9
    )
    np.testing.assert_allclose(inverse_jacobian, forward_jacobian, rtol=2e-12, atol=2e-12)
    assert empty.natural.values.shape == (0, 3)
    assert empty.valid.shape == (0,)


def test_dirichlet_mean_domain_and_solver_failure_are_distinct():
    family = phx.uq.DirichletFamily(3)
    boundary_values = jnp.log(jnp.asarray([0.2, 0.3, 0.5]))
    boundary = family.mean_domain(family.mean(boundary_values))
    exterior = family.mean_domain(family.mean(boundary_values + 0.1))
    slow = phx.uq.DirichletFamily(3, atol=1e-14, rtol=0.0, max_iterations=1)
    source = family.mean_from_natural(
        family.natural_from_concentration(jnp.asarray([0.08, 1.5, 7.0]))
    )
    failed = slow.natural_from_mean(source)

    assert int(boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(exterior.status) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN
    assert int(failed.status) == phx.uq.EXPONENTIAL_FAMILY_NONCONVERGED
    assert not bool(failed.valid)
    assert jnp.isfinite(failed.residual)
    assert not bool(family.sufficient_statistics(jnp.asarray([0.0, 0.5, 0.5])).valid)
    assert not bool(family.sufficient_statistics(jnp.asarray([0.2, 0.2, 0.2])).valid)
    large_family = phx.uq.DirichletFamily(2048)
    off_simplex = jnp.full((2048,), jnp.asarray(129.0 / 262144.0, dtype=jnp.float32))
    assert not bool(large_family.sufficient_statistics(off_simplex).valid)


def test_dirichlet_sampling_projection_and_batched_inverse():
    family = phx.uq.DirichletFamily(3)
    concentration = jnp.asarray([1.5, 3.0, 5.5])
    law = family.law_from_concentration(concentration)
    samples = law.sample(jr.key(30), sample_shape=(40_000,))
    observations = jnp.asarray(
        [[0.2, 0.3, 0.5], [0.1, 0.7, 0.2], [0.4, 0.2, 0.4], [0.3, 0.5, 0.2]]
    )
    projected = phx.uq.fit_exponential_family(family, observations, sample_axes=0)
    expected_log = jnp.mean(jnp.log(observations), axis=0)
    batched_concentrations = jnp.asarray(
        [[0.3, 0.8, 2.0], [1.0, 1.0, 1.0], [4.0, 2.0, 7.0]]
    )
    batched_natural = family.natural_from_concentration(batched_concentrations)
    converted = jax.jit(
        lambda value: family.natural_from_mean(
            family.mean_from_natural(family.natural(value))
        )
    )(batched_natural.values)

    np.testing.assert_allclose(
        jnp.mean(samples, axis=0), concentration / jnp.sum(concentration), atol=0.012
    )
    np.testing.assert_allclose(
        projected.mean_coordinates.values, expected_log, atol=3e-14
    )
    np.testing.assert_allclose(
        converted.natural.values, batched_natural.values, rtol=2e-8, atol=2e-8
    )
    assert samples.shape == (40_000, 3)
    assert bool(projected.valid)
    assert jnp.all(converted.valid)


def test_dirichlet_prior_uses_distinct_raw_and_physical_shapes():
    family = phx.uq.DirichletFamily(3)
    prior = family.law_from_concentration(jnp.asarray([1.5, 2.0, 3.0]))
    bijector = phx.uq.SimplexBijector(3)
    initial = jnp.asarray([0.2, -0.4])
    space = phx.uq.ParameterSpace(initial, priors=prior, bijectors=bijector)
    physical = space.constrain(initial)
    reconstructed = space.unconstrain(physical)
    constrained_samples = space.sample_prior(jr.key(31), num_samples=9, constrained=True)
    raw_samples = space.sample_prior(jr.key(32), num_samples=7, constrained=False)
    problem = phx.uq.PosteriorProblem(space, lambda simplex: jnp.sum(jnp.log(simplex)))
    value, gradient = problem.validate()

    np.testing.assert_allclose(reconstructed, initial, atol=3e-15)
    np.testing.assert_allclose(jnp.sum(physical), 1.0, atol=2e-15)
    assert space.raw_shapes == ((2,),)
    assert space.physical_shapes == ((3,),)
    assert constrained_samples.shape == (9, 3)
    assert raw_samples.shape == (7, 2)
    assert jnp.allclose(jnp.sum(constrained_samples, axis=-1), 1.0)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(gradient))
    with pytest.raises(ValueError, match="supports only"):
        phx.uq.GaussianPriorWhitening.from_parameter_space(space)
