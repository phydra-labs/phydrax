#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy as jsp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx
from phydrax._exponential_family._symmetric_coordinates import smat, svec


def test_symmetric_coordinates_preserve_frobenius_geometry():
    left = jnp.asarray([[1.2, -0.4, 0.7], [-0.4, 2.0, 0.3], [0.7, 0.3, -0.5]])
    right = jnp.asarray([[0.2, 0.8, -0.1], [0.8, -1.0, 0.6], [-0.1, 0.6, 1.4]])

    np.testing.assert_allclose(smat(svec(left)), left, atol=2e-15)
    np.testing.assert_allclose(
        jnp.vdot(svec(left), svec(right)), jnp.sum(left * right), atol=2e-15
    )
    integral = jnp.asarray([[0, 1], [1, 0]], dtype=jnp.int32)
    integral_packed = svec(integral)
    np.testing.assert_allclose(
        jnp.vdot(integral_packed, integral_packed),
        jnp.sum(integral * integral),
        atol=2e-15,
    )
    np.testing.assert_allclose(smat(integral_packed), integral, atol=2e-15)
    assert jnp.issubdtype(integral_packed.dtype, jnp.floating)
    with pytest.raises(ValueError, match="triangular"):
        smat(jnp.ones((5,)))


def test_multivariate_normal_density_duality_and_fisher_match_references():
    family = phx.uq.MultivariateNormalFamily(3)
    location = jnp.asarray([0.3, -0.8, 0.5])
    covariance = jnp.asarray([[1.4, 0.2, -0.1], [0.2, 0.9, 0.3], [-0.1, 0.3, 1.1]])
    natural = family.natural_from_location_covariance(location, covariance)
    mean = family.mean_from_natural(natural)
    conversion = family.natural_from_mean(mean)
    points = jnp.asarray([[0.2, -0.4, 1.0], [-1.0, 0.3, 0.7], [0.8, -1.2, -0.5]])
    expected = jsp.stats.multivariate_normal.logpdf(points, location, covariance)
    gradient = jax.grad(lambda value: family.log_normalizer(family.natural(value)))(
        natural.values
    )
    direction = jnp.linspace(-0.4, 0.6, family.signature.dimension)
    hessian = jax.hessian(lambda value: family.log_normalizer(family.natural(value)))(
        natural.values
    )

    other_location = jnp.asarray([-0.4, 0.1, 0.9])
    other_covariance = jnp.asarray(
        [[0.8, -0.1, 0.2], [-0.1, 1.6, 0.05], [0.2, 0.05, 1.3]]
    )
    other = family.natural_from_location_covariance(other_location, other_covariance)
    other_precision = jnp.linalg.inv(other_covariance)
    displacement = other_location - location
    expected_kl = 0.5 * (
        jnp.trace(other_precision @ covariance)
        + displacement @ other_precision @ displacement
        - family.event_size
        + jnp.linalg.slogdet(other_covariance)[1]
        - jnp.linalg.slogdet(covariance)[1]
    )
    np.testing.assert_allclose(family.log_prob(natural, points), expected, atol=3e-14)
    np.testing.assert_allclose(mean.values, gradient, atol=4e-14)
    np.testing.assert_allclose(conversion.natural.values, natural.values, atol=5e-14)
    np.testing.assert_allclose(
        family.fisher_action(natural, direction), hessian @ direction, atol=8e-14
    )
    np.testing.assert_allclose(family.kl_divergence(natural, natural), 0.0, atol=3e-14)
    np.testing.assert_allclose(
        family.kl_divergence(natural, other), expected_kl, atol=5e-14
    )
    assert family.signature.dimension == 9
    assert bool(conversion.valid)


def test_multivariate_normal_sampling_and_projection_recover_moments():
    family = phx.uq.MultivariateNormalFamily(2)
    location = jnp.asarray([0.4, -0.7])
    covariance = jnp.asarray([[1.3, 0.45], [0.45, 0.8]])
    law = family.law_from_location_covariance(location, covariance)
    samples = law.sample(jr.key(20), sample_shape=(50_000,))
    observations = jnp.asarray(
        [[-1.0, 0.2], [0.3, -0.4], [1.2, 0.8], [0.5, -1.1], [2.0, 0.1]]
    )
    log_weights = jnp.asarray([-0.7, 0.2, 0.8, -0.1, 0.4])
    projected = phx.uq.project_exponential_family(
        family, observations, log_weights=log_weights
    )
    normalized = jax.nn.softmax(log_weights)
    expected_location = normalized @ observations
    centered = observations - expected_location
    expected_covariance = oe.contract("n,ni,nj->ij", normalized, centered, centered)
    actual_location, actual_covariance = family.location_covariance_from_natural(
        projected.law.natural
    )

    np.testing.assert_allclose(jnp.mean(samples, axis=0), location, atol=0.018)
    np.testing.assert_allclose(
        jnp.cov(samples, rowvar=False, bias=True), covariance, atol=0.025
    )
    np.testing.assert_allclose(actual_location, expected_location, atol=3e-14)
    np.testing.assert_allclose(actual_covariance, expected_covariance, atol=5e-14)
    assert samples.shape == (50_000, 2)
    assert bool(projected.valid)


def test_multivariate_normal_domain_reports_singular_empirical_covariance():
    family = phx.uq.MultivariateNormalFamily(2)
    singular = phx.uq.fit_exponential_family(
        family,
        jnp.asarray([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]),
        sample_axes=0,
    )
    indefinite_mean = family.mean(jnp.asarray([0.0, 0.0, 1.0, 0.0, -1.0]))
    exterior = family.mean_domain(indefinite_mean)

    assert not bool(singular.valid)
    assert int(singular.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(exterior.status) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN


def test_multivariate_normal_batches_jit_and_preserve_axes():
    family = phx.uq.MultivariateNormalFamily(2)
    locations = jnp.asarray([[0.0, 0.5], [1.0, -0.2]])
    covariances = jnp.asarray([[[1.0, 0.1], [0.1, 0.7]], [[0.8, -0.2], [-0.2, 1.4]]])
    natural = family.natural_from_location_covariance(locations, covariances)
    recovered = jax.jit(
        lambda value: family.natural_from_mean(
            family.mean_from_natural(family.natural(value))
        )
    )(natural.values)
    samples = family.sample(jr.key(21), natural, sample_shape=(7, 3))

    np.testing.assert_allclose(recovered.natural.values, natural.values, atol=8e-14)
    assert recovered.valid.shape == (2,)
    assert samples.shape == (7, 3, 2, 2)
    assert family.fisher_action(natural, jnp.ones_like(natural.values)).shape == (2, 5)


def test_multivariate_normal_validation_is_scale_relative_batch_local_and_empty_safe():
    scalar_family = phx.uq.MultivariateNormalFamily(1)
    scalar_location = jnp.asarray([0.0])
    for variance in (1.0e-15, 1.0e15):
        natural = scalar_family.natural_from_location_covariance(
            scalar_location, jnp.asarray([[variance]])
        )
        converted = scalar_family.natural_from_mean(
            scalar_family.mean_from_natural(natural)
        )
        np.testing.assert_allclose(
            converted.natural.values, natural.values, rtol=3e-14, atol=0.0
        )
        assert bool(converted.valid)

    family = phx.uq.MultivariateNormalFamily(2)
    locations = jnp.zeros((2, 2))
    nonsymmetric = jnp.asarray(
        [
            [[1.0e13, 0.0], [0.0, 1.0e13]],
            [[1.0, 0.1], [0.0, 1.0]],
        ]
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="symmetric"):
        family.natural_from_location_covariance(locations, nonsymmetric)

    empty_natural = family.natural_from_location_covariance(
        jnp.empty((0, 2)), jnp.empty((0, 2, 2))
    )
    empty_conversion = family.natural_from_mean(
        family.mean(jnp.empty((0, family.signature.dimension)))
    )
    assert empty_natural.values.shape == (0, family.signature.dimension)
    assert empty_conversion.natural.values.shape == (
        0,
        family.signature.dimension,
    )
    assert empty_conversion.valid.shape == (0,)


def test_multivariate_normal_bridge_reuses_gaussian_factor_without_jitter():
    family = phx.uq.MultivariateNormalFamily(2)
    location = jnp.asarray([0.3, -0.6])
    covariance = jnp.asarray([[1.1, 0.35], [0.35, 0.7]])
    law = family.law_from_location_covariance(location, covariance)
    recovered_location, factor = phx.uq.gaussian_factor_from_multivariate_normal(law)
    conversion = phx.uq.multivariate_normal_from_gaussian_factor(
        family, recovered_location, factor
    )
    singular_factor = phx.uq.GaussianFactor(jnp.asarray([[1.0], [2.0]]))
    boundary = phx.uq.multivariate_normal_from_gaussian_factor(
        family, jnp.zeros(2), singular_factor
    )

    np.testing.assert_allclose(recovered_location, location, atol=3e-14)
    np.testing.assert_allclose(factor.covariance, covariance, atol=5e-14)
    np.testing.assert_allclose(conversion.natural.values, law.natural.values, atol=8e-14)
    assert bool(factor.valid)
    assert factor.regularization == 0.0
    assert bool(conversion.valid)
    assert not bool(boundary.valid)
    assert int(boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
