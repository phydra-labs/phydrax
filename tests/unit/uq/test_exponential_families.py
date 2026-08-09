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


FAMILY_COORDINATES = (
    (phx.uq.BernoulliFamily(), jnp.asarray([0.4])),
    (phx.uq.PoissonFamily(), jnp.asarray([jnp.log(2.3)])),
    (phx.uq.ExponentialRateFamily(), jnp.asarray([-1.7])),
    (
        phx.uq.NormalFamily(),
        jnp.asarray([-0.4 / 1.3**2, -0.5 / 1.3**2]),
    ),
)


@pytest.mark.parametrize(("family", "natural_values"), FAMILY_COORDINATES)
def test_family_duality_round_trip_kl_and_fisher_identities(family, natural_values):
    natural = family.natural(natural_values)
    mean = family.mean_from_natural(natural)
    gradient = jax.grad(lambda values: family.log_normalizer(family.natural(values)))(
        natural_values
    )
    conversion = family.natural_from_mean(mean)
    direction = jnp.linspace(0.2, 0.7, family.signature.dimension)
    fisher = family.fisher_action(natural, direction)
    hessian = jax.hessian(lambda values: family.log_normalizer(family.natural(values)))(
        natural_values
    )
    second_direction = jnp.linspace(-0.6, 0.3, family.signature.dimension)

    np.testing.assert_allclose(mean.values, gradient, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        conversion.natural.values,
        natural_values,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(fisher, hessian @ direction, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(
        jnp.vdot(direction, family.fisher_action(natural, second_direction)),
        jnp.vdot(second_direction, fisher),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(family.kl_divergence(natural, natural), 0.0, atol=2e-14)
    assert bool(conversion.valid)
    assert int(conversion.status) == phx.uq.EXPONENTIAL_FAMILY_SUCCESS


def test_normalized_log_probabilities_match_independent_formulas():
    probability = 0.3
    bernoulli = phx.uq.BernoulliFamily().law(
        jnp.asarray([jnp.log(probability) - jnp.log1p(-probability)])
    )
    binary = jnp.asarray([0.0, 1.0])
    expected_binary = binary * jnp.log(probability) + (1.0 - binary) * jnp.log1p(
        -probability
    )
    np.testing.assert_allclose(bernoulli.log_prob(binary), expected_binary, atol=2e-15)
    np.testing.assert_allclose(jnp.sum(jnp.exp(bernoulli.log_prob(binary))), 1.0)

    rate = 2.1
    poisson = phx.uq.PoissonFamily().law(jnp.asarray([jnp.log(rate)]))
    counts = jnp.arange(60.0)
    expected_counts = counts * jnp.log(rate) - rate - jsp.special.gammaln(counts + 1.0)
    np.testing.assert_allclose(poisson.log_prob(counts), expected_counts, atol=2e-14)
    np.testing.assert_allclose(
        jnp.sum(jnp.exp(poisson.log_prob(counts))), 1.0, atol=2e-14
    )

    exponential = phx.uq.ExponentialRateFamily().law(jnp.asarray([-1.4]))
    positive_grid = jnp.linspace(0.0, 25.0, 30_001)
    expected_exponential = jnp.log(1.4) - 1.4 * positive_grid
    np.testing.assert_allclose(
        exponential.log_prob(positive_grid), expected_exponential, atol=2e-15
    )
    np.testing.assert_allclose(
        np.trapezoid(np.exp(exponential.log_prob(positive_grid)), positive_grid),
        1.0,
        atol=2e-7,
    )

    location = -0.2
    scale = 1.3
    normal = phx.uq.NormalFamily().law(
        jnp.asarray([location / scale**2, -0.5 / scale**2])
    )
    real_grid = jnp.linspace(location - 10.0 * scale, location + 10.0 * scale, 40_001)
    expected_normal = jsp.stats.norm.logpdf(real_grid, location, scale)
    np.testing.assert_allclose(normal.log_prob(real_grid), expected_normal, atol=2e-14)
    np.testing.assert_allclose(
        np.trapezoid(np.exp(normal.log_prob(real_grid)), real_grid),
        1.0,
        atol=2e-12,
    )


def test_mean_and_natural_domains_distinguish_boundaries_and_exteriors():
    bernoulli = phx.uq.BernoulliFamily()
    bernoulli_boundary = bernoulli.mean_domain(
        bernoulli.mean(jnp.asarray([[0.0], [1.0]]))
    )
    bernoulli_exterior = bernoulli.mean_domain(bernoulli.mean(jnp.asarray([[-0.1]])))
    assert jnp.all(bernoulli_boundary.boundary)
    assert jnp.all(bernoulli_boundary.status == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY)
    assert (
        int(bernoulli_exterior.status[0]) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN
    )

    poisson = phx.uq.PoissonFamily()
    assert int(poisson.mean_domain(poisson.mean(jnp.asarray([[0.0]]))).status[0]) == (
        phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    )
    assert not bool(poisson.sufficient_statistics(jnp.asarray(1.5)).valid)
    assert not bool(poisson.sufficient_statistics(jnp.asarray(-1.0)).valid)

    exponential = phx.uq.ExponentialRateFamily()
    assert (
        int(
            exponential.natural_domain(exponential.natural(jnp.asarray([[0.0]]))).status[
                0
            ]
        )
        == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN
    )
    assert (
        int(exponential.mean_domain(exponential.mean(jnp.asarray([[0.0]]))).status[0])
        == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    )

    normal = phx.uq.NormalFamily()
    boundary = normal.mean_domain(normal.mean(jnp.asarray([[2.0, 4.0]])))
    exterior = normal.mean_domain(normal.mean(jnp.asarray([[2.0, 3.5]])))
    invalid_natural = normal.natural_domain(normal.natural(jnp.asarray([[0.0, 0.2]])))
    assert bool(boundary.boundary[0])
    assert int(boundary.status[0]) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(exterior.status[0]) == phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_MEAN_DOMAIN
    assert int(invalid_natural.status[0]) == (
        phx.uq.EXPONENTIAL_FAMILY_OUTSIDE_NATURAL_DOMAIN
    )

    nonfinite = bernoulli.natural_domain(bernoulli.natural(jnp.asarray([[jnp.nan]])))
    assert int(nonfinite.status[0]) == phx.uq.EXPONENTIAL_FAMILY_NONFINITE
    assert jnp.isneginf(bernoulli.log_prob(bernoulli.natural(jnp.asarray([0.0])), 2.0))


def test_family_signatures_prevent_cross_family_coordinate_use():
    bernoulli = phx.uq.BernoulliFamily()
    poisson = phx.uq.PoissonFamily()
    with pytest.raises(ValueError, match="signature"):
        poisson.log_normalizer(bernoulli.natural(jnp.asarray([0.2])))
    with pytest.raises(ValueError, match="signature"):
        bernoulli.kl_divergence(
            bernoulli.natural(jnp.asarray([0.2])),
            poisson.natural(jnp.asarray([0.2])),
        )
    with pytest.raises(ValueError, match="Unknown exponential-family status"):
        phx.uq.exponential_family_status_name(99)


def test_batched_laws_preserve_sample_batch_and_intrinsic_axes_under_transforms():
    family = phx.uq.BernoulliFamily()
    natural_values = jnp.asarray([[-1.0], [0.0], [1.0]], dtype=jnp.float32)
    observations = jnp.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=jnp.float32)
    natural = family.natural(natural_values)
    law = family.law(natural)
    compiled = jax.jit(
        lambda values, targets: family.log_prob(family.natural(values), targets)
    )(natural_values, observations)
    vmapped = jax.vmap(
        lambda values, target: family.log_prob(family.natural(values), target)
    )(natural_values, jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float32))
    gradient = jax.grad(
        lambda values: jnp.sum(family.log_prob(family.natural(values), observations))
    )(natural_values)
    samples = law.sample(jr.key(4), sample_shape=(7,))

    assert compiled.shape == (2, 3)
    assert vmapped.shape == (3,)
    assert gradient.shape == natural_values.shape
    assert samples.shape == (7, 3)
    assert compiled.dtype == jnp.float32
    assert jnp.all(jnp.isfinite(compiled))
    assert jnp.all(jnp.isfinite(gradient))


@pytest.mark.parametrize(
    ("law", "expected_mean", "expected_variance"),
    (
        (
            phx.uq.BernoulliFamily().law(jnp.asarray([jnp.log(0.35 / 0.65)])),
            0.35,
            0.35 * 0.65,
        ),
        (phx.uq.PoissonFamily().law(jnp.asarray([jnp.log(1.7)])), 1.7, 1.7),
        (
            phx.uq.ExponentialRateFamily().law(jnp.asarray([-1.4])),
            1.0 / 1.4,
            1.0 / 1.4**2,
        ),
        (
            phx.uq.NormalFamily().law(jnp.asarray([-0.2 / 1.1**2, -0.5 / 1.1**2])),
            -0.2,
            1.1**2,
        ),
    ),
)
def test_family_sampling_matches_declared_moments(law, expected_mean, expected_variance):
    samples = law.sample(jr.key(17), sample_shape=(12_000,))
    np.testing.assert_allclose(jnp.mean(samples), expected_mean, atol=0.04, rtol=0.04)
    np.testing.assert_allclose(jnp.var(samples), expected_variance, atol=0.06, rtol=0.08)
    assert isinstance(law, phx.uq.AbstractProbabilityLaw)


def test_log_weighted_projection_is_mergeable_and_retains_batch_axes():
    family = phx.uq.BernoulliFamily()
    observations = jnp.asarray([0.0, 1.0, 1.0, 0.0, 1.0, 1.0])
    log_weights = jnp.asarray([-0.7, 0.2, 1.1, -1.3, 0.6, -0.2])
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
    expected_mean = jnp.sum(normalized * observations)

    np.testing.assert_allclose(one_shot.mean_coordinates.values[0], expected_mean)
    np.testing.assert_allclose(
        merged.mean_coordinates.values,
        one_shot.mean_coordinates.values,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        merged.law.natural.values, one_shot.law.natural.values, atol=2e-15
    )
    np.testing.assert_allclose(
        merged.diagnostics.entropy, one_shot.diagnostics.entropy, atol=2e-15
    )
    assert bool(one_shot.valid)
    assert bool(merged.valid)

    batched_observations = jnp.asarray([[0.0, 1.0], [1.0, 1.0], [0.0, 1.0], [1.0, 0.0]])
    batched = phx.uq.fit_exponential_family(
        family,
        batched_observations,
        sample_axes=0,
    )
    np.testing.assert_allclose(batched.mean_coordinates.values[:, 0], [0.5, 0.75])
    assert batched.valid.shape == (2,)
    assert jnp.all(batched.valid)


def test_projection_reports_invalid_inputs_zero_weight_and_boundary_mles():
    family = phx.uq.BernoulliFamily()
    masked = phx.uq.project_exponential_family(
        family,
        jnp.asarray([0.0, 2.0, 1.0]),
        mask=jnp.asarray([True, False, True]),
    )
    zero_weight_nonfinite = phx.uq.project_exponential_family(
        family,
        jnp.asarray([0.0, jnp.nan, 1.0]),
        log_weights=jnp.asarray([0.0, -jnp.inf, 0.0]),
    )
    invalid_event = phx.uq.project_exponential_family(
        family, jnp.asarray([0.0, 2.0, 1.0])
    )
    nonfinite_event = phx.uq.project_exponential_family(
        family, jnp.asarray([0.0, jnp.nan, 1.0])
    )
    invalid_weight = phx.uq.project_exponential_family(
        family,
        jnp.asarray([0.0, 1.0]),
        log_weights=jnp.asarray([0.0, jnp.nan]),
    )
    no_weight = phx.uq.project_exponential_family(
        family,
        jnp.asarray([0.0, 1.0]),
        log_weights=jnp.asarray([-jnp.inf, -jnp.inf]),
    )
    bernoulli_boundary = phx.uq.fit_exponential_family(family, jnp.zeros((4,)))
    poisson_boundary = phx.uq.fit_exponential_family(
        phx.uq.PoissonFamily(), jnp.zeros((4,))
    )
    normal_boundary = phx.uq.fit_exponential_family(phx.uq.NormalFamily(), jnp.ones((4,)))

    assert bool(masked.valid)
    assert bool(zero_weight_nonfinite.valid)
    assert int(invalid_event.status) == phx.uq.EXPONENTIAL_FAMILY_INVALID_EVENT
    assert int(nonfinite_event.status) == phx.uq.EXPONENTIAL_FAMILY_NONFINITE
    assert int(invalid_weight.status) == phx.uq.EXPONENTIAL_FAMILY_NONFINITE
    assert int(no_weight.status) == phx.uq.EXPONENTIAL_FAMILY_INSUFFICIENT_WEIGHT
    assert int(bernoulli_boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(poisson_boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert int(normal_boundary.status) == phx.uq.EXPONENTIAL_FAMILY_MEAN_BOUNDARY
    assert not bool(invalid_event.valid)
    assert not bool(no_weight.valid)


def test_scalar_family_likelihood_delegates_normalized_density_and_sampling():
    family = phx.uq.PoissonFamily()
    likelihood = phx.uq.ScalarNaturalExponentialFamilyLikelihood(family)
    location = jnp.asarray([jnp.log(1.2), jnp.log(2.5)])
    targets = jnp.asarray([0.0, 3.0])
    expected = family.log_prob(family.natural(location[..., None]), targets)

    np.testing.assert_allclose(likelihood.log_prob(location, targets), expected)
    assert likelihood.sample(jr.key(8), location).shape == location.shape
    with pytest.raises(TypeError, match="unknown parameters"):
        likelihood.log_prob(location, targets, scale=1.0)
    with pytest.raises(ValueError, match="one natural coordinate"):
        phx.uq.ScalarNaturalExponentialFamilyLikelihood(phx.uq.NormalFamily())


def test_exponential_family_laws_are_posterior_prior_leaves_with_shape_semantics():
    scalar_prior = phx.uq.BernoulliFamily().law(jnp.asarray([0.3]))
    scalar_space = phx.uq.ParameterSpace(
        jnp.zeros((3,)),
        priors=scalar_prior,
    )
    scalar_samples = scalar_space.sample_prior(
        jr.key(31), num_samples=5, constrained=True
    )
    np.testing.assert_allclose(
        scalar_space.log_prior(jnp.asarray([0.0, 1.0, 0.0])),
        jnp.sum(scalar_prior.log_prob(jnp.asarray([0.0, 1.0, 0.0]))),
    )
    assert scalar_samples.shape == (5, 3)

    shaped_prior = phx.uq.PoissonFamily().law(
        jnp.asarray([[jnp.log(0.8)], [jnp.log(1.5)], [jnp.log(2.2)]])
    )
    shaped_space = phx.uq.ParameterSpace(
        jnp.ones((3,)),
        priors=shaped_prior,
    )
    shaped_samples = shaped_space.sample_prior(
        jr.key(32), num_samples=7, constrained=True
    )
    assert shaped_samples.shape == (7, 3)

    with pytest.raises(ValueError, match=r"batch_shape \+ event_shape"):
        phx.uq.ParameterSpace(jnp.ones((2,)), priors=shaped_prior)
    with pytest.raises(ValueError, match="supports only Normal/Identity"):
        phx.uq.GaussianPriorWhitening.from_parameter_space(scalar_space)


def test_existing_scalar_distributions_implement_common_probability_law():
    laws = (
        phx.uq.Uniform(-1.0, 2.0),
        phx.uq.Normal(0.0, 1.0),
        phx.uq.LogNormal(0.0, 0.5),
        phx.uq.EmpiricalDistribution(jnp.asarray([0.0, 1.0]), jnp.asarray([0.4, 0.6])),
    )
    expected_measures = ("lebesgue", "lebesgue", "lebesgue", "counting")
    for law, expected_measure in zip(laws, expected_measures, strict=True):
        assert isinstance(law, phx.uq.AbstractProbabilityLaw)
        assert law.event_shape == ()
        assert law.batch_shape == ()
        assert law.density_measure_kind == expected_measure
