import itertools
import math

import jax.numpy as jnp
import pytest

import phydrax as phx


def _standard_normal_moment(exponents):
    moment = 1
    for exponent in exponents:
        if exponent % 2:
            return 0.0
        moment *= math.prod(range(1, exponent, 2))
    return float(moment)


def _certify(rule):
    points = rule.prepared.points
    weights = rule.prepared.weights
    for exponents in itertools.product(
        range(rule.exact_degree + 1), repeat=rule.dimension
    ):
        if sum(exponents) > rule.exact_degree:
            continue
        powers = jnp.asarray(exponents)
        estimate = jnp.sum(weights * jnp.prod(points**powers, axis=1))
        assert jnp.allclose(
            estimate,
            _standard_normal_moment(exponents),
            rtol=1e-11,
            atol=1e-11,
        )


@pytest.mark.parametrize(
    ("dimension", "degree", "family"),
    (
        (1, 5, "auto"),
        (2, 5, "auto"),
        (3, 5, "auto"),
        (4, 3, "stroud-secrest-3-1"),
        (5, 3, "hadamard-3"),
    ),
)
def test_positive_gaussian_families_certify_total_degree_moments(
    dimension,
    degree,
    family,
):
    rule = phx.integration.GaussianCubatureRule(
        dimension,
        degree,
        family=family,
    )

    assert rule.dimension == dimension
    assert rule.exact_degree >= degree
    assert rule.prepared.reference_domain == "standard-normal"
    assert rule.prepared.integration_measure == "standard-normal"
    assert jnp.all(rule.prepared.weights > 0.0)
    assert jnp.allclose(jnp.sum(rule.prepared.weights), 1.0, atol=1e-14)
    _certify(rule)


def test_gaussian_rule_identity_and_capacity_are_static_and_explicit():
    first = phx.integration.GaussianCubatureRule(3, 5)
    replay = phx.integration.GaussianCubatureRule(3, 5)
    alternative = phx.integration.GaussianCubatureRule(3, 3)

    assert first.rule_id == replay.rule_id
    assert first.rule_id != alternative.rule_id
    assert first.storage_bytes == first.prepared.storage_bytes
    assert first.num_points == 14

    with pytest.raises(ValueError, match="exceeding maximum_points"):
        phx.integration.GaussianCubatureRule(3, 5, maximum_points=13)
    with pytest.raises(ValueError, match="degree at most five"):
        phx.integration.GaussianCubatureRule(2, 6)
    with pytest.raises(ValueError, match="dimension two"):
        phx.integration.GaussianCubatureRule(
            3,
            5,
            family="stroud-secrest-5-2",
        )


def test_scalar_gaussian_rule_integrates_a_transformed_probability_domain():
    probability = phx.domain.ProbabilityDomain(
        phx.uq.Normal(2.0, 3.0),
        label="z",
    )
    function = probability.Function("z")(lambda z: (z - 2.0) ** 4)

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(probability.component()),
        phx.integration.FixedQuadraturePlan(phx.integration.GaussianCubatureRule(1, 5)),
    )

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 243.0, atol=1e-11)


def test_grouped_gaussian_rule_preserves_coupled_probability_moments():
    x = phx.domain.ProbabilityDomain(phx.uq.Normal(1.0, 2.0), label="x")
    y = phx.domain.ProbabilityDomain(phx.uq.Normal(-2.0, 0.5), label="y")
    domain = phx.domain.ProductDomain(x, y)
    function = domain.Function("x", "y")(
        lambda x, y: ((x - 1.0) / 2.0) ** 2 * ((y + 2.0) / 0.5) ** 2
    )
    plan = phx.integration.ProductIntegrationPlan(
        {
            ("x", "y"): phx.integration.FixedQuadraturePlan(
                phx.integration.GaussianCubatureRule(2, 5)
            )
        }
    )

    estimate = phx.integration.integrate(
        function,
        phx.integration.over(domain.component()),
        plan,
    )

    assert estimate.successful
    assert jnp.allclose(jnp.asarray(estimate.value.data), 1.0, atol=1e-12)


def test_grouped_gaussian_rule_requires_matching_normal_reference_factors():
    x = phx.domain.ProbabilityDomain(phx.uq.Normal(0.0, 1.0), label="x")
    y = phx.domain.ScalarInterval(-1.0, 1.0, label="y")
    domain = phx.domain.ProductDomain(x, y)
    plan = phx.integration.ProductIntegrationPlan(
        {
            ("x", "y"): phx.integration.FixedQuadraturePlan(
                phx.integration.GaussianCubatureRule(2, 3)
            )
        }
    )

    with pytest.raises(TypeError, match="probability domains"):
        phx.integration.materialize(phx.integration.over(domain.component()), plan)
