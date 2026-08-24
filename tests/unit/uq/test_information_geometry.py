#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def test_bernoulli_information_geometry_matches_analytic_fisher_and_duality():
    family = phx.uq.BernoulliFamily()
    geometry = phx.uq.ExponentialFamilyInformationGeometry(family)
    natural = family.natural(jnp.asarray([0.0]))
    fisher = geometry.fisher_matrix(natural)
    gradient = geometry.natural_gradient(natural, jnp.asarray([1.0]))

    assert jnp.allclose(fisher, jnp.asarray([[0.25]]))
    assert jnp.allclose(gradient, jnp.asarray([4.0]))
    assert jnp.allclose(geometry.dual_coordinates(natural).values, jnp.asarray([0.5]))

    right = family.natural(jnp.asarray([0.4]))
    midpoint = geometry.exponential_interpolate(natural, right, 0.5)
    mixture = geometry.mixture_interpolate(natural, right, 0.5)
    assert jnp.allclose(midpoint.values, jnp.asarray([0.2]))
    assert bool(mixture.valid)
    assert geometry.kl_divergence(natural, right) >= 0.0


def _log_normalizer_bregman(family, left, right):
    right_mean = family.mean_from_natural(right)
    return (
        family.log_normalizer(left)
        - family.log_normalizer(right)
        - jnp.sum(right_mean.values * (left.values - right.values), axis=-1)
    )


def test_exponential_family_kl_has_documented_bregman_orientation():
    cases = (
        (
            phx.uq.BernoulliFamily(),
            jnp.asarray([-0.7]),
            jnp.asarray([1.1]),
        ),
        (
            phx.uq.NormalFamily(),
            jnp.asarray([0.2, -0.8]),
            jnp.asarray([-0.1, -0.5]),
        ),
    )
    for family, left_values, right_values in cases:
        geometry = phx.uq.ExponentialFamilyInformationGeometry(family)
        left = family.natural(left_values)
        right = family.natural(right_values)
        kl = geometry.kl_divergence(left, right)
        correctly_oriented = _log_normalizer_bregman(family, right, left)
        reversed_orientation = _log_normalizer_bregman(family, left, right)

        assert jnp.allclose(kl, correctly_oriented)
        assert not jnp.allclose(kl, reversed_orientation)
