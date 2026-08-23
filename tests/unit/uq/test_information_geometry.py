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
