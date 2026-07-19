#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_whitened_ggn_laplace_matches_nuts_on_linear_gaussian_inverse_problem():
    design = jnp.array(
        [
            [1.0, 2.0, -0.5],
            [0.5, -1.0, 1.5],
            [2.0, 0.2, 0.3],
            [-0.4, 1.2, 1.0],
            [1.3, -0.7, 0.8],
        ]
    )
    target = jnp.array([0.4, -0.2, 1.0, 0.7, -0.5])
    noise_scale = 0.35
    prior_location = jnp.array([0.2, -0.1, 0.4])
    prior_scale = 1.7
    precision = design.T @ design / noise_scale**2 + jnp.eye(3) / prior_scale**2
    analytic_covariance = jnp.linalg.inv(precision)
    analytic_mean = analytic_covariance @ (
        design.T @ target / noise_scale**2 + prior_location / prior_scale**2
    )

    def residual(parameters):
        return (design @ parameters - target) / noise_scale

    space = phx.uq.ParameterSpace(
        analytic_mean,
        priors=phx.uq.Normal(0.2, prior_scale),
    )
    # Use a vector prior offset explicitly because Normal is currently shared per leaf.
    problem = phx.uq.PosteriorProblem(
        space,
        lambda parameters: (
            -0.5 * jnp.sum(residual(parameters) ** 2)
            - jnp.sum((parameters - prior_location) ** 2) / (2.0 * prior_scale**2)
            + jnp.sum((parameters - 0.2) ** 2) / (2.0 * prior_scale**2)
        ),
        gauss_newton_residual=residual,
    )
    structured = phx.uq.fit_laplace(
        problem,
        analytic_mean,
        curvature="full",
        likelihood_curvature="ggn",
    )
    nuts = phx.uq.sample_nuts(
        problem,
        key=jr.key(92),
        num_chains=4,
        num_warmup=300,
        num_samples=500,
        initial_step_size=0.2,
        target_acceptance_rate=0.9,
        chain_method="vectorized",
    )
    flat_draws = nuts.samples.reshape((-1, 3))
    nuts_mean = jnp.mean(flat_draws, axis=0)
    nuts_covariance = jnp.cov(flat_draws, rowvar=False)
    structured_covariance = jax.vmap(structured.covariance_vector_product)(jnp.eye(3))

    assert nuts.convergence_report(
        max_rhat=1.03,
        min_bulk_ess=150.0,
        min_tail_ess=150.0,
    ).passed
    assert jnp.allclose(nuts_mean, analytic_mean, atol=0.025)
    assert jnp.allclose(structured.map_position, nuts_mean, atol=0.025)
    assert jnp.allclose(structured_covariance, analytic_covariance, atol=2e-6)
    assert jnp.allclose(nuts_covariance, structured_covariance, rtol=0.18, atol=0.004)
