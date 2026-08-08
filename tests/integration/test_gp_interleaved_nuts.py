#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_interleaved_nuts_samples_dynamic_gp_likelihood_state():
    points = jnp.linspace(0.0, 1.0, 8)
    observations = 0.8 * points + 0.12 * jnp.sin(2.0 * jnp.pi * points)
    discrepancy = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)

    def physical_mean(parameters):
        return parameters["coefficient"] * points

    def state(parameters):
        return phx.uq.GaussianProcessLikelihoodState(
            kernel=phx.kernels.AmplitudeKernel(
                phx.kernels.Matern32Kernel(
                    length_scale=parameters["length_scale"],
                ),
                parameters["amplitude"],
            ),
            noise_scale=parameters["noise_scale"],
        )

    term = phx.uq.GaussianProcessMarginalLikelihood(
        discrepancy,
        physical_mean,
        state=state,
    )
    parameter_space = phx.uq.ParameterSpace(
        {
            "coefficient": jnp.asarray(0.7),
            "amplitude": jnp.log(jnp.asarray(0.15)),
            "length_scale": jnp.log(jnp.asarray(0.25)),
            "noise_scale": jnp.log(jnp.asarray(0.03)),
        },
        priors={
            "coefficient": phx.uq.Normal(0.8, 0.4),
            "amplitude": phx.uq.LogNormal(jnp.log(0.15), 0.5),
            "length_scale": phx.uq.LogNormal(jnp.log(0.25), 0.5),
            "noise_scale": phx.uq.LogNormal(jnp.log(0.03), 0.4),
        },
        bijectors={
            "coefficient": phx.uq.IdentityBijector(),
            "amplitude": phx.uq.ExpBijector(),
            "length_scale": phx.uq.ExpBijector(),
            "noise_scale": phx.uq.ExpBijector(),
        },
    )
    problem = phx.uq.PosteriorProblem.from_terms(parameter_space, (term,))
    gradient = jax.grad(problem.log_density)(parameter_space.initial)
    result = phx.uq.sample_nuts(
        problem,
        key=jr.key(902),
        num_chains=2,
        num_warmup=12,
        num_samples=4,
        initial_step_size=0.03,
        max_num_doublings=4,
        chain_method="interleaved",
    )

    assert all(jnp.all(jnp.isfinite(value)) for value in gradient.values())
    assert result.chain_method == "interleaved"
    assert result.log_density.shape == (2, 4)
    assert jnp.all(jnp.isfinite(result.log_density))
    assert all(
        leaf.shape[:2] == (2, 4) and jnp.all(jnp.isfinite(leaf))
        for leaf in jax.tree.leaves(result.samples)
    )
