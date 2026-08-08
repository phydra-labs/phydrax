#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp

import phydrax as phx


def test_operator_conditioned_gp_identifies_diffusion_coefficient():
    value_points = jnp.linspace(0.0, 1.0, 8)
    operator_points = jnp.linspace(0.05, 0.95, 10)
    true_diffusion = 1.7

    def field(points):
        return jnp.sin(jnp.pi * points)

    forcing = true_diffusion * jnp.pi**2 * field(operator_points)
    value = phx.uq.value_functional(1)
    laplacian = phx.uq.laplacian_functional(1)
    state = phx.uq.FunctionalGaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.SquaredExponentialKernel(length_scale=0.25),
            0.02,
        ),
        noise_scale=jnp.array([0.005, 0.02]),
    )

    def log_likelihood(diffusion):
        blocks = (
            phx.uq.FunctionalObservationBlock(
                value_points,
                value,
                name="field-observations",
            ),
            phx.uq.FunctionalObservationBlock(
                operator_points,
                -diffusion * laplacian,
                name="elliptic-operator",
            ),
        )
        discrepancy = phx.uq.FunctionalGaussianProcessDiscrepancy(
            blocks,
            (field(value_points), forcing),
        )
        physical_mean = (
            field(value_points),
            diffusion * jnp.pi**2 * field(operator_points),
        )
        return discrepancy.log_marginal_likelihood(physical_mean, state=state)

    candidates = jnp.linspace(0.7, 2.7, 21)
    scores = jax.vmap(log_likelihood)(candidates)
    selected = candidates[jnp.argmax(scores)]
    wrong_diffusion = jnp.asarray(1.0)

    assert jnp.allclose(selected, true_diffusion, atol=0.05)
    assert log_likelihood(true_diffusion) > log_likelihood(wrong_diffusion)
    assert jax.grad(log_likelihood)(wrong_diffusion) > 0.0
    assert jnp.all(jnp.isfinite(scores))
