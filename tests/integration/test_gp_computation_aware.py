#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


def test_computation_aware_discrepancy_improves_physics_and_retains_uncertainty():
    points = jnp.linspace(0.0, 1.0, 24)
    true_coefficient = 1.35
    omitted_physics = 0.08 * jnp.sin(2.0 * jnp.pi * points)
    observations = true_coefficient * points + omitted_physics
    state = phx.uq.GaussianProcessLikelihoodState(
        kernel=phx.kernels.AmplitudeKernel(
            phx.kernels.Matern32Kernel(length_scale=0.22),
            0.12,
        ),
        noise_scale=0.02,
    )
    action_values = jr.normal(jr.key(17), points.shape)
    actions = phx.uq.BlockSparseGaussianProcessActionPolicy(action_values, 8)
    model = phx.uq.ComputationAwareGaussianProcessDiscrepancy(points, observations)
    exact_model = phx.uq.ExactGaussianProcessDiscrepancy(points, observations)

    def objective(coefficient, values):
        return model.elbo(
            coefficient * points,
            state=state,
            actions=phx.uq.BlockSparseGaussianProcessActionPolicy(values, 8),
        )

    wrong_coefficient = jnp.asarray(0.7)
    true_score = objective(true_coefficient, action_values)
    wrong_score, action_gradient = jax.value_and_grad(objective, argnums=1)(
        wrong_coefficient,
        action_values,
    )
    candidate_scores = jnp.stack(
        tuple(
            objective(wrong_coefficient, action_values + rate * action_gradient)
            for rate in (1e-5, 1e-4, 1e-3)
        )
    )

    factor = model.factor(state=state, actions=actions)
    query = jnp.linspace(0.0, 1.0, 41)
    physical_query = true_coefficient * query
    condition = factor.condition(model.residual(true_coefficient * points), query)
    predictive = condition.predictive_field(
        physical_query,
        jr.key(19),
        num_samples=32,
        observation_variance=state.noise_scale**2,
        sample_dim="draw",
    )
    corrected = physical_query + condition.mean
    truth = physical_query + 0.08 * jnp.sin(2.0 * jnp.pi * query)
    base_rmse = jnp.sqrt(jnp.mean((physical_query - truth) ** 2))
    corrected_rmse = jnp.sqrt(jnp.mean((corrected - truth) ** 2))

    exact_condition = exact_model.condition(
        true_coefficient * points,
        query,
        state=state,
    )
    conservative_difference = condition.covariance - exact_condition.covariance

    assert true_score > wrong_score
    assert jnp.max(candidate_scores) > wrong_score
    assert jnp.all(jnp.isfinite(action_gradient))
    assert corrected_rmse < base_rmse
    assert jnp.linalg.eigvalsh(conservative_difference).min() >= -1e-10
    assert predictive.samples.dims == ("draw", "point")
    assert predictive.conditional_variance is not None
    assert factor.diagnostics.structurally_sparse_actions
    assert factor.diagnostics.valid
