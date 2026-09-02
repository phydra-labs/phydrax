#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import blackjax
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx
from phydrax.uq._causal_hmc import build_causal_hmc_kernel


def _logdensity(position):
    precision = jnp.asarray([[4.0, 1.2], [1.2, 2.5]], dtype=position.dtype)
    center = jnp.asarray([0.4, -0.7], dtype=position.dtype)
    residual = position - center
    return -0.5 * residual @ precision @ residual


def _posterior_problem():
    return phx.uq.PosteriorProblem(
        phx.uq.ParameterSpace(
            jnp.zeros((2,)),
            priors=phx.uq.Normal(0.0, 3.0),
        ),
        _logdensity,
    )


@pytest.mark.parametrize("block_size", (3, 8))
def test_dense_causal_hmc_matches_blackjax_endpoint_energy_and_decision(block_size):
    state = blackjax.hmc.init(jnp.asarray([0.3, -0.2]), _logdensity)
    key = jax.random.key(11)
    step_size = jnp.asarray(0.1)
    inverse_mass = jnp.asarray([1.0, 0.8])
    steps = 8
    sequential_state, sequential_info = blackjax.hmc.build_kernel()(
        key,
        state,
        _logdensity,
        step_size,
        inverse_mass,
        steps,
    )
    config = phx.uq.CausalHMCConfig(
        linearization="dense-exact",
        trajectory_block_size=block_size,
        absolute_residual=1e-11,
        relative_residual=1e-11,
        maximum_outer_iterations=12,
    )
    kernel = build_causal_hmc_kernel(config)
    causal_state, causal_info = jax.jit(
        lambda current_key, current_state: kernel(
            current_key,
            current_state,
            _logdensity,
            step_size,
            inverse_mass,
            steps,
        )
    )(key, state)

    assert bool(causal_info.causal_converged)
    assert not bool(causal_info.causal_fallback_used)
    assert jnp.allclose(
        causal_info.proposal.position,
        sequential_info.proposal.position,
        atol=2e-11,
        rtol=2e-11,
    )
    assert jnp.allclose(
        causal_info.proposal.momentum,
        sequential_info.proposal.momentum,
        atol=2e-11,
        rtol=2e-11,
    )
    assert jnp.allclose(causal_info.energy, sequential_info.energy, atol=2e-11)
    assert jnp.allclose(
        causal_info.acceptance_rate,
        sequential_info.acceptance_rate,
        atol=2e-11,
    )
    assert bool(causal_info.is_accepted) == bool(sequential_info.is_accepted)
    assert jnp.allclose(causal_state.position, sequential_state.position, atol=2e-11)


def test_pair_hutchinson_causal_hmc_converges_to_sequential_trace():
    state = blackjax.hmc.init(jnp.asarray([0.3, -0.2]), _logdensity)
    key = jax.random.key(12)
    step_size = jnp.asarray(0.08)
    inverse_mass = jnp.asarray([1.0, 0.8])
    steps = 10
    _, sequential_info = blackjax.hmc.build_kernel()(
        key,
        state,
        _logdensity,
        step_size,
        inverse_mass,
        steps,
    )
    config = phx.uq.CausalHMCConfig(
        linearization="pair-hutchinson",
        probe_count=4,
        trajectory_block_size=6,
        absolute_residual=1e-9,
        relative_residual=1e-9,
        maximum_outer_iterations=20,
    )
    _, causal_info = build_causal_hmc_kernel(config)(
        key,
        state,
        _logdensity,
        step_size,
        inverse_mass,
        steps,
    )

    assert bool(causal_info.causal_converged)
    assert float(causal_info.causal_maximum_residual) < 1e-8
    assert jnp.allclose(
        causal_info.proposal.position,
        sequential_info.proposal.position,
        atol=2e-8,
        rtol=2e-8,
    )
    assert bool(causal_info.is_accepted) == bool(sequential_info.is_accepted)


def test_sample_hmc_causal_result_preserves_standard_contract():
    problem = _posterior_problem()
    settings = dict(
        key=jax.random.key(13),
        num_integration_steps=4,
        num_chains=2,
        num_warmup=8,
        num_samples=4,
        initial_step_size=0.1,
        chain_method="vectorized",
    )
    sequential = phx.uq.sample_hmc(problem, **settings)
    causal = phx.uq.sample_hmc(
        problem,
        **settings,
        trajectory_method="causal",
        causal_config=phx.uq.CausalHMCConfig(
            linearization="dense-exact",
            trajectory_block_size=4,
            absolute_residual=1e-9,
            relative_residual=1e-9,
            maximum_outer_iterations=8,
        ),
    )

    assert causal.trajectory_method == "causal"
    assert causal.causal_config is not None
    assert causal.causal_diagnostics is not None
    assert jnp.all(causal.causal_diagnostics.converged)
    assert jnp.all(causal.causal_diagnostics.outer_iterations > 0)
    assert jnp.all(jnp.isfinite(causal.causal_diagnostics.maximum_residual))
    assert jnp.any(causal.causal_diagnostics.maximum_residual > 0.0)
    assert jnp.all(causal.causal_diagnostics.transition_evaluations > 0)
    assert jnp.allclose(causal.samples, sequential.samples, atol=2e-8, rtol=2e-8)
    assert jnp.array_equal(causal.divergent, sequential.divergent)


def test_causal_hmc_configuration_rejects_unsupported_combinations():
    problem = _posterior_problem()
    settings = dict(
        key=jax.random.key(14),
        num_integration_steps=3,
        num_chains=2,
        num_warmup=4,
        num_samples=4,
    )
    with pytest.raises(ValueError, match="dense-exact"):
        phx.uq.sample_hmc(
            problem,
            **settings,
            kinetic=phx.uq.MCMCMassAdaptationPlan.blocks((("",),), max_block_size=2),
            trajectory_method="causal",
        )
    with pytest.raises(ValueError, match="requires trajectory_method"):
        phx.uq.sample_hmc(
            problem,
            **settings,
            causal_config=phx.uq.CausalHMCConfig(),
        )


def test_default_pair_hutchinson_causal_hmc_produces_draws():
    result = phx.uq.sample_hmc(
        _posterior_problem(),
        key=jax.random.key(15),
        num_integration_steps=3,
        num_chains=2,
        num_warmup=4,
        num_samples=4,
        initial_step_size=0.05,
        trajectory_method="causal",
    )

    assert result.samples.shape == (2, 4, 2)
    assert jnp.all(jnp.isfinite(result.samples))
    assert isinstance(result.causal_config, phx.uq.CausalHMCConfig)
    assert result.causal_config.linearization == "pair-hutchinson"
    assert result.causal_diagnostics is not None
    assert jnp.all(result.causal_diagnostics.converged)
    assert jnp.all(result.causal_diagnostics.transition_evaluations > 0)


def test_causal_hmc_and_nuts_report_solver_fallback_records():
    recurrence = phx.uq.CausalHMCConfig(
        linearization="dense-exact",
        trajectory_block_size=4,
        absolute_residual=0.0,
        relative_residual=0.0,
        maximum_outer_iterations=1,
        failure_policy="sequential",
    )
    hmc = phx.uq.sample_hmc(
        _posterior_problem(),
        key=jax.random.key(16),
        num_integration_steps=4,
        num_chains=2,
        num_warmup=4,
        num_samples=4,
        initial_step_size=0.1,
        trajectory_method="causal",
        causal_config=recurrence,
    )
    nuts = phx.uq.sample_nuts(
        _posterior_problem(),
        key=jax.random.key(17),
        num_chains=2,
        num_warmup=4,
        chain_method="interleaved",
        num_samples=4,
        initial_step_size=0.1,
        max_num_doublings=2,
        trajectory="causal",
        causal_config=phx.uq.CausalNUTSConfig(
            max_num_doublings=2,
            recurrence=recurrence,
        ),
    )

    for result in (hmc, nuts):
        diagnostics = result.causal_diagnostics
        assert diagnostics is not None
        assert jnp.all(diagnostics.fallback_used == ~diagnostics.converged)
        assert jnp.any(diagnostics.fallback_used)
        assert jnp.all(diagnostics.outer_iterations == 1)
        assert jnp.all(jnp.isfinite(diagnostics.maximum_residual))
        assert jnp.any(diagnostics.maximum_residual > 0.0)
        assert jnp.all(diagnostics.transition_evaluations > 0)
        assert jnp.all(
            diagnostics.accepted_nonlinear_steps + diagnostics.rejected_nonlinear_steps
            > 0
        )
        assert jnp.all(diagnostics.transition_evaluations > result.num_integration_steps)


def test_causal_nuts_rejects_conflicting_execution_capacity_before_sampling():
    with pytest.raises(ValueError, match="must agree"):
        phx.uq.sample_nuts(
            _posterior_problem(),
            key=jax.random.key(18),
            num_chains=2,
            num_warmup=4,
            num_samples=4,
            initial_position=jnp.asarray([jnp.nan, 0.0]),
            max_num_doublings=3,
            trajectory="causal",
            causal_config=phx.uq.CausalNUTSConfig(max_num_doublings=2),
        )
