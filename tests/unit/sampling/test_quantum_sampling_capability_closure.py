# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax.numpy as jnp
import jax.random as jr

from phydrax._sampling import _hamiltonian as hamiltonian
from phydrax.nn.quantum import (
    AutoregressiveSpinAmplitude,
    JastrowSpinAmplitude,
    RestrictedBoltzmannAmplitude,
)
from phydrax.sampling import (
    adapt_proposal_scale,
    GaussianRandomWalkProposal,
    initialize_hamiltonian_state,
    initialize_proposal_adaptation,
    MarkovChunkPlan,
    MetropolisHastings,
    prepare_hamiltonian_kernel,
    RobbinsMonroScalePolicy,
    sample_hamiltonian,
    sample_markov_chunked,
)


def test_hamiltonian_chain_has_semantic_replay_and_frozen_production():
    kernel = prepare_hamiltonian_kernel(
        lambda value: -0.5 * jnp.sum(value**2),
        jnp.array([[2.0, 0.2], [0.2, 1.0]]),
        step_size=0.1,
        leapfrog_steps=4,
        target_id="gaussian",
    )
    state = initialize_hamiltonian_state(kernel, jnp.zeros((2, 2)))
    first = sample_hamiltonian(kernel, state, key=jr.key(5), num_draws=8)
    second = sample_hamiltonian(kernel, state, key=jr.key(5), num_draws=8)
    assert jnp.array_equal(first.samples, second.samples)
    assert jnp.all(first.frozen_step_size == 0.1)
    assert not jnp.any(first.nonfinite_gradient)
    assert not jnp.any(first.divergent)
    assert jnp.all(first.leapfrog_steps == 4)


def _nonfinite_hmc_result(monkeypatch, cutoff):
    monkeypatch.setattr(
        hamiltonian,
        "_sample_momentum",
        lambda _kernel, _key: jnp.ones((1,)),
    )

    def bounded_log_target(value):
        finite_value = -0.5 * jnp.sum(value**2)
        return jnp.where(jnp.abs(value[0]) <= cutoff, finite_value, jnp.nan)

    kernel = prepare_hamiltonian_kernel(
        bounded_log_target,
        jnp.eye(1),
        step_size=0.5,
        leapfrog_steps=4,
        target_id=f"bounded-gaussian-{cutoff}",
    )
    state = initialize_hamiltonian_state(kernel, jnp.zeros((1, 1)))
    return state, sample_hamiltonian(kernel, state, key=jr.key(41), num_draws=1)


def test_hmc_rejects_a_first_step_nonfinite_trajectory(monkeypatch):
    state, result = _nonfinite_hmc_result(monkeypatch, 0.25)

    assert jnp.array_equal(result.samples[:, 0], state.position)
    assert jnp.array_equal(result.final_state.position, state.position)
    assert not result.accepted[0, 0]
    assert result.divergent[0, 0]
    assert result.nonfinite_gradient[0, 0]
    assert result.leapfrog_steps[0, 0] == 1
    assert result.acceptance_probability[0, 0] == 0.0


def test_hmc_rejects_an_entire_trajectory_after_a_later_nonfinite_step(
    monkeypatch,
):
    state, result = _nonfinite_hmc_result(monkeypatch, 0.6)

    assert jnp.array_equal(result.samples[:, 0], state.position)
    assert jnp.array_equal(result.final_state.position, state.position)
    assert not result.accepted[0, 0]
    assert result.divergent[0, 0]
    assert result.nonfinite_gradient[0, 0]
    assert result.leapfrog_steps[0, 0] == 2
    assert result.acceptance_probability[0, 0] == 0.0


def test_bounded_nuts_preserves_a_gaussian_and_reports_consumed_capacity():
    maximum_tree_depth = 4
    kernel = prepare_hamiltonian_kernel(
        lambda value: -0.5 * jnp.sum(value**2),
        jnp.eye(1),
        step_size=0.3,
        method="nuts",
        maximum_tree_depth=maximum_tree_depth,
        target_id="standard-normal",
    )
    state = initialize_hamiltonian_state(kernel, jnp.linspace(-2.0, 2.0, 8)[:, None])
    result = sample_hamiltonian(kernel, state, key=jr.key(29), num_draws=512)
    stationary_draws = result.samples[:, 64:, 0]
    capacity = 2**maximum_tree_depth - 1

    assert jnp.abs(jnp.mean(stationary_draws)) < 0.1
    assert jnp.abs(jnp.var(stationary_draws) - 1.0) < 0.15
    assert not jnp.any(result.divergent)
    assert jnp.all(result.leapfrog_steps <= capacity)
    assert jnp.array_equal(
        result.maximum_depth_reached,
        result.leapfrog_steps == capacity,
    )
    assert jnp.all(jnp.isfinite(result.acceptance_probability))


def test_bounded_nuts_reports_divergence_without_claiming_tree_capacity():
    kernel = prepare_hamiltonian_kernel(
        lambda value: -0.5 * jnp.sum(value**2),
        jnp.eye(1),
        step_size=5.0,
        method="nuts",
        maximum_tree_depth=4,
        divergence_threshold=1.0,
        target_id="divergent-standard-normal",
    )
    initial_positions = jnp.full((4, 1), 10.0)
    state = initialize_hamiltonian_state(kernel, initial_positions)
    result = sample_hamiltonian(kernel, state, key=jr.key(31), num_draws=1)

    assert jnp.all(result.divergent)
    assert jnp.all(result.leapfrog_steps == 1)
    assert not jnp.any(result.maximum_depth_reached)
    assert jnp.array_equal(result.samples[:, 0], initial_positions)


def test_hamiltonian_initial_positions_fail_closed_even_for_constant_target():
    kernel = prepare_hamiltonian_kernel(
        lambda _value: jnp.asarray(0.0),
        jnp.eye(1),
        step_size=0.2,
        method="nuts",
        maximum_tree_depth=3,
        target_id="constant",
    )
    state = initialize_hamiltonian_state(kernel, jnp.asarray([[jnp.nan]]))
    result = sample_hamiltonian(kernel, state, key=jr.key(30), num_draws=2)

    assert not state.valid[0]
    assert jnp.all(jnp.isnan(result.samples[0]))
    assert jnp.all(result.divergent[0])
    assert jnp.all(result.leapfrog_steps[0] == 0)
    assert not result.final_state.valid[0]


def test_chunked_markov_prefix_and_partial_mask_are_explicit():
    kernel = MetropolisHastings(GaussianRandomWalkProposal(0.2))
    initial = jnp.zeros((3, 1))
    target = lambda value: -0.5 * jnp.sum(value**2)
    state = kernel.initialize(target, initial)
    result = sample_markov_chunked(
        target,
        kernel,
        state,
        key=jr.key(9),
        plan=MarkovChunkPlan(5, 3),
    )
    assert result.samples.shape == (3, 6, 1)
    assert jnp.array_equal(result.active, jnp.array([1, 1, 1, 1, 1, 0], dtype=bool))
    assert bool(result.replay_exact)


def test_robbins_monro_scale_adapts_only_before_frozen_boundary():
    policy = RobbinsMonroScalePolicy(warmup_chunks=1)
    state = initialize_proposal_adaptation(policy, 1.0)
    adapted = adapt_proposal_scale(policy, state, 0.9)
    frozen = adapt_proposal_scale(policy, adapted, 0.1)
    assert bool(adapted.frozen)
    assert jnp.allclose(frozen.scale, adapted.scale)


def test_jastrow_rbm_caches_and_autoregressive_normalization():
    spins = jnp.array([1.0, -1.0, 1.0])
    jastrow = JastrowSpinAmplitude(
        jnp.array([0.2, -0.1, 0.3]),
        jnp.array([[0.0, 0.4, 0.1], [0.4, 0.0, -0.2], [0.1, -0.2, 0.0]]),
    )
    cache = jastrow.initialize_cache(spins)
    ratio, proposed, valid = jastrow.propose_flips(
        cache, jnp.array([1]), jnp.array([True])
    )
    direct = jastrow(spins.at[1].multiply(-1))
    assert bool(valid)
    assert jnp.allclose(jnp.real(ratio), direct.log_abs - jastrow(spins).log_abs)
    assert jnp.allclose(proposed.complex_log_amplitude.real, direct.log_abs)

    rbm = RestrictedBoltzmannAmplitude(
        jnp.zeros((3,)), jnp.zeros((2,)), jnp.ones((2, 3)) * 0.1
    )
    rbm_ratio, rbm_proposed, rbm_valid = rbm.propose_flips(
        rbm.initialize_cache(spins), jnp.array([0]), jnp.array([True])
    )
    assert bool(rbm_valid)
    assert jnp.isfinite(rbm_ratio)
    assert bool(rbm_proposed.valid)

    autoregressive = AutoregressiveSpinAmplitude(jnp.zeros((3,)), jnp.zeros((3, 3)))
    configurations = jnp.array(
        [[a, b, c] for a in (-1.0, 1.0) for b in (-1.0, 1.0) for c in (-1.0, 1.0)]
    )
    probabilities = jnp.exp(
        2.0 * jnp.array([autoregressive(value).log_abs for value in configurations])
    )
    assert jnp.allclose(jnp.sum(probabilities), 1.0)
