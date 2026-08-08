#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.uq._interleaved_nuts import build_interleaved_nuts_advancer


def _logdensity(position):
    precision = jnp.asarray([[2.0, 0.35], [0.35, 1.25]])
    return -0.5 * position @ precision @ position


def _draw_keys(seed, *, num_chains, num_draws, start=0):
    sample_keys = jr.split(jr.key(seed), num_chains)
    indices = jnp.arange(start, start + num_draws, dtype=jnp.uint32)
    return jax.vmap(
        lambda sample_key: jax.vmap(lambda index: jr.fold_in(sample_key, index))(indices)
    )(sample_keys)


def _run_blackjax(
    logdensity_fn,
    states,
    draw_keys,
    step_sizes,
    inverse_mass_matrices,
    *,
    max_num_doublings,
):
    kernel = blackjax.nuts.build_kernel()

    def run_chain(state, keys, step_size, inverse_mass_matrix):
        def one_step(current, draw_key):
            next_state, info = kernel(
                draw_key,
                current,
                logdensity_fn,
                step_size,
                inverse_mass_matrix,
                max_num_doublings,
            )
            return next_state, (next_state, info)

        return jax.lax.scan(one_step, state, keys)

    return jax.jit(jax.vmap(run_chain))(
        states,
        draw_keys,
        step_sizes,
        inverse_mass_matrices,
    )


def _assert_tree_close(left, right, *, atol=1e-12):
    comparisons = jax.tree_util.tree_map(
        lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=atol),
        left,
        right,
    )
    assert all(jax.tree_util.tree_leaves(comparisons))


@pytest.mark.parametrize("dense", [False, True])
def test_interleaved_nuts_matches_blackjax_for_unequal_chain_work(dense):
    positions = jnp.asarray([[0.2, -0.7], [1.3, 0.1], [-0.5, 2.0]])
    states = jax.vmap(lambda position: blackjax.nuts.init(position, _logdensity))(
        positions
    )
    step_sizes = jnp.asarray([0.2, 0.35, 0.5])
    if dense:
        inverse_mass_matrices = jnp.asarray(
            [
                [[1.2, 0.1], [0.1, 0.9]],
                [[0.8, -0.05], [-0.05, 1.4]],
                [[1.5, 0.2], [0.2, 0.7]],
            ]
        )
    else:
        inverse_mass_matrices = jnp.asarray([[1.0, 1.0], [0.8, 1.3], [1.5, 0.6]])
    draw_keys = _draw_keys(101, num_chains=3, num_draws=5)
    baseline_states, (baseline_draws, baseline_info) = _run_blackjax(
        _logdensity,
        states,
        draw_keys,
        step_sizes,
        inverse_mass_matrices,
        max_num_doublings=6,
    )
    advancer = build_interleaved_nuts_advancer(
        _logdensity,
        max_num_doublings=6,
    )
    final_states, samples, metrics, stats = advancer(
        states,
        step_sizes,
        inverse_mass_matrices,
        draw_keys,
    )

    _assert_tree_close(samples, baseline_draws.position)
    _assert_tree_close(final_states, baseline_states)
    assert jnp.allclose(
        metrics["log_density"],
        baseline_draws.logdensity,
        rtol=0.0,
        atol=1e-12,
    )
    assert jnp.allclose(
        metrics["acceptance_rate"],
        baseline_info.acceptance_rate,
        rtol=0.0,
        atol=1e-12,
    )
    assert jnp.allclose(
        metrics["energy"],
        baseline_info.energy,
        rtol=0.0,
        atol=1e-12,
    )
    assert jnp.array_equal(metrics["divergent"], baseline_info.is_divergent)
    assert jnp.array_equal(
        metrics["num_integration_steps"],
        baseline_info.num_integration_steps,
    )
    assert jnp.array_equal(
        metrics["num_trajectory_expansions"],
        baseline_info.num_trajectory_expansions,
    )
    _assert_tree_close(
        final_states.position,
        jax.tree_util.tree_map(lambda value: value[:, -1], samples),
        atol=0.0,
    )
    interleaved_steps = jnp.max(jnp.sum(metrics["num_integration_steps"], axis=1))
    lockstep_steps = jnp.sum(jnp.max(metrics["num_integration_steps"], axis=0))
    assert stats.num_scheduler_steps == interleaved_steps
    assert interleaved_steps <= lockstep_steps
    if not dense:
        assert interleaved_steps < lockstep_steps


def test_interleaved_nuts_preserves_pytree_positions_and_nonzero_draw_indices():
    def tree_logdensity(position):
        return -0.5 * (
            jnp.sum(position["coefficient"] ** 2) + jnp.sum(position["noise"] ** 2)
        )

    positions = {
        "coefficient": jnp.asarray([[0.2], [1.0]]),
        "noise": jnp.asarray([[0.1, -0.7], [0.5, 0.4]]),
    }
    states = jax.vmap(lambda position: blackjax.nuts.init(position, tree_logdensity))(
        positions
    )
    step_sizes = jnp.asarray([0.25, 0.4])
    inverse_mass_matrices = jnp.asarray([[1.0, 0.8, 1.2], [0.7, 1.4, 0.9]])
    draw_keys = _draw_keys(202, num_chains=2, num_draws=4, start=7)
    baseline_states, (baseline_draws, baseline_info) = _run_blackjax(
        tree_logdensity,
        states,
        draw_keys,
        step_sizes,
        inverse_mass_matrices,
        max_num_doublings=5,
    )
    final_states, samples, metrics, _ = build_interleaved_nuts_advancer(
        tree_logdensity,
        max_num_doublings=5,
    )(
        states,
        step_sizes,
        inverse_mass_matrices,
        draw_keys,
    )

    _assert_tree_close(samples, baseline_draws.position)
    _assert_tree_close(final_states, baseline_states)
    assert jnp.array_equal(
        metrics["num_integration_steps"],
        baseline_info.num_integration_steps,
    )
    assert jnp.array_equal(metrics["divergent"], baseline_info.is_divergent)


@pytest.mark.parametrize(
    ("step_size", "max_num_doublings", "expect_all_divergent"),
    [(0.01, 1, False), (20.0, 3, True)],
)
def test_interleaved_nuts_matches_depth_and_divergence_termination(
    step_size,
    max_num_doublings,
    expect_all_divergent,
):
    positions = jnp.asarray([[1.0, 1.0], [-1.0, 0.5]])
    states = jax.vmap(lambda position: blackjax.nuts.init(position, _logdensity))(
        positions
    )
    step_sizes = jnp.full((2,), step_size)
    inverse_mass_matrices = jnp.ones((2, 2))
    draw_keys = _draw_keys(333, num_chains=2, num_draws=2)
    _, (_, baseline_info) = _run_blackjax(
        _logdensity,
        states,
        draw_keys,
        step_sizes,
        inverse_mass_matrices,
        max_num_doublings=max_num_doublings,
    )
    _, _, metrics, _ = build_interleaved_nuts_advancer(
        _logdensity,
        max_num_doublings=max_num_doublings,
    )(
        states,
        step_sizes,
        inverse_mass_matrices,
        draw_keys,
    )

    assert jnp.array_equal(metrics["divergent"], baseline_info.is_divergent)
    assert jnp.array_equal(
        metrics["num_integration_steps"],
        baseline_info.num_integration_steps,
    )
    assert jnp.array_equal(
        metrics["num_trajectory_expansions"],
        baseline_info.num_trajectory_expansions,
    )
    assert bool(jnp.all(metrics["divergent"])) is expect_all_divergent
    if max_num_doublings == 1:
        assert jnp.all(metrics["num_trajectory_expansions"] == 1)
