import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _ising_chain(length=4, coupling=0.35):
    edges = jnp.stack([jnp.arange(length - 1), jnp.arange(1, length)], axis=-1)
    return phx.pgm.ising_factor_graph(
        jnp.zeros((length,)),
        edges,
        jnp.full((length - 1,), coupling),
    )


def test_gibbs_coloring_is_deterministic_and_rejects_invalid_parallel_stage():
    graph = _ising_chain()
    first = phx.pgm.prepare_chromatic_gibbs(graph)
    second = phx.pgm.prepare_chromatic_gibbs(graph)

    assert jnp.array_equal(first.colors, jnp.asarray([0, 1, 0, 1]))
    assert first.plan_id == second.plan_id

    with pytest.raises(ValueError, match="distinct Gibbs colors"):
        phx.pgm.prepare_chromatic_gibbs(
            graph,
            phx.pgm.ChromaticGibbs(jnp.zeros((4,), dtype=jnp.int32)),
        )


def test_gibbs_is_jittable_persistent_and_chain_prefix_stable():
    graph = _ising_chain()
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state_two = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0, 0, 0], [1, 1, 1, 1]]),
    )
    state_three = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 0, 1]]),
    )
    schedule = phx.pgm.GibbsSchedule(warmup_sweeps=2, num_draws=6, sweeps_per_draw=2)

    def run(state):
        return phx.pgm.sample_gibbs(
            prepared,
            state,
            key=jr.key(7),
            schedule=schedule,
        )

    eager = run(state_two)
    compiled = eqx.filter_jit(run)(state_two)
    extended = run(state_three)

    assert jnp.array_equal(eager.samples, compiled.samples)
    assert jnp.array_equal(eager.samples, extended.samples[:2])
    assert jnp.array_equal(eager.transition_valid, extended.transition_valid[:2])
    assert eager.samples.shape == (2, 6, 4)
    assert eager.transition_valid.shape == (2, 6, 2)
    assert eager.final_state.sweep_index == 14
    assert eager.diagnostics.mixing_available


def test_gibbs_clamping_preserves_sites_and_chain_measure_preserves_correlation():
    graph = _ising_chain(3)
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    initial = jnp.asarray([[1, 0, 0], [1, 1, 1]])
    state = phx.pgm.initialize_gibbs(prepared, initial)
    result = phx.pgm.sample_gibbs(
        prepared,
        state,
        key=jr.key(3),
        schedule=phx.pgm.GibbsSchedule(num_draws=5),
        clamped=jnp.asarray([True, False, False]),
    )
    target = phx.integration.markov_chain_measure(result)

    assert jnp.all(result.samples[:, :, 0] == 1)
    assert target.independent is False
    assert target.sample_axes == (
        "__phydrax_markov_chain",
        "__phydrax_markov_draw",
    )
    assert target.provenance.startswith("markov:chromatic-gibbs")


def test_impossible_conditional_preserves_state_and_reports_failure():
    variables = phx.pgm.DiscreteVariableGroup("x", shape=(1,), num_states=2)
    factor = phx.pgm.DenseTableFactorGroup(
        (phx.pgm.VariableSelection.all(variables),),
        jnp.full((1, 2), -jnp.inf),
    )
    graph = phx.pgm.DiscreteFactorGraph((variables,), (factor,))
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state = phx.pgm.GibbsState(
        jnp.asarray([[0]], dtype=jnp.int32),
        jnp.asarray([-jnp.inf]),
        valid=jnp.asarray([False]),
    )
    updated, info = phx.pgm.gibbs_sweep(prepared, state, jr.key(0))

    assert jnp.array_equal(updated.positions, state.positions)
    assert not info.valid[0]
    assert info.invalid_conditional_count[0] == 1
    assert info.status[0] == int(phx.pgm.GibbsTransitionStatus.INFEASIBLE_CONDITIONAL)


def test_gibbs_empirical_distribution_matches_exact_two_spin_law():
    graph = _ising_chain(2, coupling=0.4)
    exact = phx.pgm.enumerate_factor_graph(graph)
    prepared = phx.pgm.prepare_chromatic_gibbs(graph)
    state = phx.pgm.initialize_gibbs(
        prepared,
        jnp.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
    )
    result = phx.pgm.sample_gibbs(
        prepared,
        state,
        key=jr.key(9),
        schedule=phx.pgm.GibbsSchedule(
            warmup_sweeps=50,
            num_draws=1000,
            sweeps_per_draw=2,
        ),
    )
    flat = result.samples.reshape((-1, 2))
    empirical = jnp.asarray(
        [
            jnp.mean(jnp.all(flat == state_value, axis=-1))
            for state_value in (
                jnp.asarray([0, 0]),
                jnp.asarray([0, 1]),
                jnp.asarray([1, 0]),
                jnp.asarray([1, 1]),
            )
        ]
    )
    expected = jnp.asarray(
        [
            exact.factor_probabilities[1][0, 0, 0],
            exact.factor_probabilities[1][0, 0, 1],
            exact.factor_probabilities[1][0, 1, 0],
            exact.factor_probabilities[1][0, 1, 1],
        ]
    )

    assert jnp.allclose(empirical, expected, atol=0.04)
