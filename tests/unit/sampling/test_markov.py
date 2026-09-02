import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _normal_proposal(scale=0.25):
    return phx.sampling.GaussianRandomWalkProposal(scale)


def _standard_normal(value):
    return -0.5 * jnp.sum(value**2)


class _PayloadSpinFlipProposal(phx.sampling.AbstractProposal):
    proposal_id: str = eqx.field(static=True)

    def __init__(self):
        self.proposal_id = "payload-spin-flip"

    def sample(self, _key, current, /):
        proposed = current.at[1].multiply(-1)
        return proposed.at[0].set(jnp.where(current[0] > 0, proposed[0], jnp.nan))

    def log_prob(self, proposed, current, /):
        del proposed, current
        return jnp.zeros(())

    def payload(self, key, current, proposed, /):
        del key, current, proposed
        return jnp.asarray([1]), jnp.asarray([True])


def _select_tree(current, proposed, accepted):
    return jax.tree_util.tree_map(
        lambda proposed_leaf, current_leaf: jnp.where(
            accepted, proposed_leaf, current_leaf
        ),
        proposed,
        current,
    )


def test_metropolis_hastings_uses_asymmetric_proposal_ratio():
    def toggle(_key, current):
        return 1 - current

    def log_prob(proposed, current):
        return jnp.where(
            current == 0,
            jnp.where(proposed == 1, jnp.log(0.8), jnp.log(0.2)),
            jnp.where(proposed == 0, jnp.log(0.3), jnp.log(0.7)),
        ).reshape(())

    proposal = phx.sampling.CallableProposal(
        toggle,
        log_prob,
        proposal_id="asymmetric-toggle",
    )
    kernel = phx.sampling.MetropolisHastings(proposal)
    state = kernel.initialize(
        lambda value: jnp.where(value[0] == 1, jnp.log(0.6), jnp.log(0.4)),
        jnp.asarray([[0]], dtype=jnp.int32),
    )

    _next_state, info = kernel.step(
        lambda value: jnp.where(value[0] == 1, jnp.log(0.6), jnp.log(0.4)),
        state,
        jr.key(0),
    )

    expected = jnp.log(0.6) - jnp.log(0.4) + jnp.log(0.3) - jnp.log(0.8)
    assert jnp.allclose(info.log_acceptance_ratio[0], expected)
    assert info.proposal_valid[0]
    assert info.target_valid[0]


def test_markov_sampling_is_jittable_reproducible_and_prefix_stable():
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    initial_two = jnp.asarray([[-0.5], [0.75]])
    initial_three = jnp.asarray([[-0.5], [0.75], [1.25]])
    state_two = kernel.initialize(_standard_normal, initial_two)
    state_three = kernel.initialize(_standard_normal, initial_three)

    def run(state, key):
        return phx.sampling.sample_markov(
            _standard_normal,
            kernel,
            state,
            key=key,
            num_draws=6,
            steps_per_draw=2,
            warmup_steps=3,
        )

    eager = run(state_two, jr.key(4))
    compiled = eqx.filter_jit(run)(state_two, jr.key(4))
    extended = run(state_three, jr.key(4))

    assert isinstance(eager, phx.sampling.AbstractChainSampleResult)
    assert eager.chain_provenance == "markov:metropolis-hastings:gaussian-random-walk"

    assert jnp.array_equal(eager.samples, compiled.samples)
    assert jnp.array_equal(eager.accepted, compiled.accepted)
    assert jnp.array_equal(eager.samples, extended.samples[:2])
    assert jnp.array_equal(eager.accepted, extended.accepted[:2])
    assert eager.samples.shape == (2, 6, 1)
    assert eager.accepted.shape == (2, 6, 2)


def test_refresh_preserves_positions_and_recomputes_target_values():
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    state = kernel.initialize(_standard_normal, jnp.asarray([[1.0], [-2.0]]))
    refreshed = kernel.refresh(lambda value: -jnp.sum((value - 1.0) ** 2), state)

    assert jnp.array_equal(refreshed.position, state.position)
    assert refreshed.step_index == state.step_index
    assert not jnp.array_equal(refreshed.log_target, state.log_target)
    assert jnp.allclose(refreshed.log_target, jnp.asarray([0.0, -9.0]))


def test_markov_chain_measure_preserves_correlation_and_never_claims_iid_error():
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    state = kernel.initialize(_standard_normal, jnp.asarray([[-0.5], [0.75]]))
    result = phx.sampling.sample_markov(
        _standard_normal,
        kernel,
        state,
        key=jr.key(8),
        num_draws=5,
    )
    target = phx.integration.markov_chain_measure(result)
    estimate = phx.integration.integrate(lambda values: values**2, target)

    assert target.independent is False
    assert target.sample_axes == (
        "__phydrax_markov_chain",
        "__phydrax_markov_draw",
    )
    assert target.provenance.startswith("markov:metropolis-hastings")
    assert estimate.error_estimate is None
    assert estimate.error_kind is None
    assert estimate.diagnostics.standard_error is None
    assert jnp.allclose(
        estimate.value.data,
        jnp.mean(result.samples**2, axis=(0, 1)),
    )


def test_markov_sampling_rejects_invalid_contracts():
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    with pytest.raises(TypeError, match="real-valued"):
        kernel.initialize(lambda value: 1j * jnp.sum(value), jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="leading chain axis"):
        kernel.initialize(_standard_normal, jnp.asarray(0.0))

    state = kernel.initialize(_standard_normal, jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="num_draws"):
        phx.sampling.sample_markov(
            _standard_normal,
            kernel,
            state,
            key=jr.key(0),
            num_draws=0,
        )


def test_incremental_spin_targets_commit_only_accepted_cached_updates():
    initial = jnp.asarray([[1.0, -1.0, 1.0], [-1.0, -1.0, 1.0]])
    expected_accepted = initial.at[0, 1].multiply(-1)
    kernel = phx.sampling.MetropolisHastings(_PayloadSpinFlipProposal())
    models_and_targets = (
        (
            phx.nn.quantum.JastrowSpinAmplitude(
                jnp.asarray([0.0, 1.0, 0.0]), jnp.zeros((3, 3))
            ),
            phx.nn.quantum.jastrow_incremental_target,
        ),
        (
            phx.nn.quantum.RestrictedBoltzmannAmplitude(
                jnp.asarray([0.0, 1.0, 0.0]),
                jnp.zeros((2,)),
                jnp.zeros((2, 3)),
            ),
            phx.nn.quantum.rbm_incremental_target,
        ),
    )

    for model, make_target in models_and_targets:
        target = make_target(model)
        state = kernel.initialize(target, initial)
        result = phx.sampling.sample_markov_chunked(
            target,
            kernel,
            state,
            key=jr.key(12),
            plan=phx.sampling.MarkovChunkPlan(1, 1),
        )

        assert jnp.array_equal(result.accepted[:, 0, 0], jnp.asarray([True, False]))
        assert jnp.array_equal(result.final_state.position, expected_accepted)
        assert jnp.array_equal(result.final_state.cache.spins, expected_accepted)
        exact_values = jax.vmap(lambda position: 2.0 * model(position).log_abs)(
            expected_accepted
        )
        assert jnp.allclose(result.final_state.log_target, exact_values)
        assert jnp.all(result.final_state.valid)
        assert result.target_id == target.target_id


def test_incremental_target_refresh_runs_on_declared_transition_cadence():
    target = phx.sampling.IncrementalMarkovTarget(
        initialize=lambda position: (jnp.zeros(()), position),
        propose=lambda current, cache, proposed, payload: (
            jnp.ones(()),
            proposed,
            jnp.asarray(True),
        ),
        select=_select_tree,
        refresh=lambda position: (jnp.zeros(()), position),
        target_id="cadenced-target",
        refresh_cadence=2,
        cache_tolerance=10.0,
    )
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    state = kernel.initialize(target, jnp.zeros((1, 1)))
    result = phx.sampling.sample_markov_chunked(
        target,
        kernel,
        state,
        key=jr.key(23),
        plan=phx.sampling.MarkovChunkPlan(4, 1),
    )

    assert jnp.array_equal(result.accepted, jnp.ones((1, 4, 1), dtype=bool))
    assert jnp.allclose(result.log_target[0], jnp.asarray([1.0, 0.0, 1.0, 0.0]))
    assert jnp.all(result.final_state.valid)
    assert jnp.all(result.target_valid)


def test_incremental_target_refresh_cache_mismatch_fails_closed():
    target = phx.sampling.IncrementalMarkovTarget(
        initialize=lambda position: (jnp.zeros(()), position),
        propose=lambda current, cache, proposed, payload: (
            jnp.zeros(()),
            proposed + 1.0,
            jnp.asarray(True),
        ),
        select=_select_tree,
        refresh=lambda position: (jnp.zeros(()), position),
        target_id="mismatched-cache-target",
        refresh_cadence=2,
        cache_tolerance=0.0,
    )
    kernel = phx.sampling.MetropolisHastings(_normal_proposal())
    state = kernel.initialize(target, jnp.zeros((1, 1)))
    result = phx.sampling.sample_markov(
        target,
        kernel,
        state,
        key=jr.key(31),
        num_draws=2,
    )

    assert bool(result.target_valid[0, 0, 0])
    assert not bool(result.target_valid[0, 1, 0])
    assert not bool(result.final_state.valid[0])
    assert jnp.array_equal(result.final_state.cache, result.final_state.position)
