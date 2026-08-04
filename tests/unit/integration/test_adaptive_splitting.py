#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


_TARGET = 5.0


def _population(final_states):
    finals = jnp.asarray(final_states)
    population_size = finals.shape[0]
    states = jnp.stack(
        (
            jnp.full((population_size,), -1.0e6),
            finals,
        ),
        axis=-1,
    )[..., None]
    return phx.stochastic.StochasticTrajectory(
        jnp.asarray([0.0, 1.0]),
        states,
        realization_axes=("path",),
        realization_shape=(population_size,),
        state_axes=("state",),
        realizations=(None,),
    )


def _initial_exponential(key, population_size):
    return _population(jr.exponential(key, (population_size,)))


def _conditional_exponential_branch(population, request):
    innovations = jax.vmap(jr.exponential)(request.branch_keys)
    conditional_finals = _TARGET + request.level + innovations
    states = population.states.at[request.killed_indices, 1, 0].set(conditional_finals)
    return phx.stochastic.StochasticTrajectory(
        population.times,
        states,
        valid=population.valid,
        realization_axes=population.realization_axes,
        realization_shape=population.realization_shape,
        state_axes=population.state_axes,
        realizations=(None,),
    )


def test_adaptive_splitting_estimates_exponential_tail_and_tracks_genealogy():
    event = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        _TARGET,
        event_id="exponential-tail",
    )
    plan = phx.integration.AdaptiveMultilevelSplittingPlan(
        1024,
        kill_count=256,
        max_rounds=64,
    )

    result = phx.integration.adaptive_multilevel_splitting(
        event,
        plan,
        initial_sampler=_initial_exponential,
        branch_sampler=_conditional_exponential_branch,
        key=jr.key(52),
    )

    assert result.successful
    assert result.status_message == "converged"
    assert jnp.allclose(result.probability, jnp.exp(-_TARGET), rtol=0.35)
    assert result.diagnostics.num_rounds > 10
    assert jnp.all(jnp.diff(result.diagnostics.levels) >= 0.0)
    assert jnp.all(result.diagnostics.killed_counts == plan.kill_count)
    assert result.diagnostics.parent_indices.shape == (
        result.diagnostics.num_rounds,
        plan.population_size,
    )
    assert len(result.diagnostics.population_trajectory_ids) == (
        result.diagnostics.num_rounds
    )
    killed = result.diagnostics.killed_masks
    branches = result.diagnostics.branch_indices
    assert jnp.all(branches[killed] == 1)
    assert jnp.all(branches[~killed] == -1)


def test_adaptive_splitting_replicates_report_empirical_standard_error():
    event = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        _TARGET,
        event_id="replicated-tail",
    )
    plan = phx.integration.AdaptiveMultilevelSplittingPlan(
        128,
        kill_count=32,
        max_rounds=64,
    )

    result = phx.integration.replicate_adaptive_multilevel_splitting(
        event,
        plan,
        4,
        initial_sampler=_initial_exponential,
        branch_sampler=_conditional_exponential_branch,
        key=jr.key(54),
    )

    assert result.num_replicates == 4
    assert result.num_completed == 4
    assert jnp.all(jnp.isfinite(result.replicate_probabilities))
    assert jnp.isfinite(result.standard_error)
    assert jnp.allclose(
        result.standard_error,
        jnp.std(result.replicate_probabilities, ddof=1) / 2.0,
    )


def test_adaptive_splitting_reports_tied_population_extinction():
    event = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        1.0,
        event_id="unreachable",
    )
    plan = phx.integration.AdaptiveMultilevelSplittingPlan(
        16,
        kill_count=4,
        max_rounds=4,
    )

    def initial_sampler(key, population_size):
        del key
        return _population(jnp.zeros((population_size,)))

    def unreachable_branch(population, request):
        raise AssertionError("An extinct population must not request branching.")

    result = phx.integration.adaptive_multilevel_splitting(
        event,
        plan,
        initial_sampler=initial_sampler,
        branch_sampler=unreachable_branch,
        key=jr.key(53),
    )

    assert result.status == int(phx.integration.AdaptiveSplittingStatus.EXTINCTION)
    assert not result.successful
    assert result.probability == 0.0
    assert jnp.isneginf(result.log_probability)
    assert result.diagnostics.num_rounds == 0
