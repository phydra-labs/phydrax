#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

from phydrax.control.games._layout import PlayerControlPartition
from phydrax.control.games._stochastic_smp import (
    evaluate_open_loop_stochastic_game_smp,
    OpenLoopStochasticGameSMPProblem,
)
from phydrax.control.stochastic._evaluation import (
    ControlledPathBatch,
    ControlledTransitionProblem,
    PreparedControlledNoise,
)
from phydrax.dynamics import TimeGrid


def _paths(states, actions, noise, *, clusters=None):
    states = jnp.asarray(states, dtype=float)
    actions = jnp.asarray(actions, dtype=float)
    noise = jnp.asarray(noise, dtype=float)
    count, steps = actions.shape[:2]
    grid = TimeGrid(jnp.arange(steps + 1, dtype=float), time_id="game-smp-grid")
    controlled = ControlledTransitionProblem(
        lambda context, state, action, increment, args: state,
        grid,
        states[0, 0],
        state_shape=states.shape[2:],
        action_shape=actions.shape[2:],
        noise_shape=noise.shape[2:],
        stage_cost=lambda context, state, action, args: 0.0,
        terminal_cost=lambda time, state, args: 0.0,
        problem_id="game-smp-test",
    )
    prepared = PreparedControlledNoise(
        noise,
        valid=jnp.ones((count,), dtype=bool),
        realization_ids=tuple(f"common-private:{index}" for index in range(count)),
        coupling_id="common-private-coupling",
        independence_labels=(
            jnp.arange(count, dtype=jnp.int32)
            if clusters is None
            else jnp.asarray(clusters, dtype=jnp.int32)
        ),
        noise_shape=noise.shape[2:],
    )
    return ControlledPathBatch(
        problem=controlled,
        prepared_noise=prepared,
        states=states,
        actions=actions,
        valid=jnp.ones((count,), dtype=bool),
        status=jnp.zeros((count,), dtype=jnp.int32),
        stage_costs=jnp.zeros((count, steps)),
        terminal_costs=jnp.zeros((count,)),
        returns=jnp.zeros((count,)),
        policy_id="supplied-joint-open-loop-actions",
    )


def _problem(noise_size, action_gradients):
    grid = TimeGrid(jnp.asarray([0.0, 1.0]), time_id="game-smp-grid")
    partition = PlayerControlPartition(("row", "column"), (1, 1))
    zero_state = lambda context, state, action, args: jnp.zeros((1,))
    return OpenLoopStochasticGameSMPProblem(
        grid,
        partition,
        zero_state,
        lambda context, state, action, args: jnp.zeros((1, noise_size)),
        lambda context, state, action, args: jnp.zeros((1, 1)),
        lambda context, state, action, args: jnp.zeros((1, 2)),
        lambda context, state, action, args: jnp.zeros((1, noise_size, 1)),
        lambda context, state, action, args: jnp.zeros((1, noise_size, 2)),
        running_cost_state_gradients=(zero_state, zero_state),
        running_cost_action_gradients=action_gradients,
        terminal_cost_gradients=(
            lambda time, state, args: jnp.zeros((1,)),
            lambda time, state, args: jnp.zeros((1,)),
        ),
        state_shape=(1,),
        noise_shape=(noise_size,),
        problem_id="game-smp-test",
    )


def _evaluate(problem, paths, labels, *, sample_role="holdout", **kwargs):
    count = paths.path_count
    adjoints = (
        jnp.zeros((count, 2, 1)),
        jnp.zeros((count, 2, 1)),
    )
    integrands = (
        jnp.zeros((count, 1, 1, problem.noise_size)),
        jnp.zeros((count, 1, 1, problem.noise_size)),
    )
    return evaluate_open_loop_stochastic_game_smp(
        problem,
        paths,
        adjoints,
        integrands,
        labels,
        information_ids=("row-filtration", "column-filtration"),
        predictor_ids=("row-adjoint", "column-adjoint"),
        sample_id=f"game-{sample_role}-4",
        sample_role=sample_role,
        causal_information_checked=(True, True),
        **kwargs,
    )


def test_game_smp_retains_only_each_players_owned_action_rows():
    paths = _paths(
        states=[[[0.0], [0.0]]],
        actions=[[[0.0, 0.0]]],
        noise=[[[0.0]]],
    )
    problem = _problem(
        1,
        (
            lambda context, state, action, args: jnp.asarray([1.0, jnp.nan]),
            lambda context, state, action, args: jnp.asarray([jnp.nan, 2.0]),
        ),
    )

    result = _evaluate(problem, paths, jnp.zeros((2, 1, 1), dtype=jnp.int32))

    assert result.certificate == "OPEN_LOOP_NASH_SMP_STATIONARY"
    assert jnp.all(result.valid)
    assert jnp.allclose(
        result.owned_hamiltonian_action_gradients[0, 0],
        jnp.asarray([1.0, 2.0]),
    )
    assert jnp.allclose(
        result.conditional_owned_stationarity_residuals[0, 0],
        jnp.asarray([1.0, 2.0]),
    )
    assert not jnp.any(result.stationary)
    assert not result.sufficient
    assert not result.open_loop_nash_claim
    assert not result.feedback_claim
    assert not result.feedback_nash_claim
    assert not result.markov_perfect_claim


@pytest.mark.parametrize("sample_role", ["training", "holdout"])
def test_zero_empirical_game_smp_residual_does_not_claim_open_loop_nash(
    sample_role,
):
    paths = _paths(
        states=[[[0.0], [0.0]], [[0.0], [0.0]]],
        actions=[[[0.0, 0.0]], [[0.0, 0.0]]],
        noise=[[[0.0]], [[0.0]]],
        clusters=[0, 1],
    )
    zero_action = lambda context, state, action, args: jnp.zeros((2,))
    result = _evaluate(
        _problem(1, (zero_action, zero_action)),
        paths,
        jnp.zeros((2, 2, 1), dtype=jnp.int32),
        sample_role=sample_role,
        convexity_checked=(True, True),
        convexity_evidence=(
            "row Hamiltonian convex in row action",
            "column Hamiltonian convex in column action",
        ),
    )

    assert result.certificate == "OPEN_LOOP_NASH_SMP_STATIONARY"
    assert jnp.all(result.stationary)
    assert jnp.allclose(result.conditional_owned_stationarity_residuals, 0.0)
    assert jnp.allclose(result.maximum_residual_norms, 0.0)
    assert result.path_evidence.sample_role == sample_role
    assert result.convexity_checked == (True, True)
    assert all(item is not None for item in result.convexity_evidence)
    assert not result.sufficient
    assert not result.population_stationarity_claim
    assert not result.population_nash_claim
    assert not result.open_loop_nash_claim
    assert not result.feedback_claim
    assert not result.feedback_nash_claim
    assert not result.markov_perfect_claim


def test_game_smp_preserves_common_private_information_and_cluster_evidence():
    paths = _paths(
        states=[
            [[0.0], [0.0]],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
            [[0.0], [0.0]],
        ],
        actions=[
            [[1.0, -1.0]],
            [[1.0, 3.0]],
            [[2.0, 4.0]],
            [[2.0, 7.0]],
        ],
        noise=[
            [[-1.0, -0.5, 0.25]],
            [[-1.0, 0.5, -0.25]],
            [[1.0, -0.5, -0.25]],
            [[1.0, 0.5, 0.25]],
        ],
        clusters=[0, 0, 1, 1],
    )
    zero_action = lambda context, state, action, args: jnp.zeros((2,))
    problem = _problem(3, (zero_action, zero_action))
    labels = jnp.asarray(
        [
            [[0], [0], [1], [1]],
            [[0], [1], [2], [3]],
        ],
        dtype=jnp.int32,
    )

    result = _evaluate(problem, paths, labels)

    assert jnp.array_equal(result.causal_information.information_labels, labels)
    assert jnp.array_equal(result.causal_information.causal, jnp.asarray([True, True]))
    assert jnp.all(result.stationary)
    assert jnp.array_equal(result.path_evidence.valid_path_counts, jnp.asarray([4, 4]))
    assert jnp.array_equal(
        result.path_evidence.independent_cluster_counts, jnp.asarray([2, 2])
    )
    assert result.path_evidence.path_ids == tuple(
        f"common-private:{index}" for index in range(4)
    )
    assert result.path_evidence.coupling_id == "common-private-coupling"
    assert result.path_evidence.sample_role == "holdout"
    assert result.path_evidence.sample_id == "game-holdout-4"
    assert not result.feedback_claim
    assert not result.markov_perfect_claim
