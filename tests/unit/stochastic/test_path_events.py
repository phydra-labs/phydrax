#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx


def _trajectory():
    times = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 2.0]), (4, 3))
    states = jnp.asarray(
        [
            [[0.0], [1.0], [2.0]],
            [[0.0], [0.2], [0.4]],
            [[0.0], [1.0], [1.5]],
            [[0.0], [0.2], [0.3]],
        ]
    )
    valid = jnp.asarray(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
            [True, False, False],
        ]
    )
    return phx.stochastic.StochasticTrajectory(
        times,
        states,
        valid=valid,
        realization_axes=("path",),
        realization_shape=(4,),
        state_axes=("state",),
        realizations=(None,),
    )


def test_threshold_crossing_localizes_and_distinguishes_censoring_from_failure():
    trajectory = _trajectory()
    event = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        0.5,
        direction="up",
        event_id="upper-half",
    )
    result = eqx.filter_jit(phx.stochastic.evaluate_path_event)(trajectory, event)

    assert jnp.array_equal(result.occurred, jnp.asarray([True, False, True, False]))
    assert jnp.array_equal(result.censored, jnp.asarray([False, True, False, False]))
    assert jnp.array_equal(result.failed, jnp.asarray([False, False, False, True]))
    assert jnp.allclose(result.event_times[jnp.asarray([0, 2])], 0.5)
    assert jnp.array_equal(result.event_indices, jnp.asarray([1, -1, 1, -1]))
    assert result.event_ids == ("upper-half",)


def test_terminal_and_accumulated_events_use_complete_path_semantics():
    trajectory = _trajectory()
    terminal = phx.stochastic.TerminalSetEvent(
        lambda time, state: state[0] >= 0.3,
        event_id="terminal-target",
        score=lambda time, state: state[0] - 0.3,
    )
    accumulated = phx.stochastic.AccumulatedPathEvent(
        lambda time, state: state[0],
        0.5,
        event_id="integrated-state",
    )

    terminal_result = phx.stochastic.evaluate_path_event(trajectory, terminal)
    accumulated_result = phx.stochastic.evaluate_path_event(trajectory, accumulated)
    scores = phx.stochastic.path_event_scores(trajectory, accumulated)

    assert jnp.array_equal(
        terminal_result.occurred,
        jnp.asarray([True, True, False, False]),
    )
    assert jnp.array_equal(
        accumulated_result.occurred,
        jnp.asarray([True, False, True, False]),
    )
    assert jnp.allclose(
        accumulated_result.event_times[jnp.asarray([0, 2])],
        1.0,
    )
    assert scores.shape == (4, 3)
    assert jnp.isneginf(scores[3, 1])


def test_competing_events_report_earliest_event_code_with_stable_ties():
    times = jnp.broadcast_to(jnp.asarray([0.0, 1.0, 2.0]), (2, 3))
    states = jnp.asarray(
        [
            [[0.0], [1.0], [2.0]],
            [[0.0], [-1.0], [-2.0]],
        ]
    )
    trajectory = phx.stochastic.StochasticTrajectory(
        times,
        states,
        realization_axes=("path",),
        realization_shape=(2,),
        state_axes=("state",),
        realizations=(None,),
    )
    upper = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        0.5,
        event_id="upper",
    )
    lower = phx.stochastic.ThresholdCrossingEvent(
        lambda time, state: state[0],
        -0.5,
        direction="down",
        event_id="lower",
    )

    result = phx.stochastic.evaluate_path_event(
        trajectory,
        phx.stochastic.CompetingPathEvents((upper, lower)),
    )

    assert jnp.array_equal(result.occurred, jnp.asarray([True, True]))
    assert jnp.array_equal(result.event_codes, jnp.asarray([0, 1]))
    assert jnp.allclose(result.event_times, 0.5)
    assert result.event_ids == ("upper", "lower")
