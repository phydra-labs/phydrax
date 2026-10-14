"""Same-runtime native checkpoint restores every physical history and controller."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from phydrax import config as pxconfig
from phydrax.applications.nuclear_fuel import (
    FuelRodResult,
    FuelRodSimulation,
    FuelRodState,
)
from tests._support.assertions import assert_tree_values_equal
from tests._support.nuclear_fuel import prepare_low_power_startup_rod


_TIME_STEP = 0.5  # [s]


@dataclass(frozen=True)
class _RestartScenario:
    complete: FuelRodResult
    first_accepted: FuelRodState
    checkpoint_time: float
    continued: FuelRodResult


def _times(states: tuple[FuelRodState, ...], /) -> list[float]:
    return [float(np.asarray(state.time)) for state in states]


@pytest.fixture(scope="module")
def restart(tmp_path_factory: pytest.TempPathFactory) -> _RestartScenario:
    directory = tmp_path_factory.mktemp("restart")
    with pxconfig.context(pxconfig.NumericsConfig(enable_x64=True)):
        prepared = prepare_low_power_startup_rod()
        baseline = FuelRodSimulation(
            prepared, time_step=_TIME_STEP, checkpoint_directory=directory / "baseline"
        )
        complete = baseline.run()
        baseline.close()
        interrupted = FuelRodSimulation(
            prepared, time_step=_TIME_STEP, checkpoint_directory=directory / "restart"
        )
        initial = interrupted.runtime.initial_state(prepared.initial_state())
        first, transition = interrupted.runtime.step(initial)
        assert bool(transition.successful)
        checkpoint = interrupted.runtime.checkpoint(first)
        interrupted.close()
        first_accepted = first.accepted_state
        assert isinstance(first_accepted, FuelRodState)
        resumed = FuelRodSimulation(
            prepared, time_step=_TIME_STEP, checkpoint_directory=directory / "restart"
        )
        continued = resumed.run(restart=True)
        resumed.close()
        return _RestartScenario(
            complete=complete,
            first_accepted=first_accepted,
            checkpoint_time=float(np.asarray(checkpoint.time)),
            continued=continued,
        )


def test_interrupted_and_resumed_runs_both_reach_the_history_end(
    restart: _RestartScenario,
) -> None:
    assert restart.checkpoint_time > 0.0
    assert bool(restart.complete.successful)
    assert bool(restart.continued.successful)
    complete_times = _times(restart.complete.states)
    for output in (1.0, 2.0, 3.0):
        assert output in complete_times
    assert _times(restart.continued.states)[-1] == complete_times[-1] == 3.0


def test_interrupted_first_step_matches_uninterrupted_first_accepted_state(
    restart: _RestartScenario,
) -> None:
    assert_tree_values_equal(restart.first_accepted, restart.complete.states[1])


def test_resumed_run_reproduces_every_accepted_state_after_the_checkpoint(
    restart: _RestartScenario,
) -> None:
    expected = tuple(
        state
        for state in restart.complete.states
        if float(np.asarray(state.time)) >= restart.checkpoint_time
    )
    assert _times(restart.continued.states) == _times(expected)
    for actual, reference in zip(restart.continued.states, expected, strict=True):
        assert_tree_values_equal(actual, reference)
    assert_tree_values_equal(restart.continued.final_state, restart.complete.final_state)


def test_resumed_run_reproduces_every_attempt_after_the_checkpoint(
    restart: _RestartScenario,
) -> None:
    # Attempts include refused retries: index, time, step size, status, solver,
    # physical, balance, and source-support evidence must all coincide.
    expected = tuple(
        attempt
        for attempt in restart.complete.attempts
        if float(np.asarray(attempt.time)) >= restart.checkpoint_time
    )
    assert expected
    assert len(restart.continued.attempts) == len(expected)
    for actual, reference in zip(restart.continued.attempts, expected, strict=True):
        assert_tree_values_equal(actual, reference)
