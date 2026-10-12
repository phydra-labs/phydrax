"""Uniform axial loading reduces connected RZ to generalized plane strain."""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import Array

from phydrax.applications.nuclear_fuel import (
    FuelFixedGrid,
    FuelGeometryReduction,
    FuelReplayResult,
    FuelRodState,
    replay_fuel_grid,
)
from phydrax.applications.nuclear_fuel._acceptance import FuelStepStatus
from tests._support.nuclear_fuel import prepare_low_power_startup_rod


# One 0.1 s step of zero-power wall heating, uniform along the single axial zone.
_STEP = 0.1  # [s]


def _replay(reduction: FuelGeometryReduction, /) -> FuelReplayResult:
    prepared = prepare_low_power_startup_rod(reduction=reduction)
    return replay_fuel_grid(prepared, FuelFixedGrid((0.0, _STEP), prepared.prepared_id))


@pytest.fixture(scope="module")
def replays() -> tuple[FuelReplayResult, FuelReplayResult]:
    with jax.enable_x64():
        return _replay("layered-radial"), _replay("axisymmetric-rz")


@pytest.mark.parametrize("index", [0, 1], ids=["gps", "rz"])
def test_uniform_load_step_is_accepted_by_each_reduction(
    replays: tuple[FuelReplayResult, FuelReplayResult], index: int
) -> None:
    replay = replays[index]
    (attempt,) = replay.attempts
    assert bool(replay.successful)
    assert bool(np.all(np.asarray(replay.nonlinear_success)))
    assert int(attempt.status) == FuelStepStatus.ACCEPTED


@pytest.mark.parametrize(
    ("observable", "rtol", "atol"),
    [
        pytest.param(lambda state: state.fields.pressure, 2e-5, 0.0, id="rod-pressure"),
        pytest.param(
            lambda state: jnp.max(state.fields.temperature_fuel),
            1e-7,
            1e-5,
            id="maximum-fuel-temperature",
        ),
        pytest.param(
            lambda state: state.energy.delivered_coolant,
            2e-4,
            1e-7,
            id="coolant-delivered-energy",
        ),
        pytest.param(
            lambda state: state.gas.retained_moles,
            2e-4,
            1e-15,
            id="retained-gas-moles",
        ),
    ],
)
def test_uniform_load_rz_matches_generalized_plane_strain(
    replays: tuple[FuelReplayResult, FuelReplayResult],
    observable: Callable[[FuelRodState], Array],
    rtol: float,
    atol: float,
) -> None:
    layered, rz = replays
    assert bool(layered.successful) and bool(rz.successful)
    np.testing.assert_allclose(
        observable(rz.states[-1]), observable(layered.states[-1]), rtol=rtol, atol=atol
    )
