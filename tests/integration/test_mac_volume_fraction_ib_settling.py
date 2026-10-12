#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Real VF-IBM trajectory versus an independent CGW standard sphere-drag oracle.

Clift, Grace & Weber, Bubbles, Drops, and Particles, Academic Press (1978),
ISBN 9780121769505. Low-Re coefficients independently checked against R. S.
Subramanian's Clarkson transcription, "Drag on spherical particles and steady
settling velocities":
https://lin-web.clarkson.edu/projects/subramanian/ch330/notes/Drag%20on%20Spherical%20Particles%20and%20Steady%20Settling%20Velocities.pdf
The book bibliography is independently confirmed by the 1979 JFM review:
https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/abs/bubbles-drops-and-particles-by-r-clift-j-r-grace-and-m-e-weber-academic-press-1978-380-pp-2080-or-3200/B715AAB3FCD2C95EEC0FF1D6A817B1D0

This is candidate, not grid-converged qualification: h/D=1/4, box width=9D,
subbox width=h/4. The 55% relative acceptance budget accounts for unresolved
interface stress and finite periodic-box drag, not sampling noise; it is not
an observed convergence rate. The no-drag free-fall trajectory fails this
budget. Full qualification requires larger boxes and h/D refinement.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from examples.mac_volume_fraction_ib_cfd_dem import (
    BOX_LENGTH,
    DIAMETER,
    DYNAMIC_VISCOSITY,
    FLUID_DENSITY,
    GRAVITY,
    make_settling_case,
    settle,
    SettlingTrajectory,
    SOLID_DENSITY,
    STEP_SIZE,
)


pytestmark = pytest.mark.strict_jax

RESOLUTION_RELATIVE_TOLERANCE = 0.55


def _drag(speed: float) -> float:
    """Independent host reference; evaluates force regularly at zero speed."""
    reynolds = FLUID_DENSITY * DIAMETER * abs(speed) / DYNAMIC_VISCOSITY
    if reynolds > 20.0:
        raise ValueError("The selected CGW oracle is restricted to Re <= 20.")
    factor = (
        1.0 + 3.0 * reynolds / 16.0
        if reynolds <= 0.01
        else 1.0 + 0.1315 * reynolds ** (0.82 - 0.05 * np.log10(reynolds))
    )
    return 3.0 * np.pi * DYNAMIC_VISCOSITY * DIAMETER * speed * factor


def _reference(
    times: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    volume = np.pi * DIAMETER**3 / 6.0
    mass = SOLID_DENSITY * volume
    net_weight = (SOLID_DENSITY - FLUID_DENSITY) * volume * GRAVITY

    def rate(_time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(
            ((net_weight - _drag(float(state[0]))) / mass, state[0]), dtype=np.float64
        )

    solution = solve_ivp(
        rate,
        (0.0, float(times[-1])),
        np.zeros(2, dtype=np.float64),
        t_eval=times,
        rtol=1e-11,
        atol=1e-13,
        max_step=STEP_SIZE / 4.0,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    terminal = brentq(lambda value: _drag(value) - net_weight, 0.0, 1.0, xtol=1e-13)
    return solution.y[0], solution.y[1], terminal


@pytest.fixture(scope="module")
def trajectory() -> SettlingTrajectory:
    coupling, initial = make_settling_case(cells=36)
    result = settle(coupling, initial, steps=800)
    assert np.all(np.asarray(result.successful))
    assert np.all(np.asarray(result.projection_successful))
    assert result.state.accepted_steps == 800
    assert np.all(np.isfinite(np.asarray(result.position)))
    assert np.all(np.asarray(result.position) > DIAMETER / 2.0)
    assert np.all(np.asarray(result.position) < BOX_LENGTH - DIAMETER / 2.0)
    return result


def test_real_sphere_settling_trajectory_matches_independent_cgw_reference(
    trajectory: SettlingTrajectory,
) -> None:
    indices = np.asarray((199, 399, 799), dtype=np.int64)
    times = np.asarray(trajectory.time)[indices]
    reference_speed, reference_distance, terminal = _reference(times)
    speed = np.asarray(trajectory.relative_speed)[indices]
    distance = 2.0 - np.asarray(trajectory.position)[indices, 2]
    assert np.all(speed > 0.0) and np.all(distance > 0.0)
    np.testing.assert_allclose(
        speed, reference_speed, rtol=RESOLUTION_RELATIVE_TOLERANCE, atol=0.0
    )
    np.testing.assert_allclose(
        distance, reference_distance, rtol=RESOLUTION_RELATIVE_TOLERANCE, atol=0.0
    )
    # Deliberate independent fault adequacy: an absent hydrodynamic reaction
    # cannot satisfy even this coarse-grid candidate trajectory budget.
    free_fall_distance = 0.5 * GRAVITY * (1.0 - FLUID_DENSITY / SOLID_DENSITY) * times**2
    assert (
        np.max(np.abs(free_fall_distance / reference_distance - 1.0))
        > RESOLUTION_RELATIVE_TOLERANCE
    )
    assert FLUID_DENSITY * DIAMETER * terminal / DYNAMIC_VISCOSITY < 20.0


def test_late_settling_speed_and_balance_retain_real_fluid_evidence(
    trajectory: SettlingTrajectory,
) -> None:
    _, _, terminal = _reference(np.asarray((2.0,), dtype=np.float64))
    late = np.asarray(trajectory.relative_speed)[-100:]
    np.testing.assert_allclose(
        float(np.mean(late)), terminal, rtol=RESOLUTION_RELATIVE_TOLERANCE, atol=0.0
    )
    assert abs(float(late[-1] - late[0])) < 0.15 * terminal
    volume = np.pi * DIAMETER**3 / 6.0
    net_weight = (SOLID_DENSITY - FLUID_DENSITY) * volume * GRAVITY
    late_drag = np.asarray(trajectory.hydrodynamic_drag)[-100:]
    np.testing.assert_allclose(
        float(np.mean(late_drag)),
        _drag(float(np.mean(late))),
        rtol=RESOLUTION_RELATIVE_TOLERANCE,
        atol=0.0,
    )
    np.testing.assert_allclose(float(np.mean(late_drag)), net_weight, rtol=0.15, atol=0.0)
    np.testing.assert_allclose(trajectory.impulse_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(trajectory.torque_residual, 0.0, atol=1e-10)
    np.testing.assert_allclose(np.asarray(trajectory.position)[:, :2], 1.5, atol=1e-9)
    # The coarse geometry is explicit evidence, not silently called exact.
    assert np.max(np.asarray(trajectory.geometry_unresolved)) > 0.02


def test_example_smoke_advances_real_coupling_without_drag_closure() -> None:
    coupling, initial = make_settling_case(cells=12)
    result = settle(coupling, initial, steps=4)
    assert np.all(np.asarray(result.successful))
    assert result.state.accepted_steps == 4
    assert result.position[-1, 2] < initial.dem_state.kinematics.position[0, 2]
    assert (
        np.linalg.norm(np.asarray(result.state.fluid_state - initial.fluid_state)) > 0.0
    )
