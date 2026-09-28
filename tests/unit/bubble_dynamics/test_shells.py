#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax.bubble_dynamics as bd


AMBIENT = 101325.0
RADIUS = 2.0e-6
FREQUENCY = 2.9e6


def _marmottant_plan(
    shell: bd.AbstractBubbleInterfaceLaw,
    amplitude: float,
    *,
    tolerance: float = 1.0e-9,
    cycles: float = 3.0,
) -> bd.SingleBubblePlan:
    model = bd.RadialBubbleModel(
        "keller_miksis",
        bd.PolytropicBubbleGasLaw(1.07),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        shell,
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=1000.0,
        liquid_sound_speed=1500.0,
    )
    return bd.SingleBubblePlan(
        model,
        bd.HarmonicPressureDrive(amplitude, 2.0 * np.pi * FREQUENCY),
        np.linspace(0.0, cycles / FREQUENCY, 31),
        relative_tolerance=tolerance,
        absolute_tolerance=1.0e-2 * tolerance,
    )


def _transitions(result: bd.SingleBubbleResult) -> tuple[np.ndarray, ...]:
    tape = result.evidence.regime_tape
    count = int(tape.count)
    return (
        np.asarray(tape.times)[:count],
        np.asarray(tape.from_regime)[:count],
        np.asarray(tape.to_regime)[:count],
        np.asarray(tape.radius)[:count],
    )


def test_marmottant_regime_changes_occur_at_buckling_and_rupture_radii() -> None:
    shell = bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8)
    plan = _marmottant_plan(shell, 5.0e4)
    prepared = plan.prepare(RADIUS)
    result = bd.solve_single_bubble(prepared)
    assert bool(result.completed)
    times, before, after, radii = _transitions(result)
    buckling, rupture, _ = (float(value) for value in shell.characteristic_radii(prepared.equilibrium.state.interface))
    assert buckling == pytest.approx(RADIUS / np.sqrt(1.0 + 0.02 / 0.55), rel=1.0e-14)
    assert rupture == pytest.approx(buckling * np.sqrt(1.0 + 0.072 / 0.55), rel=1.0e-14)
    assert times.shape[0] >= 4
    assert np.all(np.diff(times) > 0.0)
    for start, end, radius in zip(before, after, radii, strict=True):
        assert abs(int(start) - int(end)) == 1
        boundary = buckling if {int(start), int(end)} == {0, 1} else rupture
        assert radius == pytest.approx(boundary, rel=5.0e-7)
    # The shell buckles on the first compression before it ever ruptures.
    assert (int(before[0]), int(after[0])) == (1, 0)
    assert bool(result.evidence.derivative_available)
    np.testing.assert_array_equal(
        np.asarray(result.trajectory.regime)[:1], np.asarray([1], dtype=np.int32)
    )


def test_marmottant_event_times_converge_under_tolerance_refinement() -> None:
    shell = bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8)
    coarse, fine, finest = (
        _transitions(bd.solve_single_bubble(_marmottant_plan(shell, 5.0e4, tolerance=tolerance).prepare(RADIUS)))[0]
        for tolerance in (1.0e-6, 1.0e-8, 1.0e-10)
    )
    count = min(coarse.shape[0], fine.shape[0], finest.shape[0])
    assert count >= 4
    coarse_error = np.max(np.abs(coarse[:count] - finest[:count]))
    fine_error = np.max(np.abs(fine[:count] - finest[:count]))
    assert fine_error < 0.1 * coarse_error
    assert fine_error * FREQUENCY < 1.0e-6


def test_irreversible_rupture_switches_to_the_broken_branch() -> None:
    shell = bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8, rupture_surface_tension=0.1)
    result = bd.solve_single_bubble(_marmottant_plan(shell, 1.5e5, cycles=2.0).prepare(RADIUS))
    assert bool(result.completed)
    _, before, after, radii = _transitions(result)
    breakups = np.flatnonzero((before == 1) & (after == 5))
    assert breakups.size == 1
    prepared_radii = shell.characteristic_radii(jnp.asarray([RADIUS]))
    assert radii[breakups[0]] == pytest.approx(float(prepared_radii[2]), rel=5.0e-7)
    assert np.all(after[breakups[0] :] >= 3)
    assert shell.regime_names[5] == "broken-ruptured"


def test_gompertz_marmottant_matches_initial_tension_and_marmottant_peak_slope() -> None:
    chi, initial, liquid = 0.5, 0.02, 0.072
    shell = bd.GompertzMarmottantShell(chi, initial, liquid, 1.0e-8)
    internal = shell.initialize(jnp.asarray(RADIUS), bd.BubbleEnvironment(AMBIENT, 293.15))
    assert float(shell.tension_at(jnp.asarray(RADIUS), internal)) == pytest.approx(initial, rel=1.0e-12)
    buckling = RADIUS / np.sqrt(1.0 + initial / chi)
    radii = jnp.linspace(0.8 * buckling, 1.5 * buckling, 40001)
    slopes = jax.vmap(jax.grad(lambda radius: shell.tension_at(radius, internal)))(radii)
    peak = buckling * np.sqrt(1.0 + liquid / (2.0 * chi))
    assert float(jnp.max(slopes)) == pytest.approx(2.0 * chi * peak / buckling**2, rel=1.0e-6)
    assert float(shell.tension_at(jnp.asarray(10.0 * RADIUS), internal)) == pytest.approx(liquid, rel=1.0e-6)


def test_shell_elasticity_and_viscosity_gradients_match_central_differences() -> None:
    plan = _marmottant_plan(bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8), 5.0e4, tolerance=1.0e-11, cycles=1.5)

    def final_radius(parameters: Any) -> Any:
        elasticity, viscosity = parameters
        updated = eqx.tree_at(
            lambda current: (
                current.model.interface.shell_elasticity,
                current.model.interface.shell_viscosity,
            ),
            plan,
            (elasticity, viscosity),
        )
        return bd.solve_single_bubble(updated.prepare(RADIUS)).terminal_state.radius

    point = (jnp.asarray(0.55), jnp.asarray(1.5e-8))
    gradient = jax.grad(final_radius)(point)
    for index, step in enumerate((1.0e-5, 1.0e-13)):
        shift = [0.0, 0.0]
        shift[index] = step
        plus = final_radius((point[0] + shift[0], point[1] + shift[1]))
        minus = final_radius((point[0] - shift[0], point[1] - shift[1]))
        assert float(gradient[index]) == pytest.approx(float(plus - minus) / (2.0 * step), rel=1.0e-4)


def test_regime_capacity_refusal_is_reported() -> None:
    shell = bd.MarmottantShell(0.55, 0.02, 0.072, 1.5e-8)
    base = _marmottant_plan(shell, 5.0e4)
    plan = bd.SingleBubblePlan(
        base.model,
        base.drive,
        np.asarray(base.save_times),
        events=bd.BubbleEventPolicy(regime_capacity=2),
    )
    result = bd.solve_single_bubble(plan.prepare(RADIUS))
    assert int(result.status) == bd.BubbleDynamicsStatus.REGIME_CAPACITY
    assert bool(result.evidence.regime_tape.capacity_exceeded)
    assert int(result.evidence.regime_tape.count) == 2
    assert not bool(result.successful)
