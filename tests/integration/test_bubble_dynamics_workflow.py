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


RADIUS = 2.0e-6
FREQUENCY = 2.5e6


def _model(interface: bd.AbstractBubbleInterfaceLaw) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        "keller_miksis",
        bd.PolytropicBubbleGasLaw(1.07),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        interface,
        bd.BubbleEnvironment(101325.0, 293.15),
        liquid_density=1000.0,
        liquid_sound_speed=1500.0,
    )


def _expansion_to_compression(result: bd.SingleBubbleResult) -> float:
    radius = np.asarray(result.trajectory.radius)
    return float((radius.max() - RADIUS) / (RADIUS - radius.min()))


def test_initially_buckled_shell_oscillates_compression_dominated() -> None:
    drive = bd.PulsedPressureDrive(8.0e4, 2.0 * np.pi * FREQUENCY, 6.0)
    times = np.linspace(0.0, 8.0 / FREQUENCY, 321)
    clean = bd.solve_single_bubble(
        bd.SingleBubblePlan(_model(bd.CleanBubbleInterfaceLaw(0.072)), drive, times).prepare(RADIUS)
    )
    coated = bd.solve_single_bubble(
        bd.SingleBubblePlan(_model(bd.MarmottantShell(1.0, 0.0, 0.072, 1.0e-8)), drive, times).prepare(RADIUS)
    )
    assert bool(clean.successful) and bool(coated.successful)
    assert _expansion_to_compression(clean) > 1.0
    assert _expansion_to_compression(coated) < 0.7 * _expansion_to_compression(clean)
    tape = coated.evidence.regime_tape
    assert int(tape.count) > 0
    assert bool(jnp.any((tape.to_regime == 0) & tape.active))


def test_small_amplitude_ringdown_matches_linear_response() -> None:
    shell = bd.MarmottantShell(0.5, 0.02, 0.072, 5.0e-9)
    model = _model(shell)
    linear = bd.linear_bubble_response(model, RADIUS, 2.0 * np.pi * FREQUENCY)
    resonance = float(linear.resonance_frequency)
    response = bd.linear_bubble_response(model, RADIUS, resonance)
    damping = float(response.total_damping[0])
    period = 2.0 * np.pi / resonance
    times = np.linspace(0.0, 6.0 * period, 2401)
    plan = bd.SingleBubblePlan(
        model,
        bd.ConstantPressureDrive(0.0),
        times,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )
    result = bd.solve_single_bubble(plan.prepare(RADIUS, initial_radius=RADIUS * (1.0 + 2.0e-4)))
    assert bool(result.completed)
    assert int(result.evidence.regime_tape.count) == 0
    deviation = np.asarray(result.trajectory.radius) - RADIUS
    crossings = np.flatnonzero((deviation[:-1] > 0.0) & (deviation[1:] <= 0.0))
    fractions = deviation[crossings] / (deviation[crossings] - deviation[crossings + 1])
    crossing_times = times[crossings] + fractions * (times[1] - times[0])
    measured_period = float(np.mean(np.diff(crossing_times)))
    damped = np.sqrt(resonance**2 - damping**2)
    assert 2.0 * np.pi / measured_period == pytest.approx(damped, rel=5.0e-3)
    peaks = [
        float(np.max(deviation[(times >= start) & (times < start + measured_period)]))
        for start in (crossing_times[0], crossing_times[2])
    ]
    decay = np.log(peaks[0] / peaks[1]) / (2.0 * measured_period)
    assert decay == pytest.approx(damping, rel=3.0e-2)


def test_fitted_gompertz_elasticity_gradient_matches_finite_differences() -> None:
    plan = bd.SingleBubblePlan(
        _model(bd.GompertzMarmottantShell(0.5, 0.02, 0.072, 1.0e-8)),
        bd.HarmonicPressureDrive(3.0e4, 2.0 * np.pi * FREQUENCY),
        np.linspace(0.0, 2.0 / FREQUENCY, 41),
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
    )
    observed = np.asarray(bd.solve_single_bubble(plan.prepare(RADIUS)).trajectory.radius)

    def misfit(elasticity: Any) -> Any:
        updated = eqx.tree_at(lambda current: current.model.interface.shell_elasticity, plan, elasticity)
        radius = bd.solve_single_bubble(updated.prepare(RADIUS)).trajectory.radius
        return jnp.sum(((radius - observed) / RADIUS) ** 2)

    point = jnp.asarray(0.6)
    gradient = float(jax.grad(misfit)(point))
    step = 1.0e-5
    central = float(misfit(point + step) - misfit(point - step)) / (2.0 * step)
    assert gradient > 0.0
    assert gradient == pytest.approx(central, rel=1.0e-5)
