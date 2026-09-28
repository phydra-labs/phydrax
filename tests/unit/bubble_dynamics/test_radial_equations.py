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
from phydrax.equations import TaitBarotropicMaterial


AMBIENT = 101325.0
DENSITY = 998.0
SOUND_SPEED = 1481.0


def _model(
    equation: bd.RadialBubbleEquation = "rayleigh_plesset",
    *,
    ambient: float = AMBIENT,
    gas: bd.AbstractBubbleGasLaw | None = None,
    viscosity: float = 0.0,
    tension: float = 0.0,
    sound_speed: float = SOUND_SPEED,
    vapor: float = 0.0,
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(1.4) if gas is None else gas,
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.CleanBubbleInterfaceLaw(tension),
        bd.BubbleEnvironment(ambient, 293.15, vapor_pressure=vapor),
        liquid_density=DENSITY,
        liquid_sound_speed=sound_speed,
    )


def test_rayleigh_collapse_matches_the_exact_empty_cavity_time() -> None:
    radius, pressure = 1.0e-3, 1.0e5
    ratio = 1.0e-2
    reference = float(bd.rayleigh_collapse_time(radius, pressure, DENSITY, final_radius_ratio=ratio))
    plan = bd.SingleBubblePlan(
        _model(ambient=0.0),
        bd.ConstantPressureDrive(pressure),
        np.linspace(0.0, 1.5 * reference, 7),
        events=bd.BubbleEventPolicy(minimum_radius_ratio=ratio, mach_limit=None),
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )
    result = bd.solve_single_bubble(plan.prepare(radius))

    assert int(result.status) == bd.BubbleDynamicsStatus.MINIMUM_RADIUS
    assert bool(result.successful)
    assert float(result.terminal_state.radius) == pytest.approx(ratio * radius, rel=1.0e-8)
    assert float(result.terminal_time) == pytest.approx(reference, rel=1.0e-8)
    unit = float(bd.rayleigh_collapse_time(1.0, 1.0, 1.0))
    assert unit == pytest.approx(0.914681, abs=5.0e-7)
    after_event = np.asarray(plan.save_times) > float(result.terminal_time)
    assert not np.any(np.asarray(result.trajectory.valid)[after_event])
    assert np.all(np.isnan(np.asarray(result.trajectory.radius)[after_event]))


def test_rayleigh_plesset_conserves_energy_without_damping_or_drive() -> None:
    kappa, tension, radius = 1.4, 0.072, 1.0e-4
    plan = bd.SingleBubblePlan(
        _model(gas=bd.PolytropicBubbleGasLaw(kappa), tension=tension),
        bd.ConstantPressureDrive(0.0),
        np.linspace(0.0, 4.0e-5, 81),
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
    )
    prepared = plan.prepare(radius, initial_radius=1.4 * radius)
    result = bd.solve_single_bubble(prepared)
    assert bool(result.completed)

    r = np.asarray(result.trajectory.radius)
    u = np.asarray(result.trajectory.wall_velocity)
    volume = 4.0 * np.pi * r**3 / 3.0
    reference_volume = 4.0 * np.pi * radius**3 / 3.0
    gas_pressure = float(prepared.equilibrium.gas_pressure)
    energy = (
        2.0 * np.pi * DENSITY * r**3 * u**2
        + AMBIENT * volume
        + 4.0 * np.pi * tension * r**2
        + gas_pressure * reference_volume**kappa * volume ** (1.0 - kappa) / (kappa - 1.0)
    )
    swing = np.max(2.0 * np.pi * DENSITY * r**3 * u**2)
    assert np.ptp(r) > 0.5 * radius
    assert np.max(np.abs(energy - energy[0])) < 1.0e-7 * swing
    assert result.evidence.work_identity_exact
    assert abs(float(result.evidence.work_residual)) < 1.0e-7 * swing


def _driven_radius(model: bd.RadialBubbleModel, times: np.ndarray, amplitude: float) -> np.ndarray:
    plan = bd.SingleBubblePlan(
        model,
        bd.HarmonicPressureDrive(amplitude, 2.0 * np.pi * 2.0e5),
        times,
        relative_tolerance=1.0e-10,
        absolute_tolerance=1.0e-12,
    )
    result = bd.solve_single_bubble(plan.prepare(1.0e-5))
    assert bool(result.completed)
    return np.asarray(result.trajectory.radius)


def test_keller_miksis_converges_to_rayleigh_plesset_as_sound_speed_grows() -> None:
    times = np.linspace(0.0, 2.0e-5, 101)
    reference = _driven_radius(_model(viscosity=1.0e-3), times, 3.0e4)
    differences = [
        np.max(np.abs(_driven_radius(_model("keller_miksis", viscosity=1.0e-3, sound_speed=c), times, 3.0e4) - reference))
        for c in (SOUND_SPEED, 10.0 * SOUND_SPEED, 100.0 * SOUND_SPEED)
    ]
    assert differences[0] > 0.0
    assert differences[1] / differences[0] == pytest.approx(0.1, rel=0.1)
    assert differences[2] / differences[1] == pytest.approx(0.1, rel=0.1)


def test_gilmore_reduces_to_keller_miksis_at_weak_compressibility() -> None:
    times = np.linspace(0.0, 2.0e-5, 101)
    material = TaitBarotropicMaterial(DENSITY, SOUND_SPEED, exponent=7.15, background_pressure=AMBIENT)
    gilmore = bd.RadialBubbleModel(
        "gilmore",
        bd.PolytropicBubbleGasLaw(1.4),
        bd.NewtonianBubbleLiquidLaw(1.0e-3),
        bd.CleanBubbleInterfaceLaw(0.0),
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_material=material,
    )
    keller = _model("keller_miksis", viscosity=1.0e-3)
    swing = _driven_radius(keller, times, 1.0e4)
    difference = np.max(np.abs(_driven_radius(gilmore, times, 1.0e4) - swing))
    assert difference < 1.0e-4 * np.ptp(swing)


def test_implicit_acceleration_matches_hand_derived_keller_miksis() -> None:
    kappa, tension, viscosity = 1.3, 0.05, 2.0e-3
    model = _model("keller_miksis", gas=bd.PolytropicBubbleGasLaw(kappa), viscosity=viscosity, tension=tension)
    drive = bd.HarmonicPressureDrive(4.0e4, 2.0e6, phase=0.3)
    equilibrium = model.equilibrium(3.0e-6)
    radius, velocity, time = 3.7e-6, -2.5, 1.3e-7
    state = bd.BubbleState(
        jnp.asarray(radius),
        jnp.asarray(velocity),
        equilibrium.state.gas,
        equilibrium.state.liquid,
        equilibrium.state.interface,
    )
    acceleration = float(model.rates(state, equilibrium.regime, jnp.asarray(time), drive).acceleration)

    gas0 = float(equilibrium.gas_pressure)
    gas = gas0 * (3.0e-6 / radius) ** (3.0 * kappa)
    gas_rate = -3.0 * kappa * gas * velocity / radius
    forcing = drive.evaluate(jnp.asarray(time))
    far = AMBIENT + float(forcing.pressure)
    wall = gas - 2.0 * tension / radius - 4.0 * viscosity * velocity / radius
    c = SOUND_SPEED
    explicit = gas_rate + 2.0 * tension * velocity / radius**2 + 4.0 * viscosity * velocity**2 / radius**2
    numerator = (
        (1.0 + velocity / c) * (wall - far) / DENSITY
        - 1.5 * (1.0 - velocity / (3.0 * c)) * velocity**2
        + radius / (DENSITY * c) * (explicit - float(forcing.pressure_rate))
    )
    denominator = (1.0 - velocity / c) * radius + 4.0 * viscosity / (DENSITY * c)
    assert acceleration == pytest.approx(numerator / denominator, rel=1.0e-11)


def test_gas_radiation_matches_marmottant_equation_three() -> None:
    kappa, viscosity, elasticity, initial, shell_viscosity = 1.095, 1.0e-3, 0.5, 0.02, 1.5e-8
    equilibrium_radius = 2.0e-6
    model = bd.RadialBubbleModel(
        "rayleigh_plesset_gas_radiation",
        bd.PolytropicBubbleGasLaw(kappa),
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.MarmottantShell(elasticity, initial, 0.072, shell_viscosity),
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=SOUND_SPEED,
    )
    drive = bd.HarmonicPressureDrive(4.0e4, 2.0e6, phase=0.3)
    equilibrium = model.equilibrium(equilibrium_radius)
    radius, velocity, time = 2.05e-6, -3.5, 1.3e-7
    state = bd.BubbleState(
        jnp.asarray(radius),
        jnp.asarray(velocity),
        equilibrium.state.gas,
        equilibrium.state.liquid,
        equilibrium.state.interface,
    )
    acceleration = float(model.rates(state, equilibrium.regime, jnp.asarray(time), drive).acceleration)

    buckling = equilibrium_radius / np.sqrt(1.0 + initial / elasticity)
    tension = elasticity * (radius**2 / buckling**2 - 1.0)
    gas = (AMBIENT + 2.0 * initial / equilibrium_radius) * (equilibrium_radius / radius) ** (3.0 * kappa)
    acoustic = float(drive.evaluate(jnp.asarray(time)).pressure)
    right = (
        gas * (1.0 - 3.0 * kappa * velocity / SOUND_SPEED)
        - AMBIENT
        - 2.0 * tension / radius
        - 4.0 * viscosity * velocity / radius
        - 4.0 * shell_viscosity * velocity / radius**2
        - acoustic
    )
    expected = (right / DENSITY - 1.5 * velocity**2) / radius
    assert acceleration == pytest.approx(expected, rel=1.0e-11)


def test_gas_radiation_equals_wall_radiation_when_only_gas_pressure_varies() -> None:
    drive = bd.ConstantPressureDrive(2.0e4)
    radius, velocity, time = 1.3e-5, 7.0, 2.0e-6
    accelerations = []
    equations: tuple[bd.RadialBubbleEquation, ...] = (
        "rayleigh_plesset_gas_radiation",
        "rayleigh_plesset_radiation",
    )
    for equation in equations:
        model = _model(equation, vapor=2.3e3)
        equilibrium = model.equilibrium(1.0e-5)
        state = bd.BubbleState(
            jnp.asarray(radius),
            jnp.asarray(velocity),
            equilibrium.state.gas,
            equilibrium.state.liquid,
            equilibrium.state.interface,
        )
        rates = model.rates(state, equilibrium.regime, jnp.asarray(time), drive)
        accelerations.append(float(rates.acceleration))
    assert accelerations[0] == pytest.approx(accelerations[1], rel=1.0e-12)


def _final_radius_plan(differentiation: bd.BubbleDifferentiation) -> bd.SingleBubblePlan:
    return bd.SingleBubblePlan(
        _model("keller_miksis", viscosity=1.0e-3, tension=0.072),
        bd.HarmonicPressureDrive(5.0e4, 2.0 * np.pi * 3.0e5),
        np.linspace(0.0, 6.0e-6, 4),
        relative_tolerance=1.0e-11,
        absolute_tolerance=1.0e-13,
        differentiation=differentiation,
    )


def _final_radius(plan: bd.SingleBubblePlan, parameters: Any) -> Any:
    viscosity, amplitude = parameters
    updated = eqx.tree_at(
        lambda current: (current.model.liquid.viscosity, current.drive.amplitude),
        plan,
        (viscosity, amplitude),
    )
    return bd.solve_single_bubble(updated.prepare(1.0e-5)).terminal_state.radius


def test_final_radius_jvp_and_vjp_match_central_differences() -> None:
    reverse = _final_radius_plan("reverse")
    forward = _final_radius_plan("forward")
    point = (jnp.asarray(1.0e-3), jnp.asarray(5.0e4))
    gradient = jax.grad(lambda parameters: _final_radius(reverse, parameters))(point)
    direction = (jnp.asarray(2.0e-4), jnp.asarray(1.0e3))
    _, tangent = jax.jvp(lambda parameters: _final_radius(forward, parameters), (point,), (direction,))
    for index, step in enumerate((1.0e-7, 5.0)):
        shift = [jnp.asarray(0.0), jnp.asarray(0.0)]
        shift[index] = jnp.asarray(step)
        plus = _final_radius(reverse, (point[0] + shift[0], point[1] + shift[1]))
        minus = _final_radius(reverse, (point[0] - shift[0], point[1] - shift[1]))
        central = float(plus - minus) / (2.0 * step)
        assert float(gradient[index]) == pytest.approx(central, rel=2.0e-5)
    projected = float(gradient[0]) * 2.0e-4 + float(gradient[1]) * 1.0e3
    assert float(tangent) == pytest.approx(projected, rel=1.0e-8)


def test_failures_return_the_last_accepted_state() -> None:
    plan = bd.SingleBubblePlan(
        _model(viscosity=1.0e-3),
        bd.HarmonicPressureDrive(5.0e4, 2.0 * np.pi * 2.0e5),
        np.linspace(0.0, 1.0e-5, 11),
        maximum_steps=6,
    )
    result = bd.solve_single_bubble(plan.prepare(1.0e-5))
    assert int(result.status) == bd.BubbleDynamicsStatus.MAX_STEPS
    assert not bool(result.successful)
    assert float(result.terminal_time) < 1.0e-5
    assert np.isfinite(float(result.terminal_state.radius))
    assert int(result.evidence.accepted_steps) <= 6

    invalid = bd.SingleBubblePlan(
        _model(ambient=1.0e3, vapor=5.0e3),
        bd.ConstantPressureDrive(0.0),
        np.linspace(0.0, 1.0e-6, 3),
    )
    refused = bd.solve_single_bubble(invalid.prepare(1.0e-3))
    assert int(refused.status) == bd.BubbleDynamicsStatus.INVALID_EQUILIBRIUM
    assert float(refused.terminal_time) == 0.0
    assert float(refused.terminal_state.radius) == pytest.approx(1.0e-3)
    assert not np.any(np.asarray(refused.trajectory.valid))


def test_terminal_events_report_mach_hard_core_and_support_exit() -> None:
    radius, pressure = 1.0e-3, 1.0e5
    mach_plan = bd.SingleBubblePlan(
        _model(ambient=0.0),
        bd.ConstantPressureDrive(pressure),
        np.linspace(0.0, 2.0e-4, 5),
        events=bd.BubbleEventPolicy(mach_limit=0.05),
    )
    mach = bd.solve_single_bubble(mach_plan.prepare(radius))
    assert int(mach.status) == bd.BubbleDynamicsStatus.MACH_LIMIT
    assert abs(float(mach.terminal_state.wall_velocity)) == pytest.approx(0.05 * SOUND_SPEED, rel=1.0e-6)

    core_plan = bd.SingleBubblePlan(
        _model(gas=bd.HardCorePolytropicBubbleGasLaw(1.4, 0.5)),
        bd.ConstantPressureDrive(1.0e7),
        np.linspace(0.0, 1.0e-4, 5),
        events=bd.BubbleEventPolicy(hard_core_margin=0.05, mach_limit=None),
    )
    core = bd.solve_single_bubble(core_plan.prepare(radius))
    assert int(core.status) == bd.BubbleDynamicsStatus.HARD_CORE
    assert float(core.evidence.validity.min_hard_core_margin) == pytest.approx(0.05, rel=1.0e-6)

    times = np.linspace(0.0, 5.0e-6, 21)
    sampled = bd.SampledPressureDrive(times[:11], 1.0e4 * np.sin(4.0e5 * times[:11]))
    support_plan = bd.SingleBubblePlan(_model(viscosity=1.0e-3), sampled, times)
    exited = bd.solve_single_bubble(support_plan.prepare(1.0e-5))
    assert int(exited.status) == bd.BubbleDynamicsStatus.SUPPORT_EXIT
    assert float(exited.terminal_time) >= times[10]
    assert bool(np.all(np.asarray(exited.trajectory.valid)[:11]))
