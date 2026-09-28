#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Wake-function, impedance, and multi-bunch memory contracts.

References are independent of the implementation: resonator loss factor
``k = ω_r R_s / (2Q)`` and the closed-form resonator Fourier pair (Chao 1993,
ch. 2), the Bane–Sands short-range resistive-wall formula evaluated with SciPy
quadrature (SLAC-PUB-95-7074) and its classical long-range limits, and the
coupled-bunch growth rate from the transverse impedance sum over revolution
harmonics (Chao 1993, ch. 4).
"""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

import phydrax as phx
from phydrax.applications import accelerator


SCALE = phx.ElectromagneticScaleContract.si()
SPEED_OF_LIGHT = float(SCALE.speed_of_light)
VACUUM_IMPEDANCE = float(SCALE.vacuum_impedance)
ELEMENTARY_CHARGE = float(SCALE.elementary_charge)
ELECTRON_MASS = float(SCALE.electron_mass)
ELECTRON_REST_ENERGY = ELECTRON_MASS * SPEED_OF_LIGHT**2


def _bunch(
    coordinates: np.ndarray,
    weights: np.ndarray,
    *,
    momentum: float,
    charge: float = ELEMENTARY_CHARGE,
    bunch_id: str = "wake-test",
) -> accelerator.AcceleratorBunch:
    return accelerator.AcceleratorBunch(
        jnp.asarray(coordinates, dtype=jnp.float64),
        jnp.asarray(weights, dtype=jnp.float64),
        jnp.arange(coordinates.shape[0], dtype=jnp.int32),
        reference_rest_energy=ELECTRON_REST_ENERGY,
        reference_momentum=momentum,
        reference_charge=charge,
        bunch_id=bunch_id,
    )


def _resonator_pair(
    kind: accelerator.WakeKind, shunt: float, quality: float, frequency: float
) -> tuple[accelerator.ResonatorWake, float, float, float]:
    omega = 2.0 * math.pi * frequency
    alpha = omega / (2.0 * quality)
    shifted = math.sqrt(omega**2 - alpha**2)
    resonator = accelerator.ResonatorWake(
        shunt, quality, frequency, kind=kind, scale=SCALE
    )
    return resonator, omega, alpha, shifted


def _reference_longitudinal_wake(
    tau: np.ndarray, shunt: float, quality: float, omega: float
) -> np.ndarray:
    alpha = omega / (2.0 * quality)
    shifted = math.sqrt(omega**2 - alpha**2)
    return (
        (omega * shunt / quality)
        * np.exp(-alpha * tau)
        * (np.cos(shifted * tau) - alpha / shifted * np.sin(shifted * tau))
    )


def _reference_longitudinal_impedance(
    omega_samples: np.ndarray, shunt: float, quality: float, omega: float
) -> np.ndarray:
    return shunt / (1.0 + 1j * quality * (omega / omega_samples - omega_samples / omega))


def _reference_transverse_impedance(
    omega_samples: np.ndarray, shunt: float, quality: float, omega: float
) -> np.ndarray:
    return (
        (omega / omega_samples)
        * shunt
        / (1.0 + 1j * quality * (omega / omega_samples - omega_samples / omega))
    )


def test_point_bunch_loses_resonator_loss_factor() -> None:
    shunt, quality, frequency = 1.0e4, 5.0, 2.0e9
    resonator, omega, _, _ = _resonator_pair("longitudinal", shunt, quality, frequency)
    zeta = np.linspace(0.0, 0.05, 2001)
    plan = resonator.wake_function(zeta)
    momentum = 1.0e9 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT
    particles = 1.0e10
    bunch = _bunch(np.zeros((4, 6)), np.full(4, particles / 4), momentum=momentum)
    result = accelerator.apply_wake(plan, bunch, SCALE, bin_count=8)
    total_charge = particles * ELEMENTARY_CHARGE
    loss_factor = omega * shunt / (2.0 * quality)
    assert bool(result.finite)
    np.testing.assert_allclose(float(result.loss_factor), loss_factor, rtol=1.0e-12)
    np.testing.assert_allclose(
        float(result.energy_change), -(total_charge**2) * loss_factor, rtol=1.0e-12
    )
    np.testing.assert_allclose(resonator.loss_factor, loss_factor, rtol=1.0e-15)
    energy = math.sqrt((momentum * SPEED_OF_LIGHT) ** 2 + ELECTRON_REST_ENERGY**2)
    # Every physical electron in the bunch loses e·Q·k.
    expected_delta = (
        (-ELEMENTARY_CHARGE * total_charge * loss_factor)
        * energy
        / (momentum * SPEED_OF_LIGHT) ** 2
    )
    np.testing.assert_allclose(np.asarray(result.kicks), expected_delta, rtol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(result.bunch.coordinates[:, 5]), expected_delta, rtol=1.0e-12
    )


def test_wake_acts_only_behind_the_source() -> None:
    resonator, _, _, _ = _resonator_pair("longitudinal", 1.0e4, 5.0, 2.0e9)
    plan = resonator.wake_function(np.linspace(0.0, 0.05, 2001))
    momentum = 1.0e9 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT
    # Head witness (zeta = -1 cm), heavy source at zeta = 0, tail witness (+1 cm);
    # "positive-late" means larger zeta is behind.
    coordinates = np.zeros((3, 6))
    coordinates[:, 4] = (-0.01, 0.0, 0.01)
    bunch = _bunch(coordinates, np.asarray([1.0, 1.0e10, 1.0]), momentum=momentum)
    result = accelerator.apply_wake(plan, bunch, SCALE, bin_count=2)
    kicks = np.asarray(result.kicks)
    assert float(result.causality_defect) == 0.0
    # The head shares its bin with nobody but itself: only its own half self-wake.
    self_only = -(ELEMENTARY_CHARGE**2) * resonator.loss_factor
    energy = math.sqrt((momentum * SPEED_OF_LIGHT) ** 2 + ELECTRON_REST_ENERGY**2)
    np.testing.assert_allclose(
        kicks[0], self_only * energy / (momentum * SPEED_OF_LIGHT) ** 2, rtol=1.0e-12
    )
    assert abs(kicks[2]) > 1.0e6 * abs(kicks[0])
    early = accelerator.AcceleratorConvention(longitudinal_sign="positive-early")
    flipped = accelerator.AcceleratorBunch(
        bunch.coordinates,
        bunch.weights,
        bunch.particle_ids,
        reference_rest_energy=ELECTRON_REST_ENERGY,
        reference_momentum=momentum,
        reference_charge=ELEMENTARY_CHARGE,
        convention=early,
        bunch_id="flipped",
    )
    flipped_result = accelerator.apply_wake(plan, flipped, SCALE, bin_count=2)
    flipped_kicks = np.asarray(flipped_result.kicks)
    np.testing.assert_allclose(flipped_kicks[2], kicks[0], rtol=1.0e-12)
    np.testing.assert_allclose(flipped_kicks[0], kicks[2], rtol=1.0e-12)


def test_resonator_wake_and_impedance_are_a_fourier_pair() -> None:
    shunt, quality, frequency = 2.0e3, 4.0, 1.0e9
    resonator, omega, _, _ = _resonator_pair("longitudinal", shunt, quality, frequency)
    # Twelve meters is about 31 decay lengths c/alpha: truncation is negligible.
    zeta = np.linspace(0.0, 12.0, 120001)
    plan = resonator.wake_function(zeta)
    np.testing.assert_allclose(
        np.asarray(plan.wake_values),
        _reference_longitudinal_wake(zeta / SPEED_OF_LIGHT, shunt, quality, omega),
        rtol=1.0e-13,
        atol=1.0e-13 * shunt * omega,
    )
    frequencies = np.linspace(0.05e9, 6.0e9, 240)
    result = accelerator.wake_impedance(plan, frequencies, scale=SCALE)
    reference = _reference_longitudinal_impedance(
        2.0 * math.pi * frequencies, shunt, quality, omega
    )
    assert result.units == "ohm"
    assert bool(jnp.all(result.supported))
    np.testing.assert_allclose(
        np.asarray(result.impedance), reference, atol=2.0e-4 * shunt
    )
    np.testing.assert_allclose(
        np.asarray(resonator.impedance(frequencies)), reference, rtol=1.0e-13
    )
    band = np.linspace(0.0, 400.0e9, 800001)
    inversion = accelerator.wake_from_impedance(
        "longitudinal", band, resonator.impedance(band), zeta[::20], scale=SCALE
    )
    assert bool(inversion.supported)
    recovered = np.asarray(inversion.plan.wake_values)
    expected = np.asarray(plan.wake_values)[::20]
    # The origin carries the causal jump: the raw inverse transform returns its
    # midpoint W(0+)/2. Away from the jump the band-limited inverse converges.
    np.testing.assert_allclose(
        float(inversion.origin_value), 0.5 * expected[0], rtol=1.0e-3
    )
    np.testing.assert_allclose(recovered[0], expected[0], rtol=1.0e-3)
    np.testing.assert_allclose(recovered[10:], expected[10:], atol=5.0e-3 * expected[0])
    round_trip = accelerator.wake_impedance(inversion.plan, frequencies, scale=SCALE)
    np.testing.assert_allclose(
        np.asarray(round_trip.impedance), reference, atol=5.0e-3 * shunt
    )


def test_transverse_resonator_impedance_matches_closed_form() -> None:
    shunt, quality, frequency = 5.0e5, 3.0, 1.5e9
    resonator, omega, _, _ = _resonator_pair("dipolar-x", shunt, quality, frequency)
    zeta = np.linspace(0.0, 3.0, 120001)
    plan = resonator.wake_function(zeta)
    assert plan.units == "V/C/m"
    assert float(plan.wake_values[0]) == 0.0
    frequencies = np.linspace(0.1e9, 5.0e9, 150)
    result = accelerator.wake_impedance(plan, frequencies, scale=SCALE)
    reference = _reference_transverse_impedance(
        2.0 * math.pi * frequencies, shunt, quality, omega
    )
    assert result.units == "ohm/m"
    np.testing.assert_allclose(
        np.asarray(result.impedance), reference, atol=2.0e-4 * shunt
    )
    np.testing.assert_allclose(
        np.asarray(resonator.impedance(frequencies)), reference, rtol=1.0e-13
    )


def _bane_sands_bracket(scaled: float, *, transverse: bool) -> float:
    if transverse:
        integral = quad(
            lambda x: (1.0 - math.exp(-(x**2) * scaled)) / (x**6 + 8.0), 0.0, np.inf
        )[0]
        return (
            1.0
            - math.exp(-scaled) * math.cos(math.sqrt(3.0) * scaled)
            + math.sqrt(3.0) * math.exp(-scaled) * math.sin(math.sqrt(3.0) * scaled)
        ) / 12.0 - math.sqrt(2.0) / math.pi * integral
    integral = quad(
        lambda x: x**2 * math.exp(-(x**2) * scaled) / (x**6 + 8.0), 0.0, np.inf
    )[0]
    return (
        math.exp(-scaled) * math.cos(math.sqrt(3.0) * scaled) / 3.0
        - math.sqrt(2.0) / math.pi * integral
    )


def test_resistive_wall_matches_bane_sands_and_long_range_forms() -> None:
    radius, conductivity, length = 1.0e-2, 5.8e7, 2.0
    wall = accelerator.ResistiveWallWake(radius, conductivity, length, scale=SCALE)
    s0 = (2.0 * radius**2 / (VACUUM_IMPEDANCE * conductivity)) ** (1.0 / 3.0)
    np.testing.assert_allclose(wall.characteristic_length, s0, rtol=1.0e-14)
    zeta = np.concatenate(([0.0, 0.25 * s0, s0, 3.0 * s0, 20.0 * s0], [1000.0 * s0, 0.5]))
    longitudinal = wall.wake_function("longitudinal", zeta)
    transverse = wall.wake_function("dipolar-y", zeta)
    values = np.asarray(longitudinal.wake_values)
    values_t = np.asarray(transverse.wake_values)
    # W(0+) = Z0 c / (pi a^2) per length; W_perp(0+) = 0.
    np.testing.assert_allclose(
        values[0], length * VACUUM_IMPEDANCE * SPEED_OF_LIGHT / (math.pi * radius**2)
    )
    assert values_t[0] == 0.0
    for index, scaled in ((1, 0.25), (2, 1.0), (3, 3.0), (4, 20.0)):
        expected = (
            length
            * 4.0
            * VACUUM_IMPEDANCE
            * SPEED_OF_LIGHT
            / (math.pi * radius**2)
            * _bane_sands_bracket(scaled, transverse=False)
        )
        np.testing.assert_allclose(values[index], expected, rtol=1.0e-9)
        expected_t = (
            length
            * 8.0
            * VACUUM_IMPEDANCE
            * SPEED_OF_LIGHT
            * s0
            / (math.pi * radius**4)
            * _bane_sands_bracket(scaled, transverse=True)
        )
        np.testing.assert_allclose(values_t[index], expected_t, rtol=1.0e-9)
    root = math.sqrt(VACUUM_IMPEDANCE / (math.pi * conductivity))
    for index in (5, 6):
        np.testing.assert_allclose(
            values[index],
            -length
            * SPEED_OF_LIGHT
            / (4.0 * math.pi * radius)
            * root
            * zeta[index] ** -1.5,
            rtol=1.0e-12,
        )
        np.testing.assert_allclose(
            values_t[index],
            length
            * SPEED_OF_LIGHT
            / (math.pi * radius**3)
            * root
            / math.sqrt(zeta[index]),
            rtol=1.0e-12,
        )
    assert wall.longitudinal_crossover_mismatch < 5.0e-5
    assert wall.transverse_crossover_mismatch < 5.0e-5
    with pytest.raises(ValueError, match="quadrupolar"):
        wall.wake_function("quadrupolar-x", zeta)


def test_dipolar_uses_source_offsets_and_quadrupolar_uses_witness_offsets() -> None:
    zeta = np.asarray([0.0, 0.01, 0.02, 0.03])
    values = np.asarray([0.0, 1.0e12, 2.0e12, 3.0e12])
    dipolar = accelerator.WakeFunctionPlan(
        "dipolar-x", zeta, values, units="V/C/m", causality_convention="behind-positive"
    )
    quadrupolar = accelerator.WakeFunctionPlan(
        "quadrupolar-x",
        zeta,
        values,
        units="V/C/m",
        causality_convention="behind-positive",
    )
    momentum = 5.0e8 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT
    # Source slice at zeta = 0 offset x_s = 2 mm, witness slice 1 cm behind at
    # x_w = -0.5 mm; identical charges.
    coordinates = np.zeros((2, 6))
    coordinates[0, 0] = 2.0e-3
    coordinates[1, 0] = -0.5e-3
    coordinates[1, 4] = 0.01
    weights = np.asarray([1.0e9, 1.0e9])
    bunch = _bunch(coordinates, weights, momentum=momentum)
    charge = 1.0e9 * ELEMENTARY_CHARGE
    wake_at_lag = 1.0e12
    dipolar_result = accelerator.apply_wake(dipolar, bunch, SCALE, bin_count=2)
    quadrupolar_result = accelerator.apply_wake(quadrupolar, bunch, SCALE, bin_count=2)
    dipolar_kicks = np.asarray(dipolar_result.kicks)
    quadrupolar_kicks = np.asarray(quadrupolar_result.kicks)
    # W_perp(0+) = 0: the head gets no kick in either case.
    assert dipolar_kicks[0] == 0.0
    assert quadrupolar_kicks[0] == 0.0
    expected_dipolar = (
        ELEMENTARY_CHARGE * charge * 2.0e-3 * wake_at_lag / (SPEED_OF_LIGHT * momentum)
    )
    expected_quadrupolar = (
        ELEMENTARY_CHARGE * charge * (-0.5e-3) * wake_at_lag / (SPEED_OF_LIGHT * momentum)
    )
    np.testing.assert_allclose(dipolar_kicks[1], expected_dipolar, rtol=1.0e-12)
    np.testing.assert_allclose(quadrupolar_kicks[1], expected_quadrupolar, rtol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(dipolar_result.bunch.coordinates[:, 1]), dipolar_kicks, rtol=1.0e-12
    )
    assert float(dipolar_result.momentum_kick_sum) == pytest.approx(
        charge * charge * 2.0e-3 * wake_at_lag
    )
    assert float(dipolar_result.energy_change) == 0.0
    assert float(dipolar_result.bunch.coordinates[1, 3]) == 0.0


def _coupled_bunch_growth_reference(
    *,
    bunch_count: int,
    mode: int,
    tune: float,
    revolution_period: float,
    beta_function: float,
    bunch_charge: float,
    witness_charge: float,
    momentum: float,
    shunt: float,
    quality: float,
    omega_r: float,
) -> float:
    """Growth per turn ``Im Ω T_0`` from the transverse impedance harmonic sum.

    ``Im Ω = -(β_x q Q_b / (2 c p_0 T_0 T_b)) Σ_p Re Z_⊥(ω_β - μ ω_0 + p M ω_0)``
    for the rigid-dipole mode ``x_n ∝ exp(2πi μ n / M) exp(-iΩ t)``.
    """
    omega_0 = 2.0 * math.pi / revolution_period
    bunch_period = revolution_period / bunch_count
    harmonics = np.arange(-4000, 4001)
    sampled = (tune - mode + bunch_count * harmonics) * omega_0
    real_part = _reference_transverse_impedance(sampled, shunt, quality, omega_r).real
    coefficient = (
        beta_function
        * witness_charge
        * bunch_charge
        / (2.0 * SPEED_OF_LIGHT * momentum * revolution_period * bunch_period)
    )
    return -coefficient * float(np.sum(real_part)) * revolution_period


def test_coupled_bunch_growth_rate_matches_transverse_impedance_sum() -> None:
    bunch_count, mode, tune = 8, 3, 0.31
    revolution_period = 1.0e-6
    bunch_period = revolution_period / bunch_count
    beta_function = 10.0
    particles = 1.0e10
    bunch_charge = particles * ELEMENTARY_CHARGE
    momentum = 1.0e9 * ELEMENTARY_CHARGE / SPEED_OF_LIGHT
    omega_0 = 2.0 * math.pi / revolution_period
    frequency = (mode - tune) * omega_0 / (2.0 * math.pi)
    shunt, quality = 3.0e7, 3.0
    resonator = accelerator.ResonatorWake(
        shunt, quality, frequency, kind="dipolar-x", scale=SCALE
    )
    turns_of_memory = 5
    samples_per_spacing = 400
    zeta = (
        np.arange(turns_of_memory * bunch_count * samples_per_spacing + 1)
        * SPEED_OF_LIGHT
        * bunch_period
        / samples_per_spacing
    )
    plan = resonator.wake_function(zeta)
    memory = accelerator.WakeMemoryState.empty(turns_of_memory * bunch_count, 1)
    phase = 2.0 * math.pi * tune
    rotation = jnp.asarray(
        [
            [math.cos(phase), beta_function * math.sin(phase)],
            [-math.sin(phase) / beta_function, math.cos(phase)],
        ]
    )
    # Eigen-mode initial condition: x + i beta x' = exp(i theta_n) with the
    # betatron phase of each bunch at its own arrival time.
    theta = 2.0 * math.pi * (mode - tune) * np.arange(bunch_count) / bunch_count
    amplitude = 1.0e-6
    states = np.stack(
        (amplitude * np.cos(theta), amplitude * np.sin(theta) / beta_function), axis=1
    )
    template = _bunch(np.zeros((1, 6)), np.asarray([particles]), momentum=momentum)

    def turn(
        carry: tuple[jax.Array, accelerator.WakeMemoryState], turn_index: jax.Array
    ) -> tuple[tuple[jax.Array, accelerator.WakeMemoryState], jax.Array]:
        state, memory_state = carry
        rows = []
        for index in range(bunch_count):
            rotated = rotation @ state[index]
            coordinates = jnp.zeros((1, 6), dtype=jnp.float64)
            coordinates = coordinates.at[0, 0].set(rotated[0]).at[0, 1].set(rotated[1])
            bunch = eqx.tree_at(lambda value: value.coordinates, template, coordinates)
            arrival = turn_index * revolution_period + index * bunch_period
            kicked = accelerator.apply_wake(
                plan,
                bunch,
                SCALE,
                bin_count=1,
                arrival_time=arrival,
                memory=memory_state,
            )
            memory_state = accelerator.record_passage(
                memory_state, bunch, SCALE, arrival_time=arrival
            )
            rows.append(kicked.bunch.coordinates[0, (0, 1)])
        next_state = jnp.stack(rows)
        return (next_state, memory_state), next_state

    turn_count = 320
    (_, final_memory), history = jax.lax.scan(
        turn,
        (jnp.asarray(states), memory),
        jnp.arange(turn_count, dtype=jnp.float64),
    )
    assert int(final_memory.passage_count) == turn_count * bunch_count
    trajectory = np.asarray(history)
    complex_amplitude = trajectory[:, :, 0] + 1j * beta_function * trajectory[:, :, 1]
    projected = np.abs(np.sum(complex_amplitude * np.exp(-1j * theta)[None, :], axis=1))
    window = np.arange(80, turn_count)
    slope, _ = np.polyfit(window, np.log(projected[window]), 1)
    reference = _coupled_bunch_growth_reference(
        bunch_count=bunch_count,
        mode=mode,
        tune=tune,
        revolution_period=revolution_period,
        beta_function=beta_function,
        bunch_charge=bunch_charge,
        witness_charge=ELEMENTARY_CHARGE,
        momentum=momentum,
        shunt=shunt,
        quality=quality,
        omega_r=2.0 * math.pi * frequency,
    )
    assert reference > 0.0
    np.testing.assert_allclose(slope, reference, rtol=2.0e-2)
    # Bounded history that drops passages still inside the wake is reported.
    short_memory = accelerator.WakeMemoryState.empty(2, 1)
    bunch = _bunch(np.zeros((1, 6)), np.asarray([particles]), momentum=momentum)
    for index in range(3):
        short_memory = accelerator.record_passage(
            short_memory, bunch, SCALE, arrival_time=index * bunch_period
        )
    insufficient = accelerator.apply_wake(
        plan,
        bunch,
        SCALE,
        bin_count=1,
        arrival_time=3 * bunch_period,
        memory=short_memory,
    )
    assert not bool(insufficient.history_sufficient)


def test_wake_plan_refuses_unit_and_causality_mismatches() -> None:
    zeta = np.asarray([0.0, 0.01, 0.02])
    with pytest.raises(ValueError, match="units"):
        accelerator.WakeFunctionPlan(
            "longitudinal",
            zeta,
            np.asarray([1.0, 0.5, 0.0]),
            units="V/C/m",
            causality_convention="behind-positive",
        )
    with pytest.raises(ValueError, match="units"):
        accelerator.WakeFunctionPlan(
            "dipolar-y",
            zeta,
            np.asarray([0.0, 0.5, 1.0]),
            units="V/C",
            causality_convention="behind-positive",
        )
    with pytest.raises(ValueError, match="Panofsky"):
        accelerator.WakeFunctionPlan(
            "dipolar-y",
            zeta,
            np.asarray([1.0, 0.5, 1.0]),
            units="V/C/m",
            causality_convention="behind-positive",
        )
    with pytest.raises(ValueError, match="zero distance"):
        accelerator.WakeFunctionPlan(
            "longitudinal",
            np.asarray([-0.01, 0.0, 0.01]),
            np.asarray([0.0, 1.0, 0.5]),
            units="V/C",
            causality_convention="behind-positive",
        )
    with pytest.raises(ValueError, match="selector|units"):
        accelerator.WakeFunctionPlan(
            "longitudinal",
            zeta,
            np.asarray([1.0, 0.5, 0.0]),
            units="volts",  # ty: ignore[invalid-argument-type]
            causality_convention="behind-positive",
        )
    negative = accelerator.WakeFunctionPlan(
        "longitudinal",
        np.asarray([0.0, -0.01, -0.02]),
        np.asarray([1.0, 0.5, 0.25]),
        units="V/C",
        causality_convention="behind-negative",
    )
    np.testing.assert_array_equal(np.asarray(negative.zeta_samples), zeta)
    np.testing.assert_array_equal(
        np.asarray(negative.wake_values), np.asarray([1.0, 0.5, 0.25])
    )
    with pytest.raises(TypeError, match="ElectromagneticScaleContract"):
        accelerator.ResonatorWake(1.0, 2.0, 1.0e9, kind="longitudinal", scale=None)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="quality"):
        accelerator.ResonatorWake(1.0, 0.5, 1.0e9, kind="longitudinal", scale=SCALE)
    with pytest.raises(ValueError, match="loss factor"):
        _ = accelerator.ResonatorWake(
            1.0, 2.0, 1.0e9, kind="dipolar-x", scale=SCALE
        ).loss_factor
