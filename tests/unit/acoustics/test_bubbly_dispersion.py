#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax.acoustics as acoustics
import phydrax.bubble_dynamics as bd


AMBIENT = 101325.0
DENSITY = 1000.0
SOUND_SPEED = 1500.0
KAPPA = 1.4


def _model(
    tension: float, viscosity: float, equation: bd.RadialBubbleEquation
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        bd.PolytropicBubbleGasLaw(KAPPA),
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.CleanBubbleInterfaceLaw(tension),
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=SOUND_SPEED,
    )


def _number_density(void_fraction: float, radius: float) -> float:
    return void_fraction / (4.0 * np.pi * radius**3 / 3.0)


def test_wood_sound_speed_of_one_percent_air_in_water() -> None:
    # Water ρ_l = 1000 kg/m³, c_l = 1500 m/s; air ρ_g = 1.2 kg/m³, c_g = 343 m/s.
    speed = float(acoustics.wood_sound_speed(0.01, 1000.0, 1500.0, 1.2, 343.0))
    assert speed == pytest.approx(119.0, abs=0.5)
    assert float(
        acoustics.wood_sound_speed(0.0, 1000.0, 1500.0, 1.2, 343.0)
    ) == pytest.approx(1500.0)
    assert float(
        acoustics.wood_sound_speed(1.0, 1000.0, 1500.0, 1.2, 343.0)
    ) == pytest.approx(343.0)


@pytest.mark.parametrize("tension", (0.0, 0.072))
def test_low_frequency_limit_matches_dilute_analytic_and_wood(tension: float) -> None:
    radius, void_fraction = 1.0e-4, 1.0e-2
    minnaert = float(
        bd.minnaert_angular_frequency(
            radius, AMBIENT, DENSITY, KAPPA, surface_tension=tension
        )
    )
    plan = acoustics.BubblyMediumDispersionPlan(
        _model(tension, 0.0, "rayleigh_plesset"),
        radius,
        _number_density(void_fraction, radius),
        np.array([1.0e-3 * minnaert]),
    )
    result = acoustics.solve_bubbly_medium_dispersion(plan)
    assert bool(result.successful)
    assert bool(result.evidence.dilute)
    assert float(result.void_fraction) == pytest.approx(void_fraction, rel=1.0e-12)
    speed = float(result.phase_speed[0])
    # Dilute limit 1/c² = 1/c_l² + 3βρ_l/(3κp_g0 − 2σ/R), p_g0 = p0 + 2σ/R; the
    # neglected (ω/ω0)² = 1e-6 correction bounds the agreement.
    gas_pressure = AMBIENT + 2.0 * tension / radius
    stiffness = 3.0 * KAPPA * gas_pressure - 2.0 * tension / radius
    dilute = (1.0 / SOUND_SPEED**2 + 3.0 * void_fraction * DENSITY / stiffness) ** -0.5
    assert speed == pytest.approx(dilute, rel=1.0e-5)
    assert float(result.attenuation_neper_per_meter[0]) == pytest.approx(0.0, abs=1.0e-12)
    if tension == 0.0:
        # Wood's law with ρ_g c_g² = κ p0 differs by the O(β) mixture-density
        # factor ρ_m/ρ_l ≈ 1 − β that the dilute model drops: c_CP/c_Wood ≈ √(1 − β).
        gas_density = 1.2
        gas_speed = np.sqrt(KAPPA * AMBIENT / gas_density)
        wood = float(
            acoustics.wood_sound_speed(
                void_fraction, DENSITY, SOUND_SPEED, gas_density, gas_speed
            )
        )
        assert abs(speed / wood - 1.0) <= void_fraction


def test_discrete_bins_reproduce_commander_prosperetti_oscillator_sum() -> None:
    radii = np.array([5.0e-5, 1.5e-4])
    numbers = np.array(
        [_number_density(2.0e-4, radii[0]), _number_density(3.0e-4, radii[1])]
    )
    frequencies = 2.0 * np.pi * np.array([5.0e3, 2.0e4, 6.0e4, 2.0e5])
    model = _model(0.072, 1.0e-3, "rayleigh_plesset")
    result = acoustics.solve_bubbly_medium_dispersion(
        acoustics.BubblyMediumDispersionPlan(model, radii, numbers, frequencies)
    )
    assert bool(result.successful)
    # Commander & Prosperetti (1989) eq. (41) summed over bins with the
    # Rayleigh–Plesset oscillator ω0² = K/(ρR²), b = 2μ/(ρR²).
    total = np.zeros(frequencies.shape, dtype=np.complex128)
    for radius, number in zip(radii, numbers, strict=True):
        omega0 = float(
            bd.minnaert_angular_frequency(
                radius, AMBIENT, DENSITY, KAPPA, surface_tension=0.072
            )
        )
        damping = 2.0 * 1.0e-3 / (DENSITY * radius**2)
        total += (
            number * radius / (omega0**2 - frequencies**2 + 2j * damping * frequencies)
        )
    squared = frequencies**2 / SOUND_SPEED**2 + 4.0 * np.pi * frequencies**2 * total
    wavenumber = np.sqrt(squared)
    np.testing.assert_allclose(np.asarray(result.wavenumber), wavenumber, rtol=1.0e-9)
    assert np.all(np.asarray(result.attenuation_neper_per_meter) > 0.0)
    np.testing.assert_allclose(
        np.asarray(result.attenuation_decibel_per_meter),
        20.0 * np.log10(np.e) * np.asarray(result.attenuation_neper_per_meter),
        rtol=1.0e-14,
    )


def test_attenuation_peaks_at_resonance_and_fast_waves_above_it() -> None:
    radius, void_fraction = 1.0e-4, 1.0e-6
    minnaert = float(
        bd.minnaert_angular_frequency(
            radius, AMBIENT, DENSITY, KAPPA, surface_tension=0.072
        )
    )
    frequencies = minnaert * np.linspace(0.8, 1.2, 401)
    plan = acoustics.BubblyMediumDispersionPlan(
        _model(0.072, 1.0e-3, "keller_miksis"),
        radius,
        _number_density(void_fraction, radius),
        frequencies,
    )
    result = acoustics.solve_bubbly_medium_dispersion(plan)
    assert bool(result.successful)
    attenuation = np.asarray(result.attenuation_neper_per_meter)
    peak = frequencies[np.argmax(attenuation)]
    assert peak / float(result.resonance_frequency[0]) == pytest.approx(1.0, abs=1.0e-2)
    assert attenuation.max() > 10.0 * max(attenuation[0], attenuation[-1])
    speed = np.asarray(result.phase_speed)
    assert speed[0] < SOUND_SPEED < speed[-1]
    assert float(result.evidence.maximum_size_parameter) < 0.1


def test_dense_mixture_is_flagged_outside_dilute_support() -> None:
    radius = 1.0e-4
    plan = acoustics.BubblyMediumDispersionPlan(
        _model(0.0, 0.0, "rayleigh_plesset"),
        radius,
        _number_density(5.0e-2, radius),
        np.array([1.0e2]),
    )
    result = acoustics.solve_bubbly_medium_dispersion(plan)
    assert not bool(result.evidence.dilute)
    assert bool(result.evidence.finite)
    assert bool(np.all(np.asarray(result.evidence.response_successful)))
    assert int(result.status) == int(
        acoustics.BubblyMediumDispersionStatus.OUTSIDE_DILUTE_LIMIT
    )
    assert not bool(result.successful)


def test_invalid_inputs_are_refused() -> None:
    model = _model(0.0, 0.0, "rayleigh_plesset")
    frequencies = np.array([1.0e3])
    radius = np.array([1.0e-4])
    number = np.array([1.0e6])
    with pytest.raises(ValueError, match="number_densities"):
        acoustics.BubblyMediumDispersionPlan(
            model, np.array([1.0e-4, 2.0e-4]), number, frequencies
        )
    with pytest.raises(ValueError, match="number_densities"):
        acoustics.BubblyMediumDispersionPlan(model, radius, -number, frequencies)
    with pytest.raises(ValueError, match="bin_radii"):
        acoustics.BubblyMediumDispersionPlan(model, -radius, number, frequencies)
    with pytest.raises(ValueError, match="angular_frequencies"):
        acoustics.BubblyMediumDispersionPlan(model, radius, number, np.array([0.0]))
    with pytest.raises(ValueError, match="maximum_void_fraction"):
        acoustics.BubblyMediumDispersionPlan(
            model, radius, number, frequencies, maximum_void_fraction=1.5
        )
    with pytest.raises(TypeError, match="RadialBubbleModel"):
        acoustics.BubblyMediumDispersionPlan(None, radius, number, frequencies)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="gas_density"):
        acoustics.wood_sound_speed(0.01, 1000.0, 1500.0, -1.2, 343.0)
    with pytest.raises(ValueError, match="liquid_density"):
        acoustics.wood_sound_speed(0.01, -1000.0, 1500.0, 1.2, 343.0)
    with pytest.raises(ValueError, match="void_fraction"):
        acoustics.wood_sound_speed(1.5, 1000.0, 1500.0, 1.2, 343.0)
