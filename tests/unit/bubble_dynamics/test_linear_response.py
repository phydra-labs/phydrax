#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import numpy as np
import pytest

import phydrax.bubble_dynamics as bd


AMBIENT = 101325.0
DENSITY = 1000.0
SOUND_SPEED = 1500.0


def _model(
    gas: bd.AbstractBubbleGasLaw,
    liquid: bd.AbstractBubbleLiquidLaw,
    interface: bd.AbstractBubbleInterfaceLaw,
    equation: bd.RadialBubbleEquation = "rayleigh_plesset",
) -> bd.RadialBubbleModel:
    return bd.RadialBubbleModel(
        equation,
        gas,
        liquid,
        interface,
        bd.BubbleEnvironment(AMBIENT, 293.15),
        liquid_density=DENSITY,
        liquid_sound_speed=SOUND_SPEED,
    )


@pytest.mark.parametrize("tension", (0.0, 0.072))
def test_minnaert_frequency_with_and_without_capillarity(tension: float) -> None:
    radius = 1.0e-3
    model = _model(
        bd.PolytropicBubbleGasLaw(1.4),
        bd.NewtonianBubbleLiquidLaw(0.0),
        bd.CleanBubbleInterfaceLaw(tension),
    )
    response = bd.linear_bubble_response(model, radius, np.array([1.0e3, 3.0e4]))
    exact = float(bd.minnaert_angular_frequency(radius, AMBIENT, DENSITY, 1.4, surface_tension=tension))
    assert bool(response.successful)
    np.testing.assert_allclose(np.asarray(response.natural_frequency), exact, rtol=1.0e-12)
    assert float(response.resonance_frequency) == pytest.approx(exact, rel=1.0e-12)
    assert exact / (2.0 * np.pi) == pytest.approx(3.26e3, rel=0.01)
    np.testing.assert_allclose(np.asarray(response.total_damping), 0.0, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(response.effective_polytropic_index), 1.4, rtol=1.0e-12)


def test_damping_split_matches_viscous_shell_and_radiation_forms() -> None:
    radius, viscosity, shell_viscosity, elasticity, initial_tension = 2.0e-6, 1.0e-3, 1.5e-8, 0.55, 0.02
    model = _model(
        bd.PolytropicBubbleGasLaw(1.07),
        bd.NewtonianBubbleLiquidLaw(viscosity),
        bd.MarmottantShell(elasticity, initial_tension, 0.072, shell_viscosity),
        "keller_miksis",
    )
    frequency = 2.0 * np.pi * 3.0e6
    response = bd.linear_bubble_response(model, radius, np.array([frequency]))
    assert bool(response.successful)
    assert float(response.viscous_damping[0]) == pytest.approx(2.0 * viscosity / (DENSITY * radius**2), rel=1.0e-10)
    assert float(response.shell_damping[0]) == pytest.approx(
        2.0 * shell_viscosity / (DENSITY * radius**3), rel=1.0e-10
    )
    mach = frequency * radius / SOUND_SPEED
    assert float(response.radiation_damping[0]) == pytest.approx(
        frequency * mach / (2.0 * (1.0 + mach**2)), rel=1.0e-9
    )
    # Coated resonance of the elastic Marmottant branch (Marmottant et al. 2005).
    buckling = radius / np.sqrt(1.0 + initial_tension / elasticity)
    gas_pressure = AMBIENT + 2.0 * initial_tension / radius
    stiffness = (
        3.0 * 1.07 * gas_pressure
        - 2.0 * initial_tension / radius
        + 4.0 * elasticity * radius / buckling**2
    )
    coated = np.sqrt(stiffness / (DENSITY * radius**2))
    assert float(response.natural_frequency[0]) == pytest.approx(coated, rel=1.0e-12)
    clean = float(bd.minnaert_angular_frequency(radius, AMBIENT, DENSITY, 1.07, surface_tension=0.072))
    assert coated > clean


@pytest.mark.parametrize("peclet", (0.5, 5.0, 50.0))
def test_spectral_thermal_gas_reproduces_prosperetti_linear_theory(peclet: float) -> None:
    radius, gamma, conductivity = 1.0e-5, 1.4, 0.0262
    model = _model(
        bd.SpectralThermalBubbleGasLaw(gamma, conductivity, node_count=16),
        bd.NewtonianBubbleLiquidLaw(0.0),
        bd.CleanBubbleInterfaceLaw(0.0),
    )
    amount = AMBIENT * bd.sphere_volume(radius) / (bd.MOLAR_GAS_CONSTANT * 293.15)
    diffusivity = conductivity * (gamma - 1.0) * bd.sphere_volume(radius) / (gamma * bd.MOLAR_GAS_CONSTANT * amount)
    frequency = peclet * diffusivity / radius**2
    response = bd.linear_bubble_response(model, radius, np.array([frequency]))
    exact = complex(bd.prosperetti_polytropic_index(gamma, peclet))
    assert bool(response.successful)
    assert float(response.effective_polytropic_index[0]) == pytest.approx(exact.real, rel=1.0e-9)
    damping = 3.0 * AMBIENT * exact.imag / (2.0 * frequency * DENSITY * radius**2)
    assert float(response.thermal_damping[0]) == pytest.approx(damping, rel=1.0e-7)


def test_reduced_transfer_gas_tends_to_isothermal_transfer_five_and_prosperetti_at_low_peclet() -> None:
    assert float(bd.preston_transfer_coefficient(1.0e-6)) == pytest.approx(5.0, rel=1.0e-12)
    # Re Ψ = 5 + Pe²/441 + O(Pe⁴): the series and closed-form branches join smoothly.
    jump = float(bd.preston_transfer_coefficient(0.1001)) - float(
        bd.preston_transfer_coefficient(0.0999)
    )
    assert jump == pytest.approx(2.0e-4 * 2.0 * 0.1 / 441.0, rel=1.0e-3)
    radius, gamma, conductivity = 1.0e-5, 1.4, 0.0262
    amount = AMBIENT * bd.sphere_volume(radius) / (bd.MOLAR_GAS_CONSTANT * 293.15)
    diffusivity = conductivity * (gamma - 1.0) * bd.sphere_volume(radius) / (gamma * bd.MOLAR_GAS_CONSTANT * amount)
    peclet = 0.2
    frequency = peclet * diffusivity / radius**2
    model = _model(
        bd.ReducedTransferBubbleGasLaw(gamma, conductivity, frequency),
        bd.NewtonianBubbleLiquidLaw(0.0),
        bd.CleanBubbleInterfaceLaw(0.0),
    )
    response = bd.linear_bubble_response(model, radius, np.array([frequency]))
    exact = complex(bd.prosperetti_polytropic_index(gamma, peclet))
    assert float(response.effective_polytropic_index[0]) == pytest.approx(exact.real, rel=1.0e-4)
    damping = 3.0 * AMBIENT * exact.imag / (2.0 * frequency * DENSITY * radius**2)
    assert float(response.thermal_damping[0]) == pytest.approx(damping, rel=1.0e-2)


def test_viscoelastic_liquids_have_their_analytic_linear_moduli() -> None:
    radius, viscosity, modulus, relaxation = 5.0e-6, 2.0e-3, 5.0e4, 1.0e-8
    frequency = 2.0 * np.pi * 1.0e6
    gas = bd.PolytropicBubbleGasLaw(1.4)
    clean = bd.CleanBubbleInterfaceLaw(0.0)

    def liquid_stiffness_and_damping(liquid: bd.AbstractBubbleLiquidLaw) -> tuple[float, float]:
        response = bd.linear_bubble_response(_model(gas, liquid, clean), radius, np.array([frequency]))
        assert bool(response.successful)
        return float(response.liquid_stiffness[0]), float(response.viscous_damping[0])

    kelvin = liquid_stiffness_and_damping(bd.KelvinVoigtBubbleLiquidLaw(viscosity, modulus))
    assert kelvin[0] == pytest.approx(4.0 * modulus / radius, rel=1.0e-10)
    assert kelvin[1] == pytest.approx(2.0 * viscosity / (DENSITY * radius**2), rel=1.0e-10)

    arm = viscosity - modulus * relaxation
    zener = liquid_stiffness_and_damping(bd.ZenerBubbleLiquidLaw(viscosity, modulus, relaxation))
    phase = frequency * relaxation
    arm_stiffness = 4.0 * arm * frequency * phase / (radius * (1.0 + phase**2))
    arm_damping = 4.0 * arm / (radius * (1.0 + phase**2)) / (2.0 * DENSITY * radius)
    assert zener[0] == pytest.approx(4.0 * modulus / radius + arm_stiffness, rel=1.0e-9)
    assert zener[1] == pytest.approx(arm_damping, rel=1.0e-9)

    solvent, polymer = 1.0e-3, 3.0e-3
    oldroyd = liquid_stiffness_and_damping(
        bd.OldroydBBubbleLiquidLaw(solvent, polymer, relaxation, quadrature_nodes=8)
    )
    polymer_damping = 4.0 * polymer / (radius * (1.0 + phase**2)) / (2.0 * DENSITY * radius)
    assert oldroyd[1] == pytest.approx(2.0 * solvent / (DENSITY * radius**2) + polymer_damping, rel=1.0e-9)
    assert oldroyd[0] == pytest.approx(4.0 * polymer * frequency * phase / (radius * (1.0 + phase**2)), rel=1.0e-9)

    power = liquid_stiffness_and_damping(bd.PowerLawBubbleLiquidLaw(viscosity, 0.6, 1.0e9))
    newtonian = liquid_stiffness_and_damping(bd.NewtonianBubbleLiquidLaw(viscosity * 1.0e9 ** (0.6 - 1.0)))
    assert power[1] == pytest.approx(newtonian[1], rel=1.0e-12)


def test_shell_catalogue_linear_limits() -> None:
    radius, thickness, modulus, shell_viscosity = 2.0e-6, 4.0e-9, 5.0e7, 1.0
    frequency = 2.0 * np.pi * 2.0e6
    gas = bd.PolytropicBubbleGasLaw(1.07)
    liquid = bd.NewtonianBubbleLiquidLaw(1.0e-3)

    def shell_response(interface: bd.AbstractBubbleInterfaceLaw) -> bd.LinearBubbleResponse:
        response = bd.linear_bubble_response(_model(gas, liquid, interface), radius, np.array([frequency]))
        assert bool(response.successful)
        return response

    hoff = shell_response(bd.HoffShell(modulus, shell_viscosity, thickness))
    assert float(hoff.interface_stiffness[0]) == pytest.approx(12.0 * modulus * thickness / radius**2, rel=1.0e-10)
    assert float(hoff.shell_damping[0]) == pytest.approx(
        12.0 * shell_viscosity * thickness / (2.0 * DENSITY * radius**3), rel=1.0e-10
    )
    church = shell_response(bd.ChurchShell(modulus, shell_viscosity, 1.0e-3 * thickness))
    thin = shell_response(bd.HoffShell(modulus, shell_viscosity, 1.0e-3 * thickness))
    assert float(church.interface_stiffness[0]) == pytest.approx(float(thin.interface_stiffness[0]), rel=1.0e-5)
    assert float(church.shell_damping[0]) == pytest.approx(float(thin.shell_damping[0]), rel=1.0e-5)

    maxwell = shell_response(bd.MaxwellShell(shell_viscosity, thickness, 1.0e-15))
    viscous_hoff = shell_response(bd.HoffShell(0.0, shell_viscosity, thickness))
    assert float(maxwell.shell_damping[0]) == pytest.approx(float(viscous_hoff.shell_damping[0]), rel=1.0e-6)

    elasticity = 0.5
    doinikov = shell_response(bd.DoinikovShearThinningShell(elasticity, 0.0, 1.0e-8, 1.0e-6))
    kelvin_shell = shell_response(bd.DoinikovShearThinningShell(elasticity, 0.0, 1.0e-8, 0.0))
    assert float(doinikov.interface_stiffness[0]) == pytest.approx(4.0 * elasticity / radius**2, rel=1.0e-10)
    assert float(doinikov.shell_damping[0]) == pytest.approx(float(kelvin_shell.shell_damping[0]), rel=1.0e-12)

    sarkar = shell_response(bd.SarkarShell(0.04, elasticity, 1.0e-8, elasticity_decay=1.5))
    tension_stiffness = 4.0 * elasticity / radius**2 - 2.0 * 0.04 / radius**2
    assert float(sarkar.interface_stiffness[0]) == pytest.approx(tension_stiffness, rel=1.0e-10)
