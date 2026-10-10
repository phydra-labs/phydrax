"""High-burnup structure formation and HBS pores of UO2.

Independent references: the effective-burnup threshold (1273.15 K) and KJMA
form of Barani et al. (2020); the source cluster generator, Xe hard-sphere
volume (Eq. 17) and pore-volume kinetics of Barani et al. (2022), Part II.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nuclear_fuel import (
    AthermalRelease,
    FissionGasModel,
    FissionGasStepResult,
    HighBurnupStructurePores,
)
from tests._support.nuclear_fuel_gas import (
    advance_gas,
    DAY,
    fission_gas_model,
    FISSION_RATE,
    NOMINAL_HYDROSTATIC_STRESS,
    PRODUCTION,
    XE_KR_YIELD,
)


pytestmark = pytest.mark.strict_jax

# Barani et al. (2020) KJMA restructuring on effective burnup [MWd/kgHM].
KJMA_RATE_CONSTANT = 2.77e-7
KJMA_AVRAMI_EXPONENT = 3.54
# Barani et al. (2022) Part II source parameters; nucleation from the Barani
# thesis (2020) Table 3.4: 5e17 nuclei per unit restructured fraction.
NUCLEI_PER_RESTRUCTURED_FRACTION = 5.0e17  # [1/m^3]
ATOMS_PER_NUCLEUS = 2.0  # source dimer
SCHOTTKY_VACANCY_VOLUME = 4.09e-29  # [m^3]
SURFACE_ENERGY = 1.0  # [J/m^2]
RESOLUTION_COEFFICIENT = 2.0e-23  # [m^3]
RESOLUTION_DISTANCE = 1.0e-9  # [m]
RESOLUTION_LAYER = 1.0e-9  # [m]
GAS_DIFFUSIVITY_PREFACTOR = 1.3e-7  # [m^2/s]
GAS_ACTIVATION_ENERGY = 4.52e-19  # [J]
VACANCY_DIFFUSIVITY_PREFACTOR = 8.86e-6  # [m^2/s]
VACANCY_ACTIVATION_ENERGY = 5.75e-19  # [J]
VACANCY_IRRADIATION_COEFFICIENT = 1.0e-39  # [m^5]
FABRICATED_TILT_DEGREES = 4.0
RESTRUCTURED_TILT_INCREASE_DEGREES = 36.0
BOLTZMANN = 1.380649e-23  # [J/K]


def _kjma_fraction(effective_burnup: float) -> float:
    return 1.0 - np.exp(-KJMA_RATE_CONSTANT * effective_burnup**KJMA_AVRAMI_EXPONENT)


def _xe_hard_sphere_volume(temperature: float) -> float:
    """Barani et al. (2022), Eq. 17 [m^3/atom]."""
    diameter = 4.45e-10 * (0.8542 - 0.03996 * np.log(temperature / 231.2))
    return np.pi * diameter**3 / 6.0


def _sphere_radius(volume: np.ndarray | float) -> np.ndarray:
    return np.cbrt(3.0 * np.asarray(volume) / (4.0 * np.pi))


def _boundary_tilt(fraction: float) -> float:
    """Grain-boundary tilt enhancement of HBS diffusivities."""
    restructured = FABRICATED_TILT_DEGREES + RESTRUCTURED_TILT_INCREASE_DEGREES * fraction
    return np.sin(np.deg2rad(restructured)) / np.sin(np.deg2rad(FABRICATED_TILT_DEGREES))


def test_hbs_forms_only_on_effective_burnup_and_monotonically() -> None:
    model = fission_gas_model(mode_count=40)
    # Site 0 stays above the 1273.15 K annealing threshold, site 1 below it.
    temperature = np.array([1400.0, 900.0])
    state = model.initial_state((2,), grain_radius=5.0e-6)
    fractions: list[np.ndarray] = []
    depletions: list[np.ndarray] = []
    burnup = 0.0
    for _ in range(40):
        burnup += 3.0
        result = advance_gas(
            model,
            state,
            temperature=temperature,
            previous_temperature=temperature,
            burnup=burnup,
            time_step=30.0 * DAY,
        )
        fractions.append(np.asarray(result.state.hbs_fraction))
        depletions.append(np.asarray(result.transfers.hbs_depletion))
        state = result.state
    history = np.stack(fractions)
    assert np.all(history[:, 0] == 0.0)
    assert np.all(np.stack(depletions)[:, 0] == 0.0)
    assert np.all(np.diff(history[:, 1]) > 0.0)
    np.testing.assert_allclose(history[-1, 1], _kjma_fraction(120.0), rtol=1.0e-12)
    assert np.all(np.stack(depletions)[1:, 1] > 0.0)
    assert float(state.hbs_pore_gas[1]) > 0.0


# Cold startup of four radial sites: 300 K, a 0.2 s step at the 100 W/m
# startup fission rate. Their effective burnups give KJMA fractions near 1e-40:
# tiny but positive, so the HBS root is active far below every other inventory.
COLD_TEMPERATURE = 300.0  # [K]
COLD_TIME_STEP = 0.2  # [s]
STARTUP_FISSION_RATE = 6.1e16  # [fissions/m^3/s]
STARTUP_HBS_FRACTION = np.asarray([2.80e-40, 2.82e-40, 2.84e-40, 2.86e-40])
STARTUP_EFFECTIVE_BURNUP = (STARTUP_HBS_FRACTION / KJMA_RATE_CONSTANT) ** (
    1.0 / KJMA_AVRAMI_EXPONENT
)  # [MWd/kgHM]


def _cold_startup_birth(model: FissionGasModel) -> FissionGasStepResult:
    sites = STARTUP_HBS_FRACTION.shape
    cold = np.full(sites, COLD_TEMPERATURE)
    return eqx.filter_jit(advance_gas)(
        model,
        model.initial_state(sites, grain_radius=5.0e-6),
        temperature=cold,
        previous_temperature=cold,
        burnup=STARTUP_EFFECTIVE_BURNUP,
        time_step=COLD_TIME_STEP,
        fission_rate=STARTUP_FISSION_RATE,
        production=XE_KR_YIELD * STARTUP_FISSION_RATE,
    )


def test_cold_startup_hbs_birth_converges_and_conserves_each_site() -> None:
    result = _cold_startup_birth(fission_gas_model(mode_count=2))
    state, transfers = result.state, result.transfers
    assert bool(jnp.all(result.admissible))
    assert bool(jnp.all(result.diagnostics.hbs_root_converged))
    np.testing.assert_allclose(state.hbs_fraction, STARTUP_HBS_FRACTION, rtol=1.0e-12)
    arrival = np.asarray(transfers.hbs_production + transfers.hbs_depletion)
    np.testing.assert_allclose(
        state.hbs_pore_gas + state.hbs_boundary_gas, arrival, rtol=1.0e-13
    )
    assert bool(jnp.all(state.hbs_pore_number_density > 0.0))
    assert bool(jnp.all(state.hbs_boundary_gas > 0.0))
    # Source number balance of the birth step: every dimer born from the
    # arrived gas survives unless re-solved with the size-dependent fraction
    # dt b F (3 delta / (3 delta + R)) (lambda / (lambda + R)). At this volume
    # fraction impingement is below an ulp.
    volume = (
        _xe_hard_sphere_volume(COLD_TEMPERATURE) * np.asarray(state.hbs_pore_gas)
        + SCHOTTKY_VACANCY_VOLUME * np.asarray(state.hbs_pore_vacancies)
    ) / np.asarray(state.hbs_pore_number_density)
    radius = _sphere_radius(volume)
    loss = (
        COLD_TIME_STEP
        * RESOLUTION_COEFFICIENT
        * STARTUP_FISSION_RATE
        * (3.0 * RESOLUTION_DISTANCE / (3.0 * RESOLUTION_DISTANCE + radius))
        * (RESOLUTION_LAYER / (RESOLUTION_LAYER + radius))
    )
    np.testing.assert_allclose(
        state.hbs_pore_number_density,
        arrival / (ATOMS_PER_NUCLEUS * (1.0 + loss)),
        rtol=1.0e-12,
    )


def test_unpowered_hold_preserves_newborn_hbs_pores() -> None:
    model = fission_gas_model(mode_count=2)
    born = _cold_startup_birth(model).state
    cold = np.full(STARTUP_HBS_FRACTION.shape, COLD_TEMPERATURE)
    held = eqx.filter_jit(advance_gas)(
        model,
        born,
        temperature=cold,
        previous_temperature=cold,
        burnup=STARTUP_EFFECTIVE_BURNUP,
        time_step=0.5 * COLD_TIME_STEP,
        fission_rate=0.0,
        production=0.0,
    )
    assert bool(jnp.all(held.admissible))
    assert bool(jnp.all(held.diagnostics.hbs_root_converged))
    for name in (
        "hbs_fraction",
        "hbs_pore_gas",
        "hbs_boundary_gas",
        "hbs_pore_number_density",
        "hbs_pore_variance",
        "hbs_pore_vacancies",
        "hbs_porosity",
    ):
        np.testing.assert_array_equal(
            getattr(held.state, name), getattr(born, name), err_msg=name
        )
    for name in ("hbs_nucleation", "hbs_trapping", "hbs_resolution"):
        np.testing.assert_array_equal(
            getattr(held.transfers, name), np.zeros(4), err_msg=name
        )


# One 30-day step at 1100 K to 80 MWd/kgHM with gas capture and re-solution
# disabled: the pores hold exactly their nucleated dimers and grow by
# vacancy absorption alone.
NUCLEATION_TEMPERATURE = 1100.0  # [K]
NUCLEATION_TIME_STEP = 30.0 * DAY  # [s]


@pytest.fixture(scope="module")
def nucleation_step() -> FissionGasStepResult:
    model = FissionGasModel(
        athermal=AthermalRelease(
            geometric_surface_to_volume=0.0, total_surface_to_volume=0.0
        ),
        hbs_pores=HighBurnupStructurePores(
            gas_diffusivity_prefactor=0.0, resolution_coefficient=0.0
        ),
    )
    return advance_gas(
        model,
        model.initial_state((), grain_radius=5.0e-6),
        temperature=NUCLEATION_TEMPERATURE,
        previous_temperature=NUCLEATION_TEMPERATURE,
        burnup=80.0,
        time_step=NUCLEATION_TIME_STEP,
    )


def test_hbs_nucleation_conserves_gas_and_builds_gas_and_vacancy_pore_volume(
    nucleation_step: FissionGasStepResult,
) -> None:
    result = nucleation_step
    assert bool(result.admissible)
    assert bool(result.diagnostics.hbs_root_converged)
    nuclei = NUCLEI_PER_RESTRUCTURED_FRACTION * float(result.state.hbs_fraction)
    atoms = ATOMS_PER_NUCLEUS * nuclei
    np.testing.assert_allclose(float(result.state.hbs_pore_gas), atoms, rtol=1.0e-9)
    np.testing.assert_allclose(
        float(result.state.hbs_pore_number_density), nuclei, rtol=1.0e-9
    )
    # Part II Eqs. (13), (17): gas hard-sphere and Schottky vacancy volumes add.
    vacancies = float(result.state.hbs_pore_vacancies)
    assert vacancies > 0.0
    np.testing.assert_allclose(
        float(result.hbs_swelling),
        _xe_hard_sphere_volume(NUCLEATION_TEMPERATURE) * atoms
        + SCHOTTKY_VACANCY_VOLUME * vacancies,
        rtol=1.0e-9,
    )
    np.testing.assert_allclose(
        float(result.retained_total + result.released_increment),
        PRODUCTION * NUCLEATION_TIME_STEP,
        rtol=1.0e-12,
    )


def test_hbs_vacancy_absorption_satisfies_backward_euler_pressure_balance(
    nucleation_step: FissionGasStepResult,
) -> None:
    """Part II vacancy absorption ``dn_v/dt = M (p - 2 gamma / R + sigma)``."""
    result = nucleation_step
    fraction = float(result.state.hbs_fraction)
    nuclei = NUCLEI_PER_RESTRUCTURED_FRACTION * fraction
    gas_volume = ATOMS_PER_NUCLEUS * _xe_hard_sphere_volume(NUCLEATION_TEMPERATURE)
    per_pore_vacancies = float(result.state.hbs_pore_vacancies) / nuclei
    volume = gas_volume + SCHOTTKY_VACANCY_VOLUME * per_pore_vacancies
    radius = float(_sphere_radius(volume))
    # Carnahan-Starling gas of the dimer at packing eta = gas volume / pore
    # volume; 1 - eta is the vacancy volume fraction, free of cancellation.
    energy = BOLTZMANN * NUCLEATION_TEMPERATURE
    packing = gas_volume / volume
    pressure = (
        energy
        * ATOMS_PER_NUCLEUS
        / volume
        * (1.0 + packing + packing**2 - packing**3)
        / (SCHOTTKY_VACANCY_VOLUME * per_pore_vacancies / volume) ** 3
    )
    # Vacancy mobility of a pore of radius R in its Wigner-Seitz cell.
    cell = float(_sphere_radius(1.0 / nuclei))
    psi = radius / cell
    zeta = 10.0 * psi * (1.0 + psi**3) / (-(psi**6) + 5.0 * psi**2 - 9.0 * psi + 5.0)
    diffusivity = _boundary_tilt(fraction) * (
        VACANCY_DIFFUSIVITY_PREFACTOR * np.exp(-VACANCY_ACTIVATION_ENERGY / energy)
        + VACANCY_IRRADIATION_COEFFICIENT * FISSION_RATE
    )
    mobility = 2.0 * np.pi * diffusivity * cell / (energy * zeta)
    equilibrium = 2.0 * SURFACE_ENERGY / radius - NOMINAL_HYDROSTATIC_STRESS
    # At this diffusion/step scale the required pressure excess is below a
    # pressure ulp; compare the conditionable pressure form of the BE equation.
    np.testing.assert_allclose(
        pressure,
        equilibrium + per_pore_vacancies / (NUCLEATION_TIME_STEP * mobility),
        rtol=1.0e-9,
    )


# A broad HBS pore population at 80 MWd/kgHM and 1500 K: 1e17 pores/m^3 with
# a mean of 1000 atoms, one vacancy per atom, a squared coefficient of
# variation of 50 and 1e25 atoms/m^3 of HBS boundary gas.
BROAD_PORE_DENSITY = 1.0e17  # [1/m^3]
BROAD_MEAN_ATOMS = 1000.0  # atoms per pore
BROAD_SQUARED_VARIATION = 50.0
BROAD_BOUNDARY_GAS = 1.0e25  # [atoms/m^3]
BROAD_BURNUP = 80.0  # [MWd/kgHM]
BROAD_TEMPERATURE = 1500.0  # [K]
BROAD_TIME_STEP = 1.0  # [s]


def test_broad_pore_distribution_captures_at_least_the_dimer_rate_and_conserves() -> None:
    model = fission_gas_model(mode_count=8)
    atoms = BROAD_MEAN_ATOMS * BROAD_PORE_DENSITY
    vacancies = BROAD_MEAN_ATOMS * BROAD_PORE_DENSITY
    fraction = _kjma_fraction(BROAD_BURNUP)
    omega = _xe_hard_sphere_volume(BROAD_TEMPERATURE)
    state = eqx.tree_at(
        lambda site: (
            site.hbs_fraction,
            site.hbs_pore_number_density,
            site.hbs_pore_gas,
            site.hbs_pore_variance,
            site.hbs_pore_vacancies,
            site.hbs_boundary_gas,
            site.produced,
            site.burnup_mwd_per_kghm,
            site.effective_burnup_mwd_per_kghm,
            site.hbs_porosity,
        ),
        model.initial_state((), grain_radius=5.0e-6),
        tuple(
            jnp.asarray(value)
            for value in (
                fraction,
                BROAD_PORE_DENSITY,
                atoms,
                BROAD_SQUARED_VARIATION * BROAD_MEAN_ATOMS**2 * BROAD_PORE_DENSITY,
                vacancies,
                BROAD_BOUNDARY_GAS,
                atoms + BROAD_BOUNDARY_GAS,
                BROAD_BURNUP,
                BROAD_BURNUP,
                omega * atoms + SCHOTTKY_VACANCY_VOLUME * vacancies,
            )
        ),
    )
    result = advance_gas(
        model,
        state,
        temperature=BROAD_TEMPERATURE,
        previous_temperature=BROAD_TEMPERATURE,
        burnup=BROAD_BURNUP,
        time_step=BROAD_TIME_STEP,
        production=0.0,
    )
    assert bool(result.admissible)
    assert bool(result.diagnostics.hbs_root_converged)
    assert float(result.state.hbs_pore_variance) >= 0.0
    assert float(result.transfers.hbs_resolution) >= 0.0
    # Every source cluster has n>=2, so its radius and diffusion-limited
    # capture rate 4 pi D c R exceed the gas-only dimer bound regardless of
    # how broad the size distribution is.
    energy = BOLTZMANN * BROAD_TEMPERATURE
    dimer_radius = float(_sphere_radius(ATOMS_PER_NUCLEUS * omega))
    diffusivity = (
        _boundary_tilt(fraction)
        * GAS_DIFFUSIVITY_PREFACTOR
        * np.exp(-GAS_ACTIVATION_ENERGY / energy)
    )
    capture_bound = (
        BROAD_TIME_STEP
        * float(result.state.hbs_pore_number_density)
        * 4.0
        * np.pi
        * diffusivity
        * float(result.state.hbs_boundary_gas)
        * dimer_radius
    )
    assert float(result.transfers.hbs_trapping) >= capture_bound * (1.0 - 1.0e-10)
    np.testing.assert_allclose(
        float(result.retained_total + result.released_increment),
        atoms + BROAD_BOUNDARY_GAS,
        rtol=1.0e-12,
    )
