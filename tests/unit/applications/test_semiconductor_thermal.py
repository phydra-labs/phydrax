import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.semiconductor._quantities import (
    BOLTZMANN_CONSTANT_SI as KB,
    ELEMENTARY_CHARGE_SI as Q,
)
from phydrax.applications.semiconductor._thermal import (
    CarrierEnergyRelaxation,
    CarrierEnergyTransport,
    ConstantLatticeHeatCapacity,
    ThermalBoundaryExchange,
    ThermalConductance,
)
from phydrax.applications.semiconductor._thermodynamics import BandThermodynamics


def _bands(statistics="boltzmann"):
    return BandThermodynamics(
        "synthetic parabolic energy fixture",
        conduction_band_edge=0.7 * Q,
        valence_band_edge=-0.5 * Q,
        conduction_density_of_states=2e25,
        valence_density_of_states=1e25,
        reference_temperature=300.0,
        temperature_range=(100.0, 2000.0),
        energy_reference="synthetic zero",
        provenance="analytic parabolic fixture; no empirical material claim",
        statistics=statistics,
    )


def test_lattice_capacity_is_energy_derivative_and_has_invertible_datum():
    law = ConstantLatticeHeatCapacity(
        2e6,
        reference_temperature=300.0,
        temperature_range=(200.0, 900.0),
        provenance="synthetic constant-cv body",
    )
    temperatures = jnp.array([250.0, 300.0, 700.0])
    energies = law.internal_energy(temperatures)
    np.testing.assert_allclose(energies, [-1e8, 0.0, 8e8])
    np.testing.assert_allclose(law.temperature(energies), temperatures)
    np.testing.assert_allclose(
        jax.grad(law.internal_energy)(500.0), law.heat_capacity(500.0)
    )
    assert np.isnan(law.internal_energy(1000.0))
    # Finite-volume inventory, not a temperature difference, is the storage.
    volume = 3e-18
    np.testing.assert_allclose(
        volume * (law.internal_energy(450.0) - law.internal_energy(350.0)), 6e-10
    )


def test_thermal_boundary_is_passive_and_exactly_closes_reservoir_power():
    conductor = ThermalConductance(
        0.04,
        temperature_range=(200.0, 900.0),
        provenance="synthetic thermal-port conductance fixture",
    )
    boundary = ThermalBoundaryExchange(conductor, 300.0)
    result = boundary.evaluate(jnp.array([250.0, 300.0, 600.0]))
    np.testing.assert_allclose(result.left_energy_source, [2.0, 0.0, -12.0])
    np.testing.assert_allclose(
        result.left_energy_source + result.right_energy_source, 0.0, atol=0.0
    )
    assert np.all(result.entropy_production >= 0)
    np.testing.assert_allclose(result.entropy_production[-1], 0.02)


def test_degenerate_carrier_relaxation_uses_shared_energy_and_cancels_heat():
    bands = _bands("fermi-dirac")
    law = CarrierEnergyRelaxation(
        bands,
        "electron",
        2e-12,
        temperature_range=(200.0, 1200.0),
        provenance="synthetic constant energy-relaxation time",
    )
    density = 8e25
    result = law.evaluate(density, 800.0, 300.0)
    expected_hot = bands.electron_energy_density(density, 800.0)
    expected_cold = bands.electron_energy_density(density, 300.0)
    np.testing.assert_allclose(
        result.carrier_energy_source, -(expected_hot - expected_cold) / 2e-12
    )
    assert float(result.carrier_energy_source) < 0
    np.testing.assert_allclose(
        result.carrier_energy_source + result.lattice_energy_source, 0.0, atol=0.0
    )
    equilibrium = law.evaluate(density, 300.0, 300.0)
    assert float(equilibrium.carrier_energy_source) == 0.0
    # In the degenerate gas the low-temperature energy is not 3nkT/2.
    assert float(result.equilibrium_internal_energy) > 1.5 * density * KB * 300.0


@pytest.mark.parametrize("carrier,charge_sign", [("electron", -1), ("hole", 1)])
def test_carrier_face_energy_closes_band_work_and_gauge_change(carrier, charge_sign):
    transport = CarrierEnergyTransport(
        _bands(),
        carrier,
        0.5,
        temperature_range=(200.0, 1200.0),
        provenance="synthetic parabolic enthalpy/Fourier moment closure",
    )
    flux, nl, nr, tl, tr, metric = -2e10, 1e22, 3e22, 500.0, 300.0, 2e-8
    # Electron band edges at different potentials share the same -q psi shift.
    bl, br = 0.6 * Q, 0.2 * Q
    result = transport.evaluate(flux, nl, nr, tl, tr, bl, br, metric)
    expected_kinetic = flux * 2.5 * KB * tr + 0.5 * metric * (tl - tr)
    np.testing.assert_allclose(result.kinetic_power, expected_kinetic)
    left_total = result.left_kinetic_source + result.left_band_storage_source
    right_total = result.right_kinetic_source + result.right_band_storage_source
    np.testing.assert_allclose(left_total, -result.total_power)
    np.testing.assert_allclose(right_total, result.total_power)
    band_work = -charge_sign * (bl - br) * flux
    np.testing.assert_allclose(
        result.left_kinetic_source + result.right_kinetic_source, band_work
    )
    shifted = transport.evaluate(flux, nl, nr, tl, tr, bl + Q, br + Q, metric)
    np.testing.assert_allclose(shifted.left_kinetic_source, result.left_kinetic_source)
    np.testing.assert_allclose(shifted.right_kinetic_source, result.right_kinetic_source)
    np.testing.assert_allclose(
        shifted.total_power - result.total_power, -charge_sign * Q * flux
    )
