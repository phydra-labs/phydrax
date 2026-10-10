"""FRAPCON-4.0 fuel-cladding gap conductance, GTHCON gas conductivity, CMHARD hardness.

Independent references: the open-gap limit h_gas -> k_mix / d; the gray-body
exchange between coaxial cylinders written unfactored, sigma F (T_f^4 - T_c^4) /
(T_f - T_c) (PNNL-19418 Vol. 1 Rev. 2 Eqs. 2.13-2.14) and its equal-temperature
limit 4 sigma F T^3; CMHARD values quoted in PNNL-19417 Rev. 2 Section 3.9 (about
2e9 Pa at room temperature, 1.94e8 Pa at the highest-temperature data point,
875 K) and the "approximately 680 MPa" hardness quoted with Eq. 2.19 of
PNNL-19418 Vol. 1 Rev. 2 for operating cladding (evaluated at 600 K).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.typing import ArrayLike

from phydrax.applications.nuclear_fuel._gap import (
    FuelGasConductivity,
    GapConductanceLaw,
    GasAccommodationLaw,
    ZircaloyMeyerHardness,
)
from phydrax.equations import brokaw_mixture_conductivity


pytestmark = pytest.mark.strict_jax

_HELIUM = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0])
_HELIUM_XENON = np.asarray([0.7, 0.0, 0.03, 0.27, 0.0])
_STEFAN_BOLTZMANN_FRAPCON = 5.6697e-8
_FUEL_EMISSIVITY = 0.8
_CLADDING_EMISSIVITY = 0.75
_FUEL_RADIUS = 4.1e-3
_CLADDING_RADIUS = 4.18e-3


def _inputs(**overrides: ArrayLike) -> dict[str, ArrayLike]:
    values: dict[str, ArrayLike] = {
        "fuel_surface_temperature": 800.0,
        "cladding_surface_temperature": 620.0,
        "gap_width": 40.0e-6,
        "contact_pressure": 0.0,
        "gas_mole_fractions": _HELIUM,
        "gas_pressure": 5.0e6,
        "fuel_roughness": 2.0e-6,
        "cladding_roughness": 0.5e-6,
        "fuel_emissivity": _FUEL_EMISSIVITY,
        "cladding_emissivity": _CLADDING_EMISSIVITY,
        "fuel_radius": _FUEL_RADIUS,
        "cladding_inner_radius": _CLADDING_RADIUS,
        "fuel_conductivity": 3.0,
        "cladding_conductivity": 16.0,
    }
    values.update(overrides)
    return values


# F = 1 / [1/e_f + (r_f/r_c)(1/e_c - 1)], PNNL-19418 Vol. 1 Rev. 2 Eq. 2.13.
_EMISSIVITY_FACTOR = 1.0 / (
    1.0 / _FUEL_EMISSIVITY
    + _FUEL_RADIUS / _CLADDING_RADIUS * (1.0 / _CLADDING_EMISSIVITY - 1.0)
)


@pytest.mark.parametrize(
    "composition", [_HELIUM, _HELIUM_XENON], ids=["helium", "helium-krypton-xenon"]
)
def test_wide_open_gap_conducts_like_a_gas_slab(composition: np.ndarray) -> None:
    law = GapConductanceLaw()
    width = 1.0e-3
    result = law.conductance(**_inputs(gap_width=width, gas_mole_fractions=composition))
    mean_temperature = 0.5 * (800.0 + 620.0)
    frap = FuelGasConductivity()
    species = frap.species_conductivity(mean_temperature)
    conductivity = brokaw_mixture_conductivity(
        composition, species, np.asarray(frap.molar_mass_g_per_mol)
    )
    # Roughness, jump distance, and offset add a few micrometres to 1 mm.
    np.testing.assert_allclose(result.gas * width / conductivity, 1.0, rtol=5e-3)
    assert float(result.solid) == 0.0


def test_rarefied_gas_lowers_narrow_gap_conductance() -> None:
    law = GapConductanceLaw()
    dense = law.conductance(**_inputs(gap_width=5.0e-6, gas_pressure=5.0e6))
    rarefied = law.conductance(**_inputs(gap_width=5.0e-6, gas_pressure=0.1e6))
    assert float(rarefied.gas) < float(dense.gas)
    assert float(rarefied.effective_gap_width) > float(dense.effective_gap_width)


def test_closed_gap_contact_conductance_grows_with_pressure() -> None:
    law = GapConductanceLaw()
    pressures = jnp.asarray([0.0, 1.0e5, 2.0e6, 1.0e7, 3.0e7])
    result = law.conductance(**_inputs(gap_width=0.0, contact_pressure=pressures))
    solid = np.asarray(result.solid)
    assert solid[0] == 0.0
    assert np.all(solid[1:] > 0.0)
    assert np.all(np.diff(solid) > 0.0)
    np.testing.assert_allclose(
        result.total, result.gas + result.solid + result.radiation, rtol=1e-15
    )


def test_contact_derivative_is_finite_at_zero_pressure() -> None:
    law = GapConductanceLaw()

    def total(pressure: jax.Array) -> jax.Array:
        return law.conductance(**_inputs(gap_width=0.0, contact_pressure=pressure)).total

    assert np.isfinite(float(jax.grad(total)(jnp.asarray(0.0))))


def test_radiation_is_the_gray_cylinder_exchange() -> None:
    law = GapConductanceLaw()
    result = law.conductance(**_inputs())
    expected = (
        _STEFAN_BOLTZMANN_FRAPCON
        * _EMISSIVITY_FACTOR
        * (800.0**4 - 620.0**4)
        / (800.0 - 620.0)
    )
    np.testing.assert_allclose(result.radiation, expected, rtol=1e-12)


def test_radiation_stays_finite_at_equal_surface_temperatures() -> None:
    law = GapConductanceLaw()
    result = law.conductance(
        **_inputs(fuel_surface_temperature=700.0, cladding_surface_temperature=700.0)
    )
    expected = 4.0 * _STEFAN_BOLTZMANN_FRAPCON * _EMISSIVITY_FACTOR * 700.0**3
    np.testing.assert_allclose(result.radiation, expected, rtol=1e-14)


@pytest.mark.parametrize(
    ("temperature", "published", "tolerance"),
    [(298.0, 2.0e9, 0.1), (600.0, 680.0e6, 0.03), (875.0, 1.94e8, 0.01)],
    ids=["room-temperature", "operating", "highest-data-point"],
)
def test_meyer_hardness_matches_published_values(
    temperature: float, published: float, tolerance: float
) -> None:
    value = ZircaloyMeyerHardness().hardness(temperature)
    np.testing.assert_allclose(value, published, rtol=tolerance)


def test_gas_composition_axis_is_required() -> None:
    law = GapConductanceLaw()
    with pytest.raises(ValueError, match="fuel gas species axis"):
        law.conductance(**_inputs(gas_mole_fractions=np.asarray([1.0, 0.0])))


def test_cold_gap_reproduction_separates_extrapolation_and_pressure_evidence() -> None:
    law = GapConductanceLaw(accommodation=GasAccommodationLaw.gap_comparison())
    result = law.conductance(
        **_inputs(
            fuel_surface_temperature=300.0,
            cladding_surface_temperature=300.0,
            gas_pressure=5.0e6,
        )
    )
    assert bool(result.successful)
    assert bool(result.within_validity)
    assert bool(result.accommodation_extrapolated)
    assert not bool(result.within_source_comparison)
    atmosphere = law.conductance(
        **_inputs(
            fuel_surface_temperature=300.0,
            cladding_surface_temperature=300.0,
            gas_pressure=0.103e6,
        )
    )
    assert bool(atmosphere.within_source_comparison)
    default = GapConductanceLaw().conductance(
        **_inputs(
            fuel_surface_temperature=300.0,
            cladding_surface_temperature=300.0,
        )
    )
    assert not bool(default.within_validity)


def test_supplied_accommodation_uses_same_jump_distance_physics() -> None:
    # A declared synthetic constant law isolates accommodation's 1/a jump
    # resistance, without pretending the inputs are measurements.
    coefficients = np.column_stack([np.full(5, 0.25), np.zeros(5)])
    weaker = GapConductanceLaw(
        accommodation=GasAccommodationLaw(
            coefficients=coefficients,
            temperature_range=(298.0, 1000.0),
            source_id="synthetic-accommodation-reference",
        )
    )
    stronger = GapConductanceLaw(
        accommodation=GasAccommodationLaw(
            coefficients=2.0 * coefficients,
            temperature_range=(298.0, 1000.0),
            source_id="synthetic-accommodation-reference",
        )
    )
    low = weaker.conductance(**_inputs(gap_width=0.0))
    high = stronger.conductance(**_inputs(gap_width=0.0))
    roughness_length = 2.0e-6 + 0.5e-6 - 1.397e-6
    np.testing.assert_allclose(
        low.effective_gap_width - roughness_length,
        2.0 * (high.effective_gap_width - roughness_length),
        rtol=1e-13,
    )
    assert float(low.gas) < float(high.gas)


def test_nonpositive_gas_path_is_reported_as_failure() -> None:
    result = GapConductanceLaw().conductance(
        **_inputs(
            gap_width=0.0,
            fuel_roughness=1.0e-8,
            cladding_roughness=1.0e-8,
            gas_pressure=1.0e12,
        )
    )
    assert float(result.effective_gap_width) < 0.0
    assert not bool(result.successful)
