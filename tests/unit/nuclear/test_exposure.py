from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


_JOULES_PER_MWD = 1.0e6 * 86400.0
_AVOGADRO = 6.02214076e23
# AME2020 (mass_1.mas20) atomic masses [u] of U-234, U-235, U-236, U-238.
_URANIUM_MASSES = np.asarray([234.040950296, 235.043928117, 236.045566130, 238.050786936])


def _reference(fuel_density: float = 10_412.0) -> phx.nuclear.HeavyMetalReference:
    return phx.nuclear.HeavyMetalReference.oxide_fuel(
        fuel_density=fuel_density,
        uranium_235_weight_fraction=0.045,
        uranium_234_weight_fraction=0.0004,
    )


@pytest.mark.strict_jax
def test_burnup_is_energy_per_initial_heavy_metal_mass_in_mwd() -> None:
    energy = jnp.asarray([_JOULES_PER_MWD, 45.5 * _JOULES_PER_MWD, 0.0])
    zero = jnp.zeros(3)
    state = phx.nuclear.ExposureState(energy, zero, zero, zero)

    np.testing.assert_array_equal(state.burnup_mwd_per_kghm, [1.0, 45.5, 0.0])


@pytest.mark.strict_jax
def test_exposure_accumulates_against_reference_inventory() -> None:
    reference = _reference()
    energy_per_fission = 3.2e-11
    rates = jnp.asarray([[2.0e19, 1.0e19, 0.0], [3.5e19, 0.5e19, 1.0e18]])
    fluxes = jnp.asarray([[4.0e17, 1.0e17, 0.0], [6.0e17, 2.0e17, 3.0e16]])
    steps = (3.6e5, 8.64e5)
    step = jax.jit(phx.nuclear.advance_exposure)
    state = phx.nuclear.ExposureState.zeros((3,))
    for rate, flux, dt in zip(rates, fluxes, steps, strict=True):
        state = step(
            state,
            reference,
            fission_rate_density=rate,
            energy_per_fission=energy_per_fission,
            fast_flux=flux,
            time_step=dt,
        )

    fissions = rates[0] * steps[0] + rates[1] * steps[1]
    np.testing.assert_allclose(state.fission_density, fissions, rtol=1e-15)
    np.testing.assert_allclose(
        state.fissions_per_initial_heavy_metal_atom,
        state.fission_density / reference.heavy_metal_atom_density,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        state.burnup_mwd_per_kghm,
        state.fission_density
        * energy_per_fission
        / reference.heavy_metal_mass_density
        / _JOULES_PER_MWD,
        rtol=1e-14,
    )
    np.testing.assert_allclose(
        state.fast_fluence, fluxes[0] * steps[0] + fluxes[1] * steps[1], rtol=1e-15
    )


def test_burnup_is_invariant_to_fabricated_fuel_density() -> None:
    # Two pellets of different sintered density run at the same specific power
    # [W/kg HM] reach the same burnup; the one whose volume later densifies is
    # still referred to its as-fabricated heavy-metal inventory.
    specific_power = 38.0e3
    energy_per_fission = 3.1e-11
    duration = 3.0e7
    burnups = []
    for density in (10_140.0, 10_687.0):
        reference = _reference(density)
        rate = (
            specific_power
            * float(reference.heavy_metal_mass_density)
            / energy_per_fission
        )
        state = phx.nuclear.ExposureState.zeros()
        for _ in range(4):
            state = phx.nuclear.advance_exposure(
                state,
                reference,
                fission_rate_density=rate,
                energy_per_fission=energy_per_fission,
                fast_flux=0.0,
                time_step=duration / 4,
            )
        burnups.append(float(state.burnup_mwd_per_kghm))

    expected = specific_power * duration / _JOULES_PER_MWD
    np.testing.assert_allclose(burnups, [expected, expected], rtol=1e-14)


def test_oxide_reference_recovers_declared_uranium_composition() -> None:
    reference = _reference()
    densities = np.asarray(reference.uranium_number_densities)
    masses = densities * _URANIUM_MASSES
    weight_fractions = masses / masses.sum()

    np.testing.assert_allclose(
        weight_fractions, [0.0004, 0.045, 0.0, 1.0 - 0.0454], rtol=1e-13, atol=1e-16
    )
    np.testing.assert_allclose(
        float(reference.heavy_metal_mass_density),
        masses.sum() * 1.0e-3 / _AVOGADRO,
        rtol=1e-14,
    )
    assert float(reference.heavy_metal_atom_density) == pytest.approx(densities.sum())


def test_natural_uranium_dioxide_heavy_metal_fraction() -> None:
    # CIAAW standard atomic weights: U 238.02891, O 15.999; natural isotopic
    # amount fractions U-234 0.000054, U-235 0.007204, U-238 0.992742.
    atoms = np.asarray([0.000054, 0.007204, 0.0, 0.992742])
    weights = atoms * _URANIUM_MASSES / np.sum(atoms * _URANIUM_MASSES)
    reference = phx.nuclear.HeavyMetalReference.oxide_fuel(
        fuel_density=10_970.0,
        uranium_234_weight_fraction=float(weights[0]),
        uranium_235_weight_fraction=float(weights[1]),
    )

    fraction = float(reference.heavy_metal_mass_density) / 10_970.0
    assert fraction == pytest.approx(238.02891 / (238.02891 + 2 * 15.999), rel=2e-6)


@pytest.mark.parametrize(
    "arguments",
    [
        pytest.param({"fuel_density": 0.0}, id="zero-density"),
        pytest.param({"fuel_density": -10_400.0}, id="negative-density"),
        pytest.param({"fuel_density": math.nan}, id="nan-density"),
        pytest.param({"fuel_density": math.inf}, id="infinite-density"),
        pytest.param({"uranium_235_weight_fraction": -0.01}, id="negative-enrichment"),
        pytest.param({"uranium_235_weight_fraction": 1.2}, id="enrichment-above-one"),
        pytest.param({"uranium_235_weight_fraction": math.nan}, id="nan-enrichment"),
        pytest.param(
            {"uranium_235_weight_fraction": 0.6, "uranium_234_weight_fraction": 0.5},
            id="isotopes-exceed-one",
        ),
        pytest.param({"oxygen_to_metal_ratio": 0.0}, id="zero-oxygen"),
        pytest.param({"oxygen_to_metal_ratio": 3.5}, id="beyond-uo3"),
    ],
)
def test_oxide_reference_refuses_invalid_fabrication_data(
    arguments: dict[str, float],
) -> None:
    values = {"fuel_density": 10_412.0, "uranium_235_weight_fraction": 0.045}
    values.update(arguments)
    with pytest.raises(ValueError):
        phx.nuclear.HeavyMetalReference.oxide_fuel(**values)
