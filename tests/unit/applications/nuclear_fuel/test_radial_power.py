"""TUBRNP composition, energy weighting, and cylindrical diffusion contracts.

Independent references use SciPy's Bessel functions and matrix exponential for
PNNL-19418 Vol. 1 Rev. 2 Eqs. 2.41–2.51. Test scattering values are explicitly
synthetic equation inputs, not a claim about the unavailable FRAPCON coefficient.
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import iv

from phydrax.applications.nuclear_fuel._radial_power import (
    RadialPowerProfileModel,
    RadialPowerState,
)
from phydrax.nuclear import FISSIONING_NUCLIDES, HeavyMetalReference


pytestmark = pytest.mark.strict_jax

_RADIUS = 4.78e-3
_ENERGY_PER_FISSION = 200.0e6 * 1.602176634e-19


def _rings(radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Ring midpoints and exact ring areas, refined toward the pellet surface."""

    edges = np.unique(
        np.concatenate(
            [
                radius * np.sqrt(np.linspace(0.0, 1.0, 201)),
                radius * (1.0 - np.geomspace(1e-5, 0.05, 60)),
                [0.99 * radius],
            ]
        )
    )
    midpoints = 0.5 * (edges[1:] + edges[:-1])
    areas = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    # Zero-weight probes at 0.99 r0 and at the surface.
    return (
        np.concatenate([midpoints, [0.99 * radius, radius]]),
        np.concatenate([areas, [0.0, 0.0]]),
    )


def _fresh(model: RadialPowerProfileModel, enrichment: float) -> RadialPowerState:
    radii, weights = _rings(_RADIUS)
    heavy_metal = HeavyMetalReference.oxide_fuel(
        fuel_density=10400.0, uranium_235_weight_fraction=enrichment
    )
    return model.initial_state(
        radii, weights=weights, outer_radius=_RADIUS, heavy_metal=heavy_metal
    )


@pytest.fixture(scope="module")
def model() -> RadialPowerProfileModel:
    return RadialPowerProfileModel(
        energy_per_fission=np.full(6, _ENERGY_PER_FISSION),
        scattering_cross_section_barn=12.0,
        scattering_source_id="synthetic-diffusion-equation-reference",
        form_depth_unit_m=1.0e-3,
    )


def test_fresh_power_matches_cylindrical_diffusion(
    model: RadialPowerProfileModel,
) -> None:
    state = _fresh(model, 0.045)
    absorption = np.asarray(
        model.fission_cross_section_barn + model.capture_cross_section_barn
    )
    density = float(state.heavy_metal_atom_density)
    sigma_a = density * np.dot(np.asarray(state.atom_fractions), absorption) * 1e-28
    sigma_s = density * 12.0e-28
    raw = iv(0, np.sqrt(3.0 * sigma_a * sigma_s) * np.asarray(state.radii))
    expected = raw * np.sum(state.weights) / np.sum(np.asarray(state.weights) * raw)
    np.testing.assert_allclose(model.power_shape(state), expected, rtol=2e-13)
    assert expected[-1] > expected[0]


def test_power_integrates_to_the_reference_area(model: RadialPowerProfileModel) -> None:
    state = model.advance(_fresh(model, 0.045), burnup_increment_mwd_per_kghm=40.0)
    shape = model.power_shape(state)
    np.testing.assert_allclose(
        jnp.sum(state.weights * shape), np.pi * _RADIUS**2, rtol=1e-13
    )
    assert bool(state.successful)


def test_rim_peaking_grows_with_burnup(model: RadialPowerProfileModel) -> None:
    state = _fresh(model, 0.045)
    surface, centre = [], []
    for increment in (10.0, 20.0, 30.0):
        state = model.advance(state, burnup_increment_mwd_per_kghm=increment)
        shape = np.asarray(model.power_shape(state))
        surface.append(shape[-1])
        centre.append(shape[0])
    assert np.all(np.diff(surface) > 0.0)
    assert np.all(np.diff(centre) < 0.0)
    np.testing.assert_allclose(state.burnup_mwd_per_kghm, 60.0, rtol=1e-15)


def test_burnup_steps_compose_exactly(model: RadialPowerProfileModel) -> None:
    fresh = _fresh(model, 0.045)
    single = model.advance(fresh, burnup_increment_mwd_per_kghm=50.0)
    split = model.advance(
        model.advance(fresh, burnup_increment_mwd_per_kghm=20.0),
        burnup_increment_mwd_per_kghm=30.0,
    )
    np.testing.assert_allclose(split.atom_fractions, single.atom_fractions, rtol=1e-10)
    np.testing.assert_allclose(split.flux_time, single.flux_time, rtol=1e-10)


def test_fission_fractions_sum_to_one_and_shift_to_plutonium(
    model: RadialPowerProfileModel,
) -> None:
    fresh = _fresh(model, 0.045)
    burned = model.advance(fresh, burnup_increment_mwd_per_kghm=50.0)
    fresh_fractions = np.asarray(model.fission_fractions(fresh))
    fractions = np.asarray(model.fission_fractions(burned))
    np.testing.assert_allclose(fractions.sum(axis=-1), 1.0, rtol=1e-14)
    uranium_235 = FISSIONING_NUCLIDES.index("U235")
    plutonium_239 = FISSIONING_NUCLIDES.index("Pu239")
    np.testing.assert_allclose(fresh_fractions[..., uranium_235], 1.0, rtol=1e-15)
    # Plutonium production is rim-weighted, so the rim burns more plutonium.
    assert fractions[-1, plutonium_239] > fractions[0, plutonium_239]
    assert fractions[0, uranium_235] > fractions[-1, uranium_235]


def test_negative_burnup_increment_is_refused(model: RadialPowerProfileModel) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="nonnegative"):
        model.advance(_fresh(model, 0.045), burnup_increment_mwd_per_kghm=-1.0)


def test_fresh_fuel_without_fission_is_refused(model: RadialPowerProfileModel) -> None:
    with pytest.raises(eqx.EquinoxRuntimeError, match="positive fission rate"):
        _fresh(model, 0.0)


def test_energy_weighted_burnup_and_fission_count() -> None:
    energies = _ENERGY_PER_FISSION * np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
    model = RadialPowerProfileModel(
        energy_per_fission=energies,
        scattering_cross_section_barn=12.0,
        scattering_source_id="synthetic-diffusion-equation-reference",
        form_depth_unit_m=1.0e-3,
    )
    fresh = _fresh(model, 0.045)
    state = model.advance(fresh, burnup_increment_mwd_per_kghm=50.0)
    fission = np.asarray(model.fission_cross_section_barn)
    capture = np.asarray(model.capture_cross_section_barn)
    # Independent augmented Bateman-chain exponential, with separate fission
    # and recoverable-energy accumulators.
    generator = np.zeros((8, 8))
    generator[np.arange(6), np.arange(6)] = -(fission + capture)
    generator[np.arange(2, 6), np.arange(1, 5)] = capture[1:-1]
    generator[6, :6] = fission
    generator[7, :6] = fission * energies / energies[0]
    initial = np.concatenate([np.asarray(fresh.atom_fractions), [0.0, 0.0]])
    reference = expm(generator * float(state.flux_time) * 1e-28) @ initial
    np.testing.assert_allclose(state.atom_fractions, reference[:6], rtol=3e-11)
    np.testing.assert_allclose(state.fima, reference[6], rtol=3e-11)
    expected_energy = (
        50.0 * 8.64e10 * float(state.heavy_metal_mass_per_atom) / energies[0]
    )
    np.testing.assert_allclose(reference[7], expected_energy, rtol=3e-11)
    assert float(state.fima) < expected_energy


def test_local_composition_preserves_pellet_average(
    model: RadialPowerProfileModel,
) -> None:
    state = model.advance(_fresh(model, 0.045), burnup_increment_mwd_per_kghm=60.0)
    local = model.local_atom_fractions(state)
    average = jnp.sum(state.weights[:, None] * local, axis=0) / jnp.sum(state.weights)
    np.testing.assert_allclose(average, state.atom_fractions, rtol=1e-13)
    assert np.all(np.asarray(local) >= 0.0)


def test_power_energy_weights_do_not_change_fission_fraction_identity(
    model: RadialPowerProfileModel,
) -> None:
    state = model.advance(_fresh(model, 0.045), burnup_increment_mwd_per_kghm=30.0)
    changed = eqx.tree_at(
        lambda law: law.energy_per_fission,
        model,
        _ENERGY_PER_FISSION * jnp.asarray([1.0, 1.1, 2.0, 3.0, 4.0, 5.0]),
    )
    np.testing.assert_array_equal(
        changed.fission_fractions(state), model.fission_fractions(state)
    )
    assert float(changed.power_shape(state)[-1]) > float(model.power_shape(state)[-1])
