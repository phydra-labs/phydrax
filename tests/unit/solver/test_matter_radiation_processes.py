#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Independent invariants for M1b/M1c electromagnetic shower processes."""

from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from scipy import integrate

import phydrax as phx


_MASS_EV = 510998.95069
_SCALE = phx.ElectromagneticScaleContract.si()
_ALPHA = float(_SCALE.fine_structure)
_PLANCK_EV_S = float(
    2.0 * np.pi * _SCALE.reduced_planck_constant / _SCALE.elementary_charge
)
_LIGHT_SPEED = float(_SCALE.speed_of_light)


def _manifest() -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        "nist-seltzer-berger-test-extract",
        checksum_algorithm="sha256",
        checksum="8" * 64,
        size_bytes=1,
        license_id="nist-public-domain-17-usc-105",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"fixture": 0.0},
        lineage_ids=("nist:seltzer-berger:test-extract",),
    )


def _photon_table(name: str, energy_grid: Any, values: tuple[float, float]) -> Any:
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(),
        f"https://example.invalid/{name}",
        "pair-production-test",
        "fixture",
        name,
    )
    unit = phx.units.derived_unit(
        "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
    )
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        energy_grid,
        ("material",),
        jnp.asarray((values,)),
        unit,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def test_pair_transport_creates_typed_daughters_and_closes_ledger() -> None:
    grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((1.0e6, 3.0e6))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    zero_photo = _photon_table("photo", grid, (0.0, 0.0))
    zero_compton = _photon_table("compton", grid, (0.0, 0.0))
    zero_rayleigh = _photon_table("rayleigh", grid, (0.0, 0.0))
    pair_nuclear = _photon_table("pair-nuclear", grid, (1.0e3, 1.0e3))
    pair_electron = _photon_table("pair-electron", grid, (0.0, 0.0))
    library = phx.equations.RadiationCrossSectionLibrary(
        zero_photo,
        zero_compton,
        zero_rayleigh,
        jnp.asarray((1.0,)),
        pair_nuclear=pair_nuclear,
        pair_electron=pair_electron,
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )
    result = phx.solver.PhotonTransportPlan(
        geometry,
        library,
        maximum_events=1,
        cutoff_energy=1.0e6,
        electron_stack=phx.solver.SecondaryStackSpec(2, minimum_energy=1.0),
    ).simulate(
        jnp.asarray(((0.5, 0.5, 0.5),)),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((2.0e6,)),
        jr.key(4),
    )
    stack = result.secondary_electrons
    assert stack is not None
    np.testing.assert_array_equal(stack.particle_kind[0], jnp.asarray((0, 1)))
    assert int(stack.count[0]) == 2
    np.testing.assert_allclose(
        float(jnp.sum(stack.energies[0]) + jnp.sum(result.material_kerma[0])),
        2.0e6,
        rtol=1.0e-12,
    )
    assert float(result.maximum_ledger_residual) < 1.0e-8


def test_bethe_heitler_threshold_and_high_energy_asymptote() -> None:
    energies = jnp.asarray((2.0 * _MASS_EV, 2.000001 * _MASS_EV, 1.0e12))
    nuclear, electron = phx.equations.bethe_heitler_pair_cross_section_m2(
        energies, jnp.asarray(82.0)
    )
    assert float(nuclear[0]) == 0.0 and float(electron[0]) == 0.0
    assert 0.0 < float(nuclear[1]) < float(nuclear[2])
    common = (
        4.0
        * _ALPHA
        * phx.ElectromagneticScaleContract.si().classical_electron_radius ** 2
    )
    expected = (
        common
        * (7.0 / 9.0)
        * 82.0**2
        * (np.log(183.0 * 82.0 ** (-1.0 / 3.0)) - 1.0 / 42.0)
    )
    np.testing.assert_allclose(float(nuclear[-1]), expected, rtol=1.0e-9)


def test_seltzer_berger_inverse_cdf_normalization_and_mean() -> None:
    fractions = np.linspace(0.01, 0.99, 99)
    energy = np.asarray((1.0e6, 2.0e6))
    density = (4.0 / 3.0 - 4.0 * fractions / 3.0 + fractions**2) / fractions
    values = np.broadcast_to(density, (1, energy.size, fractions.size))
    table = phx.equations.SeltzerBergerBremsstrahlungTable(
        energy,
        fractions,
        values,
        ("lead",),
        _manifest(),
        "https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html",
    )
    draws = jnp.asarray((np.arange(20000, dtype=np.float64) + 0.5) / 20000.0)
    sampled, supported = table.sample_fraction(0, 1.5e6, draws)
    assert bool(jnp.all(supported))
    norm = integrate.quad(
        lambda value: np.interp(value, fractions, density),
        0.01,
        0.99,
        points=fractions[1:-1],
        limit=200,
    )[0]
    expected_mean = (
        integrate.quad(
            lambda value: value * np.interp(value, fractions, density),
            0.01,
            0.99,
            points=fractions[1:-1],
            limit=200,
        )[0]
        / norm
    )
    np.testing.assert_allclose(float(jnp.mean(sampled)), expected_mean, rtol=2.0e-4)
    np.testing.assert_allclose(
        np.asarray(table.cumulative_probability)[:, :, -1], 1.0, atol=1.0e-15
    )


def test_delta_ray_and_annihilation_four_momentum_closure() -> None:
    for kind, transfer in ((0, 2.0e5), (1, 6.0e5)):
        result = phx.equations.delta_ray_kinematics(1.0e6, transfer, kind)
        assert bool(result.valid)
        assert float(result.four_momentum_residual_ev) < 1.0e-8
    annihilation = phx.equations.positron_annihilation_in_flight(2.0e6, 0.37)
    assert bool(annihilation.valid)
    np.testing.assert_allclose(
        float(jnp.sum(annihilation.photon_energies_ev)),
        2.0e6 + 2.0 * _MASS_EV,
        rtol=2.0e-15,
    )
    assert float(annihilation.four_momentum_residual_ev) < 1.0e-8


def test_lpm_ter_mikaelian_limits_and_disabled_identity() -> None:
    energy = jnp.asarray((1.0e8, 1.0e12))
    photon = jnp.asarray((1.0e7, 1.0e7))
    identity = phx.equations.bremsstrahlung_suppression_factor(energy, photon)
    np.testing.assert_array_equal(identity, jnp.ones_like(identity))
    lpm = phx.equations.bremsstrahlung_suppression_factor(
        energy, photon, lpm_energy_ev=1.0e9
    )
    dielectric_low = phx.equations.bremsstrahlung_suppression_factor(
        energy, jnp.asarray((1.0, 1.0)), plasma_energy_ev=20.0
    )
    dielectric_high = phx.equations.bremsstrahlung_suppression_factor(
        energy, 0.5 * energy, plasma_energy_ev=20.0
    )
    assert float(lpm[1]) < float(lpm[0]) <= 1.0
    assert bool(jnp.all(dielectric_low < 1.0e-6))
    np.testing.assert_allclose(dielectric_high, 1.0, rtol=2.0e-8)


def test_atomic_relaxation_energy_closure() -> None:
    fluorescent = phx.equations.atomic_relaxation(8000.0, 1.0, 0.75, 0.5)
    auger = phx.equations.atomic_relaxation(8000.0, 0.0, 0.75, 0.5)
    for result in (fluorescent, auger):
        assert bool(result.valid)
        assert float(result.closure_residual_ev) == 0.0
        np.testing.assert_allclose(
            float(result.fluorescence_energy_ev + result.auger_energy_ev), 8000.0
        )
    assert float(fluorescent.fluorescence_energy_ev) == 6000.0
    assert float(auger.auger_energy_ev) == 8000.0


def test_garibian_cherry_interference_absorption_and_formation_scaling() -> None:
    transparent = phx.equations.FoilStackTransitionRadiationPlan(
        jnp.asarray((0.0, 1.0e-4, 2.0e-4, 3.0e-4)),
        jnp.asarray((400.0, -400.0, 400.0, -400.0)),
        jnp.zeros((4,)),
    )
    absorbing = phx.equations.FoilStackTransitionRadiationPlan(
        jnp.asarray((0.0, 1.0e-4, 2.0e-4, 3.0e-4)),
        jnp.asarray((400.0, -400.0, 400.0, -400.0)),
        jnp.full((4,), 1.0e4),
    )
    first = float(transparent.formation_length_m(5000.0, 1000.0, 1.0e-3))
    doubled_energy = float(transparent.formation_length_m(10000.0, 1000.0, 1.0e-3))
    doubled_angle = float(transparent.formation_length_m(5000.0, 1000.0, 2.0e-3))
    np.testing.assert_allclose(doubled_energy / first, 0.5, rtol=1.0e-12)
    np.testing.assert_allclose(doubled_angle / first, 0.4, rtol=1.0e-12)
    clear_yield = float(transparent.spectral_angular_yield(5000.0, 1000.0, 1.0e-3))
    absorbed_yield = float(absorbing.spectral_angular_yield(5000.0, 1000.0, 1.0e-3))
    assert clear_yield >= 0.0 and 0.0 <= absorbed_yield < clear_yield


def test_frank_tamm_count_and_energy_over_declared_band() -> None:
    length, beta, index = 0.25, 0.99, 1.5
    lower, upper = 300.0e-9, 600.0e-9
    count, energy = phx.equations.cherenkov_yield_in_band(
        length, beta, index, lower, upper
    )
    factor = 1.0 - 1.0 / (beta**2 * index**2)
    expected_count = 2.0 * np.pi * _ALPHA * length * factor * (1.0 / lower - 1.0 / upper)
    expected_energy = (
        np.pi
        * _ALPHA
        * _PLANCK_EV_S
        * _LIGHT_SPEED
        * length
        * factor
        * (1.0 / lower**2 - 1.0 / upper**2)
    )
    np.testing.assert_allclose(float(count), expected_count, rtol=2.0e-10)
    np.testing.assert_allclose(float(energy), expected_energy, rtol=2.0e-10)


def test_longo_profile_normalization_peak_and_radiation_length_scaling() -> None:
    depth = jnp.linspace(1.0e-6, 60.0, 200000)
    profile = phx.equations.longo_shower_profile(depth, 1.0e10, 8.0e7, incident="photon")
    np.testing.assert_allclose(float(jnp.trapezoid(profile, depth)), 1.0, rtol=2.0e-6)
    expected_peak = np.log(1.0e10 / 8.0e7) + 0.5
    observed_peak = float(depth[jnp.argmax(profile)])
    np.testing.assert_allclose(observed_peak, expected_peak, atol=4.0e-4)
    physical_depth = 0.35 * depth
    scaled = (
        phx.equations.longo_shower_profile(
            physical_depth / 0.35, 1.0e10, 8.0e7, incident="photon"
        )
        / 0.35
    )
    np.testing.assert_allclose(
        float(jnp.trapezoid(scaled, physical_depth)), 1.0, rtol=2.0e-6
    )


def _step_bank(order: np.ndarray) -> Any:
    starts = np.asarray(
        (((0.0, 0.0, 0.0), (0.0, 0.0, 0.1), (0.0, 0.0, 0.2)),),
        dtype=np.float64,
    )[:, order]
    ends = starts + np.asarray((0.0, 0.0, 0.1))
    active = jnp.ones((1, 3), dtype=jnp.bool_)
    return phx.solver.ChargedStepBank(
        jnp.asarray(starts),
        jnp.asarray(ends),
        jnp.full((1, 3), 0.99),
        jnp.full((1, 3), 0.99),
        jnp.zeros((1, 3)),
        jnp.zeros((1, 3), dtype=jnp.int32),
        active,
        jnp.asarray((3,), dtype=jnp.int32),
        jnp.asarray((0,), dtype=jnp.int32),
    )


def test_step_process_capacity_refusal_and_batch_order_invariance() -> None:
    admitted = phx.solver.ChargedStepRadiationPlan(
        jnp.asarray((1.5,)),
        wavelength_min_m=300.0e-9,
        wavelength_max_m=600.0e-9,
        photon_stack=phx.solver.SecondaryStackSpec(3, minimum_energy=1.0),
    )
    first = admitted.evaluate(_step_bank(np.asarray((0, 1, 2))))
    shuffled = admitted.evaluate(_step_bank(np.asarray((2, 0, 1))))
    assert bool(first.successful[0]) and bool(shuffled.successful[0])
    np.testing.assert_allclose(
        float(jnp.sum(first.secondary_photons.energies)),
        float(jnp.sum(shuffled.secondary_photons.energies)),
        rtol=1.0e-15,
    )
    refused_plan = phx.solver.ChargedStepRadiationPlan(
        jnp.asarray((1.5,)),
        wavelength_min_m=300.0e-9,
        wavelength_max_m=600.0e-9,
        photon_stack=phx.solver.SecondaryStackSpec(2, minimum_energy=1.0),
    )
    refused = refused_plan.evaluate(_step_bank(np.asarray((0, 1, 2))))
    assert bool(refused.refused[0]) and not bool(refused.successful[0])
    assert not bool(jnp.any(refused.secondary_photons.active))
    assert float(refused.ledger_residual_ev[0]) == 0.0
