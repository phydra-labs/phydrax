#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Electromagnetic shower coupling, secondary production, and identity addressing.

References are independent of the implementation: the Sauter (1931) K-shell
angular distribution and the Klein–Nishina recoil spectrum are integrated with
SciPy quadrature, and expected secondary counts follow from the tabulated
interaction rates and the realized step/exit lengths.
"""

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest
from scipy import integrate, stats

import phydrax as phx


_ELECTRON_REST_ENERGY_EV = 510998.95069
_EV_PER_JOULE = float(
    phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE)
)


def _manifest(name: str) -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum="d" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:shower",),
    )


def _photon_table(name: str, energy_grid: Any, values: tuple[float, float]) -> Any:
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name),
        f"https://example.invalid/{name}",
        "synthetic-shower-test",
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


def _photon_library(
    *,
    photoelectric: float,
    compton: float,
    compton_profile_j0: float | None = None,
) -> Any:
    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((1.0e3, 2.0e6)) * _EV_PER_JOULE
    )
    return phx.equations.RadiationCrossSectionLibrary(
        _photon_table("photoelectric", energy_grid, (photoelectric, photoelectric)),
        _photon_table("compton", energy_grid, (compton, compton)),
        _photon_table("rayleigh", energy_grid, (0.0, 0.0)),
        jnp.asarray((1.0,)),
        compton_profile_j0=(
            None if compton_profile_j0 is None else jnp.asarray((compton_profile_j0,))
        ),
    )


def _geometry() -> Any:
    return phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )


def _charged_materials(*, stopping: float, bremsstrahlung: float) -> Any:
    return phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((10.0, 2.0e6)),
        jnp.full((1, 2), stopping),
        jnp.zeros((1, 2)),
        jnp.full((1, 2), bremsstrahlung),
        ("material",),
        _manifest("charged-material"),
    )


def _shower(
    *,
    photon_capacity: int = 1024,
    charged_capacity: int = 1024,
    generations: int = 2,
    bremsstrahlung: float = 2.0,
) -> Any:
    geometry = _geometry()
    photon_plan = phx.solver.PhotonTransportPlan(
        geometry,
        _photon_library(photoelectric=1.0, compton=0.0),
        maximum_events=4,
        cutoff_energy=1.0e3,
        # Sauter rejection accepts about one proposal in three; 64 attempts keep
        # the per-photoelectron refusal probability below 1e-10.
        angular_sampling_attempts=64,
        electron_stack=phx.solver.SecondaryStackSpec(1, minimum_energy=10.0),
    )
    charged_plan = phx.solver.ChargedParticleTransportPlan(
        geometry,
        _charged_materials(stopping=1.0, bremsstrahlung=bremsstrahlung),
        maximum_steps=64,
        maximum_step_length=0.05,
        cutoff_energy_ev=10.0,
        step_bank_capacity=64,
        photon_stack=phx.solver.SecondaryStackSpec(64, minimum_energy=1.0e3),
    )
    return phx.solver.EMShowerPlan(
        photon_plan,
        charged_plan,
        photon_capacity=photon_capacity,
        charged_capacity=charged_capacity,
        maximum_generations=generations,
    )


def _primary_photons(count: int, *, capacity: int, identities: Any = None) -> Any:
    return phx.solver.ShowerParticleBatch.photons(
        jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.0)), (count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3)),
        jnp.full((count,), 2.0e5),
        capacity=capacity,
        identities=identities,
    )


def _sauter_density(cosine: float, kinetic_ev: float) -> float:
    gamma = 1.0 + kinetic_ev / _ELECTRON_REST_ENERGY_EV
    beta = np.sqrt(1.0 - 1.0 / gamma**2)
    factor = 1.0 - beta * cosine
    return (
        (1.0 - cosine**2)
        / factor**4
        * (1.0 + 0.5 * gamma * (gamma - 1.0) * (gamma - 2.0) * factor)
    )


def test_sauter_photoelectron_moments_at_100_kev() -> None:
    kinetic = 1.0e5
    count = 200_000
    attempts = 16
    proposals = jr.uniform(jr.key(11), (count, attempts))
    acceptances = jr.uniform(jr.key(12), (count, attempts))
    cosine, success = jax.vmap(
        lambda p, a: phx.equations.sample_sauter_cosine(
            p, a, jnp.asarray(kinetic), _ELECTRON_REST_ENERGY_EV
        )
    )(proposals, acceptances)
    cosine = np.asarray(cosine)[np.asarray(success)]
    assert cosine.size > 0.99 * count
    norm = integrate.quad(_sauter_density, -1.0, 1.0, args=(kinetic,))[0]
    for power in (1, 2):
        expected = (
            integrate.quad(lambda c: c**power * _sauter_density(c, kinetic), -1.0, 1.0)[0]
            / norm
        )
        samples = cosine**power
        standard_error = samples.std(ddof=1) / np.sqrt(samples.size)
        assert abs(samples.mean() - expected) < 4.0 * standard_error


def _klein_nishina_recoil_cdf(energy_ev: float) -> Any:
    alpha = energy_ev / _ELECTRON_REST_ENERGY_EV
    maximum = energy_ev * 2.0 * alpha / (1.0 + 2.0 * alpha)

    def density(recoil: float) -> float:
        ratio = (energy_ev - recoil) / energy_ev
        cosine = 1.0 - (1.0 / ratio - 1.0) / alpha
        return ratio + 1.0 / ratio - (1.0 - cosine**2)

    norm = integrate.quad(density, 0.0, maximum, limit=200)[0]

    def cdf(recoil: Any) -> Any:
        values = np.atleast_1d(np.asarray(recoil, dtype=np.float64))
        return np.asarray(
            [
                integrate.quad(density, 0.0, min(max(value, 0.0), maximum), limit=200)[0]
                / norm
                for value in values
            ]
        )

    return cdf


def test_klein_nishina_recoil_spectrum_kolmogorov_smirnov() -> None:
    count = 4096
    energy = 1.0e6
    plan = phx.solver.PhotonTransportPlan(
        _geometry(),
        _photon_library(photoelectric=0.0, compton=1.0e3),
        maximum_events=1,
        cutoff_energy=1.0e3,
        electron_stack=phx.solver.SecondaryStackSpec(1, minimum_energy=1.0),
    )
    result = plan.simulate(
        jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.5)), (count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3)),
        jnp.full((count,), energy),
        jr.key(21),
    )
    stack = result.secondary_electrons
    assert stack is not None
    active = np.asarray(stack.active[:, 0])
    recoil = np.asarray(stack.energies[:, 0])[active]
    assert recoil.size > 0.95 * count
    assert float(result.maximum_ledger_residual) < 1.0e-9 * energy
    statistic, p_value = stats.kstest(recoil, _klein_nishina_recoil_cdf(energy))
    assert p_value > 0.01, (statistic, p_value)
    directions = np.asarray(stack.directions[:, 0])[active]
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-12)
    # Momentum conservation off an electron at rest: forward recoil only.
    assert np.all(directions[:, 2] > 0.0)


def _beta_to_kinetic(beta: np.ndarray) -> np.ndarray:
    return _ELECTRON_REST_ENERGY_EV * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)


def test_two_generation_shower_ledger_and_secondary_counts() -> None:
    count = 1024
    plan = _shower()
    result = plan.simulate(_primary_photons(count, capacity=1024), None, jr.key(31))

    assert bool(result.successful)
    assert int(result.status) == int(phx.solver.EMShowerStatus.SUCCESS)
    assert int(result.refused_generation) == 2
    primary = float(result.primary_energy)
    np.testing.assert_allclose(primary, count * 2.0e5)
    assert abs(float(result.ledger_residual)) <= 1.0e-9 * primary
    closed = (
        float(result.deposited_energy)
        + float(result.escaped_energy)
        + float(result.truncated_energy)
        + float(result.stack_remainder_energy)
    )
    np.testing.assert_allclose(closed, primary, rtol=1e-12)
    assert float(result.truncated_energy) == 0.0

    # Generation 0: photoelectric absorption at rate 1/m along a 1 m slab.
    photoelectrons = int(result.generation_secondary_electron_count[0])
    expected = count * (1.0 - np.exp(-1.0))
    sigma = np.sqrt(count * (1.0 - np.exp(-1.0)) * np.exp(-1.0))
    assert abs(photoelectrons - expected) < 4.0 * sigma
    assert int(result.generation_charged_count[1]) == photoelectrons
    np.testing.assert_allclose(
        float(result.photon_to_charged_energy), photoelectrons * 2.0e5
    )

    # Generation 1: bremsstrahlung at the tabulated 2/m rate over the realized
    # step lengths, counting only photons at or above the 1 keV stack threshold.
    electrons = result.charged_generations[1]
    bank = electrons.step_bank
    assert bank is not None
    launched = np.asarray(result.charged_launches[1].active)
    assert bool(jnp.all(bank.complete[launched]))
    active = np.asarray(bank.active) & launched[:, None]
    lengths = np.linalg.norm(
        np.asarray(bank.end_positions) - np.asarray(bank.start_positions), axis=-1
    )[active]
    kinetic = _beta_to_kinetic(np.asarray(bank.start_beta)[active])
    probability = (1.0 - np.exp(-2.0 * lengths)) * (1.0 - 2.0e3 / kinetic)
    expected_photons = probability.sum()
    sigma_photons = np.sqrt((probability * (1.0 - probability)).sum())
    photons = int(result.generation_secondary_photon_count[1])
    assert abs(photons - expected_photons) < 4.0 * sigma_photons
    assert photons > 0
    remainder = result.photon_launches[2]
    assert int(remainder.count) == photons
    np.testing.assert_allclose(
        float(result.stack_remainder_energy),
        float(result.charged_to_photon_energy),
        rtol=1e-12,
    )
    # Secondary identities are unique and carry their parent's identity.
    child = (np.asarray(remainder.id_hi, dtype=np.uint64) << np.uint64(32)) | np.asarray(
        remainder.id_lo, dtype=np.uint64
    )
    child = child[np.asarray(remainder.active)]
    assert np.unique(child).size == child.size
    parents = (
        np.asarray(remainder.parent_hi, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(remainder.parent_lo, dtype=np.uint64)
    electron_ids = (
        np.asarray(result.charged_launches[1].id_hi, dtype=np.uint64) << np.uint64(32)
    ) | np.asarray(result.charged_launches[1].id_lo, dtype=np.uint64)
    assert np.all(np.isin(parents[np.asarray(remainder.active)], electron_ids))


def test_shower_results_are_invariant_to_primary_order() -> None:
    count = 48
    plan = _shower(photon_capacity=64, charged_capacity=64)
    identities = (
        jnp.zeros((count,), dtype=jnp.uint32),
        jnp.asarray(np.arange(100, 100 + count), dtype=jnp.uint32),
    )
    ordered = plan.simulate(
        _primary_photons(count, capacity=64, identities=identities), None, jr.key(41)
    )
    permutation = np.random.default_rng(5).permutation(count)
    shuffled = plan.simulate(
        _primary_photons(
            count,
            capacity=64,
            identities=(identities[0][permutation], identities[1][permutation]),
        ),
        None,
        jr.key(41),
    )

    assert bool(ordered.successful) and bool(shuffled.successful)
    np.testing.assert_array_equal(ordered.deposited_energy, shuffled.deposited_energy)
    np.testing.assert_array_equal(ordered.escaped_energy, shuffled.escaped_energy)
    np.testing.assert_array_equal(
        ordered.stack_remainder_energy, shuffled.stack_remainder_energy
    )
    np.testing.assert_array_equal(
        ordered.generation_secondary_photon_count,
        shuffled.generation_secondary_photon_count,
    )
    for first, second in zip(ordered.charged_launches, shuffled.charged_launches):
        np.testing.assert_array_equal(first.id_lo, second.id_lo)
        np.testing.assert_array_equal(first.energies, second.energies)
        np.testing.assert_array_equal(first.positions, second.positions)
    for first, second in zip(ordered.photon_launches, shuffled.photon_launches):
        np.testing.assert_array_equal(first.id_hi, second.id_hi)
        np.testing.assert_array_equal(first.id_lo, second.id_lo)
        np.testing.assert_array_equal(first.directions, second.directions)


def test_shower_capacity_refusal_is_atomic() -> None:
    count = 64
    plan = _shower(photon_capacity=64, charged_capacity=8)
    result = plan.simulate(_primary_photons(count, capacity=64), None, jr.key(51))

    assert not bool(result.successful)
    assert int(result.status) == int(phx.solver.EMShowerStatus.STACK_CAPACITY_EXHAUSTED)
    assert int(result.refused_generation) == 0
    refused = result.charged_launches[1]
    assert int(refused.requested_count) > 8
    assert not bool(jnp.any(refused.active))
    assert int(result.generation_charged_count[1]) == 0
    assert int(result.generation_photon_count[1]) == 0
    assert float(result.charged_to_photon_energy) == 0.0
    np.testing.assert_allclose(
        float(result.stack_remainder_energy), float(result.photon_to_charged_energy)
    )
    assert abs(float(result.ledger_residual)) <= 1.0e-9 * float(result.primary_energy)


def test_shower_refuses_duplicate_primary_identities() -> None:
    plan = _shower(photon_capacity=8, charged_capacity=8)
    identities = (
        jnp.zeros((3,), dtype=jnp.uint32),
        jnp.asarray((7, 7, 9), dtype=jnp.uint32),
    )
    result = plan.simulate(
        _primary_photons(3, capacity=8, identities=identities), None, jr.key(61)
    )

    assert int(result.status) == int(phx.solver.EMShowerStatus.INVALID_INPUT)
    assert int(result.generation_photon_count[0]) == 0
    assert float(result.primary_energy) == 0.0


def test_shower_plan_refuses_incompatible_transports() -> None:
    geometry = _geometry()
    library = _photon_library(photoelectric=1.0, compton=0.0)
    materials = _charged_materials(stopping=1.0, bremsstrahlung=1.0)
    without_stack = phx.solver.PhotonTransportPlan(
        geometry, library, maximum_events=4, cutoff_energy=1.0e3
    )
    charged = phx.solver.ChargedParticleTransportPlan(
        geometry,
        materials,
        maximum_steps=8,
        maximum_step_length=0.1,
        cutoff_energy_ev=10.0,
        photon_stack=phx.solver.SecondaryStackSpec(8, minimum_energy=1.0e3),
    )
    with pytest.raises(ValueError, match="secondary stack"):
        phx.solver.EMShowerPlan(
            without_stack,
            charged,
            photon_capacity=4,
            charged_capacity=4,
            maximum_generations=1,
        )
    low_threshold = phx.solver.PhotonTransportPlan(
        geometry,
        library,
        maximum_events=4,
        cutoff_energy=1.0e3,
        electron_stack=phx.solver.SecondaryStackSpec(1, minimum_energy=1.0),
    )
    with pytest.raises(ValueError, match="charged cutoff"):
        phx.solver.EMShowerPlan(
            low_threshold,
            charged,
            photon_capacity=4,
            charged_capacity=4,
            maximum_generations=1,
        )


def test_photon_secondary_stack_overflow_keeps_ledger() -> None:
    count = 256
    plan = phx.solver.PhotonTransportPlan(
        _geometry(),
        _photon_library(photoelectric=0.0, compton=50.0),
        maximum_events=8,
        cutoff_energy=1.0e3,
        electron_stack=phx.solver.SecondaryStackSpec(1, minimum_energy=1.0),
    )
    result = plan.simulate(
        jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.5)), (count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3)),
        jnp.full((count,), 5.0e5),
        jr.key(71),
    )
    stack = result.secondary_electrons
    assert stack is not None
    overflowed = np.asarray(stack.overflowed)
    assert overflowed.any()
    status = np.asarray(result.status)
    assert np.all(
        status[overflowed]
        == int(phx.solver.PhotonTransportStatus.SECONDARY_CAPACITY_EXHAUSTED)
    )
    assert float(result.maximum_ledger_residual) < 1.0e-9 * 5.0e5
    assert np.all(np.asarray(result.truncated_energy)[overflowed] > 0.0)
    np.testing.assert_allclose(
        np.asarray(stack.energy),
        np.where(np.asarray(stack.active[:, 0]), np.asarray(stack.energies[:, 0]), 0.0),
    )


def test_compton_profile_momentum_and_doppler_line() -> None:
    j0 = 1.2
    draws = jr.uniform(jr.key(81), (200_000,))
    momentum = np.asarray(
        phx.equations.sample_compton_profile_momentum(draws, jnp.asarray(j0))
    )
    # Closed-form cumulative of the one-parameter profile on the negative side.
    for threshold in (-0.5, -0.2, 0.3):
        magnitude = abs(threshold)
        tail = 0.5 * np.exp(0.5 * (1.0 - (1.0 + 2.0 * j0 * magnitude) ** 2))
        expected = tail if threshold < 0.0 else 1.0 - tail
        observed = np.mean(momentum <= threshold)
        assert abs(observed - expected) < 4.0 * np.sqrt(
            expected * (1.0 - expected) / draws.size
        )
    energy = jnp.asarray(5.0e5)
    cosine = jnp.asarray(0.3)
    line, valid = phx.equations.doppler_scattered_energy(
        energy, cosine, jnp.asarray(0.0), _ELECTRON_REST_ENERGY_EV
    )
    assert bool(valid)
    compton_line = 5.0e5 / (1.0 + 5.0e5 / _ELECTRON_REST_ENERGY_EV * (1.0 - 0.3))
    np.testing.assert_allclose(float(line), compton_line, rtol=1e-12)
    shifted, valid = phx.equations.doppler_scattered_energy(
        energy, cosine, jnp.asarray(20.0), _ELECTRON_REST_ENERGY_EV
    )
    assert bool(valid) and float(shifted) > compton_line


def test_impulse_approximation_kinematics_require_and_use_profile() -> None:
    geometry = _geometry()
    with pytest.raises(ValueError, match="compton_profile_j0"):
        phx.solver.PhotonTransportPlan(
            geometry,
            _photon_library(photoelectric=0.0, compton=1.0e3),
            maximum_events=1,
            cutoff_energy=1.0e3,
            compton_kinematics="impulse-approximation",
        )
    count = 512
    origins = jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.5)), (count, 3))
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3))
    energies = jnp.full((count,), 1.0e5)
    library = _photon_library(photoelectric=0.0, compton=1.0e3, compton_profile_j0=1.5)
    free = phx.solver.PhotonTransportPlan(
        geometry, library, maximum_events=1, cutoff_energy=1.0e3
    ).simulate(origins, directions, energies, jr.key(91))
    broadened = phx.solver.PhotonTransportPlan(
        geometry,
        library,
        maximum_events=1,
        cutoff_energy=1.0e3,
        compton_kinematics="impulse-approximation",
    ).simulate(origins, directions, energies, jr.key(91))

    # One event: scattered photons still in flight report event capacity, and
    # no history may fail its angular or Doppler sampling.
    allowed = (
        int(phx.solver.PhotonTransportStatus.SUCCESS),
        int(phx.solver.PhotonTransportStatus.EVENT_CAPACITY_EXHAUSTED),
    )
    assert np.all(np.isin(np.asarray(free.status), allowed))
    assert np.all(np.isin(np.asarray(broadened.status), allowed))
    assert float(broadened.maximum_ledger_residual) < 1.0e-9 * 1.0e5
    # Same angles (same identity-addressed draws), Doppler-shifted energies.
    np.testing.assert_array_equal(free.terminal_direction, broadened.terminal_direction)
    shift = np.asarray(broadened.terminal_energy) - np.asarray(free.terminal_energy)
    assert np.any(shift > 0.0) and np.any(shift < 0.0)
