#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned Geant4 shower, Cherenkov, and optical-transport oracles.

Independent references: CODATA 2022 ``m_e c² = 510998.95069 eV`` and the exact
SI ``hc/e``; the PDG radiation lengths of lead (5.612 mm) and copper
(14.36 mm) and Rossi critical energies for electrons (lead 7.43 MeV, copper
19.42 MeV); the Longo–Sestili gamma profile (PDG, "Passage of particles
through matter", Eq. 34.36, ``b ≈ 0.5``, ``t_max = ln(E/E_c) − 1/2``); the
Frank–Tamm yield and the Cherenkov cone ``cos θ = 1/(β n)`` with polarization
along ``p̂ × (p̂ × v̂)`` (Jackson, 3rd ed., § 13.5); a single-term Sellmeier
law of water; and the law of reflection and Snell's law. Reference outputs
under ``tests/data/providers/geant4`` were produced by the pinned provider;
their provenance is ``provenance.json``. The live oracles skip only when the
pinned interpreter is not configured; they compare Geant4 with the Phydrax
consumer routes within five combined standard errors.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax._external_runtime import pin_executable
from phydrax.applications.detector import (
    geant4_cherenkov_input,
    geant4_optical_input,
    geant4_shower_input,
    Geant4Provider,
    read_geant4_cherenkov,
    read_geant4_optical,
    read_geant4_shower,
    run_geant4_cherenkov,
    run_geant4_optical,
    run_geant4_shower,
)
from phydrax.interchange import AdapterStatus
from phydrax.optics.geometric import NonSequentialSurfaceKind, NonSequentialSurfaceTable
from phydrax.optics.transport import (
    ChargedOpticalSteps,
    CherenkovEmission,
    emit_optical_photons,
    ExplicitPhotonSource,
    HenyeyGreensteinScattering,
    launch_optical_photons,
    OpticalMonteCarloPlan,
    OpticalPhotonSourcePlan,
    OpticalPhotonState,
    OpticalSurfaceHit,
    prepare_optical_monte_carlo,
    RayleighScattering,
    ScintillationEmission,
    simulate_optical_photons,
    SpectralOpticalMedium,
    UnifiedSurfaceModel,
)


_DATA = Path(__file__).parents[2] / "data" / "providers" / "geant4"
_ELECTRON_REST_ENERGY_EV = 510998.95069
_HC_EV_M = 6.62607015e-34 * 299_792_458.0 / 1.602176634e-19
_C = 299_792_458.0
_EV_PER_JOULE = float(
    phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE)
)
_LEAD_RADIATION_LENGTH = 5.612e-3
_COPPER_RADIATION_LENGTH = 14.36e-3
_LEAD_CRITICAL_ENERGY_EV = 7.43e6
_COPPER_CRITICAL_ENERGY_EV = 19.42e6
_BAND = (300e-9, 600e-9)
_BETA = 0.8
_ULTRARELATIVISTIC_BETA = 1.0 - 1.3e-7


def _manifest(name: str) -> Any:
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum="e" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:geant4-oracle",),
    )


def _photon_table(name: str, grid: Any, material_ids: tuple[str, ...]) -> Any:
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name),
        f"https://example.invalid/{name}",
        "synthetic-geant4-oracle",
        "fixture",
        name,
    )
    unit = phx.units.derived_unit(
        "m2/kg", ((phx.units.METER, 2), (phx.units.KILOGRAM, -1))
    )
    return phx.equations.DiagnosticPhotonCoefficientTable(
        phx.equations.DiagnosticPhotonCoefficientRole.MASS_ATTENUATION,
        grid,
        material_ids,
        jnp.full((len(material_ids), 2), 0.01),
        unit,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def _shower_plan(
    *,
    material_ids: tuple[str, ...] = ("lead",),
    charged_material_ids: tuple[str, ...] | None = None,
    densities: tuple[float, ...] = (11350.0,),
    depth: float = 20.0 * _LEAD_RADIATION_LENGTH,
    materials: np.ndarray | None = None,
    photon_threshold: float = 1.0e5,
    charged_cutoff: float = 1.0e6,
    photon_cutoff: float = 1.0e4,
) -> Any:
    """Slab ``|x|, |y| ≤ 0.15 m``, ``0 ≤ z ≤ depth``; the tables are synthetic."""
    count = len(material_ids)
    grid = phx.equations.PhotonEnergyGrid(jnp.asarray((1.0e2, 2.0e9)) * _EV_PER_JOULE)
    library = phx.equations.RadiationCrossSectionLibrary(
        _photon_table("photoelectric", grid, material_ids),
        _photon_table("compton", grid, material_ids),
        _photon_table("rayleigh", grid, material_ids),
        jnp.asarray(densities),
    )
    charged_materials = phx.equations.ChargedRadiationMaterialLibrary(
        jnp.asarray((1.0e2, 2.0e9)),
        jnp.full((count, 2), 1.0e8),
        jnp.zeros((count, 2)),
        jnp.full((count, 2), 10.0),
        charged_material_ids or material_ids,
        _manifest("charged-material"),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.asarray((-0.15, -0.15, 0.0)),
        jnp.asarray((0.15, 0.15, depth)),
        jnp.zeros((1, 1, 4), dtype=jnp.int32) if materials is None else materials,
        material_count=count,
    )
    photons = phx.solver.PhotonTransportPlan(
        geometry,
        library,
        maximum_events=8,
        cutoff_energy=photon_cutoff,
        electron_stack=phx.solver.SecondaryStackSpec(4, minimum_energy=charged_cutoff),
    )
    charged = phx.solver.ChargedParticleTransportPlan(
        geometry,
        charged_materials,
        maximum_steps=64,
        maximum_step_length=1.0e-3,
        cutoff_energy_ev=charged_cutoff,
        photon_stack=phx.solver.SecondaryStackSpec(8, minimum_energy=photon_threshold),
    )
    return phx.solver.EMShowerPlan(
        photons, charged, photon_capacity=64, charged_capacity=128, maximum_generations=2
    )


def _electrons(count: int, energy: float = 1.0e9, *, capacity: int = 128) -> Any:
    return phx.solver.ShowerParticleBatch.charged(
        np.zeros((count, 3)),
        np.tile(np.asarray((0.0, 0.0, 1.0)), (count, 1)),
        np.full((count,), energy),
        np.zeros((count,), dtype=np.int32),
        capacity=capacity,
    )


def _water_index(wavelengths: np.ndarray) -> np.ndarray:
    """Single-term Sellmeier law of water (strength 0.758, resonance 100 nm)."""
    return np.sqrt(1.0 + 0.758 * wavelengths**2 / (wavelengths**2 - (100e-9) ** 2))


def _cherenkov_plan(nodes: int = 65) -> tuple[OpticalPhotonSourcePlan, np.ndarray]:
    wavelengths = np.linspace(*_BAND, nodes)
    medium = SpectralOpticalMedium(
        wavelengths,
        _water_index(wavelengths)[None],
        np.full((1, nodes), np.inf),
    )
    plan = OpticalPhotonSourcePlan(
        relativity=phx.RelativityScaleContract.si(),
        photon_capacity=200_000,
        cherenkov=CherenkovEmission(medium, wavelengths),
    )
    return plan, wavelengths


def _step(
    beta: float,
    length: float,
    *,
    parents: int = 1,
    charge: float = -1.0,
    end_beta: float | None = None,
    medium: int = 0,
) -> ChargedOpticalSteps:
    starts = np.zeros((parents, 1, 3))
    starts[..., 0] = 0.01
    ends = starts.copy()
    ends[..., 2] = length
    return ChargedOpticalSteps(
        starts,
        ends,
        np.full((parents, 1), 2.0e-9),
        np.full((parents, 1), beta),
        np.full((parents, 1), beta if end_beta is None else end_beta),
        np.full((parents, 1), medium, dtype=np.int32),
        np.ones((parents, 1), dtype=np.bool_),
        (np.zeros(parents, dtype=np.uint32), np.arange(parents, dtype=np.uint32)),
        speed_of_light=_C,
        charge_numbers=charge,
    )


def _deck(inputs: dict[str, bytes]) -> dict[str, Any]:
    return json.loads(inputs["input.json"])


def test_shower_deck_carries_plan_material_primaries_and_thresholds() -> None:
    plan = _shower_plan(photon_threshold=2.0e5, charged_cutoff=5.0e5)
    charged = phx.solver.ShowerParticleBatch.charged(
        np.asarray(((0.0, 0.0, 0.0), (0.01, -0.02, 0.0))),
        np.asarray(((0.0, 0.0, 1.0), (0.0, 0.6, 0.8))),
        np.asarray((1.0e9, 3.0e8)),
        np.asarray((1, 0), dtype=np.int32),
        capacity=4,
        identities=(np.asarray((0, 0)), np.asarray((7, 2))),
    )
    photons = phx.solver.ShowerParticleBatch.photons(
        np.asarray(((0.0, 0.0, 0.05),)),
        np.asarray(((1.0, 0.0, 0.0),)),
        np.asarray((5.0e8,)),
        capacity=2,
        identities=(np.asarray((0,)), np.asarray((4,))),
    )
    deck = _deck(
        geant4_shower_input(
            plan, photons, charged, nist_material="G4_Pb", depth_bins=40, random_seed=3
        )
    )
    assert deck["nist_material"] == "G4_Pb"
    assert deck["density_kg_m3"] == 11350.0
    assert deck["physics_constructor"] == "G4EmStandardPhysics_option4"
    assert deck["production_thresholds_ev"] == {"gamma": 2.0e5, "e-": 5.0e5, "e+": 5.0e5}
    assert deck["lower_m"] == [-0.15, -0.15, 0.0]
    np.testing.assert_allclose(deck["upper_m"], (0.15, 0.15, 20.0 * 5.612e-3))
    # One Geant4 event per active primary, in identity order (2, 4, 7).
    assert [primary["particle"] for primary in deck["primaries"]] == [
        "e-",
        "gamma",
        "e+",
    ]
    assert [primary["kinetic_energy_ev"] for primary in deck["primaries"]] == [
        3.0e8,
        5.0e8,
        1.0e9,
    ]
    assert deck["primaries"][0]["position_m"] == [0.01, -0.02, 0.0]
    assert deck["primaries"][0]["direction"] == [0.0, 0.6, 0.8]


@pytest.mark.parametrize(
    ("options", "message"),
    [
        (
            {
                "material_ids": ("lead", "copper"),
                "densities": (11350.0, 8960.0),
                "materials": np.asarray([[[0, 0, 1, 1]]], dtype=np.int32),
            },
            "homogeneous",
        ),
        ({"charged_material_ids": ("tungsten",)}, "identically"),
        (
            {"photon_cutoff": 500.0, "photon_threshold": 900.0, "charged_cutoff": 900.0},
            "990 eV",
        ),
    ],
    ids=("heterogeneous-geometry", "material-identity", "threshold-below-table"),
)
def test_shower_deck_refuses_unsupported_plans(
    options: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        geant4_shower_input(
            _shower_plan(**options),
            None,
            _electrons(1),
            nist_material="G4_Pb",
            depth_bins=8,
        )


def test_shower_deck_refuses_primaries_outside_the_geometry_and_unit_sphere() -> None:
    plan = _shower_plan()
    outside = phx.solver.ShowerParticleBatch.charged(
        np.asarray(((0.0, 0.0, -1.0e-3),)),
        np.asarray(((0.0, 0.0, 1.0),)),
        np.asarray((1.0e9,)),
        np.asarray((0,), dtype=np.int32),
        capacity=1,
    )
    with pytest.raises(ValueError, match="inside the plan geometry"):
        geant4_shower_input(plan, None, outside, nist_material="G4_Pb", depth_bins=8)
    skew = phx.solver.ShowerParticleBatch.charged(
        np.zeros((1, 3)),
        np.asarray(((0.0, 0.0, 2.0),)),
        np.asarray((1.0e9,)),
        np.asarray((0,), dtype=np.int32),
        capacity=1,
    )
    with pytest.raises(ValueError, match="unit directions"):
        geant4_shower_input(plan, None, skew, nist_material="G4_Pb", depth_bins=8)
    with pytest.raises(ValueError, match="at least one active primary"):
        geant4_shower_input(plan, None, None, nist_material="G4_Pb", depth_bins=8)
    with pytest.raises(ValueError, match="physics"):
        geant4_shower_input(
            plan,
            None,
            _electrons(1),
            nist_material="G4_Pb",
            depth_bins=8,
            physics="qgsp",  # ty: ignore[invalid-argument-type]
        )


def test_cherenkov_deck_converts_speed_and_dispersion() -> None:
    plan, wavelengths = _cherenkov_plan(17)
    deck = _deck(
        geant4_cherenkov_input(
            plan, _step(_BETA, 2.0e-3, charge=1.0), nist_material="G4_WATER", events=3
        )
    )
    gamma = 1.0 / math.sqrt(1.0 - _BETA**2)
    assert deck["particle"] == "e+"
    # The package electron mass and the published m_e c² agree to CODATA precision.
    np.testing.assert_allclose(
        deck["kinetic_energy_ev"], (gamma - 1.0) * _ELECTRON_REST_ENERGY_EV, rtol=1e-9
    )
    # Geant4 tables ascend in photon energy E = hc / λ.
    np.testing.assert_allclose(
        deck["photon_energies_ev"], _HC_EV_M / wavelengths[::-1], rtol=1e-12
    )
    np.testing.assert_allclose(
        deck["refractive_indices"], _water_index(wavelengths)[::-1], rtol=1e-12
    )
    np.testing.assert_allclose(deck["start_m"], (0.01, 0.0, 0.0))
    np.testing.assert_allclose(deck["direction"], (0.0, 0.0, 1.0))
    assert deck["path_length_m"] == 2.0e-3
    assert deck["start_time_s"] == 2.0e-9


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        (_step(_BETA, 1e-3, parents=2), "one parent with one step"),
        (_step(_BETA, 1e-3, charge=2.0), "singly charged"),
        (_step(_BETA, 1e-3, end_beta=0.7), "constant speed"),
    ],
    ids=("two-parents", "doubly-charged", "decelerating"),
)
def test_cherenkov_deck_refuses_unsupported_steps(
    steps: ChargedOpticalSteps, message: str
) -> None:
    plan, _ = _cherenkov_plan(9)
    with pytest.raises(ValueError, match=message):
        geant4_cherenkov_input(plan, steps, nist_material="G4_WATER", events=1)


def test_cherenkov_deck_refuses_scintillation() -> None:
    plan, wavelengths = _cherenkov_plan(9)
    scintillating = OpticalPhotonSourcePlan(
        relativity=phx.RelativityScaleContract.si(),
        photon_capacity=16,
        cherenkov=plan.cherenkov,
        scintillation=ScintillationEmission(
            np.asarray((1.0e-3,)),
            np.asarray((0.0,)),
            np.asarray(((1.0,),)),
            np.asarray(((0.0,),)),
            np.asarray(((2.0e-9,),)),
            wavelengths,
            np.ones((1, 1, wavelengths.size)),
        ),
    )
    with pytest.raises(ValueError, match="Cherenkov emission only"):
        geant4_cherenkov_input(
            scintillating, _step(_BETA, 1e-3), nist_material="G4_WATER", events=1
        )


def _reference_executable() -> Any:
    provenance = json.loads((_DATA / "provenance.json").read_text())
    return pin_executable(
        sys.executable,
        version=provenance["geant4_pybind"],
        license_id="LicenseRef-Geant4",
    )


def _reference_shower() -> tuple[Any, dict[str, bytes]]:
    plan = _shower_plan()
    inputs = geant4_shower_input(
        plan, None, _electrons(20), nist_material="G4_Pb", depth_bins=20, random_seed=7
    )
    return plan, inputs


def test_shower_reader_imports_reference_provider_output() -> None:
    _, inputs = _reference_shower()
    assert inputs["input.json"] == (_DATA / "shower" / "input.json").read_bytes()
    result = read_geant4_shower(
        inputs,
        (_DATA / "shower" / "summary.json").read_bytes(),
        (_DATA / "shower" / "deposits.npy").read_bytes(),
        executable=_reference_executable(),
    )
    np.testing.assert_allclose(result.radiation_length, _LEAD_RADIATION_LENGTH, rtol=2e-3)
    np.testing.assert_allclose(result.mass_density, 11350.0, rtol=1e-12)
    assert result.deposited_energy.shape == (20, 20)
    np.testing.assert_array_equal(result.primary_energies, np.full(20, 1.0e9))
    # A slab can only retain part of each primary's energy.
    assert np.all(result.total_deposited_energy > 0.5e9)
    assert np.all(result.total_deposited_energy <= 1.0e9)
    np.testing.assert_allclose(result.depth_edges_radiation_lengths[-1], 20.0, rtol=2e-3)
    assert dict(result.production_thresholds) == pytest.approx(
        {"gamma": 1.0e5, "e-": 1.0e6, "e+": 1.0e6}, rel=1e-6
    )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.report.source_id == result.output_sha256
    assert {loss.path for loss in result.report.losses} >= {
        "physics.hadronic",
        "cuts.production_thresholds",
        "plan.maximum_generations",
    }


def test_readers_refuse_a_different_pinned_release() -> None:
    _, inputs = _reference_shower()
    stale = pin_executable(
        sys.executable, version="0.0.0", license_id="LicenseRef-Geant4"
    )
    with pytest.raises(ValueError, match="geant4_pybind"):
        read_geant4_shower(
            inputs,
            (_DATA / "shower" / "summary.json").read_bytes(),
            (_DATA / "shower" / "deposits.npy").read_bytes(),
            executable=stale,
        )


def _reference_cherenkov_inputs() -> dict[str, bytes]:
    plan, _ = _cherenkov_plan()
    return geant4_cherenkov_input(
        plan,
        _step(_ULTRARELATIVISTIC_BETA, 2.0e-3),
        nist_material="G4_WATER",
        events=1,
        random_seed=5,
    )


def _assert_cherenkov_kinematics(result: Any) -> None:
    """Cone ``cos θ = 1/(β̄ n(λ))`` and polarization along ``p̂ × (p̂ × v̂)``."""
    beta = 0.5 * (result.parent_start_beta + result.parent_end_beta)
    np.testing.assert_allclose(
        result.emission_cosines,
        1.0 / (beta * _water_index(result.wavelengths)),
        atol=2e-5,
    )
    assert np.all((result.wavelengths >= _BAND[0]) & (result.wavelengths <= _BAND[1]))
    expected = np.cross(
        result.directions, np.cross(result.directions, result.parent_directions)
    )
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    np.testing.assert_allclose(
        np.abs(np.sum(result.polarizations * expected, axis=1)), 1.0, atol=1e-9
    )


def test_cherenkov_reader_imports_reference_provider_output() -> None:
    inputs = _reference_cherenkov_inputs()
    assert inputs["input.json"] == (_DATA / "cherenkov" / "input.json").read_bytes()
    result = read_geant4_cherenkov(
        inputs,
        (_DATA / "cherenkov" / "summary.json").read_bytes(),
        (_DATA / "cherenkov" / "photons.npy").read_bytes(),
        executable=_reference_executable(),
    )
    assert result.photon_counts.shape == (1,)
    assert result.photon_counts[0] == result.wavelengths.shape[0] > 0
    _assert_cherenkov_kinematics(result)
    # Photons are emitted along the primary path from the step start.
    assert np.all(result.times >= 2.0e-9)
    assert np.all(np.linalg.norm(result.positions - (0.01, 0.0, 0.0), axis=1) <= 2.0e-3)
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.report.source_id == result.output_sha256


def _provider() -> Geant4Provider:
    names = (
        "PHYDRAX_GEANT4_PYTHON",
        "PHYDRAX_GEANT4_PYTHON_VERSION",
        "PHYDRAX_GEANT4_DATA",
    )
    if any(name not in os.environ for name in names):
        pytest.skip(
            "set PHYDRAX_GEANT4_PYTHON, PHYDRAX_GEANT4_PYTHON_VERSION, and "
            "PHYDRAX_GEANT4_DATA to a pinned geant4_pybind interpreter and its datasets"
        )
    return Geant4Provider(
        pin_executable(
            os.environ["PHYDRAX_GEANT4_PYTHON"],
            version=os.environ["PHYDRAX_GEANT4_PYTHON_VERSION"],
            license_id="LicenseRef-Geant4",
            source_url="https://github.com/HaarigerHarald/geant4_pybind",
        ),
        os.environ["PHYDRAX_GEANT4_DATA"],
    )


def _mean_depth(result: Any) -> float:
    centers, profile, _ = result.longitudinal_profile()
    return float(np.sum(centers * profile) / np.sum(profile))


def _longo_mean_depth(critical_energy: float, depth: float) -> float:
    """Mean of the Longo profile truncated to the slab, by the Phydrax M1 owner."""
    t = np.linspace(1e-6, depth, 20001)
    profile = np.asarray(
        phx.equations.longo_shower_profile(t, 1.0e9, critical_energy, incident="electron")
    )
    return float(np.trapezoid(t * profile, t) / np.trapezoid(profile, t))


def test_live_shower_depth_follows_longo_profile_and_radiation_length_scaling(
    tmp_path: Path,
) -> None:
    provider = _provider()
    depth = 20.0
    means = {}
    for name, nist, x0, critical in (
        ("lead", "G4_Pb", _LEAD_RADIATION_LENGTH, _LEAD_CRITICAL_ENERGY_EV),
        ("copper", "G4_Cu", _COPPER_RADIATION_LENGTH, _COPPER_CRITICAL_ENERGY_EV),
    ):
        destination = tmp_path / name
        destination.mkdir()
        plan = _shower_plan(
            material_ids=(name,),
            densities=(11350.0 if name == "lead" else 8960.0,),
            depth=depth * x0,
        )
        result = run_geant4_shower(
            provider,
            plan,
            None,
            _electrons(100),
            destination,
            nist_material=nist,
            depth_bins=40,
            random_seed=11,
            timeout=900.0,
        )
        np.testing.assert_allclose(result.radiation_length, x0, rtol=5e-3)
        contained = result.total_deposited_energy / result.primary_energies
        # Energy conservation up to Geant4's sub-eV bookkeeping of step deposits.
        assert np.all(contained <= 1.0 + 1e-9)
        assert np.mean(contained) > 0.9
        means[name] = (_mean_depth(result), _longo_mean_depth(critical, depth))
    # The PDG gamma parameterization is accurate to about ten percent in depth.
    for observed, longo in means.values():
        assert abs(observed - longo) < 0.1 * longo
    # X0 scaling: in X0 units the shower shifts only by ln(E_c,Cu / E_c,Pb).
    shift = means["lead"][0] - means["copper"][0]
    expected_shift = math.log(_COPPER_CRITICAL_ENERGY_EV / _LEAD_CRITICAL_ENERGY_EV)
    assert abs(shift - expected_shift) < 0.35


def test_live_cherenkov_yield_cone_and_spectrum_match_frank_tamm_source(
    tmp_path: Path,
) -> None:
    provider = _provider()
    plan, _ = _cherenkov_plan()
    length, events = 1.0e-2, 100
    steps = _step(_ULTRARELATIVISTIC_BETA, length)
    result = run_geant4_cherenkov(
        provider,
        plan,
        steps,
        tmp_path,
        nist_material="G4_WATER",
        events=events,
        random_seed=13,
        timeout=900.0,
    )
    emission = emit_optical_photons(plan, steps, jr.key(0))
    assert bool(emission.successful)
    phydrax_yield = float(np.asarray(emission.cherenkov_expected)[0]) / length
    counts = result.photon_counts
    geant4_yield = float(np.sum(counts) / np.sum(result.path_lengths))
    standard_error = math.sqrt(float(np.sum(counts))) / float(np.sum(result.path_lengths))
    assert abs(geant4_yield - phydrax_yield) < 5.0 * standard_error
    _assert_cherenkov_kinematics(result)
    # Spectrum: the mean inverse wavelength of both photon samples agrees.
    active = np.asarray(emission.active)
    phydrax_inverse = 1.0 / np.asarray(emission.state.wavelengths)[active]
    geant4_inverse = 1.0 / result.wavelengths
    tolerance = 5.0 * math.hypot(
        np.std(phydrax_inverse) / math.sqrt(phydrax_inverse.size),
        np.std(geant4_inverse) / math.sqrt(geant4_inverse.size),
    )
    assert abs(np.mean(geant4_inverse) - np.mean(phydrax_inverse)) < tolerance


_GLASS_INDEX = 1.5
_SLAB = 0.01
_HALF_WIDTH = 10.0
_OPTICAL_GRID = np.asarray((400e-9, 500e-9, 600e-9))
_OPTICAL_WAVELENGTH = 500e-9
_GLASS_ABSORPTION_LENGTH = 0.05
_CRITICAL_ANGLE = math.asin(1.0 / _GLASS_INDEX)
_REFERENCE_ANGLES = np.asarray((0.3, 0.6, 0.9))
_DETECTOR = int(NonSequentialSurfaceKind.DETECTOR)
_DIELECTRIC = int(NonSequentialSurfaceKind.DIELECTRIC)


def _stack(
    kinds: tuple[int, int, int] = (_DETECTOR, _DIELECTRIC, _DETECTOR),
    *,
    tilt: float = 0.0,
) -> NonSequentialSurfaceTable:
    """Detector plane ``z = −d`` | glass (0) | ``z = 0`` | air (1) | ``z = +d``.

    Outer medium 2 lies beyond the detector planes; ``tilt`` slopes the middle
    plane in ``x``.
    """
    vertices = []
    triangles = []
    detectors = []
    for plane, (height, kind) in enumerate(zip((-_SLAB, 0.0, _SLAB), kinds, strict=True)):
        base = len(vertices)
        for x, y in ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)):
            slope = tilt * x if plane == 1 else 0.0
            vertices.append((x * _HALF_WIDTH, y * _HALF_WIDTH, height + slope))
        triangles += [(base, base + 1, base + 2), (base, base + 2, base + 3)]
        detectors.append(sum(value == _DETECTOR for value in kinds[:plane]))
    kind_array = np.repeat(np.asarray(kinds, dtype=np.int32), 2)
    return NonSequentialSurfaceTable(
        np.asarray(vertices),
        np.asarray(triangles),
        np.repeat(np.asarray((2, 0, 1), dtype=np.int32), 2),
        np.repeat(np.asarray((0, 1, 2), dtype=np.int32), 2),
        np.asarray((_GLASS_INDEX, 1.0, 1.0)),
        surface_ids=np.repeat(np.arange(3, dtype=np.int32), 2),
        surface_kinds=kind_array,
        detector_indices=np.where(
            kind_array == _DETECTOR, np.repeat(np.asarray(detectors), 2), -1
        ).astype(np.int32),
    )


def _optical_medium(
    *,
    rayleigh_length: float | None = None,
    glass_absorption: Any = _GLASS_ABSORPTION_LENGTH,
    anisotropic: bool = False,
) -> SpectralOpticalMedium:
    index = np.ones((3, 3))
    index[0] = _GLASS_INDEX
    absorption = np.full((3, 3), np.inf)
    absorption[0] = glass_absorption
    rayleigh = None
    if rayleigh_length is not None:
        lengths = np.full((3, 3), np.inf)
        lengths[1] = rayleigh_length
        rayleigh = RayleighScattering(lengths)
    return SpectralOpticalMedium(
        _OPTICAL_GRID,
        index,
        absorption,
        rayleigh=rayleigh,
        henyey_greenstein=(
            HenyeyGreensteinScattering(np.ones((3, 3)), np.full((3, 3), 0.5))
            if anisotropic
            else None
        ),
    )


def _optical_plan(
    *,
    surfaces: NonSequentialSurfaceTable | None = None,
    medium: SpectralOpticalMedium | None = None,
    model: UnifiedSurfaceModel | None = None,
    maximum_interactions: int = 16,
) -> OpticalMonteCarloPlan:
    return OpticalMonteCarloPlan(
        _stack() if surfaces is None else surfaces,
        _optical_medium() if medium is None else medium,
        relativity=phx.RelativityScaleContract.si(),
        maximum_interactions=maximum_interactions,
        surface_model=UnifiedSurfaceModel(["polished"] * 3) if model is None else model,
    )


def _slab_photons(
    angles: np.ndarray,
    count: int,
    *,
    height: float = -0.5 * _SLAB,
    downward: bool = False,
    jones: tuple[complex, complex] = (1.0, 1.0),
    weight: float = 1.0,
) -> OpticalPhotonState:
    """Photons in the ``x–z`` plane at angle ``θ`` to ``±z``; ``s = −ŷ``."""
    theta = np.repeat(angles, count)
    sign = -1.0 if downward else 1.0
    directions = np.stack(
        (np.sin(theta), np.zeros_like(theta), sign * np.cos(theta)), axis=1
    )
    positions = np.zeros_like(directions)
    positions[:, 2] = height
    photons = theta.shape[0]
    return launch_optical_photons(
        positions,
        directions,
        np.full(photons, 1 if height > 0.0 else 0, dtype=np.int32),
        wavelengths=_OPTICAL_WAVELENGTH,
        jones_vectors=np.tile(np.asarray(jones, dtype=np.complex128), (photons, 1)),
        transverse_axes=np.tile((0.0, -1.0, 0.0), (photons, 1)),
        weights=weight,
    )


def test_optical_deck_carries_stack_media_and_linear_polarization() -> None:
    theta = 0.3
    deck = _deck(
        geant4_optical_input(_optical_plan(), _slab_photons(np.asarray((theta,)), 2))
    )
    assert [plane["z_m"] for plane in deck["planes"]] == [-_SLAB, 0.0, _SLAB]
    assert [layer["medium"] for layer in deck["layers"]] == [0, 1]
    assert deck["rectangle_m"] == [-_HALF_WIDTH, _HALF_WIDTH, -_HALF_WIDTH, _HALF_WIDTH]
    glass, air = deck["media"]["0"], deck["media"]["1"]
    np.testing.assert_allclose(
        glass["photon_energies_ev"], _HC_EV_M / _OPTICAL_GRID[::-1], rtol=1e-12
    )
    assert glass["refractive_indices"] == [_GLASS_INDEX] * 3
    np.testing.assert_allclose(glass["ABSLENGTH"], _GLASS_ABSORPTION_LENGTH, rtol=1e-12)
    assert glass["RAYLEIGH"] is None
    assert air["ABSLENGTH"] is None
    photon = deck["photons"][0]
    np.testing.assert_allclose(
        photon["energy_ev"], _HC_EV_M / _OPTICAL_WAVELENGTH, rtol=1e-12
    )
    np.testing.assert_allclose(
        photon["direction"], (math.sin(theta), 0.0, math.cos(theta))
    )
    # Jones (1, 1)/√2 on (s, d × s) with s = −ŷ and d × s = (cos θ, 0, −sin θ).
    np.testing.assert_allclose(
        photon["polarization"],
        np.asarray((math.cos(theta), -1.0, -math.sin(theta))) / math.sqrt(2.0),
        atol=1e-12,
    )


_ONE_ANGLE = np.asarray((0.3,))


@pytest.mark.parametrize(
    ("case", "message"),
    [
        (
            lambda: (
                OpticalMonteCarloPlan(
                    _stack(),
                    _optical_medium(),
                    relativity=phx.RelativityScaleContract.si(),
                    maximum_interactions=4,
                ),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "UNIFIED",
        ),
        (
            lambda: (
                _optical_plan(
                    model=UnifiedSurfaceModel(
                        ["polished", "ground", "polished"], sigma_alpha=(0.0, 0.1, 0.0)
                    )
                ),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "polished",
        ),
        (
            lambda: (
                _optical_plan(medium=_optical_medium(anisotropic=True)),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "Rayleigh scattering only",
        ),
        (
            lambda: (
                _optical_plan(surfaces=_stack(tilt=1e-4)),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "normal to z",
        ),
        (
            lambda: (
                _optical_plan(surfaces=_stack((_DETECTOR, _DETECTOR, _DETECTOR))),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "outermost",
        ),
        (
            lambda: (
                _optical_plan(
                    medium=_optical_medium(
                        glass_absorption=np.asarray((0.05, np.inf, 0.05))
                    )
                ),
                _slab_photons(_ONE_ANGLE, 1),
            ),
            "every wavelength node",
        ),
        (
            lambda: (_optical_plan(), _slab_photons(_ONE_ANGLE, 1, jones=(1.0, 1.0j))),
            "linear polarization",
        ),
        (
            lambda: (_optical_plan(), _slab_photons(_ONE_ANGLE, 1, weight=0.5)),
            "weights must be one",
        ),
        (
            lambda: (_optical_plan(), _slab_photons(_ONE_ANGLE, 1, height=-2.0 * _SLAB)),
            "strictly inside",
        ),
    ],
    ids=(
        "scalar-fresnel-model",
        "ground-interface",
        "henyey-greenstein-medium",
        "tilted-interface",
        "inner-detector",
        "partly-transparent-absorber",
        "circular-polarization",
        "weighted-photons",
        "photon-outside-stack",
    ),
)
def test_optical_deck_refuses_unsupported_plans_and_photons(
    case: Any, message: str
) -> None:
    plan, photons = case()
    with pytest.raises(ValueError, match=message):
        geant4_optical_input(plan, photons)


def _reference_optical_inputs() -> dict[str, bytes]:
    return geant4_optical_input(
        _optical_plan(), _slab_photons(_REFERENCE_ANGLES, 60), random_seed=17
    )


def test_optical_reader_imports_reference_provider_output() -> None:
    inputs = _reference_optical_inputs()
    assert inputs["input.json"] == (_DATA / "optical" / "input.json").read_bytes()
    result = read_geant4_optical(
        inputs,
        (_DATA / "optical" / "summary.json").read_bytes(),
        (_DATA / "optical" / "outcomes.npy").read_bytes(),
        executable=_reference_executable(),
    )
    # Every photon ends in exactly one tally; only the glass absorbs.
    total = np.sum(result.detector, axis=1) + np.sum(result.absorption, axis=1)
    np.testing.assert_array_equal(total + result.escape, 1.0)
    np.testing.assert_array_equal(result.absorption[:, 1:], 0.0)
    theta = np.repeat(_REFERENCE_ANGLES, 60)
    reflected = result.detector[:, 0] == 1.0
    transmitted = result.detector[:, 1] == 1.0
    assert not np.any(transmitted & (theta > _CRITICAL_ANGLE))
    # Law of reflection and Snell's law.
    np.testing.assert_allclose(
        result.exit_directions[reflected],
        np.stack((np.sin(theta), 0.0 * theta, -np.cos(theta)), axis=1)[reflected],
        atol=1e-9,
    )
    sine = _GLASS_INDEX * np.sin(theta[transmitted])
    np.testing.assert_allclose(
        result.exit_directions[transmitted],
        np.stack((sine, 0.0 * sine, np.sqrt(1.0 - sine**2)), axis=1),
        atol=1e-9,
    )
    # Time of flight n L / c through the glass (L = d/2 + d along cos θ).
    np.testing.assert_allclose(
        result.exit_times[reflected],
        _GLASS_INDEX * 1.5 * _SLAB / (_C * np.cos(theta[reflected])),
        rtol=1e-9,
    )
    detected = reflected | transmitted
    np.testing.assert_allclose(
        np.linalg.norm(result.exit_polarizations[detected], axis=1), 1.0, atol=1e-9
    )
    np.testing.assert_allclose(
        np.sum(result.exit_polarizations * result.exit_directions, axis=1)[detected],
        0.0,
        atol=1e-9,
    )
    assert result.report.status == AdapterStatus.DECLARED_LOSS
    assert result.report.source_id == result.output_sha256
    assert {loss.path for loss in result.report.losses} >= {
        "geometry.lateral",
        "polarization.phase",
    }


def _assert_same_rate(geant4: np.ndarray, phydrax: np.ndarray) -> None:
    """Equal means of per-photon tallies within five combined standard errors."""
    error = math.hypot(
        float(np.std(geant4)) / math.sqrt(geant4.size),
        float(np.std(phydrax)) / math.sqrt(phydrax.size),
    )
    assert abs(float(np.mean(geant4)) - float(np.mean(phydrax))) <= 5.0 * error + 1e-12


def _phydrax_tallies(plan: OpticalMonteCarloPlan, photons: OpticalPhotonState) -> Any:
    result = simulate_optical_photons(
        prepare_optical_monte_carlo(plan), ExplicitPhotonSource(photons), jr.key(3)
    )
    assert bool(result.all_successful)
    return result.per_photon_tallies


def _field(jones: np.ndarray, direction: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Real unit field of a linear Jones vector on ``(axis, direction × axis)``."""
    reference = jones[np.argmax(np.abs(jones))]
    aligned = (jones * np.conj(reference) / abs(reference)).real
    field = aligned[0] * axis + aligned[1] * np.cross(direction, axis)
    return field / np.linalg.norm(field)


def _polished_fields(theta: float) -> tuple[np.ndarray, np.ndarray]:
    """Reflected and transmitted fields of the M2 UNIFIED polished boundary."""
    direction = np.asarray((math.sin(theta), 0.0, math.cos(theta)))
    hit = OpticalSurfaceHit(
        jnp.asarray(direction[None]),
        jnp.asarray(((0.0, 0.0, 1.0),)),
        jnp.asarray((_GLASS_INDEX,)),
        jnp.asarray((1.0,)),
        jnp.asarray((_DIELECTRIC,), dtype=jnp.int32),
        jnp.asarray((_OPTICAL_WAVELENGTH,)),
        jnp.asarray(((1.0, 1.0),), dtype=jnp.complex128) / math.sqrt(2.0),
        jnp.asarray(((0.0, -1.0, 0.0),)),
        jnp.asarray((1,), dtype=jnp.int32),
    )
    response = UnifiedSurfaceModel(["polished"] * 3).interact(hit, jr.split(jr.key(0), 1))
    return (
        _field(
            np.asarray(response.reflected_jones)[0],
            np.asarray(response.reflected_directions)[0],
            np.asarray(response.reflected_axes)[0],
        ),
        _field(
            np.asarray(response.transmitted_jones)[0],
            np.asarray(response.transmitted_directions)[0],
            np.asarray(response.transmitted_axes)[0],
        ),
    )


def test_live_fresnel_total_internal_reflection_and_beer_lambert_match_optical_monte_carlo(
    tmp_path: Path,
) -> None:
    provider = _provider()
    angles = np.asarray((0.3, 0.6, 0.72, 0.9))
    count = 3000
    plan = _optical_plan()
    photons = _slab_photons(angles, count)
    result = run_geant4_optical(provider, plan, photons, tmp_path, random_seed=19)
    tallies = _phydrax_tallies(plan, photons)
    detector = np.asarray(tallies.detector)
    absorption = np.asarray(tallies.absorption)
    np.testing.assert_array_equal(result.escape, 0.0)
    for index, theta in enumerate(angles):
        rows = slice(index * count, (index + 1) * count)
        _assert_same_rate(result.detector[rows, 0], detector[rows, 0])
        _assert_same_rate(result.detector[rows, 1], detector[rows, 1])
        _assert_same_rate(result.absorption[rows, 0], absorption[rows, 0])
        if theta >= _CRITICAL_ANGLE:
            # Total internal reflection: nothing reaches the air side.
            np.testing.assert_array_equal(result.detector[rows, 1], 0.0)
            continue
        reflected_field, transmitted_field = _polished_fields(float(theta))
        # Geant4 reverses the in-plane component at each dielectric crossing
        # (declared "polarization.in_plane_sign"), so the s and p magnitudes,
        # i.e. |r_p/r_s| and |t_p/t_s|, are compared.
        s_axis = np.asarray((0.0, -1.0, 0.0))
        for column, field in ((0, reflected_field), (1, transmitted_field)):
            exits = result.detector[rows, column] == 1.0
            directions = result.exit_directions[rows][exits]
            polarizations = result.exit_polarizations[rows][exits]
            p_axes = np.cross(directions, s_axis)
            np.testing.assert_allclose(
                np.abs(polarizations @ s_axis), abs(field @ s_axis), atol=1e-9
            )
            np.testing.assert_allclose(
                np.abs(np.sum(polarizations * p_axes, axis=1)),
                abs(float(field @ p_axes[0])),
                atol=1e-9,
            )


def test_live_rayleigh_scattering_and_interfaces_match_optical_monte_carlo(
    tmp_path: Path,
) -> None:
    provider = _provider()
    plan = _optical_plan(
        medium=_optical_medium(rayleigh_length=_SLAB), maximum_interactions=64
    )
    photons = _slab_photons(np.asarray((0.0,)), 4000, height=0.5 * _SLAB, downward=True)
    result = run_geant4_optical(provider, plan, photons, tmp_path, random_seed=23)
    tallies = _phydrax_tallies(plan, photons)
    assert np.mean(result.rayleigh_scatters) > 0.1
    np.testing.assert_array_equal(result.escape, 0.0)
    detector = np.asarray(tallies.detector)
    _assert_same_rate(result.detector[:, 0], detector[:, 0])
    _assert_same_rate(result.detector[:, 1], detector[:, 1])
    _assert_same_rate(result.absorption[:, 0], np.asarray(tallies.absorption)[:, 0])
