#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


_DEFAULT_MU = np.log(2.0)


def _manifest(name="cross-sections", *, commercial_use_permitted=True):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum="c" * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"energy_eV": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=("synthetic:photon",),
    )


def _photon_table(
    name,
    energy_grid,
    values,
    *,
    commercial_use_permitted=True,
    interpolation=phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
):
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name, commercial_use_permitted=commercial_use_permitted),
        f"https://example.invalid/{name}",
        "synthetic-photon-test",
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
        interpolation,
    )


def _transport(mu=_DEFAULT_MU):
    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((500.0, 1500.0))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    photoelectric = _photon_table("photoelectric", energy_grid, (mu, mu))
    compton = _photon_table("compton", energy_grid, (0.0, 0.0))
    rayleigh = _photon_table("rayleigh", energy_grid, (0.0, 0.0))
    library = phx.equations.RadiationCrossSectionLibrary(
        photoelectric, compton, rayleigh, jnp.asarray((1.0,))
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.asarray((0.0, 0.0, 0.0)),
        jnp.asarray((1.0, 1.0, 1.0)),
        jnp.zeros((1, 1, 1), dtype=jnp.int32),
        material_count=1,
    )
    return phx.solver.PhotonTransportPlan(
        geometry, library, maximum_events=32, cutoff_energy=500.0
    )


def test_prepared_cross_sections_retain_sources_and_enforce_requested_rights():
    plan = _transport()
    assert len(set(plan.cross_sections.source_table_ids)) == 3
    assert len(plan.cross_sections.source_provenance_ids) == 3

    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((500.0, 1500.0))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    denied = tuple(
        _photon_table(
            name,
            energy_grid,
            (0.1, 0.1),
            commercial_use_permitted=False,
        )
        for name in ("denied-photoelectric", "denied-compton", "denied-rayleigh")
    )
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        phx.equations.RadiationCrossSectionLibrary(
            *denied, jnp.asarray((1.0,)), commercial_use=True
        )


def test_prepared_cross_sections_preserve_each_source_interpolation_policy():
    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((500.0, 2000.0))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    library = phx.equations.RadiationCrossSectionLibrary(
        _photon_table(
            "log-photoelectric",
            energy_grid,
            (1.0, 16.0),
            interpolation=phx.equations.DiagnosticPhotonInterpolationPolicy.LOG_LOG,
        ),
        _photon_table("linear-compton", energy_grid, (0.0, 0.0)),
        _photon_table("linear-rayleigh", energy_grid, (0.0, 0.0)),
        jnp.asarray((1.0,)),
    )

    evaluated = library.evaluate(0, 1000.0)

    assert bool(evaluated.successful)
    np.testing.assert_allclose(evaluated.coefficients, (4.0, 0.0, 0.0))


def test_delta_tracking_matches_beer_lambert_and_closes_kerma_ledger():
    count = 8192
    plan = _transport()
    result = plan.simulate(
        jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.0)), (count, 3)),
        jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (count, 3)),
        jnp.full((count,), 1000.0),
        jr.key(12),
    )

    assert bool(result.all_successful)
    np.testing.assert_allclose(result.mean_material_kerma[0], 500.0, atol=20.0)
    np.testing.assert_allclose(result.mean_escaped_energy, 500.0, atol=20.0)
    assert float(result.maximum_ledger_residual) < 1.0e-9
    assert jnp.all(
        (result.material_kerma[:, 0] == 0.0) | (result.material_kerma[:, 0] == 1000.0)
    )


def test_semantic_history_ids_are_invariant_to_batch_placement():
    plan = _transport(mu=0.5)
    origins = jnp.broadcast_to(jnp.asarray((0.5, 0.5, 0.0)), (4, 3))
    directions = jnp.broadcast_to(jnp.asarray((0.0, 0.0, 1.0)), (4, 3))
    energies = jnp.full((4,), 1000.0)
    ids = jnp.asarray((11, 12, 13, 14), dtype=jnp.uint32)
    together = plan.simulate(origins, directions, energies, jr.key(9), history_ids=ids)
    split = plan.simulate(
        origins[1:3],
        directions[1:3],
        energies[1:3],
        jr.key(9),
        history_ids=ids[1:3],
    )

    np.testing.assert_array_equal(split.material_kerma, together.material_kerma[1:3])
    np.testing.assert_array_equal(split.escaped_energy, together.escaped_energy[1:3])
    np.testing.assert_array_equal(split.status, together.status[1:3])
