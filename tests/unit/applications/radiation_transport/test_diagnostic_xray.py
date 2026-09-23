#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx


def _manifest(name, character):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=character * 64,
        size_bytes=1,
        license_id="synthetic-permissive",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"identity": 1.0},
        uncertainty={"relative": 0.0},
        lineage_ids=(f"synthetic:{name}",),
    )


def _photon_table(name, character, energy_grid, values):
    provenance = phx.nuclear.NuclearDataProvenance(
        _manifest(name, character),
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
        ("water",),
        jnp.asarray((values,)),
        unit,
        provenance,
        phx.equations.DiagnosticPhotonInterpolationPolicy.LINEAR,
    )


def test_diagnostic_xray_source_transport_detector_pipeline_and_hit_adapter():
    spectrum = phx.applications.radiation_transport.AliasSpectrumPlan(
        jnp.asarray((1000.0,)),
        jnp.asarray((1.0,)),
        phx.units.ELECTRONVOLT,
        _manifest("spectrum", "d"),
    )
    source = phx.applications.radiation_transport.DiagnosticXRaySourcePlan(
        jnp.asarray((0.5, 0.5, 0.1)),
        jnp.asarray((0.0, 0.0, 1.0)),
        spectrum,
        cone_half_angle=0.0,
        source_id="point-source",
    )
    energy_grid = phx.equations.PhotonEnergyGrid(
        jnp.asarray((500.0, 1500.0))
        * float(phx.units.conversion_factor(phx.units.ELECTRONVOLT, phx.units.JOULE))
    )
    library = phx.equations.RadiationCrossSectionLibrary(
        _photon_table("photoelectric", "e", energy_grid, (0.01, 0.01)),
        _photon_table("compton", "f", energy_grid, (0.0, 0.0)),
        _photon_table("rayleigh", "a", energy_grid, (0.0, 0.0)),
        jnp.asarray((1.0,)),
    )
    geometry = phx.discretization.VoxelRadiationGeometryPlan(
        jnp.zeros((3,)),
        jnp.ones((3,)),
        jnp.zeros((2, 2, 2), dtype=jnp.int32),
        material_count=1,
    )
    transport = phx.solver.PhotonTransportPlan(
        geometry, library, maximum_events=16, cutoff_energy=500.0
    )
    detector = phx.applications.radiation_transport.PlanarXRayDetectorPlan(
        jnp.asarray((0.5, 0.5, 2.0)),
        jnp.asarray((0.0, 0.0, 1.0)),
        jnp.asarray((1.0, 0.0, 0.0)),
        width=1.0,
        height=1.0,
        pixel_shape=(4, 4),
        detector_id="flat-panel",
    )
    experiment = phx.applications.radiation_transport.DiagnosticXRayExperimentPlan(
        source, transport, detector
    )

    result = experiment.simulate(jr.key(7), 512)
    hits = detector.to_sensitive_hits(
        result.transport, result.detector, conditions_id="synthetic-conditions"
    )
    mismatched_transport = eqx.tree_at(
        lambda item: item.history_ids,
        result.transport,
        result.transport.history_ids + jnp.asarray(1, dtype=jnp.uint32),
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="different transport histories"):
        detector.to_sensitive_hits(
            mismatched_transport,
            result.detector,
            conditions_id="synthetic-conditions",
        )
    with pytest.raises(ValueError, match="uint32 support"):
        source.sample(
            jr.key(8),
            2,
            first_history_id=int(np.iinfo(np.uint32).max),
        )
    addressed_spectrum = phx.applications.radiation_transport.AliasSpectrumPlan(
        jnp.arange(1.0, 257.0),
        jnp.ones((256,)),
        phx.units.ELECTRONVOLT,
        _manifest("addressed-spectrum", "8"),
    )
    low_ids = jnp.arange(64, dtype=jnp.uint32)
    high_ids = low_ids + jnp.asarray(2**31, dtype=jnp.uint32)
    low_energy = addressed_spectrum.sample(jr.key(9), low_ids)
    high_energy = addressed_spectrum.sample(jr.key(9), high_ids)
    assert not jnp.array_equal(low_energy, high_energy)
    with pytest.raises(eqx.EquinoxRuntimeError, match="fit uint32"):
        addressed_spectrum.sample(
            jr.key(10),
            jnp.asarray([2**32], dtype=jnp.uint64),
        )
    run_manifest = _manifest("native-run", "7")
    artifact = phx.artifacts.ScientificArtifactEnvelope(
        artifact_kind="native-photon-transport",
        content_digest=run_manifest.checksum,
        producer="phydrax",
        producer_version="test",
        build_id="test-build",
        license_id=run_manifest.license_id,
        resource_id="test-resource",
        status="complete",
    )
    governed_source = phx.applications.radiation_biophysics.RadiationSource(
        artifact,
        (run_manifest,),
        "phydrax",
        "test",
        experiment.plan_id,
        ("key:7",),
        (spectrum.spectrum_id, library.library_id),
        (("photon-cutoff", 500.0, phx.units.ELECTRONVOLT),),
        "cartesian",
        phx.units.METER,
        phx.units.ELECTRONVOLT,
        phx.units.SECOND,
    )
    ledger = phx.applications.radiation_transport.photon_result_to_interaction_ledger(
        result.transport,
        governed_source,
        run_id="run-1",
        fraction_id="fraction-1",
    )

    assert bool(result.successful)
    assert jnp.sum(result.detector.hit) > 490
    np.testing.assert_allclose(
        jnp.sum(result.detector.image_by_scatter_class),
        jnp.sum(result.detector.incident_energy),
    )
    assert bool(jnp.all(hits.valid))
    assert hits.hit_capacity == 1
    assert len(ledger.records) == int(jnp.sum(result.transport.event_active))
    np.testing.assert_allclose(
        sum(record.deposited_energy for record in ledger.records),
        jnp.sum(result.transport.event_deposited_energy),
    )
