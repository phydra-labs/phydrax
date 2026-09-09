import hashlib
from io import BytesIO

import h5py
import numpy as np

import phydrax as phx


def _reference(payload, name):
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="fixture",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("synthetic",),
    )


def _resource(payload):
    return phx.interchange.bounded_resource_from_bytes(
        payload,
        limits=phx.interchange.ResourceLimits(
            max_bytes=1_000_000,
            max_depth=8,
            max_nodes=100_000,
            max_attributes=100,
            max_losses=10,
        ),
    )


def _statepoint_bytes():
    buffer = BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.attrs["filetype"] = np.bytes_("statepoint")
        handle.attrs["openmc_version"] = np.bytes_("synthetic")
        handle.create_dataset("n_realizations", data=np.asarray(4, dtype=np.int64))
        tally = handle.create_group("tallies").create_group("tally 7")
        results = np.zeros((2, 2, 2), dtype=np.float64)
        results[..., 0] = np.asarray([[8.0, 12.0], [16.0, 20.0]])
        results[..., 1] = np.asarray([[18.0, 38.0], [68.0, 106.0]])
        tally.create_dataset("results", data=results)
    return buffer.getvalue()


def test_openmc_statepoint_import_preserves_mean_uncertainty_and_realizations():
    payload = _statepoint_bytes()
    groups = phx.nuclear.EnergyGroupStructure(
        [0.0, 1.0, 2.0], phx.units.MEGAELECTRONVOLT, source_id="groups"
    )
    result = phx.nuclear.interchange.import_openmc_multigroup_flux(
        _resource(payload),
        _reference(payload, "statepoint"),
        phx.nuclear.interchange.OpenMCStatepointProfile(
            7, (2, 2), ("region", "energy_group"), 10.0
        ),
        groups,
    )

    assert result.report.valid
    assert result.openmc_version == "synthetic"
    assert result.flux.realization_count == 4
    np.testing.assert_allclose(
        result.flux.field.values, np.asarray([[20.0, 30.0], [40.0, 50.0]])
    )
    assert result.flux.field.uncertainty is not None
    assert np.all(result.flux.field.uncertainty.values >= 0.0)
    assert result.asset.references[0].manifest_id == result.reference.manifest_id


def test_dagmc_artifact_remains_opaque_and_source_bound():
    buffer = BytesIO()
    with h5py.File(buffer, "w") as handle:
        handle.create_group("tstt")
    payload = buffer.getvalue()
    artifact = phx.nuclear.interchange.DagmcGeometryArtifact(
        _resource(payload),
        _reference(payload, "dagmc"),
        phx.units.METER,
        "source-cad",
        "synthetic-converter",
        "release",
        (("1", "steel"), ("2", "plasma")),
        True,
    )

    assert artifact.report.valid
    assert artifact.resource.data == payload
    assert artifact.geometry_id == artifact.report.target_id
