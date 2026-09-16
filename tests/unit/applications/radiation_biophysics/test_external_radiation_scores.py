#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import hashlib
from io import BytesIO

import h5py
import numpy as np
import pytest

from phydrax import SpatialCoordinateContract
from phydrax.applications.radiation_biophysics._scores import (
    ExternalRadiationRunIdentity,
    RadiationScoreDefinition,
)
from phydrax.applications.radiation_biophysics.interchange import (
    _mcgpu,
    _moqui,
    _openxraymc,
)
from phydrax.imaging import ImageAxisConvention, ImageIndexAffine
from phydrax.interchange import AdapterLoss, bounded_resource_from_bytes, ResourceLimits
from phydrax.measurement import RadiationQuantityKind
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import GRAY, JOULE_PER_METER, MILLIMETER, ONE
from tools.external_radiation_score_qualification import (
    external_radiation_score_qualification,
)


_LIMITS = ResourceLimits(1_000_000, 8, 100, 100, 100)


def _manifest(payload: bytes, name: str, *, commercial: bool = False):
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id=f"LicenseRef-{name}",
        commercial_use_permitted=commercial,
        redistribution_permitted=False,
        training_use_permitted=False,
        export_permitted=False,
        export_classification="research-only",
        nondimensionalization={"declared_scale": 1.0},
        uncertainty={"declared_source_uncertainty": 0.0},
        lineage_ids=(f"lineage:{name}",),
    )


def _resource(payload: bytes):
    return bounded_resource_from_bytes(payload, limits=_LIMITS)


def _run(engine: str, revision: str, *, config_payload: bytes = b"configuration"):
    def reference(role: str, payload: bytes):
        return _manifest(payload, f"{engine}:{role}")

    return ExternalRadiationRunIdentity(
        f"{engine}:run-1",
        engine,
        revision,
        reference("build", f"build:{revision}".encode()),
        reference("configuration", config_payload),
        (reference("table", b"transport-table"),),
        (reference("calibration", b"calibration"),),
        (reference("seed", b"seed-lineage"),),
    )


def _affine():
    return ImageIndexAffine(
        np.eye(4),
        "score-index",
        SpatialCoordinateContract(
            MILLIMETER,
            coordinate_system="cartesian-lps",
            reference_frame="synthetic-phantom",
        ),
        ImageAxisConvention.LPS,
    )


def _score(
    kind=RadiationQuantityKind.DOSE_TO_WATER,
    unit=GRAY,
    *,
    shape=(2, 1, 1),
    axes=("x", "y", "z"),
    normalization="per-source-primary",
    representation="voxel-grid",
):
    return RadiationScoreDefinition(
        f"fixture-{kind.value}",
        kind,
        unit,
        normalization,
        shape,
        "<f4",
        _affine(),
        axes,
        "voxel-cell-average",
        f"synthetic-phantom:{kind.value}:{normalization}",
        representation,
    )


def _hdf5_result():
    stream = BytesIO()
    with h5py.File(stream, "w") as handle:
        handle.attrs["producer"] = "OpenXRayMC"
        handle.attrs["producer_revision"] = "2026.1"
        handle.attrs["score_contract"] = "dose-to-water-per-primary"
        handle.create_dataset(
            "scores/dose", data=np.asarray([1.0, 2.0], dtype="<f4").reshape(2, 1, 1)
        )
        handle.create_dataset(
            "scores/se", data=np.asarray([0.1, 0.2], dtype="<f4").reshape(2, 1, 1)
        )
    payload = stream.getvalue()
    reference = _manifest(payload, "openxraymc-score")
    loss = AdapterLoss(
        "score/history-resolved-tallies",
        "import",
        "unsupported",
        "The admitted artifact retains only the batch estimator.",
        changes_interpretation=True,
        affected_capability_ids=("history-resampling",),
    )
    profile = _openxraymc.OpenXRayMCHDF5Profile(
        "OpenXRayMC",
        _run("OpenXRayMC", "2026.1"),
        _score(),
        "/scores/dose",
        "/scores/se",
        "standard-error",
        "batch-mean",
        1000,
        20,
        "shared-primary-histories",
        required_attributes=(("score_contract", "dose-to-water-per-primary"),),
        declared_losses=(loss,),
    )
    result = _openxraymc.import_openxraymc_hdf5(
        _resource(payload),
        reference,
        profile,
        required_semantics=("uncertainty", "correlation", "grid-affine"),
    )
    return result, payload, reference, profile


def test_openxraymc_hdf5_preserves_profile_rights_losses_and_correlated_evidence():
    result, payload, reference, profile = _hdf5_result()
    assert result.profile_id == profile.profile_id
    assert (
        dict(result.report.source_profile.qualifiers)["external_profile_id"]
        == profile.profile_id
    )
    assert result.definition.grid_affine.affine_id == _affine().affine_id
    assert result.estimator_evidence.correlated
    assert result.estimator_evidence.correlation_model == "shared-primary-histories"
    assert result.declared_losses == profile.declared_losses
    assert result.research_only and result.intended_use == "research-only"
    assert reference.manifest_id in {item.manifest_id for item in result.references}
    with pytest.raises(PermissionError, match="commercial-use-not-permitted"):
        _openxraymc.import_openxraymc_hdf5(
            _resource(payload), reference, profile, commercial_use=True
        )
    damaged = payload[:-1] + bytes((payload[-1] ^ 1,))
    with pytest.raises(ValueError, match="checksum mismatch"):
        _openxraymc.import_openxraymc_hdf5(_resource(damaged), reference, profile)


def test_openxraymc_hdf5_preflights_logical_shape_before_reading_payload():
    _, _, _, profile = _hdf5_result()
    stream = BytesIO()
    with h5py.File(stream, "w") as handle:
        handle.attrs["producer"] = "OpenXRayMC"
        handle.attrs["producer_revision"] = "2026.1"
        handle.attrs["score_contract"] = "dose-to-water-per-primary"
        handle.create_dataset(
            "scores/dose",
            shape=(2**30,),
            dtype="<f4",
            chunks=(1024,),
            fillvalue=0.0,
        )
    payload = stream.getvalue()
    reference = _manifest(payload, "logical-hdf5-score")
    with pytest.raises(ValueError, match="shape differs"):
        _openxraymc.import_openxraymc_hdf5(_resource(payload), reference, profile)


def test_radiation_score_definition_distinguishes_physical_meanings_and_dij():
    definitions = (
        _score(RadiationQuantityKind.ABSORBED_DOSE),
        _score(RadiationQuantityKind.KERMA),
        _score(
            RadiationQuantityKind.RELATIVE_DOSE,
            ONE,
            normalization="relative-to-prescription-point",
        ),
        _score(
            RadiationQuantityKind.LET,
            JOULE_PER_METER,
            normalization="dose-averaged-let",
        ),
        _score(
            RadiationQuantityKind.DOSE_TO_MEDIUM,
            shape=(2, 1, 1, 3),
            axes=("x", "y", "z", "beamlet"),
            normalization="per-beamlet-primary",
            representation="dij",
        ),
    )
    assert len({value.quantity.quantity_id for value in definitions}) == len(definitions)
    assert definitions[-1].representation == "dij"
    assert definitions[-1].quantity_kind is RadiationQuantityKind.DOSE_TO_MEDIUM
    with pytest.raises(ValueError, match="unit"):
        _score(RadiationQuantityKind.KERMA, ONE)
    with pytest.raises(ValueError, match="Dij"):
        _score(
            RadiationQuantityKind.LET,
            JOULE_PER_METER,
            shape=(2, 1, 1, 3),
            axes=("x", "y", "z", "beamlet"),
            representation="dij",
        )


def test_moqui_npz_checks_exact_members_affine_and_semantic_requirements():
    values = np.asarray([3.0, 4.0], dtype="<f4").reshape((2, 1, 1))
    uncertainty = np.asarray([0.3, 0.4], dtype="<f4").reshape((2, 1, 1))
    stream = BytesIO()
    np.savez(
        stream,
        dose=values,
        affine=np.eye(4),
        uncertainty=uncertainty,
    )
    payload = stream.getvalue()
    reference = _manifest(payload, "moqui-npz")
    profile = _moqui.MoquiArrayProfile(
        "npz",
        _run("Moqui", "1.2.3"),
        _score(RadiationQuantityKind.ABSORBED_DOSE),
        (2, 1, 1),
        (0, 1, 2),
        "dose",
        "affine",
        "uncertainty",
        "relative-standard-error",
        "history-batch-mean",
        2000,
        40,
        "common-random-numbers",
    )
    result = _moqui.import_moqui_arrays(
        _resource(payload),
        reference,
        profile,
        required_semantics=("uncertainty", "correlation"),
    )
    np.testing.assert_array_equal(result.values, values)
    assert result.estimator_evidence.correlation_model == "common-random-numbers"
    bad_stream = BytesIO()
    shifted = np.eye(4)
    shifted[0, 3] = 1.0
    np.savez(bad_stream, dose=values, affine=shifted, uncertainty=uncertainty)
    bad_payload = bad_stream.getvalue()
    with pytest.raises(ValueError, match="grid affine"):
        _moqui.import_moqui_arrays(
            _resource(bad_payload),
            _manifest(bad_payload, "moqui-bad-affine"),
            profile,
        )


def test_moqui_embedded_mha_validates_geometry_and_refuses_absent_uncertainty():
    header = (
        "ObjectType = Image\n"
        "NDims = 3\n"
        "BinaryData = True\n"
        "BinaryDataByteOrderMSB = False\n"
        "CompressedData = False\n"
        "TransformMatrix = 1 0 0 0 1 0 0 0 1\n"
        "Offset = 0 0 0\n"
        "ElementSpacing = 1 1 1\n"
        "DimSize = 2 1 1\n"
        "ElementType = MET_FLOAT\n"
        "ElementDataFile = LOCAL\n\n"
    ).encode("ascii")
    payload = header + np.asarray([5.0, 6.0], dtype="<f4").tobytes()
    reference = _manifest(payload, "moqui-mha")
    profile = _moqui.MoquiArrayProfile(
        "mha",
        _run("Moqui", "1.2.3"),
        _score(RadiationQuantityKind.DOSE_TO_MEDIUM),
        (1, 1, 2),
        (2, 1, 0),
        None,
        None,
        None,
        "unreported",
        "history-total",
        500,
        10,
        "unreported",
    )
    result = _moqui.import_moqui_arrays(_resource(payload), reference, profile)
    np.testing.assert_array_equal(result.values[:, 0, 0], (5.0, 6.0))
    with pytest.raises(ValueError, match="omits required semantics"):
        _moqui.import_moqui_arrays(
            _resource(payload),
            reference,
            profile,
            required_semantics=("correlation",),
        )


def test_mcgpu_raw_binds_configuration_identity_dij_and_declared_losses():
    config = b"ENGINE = MCGPU\nSCORE = DOSE_INFLUENCE\n"
    run = _run("MCGPU", "1.3", config_payload=config)
    score = _score(
        RadiationQuantityKind.DOSE_TO_WATER,
        shape=(2, 1, 1, 2),
        axes=("x", "y", "z", "beamlet"),
        normalization="per-beamlet-primary",
        representation="dij",
    )
    values = np.arange(1.0, 5.0, dtype="<f4").reshape(score.shape)
    raw = values.tobytes()
    raw_reference = _manifest(raw, "mcgpu-dij-raw")
    loss = AdapterLoss(
        "score/track-ledger",
        "import",
        "unsupported",
        "RAW dose influence storage contains no track ledger.",
        changes_interpretation=False,
    )
    profile = _mcgpu.MCGPURawProfile(
        run,
        score,
        score.shape,
        (0, 1, 2, 3),
        ("ENGINE = MCGPU", "SCORE = DOSE_INFLUENCE"),
        "unreported",
        "history-total",
        4000,
        20,
        "unreported",
        declared_losses=(loss,),
    )
    result = _mcgpu.import_mcgpu_raw(
        _resource(raw),
        raw_reference,
        _resource(config),
        run.configuration,
        profile,
        required_semantics=("configuration-lines", "score-normalization"),
    )
    np.testing.assert_array_equal(result.values, values)
    assert result.definition.representation == "dij"
    assert result.declared_losses == (loss,)
    wrong_config = _manifest(config, "unrelated-config-manifest")
    with pytest.raises(ValueError, match="pinned configuration"):
        _mcgpu.import_mcgpu_raw(
            _resource(raw),
            raw_reference,
            _resource(config),
            wrong_config,
            profile,
        )


def test_qualification_is_research_only_and_profiles_expose_no_execution_api():
    result, _, _, _ = _hdf5_result()
    qualification = external_radiation_score_qualification((result,))
    assert qualification["successful"]
    assert qualification["scope"] == "research-only"
    assert qualification["clinical_use_permitted"] is False
    assert qualification["provider_execution_performed"] is False
    assert qualification["scientifically_qualified"] is False
    assert _openxraymc.__all__ == [
        "OpenXRayMCHDF5Profile",
        "import_openxraymc_hdf5",
    ]
    assert _moqui.__all__ == ["MoquiArrayProfile", "import_moqui_arrays"]
    assert _mcgpu.__all__ == ["MCGPURawProfile", "import_mcgpu_raw"]
