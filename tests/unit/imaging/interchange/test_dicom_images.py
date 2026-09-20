from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import pytest


pydicom = pytest.importorskip("pydicom")

from phydrax.imaging import DeidentificationEvidence
from phydrax.imaging.interchange import (
    DICOMProfileError,
    DICOMResourcePolicy,
    read_dicom_enhanced_ct_image,
    read_dicom_legacy_ct_image_series,
    read_dicom_nuclear_medicine_counts_image,
    read_dicom_pet_activity_concentration_image_series,
)
from phydrax.interchange import (
    bounded_resource_from_bytes,
    ResourceLimits,
)
from phydrax.measurement import RadiationQuantityKind, ValueKind
from phydrax.qualification import ReferenceArtifactManifest
from tests.unit.imaging.interchange._synthetic import (
    enhanced_ct,
    legacy_ct,
    nm_counts,
    pet_slice,
)


_LIMITS = ResourceLimits(
    max_bytes=1_000_000,
    max_depth=32,
    max_nodes=10_000,
    max_attributes=10_000,
    max_losses=0,
)


def _inputs(*payloads: bytes):
    resources = tuple(
        bounded_resource_from_bytes(payload, limits=_LIMITS) for payload in payloads
    )
    references = tuple(
        ReferenceArtifactManifest(
            f"synthetic-dicom-{index}",
            checksum_algorithm="sha256",
            checksum=hashlib.sha256(payload).hexdigest(),
            size_bytes=len(payload),
            license_id="synthetic-test-data",
            commercial_use_permitted=True,
            redistribution_permitted=True,
            training_use_permitted=True,
            export_permitted=True,
            export_classification="unrestricted",
            nondimensionalization={"value": 1.0},
            uncertainty=None,
            lineage_ids=(f"synthetic-generator-{index}",),
        )
        for index, payload in enumerate(payloads)
    )
    return resources, references


def _deidentification():
    return DeidentificationEvidence(
        "synthetic-dicom-deid",
        "synthetic-subject",
        "synthetic-protocol",
        True,
        True,
        True,
    )


def test_legacy_ct_applies_scaling_once_and_constructs_lps_voxel_centers() -> None:
    payloads = (
        legacy_ct(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.12",
            z_mm=4.0,
            pixels=np.asarray([[1, 2], [3, 4]]),
        ),
        legacy_ct(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.11",
            z_mm=0.0,
            pixels=np.asarray([[5, 6], [7, 8]]),
        ),
    )
    resources, references = _inputs(*payloads)
    result = read_dicom_legacy_ct_image_series(resources, references, _deidentification())

    np.testing.assert_array_equal(
        result.asset.values[:, :, 0],
        np.asarray([[-990.0, -988.0], [-986.0, -984.0]]),
    )
    np.testing.assert_array_equal(
        result.asset.values[:, :, 1],
        np.asarray([[-998.0, -996.0], [-994.0, -992.0]]),
    )
    np.testing.assert_array_equal(
        result.asset.spatial_affine.matrix,
        np.asarray(
            [
                [0.0, 3.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 4.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    )
    assert result.asset.spatial_affine.voxel_reference.value == "center"
    assert result.asset.spatial_affine.axis_convention.value == "LPS"
    assert not result.asset.values.flags.writeable
    assert len(result.asset.references) == 2
    assert result.report.graph.unresolved_references == ()


def test_legacy_ct_refuses_irregular_grid_and_phi() -> None:
    payloads = tuple(
        legacy_ct(
            sop_instance_uid=f"1.2.826.0.1.3680043.10.999.{20 + index}",
            z_mm=z,
            pixels=np.ones((2, 2), dtype=np.int16),
        )
        for index, z in enumerate((0.0, 2.0, 5.0))
    )
    resources, references = _inputs(*payloads)
    with pytest.raises(DICOMProfileError, match="regular grid"):
        read_dicom_legacy_ct_image_series(resources, references, _deidentification())

    dataset = pydicom.dcmread(BytesIO(payloads[0]))
    dataset.PatientName = "FORBIDDEN^IDENTIFIER"
    stream = BytesIO()
    pydicom.dcmwrite(stream, dataset, enforce_file_format=True)
    resources, references = _inputs(stream.getvalue())
    with pytest.raises(PermissionError, match="PHI refusal"):
        read_dicom_legacy_ct_image_series(resources, references, _deidentification())


def test_enhanced_ct_uses_per_frame_scaling_before_frame_reordering() -> None:
    payload = enhanced_ct(sop_instance_uid="1.2.826.0.1.3680043.10.999.30")
    resources, references = _inputs(payload)
    result = read_dicom_enhanced_ct_image(
        resources[0], references[0], _deidentification()
    )

    np.testing.assert_array_equal(
        result.asset.values[:, :, 0],
        np.asarray([[-885.0, -882.0], [-879.0, -876.0]]),
    )
    np.testing.assert_array_equal(
        result.asset.values[:, :, 1],
        np.asarray([[-998.0, -996.0], [-994.0, -992.0]]),
    )


def test_nm_counts_and_pet_activity_concentration_remain_distinct() -> None:
    nm_payload = nm_counts(sop_instance_uid="1.2.826.0.1.3680043.10.999.40")
    nm_resources, nm_references = _inputs(nm_payload)
    nm = read_dicom_nuclear_medicine_counts_image(
        nm_resources[0], nm_references[0], _deidentification()
    )
    assert nm.asset.layout.kind is ValueKind.COUNT
    assert nm.asset.quantity.quantity_kind == "detected_counts"
    assert nm.asset.sampling.temporal.kind.value == "interval_integral"

    pet_payloads = (
        pet_slice(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.51",
            z_mm=0.0,
            pixels=np.asarray([[2, 4], [6, 8]]),
        ),
        pet_slice(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.52",
            z_mm=4.0,
            pixels=np.asarray([[10, 12], [14, 16]]),
        ),
    )
    pet_resources, pet_references = _inputs(*pet_payloads)
    pet = read_dicom_pet_activity_concentration_image_series(
        pet_resources, pet_references, _deidentification()
    )
    assert (
        pet.asset.quantity.quantity_kind
        == RadiationQuantityKind.ACTIVITY_CONCENTRATION.value
    )
    assert pet.asset.quantity.quantity_kind != nm.asset.quantity.quantity_kind
    np.testing.assert_array_equal(
        pet.asset.values[:, :, 0], np.asarray([[1.0, 2.0], [3.0, 4.0]])
    )

    bad_payload = nm_counts(sop_instance_uid="1.2.826.0.1.3680043.10.999.41", slope=2.0)
    bad_resources, bad_references = _inputs(bad_payload)
    with pytest.raises(DICOMProfileError, match="preserve counts"):
        read_dicom_nuclear_medicine_counts_image(
            bad_resources[0], bad_references[0], _deidentification()
        )


def test_dicom_metadata_limits_fail_during_bounded_walk() -> None:
    payload = enhanced_ct(sop_instance_uid="1.2.826.0.1.3680043.10.999.60")
    resources, references = _inputs(payload)
    with pytest.raises(DICOMProfileError, match="element count"):
        read_dicom_enhanced_ct_image(
            resources[0],
            references[0],
            _deidentification(),
            policy=DICOMResourcePolicy(max_elements=1),
        )
    with pytest.raises(DICOMProfileError, match="sequence depth"):
        read_dicom_enhanced_ct_image(
            resources[0],
            references[0],
            _deidentification(),
            policy=DICOMResourcePolicy(max_sequence_depth=1),
        )

    strict_nodes = bounded_resource_from_bytes(
        payload,
        limits=ResourceLimits(
            max_bytes=1_000_000,
            max_depth=32,
            max_nodes=1,
            max_attributes=100,
            max_losses=100,
        ),
    )
    with pytest.raises(DICOMProfileError, match="element count"):
        read_dicom_enhanced_ct_image(
            strict_nodes,
            references[0],
            _deidentification(),
        )

    strict_depth = bounded_resource_from_bytes(
        payload,
        limits=ResourceLimits(
            max_bytes=1_000_000,
            max_depth=1,
            max_nodes=10_000,
            max_attributes=10_000,
            max_losses=100,
        ),
    )
    with pytest.raises(DICOMProfileError, match="sequence depth"):
        read_dicom_enhanced_ct_image(
            strict_depth,
            references[0],
            _deidentification(),
        )
