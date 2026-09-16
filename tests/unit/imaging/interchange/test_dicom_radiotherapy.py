from __future__ import annotations

import hashlib

import numpy as np
import pytest


pytest.importorskip("pydicom")

from phydrax.imaging import DeidentificationEvidence
from phydrax.imaging.interchange import (
    DICOMObjectIdentity,
    DICOMProfileError,
    read_dicom_rt_dose_linked_plan,
    read_dicom_rt_dose_unlinked,
    read_dicom_rt_plan_metadata,
    read_dicom_rt_structure_set_closed_planar,
)
from phydrax.interchange import bounded_resource_from_bytes, ResourceLimits
from phydrax.measurement import RadiationQuantityKind
from phydrax.qualification import ReferenceArtifactManifest
from tests.unit.imaging.interchange._synthetic import (
    CT_STORAGE,
    FRAME_UID,
    rtdose,
    rtplan,
    rtstruct,
    SERIES_UID,
    STUDY_UID,
)


_LIMITS = ResourceLimits(
    max_bytes=1_000_000,
    max_depth=32,
    max_nodes=10_000,
    max_attributes=10_000,
    max_losses=0,
)


def _input(payload: bytes, name: str):
    return (
        bounded_resource_from_bytes(payload, limits=_LIMITS),
        ReferenceArtifactManifest(
            name,
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
            lineage_ids=("synthetic-generator",),
        ),
    )


def _deidentification():
    return DeidentificationEvidence(
        "synthetic-dicom-deid",
        "synthetic-subject",
        "synthetic-protocol",
        True,
        True,
        True,
    )


def test_rtstruct_plan_and_dose_enforce_exact_reference_closure() -> None:
    image_uid = "1.2.826.0.1.3680043.10.999.101"
    structure_uid = "1.2.826.0.1.3680043.10.999.102"
    plan_uid = "1.2.826.0.1.3680043.10.999.103"
    dose_uid = "1.2.826.0.1.3680043.10.999.104"
    image_identity = DICOMObjectIdentity(
        CT_STORAGE,
        image_uid,
        STUDY_UID,
        SERIES_UID,
        FRAME_UID,
    )

    structure_resource, structure_reference = _input(
        rtstruct(sop_instance_uid=structure_uid, image_sop_uid=image_uid),
        "synthetic-rtstruct",
    )
    structure = read_dicom_rt_structure_set_closed_planar(
        structure_resource,
        structure_reference,
        _deidentification(),
        linked_identities=(image_identity,),
    )
    contour = structure.structure_set.regions[0].contours[0]
    assert contour.points_lps_mm.shape == (4, 3)
    assert not contour.points_lps_mm.flags.writeable
    assert structure.report.graph.unresolved_references == ()

    plan_resource, plan_reference = _input(
        rtplan(sop_instance_uid=plan_uid, structure_sop_uid=structure_uid),
        "synthetic-rtplan",
    )
    plan = read_dicom_rt_plan_metadata(
        plan_resource,
        plan_reference,
        _deidentification(),
        linked_identities=(structure.structure_set.identity,),
    )
    assert plan.plan.fraction_group_count == 1
    assert plan.plan.beam_count == 0
    assert plan.plan.referenced_structure_set_uids == (structure_uid,)

    dose_resource, dose_reference = _input(
        rtdose(sop_instance_uid=dose_uid, plan_sop_uid=plan_uid),
        "synthetic-rtdose",
    )
    dose = read_dicom_rt_dose_linked_plan(
        dose_resource,
        dose_reference,
        _deidentification(),
        plan,
    )
    assert dose.linked_plan_identity == plan.plan.identity
    assert dose.report.graph.unresolved_references == ()
    assert dose.asset.quantity.quantity_kind == RadiationQuantityKind.ABSORBED_DOSE.value
    np.testing.assert_array_equal(
        dose.asset.values[:, :, 0], np.asarray([[0.01, 0.02], [0.03, 0.04]])
    )
    np.testing.assert_array_equal(
        dose.asset.values[:, :, 1], np.asarray([[0.05, 0.06], [0.07, 0.08]])
    )

    with pytest.raises(DICOMProfileError, match="encoded RT Plan reference"):
        read_dicom_rt_dose_unlinked(
            dose_resource,
            dose_reference,
            _deidentification(),
        )


def test_rt_profiles_refuse_unresolved_links_and_unknown_dose_units() -> None:
    image_uid = "1.2.826.0.1.3680043.10.999.111"
    structure_uid = "1.2.826.0.1.3680043.10.999.112"
    resource, reference = _input(
        rtstruct(sop_instance_uid=structure_uid, image_sop_uid=image_uid),
        "unresolved-rtstruct",
    )
    with pytest.raises(ValueError, match="Unresolved linked DICOM references"):
        read_dicom_rt_structure_set_closed_planar(
            resource,
            reference,
            _deidentification(),
            linked_identities=(),
        )

    mismatched_image = DICOMObjectIdentity(
        CT_STORAGE,
        image_uid,
        STUDY_UID,
        SERIES_UID,
        "1.2.826.0.1.3680043.10.999.999",
    )
    with pytest.raises(DICOMProfileError, match="different Frames of Reference"):
        read_dicom_rt_structure_set_closed_planar(
            resource,
            reference,
            _deidentification(),
            linked_identities=(mismatched_image,),
        )

    dose_resource, dose_reference = _input(
        rtdose(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.113",
            plan_sop_uid=None,
            dose_units="RELATIVE",
        ),
        "unknown-dose-unit",
    )
    with pytest.raises(DICOMProfileError, match="known physical units"):
        read_dicom_rt_dose_unlinked(
            dose_resource,
            dose_reference,
            _deidentification(),
        )


def test_unlinked_rt_dose_is_explicitly_distinct_from_plan_linked_dose() -> None:
    dose_resource, dose_reference = _input(
        rtdose(
            sop_instance_uid="1.2.826.0.1.3680043.10.999.120",
            plan_sop_uid=None,
        ),
        "unlinked-dose",
    )
    result = read_dicom_rt_dose_unlinked(
        dose_resource,
        dose_reference,
        _deidentification(),
    )
    assert result.linked_plan_identity is None
    assert result.report.profile == "rt-dose-unlinked"
    assert result.dose_summation_type == "RECORD"
