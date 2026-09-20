#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Named DICOM radiotherapy metadata, contour, and dose profiles."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any

import numpy as np

from ..._fingerprint import array_tree_fingerprint, canonical_fingerprint
from ...interchange import BoundedResource
from ...measurement import (
    AcquisitionIdentity,
    DataOrigin,
    DataStage,
    DerivationRecord,
    RadiationQuantityKind,
    resolve_radiation_quantity,
    SamplingSemantics,
    SpatialSamplingKind,
    ValueLayout,
)
from ...qualification import ReferenceArtifactManifest
from ...units import GRAY
from .._asset import ImageFieldSpec
from .._core import DeidentificationEvidence, MedicalImageAsset
from ._common import (
    _decode_stored_pixels,
    _float,
    _floats,
    _load_verified_dicom,
    _optional,
    _reference_graph,
    _regular_geometry,
    _report,
    _required,
    _same_identity_context,
    _scaled_pixels,
    _sequence_item,
    _text,
)
from ._contracts import (
    DICOMImportReport,
    DICOMObjectIdentity,
    DICOMProfileError,
    DICOMResourcePolicy,
)


_RT_STRUCTURE_SET_STORAGE = "1.2.840.10008.5.1.4.1.1.481.3"
_RT_PLAN_STORAGE = "1.2.840.10008.5.1.4.1.1.481.5"
_RT_DOSE_STORAGE = "1.2.840.10008.5.1.4.1.1.481.2"
_GRAY = GRAY
_DEFAULT_RESOURCE_POLICY = DICOMResourcePolicy()


def _require_rt_profile(dataset: Any, sop_class_uid: str, modality: str, /) -> None:
    if _text(_required(dataset, "SOPClassUID"), "SOPClassUID") != sop_class_uid:
        raise DICOMProfileError("DICOM SOP Class does not match the named RT profile.")
    if _text(_required(dataset, "Modality"), "Modality").upper() != modality:
        raise DICOMProfileError("DICOM Modality does not match the named RT profile.")


def _count_sequence(dataset: Any, keyword: str, /) -> int:
    value = _optional(dataset, keyword, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise DICOMProfileError(f"DICOM {keyword} must be a sequence.")
    return len(value)


@dataclass(frozen=True, slots=True)
class RTPlanMetadata:
    """Non-delivery RT Plan identity and finite sequence counts only."""

    identity: DICOMObjectIdentity
    plan_label: str
    plan_name: str | None
    approval_status: str
    referenced_structure_set_uids: tuple[str, ...]
    fraction_group_count: int
    beam_count: int
    ion_beam_count: int
    dose_reference_count: int
    metadata_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DICOMObjectIdentity):
            raise TypeError("identity must be DICOMObjectIdentity.")
        label = _text(self.plan_label, "plan_label")
        name = None if self.plan_name is None else _text(self.plan_name, "plan_name")
        status = _text(self.approval_status, "approval_status").upper()
        if status not in ("APPROVED", "UNAPPROVED", "REJECTED"):
            raise ValueError("RT Plan approval status is unsupported.")
        referenced = tuple(
            _text(item, "referenced_structure_set_uid")
            for item in self.referenced_structure_set_uids
        )
        if not referenced or len(referenced) != len(set(referenced)):
            raise ValueError("RT Plan must reference unique structure-set SOP instances.")
        counts = (
            self.fraction_group_count,
            self.beam_count,
            self.ion_beam_count,
            self.dose_reference_count,
        )
        if any(
            isinstance(item, bool) or not isinstance(item, Integral) or item < 0
            for item in counts
        ):
            raise ValueError("RT Plan sequence counts must be non-negative integers.")
        object.__setattr__(self, "plan_label", label)
        object.__setattr__(self, "plan_name", name)
        object.__setattr__(self, "approval_status", status)
        object.__setattr__(self, "referenced_structure_set_uids", referenced)
        object.__setattr__(
            self,
            "metadata_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-rt-plan-metadata",
                    "identity": self.identity.identity_id,
                    "plan_label": label,
                    "plan_name": name,
                    "approval_status": status,
                    "referenced_structure_sets": list(referenced),
                    "fraction_group_count": int(counts[0]),
                    "beam_count": int(counts[1]),
                    "ion_beam_count": int(counts[2]),
                    "dose_reference_count": int(counts[3]),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DICOMRTPlanImport:
    plan: RTPlanMetadata
    report: DICOMImportReport

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RTPlanMetadata):
            raise TypeError("plan must be RTPlanMetadata.")
        if not isinstance(self.report, DICOMImportReport):
            raise TypeError("report must be DICOMImportReport.")
        if self.plan.identity.sop_instance_uid not in {
            item.sop_instance_uid for item in self.report.graph.objects
        }:
            raise ValueError("RT Plan identity is absent from its DICOM report graph.")


def read_dicom_rt_plan_metadata(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    linked_identities: Sequence[DICOMObjectIdentity],
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMRTPlanImport:
    """Read RTPLAN identity and sequence counts without constructing treatment behavior."""

    loaded = _load_verified_dicom((resource,), (reference,), deidentification, policy)
    dataset = loaded[0].dataset
    _require_rt_profile(dataset, _RT_PLAN_STORAGE, "RTPLAN")
    referenced_structure_set = _sequence_item(
        dataset, "ReferencedStructureSetSequence", single=False
    )
    structure_uids = tuple(
        _text(
            _required(item, "ReferencedSOPInstanceUID"),
            "ReferencedSOPInstanceUID",
        )
        for item in referenced_structure_set
    )
    plan = RTPlanMetadata(
        loaded[0].identity,
        _text(_required(dataset, "RTPlanLabel"), "RTPlanLabel"),
        (
            None
            if _optional(dataset, "RTPlanName") is None
            else _text(_optional(dataset, "RTPlanName"), "RTPlanName")
        ),
        _text(_required(dataset, "ApprovalStatus"), "ApprovalStatus"),
        structure_uids,
        _count_sequence(dataset, "FractionGroupSequence"),
        _count_sequence(dataset, "BeamSequence"),
        _count_sequence(dataset, "IonBeamSequence"),
        _count_sequence(dataset, "DoseReferenceSequence"),
    )
    graph = _reference_graph(
        loaded,
        linked_identities=linked_identities,
        require_closed=True,
    )
    report = _report(
        "rt-plan-metadata",
        loaded,
        graph,
        target_format="phydrax.RTPlanMetadata",
        target_id=plan.metadata_id,
        preserved_fields=(
            "object-identity",
            "reference-closure",
            "plan-label",
            "approval-status",
            "sequence-counts",
        ),
    )
    return DICOMRTPlanImport(plan, report)


@dataclass(frozen=True, slots=True)
class RTClosedPlanarContour:
    """One DICOM CLOSED_PLANAR polygon in LPS millimeters."""

    points_lps_mm: np.ndarray
    referenced_image_uids: tuple[str, ...]
    contour_id: str = field(init=False)

    def __post_init__(self) -> None:
        points = np.array(self.points_lps_mm, dtype=np.float64, copy=True)
        if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] < 3:
            raise ValueError(
                "A closed planar contour requires at least three LPS points."
            )
        if not np.all(np.isfinite(points)):
            raise ValueError("RT contour points must be finite.")
        centered = points - np.mean(points, axis=0)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        scale = max(1.0, float(singular_values[0]))
        if singular_values[-1] > 1.0e-6 * scale:
            raise ValueError("RT CLOSED_PLANAR contour points are not planar.")
        referenced = tuple(
            _text(item, "referenced_image_uid") for item in self.referenced_image_uids
        )
        if not referenced or len(referenced) != len(set(referenced)):
            raise ValueError("Each RT contour must reference unique source images.")
        points.setflags(write=False)
        object.__setattr__(self, "points_lps_mm", points)
        object.__setattr__(self, "referenced_image_uids", referenced)
        object.__setattr__(
            self,
            "contour_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-rt-closed-planar-contour",
                    "points": array_tree_fingerprint(points),
                    "referenced_images": list(referenced),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RTRegionOfInterest:
    roi_number: int
    name: str
    interpreted_type: str | None
    frame_of_reference_uid: str
    contours: tuple[RTClosedPlanarContour, ...]
    region_id: str = field(init=False)

    def __post_init__(self) -> None:
        if isinstance(self.roi_number, bool) or not isinstance(self.roi_number, Integral):
            raise TypeError("roi_number must be an integer.")
        number = int(self.roi_number)
        if number < 0:
            raise ValueError("roi_number must be non-negative.")
        name = _text(self.name, "ROIName")
        interpreted = (
            None
            if self.interpreted_type is None
            else _text(self.interpreted_type, "RTROIInterpretedType").upper()
        )
        frame = _text(self.frame_of_reference_uid, "ReferencedFrameOfReferenceUID")
        contours = tuple(self.contours)
        if not contours or any(
            not isinstance(item, RTClosedPlanarContour) for item in contours
        ):
            raise ValueError("Every admitted RT ROI requires closed planar contours.")
        object.__setattr__(self, "roi_number", number)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "interpreted_type", interpreted)
        object.__setattr__(self, "frame_of_reference_uid", frame)
        object.__setattr__(self, "contours", contours)
        object.__setattr__(
            self,
            "region_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-rt-region-of-interest",
                    "number": number,
                    "name": name,
                    "interpreted_type": interpreted,
                    "frame": frame,
                    "contours": [item.contour_id for item in contours],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class RTStructureSet:
    identity: DICOMObjectIdentity
    label: str
    regions: tuple[RTRegionOfInterest, ...]
    structure_set_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.identity, DICOMObjectIdentity):
            raise TypeError("identity must be DICOMObjectIdentity.")
        label = _text(self.label, "StructureSetLabel")
        regions = tuple(self.regions)
        if not regions or any(
            not isinstance(item, RTRegionOfInterest) for item in regions
        ):
            raise ValueError("RT structure set requires regions of interest.")
        numbers = [item.roi_number for item in regions]
        if len(numbers) != len(set(numbers)):
            raise ValueError("RT structure-set ROI numbers must be unique.")
        object.__setattr__(self, "label", label)
        object.__setattr__(self, "regions", regions)
        object.__setattr__(
            self,
            "structure_set_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-rt-structure-set",
                    "identity": self.identity.identity_id,
                    "label": label,
                    "regions": [item.region_id for item in regions],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DICOMRTStructureSetImport:
    structure_set: RTStructureSet
    report: DICOMImportReport

    def __post_init__(self) -> None:
        if not isinstance(self.structure_set, RTStructureSet):
            raise TypeError("structure_set must be RTStructureSet.")
        if not isinstance(self.report, DICOMImportReport):
            raise TypeError("report must be DICOMImportReport.")


def _roi_definitions(dataset: Any, /) -> dict[int, tuple[str, str]]:
    definitions: dict[int, tuple[str, str]] = {}
    for item in _sequence_item(dataset, "StructureSetROISequence", single=False):
        number = int(_required(item, "ROINumber"))
        if number in definitions:
            raise DICOMProfileError("RTSTRUCT repeats an ROI number.")
        definitions[number] = (
            _text(_required(item, "ROIName"), "ROIName"),
            _text(
                _required(item, "ReferencedFrameOfReferenceUID"),
                "ReferencedFrameOfReferenceUID",
            ),
        )
    return definitions


def _roi_interpretations(dataset: Any, /) -> dict[int, str | None]:
    result: dict[int, str | None] = {}
    for item in _sequence_item(dataset, "RTROIObservationsSequence", single=False):
        number = int(_required(item, "ReferencedROINumber"))
        if number in result:
            raise DICOMProfileError("RTSTRUCT repeats an ROI observation.")
        value = _optional(item, "RTROIInterpretedType")
        result[number] = None if value is None else _text(value, "RTROIInterpretedType")
    return result


def read_dicom_rt_structure_set_closed_planar(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    linked_identities: Sequence[DICOMObjectIdentity],
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMRTStructureSetImport:
    """Read an RTSTRUCT whose contours are closed planar and reference-closed."""

    loaded = _load_verified_dicom((resource,), (reference,), deidentification, policy)
    dataset = loaded[0].dataset
    _require_rt_profile(dataset, _RT_STRUCTURE_SET_STORAGE, "RTSTRUCT")
    definitions = _roi_definitions(dataset)
    interpretations = _roi_interpretations(dataset)
    linked = tuple(linked_identities)
    if any(not isinstance(item, DICOMObjectIdentity) for item in linked):
        raise TypeError("linked_identities must contain DICOMObjectIdentity values.")
    linked_by_uid = {item.sop_instance_uid: item for item in linked}
    if len(linked_by_uid) != len(linked):
        raise ValueError("linked_identities must have unique SOP Instance UIDs.")
    contours_by_roi: dict[int, tuple[RTClosedPlanarContour, ...]] = {}
    contour_count = 0
    point_count = 0
    for roi_item in _sequence_item(dataset, "ROIContourSequence", single=False):
        number = int(_required(roi_item, "ReferencedROINumber"))
        if number in contours_by_roi or number not in definitions:
            raise DICOMProfileError("RTSTRUCT ROI contour mapping is inconsistent.")
        contours: list[RTClosedPlanarContour] = []
        for contour_item in _sequence_item(roi_item, "ContourSequence", single=False):
            if (
                _text(
                    _required(contour_item, "ContourGeometricType"),
                    "ContourGeometricType",
                ).upper()
                != "CLOSED_PLANAR"
            ):
                raise DICOMProfileError(
                    "RTSTRUCT profile admits only CLOSED_PLANAR contours."
                )
            declared_points = int(_required(contour_item, "NumberOfContourPoints"))
            coordinates = _floats(
                _required(contour_item, "ContourData"),
                3 * declared_points,
                "ContourData",
            ).reshape(declared_points, 3)
            image_references = _sequence_item(
                contour_item, "ContourImageSequence", single=False
            )
            image_uids = tuple(
                _text(
                    _required(item, "ReferencedSOPInstanceUID"),
                    "ReferencedSOPInstanceUID",
                )
                for item in image_references
            )
            roi_frame = definitions[number][1]
            for image_uid in image_uids:
                image_identity = linked_by_uid.get(image_uid)
                if (
                    image_identity is not None
                    and image_identity.frame_of_reference_uid != roi_frame
                ):
                    raise DICOMProfileError(
                        "RTSTRUCT contour and referenced image use different Frames of Reference."
                    )
            contours.append(RTClosedPlanarContour(coordinates, image_uids))
            contour_count += 1
            point_count += declared_points
            if (
                contour_count > policy.max_contours
                or point_count > policy.max_contour_points
            ):
                raise DICOMProfileError("RTSTRUCT contours exceed the resource policy.")
        contours_by_roi[number] = tuple(contours)
    if set(definitions) != set(contours_by_roi):
        raise DICOMProfileError("Every RTSTRUCT ROI must have a contour representation.")
    if not set(interpretations).issubset(definitions):
        raise DICOMProfileError("RTSTRUCT observations reference unknown ROIs.")
    regions = tuple(
        RTRegionOfInterest(
            number,
            definitions[number][0],
            interpretations.get(number),
            definitions[number][1],
            contours_by_roi[number],
        )
        for number in sorted(definitions)
    )
    structure_set = RTStructureSet(
        loaded[0].identity,
        _text(_required(dataset, "StructureSetLabel"), "StructureSetLabel"),
        regions,
    )
    graph = _reference_graph(
        loaded,
        linked_identities=linked_identities,
        require_closed=True,
    )
    report = _report(
        "rt-structure-set-closed-planar",
        loaded,
        graph,
        target_format="phydrax.RTStructureSet",
        target_id=structure_set.structure_set_id,
        preserved_fields=(
            "object-identity",
            "reference-closure",
            "ROI-identity",
            "CLOSED_PLANAR-LPS-contours",
        ),
        coordinate_mapping=("ContourData -> LPS millimeters",),
    )
    return DICOMRTStructureSetImport(structure_set, report)


@dataclass(frozen=True, slots=True)
class DICOMRTDoseImport:
    asset: MedicalImageAsset
    report: DICOMImportReport
    dose_summation_type: str
    linked_plan_identity: DICOMObjectIdentity | None

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if not isinstance(self.report, DICOMImportReport):
            raise TypeError("report must be DICOMImportReport.")
        summation = _text(self.dose_summation_type, "dose_summation_type").upper()
        if self.linked_plan_identity is not None and not isinstance(
            self.linked_plan_identity, DICOMObjectIdentity
        ):
            raise TypeError("linked_plan_identity must be DICOMObjectIdentity or None.")
        object.__setattr__(self, "dose_summation_type", summation)


def _dose_spec() -> ImageFieldSpec:
    quantity = resolve_radiation_quantity(
        "rt-physical-absorbed-dose",
        RadiationQuantityKind.ABSORBED_DOSE,
        _GRAY,
        support_association="DICOM RT Dose LPS voxel-cell mean",
        reference_configuration=(
            "DICOM RT Dose PHYSICAL absorbed dose; material reference not encoded"
        ),
    )
    return ImageFieldSpec(
        quantity,
        ValueLayout.scalar(),
        SamplingSemantics(SpatialSamplingKind.CELL_AVERAGE),
    )


def _dose_geometry(dataset: Any, frame_uid: str, policy: DICOMResourcePolicy, /):
    stored = _decode_stored_pixels(dataset, policy)
    if stored.ndim == 2:
        stored = stored[np.newaxis, ...]
    frame_count = stored.shape[0]
    offsets = _floats(
        _required(dataset, "GridFrameOffsetVector"),
        frame_count,
        "GridFrameOffsetVector",
    )
    if offsets[0] != 0.0 or (frame_count > 1 and np.any(np.diff(offsets) <= 0.0)):
        raise DICOMProfileError(
            "RTDOSE profile requires increasing relative frame offsets beginning at zero."
        )
    orientation = _required(dataset, "ImageOrientationPatient")
    vectors = _floats(orientation, 6, "ImageOrientationPatient")
    normal = np.cross(vectors[:3], vectors[3:])
    normal /= np.linalg.norm(normal)
    base = _floats(_required(dataset, "ImagePositionPatient"), 3, "ImagePositionPatient")
    positions = tuple(base + normal * offset for offset in offsets)
    single_spacing = (
        _float(_required(dataset, "SliceThickness"), "SliceThickness")
        if frame_count == 1
        else None
    )
    geometry = _regular_geometry(
        positions,
        (orientation,) * frame_count,
        (_required(dataset, "PixelSpacing"),) * frame_count,
        frame_uid,
        single_slice_spacing=single_spacing,
    )
    return stored, geometry


def _read_rt_dose(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    profile: str,
    linked_plan: DICOMRTPlanImport | None,
    policy: DICOMResourcePolicy,
) -> DICOMRTDoseImport:
    loaded = _load_verified_dicom((resource,), (reference,), deidentification, policy)
    dataset = loaded[0].dataset
    _require_rt_profile(dataset, _RT_DOSE_STORAGE, "RTDOSE")
    _, series_uid, frame_uid = _same_identity_context(loaded)
    if _text(_required(dataset, "DoseUnits"), "DoseUnits").upper() != "GY":
        raise DICOMProfileError("RTDOSE profile requires known physical units GY.")
    if _text(_required(dataset, "DoseType"), "DoseType").upper() != "PHYSICAL":
        raise DICOMProfileError("RTDOSE profile admits physical dose only.")
    summation_type = _text(
        _required(dataset, "DoseSummationType"), "DoseSummationType"
    ).upper()
    scaling = _float(_required(dataset, "DoseGridScaling"), "DoseGridScaling")
    if scaling <= 0.0:
        raise DICOMProfileError("RTDOSE DoseGridScaling must be positive.")
    if "RescaleSlope" in dataset or "RescaleIntercept" in dataset:
        raise DICOMProfileError(
            "RTDOSE profile refuses a second rescale transform in addition to DoseGridScaling."
        )
    if "PixelPaddingValue" in dataset or "PixelPaddingRangeLimit" in dataset:
        raise DICOMProfileError(
            "RTDOSE profile refuses ambiguous pixel padding semantics."
        )
    stored, geometry = _dose_geometry(dataset, frame_uid, policy)
    scaled = _scaled_pixels(dataset, stored, slope=scaling, intercept=0.0)
    values = np.transpose(scaled[np.asarray(geometry.order)], (1, 2, 0))
    values.setflags(write=False)
    valid_mask = np.ones(values.shape, dtype=np.bool_)
    valid_mask.setflags(write=False)

    plan_identity: DICOMObjectIdentity | None = None
    if linked_plan is None:
        if "ReferencedRTPlanSequence" in dataset:
            raise DICOMProfileError(
                "Unlinked RTDOSE profile refuses an encoded RT Plan reference."
            )
        graph = _reference_graph(loaded, require_closed=True)
    else:
        if not isinstance(linked_plan, DICOMRTPlanImport):
            raise TypeError("linked_plan must be DICOMRTPlanImport.")
        if not linked_plan.report.adapter_report.valid:
            raise ValueError("Linked RT Plan import report must be valid.")
        plan_reference = _sequence_item(dataset, "ReferencedRTPlanSequence")
        target_uid = _text(
            _required(plan_reference, "ReferencedSOPInstanceUID"),
            "ReferencedSOPInstanceUID",
        )
        target_class = _text(
            _required(plan_reference, "ReferencedSOPClassUID"),
            "ReferencedSOPClassUID",
        )
        plan_identity = linked_plan.plan.identity
        if (
            target_uid != plan_identity.sop_instance_uid
            or target_class != plan_identity.sop_class_uid
        ):
            raise DICOMProfileError(
                "RTDOSE encoded RT Plan reference does not match the supplied plan."
            )
        graph = _reference_graph(
            loaded,
            linked_identities=linked_plan.report.graph.objects,
            require_closed=True,
        )
    report = _report(
        profile,
        loaded,
        graph,
        target_format="phydrax.MedicalImageAsset",
        target_id=f"dicom:rt-dose:{loaded[0].identity.sop_instance_uid}",
        preserved_fields=(
            "object-identity",
            "reference-graph",
            "DoseGridScaling",
            "physical-dose-semantics",
            "LPS-voxel-center-geometry",
        ),
        coordinate_mapping=(
            "DICOM array row -> affine index axis 0",
            "DICOM array column -> affine index axis 1",
            "GridFrameOffsetVector -> affine index axis 2",
        ),
    )
    acquisition = AcquisitionIdentity(
        f"dicom:{loaded[0].identity.sop_instance_uid}",
        series_uid,
        "RTDOSE",
        deidentification.protocol_id,
    )
    derivation = DerivationRecord(
        DataOrigin.EXTERNAL,
        DataStage.CALIBRATED,
        adapter_report_ids=(report.adapter_report.report_id,),
    )
    asset = MedicalImageAsset(
        f"dicom:rt-dose:{loaded[0].identity.sop_instance_uid}",
        "RTDOSE",
        values,
        geometry.affine,
        _dose_spec(),
        deidentification,
        (reference,),
        derivation,
        valid_mask=valid_mask,
        uncertainty=None,
        quality_flags=(),
        acquisition=acquisition,
        metadata={
            "dicom_profile": profile,
            "dose_summation_type": summation_type,
            "scaling_applied": True,
        },
        intended_use="research",
    )
    return DICOMRTDoseImport(asset, report, summation_type, plan_identity)


def read_dicom_rt_dose_unlinked(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMRTDoseImport:
    """Read physical RTDOSE only when it encodes no unresolved RT Plan link."""

    return _read_rt_dose(
        resource,
        reference,
        deidentification,
        profile="rt-dose-unlinked",
        linked_plan=None,
        policy=policy,
    )


def read_dicom_rt_dose_linked_plan(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    linked_plan: DICOMRTPlanImport,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMRTDoseImport:
    """Read physical RTDOSE after exact RT Plan SOP-reference closure."""

    return _read_rt_dose(
        resource,
        reference,
        deidentification,
        profile="rt-dose-linked-plan",
        linked_plan=linked_plan,
        policy=policy,
    )


__all__ = [
    "DICOMRTDoseImport",
    "DICOMRTPlanImport",
    "DICOMRTStructureSetImport",
    "RTClosedPlanarContour",
    "RTPlanMetadata",
    "RTRegionOfInterest",
    "RTStructureSet",
    "read_dicom_rt_dose_linked_plan",
    "read_dicom_rt_dose_unlinked",
    "read_dicom_rt_plan_metadata",
    "read_dicom_rt_structure_set_closed_planar",
]
