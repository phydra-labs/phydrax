#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Named, read-only DICOM image profiles with exact physical semantics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

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
    TemporalSampling,
    TemporalSamplingKind,
    ValueKind,
    ValueLayout,
)
from ...qualification import ReferenceArtifactManifest
from ...units import (
    BECQUEREL,
    derived_unit,
    DIMENSIONLESS,
    MILLILITER,
    MILLISECOND,
    ONE,
    SI_REFERENCE_SYSTEM_ID,
    UnitDefinition,
)
from .._asset import ImageFieldSpec
from .._core import DeidentificationEvidence, MedicalImageAsset
from ._common import (
    _decode_stored_pixels,
    _float,
    _floats,
    _load_verified_dicom,
    _optional,
    _orientation,
    _reference_graph,
    _regular_geometry,
    _report,
    _required,
    _same_identity_context,
    _scaled_pixels,
    _sequence_item,
    _text,
)
from ._contracts import DICOMImportReport, DICOMProfileError, DICOMResourcePolicy


_CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
_ENHANCED_CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2.1"
_NM_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.20"
_PET_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.128"
_HOUNSFIELD_UNIT = UnitDefinition("HU", DIMENSIONLESS, SI_REFERENCE_SYSTEM_ID)
_BECQUEREL_PER_MILLILITER = derived_unit("Bq/mL", ((BECQUEREL, 1), (MILLILITER, -1)))
_DEFAULT_RESOURCE_POLICY = DICOMResourcePolicy()


@dataclass(frozen=True, slots=True)
class DICOMImageImport:
    """A governed medical image and its closed DICOM import report."""

    asset: MedicalImageAsset
    report: DICOMImportReport

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if not isinstance(self.report, DICOMImportReport):
            raise TypeError("report must be DICOMImportReport.")
        report_references = {item.manifest.manifest_id for item in self.report.resources}
        if len(report_references) != len(self.report.resources):
            raise ValueError("DICOM image resources must have unique byte identities.")


def _require_profile(dataset: Any, sop_class_uid: str, modality: str, /) -> None:
    if _text(_required(dataset, "SOPClassUID"), "SOPClassUID") != sop_class_uid:
        raise DICOMProfileError("DICOM SOP Class does not match the named profile.")
    if _text(_required(dataset, "Modality"), "Modality").upper() != modality:
        raise DICOMProfileError("DICOM Modality does not match the named profile.")


def _common_image_shape(datasets: Sequence[Any], /) -> tuple[int, int]:
    shapes = {
        (int(_required(dataset, "Rows")), int(_required(dataset, "Columns")))
        for dataset in datasets
    }
    if len(shapes) != 1:
        raise DICOMProfileError("DICOM image planes must have one rows/columns shape.")
    rows, columns = shapes.pop()
    if rows < 1 or columns < 1:
        raise DICOMProfileError("DICOM image dimensions must be positive.")
    return rows, columns


def _single_frame_stored(dataset: Any, policy: DICOMResourcePolicy, /) -> np.ndarray:
    if int(_optional(dataset, "NumberOfFrames", 1)) != 1:
        raise DICOMProfileError(
            "This named DICOM profile requires single-frame instances."
        )
    stored = _decode_stored_pixels(dataset, policy)
    if stored.ndim != 2:
        raise DICOMProfileError("A single-frame DICOM instance decoded with wrong rank.")
    return stored


def _padding_mask(dataset: Any, stored: np.ndarray, /) -> np.ndarray:
    padding = _optional(dataset, "PixelPaddingValue")
    limit = _optional(dataset, "PixelPaddingRangeLimit")
    if padding is None and limit is None:
        return np.ones(stored.shape, dtype=np.bool_)
    if padding is None:
        raise DICOMProfileError("PixelPaddingRangeLimit requires PixelPaddingValue.")
    low = int(padding)
    high = low if limit is None else int(limit)
    minimum, maximum = sorted((low, high))
    return np.logical_or(stored < minimum, stored > maximum)


def _ct_spec() -> ImageFieldSpec:
    return ImageFieldSpec.named(
        "ct-number",
        _HOUNSFIELD_UNIT,
        ValueKind.REAL_SCALAR,
        namespace="dicom",
        quantity_kind="ct_number",
        compatibility_key="dicom.ct-number.hounsfield",
        spatial_sampling=SpatialSamplingKind.CELL_AVERAGE,
    )


def _counts_spec(duration_ms: float, reference_time_ms: float) -> ImageFieldSpec:
    bounds = np.asarray(
        [[reference_time_ms - 0.5 * duration_ms, reference_time_ms + 0.5 * duration_ms]],
        dtype=np.float64,
    )
    temporal = TemporalSampling(
        TemporalSamplingKind.INTERVAL_INTEGRAL,
        bounds,
        MILLISECOND,
        origin="DICOM FrameReferenceTime",
    )
    return ImageFieldSpec(
        ImageFieldSpec.named(
            "detected-counts",
            ONE,
            ValueKind.COUNT,
            namespace="dicom",
            quantity_kind="detected_counts",
            compatibility_key="dicom.nm.detected-counts",
            spatial_sampling=SpatialSamplingKind.CELL_INTEGRAL,
        ).quantity,
        ValueLayout(ValueKind.COUNT),
        SamplingSemantics(
            SpatialSamplingKind.CELL_INTEGRAL,
            temporal,
            normalization="none",
        ),
    )


def _pet_spec() -> ImageFieldSpec:
    quantity = resolve_radiation_quantity(
        "pet-activity-concentration",
        RadiationQuantityKind.ACTIVITY_CONCENTRATION,
        _BECQUEREL_PER_MILLILITER,
        support_association="DICOM LPS voxel-cell mean",
        reference_configuration=(
            "DICOM PET BQML activity concentration decay-corrected to series start"
        ),
    )
    return ImageFieldSpec(
        quantity,
        ValueLayout.scalar(),
        SamplingSemantics(SpatialSamplingKind.CELL_AVERAGE),
    )


def _make_image_result(
    *,
    profile: str,
    loaded: Sequence[Any],
    graph: Any,
    asset_id: str,
    modality: str,
    values: np.ndarray,
    affine: Any,
    spec: ImageFieldSpec,
    deidentification: DeidentificationEvidence,
    metadata: dict[str, str | int | float | bool],
    valid_mask: np.ndarray,
) -> DICOMImageImport:
    report = _report(
        profile,
        loaded,
        graph,
        target_format="phydrax.MedicalImageAsset",
        target_id=asset_id,
        preserved_fields=(
            "object-identity",
            "reference-graph",
            "voxel-values",
            "quantity-semantics",
            "LPS-voxel-center-geometry",
        ),
        coordinate_mapping=(
            "DICOM array row -> affine index axis 0",
            "DICOM array column -> affine index axis 1",
            "DICOM plane order -> affine index axis 2",
        ),
    )
    first_identity = loaded[0].identity
    acquisition = AcquisitionIdentity(
        f"dicom:{first_identity.series_instance_uid}",
        first_identity.series_instance_uid,
        modality,
        deidentification.protocol_id,
    )
    derivation = DerivationRecord(
        DataOrigin.EXTERNAL,
        DataStage.CALIBRATED,
        adapter_report_ids=(report.adapter_report.report_id,),
    )
    asset = MedicalImageAsset(
        asset_id,
        modality,
        values,
        affine,
        spec,
        deidentification,
        tuple(item.reference for item in loaded),
        derivation,
        valid_mask=valid_mask,
        uncertainty=None,
        quality_flags=(),
        acquisition=acquisition,
        metadata=metadata,
        intended_use="research",
    )
    return DICOMImageImport(asset, report)


def read_dicom_legacy_ct_image_series(
    resources: Sequence[BoundedResource],
    references: Sequence[ReferenceArtifactManifest],
    deidentification: DeidentificationEvidence,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMImageImport:
    """Read a regular series of legacy single-frame CT Image Storage objects."""

    loaded = _load_verified_dicom(resources, references, deidentification, policy)
    datasets = tuple(item.dataset for item in loaded)
    for dataset in datasets:
        _require_profile(dataset, _CT_IMAGE_STORAGE, "CT")
    _, series_uid, frame_uid = _same_identity_context(loaded)
    _common_image_shape(datasets)
    positions = tuple(_required(item, "ImagePositionPatient") for item in datasets)
    orientations = tuple(_required(item, "ImageOrientationPatient") for item in datasets)
    spacings = tuple(_required(item, "PixelSpacing") for item in datasets)
    single_spacing = (
        _float(_required(datasets[0], "SliceThickness"), "SliceThickness")
        if len(datasets) == 1
        else None
    )
    geometry = _regular_geometry(
        positions,
        orientations,
        spacings,
        frame_uid,
        single_slice_spacing=single_spacing,
    )
    scaled: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    total_pixel_bytes = 0
    for dataset in datasets:
        if _optional(dataset, "RescaleType", "HU").upper() != "HU":
            raise DICOMProfileError("Legacy CT profile requires HU rescale semantics.")
        stored = _single_frame_stored(dataset, policy)
        total_pixel_bytes += stored.nbytes
        scaled.append(_scaled_pixels(dataset, stored))
        masks.append(_padding_mask(dataset, stored))
    if total_pixel_bytes > policy.max_total_pixel_bytes:
        raise DICOMProfileError("DICOM series pixel data exceed the resource policy.")
    order = geometry.order
    values = np.stack([scaled[index] for index in order], axis=-1)
    valid_mask = np.stack([masks[index] for index in order], axis=-1)
    values.setflags(write=False)
    valid_mask.setflags(write=False)
    graph = _reference_graph(loaded)
    return _make_image_result(
        profile="legacy-ct-image-series",
        loaded=loaded,
        graph=graph,
        asset_id=f"dicom:legacy-ct:{series_uid}",
        modality="CT",
        values=values,
        affine=geometry.affine,
        spec=_ct_spec(),
        deidentification=deidentification,
        metadata={"dicom_profile": "legacy-ct-image-series", "scaling_applied": True},
        valid_mask=valid_mask,
    )


def _functional_group(
    dataset: Any,
    frame_group: Any,
    keyword: str,
    /,
) -> Any:
    if keyword in frame_group:
        return _sequence_item(frame_group, keyword)
    shared = _sequence_item(dataset, "SharedFunctionalGroupsSequence")
    if keyword in shared:
        return _sequence_item(shared, keyword)
    raise DICOMProfileError(f"Enhanced CT profile requires {keyword} functional group.")


def read_dicom_enhanced_ct_image(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMImageImport:
    """Read one regular-grid Enhanced CT Image Storage object."""

    loaded = _load_verified_dicom((resource,), (reference,), deidentification, policy)
    dataset = loaded[0].dataset
    _require_profile(dataset, _ENHANCED_CT_IMAGE_STORAGE, "CT")
    _, series_uid, frame_uid = _same_identity_context(loaded)
    per_frame = _sequence_item(dataset, "PerFrameFunctionalGroupsSequence", single=False)
    stored = _decode_stored_pixels(dataset, policy)
    if stored.ndim == 2:
        stored = stored[np.newaxis, ...]
    if len(per_frame) != stored.shape[0]:
        raise DICOMProfileError(
            "Enhanced CT per-frame functional groups must match NumberOfFrames."
        )
    positions: list[Any] = []
    orientations: list[Any] = []
    spacings: list[Any] = []
    slopes: list[float] = []
    intercepts: list[float] = []
    slice_spacing: float | None = None
    for frame_group in per_frame:
        position_group = _functional_group(dataset, frame_group, "PlanePositionSequence")
        orientation_group = _functional_group(
            dataset, frame_group, "PlaneOrientationSequence"
        )
        measures = _functional_group(dataset, frame_group, "PixelMeasuresSequence")
        transform = _functional_group(
            dataset, frame_group, "PixelValueTransformationSequence"
        )
        positions.append(_required(position_group, "ImagePositionPatient"))
        orientations.append(_required(orientation_group, "ImageOrientationPatient"))
        spacings.append(_required(measures, "PixelSpacing"))
        if slice_spacing is None:
            spacing_value = _optional(
                measures,
                "SpacingBetweenSlices",
                _optional(measures, "SliceThickness"),
            )
            if spacing_value is not None:
                slice_spacing = _float(spacing_value, "slice spacing")
        if _optional(transform, "RescaleType", "HU").upper() != "HU":
            raise DICOMProfileError("Enhanced CT profile requires HU rescale semantics.")
        slopes.append(_float(_required(transform, "RescaleSlope"), "RescaleSlope"))
        intercepts.append(
            _float(_required(transform, "RescaleIntercept"), "RescaleIntercept")
        )
    geometry = _regular_geometry(
        positions,
        orientations,
        spacings,
        frame_uid,
        single_slice_spacing=slice_spacing,
    )
    scaled = tuple(
        _scaled_pixels(
            dataset, stored[index], slope=slopes[index], intercept=intercepts[index]
        )
        for index in range(stored.shape[0])
    )
    mask_frames = tuple(
        _padding_mask(dataset, stored[index]) for index in range(stored.shape[0])
    )
    values = np.stack([scaled[index] for index in geometry.order], axis=-1)
    valid_mask = np.stack([mask_frames[index] for index in geometry.order], axis=-1)
    values.setflags(write=False)
    valid_mask.setflags(write=False)
    graph = _reference_graph(loaded)
    return _make_image_result(
        profile="enhanced-ct-image",
        loaded=loaded,
        graph=graph,
        asset_id=f"dicom:enhanced-ct:{series_uid}",
        modality="CT",
        values=values,
        affine=geometry.affine,
        spec=_ct_spec(),
        deidentification=deidentification,
        metadata={"dicom_profile": "enhanced-ct-image", "scaling_applied": True},
        valid_mask=valid_mask,
    )


def read_dicom_nuclear_medicine_counts_image(
    resource: BoundedResource,
    reference: ReferenceArtifactManifest,
    deidentification: DeidentificationEvidence,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMImageImport:
    """Read one static, regular-grid NM image whose samples are accumulated counts."""

    loaded = _load_verified_dicom((resource,), (reference,), deidentification, policy)
    dataset = loaded[0].dataset
    _require_profile(dataset, _NM_IMAGE_STORAGE, "NM")
    _, series_uid, frame_uid = _same_identity_context(loaded)
    if _text(_required(dataset, "Units"), "Units").upper() != "CNTS":
        raise DICOMProfileError("NM counts profile requires DICOM Units=CNTS.")
    frames = int(_optional(dataset, "NumberOfFrames", 1))
    time_slices = int(_optional(dataset, "NumberOfTimeSlices", 1))
    slices = int(_optional(dataset, "NumberOfSlices", frames))
    if time_slices != 1 or slices != frames:
        raise DICOMProfileError(
            "NM counts profile admits one static time sample and one frame per slice."
        )
    duration_ms = _float(_required(dataset, "ActualFrameDuration"), "ActualFrameDuration")
    reference_time_ms = _float(
        _required(dataset, "FrameReferenceTime"), "FrameReferenceTime"
    )
    if duration_ms <= 0.0 or reference_time_ms < 0.0:
        raise DICOMProfileError("NM frame duration/time semantics are invalid.")
    slope = _float(_optional(dataset, "RescaleSlope", 1.0), "RescaleSlope")
    intercept = _float(_optional(dataset, "RescaleIntercept", 0.0), "RescaleIntercept")
    if slope != 1.0 or intercept != 0.0:
        raise DICOMProfileError(
            "NM counts profile refuses non-identity scaling because it would not preserve counts."
        )
    detector = _sequence_item(dataset, "DetectorInformationSequence")
    orientation_value = _required(detector, "ImageOrientationPatient")
    position_value = _required(detector, "ImagePositionPatient")
    pixel_spacing = _required(detector, "PixelSpacing")
    spacing = _float(
        _optional(
            detector,
            "SpacingBetweenSlices",
            _optional(
                dataset, "SpacingBetweenSlices", _optional(dataset, "SliceThickness")
            ),
        ),
        "SpacingBetweenSlices",
    )
    if spacing <= 0.0:
        raise DICOMProfileError("NM counts profile requires positive slice spacing.")
    _, _, normal = _orientation(orientation_value)
    base = _floats(position_value, 3, "ImagePositionPatient")
    positions = tuple(base + normal * spacing * index for index in range(frames))
    geometry = _regular_geometry(
        positions,
        (orientation_value,) * frames,
        (pixel_spacing,) * frames,
        frame_uid,
        single_slice_spacing=spacing,
    )
    stored = _decode_stored_pixels(dataset, policy)
    if stored.ndim == 2:
        stored = stored[np.newaxis, ...]
    values = np.transpose(stored[np.asarray(geometry.order)], (1, 2, 0)).copy()
    mask_frames = np.asarray(
        [_padding_mask(dataset, frame) for frame in stored], dtype=np.bool_
    )
    valid_mask = np.transpose(mask_frames[np.asarray(geometry.order)], (1, 2, 0))
    values.setflags(write=False)
    valid_mask.setflags(write=False)
    graph = _reference_graph(loaded)
    return _make_image_result(
        profile="nuclear-medicine-counts-image",
        loaded=loaded,
        graph=graph,
        asset_id=f"dicom:nm-counts:{series_uid}",
        modality="NM",
        values=values,
        affine=geometry.affine,
        spec=_counts_spec(duration_ms, reference_time_ms),
        deidentification=deidentification,
        metadata={
            "dicom_profile": "nuclear-medicine-counts-image",
            "frame_duration_ms": duration_ms,
            "frame_reference_time_ms": reference_time_ms,
            "scaling_applied": True,
        },
        valid_mask=valid_mask,
    )


def _corrected_image_terms(dataset: Any, /) -> tuple[str, ...]:
    value = _required(dataset, "CorrectedImage")
    values = (value,) if isinstance(value, str) else tuple(value)
    result = tuple(sorted({_text(item, "CorrectedImage").upper() for item in values}))
    if not {"ATTN", "DECY"}.issubset(result):
        raise DICOMProfileError(
            "PET activity profile requires attenuation and decay corrected values."
        )
    return result


def read_dicom_pet_activity_concentration_image_series(
    resources: Sequence[BoundedResource],
    references: Sequence[ReferenceArtifactManifest],
    deidentification: DeidentificationEvidence,
    /,
    *,
    policy: DICOMResourcePolicy = _DEFAULT_RESOURCE_POLICY,
) -> DICOMImageImport:
    """Read a static legacy PET series measured in decay-corrected Bq/mL."""

    loaded = _load_verified_dicom(resources, references, deidentification, policy)
    datasets = tuple(item.dataset for item in loaded)
    for dataset in datasets:
        _require_profile(dataset, _PET_IMAGE_STORAGE, "PT")
    _, series_uid, frame_uid = _same_identity_context(loaded)
    _common_image_shape(datasets)
    semantics: set[tuple[str, tuple[str, ...], str, str]] = set()
    for dataset in datasets:
        units = _text(_required(dataset, "Units"), "Units").upper()
        decay = _text(_required(dataset, "DecayCorrection"), "DecayCorrection").upper()
        corrected = _corrected_image_terms(dataset)
        series_date = _text(_required(dataset, "SeriesDate"), "SeriesDate")
        series_time = _text(_required(dataset, "SeriesTime"), "SeriesTime")
        if units != "BQML" or decay != "START":
            raise DICOMProfileError(
                "PET activity profile requires Units=BQML and DecayCorrection=START."
            )
        if _float(_required(dataset, "DecayFactor"), "DecayFactor") <= 0.0:
            raise DICOMProfileError("PET DecayFactor must be positive.")
        if _optional(dataset, "SUVType") is not None:
            raise DICOMProfileError("PET BQML profile refuses SUV-normalized values.")
        semantics.add((decay, corrected, series_date, series_time))
    if len(semantics) != 1:
        raise DICOMProfileError(
            "PET image planes disagree on activity or temporal reference semantics."
        )
    positions = tuple(_required(item, "ImagePositionPatient") for item in datasets)
    orientations = tuple(_required(item, "ImageOrientationPatient") for item in datasets)
    spacings = tuple(_required(item, "PixelSpacing") for item in datasets)
    single_spacing = (
        _float(_required(datasets[0], "SliceThickness"), "SliceThickness")
        if len(datasets) == 1
        else None
    )
    geometry = _regular_geometry(
        positions,
        orientations,
        spacings,
        frame_uid,
        single_slice_spacing=single_spacing,
    )
    scaled: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    total_pixel_bytes = 0
    for dataset in datasets:
        if _optional(dataset, "RescaleType", "BQML").upper() != "BQML":
            raise DICOMProfileError("PET profile requires BQML rescale semantics.")
        stored = _single_frame_stored(dataset, policy)
        total_pixel_bytes += stored.nbytes
        scaled.append(_scaled_pixels(dataset, stored))
        masks.append(_padding_mask(dataset, stored))
    if total_pixel_bytes > policy.max_total_pixel_bytes:
        raise DICOMProfileError("DICOM series pixel data exceed the resource policy.")
    values = np.stack([scaled[index] for index in geometry.order], axis=-1)
    valid_mask = np.stack([masks[index] for index in geometry.order], axis=-1)
    values.setflags(write=False)
    valid_mask.setflags(write=False)
    graph = _reference_graph(loaded)
    decay, corrected, series_date, series_time = semantics.pop()
    return _make_image_result(
        profile="pet-activity-concentration-image-series",
        loaded=loaded,
        graph=graph,
        asset_id=f"dicom:pet-activity:{series_uid}",
        modality="PT",
        values=values,
        affine=geometry.affine,
        spec=_pet_spec(),
        deidentification=deidentification,
        metadata={
            "dicom_profile": "pet-activity-concentration-image-series",
            "decay_correction": decay,
            "corrected_image": "\\".join(corrected),
            "series_date": series_date,
            "series_time": series_time,
            "scaling_applied": True,
        },
        valid_mask=valid_mask,
    )


__all__ = [
    "DICOMImageImport",
    "read_dicom_enhanced_ct_image",
    "read_dicom_legacy_ct_image_series",
    "read_dicom_nuclear_medicine_counts_image",
    "read_dicom_pet_activity_concentration_image_series",
]
