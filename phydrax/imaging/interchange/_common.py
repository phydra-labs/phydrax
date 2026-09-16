#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Bounded parser and geometry primitives for named DICOM profiles."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

from ..._fingerprint import canonical_fingerprint
from ..._physical import SpatialCoordinateContract
from ...interchange import (
    account_bounded_resource,
    AdapterFormatProfile,
    AdapterReport,
    AdapterStatus,
    BoundedResource,
)
from ...qualification import ReferenceArtifactManifest
from ...units import MILLIMETER
from .._core import (
    DeidentificationEvidence,
    ImageAxisConvention,
    ImageIndexAffine,
    VoxelReference,
)
from ._contracts import (
    DICOMDependencyError,
    DICOMImportReport,
    DICOMObjectIdentity,
    DICOMProfile,
    DICOMProfileError,
    DICOMReference,
    DICOMReferenceGraph,
    DICOMResourcePolicy,
)


_IMPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2"
_EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"
_SUPPORTED_TRANSFER_SYNTAXES = frozenset(
    {_IMPLICIT_VR_LITTLE_ENDIAN, _EXPLICIT_VR_LITTLE_ENDIAN}
)
_FORBIDDEN_PHI_KEYWORDS = frozenset(
    {
        "AccessionNumber",
        "AdditionalPatientHistory",
        "AdmittingDiagnosesDescription",
        "InstitutionAddress",
        "InstitutionName",
        "InstitutionalDepartmentName",
        "IssuerOfPatientID",
        "MedicalRecordLocator",
        "NameOfPhysiciansReadingStudy",
        "Occupation",
        "OperatorsName",
        "OtherPatientIDs",
        "OtherPatientIDsSequence",
        "OtherPatientNames",
        "PatientAddress",
        "PatientBirthDate",
        "PatientBirthName",
        "PatientBirthTime",
        "PatientID",
        "PatientInsurancePlanCodeSequence",
        "PatientMotherBirthName",
        "PatientName",
        "PatientReligiousPreference",
        "PatientTelephoneNumbers",
        "PerformingPhysicianName",
        "PhysiciansOfRecord",
        "ReferringPhysicianAddress",
        "ReferringPhysicianName",
        "ReferringPhysicianTelephoneNumbers",
        "RequestingPhysician",
        "ResponsiblePerson",
        "StudyID",
    }
)


@dataclass(frozen=True, slots=True)
class _LoadedDICOM:
    dataset: Any
    resource: BoundedResource
    reference: ReferenceArtifactManifest
    identity: DICOMObjectIdentity


@dataclass(frozen=True, slots=True)
class _Geometry:
    order: tuple[int, ...]
    affine: ImageIndexAffine


def _pydicom_module() -> Any:
    if importlib.util.find_spec("pydicom") is None:
        raise DICOMDependencyError(
            "The optional pydicom dependency is required for DICOM profile imports."
        )
    return importlib.import_module("pydicom")


def _required(dataset: Any, keyword: str, /) -> Any:
    if keyword not in dataset:
        raise DICOMProfileError(f"DICOM profile requires {keyword}.")
    value = dataset[keyword].value
    if value is None or (isinstance(value, str) and not value.strip()):
        raise DICOMProfileError(f"DICOM profile requires a non-empty {keyword}.")
    return value


def _optional(dataset: Any, keyword: str, default: Any = None, /) -> Any:
    if keyword not in dataset:
        return default
    value = dataset[keyword].value
    return default if value is None else value


def _text(value: Any, name: str, /) -> str:
    result = str(value).strip()
    if not result:
        raise DICOMProfileError(f"DICOM profile requires non-empty {name}.")
    return result


def _float(value: Any, name: str, /) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise DICOMProfileError(f"DICOM {name} must be finite.")
    return result


def _floats(value: Any, count: int, name: str, /) -> np.ndarray:
    result = np.asarray([float(item) for item in value], dtype=np.float64)
    if result.shape != (count,) or not np.all(np.isfinite(result)):
        raise DICOMProfileError(
            f"DICOM {name} must contain exactly {count} finite values."
        )
    return result


def _positive_int(value: Any, name: str, /) -> int:
    if isinstance(value, bool):
        raise DICOMProfileError(f"DICOM {name} must be a positive integer.")
    result = int(value)
    if result < 1 or float(value) != result:
        raise DICOMProfileError(f"DICOM {name} must be a positive integer.")
    return result


def _present(value: Any, /) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, bytes):
        return bool(value.strip(b"\x00 "))
    if isinstance(value, Sequence):
        return any(_present(item) for item in value)
    return True


def _walk_dataset(
    dataset: Any,
    /,
    *,
    depth: int = 1,
    max_depth: int,
    remaining_nodes: int,
    remaining_attributes: int,
) -> tuple[int, int, int]:
    if depth > max_depth:
        raise DICOMProfileError("DICOM sequence depth exceeds the resource policy.")
    maximum_depth = depth
    nodes = 0
    attributes = 0
    for element in dataset:
        if nodes >= remaining_nodes or attributes >= remaining_attributes:
            raise DICOMProfileError("DICOM element count exceeds the resource policy.")
        nodes += 1
        attributes += 1
        if element.tag.is_private:
            raise PermissionError(
                f"DICOM private metadata is refused at tag {element.tag}."
            )
        if element.keyword in _FORBIDDEN_PHI_KEYWORDS and _present(element.value):
            raise PermissionError(
                f"PHI refusal: DICOM attribute {element.keyword} is populated."
            )
        if (
            element.keyword == "BurnedInAnnotation"
            and _text(element.value, "BurnedInAnnotation").upper() != "NO"
        ):
            raise PermissionError("PHI refusal: burned-in annotations are not absent.")
        if (
            element.keyword == "RecognizableVisualFeatures"
            and _text(element.value, "RecognizableVisualFeatures").upper() != "NO"
        ):
            raise PermissionError(
                "PHI refusal: recognizable visual features are present."
            )
        if element.VR == "SQ":
            if element.value and depth >= max_depth:
                raise DICOMProfileError(
                    "DICOM sequence depth exceeds the resource policy."
                )
            for item in element.value:
                child_depth, child_nodes, child_attributes = _walk_dataset(
                    item,
                    depth=depth + 1,
                    max_depth=max_depth,
                    remaining_nodes=remaining_nodes - nodes,
                    remaining_attributes=remaining_attributes - attributes,
                )
                maximum_depth = max(maximum_depth, child_depth)
                nodes += child_nodes
                attributes += child_attributes
    return maximum_depth, nodes, attributes


def _identity(dataset: Any, /) -> DICOMObjectIdentity:
    frame = _optional(dataset, "FrameOfReferenceUID")
    return DICOMObjectIdentity(
        _text(_required(dataset, "SOPClassUID"), "SOPClassUID"),
        _text(_required(dataset, "SOPInstanceUID"), "SOPInstanceUID"),
        _text(_required(dataset, "StudyInstanceUID"), "StudyInstanceUID"),
        _text(_required(dataset, "SeriesInstanceUID"), "SeriesInstanceUID"),
        None if frame is None else _text(frame, "FrameOfReferenceUID"),
    )


def _file_transfer_syntax(dataset: Any, /) -> str:
    file_meta = dataset.file_meta
    if file_meta is None or "TransferSyntaxUID" not in file_meta:
        raise DICOMProfileError("DICOM file meta requires TransferSyntaxUID.")
    syntax = _text(file_meta["TransferSyntaxUID"].value, "TransferSyntaxUID")
    if syntax not in _SUPPORTED_TRANSFER_SYNTAXES:
        raise DICOMProfileError(
            f"Unsupported DICOM transfer syntax {syntax!r}; only uncompressed "
            "little-endian syntaxes are admitted."
        )
    return syntax


def _validate_resource_identity(resource: BoundedResource, /) -> None:
    manifest = resource.manifest
    if len(resource.data) != manifest.size_bytes:
        raise DICOMProfileError("Bounded DICOM resource size identity is inconsistent.")
    digest = hashlib.sha256(resource.data).hexdigest()
    if digest != manifest.content_sha256:
        raise DICOMProfileError("Bounded DICOM resource digest identity is inconsistent.")


def _load_verified_dicom(
    resources: Sequence[BoundedResource],
    references: Sequence[ReferenceArtifactManifest],
    deidentification: DeidentificationEvidence,
    policy: DICOMResourcePolicy,
    /,
) -> tuple[_LoadedDICOM, ...]:
    if not isinstance(deidentification, DeidentificationEvidence):
        raise TypeError("deidentification must be DeidentificationEvidence.")
    deidentification.require_research_ready()
    if not isinstance(policy, DICOMResourcePolicy):
        raise TypeError("policy must be DICOMResourcePolicy.")
    resources_ = tuple(resources)
    references_ = tuple(references)
    if not resources_ or any(
        not isinstance(item, BoundedResource) for item in resources_
    ):
        raise TypeError("resources must contain at least one BoundedResource.")
    if len(resources_) != len(references_) or any(
        not isinstance(item, ReferenceArtifactManifest) for item in references_
    ):
        raise TypeError(
            "references must contain one ReferenceArtifactManifest per resource."
        )
    if len(resources_) > policy.max_instances:
        raise DICOMProfileError("DICOM instance count exceeds the resource policy.")
    if sum(len(item.data) for item in resources_) > sum(
        item.manifest.limits.max_bytes for item in resources_
    ):
        raise DICOMProfileError("DICOM aggregate bytes exceed admitted resource bounds.")

    pydicom = _pydicom_module()
    loaded: list[_LoadedDICOM] = []
    observed_elements = 0
    for resource, reference in zip(resources_, references_, strict=True):
        _validate_resource_identity(resource)
        reference.verify_bytes(resource.data)
        header = pydicom.dcmread(
            BytesIO(resource.data), stop_before_pixels=True, force=False
        )
        _file_transfer_syntax(header)
        admitted_limits = resource.manifest.limits
        remaining_policy_elements = policy.max_elements - observed_elements
        depth, nodes, attributes = _walk_dataset(
            header,
            max_depth=min(policy.max_sequence_depth, admitted_limits.max_depth),
            remaining_nodes=min(remaining_policy_elements, admitted_limits.max_nodes),
            remaining_attributes=min(
                remaining_policy_elements, admitted_limits.max_attributes
            ),
        )
        observed_elements += nodes
        accounted = account_bounded_resource(
            resource,
            depth=depth,
            nodes=nodes,
            attributes=attributes,
            losses=0,
        )
        dataset = pydicom.dcmread(BytesIO(resource.data), force=False)
        _file_transfer_syntax(dataset)
        loaded.append(_LoadedDICOM(dataset, accounted, reference, _identity(dataset)))
    identities = [item.identity.sop_instance_uid for item in loaded]
    if len(identities) != len(set(identities)):
        raise DICOMProfileError("DICOM resources repeat a SOP Instance UID.")
    return tuple(loaded)


def _sequence_item(dataset: Any, keyword: str, /, *, single: bool = True) -> Any:
    sequence = _required(dataset, keyword)
    if not isinstance(sequence, Sequence) or isinstance(sequence, (str, bytes)):
        raise DICOMProfileError(f"DICOM {keyword} must be a sequence.")
    if not sequence or (single and len(sequence) != 1):
        count = "exactly one" if single else "at least one"
        raise DICOMProfileError(f"DICOM {keyword} must contain {count} item.")
    return sequence[0] if single else tuple(sequence)


def _references_in(
    dataset: Any, identity: DICOMObjectIdentity, /
) -> tuple[DICOMReference, ...]:
    references: list[DICOMReference] = []

    def visit(current: Any, path: tuple[str, ...]) -> None:
        if "ReferencedSOPInstanceUID" in current:
            target = _text(
                current["ReferencedSOPInstanceUID"].value,
                "ReferencedSOPInstanceUID",
            )
            target_class = _optional(current, "ReferencedSOPClassUID")
            references.append(
                DICOMReference(
                    identity.sop_instance_uid,
                    "/".join(path) if path else "ReferencedSOPInstanceUID",
                    target,
                    None
                    if target_class is None
                    else _text(target_class, "ReferencedSOPClassUID"),
                )
            )
        for element in current:
            if element.VR == "SQ":
                for index, item in enumerate(element.value):
                    visit(item, (*path, f"{element.keyword}[{index}]"))

    visit(dataset, ())
    unique = {item.reference_id: item for item in references}
    return tuple(sorted(unique.values(), key=lambda item: item.reference_id))


def _reference_graph(
    loaded: Sequence[_LoadedDICOM],
    /,
    *,
    linked_identities: Sequence[DICOMObjectIdentity] = (),
    require_closed: bool = True,
) -> DICOMReferenceGraph:
    linked = tuple(linked_identities)
    if any(not isinstance(item, DICOMObjectIdentity) for item in linked):
        raise TypeError("linked_identities must contain DICOMObjectIdentity values.")
    references = tuple(
        reference
        for item in loaded
        for reference in _references_in(item.dataset, item.identity)
    )
    graph = DICOMReferenceGraph(
        tuple(item.identity for item in loaded) + linked,
        references,
    )
    if require_closed:
        graph.require_closed()
    return graph


def _pixel_shape(dataset: Any, /) -> tuple[int, int, int, int]:
    rows = _positive_int(_required(dataset, "Rows"), "Rows")
    columns = _positive_int(_required(dataset, "Columns"), "Columns")
    frames = _positive_int(_optional(dataset, "NumberOfFrames", 1), "NumberOfFrames")
    samples = _positive_int(_optional(dataset, "SamplesPerPixel", 1), "SamplesPerPixel")
    if samples != 1:
        raise DICOMProfileError(
            "DICOM image profiles require one monochrome sample per pixel."
        )
    photometric = _text(
        _required(dataset, "PhotometricInterpretation"), "PhotometricInterpretation"
    )
    if photometric not in ("MONOCHROME1", "MONOCHROME2"):
        raise DICOMProfileError("DICOM image profiles require monochrome pixel data.")
    return frames, rows, columns, samples


def _pixel_budget(dataset: Any, policy: DICOMResourcePolicy, /) -> tuple[int, int]:
    frames, rows, columns, samples = _pixel_shape(dataset)
    bits = _positive_int(_required(dataset, "BitsAllocated"), "BitsAllocated")
    bits_stored = _positive_int(_required(dataset, "BitsStored"), "BitsStored")
    if bits not in (8, 16, 32) or bits_stored > bits:
        raise DICOMProfileError(
            "DICOM integer pixel storage is unsupported or inconsistent."
        )
    high_bit = int(_required(dataset, "HighBit"))
    if high_bit != bits_stored - 1:
        raise DICOMProfileError("DICOM HighBit and BitsStored are inconsistent.")
    representation = int(_required(dataset, "PixelRepresentation"))
    if representation not in (0, 1):
        raise DICOMProfileError(
            "DICOM PixelRepresentation must be unsigned or signed integer."
        )
    voxels = frames * rows * columns * samples
    pixel_bytes = voxels * (bits // 8)
    if (
        frames > policy.max_frames
        or rows > policy.max_rows
        or columns > policy.max_columns
        or voxels > policy.max_voxels
        or pixel_bytes > policy.max_total_pixel_bytes
    ):
        raise DICOMProfileError("DICOM pixel data exceed the resource policy.")
    if "PixelData" not in dataset:
        raise DICOMProfileError("DICOM image profile requires PixelData.")
    encoded = dataset["PixelData"].value
    if not isinstance(encoded, bytes) or len(encoded) not in (
        pixel_bytes,
        pixel_bytes + 1,
    ):
        raise DICOMProfileError("DICOM uncompressed PixelData length is inconsistent.")
    return voxels, pixel_bytes


def _decode_stored_pixels(dataset: Any, policy: DICOMResourcePolicy, /) -> np.ndarray:
    _pixel_budget(dataset, policy)
    array = np.asarray(dataset.pixel_array)
    frames, rows, columns, _ = _pixel_shape(dataset)
    expected = (rows, columns) if frames == 1 else (frames, rows, columns)
    if array.shape != expected or not np.issubdtype(array.dtype, np.integer):
        raise DICOMProfileError(
            "Decoded DICOM pixels have an unsupported shape or dtype."
        )
    return array


def _scaled_pixels(
    dataset: Any,
    stored: np.ndarray,
    /,
    *,
    slope: float | None = None,
    intercept: float | None = None,
) -> np.ndarray:
    slope_ = _float(
        _optional(dataset, "RescaleSlope", 1.0) if slope is None else slope,
        "RescaleSlope",
    )
    intercept_ = _float(
        _optional(dataset, "RescaleIntercept", 0.0) if intercept is None else intercept,
        "RescaleIntercept",
    )
    if slope_ == 0.0:
        raise DICOMProfileError("DICOM RescaleSlope must be nonzero.")
    result = np.asarray(stored, dtype=np.float64) * slope_ + intercept_
    if not np.all(np.isfinite(result)):
        raise DICOMProfileError("Scaled DICOM pixels must remain finite.")
    result.setflags(write=False)
    return result


def _orientation(value: Any, /) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors = _floats(value, 6, "ImageOrientationPatient")
    x = vectors[:3]
    y = vectors[3:]
    tolerance = 1.0e-6
    if (
        abs(float(np.linalg.norm(x)) - 1.0) > tolerance
        or abs(float(np.linalg.norm(y)) - 1.0) > tolerance
        or abs(float(np.dot(x, y))) > tolerance
    ):
        raise DICOMProfileError(
            "DICOM image orientation must be orthonormal and representable."
        )
    normal = np.cross(x, y)
    normal /= np.linalg.norm(normal)
    return x, y, normal


def _regular_geometry(
    positions: Sequence[Any],
    orientations: Sequence[Any],
    pixel_spacings: Sequence[Any],
    frame_of_reference_uid: str,
    /,
    *,
    single_slice_spacing: float | None = None,
) -> _Geometry:
    if (
        not positions
        or len(positions) != len(orientations)
        or len(positions) != len(pixel_spacings)
    ):
        raise DICOMProfileError("DICOM geometry must be present for every image plane.")
    position_arrays = tuple(
        _floats(value, 3, "ImagePositionPatient") for value in positions
    )
    orientation_arrays = tuple(_orientation(value) for value in orientations)
    spacing_arrays = tuple(_floats(value, 2, "PixelSpacing") for value in pixel_spacings)
    if any(np.any(value <= 0.0) for value in spacing_arrays):
        raise DICOMProfileError("DICOM PixelSpacing values must be positive.")
    x, y, normal = orientation_arrays[0]
    if any(
        not np.allclose(candidate[0], x, atol=1.0e-6, rtol=0.0)
        or not np.allclose(candidate[1], y, atol=1.0e-6, rtol=0.0)
        for candidate in orientation_arrays[1:]
    ):
        raise DICOMProfileError("DICOM plane orientations are not a regular grid.")
    pixel_spacing = spacing_arrays[0]
    if any(
        not np.allclose(candidate, pixel_spacing, atol=1.0e-6, rtol=0.0)
        for candidate in spacing_arrays[1:]
    ):
        raise DICOMProfileError("DICOM pixel spacings are not a regular grid.")
    locations = np.asarray([float(np.dot(value, normal)) for value in position_arrays])
    order_array = np.argsort(locations, kind="stable")
    sorted_locations = locations[order_array]
    base = position_arrays[int(order_array[0])]
    if len(position_arrays) == 1:
        if single_slice_spacing is None:
            raise DICOMProfileError(
                "A one-plane DICOM grid requires an explicit positive slice spacing."
            )
        slice_spacing = _float(single_slice_spacing, "slice spacing")
        if slice_spacing <= 0.0:
            raise DICOMProfileError("DICOM slice spacing must be positive.")
    else:
        deltas = np.diff(sorted_locations)
        tolerance = max(1.0e-5, 1.0e-5 * float(np.max(np.abs(sorted_locations))))
        if np.any(deltas <= tolerance) or not np.allclose(
            deltas, deltas[0], atol=tolerance, rtol=1.0e-5
        ):
            raise DICOMProfileError("DICOM slice locations are not a regular grid.")
        slice_spacing = float(deltas[0])
        for index, order_index in enumerate(order_array):
            expected = base + normal * slice_spacing * index
            if not np.allclose(
                position_arrays[int(order_index)], expected, atol=tolerance, rtol=0.0
            ):
                raise DICOMProfileError(
                    "DICOM planes contain an in-plane drift that one affine cannot represent."
                )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 0] = y * pixel_spacing[0]
    matrix[:3, 1] = x * pixel_spacing[1]
    matrix[:3, 2] = normal * slice_spacing
    matrix[:3, 3] = base
    contract = SpatialCoordinateContract(
        MILLIMETER,
        coordinate_system="cartesian-lps",
        reference_frame=f"dicom-frame-of-reference:{frame_of_reference_uid}",
    )
    affine = ImageIndexAffine(
        matrix,
        "dicom-voxel-centres",
        contract,
        ImageAxisConvention.LPS,
        VoxelReference.CENTER,
        "dicom-image-plane-patient",
    )
    return _Geometry(tuple(int(item) for item in order_array), affine)


def _report(
    profile: DICOMProfile,
    loaded: Sequence[_LoadedDICOM],
    graph: DICOMReferenceGraph,
    /,
    *,
    target_format: str,
    target_id: str,
    preserved_fields: Sequence[str],
    coordinate_mapping: Sequence[str] = (),
) -> DICOMImportReport:
    source_id = canonical_fingerprint(
        {
            "kind": "dicom-profile-source",
            "profile": profile,
            "resources": [item.resource.manifest.manifest_id for item in loaded],
            "graph": graph.graph_id,
        }
    )
    adapter = AdapterReport(
        AdapterStatus.LOSSLESS,
        "DICOM",
        target_format,
        source_id=source_id,
        target_id=target_id,
        coordinate_mapping=tuple(coordinate_mapping),
        preserved_fields=tuple(preserved_fields),
        stage=profile,
        source_profile=AdapterFormatProfile("DICOM", qualifiers={"profile": profile}),
        target_profile=AdapterFormatProfile(target_format),
    )
    return DICOMImportReport(
        profile,
        tuple(item.resource for item in loaded),
        graph,
        adapter,
    )


def _same_identity_context(loaded: Sequence[_LoadedDICOM], /) -> tuple[str, str, str]:
    studies = {item.identity.study_instance_uid for item in loaded}
    series = {item.identity.series_instance_uid for item in loaded}
    frames = {item.identity.frame_of_reference_uid for item in loaded}
    if len(studies) != 1 or len(series) != 1 or len(frames) != 1 or None in frames:
        raise DICOMProfileError(
            "DICOM image instances must share study, series, and frame-of-reference identity."
        )
    return studies.pop(), series.pop(), frames.pop()


__all__ = [
    "_LoadedDICOM",
    "_decode_stored_pixels",
    "_float",
    "_floats",
    "_load_verified_dicom",
    "_optional",
    "_pixel_budget",
    "_reference_graph",
    "_regular_geometry",
    "_report",
    "_required",
    "_same_identity_context",
    "_scaled_pixels",
    "_sequence_item",
    "_text",
]
