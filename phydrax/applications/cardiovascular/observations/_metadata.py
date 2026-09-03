#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Host-side identities for normalized cardiovascular observations.

This module deliberately stops at normalized arrays and metadata.  It does not
parse DICOM or NIfTI files, and it refuses assets whose de-identification and
usage rights have not been made explicit by the ingesting application.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from jaxtyping import ArrayLike

from ...._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_mapping,
)


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized or normalized != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return normalized


def _label(value: str, name: str, /) -> str:
    return _identifier(value, name)


def _readonly_array(
    value: ArrayLike,
    name: str,
    /,
    *,
    dtype: np.dtype[Any] | type[Any] | None = None,
) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must be a numerical or boolean array.")
    array.setflags(write=False)
    return array


def _readonly_float_array(value: ArrayLike, name: str, /) -> np.ndarray:
    array = np.asarray(value)
    dtype = np.result_type(array.dtype, np.float64)
    normalized = _readonly_array(array, name, dtype=dtype)
    if not np.issubdtype(normalized.dtype, np.floating):
        raise TypeError(f"{name} must have a real numerical dtype.")
    if not np.all(np.isfinite(normalized)):
        raise ValueError(f"{name} must be finite.")
    return normalized


class SpatialConvention(Enum):
    """Patient-coordinate convention for world-space millimetres."""

    LPS = "LPS"
    RAS = "RAS"


@dataclass(frozen=True, slots=True)
class SpatialFrame:
    """Named patient-coordinate frame with an explicit LPS or RAS convention."""

    frame_id: str
    convention: SpatialConvention
    unit: str = field(default="mm", init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_id", _identifier(self.frame_id, "frame_id"))
        if not isinstance(self.convention, SpatialConvention):
            raise TypeError("convention must be a SpatialConvention.")


@dataclass(frozen=True, slots=True)
class SpatialAffine:
    """Invertible voxel-index-to-world affine in millimetres.

    Points use a final coordinate axis of length three.  The affine maps array
    indices into ``target_frame`` coordinates; ``source_frame_id`` names the
    discrete index frame rather than pretending that indices are physical axes.
    """

    matrix_mm: np.ndarray
    source_frame_id: str
    target_frame: SpatialFrame
    provenance: str = "normalized"
    affine_id: str = field(init=False)

    def __post_init__(self) -> None:
        matrix = _readonly_float_array(self.matrix_mm, "matrix_mm")
        if matrix.shape != (4, 4):
            raise ValueError("matrix_mm must have shape (4, 4).")
        tolerance = 32.0 * np.finfo(matrix.dtype).eps
        if not np.allclose(
            matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0]), atol=tolerance, rtol=0.0
        ):
            raise ValueError("matrix_mm must have homogeneous final row [0, 0, 0, 1].")
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not np.isfinite(determinant) or abs(determinant) <= tolerance:
            raise ValueError("matrix_mm must have an invertible spatial block.")
        if not isinstance(self.target_frame, SpatialFrame):
            raise TypeError("target_frame must be a SpatialFrame.")
        source = _identifier(self.source_frame_id, "source_frame_id")
        provenance = _label(self.provenance, "provenance")
        object.__setattr__(self, "matrix_mm", matrix)
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "affine_id",
            canonical_fingerprint(
                {
                    "kind": "cardiovascular-spatial-affine",
                    "source_frame_id": source,
                    "target_frame_id": self.target_frame.frame_id,
                    "target_convention": self.target_frame.convention.value,
                    "provenance": provenance,
                    "matrix_mm": array_tree_fingerprint(matrix),
                }
            ),
        )

    @classmethod
    def from_qform_sform(
        cls,
        *,
        qform_mm: ArrayLike | None,
        sform_mm: ArrayLike | None,
        source_frame_id: str,
        target_frame: SpatialFrame,
        conflict_tolerance_mm: float = 1.0e-5,
    ) -> "SpatialAffine":
        """Resolve normalized qform/sform matrices, refusing disagreement.

        This consumes already-normalized affine matrices; it is not a NIfTI
        header parser.  When both forms are present their elementwise agreement
        is checked before the sform is selected.
        """
        tolerance = float(conflict_tolerance_mm)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conflict_tolerance_mm must be finite and non-negative.")
        if qform_mm is None and sform_mm is None:
            raise ValueError("At least one of qform_mm or sform_mm is required.")
        qform = None if qform_mm is None else _readonly_float_array(qform_mm, "qform_mm")
        sform = None if sform_mm is None else _readonly_float_array(sform_mm, "sform_mm")
        for name, matrix in (("qform_mm", qform), ("sform_mm", sform)):
            if matrix is not None and matrix.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4).")
        if qform is not None and sform is not None:
            if not np.allclose(qform, sform, atol=tolerance, rtol=0.0):
                maximum = float(np.max(np.abs(qform - sform)))
                raise ValueError(
                    "qform/sform conflict: normalized spatial affines disagree "
                    f"by up to {maximum:.6g} mm."
                )
            return cls(
                sform,
                source_frame_id,
                target_frame,
                provenance="qform+sform-agree",
            )
        if sform is not None:
            return cls(sform, source_frame_id, target_frame, provenance="sform")
        if qform is None:
            raise RuntimeError("Affine resolution reached an impossible state.")
        return cls(qform, source_frame_id, target_frame, provenance="qform")

    def index_to_world(self, points: ArrayLike, /) -> np.ndarray:
        coordinates = np.asarray(points, dtype=self.matrix_mm.dtype)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError(
                "Index points must end with a coordinate axis of length three."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Index points must be finite.")
        return coordinates @ self.matrix_mm[:3, :3].T + self.matrix_mm[:3, 3]

    def world_to_index(self, points: ArrayLike, /) -> np.ndarray:
        coordinates = np.asarray(points, dtype=self.matrix_mm.dtype)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError(
                "World points must end with a coordinate axis of length three."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("World points must be finite.")
        inverse = np.linalg.inv(self.matrix_mm[:3, :3])
        return (coordinates - self.matrix_mm[:3, 3]) @ inverse.T

    def to_convention(self, target_frame: SpatialFrame, /) -> "SpatialAffine":
        """Express the same world geometry in another LPS/RAS frame."""
        if not isinstance(target_frame, SpatialFrame):
            raise TypeError("target_frame must be a SpatialFrame.")
        if target_frame.convention is self.target_frame.convention:
            conversion = np.eye(4, dtype=self.matrix_mm.dtype)
        else:
            conversion = np.diag(np.asarray([-1.0, -1.0, 1.0, 1.0]))
        return SpatialAffine(
            conversion @ self.matrix_mm,
            self.source_frame_id,
            target_frame,
            provenance=f"{self.provenance}:{self.target_frame.convention.value}-to-{target_frame.convention.value}",
        )


@dataclass(frozen=True, slots=True)
class TimeBase:
    """Explicit observation sample times in the cardiovascular kernel unit ms."""

    timebase_id: str
    sample_times_ms: np.ndarray

    def __post_init__(self) -> None:
        identifier = _identifier(self.timebase_id, "timebase_id")
        times = _readonly_float_array(self.sample_times_ms, "sample_times_ms")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("sample_times_ms must be a non-empty rank-one array.")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("sample_times_ms must be strictly increasing.")
        object.__setattr__(self, "timebase_id", identifier)
        object.__setattr__(self, "sample_times_ms", times)

    @classmethod
    def uniform(
        cls,
        timebase_id: str,
        sample_count: int,
        interval_ms: float,
        /,
        *,
        origin_ms: float = 0.0,
    ) -> "TimeBase":
        if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
            raise TypeError("sample_count must be an integer.")
        count = int(sample_count)
        interval = float(interval_ms)
        origin = float(origin_ms)
        if count < 1:
            raise ValueError("sample_count must be positive.")
        if not np.isfinite(interval) or interval <= 0.0:
            raise ValueError("interval_ms must be finite and positive.")
        if not np.isfinite(origin):
            raise ValueError("origin_ms must be finite.")
        return cls(timebase_id, origin + interval * np.arange(count, dtype=float))

    @property
    def sample_count(self) -> int:
        return int(self.sample_times_ms.size)

    @property
    def is_uniform(self) -> bool:
        if self.sample_count <= 2:
            return True
        intervals = np.diff(self.sample_times_ms)
        scale = max(1.0, float(np.max(np.abs(intervals))))
        tolerance = 32.0 * np.finfo(intervals.dtype).eps * scale
        return bool(np.all(np.abs(intervals - intervals[0]) <= tolerance))

    @property
    def interval_ms(self) -> float | None:
        if self.sample_count < 2 or not self.is_uniform:
            return None
        return float(self.sample_times_ms[1] - self.sample_times_ms[0])

    @property
    def duration_ms(self) -> float:
        return float(self.sample_times_ms[-1] - self.sample_times_ms[0])


@dataclass(frozen=True, slots=True)
class DeidentificationIdentity:
    """Auditable assertion that a host asset is safe to admit for research."""

    identity_id: str
    pseudonymous_subject_id: str
    protocol_id: str
    direct_identifiers_removed: bool
    burned_in_annotations_removed: bool
    facial_features_removed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_id",
            _identifier(self.identity_id, "identity_id"),
        )
        object.__setattr__(
            self,
            "pseudonymous_subject_id",
            _identifier(self.pseudonymous_subject_id, "pseudonymous_subject_id"),
        )
        object.__setattr__(
            self,
            "protocol_id",
            _identifier(self.protocol_id, "protocol_id"),
        )
        if not isinstance(self.direct_identifiers_removed, bool):
            raise TypeError("direct_identifiers_removed must be boolean.")
        if not isinstance(self.burned_in_annotations_removed, bool):
            raise TypeError("burned_in_annotations_removed must be boolean.")
        if not isinstance(self.facial_features_removed, bool):
            raise TypeError("facial_features_removed must be boolean.")

    @property
    def research_ready(self) -> bool:
        return (
            self.direct_identifiers_removed
            and self.burned_in_annotations_removed
            and self.facial_features_removed
        )

    def require_research_ready(self) -> None:
        if not self.research_ready:
            raise PermissionError(
                "PHI refusal: the asset lacks complete direct-identifier, burned-in "
                "annotation, or facial-feature de-identification evidence."
            )


@dataclass(frozen=True, slots=True)
class DataRightsIdentity:
    """Stable, explicit rights grant attached to one normalized data lineage."""

    rights_id: str
    license_id: str
    permitted_uses: tuple[str, ...]
    data_controller: str
    redistribution_allowed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "rights_id",
            _identifier(self.rights_id, "rights_id"),
        )
        object.__setattr__(
            self,
            "license_id",
            _identifier(self.license_id, "license_id"),
        )
        object.__setattr__(
            self,
            "data_controller",
            _identifier(self.data_controller, "data_controller"),
        )
        uses = tuple(_label(value, "permitted use") for value in self.permitted_uses)
        if not uses or len(set(uses)) != len(uses):
            raise ValueError("permitted_uses must be non-empty and unique.")
        if not isinstance(self.redistribution_allowed, bool):
            raise TypeError("redistribution_allowed must be boolean.")
        object.__setattr__(self, "permitted_uses", uses)

    def permits(self, intended_use: str, /) -> bool:
        return _label(intended_use, "intended_use") in self.permitted_uses

    def require(self, intended_use: str, /) -> None:
        use = _label(intended_use, "intended_use")
        if use not in self.permitted_uses:
            raise PermissionError(
                f"Data-rights refusal: use {use!r} is not granted by {self.rights_id!r}."
            )


_FORBIDDEN_PHI_KEYS = frozenset(
    {
        "accessionnumber",
        "address",
        "birthdate",
        "dateofbirth",
        "institutionaddress",
        "medicalrecordnumber",
        "mrn",
        "patientbirthdate",
        "patientid",
        "patientname",
        "phonenumber",
        "referringphysician",
        "socialsecuritynumber",
    }
)


def _canonical_key(value: str, /) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _find_phi_key(value: Any, path: str = "metadata", /) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}"
            if _canonical_key(key_text) in _FORBIDDEN_PHI_KEYS:
                return child_path
            found = _find_phi_key(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _find_phi_key(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


@dataclass(frozen=True, slots=True)
class MedicalImageAsset:
    """Rights-checked normalized medical image arrays and coordinate metadata."""

    asset_id: str
    modality: str
    values: np.ndarray
    spatial_affine: SpatialAffine
    timebase: TimeBase | None
    deidentification: DeidentificationIdentity
    data_rights: DataRightsIdentity
    quantity: str
    unit: str
    valid_mask: np.ndarray | None = None
    metadata: Mapping[str, Any] | None = None
    intended_use: str = "research"
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        asset_id = _identifier(self.asset_id, "asset_id")
        modality = _label(self.modality, "modality")
        quantity = _label(self.quantity, "quantity")
        unit = _label(self.unit, "unit")
        intended_use = _label(self.intended_use, "intended_use")
        if not isinstance(self.spatial_affine, SpatialAffine):
            raise TypeError("spatial_affine must be a SpatialAffine.")
        if self.timebase is not None and not isinstance(self.timebase, TimeBase):
            raise TypeError("timebase must be a TimeBase or None.")
        if not isinstance(self.deidentification, DeidentificationIdentity):
            raise TypeError("deidentification must be a DeidentificationIdentity.")
        if not isinstance(self.data_rights, DataRightsIdentity):
            raise TypeError("data_rights must be a DataRightsIdentity.")
        self.deidentification.require_research_ready()
        self.data_rights.require(intended_use)

        values = _readonly_array(self.values, "values")
        if values.ndim < 3 or not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                "Medical image values must be a numerical array of rank at least three."
            )
        mask = (
            np.ones(values.shape, dtype=bool)
            if self.valid_mask is None
            else np.asarray(self.valid_mask, dtype=bool)
        )
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as values.")
        if not np.all(np.isfinite(values[mask])):
            raise ValueError(
                "Medical image values must be finite wherever valid_mask is true."
            )
        mask = _readonly_array(mask, "valid_mask", dtype=bool)

        metadata = canonical_mapping({} if self.metadata is None else self.metadata)
        phi_key = _find_phi_key(metadata)
        if phi_key is not None:
            raise PermissionError(
                f"PHI refusal: forbidden direct-identifier metadata key {phi_key!r}."
            )
        content_id = canonical_fingerprint(
            {
                "kind": "cardiovascular-medical-image-asset",
                "asset_id": asset_id,
                "modality": modality,
                "quantity": quantity,
                "unit": unit,
                "affine_id": self.spatial_affine.affine_id,
                "timebase_id": None
                if self.timebase is None
                else self.timebase.timebase_id,
                "deidentification_id": self.deidentification.identity_id,
                "rights_id": self.data_rights.rights_id,
                "metadata": metadata,
                "values": array_tree_fingerprint(values),
                "valid_mask": array_tree_fingerprint(mask),
            }
        )
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "intended_use", intended_use)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "content_id", content_id)


@dataclass(frozen=True, slots=True)
class ObservationRecord:
    """Normalized host observation channel consumed by personalization adapters."""

    record_id: str
    modality: str
    values: np.ndarray
    valid_mask: np.ndarray
    quantity: str
    unit: str
    frame_id: str | None = None
    timebase_id: str | None = None
    asset_id: str | None = None

    def __post_init__(self) -> None:
        record_id = _identifier(self.record_id, "record_id")
        modality = _label(self.modality, "modality")
        quantity = _label(self.quantity, "quantity")
        unit = _label(self.unit, "unit")
        values = _readonly_array(self.values, "values")
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Observation values must have a numerical dtype.")
        mask = _readonly_array(self.valid_mask, "valid_mask", dtype=bool)
        if mask.shape != values.shape:
            raise ValueError("valid_mask must have the same shape as values.")
        if not np.all(np.isfinite(values[mask])):
            raise ValueError(
                "Observation values must be finite wherever valid_mask is true."
            )
        if self.frame_id is not None:
            object.__setattr__(
                self,
                "frame_id",
                _identifier(self.frame_id, "frame_id"),
            )
        if self.timebase_id is not None:
            object.__setattr__(
                self,
                "timebase_id",
                _identifier(self.timebase_id, "timebase_id"),
            )
        if self.asset_id is not None:
            object.__setattr__(
                self,
                "asset_id",
                _identifier(self.asset_id, "asset_id"),
            )
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)


__all__ = [
    "DataRightsIdentity",
    "DeidentificationIdentity",
    "MedicalImageAsset",
    "ObservationRecord",
    "SpatialAffine",
    "SpatialConvention",
    "SpatialFrame",
    "TimeBase",
]
