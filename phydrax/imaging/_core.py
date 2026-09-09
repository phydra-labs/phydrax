#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Immutable host-side contracts for physically located medical images."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from numbers import Integral
from types import MappingProxyType
from typing import Any

import numpy as np
from jaxtyping import ArrayLike

from .._fingerprint import (
    array_tree_fingerprint,
    canonical_fingerprint,
    canonical_mapping,
)
from .._physical import SpatialCoordinateContract
from ..qualification import ReferenceArtifactManifest
from ..units import conversion_factor, TIME, UnitDefinition


HostMetadataValue = str | int | float | bool


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result or result != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return result


def _readonly(value: ArrayLike, name: str, /, *, dtype=None) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    if array.dtype.hasobject:
        raise TypeError(f"{name} must not use an object dtype.")
    array.setflags(write=False)
    return array


def _real(value: ArrayLike, name: str, /) -> np.ndarray:
    original = np.asarray(value)
    array = _readonly(value, name, dtype=np.result_type(original.dtype, np.float64))
    if not np.issubdtype(array.dtype, np.floating) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite and real-valued.")
    return array


class ImageAxisConvention(StrEnum):
    LPS = "LPS"
    RAS = "RAS"


class VoxelReference(StrEnum):
    CENTER = "center"
    CORNER = "corner"


class ImageValueKind(StrEnum):
    SCALAR = "scalar"
    VECTOR = "vector"
    COVECTOR = "covector"
    SYMMETRIC_TENSOR = "symmetric_tensor"
    CATEGORICAL = "categorical"
    PROBABILITY = "probability"


@dataclass(frozen=True, slots=True)
class ImageIndexAffine:
    """Invertible voxel-index-to-world affine in an exact physical frame."""

    matrix: np.ndarray
    source_frame_id: str
    coordinate_contract: SpatialCoordinateContract
    axis_convention: ImageAxisConvention
    voxel_reference: VoxelReference = VoxelReference.CENTER
    provenance: str = "normalized"
    affine_id: str = field(init=False)

    def __post_init__(self) -> None:
        matrix = _real(self.matrix, "matrix")
        if matrix.shape != (4, 4):
            raise ValueError("matrix must have shape (4, 4).")
        tolerance = 64.0 * np.finfo(matrix.dtype).eps
        if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=tolerance, rtol=0.0):
            raise ValueError("matrix must have homogeneous final row [0, 0, 0, 1].")
        determinant = float(np.linalg.det(matrix[:3, :3]))
        if not np.isfinite(determinant) or abs(determinant) <= tolerance:
            raise ValueError("matrix must have an invertible spatial block.")
        if not isinstance(self.coordinate_contract, SpatialCoordinateContract):
            raise TypeError("coordinate_contract must be SpatialCoordinateContract.")
        if not isinstance(self.axis_convention, ImageAxisConvention):
            raise TypeError("axis_convention must be ImageAxisConvention.")
        expected_system = f"cartesian-{self.axis_convention.value.lower()}"
        if self.coordinate_contract.coordinate_system != expected_system:
            raise ValueError(
                "Image axis convention and spatial coordinate system disagree."
            )
        if not isinstance(self.voxel_reference, VoxelReference):
            raise TypeError("voxel_reference must be VoxelReference.")
        source = _identifier(self.source_frame_id, "source_frame_id")
        provenance = _identifier(self.provenance, "provenance")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "source_frame_id", source)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(
            self,
            "affine_id",
            canonical_fingerprint(
                {
                    "kind": "image-index-affine",
                    "matrix": array_tree_fingerprint(matrix),
                    "source_frame": source,
                    "spatial_contract": self.coordinate_contract.spatial_id,
                    "axis_convention": self.axis_convention.value,
                    "voxel_reference": self.voxel_reference.value,
                    "provenance": provenance,
                }
            ),
        )

    @classmethod
    def from_qform_sform(
        cls,
        *,
        qform: ArrayLike | None,
        sform: ArrayLike | None,
        source_frame_id: str,
        coordinate_contract: SpatialCoordinateContract,
        axis_convention: ImageAxisConvention,
        voxel_reference: VoxelReference = VoxelReference.CENTER,
        conflict_tolerance: float = 1.0e-5,
    ) -> ImageIndexAffine:
        tolerance = float(conflict_tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise ValueError("conflict_tolerance must be finite and non-negative.")
        if qform is None and sform is None:
            raise ValueError("At least one of qform or sform is required.")
        q = None if qform is None else _real(qform, "qform")
        s = None if sform is None else _real(sform, "sform")
        for name, value in (("qform", q), ("sform", s)):
            if value is not None and value.shape != (4, 4):
                raise ValueError(f"{name} must have shape (4, 4).")
        if q is not None and s is not None:
            if not np.allclose(q, s, atol=tolerance, rtol=0.0):
                maximum = float(np.max(np.abs(q - s)))
                raise ValueError(
                    f"qform/sform conflict: maximum difference is {maximum:.6g}."
                )
            matrix, provenance = s, "qform+sform-agree"
        elif s is not None:
            matrix, provenance = s, "sform"
        else:
            if q is None:
                raise RuntimeError("Affine resolution reached an impossible state.")
            matrix, provenance = q, "qform"
        return cls(
            matrix,
            source_frame_id,
            coordinate_contract,
            axis_convention,
            voxel_reference,
            provenance,
        )

    def index_to_world(self, points: ArrayLike, /) -> np.ndarray:
        coordinates = np.asarray(points, dtype=self.matrix.dtype)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError(
                "Index points must end with a coordinate axis of length three."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("Index points must be finite.")
        return coordinates @ self.matrix[:3, :3].T + self.matrix[:3, 3]

    def world_to_index(self, points: ArrayLike, /) -> np.ndarray:
        coordinates = np.asarray(points, dtype=self.matrix.dtype)
        if coordinates.ndim < 1 or coordinates.shape[-1] != 3:
            raise ValueError(
                "World points must end with a coordinate axis of length three."
            )
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("World points must be finite.")
        inverse = np.linalg.inv(self.matrix[:3, :3])
        return (coordinates - self.matrix[:3, 3]) @ inverse.T

    def to_convention(self, convention: ImageAxisConvention, /) -> ImageIndexAffine:
        if not isinstance(convention, ImageAxisConvention):
            raise TypeError("convention must be ImageAxisConvention.")
        if convention is self.axis_convention:
            return self
        conversion = np.diag(np.asarray((-1.0, -1.0, 1.0, 1.0)))
        contract = SpatialCoordinateContract(
            self.coordinate_contract.length_unit,
            length_coordinate_kind=self.coordinate_contract.length_coordinate_kind,
            coordinate_system=f"cartesian-{convention.value.lower()}",
            reference_frame=self.coordinate_contract.reference_frame,
        )
        return ImageIndexAffine(
            conversion @ self.matrix,
            self.source_frame_id,
            contract,
            convention,
            self.voxel_reference,
            f"{self.provenance}:{self.axis_convention.value}-to-{convention.value}",
        )

    def to_unit(self, unit: UnitDefinition, /) -> ImageIndexAffine:
        if not isinstance(unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        factor = float(conversion_factor(self.coordinate_contract.length_unit, unit))
        if factor == 1.0:
            return self
        scale = np.diag(np.asarray((factor, factor, factor, 1.0)))
        contract = SpatialCoordinateContract(
            unit,
            length_coordinate_kind=self.coordinate_contract.length_coordinate_kind,
            coordinate_system=self.coordinate_contract.coordinate_system,
            reference_frame=self.coordinate_contract.reference_frame,
        )
        return ImageIndexAffine(
            scale @ self.matrix,
            self.source_frame_id,
            contract,
            self.axis_convention,
            self.voxel_reference,
            f"{self.provenance}:unit:{unit.unit_id}",
        )


@dataclass(frozen=True, slots=True)
class ImageTimeAxis:
    time_axis_id: str
    axis_label: str = field(init=False)
    sample_times: np.ndarray
    time_unit: UnitDefinition
    basis: str = "relative"
    origin: str | None = None

    def __post_init__(self) -> None:
        identifier = _identifier(self.time_axis_id, "time_axis_id")
        times = _real(self.sample_times, "sample_times")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("sample_times must be a non-empty rank-one array.")
        if times.size > 1 and not np.all(np.diff(times) > 0.0):
            raise ValueError("sample_times must be strictly increasing.")
        if (
            not isinstance(self.time_unit, UnitDefinition)
            or self.time_unit.dimension != TIME
        ):
            raise ValueError("time_unit must be a time UnitDefinition.")
        basis = _identifier(self.basis, "basis")
        if basis not in ("relative", "absolute"):
            raise ValueError("basis must be 'relative' or 'absolute'.")
        if basis == "absolute" and self.origin is None:
            raise ValueError("Absolute image time axes require an origin.")
        origin = None if self.origin is None else _identifier(self.origin, "origin")
        object.__setattr__(self, "axis_label", identifier)
        object.__setattr__(
            self,
            "time_axis_id",
            canonical_fingerprint(
                {
                    "kind": "image-time-axis",
                    "label": identifier,
                    "samples": array_tree_fingerprint(times),
                    "unit": self.time_unit.unit_id,
                    "basis": basis,
                    "origin": origin,
                }
            ),
        )
        object.__setattr__(self, "sample_times", times)
        object.__setattr__(self, "basis", basis)
        object.__setattr__(self, "origin", origin)

    @classmethod
    def uniform(
        cls,
        time_axis_id: str,
        sample_count: int,
        interval: float,
        time_unit: UnitDefinition,
        /,
        *,
        origin: float = 0.0,
    ) -> ImageTimeAxis:
        if isinstance(sample_count, bool) or not isinstance(sample_count, Integral):
            raise TypeError("sample_count must be an integer.")
        count, width, start = int(sample_count), float(interval), float(origin)
        if count < 1 or not np.isfinite(width) or width <= 0.0 or not np.isfinite(start):
            raise ValueError("sample_count and interval must be positive and finite.")
        return cls(time_axis_id, start + width * np.arange(count), time_unit)

    @property
    def sample_count(self) -> int:
        return int(self.sample_times.size)

    @property
    def is_uniform(self) -> bool:
        if self.sample_count <= 2:
            return True
        delta = np.diff(self.sample_times)
        tolerance = (
            64.0 * np.finfo(delta.dtype).eps * max(1.0, float(np.max(np.abs(delta))))
        )
        return bool(np.all(np.abs(delta - delta[0]) <= tolerance))

    def values_in(self, unit: UnitDefinition, /) -> np.ndarray:
        return self.sample_times * float(conversion_factor(self.time_unit, unit))

    def interval_in(self, unit: UnitDefinition, /) -> float | None:
        if self.sample_count < 2 or not self.is_uniform:
            return None
        return float(self.values_in(unit)[1] - self.values_in(unit)[0])

    def duration_in(self, unit: UnitDefinition, /) -> float:
        values = self.values_in(unit)
        return float(values[-1] - values[0])


@dataclass(frozen=True, slots=True)
class ImageAcquisitionIdentity:
    acquisition_id: str
    series_id: str
    modality: str
    protocol_id: str
    identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        values = tuple(
            _identifier(value, name)
            for value, name in zip(
                (self.acquisition_id, self.series_id, self.modality, self.protocol_id),
                ("acquisition_id", "series_id", "modality", "protocol_id"),
                strict=True,
            )
        )
        for name, value in zip(
            ("acquisition_id", "series_id", "modality", "protocol_id"),
            values,
            strict=True,
        ):
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint({"kind": "image-acquisition", "values": list(values)}),
        )


@dataclass(frozen=True, slots=True)
class DeidentificationEvidence:
    evidence_id: str
    pseudonymous_subject_id: str
    protocol_id: str
    direct_identifiers_removed: bool
    burned_in_annotations_removed: bool
    facial_features_removed: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _identifier(self.evidence_id, "evidence_id")
        )
        object.__setattr__(
            self,
            "pseudonymous_subject_id",
            _identifier(self.pseudonymous_subject_id, "pseudonymous_subject_id"),
        )
        object.__setattr__(
            self, "protocol_id", _identifier(self.protocol_id, "protocol_id")
        )
        if not all(
            isinstance(value, bool)
            for value in (
                self.direct_identifiers_removed,
                self.burned_in_annotations_removed,
                self.facial_features_removed,
            )
        ):
            raise TypeError("De-identification completion fields must be boolean.")

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
                "PHI refusal: de-identification evidence is incomplete."
            )


@dataclass(frozen=True, slots=True)
class ImageValueLayout:
    quantity: str
    unit: UnitDefinition
    kind: ImageValueKind
    component_shape: tuple[int, ...] = ()
    component_frame_id: str | None = None
    layout_id: str = field(init=False)

    def __post_init__(self) -> None:
        quantity = _identifier(self.quantity, "quantity")
        if not isinstance(self.unit, UnitDefinition):
            raise TypeError("unit must be UnitDefinition.")
        if not isinstance(self.kind, ImageValueKind):
            raise TypeError("kind must be ImageValueKind.")
        component_shape = tuple(int(size) for size in self.component_shape)
        if any(size <= 0 for size in component_shape):
            raise ValueError("component_shape entries must be positive.")
        expected = {
            ImageValueKind.SCALAR: (),
            ImageValueKind.CATEGORICAL: (),
            ImageValueKind.VECTOR: (3,),
            ImageValueKind.COVECTOR: (3,),
            ImageValueKind.SYMMETRIC_TENSOR: (3, 3),
        }
        if self.kind in expected and component_shape != expected[self.kind]:
            raise ValueError(
                f"{self.kind.value} requires component shape {expected[self.kind]}."
            )
        if self.kind is ImageValueKind.PROBABILITY and (
            len(component_shape) != 1 or component_shape[0] < 2
        ):
            raise ValueError(
                "Probability layouts require one class axis of size at least two."
            )
        frame = (
            None
            if self.component_frame_id is None
            else _identifier(self.component_frame_id, "component_frame_id")
        )
        if (
            self.kind
            in (
                ImageValueKind.VECTOR,
                ImageValueKind.COVECTOR,
                ImageValueKind.SYMMETRIC_TENSOR,
            )
            and frame is None
        ):
            raise ValueError("Geometric component layouts require component_frame_id.")
        object.__setattr__(self, "quantity", quantity)
        object.__setattr__(self, "component_shape", component_shape)
        object.__setattr__(self, "component_frame_id", frame)
        object.__setattr__(
            self,
            "layout_id",
            canonical_fingerprint(
                {
                    "kind": "image-value-layout",
                    "quantity": quantity,
                    "unit": self.unit.unit_id,
                    "value_kind": self.kind.value,
                    "component_shape": list(component_shape),
                    "component_frame": frame,
                }
            ),
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


def _phi_path(value: Any, path: str = "metadata", /) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            canonical = "".join(
                character for character in key_text.lower() if character.isalnum()
            )
            child_path = f"{path}.{key_text}"
            if canonical in _FORBIDDEN_PHI_KEYS:
                return child_path
            found = _phi_path(child, child_path)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, child in enumerate(value):
            found = _phi_path(child, f"{path}[{index}]")
            if found is not None:
                return found
    return None


@dataclass(frozen=True, slots=True)
class MedicalImageAsset:
    asset_id: str
    modality: str
    values: np.ndarray
    spatial_affine: ImageIndexAffine
    layout: ImageValueLayout
    deidentification: DeidentificationEvidence
    reference: ReferenceArtifactManifest
    time_axis: ImageTimeAxis | None = None
    valid_mask: np.ndarray | None = None
    acquisition: ImageAcquisitionIdentity | None = None
    metadata: Mapping[str, Any] | None = None
    intended_use: str = "research"
    content_id: str = field(init=False)

    def __post_init__(self) -> None:
        asset_id = _identifier(self.asset_id, "asset_id")
        modality = _identifier(self.modality, "modality")
        intended_use = _identifier(self.intended_use, "intended_use")
        if not isinstance(self.spatial_affine, ImageIndexAffine):
            raise TypeError("spatial_affine must be ImageIndexAffine.")
        if not isinstance(self.layout, ImageValueLayout):
            raise TypeError("layout must be ImageValueLayout.")
        if not isinstance(self.deidentification, DeidentificationEvidence):
            raise TypeError("deidentification must be DeidentificationEvidence.")
        if not isinstance(self.reference, ReferenceArtifactManifest):
            raise TypeError("reference must be ReferenceArtifactManifest.")
        if self.time_axis is not None and not isinstance(self.time_axis, ImageTimeAxis):
            raise TypeError("time_axis must be ImageTimeAxis or None.")
        if self.acquisition is not None and not isinstance(
            self.acquisition, ImageAcquisitionIdentity
        ):
            raise TypeError("acquisition must be ImageAcquisitionIdentity or None.")
        self.deidentification.require_research_ready()
        requested = {
            "research": {},
            "commercial": {"commercial_use": True},
            "training": {"training_use": True},
            "redistribution": {"redistribution": True},
            "export": {"export": True},
        }
        if intended_use not in requested:
            raise ValueError(
                "intended_use must be research, commercial, training, redistribution, or export."
            )
        self.reference.require_rights(**requested[intended_use])
        values = _readonly(self.values, "values")
        if values.ndim < 3 or not np.issubdtype(values.dtype, np.number):
            raise ValueError(
                "Medical image values must be a numerical array of rank at least three."
            )
        sample_shape = values.shape[:3] + (
            () if self.time_axis is None else (self.time_axis.sample_count,)
        )
        expected = sample_shape + self.layout.component_shape
        if values.shape != expected:
            raise ValueError(
                f"Image values must have shape {expected}; got {values.shape}."
            )
        mask = (
            np.ones(sample_shape, dtype=bool)
            if self.valid_mask is None
            else np.asarray(self.valid_mask, dtype=bool)
        )
        if mask.shape != sample_shape:
            raise ValueError(f"valid_mask must have sample shape {sample_shape}.")
        expanded = mask.reshape(mask.shape + (1,) * len(self.layout.component_shape))
        if not np.all(np.where(expanded, np.isfinite(values), True)):
            raise ValueError(
                "Medical image values must be finite wherever valid_mask is true."
            )
        if self.layout.kind is ImageValueKind.CATEGORICAL and not np.issubdtype(
            values.dtype, np.integer
        ):
            raise TypeError("Categorical medical images require integer storage.")
        if self.layout.kind is ImageValueKind.PROBABILITY and not np.issubdtype(
            values.dtype, np.floating
        ):
            raise TypeError("Probability medical images require floating-point storage.")
        if self.layout.kind is ImageValueKind.PROBABILITY:
            selected = values[mask]
            tolerance = 128.0 * np.finfo(values.dtype).eps
            if np.any(selected < -tolerance) or not np.allclose(
                np.sum(selected, axis=-1), 1.0, atol=tolerance, rtol=0.0
            ):
                raise ValueError(
                    "Probability medical images must be non-negative and sum to one."
                )
        mask = _readonly(mask, "valid_mask", dtype=bool)
        metadata = canonical_mapping({} if self.metadata is None else self.metadata)
        forbidden = _phi_path(metadata)
        if forbidden is not None:
            raise PermissionError(f"PHI refusal: forbidden metadata key {forbidden!r}.")
        object.__setattr__(self, "asset_id", asset_id)
        object.__setattr__(self, "modality", modality)
        object.__setattr__(self, "intended_use", intended_use)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(
            self,
            "content_id",
            canonical_fingerprint(
                {
                    "kind": "medical-image-asset",
                    "asset": asset_id,
                    "modality": modality,
                    "affine": self.spatial_affine.affine_id,
                    "layout": self.layout.layout_id,
                    "time": None
                    if self.time_axis is None
                    else self.time_axis.time_axis_id,
                    "deidentification": self.deidentification.evidence_id,
                    "reference": self.reference.manifest_id,
                    "acquisition": None
                    if self.acquisition is None
                    else self.acquisition.identity_id,
                    "metadata": metadata,
                    "values": array_tree_fingerprint(values),
                    "mask": array_tree_fingerprint(mask),
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class LabelDefinition:
    value: int
    label_id: str
    name: str
    parent_id: str | None = None
    roles: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isinstance(self.value, Integral):
            raise TypeError("Label values must be integers.")
        object.__setattr__(self, "value", int(self.value))
        object.__setattr__(self, "label_id", _identifier(self.label_id, "label_id"))
        object.__setattr__(self, "name", _identifier(self.name, "name"))
        if self.parent_id is not None:
            object.__setattr__(
                self, "parent_id", _identifier(self.parent_id, "parent_id")
            )
        roles = tuple(sorted(_identifier(role, "role") for role in self.roles))
        if len(set(roles)) != len(roles):
            raise ValueError("Label roles must be unique.")
        object.__setattr__(self, "roles", roles)


@dataclass(frozen=True, slots=True)
class LabelOntology:
    ontology_id: str
    source_system: str
    source_version: str
    labels: tuple[LabelDefinition, ...]
    ontology_content_id: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "ontology_id", _identifier(self.ontology_id, "ontology_id")
        )
        object.__setattr__(
            self, "source_system", _identifier(self.source_system, "source_system")
        )
        object.__setattr__(
            self, "source_version", _identifier(self.source_version, "source_version")
        )
        if not self.labels or any(
            not isinstance(label, LabelDefinition) for label in self.labels
        ):
            raise ValueError("labels must contain LabelDefinition values.")
        values = [label.value for label in self.labels]
        identifiers = [label.label_id for label in self.labels]
        if len(set(values)) != len(values) or len(set(identifiers)) != len(identifiers):
            raise ValueError("Label values and identifiers must be unique.")
        known = set(identifiers)
        if any(
            label.parent_id is not None and label.parent_id not in known
            for label in self.labels
        ):
            raise ValueError("Every parent_id must identify a label in the ontology.")
        parents = {label.label_id: label.parent_id for label in self.labels}
        for identifier in identifiers:
            visited = set()
            current = identifier
            while parents[current] is not None:
                if current in visited:
                    raise ValueError("Label ontology parent relations must be acyclic.")
                visited.add(current)
                parent = parents[current]
                if parent is None:
                    break
                current = parent
        object.__setattr__(
            self,
            "ontology_content_id",
            canonical_fingerprint(
                {
                    "kind": "label-ontology",
                    "ontology": self.ontology_id,
                    "source_system": self.source_system,
                    "source_version": self.source_version,
                    "labels": [
                        {
                            "value": label.value,
                            "id": label.label_id,
                            "name": label.name,
                            "parent": label.parent_id,
                            "roles": list(label.roles),
                        }
                        for label in self.labels
                    ],
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class LabelVolume:
    asset: MedicalImageAsset
    ontology: LabelOntology
    label_volume_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if not isinstance(self.ontology, LabelOntology):
            raise TypeError("ontology must be LabelOntology.")
        if self.asset.layout.kind is not ImageValueKind.CATEGORICAL:
            raise ValueError("Label volumes require a categorical image layout.")
        if self.asset.time_axis is not None or self.asset.values.ndim != 3:
            raise ValueError("Label volumes must be one static three-dimensional image.")
        if not np.issubdtype(self.asset.values.dtype, np.integer):
            raise TypeError("Label volume storage must have an integer dtype.")
        valid_values = np.unique(self.asset.values[np.asarray(self.asset.valid_mask)])
        known = np.asarray([label.value for label in self.ontology.labels])
        unknown = np.setdiff1d(valid_values, known)
        if unknown.size:
            raise ValueError(f"Label volume contains unknown values {unknown.tolist()}.")
        object.__setattr__(
            self,
            "label_volume_id",
            canonical_fingerprint(
                {
                    "kind": "label-volume",
                    "asset": self.asset.content_id,
                    "ontology": self.ontology.ontology_content_id,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DiffusionTensorImage:
    asset: MedicalImageAsset
    minimum_eigenvalue: float = 0.0
    tensor_image_id: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.asset, MedicalImageAsset):
            raise TypeError("asset must be MedicalImageAsset.")
        if self.asset.layout.kind is not ImageValueKind.SYMMETRIC_TENSOR:
            raise ValueError("DiffusionTensorImage requires a symmetric-tensor layout.")
        if not np.issubdtype(self.asset.values.dtype, np.floating):
            raise TypeError("Diffusion tensor images require floating-point storage.")
        floor = float(self.minimum_eigenvalue)
        if not np.isfinite(floor) or floor < 0.0:
            raise ValueError("minimum_eigenvalue must be finite and non-negative.")
        values = np.asarray(self.asset.values)
        valid = np.asarray(self.asset.valid_mask)
        selected = values[valid]
        if not selected.size:
            raise ValueError("Diffusion tensor images require at least one valid tensor.")
        tolerance = (
            128.0 * np.finfo(values.dtype).eps * max(1.0, float(np.max(np.abs(selected))))
        )
        if not np.allclose(
            selected, np.swapaxes(selected, -1, -2), atol=tolerance, rtol=0.0
        ):
            raise ValueError("Diffusion tensors must be symmetric on valid sites.")
        if (
            selected.size
            and float(np.min(np.linalg.eigvalsh(selected))) < floor - tolerance
        ):
            raise ValueError("Diffusion tensors violate the requested eigenvalue floor.")
        object.__setattr__(self, "minimum_eigenvalue", floor)
        object.__setattr__(
            self,
            "tensor_image_id",
            canonical_fingerprint(
                {
                    "kind": "diffusion-tensor-image",
                    "asset": self.asset.content_id,
                    "minimum_eigenvalue": floor,
                }
            ),
        )


__all__ = [
    "DeidentificationEvidence",
    "DiffusionTensorImage",
    "HostMetadataValue",
    "ImageAcquisitionIdentity",
    "ImageAxisConvention",
    "ImageIndexAffine",
    "ImageTimeAxis",
    "ImageValueKind",
    "ImageValueLayout",
    "LabelDefinition",
    "LabelOntology",
    "LabelVolume",
    "MedicalImageAsset",
    "VoxelReference",
]
