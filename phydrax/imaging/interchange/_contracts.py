#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Fail-closed contracts shared by the supported DICOM profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

from ..._fingerprint import canonical_fingerprint
from ...interchange import AdapterReport, BoundedResource


DICOMProfile = Literal[
    "legacy-ct-image-series",
    "enhanced-ct-image",
    "nuclear-medicine-counts-image",
    "pet-activity-concentration-image-series",
    "rt-plan-metadata",
    "rt-structure-set-closed-planar",
    "rt-dose-unlinked",
    "rt-dose-linked-plan",
]


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    result = value.strip()
    if not result or result != value:
        raise ValueError(f"{name} must be non-empty and have no surrounding whitespace.")
    return result


def _uid(value: str, name: str, /) -> str:
    result = _identifier(value, name)
    if len(result) > 64 or result.startswith(".") or result.endswith("."):
        raise ValueError(f"{name} must be a canonical DICOM UID.")
    components = result.split(".")
    if any(
        not item.isdigit() or (len(item) > 1 and item.startswith("0"))
        for item in components
    ):
        raise ValueError(f"{name} must be a canonical DICOM UID.")
    return result


@dataclass(frozen=True, slots=True)
class DICOMResourcePolicy:
    """Finite aggregate limits for one profile-specific DICOM import."""

    max_instances: int = 512
    max_frames: int = 4096
    max_rows: int = 8192
    max_columns: int = 8192
    max_voxels: int = 536_870_912
    max_total_pixel_bytes: int = 2_147_483_648
    max_elements: int = 1_000_000
    max_sequence_depth: int = 32
    max_contours: int = 100_000
    max_contour_points: int = 10_000_000

    def __post_init__(self) -> None:
        values = (
            self.max_instances,
            self.max_frames,
            self.max_rows,
            self.max_columns,
            self.max_voxels,
            self.max_total_pixel_bytes,
            self.max_elements,
            self.max_sequence_depth,
            self.max_contours,
            self.max_contour_points,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, Integral) or value < 1
            for value in values
        ):
            raise ValueError("DICOM resource limits must be positive integers.")
        (
            max_instances,
            max_frames,
            max_rows,
            max_columns,
            max_voxels,
            max_total_pixel_bytes,
            max_elements,
            max_sequence_depth,
            max_contours,
            max_contour_points,
        ) = (int(value) for value in values)
        object.__setattr__(self, "max_instances", max_instances)
        object.__setattr__(self, "max_frames", max_frames)
        object.__setattr__(self, "max_rows", max_rows)
        object.__setattr__(self, "max_columns", max_columns)
        object.__setattr__(self, "max_voxels", max_voxels)
        object.__setattr__(self, "max_total_pixel_bytes", max_total_pixel_bytes)
        object.__setattr__(self, "max_elements", max_elements)
        object.__setattr__(self, "max_sequence_depth", max_sequence_depth)
        object.__setattr__(self, "max_contours", max_contours)
        object.__setattr__(self, "max_contour_points", max_contour_points)


@dataclass(frozen=True, slots=True)
class DICOMObjectIdentity:
    """DICOM object identity without patient-identifying attributes."""

    sop_class_uid: str
    sop_instance_uid: str
    study_instance_uid: str
    series_instance_uid: str
    frame_of_reference_uid: str | None = None
    identity_id: str = field(init=False)

    def __post_init__(self) -> None:
        sop_class = _uid(self.sop_class_uid, "sop_class_uid")
        sop_instance = _uid(self.sop_instance_uid, "sop_instance_uid")
        study = _uid(self.study_instance_uid, "study_instance_uid")
        series = _uid(self.series_instance_uid, "series_instance_uid")
        frame = (
            None
            if self.frame_of_reference_uid is None
            else _uid(self.frame_of_reference_uid, "frame_of_reference_uid")
        )
        object.__setattr__(self, "sop_class_uid", sop_class)
        object.__setattr__(self, "sop_instance_uid", sop_instance)
        object.__setattr__(self, "study_instance_uid", study)
        object.__setattr__(self, "series_instance_uid", series)
        object.__setattr__(self, "frame_of_reference_uid", frame)
        object.__setattr__(
            self,
            "identity_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-object-identity",
                    "sop_class_uid": sop_class,
                    "sop_instance_uid": sop_instance,
                    "study_instance_uid": study,
                    "series_instance_uid": series,
                    "frame_of_reference_uid": frame,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DICOMReference:
    """One typed DICOM SOP-instance reference edge."""

    source_sop_instance_uid: str
    relationship: str
    target_sop_instance_uid: str
    target_sop_class_uid: str | None = None
    reference_id: str = field(init=False)

    def __post_init__(self) -> None:
        source = _uid(self.source_sop_instance_uid, "source_sop_instance_uid")
        relationship = _identifier(self.relationship, "relationship")
        target = _uid(self.target_sop_instance_uid, "target_sop_instance_uid")
        target_class = (
            None
            if self.target_sop_class_uid is None
            else _uid(self.target_sop_class_uid, "target_sop_class_uid")
        )
        object.__setattr__(self, "source_sop_instance_uid", source)
        object.__setattr__(self, "relationship", relationship)
        object.__setattr__(self, "target_sop_instance_uid", target)
        object.__setattr__(self, "target_sop_class_uid", target_class)
        object.__setattr__(
            self,
            "reference_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-reference",
                    "source": source,
                    "relationship": relationship,
                    "target": target,
                    "target_class": target_class,
                }
            ),
        )


@dataclass(frozen=True, slots=True)
class DICOMReferenceGraph:
    """Finite DICOM object graph whose linked profiles require reference closure."""

    objects: tuple[DICOMObjectIdentity, ...]
    references: tuple[DICOMReference, ...] = ()
    graph_id: str = field(init=False)

    def __post_init__(self) -> None:
        objects = tuple(self.objects)
        references = tuple(self.references)
        if not objects or any(
            not isinstance(item, DICOMObjectIdentity) for item in objects
        ):
            raise ValueError("A DICOM reference graph requires object identities.")
        if any(not isinstance(item, DICOMReference) for item in references):
            raise TypeError("references must contain DICOMReference values.")
        object_uids = [item.sop_instance_uid for item in objects]
        if len(object_uids) != len(set(object_uids)):
            raise ValueError(
                "DICOM object identities must have unique SOP Instance UIDs."
            )
        if len({item.reference_id for item in references}) != len(references):
            raise ValueError("DICOM reference edges must be unique.")
        known = set(object_uids)
        if any(item.source_sop_instance_uid not in known for item in references):
            raise ValueError("Every DICOM reference source must be present in the graph.")
        object.__setattr__(self, "objects", objects)
        object.__setattr__(self, "references", references)
        object.__setattr__(
            self,
            "graph_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-reference-graph",
                    "objects": sorted(item.identity_id for item in objects),
                    "references": sorted(item.reference_id for item in references),
                }
            ),
        )

    @property
    def unresolved_references(self) -> tuple[DICOMReference, ...]:
        known = {item.sop_instance_uid for item in self.objects}
        return tuple(
            item for item in self.references if item.target_sop_instance_uid not in known
        )

    def require_closed(self) -> None:
        unresolved = self.unresolved_references
        if unresolved:
            targets = ", ".join(
                sorted(item.target_sop_instance_uid for item in unresolved)
            )
            raise ValueError(f"Unresolved linked DICOM references: {targets}.")
        classes = {item.sop_instance_uid: item.sop_class_uid for item in self.objects}
        for reference in self.references:
            expected = reference.target_sop_class_uid
            if (
                expected is not None
                and classes[reference.target_sop_instance_uid] != expected
            ):
                raise ValueError(
                    "A linked DICOM reference has a mismatched SOP Class UID."
                )


@dataclass(frozen=True, slots=True)
class DICOMImportReport:
    """Profile, byte identity, object graph, and generic adapter accounting."""

    profile: DICOMProfile
    resources: tuple[BoundedResource, ...]
    graph: DICOMReferenceGraph
    adapter_report: AdapterReport
    report_id: str = field(init=False)

    def __post_init__(self) -> None:
        profile = _identifier(self.profile, "profile")
        resources = tuple(self.resources)
        if not resources or any(
            not isinstance(item, BoundedResource) for item in resources
        ):
            raise ValueError("A DICOM import report requires bounded resources.")
        if not isinstance(self.graph, DICOMReferenceGraph):
            raise TypeError("graph must be a DICOMReferenceGraph.")
        if not isinstance(self.adapter_report, AdapterReport):
            raise TypeError("adapter_report must be an AdapterReport.")
        if not self.adapter_report.valid:
            raise ValueError("A governed DICOM result requires a valid adapter report.")
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "resources", resources)
        object.__setattr__(
            self,
            "report_id",
            canonical_fingerprint(
                {
                    "kind": "dicom-import-report",
                    "profile": profile,
                    "resources": [item.manifest.manifest_id for item in resources],
                    "graph": self.graph.graph_id,
                    "adapter_report": self.adapter_report.report_id,
                }
            ),
        )


class DICOMDependencyError(ImportError):
    """The optional DICOM parser is unavailable."""


class DICOMProfileError(ValueError):
    """The source is not representable by the named fail-closed profile."""


__all__ = [
    "DICOMDependencyError",
    "DICOMImportReport",
    "DICOMObjectIdentity",
    "DICOMProfile",
    "DICOMProfileError",
    "DICOMReference",
    "DICOMReferenceGraph",
    "DICOMResourcePolicy",
]
