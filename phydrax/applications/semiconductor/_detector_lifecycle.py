#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable detector-result archives with exact plan and route identity."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from ..._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from ..._fingerprint import canonical_fingerprint
from ...lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from ._detector import DetectorBiasElectrostaticResult, DetectorWeightingFieldResult
from ._detector_response import ShockleyRamoResponseResult


DetectorArtifactKind = Literal[
    "semiconductor-detector-bias-result",
    "semiconductor-detector-weighting-result",
    "semiconductor-detector-shockley-ramo-result",
]
DetectorArchiveArtifact: TypeAlias = (
    DetectorBiasElectrostaticResult
    | DetectorWeightingFieldResult
    | ShockleyRamoResponseResult
)


def _detector_artifact_contract(
    artifact: DetectorArchiveArtifact, /
) -> tuple[DetectorArtifactKind, dict[str, str]]:
    if isinstance(artifact, DetectorBiasElectrostaticResult):
        return "semiconductor-detector-bias-result", {"plan_id": artifact.plan_id}
    if isinstance(artifact, DetectorWeightingFieldResult):
        return "semiconductor-detector-weighting-result", {"plan_id": artifact.plan_id}
    if isinstance(artifact, ShockleyRamoResponseResult):
        return "semiconductor-detector-shockley-ramo-result", {
            "trajectory_dataset_id": artifact.trajectory_dataset_id,
            "weighting_plan_id": artifact.weighting_plan_id,
            "route_id": artifact.route_id,
            "sign_convention": artifact.sign_convention,
            "resource_id": canonical_fingerprint(
                {
                    "kind": "shockley-ramo-resource-evidence",
                    "case_count": artifact.resources.case_count,
                    "sample_count": artifact.resources.sample_count,
                    "interpolation_route_count": artifact.resources.interpolation_route_count,
                    "electrode_count": artifact.resources.electrode_count,
                }
            ),
        }
    raise TypeError("Unsupported semiconductor detector archive artifact type.")


def write_detector_artifact_archive(
    path: str | Path,
    artifact: DetectorArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    """Write detector fields, masks, residuals, and response arrays unchanged."""

    kind, structure = _detector_artifact_contract(artifact)
    return write_typed_array_artifact(
        path,
        artifact,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
    )


def read_detector_artifact_archive(
    path: str | Path,
    template: DetectorArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[DetectorArchiveArtifact, ArrayArtifactReceipt]:
    """Restore only against matching caller-prepared detector/result structure."""

    kind, structure = _detector_artifact_contract(template)
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
        limits=limits,
    )
    if not isinstance(restored, type(template)):
        raise TypeError("Restored detector artifact has the wrong concrete type.")
    return restored, receipt


__all__ = [
    "DetectorArchiveArtifact",
    "DetectorArtifactKind",
    "read_detector_artifact_archive",
    "write_detector_artifact_archive",
]
