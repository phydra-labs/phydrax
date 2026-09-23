#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Typed immutable archives for quantum Hall result artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from ..._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from ...lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from ._effective_mixing import EffectiveLandauLevelInteractionResult
from ._form_factor import PlanarCoulombPseudopotentialResult
from ._infinite_cylinder import InfiniteHallCylinderResult
from ._localized_transport import LocalizedHallTransportResult
from ._observables import QuantumHallSphereObservables
from ._open_transport import OpenHallTransportResult
from ._sphere import HaldaneSphereSpectrumResult, QuantumHallGapResult
from ._transport import QuantumHallTransportResult
from ._vmc import QuantumHallVMCObservables


QuantumHallArtifactKind = Literal[
    "quantum-hall-sphere-spectrum",
    "quantum-hall-gap",
    "quantum-hall-sphere-observables",
    "quantum-hall-vmc-observables",
    "quantum-hall-transport",
    "quantum-hall-effective-mixing",
    "quantum-hall-planar-pseudopotentials",
    "quantum-hall-infinite-cylinder",
    "quantum-hall-localized-transport",
    "quantum-hall-open-transport",
]
QuantumHallArchiveArtifact: TypeAlias = (
    HaldaneSphereSpectrumResult
    | QuantumHallGapResult
    | QuantumHallSphereObservables
    | QuantumHallVMCObservables
    | QuantumHallTransportResult
    | EffectiveLandauLevelInteractionResult
    | PlanarCoulombPseudopotentialResult
    | InfiniteHallCylinderResult
    | LocalizedHallTransportResult
    | OpenHallTransportResult
)


def _contract(
    artifact: QuantumHallArchiveArtifact,
    /,
) -> tuple[QuantumHallArtifactKind, dict[str, str]]:
    if isinstance(artifact, HaldaneSphereSpectrumResult):
        return "quantum-hall-sphere-spectrum", {
            "result_id": artifact.result_id,
            "prepared_id": artifact.prepared.prepared_id,
        }
    if isinstance(artifact, QuantumHallGapResult):
        return "quantum-hall-gap", {
            "result_id": artifact.result_id,
            "gap_kind": artifact.kind,
        }
    if isinstance(artifact, QuantumHallSphereObservables):
        return "quantum-hall-sphere-observables", {"result_id": artifact.result_id}
    if isinstance(artifact, QuantumHallVMCObservables):
        return "quantum-hall-vmc-observables", {"result_id": artifact.result_id}
    if isinstance(artifact, QuantumHallTransportResult):
        return "quantum-hall-transport", {
            "result_id": artifact.result_id,
            "plan_id": artifact.plan_id,
        }
    if isinstance(artifact, EffectiveLandauLevelInteractionResult):
        return "quantum-hall-effective-mixing", {
            "result_id": artifact.result_id,
            "plan_id": artifact.plan_id,
        }
    if isinstance(artifact, PlanarCoulombPseudopotentialResult):
        return "quantum-hall-planar-pseudopotentials", {
            "result_id": artifact.result_id,
        }
    if isinstance(artifact, InfiniteHallCylinderResult):
        return "quantum-hall-infinite-cylinder", {
            "result_id": artifact.result_id,
            "plan_id": artifact.plan_id,
        }
    if isinstance(artifact, LocalizedHallTransportResult):
        return "quantum-hall-localized-transport", {
            "result_id": artifact.result_id,
            "plan_id": artifact.plan_id,
        }
    if isinstance(artifact, OpenHallTransportResult):
        return "quantum-hall-open-transport", {
            "result_id": artifact.result_id,
            "plan_id": artifact.plan_id,
        }
    raise TypeError("Unsupported quantum Hall archive artifact type.")


def write_quantum_hall_artifact_archive(
    path: str | Path,
    artifact: QuantumHallArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    kind, structure = _contract(artifact)
    return write_typed_array_artifact(
        path,
        artifact,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
    )


def read_quantum_hall_artifact_archive(
    path: str | Path,
    template: QuantumHallArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[QuantumHallArchiveArtifact, ArrayArtifactReceipt]:
    kind, structure = _contract(template)
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
        limits=limits,
    )
    if not isinstance(restored, type(template)):
        raise TypeError("Restored quantum Hall artifact has the wrong concrete type.")
    return restored, receipt


__all__ = [
    "QuantumHallArchiveArtifact",
    "QuantumHallArtifactKind",
    "read_quantum_hall_artifact_archive",
    "write_quantum_hall_artifact_archive",
]
