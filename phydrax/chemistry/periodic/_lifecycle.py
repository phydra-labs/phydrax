#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Concrete immutable archives for canonical periodic result artifacts."""

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
from ...operators.periodic import (
    PeriodicPencilEvaluation,
    PeriodicSpectrumResult,
    PeriodicTranslationFamilyState,
)
from ._embedding import SingleSiteDMFTResult
from ._lattice_force_constants import (
    FiniteDisplacementIFC2Result,
    SecondOrderForceConstants,
    ThirdOrderForceConstants,
)


PeriodicArtifactKind = Literal[
    "periodic-translation-family-state",
    "periodic-pencil-evaluation",
    "periodic-spectrum-result",
    "periodic-ifc2",
    "periodic-ifc3",
    "periodic-finite-displacement-ifc2-result",
    "periodic-single-site-dmft-result",
]
PeriodicArchiveArtifact: TypeAlias = (
    PeriodicTranslationFamilyState
    | PeriodicPencilEvaluation
    | PeriodicSpectrumResult
    | SecondOrderForceConstants
    | ThirdOrderForceConstants
    | FiniteDisplacementIFC2Result
    | SingleSiteDMFTResult
)


def _periodic_artifact_contract(
    artifact: PeriodicArchiveArtifact, /
) -> tuple[PeriodicArtifactKind, dict[str, str]]:
    if isinstance(artifact, PeriodicTranslationFamilyState):
        return "periodic-translation-family-state", {
            "plan_id": artifact.plan_id,
            "numeric_id": artifact.numeric_id,
        }
    if isinstance(artifact, PeriodicPencilEvaluation):
        return "periodic-pencil-evaluation", {"pencil_id": artifact.pencil_id}
    if isinstance(artifact, PeriodicSpectrumResult):
        return "periodic-spectrum-result", {
            "cell_id": artifact.cell_id,
            "basis_id": artifact.basis_id,
            "support_id": artifact.support_id,
            "pencil_id": artifact.pencil_id,
            "result_id": artifact.result_id,
            "energy_unit_id": artifact.energy_unit.unit_id,
        }
    if isinstance(artifact, SecondOrderForceConstants):
        return "periodic-ifc2", {
            "cell_id": artifact.cell.cell_id,
            "system_id": artifact.system_id,
            "source_kind": artifact.source_kind,
            "source_id": artifact.source_id,
            "convention_id": artifact.convention_id,
            "unit_id": artifact.unit.unit_id,
            "ifc_id": artifact.ifc_id,
        }
    if isinstance(artifact, ThirdOrderForceConstants):
        return "periodic-ifc3", {
            "system_id": artifact.system_id,
            "source_kind": artifact.source_kind,
            "source_id": artifact.source_id,
            "unit_id": artifact.unit.unit_id,
            "ifc_id": artifact.ifc_id,
        }
    if isinstance(artifact, FiniteDisplacementIFC2Result):
        return "periodic-finite-displacement-ifc2-result", {
            "ifc_id": artifact.force_constants.ifc_id,
            "plan_result_id": artifact.result_id,
        }
    if isinstance(artifact, SingleSiteDMFTResult):
        return "periodic-single-site-dmft-result", {
            "plan_id": artifact.plan_id,
            "result_id": artifact.result_id,
            "impurity_result_id": artifact.impurity.result_id,
            "bath_fit_id": artifact.bath_fit.fit_id,
        }
    raise TypeError("Unsupported periodic archive artifact type.")


def write_periodic_artifact_archive(
    path: str | Path,
    artifact: PeriodicArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    """Write one canonical periodic artifact without executable providers."""

    kind, structure = _periodic_artifact_contract(artifact)
    return write_typed_array_artifact(
        path,
        artifact,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
    )


def read_periodic_artifact_archive(
    path: str | Path,
    template: PeriodicArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[PeriodicArchiveArtifact, ArrayArtifactReceipt]:
    """Restore only against the matching caller-prepared periodic structure."""

    kind, structure = _periodic_artifact_contract(template)
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
        limits=limits,
    )
    if not isinstance(restored, type(template)):
        raise TypeError("Restored periodic artifact has the wrong concrete type.")
    return restored, receipt


__all__ = [
    "PeriodicArchiveArtifact",
    "PeriodicArtifactKind",
    "read_periodic_artifact_archive",
    "write_periodic_artifact_archive",
]
