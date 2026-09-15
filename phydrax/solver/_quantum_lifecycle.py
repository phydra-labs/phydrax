#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable archives for TPQ and fixed-sector response results."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from ..lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from ._quantum_response import (
    FiniteTemperatureResponseResult,
    ZeroTemperatureResponseResult,
)
from ._thermal_pure_quantum import ThermalPureQuantumResult


QuantumResultArtifactKind = Literal[
    "quantum-lattice-tpq-result",
    "quantum-lattice-zero-temperature-response",
    "quantum-lattice-finite-temperature-response",
]
QuantumResultArchiveArtifact: TypeAlias = (
    ThermalPureQuantumResult
    | ZeroTemperatureResponseResult
    | FiniteTemperatureResponseResult
)


def _quantum_result_contract(
    artifact: QuantumResultArchiveArtifact, /
) -> tuple[QuantumResultArtifactKind, dict[str, str]]:
    if isinstance(artifact, ThermalPureQuantumResult):
        return "quantum-lattice-tpq-result", {
            "prepared_id": artifact.prepared_id,
            "hamiltonian_id": artifact.hamiltonian_id,
            "sector_basis_id": artifact.sector_basis_id,
        }
    if isinstance(artifact, ZeroTemperatureResponseResult):
        return "quantum-lattice-zero-temperature-response", {
            "source_basis_id": artifact.evidence.source_basis_id,
            "target_basis_id": artifact.evidence.target_basis_id,
            "probe_id": artifact.evidence.probe_id,
        }
    if isinstance(artifact, FiniteTemperatureResponseResult):
        return "quantum-lattice-finite-temperature-response", {
            "source_basis_id": artifact.evidence.source_basis_id,
            "target_basis_id": artifact.evidence.target_basis_id,
            "probe_id": artifact.evidence.probe_id,
        }
    raise TypeError("Unsupported quantum-result archive artifact type.")


def write_quantum_result_archive(
    path: str | Path,
    artifact: QuantumResultArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    """Write raw probes, correlations, spectra, errors, and validity evidence."""

    kind, structure = _quantum_result_contract(artifact)
    return write_typed_array_artifact(
        path,
        artifact,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
    )


def read_quantum_result_archive(
    path: str | Path,
    template: QuantumResultArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[QuantumResultArchiveArtifact, ArrayArtifactReceipt]:
    """Restore only into the matching caller-prepared result structure."""

    kind, structure = _quantum_result_contract(template)
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
        limits=limits,
    )
    if not isinstance(restored, type(template)):
        raise TypeError("Restored quantum result has the wrong concrete type.")
    return restored, receipt


__all__ = [
    "QuantumResultArchiveArtifact",
    "QuantumResultArtifactKind",
    "read_quantum_result_archive",
    "write_quantum_result_archive",
]
