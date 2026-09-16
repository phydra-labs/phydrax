#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable archives for direct fixed-sector artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from ...._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from ....lifecycle._array_artifact import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)
from ._operator import QuantumSectorOperator
from ._orbit_operator import QuantumOrbitSectorOperator
from ._orbit_sector import PreparedFiniteGroupAction, PreparedOrbitSectorBasis
from ._sector import (
    FixedBosonNumberBasis,
    FixedCardinalityFermionBasis,
    FixedSpinProjectionBasis,
)


QuantumLatticeArtifactKind = Literal[
    "quantum-lattice-fixed-cardinality-basis",
    "quantum-lattice-fixed-spin-basis",
    "quantum-lattice-fixed-boson-basis",
    "quantum-lattice-sector-operator",
    "quantum-lattice-finite-group-action",
    "quantum-lattice-orbit-basis",
    "quantum-lattice-orbit-operator",
]
QuantumLatticeArchiveArtifact: TypeAlias = (
    FixedCardinalityFermionBasis
    | FixedSpinProjectionBasis
    | FixedBosonNumberBasis
    | QuantumSectorOperator
    | PreparedFiniteGroupAction
    | PreparedOrbitSectorBasis
    | QuantumOrbitSectorOperator
)


def _quantum_artifact_contract(
    artifact: QuantumLatticeArchiveArtifact, /
) -> tuple[QuantumLatticeArtifactKind, dict[str, str]]:
    if isinstance(artifact, FixedCardinalityFermionBasis):
        return "quantum-lattice-fixed-cardinality-basis", {
            "basis_id": artifact.basis_id,
            "mode_order_id": artifact.mode_order.order_id,
            "resource_policy_id": artifact.resources.policy_id,
        }
    if isinstance(artifact, FixedSpinProjectionBasis):
        return "quantum-lattice-fixed-spin-basis", {
            "basis_id": artifact.basis_id,
            "resource_policy_id": artifact.resources.policy_id,
        }
    if isinstance(artifact, FixedBosonNumberBasis):
        return "quantum-lattice-fixed-boson-basis", {
            "basis_id": artifact.basis_id,
            "resource_policy_id": artifact.resources.policy_id,
        }
    if isinstance(artifact, QuantumSectorOperator):
        return "quantum-lattice-sector-operator", {
            "operator_id": artifact.operator_id,
            "prepared_id": artifact.prepared.prepared_id,
            "charge_map_id": artifact.charge_map.map_id,
            "certification_id": artifact.certification.certification_id,
        }
    if isinstance(artifact, PreparedFiniteGroupAction):
        return "quantum-lattice-finite-group-action", {
            "prepared_id": artifact.prepared_id,
            "plan_id": artifact.plan.plan_id,
            "character_id": artifact.character.character_id,
        }
    if isinstance(artifact, PreparedOrbitSectorBasis):
        return "quantum-lattice-orbit-basis", {
            "basis_id": artifact.basis_id,
            "base_basis_id": artifact.base.basis_id,
            "action_id": artifact.action.prepared_id,
        }
    if isinstance(artifact, QuantumOrbitSectorOperator):
        return "quantum-lattice-orbit-operator", {
            "operator_id": artifact.operator_id,
            "prepared_id": artifact.prepared.prepared_id,
            "basis_id": artifact.basis.basis_id,
            "evidence_id": artifact.evidence.evidence_id,
        }
    raise TypeError("Unsupported quantum-lattice archive artifact type.")


def write_quantum_lattice_artifact_archive(
    path: str | Path,
    artifact: QuantumLatticeArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
) -> ArrayArtifactReceipt:
    """Write a direct-sector artifact without an ambient-basis materialization."""

    kind, structure = _quantum_artifact_contract(artifact)
    return write_typed_array_artifact(
        path,
        artifact,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
    )


def read_quantum_lattice_artifact_archive(
    path: str | Path,
    template: QuantumLatticeArchiveArtifact,
    provenance: ArrayArtifactProvenance,
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[QuantumLatticeArchiveArtifact, ArrayArtifactReceipt]:
    """Restore only into an exactly matching caller-prepared sector/result tree."""

    kind, structure = _quantum_artifact_contract(template)
    restored, receipt = read_typed_array_artifact(
        path,
        template,
        artifact_kind=kind,
        provenance=provenance,
        structure_ids=structure,
        limits=limits,
    )
    if not isinstance(restored, type(template)):
        raise TypeError("Restored quantum-lattice artifact has the wrong concrete type.")
    return restored, receipt


__all__ = [
    "QuantumLatticeArchiveArtifact",
    "QuantumLatticeArtifactKind",
    "read_quantum_lattice_artifact_archive",
    "write_quantum_lattice_artifact_archive",
]
