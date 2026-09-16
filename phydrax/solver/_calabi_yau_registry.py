#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Local content-addressed registry for frozen Calabi–Yau checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from .._fingerprint import canonical_fingerprint
from ..lifecycle import (
    ArrayArtifactProvenance,
    ArrayArtifactReceipt,
    read_typed_array_artifact,
    write_typed_array_artifact,
)


@dataclass(frozen=True, slots=True)
class CalabiYauCheckpointRegistry:
    """Bounded local registry; no remote fetching or bundled checkpoint claim."""

    root: Path
    maximum_artifacts: int
    registry_id: str

    def __init__(self, root: str | Path, /, *, maximum_artifacts: int = 1024):
        root_ = Path(root).expanduser().resolve()
        maximum = int(maximum_artifacts)
        if maximum < 1:
            raise ValueError("maximum_artifacts must be positive.")
        root_.mkdir(parents=True, exist_ok=True)
        content = {
            "kind": "calabi-yau-checkpoint-registry",
            "root": str(root_),
            "maximum_artifacts": maximum,
        }
        object.__setattr__(self, "root", root_)
        object.__setattr__(self, "maximum_artifacts", maximum)
        object.__setattr__(self, "registry_id", canonical_fingerprint(content))

    def path(self, artifact_id: str, /) -> Path:
        identifier = str(artifact_id)
        if len(identifier) != 64 or any(
            value not in "0123456789abcdef" for value in identifier
        ):
            raise ValueError("artifact_id must be a lowercase SHA-256 content address.")
        return self.root / f"{identifier}.npz"


def register_calabi_yau_checkpoint(
    registry: CalabiYauCheckpointRegistry,
    checkpoint: Any,
    provenance: ArrayArtifactProvenance,
    structure_ids: Mapping[str, str],
    /,
) -> ArrayArtifactReceipt:
    """Write one typed checkpoint and move it to its verified content address."""

    if not isinstance(registry, CalabiYauCheckpointRegistry):
        raise TypeError("registry must be CalabiYauCheckpointRegistry.")
    if len(tuple(registry.root.glob("*.npz"))) >= registry.maximum_artifacts:
        raise ValueError("Calabi–Yau checkpoint registry capacity is exhausted.")
    pending_id = canonical_fingerprint(
        {
            "kind": "calabi-yau-pending-checkpoint",
            "registry": registry.registry_id,
            "structure_ids": dict(sorted(structure_ids.items())),
        }
    )
    pending = registry.root / f"pending-{pending_id}.npz"
    receipt = write_typed_array_artifact(
        pending,
        checkpoint,
        artifact_kind="calabi-yau-checkpoint",
        provenance=provenance,
        structure_ids=structure_ids,
    )
    destination = registry.path(receipt.artifact_id)
    if destination.exists():
        pending.unlink()
    else:
        pending.rename(destination)
    return ArrayArtifactReceipt(
        destination,
        receipt.artifact_kind,
        receipt.artifact_type,
        receipt.artifact_id,
        receipt.provenance_id,
        receipt.array_digest,
    )


def load_calabi_yau_checkpoint(
    registry: CalabiYauCheckpointRegistry,
    artifact_id: str,
    template: Any,
    provenance: ArrayArtifactProvenance,
    structure_ids: Mapping[str, str],
    /,
    *,
    limits: ArrayArchiveLimits = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> tuple[Any, ArrayArtifactReceipt]:
    """Restore numeric leaves only against caller-owned executable structure."""

    if not isinstance(registry, CalabiYauCheckpointRegistry):
        raise TypeError("registry must be CalabiYauCheckpointRegistry.")
    return read_typed_array_artifact(
        registry.path(artifact_id),
        template,
        artifact_kind="calabi-yau-checkpoint",
        provenance=provenance,
        structure_ids=structure_ids,
        limits=limits,
    )


__all__ = [
    "CalabiYauCheckpointRegistry",
    "load_calabi_yau_checkpoint",
    "register_calabi_yau_checkpoint",
]
