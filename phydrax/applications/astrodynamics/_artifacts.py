#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import ArtifactManifest


class PinnedArtifact(StrictModule, NonTrainableState):
    path: str = eqx.field(static=True)
    manifest: ArtifactManifest


class AstrodynamicsDataStore(StrictModule, NonTrainableState):
    root: str = eqx.field(static=True)
    store_id: str = eqx.field(static=True)

    def __init__(self, root: str | Path, /):
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(
                "Astrodynamics data-store root must exist and be a directory."
            )
        self.root = str(path)
        self.store_id = canonical_fingerprint(
            {"kind": "astrodynamics-data-store", "root": str(path)}
        )

    def resolve(
        self, relative_path: str, manifest: ArtifactManifest, /
    ) -> PinnedArtifact:
        if not isinstance(manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        path = (Path(self.root) / relative_path).resolve()
        if Path(self.root) not in path.parents:
            raise ValueError("Artifact path escapes the configured store.")
        if not path.is_file():
            raise ValueError("Pinned artifact is absent from the configured store.")
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.sha256 or len(payload) != manifest.byte_size:
            raise ValueError(
                "Pinned artifact checksum or byte size does not match manifest."
            )
        return PinnedArtifact(str(path), manifest)


__all__ = ["ArtifactManifest", "AstrodynamicsDataStore", "PinnedArtifact"]
