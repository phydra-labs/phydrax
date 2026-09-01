#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import equinox as eqx

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState


class ArtifactManifest(StrictModule, NonTrainableState):
    artifact_id: str = eqx.field(static=True)
    producer: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    source_uri: str = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    model: str = eqx.field(static=True)
    coverage: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        artifact_id: str,
        producer: str,
        version: str,
        sha256: str,
        byte_size: int,
        source_uri: str,
        license_id: str,
        model: str,
        coverage: str,
    ):
        values = tuple(
            str(value).strip()
            for value in (
                artifact_id,
                producer,
                version,
                sha256,
                source_uri,
                license_id,
                model,
                coverage,
            )
        )
        if any(not value for value in values):
            raise ValueError("Artifact manifest fields must be non-empty.")
        if len(values[3]) != 64 or any(
            character not in "0123456789abcdef" for character in values[3].lower()
        ):
            raise ValueError("Artifact checksum must be lowercase SHA-256 hex.")
        if int(byte_size) < 0:
            raise ValueError("Artifact byte size must be non-negative.")
        (
            self.artifact_id,
            self.producer,
            self.version,
            self.sha256,
            self.source_uri,
            self.license_id,
            self.model,
            self.coverage,
        ) = values
        self.byte_size = int(byte_size)
        self.manifest_id = canonical_fingerprint(
            {
                "kind": "artifact-manifest",
                "values": list(values),
                "byte_size": int(byte_size),
            }
        )

    def as_json(self) -> str:
        return json.dumps(
            {
                "artifact_id": self.artifact_id,
                "producer": self.producer,
                "version": self.version,
                "sha256": self.sha256,
                "byte_size": self.byte_size,
                "source_uri": self.source_uri,
                "license_id": self.license_id,
                "model": self.model,
                "coverage": self.coverage,
            },
            sort_keys=True,
        )


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
