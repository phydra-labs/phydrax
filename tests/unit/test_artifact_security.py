#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from phydrax._artifact_security import (
    admit_external_artifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from phydrax.artifacts import ArtifactManifest


def _manifest(
    payload: bytes,
    *,
    sha256: str | None = None,
    license_id: str = "CC-BY-4.0",
) -> ArtifactManifest:
    return ArtifactManifest(
        artifact_id="external-fixture",
        producer="independent-test-producer",
        version="1",
        sha256=hashlib.sha256(payload).hexdigest() if sha256 is None else sha256,
        byte_size=len(payload),
        source_uri="https://example.invalid/external-fixture",
        license_id=license_id,
        model="opaque-test-bytes",
        coverage="unit-test-only",
    )


def _policy(root: Path, *, maximum_bytes: int = 32) -> ExternalArtifactPolicy:
    return ExternalArtifactPolicy(
        root,
        maximum_bytes=maximum_bytes,
        allowed_license_ids=("CC-BY-4.0",),
        allowed_suffixes=(".bin",),
    )


def test_external_artifact_is_checksum_verified_before_bytes_are_returned(tmp_path):
    payload = b"independent bytes"
    (tmp_path / "reference.bin").write_bytes(payload)
    policy = _policy(tmp_path)

    admitted = admit_external_artifact("reference.bin", _manifest(payload), policy=policy)

    assert read_admitted_artifact(admitted, policy=policy) == payload


def test_external_artifact_rejects_traversal_absolute_and_symlink_paths(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    (root / "link.bin").symlink_to(outside)
    policy = _policy(root)
    manifest = _manifest(b"outside")

    with pytest.raises(ValueError, match="confined relative path"):
        admit_external_artifact("../outside.bin", manifest, policy=policy)
    with pytest.raises(ValueError, match="confined relative path"):
        admit_external_artifact(outside, manifest, policy=policy)
    with pytest.raises(ValueError, match="symbolic links"):
        admit_external_artifact("link.bin", manifest, policy=policy)


def test_external_artifact_rejects_oversize_checksum_and_license(tmp_path):
    payload = b"eight888"
    path = tmp_path / "reference.bin"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match="manifest exceeds"):
        admit_external_artifact(
            "reference.bin", _manifest(payload), policy=_policy(tmp_path, maximum_bytes=4)
        )
    with pytest.raises(ValueError, match="checksum mismatch"):
        admit_external_artifact(
            "reference.bin",
            _manifest(payload, sha256="0" * 64),
            policy=_policy(tmp_path),
        )
    with pytest.raises(ValueError, match="checksum mismatch"):
        admit_external_artifact(
            "reference.bin",
            _manifest(payload, sha256="0" * 64, license_id="unreviewed"),
            policy=_policy(tmp_path),
        )
    with pytest.raises(PermissionError, match="license"):
        admit_external_artifact(
            "reference.bin",
            _manifest(payload, license_id="unreviewed"),
            policy=_policy(tmp_path),
        )


def test_external_artifact_read_rechecks_content_and_pickle_suffixes_are_forbidden(
    tmp_path,
):
    payload = b"trusted content"
    path = tmp_path / "reference.bin"
    path.write_bytes(payload)
    policy = _policy(tmp_path)
    admitted = admit_external_artifact("reference.bin", _manifest(payload), policy=policy)
    path.write_bytes(b"tampered bytes!")

    with pytest.raises(ValueError, match="checksum mismatch"):
        read_admitted_artifact(admitted, policy=policy)
    with pytest.raises(ValueError, match="non-pickle"):
        ExternalArtifactPolicy(
            tmp_path,
            maximum_bytes=32,
            allowed_license_ids=("CC-BY-4.0",),
            allowed_suffixes=(".pkl",),
        )
