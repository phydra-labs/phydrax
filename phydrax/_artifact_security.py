#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed admission for local, externally supplied artifact bytes."""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx

from ._fingerprint import canonical_fingerprint
from ._strict import StrictModule
from ._trainable import NonTrainableState


if TYPE_CHECKING:
    from .artifacts import ArtifactManifest


_FORBIDDEN_PICKLE_SUFFIXES = frozenset({".dill", ".joblib", ".pickle", ".pkl"})


def _identifier(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty.")
    return normalized


def _identifiers(values: Sequence[str], name: str, /) -> tuple[str, ...]:
    if isinstance(values, str):
        raise TypeError(f"{name} must be a sequence of strings.")
    normalized = tuple(sorted(_identifier(value, name) for value in values))
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must be non-empty and unique.")
    return normalized


class ExternalArtifactPolicy(StrictModule, NonTrainableState):
    """Local-root, byte, format, and license allowlist for external artifacts."""

    root: str = eqx.field(static=True)
    maximum_bytes: int = eqx.field(static=True)
    allowed_license_ids: tuple[str, ...] = eqx.field(static=True)
    allowed_suffixes: tuple[str, ...] = eqx.field(static=True)
    read_chunk_bytes: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        root: str | os.PathLike[str],
        /,
        *,
        maximum_bytes: int,
        allowed_license_ids: Sequence[str],
        allowed_suffixes: Sequence[str],
        read_chunk_bytes: int = 1_048_576,
    ):
        root_path = Path(root)
        if root_path.is_symlink():
            raise ValueError("External artifact root cannot be a symbolic link.")
        resolved_root = root_path.resolve(strict=True)
        if not resolved_root.is_dir():
            raise ValueError("External artifact root must be an existing directory.")
        if (
            isinstance(maximum_bytes, bool)
            or not isinstance(maximum_bytes, int)
            or maximum_bytes <= 0
        ):
            raise ValueError("maximum_bytes must be a positive integer.")
        if (
            isinstance(read_chunk_bytes, bool)
            or not isinstance(read_chunk_bytes, int)
            or read_chunk_bytes <= 0
        ):
            raise ValueError("read_chunk_bytes must be a positive integer.")
        licenses = _identifiers(allowed_license_ids, "allowed license ID")
        suffixes = tuple(
            sorted(
                _identifier(value, "allowed suffix").lower() for value in allowed_suffixes
            )
        )
        if (
            not suffixes
            or len(set(suffixes)) != len(suffixes)
            or any(
                not suffix.startswith(".")
                or suffix in _FORBIDDEN_PICKLE_SUFFIXES
                or "/" in suffix
                or "\\" in suffix
                for suffix in suffixes
            )
        ):
            raise ValueError("allowed_suffixes must be unique non-pickle file suffixes.")
        self.root = str(resolved_root)
        self.maximum_bytes = maximum_bytes
        self.allowed_license_ids = licenses
        self.allowed_suffixes = suffixes
        self.read_chunk_bytes = min(read_chunk_bytes, maximum_bytes)
        self.policy_id = canonical_fingerprint(
            {
                "kind": "external-artifact-policy",
                "root": self.root,
                "maximum_bytes": maximum_bytes,
                "allowed_license_ids": list(licenses),
                "allowed_suffixes": list(suffixes),
                "read_chunk_bytes": self.read_chunk_bytes,
            }
        )


class AdmittedExternalArtifact(StrictModule, NonTrainableState):
    """Immutable identity of bytes admitted under one exact local policy."""

    relative_path: str = eqx.field(static=True)
    resolved_path: str = eqx.field(static=True)
    sha256: str = eqx.field(static=True)
    byte_size: int = eqx.field(static=True)
    license_id: str = eqx.field(static=True)
    manifest_id: str = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)
    admission_id: str = eqx.field(static=True)

    def __init__(
        self,
        relative_path: str,
        resolved_path: str,
        sha256: str,
        byte_size: int,
        license_id: str,
        manifest_id: str,
        policy_id: str,
        /,
    ):
        relative = _identifier(relative_path, "relative artifact path")
        resolved = _identifier(resolved_path, "resolved artifact path")
        digest = _identifier(sha256, "artifact SHA-256")
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("Artifact SHA-256 must be lowercase hexadecimal.")
        if isinstance(byte_size, bool) or not isinstance(byte_size, int) or byte_size < 0:
            raise ValueError("Artifact byte size must be a non-negative integer.")
        license_ = _identifier(license_id, "artifact license ID")
        manifest = _identifier(manifest_id, "artifact manifest ID")
        policy = _identifier(policy_id, "artifact policy ID")
        self.relative_path = relative
        self.resolved_path = resolved
        self.sha256 = digest
        self.byte_size = byte_size
        self.license_id = license_
        self.manifest_id = manifest
        self.policy_id = policy
        self.admission_id = canonical_fingerprint(
            {
                "kind": "admitted-external-artifact",
                "relative_path": relative,
                "resolved_path": resolved,
                "sha256": digest,
                "byte_size": byte_size,
                "license_id": license_,
                "manifest_id": manifest,
                "policy_id": policy,
            }
        )

    def to_record(self) -> dict[str, object]:
        return {
            "kind": "admitted-external-artifact",
            "relative_path": self.relative_path,
            "resolved_path": self.resolved_path,
            "sha256": self.sha256,
            "byte_size": self.byte_size,
            "license_id": self.license_id,
            "manifest_id": self.manifest_id,
            "policy_id": self.policy_id,
            "admission_id": self.admission_id,
        }


def _resolve_artifact_path(
    relative_path: str | os.PathLike[str],
    policy: ExternalArtifactPolicy,
    /,
) -> tuple[str, Path]:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("External artifact path must be a confined relative path.")
    normalized = relative.as_posix()
    if normalized in ("", "."):
        raise ValueError("External artifact path must identify one file.")
    root = Path(policy.root)
    candidate = root
    for component in relative.parts:
        candidate = candidate / component
        if candidate.is_symlink():
            raise ValueError("External artifact paths cannot contain symbolic links.")
    resolved = candidate.resolve(strict=True)
    if root != resolved and root not in resolved.parents:
        raise ValueError("External artifact path escapes the configured root.")
    if not resolved.is_file():
        raise ValueError("External artifact path must identify a regular file.")
    if resolved.suffix.lower() not in policy.allowed_suffixes:
        raise ValueError("External artifact suffix is not admitted by policy.")
    return normalized, resolved


def _open_artifact_descriptor(
    relative_path: str,
    policy: ExternalArtifactPolicy,
    /,
) -> int:
    parts = Path(relative_path).parts
    read_flags = os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_flags = read_flags | os.O_DIRECTORY
    directory = os.open(policy.root, directory_flags)
    try:
        for component in parts[:-1]:
            child = os.open(
                component,
                directory_flags,
                dir_fd=directory,
            )
            os.close(directory)
            directory = child
        return os.open(
            parts[-1],
            read_flags,
            dir_fd=directory,
        )
    finally:
        os.close(directory)


def _read_verified_bytes(
    relative_path: str,
    expected_size: int,
    expected_sha256: str,
    policy: ExternalArtifactPolicy,
    /,
    *,
    retain: bool,
) -> bytes | None:
    digest = hashlib.sha256()
    payload = bytearray() if retain else None
    total = 0
    with os.fdopen(
        _open_artifact_descriptor(relative_path, policy),
        "rb",
    ) as stream:
        opened = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(opened.st_mode)
            or opened.st_size != expected_size
            or opened.st_size > policy.maximum_bytes
        ):
            raise ValueError("External artifact size or file type is not admitted.")
        while chunk := stream.read(policy.read_chunk_bytes):
            total += len(chunk)
            if total > expected_size or total > policy.maximum_bytes:
                raise ValueError("External artifact exceeded its admitted byte bound.")
            digest.update(chunk)
            if payload is not None:
                payload.extend(chunk)
    if total != expected_size:
        raise ValueError("External artifact size does not match its manifest.")
    if digest.hexdigest() != expected_sha256:
        raise ValueError("External artifact SHA-256 checksum mismatch.")
    return None if payload is None else bytes(payload)


def admit_external_artifact(
    relative_path: str | os.PathLike[str],
    manifest: ArtifactManifest,
    /,
    *,
    policy: ExternalArtifactPolicy,
) -> AdmittedExternalArtifact:
    """Verify size and SHA-256 before admitting any external artifact path."""
    from .artifacts import ArtifactManifest

    if not isinstance(policy, ExternalArtifactPolicy):
        raise TypeError("policy must be an ExternalArtifactPolicy.")
    if not isinstance(manifest, ArtifactManifest):
        raise TypeError("manifest must be an ArtifactManifest.")
    if manifest.byte_size > policy.maximum_bytes:
        raise ValueError("External artifact manifest exceeds the policy byte bound.")
    normalized, resolved = _resolve_artifact_path(relative_path, policy)
    _read_verified_bytes(
        normalized,
        manifest.byte_size,
        manifest.sha256,
        policy,
        retain=False,
    )
    if manifest.license_id not in policy.allowed_license_ids:
        raise PermissionError("External artifact license is not admitted by policy.")
    return AdmittedExternalArtifact(
        normalized,
        str(resolved),
        manifest.sha256,
        manifest.byte_size,
        manifest.license_id,
        manifest.manifest_id,
        policy.policy_id,
    )


def read_admitted_artifact(
    artifact: AdmittedExternalArtifact,
    /,
    *,
    policy: ExternalArtifactPolicy,
) -> bytes:
    """Return raw bytes only after repeating bounded size and checksum admission."""
    if not isinstance(artifact, AdmittedExternalArtifact):
        raise TypeError("artifact must be an AdmittedExternalArtifact.")
    if not isinstance(policy, ExternalArtifactPolicy):
        raise TypeError("policy must be an ExternalArtifactPolicy.")
    if artifact.policy_id != policy.policy_id:
        raise PermissionError("External artifact was admitted under a different policy.")
    if artifact.license_id not in policy.allowed_license_ids:
        raise PermissionError("External artifact license is not admitted by policy.")
    normalized, resolved = _resolve_artifact_path(artifact.relative_path, policy)
    if normalized != artifact.relative_path or str(resolved) != artifact.resolved_path:
        raise ValueError("External artifact path changed after admission.")
    payload = _read_verified_bytes(
        normalized,
        artifact.byte_size,
        artifact.sha256,
        policy,
        retain=True,
    )
    if payload is None:
        raise RuntimeError("External artifact byte retention failed.")
    return payload


__all__ = (
    "AdmittedExternalArtifact",
    "ExternalArtifactPolicy",
    "admit_external_artifact",
    "read_admitted_artifact",
)
