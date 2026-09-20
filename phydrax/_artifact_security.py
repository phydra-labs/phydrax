#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed admission for local, externally supplied artifact bytes."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import equinox as eqx

from ._external_resource import (
    read_bounded_resource,
    ResourceLimits,
    ResourceReadError,
)
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
    maximum_depth: int = eqx.field(static=True)
    policy_id: str = eqx.field(static=True)

    def __init__(
        self,
        root: str | os.PathLike[str],
        /,
        *,
        maximum_bytes: int,
        allowed_license_ids: Sequence[str],
        allowed_suffixes: Sequence[str],
        maximum_depth: int = 32,
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
            isinstance(maximum_depth, bool)
            or not isinstance(maximum_depth, int)
            or maximum_depth <= 0
        ):
            raise ValueError("maximum_depth must be a positive integer.")
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
        self.maximum_depth = maximum_depth
        self.policy_id = canonical_fingerprint(
            {
                "kind": "external-artifact-policy",
                "root": self.root,
                "maximum_bytes": maximum_bytes,
                "allowed_license_ids": list(licenses),
                "allowed_suffixes": list(suffixes),
                "maximum_depth": maximum_depth,
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


def _normalized_artifact_path(
    relative_path: str | os.PathLike[str],
    policy: ExternalArtifactPolicy,
    /,
) -> str:
    relative = Path(relative_path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("External artifact path must be a confined relative path.")
    normalized = relative.as_posix()
    if normalized in ("", "."):
        raise ValueError("External artifact path must identify one file.")
    if relative.suffix.lower() not in policy.allowed_suffixes:
        raise ValueError("External artifact suffix is not admitted by policy.")
    return normalized


def _artifact_resource_limits(policy: ExternalArtifactPolicy, /) -> ResourceLimits:
    return ResourceLimits(
        policy.maximum_bytes,
        policy.maximum_depth,
        1,
        0,
        0,
    )


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
    normalized = _normalized_artifact_path(relative_path, policy)
    try:
        resource = read_bounded_resource(
            normalized,
            trusted_root=policy.root,
            limits=_artifact_resource_limits(policy),
        )
    except ResourceReadError as error:
        raise ValueError(
            "External artifact paths cannot contain symbolic links or special files."
        ) from error
    if resource.manifest.size_bytes != manifest.byte_size:
        raise ValueError("External artifact size does not match its manifest.")
    if resource.manifest.content_sha256 != manifest.sha256:
        raise ValueError("External artifact SHA-256 checksum mismatch.")
    if manifest.license_id not in policy.allowed_license_ids:
        raise PermissionError("External artifact license is not admitted by policy.")
    resolved = resource.manifest.source_path
    if resolved is None:
        raise RuntimeError("External artifact file identity is missing.")
    return AdmittedExternalArtifact(
        normalized,
        resolved,
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
    normalized = _normalized_artifact_path(artifact.relative_path, policy)
    try:
        resource = read_bounded_resource(
            normalized,
            trusted_root=policy.root,
            limits=_artifact_resource_limits(policy),
        )
    except ResourceReadError as error:
        raise ValueError(
            "External artifact paths cannot contain symbolic links or special files."
        ) from error
    if (
        normalized != artifact.relative_path
        or resource.manifest.source_path != artifact.resolved_path
    ):
        raise ValueError("External artifact path changed after admission.")
    if resource.manifest.size_bytes != artifact.byte_size:
        raise ValueError("External artifact size does not match its manifest.")
    if resource.manifest.content_sha256 != artifact.sha256:
        raise ValueError("External artifact SHA-256 checksum mismatch.")
    return resource.data


__all__ = (
    "AdmittedExternalArtifact",
    "ExternalArtifactPolicy",
    "admit_external_artifact",
    "read_admitted_artifact",
)
