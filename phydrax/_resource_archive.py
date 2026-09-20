#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded admission of externally supplied ZIP-based resource containers."""

from __future__ import annotations

import hashlib
import io
import stat
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

from ._external_resource import BoundedResource, ResourceReadError
from ._fingerprint import canonical_fingerprint


_HASH_CHUNK_BYTES = 1_048_576
_ALLOWED_COMPRESSIONS = frozenset(
    {
        zipfile.ZIP_STORED,
        zipfile.ZIP_DEFLATED,
        zipfile.ZIP_BZIP2,
        zipfile.ZIP_LZMA,
    }
)


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Finite bounds for one external ZIP-compatible resource container."""

    max_container_bytes: int
    max_members: int
    max_member_bytes: int
    max_total_uncompressed_bytes: int
    max_name_bytes: int
    max_depth: int
    max_compression_ratio: int

    def __post_init__(self) -> None:
        values = (
            self.max_container_bytes,
            self.max_members,
            self.max_member_bytes,
            self.max_total_uncompressed_bytes,
            self.max_name_bytes,
            self.max_depth,
            self.max_compression_ratio,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("External archive limits must be positive integers.")
        if self.max_member_bytes > self.max_total_uncompressed_bytes:
            raise ValueError("Archive member bytes cannot exceed aggregate bytes.")


@dataclass(frozen=True, slots=True)
class ArchiveMemberManifest:
    """Identity and encoding evidence for one admitted regular archive member."""

    relative_path: str
    compressed_size_bytes: int
    size_bytes: int
    compression: int
    content_sha256: str
    manifest_id: str


@dataclass(frozen=True, slots=True)
class BoundedArchive:
    """One immutable resident container and its exact admitted member inventory."""

    resource: BoundedResource
    members: tuple[ArchiveMemberManifest, ...]
    total_uncompressed_bytes: int
    manifest_id: str


def admit_zip_resource(
    resource: BoundedResource,
    /,
    *,
    limits: ArchiveLimits,
) -> BoundedArchive:
    """Preflight and checksum every regular member of a resident ZIP resource."""

    if not isinstance(resource, BoundedResource):
        raise TypeError("resource must be a BoundedResource.")
    if resource.manifest.size_bytes > limits.max_container_bytes:
        raise ResourceReadError("limit", "Archive exceeds its container byte limit.")
    manifests: list[ArchiveMemberManifest] = []
    total = 0
    canonical_names: set[str] = set()
    try:
        with zipfile.ZipFile(io.BytesIO(resource.data), mode="r") as archive:
            information = archive.infolist()
            names = [member.filename for member in information]
            if len(names) != len(set(names)):
                raise ResourceReadError(
                    "malformed", "Archive contains duplicate members."
                )
            for member in information:
                relative = _canonical_member_path(member)
                if relative is None:
                    continue
                if relative in canonical_names:
                    raise ResourceReadError(
                        "malformed", "Archive contains duplicate canonical member paths."
                    )
                canonical_names.add(relative)
                if len(manifests) >= limits.max_members:
                    raise ResourceReadError("limit", "Archive exceeds its member limit.")
                encoded_name = relative.encode("utf-8")
                if len(encoded_name) > limits.max_name_bytes:
                    raise ResourceReadError(
                        "limit", "Archive member name exceeds its byte limit."
                    )
                if len(PurePosixPath(relative).parts) > limits.max_depth:
                    raise ResourceReadError(
                        "limit", "Archive member nesting exceeds its depth limit."
                    )
                if member.flag_bits & 0x1:
                    raise ResourceReadError(
                        "policy", "Encrypted archive members are disabled."
                    )
                if member.compress_type not in _ALLOWED_COMPRESSIONS:
                    raise ResourceReadError(
                        "policy", "Archive compression method is not admitted."
                    )
                if member.file_size > limits.max_member_bytes:
                    raise ResourceReadError(
                        "limit", "Archive member exceeds its byte limit."
                    )
                if total > limits.max_total_uncompressed_bytes - member.file_size:
                    raise ResourceReadError(
                        "limit", "Archive exceeds its aggregate byte limit."
                    )
                denominator = max(1, member.compress_size)
                if member.file_size > limits.max_compression_ratio * denominator:
                    raise ResourceReadError(
                        "limit", "Archive member exceeds its compression-ratio limit."
                    )
                checksum, observed = _member_identity(
                    archive, member, limits.max_member_bytes
                )
                if observed != member.file_size:
                    raise ResourceReadError(
                        "malformed", "Archive member size changed while decoding."
                    )
                total += observed
                payload = {
                    "kind": "bounded-archive-member-manifest",
                    "relative_path": relative,
                    "compressed_size_bytes": member.compress_size,
                    "size_bytes": observed,
                    "compression": member.compress_type,
                    "content_sha256": checksum,
                }
                manifests.append(
                    ArchiveMemberManifest(
                        relative,
                        member.compress_size,
                        observed,
                        member.compress_type,
                        checksum,
                        canonical_fingerprint(payload),
                    )
                )
    except ResourceReadError:
        raise
    except (OSError, RuntimeError, UnicodeError, zipfile.BadZipFile) as error:
        raise ResourceReadError("malformed", "External archive is invalid.") from error
    ordered = tuple(sorted(manifests, key=lambda member: member.relative_path))
    payload = {
        "kind": "bounded-archive-manifest",
        "resource_manifest_id": resource.manifest.manifest_id,
        "members": [member.manifest_id for member in ordered],
        "total_uncompressed_bytes": total,
        "limits": {
            "max_container_bytes": limits.max_container_bytes,
            "max_members": limits.max_members,
            "max_member_bytes": limits.max_member_bytes,
            "max_total_uncompressed_bytes": limits.max_total_uncompressed_bytes,
            "max_name_bytes": limits.max_name_bytes,
            "max_depth": limits.max_depth,
            "max_compression_ratio": limits.max_compression_ratio,
        },
    }
    return BoundedArchive(
        resource,
        ordered,
        total,
        canonical_fingerprint(payload),
    )


def read_zip_members(
    archive: BoundedArchive,
    names: tuple[str, ...],
    /,
) -> dict[str, bytes]:
    """Read an exact subset after matching it to an admitted archive inventory."""

    if not isinstance(archive, BoundedArchive):
        raise TypeError("archive must be a BoundedArchive.")
    if isinstance(names, str) or not names or len(names) != len(set(names)):
        raise ValueError("Archive member names must be a non-empty unique tuple.")
    inventory = {member.relative_path: member for member in archive.members}
    if any(name not in inventory for name in names):
        raise ResourceReadError("policy", "Requested archive member was not admitted.")
    values: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(archive.resource.data), mode="r") as container:
            for name in names:
                expected = inventory[name]
                with container.open(name, mode="r") as stream:
                    payload = stream.read(expected.size_bytes + 1)
                if (
                    len(payload) != expected.size_bytes
                    or hashlib.sha256(payload).hexdigest() != expected.content_sha256
                ):
                    raise ResourceReadError(
                        "inconsistent", "Archive member changed after admission."
                    )
                values[name] = payload
    except ResourceReadError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise ResourceReadError("malformed", "External archive is invalid.") from error
    return values


def _canonical_member_path(member: zipfile.ZipInfo, /) -> str | None:
    name = member.filename.replace("\\", "/")
    if "\x00" in name:
        raise ResourceReadError("malformed", "Archive member name contains a null byte.")
    path = PurePosixPath(name)
    parts = path.parts
    if not parts or path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ResourceReadError("policy", "Archive member path is not confined.")
    file_type = (member.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise ResourceReadError("policy", "Archive symbolic links are disabled.")
    if member.is_dir():
        if file_type not in (0, stat.S_IFDIR):
            raise ResourceReadError("policy", "Archive directory type is invalid.")
        return None
    if file_type not in (0, stat.S_IFREG):
        raise ResourceReadError("policy", "Archive special files are disabled.")
    return path.as_posix()


def _member_identity(
    archive: zipfile.ZipFile,
    member: zipfile.ZipInfo,
    maximum_bytes: int,
    /,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    with archive.open(member, mode="r") as stream:
        while total <= maximum_bytes:
            chunk = stream.read(min(_HASH_CHUNK_BYTES, maximum_bytes + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    if total > maximum_bytes:
        raise ResourceReadError("limit", "Archive member exceeds its byte limit.")
    return digest.hexdigest(), total


__all__ = [
    "ArchiveLimits",
    "ArchiveMemberManifest",
    "BoundedArchive",
    "admit_zip_resource",
    "read_zip_members",
]
