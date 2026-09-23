#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded admission of externally supplied ZIP-based resource containers."""

from __future__ import annotations

import hashlib
import io
import stat
import struct
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
_CENTRAL_DIRECTORY_SIGNATURE = b"PK\x01\x02"
_END_DIRECTORY_SIGNATURE = b"PK\x05\x06"
_ZIP64_END_DIRECTORY_SIGNATURE = b"PK\x06\x06"
_ZIP64_END_DIRECTORY_LOCATOR_SIGNATURE = b"PK\x06\x07"


@dataclass(frozen=True, slots=True)
class ArchiveLimits:
    """Finite container, total-directory-entry, decoded-byte, and path bounds."""

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
    directory_entry_count: int
    central_directory_size_bytes: int
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
    entry_count, directory_size, directory_offset = _preflight_zip_directory(
        resource.data
    )
    if entry_count > limits.max_members:
        raise ResourceReadError(
            "limit", "Archive exceeds its total directory-entry limit."
        )
    maximum_directory_bytes = min(
        limits.max_container_bytes,
        limits.max_members * (46 + limits.max_name_bytes + 2 * ((1 << 16) - 1)),
    )
    if directory_size > maximum_directory_bytes:
        raise ResourceReadError(
            "limit", "Archive central directory exceeds its byte limit."
        )
    observed_entries = _preflight_central_directory(
        resource.data,
        offset=directory_offset,
        size=directory_size,
        limits=limits,
    )
    if observed_entries != entry_count:
        raise ResourceReadError(
            "malformed", "Archive directory-entry count is inconsistent."
        )
    try:
        with zipfile.ZipFile(io.BytesIO(resource.data), mode="r") as archive:
            information = archive.infolist()
            if len(information) != entry_count:
                raise ResourceReadError(
                    "malformed", "Archive directory-entry count is inconsistent."
                )
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
        "directory_entry_count": entry_count,
        "central_directory_size_bytes": directory_size,
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
        entry_count,
        directory_size,
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


def _preflight_zip_directory(data: bytes, /) -> tuple[int, int, int]:
    search_start = max(0, len(data) - 65_557)
    search_end = len(data)
    end_offset = -1
    fields: tuple[bytes, int, int, int, int, int, int, int] | None = None
    while search_end > search_start:
        candidate = data.rfind(_END_DIRECTORY_SIGNATURE, search_start, search_end)
        if candidate < 0:
            break
        if candidate + 22 <= len(data):
            parsed = struct.unpack_from("<4s4H2IH", data, candidate)
            if candidate + 22 + parsed[-1] == len(data):
                end_offset = candidate
                fields = parsed
                break
        search_end = candidate
    if fields is None:
        raise ResourceReadError("malformed", "Archive end directory is invalid.")
    _, disk, directory_disk, disk_entries, entries, size, offset, _ = fields
    if disk != 0 or directory_disk != 0 or disk_entries != entries:
        raise ResourceReadError("policy", "Multi-disk archives are disabled.")
    if entries != 0xFFFF and size != 0xFFFFFFFF and offset != 0xFFFFFFFF:
        if offset > end_offset or size != end_offset - offset:
            raise ResourceReadError(
                "malformed", "Archive central directory is inconsistent."
            )
        return entries, size, offset

    locator_offset = end_offset - 20
    if locator_offset < 0:
        raise ResourceReadError("malformed", "ZIP64 directory locator is missing.")
    locator = struct.unpack_from("<4sIQI", data, locator_offset)
    if (
        locator[0] != _ZIP64_END_DIRECTORY_LOCATOR_SIGNATURE
        or locator[1] != 0
        or locator[3] != 1
    ):
        raise ResourceReadError("policy", "Multi-disk ZIP64 archives are disabled.")
    zip64_offset = locator[2]
    if zip64_offset > locator_offset or zip64_offset + 56 > locator_offset:
        raise ResourceReadError("malformed", "ZIP64 end directory is invalid.")
    zip64 = struct.unpack_from("<4sQ2H2I4Q", data, zip64_offset)
    if (
        zip64[0] != _ZIP64_END_DIRECTORY_SIGNATURE
        or zip64[1] < 44
        or zip64_offset + 12 + zip64[1] > locator_offset
        or zip64[4] != 0
        or zip64[5] != 0
        or zip64[6] != zip64[7]
    ):
        raise ResourceReadError("malformed", "ZIP64 end directory is inconsistent.")
    entries = zip64[7]
    size = zip64[8]
    offset = zip64[9]
    if offset > zip64_offset or size != zip64_offset - offset:
        raise ResourceReadError("malformed", "ZIP64 central directory is inconsistent.")
    return entries, size, offset


def _preflight_central_directory(
    data: bytes,
    /,
    *,
    offset: int,
    size: int,
    limits: ArchiveLimits,
) -> int:
    end = offset + size
    if offset < 0 or end > len(data):
        raise ResourceReadError(
            "malformed", "Archive central directory is outside the container."
        )
    cursor = offset
    entries = 0
    while cursor < end:
        if entries >= limits.max_members:
            raise ResourceReadError(
                "limit", "Archive exceeds its total directory-entry limit."
            )
        if end - cursor < 46 or data[cursor : cursor + 4] != _CENTRAL_DIRECTORY_SIGNATURE:
            raise ResourceReadError(
                "malformed", "Archive central directory header is invalid."
            )
        fields = struct.unpack_from("<4s6H3I5H2I", data, cursor)
        name_size = fields[10]
        extra_size = fields[11]
        comment_size = fields[12]
        disk = fields[13]
        if disk != 0:
            raise ResourceReadError("policy", "Multi-disk archives are disabled.")
        if name_size > limits.max_name_bytes:
            raise ResourceReadError(
                "limit", "Archive member name exceeds its byte limit."
            )
        record_size = 46 + name_size + extra_size + comment_size
        if record_size > end - cursor:
            raise ResourceReadError(
                "malformed", "Archive central directory record is truncated."
            )
        cursor += record_size
        entries += 1
    if cursor != end:
        raise ResourceReadError(
            "malformed", "Archive central directory size is inconsistent."
        )
    return entries


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
