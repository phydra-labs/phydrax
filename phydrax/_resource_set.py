#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded, descriptor-relative admission of local multi-file resources."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Iterator, Literal

from ._fingerprint import canonical_fingerprint
from ._host_io import directory_identity, open_directory_beneath, stat_identity


_ResourceSetFailure = Literal["policy", "malformed", "limit", "inconsistent"]
_DIRECTORY_FLAGS = (
    os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
)
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC
_HASH_CHUNK_BYTES = 1_048_576


class ResourceSetReadError(ValueError):
    """Fail-closed resource-set error with a stable failure category."""

    reason: _ResourceSetFailure

    def __init__(self, reason: _ResourceSetFailure, message: str, /):
        self.reason = reason
        super().__init__(str(message))


@dataclass(frozen=True, slots=True)
class ResourceSetLimits:
    """Finite byte, total-entry, and depth bounds for one directory resource."""

    max_total_bytes: int
    max_member_bytes: int
    max_members: int
    max_depth: int

    def __post_init__(self) -> None:
        values = (
            self.max_total_bytes,
            self.max_member_bytes,
            self.max_members,
            self.max_depth,
        )
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("Resource-set limits must be positive integers.")
        if self.max_member_bytes > self.max_total_bytes:
            raise ValueError("Resource-set member bytes cannot exceed total bytes.")


@dataclass(frozen=True, slots=True)
class ResourceMemberManifest:
    """Identity and filesystem evidence for one regular resource-set member."""

    relative_path: str
    size_bytes: int
    content_sha256: str
    file_device: int
    file_inode: int
    file_mode: int
    manifest_id: str


@dataclass(frozen=True, slots=True)
class ResourceSetManifest:
    """Aggregate identity and exact member inventory for one resource set."""

    source_path: str
    trusted_root: str
    trusted_root_device: int
    trusted_root_inode: int
    trusted_root_mode: int
    relative_components: tuple[str, ...]
    members: tuple[ResourceMemberManifest, ...]
    total_size_bytes: int
    total_entry_count: int
    aggregate_sha256: str
    limits: ResourceSetLimits
    manifest_id: str


@dataclass(frozen=True, slots=True)
class BoundedResourceSet:
    """One exact admitted local resource-set inventory."""

    manifest: ResourceSetManifest


@dataclass(slots=True)
class OpenedResourceSet:
    """Descriptor-held resource set whose admitted members can be read safely."""

    descriptor: int
    manifest: ResourceSetManifest

    def read_member(
        self,
        relative_path: str,
        /,
        *,
        maximum_bytes: int | None = None,
    ) -> bytes:
        if not isinstance(relative_path, str):
            raise TypeError("relative_path must be a string.")
        maximum = (
            self.manifest.limits.max_member_bytes
            if maximum_bytes is None
            else maximum_bytes
        )
        if type(maximum) is not int or maximum <= 0:
            raise ValueError("maximum_bytes must be a positive integer or None.")
        path = PurePosixPath(relative_path)
        if (
            not relative_path
            or "\\" in relative_path
            or "\x00" in relative_path
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ResourceSetReadError(
                "policy", "Resource-set member path is not confined."
            )
        inventory = {member.relative_path: member for member in self.manifest.members}
        expected = inventory.get(path.as_posix())
        if expected is None:
            raise ResourceSetReadError(
                "policy", "Requested resource-set member was not admitted."
            )
        if expected.size_bytes > maximum:
            raise ResourceSetReadError(
                "limit", "Resource-set member exceeds the requested byte limit."
            )
        descriptors = [os.dup(self.descriptor)]
        try:
            for component in path.parts[:-1]:
                descriptors.append(
                    os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptors[-1])
                )
            descriptor = os.open(path.parts[-1], _FILE_FLAGS, dir_fd=descriptors[-1])
            descriptors.append(descriptor)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or int(opened.st_dev) != expected.file_device
                or int(opened.st_ino) != expected.file_inode
                or int(opened.st_mode) != expected.file_mode
                or opened.st_size != expected.size_bytes
            ):
                raise ResourceSetReadError(
                    "inconsistent",
                    "A resource-set member changed after admission.",
                )
            digest = hashlib.sha256()
            payload = bytearray()
            while len(payload) <= expected.size_bytes:
                chunk = os.read(
                    descriptor,
                    min(
                        _HASH_CHUNK_BYTES,
                        expected.size_bytes + 1 - len(payload),
                    ),
                )
                if not chunk:
                    break
                payload.extend(chunk)
                digest.update(chunk)
            after = os.fstat(descriptor)
            if (
                len(payload) != expected.size_bytes
                or digest.hexdigest() != expected.content_sha256
                or stat_identity(opened) != stat_identity(after)
            ):
                raise ResourceSetReadError(
                    "inconsistent",
                    "A resource-set member changed after admission.",
                )
            return bytes(payload)
        except ResourceSetReadError:
            raise
        except OSError as error:
            raise ResourceSetReadError(
                "inconsistent",
                "An admitted resource-set member changed before it could be read.",
            ) from error
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


@contextmanager
def open_bounded_resource_set(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    limits: ResourceSetLimits,
) -> Iterator[OpenedResourceSet]:
    """Hold an admitted resource-set directory for exact member reads."""

    consumer_boundary = False
    try:
        with open_directory_beneath(
            path,
            trusted_root=trusted_root,
            maximum_depth=limits.max_depth,
        ) as opened:
            members: list[ResourceMemberManifest] = []
            totals = [0, 0]
            _walk_directory(
                opened.descriptor,
                prefix=(),
                depth=0,
                limits=limits,
                members=members,
                totals=totals,
            )
            opened.verify_stable()
            ordered = tuple(sorted(members, key=lambda member: member.relative_path))
            aggregate = hashlib.sha256()
            for member in ordered:
                for part in (
                    member.relative_path.encode("utf-8"),
                    member.size_bytes.to_bytes(8, "big"),
                    bytes.fromhex(member.content_sha256),
                ):
                    aggregate.update(len(part).to_bytes(8, "big"))
                    aggregate.update(part)
            digest = aggregate.hexdigest()
            source_path = os.path.join(opened.root_path, *opened.components)
            payload = {
                "kind": "bounded-resource-set-manifest",
                "source_path": source_path,
                "trusted_root": opened.root_path,
                "trusted_root_device": int(opened.root_status.st_dev),
                "trusted_root_inode": int(opened.root_status.st_ino),
                "trusted_root_mode": int(opened.root_status.st_mode),
                "relative_components": list(opened.components),
                "members": [
                    {
                        "relative_path": member.relative_path,
                        "size_bytes": member.size_bytes,
                        "content_sha256": member.content_sha256,
                        "file_device": member.file_device,
                        "file_inode": member.file_inode,
                        "file_mode": member.file_mode,
                        "manifest_id": member.manifest_id,
                    }
                    for member in ordered
                ],
                "total_size_bytes": totals[0],
                "total_entry_count": totals[1],
                "aggregate_sha256": digest,
                "limits": {
                    "max_total_bytes": limits.max_total_bytes,
                    "max_member_bytes": limits.max_member_bytes,
                    "max_members": limits.max_members,
                    "max_depth": limits.max_depth,
                },
            }
            manifest = ResourceSetManifest(
                source_path,
                opened.root_path,
                int(opened.root_status.st_dev),
                int(opened.root_status.st_ino),
                int(opened.root_status.st_mode),
                opened.components,
                ordered,
                totals[0],
                totals[1],
                digest,
                limits,
                canonical_fingerprint(payload),
            )
            consumer_boundary = True
            yield OpenedResourceSet(opened.descriptor, manifest)
            consumer_boundary = False
            opened.verify_stable()
    except ResourceSetReadError:
        raise
    except OverflowError as error:
        if consumer_boundary:
            raise
        raise ResourceSetReadError("limit", str(error)) from error
    except ValueError as error:
        if consumer_boundary:
            raise
        raise ResourceSetReadError("policy", str(error)) from error
    except RuntimeError as error:
        if consumer_boundary:
            raise
        raise ResourceSetReadError("inconsistent", str(error)) from error
    except OSError as error:
        if consumer_boundary:
            raise
        reason: _ResourceSetFailure = (
            "policy" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "malformed"
        )
        raise ResourceSetReadError(
            reason, "The requested resource set could not be opened or read."
        ) from error


def read_bounded_resource_set(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    limits: ResourceSetLimits,
) -> BoundedResourceSet:
    """Admit all entries beneath one descriptor-held directory."""

    with open_bounded_resource_set(
        path, trusted_root=trusted_root, limits=limits
    ) as opened:
        return BoundedResourceSet(opened.manifest)


def _walk_directory(
    directory_descriptor: int,
    /,
    *,
    prefix: tuple[str, ...],
    depth: int,
    limits: ResourceSetLimits,
    members: list[ResourceMemberManifest],
    totals: list[int],
) -> None:
    if depth > limits.max_depth:
        raise ResourceSetReadError("limit", "Resource-set nesting exceeds its limit.")
    before = os.fstat(directory_descriptor)
    if not stat.S_ISDIR(before.st_mode):
        raise ResourceSetReadError(
            "policy", "Resource-set members must be regular files or directories."
        )
    names: list[str] = []
    with os.scandir(directory_descriptor) as entries:
        for entry in entries:
            if totals[1] >= limits.max_members:
                raise ResourceSetReadError(
                    "limit", "Resource set exceeds its total entry limit."
                )
            totals[1] += 1
            names.append(entry.name)
    for name in sorted(names):
        if not name or name in {".", ".."} or "/" in name or "\x00" in name:
            raise ResourceSetReadError(
                "malformed", "Resource-set member name is invalid."
            )
        try:
            name.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ResourceSetReadError(
                "malformed", "Resource-set member names must be valid UTF-8."
            ) from error
        information = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        relative_parts = (*prefix, name)
        relative_path = PurePosixPath(*relative_parts).as_posix()
        if stat.S_ISLNK(information.st_mode):
            raise ResourceSetReadError(
                "policy", "Resource sets cannot contain symbolic links."
            )
        if stat.S_ISDIR(information.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
            try:
                opened = os.fstat(child)
                if stat_identity(information) != stat_identity(opened):
                    raise ResourceSetReadError(
                        "inconsistent", "A resource-set directory changed while opening."
                    )
                _walk_directory(
                    child,
                    prefix=relative_parts,
                    depth=depth + 1,
                    limits=limits,
                    members=members,
                    totals=totals,
                )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(information.st_mode):
            raise ResourceSetReadError(
                "policy", "Resource sets cannot contain special files."
            )
        if information.st_size > limits.max_member_bytes:
            raise ResourceSetReadError(
                "limit", "Resource-set member exceeds its byte limit."
            )
        if totals[0] > limits.max_total_bytes - information.st_size:
            raise ResourceSetReadError(
                "limit", "Resource set exceeds its aggregate byte limit."
            )
        member = _read_member(
            directory_descriptor,
            name,
            relative_path,
            information,
            limits,
        )
        totals[0] += member.size_bytes
        members.append(member)
    after = os.fstat(directory_descriptor)
    if directory_identity(before) != directory_identity(after):
        raise ResourceSetReadError(
            "inconsistent", "A resource-set directory changed during admission."
        )


def _read_member(
    directory_descriptor: int,
    name: str,
    relative_path: str,
    initial: os.stat_result,
    limits: ResourceSetLimits,
    /,
) -> ResourceMemberManifest:
    descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat_identity(initial) != stat_identity(
            opened
        ):
            raise ResourceSetReadError(
                "inconsistent", "A resource-set member changed while opening."
            )
        digest = hashlib.sha256()
        total = 0
        while total <= limits.max_member_bytes:
            chunk = os.read(
                descriptor,
                min(_HASH_CHUNK_BYTES, limits.max_member_bytes + 1 - total),
            )
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
        after = os.fstat(descriptor)
        if total > limits.max_member_bytes:
            raise ResourceSetReadError(
                "limit", "Resource-set member exceeds its byte limit."
            )
        if total != after.st_size or stat_identity(opened) != stat_identity(after):
            raise ResourceSetReadError(
                "inconsistent", "A resource-set member changed during admission."
            )
        checksum = digest.hexdigest()
        payload = {
            "kind": "resource-set-member-manifest",
            "relative_path": relative_path,
            "size_bytes": total,
            "content_sha256": checksum,
            "file_device": int(after.st_dev),
            "file_inode": int(after.st_ino),
            "file_mode": int(after.st_mode),
        }
        return ResourceMemberManifest(
            relative_path,
            total,
            checksum,
            int(after.st_dev),
            int(after.st_ino),
            int(after.st_mode),
            canonical_fingerprint(payload),
        )
    finally:
        os.close(descriptor)


__all__ = [
    "BoundedResourceSet",
    "OpenedResourceSet",
    "ResourceMemberManifest",
    "ResourceSetLimits",
    "ResourceSetManifest",
    "ResourceSetReadError",
    "open_bounded_resource_set",
    "read_bounded_resource_set",
]
