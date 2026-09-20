#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Crash-consistent publication of validated local files and resource sets."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
import secrets
import stat
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Literal

from ._fingerprint import canonical_fingerprint
from ._host_io import descriptor_relative_path, open_parent_descriptor
from ._resource_set import ResourceSetLimits


PublicationMode = Literal["exclusive", "atomic_replace"]
_HASH_CHUNK_BYTES = 1_048_576
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_FILE_FLAGS = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC


@dataclass(frozen=True, slots=True)
class PublicationReceipt:
    """Durable identity of one atomically published regular file."""

    destination: str
    size_bytes: int
    content_sha256: str
    mode: PublicationMode
    replaced_existing: bool
    receipt_id: str


@dataclass(frozen=True, slots=True)
class PublishedMemberReceipt:
    """Durable identity of one file in a published resource set."""

    relative_path: str
    size_bytes: int
    content_sha256: str
    receipt_id: str


@dataclass(frozen=True, slots=True)
class ResourceSetPublicationReceipt:
    """Durable identity of one atomically published resource-set generation."""

    destination: str
    members: tuple[PublishedMemberReceipt, ...]
    total_size_bytes: int
    aggregate_sha256: str
    mode: PublicationMode
    replaced_existing: bool
    receipt_id: str


def publish_bytes(
    path: str | os.PathLike[str],
    data: bytes,
    /,
    *,
    maximum_bytes: int,
    mode: PublicationMode = "exclusive",
    validator: Callable[[os.PathLike[str]], None] | None = None,
) -> PublicationReceipt:
    """Publish exact bytes after optional validation of their staged file."""

    if not isinstance(data, bytes):
        raise TypeError("Published data must be bytes.")

    def writer(stream: BinaryIO) -> None:
        stream.write(data)

    return publish_file(
        path,
        writer,
        maximum_bytes=maximum_bytes,
        mode=mode,
        validator=validator,
    )


def publish_file(
    path: str | os.PathLike[str],
    writer: Callable[[BinaryIO], None],
    /,
    *,
    maximum_bytes: int,
    mode: PublicationMode = "exclusive",
    validator: Callable[[os.PathLike[str]], None] | None = None,
) -> PublicationReceipt:
    """Write, validate, synchronize, and atomically expose one regular file."""

    if not callable(writer):
        raise TypeError("writer must be callable.")
    if type(maximum_bytes) is not int or maximum_bytes <= 0:
        raise ValueError("maximum_bytes must be a positive integer.")
    if mode not in ("exclusive", "atomic_replace"):
        raise ValueError("Publication mode must be 'exclusive' or 'atomic_replace'.")
    if validator is not None and not callable(validator):
        raise TypeError("validator must be callable or None.")
    destination = Path(os.fspath(path))
    parent_descriptor, destination_name = open_parent_descriptor(path, create=True)
    temporary_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    descriptor = -1
    replaced = False
    try:
        descriptor = os.open(
            temporary_name,
            _FILE_FLAGS,
            0o600,
            dir_fd=parent_descriptor,
        )
        with os.fdopen(descriptor, "w+b") as stream:
            descriptor = -1
            writer(stream)
            stream.flush()
            size, digest = _stream_identity(stream, maximum_bytes)
            os.fsync(stream.fileno())
        staged = descriptor_relative_path(parent_descriptor, temporary_name)
        if validator is not None:
            validator(staged)
        try:
            existing = os.stat(
                destination_name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("Publication destination must be absent or a regular file.")
        replaced = existing is not None
        if mode == "exclusive":
            os.link(
                temporary_name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        else:
            os.replace(
                temporary_name,
                destination_name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
        os.fsync(parent_descriptor)
        payload = {
            "kind": "file-publication-receipt",
            "destination": str(destination),
            "size_bytes": size,
            "content_sha256": digest,
            "mode": mode,
            "replaced_existing": replaced,
        }
        return PublicationReceipt(
            str(destination),
            size,
            digest,
            mode,
            replaced,
            canonical_fingerprint(payload),
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def publish_resource_set(
    path: str | os.PathLike[str],
    members: Mapping[str, bytes],
    /,
    *,
    limits: ResourceSetLimits,
    mode: PublicationMode = "exclusive",
) -> ResourceSetPublicationReceipt:
    """Publish one immutable resource-set generation with an explicit commit mode."""

    if not isinstance(members, Mapping) or not members:
        raise ValueError("Published resource sets require a non-empty member mapping.")
    if mode not in ("exclusive", "atomic_replace"):
        raise ValueError("Publication mode must be 'exclusive' or 'atomic_replace'.")
    destination = Path(os.fspath(path))
    parent_descriptor, destination_name = open_parent_descriptor(destination, create=True)
    temporary_name = f".{destination_name}.{secrets.token_hex(16)}.tmp"
    temporary_descriptor = -1
    try:
        os.mkdir(temporary_name, mode=0o700, dir_fd=parent_descriptor)
        temporary_descriptor = os.open(
            temporary_name,
            _DIRECTORY_FLAGS,
            dir_fd=parent_descriptor,
        )
        receipts: list[PublishedMemberReceipt] = []
        total = 0
        canonical_members: dict[str, bytes] = {}
        for name, data in members.items():
            relative = _canonical_relative_path(name, limits.max_depth)
            if relative in canonical_members:
                raise ValueError("Published resource-set member paths must be unique.")
            if not isinstance(data, bytes):
                raise TypeError("Published resource-set members must be bytes.")
            if len(data) > limits.max_member_bytes:
                raise ValueError("Published resource-set member exceeds its byte limit.")
            if total > limits.max_total_bytes - len(data):
                raise ValueError(
                    "Published resource set exceeds its aggregate byte limit."
                )
            if len(canonical_members) >= limits.max_members:
                raise ValueError("Published resource set exceeds its member limit.")
            canonical_members[relative] = data
            total += len(data)
        for relative in sorted(canonical_members):
            data = canonical_members[relative]
            _write_resource_member(temporary_descriptor, relative, data)
            digest = hashlib.sha256(data).hexdigest()
            payload = {
                "kind": "published-resource-set-member",
                "relative_path": relative,
                "size_bytes": len(data),
                "content_sha256": digest,
            }
            receipts.append(
                PublishedMemberReceipt(
                    relative,
                    len(data),
                    digest,
                    canonical_fingerprint(payload),
                )
            )
        _fsync_tree(temporary_descriptor)
        aggregate = hashlib.sha256()
        for receipt in receipts:
            for part in (
                receipt.relative_path.encode("utf-8"),
                receipt.size_bytes.to_bytes(8, "big"),
                bytes.fromhex(receipt.content_sha256),
            ):
                aggregate.update(len(part).to_bytes(8, "big"))
                aggregate.update(part)
        digest = aggregate.hexdigest()
        try:
            existing_descriptor = os.open(
                destination_name,
                _DIRECTORY_FLAGS,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            existing_descriptor = -1
        except OSError as error:
            if error.errno in (errno.ELOOP, errno.ENOTDIR):
                raise FileExistsError(destination) from error
            raise
        replaced = existing_descriptor >= 0
        existing_empty = False
        if existing_descriptor >= 0:
            try:
                existing_empty = not os.listdir(existing_descriptor)
            finally:
                os.close(existing_descriptor)
        if replaced and mode == "atomic_replace":
            _atomic_exchange_directories(
                parent_descriptor,
                temporary_name,
                destination_name,
            )
        else:
            if replaced:
                if not existing_empty:
                    raise FileExistsError(destination)
                os.rmdir(destination_name, dir_fd=parent_descriptor)
                os.fsync(parent_descriptor)
            try:
                os.rename(
                    temporary_name,
                    destination_name,
                    src_dir_fd=parent_descriptor,
                    dst_dir_fd=parent_descriptor,
                )
            except OSError as error:
                if error.errno in (errno.EEXIST, errno.ENOTEMPTY):
                    raise FileExistsError(destination) from error
                raise
        os.close(temporary_descriptor)
        temporary_descriptor = -1
        os.fsync(parent_descriptor)
        payload = {
            "kind": "resource-set-publication-receipt",
            "destination": str(destination),
            "members": [receipt.receipt_id for receipt in receipts],
            "total_size_bytes": total,
            "aggregate_sha256": digest,
            "mode": mode,
            "replaced_existing": replaced,
        }
        return ResourceSetPublicationReceipt(
            str(destination),
            tuple(receipts),
            total,
            digest,
            mode,
            replaced,
            canonical_fingerprint(payload),
        )
    finally:
        if temporary_descriptor >= 0:
            os.close(temporary_descriptor)
        try:
            _remove_tree_at(parent_descriptor, temporary_name)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def _stream_identity(stream: BinaryIO, maximum_bytes: int, /) -> tuple[int, str]:
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    if size > maximum_bytes:
        raise ValueError("Published file exceeds its configured byte limit.")
    stream.seek(0)
    digest = hashlib.sha256()
    total = 0
    while chunk := stream.read(_HASH_CHUNK_BYTES):
        total += len(chunk)
        if total > maximum_bytes:
            raise ValueError("Published file exceeds its configured byte limit.")
        digest.update(chunk)
    if total != size:
        raise RuntimeError("Published file changed while computing its identity.")
    stream.seek(0)
    return total, digest.hexdigest()


def _canonical_relative_path(value: str, maximum_depth: int, /) -> str:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        raise ValueError("Published resource-set member path is invalid.")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("Published resource-set member path is not confined.")
    if len(path.parts) > maximum_depth:
        raise ValueError("Published resource-set member path exceeds its depth limit.")
    return path.as_posix()


def _atomic_exchange_directories(
    parent_descriptor: int,
    temporary_name: str,
    destination_name: str,
    /,
) -> None:
    """Atomically exchange two sibling directories on supported POSIX hosts."""

    library = ctypes.CDLL(None, use_errno=True)
    source = temporary_name.encode()
    destination = destination_name.encode()
    if sys.platform == "darwin":
        operation = library.renameatx_np
    elif sys.platform.startswith("linux"):
        operation = library.renameat2
    else:
        raise OSError(errno.ENOTSUP, "Atomic directory exchange is unsupported.")
    operation.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    operation.restype = ctypes.c_int
    result = operation(
        parent_descriptor,
        source,
        parent_descriptor,
        destination,
        0x00000002,
    )
    if result != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    os.fsync(parent_descriptor)


def _write_resource_member(
    root_descriptor: int,
    relative_path: str,
    data: bytes,
    /,
) -> None:
    parts = PurePosixPath(relative_path).parts
    descriptor = os.dup(root_descriptor)
    try:
        for component in parts[:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=descriptor)
            except FileExistsError:
                pass
            following = os.open(component, _DIRECTORY_FLAGS, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = following
        file_descriptor = os.open(parts[-1], _FILE_FLAGS, 0o600, dir_fd=descriptor)
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(directory_descriptor: int, /) -> None:
    for name in sorted(os.listdir(directory_descriptor)):
        information = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISDIR(information.st_mode):
            child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
            try:
                _fsync_tree(child)
            finally:
                os.close(child)
        elif not stat.S_ISREG(information.st_mode):
            raise ValueError("Published resource set contains a special file.")
    os.fsync(directory_descriptor)


def _remove_tree_at(parent_descriptor: int, name: str, /) -> None:
    directory_descriptor = os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
    try:
        for child_name in os.listdir(directory_descriptor):
            information = os.stat(
                child_name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
            if stat.S_ISDIR(information.st_mode):
                _remove_tree_at(directory_descriptor, child_name)
            else:
                os.unlink(child_name, dir_fd=directory_descriptor)
    finally:
        os.close(directory_descriptor)
    os.rmdir(name, dir_fd=parent_descriptor)


__all__ = [
    "PublicationMode",
    "PublicationReceipt",
    "PublishedMemberReceipt",
    "ResourceSetPublicationReceipt",
    "publish_bytes",
    "publish_file",
    "publish_resource_set",
]
