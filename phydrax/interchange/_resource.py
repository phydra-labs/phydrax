#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Descriptor-relative, bounded reads of untrusted local resources."""

from __future__ import annotations

import errno
import hashlib
import os
import stat
from dataclasses import dataclass, replace
from typing import Literal

from .._fingerprint import canonical_fingerprint


_ResourceFailure = Literal["policy", "malformed", "limit", "inconsistent"]


class ResourceReadError(ValueError):
    """Fail-closed resource-read error with a stable failure category."""

    reason: _ResourceFailure

    def __init__(self, reason: _ResourceFailure, message: str, /):
        self.reason = reason
        super().__init__(str(message))


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Finite bounds shared by resource loading and structured decoding."""

    max_bytes: int
    max_depth: int
    max_nodes: int
    max_attributes: int
    max_losses: int

    def __post_init__(self) -> None:
        values = (
            int(self.max_bytes),
            int(self.max_depth),
            int(self.max_nodes),
            int(self.max_attributes),
            int(self.max_losses),
        )
        if values[0] <= 0 or values[1] <= 0 or values[2] <= 0:
            raise ValueError("Resource byte, depth, and node limits must be positive.")
        if values[3] < 0 or values[4] < 0:
            raise ValueError("Resource attribute and loss limits must be nonnegative.")
        object.__setattr__(self, "max_bytes", values[0])
        object.__setattr__(self, "max_depth", values[1])
        object.__setattr__(self, "max_nodes", values[2])
        object.__setattr__(self, "max_attributes", values[3])
        object.__setattr__(self, "max_losses", values[4])


@dataclass(frozen=True, slots=True)
class ResourceManifest:
    """Immutable provenance, identity, and bounds for one exact byte resource."""

    source_kind: Literal["memory", "file"]
    source_path: str | None
    trusted_root: str | None
    trusted_root_device: int | None
    trusted_root_inode: int | None
    trusted_root_mode: int | None
    relative_components: tuple[str, ...]
    size_bytes: int
    content_sha256: str
    file_device: int | None
    file_inode: int | None
    file_mode: int | None
    limits: ResourceLimits
    observed_depth: int
    observed_nodes: int
    observed_attributes: int
    observed_losses: int
    manifest_id: str


@dataclass(frozen=True, slots=True)
class BoundedResource:
    """Exact immutable resource bytes and their immutable manifest."""

    data: bytes
    manifest: ResourceManifest


def bounded_resource_from_bytes(
    data: bytes,
    /,
    *,
    limits: ResourceLimits,
    source_path: str | None = None,
) -> BoundedResource:
    """Bound an already resident byte resource and record its exact identity."""

    if not isinstance(data, bytes):
        raise TypeError("Bounded resource data must be bytes.")
    if len(data) > limits.max_bytes:
        raise ResourceReadError(
            "limit",
            f"Resource exceeds the configured {limits.max_bytes}-byte size limit.",
        )
    manifest = _manifest(
        source_kind="memory",
        source_path=None if source_path is None else str(source_path),
        trusted_root=None,
        relative_components=(),
        data=data,
        file_status=None,
        root_status=None,
        limits=limits,
        observed_depth=0,
        observed_nodes=0,
        observed_attributes=0,
        observed_losses=0,
    )
    return BoundedResource(data, manifest)


def read_bounded_resource(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    limits: ResourceLimits,
) -> BoundedResource:
    """Read one regular file by walking beneath a trusted directory descriptor."""

    root_text, components = _resource_components(path, trusted_root, limits.max_depth)
    descriptors: list[int] = []
    directory_states: list[tuple[int, os.stat_result]] = []
    try:
        root_descriptor = os.open(
            root_text,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
        descriptors.append(root_descriptor)
        root_status = os.fstat(root_descriptor)
        if not stat.S_ISDIR(root_status.st_mode):
            raise ResourceReadError(
                "policy", "The trusted resource root must be a real directory."
            )
        directory_states.append((root_descriptor, root_status))
        parent_descriptor = root_descriptor
        for component in components[:-1]:
            descriptor = os.open(
                component,
                os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                dir_fd=parent_descriptor,
            )
            descriptors.append(descriptor)
            status = os.fstat(descriptor)
            if not stat.S_ISDIR(status.st_mode):
                raise ResourceReadError(
                    "policy", "Resource path components must be real directories."
                )
            directory_states.append((descriptor, status))
            parent_descriptor = descriptor
        file_descriptor = os.open(
            components[-1],
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_descriptor,
        )
        descriptors.append(file_descriptor)
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ResourceReadError(
                "policy", "The requested resource must be a regular file."
            )
        if before.st_size > limits.max_bytes:
            raise ResourceReadError(
                "limit",
                f"Resource exceeds the configured {limits.max_bytes}-byte size limit.",
            )
        payload = bytearray()
        while len(payload) <= limits.max_bytes:
            chunk = os.read(
                file_descriptor,
                min(64 * 1024, limits.max_bytes + 1 - len(payload)),
            )
            if not chunk:
                break
            payload.extend(chunk)
        if len(payload) > limits.max_bytes:
            raise ResourceReadError(
                "limit",
                f"Resource exceeds the configured {limits.max_bytes}-byte size limit.",
            )
        after = os.fstat(file_descriptor)
        if (
            len(payload) != after.st_size
            or _stat_identity(before) != _stat_identity(after)
        ):
            raise ResourceReadError(
                "inconsistent", "The resource changed while it was being read."
            )
        for descriptor, initial in directory_states:
            if _directory_identity(initial) != _directory_identity(
                os.fstat(descriptor)
            ):
                raise ResourceReadError(
                    "inconsistent", "A resource path component changed during the read."
                )
        data = bytes(payload)
        source_path = os.path.join(root_text, *components)
        manifest = _manifest(
            source_kind="file",
            source_path=source_path,
            trusted_root=root_text,
            relative_components=components,
            data=data,
            file_status=after,
            root_status=root_status,
            limits=limits,
            observed_depth=0,
            observed_nodes=0,
            observed_attributes=0,
            observed_losses=0,
        )
        return BoundedResource(data, manifest)
    except ResourceReadError:
        raise
    except OSError as error:
        reason: _ResourceFailure = (
            "policy"
            if error.errno in (errno.ELOOP, errno.ENOTDIR)
            else "malformed"
        )
        raise ResourceReadError(
            reason, "The requested resource could not be opened or read."
        ) from error
    finally:
        _close_descriptors(descriptors)


def account_bounded_resource(
    resource: BoundedResource,
    /,
    *,
    depth: int,
    nodes: int,
    attributes: int,
    losses: int,
) -> BoundedResource:
    """Record bounded structured-decoding counts without mutating provenance."""

    counts = int(depth), int(nodes), int(attributes), int(losses)
    if any(value < 0 for value in counts):
        raise ValueError("Observed resource counts must be nonnegative.")
    limits = resource.manifest.limits
    if counts[0] > limits.max_depth:
        raise ResourceReadError("limit", "Resource nesting exceeds its depth limit.")
    if counts[1] > limits.max_nodes:
        raise ResourceReadError("limit", "Resource node count exceeds its limit.")
    if counts[2] > limits.max_attributes:
        raise ResourceReadError("limit", "Resource attribute count exceeds its limit.")
    if counts[3] > limits.max_losses:
        raise ResourceReadError(
            "limit", "Resource semantic-loss count exceeds its limit."
        )
    previous = resource.manifest
    manifest = _manifest(
        source_kind=previous.source_kind,
        source_path=previous.source_path,
        trusted_root=previous.trusted_root,
        relative_components=previous.relative_components,
        data=resource.data,
        file_status=None,
        root_status=None,
        limits=limits,
        observed_depth=counts[0],
        observed_nodes=counts[1],
        observed_attributes=counts[2],
        observed_losses=counts[3],
        file_device=previous.file_device,
        file_inode=previous.file_inode,
        file_mode=previous.file_mode,
        root_device=previous.trusted_root_device,
        root_inode=previous.trusted_root_inode,
        root_mode=previous.trusted_root_mode,
    )
    return replace(resource, manifest=manifest)


def _resource_components(
    path: str | os.PathLike[str],
    trusted_root: str | os.PathLike[str],
    maximum_depth: int,
    /,
) -> tuple[str, tuple[str, ...]]:
    path_text = os.fspath(path)
    root_input = os.fspath(trusted_root)
    if not isinstance(path_text, str) or not isinstance(root_input, str):
        raise TypeError("Resource paths and trusted roots must be text paths.")
    if _remote_location(path_text) or _remote_location(root_input):
        raise ResourceReadError("policy", "Network resource locations are disabled.")
    if "\x00" in path_text or "\x00" in root_input:
        raise ResourceReadError("policy", "Resource paths cannot contain null bytes.")
    root_text = os.path.abspath(os.path.expanduser(root_input))
    expanded_path = os.path.expanduser(path_text)
    raw_parts = tuple(part for part in expanded_path.split(os.sep) if part)
    if ".." in raw_parts:
        raise ResourceReadError("policy", "Resource traversal components are disabled.")
    if os.path.isabs(expanded_path):
        relative_text = os.path.relpath(expanded_path, root_text)
    else:
        relative_text = expanded_path
    components = tuple(
        part for part in relative_text.split(os.sep) if part and part != "."
    )
    if not components:
        raise ResourceReadError("policy", "A resource file path is required.")
    if any(part == ".." for part in components):
        raise ResourceReadError("policy", "The resource path escapes its trusted root.")
    if len(components) > maximum_depth:
        raise ResourceReadError(
            "limit", "Resource path nesting exceeds its depth limit."
        )
    return root_text, components


def _remote_location(value: str, /) -> bool:
    return "://" in value or value.startswith(("//", "\\\\"))


def _stat_identity(status: os.stat_result, /) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _directory_identity(status: os.stat_result, /) -> tuple[int, ...]:
    return (
        int(status.st_dev),
        int(status.st_ino),
        int(status.st_mode),
        int(status.st_size),
        int(status.st_mtime_ns),
        int(status.st_ctime_ns),
    )


def _manifest(
    *,
    source_kind: Literal["memory", "file"],
    source_path: str | None,
    trusted_root: str | None,
    relative_components: tuple[str, ...],
    data: bytes,
    file_status: os.stat_result | None,
    root_status: os.stat_result | None,
    limits: ResourceLimits,
    observed_depth: int,
    observed_nodes: int,
    observed_attributes: int,
    observed_losses: int,
    file_device: int | None = None,
    file_inode: int | None = None,
    file_mode: int | None = None,
    root_device: int | None = None,
    root_inode: int | None = None,
    root_mode: int | None = None,
) -> ResourceManifest:
    if file_status is not None:
        file_device = int(file_status.st_dev)
        file_inode = int(file_status.st_ino)
        file_mode = int(file_status.st_mode)
    if root_status is not None:
        root_device = int(root_status.st_dev)
        root_inode = int(root_status.st_ino)
        root_mode = int(root_status.st_mode)
    digest = hashlib.sha256(data).hexdigest()
    payload = {
        "kind": "bounded-resource-manifest",
        "source_kind": source_kind,
        "source_path": source_path,
        "trusted_root": trusted_root,
        "trusted_root_device": root_device,
        "trusted_root_inode": root_inode,
        "trusted_root_mode": root_mode,
        "relative_components": list(relative_components),
        "size_bytes": len(data),
        "content_sha256": digest,
        "file_device": file_device,
        "file_inode": file_inode,
        "file_mode": file_mode,
        "limits": {
            "max_bytes": limits.max_bytes,
            "max_depth": limits.max_depth,
            "max_nodes": limits.max_nodes,
            "max_attributes": limits.max_attributes,
            "max_losses": limits.max_losses,
        },
        "observed": {
            "depth": observed_depth,
            "nodes": observed_nodes,
            "attributes": observed_attributes,
            "losses": observed_losses,
        },
    }
    return ResourceManifest(
        source_kind,
        source_path,
        trusted_root,
        root_device,
        root_inode,
        root_mode,
        relative_components,
        len(data),
        digest,
        file_device,
        file_inode,
        file_mode,
        limits,
        observed_depth,
        observed_nodes,
        observed_attributes,
        observed_losses,
        canonical_fingerprint(payload),
    )


def _close_descriptors(descriptors: list[int], /) -> None:
    for descriptor in reversed(descriptors):
        try:
            os.close(descriptor)
        except OSError:
            pass


__all__ = [
    "BoundedResource",
    "ResourceLimits",
    "ResourceManifest",
    "ResourceReadError",
    "account_bounded_resource",
    "bounded_resource_from_bytes",
    "read_bounded_resource",
]
