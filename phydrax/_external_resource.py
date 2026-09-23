#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Descriptor-relative, bounded reads of untrusted local resources."""

from __future__ import annotations

import errno
import hashlib
import os
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import BinaryIO, Literal

from ._fingerprint import canonical_fingerprint
from ._host_io import open_regular_beneath, OpenedHostFile


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


@dataclass(frozen=True, slots=True)
class OpenedResource:
    """One admitted seekable resource valid for the surrounding context."""

    stream: BinaryIO
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

    try:
        with open_regular_beneath(
            path,
            trusted_root=trusted_root,
            maximum_depth=limits.max_depth,
        ) as opened:
            data, manifest = _read_opened_resource(opened, limits=limits, retain=True)
    except ResourceReadError:
        raise
    except OverflowError as error:
        raise ResourceReadError("limit", str(error)) from error
    except ValueError as error:
        raise ResourceReadError("policy", str(error)) from error
    except RuntimeError as error:
        raise ResourceReadError("inconsistent", str(error)) from error
    except OSError as error:
        reason: _ResourceFailure = (
            "policy" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "malformed"
        )
        raise ResourceReadError(
            reason, "The requested resource could not be opened or read."
        ) from error
    if data is None:
        raise RuntimeError("Resident resource admission did not retain bytes.")
    return BoundedResource(data, manifest)


@contextmanager
def open_bounded_resource(
    path: str | os.PathLike[str],
    /,
    *,
    trusted_root: str | os.PathLike[str],
    limits: ResourceLimits,
) -> Iterator[OpenedResource]:
    """Open one admitted seekable resource without retaining its bytes in memory."""

    consumer_boundary = False
    try:
        with open_regular_beneath(
            path,
            trusted_root=trusted_root,
            maximum_depth=limits.max_depth,
        ) as opened:
            _, manifest = _read_opened_resource(opened, limits=limits, retain=False)
            with opened.duplicate_stream() as stream:
                stream.seek(0)
                consumer_boundary = True
                yield OpenedResource(stream, manifest)
                consumer_boundary = False
            opened.verify_stable()
    except ResourceReadError:
        raise
    except OverflowError as error:
        if consumer_boundary:
            raise
        raise ResourceReadError("limit", str(error)) from error
    except ValueError as error:
        if consumer_boundary:
            raise
        raise ResourceReadError("policy", str(error)) from error
    except RuntimeError as error:
        if consumer_boundary:
            raise
        raise ResourceReadError("inconsistent", str(error)) from error
    except OSError as error:
        if consumer_boundary:
            raise
        reason: _ResourceFailure = (
            "policy" if error.errno in (errno.ELOOP, errno.ENOTDIR) else "malformed"
        )
        raise ResourceReadError(
            reason, "The requested resource could not be opened or read."
        ) from error


def _read_opened_resource(
    opened: OpenedHostFile,
    /,
    *,
    limits: ResourceLimits,
    retain: bool,
) -> tuple[bytes | None, ResourceManifest]:
    before = opened.file_status
    if before.st_size > limits.max_bytes:
        raise ResourceReadError(
            "limit",
            f"Resource exceeds the configured {limits.max_bytes}-byte size limit.",
        )
    opened.rewind()
    digest = hashlib.sha256()
    payload = bytearray() if retain else None
    total = 0
    while total <= limits.max_bytes:
        chunk = os.read(
            opened.descriptor,
            min(64 * 1024, limits.max_bytes + 1 - total),
        )
        if not chunk:
            break
        total += len(chunk)
        digest.update(chunk)
        if payload is not None:
            payload.extend(chunk)
    if total > limits.max_bytes:
        raise ResourceReadError(
            "limit",
            f"Resource exceeds the configured {limits.max_bytes}-byte size limit.",
        )
    after = opened.verify_stable()
    if total != after.st_size:
        raise ResourceReadError(
            "inconsistent", "The resource changed while it was being read."
        )
    manifest = _manifest(
        source_kind="file",
        source_path=os.path.join(opened.root_path, *opened.components),
        trusted_root=opened.root_path,
        relative_components=opened.components,
        data=None,
        size_bytes=total,
        content_sha256=digest.hexdigest(),
        file_status=after,
        root_status=opened.root_status,
        limits=limits,
        observed_depth=0,
        observed_nodes=0,
        observed_attributes=0,
        observed_losses=0,
    )
    return None if payload is None else bytes(payload), manifest


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


def _manifest(
    *,
    source_kind: Literal["memory", "file"],
    source_path: str | None,
    trusted_root: str | None,
    relative_components: tuple[str, ...],
    data: bytes | None,
    file_status: os.stat_result | None,
    root_status: os.stat_result | None,
    limits: ResourceLimits,
    observed_depth: int,
    observed_nodes: int,
    observed_attributes: int,
    observed_losses: int,
    size_bytes: int | None = None,
    content_sha256: str | None = None,
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
    if data is not None:
        size_bytes = len(data)
        content_sha256 = hashlib.sha256(data).hexdigest()
    if (
        size_bytes is None
        or size_bytes < 0
        or content_sha256 is None
        or len(content_sha256) != 64
        or any(character not in "0123456789abcdef" for character in content_sha256)
    ):
        raise ValueError("Resource size and SHA-256 identity are invalid.")
    payload = {
        "kind": "bounded-resource-manifest",
        "source_kind": source_kind,
        "source_path": source_path,
        "trusted_root": trusted_root,
        "trusted_root_device": root_device,
        "trusted_root_inode": root_inode,
        "trusted_root_mode": root_mode,
        "relative_components": list(relative_components),
        "size_bytes": size_bytes,
        "content_sha256": content_sha256,
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
        size_bytes,
        content_sha256,
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


__all__ = [
    "BoundedResource",
    "OpenedResource",
    "ResourceLimits",
    "ResourceManifest",
    "ResourceReadError",
    "account_bounded_resource",
    "bounded_resource_from_bytes",
    "open_bounded_resource",
    "read_bounded_resource",
]
