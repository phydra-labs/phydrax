# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Geophysical sampled products inside the existing native lifecycle envelope."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..._array_archive import ArrayArchiveLimits, DEFAULT_ARRAY_ARCHIVE_LIMITS
from ...lifecycle import (
    create,
    LifecycleArchive,
    open as open_lifecycle,
    payload_digest,
    ResultManifest,
    ResultRevision,
)
from ._data import GeophysicalData


_SEMANTICS_KEY = "geophysical_data"


def write_geophysical_archive(
    path: str | Path,
    data: GeophysicalData,
    /,
    *,
    run_id: str,
    evidence_ids: tuple[str, ...] = (),
    diagnostic_ids: tuple[str, ...] = (),
) -> LifecycleArchive:
    """Persist exact native product values, masks, descriptors, and provenance.

    An imported initialization remains initialization. This is not a checkpoint:
    solver history, random keys, caches, and distributed restart admission remain
    the responsibility of the existing simulation/lifecycle owners.
    """
    descriptor = data.descriptor
    fields = tuple(
        (
            name,
            descriptor["variables"][name]["payload"],
            descriptor["variables"][name]["attrs"]["units"],
        )
        for name in data.fields
    )
    manifest = ResultManifest(
        data.data_id,
        run_id,
        fields,
        {name: payload_digest(value) for name, value in data.arrays.items()},
        evidence_ids=evidence_ids,
        diagnostic_ids=diagnostic_ids,
        sampled_semantics={_SEMANTICS_KEY: data.descriptor_json},
    )
    return create(path, manifest=manifest, arrays=data.arrays)


def read_geophysical_archive(
    path: str | Path,
    /,
    *,
    bindings: Mapping[str, Any] | None = None,
    limits: ArrayArchiveLimits | None = DEFAULT_ARRAY_ARCHIVE_LIMITS,
) -> GeophysicalData:
    """Verify the native lifecycle archive and all canonical scientific references.

    No external host I/O dependency is needed. Optional supplied bindings must
    match exactly; no unresolved storage identity is reconstructed as geometry.
    """
    archive = open_lifecycle(path, limits=limits)
    manifest = archive.manifest
    if isinstance(manifest, ResultRevision):
        manifest = manifest.manifest
    if not isinstance(manifest, ResultManifest):
        raise ValueError("Geophysical products require a native result manifest.")
    semantics = dict(manifest.sampled_semantics)
    if _SEMANTICS_KEY not in semantics:
        raise ValueError("Result manifest has no geophysical sampled semantics.")
    descriptor = json.loads(semantics[_SEMANTICS_KEY])
    data = GeophysicalData(descriptor, archive.arrays)
    if data.data_id != manifest.result_id:
        raise ValueError("Geophysical data identity does not match the result manifest.")
    expected_fields = tuple(
        sorted(
            (
                name,
                descriptor["variables"][name]["payload"],
                descriptor["variables"][name]["attrs"]["units"],
            )
            for name in data.fields
        )
    )
    if tuple(sorted(manifest.fields)) != expected_fields:
        raise ValueError("Canonical geophysical result field references do not match.")
    if bindings is not None:
        data.require_bindings(bindings)
    return data


__all__ = ["write_geophysical_archive", "read_geophysical_archive"]
