#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np

from .._array_archive import read_array_archive, write_array_archive


OPEN_SYSTEM_ARTIFACT_SCHEMA = 1
_REQUIRED_FIELDS = {
    "schema_version",
    "campaign_id",
    "representation_id",
    "problem_id",
    "plan_id",
    "precision",
    "backend",
    "status",
    "thresholds",
    "approximation_axes",
    "semantic_rng_schema",
}


def write_open_system_artifact(
    path: str | os.PathLike[str],
    /,
    *,
    campaign_id: str,
    representation_id: str,
    problem_id: str,
    plan_id: str,
    precision: str,
    backend: str,
    status: str,
    thresholds: dict[str, float],
    approximation_axes: dict[str, float | int],
    semantic_rng_schema: dict[str, Any],
    arrays: dict[str, Any],
    extra_manifest: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "schema_version": OPEN_SYSTEM_ARTIFACT_SCHEMA,
        "campaign_id": str(campaign_id),
        "representation_id": str(representation_id),
        "problem_id": str(problem_id),
        "plan_id": str(plan_id),
        "precision": str(precision),
        "backend": str(backend),
        "status": str(status),
        "thresholds": dict(thresholds),
        "approximation_axes": dict(approximation_axes),
        "semantic_rng_schema": dict(semantic_rng_schema),
    }
    if extra_manifest is not None:
        overlap = set(extra_manifest).intersection(manifest)
        if overlap:
            raise ValueError(
                f"Extra manifest overwrites reserved fields: {sorted(overlap)}"
            )
        manifest.update(extra_manifest)
    return write_array_archive(path, manifest=manifest, arrays=arrays)


def read_open_system_artifact(
    path: str | os.PathLike[str],
    /,
    *,
    expected_campaign_id: str | None = None,
    expected_representation_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest, arrays = read_array_archive(path)
    missing = _REQUIRED_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"Open-system artifact is missing fields: {sorted(missing)}")
    if manifest["schema_version"] != OPEN_SYSTEM_ARTIFACT_SCHEMA:
        raise ValueError("Unsupported open-system artifact schema version.")
    if (
        expected_campaign_id is not None
        and manifest["campaign_id"] != expected_campaign_id
    ):
        raise ValueError("Open-system campaign identity mismatch.")
    if (
        expected_representation_id is not None
        and manifest["representation_id"] != expected_representation_id
    ):
        raise ValueError("Open-system representation identity mismatch.")
    for name, value in arrays.items():
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Open-system artifact array {name!r} is nonfinite.")
    return manifest, arrays


__all__ = [
    "OPEN_SYSTEM_ARTIFACT_SCHEMA",
    "read_open_system_artifact",
    "write_open_system_artifact",
]
