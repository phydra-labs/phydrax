#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Locations of the packaged Phydrax CUDA 13 component libraries."""

from __future__ import annotations

import json
from pathlib import Path


__all__ = ["component_library"]


def component_library(name: str) -> Path:
    """Return an installed component's absolute path without loading native code.

    Raises:
        ValueError: If ``name`` is unknown or not shipped in this platform/build.
        RuntimeError: If the installed manifest is malformed.
        FileNotFoundError: If a declared component or its manifest is missing.
    """
    match name:
        case (
            "cudss"
            | "amgx"
            | "quda"
            | "cutensor"
            | "cutensornet"
            | "cuest"
            | "collective"
            | "cudecomp"
            | "cudss_mgmn"
        ):
            pass
        case _:
            raise ValueError(
                f"Unknown CUDA component {name!r}; expected cudss, amgx, quda, "
                "cutensor, cutensornet, cuest, collective, cudecomp, or cudss_mgmn."
            )
    root = Path(__file__).resolve().parent
    manifest = root / "components.json"
    try:
        components = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid CUDA component manifest: {manifest}") from error
    if not isinstance(components, dict):
        raise RuntimeError(f"Invalid CUDA component manifest: {manifest}")
    if name not in components:
        raise ValueError(
            f"CUDA component {name!r} is unavailable in this installed build."
        )
    relative = components[name]
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or Path(relative).parts[:1] != ("lib",)
    ):
        raise RuntimeError(f"Invalid CUDA component path in {manifest}: {relative!r}")
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"phydrax_cuda13 {name} library is missing at {path}.")
    return path
