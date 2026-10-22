#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Locations of the packaged Phydrax CUDA 13 component libraries."""

from __future__ import annotations

from pathlib import Path


__all__ = ["component_library"]


def component_library(name: str) -> Path:
    """Return an installed component's absolute path without loading native code.

    Raises:
        ValueError: If ``name`` is not a supported native CUDA component.
        FileNotFoundError: If the selected component library is not packaged.
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
            path = Path(__file__).resolve().parent / "lib" / f"libphydrax_cuda_{name}.so"
        case _:
            raise ValueError(
                f"Unknown CUDA component {name!r}; expected cudss, amgx, quda, "
                "cutensor, cutensornet, cuest, collective, cudecomp, or cudss_mgmn."
            )
    if not path.is_file():
        raise FileNotFoundError(f"phydrax_cuda13 {name} library is missing at {path}.")
    return path
