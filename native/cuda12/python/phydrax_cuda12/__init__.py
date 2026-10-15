#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Locations of the packaged Phydrax CUDA 12 component libraries."""

from __future__ import annotations

from pathlib import Path


__all__ = ["component_library"]


def component_library(name: str) -> Path:
    """Return an installed component's absolute path without loading native code.

    Raises:
        ValueError: If ``name`` is not ``cudss``, ``amgx``, or ``quda``.
        FileNotFoundError: If the selected component library is not packaged.
    """
    match name:
        case "cudss" | "amgx" | "quda":
            path = Path(__file__).resolve().parent / "lib" / f"libphydrax_cuda_{name}.so"
        case _:
            raise ValueError(
                f"Unknown CUDA component {name!r}; expected cudss, amgx, or quda."
            )
    if not path.is_file():
        raise FileNotFoundError(f"phydrax_cuda12 {name} library is missing at {path}.")
    return path
