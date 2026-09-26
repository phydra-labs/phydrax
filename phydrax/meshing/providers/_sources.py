#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Location of the native provider worker sources distributed with phydrax."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias


NativeProviderName: TypeAlias = Literal["mmg", "omega_h", "tioga", "vorocrust"]

# A wheel carries the sources inside the package (Hatch force-include); a source
# checkout or sdist keeps them at the project root beside the package.
_PACKAGE = Path(__file__).resolve().parents[2]
_ROOTS = (
    _PACKAGE / "native" / "providers",
    _PACKAGE.parent / "native" / "providers",
)


def native_provider_source_path(provider: NativeProviderName) -> Path:
    """Return the CMake source directory of one native provider worker.

    The directory is a standalone CMake project (``cmake -S <path> ...``); its
    sibling ``common`` directory holds the shared worker protocol headers.
    Raises ``FileNotFoundError`` when this installation carries no sources.
    """
    if not isinstance(provider, str):
        raise TypeError("provider must be a native provider name.")
    match provider:
        case "mmg" | "omega_h" | "tioga" | "vorocrust":
            pass
        case _:
            raise ValueError(
                f"Unknown native provider {provider!r}; expected one of "
                "'mmg', 'omega_h', 'tioga', or 'vorocrust'."
            )
    for root in _ROOTS:
        source = root / provider
        if (source / "CMakeLists.txt").is_file() and (root / "common").is_dir():
            return source
    raise FileNotFoundError(
        f"The {provider} worker sources are not part of this phydrax installation "
        f"(searched {', '.join(str(root) for root in _ROOTS)})."
    )


__all__ = ["native_provider_source_path"]
