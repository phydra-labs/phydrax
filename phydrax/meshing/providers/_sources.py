#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Location of the native provider worker sources distributed with phydrax."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, TypeAlias

from ...typing import checked


NativeProviderName: TypeAlias = Literal["mmg", "omega_h", "tioga", "vorocrust"]

_PACKAGE = Path(__file__).resolve().parents[2]


@checked
def native_provider_source_root(package: Path, /) -> Path | None:
    """Return the native provider source tree of the installation containing ``package``.

    ``package`` is a ``phydrax`` package directory. A wheel carries the tree
    inside it (``<package>/native/providers``); a source checkout or sdist keeps
    it at the project root beside it (``<package>/../native/providers``). Exactly
    one layout must be present, and source/build identity reads provider bytes
    from the same tree that ``native_provider_source_path`` uses.

    Returns ``None`` when the installation carries neither tree. Raises
    ``NotADirectoryError`` when either location holds something other than a
    directory, and ``RuntimeError`` when both trees are present rather than
    resolving them provider by provider.
    """
    installed = package / "native" / "providers"
    checkout = package.parent / "native" / "providers"
    present = [path for path in (installed, checkout) if path.exists()]
    for path in present:
        if not path.is_dir():
            raise NotADirectoryError(
                "native_provider_source_root.tree must be a provider source "
                f"directory; got a non-directory at {path}"
            )
    if len(present) > 1:
        raise RuntimeError(
            "native_provider_source_root.layout must be exactly one installation "
            f"layout; got both {installed} and {checkout}"
        )
    return present[0] if present else None


def native_provider_source_path(provider: NativeProviderName) -> Path:
    """Return the CMake source directory of one native provider worker.

    The directory is a standalone CMake project (``cmake -S <path> ...``); its
    sibling ``common`` directory holds the shared worker protocol headers.
    Raises ``FileNotFoundError`` when this installation lacks the provider's
    worker sources or shared headers, and the ``native_provider_source_root`` errors
    when its provider tree is malformed or present in both layouts.
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
    root = native_provider_source_root(_PACKAGE)
    if root is None:
        raise FileNotFoundError(
            f"native_provider_source_path.provider must name worker sources in this "
            f"phydrax installation; got no provider tree under "
            f"{_PACKAGE / 'native' / 'providers'} or "
            f"{_PACKAGE.parent / 'native' / 'providers'} for {provider!r}"
        )
    source = root / provider
    if not (source / "CMakeLists.txt").is_file() or not (root / "common").is_dir():
        raise FileNotFoundError(
            f"native_provider_source_path.provider must name worker sources in this "
            f"phydrax installation; got no {provider!r} CMake project with shared "
            f"common headers under {root}"
        )
    return source


__all__ = ["native_provider_source_path", "native_provider_source_root"]
