#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.metadata
from collections.abc import Sequence
from types import ModuleType

from .base import Availability


def unsupported(
    *,
    adapter: str,
    dependency: str,
    capability: str,
) -> Availability:
    return Availability(
        available=False,
        capability=capability,
        dependency=dependency,
        dependency_version=None,
        reason=f"adapter {adapter!r} does not implement capability {capability!r}",
    )


def probe_modules(
    *,
    adapter: str,
    dependency: str,
    capability: str,
    supported: frozenset[str],
    modules: Sequence[str],
    distribution: str,
) -> Availability:
    if capability not in supported:
        return unsupported(
            adapter=adapter,
            dependency=dependency,
            capability=capability,
        )
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing = error.name or module_name
            return Availability(
                available=False,
                capability=capability,
                dependency=dependency,
                dependency_version=None,
                reason=f"required module {missing!r} is not installed for adapter {adapter!r}",
            )
        except ImportError as error:
            return Availability(
                available=False,
                capability=capability,
                dependency=dependency,
                dependency_version=None,
                reason=(
                    f"required module {module_name!r} could not be imported for adapter "
                    f"{adapter!r}: {type(error).__name__}: {error}"
                ),
            )
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        version = "installed; distribution metadata unavailable"
    return Availability(
        available=True,
        capability=capability,
        dependency=dependency,
        dependency_version=version,
        reason=None,
    )


def import_module(name: str, /) -> ModuleType:
    """Import an optional module only after its availability row has been established."""
    return importlib.import_module(name)


__all__ = ["import_module", "probe_modules", "unsupported"]
