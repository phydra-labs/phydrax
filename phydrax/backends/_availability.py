#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from types import ModuleType

from ._types import (
    BackendAvailability,
    BackendCapabilities,
    BackendUnavailableError,
)


def distribution_versions(
    distributions: tuple[str, ...],
    /,
) -> tuple[tuple[str, str], ...]:
    """Return installed distribution versions without importing provider modules."""
    versions: list[tuple[str, str]] = []
    for distribution in distributions:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        versions.append((distribution, version))
    return tuple(versions)


def probe_backend(
    capabilities: BackendCapabilities,
    /,
    *,
    module: str,
    requirement: str,
    distributions: tuple[str, ...] = (),
    supported_platforms: tuple[str, ...] | None = None,
) -> BackendAvailability:
    """Probe one optional module and preserve missing/import/linker evidence."""
    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError("capabilities must be BackendCapabilities.")
    module_ = str(module)
    requirement_ = str(requirement)
    if not module_ or not requirement_:
        raise ValueError("Backend module and requirement must be non-empty.")
    versions = distribution_versions(distributions)
    if supported_platforms is not None and sys.platform not in supported_platforms:
        supported = ", ".join(supported_platforms)
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"platform {sys.platform!r} is unsupported; expected one of {supported}",
            versions=versions,
        )
    try:
        specification = importlib.util.find_spec(module_)
    except (ImportError, ModuleNotFoundError) as error:
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"module discovery failed: {type(error).__name__}: {error}",
            versions=versions,
        )
    if specification is None:
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"required module {module_!r} is not installed",
            versions=versions,
        )
    try:
        importlib.import_module(module_)
    except ModuleNotFoundError as error:
        missing = error.name or "an undeclared transitive module"
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"provider import is missing transitive module {missing!r}",
            versions=versions,
        )
    except ImportError as error:
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"provider import failed: ImportError: {error}",
            versions=versions,
        )
    except OSError as error:
        return BackendAvailability(
            capabilities=capabilities,
            available=False,
            requirement=requirement_,
            reason=f"provider linker/runtime load failed: OSError: {error}",
            versions=versions,
        )
    return BackendAvailability(
        capabilities=capabilities,
        available=True,
        requirement=requirement_,
        reason="provider module imported successfully",
        versions=versions,
    )


def import_backend_module(
    availability: BackendAvailability,
    capability: str,
    module: str,
    /,
) -> ModuleType:
    """Import an already-probed backend or raise its exact capability failure."""
    if not isinstance(availability, BackendAvailability):
        raise TypeError("availability must be BackendAvailability.")
    availability.require(capability)
    module_ = str(module)
    try:
        return importlib.import_module(module_)
    except (ModuleNotFoundError, ImportError, OSError) as error:
        raise BackendUnavailableError(
            availability.backend,
            str(capability),
            availability.requirement,
            f"provider became unavailable after probing: {type(error).__name__}: {error}",
        ) from error


__all__ = ["distribution_versions", "import_backend_module", "probe_backend"]
