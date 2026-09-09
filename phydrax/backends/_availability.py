#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from types import ModuleType

from ..logging import emit
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

def _probe_result(
    capabilities: BackendCapabilities,
    /,
    *,
    available: bool,
    failure_category: str | None,
    reason: str,
    requirement: str,
    versions: tuple[tuple[str, str], ...],
) -> BackendAvailability:
    result = BackendAvailability(
        capabilities=capabilities,
        available=available,
        requirement=requirement,
        reason=reason,
        versions=versions,
    )
    emit(
        "DEBUG",
        "backend.probe.completed",
        "Backend probe completed",
        available=available,
        backend=capabilities.backend,
        failure_category=failure_category,
        versions=tuple(
            {"distribution": distribution, "version": version}
            for distribution, version in versions
        ),
    )
    return result


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
        return _probe_result(
            capabilities,
            available=False,
            failure_category="unsupported_platform",
            requirement=requirement_,
            reason=f"platform {sys.platform!r} is unsupported; expected one of {supported}",
            versions=versions,
        )
    try:
        specification = importlib.util.find_spec(module_)
    except (ImportError, ModuleNotFoundError) as error:
        return _probe_result(
            capabilities,
            available=False,
            failure_category="discovery_failed",
            requirement=requirement_,
            reason=f"module discovery failed: {type(error).__name__}: {error}",
            versions=versions,
        )
    if specification is None:
        return _probe_result(
            capabilities,
            available=False,
            failure_category="not_installed",
            requirement=requirement_,
            reason=f"required module {module_!r} is not installed",
            versions=versions,
        )
    try:
        importlib.import_module(module_)
    except ModuleNotFoundError as error:
        missing = error.name or "an undeclared transitive module"
        return _probe_result(
            capabilities,
            available=False,
            failure_category="missing_transitive_dependency",
            requirement=requirement_,
            reason=f"provider import is missing transitive module {missing!r}",
            versions=versions,
        )
    except ImportError as error:
        return _probe_result(
            capabilities,
            available=False,
            failure_category="import_failed",
            requirement=requirement_,
            reason=f"provider import failed: ImportError: {error}",
            versions=versions,
        )
    except OSError as error:
        return _probe_result(
            capabilities,
            available=False,
            failure_category="runtime_load_failed",
            requirement=requirement_,
            reason=f"provider linker/runtime load failed: OSError: {error}",
            versions=versions,
        )
    return _probe_result(
        capabilities,
        available=True,
        failure_category=None,
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
        imported = importlib.import_module(module_)
    except (ModuleNotFoundError, ImportError, OSError) as error:
        emit(
            "WARNING",
            "backend.import.failed",
            "Backend import failed",
            backend=availability.backend,
            capability=str(capability),
            failure_category=type(error).__name__,
        )
        raise BackendUnavailableError(
            availability.backend,
            str(capability),
            availability.requirement,
            f"provider became unavailable after probing: {type(error).__name__}: {error}",
        ) from error
    emit(
        "DEBUG",
        "backend.import.completed",
        "Backend import completed",
        backend=availability.backend,
        capability=str(capability),
    )
    return imported


__all__ = ["distribution_versions", "import_backend_module", "probe_backend"]
