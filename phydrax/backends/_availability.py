#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from collections.abc import Callable
from importlib.machinery import EXTENSION_SUFFIXES, ModuleSpec, PathFinder
from pathlib import Path
from types import ModuleType

import jax

from .._cuda_environment import (
    assess_cuda_environment,
    assess_provider_dependencies,
    collect_cuda_environment,
    cuda_enabled,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    provider_dependency_closure,
)
from .._execution_device import device_platform as runtime_platform
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
    environment_issues: tuple[CudaEnvironmentIssue, ...] = (),
) -> BackendAvailability:
    result = BackendAvailability(
        capabilities=capabilities,
        available=available,
        requirement=requirement,
        reason=reason,
        versions=versions,
        environment_issues=environment_issues,
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


def probe_cuda_component(
    capabilities: BackendCapabilities,
    /,
    *,
    component: str,
    requirement: str,
) -> BackendAvailability:
    """Probe a Phydrax-owned CUDA component and preserve refusal evidence."""
    from .._cuda_native import cuda_component_status, CudaNativeComponentName
    from ..typing import parse

    if not isinstance(capabilities, BackendCapabilities):
        raise TypeError("capabilities must be BackendCapabilities.")
    if not requirement:
        raise ValueError("Backend requirement must be non-empty.")
    name = parse(component, CudaNativeComponentName, "component")
    available, reason, issues, versions = cuda_component_status(name)
    return _probe_result(
        capabilities,
        available=available,
        failure_category=None if available else "cuda_companion_refused",
        reason=reason,
        requirement=requirement,
        versions=versions,
        environment_issues=issues,
    )


def _native_module_specification(module: str, /) -> ModuleSpec | None:
    # util.find_spec on a dotted name imports its parent before any CUDA guard.
    parts = module.split(".")
    specification = importlib.util.find_spec(parts[0])
    for index in range(1, len(parts)):
        if specification is None or specification.submodule_search_locations is None:
            return None
        specification = PathFinder.find_spec(
            ".".join(parts[: index + 1]), specification.submodule_search_locations
        )
    return specification


def module_native_libraries(
    module: str, /, pattern: str | None = None
) -> tuple[str, ...]:
    """Locate native provider objects without importing the provider package."""
    specification = _native_module_specification(module)
    if specification is None:
        return ()
    patterns = (
        (pattern,)
        if pattern is not None
        else ("*.pyd", "*.dll")
        if sys.platform == "win32"
        else ("*.so",)
    )
    if specification.submodule_search_locations is not None:
        return tuple(
            sorted(
                str(path)
                for directory in specification.submodule_search_locations
                for selected in patterns
                for path in Path(directory).glob(selected)
                if path.is_file()
            )
        )
    origin = specification.origin
    if origin is not None and origin.endswith(tuple(EXTENSION_SUFFIXES)):
        return (origin,)
    return ()


def _cuda_dependency_issues(
    cuda_libraries: Callable[[], tuple[str, ...]],
    facts: CudaEnvironmentFacts,
    /,
    *,
    jax_backend_preloaded: bool,
) -> tuple[CudaEnvironmentIssue, ...]:
    """Judge each located native library as a fresh process would load it."""
    try:
        return tuple(
            issue
            for library in cuda_libraries()
            for issue in assess_provider_dependencies(
                provider_dependency_closure(
                    library,
                    facts=facts,
                    jax_backend_preloaded=jax_backend_preloaded,
                ),
                facts,
            )
        )
    except (OSError, ValueError) as error:
        return (
            CudaEnvironmentIssue(
                code="provider-dependency-unresolved",
                severity="error",
                message=f"Provider dependency inspection failed: {error}",
                paths=(),
            ),
        )


def _cuda_refusal(
    capabilities: BackendCapabilities,
    /,
    *,
    requirement: str,
    versions: tuple[tuple[str, str], ...],
    issues: tuple[CudaEnvironmentIssue, ...],
) -> BackendAvailability:
    return _probe_result(
        capabilities,
        available=False,
        failure_category="cuda_environment_refused",
        requirement=requirement,
        reason="CUDA environment refused: "
        + "; ".join(issue.message for issue in issues),
        versions=versions,
        environment_issues=issues,
    )


def probe_backend(
    capabilities: BackendCapabilities,
    /,
    *,
    module: str,
    requirement: str,
    distributions: tuple[str, ...] = (),
    supported_platforms: tuple[str, ...] | None = None,
    device_platform: str | None = None,
    cuda_libraries: Callable[[], tuple[str, ...]] | None = None,
) -> BackendAvailability:
    """Probe one optional module and preserve missing/import/linker evidence.

    ``device_platform`` names the JAX runtime (for example ``"cuda"``) that a
    device provider executes on; without such a device it is unavailable and the
    module is not imported. ``cuda_libraries`` locates the provider's native
    objects without importing it; unsupported CUDA installations refuse the
    provider before import, even with CPU selected. With CUDA enabled, foreign
    provider dependencies are also refused before import.
    """
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
        specification = (
            _native_module_specification(module_)
            if cuda_libraries is not None
            else importlib.util.find_spec(module_)
        )
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
    facts = (
        collect_cuda_environment(initialize_backend=False)
        if cuda_libraries is not None
        else None
    )
    cuda = facts is not None and cuda_enabled(facts)
    if facts is not None:
        issues = tuple(
            issue
            for issue in assess_cuda_environment(facts).issues
            if issue.severity == "error"
        )
        if issues:
            return _cuda_refusal(
                capabilities, requirement=requirement_, versions=versions, issues=issues
            )
    if device_platform is not None and not any(
        runtime_platform(device) == device_platform for device in jax.devices()
    ):
        return _probe_result(
            capabilities,
            available=False,
            failure_category="unsupported_device",
            requirement=requirement_,
            reason=f"no JAX {device_platform} device is available",
            versions=versions,
        )
    if cuda_libraries is not None and facts is not None and cuda:
        # A device check initialized JAX, whose plugin preloaded its NVIDIA wheels.
        issues = _cuda_dependency_issues(
            cuda_libraries, facts, jax_backend_preloaded=device_platform is not None
        )
        if issues:
            return _cuda_refusal(
                capabilities, requirement=requirement_, versions=versions, issues=issues
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
