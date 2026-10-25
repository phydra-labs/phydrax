#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host CUDA facts, pure compatibility judgments, and native dependency attribution."""

from __future__ import annotations

import ctypes
import importlib.metadata
import importlib.util
import os
import platform
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import assert_never, final, Literal, TypeAlias

import equinox as eqx
import jax

from . import _jax_config
from ._elf_dependencies import read_dynamic_section
from ._execution_device import device_platform
from ._native_dependencies import (
    DependencyClosure,
    resolve_native_dependencies,
    ResolvedDependency,
    UnresolvedDependency,
)
from ._strict import StrictModule
from .typing import checked, parse


CudaIssueCode: TypeAlias = Literal[
    "no-cuda-plugin",
    "duplicate-cuda-plugins",
    "plugin-major-mismatch",
    "unsupported-cuda-major",
    "plugin-version-mismatch",
    "plugin-invalid",
    "native-inspection-unavailable",
    "unowned-cuda-library",
    "foreign-nvidia-distributions",
    "driver-too-old",
    "cpu-platform-selected",
    "backend-unavailable",
    "library-path-shadowing",
    "mixed-cuda-majors",
    "provider-major-mismatch",
    "provider-dependency-unresolved",
    "companion-missing",
    "companion-invalid",
    "component-unavailable",
    "companion-version-mismatch",
    "companion-major-mismatch",
    "native-dependency-missing",
]
CudaIssueSeverity: TypeAlias = Literal["error", "warning"]


def _text(value: str, name: str, /) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text.")
    if not value:
        raise ValueError(f"{name} must be non-empty.")
    return value


def _texts(value: tuple[str, ...], name: str, /) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple.")
    return tuple(_text(item, name) for item in value)


def _major(value: int | None, name: str, /) -> int | None:
    if value is not None:
        if type(value) is not int:
            raise TypeError(f"{name} must be an integer or None.")
        if value <= 0:
            raise ValueError(f"{name} must be positive.")
    return value


def _version(value: tuple[int, int] | None, name: str, /) -> tuple[int, int] | None:
    if value is not None:
        if not isinstance(value, tuple) or len(value) != 2:
            raise TypeError(f"{name} must be a (major, minor) tuple or None.")
        if any(type(part) is not int for part in value):
            raise TypeError(f"{name} must contain integers.")
        if value[0] <= 0 or value[1] < 0:
            raise ValueError(f"{name} must have a positive major and non-negative minor.")
    return value


@final
class CudaPluginInstallation(StrictModule):
    """One installed CUDA plugin owner, not merely a deduplicated toolkit major."""

    distribution: str = eqx.field(static=True)
    version: str = eqx.field(static=True)
    cuda_major: int = eqx.field(static=True)

    def __init__(self, *, distribution: str, version: str, cuda_major: int) -> None:
        distribution_ = re.sub(
            r"[-_.]+", "-", _text(distribution, "distribution")
        ).lower()
        version_ = _text(version, "version")
        major = _major(cuda_major, "cuda_major")
        if major is None:
            raise TypeError("cuda_major must be an integer.")
        self.distribution = distribution_
        self.version = version_
        self.cuda_major = major


@final
class MappedCudaLibrary(StrictModule):
    """A mapped CUDA-family library and its known toolkit major, if any."""

    soname: str = eqx.field(static=True)
    path: str = eqx.field(static=True)
    major: int | None = eqx.field(static=True)

    def __init__(self, *, soname: str, path: str, major: int | None) -> None:
        soname_ = _text(soname, "soname")
        path_ = _text(path, "path")
        major_ = _major(major, "major")
        self.soname = soname_
        self.path = path_
        self.major = major_


@final
class CudaEnvironmentIssue(StrictModule):
    """Actionable CUDA compatibility evidence."""

    code: CudaIssueCode = eqx.field(static=True)
    severity: CudaIssueSeverity = eqx.field(static=True)
    message: str = eqx.field(static=True)
    paths: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        code: CudaIssueCode,
        severity: CudaIssueSeverity,
        message: str,
        paths: tuple[str, ...] = (),
    ) -> None:
        code_ = parse(code, CudaIssueCode, "code")
        severity_ = parse(severity, CudaIssueSeverity, "severity")
        message_ = _text(message, "message")
        paths_ = _texts(paths, "paths")
        self.code = code_
        self.severity = severity_
        self.message = message_
        self.paths = paths_


@final
class CudaEnvironmentFacts(StrictModule):
    """Immutable host evidence; constructing facts never probes the host."""

    host_platform: str = eqx.field(static=True)
    host_machine: str = eqx.field(static=True)
    requested_major: int | None = eqx.field(static=True)
    plugins: tuple[CudaPluginInstallation, ...] = eqx.field(static=True)
    foreign_nvidia_distributions: tuple[str, ...] = eqx.field(static=True)
    wheel_library_majors: tuple[tuple[str, int], ...] = eqx.field(static=True)
    selected_platforms: tuple[str, ...] | None = eqx.field(static=True)
    backends_initialized: bool = eqx.field(static=True)
    device_probed: bool = eqx.field(static=True)
    runtime_version: tuple[int, int] | None = eqx.field(static=True)
    compiled_cuda_major: int | None = eqx.field(static=True)
    driver_compatibility_version: tuple[int, int] | None = eqx.field(static=True)
    runtime_identity: tuple[tuple[str, str], ...] = eqx.field(static=True)
    backend_error: str | None = eqx.field(static=True)
    driver_version: tuple[int, int] | None = eqx.field(static=True)
    library_path_cuda_dirs: tuple[str, ...] = eqx.field(static=True)
    mapped_libraries: tuple[MappedCudaLibrary, ...] = eqx.field(static=True)
    installation_error: str | None = eqx.field(static=True)
    inspection_error: str | None = eqx.field(static=True)

    def __init__(
        self,
        *,
        host_platform: str,
        host_machine: str,
        requested_major: int | None = None,
        plugins: tuple[CudaPluginInstallation, ...] = (),
        foreign_nvidia_distributions: tuple[str, ...] = (),
        wheel_library_majors: tuple[tuple[str, int], ...] = (),
        selected_platforms: tuple[str, ...] | None = None,
        backends_initialized: bool = False,
        device_probed: bool = False,
        runtime_version: tuple[int, int] | None = None,
        compiled_cuda_major: int | None = None,
        driver_compatibility_version: tuple[int, int] | None = None,
        runtime_identity: tuple[tuple[str, str], ...] = (),
        backend_error: str | None = None,
        driver_version: tuple[int, int] | None = None,
        library_path_cuda_dirs: tuple[str, ...] = (),
        mapped_libraries: tuple[MappedCudaLibrary, ...] = (),
        installation_error: str | None = None,
        inspection_error: str | None = None,
    ) -> None:
        host_platform_ = _text(host_platform, "host_platform")
        host_machine_ = _text(host_machine, "host_machine")
        requested_major_ = _major(requested_major, "requested_major")
        if not isinstance(plugins, tuple) or any(
            not isinstance(plugin, CudaPluginInstallation) for plugin in plugins
        ):
            raise TypeError("plugins must be a tuple of CudaPluginInstallation values.")
        foreign_ = _texts(foreign_nvidia_distributions, "foreign_nvidia_distributions")
        if not isinstance(wheel_library_majors, tuple):
            raise TypeError("wheel_library_majors must be a tuple.")
        for entry in wheel_library_majors:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("wheel_library_majors must contain (path, major) pairs.")
            _text(entry[0], "wheel_library_majors path")
            if entry[1] is None:
                raise TypeError("wheel_library_majors must contain known majors.")
            _major(entry[1], "wheel_library_majors major")
        selected_ = (
            None
            if selected_platforms is None
            else _texts(selected_platforms, "selected_platforms")
        )
        if type(backends_initialized) is not bool or type(device_probed) is not bool:
            raise TypeError("backends_initialized and device_probed must be bools.")
        runtime_ = _version(runtime_version, "runtime_version")
        compiled_major_ = _major(compiled_cuda_major, "compiled_cuda_major")
        compatibility_ = _version(
            driver_compatibility_version, "driver_compatibility_version"
        )
        if not isinstance(runtime_identity, tuple):
            raise TypeError("runtime_identity must be a tuple of text pairs.")
        for entry in runtime_identity:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("runtime_identity must contain text pairs.")
            _text(entry[0], "runtime_identity key")
            _text(entry[1], "runtime_identity value")
        error_ = None if backend_error is None else _text(backend_error, "backend_error")
        driver_ = _version(driver_version, "driver_version")
        library_dirs_ = _texts(library_path_cuda_dirs, "library_path_cuda_dirs")
        installation_error_ = (
            None
            if installation_error is None
            else _text(installation_error, "installation_error")
        )
        inspection_error_ = (
            None
            if inspection_error is None
            else _text(inspection_error, "inspection_error")
        )
        if not isinstance(mapped_libraries, tuple) or any(
            not isinstance(item, MappedCudaLibrary) for item in mapped_libraries
        ):
            raise TypeError(
                "mapped_libraries must be a tuple of MappedCudaLibrary values."
            )
        self.host_platform = host_platform_
        self.host_machine = host_machine_
        self.requested_major = requested_major_
        self.plugins = tuple(
            sorted(
                plugins,
                key=lambda item: (item.distribution, item.version, item.cuda_major),
            )
        )
        self.foreign_nvidia_distributions = tuple(sorted(foreign_))
        self.wheel_library_majors = tuple(sorted(wheel_library_majors))
        self.selected_platforms = selected_
        self.backends_initialized = backends_initialized
        self.device_probed = device_probed
        self.runtime_version = runtime_
        self.compiled_cuda_major = compiled_major_
        self.driver_compatibility_version = compatibility_
        self.runtime_identity = tuple(sorted(runtime_identity))
        self.backend_error = error_
        self.driver_version = driver_
        self.library_path_cuda_dirs = library_dirs_
        self.mapped_libraries = mapped_libraries
        self.installation_error = installation_error_
        self.inspection_error = inspection_error_


@final
class CudaEnvironmentReport(StrictModule):
    """CUDA evidence and ordered judgments, with warnings distinct from refusal."""

    facts: CudaEnvironmentFacts = eqx.field(static=True)
    issues: tuple[CudaEnvironmentIssue, ...] = eqx.field(static=True)

    @checked
    def __init__(
        self,
        *,
        facts: CudaEnvironmentFacts,
        issues: tuple[CudaEnvironmentIssue, ...] = (),
    ) -> None:
        if not isinstance(issues, tuple) or any(
            not isinstance(issue, CudaEnvironmentIssue) for issue in issues
        ):
            raise TypeError("issues must be a tuple of CudaEnvironmentIssue values.")
        self.facts = facts
        self.issues = issues

    @property
    def successful(self) -> bool:
        for issue in self.issues:
            match issue.severity:
                case "error":
                    return False
                case "warning":
                    pass
                case _ as unreachable:
                    assert_never(unreachable)
        return True


@checked
def cuda_enabled(facts: CudaEnvironmentFacts, /) -> bool:
    """Whether the CUDA13 plugin is installed and CUDA is not explicitly excluded."""
    return (
        len(facts.plugins) == 1
        and facts.plugins[0].cuda_major == 13
        and (
            facts.selected_platforms is None
            or any(item in ("cuda", "gpu") for item in facts.selected_platforms)
        )
    )


@checked
def assess_cuda_environment(facts: CudaEnvironmentFacts, /) -> CudaEnvironmentReport:
    """Assess supplied facts without collecting or initializing anything."""
    issues: list[CudaEnvironmentIssue] = []

    def add(
        code: CudaIssueCode,
        severity: CudaIssueSeverity,
        message: str,
        paths: tuple[str, ...] = (),
    ) -> None:
        issues.append(
            CudaEnvironmentIssue(
                code=code, severity=severity, message=message, paths=paths
            )
        )

    plugin = facts.plugins[0].cuda_major if len(facts.plugins) == 1 else None
    if facts.requested_major is not None and not facts.plugins:
        add(
            "no-cuda-plugin",
            "error"
            if (
                facts.host_platform == "linux"
                and facts.host_machine in ("x86_64", "aarch64")
            )
            or facts.host_platform == "win32"
            else "warning",
            f"CUDA {facts.requested_major} was requested but no JAX CUDA plugin is installed on "
            f"{facts.host_platform}/{facts.host_machine}.",
        )
    if len(facts.plugins) > 1:
        add(
            "duplicate-cuda-plugins",
            "error",
            f"Multiple JAX CUDA plugins are installed: {tuple(item.distribution for item in facts.plugins)}.",
        )
    _append_plugin_identity_issues(facts, issues)
    if facts.installation_error is not None:
        add("plugin-invalid", "error", facts.installation_error)
    if facts.inspection_error is not None:
        add("native-inspection-unavailable", "error", facts.inspection_error)
    if (plugin is not None and plugin != 13) or (
        facts.requested_major is not None and facts.requested_major != 13
    ):
        add(
            "plugin-major-mismatch",
            "error",
            f"Phydrax requires CUDA 13; installed CUDA intent is {facts.requested_major}, "
            f"and the JAX plugin major is {plugin}. Install phydrax[cuda13].",
        )
    if facts.compiled_cuda_major not in (None, 13) or (
        facts.host_platform != "win32"
        and facts.runtime_version is not None
        and facts.runtime_version[0] != 13
    ):
        add(
            "unsupported-cuda-major",
            "error",
            f"Phydrax requires a compiled CUDA 13 stack; observed compiled major "
            f"{facts.compiled_cuda_major} and runtime {facts.runtime_version}. Install phydrax[cuda13].",
        )
    if facts.foreign_nvidia_distributions:
        add(
            "foreign-nvidia-distributions",
            "error",
            "Remove other-major NVIDIA distributions: pip uninstall "
            + " ".join(facts.foreign_nvidia_distributions),
        )
    compatibility = (
        facts.driver_compatibility_version or facts.driver_version
        if facts.host_platform == "win32"
        else facts.driver_version
    )
    if plugin is not None and compatibility is not None and compatibility[0] < plugin:
        driver_major = compatibility[0]
        add(
            "driver-too-old",
            "error",
            f"The driver supports CUDA {driver_major}.{compatibility[1]}, below the CUDA {plugin} plugin. "
            "Upgrade the NVIDIA driver for CUDA 13.",
        )
    if plugin == 13 and not cuda_enabled(facts):
        add(
            "cpu-platform-selected",
            "warning",
            f"Selected platforms {facts.selected_platforms} exclude CUDA.",
        )
    unavailable = (
        cuda_enabled(facts) and facts.device_probed and facts.runtime_version is None
    )
    if unavailable:
        add(
            "backend-unavailable",
            "error",
            "No CUDA backend is available. "
            + (
                f"Backend initialization failed: {facts.backend_error}. "
                if facts.backend_error is not None
                else ""
            )
            + f"CUDA LD_LIBRARY_PATH directories: {facts.library_path_cuda_dirs}.",
            facts.library_path_cuda_dirs,
        )
    if facts.library_path_cuda_dirs and not unavailable:
        add(
            "library-path-shadowing",
            "warning",
            f"LD_LIBRARY_PATH can shadow NVIDIA wheel libraries: {facts.library_path_cuda_dirs}.",
            facts.library_path_cuda_dirs,
        )
    _append_mapped_library_issues(facts, issues)
    return CudaEnvironmentReport(facts=facts, issues=tuple(issues))


def _append_plugin_identity_issues(
    facts: CudaEnvironmentFacts, issues: list[CudaEnvironmentIssue]
) -> None:
    """Admit the exact supported implementation namespace and plugin release."""
    if len(facts.plugins) != 1 or facts.plugins[0].cuda_major != 13:
        return
    installation = facts.plugins[0]
    if (
        installation.distribution in ("jax-cuda13-plugin", "winjax-cuda13-plugin")
        and installation.version != "0.12.0"
    ):
        issues.append(
            CudaEnvironmentIssue(
                code="plugin-version-mismatch",
                severity="error",
                message=f"CUDA requires the qualified JAX 0.12.0 plugin; "
                f"found {installation.distribution} {installation.version}.",
            )
        )
    if (
        facts.host_platform == "win32"
        and installation.distribution != "phydrax-cuda13-pjrt"
    ):
        issues.append(
            CudaEnvironmentIssue(
                code="plugin-invalid",
                severity="error",
                message="Native Windows CUDA requires the matching phydrax-cuda13-pjrt "
                "runtime; foreign CUDA plugins are not admitted.",
            )
        )


def _append_mapped_library_issues(
    facts: CudaEnvironmentFacts, issues: list[CudaEnvironmentIssue]
) -> None:
    """Preserve mapped-library diagnostic order and platform ownership semantics."""
    majors = sorted(
        {library.major for library in facts.mapped_libraries if library.major is not None}
    )
    paths = tuple(
        library.path for library in facts.mapped_libraries if library.major is not None
    )
    if len(majors) > 1:
        issues.append(
            CudaEnvironmentIssue(
                code="mixed-cuda-majors",
                severity="error",
                message=f"Mapped CUDA libraries contain multiple toolkit majors {majors}: {paths}.",
                paths=paths,
            )
        )
    elif majors and majors != [13]:
        issues.append(
            CudaEnvironmentIssue(
                code="unsupported-cuda-major",
                severity="error",
                message=f"Phydrax requires CUDA 13, but mapped libraries use CUDA {majors[0]}: {paths}.",
                paths=paths,
            )
        )
    if facts.host_platform != "win32":
        return
    owned = {
        path.replace("\\", "/").casefold()
        for path, major in facts.wheel_library_majors
        if major == 13
    }
    unowned = tuple(
        library.path
        for library in facts.mapped_libraries
        if library.path.replace("\\", "/").casefold() not in owned
    )
    if unowned:
        issues.append(
            CudaEnvironmentIssue(
                code="unowned-cuda-library",
                severity="error",
                message=f"Mapped NVIDIA userland DLLs lack CUDA 13 wheel ownership: {unowned}.",
                paths=unowned,
            )
        )


_CUDA_SONAME = re.compile(
    r"lib(?:cudart|cublasLt|cublas|cusparse|cusolver|cufft|curand|nvJitLink|nvrtc|cudnn[^/]*|nccl|cudss|cupti|nvshmem_host|cutensor|cutensornet|cuest)\.so(?:\.\d+)*\Z"
)
_MAJOR_SONAME = re.compile(
    r"lib(?:cudart|cublasLt|cublas|nvrtc|nvJitLink|cupti)\.so\.(\d+)(?:\.\d+)*\Z"
)
_MATH_SONAME = re.compile(r"lib(?:cusolver|cufft)\.so\.(11|12)(?:\.\d+)*\Z")
_CUDA_DLL = re.compile(
    r"(?:cudart64_\d+|cublas(?:Lt)?64_\d+|cusparse64_\d+|cusolver64_\d+|"
    r"cufft64_\d+|curand64_\d+|nvJitLink_[\d_]+|nvrtc(?:-builtins)?64_[\d_]+|"
    r"cupti64_[\d_]+|cudnn[^/\\]*|cudss64_\d+|cutensor(?:net)?|cuest)\.dll\Z",
    re.IGNORECASE,
)


def _cuda_library_name(name: str, /) -> bool:
    return (
        _CUDA_SONAME.fullmatch(name) is not None or _CUDA_DLL.fullmatch(name) is not None
    )


def _cuda_library_major(
    soname: str, path: str, wheel_library_majors: tuple[tuple[str, int], ...], /
) -> int | None:
    match = _MAJOR_SONAME.fullmatch(soname)
    if match is not None:
        return int(match[1])
    match = _MATH_SONAME.fullmatch(soname)
    if match is not None:
        return {11: 12, 12: 13}[int(match[1])]
    if not _cuda_library_name(soname):
        return None
    windows = soname.lower().endswith(".dll")
    realpath = path.replace("\\", "/")
    if windows:
        realpath = realpath.casefold()
    for wheel_path, major in wheel_library_majors:
        candidate = wheel_path.replace("\\", "/")
        if realpath == (candidate.casefold() if windows else candidate):
            return major
    if windows:
        match = re.fullmatch(
            r"(?:cudart|cublas(?:Lt)?)64_(\d+)\.dll", soname, re.IGNORECASE
        )
        return None if match is None else int(match[1])
    match = re.search(r"/cuda-(\d+)\.\d+/", realpath)
    return None if match is None else int(match[1])


@checked
def assess_provider_dependencies(
    closure: DependencyClosure, facts: CudaEnvironmentFacts, /
) -> tuple[CudaEnvironmentIssue, ...]:
    """Refuse CUDA dependencies incompatible with the enabled JAX plugin."""
    if not cuda_enabled(facts):
        return ()
    plugin = facts.plugins[0].cuda_major
    mismatches: list[ResolvedDependency] = []
    for dependency in closure.resolved:
        major = _cuda_library_major(
            dependency.soname, dependency.path, facts.wheel_library_majors
        )
        if major is not None and major != plugin:
            mismatches.append(dependency)
    unresolved: list[UnresolvedDependency] = [
        dependency
        for dependency in closure.unresolved
        if _cuda_library_name(dependency.soname)
    ]
    issues: list[CudaEnvironmentIssue] = []
    if mismatches:
        issues.append(
            CudaEnvironmentIssue(
                code="provider-major-mismatch",
                severity="error",
                message=f"Provider dependencies do not match CUDA {plugin}: "
                + "; ".join(
                    f"{item.soname} at {item.path}, requested by {item.requested_by}"
                    for item in mismatches
                ),
                paths=tuple(item.path for item in mismatches),
            )
        )
    if unresolved:
        issues.append(
            CudaEnvironmentIssue(
                code="provider-dependency-unresolved",
                severity="error",
                message="Unresolved CUDA provider dependencies: "
                + "; ".join(
                    f"{item.soname}, requested by {item.requested_by}: {item.reason}"
                    for item in unresolved
                ),
            )
        )
    return tuple(issues)


# JAX <0.12 CUDA13 wheel preloads, in load order.
_JAX_CUDA_PRELOADS = (
    ("cu13", ("libcudart.so.13",)),
    ("cu13", ("libnvrtc.so.13",)),
    ("cu13", ("libcublas.so.13", "libcublasLt.so.13")),
    ("nccl", ("libnccl.so.2",)),
    ("cu13", ("libcupti.so.13",)),
    ("cu13", ("libcusparse.so.12",)),
    ("cu13", ("libcusolver.so.12",)),
    ("cu13", ("libcufft.so.12",)),
    ("nvshmem", ("libnvshmem_host.so.3",)),
    ("cudnn", ("libcudnn.so.9",)),
)


def jax_plugin_preloads() -> tuple[tuple[str, str], ...]:
    """Locate CUDA13 libraries JAX preloads without importing native code."""
    loaded: dict[str, str] = {}
    if importlib.util.find_spec("nvidia") is None:
        return ()
    for module, sonames in _JAX_CUDA_PRELOADS:
        spec = importlib.util.find_spec(f"nvidia.{module}")
        if spec is None or not spec.submodule_search_locations:
            continue
        directory = Path(next(iter(spec.submodule_search_locations))) / "lib"
        for soname in sonames:
            path = directory / soname
            if soname not in loaded and path.is_file():
                loaded[soname] = str(path)
    return tuple(loaded.items())


@checked
def provider_dependency_closure(
    path: str, /, *, facts: CudaEnvironmentFacts, jax_backend_preloaded: bool
) -> DependencyClosure:
    """Model a provider load after JAX's optional NVIDIA wheel preloads."""
    preloaded = (
        jax_plugin_preloads()
        if facts.host_platform == "linux"
        and jax_backend_preloaded
        and cuda_enabled(facts)
        else ()
    )
    return resolve_native_dependencies(
        path, library_path=_library_path(), executable=sys.executable, preloaded=preloaded
    )


def _library_path() -> tuple[str, ...]:
    return tuple(
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    )


def competing_cuda_libraries(
    library_path: str, mapped_libraries: tuple[MappedCudaLibrary, ...], /
) -> tuple[tuple[str, str], ...]:
    """Find real competing ELF or PE objects in explicitly supplied directories."""
    directories = tuple(Path(entry) for entry in library_path.split(os.pathsep) if entry)
    candidates: set[tuple[str, str]] = set()
    for library in mapped_libraries:
        mapped = Path(library.path).resolve()
        windows = library.soname.lower().endswith(".dll")
        soname = library.soname if windows else read_dynamic_section(mapped).soname
        if soname is None or not _cuda_library_name(soname):
            continue
        for directory in directories:
            candidate = directory / soname
            if not candidate.is_file() or candidate.samefile(mapped):
                continue
            if windows:
                from ._pe_dependencies import inspect_pe

                inspect_pe(candidate)
            elif read_dynamic_section(candidate).soname != soname:
                raise ValueError(
                    f"Competing CUDA library {candidate} has a different SONAME."
                )
            candidates.add((soname, str(candidate.resolve())))
    return tuple(sorted(candidates))


def _distribution_major(distribution: importlib.metadata.Distribution, /) -> int | None:
    name = distribution.metadata["Name"]
    if name.lower().replace("_", "-") == "phydrax-cuest":
        return 13
    match = re.search(r"-cu(\d+)$", name.lower().replace("_", "-"))
    if match is not None:
        return int(match[1])
    majors = {
        int(match[1])
        for file in distribution.files or ()
        if (match := re.search(r"(?:^|/)nvidia/cu(\d+)/", str(file))) is not None
    }
    return next(iter(majors)) if len(majors) == 1 else None


def _platform_version(value: str, /) -> tuple[int, int]:
    match = re.fullmatch(r"cuda ([0-9]+)", value)
    if match is None:
        raise ValueError(f"Malformed CUDA platform version: {value!r}.")
    encoded = int(match[1])
    if encoded < 1000 or encoded % 10:
        raise ValueError(f"Malformed CUDA platform version: {value!r}.")
    return encoded // 1000, (encoded % 1000) // 10


def _driver_version() -> tuple[int, int] | None:
    try:
        library = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return None
    version = ctypes.c_int()
    function = library.cuDriverGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    if function(ctypes.byref(version)) != 0 or version.value <= 0:
        return None
    return version.value // 1000, (version.value % 1000) // 10


def collect_cuda_installation() -> CudaEnvironmentFacts:
    """Collect installed intent and artifact metadata without loading native code."""
    requested: int | None = None
    plugins: list[CudaPluginInstallation] = []
    nvidia: list[tuple[str, int]] = []
    wheel_majors: dict[str, int] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        normalized = re.sub(r"[-_.]+", "-", name).lower()
        if normalized == "phydrax-cuda-major":
            requested = int(distribution.version.split(".")[0])
        match = re.fullmatch(r"(?:jax|winjax)-cuda(\d+)-plugin", normalized)
        if match is not None or normalized == "phydrax-cuda13-pjrt":
            plugins.append(
                CudaPluginInstallation(
                    distribution=normalized,
                    version=distribution.version,
                    cuda_major=int(match[1]) if match is not None else 13,
                )
            )
        if (
            normalized.startswith("nvidia-")
            or normalized == "phydrax-cuest"
            or re.fullmatch(r"(?:cutensor|cutensornet)-cu\d+", normalized)
        ):
            major = _distribution_major(distribution)
            if major is not None:
                nvidia.append((name, major))
                for file in distribution.files or ():
                    if re.search(r"\.so(?:\.\d+)*$|\.dll$", str(file), re.IGNORECASE):
                        wheel_majors[
                            os.path.realpath(str(distribution.locate_file(file)))
                        ] = major
    installation_error: str | None = None
    compiled_major: int | None = None
    if sys.platform == "win32" and any(
        plugin.distribution == "phydrax-cuda13-pjrt" for plugin in plugins
    ):
        try:
            from phydrax_cuda13_pjrt.runtime import installation_record

            record = installation_record()
            compiled_major = record.cuda_major
            if record.issues:
                installation_error = "; ".join(record.issues)
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            installation_error = f"Windows CUDA runtime admission failed: {error}"
    return CudaEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        requested_major=requested,
        plugins=tuple(plugins),
        foreign_nvidia_distributions=tuple(
            sorted(name for name, major in nvidia if major != 13)
        ),
        wheel_library_majors=tuple(sorted(wheel_majors.items())),
        selected_platforms=_jax_config.platforms(),
        backends_initialized=_jax_config.backends_initialized(),
        compiled_cuda_major=compiled_major,
        installation_error=installation_error,
    )


def collect_cuda_environment(*, initialize_backend: bool) -> CudaEnvironmentFacts:
    """Observe the selected runtime; metadata-only startup uses installation facts."""
    facts = collect_cuda_installation()
    library_dirs = (
        ()
        if sys.platform == "win32"
        else tuple(
            directory
            for directory in _library_path()
            if Path(directory).is_dir()
            and any(
                _CUDA_SONAME.fullmatch(path.name) is not None
                for path in Path(directory).iterdir()
            )
        )
    )
    runtime: tuple[int, int] | None = None
    backend_error: str | None = None
    if initialize_backend:
        try:
            devices = jax.devices()
        except RuntimeError as error:
            backend_error = str(error)
        else:
            for device in devices:
                if device_platform(device) == "cuda":
                    runtime = _platform_version(device.client.platform_version)
                    break
    driver: tuple[int, int] | None = None
    inspection_error: str | None = None
    if sys.platform == "win32":
        if cuda_enabled(facts) and facts.installation_error is None:
            try:
                from phydrax_cuda13_pjrt.runtime import driver_version

                encoded = driver_version()
                driver = encoded // 1000, (encoded % 1000) // 10
            except (ImportError, OSError, RuntimeError, ValueError) as error:
                inspection_error = f"Windows CUDA driver inspection failed: {error}"
    else:
        driver = _driver_version()
    return replace(
        facts,
        device_probed=initialize_backend,
        backends_initialized=_jax_config.backends_initialized(),
        runtime_version=runtime,
        backend_error=backend_error,
        driver_version=driver,
        driver_compatibility_version=driver if sys.platform == "win32" else None,
        library_path_cuda_dirs=library_dirs,
        inspection_error=inspection_error,
    )


def cuda_environment() -> CudaEnvironmentReport:
    """Probe CUDA and diagnose the libraries currently mapped in this process."""
    facts = collect_cuda_environment(initialize_backend=True)
    if sys.platform == "win32":
        if not cuda_enabled(facts) or facts.installation_error is not None:
            return assess_cuda_environment(facts)
        observed: tuple[MappedCudaLibrary, ...] = ()
        identity: tuple[tuple[str, str], ...] = ()
        inspection_errors = [facts.inspection_error] if facts.inspection_error else []
        try:
            from phydrax_cuda13_pjrt.runtime import mapped_libraries

            observed = tuple(
                MappedCudaLibrary(
                    soname=item.name,
                    path=item.path,
                    major=(
                        item.cuda_major
                        if item.cuda_major is not None
                        else _cuda_library_major(
                            item.name, item.path, facts.wheel_library_majors
                        )
                    ),
                )
                for item in mapped_libraries()
                if _cuda_library_name(item.name)
            )
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            inspection_errors.append(f"Windows CUDA library inspection failed: {error}")
        try:
            from phydrax_cuda13_pjrt.runtime import runtime_identity

            identity = tuple(runtime_identity()._asdict().items())
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            inspection_errors.append(
                f"Windows CUDA runtime identity inspection failed: {error}"
            )
        return assess_cuda_environment(
            replace(
                facts,
                mapped_libraries=observed,
                runtime_identity=identity,
                inspection_error="; ".join(inspection_errors) or None,
            )
        )
    mapped: dict[str, MappedCudaLibrary] = {}
    maps = Path("/proc/self/maps")
    if maps.is_file():
        for line in maps.read_text().splitlines():
            fields = line.split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            path = os.path.realpath(fields[5].removesuffix(" (deleted)"))
            soname = Path(path).name
            if _CUDA_SONAME.fullmatch(soname) is not None:
                mapped[path] = MappedCudaLibrary(
                    soname=soname,
                    path=path,
                    major=_cuda_library_major(soname, path, facts.wheel_library_majors),
                )
    facts = replace(
        facts, mapped_libraries=tuple(mapped[path] for path in sorted(mapped))
    )
    return assess_cuda_environment(facts)
