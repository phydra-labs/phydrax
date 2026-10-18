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
from ._elf_dependencies import (
    DependencyClosure,
    resolve_dependency_closure,
    ResolvedDependency,
    UnresolvedDependency,
)
from ._execution_device import device_platform
from ._strict import StrictModule
from .typing import checked, parse


CudaIssueCode: TypeAlias = Literal[
    "no-cuda-plugin",
    "duplicate-cuda-plugins",
    "plugin-major-mismatch",
    "unsupported-cuda-major",
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
    plugin_majors: tuple[int, ...] = eqx.field(static=True)
    foreign_nvidia_distributions: tuple[str, ...] = eqx.field(static=True)
    wheel_library_majors: tuple[tuple[str, int], ...] = eqx.field(static=True)
    selected_platforms: tuple[str, ...] | None = eqx.field(static=True)
    backends_initialized: bool = eqx.field(static=True)
    device_probed: bool = eqx.field(static=True)
    runtime_version: tuple[int, int] | None = eqx.field(static=True)
    backend_error: str | None = eqx.field(static=True)
    driver_version: tuple[int, int] | None = eqx.field(static=True)
    library_path_cuda_dirs: tuple[str, ...] = eqx.field(static=True)
    mapped_libraries: tuple[MappedCudaLibrary, ...] = eqx.field(static=True)

    def __init__(
        self,
        *,
        host_platform: str,
        host_machine: str,
        requested_major: int | None = None,
        plugin_majors: tuple[int, ...] = (),
        foreign_nvidia_distributions: tuple[str, ...] = (),
        wheel_library_majors: tuple[tuple[str, int], ...] = (),
        selected_platforms: tuple[str, ...] | None = None,
        backends_initialized: bool = False,
        device_probed: bool = False,
        runtime_version: tuple[int, int] | None = None,
        backend_error: str | None = None,
        driver_version: tuple[int, int] | None = None,
        library_path_cuda_dirs: tuple[str, ...] = (),
        mapped_libraries: tuple[MappedCudaLibrary, ...] = (),
    ) -> None:
        host_platform_ = _text(host_platform, "host_platform")
        host_machine_ = _text(host_machine, "host_machine")
        requested_major_ = _major(requested_major, "requested_major")
        if not isinstance(plugin_majors, tuple):
            raise TypeError("plugin_majors must be a tuple.")
        for major in plugin_majors:
            if major is None:
                raise TypeError("plugin_majors must contain integers.")
            _major(major, "plugin_majors")
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
        error_ = None if backend_error is None else _text(backend_error, "backend_error")
        driver_ = _version(driver_version, "driver_version")
        library_dirs_ = _texts(library_path_cuda_dirs, "library_path_cuda_dirs")
        if not isinstance(mapped_libraries, tuple) or any(
            not isinstance(item, MappedCudaLibrary) for item in mapped_libraries
        ):
            raise TypeError(
                "mapped_libraries must be a tuple of MappedCudaLibrary values."
            )
        self.host_platform = host_platform_
        self.host_machine = host_machine_
        self.requested_major = requested_major_
        self.plugin_majors = tuple(sorted(plugin_majors))
        self.foreign_nvidia_distributions = tuple(sorted(foreign_))
        self.wheel_library_majors = tuple(sorted(wheel_library_majors))
        self.selected_platforms = selected_
        self.backends_initialized = backends_initialized
        self.device_probed = device_probed
        self.runtime_version = runtime_
        self.backend_error = error_
        self.driver_version = driver_
        self.library_path_cuda_dirs = library_dirs_
        self.mapped_libraries = mapped_libraries


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
    return facts.plugin_majors == (13,) and (
        facts.selected_platforms is None
        or any(item in ("cuda", "gpu") for item in facts.selected_platforms)
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

    plugin = facts.plugin_majors[0] if len(facts.plugin_majors) == 1 else None
    if facts.requested_major is not None and not facts.plugin_majors:
        add(
            "no-cuda-plugin",
            "error"
            if (facts.host_platform, facts.host_machine) == ("linux", "x86_64")
            else "warning",
            f"CUDA {facts.requested_major} was requested but no JAX CUDA plugin is installed on "
            f"{facts.host_platform}/{facts.host_machine}.",
        )
    if len(facts.plugin_majors) > 1:
        add(
            "duplicate-cuda-plugins",
            "error",
            f"Multiple JAX CUDA plugins are installed: {facts.plugin_majors}.",
        )
    if (plugin is not None and plugin != 13) or (
        facts.requested_major is not None and facts.requested_major != 13
    ):
        add(
            "plugin-major-mismatch",
            "error",
            f"Phydrax requires CUDA 13; installed CUDA intent is {facts.requested_major}, "
            f"and the JAX plugin major is {plugin}. Install phydrax[cuda13].",
        )
    if facts.runtime_version is not None and facts.runtime_version[0] != 13:
        add(
            "unsupported-cuda-major",
            "error",
            f"Phydrax requires CUDA 13, but the active runtime is CUDA "
            f"{facts.runtime_version[0]}.{facts.runtime_version[1]}. Install phydrax[cuda13].",
        )
    if facts.foreign_nvidia_distributions:
        add(
            "foreign-nvidia-distributions",
            "error",
            "Remove other-major NVIDIA distributions: pip uninstall "
            + " ".join(facts.foreign_nvidia_distributions),
        )
    if (
        plugin is not None
        and facts.driver_version is not None
        and facts.driver_version[0] < plugin
    ):
        driver_major = facts.driver_version[0]
        add(
            "driver-too-old",
            "error",
            f"The driver supports CUDA {driver_major}.{facts.driver_version[1]}, below the CUDA {plugin} plugin. "
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
    majors = sorted(
        {library.major for library in facts.mapped_libraries if library.major is not None}
    )
    if len(majors) > 1:
        paths = tuple(
            library.path
            for library in facts.mapped_libraries
            if library.major is not None
        )
        add(
            "mixed-cuda-majors",
            "error",
            f"Mapped CUDA libraries contain multiple toolkit majors {majors}: {paths}.",
            paths,
        )
    elif majors and majors != [13]:
        paths = tuple(
            library.path
            for library in facts.mapped_libraries
            if library.major is not None
        )
        add(
            "unsupported-cuda-major",
            "error",
            f"Phydrax requires CUDA 13, but mapped libraries use CUDA {majors[0]}: {paths}.",
            paths,
        )
    return CudaEnvironmentReport(facts=facts, issues=tuple(issues))


_CUDA_SONAME = re.compile(
    r"lib(?:cudart|cublasLt|cublas|cusparse|cusolver|cufft|curand|nvJitLink|nvrtc|cudnn[^/]*|nccl|cudss|cupti|nvshmem_host)\.so(?:\.\d+)*\Z"
)
_MAJOR_SONAME = re.compile(
    r"lib(?:cudart|cublasLt|cublas|nvrtc|nvJitLink|cupti)\.so\.(\d+)(?:\.\d+)*\Z"
)
_MATH_SONAME = re.compile(r"lib(?:cusolver|cufft)\.so\.(11|12)(?:\.\d+)*\Z")


def _cuda_library_major(
    soname: str, path: str, wheel_library_majors: tuple[tuple[str, int], ...], /
) -> int | None:
    match = _MAJOR_SONAME.fullmatch(soname)
    if match is not None:
        return int(match[1])
    match = _MATH_SONAME.fullmatch(soname)
    if match is not None:
        return {11: 12, 12: 13}[int(match[1])]
    if _CUDA_SONAME.fullmatch(soname) is None:
        return None
    realpath = os.path.realpath(path)
    for wheel_path, major in wheel_library_majors:
        if realpath == wheel_path:
            return major
    match = re.search(r"/cuda-(\d+)\.\d+/", realpath)
    return None if match is None else int(match[1])


@checked
def assess_provider_dependencies(
    closure: DependencyClosure, facts: CudaEnvironmentFacts, /
) -> tuple[CudaEnvironmentIssue, ...]:
    """Refuse CUDA dependencies incompatible with the enabled JAX plugin."""
    if not cuda_enabled(facts):
        return ()
    plugin = facts.plugin_majors[0]
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
        if _CUDA_SONAME.fullmatch(dependency.soname) is not None
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
        jax_plugin_preloads() if jax_backend_preloaded and cuda_enabled(facts) else ()
    )
    return resolve_dependency_closure(
        path, library_path=_library_path(), executable=sys.executable, preloaded=preloaded
    )


def _library_path() -> tuple[str, ...]:
    return tuple(
        entry
        for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
        if entry
    )


def _distribution_major(distribution: importlib.metadata.Distribution, /) -> int | None:
    name = distribution.metadata["Name"]
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


def collect_cuda_environment(*, initialize_backend: bool) -> CudaEnvironmentFacts:
    """Collect metadata and driver evidence, optionally probing JAX's devices."""
    requested: int | None = None
    plugins: set[int] = set()
    nvidia: list[tuple[str, int]] = []
    wheel_majors: dict[str, int] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"]
        normalized = name.lower().replace("_", "-")
        if normalized == "phydrax-cuda-major":
            requested = int(distribution.version.split(".")[0])
        match = re.fullmatch(r"jax-cuda(\d+)-plugin", normalized)
        if match is not None:
            plugins.add(int(match[1]))
        if normalized.startswith("nvidia-"):
            major = _distribution_major(distribution)
            if major is not None:
                nvidia.append((name, major))
                for file in distribution.files or ():
                    if re.search(r"\.so(?:\.\d+)*$", str(file)):
                        wheel_majors[
                            os.path.realpath(str(distribution.locate_file(file)))
                        ] = major
    foreign = tuple(sorted(name for name, major in nvidia if major != 13))
    library_dirs = tuple(
        directory
        for directory in _library_path()
        if Path(directory).is_dir()
        and any(
            _CUDA_SONAME.fullmatch(path.name) is not None
            for path in Path(directory).iterdir()
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
    return CudaEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        requested_major=requested,
        plugin_majors=tuple(sorted(plugins)),
        foreign_nvidia_distributions=foreign,
        wheel_library_majors=tuple(sorted(wheel_majors.items())),
        selected_platforms=_jax_config.platforms(),
        backends_initialized=_jax_config.backends_initialized(),
        device_probed=initialize_backend,
        runtime_version=runtime,
        backend_error=backend_error,
        driver_version=_driver_version(),
        library_path_cuda_dirs=library_dirs,
    )


def cuda_environment() -> CudaEnvironmentReport:
    """Probe CUDA and diagnose the libraries currently mapped in this process."""
    facts = collect_cuda_environment(initialize_backend=True)
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
