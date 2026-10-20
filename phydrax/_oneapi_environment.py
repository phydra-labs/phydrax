#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only upstream oneAPI installation evidence and explicit runtime observation."""

from __future__ import annotations

import os
import platform
import sys
import sysconfig
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import final, Literal

import jax

from . import _jax_config
from ._validation import canonical_identifier
from .typing import parse


type OneapiIssueCode = Literal[
    "unsupported-host",
    "host-requirements-unknown",
    "no-oneapi-plugin",
    "plugin-version-mismatch",
    "jax-version-mismatch",
    "runtime-dependency-mismatch",
    "conflicting-plugins",
    "platform-not-selected",
    "library-overrides",
    "mixed-frontend-stack",
    "backend-unavailable",
    "runtime-identity-mismatch",
]
type OneapiIssueSeverity = Literal["error", "warning"]

_REQUIRED_VERSIONS = (
    ("jax", "0.12.0"),
    ("jaxlib", "0.11.2"),
    ("jax-oneapi-plugin", "0.11.2"),
    ("jax-oneapi-pjrt", "0.11.2"),
)
_RUNTIME_VERSIONS = (
    ("dpcpp-cpp-rt", "2025.1.1"),
    ("intel-sycl-rt", "2025.1.1"),
    ("tbb", "2022.1.0"),
    ("intel-cmplr-lib-rt", "2025.1.1"),
    ("intel-cmplr-lib-ur", "2025.1.1"),
    ("intel-cmplr-lic-rt", "2025.1.1"),
    ("intel-opencl-rt", "2025.1.1"),
    ("intel-openmp", "2025.1.1"),
    ("impi-rt", "2021.15.0"),
    ("intel-pti", "0.12.0"),
    ("mkl", "2025.1.0"),
    ("onemkl-sycl-blas", "2025.1.0"),
    ("onemkl-sycl-dft", "2025.1.0"),
    ("onemkl-sycl-lapack", "2025.1.0"),
    ("onemkl-sycl-rng", "2025.1.0"),
    ("onemkl-sycl-sparse", "2025.1.0"),
    ("umf", "0.10.0"),
    ("tcmlib", "1.3.0"),
)
_CONFLICTING_PLUGINS = (
    "jax-cuda12-plugin",
    "jax-cuda12-pjrt",
    "jax-cuda13-plugin",
    "jax-cuda13-pjrt",
    "phydrax-metal",
    "libtpu",
    "intel-extension-for-openxla",
)
_OVERRIDE_KEYS = (
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "LIBRARY_PATH",
    "ONEAPI_ROOT",
    "CMPLR_ROOT",
    "MKLROOT",
    "MKL_INSTALL_PATH",
    "L0_INSTALL_PATH",
    "OCL_ICD_FILENAMES",
    "OCL_ICD_VENDORS",
    "ZE_ENABLE_ALT_DRIVERS",
    "ZE_AFFINITY_MASK",
    "ONEAPI_DEVICE_SELECTOR",
    "SYCL_DEVICE_FILTER",
    "SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS",
    "UR_ADAPTERS_FORCE_LOAD",
    "UR_ENABLE_LAYERS",
    "XLA_FLAGS",
    "JAX_PLATFORMS",
)


def _pairs(
    value: tuple[tuple[str, str], ...],
    name: str,
    /,
    *,
    canonical_values: bool = True,
) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple of text pairs.")
    for entry in value:
        if not isinstance(entry, tuple) or len(entry) != 2:
            raise TypeError(f"{name} must contain text pairs.")
        canonical_identifier(entry[0], name)
        if canonical_values:
            canonical_identifier(entry[1], name)
        elif not isinstance(entry[1], str):
            raise TypeError(f"{name} values must be text.")
    if len({key for key, _ in value}) != len(value):
        raise ValueError(f"{name} keys must be unique.")
    return tuple(sorted(value))


@final
@dataclass(frozen=True, slots=True)
class OneapiDeviceFacts:
    """Public JAX device/client evidence; display names do not establish identity."""

    platform: str
    client_platform: str
    runtime_version: str
    device_kind: str
    id: int
    process_index: int

    def __post_init__(self) -> None:
        for value in (
            self.platform,
            self.client_platform,
            self.runtime_version,
            self.device_kind,
        ):
            canonical_identifier(value, "device metadata")
        for value in (self.id, self.process_index):
            if type(value) is not int:
                raise TypeError("Device identifiers must be integers.")
            if value < 0:
                raise ValueError("Device identifiers must be non-negative.")


@final
@dataclass(frozen=True, slots=True)
class OneapiEnvironmentIssue:
    code: OneapiIssueCode
    severity: OneapiIssueSeverity
    message: str

    def __post_init__(self) -> None:
        parse(self.code, OneapiIssueCode, "code")
        parse(self.severity, OneapiIssueSeverity, "severity")
        canonical_identifier(self.message, "message")


@final
@dataclass(frozen=True, slots=True)
class OneapiEnvironmentFacts:
    """Immutable evidence, never an assertion of Intel GPU or FP64 qualification."""

    host_platform: str
    host_machine: str
    python_implementation: str = ""
    python_version: tuple[int, int, int] | None = None
    python_free_threaded: bool | None = None
    libc_name: str = ""
    libc_version: str = ""
    host_release: str = ""
    wsl: bool = False
    versions: tuple[tuple[str, str], ...] = ()
    selected_platforms: tuple[str, ...] | None = None
    environment_overrides: tuple[tuple[str, str], ...] = ()
    provider_libraries: tuple[tuple[str, str], ...] = ()
    backends_initialized: bool = False
    device_probed: bool = False
    devices: tuple[OneapiDeviceFacts, ...] = ()
    runtime_platform: str | None = None
    runtime_version: str | None = None
    backend_error: str | None = None

    def __post_init__(self) -> None:
        canonical_identifier(self.host_platform, "host_platform")
        canonical_identifier(self.host_machine, "host_machine")
        for value in (
            self.python_implementation,
            self.libc_name,
            self.libc_version,
            self.host_release,
        ):
            if not isinstance(value, str):
                raise TypeError("Host metadata must be text.")
        if self.libc_name == "glibc" and self.libc_version:
            parts = self.libc_version.split(".")
            if len(parts) < 2 or any(not part.isdecimal() for part in parts):
                raise ValueError("glibc version must contain decimal release components.")
        if self.python_version is not None:
            if (
                not isinstance(self.python_version, tuple)
                or len(self.python_version) != 3
            ):
                raise TypeError("python_version must be a three-integer tuple or None.")
            if any(type(part) is not int for part in self.python_version):
                raise TypeError("python_version must contain integers.")
            if any(part < 0 for part in self.python_version):
                raise ValueError("python_version must contain non-negative integers.")
        if (
            self.python_free_threaded is not None
            and type(self.python_free_threaded) is not bool
        ):
            raise TypeError("python_free_threaded must be a boolean or None.")
        object.__setattr__(self, "versions", _pairs(self.versions, "versions"))
        object.__setattr__(
            self,
            "environment_overrides",
            _pairs(
                self.environment_overrides,
                "environment_overrides",
                canonical_values=False,
            ),
        )
        if not isinstance(self.provider_libraries, tuple):
            raise TypeError("provider_libraries must contain distribution/path pairs.")
        for entry in self.provider_libraries:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError(
                    "provider_libraries must contain distribution/path pairs."
                )
            for value in entry:
                canonical_identifier(value, "provider library")
        object.__setattr__(
            self, "provider_libraries", tuple(sorted(set(self.provider_libraries)))
        )
        if self.selected_platforms is not None:
            if (
                not isinstance(self.selected_platforms, tuple)
                or not self.selected_platforms
            ):
                raise TypeError("selected_platforms must be a non-empty tuple or None.")
            for name in self.selected_platforms:
                canonical_identifier(name, "platform")
        for value in (self.wsl, self.backends_initialized, self.device_probed):
            if type(value) is not bool:
                raise TypeError("Observation flags must be booleans.")
        if not isinstance(self.devices, tuple) or any(
            not isinstance(device, OneapiDeviceFacts) for device in self.devices
        ):
            raise TypeError("devices must be a tuple of OneapiDeviceFacts.")
        for value in (self.runtime_platform, self.runtime_version, self.backend_error):
            if value is not None:
                canonical_identifier(value, "runtime metadata")
        if not self.device_probed and (
            self.devices
            or self.runtime_platform is not None
            or self.runtime_version is not None
            or self.backend_error is not None
        ):
            raise ValueError("Runtime observations require device_probed=True.")


@final
@dataclass(frozen=True, slots=True)
class OneapiEnvironmentReport:
    facts: OneapiEnvironmentFacts
    issues: tuple[OneapiEnvironmentIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.facts, OneapiEnvironmentFacts):
            raise TypeError("facts must be OneapiEnvironmentFacts.")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(issue, OneapiEnvironmentIssue) for issue in self.issues
        ):
            raise TypeError("issues must contain OneapiEnvironmentIssue values.")

    @property
    def successful(self) -> bool:
        """Whether assessed checks succeeded, not whether hardware is qualified."""
        return not any(issue.severity == "error" for issue in self.issues)


def _oneapi_device(device: OneapiDeviceFacts, /) -> bool:
    if device.platform.lower() == "oneapi":
        return True
    if device.platform.lower() != "gpu":
        return False
    return device.client_platform.lower() in ("oneapi", "sycl") or (
        device.runtime_version.lower().split()[0] in ("oneapi", "sycl")
    )


def _host_issues(facts: OneapiEnvironmentFacts, /) -> list[OneapiEnvironmentIssue]:
    issues: list[OneapiEnvironmentIssue] = []
    unsupported = (
        (facts.host_platform, facts.host_machine.lower()) != ("linux", "x86_64")
        or facts.wsl
        or "microsoft" in facts.host_release.lower()
    )
    unsupported |= bool(
        facts.python_implementation and facts.python_implementation != "CPython"
    )
    unsupported |= facts.python_version is not None and facts.python_version[:2] != (
        3,
        13,
    )
    unsupported |= facts.python_free_threaded is True
    if facts.libc_name:
        unsupported |= facts.libc_name != "glibc"
    if facts.libc_name == "glibc" and facts.libc_version:
        parts = facts.libc_version.split(".")
        unsupported |= tuple(int(part) for part in parts[:2]) < (2, 27)
    if unsupported:
        issues.append(
            OneapiEnvironmentIssue(
                "unsupported-host",
                "error",
                "Phydrax requires native Linux x86_64, standard-GIL CPython 3.13 and glibc >=2.27; WSL is unsupported. The vendor wheel's broader Python range does not expand Phydrax support.",
            )
        )
    elif not (
        facts.python_implementation
        and facts.python_version
        and facts.python_free_threaded is not None
        and facts.libc_name
        and facts.libc_version
    ):
        issues.append(
            OneapiEnvironmentIssue(
                "host-requirements-unknown",
                "error",
                "Python implementation/version/GIL-build or libc compatibility evidence is incomplete; explicit oneAPI admission requires complete host evidence.",
            )
        )
    return issues


def assess_oneapi_environment(
    facts: OneapiEnvironmentFacts, /
) -> OneapiEnvironmentReport:
    """Assess supplied evidence without loading a plugin, driver, or JAX client."""
    if not isinstance(facts, OneapiEnvironmentFacts):
        raise TypeError("facts must be OneapiEnvironmentFacts.")
    issues = _host_issues(facts)

    def add(
        code: OneapiIssueCode, message: str, severity: OneapiIssueSeverity = "error"
    ) -> None:
        issues.append(OneapiEnvironmentIssue(code, severity, message))

    versions = dict(facts.versions)
    for name, required in _REQUIRED_VERSIONS:
        found = versions.get(name)
        if found == required:
            continue
        if name.startswith("jax-oneapi"):
            code: OneapiIssueCode = (
                "no-oneapi-plugin" if found is None else "plugin-version-mismatch"
            )
        else:
            code = "jax-version-mismatch"
        add(code, f"phydrax[oneapi] requires {name}=={required}; found {found!r}.")
    for name, required in _RUNTIME_VERSIONS:
        if versions.get(name) != required:
            add(
                "runtime-dependency-mismatch",
                f"The with-oneapi runtime requires {name}=={required}; found {versions.get(name)!r}.",
            )
    conflicts = tuple(name for name in _CONFLICTING_PLUGINS if name in versions)
    if conflicts:
        add(
            "conflicting-plugins",
            "Use an isolated oneAPI environment without competing accelerator distributions: "
            + ", ".join(conflicts),
        )
    if facts.selected_platforms is not None and "oneapi" not in facts.selected_platforms:
        add(
            "platform-not-selected",
            "The explicit runtime selection excludes oneapi; CPU use is preserved.",
            "warning",
        )
    if facts.environment_overrides:
        add(
            "library-overrides",
            "SDK, loader, or device-selection overrides are present; wheel metadata does not establish which libraries or devices they select.",
            "warning",
        )
    add(
        "mixed-frontend-stack",
        "Experimental mixed frontend: JAX 0.12.0 with jaxlib and oneAPI plugins 0.11.2 is not Intel's documented matched 0.11.2 stack; Intel GPU qualification is pending.",
        "warning",
    )
    if facts.device_probed:
        if facts.backend_error is not None:
            add("backend-unavailable", facts.backend_error)
        elif not facts.devices:
            add("backend-unavailable", "No oneAPI devices were observed.")
        elif any(not _oneapi_device(device) for device in facts.devices):
            add(
                "runtime-identity-mismatch",
                "The explicit oneAPI probe returned a CPU or a runtime without oneAPI/SYCL identity.",
            )
        elif (facts.runtime_platform, facts.runtime_version) != (
            facts.devices[0].platform,
            facts.devices[0].runtime_version,
        ):
            add(
                "runtime-identity-mismatch",
                "Runtime summary does not match observed device/client evidence.",
            )
    return OneapiEnvironmentReport(facts, tuple(issues))


def collect_oneapi_environment() -> OneapiEnvironmentFacts:
    """Collect metadata and host facts; never import a provider or enumerate devices."""
    versions: list[tuple[str, str]] = []
    libraries: list[tuple[str, str]] = []
    names = (
        "phydrax",
        *(name for name, _ in _REQUIRED_VERSIONS),
        *(name for name, _ in _RUNTIME_VERSIONS),
        *_CONFLICTING_PLUGINS,
    )
    for name in names:
        try:
            distribution = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        versions.append((name, distribution.version))
        if name in ("jax-oneapi-plugin", "jax-oneapi-pjrt"):
            for file in distribution.files or ():
                if ".so" in file.name:
                    libraries.append(
                        (name, str(Path(str(distribution.locate_file(file))).resolve()))
                    )
    libc_name, libc_version = platform.libc_ver()
    release = platform.release()
    return OneapiEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        python_free_threaded=bool(sysconfig.get_config_var("Py_GIL_DISABLED")),
        libc_name=libc_name,
        libc_version=libc_version,
        host_release=release,
        wsl="microsoft" in release.lower()
        or "WSL_INTEROP" in os.environ
        or "WSL_DISTRO_NAME" in os.environ,
        versions=tuple(versions),
        selected_platforms=_jax_config.platforms(),
        environment_overrides=tuple(
            (name, os.environ[name]) for name in _OVERRIDE_KEYS if os.environ.get(name)
        ),
        provider_libraries=tuple(libraries),
        backends_initialized=_jax_config.backends_initialized(),
    )


def oneapi_environment(*, initialize_backend: bool = False) -> OneapiEnvironmentReport:
    """Inspect host evidence and, only by explicit request, probe ``oneapi``.

    Explicit CPU or other platform selection is never changed or bypassed.
    Success means assessed checks succeeded, not scientific/hardware qualification.
    """
    if type(initialize_backend) is not bool:
        raise TypeError("initialize_backend must be a boolean.")
    facts = collect_oneapi_environment()
    report = assess_oneapi_environment(facts)
    if not initialize_backend or not report.successful:
        return report
    if facts.selected_platforms is not None and "oneapi" not in facts.selected_platforms:
        return report
    try:
        devices = jax.devices("oneapi")
    except (RuntimeError, OSError, ImportError, ValueError) as error:
        return assess_oneapi_environment(
            replace(
                facts,
                device_probed=True,
                backends_initialized=_jax_config.backends_initialized(),
                backend_error=f"{type(error).__name__}: {str(error).strip()}",
            )
        )
    observed = tuple(
        OneapiDeviceFacts(
            platform=device.platform,
            client_platform=device.client.platform,
            runtime_version=device.client.platform_version,
            device_kind=device.device_kind,
            id=device.id,
            process_index=device.process_index,
        )
        for device in devices
    )
    return assess_oneapi_environment(
        replace(
            facts,
            device_probed=True,
            backends_initialized=_jax_config.backends_initialized(),
            devices=observed,
            runtime_platform=observed[0].platform if observed else None,
            runtime_version=observed[0].runtime_version if observed else None,
        )
    )


__all__ = [
    "OneapiDeviceFacts",
    "OneapiEnvironmentFacts",
    "OneapiEnvironmentIssue",
    "OneapiEnvironmentReport",
    "OneapiIssueCode",
    "OneapiIssueSeverity",
    "assess_oneapi_environment",
    "collect_oneapi_environment",
    "oneapi_environment",
]
