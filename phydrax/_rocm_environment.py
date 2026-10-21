#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Device-inert owned ROCm installation facts and opt-in client observation."""

from __future__ import annotations

import os
import platform
import re
import sys
import sysconfig
import traceback
from dataclasses import dataclass, replace
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from typing import final, Literal

import jax

from . import _jax_config
from ._validation import canonical_identifier
from .typing import parse


type RocmIssueCode = Literal[
    "unsupported-host",
    "jax-version-mismatch",
    "no-rocm-plugin",
    "conflicting-plugin-majors",
    "incomplete-plugin-pair",
    "plugin-version-mismatch",
    "selected-major-mismatch",
    "runtime-dependency-missing",
    "runtime-dependency-mismatch",
    "runtime-requirements-unknown",
    "platform-not-selected",
    "backend-unavailable",
    "runtime-identity-mismatch",
    "runtime-bundle-invalid",
]
type RocmIssueSeverity = Literal["error", "warning"]

_PLUGIN_DISTRIBUTIONS = ("jax-rocm10-plugin", "jax-rocm10-pjrt")
SDK_VERSION = "10.1.0"
ARCHITECTURES = ("gfx1250", "gfx942", "gfx950")
DISTRIBUTIONS = (
    "phydrax-rocm-runtime",
    "phydrax-rocm-core",
    "phydrax-rocm-libraries",
    "phydrax-rocm-device-gfx1250",
    "phydrax-rocm-device-gfx942",
    "phydrax-rocm-device-gfx950",
)


def _runtime_provider() -> ModuleType | None:
    """Cross the optional wheel boundary without importing its native payload."""
    try:
        metadata.distribution("phydrax-rocm-runtime")
    except metadata.PackageNotFoundError:
        return None
    return import_module("phydrax_rocm_runtime._loader")


def _bundle_loaded() -> bool:
    provider = sys.modules.get("phydrax_rocm_runtime._loader")
    return provider is not None and bool(provider.rocm_bundle_loaded())


def _major(value: int | None, /) -> None:
    if value is not None:
        if type(value) is not int:
            raise TypeError("selected_major must be an integer or None.")
        if value != 10:
            raise ValueError("selected_major must be 10.")


def _nonnegative(value: int, /) -> None:
    if type(value) is not int:
        raise TypeError("Device and Python identifiers must be integers.")
    if value < 0:
        raise ValueError("Device and Python identifiers must be nonnegative.")


@final
@dataclass(frozen=True, slots=True)
class RocmDeviceFacts:
    """Provider identity; neither raw ``gpu`` nor a device name identifies AMD."""

    platform: str
    client_platform: str
    platform_version: str
    device_kind: str
    id: int
    process_index: int

    def __post_init__(self) -> None:
        for value in (
            self.platform,
            self.client_platform,
            self.platform_version,
            self.device_kind,
        ):
            canonical_identifier(value, "device metadata")
        _nonnegative(self.id)
        _nonnegative(self.process_index)


@final
@dataclass(frozen=True, slots=True)
class RocmEnvironmentIssue:
    code: RocmIssueCode
    severity: RocmIssueSeverity
    message: str

    def __post_init__(self) -> None:
        parse(self.code, RocmIssueCode, "code")
        parse(self.severity, RocmIssueSeverity, "severity")
        canonical_identifier(self.message, "message")


@final
@dataclass(frozen=True, slots=True)
class RocmEnvironmentFacts:
    """Immutable observations; installed versions do not identify loaded libraries.

    ``userspace_version`` and ``userspace_root`` preserve external SDK evidence;
    only ``bundle_*`` fields identify Phydrax's owned userspace. Neither installed
    metadata nor a successful inspection establishes hardware qualification.
    """

    host_platform: str
    host_machine: str
    python_implementation: str
    python_version: tuple[int, int, int]
    python_free_threaded: bool = False
    versions: tuple[tuple[str, str], ...] = ()
    selected_major: int | None = None
    selected_platforms: tuple[str, ...] | None = None
    userspace_root: str | None = None
    userspace_version: str | None = None
    missing_runtime_dependencies: tuple[str, ...] = ()
    bundle_manifest: str | None = None
    bundle_version: str | None = None
    bundle_architectures: tuple[str, ...] = ()
    bundle_components: tuple[tuple[str, str, str], ...] = ()
    bundle_error: str | None = None
    bundle_loaded: bool = False
    environment_overrides: tuple[tuple[str, str], ...] = ()
    enable_x64: bool = False
    backends_initialized: bool = False
    distributed_initialized: bool = False
    device_probed: bool = False
    devices: tuple[RocmDeviceFacts, ...] = ()
    runtime_platform: str | None = None
    runtime_version: str | None = None
    backend_error: str | None = None

    def __post_init__(self) -> None:
        for value in (self.host_platform, self.host_machine, self.python_implementation):
            canonical_identifier(value, "host identity")
        if not isinstance(self.python_version, tuple) or len(self.python_version) != 3:
            raise TypeError("python_version must be a three-integer tuple.")
        for value in self.python_version:
            _nonnegative(value)
        _major(self.selected_major)
        for name, pairs in (
            ("versions", self.versions),
            ("environment_overrides", self.environment_overrides),
        ):
            if not isinstance(pairs, tuple):
                raise TypeError(f"{name} must be a tuple of text pairs.")
            for pair in pairs:
                if not isinstance(pair, tuple) or len(pair) != 2:
                    raise TypeError(f"{name} must contain text pairs.")
                canonical_identifier(pair[0], name)
                if not isinstance(pair[1], str):
                    raise TypeError(f"{name} values must be text.")
            if len({key for key, _ in pairs}) != len(pairs):
                raise ValueError(f"{name} keys must be unique.")
            object.__setattr__(self, name, tuple(sorted(pairs)))
        if self.selected_platforms is not None:
            if (
                not isinstance(self.selected_platforms, tuple)
                or not self.selected_platforms
            ):
                raise TypeError("selected_platforms must be a nonempty tuple or None.")
            for value in self.selected_platforms:
                canonical_identifier(value, "selected platform")
        if not isinstance(self.missing_runtime_dependencies, tuple):
            raise TypeError("missing_runtime_dependencies must be a tuple.")
        for value in self.missing_runtime_dependencies:
            canonical_identifier(value, "missing dependency")
        if not isinstance(self.bundle_architectures, tuple):
            raise TypeError("bundle_architectures must be a tuple.")
        for value in self.bundle_architectures:
            canonical_identifier(value, "bundle architecture")
        if not isinstance(self.bundle_components, tuple):
            raise TypeError("bundle_components must be a tuple.")
        for component in self.bundle_components:
            if not isinstance(component, tuple) or len(component) != 3:
                raise TypeError(
                    "bundle_components must contain distribution/version/digest triples."
                )
            for value in component:
                canonical_identifier(value, "bundle component")
        for value in (
            self.userspace_root,
            self.userspace_version,
            self.runtime_platform,
            self.runtime_version,
            self.backend_error,
            self.bundle_manifest,
            self.bundle_version,
            self.bundle_error,
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError("Optional runtime metadata must be text or None.")
        for value in (
            self.python_free_threaded,
            self.enable_x64,
            self.backends_initialized,
            self.distributed_initialized,
            self.device_probed,
            self.bundle_loaded,
        ):
            if type(value) is not bool:
                raise TypeError("Observation flags must be booleans.")
        if not isinstance(self.devices, tuple) or any(
            not isinstance(device, RocmDeviceFacts) for device in self.devices
        ):
            raise TypeError("devices must contain RocmDeviceFacts.")
        if not self.device_probed and (
            self.devices
            or self.runtime_platform is not None
            or self.runtime_version is not None
            or self.backend_error is not None
        ):
            raise ValueError("Runtime observations require device_probed=True.")


@final
@dataclass(frozen=True, slots=True)
class RocmEnvironmentReport:
    facts: RocmEnvironmentFacts
    issues: tuple[RocmEnvironmentIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.facts, RocmEnvironmentFacts):
            raise TypeError("facts must be RocmEnvironmentFacts.")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(issue, RocmEnvironmentIssue) for issue in self.issues
        ):
            raise TypeError("issues must contain RocmEnvironmentIssue values.")

    @property
    def successful(self) -> bool:
        """Whether represented checks passed, not hardware qualification."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def hardware_qualified(self) -> bool:
        return False


def _installed_majors(versions: dict[str, str], /) -> tuple[int, ...]:
    return tuple(
        major
        for major in (10,)
        if any(
            f"jax-rocm{major}-{component}" in versions for component in ("plugin", "pjrt")
        )
    )


def _assess_installation(
    facts: RocmEnvironmentFacts, issues: list[RocmEnvironmentIssue], /
) -> int | None:
    versions = dict(facts.versions)
    for name in ("jax", "jaxlib"):
        if versions.get(name) != "0.12.0":
            issues.append(
                RocmEnvironmentIssue(
                    "jax-version-mismatch",
                    "error",
                    f"ROCm requires {name}==0.12.0; found {versions.get(name)!r}.",
                )
            )
    majors = _installed_majors(versions)
    if not majors:
        issues.append(
            RocmEnvironmentIssue(
                "no-rocm-plugin",
                "error",
                "Install phydrax[rocm10].",
            )
        )
    if any(
        name.startswith("jax-rocm") and name not in _PLUGIN_DISTRIBUTIONS
        for name in versions
    ):
        issues.append(
            RocmEnvironmentIssue(
                "conflicting-plugin-majors",
                "error",
                "Only the ROCm 10 plugin family is supported.",
            )
        )
    selected = facts.selected_major
    if selected is None and len(majors) == 1:
        selected = majors[0]
    if selected is not None and majors != (selected,):
        issues.append(
            RocmEnvironmentIssue(
                "selected-major-mismatch",
                "error",
                f"Requested ROCm {selected}; installed plugin majors are {majors}.",
            )
        )
    for major in majors:
        for component in ("plugin", "pjrt"):
            name = f"jax-rocm{major}-{component}"
            found = versions.get(name)
            if found != "0.12.0":
                issues.append(
                    RocmEnvironmentIssue(
                        "incomplete-plugin-pair"
                        if found is None
                        else "plugin-version-mismatch",
                        "error",
                        f"ROCm {major} requires {name}==0.12.0; found {found!r}.",
                    )
                )
    return selected


def _assess_runtime(
    facts: RocmEnvironmentFacts, issues: list[RocmEnvironmentIssue], /
) -> None:
    if not facts.device_probed:
        return
    if facts.backend_error is not None or not facts.devices:
        issues.append(
            RocmEnvironmentIssue(
                "backend-unavailable",
                "error",
                facts.backend_error or "No ROCm devices were observed.",
            )
        )
        return
    for device in facts.devices:
        identity = re.match(
            r"rocm\s*[:=]?\s*\d+(?:\.|\b)",
            device.platform_version,
            re.IGNORECASE,
        )
        if (
            device.platform not in ("gpu", "rocm")
            or device.client_platform not in ("gpu", "rocm")
            or identity is None
        ):
            issues.append(
                RocmEnvironmentIssue(
                    "runtime-identity-mismatch",
                    "error",
                    "Every observed device must identify ROCm through its loaded client platform version; GPU display names are not AMD evidence.",
                )
            )
            return
    if (facts.runtime_platform, facts.runtime_version) != (
        facts.devices[0].client_platform,
        facts.devices[0].platform_version,
    ):
        issues.append(
            RocmEnvironmentIssue(
                "runtime-identity-mismatch",
                "error",
                "Runtime summary does not match the observed ROCm client.",
            )
        )


def _assess_userspace(
    facts: RocmEnvironmentFacts,
    selected: int | None,
    issues: list[RocmEnvironmentIssue],
    /,
) -> None:
    if facts.bundle_error is not None:
        issues.append(
            RocmEnvironmentIssue("runtime-bundle-invalid", "error", facts.bundle_error)
        )
    if selected is None:
        return
    versions = dict(facts.versions)
    for name in DISTRIBUTIONS:
        found = versions.get(name)
        if found != SDK_VERSION:
            issues.append(
                RocmEnvironmentIssue(
                    "runtime-dependency-missing"
                    if found is None
                    else "runtime-dependency-mismatch",
                    "error",
                    f"ROCm 10 requires owned {name}=={SDK_VERSION}; found {found!r}.",
                )
            )
    if facts.bundle_manifest is None or facts.bundle_version != SDK_VERSION:
        issues.append(
            RocmEnvironmentIssue(
                "runtime-dependency-missing",
                "error",
                "A validated owned ROCm runtime manifest is required; install phydrax[rocm10].",
            )
        )
    if facts.bundle_architectures != ARCHITECTURES:
        issues.append(
            RocmEnvironmentIssue(
                "runtime-bundle-invalid",
                "error",
                "The owned runtime must contain gfx1250, gfx942, and gfx950 architecture payloads.",
            )
        )


def assess_rocm_environment(facts: RocmEnvironmentFacts, /) -> RocmEnvironmentReport:
    """Pure policy assessment without filesystem, environment, or device access."""
    if not isinstance(facts, RocmEnvironmentFacts):
        raise TypeError("facts must be RocmEnvironmentFacts.")
    issues: list[RocmEnvironmentIssue] = []
    if (
        facts.host_platform != "linux"
        or facts.host_machine != "x86_64"
        or facts.python_implementation != "CPython"
        or facts.python_version[:2] != (3, 13)
        or facts.python_free_threaded
    ):
        issues.append(
            RocmEnvironmentIssue(
                "unsupported-host",
                "error",
                "ROCm requires Linux x86_64 and standard-GIL CPython 3.13.",
            )
        )
    selected = _assess_installation(facts, issues)
    _assess_userspace(facts, selected, issues)
    if not facts.device_probed:
        issues.append(
            RocmEnvironmentIssue(
                "runtime-requirements-unknown",
                "warning",
                "Host metadata cannot establish loader resolution, kernel-driver compatibility, device availability, or scientific qualification. The owned userspace is prepared automatically at ROCm backend use; a compatible driver and device permissions remain prerequisites.",
            )
        )
    if facts.selected_platforms is not None and "rocm" not in facts.selected_platforms:
        issues.append(
            RocmEnvironmentIssue(
                "platform-not-selected",
                "warning",
                "The explicit platform selection excludes rocm; inspection will not change it.",
            )
        )
    _assess_runtime(facts, issues)
    return RocmEnvironmentReport(facts, tuple(issues))


def _userspace_facts(
    versions: dict[str, str], /
) -> tuple[str | None, str | None, tuple[str, ...]]:
    # Explicit SDK roots override the default system SDK; wheel metadata is not
    # attributed to /opt/rocm, which may be an unrelated installation.
    provider = sys.modules.get("phydrax_rocm_runtime._loader")
    initial = provider.initial_rocm_environment() if provider is not None else None
    overrides = os.environ if initial is None else dict(initial)
    root = overrides.get("ROCM_PATH") or overrides.get("ROCM_HOME")
    if root is None and "rocm" in versions:
        return None, versions["rocm"], ()
    selected = Path(root) if root else Path("/opt/rocm")
    if not selected.is_dir():
        return root, None, (root,) if root is not None else ()
    version_file = selected / ".info" / "version"
    if version_file.is_file():
        version = version_file.read_text(encoding="utf-8").strip().split("-", 1)[0]
        return str(selected.resolve()), version or None, ()
    return str(selected.resolve()), None, ()


def collect_rocm_environment() -> RocmEnvironmentFacts:
    """Read owned and external SDK metadata without loading plugins or devices."""
    versions: dict[str, str] = {}
    for name in (
        "jax",
        "jaxlib",
        "rocm",
        "rocm-sdk-core",
        "rocm-sdk-libraries",
        "phydrax",
        *_PLUGIN_DISTRIBUTIONS,
        *DISTRIBUTIONS,
    ):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    for distribution in metadata.distributions():
        name = (distribution.metadata.get("Name") or "").lower().replace("_", "-")
        if name.startswith("jax-rocm"):
            versions[name] = distribution.version
    majors = _installed_majors(versions)
    selected = majors[0] if len(majors) == 1 else None
    root, version, missing = _userspace_facts(versions)
    bundle = None
    bundle_error = None
    try:
        provider = _runtime_provider()
        bundle = provider.inspect_rocm_bundle() if provider is not None else None
    except (RuntimeError, ImportError, OSError) as error:
        bundle_error = str(error)
    return RocmEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        python_implementation=platform.python_implementation(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        python_free_threaded=sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"),
        versions=tuple(versions.items()),
        selected_major=selected,
        selected_platforms=_jax_config.platforms(),
        userspace_root=root,
        userspace_version=version,
        missing_runtime_dependencies=missing,
        bundle_manifest=bundle.manifest_path if bundle else None,
        bundle_version=bundle.sdk_version if bundle else None,
        bundle_architectures=bundle.architectures if bundle else (),
        bundle_components=bundle.components if bundle else (),
        bundle_error=bundle_error,
        bundle_loaded=_bundle_loaded(),
        environment_overrides=tuple(
            (name, os.environ[name])
            for name in (
                "ROCM_PATH",
                "ROCM_HOME",
                "LD_LIBRARY_PATH",
                "LLVM_PATH",
                "HIP_DEVICE_LIB_PATH",
                "LD_PRELOAD",
                "HIP_VISIBLE_DEVICES",
                "ROCR_VISIBLE_DEVICES",
                "XLA_FLAGS",
            )
            if name in os.environ
        ),
        enable_x64=_jax_config.x64_enabled(),
        backends_initialized=_jax_config.backends_initialized(),
        distributed_initialized=_jax_config.distributed_initialized(),
    )


def rocm_environment(*, initialize_backend: bool = False) -> RocmEnvironmentReport:
    """Inspect one ROCm stack, optionally requesting exactly the ``rocm`` client.

    Never changes platform selection, initializes a distributed cluster, or
    falls back to CPU. Backend exception chains are preserved as diagnostic text.
    """
    if type(initialize_backend) is not bool:
        raise TypeError("initialize_backend must be a boolean.")
    facts = collect_rocm_environment()
    report = assess_rocm_environment(facts)
    if not initialize_backend or not report.successful:
        return report
    if facts.selected_platforms is not None and "rocm" not in facts.selected_platforms:
        return report
    try:
        provider = _runtime_provider()
        if provider is None:
            raise RuntimeError("Missing owned ROCm runtime; install phydrax[rocm10].")
        provider.activate_rocm_bundle()
        devices = tuple(jax.devices("rocm"))
        observed = tuple(
            RocmDeviceFacts(
                platform=device.platform,
                client_platform=device.client.platform,
                platform_version=device.client.platform_version,
                device_kind=device.device_kind,
                id=device.id,
                process_index=device.process_index,
            )
            for device in devices
        )
    except (RuntimeError, OSError) as error:
        return assess_rocm_environment(
            replace(
                facts,
                device_probed=True,
                backend_error="".join(traceback.format_exception(error)).strip(),
                backends_initialized=_jax_config.backends_initialized(),
                bundle_loaded=_bundle_loaded(),
            )
        )
    return assess_rocm_environment(
        replace(
            facts,
            device_probed=True,
            devices=observed,
            runtime_platform=observed[0].client_platform if observed else None,
            runtime_version=observed[0].platform_version if observed else None,
            backends_initialized=_jax_config.backends_initialized(),
            distributed_initialized=_jax_config.distributed_initialized(),
            bundle_loaded=_bundle_loaded(),
        )
    )


__all__ = [
    "RocmDeviceFacts",
    "RocmEnvironmentFacts",
    "RocmEnvironmentIssue",
    "RocmEnvironmentReport",
    "RocmIssueCode",
    "RocmIssueSeverity",
    "assess_rocm_environment",
    "collect_rocm_environment",
    "rocm_environment",
]
