#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only Metal installation evidence and explicit runtime observation."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import dataclass, replace
from importlib import metadata
from pathlib import Path
from typing import final, Literal

import jax

from . import _jax_config
from ._validation import canonical_identifier
from .typing import parse


type MetalIssueCode = Literal[
    "unsupported-host",
    "companion-missing",
    "companion-invalid",
    "companion-version-mismatch",
    "jax-version-mismatch",
    "conflicting-plugins",
    "platform-not-selected",
    "backend-unavailable",
    "runtime-identity-mismatch",
]
type MetalIssueSeverity = Literal["error", "warning"]


@final
@dataclass(frozen=True, slots=True)
class MetalEnvironmentIssue:
    code: MetalIssueCode
    severity: MetalIssueSeverity
    message: str

    def __post_init__(self) -> None:
        parse(self.code, MetalIssueCode, "code")
        parse(self.severity, MetalIssueSeverity, "severity")
        canonical_identifier(self.message, "message")


@final
@dataclass(frozen=True, slots=True)
class MetalEnvironmentFacts:
    """Collected evidence, not a capability or scientific qualification claim."""

    host_platform: str
    host_machine: str
    macos_version: str
    versions: tuple[tuple[str, str], ...] = ()
    competing_plugins: tuple[str, ...] = ()
    selected_platforms: tuple[str, ...] | None = None
    build_id: str | None = None
    source_id: str | None = None
    companion_platform: str | None = None
    required_jax: str | None = None
    required_jaxlib: str | None = None
    minimum_macos: str | None = None
    installation_error: str | None = None
    backends_initialized: bool = False
    device_probed: bool = False
    runtime_platform: str | None = None
    runtime_version: str | None = None
    device_kind: str | None = None
    backend_error: str | None = None

    def __post_init__(self) -> None:
        canonical_identifier(self.host_platform, "host_platform")
        canonical_identifier(self.host_machine, "host_machine")
        if not isinstance(self.macos_version, str):
            raise TypeError("macos_version must be text.")
        if not isinstance(self.versions, tuple):
            raise TypeError("versions must be a tuple of distribution/version pairs.")
        for entry in self.versions:
            if not isinstance(entry, tuple):
                raise TypeError("Each distribution/version pair must be a tuple.")
            if len(entry) != 2:
                raise ValueError("Each distribution/version pair must have two entries.")
        if not isinstance(self.competing_plugins, tuple):
            raise TypeError("competing_plugins must be a tuple.")
        for name in self.competing_plugins:
            canonical_identifier(name, "competing plugin")
        for value in (
            self.build_id,
            self.source_id,
            self.companion_platform,
            self.required_jax,
            self.required_jaxlib,
            self.minimum_macos,
            self.installation_error,
            self.runtime_platform,
            self.runtime_version,
            self.device_kind,
            self.backend_error,
        ):
            if value is not None:
                canonical_identifier(value, "Metal metadata")
        if self.host_platform == "darwin":
            _os_release(self.macos_version)
        if self.minimum_macos is not None:
            _os_release(self.minimum_macos)
        for name, version in self.versions:
            canonical_identifier(name, "distribution")
            canonical_identifier(version, "version")
        if len({name for name, _ in self.versions}) != len(self.versions):
            raise ValueError("Metal distribution names must be unique.")
        object.__setattr__(self, "versions", tuple(sorted(self.versions)))
        object.__setattr__(
            self, "competing_plugins", tuple(sorted(set(self.competing_plugins)))
        )
        if self.selected_platforms is not None:
            if (
                not isinstance(self.selected_platforms, tuple)
                or not self.selected_platforms
            ):
                raise TypeError("selected_platforms must be a non-empty tuple or None.")
            for name in self.selected_platforms:
                canonical_identifier(name, "platform")
        for value in (self.backends_initialized, self.device_probed):
            if type(value) is not bool:
                raise TypeError("Metal observation flags must be booleans.")


@final
@dataclass(frozen=True, slots=True)
class MetalEnvironmentReport:
    facts: MetalEnvironmentFacts
    issues: tuple[MetalEnvironmentIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.facts, MetalEnvironmentFacts):
            raise TypeError("facts must be MetalEnvironmentFacts.")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(issue, MetalEnvironmentIssue) for issue in self.issues
        ):
            raise TypeError("issues must contain MetalEnvironmentIssue values.")

    @property
    def successful(self) -> bool:
        """Whether assessed checks succeeded; unprobed hardware remains unprobed."""
        return not any(issue.severity == "error" for issue in self.issues)


def _os_release(value: str, /) -> tuple[int, ...]:
    parts = value.split(".")
    if not 1 <= len(parts) <= 3 or any(not part.isdecimal() for part in parts):
        raise ValueError(f"Invalid macOS release {value!r}.")
    return (*tuple(int(part) for part in parts), *(0 for _ in range(3 - len(parts))))


def assess_metal_environment(facts: MetalEnvironmentFacts, /) -> MetalEnvironmentReport:
    """Assess immutable evidence without importing a plugin or probing devices."""
    if not isinstance(facts, MetalEnvironmentFacts):
        raise TypeError("facts must be MetalEnvironmentFacts.")
    issues: list[MetalEnvironmentIssue] = []

    def add(
        code: MetalIssueCode, message: str, severity: MetalIssueSeverity = "error"
    ) -> None:
        issues.append(MetalEnvironmentIssue(code, severity, message))

    versions = dict(facts.versions)
    if (facts.host_platform, facts.host_machine) != ("darwin", "arm64"):
        add("unsupported-host", "Phydrax Metal requires macOS on Apple Silicon.")
    elif facts.minimum_macos is not None and (
        _os_release(facts.macos_version) < _os_release(facts.minimum_macos)
    ):
        add(
            "unsupported-host",
            f"This Metal build requires macOS {facts.minimum_macos} or newer.",
        )
    if "phydrax-metal" not in versions:
        add(
            "companion-missing",
            "Install phydrax[metal] to obtain the matching Metal companion.",
        )
    elif versions.get("phydrax") != versions["phydrax-metal"]:
        add(
            "companion-version-mismatch",
            "Phydrax and phydrax-metal must have the same release version.",
        )
    if facts.installation_error is not None:
        add("companion-invalid", facts.installation_error)
    elif "phydrax-metal" in versions and (
        not facts.build_id
        or not facts.source_id
        or facts.companion_platform != "mps"
        or facts.required_jax is None
        or facts.required_jaxlib is None
        or facts.minimum_macos is None
    ):
        add("companion-invalid", "The companion does not identify an owned MPS build.")
    for name, required in (
        ("jax", facts.required_jax),
        ("jaxlib", facts.required_jaxlib),
    ):
        if required is not None and versions.get(name) != required:
            add(
                "jax-version-mismatch",
                f"The companion requires {name}=={required}; found {versions.get(name)!r}.",
            )
    if facts.competing_plugins:
        add(
            "conflicting-plugins",
            "Remove competing Apple PJRT plugins: " + ", ".join(facts.competing_plugins),
        )
    if facts.selected_platforms is not None and "mps" not in facts.selected_platforms:
        add(
            "platform-not-selected",
            "The explicit runtime selection does not include mps.",
            "warning",
        )
    if facts.device_probed:
        if facts.backend_error is not None or facts.runtime_platform is None:
            add(
                "backend-unavailable",
                facts.backend_error or "No MPS device was observed.",
            )
        elif facts.runtime_platform.lower() != "mps":
            add(
                "runtime-identity-mismatch",
                f"Expected mps, observed {facts.runtime_platform!r}.",
            )
        elif facts.runtime_version != (
            f"phydrax-metal {versions.get('phydrax-metal')} {facts.source_id}"
        ):
            add(
                "runtime-identity-mismatch",
                "Loaded MPS source identity differs from the companion manifest.",
            )
    return MetalEnvironmentReport(facts, tuple(issues))


def _distribution_versions() -> tuple[tuple[str, str], ...]:
    result: list[tuple[str, str]] = []
    for name in ("phydrax", "phydrax-metal", "jax", "jaxlib"):
        try:
            result.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    return tuple(result)


def _competing_plugins() -> tuple[str, ...]:
    names: set[str] = set()
    for entry in metadata.entry_points(group="jax_plugins"):
        distribution = entry.dist
        if entry.name.lower() in ("mps", "metal"):
            if distribution is None:
                names.add(f"unattributed:{entry.name}")
            elif distribution.name.lower().replace("_", "-") != "phydrax-metal":
                names.add(distribution.name)
    for name in ("jax-mps", "jax-metal", "jax-metallib", "metaljax"):
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        names.add(name)
    return tuple(sorted(names))


def _build_facts(facts: MetalEnvironmentFacts, /) -> MetalEnvironmentFacts:
    if "phydrax-metal" not in dict(facts.versions):
        return facts
    distribution = metadata.distribution("phydrax-metal")
    root = Path(str(distribution.locate_file("phydrax_metal"))).resolve()
    record = json.loads((root / "build.json").read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise TypeError("Metal build manifest must be an object.")
    fields = (
        "version",
        "platform",
        "jax_version",
        "jaxlib_version",
        "minimum_macos",
        "build_id",
        "source_id",
        "library",
    )
    for field in fields:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise ValueError(f"Metal build manifest requires a non-empty {field}.")
    content = {name: value for name, value in record.items() if name != "build_id"}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(encoded).hexdigest() != record["build_id"]:
        raise ValueError("Metal build manifest identity is invalid.")
    _os_release(record["minimum_macos"])
    if record["version"] != dict(facts.versions)["phydrax-metal"]:
        raise ValueError("Metal build manifest and distribution versions disagree.")
    assets = record.get("assets")
    if not isinstance(assets, dict) or record["library"] not in assets:
        raise ValueError("Metal build manifest must identify the library in its assets.")
    for relative, expected in assets.items():
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise TypeError("Metal asset paths and hashes must be text.")
        asset = (root / relative).resolve()
        if not asset.is_relative_to(root) or not asset.is_file():
            raise ValueError(
                f"Metal runtime asset is absent or outside the package: {relative}."
            )
        with asset.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"Metal runtime asset hash mismatch: {relative}.")
    return replace(
        facts,
        build_id=record["build_id"],
        companion_platform=record["platform"],
        source_id=record["source_id"],
        required_jax=record["jax_version"],
        required_jaxlib=record["jaxlib_version"],
        minimum_macos=record["minimum_macos"],
    )


def collect_metal_environment() -> MetalEnvironmentFacts:
    """Collect installation evidence without loading a native library or client."""
    facts = MetalEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        macos_version=platform.mac_ver()[0],
        versions=_distribution_versions(),
        competing_plugins=_competing_plugins(),
        selected_platforms=_jax_config.platforms(),
        backends_initialized=_jax_config.backends_initialized(),
    )
    try:
        return _build_facts(facts)
    except (OSError, TypeError, ValueError, metadata.PackageNotFoundError) as error:
        return replace(
            facts, installation_error=f"Metal companion inspection failed: {error}"
        )


def metal_environment(*, initialize_backend: bool = True) -> MetalEnvironmentReport:
    """Inspect the installation and optionally probe the explicitly selected MPS device."""
    if type(initialize_backend) is not bool:
        raise TypeError("initialize_backend must be a boolean.")
    facts = collect_metal_environment()
    assessed = assess_metal_environment(facts)
    if not initialize_backend or not assessed.successful:
        return assessed
    if facts.selected_platforms is not None and "mps" not in facts.selected_platforms:
        return assessed
    try:
        devices = jax.devices("mps")
    except RuntimeError as error:
        return assess_metal_environment(
            replace(facts, device_probed=True, backend_error=str(error))
        )
    if not devices:
        return assess_metal_environment(
            replace(facts, device_probed=True, backend_error="No MPS devices returned.")
        )
    device = devices[0]
    return assess_metal_environment(
        replace(
            facts,
            backends_initialized=True,
            device_probed=True,
            runtime_platform=device.platform,
            runtime_version=device.client.platform_version,
            device_kind=device.device_kind,
        )
    )


__all__ = [
    "MetalEnvironmentFacts",
    "MetalEnvironmentIssue",
    "MetalEnvironmentReport",
    "MetalIssueCode",
    "MetalIssueSeverity",
    "assess_metal_environment",
    "metal_environment",
]
