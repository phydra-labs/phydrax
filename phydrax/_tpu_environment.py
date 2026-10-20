#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host-only upstream TPU evidence and explicitly requested runtime observation."""

from __future__ import annotations

import base64
import hashlib
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


type TpuIssueCode = Literal[
    "unsupported-host",
    "jax-version-mismatch",
    "libtpu-missing",
    "libtpu-version-mismatch",
    "library-invalid",
    "library-override-unattributed",
    "platform-not-selected",
    "backend-unavailable",
    "runtime-identity-mismatch",
    "precision-limitations",
]
type TpuIssueSeverity = Literal["error", "warning"]


@final
@dataclass(frozen=True, slots=True)
class TpuEnvironmentIssue:
    code: TpuIssueCode
    severity: TpuIssueSeverity
    message: str

    def __post_init__(self) -> None:
        parse(self.code, TpuIssueCode, "code")
        parse(self.severity, TpuIssueSeverity, "severity")
        canonical_identifier(self.message, "message")


def _nonnegative(value: int, name: str, /) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative.")


@final
@dataclass(frozen=True, slots=True)
class TpuDeviceFacts:
    """Observed provider device identity, without retaining a live JAX client."""

    id: int
    process_index: int
    platform: str
    device_kind: str
    local_hardware_id: int | None = None

    def __post_init__(self) -> None:
        _nonnegative(self.id, "id")
        _nonnegative(self.process_index, "process_index")
        canonical_identifier(self.platform, "platform")
        canonical_identifier(self.device_kind, "device_kind")
        if self.local_hardware_id is not None:
            _nonnegative(self.local_hardware_id, "local_hardware_id")


@final
@dataclass(frozen=True, slots=True)
class TpuEnvironmentFacts:
    """Immutable evidence, not a claim of device availability or qualification.

    Distribution versions describe installed metadata. The selected library has
    its own path and digest; an unrelated override does not inherit that version.
    Counts and runtime identity remain unknown until an explicit probe.
    """

    host_platform: str
    host_machine: str
    python_implementation: str
    python_version: tuple[int, int, int]
    libc_name: str
    libc_version: str
    python_free_threaded: bool = False
    host_release: str = ""
    wsl: bool = False
    versions: tuple[tuple[str, str], ...] = ()
    selected_platforms: tuple[str, ...] | None = None
    library_override: str | None = None
    distribution_library_path: str | None = None
    library_path: str | None = None
    library_sha256: str | None = None
    library_from_distribution: bool = False
    installation_error: str | None = None
    enable_x64: bool = False
    matmul_precision: str | None = None
    xla_flags: str | None = None
    libtpu_init_args: str | None = None
    backends_initialized: bool = False
    distributed_initialized: bool = False
    device_probed: bool = False
    devices: tuple[TpuDeviceFacts, ...] = ()
    runtime_platform: str | None = None
    runtime_version: str | None = None
    mapped_library_paths: tuple[str, ...] = ()
    local_device_count: int | None = None
    global_device_count: int | None = None
    process_count: int | None = None
    process_index: int | None = None
    backend_error: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("host_platform", self.host_platform),
            ("host_machine", self.host_machine),
            ("python_implementation", self.python_implementation),
        ):
            canonical_identifier(value, name)
        if not isinstance(self.python_version, tuple) or len(self.python_version) != 3:
            raise TypeError("python_version must be a three-integer tuple.")
        for value in self.python_version:
            _nonnegative(value, "python_version")
        for value in (self.libc_name, self.libc_version, self.host_release):
            if not isinstance(value, str):
                raise TypeError("Host ABI identity must be text.")
        if not isinstance(self.versions, tuple):
            raise TypeError("versions must be a tuple of distribution/version pairs.")
        for entry in self.versions:
            if not isinstance(entry, tuple) or len(entry) != 2:
                raise TypeError("versions must contain distribution/version pairs.")
            for value in entry:
                canonical_identifier(value, "distribution identity")
        if len({name for name, _ in self.versions}) != len(self.versions):
            raise ValueError("TPU distribution names must be unique.")
        object.__setattr__(self, "versions", tuple(sorted(self.versions)))
        if self.selected_platforms is not None:
            if (
                not isinstance(self.selected_platforms, tuple)
                or not self.selected_platforms
            ):
                raise TypeError("selected_platforms must be a non-empty tuple or None.")
            for value in self.selected_platforms:
                canonical_identifier(value, "selected platform")
        for value in (
            self.library_override,
            self.distribution_library_path,
            self.library_path,
            self.library_sha256,
            self.installation_error,
            self.matmul_precision,
            self.xla_flags,
            self.libtpu_init_args,
            self.runtime_platform,
            self.runtime_version,
            self.backend_error,
        ):
            if value is not None and not isinstance(value, str):
                raise TypeError("TPU optional metadata must be text or None.")
        if self.library_sha256 is not None and (
            len(self.library_sha256) != 64
            or any(
                character not in "0123456789abcdef" for character in self.library_sha256
            )
        ):
            raise ValueError("library_sha256 must be a lowercase SHA-256 digest.")
        for value in (
            self.library_from_distribution,
            self.enable_x64,
            self.backends_initialized,
            self.distributed_initialized,
            self.device_probed,
            self.python_free_threaded,
            self.wsl,
        ):
            if type(value) is not bool:
                raise TypeError("TPU observation flags must be booleans.")
        self._validate_observations()

    def _validate_observations(self) -> None:
        """Bind device and topology observations to the explicit probe lifecycle."""
        if not isinstance(self.devices, tuple) or any(
            not isinstance(device, TpuDeviceFacts) for device in self.devices
        ):
            raise TypeError("devices must be a tuple of TpuDeviceFacts.")
        if not isinstance(self.mapped_library_paths, tuple):
            raise TypeError("mapped_library_paths must be a tuple.")
        for value in self.mapped_library_paths:
            canonical_identifier(value, "mapped library path")
        object.__setattr__(
            self, "mapped_library_paths", tuple(sorted(set(self.mapped_library_paths)))
        )
        for value in (
            self.local_device_count,
            self.global_device_count,
            self.process_count,
            self.process_index,
        ):
            if value is not None:
                _nonnegative(value, "runtime count")
        if not self.device_probed and (
            self.devices
            or self.runtime_platform is not None
            or self.runtime_version is not None
            or self.mapped_library_paths
            or self.local_device_count is not None
            or self.global_device_count is not None
            or self.process_count is not None
            or self.process_index is not None
            or self.backend_error is not None
        ):
            raise ValueError("Runtime observations require device_probed=True.")


@final
@dataclass(frozen=True, slots=True)
class TpuEnvironmentReport:
    facts: TpuEnvironmentFacts
    issues: tuple[TpuEnvironmentIssue, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.facts, TpuEnvironmentFacts):
            raise TypeError("facts must be TpuEnvironmentFacts.")
        if not isinstance(self.issues, tuple) or any(
            not isinstance(issue, TpuEnvironmentIssue) for issue in self.issues
        ):
            raise TypeError("issues must contain TpuEnvironmentIssue values.")

    @property
    def successful(self) -> bool:
        """Whether assessed checks passed, not whether hardware was qualified."""
        return not any(issue.severity == "error" for issue in self.issues)

    @property
    def hardware_qualified(self) -> bool:
        """Diagnostics never replace the installed-wheel scientific qualifier."""
        return False


def _supported_abi(facts: TpuEnvironmentFacts, /) -> bool:
    parts = facts.libc_version.split(".")
    return (
        facts.host_platform == "linux"
        and facts.host_machine == "x86_64"
        and not facts.wsl
        and "microsoft" not in facts.host_release.lower()
        and "wsl" not in facts.host_release.lower()
        and facts.python_implementation == "CPython"
        and facts.python_version[:2] == (3, 13)
        and not facts.python_free_threaded
        and facts.libc_name == "glibc"
        and len(parts) >= 2
        and all(part.isdecimal() for part in parts)
        and tuple(int(part) for part in parts[:2]) >= (2, 31)
    )


def _assess_runtime(
    facts: TpuEnvironmentFacts, issues: list[TpuEnvironmentIssue], /
) -> None:
    if not facts.device_probed:
        return
    if facts.backend_error is not None or not facts.devices:
        issues.append(
            TpuEnvironmentIssue(
                "backend-unavailable",
                "error",
                facts.backend_error or "No TPU devices were observed.",
            )
        )
        return
    if facts.runtime_platform != "tpu" or any(
        device.platform != "tpu" for device in facts.devices
    ):
        issues.append(
            TpuEnvironmentIssue(
                "runtime-identity-mismatch",
                "error",
                "An explicit TPU probe did not observe the TPU platform.",
            )
        )
    if not facts.runtime_version or facts.library_path not in facts.mapped_library_paths:
        issues.append(
            TpuEnvironmentIssue(
                "runtime-identity-mismatch",
                "error",
                "The observed client must identify its runtime and map the selected libtpu file.",
            )
        )
    counts = (facts.local_device_count, facts.global_device_count, facts.process_count)
    if (
        any(value is None or value <= 0 for value in counts)
        or facts.global_device_count != len(facts.devices)
        or facts.process_index is None
        or facts.process_count is None
        or facts.process_index >= facts.process_count
        or facts.local_device_count
        != sum(device.process_index == facts.process_index for device in facts.devices)
        or any(device.process_index >= facts.process_count for device in facts.devices)
        or (facts.process_count > 1 and not facts.distributed_initialized)
    ):
        issues.append(
            TpuEnvironmentIssue(
                "runtime-identity-mismatch",
                "error",
                "The observed TPU process and device topology is inconsistent.",
            )
        )


def assess_tpu_environment(facts: TpuEnvironmentFacts, /) -> TpuEnvironmentReport:
    """Assess host or observed facts without imports, environment reads, or clients."""
    if not isinstance(facts, TpuEnvironmentFacts):
        raise TypeError("facts must be TpuEnvironmentFacts.")
    issues: list[TpuEnvironmentIssue] = []

    def add(
        code: TpuIssueCode, message: str, severity: TpuIssueSeverity = "error"
    ) -> None:
        issues.append(TpuEnvironmentIssue(code, severity, message))

    if not _supported_abi(facts):
        add(
            "unsupported-host",
            "phydrax[tpu] requires standard-GIL CPython 3.13 on native Linux x86_64 with glibc >= 2.31; WSL is unsupported.",
        )
    versions = dict(facts.versions)
    for name in ("jax", "jaxlib"):
        if versions.get(name) != "0.12.0":
            add(
                "jax-version-mismatch",
                f"phydrax[tpu] requires {name}==0.12.0; found {versions.get(name)!r}.",
            )
    if "libtpu" not in versions:
        add(
            "libtpu-missing",
            "Install phydrax[tpu] to obtain the upstream libtpu distribution.",
        )
    elif versions["libtpu"] != "0.0.50":
        add(
            "libtpu-version-mismatch",
            f"phydrax[tpu] requires libtpu==0.0.50; found {versions['libtpu']!r}.",
        )
    if facts.installation_error or not facts.library_path or not facts.library_sha256:
        add(
            "library-invalid",
            facts.installation_error
            or "No attributable libtpu shared library was found.",
        )
    elif not facts.library_from_distribution:
        add(
            "library-override-unattributed",
            "TPU_LIBRARY_PATH selects bytes not attributed to the installed libtpu wheel; its version is unknown.",
        )
    if facts.selected_platforms is not None and "tpu" not in facts.selected_platforms:
        add(
            "platform-not-selected",
            "The explicit runtime selection does not include tpu; diagnostics will not change it.",
            "warning",
        )
    _assess_runtime(facts, issues)
    add(
        "precision-limitations",
        "x64 enablement is not TPU FP64 qualification. On TPU, highest float32 contractions use six BF16 passes, not native IEEE float32 multiplication; qualify the scientific route and tolerances on the target hardware.",
        "warning",
    )
    return TpuEnvironmentReport(facts, tuple(issues))


def _sha256(path: Path, /) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _library_facts(facts: TpuEnvironmentFacts, /) -> TpuEnvironmentFacts:
    distribution = metadata.distribution("libtpu")
    candidates = tuple(
        entry for entry in distribution.files or () if entry.name == "libtpu.so"
    )
    if len(candidates) != 1:
        raise ValueError(
            "The libtpu distribution must identify exactly one libtpu.so in RECORD."
        )
    entry = candidates[0]
    distribution_path = Path(str(distribution.locate_file(entry))).resolve(strict=True)
    distribution_digest = _sha256(distribution_path)
    if entry.hash is None or entry.hash.mode != "sha256":
        raise ValueError("libtpu.so has no SHA-256 distribution RECORD identity.")
    recorded_digest = (
        base64.urlsafe_b64encode(bytes.fromhex(distribution_digest)).decode().rstrip("=")
    )
    if recorded_digest != entry.hash.value:
        raise ValueError("libtpu.so does not match the installed distribution RECORD.")
    selected = (
        distribution_path if facts.library_path is None else Path(facts.library_path)
    )
    digest = (
        facts.library_sha256 if facts.library_sha256 is not None else distribution_digest
    )
    return replace(
        facts,
        distribution_library_path=str(distribution_path),
        library_path=str(selected),
        library_sha256=digest,
        library_from_distribution=digest == distribution_digest,
    )


def collect_tpu_environment() -> TpuEnvironmentFacts:
    """Read host facts without importing libtpu, loading a library, or discovering devices.

    Importing JAX itself may already perform upstream Cloud TPU startup work; this
    collector promises no additional client creation, not absence of JAX effects.
    """
    versions: list[tuple[str, str]] = []
    for name in ("jax", "jaxlib", "libtpu", "phydrax"):
        try:
            versions.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            continue
    libc_name, libc_version = platform.libc_ver()
    release = platform.release()
    facts = TpuEnvironmentFacts(
        host_platform=sys.platform,
        host_machine=platform.machine(),
        host_release=release,
        wsl="microsoft" in release.lower()
        or "wsl" in release.lower()
        or "WSL_INTEROP" in os.environ
        or "WSL_DISTRO_NAME" in os.environ,
        python_implementation=platform.python_implementation(),
        python_version=(
            sys.version_info.major,
            sys.version_info.minor,
            sys.version_info.micro,
        ),
        libc_name=libc_name,
        python_free_threaded=sysconfig.get_config_var("Py_GIL_DISABLED") in (1, "1"),
        libc_version=libc_version,
        versions=tuple(versions),
        selected_platforms=_jax_config.platforms(),
        library_override=os.environ.get("TPU_LIBRARY_PATH"),
        enable_x64=_jax_config.x64_enabled(),
        matmul_precision=_jax_config.matmul_precision(),
        xla_flags=os.environ.get(_jax_config.XLA_FLAGS_ENVIRONMENT),
        libtpu_init_args=os.environ.get("LIBTPU_INIT_ARGS"),
        backends_initialized=_jax_config.backends_initialized(),
        distributed_initialized=_jax_config.distributed_initialized(),
    )
    try:
        if facts.library_override is not None:
            selected = Path(facts.library_override).resolve(strict=True)
            facts = replace(
                facts, library_path=str(selected), library_sha256=_sha256(selected)
            )
        return _library_facts(facts)
    except (OSError, ValueError, metadata.PackageNotFoundError) as error:
        return replace(facts, installation_error=f"libtpu inspection failed: {error}")


def _mapped_libtpu_paths(selected_path: str | None, /) -> tuple[str, ...]:
    paths: set[str] = set()
    for line in Path("/proc/self/maps").read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith("/"):
            path = Path(fields[5])
            resolved = str(path.resolve())
            if path.name == "libtpu.so" or resolved == selected_path:
                paths.add(resolved)
    return tuple(sorted(paths))


def tpu_environment(*, initialize_backend: bool = False) -> TpuEnvironmentReport:
    """Inspect TPU facts; optionally initialize the explicitly requested TPU client.

    Callers must apply RuntimeBootstrap before probing a multi-host runtime. This
    function never initializes a distributed cluster or changes platform selection.
    """
    if type(initialize_backend) is not bool:
        raise TypeError("initialize_backend must be a boolean.")
    facts = collect_tpu_environment()
    assessed = assess_tpu_environment(facts)
    if not initialize_backend or not assessed.successful:
        return assessed
    if facts.selected_platforms is not None and "tpu" not in facts.selected_platforms:
        return assessed
    try:
        devices = tuple(jax.devices("tpu"))
        local_devices = tuple(jax.local_devices(backend="tpu"))
        process_count = jax.process_count("tpu")
        process_index = jax.process_index("tpu")
        mapped_library_paths = _mapped_libtpu_paths(facts.library_path)
    except (RuntimeError, OSError) as error:
        return assess_tpu_environment(
            replace(
                facts,
                device_probed=True,
                backend_error=str(error),
                backends_initialized=_jax_config.backends_initialized(),
                distributed_initialized=_jax_config.distributed_initialized(),
            )
        )
    return assess_tpu_environment(
        replace(
            facts,
            device_probed=True,
            backends_initialized=_jax_config.backends_initialized(),
            distributed_initialized=_jax_config.distributed_initialized(),
            devices=tuple(
                TpuDeviceFacts(
                    id=device.id,
                    process_index=device.process_index,
                    platform=device.platform,
                    device_kind=device.device_kind,
                    local_hardware_id=device.local_hardware_id,
                )
                for device in devices
            ),
            runtime_platform=devices[0].platform if devices else None,
            runtime_version=devices[0].client.platform_version if devices else None,
            mapped_library_paths=mapped_library_paths,
            local_device_count=len(local_devices),
            global_device_count=len(devices),
            process_count=process_count,
            process_index=process_index,
        )
    )


__all__ = [
    "TpuDeviceFacts",
    "TpuEnvironmentFacts",
    "TpuEnvironmentIssue",
    "TpuEnvironmentReport",
    "TpuIssueCode",
    "TpuIssueSeverity",
    "assess_tpu_environment",
    "collect_tpu_environment",
    "tpu_environment",
]
