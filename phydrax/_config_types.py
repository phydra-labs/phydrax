#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Immutable native configuration requests owned by `phydrax.config`.

Every field is a sparse request: `None` leaves the current setting untouched
rather than resetting it. The records hold host metadata only; scientific
precision, execution plans, and live runtime facts remain with their owners.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, Literal, TypeAlias

from . import _jax_config
from ._frozendict import frozendict
from ._validation import canonical_identifier
from .typing import parse


CPUCollectives: TypeAlias = Literal["gloo", "mpi"]
DTypePromotion: TypeAlias = Literal["standard", "strict"]
RankPromotion: TypeAlias = Literal["allow", "warn", "raise"]
TransferGuard: TypeAlias = Literal[
    "allow", "log", "disallow", "log_explicit", "disallow_explicit"
]
LibrarySetting: TypeAlias = Literal["meshcore", "metis"]
WorkerSetting: TypeAlias = Literal["mmg", "omega_h", "tioga", "vorocrust"]
EnvironmentMerge: TypeAlias = Literal["override", "defaults"]


def _optional_bool(value: object, name: str, /) -> None:
    if value is not None and not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool or None.")


def _optional_identifier(value: object, name: str, /) -> None:
    if value is not None:
        canonical_identifier(value, name)


def _exact_int(value: object, name: str, /, *, minimum: int) -> int:
    # Host metadata is a canonical Python `int`; bool and NumPy integers are refused.
    if type(value) is not int:
        raise TypeError(f"{name} must be an int.")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeBootstrap:
    """Distributed coordination consumed before any JAX backend initializes.

    Either every explicit coordinator field is given, or `cluster_detection_method`
    names a JAX launcher detection method, or `autodetect` delegates detection to
    JAX. Local device IDs are ordered: their order is the process's device order.
    """

    __config_kind__: ClassVar[str] = "phydrax.config.bootstrap"

    coordinator_address: str | None = None
    num_processes: int | None = None
    process_id: int | None = None
    local_device_ids: tuple[int, ...] | None = None
    cluster_detection_method: str | None = None
    autodetect: bool = False
    coordinator_bind_address: str | None = None
    heartbeat_timeout_seconds: int = 100
    initialization_timeout_seconds: int = 300

    def __post_init__(self) -> None:
        _optional_identifier(self.coordinator_address, "coordinator_address")
        _optional_identifier(self.cluster_detection_method, "cluster_detection_method")
        _optional_identifier(self.coordinator_bind_address, "coordinator_bind_address")
        if not isinstance(self.autodetect, bool):
            raise TypeError("autodetect must be a bool.")
        explicit = (
            self.coordinator_address is not None,
            self.num_processes is not None,
            self.process_id is not None,
        )
        if any(explicit) and not all(explicit):
            raise ValueError(
                "coordinator_address, num_processes, and process_id must be provided together"
            )
        selection_count = sum(
            (
                self.coordinator_address is not None,
                self.cluster_detection_method is not None,
                self.autodetect,
            )
        )
        if selection_count > 1:
            raise ValueError(
                "explicit coordinator configuration, cluster_detection_method, and autodetect are mutually exclusive"
            )
        if self.num_processes is not None:
            count = _exact_int(self.num_processes, "num_processes", minimum=1)
            if _exact_int(self.process_id, "process_id", minimum=0) >= count:
                raise ValueError("process_id must be in [0, num_processes)")
        if self.local_device_ids is not None:
            if not isinstance(self.local_device_ids, tuple):
                raise TypeError("local_device_ids must be a tuple of device IDs.")
            if not self.local_device_ids:
                raise ValueError("local_device_ids cannot be empty")
            for device_id in self.local_device_ids:
                _exact_int(device_id, "local_device_ids", minimum=0)
            if len(set(self.local_device_ids)) != len(self.local_device_ids):
                raise ValueError("local_device_ids must be unique")
        _exact_int(self.heartbeat_timeout_seconds, "heartbeat_timeout_seconds", minimum=1)
        _exact_int(
            self.initialization_timeout_seconds,
            "initialization_timeout_seconds",
            minimum=1,
        )

    @property
    def requests_distributed_initialization(self) -> bool:
        return (
            self.coordinator_address is not None
            or self.cluster_detection_method is not None
            or self.autodetect
        )


@dataclass(frozen=True, slots=True)
class LaunchEnvironmentConfig:
    """External launch inputs that take effect only in a process not yet started.

    They are projected into a child environment by `phydrax.config.environment`;
    a running process cannot apply them.
    """

    __config_kind__: ClassVar[str] = "phydrax.config.launch_environment"

    xla_flags: str | None = None
    preallocate: bool | None = None

    def __post_init__(self) -> None:
        if self.xla_flags is not None and not isinstance(self.xla_flags, str):
            raise TypeError("xla_flags must be text or None.")
        _optional_bool(self.preallocate, "preallocate")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Process-wide runtime settings.

    Platform selection, CPU device count, and CPU collectives bind when a JAX
    backend initializes; distributed coordination binds once per process. x64
    enablement affects operations staged after it is applied, never existing
    arrays.
    """

    __config_kind__: ClassVar[str] = "phydrax.config.runtime"

    enable_x64: bool | None = None
    platforms: tuple[str, ...] | None = None
    cpu_device_count: int | None = None
    cpu_collectives_implementation: CPUCollectives | None = None
    bootstrap: RuntimeBootstrap | None = None
    launch_environment: LaunchEnvironmentConfig | None = None

    def __post_init__(self) -> None:
        _optional_bool(self.enable_x64, "enable_x64")
        if self.platforms is not None:
            if not isinstance(self.platforms, tuple) or not self.platforms:
                raise TypeError("platforms must be a non-empty tuple of platform names.")
            for platform in self.platforms:
                if "," in canonical_identifier(platform, "platforms"):
                    raise ValueError("Platform names cannot contain commas.")
            if len(set(self.platforms)) != len(self.platforms):
                raise ValueError("platforms must be unique.")
        if self.cpu_device_count is not None:
            _exact_int(self.cpu_device_count, "cpu_device_count", minimum=1)
        if self.cpu_collectives_implementation is not None:
            object.__setattr__(
                self,
                "cpu_collectives_implementation",
                parse(
                    self.cpu_collectives_implementation,
                    CPUCollectives,
                    "cpu_collectives_implementation",
                ),
            )
        if self.bootstrap is not None and not isinstance(
            self.bootstrap, RuntimeBootstrap
        ):
            raise TypeError("bootstrap must be a RuntimeBootstrap or None.")
        if self.launch_environment is not None and not isinstance(
            self.launch_environment, LaunchEnvironmentConfig
        ):
            raise TypeError("launch_environment must be a LaunchEnvironmentConfig.")


@dataclass(frozen=True, slots=True)
class NumericsConfig:
    """Thread-local numerical settings entered by `phydrax.config.context`.

    `matmul_precision` is any precision or dot-algorithm name JAX accepts;
    `None` inherits, so an explicit `"default"` remains a distinct request.
    """

    __config_kind__: ClassVar[str] = "phydrax.config.numerics"

    enable_x64: bool | None = None
    matmul_precision: str | None = None
    dtype_promotion: DTypePromotion | None = None
    rank_promotion: RankPromotion | None = None
    transfer_guard_device_to_host: TransferGuard | None = None
    disable_jit: bool | None = None

    def __post_init__(self) -> None:
        _optional_bool(self.enable_x64, "enable_x64")
        _optional_bool(self.disable_jit, "disable_jit")
        if self.matmul_precision is not None:
            _jax_config.validate_matmul_precision(
                canonical_identifier(self.matmul_precision, "matmul_precision")
            )
        # Selectors are stored as their canonical declared literals.
        if self.dtype_promotion is not None:
            object.__setattr__(
                self,
                "dtype_promotion",
                parse(self.dtype_promotion, DTypePromotion, "dtype_promotion"),
            )
        if self.rank_promotion is not None:
            object.__setattr__(
                self,
                "rank_promotion",
                parse(self.rank_promotion, RankPromotion, "rank_promotion"),
            )
        if self.transfer_guard_device_to_host is not None:
            object.__setattr__(
                self,
                "transfer_guard_device_to_host",
                parse(
                    self.transfer_guard_device_to_host,
                    TransferGuard,
                    "transfer_guard_device_to_host",
                ),
            )


@dataclass(frozen=True, slots=True)
class LoggingConfig:
    """Process-wide emission of canonical Phydrax events; sinks stay with logging."""

    __config_kind__: ClassVar[str] = "phydrax.config.logging"

    enabled: bool

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a bool.")


def _library_overrides(
    values: Mapping[LibrarySetting, str | None], /
) -> frozendict[LibrarySetting, str | None]:
    if not isinstance(values, Mapping):
        raise TypeError("library_paths must be a mapping.")
    canonical: dict[LibrarySetting, str | None] = {}
    for key, value in values.items():
        if value is not None and not isinstance(value, str):
            raise TypeError(f"library_paths[{key!r}] must be text or None.")
        canonical[parse(key, LibrarySetting, "library_paths key")] = value
    return frozendict(canonical)


def _worker_overrides(
    values: Mapping[WorkerSetting, str | None], /
) -> frozendict[WorkerSetting, str | None]:
    if not isinstance(values, Mapping):
        raise TypeError("workers must be a mapping.")
    canonical: dict[WorkerSetting, str | None] = {}
    for key, value in values.items():
        if value is not None and not isinstance(value, str):
            raise TypeError(f"workers[{key!r}] must be text or None.")
        canonical[parse(key, WorkerSetting, "workers key")] = value
    return frozendict(canonical)


@dataclass(frozen=True, slots=True)
class ProviderConfig:
    """Overrides of native library and worker selection.

    An absent key inherits the declared environment variable; `None` ignores it
    and selects the provider's own fallback; text is used as given, including
    the empty string. Paths are inputs: the loaded library or worker remains
    identified by its provider.
    """

    __config_kind__: ClassVar[str] = "phydrax.config.provider"

    library_paths: Mapping[LibrarySetting, str | None] = frozendict()
    workers: Mapping[WorkerSetting, str | None] = frozendict()

    def __post_init__(self) -> None:
        # The canonical immutable mapping is fixed before the record is shared.
        object.__setattr__(self, "library_paths", _library_overrides(self.library_paths))
        object.__setattr__(self, "workers", _worker_overrides(self.workers))


class ConfigurationApplicationError(RuntimeError):
    """A configuration request could not take effect.

    `setting` names the refused or failed field; `applied` lists the fields of
    the same request that had already taken effect, which are not rolled back.
    """

    def __init__(self, setting: str, applied: tuple[str, ...], message: str, /) -> None:
        super().__init__(f"{setting}: {message}")
        self.setting = setting
        self.applied = applied


__all__ = [
    "CPUCollectives",
    "ConfigurationApplicationError",
    "DTypePromotion",
    "EnvironmentMerge",
    "LaunchEnvironmentConfig",
    "LibrarySetting",
    "LoggingConfig",
    "NumericsConfig",
    "ProviderConfig",
    "RankPromotion",
    "RuntimeBootstrap",
    "RuntimeConfig",
    "TransferGuard",
    "WorkerSetting",
]
