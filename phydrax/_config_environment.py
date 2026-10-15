#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fixed native environment catalog, pure decoding, and child projection."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Final

from . import _jax_config
from ._config_types import (
    CPUCollectives,
    EnvironmentMerge,
    LaunchEnvironmentConfig,
    LibrarySetting,
    NumericsConfig,
    ProviderConfig,
    RuntimeBootstrap,
    RuntimeConfig,
    WorkerSetting,
)
from .typing import parse


COORDINATOR_ADDRESS: Final = "PHYDRAX_COORDINATOR_ADDRESS"
COORDINATOR_BIND_ADDRESS: Final = "PHYDRAX_COORDINATOR_BIND_ADDRESS"
NUM_PROCESSES: Final = "PHYDRAX_NUM_PROCESSES"
PROCESS_ID: Final = "PHYDRAX_PROCESS_ID"
LOCAL_DEVICE_IDS: Final = "PHYDRAX_LOCAL_DEVICE_IDS"
CPU_COLLECTIVES: Final = "PHYDRAX_CPU_COLLECTIVES"
HEARTBEAT_TIMEOUT_SECONDS: Final = "PHYDRAX_HEARTBEAT_TIMEOUT_SECONDS"
INITIALIZATION_TIMEOUT_SECONDS: Final = "PHYDRAX_INITIALIZATION_TIMEOUT_SECONDS"

LIBRARY_ENVIRONMENT: Final[Mapping[LibrarySetting, str]] = MappingProxyType(
    {
        "meshcore": "PHYDRAX_MESHCORE_LIBRARY",
        "metis": "PHYDRAX_METIS_LIBRARY",
        "cuda_cudss": "PHYDRAX_CUDA_CUDSS_LIBRARY",
        "cuda_amgx": "PHYDRAX_CUDA_AMGX_LIBRARY",
        "cuda_quda": "PHYDRAX_CUDA_QUDA_LIBRARY",
    }
)
WORKER_ENVIRONMENT: Final[Mapping[WorkerSetting, str]] = MappingProxyType(
    {
        "mmg": "PHYDRAX_MMG_WORKER",
        "omega_h": "PHYDRAX_OMEGA_H_WORKER",
        "tioga": "PHYDRAX_TIOGA_WORKER",
        "vorocrust": "PHYDRAX_VOROCRUST_WORKER",
    }
)
_CLUSTER_SIZE_ENVIRONMENTS: Final = (
    "SLURM_NTASKS",
    "OMPI_COMM_WORLD_SIZE",
    "PMI_SIZE",
    "PMIX_SIZE",
)
_BOOTSTRAP_TRIO: Final = (COORDINATOR_ADDRESS, NUM_PROCESSES, PROCESS_ID)
_BOOTSTRAP_FIELDS: Final = {
    "coordinator_address": COORDINATOR_ADDRESS,
    "num_processes": NUM_PROCESSES,
    "process_id": PROCESS_ID,
    "local_device_ids": LOCAL_DEVICE_IDS,
    "coordinator_bind_address": COORDINATOR_BIND_ADDRESS,
    "heartbeat_timeout_seconds": HEARTBEAT_TIMEOUT_SECONDS,
    "initialization_timeout_seconds": INITIALIZATION_TIMEOUT_SECONDS,
}
_BOOTSTRAP_ENVIRONMENT: Final = frozenset(_BOOTSTRAP_FIELDS.values())
_NATIVE_RECORDS: Final = (
    RuntimeConfig,
    RuntimeBootstrap,
    LaunchEnvironmentConfig,
    NumericsConfig,
    ProviderConfig,
)


@dataclass(frozen=True, slots=True)
class NativeEnvironmentInput:
    path: tuple[str, ...]
    value: object
    environment_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class BootstrapTimeoutOverrides:
    heartbeat_timeout_seconds: int | None = None
    initialization_timeout_seconds: int | None = None


def is_native_record(record_type: type, /) -> bool:
    return record_type in _NATIVE_RECORDS


def _environment_text(environment: Mapping[str, str], key: str, /) -> str:
    value = environment[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be text.")
    return value


def _decimal(text: str, key: str, /, *, positive: bool = False) -> int:
    if not isinstance(text, str):
        raise TypeError(f"{key} must be text.")
    stripped = text.strip()
    if not stripped.isdecimal():
        raise ValueError(f"{key} must be a nonnegative decimal integer; got {text!r}.")
    value = int(stripped)
    if positive and value == 0:
        raise ValueError(f"{key} must be positive.")
    return value


def bootstrap_timeout_overrides(
    environment: Mapping[str, str], /
) -> BootstrapTimeoutOverrides:
    return BootstrapTimeoutOverrides(
        heartbeat_timeout_seconds=(
            _decimal(
                environment[HEARTBEAT_TIMEOUT_SECONDS],
                HEARTBEAT_TIMEOUT_SECONDS,
                positive=True,
            )
            if HEARTBEAT_TIMEOUT_SECONDS in environment
            else None
        ),
        initialization_timeout_seconds=(
            _decimal(
                environment[INITIALIZATION_TIMEOUT_SECONDS],
                INITIALIZATION_TIMEOUT_SECONDS,
                positive=True,
            )
            if INITIALIZATION_TIMEOUT_SECONDS in environment
            else None
        ),
    )


def _bootstrap_inputs(environment: Mapping[str, str], /) -> dict[str, object]:
    bootstrap = _bootstrap(environment)
    values: dict[str, object] = {
        "coordinator_address": bootstrap.coordinator_address,
        "num_processes": bootstrap.num_processes,
        "process_id": bootstrap.process_id,
        "local_device_ids": bootstrap.local_device_ids,
        "coordinator_bind_address": bootstrap.coordinator_bind_address,
        "heartbeat_timeout_seconds": bootstrap.heartbeat_timeout_seconds,
        "initialization_timeout_seconds": bootstrap.initialization_timeout_seconds,
    }
    return {
        field: value
        for field, value in values.items()
        if _BOOTSTRAP_FIELDS[field] in environment
    }


def _bootstrap(environment: Mapping[str, str], /) -> RuntimeBootstrap:
    present = tuple(key in environment for key in _BOOTSTRAP_TRIO)
    if any(present) and not all(present):
        raise ValueError(
            f"{COORDINATOR_ADDRESS}, {NUM_PROCESSES}, and {PROCESS_ID} must be set together"
        )
    timeouts = bootstrap_timeout_overrides(environment)
    defaults = RuntimeBootstrap()
    local = (
        _environment_text(environment, LOCAL_DEVICE_IDS)
        if LOCAL_DEVICE_IDS in environment
        else None
    )
    return RuntimeBootstrap(
        coordinator_address=(
            _environment_text(environment, COORDINATOR_ADDRESS)
            if COORDINATOR_ADDRESS in environment
            else None
        ),
        num_processes=(
            _decimal(environment[NUM_PROCESSES], NUM_PROCESSES)
            if NUM_PROCESSES in environment
            else None
        ),
        process_id=(
            _decimal(environment[PROCESS_ID], PROCESS_ID)
            if PROCESS_ID in environment
            else None
        ),
        local_device_ids=(
            tuple(_decimal(item, LOCAL_DEVICE_IDS) for item in local.split(","))
            if local is not None
            else None
        ),
        coordinator_bind_address=(
            _environment_text(environment, COORDINATOR_BIND_ADDRESS)
            if COORDINATOR_BIND_ADDRESS in environment
            else None
        ),
        heartbeat_timeout_seconds=(
            defaults.heartbeat_timeout_seconds
            if timeouts.heartbeat_timeout_seconds is None
            else timeouts.heartbeat_timeout_seconds
        ),
        initialization_timeout_seconds=(
            defaults.initialization_timeout_seconds
            if timeouts.initialization_timeout_seconds is None
            else timeouts.initialization_timeout_seconds
        ),
    )


def _construct_native(
    record_type: type, inputs: Mapping[str, tuple[object, tuple[str, ...]]], /
) -> object:
    return record_type(**{field: value for field, (value, _) in inputs.items()})


def native_environment_inputs(
    record_type: type, environment: Mapping[str, str], /
) -> tuple[NativeEnvironmentInput, ...]:
    if not is_native_record(record_type):
        raise TypeError("record_type must be an exact native configuration record.")
    inputs: dict[str, tuple[object, tuple[str, ...]]] = {}
    if record_type is RuntimeBootstrap:
        if not any(key in environment for key in _BOOTSTRAP_TRIO):
            return ()
        values = _bootstrap_inputs(environment)
        inputs.update(
            (field, (value, (_BOOTSTRAP_FIELDS[field],)))
            for field, value in values.items()
        )
    elif record_type is RuntimeConfig:
        inputs.update(_jax_config.runtime_environment_inputs(environment))
        if CPU_COLLECTIVES in environment:
            inputs["cpu_collectives_implementation"] = (
                parse(environment[CPU_COLLECTIVES], CPUCollectives, CPU_COLLECTIVES),
                (CPU_COLLECTIVES,),
            )
        if any(key in environment for key in _BOOTSTRAP_TRIO):
            inputs["bootstrap"] = (
                _bootstrap(environment),
                tuple(sorted(_BOOTSTRAP_ENVIRONMENT.intersection(environment))),
            )
        launch = _jax_config.launch_environment_inputs(environment)
        if launch:
            inputs["launch_environment"] = (
                _construct_native(LaunchEnvironmentConfig, launch),
                tuple(sorted({key for _, keys in launch.values() for key in keys})),
            )
    elif record_type is LaunchEnvironmentConfig:
        inputs.update(_jax_config.launch_environment_inputs(environment))
    elif record_type is NumericsConfig:
        inputs.update(_jax_config.numerics_environment_inputs(environment))
    elif record_type is ProviderConfig:
        libraries = {
            field: _environment_text(environment, key)
            for field, key in LIBRARY_ENVIRONMENT.items()
            if key in environment
        }
        workers = {
            field: _environment_text(environment, key)
            for field, key in WORKER_ENVIRONMENT.items()
            if key in environment
        }
        provider = ProviderConfig(library_paths=libraries, workers=workers)
        if libraries:
            inputs["library_paths"] = (
                provider.library_paths,
                tuple(sorted(LIBRARY_ENVIRONMENT[field] for field in libraries)),
            )
        if workers:
            inputs["workers"] = (
                provider.workers,
                tuple(sorted(WORKER_ENVIRONMENT[field] for field in workers)),
            )
    if record_type in (RuntimeConfig, LaunchEnvironmentConfig, NumericsConfig):
        _construct_native(record_type, inputs)
    return tuple(
        NativeEnvironmentInput((field,), value, keys)
        for field, (value, keys) in sorted(inputs.items())
    )


def validate_native_origin(
    record_type: type,
    path: tuple[str, ...],
    environment_keys: tuple[str, ...],
    value: object,
    /,
) -> None:
    if not is_native_record(record_type):
        raise ValueError("Origin record is not native.")
    if (
        len(path) != 1
        or not environment_keys
        or tuple(sorted(set(environment_keys))) != environment_keys
    ):
        raise ValueError("Invalid native environment origin.")
    field = path[0]
    keys = frozenset(environment_keys)
    valid = False
    if record_type is RuntimeBootstrap:
        valid = field in _BOOTSTRAP_FIELDS and keys == {_BOOTSTRAP_FIELDS[field]}
    elif record_type is ProviderConfig:
        catalog = (
            LIBRARY_ENVIRONMENT
            if field == "library_paths"
            else WORKER_ENVIRONMENT
            if field == "workers"
            else {}
        )
        valid = (
            isinstance(value, Mapping)
            and bool(value)
            and all(item in catalog for item in value)
            and keys == {catalog[item] for item in value}
        )
    elif record_type is RuntimeConfig and field == "bootstrap":
        valid = (
            isinstance(value, RuntimeBootstrap)
            and value.coordinator_address is not None
            and value.num_processes is not None
            and value.process_id is not None
            and set(_BOOTSTRAP_TRIO) <= keys <= _BOOTSTRAP_ENVIRONMENT
        )
        if valid and isinstance(value, RuntimeBootstrap):
            defaults = RuntimeBootstrap()
            valid = (
                (LOCAL_DEVICE_IDS in keys) == (value.local_device_ids is not None)
                and (COORDINATOR_BIND_ADDRESS in keys)
                == (value.coordinator_bind_address is not None)
                and (
                    HEARTBEAT_TIMEOUT_SECONDS in keys
                    or value.heartbeat_timeout_seconds
                    == defaults.heartbeat_timeout_seconds
                )
                and (
                    INITIALIZATION_TIMEOUT_SECONDS in keys
                    or value.initialization_timeout_seconds
                    == defaults.initialization_timeout_seconds
                )
            )
    elif record_type is RuntimeConfig and field == "cpu_collectives_implementation":
        valid = keys == {CPU_COLLECTIVES}
    elif record_type is RuntimeConfig and field == "launch_environment":
        valid = isinstance(
            value, LaunchEnvironmentConfig
        ) and _jax_config.valid_launch_environment_origin(
            environment_keys, value.xla_flags is not None, value.preallocate is not None
        )
    else:
        catalog = (
            _jax_config.RUNTIME_ENVIRONMENT_KEYS
            if record_type is RuntimeConfig
            else _jax_config.NUMERICS_ENVIRONMENT_KEYS
            if record_type is NumericsConfig
            else _jax_config.LAUNCH_ENVIRONMENT_KEYS
        )
        valid = environment_keys in catalog.get(field, ())
    if not valid:
        raise ValueError(
            f"Invalid native environment keys for {record_type.__name__}.{field}."
        )


def bootstrap_from_environment(
    environment: Mapping[str, str], /
) -> RuntimeBootstrap | None:
    if any(key in environment for key in _BOOTSTRAP_TRIO):
        return _bootstrap(environment)
    for key in _CLUSTER_SIZE_ENVIRONMENTS:
        if key in environment and _decimal(environment[key], key) > 1:
            return replace(_bootstrap(environment), autodetect=True)
    return None


def startup_runtime(environment: Mapping[str, str], /) -> RuntimeConfig:
    inputs = _jax_config.runtime_environment_inputs(environment)
    if CPU_COLLECTIVES in environment:
        inputs["cpu_collectives_implementation"] = (
            parse(environment[CPU_COLLECTIVES], CPUCollectives, CPU_COLLECTIVES),
            (CPU_COLLECTIVES,),
        )
    runtime = _construct_native(RuntimeConfig, inputs)
    if not isinstance(runtime, RuntimeConfig):
        raise TypeError("The native runtime constructor returned an invalid record.")
    return replace(
        runtime,
        enable_x64=True if runtime.enable_x64 is None else runtime.enable_x64,
        bootstrap=bootstrap_from_environment(environment),
    )


def runtime_environment(
    settings: RuntimeConfig, base: Mapping[str, str] | None, mode: EnvironmentMerge, /
) -> dict[str, str]:
    if not isinstance(settings, RuntimeConfig):
        raise TypeError("settings must be a RuntimeConfig.")
    mode = parse(mode, EnvironmentMerge, "mode")
    original = dict(os.environ if base is None else base)
    result = original.copy()
    launch = settings.launch_environment
    _jax_config.project_environment(
        result,
        original,
        mode == "override",
        enable_x64=settings.enable_x64,
        platforms=settings.platforms,
        cpu_device_count=settings.cpu_device_count,
        xla_flags=None if launch is None else launch.xla_flags,
        preallocate=None if launch is None else launch.preallocate,
    )
    if settings.cpu_collectives_implementation is not None and (
        mode == "override" or CPU_COLLECTIVES not in original
    ):
        result[CPU_COLLECTIVES] = settings.cpu_collectives_implementation
    bootstrap = settings.bootstrap
    if bootstrap is not None:
        if (
            bootstrap.coordinator_address is None
            or bootstrap.num_processes is None
            or bootstrap.process_id is None
        ):
            raise ValueError(
                "Only an explicit coordinator, process count, and process ID can be projected into a launch environment."
            )
        if mode == "override" or _BOOTSTRAP_ENVIRONMENT.isdisjoint(original):
            for key in _BOOTSTRAP_ENVIRONMENT:
                result.pop(key, None)
            result.update(
                {
                    COORDINATOR_ADDRESS: bootstrap.coordinator_address,
                    NUM_PROCESSES: str(bootstrap.num_processes),
                    PROCESS_ID: str(bootstrap.process_id),
                    HEARTBEAT_TIMEOUT_SECONDS: str(bootstrap.heartbeat_timeout_seconds),
                    INITIALIZATION_TIMEOUT_SECONDS: str(
                        bootstrap.initialization_timeout_seconds
                    ),
                }
            )
            if bootstrap.local_device_ids is not None:
                result[LOCAL_DEVICE_IDS] = ",".join(
                    str(item) for item in bootstrap.local_device_ids
                )
            if bootstrap.coordinator_bind_address is not None:
                result[COORDINATOR_BIND_ADDRESS] = bootstrap.coordinator_bind_address
    return result


__all__ = [
    "COORDINATOR_ADDRESS",
    "COORDINATOR_BIND_ADDRESS",
    "CPU_COLLECTIVES",
    "HEARTBEAT_TIMEOUT_SECONDS",
    "INITIALIZATION_TIMEOUT_SECONDS",
    "LIBRARY_ENVIRONMENT",
    "LOCAL_DEVICE_IDS",
    "NUM_PROCESSES",
    "PROCESS_ID",
    "WORKER_ENVIRONMENT",
    "BootstrapTimeoutOverrides",
    "NativeEnvironmentInput",
    "bootstrap_from_environment",
    "bootstrap_timeout_overrides",
    "is_native_record",
    "native_environment_inputs",
    "runtime_environment",
    "startup_runtime",
    "validate_native_origin",
]
