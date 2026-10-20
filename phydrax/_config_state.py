#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Process configuration state, its application, and observation.

The state holds host metadata only: the merged runtime request, the logging
emission flag, and provider selection overrides. Application is validated in
full before any effect; an effect that then fails reports the fields of the
same request that already took effect.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Final

from . import _jax_config
from ._config_environment import (
    LIBRARY_ENVIRONMENT,
    runtime_environment,
    startup_runtime,
    WORKER_ENVIRONMENT,
)
from ._config_resolution import encode_snapshot_value
from ._config_types import (
    ConfigurationApplicationError,
    EnvironmentMerge,
    LibrarySetting,
    LoggingConfig,
    NumericsConfig,
    ProviderConfig,
    RuntimeConfig,
    WorkerSetting,
)
from ._execution_bootstrap import coordinate
from ._frozendict import frozendict
from .typing import parse


@dataclass(slots=True)
class _ProcessState:
    requested: RuntimeConfig
    logging_enabled: bool
    libraries: dict[LibrarySetting, str | None]
    workers: dict[WorkerSetting, str | None]
    started: bool


_LOCK: Final = threading.RLock()
_STATE: Final = _ProcessState(RuntimeConfig(), False, {}, {}, False)


@dataclass(frozen=True, slots=True)
class ConfigurationSnapshot:
    """Effective configuration observed without initializing any backend.

    `requested` is the merged runtime request applied in this process; the other
    runtime fields are the effective values, including active scoped settings.
    `None` means automatic selection, never a guessed hardware value. Provider
    inputs are the configured or declared-environment values, not loaded
    providers.
    """

    requested: RuntimeConfig
    enable_x64: bool
    platforms: tuple[str, ...] | None
    cpu_device_count: int | None
    cpu_collectives_implementation: str | None
    matmul_precision: str | None
    dtype_promotion: str
    rank_promotion: str
    transfer_guard_device_to_host: str | None
    jit_disabled: bool
    default_prng_implementation: str
    logging_enabled: bool
    library_paths: frozendict[LibrarySetting, str]
    workers: frozendict[WorkerSetting, str]
    backends_initialized: bool
    distributed_initialized: bool

    def to_record(self) -> dict[str, object]:
        record = encode_snapshot_value(self)
        if not isinstance(record, dict):
            raise TypeError("A configuration snapshot must encode as a mapping.")
        return record


def _merged(previous: RuntimeConfig, update: RuntimeConfig, /) -> RuntimeConfig:
    def chosen[T](new: T | None, old: T | None, /) -> T | None:
        return old if new is None else new

    return RuntimeConfig(
        enable_x64=chosen(update.enable_x64, previous.enable_x64),
        platforms=chosen(update.platforms, previous.platforms),
        cpu_device_count=chosen(update.cpu_device_count, previous.cpu_device_count),
        cpu_collectives_implementation=chosen(
            update.cpu_collectives_implementation,
            previous.cpu_collectives_implementation,
        ),
        bootstrap=chosen(update.bootstrap, previous.bootstrap),
    )


def _admit_accelerators(platforms: tuple[str, ...] | None, /) -> None:
    """Validate selected installations without discovering devices or applying effects."""
    for platform in platforms or ():
        match platform:
            case "mps":
                from ._metal_environment import (
                    assess_metal_environment,
                    collect_metal_environment,
                )

                issues = assess_metal_environment(collect_metal_environment()).issues
                messages = tuple(
                    issue.message for issue in issues if issue.severity == "error"
                )
            case "oneapi":
                from ._oneapi_environment import (
                    assess_oneapi_environment,
                    collect_oneapi_environment,
                )

                oneapi_issues = assess_oneapi_environment(
                    collect_oneapi_environment()
                ).issues
                messages = tuple(
                    issue.message for issue in oneapi_issues if issue.severity == "error"
                )
            case "tpu":
                from ._tpu_environment import (
                    assess_tpu_environment,
                    collect_tpu_environment,
                )

                tpu_issues = assess_tpu_environment(collect_tpu_environment()).issues
                messages = tuple(
                    issue.message for issue in tpu_issues if issue.severity == "error"
                )
            case _:
                continue
        if messages:
            raise ConfigurationApplicationError("platforms", (), "; ".join(messages))


def _admit(settings: RuntimeConfig, /) -> None:
    if settings.launch_environment is not None:
        raise ConfigurationApplicationError(
            "launch_environment",
            (),
            "launch settings take effect only in a process that has not started; "
            "project them with phydrax.config.environment().",
        )
    backends = _jax_config.backends_initialized()
    bound = (
        ("platforms", settings.platforms, _jax_config.platforms()),
        ("cpu_device_count", settings.cpu_device_count, _jax_config.cpu_device_count()),
        (
            "cpu_collectives_implementation",
            settings.cpu_collectives_implementation,
            _jax_config.cpu_collectives_implementation(),
        ),
    )
    for name, requested, effective in bound:
        if backends and requested is not None and requested != effective:
            raise ConfigurationApplicationError(
                name,
                (),
                f"a JAX backend is already initialized with {effective!r}; "
                f"{requested!r} binds only at backend initialization.",
            )
    _admit_accelerators(settings.platforms)
    bootstrap = settings.bootstrap
    if bootstrap is not None and bootstrap.requests_distributed_initialization:
        if _jax_config.distributed_initialized():
            raise ConfigurationApplicationError(
                "bootstrap", (), "the JAX distributed runtime is already initialized."
            )
        if backends:
            raise ConfigurationApplicationError(
                "bootstrap",
                (),
                "distributed coordination must precede JAX backend initialization.",
            )


def _applications(
    settings: RuntimeConfig, /
) -> list[tuple[str, RuntimeConfig, Callable[[], None] | None]]:
    steps: list[tuple[str, RuntimeConfig, Callable[[], None] | None]] = []
    collectives = settings.cpu_collectives_implementation
    if collectives is not None:
        steps.append(
            (
                "cpu_collectives_implementation",
                RuntimeConfig(cpu_collectives_implementation=collectives),
                (lambda: _jax_config.set_cpu_collectives_implementation(collectives))
                if collectives != _jax_config.cpu_collectives_implementation()
                else None,
            )
        )
    platforms = settings.platforms
    if platforms is not None:
        steps.append(
            (
                "platforms",
                RuntimeConfig(platforms=platforms),
                (lambda: _jax_config.set_platforms(platforms))
                if platforms != _jax_config.platforms()
                else None,
            )
        )
    count = settings.cpu_device_count
    if count is not None:
        steps.append(
            (
                "cpu_device_count",
                RuntimeConfig(cpu_device_count=count),
                (lambda: _jax_config.set_cpu_device_count(count))
                if count != _jax_config.cpu_device_count()
                else None,
            )
        )
    enable_x64 = settings.enable_x64
    if enable_x64 is not None:
        steps.append(
            (
                "enable_x64",
                RuntimeConfig(enable_x64=enable_x64),
                (lambda: _jax_config.set_x64(enable_x64))
                if enable_x64 != _jax_config.process_x64_enabled()
                else None,
            )
        )
    bootstrap = settings.bootstrap
    if bootstrap is not None:
        steps.append(
            (
                "bootstrap",
                RuntimeConfig(bootstrap=bootstrap),
                (lambda: coordinate(bootstrap))
                if bootstrap.requests_distributed_initialization
                else None,
            )
        )
    return steps


def _configure_runtime(settings: RuntimeConfig, /) -> None:
    with _LOCK:
        _admit(settings)
        applied: list[str] = []
        for name, request, apply in _applications(settings):
            next_requested = _merged(_STATE.requested, request)
            if apply is not None:
                try:
                    apply()
                except (RuntimeError, ValueError) as error:
                    raise ConfigurationApplicationError(
                        name, tuple(applied), str(error)
                    ) from error
                applied.append(name)
            _STATE.requested = next_requested


def _configure_logging(settings: LoggingConfig, /) -> None:
    # The logging transport imports this owner; it is loaded only when applied.
    from . import logging as phydrax_logging

    with _LOCK:
        _STATE.logging_enabled = settings.enabled
        phydrax_logging._synchronize_emission()


def configure(settings: RuntimeConfig | LoggingConfig | ProviderConfig, /) -> None:
    """Apply process-wide runtime, logging, or provider settings.

    Runtime fields that bind at backend initialization are refused, unchanged,
    once a backend exists; `ConfigurationApplicationError` names the field.
    """
    match settings:
        case RuntimeConfig():
            _configure_runtime(settings)
        case LoggingConfig():
            _configure_logging(settings)
        case ProviderConfig():
            with _LOCK:
                _STATE.libraries.update(settings.library_paths)
                _STATE.workers.update(settings.workers)
        case _:
            raise TypeError(
                "configure accepts RuntimeConfig, LoggingConfig, or ProviderConfig."
            )


@contextmanager
def context(settings: NumericsConfig, /) -> Iterator[None]:
    """Enter thread-local numerical settings and restore the previous values."""
    if not isinstance(settings, NumericsConfig):
        raise TypeError("context requires a NumericsConfig.")
    with _jax_config.numerics(
        enable_x64=settings.enable_x64,
        matmul_precision=settings.matmul_precision,
        dtype_promotion=settings.dtype_promotion,
        rank_promotion=settings.rank_promotion,
        transfer_guard_device_to_host=settings.transfer_guard_device_to_host,
        disable_jit=settings.disable_jit,
    ):
        yield


def environment(
    settings: RuntimeConfig,
    /,
    *,
    base: Mapping[str, str] | None = None,
    mode: EnvironmentMerge = "override",
) -> dict[str, str]:
    """Environment for a child process that applies `settings` when it starts.

    `base=None` copies the current process environment. In `"defaults"` mode a
    setting already selected by `base` is kept. Neither `base` nor the current
    process is modified.
    """
    return runtime_environment(settings, base, parse(mode, EnvironmentMerge, "mode"))


def logging_enabled() -> bool:
    """Whether canonical Phydrax events are emitted."""
    return _STATE.logging_enabled


def library_input(setting: LibrarySetting, /) -> str | None:
    """Configured native library path, else its declared environment value."""
    if setting in _STATE.libraries:
        return _STATE.libraries[setting]
    return os.environ.get(LIBRARY_ENVIRONMENT[setting])


def worker_input(setting: WorkerSetting, /) -> str | None:
    """Configured native worker executable, else its declared environment value."""
    if setting in _STATE.workers:
        return _STATE.workers[setting]
    return os.environ.get(WORKER_ENVIRONMENT[setting])


def snapshot() -> ConfigurationSnapshot:
    """Observe the effective configuration; never initializes a JAX backend."""
    with _LOCK:
        libraries = {key: library_input(key) for key in LIBRARY_ENVIRONMENT}
        workers = {key: worker_input(key) for key in WORKER_ENVIRONMENT}
        return ConfigurationSnapshot(
            requested=_STATE.requested,
            enable_x64=_jax_config.x64_enabled(),
            platforms=_jax_config.platforms(),
            cpu_device_count=_jax_config.cpu_device_count(),
            cpu_collectives_implementation=_jax_config.cpu_collectives_implementation(),
            matmul_precision=_jax_config.matmul_precision(),
            dtype_promotion=_jax_config.dtype_promotion(),
            rank_promotion=_jax_config.rank_promotion(),
            transfer_guard_device_to_host=_jax_config.transfer_guard_device_to_host(),
            jit_disabled=_jax_config.jit_disabled(),
            default_prng_implementation=_jax_config.default_prng_implementation(),
            logging_enabled=_STATE.logging_enabled,
            library_paths=frozendict(
                {key: value for key, value in libraries.items() if value is not None}
            ),
            workers=frozendict(
                {key: value for key, value in workers.items() if value is not None}
            ),
            backends_initialized=_jax_config.backends_initialized(),
            distributed_initialized=_jax_config.distributed_initialized(),
        )


def initialize_process() -> None:
    """Apply the startup runtime request once, before any numerical module loads.

    A process that already joined a distributed process set keeps it.
    """
    with _LOCK:
        if _STATE.started:
            return
        settings = startup_runtime(os.environ)
        if settings.bootstrap is not None and _jax_config.distributed_initialized():
            settings = replace(settings, bootstrap=None)
        _configure_runtime(settings)
        from ._jax_indexing import install_accelerator_lowerings

        install_accelerator_lowerings()
        _STATE.started = True


__all__ = [
    "ConfigurationSnapshot",
    "configure",
    "context",
    "environment",
    "initialize_process",
    "library_input",
    "logging_enabled",
    "snapshot",
    "worker_input",
]
