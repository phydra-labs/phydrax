#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Sole first-party owner of upstream JAX configuration access.

Every read, update, and scoped context of a JAX configuration option, and every
JAX/XLA process-environment spelling, passes through this module. Callers use
the typed `phydrax.config` records; nothing here interprets scientific meaning.
"""

from __future__ import annotations

import os
import re
import shlex
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, ExitStack
from types import MappingProxyType
from typing import Final

import jax

# JAX exposes no public query of backend readiness that does not itself initialize
# a backend: every public device or backend accessor creates the clients. This
# private predicate is read only here, to refuse configuration that can no longer
# take effect; it never initializes anything.
from jax._src import xla_bridge as _xla_bridge


ENABLE_X64_ENVIRONMENT: Final = "JAX_ENABLE_X64"
MATMUL_PRECISION_ENVIRONMENT: Final = "JAX_DEFAULT_MATMUL_PRECISION"
PLATFORMS_ENVIRONMENT: Final = "JAX_PLATFORMS"
CPU_DEVICE_COUNT_ENVIRONMENT: Final = "JAX_NUM_CPU_DEVICES"
XLA_FLAGS_ENVIRONMENT: Final = "XLA_FLAGS"
PREALLOCATE_ENVIRONMENT: Final = "XLA_PYTHON_CLIENT_PREALLOCATE"
# The deprecated single-platform selector still selects a launch platform when a
# caller's base environment supplies it.
_PLATFORM_NAME_ENVIRONMENT: Final = "JAX_PLATFORM_NAME"
_FORCED_HOST_DEVICES: Final = re.compile(
    r"--xla_force_host_platform_device_count=(\d+)\Z"
)
_FLAG_TOKENS: Final = re.compile(r"""(?:[^\s"'\\]+|"(?:\\.|[^"\\])*"|'[^']*'|\\.)+""")
_NUMERICS_ENVIRONMENTS: Final = {
    "enable_x64": ENABLE_X64_ENVIRONMENT,
    "matmul_precision": MATMUL_PRECISION_ENVIRONMENT,
    "dtype_promotion": "JAX_NUMPY_DTYPE_PROMOTION",
    "rank_promotion": "JAX_NUMPY_RANK_PROMOTION",
    "transfer_guard_device_to_host": "JAX_TRANSFER_GUARD_DEVICE_TO_HOST",
    "disable_jit": "JAX_DISABLE_JIT",
}

type DecodedEnvironment = dict[str, tuple[object, tuple[str, ...]]]


def backends_initialized() -> bool:
    """Whether any JAX backend client exists; never initializes one."""
    return _xla_bridge.backends_are_initialized()


def distributed_initialized() -> bool:
    return jax.distributed.is_initialized()


def x64_enabled() -> bool:
    """Effective 64-bit enablement, including any active scoped context."""
    return bool(jax.enable_x64.value)


def compilation_cache_enabled() -> bool:
    """Observe persistent compilation caching without initializing a backend."""
    return bool(jax.config.jax_enable_compilation_cache)


def process_x64_enabled() -> bool:
    """Process-wide 64-bit enablement, ignoring any scoped context."""
    return bool(jax.enable_x64.get_global())


def startup_x64(environment: Mapping[str, str], /) -> bool | None:
    """Default to x64 only when upstream has no explicit startup selection."""
    return None if ENABLE_X64_ENVIRONMENT in environment else True


def startup_matmul_precision(environment: Mapping[str, str], /) -> str | None:
    """Default to JAX's highest contraction precision without overriding a request.

    The meaning is backend-specific: IEEE float32 on NVIDIA rather than TF32,
    but multiple BF16 passes for float32 dot operations on TPU. This is not
    a universal claim of native IEEE float32 hardware. Explicit environment
    or earlier process settings remain authoritative.
    """
    if MATMUL_PRECISION_ENVIRONMENT in environment or matmul_precision() is not None:
        return None
    return "highest"


def platforms() -> tuple[str, ...] | None:
    value = jax.config.jax_platforms
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise TypeError("JAX reported a non-text platform selection.")
    return tuple(value.split(","))


def configuration_values() -> Mapping[str, object]:
    """Observe upstream options without initializing devices or mutating policy."""
    return dict(jax.config.values)


def cpu_collectives_implementation() -> str | None:
    value = jax.config.jax_cpu_collectives_implementation
    if value is not None and not isinstance(value, str):
        raise TypeError("JAX reported a non-text CPU collectives implementation.")
    return value


def cpu_device_count() -> int | None:
    """Requested CPU device count: the JAX option, else the XLA host-platform flag.

    `None` means the platform's own count. Reading either source never
    initializes a backend.
    """
    value = jax.config.jax_num_cpu_devices
    if not isinstance(value, int):
        raise TypeError("JAX reported a non-integer CPU device count.")
    if value >= 0:
        return value
    counts = _forced_host_tokens(os.environ.get(XLA_FLAGS_ENVIRONMENT, ""))
    return None if not counts else counts[-1][2]


def matmul_precision() -> str | None:
    value = jax.default_matmul_precision.value
    return None if value is None else str(value)


def dtype_promotion() -> str:
    return str(jax.numpy_dtype_promotion.value)


def rank_promotion() -> str:
    return str(jax.numpy_rank_promotion.value)


def transfer_guard_device_to_host() -> str | None:
    value = jax.transfer_guard_device_to_host.value
    return None if value is None else str(value)


def jit_disabled() -> bool:
    return bool(jax.config.jax_disable_jit)


def default_prng_implementation() -> str:
    return str(jax.default_prng_impl.value)


def set_x64(enabled: bool, /) -> None:
    jax.config.update("jax_enable_x64", enabled)


def set_matmul_precision(value: str, /) -> None:
    jax.config.update("jax_default_matmul_precision", value)


def default_device_selection() -> jax.Device | str | None:
    """Effective default placement: unset, a live device, or a platform name."""
    return jax.default_device.value


def set_platforms(selection: tuple[str, ...], /) -> None:
    jax.config.update("jax_platforms", ",".join(selection))


def set_cpu_device_count(count: int, /) -> None:
    jax.config.update("jax_num_cpu_devices", count)


def set_cpu_collectives_implementation(implementation: str, /) -> None:
    jax.config.update("jax_cpu_collectives_implementation", implementation)


def validate_matmul_precision(value: str, /) -> None:
    """Refuse a matmul precision JAX does not define, without entering a scope."""
    jax.default_matmul_precision(value)


@contextmanager
def numerics(
    *,
    enable_x64: bool | None,
    matmul_precision: str | None,
    dtype_promotion: str | None,
    rank_promotion: str | None,
    transfer_guard_device_to_host: str | None,
    disable_jit: bool | None,
) -> Iterator[None]:
    """Enter the requested thread-local JAX settings and restore them on exit."""
    with ExitStack() as stack:
        if enable_x64 is not None:
            stack.enter_context(jax.enable_x64(enable_x64))
        if matmul_precision is not None:
            stack.enter_context(jax.default_matmul_precision(matmul_precision))
        if dtype_promotion is not None:
            stack.enter_context(jax.numpy_dtype_promotion(dtype_promotion))
        if rank_promotion is not None:
            stack.enter_context(jax.numpy_rank_promotion(rank_promotion))
        if transfer_guard_device_to_host is not None:
            stack.enter_context(
                jax.transfer_guard_device_to_host(transfer_guard_device_to_host)
            )
        if disable_jit is not None:
            stack.enter_context(jax.disable_jit(disable_jit))
        yield


@contextmanager
def process_numerics() -> Iterator[None]:
    """Suspend thread-local numerical and placement scopes for process settings.

    Loading a module evaluates its dtype-dependent constants once for the whole
    process, so a deferred load must not observe the caller's scoped x64,
    promotion, precision, transfer-guard, or default-device overrides.
    """
    with ExitStack() as stack:
        for option in (
            jax.enable_x64,
            jax.default_matmul_precision,
            jax.numpy_dtype_promotion,
            jax.numpy_rank_promotion,
            jax.transfer_guard_device_to_host,
            jax.default_device,
        ):
            stack.enter_context(option(option.get_global()))
        yield


@contextmanager
def default_device(device: jax.Device, /) -> Iterator[None]:
    """Place uncommitted computations on a live `device` within the scope.

    A device is a runtime object, not host configuration, so execution owners
    use this scope directly rather than through a configuration record.
    """
    if not isinstance(device, jax.Device):
        raise TypeError("default_device requires a live JAX device.")
    with jax.default_device(device):
        yield


def _environment_text(environment: Mapping[str, str], key: str, /) -> str:
    value = environment[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be text.")
    return value


def _boolean(text: str, key: str, /) -> bool:
    if not isinstance(text, str):
        raise TypeError(f"{key} must be text.")
    value = text.lower()
    if value in ("y", "yes", "t", "true", "on", "1"):
        return True
    if value in ("n", "no", "f", "false", "off", "0"):
        return False
    raise ValueError(f"invalid truth value {text!r} for environment {key!r}")


def _forced_host_tokens(flags: str, /) -> tuple[tuple[int, int, int], ...]:
    tokens: list[tuple[int, int, int]] = []
    for match in _FLAG_TOKENS.finditer(flags):
        parts = shlex.split(match.group())
        if len(parts) == 1:
            count = _FORCED_HOST_DEVICES.fullmatch(parts[0])
            if count is not None:
                tokens.append((match.start(), match.end(), int(count.group(1))))
    return tuple(tokens)


def runtime_environment_inputs(environment: Mapping[str, str], /) -> DecodedEnvironment:
    inputs: DecodedEnvironment = {}
    if ENABLE_X64_ENVIRONMENT in environment:
        inputs["enable_x64"] = (
            _boolean(environment[ENABLE_X64_ENVIRONMENT], ENABLE_X64_ENVIRONMENT),
            (ENABLE_X64_ENVIRONMENT,),
        )
    platform_key = (
        PLATFORMS_ENVIRONMENT
        if PLATFORMS_ENVIRONMENT in environment
        and _environment_text(environment, PLATFORMS_ENVIRONMENT)
        else _PLATFORM_NAME_ENVIRONMENT
    )
    if platform_key in environment and _environment_text(environment, platform_key):
        inputs["platforms"] = (
            tuple(environment[platform_key].split(",")),
            (platform_key,),
        )
    count = (
        int(_environment_text(environment, CPU_DEVICE_COUNT_ENVIRONMENT))
        if CPU_DEVICE_COUNT_ENVIRONMENT in environment
        else -1
    )
    if count != -1:
        inputs["cpu_device_count"] = (count, (CPU_DEVICE_COUNT_ENVIRONMENT,))
    else:
        tokens = _forced_host_tokens(
            _environment_text(environment, XLA_FLAGS_ENVIRONMENT)
            if XLA_FLAGS_ENVIRONMENT in environment
            else ""
        )
        if tokens:
            inputs["cpu_device_count"] = (tokens[-1][2], (XLA_FLAGS_ENVIRONMENT,))
    return inputs


def numerics_environment_inputs(environment: Mapping[str, str], /) -> DecodedEnvironment:
    return {
        field: (
            _boolean(environment[key], key)
            if field in ("enable_x64", "disable_jit")
            else _environment_text(environment, key),
            (key,),
        )
        for field, key in _NUMERICS_ENVIRONMENTS.items()
        if key in environment
    }


def launch_environment_inputs(environment: Mapping[str, str], /) -> DecodedEnvironment:
    inputs: DecodedEnvironment = {}
    if XLA_FLAGS_ENVIRONMENT in environment:
        inputs["xla_flags"] = (
            _environment_text(environment, XLA_FLAGS_ENVIRONMENT),
            (XLA_FLAGS_ENVIRONMENT,),
        )
    preallocate = (
        _environment_text(environment, PREALLOCATE_ENVIRONMENT)
        if PREALLOCATE_ENVIRONMENT in environment
        else None
    )
    if preallocate:
        inputs["preallocate"] = (
            preallocate not in ("false", "False", "0"),
            (PREALLOCATE_ENVIRONMENT,),
        )
    return inputs


# Upstream keys that may source each environment-decoded field; a field decoded
# from either alias of one selection names the alias it was read from.
RUNTIME_ENVIRONMENT_KEYS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = (
    MappingProxyType(
        {
            "enable_x64": ((ENABLE_X64_ENVIRONMENT,),),
            "platforms": ((PLATFORMS_ENVIRONMENT,), (_PLATFORM_NAME_ENVIRONMENT,)),
            "cpu_device_count": (
                (CPU_DEVICE_COUNT_ENVIRONMENT,),
                (XLA_FLAGS_ENVIRONMENT,),
            ),
        }
    )
)
NUMERICS_ENVIRONMENT_KEYS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = (
    MappingProxyType({field: ((key,),) for field, key in _NUMERICS_ENVIRONMENTS.items()})
)
LAUNCH_ENVIRONMENT_KEYS: Final[Mapping[str, tuple[tuple[str, ...], ...]]] = (
    MappingProxyType(
        {
            "xla_flags": ((XLA_FLAGS_ENVIRONMENT,),),
            "preallocate": ((PREALLOCATE_ENVIRONMENT,),),
        }
    )
)


def valid_launch_environment_origin(
    keys: tuple[str, ...], flags: bool, preallocate: bool, /
) -> bool:
    selected: set[str] = set()
    if flags:
        selected.add(XLA_FLAGS_ENVIRONMENT)
    if preallocate:
        selected.add(PREALLOCATE_ENVIRONMENT)
    return bool(selected) and keys == tuple(sorted(selected))


def project_environment(
    result: dict[str, str],
    base: Mapping[str, str],
    override: bool,
    /,
    *,
    enable_x64: bool | None,
    platforms: tuple[str, ...] | None,
    cpu_device_count: int | None,
    xla_flags: str | None,
    preallocate: bool | None,
) -> None:
    """Project upstream choices, normalizing aliases only when explicitly replaced."""
    if enable_x64 is not None and (override or ENABLE_X64_ENVIRONMENT not in base):
        result[ENABLE_X64_ENVIRONMENT] = "1" if enable_x64 else "0"
    if platforms is not None and (
        override
        or not ({PLATFORMS_ENVIRONMENT, _PLATFORM_NAME_ENVIRONMENT} & base.keys())
    ):
        result[PLATFORMS_ENVIRONMENT] = ",".join(platforms)
        result.pop(_PLATFORM_NAME_ENVIRONMENT, None)
    if xla_flags is not None and (override or XLA_FLAGS_ENVIRONMENT not in base):
        result[XLA_FLAGS_ENVIRONMENT] = xla_flags
    selected_count = CPU_DEVICE_COUNT_ENVIRONMENT in base or bool(
        _forced_host_tokens(base.get(XLA_FLAGS_ENVIRONMENT, ""))
    )
    if cpu_device_count is not None and (override or not selected_count):
        result[CPU_DEVICE_COUNT_ENVIRONMENT] = str(cpu_device_count)
        flags = result.get(XLA_FLAGS_ENVIRONMENT, "")
        cursor = 0
        pieces: list[str] = []
        for start, end, count in _forced_host_tokens(flags):
            if count != cpu_device_count:
                pieces.append(flags[cursor:start])
                cursor = end
        if cursor:
            pieces.append(flags[cursor:])
            remaining = "".join(pieces).strip()
            if remaining:
                result[XLA_FLAGS_ENVIRONMENT] = remaining
            else:
                result.pop(XLA_FLAGS_ENVIRONMENT, None)
    if preallocate is not None and (override or PREALLOCATE_ENVIRONMENT not in base):
        result[PREALLOCATE_ENVIRONMENT] = "true" if preallocate else "false"


__all__ = [
    "CPU_DEVICE_COUNT_ENVIRONMENT",
    "ENABLE_X64_ENVIRONMENT",
    "MATMUL_PRECISION_ENVIRONMENT",
    "LAUNCH_ENVIRONMENT_KEYS",
    "NUMERICS_ENVIRONMENT_KEYS",
    "PLATFORMS_ENVIRONMENT",
    "PREALLOCATE_ENVIRONMENT",
    "RUNTIME_ENVIRONMENT_KEYS",
    "XLA_FLAGS_ENVIRONMENT",
    "backends_initialized",
    "compilation_cache_enabled",
    "configuration_values",
    "cpu_collectives_implementation",
    "cpu_device_count",
    "default_device",
    "default_device_selection",
    "default_prng_implementation",
    "distributed_initialized",
    "dtype_promotion",
    "launch_environment_inputs",
    "numerics_environment_inputs",
    "project_environment",
    "runtime_environment_inputs",
    "valid_launch_environment_origin",
    "jit_disabled",
    "matmul_precision",
    "numerics",
    "platforms",
    "process_x64_enabled",
    "rank_promotion",
    "set_cpu_collectives_implementation",
    "set_cpu_device_count",
    "set_platforms",
    "set_matmul_precision",
    "set_x64",
    "transfer_guard_device_to_host",
    "validate_matmul_precision",
    "x64_enabled",
]
