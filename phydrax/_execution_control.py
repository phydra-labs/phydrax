#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Host observation, global consensus, and cooperative failure scopes."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from enum import Enum
from typing import Any, Self

import jax
import numpy as np
from jax.experimental import multihost_utils


class ObservationScope(str, Enum):
    LOCAL = "local"
    ALL_PROCESSES = "all_processes"
    COORDINATOR = "coordinator"
    GLOBAL_SUM = "global_sum"
    GLOBAL_MAXIMUM = "global_maximum"
    GLOBAL_MINIMUM = "global_minimum"


class FailureScope(str, Enum):
    COUPLED = "coupled"
    INDEPENDENT = "independent"


class CancellationMode(str, Enum):
    COOPERATIVE = "cooperative"
    DRAIN = "drain"
    PROCESS = "process"


@dataclass(frozen=True, slots=True)
class DistributedObservationPolicy:
    scope: ObservationScope = ObservationScope.LOCAL
    coordinator_process: int = 0
    execution_group_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope", ObservationScope(self.scope))
        if self.coordinator_process < 0:
            raise ValueError("coordinator_process must be non-negative")
        if self.execution_group_id is not None and not self.execution_group_id:
            raise ValueError("execution_group_id must be non-empty")


@dataclass(frozen=True, slots=True)
class HostObservation:
    """Host-materialized value and exact distributed delivery coordinates."""

    value: Any
    scope: ObservationScope
    process_index: int
    process_count: int
    execution_group_id: str | None


def materialize_observation(
    value: Any,
    policy: DistributedObservationPolicy,
    /,
) -> HostObservation | None:
    """Materialize one symmetric host observation under an explicit scope."""

    local = np.asarray(jax.device_get(value))
    scope = policy.scope
    if scope in (ObservationScope.LOCAL, ObservationScope.ALL_PROCESSES):
        observed = local
    elif scope is ObservationScope.COORDINATOR:
        if jax.process_index() != policy.coordinator_process:
            return None
        observed = local
    else:
        gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False))
        if scope is ObservationScope.GLOBAL_SUM:
            observed = np.sum(gathered, axis=0)
        elif scope is ObservationScope.GLOBAL_MAXIMUM:
            observed = np.max(gathered, axis=0)
        elif scope is ObservationScope.GLOBAL_MINIMUM:
            observed = np.min(gathered, axis=0)
        else:
            raise ValueError(f"unsupported observation scope {scope!r}")
        if jax.process_index() != policy.coordinator_process:
            return None
    return HostObservation(
        observed,
        scope,
        jax.process_index(),
        jax.process_count(),
        policy.execution_group_id,
    )


def global_boolean_consensus(value: Any, /, *, require_all: bool = True) -> bool:
    """Return one process-symmetric boolean agreement at a host safe point."""

    local = np.asarray(bool(np.asarray(jax.device_get(value))), dtype=np.int8)
    gathered = np.asarray(multihost_utils.process_allgather(local, tiled=False))
    return bool(np.all(gathered)) if require_all else bool(np.any(gathered))


class CancellationToken:
    """Cooperative cancellation flag; never injects an asynchronous exception."""

    __slots__ = ("_event", "_reason")

    def __init__(self) -> None:
        self._event = threading.Event()
        self._reason: str | None = None

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str | None:
        return self._reason

    def cancel(self, reason: str, /) -> None:
        normalized = str(reason).strip()
        if not normalized:
            raise ValueError("cancellation reason must be non-empty")
        if not self._event.is_set():
            self._reason = normalized
            self._event.set()

    def wait(self, timeout: float | None = None, /) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ExecutionCancelledError(self._reason or "execution cancelled")


class ExecutionCancelledError(RuntimeError):
    """Cooperative execution stopped at a declared safe boundary."""


class ExecutionFailureContext:
    """Failure scope sharing one cooperative cancellation token."""

    __slots__ = ("mode", "token")

    def __init__(
        self,
        mode: FailureScope,
        token: CancellationToken | None = None,
    ) -> None:
        self.mode = FailureScope(mode)
        self.token = CancellationToken() if token is None else token

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        if exception is not None and self.mode is FailureScope.COUPLED:
            self.token.cancel(f"coupled execution failed: {exception}")
        return False


__all__ = (
    "CancellationMode",
    "CancellationToken",
    "DistributedObservationPolicy",
    "ExecutionCancelledError",
    "ExecutionFailureContext",
    "FailureScope",
    "HostObservation",
    "ObservationScope",
    "global_boolean_consensus",
    "materialize_observation",
)
