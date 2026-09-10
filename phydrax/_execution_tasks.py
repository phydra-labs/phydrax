#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Bounded independent host tasks with cooperative cancellation and backpressure."""

from __future__ import annotations

import multiprocessing
import threading
from collections.abc import Callable, Mapping
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Generic, Protocol, Self, TypeVar

from ._execution_control import CancellationMode, CancellationToken


T = TypeVar("T")


class TaskHandle(Protocol[T]):
    task_id: str

    def done(self) -> bool: ...

    def cancel(self, reason: str = "task cancelled") -> bool: ...

    def result(self, timeout: float | None = None) -> T: ...


class HostTaskExecutor(Protocol):
    def submit(
        self,
        task_id: str,
        operation: Callable[..., T],
        /,
        *args: Any,
        byte_count: int = 0,
        cancellation_mode: CancellationMode = CancellationMode.DRAIN,
        **kwargs: Any,
    ) -> TaskHandle[T]: ...

    def close(self, *, cancel_queued: bool = False) -> None: ...


@dataclass(frozen=True, slots=True)
class ImmediateTaskHandle(Generic[T]):
    task_id: str
    value: T

    def done(self) -> bool:
        return True

    def cancel(self, reason: str = "task cancelled") -> bool:
        del reason
        return False

    def result(self, timeout: float | None = None) -> T:
        del timeout
        return self.value


class FutureTaskHandle(Generic[T]):
    __slots__ = ("_future", "_mode", "_token", "task_id")

    def __init__(
        self,
        task_id: str,
        future: Future[T],
        mode: CancellationMode,
        token: CancellationToken | None,
    ) -> None:
        self.task_id = task_id
        self._future = future
        self._mode = mode
        self._token = token

    def done(self) -> bool:
        return self._future.done()

    def cancel(self, reason: str = "task cancelled") -> bool:
        if self._token is not None:
            self._token.cancel(reason)
        return self._future.cancel()

    def result(self, timeout: float | None = None) -> T:
        return self._future.result(timeout=timeout)


class InlineTaskExecutor:
    """Synchronous reference executor preserving the host-task contract."""

    __slots__ = ("_closed", "_task_ids")

    def __init__(self) -> None:
        self._closed = False
        self._task_ids: set[str] = set()

    def submit(
        self,
        task_id: str,
        operation: Callable[..., T],
        /,
        *args: Any,
        byte_count: int = 0,
        cancellation_mode: CancellationMode = CancellationMode.DRAIN,
        **kwargs: Any,
    ) -> ImmediateTaskHandle[T]:
        task = _task_id(task_id)
        if self._closed:
            raise RuntimeError("task executor is closed")
        if task in self._task_ids:
            raise ValueError(f"duplicate task_id {task!r}")
        if byte_count < 0:
            raise ValueError("byte_count must be non-negative")
        self._task_ids.add(task)
        mode = CancellationMode(cancellation_mode)
        try:
            if mode is CancellationMode.COOPERATIVE:
                token = CancellationToken()
                value = operation(token, *args, **kwargs)
            else:
                value = operation(*args, **kwargs)
        finally:
            self._task_ids.remove(task)
        return ImmediateTaskHandle(task, value)

    def close(self, *, cancel_queued: bool = False) -> None:
        del cancel_queued
        self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.close()
        return False


def _call_with_token(
    operation: Callable[..., T],
    token: CancellationToken,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> T:
    token.raise_if_cancelled()
    return operation(token, *args, **kwargs)


def _call_without_token(
    operation: Callable[..., T],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> T:
    return operation(*args, **kwargs)


class _BoundedFutureExecutor:
    __slots__ = (
        "_closed",
        "_condition",
        "_executor",
        "_maximum_pending",
        "_maximum_pending_bytes",
        "_pending",
        "_pending_bytes",
        "_task_ids",
    )

    def __init__(
        self,
        executor: ThreadPoolExecutor | ProcessPoolExecutor,
        *,
        maximum_pending: int,
        maximum_pending_bytes: int,
    ) -> None:
        if maximum_pending <= 0:
            raise ValueError("maximum_pending must be positive")
        if maximum_pending_bytes <= 0:
            raise ValueError("maximum_pending_bytes must be positive")
        self._executor = executor
        self._maximum_pending = int(maximum_pending)
        self._maximum_pending_bytes = int(maximum_pending_bytes)
        self._condition = threading.Condition()
        self._pending = 0
        self._pending_bytes = 0
        self._task_ids: set[str] = set()
        self._closed = False

    def _reserve(self, task_id: str, byte_count: int) -> None:
        task = _task_id(task_id)
        size = int(byte_count)
        if size < 0:
            raise ValueError("byte_count must be non-negative")
        if size > self._maximum_pending_bytes:
            raise ValueError("one task exceeds maximum_pending_bytes")
        with self._condition:
            if self._closed:
                raise RuntimeError("task executor is closed")
            if task in self._task_ids:
                raise ValueError(f"duplicate task_id {task!r}")
            self._condition.wait_for(
                lambda: (
                    self._closed
                    or (
                        self._pending < self._maximum_pending
                        and self._pending_bytes + size <= self._maximum_pending_bytes
                    )
                )
            )
            if self._closed:
                raise RuntimeError("task executor closed while waiting for capacity")
            self._task_ids.add(task)
            self._pending += 1
            self._pending_bytes += size

    def _release(self, task_id: str, byte_count: int) -> None:
        with self._condition:
            self._task_ids.remove(task_id)
            self._pending -= 1
            self._pending_bytes -= int(byte_count)
            self._condition.notify_all()

    def close(self, *, cancel_queued: bool = False) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            self._condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=cancel_queued)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exception_type, exception, traceback) -> bool:
        self.close()
        return False


class BoundedThreadTaskExecutor(_BoundedFutureExecutor):
    """Bounded thread execution for blocking host I/O and serialization only."""

    def __init__(
        self,
        *,
        maximum_workers: int = 1,
        maximum_pending: int = 2,
        maximum_pending_bytes: int = 1 << 30,
    ) -> None:
        if maximum_workers <= 0:
            raise ValueError("maximum_workers must be positive")
        super().__init__(
            ThreadPoolExecutor(
                max_workers=maximum_workers,
                thread_name_prefix="phydrax-host-task",
            ),
            maximum_pending=maximum_pending,
            maximum_pending_bytes=maximum_pending_bytes,
        )

    def submit(
        self,
        task_id: str,
        operation: Callable[..., T],
        /,
        *args: Any,
        byte_count: int = 0,
        cancellation_mode: CancellationMode = CancellationMode.DRAIN,
        **kwargs: Any,
    ) -> FutureTaskHandle[T]:
        task = _task_id(task_id)
        mode = CancellationMode(cancellation_mode)
        if mode is CancellationMode.PROCESS:
            raise ValueError("process cancellation requires an isolated process executor")
        self._reserve(task, byte_count)
        token = CancellationToken() if mode is CancellationMode.COOPERATIVE else None
        if token is None:
            future = self._executor.submit(
                _call_without_token,
                operation,
                tuple(args),
                dict(kwargs),
            )
        else:
            future = self._executor.submit(
                _call_with_token,
                operation,
                token,
                tuple(args),
                dict(kwargs),
            )
        future.add_done_callback(lambda _: self._release(task, byte_count))
        return FutureTaskHandle(task, future, mode, token)


class SpawnedProcessTaskExecutor(_BoundedFutureExecutor):
    """Spawn-only process isolation for independent Python CPU work."""

    def __init__(
        self,
        *,
        maximum_workers: int = 1,
        maximum_pending: int = 2,
        maximum_pending_bytes: int = 1 << 30,
    ) -> None:
        if maximum_workers <= 0:
            raise ValueError("maximum_workers must be positive")
        super().__init__(
            ProcessPoolExecutor(
                max_workers=maximum_workers,
                mp_context=multiprocessing.get_context("spawn"),
            ),
            maximum_pending=maximum_pending,
            maximum_pending_bytes=maximum_pending_bytes,
        )

    def submit(
        self,
        task_id: str,
        operation: Callable[..., T],
        /,
        *args: Any,
        byte_count: int = 0,
        cancellation_mode: CancellationMode = CancellationMode.DRAIN,
        **kwargs: Any,
    ) -> FutureTaskHandle[T]:
        task = _task_id(task_id)
        mode = CancellationMode(cancellation_mode)
        if mode is not CancellationMode.DRAIN:
            raise ValueError(
                "spawned process-pool tasks are drain-only; use a dedicated managed "
                "worker for killable process cancellation"
            )
        self._reserve(task, byte_count)
        future = self._executor.submit(
            _call_without_token,
            operation,
            tuple(args),
            dict(kwargs),
        )
        future.add_done_callback(lambda _: self._release(task, byte_count))
        return FutureTaskHandle(task, future, mode, None)


def _task_id(value: str) -> str:
    task = str(value).strip()
    if not task:
        raise ValueError("task_id must be non-empty")
    return task


__all__ = (
    "BoundedThreadTaskExecutor",
    "FutureTaskHandle",
    "HostTaskExecutor",
    "ImmediateTaskHandle",
    "InlineTaskExecutor",
    "SpawnedProcessTaskExecutor",
    "TaskHandle",
)
