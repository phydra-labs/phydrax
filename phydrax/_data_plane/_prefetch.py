#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from queue import Empty, Full, Queue
from threading import current_thread, Event, Thread
from typing import cast, Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
_END_OF_INPUT = object()


@dataclass(frozen=True, slots=True)
class _ProducerFailure:
    error: Exception


class BoundedPrefetchIterator(Iterator[OutputT], Generic[InputT, OutputT]):
    """Prepare ordered items synchronously or on one bounded producer thread.

    Closing prevents another item from starting, but it cannot interrupt a preparation
    callback already in progress. The callback must therefore complete finitely.
    """

    def __init__(
        self,
        items: Iterable[InputT],
        prepare: Callable[[InputT], OutputT],
        /,
        *,
        capacity: int,
        thread_name: str,
    ):
        resolved_capacity = int(capacity)
        if resolved_capacity < 0:
            raise ValueError("capacity must be nonnegative.")
        resolved_name = str(thread_name).strip()
        if not resolved_name:
            raise ValueError("thread_name must be non-empty.")
        self._items = iter(items)
        self._prepare = prepare
        self._capacity = resolved_capacity
        self._thread_name = resolved_name
        self._closed = False
        self._stop = Event()
        self._queue: Queue[object] | None = (
            None if resolved_capacity == 0 else Queue(maxsize=resolved_capacity)
        )
        self._thread: Thread | None = None

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def closed(self) -> bool:
        return self._closed

    def __iter__(self) -> BoundedPrefetchIterator[InputT, OutputT]:
        return self

    def __next__(self) -> OutputT:
        if self._closed:
            raise StopIteration
        if self._queue is None:
            try:
                item = next(self._items)
                return self._prepare(item)
            except StopIteration:
                self.close()
                raise
            except Exception:
                self.close()
                raise

        self._start()
        queued = self._queue.get()
        if queued is _END_OF_INPUT:
            self.close()
            raise StopIteration
        if isinstance(queued, _ProducerFailure):
            error = queued.error
            self.close()
            raise error.with_traceback(error.__traceback__)
        return cast(OutputT, queued)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        if self._queue is not None:
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break
        thread = self._thread
        if thread is not None and thread is not current_thread():
            thread.join()

    def __enter__(self) -> BoundedPrefetchIterator[InputT, OutputT]:
        self._start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        del exc_type, exc_value, traceback
        self.close()

    def _start(self) -> None:
        if self._queue is None or self._thread is not None or self._closed:
            return
        self._thread = Thread(
            target=self._produce,
            name=self._thread_name,
            daemon=True,
        )
        self._thread.start()

    def _wait_for_capacity(self) -> bool:
        assert self._queue is not None
        while not self._stop.is_set():
            if not self._queue.full():
                return True
            self._stop.wait(0.01)
        return False

    def _put(self, item: object, /) -> bool:
        assert self._queue is not None
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.05)
                return True
            except Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            while self._wait_for_capacity():
                try:
                    item = next(self._items)
                except StopIteration:
                    return
                prepared = self._prepare(item)
                if not self._put(prepared):
                    return
        except Exception as error:
            self._put(_ProducerFailure(error))
        finally:
            self._put(_END_OF_INPUT)


__all__ = ["BoundedPrefetchIterator"]
