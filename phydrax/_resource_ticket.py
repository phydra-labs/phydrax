#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Explicit batch reservations for host work, independent of native availability."""

from __future__ import annotations

import threading
from contextlib import AbstractContextManager
from types import TracebackType
from typing import final, NoReturn, Protocol


class BatchTicket(Protocol):
    """A reserved upper bound with local counters and explicit child batches."""

    @property
    def work_upper(self) -> int: ...

    @property
    def queries_upper(self) -> int: ...

    @property
    def cavity_upper(self) -> int: ...

    @property
    def bytes_upper(self) -> int: ...

    @property
    def work_units(self) -> int: ...

    @property
    def geometry_queries(self) -> int: ...

    @property
    def cavity_cells(self) -> int: ...

    @property
    def managed_bytes(self) -> int: ...

    @property
    def settled(self) -> bool: ...

    @property
    def native_accounting(self) -> bool: ...

    @property
    def remaining_work_units(self) -> int: ...

    @property
    def remaining_geometry_queries(self) -> int: ...

    def consume(
        self,
        work: int = 0,
        /,
        *,
        geometry_queries: int = 0,
        cavity_cells: int | None = None,
        managed_bytes: int | None = None,
    ) -> None: ...

    def batch(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> BatchTicket: ...

    def checkpoint(self) -> None: ...

    def settle(self) -> None: ...

    def refuse(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NoReturn: ...

    def native_target(self) -> AbstractContextManager[None]: ...

    def managed_storage(self) -> ManagedBytesContributor: ...

    def set_managed_storage(
        self, owner: ManagedBytesContributor, bound: int | None, /
    ) -> None: ...

    def __enter__(self) -> BatchTicket: ...

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


def _bound(value: int, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
            f"HostBatchTicket.{name} must be an explicit integer; got {value!r}."
        )
    if value < 0:
        raise ValueError(f"HostBatchTicket.{name} must be nonnegative; got {value}.")
    return value


class HostBatchEstimateError(RuntimeError):
    """A host phase exceeds its explicit estimate after domain-limit admission."""

    def __init__(
        self,
        requested: tuple[int, int, int, int],
        evidence: tuple[int, int, int, int, int, int, int, int],
        /,
    ) -> None:
        self.batch_requested_upper = requested
        self.batch_refusal_evidence = evidence
        super().__init__(
            f"HostBatchTicket upper must cover the host phase; got requested={requested}, evidence={evidence}."
        )


@final
class ManagedBytesContributor:
    """One explicit live-capacity contribution to a batch ticket."""

    def __init__(self, ticket: BatchTicket, /) -> None:
        self.ticket = ticket
        self._bound = 0
        self._entered = False
        self._closed = False

    @property
    def bound(self) -> int:
        return self._bound

    def set_bound(self, bytes_upper: int, /) -> None:
        if self._closed:
            raise RuntimeError(
                f"ManagedBytesContributor.closed must be false; got {self._closed}."
            )
        bound = _bound(bytes_upper, "managed_bytes")
        self.ticket.set_managed_storage(self, bound)
        self._bound = bound

    def close(self) -> None:
        if not self._closed:
            self.ticket.set_managed_storage(self, None)
            self._bound = 0
            self._closed = True

    def __enter__(self) -> ManagedBytesContributor:
        if self._entered or self._closed:
            raise RuntimeError(
                f"ManagedBytesContributor.state must be unused and open; got entered={self._entered}, closed={self._closed}."
            )
        self._entered = True
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        try:
            self.close()
        except BaseException as release:
            if error is None:
                raise
            error.add_note(f"Managed byte-contributor release also refused: {release}")


@final
class HostBatchTicket:
    """Reserve host-only work; child estimates cannot overcommit their parent.

    Root bounds are supplied by the request's existing domain limits. Capacity
    counters are simultaneous live values; settled evidence retains their peaks.
    """

    def __init__(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> None:
        self.work_upper = _bound(work_upper, "work_upper")
        self.queries_upper = _bound(queries_upper, "queries_upper")
        self.bytes_upper = _bound(bytes_upper, "bytes_upper")
        self.cavity_upper = _bound(cavity_upper, "cavity_upper")
        self.work_units = 0
        self.geometry_queries = 0
        self.cavity_cells = 0
        self.managed_bytes = 0
        self._own_cavity = 0
        self._own_bytes = 0
        self._default_managed_bytes = 0
        self._managed_components: dict[ManagedBytesContributor, int] = {}
        self._live_cavity = 0
        self._live_bytes = 0
        self._children_work = 0
        self._children_queries = 0
        self._reserved_work = 0
        self._reserved_queries = 0
        self._reserved_cavity = 0
        self._reserved_bytes = 0
        self._children: set[HostBatchTicket] = set()
        self._parent: HostBatchTicket | None = None
        self._lock = threading.RLock()
        self._entered = False
        self._settled = False

    def _require_live(self) -> None:
        if self._settled:
            raise RuntimeError(
                f"HostBatchTicket.settled must be false; got {self._settled}."
            )

    @property
    def settled(self) -> bool:
        return self._settled

    @property
    def native_accounting(self) -> bool:
        return False

    @property
    def remaining_work_units(self) -> int:
        return (
            self.work_upper - self.work_units - self._children_work - self._reserved_work
        )

    @property
    def remaining_geometry_queries(self) -> int:
        return (
            self.queries_upper
            - self.geometry_queries
            - self._children_queries
            - self._reserved_queries
        )

    def __enter__(self) -> HostBatchTicket:
        with self._lock:
            self._require_live()
            if self._entered:
                raise RuntimeError(
                    f"HostBatchTicket.entered must be false; got {self._entered}."
                )
            self._entered = True
        return self

    def consume(
        self,
        work: int = 0,
        /,
        *,
        geometry_queries: int = 0,
        cavity_cells: int | None = None,
        managed_bytes: int | None = None,
    ) -> None:
        work = _bound(work, "work")
        queries = _bound(geometry_queries, "geometry_queries")
        with self._lock:
            self._require_live()
            cavity = (
                self._own_cavity
                if cavity_cells is None
                else _bound(cavity_cells, "cavity_cells")
            )
            direct = (
                self._default_managed_bytes
                if managed_bytes is None
                else _bound(managed_bytes, "managed_bytes")
            )
            managed = (
                self._own_bytes
                if managed_bytes is None
                else direct + sum(self._managed_components.values())
            )
            if (
                work > self.remaining_work_units
                or queries > self.remaining_geometry_queries
                or cavity > self.cavity_upper - self._reserved_cavity
                or managed > self.bytes_upper - self._reserved_bytes
            ):
                raise RuntimeError(
                    "HostBatchTicket.consume must fit the admitted upper bound; got estimate defect."
                )
            self.work_units += work
            self.geometry_queries += queries
            self._own_cavity = cavity
            self._own_bytes = managed
            self._default_managed_bytes = direct
            if cavity_cells is not None or managed_bytes is not None:
                self._observe_capacity()
        if self._parent is not None and (
            cavity_cells is not None or managed_bytes is not None
        ):
            self._parent._observe_descendant_capacity()

    def _observe_capacity(self) -> None:
        cavity = self._own_cavity + sum(child._live_cavity for child in self._children)
        managed = self._own_bytes + sum(child._live_bytes for child in self._children)
        self._live_cavity = cavity
        self._live_bytes = managed
        self.cavity_cells = max(self.cavity_cells, cavity)
        self.managed_bytes = max(self.managed_bytes, managed)

    def native_target(self) -> NoReturn:
        raise TypeError(
            "HostBatchTicket.native_target must receive a native ticket; got a host-only ticket."
        )

    def managed_storage(self) -> ManagedBytesContributor:
        self._require_live()
        return ManagedBytesContributor(self)

    def set_managed_storage(
        self, owner: ManagedBytesContributor, bound: int | None, /
    ) -> None:
        if owner.ticket is not self:
            raise ValueError(
                f"HostBatchTicket.owner.ticket must be this ticket; got {type(owner.ticket).__qualname__} with different identity."
            )
        amount = 0 if bound is None else _bound(bound, "managed_bytes")
        with self._lock:
            prior = self._managed_components.get(owner, 0)
            if self._settled:
                if amount > prior:
                    raise RuntimeError(
                        f"HostBatchTicket.managed_storage must only release capacity after settlement; got amount={amount}, prior={prior}."
                    )
            else:
                aggregate = self._own_bytes - prior + amount
                if aggregate > self.bytes_upper - self._reserved_bytes:
                    self.refuse(0, bytes_upper=aggregate)
            if bound is None:
                self._managed_components.pop(owner, None)
            else:
                self._managed_components[owner] = amount
            self._own_bytes = self._default_managed_bytes + sum(
                self._managed_components.values()
            )
            self._observe_capacity()
        if not self._settled and self._parent is not None:
            self._parent._observe_descendant_capacity()

    def _observe_descendant_capacity(self) -> None:
        with self._lock:
            self._observe_capacity()
        if self._parent is not None:
            self._parent._observe_descendant_capacity()

    def batch(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> HostBatchTicket:
        child = HostBatchTicket(
            work_upper,
            queries_upper=queries_upper,
            bytes_upper=bytes_upper,
            cavity_upper=cavity_upper,
        )
        with self._lock:
            self._require_live()
            if (
                child.work_upper > self.remaining_work_units
                or child.queries_upper > self.remaining_geometry_queries
                or child.cavity_upper
                > self.cavity_upper - self._own_cavity - self._reserved_cavity
                or child.bytes_upper
                > self.bytes_upper - self._own_bytes - self._reserved_bytes
            ):
                raise RuntimeError(
                    "HostBatchTicket.child must fit the parent estimate; got excessive reservation."
                )
            child._parent = self
            self._children.add(child)
            self._reserved_work += child.work_upper
            self._reserved_queries += child.queries_upper
            self._reserved_cavity += child.cavity_upper
            self._reserved_bytes += child.bytes_upper
        return child

    def checkpoint(self) -> None:
        self._require_live()

    def settle(self) -> None:
        with self._lock:
            self._require_live()
            if self._children:
                raise RuntimeError(
                    "HostBatchTicket.settle must follow child settlement; got live children."
                )
            self._settled = True
        parent = self._parent
        if parent is not None:
            with parent._lock:
                parent._children.remove(self)
                parent._reserved_work -= self.work_upper
                parent._reserved_queries -= self.queries_upper
                parent._reserved_cavity -= self.cavity_upper
                parent._reserved_bytes -= self.bytes_upper
                parent._children_work += self.work_units + self._children_work
                parent._children_queries += self.geometry_queries + self._children_queries
                parent._observe_capacity()
            if parent._parent is not None:
                parent._parent._observe_descendant_capacity()

    def refuse(
        self,
        work_upper: int,
        /,
        *,
        queries_upper: int = 0,
        bytes_upper: int = 0,
        cavity_upper: int = 0,
    ) -> NoReturn:
        requested = (
            _bound(work_upper, "work_upper"),
            _bound(queries_upper, "queries_upper"),
            _bound(cavity_upper, "cavity_upper"),
            _bound(bytes_upper, "bytes_upper"),
        )
        evidence = (
            self.remaining_work_units,
            self.remaining_geometry_queries,
            self.cavity_upper - self._own_cavity - self._reserved_cavity,
            self.bytes_upper - self._own_bytes - self._reserved_bytes,
            self.work_units + self._children_work,
            self.geometry_queries + self._children_queries,
            self.cavity_cells,
            self.managed_bytes,
        )
        self.settle()
        raise HostBatchEstimateError(requested, evidence)

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._settled:
            self.settle()
