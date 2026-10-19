#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Reservation isolation, local accounting and coordinate-domain settlement."""

import ctypes
import gc
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

import numpy as np
import pytest

from phydrax._meshcore import (
    _ClipReferenceTetrahedronExactRequest,
    current_native_execution_scope,
    exact_orient3d,
    load_meshcore,
    MeshcoreError,
    MeshcoreLibrary,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionLimits,
    NativeExecutionScope,
    surface_envelope,
    TetMesh3D,
)
from phydrax._resource_ticket import BatchTicket, ManagedBytesContributor
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureResourceError,
)


def _scope(work: int = 100) -> NativeExecutionScope:
    return NativeExecutionScope(NativeExecutionLimits(work, 100, 20, 65536, math.inf))


class _TicketAttachment:
    def __init__(self, scope: NativeExecutionScope, *, fail: bool = False) -> None:
        self.ticket = scope.batch(4, bytes_upper=1024)
        self.ticket.consume(3, managed_bytes=64)
        self.fail = fail

    def close(self) -> None:
        self.ticket.settle()
        if self.fail:
            raise RuntimeError(
                "attachment_fixture.cleanup must succeed; got attachment cleanup failure"
            )


def test_preparation_attachment_settles_before_scope_end() -> None:
    scope = _scope()
    with scope:
        attachment = _TicketAttachment(scope)

        def factory() -> _TicketAttachment:
            return attachment

        assert scope.preparation_ledger(factory) is attachment
    assert attachment.ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3
    assert scope.evidence.host_storage_live_bytes_upper == 0


@pytest.mark.parametrize("cleanup_fails", (False, True))
def test_preparation_attachment_cleanup_preserves_primary_exception(
    cleanup_fails: bool,
) -> None:
    scope = _scope()
    primary = ValueError(
        "native_ticket_fixture.request must succeed; got injected failure"
    )
    with pytest.raises(ValueError) as raised:
        with scope:
            attachment = _TicketAttachment(scope, fail=cleanup_fails)

            def factory() -> _TicketAttachment:
                return attachment

            scope.preparation_ledger(factory)
            raise primary
    assert raised.value is primary
    assert attachment.ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3
    assert scope.evidence.host_storage_live_bytes_upper == 0
    if cleanup_fails:
        assert primary.__notes__ == [
            "Native preparation attachment cleanup also failed: "
            "attachment_fixture.cleanup must succeed; got attachment cleanup failure"
        ]


def test_preparation_attachment_cleanup_failure_ends_scope_before_raising() -> None:
    scope = _scope()
    with pytest.raises(RuntimeError, match="attachment cleanup failure"):
        with scope:
            attachment = _TicketAttachment(scope, fail=True)

            def factory() -> _TicketAttachment:
                return attachment

            scope.preparation_ledger(factory)
    assert attachment.ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3


def test_scope_retains_lowered_limits_record_without_rebuilding() -> None:
    limits = NativeExecutionLimits(5, 0, 0, 4096, math.inf)
    scope = NativeExecutionScope(limits)
    assert scope.limits is limits
    with scope:
        with NativeExecutionScope(scope.limits) as child:
            assert child.limits is limits


def test_request_clock_is_shared_and_prior_time_debits_all_nested_views() -> None:
    scope = _scope()
    with pytest.raises(RuntimeError, match="scope entry"):
        _ = scope.operation_started
    before = time.monotonic()
    with scope:
        started = scope.operation_started
        assert before <= started <= time.monotonic()
        with _scope() as child:
            assert child.operation_started == started
            child.import_preparation(work=0, geometry_queries=0, elapsed_seconds=2.0)
            assert scope.operation_started == child.operation_started == started - 2.0
        scope.import_preparation(work=0, geometry_queries=0, elapsed_seconds=1.0)
        assert scope.operation_started == child.operation_started == started - 3.0
    assert scope.operation_started == started - 3.0
    assert scope.evidence is not None
    assert scope.evidence.prior_elapsed_seconds == 3.0
    assert child.evidence is not None
    assert child.evidence.prior_elapsed_seconds == 2.0


def test_refused_preparation_import_does_not_debit_request_clock() -> None:
    scope = _scope(0)
    with pytest.raises(MeshcoreError):
        with scope:
            started = scope.operation_started
            with pytest.raises(MeshcoreError):
                scope.import_preparation(work=1, geometry_queries=0, elapsed_seconds=1.0)
            assert scope.operation_started == started
    assert scope.evidence is not None
    assert scope.evidence.prior_elapsed_seconds == 0.0


def test_batch_reserves_complete_upper_and_releases_unused() -> None:
    scope = _scope()
    with scope:
        with scope.batch(
            60, queries_upper=50, cavity_upper=8, bytes_upper=4096
        ) as ticket:
            assert not ticket.settled
            remaining = scope.remaining()
            assert remaining.remaining_work_units == 40
            assert remaining.remaining_geometry_queries == 50
            assert remaining.remaining_cavity_cells == 12
            assert remaining.remaining_managed_bytes <= 65536 - 4096
            ticket.consume(7, geometry_queries=3, cavity_cells=4, managed_bytes=512)
        assert ticket.settled
        remaining = scope.remaining()
        assert remaining.remaining_work_units == 93
        assert remaining.remaining_geometry_queries == 97
        assert remaining.remaining_cavity_cells == 20
    assert scope.evidence is not None
    counters = scope.evidence.work_counters
    assert (
        counters.construction_work,
        counters.source_queries,
        counters.peak_cavity_cells,
    ) == (7, 3, 4)
    assert scope.evidence.externally_charged_work == 7


def test_batch_refuses_before_dispatch_and_retains_consumed_evidence() -> None:
    scope = _scope(10)
    dispatched: list[str] = []
    with pytest.raises(MeshcoreError) as refusal:
        with scope:
            with scope.batch(4) as first:
                first.consume(3)
            with scope.batch(8):
                dispatched.append("inadmissible")
    assert not dispatched
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert refusal.value.batch_requested_upper == (8, 0, 0, 0)
    assert refusal.value.batch_refusal_evidence is not None
    assert refusal.value.batch_refusal_evidence[0] == 7
    assert refusal.value.batch_refusal_evidence[4] == 3
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3


def test_nested_tickets_reserve_parent_not_root_and_settle_once() -> None:
    scope = _scope(20)
    with scope:
        with scope.batch(15, queries_upper=10) as parent:
            parent.consume(2)
            with parent.batch(8, queries_upper=6) as child:
                assert scope.remaining().remaining_work_units == 5
                assert parent.remaining_work_units == 5
                child.consume(3, geometry_queries=2)
            assert parent.remaining_work_units == 10
            parent.consume(4, geometry_queries=1)
        assert scope.remaining().remaining_work_units == 11
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 9
    assert scope.evidence.work_counters.source_queries == 3


def test_nested_capacity_peaks_include_overlap_and_release_live_capacity() -> None:
    scope = _scope()
    with scope:
        with scope.batch(0, cavity_upper=20, bytes_upper=4096) as parent:
            parent.consume(cavity_cells=5, managed_bytes=512)
            with parent.batch(0, cavity_upper=10, bytes_upper=1024) as child:
                child.consume(cavity_cells=7, managed_bytes=128)
                assert parent.cavity_cells == 12
                assert parent.managed_bytes == 640
                parent.consume()
                assert parent.cavity_cells == 12
            parent.consume(cavity_cells=0, managed_bytes=0)
            with parent.batch(0, cavity_upper=20, bytes_upper=4096) as second:
                second.consume(cavity_cells=18, managed_bytes=2048)
            assert parent.cavity_cells == 18
            assert parent.managed_bytes == 2048
    assert scope.evidence is not None
    assert scope.evidence.work_counters.peak_cavity_cells == 18


def test_exception_settles_measured_work_without_replacing_primary_error() -> None:
    scope = _scope()
    primary = ValueError("native_ticket_fixture.batch must succeed; got injected failure")
    with scope:
        with pytest.raises(ValueError) as raised:
            with scope.batch(80) as ticket:
                assert not ticket.settled
                ticket.consume(9)
                raise primary
        assert raised.value is primary
        assert ticket.settled
        assert scope.remaining().remaining_work_units == 91
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 9


def test_estimate_defect_is_internal_and_releases_reservation() -> None:
    scope = _scope(10)
    with scope:
        with pytest.raises(RuntimeError, match="estimate defect"):
            with scope.batch(5) as ticket:
                ticket.consume(3)
                ticket.consume(3)
        assert scope.remaining().remaining_work_units == 7
        with pytest.raises(RuntimeError, match="settled"):
            ticket.consume(1)


@pytest.mark.parametrize(
    ("value", "error"),
    (
        (True, TypeError),
        (np.bool_(False), TypeError),
        (1.5, TypeError),
        (-1, ValueError),
        (1 << 64, ValueError),
    ),
)
def test_public_ticket_counter_validation_preserves_completed_prefix(
    value: int | float, error: type[Exception]
) -> None:
    scope = _scope()
    with scope:
        with pytest.raises(error):
            scope.batch(value)  # ty: ignore[invalid-argument-type]
        with pytest.raises(error):
            scope.batch(5, bytes_upper=value)  # ty: ignore[invalid-argument-type]
        with scope.batch(5, queries_upper=5, cavity_upper=5, bytes_upper=1024) as ticket:
            owner = ticket.managed_storage()
            ticket.consume(2, geometry_queries=1, cavity_cells=3, managed_bytes=64)
            owner.set_bound(32)
            with pytest.raises(error):
                ticket.consume(value)  # ty: ignore[invalid-argument-type]
            with pytest.raises(error):
                ticket.consume(geometry_queries=value)  # ty: ignore[invalid-argument-type]
            with pytest.raises(error):
                ticket.consume(cavity_cells=value)  # ty: ignore[invalid-argument-type]
            with pytest.raises(error):
                ticket.consume(managed_bytes=value)  # ty: ignore[invalid-argument-type]
            with pytest.raises(error):
                ticket.set_managed_storage(owner, value)  # ty: ignore[invalid-argument-type]
            assert not ticket.settled
            assert (ticket.work_units, ticket.geometry_queries) == (2, 1)
            assert (ticket.cavity_cells, ticket.managed_bytes) == (3, 96)
            assert owner.bound == 32
            owner.close()


def test_explicit_child_tickets_transfer_to_deferred_workers() -> None:
    scope = _scope(20)
    with scope:
        with scope.batch(18) as parent:
            children = tuple(parent.batch(6) for _ in range(3))

            def execute(ticket: NativeBatchTicket) -> int:
                with ticket:
                    ticket.consume(4)
                    return ticket.work_units

            with ThreadPoolExecutor(max_workers=3) as executor:
                assert tuple(executor.map(execute, children)) == (4, 4, 4)
            assert parent.remaining_work_units == 6
        assert scope.remaining().remaining_work_units == 8
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 12


def test_concurrent_root_tickets_cannot_over_admit() -> None:
    scope = _scope(10)
    barrier = threading.Barrier(2)
    released = threading.Event()
    outcomes: list[MeshcoreStatus] = []

    def execute() -> None:
        barrier.wait()
        try:
            with scope.batch(7) as ticket:
                outcomes.append(MeshcoreStatus.OK)
                released.wait(timeout=5)
                ticket.consume(2)
        except MeshcoreError as refusal:
            outcomes.append(refusal.status)
            released.set()

    with pytest.raises(MeshcoreError):
        with scope:
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = tuple(executor.submit(execute) for _ in range(2))
                for future in futures:
                    future.result(timeout=10)
    assert sorted(outcomes) == [MeshcoreStatus.OK, MeshcoreStatus.CAPACITY_EXCEEDED]
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 2


def test_coordinate_ticket_settles_before_admission_and_on_exception_exit() -> None:
    scope = _scope(10)
    with scope:
        with scope.batch(10, bytes_upper=4096) as ticket:
            domain = CoordinateEnclosureBudget(
                ticket, maximum_work_units=10, maximum_memory_bytes=4096
            )
            primary = ValueError(
                "native_ticket_fixture.domain must succeed; got injected failure"
            )
            with pytest.raises(ValueError) as raised:
                with domain.activate():
                    domain.reserve(3, 128)
                    domain.admit_work_bound(4)
                    assert domain.native_charged_work_units == 3
                    assert ticket.work_units == 3
                    domain.reserve(2)
                    raise primary
            assert raised.value is primary
            assert domain.native_charged_work_units == domain.work_units == 5
            assert domain.required_work_units == 7
            assert ticket.work_units == 5
        assert scope.remaining().remaining_work_units == 5


def test_coordinate_domain_refusal_preserves_exact_counters() -> None:
    scope = _scope(10)
    with scope:
        with scope.batch(10, bytes_upper=4096) as ticket:
            domain = CoordinateEnclosureBudget(
                ticket, maximum_work_units=10, maximum_memory_bytes=4096
            )
            with domain.activate(), domain.bound_stage(5, 4096):
                domain.reserve(3)
                with pytest.raises(CoordinateEnclosureResourceError) as refused:
                    domain.admit_work_bound(3)
                assert refused.value.resource == "coefficient_work"
                assert refused.value.requested == 6
                assert refused.value.completed == 3
            assert ticket.work_units == 3


@pytest.mark.parametrize(
    ("work", "storage"), ((6, 0), (0, 4097)), ids=("work", "storage")
)
def test_coordinate_domain_limit_refusal_preserves_ticket_estimate_counters(
    work: int, storage: int
) -> None:
    scope = _scope(10)
    with scope:
        with scope.batch(5, bytes_upper=4096) as ticket:
            domain = CoordinateEnclosureBudget(
                ticket, maximum_work_units=5, maximum_memory_bytes=4096
            )
            with domain.activate():
                domain.reserve(3)
                with pytest.raises(CoordinateEnclosureResourceError) as raised:
                    domain.reserve(work, storage)
                assert domain.work_units == domain.native_charged_work_units == 3
                assert domain.required_work_units == 3 + work
                assert domain.peak_bytes_upper == 0
                assert ticket.work_units == 3
                assert raised.value.resource == (
                    "coefficient_work" if work else "polynomial_storage"
                )
                assert raised.value.limit == (5 if work else 4096)
                assert raised.value.requested == (3 + work if work else storage)
                assert not ticket.settled


def _native_mesh_work() -> None:
    mesh = TetMesh3D(
        np.asarray(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.zeros((1,), dtype=np.int32),
        np.asarray(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)), dtype=np.int32),
        np.arange(4, dtype=np.int32),
        np.asarray(((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)), dtype=np.int32),
        np.arange(6, dtype=np.int32),
        boundary_policy="conforming",
    )
    mesh.close()


def _target_scope() -> NativeExecutionScope:
    return NativeExecutionScope(
        NativeExecutionLimits(10000, 100, 100, 1024 * 1024, math.inf)
    )


def _protocol_charge(ticket: BatchTicket, /) -> ManagedBytesContributor:
    assert ticket.native_accounting is True
    ticket.consume(2, geometry_queries=1)
    contributor = ticket.managed_storage()
    contributor.set_bound(64)
    return contributor


def _bookkeeping(scope: NativeExecutionScope, /) -> int:
    value = ctypes.c_uint64()
    status = load_meshcore()["phx_mc_execution_allocator_bookkeeping"](
        scope._handle, ctypes.addressof(value)
    )
    assert MeshcoreStatus(status) is MeshcoreStatus.OK
    return value.value


def test_native_target_mixed_counters_are_immediate_without_counter_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _target_scope()
    lookups: list[str] = []
    original = MeshcoreLibrary.__getitem__

    def observed(library: MeshcoreLibrary, name: str, /) -> Callable[..., object]:
        lookups.append(name)
        return original(library, name)

    monkeypatch.setattr(MeshcoreLibrary, "__getitem__", observed)
    with scope:
        with scope.batch(9000, queries_upper=80, bytes_upper=900000) as ticket:
            with ticket.native_target() as target:
                assert target is None
                _native_mesh_work()
                native = ticket.native_work_units
                assert native > 0
                scope.charge(work=3, geometry_queries=2)
                for iteration in range(1, 9):
                    lookups.clear()
                    ticket.consume(2, geometry_queries=1)
                    assert ticket.work_units == native + 3 + 2 * iteration
                    assert ticket.consumed_work_units == ticket.work_units
                    assert ticket.native_work_units == native
                    assert ticket.geometry_queries == 2 + iteration
                    assert ticket.consumed_geometry_queries == 2 + iteration
                    assert ticket.native_geometry_queries == 0
                    assert ticket.remaining_work_units == 9000 - ticket.work_units
                    assert ticket.remaining_geometry_queries == 80 - 2 - iteration
                    assert lookups == []
            total = ticket.work_units
        assert ticket.work_units == ticket.consumed_work_units == total
        assert ticket.native_work_units == native
        assert scope.remaining().remaining_work_units == 10000 - total
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == total
    assert scope.evidence.externally_charged_work == 19
    assert scope.evidence.externally_charged_geometry_queries == 10


def test_same_ticket_native_target_join_preserves_outer_without_ffi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _scope()
    lookups: list[str] = []
    original = MeshcoreLibrary.__getitem__

    def observed(library: MeshcoreLibrary, name: str, /) -> Callable[..., object]:
        lookups.append(name)
        return original(library, name)

    monkeypatch.setattr(MeshcoreLibrary, "__getitem__", observed)
    primary = ValueError(
        "native_ticket_fixture.joined must succeed; got injected failure"
    )
    with scope:
        with scope.batch(10) as ticket:
            outer = ticket.native_target()
            with outer:
                joined = ticket.native_target()
                lookups.clear()
                with joined as value:
                    assert value is None
                assert lookups == []
                with pytest.raises(RuntimeError, match="single-use"):
                    with joined:
                        pass
                with pytest.raises(ValueError) as raised:
                    with ticket.native_target():
                        raise primary
                assert raised.value is primary
                assert current_native_execution_scope() is scope
                scope.charge(work=2)
                assert ticket.work_units == 2
            with pytest.raises(RuntimeError, match="single-use"):
                with outer:
                    pass
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 2


def test_different_ticket_native_target_nests_and_restores_current_owner() -> None:
    scope = _scope()
    with scope:
        with scope.batch(10) as parent:
            with parent.native_target():
                with parent.batch(5) as child:
                    with child.native_target():
                        scope.charge(work=2)
                        assert child.work_units == 2
                        assert parent.work_units == 0
                    assert current_native_execution_scope() is scope
                scope.charge(work=1)
                assert parent.work_units == 1
                assert parent.consumed_work_units == 3
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3


def test_native_target_nested_exception_restores_parent_owner() -> None:
    scope = _target_scope()
    primary = ValueError(
        "native_ticket_fixture.nested must succeed; got injected failure"
    )
    with scope:
        with scope.batch(9000, queries_upper=80, bytes_upper=900000) as parent:
            with parent.native_target():
                scope.charge(work=2)
                with parent.batch(4000, queries_upper=30, bytes_upper=400000) as child:
                    with pytest.raises(ValueError) as raised:
                        with child.native_target() as target:
                            assert target is None
                            _native_mesh_work()
                            scope.charge(work=3, geometry_queries=2)
                            raise primary
                    assert raised.value is primary
                    assert current_native_execution_scope() is scope
                    child_total = child.work_units
                    child_native = child.native_work_units
                scope.charge(work=4, geometry_queries=1)
                assert parent.work_units == 6
                assert parent.consumed_work_units == 6 + child_total
                assert parent.native_work_units == child_native
                assert parent.geometry_queries == 1
                assert parent.consumed_geometry_queries == 3
            scope.charge(work=5)
            assert parent.work_units == 6
        assert current_native_execution_scope() is scope
    assert current_native_execution_scope() is None
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 11 + child_total
    assert scope.evidence.externally_charged_work == 14


def test_root_native_target_rejects_worker_without_transferring_ambient_scope() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(100, bytes_upper=4096) as root:

            def execute() -> None:
                assert current_native_execution_scope() is None
                with pytest.raises(RuntimeError):
                    with root.native_target():
                        pytest.fail("root target entered on foreign thread")
                with pytest.raises(RuntimeError):
                    scope.allocate_host_array((1,), np.uint8)
                assert current_native_execution_scope() is None

            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(execute).result(timeout=10)
            assert root.work_units == 0
            with root.native_target():
                scope.charge(work=1)
            assert root.work_units == 1


def test_explicit_child_native_target_allows_worker_scope_and_restores_ambient() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(9000, queries_upper=80, bytes_upper=900000) as parent:
            child = parent.batch(4000, queries_upper=30, bytes_upper=400000)

            def execute() -> tuple[int, int]:
                assert current_native_execution_scope() is None
                with pytest.raises(RuntimeError):
                    scope.allocate_host_array((1,), np.uint8)
                with child:
                    with child.native_target():
                        assert current_native_execution_scope() is scope
                        array = scope.allocate_host_array((8,), np.float64)
                        array[:] = 2.0
                        _native_mesh_work()
                        child.consume(3, geometry_queries=2)
                        native = child.native_work_units
                        assert child.work_units == native + 3
                        del array
                    assert current_native_execution_scope() is None
                return child.work_units, native

            with ThreadPoolExecutor(max_workers=1) as executor:
                total, native = executor.submit(execute).result(timeout=10)
            assert current_native_execution_scope() is scope
            assert parent.work_units == 0
            assert parent.consumed_work_units == total
            assert parent.native_work_units == native
            assert parent.remaining_work_units == 9000 - total
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == total
    assert scope.evidence.externally_charged_work == 3


def test_native_target_foreign_host_write_is_rejected_before_mapped_store() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(20, queries_upper=10, bytes_upper=4096) as ticket:
            with ticket.native_target():
                ticket.consume(3, geometry_queries=2, managed_bytes=64)
                contributor = ticket.managed_storage()
                contributor.set_bound(32)

                def execute() -> None:
                    with pytest.raises(RuntimeError):
                        ticket.consume(1, geometry_queries=1, managed_bytes=128)
                    with pytest.raises(RuntimeError):
                        contributor.set_bound(64)

                with ThreadPoolExecutor(max_workers=1) as executor:
                    executor.submit(execute).result(timeout=10)
                assert ticket.work_units == 3
                assert ticket.geometry_queries == 2
                assert ticket.managed_bytes == 96
                assert contributor.bound == 32
                contributor.close()


def test_child_scope_under_target_has_phase_receipt_and_counts_once() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(9000, queries_upper=80, bytes_upper=900000) as parent:
            with parent.native_target():
                scope.charge(work=2)
                with _target_scope() as child_scope:
                    assert child_scope.parent is scope
                    _native_mesh_work()
                    with child_scope.batch(10, queries_upper=5) as child_ticket:
                        assert child_ticket.parent is parent
                        child_ticket.consume(3, geometry_queries=1)
                    child_scope.charge(work=4, geometry_queries=2)
                assert child_scope.evidence is not None
                native = parent.native_work_units
                assert native > 0
                assert child_scope.evidence.work_counters.construction_work == native + 7
                assert child_scope.evidence.work_counters.source_queries == 3
                assert child_scope.evidence.externally_charged_work == 7
                assert parent.work_units == native + 6
                assert parent.consumed_work_units == native + 9
                assert parent.consumed_geometry_queries == 3
                scope.charge(work=5)
            total = parent.consumed_work_units
        assert scope.remaining().remaining_work_units == 10000 - total
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == native + 14
    assert scope.evidence.externally_charged_work == 14


def test_native_target_host_arrays_draw_reservation_and_free_credit() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(0, bytes_upper=4096) as ticket:
            outside = scope.remaining().remaining_managed_bytes
            with ticket.native_target():
                before = ticket.remaining_managed_bytes
                array = scope.allocate_host_array((128,), np.uint8)
                array[:] = 7
                live = ticket.remaining_managed_bytes
                assert live <= before - array.nbytes
                assert scope.remaining().remaining_managed_bytes == outside
                assert ticket.managed_bytes >= array.nbytes
                assert not array.flags.owndata
                del array
                gc.collect()
                assert ticket.remaining_managed_bytes == before
                assert scope.remaining().remaining_managed_bytes == outside
            assert scope.remaining().remaining_managed_bytes == outside
        assert ticket.managed_bytes >= 128
    assert scope.evidence is not None
    assert scope.evidence.memory_counters.live_bytes == 0
    assert scope.evidence.memory_counters.peak_bytes >= 128


def test_native_target_escaped_array_retains_child_capacity_until_free() -> None:
    scope = _target_scope()
    with scope:
        with scope.batch(0, bytes_upper=8192) as parent:
            outside = scope.remaining().remaining_managed_bytes
            with parent.batch(0, bytes_upper=4096) as child:
                with child.native_target():
                    array = scope.allocate_host_array((128,), np.float64)
                    array[:] = 3.0
            assert child.settled
            with parent.native_target():
                retained = parent.remaining_managed_bytes
                assert retained <= 8192 - array.nbytes
                assert parent.managed_bytes >= array.nbytes
                assert scope.remaining().remaining_managed_bytes == outside
                del array
                gc.collect()
                assert parent.remaining_managed_bytes == 8192
                assert scope.remaining().remaining_managed_bytes == outside
            assert parent.managed_bytes >= 1024
        with scope.batch(0, bytes_upper=4096) as escaping:
            with escaping.native_target():
                escaped = scope.allocate_host_array((4,), np.int64)
                escaped[:] = (1, 2, 3, 4)
    np.testing.assert_array_equal(escaped, (1, 2, 3, 4))
    del escaped
    gc.collect()


def test_native_target_zero_bytes_is_hard_zero_not_scope_fallback() -> None:
    scope = _target_scope()
    with pytest.raises(MeshcoreError):
        with scope:
            with scope.batch(0, bytes_upper=0) as ticket:
                with pytest.raises(MeshcoreError) as raised:
                    with ticket.native_target():
                        scope.allocate_host_array((1,), np.uint8)
                assert raised.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                assert raised.value.batch_requested_upper is not None
                assert raised.value.batch_requested_upper[:3] == (0, 0, 0)
                assert raised.value.batch_requested_upper[3] > 0
                assert raised.value.batch_refusal_evidence is not None
                assert raised.value.batch_refusal_evidence.shape == (8,)
                assert raised.value.batch_refusal_evidence[3] == 0
            assert ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.memory_counters.live_bytes == 0


@pytest.mark.parametrize(
    ("resource", "upper_request"),
    (
        ("work", (8, 0, 0, 0)),
        ("queries", (0, 99, 0, 0)),
        ("cavity", (0, 0, 21, 0)),
        ("bytes", (0, 0, 0, 65537)),
    ),
)
def test_atomic_refuse_matches_real_failed_admission_snapshot(
    resource: str,
    upper_request: tuple[int, int, int, int],
) -> None:
    def refuse(atomic: bool, /) -> tuple[MeshcoreError, NativeExecutionScope]:
        scope = _scope(10)
        with pytest.raises(MeshcoreError):
            with scope:
                baseline = _bookkeeping(scope)
                ticket = scope.batch(5, queries_upper=5, cavity_upper=6, bytes_upper=512)
                ticket.consume(3, geometry_queries=2, cavity_cells=4, managed_bytes=128)
                with pytest.raises(MeshcoreError) as raised:
                    if atomic:
                        ticket.refuse(
                            upper_request[0],
                            queries_upper=upper_request[1],
                            cavity_upper=upper_request[2],
                            bytes_upper=upper_request[3],
                        )
                    else:
                        ticket.settle()
                        scope.batch(
                            upper_request[0],
                            queries_upper=upper_request[1],
                            cavity_upper=upper_request[2],
                            bytes_upper=upper_request[3],
                        )
                assert ticket.settled
                assert _bookkeeping(scope) == baseline
        return raised.value, scope

    atomic, atomic_scope = refuse(True)
    admission, admission_scope = refuse(False)
    assert resource in ("work", "queries", "cavity", "bytes")
    assert atomic.status is admission.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert (
        atomic.batch_requested_upper == admission.batch_requested_upper == upper_request
    )
    assert atomic.batch_refusal_evidence is not None
    assert admission.batch_refusal_evidence is not None
    assert atomic.batch_refusal_evidence.shape == (8,)
    assert not atomic.batch_refusal_evidence.flags.writeable
    np.testing.assert_array_equal(
        atomic.batch_refusal_evidence, admission.batch_refusal_evidence
    )
    assert atomic_scope.evidence is not None
    assert admission_scope.evidence is not None
    assert (
        atomic_scope.evidence.work_counters.items()
        == admission_scope.evidence.work_counters.items()
    )
    counters = atomic_scope.evidence.work_counters
    assert (
        counters.construction_work,
        counters.source_queries,
        counters.peak_cavity_cells,
    ) == (3, 2, 4)
    assert atomic_scope.evidence.memory_counters.live_bytes == 0


@pytest.mark.parametrize(
    "upper_request", ((6, 0, 0, 0), (0, 0, 0, 4097)), ids=("work", "storage")
)
def test_atomic_refuse_fitting_request_is_nonmutating_estimate_defect(
    upper_request: tuple[int, int, int, int],
) -> None:
    scope = _scope(10)
    with scope:
        with scope.batch(5, bytes_upper=4096) as ticket:
            ticket.consume(3)
            before = scope.remaining()
            bookkeeping = _bookkeeping(scope)
            with pytest.raises(RuntimeError) as raised:
                ticket.refuse(
                    upper_request[0],
                    queries_upper=upper_request[1],
                    cavity_upper=upper_request[2],
                    bytes_upper=upper_request[3],
                )
            assert not isinstance(raised.value, MeshcoreError)
            assert not ticket.settled
            assert ticket.work_units == 3
            assert ticket.managed_bytes == 0
            assert ticket.remaining_work_units == 2
            assert _bookkeeping(scope) == bookkeeping
            after = scope.remaining()
            assert after.status is MeshcoreStatus.OK
            assert after.remaining_work_units == before.remaining_work_units
            assert after.remaining_managed_bytes == before.remaining_managed_bytes
            ticket.consume(1)
        assert scope.remaining().remaining_work_units == 6
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 4


def test_managed_contributors_aggregate_default_and_descendant_capacity() -> None:
    scope = _scope()
    with scope:
        with scope.batch(0, bytes_upper=4096) as parent:
            first = parent.managed_storage()
            second = parent.managed_storage()
            parent.consume(managed_bytes=100)
            first.set_bound(200)
            second.set_bound(300)
            assert first.bound == 200
            assert second.bound == 300
            assert parent.managed_bytes == 600
            with parent.batch(0, bytes_upper=1024) as child:
                descendant = child.managed_storage()
                child.consume(managed_bytes=50)
                descendant.set_bound(150)
                assert child.managed_bytes == 200
                assert parent.managed_bytes == 800
                descendant.close()
                assert descendant.bound == 0
                child.consume(managed_bytes=0)
            first.close()
            second.set_bound(0)
            second.close()
            parent.consume(managed_bytes=0)
            with parent.native_target():
                assert parent.remaining_managed_bytes == 4096
                assert scope.remaining().remaining_managed_bytes == 65536 - 4096
            assert parent.managed_bytes == 800


def test_managed_aggregate_survives_mapping_and_reads_live_native_capacity() -> None:
    scope = _scope()
    with scope:
        with scope.batch(5, bytes_upper=4096) as ticket:
            first = ticket.managed_storage()
            second = ticket.managed_storage()
            ticket.consume(managed_bytes=np.int64(32))  # ty: ignore[invalid-argument-type]
            first.set_bound(64)
            second.set_bound(96)
            assert ticket._usage.host_bytes == 192
            with ticket.native_target():
                array = scope.allocate_host_array((128,), np.uint8)
                array[:] = 7
                available = ticket.remaining_managed_bytes
                ticket.consume(np.int64(1), managed_bytes=np.int64(48))  # ty: ignore[invalid-argument-type]
                assert ticket.remaining_managed_bytes == available - 16
                first.set_bound(32)
                assert ticket._usage.host_bytes == 176
                assert ticket.remaining_managed_bytes == available + 16
                with pytest.raises(RuntimeError, match="estimate defect"):
                    ticket.consume(1, managed_bytes=4096)
                assert ticket.work_units == 1
                assert ticket._usage.host_bytes == 176
                np.testing.assert_array_equal(array, np.full((128,), 7, dtype=np.uint8))
                del array
                gc.collect()
                assert ticket.remaining_managed_bytes == 4096 - 176
            first.close()
            second.close()
            ticket.consume(managed_bytes=0)
            assert ticket.remaining_managed_bytes == 4096
            assert ticket.work_units == 1
            assert ticket.managed_bytes >= 208 + 128
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 1
    assert scope.evidence.host_storage_live_bytes_upper == 0


@pytest.mark.parametrize("target_active", (False, True), ids=("host", "target"))
def test_managed_contributor_overrun_refuses_aggregate_without_probe_leak(
    target_active: bool,
) -> None:
    scope = _scope()
    with pytest.raises(MeshcoreError):
        with scope:
            baseline = _bookkeeping(scope)
            with scope.batch(5, bytes_upper=1024) as ticket:
                ticket.consume(3, managed_bytes=128)
                first = ticket.managed_storage()
                first.set_bound(256)
                second = ticket.managed_storage()
                target = ticket.native_target() if target_active else nullcontext()
                with target:
                    with pytest.raises(MeshcoreError) as raised:
                        second.set_bound(65536)
                assert raised.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                assert raised.value.batch_requested_upper == (0, 0, 0, 65920)
                assert raised.value.batch_refusal_evidence is not None
                assert raised.value.batch_refusal_evidence.shape == (8,)
                assert raised.value.batch_refusal_evidence[4] == 3
                assert ticket.settled
                assert ticket.work_units == 3
                assert ticket.managed_bytes == 384
                assert first.bound == 256
                assert second.bound == 0
                first.close()
                second.close()
                assert _bookkeeping(scope) == baseline
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3
    assert scope.evidence.host_storage_live_bytes_upper == 0


def test_managed_contributor_close_after_settle_is_bookkeeping_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scope = _scope()
    with scope:
        with scope.batch(5, queries_upper=5, bytes_upper=1024) as ticket:
            contributor = _protocol_charge(ticket)
            assert isinstance(contributor, ManagedBytesContributor)
            assert ticket.managed_bytes == 64
        assert ticket.native_accounting is True
        lookups: list[str] = []
        original = MeshcoreLibrary.__getitem__

        def observed(library: MeshcoreLibrary, name: str, /) -> Callable[..., object]:
            lookups.append(name)
            return original(library, name)

        monkeypatch.setattr(MeshcoreLibrary, "__getitem__", observed)
        with pytest.raises(RuntimeError):
            contributor.set_bound(65)
        contributor.set_bound(32)
        contributor.close()
        contributor.close()
        assert contributor.bound == 0
        assert ticket.work_units == 2
        assert ticket.managed_bytes == 64
        assert lookups == []
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 2


def test_managed_contributor_identity_cannot_cross_tickets() -> None:
    scope = _scope()
    with scope:
        with scope.batch(0, bytes_upper=1024) as first:
            contributor = first.managed_storage()
            contributor.set_bound(64)
            with scope.batch(0, bytes_upper=1024) as second:
                with pytest.raises(
                    ValueError, match="managed_storage must belong to this ticket"
                ):
                    second.set_managed_storage(contributor, 128)
                assert contributor.bound == 64
                assert first.managed_bytes == 64
                assert second.managed_bytes == 0
            contributor.close()


def test_native_accounting_is_read_only_protocol_capability() -> None:
    scope = _scope()
    with scope:
        with scope.batch(5, queries_upper=5, bytes_upper=1024) as ticket:
            assert ticket.native_accounting is True
            with pytest.raises(AttributeError):
                setattr(ticket, "native_accounting", False)
            contributor = _protocol_charge(ticket)
            contributor.close()
            assert ticket.work_units == 2
            assert ticket.geometry_queries == 1
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 2
    assert scope.evidence.work_counters.source_queries == 1


@pytest.mark.parametrize("resource", ("work", "queries", "cavity"))
def test_native_target_overrun_preserves_primary_and_exact_refusal(
    resource: str,
) -> None:
    scope = _target_scope()
    captured: list[MeshcoreError] = []
    with pytest.raises(MeshcoreError):
        with scope:
            with scope.batch(
                0 if resource == "work" else 10,
                queries_upper=0 if resource == "queries" else 5,
                cavity_upper=0 if resource == "cavity" else 10,
                bytes_upper=900000,
            ) as ticket:
                with pytest.raises(MeshcoreError) as raised:
                    with ticket.native_target():
                        try:
                            match resource:
                                case "work":
                                    _native_mesh_work()
                                case "queries":
                                    scope.charge(geometry_queries=1)
                                case "cavity":
                                    scope.admit_cavity(1)
                                case _:
                                    pytest.fail("unrecognized resource")
                        except MeshcoreError as primary:
                            captured.append(primary)
                            raise
                assert raised.value is captured[0]
                assert raised.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                assert raised.value.batch_requested_upper == (
                    1 if resource == "work" else 0,
                    1 if resource == "queries" else 0,
                    1 if resource == "cavity" else 0,
                    0,
                )
                evidence = raised.value.batch_refusal_evidence
                assert evidence is not None
                assert evidence.shape == (8,)
                assert not evidence.flags.writeable
                slot = {"work": 0, "queries": 1, "cavity": 2}[resource]
                assert evidence[slot] == 0
                assert current_native_execution_scope() is scope
            assert ticket.settled
            assert ticket.work_units == 0
    assert current_native_execution_scope() is None
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 0
    assert scope.evidence.memory_counters.live_bytes == 0


def test_native_target_failed_body_still_settles_host_and_native_once() -> None:
    scope = _target_scope()
    primary = ValueError("native_ticket_fixture.body must succeed; got injected failure")
    with scope:
        with pytest.raises(ValueError) as raised:
            with scope.batch(9000, bytes_upper=900000) as ticket:
                with ticket.native_target():
                    ticket.consume(3)
                    _native_mesh_work()
                    native = ticket.native_work_units
                    raise primary
        assert raised.value is primary
        assert ticket.settled
        assert ticket.work_units == native + 3
        assert ticket.native_work_units == native > 0
        assert current_native_execution_scope() is scope
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == native + 3
    assert scope.evidence.externally_charged_work == 3


@pytest.mark.parametrize(
    ("work", "storage"), ((8, 0), (0, 65537)), ids=("work", "storage")
)
def test_coordinate_smaller_native_envelope_refuses_with_exact_request(
    work: int,
    storage: int,
) -> None:
    scope = _scope(10)
    with pytest.raises(MeshcoreError):
        with scope:
            with scope.batch(5, bytes_upper=4096) as ticket:
                domain = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=20, maximum_memory_bytes=131072
                )
                with domain.activate():
                    domain.reserve(3)
                    with pytest.raises(MeshcoreError) as raised:
                        domain.reserve(work, storage)
                    assert raised.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                    assert raised.value.batch_requested_upper == (work, 0, 0, storage)
                    assert raised.value.batch_refusal_evidence is not None
                    assert raised.value.batch_refusal_evidence.shape == (8,)
                    assert raised.value.batch_refusal_evidence[4] == 3
                    assert domain.work_units == domain.native_charged_work_units == 3
                    assert domain.required_work_units == 3 + work
                    assert domain.peak_bytes_upper == 0
                    assert ticket.work_units == 3
                    assert ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 3


def test_native_source_query_and_work_counters_remain_native_at_settlement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    limits = NativeExecutionLimits(20000000, 1000000, 1000000, 64 * 1024 * 1024, math.inf)
    scope = NativeExecutionScope(limits)
    with scope:
        with scope.batch(
            10000000,
            queries_upper=1000000,
            cavity_upper=1000000,
            bytes_upper=64 * 1024 * 1024,
        ) as ticket:
            with ticket.native_target():
                surface_envelope(
                    np.asarray(
                        ((0.0, 0.0, 0.0), (0.2, 0.0, 0.0), (0.0, 0.2, 0.0)),
                        dtype=np.float64,
                    ),
                    np.asarray(((0, 1, 2),), dtype=np.int64),
                    offset=0.263,
                    spacing=0.11,
                    certificate_spacing=0.05,
                    max_samples=100000,
                    max_work_units=10000000,
                    max_vertices=10000,
                    max_triangles=20000,
                    max_volume_vertices=1000000,
                    max_tetrahedra=1000000,
                )
                native_work = ticket.native_work_units
                native_queries = ticket.native_geometry_queries
                assert native_work > 0
                assert native_queries > 0
                lookups: list[str] = []
                original = MeshcoreLibrary.__getitem__

                def observed(
                    library: MeshcoreLibrary, name: str, /
                ) -> Callable[..., object]:
                    lookups.append(name)
                    return original(library, name)

                with monkeypatch.context() as patch:
                    patch.setattr(MeshcoreLibrary, "__getitem__", observed)
                    for iteration in range(1, 5):
                        ticket.consume(1, geometry_queries=1)
                        assert ticket.work_units == native_work + iteration
                        assert ticket.geometry_queries == native_queries + iteration
                        assert ticket.native_work_units == native_work
                        assert ticket.native_geometry_queries == native_queries
                        assert (
                            ticket.remaining_work_units
                            == 10000000 - native_work - iteration
                        )
                        assert (
                            ticket.remaining_geometry_queries
                            == 1000000 - native_queries - iteration
                        )
                    assert lookups == []
        assert ticket.work_units == native_work + 4
        assert ticket.geometry_queries == native_queries + 4
        assert ticket.native_geometry_queries == native_queries
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == native_work + 4
    assert scope.evidence.work_counters.source_queries == native_queries + 4
    assert scope.evidence.externally_charged_work == 4
    assert scope.evidence.externally_charged_geometry_queries == 4


def test_managed_contributor_fitting_overrun_is_nonmutating_estimate_defect() -> None:
    scope = _scope()
    with scope:
        with scope.batch(0, bytes_upper=1024) as ticket:
            ticket.consume(managed_bytes=128)
            contributor = ticket.managed_storage()
            contributor.set_bound(256)
            excess = ticket.managed_storage()
            baseline = _bookkeeping(scope)
            with pytest.raises(RuntimeError) as raised:
                excess.set_bound(1024)
            assert not isinstance(raised.value, MeshcoreError)
            assert not ticket.settled
            assert contributor.bound == 256
            assert excess.bound == 0
            assert ticket.managed_bytes == 384
            assert _bookkeeping(scope) == baseline
            contributor.close()
            excess.close()
            ticket.consume(managed_bytes=0)
        assert scope.remaining().status is MeshcoreStatus.OK


def _native_clip_scratch() -> None:
    coefficients = np.asarray(
        (
            (-1, 0, 0, 0),
            (0, -1, 0, 0),
            (0, 0, -1, 0),
            (1, 1, 1, -1),
            (1, 0, 0, -2),
            (-1, 0, 0, -1),
            (0, 1, 0, -2),
            (0, -1, 0, -1),
            (0, 0, 1, -2),
            (0, 0, -1, -1),
        ),
        dtype=np.int64,
    ).reshape((40,))
    nonzero = coefficients != 0
    words = np.asarray(np.abs(coefficients[nonzero]), dtype=np.uint32)
    signs = np.asarray(np.sign(coefficients), dtype=np.int8)
    offsets = np.zeros((41,), dtype=np.int64)
    offsets[1:] = np.cumsum(nonzero, dtype=np.int64)
    supports = np.empty((16, 3), dtype=np.int32)
    incidence = np.empty((16,), dtype=np.uint16)
    face_offsets = np.zeros((11,), dtype=np.int32)
    face_planes = np.empty((10,), dtype=np.int32)
    face_vertices = np.empty((48,), dtype=np.int32)
    vertex_count = np.zeros((1,), dtype=np.int32)
    face_count = np.zeros((1,), dtype=np.int32)
    work = np.zeros((1,), dtype=np.int64)
    status = load_meshcore()["phx_mc_clip_reference_tetrahedron_exact"](
        ctypes.byref(
            _ClipReferenceTetrahedronExactRequest(
                words=words.ctypes.data,
                word_count=words.size,
                offsets=offsets.ctypes.data,
                signs=signs.ctypes.data,
                scratch_limit=900000,
                work_limit=9000,
                vertex_capacity=16,
                face_capacity=10,
                entry_capacity=48,
                supporting_planes=supports.ctypes.data,
                incident_planes=incidence.ctypes.data,
                face_offsets=face_offsets.ctypes.data,
                face_planes=face_planes.ctypes.data,
                face_vertices=face_vertices.ctypes.data,
                vertex_count=vertex_count.ctypes.data,
                face_count=face_count.ctypes.data,
                work_units=work.ctypes.data,
            )
        )
    )
    assert MeshcoreStatus(status) is MeshcoreStatus.OK
    assert vertex_count[0] == face_count[0] == 4
    assert work[0] > 0


@pytest.mark.parametrize("overlap", (False, True), ids=("disjoint", "simultaneous"))
def test_single_ffi_native_scratch_peak_combines_only_simultaneous_host_storage(
    overlap: bool,
) -> None:
    calibration = _target_scope()
    with calibration:
        with calibration.batch(9000, bytes_upper=900000) as native:
            with native.native_target():
                _native_clip_scratch()
    assert calibration.evidence is not None
    native_peak = int(calibration.evidence.memory_counters.peak_bytes)
    assert native_peak > 0
    assert calibration.evidence.memory_counters.live_bytes == 0
    assert native.managed_bytes == native_peak

    scope = _target_scope()
    with scope:
        with scope.batch(9000, bytes_upper=900000) as ticket:
            with ticket.native_target():
                contributor = ticket.managed_storage()
                contributor.set_bound(1024)
                if not overlap:
                    contributor.set_bound(0)
                # All native payload allocations and frees occur in this call.
                # No Python peak read observes its temporary live allocations.
                _native_clip_scratch()
                expected_peak = native_peak + 1024 if overlap else max(native_peak, 1024)
                assert ticket.managed_bytes == expected_peak
                assert ticket.remaining_managed_bytes == 900000 - (1024 if overlap else 0)
                contributor.close()
            assert ticket.managed_bytes == expected_peak
        assert ticket.managed_bytes == expected_peak
    assert scope.evidence is not None
    assert scope.evidence.memory_counters.live_bytes == 0
    assert scope.evidence.memory_counters.peak_bytes == native_peak


@pytest.mark.parametrize("live_bytes", (0, 128), ids=("empty", "live-storage"))
def test_coordinate_admission_refusal_preserves_error_and_releases_once(
    live_bytes: int,
) -> None:
    scope = _scope(10)
    captured: list[MeshcoreError] = []
    with pytest.raises(MeshcoreError) as surfaced:
        with scope:
            before = scope.remaining()
            bookkeeping = _bookkeeping(scope)
            with pytest.raises(MeshcoreError) as boundary:
                with scope.batch(10, bytes_upper=4096) as ticket:
                    budget = CoordinateEnclosureBudget(
                        ticket, maximum_work_units=20, maximum_memory_bytes=4096
                    )
                    with budget.activate(), budget.live_storage() as storage:
                        storage.set_bound(live_bytes)
                        try:
                            budget.reserve(11)
                        except MeshcoreError as error:
                            captured.append(error)
                            raise
            assert boundary.value is captured[0]
            assert boundary.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert boundary.value.batch_admitted_upper == (10, 0, 0, 4096)
            assert boundary.value.batch_requested_upper == (11, 0, 0, live_bytes)
            evidence = boundary.value.batch_refusal_evidence
            assert evidence is not None
            np.testing.assert_array_equal(evidence[:3], (10, 100, 20))
            np.testing.assert_array_equal(evidence[4:6], (0, 0))
            assert not evidence.flags.writeable
            assert ticket.settled
            assert ticket.work_units == budget.work_units == 0
            assert budget.temporary_bytes_upper == storage.bound == 0
            assert ticket.remaining_work_units == 10
            assert scope.remaining().remaining_work_units == before.remaining_work_units
            assert (
                scope.remaining().remaining_managed_bytes
                == before.remaining_managed_bytes
            )
            assert _bookkeeping(scope) == bookkeeping
            with pytest.raises(RuntimeError, match="settled"):
                ticket.settle()
            raise boundary.value
    assert surfaced.value is captured[0]
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 0
    assert scope.evidence.host_storage_live_bytes_upper == 0
    assert scope.evidence.memory_counters.live_bytes == 0


def test_native_work_cannot_spend_python_admitted_coordinate_work() -> None:
    scope = _target_scope()
    with pytest.raises(MeshcoreError):
        with scope:
            with scope.batch(10000, bytes_upper=900000) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=10000, maximum_memory_bytes=900000
                )
                with budget.activate():
                    budget.reserve(10000)
                    assert ticket.work_units == budget.work_units == 10000
                    assert ticket.remaining_work_units == 0
                    with pytest.raises(MeshcoreError) as refused:
                        with ticket.native_target():
                            _native_mesh_work()
                    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
                    assert refused.value.batch_admitted_upper == (10000, 0, 0, 900000)
                    assert refused.value.batch_requested_upper == (1, 0, 0, 0)
                    evidence = refused.value.batch_refusal_evidence
                    assert evidence is not None
                    assert evidence[0] == 0
                    assert evidence[4] == 10000
                    assert ticket.native_work_units == 0
                assert ticket.work_units == budget.native_charged_work_units == 10000
            assert ticket.settled
    assert scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 10000
    assert scope.evidence.externally_charged_work == 10000
    assert scope.evidence.host_storage_live_bytes_upper == 0


def test_scope_refusal_closes_entered_children_and_next_operation_starts_clean() -> None:
    scope = _scope(20)
    captured: list[MeshcoreError] = []
    receipts: list[np.ndarray] = []
    with pytest.raises(MeshcoreError) as surfaced:
        with scope:
            parent = scope.batch(20, bytes_upper=8192)
            parent_target = parent.native_target()
            parent_target.__enter__()
            child_scope = _scope(20)
            child_scope.__enter__()
            child = child_scope.batch(6, bytes_upper=1024)
            child_target = child.native_target()
            child_target.__enter__()
            child.consume(2, managed_bytes=128)
            try:
                child.refuse(30)
            except MeshcoreError as refusal:
                captured.append(refusal)
                receipt = refusal.batch_refusal_evidence
                assert receipt is not None
                receipts.append(receipt)
                raise
    assert surfaced.value is captured[0]
    assert surfaced.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert surfaced.value.batch_admitted_upper == (6, 0, 0, 1024)
    assert surfaced.value.batch_requested_upper == (30, 0, 0, 0)
    assert surfaced.value.batch_refusal_evidence is receipts[0]
    assert parent.settled and child.settled
    assert not scope._tickets and not child_scope._tickets
    assert not scope._target_stacks and not child_scope._target_stacks
    assert current_native_execution_scope() is None
    assert scope.evidence is not None
    assert child_scope.evidence is not None
    assert scope.evidence.work_counters.construction_work == 2
    assert scope.evidence.host_storage_live_bytes_upper == 0
    assert scope.evidence.memory_counters.live_bytes == 0
    # The lexical child boundary may still unwind after its owner has retired it.
    child_scope.__exit__(MeshcoreError, surfaced.value, None)
    corners = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )
    assert np.all(
        np.abs(exact_orient3d(*(corners[index : index + 1] for index in range(4)))) == 1
    )
    independent = _scope(20)
    with independent:
        baseline = _bookkeeping(independent)
        before = independent.remaining()
        assert before.status is MeshcoreStatus.OK
        assert before.remaining_work_units == 20
        with independent.batch(5, bytes_upper=1024) as fresh:
            fresh.consume(3)
        assert independent.remaining().remaining_work_units == 17
        assert (
            independent.remaining().remaining_managed_bytes
            == before.remaining_managed_bytes
        )
        assert _bookkeeping(independent) == baseline
    assert independent.evidence is not None
    assert independent.evidence.status is MeshcoreStatus.OK
    assert independent.evidence.work_counters.construction_work == 3


def test_parent_context_retires_retained_child_without_changing_success_counters() -> (
    None
):
    scope = _scope(20)
    with scope:
        baseline = _bookkeeping(scope)
        before = scope.remaining()
        with scope.batch(15, bytes_upper=4096) as parent:
            with parent.native_target():
                retained = scope.batch(0, bytes_upper=512)
                storage = retained.managed_storage()
                storage.set_bound(128)
                parent.consume(3)
            with pytest.raises(RuntimeError, match="live children"):
                parent.settle()
        assert parent.settled and retained.settled
        storage.close()
        assert scope.remaining().remaining_work_units == 17
        assert scope.remaining().remaining_managed_bytes == before.remaining_managed_bytes
        assert _bookkeeping(scope) == baseline
        with pytest.raises(RuntimeError, match="settled"):
            retained.settle()
    assert scope.evidence is not None
    assert scope.evidence.status is MeshcoreStatus.OK
    assert scope.evidence.work_counters.construction_work == 3
    assert scope.evidence.externally_charged_work == 3
    assert scope.evidence.host_storage_live_bytes_upper == 0
