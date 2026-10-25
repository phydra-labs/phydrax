#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coordinate work admission, storage lifetime, and phase-evidence contracts."""

from collections.abc import Iterator, Mapping
from contextlib import nullcontext, suppress
from fractions import Fraction
from math import inf
from time import sleep
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

from phydrax._host_storage import declared_object_storage_bytes
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    _coordinate_storage_children,
    CoordinateEnclosureBudget,
    CoordinateEnclosureMeasurement,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
    reserve_polynomial,
)
from phydrax.discretization._coordinate_enclosure import multiply, Polynomial
from phydrax.discretization._coordinate_operands import (
    CoordinateOperandProfile,
    CoordinateOperandProfileBuilder,
    PreparedCoordinateOperands,
)


@pytest.fixture
def budget() -> Iterator[CoordinateEnclosureBudget]:
    with HostBatchTicket(100, bytes_upper=8192, queries_upper=10) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=8192
        )
        with ledger.activate():
            yield ledger


@pytest.mark.parametrize(
    ("work", "storage", "field", "error"),
    (
        pytest.param(-1, 0, "work", ValueError, id="negative-work"),
        pytest.param(True, 0, "work", TypeError, id="boolean-work"),
        pytest.param(1.5, 0, "work", TypeError, id="fractional-work"),
        pytest.param(0, -1, "storage_upper", ValueError, id="negative-storage"),
        pytest.param(0, True, "storage_upper", TypeError, id="boolean-storage"),
        pytest.param(0, 1.5, "storage_upper", TypeError, id="fractional-storage"),
    ),
)
def test_invalid_reservation_leaves_charged_work_and_storage_unchanged(
    work: int | float,
    storage: int | float,
    field: str,
    error: type[Exception],
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 32)
    with budget.measurement() as phase:
        with pytest.raises(error, match=rf"CoordinateEnclosureBudget\.{field}"):
            budget.reserve(work, storage)  # ty: ignore[invalid-argument-type]
        assert (
            phase.work_units == phase.required_work_units == phase.peak_bytes_upper == 0
        )
    assert budget.work_units == budget.required_work_units == 3
    assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 32
    assert budget.ticket.work_units == 3
    assert budget.ticket.managed_bytes == 32


def test_invalid_storage_selector_is_refused_before_accounting(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 32)
    with pytest.raises(ValueError, match="CoordinateEnclosureBudget.storage_resource"):
        budget.reserve(1, 16, storage_resource="unknown")  # ty: ignore[invalid-argument-type]
    assert budget.work_units == budget.required_work_units == 3
    assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 32
    assert budget.ticket.work_units == 3


def test_degree_one_reference_reuse_is_request_local_and_charged_once() -> None:
    elements = []
    for _ in range(2):
        with HostBatchTicket(10000, bytes_upper=1000000) as ticket:
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=10000, maximum_memory_bytes=1000000
            )
            with ledger.activate():
                element = ledger.degree_one_coordinate_element("tetrahedron")
                completed = ledger.work_units
                retained = ledger.retained_basis_bytes
                assert completed > 0
                assert retained > 0
                assert ledger.degree_one_coordinate_element("tetrahedron") is element
                assert ledger.work_units == completed
                assert ledger.retained_basis_bytes == retained
                elements.append(element)
    assert elements[0] is not elements[1]
    assert elements[0].element_id == elements[1].element_id


def test_algebraic_retention_profiles_exact_terms_and_reuses_shared_graph() -> None:
    from phydrax.discretization._exact_algebraic_geometry import RadicalElement

    bounds = []
    for bits in (1, 4096):
        with HostBatchTicket(10000, bytes_upper=1000000) as ticket:
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=10000, maximum_memory_bytes=1000000
            )
            with ledger.activate():
                value = RadicalElement(terms=((2, Fraction(1 << bits)),))
                basis = (value,)
                ledger.retain_basis(basis)
                completed = ledger.work_units
                retained = ledger.retained_basis_bytes
                ledger.retain_basis(basis)
                assert ledger.work_units == completed
                assert ledger.retained_basis_bytes == retained
                bounds.append(retained)
    assert bounds[1] > bounds[0]


def test_numpy_integer_reservation_normalizes_execution_counts_once(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(np.int64(3), np.int64(32))
    assert type(budget.work_units) is int
    assert type(budget.temporary_bytes_upper) is int
    assert budget.work_units == 3
    assert budget.temporary_bytes_upper == 32
    assert budget.ticket.work_units == 3
    assert budget.native_charged_work_units == 0


@pytest.mark.parametrize("resource", ("coefficient_work", "polynomial_storage"))
def test_resumed_admission_preserves_completed_prefix(
    resource: str, budget: CoordinateEnclosureBudget
) -> None:
    budget.reserve(12, 64)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        with budget.bound_stage(
            5 if resource == "coefficient_work" else 100,
            32 if resource == "polynomial_storage" else 8192,
            starting_work_units=0,
        ):
            pytest.fail("A refused completed prefix entered its resumed stage.")
    refusal = caught.value
    assert refusal.resource == resource
    assert refusal.admission
    assert (
        refusal.requested
        == refusal.completed
        == (12 if resource == "coefficient_work" else 64)
    )
    assert budget.work_units == 12
    assert budget.temporary_bytes_upper == 64
    assert budget.ticket.work_units == 12


def test_dry_work_admission_refusal_preserves_completed_prefix(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(12, 64)
    budget.admit_work_bound(88, storage_upper=32)
    assert budget.required_work_units == 100
    assert budget.work_units == budget.ticket.work_units == 12
    assert budget.retained_basis_bytes == 0
    assert (
        budget.temporary_bytes_upper
        == budget.peak_bytes_upper
        == budget.ticket.managed_bytes
        == 64
    )

    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        budget.admit_work_bound(89, storage_upper=32)
    refusal = caught.value
    assert refusal.resource == "coefficient_work"
    assert (refusal.limit, refusal.requested, refusal.completed) == (100, 101, 12)
    assert not refusal.admission
    assert budget.required_work_units == 101
    assert budget.work_units == budget.ticket.work_units == 12
    assert budget.retained_basis_bytes == 0
    assert (
        budget.temporary_bytes_upper
        == budget.peak_bytes_upper
        == budget.ticket.managed_bytes
        == 64
    )


def test_nested_stage_minima_preserve_dry_refusals_and_restore_outer_bounds(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 32)
    with budget.bound_stage(10, 512, starting_work_units=0):
        with budget.bound_stage(100, 1024, starting_work_units=0):
            budget.admit_work_bound(7, storage_upper=480)
            assert budget.work_units == 3
            assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 32
            budget.reserve(7, 480)
            with pytest.raises(CoordinateEnclosureResourceError) as work_refusal:
                budget.admit_work_bound(1)
            assert work_refusal.value.resource == "coefficient_work"
            assert (
                work_refusal.value.limit,
                work_refusal.value.requested,
                work_refusal.value.completed,
            ) == (10, 11, 10)
            with pytest.raises(CoordinateEnclosureResourceError) as storage_refusal:
                budget.admit_work_bound(0, storage_upper=1)
            assert storage_refusal.value.resource == "polynomial_storage"
            assert (
                storage_refusal.value.limit,
                storage_refusal.value.requested,
                storage_refusal.value.completed,
            ) == (512, 513, 512)
        with pytest.raises(CoordinateEnclosureResourceError):
            budget.reserve(1)
        assert budget.work_units == budget.ticket.work_units == 10
        assert budget.temporary_bytes_upper == budget.ticket.managed_bytes == 512
    budget.reserve(1, 1)
    assert budget.work_units == budget.ticket.work_units == 11
    assert budget.required_work_units == 11
    assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 513


def test_nested_refusal_records_the_innermost_completed_phase(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(95, 128)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        with budget.measurement() as parent:
            budget.reserve(2, 32)
            with budget.measurement() as child:
                budget.reserve(2, 64)
                budget.reserve(2)
    refusal = caught.value
    evidence = refusal.phase_evidence
    assert evidence is not None
    assert evidence.work_units == child.work_units == 2
    assert evidence.peak_bytes_upper == child.peak_bytes_upper == 64
    assert evidence.native_charged_work_units == child.native_charged_work_units == 0
    assert parent.work_units == 4
    assert parent.required_work_units == 6
    assert parent.peak_bytes_upper == 96
    assert child.required_work_units == 4
    assert (refusal.limit, refusal.requested, refusal.completed) == (100, 101, 99)
    assert budget.ticket.work_units == budget.work_units == 99
    budget.reserve(1)
    refusal.record_phase(parent)
    assert refusal.phase_evidence is evidence
    assert child.work_units == 2
    assert parent.work_units == 4


def test_active_phase_cannot_be_attached_to_refusal(
    budget: CoordinateEnclosureBudget,
) -> None:
    refusal = CoordinateEnclosureResourceError("coefficient_work", 10, 11, 0)
    with budget.measurement() as phase:
        with pytest.raises(RuntimeError, match="CoordinateEnclosureResourceError.phase"):
            refusal.record_phase(phase)
        assert refusal.phase_evidence is None
    refusal.record_phase(phase)
    assert refusal.phase_evidence is not None
    assert refusal.phase_evidence.work_units == 0


def test_live_owner_survives_temporary_scope_and_releases_its_bound(
    budget: CoordinateEnclosureBudget,
) -> None:
    with budget.live_storage() as owner:
        owner.set_bound(128, work=2)
        with budget.temporary_scope():
            budget.reserve(3, 256)
            owner.set_bound(64)
            assert budget.temporary_bytes_upper == 320
        assert owner.bound == budget.temporary_bytes_upper == 64
    assert owner.bound == budget.temporary_bytes_upper == 0
    assert budget.work_units == budget.ticket.work_units == 5
    assert budget.peak_bytes_upper == budget.ticket.managed_bytes == 384


@pytest.mark.parametrize("primary_failure", [False, True], ids=["normal-exit", "body-error"])
def test_out_of_order_temporary_close_preserves_inner_admission(
    budget: CoordinateEnclosureBudget, primary_failure: bool,
) -> None:
    with budget.temporary_scope():
        budget.reserve(0, 32)
        outer = budget.temporary_scope()
        outer_token = outer.__enter__()
        budget.reserve(0, 128)
        inner = budget.temporary_scope()
        inner_token = inner.__enter__()
        budget.reserve(0, 256)
        if primary_failure:
            primary = ValueError("source operation failed")
            assert not outer.__exit__(type(primary), primary, None)
            assert any("last-in-first-out" in note for note in primary.__notes__)
        else:
            with pytest.raises(RuntimeError, match="last-in-first-out"):
                outer.__exit__(None, None, None)
        assert budget.temporary_bytes_upper == 416
        assert inner_token.available_bytes == 256
        with pytest.raises(RuntimeError, match="closed scope"):
            _ = outer_token.available_bytes
        with budget.live_storage() as owner:
            with pytest.raises(RuntimeError):
                owner.promote_temporary_storage(outer_token, 64)
            assert owner.bound == 0
        inner.__exit__(None, None, None)
        assert budget.temporary_bytes_upper == 160
    assert budget.temporary_bytes_upper == 0


def test_closed_live_owner_refuses_changes_without_reopening_storage(
    budget: CoordinateEnclosureBudget,
) -> None:
    with budget.live_storage() as owner:
        owner.set_bound(128)
    with pytest.raises(RuntimeError, match="CoordinateLiveStorage.set_bound"):
        owner.set_bound(64)
    owner.close()
    assert owner.bound == budget.temporary_bytes_upper == 0
    assert budget.work_units == 0


def test_retained_graph_reuse_keeps_storage_bound_without_releasing_owner(
    budget: CoordinateEnclosureBudget,
) -> None:
    coefficients = (Fraction(3, 7), Fraction(5, 11))
    budget.retain_basis(coefficients)
    retained = budget.retained_basis_bytes
    work = budget.work_units
    assert retained > 0
    budget.retain_basis(coefficients)
    assert budget.retained_basis_bytes == retained
    assert budget.temporary_bytes_upper == 0
    assert budget.work_units == work == 7
    assert len(budget.retained_objects) == 7
    assert budget.retained_objects[id(coefficients)] is coefficients
    assert all(
        budget.retained_objects[id(coefficient)] is coefficient
        for coefficient in coefficients
    )


def _polynomial_retention_trace(
    ledger: CoordinateEnclosureBudget,
) -> tuple[int, int, int, int, int]:
    first: Polynomial = {(0,): Fraction(1), (1,): Fraction(2)}
    second: Polynomial = {(0,): Fraction(3), (1,): Fraction(4)}
    for _ in range(128):
        multiply(first, second, budget=ledger)
    ledger.retain_basis(tuple(Fraction(index, 97) for index in range(512)))
    ledger.reserve_geometry_queries(3)
    return (
        ledger.work_units,
        ledger.retained_basis_bytes,
        ledger.temporary_bytes_upper,
        ledger.peak_bytes_upper,
        ledger.ticket.geometry_queries,
    )


def test_coordinate_polling_is_request_local_and_bounds_admissions_and_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with HostBatchTicket(20_000, bytes_upper=8192) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=20_000, maximum_memory_bytes=8192
        )
        polls: list[int] = []
        checkpoint = ticket.checkpoint

        def observe() -> None:
            polls.append(ledger.work_units)
            checkpoint()

        monkeypatch.setattr(ticket, "checkpoint", observe)
        for _ in range(257):
            ledger.reserve(0)
        assert polls == [0, 0]
        ledger.reserve(4096)
        assert polls == [0, 0, 0]
        ledger.reserve(0)
        assert polls == [0, 0, 0, 4096]
        with HostBatchTicket(1, bytes_upper=8192) as other_ticket:
            other = CoordinateEnclosureBudget(
                other_ticket, maximum_work_units=1, maximum_memory_bytes=8192
            )
            other.reserve(1)
        ledger.reserve(1)
        assert polls == [0, 0, 0, 4096]
        assert ticket.work_units == ledger.work_units == 4097


def test_coordinate_deadline_polling_preserves_ample_ticket_accounting() -> None:
    maximum_work, maximum_bytes = 100_000, 4 << 20
    with HostBatchTicket(
        maximum_work, bytes_upper=maximum_bytes, queries_upper=10
    ) as ticket:
        host = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_bytes,
        )
        with host.activate():
            expected = _polynomial_retention_trace(host)
    with NativeExecutionScope(
        NativeExecutionLimits(maximum_work, 10, 0, maximum_bytes, inf)
    ) as scope:
        with scope.batch(
            maximum_work, queries_upper=10, bytes_upper=maximum_bytes
        ) as ticket:
            native = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=maximum_work,
                maximum_memory_bytes=maximum_bytes,
            )
            with native.activate():
                observed = _polynomial_retention_trace(native)
            assert native.native_charged_work_units == native.work_units
            assert ticket.work_units == native.work_units
    assert observed == expected


@pytest.mark.parametrize("operation", ("polynomial", "retention"))
def test_coordinate_host_loops_preserve_typed_deadline_refusal(operation: str) -> None:
    maximum_work, maximum_bytes = 100_000, 4 << 20
    scope = NativeExecutionScope(
        NativeExecutionLimits(maximum_work, 10, 0, maximum_bytes, 1.0)
    )
    with pytest.raises(MeshcoreError) as caught:
        with scope:
            with scope.batch(
                maximum_work, queries_upper=10, bytes_upper=maximum_bytes
            ) as ticket:
                ledger = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=maximum_work,
                    maximum_memory_bytes=maximum_bytes,
                )
                with ledger.activate():
                    ledger.reserve(7, 128)
                    sleep(1.01)
                    with pytest.raises(MeshcoreError) as host_refusal:
                        if operation == "polynomial":
                            first: Polynomial = {(0,): Fraction(1), (1,): Fraction(2)}
                            for _ in range(128):
                                multiply(first, first, budget=ledger)
                        else:
                            ledger.retain_basis(
                                tuple(Fraction(index, 97) for index in range(512))
                            )
                    assert host_refusal.value.status is MeshcoreStatus.TIMEOUT
                    assert ledger.work_units < maximum_work
                    assert ticket.work_units == ledger.work_units
                    raise host_refusal.value
    assert caught.value is host_refusal.value
    assert caught.value.batch_admitted_upper == (
        maximum_work,
        10,
        0,
        maximum_bytes,
    )
    # A pure deadline checkpoint does not fabricate a capacity-refusal snapshot.
    assert caught.value.batch_refusal_evidence is None
    assert caught.value.work_counters is not None
    assert scope.evidence is not None
    assert scope.evidence.status is MeshcoreStatus.TIMEOUT
    assert scope.evidence.elapsed_seconds >= scope.limits.maximum_seconds == 1.0
    assert caught.value.elapsed_seconds == scope.evidence.elapsed_seconds
    assert caught.value.maximum_seconds == 1.0


def test_refusal_preserves_numpy_quantity_identity() -> None:
    limit, requested, completed = np.int64(10), np.int64(11), np.int64(9)
    refusal = CoordinateEnclosureResourceError(
        "coefficient_work", limit, requested, completed
    )
    assert refusal.limit is limit
    assert refusal.requested is requested
    assert refusal.completed is completed
    assert not refusal.admission
    assert "CoordinateEnclosureBudget.coefficient_work must fit limit 10" in str(refusal)


def test_live_storage_boundary_refuses_foreign_budget() -> None:
    with pytest.raises(TypeError, match="budget"):
        CoordinateLiveStorage(object())  # ty: ignore[invalid-argument-type]


def test_measurement_boundary_refuses_foreign_budget() -> None:
    with pytest.raises(TypeError, match="budget"):
        CoordinateEnclosureMeasurement(object())  # ty: ignore[invalid-argument-type]


def test_polynomial_reservation_boundary_refuses_foreign_budget() -> None:
    with pytest.raises(TypeError, match="budget"):
        reserve_polynomial(0, 1, 1, 64, budget=object())  # ty: ignore[invalid-argument-type]


def test_refusal_phase_boundary_refuses_foreign_measurement() -> None:
    refusal = CoordinateEnclosureResourceError("coefficient_work", 10, 11, 0)
    with pytest.raises(TypeError, match="measurement"):
        refusal.record_phase(object())  # ty: ignore[invalid-argument-type]
    assert refusal.phase_evidence is None


def test_retained_storage_boundary_refuses_noncallable_factory(
    budget: CoordinateEnclosureBudget,
) -> None:
    with pytest.raises(TypeError, match="factory"):
        with budget.retained_storage(object()):  # ty: ignore[invalid-argument-type]
            pytest.fail("A noncallable storage factory entered the retention scope.")
    assert budget.storage_ticket is None
    assert not budget.ticket.settled


def test_temporary_release_preserves_work_and_peak_evidence(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 256)
    with budget.measurement() as phase:
        budget.release_temporary_storage(64)
    assert budget.temporary_bytes_upper == 192
    assert budget.peak_bytes_upper == budget.ticket.managed_bytes == 256
    assert budget.work_units == budget.required_work_units == 3
    assert budget.ticket.work_units == 3
    assert phase.work_units == phase.required_work_units == phase.peak_bytes_upper == 0
    budget.release_temporary_storage(192)
    assert budget.temporary_bytes_upper == 0
    assert budget.ticket.work_units == 3


def test_excessive_temporary_release_preserves_storage_and_charged_work(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 256)
    with pytest.raises(RuntimeError, match="CoordinateEnclosureBudget.amount"):
        budget.release_temporary_storage(257)
    assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 256
    assert budget.work_units == budget.required_work_units == 3
    assert budget.ticket.work_units == 3


def test_negative_temporary_release_cannot_expand_storage(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve(3, 256)
    with pytest.raises(ValueError, match="CoordinateEnclosureBudget.amount"):
        budget.release_temporary_storage(-1)
    assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 256
    assert budget.work_units == budget.required_work_units == 3
    assert budget.ticket.work_units == 3


def test_retained_shared_graph_charges_only_new_wrapper_content(
    budget: CoordinateEnclosureBudget,
) -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    budget.retain_basis(graph)
    assert budget.work_units == len(budget.retained_objects) == 4
    retained = budget.retained_basis_bytes
    wrapper = (graph, coefficient)
    budget.retain_basis(wrapper)
    assert budget.work_units == len(budget.retained_objects) == 5
    assert budget.retained_basis_bytes > retained
    assert budget.retained_objects[id(graph)] is graph
    assert budget.retained_objects[id(wrapper)] is wrapper
    budget.retain_basis(graph)
    budget.retain_basis(wrapper)
    assert budget.work_units == 5


def test_equal_fresh_coefficients_do_not_reuse_another_objects_profile(
    budget: CoordinateEnclosureBudget,
) -> None:
    original = (Fraction("1009/1013"),)
    equal = (Fraction("1009/1013"),)
    assert original == equal and original is not equal
    budget.retain_basis(original)
    assert budget.work_units == 4
    budget.retain_basis(equal)
    assert budget.work_units == len(budget.retained_objects) == 8
    assert budget.retained_objects[id(original)] is original
    assert budget.retained_objects[id(equal)] is equal
    assert budget.retained_objects[id(original[0])] is original[0]
    assert budget.retained_objects[id(equal[0])] is equal[0]


def test_retained_profile_is_earned_again_in_an_independent_request() -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    for _ in range(2):
        with HostBatchTicket(4, bytes_upper=8192) as ticket:
            ledger = CoordinateEnclosureBudget(
                ticket, maximum_work_units=4, maximum_memory_bytes=8192
            )
            with ledger.activate():
                ledger.retain_basis(graph)
                ledger.retain_basis(graph)
                assert ledger.work_units == len(ledger.retained_objects) == 4
            assert ticket.work_units == 4


def test_retained_hit_at_exact_work_capacity_still_refuses_new_content() -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    with HostBatchTicket(4, bytes_upper=8192) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=4, maximum_memory_bytes=8192
        )
        with ledger.activate():
            ledger.retain_basis(graph)
            retained = ledger.retained_basis_bytes
            peak = ledger.peak_bytes_upper
            ledger.retain_basis(graph)
            assert ledger.peak_bytes_upper == peak
            fresh = (graph,)
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                ledger.retain_basis(fresh)
            assert caught.value.resource == "coefficient_work"
            assert (
                caught.value.limit,
                caught.value.requested,
                caught.value.completed,
            ) == (4, 5, 4)
            assert ledger.retained_basis_bytes == retained
            assert id(fresh) not in ledger.retained_objects
            assert ledger.temporary_bytes_upper == 0
            assert ledger.work_units == 4
            assert ledger.required_work_units == 5
        assert ticket.work_units == 4


def test_live_graph_profiles_are_not_retained_between_traversals(
    budget: CoordinateEnclosureBudget,
) -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    first = budget.live_object_storage_upper(graph)
    assert budget.work_units == 4
    assert budget.live_object_storage_upper(graph) == first
    assert budget.work_units == 8
    assert budget.retained_basis_bytes == budget.temporary_bytes_upper == 0
    assert budget.retained_objects == {}


def test_retained_scientific_owner_profiles_only_fresh_graph_content(
    budget: CoordinateEnclosureBudget,
) -> None:
    source = np.arange(4, dtype=np.float64)
    graph = (source, source)
    budget.retain_owner(graph)
    assert budget.work_units == len(budget.retained_owner_objects) == 2
    retained = budget.retained_basis_bytes
    peak = budget.peak_bytes_upper
    budget.retain_owner(graph)
    budget.retain_owner(source)
    assert budget.work_units == 2
    assert budget.retained_basis_bytes == retained
    assert budget.peak_bytes_upper == peak
    wrapper = (graph,)
    budget.retain_owner(wrapper)
    assert budget.work_units == len(budget.retained_owner_objects) == 3
    assert budget.retained_basis_bytes > retained
    assert budget.retained_owner_objects[id(source)] is source
    assert budget.retained_owner_objects[id(graph)] is graph
    assert budget.retained_owner_objects[id(wrapper)] is wrapper


def test_retained_owner_credit_survives_nested_temporary_scopes(
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    size = declared_object_storage_bytes(graph)
    assert size == 320
    budget.reserve(2, 16)
    with budget.temporary_scope():
        budget.reserve(3, 32)
        with budget.live_storage() as lease:
            lease.set_bound(size + 80, work=1)
            with budget.temporary_scope():
                budget.reserve(4, 64)
                peak = budget.peak_bytes_upper
                budget.retain_owner(graph, temporary_storage_credit=lease)
                assert lease.bound == budget._live_bytes_upper == 80
                assert budget.temporary_bytes_upper == 16 + 32 + 80 + 64
                assert budget.retained_basis_bytes == size
                assert budget.peak_bytes_upper == peak == 512
                assert budget.retained_owner_objects == {id(graph): graph}
            assert budget.temporary_bytes_upper == 16 + 32 + 80
            assert budget.retained_basis_bytes == size
        assert lease.bound == budget._live_bytes_upper == 0
        assert budget.temporary_bytes_upper == 16 + 32
    assert budget.temporary_bytes_upper == 16
    assert budget.retained_basis_bytes == size
    assert budget.retained_owner_objects[id(graph)] is graph
    assert budget.logical_array_bytes_upper == 0
    assert budget.work_units == budget.ticket.work_units == 11
    assert budget.peak_bytes_upper == budget.ticket.managed_bytes == 512


def test_retained_owner_credit_reserves_only_genuine_excess(
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    size = declared_object_storage_bytes(graph)
    budget.reserve(2, 32)
    with budget.live_storage() as lease:
        lease.set_bound(128)
        with budget.bound_stage(100, 32 + size):
            budget.retain_owner(graph, temporary_storage_credit=lease)
        assert lease.bound == budget._live_bytes_upper == 0
        assert budget.temporary_bytes_upper == 32
        assert budget.retained_basis_bytes == size
        assert budget.retained_owner_objects == {id(graph): graph}
        assert budget.peak_bytes_upper == 32 + size
    assert budget.temporary_bytes_upper == 32
    assert budget.retained_basis_bytes == size
    assert budget.work_units == budget.ticket.work_units == 3
    assert budget.logical_array_bytes_upper == 0


def test_duplicate_retained_owner_does_not_consume_credit(
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    size = declared_object_storage_bytes(graph)
    budget.retain_owner(graph)
    owners = budget.retained_owner_objects.copy()
    with budget.live_storage() as lease:
        lease.set_bound(size + 32)
        peak = budget.peak_bytes_upper
        with budget.bound_stage(0, peak):
            budget.retain_owner(graph, temporary_storage_credit=lease)
        assert budget.work_units == budget.ticket.work_units == 1
        assert lease.bound == budget._live_bytes_upper == size + 32
        assert budget.temporary_bytes_upper == size + 32
        assert budget.retained_basis_bytes == size
        assert budget.retained_owner_objects == owners
        assert budget.peak_bytes_upper == peak
    assert budget.temporary_bytes_upper == 0
    assert budget.retained_basis_bytes == size


@pytest.mark.parametrize("already_retained", (False, True))
@pytest.mark.parametrize(
    ("kind", "error"),
    (("foreign", ValueError), ("closed", RuntimeError), ("nonlease", TypeError)),
)
def test_retained_owner_refuses_invalid_credit_before_mutation(
    kind: str,
    error: type[Exception],
    already_retained: bool,
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    if already_retained:
        budget.retain_owner(graph)
    budget.reserve(3, 32)
    owners = budget.retained_owner_objects.copy()
    retained = budget.retained_basis_bytes
    work = budget.work_units
    peak = budget.peak_bytes_upper
    with HostBatchTicket(10, bytes_upper=512) as foreign_ticket:
        foreign = CoordinateEnclosureBudget(
            foreign_ticket, maximum_work_units=10, maximum_memory_bytes=512
        )
        if kind == "foreign":
            credit: object = CoordinateLiveStorage(foreign)
        elif kind == "closed":
            credit = CoordinateLiveStorage(budget)
            credit.close()
        else:
            credit = object()
        with pytest.raises(
            error, match="CoordinateEnclosureBudget.temporary_storage_credit"
        ):
            budget.retain_owner(
                graph,
                temporary_storage_credit=credit,  # ty: ignore[invalid-argument-type]
            )
    assert budget.work_units == budget.required_work_units == work
    assert budget.ticket.work_units == work
    assert budget.temporary_bytes_upper == 32
    assert budget._live_bytes_upper == 0
    assert budget.retained_basis_bytes == retained
    assert budget.retained_owner_objects == owners
    assert budget.logical_array_bytes_upper == 0
    assert budget.peak_bytes_upper == peak


@pytest.mark.parametrize(
    ("bound", "error"),
    ((True, TypeError), (1.5, TypeError), (-1, ValueError), (257, RuntimeError)),
)
def test_retained_owner_refuses_unadmitted_credit_bound(
    bound: int | float,
    error: type[Exception],
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    with budget.live_storage() as lease:
        lease.set_bound(256)
        lease.bound = bound  # ty: ignore[invalid-assignment]
        try:
            with pytest.raises(
                error, match="CoordinateEnclosureBudget.temporary_storage_credit.bound"
            ):
                budget.retain_owner(graph, temporary_storage_credit=lease)
            assert lease.bound == bound
            assert budget.work_units == budget.required_work_units == 0
            assert budget.temporary_bytes_upper == budget._live_bytes_upper == 256
            assert budget.retained_basis_bytes == 0
            assert budget.retained_owner_objects == {}
            assert budget.peak_bytes_upper == 256
        finally:
            lease.bound = 256


def test_retained_owner_credit_must_fit_temporary_accounting(
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    with budget.live_storage() as lease:
        lease.set_bound(256)
        budget.release_temporary_storage(128)
        try:
            with pytest.raises(
                RuntimeError,
                match="CoordinateEnclosureBudget.temporary_storage_credit.bound",
            ):
                budget.retain_owner(graph, temporary_storage_credit=lease)
            assert lease.bound == budget._live_bytes_upper == 256
            assert budget.temporary_bytes_upper == 128
            assert budget.work_units == budget.required_work_units == 0
            assert budget.retained_basis_bytes == 0
            assert budget.retained_owner_objects == {}
        finally:
            budget.reserve(0, 128)


@pytest.mark.parametrize("use_credit", (False, True))
def test_retained_owner_work_refusal_precedes_storage_admission(
    use_credit: bool,
    budget: CoordinateEnclosureBudget,
) -> None:
    graph = bytearray(64)
    with budget.live_storage() as lease:
        lease.set_bound(128, work=3)
        with budget.bound_stage(0, 128):
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                budget.retain_owner(
                    graph, temporary_storage_credit=lease if use_credit else None
                )
        assert caught.value.resource == "coefficient_work"
        assert (
            caught.value.limit,
            caught.value.requested,
            caught.value.completed,
        ) == (3, 4, 3)
        assert budget.work_units == budget.ticket.work_units == 3
        assert budget.required_work_units == 4
        assert lease.bound == budget._live_bytes_upper == 128
        assert budget.temporary_bytes_upper == budget.peak_bytes_upper == 128
        assert budget.retained_basis_bytes == 0
        assert budget.retained_owner_objects == {}


@pytest.mark.parametrize("credit_bound", (None, 128))
def test_retained_owner_capacity_refusal_preserves_profile_work_and_credit(
    credit_bound: int | None,
    budget: CoordinateEnclosureBudget,
) -> None:
    previous = bytearray()
    graph = bytearray(64)
    size = declared_object_storage_bytes(graph)
    budget.retain_owner(previous)
    budget.reserve(3, 32)
    retained = budget.retained_basis_bytes
    owners = budget.retained_owner_objects.copy()
    with budget.live_storage() as lease:
        lease.set_bound(0 if credit_bound is None else credit_bound, work=2)
        temporary = budget.temporary_bytes_upper
        peak = budget.peak_bytes_upper
        excess = size - lease.bound
        limit = retained + temporary + excess - 1
        with budget.bound_stage(100, limit):
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                budget.retain_owner(
                    graph,
                    temporary_storage_credit=None if credit_bound is None else lease,
                )
        refusal = caught.value
        assert refusal.resource == "polynomial_storage"
        assert (refusal.limit, refusal.requested, refusal.completed) == (
            limit,
            retained + temporary + excess,
            retained + temporary,
        )
        assert not refusal.admission
        assert budget.work_units == budget.required_work_units == 7
        assert budget.ticket.work_units == 7
        assert not lease._closed
        assert (
            lease.bound
            == budget._live_bytes_upper
            == (0 if credit_bound is None else credit_bound)
        )
        assert budget.temporary_bytes_upper == temporary
        assert budget.retained_basis_bytes == retained
        assert budget.retained_owner_objects == owners
        assert budget.logical_array_bytes_upper == 0
        assert budget.peak_bytes_upper == peak
    assert budget.temporary_bytes_upper == 32
    assert budget.retained_basis_bytes == retained


def test_retained_hit_at_exact_memory_capacity_still_refuses_new_storage(
    budget: CoordinateEnclosureBudget,
) -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    budget.retain_basis(graph)
    retained = budget.retained_basis_bytes
    peak = budget.peak_bytes_upper
    with budget.bound_stage(0, retained):
        budget.retain_basis(graph)
        with pytest.raises(CoordinateEnclosureResourceError) as caught:
            budget.retain_basis((graph,))
        assert caught.value.resource == "retained_basis"
        assert (
            caught.value.limit,
            caught.value.requested,
            caught.value.completed,
        ) == (retained, retained + 256, retained)
        assert budget.work_units == len(budget.retained_objects) == 4
        assert budget.retained_basis_bytes == retained
        assert budget.temporary_bytes_upper == 0
        assert budget.peak_bytes_upper == peak


def test_nested_query_measurements_charge_once_and_freeze_on_refusal(
    budget: CoordinateEnclosureBudget,
) -> None:
    budget.reserve_geometry_queries(3)
    with pytest.raises(HostBatchEstimateError):
        with budget.measurement() as outer:
            budget.reserve_geometry_queries(2)
            with budget.measurement() as inner:
                budget.reserve_geometry_queries(5)
                assert budget.ticket.geometry_queries == 10
                budget.reserve_geometry_queries(1)
    assert inner.geometry_queries == 5
    assert outer.geometry_queries == 7
    assert budget.ticket.geometry_queries == 10
    assert budget.work_units == 0


def test_query_phase_evidence_excludes_prefix_and_preserves_inner_refusal(
    budget: CoordinateEnclosureBudget,
) -> None:
    from phydrax._model._structure import (
        model_from_structure_recipe,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    budget.reserve_geometry_queries(2)
    with pytest.raises(CoordinateEnclosureResourceError) as caught:
        with budget.measurement() as outer:
            budget.reserve_geometry_queries(3)
            with budget.measurement() as inner:
                budget.reserve_geometry_queries(4)
                budget.reserve(101)
    evidence = caught.value.phase_evidence
    assert evidence is not None
    assert evidence.geometry_queries == inner.geometry_queries == 4
    assert outer.geometry_queries == 7
    assert budget.ticket.geometry_queries == 9
    register_meshing_source_artifacts()
    restored = model_from_structure_recipe(model_structure_recipe(evidence))
    assert restored == evidence


def test_query_measurement_excludes_child_charges_and_freezes_after_exception(
    budget: CoordinateEnclosureBudget,
) -> None:
    with pytest.raises(ValueError, match="injected phase failure"):
        with budget.measurement() as phase:
            with budget.ticket.batch(0, queries_upper=4) as child:
                child.consume(geometry_queries=4)
                assert phase.geometry_queries == 0
            assert phase.geometry_queries == phase.work_units == 0
            assert budget.ticket.remaining_geometry_queries == 6
            raise ValueError(
                "query_phase.probe must finish without error; got injected phase failure"
            )
    budget.reserve_geometry_queries(1)
    assert phase.geometry_queries == 0
    assert budget.ticket.geometry_queries == 1


def test_parent_refusal_with_live_child_keeps_typed_prefix_until_child_unwinds() -> None:
    with HostBatchTicket(100) as request:
        root = request.batch(10)
        budget = CoordinateEnclosureBudget(
            root, maximum_work_units=100, maximum_memory_bytes=1 << 20
        )
        with budget.activate():
            budget.reserve(1)
            with root.batch(8) as child:
                child.consume(3)
                with pytest.raises(HostBatchEstimateError) as refused:
                    budget.reserve(5)
                assert refused.value.batch_requested_upper == (5, 0, 0, 0)
                assert refused.value.batch_refusal_evidence == (1, 0, 0, 0, 1, 0, 0, 0)
                assert budget.work_units == root.work_units == 1
                assert budget.required_work_units == 6
                assert not root.settled and not child.settled
                assert request.remaining_work_units == 90
                with pytest.raises(RuntimeError, match="refused"):
                    root.batch(1)
                with pytest.raises(RuntimeError, match="refused"):
                    root.consume(1)
                child.consume(2)
            assert child.settled and root.settled
        assert request.remaining_work_units == 94
    assert request.work_units == 0


def test_refused_parent_exit_retires_retained_child_without_replacing_refusal() -> None:
    with HostBatchTicket(100) as request:
        with pytest.raises(HostBatchEstimateError) as refused:
            with request.batch(10) as root:
                retained = root.batch(8)
                retained.consume(1)
                root.refuse(5)
        assert refused.value.batch_refusal_evidence[0] == 2
        assert retained.settled and root.settled
        assert request.remaining_work_units == 99


def test_released_cross_phase_storage_keeps_work_refusal_typed() -> None:
    with HostBatchTicket(10, bytes_upper=10_000) as root:
        first = root.batch(5)
        first.settle()
        budget = CoordinateEnclosureBudget(
            first, maximum_work_units=100, maximum_memory_bytes=1 << 20
        )
        with budget.retained_storage(lambda upper: root.batch(0, bytes_upper=upper)):
            lease = CoordinateLiveStorage(budget)
            with budget.bind_ticket(root.batch(5, bytes_upper=2000)):
                lease.set_bound(1000)
            retained = budget.storage_ticket
            assert retained is not None and retained.bytes_upper == 1000
            lease.close()
            stage = root.batch(1, bytes_upper=2000)
            with pytest.raises(HostBatchEstimateError) as refused:
                with budget.bind_ticket(stage):
                    budget.reserve(2)
            assert refused.value.batch_requested_upper == (2, 0, 0, 0)
            assert refused.value.batch_refusal_evidence == (1, 0, 0, 2000, 0, 0, 0, 0)
            assert stage.settled and retained.settled
            assert budget.work_units == 0 and budget.required_work_units == 2
            assert budget.temporary_bytes_upper == 0
        assert root.remaining_work_units == 10
    assert root.work_units == 0


def test_coordinate_map_operation_preserves_full_bank_binding() -> None:
    from phydrax.discretization import _coordinate_enclosure as algebra

    with HostBatchTicket(1000000, bytes_upper=16000000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1000000, maximum_memory_bytes=16000000
        )
        with ledger.activate():
            element = ledger.degree_one_coordinate_element("triangle")
            controls = np.asarray(element.reference_nodes, dtype=np.float64).copy()
            expected = (
                algebra.coordinate_expressions(element, controls, budget=ledger),
                algebra.coordinate_corner_images(element, controls, budget=ledger),
            )
            observed = algebra.coordinate_map_images(element, controls, budget=ledger)
            assert observed == expected
            controls[0, 0] = 0.25
            changed = algebra.coordinate_map_images(element, controls, budget=ledger)
            assert changed != observed
            assert changed[1] is not None
            assert changed[1][0][0] == Fraction(1, 4)
            assert changed == (
                algebra.coordinate_expressions(element, controls, budget=ledger),
                algebra.coordinate_corner_images(element, controls, budget=ledger),
            )


def test_coordinate_map_operation_rehashes_mutable_protocol_owner() -> None:
    from dataclasses import dataclass

    import jax

    from phydrax.discretization import _coordinate_enclosure as algebra

    @jax.tree_util.register_dataclass
    @dataclass
    class MutableElement:
        cell_kind: str
        conformity: str
        local_dof_count: int
        element_id: str
        source: np.ndarray

    element = MutableElement(
        "triangle", "H1", 3, "mutable-source", np.zeros((1,), dtype=np.float64)
    )
    controls = ((Fraction(0), Fraction(0)),) * 3
    with HostBatchTicket(1000000, bytes_upper=16000000) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=1000000, maximum_memory_bytes=16000000
        )
        with ledger.activate():
            _, first = algebra.coordinate_preparation_key(
                element, controls, budget=ledger
            )
            with pytest.raises(TypeError, match="tabulated scalar coordinate element"):
                algebra.coordinate_map_images(element, controls, budget=ledger)
            element.source[0] = 1
            _, second = algebra.coordinate_preparation_key(
                element, controls, budget=ledger
            )
            assert first != second
            assert first[0] == second[0] == element.element_id
            assert first[2] == second[2] == controls
            with pytest.raises(TypeError, match="tabulated scalar coordinate element"):
                algebra.coordinate_map_images(element, controls, budget=ledger)


def _individual_storage_profile(
    ledger: CoordinateEnclosureBudget,
    root: object,
    existing: Mapping[int, object],
    /,
    *,
    retained: bool,
    operand_profile: CoordinateOperandProfileBuilder | None = None,
    include_operands: bool = False,
) -> tuple[dict[int, object], int]:
    """Admit each owner independently, including immutable operand witnesses.

    The traversal index costs 128 bytes and one work unit per fresh owner.
    Public Fraction snapshots add 768 temporary bytes, later promoted by
    retain_basis; kernel-private atoms need no field snapshot. An unsupported
    owner stops snapshot capture, not traversal or its storage admissions.
    """
    resource = "retained_basis" if retained else "polynomial_storage"
    ledger._reserve_validated(0, 256, resource)
    pending: list[object] = [root]
    seen: dict[int, object] = {}
    size = 256 if retained and not existing else 0
    storage_limit = ledger.maximum_memory_bytes
    if ledger._stage_memory_bounds:
        storage_limit = min(storage_limit, ledger._stage_memory_bounds[-1])
    while pending:
        value = pending.pop()
        identifier = id(value)
        if (identifier in seen and seen[identifier] is value) or (
            identifier in existing and existing[identifier] is value
        ):
            continue
        prepared = ledger._prepared_coordinate_operands
        if include_operands and prepared is not None and any(
            profile.owners.get(identifier) is value for profile in prepared._profiles
        ):
            continue
        ledger._reserve_validated(1, 0, "polynomial_storage")
        ledger._reserve_validated(0, 128, resource)
        size += declared_object_storage_bytes(value) + 128
        if (
            retained
            and ledger.retained_basis_bytes + ledger.temporary_bytes_upper + size
            > storage_limit
        ):
            raise CoordinateEnclosureResourceError(
                "retained_basis",
                storage_limit,
                ledger.retained_basis_bytes + ledger.temporary_bytes_upper + size,
                ledger.retained_basis_bytes,
            )
        storage, children = _coordinate_storage_children(value)
        snapshot = (
            operand_profile is not None
            and operand_profile.supported
            and not operand_profile.private_atoms
            and type(value) is Fraction
        )
        if snapshot:
            storage += 768
        if storage:
            ledger._reserve_validated(0, storage, resource)
        seen[identifier] = value
        if operand_profile is not None and operand_profile.supported:
            if type(value) is Fraction:
                if snapshot:
                    operand_profile.owner_fields.append(
                        (
                            value,
                            Fraction,
                            ("numerator", "denominator"),
                            (value.numerator, value.denominator),
                        )
                    )
            elif value is not None and type(value) not in (
                tuple, bool, int, float, complex, str, bytes
            ):
                operand_profile.supported = False
        if storage:
            pending.extend(children)
    return seen, size


@pytest.mark.parametrize("individual", (False, True), ids=("coalesced", "individual"))
@pytest.mark.parametrize("mode", ("ordinary", "public", "private", "unsupported"))
def test_storage_profile_operand_snapshot_costs(
    individual: bool, mode: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Price a tiny graph explicitly, independently of either traversal."""
    coefficient = Fraction(1009, 1013)
    unsupported: dict[object, object] = {}
    basis = (
        (unsupported, coefficient, coefficient)
        if mode == "unsupported"
        else (coefficient, coefficient)
    )
    owners = (basis, coefficient, coefficient.numerator, coefficient.denominator)
    if mode == "unsupported":
        owners = (*owners, unsupported)
    graph_bytes = 256 + sum(
        declared_object_storage_bytes(owner) + 128 for owner in owners
    )
    snapshot_bytes = 768 if mode in ("public", "unsupported") else 0
    traversal_bytes = (
        256 + 128 * len(owners)
        + 64 + 16 * len(basis) + 96
        + (64 if mode == "unsupported" else 0) + snapshot_bytes
    )
    captured = mode in ("public", "private")
    metadata = 1024 + 128 * len(owners) + snapshot_bytes if captured else 0
    if individual:
        monkeypatch.setattr(
            CoordinateEnclosureBudget, "_object_storage_profile",
            _individual_storage_profile,
        )
    with HostBatchTicket(100, bytes_upper=1 << 20) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=1 << 20
        )
        ledger.capture_coordinate_operands = True
        ledger.retain_basis(
            basis, operand=mode != "ordinary", _kernel_operands=mode == "private"
        )
        assert ledger.work_units == ledger.required_work_units == len(owners)
        assert ticket.work_units == len(owners)
        assert ledger.retained_objects == {id(owner): owner for owner in owners}
        assert ledger.retained_basis_bytes == graph_bytes + metadata
        assert ledger.temporary_bytes_upper == 0
        assert ledger.peak_bytes_upper == (
            traversal_bytes + graph_bytes + (1024 if captured else 0)
        )
        assert len(ledger.coordinate_operand_profiles) == int(captured)
        if captured:
            profile = ledger.coordinate_operand_profiles[0]
            assert profile.basis is basis
            assert dict(profile.owners) == ledger.retained_objects
            assert profile.physical_bytes == graph_bytes + metadata
            assert profile.owner_fields == (
                (
                    (
                        coefficient, Fraction, ("numerator", "denominator"),
                        (coefficient.numerator, coefficient.denominator),
                    ),
                )
                if mode == "public" else ()
            )


@pytest.mark.parametrize("individual", (False, True), ids=("coalesced", "individual"))
@pytest.mark.parametrize("retained", (False, True), ids=("live", "retained"))
@pytest.mark.parametrize("include_operands", (False, True), ids=("fresh", "imported"))
def test_storage_profile_imported_owners_are_charged_only_when_requested(
    individual: bool, retained: bool, include_operands: bool,
) -> None:
    first, second = Fraction(1009, 1013), Fraction(1019, 1021)
    bank = (first, first)
    old_owners = (bank, first, first.numerator, first.denominator)
    root = (bank, second)
    owners = (root, second, second.numerator, second.denominator)
    if not include_operands:
        owners = (*owners, *old_owners)
    # Isolate an authenticated bank's ownership phase, not receipt admission.
    prepared = object.__new__(PreparedCoordinateOperands)
    object.__setattr__(
        prepared, "_profiles",
        (
            CoordinateOperandProfile(
                bank, MappingProxyType({id(owner): owner for owner in old_owners}),
                (), 256 + sum(
                    declared_object_storage_bytes(owner) + 128 for owner in old_owners
                ),
            ),
        ),
    )
    with HostBatchTicket(100, bytes_upper=1 << 20) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=100, maximum_memory_bytes=1 << 20
        )
        ledger._prepared_coordinate_operands = prepared
        profile = (
            _individual_storage_profile if individual
            else CoordinateEnclosureBudget._object_storage_profile
        )
        seen, size = profile(
            ledger, root, {}, retained=retained, include_operands=include_operands
        )
        assert seen == {id(owner): owner for owner in owners}
        assert ledger.work_units == ledger.required_work_units == len(owners)
        assert ticket.work_units == len(owners)
        assert size == (256 if retained else 0) + sum(
            declared_object_storage_bytes(owner) + 128 for owner in owners
        )
        assert ledger.retained_basis_bytes == 0
        assert ledger.temporary_bytes_upper == ledger.peak_bytes_upper == (
            256 + 128 * len(owners) + 192 * (1 if include_operands else 2)
        )


_PROFILE_SHARED = Fraction(-(3**80), 7**40)
_PROFILE_ROWS = tuple(
    (
        Fraction(index, 97),
        _PROFILE_SHARED,
        index * 2**70,
        {index: (_PROFILE_SHARED, Fraction(index + 1, 3))},
    )
    for index in range(96)
)
_PROFILE_GRAPH = (
    _PROFILE_ROWS,
    _PROFILE_ROWS[3],
    _PROFILE_SHARED,
    {"rows": _PROFILE_ROWS[:5]},
)
_PROFILE_PREFIX = _PROFILE_ROWS[:2]
_PROFILE_OPERANDS = tuple(row[:3] for row in _PROFILE_ROWS)
# LIFO traversal captures immutable rows before encountering the mutable graph.
_PROFILE_UNSUPPORTED_OPERANDS = (_PROFILE_GRAPH, _PROFILE_OPERANDS)


def _profile_operation(ledger: CoordinateEnclosureBudget, operation: str, /) -> None:
    match operation:
        case "retained":
            ledger.retain_basis(_PROFILE_GRAPH)
        case "live":
            ledger.live_object_storage_upper(_PROFILE_GRAPH)
        case "preparation":
            with ledger.polynomial_preparation_scope():
                ledger.retain_polynomial_preparation(_PROFILE_GRAPH)
        case "operand" | "private-operand" | "unsupported-operand":
            ledger.capture_coordinate_operands = True
            ledger.retain_basis(
                _PROFILE_UNSUPPORTED_OPERANDS
                if operation == "unsupported-operand"
                else _PROFILE_OPERANDS,
                operand=True,
                _kernel_operands=operation == "private-operand",
            )
        case _:
            raise AssertionError(operation)


def _profile_trace(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    /,
    *,
    individual: bool,
    stage_work: int = 1 << 20,
    stage_memory: int = 1 << 24,
    ticket_work: int = 1 << 20,
    ticket_bytes: int = 1 << 24,
    occupied: int = 0,
    retained_phase: bool = False,
) -> tuple[object, ...]:
    """Observe every ledger, measurement, poll and ticket effect of one profile."""
    with monkeypatch.context() as patch:
        if individual:
            patch.setattr(
                CoordinateEnclosureBudget,
                "_object_storage_profile",
                _individual_storage_profile,
            )
        polls: list[tuple[int, int]] = []
        outcome: object = None
        with HostBatchTicket(1 << 30, bytes_upper=1 << 30) as root:
            first = root.batch(1 << 20, bytes_upper=1 << 24)
            ledger = CoordinateEnclosureBudget(
                first, maximum_work_units=1 << 20, maximum_memory_bytes=1 << 24
            )
            # Prior retained content makes the profile skip a shared prefix.
            with first, ledger.activate():
                ledger.retain_basis(_PROFILE_PREFIX)
            if not first.settled:
                first.settle()
            ticket = root.batch(ticket_work, bytes_upper=ticket_bytes)
            checkpoint = ticket.checkpoint

            def observe() -> None:
                polls.append((ledger.work_units, ledger.temporary_bytes_upper))
                checkpoint()

            patch.setattr(ticket, "checkpoint", observe)
            other = ticket.managed_storage()
            if occupied:
                other.set_bound(occupied)
            measurement: CoordinateEnclosureMeasurement | None = None
            try:
                with (
                    ledger.retained_storage(lambda upper: root.batch(0, bytes_upper=upper))
                    if retained_phase
                    else nullcontext()
                ):
                    with ledger.bind_ticket(ticket):
                        with ledger.measurement() as measurement:
                            with ledger.bound_stage(stage_work, stage_memory):
                                _profile_operation(ledger, operation)
            except CoordinateEnclosureResourceError as error:
                outcome = (
                    error.resource,
                    error.limit,
                    error.requested,
                    error.completed,
                    error.admission,
                )
            except HostBatchEstimateError as error:
                outcome = (error.batch_requested_upper, error.batch_refusal_evidence)
            other.close()
            return (
                outcome,
                ledger.work_units,
                ledger.required_work_units,
                ledger.native_charged_work_units,
                ledger.retained_basis_bytes,
                ledger.temporary_bytes_upper,
                ledger.peak_bytes_upper,
                ledger._checkpoint_admissions_remaining,
                ledger._checkpoint_work_upper,
                frozenset(ledger.retained_objects),
                None
                if measurement is None
                else (
                    measurement.required_work_units,
                    measurement.peak_bytes_upper,
                    measurement.work_units,
                ),
                tuple(ledger.stage_evidence),
                ticket.work_units,
                ticket.managed_bytes,
                ticket.settled,
                tuple(
                    (
                        id(profile.basis),
                        frozenset(profile.owners),
                        tuple(
                            (id(owner), kind, names, tuple(map(id, values)))
                            for owner, kind, names, values in profile.owner_fields
                        ),
                        profile.physical_bytes,
                    )
                    for profile in ledger.coordinate_operand_profiles
                ),
                tuple(polls),
            )


@pytest.mark.parametrize(
    "operation",
    (
        "retained", "live", "preparation",
        "operand", "private-operand", "unsupported-operand",
    ),
)
def test_coalesced_storage_profile_replays_individual_admissions(
    operation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    ample = _profile_trace(monkeypatch, operation, individual=True)
    polls = ample[-1]
    assert ample[0] is None and isinstance(polls, tuple) and len(polls) >= 3
    assert _profile_trace(monkeypatch, operation, individual=False) == ample
    work, peak = ample[1], ample[6]
    assert isinstance(work, int) and isinstance(peak, int)
    work_limits = sorted({*range(0, work + 2, 37), work - 1, 255, 256})
    memory_limits = sorted(
        {*range(peak // 3, peak + 2, max(1, peak // 23)), peak - 1, peak}
    )
    cases: list[dict[str, Any]] = [
        *({"stage_work": limit} for limit in work_limits),
        *({"ticket_work": limit} for limit in work_limits),
        *({"stage_memory": limit} for limit in memory_limits),
        *({"ticket_bytes": limit} for limit in memory_limits),
        *({"ticket_bytes": peak, "occupied": peak - limit} for limit in memory_limits),
        *({"ticket_bytes": limit, "retained_phase": True} for limit in memory_limits),
    ]
    refusals = 0
    for case in cases:
        expected = _profile_trace(monkeypatch, operation, individual=True, **case)
        observed = _profile_trace(monkeypatch, operation, individual=False, **case)
        assert observed == expected, case
        refusals += expected[0] is not None
    assert refusals > len(cases) // 3


def test_coalesced_storage_profile_matches_native_ticket_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def trace(individual: bool, maximum_bytes: int) -> tuple[object, tuple[int, ...]]:
        with monkeypatch.context() as patch:
            if individual:
                patch.setattr(
                    CoordinateEnclosureBudget,
                    "_object_storage_profile",
                    _individual_storage_profile,
                )
            maximum_work = 100_000
            outcome: object = None
            snapshot: tuple[int, ...] = ()
            try:
                with NativeExecutionScope(
                    NativeExecutionLimits(maximum_work, 10, 0, maximum_bytes, inf)
                ) as scope:
                    with scope.batch(
                        maximum_work, queries_upper=10, bytes_upper=maximum_bytes
                    ) as ticket:
                        ledger = CoordinateEnclosureBudget(
                            ticket,
                            maximum_work_units=maximum_work,
                            maximum_memory_bytes=1 << 30,
                        )
                        try:
                            with ledger.activate():
                                _profile_operation(ledger, "retained")
                        finally:
                            snapshot = (
                                ledger.work_units,
                                ledger.native_charged_work_units,
                                ledger.retained_basis_bytes,
                                ledger.temporary_bytes_upper,
                                ledger.peak_bytes_upper,
                                ledger._checkpoint_admissions_remaining,
                                ticket.work_units,
                                ticket.managed_bytes,
                            )
            except MeshcoreError as error:
                evidence = error.batch_refusal_evidence
                outcome = (
                    error.status,
                    None if evidence is None else np.asarray(evidence).tolist(),
                )
            return outcome, snapshot

    ample = trace(True, 4 << 20)
    assert ample[0] is None
    assert trace(False, 4 << 20) == ample
    peak = ample[1][4]
    for maximum_bytes in (peak // 2, peak - 129, peak - 1, peak):
        expected = trace(True, maximum_bytes)
        assert trace(False, maximum_bytes) == expected
    assert trace(True, peak // 2)[0] is not None


@pytest.mark.parametrize("native", (False, True), ids=("host", "native"))
def test_managed_storage_headroom_is_the_exact_refusal_threshold(native: bool) -> None:
    def exercise(ticket: HostBatchTicket | NativeBatchTicket) -> None:
        first, second = ticket.managed_storage(), ticket.managed_storage()
        first.set_bound(700)
        headroom = second.headroom()
        assert headroom is not None and headroom == 4096 - 700
        second.set_bound(headroom)
        assert (first.headroom(), second.headroom()) == (700, headroom)
        refusal = MeshcoreError if native else HostBatchEstimateError
        with pytest.raises(refusal):
            first.set_bound(701)
        # A refused ticket admits releases only.
        assert (first.headroom(), second.headroom()) == (700, headroom)
        second.close()
        assert second.headroom() is None
        first.close()
        exercised.append(native)

    exercised: list[bool] = []
    # The refused ticket's own exit may report the same refusal again.
    with suppress(MeshcoreError, HostBatchEstimateError):
        if native:
            with NativeExecutionScope(
                NativeExecutionLimits(16, 0, 0, 4096, inf)
            ) as scope:
                with scope.batch(16, bytes_upper=4096) as native_ticket:
                    exercise(native_ticket)
        else:
            with HostBatchTicket(16, bytes_upper=4096) as host_ticket:
                exercise(host_ticket)
    assert exercised == [native]


@pytest.mark.parametrize("fail", (False, True), ids=("completed", "exception"))
def test_discarded_validity_retires_graphs_and_restores_exact_preexisting_proofs(
    fail: bool,
) -> None:
    import gc
    import weakref

    import jax.numpy as jnp

    from phydrax.discretization import (
        CellBlock,
        CellGeometrySpec,
        CellMesh,
        reference_cell_topology,
    )
    from phydrax.discretization._cell_geometry import coordinate_lagrange_element
    from phydrax.discretization._cell_geometry_validity import (
        _prepare_polynomial_differential,
        CellValidityCertificate,
        certify_cell_geometry_validity,
    )

    def certify(offset: float) -> CellValidityCertificate:
        points = np.asarray(
            reference_cell_topology("hexahedron").vertices, dtype=np.float64
        ).copy()
        points[:, 0] += offset * points[:, 0] ** 2
        mesh = CellMesh(
            points,
            (CellBlock("cells", "hexahedron", np.arange(8)[None]),),
        )
        element = coordinate_lagrange_element("hexahedron", 2)
        nodes = np.asarray(element.reference_nodes, dtype=np.float64).copy()
        nodes[:, 0] += offset * nodes[:, 0] ** 2
        geometry = CellGeometrySpec(
            {"cells": element}, {"cells": np.arange(len(nodes))[None]}, nodes
        )
        # Retain the real expanded differential independently of whichever
        # equivalent factored route the validity dispatcher selects.
        _prepare_polynomial_differential(
            element, nodes, "hexahedron", 3, False, ledger
        )
        return certify_cell_geometry_validity(geometry, mesh=mesh, budget=ledger)

    maximum_work, maximum_bytes = 2_000_000, 64 << 20
    with HostBatchTicket(
        maximum_work, bytes_upper=maximum_bytes, queries_upper=10
    ) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_bytes,
        )
        with ledger.activate(), ledger.live_storage() as enclosing:
            enclosing.set_bound(64)
            existing = certify(0.125)
            assert existing.invalid_count == existing.unresolved_count == 0
            assert ledger.polynomial_differential_cache
            caches = {
                name: (value, value.copy())
                for name, value in vars(ledger).items()
                if isinstance(value, (dict, set))
                and name not in (
                    "exact_predicate_evaluations",
                    "exact_predicate_cache_hits",
                    "source_homotopy_enclosure_cache",
                )
            }
            retained, logical = ledger.retained_basis_bytes, ledger.logical_array_bytes_upper
            temporary = ledger.temporary_bytes_upper
            with pytest.raises(RuntimeError, match="discarded proof probe") if fail else nullcontext():
                with ledger.discarded_preparation_scope():
                    assert certify(0.125) is existing
                    certificate = certify(0.25)
                    assert certificate.invalid_count == certificate.unresolved_count == 0
                    differential = next(
                        value for key, value in ledger.polynomial_differential_cache.items()
                        if key not in caches["polynomial_differential_cache"][1]
                    )
                    array = np.arange(32, dtype=np.float64)
                    device = jnp.arange(4, dtype=jnp.float64)
                    ledger.retain_owner((array, device))
                    references = (
                        weakref.ref(certificate), weakref.ref(differential),
                        weakref.ref(array), weakref.ref(device),
                    )
                    # Replacement and deletion must restore the exact old proof,
                    # not merely remove keys which happened to be new.
                    key = next(iter(caches["validity_certificate_cache"][1]))
                    ledger.validity_certificate_cache[key] = certificate
                    ledger.affine_injectivity_cache.add(((Fraction(17),),))
                    del certificate, differential, array, device
                    ledger.reserve_geometry_queries(2)
                    ledger.admit_work_bound(100)
                    ledger.exact_predicate_evaluations["discarded_probe"] = 3
                    assert ledger.retained_basis_bytes > retained
                    assert ledger.logical_array_bytes_upper > logical
                    counters = (
                        ledger.work_units, ledger.required_work_units,
                        ledger.native_charged_work_units, ticket.geometry_queries,
                        ledger.peak_bytes_upper, ticket.work_units, ticket.managed_bytes,
                    )
                    if fail:
                        raise RuntimeError("discarded proof probe")
            assert (
                ledger.work_units, ledger.required_work_units,
                ledger.native_charged_work_units, ticket.geometry_queries,
                ledger.peak_bytes_upper, ticket.work_units, ticket.managed_bytes,
            ) == counters
            assert (ledger.retained_basis_bytes, ledger.logical_array_bytes_upper) == (
                retained, logical
            )
            assert ledger.temporary_bytes_upper == temporary
            assert enclosing.bound == ledger._live_bytes_upper == 64
            assert ledger.exact_predicate_evaluations["discarded_probe"] == 3
            for name, (original, entries) in caches.items():
                restored = getattr(ledger, name)
                assert restored is original
                assert restored.keys() == entries.keys() if isinstance(restored, dict) else restored == entries
                if isinstance(restored, dict):
                    assert all(restored[key] is value for key, value in entries.items())
            gc.collect()
            assert all(reference() is None for reference in references)
            assert certify(0.125) is existing


@pytest.mark.parametrize("resource", ("work", "storage"))
def test_discarded_snapshot_is_admitted_before_cache_copy(resource: str) -> None:
    from phydrax.discretization._coordinate_enclosure import bernstein_coefficients

    with HostBatchTicket(10_000, bytes_upper=1 << 20) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=10_000, maximum_memory_bytes=1 << 20
        )
        polynomial: Polynomial = {(0,): Fraction(3, 7), (1,): Fraction(5, 11)}
        original = bernstein_coefficients(polynomial, "simplex", 1, budget=ledger)
        retained = ledger.retained_basis_bytes
        objects = ledger.retained_objects.copy()
        with ledger.bound_stage(
            0 if resource == "work" else 10_000,
            retained if resource == "storage" else 1 << 20,
        ):
            with pytest.raises(CoordinateEnclosureResourceError):
                with ledger.discarded_preparation_scope():
                    pytest.fail("Unadmitted snapshot entered its consuming body.")
        assert ledger.retained_basis_bytes == retained
        assert ledger.retained_objects == objects
        assert ledger.temporary_bytes_upper == 0
        assert bernstein_coefficients(polynomial, "simplex", 1, budget=ledger) is original


def _discarded_request_work(ledger: CoordinateEnclosureBudget) -> None:
    from phydrax.discretization._coordinate_enclosure import bernstein_coefficients

    with ledger.discarded_preparation_scope():
        polynomial: Polynomial = {(0,): Fraction(3, 7), (1,): Fraction(5, 11)}
        controls = bernstein_coefficients(polynomial, "simplex", 1, budget=ledger)
        array = np.arange(16, dtype=np.float64)
        ledger.retain_owner(array)
        ledger.reserve_geometry_queries(2)
        ledger.reserve(5)
        charged = ledger.work_units
        peak = ledger.peak_bytes_upper
        del controls, array
    assert ledger.work_units == charged
    assert ledger.peak_bytes_upper == peak
    assert ledger.retained_basis_bytes == ledger.logical_array_bytes_upper == 0
    assert ledger.retained_objects == ledger.retained_owner_objects == {}
    assert not ledger.bernstein_coefficient_cache
    # The independently admitted next step still pays the same request ledger.
    ledger.reserve(7)


@pytest.mark.parametrize("shortfall", (0, 1), ids=("R", "R-minus-one"))
def test_discarded_preparation_preserves_native_same_request_refusal(
    shortfall: int,
) -> None:
    maximum_bytes = 4 << 20
    with HostBatchTicket(100_000, bytes_upper=maximum_bytes, queries_upper=10) as host:
        reference = CoordinateEnclosureBudget(
            host, maximum_work_units=100_000, maximum_memory_bytes=maximum_bytes
        )
        _discarded_request_work(reference)
        required = reference.work_units
    scope = NativeExecutionScope(
        NativeExecutionLimits(required - shortfall, 10, 0, maximum_bytes, inf)
    )
    with pytest.raises(MeshcoreError) if shortfall else nullcontext():
        with scope:
            with scope.batch(
                required - shortfall, queries_upper=10, bytes_upper=maximum_bytes
            ) as ticket:
                ledger = CoordinateEnclosureBudget(
                    ticket,
                    maximum_work_units=required,
                    maximum_memory_bytes=maximum_bytes,
                )
                try:
                    _discarded_request_work(ledger)
                finally:
                    assert ledger.work_units == ledger.native_charged_work_units == ticket.work_units
                    assert ledger.required_work_units == required
                    assert ledger.work_units == required - (7 if shortfall else 0)
                    assert ledger.retained_basis_bytes == ledger.logical_array_bytes_upper == 0
                    assert ledger.retained_objects == ledger.retained_owner_objects == {}
                    assert ticket.geometry_queries == 2
    assert scope.evidence is not None
    assert scope.evidence.status is (
        MeshcoreStatus.CAPACITY_EXCEEDED if shortfall else MeshcoreStatus.OK
    )


def test_discarded_scope_restores_mutable_contact_cache_closure() -> None:
    from phydrax.geometry._exact_polyhedral_geometry import TriangleContactPreparation

    maximum_work, maximum_bytes = 100_000, 4 << 20
    with HostBatchTicket(maximum_work, bytes_upper=maximum_bytes) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=maximum_work,
            maximum_memory_bytes=maximum_bytes,
        )
        contact = TriangleContactPreparation(ledger)
        ledger.exact_triangle_contact_preparation = contact
        triangle = (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1), Fraction(0)),
        )
        original_box = contact.box(triangle)
        source = ((0, 1, 2), "version", b"points", "contact", 0, 0)
        record = (
            np.asarray(((0, 1),), dtype=np.int64),
            np.asarray((False,)), np.asarray((True,)),
        )
        ledger.retain_basis((source, record))
        records = [record]
        ledger.triangle_contact_batches[source] = records
        retained = ledger.retained_basis_bytes
        temporary = ledger.temporary_bytes_upper
        with ledger.discarded_preparation_scope():
            shifted = tuple(tuple(value + 1 for value in point) for point in triangle)
            contact.box(shifted)
            contact.plane(shifted)
            records.append(record)
            ledger.exact_triangle_contact_preparation = TriangleContactPreparation(ledger)
            ledger.exact_triangle_contact_preparation.box(shifted)
        assert ledger.exact_triangle_contact_preparation is contact
        assert contact.bounds == {triangle: original_box}
        assert not contact.planes and not contact.projections and not contact.sides
        assert ledger.triangle_contact_batches[source] is records
        assert len(records) == 1 and records[0] is record
        assert ledger.retained_basis_bytes == retained
        assert ledger.temporary_bytes_upper == temporary
        assert ledger._live_bytes_upper == 0


def _admitted_measure_bank(budget: CoordinateEnclosureBudget, count: int) -> Any:
    """Isolate the already-authenticated cache phase, not public receipt admission."""
    from types import MappingProxyType

    from phydrax.discretization._coordinate_operands import PreparedCoordinateOperands

    prepared = object.__new__(PreparedCoordinateOperands)
    snapshots = tuple(
        ((str(index), "revision", ()), (float(index + 1), 0.0))
        for index in range(count)
    )
    object.__setattr__(prepared, "_measure_snapshots", snapshots)
    object.__setattr__(prepared, "_power_witnesses", MappingProxyType({}))
    object.__setattr__(prepared, "_polynomial_snapshots", ())
    object.__setattr__(prepared, "_affine_areas", ())
    budget._prepared_coordinate_operands = prepared
    budget.exact_cell_measure_cache.update(snapshots)
    return prepared


@pytest.mark.parametrize("count", (0, 1, 1285))
def test_measure_readmission_charges_one_visit_per_snapshot(count: int) -> None:
    with HostBatchTicket(10000, bytes_upper=8192) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=10000, maximum_memory_bytes=8192
        )
        prepared = _admitted_measure_bank(ledger, count)
        original = ledger.exact_cell_measure_cache.copy()
        with ledger.measurement() as phase:
            for _ in range(3):
                ledger.admit_coordinate_operands(prepared)
        assert phase.work_units == 3 * count
        # The previous two traversals cost 6 * count for these re-admissions.
        assert 6 * count - phase.work_units == 3 * count
        assert ledger.exact_cell_measure_cache == original
        assert ledger.retained_basis_bytes == 0


@pytest.mark.parametrize("remaining", (1284, 1285))
def test_measure_readmission_exact_work_boundary(remaining: int) -> None:
    with HostBatchTicket(10000, bytes_upper=8192) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=remaining, maximum_memory_bytes=8192
        )
        prepared = _admitted_measure_bank(ledger, 1285)
        original = ledger.exact_cell_measure_cache.copy()
        if remaining == 1285:
            ledger.admit_coordinate_operands(prepared)
        else:
            with pytest.raises(CoordinateEnclosureResourceError):
                ledger.admit_coordinate_operands(prepared)
        assert ledger.work_units == remaining
        assert ledger.exact_cell_measure_cache == original
        assert ledger._prepared_coordinate_operands is prepared
        assert ledger.retained_basis_bytes == 0


@pytest.mark.parametrize(
    ("fault", "message"),
    (
        ("missing", "changed after operand admission"),
        ("conflict", "conflict with current exact cell measures"),
        ("signed-zero", "conflict with current exact cell measures"),
        ("power", "conflict with the current exact power construction"),
        ("polynomial", "conflict with current polynomial preparations"),
        ("area", "changed after operand admission"),
        ("later-conflict", "conflict with current exact cell measures"),
    ),
)
def test_measure_readmission_preserves_mutation_refusal_precedence(
    fault: str, message: str, budget: CoordinateEnclosureBudget,
) -> None:
    from types import MappingProxyType

    prepared = _admitted_measure_bank(budget, 2)
    first_key, first_integral = prepared._measure_snapshots[0]
    second_key, _ = prepared._measure_snapshots[1]
    del budget.exact_cell_measure_cache[first_key]
    if fault == "conflict":
        budget.exact_cell_measure_cache[first_key] = (99.0, 0.0)
    elif fault == "signed-zero":
        budget.exact_cell_measure_cache[first_key] = (first_integral[0], -0.0)
    elif fault == "later-conflict":
        budget.exact_cell_measure_cache[second_key] = (99.0, 0.0)
    elif fault == "power":
        object.__setattr__(prepared, "_power_witnesses", MappingProxyType({"source": object()}))
    elif fault == "polynomial":
        object.__setattr__(prepared, "_polynomial_snapshots", ((first_key, ((),), True),))
    elif fault == "area":
        object.__setattr__(prepared, "_affine_areas", ((("absent",), Fraction(1)),))
    original = budget.exact_cell_measure_cache.copy()
    with pytest.raises(ValueError, match=message):
        budget.admit_coordinate_operands(prepared)
    assert budget.exact_cell_measure_cache == original
    assert budget._prepared_coordinate_operands is prepared
    assert not budget.power_witness_cache
    assert not budget.polynomial_cache
    assert not budget.affine_chain_intersection_cache
    assert budget.retained_basis_bytes == 0


@pytest.mark.parametrize("vertices", (0, 3))
def test_coordinate_source_cache_analytic_bound_covers_realized_storage(
    vertices: int,
) -> None:
    from phydrax.discretization._cell_geometry import (
        cell_geometry_preparation_bounds,
        CellGeometrySpec,
        coordinate_lagrange_element,
    )

    geometry = CellGeometrySpec(
        {"cells": coordinate_lagrange_element("triangle", 1)},
        {"cells": np.arange(vertices, dtype=np.int64).reshape((-1, 3))},
        np.arange(2 * vertices, dtype=np.float64).reshape((-1, 2)),
    )
    work, memory = cell_geometry_preparation_bounds(geometry)
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        )
        bank = geometry.source_coordinates(budget=ledger)
        assert len(bank) == vertices
        assert ledger.peak_bytes_upper <= memory
        if vertices == 0:
            assert ledger.peak_bytes_upper == memory == 1024


@pytest.mark.parametrize("headroom", (0, -1), ids=("R", "R-minus-one"))
def test_empty_coordinate_cache_refuses_before_source_allocation(
    monkeypatch: pytest.MonkeyPatch, headroom: int,
) -> None:
    from phydrax.discretization import _cell_geometry as geometry_owner

    geometry = geometry_owner.CellGeometrySpec(
        {"cells": geometry_owner.coordinate_lagrange_element("triangle", 1)},
        {"cells": np.empty((0, 3), dtype=np.int64)},
        np.empty((0, 2), dtype=np.float64),
    )
    work, memory = geometry_owner.cell_geometry_preparation_bounds(geometry)
    assert memory == 1024
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory + headroom
        )
        if headroom == 0:
            assert geometry.source_coordinates(budget=ledger) == ()
        else:
            def forbidden_fetch(*args: Any, **kwargs: Any) -> Any:
                raise AssertionError("Source allocation preceded storage admission")

            monkeypatch.setattr(geometry_owner.jax, "device_get", forbidden_fetch)
            with pytest.raises(CoordinateEnclosureResourceError) as caught:
                geometry.source_coordinates(budget=ledger)
            assert (caught.value.limit, caught.value.requested) == (1023, 1024)
            assert ledger.peak_bytes_upper == ledger.temporary_bytes_upper == 0


@pytest.mark.parametrize("vertices", (0, 3))
def test_validity_bound_includes_source_bank_retention(vertices: int) -> None:
    from phydrax.discretization._cell_geometry import (
        CellGeometrySpec,
        coordinate_lagrange_element,
    )
    from phydrax.discretization._cell_geometry_validity import (
        cell_validity_preparation_bounds,
        CellValidityPolicy,
    )
    from phydrax.discretization._coordinate_enclosure import (
        prepared_coordinate_source_bank,
    )

    geometry = CellGeometrySpec(
        {"cells": coordinate_lagrange_element("triangle", 1)},
        {"cells": np.arange(vertices, dtype=np.int64).reshape((-1, 3))},
        np.arange(2 * vertices, dtype=np.float64).reshape((-1, 2)),
    )
    work, memory, _ = cell_validity_preparation_bounds(geometry, CellValidityPolicy())
    with HostBatchTicket(work, bytes_upper=memory) as ticket:
        ledger = CoordinateEnclosureBudget(
            ticket, maximum_work_units=work, maximum_memory_bytes=memory
        )
        bank = prepared_coordinate_source_bank(geometry, budget=ledger)
        assert len(bank) == vertices
        assert 0 < ledger.work_units <= work
        assert 0 < ledger.peak_bytes_upper <= memory
