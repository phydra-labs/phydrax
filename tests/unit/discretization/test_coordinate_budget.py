#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Coordinate work admission, storage lifetime, and phase-evidence contracts."""

from collections.abc import Iterator
from fractions import Fraction
from math import inf
from time import sleep

import numpy as np
import pytest

from phydrax._host_storage import declared_object_storage_bytes
from phydrax._meshcore import (
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._resource_ticket import HostBatchEstimateError, HostBatchTicket
from phydrax.discretization._coordinate_budget import (
    CoordinateEnclosureBudget,
    CoordinateEnclosureMeasurement,
    CoordinateEnclosureResourceError,
    CoordinateLiveStorage,
    reserve_polynomial,
)
from phydrax.discretization._coordinate_enclosure import multiply, Polynomial


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
