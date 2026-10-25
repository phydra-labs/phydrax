# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Opt-in simultaneous admission observations never reprice native allocation."""

from __future__ import annotations

import ctypes
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from queue import Queue
from typing import cast, Never

import numpy as np
import pytest

import phydrax._meshcore as meshcore
from phydrax._meshcore import (
    is_native_host_buffer,
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeBatchTicket,
    NativeExecutionEvidence,
    NativeExecutionLimits,
    NativeExecutionScope,
    NativeParallelExecution,
)


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore required"
)


def _limits() -> NativeExecutionLimits:
    return NativeExecutionLimits(100, 100, 100, 4 * 1024**2, 120.0)


def test_unobserved_scope_has_no_managed_admission_measurement() -> None:
    with NativeExecutionScope(_limits()) as scope:
        with scope.batch(1, bytes_upper=65536) as ticket:
            ticket.consume(1, managed_bytes=8192)
    evidence = scope.completed_evidence()
    assert evidence.managed_admission_peak_bytes_upper is None
    assert evidence.work_counters.construction_work == 1


def test_simultaneous_native_host_and_child_reservation_envelope() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.host_workspace() as cache:
            cache.set_bound(4096)
            with scope.batch(1, bytes_upper=131072) as ticket:
                outside = (
                    scope.limits.maximum_managed_bytes
                    - scope.remaining().remaining_managed_bytes
                    - ticket.bytes_upper
                )
                with ticket.native_target():
                    values = scope.allocate_host_array((2048,), np.uint8)
                    ticket.consume(1, managed_bytes=8192)
                    with ticket.batch(0, bytes_upper=24576):
                        expected = outside + (
                            ticket.bytes_upper - ticket.remaining_managed_bytes
                        )
                    del values
            expected = max(expected, outside + ticket.managed_bytes)
    evidence = scope.completed_evidence()
    assert evidence.managed_admission_peak_bytes_upper == expected
    assert expected < 131072
    assert expected >= 4096 + 8192 + 24576
    assert evidence.memory_counters.peak_bytes > 0
    assert evidence.host_storage_peak_bytes_upper >= 4096


def test_complementary_ticket_peaks_are_maxed_not_summed() -> None:
    candidates: list[int] = []
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        for host_bytes in (8192, 16384):
            with scope.batch(0, bytes_upper=65536) as ticket:
                outside = (
                    scope.limits.maximum_managed_bytes
                    - scope.remaining().remaining_managed_bytes
                    - ticket.bytes_upper
                )
                ticket.consume(managed_bytes=host_bytes)
                ticket.consume(managed_bytes=0)
            candidates.append(outside + ticket.managed_bytes)
    observed = scope.completed_evidence().managed_admission_peak_bytes_upper
    assert observed == max(candidates)
    assert observed < sum(candidates)


def test_direct_root_array_and_host_admissions_are_simultaneous_candidates() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.host_workspace() as cache:
            cache.set_bound(8192)
            values = scope.allocate_host_array((4096,), np.uint8)
            assert is_native_host_buffer(cast(ctypes.Array, values.base))
            expected = (
                scope.limits.maximum_managed_bytes
                - scope.remaining().remaining_managed_bytes
            )
            del values
    evidence = scope.completed_evidence()
    assert evidence.managed_admission_peak_bytes_upper == expected
    assert expected >= 8192 + 4096


@pytest.mark.parametrize("mutation", ("grow", "shrink", "close", "array-release"))
def test_known_decreases_preserve_fixed_phase_admission_not_outside_growth(
    mutation: str,
) -> None:
    with (
        pytest.raises(MeshcoreError) if mutation == "grow" else nullcontext()
    ) as refusal, NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.host_workspace() as cache:
            cache.set_bound(4096)
            values = scope.allocate_host_array((1024,), np.uint8)
            with scope.batch(0, bytes_upper=65536) as ticket:
                with ticket.native_target():
                    outside = (
                        scope.limits.maximum_managed_bytes
                        - scope.remaining().remaining_managed_bytes
                        - ticket.bytes_upper
                    )
                    ticket.consume(managed_bytes=8192)
                    if mutation == "grow":
                        cache.set_bound(8192)
                    elif mutation == "shrink":
                        cache.set_bound(2048)
                    elif mutation == "close":
                        cache.close()
                    else:
                        del values
            scope.observe_managed_admission()  # Must not reset incompleteness.
    observed = scope.completed_evidence().managed_admission_peak_bytes_upper
    if mutation == "grow":
        assert observed is None
        assert refusal is not None
        assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    else:
        assert observed == outside + ticket.managed_bytes


def test_overlapping_independent_root_reservations_are_not_synchronized() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(1, bytes_upper=65536) as first:
            first.consume(managed_bytes=1024)
            with scope.batch(0, bytes_upper=32768) as second:
                second.consume(managed_bytes=2048)
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_explicit_tighter_nested_window_measures_its_own_admission() -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with NativeExecutionScope(
            NativeExecutionLimits(100, 100, 100, 32768, 120.0)
        ) as borrowed:
            borrowed.observe_managed_admission()
            with borrowed.batch(0, bytes_upper=16384) as ticket:
                ticket.consume(managed_bytes=4096)
    assert borrowed.completed_evidence().managed_admission_peak_bytes_upper == 4096
    assert root.completed_evidence().managed_admission_peak_bytes_upper == 4096


def test_nested_children_remain_in_root_combined_peak() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(0, bytes_upper=131072) as root:
            outside = (
                scope.limits.maximum_managed_bytes
                - scope.remaining().remaining_managed_bytes
                - root.bytes_upper
            )
            with root.native_target():
                with NativeExecutionScope(_limits()) as borrowed:
                    with borrowed.batch(0, bytes_upper=65536) as child:
                        expected = outside + (
                            root.bytes_upper - root.remaining_managed_bytes
                        )
                        with child.native_target():
                            values = borrowed.allocate_host_array((4096,), np.uint8)
                            child.consume(managed_bytes=8192)
                            del values
        expected = max(expected, outside + root.managed_bytes)
    assert borrowed.completed_evidence().managed_admission_peak_bytes_upper is None
    assert scope.completed_evidence().managed_admission_peak_bytes_upper == expected


@pytest.mark.parametrize("observe", (False, True))
def test_child_admission_upper_covers_released_historical_host_phase(
    observe: bool,
) -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as outside:
            outside.set_bound(1024)
            with NativeExecutionScope(_limits()) as child:
                if observe:
                    child.observe_managed_admission()
                with ExitStack() as retained:
                    with child.batch(1, bytes_upper=65536) as phase:
                        with phase.native_target():
                            phase.consume(managed_bytes=8192)
                            phase.consume(managed_bytes=0)
                            storage = retained.enter_context(child.host_workspace())
                            storage.set_bound(4096)
    evidence = child.completed_evidence()
    if observe:
        # Settlement's outside-plus-historical bound is deliberately more
        # conservative than the synchronized events, not a physical RSS peak.
        assert evidence.host_storage_peak_bytes_upper > child._managed_observation_peak
        assert evidence.managed_admission_peak_bytes_upper == (
            evidence.host_storage_peak_bytes_upper
        )
        assert root.completed_evidence().managed_admission_peak_bytes_upper == (
            evidence.managed_admission_peak_bytes_upper
        )
    else:
        assert evidence.managed_admission_peak_bytes_upper is None


@pytest.mark.parametrize("simultaneous", (False, True))
def test_child_peak_does_not_copy_parent_history_or_sum_disjoint_peaks(
    simultaneous: bool,
) -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as outside:
            outside.set_bound(8192)
            earlier = (
                root.limits.maximum_managed_bytes
                - root.remaining().remaining_managed_bytes
            )
            if not simultaneous:
                outside.set_bound(0)
            with NativeExecutionScope(_limits()) as child:
                child.observe_managed_admission()
                with child.host_workspace() as storage:
                    storage.set_bound(4096)
                    measured = (
                        child.limits.maximum_managed_bytes
                        - child.remaining().remaining_managed_bytes
                    )
    child_peak = child.completed_evidence().managed_admission_peak_bytes_upper
    assert child_peak == measured
    assert (child_peak > earlier) is simultaneous
    assert root.completed_evidence().managed_admission_peak_bytes_upper == (
        max(earlier, measured)
    )


def test_observed_child_exact_managed_boundary_uses_actual_outside_baseline() -> None:
    def trial(limit: int) -> NativeExecutionEvidence:
        limits = NativeExecutionLimits(100, 100, 100, limit, 120.0)
        with NativeExecutionScope(limits) as root:
            root.observe_managed_admission()
            with root.host_workspace() as outside:
                outside.set_bound(1024)
                with NativeExecutionScope(limits) as child:
                    child.observe_managed_admission()
                    values = child.allocate_host_array((4096,), np.uint8)
                    del values
        evidence = child.completed_evidence()
        assert root.completed_evidence().managed_admission_peak_bytes_upper == (
            evidence.managed_admission_peak_bytes_upper
        )
        return evidence

    required = trial(_limits().maximum_managed_bytes).managed_admission_peak_bytes_upper
    assert required is not None and required >= 5120
    assert trial(required).managed_admission_peak_bytes_upper == required
    with pytest.raises(MeshcoreError) as caught:
        trial(required - 1)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert meshcore.current_native_execution_scope() is None


def test_late_child_observation_cannot_recover_parent_owned_history() -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with NativeExecutionScope(_limits()) as child:
            values = child.allocate_host_array((4096,), np.uint8)
            del values
            child.observe_managed_admission()
    assert child.completed_evidence().managed_admission_peak_bytes_upper is None
    assert root.completed_evidence().managed_admission_peak_bytes_upper is None


@pytest.mark.parametrize("borrow_target", (False, True))
@pytest.mark.parametrize("fail", (False, True))
def test_observed_nested_intervals_measure_local_allocations(
    borrow_target: bool, fail: bool,
) -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as outside:
            outside.set_bound(1024)
            target = root.batch(1, bytes_upper=65536) if borrow_target else None
            with target if target is not None else nullcontext():
                with target.native_target() if target is not None else nullcontext():
                    with NativeExecutionScope(_limits()) as child:
                        child.observe_managed_admission()
                        retained = child.allocate_host_array((2048,), np.uint8)
                        expectation = pytest.raises(ValueError) if fail else nullcontext()
                        with expectation, NativeExecutionScope(_limits()) as grandchild:
                            grandchild.observe_managed_admission()
                            with grandchild.ticket_view() as view:
                                with view.native_target():
                                    values = grandchild.allocate_host_array(
                                        (4096,), np.uint8
                                    )
                                    del values
                            if fail:
                                raise ValueError("consumer failed")
                        grandchild_peak = (
                            grandchild.completed_evidence()
                            .managed_admission_peak_bytes_upper
                        )
                        assert grandchild_peak is not None
                        assert grandchild_peak >= 1024 + 2048 + 4096
                        del retained
                    child_peak = (
                        child.completed_evidence().managed_admission_peak_bytes_upper
                    )
                    assert child_peak is not None
                    assert child_peak >= grandchild_peak
            with root.host_workspace() as later:
                later.set_bound(8192)
    root_peak = root.completed_evidence().managed_admission_peak_bytes_upper
    assert root_peak is not None
    assert root_peak >= max(child_peak, 1024 + 8192)
    assert meshcore.current_native_execution_scope() is None


def test_full_inherited_escrow_remains_admissible_under_its_original_quota() -> None:
    from phydrax.meshing import MeshingFailure, MeshingFailureCategory, MeshingLimits
    from phydrax.meshing._measurements import NativeExecutionRecord
    from phydrax.meshing._volume_generation import require_native_preparation_allowance

    capacity = 1 << 20
    limits = NativeExecutionLimits(100, 100, 100, capacity, 120.0)
    with NativeExecutionScope(limits) as root:
        root.observe_managed_admission()
        with root.ticket_view() as outer, outer.native_target():
            with NativeExecutionScope(limits) as child:
                child.observe_managed_admission()
                with (
                    child.ticket_view() as inner,
                    inner.native_target(),
                    child.host_workspace() as storage,
                ):
                    storage.set_bound(4096)
                    with NativeExecutionScope(limits):
                        storage.set_bound(2048)
    evidence = child.completed_evidence()
    peak = evidence.managed_admission_peak_bytes_upper
    assert peak is not None and 4096 <= peak <= capacity
    assert root.completed_evidence().managed_admission_peak_bytes_upper == peak
    receipt = NativeExecutionRecord(evidence, owner_id="bounded-inherited-request")
    require_native_preparation_allowance(
        MeshingLimits(maximum_scratch_bytes=capacity), receipt
    )
    with pytest.raises(MeshingFailure) as refusal:
        require_native_preparation_allowance(
            MeshingLimits(maximum_scratch_bytes=peak - 1), receipt
        )
    assert refusal.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


@pytest.mark.parametrize("outside", (0, 8192), ids=("empty-baseline", "live-outside"))
def test_partial_escrow_is_counted_once_through_tighter_descendants(outside: int) -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as storage:
            storage.set_bound(outside)
            with root.batch(10, bytes_upper=65536) as outer, outer.native_target():
                with NativeExecutionScope(_limits()) as child:
                    child.observe_managed_admission()
                    baseline = child._managed_observation_used()
                    with child.batch(1, bytes_upper=32768) as phase:
                        admitted = child._managed_observation_used()
                        assert admitted - baseline == 32768
                        with phase.native_target():
                            with NativeExecutionScope(
                                NativeExecutionLimits(100, 100, 100, 16384, 120.0)
                            ):
                                assert child._managed_observation_used() == admitted
                            phase.consume(1, managed_bytes=4096)
    assert child.completed_evidence().status is MeshcoreStatus.OK
    assert root.completed_evidence().managed_admission_peak_bytes_upper is not None


def test_impossible_observed_event_is_not_clipped_to_the_request_quota() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        scope._managed_observation_peak = _limits().maximum_managed_bytes + 1
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_unobserved_descendants_borrow_their_nearest_observed_phase() -> None:
    limits = _limits()
    with NativeExecutionScope(limits) as root:
        root.observe_managed_admission()
        with root.batch(10, bytes_upper=65536) as parent, parent.native_target():
            with NativeExecutionScope(limits) as child:
                child.observe_managed_admission()
                with (
                    child.ticket_view() as view,
                    view.native_target(),
                    child.host_workspace() as storage,
                ):
                    storage.set_bound(4096)
                    with NativeExecutionScope(limits), NativeExecutionScope(limits):
                        storage.set_bound(2048)
    child_evidence = child.completed_evidence()
    root_evidence = root.completed_evidence()
    child_peak = child_evidence.managed_admission_peak_bytes_upper
    root_peak = root_evidence.managed_admission_peak_bytes_upper
    assert child_evidence.status is root_evidence.status is MeshcoreStatus.OK
    assert child_peak is not None and root_peak is not None
    assert 4096 <= child_peak <= root_peak <= limits.maximum_managed_bytes
    assert meshcore.current_native_execution_scope() is None


def test_refusal_preserves_observed_ticket_peak_and_status() -> None:
    with pytest.raises(MeshcoreError) as caught:
        with NativeExecutionScope(_limits()) as scope:
            scope.observe_managed_admission()
            with scope.batch(0, bytes_upper=65536) as ticket:
                outside = (
                    scope.limits.maximum_managed_bytes
                    - scope.remaining().remaining_managed_bytes
                    - ticket.bytes_upper
                )
                ticket.consume(managed_bytes=8192)
                ticket.refuse(101)
    evidence = scope.completed_evidence()
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert evidence.managed_admission_peak_bytes_upper == outside + ticket.managed_bytes
    assert evidence.managed_admission_peak_bytes_upper >= 8192


def test_uncovered_native_only_span_is_not_filled_by_independent_peak_sum() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        library = scope._library
        assert library is not None
        owner = ctypes.c_void_p()
        data = ctypes.c_void_p()
        status = MeshcoreStatus(
            library["phx_mc_execution_allocate_host_array"](
                scope._handle, 32768, 1,
                ctypes.addressof(owner), ctypes.addressof(data),
            )
        )
        assert status is MeshcoreStatus.OK
        try:
            assert owner.value is not None
            assert data.value is not None
        finally:
            library["phx_mc_execution_free_host_array"](owner)
    evidence = scope.completed_evidence()
    assert evidence.memory_counters.peak_bytes >= 32768
    assert evidence.managed_admission_peak_bytes_upper is None


@pytest.mark.parametrize("value", (True, -1, 2**64, np.uint64(1), 1.5))
def test_evidence_requires_optional_python_uint64_observation(value: object) -> None:
    with NativeExecutionScope(_limits()) as scope:
        pass
    evidence = scope.completed_evidence()
    with pytest.raises(ValueError, match="managed_admission_peak_bytes_upper"):
        NativeExecutionEvidence(
            evidence.work_counters,
            evidence.memory_counters,
            evidence.elapsed_seconds,
            evidence.status,
            evidence.externally_charged_work,
            evidence.externally_charged_geometry_queries,
            evidence.native_primitive_queries,
            evidence.host_storage_live_bytes_upper,
            evidence.host_storage_peak_bytes_upper,
            value,  # ty: ignore[invalid-argument-type] -- Invalid runtime input is intentional.
        )


def test_late_opt_in_cannot_hide_an_already_settled_reservation_peak() -> None:
    with NativeExecutionScope(_limits()) as scope:
        with scope.batch(0, bytes_upper=131072) as ticket:
            with ticket.batch(0, bytes_upper=65536):
                pass
        scope.observe_managed_admission()
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_managed_admission_constructor_argument_is_required() -> None:
    with NativeExecutionScope(_limits()) as scope:
        pass
    evidence = scope.completed_evidence()
    with pytest.raises(TypeError, match="managed_admission_peak_bytes_upper"):
        NativeExecutionEvidence(  # ty: ignore[missing-argument] -- Required-argument regression.
            evidence.work_counters,
            evidence.memory_counters,
            evidence.elapsed_seconds,
            evidence.status,
            evidence.externally_charged_work,
            evidence.externally_charged_geometry_queries,
            evidence.native_primitive_queries,
            evidence.host_storage_live_bytes_upper,
            evidence.host_storage_peak_bytes_upper,
        )


def test_restored_observation_is_validated_without_integer_coercion() -> None:
    with NativeExecutionScope(_limits()) as scope:
        pass
    evidence = scope.completed_evidence()
    object.__setattr__(evidence, "managed_admission_peak_bytes_upper", np.uint64(1))
    with pytest.raises(ValueError, match="managed_admission_peak_bytes_upper"):
        evidence.validate_restored()


def test_retained_cache_ticket_is_outside_coverage_not_a_work_phase() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(0, bytes_upper=4096) as retained:
            retained.consume(managed_bytes=4096)
            with scope.host_workspace() as workspace:
                workspace.set_bound(4096)
                workspace.set_bound(8192)
                with scope.batch(1, bytes_upper=65536) as phase:
                    outside = (
                        scope.limits.maximum_managed_bytes
                        - scope.remaining().remaining_managed_bytes
                        - phase.bytes_upper
                    )
                    phase.consume(managed_bytes=16384)
                expected = outside + phase.managed_bytes
    assert scope.completed_evidence().managed_admission_peak_bytes_upper == expected
    assert outside >= 4096 + 8192


def test_empty_deadline_checkpoint_before_opt_in_does_not_hide_storage() -> None:
    with NativeExecutionScope(_limits()) as scope:
        with scope.batch(0) as checkpoint:
            checkpoint.checkpoint()
        scope.observe_managed_admission()
        with scope.batch(1, bytes_upper=65536) as phase:
            outside = (
                scope.limits.maximum_managed_bytes
                - scope.remaining().remaining_managed_bytes
                - phase.bytes_upper
            )
            phase.consume(managed_bytes=8192)
        expected = outside + phase.managed_bytes
    assert scope.completed_evidence().managed_admission_peak_bytes_upper == expected


def test_empty_independent_checkpoint_does_not_change_phase_outside_baseline() -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(1, bytes_upper=65536) as phase:
            outside = (
                scope.limits.maximum_managed_bytes
                - scope.remaining().remaining_managed_bytes
                - phase.bytes_upper
            )
            phase.consume(managed_bytes=8192)
            with scope.batch(0) as checkpoint:
                checkpoint.checkpoint()
        expected = outside + phase.managed_bytes
    assert scope.completed_evidence().managed_admission_peak_bytes_upper == expected


def test_nonbinding_nested_window_preserves_root_outside_and_child_escrow() -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as outside_storage:
            outside_storage.set_bound(4096)
            with NativeExecutionScope(_limits()) as nested:
                with nested.batch(1, bytes_upper=131072) as phase:
                    outside = (
                        root.limits.maximum_managed_bytes
                        - nested.remaining().remaining_managed_bytes
                        - phase.bytes_upper
                    )
                    with phase.native_target():
                        values = nested.allocate_host_array((4096,), np.uint8)
                        phase.consume(managed_bytes=8192)
                        with phase.batch(0, bytes_upper=24576):
                            expected = outside + (
                                phase.bytes_upper - phase.remaining_managed_bytes
                            )
                        del values
                expected = max(expected, outside + phase.managed_bytes)
    assert nested.completed_evidence().managed_admission_peak_bytes_upper is None
    assert root.completed_evidence().managed_admission_peak_bytes_upper == expected
    assert expected >= 4096 + 8192 + 24576


def test_borrowing_an_unobserved_parent_ticket_cannot_create_root_history() -> None:
    with NativeExecutionScope(_limits()) as root:
        with root.batch(0, bytes_upper=65536) as reservation:
            with reservation.native_target():
                with NativeExecutionScope(_limits()) as borrowed:
                    borrowed.observe_managed_admission()
                    with borrowed.batch(0, bytes_upper=16384) as ticket:
                        ticket.consume(managed_bytes=4096)
    assert root.completed_evidence().managed_admission_peak_bytes_upper is None
    assert borrowed.completed_evidence().managed_admission_peak_bytes_upper is None


@pytest.mark.parametrize("mutation", ("resize", "close", "array-release"))
def test_nested_nonbinding_window_requires_its_outside_baseline_to_stay_held(
    mutation: str,
) -> None:
    with NativeExecutionScope(_limits()) as root:
        root.observe_managed_admission()
        with root.host_workspace() as outside_storage:
            outside_storage.set_bound(4096)
            values = root.allocate_host_array((1024,), np.uint8)
            with NativeExecutionScope(_limits()) as nested:
                with nested.host_workspace() as local:
                    local.set_bound(8192)
                    if mutation == "resize":
                        outside_storage.set_bound(2048)
                    elif mutation == "close":
                        outside_storage.close()
                    else:
                        del values
    assert root.completed_evidence().managed_admission_peak_bytes_upper is None


def test_outside_release_cannot_credit_a_fixed_phase_quota() -> None:
    def execute(cap: int) -> int:
        limits = NativeExecutionLimits(100, 100, 100, cap, 120.0)
        with NativeExecutionScope(limits) as scope:
            scope.observe_managed_admission()
            with scope.host_workspace() as cache:
                cache.set_bound(4096)
                values = scope.allocate_host_array((1024,), np.uint8)
                with scope.batch(
                    1, bytes_upper=min(65536, scope.remaining().remaining_managed_bytes)
                ) as phase:
                    with phase.native_target(), scope.host_workspace() as storage:
                        storage.set_bound(8192)
                        first = np.empty((8192,), dtype=np.uint8)
                        cache.close()
                        del first
                        storage.set_bound(0)
                        storage.set_bound(16384)
                        second = np.empty((16384,), dtype=np.uint8)
                        del second
                        storage.set_bound(0)
                del values
        observed = scope.completed_evidence().managed_admission_peak_bytes_upper
        assert observed is not None
        return observed

    required = execute(1 << 20)
    assert execute(required) == required
    with pytest.raises(MeshcoreError) as failure:
        execute(required - 1)
    assert failure.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


def test_scope_view_prepays_actual_ffi_without_new_quota_or_callable_identity() -> None:
    import phydrax._meshcore as owner

    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        view = scope.ticket_view()
        assert scope._library is not None
        function = scope._library["phx_mc_orient2d"]
        assert function is scope._library._functions["phx_mc_orient2d"]
        pointer = ctypes.cast(function, ctypes.c_void_p).value
        with view.native_target():
            phase = scope._target_for_thread()
            assert phase is not None and phase._mapped
            assert phase.work_upper == _limits().maximum_work_units
            signs = owner.exact_orient2d(
                [[0.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]]
            )
            np.testing.assert_array_equal(signs, [1])
        assert scope._target_for_thread() is None
        assert function is scope._library["phx_mc_orient2d"]
        assert ctypes.cast(function, ctypes.c_void_p).value == pointer
        assert view.work_units == 0
        view.settle()
    evidence = scope.completed_evidence()
    # Predicate work remains on its caller's existing host tariff; observing
    # native allocation spans must not invent a native work increment.
    assert evidence.work_counters.construction_work == 0
    assert evidence.managed_admission_peak_bytes_upper is not None


def test_small_unmapped_entry_cannot_hide_under_larger_covered_host_peak() -> None:
    points = np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)))
    signs = np.zeros((1,), dtype=np.int8)
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.host_workspace() as storage:
            storage.set_bound(65536)
            library = meshcore.load_meshcore()
            assert library["phx_mc_orient2d"](
                1, *(point.ctypes.data for point in points), signs.ctypes.data
            ) == int(MeshcoreStatus.OK)
            np.testing.assert_array_equal(signs, [1])
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_observed_ffi_error_preserves_status_and_releases_actual_target() -> None:
    import phydrax._meshcore as owner

    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        view = scope.ticket_view()
        assert scope._library is not None
        with pytest.raises(ValueError, match="orient2d"):
            with view.native_target():
                status = scope._library["phx_mc_orient2d"](-1, None, None, None, None)
                assert status == int(MeshcoreStatus.INVALID_ARGUMENT)
                owner._raise_for_call(status, "orient2d")
        assert scope._target_for_thread() is None
        view.settle()


def test_expired_scope_view_refuses_before_entering_ffi_body() -> None:
    entered = False
    with pytest.raises(MeshcoreError) as failure:
        with NativeExecutionScope(
            NativeExecutionLimits(100, 100, 100, 65536, 0.0)
        ) as scope:
            scope.observe_managed_admission()
            with scope.ticket_view().native_target():
                entered = True
    assert failure.value.status is MeshcoreStatus.TIMEOUT
    assert not entered


def test_foreign_typed_entry_remains_unobserved_even_with_actual_target() -> None:
    import phydrax._meshcore as owner

    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        assert scope._library is not None
        original = scope._library["phx_mc_orient2d"]
        foreign_library = ctypes.CDLL(str(scope._library.path))
        foreign = foreign_library.phx_mc_orient2d
        foreign.argtypes, foreign.restype = original.argtypes, original.restype
        foreign.errcheck = owner._observe_managed_native_call
        a = np.asarray([[0.0, 0.0]])
        b = np.asarray([[1.0, 0.0]])
        c = np.asarray([[0.0, 1.0]])
        signs = np.zeros((1,), dtype=np.int8)
        with scope.ticket_view().native_target():
            assert foreign(
                1, a.ctypes.data, b.ctypes.data, c.ctypes.data, signs.ctypes.data
            ) == int(MeshcoreStatus.OK)
        np.testing.assert_array_equal(signs, [1])
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_foreign_errcheck_is_preserved_and_forces_unknown_coverage() -> None:
    import phydrax._meshcore as owner

    class ForeignCheckError(RuntimeError):
        pass

    with NativeExecutionScope(_limits()) as scope:
        original = scope._library
        assert original is not None
        foreign_library = ctypes.CDLL(str(original.path))
        function = foreign_library.phx_mc_orient2d
        function.argtypes = original["phx_mc_orient2d"].argtypes
        function.restype = original["phx_mc_orient2d"].restype

        def foreign_check(
            result: object, function: object, arguments: tuple[object, ...],
        ) -> Never:
            raise ForeignCheckError("actual foreign post-call check")

        function.errcheck = foreign_check
        functions = dict(original._functions)
        functions["phx_mc_orient2d"] = function
        observed = owner.MeshcoreLibrary(
            foreign_library, functions, original.path, original.version,
            original.build_hash, original.binary_hash, original.configuration,
        )
        assert observed["phx_mc_orient2d"] is function
        assert function.errcheck is foreign_check
        scope._library = observed
        scope.observe_managed_admission()
        a = np.asarray([[0.0, 0.0]])
        b = np.asarray([[1.0, 0.0]])
        c = np.asarray([[0.0, 1.0]])
        signs = np.zeros((1,), dtype=np.int8)
        with pytest.raises(ForeignCheckError, match="actual foreign"):
            with scope.ticket_view().native_target():
                function(1, a.ctypes.data, b.ctypes.data, c.ctypes.data, signs.ctypes.data)
        np.testing.assert_array_equal(signs, [1])
        assert scope._target_for_thread() is None
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_reentrant_actual_scientific_entry_fails_coverage_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        a = np.asarray([[0.0, 0.0]])
        b = np.asarray([[1.0, 0.0]])
        c = np.asarray([[0.0, 1.0]])
        signs = np.zeros((1,), dtype=np.int8)
        assert scope._library is not None
        function = scope._library["phx_mc_orient2d"]
        original_target = scope._target_for_thread
        nested = False

        def target_with_actual_reentry() -> NativeBatchTicket | None:
            nonlocal nested
            if not nested:
                nested = True
                try:
                    function(1, a.ctypes.data, b.ctypes.data, c.ctypes.data, signs.ctypes.data)
                finally:
                    nested = False
            return original_target()

        with scope.ticket_view().native_target():
            with monkeypatch.context() as patch:
                patch.setattr(scope, "_target_for_thread", target_with_actual_reentry)
                assert function(
                    1, a.ctypes.data, b.ctypes.data, c.ctypes.data, signs.ctypes.data
                ) == int(MeshcoreStatus.OK)
        np.testing.assert_array_equal(signs, [1])
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def _predicate_case(
    name: str, symbolic: bool, count: int,
) -> tuple[Callable[..., np.ndarray], tuple[np.ndarray, ...]]:
    width = 2 if name in ("orient2d", "incircle") else 3
    corners = np.vstack((np.zeros((1, width)), np.eye(width)))
    if name in ("incircle", "insphere"):
        corners = np.vstack((corners, np.full((1, width), 0.25)))
    rows = np.random.default_rng(17).integers(
        -2, 3, size=(count, len(corners), width)
    ).astype(np.float64)
    if count:
        rows[0] = corners
    points = tuple(rows[:, index] for index in range(len(corners)))
    if symbolic:
        points += (np.arange(len(corners), dtype=np.int64),)
    prefix = "symbolic" if symbolic else "exact"
    return getattr(meshcore, f"{prefix}_{name}"), points


@pytest.mark.parametrize("name", ("orient2d", "orient3d", "incircle", "insphere"))
@pytest.mark.parametrize("symbolic", (False, True))
@pytest.mark.parametrize("count", (0, 1, 257))
def test_predicate_dispatch_observes_real_scratch_without_work_or_query_tariff(
    name: str, symbolic: bool, count: int,
) -> None:
    predicate, arguments = _predicate_case(name, symbolic, count)
    limits = NativeExecutionLimits(0, 0, 0, 64 * 1024**2, 120.0)
    library = meshcore.load_meshcore()
    symbol = f"phx_mc_{name}" + ("_sos" if symbolic else "")
    function = library[symbol]
    pointer = ctypes.cast(function, ctypes.c_void_p).value
    errcheck = function.errcheck
    evidence: list[NativeExecutionEvidence] = []
    outputs: list[np.ndarray] = []
    for observe, workers in ((False, 1), (True, 1), (True, 2)):
        with NativeParallelExecution(workers, 120.0), NativeExecutionScope(limits) as scope:
            if observe:
                scope.observe_managed_admission()
            outputs.append(predicate(*arguments))
            assert scope._target_for_thread() is None
            assert not scope._ticket_views
            assert not scope._tickets
        evidence.append(scope.completed_evidence())
    assert outputs[0].dtype == np.int8 and outputs[0].shape == (count,)
    if count:
        assert outputs[0][0] == 1
    for result in outputs[1:]:
        np.testing.assert_array_equal(result, outputs[0])
    assert evidence[0].managed_admission_peak_bytes_upper is None
    for observed in evidence[1:]:
        assert observed.managed_admission_peak_bytes_upper is not None
        assert observed.work_counters.construction_work == 0
        assert observed.work_counters.source_queries == 0
        assert observed.work_counters.peak_cavity_cells == 0
        assert observed.native_primitive_queries == evidence[0].native_primitive_queries
        if count:
            assert observed.managed_admission_peak_bytes_upper > 0
    assert library[symbol] is function
    assert ctypes.cast(function, ctypes.c_void_p).value == pointer
    assert function.errcheck is errcheck


@pytest.mark.parametrize("name", ("orient3d", "incircle", "insphere"))
@pytest.mark.parametrize("symbolic", (False, True))
def test_predicate_managed_scratch_exact_boundary_preserves_native_refusal(
    name: str, symbolic: bool,
) -> None:
    predicate, arguments = _predicate_case(name, symbolic, 1)

    def execute(capacity: int) -> NativeExecutionEvidence:
        limits = NativeExecutionLimits(0, 0, 0, capacity, 120.0)
        with NativeParallelExecution(1, 120.0), NativeExecutionScope(limits) as scope:
            scope.observe_managed_admission()
            with scope.host_workspace() as retained:
                retained.set_bound(2048)
                np.testing.assert_array_equal(predicate(*arguments), [1])
        return scope.completed_evidence()

    original = execute(64 * 1024**2)
    boundary = original.managed_admission_peak_bytes_upper
    assert boundary is not None and boundary > 2048
    exact = execute(boundary)
    assert exact.managed_admission_peak_bytes_upper == boundary
    assert exact.work_counters.construction_work == 0
    assert exact.work_counters.source_queries == 0
    assert exact.native_primitive_queries == original.native_primitive_queries
    with pytest.raises(MeshcoreError) as refused:
        execute(boundary - 1)
    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert name in str(refused.value)
    requested = refused.value.batch_requested_upper
    snapshot = refused.value.batch_refusal_evidence
    assert requested is not None and snapshot is not None
    assert requested[0:3] == (0, 0, 0)
    assert requested[3] > int(snapshot[3])


@pytest.mark.parametrize("nested_scope", (False, True))
def test_predicate_reuses_mapped_target_without_second_reservation(
    nested_scope: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicate, arguments = _predicate_case("orient3d", False, 1)
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(0, bytes_upper=65536) as phase, phase.native_target():
            with (
                NativeExecutionScope(_limits()) if nested_scope else nullcontext(scope)
            ) as active:
                def forbidden_batch(*args: object, **kwargs: object) -> None:
                    raise AssertionError("A mapped predicate must not open a second target.")

                with monkeypatch.context() as patch:
                    patch.setattr(active, "batch", forbidden_batch)
                    np.testing.assert_array_equal(predicate(*arguments), [1])
                    if nested_scope:
                        assert active._borrowed_target_for_thread() is phase
                assert scope._target_for_thread() is phase
                assert not active._ticket_views
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is not None


@pytest.mark.parametrize(
    ("coordinates", "ids", "message"),
    (
        ([[0.0]], [0, 1], "shape"),
        ([[np.nan, 2.0**121]], [0, 1], "ids must have shape"),
        ([[np.nan, 2.0**121]], [0, 1, 2], "must be finite"),
        ([[0.0, 2.0**121]], [0, 1, 2], "RANGE_ERROR"),
        ([[0.0, 0.0]], [0, 0, 2], "INVALID_INPUT"),
    ),
)
def test_observed_predicate_preserves_validation_order_and_target_cleanup(
    coordinates: object, ids: object, message: str,
) -> None:
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with pytest.raises(ValueError, match=message):
            meshcore.symbolic_orient2d(
                coordinates, [[1.0, 0.0]], [[0.0, 1.0]], ids
            )
        assert scope._target_for_thread() is None
        assert not scope._ticket_views
        assert not scope._tickets
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is not None


@pytest.mark.parametrize("settled", (False, True))
def test_predicate_rejects_stale_borrowed_target_without_resetting_coverage(
    settled: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicate, arguments = _predicate_case("orient3d", False, 1)
    with NativeExecutionScope(_limits()) as scope:
        scope.observe_managed_admission()
        with scope.batch(0, bytes_upper=65536) as stale:
            with stale.native_target():
                pass
            if settled:
                stale.settle()
            with scope.batch(0, bytes_upper=65536) as phase, phase.native_target():
                with NativeExecutionScope(_limits()) as child:
                    with monkeypatch.context() as patch:
                        patch.setattr(child, "_borrowed_ticket", stale)
                        assert child._borrowed_target_for_thread() is None
                        signs = np.zeros((1,), dtype=np.int8)
                        status = meshcore.load_meshcore()["phx_mc_orient3d"](
                            1, *(point.ctypes.data for point in arguments), signs.ctypes.data
                        )
                        assert status == int(MeshcoreStatus.OK)
                        assert not scope._managed_native_events_complete
                        with pytest.raises(RuntimeError, match="live ancestor target"):
                            predicate(*arguments)
                        assert child._target_for_thread() is None
                        assert scope._target_for_thread() is phase
            scope.observe_managed_admission()
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None


def test_predicate_rejects_live_foreign_borrowed_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ready: Queue[NativeBatchTicket] = Queue()
    release = threading.Event()

    def foreign_owner() -> None:
        with NativeExecutionScope(_limits()) as foreign:
            with foreign.batch(0, bytes_upper=65536) as ticket, ticket.native_target():
                ready.put(ticket)
                assert release.wait(30)

    predicate, arguments = _predicate_case("orient3d", False, 1)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(foreign_owner)
        try:
            foreign = ready.get(timeout=30)
            with NativeExecutionScope(_limits()) as scope:
                scope.observe_managed_admission()
                with scope.batch(0, bytes_upper=65536) as phase, phase.native_target():
                    with NativeExecutionScope(_limits()) as child:
                        with monkeypatch.context() as patch:
                            patch.setattr(child, "_borrowed_ticket", foreign)
                            assert child._borrowed_target_for_thread() is None
                            signs = np.zeros((1,), dtype=np.int8)
                            status = meshcore.load_meshcore()["phx_mc_orient3d"](
                                1, *(point.ctypes.data for point in arguments), signs.ctypes.data
                            )
                            assert status == int(MeshcoreStatus.OK)
                            assert not scope._managed_native_events_complete
                            with pytest.raises(RuntimeError, match="live ancestor target"):
                                predicate(*arguments)
                            assert scope._target_for_thread() is phase
                scope.observe_managed_admission()
            assert scope.completed_evidence().managed_admission_peak_bytes_upper is None
        finally:
            release.set()
        future.result(timeout=30)


def test_borrowed_predicate_refusal_preserves_original_target_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicate, arguments = _predicate_case("insphere", False, 1)
    with pytest.raises(MeshcoreError) as refused:
        with NativeExecutionScope(_limits()) as scope:
            scope.observe_managed_admission()
            with scope.batch(0, bytes_upper=1024) as phase, phase.native_target():
                with NativeExecutionScope(_limits()) as child:
                    def forbidden_batch(*args: object, **kwargs: object) -> None:
                        raise AssertionError("A borrowed refusal cannot renew its target.")

                    with monkeypatch.context() as patch:
                        patch.setattr(child, "batch", forbidden_batch)
                        predicate(*arguments)
    assert refused.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert "insphere" in str(refused.value)
    assert refused.value.batch_admitted_upper == (0, 0, 0, 1024)
    requested = refused.value.batch_requested_upper
    assert requested is not None and requested[3] > 1024
    assert scope.completed_evidence().status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is not None


def test_unobserved_predicate_does_not_open_a_native_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predicate, arguments = _predicate_case("orient3d", False, 1)
    with NativeExecutionScope(_limits()) as scope:
        def forbidden_batch(*args: object, **kwargs: object) -> None:
            raise AssertionError("Observation-disabled predicates must not open a target.")

        with monkeypatch.context() as patch:
            patch.setattr(scope, "batch", forbidden_batch)
            np.testing.assert_array_equal(predicate(*arguments), [1])
    assert scope.completed_evidence().managed_admission_peak_bytes_upper is None
