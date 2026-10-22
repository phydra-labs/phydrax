import gc
import math
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, Never

import numpy as np
import pytest

import phydrax._meshcore as native
import phydrax.geometry._triangulation as owner
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.discretization._periodic_cell import PeriodicCell
from phydrax.geometry._periodic_power_preparation import PeriodicPowerPreparation


_MEMORY = 80_000_000
_WORK = 1 << 26


def _limits() -> native.NativeExecutionLimits:
    return native.NativeExecutionLimits(_WORK, _WORK, _WORK, _MEMORY, math.inf)


def _inputs(count: int = 1) -> tuple[np.ndarray, ...]:
    sites = np.asarray(((0.25, 0.25, 0.25), (0.6, 0.1, 0.1)))[:count].copy()
    domain = np.asarray(((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
    return (
        sites, np.asarray((0.03, -0.02))[:count].copy(), domain,
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
        np.zeros(1, dtype=np.int32),
        np.asarray(((0, 1, 2, 3),), dtype=np.int32),
    )


class _ReachedSourceAllocation(Exception):
    pass


@pytest.mark.parametrize("shortfall", (0, 1))
def test_source_r_and_r_minus_one_precede_host_conversion(
    monkeypatch: pytest.MonkeyPatch, shortfall: int,
) -> None:
    inputs = _inputs()
    required = owner._restricted_source_bound(*inputs, None)
    attempted = []

    def stop_before_conversion(*args: object, **kwargs: object) -> Never:
        attempted.append(True)
        raise _ReachedSourceAllocation

    monkeypatch.setattr(owner, "host_point_array", stop_before_conversion)
    expected = _ReachedSourceAllocation if shortfall == 0 else native.MeshcoreError
    with pytest.raises(expected):
        with native.NativeExecutionScope(_limits()) as scope:
            storage = scope.host_workspace().open()
            blocker = scope.host_workspace().open()
            try:
                # Admit both real token headers first, leaving exactly R or R-1
                # payload bytes in the actual shared native pool.
                storage.reserve_empty()
                blocker.reserve_empty()
                blocker.set_bound(
                    scope.remaining().remaining_managed_bytes - required + shortfall
                )
                owner.RestrictedPowerDiagram(*inputs, scope=scope, host_storage=storage)
            finally:
                blocker.close()
                storage.close()
    assert bool(attempted) is (shortfall == 0)


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("shortfall", (0, 1))
def test_export_r_and_r_minus_one_include_live_native_and_host_storage(
    monkeypatch: pytest.MonkeyPatch, periodic: bool, shortfall: int,
) -> None:
    inputs = _inputs()
    preparation = None
    if periodic:
        preparation = PeriodicPowerPreparation(
            inputs[0], inputs[1], inputs[2], PeriodicCell(np.eye(3)),
            maximum_images=10_000,
        )
    original_copy = native._copy_power_cells
    original_empty = np.empty
    allocated = []
    observed = []
    copying = False

    expectation = pytest.raises(native.MeshcoreError) if shortfall else nullcontext()
    with expectation as caught, native.NativeExecutionScope(_limits()) as scope:
        storage = scope.host_workspace().open()
        blocker = scope.host_workspace().open()
        storage.reserve_empty()
        blocker.reserve_empty()
        token_bytes = _MEMORY - scope.remaining().remaining_managed_bytes

        def record_empty(shape: Any, *args: Any, **kwargs: Any) -> np.ndarray:
            result = original_empty(shape, *args, **kwargs)
            if copying:
                allocated.append(result.nbytes)
            return result

        def copy_with_exact_headroom(
            *args: Any, admit_host_output: Callable[[int], None], **kwargs: Any,
        ) -> native.RestrictedPowerCells:
            nonlocal copying

            def admit(required: int) -> None:
                remaining = scope.remaining().remaining_managed_bytes
                native_live = _MEMORY - remaining - token_bytes - storage.bound
                assert storage.bound > 0
                assert native_live > 0
                blocker.set_bound(remaining - required + shortfall)
                observed.append((required, native_live, storage.bound))
                admit_host_output(required)

            copying = True
            try:
                return original_copy(*args, admit_host_output=admit, **kwargs)
            finally:
                copying = False
                blocker.close()

        monkeypatch.setattr(native.np, "empty", record_empty)
        monkeypatch.setattr(native, "_copy_power_cells", copy_with_exact_headroom)
        try:
            diagram = owner.RestrictedPowerDiagram(
                *inputs, scope=scope, host_storage=storage,
                periodic_preparation=preparation,
            )
            assert len(diagram.construction.cell_sites) > 0
        finally:
            blocker.close()
            # This test's local arrays do not escape the request.
            storage.close()
    assert len(observed) == 1
    if shortfall:
        assert caught is not None
        assert caught.value.status == native.MeshcoreStatus.CAPACITY_EXCEEDED
        assert allocated == []
    else:
        assert len(allocated) == 17
        assert observed[0][0] == sum(allocated) + 20 * 256 + 23 * (128 + 4 * 8)


def test_direct_restricted_output_is_bitwise_deterministic_with_shared_budget() -> None:
    inputs = _inputs(2)
    direct = owner.RestrictedPowerDiagram(*inputs)
    with native.NativeExecutionScope(_limits()) as scope:
        ticket = scope.ticket_view()
        budget = CoordinateEnclosureBudget(
            ticket, maximum_work_units=_WORK, maximum_memory_bytes=_MEMORY,
        )
        with ticket, budget.activate():
            bounded = owner.RestrictedPowerDiagram(*inputs, scope=scope, budget=budget)
            assert direct.guard_count == bounded.guard_count == 4
            assert direct.meshcore_identity == bounded.meshcore_identity
            assert direct.adjacency_work_units == bounded.adjacency_work_units
            for name, value in vars(direct.construction).items():
                other = getattr(bounded.construction, name)
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, other)
                else:
                    assert value == other
            for name in (
                "points", "weights", "neighbor_offsets", "neighbors",
                "cell_original_sites", "piece_original_sites",
                "cell_image_exponents", "piece_image_exponents",
            ):
                np.testing.assert_array_equal(getattr(direct, name), getattr(bounded, name))


@pytest.mark.parametrize("source_alias", (False, True))
def test_export_lease_survives_escaped_array_and_releases_last_root(source_alias: bool) -> None:
    with native.NativeExecutionScope(_limits()) as scope:
        storage = scope.host_workspace().open()
        diagram = owner.RestrictedPowerDiagram(*_inputs(), scope=scope, host_storage=storage)
        escaped = diagram.points if source_alias else diagram.construction.vertices
        retained = storage.bound
        assert retained > 0
        del diagram
        gc.collect()
        assert 0 < storage.bound < retained
        del escaped
        gc.collect()
        assert storage.bound == 0


def test_reused_periodic_preparation_has_one_same_request_payload_lease() -> None:
    inputs = _inputs()
    preparation = PeriodicPowerPreparation(
        inputs[0], inputs[1], inputs[2], PeriodicCell(np.eye(3)), maximum_images=10_000,
    )
    with native.NativeExecutionScope(_limits()) as scope:
        first = owner.RestrictedPowerDiagram(
            *inputs, scope=scope, periodic_preparation=preparation,
        )
        lease = owner._restricted_payload_leases[scope][id(preparation.points)]
        arrays = tuple(
            value for value in vars(preparation).values() if isinstance(value, np.ndarray)
        )
        assert owner._restricted_unowned_array_storage(arrays, scope) == 0
        second = owner.RestrictedPowerDiagram(
            *inputs, scope=scope, periodic_preparation=preparation,
        )
        assert owner._restricted_payload_leases[scope][id(preparation.points)] is lease
        for name, value in vars(first.construction).items():
            other = getattr(second.construction, name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, other)
            else:
                assert value == other


def test_payload_retirement_defers_under_foreign_reservation_target() -> None:
    with native.NativeExecutionScope(_limits()) as scope:
        storage = scope.host_workspace().open()
        diagram = owner.RestrictedPowerDiagram(*_inputs(), scope=scope, host_storage=storage)
        escaped = diagram.points
        retained = storage.bound
        with scope.batch(0) as batch, batch.native_target():
            del diagram
            gc.collect()
            assert storage.bound == retained
        del escaped
        gc.collect()
        assert storage.bound == 0


def test_host_only_batch_keeps_root_payload_retirement_available() -> None:
    with native.NativeExecutionScope(_limits()) as scope:
        storage = scope.host_workspace().open()
        diagram = owner.RestrictedPowerDiagram(*_inputs(), scope=scope, host_storage=storage)
        escaped = diagram.points
        retained = storage.bound
        # Host reservation entry alone does not install a native allocation
        # target. The existing untargeted storage token can still shrink.
        with scope.batch(0):
            del diagram
            gc.collect()
            assert 0 < storage.bound < retained
        del escaped
        gc.collect()
        assert storage.bound == 0


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize(
    ("field", "axis"),
    (
        ("max_edges", "edges"),
        ("max_faces", "faces"),
        ("max_face_entries", "face_corners"),
        ("max_cells", "cells"),
        ("max_site_witnesses", "vertex_site_witnesses"),
        ("max_sided_loop_occurrences", "sided_loop_occurrences"),
        ("max_sided_corner_occurrences", "sided_corner_occurrences"),
        ("max_connectivity_entries", "publication_connectivity_entries_upper"),
        ("max_data_bytes", "publication_numeric_data_bytes_upper"),
    ),
)
def test_output_limits_refuse_before_host_export_and_preserve_exact_evidence(
    monkeypatch: pytest.MonkeyPatch, periodic: bool, field: str, axis: str,
) -> None:
    from phydrax.meshing.providers._native_polyhedral import construction_failure

    inputs = _inputs()
    preparation = None
    if periodic:
        preparation = PeriodicPowerPreparation(
            inputs[0], inputs[1], inputs[2], PeriodicCell(np.eye(3)),
            maximum_images=10_000,
        )
    reference = owner.RestrictedPowerDiagram(*inputs, periodic_preparation=preparation)
    requirement = getattr(reference.construction.preflight, axis)
    limits = native.RestrictedPowerOutputLimits(**{field: requirement})
    accepted = owner.RestrictedPowerDiagram(
        *inputs, output_limits=limits, periodic_preparation=preparation
    )
    assert accepted.construction.preflight == reference.construction.preflight
    for name, value in vars(reference.construction).items():
        other = getattr(accepted.construction, name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, other)
        else:
            assert value == other

    exported = []
    original_copy = native._copy_power_cells

    def record_copy(*args: Any, **kwargs: Any) -> native.RestrictedPowerCells:
        exported.append(True)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(native, "_copy_power_cells", record_copy)
    limits = native.RestrictedPowerOutputLimits(**{field: requirement - 1})
    with native.NativeExecutionScope(_limits()) as scope:
        with pytest.raises(
            owner.PeriodicPowerSourceRefusal if periodic else native.RestrictedPowerFailure
        ) as caught:
            owner.RestrictedPowerDiagram(
                *inputs, scope=scope, output_limits=limits,
                periodic_preparation=preparation,
            )
        refusal = caught.value
        if periodic:
            refusal = refusal.__cause__
        assert isinstance(refusal, native.RestrictedPowerFailure)
        assert refusal.preflight == reference.construction.preflight
        assert refusal.capacity == native.RestrictedPowerCapacity(
            axis, requirement, requirement - 1
        )
        assert exported == []
        failure = construction_failure(refusal, "volume_fill")
        assert dict(failure.achieved)[f"required_{axis}"] == requirement
        assert dict(failure.achieved)[f"remaining_{axis}"] == requirement - 1
        assert {
            key: value for key, value in failure.achieved
            if key.startswith("native_preflight:")
        } == {
            f"native_preflight:{name}": float(value)
            for name, value in refusal.preflight.items()
        }
        assert f"required={requirement}, remaining={requirement - 1}" in str(failure)
        # No native prepared topology or owner workspace survives a refusal.
        assert scope.remaining().remaining_managed_bytes == _MEMORY


def test_native_publication_header_matches_independent_owner() -> None:
    from phydrax.meshing._polyhedral_generation import (
        _polyhedral_numeric_publication_bounds,
        _validate_polyhedral_preflight,
    )

    diagram = owner.RestrictedPowerDiagram(*_inputs(2))
    data = diagram.construction
    header = data.preflight
    edges = {
        tuple(sorted((int(a), int(b))))
        for face in range(len(data.face_cells))
        for a, b in zip(
            data.face_vertices[data.face_offsets[face]:data.face_offsets[face + 1]],
            np.roll(
                data.face_vertices[data.face_offsets[face]:data.face_offsets[face + 1]],
                -1,
            ),
            strict=True,
        )
    }
    copies = 1 + (data.face_cells[:, 1] >= 0)
    assert header.edges == len(edges)
    assert header.sided_loop_occurrences == int(np.sum(copies))
    assert header.sided_corner_occurrences == int(np.dot(copies, np.diff(data.face_offsets)))
    assert (
        header.publication_connectivity_entries_upper,
        header.publication_numeric_data_bytes_upper,
    ) == _polyhedral_numeric_publication_bounds(
        header.vertices, header.cells, header.sided_loop_occurrences,
        header.sided_corner_occurrences,
    )
    _validate_polyhedral_preflight(
        data, header.sided_loop_occurrences, header.sided_corner_occurrences
    )


@pytest.mark.parametrize(
    "field",
    (
        "vertices", "faces", "face_corners", "cells", "pieces",
        "vertex_site_witnesses", "sided_loop_occurrences",
        "sided_corner_occurrences", "publication_connectivity_entries_upper",
        "native_output_payload_bytes",
        "publication_numeric_data_bytes_upper",
    ),
)
def test_publication_rejects_native_owner_count_or_formula_drift(field: str) -> None:
    from phydrax.meshing._failures import MeshingFailure, MeshingFailureCategory
    from phydrax.meshing._polyhedral_generation import _validate_polyhedral_preflight

    data = owner.RestrictedPowerDiagram(*_inputs(2)).construction
    header = data.preflight
    expected = getattr(header, field)
    inconsistent = replace(
        data, preflight=replace(header, **{field: expected + 1})
    )
    with pytest.raises(MeshingFailure, match=f"polyhedral.preflight.{field}") as caught:
        _validate_polyhedral_preflight(
            inconsistent, header.sided_loop_occurrences, header.sided_corner_occurrences
        )
    assert caught.value.category is MeshingFailureCategory.AUDIT_FAILED
    assert dict(caught.value.achieved) == {
        f"native_{field}": expected + 1, f"owner_{field}": expected,
    }


@pytest.mark.parametrize("sites", (1, 2))
def test_restricted_construction_has_zero_source_query_requirement(sites: int) -> None:
    limits = native.NativeExecutionLimits(_WORK, 0, _WORK, _MEMORY, math.inf)
    with native.NativeExecutionScope(limits) as scope:
        diagram = owner.RestrictedPowerDiagram(*_inputs(sites), scope=scope)
        assert diagram.construction.preflight.cells > 0
        assert scope.remaining().remaining_geometry_queries == 0
    # Exact predicates and guarded regular triangulation are native work,
    # never evaluations of an authored geometry source.
    assert scope.completed_evidence().externally_charged_geometry_queries == 0


def test_managed_request_peak_r_and_r_minus_one() -> None:
    inputs = _inputs(2)
    with native.NativeExecutionScope(_limits()) as measured:
        measured.observe_managed_admission()
        reference = owner.RestrictedPowerDiagram(*inputs, scope=measured)
    evidence = measured.completed_evidence()
    required = evidence.managed_admission_peak_bytes_upper
    assert required is not None and required > 0
    for shortfall in (0, 1):
        limit = required - shortfall
        limits = native.NativeExecutionLimits(_WORK, _WORK, _WORK, limit, math.inf)
        expectation = pytest.raises(native.MeshcoreError) if shortfall else nullcontext()
        with expectation, native.NativeExecutionScope(limits) as scope:
            scope.observe_managed_admission()
            result = owner.RestrictedPowerDiagram(*inputs, scope=scope)
            assert result.construction.preflight == reference.construction.preflight
        observed = scope.completed_evidence()
        assert observed.managed_admission_peak_bytes_upper is not None
        assert observed.managed_admission_peak_bytes_upper <= limit
        if shortfall:
            assert observed.status is native.MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("periodic", (False, True))
@pytest.mark.parametrize("authored", (2**63 - 1, 2**63))
def test_output_allowances_saturate_at_signed_abi_maximum(
    periodic: bool, authored: int,
) -> None:
    fields = vars(native.RestrictedPowerOutputLimits())
    limits = native.RestrictedPowerOutputLimits(
        **dict.fromkeys(fields, authored)
    )
    assert limits.native_fields() == dict.fromkeys(fields, 2**63 - 1)
    assert dict(vars(limits)) == dict.fromkeys(fields, authored)
    inputs = _inputs()
    preparation = None
    if periodic:
        preparation = PeriodicPowerPreparation(
            inputs[0], inputs[1], inputs[2], PeriodicCell(np.eye(3)),
            maximum_images=10_000,
        )
    reference = owner.RestrictedPowerDiagram(*inputs, periodic_preparation=preparation)
    bounded = owner.RestrictedPowerDiagram(
        *inputs, periodic_preparation=preparation, output_limits=limits
    )
    assert bounded.construction.preflight == reference.construction.preflight
    for name, value in vars(reference.construction).items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, getattr(bounded.construction, name))


@pytest.mark.parametrize(
    ("schedule_vertices", "authored_vertices"), ((64, 3), (3, 64))
)
def test_generation_threads_authored_vertex_cap_before_native_export(
    monkeypatch: pytest.MonkeyPatch, schedule_vertices: int, authored_vertices: int,
) -> None:
    from phydrax.meshing import _polyhedral_generation as generation
    from phydrax.meshing._contracts import MeshingLimits

    sites, weights, points, tets, regions, facets = _inputs()
    # Only the carrier fields read before native construction are needed;
    # the real native clipper must refuse before any publication dependency.
    carrier: Any = SimpleNamespace(
        recovery=SimpleNamespace(
            points=points, tetrahedra=tets, tetrahedron_regions=regions,
        ),
        domain=None,
        facets=facets,
    )
    state: Any = SimpleNamespace(
        points=sites, weights=weights, split=np.zeros(len(sites), dtype=np.int8)
    )
    unused_publication_input: Any = None
    selected_limits = []
    exported = []
    original_construct = native.restricted_power_cells
    original_copy = native._copy_power_cells

    def construct(*args: Any, **kwargs: Any) -> native.RestrictedPowerCells:
        selected_limits.append(kwargs["max_vertices"])
        return original_construct(*args, **kwargs)

    def export(*args: Any, **kwargs: Any) -> native.RestrictedPowerCells:
        exported.append(True)
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(native, "restricted_power_cells", construct)
    monkeypatch.setattr(native, "_copy_power_cells", export)
    with native.NativeExecutionScope(_limits()) as scope:
        with pytest.raises(native.RestrictedPowerFailure) as caught:
            generation._construct_polyhedral_publication(
                carrier, state,
                generation.NativePolyhedralSchedule(maximum_vertices=schedule_vertices),
                0, unused_publication_input, (), None, None,
                scope=scope, budget=unused_publication_input,
                limits=MeshingLimits(maximum_vertices=authored_vertices),
            )
        assert caught.value.status is native.MeshcoreStatus.CAPACITY_EXCEEDED
        assert int(caught.value.failure[0]) == 5  # Native vertex allowance.
    assert selected_limits == [min(schedule_vertices, authored_vertices)]
    assert exported == []


@pytest.mark.parametrize("site_count", (1, 2))
def test_restricted_owner_work_r_and_r_minus_one_rolls_back_leases(site_count: int) -> None:
    inputs = _inputs(site_count)
    original = tuple(value.copy() for value in inputs)
    with native.NativeExecutionScope(_limits()) as measured:
        reference = owner.RestrictedPowerDiagram(*inputs, scope=measured)
        required = _WORK - measured.remaining().remaining_work_units
    assert measured.completed_evidence().externally_charged_work > 0
    assert measured.completed_evidence().externally_charged_geometry_queries == 0
    for shortfall in (0, 1):
        limits = native.NativeExecutionLimits(
            required - shortfall, 0, _WORK, _MEMORY, math.inf
        )
        expectation = pytest.raises(native.MeshcoreError) if shortfall else nullcontext()
        with expectation as caught, native.NativeExecutionScope(limits) as scope:
            result = owner.RestrictedPowerDiagram(*inputs, scope=scope)
            assert scope.remaining().remaining_work_units == 0
            assert result.construction.preflight == reference.construction.preflight
            for name, value in vars(reference.construction).items():
                if isinstance(value, np.ndarray):
                    np.testing.assert_array_equal(value, getattr(result.construction, name))
        if shortfall:
            assert caught is not None
            assert caught.value.status is native.MeshcoreStatus.CAPACITY_EXCEEDED
            assert scope not in owner._restricted_payload_leases
            assert scope.completed_evidence().host_storage_live_bytes_upper == 0
        assert scope.completed_evidence().externally_charged_geometry_queries == 0
    for value, unchanged in zip(inputs, original, strict=True):
        np.testing.assert_array_equal(value, unchanged)


def test_restricted_owner_profile_charges_each_fresh_identity_before_inspection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = owner.retained_host_storage_profile
    visits: list[int] = []
    with native.NativeExecutionScope(_limits()) as scope:
        def profile(
            value: object, existing: Any, native_buffer: Any, *,
            admit_owner: Callable[[object], None] | None = None,
        ) -> Any:
            assert admit_owner is not None

            def admitted(item: object) -> None:
                before = scope.remaining().remaining_work_units
                admit_owner(item)
                assert before - scope.remaining().remaining_work_units == 1
                visits.append(id(item))

            return original(value, existing, native_buffer, admit_owner=admitted)

        monkeypatch.setattr(owner, "retained_host_storage_profile", profile)
        diagram = owner.RestrictedPowerDiagram(*_inputs(), scope=scope)
        assert diagram.construction.preflight.cells > 0
        assert visits and len(visits) == len(set(visits))


def test_restricted_final_resize_deadline_detaches_partial_attachments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    resize = native.NativeHostStorageWorkspace.set_bound
    attached: list[Any] = []
    finalize = owner.weakref.finalize

    def attach(value: object, callback: Any, *args: Any, **kwargs: Any) -> Any:
        result = finalize(value, callback, *args, **kwargs)
        if isinstance(
            getattr(callback, "__self__", None),
            (owner._RestrictedPayloadLease, owner._RestrictedRecordLifetime),
        ):
            attached.append(result)
        return result

    with native.NativeExecutionScope(_limits()) as scope:
        def deadline(storage: native.NativeHostStorageWorkspace, bound: int) -> None:
            if attached and bound < storage.bound:
                raise native.MeshcoreError(native.MeshcoreStatus.TIMEOUT, "retention deadline")
            resize(storage, bound)

        monkeypatch.setattr(owner.weakref, "finalize", attach)
        monkeypatch.setattr(native.NativeHostStorageWorkspace, "set_bound", deadline)
        with pytest.raises(native.MeshcoreError) as caught:
            owner.RestrictedPowerDiagram(*_inputs(), scope=scope)
        assert caught.value.status is native.MeshcoreStatus.TIMEOUT
        assert attached and not any(value.alive for value in attached)
        assert scope not in owner._restricted_payload_leases
        assert scope.remaining().remaining_managed_bytes == _MEMORY
