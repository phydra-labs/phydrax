# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Each complete native request owns one allowance; completed requests are history."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import NoReturn

import numpy as np
import pytest

import phydrax as phx
from phydrax._fingerprint import array_tree_fingerprint
from phydrax._meshcore import (
    current_native_execution_scope,
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax._model._structure import model_array_bytes
from phydrax.meshing import (
    BisectionCompatibility,
    CellMeshingResult,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    prepare_mesh_adaptation,
)
from phydrax.meshing._measurements import NativeExecutionImports, NativeExecutionRecord
from phydrax.meshing._native_preparation import (
    _completed_native_sources,
    native_preparation_ledger,
    prepare_native_completed_source_issue,
    stage_native_source_publication,
)
from phydrax.meshing._volume_generation import native_volume_execution_budget
from phydrax.meshing.providers._native_publication import bind_native_execution_result
from phydrax.meshing.qualification import (
    execute_native_consumer,
    execute_native_request,
)
from tests._support.cell_meshes import triangle_grid


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore required"
)


def _ended(
    work: int, queries: int = 0, *, owner: str = "completed-request"
) -> NativeExecutionRecord:
    with NativeExecutionScope(
        NativeExecutionLimits(work, queries, 10, 4 * 1024**2, 120.0)
    ) as scope:
        with scope.batch(work, queries_upper=queries) as ticket:
            ticket.consume(work, geometry_queries=queries)
    return NativeExecutionRecord(scope.completed_evidence(), owner_id=owner)


def _limits(work: int, queries: int = 1) -> MeshingLimits:
    return MeshingLimits(
        maximum_work_units=work,
        maximum_geometry_queries=queries,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )


def _failure(operation: str, message: str, details: dict[str, object]) -> Exception:
    return RuntimeError(f"{operation}: {message} {details!r}")


@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_publication_receipt_storage_is_admitted_before_device_transfer(
    slack: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import phydrax.meshing._measurements as measurements

    source = phx.meshing.certify_cell_mesh(
        triangle_grid(2, 2), phx.SpatialCoordinateContract.si()
    )
    with NativeExecutionScope(_limits(1).native_limits()) as scope:
        pass
    evidence = scope.completed_evidence()
    baseline = bind_native_execution_result(source, evidence, _limits(1), perf_counter())
    required = model_array_bytes(baseline)
    limits = MeshingLimits(maximum_data_bytes=required + slack)
    if slack == 0:
        bounded = bind_native_execution_result(source, evidence, limits, perf_counter())
        assert model_array_bytes(bounded) == required
        return

    def reject_device_transfer(*args: object, **kwargs: object) -> NoReturn:
        raise AssertionError("Device storage was allocated before retention admission.")

    monkeypatch.setattr(measurements.jax, "device_put", reject_device_transfer)
    with pytest.raises(MeshingFailure) as caught:
        bind_native_execution_result(source, evidence, limits, perf_counter())
    assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED


@pytest.mark.parametrize("predecessor", ("preparation", "consumer"))
def test_explicit_publication_owner_cannot_relabel_paid_history(predecessor: str) -> None:
    source = phx.meshing.certify_cell_mesh(
        triangle_grid(1, 1), phx.SpatialCoordinateContract.si()
    )
    previous = _ended(5, owner=source.result_id)
    carried = source if predecessor == "preparation" else source.with_execution_evidence(previous)
    with NativeExecutionScope(_limits(1).native_limits()) as scope:
        pass
    with pytest.raises(ValueError):
        bind_native_execution_result(
            carried, scope.completed_evidence(), _limits(1), perf_counter(),
            preparation_evidence=previous if predecessor == "preparation" else None,
            owner_id="different-generation-request",
        )


@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_completed_request_history_is_provenance_not_a_charge(slack: int) -> None:
    history = _ended(50, 2)
    own = 9
    # History's 50 units stay historical; fresh authentication and retention
    # are operations of this request and must be admitted as well.
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as measured:
        native_preparation_ledger(measured).record_history(measured, history)
        authentication = (
            1_000_000 - measured.remaining().remaining_work_units
        )
    limits = _limits(own + authentication + slack)

    def run() -> NativeExecutionScope:
        with native_volume_execution_budget(limits) as request:
            ledger = native_preparation_ledger(request)
            ledger.record_history(request, history)
            assert request.remaining().remaining_work_units == own + slack
            # The two historical queries exceed this request's one-query limit.
            assert request.remaining().remaining_geometry_queries == 1
            with request.batch(own) as ticket:
                ticket.consume(own)
        return request

    if slack < 0:
        with pytest.raises(MeshingFailure) as caught:
            run()
        assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        return
    request = run()
    record = NativeExecutionRecord(
        request.completed_evidence(), owner_id="next-request", history_evidence=(history,)
    )
    assert record.history_evidence == (history,)
    assert int(np.asarray(record.total_work_units)) == own + authentication
    assert int(np.asarray(record.total_geometry_queries)) == 0
    assert history.receipt_hex() not in record.imported_preparation_hex()
    interchange = record.to_record()
    assert interchange["phases"][0]["history_evidence"] == [1]
    assert interchange["phases"][1]["receipt_id"] == history.receipt_hex()
    assert interchange["phases"][0]["total_work_units"] == own + authentication


def test_new_request_preserves_history_discovered_during_restoration() -> None:
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget

    source = phx.meshing.certify_cell_mesh(
        triangle_grid(1, 1), phx.SpatialCoordinateContract.si()
    )
    first = _ended(1_000_001, 2, owner="archive-request")
    dependency = _ended(1_000_003, 3, owner="restored-source-request")

    def restore(budget: CoordinateEnclosureBudget) -> CellMeshingResult:
        scope = current_native_execution_scope()
        if scope is None:
            raise RuntimeError("Restoration must retain its current request scope.")
        native_preparation_ledger(scope).record_history(scope, dependency)
        budget.reserve(7)
        return source

    restored = execute_native_request(
        first, _limits(1_000_000), restore,
        target=lambda value: value, failure_factory=_failure,
    )
    receipt = restored.result.execution_evidence
    assert receipt is not None
    assert tuple(item.receipt_hex() for item in receipt.history_evidence) == tuple(
        sorted((first.receipt_hex(), dependency.receipt_hex()))
    )
    assert receipt.preparation_evidence is None
    assert int(np.asarray(receipt.total_work_units)) < 1_000_000
    assert int(np.asarray(receipt.total_geometry_queries)) == 0


def test_history_and_request_charges_remain_disjoint() -> None:
    history = _ended(3)
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.record_history(request, history)
        with pytest.raises(ValueError, match="recorded as history"):
            ledger.import_receipt(request, history)
        with pytest.raises(ValueError, match="recorded as history"):
            ledger.remember_receipt(request, history)
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.import_receipt(request, history)
        with pytest.raises(ValueError, match="already charged to this request"):
            ledger.record_history(request, history)
    with pytest.raises(ValueError, match="both charged and historical"):
        NativeExecutionRecord(
            request.completed_evidence(),
            preparation_evidence=history,
            history_evidence=(history,),
            owner_id=history.owner_id,
        )


def test_history_is_recorded_once_without_replaying_elapsed_time() -> None:
    history = _ended(3)
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.record_history(request, history)
        after_first = request.remaining().remaining_work_units
        ledger.record_history(request, history)
        after_second = request.remaining().remaining_work_units
        graph = ledger._receipt_graphs[id(history)]
        assert after_first - after_second == graph.authentication_work + 69
    evidence = request.completed_evidence()
    assert evidence.prior_elapsed_seconds == 0.0
    assert evidence.work_counters.construction_work == 1_000_000 - after_second


@pytest.mark.parametrize("nested", (False, True), ids=("root", "after-consumer"))
def test_explicit_import_of_remembered_phase_keeps_identity_without_recharge(
    nested: bool,
) -> None:
    with NativeExecutionScope(_limits(1_000_000, 10).native_limits()) as root:
        with NativeExecutionScope(_limits(50, 5).native_limits()) as preparation:
            with preparation.batch(7, queries_upper=2) as ticket:
                ticket.consume(7, geometry_queries=2)
        prior = NativeExecutionRecord(
            preparation.completed_evidence(), owner_id="prior-preparation"
        )
        ledger = native_preparation_ledger(root)
        ledger.remember_receipt(root, prior)
        ledger.authenticate_receipt(root, prior)
        graph = ledger._receipt_graphs[id(prior)]
        assert ledger.imported_receipt_ids(root) == ()

        def import_twice(scope: NativeExecutionScope) -> NativeExecutionImports:
            before = scope.remaining()
            ledger.import_receipt(scope, prior)
            middle = scope.remaining()
            ledger.import_receipt(scope, prior)
            after = scope.remaining()
            first_cost = before.remaining_work_units - middle.remaining_work_units
            second_cost = middle.remaining_work_units - after.remaining_work_units
            assert first_cost == second_cost
            assert first_cost > graph.authentication_work
            assert after.remaining_geometry_queries == before.remaining_geometry_queries
            assert after.remaining_managed_bytes == before.remaining_managed_bytes
            assert after.remaining_seconds <= before.remaining_seconds
            imported = NativeExecutionImports(scope, preparation=prior)
            assert imported.receipt_ids == (prior.receipt_hex(),)
            return imported

        if nested:
            with NativeExecutionScope(_limits(1_000_000, 5).native_limits()) as consumer:
                imported = import_twice(consumer)
            evidence = imported.evidence()
        else:
            imported = import_twice(root)
    if not nested:
        evidence = imported.evidence()
    current = NativeExecutionRecord(
        evidence, owner_id="closed-preparation", preparation_evidence=prior,
        imports=imported,
    )
    current.require_valid()
    assert current.preparation_evidence is prior
    assert prior.receipt_hex() in current.imported_preparation_hex()
    assert root.completed_evidence().prior_elapsed_seconds == 0.0
    assert int(np.asarray(current.source_preparation_work_units)) == 7
    assert int(np.asarray(current.total_work_units)) == (
        int(np.asarray(current.work_counters.construction_work)) + 7
    )
    if not nested:
        assert int(np.asarray(current.total_work_units)) == (
            root.completed_evidence().work_counters.construction_work
        )
    assert root.completed_evidence().work_counters.source_queries == 2


def test_remembered_import_refuses_changed_owner_and_foreign_root() -> None:
    from copy import deepcopy

    prior = _ended(3)
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as first:
        ledger = native_preparation_ledger(first)
        ledger.remember_receipt(first, prior)
        changed = deepcopy(prior)
        object.__setattr__(changed, "owner_id", "foreign-owner")
        with pytest.raises(ValueError):
            ledger.import_receipt(first, changed)
        assert ledger.imported_receipt_ids(first) == ()
    with NativeExecutionScope(_limits(100).native_limits()) as second:
        with pytest.raises(ValueError, match="foreign root"):
            ledger.import_receipt(second, prior)
        assert native_preparation_ledger(second).imported_receipt_ids(second) == ()


@pytest.mark.parametrize("workflow", (0.5, None), ids=("workflow-binds", "request-binds"))
def test_request_clock_is_min_of_own_budget_and_workflow_remaining(
    workflow: float | None,
) -> None:
    observed: list[float] = []
    result = _source_result(4)

    def call(_budget: phx.discretization.CoordinateEnclosureBudget) -> CellMeshingResult:
        scope = current_native_execution_scope()
        assert scope is not None
        observed.append(scope.remaining().remaining_seconds)
        with NativeExecutionScope(_limits(10).native_limits()) as phase:
            with phase.batch(1) as ticket:
                ticket.consume(1)
        return result.with_execution_evidence(
            NativeExecutionRecord(phase.completed_evidence(), owner_id="next")
        )

    execute_native_request(
        result,
        MeshingLimits(maximum_wall_seconds=120.0),
        call,
        target=lambda value: value,
        failure_factory=_failure,
        workflow_remaining_seconds=workflow,
    )
    # History elapsed is never subtracted a second time from the request budget.
    assert observed[0] <= (120.0 if workflow is None else workflow)
    assert observed[0] > (119.0 if workflow is None else 0.0)


@pytest.mark.parametrize("targeted", (False, True), ids=("scope", "active-target"))
def test_oversize_request_import_is_a_typed_resource_refusal(targeted: bool) -> None:
    preparation = _ended(50)
    with pytest.raises(MeshcoreError) as caught:
        with NativeExecutionScope(
            _limits(100 if targeted else 10).native_limits()
        ) as request:
            ledger = native_preparation_ledger(request)
            if targeted:
                with (
                    request.batch(10, bytes_upper=4 * 1024**2) as ticket,
                    ticket.native_target(),
                ):
                    ledger.import_receipt(request, preparation)
            else:
                ledger.import_receipt(request, preparation)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert "child must fit the parent estimate" not in str(caught.value)
    evidence = request.completed_evidence()
    assert evidence.work_counters.work_refusals >= 1
    assert evidence.externally_charged_work == 2


def test_preparation_is_charged_once_per_request_and_again_by_each_request() -> None:
    preparation = _ended(7)
    consumed = []
    for _ in range(2):
        with NativeExecutionScope(_limits(1_000_000).native_limits()) as request:
            ledger = native_preparation_ledger(request)
            ledger.import_receipt(request, preparation)
            ledger.import_receipt(request, preparation)
            imports = NativeExecutionImports(request, preparation=preparation)
            consumed.append(1_000_000 - request.remaining().remaining_work_units)
        assert imports.evidence().work_counters.construction_work == consumed[-1] - 7
    # Both requests independently re-earn metadata; only the seven preparation
    # units are removed from each importing phase's raw counters.
    assert consumed[0] == consumed[1]
    assert consumed[0] > 8


@pytest.mark.parametrize("ownership", ("live", "ended"))
def test_live_and_ended_preparation_share_the_request_allowance(ownership: str) -> None:
    limits = _limits(1_000_000)
    ended = _ended(6, owner="request")
    with NativeExecutionScope(limits.native_limits()) as request:
        ledger = native_preparation_ledger(request)
        if ownership == "live":
            with NativeExecutionScope(limits.native_limits()) as phase:
                with phase.batch(6) as ticket:
                    ticket.consume(6)
            preparation = NativeExecutionRecord(
                phase.completed_evidence(), owner_id="request"
            )
            ledger.remember_receipt(request, preparation)
        else:
            preparation = ended
            ledger.import_receipt(request, preparation)
        with request.batch(4) as ticket:
            ticket.consume(4)
        imports = NativeExecutionImports(request, preparation=preparation)
    record = NativeExecutionRecord(
        imports.evidence(),
        preparation_evidence=preparation,
        owner_id="request",
        imports=imports,
    )
    # Authentication/retention belong to this phase; the represented six units
    # are supplied by the preparation exactly once.
    total = int(request.completed_evidence().work_counters.construction_work)
    assert int(np.asarray(record.work_counters.construction_work)) == total - 6
    assert int(np.asarray(record.total_work_units)) == total


def _source_result(work: int, *, mutable: bool = False) -> CellMeshingResult:
    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    certified = phx.meshing.certify_cell_mesh(mesh, phx.SpatialCoordinateContract.si())
    if mutable:
        object.__setattr__(
            certified.mesh, "coordinates", np.array(certified.mesh.coordinates),
        )
    limits = MeshingLimits(maximum_wall_seconds=120.0)
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(work) as ticket:
            ticket.consume(work)
        stage_native_source_publication(scope, certified)
        issue = prepare_native_completed_source_issue(scope, certified)
    result = bind_native_execution_result(
        certified, scope.completed_evidence(), limits, scope.operation_started,
    )
    issue.complete(result)
    return result


@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_unchanged_consumer_exact_work_includes_paid_source_and_final_scan(
    slack: int,
) -> None:
    source = _source_result(4, mutable=True)
    receipt = source.execution_evidence
    assert receipt is not None

    def consume(budget: phx.discretization.CoordinateEnclosureBudget) -> int:
        budget.reserve(3)
        return 7

    measured = execute_native_consumer(
        source, _limits(1_000_000), consume, failure_factory=_failure,
    )
    recorded = measured.result.execution_evidence
    assert recorded is not None
    required = int(np.asarray(recorded.total_work_units))
    assert required > int(np.asarray(receipt.total_work_units)) + 3
    if slack < 0:
        with pytest.raises(MeshcoreError) as caught:
            execute_native_consumer(
                source, _limits(required - 1), consume, failure_factory=_failure,
            )
        assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
        return
    exact = execute_native_consumer(
        source, _limits(required), consume, failure_factory=_failure,
    )
    assert exact.value == 7
    assert exact.result.execution_evidence is not None
    assert exact.result.execution_evidence.preparation_evidence is receipt
    assert int(np.asarray(exact.result.execution_evidence.total_work_units)) == required


@pytest.mark.parametrize("target", ("implicit", "same", "changed"))
def test_consumer_only_reuses_the_actual_unchanged_target(
    target: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing import _native_preparation as publication
    from phydrax.meshing._native_preparation import _NativeSourceSeal

    source = _source_result(4, mutable=True)
    assert source.execution_evidence is not None
    produced = (
        source.with_execution_evidence(source.execution_evidence)
        if target == "changed" else source
    )
    staged = []
    seal_source = publication._native_source_seal

    def stage(
        value: CellMeshingResult, budget: phx.discretization.CoordinateEnclosureBudget,
    ) -> _NativeSourceSeal:
        staged.append(value)
        return seal_source(value, budget)

    monkeypatch.setattr(publication, "_native_source_seal", stage)
    consumed = execute_native_consumer(
        source, _limits(1_000_000), lambda budget: produced,
        target=None if target == "implicit" else lambda value: value,
        failure_factory=_failure,
    )
    assert staged == ([produced] if target == "changed" else [])
    assert consumed.result.execution_evidence is not None
    assert consumed.result.execution_evidence.preparation_evidence is source.execution_evidence
    assert id(consumed.result) in _completed_native_sources


@pytest.mark.parametrize("before_consumer", (False, True), ids=("during", "before"))
def test_unchanged_consumer_mutation_refuses_only_at_final_completion(
    before_consumer: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax.meshing._native_preparation import NativeCompletedSourceIssue

    source = _source_result(4, mutable=True)
    bank = source.mesh.coordinates
    assert isinstance(bank, np.ndarray)
    completed = []
    complete = NativeCompletedSourceIssue.complete

    def finish(issue: NativeCompletedSourceIssue, bound: CellMeshingResult) -> None:
        assert current_native_execution_scope() is None
        completed.append(bound)
        complete(issue, bound)

    monkeypatch.setattr(NativeCompletedSourceIssue, "complete", finish)
    if before_consumer:
        bank.flat[-1] += 0.125

    def consume(budget: phx.discretization.CoordinateEnclosureBudget) -> None:
        budget.reserve(1)
        if not before_consumer:
            bank.flat[-1] += 0.125

    with pytest.raises(ValueError, match="complete bytes"):
        execute_native_consumer(
            source, _limits(1_000_000), consume, failure_factory=_failure,
        )
    assert len(completed) == 1
    assert id(completed[0]) not in _completed_native_sources


def test_unissued_same_result_consumer_refuses() -> None:
    issued = _source_result(4)
    assert issued.execution_evidence is not None
    clone = issued.with_execution_evidence(issued.execution_evidence)
    with pytest.raises(ValueError, match="actual issued weak owner"):
        execute_native_consumer(
            clone, _limits(1_000_000), lambda budget: None, failure_factory=_failure,
        )
    assert id(clone) not in _completed_native_sources


@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_each_epoch_request_keeps_its_original_allowance(slack: int) -> None:
    own = 5
    limits = MeshingLimits(maximum_wall_seconds=120.0)
    result = _source_result(4)
    roots = [result.execution_evidence]

    def epoch(index: int) -> CellMeshingResult:
        with NativeExecutionScope(limits.native_limits()) as phase:
            with phase.batch(own) as ticket:
                ticket.consume(own)
        return result.with_execution_evidence(
            NativeExecutionRecord(phase.completed_evidence(), owner_id=f"epoch-{index}")
        )

    for index in range(3):
        # Publication authenticates the complete current source and ancestry.
        # Measure that real work rather than pretending the five science units
        # are the complete request or omitting metadata admission from R.
        limits = MeshingLimits(maximum_wall_seconds=120.0)
        measured = execute_native_request(
            result, limits, lambda _budget, index=index: epoch(index),
            target=lambda value: value, failure_factory=_failure,
        )
        assert measured.result.execution_evidence is not None
        required = int(np.asarray(measured.result.execution_evidence.total_work_units))
        limits = _limits(required + slack)
        if slack < 0:
            with pytest.raises(MeshcoreError) as caught:
                execute_native_request(
                    result,
                    limits,
                    lambda _budget, index=index: epoch(index),
                    target=lambda value: value,
                    failure_factory=_failure,
                )
            assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            return
        previous = result.execution_evidence
        assert previous is not None
        requested = execute_native_request(
            result,
            limits,
            lambda _budget, index=index: epoch(index),
            target=lambda value: value,
            failure_factory=_failure,
        )
        result = requested.result
        root = result.execution_evidence
        assert root is not None
        assert root.owner_id == f"epoch-{index}"
        assert root.history_evidence == (previous,)
        assert int(np.asarray(root.total_work_units)) == required
        # History never replays elapsed time; the workflow deadline intersects live.
        assert float(np.asarray(root.prior_elapsed_seconds)) == 0.0
        roots.append(root)
        # Consumers continue the producing request and its remaining allowance.
        with pytest.raises(MeshcoreError):
            execute_native_consumer(
                result,
                limits,
                lambda budget: budget.reserve(1),
                failure_factory=_failure,
            )
    assert len({root.receipt_hex() for root in roots if root is not None}) == 4


@pytest.mark.parametrize("observe", (False, True), ids=("unobserved", "observed"))
def test_native_helper_observation_opts_into_actual_scope_before_admissions(
    observe: bool,
) -> None:
    from phydrax._meshcore import load_meshcore

    source = _source_result(2)
    limits = MeshingLimits(maximum_wall_seconds=120.0)

    def consume(
        budget: phx.discretization.CoordinateEnclosureBudget,
    ) -> CellMeshingResult:
        budget.reserve(1, 1024)
        first = np.asarray(((0.0, 0.0),), dtype=np.float64)
        second = np.asarray(((1.0, 0.0),), dtype=np.float64)
        third = np.asarray(((0.0, 1.0),), dtype=np.float64)
        signs = np.empty((1,), dtype=np.int8)
        status = load_meshcore()["phx_mc_orient2d"](
            1, first.ctypes.data, second.ctypes.data, third.ctypes.data, signs.ctypes.data
        )
        assert status == MeshcoreStatus.OK
        np.testing.assert_array_equal(signs, np.asarray((1,), dtype=np.int8))
        return source

    continued = execute_native_request(
        source,
        limits,
        consume,
        target=lambda value: value,
        failure_factory=_failure,
        observe_managed_admission=observe,
    )
    receipt = continued.result.execution_evidence
    assert receipt is not None
    assert receipt.history_evidence == (source.execution_evidence,)
    recorded = receipt.native_evidence()
    assert recorded.status is MeshcoreStatus.OK
    if observe:
        assert recorded.managed_admission_peak_bytes_upper is not None
        assert recorded.managed_admission_peak_bytes_upper >= 1024
    else:
        assert recorded.managed_admission_peak_bytes_upper is None
    assert current_native_execution_scope() is None


def _consume(work: int) -> None:
    scope = current_native_execution_scope()
    assert scope is not None
    with scope.batch(work) as ticket:
        ticket.consume(work)


def test_history_records_restore_their_complete_causal_graph(tmp_path: Path) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    history = _ended(3)
    preparation = _ended(2, owner="request")
    with NativeExecutionScope(_limits(1_000_000).native_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.record_history(request, history)
        ledger.import_receipt(request, preparation)
        imports = NativeExecutionImports(request, preparation=preparation)
    record = NativeExecutionRecord(
        imports.evidence(),
        preparation_evidence=preparation,
        history_evidence=(history,),
        owner_id="request",
        imports=imports,
    )
    archive = write_meshing_source_closure(tmp_path / "request-history", record)
    restored = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id
    )
    assert isinstance(restored, NativeExecutionRecord)
    restored.require_valid()
    assert restored.to_record() == record.to_record()
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(record)
    assert restored.history_evidence[0].receipt_hex() == history.receipt_hex()


def _adaptation_inputs(
    work: int,
) -> tuple[CellMeshingResult, MarkedMeshAdaptation, MeshAdaptationPolicy]:
    limits = MeshingLimits(maximum_work_units=work)
    source = phx.meshing.certify_cell_mesh(
        triangle_grid(4, 4), phx.SpatialCoordinateContract.si()
    ).with_execution_evidence(_ended(1_000, owner="generation"))
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
        limits=limits,
    )
    return source, MarkedMeshAdaptation(np.asarray([0], dtype=np.int64)), policy


def test_adaptation_charges_its_preparation_and_keeps_source_history() -> None:
    source, request, policy = _adaptation_inputs(1_000_000)
    prepared = prepare_mesh_adaptation(source, request, policy=policy)
    preparation = prepared.preparation_evidence
    assert preparation is not None and preparation.owner_id == prepared.prepared_id
    assert int(np.asarray(preparation.status)) == int(MeshcoreStatus.OK)
    adapted = execute_mesh_adaptation(prepared)
    record = adapted.target.execution_evidence
    assert record is not None and record.owner_id == prepared.prepared_id
    assert record.preparation_evidence is preparation
    history = source.execution_evidence
    assert history is not None and record.history_evidence == (history,)
    assert history.receipt_hex() not in record.imported_preparation_hex()
    assert int(np.asarray(record.total_work_units)) == int(
        np.asarray(preparation.total_work_units)
    ) + int(np.asarray(record.work_counters.construction_work))


@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_adaptation_request_limit_boundary_includes_its_preparation(slack: int) -> None:
    source, request, policy = _adaptation_inputs(1_000_000)
    adapted = execute_mesh_adaptation(
        prepare_mesh_adaptation(source, request, policy=policy)
    )
    record = adapted.target.execution_evidence
    assert record is not None
    observed = int(np.asarray(record.total_work_units))
    # The 1000-unit source history would exceed the boundary if it were charged.
    source, request, policy = _adaptation_inputs(observed + slack)
    if slack == 0:
        bounded = execute_mesh_adaptation(
            prepare_mesh_adaptation(source, request, policy=policy)
        )
        assert bounded.target.mesh.topology_id == adapted.target.mesh.topology_id
        assert bounded.target.execution_evidence is not None
        assert int(np.asarray(bounded.target.execution_evidence.total_work_units)) == (
            observed
        )
        return
    with pytest.raises((MeshingFailure, MeshcoreError)) as caught:
        execute_mesh_adaptation(prepare_mesh_adaptation(source, request, policy=policy))
    match caught.value:
        case MeshingFailure() as failure:
            assert failure.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
        case MeshcoreError() as error:
            assert error.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("mutate_before_capture", (False, True))
def test_private_polynomial_operands_reuse_only_paid_current_source(
    monkeypatch: pytest.MonkeyPatch, mutate_before_capture: bool,
) -> None:
    from fractions import Fraction

    from phydrax.discretization import _coordinate_enclosure as algebra
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.discretization._coordinate_operands import CoordinateOperandCollector

    mesh = triangle_grid(1, 1)
    geometry = phx.discretization.CellGeometrySpec.affine(mesh)
    limits = _limits(1_000_000)
    collector = CoordinateOperandCollector(capture_polynomials=True)
    with NativeExecutionScope(limits.native_limits()) as earning:
        collector.begin(
            earning, "numeric-polynomial-earning",
            native_preparation_ledger(earning).overlap(earning, None),
        )
        with earning.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            budget.capture_coordinate_operands = True
            budget.capture_polynomial_operands = True
            bank = algebra.prepared_coordinate_source_bank(geometry, budget=budget)
            elements, routes, _ = geometry.resolve(mesh, budget=budget)
            local = tuple(bank[int(index)] for index in np.asarray(routes[0])[0])
            expected = algebra.coordinate_polynomials(elements[0], local, budget=budget)
            assert expected is not None
            if mutate_before_capture:
                expected[0][(0, 0)] = Fraction(999)
                with pytest.raises(ValueError, match="changed after numeric issue"):
                    collector.capture(mesh, geometry, budget, earning)
                collector.abort()
                return
            expected_items = tuple(tuple(value.items()) for value in expected)
            collector.capture(mesh, geometry, budget, earning)
    receipt = NativeExecutionRecord(
        earning.completed_evidence(), owner_id="numeric-polynomial-earning"
    )
    collector.complete(receipt)
    operands = collector.for_result(mesh, geometry)
    assert operands is not None and operands._polynomial_snapshots
    # The exported snapshot owns new atoms/maps; a retained mutable work map
    # cannot mutate a later consumer's prepared numerical source.
    expected[0][(0, 0)] = Fraction(987)

    def unexpected_expansion(*args: object, **kwargs: object) -> None:
        raise AssertionError("paid polynomial arithmetic must not be repeated")

    monkeypatch.setattr(algebra, "_prepare_coordinate_polynomials", unexpected_expansion)
    with NativeExecutionScope(limits.native_limits()) as current:
        native_preparation_ledger(current).import_receipt(current, receipt)
        with current.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            operands.require(mesh, geometry, receipt, budget)
            restored = algebra.coordinate_polynomials(elements[0], local, budget=budget)
            assert restored is not None
            assert tuple(tuple(value.items()) for value in restored) == expected_items
            assert restored is not expected
            restored[0][(0, 0)] = Fraction(123)
            with pytest.raises(ValueError, match="conflict"):
                operands.require(mesh, geometry, receipt, budget)
    with NativeExecutionScope(limits.native_limits()) as cold:
        with cold.ticket_view() as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with pytest.raises(ValueError, match="already be paid"):
                operands.require(mesh, geometry, receipt, budget)


def test_prepaid_preparation_clock_and_import_keep_original_work_boundary() -> None:
    """Clock tightening precedes authentication; preparation is still paid once."""
    from contextlib import nullcontext
    from dataclasses import replace

    prior = _ended(50, 2)
    original_id = prior.receipt_hex()
    seconds = float(np.asarray(prior.total_elapsed_seconds))
    full = _limits(1_000_000, 2).native_limits()
    idle = 0.25

    def import_prior(scope: NativeExecutionScope) -> NativeExecutionImports:
        ledger = native_preparation_ledger(scope)
        expiry = scope.operation_started + full.maximum_seconds - seconds - idle
        ledger.prepay_receipt_clock(scope, prior, expires_at=expiry)
        ledger.import_receipt(scope, prior)
        return NativeExecutionImports(scope, preparation=prior)

    with NativeExecutionScope(full) as measured:
        imports = import_prior(measured)
        required = full.maximum_work_units - measured.remaining().remaining_work_units
    assert measured.completed_evidence().prior_elapsed_seconds == pytest.approx(seconds + idle)
    assert imports.evidence().prior_elapsed_seconds == pytest.approx(idle)
    assert imports.evidence().work_counters.source_queries == 0
    assert imports.evidence().work_counters.construction_work == required - 50

    for shortfall in (0, 1):
        scope = NativeExecutionScope(replace(full, maximum_work_units=required - shortfall))
        expectation = pytest.raises(MeshcoreError) if shortfall else nullcontext()
        with expectation as caught, scope:
            ledger = native_preparation_ledger(scope)
            imported = import_prior(scope)
            assert scope.remaining().remaining_work_units == 0
            assert scope.remaining().remaining_geometry_queries == 0
        if shortfall:
            assert caught is not None
            assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
            assert ledger._storage == []
            assert ledger._receipt_graphs == {}
        else:
            assert imported.evidence().work_counters.construction_work == required - 50
            assert imported.evidence().prior_elapsed_seconds == pytest.approx(idle)
        assert prior.receipt_hex() == original_id


def test_receipt_authentication_refuses_before_numerical_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prior = _ended(3)
    attempted = False

    def forbidden(self: NativeExecutionRecord) -> NoReturn:
        nonlocal attempted
        attempted = True
        raise AssertionError("Unadmitted receipt numerics must not be inspected.")

    monkeypatch.setattr(NativeExecutionRecord, "_require_phase_valid", forbidden)
    with (
        pytest.raises(MeshcoreError) as caught,
        NativeExecutionScope(_limits(2).native_limits()) as scope,
    ):
        prior.require_valid_in_scope(scope)
    assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    refusal = scope.completed_evidence()
    assert refusal.memory_counters.live_bytes == 0
    assert refusal.host_storage_live_bytes_upper == 0
    assert not attempted


@pytest.mark.parametrize("shortfall", (0, 1), ids=("R", "R-1"))
def test_receipt_authentication_storage_exact_boundary(shortfall: int) -> None:
    """Cold receipt validation admits exactly its declared scratch and refuses R-1."""
    from contextlib import nullcontext

    from phydrax._meshcore import NativeHostStorageWorkspace
    from phydrax.meshing._measurements import native_receipt_validation_bytes_upper

    prior = _ended(3)
    limits = _limits(1_000_000).native_limits()
    with NativeExecutionScope(limits) as measured:
        prior.require_valid_in_scope(measured)
    required = measured.completed_evidence().host_storage_peak_bytes_upper
    assert required == native_receipt_validation_bytes_upper(
        1, 3, 0, len("completed-request")
    )
    # A one-phase receipt fits the registered distributed 1 MiB scratch.
    assert required < 1024**2 // 4
    expectation = pytest.raises(MeshcoreError) if shortfall else nullcontext()
    with expectation as caught, NativeExecutionScope(limits) as scope:
        with NativeHostStorageWorkspace(scope) as ballast:
            ballast.reserve_empty()
            # The validation workspace's own empty token is native-managed.
            before = scope.remaining().remaining_managed_bytes
            with NativeHostStorageWorkspace(scope) as token:
                token.reserve_empty()
                token_bytes = before - scope.remaining().remaining_managed_bytes
            available = scope.remaining().remaining_managed_bytes
            ballast.set_bound(available - required - token_bytes + shortfall)
            prior.require_valid_in_scope(scope)
    if shortfall:
        assert caught is not None
        assert caught.value.status is MeshcoreStatus.CAPACITY_EXCEEDED


@pytest.mark.parametrize("history_count", (0, 8, 32))
def test_receipt_authentication_storage_bounds_measured_host_peak(
    history_count: int,
) -> None:
    """The admitted scratch is an upper bound on the actual traced host peak."""
    import gc
    import tracemalloc

    imported = [_ended(5 + index, owner="imported-phase") for index in range(64)]
    history = tuple(_ended(200 + index) for index in range(history_count))
    with NativeExecutionScope(_limits(10_000_000, 10).native_limits()) as scope:
        ledger = native_preparation_ledger(scope)
        for record in imported:
            ledger.import_receipt(scope, record)
        imports = NativeExecutionImports(scope)
    root = NativeExecutionRecord(
        imports.evidence(), owner_id="root-phase", imports=imports,
        history_evidence=history,
    )
    assert len(root.imported_preparation_hex()) == 64
    with NativeExecutionScope(_limits(10_000_000).native_limits()) as receiving:
        gc.collect()
        tracemalloc.start()
        try:
            root.require_valid_in_scope(receiving)
            _, traced_peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
    admitted = receiving.completed_evidence().host_storage_peak_bytes_upper
    assert traced_peak <= admitted
