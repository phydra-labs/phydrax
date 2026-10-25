# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Completed-source authority belongs to actual paid publication roots only."""

from __future__ import annotations

import gc
import subprocess
import sys
import weakref
from collections.abc import Mapping
from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx
import phydrax.meshing._native_preparation as preparation
from phydrax._fingerprint import canonical_fingerprint
from phydrax._host_storage import immutable_host_array
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    NativeExecutionLimits,
    NativeExecutionScope,
)
from phydrax.discretization._coordinate_budget import CoordinateEnclosureResourceError
from phydrax.meshing._measurements import NativeExecutionImports, NativeExecutionRecord
from phydrax.meshing._native_preparation import (
    _completed_native_sources,
    _native_source_array_content,
    _native_source_budget,
    _native_source_seal,
    _require_native_completed_source,
    _require_native_source_seal,
    continue_native_completed_source_issue,
    native_preparation_ledger,
    NativeCompletedSourceIssue,
    NativePreparationOverlap,
    prepare_native_completed_source_issue,
    stage_native_completed_source_issue,
    stage_native_source_publication,
)
from phydrax.meshing._result import CellMeshingResult
from tests._support.cell_meshes import triangle_grid


pytestmark = pytest.mark.skipif(
    not meshcore_available(), reason="native meshcore required"
)


def _limits() -> NativeExecutionLimits:
    return NativeExecutionLimits(10_000_000, 100_000, 100_000, 128 * 1024**2, 120.0)


@pytest.fixture
def source() -> CellMeshingResult:
    # This constructor independently closes the scientific publication. Staging
    # below models only the explicit post-full-validation cold boundary.
    return phx.meshing.certify_cell_mesh(
        triangle_grid(1, 1), phx.SpatialCoordinateContract.si()
    )


def _prepare_issue(
    scope: NativeExecutionScope,
    source: CellMeshingResult,
    *,
    combined: bool,
) -> NativeCompletedSourceIssue:
    if combined:
        return stage_native_completed_source_issue(scope, source)
    stage_native_source_publication(scope, source)
    return prepare_native_completed_source_issue(scope, source)


def _complete(
    source: CellMeshingResult,
    *,
    combined: bool = False,
) -> tuple[CellMeshingResult, NativeCompletedSourceIssue]:
    with NativeExecutionScope(_limits()) as scope:
        issue = _prepare_issue(scope, source, combined=combined)
    receipt = NativeExecutionRecord(scope.completed_evidence(), owner_id=source.result_id)
    bound = source.with_execution_evidence(receipt)
    issue.complete(bound)
    return bound, issue


@pytest.mark.parametrize("cleanup", ("explicit", "gc", "module"))
def test_completed_issue_cleanup_releases_once(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch, cleanup: str,
) -> None:
    bound, issue = _complete(source)
    identity = id(bound)
    reference = weakref.ref(bound)
    issue_reference = weakref.ref(issue)
    gc.collect()
    assert issue._root() is None
    release = issue._release_metadata
    assert release is not None
    calls: list[None] = []

    def observed_release() -> None:
        calls.append(None)
        release()

    issue._release_metadata = observed_release
    registry = _completed_native_sources
    if cleanup == "module":
        monkeypatch.setattr(preparation, "_completed_native_sources", None)
    if cleanup == "explicit":
        issue.close()
        issue.close()
        assert reference() is bound
        assert issue._source is None
        assert calls == [None]
    del bound
    gc.collect()
    assert reference() is None
    assert identity not in registry
    assert issue._registry is None
    assert issue._release_metadata is None
    assert issue._seal is None
    assert calls == [None]
    issue.close()
    assert calls == [None]
    del issue
    gc.collect()
    assert issue_reference() is None


def test_completed_issue_interpreter_exit_has_no_unraisable_stderr() -> None:
    script = """
import atexit
import gc
import weakref
import phydrax as phx
import phydrax.meshing._native_preparation as preparation
from tests._support.cell_meshes import triangle_grid
from tests.unit.meshing.test_native_completed_source_publication import _complete

source = phx.meshing.certify_cell_mesh(
    triangle_grid(1, 1), phx.SpatialCoordinateContract.si()
)
bound, issue = _complete(source)
registry = preparation._completed_native_sources
identity = id(bound)
reference = weakref.ref(bound)
release = issue._release_metadata
calls = []
def observed_release():
    calls.append(None)
    release()
issue._release_metadata = observed_release
def cleanup():
    global bound
    preparation._completed_native_sources = None
    # Simulate the late module-cleanup phase after weakref's exit hook.
    weakref.finalize._shutdown = True
    bound = None
    gc.collect()
    assert reference() is None
    assert identity not in registry
    assert calls == [None]
    issue.close()
    assert calls == [None]
    print("cleanup complete")
atexit.register(cleanup)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "cleanup complete"
    assert result.stderr == ""


def test_arbitrary_ended_preparation_wrapper_cannot_issue(
    source: CellMeshingResult,
) -> None:
    with NativeExecutionScope(_limits()) as preparation:
        with preparation.batch(3) as ticket:
            ticket.consume(3)
    wrapper = source.with_execution_evidence(
        NativeExecutionRecord(preparation.completed_evidence(), owner_id=source.result_id)
    )
    with NativeExecutionScope(_limits()) as request:
        with pytest.raises(ValueError, match="publication|completed|staged"):
            prepare_native_completed_source_issue(request, wrapper)


def test_completed_root_is_actual_weak_source_owner(source: CellMeshingResult) -> None:
    bound, issue = _complete(source)
    receipt = bound.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(_limits()):
        _require_native_completed_source(bound, receipt)
        # A numerically equal wrapper has not itself been issued by the root.
        with pytest.raises(ValueError, match="completed|issued|owner"):
            _require_native_completed_source(
                bound.with_execution_evidence(receipt), receipt
            )
    reference = weakref.ref(bound)
    del bound
    gc.collect()
    assert reference() is None
    issue.close()
    issue.close()


def test_completed_source_continuation_uses_current_request_history(
    source: CellMeshingResult,
) -> None:
    bound, _ = _complete(source)
    receipt = bound.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.record_history(request, receipt, completed_source=bound)
        issue = prepare_native_completed_source_issue(request, bound)
    continued = bound.with_execution_evidence(
        NativeExecutionRecord(
            request.completed_evidence(),
            owner_id=bound.result_id,
            history_evidence=(receipt,),
        )
    )
    issue.complete(continued)
    from phydrax.lifecycle._meshing_field_records import (
        _require_execution_endpoint_extension,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts

    register_meshing_source_artifacts()
    # A completed request named as history is provenance only; it never
    # authorizes rebinding an accepted endpoint's same-request receipt.
    with pytest.raises(ValueError, match="continue the actual preceding receipt"):
        _require_execution_endpoint_extension(bound, continued)
    with NativeExecutionScope(_limits()):
        assert continued.execution_evidence is not None
        _require_native_completed_source(continued, continued.execution_evidence)


def test_large_live_publication_rebind_keeps_archive_limits(
    source: CellMeshingResult,
) -> None:
    from phydrax._array_archive import DEFAULT_ARRAY_ARCHIVE_LIMITS
    from phydrax._model._structure import model_recipe_wire_json, model_structure_recipe
    from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
    from phydrax.lifecycle._meshing_field_records import (
        _require_execution_endpoint_extension,
    )
    from phydrax.lifecycle._meshing_sources import register_meshing_source_artifacts
    from phydrax.meshing.qualification._native_lifecycle import execute_native_consumer

    provenance = phx.SemanticProvenance(
        {
            "kind": "large-live-publication",
            "source": source.provenance.semantic_id,
            "description": "x" * DEFAULT_ARRAY_ARCHIVE_LIMITS.max_manifest_bytes,
        }
    )
    large = CellMeshingResult(
        source.mesh, source.geometry, source.coordinate_contract, source.audit,
        source.quality, source.compliance, source.trace, source.provider,
        source.runtime, source.derivative_mode, provenance,
        boundary=source.boundary, patches=source.patches,
    )
    bound, _ = _complete(large)
    limits = phx.meshing.MeshingLimits(
        maximum_work_units=10_000_000,
        maximum_geometry_queries=100_000,
        maximum_cavity_cells=100_000,
        maximum_scratch_bytes=128 * 1024**2,
        maximum_wall_seconds=120.0,
    )

    def consume(budget: CoordinateEnclosureBudget) -> None:
        budget.reserve(1)

    def fail(stage: str, message: str, data: dict[str, object]) -> ValueError:
        return ValueError(f"{stage}: {message}; {data}")

    continued = execute_native_consumer(
        bound, limits, consume, failure_factory=fail,
    ).result
    register_meshing_source_artifacts()
    _require_execution_endpoint_extension(bound, continued)
    assert continued.result_id == bound.result_id
    assert continued.execution_evidence is not bound.execution_evidence
    with pytest.raises(ValueError, match=r"recipe\.bytes"):
        model_recipe_wire_json(model_structure_recipe(continued))


def test_constructor_replacement_refuses_staged_publication(
    source: CellMeshingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with NativeExecutionScope(_limits()) as scope:
        stage_native_source_publication(scope, source)
        monkeypatch.setattr(type(source), "__init__", lambda *args, **kwargs: None)
        with pytest.raises(ValueError, match="constructor"):
            prepare_native_completed_source_issue(scope, source)


def test_source_bytes_are_fully_scanned_even_when_readonly() -> None:
    values = np.arange(3 * 1024**2, dtype=np.uint8)
    values.flags.writeable = False
    with NativeExecutionScope(_limits()) as scope:
        with _native_source_budget(scope, None) as budget:
            original = _native_source_array_content((values,), budget)
            values.flags.writeable = True
            values[-1] ^= 1
            values.flags.writeable = False
            changed = _native_source_array_content((values,), budget)
    assert changed != original


@pytest.mark.parametrize("host", (False, True), ids=("device", "bytes-backed-host"))
def test_huge_immutable_bank_publication_pays_owner_visits_not_elements(
    source: CellMeshingResult,
    host: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _fingerprint

    # Deliberately isolate ownership/accounting from the already completed
    # scientific constructor; no publication scan may read this device bank.
    bank = (
        immutable_host_array(np.arange(2_000_000, dtype=np.float32))
        if host else jnp.arange(2_000_000, dtype=jnp.float32)
    )
    object.__setattr__(source.mesh, "coordinates", bank)
    digest = _fingerprint.logical_array_value_collection_digest

    def host_digest(
        arrays: Mapping[str, jax.Array | np.ndarray],
        *,
        maximum_chunk_bytes: int,
    ) -> str:
        assert all(type(array) is np.ndarray for array in arrays.values())
        assert all(array is not bank for array in arrays.values())
        return digest(arrays, maximum_chunk_bytes=maximum_chunk_bytes)

    monkeypatch.setattr(
        _fingerprint, "logical_array_value_collection_digest", host_digest
    )
    with NativeExecutionScope(replace(_limits(), maximum_work_units=100_000)) as root:
        stage_native_source_publication(root, source)
        seal = native_preparation_ledger(root)._publications[id(source)][1]
        if host:
            assert any(array is bank for array, _ in seal.immutable_host_arrays)
            assert type(bank) is np.ndarray and not bank.flags.writeable
        else:
            assert any(array is bank for array in seal.immutable_arrays)
        assert all(type(array) is np.ndarray for array in seal.mutable_arrays)
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    issue.complete(bound)
    assert bound.execution_evidence is not None
    assert int(np.asarray(bound.execution_evidence.total_work_units)) < bank.size
    required = (
        1
        + len(seal.names)
        + 4 * len(seal.constructors)
        + len(seal.immutable_arrays)
        + len(seal.immutable_host_arrays)
        + len(seal.mutable_arrays)
        + sum(1 + len(names) for _, _, names, _ in seal.owners)
        + sum(1 + len(children) for _, children in seal.containers)
        + sum(1 + array.size for array in seal.mutable_arrays)
    )
    with NativeExecutionScope(replace(_limits(), maximum_work_units=required)) as exact:
        with _native_source_budget(exact, None) as budget:
            _require_native_source_seal(bound, seal, budget)
    assert (
        int(
            np.asarray(NativeExecutionRecord(exact.completed_evidence()).total_work_units)
        )
        == required
    )
    with pytest.raises(CoordinateEnclosureResourceError):
        with NativeExecutionScope(
            replace(_limits(), maximum_work_units=required - 1)
        ) as short:
            with _native_source_budget(short, None) as budget:
                _require_native_source_seal(bound, seal, budget)
    issue.close()


@pytest.mark.parametrize("boundary", ("stage", "prepare", "complete", "history"))
def test_deleted_immutable_owner_refuses_each_boundary(
    source: CellMeshingResult,
    boundary: str,
) -> None:
    bank = jnp.array(source.mesh.coordinates, copy=True)
    object.__setattr__(source.mesh, "coordinates", bank)
    if boundary == "history":
        bound, issue = _complete(source)
        bank.delete()
        with NativeExecutionScope(_limits()):
            assert bound.execution_evidence is not None
            with pytest.raises(ValueError, match="immutable.*live"):
                _require_native_completed_source(bound, bound.execution_evidence)
        issue.close()
        return
    with NativeExecutionScope(_limits()) as root:
        if boundary == "stage":
            bank.delete()
            with pytest.raises(ValueError, match="immutable.*live"):
                stage_native_source_publication(root, source)
            return
        stage_native_source_publication(root, source)
        if boundary == "prepare":
            bank.delete()
            with pytest.raises(ValueError, match="immutable.*live"):
                prepare_native_completed_source_issue(root, source)
            return
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    bank.delete()
    with pytest.raises(ValueError, match="immutable.*live"):
        issue.complete(bound)


@pytest.mark.parametrize("after_prepare", (False, True))
@pytest.mark.parametrize("container", (False, True), ids=("field", "container"))
def test_replaced_immutable_owner_refuses(
    source: CellMeshingResult,
    after_prepare: bool,
    container: bool,
) -> None:
    original = source.mesh.coordinates
    assert isinstance(original, jax.Array)
    bank = [original] if container else original
    object.__setattr__(source.mesh, "coordinates", bank)
    with NativeExecutionScope(_limits()) as root:
        stage_native_source_publication(root, source)
        issue = (
            prepare_native_completed_source_issue(root, source) if after_prepare else None
        )
        replacement = jnp.array(original, copy=True)
        if container:
            assert isinstance(bank, list)
            bank[0] = replacement
        else:
            object.__setattr__(source.mesh, "coordinates", replacement)
        if not after_prepare:
            with pytest.raises(ValueError, match="fields|owners|bytes"):
                prepare_native_completed_source_issue(root, source)
            return
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    assert issue is not None
    with pytest.raises(ValueError, match="owners|containers"):
        issue.complete(bound)


@pytest.mark.parametrize("boundary", ("stage", "prepare", "combined"))
@pytest.mark.parametrize("mutation", ("bytes", "dtype", "shape", "resize", "container"))
def test_mutable_bank_mutations_refuse_each_window(
    source: CellMeshingResult,
    boundary: str,
    mutation: str,
) -> None:
    bank = np.array(source.mesh.coordinates)
    container = [bank]
    object.__setattr__(
        source.mesh, "coordinates", container if mutation == "container" else bank
    )
    bank.flags.writeable = False
    with NativeExecutionScope(_limits()) as root:
        if boundary == "stage":
            stage_native_source_publication(root, source)
            issue = None
        else:
            issue = _prepare_issue(root, source, combined=boundary == "combined")
        bank.flags.writeable = True
        if mutation == "bytes":
            bank.flat[-1] += 0.125
        elif mutation == "dtype":
            object.__setattr__(bank, "dtype", np.dtype(f"u{bank.dtype.itemsize}"))
        elif mutation == "shape":
            object.__setattr__(bank, "shape", (bank.size,))
        elif mutation == "resize":
            # Exercise the still-available in-place layout attack explicitly.
            bank.resize((bank.size * 100,), refcheck=False)  # ty: ignore[deprecated]
        else:
            container[0] = bank.copy()
        bank.flags.writeable = False
        if boundary == "stage":
            with pytest.raises(ValueError, match="bytes|dtype|shape|fields"):
                prepare_native_completed_source_issue(root, source)
            return
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    assert issue is not None
    with pytest.raises(ValueError, match="bytes|dtype|shape|containers"):
        issue.complete(bound)


@pytest.mark.parametrize("mutation", ("deleted", "replaced", "constructor"))
def test_combined_issue_retains_final_immutable_and_constructor_checks(
    source: CellMeshingResult,
    mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = jnp.array(source.mesh.coordinates, copy=True)
    object.__setattr__(source.mesh, "coordinates", bank)
    with NativeExecutionScope(_limits()) as root:
        issue = stage_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    if mutation == "deleted":
        bank.delete()
    elif mutation == "replaced":
        object.__setattr__(source.mesh, "coordinates", jnp.array(bank, copy=True))
    else:
        monkeypatch.setattr(type(source), "__init__", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="immutable|owners|constructor"):
        issue.complete(bound)


def test_combined_issue_removes_only_the_recheck_and_duplicate_retention(
    source: CellMeshingResult,
) -> None:
    object.__setattr__(source.mesh, "coordinates", np.array(source.mesh.coordinates))
    separate, separate_issue = _complete(source)
    combined, combined_issue = _complete(source, combined=True)
    assert separate.execution_evidence is not None
    assert combined.execution_evidence is not None
    seal = combined_issue._seal
    assert seal is not None
    with NativeExecutionScope(_limits()) as recheck:
        with _native_source_budget(recheck, None) as budget:
            _require_native_source_seal(source, seal, budget)
    # Separate staging also retains its seal on the ledger; the combined issue's
    # own token covers that seal, so only that retention and the recheck vanish.
    with NativeExecutionScope(_limits()) as staging:
        stage_native_source_publication(staging, source)
    with NativeExecutionScope(_limits()) as sealing:
        with _native_source_budget(sealing, None) as budget:
            _native_source_seal(source, budget)

    def work(scope: NativeExecutionScope) -> int:
        record = NativeExecutionRecord(scope.completed_evidence())
        return int(np.asarray(record.total_work_units))

    retention = work(staging) - work(sealing)
    assert retention > 0
    assert int(np.asarray(separate.execution_evidence.total_work_units)) == (
        int(np.asarray(combined.execution_evidence.total_work_units))
        + work(recheck)
        + retention
    )
    separate_issue.close()
    combined_issue.close()


def test_complete_requires_ended_actual_scope(source: CellMeshingResult) -> None:
    with NativeExecutionScope(_limits()) as request:
        stage_native_source_publication(request, source)
        issue = prepare_native_completed_source_issue(request, source)
        with pytest.raises(ValueError, match="ended|complete"):
            issue.complete(source)
        issue.close()
        issue.close()


def test_normalized_counters_must_match_actual_ended_root(
    source: CellMeshingResult,
) -> None:
    with NativeExecutionScope(_limits()) as scope:
        stage_native_source_publication(scope, source)
        issue = prepare_native_completed_source_issue(scope, source)
    evidence = scope.completed_evidence()
    counters = replace(
        evidence.work_counters,
        construction_work=evidence.work_counters.construction_work + 1,
    )
    false = source.with_execution_evidence(
        NativeExecutionRecord(
            replace(evidence, work_counters=counters), owner_id=source.result_id
        )
    )
    with pytest.raises(ValueError, match="counter|evidence|scope|root"):
        issue.complete(false)
    issue.close()


def test_normalization_modes_do_not_accept_both_imports_and_overlap(
    source: CellMeshingResult,
) -> None:
    from phydrax.meshing._measurements import NativeExecutionImports

    with NativeExecutionScope(_limits()) as scope:
        stage_native_source_publication(scope, source)
        imports = NativeExecutionImports(scope)
        with pytest.raises(ValueError, match="normaliz|exclusive|overlap|imports"):
            prepare_native_completed_source_issue(
                scope, source, imports=imports, overlap=NativePreparationOverlap()
            )


def test_history_cannot_be_imported_or_normalized_as_preparation(
    source: CellMeshingResult,
) -> None:
    bound, _ = _complete(source)
    receipt = bound.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(_limits()) as request:
        ledger = native_preparation_ledger(request)
        ledger.record_history(request, receipt, completed_source=bound)
        assert ledger.overlap(request, receipt) == NativePreparationOverlap()
        with pytest.raises(ValueError, match="history"):
            ledger.import_receipt(request, receipt)


def test_prepared_issue_rejects_changed_constructor_after_end(
    source: CellMeshingResult,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with NativeExecutionScope(_limits()) as scope:
        stage_native_source_publication(scope, source)
        issue = prepare_native_completed_source_issue(scope, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(scope.completed_evidence(), owner_id=source.result_id)
    )
    monkeypatch.setattr(type(source), "__init__", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="constructor"):
        issue.complete(bound)
    issue.close()


def test_issue_refuses_foreign_root_and_current_scope_completion(
    source: CellMeshingResult,
) -> None:
    with NativeExecutionScope(_limits()) as root:
        stage_native_source_publication(root, source)
        with NativeExecutionScope(_limits()) as child:
            with pytest.raises(ValueError, match="root"):
                prepare_native_completed_source_issue(child, source)
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    with NativeExecutionScope(_limits()):
        with pytest.raises(ValueError, match="ended"):
            issue.complete(bound)
    issue.complete(bound)


def test_completion_scans_readonly_bytes_after_prepare(source: CellMeshingResult) -> None:
    coordinates = np.array(source.mesh.coordinates)
    coordinates.flags.writeable = False
    object.__setattr__(source.mesh, "coordinates", coordinates)
    with NativeExecutionScope(_limits()) as root:
        stage_native_source_publication(root, source)
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    coordinates.flags.writeable = True
    coordinates.flat[-1] += 0.125
    coordinates.flags.writeable = False
    with pytest.raises(ValueError, match="bytes"):
        issue.complete(bound)
    with pytest.raises(ValueError, match="closed"):
        issue.complete(bound)


@pytest.mark.parametrize("combined", (False, True), ids=("separated", "atomic"))
def test_publication_exact_work_boundary_and_one_short_cleanup(
    source: CellMeshingResult,
    combined: bool,
) -> None:
    measured, measured_issue = _complete(source, combined=combined)
    assert measured.execution_evidence is not None
    required = int(np.asarray(measured.execution_evidence.total_work_units))
    measured_issue.close()
    with NativeExecutionScope(replace(_limits(), maximum_work_units=required)) as exact:
        issue = _prepare_issue(exact, source, combined=combined)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(exact.completed_evidence(), owner_id=source.result_id)
    )
    issue.complete(bound)
    with NativeExecutionScope(_limits()):
        assert bound.execution_evidence is not None
        _require_native_completed_source(bound, bound.execution_evidence)
    with pytest.raises(MeshcoreError):
        with NativeExecutionScope(
            replace(_limits(), maximum_work_units=required - 1)
        ) as short:
            _prepare_issue(short, source, combined=combined)
    assert short.evidence is not None
    assert short.evidence.host_storage_live_bytes_upper == 0
    issue.close()


def _prepare_continuation(
    root: NativeExecutionScope, source: CellMeshingResult,
) -> tuple[NativeCompletedSourceIssue, NativePreparationOverlap]:
    receipt = source.execution_evidence
    assert receipt is not None
    ledger = native_preparation_ledger(root)
    ledger.import_receipt(root, receipt)
    overlap = ledger.overlap(root, receipt)
    return continue_native_completed_source_issue(
        root, source, receipt, overlap=overlap,
    ), overlap


def _bind_continuation(
    root: NativeExecutionScope,
    source: CellMeshingResult,
    overlap: NativePreparationOverlap,
) -> CellMeshingResult:
    receipt = source.execution_evidence
    assert receipt is not None
    return source.with_execution_evidence(
        NativeExecutionRecord(
            overlap.normalize(root.completed_evidence()),
            owner_id=receipt.owner_id,
            preparation_evidence=receipt,
            consumer_evidence=receipt,
        )
    )


def test_continuation_reuses_seal_and_only_scans_mutable_bytes_after_end(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from phydrax import _fingerprint
    from phydrax._meshcore import current_native_execution_scope

    object.__setattr__(source.mesh, "coordinates", np.array(source.mesh.coordinates))
    bound, previous = _complete(source, combined=True)
    digest = _fingerprint.logical_array_value_collection_digest
    scans = []

    def final_digest(
        arrays: Mapping[str, jax.Array | np.ndarray],
        *,
        maximum_chunk_bytes: int,
    ) -> str:
        assert current_native_execution_scope() is None
        scans.append(tuple(arrays.values()))
        return digest(arrays, maximum_chunk_bytes=maximum_chunk_bytes)

    monkeypatch.setattr(
        _fingerprint, "logical_array_value_collection_digest", final_digest,
    )
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
        assert issue._seal is previous._seal
        assert not native_preparation_ledger(root)._publications
        assert not scans
    continued = _bind_continuation(root, bound, overlap)
    issue.complete(continued)
    assert len(scans) == 1
    assert scans[0]
    assert _completed_native_sources[id(continued)] is issue


@pytest.mark.parametrize("immutable", (False, True), ids=("mutable", "immutable-host"))
def test_continuation_exact_work_delta_and_one_short_cleanup(
    source: CellMeshingResult, immutable: bool,
) -> None:
    bank = np.array(source.mesh.coordinates)
    object.__setattr__(
        source.mesh, "coordinates", immutable_host_array(bank) if immutable else bank,
    )
    bound, original = _complete(source, combined=True)
    receipt = bound.execution_evidence
    seal = original._seal
    assert receipt is not None and seal is not None
    with NativeExecutionScope(_limits()) as staged:
        ledger = native_preparation_ledger(staged)
        ledger.import_receipt(staged, receipt)
        full_issue = stage_native_completed_source_issue(
            staged, bound, owner_id=receipt.owner_id,
            overlap=ledger.overlap(staged, receipt),
        )
    full_issue.close()
    with NativeExecutionScope(_limits()) as stage_only:
        stage_native_source_publication(stage_only, bound)
    cheap_work = (
        1 + len(seal.names) + 4 * len(seal.constructors)
        + len(seal.immutable_arrays) + len(seal.immutable_host_arrays)
        + len(seal.mutable_arrays)
    )
    with NativeExecutionScope(_limits()) as measured:
        measured_issue, overlap = _prepare_continuation(measured, bound)
    continued = _bind_continuation(measured, bound, overlap)
    measured_issue.complete(continued)
    required = measured.completed_evidence().work_counters.construction_work
    staging_saving = (
        stage_only.completed_evidence().work_counters.construction_work - cheap_work
    )
    assert staged.completed_evidence().work_counters.construction_work - required > (
        staging_saving
    )
    with NativeExecutionScope(replace(_limits(), maximum_work_units=required)) as exact:
        issue, overlap = _prepare_continuation(exact, bound)
        assert exact.remaining().remaining_work_units == 0
    exact_bound = _bind_continuation(exact, bound, overlap)
    issue.complete(exact_bound)
    assert exact_bound.execution_evidence is not None
    assert int(np.asarray(exact_bound.execution_evidence.total_work_units)) == required
    with pytest.raises(MeshcoreError):
        with NativeExecutionScope(
            replace(_limits(), maximum_work_units=required - 1)
        ) as short:
            _prepare_continuation(short, bound)
    assert short.completed_evidence().host_storage_live_bytes_upper == 0


def test_immutable_host_continuation_removes_only_complete_bank_scan(
    source: CellMeshingResult,
) -> None:
    values = np.arange(16_120, dtype=np.float64)
    demands = []
    for immutable in (False, True):
        bank = immutable_host_array(values) if immutable else values
        # Isolate publication accounting, as in the huge-device-bank test.
        object.__setattr__(source.mesh, "coordinates", bank)
        bound, original = _complete(source, combined=True)
        seal = original._seal
        assert seal is not None
        assert any(array is bank for array in seal.mutable_arrays) is not immutable
        assert any(array is bank for array, _ in seal.immutable_host_arrays) is immutable
        receipt = bound.execution_evidence
        assert receipt is not None
        with NativeExecutionScope(_limits()) as root:
            ledger = native_preparation_ledger(root)
            ledger.import_receipt(root, receipt)
            overlap = ledger.overlap(root, receipt)
            before = root.remaining().remaining_work_units
            issue = continue_native_completed_source_issue(
                root, bound, receipt, overlap=overlap
            )
            demands.append(before - root.remaining().remaining_work_units)
        continued = _bind_continuation(root, bound, overlap)
        issue.complete(continued)
        assert np.array_equal(bank, values)
        assert bank.dtype == values.dtype and bank.shape == values.shape
        issue.close()
        original.close()
    assert demands[0] - demands[1] == 1 + values.size
    assert demands[0] - demands[1] >= 16_120


@pytest.mark.parametrize("mutation", ("dtype", "shape", "owner"))
def test_immutable_host_continuation_rechecks_exact_layout_and_owner(
    source: CellMeshingResult, mutation: str,
) -> None:
    bank = immutable_host_array(np.array(source.mesh.coordinates))
    object.__setattr__(source.mesh, "coordinates", bank)
    bound, _ = _complete(source, combined=True)
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    if mutation == "dtype":
        object.__setattr__(bank, "dtype", np.dtype(f"u{bank.dtype.itemsize}"))
    elif mutation == "shape":
        object.__setattr__(bank, "shape", (bank.size,))
    else:
        object.__setattr__(bound.mesh, "coordinates", immutable_host_array(bank.copy()))
    with pytest.raises(ValueError, match="dtype|shape|owners"):
        issue.complete(continued)
    assert issue._seal is None and issue._storage is None


def test_readonly_memoryview_of_mutable_bytes_keeps_complete_scan(
    source: CellMeshingResult,
) -> None:
    mutable = bytearray(np.asarray(source.mesh.coordinates).tobytes())
    bank = np.frombuffer(
        memoryview(mutable).toreadonly(), dtype=source.mesh.coordinates.dtype
    ).reshape(source.mesh.coordinates.shape)
    object.__setattr__(source.mesh, "coordinates", bank)
    bound, original = _complete(source, combined=True)
    assert original._seal is not None
    assert any(array is bank for array in original._seal.mutable_arrays)
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    mutable[-1] ^= 1
    assert not bank.flags.writeable
    with pytest.raises(ValueError, match="bytes"):
        issue.complete(continued)


def test_continuation_prepared_graph_preserves_ids_and_skips_old_numerics(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch,
) -> None:
    bound, previous = _complete(source, combined=True)
    require_phase = NativeExecutionRecord._require_phase_valid
    for _ in range(3):
        paid = bound.execution_evidence
        graph = previous._receipt_graph
        assert paid is not None and graph is not None
        assert graph.graph_id == canonical_fingerprint(paid.to_record())
        with pytest.raises(FrozenInstanceError):
            setattr(graph, "phases", ())
        with NativeExecutionScope(_limits()) as root:
            issue, overlap = _prepare_continuation(root, bound)
        continued = _bind_continuation(root, bound, overlap)
        final = continued.execution_evidence
        assert final is not None
        expected_graph_id = canonical_fingerprint(final.to_record())
        expected_receipt_id = final.receipt_hex()
        checked = []

        def current_phase_only(record: NativeExecutionRecord) -> None:
            assert record is final
            checked.append(record)
            require_phase(record)

        with monkeypatch.context() as patch:
            patch.setattr(
                NativeExecutionRecord, "_require_phase_valid", current_phase_only
            )
            issue.complete(continued)
        assert checked == [final]
        assert issue._receipt_graph is not None
        assert issue._receipt_graph.graph_id == expected_graph_id
        assert final.receipt_hex() == expected_receipt_id
        assert issue._receipt_graph.phases[1:] == graph.phases
        bound, previous = continued, issue


def test_four_phase_continuation_saves_graph_work_and_has_exact_boundary(
    source: CellMeshingResult,
) -> None:
    bound, previous = _complete(source, combined=True)
    for _ in range(3):
        with NativeExecutionScope(_limits()) as root:
            issue, overlap = _prepare_continuation(root, bound)
        continued = _bind_continuation(root, bound, overlap)
        issue.complete(continued)
        bound, previous = continued, issue
    graph = previous._receipt_graph
    assert graph is not None and len(graph.phases) == 4
    # The unprepared comparison retains identical source owners and all original
    # numerical limits. Only the private graph preparation is unavailable.
    previous._receipt_graph = None
    try:
        with NativeExecutionScope(_limits()) as full:
            full_issue, _ = _prepare_continuation(full, bound)
        full_issue.close()
    finally:
        previous._receipt_graph = graph
    with NativeExecutionScope(_limits()) as measured:
        measured_issue, overlap = _prepare_continuation(measured, bound)
    continued = _bind_continuation(measured, bound, overlap)
    measured_issue.complete(continued)
    required = measured.completed_evidence().work_counters.construction_work
    assert full.completed_evidence().work_counters.construction_work - required > 11_100
    with NativeExecutionScope(replace(_limits(), maximum_work_units=required)) as exact:
        exact_issue, exact_overlap = _prepare_continuation(exact, bound)
        assert exact.remaining().remaining_work_units == 0
    exact_issue.complete(_bind_continuation(exact, bound, exact_overlap))
    with pytest.raises(MeshcoreError):
        with NativeExecutionScope(
            replace(_limits(), maximum_work_units=required - 1)
        ) as short:
            _prepare_continuation(short, bound)
    assert short.completed_evidence().host_storage_live_bytes_upper == 0


@pytest.mark.parametrize("mutation", ("leaf", "counter", "clone", "history", "deleted"))
def test_continuation_authenticates_every_prepared_historical_owner(
    source: CellMeshingResult, mutation: str,
) -> None:
    with NativeExecutionScope(_limits()) as earlier:
        pass
    histories = tuple(
        NativeExecutionRecord(earlier.completed_evidence(), owner_id=f"history-{index}")
        for index in range(2)
    )
    with NativeExecutionScope(_limits()) as published:
        ledger = native_preparation_ledger(published)
        for history in histories:
            ledger.record_history(published, history)
        initial_issue = stage_native_completed_source_issue(published, source)
    paid = NativeExecutionRecord(
        published.completed_evidence(), owner_id=source.result_id,
        history_evidence=histories,
    )
    bound = source.with_execution_evidence(paid)
    initial_issue.complete(bound)
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    historical = paid.history_evidence[0]
    if mutation == "leaf":
        object.__setattr__(
            historical, "elapsed_seconds", np.asarray(historical.elapsed_seconds).copy()
        )
    elif mutation == "counter":
        object.__setattr__(
            historical.work_counters, "construction_work",
            jnp.array(historical.work_counters.construction_work, copy=True),
        )
    elif mutation == "clone":
        clone = NativeExecutionRecord(
            historical.native_evidence(), owner_id=historical.owner_id,
        )
        assert clone.receipt_hex() == historical.receipt_hex()
        object.__setattr__(paid, "history_evidence", (clone, paid.history_evidence[1]))
    elif mutation == "history":
        object.__setattr__(paid, "history_evidence", tuple(reversed(paid.history_evidence)))
    else:
        historical.elapsed_seconds.delete()
    with pytest.raises(ValueError, match="receipt graph owners"):
        issue.complete(continued)
    assert issue._receipt_graph is None and issue._storage is None
    assert id(continued) not in _completed_native_sources


@pytest.mark.parametrize("mutation", ("counter", "aggregate", "imports", "consumer"))
def test_continuation_checks_new_phase_and_exact_root(
    source: CellMeshingResult, mutation: str,
) -> None:
    bound, _ = _complete(source, combined=True)
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    receipt = continued.execution_evidence
    assert receipt is not None
    if mutation == "counter":
        evidence = receipt.native_evidence()
        counters = replace(
            evidence.work_counters,
            construction_work=evidence.work_counters.construction_work + 1,
        )
        receipt = NativeExecutionRecord(
            replace(evidence, work_counters=counters), owner_id=receipt.owner_id,
            preparation_evidence=bound.execution_evidence,
            consumer_evidence=bound.execution_evidence,
        )
        continued = continued.with_execution_evidence(receipt)
    elif mutation == "aggregate":
        object.__setattr__(
            receipt, "source_preparation_work_units",
            receipt.source_preparation_work_units + jnp.uint64(1),
        )
    elif mutation == "imports":
        object.__setattr__(
            receipt, "imported_preparation_ids",
            receipt.imported_preparation_ids.at[-1, -1].set(1),
        )
    else:
        object.__setattr__(
            receipt, "consumer_evidence",
            NativeExecutionRecord(receipt.native_evidence(), owner_id=receipt.owner_id),
        )
    with pytest.raises(ValueError, match="counter|aggregate|padding|graph"):
        issue.complete(continued)
    assert issue._receipt_graph is None and issue._storage is None


@pytest.mark.parametrize("window", ("before", "prepared", "ended"))
@pytest.mark.parametrize("mutation", ("bytes", "owner", "container"))
def test_continuation_keeps_original_seal_across_every_mutation_window(
    source: CellMeshingResult, window: str, mutation: str,
) -> None:
    bank = np.array(source.mesh.coordinates)
    container = [bank]
    object.__setattr__(
        source.mesh, "coordinates", container if mutation == "container" else bank,
    )
    bank.flags.writeable = False
    bound, _ = _complete(source, combined=True)

    def mutate() -> None:
        if mutation == "bytes":
            bank.flags.writeable = True
            bank.flat[-1] += 0.125
            bank.flags.writeable = False
        elif mutation == "owner":
            object.__setattr__(bound.mesh, "coordinates", bank.copy())
        else:
            container[0] = bank.copy()

    with NativeExecutionScope(_limits()) as root:
        if window == "before":
            mutate()
        issue, overlap = _prepare_continuation(root, bound)
        if window == "prepared":
            mutate()
    continued = _bind_continuation(root, bound, overlap)
    if window == "ended":
        mutate()
    with pytest.raises(ValueError, match="bytes|owners|containers"):
        issue.complete(continued)
    assert id(continued) not in _completed_native_sources
    assert issue._seal is None and issue._storage is None


@pytest.mark.parametrize("after_prepare", (False, True))
@pytest.mark.parametrize("mutation", ("top", "constructor", "deleted", "dtype", "shape"))
def test_continuation_checks_cheap_identities_layout_and_liveness(
    source: CellMeshingResult, after_prepare: bool, mutation: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bank = (
        jnp.array(source.mesh.coordinates, copy=True)
        if mutation == "deleted" else np.array(source.mesh.coordinates)
    )
    object.__setattr__(source.mesh, "coordinates", bank)
    bound, _ = _complete(source, combined=True)

    def mutate() -> None:
        if mutation == "top":
            object.__setattr__(
                bound, "coordinate_contract",
                phx.SpatialCoordinateContract.si(),
            )
        elif mutation == "constructor":
            monkeypatch.setattr(type(bound), "__init__", lambda *args, **kwargs: None)
        elif mutation == "deleted":
            assert isinstance(bank, jax.Array)
            bank.delete()
        elif mutation == "dtype":
            assert isinstance(bank, np.ndarray)
            object.__setattr__(bank, "dtype", np.dtype(f"u{bank.dtype.itemsize}"))
        else:
            assert isinstance(bank, np.ndarray)
            object.__setattr__(bank, "shape", (bank.size,))

    with NativeExecutionScope(_limits()) as root:
        if not after_prepare:
            mutate()
            with pytest.raises(ValueError, match="owners|constructor|live|dtype|shape"):
                _prepare_continuation(root, bound)
            return
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    mutate()
    if mutation == "top":
        object.__setattr__(continued, "coordinate_contract", bound.coordinate_contract)
    with pytest.raises(ValueError, match="owners|constructor|live|dtype|shape"):
        issue.complete(continued)
    assert id(continued) not in _completed_native_sources


@pytest.mark.parametrize("authority", ("unpaid", "history", "clone", "receipt", "process"))
def test_continuation_refuses_unpaid_or_nonactual_authority(
    source: CellMeshingResult, authority: str,
) -> None:
    bound, previous = _complete(source, combined=True)
    receipt = bound.execution_evidence
    assert receipt is not None
    candidate = bound
    if authority == "clone":
        candidate = bound.with_execution_evidence(receipt)
    elif authority == "receipt":
        object.__setattr__(
            bound, "execution_evidence",
            NativeExecutionRecord(receipt.native_evidence(), owner_id=receipt.owner_id),
        )
    elif authority == "process":
        previous._process_id = -1
    with NativeExecutionScope(_limits()) as root:
        ledger = native_preparation_ledger(root)
        if authority == "history":
            ledger.record_history(root, receipt, completed_source=bound)
        elif authority != "unpaid":
            ledger.import_receipt(root, receipt)
        with pytest.raises(ValueError, match="paid|issued"):
            continue_native_completed_source_issue(
                root, candidate, receipt, overlap=ledger.overlap(root, receipt),
            )
    assert previous._source is not None and previous._source() is bound
    previous.close()


def test_continuation_refuses_foreign_overlap_and_root(source: CellMeshingResult) -> None:
    bound, _ = _complete(source, combined=True)
    receipt = bound.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(_limits()) as root:
        ledger = native_preparation_ledger(root)
        ledger.import_receipt(root, receipt)
        with pytest.raises(ValueError, match="overlap"):
            continue_native_completed_source_issue(
                root, bound, receipt, overlap=NativePreparationOverlap(),
            )
        overlap = ledger.overlap(root, receipt)
        with NativeExecutionScope(_limits()) as child:
            with pytest.raises(ValueError, match="root"):
                continue_native_completed_source_issue(
                    child, bound, receipt, overlap=overlap,
                )


@pytest.mark.parametrize("mutation", ("replacement", "bytes", "preparation"))
def test_continuation_rechecks_actual_receipt_after_prepare(
    source: CellMeshingResult, mutation: str,
) -> None:
    bound, _ = _complete(source, combined=True)
    receipt = bound.execution_evidence
    assert receipt is not None
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    replacement = NativeExecutionRecord(
        receipt.native_evidence(), owner_id="different-receipt",
    )
    if mutation == "replacement":
        object.__setattr__(bound, "execution_evidence", replacement)
    elif mutation == "bytes":
        object.__setattr__(receipt, "owner_id", replacement.owner_id)
        object.__setattr__(receipt, "receipt_id", replacement.receipt_id)
        receipt.require_valid()
    else:
        assert continued.execution_evidence is not None
        object.__setattr__(continued.execution_evidence, "preparation_evidence", replacement)
    with pytest.raises(ValueError, match="issued|receipt"):
        issue.complete(continued)
    assert id(continued) not in _completed_native_sources


def test_continuation_weak_gc_and_failed_root_cleanup(source: CellMeshingResult) -> None:
    bound, previous = _complete(source, combined=True)
    with NativeExecutionScope(_limits()) as root:
        issue, overlap = _prepare_continuation(root, bound)
    continued = _bind_continuation(root, bound, overlap)
    issue.complete(continued)
    identity, reference = id(bound), weakref.ref(bound)
    del bound
    gc.collect()
    assert reference() is None and identity not in _completed_native_sources
    assert previous._seal is None
    assert issue._continuation is None
    with pytest.raises(RuntimeError, match="failed consumer root"):
        with NativeExecutionScope(_limits()) as failed:
            failed_issue, _ = _prepare_continuation(failed, continued)
            raise RuntimeError("failed consumer root")
    assert failed.completed_evidence().host_storage_live_bytes_upper == 0
    assert failed_issue._storage is not None and failed_issue._storage.bound == 0
    with pytest.raises((RuntimeError, ValueError)):
        failed_issue.complete(continued)
    failed_issue.close()
    identity, reference = id(continued), weakref.ref(continued)
    del continued
    gc.collect()
    assert reference() is None and identity not in _completed_native_sources
    assert issue._seal is None


def test_archive_decode_does_not_restore_live_publication_authority(
    source: CellMeshingResult,
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    bound, _ = _complete(source)
    archived = write_meshing_source_closure(tmp_path / "source.phx", bound)
    with NativeExecutionScope(_limits()) as cold:
        with _native_source_budget(cold, None) as budget:
            restored = read_meshing_source_closure(
                archived.path,
                expected_content_id=archived.content_id,
                budget=budget,
            )
        assert isinstance(restored, CellMeshingResult)
        assert restored.execution_evidence is not None
        with pytest.raises(ValueError, match="issued"):
            _require_native_completed_source(restored, restored.execution_evidence)
        # Decode independently revalidated the scientific owner. Publication of
        # that actual root is newly staged under this cold request, never decoded.
        native_preparation_ledger(cold).record_history(cold, restored.execution_evidence)
        stage_native_source_publication(cold, restored)
        issue = prepare_native_completed_source_issue(cold, restored)
    continued = restored.with_execution_evidence(
        NativeExecutionRecord(
            cold.completed_evidence(),
            owner_id=restored.result_id,
            history_evidence=(restored.execution_evidence,),
        )
    )
    issue.complete(continued)
    with NativeExecutionScope(_limits()):
        _require_native_completed_source(continued, continued.execution_evidence)
    with NativeExecutionScope(_limits()) as imported_cold:
        ledger = native_preparation_ledger(imported_cold)
        ledger.import_receipt(imported_cold, restored.execution_evidence)
        with pytest.raises(ValueError, match="actual issued weak owner"):
            continue_native_completed_source_issue(
                imported_cold, restored, restored.execution_evidence,
                overlap=ledger.overlap(imported_cold, restored.execution_evidence),
            )


def test_completed_history_preparation_cannot_be_reimported(
    source: CellMeshingResult,
) -> None:
    bound, _ = _complete(source)
    previous = bound.execution_evidence
    assert previous is not None
    with NativeExecutionScope(_limits()) as completed:
        pass
    history = NativeExecutionRecord(
        completed.completed_evidence(),
        owner_id=source.result_id,
        preparation_evidence=previous,
    )
    with NativeExecutionScope(_limits()) as current:
        ledger = native_preparation_ledger(current)
        ledger.record_history(current, history)
        with pytest.raises(ValueError, match="history"):
            ledger.import_receipt(current, previous)


def test_completion_refuses_foreign_coordinate_frame(source: CellMeshingResult) -> None:
    with NativeExecutionScope(_limits()) as root:
        stage_native_source_publication(root, source)
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    object.__setattr__(
        bound,
        "coordinate_contract",
        phx.SpatialCoordinateContract(
            source.coordinate_contract.length_unit,
            reference_frame="foreign",
        ),
    )
    with pytest.raises(ValueError, match="field owners"):
        issue.complete(bound)


def test_completed_issue_does_not_retain_request_scope(source: CellMeshingResult) -> None:
    with NativeExecutionScope(_limits()) as root:
        stage_native_source_publication(root, source)
        issue = prepare_native_completed_source_issue(root, source)
    bound = source.with_execution_evidence(
        NativeExecutionRecord(root.completed_evidence(), owner_id=source.result_id)
    )
    issue.complete(bound)
    reference = weakref.ref(root)
    del root
    gc.collect()
    assert reference() is None
    with NativeExecutionScope(_limits()):
        assert bound.execution_evidence is not None
        _require_native_completed_source(bound, bound.execution_evidence)
    issue.close()


@pytest.mark.parametrize("readonly", (False, True), ids=("mapping", "mapping-proxy"))
def test_shared_owner_predicate_detects_mapping_edge_mutation(readonly: bool) -> None:
    from types import MappingProxyType

    from phydrax.discretization._coordinate_operands import (
        _coordinate_input_owner_matches,
        require_coordinate_input_owners,
    )

    child = object()
    mapping = {"value": child}
    owner = MappingProxyType(mapping) if readonly else mapping
    record = (owner, type(owner), ("value",), (child,))
    assert _coordinate_input_owner_matches(*record)
    with NativeExecutionScope(_limits()) as root:
        with _native_source_budget(root, None) as budget:
            assert require_coordinate_input_owners((record,), budget)
            mapping["value"] = object()
            assert not _coordinate_input_owner_matches(*record)
            assert not require_coordinate_input_owners((record,), budget)


def test_shared_owner_predicate_detects_namedtuple_field_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections import namedtuple

    from phydrax.discretization._coordinate_operands import (
        _coordinate_input_owner_matches,
        require_coordinate_input_owners,
    )

    pair = namedtuple("pair", ("value",))
    child = object()
    owner = pair(child)
    record = (owner, type(owner), ("_fields", "value"), (pair._fields, child))
    assert _coordinate_input_owner_matches(*record)
    with NativeExecutionScope(_limits()) as root:
        with _native_source_budget(root, None) as budget:
            assert require_coordinate_input_owners((record,), budget)
            monkeypatch.setattr(pair, "_fields", tuple(list(pair._fields)))
            assert not _coordinate_input_owner_matches(*record)
            assert not require_coordinate_input_owners((record,), budget)


def test_shared_owner_predicate_detects_added_object_field() -> None:
    from types import SimpleNamespace

    from phydrax.discretization._coordinate_operands import (
        _coordinate_input_owner_matches,
        require_coordinate_input_owners,
    )

    child = object()
    owner = SimpleNamespace(value=child)
    record = (owner, type(owner), ("value",), (child,))
    assert _coordinate_input_owner_matches(*record)
    with NativeExecutionScope(_limits()) as root:
        with _native_source_budget(root, None) as budget:
            assert require_coordinate_input_owners((record,), budget)
            owner.other = child
            assert not _coordinate_input_owner_matches(*record)
            assert not require_coordinate_input_owners((record,), budget)


def test_shared_container_predicate_detects_sequence_edge_mutation() -> None:
    from phydrax.discretization._coordinate_operands import (
        _coordinate_input_container_matches,
        require_coordinate_input_containers,
    )

    child = object()
    owner = [child]
    children = (child,)
    assert _coordinate_input_container_matches(owner, children)
    with NativeExecutionScope(_limits()) as root:
        with _native_source_budget(root, None) as budget:
            assert require_coordinate_input_containers(((owner, children),), budget)
            owner[0] = object()
            assert not _coordinate_input_container_matches(owner, children)
            assert not require_coordinate_input_containers(((owner, children),), budget)


def _prefix_receipt(
    owner_id: str, preparation: NativeExecutionRecord | None = None,
) -> NativeExecutionRecord:
    with NativeExecutionScope(_limits()) as root:
        root.charge(work=7)
    return NativeExecutionRecord(
        root.completed_evidence(), owner_id=owner_id,
        preparation_evidence=preparation,
    )


def _prepare_changed_target(
    root: NativeExecutionScope,
    source: CellMeshingResult,
    preparation: NativeExecutionRecord,
    histories: tuple[NativeExecutionRecord, ...],
) -> tuple[NativeCompletedSourceIssue, NativeExecutionImports]:
    from phydrax.meshing._native_preparation import (
        _NativeReceiptSnapshot,
        _prepare_native_receipt_graph,
    )

    ledger = native_preparation_ledger(root)
    snapshots: list[_NativeReceiptSnapshot] = []
    graph = _prepare_native_receipt_graph(preparation, snapshots=snapshots)
    assert graph is not None
    ledger.retain_receipt_snapshot(root, snapshots[0])
    for history in histories:
        ledger.record_history(root, history)
    imports = NativeExecutionImports(root, preparation=preparation)
    return stage_native_completed_source_issue(root, source, imports=imports), imports


def _bind_changed_target(
    root: NativeExecutionScope,
    source: CellMeshingResult,
    imports: NativeExecutionImports,
    histories: tuple[NativeExecutionRecord, ...],
) -> CellMeshingResult:
    return source.with_execution_evidence(
        NativeExecutionRecord(
            imports.evidence(), owner_id=source.result_id, imports=imports,
            preparation_evidence=imports.preparation, history_evidence=histories,
        )
    )


@pytest.mark.parametrize("multiple", (False, True))
def test_changed_target_prefix_preserves_graph_ids_and_only_checks_new_phase(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch, multiple: bool,
) -> None:
    preparation = _prefix_receipt(source.result_id)
    child = _prefix_receipt(source.result_id)
    history = _prefix_receipt(source.result_id, child)
    histories = (history, _prefix_receipt(source.result_id, child)) if multiple else (history,)
    with NativeExecutionScope(_limits()) as root:
        issue, imports = _prepare_changed_target(root, source, preparation, histories)
        assert issue._prefix is not None
        assert len(issue._prefix.phases) == (4 if multiple else 3)
    bound = _bind_changed_target(root, source, imports, histories)
    receipt = bound.execution_evidence
    assert receipt is not None
    expected = canonical_fingerprint(receipt.to_record())
    receipt_id = receipt.receipt_hex()
    checked = []
    require_phase = NativeExecutionRecord._require_phase_valid

    def check_new(record: NativeExecutionRecord) -> None:
        assert record is receipt
        checked.append(record)
        require_phase(record)

    with monkeypatch.context() as patch:
        patch.setattr(NativeExecutionRecord, "_require_phase_valid", check_new)
        issue.complete(bound)
    assert checked == [receipt]
    assert issue._receipt_graph is not None
    assert issue._receipt_graph.graph_id == expected
    assert receipt.receipt_hex() == receipt_id
    assert bound.mesh is source.mesh
    assert bound.geometry is source.geometry
    assert bound.certification is source.certification
    assert bound.result_id == source.result_id
    issue.close()


def test_changed_target_prefix_net_saving_and_exact_work_boundary(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    from phydrax.meshing._native_preparation import NativePreparationLedger

    preparation = _prefix_receipt(source.result_id)
    histories = (_prefix_receipt(source.result_id, _prefix_receipt(source.result_id)),)
    with monkeypatch.context() as patch:
        patch.setattr(NativePreparationLedger, "retain_receipt_snapshot", lambda *args, **kwargs: None)
        with NativeExecutionScope(_limits()) as full:
            full_issue, _ = _prepare_changed_target(full, source, preparation, histories)
            assert full_issue._prefix is None
        full_issue.close()
    with NativeExecutionScope(_limits()) as measured:
        issue, imports = _prepare_changed_target(measured, source, preparation, histories)
    issue.complete(_bind_changed_target(measured, source, imports, histories))
    required = measured.completed_evidence().work_counters.construction_work
    full_work = full.completed_evidence().work_counters.construction_work
    request.node.user_properties.extend(
        (("full_work", full_work), ("prepared_work", required),
         ("net_work_saved", full_work - required))
    )
    assert full_work - required > 10_166
    issue.close()
    with NativeExecutionScope(replace(_limits(), maximum_work_units=required)) as exact:
        exact_issue, imports = _prepare_changed_target(exact, source, preparation, histories)
        assert exact.remaining().remaining_work_units == 0
    exact_issue.complete(_bind_changed_target(exact, source, imports, histories))
    exact_issue.close()
    with pytest.raises((MeshcoreError, CoordinateEnclosureResourceError)):
        with NativeExecutionScope(
            replace(_limits(), maximum_work_units=required - 1)
        ) as short:
            _prepare_changed_target(short, source, preparation, histories)
    assert short.completed_evidence().host_storage_live_bytes_upper == 0


@pytest.mark.parametrize("mutation", ("leaf", "counter", "clone", "history", "deleted"))
def test_changed_target_prefix_authenticates_all_old_phase_owners(
    source: CellMeshingResult, mutation: str,
) -> None:
    preparation = _prefix_receipt(source.result_id)
    child = _prefix_receipt(source.result_id)
    history = _prefix_receipt(source.result_id, child)
    histories = (history,)
    with NativeExecutionScope(_limits()) as root:
        issue, imports = _prepare_changed_target(root, source, preparation, histories)
    bound = _bind_changed_target(root, source, imports, histories)
    if mutation == "leaf":
        object.__setattr__(child, "elapsed_seconds", jnp.array(0.5))
    elif mutation == "counter":
        object.__setattr__(child.work_counters, "construction_work", jnp.array(99, dtype=jnp.uint64))
    elif mutation == "clone":
        clone = NativeExecutionRecord(child.native_evidence(), owner_id=child.owner_id)
        object.__setattr__(history, "preparation_evidence", clone)
    elif mutation == "history":
        object.__setattr__(history, "history_evidence", (preparation,))
    else:
        child.elapsed_seconds.delete()
    with pytest.raises(ValueError, match="owners changed"):
        issue.complete(bound)
    assert issue._storage is None


@pytest.mark.parametrize("mutation", ("clone", "reordered", "new-history", "new-counter"))
def test_changed_target_prefix_rejects_changed_new_root(
    source: CellMeshingResult, mutation: str,
) -> None:
    preparation = _prefix_receipt(source.result_id)
    histories = (_prefix_receipt(source.result_id), _prefix_receipt(source.result_id))
    with NativeExecutionScope(_limits()) as root:
        issue, imports = _prepare_changed_target(root, source, preparation, histories)
    bound = _bind_changed_target(root, source, imports, histories)
    receipt = bound.execution_evidence
    assert receipt is not None
    if mutation == "clone":
        clone = NativeExecutionRecord(
            preparation.native_evidence(), owner_id=preparation.owner_id
        )
        object.__setattr__(receipt, "preparation_evidence", clone)
    elif mutation == "reordered":
        object.__setattr__(receipt, "history_evidence", receipt.history_evidence[::-1])
    elif mutation == "new-history":
        object.__setattr__(receipt, "history_evidence", (*histories, preparation))
    else:
        object.__setattr__(receipt.work_counters, "construction_work", jnp.array(99, dtype=jnp.uint64))
    with pytest.raises(ValueError, match="preparation|prefix|receipt_id"):
        issue.complete(bound)
    assert issue._storage is None


def test_changed_target_prefix_unprepared_and_mutable_graphs_take_full_path(
    source: CellMeshingResult,
) -> None:
    from phydrax.meshing._measurements import NativeExecutionImports
    from phydrax.meshing._native_preparation import _prepare_native_receipt_graph

    preparation = _prefix_receipt(source.result_id)
    object.__setattr__(preparation, "elapsed_seconds", np.asarray(preparation.elapsed_seconds).copy())
    preparation.require_valid()
    assert _prepare_native_receipt_graph(preparation) is None
    with NativeExecutionScope(_limits()) as root:
        imports = NativeExecutionImports(root, preparation=preparation)
        issue = stage_native_completed_source_issue(root, source, imports=imports)
        assert issue._prefix is None
    bound = _bind_changed_target(root, source, imports, ())
    issue.complete(bound)
    issue.close()


def test_changed_target_multiple_imports_outside_prefix_keep_full_validation(
    source: CellMeshingResult,
) -> None:
    preparation = _prefix_receipt(source.result_id)
    extra = _prefix_receipt(source.result_id)
    with NativeExecutionScope(_limits()) as root:
        ledger = native_preparation_ledger(root)
        ledger.import_receipt(root, preparation)
        ledger.import_receipt(root, extra)
        issue, imports = _prepare_changed_target(root, source, preparation, ())
        assert len(imports.receipt_ids) == 2
        assert issue._prefix is None
    issue.complete(_bind_changed_target(root, source, imports, ()))
    issue.close()


@pytest.mark.parametrize("mutation", ("bytes", "owner"))
def test_changed_target_prefix_keeps_independent_full_target_seal(
    source: CellMeshingResult, mutation: str,
) -> None:
    coordinates = np.array(source.mesh.coordinates)
    object.__setattr__(source.mesh, "coordinates", coordinates)
    preparation = _prefix_receipt(source.result_id)
    histories = (_prefix_receipt(source.result_id),)
    with NativeExecutionScope(_limits()) as root:
        issue, imports = _prepare_changed_target(root, source, preparation, histories)
        assert issue._prefix is not None
    bound = _bind_changed_target(root, source, imports, histories)
    if mutation == "bytes":
        coordinates.flat[-1] += 0.25
    else:
        object.__setattr__(bound.mesh, "coordinates", coordinates.copy())
    with pytest.raises(ValueError, match="bytes changed|field owners changed"):
        issue.complete(bound)
    assert issue._storage is None


@pytest.mark.parametrize("outside", (False, True), ids=("live-history", "pre-root"))
@pytest.mark.parametrize("mutable", (False, True), ids=("prepared", "fallback"))
def test_receipt_snapshot_work_and_only_pre_root_elapsed_are_imported(
    source: CellMeshingResult, monkeypatch: pytest.MonkeyPatch,
    outside: bool, mutable: bool,
) -> None:
    from phydrax.meshing import _native_preparation as owner

    receipt = _prefix_receipt(source.result_id)
    if mutable:
        object.__setattr__(receipt, "elapsed_seconds", np.asarray(receipt.elapsed_seconds).copy())
    receipt.require_valid()
    snapshots: list[owner._NativeReceiptSnapshot] = []
    # The already-completed numerical validation is outside this measurement.
    clock = iter((10.0, 10.125))
    with monkeypatch.context() as patch:
        patch.setattr(owner, "monotonic", lambda: next(clock))
        graph = owner._prepare_native_receipt_graph(receipt, snapshots=snapshots)
    snapshot, = snapshots
    assert (graph is None) == mutable
    assert snapshot.seconds == 0.125
    assert snapshot.work > 0
    # Retaining a prepared graph separately meters its owner traversal; that
    # host work is not part of the imported snapshot work.
    retention = 0
    if not mutable:
        with NativeExecutionScope(_limits()) as measured:
            native_preparation_ledger(measured)._retain_storage(
                measured, (receipt,), snapshot.bytes_upper
            )
            retention = (
                measured.limits.maximum_work_units
                - measured.remaining().remaining_work_units
            )
    with NativeExecutionScope(_limits()) as root:
        ledger = native_preparation_ledger(root)
        ledger.retain_receipt_snapshot(
            root, snapshot, preparation_outside_scope=outside
        )
        if not mutable:
            # Reusing this same admitted snapshot does not import it again.
            ledger.retain_receipt_snapshot(
                root, snapshot, preparation_outside_scope=outside
            )
        assert root.remaining().remaining_work_units == (
            root.limits.maximum_work_units - snapshot.work - retention
        )
    evidence = root.completed_evidence()
    assert evidence.prior_elapsed_seconds == (0.125 if outside else 0.0)
    final = NativeExecutionRecord(evidence, owner_id=source.result_id)
    final_graph = owner._prepare_native_receipt_graph(final)
    assert final_graph is not None
    assert final_graph.graph_id == canonical_fingerprint(final.to_record())
    assert float(final.total_elapsed_seconds) == (
        evidence.elapsed_seconds + evidence.prior_elapsed_seconds
    )
