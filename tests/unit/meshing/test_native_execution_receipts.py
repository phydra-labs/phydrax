# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Ended-source execution records remain distinct from parent/child measurements."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import nullcontext
from dataclasses import dataclass, fields, replace
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.typing import NDArray

from phydrax import SpatialCoordinateContract
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax._meshcore import (
    meshcore_available,
    MeshcoreError,
    MeshcoreStatus,
    NativeExecutionEvidence,
    NativeExecutionLimits,
    NativeExecutionMemoryCounters,
    NativeExecutionScope,
    NativeExecutionWorkCounters,
)
from phydrax._resource_ticket import HostBatchTicket
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import (
    CellMeshingResult,
    CurveMeshingSpec,
    MeshingFailure,
    MeshingFailureCategory,
    MeshingLimits,
    NativeCurveSource,
    NativeMeshingProvider,
)
from phydrax.meshing._association import PlcAssociationTransfer
from phydrax.meshing._failures import CutStageError
from phydrax.meshing._hex_cut_generation import cut_stage
from phydrax.meshing._measurements import (
    _NATIVE_EXECUTION_RECORD_ARRAY_BYTES,
    NativeExecutionDeviceMemoryCounters,
    NativeExecutionDeviceWorkCounters,
    NativeExecutionImports,
    NativeExecutionRecord,
)
from phydrax.meshing._native_preparation import (
    native_preparation_ledger,
    NativePreparationOverlap,
    stage_native_completed_source_issue,
)
from phydrax.meshing._volume_generation import (
    native_volume_execution_budget,
)
from phydrax.units import METER


_NATIVE = pytest.mark.skipif(not meshcore_available(), reason="native meshcore required")


@dataclass(frozen=True)
class _RetainedPreparation:
    prepared_id: str
    values: NDArray[np.int64]


@_NATIVE
@pytest.mark.parametrize("native_backed", (False, True), ids=("numpy", "native-host"))
def test_live_preparation_retains_phase_storage_once_until_root_close(
    native_backed: bool,
) -> None:
    maximum_bytes = 96 * 1024
    limits = NativeExecutionLimits(2, 0, 0, maximum_bytes, 120.0)
    root = NativeExecutionScope(limits)
    with root:
        values = (
            root.allocate_host_array((8192,), np.int64)
            if native_backed
            else np.arange(8192, dtype=np.int64)
        )
        first = _RetainedPreparation("prepared-first", values)
        second = _RetainedPreparation("prepared-second", values)
        detached = _RetainedPreparation(first.prepared_id, values)
        ledger = native_preparation_ledger(root)
        initial_bytes = root.remaining().remaining_managed_bytes
        native_bytes = maximum_bytes - initial_bytes
        with NativeExecutionScope(limits) as phase:
            with (
                phase.batch(1, bytes_upper=initial_bytes) as ticket,
                ticket.native_target(),
            ):
                assert phase.remaining().remaining_managed_bytes == 0
                ledger.remember_live(phase, first)
                assert ledger.live_contains(first)
                assert not ledger.live_contains(detached)
        after_first = root.remaining()
        first_bytes = initial_bytes - after_first.remaining_managed_bytes
        if native_backed:
            assert 0 < first_bytes < values.nbytes
        else:
            assert values.nbytes < first_bytes < initial_bytes
        with NativeExecutionScope(limits) as phase:
            with (
                phase.batch(1, bytes_upper=after_first.remaining_managed_bytes) as ticket,
                ticket.native_target(),
            ):
                assert phase.remaining().remaining_managed_bytes == 0
                ledger.remember_live(phase, second)
        after_second = root.remaining()
        assert (
            0
            < after_first.remaining_managed_bytes - after_second.remaining_managed_bytes
            < values.nbytes
        )
        assert after_second.remaining_work_units == 0
        ledger.remember_live(root, first)
        ledger.remember_live(root, second)
        repeated = root.remaining()
        assert repeated.remaining_work_units == after_second.remaining_work_units
        assert repeated.remaining_managed_bytes == after_second.remaining_managed_bytes
        assert (
            repeated.remaining_geometry_queries == after_second.remaining_geometry_queries
        )
        assert repeated.remaining_cavity_cells == after_second.remaining_cavity_cells
        assert repeated.remaining_seconds <= after_second.remaining_seconds
        assert ledger.live_contains(first) and ledger.live_contains(second)
        assert not ledger.live_contains(detached)
    evidence = root.completed_evidence()
    assert evidence.status is MeshcoreStatus.OK
    assert evidence.work_counters.construction_work == 2
    assert evidence.host_storage_live_bytes_upper == 0
    assert evidence.memory_counters.live_bytes == native_bytes
    assert not ledger.live_contains(first) and not ledger.live_contains(second)


@pytest.fixture
def retained_preparation_byte_boundary() -> tuple[_RetainedPreparation, int]:
    prepared = _RetainedPreparation("prepared-boundary", np.arange(8192, dtype=np.int64))
    maximum_bytes = 96 * 1024
    limits = NativeExecutionLimits(1, 0, 0, maximum_bytes, 120.0)
    root = NativeExecutionScope(limits)
    with root:
        with NativeExecutionScope(limits) as phase:
            with (
                phase.batch(1, bytes_upper=maximum_bytes) as ticket,
                ticket.native_target(),
            ):
                native_preparation_ledger(phase).remember_live(phase, prepared)
        retained_bytes = maximum_bytes - root.remaining().remaining_managed_bytes
    evidence = root.completed_evidence()
    assert evidence.status is MeshcoreStatus.OK
    assert evidence.host_storage_live_bytes_upper == 0
    assert evidence.memory_counters.live_bytes == 0
    return prepared, retained_bytes


@_NATIVE
@pytest.mark.parametrize("shortfall", (0, 1), ids=("exact", "one-byte-short"))
def test_live_preparation_retained_byte_boundary(
    retained_preparation_byte_boundary: tuple[_RetainedPreparation, int],
    shortfall: int,
) -> None:
    prepared, retained_bytes = retained_preparation_byte_boundary
    maximum_bytes = retained_bytes - shortfall
    limits = NativeExecutionLimits(1, 0, 0, maximum_bytes, 120.0)
    root = NativeExecutionScope(limits)
    registered = False
    with pytest.raises(MeshcoreError) if shortfall else nullcontext(), root:
        ledger = native_preparation_ledger(root)
        with NativeExecutionScope(limits) as phase:
            with (
                phase.batch(1, bytes_upper=maximum_bytes) as ticket,
                ticket.native_target(),
            ):
                ledger.remember_live(phase, prepared)
                registered = ledger.live_contains(prepared)
        assert root.remaining().remaining_managed_bytes == 0
    evidence = root.completed_evidence()
    assert evidence.status is (
        MeshcoreStatus.CAPACITY_EXCEEDED if shortfall else MeshcoreStatus.OK
    )
    assert registered is (shortfall == 0)
    assert not ledger.live_contains(prepared)
    assert evidence.host_storage_live_bytes_upper == 0
    assert evidence.memory_counters.live_bytes == 0


@_NATIVE
def test_live_preparation_refuses_exhausted_parent_without_registration() -> None:
    prepared = _RetainedPreparation("prepared-exhausted", np.arange(16, dtype=np.int64))
    limits = NativeExecutionLimits(0, 0, 0, 8192, 120.0)
    root = NativeExecutionScope(limits)
    with pytest.raises(MeshcoreError) as refusal, root:
        ledger = native_preparation_ledger(root)
        with NativeExecutionScope(NativeExecutionLimits(1, 0, 0, 8192, 120.0)) as phase:
            ledger.remember_live(phase, prepared)
    assert refusal.value.status is MeshcoreStatus.CAPACITY_EXCEEDED
    evidence = root.completed_evidence()
    assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert evidence.work_counters.construction_work == 0
    assert evidence.work_counters.work_refusals == 1
    assert evidence.host_storage_live_bytes_upper == 0
    assert evidence.memory_counters.live_bytes == 0
    assert not ledger.live_contains(prepared)


def _counter_evidence(
    construction_work: int = 11, *, managed_peak: int | None = None
) -> NativeExecutionEvidence:
    return NativeExecutionEvidence(
        NativeExecutionWorkCounters(construction_work, 7, 5, 3, 2, 1),
        NativeExecutionMemoryCounters(1024, 16, 32, 64, 4, 1),
        0.25,
        MeshcoreStatus.CAPACITY_EXCEEDED,
        8,
        6,
        9,
        10,
        20,
        managed_peak,
        0.125,
    )


def test_native_execution_host_counters_own_all_six_documented_slots() -> None:
    work_slots = np.asarray((11, 7, 5, 3, 2, 1), dtype=np.uint64)
    memory_slots = np.asarray((1024, 16, 32, 64, 4, 1), dtype=np.uint64)
    work = NativeExecutionWorkCounters.from_slots(work_slots)
    memory = NativeExecutionMemoryCounters.from_slots(memory_slots)
    assert tuple(field.name for field in fields(work)) == (
        "construction_work",
        "source_queries",
        "peak_cavity_cells",
        "work_refusals",
        "query_refusals",
        "cavity_refusals",
    )
    assert tuple(field.name for field in fields(memory)) == (
        "limit_bytes",
        "live_bytes",
        "peak_bytes",
        "requested_bytes",
        "allocations",
        "refusals",
    )
    assert dict(work.items()) == {
        "construction_work": 11,
        "source_queries": 7,
        "peak_cavity_cells": 5,
        "work_refusals": 3,
        "query_refusals": 2,
        "cavity_refusals": 1,
    }
    assert dict(memory.items()) == {
        "limit_bytes": 1024,
        "live_bytes": 16,
        "peak_bytes": 32,
        "requested_bytes": 64,
        "allocations": 4,
        "refusals": 1,
    }
    work_slots[:] = 0
    memory_slots[:] = 0
    assert work.construction_work == 11 and memory.peak_bytes == 32
    assert not jax.tree_util.tree_leaves(work)
    assert not jax.tree_util.tree_leaves(memory)
    with pytest.raises(AttributeError):
        setattr(work, "construction_work", 12)


@pytest.mark.parametrize(
    "owner",
    (NativeExecutionWorkCounters, NativeExecutionMemoryCounters),
    ids=("execution-work", "native-memory"),
)
@pytest.mark.parametrize(
    "slots",
    (
        np.zeros((5,), dtype=np.uint64),
        np.zeros((7,), dtype=np.uint64),
        np.zeros((2, 3), dtype=np.uint64),
        np.zeros((6,), dtype=np.int64),
    ),
    ids=("missing-slot", "extra-slot", "wrong-rank", "wrong-dtype"),
)
def test_native_execution_counters_refuse_inadequate_slot_transport(
    owner: type[NativeExecutionWorkCounters] | type[NativeExecutionMemoryCounters],
    slots: np.ndarray,
) -> None:
    with pytest.raises(
        ValueError, match=r"\.slots must have shape \(6,\) and dtype uint64"
    ):
        owner.from_slots(slots)


def test_native_execution_counter_constructors_refuse_boolean_counts() -> None:
    with pytest.raises(TypeError, match="construction_work"):
        NativeExecutionWorkCounters(construction_work=True)
    with pytest.raises(TypeError, match="refusals"):
        NativeExecutionMemoryCounters(refusals=True)


@pytest.mark.parametrize("value", (-1, 1 << 64), ids=("negative", "overflow"))
def test_native_execution_counter_values_refuse_outside_uint64(value: int) -> None:
    with pytest.raises(ValueError, match="construction_work"):
        NativeExecutionWorkCounters(construction_work=value)
    with pytest.raises(ValueError, match="requested_bytes"):
        NativeExecutionMemoryCounters(requested_bytes=value)


def test_native_execution_record_owns_named_scalar_device_leaves_and_json() -> None:
    evidence = _counter_evidence()
    receipt = NativeExecutionRecord(evidence, owner_id="execution-counter-ledger")
    assert type(receipt.work_counters) is NativeExecutionDeviceWorkCounters
    assert type(receipt.memory_counters) is NativeExecutionDeviceMemoryCounters
    leaves = jax.tree_util.tree_leaves((receipt.work_counters, receipt.memory_counters))
    assert len(leaves) == 12
    assert all(value.shape == () and value.dtype == jnp.uint64 for value in leaves)
    restored = receipt.native_evidence()
    assert restored.work_counters.items() == evidence.work_counters.items()
    assert restored.memory_counters.items() == evidence.memory_counters.items()
    phase = receipt.to_record()["phases"][0]
    assert phase["work_counters"] == dict(evidence.work_counters.items())
    assert phase["memory_counters"] == dict(evidence.memory_counters.items())
    assert phase["total_work_units"] == 11
    assert phase["total_geometry_queries"] == 7
    assert phase["native_primitive_queries"] == 9
    assert phase["host_storage_peak_bytes_upper"] == 20
    assert not hasattr(receipt, "work") and not hasattr(receipt, "memory")
    assert not hasattr(restored, "work_evidence")
    assert not hasattr(restored, "memory_evidence")


@pytest.mark.parametrize("defect", ("shape", "dtype"))
def test_native_execution_record_refuses_inadequate_restored_counter_leaves(
    defect: str,
) -> None:
    receipt = NativeExecutionRecord(_counter_evidence())
    invalid = (
        jnp.zeros((1,), dtype=jnp.uint64)
        if defect == "shape"
        else jnp.asarray(1, dtype=jnp.int64)
    )
    corrupted = eqx.tree_at(
        lambda value: value.memory_counters.refusals, receipt, invalid
    )
    with pytest.raises(ValueError, match=r"refusals must be a scalar uint64 leaf"):
        corrupted.require_valid()


@pytest.mark.parametrize("managed_peak", (None, 80), ids=("unobserved", "observed"))
def test_native_execution_named_records_have_current_identity_recipes(
    managed_peak: int | None,
) -> None:
    evidence = _counter_evidence(managed_peak=managed_peak)
    receipt = NativeExecutionRecord(evidence, owner_id="execution-counter-ledger")
    repeated = NativeExecutionRecord(evidence, owner_id=receipt.owner_id)
    changed = NativeExecutionRecord(
        replace(evidence, managed_admission_peak_bytes_upper=81),
        owner_id=receipt.owner_id,
    )
    fingerprint = array_tree_fingerprint(receipt)
    assert fingerprint == array_tree_fingerprint(repeated)
    assert len(fingerprint["signature"]) == 26 + int(managed_peak is not None)
    assert fingerprint != array_tree_fingerprint(changed)
    assert receipt.receipt_hex() == repeated.receipt_hex()
    assert receipt.receipt_hex() != changed.receipt_hex()
    assert canonical_fingerprint(receipt.to_record()) != canonical_fingerprint(
        changed.to_record()
    )


@pytest.mark.parametrize("managed_peak", (None, 80), ids=("unobserved", "observed"))
def test_native_execution_named_record_archive_restores_shared_phase_graph(
    tmp_path: Path,
    managed_peak: int | None,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    previous = NativeExecutionRecord(
        _counter_evidence(managed_peak=managed_peak), owner_id="execution-counter-ledger"
    )
    receipt = NativeExecutionRecord(
        _counter_evidence(managed_peak=managed_peak),
        preparation_evidence=previous,
        consumer_evidence=previous,
        owner_id=previous.owner_id,
    )
    archive = write_meshing_source_closure(tmp_path / "execution-counters", receipt)
    restored = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id
    )
    assert isinstance(restored, NativeExecutionRecord)
    restored.require_valid()
    assert restored.preparation_evidence is restored.consumer_evidence
    assert restored.to_record() == receipt.to_record()
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(receipt)


@pytest.mark.parametrize("managed_peak", (None, 0, 80), ids=("missing", "zero", "observed"))
def test_managed_admission_receipt_roundtrip_and_owned_array_capacity(
    managed_peak: int | None,
) -> None:
    receipt = NativeExecutionRecord(_counter_evidence(managed_peak=managed_peak))
    leaf = receipt.managed_admission_peak_bytes_upper
    if managed_peak is None:
        assert leaf is None
        assert receipt.total_managed_admission_peak_bytes_upper is None
    else:
        assert leaf is not None and leaf.shape == () and leaf.dtype == jnp.uint64
        assert int(leaf) == managed_peak
        total = receipt.total_managed_admission_peak_bytes_upper
        assert total is not None and int(total) == managed_peak
    assert receipt.native_evidence().managed_admission_peak_bytes_upper == managed_peak
    assert receipt.to_record()["phases"][0]["managed_admission_peak_bytes_upper"] == (
        managed_peak
    )
    owned_bytes = sum(value.nbytes for value in jax.tree_util.tree_leaves(receipt))
    assert owned_bytes == _NATIVE_EXECUTION_RECORD_ARRAY_BYTES - (
        np.dtype(np.uint64).itemsize if managed_peak is None else 0
    )


@pytest.mark.parametrize("defect", ("shape", "dtype"))
def test_managed_admission_receipt_refuses_invalid_restored_leaf(defect: str) -> None:
    receipt = NativeExecutionRecord(_counter_evidence(managed_peak=80))
    invalid = (
        jnp.zeros((1,), dtype=jnp.uint64)
        if defect == "shape"
        else jnp.asarray(80, dtype=jnp.int64)
    )
    corrupted = eqx.tree_at(
        lambda value: value.managed_admission_peak_bytes_upper, receipt, invalid
    )
    with pytest.raises(
        ValueError, match=r"managed_admission_peak_bytes_upper must be a scalar uint64"
    ):
        corrupted.require_valid()


def test_managed_admission_receipt_identity_binds_changed_observation() -> None:
    receipt = NativeExecutionRecord(_counter_evidence(managed_peak=80))
    corrupted = eqx.tree_at(
        lambda value: value.managed_admission_peak_bytes_upper,
        receipt,
        jnp.asarray(81, dtype=jnp.uint64),
    )
    with pytest.raises(ValueError, match=r"receipt_id must bind"):
        corrupted.require_valid()


@pytest.mark.parametrize("missing", (None, "root", "preparation", "consumer"))
def test_managed_admission_total_requires_complete_same_request_graph(
    missing: str | None,
) -> None:
    preparation = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "preparation" else 100)
    )
    consumer = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "consumer" else 200),
        preparation_evidence=preparation,
    )
    # Shared preparation is visited once; completed historical requests do not
    # impose either their large peak or their missing observation on this request.
    historical = (
        NativeExecutionRecord(_counter_evidence(31, managed_peak=10_000)),
        NativeExecutionRecord(_counter_evidence(32)),
    )
    receipt = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "root" else 150),
        preparation_evidence=preparation,
        consumer_evidence=consumer,
        history_evidence=historical,
    )
    total = receipt.total_managed_admission_peak_bytes_upper
    if missing is None:
        assert total is not None and int(total) == 200
    else:
        assert total is None
    assert int(receipt.total_work_units) == 22
    assert int(receipt.total_geometry_queries) == 14
    assert len(receipt.history_evidence) == 2
    assert receipt.receipt_hex() != NativeExecutionRecord(
        receipt.native_evidence(),
        preparation_evidence=preparation,
        consumer_evidence=consumer,
    ).receipt_hex()


@pytest.mark.parametrize("managed_peak", (None, 80), ids=("missing", "observed"))
def test_preparation_normalization_preserves_managed_admission_observation(
    managed_peak: int | None,
) -> None:
    evidence = _counter_evidence(managed_peak=managed_peak)
    normalized = NativePreparationOverlap(
        work=1, geometry_queries=1, external_work=1, live_elapsed_seconds=0.125
    ).normalize(evidence)
    assert normalized.managed_admission_peak_bytes_upper == managed_peak
    assert normalized.memory_counters is evidence.memory_counters
    assert normalized.host_storage_peak_bytes_upper == evidence.host_storage_peak_bytes_upper
    assert normalized.work_counters.construction_work == 10


@pytest.mark.parametrize("historical_peak", (None, 10_000), ids=("missing", "observed"))
def test_managed_admission_archive_roundtrip_retains_history_only_as_provenance(
    tmp_path: Path, historical_peak: int | None,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    historical = NativeExecutionRecord(_counter_evidence(managed_peak=historical_peak))
    receipt = NativeExecutionRecord(
        _counter_evidence(managed_peak=80), history_evidence=(historical,)
    )
    archive = write_meshing_source_closure(tmp_path / "managed-history", receipt)
    restored = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id
    )
    assert isinstance(restored, NativeExecutionRecord)
    assert restored.to_record() == receipt.to_record()
    assert len(restored.history_evidence) == 1
    assert restored.history_evidence[0].native_evidence().managed_admission_peak_bytes_upper == (
        historical_peak
    )
    assert int(restored.total_managed_admission_peak_bytes_upper) == 80


@pytest.mark.parametrize("managed_peak", (None, 80), ids=("missing", "observed"))
@pytest.mark.parametrize("with_history", (False, True), ids=("empty-history", "history"))
def test_raw_execution_controls_decode_optional_managed_leaf_and_strict_history(
    managed_peak: int | None, with_history: bool,
) -> None:
    from phydrax._array_archive import (
        ArrayArchiveCorruptionError,
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
    )
    from phydrax._model._structure import (
        model_array_catalog,
        model_recipe_array_values,
        model_structure_recipe,
    )
    from phydrax.lifecycle._meshing_sources import (
        _ExecutionControlSelection,
        _restore_execution_controls,
        register_meshing_source_artifacts,
    )

    historical = NativeExecutionRecord(_counter_evidence())
    receipt = NativeExecutionRecord(
        _counter_evidence(managed_peak=managed_peak),
        history_evidence=(historical,) if with_history else (),
    )
    controls = (receipt, _plan_limits())
    register_meshing_source_artifacts()
    recipe = model_structure_recipe(controls)
    table, catalog = model_array_catalog(
        recipe, prefix="source-record", limits=DEFAULT_ARRAY_ARCHIVE_LIMITS
    )
    receipt_index, limit_index = table.nodes[-1]["items"]
    selection = _ExecutionControlSelection(
        table, catalog, receipt_index, limit_index, {}, None, receipt_index, (), ()
    )
    arrays = {
        name: np.asarray(value)
        for name, value in model_recipe_array_values(
            controls, recipe, prefix="source-record"
        ).items()
    }
    restored = _restore_execution_controls(selection, arrays)
    assert restored.record.to_record() == receipt.to_record()
    assert restored.record.native_evidence().managed_admission_peak_bytes_upper == (
        managed_peak
    )
    assert len(restored.record.history_evidence) == int(with_history)
    # The new optional field must not open a generic None escape for required
    # numerical observations.
    names = tuple(member.name for member in fields(NativeExecutionRecord))
    required_index = table.nodes[receipt_index]["items"][
        names.index("externally_charged_work")
    ]
    table.nodes[required_index] = {"kind": "literal", "value": None}
    with pytest.raises(ArrayArchiveCorruptionError, match="declared optional receipt data"):
        _restore_execution_controls(selection, arrays)


@_NATIVE
def test_native_execution_counter_values_do_not_retrace_jit_scan_carry() -> None:
    receipt = _actual_source_receipt()
    with NativeExecutionScope(_plan_limits().native_limits()) as scope:
        native_preparation_ledger(scope).import_receipt(scope, receipt)
        imports = NativeExecutionImports(scope)
    imported = NativeExecutionRecord(
        scope.completed_evidence(), owner_id="execution-counter-ledger", imports=imports
    )
    traces: list[None] = []

    @jax.jit
    def advance(receipt: NativeExecutionRecord) -> NativeExecutionRecord:
        traces.append(None)

        def step(
            carry: NativeExecutionRecord, increment: jax.Array
        ) -> tuple[NativeExecutionRecord, None]:
            return (
                eqx.tree_at(
                    lambda value: value.work_counters.construction_work,
                    carry,
                    carry.work_counters.construction_work + increment,
                ),
                None,
            )

        return jax.lax.scan(step, receipt, jnp.arange(3, dtype=jnp.uint64))[0]

    first = NativeExecutionRecord(
        _counter_evidence(11), owner_id="execution-counter-ledger"
    )
    second = NativeExecutionRecord(_counter_evidence(17), owner_id=first.owner_id)
    first_result, second_result, imported_result = (
        advance(first),
        advance(second),
        advance(imported),
    )
    assert int(np.asarray(first_result.work_counters.construction_work)) == 14
    assert int(np.asarray(second_result.work_counters.construction_work)) == 20
    assert int(np.asarray(imported_result.imported_preparation_count)) == 1
    assert int(np.asarray(first_result.imported_preparation_count)) == 0
    assert len(traces) == 1
    assert jax.tree_util.tree_structure(first_result) == jax.tree_util.tree_structure(
        first
    )


@_NATIVE
@pytest.mark.parametrize(
    ("upper", "refusal"),
    (
        ({"work_upper": 1}, "work_refusals"),
        ({"queries_upper": 1}, "query_refusals"),
        ({"cavity_upper": 1}, "cavity_refusals"),
    ),
    ids=("construction-work", "source-query", "cavity"),
)
def test_native_execution_refusals_retain_the_same_named_counter_records(
    upper: dict[str, int], refusal: str
) -> None:
    scope = NativeExecutionScope(NativeExecutionLimits(0, 0, 0, 1024**2, 120.0))
    with pytest.raises(MeshcoreError) as caught, scope:
        with scope.batch(
            upper.get("work_upper", 0),
            queries_upper=upper.get("queries_upper", 0),
            cavity_upper=upper.get("cavity_upper", 0),
        ):
            pytest.fail("an inadmissible batch dispatched")
    evidence = scope.completed_evidence()
    assert evidence.status is MeshcoreStatus.CAPACITY_EXCEEDED
    assert caught.value.work_counters is evidence.work_counters
    assert caught.value.memory_counters is evidence.memory_counters
    assert dict(evidence.work_counters.items())[refusal] == 1
    assert evidence.work_counters.construction_work == 0
    assert evidence.work_counters.source_queries == 0
    assert evidence.externally_charged_work == 0
    assert evidence.externally_charged_geometry_queries == 0


@_NATIVE
def test_native_execution_allocation_refusal_retains_named_memory_measurements() -> None:
    scope = NativeExecutionScope(NativeExecutionLimits(0, 0, 0, 0, 120.0))
    with pytest.raises(MeshcoreError) as caught, scope:
        scope.allocate_host_array((1,), np.uint8)
    evidence = scope.completed_evidence()
    memory = evidence.memory_counters
    assert caught.value.memory_counters is memory
    assert memory.limit_bytes == memory.live_bytes == memory.peak_bytes == 0
    assert memory.requested_bytes > 0 and memory.refusals > 0
    assert memory.allocations == 0


@_NATIVE
@pytest.mark.parametrize("remaining", [0, 1])
def test_plc_support_batch_uses_authentic_parent_query_allowance(
    monkeypatch: pytest.MonkeyPatch, remaining: int
) -> None:
    import phydrax.meshing._association as association

    triangles = np.asarray(((0, 1, 2),), dtype=np.int64)
    owner = SimpleNamespace(
        domain=SimpleNamespace(ambient_dimension=3, vertices=np.eye(3)),
        triangle_vertices=triangles,
        triangle_facets=np.asarray((7,)),
        _spend=lambda work, count: PlcAssociationTransfer._spend(None, work, count),  # ty: ignore[invalid-argument-type] -- borrowed method does not inspect self
    )
    dispatched: list[int] = []

    def support(
        points: np.ndarray, source: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dispatched.append(points.shape[0])
        return np.zeros(1), np.asarray((6,)), np.asarray((MeshcoreStatus.OK,))

    monkeypatch.setattr(association, "point_triangle_locations", support)
    parent = NativeExecutionScope(
        NativeExecutionLimits(100_000, 1, 10, 4 * 1024**2, 120.0)
    )
    child = NativeExecutionScope(
        NativeExecutionLimits(100_000, 100, 10, 4 * 1024**2, 120.0)
    )
    work = [100]
    if remaining:
        with parent, child, child.batch(0, queries_upper=1) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket, maximum_work_units=100_000, maximum_memory_bytes=4 * 1024**2
            )
            assert PlcAssociationTransfer.point_support(
                owner,  # ty: ignore[invalid-argument-type] -- minimal provider boundary owner
                2,
                7,
                np.zeros(3),
                work,
                budget=budget,
            )
    else:
        with pytest.raises(MeshcoreError) as refusal, parent:
            with parent.batch(0, queries_upper=1) as ticket:
                ticket.consume(geometry_queries=1)
            with child, child.batch(0, queries_upper=1) as ticket:
                budget = CoordinateEnclosureBudget(
                    ticket, maximum_work_units=100_000, maximum_memory_bytes=4 * 1024**2
                )
                PlcAssociationTransfer.point_support(
                    owner,  # ty: ignore[invalid-argument-type] -- minimal provider boundary owner
                    2,
                    7,
                    np.zeros(3),
                    work,
                    budget=budget,
                )
        assert refusal.value.batch_requested_upper == (0, 1, 0, 0)
        assert refusal.value.batch_refusal_evidence is not None
        assert tuple(refusal.value.batch_refusal_evidence[4:6]) == (0, 1)
        assert refusal.value.batch_refusal_evidence[1] == 0
    assert dispatched == ([1] if remaining else [])
    assert work == ([99] if remaining else [100])
    assert parent.evidence is not None and child.evidence is not None
    assert int(parent.evidence.work_counters.source_queries) == 1
    assert int(child.evidence.work_counters.source_queries) == remaining
    assert int(parent.evidence.work_counters.query_refusals) == 1 - remaining


def _actual_source_receipt(work: int = 3) -> NativeExecutionRecord:
    with NativeExecutionScope(
        NativeExecutionLimits(work, 2, 10, 4 * 1024**2, 120.0)
    ) as source:
        with source.batch(work, queries_upper=2) as ticket:
            ticket.consume(work, geometry_queries=2)
    assert source.evidence is not None
    return NativeExecutionRecord(source.evidence, owner_id="receipt-source")


@_NATIVE
def test_actual_source_receipt_imports_once_and_retains_child_evidence() -> None:
    receipt = _actual_source_receipt()
    limits = MeshingLimits(
        maximum_work_units=100_000,
        maximum_geometry_queries=3,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    with native_volume_execution_budget(limits) as parent:
        from phydrax._meshcore import current_native_host_workspace

        storage = current_native_host_workspace()
        assert storage is not None
        native_preparation_ledger(parent).import_receipt(parent, receipt)
        once = parent.remaining()
        native_preparation_ledger(parent).import_receipt(parent, receipt)
        twice = parent.remaining()
        assert twice.remaining_work_units == once.remaining_work_units
        assert twice.remaining_geometry_queries == once.remaining_geometry_queries == 1
        with native_volume_execution_budget(
            limits,
            source_work_units=3,
            source_geometry_queries=2,
        ) as child:
            with child.batch(2, queries_upper=1) as ticket:
                ticket.consume(2, geometry_queries=1)
        assert child.evidence is not None
        consumer = NativeExecutionRecord(
            child.evidence,
            preparation_evidence=receipt,
            owner_id=receipt.owner_id,
        )
        assert int(np.asarray(consumer.total_work_units)) == 5
        assert int(np.asarray(consumer.total_geometry_queries)) == 3
        assert consumer.preparation_evidence is receipt
        assert float(np.asarray(consumer.total_elapsed_seconds)) == (
            child.evidence.elapsed_seconds
            + float(np.asarray(receipt.total_elapsed_seconds))
        )
    assert parent.evidence is not None
    assert parent.evidence.status is MeshcoreStatus.OK
    assert parent.evidence.externally_charged_geometry_queries == 3
    assert parent.evidence.prior_elapsed_seconds == float(
        np.asarray(receipt.total_elapsed_seconds)
    )
    assert child.evidence.prior_elapsed_seconds == 0.0
    original = NativeExecutionRecord(
        parent.evidence,
        consumer_evidence=consumer,
        owner_id=receipt.owner_id,
    )
    assert original.consumer_evidence is consumer
    assert original.consumer_evidence.preparation_evidence is receipt
    assert int(np.asarray(original.total_work_units)) == int(
        parent.evidence.work_counters.construction_work
    )
    assert int(np.asarray(original.total_geometry_queries)) == 3
    assert float(np.asarray(original.total_elapsed_seconds)) == (
        parent.evidence.elapsed_seconds + parent.evidence.prior_elapsed_seconds
    )


@_NATIVE
def test_source_receipt_query_refusal_never_dispatches_or_imports_duration() -> None:
    receipt = _actual_source_receipt()
    dispatched: list[str] = []
    limits = MeshingLimits(
        maximum_work_units=100_000,
        maximum_geometry_queries=1,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(limits) as parent:
            from phydrax._meshcore import current_native_host_workspace

            storage = current_native_host_workspace()
            assert storage is not None
            native_preparation_ledger(parent).import_receipt(parent, receipt)
            dispatched.append("inadmissible")
    assert not dispatched
    assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert parent.evidence is not None
    assert parent.evidence.externally_charged_geometry_queries == 0
    assert parent.evidence.prior_elapsed_seconds == 0.0
    assert int(parent.evidence.work_counters.query_refusals) == 1


@_NATIVE
def test_negative_prior_duration_cannot_refund_original_allowance() -> None:
    with NativeExecutionScope(
        NativeExecutionLimits(3, 2, 10, 4 * 1024**2, 120.0)
    ) as execution:
        with pytest.raises(ValueError, match=r"NativeExecutionScope\.elapsed_seconds"):
            execution.import_preparation(work=3, geometry_queries=2, elapsed_seconds=-1.0)
        assert execution.remaining().remaining_work_units == 3
        assert execution.remaining().remaining_geometry_queries == 2
        with execution.batch(3, queries_upper=2) as ticket:
            ticket.consume(3, geometry_queries=2)
    assert execution.evidence is not None
    assert execution.evidence.prior_elapsed_seconds == 0.0
    assert execution.evidence.externally_charged_work == 3
    assert execution.evidence.externally_charged_geometry_queries == 2


@_NATIVE
def test_consumer_constructor_preserves_original_operation_ownership() -> None:
    receipt = _actual_source_receipt()
    with NativeExecutionScope(
        NativeExecutionLimits(0, 0, 10, 4 * 1024**2, 120.0)
    ) as consumer:
        with consumer.batch(0):
            pass
    assert consumer.evidence is not None
    with pytest.raises(ValueError, match=r"NativeExecutionRecord\.owner_id"):
        NativeExecutionRecord(
            consumer.evidence,
            consumer_evidence=receipt,
            owner_id="another-operation",
        )


@_NATIVE
@pytest.mark.parametrize("native_refusal", [False, True])
def test_volume_wrapper_retains_cut_stage_records(native_refusal: bool) -> None:

    limits = MeshingLimits(
        maximum_work_units=10,
        maximum_geometry_queries=10,
        maximum_scratch_bytes=4 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    receipts = []
    original = None
    prefix = None
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(limits) as root:
            try:
                with cut_stage("original_source_link", None, receipts, scope=root):
                    if native_refusal:
                        active = root
                        assert active is not None
                        with active.batch(11) as ticket:
                            ticket.consume(11)
                    else:
                        raise MeshingFailure(
                            MeshingFailureCategory.RESOURCE_EXHAUSTED,
                            f"cut_stage.source_link must fit the resource budget; got {MeshingFailureCategory.RESOURCE_EXHAUSTED.value!r}.",
                        )
            except CutStageError as error:
                original = error.cause
                prefix = error.stage_records
                raise
    assert original is not None and prefix is not None
    assert caught.value.__cause__ is original
    assert caught.value.cut_stage_records is prefix
    assert prefix[0] is receipts[0]
    assert prefix[0].stage == "original_source_link"
    assert prefix[0].execution is not None
    assert root.evidence is not None
    assert (
        root.evidence.work_counters.construction_work
        == prefix[0].execution.work_counters.construction_work
    )
    if native_refusal:
        assert isinstance(original, MeshcoreError)
        assert original.work_counters is not None
        assert root.evidence.work_counters.items() == original.work_counters.items()


def _circle_plan_inputs(
    limits: MeshingLimits,
) -> tuple[
    NativeMeshingProvider,
    NativeCurveSource,
    CurveMeshingSpec,
    SpatialCoordinateContract,
]:
    import phydrax as phx

    meshing = phx.meshing
    atlas = (
        phx.geometry.Circle(
            (0.0, 0.0),
            1.0,
            feature_id="receipt-circle",
        )
        .compile(coordinate_unit=METER)
        .boundary_atlas
    )
    source = meshing.NativeCurveSource(atlas, "receipt-revision")
    scope = meshing.MeshingScope(
        atlas.source_id,
        source.source_revision,
        meshing.MeshingEntityKind.GEOMETRY,
        1,
        "arcs",
        np.arange(4),
    )
    specification = meshing.CurveMeshingSpec(
        meshing.CellMeshingTarget(
            1,
            2,
            meshing.CellFamilyPolicy(required=("interval",)),
        ),
        scope,
        junctions=tuple(
            meshing.CurveJunction(f"j{i}", ((i, "end"), ((i + 1) % 4, "start")))
            for i in range(4)
        ),
        size_controls=(
            meshing.UniformSizeControl(
                scope,
                1.0,
                strength=meshing.SizeControlStrength.SOFT,
            ),
        ),
        limits=limits,
    )
    provider = meshing.NativeMeshingProvider(
        meshing.NativeMeshingOptions("curve_arc_length")
    )
    return provider, source, specification, phx.SpatialCoordinateContract.si()


def _plan_limits(work: int = 1_000_000) -> MeshingLimits:
    return MeshingLimits(
        maximum_work_units=work,
        maximum_geometry_queries=100_000,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )


@pytest.fixture
def reference_budget() -> Iterator[CoordinateEnclosureBudget]:
    limits = _plan_limits()
    with HostBatchTicket(
        limits.maximum_work_units,
        queries_upper=limits.maximum_geometry_queries,
        bytes_upper=limits.maximum_scratch_bytes,
    ) as ticket:
        budget = CoordinateEnclosureBudget(
            ticket,
            maximum_work_units=limits.maximum_work_units,
            maximum_memory_bytes=limits.maximum_scratch_bytes,
        )
        with budget.activate():
            yield budget


@_NATIVE
def test_active_plan_preparation_and_execution_retain_once_only_receipt() -> None:
    from phydrax._meshcore import current_native_host_workspace
    from phydrax.meshing.providers._native_curve import PreparedCurveNetwork

    provider, source, specification, contract = _circle_plan_inputs(_plan_limits())
    with native_volume_execution_budget(_plan_limits(2_000_000)) as root:
        plan = provider.plan(source, specification, coordinate_contract=contract)
        assert isinstance(plan.prepared, PreparedCurveNetwork)
        preparation = plan.preparation_evidence
        assert preparation is not None and preparation.owner_id == plan.plan_id
        assert int(np.asarray(preparation.work_counters.construction_work)) > 0
        storage = current_native_host_workspace()
        assert storage is not None
        before = root.remaining()
        native_preparation_ledger(root).import_receipt(root, preparation)
        after = root.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert after.remaining_geometry_queries == before.remaining_geometry_queries
        result = plan.execute()
    assert result.execution_evidence is not None and root.evidence is not None
    assert result.execution_evidence.preparation_evidence is preparation
    assert int(root.evidence.work_counters.construction_work) >= int(
        np.asarray(
            preparation.work_counters.construction_work
            + result.execution_evidence.work_counters.construction_work
        ),
    )


@_NATIVE
def test_active_plan_stricter_preparation_cap_refuses_before_parent_renewal() -> None:
    provider, source, specification, contract = _circle_plan_inputs(_plan_limits(1))
    with pytest.raises(MeshingFailure) as caught:
        with native_volume_execution_budget(_plan_limits()) as root:
            provider.plan(source, specification, coordinate_contract=contract)
    assert caught.value.category is MeshingFailureCategory.RESOURCE_EXHAUSTED
    assert root.evidence is not None
    assert int(root.evidence.work_counters.construction_work) <= 1
    assert int(root.evidence.work_counters.work_refusals) == 1


@_NATIVE
def test_retained_active_plan_carries_authentic_preparation_across_roots() -> None:
    provider, source, specification, contract = _circle_plan_inputs(_plan_limits())
    with native_volume_execution_budget(_plan_limits()) as original:
        plan = provider.plan(source, specification, coordinate_contract=contract)
    preparation = plan.preparation_evidence
    assert original.evidence is not None and preparation is not None
    original_work = preparation.native_evidence().work_counters.items()
    with native_volume_execution_budget(_plan_limits(2_000_000)) as consumer:
        result = plan.execute()
    assert result.execution_evidence is not None and consumer.evidence is not None
    assert result.execution_evidence.preparation_evidence is preparation
    assert preparation.native_evidence().work_counters.items() == original_work
    assert int(consumer.evidence.work_counters.construction_work) >= int(
        np.asarray(result.execution_evidence.total_work_units),
    )


@pytest.mark.parametrize("dimension", [1, 2, 3])
def test_affine_simplex_bernstein_preserves_exact_vertex_order(
    dimension: int, reference_budget: CoordinateEnclosureBudget
) -> None:
    from fractions import Fraction

    from phydrax.discretization._coordinate_enclosure import (
        bernstein_coefficients,
        Polynomial,
    )

    origin = Fraction(2, 7)
    polynomial: Polynomial = {(0,) * dimension: origin}
    increments = tuple(Fraction(axis + 1, 11) for axis in range(dimension))
    for axis, increment in enumerate(increments):
        polynomial[tuple(int(index == axis) for index in range(dimension))] = increment
    assert bernstein_coefficients(
        polynomial, "simplex", dimension, budget=reference_budget
    ) == (origin, *(origin + increment for increment in reversed(increments)))


@_NATIVE
def test_retained_host_owner_reuses_complete_admitted_traversal() -> None:
    limits = _plan_limits()
    owner = (np.arange(16, dtype=np.int64), ("source", 3))
    with native_volume_execution_budget(limits) as root:
        from phydrax._meshcore import current_native_host_workspace

        storage = current_native_host_workspace()
        assert storage is not None
        storage.retain_owner(owner)
        before = root.remaining()
        bound = storage.bound
        storage.retain_owner(owner)
        after = root.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert storage.bound == bound


@_NATIVE
def test_exact_triangle_preparation_preserves_orientation_rank_and_capacity(
    reference_budget: CoordinateEnclosureBudget,
) -> None:
    from fractions import Fraction

    from phydrax.geometry._polygon_triangulation import loop_triangles

    points = np.asarray(
        (
            (Fraction(0), Fraction(0), Fraction(0)),
            (Fraction(1, 3), Fraction(0), Fraction(0)),
            (Fraction(0), Fraction(1, 7), Fraction(0)),
            (Fraction(2, 3), Fraction(0), Fraction(0)),
        ),
        dtype=object,
    )
    loops = np.asarray(((0, 2, 1), (0, 1, 3)), dtype=np.int64)
    triangles, owners, failed, exhausted = loop_triangles(
        points, loops, 54, budget=reference_budget
    )
    np.testing.assert_array_equal(triangles, loops[:1])
    np.testing.assert_array_equal(owners, (0,))
    np.testing.assert_array_equal(failed, (False, True))
    assert not np.any(exhausted)
    triangles, owners, failed, exhausted = loop_triangles(
        points, loops, 26, budget=reference_budget
    )
    assert not triangles.size and not owners.size and not np.any(failed)
    assert np.all(exhausted)


@_NATIVE
@pytest.mark.parametrize("imported", [False, True])
def test_matched_preparation_overlap_normalizes_additive_counts_once(
    imported: bool,
) -> None:
    limits = _plan_limits()
    receipt = _actual_source_receipt() if imported else None
    with NativeExecutionScope(limits.native_limits()) as parent:
        ledger = native_preparation_ledger(parent)
        if imported:
            assert receipt is not None
            ledger.import_receipt(parent, receipt)
        else:
            with NativeExecutionScope(
                NativeExecutionLimits(3, 2, 10, 4 * 1024**2, 120.0)
            ) as child:
                with child.batch(3, queries_upper=2) as ticket:
                    ticket.consume(3, geometry_queries=2)
            assert child.evidence is not None
            receipt = NativeExecutionRecord(
                child.evidence, owner_id="live-receipt-source"
            )
            ledger.remember_receipt(parent, receipt)
        overlap = ledger.overlap(parent, receipt)
    assert parent.evidence is not None and receipt is not None
    raw = parent.evidence
    assert int(raw.work_counters.construction_work) == 4
    assert int(raw.work_counters.source_queries) == 2
    normalized = overlap.normalize(raw)
    assert int(normalized.work_counters.construction_work) == 1
    assert int(normalized.work_counters.source_queries) == 0
    assert normalized.externally_charged_work == 1
    assert normalized.externally_charged_geometry_queries == 0
    assert normalized.prior_elapsed_seconds == 0.0
    assert (
        normalized.work_counters.peak_cavity_cells == raw.work_counters.peak_cavity_cells
    )
    assert normalized.work_counters.work_refusals == raw.work_counters.work_refusals
    assert normalized.work_counters.query_refusals == raw.work_counters.query_refusals
    assert normalized.work_counters.cavity_refusals == raw.work_counters.cavity_refusals
    assert normalized.memory_counters is raw.memory_counters
    record = NativeExecutionRecord(
        normalized, preparation_evidence=receipt, owner_id=receipt.owner_id
    )
    assert record.preparation_evidence is receipt
    assert int(np.asarray(record.total_work_units)) == int(
        raw.work_counters.construction_work
    )
    assert int(np.asarray(record.total_geometry_queries)) == int(
        raw.work_counters.source_queries
    )
    assert float(np.asarray(record.total_elapsed_seconds)) == (
        normalized.elapsed_seconds + float(np.asarray(receipt.total_elapsed_seconds))
    )


def _issued_consumer_source(source: CellMeshingResult) -> CellMeshingResult:
    with NativeExecutionScope(_plan_limits(100_000).native_limits()) as root:
        with root.batch(3, queries_upper=2) as ticket:
            ticket.consume(3, geometry_queries=2)
        issue = stage_native_completed_source_issue(
            root, source, owner_id="receipt-source",
        )
    receipt = NativeExecutionRecord(
        root.completed_evidence(), owner_id="receipt-source",
    )
    bound = source.with_execution_evidence(receipt)
    issue.complete(bound)
    return bound


@_NATIVE
@pytest.mark.parametrize("continuations", (1, 2))
def test_native_consumers_count_each_ended_preparation_once(continuations: int) -> None:
    import phydrax as phx
    from phydrax._meshcore import current_native_execution_scope
    from phydrax.meshing.qualification import execute_native_consumer

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    source = phx.meshing.certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    result = _issued_consumer_source(source)
    receipt = result.execution_evidence
    assert receipt is not None
    limits = MeshingLimits(
        maximum_work_units=100_000,
        maximum_geometry_queries=2 + continuations,
        maximum_scratch_bytes=32 * 1024**2,
        maximum_wall_seconds=120.0,
    )
    scopes = []

    def consume(budget: phx.discretization.CoordinateEnclosureBudget) -> int:
        scope = current_native_execution_scope()
        assert scope is not None
        assert budget.ticket.native_accounting
        scopes.append(scope)
        with scope.batch(5, queries_upper=1) as ticket:
            ticket.consume(5, geometry_queries=1)
        return 7

    def failure(operation: str, message: str, details: dict[str, object]) -> Exception:
        return RuntimeError(
            "NativeConsumer.execution must retain ended native evidence; "
            f"got {(operation, message, details)!r}"
        )

    expected_work = int(np.asarray(receipt.total_work_units))
    for index in range(1, continuations + 1):
        previous = result.execution_evidence
        consumed = execute_native_consumer(
            result, limits, consume, failure_factory=failure
        )
        result = consumed.result
        current = result.execution_evidence
        assert current is not None
        own_work = int(np.asarray(current.work_counters.construction_work))
        assert own_work >= 5 + index
        expected_work += own_work
        # Publication authentication is paid work too. The actual live root
        # independently reports the complete request, including each imported
        # predecessor exactly once, not just the five scientific units.
        assert expected_work == (
            scopes[-1].completed_evidence().work_counters.construction_work
        )
        assert consumed.value == 7
        assert current.preparation_evidence is previous
        assert int(np.asarray(current.total_work_units)) == expected_work
        assert int(np.asarray(current.total_geometry_queries)) == 2 + index
        assert result.mesh.mesh_id == source.mesh.mesh_id
        assert result.certification is source.certification


@_NATIVE
@pytest.mark.parametrize("slack", (0, -1), ids=("R", "R-1"))
def test_native_geometry_consumer_uses_the_producing_request_budget(
    monkeypatch: pytest.MonkeyPatch, slack: int
) -> None:
    import phydrax as phx
    import phydrax.geometry._embedding_state as embedding_state
    from phydrax._meshcore import MeshcoreError
    from phydrax.meshing.qualification import execute_native_consumer

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    source = phx.meshing.certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    result = _issued_consumer_source(source)
    initial = result.execution_evidence
    assert initial is not None
    oracle_work = []

    def no_fresh_host_ticket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Native geometry consumer must use its explicit native budget")

    monkeypatch.setattr(embedding_state, "HostBatchTicket", no_fresh_host_ticket)

    def consume(budget: phx.discretization.CoordinateEnclosureBudget) -> str:
        assert budget.ticket.native_accounting
        certificate = phx.geometry.certify_global_embedding(
            result.mesh, result.geometry, result.audit.validity, budget=budget
        )
        assert certificate.status == "certified"
        oracle_work.append(budget.work_units)
        return certificate.certificate_id

    def failure(operation: str, message: str, details: dict[str, object]) -> Exception:
        return RuntimeError(f"{operation}: {message}; {details!r}")

    def limits(work: int) -> MeshingLimits:
        return MeshingLimits(
            maximum_work_units=work,
            maximum_geometry_queries=100_000,
            maximum_scratch_bytes=32 * 1024**2,
            maximum_wall_seconds=120.0,
        )

    measured = execute_native_consumer(
        result, limits(100_000), consume, failure_factory=failure
    )
    receipt = measured.result.execution_evidence
    assert receipt is not None
    required = int(np.asarray(receipt.total_work_units))
    assert oracle_work[-1] > 0
    assert required >= int(np.asarray(initial.total_work_units)) + oracle_work[-1]
    if slack < 0:
        with pytest.raises(MeshcoreError):
            execute_native_consumer(
                result, limits(required + slack), consume, failure_factory=failure
            )
    else:
        repeated = execute_native_consumer(
            result, limits(required), consume, failure_factory=failure
        )
        assert repeated.value == measured.value
        assert repeated.result.execution_evidence is not None
        assert int(np.asarray(repeated.result.execution_evidence.total_work_units)) == required


@_NATIVE
def test_retained_coefficient_profile_is_charged_once_across_native_tickets() -> None:
    coefficient = Fraction(3, 7)
    graph = (coefficient, coefficient)
    root = NativeExecutionScope(NativeExecutionLimits(4, 0, 0, 8192, 120.0))
    with root:
        first = root.batch(4, bytes_upper=8192)
        ledger = CoordinateEnclosureBudget(
            first, maximum_work_units=4, maximum_memory_bytes=8192
        )
        with first, ledger.activate():
            ledger.retain_basis(graph)
            ledger.retain_basis(graph)
            assert ledger.work_units == len(ledger.retained_objects) == 4
        assert first.settled and first.work_units == 4
        retained = ledger.retained_basis_bytes
        second = root.batch(0, bytes_upper=8192)
        with ledger.bind_ticket(second):
            before = root.remaining().remaining_work_units
            ledger.retain_basis(graph)
            assert root.remaining().remaining_work_units == before == 0
            assert ledger.retained_basis_bytes == retained
            assert ledger.retained_objects[id(graph)] is graph
        assert second.settled and second.work_units == 0
        assert ledger.native_charged_work_units == 4
    assert root.evidence is not None
    counters = root.evidence.work_counters
    assert counters.construction_work == 4
    assert counters.source_queries == counters.work_refusals == 0


@_NATIVE
def test_cross_operation_consumer_binds_its_exact_imported_preparation(
    tmp_path: Path,
) -> None:
    from phydrax.lifecycle._meshing_sources import (
        read_meshing_source_closure,
        write_meshing_source_closure,
    )

    source = _actual_source_receipt()
    limits = _plan_limits()
    with NativeExecutionScope(limits.native_limits()) as parent:
        ledger = native_preparation_ledger(parent)
        ledger.import_receipt(parent, source)
        overlap = ledger.overlap(parent, source)
        with NativeExecutionScope(limits.native_limits()) as child:
            with child.batch(5, queries_upper=1) as ticket:
                ticket.consume(5, geometry_queries=1)
            imports = NativeExecutionImports(child, preparation=source)
        consumer = NativeExecutionRecord(
            child.completed_evidence(),
            owner_id="adaptation-operation",
            imports=imports,
            preparation_evidence=source,
        )
    bound = NativeExecutionRecord(
        overlap.normalize(parent.completed_evidence()),
        owner_id=source.owner_id,
        preparation_evidence=source,
        consumer_evidence=consumer,
    )
    bound.require_valid()
    assert consumer.imported_preparation_hex() == (source.receipt_hex(),)
    assert bound.preparation_evidence is source
    assert bound.consumer_evidence is consumer
    assert int(np.asarray(consumer.total_work_units)) == 8
    assert int(np.asarray(bound.total_work_units)) == 9
    assert int(np.asarray(bound.total_geometry_queries)) == 3
    archive = write_meshing_source_closure(tmp_path / "imported-receipt", bound)
    restored = read_meshing_source_closure(
        archive.path, expected_content_id=archive.content_id
    )
    assert isinstance(restored, NativeExecutionRecord)
    restored.require_valid()
    assert restored.to_record() == bound.to_record()
    assert array_tree_fingerprint(restored) == array_tree_fingerprint(bound)


@_NATIVE
@pytest.mark.parametrize("imported", (False, True), ids=("unimported", "wrong-receipt"))
def test_cross_operation_consumer_refuses_unrelated_preparation(imported: bool) -> None:
    source = _actual_source_receipt()
    other = _actual_source_receipt(4)
    with NativeExecutionScope(_plan_limits().native_limits()) as scope:
        if imported:
            native_preparation_ledger(scope).import_receipt(scope, other)
        imports = NativeExecutionImports(scope)
        chained = NativeExecutionImports(scope, preparation=source)
    consumer = NativeExecutionRecord(
        scope.completed_evidence(), owner_id="adaptation-operation", imports=imports
    )
    assert source.owner_id == other.owner_id
    assert source.receipt_hex() != other.receipt_hex()
    with pytest.raises(ValueError, match=r"NativeExecutionRecord.owner_id must match"):
        NativeExecutionRecord(
            _counter_evidence(),
            owner_id=source.owner_id,
            preparation_evidence=source,
            consumer_evidence=consumer,
        )
    with pytest.raises(ValueError, match=r"NativeExecutionRecord.owner_id must match"):
        NativeExecutionRecord(
            scope.completed_evidence(),
            owner_id="adaptation-operation",
            preparation_evidence=source,
            imports=chained,
        )


@_NATIVE
def test_preparation_import_witness_refuses_another_scopes_evidence() -> None:
    source = _actual_source_receipt()
    with NativeExecutionScope(_plan_limits().native_limits()) as importing:
        native_preparation_ledger(importing).import_receipt(importing, source)
        imports = NativeExecutionImports(importing)
    with NativeExecutionScope(_plan_limits().native_limits()) as unimported:
        pass
    with pytest.raises(ValueError, match=r"NativeExecutionImports.evidence must be"):
        NativeExecutionRecord(
            unimported.completed_evidence(),
            owner_id="unimported-operation",
            imports=imports,
        )


@_NATIVE
def test_preparation_import_capacity_refuses_without_truncating() -> None:
    from phydrax.meshing._measurements import NATIVE_EXECUTION_IMPORT_CAPACITY

    receipts = tuple(
        _actual_source_receipt(work + 1)
        for work in range(NATIVE_EXECUTION_IMPORT_CAPACITY + 1)
    )
    with NativeExecutionScope(_plan_limits().native_limits()) as scope:
        for receipt in receipts:
            native_preparation_ledger(scope).import_receipt(scope, receipt)
        imports = NativeExecutionImports(scope)
    assert len(imports.receipt_ids) == NATIVE_EXECUTION_IMPORT_CAPACITY + 1
    with pytest.raises(ValueError, match=r"capacity 64; got 65"):
        NativeExecutionRecord(scope.completed_evidence(), imports=imports)


def test_restored_preparation_import_digest_refuses_forged_identity() -> None:
    record = NativeExecutionRecord(_counter_evidence())
    corrupted = eqx.tree_at(
        lambda value: value.receipt_id,
        record,
        jnp.zeros((32,), dtype=jnp.uint8),
    )
    with pytest.raises(ValueError, match=r"NativeExecutionRecord.receipt_id must bind"):
        corrupted.require_valid()


@_NATIVE
@pytest.mark.parametrize("managed_peak", (None, 80), ids=("missing", "observed"))
def test_managed_admission_leaf_is_owned_before_receipt_allocation(
    monkeypatch: pytest.MonkeyPatch,
    reference_budget: CoordinateEnclosureBudget,
    managed_peak: int | None,
) -> None:
    import phydrax as phx
    import phydrax.meshing.providers._native_publication as publication

    mesh = phx.discretization.CellMesh.from_triangles(
        np.asarray(((0.0, 0.0), (1.0, 0.0), (0.0, 1.0)), dtype=np.float64),
        np.asarray(((0, 1, 2),), dtype=np.int64),
    )
    result = phx.meshing.certify_cell_mesh(mesh, SpatialCoordinateContract.si())
    retained = publication.model_array_bytes((result, None, ()))
    expected = retained + _NATIVE_EXECUTION_RECORD_ARRAY_BYTES - (
        np.dtype(np.uint64).itemsize if managed_peak is None else 0
    )

    def refuse_before_allocation(retained_bytes: int, limits: MeshingLimits) -> None:
        assert retained_bytes == expected
        raise RuntimeError("receipt ownership refused")

    def unowned_allocation(*args: object, **kwargs: object) -> None:
        pytest.fail("receipt arrays allocated before their retained-byte admission")

    monkeypatch.setattr(publication, "_require_retained_bytes", refuse_before_allocation)
    monkeypatch.setattr(jax, "device_put", unowned_allocation)
    with pytest.raises(RuntimeError, match="receipt ownership refused"):
        publication.bind_native_execution_result(
            result, _counter_evidence(managed_peak=managed_peak), _plan_limits(), 0.0
        )


@pytest.mark.parametrize(
    ("aggregate", "value"),
    (
        ("source_preparation_work_units", 1),
        ("source_preparation_geometry_queries", 1),
        ("preparation_seconds", 0.125),
    ),
)
@pytest.mark.parametrize("nested", (False, True), ids=("root", "consumer"))
def test_managed_admission_total_requires_observation_of_aggregate_only_preparation(
    aggregate: str, value: int | float, nested: bool,
) -> None:
    phase = NativeExecutionRecord(
        _counter_evidence(managed_peak=80),
        source_preparation_work_units=int(value) if aggregate == "source_preparation_work_units" else None,
        source_preparation_geometry_queries=int(value) if aggregate == "source_preparation_geometry_queries" else None,
        preparation_seconds=float(value) if aggregate == "preparation_seconds" else None,
    )
    receipt = (
        NativeExecutionRecord(
            _counter_evidence(managed_peak=100), consumer_evidence=phase
        )
        if nested
        else phase
    )
    assert phase.native_evidence().managed_admission_peak_bytes_upper == 80
    assert phase.preparation_evidence is None
    assert phase.total_managed_admission_peak_bytes_upper is None
    assert receipt.total_managed_admission_peak_bytes_upper is None
    assert int(phase.total_work_units) == 11 + int(
        aggregate == "source_preparation_work_units"
    )
    assert int(phase.total_geometry_queries) == 7 + int(
        aggregate == "source_preparation_geometry_queries"
    )
    assert float(phase.total_elapsed_seconds) == 0.375 + (
        float(value) if aggregate == "preparation_seconds" else 0.0
    )


@_NATIVE
def test_imported_envelope_preparation_keeps_source_owner_and_new_plan_owner() -> None:
    import phydrax.meshing as meshing
    from phydrax.meshing.providers._native import NativeMeshingPlan
    from tests._support.envelope_meshing import (
        envelope_policy,
        envelope_sheet,
        envelope_specification,
    )

    envelope = meshing.wrap_surface_envelope(envelope_sheet(), envelope_policy())
    source = meshing.NativeSurfaceEnvelopeSource(envelope, "wrapped-solid")
    receipt = envelope.execution_evidence
    assert receipt is not None and receipt.owner_id == envelope.envelope_id
    specification = envelope_specification(source)
    provider = NativeMeshingProvider(
        meshing.NativeMeshingOptions(
            "surface_envelope_tetrahedral",
            volume_schedule=meshing.NativeVolumeSchedule(
                refinement_rounds=1,
                improvement_passes=1,
                minimum_dihedral_degrees=0.0,
            ),
        )
    )
    with native_volume_execution_budget(specification.limits) as root:
        plan = provider.plan(
            source, specification, coordinate_contract=SpatialCoordinateContract.si()
        )
        preparation = plan.preparation_evidence
        assert preparation is not None
        assert preparation.owner_id == plan.plan_id != receipt.owner_id
        assert preparation.preparation_evidence is receipt
        assert receipt.receipt_hex() in preparation.imported_preparation_hex()
        assert preparation.history_evidence == ()
        before = root.remaining()
        ledger = native_preparation_ledger(root)
        ledger.import_receipt(root, preparation)
        ledger.import_receipt(root, receipt)
        after = root.remaining()
        assert after.remaining_work_units == before.remaining_work_units
        assert after.remaining_geometry_queries == before.remaining_geometry_queries
    assert envelope.execution_evidence is receipt
    assert receipt.owner_id == envelope.envelope_id
    assert int(np.asarray(preparation.source_preparation_work_units)) == int(
        np.asarray(receipt.total_work_units)
    )
    assert int(np.asarray(preparation.source_preparation_geometry_queries)) == int(
        np.asarray(receipt.total_geometry_queries)
    )
    restored = NativeMeshingPlan(
        plan.source,
        plan.specification,
        plan.options,
        plan.support,
        plan.coordinate_contract,
        plan.prepared,
        preparation_evidence=preparation,
    )
    assert restored.plan_id == plan.plan_id
    assert restored.preparation_evidence is preparation
    with pytest.raises(ValueError, match="owner_id must bind this plan"):
        NativeMeshingPlan(
            plan.source,
            plan.specification,
            plan.options,
            plan.support,
            plan.coordinate_contract,
            plan.prepared,
            preparation_evidence=receipt,
        )


@pytest.mark.parametrize("missing", (None, "root", "preparation", "consumer"))
def test_json_receipt_validator_preserves_complete_same_request_observation(
    missing: str | None,
) -> None:
    from phydrax.meshing._measurements import validate_native_execution_interchange

    preparation = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "preparation" else 100),
        owner_id="same-request",
    )
    consumer = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "consumer" else 200),
        owner_id="same-request",
        preparation_evidence=preparation,
    )
    history = NativeExecutionRecord(
        _counter_evidence(31, managed_peak=None), owner_id="previous-request",
    )
    root = NativeExecutionRecord(
        _counter_evidence(managed_peak=None if missing == "root" else 150),
        owner_id="same-request",
        preparation_evidence=preparation,
        consumer_evidence=consumer,
        history_evidence=(history,),
    )
    observation = validate_native_execution_interchange(root.to_record())
    assert observation["root_receipt_id"] == root.receipt_hex()
    assert observation["owner_id"] == "same-request"
    assert observation["preparation_complete"] is True
    assert len(observation["current_phase_indexes"]) == 3
    assert observation["managed_admission_peak_bytes_upper"] == (
        200 if missing is None else None
    )


@pytest.mark.parametrize(
    "defect",
    (
        "missing-field", "unknown-field", "status", "boolean-counter",
        "counter-overflow", "digest", "imports", "aggregate", "foreign-owner",
        "cycle", "disconnected", "out-of-range-edge",
    ),
)
def test_json_receipt_validator_rejects_incomplete_or_foreign_graph(defect: str) -> None:
    from copy import deepcopy

    from phydrax.meshing._measurements import validate_native_execution_interchange

    preparation = NativeExecutionRecord(_counter_evidence(managed_peak=100), owner_id="request")
    root = NativeExecutionRecord(
        _counter_evidence(managed_peak=200), owner_id="request",
        preparation_evidence=preparation,
    )
    record = cast(dict[str, Any], root.to_record())
    phase = record["phases"][0]
    if defect == "missing-field":
        del phase["managed_admission_peak_bytes_upper"]
    elif defect == "unknown-field":
        phase["unrecognized_scope"] = True
    elif defect == "status":
        phase["status_name"] = "OK-but-unended"
    elif defect == "boolean-counter":
        phase["work_counters"]["construction_work"] = True
    elif defect == "counter-overflow":
        phase["memory_counters"]["peak_bytes"] = 1 << 64
    elif defect == "digest":
        phase["receipt_id"] = "0" * 64
    elif defect == "imports":
        phase["imported_preparation_ids"] = ["f" * 64, "0" * 64]
    elif defect == "aggregate":
        phase["source_preparation_work_units"] += 1
    elif defect == "foreign-owner":
        record["phases"][1]["owner_id"] = "foreign-request"
    elif defect == "cycle":
        phase["preparation_evidence"] = 0
    elif defect == "disconnected":
        record["phases"].append(deepcopy(record["phases"][1]))
    else:
        phase["consumer_evidence"] = len(record["phases"])
    with pytest.raises((TypeError, ValueError)):
        validate_native_execution_interchange(record)


def test_json_receipt_validator_does_not_fill_aggregate_only_preparation_from_history() -> None:
    from phydrax.meshing._measurements import validate_native_execution_interchange

    historical = NativeExecutionRecord(_counter_evidence(managed_peak=1_000))
    root = NativeExecutionRecord(
        _counter_evidence(managed_peak=100),
        source_preparation_work_units=11,
        source_preparation_geometry_queries=7,
        preparation_seconds=0.5,
        history_evidence=(historical,),
    )
    observation = validate_native_execution_interchange(root.to_record())
    assert observation["preparation_complete"] is False
    assert observation["managed_admission_peak_bytes_upper"] is None
