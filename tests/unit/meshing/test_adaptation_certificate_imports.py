#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native adaptation reuses source certificates and keeps their request as history."""

from contextlib import nullcontext
from typing import Any

import equinox as eqx
import numpy as np
import pytest

import phydrax as phx
from phydrax._meshcore import NativeExecutionScope
from phydrax.discretization._coordinate_budget import CoordinateEnclosureBudget
from phydrax.meshing import (
    BisectionCompatibility,
    execute_mesh_adaptation,
    MarkedMeshAdaptation,
    MeshAdaptationPolicy,
    MeshAdaptationRoute,
    MeshingLimits,
    prepare_mesh_adaptation,
)
from phydrax.meshing._measurements import NativeExecutionRecord
from phydrax.meshing._native_preparation import native_preparation_ledger
from phydrax.meshing._result import CellMeshingResult
from tests._support.cell_meshes import triangle_grid


def _source() -> tuple[CellMeshingResult, NativeExecutionRecord, MeshingLimits]:
    limits = MeshingLimits()
    with NativeExecutionScope(limits.native_limits()) as scope:
        with scope.batch(
            limits.maximum_work_units,
            queries_upper=limits.maximum_geometry_queries,
            bytes_upper=limits.maximum_scratch_bytes,
        ) as ticket:
            budget = CoordinateEnclosureBudget(
                ticket,
                maximum_work_units=limits.maximum_work_units,
                maximum_memory_bytes=limits.maximum_scratch_bytes,
            )
            with budget.activate(), ticket.native_target():
                source = phx.meshing.certify_cell_mesh(
                    triangle_grid(4, 4), phx.SpatialCoordinateContract.si(), budget=budget
                )
    record = NativeExecutionRecord(scope.completed_evidence(), owner_id=source.result_id)
    return source, record, limits


def test_native_adaptation_keeps_reused_certificate_source_as_history() -> None:
    source, receipt, limits = _source()
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
        limits=limits,
    )
    request = MarkedMeshAdaptation(np.asarray([0], dtype=np.int64))
    full = execute_mesh_adaptation(prepare_mesh_adaptation(source, request, policy=policy))
    inherited = execute_mesh_adaptation(
        prepare_mesh_adaptation(
            source.with_execution_evidence(receipt), request, policy=policy
        )
    )
    assert inherited.target.result_id == full.target.result_id
    assert (
        inherited.target.audit.validity.certificate_id
        == full.target.audit.validity.certificate_id
    )
    np.testing.assert_array_equal(
        inherited.target.audit.validity.status, full.target.audit.validity.status
    )
    evidence = inherited.target.execution_evidence
    assert evidence is not None
    # The completed source request is provenance, never a charge of this request.
    assert evidence.history_evidence == (receipt,)
    assert receipt.receipt_hex() not in evidence.imported_preparation_hex()
    with NativeExecutionScope(limits.native_limits()) as charged:
        native_preparation_ledger(charged).import_receipt(charged, receipt)
        with pytest.raises(ValueError, match="already charged to this request"):
            execute_mesh_adaptation(
                prepare_mesh_adaptation(
                    source.with_execution_evidence(receipt), request, policy=policy
                )
            )


def test_native_adaptation_refuses_tampered_imported_certificate_payload() -> None:
    source, receipt, limits = _source()
    prepared = prepare_mesh_adaptation(
        source.with_execution_evidence(receipt),
        MarkedMeshAdaptation(np.asarray([0], dtype=np.int64)),
        policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_BISECTION,
            compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            limits=limits,
        ),
    )
    changed = eqx.tree_at(
        lambda value: value.source.audit.validity.determinant_lower,
        prepared,
        prepared.source.audit.validity.determinant_lower + 1.0,
    )
    with pytest.raises(ValueError, match="authenticate.*source certificate payload"):
        execute_mesh_adaptation(changed)


@pytest.mark.parametrize("nested", (False, True), ids=("root", "nested"))
def test_successful_adaptation_retains_reusable_receipt_once(
    monkeypatch: pytest.MonkeyPatch, nested: bool,
) -> None:
    from phydrax.meshing import _native_preparation as owner

    source, _, limits = _source()
    prepared = prepare_mesh_adaptation(
        source,
        MarkedMeshAdaptation(np.asarray([0], dtype=np.int64)),
        policy=MeshAdaptationPolicy(
            MeshAdaptationRoute.NATIVE_BISECTION,
            compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
            limits=limits,
        ),
    )
    retained: list[NativeExecutionRecord] = []
    remember = owner.NativePreparationLedger.remember_receipt

    def capture(self: Any, scope: Any, record: NativeExecutionRecord, **kwargs: Any) -> None:
        if record.owner_id == prepared.prepared_id:
            retained.append(record)
        remember(self, scope, record, **kwargs)

    monkeypatch.setattr(owner.NativePreparationLedger, "remember_receipt", capture)
    parent = NativeExecutionScope(limits.native_limits()) if nested else None
    with parent if parent is not None else nullcontext():
        result = execute_mesh_adaptation(prepared)
    assert result.status is phx.meshing.MeshAdaptationStatus.COMPLETE
    record = result.target.execution_evidence
    assert record is not None
    assert len(retained) == int(nested)
    if nested:
        assert retained[0].to_record() == record.to_record()
