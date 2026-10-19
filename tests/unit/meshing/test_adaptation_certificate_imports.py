#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Native adaptation imports authenticated unchanged-cell preparation once."""

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


def test_native_adaptation_imports_unchanged_cell_receipt_once() -> None:
    source, receipt, limits = _source()
    policy = MeshAdaptationPolicy(
        MeshAdaptationRoute.NATIVE_BISECTION,
        compatibility=BisectionCompatibility.CONFORMING_CLOSURE,
        limits=limits,
    )
    request = MarkedMeshAdaptation(np.asarray([0], dtype=np.int64))
    with NativeExecutionScope(limits.native_limits()) as full_scope:
        native_preparation_ledger(full_scope).import_receipt(full_scope, receipt)
        full = execute_mesh_adaptation(
            prepare_mesh_adaptation(source, request, policy=policy)
        )
    with NativeExecutionScope(limits.native_limits()) as inherited_scope:
        native_preparation_ledger(inherited_scope).import_receipt(
            inherited_scope, receipt
        )
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
    full_evidence = full.target.execution_evidence
    assert evidence is not None and full_evidence is not None
    assert evidence.imported_preparation_hex().count(receipt.receipt_hex()) == 1
    assert int(np.asarray(evidence.work_counters.construction_work)) < int(
        np.asarray(full_evidence.work_counters.construction_work)
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
