#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from enum import IntEnum

import equinox as eqx
from jaxtyping import Array, PyTree

from .._strict import StrictModule


class MatrixFunctionStatus(IntEnum):
    """Portable disposition of one matrix-function action."""

    SUCCESS = 0
    TOLERANCE_NOT_MET = 1
    BREAKDOWN = 2
    NONFINITE = 3
    PLANNING_FAILURE = 4
    RESOURCE_EXHAUSTED = 5


class MatrixFunctionDiagnostics(StrictModule):
    """Accuracy, work, resource, and derivative evidence for one action."""

    error_estimate: Array
    residual_estimate: Array
    error_bound: Array
    error_bound_available: Array
    error_bound_certified: Array
    finite: Array
    converged: Array
    derivative_valid: Array
    effective_dimension: Array
    selected_degree: Array
    scaling_count: Array
    setup_matvec_count: Array
    action_matvec_count: Array
    transpose_matvec_count: Array
    breakdown_status: Array
    retained_storage_bytes: int = eqx.field(static=True)
    workspace_bytes: int = eqx.field(static=True)


class MatrixFunctionProvenance(StrictModule):
    """Method, operator, prepared-state, and numerical-version identity."""

    method: str = eqx.field(static=True)
    kind: str = eqx.field(static=True)
    description: str = eqx.field(static=True)
    operator_id: str = eqx.field(static=True)
    plan_id: str | None = eqx.field(static=True)
    prepared_id: str | None = eqx.field(static=True)
    trace_source: str = eqx.field(static=True)
    norm_source: str = eqx.field(static=True)
    numeric_version: Array


class MatrixFunctionResult(StrictModule):
    """Matrix-function value with explicit status and numerical evidence."""

    value: PyTree[Array]
    status: Array
    diagnostics: MatrixFunctionDiagnostics
    provenance: MatrixFunctionProvenance

    @property
    def successful(self) -> Array:
        return self.status == int(MatrixFunctionStatus.SUCCESS)


__all__ = [
    "MatrixFunctionDiagnostics",
    "MatrixFunctionProvenance",
    "MatrixFunctionResult",
    "MatrixFunctionStatus",
]
