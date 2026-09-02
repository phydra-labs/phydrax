#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from .._fingerprint import array_tree_fingerprint, canonical_fingerprint
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    FailurePolicy,
    LinearSolvePolicy,
    LinearSolveResult,
    LinearSystem,
    MaterializationPolicy,
    prepare,
    PreparedLinearSolve,
    solve,
)


class ScalarScreenJunctionCondition3D(StrictModule, NonTrainableState):
    name: str = eqx.field(static=True)
    law: str = eqx.field(static=True)
    constraint_rows: Array
    condition_id: str = eqx.field(static=True)

    def __init__(self, name: str, law: str, constraint_rows: ArrayLike, /):
        rows = np.asarray(constraint_rows)
        if (
            not str(name)
            or law
            not in (
                "independent",
                "continuity",
                "flux-balance",
                "continuity-and-flux-balance",
            )
            or rows.ndim != 2
            or rows.shape[0] == 0
        ):
            raise ValueError("Screen junction condition/law/rows are invalid.")
        if np.linalg.matrix_rank(rows) != rows.shape[0]:
            raise ValueError("Screen junction constraints must have full row rank.")
        self.name = str(name)
        self.law = str(law)
        self.constraint_rows = jnp.asarray(rows)
        self.condition_id = canonical_fingerprint(
            {
                "kind": "scalar-screen-junction-condition-3d",
                "name": name,
                "law": law,
                "rows": array_tree_fingerprint(rows),
            }
        )


class ScalarScreenJunctionResult3D(StrictModule):
    trace: Array
    multiplier: Array
    linear_result: LinearSolveResult
    constraint_defect: Array
    successful: Array


class PreparedScalarScreenJunctionSolve3D(StrictModule, NonTrainableState):
    sheet_operator: DenseLinearOperator
    constraint_matrix: Array
    prepared_linear: PreparedLinearSolve
    conditions: tuple[ScalarScreenJunctionCondition3D, ...]
    trace_size: int = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def solve(self, sheet_rhs: ArrayLike, /) -> ScalarScreenJunctionResult3D:
        rhs = jnp.asarray(sheet_rhs, dtype=self.sheet_operator.matrix.dtype)
        if rhs.shape != (self.trace_size,):
            raise ValueError("sheet_rhs has incompatible trace shape.")
        complete = jnp.concatenate(
            (rhs, jnp.zeros((self.constraint_matrix.shape[0],), dtype=rhs.dtype))
        )
        result = solve(self.prepared_linear, complete)
        trace = result.value[: self.trace_size]
        multiplier = result.value[self.trace_size :]
        defect = jnp.linalg.norm(self.constraint_matrix @ trace)
        return ScalarScreenJunctionResult3D(
            trace,
            multiplier,
            result,
            defect,
            result.successful & result.diagnostics.finite,
        )


def prepare_scalar_screen_junction_solve_3d(
    sheet_matrix: ArrayLike,
    conditions: tuple[ScalarScreenJunctionCondition3D, ...],
    /,
    *,
    maximum_dense_entries: int = 4_000_000,
) -> PreparedScalarScreenJunctionSolve3D:
    matrix = np.asarray(sheet_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("sheet_matrix must be nonempty square.")
    if not conditions or not all(
        isinstance(value, ScalarScreenJunctionCondition3D) for value in conditions
    ):
        raise TypeError("conditions must contain declared junction laws.")
    rows = np.concatenate(
        tuple(np.asarray(value.constraint_rows) for value in conditions), axis=0
    )
    if rows.shape[1] != matrix.shape[0] or np.linalg.matrix_rank(rows) != rows.shape[0]:
        raise ValueError(
            "Combined screen junction constraints are incompatible or rank deficient."
        )
    saddle = np.block(
        [
            [matrix, rows.T],
            [rows, np.zeros((rows.shape[0], rows.shape[0]), dtype=matrix.dtype)],
        ]
    )
    if saddle.size > int(maximum_dense_entries):
        raise ValueError("Screen junction saddle system exceeds maximum_dense_entries.")
    operator = DenseLinearOperator(
        jnp.asarray(saddle),
        operator_id=canonical_fingerprint(
            {
                "kind": "scalar-screen-junction-saddle-3d",
                "matrix": array_tree_fingerprint(saddle),
            }
        ),
    )
    policy = LinearSolvePolicy(
        DenseLU(),
        materialization=MaterializationPolicy(max_entries=maximum_dense_entries),
        failure=FailurePolicy("status"),
    )
    prepared_linear = prepare(
        LinearSystem(operator, problem_id=operator.operator_id), policy
    )
    sheet = DenseLinearOperator(
        jnp.asarray(matrix), operator_id=f"{operator.operator_id}:sheet"
    )
    return PreparedScalarScreenJunctionSolve3D(
        sheet,
        jnp.asarray(rows),
        prepared_linear,
        conditions,
        matrix.shape[0],
        canonical_fingerprint(
            {
                "kind": "prepared-scalar-screen-junction-solve-3d",
                "operator": operator.operator_id,
                "conditions": [value.condition_id for value in conditions],
            }
        ),
    )


__all__ = [
    "PreparedScalarScreenJunctionSolve3D",
    "ScalarScreenJunctionCondition3D",
    "ScalarScreenJunctionResult3D",
    "prepare_scalar_screen_junction_solve_3d",
]
