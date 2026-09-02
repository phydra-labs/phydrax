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
from ..discretization.fem import (
    PreparedMaxwellMortarInterfaceTrace3D,
    PreparedScalarMortarInterfaceTrace3D,
)
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
from ..operators.integral.layer_potential import PreparedPeriodicMaxwellBoundary3D


class CoupledFEMBEMResult3D(StrictModule):
    interior: Array
    boundary: Array
    linear_result: LinearSolveResult
    interface_residual: Array
    successful: Array


class PreparedNonmatchingFEMBEM3D(StrictModule, NonTrainableState):
    operator: DenseLinearOperator
    prepared_linear: PreparedLinearSolve
    coupling: Array
    interior_size: int = eqx.field(static=True)
    boundary_size: int = eqx.field(static=True)
    family: str = eqx.field(static=True)
    prepared_id: str = eqx.field(static=True)

    def solve(
        self, interior_load: ArrayLike, boundary_load: ArrayLike, /
    ) -> CoupledFEMBEMResult3D:
        interior = jnp.asarray(interior_load, dtype=self.operator.matrix.dtype)
        boundary = jnp.asarray(boundary_load, dtype=self.operator.matrix.dtype)
        if interior.shape != (self.interior_size,) or boundary.shape != (
            self.boundary_size,
        ):
            raise ValueError("Coupled FEM-BEM loads have incompatible shapes.")
        result = solve(self.prepared_linear, jnp.concatenate((interior, boundary)))
        u = result.value[: self.interior_size]
        q = result.value[self.interior_size :]
        residual = self.coupling @ u - q
        return CoupledFEMBEMResult3D(
            u, q, result, residual, result.successful & result.diagnostics.finite
        )


def _prepare(interior_matrix, boundary_matrix, coupling, family, maximum_dense_entries):
    interior = np.asarray(interior_matrix)
    boundary = np.asarray(boundary_matrix)
    trace = np.asarray(coupling)
    if (
        interior.ndim != 2
        or interior.shape[0] != interior.shape[1]
        or boundary.ndim != 2
        or boundary.shape[0] != boundary.shape[1]
        or trace.shape != (boundary.shape[0], interior.shape[0])
    ):
        raise ValueError("Nonmatching coupled block shapes are incompatible.")
    matrix = np.block([[interior, trace.T.conj()], [trace, -boundary]])
    if matrix.size > int(maximum_dense_entries):
        raise ValueError("Nonmatching coupled matrix exceeds maximum_dense_entries.")
    operator = DenseLinearOperator(
        jnp.asarray(matrix),
        operator_id=canonical_fingerprint(
            {
                "kind": f"nonmatching-{family}-fem-bem-3d",
                "matrix": array_tree_fingerprint(matrix),
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
    return PreparedNonmatchingFEMBEM3D(
        operator,
        prepared_linear,
        jnp.asarray(trace),
        interior.shape[0],
        boundary.shape[0],
        family,
        canonical_fingerprint(
            {
                "kind": "prepared-nonmatching-fem-bem-3d",
                "family": family,
                "operator": operator.operator_id,
            }
        ),
    )


def prepare_scalar_nonmatching_fem_bem_3d(
    interior_matrix: ArrayLike,
    boundary_matrix: ArrayLike,
    mortar: PreparedScalarMortarInterfaceTrace3D,
    /,
    *,
    maximum_dense_entries: int = 4_000_000,
) -> PreparedNonmatchingFEMBEM3D:
    if not isinstance(mortar, PreparedScalarMortarInterfaceTrace3D):
        raise TypeError("mortar must be PreparedScalarMortarInterfaceTrace3D.")
    return _prepare(
        interior_matrix,
        boundary_matrix,
        mortar.trace.matrix,
        "scalar",
        maximum_dense_entries,
    )


def prepare_maxwell_fem_bem_3d(
    interior_matrix: ArrayLike,
    periodic_boundary: PreparedPeriodicMaxwellBoundary3D,
    mortar: PreparedMaxwellMortarInterfaceTrace3D,
    /,
    *,
    maximum_dense_entries: int = 4_000_000,
) -> PreparedNonmatchingFEMBEM3D:
    if not isinstance(
        periodic_boundary, PreparedPeriodicMaxwellBoundary3D
    ) or not isinstance(mortar, PreparedMaxwellMortarInterfaceTrace3D):
        raise TypeError("Periodic Maxwell boundary and qualified mortar are required.")
    return _prepare(
        interior_matrix,
        periodic_boundary.operator.matrix,
        mortar.tangential_trace.matrix,
        "maxwell",
        maximum_dense_entries,
    )


__all__ = [
    "CoupledFEMBEMResult3D",
    "PreparedNonmatchingFEMBEM3D",
    "prepare_maxwell_fem_bem_3d",
    "prepare_scalar_nonmatching_fem_bem_3d",
]
