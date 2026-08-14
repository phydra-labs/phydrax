#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array

from ..._strict import StrictModule
from .._plans import LinearSolvePlan
from .._policies import HostSparseLU, SparseDirect
from .._results import LinearSolveStatus
from .._sparse_contract import AbstractSparseLinearOperator, SparseStorage


class DeviceSparseState(StrictModule):
    storage: SparseStorage


class HostSparseState(StrictModule):
    storage: SparseStorage
    factor: Any = eqx.field(static=True)


class SparseBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def _validated_storage(storage: SparseStorage, /) -> SparseStorage:
    nnz = storage.values.shape[0]
    positions = jnp.arange(nnz, dtype=storage.indptr.dtype)
    rows = jnp.searchsorted(storage.indptr[1:], positions, side="right")
    same_row = rows[1:] == rows[:-1]
    out_of_order = same_row & (storage.indices[1:] <= storage.indices[:-1])
    invalid = (
        jnp.any(~jnp.isfinite(storage.values))
        | (storage.indptr[0] != 0)
        | (storage.indptr[-1] != nnz)
        | jnp.any(storage.indptr[1:] < storage.indptr[:-1])
        | jnp.any(storage.indices < 0)
        | jnp.any(storage.indices >= storage.shape[1])
        | jnp.any(out_of_order)
    )
    values = eqx.error_if(
        storage.values,
        invalid,
        "CSR storage must be finite, bounded, monotone, sorted, and duplicate-free.",
    )
    return eqx.tree_at(lambda item: item.values, storage, values)


def prepare_sparse(problem: Any, plan: LinearSolvePlan, /) -> Any:
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("Sparse preparation requires an AbstractSparseLinearOperator.")
    storage = _validated_storage(operator.sparse_storage())
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("Sparse direct providers require sorted canonical CSR storage.")
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Sparse direct providers require square storage.")
    if plan.backend == "jax-sparse":
        if plan.method != SparseDirect().name:
            raise ValueError("The native sparse provider requires SparseDirect.")
        if storage.index_width != 32:
            raise ValueError("CUDA sparse direct execution requires 32-bit CSR indices.")
        return DeviceSparseState(storage)
    if plan.backend == "host-sparse":
        if plan.method != HostSparseLU().name:
            raise ValueError("The host sparse provider requires HostSparseLU.")
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        matrix = sp.csr_matrix(
            (
                np.asarray(storage.values),
                np.asarray(storage.indices),
                np.asarray(storage.indptr),
            ),
            shape=storage.shape,
        )
        return HostSparseState(storage, spla.splu(matrix.tocsc()))
    raise ValueError(f"Unsupported sparse backend {plan.backend!r}.")


def solve_sparse(
    state: DeviceSparseState | HostSparseState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> SparseBackendOutput:
    if rhs.ndim != 2:
        raise ValueError("Sparse canonical right-hand sides must have shape (n, k).")
    if isinstance(state, DeviceSparseState):
        from jax.experimental.sparse.linalg import spsolve

        method = plan.policy.method
        reorder = method.reorder if isinstance(method, SparseDirect) else 1
        columns = tuple(
            spsolve(
                state.storage.values,
                state.storage.indices,
                state.storage.indptr,
                rhs[:, column],
                tol=plan.policy.tolerance.relative,
                reorder=reorder,
            )
            for column in range(rhs.shape[1])
        )
        value = jnp.stack(columns, axis=1)
    elif isinstance(state, HostSparseState):
        if plan.policy.differentiation.mode != "none":
            raise ValueError("HostSparseLU execution is non-differentiable.")
        value = jnp.asarray(state.factor.solve(np.asarray(rhs)))
    else:
        raise TypeError(f"Unsupported sparse prepared state {type(state).__name__}.")
    finite = jnp.all(jnp.isfinite(value), axis=0)
    status = jnp.where(
        finite,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    count = rhs.shape[1]
    return SparseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros((count,), dtype=jnp.int32),
        rank=jnp.asarray(-1, dtype=jnp.int32),
        condition_estimate=jnp.full((count,), jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
    )


def solve_host_sparse_transformed(
    state: HostSparseState,
    rhs: Array,
    /,
    *,
    adjoint: bool,
) -> SparseBackendOutput:
    mode = "H" if adjoint else "T"
    value = jnp.asarray(state.factor.solve(np.asarray(rhs), trans=mode))
    finite = jnp.all(jnp.isfinite(value), axis=0)
    status = jnp.where(
        finite,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    count = rhs.shape[1]
    return SparseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros((count,), dtype=jnp.int32),
        rank=jnp.asarray(-1, dtype=jnp.int32),
        condition_estimate=jnp.full((count,), jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
    )


__all__ = [
    "DeviceSparseState",
    "HostSparseState",
    "SparseBackendOutput",
    "prepare_sparse",
    "solve_host_sparse_transformed",
    "solve_sparse",
]
