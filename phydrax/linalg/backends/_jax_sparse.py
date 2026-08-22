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
from .._policies import SparseCholesky, SparseLU, SparseQR
from .._results import LinearSolveStatus
from .._sparse_contract import AbstractSparseLinearOperator, SparseStorage


class DeviceSparseState(StrictModule):
    storage: SparseStorage


class HostSparseState(StrictModule):
    storage: SparseStorage
    provider: str = eqx.field(static=True)
    factor: Any = eqx.field(static=True)


class SparseBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def _validated_storage(storage: SparseStorage, /) -> SparseStorage:
    if storage.batch_shape:
        raise ValueError("Sparse direct providers require unbatched CSR values.")
    nnz = storage.nnz
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


def _host_factor(provider: str, matrix: Any, /) -> Any:
    if provider == "scipy-superlu":
        import scipy.sparse.linalg as spla

        return spla.splu(matrix.tocsc())
    if provider == "umfpack":
        import scikits.umfpack  # noqa: F401

        return matrix.tocsc()
    if provider == "cholmod":
        from sksparse.cholmod import cholesky

        return cholesky(matrix.tocsc())
    if provider == "spqr":
        import sparseqr  # noqa: F401

        return matrix.tocsc()
    raise ValueError(f"Unsupported host sparse provider {provider!r}.")


def _columnwise(function: Any, rhs: np.ndarray, /) -> np.ndarray:
    return np.column_stack(
        tuple(function(rhs[:, index]) for index in range(rhs.shape[1]))
    )


def _host_solve(
    state: HostSparseState,
    rhs: np.ndarray,
    /,
    *,
    transpose: bool = False,
    adjoint: bool = False,
) -> np.ndarray:
    if state.provider == "scipy-superlu":
        mode = "H" if adjoint else ("T" if transpose else "N")
        return np.asarray(state.factor.solve(rhs, trans=mode))
    if state.provider == "umfpack":
        import scipy.sparse.linalg as spla

        matrix = (
            state.factor.getH()
            if adjoint
            else (state.factor.T if transpose else state.factor)
        )
        return _columnwise(
            lambda column: spla.spsolve(matrix, column, use_umfpack=True),
            rhs,
        )
    if state.provider == "cholmod":
        solved_rhs = np.conjugate(rhs) if transpose and not adjoint else rhs
        solution = np.asarray(state.factor.solve_A(solved_rhs))
        return np.conjugate(solution) if transpose and not adjoint else solution
    if state.provider == "spqr":
        import sparseqr

        matrix = (
            state.factor.getH()
            if adjoint
            else (state.factor.T if transpose else state.factor)
        )
        return _columnwise(lambda column: sparseqr.solve(matrix, column), rhs)
    raise ValueError(f"Unsupported host sparse provider {state.provider!r}.")


def prepare_sparse(problem: Any, plan: LinearSolvePlan, /) -> Any:
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("Sparse preparation requires an AbstractSparseLinearOperator.")
    storage = _validated_storage(operator.sparse_storage())
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("Sparse direct providers require sorted canonical CSR storage.")
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Sparse direct providers require square storage.")
    method = plan.policy.method
    if plan.backend == "jax-sparse":
        if not isinstance(method, SparseQR) or method.provider != "jax-cuda":
            raise ValueError(
                "The native sparse backend requires SparseQR(provider='jax-cuda')."
            )
        if storage.index_width != 32:
            raise ValueError("CUDA sparse QR execution requires 32-bit CSR indices.")
        return DeviceSparseState(storage)
    if plan.backend == "host-sparse":
        if not isinstance(method, (SparseLU, SparseCholesky, SparseQR)):
            raise ValueError("The host sparse backend requires a host sparse method.")
        provider = (
            "scipy-superlu"
            if isinstance(method, SparseLU) and method.provider == "auto"
            else method.provider
        )
        if provider == "jax-cuda":
            raise ValueError("The JAX CUDA sparse provider is not a host provider.")
        import scipy.sparse as sp

        matrix = sp.csr_matrix(
            (
                np.asarray(storage.values),
                np.asarray(storage.indices),
                np.asarray(storage.indptr),
            ),
            shape=storage.shape,
        )
        return HostSparseState(storage, provider, _host_factor(provider, matrix))
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
        reorder = method.reorder if isinstance(method, SparseQR) else 1
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
            raise ValueError("Host sparse execution is non-differentiable.")
        value = jnp.asarray(_host_solve(state, np.asarray(rhs)))
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
    value = jnp.asarray(
        _host_solve(
            state,
            np.asarray(rhs),
            transpose=not adjoint,
            adjoint=adjoint,
        )
    )
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
