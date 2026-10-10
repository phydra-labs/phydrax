#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

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


@dataclass(frozen=True, eq=False)
class _ColumnPermutedLU:
    """SuperLU factor of ``A[:, columns]`` under SuperLU's natural ordering."""

    factor: Any
    columns: np.ndarray


def _host_factor(provider: str, matrix: Any, columns: np.ndarray | None, /) -> Any:
    if provider == "scipy-superlu":
        import scipy.sparse.linalg as spla

        if columns is None:
            raise ValueError("Host SuperLU requires the plan's sparse LU analysis.")
        # The planned symbolic fill bound holds for exactly this column order.
        return _ColumnPermutedLU(
            spla.splu(matrix.tocsc()[:, columns], permc_spec="NATURAL"), columns
        )
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
    provider: str,
    factor: Any,
    rhs: np.ndarray,
    /,
    *,
    transpose: bool = False,
    adjoint: bool = False,
) -> np.ndarray:
    if provider == "scipy-superlu":
        mode = "H" if adjoint else ("T" if transpose else "N")
        # (A Q)ᵀ x = Qᵀ b for transposed solves; A x = b is (A Q)(Qᵀ x) = b.
        if transpose or adjoint:
            return np.asarray(factor.factor.solve(rhs[factor.columns], trans=mode))
        permuted = np.asarray(factor.factor.solve(rhs))
        solution = np.empty_like(permuted)
        solution[factor.columns] = permuted
        return solution
    if provider == "umfpack":
        import scipy.sparse.linalg as spla

        matrix = factor.getH() if adjoint else (factor.T if transpose else factor)
        return _columnwise(
            lambda column: spla.spsolve(matrix, column, use_umfpack=True),
            rhs,
        )
    if provider == "cholmod":
        solved_rhs = np.conjugate(rhs) if transpose and not adjoint else rhs
        solution = np.asarray(factor.solve_A(solved_rhs))
        return np.conjugate(solution) if transpose and not adjoint else solution
    if provider == "spqr":
        import sparseqr

        matrix = factor.getH() if adjoint else (factor.T if transpose else factor)
        return _columnwise(lambda column: sparseqr.solve(matrix, column), rhs)
    raise ValueError(f"Unsupported host sparse provider {provider!r}.")


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
        supported = (isinstance(method, SparseQR) and method.provider == "jax-cuda") or (
            isinstance(method, SparseLU) and method.provider == "jax-cpu"
        )
        if not supported:
            raise ValueError(
                "The native sparse backend requires SparseQR(provider='jax-cuda') or SparseLU(provider='jax-cpu')."
            )
        if isinstance(method, SparseQR) and storage.index_width != 32:
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

        batch_count = int(np.prod(storage.batch_shape)) if storage.batch_shape else 1
        values = np.asarray(storage.values).reshape((batch_count, storage.nnz))
        analysis = plan.sparse_lu_analysis
        columns = None if analysis is None else np.asarray(analysis.column_permutation)
        factors = tuple(
            _host_factor(
                provider,
                sp.csr_matrix(
                    (
                        batch_values,
                        np.asarray(storage.indices),
                        np.asarray(storage.indptr),
                    ),
                    shape=storage.shape,
                ),
                columns,
            )
            for batch_values in values
        )
        return HostSparseState(storage, provider, factors)
    raise ValueError(f"Unsupported sparse backend {plan.backend!r}.")


type _ColumnSolver = Callable[[Array, Array, Array, Array], Array]


def _spsolve_columns(tolerance: float, reorder: int, /) -> _ColumnSolver:
    """JAX's sparse direct solve over canonical ``(n, k)`` right-hand-side columns.

    ``spsolve`` accepts one vector and defines no batching rule. Under ``vmap``,
    lanes sharing one CSR matrix become additional right-hand-side columns of that
    matrix; lanes carrying their own matrix are solved sequentially.
    """
    from jax.experimental.sparse.linalg import spsolve

    @jax.custom_batching.custom_vmap
    def solve_columns(values: Array, indices: Array, indptr: Array, rhs: Array) -> Array:
        def solve_column(column: Array) -> Array:
            return spsolve(
                values, indices, indptr, column, tol=tolerance, reorder=reorder
            )

        return jax.lax.map(solve_column, rhs.T).T

    @solve_columns.def_vmap
    def solve_lanes(
        axis_size: int,
        in_batched: Sequence[bool],
        values: Array,
        indices: Array,
        indptr: Array,
        rhs: Array,
    ) -> tuple[Array, bool]:
        arguments = (values, indices, indptr, rhs)
        if not any(in_batched[:3]):
            size, count = rhs.shape[1:]
            columns = jnp.moveaxis(rhs, 0, 1).reshape((size, axis_size * count))
            solved = solve_columns(values, indices, indptr, columns)
            return jnp.moveaxis(solved.reshape((size, axis_size, count)), 1, 0), True

        def solve_lane(lane: Array) -> Array:
            return solve_columns(
                *(
                    argument[lane] if batched else argument
                    for argument, batched in zip(arguments, in_batched, strict=True)
                )
            )

        return jax.lax.map(solve_lane, jnp.arange(axis_size)), True

    return solve_columns


def _device_sparse_solve(
    values: Array,
    storage: SparseStorage,
    rhs: Array,
    solve_columns: _ColumnSolver,
    /,
) -> Array:
    """Solve ``A X = B`` for one CSR value set as a transposable linear solve.

    The operator derivative is ``-A^{-1} dA X`` and the transpose solves with the
    CSR transpose of the shared pattern, so forward, reverse, and batched
    transformations all reuse the direct solve.
    """
    size = storage.shape[0]
    positions = jnp.arange(storage.nnz, dtype=storage.indptr.dtype)
    rows = jnp.searchsorted(storage.indptr[1:], positions, side="right")

    def matvec(columns: Array) -> Array:
        return jax.ops.segment_sum(
            values[:, None] * columns[storage.indices],
            rows,
            num_segments=size,
            indices_are_sorted=True,
        )

    def solve(_: Callable[[Array], Array], right: Array) -> Array:
        return solve_columns(values, storage.indices, storage.indptr, right)

    def transpose_solve(_: Callable[[Array], Array], right: Array) -> Array:
        # Row-major CSR entries in stable column order are the CSR of A^T.
        order = jnp.argsort(storage.indices, stable=True)
        counts = jnp.bincount(storage.indices, length=size)
        transposed_indptr = jnp.concatenate(
            (jnp.zeros((1,), dtype=counts.dtype), jnp.cumsum(counts))
        ).astype(storage.indptr.dtype)
        return solve_columns(
            values[order],
            rows[order].astype(storage.indices.dtype),
            transposed_indptr,
            right,
        )

    return jax.lax.custom_linear_solve(matvec, rhs, solve, transpose_solve)


def solve_sparse(
    state: DeviceSparseState | HostSparseState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> SparseBackendOutput:
    batch_shape = state.storage.batch_shape
    expected_rank = len(batch_shape) + 2
    if rhs.ndim != expected_rank or rhs.shape[: len(batch_shape)] != batch_shape:
        raise ValueError(
            "Sparse canonical right-hand sides must have shape batch_shape + (n, k)."
        )
    batch_count = int(np.prod(batch_shape)) if batch_shape else 1
    size = state.storage.shape[0]
    count = rhs.shape[-1]
    flattened_rhs = rhs.reshape((batch_count, size, count))
    if isinstance(state, DeviceSparseState):
        method = plan.policy.method
        solve_columns = _spsolve_columns(
            plan.policy.tolerance.relative,
            method.reorder if isinstance(method, SparseQR) else 1,
        )
        flattened_values = state.storage.values.reshape((batch_count, state.storage.nnz))

        def solve_one(inputs: tuple[Array, Array]) -> Array:
            values, right_hand_side = inputs
            return _device_sparse_solve(
                values, state.storage, right_hand_side, solve_columns
            )

        value = jax.lax.map(solve_one, (flattened_values, flattened_rhs))
    elif isinstance(state, HostSparseState):
        if plan.policy.differentiation.mode != "none":
            raise ValueError("Host sparse execution is non-differentiable.")
        value = jnp.asarray(
            np.stack(
                tuple(
                    _host_solve(
                        state.provider,
                        factor,
                        np.asarray(batch_rhs),
                    )
                    for factor, batch_rhs in zip(
                        state.factor,
                        np.asarray(flattened_rhs),
                        strict=True,
                    )
                ),
                axis=0,
            )
        )
    else:
        raise TypeError(f"Unsupported sparse prepared state {type(state).__name__}.")
    value = value.reshape(batch_shape + (size, count))
    finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.where(
        finite,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    return SparseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros(batch_shape + (count,), dtype=jnp.int32),
        rank=jnp.full(batch_shape, -1, dtype=jnp.int32),
        condition_estimate=jnp.full(
            batch_shape + (count,),
            jnp.nan,
            dtype=rhs.real.dtype,
        ),
        singular_values=None,
    )


def solve_host_sparse_transformed(
    state: HostSparseState,
    rhs: Array,
    /,
    *,
    adjoint: bool,
) -> SparseBackendOutput:
    batch_shape = state.storage.batch_shape
    batch_count = int(np.prod(batch_shape)) if batch_shape else 1
    size = state.storage.shape[0]
    count = rhs.shape[-1]
    flattened_rhs = np.asarray(rhs).reshape((batch_count, size, count))
    value = jnp.asarray(
        np.stack(
            tuple(
                _host_solve(
                    state.provider,
                    factor,
                    batch_rhs,
                    transpose=not adjoint,
                    adjoint=adjoint,
                )
                for factor, batch_rhs in zip(
                    state.factor,
                    flattened_rhs,
                    strict=True,
                )
            ),
            axis=0,
        )
    ).reshape(batch_shape + (size, count))
    finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.where(
        finite,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    return SparseBackendOutput(
        value=value,
        status=status,
        iterations=jnp.zeros(batch_shape + (count,), dtype=jnp.int32),
        rank=jnp.full(batch_shape, -1, dtype=jnp.int32),
        condition_estimate=jnp.full(
            batch_shape + (count,),
            jnp.nan,
            dtype=rhs.real.dtype,
        ),
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
