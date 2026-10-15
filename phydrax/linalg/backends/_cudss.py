#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
from math import prod
from types import ModuleType
from typing import Any, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array, lax

from ..._strict import StrictModule
from .._plans import LinearSolvePlan
from .._policies import SparseLDLT
from .._results import LinearSolveStatus
from .._sparse_contract import AbstractSparseLinearOperator, SparseStorage


@final
class CudssSymbolicState(StrictModule):
    token: Any
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


@final
class CudssFactorState(StrictModule):
    token: Any
    storage: SparseStorage
    positive_inertia: Array
    negative_inertia: Array
    zero_inertia: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


@final
class CudssBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    refinement_steps: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None
    native_rebuilds: Array


def _ffi() -> ModuleType:
    return importlib.import_module("phydrax.linalg.backends._cudss_ffi")


def _storage(problem: Any, /) -> SparseStorage:
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("cuDSS preparation requires an AbstractSparseLinearOperator.")
    storage = operator.sparse_storage()
    if storage.values.dtype not in (jnp.dtype(jnp.float32), jnp.dtype(jnp.float64)):
        raise ValueError(
            "cuDSS requires real-valued CSR storage with float32/float64 values."
        )
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("cuDSS requires sorted canonical CSR storage.")
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("cuDSS requires square CSR storage.")
    if storage.index_width != 32:
        raise ValueError("cuDSS requires 32-bit CSR indices.")
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
        "cuDSS CSR storage must be finite, bounded, monotone, sorted, and unique.",
    )
    return eqx.tree_at(lambda item: item.values, storage, values)


def _factor_values(storage: SparseStorage, /) -> Array:
    return (
        storage.values.reshape((prod(storage.batch_shape), storage.nnz))
        if storage.batch_shape
        else storage.values
    )


def _inertia(token: Any, batch_shape: tuple[int, ...], /) -> tuple[Array, Array, Array]:
    diagonal, _ = _ffi().diagonal(lax.stop_gradient(token))
    shaped = diagonal.reshape(batch_shape + (-1,))
    positive = jnp.sum(shaped >= 1e-13, axis=-1, dtype=jnp.int32)
    negative = jnp.sum(shaped <= -1e-13, axis=-1, dtype=jnp.int32)
    return positive, negative, jnp.full(batch_shape, -1, dtype=jnp.int32)


def analyze_cudss(problem: Any, plan: LinearSolvePlan, /) -> CudssSymbolicState:
    from ..._cuda_native import cuda_partition_key

    method = plan.policy.method
    if not isinstance(method, SparseLDLT):
        raise TypeError("cuDSS analysis requires SparseLDLT.")
    storage = _storage(problem)
    token = _ffi().analyze(
        _factor_values(storage),
        storage.indptr,
        storage.indices,
        partition=cuda_partition_key(plan.plan_id),
        capacity=method.factor_cache_capacity,
        reordering=method.reordering,
        memory=method.memory_mode,
    )
    return CudssSymbolicState(token, storage.batch_shape, problem.operator.operator_id)


def bind_cudss(
    symbolic: CudssSymbolicState, problem: Any, plan: LinearSolvePlan, /
) -> CudssFactorState:
    del plan
    if not isinstance(symbolic, CudssSymbolicState):
        raise TypeError("cuDSS binding requires CudssSymbolicState.")
    storage = _storage(problem)
    if problem.operator.operator_id != symbolic.structure_id:
        raise ValueError("cuDSS binding cannot change sparse structure.")
    token = _ffi().factorize(symbolic.token, _factor_values(storage))
    positive, negative, zero = _inertia(token, storage.batch_shape)
    return CudssFactorState(
        token,
        storage,
        positive,
        negative,
        zero,
        storage.batch_shape,
        symbolic.structure_id,
    )


def refresh_cudss(
    symbolic: CudssSymbolicState,
    previous: CudssFactorState,
    problem: Any,
    plan: LinearSolvePlan,
    /,
) -> CudssFactorState:
    del symbolic, plan
    if not isinstance(previous, CudssFactorState):
        raise TypeError("cuDSS refresh requires CudssFactorState.")
    storage = _storage(problem)
    if problem.operator.operator_id != previous.structure_id:
        raise ValueError("cuDSS refresh cannot change sparse structure.")
    token = _ffi().refactorize(previous.token, _factor_values(storage))
    positive, negative, zero = _inertia(token, storage.batch_shape)
    return CudssFactorState(
        token,
        storage,
        positive,
        negative,
        zero,
        storage.batch_shape,
        previous.structure_id,
    )


def solve_cudss(
    state: CudssFactorState, rhs: Array, plan: LinearSolvePlan, /
) -> CudssBackendOutput:
    if not isinstance(state, CudssFactorState):
        raise TypeError("cuDSS solve requires CudssFactorState.")
    if rhs.ndim < 2:
        raise ValueError("Canonical right-hand sides must end with (n, k).")
    method = plan.policy.method
    if not isinstance(method, SparseLDLT):
        raise TypeError("cuDSS solve requires SparseLDLT.")
    canonical = (
        rhs.reshape((prod(state.batch_shape), rhs.shape[-2], rhs.shape[-1]))
        if state.batch_shape
        else rhs
    )
    solved, rebuilt = _ffi().solve(
        state.token,
        jnp.moveaxis(canonical, -1, 0),
        refinement_steps=method.refinement_steps,
    )
    canonical_value = jnp.moveaxis(solved, 0, -1)
    value = (
        canonical_value.reshape(state.batch_shape + canonical_value.shape[-2:])
        if state.batch_shape
        else canonical_value
    )
    finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.where(
        finite, int(LinearSolveStatus.SUCCESS), int(LinearSolveStatus.NONFINITE_OUTPUT)
    ).astype(jnp.int32)
    return CudssBackendOutput(
        value=value,
        status=status,
        iterations=jnp.full(status.shape, method.refinement_steps, dtype=jnp.int32),
        refinement_steps=jnp.full(status.shape, method.refinement_steps, dtype=jnp.int32),
        rank=jnp.asarray(-1, dtype=jnp.int32),
        condition_estimate=jnp.full(status.shape, jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
        native_rebuilds=jnp.asarray(rebuilt, dtype=jnp.int32).reshape(()),
    )


def release_cudss(state: CudssFactorState, /) -> bool | Array:
    if not isinstance(state, CudssFactorState):
        raise TypeError("cuDSS release requires CudssFactorState.")
    return _ffi().release(state.token)


__all__ = [
    "CudssBackendOutput",
    "CudssFactorState",
    "CudssSymbolicState",
    "analyze_cudss",
    "bind_cudss",
    "refresh_cudss",
    "release_cudss",
    "solve_cudss",
]
