#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib
from math import prod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array

from ..._strict import StrictModule
from .._plans import LinearSolvePlan
from .._policies import SparseLDLT
from .._results import LinearSolveStatus
from .._sparse_contract import AbstractSparseLinearOperator, SparseStorage


class SpineaxSymbolicState(StrictModule):
    token: Any
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class SpineaxFactorState(StrictModule):
    token: Any
    storage: SparseStorage
    positive_inertia: Array
    negative_inertia: Array
    zero_inertia: Array
    batch_shape: tuple[int, ...] = eqx.field(static=True)
    structure_id: str = eqx.field(static=True)


class SpineaxBackendOutput(StrictModule):
    value: Array
    status: Array
    iterations: Array
    rank: Array
    condition_estimate: Array
    singular_values: Array | None


def _cudss():
    return importlib.import_module("spineax.cudss")


def _storage(problem: Any, /) -> SparseStorage:
    operator = problem.operator
    if not isinstance(operator, AbstractSparseLinearOperator):
        raise TypeError("Spineax preparation requires an AbstractSparseLinearOperator.")
    storage = operator.sparse_storage()
    if not storage.canonical or not storage.sorted_indices:
        raise ValueError("Spineax requires sorted canonical CSR storage.")
    if storage.shape[0] != storage.shape[1]:
        raise ValueError("Spineax requires square CSR storage.")
    if storage.index_width != 32:
        raise ValueError("Spineax requires 32-bit CSR indices.")
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
        "Spineax CSR storage must be finite, bounded, monotone, sorted, and unique.",
    )
    return eqx.tree_at(lambda item: item.values, storage, values)


def _factor_values(storage: SparseStorage, /) -> Array:
    if not storage.batch_shape:
        return storage.values
    return storage.values.reshape((prod(storage.batch_shape), storage.nnz))


def _inertia(token: Any, batch_shape: tuple[int, ...], /):
    cudss = _cudss()
    batch_size = prod(batch_shape) if batch_shape else 1
    counts = cudss.inertia(cudss.query(token), batch_size=batch_size)
    if not batch_shape:
        positive, negative = counts[0], counts[1]
        zero = jnp.asarray(-1, dtype=jnp.int32)
    else:
        shaped = counts.reshape(batch_shape + (2,))
        positive = shaped[..., 0]
        negative = shaped[..., 1]
        zero = jnp.full(batch_shape, -1, dtype=jnp.int32)
    return positive, negative, zero


def analyze_spineax(problem: Any, plan: LinearSolvePlan, /) -> SpineaxSymbolicState:
    method = plan.policy.method
    if not isinstance(method, SparseLDLT):
        raise TypeError("Spineax analysis requires SparseLDLT.")
    storage = _storage(problem)
    cudss = _cudss()
    token = cudss.analyze(
        _factor_values(storage),
        storage.indptr,
        storage.indices,
        mtype_id="symmetric",
        mview_id="full",
        reordering=method.reordering,
        memory=method.memory_mode,
    )
    return SpineaxSymbolicState(
        token,
        storage.batch_shape,
        problem.operator.operator_id,
    )


def bind_spineax(
    symbolic: SpineaxSymbolicState,
    problem: Any,
    plan: LinearSolvePlan,
    /,
) -> SpineaxFactorState:
    if not isinstance(symbolic, SpineaxSymbolicState):
        raise TypeError("Spineax binding requires SpineaxSymbolicState.")
    storage = _storage(problem)
    if problem.operator.operator_id != symbolic.structure_id:
        raise ValueError("Spineax binding cannot change sparse structure.")
    token = _cudss().factorize(symbolic.token, _factor_values(storage))
    positive, negative, zero = _inertia(token, storage.batch_shape)
    return SpineaxFactorState(
        token,
        storage,
        positive,
        negative,
        zero,
        storage.batch_shape,
        symbolic.structure_id,
    )


def refresh_spineax(
    symbolic: SpineaxSymbolicState,
    previous: SpineaxFactorState,
    problem: Any,
    plan: LinearSolvePlan,
    /,
) -> SpineaxFactorState:
    del symbolic, plan
    if not isinstance(previous, SpineaxFactorState):
        raise TypeError("Spineax refresh requires SpineaxFactorState.")
    storage = _storage(problem)
    if problem.operator.operator_id != previous.structure_id:
        raise ValueError("Spineax refresh cannot change sparse structure.")
    token = _cudss().refactorize(previous.token, _factor_values(storage))
    positive, negative, zero = _inertia(token, storage.batch_shape)
    return SpineaxFactorState(
        token,
        storage,
        positive,
        negative,
        zero,
        storage.batch_shape,
        previous.structure_id,
    )


def solve_spineax(
    state: SpineaxFactorState,
    rhs: Array,
    plan: LinearSolvePlan,
    /,
) -> SpineaxBackendOutput:
    if not isinstance(state, SpineaxFactorState):
        raise TypeError("Spineax solve requires SpineaxFactorState.")
    if rhs.ndim < 2:
        raise ValueError("Canonical right-hand sides must end with (n, k).")
    method = plan.policy.method
    if not isinstance(method, SparseLDLT):
        raise TypeError("Spineax solve requires SparseLDLT.")
    batch_size = prod(state.batch_shape) if state.batch_shape else 1
    canonical = (
        rhs.reshape((batch_size, rhs.shape[-2], rhs.shape[-1]))
        if state.batch_shape
        else rhs
    )
    spineax_rhs = jnp.moveaxis(canonical, -1, 0)
    solved = _cudss().solve(
        state.token,
        spineax_rhs,
        ir_nsteps=method.refinement_steps,
    )
    canonical_value = jnp.moveaxis(solved, 0, -1)
    value = (
        canonical_value.reshape(state.batch_shape + canonical_value.shape[-2:])
        if state.batch_shape
        else canonical_value
    )
    finite = jnp.all(jnp.isfinite(value), axis=-2)
    status = jnp.where(
        finite,
        int(LinearSolveStatus.SUCCESS),
        int(LinearSolveStatus.NONFINITE_OUTPUT),
    ).astype(jnp.int32)
    return SpineaxBackendOutput(
        value=value,
        status=status,
        iterations=jnp.full(status.shape, method.refinement_steps, dtype=jnp.int32),
        rank=jnp.asarray(-1, dtype=jnp.int32),
        condition_estimate=jnp.full(status.shape, jnp.nan, dtype=rhs.real.dtype),
        singular_values=None,
    )


def release_spineax(state: SpineaxFactorState, /) -> bool:
    if not isinstance(state, SpineaxFactorState):
        raise TypeError("Spineax release requires SpineaxFactorState.")
    return _cudss().release(state.token)


__all__ = [
    "SpineaxBackendOutput",
    "SpineaxFactorState",
    "SpineaxSymbolicState",
    "analyze_spineax",
    "bind_spineax",
    "refresh_spineax",
    "release_spineax",
    "solve_spineax",
]
