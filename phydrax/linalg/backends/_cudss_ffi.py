#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
# Adapted algorithm copyright (c) 2025 John Viljoen, used under the MIT license.
# Token batching, structure fingerprints, and implicit derivatives adapted from
# the MIT-licensed cuDSS token algorithm; attribution is recorded in NOTICE.

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from functools import lru_cache
from math import prod
from typing import final, Literal

import equinox as eqx
import jax
import jax.core as jax_core
import jax.numpy as jnp
import numpy as np
from jax import Array, lax
from jax.custom_batching import custom_vmap

from ..._cuda_native import (
    load_cuda_component,
    release_cuda_entries,
    release_cuda_partition,
)
from ..._strict import StrictModule


type Config = tuple[int, int, int, int, int, int, int, int, tuple[int, ...]]
type Analyze = Callable[[Array, Array, Array], tuple[Array, Array]]
type Numeric = Callable[[Array, Array, Array, Array], tuple[Array, Array]]
type Solve = Callable[[Array, Array, Array, Array, Array], tuple[Array, Array]]
type Diagonal = Callable[[Array, Array, Array, Array], tuple[Array, Array]]


@final
class CudssToken(StrictModule):
    """A rebuildable, immutable state; ids are bounded native memo hints."""

    id: Array
    values: Array
    offsets: Array
    columns: Array
    native_rebuilds: Array
    analysis_memory_words: Array
    phase: Literal["analyzed", "factorized"] = eqx.field(static=True)
    dtype: np.dtype = eqx.field(static=True)
    n: int = eqx.field(static=True)
    nnz: int = eqx.field(static=True)
    batch_size: int = eqx.field(static=True)
    config: Config = eqx.field(static=True)


def _suffix(dtype: np.dtype, /) -> str:
    if dtype == jnp.dtype(jnp.float32):
        return "f32"
    if dtype == jnp.dtype(jnp.float64):
        return "f64"
    raise ValueError(f"cuDSS supports real float32/float64, not {dtype}.")


def _attributes(config: Config, /) -> dict[str, np.int64 | np.ndarray]:
    attributes: dict[str, np.int64 | np.ndarray] = {
        key: np.int64(value)
        for key, value in zip(
            (
                "partition",
                "capacity",
                "mtype",
                "mview",
                "reordering",
                "memory",
                "hybrid_device_limit",
                "group",
            ),
            config[:8],
            strict=True,
        )
    }
    attributes["devices"] = np.asarray(config[8], dtype=np.int64)
    return attributes


def _structure_fingerprint(offsets: Array, columns: Array, /) -> Array:
    off_weights = jnp.arange(offsets.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2654435761
    ) + jnp.uint32(0x9E3779B9)
    col_weights = jnp.arange(columns.shape[0], dtype=jnp.uint32) * jnp.uint32(
        2246822519
    ) + jnp.uint32(0x85EBCA6B)
    return jnp.stack(
        (
            jnp.sum(
                (offsets.astype(jnp.uint32) + jnp.uint32(1)) * off_weights,
                dtype=jnp.uint32,
            ),
            jnp.sum(
                (columns.astype(jnp.uint32) + jnp.uint32(1)) * col_weights,
                dtype=jnp.uint32,
            ),
        )
    )


def _expand_structure(
    offsets: Array, columns: Array, batch_size: int, /
) -> tuple[Array, Array, Array]:
    if (
        batch_size < 1
        or max(
            (offsets.shape[-1] - 1) * batch_size,
            columns.shape[-1] * batch_size,
        )
        > np.iinfo(np.int32).max
    ):
        raise ValueError("cuDSS expanded CSR dimensions overflow int32.")
    offsets = offsets.astype(jnp.int32)
    columns = columns.astype(jnp.int32)
    if batch_size == 1:
        expanded_offsets, expanded_columns = offsets.reshape(-1), columns.reshape(-1)
    else:
        n, nnz = offsets.shape[-1] - 1, columns.shape[-1]
        shift = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
        body = (offsets.reshape((-1, n + 1))[:, 1:] + shift * jnp.int32(nnz)).reshape(-1)
        expanded_offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), body))
        expanded_columns = (columns.reshape((-1, nnz)) + shift * jnp.int32(n)).reshape(-1)
    return (
        expanded_offsets,
        expanded_columns,
        _structure_fingerprint(expanded_offsets, expanded_columns),
    )


def _batch_of(token: CudssToken, /) -> int:
    return prod(token.id.shape[:-1]) * token.batch_size


def _call(
    operation: str,
    dtype: np.dtype,
    results: tuple[jax.ShapeDtypeStruct, ...],
    operands: tuple[Array, ...],
    config: Config,
    /,
    *,
    effect: bool,
    batch: int | None = None,
) -> tuple[Array, ...]:
    # Loading is deliberately before ffi_call construction, including tracing.
    load_cuda_component("cudss")
    attributes = _attributes(config)
    if batch is not None:
        attributes["batch"] = np.int64(batch)
    return tuple(
        jax.ffi.ffi_call(
            f"phydrax_cudss_{operation}_{_suffix(dtype)}",
            results,
            has_side_effect=effect,
        )(*operands, **attributes)
    )


def _count_spec() -> jax.ShapeDtypeStruct:
    return jax.ShapeDtypeStruct((1,), jnp.int32)


@lru_cache(maxsize=None)
def _make_analyze(config: Config) -> Analyze:
    @custom_vmap
    def analyze_id(values: Array, offsets: Array, columns: Array) -> tuple[Array, Array]:
        expanded_offsets, expanded_columns, fingerprint = _expand_structure(
            offsets, columns, 1
        )
        token_id, memory = _call(
            "analyze",
            values.dtype,
            (
                _count_spec(),
                jax.ShapeDtypeStruct((max(len(config[8]), 1), 6, 2), jnp.uint32),
            ),
            (values, expanded_offsets, expanded_columns, fingerprint),
            config,
            effect=True,
            batch=1,
        )
        return token_id, memory

    @analyze_id.def_vmap
    def rule(
        axis_size: int,
        batched: tuple[bool, ...],
        values: Array,
        offsets: Array,
        columns: Array,
    ) -> tuple[tuple[Array, Array], tuple[bool, bool]]:
        if config[8]:
            raise ValueError("cuDSS MG refuses matrix-batch vmap; use multiple RHS.")
        values = (
            values
            if batched[0]
            else jnp.broadcast_to(values, (axis_size,) + values.shape)
        )
        offsets, columns, _ = _expand_structure(offsets, columns, axis_size)
        token_id, memory = analyze_id(values.reshape(-1), offsets, columns)
        return (
            jnp.broadcast_to(token_id, (axis_size,) + token_id.shape),
            jnp.broadcast_to(memory, (axis_size,) + memory.shape),
        ), (True, True)

    return analyze_id


@lru_cache(maxsize=None)
def _make_numeric(refactor: bool, config: Config) -> Numeric:
    operation = "refactorize" if refactor else "factorize"

    @custom_vmap
    def numeric_id(
        token_id: Array, offsets: Array, columns: Array, values: Array
    ) -> tuple[Array, Array]:
        offsets, columns, fingerprint = _expand_structure(offsets, columns, 1)
        out_id, rebuilt = _call(
            operation,
            values.dtype,
            (_count_spec(), _count_spec()),
            (token_id, offsets, columns, fingerprint, values),
            config,
            effect=True,
        )
        return out_id, rebuilt

    @numeric_id.def_vmap
    def rule(
        axis_size: int,
        batched: tuple[bool, ...],
        token_id: Array,
        offsets: Array,
        columns: Array,
        values: Array,
    ) -> tuple[tuple[Array, Array], tuple[bool, bool]]:
        if config[8]:
            raise ValueError("cuDSS MG refuses matrix-batch vmap; use multiple RHS.")
        if not batched[0]:
            raise ValueError(
                f"cuDSS vmap({operation}) with an unbatched token and batched values "
                "is unsupported: one entry cannot hold a batch of factorizations; "
                "vmap(analyze) over the batch first."
            )
        values = (
            values
            if batched[3]
            else jnp.broadcast_to(values, (axis_size,) + values.shape)
        )
        offsets, columns, _ = _expand_structure(offsets, columns, axis_size)
        out_id, rebuilt = numeric_id(token_id[0], offsets, columns, values.reshape(-1))
        return (
            jnp.broadcast_to(out_id, (axis_size,) + out_id.shape),
            jnp.broadcast_to(rebuilt, (axis_size,) + rebuilt.shape),
        ), (True, True)

    return numeric_id


@lru_cache(maxsize=None)
def _make_solve(config: Config) -> Solve:
    @custom_vmap
    def solve_id(
        token_id: Array, offsets: Array, columns: Array, values: Array, rhs: Array
    ) -> tuple[Array, Array]:
        offsets, columns, fingerprint = _expand_structure(offsets, columns, 1)
        value, rebuilt = _call(
            "solve",
            values.dtype,
            (jax.ShapeDtypeStruct(rhs.shape, rhs.dtype), _count_spec()),
            (token_id, offsets, columns, fingerprint, values, rhs),
            config,
            effect=False,
        )
        return value, rebuilt

    @solve_id.def_vmap
    def rule(
        axis_size: int,
        batched: tuple[bool, ...],
        token_id: Array,
        offsets: Array,
        columns: Array,
        values: Array,
        rhs: Array,
    ) -> tuple[tuple[Array, Array], tuple[bool, bool]]:
        if config[8] and batched[0]:
            raise ValueError("cuDSS MG refuses matrix-batch vmap; use multiple RHS.")
        rhs = rhs if batched[4] else jnp.broadcast_to(rhs, (axis_size,) + rhs.shape)
        if not batched[0]:
            if batched[3] or batched[1] or batched[2]:
                raise ValueError(
                    "cuDSS solve refuses an unbatched token with batched values or pattern; vmap(analyze) first."
                )
            value, rebuilt = solve_id(token_id, offsets, columns, values, rhs)
            return (value, rebuilt), (True, False)
        offsets, columns, _ = _expand_structure(offsets, columns, axis_size)
        values = (
            values
            if batched[3]
            else jnp.broadcast_to(values, (axis_size,) + values.shape)
        )
        moved = jnp.moveaxis(rhs, 0, -2)
        value, rebuilt = solve_id(
            token_id[0],
            offsets,
            columns,
            values.reshape(-1),
            moved.reshape(moved.shape[:-2] + (-1,)),
        )
        return (
            jnp.moveaxis(value.reshape(moved.shape), -2, 0),
            jnp.broadcast_to(rebuilt, (axis_size,) + rebuilt.shape),
        ), (True, True)

    return solve_id


@lru_cache(maxsize=None)
def _make_diagonal(config: Config) -> Diagonal:
    @custom_vmap
    def diagonal_id(
        token_id: Array, offsets: Array, columns: Array, values: Array
    ) -> tuple[Array, Array]:
        offsets, columns, fingerprint = _expand_structure(offsets, columns, 1)
        diagonal, rebuilt = _call(
            "diagonal",
            values.dtype,
            (jax.ShapeDtypeStruct((offsets.shape[0] - 1,), values.dtype), _count_spec()),
            (token_id, offsets, columns, fingerprint, values),
            config,
            effect=False,
        )
        return diagonal, rebuilt

    @diagonal_id.def_vmap
    def rule(
        axis_size: int,
        batched: tuple[bool, ...],
        token_id: Array,
        offsets: Array,
        columns: Array,
        values: Array,
    ) -> tuple[tuple[Array, Array], tuple[bool, bool]]:
        if config[8]:
            raise ValueError("cuDSS MG refuses matrix-batch vmap.")
        if not batched[0]:
            raise ValueError(
                "cuDSS diagonal requires a batched token when values or pattern are batched."
            )
        n = offsets.shape[-1] - 1
        values = (
            values
            if batched[3]
            else jnp.broadcast_to(values, (axis_size,) + values.shape)
        )
        offsets, columns, _ = _expand_structure(offsets, columns, axis_size)
        diagonal, rebuilt = diagonal_id(token_id[0], offsets, columns, values.reshape(-1))
        return (
            diagonal.reshape(axis_size, n),
            jnp.broadcast_to(rebuilt, (axis_size,) + rebuilt.shape),
        ), (True, True)

    return diagonal_id


@lru_cache(maxsize=None)
def _make_analyze_ad(config: Config) -> Callable[[Array, Array, Array], CudssToken]:
    @jax.custom_jvp
    def analyze_ad(values: Array, offsets: Array, columns: Array) -> CudssToken:
        n, nnz = offsets.shape[-1] - 1, columns.shape[-1]
        batch_size = prod(values.shape[:-1])
        if values.ndim > 1:
            expanded_offsets, expanded_columns, _ = _expand_structure(
                offsets, columns, batch_size
            )
            token_id, memory = _make_analyze(config)(
                values.reshape(-1), expanded_offsets, expanded_columns
            )
        else:
            token_id, memory = _make_analyze(config)(values, offsets, columns)
        return CudssToken(
            token_id,
            values,
            offsets,
            columns,
            jnp.zeros_like(token_id),
            memory,
            "analyzed",
            values.dtype,
            n,
            nnz,
            batch_size,
            config,
        )

    @analyze_ad.defjvp
    def rule(
        primals: tuple[Array, Array, Array], tangents: tuple[Array, Array, Array]
    ) -> tuple[CudssToken, CudssToken]:
        token = analyze_ad(*primals)
        tangent = dataclasses.replace(
            token,
            values=tangents[0],
            id=np.zeros(token.id.shape, jax.dtypes.float0),
            offsets=np.zeros(token.offsets.shape, jax.dtypes.float0),
            columns=np.zeros(token.columns.shape, jax.dtypes.float0),
            native_rebuilds=np.zeros(token.native_rebuilds.shape, jax.dtypes.float0),
            analysis_memory_words=np.zeros(
                token.analysis_memory_words.shape, jax.dtypes.float0
            ),
        )
        return token, tangent

    return analyze_ad


def analyze(
    values: Array,
    offsets: Array,
    columns: Array,
    /,
    *,
    partition: int,
    capacity: int,
    reordering: str,
    memory: str,
    hybrid_device_limit: int | None = None,
    group: int = 0,
    devices: tuple[int, ...] = (),
) -> CudssToken:
    if devices and values.ndim != 1:
        raise ValueError("cuDSS MG refuses matrix batches; use multiple RHS.")
    load_cuda_component("cudss")
    _suffix(values.dtype)
    config = (
        partition,
        capacity,
        1,
        0,
        {"default": 0, "amd": 3, "nested_dissection": 4}[reordering],
        {"device": 0, "hybrid": 1}[memory],
        0 if hybrid_device_limit is None else hybrid_device_limit,
        group,
        devices,
    )
    return _make_analyze_ad(config)(
        values, offsets.astype(jnp.int32), columns.astype(jnp.int32)
    )


def _require_factorized(token: CudssToken, operation: str, /) -> None:
    if token.phase != "factorized":
        raise ValueError(
            f"cuDSS {operation} requires a factorized token; call factorize first."
        )


@lru_cache(maxsize=None)
def _make_numeric_ad(refactor: bool) -> Callable[[CudssToken, Array], CudssToken]:
    @jax.custom_jvp
    def numeric_ad(token: CudssToken, values: Array) -> CudssToken:
        batch_size = _batch_of(token)
        numeric = _make_numeric(refactor, token.config)
        if batch_size > 1 or token.id.ndim >= 2:
            offsets, columns, _ = _expand_structure(
                token.offsets, token.columns, batch_size
            )
            token_id, rebuilt = numeric(
                token.id.reshape(-1)[:1], offsets, columns, values.reshape(-1)
            )
            token_id = jnp.broadcast_to(token_id, token.id.shape)
        else:
            token_id, rebuilt = numeric(token.id, token.offsets, token.columns, values)
        return dataclasses.replace(
            token, id=token_id, values=values, native_rebuilds=rebuilt, phase="factorized"
        )

    @numeric_ad.defjvp
    def rule(
        primals: tuple[CudssToken, Array], tangents: tuple[CudssToken, Array]
    ) -> tuple[CudssToken, CudssToken]:
        out = numeric_ad(*primals)
        return out, dataclasses.replace(
            tangents[0],
            values=tangents[1],
            phase="factorized",
            native_rebuilds=np.zeros(out.native_rebuilds.shape, jax.dtypes.float0),
        )

    return numeric_ad


def _numeric(token: CudssToken, values: Array, refactor: bool, /) -> CudssToken:
    if refactor:
        _require_factorized(token, "refactorize")
    if values.dtype != token.dtype or values.shape[-1] != token.nnz:
        raise ValueError(
            "cuDSS numeric values must match the token dtype and per-block nnz."
        )
    batch_size = _batch_of(token)
    if (batch_size > 1 or token.id.ndim >= 2) and prod(values.shape[:-1]) != batch_size:
        raise ValueError(
            f"cuDSS numeric phase expects {batch_size} value blocks, got {values.shape}."
        )
    return _make_numeric_ad(refactor)(token, values)


def factorize(token: CudssToken, values: Array, /) -> CudssToken:
    return _numeric(token, values, False)


def refactorize(token: CudssToken, values: Array, /) -> CudssToken:
    return _numeric(token, values, True)


def _matvec(token: CudssToken, value: Array, /) -> Array:
    batch_size = _batch_of(token)
    offsets, columns, _ = _expand_structure(token.offsets, token.columns, batch_size)
    rows = jnp.repeat(
        jnp.arange(offsets.shape[0] - 1, dtype=jnp.int32),
        jnp.diff(offsets),
        total_repeat_length=batch_size * token.nnz,
    )
    flat = value.reshape(value.shape[:-2] + (-1,)) if batch_size > 1 else value
    out = (
        jnp.zeros_like(flat)
        .at[..., rows]
        .add(token.values.reshape(-1) * jnp.take(flat, columns, axis=-1))
    )
    return out.reshape(value.shape)


def _solve_impl(token: CudssToken, rhs: Array, /) -> tuple[Array, Array]:
    batch_size = _batch_of(token)
    solve_id = _make_solve(token.config)
    if batch_size > 1 or token.id.ndim >= 2:
        offsets, columns, _ = _expand_structure(token.offsets, token.columns, batch_size)
        value, rebuilt = solve_id(
            token.id.reshape(-1)[:1],
            offsets,
            columns,
            token.values.reshape(-1),
            rhs.reshape(rhs.shape[:-2] + (-1,)),
        )
        return value.reshape(rhs.shape), rebuilt
    return solve_id(token.id, token.offsets, token.columns, token.values, rhs)


def _refined_solve(token: CudssToken, rhs: Array, steps: int, /) -> tuple[Array, Array]:
    value, rebuilt = _solve_impl(token, rhs)
    for _ in range(steps):
        correction, count = _solve_impl(token, rhs - _matvec(token, value))
        value = value + correction
        rebuilt = rebuilt + count
    return value, rebuilt


def solve(
    token: CudssToken, rhs: Array, /, *, refinement_steps: int
) -> tuple[Array, Array]:
    _require_factorized(token, "solve")
    if rhs.dtype != token.dtype or rhs.shape[-1] != token.n:
        raise ValueError("cuDSS rhs must match the token dtype and trailing dimension.")
    batch_size = _batch_of(token)
    if (batch_size > 1 or token.id.ndim >= 2) and (
        rhs.ndim < 2 or rhs.shape[-2] != batch_size
    ):
        raise ValueError(
            f"cuDSS batch solve expects rhs (..., {batch_size}, {token.n}), got {rhs.shape}."
        )
    stopped = lax.stop_gradient(token)

    def matvec(value: Array) -> Array:
        return _matvec(token, value)

    def solve_fn(_matvec: Callable[[Array], Array], value: Array) -> tuple[Array, Array]:
        return _refined_solve(stopped, value, refinement_steps)

    return lax.custom_linear_solve(matvec, rhs, solve_fn, symmetric=True, has_aux=True)


def diagonal(token: CudssToken, /) -> tuple[Array, Array]:
    _require_factorized(token, "diagonal")
    if token.config[8]:
        raise ValueError("cuDSS MG does not expose a qualified diagonal query.")
    token = lax.stop_gradient(token)
    batch_size = _batch_of(token)
    diagonal_id = _make_diagonal(token.config)
    if batch_size > 1 or token.id.ndim >= 2:
        offsets, columns, _ = _expand_structure(token.offsets, token.columns, batch_size)
        return diagonal_id(
            token.id.reshape(-1)[:1], offsets, columns, token.values.reshape(-1)
        )
    return diagonal_id(token.id, token.offsets, token.columns, token.values)


def release(token: CudssToken, /) -> bool | Array:
    if token.config[8] and not isinstance(token.id, jax_core.Tracer):
        return release_cuda_partition("cudss", token.config[0]) > 0
    ids = token.id.reshape(-1)
    if isinstance(token.id, jax_core.Tracer):
        load_cuda_component("cudss")
        return (
            jax.ffi.ffi_call(
                "phydrax_cudss_release",
                _count_spec(),
                has_side_effect=True,
                vmap_method="broadcast_all",
            )(
                ids,
                partition=np.int64(token.config[0]),
                release_partition=np.int64(bool(token.config[8])),
            )[0]
            > 0
        )
    return release_cuda_entries("cudss", token.config[0], ids) > 0
