#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._strict import StrictModule
from ._policies import RankPolicy
from ._rank import numerical_rank_data


class DensePseudoinverseFactors(StrictModule):
    matrix: Array
    left_vectors: Array
    singular_values: Array
    right_adjoint: Array
    retained: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    finite: Array
    hermitian: bool = eqx.field(static=True)


def _economy_components(matrix: Array, *, hermitian: bool) -> tuple[Array, Array, Array]:
    result = jnp.linalg.svd(
        matrix, full_matrices=False, compute_uv=True, hermitian=hermitian
    )
    return result.U, result.S, result.Vh


def factor_pseudoinverse(
    matrix: Array,
    rank_policy: RankPolicy,
    /,
    *,
    hermitian: bool = False,
) -> DensePseudoinverseFactors:
    """Compute one batched economy factorization for Moore-Penrose operations."""
    value = jnp.asarray(matrix)
    if value.ndim < 2:
        raise ValueError("matrix must have at least two dimensions.")
    if not jnp.issubdtype(value.dtype, jnp.inexact):
        value = value.astype(jnp.float64)
    rows, columns = value.shape[-2:]
    if hermitian and rows != columns:
        raise ValueError("Hermitian pseudoinverse requires square matrices.")
    if hermitian:
        value = 0.5 * (value + jnp.conj(jnp.swapaxes(value, -1, -2)))
    left, singular_values, right_adjoint = _economy_components(value, hermitian=hermitian)
    rank = numerical_rank_data(singular_values, rows, columns, rank_policy)
    finite = rank.finite & jnp.all(jnp.isfinite(value), axis=(-2, -1))
    return DensePseudoinverseFactors(
        matrix=value,
        left_vectors=left,
        singular_values=singular_values,
        right_adjoint=right_adjoint,
        retained=rank.retained,
        rank=rank.rank,
        rank_cutoff=rank.cutoff,
        condition_estimate=rank.condition_estimate,
        finite=finite,
        hermitian=hermitian,
    )


def apply_pseudoinverse(
    factors: DensePseudoinverseFactors,
    rhs: Array,
    /,
) -> Array:
    """Apply A⁺ without materializing its dense coordinate matrix."""
    value = jnp.asarray(rhs)
    vector_rhs = value.ndim == factors.left_vectors.ndim - 1
    if vector_rhs:
        value = value[..., None]
    reciprocal = _reciprocal_singular_values(factors.singular_values, factors.retained)
    result = _apply_factor_columns(
        factors.left_vectors,
        reciprocal,
        factors.right_adjoint,
        value,
    )
    return result[..., 0] if vector_rhs else result


class ConnectedPseudoinverseResult(StrictModule):
    """Complete exact-component action with one global numerical-rank rule."""

    value: Array
    finite: Array
    resource_refused: Array
    work_units: Array
    rank: Array
    rank_cutoff: Array
    condition_estimate: Array
    component_sizes: Array
    component_buckets: Array
    factor_blocks: Array


class _ConnectedFactorGroup(NamedTuple):
    rows: Array
    counts: Array
    present: Array
    left: Array
    singular_values: Array
    right_adjoint: Array


def connected_pseudoinverse_graph_work(size: int, /) -> int:
    return 2 * size**3 + 24 * size**2


def connected_pseudoinverse_upper_work(size: int, rhs_columns: int, /) -> int:
    """Conservative full signature envelope, separate from executed receipts."""
    signatures = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
    minima = (1, 2, 3, 5, 9, 13, 17, 21, 25, 29)
    groups = sum(
        64 * (size // minimum) * signature**3
        + 48 * (size // minimum) * signature**2
        + 8 * (size // minimum) * signature * (size + rhs_columns)
        for signature, minimum in zip(signatures, minima, strict=True)
        if minimum <= size
    )
    return (
        connected_pseudoinverse_graph_work(size)
        + groups
        + 24 * size**2
        + 16 * size * rhs_columns
    )


def apply_connected_pseudoinverse(
    matrix: Array,
    rhs: Array,
    rank_policy: RankPolicy,
    valid_rows: Array,
    remaining_work: Array,
    /,
) -> ConnectedPseudoinverseResult:
    """Factor every exact-zero-connected component, retaining global cutoff.

    Connectivity uses original matrix entries AFTER the caller's equality
    projection, never SCI support or epsilon sparsity. Signature padding is
    excluded from the collected spectrum; true rank-deficient zeros remain.
    No factor cache, inverse, dropped component, or per-block rank decision.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Connected pseudoinverse requires one square matrix.")
    if matrix.dtype.kind != "f":
        raise TypeError("Connected Gram actions require real floating data.")
    size = matrix.shape[0]
    if size < 1 or size > 32 or valid_rows.shape != (size,):
        raise ValueError("Connected pseudoinverse requires at most32 matching rows.")
    if valid_rows.dtype.kind != "b":
        raise TypeError("Component validity must be Boolean.")
    if rhs.ndim not in (1, 2) or rhs.shape[0] != size:
        raise ValueError("Connected right-hand side must match matrix rows.")
    vector_rhs = rhs.ndim == 1
    columns = rhs[:, None] if vector_rhs else rhs
    graph_work = connected_pseudoinverse_graph_work(size)
    slots = jnp.arange(size)
    signatures = (1, 2, 4, 8, 12, 16, 20, 24, 28, 32)
    minima = (1, 2, 3, 5, 9, 13, 17, 21, 25, 29)

    def unavailable(work: Array, resource: Array) -> ConnectedPseudoinverseResult:
        return ConnectedPseudoinverseResult(
            value=jnp.zeros_like(rhs),
            finite=jnp.asarray(False),
            resource_refused=resource,
            work_units=work,
            rank=jnp.asarray(0, dtype=jnp.int32),
            rank_cutoff=jnp.asarray(jnp.nan, dtype=matrix.dtype),
            condition_estimate=jnp.asarray(jnp.inf, dtype=matrix.dtype),
            component_sizes=jnp.zeros((size,), dtype=jnp.int32),
            component_buckets=jnp.zeros((size,), dtype=jnp.int32),
            factor_blocks=jnp.asarray(0, dtype=jnp.int32),
        )

    def graph(_: None) -> ConnectedPseudoinverseResult:
        adjacency = (
            ((matrix != 0.0) | (matrix.T != 0.0))
            & valid_rows[:, None]
            & valid_rows[None, :]
        )
        adjacency = adjacency | (jnp.eye(size, dtype=jnp.bool_) & valid_rows[:, None])
        initial = jnp.where(valid_rows, slots, size)

        def propagate(_: int, labels: Array) -> Array:
            return jnp.min(jnp.where(adjacency, labels[None, :], size), axis=1)

        labels = jax.lax.fori_loop(0, size, propagate, initial)
        counts = (
            jnp.zeros((size,), dtype=jnp.int32)
            .at[jnp.where(valid_rows, labels, 0)]
            .add(valid_rows.astype(jnp.int32))
        )
        leaders = (counts > 0) & (labels == slots) & valid_rows
        sizes = jnp.where(leaders, counts, 0)
        buckets = jnp.zeros((size,), dtype=jnp.int32)
        for signature, minimum in zip(signatures, minima, strict=True):
            buckets = jnp.where(
                leaders & (counts >= minimum) & (counts <= signature), signature, buckets
            )
        finite_input = jnp.all(jnp.isfinite(matrix)) & jnp.all(jnp.isfinite(rhs))
        finite_input = finite_input & jnp.all(
            jnp.where(valid_rows[:, None] & valid_rows[None, :], True, matrix == 0.0)
        )
        factor_work = jnp.asarray(0, dtype=jnp.int64)
        for signature, minimum in zip(signatures, minima, strict=True):
            if minimum <= size:
                group_slots = size // minimum
                present = jnp.any(buckets == signature)
                units = (
                    64 * group_slots * signature**3
                    + 48 * group_slots * signature**2
                    + 8 * group_slots * signature * (size + columns.shape[1])
                )
                factor_work = factor_work + jnp.where(present, units, 0)
        global_work = 24 * size**2 + 16 * size * columns.shape[1]
        required = graph_work + factor_work + global_work

        def factor_all(_: None) -> ConnectedPseudoinverseResult:
            symmetric = 0.5 * (matrix + matrix.T)
            groups = []
            spectra = []
            for signature, minimum in zip(signatures, minima, strict=True):
                if minimum > size:
                    continue
                group_slots = size // minimum
                group_mask = buckets == signature
                group_count = jnp.sum(group_mask)

                def actual_factor(_: None) -> _ConnectedFactorGroup:
                    group_leaders = jnp.nonzero(
                        group_mask, size=group_slots, fill_value=0
                    )[0]
                    group_present = jnp.arange(group_slots) < group_count
                    group_counts = jnp.where(group_present, counts[group_leaders], 0)

                    def members(leader: Array) -> Array:
                        return jnp.nonzero(
                            valid_rows & (labels == leader), size=signature, fill_value=0
                        )[0]

                    rows = jax.vmap(members)(group_leaders)
                    row_valid = jnp.arange(signature)[None, :] < group_counts[:, None]
                    blocks = jnp.where(
                        row_valid[:, :, None] & row_valid[:, None, :],
                        symmetric[rows[:, :, None], rows[:, None, :]],
                        0.0,
                    )
                    left, singular, right = _economy_components(blocks, hermitian=True)
                    singular = jnp.where(row_valid, singular, 0.0)
                    return _ConnectedFactorGroup(
                        rows, group_counts, group_present, left, singular, right
                    )

                group = jax.lax.cond(
                    group_count > 0,
                    actual_factor,
                    lambda _: _ConnectedFactorGroup(
                        jnp.zeros((group_slots, signature), dtype=slots.dtype),
                        jnp.zeros((group_slots,), dtype=counts.dtype),
                        jnp.zeros((group_slots,), dtype=jnp.bool_),
                        jnp.zeros(
                            (group_slots, signature, signature), dtype=matrix.dtype
                        ),
                        jnp.zeros((group_slots, signature), dtype=matrix.dtype),
                        jnp.zeros(
                            (group_slots, signature, signature), dtype=matrix.dtype
                        ),
                    ),
                    None,
                )
                groups.append(group)
                spectra.append(group.singular_values.reshape((-1,)))
            spectrum = jnp.concatenate(tuple(spectra))
            global_rank = numerical_rank_data(spectrum, size, size, rank_policy)
            value = jnp.zeros_like(columns)
            finite_factors = global_rank.finite
            for group in groups:
                signature = group.singular_values.shape[1]

                def apply_group(_: None) -> Array:
                    row_valid = jnp.arange(signature)[None, :] < group.counts[:, None]
                    retained = (group.singular_values > global_rank.cutoff) & row_valid
                    reciprocal = _reciprocal_singular_values(
                        group.singular_values, retained
                    )
                    block_rhs = jnp.where(row_valid[:, :, None], columns[group.rows], 0.0)
                    block_value = _apply_factor_columns(
                        group.left, reciprocal, group.right_adjoint, block_rhs
                    )
                    block_value = jnp.where(row_valid[:, :, None], block_value, 0.0)
                    return jnp.zeros_like(columns).at[group.rows].add(block_value)

                value = value + jax.lax.cond(
                    jnp.any(group.present),
                    apply_group,
                    lambda _: jnp.zeros_like(columns),
                    None,
                )
                finite_factors = finite_factors & jnp.all(
                    jnp.isfinite(group.singular_values)
                )
            output = value[:, 0] if vector_rhs else value
            return ConnectedPseudoinverseResult(
                value=output,
                finite=finite_input & finite_factors & jnp.all(jnp.isfinite(output)),
                resource_refused=jnp.asarray(False),
                work_units=required,
                rank=global_rank.rank,
                rank_cutoff=global_rank.cutoff,
                condition_estimate=global_rank.condition_estimate,
                component_sizes=sizes,
                component_buckets=buckets,
                factor_blocks=jnp.sum(leaders, dtype=jnp.int32),
            )

        return jax.lax.cond(
            finite_input & (remaining_work >= required),
            factor_all,
            lambda _: unavailable(
                jnp.asarray(graph_work, dtype=jnp.int64),
                finite_input & (remaining_work < required),
            ),
            None,
        )

    return jax.lax.cond(
        remaining_work >= graph_work,
        graph,
        lambda _: unavailable(jnp.asarray(0, dtype=jnp.int64), jnp.asarray(True)),
        None,
    )


def _reciprocal_singular_values(singular_values: Array, retained: Array, /) -> Array:
    return jnp.where(retained, 1.0 / jnp.where(retained, singular_values, 1.0), 0.0)


def _apply_factor_columns(
    left: Array,
    reciprocal: Array,
    right_adjoint: Array,
    rhs: Array,
    /,
) -> Array:
    """Economy-factor action on coordinate columns, without a dense inverse."""
    projected = _adjoint(left) @ rhs
    return _adjoint(right_adjoint) @ (reciprocal[..., :, None] * projected)


def fixed_rank_pseudoinverse_action(
    matrix: Array,
    left: Array,
    singular_values: Array,
    right_adjoint: Array,
    retained: Array,
    rhs: Array,
    hermitian: bool,
    /,
) -> Array:
    """Fixed-rank solution action and derivative using existing economy factors."""
    if hermitian:
        matrix = 0.5 * (matrix + _adjoint(matrix))
    reciprocal = _reciprocal_singular_values(singular_values, retained)
    return _fixed_rank_pseudoinverse_action(matrix, left, reciprocal, right_adjoint, rhs)


@jax.custom_jvp
def _fixed_rank_pseudoinverse_action(
    matrix: Array,
    left: Array,
    reciprocal: Array,
    right_adjoint: Array,
    rhs: Array,
    /,
) -> Array:
    del matrix
    return _apply_factor_columns(left, reciprocal, right_adjoint, rhs)


@_fixed_rank_pseudoinverse_action.defjvp
def _fixed_rank_pseudoinverse_action_jvp(
    primals: tuple[Array, Array, Array, Array, Array],
    tangents: tuple[Array, Array, Array, Array, Array],
) -> tuple[Array, Array]:
    matrix, left, reciprocal, right_adjoint, rhs = primals
    matrix_tangent, _, _, _, rhs_tangent = tangents

    def apply(columns: Array) -> Array:
        return _fixed_rank_pseudoinverse_action(
            matrix, left, reciprocal, right_adjoint, columns
        )

    def apply_adjoint(columns: Array) -> Array:
        return _fixed_rank_pseudoinverse_action(
            _adjoint(matrix),
            _adjoint(right_adjoint),
            reciprocal,
            _adjoint(left),
            columns,
        )

    value = apply(rhs)
    residual = rhs - matrix @ value
    tangent_adjoint = _adjoint(matrix_tangent)
    null_direction = tangent_adjoint @ apply_adjoint(value)
    # d(P b) = P(db - dA x) + P P* dA* r + (I - P A) dA* P* x.
    # Apply the complements to columns: no source/target square buffer exists,
    # even for a one-row design with a large source nullspace.
    tangent = (
        apply(rhs_tangent - matrix_tangent @ value)
        + apply(apply_adjoint(tangent_adjoint @ residual))
        + null_direction
        - apply(matrix @ null_direction)
    )
    return value, tangent


def materialize_pseudoinverse(
    factors: DensePseudoinverseFactors,
    /,
) -> Array:
    """Materialize A⁺ directly from economy factors with one matrix product."""
    reciprocal = _reciprocal_singular_values(factors.singular_values, factors.retained)
    scaled_right = (
        jnp.conj(jnp.swapaxes(factors.right_adjoint, -1, -2)) * reciprocal[..., None, :]
    )
    value = jnp.matmul(
        scaled_right,
        jnp.conj(jnp.swapaxes(factors.left_vectors, -1, -2)),
    )
    return fixed_rank_pseudoinverse_value(
        factors.matrix,
        value,
        factors.hermitian,
    )


def fixed_rank_pseudoinverse_value(
    matrix: Array,
    value: Array,
    hermitian: bool,
    /,
) -> Array:
    return (
        _fixed_rank_hermitian_pseudoinverse(matrix, value)
        if hermitian
        else _fixed_rank_pseudoinverse(matrix, value)
    )


def _adjoint(value: Array, /) -> Array:
    return jnp.conj(jnp.swapaxes(value, -1, -2))


def _pseudoinverse_tangent(
    matrix: Array,
    pseudoinverse: Array,
    tangent: Array,
    /,
) -> Array:
    rows, columns = matrix.shape[-2:]
    pseudoinverse_adjoint = _adjoint(pseudoinverse)
    tangent_adjoint = _adjoint(tangent)
    target_identity = jnp.eye(rows, dtype=matrix.dtype)
    source_identity = jnp.eye(columns, dtype=matrix.dtype)
    target_complement = target_identity - matrix @ pseudoinverse
    source_complement = source_identity - pseudoinverse @ matrix
    return (
        -pseudoinverse @ tangent @ pseudoinverse
        + pseudoinverse @ pseudoinverse_adjoint @ tangent_adjoint @ target_complement
        + source_complement @ tangent_adjoint @ pseudoinverse_adjoint @ pseudoinverse
    )


@jax.custom_jvp
def _fixed_rank_pseudoinverse(matrix: Array, value: Array, /) -> Array:
    del matrix
    return value


@_fixed_rank_pseudoinverse.defjvp
def _fixed_rank_pseudoinverse_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    matrix, value = primals
    matrix_tangent, _ = tangents
    return value, _pseudoinverse_tangent(matrix, value, matrix_tangent)


@jax.custom_jvp
def _fixed_rank_hermitian_pseudoinverse(
    matrix: Array,
    value: Array,
    /,
) -> Array:
    del matrix
    return value


@_fixed_rank_hermitian_pseudoinverse.defjvp
def _fixed_rank_hermitian_pseudoinverse_jvp(
    primals: tuple[Array, Array], tangents: tuple[Array, Array]
) -> tuple[Array, Array]:
    matrix, value = primals
    matrix_tangent, _ = tangents
    matrix_tangent = 0.5 * (matrix_tangent + _adjoint(matrix_tangent))
    return value, _pseudoinverse_tangent(matrix, value, matrix_tangent)


__all__ = [
    "DensePseudoinverseFactors",
    "ConnectedPseudoinverseResult",
    "apply_connected_pseudoinverse",
    "connected_pseudoinverse_graph_work",
    "connected_pseudoinverse_upper_work",
    "apply_pseudoinverse",
    "factor_pseudoinverse",
    "materialize_pseudoinverse",
]
