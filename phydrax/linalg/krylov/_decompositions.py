#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ._results import (
    BlockKrylovDecomposition,
    GolubKahanDecomposition,
    KrylovBreakdownStatus,
    KrylovDecomposition,
)


Orthogonalization = Literal["modified", "double", "selective", "full"]
InnerProduct = Callable[[Array, Array], Array]


def _euclidean_inner(left: Array, right: Array, /) -> Array:
    return jnp.vdot(left, right)


def _norm_from_squared(squared: Array, /) -> Array:
    """Exact norm with the zero subgradient on an inactive zero residual.

    Happy breakdown closes the active Krylov subspace. Its discarded residual
    has zero cotangent, but differentiating sqrt(0) directly still forms 0/0 in
    reverse mode. Select a finite square-root argument before taking the root;
    the zero branch then has zero tangent without perturbing any positive norm.
    Nonfinite squared lengths remain nonfinite instead of becoming zero.
    """
    squared = jnp.real(squared)
    inactive = jnp.isfinite(squared) & (squared <= 0.0)
    root = jnp.sqrt(jnp.where(inactive, 1.0, squared))
    return jnp.where(inactive, 0.0, root)


def _norm(vector: Array, inner: InnerProduct, /) -> Array:
    return _norm_from_squared(inner(vector, vector))


def _breakdown_tolerance(
    value: float | None,
    dtype: Any,
    /,
) -> Array:
    if value is None:
        return jnp.sqrt(jnp.finfo(dtype).eps)
    scalar = float(value)
    if not math.isfinite(scalar) or scalar < 0.0 or scalar > float(jnp.finfo(dtype).max):
        raise ValueError("breakdown_tolerance must be finite and non-negative.")
    return jnp.asarray(scalar, dtype=dtype)


def _validate(
    initial: ArrayLike,
    max_dimension: int,
    orthogonalization: Orthogonalization,
    /,
) -> tuple[Array, int]:
    vector = jnp.asarray(initial)
    if vector.ndim != 1 or not jnp.issubdtype(vector.dtype, jnp.inexact):
        raise TypeError("The starting vector must be one real or complex vector.")
    dimension = int(max_dimension)
    if dimension < 1 or dimension > vector.size:
        raise ValueError("max_dimension must be in [1, vector.size].")
    if orthogonalization not in ("modified", "double", "selective", "full"):
        raise ValueError("Unknown orthogonalization policy.")
    return vector, dimension


def arnoldi(
    action: Callable[[Array], Array],
    initial: ArrayLike,
    /,
    *,
    max_dimension: int,
    inner: InnerProduct = _euclidean_inner,
    orthogonalization: Orthogonalization = "selective",
    breakdown_tolerance: float | None = None,
) -> KrylovDecomposition:
    """Breakdown-safe Arnoldi/Hessenberg decomposition with fixed capacity."""
    if not callable(action) or not callable(inner):
        raise TypeError("action and inner must be callable.")
    vector, dimension = _validate(initial, max_dimension, orthogonalization)
    tolerance = _breakdown_tolerance(breakdown_tolerance, vector.real.dtype)
    epsilon = jnp.finfo(vector.real.dtype).eps
    output = jax.eval_shape(action, vector)
    if (
        not isinstance(output, jax.ShapeDtypeStruct)
        or output.shape != vector.shape
        or output.dtype != vector.dtype
    ):
        raise ValueError("Arnoldi action must preserve the vector shape and dtype.")
    beta = _norm(vector, inner)
    safe_beta = jnp.where(beta > 0.0, beta, 1.0)
    basis = jnp.zeros((dimension + 1, vector.size), dtype=vector.dtype)
    basis = basis.at[0].set(vector / safe_beta)
    initial_finite = jnp.all(jnp.isfinite(vector)) & jnp.isfinite(beta)
    projected = jnp.zeros((dimension + 1, dimension), dtype=vector.dtype)
    initial_status = jnp.where(
        ~initial_finite,
        int(KrylovBreakdownStatus.NONFINITE_ACTION),
        jnp.where(
            beta > 0.0,
            int(KrylovBreakdownStatus.NONE),
            int(KrylovBreakdownStatus.RANK_DEFICIENT_START),
        ),
    ).astype(jnp.int32)
    initial_state = (
        basis,
        projected,
        jnp.zeros_like(vector),
        jnp.asarray(0, dtype=jnp.int32),
        initial_status,
        initial_finite & (beta > 0.0),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(index, state):
        basis_, projected_, residual, effective, status, active, matvecs = state

        def execute(operand):
            basis_i, projected_i, _, _, _, _, matvecs_i = operand
            candidate = action(basis_i[index])
            finite = jnp.all(jnp.isfinite(candidate))
            initial_norm = _norm(candidate, inner)
            coefficients = jax.vmap(lambda q: inner(q, candidate))(basis_i[:-1])
            mask = jnp.arange(dimension) <= index
            coefficients = jnp.where(mask, coefficients, 0)
            residual_i = candidate - jnp.sum(coefficients[:, None] * basis_i[:-1], axis=0)
            first_norm = _norm(residual_i, inner)
            repeat = orthogonalization in ("double", "full")
            if orthogonalization == "selective":
                repeat = first_norm < 0.717 * initial_norm

            def reorthogonalize(values):
                residual_value, coefficient_value = values
                correction = jax.vmap(lambda q: inner(q, residual_value))(basis_i[:-1])
                correction = jnp.where(mask, correction, 0)
                return (
                    residual_value - jnp.sum(correction[:, None] * basis_i[:-1], axis=0),
                    coefficient_value + correction,
                )

            residual_i, coefficients = jax.lax.cond(
                jnp.asarray(repeat),
                reorthogonalize,
                lambda values: values,
                (residual_i, coefficients),
            )
            residual_norm = _norm(residual_i, inner)
            threshold = tolerance * jnp.maximum(initial_norm, 1.0)
            breakdown = residual_norm <= threshold
            safe_norm = jnp.where(breakdown, 1.0, residual_norm)
            basis_i = basis_i.at[index + 1].set(residual_i / safe_norm)
            projected_i = projected_i.at[:, index].set(0)
            projected_i = projected_i.at[:-1, index].set(coefficients)
            projected_i = projected_i.at[index + 1, index].set(residual_norm)
            next_status = jnp.where(
                finite,
                jnp.where(
                    breakdown,
                    int(KrylovBreakdownStatus.HAPPY),
                    int(KrylovBreakdownStatus.NONE),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return (
                basis_i,
                projected_i,
                residual_i,
                jnp.asarray(index + 1, dtype=jnp.int32),
                next_status,
                finite & ~breakdown,
                matvecs_i + 1,
            )

        return jax.lax.cond(active, execute, lambda operand: operand, state)

    basis, projected, residual, effective, status, _, matvecs = jax.lax.fori_loop(
        0, dimension, step, initial_state
    )
    residual_norm = _norm(residual, inner)
    orthogonality = _orthogonality_error(basis[:-1], effective, inner)
    status = jnp.where(
        (status == int(KrylovBreakdownStatus.NONE))
        & (orthogonality > 100.0 * jnp.sqrt(epsilon)),
        int(KrylovBreakdownStatus.LOSS_OF_ORTHOGONALITY),
        status,
    )
    return KrylovDecomposition(
        basis=basis,
        projected=projected,
        residual_vector=residual,
        residual_norm=residual_norm,
        effective_dimension=effective,
        breakdown_status=status,
        orthogonality_error=orthogonality,
        matvec_count=matvecs,
        adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
        method="arnoldi",
        provenance="phydrax-native",
    )


def block_arnoldi(
    action: Callable[[Array], Array],
    initial: ArrayLike,
    /,
    *,
    max_blocks: int,
    inner: InnerProduct = _euclidean_inner,
    orthogonalization: Orthogonalization = "selective",
    breakdown_tolerance: float | None = None,
) -> BlockKrylovDecomposition:
    """Block Arnoldi decomposition with metric-aware rank deflation."""
    if not callable(action) or not callable(inner):
        raise TypeError("action and inner must be callable.")
    block = jnp.asarray(initial)
    if block.ndim != 2 or not jnp.issubdtype(block.dtype, jnp.inexact):
        raise TypeError("initial must be one real or complex coordinate matrix.")
    dimension, block_size = block.shape
    blocks = int(max_blocks)
    if block_size < 1 or blocks < 1 or blocks * block_size > dimension:
        raise ValueError("max_blocks * block_size must lie in [1, vector dimension].")
    if orthogonalization not in ("modified", "double", "selective", "full"):
        raise ValueError("Unknown orthogonalization policy.")
    tolerance = _breakdown_tolerance(breakdown_tolerance, block.real.dtype)
    output = jax.eval_shape(action, block)
    if (
        not isinstance(output, jax.ShapeDtypeStruct)
        or output.shape != block.shape
        or output.dtype != block.dtype
    ):
        raise ValueError(
            "Block Arnoldi action must preserve the coordinate-matrix shape and dtype."
        )

    first_basis, initial_factor, initial_rank = _orthonormalize_block(
        block, inner, tolerance
    )
    capacity = (blocks + 1) * block_size
    basis = jnp.zeros((dimension, capacity), dtype=block.dtype)
    basis = basis.at[:, :block_size].set(first_basis)
    projected = jnp.zeros(
        (capacity, blocks * block_size),
        dtype=block.dtype,
    )
    block_ranks = jnp.zeros((blocks + 1,), dtype=jnp.int32)
    block_ranks = block_ranks.at[0].set(initial_rank)
    initial_finite = jnp.all(jnp.isfinite(block))
    initial_status = jnp.where(
        ~initial_finite,
        int(KrylovBreakdownStatus.NONFINITE_ACTION),
        jnp.where(
            initial_rank == block_size,
            int(KrylovBreakdownStatus.NONE),
            int(KrylovBreakdownStatus.RANK_DEFICIENT_START),
        ),
    ).astype(jnp.int32)
    state = (
        basis,
        projected,
        block_ranks,
        initial_status,
        initial_finite & (initial_rank > 0),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(index, current):
        basis_, projected_, ranks_, status_, active_, count_ = current

        def execute(operand):
            basis_i, projected_i, ranks_i, status_i, _, count_i = operand
            start = index * block_size
            current_block = jax.lax.dynamic_slice(
                basis_i,
                (0, start),
                (dimension, block_size),
            )
            candidate = action(current_block)
            finite = jnp.all(jnp.isfinite(candidate))
            prior = basis_i[:, : blocks * block_size]
            column_indices = jnp.arange(blocks * block_size)
            prior_blocks = column_indices // block_size
            within_blocks = column_indices % block_size
            prior_mask = (prior_blocks <= index) & (within_blocks < ranks_i[prior_blocks])
            coefficients = _block_inner(prior, candidate, inner)
            coefficients = jnp.where(prior_mask[:, None], coefficients, 0)
            residual = candidate - prior @ coefficients
            first_norm = _block_norm(residual, inner)
            candidate_norm = _block_norm(candidate, inner)
            repeat = orthogonalization in ("double", "full")
            if orthogonalization == "selective":
                repeat = first_norm < 0.717 * candidate_norm

            def reorthogonalize(values):
                residual_value, coefficient_value = values
                correction = _block_inner(prior, residual_value, inner)
                correction = jnp.where(prior_mask[:, None], correction, 0)
                return (
                    residual_value - prior @ correction,
                    coefficient_value + correction,
                )

            residual, coefficients = jax.lax.cond(
                jnp.asarray(repeat),
                reorthogonalize,
                lambda values: values,
                (residual, coefficients),
            )
            next_basis, next_factor, next_rank = _orthonormalize_block(
                residual,
                inner,
                tolerance,
            )
            next_start = (index + 1) * block_size
            basis_i = jax.lax.dynamic_update_slice(
                basis_i,
                next_basis,
                (0, next_start),
            )
            projected_i = jax.lax.dynamic_update_slice(
                projected_i,
                coefficients,
                (0, start),
            )
            projected_i = jax.lax.dynamic_update_slice(
                projected_i,
                next_factor,
                (next_start, start),
            )
            ranks_i = ranks_i.at[index + 1].set(next_rank)
            next_status = jnp.where(
                finite,
                jnp.where(
                    next_rank == 0,
                    int(KrylovBreakdownStatus.HAPPY),
                    jnp.where(
                        next_rank < ranks_i[index],
                        int(KrylovBreakdownStatus.NEAR_BREAKDOWN),
                        status_i,
                    ),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return (
                basis_i,
                projected_i,
                ranks_i,
                next_status,
                finite & (next_rank > 0),
                count_i + ranks_i[index],
            )

        return jax.lax.cond(active_, execute, lambda operand: operand, current)

    basis, projected, block_ranks, status, _, matvecs = jax.lax.fori_loop(
        0,
        blocks,
        step,
        state,
    )
    active_columns = (
        jnp.arange(capacity) % block_size
        < block_ranks[jnp.arange(capacity) // block_size]
    )
    gram = _block_inner(basis, basis, inner)
    orthogonality_error = jnp.max(
        jnp.abs(
            jnp.where(
                active_columns[:, None] & active_columns[None, :],
                gram - jnp.eye(capacity, dtype=gram.dtype),
                0,
            )
        )
    )
    effective = jnp.sum(block_ranks[:-1], dtype=jnp.int32)
    return BlockKrylovDecomposition(
        basis=basis,
        projected=projected,
        initial_factor=initial_factor,
        block_ranks=block_ranks,
        effective_dimension=effective,
        breakdown_status=status,
        orthogonality_error=orthogonality_error,
        matvec_count=matvecs,
        block_size=block_size,
        max_blocks=blocks,
        method="block-arnoldi",
        provenance="phydrax-native",
    )


def lanczos(
    action: Callable[[Array], Array],
    initial: ArrayLike,
    /,
    *,
    max_dimension: int,
    inner: InnerProduct = _euclidean_inner,
    orthogonalization: Orthogonalization = "selective",
    breakdown_tolerance: float | None = None,
) -> KrylovDecomposition:
    """Hermitian Lanczos decomposition with optional full reorthogonalization."""
    if not callable(action) or not callable(inner):
        raise TypeError("action and inner must be callable.")
    vector, dimension = _validate(initial, max_dimension, orthogonalization)
    tolerance = _breakdown_tolerance(breakdown_tolerance, vector.real.dtype)
    epsilon = jnp.finfo(vector.real.dtype).eps
    output = jax.eval_shape(action, vector)
    if (
        not isinstance(output, jax.ShapeDtypeStruct)
        or output.shape != vector.shape
        or output.dtype != vector.dtype
    ):
        raise ValueError("Lanczos action must preserve the vector shape and dtype.")
    beta0 = _norm(vector, inner)
    initial_finite = jnp.all(jnp.isfinite(vector)) & jnp.isfinite(beta0)
    basis = jnp.zeros((dimension + 1, vector.size), dtype=vector.dtype)
    basis = basis.at[0].set(vector / jnp.where(beta0 > 0.0, beta0, 1.0))
    tridiagonal = jnp.zeros((dimension + 1, dimension), dtype=vector.dtype)
    state = (
        basis,
        tridiagonal,
        jnp.zeros_like(vector),
        jnp.asarray(0.0, dtype=vector.real.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(
            ~initial_finite,
            int(KrylovBreakdownStatus.NONFINITE_ACTION),
            jnp.where(
                beta0 > 0.0,
                int(KrylovBreakdownStatus.NONE),
                int(KrylovBreakdownStatus.RANK_DEFICIENT_START),
            ),
        ).astype(jnp.int32),
        initial_finite & (beta0 > 0.0),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(index, current):
        basis_, projected_, previous, previous_beta, effective, status, active, count = (
            current
        )

        def execute(operand):
            basis_i, projected_i, previous_i, beta_i, _, _, _, count_i = operand
            q = basis_i[index]
            candidate = action(q) - beta_i * previous_i
            alpha = jnp.real(inner(q, candidate))
            residual = candidate - alpha * q
            mask = jnp.arange(dimension) <= index
            if orthogonalization in ("double", "full", "selective"):
                correction = jax.vmap(lambda item: inner(item, residual))(basis_i[:-1])
                correction = jnp.where(mask, correction, 0)
                residual = residual - jnp.sum(correction[:, None] * basis_i[:-1], axis=0)
                alpha = alpha + jnp.real(correction[index])
            beta_next = _norm(residual, inner)
            threshold = tolerance * jnp.maximum(_norm(candidate, inner), 1.0)
            finite = jnp.all(jnp.isfinite(candidate))
            breakdown = beta_next <= threshold
            basis_i = basis_i.at[index + 1].set(
                residual / jnp.where(breakdown, 1.0, beta_next)
            )
            projected_i = projected_i.at[index, index].set(alpha)
            projected_i = jax.lax.cond(
                index > 0,
                lambda value: value.at[index - 1, index].set(beta_i),
                lambda value: value,
                projected_i,
            )
            projected_i = projected_i.at[index + 1, index].set(beta_next)
            status_i = jnp.where(
                finite,
                jnp.where(
                    breakdown,
                    int(KrylovBreakdownStatus.HAPPY),
                    int(KrylovBreakdownStatus.NONE),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return (
                basis_i,
                projected_i,
                q,
                beta_next,
                jnp.asarray(index + 1, dtype=jnp.int32),
                status_i,
                finite & ~breakdown,
                count_i + 1,
            )

        return jax.lax.cond(active, execute, lambda operand: operand, current)

    basis, projected, _, _, effective, status, _, matvecs = jax.lax.fori_loop(
        0, dimension, step, state
    )
    residual = projected[effective, jnp.maximum(effective - 1, 0)] * basis[effective]
    residual_norm = _norm(residual, inner)
    orthogonality = _orthogonality_error(basis[:-1], effective, inner)
    status = jnp.where(
        (status == int(KrylovBreakdownStatus.NONE))
        & (orthogonality > 100.0 * jnp.sqrt(epsilon)),
        int(KrylovBreakdownStatus.LOSS_OF_ORTHOGONALITY),
        status,
    )
    return KrylovDecomposition(
        basis=basis,
        projected=projected,
        residual_vector=residual,
        residual_norm=residual_norm,
        effective_dimension=effective,
        breakdown_status=status,
        orthogonality_error=orthogonality,
        matvec_count=matvecs,
        adjoint_matvec_count=jnp.asarray(0, dtype=jnp.int32),
        method="lanczos",
        provenance="phydrax-native",
    )


def golub_kahan(
    action: Callable[[Array], Array],
    adjoint_action: Callable[[Array], Array],
    initial: ArrayLike,
    /,
    *,
    max_dimension: int,
    left_inner: InnerProduct = _euclidean_inner,
    right_inner: InnerProduct = _euclidean_inner,
    breakdown_tolerance: float | None = None,
) -> GolubKahanDecomposition:
    """Pairing-aware Golub–Kahan bidiagonalization with fixed-capacity bases."""
    if not all(
        callable(item) for item in (action, adjoint_action, left_inner, right_inner)
    ):
        raise TypeError(
            "action, adjoint_action, left_inner, and right_inner must be callable."
        )
    left = jnp.asarray(initial)
    if left.ndim != 1 or not jnp.issubdtype(left.dtype, jnp.inexact):
        raise TypeError("initial must be one real or complex vector.")
    tolerance = _breakdown_tolerance(breakdown_tolerance, left.real.dtype)
    right_spec = jax.eval_shape(adjoint_action, left)
    if (
        not isinstance(right_spec, jax.ShapeDtypeStruct)
        or len(right_spec.shape) != 1
        or right_spec.dtype != left.dtype
    ):
        raise ValueError(
            "adjoint_action must return one vector with the initial vector dtype."
        )
    action_spec = jax.eval_shape(action, right_spec)
    if (
        not isinstance(action_spec, jax.ShapeDtypeStruct)
        or action_spec.shape != left.shape
        or action_spec.dtype != left.dtype
    ):
        raise ValueError("action must return the initial vector shape and dtype.")
    dimension = int(max_dimension)
    if dimension < 1 or dimension > min(left.size, right_spec.shape[0]):
        raise ValueError("max_dimension exceeds the bidiagonalization dimensions.")
    beta0 = _norm(left, left_inner)
    u_basis = jnp.zeros((dimension + 1, left.size), dtype=left.dtype)
    initial_finite = jnp.all(jnp.isfinite(left)) & jnp.isfinite(beta0)
    u_basis = u_basis.at[0].set(left / jnp.where(beta0 > 0.0, beta0, 1.0))
    v_basis = jnp.zeros((dimension, right_spec.shape[0]), dtype=right_spec.dtype)
    alpha = jnp.zeros((dimension,), dtype=left.real.dtype)
    beta = jnp.zeros((dimension,), dtype=left.real.dtype)
    state = (
        u_basis,
        v_basis,
        alpha,
        beta,
        jnp.zeros((right_spec.shape[0],), dtype=right_spec.dtype),
        jnp.asarray(0.0, dtype=left.real.dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.where(
            ~initial_finite,
            int(KrylovBreakdownStatus.NONFINITE_ACTION),
            jnp.where(
                beta0 > 0.0,
                int(KrylovBreakdownStatus.NONE),
                int(KrylovBreakdownStatus.RANK_DEFICIENT_START),
            ),
        ).astype(jnp.int32),
        initial_finite & (beta0 > 0.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def step(index, current):
        (
            ub,
            vb,
            diagonal,
            superdiagonal,
            previous_v,
            previous_beta,
            effective,
            status,
            active,
            matvecs,
            adjoint_matvecs,
        ) = current

        def execute(operand):
            (
                ub_i,
                vb_i,
                diagonal_i,
                super_i,
                previous_v_i,
                previous_beta_i,
                _,
                _,
                _,
                mat_i,
                adj_i,
            ) = operand
            candidate_v = adjoint_action(ub_i[index]) - previous_beta_i * previous_v_i
            alpha_i = _norm(candidate_v, right_inner)
            alpha_breakdown = alpha_i <= tolerance * jnp.maximum(
                _norm(candidate_v, right_inner), 1.0
            )
            v = candidate_v / jnp.where(alpha_breakdown, 1.0, alpha_i)
            candidate_u = action(v) - alpha_i * ub_i[index]
            beta_i = _norm(candidate_u, left_inner)
            breakdown = alpha_breakdown | (
                beta_i <= tolerance * jnp.maximum(_norm(candidate_u, left_inner), 1.0)
            )
            finite = jnp.all(jnp.isfinite(candidate_v)) & jnp.all(
                jnp.isfinite(candidate_u)
            )
            vb_i = vb_i.at[index].set(v)
            ub_i = ub_i.at[index + 1].set(candidate_u / jnp.where(breakdown, 1.0, beta_i))
            diagonal_i = diagonal_i.at[index].set(alpha_i)
            super_i = super_i.at[index].set(beta_i)
            status_i = jnp.where(
                finite,
                jnp.where(
                    breakdown,
                    int(KrylovBreakdownStatus.HAPPY),
                    int(KrylovBreakdownStatus.NONE),
                ),
                int(KrylovBreakdownStatus.NONFINITE_ACTION),
            ).astype(jnp.int32)
            return (
                ub_i,
                vb_i,
                diagonal_i,
                super_i,
                v,
                beta_i,
                jnp.asarray(index + 1, dtype=jnp.int32),
                status_i,
                finite & ~breakdown,
                mat_i + 1,
                adj_i + 1,
            )

        return jax.lax.cond(active, execute, lambda operand: operand, current)

    (
        u_basis,
        v_basis,
        diagonal,
        superdiagonal,
        _,
        _,
        effective,
        status,
        _,
        matvecs,
        adjoint_matvecs,
    ) = jax.lax.fori_loop(0, dimension, step, state)
    return GolubKahanDecomposition(
        left_basis=u_basis,
        right_basis=v_basis,
        diagonal=diagonal,
        superdiagonal=superdiagonal,
        effective_dimension=effective,
        breakdown_status=status,
        left_orthogonality_error=_orthogonality_error(
            u_basis[:-1], effective, left_inner
        ),
        right_orthogonality_error=_orthogonality_error(v_basis, effective, right_inner),
        matvec_count=matvecs,
        adjoint_matvec_count=adjoint_matvecs,
        provenance="phydrax-native",
    )


def _block_inner(left: Array, right: Array, inner: InnerProduct, /) -> Array:
    return jax.vmap(
        lambda left_column: jax.vmap(
            lambda right_column: inner(left_column, right_column),
            in_axes=1,
        )(right),
        in_axes=1,
    )(left)


def _block_norm(block: Array, inner: InnerProduct, /) -> Array:
    gram = _block_inner(block, block, inner)
    return _norm_from_squared(jnp.trace(gram))


def _orthonormalize_block(
    block: Array,
    inner: InnerProduct,
    tolerance: Array,
    /,
) -> tuple[Array, Array, Array]:
    gram = _block_inner(block, block, inner)
    gram = 0.5 * (gram + jnp.conj(gram.T))
    eigenvalues, eigenvectors = jnp.linalg.eigh(gram)
    order = jnp.argsort(eigenvalues)[::-1]
    eigenvalues = jnp.maximum(jnp.real(eigenvalues[order]), 0.0)
    eigenvectors = eigenvectors[:, order]
    singular_values = jnp.sqrt(eigenvalues)
    largest = singular_values[0]
    valid_scale = jnp.isfinite(largest) & (largest > 0.0)
    threshold = tolerance * jnp.where(valid_scale, largest, 1.0)
    active = valid_scale & jnp.isfinite(singular_values) & (singular_values > threshold)
    safe = jnp.where(active, singular_values, 1.0)
    transform = eigenvectors / safe[None, :]
    basis = block @ transform
    basis = jnp.where(active[None, :], basis, 0)
    factor = jnp.diag(jnp.where(active, singular_values, 0)) @ jnp.conj(eigenvectors.T)
    return basis, factor, jnp.sum(active, dtype=jnp.int32)


def _orthogonality_error(
    basis: Array,
    effective_dimension: Array,
    inner: InnerProduct,
    /,
) -> Array:
    gram = jax.vmap(lambda left: jax.vmap(lambda right: inner(left, right))(basis))(basis)
    indices = jnp.arange(basis.shape[0])
    active = indices < effective_dimension
    mask = active[:, None] & active[None, :]
    identity = jnp.eye(basis.shape[0], dtype=gram.dtype)
    error = jnp.where(mask, gram - identity, 0)
    return jnp.max(jnp.abs(error))


__all__ = [
    "Orthogonalization",
    "arnoldi",
    "block_arnoldi",
    "golub_kahan",
    "lanczos",
]
