# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ._operators import DenseLinearOperator
from ._rank import numerical_rank_data
from ._spaces import AbstractVectorSpace
from ._svd_contracts import (
    SVDLeadingEvidence,
    SVDProblem,
    SVDRankEvidence,
    SVDSolvePlan,
    SVDTarget,
)


def scaled_column_norms(columns: Array, /) -> Array:
    """Keep audit norms representable when squaring coordinates would underflow."""
    magnitudes = jnp.maximum(jnp.abs(jnp.real(columns)), jnp.abs(jnp.imag(columns)))
    scale = jnp.max(magnitudes, axis=0)
    safe = jnp.where(scale > 0, scale, 1).astype(columns.dtype)
    normalized = columns / safe[None, :]
    squares = jnp.real(normalized) ** 2 + jnp.imag(normalized) ** 2
    return scale * jnp.sqrt(jnp.sum(squares, axis=0))


def column_norms(space: AbstractVectorSpace, columns: Array, /) -> Array:
    def norm(column: Array) -> Array:
        magnitude = jnp.maximum(jnp.abs(jnp.real(column)), jnp.abs(jnp.imag(column)))
        scale = jnp.max(magnitude)
        safe = jnp.where(scale > 0, scale, 1).astype(column.dtype)
        value = space.unflatten(column / safe)
        return scale * jnp.sqrt(jnp.maximum(jnp.real(space.inner(value, value)), 0))

    return jax.vmap(norm, in_axes=1)(columns)


def orthogonality_error(space: AbstractVectorSpace, columns: Array, /) -> Array:
    def row(left: Array) -> Array:
        return jax.vmap(
            lambda right: space.inner(space.unflatten(left), space.unflatten(right)),
            in_axes=1,
        )(columns)

    gram = jax.vmap(row, in_axes=1)(columns)
    return jnp.max(jnp.abs(gram - jnp.eye(columns.shape[1], dtype=gram.dtype)))


def triplet_evidence(
    problem: SVDProblem,
    left: Array,
    values: Array,
    right: Array,
    operator_bound: Array,
    /,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    operator = problem.operator
    forward, backward = operator.mv_block(right), operator.adjoint_mv_block(left)
    left_residual = column_norms(
        operator.target, forward - left * values.astype(left.dtype)[None, :]
    )
    right_residual = column_norms(
        operator.source, backward - right * values.astype(right.dtype)[None, :]
    )
    left_norm, right_norm = (
        column_norms(operator.target, left),
        column_norms(operator.source, right),
    )
    tiny = jnp.finfo(values.dtype).tiny
    relative = jnp.maximum(
        left_residual
        / jnp.maximum(operator_bound * right_norm + values * left_norm, tiny),
        right_residual
        / jnp.maximum(operator_bound * left_norm + values * right_norm, tiny),
    )
    return (
        left_residual,
        right_residual,
        relative,
        orthogonality_error(operator.target, left),
        orthogonality_error(operator.source, right),
        column_norms(operator.target, forward) ** 2,
    )


def resident_range_residual(
    operator: DenseLinearOperator,
    basis: Array,
    source_scale: Array,
    target_scale: Array,
    /,
) -> tuple[Array, Array]:
    """Direct tiled residual, stable scaled sums, no duplicate transformed matrix."""
    matrix = operator.matrix
    rows, columns = matrix.shape
    tile = min(32, columns)
    steps = (columns + tile - 1) // tile

    def step(
        index: int, carry: tuple[Array, Array, Array, Array]
    ) -> tuple[Array, Array, Array, Array]:
        coordinates = index * tile + jnp.arange(tile)
        mask = coordinates < columns
        block = jnp.take(matrix, coordinates, axis=1, mode="clip")
        block = (
            target_scale[:, None]
            * block
            / jnp.take(source_scale, coordinates, mode="clip")[None, :]
        )
        block = jnp.where(mask[None, :], block, 0)
        residual = block - basis @ (basis.conj().T @ block)

        def accumulate(scale: Array, squares: Array, value: Array) -> tuple[Array, Array]:
            magnitude = jnp.max(jnp.abs(value))
            new_scale = jnp.maximum(scale, magnitude)
            safe = jnp.where(new_scale > 0, new_scale, 1)
            return new_scale, squares * (scale / safe) ** 2 + jnp.sum(
                (jnp.abs(value) / safe) ** 2
            )

        scale, squares = accumulate(carry[0], carry[1], residual)
        total_scale, total_squares = accumulate(carry[2], carry[3], block)
        return scale, squares, total_scale, total_squares

    zero = jnp.asarray(0, matrix.real.dtype)
    scale, squares, total_scale, total_squares = jax.lax.fori_loop(
        0, steps, step, (zero, zero, zero, zero)
    )
    return scale * jnp.sqrt(squares), (total_scale * jnp.sqrt(total_squares)) ** 2


def version_failure_probability(
    delta: float, version: Array, dtype: jnp.dtype, /
) -> Array:
    # Floating additions/divisions avoid integer products and version+2 overflow.
    value = version.astype(dtype)
    probability = jnp.asarray(delta, dtype) / (value + 1) / (value + 2)
    conservative = probability * (1 - 8 * jnp.finfo(dtype).eps)
    return eqx.error_if(
        conservative,
        conservative < jnp.finfo(dtype).tiny,
        "SVD audit lifetime allowance is not representable.",
    )


def gaussian_range_bound(
    maximum_norm: Array, probability: Array, count: int, complex_probe: bool, /
) -> Array:
    root = jnp.exp(jnp.log(probability) / count)
    if complex_probe:
        return maximum_norm / jnp.sqrt(-jnp.log1p(-root))
    return jnp.sqrt(jnp.asarray(2 / jnp.pi, maximum_norm.dtype)) * maximum_norm / root


def compressed_spectrum_allowance(
    values: Array, problem: SVDProblem, plan: SVDSolvePlan, /
) -> Array:
    if plan.certificate_kind == "exact-spectrum":
        return jnp.asarray(0, values.dtype)
    return (
        plan.policy.approximation.numerical_allowance
        * max(problem.operator.target.size, problem.operator.source.size)
        * jnp.finfo(values.dtype).eps
        * values[0]
    )


def rank_evidence(
    values: Array,
    eta: Array,
    available: Array,
    problem: SVDProblem,
    plan: SVDSolvePlan,
    probability: Array,
    /,
) -> SVDRankEvidence:
    rows, columns = problem.operator.target.size, problem.operator.source.size
    full = values.shape[0] == problem.maximum_rank
    exact = plan.certificate_kind == "exact-spectrum"
    policy = plan.policy.rank
    rank = numerical_rank_data(values, rows, columns, policy)
    relative = (
        max(rows, columns) * jnp.finfo(values.dtype).eps
        if policy.relative_cutoff is None
        else policy.relative_cutoff
    )
    absolute = 0 if policy.absolute_cutoff is None else policy.absolute_cutoff
    allowance = compressed_spectrum_allowance(values, problem, plan)
    low = jnp.asarray(absolute, values.dtype) + relative * jnp.maximum(
        values[0] - allowance, 0
    )
    high = jnp.asarray(absolute, values.dtype) + relative * (values[0] + eta + allowance)
    lower = jnp.sum(values - allowance > high, dtype=jnp.int32)
    upper = jnp.sum(values + eta + allowance > low, dtype=jnp.int32)
    upper = upper + jnp.where(eta > low, problem.maximum_rank - values.shape[0], 0)
    if exact:
        lower, upper, low, high = rank.rank, rank.rank, rank.cutoff, rank.cutoff
    deterministic = exact or (full and plan.certificate_kind == "deterministic-frobenius")
    exact_available = available & deterministic & (lower == upper)
    return SVDRankEvidence(
        jnp.where(available, lower, 0),
        jnp.where(
            available, jnp.minimum(upper, problem.maximum_rank), problem.maximum_rank
        ),
        jnp.where(available, low, jnp.nan),
        jnp.where(available, high, jnp.nan),
        available,
        full,
        exact_available,
        plan.certificate_kind,
        probability,
    )


def discarded_singular_maximum(values: Array, count: int, which: SVDTarget, /) -> Array:
    if count == values.shape[0]:
        return jnp.asarray(0, dtype=values.dtype)
    match which:
        case "largest":
            return values[count]
        case "smallest":
            return values[0]
        case _:
            raise ValueError("Unknown singular-value selection.")


def leading_evidence(
    values: Array,
    eta: Array,
    count: int,
    full: bool,
    available: Array,
    allowance: Array,
    which: SVDTarget,
    /,
) -> SVDLeadingEvidence:
    tail = discarded_singular_maximum(values, count, which) + eta + allowance
    selection_is_leading = which == "largest" or count == values.shape[0]
    smallest_selected = values[count - 1] if which == "largest" else values[-1]
    gap = smallest_selected - tail - allowance
    certified = (
        available
        & selection_is_leading
        & ((gap > 0) | (full & (count == values.shape[0])))
    )
    return SVDLeadingEvidence(certified, gap, tail, full)


def spectral_margins(
    values: Array, indices: Array, left_size: int, right_size: int, /
) -> tuple[Array, Array]:
    selected = values[indices]
    distances = jnp.abs(selected[:, None] - values[None, :])
    members = indices[:, None] == jnp.arange(values.shape[0])[None, :]
    isolation = jnp.min(jnp.where(members, jnp.inf, distances), axis=1)
    if max(left_size, right_size) > values.shape[0]:
        isolation = jnp.minimum(isolation, selected)
    selected_members = jnp.any(members, axis=0)
    boundary = jnp.min(
        jnp.where(
            selected_members[None, :],
            jnp.inf,
            jnp.abs(selected[:, None] ** 2 - values[None, :] ** 2),
        )
    )
    if max(left_size, right_size) > values.shape[0]:
        boundary = jnp.minimum(boundary, jnp.min(selected**2))
    allowance = (
        64
        * max(left_size, right_size)
        * jnp.finfo(values.dtype).eps
        * jnp.maximum(values[0], 1)
    )
    return isolation - allowance, boundary - allowance * jnp.maximum(values[0], 1)
