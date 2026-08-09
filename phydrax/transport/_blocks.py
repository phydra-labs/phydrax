#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._geometry import (
    block_count as _block_count,
    column_logsumexp as _geometry_column_logsumexp,
    cost_block as _geometry_cost_block,
    indices as _indices,
    row_logsumexp as _geometry_row_logsumexp,
)
from ._problem import DiscreteTransportProblem


Direction = Literal["source_to_target", "target_to_source"]


def row_logsumexp(
    problem: DiscreteTransportProblem,
    log_values: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    """Compute row-wise log-sum-exp through the shared finite-cost geometry."""
    return _geometry_row_logsumexp(
        problem.cost,
        problem.source.points,
        problem.target.points,
        log_values,
        epsilon,
        block_size=block_size,
    )


def column_logsumexp(
    problem: DiscreteTransportProblem,
    log_values: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> Array:
    """Compute column-wise log-sum-exp through the shared finite-cost geometry."""
    return _geometry_column_logsumexp(
        problem.cost,
        problem.source.points,
        problem.target.points,
        log_values,
        epsilon,
        block_size=block_size,
    )


def coupling_statistics(
    problem: DiscreteTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return probability marginals, cost, KL, mass, and finite flag."""
    source_log_weights = _safe_log(problem.source_probabilities)
    target_log_weights = _safe_log(problem.target_probabilities)
    if block_size is None:
        costs = problem.cost_matrix()
        log_ratio = (
            source_potential[:, None]
            + target_potential[None, :]
            - costs
        ) / epsilon
        log_plan = (
            source_log_weights[:, None]
            + target_log_weights[None, :]
            + log_ratio
        )
        plan = jnp.exp(log_plan)
        source_marginal = jnp.sum(plan, axis=1)
        target_marginal = jnp.sum(plan, axis=0)
        plan_mass = jnp.sum(plan)
        transport_cost = jnp.sum(plan * costs)
        safe_ratio = jnp.where(jnp.isfinite(log_ratio), log_ratio, 0.0)
        kl = jnp.sum(plan * safe_ratio) - plan_mass + 1.0
        finite = (
            jnp.all(jnp.isfinite(source_marginal))
            & jnp.all(jnp.isfinite(target_marginal))
            & jnp.isfinite(transport_cost)
            & jnp.isfinite(kl)
            & jnp.isfinite(plan_mass)
        )
        return (
            source_marginal,
            target_marginal,
            transport_cost,
            kl,
            plan_mass,
            finite,
        )
    return _blockwise_statistics(
        problem,
        source_potential,
        target_potential,
        epsilon,
        source_log_weights,
        target_log_weights,
        block_size=int(block_size),
    )


def dense_plan(
    problem: DiscreteTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    /,
) -> Array:
    """Materialize the physical transport plan."""
    costs = problem.cost_matrix()
    log_plan = (
        _safe_log(problem.source_probabilities)[:, None]
        + _safe_log(problem.target_probabilities)[None, :]
        + (
            source_potential[:, None]
            + target_potential[None, :]
            - costs
        )
        / epsilon
    )
    return problem.mass * jnp.exp(log_plan)


def apply_plan(
    problem: DiscreteTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    values: Array,
    /,
    *,
    direction: Direction,
    block_size: int | None,
) -> Array:
    """Apply the physical coupling without requiring dense materialization."""
    values_ = jnp.asarray(values)
    source_count, target_count = problem.shape
    if direction == "source_to_target":
        if values_.ndim < 1 or values_.shape[0] != source_count:
            raise ValueError("Source-to-target values must begin with source atom count.")
        output_count = target_count
    elif direction == "target_to_source":
        if values_.ndim < 1 or values_.shape[0] != target_count:
            raise ValueError("Target-to-source values must begin with target atom count.")
        output_count = source_count
    else:
        raise ValueError("direction must be 'source_to_target' or 'target_to_source'.")
    payload_shape = values_.shape[1:]
    flat_values = values_.reshape((values_.shape[0], -1))
    if block_size is None:
        plan = dense_plan(
            problem,
            source_potential,
            target_potential,
            epsilon,
        )
        output = (
            plan.T @ flat_values
            if direction == "source_to_target"
            else plan @ flat_values
        )
        return output.reshape((output_count,) + payload_shape)
    output = _blockwise_apply(
        problem,
        source_potential,
        target_potential,
        epsilon,
        flat_values,
        direction=direction,
        block_size=int(block_size),
    )
    return output.reshape((output_count,) + payload_shape)


def _blockwise_statistics(
    problem: DiscreteTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    source_log_weights: Array,
    target_log_weights: Array,
    /,
    *,
    block_size: int,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    source_count, target_count = problem.shape
    source_blocks = _block_count(source_count, block_size)
    target_blocks = _block_count(target_count, block_size)
    source_padded = source_blocks * block_size
    target_padded = target_blocks * block_size
    source_marginal = jnp.zeros((source_padded,), dtype=source_potential.dtype)
    target_marginal = jnp.zeros((target_padded,), dtype=target_potential.dtype)
    initial = (
        source_marginal,
        target_marginal,
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(True),
    )

    def source_body(source_block, state):
        source_result, target_result, cost_total, ratio_total, mass_total, finite = state
        source_start = source_block * block_size
        source_indices, source_valid = _indices(
            source_start, block_size, source_count
        )
        source_block_marginal = jnp.zeros(
            (block_size,), dtype=source_potential.dtype
        )

        def target_body(target_block, inner):
            (
                source_accumulator,
                target_accumulator,
                cost_accumulator,
                ratio_accumulator,
                mass_accumulator,
                finite_accumulator,
            ) = inner
            target_start = target_block * block_size
            target_indices, target_valid = _indices(
                target_start, block_size, target_count
            )
            costs = _cost_block(problem, source_indices, target_indices)
            f = jnp.take(source_potential, source_indices, axis=0)
            g = jnp.take(target_potential, target_indices, axis=0)
            log_a = jnp.take(source_log_weights, source_indices, axis=0)
            log_b = jnp.take(target_log_weights, target_indices, axis=0)
            valid = source_valid[:, None] & target_valid[None, :]
            log_ratio = (f[:, None] + g[None, :] - costs) / epsilon
            log_plan = log_a[:, None] + log_b[None, :] + log_ratio
            plan = jnp.where(valid, jnp.exp(log_plan), 0.0)
            safe_ratio = jnp.where(valid & jnp.isfinite(log_ratio), log_ratio, 0.0)
            target_slice = jax.lax.dynamic_slice(
                target_accumulator,
                (target_start,),
                (block_size,),
            )
            target_accumulator = jax.lax.dynamic_update_slice(
                target_accumulator,
                target_slice + jnp.sum(plan, axis=0),
                (target_start,),
            )
            block_finite = (
                jnp.all(jnp.isfinite(plan))
                & jnp.all(jnp.isfinite(jnp.where(valid, costs, 0.0)))
            )
            return (
                source_accumulator + jnp.sum(plan, axis=1),
                target_accumulator,
                cost_accumulator + jnp.sum(plan * costs),
                ratio_accumulator + jnp.sum(plan * safe_ratio),
                mass_accumulator + jnp.sum(plan),
                finite_accumulator & block_finite,
            )

        inner = jax.lax.fori_loop(
            0,
            target_blocks,
            target_body,
            (
                source_block_marginal,
                target_result,
                cost_total,
                ratio_total,
                mass_total,
                finite,
            ),
        )
        (
            source_block_marginal,
            target_result,
            cost_total,
            ratio_total,
            mass_total,
            finite,
        ) = inner
        source_result = jax.lax.dynamic_update_slice(
            source_result,
            source_block_marginal,
            (source_start,),
        )
        return (
            source_result,
            target_result,
            cost_total,
            ratio_total,
            mass_total,
            finite,
        )

    (
        source_marginal,
        target_marginal,
        transport_cost,
        ratio_term,
        plan_mass,
        finite,
    ) = jax.lax.fori_loop(0, source_blocks, source_body, initial)
    kl = ratio_term - plan_mass + 1.0
    finite = finite & jnp.isfinite(transport_cost) & jnp.isfinite(kl)
    return (
        source_marginal[:source_count],
        target_marginal[:target_count],
        transport_cost,
        kl,
        plan_mass,
        finite,
    )


def _blockwise_apply(
    problem: DiscreteTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    values: Array,
    /,
    *,
    direction: Direction,
    block_size: int,
) -> Array:
    source_count, target_count = problem.shape
    source_blocks = _block_count(source_count, block_size)
    target_blocks = _block_count(target_count, block_size)
    payload_size = values.shape[1]
    source_log_weights = _safe_log(problem.source_probabilities)
    target_log_weights = _safe_log(problem.target_probabilities)
    if direction == "source_to_target":
        output = jnp.zeros(
            (target_blocks * block_size, payload_size), dtype=values.dtype
        )
    else:
        output = jnp.zeros(
            (source_blocks * block_size, payload_size), dtype=values.dtype
        )

    def source_body(source_block, result):
        source_start = source_block * block_size
        source_indices, source_valid = _indices(
            source_start, block_size, source_count
        )
        f = jnp.take(source_potential, source_indices, axis=0)
        log_a = jnp.take(source_log_weights, source_indices, axis=0)
        if direction == "source_to_target":
            source_values = jnp.take(values, source_indices, axis=0)
        else:
            source_output = jnp.zeros((block_size, payload_size), dtype=values.dtype)

        def target_body(target_block, inner_result):
            target_start = target_block * block_size
            target_indices, target_valid = _indices(
                target_start, block_size, target_count
            )
            g = jnp.take(target_potential, target_indices, axis=0)
            log_b = jnp.take(target_log_weights, target_indices, axis=0)
            costs = _cost_block(problem, source_indices, target_indices)
            valid = source_valid[:, None] & target_valid[None, :]
            log_plan = (
                log_a[:, None]
                + log_b[None, :]
                + (f[:, None] + g[None, :] - costs) / epsilon
            )
            plan = problem.mass * jnp.where(valid, jnp.exp(log_plan), 0.0)
            if direction == "source_to_target":
                target_slice = jax.lax.dynamic_slice(
                    inner_result,
                    (target_start, 0),
                    (block_size, payload_size),
                )
                return jax.lax.dynamic_update_slice(
                    inner_result,
                    target_slice + plan.T @ source_values,
                    (target_start, 0),
                )
            target_values = jnp.take(values, target_indices, axis=0)
            return inner_result + plan @ target_values

        if direction == "source_to_target":
            return jax.lax.fori_loop(0, target_blocks, target_body, result)
        source_output = jax.lax.fori_loop(
            0,
            target_blocks,
            target_body,
            source_output,
        )
        return jax.lax.dynamic_update_slice(
            result,
            source_output,
            (source_start, 0),
        )

    output = jax.lax.fori_loop(0, source_blocks, source_body, output)
    return output[: target_count if direction == "source_to_target" else source_count]


def _cost_block(
    problem: DiscreteTransportProblem,
    source_indices: Array,
    target_indices: Array,
    /,
) -> Array:
    return _geometry_cost_block(
        problem.cost,
        problem.source.points,
        problem.target.points,
        source_indices,
        target_indices,
    )


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


__all__ = [
    "Direction",
    "apply_plan",
    "column_logsumexp",
    "coupling_statistics",
    "dense_plan",
    "row_logsumexp",
]
