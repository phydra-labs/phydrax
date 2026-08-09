#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array

from ._geometry import block_count, cost_block, indices
from ._unbalanced_problem import UnbalancedTransportProblem


Direction = Literal["source_to_target", "target_to_source"]


def generalized_kl(values: Array, reference: Array, /) -> Array:
    """Generalized KL for nonnegative finite vectors, including total-mass terms."""
    values_ = jnp.asarray(values)
    reference_ = jnp.asarray(reference)
    positive = values_ > 0.0
    reference_positive = reference_ > 0.0
    log_ratio = jnp.log(jnp.where(positive, values_, 1.0)) - jnp.log(
        jnp.where(reference_positive, reference_, 1.0)
    )
    terms = jnp.where(
        positive & reference_positive,
        values_ * log_ratio,
        jnp.where(positive, jnp.inf, 0.0),
    )
    return jnp.sum(terms - values_ + reference_)


def coupling_statistics(
    problem: UnbalancedTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    /,
    *,
    block_size: int | None,
) -> tuple[Array, Array, Array, Array, Array, Array]:
    """Return physical marginals, cost, entropy KL, mass, and finite flag."""
    source_log_weights = _safe_log(problem.source_weights)
    target_log_weights = _safe_log(problem.target_weights)
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
        transported_mass = jnp.sum(plan)
        transport_cost = jnp.sum(plan * costs)
        safe_ratio = jnp.where(jnp.isfinite(log_ratio), log_ratio, 0.0)
        entropy_kl = (
            jnp.sum(plan * safe_ratio)
            - transported_mass
            + problem.source_mass * problem.target_mass
        )
        finite = (
            jnp.all(jnp.isfinite(source_marginal))
            & jnp.all(jnp.isfinite(target_marginal))
            & jnp.isfinite(transport_cost)
            & jnp.isfinite(entropy_kl)
            & jnp.isfinite(transported_mass)
        )
        return (
            source_marginal,
            target_marginal,
            transport_cost,
            entropy_kl,
            transported_mass,
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
    problem: UnbalancedTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    /,
) -> Array:
    """Materialize the physical unbalanced coupling."""
    log_plan = (
        _safe_log(problem.source_weights)[:, None]
        + _safe_log(problem.target_weights)[None, :]
        + (
            source_potential[:, None]
            + target_potential[None, :]
            - problem.cost_matrix()
        )
        / epsilon
    )
    return jnp.exp(log_plan)


def apply_plan(
    problem: UnbalancedTransportProblem,
    source_potential: Array,
    target_potential: Array,
    epsilon: Array,
    values: Array,
    /,
    *,
    direction: Direction,
    block_size: int | None,
) -> Array:
    """Apply the physical unbalanced coupling without forced normalization."""
    values_ = jnp.asarray(
        values,
        dtype=jnp.result_type(values, source_potential, target_potential),
    )
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
    else:
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
    problem: UnbalancedTransportProblem,
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
    source_blocks = block_count(source_count, block_size)
    target_blocks = block_count(target_count, block_size)
    source_padded = source_blocks * block_size
    target_padded = target_blocks * block_size
    initial = (
        jnp.zeros((source_padded,), dtype=source_potential.dtype),
        jnp.zeros((target_padded,), dtype=target_potential.dtype),
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(0.0, dtype=source_potential.dtype),
        jnp.asarray(True),
    )

    def source_body(source_block, state):
        source_result, target_result, cost_total, ratio_total, mass_total, finite = state
        source_start = source_block * block_size
        source_indices, source_valid = indices(source_start, block_size, source_count)
        source_accumulator = jnp.zeros(
            (block_size,), dtype=source_potential.dtype
        )

        def target_body(target_block, inner):
            (
                source_partial,
                target_partial,
                cost_partial,
                ratio_partial,
                mass_partial,
                finite_partial,
            ) = inner
            target_start = target_block * block_size
            target_indices, target_valid = indices(
                target_start, block_size, target_count
            )
            costs = cost_block(
                problem.cost,
                problem.source.points,
                problem.target.points,
                source_indices,
                target_indices,
            )
            f = jnp.take(source_potential, source_indices, axis=0)
            g = jnp.take(target_potential, target_indices, axis=0)
            log_a = jnp.take(source_log_weights, source_indices, axis=0)
            log_b = jnp.take(target_log_weights, target_indices, axis=0)
            valid = source_valid[:, None] & target_valid[None, :]
            log_ratio = (f[:, None] + g[None, :] - costs) / epsilon
            plan = jnp.where(
                valid,
                jnp.exp(log_a[:, None] + log_b[None, :] + log_ratio),
                0.0,
            )
            safe_ratio = jnp.where(valid & jnp.isfinite(log_ratio), log_ratio, 0.0)
            target_slice = jax.lax.dynamic_slice(
                target_partial,
                (target_start,),
                (block_size,),
            )
            target_partial = jax.lax.dynamic_update_slice(
                target_partial,
                target_slice + jnp.sum(plan, axis=0),
                (target_start,),
            )
            block_finite = (
                jnp.all(jnp.isfinite(plan))
                & jnp.all(jnp.isfinite(jnp.where(valid, costs, 0.0)))
            )
            return (
                source_partial + jnp.sum(plan, axis=1),
                target_partial,
                cost_partial + jnp.sum(plan * costs),
                ratio_partial + jnp.sum(plan * safe_ratio),
                mass_partial + jnp.sum(plan),
                finite_partial & block_finite,
            )

        (
            source_accumulator,
            target_result,
            cost_total,
            ratio_total,
            mass_total,
            finite,
        ) = jax.lax.fori_loop(
            0,
            target_blocks,
            target_body,
            (
                source_accumulator,
                target_result,
                cost_total,
                ratio_total,
                mass_total,
                finite,
            ),
        )
        source_result = jax.lax.dynamic_update_slice(
            source_result,
            source_accumulator,
            (source_start,),
        )
        return source_result, target_result, cost_total, ratio_total, mass_total, finite

    (
        source_marginal,
        target_marginal,
        transport_cost,
        ratio_term,
        transported_mass,
        finite,
    ) = jax.lax.fori_loop(0, source_blocks, source_body, initial)
    entropy_kl = (
        ratio_term
        - transported_mass
        + problem.source_mass * problem.target_mass
    )
    finite = finite & jnp.isfinite(transport_cost) & jnp.isfinite(entropy_kl)
    return (
        source_marginal[:source_count],
        target_marginal[:target_count],
        transport_cost,
        entropy_kl,
        transported_mass,
        finite,
    )


def _blockwise_apply(
    problem: UnbalancedTransportProblem,
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
    source_blocks = block_count(source_count, block_size)
    target_blocks = block_count(target_count, block_size)
    payload_size = values.shape[1]
    source_log_weights = _safe_log(problem.source_weights)
    target_log_weights = _safe_log(problem.target_weights)
    output_count = target_blocks if direction == "source_to_target" else source_blocks
    output = jnp.zeros(
        (output_count * block_size, payload_size), dtype=values.dtype
    )

    def source_body(source_block, result):
        source_start = source_block * block_size
        source_indices, source_valid = indices(source_start, block_size, source_count)
        f = jnp.take(source_potential, source_indices, axis=0)
        log_a = jnp.take(source_log_weights, source_indices, axis=0)
        if direction == "source_to_target":
            source_values = jnp.take(values, source_indices, axis=0)
        else:
            source_output = jnp.zeros((block_size, payload_size), dtype=values.dtype)

        def target_body(target_block, inner_result):
            target_start = target_block * block_size
            target_indices, target_valid = indices(
                target_start, block_size, target_count
            )
            g = jnp.take(target_potential, target_indices, axis=0)
            log_b = jnp.take(target_log_weights, target_indices, axis=0)
            costs = cost_block(
                problem.cost,
                problem.source.points,
                problem.target.points,
                source_indices,
                target_indices,
            )
            valid = source_valid[:, None] & target_valid[None, :]
            plan = jnp.where(
                valid,
                jnp.exp(
                    log_a[:, None]
                    + log_b[None, :]
                    + (f[:, None] + g[None, :] - costs) / epsilon
                ),
                0.0,
            )
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
            0, target_blocks, target_body, source_output
        )
        return jax.lax.dynamic_update_slice(
            result,
            source_output,
            (source_start, 0),
        )

    output = jax.lax.fori_loop(0, source_blocks, source_body, output)
    final_count = target_count if direction == "source_to_target" else source_count
    return output[:final_count]


def _safe_log(values: Array, /) -> Array:
    return jnp.where(values > 0.0, jnp.log(values), -jnp.inf)


__all__ = [
    "Direction",
    "apply_plan",
    "coupling_statistics",
    "dense_plan",
    "generalized_kl",
]
