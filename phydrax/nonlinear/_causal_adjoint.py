#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_never, TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .._custom_root import custom_root
from .._numerics._checkpointed_scan import checkpointed_scan
from ..linalg._causal_linear import (
    associative_affine_solve,
    associative_transpose_solve,
)
from ._types import NonlinearStatus


if TYPE_CHECKING:
    from ._causal import (
        CausalAdjointPolicy,
        CausalRecurrenceProblem,
        CausalRecurrenceResult,
    )


def _exact_transition_matrices(
    problem: CausalRecurrenceProblem, trajectory: Array, /
) -> Array:
    predecessors = jnp.concatenate(
        (problem.flat_initial_state[None, :], trajectory[:-1]),
        axis=0,
    )
    matrices = jax.vmap(jax.jacfwd(problem.transition_flat))(
        predecessors,
        problem.drivers,
    )
    return matrices.at[0].set(jnp.zeros_like(matrices[0]))


def _action_solve(
    problem: CausalRecurrenceProblem,
    trajectory: Array,
    right: Array,
    policy: CausalAdjointPolicy,
    *,
    transpose: bool,
) -> Array:
    if problem.num_steps == 1:
        return right
    predecessors = trajectory[:-1]
    drivers = jax.tree.map(lambda value: value[1:], problem.drivers)
    if transpose:
        inputs = (
            predecessors[::-1],
            jax.tree.map(lambda value: value[::-1], drivers),
            right[:-1][::-1],
        )
        initial = right[-1]
    else:
        inputs = (predecessors, drivers, right[1:])
        initial = right[0]

    def step(carry: Array, data: tuple[Array, Any, Array]) -> tuple[Array, Array]:
        predecessor, driver, offset = data
        if transpose:
            _, pullback = jax.vjp(
                lambda state: problem.transition_flat(state, driver), predecessor
            )
            action = pullback(carry)[0]
        else:
            _, action = jax.jvp(
                lambda state: problem.transition_flat(state, driver),
                (predecessor,),
                (carry,),
            )
        value = offset + action
        return value, value

    match policy.route:
        case "action-scan":
            _, values = jax.lax.scan(step, initial, inputs)
        case "checkpointed-action-scan":
            _, values = checkpointed_scan(
                step,
                initial,
                inputs,
                length=problem.num_steps - 1,
                mode="block",
                block_size=policy.checkpoint_block_size,
            )
        case "dense-associative":
            raise ValueError("Dense causal adjoints do not use action scans.")
        case _:
            assert_never(policy.route)
    return (
        jnp.concatenate((values[::-1], initial[None]), axis=0)
        if transpose
        else jnp.concatenate((initial[None], values), axis=0)
    )


def attach_causal_implicit_derivative(
    problem: CausalRecurrenceProblem,
    result: CausalRecurrenceResult,
    policy: CausalAdjointPolicy,
    /,
) -> CausalRecurrenceResult:
    """Attach the exact implicit solution derivative to a certified trajectory."""

    forward_states = jax.lax.stop_gradient(result.flat_states)

    def residual_function(trajectory: Array) -> Array:
        residual, _ = problem.evaluate_flat(trajectory)
        return residual

    def primal_solve(_: Callable[[Array], Array], __: Array) -> Array:
        return forward_states

    def tangent_solve(
        linearized: Callable[[Array], Array], right_hand_side: Array
    ) -> Array:
        checked = eqx.error_if(
            right_hand_side,
            result.status != int(NonlinearStatus.SUCCESS),
            "Implicit causal derivative requires a successfully converged trajectory.",
        )
        match policy.route:
            case "dense-associative":
                policy.admit(
                    problem.num_steps, problem.state_size, forward_states.dtype.itemsize
                )
                matrices = _exact_transition_matrices(problem, forward_states)
                return jax.lax.custom_linear_solve(
                    linearized,
                    checked,
                    solve=lambda _, rhs: associative_affine_solve(matrices, rhs),
                    transpose_solve=lambda _, rhs: associative_transpose_solve(
                        matrices, rhs
                    ),
                )
            case "action-scan" | "checkpointed-action-scan":
                return jax.lax.custom_linear_solve(
                    linearized,
                    checked,
                    solve=lambda _, rhs: _action_solve(
                        problem, forward_states, rhs, policy, transpose=False
                    ),
                    transpose_solve=lambda _, rhs: _action_solve(
                        problem, forward_states, rhs, policy, transpose=True
                    ),
                )
            case _:
                assert_never(policy.route)

    implicit_states = custom_root(
        residual_function,
        forward_states,
        solve=primal_solve,
        tangent_solve=tangent_solve,
    )
    implicit_residuals, _ = problem.evaluate_flat(implicit_states)
    states = problem.unravel_trajectory(implicit_states)
    residuals = problem.unravel_trajectory(implicit_residuals)
    final_state = jax.tree.map(lambda leaf: leaf[-1], states)
    return eqx.tree_at(
        lambda value: (
            value.states,
            value.residuals,
            value.flat_states,
            value.flat_residuals,
            value.final_state,
        ),
        result,
        (states, residuals, implicit_states, implicit_residuals, final_state),
    )


__all__ = ["attach_causal_implicit_derivative"]
