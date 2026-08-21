#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from ..linalg._causal_linear import (
    associative_affine_solve,
    associative_transpose_solve,
)
from ._types import NonlinearStatus


def _exact_transition_matrices(problem, trajectory: Array, /) -> Array:
    predecessors = jnp.concatenate(
        (problem.flat_initial_state[None, :], trajectory[:-1]),
        axis=0,
    )
    matrices = jax.vmap(jax.jacfwd(problem.transition_flat))(
        predecessors,
        problem.drivers,
    )
    return matrices.at[0].set(jnp.zeros_like(matrices[0]))


def attach_causal_implicit_derivative(problem, result, /):
    """Attach the exact implicit solution derivative to a certified trajectory."""

    forward_states = jax.lax.stop_gradient(result.flat_states)

    def residual_function(trajectory):
        residual, _ = problem.evaluate_flat(trajectory)
        return residual

    def primal_solve(_, __):
        return forward_states

    def tangent_solve(linearized, right_hand_side):
        checked = eqx.error_if(
            right_hand_side,
            result.status != int(NonlinearStatus.SUCCESS),
            "Implicit causal derivative requires a successfully converged trajectory.",
        )
        matrices = _exact_transition_matrices(problem, forward_states)
        return jax.lax.custom_linear_solve(
            linearized,
            checked,
            solve=lambda _, rhs: associative_affine_solve(matrices, rhs),
            transpose_solve=lambda _, rhs: associative_transpose_solve(matrices, rhs),
        )

    implicit_states = jax.lax.custom_root(
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
