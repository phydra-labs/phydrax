#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from ..linalg._causal_linear import (
    associative_affine_solve,
    associative_transpose_solve,
)
from ._types import NonlinearStatus


if TYPE_CHECKING:
    from ._causal import CausalRecurrenceProblem, CausalRecurrenceResult


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


def attach_causal_implicit_derivative(
    problem: CausalRecurrenceProblem, result: CausalRecurrenceResult, /
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
