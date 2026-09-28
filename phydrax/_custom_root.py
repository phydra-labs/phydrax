#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Implicit-function root differentiation that stays linearizable under vmap."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree


def _zero_tangent(leaf: Any) -> Any:
    value = jnp.asarray(leaf)
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.zeros_like(value)
    return np.zeros(value.shape, dtype=jax.dtypes.float0)


def custom_root(
    f: Callable[[PyTree[Any]], PyTree[Any]],
    initial_guess: PyTree[Any],
    solve: Callable[[Callable[[PyTree[Any]], PyTree[Any]], PyTree[Any]], Any],
    tangent_solve: Callable[
        [Callable[[PyTree[Any]], PyTree[Any]], PyTree[Any]], PyTree[Any]
    ],
    has_aux: bool = False,
) -> Any:
    """Return ``solve(f, initial_guess)`` differentiated through ``f(root) = 0``.

    The contract matches ``jax.lax.custom_root``: only values closed over by
    ``f`` carry derivatives, ``dx = -(D_x f)^{-1} D_p f dp`` with the inverse
    applied by ``tangent_solve``, and derivatives of ``initial_guess``, of values
    closed over by ``solve`` or ``tangent_solve``, and of auxiliary outputs are
    zero. The rule is a custom JVP, so ``jax.linearize`` of a vmapped root works
    (JAX's hijax ``custom_root`` batches into a ``VmapOf`` without a
    linearization rule).
    """
    # Hoist every closed-over array, including undifferentiated batch or staged
    # tracers, so neither the primal nor the rule captures an outer tracer.
    residual = eqx.filter_closure_convert(f, initial_guess)
    residual_dynamic, residual_static = eqx.partition(residual, eqx.is_array)
    residual_shape = jax.eval_shape(residual, initial_guess)
    if jax.tree.structure(residual_shape) != jax.tree.structure(initial_guess):
        raise TypeError("f must return the same pytree structure as initial_guess.")
    primal_solve = eqx.filter_closure_convert(
        lambda guess: solve(residual, guess), initial_guess
    )
    solve_dynamic, solve_static = eqx.partition(primal_solve, eqx.is_array)

    def linearize_and_solve(
        dynamic: Any, root: PyTree[Any], right_hand_side: PyTree[Any]
    ) -> PyTree[Any]:
        function = eqx.combine(dynamic, residual_static)
        return tangent_solve(
            lambda tangent: jax.jvp(function, (root,), (tangent,))[1],
            right_hand_side,
        )

    implicit_solve = eqx.filter_closure_convert(
        linearize_and_solve, residual_dynamic, initial_guess, residual_shape
    )
    tangent_dynamic, tangent_static = eqx.partition(implicit_solve, eqx.is_array)

    @jax.custom_jvp
    def root(
        residual_dynamic: Any,
        solve_dynamic: Any,
        tangent_dynamic: Any,
        guess: PyTree[Any],
    ) -> Any:
        # Differentiation may pass constant primals as typed scalar literals.
        return eqx.combine(solve_dynamic, solve_static)(jax.tree.map(jnp.asarray, guess))

    @root.defjvp
    def root_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        residual_dynamic = jax.tree.map(jnp.asarray, primals[0])
        tangent_dynamic = primals[2]
        solution = root(*primals)
        value, auxiliary = solution if has_aux else (solution, None)
        _, parameter_action = jax.jvp(
            lambda dynamic: eqx.combine(dynamic, residual_static)(value),
            (residual_dynamic,),
            (tangents[0],),
        )
        value_tangent = jax.tree.map(
            jnp.negative,
            eqx.combine(tangent_dynamic, tangent_static)(
                residual_dynamic, value, parameter_action
            ),
        )
        if has_aux:
            return solution, (value_tangent, jax.tree.map(_zero_tangent, auxiliary))
        return solution, value_tangent

    return root(residual_dynamic, solve_dynamic, tangent_dynamic, initial_guess)


__all__ = ["custom_root"]
