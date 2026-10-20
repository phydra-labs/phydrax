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
    zero. The custom JVP keeps residual, solve and tangent-solve captures
    explicit for Phydrax's compiler identity and structural analysis. Upstream
    HiJax roots require separate admission of those hidden program contracts;
    retaining this implementation avoids an opaque root boundary.

    Symbolically inactive residual parameters are not differentiated. Undefined
    active parameter actions remain undefined; no tangent magnitude is clipped.
    """
    # Hoist every closed-over array, including undifferentiated batch or staged
    # tracers, so neither the primal nor the rule captures an outer tracer.
    residual = eqx.filter_closure_convert(f, initial_guess)
    residual_dynamic, residual_static = eqx.partition(residual, eqx.is_array)

    # Name the shape-only boundary instead of formatting the residual's Jaxpr.
    def evaluate_residual_shape(point: PyTree[Any]) -> PyTree[Any]:
        return residual(point)

    residual_shape = jax.eval_shape(evaluate_residual_shape, initial_guess)
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

    @eqx.filter_custom_jvp
    def root(
        residual_dynamic: Any,
        solve_dynamic: Any,
        tangent_dynamic: Any,
        guess: PyTree[Any],
    ) -> Any:
        # Differentiation may pass constant primals as typed scalar literals.
        return eqx.combine(solve_dynamic, solve_static)(jax.tree.map(jnp.asarray, guess))

    @root.def_jvp
    def root_jvp(primals: Any, tangents: Any) -> tuple[Any, Any]:
        residual_dynamic = jax.tree.map(jnp.asarray, primals[0])
        tangent_dynamic = primals[2]
        solution = root(*primals)
        value, auxiliary = solution if has_aux else (solution, None)
        if jax.tree.leaves(tangents[0]):
            # Preserve symbolic zero parameter tangents: differentiating an
            # inactive fractional power at zero can otherwise inject 0 * inf.
            _, parameter_action = eqx.filter_jvp(
                lambda dynamic: eqx.combine(dynamic, residual_static)(value),
                (residual_dynamic,),
                (tangents[0],),
            )
            parameter_action = jax.tree.map(
                lambda primal, tangent: (
                    _zero_tangent(primal) if tangent is None else tangent
                ),
                value,
                parameter_action,
                is_leaf=lambda leaf: leaf is None,
            )
            value_tangent = jax.tree.map(
                jnp.negative,
                eqx.combine(tangent_dynamic, tangent_static)(
                    residual_dynamic, value, parameter_action
                ),
            )
        else:
            # Guess/solver/auxiliary dependence is explicitly outside the IFT.
            value_tangent = jax.tree.map(_zero_tangent, value)
        if has_aux:
            return solution, (value_tangent, jax.tree.map(_zero_tangent, auxiliary))
        return solution, value_tangent

    return root(residual_dynamic, solve_dynamic, tangent_dynamic, initial_guess)


__all__ = ["custom_root"]
