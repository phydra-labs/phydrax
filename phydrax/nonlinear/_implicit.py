#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._tree_math import validate_inexact_tree
from ..linalg import (
    FGMRES,
    FunctionLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    PyTreeSpace,
    solve as solve_linear,
)
from ..linalg._runtime import _callable_gmres_for_policy
from ._newton import NewtonKrylov, NewtonTrustRegion
from ._types import (
    AbstractNonlinearMethod,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


def _checked_root_coordinates(
    result,
    space: PyTreeSpace,
    /,
) -> Array:
    coordinates = space.flatten(result.state)
    return eqx.error_if(
        coordinates,
        result.status != int(NonlinearStatus.SUCCESS),
        "Implicit nonlinear root solve failed; inspect an explicit root result first.",
    )


def _checked_tangent_solve(
    action,
    right_hand_side: Array,
    policy: LinearSolvePolicy,
    /,
) -> Array:
    if (
        isinstance(policy.method, (GMRES, FGMRES))
        and policy.preconditioning is None
        and policy.recycling is None
        and policy.precision is None
    ):
        value = _callable_gmres_for_policy(action, right_hand_side, policy)
        residual_norm = jnp.linalg.norm(right_hand_side - action(value))
        threshold = policy.tolerance.absolute + (
            policy.tolerance.relative * jnp.linalg.norm(right_hand_side)
        )
        return eqx.error_if(
            value,
            ~jnp.isfinite(residual_norm) | (residual_norm > threshold),
            "Implicit root derivative solve failed; the root Jacobian is unresolved.",
        )

    space = PyTreeSpace(right_hand_side)
    operator = FunctionLinearOperator(
        action,
        source=space,
        target=space,
        closure_convert=False,
    )
    result = solve_linear(LinearSystem(operator), right_hand_side, policy=policy)
    value = space.flatten(result.value)
    return eqx.error_if(
        value,
        result.status != int(LinearSolveStatus.SUCCESS),
        "Implicit root derivative solve failed; the root Jacobian is unresolved.",
    )


def implicit_root(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    args: Any = None,
) -> PyTree[Array]:
    """Return a root with forward- and reverse-mode implicit derivatives.

    Differentiation follows the residual arguments captured by ``args``; the initial
    guess is treated only as a globalization input. The derivative solve must converge
    successfully, so singular or unresolved Jacobians fail rather than returning a
    misleading tangent.
    """
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, AbstractNonlinearMethod):
        raise TypeError("method must be AbstractNonlinearMethod or None.")
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be NonlinearTermination or None.")
    if linear_policy is None:
        if not isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
            raise ValueError(
                "linear_policy is required when the nonlinear method has no linear policy."
            )
        derivative_policy = method_.linear_policy
    else:
        derivative_policy = linear_policy
    if not isinstance(derivative_policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")

    initial = validate_inexact_tree(initial_state, name="initial implicit root state")
    source = PyTreeSpace(initial)
    initial_residual = problem.residual(initial, args)
    target = PyTreeSpace(initial_residual)
    if source.size != target.size:
        raise ValueError("Implicit root differentiation requires a square Jacobian.")
    initial_coordinates = source.flatten(initial)

    def coordinate_residual(coordinates):
        state = source.unflatten(coordinates)
        return target.flatten(problem.residual(state, args))

    def primal_solve(_, coordinates):
        result = method_.solve(
            problem,
            source.unflatten(coordinates),
            termination=termination_,
            args=args,
        )
        return _checked_root_coordinates(result, source)

    def tangent_solve(linearized, right_hand_side):
        return jax.lax.custom_linear_solve(
            linearized,
            right_hand_side,
            solve=lambda action, rhs: _checked_tangent_solve(
                action, rhs, derivative_policy
            ),
            transpose_solve=lambda action, rhs: _checked_tangent_solve(
                action, rhs, derivative_policy
            ),
        )

    coordinates = jax.lax.custom_root(
        coordinate_residual,
        initial_coordinates,
        solve=primal_solve,
        tangent_solve=tangent_solve,
    )
    return source.unflatten(coordinates)


__all__ = ["implicit_root"]
