#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Accepted, fixed-realization physical response pullbacks, without optimization."""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    PyTreeSpace,
    solve as solve_linear,
    transpose,
)
from ._iterative._types import _tree_allfinite, _validate_real_inexact_tree
from ._pde_constrained import (
    _default_adjoint_policy,
    AdjointAcceptanceEvidence,
    StateAcceptanceEvidence,
    StateDesignProblem,
    StateEquationResult,
)


class _StateAction(StrictModule):
    action: Any
    template: PyTree[Array]
    pullback: bool = eqx.field(static=True)

    def __call__(self, value):
        result = self.action(value)
        if self.pullback:
            result = result[0]
        return jax.tree.map(
            lambda leaf, reference: jnp.asarray(leaf, dtype=reference.dtype).reshape(
                reference.shape
            ),
            result,
            self.template,
        )


class StateDesignLinearization(StrictModule):
    """One immutable state/design point and its reusable residual linearization.

    Construct through ``prepare_state_design_linearization``. There is no numeric
    rebinding operation: a changed design, argument, or realization requires a
    newly accepted state. Residual pullbacks are JAX PyTrees, not dense Jacobians.
    """

    problem: StateDesignProblem
    state: PyTree[Array]
    design: PyTree[Array]
    args: Any
    residual: PyTree[Array]
    state_jacobian: FunctionLinearOperator
    design_pullback: Any
    state_acceptance: StateAcceptanceEvidence
    linear_policy: LinearSolvePolicy
    state_result: StateEquationResult | None

    @property
    def accepted(self) -> Array:
        return self.state_acceptance.accepted


class StateDesignResponseVJP(StrictModule):
    """Response and cotangent together with independent physical solve evidence."""

    values: PyTree[Array]
    design_cotangent: PyTree[Array]
    adjoint: PyTree[Array]
    linear_result: Any
    state_acceptance: StateAcceptanceEvidence
    adjoint_acceptance: AdjointAcceptanceEvidence | None
    accepted: Array


def _linearize_state_design(
    problem,
    state,
    design,
    args,
    linear_policy,
    state_acceptance,
    *,
    state_result=None,
    operator_id=None,
):
    def residual_function(current_state):
        return problem.residual(current_state, design, args)

    residual, state_action = jax.linearize(residual_function, state)
    _, state_pullback = jax.vjp(residual_function, state)
    state_jacobian = FunctionLinearOperator(
        _StateAction(state_action, residual, False),
        source=PyTreeSpace(state),
        target=PyTreeSpace(residual),
        transpose_action=_StateAction(state_pullback, state, True),
        operator_id=(
            f"{problem.problem_id}/state-jacobian" if operator_id is None else operator_id
        ),
        closure_convert=False,
    )
    _, design_pullback = jax.vjp(
        lambda current_design: problem.residual(state, current_design, args), design
    )
    return StateDesignLinearization(
        problem,
        state,
        design,
        args,
        residual,
        state_jacobian,
        design_pullback,
        state_acceptance,
        linear_policy,
        state_result,
    )


def prepare_state_design_linearization(
    problem: StateDesignProblem,
    design: PyTree[Any],
    initial_state: PyTree[Any],
    /,
    *,
    args: Any = None,
    linear_policy: LinearSolvePolicy | None = None,
) -> StateDesignLinearization:
    """Solve and independently accept physics, then prepare response pullbacks.

    A rejected state returns ``accepted=False``; consumers must not use response
    derivatives as accepted physics. This operation never runs a design optimizer.
    """
    if not isinstance(problem, StateDesignProblem):
        raise TypeError("problem must be a StateDesignProblem.")
    policy = _default_adjoint_policy() if linear_policy is None else linear_policy
    if not isinstance(policy, LinearSolvePolicy):
        raise TypeError("linear_policy must be LinearSolvePolicy or None.")
    design_ = _validate_real_inexact_tree(design, name="design")
    initial = _validate_real_inexact_tree(initial_state, name="initial_state")
    solved = problem.solve_state(design_, initial, args=args)
    return _linearize_state_design(
        problem,
        solved.state,
        design_,
        args,
        policy,
        solved.acceptance,
        state_result=solved,
    )


def _response_pullback(
    linearization,
    response,
    cotangent,
    depends_on_state,
    *,
    prepared_adjoint=None,
):
    point = linearization
    function = (
        (lambda state, design, args: point.problem.value(state, design, args)[0])
        if response is None
        else response
    )
    if not callable(function):
        raise TypeError("response must be callable or None.")
    if depends_on_state:
        values, pullback = jax.vjp(
            lambda state, design: function(state, design, point.args),
            point.state,
            point.design,
        )
    else:
        values, pullback = jax.vjp(
            lambda design: function(point.state, design, point.args), point.design
        )
    values = _validate_real_inexact_tree(values, name="response")
    if cotangent is None:
        if jax.tree.structure(values) != jax.tree.structure(jnp.asarray(0.0)):
            raise ValueError("A nonscalar response requires an explicit cotangent.")
        if values.shape != ():
            raise ValueError("A nonscalar response requires an explicit cotangent.")
        cotangent = jnp.ones_like(values)
    cotangent = _validate_real_inexact_tree(cotangent, name="response cotangent")
    if jax.tree.structure(cotangent) != jax.tree.structure(values):
        raise ValueError("Response and cotangent PyTree structures must match.")
    if any(
        left.shape != right.shape or left.dtype != right.dtype
        for left, right in zip(jax.tree.leaves(cotangent), jax.tree.leaves(values))
    ):
        raise ValueError("Response cotangent shapes and dtypes must match the response.")
    if depends_on_state:
        state_gradient, direct = pullback(cotangent)
        result = (
            solve_linear(
                LinearSystem(transpose(point.state_jacobian)),
                state_gradient,
                policy=point.linear_policy,
            )
            if prepared_adjoint is None
            else solve_linear(prepared_adjoint, state_gradient)
        )
        adjoint = result.value
        acceptance = point.problem.acceptance_policy.adjoint_evidence(
            adjoint,
            point.state_jacobian.transpose_mv(adjoint),
            state_gradient,
            result.status,
            admissible=point.state_acceptance.accepted,
            realization_matches=point.state_acceptance.realization_matches,
        )
        residual_part = point.design_pullback(adjoint)[0]
        gradient = jax.tree.map(lambda left, right: left - right, direct, residual_part)
        derivative_accepted = acceptance.accepted
    else:
        gradient = pullback(cotangent)[0]
        adjoint = jax.tree.map(jnp.zeros_like, point.residual)
        result = None
        acceptance = None
        derivative_accepted = jnp.asarray(True)
    current_realization = (
        jnp.asarray(True)
        if point.problem.state_realization is None
        else jnp.all(
            jnp.asarray(
                point.problem.state_realization(point.state, point.design, point.args),
                dtype=bool,
            )
        )
    )
    accepted = (
        point.accepted
        & derivative_accepted
        & current_realization
        & _tree_allfinite(values)
        & _tree_allfinite(cotangent)
        & _tree_allfinite(gradient)
    )
    return StateDesignResponseVJP(
        values, gradient, adjoint, result, point.state_acceptance, acceptance, accepted
    )


def state_design_response_vjp(
    linearization: StateDesignLinearization,
    /,
    response=None,
    cotangent=None,
    *,
    depends_on_state: bool = True,
) -> StateDesignResponseVJP:
    """Pull a response cotangent through accepted fixed-realization physics.

    ``response(state, design, args)`` defaults to the scalar objective. A PyTree
    response requires a matching cotangent. ``depends_on_state=False`` declares a
    design-only response and omits the transpose solve. The returned ``accepted``
    flag, not finite fallback values or backend success alone, permits derivative
    use. This is a first-order response derivative, not an optimizer derivative.
    """
    if not isinstance(linearization, StateDesignLinearization):
        raise TypeError("linearization must be StateDesignLinearization.")
    if not isinstance(depends_on_state, bool):
        raise TypeError("depends_on_state must be a static bool.")
    return _response_pullback(linearization, response, cotangent, depends_on_state)


__all__ = [
    "StateDesignLinearization",
    "StateDesignResponseVJP",
    "prepare_state_design_linearization",
    "state_design_response_vjp",
]
