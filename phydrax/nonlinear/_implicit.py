#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from ..linalg import (
    AbstractVectorSpace,
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
from ._prepared import PreparedNonlinearSolve, solve_prepared_nonlinear
from ._types import (
    AbstractNonlinearMethod,
    NonlinearDiagnostics,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


_DEFAULT_ARGS = object()


def _diagnostic_counts(
    diagnostics: NonlinearDiagnostics,
    /,
) -> tuple[Array, ...]:
    return (
        diagnostics.iterations,
        diagnostics.residual_evaluations,
        diagnostics.jvp_evaluations,
        diagnostics.vjp_evaluations,
        diagnostics.jacobian_preparations,
        diagnostics.linear_solves,
        diagnostics.linear_iterations,
        diagnostics.accepted_steps,
        diagnostics.rejected_steps,
        diagnostics.domain_failures,
        diagnostics.nonfinite_trials,
        diagnostics.acceleration_restarts,
        diagnostics.setup_refreshes,
        diagnostics.numeric_refreshes,
        diagnostics.final_linear_status,
        diagnostics.final_linear_rank,
        diagnostics.final_linear_converged,
    )


def _cast_diagnostic_counts(
    diagnostics: NonlinearDiagnostics,
    dtype,
    /,
) -> NonlinearDiagnostics:
    return eqx.tree_at(
        _diagnostic_counts,
        diagnostics,
        tuple(value.astype(dtype) for value in _diagnostic_counts(diagnostics)),
    )


def _restore_diagnostic_types(
    diagnostics: NonlinearDiagnostics,
    /,
) -> NonlinearDiagnostics:
    restored = _cast_diagnostic_counts(diagnostics, jnp.int32)
    return eqx.tree_at(
        lambda value: value.final_linear_converged,
        restored,
        diagnostics.final_linear_converged.astype(bool),
    )


def _checked_root_state(
    result: NonlinearResult,
    space: AbstractVectorSpace,
    /,
) -> PyTree[Array]:
    coordinates = space.flatten(result.state)
    coordinates = eqx.error_if(
        coordinates,
        result.status != int(NonlinearStatus.SUCCESS),
        "Implicit nonlinear root solve failed; inspect an explicit root result first.",
    )
    return space.unflatten(coordinates)


def _checked_tangent_solve(
    action,
    right_hand_side: Array,
    policy: LinearSolvePolicy,
    /,
) -> Array:
    zero_right_hand_side = jnp.all(right_hand_side == 0)
    safe_right_hand_side = jnp.where(
        zero_right_hand_side,
        jnp.ones_like(right_hand_side),
        right_hand_side,
    )
    if (
        isinstance(policy.method, (GMRES, FGMRES))
        and policy.preconditioning is None
        and policy.recycling is None
        and policy.precision is None
    ):
        value = _callable_gmres_for_policy(action, safe_right_hand_side, policy)
        residual_norm = jnp.linalg.norm(safe_right_hand_side - action(value))
        threshold = policy.tolerance.absolute + (
            policy.tolerance.relative * jnp.linalg.norm(safe_right_hand_side)
        )
        checked = eqx.error_if(
            value,
            ~zero_right_hand_side
            & (~jnp.isfinite(residual_norm) | (residual_norm > threshold)),
            "Implicit root derivative solve failed; the root Jacobian is unresolved.",
        )
    else:
        space = PyTreeSpace(safe_right_hand_side)
        operator = FunctionLinearOperator(
            action,
            source=space,
            target=space,
            closure_convert=False,
        )
        result = solve_linear(
            LinearSystem(operator),
            safe_right_hand_side,
            policy=policy,
        )
        checked = eqx.error_if(
            space.flatten(result.value),
            ~zero_right_hand_side & (result.status != int(LinearSolveStatus.SUCCESS)),
            "Implicit root derivative solve failed; the root Jacobian is unresolved.",
        )
    return jnp.where(
        zero_right_hand_side,
        jnp.zeros_like(checked),
        checked,
    )


def implicit_root_result(
    problem_or_prepared: NonlinearSystemProblem | PreparedNonlinearSolve,
    initial_state: PyTree[Any] | None = None,
    /,
    *,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    linear_policy: LinearSolvePolicy | None = None,
    args: Any = _DEFAULT_ARGS,
) -> NonlinearResult:
    """Return one nonlinear result whose root has implicit derivatives.

    The primal nonlinear solve runs once. Its status and diagnostics are retained as
    nondifferentiable evidence, while state, residual, and auxiliary values are
    evaluated at the implicitly differentiated root. A failed solve remains a failed
    result; implicit derivatives are meaningful only when ``successful`` is true.
    """
    if isinstance(problem_or_prepared, PreparedNonlinearSolve):
        if initial_state is not None or method is not None or termination is not None:
            raise ValueError(
                "initial_state, method, and termination must be omitted for a "
                "prepared implicit root."
            )
        if args is not _DEFAULT_ARGS:
            raise ValueError("args must be omitted for a prepared implicit root.")
        prepared = problem_or_prepared
        problem = prepared.problem
        initial = prepared.state
        method_ = prepared.method
        termination_ = prepared.termination
        runtime_args = prepared.args
    elif isinstance(problem_or_prepared, NonlinearSystemProblem):
        if initial_state is None:
            raise ValueError("initial_state is required for an unprepared implicit root.")
        prepared = None
        problem = problem_or_prepared
        initial = problem.validate_state(initial_state)
        method_ = NewtonKrylov() if method is None else method
        termination_ = NonlinearTermination() if termination is None else termination
        runtime_args = None if args is _DEFAULT_ARGS else args
    else:
        raise TypeError(
            "problem_or_prepared must be a NonlinearSystemProblem or "
            "PreparedNonlinearSolve."
        )

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

    initial_residual = problem.residual(initial, runtime_args)
    source = PyTreeSpace(initial) if problem.state_space is None else problem.state_space
    target = (
        PyTreeSpace(initial_residual)
        if problem.residual_space is None
        else problem.residual_space
    )
    if source.size != target.size:
        raise ValueError("Implicit root differentiation requires a square Jacobian.")
    initial_coordinates = source.flatten(initial)

    def coordinate_residual(coordinates):
        state = source.unflatten(coordinates)
        return target.flatten(problem.residual(state, runtime_args))

    def primal_solve(_, coordinates):
        if prepared is None:
            result = method_.solve(
                problem,
                source.unflatten(coordinates),
                termination=termination_,
                args=runtime_args,
            )
        else:
            result = solve_prepared_nonlinear(prepared)
        if result.transformation_evidence is not None:
            raise ValueError(
                "Implicit root results do not support transformed nonlinear evidence."
            )
        evidence = (
            result.status.astype(coordinates.dtype),
            _cast_diagnostic_counts(result.diagnostics, coordinates.dtype),
            result.provenance,
        )
        return source.flatten(result.state), evidence

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

    coordinates, evidence = jax.lax.custom_root(
        coordinate_residual,
        initial_coordinates,
        solve=primal_solve,
        tangent_solve=tangent_solve,
        has_aux=True,
    )
    status, diagnostics, provenance = evidence
    state = source.unflatten(coordinates)
    residual, auxiliary = problem.evaluate(state, runtime_args)
    return NonlinearResult(
        state=state,
        residual=residual,
        auxiliary=auxiliary,
        status=status,
        diagnostics=_restore_diagnostic_types(diagnostics),
        provenance=provenance,
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
    """Return a successful root with forward- and reverse-mode implicit derivatives."""
    result = implicit_root_result(
        problem,
        initial_state,
        method=method,
        termination=termination,
        linear_policy=linear_policy,
        args=args,
    )
    state = problem.validate_state(result.state)
    source = PyTreeSpace(state) if problem.state_space is None else problem.state_space
    return _checked_root_state(result, source)


__all__ = ["implicit_root", "implicit_root_result"]
