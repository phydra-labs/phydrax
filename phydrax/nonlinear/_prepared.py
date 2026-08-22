#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._linear_refresh import LinearRefreshState
from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import (
    LinearSolvePlan,
    LinearSolveTemplate,
    LinearSystem,
    refresh_recycling,
)
from ._linearization import (
    _jacobian_solve_operator,
    prepare_jacobian,
    PreparedJacobian,
)
from ._newton import (
    _initial_root_state,
    _RootState,
    NewtonKrylov,
    NewtonTrustRegion,
)
from ._types import (
    AbstractNonlinearMethod,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


class PreparedNonlinearSolve(StrictModule):
    """Numeric Newton state bound to one reusable symbolic linear template."""

    problem: NonlinearSystemProblem
    state: PyTree[Array]
    method: NewtonKrylov | NewtonTrustRegion
    termination: NonlinearTermination
    args: Any
    jacobian: PreparedJacobian
    run: _RootState
    provenance: NonlinearProvenance
    numeric_version: Array

    def __init__(
        self,
        problem: NonlinearSystemProblem,
        state: PyTree[Any],
        method: NewtonKrylov | NewtonTrustRegion,
        termination: NonlinearTermination,
        args: Any,
        jacobian: PreparedJacobian,
        run: _RootState,
        provenance: NonlinearProvenance,
        /,
        *,
        numeric_version: Any,
    ):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if problem.state_space is None or problem.residual_space is None:
            raise ValueError(
                "A prepared nonlinear problem must have bound vector spaces."
            )
        if not isinstance(method, (NewtonKrylov, NewtonTrustRegion)):
            raise TypeError("method must be NewtonKrylov or NewtonTrustRegion.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        if not isinstance(jacobian, PreparedJacobian):
            raise TypeError("jacobian must be a PreparedJacobian.")
        if not isinstance(run, _RootState):
            raise TypeError("run must be prepared Newton root state.")
        if not isinstance(provenance, NonlinearProvenance):
            raise TypeError("provenance must be a NonlinearProvenance.")
        if provenance.problem_id != problem.problem_id:
            raise ValueError("Prepared nonlinear provenance must match the problem.")
        if provenance.method_id != method.method_id:
            raise ValueError("Prepared nonlinear provenance must match the method.")
        if provenance.linear_plan_id != run.refresh_state.template.plan.plan_id:
            raise ValueError("Prepared nonlinear provenance must match the linear plan.")
        self.problem = problem
        self.state = problem.state_space.validate(state)
        self.method = method
        self.termination = termination
        self.args = args
        self.jacobian = jacobian
        self.run = run
        self.provenance = provenance
        self.numeric_version = jnp.asarray(numeric_version, dtype=jnp.int32)

    @property
    def linear_refresh_state(self) -> LinearRefreshState:
        return self.run.refresh_state

    @property
    def linear_template(self) -> LinearSolveTemplate:
        return self.run.refresh_state.template

    @property
    def linear_plan(self) -> LinearSolvePlan:
        return self.run.refresh_state.template.plan

    @property
    def linear_plan_id(self) -> str:
        return self.provenance.linear_plan_id

    @property
    def linear_template_id(self) -> str:
        return self.run.refresh_state.template.template_id


def _globalization_id(method: NewtonKrylov | NewtonTrustRegion, /) -> str:
    if isinstance(method, NewtonKrylov):
        return "residual-armijo"
    return "dogleg-residual-trust-region"


def _prepared_provenance(
    problem: NonlinearSystemProblem,
    method: NewtonKrylov | NewtonTrustRegion,
    refresh_state: LinearRefreshState,
    /,
) -> NonlinearProvenance:
    plan = refresh_state.template.plan
    return NonlinearProvenance(
        problem_id=problem.problem_id,
        method_id=method.method_id,
        derivative_id=method.jacobian_policy.policy_id,
        globalization_id=_globalization_id(method),
        linear_plan_id=plan.plan_id,
        notes=f"linear-method={plan.method};linear-backend={plan.backend}",
    )


def _initial_trust_radius(method: NewtonKrylov | NewtonTrustRegion, /) -> float:
    if isinstance(method, NewtonTrustRegion):
        return method.trust_region.initial_radius
    return jnp.nan


def _refreshed_run(
    prepared: PreparedNonlinearSolve,
    problem: NonlinearSystemProblem,
    state: PyTree[Array],
    jacobian: PreparedJacobian,
    refresh_state: LinearRefreshState,
    args: Any,
    recycling,
    /,
) -> _RootState:
    residual = jacobian.residual
    residual_space = problem.residual_space
    if residual_space is None:
        raise ValueError("A refreshed nonlinear problem must have a residual space.")
    residual_norm = jnp.sqrt(
        jnp.maximum(jnp.real(residual_space.inner(residual, residual)), 0.0)
    )
    finite = tree_allfinite(state) & tree_allfinite(residual)
    valid = problem.valid(state, residual, jacobian.auxiliary, args)
    status = jnp.where(
        finite & valid,
        int(NonlinearStatus.ITERATING),
        jnp.where(
            finite,
            int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
            jnp.where(
                tree_allfinite(state),
                int(NonlinearStatus.NONFINITE_EVALUATION),
                int(NonlinearStatus.NONFINITE_INPUT),
            ),
        ),
    ).astype(jnp.int32)
    method = prepared.method
    forcing = jnp.asarray(
        (
            method.linear_policy.tolerance.relative
            if method.forcing_policy.strategy == "constant"
            else method.forcing_policy.initial
        ),
        dtype=residual_norm.dtype,
    )
    zero_count = jnp.asarray(0, dtype=jnp.int32)
    return _RootState(
        residual=residual,
        auxiliary=jacobian.auxiliary,
        initial_residual_norm=residual_norm,
        residual_norm=residual_norm,
        step_norm=jnp.asarray(0.0, dtype=residual_norm.dtype),
        iteration=zero_count,
        residual_evaluations=jnp.asarray(jacobian.residual_evaluations, dtype=jnp.int32),
        jvp_evaluations=zero_count,
        vjp_evaluations=zero_count,
        jacobian_preparations=jnp.asarray(1, dtype=jnp.int32),
        linear_solves=zero_count,
        linear_iterations=zero_count,
        accepted_steps=zero_count,
        rejected_steps=zero_count,
        globalization_rejections=zero_count,
        domain_failures=(finite & ~valid).astype(jnp.int32),
        nonfinite_trials=zero_count,
        setup_refreshes=zero_count,
        numeric_refreshes=jnp.asarray(1, dtype=jnp.int32),
        forcing=forcing,
        last_forcing=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        jacobian_age=zero_count,
        jacobian_reference_residual_norm=residual_norm,
        jacobian_reference_rejected_steps=zero_count,
        trust_radius=jnp.asarray(
            _initial_trust_radius(method), dtype=residual_norm.dtype
        ),
        status=status,
        refresh_state=refresh_state,
        recycling=recycling,
        final_linear_status=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_rank=jnp.asarray(-1, dtype=jnp.int32),
        final_linear_condition_estimate=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        final_linear_residual_norm=jnp.asarray(jnp.nan, dtype=residual_norm.dtype),
        final_linear_converged=jnp.asarray(False),
    )


def prepare_nonlinear(
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    method: AbstractNonlinearMethod | None = None,
    termination: NonlinearTermination | None = None,
    args: Any = None,
) -> PreparedNonlinearSolve:
    """Bind one Newton solve and retain its reusable symbolic linear template."""
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    method_ = NewtonKrylov() if method is None else method
    termination_ = NonlinearTermination() if termination is None else termination
    if not isinstance(method_, AbstractNonlinearMethod):
        raise TypeError("method must be an AbstractNonlinearMethod or None.")
    if not isinstance(method_, (NewtonKrylov, NewtonTrustRegion)):
        raise ValueError(
            "Prepared nonlinear solves support only NewtonKrylov and NewtonTrustRegion."
        )
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")
    problem_, state, run, jacobian = _initial_root_state(
        problem,
        initial_state,
        method_.jacobian_policy,
        method_.linear_policy,
        method_.forcing_policy,
        _initial_trust_radius(method_),
        args,
    )
    provenance = _prepared_provenance(problem_, method_, run.refresh_state)
    return PreparedNonlinearSolve(
        problem_,
        state,
        method_,
        termination_,
        args,
        jacobian,
        run,
        provenance,
        numeric_version=0,
    )


def refresh_nonlinear(
    prepared: PreparedNonlinearSolve,
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    args: Any = None,
) -> PreparedNonlinearSolve:
    """Refresh Newton numerics while preserving one symbolic linear plan."""
    if not isinstance(prepared, PreparedNonlinearSolve):
        raise TypeError("prepared must be a PreparedNonlinearSolve.")
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    if problem.problem_id != prepared.problem.problem_id:
        raise ValueError("Nonlinear refreshes must preserve problem_id.")
    state = problem.validate_state(initial_state)
    jacobian = prepare_jacobian(problem, state, prepared.method.jacobian_policy, args)
    problem_ = problem.bind_spaces(state, jacobian.residual)
    old_state_space = prepared.problem.state_space
    old_residual_space = prepared.problem.residual_space
    new_state_space = problem_.state_space
    new_residual_space = problem_.residual_space
    if (
        old_state_space is None
        or old_residual_space is None
        or new_state_space is None
        or new_residual_space is None
    ):
        raise ValueError("Prepared nonlinear refresh requires bound vector spaces.")
    if not old_state_space.compatible(new_state_space):
        raise ValueError("Nonlinear refresh changed the state space.")
    if not old_residual_space.compatible(new_residual_space):
        raise ValueError("Nonlinear refresh changed the residual space.")
    if jacobian.derivative_id != prepared.jacobian.derivative_id:
        raise ValueError("Nonlinear refresh changed the derivative structure.")
    linear_operator = _jacobian_solve_operator(jacobian.operator)
    if linear_operator.source.size != linear_operator.target.size:
        raise ValueError("Newton methods require a square Jacobian coordinate map.")
    prepared_linear, refresh_state = prepared.linear_refresh_state.refresh(
        LinearSystem(linear_operator, problem_id=prepared.linear_plan.problem_id)
    )
    recycling_policy = prepared_linear.plan.policy.recycling
    recycling = (
        None
        if prepared.run.recycling is None
        else refresh_recycling(
            prepared.run.recycling,
            prepared_linear,
            extraction=recycling_policy.extraction,
            refresh=recycling_policy.refresh,
        )
    )
    if refresh_state.template.template_id != prepared.linear_template_id:
        raise ValueError("Nonlinear refresh changed the symbolic linear template.")
    run = _refreshed_run(
        prepared,
        problem_,
        state,
        jacobian,
        refresh_state,
        args,
        recycling,
    )
    return PreparedNonlinearSolve(
        problem_,
        state,
        prepared.method,
        prepared.termination,
        args,
        jacobian,
        run,
        prepared.provenance,
        numeric_version=prepared.numeric_version + jnp.asarray(1, dtype=jnp.int32),
    )


def solve_prepared_nonlinear(
    prepared: PreparedNonlinearSolve,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> NonlinearResult:
    """Run the ordinary Newton iteration from prepared numeric state."""
    if not isinstance(prepared, PreparedNonlinearSolve):
        raise TypeError("prepared must be a PreparedNonlinearSolve.")
    termination_ = prepared.termination if termination is None else termination
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")
    return prepared.method.solve(
        prepared.problem,
        prepared.state,
        termination=termination_,
        args=prepared.args,
        _prepared_start=(
            prepared.problem,
            prepared.state,
            prepared.run,
            prepared.jacobian,
        ),
    )


def _seed_nonlinear_continuation(
    prepared: PreparedNonlinearSolve,
    problem: NonlinearSystemProblem,
    initial_state: PyTree[Any],
    /,
    *,
    args: Any = None,
    defer_refresh_steps: int = 0,
) -> PreparedNonlinearSolve:
    """Seed a new root with retained numerical Jacobian and Krylov state."""
    if not isinstance(prepared, PreparedNonlinearSolve):
        raise TypeError("prepared must be a PreparedNonlinearSolve.")
    if not isinstance(problem, NonlinearSystemProblem):
        raise TypeError("problem must be a NonlinearSystemProblem.")
    deferred = int(defer_refresh_steps)
    if deferred < 0:
        raise ValueError("defer_refresh_steps must be non-negative.")
    state = problem.validate_state(initial_state)
    residual, auxiliary = problem.evaluate(state, args)
    problem_ = problem.bind_spaces(state, residual)
    old_state_space = prepared.problem.state_space
    old_residual_space = prepared.problem.residual_space
    if (
        old_state_space is None
        or old_residual_space is None
        or problem_.state_space is None
        or problem_.residual_space is None
    ):
        raise ValueError("Nonlinear continuation requires bound vector spaces.")
    if not old_state_space.compatible(problem_.state_space):
        raise ValueError("Nonlinear continuation changed the state space.")
    if not old_residual_space.compatible(problem_.residual_space):
        raise ValueError("Nonlinear continuation changed the residual space.")
    operator = prepared.jacobian.operator
    if not operator.source.compatible(problem_.state_space):
        raise ValueError("Retained Jacobian source space is incompatible.")
    if not operator.target.compatible(problem_.residual_space):
        raise ValueError("Retained Jacobian target space is incompatible.")
    jacobian = PreparedJacobian(
        residual,
        operator,
        auxiliary=auxiliary,
        sparse_derivative=prepared.jacobian.sparse_derivative,
        derivative_id=prepared.jacobian.derivative_id,
    )
    run = _refreshed_run(
        prepared,
        problem_,
        state,
        jacobian,
        prepared.run.refresh_state,
        args,
        prepared.run.recycling,
    )
    run = eqx.tree_at(
        lambda value: (
            value.jacobian_preparations,
            value.setup_refreshes,
            value.numeric_refreshes,
            value.jacobian_age,
        ),
        run,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(-deferred, dtype=jnp.int32),
        ),
    )
    return PreparedNonlinearSolve(
        problem_,
        state,
        prepared.method,
        prepared.termination,
        args,
        jacobian,
        run,
        prepared.provenance,
        numeric_version=prepared.numeric_version,
    )


def _solve_prepared_nonlinear_stateful(
    prepared: PreparedNonlinearSolve,
    /,
    *,
    termination: NonlinearTermination | None = None,
) -> tuple[NonlinearResult, PreparedNonlinearSolve]:
    """Solve and retain the final numerical Newton state for temporal reuse."""
    if not isinstance(prepared, PreparedNonlinearSolve):
        raise TypeError("prepared must be a PreparedNonlinearSolve.")
    termination_ = prepared.termination if termination is None else termination
    if not isinstance(termination_, NonlinearTermination):
        raise TypeError("termination must be a NonlinearTermination or None.")
    result, state, run, jacobian = prepared.method.solve(
        prepared.problem,
        prepared.state,
        termination=termination_,
        args=prepared.args,
        _prepared_start=(
            prepared.problem,
            prepared.state,
            prepared.run,
            prepared.jacobian,
        ),
        _return_internal=True,
    )
    retained = PreparedNonlinearSolve(
        prepared.problem,
        state,
        prepared.method,
        termination_,
        prepared.args,
        jacobian,
        run,
        result.provenance,
        numeric_version=run.refresh_state.numeric_version,
    )
    return result, retained


__all__ = [
    "PreparedNonlinearSolve",
    "prepare_nonlinear",
    "refresh_nonlinear",
    "solve_prepared_nonlinear",
]
