#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PyTree

from .._tree_math import tree_add_scaled, tree_allfinite, tree_where
from ..linalg import (
    AbstractLinearOperator,
    GMRES,
    LinearSolvePolicy,
    LinearSolveStatus,
    LinearSystem,
    prepare as prepare_linear,
    PreparedLinearSolve,
    refresh as refresh_linear,
    solve as solve_linear,
    TolerancePolicy,
)
from ._updates import (
    _provenance,
    _space_norm,
    AbstractNonlinearUpdate,
    NonlinearUpdateCapabilities,
    NonlinearUpdateControl,
    NonlinearUpdateDiagnostics,
    NonlinearUpdateResult,
    NonlinearUpdateStatus,
    PreparedNonlinearUpdate,
    skipped_nonlinear_update_result,
)
from ._work import NonlinearWork


def _default_linear_policy() -> LinearSolvePolicy:
    return LinearSolvePolicy(
        GMRES(restart=16),
        tolerance=TolerancePolicy(relative=1e-6, absolute=1e-10, max_steps=64),
    )


class LaggedLinearSolveUpdate(AbstractNonlinearUpdate):
    """One refreshed lagged-linear correction for a physical nonlinear residual.

    ``operator_function(state, args)`` returns a linear operator ``B(state)``. The
    update solves ``B(state) direction = -residual(state)`` and proposes the damped
    correction. Application only certifies a finite, physically valid proposal; an
    outer nonlinear method remains responsible for root convergence.
    """

    operator_function: Any
    linear_policy: LinearSolvePolicy
    damping: float = eqx.field(static=True)
    update_name: str = eqx.field(static=True)

    def __init__(
        self,
        operator_function: Any,
        /,
        *,
        linear_policy: LinearSolvePolicy | None = None,
        damping: float = 1.0,
        update_id: str = "lagged-linear-solve",
    ):
        if not callable(operator_function):
            raise TypeError("operator_function must be callable.")
        policy = _default_linear_policy() if linear_policy is None else linear_policy
        if not isinstance(policy, LinearSolvePolicy):
            raise TypeError("linear_policy must be LinearSolvePolicy or None.")
        damping_ = float(damping)
        if not isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("Lagged linear damping must lie in (0, 1].")
        identifier = str(update_id)
        if not identifier:
            raise ValueError("update_id must be non-empty.")
        self.operator_function = operator_function
        self.linear_policy = policy
        self.damping = damping_
        self.update_name = identifier

    @property
    def update_id(self) -> str:
        return self.update_name

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=True,
            prepared_refresh=True,
            differentiable_action=self.linear_policy.differentiation.mode != "none",
            exposes_linearization=True,
            counts_complete=self.maximum_work.complete,
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        maximum_steps = self.linear_policy.tolerance.max_steps
        bounded_steps = 0 if maximum_steps is None else maximum_steps
        action_bound = 0 if maximum_steps is None else 4 * maximum_steps + 4
        complete = maximum_steps is not None and self.linear_policy.preconditioning is None
        return NonlinearWork(
            residual_evaluations=2,
            validity_evaluations=1,
            jvp_evaluations=action_bound,
            vjp_evaluations=action_bound,
            linear_refreshes=1,
            linear_solves=1,
            linear_iterations=bounded_steps,
            complete=complete,
        )

    def _linear_problem(self, problem, state, args, /) -> LinearSystem:
        if problem.state_space is None or problem.residual_space is None:
            raise ValueError("Lagged linear updates require bound vector spaces.")
        operator = self.operator_function(state, args)
        if not isinstance(operator, AbstractLinearOperator):
            raise TypeError(
                "operator_function must return an AbstractLinearOperator."
            )
        if not problem.state_space.compatible(operator.source):
            raise ValueError(
                "Lagged operator source must match the nonlinear state space."
            )
        if not problem.residual_space.compatible(operator.target):
            raise ValueError(
                "Lagged operator target must match the nonlinear residual space."
            )
        if operator.source.size != operator.target.size:
            raise ValueError("Lagged root updates require a square linear operator.")
        return LinearSystem(operator)

    def _prepare_internal(self, problem, state, args, /) -> PreparedLinearSolve:
        return prepare_linear(
            self._linear_problem(problem, state, args),
            self.linear_policy,
        )

    def _refresh_internal(
        self,
        internal_state: Any,
        problem,
        state,
        args,
        /,
    ) -> PreparedLinearSolve:
        if not isinstance(internal_state, PreparedLinearSolve):
            raise TypeError("Prepared lagged linear update state is invalid.")
        return refresh_linear(
            internal_state,
            self._linear_problem(problem, state, args),
        )

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ) -> tuple[NonlinearUpdateResult, Any]:
        state_ = prepared.plan.state_space.validate(state)
        internal_dynamic, internal_static = eqx.partition(
            prepared.internal_state,
            eqx.is_array,
        )

        def skipped(_):
            return (
                skipped_nonlinear_update_result(
                    prepared,
                    state_,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    failure_origin="lagged-linear-budget",
                ),
                internal_dynamic,
            )

        def execute(_):
            problem = prepared.problem
            initial_residual, initial_auxiliary = problem.evaluate(state_, args)
            initial_norm = _space_norm(
                prepared.plan.residual_space,
                initial_residual,
            )
            initial_finite = tree_allfinite(state_) & tree_allfinite(initial_residual)

            def nonfinite_input(__):
                work = NonlinearWork(residual_evaluations=1)
                diagnostics = NonlinearUpdateDiagnostics(
                    initial_residual_norm=initial_norm,
                    final_residual_norm=initial_norm,
                    step_norm=0.0,
                    work=work,
                    rejected_steps=1,
                    nonfinite_trials=1,
                )
                return (
                    NonlinearUpdateResult(
                        state=state_,
                        residual=initial_residual,
                        auxiliary=initial_auxiliary,
                        status=NonlinearUpdateStatus.NONFINITE_INPUT,
                        diagnostics=diagnostics,
                        provenance=_provenance(prepared),
                    ),
                    internal_dynamic,
                )

            def solve_direction(__):
                combined = eqx.combine(internal_dynamic, internal_static)
                refreshed = self._refresh_internal(
                    combined,
                    problem,
                    state_,
                    args,
                )
                right_hand_side = jax.tree.map(jnp.negative, initial_residual)
                linear_result = solve_linear(refreshed, right_hand_side)
                direction = prepared.plan.state_space.validate(linear_result.value)
                iterations = jnp.sum(
                    linear_result.diagnostics.iterations,
                    dtype=jnp.int32,
                )
                work = NonlinearWork(
                    residual_evaluations=2,
                    validity_evaluations=1,
                    jvp_evaluations=jnp.sum(
                        linear_result.diagnostics.matvec_count,
                        dtype=jnp.int32,
                    ),
                    vjp_evaluations=jnp.sum(
                        linear_result.diagnostics.adjoint_matvec_count,
                        dtype=jnp.int32,
                    ),
                    linear_refreshes=1,
                    linear_solves=1,
                    linear_iterations=iterations,
                    complete=self.maximum_work.complete,
                )
                linear_success = (
                    linear_result.status == int(LinearSolveStatus.SUCCESS)
                ) & linear_result.diagnostics.converged
                direction_finite = tree_allfinite(direction)

                def failed_direction(___):
                    status = jnp.where(
                        linear_success,
                        int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                        int(NonlinearUpdateStatus.LINEAR_FAILURE),
                    ).astype(jnp.int32)
                    diagnostics = NonlinearUpdateDiagnostics(
                        initial_residual_norm=initial_norm,
                        final_residual_norm=initial_norm,
                        step_norm=0.0,
                        work=work,
                        rejected_steps=1,
                        nonfinite_trials=(~direction_finite).astype(jnp.int32),
                    )
                    return NonlinearUpdateResult(
                        state=state_,
                        residual=initial_residual,
                        auxiliary=initial_auxiliary,
                        status=status,
                        inner_status=linear_result.status,
                        diagnostics=diagnostics,
                        provenance=_provenance(prepared),
                    )

                def evaluate_candidate(___):
                    candidate = prepared.plan.state_space.validate(
                        tree_add_scaled(state_, direction, self.damping)
                    )
                    residual, auxiliary = problem.evaluate(candidate, args)
                    finite = tree_allfinite(candidate) & tree_allfinite(residual)
                    valid = problem.valid(candidate, residual, auxiliary, args)
                    applied = finite & valid
                    status = jnp.where(
                        ~finite,
                        int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                        jnp.where(
                            ~valid,
                            int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                            int(NonlinearUpdateStatus.APPLIED),
                        ),
                    ).astype(jnp.int32)
                    output_state = tree_where(applied, candidate, state_)
                    output_residual = tree_where(applied, residual, initial_residual)
                    output_auxiliary = tree_where(
                        applied,
                        auxiliary,
                        initial_auxiliary,
                    )
                    step = jax.tree.map(
                        lambda new, old: new - old,
                        output_state,
                        state_,
                    )
                    diagnostics = NonlinearUpdateDiagnostics(
                        initial_residual_norm=initial_norm,
                        final_residual_norm=jnp.where(
                            applied,
                            _space_norm(prepared.plan.residual_space, residual),
                            initial_norm,
                        ),
                        step_norm=_space_norm(prepared.plan.state_space, step),
                        work=work,
                        accepted_steps=applied.astype(jnp.int32),
                        rejected_steps=(~applied).astype(jnp.int32),
                        domain_failures=(finite & ~valid).astype(jnp.int32),
                        nonfinite_trials=(~finite).astype(jnp.int32),
                    )
                    return NonlinearUpdateResult(
                        state=output_state,
                        residual=output_residual,
                        auxiliary=output_auxiliary,
                        status=status,
                        inner_status=linear_result.status,
                        diagnostics=diagnostics,
                        provenance=_provenance(prepared),
                    )

                result = jax.lax.cond(
                    linear_success & direction_finite,
                    evaluate_candidate,
                    failed_direction,
                    operand=None,
                )
                return result, eqx.partition(refreshed, eqx.is_array)[0]

            return jax.lax.cond(
                initial_finite,
                solve_direction,
                nonfinite_input,
                operand=None,
            )

        result, next_dynamic = jax.lax.cond(
            control.permits(self.maximum_work),
            execute,
            skipped,
            operand=None,
        )
        return result, eqx.combine(next_dynamic, internal_static)


__all__ = ["LaggedLinearSolveUpdate"]
