#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._tree_math import tree_allfinite
from ..linalg import (
    IdentityLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    ScaledLinearOperator,
    solve as solve_linear,
    SumLinearOperator,
)
from ._linearization import JacobianPolicy, prepare_jacobian
from ._types import (
    AbstractNonlinearMethod,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


class _PseudoRun(eqx.Module):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    initial_norm: Array
    norm: Array
    step_norm: Array
    pseudo_step: Array
    iteration: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    final_linear_status: Array
    status: Array


class PseudoTransient(AbstractNonlinearMethod):
    """SER-adaptive pseudo-transient continuation for difficult roots."""

    jacobian: JacobianPolicy
    linear: LinearSolvePolicy
    initial_step: float = eqx.field(static=True)
    minimum_step: float = eqx.field(static=True)
    maximum_step: float = eqx.field(static=True)
    growth_limit: float = eqx.field(static=True)
    rejection_shrink: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        jacobian: JacobianPolicy | None = None,
        linear: LinearSolvePolicy | None = None,
        initial_step: float = 1e-2,
        minimum_step: float = 1e-12,
        maximum_step: float = 1e12,
        growth_limit: float = 10.0,
        rejection_shrink: float = 0.25,
    ):
        jacobian_ = JacobianPolicy() if jacobian is None else jacobian
        linear_ = LinearSolvePolicy() if linear is None else linear
        if not isinstance(jacobian_, JacobianPolicy):
            raise TypeError("jacobian must be JacobianPolicy or None.")
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        values = tuple(
            float(value)
            for value in (
                initial_step,
                minimum_step,
                maximum_step,
                growth_limit,
                rejection_shrink,
            )
        )
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("Pseudo-transient controls must be finite and positive.")
        if not values[1] <= values[0] <= values[2]:
            raise ValueError("Pseudo steps must satisfy minimum <= initial <= maximum.")
        if values[3] < 1.0 or values[4] >= 1.0:
            raise ValueError(
                "growth_limit must exceed one and rejection_shrink be below one."
            )
        self.jacobian = jacobian_
        self.linear = linear_
        (
            self.initial_step,
            self.minimum_step,
            self.maximum_step,
            self.growth_limit,
            self.rejection_shrink,
        ) = values

    @property
    def method_id(self) -> str:
        return "pseudo-transient-ser"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=False,
            jit=True,
            implicit_differentiation=True,
        )

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
        _initial_evaluation=None,
    ) -> NonlinearResult:
        if _initial_evaluation is None:
            state = problem.validate_state(initial_state)
            residual, auxiliary = problem.evaluate(state, args)
            problem_ = problem.bind_spaces(state, residual)
            initial_evaluations = 1
        else:
            problem_, state, residual, auxiliary = _initial_evaluation
            state = problem_.validate_state(state)
            residual = problem_.validate_residual(residual)
            initial_evaluations = 0
        if problem_.state_space is None or problem_.residual_space is None:
            raise ValueError("Pseudo-transient solve requires bound spaces.")
        if not problem_.state_space.compatible(problem_.residual_space):
            raise ValueError(
                "Pseudo-transient mass identity requires compatible state/residual spaces."
            )
        norm = jnp.sqrt(
            jnp.maximum(
                jnp.real(problem_.residual_space.inner(residual, residual)),
                0.0,
            )
        )
        finite = tree_allfinite(state) & tree_allfinite(residual)
        valid = problem_.valid(state, residual, auxiliary, args)
        converged = finite & valid & (norm <= termination.residual_threshold(norm))
        run = _PseudoRun(
            state=state,
            residual=residual,
            auxiliary=auxiliary,
            initial_norm=jnp.maximum(norm, 1e-30),
            norm=norm,
            step_norm=jnp.asarray(0.0, dtype=norm.dtype),
            pseudo_step=jnp.asarray(self.initial_step, dtype=norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            residual_evaluations=jnp.asarray(
                initial_evaluations,
                dtype=jnp.int32,
            ),
            jvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
            jacobian_preparations=jnp.asarray(0, dtype=jnp.int32),
            linear_solves=jnp.asarray(0, dtype=jnp.int32),
            linear_iterations=jnp.asarray(0, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            domain_failures=(finite & ~valid).astype(jnp.int32),
            nonfinite_trials=(~finite).astype(jnp.int32),
            final_linear_status=jnp.asarray(-1, dtype=jnp.int32),
            status=jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    finite & valid,
                    int(NonlinearStatus.ITERATING),
                    int(NonlinearStatus.NONFINITE_INPUT),
                ),
            ).astype(jnp.int32),
        )

        def condition(current):
            within_evaluations = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else current.residual_evaluations + 2 <= termination.maximum_evaluations
            )
            within_linear = (
                jnp.asarray(True)
                if termination.maximum_linear_iterations is None
                else current.linear_iterations < termination.maximum_linear_iterations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination.maximum_steps)
                & within_evaluations
                & within_linear
            )

        def body(current):
            jacobian = prepare_jacobian(problem_, current.state, self.jacobian, args)
            identity = IdentityLinearOperator(problem_.state_space)
            operator = SumLinearOperator(
                jacobian.operator,
                ScaledLinearOperator(identity, 1.0 / current.pseudo_step),
            )
            linear_result = solve_linear(
                LinearSystem(operator),
                jax.tree.map(jnp.negative, jacobian.residual),
                policy=self.linear,
            )
            direction = linear_result.value
            candidate = jax.tree.map(
                lambda value, delta: value + delta,
                current.state,
                direction,
            )
            candidate_residual, candidate_auxiliary = problem_.evaluate(candidate, args)
            candidate_norm = jnp.sqrt(
                jnp.maximum(
                    jnp.real(
                        problem_.residual_space.inner(
                            candidate_residual, candidate_residual
                        )
                    ),
                    0.0,
                )
            )
            candidate_finite = tree_allfinite(candidate) & tree_allfinite(
                candidate_residual
            )
            candidate_valid = problem_.valid(
                candidate, candidate_residual, candidate_auxiliary, args
            )
            accepted = (
                linear_result.diagnostics.converged
                & candidate_finite
                & candidate_valid
                & (candidate_norm < current.norm)
            )
            ratio = current.norm / jnp.maximum(candidate_norm, 1e-30)
            grown = current.pseudo_step * jnp.minimum(ratio, self.growth_limit)
            next_pseudo_step = jnp.where(
                accepted,
                jnp.clip(grown, self.minimum_step, self.maximum_step),
                jnp.maximum(
                    self.minimum_step,
                    self.rejection_shrink * current.pseudo_step,
                ),
            )
            step_norm = jnp.sqrt(
                jnp.maximum(
                    jnp.real(problem_.state_space.inner(direction, direction)),
                    0.0,
                )
            )
            converged = accepted & (
                candidate_norm <= termination.residual_threshold(current.initial_norm)
            )
            stagnated = (
                accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(
                        jnp.sqrt(
                            jnp.maximum(
                                jnp.real(
                                    problem_.state_space.inner(
                                        current.state, current.state
                                    )
                                ),
                                0.0,
                            )
                        )
                    )
                )
            )
            linear_iterations = jnp.sum(
                linear_result.diagnostics.iterations,
                dtype=jnp.int32,
            )
            next_linear = current.linear_iterations + linear_iterations
            linear_exhausted = (
                jnp.asarray(False)
                if termination.maximum_linear_iterations is None
                else next_linear >= termination.maximum_linear_iterations
            )
            failed_at_floor = (~accepted) & (next_pseudo_step <= self.minimum_step)
            status = jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    stagnated,
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                    jnp.where(
                        linear_exhausted,
                        int(NonlinearStatus.MAXIMUM_LINEAR_ITERATIONS_REACHED),
                        jnp.where(
                            failed_at_floor,
                            int(NonlinearStatus.LINE_SEARCH_FAILED),
                            int(NonlinearStatus.ITERATING),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            return _PseudoRun(
                state=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    candidate,
                    current.state,
                ),
                residual=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    candidate_residual,
                    current.residual,
                ),
                auxiliary=jax.tree.map(
                    lambda proposed, old: jnp.where(accepted, proposed, old),
                    candidate_auxiliary,
                    current.auxiliary,
                ),
                initial_norm=current.initial_norm,
                norm=jnp.where(accepted, candidate_norm, current.norm),
                step_norm=step_norm,
                pseudo_step=next_pseudo_step,
                iteration=current.iteration + 1,
                residual_evaluations=current.residual_evaluations
                + jacobian.residual_evaluations
                + 1,
                jvp_evaluations=current.jvp_evaluations
                + jnp.sum(
                    linear_result.diagnostics.matvec_count,
                    dtype=jnp.int32,
                ),
                jacobian_preparations=current.jacobian_preparations + 1,
                linear_solves=current.linear_solves + 1,
                linear_iterations=next_linear,
                accepted_steps=current.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps + (~accepted).astype(jnp.int32),
                domain_failures=current.domain_failures
                + (candidate_finite & ~candidate_valid).astype(jnp.int32),
                nonfinite_trials=current.nonfinite_trials
                + (~candidate_finite).astype(jnp.int32),
                final_linear_status=linear_result.status,
                status=status,
            )

        run = jax.lax.while_loop(condition, body, run)
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            run.status,
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_norm,
            final_residual_norm=run.norm,
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.residual_evaluations,
            jvp_evaluations=run.jvp_evaluations,
            jacobian_preparations=run.jacobian_preparations,
            linear_solves=run.linear_solves,
            linear_iterations=run.linear_iterations,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            domain_failures=run.domain_failures,
            nonfinite_trials=run.nonfinite_trials,
            final_trust_radius=run.pseudo_step,
            final_linear_status=run.final_linear_status,
        )
        return NonlinearResult(
            state=run.state,
            residual=run.residual,
            auxiliary=run.auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem_.problem_id,
                method_id=self.method_id,
                derivative_id=self.jacobian.mode,
                globalization_id="pseudo-time-ser",
            ),
        )


__all__ = ["PseudoTransient"]
