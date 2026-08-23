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

from .._strict import StrictModule
from .._tree_math import validate_inexact_tree
from ..linalg import PyTreeSpace
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
from ._updates import (
    AbstractNonlinearUpdate,
    apply_prepared_nonlinear_update,
    prepare_nonlinear_update,
    PreparedNonlinearUpdate,
)


class _RichardsonSearch(StrictModule):
    state: Array
    residual: Array
    residual_norm: Array
    rate: Array
    steps: Array
    accepted: Array
    finite_valid_seen: Array
    domain_failures: Array
    nonfinite_trials: Array


class _RichardsonRun(StrictModule):
    state: Array
    residual: Array
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    iteration: Array
    residual_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    prepared_update: PreparedNonlinearUpdate
    status: Array


class NonlinearRichardson(AbstractNonlinearMethod):
    """Armijo-globalized iteration over one typed nonlinear update."""

    update: AbstractNonlinearUpdate
    sufficient_decrease: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    minimum_rate: float = eqx.field(static=True)
    maximum_search_steps: int = eqx.field(static=True)

    def __init__(
        self,
        update: AbstractNonlinearUpdate,
        /,
        *,
        sufficient_decrease: float = 1e-4,
        contraction: float = 0.5,
        minimum_rate: float = 1e-8,
        maximum_search_steps: int = 20,
    ):
        if not isinstance(update, AbstractNonlinearUpdate):
            raise TypeError("update must be AbstractNonlinearUpdate.")
        decrease = float(sufficient_decrease)
        contraction_ = float(contraction)
        minimum = float(minimum_rate)
        steps = int(maximum_search_steps)
        if not isfinite(decrease) or not 0.0 < decrease < 1.0:
            raise ValueError("sufficient_decrease must lie in (0, 1).")
        if not isfinite(contraction_) or not 0.0 < contraction_ < 1.0:
            raise ValueError("contraction must lie in (0, 1).")
        if not isfinite(minimum) or not 0.0 < minimum < 1.0:
            raise ValueError("minimum_rate must lie in (0, 1).")
        if steps < 1:
            raise ValueError("maximum_search_steps must be positive.")
        self.update = update
        self.sufficient_decrease = decrease
        self.contraction = contraction_
        self.minimum_rate = minimum
        self.maximum_search_steps = steps

    @property
    def method_id(self) -> str:
        return "nonlinear-richardson"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=True,
            jit=self.update.capabilities.jit,
            implicit_differentiation=False,
            fixed_point=True,
            nonlinear_preconditioning=True,
        )

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination,
        args: Any = None,
    ) -> NonlinearResult:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        initial = validate_inexact_tree(initial_state, name="initial Richardson state")
        residual, auxiliary = problem.evaluate(initial, args)
        source = PyTreeSpace(initial)
        target = PyTreeSpace(residual)
        state_coordinates = source.flatten(initial)
        residual_coordinates = target.flatten(residual)
        initial_norm = jnp.linalg.norm(residual_coordinates)
        finite = jnp.all(jnp.isfinite(state_coordinates)) & jnp.all(
            jnp.isfinite(residual_coordinates)
        )
        valid = problem.valid(initial, residual, auxiliary, args)
        converged = (
            finite
            & valid
            & (initial_norm <= termination.residual_threshold(initial_norm))
        )
        prepared_update = prepare_nonlinear_update(
            problem,
            initial,
            self.update,
            args=args,
        )
        prepared_dynamic, prepared_static = eqx.partition(
            prepared_update,
            eqx.is_array,
        )
        run = _RichardsonRun(
            state=state_coordinates,
            residual=residual_coordinates,
            initial_residual_norm=initial_norm,
            residual_norm=initial_norm,
            step_norm=jnp.asarray(0.0, dtype=initial_norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            residual_evaluations=jnp.asarray(1, dtype=jnp.int32),
            jacobian_preparations=jnp.asarray(0, dtype=jnp.int32),
            linear_solves=jnp.asarray(0, dtype=jnp.int32),
            linear_iterations=jnp.asarray(0, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            domain_failures=(finite & ~valid).astype(jnp.int32),
            nonfinite_trials=(~finite).astype(jnp.int32),
            prepared_update=prepared_dynamic,
            status=jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    finite & valid,
                    int(NonlinearStatus.ITERATING),
                    jnp.where(
                        finite,
                        int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                        int(NonlinearStatus.NONFINITE_INPUT),
                    ),
                ),
            ).astype(jnp.int32),
        )

        def condition(current):
            within_evaluations = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else current.residual_evaluations + 1 <= termination.maximum_evaluations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination.maximum_steps)
                & within_evaluations
            )

        def body(current):
            state_tree = source.unflatten(current.state)
            combined_prepared = eqx.combine(
                current.prepared_update,
                prepared_static,
            )
            update_result, next_prepared = apply_prepared_nonlinear_update(
                combined_prepared,
                state_tree,
                args=args,
            )
            next_prepared_dynamic, _ = eqx.partition(next_prepared, eqx.is_array)
            proposed_state = source.flatten(update_result.state)
            proposed_residual = target.flatten(update_result.residual)
            direction = proposed_state - current.state
            proposed_norm = jnp.linalg.norm(proposed_residual)
            proposed_finite = jnp.all(jnp.isfinite(proposed_state)) & jnp.all(
                jnp.isfinite(proposed_residual)
            )
            proposed_valid = update_result.applied
            proposed_accepted = (
                proposed_valid
                & proposed_finite
                & (
                    proposed_norm * proposed_norm
                    <= (1.0 - self.sufficient_decrease)
                    * current.residual_norm
                    * current.residual_norm
                )
            )
            search = _RichardsonSearch(
                state=proposed_state,
                residual=proposed_residual,
                residual_norm=proposed_norm,
                rate=jnp.asarray(1.0, dtype=proposed_norm.dtype),
                steps=jnp.asarray(0, dtype=jnp.int32),
                accepted=proposed_accepted,
                finite_valid_seen=proposed_finite & proposed_valid,
                domain_failures=update_result.diagnostics.domain_failures,
                nonfinite_trials=update_result.diagnostics.nonfinite_trials,
            )

            def search_condition(item):
                return (
                    update_result.applied
                    & ~item.accepted
                    & (item.steps < self.maximum_search_steps)
                    & (item.rate > self.minimum_rate)
                )

            def search_body(item):
                rate = self.contraction * item.rate
                trial_coordinates = current.state + rate * direction
                trial_state = source.unflatten(trial_coordinates)
                trial_residual_tree, trial_auxiliary = problem.evaluate(trial_state, args)
                trial_residual = target.flatten(trial_residual_tree)
                trial_norm = jnp.linalg.norm(trial_residual)
                trial_finite = jnp.all(jnp.isfinite(trial_coordinates)) & jnp.all(
                    jnp.isfinite(trial_residual)
                )
                trial_valid = problem.valid(
                    trial_state,
                    trial_residual_tree,
                    trial_auxiliary,
                    args,
                )
                accepted = (
                    trial_finite
                    & trial_valid
                    & (
                        trial_norm * trial_norm
                        <= (1.0 - self.sufficient_decrease * rate)
                        * current.residual_norm
                        * current.residual_norm
                    )
                )
                return _RichardsonSearch(
                    state=trial_coordinates,
                    residual=trial_residual,
                    residual_norm=trial_norm,
                    rate=rate,
                    steps=item.steps + 1,
                    accepted=accepted,
                    finite_valid_seen=item.finite_valid_seen
                    | (trial_finite & trial_valid),
                    domain_failures=item.domain_failures
                    + (trial_finite & ~trial_valid).astype(jnp.int32),
                    nonfinite_trials=item.nonfinite_trials
                    + (~trial_finite).astype(jnp.int32),
                )

            search = jax.lax.while_loop(search_condition, search_body, search)
            accepted_state = jnp.where(search.accepted, search.state, current.state)
            accepted_residual = jnp.where(
                search.accepted,
                search.residual,
                current.residual,
            )
            accepted_norm = jnp.where(
                search.accepted,
                search.residual_norm,
                current.residual_norm,
            )
            step_norm = jnp.linalg.norm(accepted_state - current.state)
            converged = search.accepted & (
                accepted_norm
                <= termination.residual_threshold(current.initial_residual_norm)
            )
            stagnated = (
                search.accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(jnp.linalg.norm(current.state))
                )
            )
            status = jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    stagnated,
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                    jnp.where(
                        search.accepted,
                        int(NonlinearStatus.ITERATING),
                        jnp.where(
                            search.finite_valid_seen,
                            int(NonlinearStatus.LINE_SEARCH_FAILED),
                            int(NonlinearStatus.NONFINITE_EVALUATION),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            return _RichardsonRun(
                state=accepted_state,
                residual=accepted_residual,
                initial_residual_norm=current.initial_residual_norm,
                residual_norm=accepted_norm,
                step_norm=step_norm,
                iteration=current.iteration + search.accepted.astype(jnp.int32),
                residual_evaluations=current.residual_evaluations
                + update_result.diagnostics.residual_evaluations
                + search.steps,
                jacobian_preparations=current.jacobian_preparations
                + update_result.diagnostics.jacobian_preparations,
                linear_solves=current.linear_solves
                + update_result.diagnostics.linear_solves,
                linear_iterations=current.linear_iterations
                + update_result.diagnostics.linear_iterations,
                accepted_steps=current.accepted_steps
                + update_result.diagnostics.accepted_steps
                + search.accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps
                + update_result.diagnostics.rejected_steps
                + (~search.accepted).astype(jnp.int32),
                domain_failures=current.domain_failures + search.domain_failures,
                nonfinite_trials=current.nonfinite_trials + search.nonfinite_trials,
                prepared_update=next_prepared_dynamic,
                status=status,
            )

        run = jax.lax.while_loop(condition, body, run)
        exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else run.residual_evaluations >= termination.maximum_evaluations
        )
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            jnp.where(
                exhausted,
                int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            ),
            run.status,
        ).astype(jnp.int32)
        final_state = source.unflatten(run.state)
        final_residual, final_auxiliary = problem.evaluate(final_state, args)
        final_coordinates = target.flatten(final_residual)
        final_norm = jnp.linalg.norm(final_coordinates)
        final_finite = jnp.all(jnp.isfinite(run.state)) & jnp.all(
            jnp.isfinite(final_coordinates)
        )
        final_valid = problem.valid(
            final_state,
            final_residual,
            final_auxiliary,
            args,
        )
        status = jnp.where(
            ~final_finite,
            int(NonlinearStatus.NONFINITE_EVALUATION),
            jnp.where(
                ~final_valid,
                int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                status,
            ),
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_residual_norm,
            final_residual_norm=final_norm,
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.residual_evaluations + 1,
            jacobian_preparations=run.jacobian_preparations,
            linear_solves=run.linear_solves,
            linear_iterations=run.linear_iterations,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            domain_failures=run.domain_failures,
            nonfinite_trials=run.nonfinite_trials,
            counts_complete=self.update.capabilities.counts_complete,
        )
        return NonlinearResult(
            state=final_state,
            residual=final_residual,
            auxiliary=final_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="none",
                globalization_id="physical-residual-armijo",
                notes=f"update={self.update.update_id}",
            ),
        )


__all__ = ["NonlinearRichardson"]
