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


class _NGMRESRun(StrictModule):
    state: Array
    residual: Array
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    iteration: Array
    residual_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    domain_failures: Array
    nonfinite_trials: Array
    restarts: Array
    history_states: Array
    history_residuals: Array
    history_count: Array
    status: Array
    prepared_update: PreparedNonlinearUpdate
    jvp_evaluations: Array
    vjp_evaluations: Array
    jacobian_preparations: Array
    linear_solves: Array
    linear_iterations: Array


class NonlinearGMRES(AbstractNonlinearMethod):
    """Nonlinear-preconditioned residual-minimizing affine acceleration."""

    update: AbstractNonlinearUpdate
    history: int
    regularization: float
    safeguard_factor: float

    def __init__(
        self,
        update: AbstractNonlinearUpdate,
        /,
        *,
        history: int = 8,
        regularization: float = 1e-10,
        safeguard_factor: float = 1.0,
    ):
        if not isinstance(update, AbstractNonlinearUpdate):
            raise TypeError("update must be AbstractNonlinearUpdate.")
        history_ = int(history)
        regularization_ = float(regularization)
        safeguard_ = float(safeguard_factor)
        if history_ < 1:
            raise ValueError("history must be positive.")
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("regularization must be finite and non-negative.")
        if not isfinite(safeguard_) or safeguard_ < 1.0:
            raise ValueError("safeguard_factor must be finite and at least one.")
        self.update = update
        self.history = history_
        self.regularization = regularization_
        self.safeguard_factor = safeguard_

    @property
    def method_id(self) -> str:
        return "nonlinear-gmres"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=True,
            jit=True,
            implicit_differentiation=False,
            nonlinear_preconditioning=True,
            fixed_point=True,
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
            raise TypeError("problem must be a NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be a NonlinearTermination.")
        initial = validate_inexact_tree(initial_state, name="initial NGMRES state")
        source = PyTreeSpace(initial)
        residual_tree, initial_auxiliary = problem.evaluate(initial, args)
        target = PyTreeSpace(residual_tree)
        initial_coordinates = source.flatten(initial)
        initial_residual = target.flatten(residual_tree)
        if initial_coordinates.dtype != initial_residual.dtype:
            raise TypeError("NGMRES state and residual coordinate dtypes must match.")
        initial_norm = jnp.linalg.norm(initial_residual)
        history_states = jnp.zeros(
            (self.history, source.size), dtype=initial_coordinates.dtype
        )
        history_residuals = jnp.zeros(
            (self.history, target.size), dtype=initial_residual.dtype
        )
        finite = jnp.all(jnp.isfinite(initial_coordinates)) & jnp.all(
            jnp.isfinite(initial_residual)
        )
        initial_valid = problem.valid(initial, residual_tree, initial_auxiliary, args)
        initial_converged = (
            finite
            & initial_valid
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
        run = _NGMRESRun(
            state=initial_coordinates,
            residual=initial_residual,
            initial_residual_norm=initial_norm,
            residual_norm=initial_norm,
            step_norm=jnp.asarray(0.0, dtype=initial_norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            residual_evaluations=jnp.asarray(1, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            domain_failures=jnp.asarray(0, dtype=jnp.int32),
            nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
            restarts=jnp.asarray(0, dtype=jnp.int32),
            history_states=history_states,
            history_residuals=history_residuals,
            history_count=jnp.asarray(0, dtype=jnp.int32),
            prepared_update=prepared_dynamic,
            jvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
            vjp_evaluations=jnp.asarray(0, dtype=jnp.int32),
            jacobian_preparations=jnp.asarray(0, dtype=jnp.int32),
            linear_solves=jnp.asarray(0, dtype=jnp.int32),
            linear_iterations=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.where(
                initial_converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    finite & initial_valid,
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
                else current.residual_evaluations + 2 <= termination.maximum_evaluations
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
            base_result, next_prepared_update = apply_prepared_nonlinear_update(
                combined_prepared,
                state_tree,
                args=args,
            )
            next_prepared_dynamic, _ = eqx.partition(
                next_prepared_update,
                eqx.is_array,
            )
            base_tree = base_result.state
            base = source.flatten(base_tree)
            base_residual_tree = base_result.residual
            base_residual = target.flatten(base_residual_tree)
            base_finite = jnp.all(jnp.isfinite(base)) & jnp.all(
                jnp.isfinite(base_residual)
            )
            base_valid = base_result.applied

            indices = jnp.arange(self.history)
            active = indices >= (self.history - current.history_count)
            delta_states = current.history_states - base[None, :]
            delta_residuals = current.history_residuals - base_residual[None, :]
            delta_states = jnp.where(active[:, None], delta_states, 0.0)
            delta_residuals = jnp.where(active[:, None], delta_residuals, 0.0)
            gram = jnp.conj(delta_residuals) @ delta_residuals.T
            diagonal = jnp.where(
                active,
                jnp.asarray(self.regularization, dtype=gram.real.dtype),
                jnp.asarray(1.0, dtype=gram.real.dtype),
            )
            gram = gram + jnp.diag(diagonal.astype(gram.dtype))
            right = -(jnp.conj(delta_residuals) @ base_residual)
            coefficients = jnp.linalg.solve(gram, right)
            coefficients = jnp.where(active, coefficients, 0.0)
            accelerated = base + jnp.sum(coefficients[:, None] * delta_states, axis=0)
            accelerated_tree = source.unflatten(accelerated)
            accelerated_residual_tree, accelerated_auxiliary = problem.evaluate(
                accelerated_tree, args
            )
            accelerated_residual = target.flatten(accelerated_residual_tree)
            accelerated_finite = jnp.all(jnp.isfinite(accelerated)) & jnp.all(
                jnp.isfinite(accelerated_residual)
            )
            accelerated_valid = problem.valid(
                accelerated_tree,
                accelerated_residual_tree,
                accelerated_auxiliary,
                args,
            )
            base_norm = jnp.linalg.norm(base_residual)
            accelerated_norm = jnp.linalg.norm(accelerated_residual)
            use_accelerated = (
                (current.history_count > 0)
                & base_finite
                & base_valid
                & accelerated_finite
                & accelerated_valid
                & (
                    accelerated_norm
                    <= self.safeguard_factor
                    * jnp.minimum(base_norm, current.residual_norm)
                )
            )
            use_base = base_finite & base_valid
            accepted = use_accelerated | use_base
            candidate = jnp.where(use_accelerated, accelerated, base)
            candidate_residual = jnp.where(
                use_accelerated, accelerated_residual, base_residual
            )
            candidate_norm = jnp.where(use_accelerated, accelerated_norm, base_norm)
            accepted_state = jnp.where(accepted, candidate, current.state)
            accepted_residual = jnp.where(accepted, candidate_residual, current.residual)
            accepted_norm = jnp.where(accepted, candidate_norm, current.residual_norm)
            step_norm = jnp.linalg.norm(accepted_state - current.state)
            converged = accepted & (
                accepted_norm
                <= termination.residual_threshold(current.initial_residual_norm)
            )
            stagnated = (
                accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(jnp.linalg.norm(current.state))
                )
            )
            diverged = accepted_norm > (
                termination.divergence_factor
                * jnp.maximum(current.initial_residual_norm, 1e-30)
            )
            any_finite_valid = (base_finite & base_valid) | (
                accelerated_finite & accelerated_valid
            )
            status = jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    stagnated,
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                    jnp.where(
                        diverged,
                        int(NonlinearStatus.DIVERGENCE),
                        jnp.where(
                            accepted,
                            int(NonlinearStatus.ITERATING),
                            jnp.where(
                                any_finite_valid,
                                int(NonlinearStatus.RECOVERABLE_DOMAIN_FAILURE),
                                int(NonlinearStatus.NONFINITE_EVALUATION),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            appended_states = jnp.concatenate(
                (current.history_states[1:], accepted_state[None, :]), axis=0
            )
            appended_residuals = jnp.concatenate(
                (current.history_residuals[1:], accepted_residual[None, :]), axis=0
            )
            rejected_acceleration = (current.history_count > 0) & ~use_accelerated
            restarted_states = (
                jnp.zeros_like(current.history_states).at[-1].set(accepted_state)
            )
            restarted_residuals = (
                jnp.zeros_like(current.history_residuals).at[-1].set(accepted_residual)
            )
            next_history_states = jnp.where(
                rejected_acceleration, restarted_states, appended_states
            )
            next_history_residuals = jnp.where(
                rejected_acceleration, restarted_residuals, appended_residuals
            )
            next_history_count = jnp.where(
                rejected_acceleration,
                jnp.asarray(1, dtype=jnp.int32),
                jnp.minimum(current.history_count + 1, self.history),
            )
            next_history_states = jnp.where(
                accepted, next_history_states, current.history_states
            )
            next_history_residuals = jnp.where(
                accepted, next_history_residuals, current.history_residuals
            )
            next_history_count = jnp.where(
                accepted, next_history_count, current.history_count
            )
            return _NGMRESRun(
                state=accepted_state,
                residual=accepted_residual,
                initial_residual_norm=current.initial_residual_norm,
                residual_norm=accepted_norm,
                step_norm=step_norm,
                iteration=current.iteration + accepted.astype(jnp.int32),
                residual_evaluations=(
                    current.residual_evaluations
                    + base_result.diagnostics.residual_evaluations
                    + 1
                ),
                accepted_steps=current.accepted_steps + accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps
                + rejected_acceleration.astype(jnp.int32)
                + (~accepted).astype(jnp.int32),
                domain_failures=current.domain_failures
                + base_result.diagnostics.domain_failures
                + (accelerated_finite & ~accelerated_valid).astype(jnp.int32),
                nonfinite_trials=current.nonfinite_trials
                + base_result.diagnostics.nonfinite_trials
                + (~accelerated_finite).astype(jnp.int32),
                restarts=current.restarts + rejected_acceleration.astype(jnp.int32),
                history_states=next_history_states,
                history_residuals=next_history_residuals,
                history_count=next_history_count,
                status=status,
                prepared_update=next_prepared_dynamic,
                jvp_evaluations=current.jvp_evaluations,
                vjp_evaluations=current.vjp_evaluations,
                jacobian_preparations=(
                    current.jacobian_preparations
                    + base_result.diagnostics.jacobian_preparations
                ),
                linear_solves=(
                    current.linear_solves + base_result.diagnostics.linear_solves
                ),
                linear_iterations=(
                    current.linear_iterations + base_result.diagnostics.linear_iterations
                ),
            )

        run = jax.lax.while_loop(condition, body, run)
        exhausted = (
            jnp.asarray(False)
            if termination.maximum_evaluations is None
            else run.residual_evaluations >= termination.maximum_evaluations
        )
        status = jnp.where(
            (run.status == int(NonlinearStatus.ITERATING)) & exhausted,
            int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
            run.status,
        )
        status = jnp.where(
            status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            status,
        ).astype(jnp.int32)
        final_state = source.unflatten(run.state)
        final_residual, final_auxiliary = problem.evaluate(final_state, args)
        final_coordinates = target.flatten(final_residual)
        final_norm = jnp.linalg.norm(final_coordinates)
        final_finite = jnp.all(jnp.isfinite(run.state)) & jnp.all(
            jnp.isfinite(final_coordinates)
        )
        final_valid = problem.valid(final_state, final_residual, final_auxiliary, args)
        preserve_input_failure = status == int(NonlinearStatus.NONFINITE_INPUT)
        status = jnp.where(
            preserve_input_failure,
            status,
            jnp.where(
                ~final_finite,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    ~final_valid,
                    int(NonlinearStatus.UNRECOVERABLE_DOMAIN_FAILURE),
                    status,
                ),
            ),
        ).astype(jnp.int32)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_residual_norm,
            final_residual_norm=final_norm,
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.residual_evaluations + 1,
            jvp_evaluations=run.jvp_evaluations,
            vjp_evaluations=run.vjp_evaluations,
            jacobian_preparations=run.jacobian_preparations,
            linear_solves=run.linear_solves,
            linear_iterations=run.linear_iterations,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            domain_failures=run.domain_failures
            + ((run.domain_failures == 0) & final_finite & ~final_valid).astype(
                jnp.int32
            ),
            nonfinite_trials=run.nonfinite_trials,
            acceleration_restarts=run.restarts,
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
                derivative_id="residual-affine-model",
                globalization_id="preconditioner-safeguard",
                notes=f"update={self.update.update_id}",
            ),
        )


__all__ = ["NonlinearGMRES"]
