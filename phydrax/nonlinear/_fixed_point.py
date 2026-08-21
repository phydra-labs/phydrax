#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import tree_add_scaled, tree_norm, validate_inexact_tree
from ..linalg import AbstractPreconditioner, PyTreeSpace
from ._types import (
    FixedPointProblem,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearStatus,
    NonlinearSystemProblem,
    NonlinearTermination,
)


class AndersonAcceleration(StrictModule):
    """Fixed-capacity regularized Type-II Anderson acceleration."""

    history: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    safeguard_factor: float = eqx.field(static=True)
    restart_condition: float = eqx.field(static=True)

    def __init__(
        self,
        *,
        history: int = 5,
        regularization: float = 1e-10,
        safeguard_factor: float = 2.0,
        restart_condition: float = 1e12,
    ):
        history_ = int(history)
        regularization_ = float(regularization)
        safeguard_ = float(safeguard_factor)
        condition_ = float(restart_condition)
        if history_ < 1:
            raise ValueError("Anderson history must be positive.")
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("Anderson regularization must be finite and non-negative.")
        if not isfinite(safeguard_) or safeguard_ < 1.0:
            raise ValueError("Anderson safeguard_factor must be finite and at least one.")
        if not isfinite(condition_) or condition_ <= 1.0:
            raise ValueError("Anderson restart_condition must be finite and exceed one.")
        self.history = history_
        self.regularization = regularization_
        self.safeguard_factor = safeguard_
        self.restart_condition = condition_


class _FixedPointRun(StrictModule):
    state: Array
    residual: Array
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    iteration: Array
    evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    nonfinite_trials: Array
    restarts: Array
    history_states: Array
    history_residuals: Array
    history_count: Array
    status: Array


def _anderson_candidate(
    mapped: Array,
    residual: Array,
    run: _FixedPointRun,
    policy: AndersonAcceleration,
    /,
) -> tuple[Array, Array, Array]:
    capacity = policy.history
    active_count = jnp.minimum(run.history_count, capacity)
    indices = jnp.arange(capacity)
    active = indices >= (capacity - active_count)
    previous_states = run.history_states[:capacity]
    previous_residuals = run.history_residuals[:capacity]
    next_states = run.history_states[1 : capacity + 1]
    next_residuals = run.history_residuals[1 : capacity + 1]
    delta_states = next_states - previous_states
    delta_residuals = next_residuals - previous_residuals
    delta_states = jnp.where(active[:, None], delta_states, 0.0)
    delta_residuals = jnp.where(active[:, None], delta_residuals, 0.0)
    gram = delta_residuals @ jnp.conj(delta_residuals.T)
    diagonal = jnp.where(
        active,
        jnp.asarray(policy.regularization, dtype=gram.real.dtype),
        jnp.asarray(1.0, dtype=gram.real.dtype),
    )
    gram = gram + jnp.diag(diagonal.astype(gram.dtype))
    right = delta_residuals @ jnp.conj(residual)
    coefficients = jnp.linalg.solve(gram, right)
    coefficients = jnp.where(active, coefficients, 0.0)
    correction = jnp.sum(coefficients[:, None] * (delta_states + delta_residuals), axis=0)
    candidate = mapped - correction
    singular_values = jnp.linalg.svd(gram, compute_uv=False)
    condition = singular_values[0] / jnp.maximum(singular_values[-1], 1e-30)
    usable = (
        (active_count > 0)
        & jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(condition)
        & (condition <= policy.restart_condition)
    )
    return jnp.where(usable, candidate, mapped), usable, condition


class FixedPointIteration(StrictModule):
    """Damped fixed-point iteration with optional safeguarded Anderson acceleration."""

    damping: float = eqx.field(static=True)
    acceleration: AndersonAcceleration | None

    def __init__(
        self,
        *,
        damping: float = 1.0,
        acceleration: AndersonAcceleration | None = None,
    ):
        damping_ = float(damping)
        if not isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("Fixed-point damping must lie in (0, 1].")
        if acceleration is not None and not isinstance(
            acceleration, AndersonAcceleration
        ):
            raise TypeError("acceleration must be AndersonAcceleration or None.")
        self.damping = damping_
        self.acceleration = acceleration

    @property
    def method_id(self) -> str:
        return "anderson-fixed-point" if self.acceleration is not None else "fixed-point"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=False,
            jit=True,
            implicit_differentiation=False,
            fixed_point=True,
        )

    def solve(
        self,
        problem: FixedPointProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination | None = None,
        args: Any = None,
    ) -> NonlinearResult:
        if not isinstance(problem, FixedPointProblem):
            raise TypeError("problem must be a FixedPointProblem.")
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        initial = validate_inexact_tree(initial_state, name="initial fixed-point state")
        space = PyTreeSpace(initial)
        flat_initial = space.flatten(initial)
        mapped = problem.mapping(initial, args)
        flat_mapped = space.flatten(mapped)
        residual = flat_mapped - flat_initial
        residual_norm = jnp.linalg.norm(residual)
        capacity = 1 if self.acceleration is None else self.acceleration.history + 1
        history_states = (
            jnp.zeros((capacity, space.size), dtype=flat_initial.dtype)
            .at[-1]
            .set(flat_initial)
        )
        history_residuals = (
            jnp.zeros((capacity, space.size), dtype=residual.dtype).at[-1].set(residual)
        )
        status = jnp.where(
            jnp.all(jnp.isfinite(flat_initial)) & jnp.all(jnp.isfinite(residual)),
            int(NonlinearStatus.ITERATING),
            jnp.where(
                jnp.all(jnp.isfinite(flat_initial)),
                int(NonlinearStatus.NONFINITE_EVALUATION),
                int(NonlinearStatus.NONFINITE_INPUT),
            ),
        ).astype(jnp.int32)
        run = _FixedPointRun(
            state=flat_initial,
            residual=residual,
            initial_residual_norm=residual_norm,
            residual_norm=residual_norm,
            step_norm=jnp.asarray(0.0, dtype=residual_norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            evaluations=jnp.asarray(1, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            nonfinite_trials=jnp.asarray(0, dtype=jnp.int32),
            restarts=jnp.asarray(0, dtype=jnp.int32),
            history_states=history_states,
            history_residuals=history_residuals,
            history_count=jnp.asarray(0, dtype=jnp.int32),
            status=status,
        )

        evaluations_per_step = 1 if self.acceleration is None else 2

        def condition(current):
            within_evaluations = (
                jnp.asarray(True)
                if termination_.maximum_evaluations is None
                else current.evaluations + evaluations_per_step
                <= termination_.maximum_evaluations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination_.maximum_steps)
                & within_evaluations
            )

        def body(current):
            raw = current.state + self.damping * current.residual
            if self.acceleration is None:
                proposed = raw
                accelerated = jnp.asarray(False)
                condition_estimate = jnp.asarray(1.0, dtype=residual_norm.dtype)
            else:
                proposed, accelerated, condition_estimate = _anderson_candidate(
                    raw,
                    raw - current.state,
                    current,
                    self.acceleration,
                )
            proposed_tree = space.unflatten(proposed)
            next_mapped = space.flatten(problem.mapping(proposed_tree, args))
            next_residual = next_mapped - proposed
            next_norm = jnp.linalg.norm(next_residual)
            if self.acceleration is None:
                raw_residual = next_residual
                raw_norm = next_norm
            else:
                raw_mapped = space.flatten(problem.mapping(space.unflatten(raw), args))
                raw_residual = raw_mapped - raw
                raw_norm = jnp.linalg.norm(raw_residual)
            safeguard = (
                jnp.asarray(True)
                if self.acceleration is None
                else (~accelerated)
                | (next_norm <= self.acceleration.safeguard_factor * raw_norm)
            )
            accepted_state = jnp.where(safeguard, proposed, raw)
            accepted_residual = jnp.where(safeguard, next_residual, raw_residual)
            accepted_norm = jnp.where(safeguard, next_norm, raw_norm)
            finite = jnp.all(jnp.isfinite(accepted_state)) & jnp.all(
                jnp.isfinite(accepted_residual)
            )
            step_norm = jnp.linalg.norm(accepted_state - current.state)
            converged = finite & (
                accepted_norm
                <= termination_.residual_threshold(current.initial_residual_norm)
            )
            stagnated = (
                finite
                & ~converged
                & (
                    step_norm
                    <= termination_.step_threshold(jnp.linalg.norm(current.state))
                )
            )
            diverged = accepted_norm > (
                termination_.divergence_factor
                * jnp.maximum(current.initial_residual_norm, 1e-30)
            )
            next_status = jnp.where(
                ~finite,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    converged,
                    int(NonlinearStatus.SUCCESS),
                    jnp.where(
                        stagnated,
                        int(NonlinearStatus.RESIDUAL_STAGNATION),
                        jnp.where(
                            diverged,
                            int(NonlinearStatus.DIVERGENCE),
                            int(NonlinearStatus.ITERATING),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            restart = (
                jnp.asarray(False)
                if self.acceleration is None
                else (current.history_count > 0)
                & (
                    (accelerated & ~safeguard)
                    | ~jnp.isfinite(condition_estimate)
                    | (condition_estimate > self.acceleration.restart_condition)
                )
            )
            shifted_states = jnp.concatenate(
                (current.history_states[1:], accepted_state[None, :]), axis=0
            )
            shifted_residuals = jnp.concatenate(
                (current.history_residuals[1:], accepted_residual[None, :]), axis=0
            )
            reset_states = jnp.zeros_like(shifted_states).at[-1].set(accepted_state)
            reset_residuals = (
                jnp.zeros_like(shifted_residuals).at[-1].set(accepted_residual)
            )
            next_states = jnp.where(restart, reset_states, shifted_states)
            next_residuals = jnp.where(restart, reset_residuals, shifted_residuals)
            next_count = jnp.where(
                restart,
                0,
                jnp.minimum(current.history_count + 1, capacity - 1),
            )
            return _FixedPointRun(
                state=jnp.where(finite, accepted_state, current.state),
                residual=jnp.where(finite, accepted_residual, current.residual),
                initial_residual_norm=current.initial_residual_norm,
                residual_norm=jnp.where(finite, accepted_norm, current.residual_norm),
                step_norm=jnp.where(finite, step_norm, current.step_norm),
                iteration=current.iteration + finite.astype(jnp.int32),
                evaluations=current.evaluations + evaluations_per_step,
                accepted_steps=current.accepted_steps + finite.astype(jnp.int32),
                rejected_steps=current.rejected_steps + (~finite).astype(jnp.int32),
                nonfinite_trials=current.nonfinite_trials + (~finite).astype(jnp.int32),
                restarts=current.restarts + restart.astype(jnp.int32),
                history_states=next_states,
                history_residuals=next_residuals,
                history_count=next_count,
                status=next_status,
            )

        run = jax.lax.while_loop(condition, body, run)
        status = run.status
        exhausted = (
            jnp.asarray(False)
            if termination_.maximum_evaluations is None
            else run.evaluations >= termination_.maximum_evaluations
        )
        status = jnp.where(
            (status == int(NonlinearStatus.ITERATING)) & exhausted,
            int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
            status,
        )
        status = jnp.where(
            status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            status,
        ).astype(jnp.int32)
        final_state = space.unflatten(run.state)
        final_residual = space.unflatten(run.residual)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_residual_norm,
            final_residual_norm=run.residual_norm,
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.evaluations,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            nonfinite_trials=run.nonfinite_trials,
            acceleration_restarts=run.restarts,
        )
        return NonlinearResult(
            state=final_state,
            residual=final_residual,
            auxiliary=None,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="none",
                globalization_id="fixed-point-safeguard",
            ),
        )


class PicardIteration(StrictModule):
    """Preconditioned Picard iteration for a physical nonlinear residual."""

    inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner
    damping: float = eqx.field(static=True)
    acceleration: AndersonAcceleration | None

    def __init__(
        self,
        inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner,
        /,
        *,
        damping: float = 1.0,
        acceleration: AndersonAcceleration | None = None,
    ):
        if not callable(inverse_action) and not isinstance(
            inverse_action, AbstractPreconditioner
        ):
            raise TypeError("inverse_action must be callable or AbstractPreconditioner.")
        if not isfinite(float(damping)) or not 0.0 < float(damping) <= 1.0:
            raise ValueError("Picard damping must lie in (0, 1].")
        if acceleration is not None and not isinstance(
            acceleration, AndersonAcceleration
        ):
            raise TypeError("acceleration must be AndersonAcceleration or None.")
        self.inverse_action = inverse_action
        self.damping = float(damping)
        self.acceleration = acceleration

    @property
    def method_id(self) -> str:
        return "picard-anderson" if self.acceleration is not None else "picard"

    def solve(
        self,
        problem: NonlinearSystemProblem,
        initial_state: PyTree[Any],
        /,
        *,
        termination: NonlinearTermination | None = None,
        args: Any = None,
    ) -> NonlinearResult:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be a NonlinearSystemProblem.")

        def mapping(state, current_args):
            residual = problem.residual(state, current_args)
            correction = (
                self.inverse_action.apply(residual)
                if isinstance(self.inverse_action, AbstractPreconditioner)
                else self.inverse_action(residual)
            )
            return tree_add_scaled(state, correction, -self.damping)

        fixed_problem = FixedPointProblem(
            mapping,
            problem_id=f"{problem.problem_id}/picard-map",
        )
        result = FixedPointIteration(
            damping=1.0,
            acceleration=self.acceleration,
        ).solve(fixed_problem, initial_state, termination=termination, args=args)
        physical_residual, auxiliary = problem.evaluate(result.state, args)
        physical_norm = tree_norm(physical_residual)
        termination_ = NonlinearTermination() if termination is None else termination
        successful = physical_norm <= termination_.residual_threshold(
            result.diagnostics.initial_residual_norm
        )
        status = jnp.where(
            successful,
            int(NonlinearStatus.SUCCESS),
            jnp.where(
                result.status == int(NonlinearStatus.SUCCESS),
                int(NonlinearStatus.RESIDUAL_STAGNATION),
                result.status,
            ),
        ).astype(jnp.int32)
        diagnostics = eqx.tree_at(
            lambda item: (item.final_residual_norm, item.residual_evaluations),
            result.diagnostics,
            (physical_norm, result.diagnostics.residual_evaluations + 1),
        )
        return NonlinearResult(
            state=result.state,
            residual=physical_residual,
            auxiliary=auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="preconditioned-residual-map",
                globalization_id="fixed-point-safeguard",
            ),
        )


__all__ = ["AndersonAcceleration", "FixedPointIteration", "PicardIteration"]
