#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Callable
from math import isfinite
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._strict import StrictModule
from .._tree_math import (
    tree_add_scaled,
    tree_allfinite,
    validate_inexact_tree,
)
from ..linalg import (
    AbstractPreconditioner,
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    PyTreeSpace,
    solve as solve_linear,
)
from ._precision import NonlinearPrecisionPolicy
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
from ._updates import (
    AbstractNonlinearUpdate,
    NonlinearUpdateCapabilities,
    NonlinearUpdateControl,
    NonlinearUpdateDiagnostics,
    NonlinearUpdateProvenance,
    NonlinearUpdateResult,
    NonlinearUpdateStatus,
    PreparedNonlinearUpdate,
)
from ._work import NonlinearWork


def _coordinate_norm(value: Array, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


class AndersonAcceleration(StrictModule):
    """Fixed-capacity regularized Type-I or Type-II Anderson acceleration."""

    kind: Literal["type-i", "type-ii"] = eqx.field(static=True)
    history: int = eqx.field(static=True)
    regularization: float = eqx.field(static=True)
    safeguard_factor: float = eqx.field(static=True)
    restart_condition: float = eqx.field(static=True)
    linear: LinearSolvePolicy

    def __init__(
        self,
        *,
        kind: Literal["type-i", "type-ii"] = "type-ii",
        history: int = 5,
        regularization: float = 1e-10,
        safeguard_factor: float = 2.0,
        restart_condition: float = 1e12,
        linear: LinearSolvePolicy | None = None,
    ):
        history_ = int(history)
        regularization_ = float(regularization)
        safeguard_ = float(safeguard_factor)
        condition_ = float(restart_condition)
        if kind not in ("type-i", "type-ii"):
            raise ValueError("Anderson kind must be 'type-i' or 'type-ii'.")
        if history_ < 1:
            raise ValueError("Anderson history must be positive.")
        if not isfinite(regularization_) or regularization_ < 0.0:
            raise ValueError("Anderson regularization must be finite and non-negative.")
        if not isfinite(safeguard_) or safeguard_ < 1.0:
            raise ValueError("Anderson safeguard_factor must be finite and at least one.")
        if not isfinite(condition_) or condition_ <= 1.0:
            raise ValueError("Anderson restart_condition must be finite and exceed one.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        self.history = history_
        self.kind = kind
        self.regularization = regularization_
        self.safeguard_factor = safeguard_
        self.restart_condition = condition_
        self.linear = linear_


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
    precision: NonlinearPrecisionPolicy,
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
    delta_states_ = precision.accumulation(delta_states)
    delta_residuals_ = precision.accumulation(delta_residuals)
    residual_ = precision.accumulation(residual)
    if policy.kind == "type-ii":
        gram = delta_residuals_ @ jnp.conj(delta_residuals_.T)
        right = delta_residuals_ @ jnp.conj(residual_)
        correction_basis = delta_states_ + delta_residuals_
        sign = -1.0
    else:
        gram = delta_states_ @ jnp.conj(delta_residuals_.T)
        right = delta_states_ @ jnp.conj(residual_)
        correction_basis = delta_states_ - delta_residuals_
        sign = 1.0
    diagonal = jnp.where(
        active,
        jnp.asarray(policy.regularization, dtype=gram.real.dtype),
        jnp.asarray(1.0, dtype=gram.real.dtype),
    )
    gram = gram + jnp.diag(diagonal.astype(gram.dtype))
    linear_result = solve_linear(
        LeastSquaresProblem(DenseLinearOperator(gram)),
        right,
        policy=precision.bind_linear(policy.linear),
    )
    coefficients = jnp.where(active, linear_result.value, 0.0)
    correction = jnp.sum(
        coefficients[:, None] * correction_basis,
        axis=0,
    )
    candidate = jnp.asarray(mapped + sign * correction, dtype=mapped.dtype)
    condition = precision.decision(linear_result.diagnostics.condition_estimate)
    usable = (
        linear_result.diagnostics.converged
        & (active_count > 0)
        & jnp.all(jnp.isfinite(candidate))
        & jnp.isfinite(condition)
        & (condition <= policy.restart_condition)
    )
    return jnp.where(usable, candidate, mapped), usable, condition


class FixedPointIteration(StrictModule):
    """Damped fixed-point iteration with optional safeguarded Anderson acceleration."""

    damping: float = eqx.field(static=True)
    acceleration: AndersonAcceleration | None
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        damping: float = 1.0,
        acceleration: AndersonAcceleration | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        damping_ = float(damping)
        if not isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("Fixed-point damping must lie in (0, 1].")
        if acceleration is not None and not isinstance(
            acceleration, AndersonAcceleration
        ):
            raise TypeError("acceleration must be AndersonAcceleration or None.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.damping = damping_
        self.acceleration = acceleration
        self.precision = precision_

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
        self.precision.validate_tolerance(termination_.absolute_residual)
        initial = validate_inexact_tree(initial_state, name="initial fixed-point state")
        space = PyTreeSpace(initial)
        flat_initial = space.flatten(initial)
        mapped = problem.mapping(initial, args)
        flat_mapped = space.flatten(mapped)
        residual = flat_mapped - flat_initial
        self.precision.validate_trees(initial, residual)
        residual_norm = _coordinate_norm(residual, self.precision)
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
                    self.precision,
                )
            proposed_tree = space.unflatten(proposed)
            next_mapped = space.flatten(problem.mapping(proposed_tree, args))
            next_residual = next_mapped - proposed
            next_norm = _coordinate_norm(next_residual, self.precision)
            if self.acceleration is None:
                raw_residual = next_residual
                raw_norm = next_norm
            else:
                raw_mapped = space.flatten(problem.mapping(space.unflatten(raw), args))
                raw_residual = raw_mapped - raw
                raw_norm = _coordinate_norm(raw_residual, self.precision)
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
            step_norm = _coordinate_norm(
                accepted_state - current.state,
                self.precision,
            )
            converged = finite & (
                accepted_norm
                <= termination_.residual_threshold(current.initial_residual_norm)
            )
            stagnated = (
                finite
                & ~converged
                & (
                    step_norm
                    <= termination_.step_threshold(
                        _coordinate_norm(current.state, self.precision)
                    )
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
        output_state = jax.tree.map(self.precision.output, final_state)
        return NonlinearResult(
            state=output_state,
            residual=final_residual,
            auxiliary=None,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="none",
                globalization_id="fixed-point-safeguard",
                precision_policy_id=self.precision.policy_id,
            ),
            precision_evidence=self.precision.evidence_for(
                final_state,
                final_residual,
                output_value=output_state,
            ),
        )


class _SteffensenRun(StrictModule):
    state: Array
    residual: Array
    initial_norm: Array
    norm: Array
    step_norm: Array
    iteration: Array
    evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    restarts: Array
    nonfinite: Array
    status: Array


class SteffensenIteration(StrictModule):
    """Elementwise Steffensen/Aitken acceleration with residual safeguarding."""

    denominator_tolerance: float = eqx.field(static=True)
    safeguard_factor: float = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        denominator_tolerance: float = 1e-12,
        safeguard_factor: float = 1.0,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        tolerance = float(denominator_tolerance)
        safeguard = float(safeguard_factor)
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("denominator_tolerance must be finite and positive.")
        if not isfinite(safeguard) or safeguard < 1.0:
            raise ValueError("safeguard_factor must be finite and at least one.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.denominator_tolerance = tolerance
        self.safeguard_factor = safeguard
        self.precision = precision_

    @property
    def method_id(self) -> str:
        return "steffensen-aitken"

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
            raise TypeError("problem must be FixedPointProblem.")
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        self.precision.validate_tolerance(termination_.absolute_residual)
        initial = validate_inexact_tree(initial_state, name="initial fixed-point state")
        space = PyTreeSpace(initial)
        state = space.flatten(initial)
        first = space.flatten(problem.mapping(initial, args))
        residual = first - state
        self.precision.validate_trees(initial, residual)
        norm = _coordinate_norm(residual, self.precision)
        finite = jnp.all(jnp.isfinite(state)) & jnp.all(jnp.isfinite(residual))
        run = _SteffensenRun(
            state=state,
            residual=residual,
            initial_norm=jnp.maximum(norm, 1e-30),
            norm=norm,
            step_norm=jnp.asarray(0.0, dtype=norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            evaluations=jnp.asarray(1, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            restarts=jnp.asarray(0, dtype=jnp.int32),
            nonfinite=(~finite).astype(jnp.int32),
            status=jnp.where(
                finite & (norm <= termination_.residual_threshold(norm)),
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    finite,
                    int(NonlinearStatus.ITERATING),
                    int(NonlinearStatus.NONFINITE_INPUT),
                ),
            ).astype(jnp.int32),
        )

        def condition(current):
            within = (
                jnp.asarray(True)
                if termination_.maximum_evaluations is None
                else current.evaluations + 3 <= termination_.maximum_evaluations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination_.maximum_steps)
                & within
            )

        def body(current):
            current_tree = space.unflatten(current.state)
            first_coordinates = space.flatten(problem.mapping(current_tree, args))
            first_tree = space.unflatten(first_coordinates)
            second_coordinates = space.flatten(problem.mapping(first_tree, args))
            second_ = self.precision.accumulation(second_coordinates)
            first_ = self.precision.accumulation(first_coordinates)
            current_ = self.precision.accumulation(current.state)
            denominator = second_ - 2.0 * first_ + current_
            usable = jnp.abs(denominator) >= self.denominator_tolerance
            accelerated = jnp.asarray(
                current_
                - jnp.where(
                    usable,
                    (first_ - current_) ** 2 / denominator,
                    0.0,
                ),
                dtype=current.state.dtype,
            )
            accelerated_tree = space.unflatten(accelerated)
            mapped_accelerated = space.flatten(problem.mapping(accelerated_tree, args))
            accelerated_residual = mapped_accelerated - accelerated
            plain_residual = second_coordinates - first_coordinates
            accelerated_norm = _coordinate_norm(
                accelerated_residual,
                self.precision,
            )
            plain_norm = _coordinate_norm(plain_residual, self.precision)
            finite_accelerated = jnp.all(jnp.isfinite(accelerated)) & jnp.all(
                jnp.isfinite(accelerated_residual)
            )
            take_accelerated = (
                jnp.any(usable)
                & finite_accelerated
                & (accelerated_norm <= self.safeguard_factor * plain_norm)
            )
            candidate = jnp.where(
                take_accelerated,
                accelerated,
                first_coordinates,
            )
            candidate_residual = jnp.where(
                take_accelerated,
                accelerated_residual,
                plain_residual,
            )
            candidate_norm = _coordinate_norm(candidate_residual, self.precision)
            step_norm = _coordinate_norm(
                candidate - current.state,
                self.precision,
            )
            converged = candidate_norm <= termination_.residual_threshold(
                current.initial_norm
            )
            stagnated = ~converged & (
                step_norm
                <= termination_.step_threshold(
                    _coordinate_norm(current.state, self.precision)
                )
            )
            finite_candidate = jnp.all(jnp.isfinite(candidate)) & jnp.all(
                jnp.isfinite(candidate_residual)
            )
            status = jnp.where(
                ~finite_candidate,
                int(NonlinearStatus.NONFINITE_EVALUATION),
                jnp.where(
                    converged,
                    int(NonlinearStatus.SUCCESS),
                    jnp.where(
                        stagnated,
                        int(NonlinearStatus.RESIDUAL_STAGNATION),
                        int(NonlinearStatus.ITERATING),
                    ),
                ),
            ).astype(jnp.int32)
            return _SteffensenRun(
                state=candidate,
                residual=candidate_residual,
                initial_norm=current.initial_norm,
                norm=candidate_norm,
                step_norm=step_norm,
                iteration=current.iteration + 1,
                evaluations=current.evaluations + 3,
                accepted_steps=current.accepted_steps + 1,
                rejected_steps=current.rejected_steps
                + (~take_accelerated).astype(jnp.int32),
                restarts=current.restarts + (~take_accelerated).astype(jnp.int32),
                nonfinite=current.nonfinite + (~finite_accelerated).astype(jnp.int32),
                status=status,
            )

        run = jax.lax.while_loop(condition, body, run)
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            run.status,
        ).astype(jnp.int32)
        final_state = space.unflatten(run.state)
        final_mapped = problem.mapping(final_state, args)
        final_residual = jax.tree.map(
            lambda mapped, value: mapped - value,
            final_mapped,
            final_state,
        )
        output_state = jax.tree.map(self.precision.output, final_state)
        return NonlinearResult(
            state=output_state,
            residual=final_residual,
            auxiliary=final_mapped,
            status=status,
            diagnostics=NonlinearDiagnostics(
                initial_residual_norm=run.initial_norm,
                final_residual_norm=run.norm,
                final_step_norm=run.step_norm,
                iterations=run.iteration,
                residual_evaluations=run.evaluations + 1,
                accepted_steps=run.accepted_steps,
                rejected_steps=run.rejected_steps,
                nonfinite_trials=run.nonfinite,
                acceleration_restarts=run.restarts,
            ),
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="none",
                globalization_id="residual-safeguard",
                precision_policy_id=self.precision.policy_id,
            ),
            precision_evidence=self.precision.evidence_for(
                final_state,
                final_residual,
                output_value=output_state,
            ),
        )


def _picard_candidate(
    inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner,
    damping: float,
    state: PyTree[Any],
    residual: PyTree[Any],
    /,
) -> PyTree[Array]:
    correction = (
        inverse_action.apply(residual)
        if isinstance(inverse_action, AbstractPreconditioner)
        else inverse_action(residual)
    )
    return tree_add_scaled(state, correction, -damping)


class PicardUpdate(AbstractNonlinearUpdate):
    """One preconditioned Picard correction as a finite nonlinear update."""

    inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner
    damping: float = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner,
        /,
        *,
        damping: float = 1.0,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if not callable(inverse_action) and not isinstance(
            inverse_action, AbstractPreconditioner
        ):
            raise TypeError("inverse_action must be callable or AbstractPreconditioner.")
        damping_ = float(damping)
        if not isfinite(damping_) or not 0.0 < damping_ <= 1.0:
            raise ValueError("Picard damping must lie in (0, 1].")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.inverse_action = inverse_action
        self.damping = damping_
        self.precision = precision_

    @property
    def update_id(self) -> str:
        return "picard-update"

    @property
    def capabilities(self) -> NonlinearUpdateCapabilities:
        return NonlinearUpdateCapabilities(
            jit=True,
            prepared_refresh=True,
            differentiable_action=True,
        )

    @property
    def maximum_work(self) -> NonlinearWork:
        return NonlinearWork(
            residual_evaluations=2,
            validity_evaluations=1,
            preconditioner_applications=1,
        )

    def _prepare_internal(self, problem, state, args, /):
        del problem, state, args
        return None

    def _refresh_internal(self, internal_state, problem, state, args, /):
        del problem, state, args
        return internal_state

    def _apply(
        self,
        prepared: PreparedNonlinearUpdate,
        state: PyTree[Any],
        args: Any,
        control: NonlinearUpdateControl,
        /,
    ):
        problem = prepared.problem
        state_ = prepared.plan.state_space.validate(state)

        def skipped(_):
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=jnp.asarray(jnp.nan),
                final_residual_norm=jnp.asarray(jnp.nan),
                step_norm=0.0,
                work=NonlinearWork.zero(),
            )
            return (
                NonlinearUpdateResult(
                    state=state_,
                    residual=prepared.plan.residual_space.zeros(),
                    auxiliary=prepared.reference_auxiliary,
                    status=NonlinearUpdateStatus.BUDGET_EXHAUSTED,
                    diagnostics=diagnostics,
                    provenance=NonlinearUpdateProvenance(
                        problem_id=problem.problem_id,
                        update_id=self.update_id,
                        plan_id=prepared.plan.plan_id,
                        notes=f"precision-policy={self.precision.policy_id}",
                    ),
                ),
                prepared.internal_state,
            )

        def execute(_):
            residual, _ = problem.evaluate(state_, args)
            self.precision.validate_trees(state_, residual)
            initial_norm = self.precision.norm(
                prepared.plan.residual_space,
                residual,
            )
            candidate = prepared.plan.state_space.validate(
                _picard_candidate(
                    self.inverse_action,
                    self.damping,
                    state_,
                    residual,
                )
            )
            candidate_residual, candidate_auxiliary = problem.evaluate(
                candidate,
                args,
            )
            final_norm = self.precision.norm(
                prepared.plan.residual_space,
                candidate_residual,
            )
            finite = tree_allfinite(candidate) & tree_allfinite(candidate_residual)
            valid = problem.valid(
                candidate,
                candidate_residual,
                candidate_auxiliary,
                args,
            )
            status = jnp.where(
                ~finite,
                int(NonlinearUpdateStatus.NONFINITE_EVALUATION),
                jnp.where(
                    ~valid,
                    int(NonlinearUpdateStatus.DOMAIN_REJECTED),
                    int(NonlinearUpdateStatus.APPLIED),
                ),
            ).astype(jnp.int32)
            step = jax.tree.map(
                lambda new, old: new - old,
                candidate,
                state_,
            )
            diagnostics = NonlinearUpdateDiagnostics(
                initial_residual_norm=initial_norm,
                final_residual_norm=final_norm,
                step_norm=self.precision.norm(
                    prepared.plan.state_space,
                    step,
                ),
                work=self.maximum_work,
                accepted_steps=(status == int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                rejected_steps=(status != int(NonlinearUpdateStatus.APPLIED)).astype(
                    jnp.int32
                ),
                domain_failures=(finite & ~valid).astype(jnp.int32),
                nonfinite_trials=(~finite).astype(jnp.int32),
            )
            return (
                NonlinearUpdateResult(
                    state=candidate,
                    residual=candidate_residual,
                    auxiliary=candidate_auxiliary,
                    status=status,
                    diagnostics=diagnostics,
                    provenance=NonlinearUpdateProvenance(
                        problem_id=problem.problem_id,
                        update_id=self.update_id,
                        plan_id=prepared.plan.plan_id,
                        notes=f"precision-policy={self.precision.policy_id}",
                    ),
                ),
                prepared.internal_state,
            )

        return jax.lax.cond(
            control.permits(self.maximum_work),
            execute,
            skipped,
            operand=None,
        )


class PicardIteration(StrictModule):
    """Preconditioned Picard iteration for a physical nonlinear residual."""

    inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner
    damping: float = eqx.field(static=True)
    acceleration: AndersonAcceleration | None
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        inverse_action: Callable[[PyTree[Any]], PyTree[Any]] | AbstractPreconditioner,
        /,
        *,
        damping: float = 1.0,
        acceleration: AndersonAcceleration | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
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
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.inverse_action = inverse_action
        self.damping = float(damping)
        self.acceleration = acceleration
        self.precision = precision_

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
        termination_ = NonlinearTermination() if termination is None else termination
        if not isinstance(termination_, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination or None.")
        self.precision.validate_tolerance(termination_.absolute_residual)

        def mapping(state, current_args):
            residual = problem.residual(state, current_args)
            return _picard_candidate(
                self.inverse_action,
                self.damping,
                state,
                residual,
            )

        fixed_problem = FixedPointProblem(
            mapping,
            problem_id=f"{problem.problem_id}/picard-map",
        )
        result = FixedPointIteration(
            damping=1.0,
            acceleration=self.acceleration,
            precision=self.precision,
        ).solve(
            fixed_problem,
            initial_state,
            termination=termination_,
            args=args,
        )
        model_state = self.precision.state(result.state)
        physical_residual, auxiliary = problem.evaluate(model_state, args)
        self.precision.validate_trees(model_state, physical_residual)
        physical_norm = self.precision.norm(
            PyTreeSpace(physical_residual),
            physical_residual,
        )
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
        output_state = jax.tree.map(self.precision.output, model_state)
        children = (
            {}
            if result.precision_evidence is None
            else {"fixed-point": result.precision_evidence}
        )
        return NonlinearResult(
            state=output_state,
            residual=physical_residual,
            auxiliary=auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem.problem_id,
                method_id=self.method_id,
                derivative_id="preconditioned-residual-map",
                globalization_id="fixed-point-safeguard",
                precision_policy_id=self.precision.policy_id,
            ),
            precision_evidence=self.precision.evidence_for(
                model_state,
                physical_residual,
                children=children,
                output_value=output_state,
            ),
            attempts=result.attempts,
        )


__all__ = [
    "AndersonAcceleration",
    "FixedPointIteration",
    "PicardIteration",
    "PicardUpdate",
    "SteffensenIteration",
]
