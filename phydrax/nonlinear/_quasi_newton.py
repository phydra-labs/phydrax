#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from math import isfinite
from typing import Literal, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from .._strict import StrictModule
from .._tree_math import tree_allfinite
from ..linalg import (
    DenseLinearOperator,
    DenseLU,
    LinearSolvePolicy,
    LinearSystem,
    prepare as prepare_linear,
    PyTreeSpace,
    solve as solve_linear,
)
from ._precision import NonlinearPrecisionPolicy
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


BroydenKind: TypeAlias = Literal["good", "bad"]


def _coordinate_norm(value: Array, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(jnp.linalg.norm(precision.accumulation(value)))


def _coordinate_inner(
    left: Array,
    right: Array,
    precision: NonlinearPrecisionPolicy,
    /,
) -> Array:
    left_ = precision.accumulation(left)
    right_ = precision.accumulation(right)
    return precision.decision(jnp.real(jnp.sum(jnp.conj(left_) * right_)))


class _QuasiRun(StrictModule):
    state: Array
    residual: Array
    initial_residual_norm: Array
    residual_norm: Array
    step_norm: Array
    inverse_updates: Array
    left_factors: Array
    right_factors: Array
    history_count: Array
    iteration: Array
    residual_evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    restarts: Array
    domain_failures: Array
    nonfinite_trials: Array
    status: Array


def _apply_inverse(scale, left, right, count, vector, precision, /):
    active = jnp.arange(left.shape[0]) < count
    left_ = precision.accumulation(left)
    right_ = precision.accumulation(right)
    vector_ = precision.accumulation(vector)
    coefficients = right_ @ vector_
    result = scale * vector_ + jnp.sum(
        jnp.where(active[:, None], left_ * coefficients[:, None], 0.0),
        axis=0,
    )
    return precision.direction(result)


def _apply_inverse_transpose(scale, left, right, count, vector, precision, /):
    active = jnp.arange(left.shape[0]) < count
    left_ = precision.accumulation(left)
    right_ = precision.accumulation(right)
    vector_ = precision.accumulation(vector)
    coefficients = left_ @ vector_
    result = scale * vector_ + jnp.sum(
        jnp.where(active[:, None], right_ * coefficients[:, None], 0.0),
        axis=0,
    )
    return precision.direction(result)


def _line_search(
    problem,
    source,
    target,
    state,
    residual,
    direction,
    args,
    termination,
    maximum_steps,
    precision,
    /,
):
    class _Search(StrictModule):
        state: Array
        residual: Array
        norm: Array
        rate: Array
        evaluations: Array
        accepted: Array
        finite_seen: Array
        domain_failures: Array
        nonfinite: Array

    search = _Search(
        state=state,
        residual=residual,
        norm=_coordinate_norm(residual, precision),
        rate=precision.decision(1.0),
        evaluations=jnp.asarray(0, dtype=jnp.int32),
        accepted=jnp.asarray(False),
        finite_seen=jnp.asarray(False),
        domain_failures=jnp.asarray(0, dtype=jnp.int32),
        nonfinite=jnp.asarray(0, dtype=jnp.int32),
    )

    def condition(item):
        within = (
            jnp.asarray(True)
            if termination.maximum_evaluations is None
            else item.evaluations < termination.maximum_evaluations
        )
        return (
            ~item.accepted
            & (item.evaluations < maximum_steps)
            & (item.rate >= 1e-10)
            & within
        )

    def body(item):
        candidate_coordinates = jnp.asarray(
            state + item.rate * direction,
            dtype=state.dtype,
        )
        candidate = source.unflatten(candidate_coordinates)
        candidate_residual_tree, auxiliary = problem.evaluate(candidate, args)
        candidate_residual = target.flatten(candidate_residual_tree)
        norm = _coordinate_norm(candidate_residual, precision)
        finite = jnp.all(jnp.isfinite(candidate_coordinates)) & jnp.all(
            jnp.isfinite(candidate_residual)
        )
        valid = problem.valid(candidate, candidate_residual_tree, auxiliary, args)
        accepted = (
            finite & valid & (norm * norm <= (1.0 - 1e-4 * item.rate) * item.norm**2)
        )
        return _Search(
            state=jnp.where(accepted, candidate_coordinates, item.state),
            residual=jnp.where(accepted, candidate_residual, item.residual),
            norm=jnp.where(accepted, norm, item.norm),
            rate=jnp.where(accepted, item.rate, 0.5 * item.rate),
            evaluations=item.evaluations + 1,
            accepted=accepted,
            finite_seen=item.finite_seen | (finite & valid),
            domain_failures=item.domain_failures + (finite & ~valid).astype(jnp.int32),
            nonfinite=item.nonfinite + (~finite).astype(jnp.int32),
        )

    return jax.lax.while_loop(condition, body, search)


class Broyden(AbstractNonlinearMethod):
    """Good or bad limited-memory inverse-Broyden root method."""

    kind: BroydenKind = eqx.field(static=True)
    memory: int = eqx.field(static=True)
    initial_scale: float = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    denominator_tolerance: float = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        kind: BroydenKind = "good",
        /,
        *,
        memory: int = 12,
        initial_scale: float = 1.0,
        maximum_line_search_steps: int = 20,
        denominator_tolerance: float = 1e-12,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if kind not in ("good", "bad"):
            raise ValueError("kind must be 'good' or 'bad'.")
        memory_ = int(memory)
        steps = int(maximum_line_search_steps)
        scale = float(initial_scale)
        tolerance = float(denominator_tolerance)
        if memory_ < 1 or steps < 1:
            raise ValueError("memory and maximum_line_search_steps must be positive.")
        if not isfinite(scale) or scale <= 0.0:
            raise ValueError("initial_scale must be finite and positive.")
        if not isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("denominator_tolerance must be finite and positive.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.kind = kind
        self.memory = memory_
        self.initial_scale = scale
        self.maximum_line_search_steps = steps
        self.denominator_tolerance = tolerance
        self.precision = precision_

    @property
    def method_id(self) -> str:
        return f"{self.kind}-broyden/lm{self.memory}"

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
        problem,
        initial_state,
        /,
        *,
        termination,
        args=None,
        _initial_evaluation=None,
    ) -> NonlinearResult:
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        self.precision.validate_tolerance(termination.absolute_residual)
        if _initial_evaluation is None:
            state_tree = problem.validate_state(initial_state)
            residual_tree, auxiliary = problem.evaluate(state_tree, args)
            problem_ = problem.bind_spaces(state_tree, residual_tree)
            initial_evaluations = 1
        else:
            problem_, state_tree, residual_tree, auxiliary = _initial_evaluation
            state_tree = problem_.validate_state(state_tree)
            residual_tree = problem_.validate_residual(residual_tree)
            initial_evaluations = 0
        source = PyTreeSpace(state_tree)
        target = PyTreeSpace(residual_tree)
        if source.size != target.size:
            raise ValueError("Broyden methods require square state/residual coordinates.")
        state = source.flatten(state_tree)
        residual = target.flatten(residual_tree)
        self.precision.validate_trees(state_tree, residual_tree)
        norm = _coordinate_norm(residual, self.precision)
        finite = tree_allfinite(state_tree) & tree_allfinite(residual_tree)
        valid = problem_.valid(state_tree, residual_tree, auxiliary, args)
        converged = finite & valid & (norm <= termination.residual_threshold(norm))
        dtype = state.dtype
        run = _QuasiRun(
            state=state,
            residual=residual,
            initial_residual_norm=jnp.maximum(norm, 1e-30),
            residual_norm=norm,
            step_norm=self.precision.decision(0.0),
            inverse_updates=jnp.asarray(0, dtype=jnp.int32),
            left_factors=jnp.zeros((self.memory, source.size), dtype=dtype),
            right_factors=jnp.zeros((self.memory, source.size), dtype=dtype),
            history_count=jnp.asarray(0, dtype=jnp.int32),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            residual_evaluations=jnp.asarray(
                initial_evaluations,
                dtype=jnp.int32,
            ),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            restarts=jnp.asarray(0, dtype=jnp.int32),
            domain_failures=(finite & ~valid).astype(jnp.int32),
            nonfinite_trials=(~finite).astype(jnp.int32),
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
            within = (
                jnp.asarray(True)
                if termination.maximum_evaluations is None
                else current.residual_evaluations < termination.maximum_evaluations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination.maximum_steps)
                & within
            )

        def body(current):
            direction = -_apply_inverse(
                self.initial_scale,
                current.left_factors,
                current.right_factors,
                current.history_count,
                current.residual,
                self.precision,
            )
            search = _line_search(
                problem_,
                source,
                target,
                current.state,
                current.residual,
                direction,
                args,
                termination,
                self.maximum_line_search_steps,
                self.precision,
            )
            step = search.state - current.state
            residual_delta = search.residual - current.residual
            image = _apply_inverse(
                self.initial_scale,
                current.left_factors,
                current.right_factors,
                current.history_count,
                residual_delta,
                self.precision,
            )
            left = step - image
            if self.kind == "good":
                transpose_step = _apply_inverse_transpose(
                    self.initial_scale,
                    current.left_factors,
                    current.right_factors,
                    current.history_count,
                    step,
                    self.precision,
                )
                denominator = _coordinate_inner(step, image, self.precision)
                right = transpose_step / jnp.where(
                    jnp.abs(denominator) < self.denominator_tolerance,
                    1.0,
                    denominator,
                )
            else:
                denominator = _coordinate_inner(
                    residual_delta,
                    residual_delta,
                    self.precision,
                )
                right = residual_delta / jnp.where(
                    jnp.abs(denominator) < self.denominator_tolerance,
                    1.0,
                    denominator,
                )
            update_usable = (
                search.accepted
                & jnp.isfinite(denominator)
                & (jnp.abs(denominator) >= self.denominator_tolerance)
                & jnp.all(jnp.isfinite(left))
                & jnp.all(jnp.isfinite(right))
            )
            full = current.history_count >= self.memory
            restart = search.accepted & (full | ~update_usable)
            cleared_left = jnp.where(
                restart, jnp.zeros_like(current.left_factors), current.left_factors
            )
            cleared_right = jnp.where(
                restart, jnp.zeros_like(current.right_factors), current.right_factors
            )
            cleared_count = jnp.where(restart, 0, current.history_count)
            slot = jnp.minimum(cleared_count, self.memory - 1)
            next_left = jax.lax.cond(
                update_usable,
                lambda value: value.at[slot].set(left),
                lambda value: value,
                cleared_left,
            )
            next_right = jax.lax.cond(
                update_usable,
                lambda value: value.at[slot].set(right),
                lambda value: value,
                cleared_right,
            )
            next_count = jnp.where(
                update_usable,
                jnp.minimum(cleared_count + 1, self.memory),
                cleared_count,
            )
            step_norm = _coordinate_norm(step, self.precision)
            converged = search.accepted & (
                search.norm
                <= termination.residual_threshold(current.initial_residual_norm)
            )
            stagnated = (
                search.accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(
                        _coordinate_norm(current.state, self.precision)
                    )
                )
            )
            next_evaluations = current.residual_evaluations + search.evaluations
            exhausted = (
                jnp.asarray(False)
                if termination.maximum_evaluations is None
                else next_evaluations >= termination.maximum_evaluations
            )
            status = jnp.where(
                converged,
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    stagnated,
                    int(NonlinearStatus.RESIDUAL_STAGNATION),
                    jnp.where(
                        exhausted,
                        int(NonlinearStatus.MAXIMUM_EVALUATIONS_REACHED),
                        jnp.where(
                            search.accepted,
                            int(NonlinearStatus.ITERATING),
                            jnp.where(
                                search.finite_seen,
                                int(NonlinearStatus.LINE_SEARCH_FAILED),
                                int(NonlinearStatus.NONFINITE_EVALUATION),
                            ),
                        ),
                    ),
                ),
            ).astype(jnp.int32)
            return _QuasiRun(
                state=jnp.where(search.accepted, search.state, current.state),
                residual=jnp.where(search.accepted, search.residual, current.residual),
                initial_residual_norm=current.initial_residual_norm,
                residual_norm=jnp.where(
                    search.accepted, search.norm, current.residual_norm
                ),
                step_norm=step_norm,
                inverse_updates=current.inverse_updates + update_usable.astype(jnp.int32),
                left_factors=next_left,
                right_factors=next_right,
                history_count=next_count,
                iteration=current.iteration + 1,
                residual_evaluations=next_evaluations,
                accepted_steps=current.accepted_steps + search.accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps
                + (~search.accepted).astype(jnp.int32),
                restarts=current.restarts + restart.astype(jnp.int32),
                domain_failures=current.domain_failures + search.domain_failures,
                nonfinite_trials=current.nonfinite_trials + search.nonfinite,
                status=status,
            )

        run = jax.lax.while_loop(condition, body, run)
        status = jnp.where(
            run.status == int(NonlinearStatus.ITERATING),
            int(NonlinearStatus.MAXIMUM_STEPS_REACHED),
            run.status,
        ).astype(jnp.int32)
        final_state = source.unflatten(run.state)
        final_residual, final_auxiliary = problem_.evaluate(final_state, args)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=run.initial_residual_norm,
            final_residual_norm=_coordinate_norm(
                target.flatten(final_residual),
                self.precision,
            ),
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.residual_evaluations + 1,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            domain_failures=run.domain_failures,
            nonfinite_trials=run.nonfinite_trials,
            acceleration_restarts=run.restarts,
        )
        output_state = jax.tree.map(self.precision.output, final_state)
        return NonlinearResult(
            state=output_state,
            residual=final_residual,
            auxiliary=final_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem_.problem_id,
                method_id=self.method_id,
                derivative_id="secant-updates",
                globalization_id="residual-armijo",
                precision_policy_id=self.precision.policy_id,
                notes="limited-memory inverse secant updates",
            ),
            precision_evidence=self.precision.evidence_for(
                final_state,
                final_residual,
                output_value=output_state,
            ),
        )


class Chord(AbstractNonlinearMethod):
    """Frozen dense Jacobian with repeated residual-globalized solves."""

    linear: LinearSolvePolicy
    maximum_dimension: int = eqx.field(static=True)
    maximum_line_search_steps: int = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        linear: LinearSolvePolicy | None = None,
        maximum_dimension: int = 512,
        maximum_line_search_steps: int = 20,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        linear_ = LinearSolvePolicy(DenseLU()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        dimension = int(maximum_dimension)
        steps = int(maximum_line_search_steps)
        if dimension < 1 or steps < 1:
            raise ValueError("Chord dimensions and search steps must be positive.")
        self.linear = linear_
        self.maximum_dimension = dimension
        self.maximum_line_search_steps = steps
        self.precision = precision_

    @property
    def method_id(self) -> str:
        return "chord/frozen-jacobian"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=False,
            prepared_refresh=True,
            jit=False,
            implicit_differentiation=True,
        )

    def solve(self, problem, initial_state, /, *, termination, args=None):
        if not isinstance(problem, NonlinearSystemProblem):
            raise TypeError("problem must be NonlinearSystemProblem.")
        if not isinstance(termination, NonlinearTermination):
            raise TypeError("termination must be NonlinearTermination.")
        self.precision.validate_tolerance(termination.absolute_residual)
        state_tree = problem.validate_state(initial_state)
        residual_tree, auxiliary = problem.evaluate(state_tree, args)
        self.precision.validate_trees(state_tree, residual_tree)
        problem_ = problem.bind_spaces(state_tree, residual_tree)
        source = PyTreeSpace(state_tree)
        target = PyTreeSpace(residual_tree)
        if source.size != target.size or source.size > self.maximum_dimension:
            raise ValueError("Chord requires a square system within maximum_dimension.")
        initial_coordinates = source.flatten(state_tree)

        def coordinate_residual(coordinates):
            return target.flatten(problem_.residual(source.unflatten(coordinates), args))

        matrix = jax.jacfwd(coordinate_residual)(initial_coordinates)
        prepared = prepare_linear(
            LinearSystem(DenseLinearOperator(matrix)),
            self.precision.bind_linear(self.linear),
        )
        state = initial_coordinates
        residual = target.flatten(residual_tree)
        initial_norm = jnp.maximum(
            _coordinate_norm(residual, self.precision),
            self.precision.decision(1e-30),
        )
        iterations = 0
        evaluations = 1
        accepted = 0
        rejected = 0
        linear_iterations = 0
        step_norm = self.precision.decision(0.0)
        status = int(NonlinearStatus.ITERATING)
        while (
            status == int(NonlinearStatus.ITERATING)
            and iterations < termination.maximum_steps
        ):
            linear_result = solve_linear(prepared, -residual)
            direction = self.precision.direction(linear_result.value)
            search = _line_search(
                problem_,
                source,
                target,
                state,
                residual,
                direction,
                args,
                termination,
                self.maximum_line_search_steps,
                self.precision,
            )
            step_norm = _coordinate_norm(search.state - state, self.precision)
            state = search.state
            residual = search.residual
            evaluations += int(search.evaluations)
            accepted += int(search.accepted)
            rejected += int(~search.accepted)
            linear_iterations += int(
                jnp.sum(linear_result.diagnostics.iterations, dtype=jnp.int32)
            )
            iterations += 1
            if bool(search.accepted) and float(search.norm) <= float(
                termination.residual_threshold(initial_norm)
            ):
                status = int(NonlinearStatus.SUCCESS)
            elif not bool(search.accepted):
                status = int(NonlinearStatus.LINE_SEARCH_FAILED)
            elif float(step_norm) <= float(
                termination.step_threshold(_coordinate_norm(state, self.precision))
            ):
                status = int(NonlinearStatus.RESIDUAL_STAGNATION)
        if status == int(NonlinearStatus.ITERATING):
            status = int(NonlinearStatus.MAXIMUM_STEPS_REACHED)
        final_state = source.unflatten(state)
        final_residual, final_auxiliary = problem_.evaluate(final_state, args)
        diagnostics = NonlinearDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=_coordinate_norm(
                target.flatten(final_residual),
                self.precision,
            ),
            final_step_norm=step_norm,
            iterations=iterations,
            residual_evaluations=evaluations + 1,
            jacobian_preparations=1,
            linear_solves=iterations,
            linear_iterations=linear_iterations,
            accepted_steps=accepted,
            rejected_steps=rejected,
        )
        output_state = jax.tree.map(self.precision.output, final_state)
        return NonlinearResult(
            state=output_state,
            residual=final_residual,
            auxiliary=final_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem_.problem_id,
                method_id=self.method_id,
                derivative_id="autodiff-frozen",
                globalization_id="residual-armijo",
                precision_policy_id=self.precision.policy_id,
            ),
            precision_evidence=self.precision.evidence_for(
                final_state,
                final_residual,
                output_value=output_state,
            ),
        )


__all__ = ["Broyden", "BroydenKind", "Chord"]
