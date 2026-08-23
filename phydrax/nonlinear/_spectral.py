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


class _SpectralSearch(eqx.Module):
    state: Array
    residual: Array
    norm: Array
    rate: Array
    evaluations: Array
    accepted: Array
    finite_seen: Array
    domain_failures: Array
    nonfinite: Array


class _SpectralRun(eqx.Module):
    state: Array
    residual: Array
    initial_norm: Array
    norm: Array
    step_norm: Array
    sigma: Array
    merit_history: Array
    history_count: Array
    history_cursor: Array
    iteration: Array
    evaluations: Array
    accepted_steps: Array
    rejected_steps: Array
    restarts: Array
    domain_failures: Array
    nonfinite: Array
    status: Array


class DFSANE(AbstractNonlinearMethod):
    """Derivative-free spectral residual method with nonmonotone globalization."""

    history: int = eqx.field(static=True)
    minimum_spectral: float = eqx.field(static=True)
    maximum_spectral: float = eqx.field(static=True)
    contraction: float = eqx.field(static=True)
    sufficient_decrease: float = eqx.field(static=True)
    maximum_search_steps: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        history: int = 10,
        minimum_spectral: float = 1e-10,
        maximum_spectral: float = 1e10,
        contraction: float = 0.5,
        sufficient_decrease: float = 1e-4,
        maximum_search_steps: int = 24,
    ):
        history_ = int(history)
        search_steps = int(maximum_search_steps)
        values = tuple(
            float(value)
            for value in (
                minimum_spectral,
                maximum_spectral,
                contraction,
                sufficient_decrease,
            )
        )
        if history_ < 1 or search_steps < 1:
            raise ValueError("history and maximum_search_steps must be positive.")
        if any(not isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("DF-SANE controls must be finite and positive.")
        if not values[0] < values[1] or not values[2] < 1.0 or not values[3] < 1.0:
            raise ValueError("DF-SANE spectral and line-search controls are invalid.")
        self.history = history_
        (
            self.minimum_spectral,
            self.maximum_spectral,
            self.contraction,
            self.sufficient_decrease,
        ) = values
        self.maximum_search_steps = search_steps

    @property
    def method_id(self) -> str:
        return "df-sane"

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
            raise ValueError("DF-SANE requires square state/residual coordinates.")
        state = source.flatten(state_tree)
        residual = target.flatten(residual_tree)
        norm = jnp.linalg.norm(residual)
        finite = tree_allfinite(state_tree) & tree_allfinite(residual_tree)
        valid = problem_.valid(state_tree, residual_tree, auxiliary, args)
        converged = finite & valid & (norm <= termination.residual_threshold(norm))
        merit = 0.5 * norm * norm
        run = _SpectralRun(
            state=state,
            residual=residual,
            initial_norm=jnp.maximum(norm, 1e-30),
            norm=norm,
            step_norm=jnp.asarray(0.0, dtype=norm.dtype),
            sigma=jnp.asarray(1.0, dtype=norm.dtype),
            merit_history=jnp.full((self.history,), -jnp.inf, dtype=norm.dtype)
            .at[0]
            .set(merit),
            history_count=jnp.asarray(1, dtype=jnp.int32),
            history_cursor=jnp.asarray(1 % self.history, dtype=jnp.int32),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            evaluations=jnp.asarray(initial_evaluations, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            restarts=jnp.asarray(0, dtype=jnp.int32),
            domain_failures=(finite & ~valid).astype(jnp.int32),
            nonfinite=(~finite).astype(jnp.int32),
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
                else current.evaluations < termination.maximum_evaluations
            )
            return (
                (current.status == int(NonlinearStatus.ITERATING))
                & (current.iteration < termination.maximum_steps)
                & within
            )

        def body(current):
            direction = -current.sigma * current.residual
            reference = jnp.max(current.merit_history)
            search = _SpectralSearch(
                state=current.state,
                residual=current.residual,
                norm=current.norm,
                rate=jnp.asarray(1.0, dtype=current.norm.dtype),
                evaluations=jnp.asarray(0, dtype=jnp.int32),
                accepted=jnp.asarray(False),
                finite_seen=jnp.asarray(False),
                domain_failures=jnp.asarray(0, dtype=jnp.int32),
                nonfinite=jnp.asarray(0, dtype=jnp.int32),
            )

            def search_condition(item):
                within = (
                    jnp.asarray(True)
                    if termination.maximum_evaluations is None
                    else current.evaluations + item.evaluations
                    < termination.maximum_evaluations
                )
                return (
                    ~item.accepted
                    & (item.evaluations < self.maximum_search_steps)
                    & (item.rate >= 1e-12)
                    & within
                )

            def search_body(item):
                candidate_coordinates = current.state + item.rate * direction
                candidate = source.unflatten(candidate_coordinates)
                candidate_residual_tree, candidate_auxiliary = problem_.evaluate(
                    candidate, args
                )
                candidate_residual = target.flatten(candidate_residual_tree)
                candidate_norm = jnp.linalg.norm(candidate_residual)
                candidate_finite = jnp.all(jnp.isfinite(candidate_coordinates)) & jnp.all(
                    jnp.isfinite(candidate_residual)
                )
                candidate_valid = problem_.valid(
                    candidate, candidate_residual_tree, candidate_auxiliary, args
                )
                accepted = (
                    candidate_finite
                    & candidate_valid
                    & (
                        0.5 * candidate_norm * candidate_norm
                        <= reference
                        - self.sufficient_decrease
                        * item.rate
                        * item.rate
                        * current.norm
                        * current.norm
                    )
                )
                return _SpectralSearch(
                    state=jnp.where(accepted, candidate_coordinates, item.state),
                    residual=jnp.where(accepted, candidate_residual, item.residual),
                    norm=jnp.where(accepted, candidate_norm, item.norm),
                    rate=jnp.where(accepted, item.rate, self.contraction * item.rate),
                    evaluations=item.evaluations + 1,
                    accepted=accepted,
                    finite_seen=item.finite_seen | (candidate_finite & candidate_valid),
                    domain_failures=item.domain_failures
                    + (candidate_finite & ~candidate_valid).astype(jnp.int32),
                    nonfinite=item.nonfinite + (~candidate_finite).astype(jnp.int32),
                )

            search = jax.lax.while_loop(search_condition, search_body, search)
            step = search.state - current.state
            residual_delta = search.residual - current.residual
            denominator = jnp.real(jnp.vdot(step, residual_delta))
            spectral = jnp.real(jnp.vdot(step, step)) / jnp.where(
                jnp.abs(denominator) < 1e-30, 1.0, denominator
            )
            spectral_usable = search.accepted & jnp.isfinite(spectral) & (spectral > 0.0)
            sigma = jnp.where(
                spectral_usable,
                jnp.clip(spectral, self.minimum_spectral, self.maximum_spectral),
                1.0,
            )
            history = jax.lax.cond(
                search.accepted,
                lambda values: values.at[current.history_cursor].set(
                    0.5 * search.norm * search.norm
                ),
                lambda values: values,
                current.merit_history,
            )
            next_cursor = jnp.where(
                search.accepted,
                (current.history_cursor + 1) % self.history,
                current.history_cursor,
            )
            next_count = jnp.where(
                search.accepted,
                jnp.minimum(current.history_count + 1, self.history),
                current.history_count,
            )
            converged = search.accepted & (
                search.norm <= termination.residual_threshold(current.initial_norm)
            )
            step_norm = jnp.linalg.norm(step)
            stagnated = (
                search.accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(jnp.linalg.norm(current.state))
                )
            )
            next_evaluations = current.evaluations + search.evaluations
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
            return _SpectralRun(
                state=jnp.where(search.accepted, search.state, current.state),
                residual=jnp.where(search.accepted, search.residual, current.residual),
                initial_norm=current.initial_norm,
                norm=jnp.where(search.accepted, search.norm, current.norm),
                step_norm=step_norm,
                sigma=sigma,
                merit_history=history,
                history_count=next_count,
                history_cursor=next_cursor,
                iteration=current.iteration + 1,
                evaluations=next_evaluations,
                accepted_steps=current.accepted_steps + search.accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps
                + (~search.accepted).astype(jnp.int32),
                restarts=current.restarts
                + (search.accepted & ~spectral_usable).astype(jnp.int32),
                domain_failures=current.domain_failures + search.domain_failures,
                nonfinite=current.nonfinite + search.nonfinite,
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
            initial_residual_norm=run.initial_norm,
            final_residual_norm=jnp.linalg.norm(target.flatten(final_residual)),
            final_step_norm=run.step_norm,
            iterations=run.iteration,
            residual_evaluations=run.evaluations + 1,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
            domain_failures=run.domain_failures,
            nonfinite_trials=run.nonfinite,
            acceleration_restarts=run.restarts,
        )
        return NonlinearResult(
            state=final_state,
            residual=final_residual,
            auxiliary=final_auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem_.problem_id,
                method_id=self.method_id,
                derivative_id="none",
                globalization_id="nonmonotone-spectral-residual",
            ),
        )


__all__ = ["DFSANE"]
