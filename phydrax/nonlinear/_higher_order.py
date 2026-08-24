#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._tree_math import tree_allfinite
from ..linalg import (
    FunctionLinearOperator,
    LinearSolvePolicy,
    LinearSystem,
    OperatorProperties,
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


def _coordinate_norm(space, value, precision: NonlinearPrecisionPolicy, /) -> Array:
    return precision.decision(
        jnp.linalg.norm(precision.accumulation(space.flatten(value)))
    )


class _HalleyRun(eqx.Module):
    state: PyTree[Array]
    residual: PyTree[Array]
    auxiliary: Any
    initial_norm: Array
    norm: Array
    step_norm: Array
    iteration: Array
    residual_evaluations: Array
    jvp_evaluations: Array
    linear_solves: Array
    linear_iterations: Array
    accepted_steps: Array
    rejected_steps: Array
    status: Array


class VectorHalley(AbstractNonlinearMethod):
    """Matrix-free vector Halley method using directional second derivatives."""

    linear: LinearSolvePolicy
    maximum_dimension: int = eqx.field(static=True)
    maximum_search_steps: int = eqx.field(static=True)
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        *,
        linear: LinearSolvePolicy | None = None,
        maximum_dimension: int = 128,
        maximum_search_steps: int = 16,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        linear_ = LinearSolvePolicy() if linear is None else linear
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        dimension = int(maximum_dimension)
        steps = int(maximum_search_steps)
        if dimension < 1 or steps < 1:
            raise ValueError("Halley dimension and search limits must be positive.")
        self.linear = linear_
        self.maximum_dimension = dimension
        self.maximum_search_steps = steps
        self.precision = precision_

    @property
    def method_id(self) -> str:
        return "vector-halley"

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
    ) -> NonlinearResult:
        self.precision.validate_tolerance(termination.absolute_residual)
        state = problem.validate_state(initial_state)
        residual, auxiliary = problem.evaluate(state, args)
        problem_ = problem.bind_spaces(state, residual)
        source = PyTreeSpace(state)
        target = PyTreeSpace(residual)
        if source.size != target.size or source.size > self.maximum_dimension:
            raise ValueError(
                "VectorHalley requires a square system within maximum_dimension."
            )
        self.precision.validate_trees(state, residual)
        norm = _coordinate_norm(target, residual, self.precision)
        finite = tree_allfinite(state) & tree_allfinite(residual)
        valid = problem_.valid(state, residual, auxiliary, args)
        run = _HalleyRun(
            state=state,
            residual=residual,
            auxiliary=auxiliary,
            initial_norm=jnp.maximum(norm, 1e-30),
            norm=norm,
            step_norm=jnp.asarray(0.0, dtype=norm.dtype),
            iteration=jnp.asarray(0, dtype=jnp.int32),
            residual_evaluations=jnp.asarray(1, dtype=jnp.int32),
            jvp_evaluations=jnp.asarray(0, dtype=jnp.int32),
            linear_solves=jnp.asarray(0, dtype=jnp.int32),
            linear_iterations=jnp.asarray(0, dtype=jnp.int32),
            accepted_steps=jnp.asarray(0, dtype=jnp.int32),
            rejected_steps=jnp.asarray(0, dtype=jnp.int32),
            status=jnp.where(
                finite & valid & (norm <= termination.residual_threshold(norm)),
                int(NonlinearStatus.SUCCESS),
                jnp.where(
                    finite & valid,
                    int(NonlinearStatus.ITERATING),
                    int(NonlinearStatus.NONFINITE_INPUT),
                ),
            ).astype(jnp.int32),
        )

        def condition(current):
            return (current.status == int(NonlinearStatus.ITERATING)) & (
                current.iteration < termination.maximum_steps
            )

        def body(current):
            def residual_function(value):
                return problem_.residual(value, args)

            _, jacobian_action = jax.linearize(residual_function, current.state)
            jacobian = FunctionLinearOperator(
                jacobian_action,
                source=source,
                target=target,
                operator_id=f"{problem_.problem_id}/halley-jacobian",
                closure_convert=False,
            )
            first = solve_linear(
                LinearSystem(jacobian),
                current.residual,
                policy=self.precision.bind_linear(self.linear),
            )
            inverse_residual = first.value

            def directional_jacobian(value):
                return jax.jvp(
                    residual_function,
                    (value,),
                    (inverse_residual,),
                )[1]

            def modified_action(direction):
                first_action = jacobian_action(direction)
                second_action = jax.jvp(
                    directional_jacobian,
                    (current.state,),
                    (direction,),
                )[1]
                return jax.tree.map(
                    lambda first_value, second_value: first_value - 0.5 * second_value,
                    first_action,
                    second_action,
                )

            modified = FunctionLinearOperator(
                modified_action,
                source=source,
                target=target,
                properties=OperatorProperties(),
                operator_id=f"{problem_.problem_id}/halley-modified-jacobian",
                closure_convert=False,
            )
            second = solve_linear(
                LinearSystem(modified),
                jax.tree.map(jnp.negative, current.residual),
                policy=self.precision.bind_linear(self.linear),
            )
            direction = second.value

            class _Search(eqx.Module):
                state: PyTree[Array]
                residual: PyTree[Array]
                auxiliary: Any
                norm: Array
                rate: Array
                evaluations: Array
                accepted: Array

            search = _Search(
                current.state,
                current.residual,
                current.auxiliary,
                current.norm,
                jnp.asarray(1.0, dtype=current.norm.dtype),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(False),
            )

            def search_condition(item):
                return (
                    ~item.accepted
                    & (item.evaluations < self.maximum_search_steps)
                    & (item.rate >= 1e-10)
                )

            def search_body(item):
                candidate = jax.tree.map(
                    lambda value, delta: jnp.asarray(
                        value + item.rate * delta,
                        dtype=value.dtype,
                    ),
                    current.state,
                    direction,
                )
                candidate_residual, candidate_auxiliary = problem_.evaluate(
                    candidate, args
                )
                candidate_norm = _coordinate_norm(
                    target,
                    candidate_residual,
                    self.precision,
                )
                finite_candidate = tree_allfinite(candidate) & tree_allfinite(
                    candidate_residual
                )
                valid_candidate = problem_.valid(
                    candidate, candidate_residual, candidate_auxiliary, args
                )
                accepted = (
                    first.diagnostics.converged
                    & second.diagnostics.converged
                    & finite_candidate
                    & valid_candidate
                    & (candidate_norm < current.norm)
                )
                return _Search(
                    jax.tree.map(
                        lambda proposed, old: jnp.where(accepted, proposed, old),
                        candidate,
                        item.state,
                    ),
                    jax.tree.map(
                        lambda proposed, old: jnp.where(accepted, proposed, old),
                        candidate_residual,
                        item.residual,
                    ),
                    jax.tree.map(
                        lambda proposed, old: jnp.where(accepted, proposed, old),
                        candidate_auxiliary,
                        item.auxiliary,
                    ),
                    jnp.where(accepted, candidate_norm, item.norm),
                    jnp.where(accepted, item.rate, 0.5 * item.rate),
                    item.evaluations + 1,
                    accepted,
                )

            search = jax.lax.while_loop(search_condition, search_body, search)
            step = jax.tree.map(lambda new, old: new - old, search.state, current.state)
            step_norm = _coordinate_norm(source, step, self.precision)
            converged = search.accepted & (
                search.norm <= termination.residual_threshold(current.initial_norm)
            )
            stagnated = (
                search.accepted
                & ~converged
                & (
                    step_norm
                    <= termination.step_threshold(
                        _coordinate_norm(source, current.state, self.precision)
                    )
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
                        int(NonlinearStatus.LINE_SEARCH_FAILED),
                    ),
                ),
            ).astype(jnp.int32)
            first_iterations = jnp.sum(first.diagnostics.iterations, dtype=jnp.int32)
            second_iterations = jnp.sum(second.diagnostics.iterations, dtype=jnp.int32)
            return _HalleyRun(
                state=search.state,
                residual=search.residual,
                auxiliary=search.auxiliary,
                initial_norm=current.initial_norm,
                norm=search.norm,
                step_norm=step_norm,
                iteration=current.iteration + 1,
                residual_evaluations=current.residual_evaluations + search.evaluations,
                jvp_evaluations=current.jvp_evaluations
                + jnp.sum(first.diagnostics.matvec_count, dtype=jnp.int32)
                + jnp.sum(second.diagnostics.matvec_count, dtype=jnp.int32)
                + 2,
                linear_solves=current.linear_solves + 2,
                linear_iterations=current.linear_iterations
                + first_iterations
                + second_iterations,
                accepted_steps=current.accepted_steps + search.accepted.astype(jnp.int32),
                rejected_steps=current.rejected_steps
                + (~search.accepted).astype(jnp.int32),
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
            jacobian_preparations=run.iteration,
            linear_solves=run.linear_solves,
            linear_iterations=run.linear_iterations,
            accepted_steps=run.accepted_steps,
            rejected_steps=run.rejected_steps,
        )
        output_state = jax.tree.map(self.precision.output, run.state)
        return NonlinearResult(
            state=output_state,
            residual=run.residual,
            auxiliary=run.auxiliary,
            status=status,
            diagnostics=diagnostics,
            provenance=NonlinearProvenance(
                problem_id=problem_.problem_id,
                method_id=self.method_id,
                derivative_id="nested-jvp-second-order",
                globalization_id="residual-decrease",
                precision_policy_id=self.precision.policy_id,
            ),
            precision_evidence=self.precision.evidence_for(
                run.state,
                run.residual,
                output_value=output_state,
            ),
        )


__all__ = ["VectorHalley"]
