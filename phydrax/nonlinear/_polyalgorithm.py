#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any

import equinox as eqx
from jaxtyping import PyTree

from ..linalg import PyTreeSpace
from ._newton import (
    _root_attempt_handoff,
    NewtonKrylov,
    NewtonTrustRegion,
)
from ._pseudo_transient import PseudoTransient
from ._quasi_newton import Broyden
from ._spectral import DFSANE
from ._types import (
    AbstractNonlinearMethod,
    NonlinearCapabilities,
    NonlinearDiagnostics,
    NonlinearProvenance,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
)
from ._work import NonlinearAttemptEvidence, NonlinearWork, work_sum


def _diagnostic_work(diagnostics: NonlinearDiagnostics, /) -> NonlinearWork:
    return NonlinearWork(
        residual_evaluations=diagnostics.residual_evaluations,
        validity_evaluations=diagnostics.residual_evaluations,
        jvp_evaluations=diagnostics.jvp_evaluations,
        vjp_evaluations=diagnostics.vjp_evaluations,
        jacobian_preparations=diagnostics.jacobian_preparations,
        linear_setups=diagnostics.setup_refreshes,
        linear_refreshes=diagnostics.numeric_refreshes,
        linear_solves=diagnostics.linear_solves,
        linear_iterations=diagnostics.linear_iterations,
        complete=diagnostics.counts_complete,
    )


def _remaining_termination(
    termination: NonlinearTermination,
    work: NonlinearWork,
    iterations: int,
    /,
) -> NonlinearTermination | None:
    remaining_steps = termination.maximum_steps - int(iterations)
    remaining_evaluations = (
        None
        if termination.maximum_evaluations is None
        else termination.maximum_evaluations - int(work.residual_evaluations)
    )
    remaining_linear = (
        None
        if termination.maximum_linear_iterations is None
        else termination.maximum_linear_iterations - int(work.linear_iterations)
    )
    if remaining_steps < 1:
        return None
    if remaining_evaluations is not None and remaining_evaluations < 1:
        return None
    if remaining_linear is not None and remaining_linear < 1:
        return None
    return NonlinearTermination(
        absolute_residual=termination.absolute_residual,
        relative_residual=termination.relative_residual,
        absolute_step=termination.absolute_step,
        relative_step=termination.relative_step,
        maximum_steps=remaining_steps,
        maximum_evaluations=remaining_evaluations,
        maximum_linear_iterations=remaining_linear,
        divergence_factor=termination.divergence_factor,
    )


class RootPolyalgorithm(AbstractNonlinearMethod):
    """Deterministic complete-solve attempts sharing one physical work budget."""

    methods: tuple[AbstractNonlinearMethod, ...]
    selector_id: str = eqx.field(static=True)

    def __init__(
        self,
        methods: tuple[AbstractNonlinearMethod, ...],
        /,
        *,
        selector_id: str = "root-polyalgorithm",
    ):
        methods_ = tuple(methods)
        if not methods_ or not all(
            isinstance(method, AbstractNonlinearMethod) for method in methods_
        ):
            raise TypeError(
                "methods must be a nonempty tuple of AbstractNonlinearMethod values."
            )
        identifier = str(selector_id)
        if not identifier:
            raise ValueError("selector_id must be non-empty.")
        self.methods = methods_
        self.selector_id = identifier

    @property
    def method_id(self) -> str:
        return self.selector_id

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=all(method.capabilities.matrix_free for method in self.methods),
            prepared_refresh=False,
            jit=False,
            implicit_differentiation=all(
                method.capabilities.implicit_differentiation for method in self.methods
            ),
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
        state = problem.validate_state(initial_state)
        problem_ = problem
        attempts = []
        work_values = []
        diagnostic_values = []
        used_iterations = 0
        best = None
        best_norm = float("inf")
        initial_norm = None
        initial_evaluation = None
        best_newton_internal = None
        residual_reuses = prepared_handoffs = 0
        for method in self.methods:
            used_work = work_sum(tuple(work_values))
            remaining = _remaining_termination(
                termination,
                used_work,
                used_iterations,
            )
            if remaining is None:
                break
            current_newton_internal = None
            if isinstance(method, (NewtonKrylov, NewtonTrustRegion)):
                prepared_start = (
                    None
                    if best_newton_internal is None
                    else _root_attempt_handoff(
                        method,
                        problem_,
                        state,
                        best.residual,
                        best.auxiliary,
                        best_newton_internal[0],
                        best_newton_internal[1],
                        args,
                    )
                )
                if prepared_start is not None:
                    prepared_handoffs += 1
                    residual_reuses += 1
                (
                    result,
                    _,
                    internal_run,
                    internal_jacobian,
                ) = method.solve(
                    problem_,
                    state,
                    termination=remaining,
                    args=args,
                    _prepared_start=prepared_start,
                    _return_internal=True,
                )
                current_newton_internal = (
                    internal_run,
                    internal_jacobian,
                )
            elif isinstance(method, (Broyden, PseudoTransient, DFSANE)):
                if initial_evaluation is not None:
                    residual_reuses += 1
                result = method.solve(
                    problem_,
                    state,
                    termination=remaining,
                    args=args,
                    _initial_evaluation=initial_evaluation,
                )
            else:
                result = method.solve(
                    problem_,
                    state,
                    termination=remaining,
                    args=args,
                )
            if initial_norm is None:
                initial_norm = result.diagnostics.initial_residual_norm
            work = _diagnostic_work(result.diagnostics)
            attempts.append(
                NonlinearAttemptEvidence(
                    component_id=method.method_id,
                    status=result.status,
                    accepted=result.successful,
                    input_residual_norm=(result.diagnostics.initial_residual_norm),
                    output_residual_norm=(result.diagnostics.final_residual_norm),
                    work=work,
                    failure_origin="solve-status",
                )
            )
            work_values.append(work)
            diagnostic_values.append(result.diagnostics)
            used_iterations += int(result.diagnostics.iterations)
            norm = float(result.diagnostics.final_residual_norm)
            if best is None or norm < best_norm:
                best = result
                best_norm = norm
                state = result.state
                problem_ = problem_.bind_spaces(state, result.residual)
                initial_evaluation = (
                    problem_,
                    state,
                    result.residual,
                    result.auxiliary,
                )
                best_newton_internal = current_newton_internal
            if bool(result.successful):
                best = result
                break
        if best is None or initial_norm is None:
            raise ValueError("The root attempt graph had no executable budget.")
        aggregate = work_sum(tuple(work_values))

        def diagnostic_sum(name: str):
            values = [vars(value)[name] for value in diagnostic_values]
            return sum(values[1:], values[0])

        final_diagnostics = NonlinearDiagnostics(
            initial_residual_norm=initial_norm,
            final_residual_norm=best.diagnostics.final_residual_norm,
            final_step_norm=best.diagnostics.final_step_norm,
            iterations=used_iterations,
            residual_evaluations=aggregate.residual_evaluations,
            jvp_evaluations=aggregate.jvp_evaluations,
            vjp_evaluations=aggregate.vjp_evaluations,
            jacobian_preparations=aggregate.jacobian_preparations,
            linear_solves=aggregate.linear_solves,
            linear_iterations=aggregate.linear_iterations,
            accepted_steps=diagnostic_sum("accepted_steps"),
            rejected_steps=diagnostic_sum("rejected_steps"),
            domain_failures=diagnostic_sum("domain_failures"),
            nonfinite_trials=diagnostic_sum("nonfinite_trials"),
            setup_refreshes=aggregate.linear_setups,
            numeric_refreshes=aggregate.linear_refreshes,
            final_forcing=best.diagnostics.final_forcing,
            final_trust_radius=best.diagnostics.final_trust_radius,
            final_linear_status=best.diagnostics.final_linear_status,
            final_linear_rank=best.diagnostics.final_linear_rank,
            final_linear_condition_estimate=(
                best.diagnostics.final_linear_condition_estimate
            ),
            final_linear_residual_norm=(best.diagnostics.final_linear_residual_norm),
            final_linear_converged=(best.diagnostics.final_linear_converged),
            counts_complete=aggregate.complete,
        )
        provenance = NonlinearProvenance(
            problem_id=problem_.problem_id,
            method_id=self.method_id,
            derivative_id="attempt-dependent",
            globalization_id="attempt-graph",
            notes=(
                "methods="
                + ",".join(attempt.component_id for attempt in attempts)
                + f";residual-reuses={residual_reuses}"
                + f";prepared-handoffs={prepared_handoffs}"
            ),
        )
        return NonlinearResult(
            state=best.state,
            residual=best.residual,
            auxiliary=best.auxiliary,
            status=best.status,
            diagnostics=final_diagnostics,
            provenance=provenance,
            transformation_evidence=best.transformation_evidence,
            attempts=tuple(attempts),
        )


class FastRoot(AbstractNonlinearMethod):
    """Capability-selected single native root method."""

    dense_dimension: int = eqx.field(static=True)

    def __init__(self, *, dense_dimension: int = 64):
        dimension = int(dense_dimension)
        if dimension < 1:
            raise ValueError("dense_dimension must be positive.")
        self.dense_dimension = dimension

    @property
    def method_id(self) -> str:
        return "fast-root"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return NonlinearCapabilities(
            matrix_free=True,
            prepared_refresh=True,
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
    ) -> NonlinearResult:
        state = problem.validate_state(initial_state)
        method = (
            NewtonTrustRegion()
            if PyTreeSpace(state).size <= self.dense_dimension
            else NewtonKrylov()
        )
        result = method.solve(
            problem,
            state,
            termination=termination,
            args=args,
        )
        provenance = NonlinearProvenance(
            problem_id=result.provenance.problem_id,
            method_id=self.method_id,
            derivative_id=result.provenance.derivative_id,
            globalization_id=result.provenance.globalization_id,
            linear_plan_id=result.provenance.linear_plan_id,
            notes=f"selected={method.method_id}",
        )
        return NonlinearResult(
            state=result.state,
            residual=result.residual,
            auxiliary=result.auxiliary,
            status=result.status,
            diagnostics=result.diagnostics,
            provenance=provenance,
            transformation_evidence=result.transformation_evidence,
            attempts=result.attempts,
        )


class RobustRoot(AbstractNonlinearMethod):
    """Deterministic Newton, secant, spectral, and pseudo-time attempts."""

    algorithm: RootPolyalgorithm

    def __init__(self):
        self.algorithm = RootPolyalgorithm(
            (
                NewtonKrylov(),
                NewtonTrustRegion(),
                Broyden("good"),
                PseudoTransient(),
                DFSANE(),
            ),
            selector_id="robust-root",
        )

    @property
    def method_id(self) -> str:
        return "robust-root"

    @property
    def capabilities(self) -> NonlinearCapabilities:
        return self.algorithm.capabilities

    def solve(
        self,
        problem,
        initial_state,
        /,
        *,
        termination,
        args=None,
    ) -> NonlinearResult:
        return self.algorithm.solve(
            problem,
            initial_state,
            termination=termination,
            args=args,
        )


__all__ = ["FastRoot", "RobustRoot", "RootPolyalgorithm"]
