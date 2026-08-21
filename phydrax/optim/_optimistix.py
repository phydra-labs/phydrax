#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optimistix as optx
from jaxtyping import PyTree

from .._frozendict import frozendict
from ._iterative._base import AbstractMinimizationMethod
from ._iterative._types import (
    _tree_allfinite,
    _validate_real_inexact_tree,
    MinimizationProblem,
    MinimizationResult,
    OptimizationCapabilities,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationStatus,
    OptimizationTermination,
)


class OptimistixMethod(AbstractMinimizationMethod):
    """Explicit adapter for a verified public Optimistix minimizer instance."""

    solver: optx.AbstractMinimiser
    adjoint: optx.AbstractAdjoint
    options: frozendict[str, Any]

    def __init__(
        self,
        solver: optx.AbstractMinimiser,
        /,
        *,
        adjoint: optx.AbstractAdjoint | None = None,
        options: Mapping[str, Any] | None = None,
    ):
        if not isinstance(solver, optx.AbstractMinimiser):
            raise TypeError("solver must be an optimistix.AbstractMinimiser.")
        adjoint_ = optx.ImplicitAdjoint() if adjoint is None else adjoint
        if not isinstance(adjoint_, optx.AbstractAdjoint):
            raise TypeError("adjoint must be an optimistix.AbstractAdjoint or None.")
        if options is not None and not isinstance(options, Mapping):
            raise TypeError("options must be a mapping or None.")
        self.solver = solver
        self.adjoint = adjoint_
        self.options = frozendict({} if options is None else options)

    @property
    def method_id(self) -> str:
        return f"optimistix:{type(self.solver).__name__.lower()}"

    @property
    def capabilities(self) -> OptimizationCapabilities:
        return OptimizationCapabilities(
            scalar_objective=True,
            residual_objective=False,
            matrix_free=False,
            prepared_refresh=False,
            implicit_differentiation=isinstance(self.adjoint, optx.ImplicitAdjoint),
        )

    def solve(
        self,
        problem: MinimizationProblem,
        initial_parameters: PyTree[Any],
        /,
        *,
        termination: OptimizationTermination,
        args: Any,
    ) -> MinimizationResult:
        if not isinstance(problem, MinimizationProblem):
            raise TypeError("problem must be a MinimizationProblem.")
        if not isinstance(termination, OptimizationTermination):
            raise TypeError("termination must be an OptimizationTermination.")
        if problem.bounds is not None or problem.constraints:
            raise ValueError(
                "OptimistixMethod does not translate Phydrax bounds or constraints; "
                "use a native constrained method."
            )
        parameters = _validate_real_inexact_tree(
            initial_parameters,
            name="initial_parameters",
        )
        provenance = OptimizationProvenance(
            problem_id=problem.problem_id,
            method=self.method_id,
            backend="optimistix",
            backend_method=type(self.solver).__name__,
            globalization="backend-owned",
            matrix_free=False,
            implicit_differentiation=isinstance(
                self.adjoint,
                optx.ImplicitAdjoint,
            ),
            notes=(
                "Optimistix owns tolerance tests and internal evaluation counters; "
                "Phydrax normalizes typed status and the available total step count. "
                "Optimistix does not expose a portable objective-evaluation budget."
            ),
        )
        if termination.maximum_evaluations is not None:
            raise ValueError(
                "OptimistixMethod cannot enforce maximum_evaluations because "
                "Optimistix does not expose a portable evaluation-budget contract."
            )

        def nonfinite_input(_):
            value, auxiliary = problem.value(parameters, args)
            return MinimizationResult(
                parameters,
                value,
                auxiliary,
                OptimizationStatus.NONFINITE_INPUT,
                OptimizationDiagnostics(
                    objective_evaluations=1,
                    counts_complete=False,
                ),
                provenance,
            )

        def run_backend(_):
            solution = optx.minimise(
                problem.objective,
                self.solver,
                parameters,
                args,
                dict(self.options),
                has_aux=problem.has_aux,
                max_steps=termination.maximum_steps,
                adjoint=self.adjoint,
                throw=False,
            )
            value, auxiliary = problem.value(solution.value, args)
            status = jnp.where(
                solution.result == optx.RESULTS.successful,
                int(OptimizationStatus.SUCCESS),
                jnp.where(
                    solution.result == optx.RESULTS.nonlinear_max_steps_reached,
                    int(OptimizationStatus.MAXIMUM_STEPS_REACHED),
                    jnp.where(
                        solution.result == optx.RESULTS.nonfinite,
                        int(OptimizationStatus.NONFINITE_EVALUATION),
                        jnp.where(
                            solution.result == optx.RESULTS.nonlinear_divergence,
                            int(OptimizationStatus.DIVERGENCE),
                            int(OptimizationStatus.BACKEND_FAILED),
                        ),
                    ),
                ),
            )
            finite = jnp.isfinite(value) & _tree_allfinite(solution.value)
            status = jnp.where(
                finite,
                status,
                int(OptimizationStatus.NONFINITE_EVALUATION),
            ).astype(jnp.int32)
            steps = jnp.asarray(
                solution.stats.get("num_steps", -1),
                dtype=jnp.int32,
            )
            diagnostics = OptimizationDiagnostics(
                iterations=steps,
                accepted_steps=-1,
                rejected_steps=-1,
                objective_evaluations=-1,
                final_step_norm=jnp.nan,
                final_optimality_norm=jnp.nan,
                counts_complete=False,
            )
            return MinimizationResult(
                solution.value,
                value,
                auxiliary,
                status,
                diagnostics,
                provenance,
            )

        return jax.lax.cond(
            _tree_allfinite(parameters),
            run_backend,
            nonfinite_input,
            None,
        )


__all__ = ["OptimistixMethod"]
