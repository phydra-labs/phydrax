#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PyTree

from .._bounds import Bounds
from .._nonlinear_precision import NonlinearPrecisionPolicy
from .._strict import StrictModule
from .._tree_math import validate_real_inexact_tree
from ..linalg import (
    DenseLinearOperator,
    DenseSVD,
    LeastSquaresProblem,
    LinearSolvePolicy,
    solve as solve_linear,
)
from ._iterative import (
    AbstractLeastSquaresMethod,
    LeastSquaresResult,
    NonlinearLeastSquaresProblem,
    OptimizationDiagnostics,
    OptimizationProvenance,
    OptimizationTermination,
)
from ._least_squares import (
    BoundedLevenbergMarquardt,
    least_squares,
    LevenbergMarquardt,
)


class VariableProjectionProblem(StrictModule):
    """Separable residual with nonlinear variables and one linear coefficient block."""

    design_matrix: Callable[[PyTree[Any], Any], Any]
    observations: Any
    offset: Callable[[PyTree[Any], Any], Any]
    bounds: Bounds | None
    regularization: float = eqx.field(static=True)
    problem_id: str = eqx.field(static=True)
    linear: LinearSolvePolicy
    precision: NonlinearPrecisionPolicy

    def __init__(
        self,
        design_matrix: Callable[[PyTree[Any], Any], Any],
        observations: Any,
        /,
        *,
        offset: Callable[[PyTree[Any], Any], Any] | None = None,
        bounds: Bounds | None = None,
        regularization: float = 0.0,
        problem_id: str = "variable-projection",
        linear: LinearSolvePolicy | None = None,
        precision: NonlinearPrecisionPolicy | None = None,
    ):
        if not callable(design_matrix):
            raise TypeError("design_matrix must be callable.")
        if offset is not None and not callable(offset):
            raise TypeError("offset must be callable or None.")
        observations_ = jnp.asarray(observations)
        if observations_.ndim != 1 or not jnp.issubdtype(
            observations_.dtype, jnp.inexact
        ):
            raise ValueError("observations must be one real inexact vector.")
        if bounds is not None and not isinstance(bounds, Bounds):
            raise TypeError("bounds must be Bounds or None.")
        regularization_ = float(regularization)
        if regularization_ < 0.0 or not jnp.isfinite(regularization_):
            raise ValueError("regularization must be finite and non-negative.")
        identifier = str(problem_id)
        if not identifier:
            raise ValueError("problem_id must be non-empty.")
        linear_ = LinearSolvePolicy(DenseSVD()) if linear is None else linear
        precision_ = NonlinearPrecisionPolicy() if precision is None else precision
        if not isinstance(linear_, LinearSolvePolicy):
            raise TypeError("linear must be LinearSolvePolicy or None.")
        if not isinstance(precision_, NonlinearPrecisionPolicy):
            raise TypeError("precision must be NonlinearPrecisionPolicy or None.")
        self.design_matrix = design_matrix
        self.observations = observations_
        self.offset = (
            (lambda parameters, args: jnp.zeros_like(observations_))
            if offset is None
            else offset
        )
        self.bounds = bounds
        self.regularization = regularization_
        self.problem_id = identifier
        self.linear = linear_
        self.precision = precision_

    def linear_solution(self, nonlinear_parameters, args=None, /):
        matrix = self.precision.accumulation(
            self.design_matrix(nonlinear_parameters, args)
        )
        offset = self.precision.residual(self.offset(nonlinear_parameters, args))
        if matrix.ndim != 2 or matrix.shape[0] != self.observations.size:
            raise ValueError("design_matrix must return (observations, coefficients).")
        if offset.shape != self.observations.shape:
            raise ValueError("offset must match observations.")
        right = self.observations - offset
        if self.regularization > 0.0:
            columns = matrix.shape[1]
            augmented_matrix = jnp.concatenate(
                [
                    matrix,
                    jnp.sqrt(self.regularization) * jnp.eye(columns, dtype=matrix.dtype),
                ],
                axis=0,
            )
            augmented_right = jnp.concatenate(
                [right, jnp.zeros((columns,), dtype=right.dtype)]
            )
        else:
            augmented_matrix = matrix
            augmented_right = right
        linear_result = solve_linear(
            LeastSquaresProblem(DenseLinearOperator(augmented_matrix)),
            self.precision.accumulation(augmented_right),
            policy=self.precision.bind_linear(self.linear),
        )
        solution = self.precision.direction(linear_result.value)
        rank = linear_result.diagnostics.rank
        singular_values = linear_result.diagnostics.singular_values
        assert singular_values is not None
        residual = self.precision.residual(matrix @ solution + offset - self.observations)
        return solution, residual, rank, singular_values


class VariableProjectionResult(StrictModule):
    nonlinear_result: LeastSquaresResult
    linear_parameters: Array
    residual: Array
    rank: Array
    singular_values: Array

    @property
    def successful(self):
        return self.nonlinear_result.successful

    @property
    def nonlinear_parameters(self):
        return self.nonlinear_result.parameters

    @property
    def objective(self):
        return self.nonlinear_result.objective


def variable_projection(
    problem: VariableProjectionProblem,
    initial_nonlinear_parameters: PyTree[Any],
    /,
    *,
    method: AbstractLeastSquaresMethod | None = None,
    termination: OptimizationTermination | None = None,
    args: Any = None,
) -> VariableProjectionResult:
    if not isinstance(problem, VariableProjectionProblem):
        raise TypeError("problem must be VariableProjectionProblem.")
    initial = validate_real_inexact_tree(
        initial_nonlinear_parameters,
        name="initial_nonlinear_parameters",
    )

    def reduced_residual(parameters, current_args):
        return problem.linear_solution(parameters, current_args)[1]

    reduced = NonlinearLeastSquaresProblem(
        reduced_residual,
        bounds=problem.bounds,
        problem_id=f"{problem.problem_id}/reduced",
    )
    method_ = (
        (
            BoundedLevenbergMarquardt()
            if problem.bounds is not None
            else LevenbergMarquardt()
        )
        if method is None
        else method
    )
    termination_ = OptimizationTermination() if termination is None else termination
    result = least_squares(
        reduced,
        initial,
        method=method_,
        termination=termination_,
        args=args,
    )
    linear, residual, rank, singular_values = problem.linear_solution(
        result.parameters,
        args,
    )
    diagnostics = OptimizationDiagnostics(
        iterations=result.diagnostics.iterations,
        accepted_steps=result.diagnostics.accepted_steps,
        rejected_steps=result.diagnostics.rejected_steps,
        objective_evaluations=result.diagnostics.objective_evaluations,
        gradient_evaluations=result.diagnostics.gradient_evaluations,
        residual_evaluations=result.diagnostics.residual_evaluations,
        jvp_evaluations=result.diagnostics.jvp_evaluations,
        vjp_evaluations=result.diagnostics.vjp_evaluations,
        hvp_evaluations=result.diagnostics.hvp_evaluations,
        jacobian_evaluations=result.diagnostics.jacobian_evaluations,
        constraint_evaluations=result.diagnostics.constraint_evaluations,
        linear_solves=result.diagnostics.linear_solves
        + result.diagnostics.residual_evaluations,
        setup_refreshes=result.diagnostics.setup_refreshes,
        numeric_refreshes=result.diagnostics.numeric_refreshes,
        linear_iterations=result.diagnostics.linear_iterations,
        globalization_evaluations=(result.diagnostics.globalization_evaluations),
        initial_optimality_norm=result.diagnostics.initial_optimality_norm,
        final_optimality_norm=result.diagnostics.final_optimality_norm,
        final_step_norm=result.diagnostics.final_step_norm,
        accepted_step_size=result.diagnostics.accepted_step_size,
        damping=result.diagnostics.damping,
        reduction_ratio=result.diagnostics.reduction_ratio,
        direction_fallbacks=result.diagnostics.direction_fallbacks,
        primal_feasibility=result.diagnostics.primal_feasibility,
        dual_feasibility=result.diagnostics.dual_feasibility,
        complementarity=result.diagnostics.complementarity,
        active_constraints=result.diagnostics.active_constraints,
        counts_complete=result.diagnostics.counts_complete,
    )
    model_parameters = problem.precision.state(result.parameters)
    model_residual = problem.precision.residual(residual)
    output_parameters = jax.tree.map(
        problem.precision.output,
        model_parameters,
    )
    residual_ = problem.precision.accumulation(model_residual)
    provenance = OptimizationProvenance(
        problem_id=result.provenance.problem_id,
        method=result.provenance.method,
        backend=result.provenance.backend,
        backend_method=result.provenance.backend_method,
        globalization=result.provenance.globalization,
        matrix_free=result.provenance.matrix_free,
        implicit_differentiation=result.provenance.implicit_differentiation,
        precision_policy_id=problem.precision.policy_id,
        notes=result.provenance.notes,
    )
    children = (
        {}
        if result.precision_evidence is None
        else {"reduced-solve": result.precision_evidence}
    )
    augmented_result = LeastSquaresResult(
        output_parameters,
        model_residual,
        problem.precision.decision(
            0.5 * jnp.real(jnp.sum(jnp.conj(residual_) * residual_))
        ),
        result.auxiliary,
        result.status,
        diagnostics,
        provenance,
        precision_evidence=problem.precision.evidence_for(
            model_parameters,
            model_residual,
            children=children,
            output_value=output_parameters,
        ),
    )
    return VariableProjectionResult(
        augmented_result,
        linear,
        residual,
        rank,
        singular_values,
    )


__all__ = [
    "VariableProjectionProblem",
    "VariableProjectionResult",
    "variable_projection",
]
