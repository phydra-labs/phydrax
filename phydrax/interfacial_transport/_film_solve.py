#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Prepared native Newton solve on a fixed film sparsity pattern.

Film residuals couple each vertex only with its one-ring, so the Jacobian
pattern is fixed by topology. Preparation compiles the native sparse
derivative and the symbolic sparse-LU pattern once on the host. By default a
solve refreshes the numeric factor at its initial guess; a bounded
nearby-state recurrence may supply one refreshed factor for reuse. It is only
a frozen right preconditioner for flexible GMRES: Newton actions stay exact
matrix-free JVPs. The residual-model trust region is
used because Armijo line search stalls on strongly nonlinear large-step
drainage systems. The forcing term is held constant at the declared linear
tolerance: adaptive Eisenstat--Walker forcing stays near its 0.9 cap while
the residual decreases slowly, and the resulting inexact directions stall the
trust region on large drainage steps (40 steps without convergence on a 6x6
film with 30 % perturbation at dt = 0.5, against 8 steps with accurate inner
solves). Newton steps, trust-region attempts and FGMRES iterations are all
bounded, and every status reaches the caller.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import equinox as eqx
import numpy as np
from jax import Array

from .._strict import StrictModule
from .._trainable import NonTrainableState
from ..linalg import (
    ArraySpace,
    FailurePolicy,
    FGMRES,
    LinearSolvePolicy,
    PreconditionerProperties,
    PreconditioningPolicy,
    prepare_sparse_factorization,
    PreparedSparseFactorization,
    refresh_sparse_factorization,
    SparseFactorizationPlan,
    SparseFactorizationPolicy,
    SparseFactorizationPreconditioner,
    TolerancePolicy,
)
from ..nonlinear import (
    NewtonForcingPolicy,
    NewtonTrustRegion,
    NonlinearResult,
    NonlinearSystemProblem,
    NonlinearTermination,
    root,
)
from ..sparse import compile_sparse_jacobian, EdgeRelation, SparseDerivativePlan
from ..typing import checked


def vertex_block_pattern(
    edges: np.ndarray, vertex_count: int, block_count: int, /
) -> EdgeRelation:
    """Return the one-ring pattern coupling every pair of vertex blocks.

    Unknowns are ordered block-major: ``block * vertex_count + vertex``.
    """
    vertices = np.arange(vertex_count, dtype=np.int64)
    rows = np.concatenate((vertices, edges[:, 0], edges[:, 1]))
    columns = np.concatenate((vertices, edges[:, 1], edges[:, 0]))
    row_blocks = []
    column_blocks = []
    for row_block in range(block_count):
        for column_block in range(block_count):
            row_blocks.append(rows + row_block * vertex_count)
            column_blocks.append(columns + column_block * vertex_count)
    size = block_count * vertex_count
    return EdgeRelation(
        np.concatenate(column_blocks).astype(np.int32),
        np.concatenate(row_blocks).astype(np.int32),
        source_size=size,
        target_size=size,
    )


class PreparedFilmNewton(StrictModule, NonTrainableState):
    """Native trust-region Newton--FGMRES with a reusable sparse-LU preconditioner."""

    problem: NonlinearSystemProblem
    derivative: SparseDerivativePlan
    factorization: SparseFactorizationPlan
    termination: NonlinearTermination
    linear_relative_tolerance: float = eqx.field(static=True)
    solver_id: str = eqx.field(static=True)

    @checked
    def __init__(
        self,
        residual: Callable[[Array, Any], Array],
        pattern: EdgeRelation,
        sample_state: Array,
        sample_args: Any,
        /,
        *,
        termination: NonlinearTermination,
        solver_id: str,
    ) -> None:
        size = sample_state.shape[0]
        space = ArraySpace((size,), dtype=np.float64)
        derivative = compile_sparse_jacobian(
            residual,
            sample_state,
            source=space,
            target=space,
            sample_args=sample_args,
            structure=pattern,
            compiler="native",
            mode="fwd",
            plan_id=f"{solver_id}/jacobian",
        )
        self.problem = NonlinearSystemProblem(
            residual,
            state_space=space,
            residual_space=space,
            problem_id=solver_id,
        )
        self.derivative = derivative
        self.factorization = prepare_sparse_factorization(
            derivative.operator(sample_state, sample_args),
            SparseFactorizationPolicy("lu", ordering="reverse-cuthill-mckee"),
        )
        self.termination = termination
        self.linear_relative_tolerance = 1e-10
        self.solver_id = solver_id

    def factorize(self, initial: Array, args: Any, /) -> PreparedSparseFactorization:
        """Refresh the numeric sparse-LU factor at one nonlinear initial guess."""
        return refresh_sparse_factorization(
            self.factorization, self.derivative.operator(initial, args)
        )

    def solve(
        self,
        initial: Array,
        args: Any,
        /,
        *,
        factorization: PreparedSparseFactorization | None = None,
    ) -> NonlinearResult:
        jacobian = self.derivative.operator(initial, args)
        factor = (
            refresh_sparse_factorization(self.factorization, jacobian)
            if factorization is None
            else factorization
        )
        if factor.plan.plan_id != self.factorization.plan_id:
            raise ValueError("Reusable film factorization belongs to another solver.")
        preconditioner = SparseFactorizationPreconditioner(
            jacobian,
            factor,
            properties=PreconditionerProperties(linear=True, stationary=True),
            preconditioner_id=f"{self.solver_id}/frozen-lu",
        )
        tolerance = self.linear_relative_tolerance
        method = NewtonTrustRegion(
            linear_policy=LinearSolvePolicy(
                FGMRES(restart=30),
                tolerance=TolerancePolicy(relative=tolerance, absolute=0.0, max_steps=90),
                preconditioning=PreconditioningPolicy(preconditioner, side="right"),
                failure=FailurePolicy("status"),
            ),
            forcing_policy=NewtonForcingPolicy(
                "constant", initial=tolerance, minimum=tolerance, maximum=tolerance
            ),
        )
        return root(
            self.problem,
            initial,
            method=method,
            termination=self.termination,
            args=args,
        )


def film_termination(
    *, tolerance: float, maximum_iterations: int
) -> NonlinearTermination:
    """Return the dimensionless residual termination shared by film routes."""
    return NonlinearTermination(
        absolute_residual=tolerance,
        relative_residual=0.0,
        absolute_step=0.0,
        relative_step=0.0,
        maximum_steps=maximum_iterations,
    )


__all__ = ["PreparedFilmNewton", "film_termination", "vertex_block_pattern"]
