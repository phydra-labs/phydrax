#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..problems import (
    GeneralEigenProblem,
    NonlinearProblem,
    OptimizationProblem,
    SparseLinearProblem,
)
from ._availability import import_module, probe_modules
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    SolveResult,
)


_CAPABILITIES = frozenset(
    {
        "linear.scalar",
        "linear.block",
        "nonlinear.root",
        "nonlinear.vi",
        "eigen.general",
        "optimization.unconstrained",
        "optimization.constrained",
        "optimization.proximal",
        "optimization.bounded-least-squares",
    }
)


@dataclass
class _ScipyState:
    spec: CaseSpec
    matrix: Any = None
    factor: Any = None


class ScipyAdapter(BenchmarkAdapter):
    """SciPy SuperLU, nonlinear optimization, and ARPACK reference paths."""

    name = "scipy"
    dependency = "scipy"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("scipy",),
            distribution="scipy",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        method, preconditioner = _configuration(spec)
        return Implementation(
            adapter=self.name,
            backend="scipy-cpu",
            method=method,
            preconditioner=preconditioner,
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _ScipyState:
        sparse = import_module("scipy.sparse")
        problem = spec.problem
        state = _ScipyState(spec=spec)
        if isinstance(problem, SparseLinearProblem):
            state.matrix = sparse.csc_matrix(
                (problem.coefficients, (problem.rows, problem.columns)),
                shape=(problem.dimension, problem.dimension),
            )
        elif isinstance(problem, GeneralEigenProblem):
            state.matrix = sparse.csc_matrix(problem.matrix)
        return state

    def preparation_applicable(self, compiled_state: _ScipyState, /) -> bool:
        return isinstance(compiled_state.spec.problem, SparseLinearProblem)

    def prepare(self, compiled_state: _ScipyState, /) -> _ScipyState:
        sparse_linalg = import_module("scipy.sparse.linalg")
        compiled_state.factor = sparse_linalg.splu(compiled_state.matrix)
        return compiled_state

    def solve(self, prepared_state: _ScipyState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            value = prepared_state.factor.solve(problem.rhs)
            rhs_count = 1 if problem.rhs.ndim == 1 else problem.rhs.shape[1]
            return SolveResult(
                solution=value,
                auxiliary={},
                converged=True,
                message="SuperLU factor-and-solve completed",
                operations={
                    "iterations": 0,
                    "matvecs": 0,
                    "preconditioner_applications": 0,
                    "linear_solves": int(rhs_count),
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, GeneralEigenProblem):
            sparse_linalg = import_module("scipy.sparse.linalg")
            initial = np.linspace(
                1.0,
                2.0,
                problem.matrix.shape[0],
                dtype=np.float64,
            )
            try:
                eigenvalues, eigenvectors = sparse_linalg.eigs(
                    prepared_state.matrix,
                    k=problem.eigenpairs,
                    which="LM",
                    tol=prepared_state.spec.tolerances.relative,
                    maxiter=prepared_state.spec.tolerances.max_steps,
                    v0=initial,
                )
                converged = True
                message = "ARPACK returned the requested eigenpairs"
            except sparse_linalg.ArpackNoConvergence as error:
                eigenvalues = np.asarray(error.eigenvalues)
                eigenvectors = np.asarray(error.eigenvectors)
                if eigenvalues.size == 0:
                    vector = initial / np.linalg.norm(initial)
                    eigenvalue = vector @ problem.matrix @ vector
                    eigenvalues = np.asarray([eigenvalue], dtype=np.complex128)
                    eigenvectors = vector[:, None].astype(np.complex128)
                    message = (
                        "ARPACK converged no Ritz pairs; returned the deterministic "
                        "initial-space Rayleigh pair for independent residual evidence"
                    )
                else:
                    message = (
                        f"ARPACK returned {eigenvalues.size} converged Ritz pairs "
                        f"of {problem.eigenpairs} requested"
                    )
                converged = False
            return SolveResult(
                solution=eigenvectors,
                auxiliary={"eigenvalues": eigenvalues},
                converged=converged,
                message=message,
                operations={
                    "iterations": None,
                    "matvecs": None,
                    "preconditioner_applications": 0,
                    "linear_solves": 0,
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, NonlinearProblem):
            return self._solve_nonlinear(prepared_state, problem)
        if isinstance(problem, OptimizationProblem):
            return self._solve_optimization(prepared_state, problem)
        raise TypeError(f"unsupported SciPy problem type {type(problem).__name__!r}")

    def memory(
        self,
        prepared_state: _ScipyState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            matrix_bytes = _sparse_bytes(prepared_state.matrix)
            factor_bytes = _sparse_bytes(prepared_state.factor.L) + _sparse_bytes(
                prepared_state.factor.U
            )
            return {
                "matrix_bytes": matrix_bytes,
                "setup_bytes": factor_bytes,
                "peak_estimate_bytes": None,
                "evidence": "exact retained CSC and SuperLU L/U storage; factorization/workspace peak is unavailable",
            }
        if isinstance(problem, GeneralEigenProblem):
            matrix_bytes = _sparse_bytes(prepared_state.matrix)
            return {
                "matrix_bytes": matrix_bytes,
                "setup_bytes": 0,
                "peak_estimate_bytes": None,
                "evidence": "exact retained CSC bytes; ARPACK basis/output/workspace peak is unavailable",
            }
        if isinstance(problem, NonlinearProblem):
            arrays = [problem.initial, problem.target]
            if problem.lower is not None:
                arrays.append(problem.lower)
            if problem.upper is not None:
                arrays.append(problem.upper)
            if problem.diagonal is not None:
                arrays.append(problem.diagonal)
        else:
            arrays = [problem.initial, problem.optimum]
            if problem.target is not None:
                arrays.append(problem.target)
        problem_bytes = sum(array.nbytes for array in arrays)
        return {
            "matrix_bytes": int(problem_bytes),
            "setup_bytes": 0,
            "peak_estimate_bytes": None,
            "evidence": "exact benchmark input bytes; optimizer retained/workspace peak is unavailable",
        }

    def _solve_nonlinear(
        self,
        state: _ScipyState,
        problem: NonlinearProblem,
    ) -> SolveResult:
        optimize = import_module("scipy.optimize")
        tolerance = state.spec.tolerances
        if problem.variant == "root":
            result = optimize.root(
                problem.residual,
                problem.initial,
                jac=problem.jacobian,
                method="hybr",
                options={"xtol": tolerance.relative, "maxfev": tolerance.max_steps},
            )
            evaluations = int(result.nfev)
            jacobian_evaluations = int(result.njev)
            iterations = None
        else:
            if problem.diagonal is None or problem.lower is None or problem.upper is None:
                raise ValueError("VI problem is missing bounds or its diagonal operator")

            def objective(value: np.ndarray) -> float:
                return float(
                    0.5 * np.sum(problem.diagonal * value * value)
                    - problem.target @ value
                )

            result = optimize.minimize(
                objective,
                problem.initial,
                jac=problem.residual,
                method="L-BFGS-B",
                bounds=list(zip(problem.lower, problem.upper, strict=True)),
                options={
                    "ftol": tolerance.relative,
                    "gtol": tolerance.absolute,
                    "maxiter": tolerance.max_steps,
                },
            )
            evaluations = int(result.nfev)
            jacobian_evaluations = int(result.njev)
            iterations = int(result.nit)
        return SolveResult(
            solution=result.x,
            auxiliary={},
            converged=bool(result.success),
            message=str(result.message),
            operations={
                "iterations": iterations,
                "matvecs": None,
                "preconditioner_applications": 0,
                "linear_solves": None,
                "nonlinear_evaluations": evaluations,
                "jacobian_evaluations": jacobian_evaluations,
            },
        )

    def _solve_optimization(
        self,
        state: _ScipyState,
        problem: OptimizationProblem,
    ) -> SolveResult:
        optimize = import_module("scipy.optimize")
        tolerance = state.spec.tolerances
        options = {"maxiter": tolerance.max_steps}
        if problem.variant == "unconstrained":
            options.update(
                {
                    "gtol": tolerance.absolute,
                }
            )
            result = optimize.minimize(
                problem.objective,
                problem.initial,
                jac=problem.gradient,
                method="BFGS",
                options=options,
            )
        elif problem.variant == "constrained":
            constraints = (
                {
                    "type": "eq",
                    "fun": problem.equality,
                    "jac": lambda value: 2.0 * np.asarray(value),
                },
                {
                    "type": "ineq",
                    "fun": lambda value: -problem.inequality(value),
                    "jac": lambda value: -np.ones_like(value),
                },
            )
            options["ftol"] = max(tolerance.absolute, tolerance.relative)
            result = optimize.minimize(
                problem.objective,
                problem.initial,
                jac=problem.gradient,
                constraints=constraints,
                method="SLSQP",
                options=options,
            )
        elif problem.variant == "bounded-least-squares":
            options.update(
                {
                    "ftol": tolerance.absolute,
                    "gtol": tolerance.absolute,
                }
            )
            result = optimize.minimize(
                problem.objective,
                problem.initial,
                jac=problem.gradient,
                bounds=[(0.0, 1.0)] * problem.initial.size,
                method="L-BFGS-B",
                options=options,
            )
        else:
            options.update(
                {
                    "xtol": tolerance.relative,
                    "ftol": tolerance.absolute,
                }
            )
            result = optimize.minimize(
                problem.objective,
                problem.initial,
                method="Powell",
                options=options,
            )
        return SolveResult(
            solution=result.x,
            auxiliary={"objective": result.fun, "status_code": result.status},
            converged=bool(result.success),
            message=str(result.message),
            operations={
                "iterations": int(result.nit),
                "matvecs": 0,
                "preconditioner_applications": 0,
                "linear_solves": 0,
                "nonlinear_evaluations": int(result.nfev),
                "jacobian_evaluations": (
                    None if result.get("njev") is None else int(result["njev"])
                ),
            },
        )


def _configuration(spec: CaseSpec) -> tuple[str, str]:
    capability = spec.capability
    if capability.startswith("linear."):
        return "superlu", "none"
    if capability == "eigen.general":
        return "arpack-eigs-largest-magnitude", "none"
    if capability == "nonlinear.root":
        return "hybr", "none"
    if capability == "nonlinear.vi":
        return "l-bfgs-b-bound-vi", "none"
    if capability == "optimization.unconstrained":
        return "bfgs", "none"
    if capability == "optimization.constrained":
        return "slsqp", "dense-bfgs-qp"
    if capability == "optimization.proximal":
        return "powell-exact-composite-objective", "none"
    if capability == "optimization.bounded-least-squares":
        return "l-bfgs-b-bound-least-squares", "none"
    return "unsupported", "none"


def _version_evidence() -> dict[str, str]:
    try:
        return {"scipy": importlib.metadata.version("scipy")}
    except importlib.metadata.PackageNotFoundError:
        return {}


def _sparse_bytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


__all__ = ["ScipyAdapter"]
