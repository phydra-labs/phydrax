#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..problems import SparseLinearProblem
from ._availability import import_module, probe_modules
from .base import Availability, BenchmarkAdapter, CaseSpec, Implementation, SolveResult


_CAPABILITIES = frozenset({"linear.scalar", "linear.block"})


@dataclass
class _PyamgState:
    spec: CaseSpec
    matrix: Any
    hierarchy: Any = None


class PyamgAdapter(BenchmarkAdapter):
    """PyAMG smoothed-aggregation setup used as a CG preconditioner."""

    name = "pyamg"
    dependency = "pyamg+scipy"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("scipy", "pyamg"),
            distribution="pyamg",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        supported = spec.capability in self.capabilities
        return Implementation(
            adapter=self.name,
            backend="pyamg-cpu",
            method="conjugate-gradient" if supported else "unsupported",
            preconditioner="smoothed-aggregation" if supported else "none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _PyamgState:
        if not isinstance(spec.problem, SparseLinearProblem):
            raise TypeError("PyAMG adapter requires a SparseLinearProblem")
        sparse = import_module("scipy.sparse")
        problem = spec.problem
        matrix = sparse.csr_matrix(
            (problem.coefficients, (problem.rows, problem.columns)),
            shape=(problem.dimension, problem.dimension),
        )
        return _PyamgState(spec=spec, matrix=matrix)

    def preparation_applicable(self, compiled_state: _PyamgState, /) -> bool:
        return True

    def prepare(self, compiled_state: _PyamgState, /) -> _PyamgState:
        pyamg = import_module("pyamg")
        compiled_state.hierarchy = pyamg.smoothed_aggregation_solver(
            compiled_state.matrix,
            symmetry="symmetric",
            strength="symmetric",
        )
        return compiled_state

    def solve(self, prepared_state: _PyamgState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        rhs = problem.rhs[:, None] if problem.rhs.ndim == 1 else problem.rhs
        solutions: list[np.ndarray] = []
        total_iterations = 0
        converged = True
        for column in range(rhs.shape[1]):
            residuals: list[float] = []
            value = prepared_state.hierarchy.solve(
                rhs[:, column],
                x0=np.zeros(problem.dimension, dtype=np.float64),
                tol=prepared_state.spec.tolerances.relative,
                maxiter=prepared_state.spec.tolerances.max_steps,
                accel="cg",
                residuals=residuals,
            )
            solutions.append(np.asarray(value))
            total_iterations += max(0, len(residuals) - 1)
            final_residual = float(
                np.linalg.norm(prepared_state.matrix @ value - rhs[:, column])
            )
            threshold = (
                prepared_state.spec.tolerances.absolute
                + prepared_state.spec.tolerances.relative
                * float(np.linalg.norm(rhs[:, column]))
            )
            converged = converged and final_residual <= threshold
        stacked = np.column_stack(solutions)
        solution = stacked[:, 0] if problem.rhs.ndim == 1 else stacked
        return SolveResult(
            solution=solution,
            auxiliary={},
            converged=converged,
            message=(
                "PyAMG-preconditioned CG satisfied the common residual tolerance"
                if converged
                else "PyAMG-preconditioned CG did not satisfy the common residual tolerance"
            ),
            operations={
                "iterations": total_iterations,
                "matvecs": total_iterations + rhs.shape[1],
                "preconditioner_applications": total_iterations,
                "linear_solves": int(rhs.shape[1]),
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def memory(
        self,
        prepared_state: _PyamgState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        matrix_bytes = _sparse_bytes(prepared_state.matrix)
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": "exact fine-grid CSR bytes; PyAMG hierarchy and workspace storage estimates are unavailable",
        }


def _version_evidence() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("pyamg", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _sparse_bytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


__all__ = ["PyamgAdapter"]
