#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..problems import SparseLinearProblem
from ._availability import import_module, probe_modules
from .base import Availability, BenchmarkAdapter, CaseSpec, Implementation, SolveResult


_CAPABILITIES = frozenset({"linear.scalar", "linear.block"})


@dataclass
class _AmgclState:
    spec: CaseSpec
    matrix: Any
    solvers: list[Any] = field(default_factory=list)


class AmgclAdapter(BenchmarkAdapter):
    """PyAMGCL default-AMG-preconditioned CG on host CSR systems."""

    name = "amgcl"
    dependency = "pyamgcl+scipy"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("scipy.sparse", "pyamgcl"),
            distribution="pyamgcl",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        supported = spec.capability in self.capabilities
        return Implementation(
            adapter=self.name,
            backend="pyamgcl-cpu",
            method="amgcl-cg-per-rhs" if supported else "unsupported",
            preconditioner="amgcl-default-amg-hierarchy" if supported else "none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _AmgclState:
        if not isinstance(spec.problem, SparseLinearProblem):
            raise TypeError(
                f"AMGCL adapter does not implement {spec.capability!r}; "
                "availability must be checked before setup"
            )
        sparse = import_module("scipy.sparse")
        problem = spec.problem
        matrix = sparse.csr_matrix(
            (problem.coefficients, (problem.rows, problem.columns)),
            shape=(problem.dimension, problem.dimension),
        )
        return _AmgclState(spec=spec, matrix=matrix)

    def preparation_applicable(self, compiled_state: _AmgclState, /) -> bool:
        return True

    def prepare(self, compiled_state: _AmgclState, /) -> _AmgclState:
        pyamgcl = import_module("pyamgcl")
        problem = compiled_state.spec.problem
        rhs = problem.rhs[:, None] if problem.rhs.ndim == 1 else problem.rhs
        tolerances = compiled_state.spec.tolerances
        compiled_state.solvers = []
        for column in range(rhs.shape[1]):
            rhs_norm = float(np.linalg.norm(rhs[:, column]))
            effective_relative = tolerances.relative
            if rhs_norm > 0.0:
                effective_relative += tolerances.absolute / rhs_norm
            compiled_state.solvers.append(
                pyamgcl.make_solver(
                    compiled_state.matrix,
                    solver=pyamgcl.solver_type.cg,
                    prm={
                        "tol": effective_relative,
                        "maxiter": tolerances.max_steps,
                    },
                )
            )
        return compiled_state

    def solve(self, prepared_state: _AmgclState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        rhs = problem.rhs[:, None] if problem.rhs.ndim == 1 else problem.rhs
        solutions = [
            np.asarray(solver(rhs[:, column]))
            for column, solver in enumerate(prepared_state.solvers)
        ]
        stacked = np.column_stack(solutions)
        solution = stacked[:, 0] if problem.rhs.ndim == 1 else stacked
        residual = np.asarray(prepared_state.matrix @ solution - problem.rhs)
        threshold = (
            prepared_state.spec.tolerances.absolute
            + prepared_state.spec.tolerances.relative * float(np.linalg.norm(problem.rhs))
        )
        converged = float(np.linalg.norm(residual)) <= threshold
        return SolveResult(
            solution=solution,
            auxiliary={},
            converged=converged,
            message=(
                "AMGCL CG satisfied the common residual tolerance"
                if converged
                else "AMGCL CG did not satisfy the common residual tolerance"
            ),
            operations={
                "iterations": None,
                "matvecs": None,
                "preconditioner_applications": None,
                "linear_solves": int(rhs.shape[1]),
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def memory(
        self,
        prepared_state: _AmgclState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        return {
            "matrix_bytes": _sparse_bytes(prepared_state.matrix),
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact fine-grid CSR bytes; PyAMGCL hierarchy and Krylov workspace "
                "storage are not exposed by the binding"
            ),
        }


def _version_evidence() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("pyamgcl", "scipy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _sparse_bytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


__all__ = ["AmgclAdapter"]
