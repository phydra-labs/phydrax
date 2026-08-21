#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..problems import GeneralEigenProblem
from ._availability import import_module, probe_modules
from .base import Availability, BenchmarkAdapter, CaseSpec, Implementation, SolveResult


_CAPABILITIES = frozenset({"eigen.general"})


@dataclass
class _SlepcState:
    spec: CaseSpec
    petsc: Any
    slepc: Any
    matrix: Any
    solver: Any = None
    initial_space: Any = None
    initial_values: np.ndarray | None = None


class SlepcAdapter(BenchmarkAdapter):
    """SLEPc NHEP largest-magnitude eigenpair benchmark path."""

    name = "slepc"
    dependency = "slepc4py+SLEPc+petsc4py+PETSc"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("petsc4py.PETSc", "slepc4py", "slepc4py.SLEPc"),
            distribution="slepc4py",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        return Implementation(
            adapter=self.name,
            backend="slepc-comm-self",
            method=(
                "slepc-eps-nhep-largest-magnitude"
                if spec.capability in self.capabilities
                else "unsupported"
            ),
            preconditioner="none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _SlepcState:
        if not isinstance(spec.problem, GeneralEigenProblem):
            raise TypeError("SLEPc adapter requires a GeneralEigenProblem")
        petsc = import_module("petsc4py.PETSc")
        slepc = import_module("slepc4py.SLEPc")
        problem = spec.problem
        matrix = petsc.Mat().createDense(
            size=problem.matrix.shape,
            array=np.asarray(problem.matrix, dtype=np.float64).copy(),
            comm=petsc.COMM_SELF,
        )
        matrix.assemble()
        return _SlepcState(spec=spec, petsc=petsc, slepc=slepc, matrix=matrix)

    def preparation_applicable(self, compiled_state: _SlepcState, /) -> bool:
        return True

    def prepare(self, compiled_state: _SlepcState, /) -> _SlepcState:
        problem = compiled_state.spec.problem
        tolerance = compiled_state.spec.tolerances
        solver = compiled_state.slepc.EPS().create(comm=compiled_state.petsc.COMM_SELF)
        solver.setOperators(compiled_state.matrix)
        solver.setProblemType(compiled_state.slepc.EPS.ProblemType.NHEP)
        solver.setDimensions(problem.eigenpairs)
        solver.setWhichEigenpairs(compiled_state.slepc.EPS.Which.LARGEST_MAGNITUDE)
        solver.setTolerances(tol=tolerance.relative, max_it=tolerance.max_steps)
        initial_space, _ = compiled_state.matrix.createVecs()
        initial_values = _deterministic_initial_vector(problem)
        initial_space.setValues(
            np.arange(problem.matrix.shape[0], dtype=compiled_state.petsc.IntType),
            np.asarray(initial_values, dtype=compiled_state.petsc.ScalarType),
        )
        initial_space.assemble()
        solver.setInitialSpace(initial_space)
        solver.setUp()
        compiled_state.initial_space = initial_space
        compiled_state.solver = solver
        compiled_state.initial_values = initial_values
        return compiled_state

    def solve(self, prepared_state: _SlepcState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        prepared_state.solver.solve()
        converged_count = int(prepared_state.solver.getConverged())
        returned = min(converged_count, problem.eigenpairs)
        fallback_used = returned == 0
        eigenvalues: list[complex] = []
        eigenvectors: list[np.ndarray] = []
        for index in range(returned):
            real_vector, imaginary_vector = prepared_state.matrix.createVecs()
            if np.issubdtype(
                np.dtype(prepared_state.petsc.ScalarType),
                np.complexfloating,
            ):
                eigenvalue = prepared_state.solver.getEigenpair(index, real_vector)
                vector = real_vector.getArray(readonly=True).copy().astype(np.complex128)
            else:
                eigenvalue = prepared_state.solver.getEigenpair(
                    index,
                    real_vector,
                    imaginary_vector,
                )
                vector = real_vector.getArray(readonly=True).copy().astype(np.complex128)
                vector += 1j * imaginary_vector.getArray(readonly=True).copy()
            eigenvalues.append(complex(eigenvalue))
            eigenvectors.append(vector)
        if fallback_used:
            vector = np.asarray(
                prepared_state.initial_values,
                dtype=np.complex128,
            )
            matrix = problem.matrix.astype(np.complex128)
            eigenvalue = np.vdot(vector, matrix @ vector) / np.vdot(vector, vector)
            eigenvalues.append(complex(eigenvalue))
            eigenvectors.append(vector)
        return SolveResult(
            solution=np.column_stack(eigenvectors),
            auxiliary={"eigenvalues": np.asarray(eigenvalues, dtype=np.complex128)},
            converged=converged_count >= problem.eigenpairs,
            message=(
                (
                    "SLEPc converged no eigenpairs; returned the deterministic "
                    "initial-space Rayleigh pair for independent residual evidence"
                )
                if fallback_used
                else (
                    f"SLEPc converged {converged_count} eigenpairs; "
                    f"requested {problem.eigenpairs}"
                )
            ),
            operations={
                "iterations": int(prepared_state.solver.getIterationNumber()),
                "matvecs": None,
                "preconditioner_applications": None,
                "linear_solves": None,
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def memory(
        self,
        prepared_state: _SlepcState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        matrix_bytes = int(prepared_state.spec.problem.matrix.nbytes)
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": "exact dense input bytes; SLEPc basis, output, and workspace peak are unavailable",
        }


def _deterministic_initial_vector(problem: GeneralEigenProblem, /) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(problem.seed))
    values = generator.standard_normal(problem.matrix.shape[0])
    return values / np.linalg.norm(values)


def _version_evidence() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("slepc4py", "petsc4py"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


__all__ = ["SlepcAdapter"]
