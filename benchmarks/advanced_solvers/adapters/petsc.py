#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..problems import NonlinearProblem, SparseLinearProblem
from ._availability import import_module, probe_modules
from .base import Availability, BenchmarkAdapter, CaseSpec, Implementation, SolveResult


_CAPABILITIES = frozenset(
    {"linear.scalar", "linear.block", "nonlinear.root", "nonlinear.vi"}
)


@dataclass
class _PetscState:
    spec: CaseSpec
    petsc: Any
    matrix: Any = None
    solver: Any = None
    residual_vector: Any = None
    jacobian: Any = None
    bounds: tuple[Any, Any] | None = None


class PetscAdapter(BenchmarkAdapter):
    """Sequential petsc4py KSP and SNES benchmark paths."""

    name = "petsc"
    dependency = "petsc4py+PETSc"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("petsc4py", "petsc4py.PETSc"),
            distribution="petsc4py",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        if spec.capability.startswith("linear."):
            method = "petsc-ksp-cg"
            preconditioner = "petsc-jacobi"
        elif spec.capability == "nonlinear.vi":
            method = "petsc-snes-vinewtonrsls"
            preconditioner = "petsc-default-linear-solver"
        elif spec.capability == "nonlinear.root":
            method = "petsc-snes-newtonls"
            preconditioner = "petsc-default-linear-solver"
        else:
            method = "unsupported"
            preconditioner = "none"
        return Implementation(
            adapter=self.name,
            backend="petsc-comm-self",
            method=method,
            preconditioner=preconditioner,
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _PetscState:
        petsc = import_module("petsc4py.PETSc")
        state = _PetscState(spec=spec, petsc=petsc)
        problem = spec.problem
        if isinstance(problem, SparseLinearProblem):
            indptr, indices, coefficients = _csr_arrays(problem, petsc.IntType)
            state.matrix = petsc.Mat().createAIJ(
                size=(problem.dimension, problem.dimension),
                csr=(indptr, indices, coefficients),
                comm=petsc.COMM_SELF,
            )
            state.matrix.assemble()
        return state

    def preparation_applicable(self, compiled_state: _PetscState, /) -> bool:
        return True

    def prepare(self, compiled_state: _PetscState, /) -> _PetscState:
        problem = compiled_state.spec.problem
        petsc = compiled_state.petsc
        tolerance = compiled_state.spec.tolerances
        if isinstance(problem, SparseLinearProblem):
            solver = petsc.KSP().create(comm=petsc.COMM_SELF)
            solver.setOperators(compiled_state.matrix)
            solver.setType(petsc.KSP.Type.CG)
            solver.getPC().setType(petsc.PC.Type.JACOBI)
            solver.setTolerances(
                rtol=tolerance.relative,
                atol=tolerance.absolute,
                max_it=tolerance.max_steps,
            )
            solver.setInitialGuessNonzero(False)
            solver.setUp()
            compiled_state.solver = solver
            return compiled_state
        if not isinstance(problem, NonlinearProblem):
            raise TypeError("PETSc adapter requires a linear or nonlinear problem")
        dimension = problem.initial.size
        solver = petsc.SNES().create(comm=petsc.COMM_SELF)
        residual_vector = petsc.Vec().createSeq(dimension, comm=petsc.COMM_SELF)
        jacobian = petsc.Mat().createDense(
            size=(dimension, dimension),
            comm=petsc.COMM_SELF,
        )
        jacobian.setUp()

        def residual_callback(snes: Any, value: Any, residual: Any) -> None:
            del snes
            residual.getArray()[:] = problem.residual(value.getArray(readonly=True))

        def jacobian_callback(
            snes: Any,
            value: Any,
            operator: Any,
            preconditioner: Any,
        ) -> None:
            del snes
            dense = problem.jacobian(value.getArray(readonly=True))
            indices = np.arange(dimension, dtype=petsc.IntType)
            operator.zeroEntries()
            operator.setValues(indices, indices, dense)
            operator.assemble()
            if preconditioner != operator:
                preconditioner.zeroEntries()
                preconditioner.setValues(indices, indices, dense)
                preconditioner.assemble()

        solver.setFunction(residual_callback, residual_vector)
        solver.setJacobian(jacobian_callback, J=jacobian, P=jacobian)
        if problem.variant == "vi":
            solver.setType(petsc.SNES.Type.VINEWTONRSLS)
            if problem.lower is None or problem.upper is None:
                raise ValueError("VI problem is missing its variable bounds")
            lower = petsc.Vec().createWithArray(
                np.asarray(problem.lower, dtype=np.float64).copy(),
                comm=petsc.COMM_SELF,
            )
            upper = petsc.Vec().createWithArray(
                np.asarray(problem.upper, dtype=np.float64).copy(),
                comm=petsc.COMM_SELF,
            )
            solver.setVariableBounds(lower, upper)
            compiled_state.bounds = (lower, upper)
        else:
            solver.setType(petsc.SNES.Type.NEWTONLS)
        solver.setTolerances(
            rtol=tolerance.relative,
            atol=tolerance.absolute,
            max_it=tolerance.max_steps,
        )
        solver.setUp()
        compiled_state.solver = solver
        compiled_state.residual_vector = residual_vector
        compiled_state.jacobian = jacobian
        return compiled_state

    def solve(self, prepared_state: _PetscState, /) -> SolveResult:
        problem = prepared_state.spec.problem
        petsc = prepared_state.petsc
        if isinstance(problem, SparseLinearProblem):
            rhs = problem.rhs[:, None] if problem.rhs.ndim == 1 else problem.rhs
            solutions: list[np.ndarray] = []
            total_iterations = 0
            converged = True
            reasons: list[int] = []
            for column in range(rhs.shape[1]):
                target = petsc.Vec().createWithArray(
                    np.asarray(rhs[:, column], dtype=np.float64).copy(),
                    comm=petsc.COMM_SELF,
                )
                value = target.duplicate()
                value.set(0.0)
                prepared_state.solver.solve(target, value)
                solutions.append(value.getArray(readonly=True).copy())
                total_iterations += int(prepared_state.solver.getIterationNumber())
                reason = int(prepared_state.solver.getConvergedReason())
                reasons.append(reason)
                converged = converged and reason > 0
            stacked = np.column_stack(solutions)
            solution = stacked[:, 0] if problem.rhs.ndim == 1 else stacked
            return SolveResult(
                solution=solution,
                auxiliary={"petsc_converged_reasons": reasons},
                converged=converged,
                message=f"PETSc KSP convergence reasons: {reasons}",
                operations={
                    "iterations": total_iterations,
                    "matvecs": None,
                    "preconditioner_applications": total_iterations,
                    "linear_solves": int(rhs.shape[1]),
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if not isinstance(problem, NonlinearProblem):
            raise TypeError("PETSc adapter requires a linear or nonlinear problem")
        value = petsc.Vec().createWithArray(
            np.asarray(problem.initial, dtype=np.float64).copy(),
            comm=petsc.COMM_SELF,
        )
        prepared_state.solver.solve(None, value)
        reason = int(prepared_state.solver.getConvergedReason())
        return SolveResult(
            solution=value.getArray(readonly=True).copy(),
            auxiliary={"petsc_converged_reason": reason},
            converged=reason > 0,
            message=f"PETSc SNES convergence reason: {reason}",
            operations={
                "iterations": int(prepared_state.solver.getIterationNumber()),
                "matvecs": None,
                "preconditioner_applications": None,
                "linear_solves": int(prepared_state.solver.getLinearSolveIterations()),
                "nonlinear_evaluations": int(
                    prepared_state.solver.getFunctionEvaluations()
                ),
                "jacobian_evaluations": None,
            },
        )

    def memory(
        self,
        prepared_state: _PetscState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            matrix_bytes = int(
                problem.coefficients.nbytes + problem.rows.nbytes + problem.columns.nbytes
            )
        else:
            arrays = [problem.initial, problem.target]
            if problem.lower is not None:
                arrays.append(problem.lower)
            if problem.upper is not None:
                arrays.append(problem.upper)
            if problem.diagonal is not None:
                arrays.append(problem.diagonal)
            matrix_bytes = int(sum(array.nbytes for array in arrays))
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": "exact benchmark coordinate/input bytes; PETSc retained and workspace peak is unavailable",
        }


def _csr_arrays(
    problem: SparseLinearProblem,
    index_dtype: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.lexsort((problem.columns, problem.rows))
    rows = problem.rows[order]
    columns = problem.columns[order].astype(index_dtype, copy=False)
    coefficients = problem.coefficients[order]
    indptr = np.zeros(problem.dimension + 1, dtype=index_dtype)
    np.add.at(indptr, rows + 1, 1)
    np.cumsum(indptr, out=indptr)
    return indptr, columns, coefficients


def _version_evidence() -> dict[str, str]:
    try:
        return {"petsc4py": importlib.metadata.version("petsc4py")}
    except importlib.metadata.PackageNotFoundError:
        return {}


__all__ = ["PetscAdapter"]
