#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

from ..problems import GeneralEigenProblem, OptimizationProblem, SparseLinearProblem
from ._availability import import_module, probe_modules
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    SolveResult,
    TransferEvidence,
)


_CAPABILITIES = frozenset(
    {
        "linear.scalar",
        "linear.block",
        "eigen.general",
        "optimization.unconstrained",
    }
)


@dataclass
class _JaxState:
    spec: CaseSpec
    matrix: Any
    rhs: Any = None
    executable: Any = None
    host_to_device_bytes: int = 0


class JaxAdapter(BenchmarkAdapter):
    """Dense JIT-compiled JAX baselines for mathematically matching cases."""

    name = "jax"
    dependency = "jax"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("jax", "jax.numpy"),
            distribution="jax",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        if spec.capability.startswith("linear."):
            method = "dense-jit-jax-numpy-linalg-solve"
        elif spec.capability == "eigen.general":
            method = "dense-jit-jax-numpy-linalg-eig-largest-magnitude"
        elif spec.capability == "optimization.unconstrained":
            method = "jax-scipy-optimize-bfgs"
        else:
            method = "unsupported"
        return Implementation(
            adapter=self.name,
            backend="jax-default-device",
            method=method,
            preconditioner="none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _JaxState:
        jnp = import_module("jax.numpy")
        problem = spec.problem
        if isinstance(problem, SparseLinearProblem):
            dense_matrix = problem.matrix
            return _JaxState(
                spec=spec,
                matrix=jnp.asarray(dense_matrix),
                rhs=jnp.asarray(problem.rhs),
                host_to_device_bytes=int(dense_matrix.nbytes + problem.rhs.nbytes),
            )
        if isinstance(problem, GeneralEigenProblem):
            if problem.variant != "standard-largest-magnitude":
                raise TypeError(
                    "direct JAX eig baseline supports only the standard "
                    "general-eigen contract"
                )
            return _JaxState(
                spec=spec,
                matrix=jnp.asarray(problem.matrix),
                host_to_device_bytes=int(problem.matrix.nbytes),
            )
        if isinstance(problem, OptimizationProblem):
            if problem.variant != "unconstrained":
                raise TypeError(
                    "direct JAX optimization baseline supports only the "
                    "unconstrained contract"
                )
            return _JaxState(
                spec=spec,
                matrix=jnp.asarray(problem.initial),
                host_to_device_bytes=int(problem.initial.nbytes),
            )
        raise TypeError(
            f"direct JAX adapter does not implement {spec.capability!r}; "
            "availability must be checked before setup"
        )

    def compilation_applicable(self, setup_state: _JaxState, /) -> bool:
        return True

    def compile(self, setup_state: _JaxState, /) -> _JaxState:
        jax = import_module("jax")
        jnp = import_module("jax.numpy")
        problem = setup_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            setup_state.executable = (
                jax.jit(jnp.linalg.solve)
                .lower(
                    setup_state.matrix,
                    setup_state.rhs,
                )
                .compile()
            )
            return setup_state
        if isinstance(problem, GeneralEigenProblem):
            count = problem.eigenpairs

            def selected_eigenpairs(matrix: Any) -> tuple[Any, Any]:
                eigenvalues, eigenvectors = jnp.linalg.eig(matrix)
                indices = jnp.argsort(jnp.abs(eigenvalues))[-count:]
                return eigenvalues[indices], eigenvectors[:, indices]

            setup_state.executable = (
                jax.jit(selected_eigenpairs).lower(setup_state.matrix).compile()
            )
            return setup_state
        if isinstance(problem, OptimizationProblem):
            scipy_optimize = import_module("jax.scipy.optimize")

            def minimize_rosenbrock(initial: Any) -> Any:
                return scipy_optimize.minimize(
                    lambda value: jnp.sum(
                        100.0 * (value[1:] - value[:-1] ** 2) ** 2
                        + (1.0 - value[:-1]) ** 2
                    ),
                    initial,
                    method="BFGS",
                    tol=max(
                        setup_state.spec.tolerances.absolute,
                        setup_state.spec.tolerances.relative,
                    ),
                    options={"maxiter": setup_state.spec.tolerances.max_steps},
                )

            setup_state.executable = (
                jax.jit(minimize_rosenbrock).lower(setup_state.matrix).compile()
            )
            return setup_state
        raise TypeError(f"unsupported JAX benchmark problem {type(problem).__name__!r}")

    def solve(self, prepared_state: _JaxState, /) -> SolveResult:
        jnp = import_module("jax.numpy")
        problem = prepared_state.spec.problem
        if isinstance(problem, SparseLinearProblem):
            solution = prepared_state.executable(
                prepared_state.matrix,
                prepared_state.rhs,
            )
            return SolveResult(
                solution=solution,
                auxiliary={},
                converged=jnp.all(jnp.isfinite(solution)),
                message="dense JAX solve completed; certificate verifies the equation",
                operations={
                    "iterations": 1,
                    "matvecs": 0,
                    "preconditioner_applications": 0,
                    "linear_solves": 1,
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, GeneralEigenProblem):
            eigenvalues, eigenvectors = prepared_state.executable(prepared_state.matrix)
            return SolveResult(
                solution=eigenvectors,
                auxiliary={"eigenvalues": eigenvalues},
                converged=(
                    jnp.all(jnp.isfinite(eigenvalues))
                    & jnp.all(jnp.isfinite(eigenvectors))
                ),
                message="dense JAX eig completed; certificate verifies selected eigenpairs",
                operations={
                    "iterations": None,
                    "matvecs": None,
                    "preconditioner_applications": 0,
                    "linear_solves": 0,
                    "nonlinear_evaluations": 0,
                    "jacobian_evaluations": 0,
                },
            )
        if isinstance(problem, OptimizationProblem):
            result = prepared_state.executable(prepared_state.matrix)
            return SolveResult(
                solution=result.x,
                auxiliary={"objective": result.fun, "status_code": result.status},
                converged=result.success,
                message="JAX BFGS completed; status is in auxiliary evidence",
                operations={
                    "iterations": result.nit,
                    "matvecs": 0,
                    "preconditioner_applications": 0,
                    "linear_solves": 0,
                    "nonlinear_evaluations": result.nfev,
                    "jacobian_evaluations": result.njev,
                },
            )
        raise TypeError(f"unsupported JAX benchmark problem {type(problem).__name__!r}")

    def memory(
        self,
        prepared_state: _JaxState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        return {
            "matrix_bytes": int(prepared_state.matrix.nbytes),
            "setup_bytes": 0,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact retained dense device input bytes; XLA temporary and output "
                "peak allocation is unavailable"
            ),
        }

    def transfers(
        self,
        prepared_state: _JaxState,
        result: SolveResult,
        /,
        *,
        device_to_host_bytes: int,
    ) -> TransferEvidence:
        del result
        return TransferEvidence(
            input_origin="numpy-host",
            host_to_device_bytes=prepared_state.host_to_device_bytes,
            host_to_device_timing_phase="setup",
            device_to_host_bytes=device_to_host_bytes,
            device_to_host_timing_phase="verification",
            evidence=(
                "dense canonical NumPy inputs were converted with jax.numpy.asarray "
                "during setup; solution, eigenpair, convergence, and operation arrays "
                "were materialized by jax.device_get during verification"
            ),
        )


def _version_evidence() -> dict[str, str]:
    return {"jax": importlib.metadata.version("jax")}


__all__ = ["JaxAdapter"]
