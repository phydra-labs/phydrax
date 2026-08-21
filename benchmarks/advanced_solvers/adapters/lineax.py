#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import importlib.metadata
from dataclasses import dataclass
from typing import Any

from ..problems import SparseLinearProblem
from ._availability import import_module, probe_modules
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    SolveResult,
    TransferEvidence,
)


_CAPABILITIES = frozenset({"linear.scalar", "linear.block"})


@dataclass
class _LineaxState:
    spec: CaseSpec
    matrix: Any
    rhs: Any
    executable: Any = None
    host_to_device_bytes: int = 0


class LineaxAdapter(BenchmarkAdapter):
    """JIT-compiled Lineax CG with one mathematically identical solve per RHS."""

    name = "lineax"
    dependency = "lineax+jax"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        return probe_modules(
            adapter=self.name,
            dependency=self.dependency,
            capability=capability,
            supported=self.capabilities,
            modules=("jax", "lineax"),
            distribution="lineax",
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        supported = spec.capability in self.capabilities
        return Implementation(
            adapter=self.name,
            backend="lineax-jax-default-device",
            method=("lineax-cg-vmap-per-rhs" if supported else "unsupported"),
            preconditioner="none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _LineaxState:
        if not isinstance(spec.problem, SparseLinearProblem):
            raise TypeError(
                f"Lineax adapter does not implement {spec.capability!r}; "
                "availability must be checked before setup"
            )
        jnp = import_module("jax.numpy")
        dense_matrix = spec.problem.matrix
        return _LineaxState(
            spec=spec,
            matrix=jnp.asarray(dense_matrix),
            rhs=jnp.asarray(spec.problem.rhs),
            host_to_device_bytes=int(dense_matrix.nbytes + spec.problem.rhs.nbytes),
        )

    def compilation_applicable(self, setup_state: _LineaxState, /) -> bool:
        return True

    def compile(self, setup_state: _LineaxState, /) -> _LineaxState:
        jax = import_module("jax")
        lx = import_module("lineax")
        tolerance = setup_state.spec.tolerances

        def solve_one(matrix: Any, rhs: Any) -> tuple[Any, Any, Any]:
            operator = lx.MatrixLinearOperator(
                matrix,
                tags=frozenset({lx.positive_semidefinite_tag, lx.symmetric_tag}),
            )
            solution = lx.linear_solve(
                operator,
                rhs,
                solver=lx.CG(
                    rtol=tolerance.relative,
                    atol=tolerance.absolute,
                    max_steps=tolerance.max_steps,
                ),
                throw=False,
            )
            return (
                solution.value,
                solution.result == lx.RESULTS.successful,
                solution.stats["num_steps"],
            )

        if setup_state.rhs.ndim == 1:
            operation = solve_one
        else:
            operation = jax.vmap(solve_one, in_axes=(None, 1), out_axes=(1, 0, 0))
        setup_state.executable = (
            jax.jit(operation)
            .lower(
                setup_state.matrix,
                setup_state.rhs,
            )
            .compile()
        )
        return setup_state

    def solve(self, prepared_state: _LineaxState, /) -> SolveResult:
        jnp = import_module("jax.numpy")
        solution, successful, steps = prepared_state.executable(
            prepared_state.matrix,
            prepared_state.rhs,
        )
        rhs_count = 1 if prepared_state.rhs.ndim == 1 else prepared_state.rhs.shape[1]
        return SolveResult(
            solution=solution,
            auxiliary={"successful_per_rhs": successful},
            converged=jnp.all(successful),
            message="Lineax CG completed; per-RHS status is in auxiliary evidence",
            operations={
                "iterations": jnp.max(steps),
                "matvecs": None,
                "preconditioner_applications": 0,
                "linear_solves": rhs_count,
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def memory(
        self,
        prepared_state: _LineaxState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        return {
            "matrix_bytes": int(prepared_state.matrix.nbytes),
            "setup_bytes": 0,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact dense device matrix bytes; Lineax/XLA Krylov and output peak "
                "allocation is unavailable"
            ),
        }

    def transfers(
        self,
        prepared_state: _LineaxState,
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
                "dense canonical NumPy matrix and RHS arrays were converted during "
                "setup; Lineax solution, per-RHS status, and iteration arrays were "
                "materialized by jax.device_get during verification"
            ),
        )


def _version_evidence() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("lineax", "jax"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


__all__ = ["LineaxAdapter"]
