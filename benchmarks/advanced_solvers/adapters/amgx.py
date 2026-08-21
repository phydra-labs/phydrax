#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..problems import SparseLinearProblem
from ._availability import import_module, unsupported
from .base import (
    Availability,
    BenchmarkAdapter,
    CaseSpec,
    Implementation,
    RefreshEvidence,
    SolveResult,
    TransferEvidence,
)


_CAPABILITIES = frozenset({"linear.scalar", "linear.block"})


@dataclass
class _AmgxState:
    spec: CaseSpec
    phx: Any
    backend: Any
    native_problem: Any
    rhs: Any
    policy: Any
    plan: Any = None
    prepared: Any = None
    refreshed_certificate_problem: Any = None
    preparation_host_to_device_bytes: int = 0
    solve_host_to_device_bytes: int = 0
    solve_device_to_host_bytes: int = 0
    jax_result_device_bytes: int = 0


class AmgxAdapter(BenchmarkAdapter):
    """Phydrax public AmgX lifecycle with the canonical FGMRES+AMG policy."""

    name = "amgx"
    dependency = "phydrax+pyamgx+NVIDIA-AmgX+CUDA"
    capabilities = _CAPABILITIES

    def availability(self, capability: str, /) -> Availability:
        if capability not in self.capabilities:
            return unsupported(
                adapter=self.name,
                dependency=self.dependency,
                capability=capability,
            )
        backends = import_module("phydrax.backends")
        evidence = backends.amgx_availability()
        versions = dict(evidence.versions)
        return Availability(
            available=evidence.available,
            capability=capability,
            dependency=self.dependency,
            dependency_version=versions.get("pyamgx"),
            reason=None if evidence.available else evidence.reason,
        )

    def implementation(self, spec: CaseSpec, /) -> Implementation:
        supported = spec.capability in self.capabilities
        return Implementation(
            adapter=self.name,
            backend="phydrax-amgx-public-cuda",
            method="fgmres" if supported else "unsupported",
            preconditioner="amg-aggregation" if supported else "none",
            versions=_version_evidence(),
        )

    def setup(self, spec: CaseSpec, /) -> _AmgxState:
        if not isinstance(spec.problem, SparseLinearProblem):
            raise TypeError(
                f"AmgX adapter does not implement {spec.capability!r}; "
                "no CPU fallback is provided"
            )
        phx = import_module("phydrax")
        backends = import_module("phydrax.backends")
        jnp = import_module("jax.numpy")
        problem = spec.problem
        native_problem = _native_linear_problem(phx, jnp, problem)
        rhs = jnp.asarray(problem.rhs)
        rhs_matrix = problem.rhs[:, None] if problem.rhs.ndim == 1 else problem.rhs
        rhs_norms = np.linalg.norm(rhs_matrix, axis=0)
        effective_tolerances = [
            spec.tolerances.relative
            + (spec.tolerances.absolute / norm if norm > 0.0 else 0.0)
            for norm in rhs_norms
        ]
        policy = backends.AmgXPolicy(
            {
                "config_version": 2,
                "solver": {
                    "solver": "FGMRES",
                    "preconditioner": {
                        "solver": "AMG",
                        "algorithm": "AGGREGATION",
                    },
                    "max_iters": spec.tolerances.max_steps,
                    "tolerance": min(effective_tolerances),
                    "norm": "L2",
                },
            }
        )
        return _AmgxState(
            spec=spec,
            phx=phx,
            backend=backends.AmgXBackend(),
            native_problem=native_problem,
            rhs=rhs,
            policy=policy,
        )

    def compilation_applicable(self, setup_state: _AmgxState, /) -> bool:
        return True

    def compile(self, setup_state: _AmgxState, /) -> _AmgxState:
        setup_state.plan = setup_state.backend.plan(
            setup_state.native_problem,
            setup_state.policy,
        )
        return setup_state

    def preparation_applicable(self, compiled_state: _AmgxState, /) -> bool:
        return True

    def prepare(self, compiled_state: _AmgxState, /) -> _AmgxState:
        compiled_state.prepared = compiled_state.backend.prepare(
            compiled_state.native_problem,
            compiled_state.plan,
        )
        compiled_state.preparation_host_to_device_bytes += int(
            compiled_state.prepared.transfers.host_to_device_bytes
        )
        return compiled_state

    def solve(self, prepared_state: _AmgxState, /) -> SolveResult:
        jnp = import_module("jax.numpy")
        problem = prepared_state.spec.problem
        if not isinstance(problem, SparseLinearProblem):
            raise TypeError("AmgX solve requires a sparse linear benchmark problem.")
        result = prepared_state.backend.solve(
            prepared_state.prepared,
            prepared_state.rhs,
        )
        prepared_state.solve_host_to_device_bytes = int(
            result.transfers.host_to_device_bytes
        )
        prepared_state.solve_device_to_host_bytes = int(
            result.transfers.device_to_host_bytes
        )
        prepared_state.jax_result_device_bytes = _jax_array_bytes(result)
        iterations = result.diagnostics.iterations
        return SolveResult(
            solution=result.value,
            auxiliary={
                "status_codes": result.status,
                "provider_reasons": result.diagnostics.provider_reasons,
                "numeric_version": result.provenance.numeric_version,
            },
            converged=jnp.all(result.success),
            message="Phydrax public AmgX solve completed with independent backend residual evidence",
            operations={
                "iterations": None if iterations is None else jnp.max(iterations),
                "matvecs": None,
                "preconditioner_applications": None,
                "linear_solves": (
                    1 if problem.rhs.ndim == 1 else int(problem.rhs.shape[1])
                ),
                "nonlinear_evaluations": 0,
                "jacobian_evaluations": 0,
            },
        )

    def refresh_applicable(self, prepared_state: _AmgxState, /) -> bool:
        return True

    def refresh(
        self,
        prepared_state: _AmgxState,
        /,
    ) -> tuple[_AmgxState, RefreshEvidence]:
        problem = prepared_state.spec.problem
        if not isinstance(problem, SparseLinearProblem):
            raise TypeError("AmgX refresh requires a sparse linear benchmark problem.")
        refreshed_problem = replace(problem, coefficients=problem.coefficients * 1.01)
        jnp = import_module("jax.numpy")
        refreshed_native = _native_linear_problem(
            prepared_state.phx,
            jnp,
            refreshed_problem,
        )
        previous = prepared_state.prepared
        refreshed = prepared_state.backend.refresh(previous, refreshed_native)
        prepared_state.backend.release(previous)
        prepared_state.native_problem = refreshed_native
        prepared_state.prepared = refreshed
        prepared_state.refreshed_certificate_problem = refreshed_problem
        prepared_state.preparation_host_to_device_bytes += int(
            refreshed.transfers.host_to_device_bytes
        )
        return prepared_state, RefreshEvidence(
            applicable=True,
            symbolic_reused=True,
            numeric_refreshed=True,
            symbolic_refresh_count=0,
            numeric_refresh_count=1,
            evidence=(
                "Phydrax public AmgX refresh preserved the canonical CSR pattern and "
                "plan while rebinding a deterministic 1% numeric perturbation; the "
                "previous explicit AmgX preparation was released"
            ),
        )

    def certificate_problem(self, prepared_state: _AmgxState, /) -> Any:
        return (
            prepared_state.spec.problem
            if prepared_state.refreshed_certificate_problem is None
            else prepared_state.refreshed_certificate_problem
        )

    def memory(
        self,
        prepared_state: _AmgxState,
        result: SolveResult,
        /,
    ) -> dict[str, Any]:
        del result
        storage = prepared_state.native_problem.operator.sparse_storage()
        matrix_bytes = int(
            storage.values.nbytes + storage.indices.nbytes + storage.indptr.nbytes
        )
        return {
            "matrix_bytes": matrix_bytes,
            "setup_bytes": None,
            "peak_estimate_bytes": None,
            "evidence": (
                "exact canonical CSR storage bytes; the public AmgX provider does not "
                "expose hierarchy, Krylov workspace, or allocator peak bytes"
            ),
        }

    def transfers(
        self,
        prepared_state: _AmgxState,
        result: SolveResult,
        /,
        *,
        device_to_host_bytes: int,
    ) -> TransferEvidence:
        del result
        host_to_device = (
            prepared_state.preparation_host_to_device_bytes
            + prepared_state.solve_host_to_device_bytes
            + prepared_state.jax_result_device_bytes
        )
        device_to_host = prepared_state.solve_device_to_host_bytes + device_to_host_bytes
        return TransferEvidence(
            input_origin="numpy-host",
            host_to_device_bytes=host_to_device,
            host_to_device_timing_phase=("preparation+solve+refresh+refreshed_solve"),
            device_to_host_bytes=device_to_host,
            device_to_host_timing_phase=(
                "solve+refreshed_solve+verification+refreshed_verification"
            ),
            evidence=(
                "exact public-backend matrix, RHS, initial-vector, and result transfer "
                "bytes accumulated across preparation, solve, numeric refresh, "
                "refreshed solve, verification, and refreshed verification; '+' "
                "separates measured phases"
            ),
        )

    def release(self, prepared_state: _AmgxState, /) -> None:
        if prepared_state.prepared is not None:
            prepared_state.backend.release(prepared_state.prepared)


def _native_linear_problem(phx: Any, jnp: Any, problem: SparseLinearProblem, /) -> Any:
    properties = phx.linalg.OperatorProperties(
        self_adjoint=True,
        positive_definite=True,
        evidence={
            "self_adjoint": "benchmark construction",
            "positive_definite": "benchmark construction",
            "positive_semidefinite": "benchmark construction",
        },
    )
    relation = phx.sparse.EdgeRelation(
        jnp.asarray(problem.columns),
        jnp.asarray(problem.rows),
        source_size=problem.dimension,
        target_size=problem.dimension,
    )
    space = phx.linalg.ArraySpace((problem.dimension,), dtype=jnp.float64)
    operator = phx.sparse.SparseCoordinateOperator(
        relation,
        jnp.asarray(problem.coefficients),
        source=space,
        target=space,
        properties=properties,
        operator_id=f"benchmark-amgx:{problem.identity()['fingerprint']}",
    )
    return phx.linalg.LinearSystem(
        operator,
        problem_id=f"benchmark-amgx-system:{problem.identity()['fingerprint']}",
    )


def _jax_array_bytes(value: Any, /) -> int:
    jax = import_module("jax")
    total = 0
    seen: set[int] = set()
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array) and id(leaf) not in seen:
            seen.add(id(leaf))
            total += int(leaf.size * leaf.dtype.itemsize)
    return total


def _version_evidence() -> dict[str, str]:
    versions = {"phydrax_source_sha256": _phydrax_amg_source_digest()}
    for distribution in ("phydrax", "pyamgx", "jax"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _phydrax_amg_source_digest() -> str:
    module_spec = importlib.util.find_spec("phydrax.backends.amg")
    if module_spec is None or module_spec.origin is None:
        return "unavailable"
    source = Path(module_spec.origin)
    if not source.is_file():
        return "unavailable"
    return hashlib.sha256(source.read_bytes()).hexdigest()


__all__ = ["AmgxAdapter"]
