#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated grounded-solve effective resistance on nonperiodic square grids.

Run ``python -m benchmarks.effective_resistance_scaling --sizes 32 --queries 8``
for a smoke case. Defaults sweep sides 64, 128, 256, and 512 with 8 and 32
query nodes, comparing PCG with Jacobi and automatic method selection against
an independent SciPy grounded sparse LU reference. Evidence is written beside
this script unless ``--output`` selects another destination.

Compiler memory estimates and retained logical array bytes are not measured
process peak memory. Forward-plus-backward differentiates the mean of the full
query resistance matrix with respect to every edge conductance.

Retained records must be regenerated after source/driver changes. The shared
stored-record validator owns last-step evidence, not this scaling layout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from phydrax.graph import EffectiveResistancePlan, EffectiveResistanceResult
from phydrax.linalg import (
    JacobiPreconditionerBuilder,
    LinearSolvePolicy,
    PCG,
    PreconditioningPolicy,
)
from phydrax.topology import grid_adjacency_relation

from ._io import write_json_atomic
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    synchronize,
)


EVIDENCE_FIELDS = (
    "environment",
    "command",
    "configuration",
    "dtype",
    "repeats",
    "objective",
    "nonfinite_evidence",
    "cases",
)


def _forward(
    plan: EffectiveResistancePlan, conductances: Array, /
) -> EffectiveResistanceResult:
    return plan.solve(conductances)


def _objective(plan: EffectiveResistancePlan, conductances: Array, /) -> Array:
    return jnp.mean(plan.solve(conductances).resistance)


class _Compiled[Result](Protocol):
    compiled: jax.stages.Compiled

    def __call__(
        self, plan: EffectiveResistancePlan, conductances: Array, /
    ) -> Result: ...


class _Lowered[Result](Protocol):
    def compile(self) -> _Compiled[Result]: ...


@runtime_checkable
class _Entry[Result](Protocol):
    def __call__(
        self, plan: EffectiveResistancePlan, conductances: Array, /
    ) -> Result: ...

    def lower(
        self, plan: EffectiveResistancePlan, conductances: Array, /
    ) -> _Lowered[Result]: ...


def _require_entry[Result](
    entry: Callable[[EffectiveResistancePlan, Array], Result], /
) -> _Entry[Result]:
    if not isinstance(entry, _Entry):
        raise TypeError("Resistance compilation requires a callable lowering entry.")
    return entry


_FORWARD = _require_entry(eqx.filter_jit(_forward))
_BACKWARD = _require_entry(eqx.filter_jit(jax.value_and_grad(_objective, argnums=1)))


def _measure_entry[Result](
    entry: _Entry[Result],
    plan: EffectiveResistancePlan,
    conductances: Array,
    repeats: int,
    /,
) -> tuple[Result, dict[str, Any]]:
    executable, timing = measure_lower_and_compile(
        lambda: entry.lower(plan, conductances), lambda lowered: lowered.compile()
    )
    result, warm = measure_repeated(
        lambda: executable(plan, conductances), warmup=1, repeats=repeats
    )
    evidence = compiler_evidence(
        executable.compiled.cost_analysis(),
        executable.compiled.memory_analysis(),
        source="jax-compiled-grounded-effective-resistance",
        unavailable_reason="The compiler did not supply all cost or memory estimates.",
    )
    return result, {
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(evidence),
        "retained_output_bytes": logical_array_bytes(result),
    }


def _grounded_laplacian(
    source: NDArray[np.int32],
    target: NDArray[np.int32],
    conductances: NDArray[np.float64],
    node_count: int,
    /,
) -> csc_matrix:
    rows = np.concatenate((source, target, source, target))
    columns = np.concatenate((source, target, target, source))
    values = np.concatenate((conductances, conductances, -conductances, -conductances))
    laplacian = coo_matrix(
        (values, (rows, columns)), shape=(node_count, node_count), dtype=np.float64
    ).tocsc()
    # The connected grid's minimum-index node is the same zero ground used by
    # the native plan. No dense node-by-node inverse is constructed.
    return laplacian[1:, 1:].tocsc()


def _scipy_reference(
    source: NDArray[np.int32],
    target: NDArray[np.int32],
    conductances: NDArray[np.float64],
    node_count: int,
    nodes: NDArray[np.int32],
    repeats: int,
    /,
) -> tuple[NDArray[np.float64], NDArray[np.float64], dict[str, Any]]:
    matrix, assembly_seconds = measure_host(
        lambda: _grounded_laplacian(source, target, conductances, node_count)
    )
    factor, factorization_seconds = measure_host(lambda: splu(matrix))
    rhs = np.zeros((node_count - 1, nodes.size), dtype=np.float64)
    non_ground = nodes != 0
    rhs[nodes[non_ground] - 1, np.flatnonzero(non_ground)] = 1.0

    def solve() -> NDArray[np.float64]:
        return factor.solve(rhs)

    reduced, warm = measure_repeated(solve, warmup=1, repeats=repeats)
    potential = np.zeros((node_count, nodes.size), dtype=np.float64)
    potential[1:] = reduced
    kernel = potential[nodes]
    diagonal = np.diag(kernel)
    resistance = diagonal[:, None] + diagonal[None, :] - kernel - kernel.T
    # d mean(R_ij)/dw_e = -mean((phi_i(e)-phi_j(e))**2).
    # Centering avoids materializing an edge-by-query-by-query tensor.
    drop = potential[source] - potential[target]
    centered = drop - np.mean(drop, axis=1, keepdims=True)
    gradient = -2.0 * np.mean(centered * centered, axis=1)
    residual = matrix @ reduced - rhs
    factors = (factor.L, factor.U)
    factor_bytes = (
        sum(
            sparse.data.nbytes + sparse.indices.nbytes + sparse.indptr.nbytes
            for sparse in factors
        )
        + factor.perm_r.nbytes
        + factor.perm_c.nbytes
    )
    return (
        resistance,
        gradient,
        {
            "method": "scipy-grounded-splu",
            "ground_node": 0,
            "assembly_seconds": assembly_seconds,
            "factorization_seconds": factorization_seconds,
            "warm_solve": warm.to_seconds_dict(),
            "residual_norm": float(np.linalg.norm(residual)),
            "retained_matrix_bytes": (
                matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
            ),
            "retained_factor_bytes": factor_bytes,
            "retained_rhs_bytes": rhs.nbytes,
            "retained_potential_bytes": potential.nbytes,
            "retained_resistance_bytes": resistance.nbytes,
            "factor_nonzeros": sum(sparse.nnz for sparse in factors),
        },
    )


def _finite(value: float, /) -> float | None:
    return value if np.isfinite(value) else None


def _measure_case(
    side: int, query_count: int, repeats: int, workspace_bytes: int, /
) -> list[dict[str, Any]]:
    node_count = side * side
    relation, topology_seconds = measure_host(
        lambda: grid_adjacency_relation((side, side), (False, False))
    )
    synchronize(relation)
    source = np.asarray(jax.device_get(relation.source_indices), dtype=np.int32)
    target = np.asarray(jax.device_get(relation.target_indices), dtype=np.int32)
    nodes = np.linspace(0, node_count - 1, query_count, dtype=np.int32)
    host_conductances = np.ones(source.size, dtype=np.float64)
    conductances = jnp.asarray(host_conductances, dtype=jnp.float64)
    synchronize(conductances)
    reference, reference_gradient, reference_evidence = _scipy_reference(
        source, target, host_conductances, node_count, nodes, repeats
    )
    policies = (
        (
            "pcg-jacobi",
            LinearSolvePolicy(
                PCG(),
                preconditioning=PreconditioningPolicy(JacobiPreconditionerBuilder()),
            ),
        ),
        ("auto", LinearSolvePolicy()),
    )
    rows: list[dict[str, Any]] = []
    for name, policy in policies:
        plan, preparation_seconds = measure_host(
            lambda: EffectiveResistancePlan(
                relation,
                nodes,
                policy=policy,
                dtype=jnp.float64,
                workspace_bytes=workspace_bytes,
            )
        )
        # Preparation timing is host wall time; any asynchronous placement is
        # drained before lowering or steady execution is measured.
        synchronize(plan)
        result, forward = _measure_entry(_FORWARD, plan, conductances, repeats)
        (objective, gradient), backward = _measure_entry(
            _BACKWARD, plan, conductances, repeats
        )
        resistance = np.asarray(jax.device_get(result.resistance), dtype=np.float64)
        host_gradient = np.asarray(jax.device_get(gradient), dtype=np.float64)
        rows.append(
            {
                "grid_side": side,
                "node_count": node_count,
                "edge_count": source.size,
                "query_count": query_count,
                "query_nodes": nodes.tolist(),
                "policy": name,
                "plan_id": plan.plan_id,
                "block_size": plan.block_size,
                "resource_estimate": plan.resource_estimate,
                "workspace_bytes": workspace_bytes,
                "topology_build_seconds": topology_seconds,
                "host_plan_build_seconds": preparation_seconds,
                "retained_relation_bytes": logical_array_bytes(relation),
                "retained_plan_bytes": logical_array_bytes(plan),
                "retained_conductance_bytes": conductances.nbytes,
                "forward": forward,
                "forward_backward": backward,
                "status": np.asarray(jax.device_get(result.status)).tolist(),
                "solve_status": np.asarray(jax.device_get(result.solve_status)).tolist(),
                "residual_norm": [
                    _finite(float(value))
                    for value in np.asarray(jax.device_get(result.residual_norm)).reshape(
                        -1
                    )
                ],
                "iterations": np.asarray(jax.device_get(result.iterations)).tolist(),
                "connected": np.asarray(jax.device_get(result.connected)).tolist(),
                "derivative_regular": np.asarray(
                    jax.device_get(result.derivative_regular)
                ).tolist(),
                "mean_resistance": _finite(float(objective)),
                "reference_max_abs_error": _finite(
                    float(np.max(np.abs(resistance - reference)))
                ),
                "reference_gradient_max_abs_error": _finite(
                    float(np.max(np.abs(host_gradient - reference_gradient)))
                ),
                "scipy_reference": reference_evidence,
            }
        )
        # Independent cases do not retain earlier executables in JAX's caches.
        jax.clear_caches()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=(64, 128, 256, 512))
    parser.add_argument("--queries", type=int, nargs="+", default=(8, 32))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--workspace-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).with_suffix(".json")
    )
    arguments = parser.parse_args()
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, EVIDENCE_FIELDS)
    if any(side < 2 for side in arguments.sizes):
        raise ValueError("Grid sides must be at least two.")
    if any(count < 2 for count in arguments.queries):
        raise ValueError("Query counts must be at least two.")
    if any(
        count > side * side for side in arguments.sizes for count in arguments.queries
    ):
        raise ValueError("Query count must not exceed the grid node count.")
    if arguments.repeats < 1 or arguments.workspace_bytes < 1:
        raise ValueError("Repeats and workspace bytes must be positive.")
    jax.config.update("jax_enable_x64", True)
    rows = [
        row
        for side in sorted(set(arguments.sizes))
        for count in sorted(set(arguments.queries))
        for row in _measure_case(
            side, count, arguments.repeats, arguments.workspace_bytes
        )
    ]
    record = {
        "environment": capture_environment().to_dict(),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "driver": str(driver),
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "import_path": list(sys.path),
            "mode": "full"
            if tuple(sorted(set(arguments.sizes))) == (64, 128, 256, 512)
            and tuple(sorted(set(arguments.queries))) == (8, 32)
            else "smoke"
            if max(arguments.sizes) <= 32
            else "custom",
            "sides": sorted(set(arguments.sizes)),
            "queries": sorted(set(arguments.queries)),
            "repeats": arguments.repeats,
            "warmups": 1,
            "policies": ["pcg-jacobi", "auto"],
            "backend": jax.default_backend(),
            "reference_provider": "scipy-sparse-superlu",
            "workspace_bytes": arguments.workspace_bytes,
            "output": str(arguments.output.resolve()),
        },
        "dtype": "float64",
        "repeats": arguments.repeats,
        "objective": "mean of the full query resistance matrix",
        "nonfinite_evidence": "Nonfinite scalar residuals or errors are serialized as null.",
        "cases": rows,
    }
    record["identity"] = identity.to_dict()
    write_json_atomic(arguments.output, record)
    print(json.dumps(record, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
