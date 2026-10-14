"""CPU distance-field scaling with explicit host-provider and device phases.

Run --sides 8 for smoke; larger grids are an explicit benchmark invocation.
Retained records must be regenerated after source/driver changes. The shared
stored-record validator owns last-step evidence, not this scaling layout.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
)
from phydrax import config as pxconfig
from phydrax.combinatorial import ShortestPathFieldPlan
from phydrax.sparse import EdgeRelation


MAXIMUM_WORK = 100_000_000
MAXIMUM_HOST_BYTES = 256 * 1024**2
EVIDENCE_FIELDS = (
    "backend",
    "environment",
    "command",
    "configuration",
    "nonfinite_evidence",
    "records",
)


def measure(side: int, repeats: int) -> dict[str, Any]:
    vertices = side * side
    ids = np.arange(vertices, dtype=np.int32).reshape(side, side)
    edges = np.concatenate(
        (
            np.stack((ids[:, :-1].ravel(), ids[:, 1:].ravel()), 1),
            np.stack((ids[:-1].ravel(), ids[1:].ravel()), 1),
        )
    )
    routes = np.concatenate((edges, edges[:, ::-1]))
    source, target = routes.T
    relation = EdgeRelation(source, target, source_size=vertices, target_size=vertices)
    plan = ShortestPathFieldPlan(
        relation,
        maximum_work=MAXIMUM_WORK,
        maximum_host_bytes=MAXIMUM_HOST_BYTES,
    )
    costs = jnp.full(len(routes), 0.5, jnp.float64)
    sources = jnp.asarray([0], jnp.int32)
    run = jax.jit(lambda c: plan.solve(c, sources))
    start = time.perf_counter()
    lowered = run.lower(costs)
    lowering = time.perf_counter() - start
    start = time.perf_counter()
    compiled = lowered.compile()
    compilation = time.perf_counter() - start
    result = compiled(costs)
    result.distance.block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        compiled(costs).distance.block_until_ready()
    warm = (time.perf_counter() - start) / repeats
    memory = compiled.memory_analysis()
    memory_record = (
        None
        if memory is None
        else {
            "argument_bytes": memory.argument_size_in_bytes,
            "output_bytes": memory.output_size_in_bytes,
            "temporary_bytes": memory.temp_size_in_bytes,
            "alias_bytes": memory.alias_size_in_bytes,
            "host_workspace_note": "SciPy callback allocations are not included in device compiler memory analysis.",
        }
    )
    matrix = csr_matrix(
        (np.full(len(routes), 0.5), (source, target)), shape=(vertices, vertices)
    )
    start = time.perf_counter()
    expected = dijkstra(matrix, indices=0)
    scipy_s = time.perf_counter() - start
    u, v = jnp.asarray(source), jnp.asarray(target)

    def baseline(c: jax.Array) -> jax.Array:
        initial = jnp.full(vertices, jnp.inf, jnp.float64).at[0].set(0.0)

        def sweep(_: int, d: jax.Array) -> jax.Array:
            return d.at[v].min(d[u] + c)

        return jax.lax.fori_loop(0, vertices - 1, sweep, initial)

    fixed = jax.jit(baseline)
    start = time.perf_counter()
    baseline_lowered = fixed.lower(costs)
    baseline_lowering = time.perf_counter() - start
    start = time.perf_counter()
    baseline_compiled = baseline_lowered.compile()
    baseline_compile = time.perf_counter() - start
    baseline_compiled(costs).block_until_ready()
    start = time.perf_counter()
    for _ in range(repeats):
        baseline_compiled(costs).block_until_ready()
    baseline_warm = (time.perf_counter() - start) / repeats
    return {
        "side": side,
        "vertices": vertices,
        "routes": len(routes),
        "method": result.method,
        "host_synchronization": result.host_synchronization,
        "lowering_s": lowering,
        "compile_s": compilation,
        "warm_execution_s": warm,
        "memory_analysis": memory_record,
        "scipy_dijkstra_s": scipy_s,
        "baseline_lowering_s": baseline_lowering,
        "baseline_compile_s": baseline_compile,
        "baseline_warm_execution_s": baseline_warm,
        "baseline_route_work": (vertices - 1) * len(routes),
        "provider_route_work": int(result.work),
        "maximum_error": _finite(
            float(np.max(np.abs(np.asarray(result.distance) - expected)))
        ),
        "status": int(result.status),
        "bellman_residual": _finite(float(result.bellman_residual)),
        "dual_residual": _finite(float(result.dual_residual)),
    }


def _finite(value: float, /) -> float | None:
    return value if np.isfinite(value) else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sides", type=int, nargs="+", default=[8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/shortest_path_field_scaling.json")
    )
    args = parser.parse_args()
    driver = Path(__file__).resolve()
    identity = capture_benchmark_identity(driver.parents[1], driver, EVIDENCE_FIELDS)
    pxconfig.configure(pxconfig.RuntimeConfig(enable_x64=True))
    if args.repeats < 1 or any(side < 2 for side in args.sides):
        raise ValueError("Sides must be at least two and repeats positive.")
    sides = sorted(set(args.sides))
    records = [measure(side, args.repeats) for side in sides]
    record = {
        "identity": identity.to_dict(),
        "backend": jax.default_backend(),
        "environment": capture_environment().to_dict(),
        "command": [sys.executable, *sys.argv],
        "configuration": {
            "driver": str(driver),
            "cwd": str(Path.cwd()),
            "pythonpath": os.environ.get("PYTHONPATH"),
            "import_path": list(sys.path),
            "mode": "smoke" if max(sides) <= 8 else "full",
            "sides": sides,
            "queries": [0],
            "repeats": args.repeats,
            "warmups": 1,
            "dtype": "float64",
            "edge_cost": 0.5,
            "policy": "default",
            "provider": "scipy-csgraph-dijkstra",
            "resource_policy": {
                "maximum_work": MAXIMUM_WORK,
                "maximum_host_bytes": MAXIMUM_HOST_BYTES,
            },
            "output": str(args.output.resolve()),
        },
        "records": records,
        "nonfinite_evidence": "Nonfinite scalar residuals or errors are serialized as null.",
    }
    write_json_atomic(args.output, record)


if __name__ == "__main__":
    main()
