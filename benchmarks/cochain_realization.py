#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Run the bounded metric cochain campaign with --output for persistent evidence."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization._cell_complex import interval_cell_complex
from phydrax.discretization._cochain import CochainDiscretization
from phydrax.discretization._cochain_hodge import CochainHodge, DiagonalHodge, SparseHodge
from phydrax.exterior._complex import ComplexBoundary
from phydrax.graph import cochain_exterior_derivative, CochainComplexIR

from ._io import write_json_atomic
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _metric(size: int, sparse: bool, /) -> CochainHodge:
    diagonal = 2.0 + np.arange(size, dtype=np.float64) / max(size, 1)
    if not sparse:
        return DiagonalHodge(diagonal)
    indices = np.arange(size, dtype=np.int32)
    rows = np.concatenate((indices, indices[:-1]))
    columns = np.concatenate((indices, indices[1:]))
    entries = np.concatenate(
        (diagonal, np.full(max(0, size - 1), -0.2, dtype=np.float64))
    )
    return SparseHodge(rows, columns, entries, size)


def _prepare(capacity: int, sparse: bool, /) -> CochainDiscretization:
    sites = np.arange(capacity + 1, dtype=np.int32)
    topology = interval_cell_complex(
        np.stack((sites[:-1], sites[1:]), axis=1), capacity + 1
    )
    hodge0 = _metric(capacity + 1, sparse)
    hodge1 = _metric(capacity, sparse)
    boundary0 = np.zeros(capacity + 1, dtype=np.bool_)
    boundary0[[0, -1]] = True
    return CochainDiscretization(
        topology,
        (hodge0, hodge1),
        boundary_masks=(boundary0, np.zeros(capacity, dtype=np.bool_)),
        numeric_revision=f"benchmark-cochain:{capacity}:{sparse}",
    )


def _reference(
    capacity: int, sparse: bool, boundary: ComplexBoundary, values: np.ndarray, /
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gram0 = np.diag(2.0 + np.arange(capacity + 1, dtype=np.float64) / (capacity + 1))
    gram1 = np.diag(2.0 + np.arange(capacity, dtype=np.float64) / capacity)
    if sparse:
        off0 = np.full(capacity, -0.2, dtype=np.float64)
        off1 = np.full(capacity - 1, -0.2, dtype=np.float64)
        gram0 += np.diag(off0, 1) + np.diag(off0, -1)
        gram1 += np.diag(off1, 1) + np.diag(off1, -1)
    differential = np.zeros((capacity, capacity + 1), dtype=np.float64)
    indices = np.arange(capacity)
    differential[indices, indices] = -1.0
    differential[indices, indices + 1] = 1.0
    active0 = (
        np.arange(capacity + 1) if boundary == "absolute" else np.arange(1, capacity)
    )
    restricted = differential[:, active0]
    delta_active = np.linalg.solve(
        gram0[np.ix_(active0, active0)], restricted.T @ gram1 @ values
    )
    delta = np.zeros(capacity + 1, dtype=np.float64)
    delta[active0] = delta_active
    return gram1 @ values, delta, restricted @ delta_active


def _graph_evidence(
    realization: CochainDiscretization, boundary: ComplexBoundary, repeats: int, /
) -> dict[str, Any]:
    lowered, lowering = measure_host(
        lambda: CochainComplexIR(realization, boundary=boundary)
    )
    primal = jnp.cos(0.19 * jnp.arange(realization.cell_counts[0], dtype=jnp.float64))
    packed = (
        jnp.zeros((lowered.num_cells,), dtype=primal.dtype)
        .at[lowered.cell_entities(0)]
        .set(primal)
    )

    def action(values: Array) -> Array:
        return cochain_exterior_derivative(lowered.graph, values, 0, boundary=boundary)

    function = jax.jit(action)
    compiled, timing = measure_lower_and_compile(
        lambda: function.lower(packed), lambda lowering: lowering.compile()
    )
    _, cold = measure_synchronized(lambda: compiled(packed))
    result, warm = measure_repeated(lambda: compiled(packed), warmup=1, repeats=repeats)
    potential = np.asarray(primal).copy()
    if boundary == "relative":
        potential[[0, -1]] = 0.0
    expected = np.diff(potential)
    actual = np.asarray(result)[np.asarray(lowered.cell_entities(1))]
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-cochain-graph-lowering",
    )
    return {
        "host_lowering_seconds": lowering,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "cold_seconds": cold,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
        "retained_preparation_bytes": logical_array_bytes(lowered),
        "derivative_defect": float(np.max(np.abs(actual - expected))),
    }


def _measure(
    capacity: int, sparse: bool, boundary: ComplexBoundary, repeats: int, /
) -> dict[str, Any]:
    realization, preparation_seconds = measure_synchronized(
        lambda: _prepare(capacity, sparse)
    )
    values = jnp.sin(0.17 * jnp.arange(capacity, dtype=jnp.float64))

    def action(vector: Array) -> tuple[Array, Array, Array]:
        return (
            realization.hodge_star(1, vector),
            realization.codifferential(1, vector, boundary=boundary),
            realization.hodge_laplacian(1, vector, boundary=boundary),
        )

    function = jax.jit(action)
    compiled, timing = measure_lower_and_compile(
        lambda: function.lower(values), lambda lowered: lowered.compile()
    )
    _, cold_seconds = measure_synchronized(lambda: compiled(values))
    actual, warm = measure_repeated(lambda: compiled(values), warmup=1, repeats=repeats)
    expected = _reference(capacity, sparse, boundary, np.asarray(values))
    defects = []
    for result, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(np.asarray(result), reference, rtol=2e-9, atol=2e-10)
        defects.append(float(np.max(np.abs(np.asarray(result) - reference))))
    compiler = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-cochain-realization",
    )
    graph = None if sparse else _graph_evidence(realization, boundary, repeats)
    return {
        "capacity": capacity,
        "metric": "sparse" if sparse else "diagonal",
        "boundary": boundary,
        "preparation_seconds": preparation_seconds,
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "cold_seconds": cold_seconds,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(compiler),
        "retained_preparation_bytes": logical_array_bytes(realization),
        "graph_lowering": graph,
        "last_step_evidence": {
            "hodge_defect": defects[0],
            "codifferential_defect": defects[1],
            "laplacian_defect": defects[2],
            "graph_derivative_defect": None
            if graph is None
            else graph["derivative_defect"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--capacities", type=int, nargs="+", default=(16, 64, 256))
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    if any(capacity < 2 for capacity in arguments.capacities):
        raise ValueError("Benchmark capacities must be at least two.")
    boundaries: tuple[ComplexBoundary, ...] = ("absolute", "relative")
    rows = [
        _measure(capacity, sparse, boundary, arguments.repeats)
        for capacity in arguments.capacities
        for sparse in (False, True)
        for boundary in boundaries
    ]
    root = Path(__file__).resolve().parent.parent
    identity = capture_benchmark_identity(
        root, Path(__file__), rows[0]["last_step_evidence"]
    )
    record = {
        "benchmark_identity": identity.to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": rows,
    }
    if arguments.output is not None:
        write_json_atomic(arguments.output, record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
