# Copyright © 2026 PHYDRA, Inc. All rights reserved.
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array

from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)
from phydrax.discretization import (
    StructuredCochainBridge,
    TensorGridPlan,
    UniformCellAxisSpec,
)
from phydrax.sparse import SparseLinearMap


@eqx.filter_jit
def _stencil_d(bridge: StructuredCochainBridge, degree: int, value: Array, /) -> Array:
    return bridge.exterior_derivative(degree, value)


@eqx.filter_jit
def _route_d(incidence: SparseLinearMap, value: Array, /) -> Array:
    return incidence.mv(value)


@eqx.filter_jit
def _stencil_delta(
    bridge: StructuredCochainBridge, degree: int, value: Array, /
) -> Array:
    return bridge.codifferential(degree, value)


@eqx.filter_jit
def _route_delta(
    incidence: SparseLinearMap, source_mass: Array, target_mass: Array, value: Array, /
) -> Array:
    return incidence.transpose_mv(target_mass * value) / source_mass


def prepare_bridge(
    count: int, dimension: int, periodic: bool, /
) -> StructuredCochainBridge:
    grid = TensorGridPlan(
        tuple(UniformCellAxisSpec(count, periodic=periodic) for _ in range(dimension)),
        axis_names=tuple(f"axis{axis}" for axis in range(dimension)),
    ).prepare(jnp.asarray([[0.0] * dimension, [1.0] * dimension], dtype=jnp.float64))
    return StructuredCochainBridge(grid)


def campaign(
    count: int, dimension: int, periodic: bool, repeats: int, /
) -> dict[str, Any]:
    bridge, preparation_seconds = measure_synchronized(
        lambda: prepare_bridge(count, dimension, periodic)
    )
    actions = []
    for degree in range(dimension):
        incidence = bridge.topology.incidences[degree].exterior_derivative()
        source = jnp.sin(jnp.arange(bridge.cell_counts[degree], dtype=jnp.float64))
        target = jnp.cos(jnp.arange(bridge.cell_counts[degree + 1], dtype=jnp.float64))
        source_mass = bridge.cochain.hodge_diagonal(degree)
        target_mass = bridge.cochain.hodge_diagonal(degree + 1)

        full_stencil = bridge.cochain.hilbert_complex().differential(degree)
        numerical = (
            ("d_stencil", incidence.mv(source)),
            ("d_routes", full_stencil.mv(source)),
            (
                "delta_stencil",
                incidence.transpose_mv(target_mass * target) / source_mass,
            ),
            ("delta_routes", bridge.codifferential(degree + 1, target)),
        )
        for name, reference in numerical:
            # Keep each signature explicit and numerical PyTree leaves dynamic.
            if name == "d_stencil":
                # Equinox annotates filter_jit as Callable, omitting runtime lower().
                compiled_d_stencil, timing = measure_lower_and_compile(
                    lambda: _stencil_d.lower(  # ty: ignore[unresolved-attribute]
                        bridge, degree, source
                    ),
                    lambda lowered: lowered.compile(),
                )
                result, warm = measure_repeated(
                    lambda: compiled_d_stencil(bridge, degree, source),
                    warmup=1,
                    repeats=repeats,
                )
                executable = compiled_d_stencil.compiled
            elif name == "d_routes":
                compiled_d_routes, timing = measure_lower_and_compile(
                    lambda: _route_d.lower(  # ty: ignore[unresolved-attribute]
                        incidence, source
                    ),
                    lambda lowered: lowered.compile(),
                )
                result, warm = measure_repeated(
                    lambda: compiled_d_routes(incidence, source),
                    warmup=1,
                    repeats=repeats,
                )
                executable = compiled_d_routes.compiled
            elif name == "delta_stencil":
                compiled_delta_stencil, timing = measure_lower_and_compile(
                    lambda: _stencil_delta.lower(  # ty: ignore[unresolved-attribute]
                        bridge, degree + 1, target
                    ),
                    lambda lowered: lowered.compile(),
                )
                result, warm = measure_repeated(
                    lambda: compiled_delta_stencil(bridge, degree + 1, target),
                    warmup=1,
                    repeats=repeats,
                )
                executable = compiled_delta_stencil.compiled
            else:
                compiled_delta_routes, timing = measure_lower_and_compile(
                    lambda: _route_delta.lower(  # ty: ignore[unresolved-attribute]
                        incidence, source_mass, target_mass, target
                    ),
                    lambda lowered: lowered.compile(),
                )
                result, warm = measure_repeated(
                    lambda: compiled_delta_routes(
                        incidence, source_mass, target_mass, target
                    ),
                    warmup=1,
                    repeats=repeats,
                )
                executable = compiled_delta_routes.compiled
            error = float(jnp.max(jnp.abs(result - reference)))
            np.testing.assert_allclose(result, reference, rtol=1e-12, atol=1e-12)
            evidence = compiler_evidence(
                executable.cost_analysis(),
                executable.memory_analysis(),
                source="jax-compiled",
            )
            actions.append(
                {
                    "degree": degree,
                    "action": name,
                    "maximum_error": error,
                    "compilation": asdict(timing),
                    "warm": warm.to_seconds_dict(),
                    "compiler": asdict(evidence),
                }
            )
        directional_sum = sum(
            (
                operator.mv(source)
                for operator in bridge.directional_differentials[degree]
            ),
            jnp.zeros((bridge.cell_counts[degree + 1],), dtype=source.dtype),
        )
        np.testing.assert_allclose(directional_sum, full_stencil.mv(source), atol=1e-12)
    return {
        "count_per_axis": count,
        "dimension": dimension,
        "periodic": periodic,
        "entity_counts": bridge.entity_counts,
        "incidence_routes": bridge.incidence_route_count,
        "estimated_preparation_bytes": bridge.preparation_bytes,
        "retained_bytes": logical_array_bytes(bridge),
        "preparation_seconds": preparation_seconds,
        "actions": actions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--counts", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/structured_cochain.json")
    )
    options = parser.parse_args()
    rows = [
        campaign(count, options.dimension, periodic, options.repeats)
        for count in options.counts
        for periodic in (False, True)
    ]
    root = Path(__file__).resolve().parents[1]
    identity = capture_benchmark_identity(root, Path(__file__), rows[0].keys())
    write_json_atomic(
        options.output,
        {
            "identity": identity.to_dict(),
            "environment": capture_environment().to_dict(),
            "cases": rows,
        },
    )


if __name__ == "__main__":
    main()
