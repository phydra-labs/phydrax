#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)
from phydrax._bvh import BVHBuildPolicy, point_select_leaf_items, prepare_bvh
from phydrax.discretization.spatial import (
    BVHNeighborQueryPlan,
    BVHSourcePlan,
    MortonAddressPlan,
    MortonNeighborQueryPlan,
    MortonSourcePlan,
)
from phydrax.graph import query_neighbors, QueryNeighborhood


def _case(node_count: int, query_count: int, repeats: int) -> dict[str, object]:
    item_count = max(1, (node_count + 1) // 2)
    lower = jnp.stack(
        (
            jnp.arange(item_count, dtype="float64"),
            jnp.zeros(item_count),
            jnp.zeros(item_count),
        ),
        axis=-1,
    )
    upper = lower + jnp.asarray((0.75, 1.0, 1.0))
    bvh = prepare_bvh(lower, upper, policy=BVHBuildPolicy(leaf_size=2))
    points = jnp.stack(
        (
            jnp.linspace(0.25, item_count - 0.25, query_count),
            jnp.full((query_count,), 0.5),
            jnp.full((query_count,), 0.5),
        ),
        axis=-1,
    )

    def query(values: Any) -> Any:
        return point_select_leaf_items(
            values,
            bvh=bvh,
            maximum_candidates=4,
            query_batch_capacity=64,
        )

    compiled_query = eqx.filter_jit(query)
    executable, compilation = measure_lower_and_compile(
        # ty: ignore[unresolved-attribute]
        lambda: compiled_query.lower(points),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: executable(points),
        warmup=1,
        repeats=repeats,
    )
    return {
        "item_count": item_count,
        "node_count": bvh.left.shape[0],
        "query_count": query_count,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_milliseconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source="jax-compiled-executable",
            )
        ),
        "output_bytes": logical_array_bytes(result),
        "complete": bool(jnp.all(result[2])),
    }


def run_spatial_query_scaling(
    node_counts: Sequence[int] = (127, 511, 2047),
    /,
    *,
    query_count: int = 512,
    repeats: int = 5,
) -> dict[str, object]:
    return {
        "campaign": "packed-bvh-query-scaling",
        "rows": [
            _case(int(node_count), int(query_count), int(repeats))
            for node_count in node_counts
        ],
    }


def _phase(
    function: Any, args: tuple[Any, ...], repeats: int
) -> tuple[Any, dict[str, Any]]:
    compiled = eqx.filter_jit(function)
    executable, compilation = measure_lower_and_compile(
        lambda: compiled.lower(*args),  # ty: ignore[unresolved-attribute]
        lambda lowered: lowered.compile(),
    )
    result, first = measure_repeated(lambda: executable(*args), warmup=0, repeats=1)
    _, warm = measure_repeated(lambda: executable(*args), warmup=1, repeats=repeats)
    return result, {
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "first": first.to_milliseconds_dict(),
        "warm": warm.to_milliseconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source="jax-compiled-executable",
            )
        ),
        "output_bytes": logical_array_bytes(result),
    }


class _PreparedSources(Protocol):
    @property
    def invalid_points(self) -> jax.Array: ...

    @property
    def stable_ids_unique(self) -> jax.Array: ...


class _QueryEvidence(Protocol):
    @property
    def successful(self) -> jax.Array: ...


class _QueryResult(Protocol):
    @property
    def status(self) -> jax.Array: ...

    @property
    def source_indices(self) -> jax.Array: ...

    @property
    def valid(self) -> jax.Array: ...

    @property
    def evidence(self) -> _QueryEvidence: ...


def _family_phases[P: _PreparedSources, R: _QueryResult](
    name: str,
    prepare: Callable[[jax.Array], P],
    query: Callable[[P, jax.Array], R],
    points: jax.Array,
    targets: jax.Array,
    dense: QueryNeighborhood,
    repeats: int,
) -> dict[str, Any]:
    """Separate one-time source preparation from repeated prepared and raw queries."""
    phases: dict[str, Any] = {}
    prepared, timing = _phase(prepare, (points,), repeats)
    timing.update(
        successful=bool((prepared.invalid_points == 0) & prepared.stable_ids_unique),
        invalid_points=np.asarray(prepared.invalid_points).tolist(),
    )
    phases[name + "_prepare"] = timing

    def prepared_query(sources: P, batches: jax.Array) -> R:
        return jax.lax.map(lambda target: query(sources, target), batches)

    def raw_query(source: jax.Array, batches: jax.Array) -> R:
        return jax.lax.map(lambda target: query(prepare(source), target), batches)

    for route, result_timing in (
        ("query", _phase(prepared_query, (prepared, targets), repeats)),
        ("raw", _phase(raw_query, (points, targets), repeats)),
    ):
        result, timing = result_timing
        timing.update(
            status=np.asarray(result.status).tolist(),
            successful=bool(jnp.all(result.evidence.successful)),
            exactness_vs_dense=bool(
                jnp.all(result.source_indices == dense.indices)
                & jnp.all(result.valid == dense.mask)
            ),
        )
        phases[name + "_" + route] = timing
    return phases


def run_nearest_neighbors(
    source_count: int,
    query_count: int,
    neighbors: int,
    dimension: int,
    distribution: str,
    dtype: str,
    repeats: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    match distribution:
        case "uniform":
            host = rng.uniform(0.05, 0.95, (source_count, dimension))
        case "clustered":
            host = np.clip(rng.normal(0.5, 0.025, (source_count, dimension)), 0.0, 1.0)
        case "coincident":
            host = np.full((source_count, dimension), 0.5, dtype=np.float64)
        case _:
            raise ValueError("Unknown distribution.")
    points = jnp.asarray(host, dtype=jnp.dtype(dtype))
    targets = jnp.asarray(
        rng.uniform(0.0, 1.0, (3, query_count, dimension)), dtype=points.dtype
    )
    bvh_source = BVHSourcePlan(dimension, source_count)
    bvh = BVHNeighborQueryPlan(bvh_source, query_count, neighbors)
    # The dense reference uses budget-derived exact tiles (256 MiB workspace),
    # not deliberately tiny chunks that would serialize millions of tile steps.
    dense, dense_timing = _phase(
        lambda source, target: query_neighbors(
            source,
            target,
            max_neighbors=neighbors,
            query_workspace_bytes=256 * 1024**2,
        ),
        (points, targets),
        repeats,
    )
    dense_timing.update(
        successful=bool(dense.evidence.successful),
        exactness_vs_dense=True,
    )
    phases: dict[str, Any] = {"dense_blocked_graph_query": dense_timing}
    # Morton addressing is defined for one to three dimensions only.
    if dimension <= 3:
        address = MortonAddressPlan((0.0,) * dimension, (1.0,) * dimension, 6)
        morton_source = MortonSourcePlan(address, source_count)
        morton = MortonNeighborQueryPlan(morton_source, query_count, neighbors)
        phases.update(
            _family_phases(
                "morton",
                morton_source.prepare,
                morton.query,
                points,
                targets,
                dense,
                repeats,
            )
        )
    phases.update(
        _family_phases(
            "bvh", bvh_source.prepare, bvh.query, points, targets, dense, repeats
        )
    )
    host_points = np.asarray(points)
    start = perf_counter()
    host_tree = prepare_bvh(host_points, host_points, policy=BVHBuildPolicy(leaf_size=8))
    jax.block_until_ready(host_tree)
    phases["host_prepare_bvh_and_transfer"] = {
        "seconds": perf_counter() - start,
        "output_bytes": logical_array_bytes(host_tree),
    }
    return {
        "campaign": "phase-separated-nearest-neighbors",
        "source_count": source_count,
        "query_count": query_count,
        "neighbors": neighbors,
        "dimension": dimension,
        "distribution": distribution,
        "dtype": dtype,
        "target_batches": targets.shape[0],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=(127, 511, 2047))
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scenario", choices=("primitive", "nearest"), default="primitive"
    )
    parser.add_argument("--sources", type=int, default=128)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument(
        "--distribution",
        choices=("uniform", "clustered", "coincident"),
        default="uniform",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    result = (
        run_nearest_neighbors(
            arguments.sources,
            arguments.queries,
            arguments.neighbors,
            arguments.dimension,
            arguments.distribution,
            arguments.dtype,
            arguments.repeats,
        )
        if arguments.scenario == "nearest"
        else run_spatial_query_scaling(
            tuple(arguments.nodes),
            query_count=arguments.queries,
            repeats=arguments.repeats,
        )
    )
    serialized = json.dumps(result, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()


__all__ = ["run_spatial_query_scaling", "run_nearest_neighbors"]
