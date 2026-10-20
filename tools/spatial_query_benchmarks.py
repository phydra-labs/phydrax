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
from typing import Any, get_args, Literal, NamedTuple, Protocol, TypeAlias

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
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
from phydrax.typing import parse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NearestRoute: TypeAlias = Literal["all", "morton"]


class _HostNeighbors(NamedTuple):
    indices: np.ndarray
    mask: np.ndarray


def _host_nearest(points: np.ndarray, targets: np.ndarray, count: int) -> _HostNeighbors:
    """Independent dense reference on identical rounded inputs, in bounded chunks."""
    source = points.astype(np.float64)
    queries = targets.reshape((-1, targets.shape[-1]))
    indices = np.empty((queries.shape[0], count), dtype=np.int64)
    for start in range(0, queries.shape[0], 32):
        query = queries[start : start + 32].astype(np.float64)
        difference = query[:, None, :] - source[None, :, :]
        distance = np.sum(difference * difference, axis=-1)
        indices[start : start + query.shape[0]] = np.argsort(
            distance, axis=-1, kind="stable"
        )[:, :count]
    shape = (*targets.shape[:-1], count)
    return _HostNeighbors(indices.reshape(shape), np.ones(shape, dtype=np.bool_))


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
                unavailable_reason="The selected backend did not expose compiler estimates.",
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
                unavailable_reason="The selected backend did not expose compiler estimates.",
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
    dense: QueryNeighborhood | _HostNeighbors,
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
    *,
    route: NearestRoute = "all",
) -> dict[str, Any]:
    route = parse(route, NearestRoute, "route")
    if (
        source_count < 1
        or query_count < 1
        or repeats < 1
        or not 1 <= neighbors <= source_count
    ):
        raise ValueError(
            "Counts and repeats must be positive; neighbors must fit sources."
        )
    if route == "morton":
        if not 1 <= dimension <= 3:
            raise ValueError(
                "The prepared Morton route requires one to three dimensions."
            )
        query_rows = 3 * query_count
        reference_bytes = (
            min(32, query_rows) * source_count * (2 * dimension + 4) * 8
            + (source_count + query_rows) * dimension * 8
            + query_rows * neighbors * 9
        )
        if reference_bytes > 256 * 1024**2:
            raise ValueError(
                "The host nearest reference exceeds its 256 MiB working-set bound."
            )
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
    host = np.asarray(host, dtype=np.dtype(dtype))
    host_targets = rng.uniform(0.0, 1.0, (3, query_count, dimension)).astype(host.dtype)
    points, targets = jnp.asarray(host), jnp.asarray(host_targets)
    phases: dict[str, Any] = {}
    dense: QueryNeighborhood | _HostNeighbors
    if route == "morton":
        dense, reference_seconds = measure_host(
            lambda: _host_nearest(host, host_targets, neighbors)
        )
        reference = {
            "kind": "host-float64-dense",
            "seconds": reference_seconds,
            "working_set_bound_bytes": reference_bytes,
        }
    else:
        # The generic graph reference deliberately owns float64 geometry.
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
            successful=bool(dense.evidence.successful), exactness_vs_dense=True
        )
        phases["dense_blocked_graph_query"] = dense_timing
        reference = {"kind": "jax-float64-dense"}
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
        # Refresh moves every source by a quarter of a finest cell in the
        # declared dtype; its candidate must equal a fresh preparation exactly.
        prepared = morton_source.prepare(points)
        moved = points + jnp.asarray(0.25 / address.resolution, dtype=points.dtype)
        transition, refresh = _phase(morton_source.refresh, (prepared, moved), repeats)
        fresh = morton_source.prepare(moved)
        refresh.update(
            successful=bool(transition.successful),
            reused_order=bool(transition.reused_order),
            coordinate_dtype=str(transition.accepted.points.dtype),
            matches_fresh_preparation=all(
                bool(jnp.array_equal(candidate, reference))
                for candidate, reference in zip(
                    jax.tree.leaves(transition.candidate),
                    jax.tree.leaves(fresh),
                    strict=True,
                )
            ),
        )
        phases["morton_refresh"] = refresh
    if route == "all":
        bvh_source = BVHSourcePlan(dimension, source_count)
        bvh = BVHNeighborQueryPlan(bvh_source, query_count, neighbors)
        phases.update(
            _family_phases(
                "bvh", bvh_source.prepare, bvh.query, points, targets, dense, repeats
            )
        )
        start = perf_counter()
        host_tree = prepare_bvh(host, host, policy=BVHBuildPolicy(leaf_size=8))
        jax.block_until_ready(host_tree)
        phases["host_prepare_bvh_and_transfer"] = {
            "seconds": perf_counter() - start,
            "output_bytes": logical_array_bytes(host_tree),
        }
    return {
        "campaign": "phase-separated-nearest-neighbors",
        "route": route,
        "reference": reference,
        "source_count": source_count,
        "query_count": query_count,
        "neighbors": neighbors,
        "dimension": dimension,
        "distribution": distribution,
        "dtype": dtype,
        "target_batches": targets.shape[0],
        "phases": phases,
        # Every phase that reports status, dense exactness or refresh identity passed.
        "passed": all(
            bool(phase.get("successful", True))
            and bool(phase.get("exactness_vs_dense", True))
            and bool(phase.get("matches_fresh_preparation", True))
            for phase in phases.values()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", nargs="+", type=int, default=(127, 511, 2047))
    parser.add_argument("--queries", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument(
        "--scenario", choices=("primitive", "nearest"), default="primitive"
    )
    parser.add_argument("--route", choices=get_args(NearestRoute), default="all")
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
    if arguments.scenario != "nearest" and arguments.route != "all":
        parser.error("--route applies only to the nearest scenario.")
    result = (
        run_nearest_neighbors(
            arguments.sources,
            arguments.queries,
            arguments.neighbors,
            arguments.dimension,
            arguments.distribution,
            arguments.dtype,
            arguments.repeats,
            route=arguments.route,
        )
        if arguments.scenario == "nearest"
        else run_spatial_query_scaling(
            tuple(arguments.nodes),
            query_count=arguments.queries,
            repeats=arguments.repeats,
        )
    )
    result["environment"] = capture_environment().to_dict()
    result["identity"] = capture_benchmark_identity(
        PROJECT_ROOT, Path(__file__), tuple(result)
    ).to_dict()
    serialized = json.dumps(result, sort_keys=True)
    if arguments.output is not None:
        arguments.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)


if __name__ == "__main__":
    main()


__all__ = ["run_spatial_query_scaling", "run_nearest_neighbors"]
