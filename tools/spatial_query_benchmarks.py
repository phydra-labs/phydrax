from __future__ import annotations

import argparse
import json
import time

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx


def _block(value) -> None:
    for leaf in jax.tree.leaves(value):
        if isinstance(leaf, jax.Array):
            leaf.block_until_ready()


def _measure(function, *arguments):
    started = time.perf_counter()
    value = function(*arguments)

    _block(value)
    return value, time.perf_counter() - started


def _positions(count: int, distribution: str) -> jax.Array:
    key = jax.random.key(1700 + count)
    if distribution == "uniform":
        return 0.02 + 0.96 * jax.random.uniform(key, (count, 3))
    cluster = 0.5 + 0.06 * jax.random.normal(key, (count, 3))
    return jnp.clip(cluster, 0.02, 0.98)


def _bytes(value) -> int:
    return sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(value)
        if isinstance(leaf, jax.Array)
    )


def _case(count: int, distribution: str, neighbors: int) -> dict[str, object]:
    positions = _positions(count, distribution)
    address = phx.discretization.spatial.MortonAddressPlan(
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        min(21, max(8, int(jnp.ceil(jnp.log2(count))) + 3)),
    )
    plan = phx.discretization.spatial.MortonNeighborQueryPlan(
        address,
        count,
        count,
        neighbors,
        maximum_leaf_occupancy=16,
        coarsening_factor=8,
        target_top_nodes=64,
    )

    dense = eqx.filter_jit(
        lambda points: phx.graph.query_neighbors(
            points,
            points,
            max_neighbors=neighbors,
            exclude_self=True,
            target_chunk_size=min(count, 128),
        )
    )
    morton = eqx.filter_jit(
        lambda points: phx.graph.query_neighbors(
            points,
            points,
            max_neighbors=neighbors,
            exclude_self=True,
            plan=plan,
        )
    )
    dense_result, dense_first = _measure(dense, positions)
    _, dense_steady = _measure(dense, positions)
    morton_result, morton_first = _measure(morton, positions)
    _, morton_steady = _measure(morton, positions)
    schedule, schedule_first = _measure(
        eqx.filter_jit(plan.schedule_plan.build), positions
    )
    _, schedule_steady = _measure(eqx.filter_jit(plan.schedule_plan.build), positions)
    exact = bool(
        jnp.array_equal(morton_result.indices, dense_result.indices)
        & jnp.array_equal(morton_result.mask, dense_result.mask)
    )
    return {
        "distribution": distribution,
        "points": count,
        "neighbors": neighbors,
        "plane_count": plan.schedule_plan.plane_count,
        "active_nodes": int(schedule.evidence.active_nodes),
        "active_leaves": int(schedule.evidence.active_leaves),
        "schedule_bytes": _bytes(schedule),
        "schedule_first_seconds": schedule_first,
        "schedule_steady_seconds": schedule_steady,
        "dense_first_seconds": dense_first,
        "dense_steady_seconds": dense_steady,
        "morton_first_seconds": morton_first,
        "morton_steady_seconds": morton_steady,
        "steady_speedup_over_dense": dense_steady / morton_steady,
        "maximum_required_candidates": int(
            plan.query(
                positions,
                positions,
                exclude_self=True,
            ).evidence.required_candidates
        ),
        "exact": exact,
        "successful": bool(morton_result.evidence.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark exact spatial queries.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    counts = (32, 64) if arguments.smoke else (128, 512, 2048)
    report = {
        "kind": "spatial-query-benchmark",
        "device": str(jax.devices()[0]),
        "cases": [
            _case(count, distribution, min(8, count - 1))
            for distribution in ("uniform", "clustered")
            for count in counts
        ],
    }
    report["passed"] = all(
        case["successful"] and case["exact"] for case in report["cases"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
