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


def _measure(function, *arguments, **keyword_arguments):
    started = time.perf_counter()
    value = function(*arguments, **keyword_arguments)
    _block(value)
    return value, time.perf_counter() - started


def _sphere_surfels(count: int):
    index = jnp.arange(count, dtype=float)
    golden_angle = jnp.pi * (3.0 - jnp.sqrt(5.0))
    vertical = 1.0 - 2.0 * (index + 0.5) / count
    radial = jnp.sqrt(jnp.maximum(1.0 - vertical**2, 0.0))
    angle = golden_angle * index
    position = jnp.stack(
        (radial * jnp.cos(angle), radial * jnp.sin(angle), vertical), axis=-1
    )
    normal = position
    reference = jnp.tile(jnp.asarray((0.0, 0.0, 1.0)), (count, 1))
    alternate = jnp.tile(jnp.asarray((1.0, 0.0, 0.0)), (count, 1))
    reference = jnp.where((jnp.abs(normal[:, 2]) > 0.9)[:, None], alternate, reference)
    first = jnp.cross(reference, normal)
    first = first / jnp.sqrt(jnp.sum(first**2, axis=-1, keepdims=True))
    second = jnp.cross(normal, first)
    radius = 2.0 / jnp.sqrt(jnp.asarray(count, dtype=float))
    axes = jnp.stack((radius * first, radius * second), axis=-1)
    weight = jnp.full((count,), 4.0 * jnp.pi / count)
    return position, normal, axes, weight


def _case(count: int):
    position, normal, axes, weight = _sphere_surfels(count)
    ids = jnp.arange(count, dtype=jnp.int64)
    surfel_plan = phx.discretization.SurfelSetPlan(ids, position, weight)
    prepared = surfel_plan.prepare()
    geometry_plan = phx.discretization.SurfelGeometryPlan(prepared)
    materialize = eqx.filter_jit(geometry_plan.materialize)
    geometry, materialize_first = _measure(materialize, position, normal, axes)
    _, materialize_steady = _measure(materialize, position, normal, axes)
    depth = max(3, int(jnp.ceil(jnp.log2(count) / 3.0)) + 2)
    hierarchy_plan = phx.discretization.MortonPointHierarchyPlan(
        phx.discretization.MortonAddressPlan((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5), depth),
        count,
        target_leaf_occupancy=4,
    )
    build = eqx.filter_jit(hierarchy_plan.build)
    hierarchy, hierarchy_first = _measure(
        build, geometry.position, stable_ids=prepared.surfel_ids
    )
    _, hierarchy_steady = _measure(
        build, geometry.position, stable_ids=prepared.surfel_ids
    )
    bounds_plan = phx.discretization.MortonPrimitiveBoundsPlan(hierarchy, 3)
    refit = eqx.filter_jit(bounds_plan.refit)
    bounds, bounds_first = _measure(
        refit,
        geometry.position - geometry.footprint_half_width,
        geometry.position + geometry.footprint_half_width,
    )
    _, bounds_steady = _measure(
        refit,
        geometry.position - geometry.footprint_half_width,
        geometry.position + geometry.footprint_half_width,
    )
    ray_origins = 2.0 * position[: min(count, 64)]
    ray_directions = -position[: min(count, 64)]
    query_plan = phx.discretization.SurfelRayQueryPlan(
        bounds, geometry, maximum_hits_per_ray=16
    )
    query = eqx.filter_jit(query_plan.query)
    hits, query_first = _measure(query, ray_origins, ray_directions)
    _, query_steady = _measure(query, ray_origins, ray_directions)
    topology_bytes = sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves((hierarchy, bounds))
        if isinstance(leaf, jax.Array)
    )
    return {
        "surfels": count,
        "active_nodes": int(hierarchy.evidence.active_nodes),
        "active_leaves": int(hierarchy.evidence.active_leaves),
        "topology_bytes": topology_bytes,
        "materialize_first_seconds": materialize_first,
        "materialize_steady_seconds": materialize_steady,
        "hierarchy_first_seconds": hierarchy_first,
        "hierarchy_steady_seconds": hierarchy_steady,
        "bounds_first_seconds": bounds_first,
        "bounds_steady_seconds": bounds_steady,
        "ray_first_seconds": query_first,
        "ray_steady_seconds": query_steady,
        "ray_hits": int(jnp.sum(hits.valid)),
        "successful": bool(
            geometry.evidence.successful
            & hierarchy.evidence.successful
            & bounds.evidence.successful
            & jnp.all(hits.evidence.successful)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native surfel substrate.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    counts = (32,) if arguments.smoke else (32, 128, 512)
    cases = [_case(count) for count in counts]
    report = {
        "kind": "surfel-substrate-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(case["successful"] for case in cases),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
