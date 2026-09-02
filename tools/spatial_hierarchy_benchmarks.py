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


def _positions(count: int, distribution: str):
    key = jax.random.key(1000 + count)
    if distribution == "uniform":
        return 0.05 + 0.9 * jax.random.uniform(key, (count, 3))
    cluster = 0.5 + 0.08 * jax.random.normal(key, (count, 3))
    return jnp.clip(cluster, 0.05, 0.95)


def _reference(positions, masses, softening):
    displacement = positions[None, :, :] - positions[:, None, :]
    squared = jnp.sum(displacement**2, axis=-1) + softening**2
    mask = ~jnp.eye(positions.shape[0], dtype=bool)
    return jnp.sum(
        jnp.where(
            mask[..., None],
            masses[None, :, None] * displacement / squared[..., None] ** 1.5,
            0.0,
        ),
        axis=1,
    )


def _case(count: int, distribution: str):
    positions = _positions(count, distribution)
    masses = jnp.full((count,), 1.0 / count)
    depth = min(10, max(2, int(jnp.ceil(jnp.log2(count) / 3.0)) + 2))
    tree_plan = phx.applications.cosmology.ParticleOctreePlan3D(
        (1.0, 1.0, 1.0), depth, target_leaf_occupancy=1
    )
    prepare = eqx.filter_jit(tree_plan.prepare)
    tree, prepare_first = _measure(prepare, positions, masses)
    _, prepare_steady = _measure(prepare, positions, masses)
    barnes_hut = phx.applications.cosmology.BarnesHutGravityPlan(
        1.0,
        softening=0.01,
        opening_angle=0.5,
        use_quadrupole=True,
    )
    evaluate = eqx.filter_jit(barnes_hut.evaluate)
    result, evaluate_first = _measure(evaluate, tree)
    _, evaluate_steady = _measure(evaluate, tree)
    direct = jax.jit(lambda p, m: _reference(p, m, 0.01))
    reference, direct_first = _measure(direct, positions, masses)
    _, direct_steady = _measure(direct, positions, masses)
    relative_error = float(
        jnp.linalg.norm(result.acceleration - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1.0e-15)
    )
    hierarchy_bytes = sum(
        int(leaf.size * leaf.dtype.itemsize)
        for leaf in jax.tree.leaves(tree.hierarchy)
        if isinstance(leaf, jax.Array)
    )
    return {
        "distribution": distribution,
        "particles": count,
        "depth": depth,
        "active_nodes": int(tree.hierarchy.evidence.active_nodes),
        "active_leaves": int(tree.hierarchy.evidence.active_leaves),
        "hierarchy_bytes": hierarchy_bytes,
        "prepare_first_seconds": prepare_first,
        "prepare_steady_seconds": prepare_steady,
        "evaluate_first_seconds": evaluate_first,
        "evaluate_steady_seconds": evaluate_steady,
        "direct_first_seconds": direct_first,
        "direct_steady_seconds": direct_steady,
        "steady_speedup_over_direct": direct_steady / evaluate_steady,
        "accepted_nodes": int(result.evidence.accepted_leaf_interactions),
        "direct_interactions": int(result.evidence.direct_particle_interactions),
        "relative_error": relative_error,
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sparse spatial hierarchies.")
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    counts = (32, 64) if arguments.smoke else (64, 256, 1024)
    report = {
        "kind": "spatial-hierarchy-benchmark",
        "device": str(jax.devices()[0]),
        "cases": [
            _case(count, distribution)
            for distribution in ("uniform", "clustered")
            for count in counts
        ],
    }
    report["passed"] = all(
        case["successful"] and case["relative_error"] < 0.25 for case in report["cases"]
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
