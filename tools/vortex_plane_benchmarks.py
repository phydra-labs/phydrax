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


def _case(count: int):
    key = jax.random.key(6150 + count)
    sources = 1.6 * jax.random.uniform(key, (count, 3)) - 0.8
    targets = (
        1.6 * jax.random.uniform(jax.random.fold_in(key, 1), (max(2, count // 2), 3))
        - 0.8
    )
    strength = jax.random.normal(jax.random.fold_in(key, 2), (count, 3))
    core = jnp.full((count,), 0.1)
    source = phx.discretization.VortexSourceState(
        sources,
        strength,
        core_radius=core,
    )
    target = phx.discretization.VortexTargetState(targets)
    plane = phx.operators.VortexFMMPlan(
        sources,
        (-1.0, -1.0, -1.0),
        (1.0, 1.0, 1.0),
        plane_opening_angle=0.3,
        reference_targets=targets,
        depth=4,
        expansion_order=1,
        leaf_capacity=1,
        target_leaf_capacity=1,
        maximum_reference_displacement=0.05,
        execution="plane_dual",
        plane_coarsening_factor=2,
        plane_target_top_nodes=1,
    ).prepare(
        source_capacity=count,
        target_capacity=targets.shape[0],
        target_topology="arbitrary-targets",
    )
    direct = phx.operators.GaussianErfDirectVortexPlan3D(
        maximum_sources=count,
        maximum_targets=targets.shape[0],
        maximum_interactions=count * targets.shape[0],
    ).prepare(
        source_capacity=count,
        target_capacity=targets.shape[0],
        target_topology="arbitrary-targets",
    )
    plane_evaluate = eqx.filter_jit(plane.evaluate)
    direct_evaluate = eqx.filter_jit(direct.evaluate)
    reference, direct_first = _measure(direct_evaluate, source, target)
    _, direct_steady = _measure(direct_evaluate, source, target)
    candidate, plane_first = _measure(plane_evaluate, source, target)
    _, plane_steady = _measure(plane_evaluate, source, target)
    error = jnp.max(
        jnp.linalg.norm(candidate.velocity - reference.velocity, axis=-1),
        initial=0.0,
    )
    evidence = candidate.diagnostics.backend_diagnostics
    return {
        "points": count,
        "targets": targets.shape[0],
        "direct_first_seconds": direct_first,
        "direct_steady_seconds": direct_steady,
        "plane_first_seconds": plane_first,
        "plane_steady_seconds": plane_steady,
        "maximum_velocity_error": float(error),
        "m2l_count": int(evidence.m2l_count),
        "near_pair_count": int(evidence.near_pair_count),
        "tail_bound": float(evidence.geometric_tail_bound),
        "successful": bool(candidate.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark fixed-envelope vortex plane FMM execution."
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    counts = (4, 8) if arguments.smoke else (16, 32)
    cases = [_case(count) for count in counts]
    report = {
        "kind": "vortex-plane-execution-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(
            case["successful"] and case["maximum_velocity_error"] < 0.1 for case in cases
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
