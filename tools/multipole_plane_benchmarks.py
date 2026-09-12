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


def _inputs(source_count: int, target_count: int):
    key = jax.random.key(4180 + source_count + target_count)
    sources = 0.05 + 0.9 * jax.random.uniform(key, (source_count, 3))
    targets = 0.05 + 0.9 * jax.random.uniform(
        jax.random.fold_in(key, 1), (target_count, 3)
    )
    strengths = jax.random.normal(jax.random.fold_in(key, 2), (source_count,))
    return sources, targets, strengths


def _direct(kind: str, sources, strengths, targets, parameter: float):
    radius = jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
    if kind == "laplace":
        factor = 1.0
    elif kind == "helmholtz":
        factor = jnp.exp(1j * parameter * radius)
    else:
        factor = jnp.exp(-parameter * radius)
    return jnp.sum(factor * strengths[None, :] / (4.0 * jnp.pi * radius), axis=1)


def _plan_type(kind: str):
    if kind == "laplace":
        return phx.operators.LaplaceMultipolePlan3D, {}
    if kind == "helmholtz":
        return phx.operators.HelmholtzMultipolePlan3D, {"wavenumber": 0.7}
    return phx.operators.ModifiedHelmholtzMultipolePlan3D, {"decay": 0.7}


def _case(kind: str, execution: str, source_count: int, target_count: int, order: int):
    sources, targets, strengths = _inputs(source_count, target_count)
    plan_type, keyword = _plan_type(kind)
    policy = {
        "reference_targets": targets,
        "depth": 3,
        "expansion_order": order,
        "execution": execution,
        **keyword,
    }
    if execution == "plane_dual":
        policy.update(
            source_leaf_occupancy=1,
            target_leaf_occupancy=1,
            plane_coarsening_factor=2,
            plane_target_top_nodes=1,
        )
        if kind != "laplace":
            policy["maximum_plane_node_argument"] = 0.75
    started = time.perf_counter()
    prepared = plan_type(
        sources,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
        **policy,
    ).prepare()
    preparation_seconds = time.perf_counter() - started
    evaluate = eqx.filter_jit(
        lambda source, strength, target: prepared.evaluate(source, strength, target)
    )
    first, first_seconds = _measure(evaluate, sources, strengths, targets)
    result, steady_seconds = _measure(evaluate, sources, strengths, targets)
    expected = _direct(kind, sources, strengths, targets, 0.7)
    absolute_error = jnp.max(jnp.abs(result.values - expected), initial=0.0)
    capacity = result.capacity
    return {
        "kernel": kind,
        "execution": execution,
        "source_count": source_count,
        "target_count": target_count,
        "expansion_order": order,
        "preparation_seconds": preparation_seconds,
        "first_seconds": first_seconds,
        "steady_seconds": steady_seconds,
        "maximum_absolute_error": float(absolute_error),
        "required_nodes": int(capacity.required_nodes),
        "node_capacity": int(capacity.node_capacity),
        "required_far": int(capacity.required_far_interactions),
        "far_capacity": int(capacity.far_interaction_capacity),
        "required_near": int(capacity.required_near_interactions),
        "near_capacity": int(capacity.near_interaction_capacity),
        "m2l_count": int(result.m2l_count),
        "p2p_count": int(result.p2p_count),
        "successful": bool(result.successful),
        "checksum": float(jnp.sum(jnp.abs(first.values))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark level-octree and plane multipole execution."
    )
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    source_count, target_count, order = (6, 4, 3) if arguments.smoke else (16, 12, 4)
    cases = [
        _case(kind, execution, source_count, target_count, order)
        for kind in ("laplace", "modified-helmholtz", "helmholtz")
        for execution in ("level_octree", "plane_dual")
    ]
    report = {
        "kind": "multipole-plane-execution-benchmark",
        "device": str(jax.devices()[0]),
        "cases": cases,
        "passed": all(
            case["successful"] and case["maximum_absolute_error"] < 5.0e-2
            for case in cases
        ),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
