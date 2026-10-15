from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    measure_synchronized,
)
from tools.multipole3d_benchmarks import _phase, _preparation, _translation_phases


def _inputs(source_count: int, target_count: int) -> Any:
    key = jax.random.key(4180 + source_count + target_count)
    sources = 0.05 + 0.9 * jax.random.uniform(key, (source_count, 3), dtype=jnp.float64)
    targets = 0.05 + 0.9 * jax.random.uniform(
        jax.random.fold_in(key, 1), (target_count, 3), dtype=jnp.float64
    )
    strengths = jax.random.normal(
        jax.random.fold_in(key, 2), (source_count,), dtype=jnp.float64
    )
    return sources, targets, strengths


def _direct(
    kind: str, sources: Any, strengths: Any, targets: Any, parameter: float
) -> Any:
    radius = jnp.linalg.norm(targets[:, None, :] - sources[None, :, :], axis=-1)
    if kind == "laplace":
        factor = 1.0
    elif kind == "helmholtz":
        factor = jnp.exp(1j * parameter * radius)
    else:
        factor = jnp.exp(-parameter * radius)
    return jnp.sum(factor * strengths[None, :] / (4.0 * jnp.pi * radius), axis=1)


def _plan_type(kind: str) -> Any:
    if kind == "laplace":
        return phx.operators.LaplaceMultipolePlan3D, {}
    if kind == "helmholtz":
        return phx.operators.HelmholtzMultipolePlan3D, {"wavenumber": 0.7}
    return phx.operators.ModifiedHelmholtzMultipolePlan3D, {"decay": 0.7}


def _case(
    kind: str,
    execution: str,
    source_count: int,
    target_count: int,
    order: int,
    translation_route: str,
    repeats: int,
) -> Any:
    sources, targets, strengths = _inputs(source_count, target_count)
    plan_type, keyword = _plan_type(kind)
    policy = {
        "reference_targets": targets,
        "depth": 3,
        "expansion_order": order,
        "execution": execution,
        "translation_route": translation_route,
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
    else:
        policy["source_leaf_occupancy"] = 2
    plan, policy_seconds = measure_synchronized(
        lambda: plan_type(sources, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), **policy)
    )
    prepared, preparation = _preparation(plan)
    result, evaluation = _phase(
        lambda source, strength, target: prepared.evaluate(source, strength, target),
        (sources, strengths, targets),
        repeats,
    )

    def target_gradient(source: Any, strength: Any, target: Any) -> Any:
        return jax.grad(
            lambda positions: jnp.real(
                jnp.sum(prepared.evaluate(source, strength, positions).values)
            )
        )(target)

    _, derivative = _phase(target_gradient, (sources, strengths, targets), repeats)
    expected = _direct(kind, sources, strengths, targets, 0.7)
    absolute_error = jnp.max(jnp.abs(result.values - expected), initial=0.0)
    capacity = result.capacity
    return {
        "kernel": kind,
        "execution": execution,
        "translation_route": translation_route,
        "plan_id": plan.plan_id,
        "prepared_id": prepared.prepared_id,
        "source_count": source_count,
        "target_count": target_count,
        "expansion_order": order,
        "preparation": {
            "policy_and_reference_transfer_seconds": policy_seconds,
            **preparation,
        },
        "translations": _translation_phases(prepared, sources, strengths, repeats),
        "complete_evaluation": evaluation,
        "complete_target_gradient": derivative,
        "translation_table_bytes": prepared.resources.translation_table_bytes,
        "translation_workspace_bytes": prepared.resources.translation_workspace_bytes,
        "maximum_absolute_error": float(absolute_error),
        "required_nodes": int(capacity.required_nodes),
        "node_capacity": int(capacity.node_capacity),
        "required_far": int(capacity.required_far_interactions),
        "far_capacity": int(capacity.far_interaction_capacity),
        "required_near": int(capacity.required_near_interactions),
        "near_capacity": int(capacity.near_interaction_capacity),
        "m2l_count": int(result.m2l_count),
        "p2l_count": int(result.p2l_count),
        "m2p_count": int(result.m2p_count),
        "p2p_count": int(result.p2p_count),
        "successful": bool(result.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark level-octree and plane multipole execution."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--kernel",
        choices=("all", "laplace", "helmholtz", "modified-helmholtz"),
        default="all",
    )
    parser.add_argument(
        "--translation-route", choices=("dense", "rotation"), default="dense"
    )
    arguments = parser.parse_args()
    if arguments.translation_route == "rotation" and arguments.kernel != "laplace":
        parser.error(
            "--translation-route rotation requires --kernel laplace; screened kernels refuse this route"
        )
    source_count, target_count, order = (6, 4, 3) if arguments.smoke else (16, 12, 4)
    kinds = (
        ("laplace", "modified-helmholtz", "helmholtz")
        if arguments.kernel == "all"
        else (arguments.kernel,)
    )
    cases = [
        _case(
            kind,
            execution,
            source_count,
            target_count,
            order,
            arguments.translation_route,
            arguments.repeats,
        )
        for kind in kinds
        for execution in ("level_octree", "plane_dual")
    ]
    report = {
        "kind": "multipole-plane-execution-benchmark",
        "environment": capture_environment().to_dict(),
        "cases": cases,
        "passed": all(
            case["successful"] and case["maximum_absolute_error"] < 5.0e-2
            for case in cases
        ),
    }
    root = Path(__file__).resolve().parents[1]
    report["identity"] = capture_benchmark_identity(
        root, Path(__file__), report.keys()
    ).to_dict()
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
