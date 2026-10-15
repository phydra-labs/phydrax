from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp

import phydrax as phx
from benchmarks._io import write_json_atomic
from benchmarks._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax import config as pxconfig
from phydrax.discretization.spatial._plane_interactions import MortonPlaneInteractionPlan
from phydrax.discretization.spatial._plane_schedule import MortonPlaneSchedulePlan
from phydrax.typing import parse


def _phase(
    function: Callable[..., Any],
    *arguments: Any,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    compiled_function: Any = eqx.filter_jit(function)
    executable, compilation = measure_lower_and_compile(
        lambda: compiled_function.lower(*arguments),
        lambda lowered: lowered.compile(),
    )
    result, first = measure_synchronized(lambda: executable(*arguments))
    result, warm = measure_repeated(
        lambda: executable(*arguments), warmup=warmup, repeats=repeats
    )
    return result, {
        **asdict(compilation),
        "first_seconds": first,
        "warm": warm.to_seconds_dict(),
        "compiler": asdict(
            compiler_evidence(
                executable.compiled.cost_analysis(),
                executable.compiled.memory_analysis(),
                source="jax-compiled-executable",
                unavailable_reason="Compiler did not expose every estimate.",
            )
        ),
        "output_bytes": logical_array_bytes(result),
    }


def _positions(count: int, distribution: str, dtype: Any) -> jax.Array:
    key = jax.random.key(1000 + count)
    if distribution == "uniform":
        return 0.05 + 0.9 * jax.random.uniform(key, (count, 3), dtype=dtype)
    cluster = 0.5 + 0.08 * jax.random.normal(key, (count, 3), dtype=dtype)
    return jnp.clip(cluster, 0.05, 0.95)


def _reference(positions: jax.Array, masses: jax.Array, softening: float) -> jax.Array:
    displacement = positions[None, :, :] - positions[:, None, :]
    squared = jnp.sum(displacement**2, axis=-1) + softening**2
    mask = ~jnp.eye(positions.shape[0], dtype=jnp.bool_)
    return jnp.sum(
        jnp.where(
            mask[..., None],
            masses[None, :, None] * displacement / squared[..., None] ** 1.5,
            0.0,
        ),
        axis=1,
    )


def _interaction_case(
    positions: jax.Array,
    depth: int,
    *,
    target_top_nodes: int,
    leaf_occupancy: int,
    coarsening_factor: int,
    opening_angle: float,
    queue_capacity: int | None,
    far_capacity: int | None,
    near_capacity: int | None,
    frontier_widths: tuple[int, ...] | None,
    warmup: int,
    repeats: int,
) -> dict[str, Any]:
    address = phx.discretization.spatial.MortonAddressPlan(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), depth
    )
    schedule_plan, schedule_metadata_seconds = measure_host(
        lambda: MortonPlaneSchedulePlan(
            address,
            positions.shape[0],
            maximum_leaf_occupancy=leaf_occupancy,
            coarsening_factor=coarsening_factor,
            target_top_nodes=target_top_nodes,
        )
    )
    leaf_capacity = schedule_plan.plane_capacities[0]
    maximum_pairs = max(1, leaf_capacity * leaf_capacity)
    interaction_plan, interaction_metadata_seconds = measure_host(
        lambda: MortonPlaneInteractionPlan(
            schedule_plan,
            opening_angle=opening_angle,
            queue_capacity=maximum_pairs if queue_capacity is None else queue_capacity,
            far_capacity=maximum_pairs if far_capacity is None else far_capacity,
            near_capacity=maximum_pairs if near_capacity is None else near_capacity,
            frontier_widths=frontier_widths,
        )
    )
    schedule, preparation = _phase(
        schedule_plan.build, positions, warmup=warmup, repeats=repeats
    )
    interactions, interaction_preparation = _phase(
        interaction_plan.build, schedule, warmup=warmup, repeats=repeats
    )
    unchanged, refresh_unchanged = _phase(
        schedule_plan.refresh, schedule, positions, warmup=warmup, repeats=repeats
    )
    displacement = jnp.asarray(1.0e-5, dtype=positions.dtype) * jax.random.normal(
        jax.random.key(15000 + positions.shape[0]), positions.shape, dtype=positions.dtype
    )
    moved_positions = synchronize(positions + displacement)
    moved, refresh_moved = _phase(
        schedule_plan.refresh,
        schedule,
        moved_positions,
        warmup=warmup,
        repeats=repeats,
    )
    evidence = interactions.evidence
    active_by_plane = [int(value) for value in schedule.plane_active_counts]
    seed_pairs = active_by_plane[-1] ** 2
    required_queue = int(evidence.required_queue)
    far_degrees = jnp.bincount(
        interactions.far.target_indices,
        weights=interactions.far.valid.astype(jnp.int32),
        length=schedule_plan.node_capacity,
    )
    near_degrees = jnp.bincount(
        interactions.near.target_indices,
        weights=interactions.near.valid.astype(jnp.int32),
        length=schedule_plan.node_capacity,
    )
    return {
        "schedule_plan_id": schedule_plan.plan_id,
        "interaction_plan_id": interaction_plan.plan_id,
        "leaf_occupancy": leaf_occupancy,
        "coarsening_factor": coarsening_factor,
        "target_top_nodes": target_top_nodes,
        "opening_angle": opening_angle,
        "host_metadata_seconds": {
            "schedule": schedule_metadata_seconds,
            "interactions": interaction_metadata_seconds,
        },
        "phases": {
            "schedule_preparation": preparation,
            "interaction_preparation": interaction_preparation,
            "refresh_unchanged": refresh_unchanged,
            "refresh_moved": refresh_moved,
        },
        "retained_bytes": {
            "schedule": logical_array_bytes(schedule),
            "interactions": logical_array_bytes(interactions),
            "far_relation": logical_array_bytes(interactions.far),
            "near_relation": logical_array_bytes(interactions.near),
        },
        "capacities": {
            "nodes": schedule_plan.node_capacity,
            "required_nodes": int(schedule.evidence.required_nodes),
            "live_nodes": int(schedule.evidence.active_nodes),
            "queue": int(evidence.queue_capacity),
            "required_queue": required_queue,
            "far": int(evidence.far_capacity),
            "required_far": int(evidence.required_far),
            "live_far": int(jnp.sum(interactions.far.valid)),
            "near": int(evidence.near_capacity),
            "required_near": int(evidence.required_near),
            "live_near": int(jnp.sum(interactions.near.valid)),
        },
        "frontier": {
            "supported_widths": list(interaction_plan.frontier_widths),
            "seed_live_pairs": seed_pairs,
            "seed_fill_fraction": seed_pairs / interaction_plan.queue_capacity,
            "peak_required_pairs": required_queue,
            "peak_fill_fraction": required_queue / interaction_plan.queue_capacity,
            "active_nodes_by_plane": active_by_plane,
            "same_plane_pair_upper_bounds": [value * value for value in active_by_plane],
            "note": "Seed and required peak are observed; per-plane pair counts are bounds, not traversal samples.",
        },
        "accepted_target_degrees": {
            "maximum_far": int(jnp.max(far_degrees, initial=0)),
            "maximum_near": int(jnp.max(near_degrees, initial=0)),
            "far_receivers": int(jnp.count_nonzero(far_degrees)),
            "near_receivers": int(jnp.count_nonzero(near_degrees)),
        },
        "refresh": {
            "unchanged_accepted": bool(unchanged.accepted_candidate),
            "unchanged_refitted": bool(unchanged.refitted),
            "unchanged_rebuilt": bool(unchanged.rebuilt),
            "moved_accepted": bool(moved.accepted_candidate),
            "moved_refitted": bool(moved.refitted),
            "moved_rebuilt": bool(moved.rebuilt),
        },
        "maximum_accepted_ratio": float(evidence.maximum_accepted_ratio),
        "overflow": {
            "queue": bool(evidence.queue_overflow),
            "far": bool(evidence.far_overflow),
            "near": bool(evidence.near_overflow),
        },
        "complete": bool(evidence.complete),
        "finite": bool(evidence.finite),
        "successful": bool(
            evidence.successful & unchanged.accepted_candidate & moved.accepted_candidate
        ),
    }


def _case(
    count: int,
    distribution: str,
    *,
    dtype: Any,
    warmup: int,
    repeats: int,
    target_top_nodes: Sequence[int],
    leaf_occupancy: int,
    coarsening_factor: int,
    opening_angle: float,
    queue_capacity: int | None,
    far_capacity: int | None,
    near_capacity: int | None,
    frontier_widths: tuple[int, ...] | None,
) -> dict[str, Any]:
    positions = _positions(count, distribution, dtype)
    masses = jnp.full((count,), 1.0 / count, dtype=positions.dtype)
    cotangent = jax.random.normal(
        jax.random.key(16000 + count), positions.shape, dtype=positions.dtype
    )
    synchronize((positions, masses, cotangent))
    depth = min(10, max(2, math.ceil(math.log2(count) / 3.0) + 2))
    tree_plan = phx.applications.cosmology.ParticleOctreePlan3D(
        (1.0, 1.0, 1.0), depth, target_leaf_occupancy=1
    )
    tree, preparation = _phase(
        tree_plan.prepare, positions, masses, warmup=warmup, repeats=repeats
    )
    barnes_hut = phx.applications.cosmology.BarnesHutGravityPlan(
        1.0,
        softening=0.01,
        opening_angle=0.5,
        use_quadrupole=True,
    )
    result, evaluation = _phase(barnes_hut.evaluate, tree, warmup=warmup, repeats=repeats)

    def total(current_positions: jax.Array, strengths: jax.Array) -> Any:
        return barnes_hut.evaluate(tree_plan.prepare(current_positions, strengths))

    _, total_evaluation = _phase(total, positions, masses, warmup=warmup, repeats=repeats)

    def position_vjp(
        current_positions: jax.Array, strengths: jax.Array, gradient_bar: jax.Array
    ) -> jax.Array:
        def acceleration(values: jax.Array) -> jax.Array:
            return total(values, strengths).acceleration

        _, pullback = jax.vjp(acceleration, current_positions)
        return pullback(gradient_bar)[0]

    derivative, transformed = _phase(
        position_vjp,
        positions,
        masses,
        cotangent,
        warmup=warmup,
        repeats=repeats,
    )

    def direct(current_positions: jax.Array, strengths: jax.Array) -> jax.Array:
        return _reference(current_positions, strengths, 0.01)

    reference, direct_evaluation = _phase(
        direct, positions, masses, warmup=warmup, repeats=repeats
    )
    relative_error = float(
        jnp.linalg.norm(result.acceleration - reference)
        / jnp.maximum(jnp.linalg.norm(reference), 1.0e-15)
    )
    interaction_cases = [
        _interaction_case(
            positions,
            depth,
            target_top_nodes=top_nodes,
            leaf_occupancy=leaf_occupancy,
            coarsening_factor=coarsening_factor,
            opening_angle=opening_angle,
            queue_capacity=queue_capacity,
            far_capacity=far_capacity,
            near_capacity=near_capacity,
            frontier_widths=frontier_widths,
            warmup=warmup,
            repeats=repeats,
        )
        for top_nodes in target_top_nodes
    ]
    transformed_finite = bool(jnp.all(jnp.isfinite(derivative)))
    return {
        "distribution": distribution,
        "particles": count,
        "dtype": str(positions.dtype),
        "depth": depth,
        "active_mask": "all-active",
        "tree_plan_id": tree_plan.plan_id,
        "gravity_backend": "jax",
        "derivative_execution": "jax-reference",
        "active_nodes": int(tree.hierarchy.evidence.active_nodes),
        "active_leaves": int(tree.hierarchy.evidence.active_leaves),
        "hierarchy_bytes": logical_array_bytes(tree.hierarchy),
        "tree_bytes": logical_array_bytes(tree),
        "phases": {
            "tree_preparation": preparation,
            "evaluation": evaluation,
            "total": total_evaluation,
            "total_position_vjp": transformed,
            "direct": direct_evaluation,
        },
        "interaction_cases": interaction_cases,
        "steady_speedup_over_direct": (
            direct_evaluation["warm"]["median_seconds"]
            / evaluation["warm"]["median_seconds"]
        ),
        "accepted_nodes": int(result.evidence.accepted_leaf_interactions),
        "direct_interactions": int(result.evidence.direct_particle_interactions),
        "relative_error": relative_error,
        "transformed_finite": transformed_finite,
        "successful": bool(result.successful)
        and transformed_finite
        and all(case["successful"] for case in interaction_cases),
    }


def run_spatial_hierarchy_benchmarks(
    counts: Sequence[int],
    /,
    *,
    distributions: Sequence[str] = ("uniform", "clustered"),
    dtype: Literal["float32", "float64"] = "float64",
    warmup: int = 1,
    repeats: int = 5,
    target_top_nodes: Sequence[int] = (1, 16),
    leaf_occupancy: int = 4,
    coarsening_factor: int = 4,
    opening_angle: float = 0.5,
    queue_capacity: int | None = None,
    far_capacity: int | None = None,
    near_capacity: int | None = None,
    frontier_widths: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    dtype = parse(dtype, Literal["float32", "float64"], "dtype")
    if not counts or any(count < 1 for count in counts):
        raise ValueError("Particle counts must be nonempty and positive.")
    if not distributions or any(
        distribution not in ("uniform", "clustered") for distribution in distributions
    ):
        raise ValueError("Distributions must be uniform or clustered.")
    if not target_top_nodes or any(value < 1 for value in target_top_nodes):
        raise ValueError("Top-node bounds must be nonempty and positive.")
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive.")
    if dtype == "float64" and not pxconfig.x64_enabled():
        raise ValueError("Float64 benchmarks require JAX_ENABLE_X64=true.")
    array_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    cases = [
        _case(
            count,
            distribution,
            dtype=array_dtype,
            warmup=warmup,
            repeats=repeats,
            target_top_nodes=target_top_nodes,
            leaf_occupancy=leaf_occupancy,
            coarsening_factor=coarsening_factor,
            opening_angle=opening_angle,
            queue_capacity=queue_capacity,
            far_capacity=far_capacity,
            near_capacity=near_capacity,
            frontier_widths=frontier_widths,
        )
        for distribution in distributions
        for count in counts
    ]
    report = {
        "kind": "spatial-hierarchy-benchmark",
        "device": str(jax.devices()[0]),
        "environment": capture_environment().to_dict(),
        "command": [sys.executable, *sys.argv],
        "cache_policy": "current-process-cache; separate lowering/compile/first/warm",
        "warmup": warmup,
        "repeats": repeats,
        "cases": cases,
        "passed": all(
            case["successful"]
            and math.isfinite(case["relative_error"])
            and case["relative_error"] < 0.25
            for case in cases
        ),
    }
    report["benchmark_identity"] = capture_benchmark_identity(
        Path(__file__).resolve().parents[1], Path(__file__), report.keys()
    ).to_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark sparse hierarchies and bounded plane interaction preparation."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--points", nargs="+", type=int)
    parser.add_argument("--distributions", nargs="+", choices=("uniform", "clustered"))
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--target-top-nodes", nargs="+", type=int, default=(1, 16))
    parser.add_argument("--leaf-occupancy", type=int, default=4)
    parser.add_argument("--coarsening-factor", type=int, default=4)
    parser.add_argument("--opening-angle", type=float, default=0.5)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--far-capacity", type=int)
    parser.add_argument("--near-capacity", type=int)
    parser.add_argument("--frontier-widths", nargs="+", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = run_spatial_hierarchy_benchmarks(
        arguments.points or ((32, 64) if arguments.smoke else (64, 256, 1024)),
        distributions=arguments.distributions or ("uniform", "clustered"),
        dtype=arguments.dtype,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        target_top_nodes=arguments.target_top_nodes,
        leaf_occupancy=arguments.leaf_occupancy,
        coarsening_factor=arguments.coarsening_factor,
        opening_angle=arguments.opening_angle,
        queue_capacity=arguments.queue_capacity,
        far_capacity=arguments.far_capacity,
        near_capacity=arguments.near_capacity,
        frontier_widths=(
            None
            if arguments.frontier_widths is None
            else tuple(arguments.frontier_widths)
        ),
    )
    if arguments.output is not None:
        write_json_atomic(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
