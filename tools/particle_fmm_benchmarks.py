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
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
    synchronize,
)
from phydrax import config as pxconfig
from phydrax.solver._particle_gravity import (
    CartesianFMMOperators,
    PreparedUniformFMMStructure,
)
from phydrax.typing import parse


def _phase(
    function: Callable[..., Any],
    *arguments: Any,
    warmup: int,
    repeats: int,
) -> tuple[Any, dict[str, Any]]:
    """Keep compilation, first execution, and synchronized warm samples distinct."""
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
    key = jax.random.key(7300 + count)
    if distribution == "uniform":
        return 0.02 + 0.96 * jax.random.uniform(key, (count, 3), dtype=dtype)
    centers = jnp.asarray([[0.18, 0.18, 0.18], [0.82, 0.82, 0.82]], dtype=dtype)
    assignment = jnp.arange(count, dtype=jnp.int32) % 2
    noise = 0.035 * jax.random.normal(key, (count, 3), dtype=dtype)
    return jnp.clip(centers[assignment] + noise, 0.02, 0.98)


def _far_field(
    plan: phx.solver.UniformFMMPlan,
    structure: PreparedUniformFMMStructure,
    strengths: jax.Array,
    /,
) -> tuple[jax.Array, jax.Array]:
    """Far potential and gradient, including P2M/M2M and L2P, in storage order."""
    operators = CartesianFMMOperators(
        plan.expansion, plan.gravitational_constant, plan.softening
    )
    schedule = structure.schedule
    order = schedule.point_order.storage_to_logical
    positions = schedule.point_order.sorted_coordinates
    leaf = jnp.maximum(schedule.sorted_point_leaf_slots, 0)
    scales = schedule.node_scales[leaf]
    relative = positions - schedule.node_centers[leaf]
    sorted_strengths = jnp.where(
        schedule.point_order.sorted_active, strengths[order], 0.0
    )
    multipoles, _ = plan._upward_pass(
        structure, relative / scales[:, None], sorted_strengths, operators
    )
    local, _, _ = plan._downward_pass(structure, multipoles, operators)
    potential, gradient = jax.vmap(
        lambda value, offset, scale: operators.l2p(value, offset, scale=scale)
    )(local[leaf], relative, scales)
    active = schedule.point_order.sorted_active
    return jnp.where(active, -potential, 0.0), jnp.where(active[:, None], gradient, 0.0)


def _case(
    count: int,
    distribution: str,
    order: int,
    *,
    backend: Literal["jax", "pallas"],
    pallas_interpret: bool,
    dtype: Any,
    warmup: int,
    repeats: int,
    depth: int,
    opening_angle: float,
    leaf_occupancy: int,
    maximum_nodes: int | None,
    queue_capacity: int | None,
    far_capacity: int | None,
    near_capacity: int | None,
) -> dict[str, Any]:
    positions = _positions(count, distribution, dtype)
    masses = 0.5 + jax.random.uniform(
        jax.random.key(9100 + count), (count,), dtype=positions.dtype
    )
    signed_strengths = masses * jnp.where(
        jnp.arange(count, dtype=jnp.int32) % 2 == 0, 1.0, -1.0
    )
    cotangent = jax.random.normal(
        jax.random.key(12100 + count), positions.shape, dtype=positions.dtype
    )
    potential_cotangent = jax.random.normal(
        jax.random.key(13100 + count), masses.shape, dtype=positions.dtype
    )
    synchronize((positions, masses, signed_strengths, cotangent, potential_cotangent))
    softening = 0.015
    tree_plan = phx.solver.ParticleOctreePlan3D((1.0, 1.0, 1.0), depth)
    direct_plan = phx.solver.DirectParticleGravityPlan(
        phx.solver.NewtonianPairKernel(1.0, softening=softening)
    )
    fmm_plan = phx.solver.UniformFMMPlan(
        1.0,
        phx.solver.CartesianExpansionSpace(order),
        softening=softening,
        opening_angle=opening_angle,
        maximum_leaf_occupancy=leaf_occupancy,
        coarsening_factor=4,
        target_top_nodes=1,
        maximum_nodes=maximum_nodes,
        maximum_queue_interactions=queue_capacity,
        maximum_far_interactions=far_capacity,
        maximum_near_interactions=near_capacity,
        execution_backend=backend,
        pallas_interpret=pallas_interpret,
    )
    phases: dict[str, Any] = {}

    def measure(function: Callable[..., Any], *values: Any) -> tuple[Any, dict[str, Any]]:
        return _phase(function, *values, warmup=warmup, repeats=repeats)

    tree, phases["tree_preparation"] = measure(tree_plan.prepare, positions, masses)

    def prepare(current_positions: jax.Array) -> PreparedUniformFMMStructure:
        return fmm_plan.prepare_structure(
            current_positions, box_size=(1.0, 1.0, 1.0), depth=depth
        )

    structure, phases["structure_preparation"] = measure(prepare, positions)

    def refresh(
        previous: PreparedUniformFMMStructure, current_positions: jax.Array
    ) -> Any:
        return fmm_plan.refresh_structure(
            previous, current_positions, box_size=(1.0, 1.0, 1.0), depth=depth
        )

    transition, phases["refresh_unchanged"] = measure(refresh, structure, positions)
    moved_positions = positions + jnp.asarray(1.0e-5, dtype=positions.dtype) * cotangent
    synchronize(moved_positions)
    moved, phases["refresh_moved"] = measure(refresh, structure, moved_positions)
    operators = CartesianFMMOperators(fmm_plan.expansion, 1.0, softening)
    schedule = structure.schedule
    sorted_logical = schedule.point_order.storage_to_logical
    sorted_positions = schedule.point_order.sorted_coordinates
    sorted_strengths = jnp.where(
        schedule.point_order.sorted_active, signed_strengths[sorted_logical], 0.0
    )
    safe_leaf = jnp.maximum(schedule.sorted_point_leaf_slots, 0)
    normalized_relative = (
        sorted_positions - schedule.node_centers[safe_leaf]
    ) / schedule.node_scales[safe_leaf, None]
    synchronize((sorted_positions, sorted_strengths, normalized_relative))
    upward, phases["upward"] = measure(
        fmm_plan._upward_pass,
        structure,
        normalized_relative,
        sorted_strengths,
        operators,
    )
    _, phases["far_translation"] = measure(
        fmm_plan._downward_pass, structure, upward[0], operators
    )
    _, phases["far"] = measure(_far_field, fmm_plan, structure, signed_strengths)

    def near(
        current_structure: PreparedUniformFMMStructure,
        storage_positions: jax.Array,
        storage_strengths: jax.Array,
    ) -> Any:
        return fmm_plan._near_field(
            current_structure,
            storage_positions,
            storage_strengths,
            with_potential=True,
        )

    _, phases["near"] = measure(near, structure, sorted_positions, sorted_strengths)
    monopole, phases["prepared_total"] = measure(
        fmm_plan.evaluate_monopole, structure, signed_strengths
    )
    fmm_result, phases["total"] = measure(fmm_plan.evaluate, tree)
    direct_result, phases["direct"] = measure(direct_plan.evaluate, positions, masses)

    def near_position_vjp(
        current_positions: jax.Array,
        current_structure: PreparedUniformFMMStructure,
        strengths: jax.Array,
        gradient_bar: jax.Array,
        potential_bar: jax.Array,
    ) -> jax.Array:
        storage_order = current_structure.schedule.point_order.storage_to_logical
        active = current_structure.schedule.point_order.sorted_active
        storage_strengths = jnp.where(active, strengths[storage_order], 0.0)

        def field(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            gradient, potential, _, _ = near(
                current_structure, values[storage_order], storage_strengths
            )
            if potential is None:
                raise RuntimeError("Near monopole evaluation did not return a potential.")
            return potential, gradient

        _, pullback = jax.vjp(field, current_positions)
        return pullback((potential_bar[storage_order], gradient_bar[storage_order]))[0]

    def far_position_vjp(
        current_positions: jax.Array,
        current_structure: PreparedUniformFMMStructure,
        strengths: jax.Array,
        gradient_bar: jax.Array,
        potential_bar: jax.Array,
    ) -> jax.Array:
        def field(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            refreshed = refresh(current_structure, values).accepted
            return _far_field(fmm_plan, refreshed, strengths)

        storage_order = current_structure.schedule.point_order.storage_to_logical
        _, pullback = jax.vjp(field, current_positions)
        return pullback((potential_bar[storage_order], gradient_bar[storage_order]))[0]

    def prepared_position_vjp(
        current_positions: jax.Array,
        current_structure: PreparedUniformFMMStructure,
        strengths: jax.Array,
        gradient_bar: jax.Array,
        potential_bar: jax.Array,
    ) -> jax.Array:
        def field(values: jax.Array) -> tuple[jax.Array, jax.Array]:
            refreshed = refresh(current_structure, values).accepted
            result = fmm_plan.evaluate_monopole(refreshed, strengths)
            return result.potential, result.gradient

        _, pullback = jax.vjp(field, current_positions)
        return pullback((potential_bar, gradient_bar))[0]

    def total_position_vjp(
        current_positions: jax.Array, strengths: jax.Array, gradient_bar: jax.Array
    ) -> jax.Array:
        def acceleration(values: jax.Array) -> jax.Array:
            return fmm_plan.evaluate(tree_plan.prepare(values, strengths)).acceleration

        _, pullback = jax.vjp(acceleration, current_positions)
        return pullback(gradient_bar)[0]

    transformed: dict[str, jax.Array] = {}
    for name, function in (
        ("near_position_vjp", near_position_vjp),
        ("far_position_vjp", far_position_vjp),
        ("prepared_total_position_vjp", prepared_position_vjp),
    ):
        transformed[name], phases[name] = measure(
            function,
            positions,
            structure,
            signed_strengths,
            cotangent,
            potential_cotangent,
        )
    transformed["total_position_vjp"], phases["total_position_vjp"] = measure(
        total_position_vjp, positions, masses, cotangent
    )
    reference = direct_result[0]
    absolute = jnp.sqrt(jnp.sum((fmm_result.acceleration - reference) ** 2, axis=-1))
    reference_norm = jnp.sqrt(jnp.sum(reference * reference, axis=-1))
    relative = absolute / jnp.maximum(reference_norm, 1.0e-14)
    resources = fmm_result.fmm_evidence
    if resources is None:
        raise RuntimeError("FMM evaluation did not return resource evidence.")
    interaction_evidence = structure.interactions.evidence
    transformed_finite = all(
        bool(jnp.all(jnp.isfinite(value))) for value in transformed.values()
    )
    return {
        "points": count,
        "distribution": distribution,
        "order": order,
        "dtype": str(positions.dtype),
        "backend": fmm_plan.execution_backend,
        "pallas_interpret": pallas_interpret,
        "derivative_execution": "jax-reference",
        "plan_id": fmm_plan.plan_id,
        "structure_id": structure.structure_id,
        "depth": depth,
        "active_mask": "all-active",
        "strengths": "positive-masses-and-alternating-signed-monopoles",
        "softening": softening,
        "opening_angle": opening_angle,
        "leaf_occupancy": leaf_occupancy,
        "phases": phases,
        "phase_scopes": {
            "far_translation": "M2L and L2L only; precomputed multipoles",
            "far": "P2M/M2M/M2L/L2L/L2P potential and gradient; prepared geometry",
            "near": "streamed P2P gradient and potential",
            "prepared_total": "signed monopole potential and gradient; prepared geometry",
            "total": "public rematerializing evaluate; includes structure preparation",
            "near_position_vjp": "P2P potential/gradient cotangents; frozen support",
            "far_position_vjp": "geometry refresh, P2M/M2M/M2L/L2L/L2P and potential/gradient VJP",
            "prepared_total_position_vjp": "geometry refresh and signed monopole potential/gradient VJP",
            "total_position_vjp": "tree preparation and public acceleration evaluate VJP",
        },
        "refresh": {
            "unchanged_successful": bool(transition.successful),
            "unchanged_reused_topology": bool(transition.reused_topology),
            "moved_successful": bool(moved.successful),
            "moved_reused_topology": bool(moved.reused_topology),
            "maximum_displacement": float(jnp.max(jnp.abs(moved_positions - positions))),
        },
        "retained_bytes": {
            "tree": logical_array_bytes(tree),
            "structure": logical_array_bytes(structure),
            "interactions": logical_array_bytes(structure.interactions),
            "near_stream": logical_array_bytes(structure.near_execution),
            "far_stream": logical_array_bytes(structure.far_execution),
        },
        "capacities": {
            "nodes": structure.schedule_plan.node_capacity,
            "required_nodes": int(schedule.evidence.required_nodes),
            "queue": int(interaction_evidence.queue_capacity),
            "required_queue": int(interaction_evidence.required_queue),
            "far": int(resources.far_capacity),
            "required_far": int(resources.required_far),
            "live_far": int(jnp.sum(structure.interactions.far.valid)),
            "near": int(resources.near_capacity),
            "required_near": int(resources.required_near),
            "live_near": int(jnp.sum(structure.interactions.near.valid)),
        },
        "steady_speedup_over_direct": (
            phases["direct"]["warm"]["median_seconds"]
            / phases["total"]["warm"]["median_seconds"]
        ),
        "maximum_relative_error": float(jnp.max(relative, initial=0.0)),
        "rms_relative_error": float(jnp.sqrt(jnp.mean(relative * relative))),
        "active_nodes": int(fmm_result.evidence.active_nodes),
        "far_interactions": int(resources.required_far),
        "near_interactions": int(resources.required_near),
        "direct_particle_interactions": int(resources.p2p_count),
        "minimum_scale_exponent": float(resources.minimum_scale_exponent),
        "maximum_scale_exponent": float(resources.maximum_scale_exponent),
        "net_force_norm": float(jnp.sqrt(jnp.sum(fmm_result.evidence.net_force**2))),
        "transformed_finite": transformed_finite,
        "successful": bool(
            fmm_result.successful
            & monopole.successful
            & transition.successful
            & moved.successful
        )
        and transformed_finite,
    }


def run_particle_fmm_benchmarks(
    counts: Sequence[int],
    orders: Sequence[int],
    /,
    *,
    distributions: Sequence[str] = ("uniform", "clustered"),
    backend: Literal["jax", "pallas"] = "jax",
    pallas_interpret: bool = False,
    dtype: Literal["float32", "float64"] = "float64",
    warmup: int = 1,
    repeats: int = 5,
    depth: int = 8,
    opening_angle: float = 0.5,
    leaf_occupancy: int = 4,
    maximum_nodes: int | None = None,
    queue_capacity: int | None = None,
    far_capacity: int | None = None,
    near_capacity: int | None = None,
) -> dict[str, Any]:
    backend = parse(backend, Literal["jax", "pallas"], "backend")
    dtype = parse(dtype, Literal["float32", "float64"], "dtype")
    if not counts or any(count < 1 for count in counts):
        raise ValueError("Particle counts must be nonempty and positive.")
    if not orders or any(order < 0 for order in orders):
        raise ValueError("Expansion orders must be nonempty and nonnegative.")
    if not distributions or any(
        distribution not in ("uniform", "clustered") for distribution in distributions
    ):
        raise ValueError("Distributions must be uniform or clustered.")
    if warmup < 0 or repeats < 1:
        raise ValueError("Warmup must be nonnegative and repeats must be positive.")
    if pallas_interpret and backend != "pallas":
        raise ValueError("--pallas-interpret requires --backend pallas.")
    array_dtype = jnp.float64 if dtype == "float64" else jnp.float32
    if dtype == "float64" and not pxconfig.x64_enabled():
        raise ValueError("Float64 benchmarks require JAX_ENABLE_X64=true.")
    cases = [
        _case(
            count,
            distribution,
            order,
            backend=backend,
            pallas_interpret=pallas_interpret,
            dtype=array_dtype,
            warmup=warmup,
            repeats=repeats,
            depth=depth,
            opening_angle=opening_angle,
            leaf_occupancy=leaf_occupancy,
            maximum_nodes=maximum_nodes,
            queue_capacity=queue_capacity,
            far_capacity=far_capacity,
            near_capacity=near_capacity,
        )
        for distribution in distributions
        for count in counts
        for order in orders
    ]
    maximum_relative_error = 0.25
    maximum_rms_relative_error = 0.1
    report = {
        "kind": "particle-cartesian-fmm-benchmark",
        "device": str(jax.devices()[0]),
        "environment": capture_environment().to_dict(),
        "command": [sys.executable, *sys.argv],
        "cache_policy": "current-process-cache; separate lowering/compile/first/warm",
        "warmup": warmup,
        "repeats": repeats,
        "maximum_relative_error": maximum_relative_error,
        "maximum_rms_relative_error": maximum_rms_relative_error,
        "cases": cases,
        "passed": all(
            case["successful"]
            and math.isfinite(case["maximum_relative_error"])
            and math.isfinite(case["rms_relative_error"])
            and case["maximum_relative_error"] <= maximum_relative_error
            and case["rms_relative_error"] <= maximum_rms_relative_error
            for case in cases
        ),
    }
    report["benchmark_identity"] = capture_benchmark_identity(
        Path(__file__).resolve().parents[1], Path(__file__), report.keys()
    ).to_dict()
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark streamed Cartesian particle FMM phases and position VJPs."
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--points", nargs="+", type=int)
    parser.add_argument("--orders", nargs="+", type=int)
    parser.add_argument("--distributions", nargs="+", choices=("uniform", "clustered"))
    parser.add_argument("--backend", choices=("jax", "pallas"), default="jax")
    parser.add_argument("--pallas-interpret", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--opening-angle", type=float, default=0.5)
    parser.add_argument("--leaf-occupancy", type=int, default=4)
    parser.add_argument("--maximum-nodes", type=int)
    parser.add_argument("--queue-capacity", type=int)
    parser.add_argument("--far-capacity", type=int)
    parser.add_argument("--near-capacity", type=int)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    counts = arguments.points or ((8, 16) if arguments.smoke else (32, 64, 128))
    orders = arguments.orders or ((1, 3) if arguments.smoke else (1, 3, 5))
    report = run_particle_fmm_benchmarks(
        counts,
        orders,
        distributions=arguments.distributions or ("uniform", "clustered"),
        backend=arguments.backend,
        pallas_interpret=arguments.pallas_interpret,
        dtype=arguments.dtype,
        warmup=arguments.warmup,
        repeats=arguments.repeats,
        depth=arguments.depth,
        opening_angle=arguments.opening_angle,
        leaf_occupancy=arguments.leaf_occupancy,
        maximum_nodes=arguments.maximum_nodes,
        queue_capacity=arguments.queue_capacity,
        far_capacity=arguments.far_capacity,
        near_capacity=arguments.near_capacity,
    )
    if arguments.output is not None:
        write_json_atomic(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
