#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

"""Forest AMR adaptation cycles: host adaptation, conservative transfer, jit reuse.

Every cycle advects a moving bump on a 2:1-balanced forest, refines around it,
coarsens behind it, and transfers the state conservatively.  The compiled step and
transfer have stable callable identity and receive worksets/routes as dynamic
arguments, so their executable counts equal the number of distinct capacity
buckets visited rather than the number of cycles.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    measure_host,
    measure_lower_and_compile,
    measure_repeated,
)


def _advect_step(
    workset: Any, geometry: Any, state: Any, velocity: Any, step: Any
) -> Any:
    """One explicit upwind step over the padded forest face list."""
    valid = workset.face_valid
    minus = jnp.where(valid, workset.face_minus, 0)
    plus = jnp.where(valid, workset.face_plus, 0)
    speed = velocity[jnp.where(valid, workset.face_axes, 0)]
    upwind = jnp.where(speed > 0.0, state[minus], state[plus])
    flux = jnp.where(valid, speed * upwind * geometry.face_measures, 0.0)
    change = (
        jnp.zeros_like(state).at[minus].add(-flux).at[plus].add(flux) / geometry.volumes
    )
    return jnp.where(workset.leaf_valid, state + step * change, 0.0)


_advect = jax.jit(_advect_step)


@jax.jit
def _transfer(routes: Any, values: Any) -> Any:
    result = routes.apply(values)
    return result.values, result.conservation_residual


def _plan(maximum_level: int, /) -> phx.discretization.ForestPlan:
    grid = phx.discretization.TensorGridPlan(
        (
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
            phx.discretization.UniformCellAxisSpec(4, periodic=True),
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    return phx.discretization.ForestPlan(
        grid,
        maximum_level=maximum_level,
        minimum_leaf_capacity=64,
        maximum_leaf_capacity=1 << 18,
    )


def _marks(topology: Any, center: Any, radius: Any, base_level: Any, /) -> np.ndarray:
    lower, upper = topology.reference_bounds()
    count = topology.leaf_count
    offset = 0.5 * (lower + upper)[:count] - center
    distance = np.linalg.norm(offset - np.round(offset), axis=1)
    levels = topology.leaf_levels()
    marks = np.zeros((topology.signature.leaf_capacity,), dtype=np.int8)
    inside = distance < radius
    marks[:count] = np.where(
        inside & (levels < topology.plan.maximum_level),
        1,
        np.where(~inside & (levels > base_level), -1, 0),
    )
    return marks


def _scale(maximum_level: int, cycles: int, /) -> dict[str, object]:
    plan = _plan(maximum_level)
    compiler = phx.discretization.ForestTopologyCompiler(plan)
    base_level = 1
    topology = compiler.initialize(base_level).topology
    velocity = jnp.asarray([1.0, 0.5])
    geometry = phx.discretization.forest_leaf_geometry(topology)
    lower, upper = topology.reference_bounds()
    centers = 0.5 * (lower + upper)
    state = jnp.where(
        topology.workset.leaf_valid,
        jnp.exp(-40.0 * jnp.sum((jnp.asarray(centers) - 0.3) ** 2, axis=1)),
        0.0,
    )
    # ty: ignore[unresolved-attribute]
    advect_before = _advect._cache_size()
    # ty: ignore[unresolved-attribute]
    transfer_before = _transfer._cache_size()
    adapt_seconds = []
    transition_seconds = []
    residuals = []
    leaf_counts = []
    workset_signatures = []
    transfer_signatures = []
    for cycle in range(cycles):
        step = 0.2 * float(np.min(plan.root_spacing)) / (2.0**maximum_level)
        state = _advect(topology.workset, geometry, state, velocity, step)
        center = np.asarray([0.3, 0.3]) + (cycle + 1) * np.asarray([0.05, 0.025])
        marks = _marks(topology, center, 0.15, base_level)
        result, seconds = measure_host(lambda: compiler.adapt(topology, marks))
        adapt_seconds.append(seconds)
        if not result.status.changed:
            continue
        transition, seconds = measure_host(
            lambda: phx.discretization.ForestFieldTransition(topology, result.topology)
        )
        transition_seconds.append(seconds)
        state, residual = _transfer(transition.routes, state)
        residuals.append(float(np.max(np.abs(np.asarray(residual)))))
        topology = result.topology
        geometry = phx.discretization.forest_leaf_geometry(topology)
        leaf_counts.append(topology.leaf_count)
        workset_signatures.append(topology.signature.signature_id)
        transfer_signatures.append(
            (
                transition.routes.relation.source_size,
                transition.routes.relation.target_size,
                transition.routes.relation.capacity,
            )
        )
    # A fresh wrapper measures cold lowering and compilation of the final bucket;
    # the cycle loop above only ever used the stable `_advect` entry point.
    compiled, compilation = measure_lower_and_compile(
        lambda: jax.jit(_advect_step).lower(
            topology.workset, geometry, state, velocity, 1.0e-3
        ),
        lambda lowered: lowered.compile(),
    )
    _, warmed = measure_repeated(
        lambda: _advect(topology.workset, geometry, state, velocity, 1.0e-3),
        warmup=2,
        repeats=10,
    )
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
        unavailable_reason="The selected JAX backend did not report compiler analysis.",
    )
    return {
        "maximum_level": maximum_level,
        "cycles": cycles,
        "adapted_cycles": len(leaf_counts),
        "leaf_counts": leaf_counts,
        "distinct_workset_signatures": len(set(workset_signatures)),
        "distinct_transfer_signatures": len(set(transfer_signatures)),
        # ty: ignore[unresolved-attribute]
        "advect_executables": _advect._cache_size() - advect_before,
        # ty: ignore[unresolved-attribute]
        "transfer_executables": _transfer._cache_size() - transfer_before,
        "maximum_conservation_residual": max(residuals, default=0.0),
        "adapt_seconds_median": float(np.median(adapt_seconds)),
        "transition_seconds_median": float(np.median(transition_seconds)),
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "warmed_step_seconds_median": warmed.median_seconds,
        "compiler": asdict(evidence),
    }


def benchmark(*, smoke: bool) -> dict[str, object]:
    levels = (3, 4) if smoke else (4, 5, 6)
    cycles = 6 if smoke else 24
    scales = [_scale(level, cycles) for level in levels]
    reused = all(
        # ty: ignore[unsupported-operator]
        scale["advect_executables"] <= scale["distinct_workset_signatures"] + 1
        # ty: ignore[unsupported-operator]
        and scale["transfer_executables"] <= scale["distinct_transfer_signatures"]
        # ty: ignore[unsupported-operator]
        and scale["maximum_conservation_residual"] <= 1.0e-12
        for scale in scales
    )
    return {
        "benchmark": "forest-amr-adaptation-cycles",
        "status": "pass" if reused else "fail",
        "scales": scales,
        "environment": capture_environment().to_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    report = benchmark(smoke=arguments.smoke)
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    print(payload)
    if arguments.output is not None:
        arguments.output.write_text(payload + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
