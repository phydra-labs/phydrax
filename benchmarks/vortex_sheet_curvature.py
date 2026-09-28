"""Bounded vortex-sheet curvature routing across vertex and region capacities."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from _runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
)

from phydrax.applications.foams import (
    FoamDynamicsPlan,
    FoamDynamicsState,
    FoamMaterialPlan,
    PreparedFoamDynamics,
    PreparedVortexSheetAir,
    RegionPressureAirPlan,
    VortexSheetAirPlan,
)
from phydrax.geometry.multiregion_surface import (
    MultiRegionSurfaceCapacityPlan,
    PreparedMultiRegionSurface,
    seed_sphere,
)


def _prepare(
    subdivisions: int, region_capacity: int, /
) -> tuple[PreparedVortexSheetAir, FoamDynamicsState, float]:
    seed = seed_sphere(1.0, subdivisions=subdivisions)
    base = seed.capacity_plan(resource_id="vortex-sheet-curvature-benchmark")
    capacity = MultiRegionSurfaceCapacityPlan(
        vertex_capacity=base.vertex_capacity,
        edge_capacity=base.edge_capacity,
        face_capacity=base.face_capacity,
        region_capacity=region_capacity,
        region_pair_capacity=base.region_pair_capacity,
        maximum_edge_valence=base.maximum_edge_valence,
        maximum_vertex_region_pairs=base.maximum_vertex_region_pairs,
        resource_id="vortex-sheet-curvature-benchmark",
        coordinate_dtype=base.coordinate_dtype,
        index_dtype=base.index_dtype,
    )
    started = time.perf_counter()
    topology = seed.topology(capacity)
    surface_state = seed.state(topology)
    surface = PreparedMultiRegionSurface(topology, surface_state)
    volumes = surface.region_volumes(surface_state.positions)[
        jnp.asarray(topology.finite_region_indices, dtype=jnp.int32)
    ]
    dynamics_state = FoamDynamicsState(surface_state)
    dynamics = PreparedFoamDynamics(
        FoamDynamicsPlan(
            route="film-inertia",
            time_step=1.0e-5,
            areal_mass=1.0,
            volume_tolerance=1.0e-10,
        ),
        surface,
        FoamMaterialPlan.soap_film(topology.region_ids, 0.025),
        RegionPressureAirPlan.incompressible(volumes),
        dynamics_state,
    )
    prepared = VortexSheetAirPlan(
        air_density=1.2,
        time_step=1.0e-5,
        core_radius_fraction=0.4,
        fmm_depth=1,
        fmm_leaf_capacity=64,
        maximum_fmm_relative_error=0.15,
    ).prepare(dynamics, dynamics_state)
    points = dynamics_state.surface.positions
    radius = jnp.linalg.norm(points, axis=1)
    cosine = jnp.where(radius > 0.0, points[:, 2] / radius, 0.0)
    mode = 0.5 * (3.0 * cosine * cosine - 1.0)
    deformed = points * (1.0 + 0.03 * mode)[:, None]
    benchmark_state = FoamDynamicsState(
        dynamics_state.surface.with_positions(deformed)
    )
    return prepared, benchmark_state, time.perf_counter() - started


def _measure(
    subdivisions: int,
    region_capacity: int,
    warmup: int,
    repeats: int,
    /,
) -> dict[str, Any]:
    prepared, state, preparation_seconds = _prepare(subdivisions, region_capacity)
    dynamic, static = eqx.partition((prepared, state), eqx.is_array)

    def evaluate(leaves: Any) -> Any:
        current, current_state = eqx.combine(leaves, static)
        return current.circulation_source(current_state)

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic), lambda lowered: lowered.compile()
    )
    source, execution = measure_repeated(
        lambda: compiled(dynamic), warmup=warmup, repeats=repeats
    )
    compiler = compiler_evidence(
        compiled.cost_analysis(), compiled.memory_analysis(), source="xla"
    )
    topology = prepared.topology
    dtype_bytes = np.dtype(np.asarray(state.surface.positions).dtype).itemsize
    return {
        "subdivisions": subdivisions,
        "vertex_capacity": topology.vertex_capacity,
        "slot_width": topology.slot_width,
        "declared_region_capacity": topology.region_capacity,
        "curvature_target_entries": topology.vertex_capacity * topology.slot_width * 2,
        "dense_reference_bytes": topology.vertex_capacity
        * topology.region_capacity
        * dtype_bytes,
        "curvature_relation_routes": prepared.curvature_slots.capacity,
        "curvature_relation_logical_bytes": logical_array_bytes(
            (prepared.curvature_slots, prepared.curvature_route_regions)
        ),
        "curvature_relation_evidence_bytes": prepared.curvature_relation_evidence.logical_retained_bytes,
        "preparation_seconds": preparation_seconds,
        "lowering_seconds": compilation.lowering_seconds,
        "compilation_seconds": compilation.compilation_seconds,
        "execution": execution.to_seconds_dict(),
        "compiler": {
            "temporary_bytes": compiler.temporary_bytes,
            "output_bytes": compiler.output_bytes,
            "generated_code_bytes": compiler.generated_code_bytes,
        },
        "maximum_source": float(jnp.max(jnp.abs(source))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subdivisions", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--region-capacities", type=int, nargs="+", default=[2, 128])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    cases = [
        _measure(subdivision, region_capacity, args.warmup, args.repeats)
        for subdivision in args.subdivisions
        for region_capacity in args.region_capacities
    ]
    payload = {
        "benchmark": "vortex-sheet-curvature",
        "environment": capture_environment().to_dict(),
        "cases": cases,
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    if args.output is not None:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
