#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import capture_environment, measure_host, measure_repeated
from tools.block_amr_production_qualification import (
    _cut_plan,
    _resources,
    _topology,
    _topology_2d,
)


def benchmark(*, smoke: bool) -> dict[str, object]:
    topology = _topology()
    hierarchy = phx.discretization.canonicalize_patch_hierarchy(topology)
    resource, resource_seconds = measure_host(
        lambda: _resources().preflight(
            hierarchy,
            physical_component_count=5,
            dtype=np.float64,
        )
    )
    cut, cut_seconds = measure_host(lambda: _cut_plan().prepare())
    multivalued_body = phx.discretization.EmbeddedLevelSetBody(
        lambda points, time, args: (points[:, 0] - 0.3) * (points[:, 0] - 0.7),
        "benchmark-slab",
        9,
    )
    multivalued_plan = phx.discretization.MultivaluedCutCellPlan(
        topology,
        lambda points, time, args: points,
        "benchmark-identity-map",
        phx.discretization.EmbeddedLevelSetBodySet((multivalued_body,)),
        _resources(),
        subdivision=2 if smoke else 4,
    )
    multivalued, multivalued_seconds = measure_host(multivalued_plan.prepare)
    multivalued_2d_plan = phx.discretization.MultivaluedCutCell2DPlan(
        _topology_2d(),
        lambda points, time, args: points,
        "benchmark-identity-map-2d",
        phx.discretization.EmbeddedLevelSetBodySet((multivalued_body,)),
        _resources(),
        subdivision=2 if smoke else 4,
    )
    multivalued_2d, multivalued_2d_seconds = measure_host(multivalued_2d_plan.prepare)
    mapped_plan = phx.discretization.CanonicalMappedGeometryPlan(
        topology,
        phx.discretization.PatchCoordinateMapSet(
            lambda point, time, args: (1.0 + 0.02 * time) * point,
            "benchmark-dilation",
        ),
        quadrature_order=3,
        tolerance=2.0e-6,
    )
    mapped_kernel = eqx.filter_jit(lambda time: mapped_plan.evaluate(time, revision=1))
    mapped, mapped_times = measure_repeated(
        lambda: mapped_kernel(jnp.asarray(0.25)),
        warmup=1,
        repeats=2 if smoke else 10,
    )

    system = phx.equations.EulerSystem(3)
    discretization = cut.finite_volume_plan(
        component_names=system.component_names
    ).prepare()
    boundaries = phx.discretization.UnstructuredFiniteVolumeBoundarySet(
        discretization.boundary_patch_names,
        {
            name: (
                phx.discretization.SlipWallBoundary()
                if name.startswith("embedded-")
                else phx.discretization.ExtrapolationBoundary()
            )
            for name in discretization.boundary_patch_names
        },
    )
    dynamics = phx.equations.compile_conservation_problem(
        phx.equations.ConservationProblemIR(
            "production-benchmark",
            "state",
            system,
            boundaries,
        ),
        discretization,
        phx.discretization.UnstructuredFiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.RusanovFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
    )
    primitive = jnp.broadcast_to(
        jnp.asarray((1.0, 0.0, 0.0, 0.0, 1.0)),
        discretization.state_shape,
    )
    initial = runtime.initialize_state(
        system.primitive_to_conserved(primitive),
        0.0,
        1.0e-4,
    )
    advance_kernel = eqx.filter_jit(runtime.advance)
    advanced, advance_times = measure_repeated(
        lambda: advance_kernel(initial),
        warmup=1,
        repeats=2 if smoke else 10,
    )

    diffusion = phx.discretization.MultivaluedCutCellDiffusionPlan(cut, 1.0)
    diffusion_kernel = eqx.filter_jit(diffusion.apply)
    diffusion_value, diffusion_times = measure_repeated(
        lambda: diffusion_kernel(jnp.ones((diffusion.cell_count,))),
        warmup=1,
        repeats=2 if smoke else 20,
    )
    group = phx.execution.ExecutionRuntime.current().root_group
    partition = phx.discretization.DistributedCutCellPartitionPlan(
        group,
        max(1, cut.component_count),
    ).prepare(cut)
    canonical = jnp.zeros((cut.component_capacity, 2)).at[0].set(jnp.asarray((1.0, 2.0)))
    local_indices = partition.local_component_indices()
    local_values = canonical[jnp.asarray(local_indices, dtype=jnp.int32)]
    distributed, distributed_times = measure_repeated(
        lambda: partition.unpack_process_local(
            partition.pack_process_local(local_indices, local_values)
        ),
        warmup=1,
        repeats=2 if smoke else 20,
    )

    cache = phx.discretization.PatchExecutableCachePlan(_resources())
    signatures = cache.required_signatures(
        hierarchy,
        physical_component_count=2,
        method_id="production-benchmark-kernel",
        dtype=np.float64,
    )
    cache_result, compile_seconds = measure_host(
        lambda: cache.install(
            phx.discretization.PatchExecutableCacheState(),
            signatures,
            lambda signature: lambda values: values + 1.0,
            lambda signature: (
                (jnp.zeros(signature.state_shape, dtype=signature.dtype),),
                {},
            ),
        )
    )
    valid = (
        resource.valid
        and cut.evidence.valid
        and multivalued.evidence.valid
        and multivalued_2d.evidence.valid
        and bool(mapped.evidence.valid)
        and bool(advanced.accepted)
        and bool(jnp.all(jnp.isfinite(diffusion_value)))
        and bool(jnp.array_equal(distributed, local_values))
        and cache_result.changed
    )
    return {
        "status": "pass" if valid else "fail",
        "configuration": {
            "smoke": smoke,
            "dimension": 3,
            "cut_components": cut.component_count,
            "multivalued_components": multivalued.component_count,
            "multivalued_2d_components": multivalued_2d.component_count,
            "process_count": jax.process_count(),
            "device_count": jax.device_count(),
            "real_multihost": jax.process_count() > 1,
        },
        "timing": {
            "resource_preflight_seconds": resource_seconds,
            "single_cut_prepare_seconds": cut_seconds,
            "multivalued_cut_prepare_seconds": multivalued_seconds,
            "multivalued_2d_prepare_seconds": multivalued_2d_seconds,
            "mapped_metric_median_seconds": mapped_times.median_seconds,
            "finite_volume_advance_median_seconds": advance_times.median_seconds,
            "diffusion_action_median_seconds": diffusion_times.median_seconds,
            "distributed_roundtrip_median_seconds": distributed_times.median_seconds,
            "new_signature_compile_seconds": compile_seconds,
        },
        "memory": {
            "reserved_host_bytes": resource.reserved_host_bytes,
            "reserved_device_bytes": resource.reserved_device_bytes,
        },
        "evidence": {
            "cut_evidence_id": cut.evidence.evidence_id,
            "multivalued_evidence_id": multivalued.evidence.evidence_id,
            "multivalued_2d_evidence_id": multivalued_2d.evidence.evidence_id,
            "mapped_evidence_id": mapped.evidence.evidence_id,
            "partition_id": partition.partition_id,
            "cache_id": cache_result.state.cache_id,
        },
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
