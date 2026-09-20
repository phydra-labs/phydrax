#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp

import phydrax as phx
from benchmarks._runtime import capture_environment, measure_host, measure_repeated
from tools.block_amr_advanced_qualification import _configuration


def benchmark(*, smoke: bool) -> dict[str, object]:
    compiled, entities, geometry_plan, _ = _configuration()
    compiler = phx.discretization.VariablePatchTopologyCompiler(compiled.topology.plan)
    tags = ((jnp.zeros((1, 4, 4), dtype="bool").at[0, 1, 1].set(True),),)
    _, topology_seconds = measure_host(
        lambda: compiler.compile(compiler.initial_topology(), tags)
    )
    _, entity_seconds = measure_host(
        lambda: phx.discretization.VariablePatchEntityComplexPlan(
            phx.discretization.BlockHierarchyCapacityPlan(
                ((32, 64, 32), (96, 160, 64)),
                (80, 64, 240, 160),
            )
        ).prepare(compiled.topology)
    )
    transfer, transfer_seconds = measure_host(
        lambda: phx.discretization.CompatibleEntityTransferFamily(
            entities[0],
            entities[1],
            compiled.topology.plan.levels[0].refinement_ratio,
            (256, 256, 256),
        )
    )
    mapped = eqx.filter_jit(lambda time: geometry_plan.state(time, revision=1))
    repeats = 2 if smoke else 10
    geometry, geometry_times = measure_repeated(
        lambda: mapped(jnp.asarray(0.25)),
        warmup=1,
        repeats=repeats,
    )
    entity_count = sum(
        entity.num_active
        for complex_ in entities
        for entity in complex_.complex.entity_sets
    )
    return {
        "status": "pass" if bool(geometry.valid) else "fail",
        "configuration": {
            "smoke": smoke,
            "levels": len(compiled.topology.levels),
            "shape_signatures": [
                [bucket.signature.envelope_shape for bucket in level.buckets]
                for level in compiled.topology.plan.levels
            ],
            "entity_count": entity_count,
        },
        "timing": {
            "topology_compile_seconds": topology_seconds,
            "entity_prepare_seconds": entity_seconds,
            "compatible_transfer_prepare_seconds": transfer_seconds,
            "geometry_stage_seconds": list(geometry_times.samples_seconds),
            "geometry_stage_median_seconds": geometry_times.median_seconds,
        },
        "evidence": {
            "topology_id": compiled.topology.topology_id,
            "entity_complex_ids": [value.complex_id for value in entities],
            "compatible_transfer_family_id": transfer.family_id,
            "geometry_plan_id": geometry_plan.plan_id,
            "maximum_gcl_defect": float(
                max(jnp.max(value) for level in geometry.gcl_defects for value in level)
            ),
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
