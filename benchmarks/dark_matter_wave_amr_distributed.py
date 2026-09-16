from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from phydrax._execution_resources import ExecutionGroupSpec
from phydrax._execution_runtime import ExecutionGroup
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)


def _group(devices) -> ExecutionGroup:
    values = tuple(devices)
    return ExecutionGroup(
        ExecutionGroupSpec(
            f"wave-amr-benchmark-{len(values)}",
            tuple(sorted({device.process_index for device in values})),
            tuple((device.process_index, device.id) for device in values),
            mesh_axes=(("block_parts", len(values)),),
        ),
        values,
    )


def _problem(cells: int):
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(cells, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    base_blocks = cells // 4
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0,
                (4,),
                base_blocks,
                halo_width=1,
                refinement_ratio=2,
            ),
            phx.discretization.BlockLevelPlan(
                1,
                (4,),
                max(2 * base_blocks, 8),
                halo_width=1,
            ),
        ),
    )
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    initial = fd.initial_topology()
    tags = jnp.zeros((base_blocks, 4), dtype=bool)
    first = base_blocks // 4
    stop = max(first + 1, 3 * base_blocks // 4)
    tags = tags.at[first:stop, 1:3].set(True)
    compilation = fd.compile_topology(initial, (tags,))
    if not compilation.status.successful:
        raise ValueError(compilation.status.message)
    topology = compilation.topology
    prepared = WaveAMRDiscretizationPlan(
        fd,
        norm_relative_tolerance=2.0e-8,
        maximum_phase_radians=2.0,
    ).prepare(
        WaveAMRPhysicsPlan(
            1.0,
            gravitational_constant=0.02,
            reduced_planck_constant=0.03,
        ),
        topology,
        phx.applications.cosmology.FLRWBackground(1.0, 1.0),
    )
    levels = []
    for level_plan, metadata, spacing in zip(
        topology.plan.levels,
        topology.levels,
        topology.plan.level_spacings,
        strict=True,
    ):
        values = jnp.zeros(
            (level_plan.maximum_blocks, *level_plan.block_shape),
            dtype=jnp.complex128,
        )
        logical = np.asarray(metadata.logical_indices)
        for slot in np.flatnonzero(np.asarray(metadata.active)):
            origin = logical[slot, 0] * level_plan.block_shape[0]
            coordinate = (origin + jnp.arange(4) + 0.5) * spacing[0]
            amplitude = 0.3 + jnp.exp(-(((coordinate - 0.45) / 0.18) ** 2))
            values = values.at[slot].set(amplitude * jnp.exp(2j * jnp.pi * coordinate))
        levels.append(values)
    return hierarchy, prepared, prepared.initialize(tuple(levels), 1.0)


def _measure(action, warmup: int, repeats: int):
    result = None
    for _ in range(warmup):
        result = action()
        jax.block_until_ready(result.successful)
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = action()
        jax.block_until_ready(result.successful)
        samples.append(time.perf_counter() - started)
    assert result is not None
    return result, samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, default=64)
    parser.add_argument("--parts", type=str)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--maximum-bytes", type=int, default=2_000_000_000)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.cells < 16
        or arguments.cells % 4
        or arguments.warmup < 0
        or arguments.repeats < 1
        or arguments.maximum_bytes <= 0
    ):
        raise ValueError(
            "cells >= 16 divisible by 4, nonnegative warmup, positive repeats, "
            "and positive maximum bytes are required"
        )
    available = len(jax.devices())
    if arguments.parts is None:
        part_counts = tuple(value for value in (1, 2, 4, 8, 16, 32) if value <= available)
    else:
        part_counts = tuple(int(value) for value in arguments.parts.split(","))
    if (
        not part_counts
        or any(value < 1 or value > available for value in part_counts)
        or len(set(part_counts)) != len(part_counts)
    ):
        raise ValueError(
            "parts must be unique positive counts no larger than JAX devices"
        )

    hierarchy, prepared, initial = _problem(arguments.cells)
    records = []
    local_authority_baseline = None
    collective_baseline = None
    collective_baseline_parts = None
    for parts in part_counts:
        group = _group(jax.devices()[:parts])
        distributed = prepared.prepare_distributed(
            phx.discretization.BlockAMRPartitionPlan(hierarchy, parts),
            maximum_bytes=arguments.maximum_bytes,
            execution_group=group,
        )
        if not distributed.executable or distributed.execution is None:
            raise RuntimeError(distributed.reason)
        packed_initial = distributed.execution.bind_packed_state(
            distributed.execution.pack_canonical_values(
                prepared.layout.bind_state(initial.psi)
            ),
            initial.scale_factor,
            accepted_boundary=initial.accepted_boundary,
        )
        action = lambda: prepared.distributed_step(
            distributed,
            packed_initial,
            1.00001,
        )
        result, samples = _measure(action, arguments.warmup, arguments.repeats)
        median = float(np.median(np.asarray(samples)))
        if parts > 1 and collective_baseline is None:
            collective_baseline = median
            collective_baseline_parts = parts
        resources = distributed.hierarchy.resources
        diagnostics = result.diagnostics
        if parts == 1:
            first_poisson_residual = (
                result.gravity.solve_result.diagnostics.relative_residual
            )
            second_poisson_residual = (
                result.second_gravity.solve_result.diagnostics.relative_residual
            )
        else:
            first_poisson_residual = result.gravity.solve.relative_residual
            second_poisson_residual = result.second_gravity.solve.relative_residual
        record = {
            "parts": parts,
            "execution_mode": (
                "single-part-local-authority"
                if parts == 1
                else "multi-part-collective-owner-computes"
            ),
            "timed_setup_excluded": (
                "canonical-pack",
                "initial-device-reshard",
            ),
            "timed_input": "prepacked-distributed-state",
            "median_seconds": median,
            "minimum_seconds": float(min(samples)),
            "maximum_seconds": float(max(samples)),
            "strong_scaling_speedup": (
                None if parts == 1 else collective_baseline / median
            ),
            "strong_scaling_efficiency": (
                None
                if parts == 1
                else collective_baseline * collective_baseline_parts / (median * parts)
            ),
            "successful": bool(result.successful),
            "probability_relative_error": float(diagnostics.probability_relative_error),
            "cayley_relative_residual": float(diagnostics.cayley_relative_residual),
            "self_adjoint_residual": float(diagnostics.self_adjoint_residual),
            "first_poisson_relative_residual": float(first_poisson_residual),
            "second_poisson_relative_residual": float(second_poisson_residual),
            "required_bytes": distributed.required_bytes,
            "maximum_bytes": distributed.maximum_bytes,
            "active_blocks": list(resources.active_blocks),
            "allocated_block_slots": list(resources.allocated_block_slots),
            "same_level_routes": list(resources.same_level_routes),
            "coarse_fine_routes": list(resources.coarse_fine_routes),
            "interface_routes": list(resources.interface_routes),
            "composite_crossing_edges": (distributed.execution.route_crossing_count),
            "dynamic_fillpatch_route_bytes": (resources.dynamic_route_array_bytes),
            "dynamic_composite_route_bytes": (
                distributed.execution.routes.dynamic_array_bytes
            ),
            "execution_id": distributed.execution.execution_id,
        }
        if parts == 1:
            record.pop("strong_scaling_speedup")
            record.pop("strong_scaling_efficiency")
            local_authority_baseline = record
        else:
            records.append(record)

    payload = {
        "configuration": {
            "cells": arguments.cells,
            "physical_leaf_cells": prepared.layout.real_layout.physical_cell_count,
            "active_blocks": [
                int(jnp.sum(metadata.active)) for metadata in prepared.topology.levels
            ],
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
            "available_devices": available,
            "part_counts": list(part_counts),
            "strong_scaling_baseline": "first-multipart-collective-record",
        },
        "prepared_id": prepared.prepared_id,
        "records": records,
        "local_authority_baseline": local_authority_baseline,
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
