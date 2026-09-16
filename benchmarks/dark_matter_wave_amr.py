from __future__ import annotations

import argparse
import json
from math import prod
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
from _runtime import capture_environment, measure_lower_and_compile, measure_repeated

import phydrax as phx
from phydrax.applications.cosmology._wave_amr import (
    WaveAMRDiscretizationPlan,
    WaveAMRPhysicsPlan,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 8
        or arguments.size % 4
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "size >= 8 divisible by 4, nonnegative warmup, and positive repeats are required"
        )

    shape = (arguments.size,) * 3
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True)
            for count in shape
        ),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    base_blocks = prod(count // 4 for count in shape)
    hierarchy = phx.discretization.BlockHierarchyPlan(
        grid,
        (
            phx.discretization.BlockLevelPlan(
                0,
                (4, 4, 4),
                base_blocks,
                halo_width=1,
                refinement_ratio=2,
            ),
            phx.discretization.BlockLevelPlan(
                1,
                (4, 4, 4),
                max(32, 2 * base_blocks),
                halo_width=1,
            ),
        ),
    )
    fd = phx.discretization.FDAMRHierarchyPlan(hierarchy).prepare()
    topology = fd.initial_topology()
    tags = (
        jnp.zeros((base_blocks, 4, 4, 4), dtype=bool)
        .at[base_blocks // 2, 1:3, 1:3, 1:3]
        .set(True)
    )
    compiled_topology = fd.compile_topology(topology, (tags,))
    if not compiled_topology.status.successful:
        raise ValueError(compiled_topology.status.message)
    topology = compiled_topology.topology
    prepared = WaveAMRDiscretizationPlan(
        fd,
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
    values = []
    for level, (level_plan, metadata) in enumerate(
        zip(topology.plan.levels, topology.levels, strict=True)
    ):
        index = jnp.indices(level_plan.block_shape)
        local_phase = 0.02 * sum(index)
        profile = (1.0 + 0.05 * jnp.cos(local_phase)) * jnp.exp(1j * local_phase)
        active = metadata.active.reshape(
            (level_plan.maximum_blocks,) + (1,) * len(level_plan.block_shape)
        )
        values.append(
            jnp.where(
                active,
                jnp.broadcast_to(
                    profile, (level_plan.maximum_blocks,) + level_plan.block_shape
                ),
                0.0,
            ).astype(jnp.complex128)
        )
    state = prepared.initialize(tuple(values), 1.0)
    function = eqx.filter_jit(lambda value: prepared.step(value, 1.00001))
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(state),
        lambda lowered: lowered.compile(),
    )
    result, execution = measure_repeated(
        lambda: compiled(state),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    diagnostics = result.diagnostics
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "base_shape": shape,
            "level_count": len(topology.plan.levels),
            "active_blocks": [
                int(jnp.sum(metadata.active)) for metadata in topology.levels
            ],
            "physical_leaf_cells": prepared.layout.real_layout.physical_cell_count,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "identity": prepared.prepared_id,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "successful": bool(result.successful),
            "probability_relative_error": float(diagnostics.probability_relative_error),
            "cayley_relative_residual": float(diagnostics.cayley_relative_residual),
            "self_adjoint_residual": float(diagnostics.self_adjoint_residual),
            "poisson_relative_residual": float(
                result.gravity.solve_result.diagnostics.relative_residual
            ),
            "second_poisson_relative_residual": float(
                result.second_gravity.solve_result.diagnostics.relative_residual
            ),
            "maximum_kinetic_phase": float(diagnostics.maximum_kinetic_phase),
            "interface_flux_conservation_defect": float(
                result.gravity.interface_flux_conservation_defect
            ),
        },
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
