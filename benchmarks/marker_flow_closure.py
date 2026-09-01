#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from math import prod
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def _case(grid_count: int, marker_count: int) -> dict[str, float | int | bool]:
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(grid_count, periodic=True)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    operators = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    angle = 2.0 * jnp.pi * jnp.arange(marker_count) / marker_count
    position = jnp.stack(
        (0.5 + 0.2 * jnp.cos(angle), 0.5 + 0.2 * jnp.sin(angle)), axis=-1
    )
    markers = phx.discretization.LagrangianMarkerSetPlan(
        jnp.arange(marker_count),
        position,
        jnp.full((marker_count,), 2.0 * jnp.pi * 0.2 / marker_count),
    ).prepare()
    plan = phx.discretization.MACMarkerTransferPlan(operators, markers)
    transfer = plan.prepare()
    relation = transfer.relation(position)
    velocity = tuple(
        jnp.sin(jnp.arange(prod(layout.shape)).reshape(layout.shape))
        for layout in finite_volume.face_layouts
    )
    force = jnp.stack((jnp.cos(angle), jnp.sin(angle)), axis=-1)
    action = jax.jit(
        lambda velocity_, force_: (
            transfer.gather(relation, velocity_),
            transfer.spread(relation, force_),
        )
    )
    start = time.perf_counter()
    compiled = action(velocity, force)
    jax.block_until_ready(compiled)
    compile_ms = 1000.0 * (time.perf_counter() - start)
    repetitions = 10
    start = time.perf_counter()
    for _ in range(repetitions):
        executed = action(velocity, force)
    jax.block_until_ready(executed)
    execution_ms = 1000.0 * (time.perf_counter() - start) / repetitions
    diagnostics = transfer.diagnostics(relation, velocity, force)
    return {
        "grid_count": grid_count,
        "marker_count": marker_count,
        "route_width": plan.route_width,
        "relation_bytes": plan.relation_bytes,
        "workspace_bytes": plan.scalar_workspace_bytes,
        "compile_and_first_ms": compile_ms,
        "execution_ms": execution_ms,
        "work_residual": float(jnp.abs(diagnostics.work_adjoint_residual)),
        "force_residual": float(jnp.max(jnp.abs(diagnostics.force_residual))),
        "successful": bool(diagnostics.successful),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-counts", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--marker-counts", nargs="+", type=int, default=[16, 64, 256])
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cases = [
        _case(grid_count, marker_count)
        for grid_count in arguments.grid_counts
        for marker_count in arguments.marker_counts
    ]
    payload = {
        "cases": cases,
        "all_successful": all(case["successful"] for case in cases),
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
