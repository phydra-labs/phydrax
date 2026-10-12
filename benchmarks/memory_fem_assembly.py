#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Requested versus paired finite-element assembly memory campaign."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from phydrax.discretization import (
    CellMesh,
    FiniteElementFieldSpec,
    FiniteElementPlan,
    lagrange_element,
)

from ._io import write_json_atomic
from ._memory import host_allocation_peak, process_peak_rss_bytes
from ._runtime import (
    capture_benchmark_identity,
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    measure_synchronized,
)


def _case(cells: int, route: str, repeats: int, /) -> dict[str, Any]:
    x = np.arange(cells + 1, dtype=np.float64)
    points = np.stack(
        (np.repeat(x, 2), np.tile(np.asarray([0.0, 1.0]), cells + 1)), axis=1
    )
    connectivity = np.asarray(
        [(2 * i, 2 * i + 2, 2 * i + 1) for i in range(cells)]
        + [(2 * i + 1, 2 * i + 2, 2 * i + 3) for i in range(cells)],
        dtype=np.int32,
    )
    mesh = CellMesh.from_triangles(points, connectivity)
    space = FiniteElementPlan(
        mesh, FiniteElementFieldSpec("u", lagrange_element("triangle", 1))
    ).prepare()

    def assemble() -> object:
        match route:
            case "mass":
                return space.mass
            case "stiffness":
                return space.stiffness
            case "paired":
                return space.assemble_field_operators("u", space.default_runtime)
            case _:
                raise ValueError("Unknown assembly route.")

    measured, peak = host_allocation_peak(lambda: measure_synchronized(assemble))
    operators, seconds = measured

    def action(coordinates: Array) -> Array:
        runtime = space.prepare_runtime(coordinates, numeric_version="benchmark")
        mass, stiffness = space.assemble_field_operators("u", runtime)
        value = jnp.ones((space.dof_maps[0].global_dof_count,), dtype=jnp.float64)
        return mass.mv(value) + stiffness.mv(value)

    executable, compilation = measure_lower_and_compile(
        lambda: jax.jit(action).lower(space.vertices), lambda lowered: lowered.compile()
    )
    _, warm = measure_repeated(
        lambda: executable(space.vertices), warmup=1, repeats=repeats
    )
    return {
        "cells": 2 * cells,
        "route": route,
        "compiler_route": "paired-runtime-assembly",
        "requested_tensor_families": 2 if route == "paired" else 1,
        "host_allocation_peak_bytes": peak,
        "assembly_seconds": seconds,
        "retained_array_bytes": logical_array_bytes(operators),
        "process_peak_rss_bytes": process_peak_rss_bytes(),
        "compilation": asdict(compilation),
        "compiler": asdict(
            compiler_evidence(
                executable.cost_analysis(), executable.memory_analysis(), source="xla"
            )
        ),
        "warm": warm.to_dict(unit="seconds"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cells", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("benchmarks/memory_fem_assembly.json")
    )
    args = parser.parse_args()
    sizes = [2, 4] if args.smoke else args.cells
    cases = [
        _case(size, route, args.repeats)
        for size in sizes
        for route in ("mass", "stiffness", "paired")
    ]
    report = {
        "identity": capture_benchmark_identity(
            Path(__file__).resolve().parent.parent,
            Path(__file__).resolve(),
            cases[0].keys(),
        ).to_dict(),
        "environment": capture_environment().to_dict(),
        "cases": cases,
    }
    write_json_atomic(args.output, report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
