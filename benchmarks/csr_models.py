#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Phase-separated coherent synchrotron radiation field benchmark.

Each CSR model evaluates the fields of a Gaussian bunch deep in a bend. Plan
preparation (kernel tables), lowering, compilation, and warmed execution are
measured separately; the longitudinal cell count is the controlling capacity.
"""

from __future__ import annotations

import argparse
import json
import math
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

from phydrax import ElectromagneticScaleContract
from phydrax.applications.accelerator import CSRLattice, CSRModel, CSRPlan
from phydrax.applications.accelerator._csr import (
    _axis_gradient,
    _evaluate_fields,
    _frozen_timeline,
)
from phydrax.discretization import TensorGridPlan, UniformCellAxisSpec
from phydrax.typing import parse


def _compiler_record(compiled: Any) -> dict[str, object]:
    evidence = compiler_evidence(
        compiled.cost_analysis(),
        compiled.memory_analysis(),
        source="jax-compiled-executable",
    )
    return {
        "flops": evidence.flops,
        "bytes_accessed": evidence.bytes_accessed,
        "argument_bytes": evidence.argument_bytes,
        "output_bytes": evidence.output_bytes,
        "temporary_bytes": evidence.temporary_bytes,
        "generated_code_bytes": evidence.generated_code_bytes,
    }


def _density(
    grid: Any, shape: tuple[int, ...], sigma_x: float, sigma_z: float
) -> np.ndarray:
    points = np.asarray(grid.points).reshape(shape + (len(shape),))
    longitudinal = np.exp(-0.5 * (points[..., -1] / sigma_z) ** 2)
    if len(shape) == 1:
        return 1.0e-9 * longitudinal / (math.sqrt(2.0 * math.pi) * sigma_z)
    transverse = np.exp(-0.5 * (points[..., 0] ** 2 + points[..., 1] ** 2) / sigma_x**2)
    return (
        1.0e-9
        * longitudinal
        * transverse
        / ((2.0 * math.pi) ** 1.5 * sigma_x**2 * sigma_z)
    )


def _case(
    model: str,
    cells: int,
    arguments: argparse.Namespace,
    scale: ElectromagneticScaleContract,
) -> dict[str, object]:
    sigma_z, sigma_x, radius = 1.0e-5, 2.0e-6, 1.0
    three = model.startswith("3d")
    shape = (arguments.transverse, arguments.transverse, cells) if three else (cells,)
    lower = (-4 * sigma_x, -4 * sigma_x, -5 * sigma_z) if three else (-6 * sigma_z,)
    grid = TensorGridPlan(tuple(UniformCellAxisSpec(count) for count in shape)).prepare(
        np.asarray([lower, [-value for value in lower]])
    )
    lattice = CSRLattice([1.0], [1.0 / radius], element_ids=["bend"])
    rest_energy = float(scale.electron_mass) * float(scale.speed_of_light) ** 2
    start = time.perf_counter()
    plan = CSRPlan(
        parse(model, CSRModel, "model"),
        lattice,
        scale,
        grid,
        reference_rest_energy=rest_energy,
        reference_momentum=rest_energy * math.sqrt(arguments.gamma**2 - 1.0),
        capacity=8,
        kernel_quadrature=2,
        far_nodes=arguments.far_nodes,
    )
    preparation = time.perf_counter() - start
    density = jnp.asarray(_density(grid, shape, sigma_x, sigma_z))
    slope = _axis_gradient(density, plan.spacing[-1], -1)
    timeline = _frozen_timeline(density, slope, jnp.asarray(0.9))
    dynamic, static = eqx.partition((plan, timeline), eqx.is_array)

    def evaluate(leaves: Any) -> Any:
        plan_, timeline_ = eqx.combine(leaves, static)
        fields = _evaluate_fields(plan_, timeline_, timeline_.positions[-1])
        return fields.wake, fields.failures

    function = jax.jit(evaluate)
    compiled, compilation = measure_lower_and_compile(
        lambda: function.lower(dynamic), lambda lowered: lowered.compile()
    )
    (wake, failures), execution = measure_repeated(
        lambda: compiled(dynamic), warmup=arguments.warmup, repeats=arguments.repeats
    )
    return {
        "model": model,
        "cells": cells,
        "identity": plan.plan_id,
        "preparation_seconds": preparation,
        "compilation": {
            "lowering_seconds": compilation.lowering_seconds,
            "compilation_seconds": compilation.compilation_seconds,
        },
        "compiler": _compiler_record(compiled),
        "execution": execution.to_seconds_dict(),
        "memory": {
            "estimated_retarded_pairs": plan.estimate.retarded_pairs,
            "estimated_history_bytes": plan.estimate.history_bytes,
            "estimated_kernel_bytes": plan.estimate.kernel_bytes,
            "input_logical_bytes": logical_array_bytes(dynamic),
            "output_logical_bytes": logical_array_bytes(wake),
        },
        "physics": {
            "finite": bool(np.all(np.isfinite(np.asarray(wake)))),
            "retarded_failures": int(failures),
            "peak_wake_V": float(np.max(np.abs(np.asarray(wake)))),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        type=str,
        default="1d-steady,1d-transient-shielded,3d-steady-igf,3d-retarded-mesh",
    )
    parser.add_argument("--cells", type=str, default="32,64")
    parser.add_argument("--transverse", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=500.0)
    parser.add_argument("--far-nodes", type=int, default=48)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    cells = tuple(int(value) for value in arguments.cells.split(","))
    models = tuple(value.strip() for value in arguments.models.split(","))
    if (
        min(cells) < 8
        or arguments.transverse < 2
        or arguments.gamma <= 1.0
        or arguments.far_nodes < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "cells >= 8, transverse >= 2, gamma > 1, positive far nodes and repeats, "
            "and nonnegative warmup are required"
        )
    scale = ElectromagneticScaleContract.si()
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "models": list(models),
            "cells": list(cells),
            "transverse": arguments.transverse,
            "gamma": arguments.gamma,
            "far_nodes": arguments.far_nodes,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "cases": [
            _case(model, count, arguments, scale) for model in models for count in cells
        ],
    }
    encoded = json.dumps(payload, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        arguments.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
