#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


def run_case(points: int, samples_per_axis: int, /) -> dict[str, object]:
    harmonics = phx.discretization.LatticeHarmonicPlan.parallelogramic(
        (1, 1), (points, points)
    ).prepare(jnp.eye(2))
    plan = phx.solver.maxwell.fourier_modal.FourierModalRasterizationPlan(
        harmonics,
        phx.solver.maxwell.fourier_modal.FourierModalRasterizationPolicy(
            samples_per_axis=samples_per_axis,
            smoothing_width=0.02,
        ),
    )
    geometry = phx.geometry.Circle((0.5, 0.5), 0.2, feature_id="inclusion").compile()
    radius_id = phx.geometry.ParameterId("inclusion", "radius")

    def mean_fill(radius):
        current = geometry.with_parameters({radius_id: radius})
        result = phx.solver.maxwell.fourier_modal.rasterize_fourier_modal_material(
            plan,
            current,
            inside_permittivity=12.0,
            material_id="inclusion",
        )
        return jnp.mean(result.fill_fraction)

    value_and_gradient = jax.jit(jax.value_and_grad(mean_fill))
    started = time.perf_counter()
    value, gradient = value_and_gradient(jnp.asarray(0.2))
    jax.block_until_ready(value)
    compile_seconds = time.perf_counter() - started
    started = time.perf_counter()
    value, gradient = value_and_gradient(jnp.asarray(0.2))
    jax.block_until_ready(value)
    warm_seconds = time.perf_counter() - started
    return {
        "points_per_axis": points,
        "samples_per_axis": samples_per_axis,
        "fill_fraction": float(value),
        "radius_gradient": float(gradient),
        "compile_seconds": compile_seconds,
        "warm_seconds": warm_seconds,
        "parameter_differentiable": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fourier_modal_rasterization.json"),
    )
    parser.add_argument("--points", type=int, default=64)
    parser.add_argument("--samples-per-axis", type=int, default=3)
    args = parser.parse_args()
    payload = run_case(args.points, args.samples_per_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
