"""Fourier-modal planning, preparation, refresh, and RHS benchmark."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp

from phydrax.discretization.spectral import LatticeHarmonicPlan
from phydrax.solver.maxwell import fourier_modal as fm


jax.config.update("jax_enable_x64", True)


def _timed(function):
    start = time.perf_counter()
    value = function()
    jax.block_until_ready(value)
    return value, time.perf_counter() - start


def benchmark() -> dict[str, object]:
    harmonics = LatticeHarmonicPlan.parallelogramic((1,), (3,)).prepare(
        jnp.asarray(((1.0, 0.0),))
    )
    vacuum = fm.FrequencyMaxwellMaterial(1.0, material_id="vacuum")
    film = fm.FrequencyMaxwellMaterial(2.25, material_id="film")
    layer = fm.FourierModalLayer(
        film,
        0.2,
        fm.DirectFourierFactorizationPlan(),
        layer_id="film",
    )
    problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        fm.HomogeneousMaxwellPort(vacuum, port_id="left"),
        (layer,),
        fm.HomogeneousMaxwellPort(vacuum, port_id="right"),
    )
    start = time.perf_counter()
    plan = fm.plan_fourier_modal_maxwell(problem)
    planning_seconds = time.perf_counter() - start
    start = time.perf_counter()
    prepared = fm.prepare_fourier_modal_maxwell(problem, plan)
    jax.block_until_ready(prepared.scattering.s11.matrix)
    preparation_seconds = time.perf_counter() - start

    rhs_count = 32
    left = jnp.zeros((2, rhs_count), dtype=jnp.complex128).at[1, :].set(1.0)
    excitation = fm.FourierModalExcitation(left, jnp.zeros_like(left))
    start = time.perf_counter()
    result = fm.solve_fourier_modal_maxwell(prepared, excitation)
    jax.block_until_ready(result.right_outgoing)
    solve_seconds = time.perf_counter() - start

    updated_layer = fm.FourierModalLayer(
        film,
        0.21,
        fm.DirectFourierFactorizationPlan(),
        layer_id="film",
    )
    updated_problem = fm.FourierModalMaxwellProblem(
        harmonics,
        2.0 * jnp.pi,
        jnp.asarray((0.0, 0.0)),
        problem.superstrate,
        (updated_layer,),
        problem.substrate,
        numeric_version="thickness-update",
    )
    start = time.perf_counter()
    refreshed = fm.refresh_fourier_modal_maxwell(
        prepared,
        updated_problem,
        fm.FourierModalRefreshSpec(("thickness",)),
    )
    jax.block_until_ready(refreshed.scattering.s11.matrix)
    refresh_seconds = time.perf_counter() - start
    return {
        "environment": {
            "python": platform.python_version(),
            "jax": jax.__version__,
            "backend": jax.default_backend(),
            "platform": platform.platform(),
        },
        "problem": {
            "harmonic_count": harmonics.harmonic_count,
            "layer_count": problem.layer_count,
            "rhs_count": rhs_count,
            "preparation_bytes": plan.cost.preparation_bytes,
            "workspace_bytes": plan.cost.workspace_bytes,
        },
        "timings_seconds": {
            "planning": planning_seconds,
            "preparation": preparation_seconds,
            "solve_32_rhs": solve_seconds,
            "thickness_refresh": refresh_seconds,
        },
        "evidence": {
            "status": int(result.status),
            "left_incoming_power": float(result.weighted_left_incoming_power),
            "right_incoming_power": float(result.weighted_right_incoming_power),
            "left_outgoing_power": float(result.weighted_left_outgoing_power),
            "right_outgoing_power": float(result.weighted_right_outgoing_power),
            "net_port_power_into_stack": float(result.weighted_net_port_power_into_stack),
            "paired_error": float(result.diagnostics.maximum_boundary_paired_error),
            "refresh_count": refreshed.refresh_count,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fourier_modal_maxwell.json"),
    )
    arguments = parser.parse_args()
    payload = benchmark()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
