#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class FLIPBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    particle_count: int
    grid_count: int
    compile_and_first_ms: float
    steady_step_ms: float
    mass_balance_defect: float
    momentum_balance_defect: float
    divergence_norm: float
    successful: bool


def _case(count):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count) for _ in range(2)),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    finite_volume = phx.discretization.FiniteVolumePlan(grid).prepare()
    mac = phx.discretization.MACOperatorPlan(finite_volume).prepare()
    boundaries = phx.discretization.MACBoundaryPlan(mac).prepare()
    projection = phx.solver.MACFreeSurfaceProjectionPlan(
        mac, boundaries=boundaries, tolerance=1.0e-7
    )
    side = max(3, count // 2)
    x = jnp.linspace(0.2, 0.6, side)
    y = jnp.linspace(0.2, 0.7, side)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    position = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
    particle_count = int(position.shape[0])
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count),
        jnp.full((particle_count,), 1000.0 * 0.005),
        ambient_dimension=2,
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(support)
    problem = phx.equations.FLIPProblemIR(
        "benchmark", 1000.0, jnp.asarray([0.0, -1.0])
    )
    method = phx.discretization.flip.FLIPMethodPlan(
        0.05, liquid_fraction_threshold=0.01
    )
    compiled = phx.equations.compile_flip_problem(problem, transfer, projection, method)
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    return compiled, state, particle_count


def run_flip_benchmark(*, smoke=False):
    count = 8 if smoke else 24
    compiled, state, particle_count = _case(count)

    @jax.jit
    def apply(value):
        return compiled.step_detailed(value, jnp.asarray(2.0e-4))

    started = perf_counter()
    first = apply(state)
    jax.block_until_ready(first.accepted_state.particles.position)
    first_ms = (perf_counter() - started) * 1.0e3
    repetitions = 3 if smoke else 8
    started = perf_counter()
    result = first
    for _ in range(repetitions):
        result = apply(state)
    jax.block_until_ready(result.accepted_state.particles.position)
    steady_ms = (perf_counter() - started) * 1.0e3 / repetitions
    return FLIPBenchmarkReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        particle_count,
        count,
        first_ms,
        steady_ms,
        float(result.diagnostics.mass_balance_defect),
        float(result.diagnostics.momentum_balance_defect),
        float(result.diagnostics.divergence_norm),
        bool(result.successful),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_flip_benchmark(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
