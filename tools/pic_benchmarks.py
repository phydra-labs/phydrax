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
class PICBenchmarkReport:
    maturity: str
    phydrax_version: str
    jax_version: str
    device: str
    particle_count: int
    grid_count: int
    charge_routes: int
    compile_and_first_ms: float
    steady_step_ms: float
    continuity_defect: float
    gauss_defect: float
    charge_balance_defect: float
    successful: bool


def _case(count, particle_count):
    grid = phx.discretization.TensorGridPlan(
        tuple(phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)),
        axis_names=("x", "y", "z"),
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    support = phx.discretization.ParticleSetPlan(
        jnp.arange(particle_count), jnp.ones((particle_count,)), ambient_dimension=3
    ).prepare()
    charged = phx.discretization.ChargedParticlePlan(
        -jnp.ones((particle_count,)), "benchmark"
    ).prepare(support)
    transfer = phx.discretization.pic.PICParticleCochainTransferPlan(bridge).prepare(charged)
    current = phx.discretization.pic.ChargeConservingCurrentPlan(transfer)
    key = jax.random.key(17)
    position = jax.random.uniform(key, (particle_count, 3), minval=0.1, maxval=0.9)
    displacement = 0.02 / count * jnp.asarray([1.0, -0.4, 0.25])
    return transfer, current, position, position + displacement


def run_pic_benchmark(*, smoke=False):
    count = 4 if smoke else 8
    particle_count = 16 if smoke else 512
    transfer, current, start, end = _case(count, particle_count)

    @jax.jit
    def apply(left, right):
        return current.deposit(left, right, jnp.asarray(1.0e-3))

    started = perf_counter()
    first = apply(start, end)
    jax.block_until_ready(first.current)
    first_ms = (perf_counter() - started) * 1.0e3
    repetitions = 3 if smoke else 8
    started = perf_counter()
    result = first
    for _ in range(repetitions):
        result = apply(start, end)
    jax.block_until_ready(result.current)
    steady_ms = (perf_counter() - started) * 1.0e3 / repetitions
    resources = dict(transfer.charge.preparation.resource_counts)
    successful = bool(result.successful)
    return PICBenchmarkReport(
        "experimental",
        version("phydrax"),
        jax.__version__,
        str(jax.devices()[0]),
        particle_count,
        count,
        int(resources["route_count"]),
        first_ms,
        steady_ms,
        float(result.maximum_continuity_defect),
        float(jnp.max(jnp.abs(result.continuity_residual))),
        float(
            jnp.maximum(
                result.start_charge.balance.maximum_absolute_balance_defect,
                result.end_charge.balance.maximum_absolute_balance_defect,
            )
        ),
        successful,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_pic_benchmark(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
