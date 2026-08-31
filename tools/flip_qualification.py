#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class FLIPQualificationReport:
    liquid_cells: int
    mass_balance_defect: float
    momentum_balance_defect: float
    divergence_norm: float
    air_pressure_defect: float
    maximum_displacement_fraction: float
    successful: bool


def run_flip_qualification(*, smoke=False):
    count = 8 if smoke else 20
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
    side = 4 if smoke else 8
    x = jnp.linspace(0.2, 0.55, side)
    y = jnp.linspace(0.2, 0.65, side)
    xx, yy = jnp.meshgrid(x, y, indexing="ij")
    position = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
    mass = jnp.full((position.shape[0],), 1000.0 * 0.004)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(position.shape[0]), mass, ambient_dimension=2
    ).prepare()
    transfer = phx.discretization.flip.FLIPParticleTransferPlan(mac).prepare(particles)
    problem = phx.equations.FLIPProblemIR(
        "qualification", 1000.0, jnp.asarray([0.0, -1.0])
    )
    method = phx.discretization.flip.FLIPMethodPlan(
        0.05, liquid_fraction_threshold=0.01
    )
    compiled = phx.equations.compile_flip_problem(problem, transfer, projection, method)
    state = compiled.initialize_state(position, jnp.zeros_like(position))
    result = compiled.step_detailed(state, 2.0e-4)
    projected = result.diagnostics.details
    successful = bool(
        result.successful
        and result.diagnostics.mass_balance_defect < 1.0e-10
        and result.diagnostics.momentum_balance_defect < 1.0e-10
        and result.diagnostics.divergence_norm < 1.0e-6
        and projected.air_pressure_defect < 1.0e-10
    )
    return FLIPQualificationReport(
        int(result.diagnostics.liquid_count),
        float(result.diagnostics.mass_balance_defect),
        float(result.diagnostics.momentum_balance_defect),
        float(result.diagnostics.divergence_norm),
        float(projected.air_pressure_defect),
        float(result.diagnostics.maximum_displacement_fraction),
        successful,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_flip_qualification(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
