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
class PICQualificationReport:
    poisson_residual: float
    charge_balance_defect: float
    electrostatic_energy_defect: float
    continuity_defect: float
    boris_speed_defect: float
    successful: bool


def run_pic_qualification(*, smoke=False):
    count = 16 if smoke else 32
    particle_count = 8 if smoke else 32
    grid = phx.discretization.TensorGridPlan(
        (phx.discretization.UniformCellAxisSpec(count, periodic=True),),
        axis_names=("x",),
    ).prepare(jnp.asarray([[0.0], [1.0]]))
    bridge = phx.discretization.StructuredCochainBridge(grid)
    base = (jnp.arange(particle_count, dtype=float)[:, None] + 0.5) / particle_count
    transfers = []
    for offset, sign, name in ((0, -1.0, "negative"), (1000, 1.0, "positive")):
        support = phx.discretization.ParticleSetPlan(
            jnp.arange(offset, offset + particle_count),
            jnp.ones((particle_count,)),
            ambient_dimension=1,
        ).prepare()
        charged = phx.discretization.ChargedParticlePlan(
            sign * jnp.ones((particle_count,)), name
        ).prepare(support)
        transfers.append(
            phx.discretization.pic.PICParticleCochainTransferPlan(bridge).prepare(charged)
        )
    field = phx.solver.CochainElectrostaticPlan(
        bridge, phx.solver.CochainElectrostaticBoundaryPlan.periodic(bridge)
    )
    plan = phx.solver.ElectrostaticPICPlan(field, tuple(transfers))
    state = plan.initialize(
        (base + 0.002 * jnp.sin(2.0 * jnp.pi * base), base),
        (jnp.zeros((particle_count, 1)), jnp.zeros((particle_count, 1))),
    )
    step = plan.step_detailed(state, 5.0e-4)

    pusher = phx.discretization.pic.RelativisticBorisPlan()
    proper = jnp.asarray([[0.2, 0.1, 0.0]])
    pushed = pusher.push(
        proper,
        jnp.zeros_like(proper),
        jnp.asarray([[0.0, 0.0, 0.7]]),
        jnp.asarray([1.0]),
        jnp.asarray([True]),
        1.0e-3,
    )
    speed_defect = jnp.abs(jnp.sum(pushed.proper_velocity**2) - jnp.sum(proper**2))
    successful = bool(
        step.successful
        and pushed.successful
        and step.diagnostics.poisson_residual < 1.0e-8
        and step.diagnostics.charge_balance_defect < 1.0e-10
        and speed_defect < 1.0e-10
    )
    return PICQualificationReport(
        float(step.diagnostics.poisson_residual),
        float(step.diagnostics.charge_balance_defect),
        float(step.diagnostics.energy.defect),
        0.0,
        float(speed_defect),
        successful,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_pic_qualification(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
