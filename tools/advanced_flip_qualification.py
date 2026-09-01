#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class AdvancedFLIPQualification:
    liquid_cells: int
    interface_band_cells: int
    surface_energy: float
    ghost_divergence: float
    air_pressure_defect: float
    viscous_dissipation: float
    viscous_energy_increase: float
    compile_and_first_ms: float
    steady_ms: float
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
        mac, boundaries=boundaries, tolerance=1.0e-7, maximum_iterations=2000
    )
    ghost = phx.solver.MACGhostFluidProjectionPlan(projection)
    x = jnp.linspace(0.3, 0.7, max(4, count // 3))
    xx, yy = jnp.meshgrid(x, x, indexing="ij")
    position = jnp.stack((xx.reshape((-1,)), yy.reshape((-1,))), axis=-1)
    interface_plan = phx.discretization.flip.ParticleLevelSetPlan(grid, 1.5 / count)
    interface = interface_plan.evaluate(
        position, jnp.ones((position.shape[0],), dtype=bool)
    )
    capillary = phx.discretization.finite_volume.MACGhostFluidCapillaryPlan(
        0.05, interface_width=2.0 / count
    ).evaluate(interface)
    velocity = tuple(jnp.zeros(layout.shape) for layout in finite_volume.face_layouts)
    return finite_volume, mac, ghost, interface, capillary, velocity


def run(*, smoke=False):
    count = 8 if smoke else 24
    finite_volume, mac, ghost, interface, capillary, velocity = _case(count)

    @jax.jit
    def project(values):
        return ghost.project(
            values,
            interface,
            jnp.asarray(1.0e-3),
            pressure_jump=capillary.pressure_jump,
        )

    started = perf_counter()
    projected = project(velocity)
    jax.block_until_ready(projected.pressure)
    first_ms = 1.0e3 * (perf_counter() - started)
    repetitions = 3 if smoke else 8
    started = perf_counter()
    for _ in range(repetitions):
        projected = project(velocity)
    jax.block_until_ready(projected.pressure)
    steady_ms = 1.0e3 * (perf_counter() - started) / repetitions

    measures = phx.discretization.finite_volume.MACFreeSurfaceViscousMeasurePlan(
        mac, 1.0
    ).evaluate(interface, 0.1)
    viscous = phx.solver.MACVariationalViscosityPlan(mac, tolerance=1.0e-7).solve(
        projected.velocity, measures, 1.0e-3
    )
    successful = bool(
        interface.successful
        and projected.successful
        and viscous.successful
        and projected.projection.air_pressure_defect < 1.0e-8
        and viscous.energy_increase < 1.0e-8
    )
    return AdvancedFLIPQualification(
        int(jnp.sum(interface.liquid_mask)),
        int(jnp.sum(interface.valid_band)),
        float(capillary.surface_energy),
        float(projected.projection.active_divergence_norm),
        float(projected.projection.air_pressure_defect),
        float(viscous.dissipation),
        float(viscous.energy_increase),
        first_ms,
        steady_ms,
        successful,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(smoke=args.smoke)
    payload = json.dumps(asdict(report), indent=2)
    print(payload)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    if not report.successful:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
