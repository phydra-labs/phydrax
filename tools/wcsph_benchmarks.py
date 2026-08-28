#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

import phydrax as phx


@dataclass(frozen=True)
class WCSPHBenchmarkRecord:
    schema_version: int
    acoustic_particle_count: int
    acoustic_drift_parity_error: float
    acoustic_trajectory_parity_error: float
    acoustic_energy_balance_defect: float
    acoustic_gradient_error: float
    shear_particle_count: int
    shear_velocity_error: float
    shear_density_spread: float
    shear_momentum_defect: float
    shear_dissipation_rate: float
    shear_energy_rate_defect: float
    finite: bool

    @property
    def passed(self) -> bool:
        return (
            self.finite
            and self.acoustic_drift_parity_error <= 1.0e-11
            and self.acoustic_trajectory_parity_error <= 1.0e-10
            and self.acoustic_energy_balance_defect <= 1.0e-10
            and self.acoustic_gradient_error <= 2.0e-6
            and self.shear_velocity_error <= 2.0e-3
            and self.shear_density_spread <= 0.1
            and self.shear_momentum_defect <= 1.0e-10
            and self.shear_dissipation_rate >= 0.0
            and self.shear_energy_rate_defect <= 1.0e-9
        )


def _acoustic_problem(count, backend):
    spacing = 1.0 / count
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing), ambient_dimension=1
    ).prepare()
    box = phx.discretization.ParticleBox([0.0], [1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(
        phx.discretization.WendlandC2SPHKernel(1),
        1.25 * spacing,
        density=phx.discretization.ContinuityDensityPlan(),
    )
    neighborhood = (
        phx.discretization.DenseParticleNeighborhoodPlan(
            count * (count - 1) // 2, box=box
        )
        if backend == "dense"
        else phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            8,
            4 * count,
            box,
        )
    )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "acoustic-wave", phx.equations.TaitBarotropicMaterial(1.0, 1.0)
        ),
        particles,
        method,
        neighborhood=neighborhood,
    )
    position = (jnp.arange(count, dtype=float) + 0.5)[:, None] * spacing
    position = position + 0.001 * jnp.sin(2.0 * jnp.pi * position)
    velocity = jnp.zeros_like(position)
    return compiled, position, velocity


def _solve(compiled, position, velocity, final_time, step):
    return phx.solver.solve_diffrax(
        compiled.as_differential_problem(position, velocity, t0=0.0, t1=final_time),
        save_times=jnp.asarray([final_time]),
        solver=phx.solver.SSPRK33(),
        dt0=step,
        max_steps=256,
    )


def _shear_case(resolution):
    count = resolution**2
    spacing = 1.0 / resolution
    viscosity = 0.01
    wave_number = 2.0 * jnp.pi
    amplitude = 0.05
    final_time = 0.01
    axis = (jnp.arange(resolution, dtype=float) + 0.5) * spacing
    x_grid, y_grid = jnp.meshgrid(axis, axis, indexing="ij")
    position = jnp.stack((x_grid.reshape(-1), y_grid.reshape(-1)), axis=-1)
    velocity = jnp.stack(
        (
            jnp.zeros((count,)),
            amplitude * jnp.sin(wave_number * position[:, 0]),
        ),
        axis=-1,
    )
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(count), jnp.full((count,), spacing**2), ambient_dimension=2
    ).prepare()
    box = phx.discretization.ParticleBox([0.0, 0.0], [1.0, 1.0])
    method = phx.discretization.WeaklyCompressibleSPHMethodPlan(phx.discretization.WendlandC2SPHKernel(2),
    1.25 * spacing,
    density=phx.discretization.ContinuityDensityPlan(), physical_viscosity=phx.discretization.MorrisViscosityPlan(viscosity), )
    compiled = phx.equations.compile_weakly_compressible_sph_problem(
        phx.equations.WeaklyCompressibleFluidProblemIR(
            "viscous-shear", phx.equations.TaitBarotropicMaterial(1.0, 10.0)
        ),
        particles,
        method,
        neighborhood=phx.discretization.CellListParticleNeighborhoodPlan(
            method.kernel.support_factor * method.smoothing_length,
            16,
            20 * count,
            box,
        ),
    )
    initial_state = compiled.initialize_state(position, velocity)
    initial = compiled.dynamics.diagnostics(0.0, initial_state, None)
    solution = _solve(compiled, position, velocity, final_time, 5.0e-4)
    final_position, final_velocity, _ = compiled.dynamics.state_layout.unpack(
        solution.states[-1]
    )
    final = compiled.dynamics.diagnostics(final_time, solution.states[-1], None)
    expected = (
        amplitude
        * jnp.exp(-viscosity * wave_number**2 * final_time)
        * jnp.sin(wave_number * final_position[:, 0])
    )
    return {
        "particle_count": count,
        "velocity_error": jnp.sqrt(jnp.mean((final_velocity[:, 1] - expected) ** 2)),
        "density_spread": final.density_maximum - final.density_minimum,
        "momentum_defect": jnp.max(
            jnp.abs(final.linear_momentum - initial.linear_momentum)
        ),
        "dissipation_rate": final.viscous_dissipation_rate,
        "energy_rate_defect": jnp.abs(
            final.total_energy_rate + final.viscous_dissipation_rate
        ),
    }


def run_wcsph_benchmark(acoustic_count=24, shear_resolution=8):
    dense, position, velocity = _acoustic_problem(acoustic_count, "dense")
    cell, _, _ = _acoustic_problem(acoustic_count, "cell")
    dense_state = dense.initialize_state(position, velocity)
    cell_state = cell.initialize_state(position, velocity)
    dense_rate = dense.dynamics(0.0, dense_state, None)
    cell_rate = cell.dynamics(0.0, cell_state, None)
    final_time = 0.005
    dense_solution = _solve(dense, position, velocity, final_time, 2.0e-4)
    cell_solution = _solve(cell, position, velocity, final_time, 2.0e-4)
    diagnostics = cell.dynamics.diagnostics(0.0, cell_state, None)

    def terminal(amplitude):
        shifted = position + amplitude * jnp.sin(2.0 * jnp.pi * position)
        solution = _solve(cell, shifted, velocity, 0.001, 5.0e-4)
        return solution.states[-1, 0, 0]

    amplitude = jnp.asarray(0.0)
    derivative = jax.grad(terminal)(amplitude)
    epsilon = jnp.asarray(1.0e-5)
    finite_difference = (
        terminal(amplitude + epsilon) - terminal(amplitude - epsilon)
    ) / (2.0 * epsilon)
    shear = _shear_case(shear_resolution)
    values = jnp.asarray(
        (
            jnp.max(jnp.abs(cell_rate - dense_rate)),
            jnp.max(jnp.abs(cell_solution.states - dense_solution.states)),
            jnp.abs(diagnostics.pressure_energy_balance_defect),
            jnp.abs(derivative - finite_difference),
            shear["velocity_error"],
            shear["density_spread"],
            shear["momentum_defect"],
            shear["dissipation_rate"],
            shear["energy_rate_defect"],
        )
    )
    return WCSPHBenchmarkRecord(
        schema_version=1,
        acoustic_particle_count=int(acoustic_count),
        acoustic_drift_parity_error=float(values[0]),
        acoustic_trajectory_parity_error=float(values[1]),
        acoustic_energy_balance_defect=float(values[2]),
        acoustic_gradient_error=float(values[3]),
        shear_particle_count=int(shear["particle_count"]),
        shear_velocity_error=float(values[4]),
        shear_density_spread=float(values[5]),
        shear_momentum_defect=float(values[6]),
        shear_dissipation_rate=float(values[7]),
        shear_energy_rate_defect=float(values[8]),
        finite=bool(jnp.all(jnp.isfinite(values))),
    )


def main():
    parser = argparse.ArgumentParser(description="Qualify periodic engineering WCSPH.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    record = run_wcsph_benchmark(
        acoustic_count=12 if arguments.smoke else 24,
        shear_resolution=4 if arguments.smoke else 8,
    )
    payload = json.dumps({**asdict(record), "passed": record.passed}, indent=2)
    if arguments.output is not None:
        temporary = arguments.output.with_suffix(arguments.output.suffix + ".tmp")
        temporary.write_text(payload + "\n")
        os.replace(temporary, arguments.output)
    print(payload)
    if not record.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
