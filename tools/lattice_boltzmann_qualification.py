"""Deterministic accuracy and runtime qualification for lattice-Boltzmann flow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np

import phydrax as phx
from benchmarks._runtime import (
    capture_environment,
    compiler_evidence,
    logical_array_bytes,
    measure_lower_and_compile,
    measure_repeated,
    synchronize,
)


def _grid(shape, lengths, periodic):
    dimension = len(shape)
    return phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=periodic[axis])
            for axis, count in enumerate(shape)
        ),
        axis_names=tuple("xyz"[:dimension]),
    ).prepare(jnp.asarray(((0.0,) * dimension, lengths)))


def _compiled(
    shape,
    lengths,
    periodic,
    velocity_set,
    collision,
    *,
    physical_viscosity,
    lattice_viscosity=0.1,
    acceleration=None,
    moving_faces=(),
):
    grid = _grid(shape, lengths, periodic)
    discretization = phx.discretization.LatticeBoltzmannPlan(grid, velocity_set).prepare()
    cell_size = float(discretization.cell_size)
    time_step = lattice_viscosity * cell_size**2 / physical_viscosity
    forcing = None if acceleration is None else phx.discretization.GuoForcingPlan()
    method = phx.discretization.LatticeBoltzmannMethodPlan(collision, forcing=forcing)
    problem = phx.equations.LatticeBoltzmannProblem(
        "qualification-flow",
        len(shape),
        acceleration=(
            None
            if acceleration is None
            else lambda time, coordinates, parameters: parameters
        ),
        acceleration_id=None if acceleration is None else "constant-acceleration",
    )
    boundary = phx.discretization.LatticeBoltzmannBoundaryPlan(moving_faces=moving_faces)
    compiled = phx.equations.compile_lattice_boltzmann_problem(
        problem,
        discretization,
        method,
        boundary,
        time_step=time_step,
    )
    parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
        physical_viscosity,
        force_parameters=acceleration,
        moving_wall_velocities=jnp.zeros((len(moving_faces), len(shape))),
    )
    return compiled, parameters


def _rollout(compiled, parameters, initial, steps):
    method = phx.solver.LatticeBoltzmannFixedStepMethod(compiled.dynamics)
    dt = float(compiled.scaling.time_step)
    problem = phx.solver.FixedStepProblem(
        method,
        initial,
        t0=0.0,
        t1=steps * dt,
        step_size=dt,
        args=parameters,
        discretization_bundle=compiled.discretization_bundle,
    )
    return phx.solver.FixedStepRolloutPlan(retention="final").rollout(problem)


def _observed_order(spacing, errors):
    valid = np.asarray(errors) > 0.0
    if np.sum(valid) < 2:
        return None
    return float(
        np.polyfit(
            np.log(np.asarray(spacing)[valid]), np.log(np.asarray(errors)[valid]), 1
        )[0]
    )


def _shear_decay_case():
    physical_viscosity = 0.02
    amplitude = 0.05
    final_time = 0.1
    rows = []
    for resolution in (16, 24, 32):
        compiled, parameters = _compiled(
            (resolution, resolution),
            (1.0, 1.0),
            (True, True),
            phx.discretization.D2Q9(),
            phx.discretization.BGKCollisionPlan(),
            physical_viscosity=physical_viscosity,
        )
        y = compiled.discretization.grid.structured_axes[1].interval_centers
        profile = amplitude * jnp.sin(2.0 * jnp.pi * y)
        velocity = (
            jnp.zeros((resolution, resolution, 2))
            .at[..., 0]
            .set(jnp.broadcast_to(profile, (resolution, resolution)))
        )
        initial = compiled.initialize_state(1.0, velocity, parameters)
        dt = float(compiled.scaling.time_step)
        steps = max(1, int(round(final_time / dt)))
        result = _rollout(compiled, parameters, initial, steps)
        elapsed = steps * dt
        macros = compiled.macroscopic_state(elapsed, result.final_state, parameters)
        mode = jnp.sin(2.0 * jnp.pi * y)
        recovered = float(2.0 * jnp.mean(macros.velocity[..., 0] * mode[None, :]))
        expected = amplitude * np.exp(-physical_viscosity * (2.0 * np.pi) ** 2 * elapsed)
        error = abs(recovered - expected) / abs(expected)
        rows.append(
            {
                "resolution": resolution,
                "cell_size": float(compiled.scaling.cell_size),
                "time_step": dt,
                "steps": steps,
                "elapsed_time": elapsed,
                "recovered_amplitude": recovered,
                "expected_amplitude": expected,
                "relative_error": error,
                "mass_defect": float(result.residuals[-1]),
                "successful": bool(result.successful),
            }
        )
    order = _observed_order(
        [row["cell_size"] for row in rows], [row["relative_error"] for row in rows]
    )
    return {
        "cases": rows,
        "observed_order": order,
        "finest_relative_error": rows[-1]["relative_error"],
        "passed": bool(
            order is not None
            and order >= 1.8
            and rows[-1]["relative_error"] <= 5e-3
            and all(row["successful"] for row in rows)
        ),
    }


def _d3q19_case():
    resolution = 16
    physical_viscosity = 0.02
    amplitude = 0.03
    compiled, parameters = _compiled(
        (resolution, resolution, resolution),
        (1.0, 1.0, 1.0),
        (True, True, True),
        phx.discretization.D3Q19(),
        phx.discretization.TRTCollisionPlan(),
        physical_viscosity=physical_viscosity,
    )
    z = compiled.discretization.grid.structured_axes[2].interval_centers
    velocity = (
        jnp.zeros((resolution, resolution, resolution, 3))
        .at[..., 0]
        .set(
            jnp.broadcast_to(
                amplitude * jnp.sin(2.0 * jnp.pi * z),
                (resolution, resolution, resolution),
            )
        )
    )
    initial = compiled.initialize_state(1.0, velocity, parameters)
    dt = float(compiled.scaling.time_step)
    steps = max(1, int(round(0.1 / dt)))
    result = _rollout(compiled, parameters, initial, steps)
    elapsed = steps * dt
    macros = compiled.macroscopic_state(elapsed, result.final_state, parameters)
    mode = jnp.sin(2.0 * jnp.pi * z)
    recovered = float(2.0 * jnp.mean(macros.velocity[..., 0] * mode[None, None, :]))
    expected = amplitude * np.exp(-physical_viscosity * (2.0 * np.pi) ** 2 * elapsed)
    diagnostics = compiled.dynamics.scalar_diagnostics(
        jnp.asarray(steps - 1),
        jnp.asarray(elapsed),
        result.final_state,
        parameters,
    )
    relative_error = abs(recovered - expected) / abs(expected)
    return {
        "shape": [resolution, resolution, resolution],
        "time_step": dt,
        "steps": steps,
        "elapsed_time": elapsed,
        "relative_error": relative_error,
        "maximum_mach": float(diagnostics.maximum_mach),
        "relative_mass_drift": float(
            abs(diagnostics.total_mass - resolution**3) / resolution**3
        ),
        "minimum_density": float(diagnostics.minimum_density),
        "minimum_population": float(diagnostics.minimum_population),
        "successful": bool(result.successful),
        "passed": bool(
            result.successful
            and relative_error <= 1e-2
            and diagnostics.maximum_mach <= 0.05
            and diagnostics.minimum_density > 0.0
        ),
    }


def _channel_case(*, couette):
    physical_viscosity = 0.02
    acceleration = None if couette else jnp.asarray((0.001, 0.0))
    wall_speed = 0.01 if couette else 0.0
    rows = []
    for height in (8, 12, 16):
        compiled, parameters = _compiled(
            (4 * height, height),
            (4.0, 1.0),
            (True, False),
            phx.discretization.D2Q9(),
            (
                phx.discretization.BGKCollisionPlan()
                if couette
                else phx.discretization.TRTCollisionPlan()
            ),
            physical_viscosity=physical_viscosity,
            acceleration=acceleration,
            moving_faces=(("y", "upper"),) if couette else (),
        )
        parameters = phx.discretization.LatticeBoltzmannRuntimeParameters(
            physical_viscosity,
            force_parameters=acceleration,
            moving_wall_velocities=(
                jnp.asarray(((wall_speed, 0.0),)) if couette else jnp.empty((0, 2))
            ),
        )
        initial = compiled.initialize_state(1.0, jnp.zeros((2,)), parameters)
        dt = float(compiled.scaling.time_step)
        target_time = 100.0
        steps = int(round(target_time / dt))
        result = _rollout(compiled, parameters, initial, steps)
        elapsed = steps * dt
        macros = compiled.macroscopic_state(elapsed, result.final_state, parameters)
        y = np.asarray(compiled.discretization.grid.structured_axes[1].interval_centers)
        numerical = np.asarray(jnp.mean(macros.velocity[..., 0], axis=0))
        expected = (
            wall_speed * y
            if couette
            else float(acceleration[0]) * y * (1.0 - y) / (2.0 * physical_viscosity)
        )
        relative_l2 = float(
            np.linalg.norm(numerical - expected) / max(np.linalg.norm(expected), 1e-15)
        )
        diagnostics = compiled.dynamics.scalar_diagnostics(
            jnp.asarray(steps - 1),
            jnp.asarray(elapsed),
            result.final_state,
            parameters,
        )
        rows.append(
            {
                "height": height,
                "cell_size": float(compiled.scaling.cell_size),
                "time_step": dt,
                "steps": steps,
                "relative_l2": relative_l2,
                "maximum_mach": float(diagnostics.maximum_mach),
                "relative_mass_drift": float(
                    abs(diagnostics.total_mass - height * 4 * height)
                    / (height * 4 * height)
                ),
                "minimum_density": float(diagnostics.minimum_density),
                "minimum_population": float(diagnostics.minimum_population),
                "successful": bool(result.successful),
            }
        )
    order = _observed_order(
        [row["cell_size"] for row in rows], [row["relative_l2"] for row in rows]
    )
    converged = order is not None and (order >= 1.8 or rows[-1]["relative_l2"] <= 1e-8)
    passed = (
        converged
        and rows[-1]["relative_l2"] <= 1e-2
        and all(row["successful"] for row in rows)
        and all(row["maximum_mach"] <= 0.05 for row in rows)
        and all(row["relative_mass_drift"] <= 1e-10 for row in rows)
        and all(row["minimum_density"] > 0.0 for row in rows)
    )
    return {
        "cases": rows,
        "observed_order": order,
        "finest_relative_l2": rows[-1]["relative_l2"],
        "passed": bool(passed),
    }


def _runtime_case(*, warmup, repeats):
    resolution = 128
    compiled, parameters = _compiled(
        (resolution, resolution),
        (1.0, 1.0),
        (True, True),
        phx.discretization.D2Q9(),
        phx.discretization.BGKCollisionPlan(),
        physical_viscosity=0.02,
    )
    initial = compiled.initialize_state(1.0, jnp.asarray((0.01, 0.0)), parameters)
    step_index = jnp.asarray(0, dtype=jnp.int32)
    time = jnp.asarray(0.0)
    step_size = jnp.asarray(compiled.scaling.time_step)

    def step(populations):
        return compiled.dynamics.step_detailed(
            step_index, time, populations, step_size, parameters
        ).accepted_state

    jitted = eqx.filter_jit(step)
    compiled_step, timing = measure_lower_and_compile(
        lambda: jitted.lower(initial), lambda lowered: lowered.compile()
    )
    final, distribution = measure_repeated(
        lambda: compiled_step(initial), warmup=warmup, repeats=repeats
    )
    synchronize(final)
    executable = compiled_step.compiled
    cost = executable.cost_analysis()
    memory = executable.memory_analysis()
    evidence = compiler_evidence(cost, memory, source="jax-compiled-executable")
    median = distribution.median_seconds
    updates = resolution * resolution
    return {
        "shape": [resolution, resolution],
        "population_count": 9,
        "fluid_cells": updates,
        "discretization_id": compiled.discretization.prepared_id,
        "problem_id": compiled.problem.problem_id,
        "method_id": compiled.method.method_id,
        "boundary_id": compiled.boundary.boundary_id,
        "compilation_id": compiled.compilation_id,
        "logical_state_bytes": logical_array_bytes(initial),
        "lowering_seconds": timing.lowering_seconds,
        "compilation_seconds": timing.compilation_seconds,
        "timing": distribution.to_milliseconds_dict(),
        "total_site_mlups": None if median is None else updates / median / 1e6,
        "fluid_site_mlups": None if median is None else updates / median / 1e6,
        "compiler": {
            "flops": evidence.flops,
            "bytes_accessed": evidence.bytes_accessed,
            "argument_bytes": evidence.argument_bytes,
            "output_bytes": evidence.output_bytes,
            "temporary_bytes": evidence.temporary_bytes,
            "generated_code_bytes": evidence.generated_code_bytes,
            "estimated_device_memory_bytes": evidence.estimated_device_memory_bytes,
            "source": evidence.source,
        },
    }


def qualification(*, warmup=2, repeats=5):
    return {
        "environment": capture_environment().to_dict(),
        "scope": {
            "dtypes": ["float64"],
            "lattices": ["D2Q9", "D3Q19"],
            "collisions": ["BGK", "TRT"],
            "forcing": "Guo",
            "geometry": "periodic and planar halfway walls",
        },
        "shear_decay": _shear_decay_case(),
        "d3q19_shear": _d3q19_case(),
        "poiseuille": _channel_case(couette=False),
        "couette": _channel_case(couette=True),
        "runtime": _runtime_case(warmup=warmup, repeats=repeats),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/lattice_boltzmann_qualification.json"),
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    arguments = parser.parse_args()
    payload = qualification(warmup=arguments.warmup, repeats=arguments.repeats)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
