from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
from _runtime import (
    capture_environment,
    measure_repeated,
    measure_synchronized,
)

import phydrax as phx
from phydrax.applications.cosmology._coupled import ComovingEulerPlan
from phydrax.applications.cosmology._mixed_matter import (
    WaveParticleGasCosmologyPlan,
    WaveParticleGasCosmologyState,
)
from phydrax.applications.cosmology._particles import CosmologicalKDKPlan
from phydrax.applications.cosmology._wave_dark_matter import (
    WaveDarkMatterPlan,
    WaveDarkMatterStepPolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if (
        arguments.size < 4
        or arguments.steps < 1
        or arguments.warmup < 0
        or arguments.repeats < 1
    ):
        raise ValueError(
            "size >= 4, positive steps/repeats, and nonnegative warmup are required"
        )

    count = arguments.size
    shape = (count,) * 3
    axis_names = ("x", "y", "z")
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformCellAxisSpec(count, periodic=True) for _ in range(3)
        ),
        axis_names=axis_names,
    ).prepare(jnp.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))
    system = phx.equations.EulerSystem(3)
    discretization = phx.discretization.FiniteVolumePlan(
        grid, component_names=system.component_names
    ).prepare()
    problem = phx.equations.ConservationProblemIR(
        "mixed-dark-matter-benchmark",
        "state",
        system,
        phx.discretization.FiniteVolumeBoundarySet.periodic(axis_names),
    )
    dynamics = phx.equations.compile_conservation_problem(
        problem,
        discretization,
        phx.discretization.FiniteVolumeMethodPlan(
            phx.discretization.PiecewiseConstantReconstruction(),
            phx.discretization.HLLCFluxPlan(),
        ),
    ).dynamics
    runtime = phx.solver.PreparedFiniteVolumeRuntime(
        dynamics,
        phx.discretization.FluxPositivityPlan(),
        phx.solver.FiniteVolumeStepPolicy(cfl=0.3, maximum_retries=0),
    )
    gravity_owner = phx.solver.NewtonianSelfGravityPlan(0.02).prepare(
        phx.solver.prepare_balance_law_transport(runtime)
    )

    coordinates = jnp.meshgrid(
        *(grid.structured_axes[axis].interval_centers for axis in range(3)),
        indexing="ij",
    )
    positions = jnp.stack(tuple(value.reshape((-1,)) for value in coordinates), axis=-1)
    capacity = positions.shape[0]
    particle_support = phx.discretization.ParticleSetPlan(
        jnp.arange(capacity),
        jnp.full((capacity,), 1.0 / capacity),
        ambient_dimension=3,
    ).prepare()
    transfer = phx.discretization.ParticleGridSplatPlan(grid).prepare(particle_support)
    particle_gravity = phx.solver.ParticleMeshGravityPlan(gravity_owner, transfer)
    kdk = CosmologicalKDKPlan(particle_support, (1.0, 1.0, 1.0))

    half_cell = 0.5 / count
    space = phx.discretization.TensorSpectralPlan(
        tuple(phx.discretization.FourierBasisPlan(count) for _ in range(3)),
        axis_names=axis_names,
        field_name="psi",
    ).prepare(
        tuple(
            phx.discretization.AxisDomain.periodic(half_cell, 1.0 + half_cell)
            for _ in range(3)
        )
    )
    background = phx.applications.cosmology.FLRWBackground(1.0, 1.0)
    schedule = 0.5 + 1.0e-5 * jnp.arange(arguments.steps + 1)
    wave = WaveDarkMatterPlan(
        1.0,
        schedule,
        gravitational_constant=0.02,
        reduced_planck_constant=0.03,
        step_policy=WaveDarkMatterStepPolicy(
            maximum_phase_radians=2.0,
            minimum_de_broglie_cells=2.0,
            norm_relative_tolerance=1.0e-6,
        ),
    ).prepare(space, background)
    gas = ComovingEulerPlan(
        dynamics,
        adiabatic_index=5.0 / 3.0,
        expansion_dimension=3,
        substeps=4,
    )
    prepared = WaveParticleGasCosmologyPlan(wave, kdk, gas, particle_gravity).prepare()

    x, y, z = coordinates
    perturbation = (
        1.0e-3
        * (
            jnp.cos(2.0 * jnp.pi * x)
            + jnp.cos(2.0 * jnp.pi * y)
            + jnp.cos(2.0 * jnp.pi * z)
        )
        / 3.0
    )
    wave_state = wave.initialize(jnp.sqrt(1.0 + perturbation).astype(jnp.complex128))
    particle_state = kdk.initialize(
        positions,
        jnp.zeros_like(positions),
        schedule[0],
    )
    gas_average = jnp.zeros(shape + (5,))
    gas_average = gas_average.at[..., 0].set(1.0 + perturbation)
    gas_average = gas_average.at[..., -1].set(1.0 + perturbation)
    gas_state = gas.initialize(gas_average, schedule[0])
    state = WaveParticleGasCosmologyState(wave_state, particle_state, gas_state)

    first_result, first_execution_seconds = measure_synchronized(
        lambda: prepared.rollout(state)
    )
    result, execution = measure_repeated(
        lambda: prepared.rollout(state),
        warmup=arguments.warmup,
        repeats=arguments.repeats,
    )
    if not bool(first_result.successful):
        raise RuntimeError("The first mixed benchmark rollout was rejected.")
    diagnostics = result.diagnostics
    payload = {
        "environment": capture_environment().to_dict(),
        "configuration": {
            "shape": shape,
            "particle_capacity": capacity,
            "steps": arguments.steps,
            "gas_substeps": gas.substeps,
            "wave_evaluation_shape": wave.dealiasing.report.evaluation_shape,
            "warmup": arguments.warmup,
            "repeats": arguments.repeats,
        },
        "identity": prepared.prepared_id,
        "compilation": {
            "execution_model": "fixed-shape-jax-scan",
            "first_compile_and_execution_seconds": first_execution_seconds,
        },
        "execution": execution.to_seconds_dict(),
        "physics": {
            "completed": bool(result.successful),
            "accepted_steps": int(diagnostics.accepted_steps),
            "maximum_mass_balance_defect": float(
                jnp.max(jnp.abs(diagnostics.mass_balance_defect))
            ),
            "maximum_source_integral": float(
                jnp.max(jnp.abs(diagnostics.source_integral))
            ),
            "maximum_poisson_relative_residual": float(
                jnp.max(diagnostics.poisson_relative_residual)
            ),
            "maximum_total_force_norm": float(
                jnp.max(jnp.sqrt(jnp.sum(diagnostics.total_force**2, axis=-1)))
            ),
            "maximum_wave_norm_relative_error": float(
                jnp.max(diagnostics.wave_norm_relative_error)
            ),
            "maximum_particle_force_adjoint_defect": float(
                jnp.max(diagnostics.particle_force_adjoint_defect)
            ),
            "total_gravity_work": float(jnp.sum(diagnostics.total_gravity_work)),
            "all_time_levels_consistent": bool(
                jnp.all(diagnostics.time_level_consistent)
            ),
            "all_gas_states_positive": bool(
                jnp.all(
                    diagnostics.gas_density_positive & diagnostics.gas_pressure_positive
                )
            ),
        },
        "passed": bool(first_result.successful)
        and bool(result.successful)
        and bool(jnp.all(diagnostics.time_level_consistent))
        and bool(
            jnp.all(
                diagnostics.gas_density_positive & diagnostics.gas_pressure_positive
            )
        ),
    }
    encoded = json.dumps(payload, allow_nan=False, indent=2)
    if arguments.output is None:
        print(encoded)
    else:
        from benchmarks._io import write_json_atomic

        write_json_atomic(arguments.output, payload)
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
