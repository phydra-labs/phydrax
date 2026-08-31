#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Single-process particle-conversion backend microbenchmarks."""

import json
import statistics
import time

import jax.numpy as jnp

import phydrax as phx


def _case(shells):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
    ).prepare()
    schema = phx.equations.ParticleSpeciesSchema(
        ("solid",),
        (phx.equations.ParticlePhase.SOLID,),
        jnp.asarray([0.01]),
        ("X",),
        jnp.asarray([[1]]),
    )
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        schema, jnp.asarray([10.0]), jnp.asarray([0.0])
    )
    material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermodynamics,
        phx.equations.ParticleTransportMaterialPlan(
            schema, jnp.asarray([1.0]), jnp.asarray([0.0])
        ),
    )
    batch_plan = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, shells
        ),
        1,
    )
    batch = batch_plan.prepare(particles)
    species = jnp.ones((1, shells, 1))
    internal = phx.discretization.initialize_particle_internal_batch(
        batch,
        thermodynamics.energy_from_temperature(
            jnp.linspace(300.0, 400.0, shells)[None, :], species
        ),
        species,
        jnp.full((1, shells), 0.2),
        jnp.ones((1, shells)),
        jnp.asarray([1.0]),
    )
    compiled = phx.equations.compile_particle_conversion_problem(
        phx.equations.ParticleConversionProblemIR("benchmark", (material,)),
        particles,
        (batch_plan,),
    )
    state = compiled.initialize_state((internal,))
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([500.0]),
        jnp.zeros((1, 1)),
        jnp.asarray([1.0]),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
    )
    return compiled, state, boundary


def _measure(compiled, state, boundary, backend, repetitions=3):
    plan = phx.solver.ParticleConversionSolverPlan(backend)

    def run():
        result = phx.solver.advance_particle_conversion(
            compiled.dynamics,
            plan,
            state,
            (boundary,),
            jnp.asarray(0.0),
            jnp.asarray(1.0e-4),
        )
        result.accepted_state.batches[0].internal_energy.block_until_ready()
        return result

    warm = run()
    durations = []
    for _ in range(repetitions):
        started = time.perf_counter()
        result = run()
        durations.append(time.perf_counter() - started)
    return {
        "backend": backend.value,
        "median_seconds": statistics.median(durations),
        "minimum_seconds": min(durations),
        "successful": bool(warm.successful & result.successful),
    }


def main():
    cases = []
    for shells in (2, 8, 32):
        compiled, state, boundary = _case(shells)
        measurements = [
            _measure(
                compiled,
                state,
                boundary,
                phx.solver.ParticleConversionBackend.REFERENCE_ROSENBROCK,
            ),
            _measure(
                compiled,
                state,
                boundary,
                phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE,
            ),
        ]
        cases.append(
            {
                "shells": shells,
                "measurements": measurements,
                "passed": all(value["successful"] for value in measurements),
            }
        )
    print(
        json.dumps(
            {
                "benchmark": "single-process-particle-conversion-backends",
                "passed": all(case["passed"] for case in cases),
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
