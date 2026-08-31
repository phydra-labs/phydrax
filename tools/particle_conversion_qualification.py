#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservation and backend-agreement campaign for particle conversion."""

import json

import jax.numpy as jnp

import phydrax as phx


def _mesh_case(geometry, shells):
    mesh = phx.discretization.RadialShellMeshPlan(geometry, shells).prepare()
    scale = jnp.asarray([2.0])
    metrics = mesh.metrics(scale)
    if geometry is phx.discretization.ParticleInternalGeometry.SLAB:
        exact = 2.0
    elif geometry is phx.discretization.ParticleInternalGeometry.CYLINDER:
        exact = 4.0 * jnp.pi
    else:
        exact = (32.0 / 3.0) * jnp.pi
    relative_error = jnp.abs(jnp.sum(metrics.cell_measures) - exact) / exact
    return {
        "geometry": geometry.value,
        "shells": shells,
        "relative_measure_error": float(relative_error),
        "passed": bool(relative_error <= 1.0e-13),
    }


def _conversion_case(shells, step_size):
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
    temperatures = jnp.linspace(300.0, 400.0, shells)[None, :]
    internal = phx.discretization.initialize_particle_internal_batch(
        batch,
        thermodynamics.energy_from_temperature(temperatures, species),
        species,
        jnp.full((1, shells), 0.2),
        jnp.ones((1, shells)),
        jnp.asarray([1.0]),
    )
    compiled = phx.equations.compile_particle_conversion_problem(
        phx.equations.ParticleConversionProblemIR("qualification", (material,)),
        particles,
        (batch_plan,),
    )
    initial = compiled.initialize_state((internal,))
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([500.0]),
        jnp.zeros((1, 1)),
        jnp.asarray([1.0]),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
    )
    reference = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.REFERENCE_ROSENBROCK,
            substeps=2,
        ),
        initial,
        (boundary,),
        jnp.asarray(0.0),
        jnp.asarray(step_size),
    )
    structured = phx.solver.advance_particle_conversion(
        compiled.dynamics,
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE,
            substeps=2,
        ),
        initial,
        (boundary,),
        jnp.asarray(0.0),
        jnp.asarray(step_size),
    )
    reference_energy = reference.accepted_state.batches[0].internal_energy
    structured_energy = structured.accepted_state.batches[0].internal_energy
    scale = jnp.maximum(jnp.linalg.norm(reference_energy), 1.0)
    relative_difference = jnp.linalg.norm(reference_energy - structured_energy) / scale
    passed = (
        reference.successful
        & structured.successful
        & (jnp.abs(reference.replay.internal_energy_residual) <= 1.0e-9)
        & (jnp.abs(structured.replay.internal_energy_residual) <= 1.0e-9)
        & (relative_difference <= 2.0e-3)
    )
    return {
        "shells": shells,
        "step_size": step_size,
        "reference_energy_residual": float(reference.replay.internal_energy_residual),
        "structured_energy_residual": float(structured.replay.internal_energy_residual),
        "backend_relative_difference": float(relative_difference),
        "passed": bool(passed),
    }


def main():
    mesh_cases = [
        _mesh_case(geometry, shells)
        for geometry in phx.discretization.ParticleInternalGeometry
        for shells in (1, 4, 16)
    ]
    conversion_cases = [
        _conversion_case(shells, step_size)
        for shells in (2, 4, 8)
        for step_size in (1.0e-4, 5.0e-5)
    ]
    passed = all(case["passed"] for case in mesh_cases + conversion_cases)
    print(
        json.dumps(
            {
                "campaign": "particle-conversion-conservation-and-backend-agreement",
                "passed": passed,
                "mesh_cases": mesh_cases,
                "conversion_cases": conversion_cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
