#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Atomic-balance campaign for nondistributed reactive CFD-DEM coupling."""

import json

import jax.numpy as jnp

import phydrax as phx


def _problem():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=2
    ).prepare()
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.0]]),
    )
    dem = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "reactive-qualification", material, gravity=jnp.zeros((2,))
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(jnp.asarray([0.1]), jnp.asarray([0])),
        phx.discretization.SoftSphereDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.LinearSpringDashpotNormalPlan(1.0e3)
            )
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(0),
    )
    dem_state = dem.initialize_state(0.0, jnp.asarray([[0.0, 0.0]]), jnp.zeros((1, 2)))
    schema = phx.equations.ChemicalSpeciesSchema(
        ("solid",),
        (phx.equations.ChemicalPhaseKind.SOLID,),
        jnp.asarray([0.01]),
        ("X",),
        jnp.asarray([[1]]),
        jnp.zeros_like(jnp.asarray([0.01]), dtype=jnp.int32),
    )
    thermodynamics = phx.equations.ParticleThermodynamicMaterialPlan(
        phx.equations.PolynomialSpeciesThermodynamicsPlan(
            schema, jnp.asarray([10.0]), jnp.asarray([0.0])
        )
    )
    conversion_material = phx.equations.ParticleThermochemicalMaterialBundle(
        thermodynamics,
        phx.equations.ParticleTransportMaterialPlan(
            schema, jnp.asarray([1.0]), jnp.asarray([0.0])
        ),
    )
    batch_plan = phx.discretization.ParticleInternalBatchPlan(
        jnp.asarray([0]),
        phx.discretization.RadialShellMeshPlan(
            phx.discretization.ParticleInternalGeometry.SPHERE, 1
        ),
        1,
    )
    batch = batch_plan.prepare(particles)
    species = jnp.ones((1, 1, 1))
    internal = phx.discretization.initialize_particle_internal_batch(
        batch,
        thermodynamics.energy_from_temperature(jnp.asarray([[300.0]]), species),
        species,
        jnp.asarray([[0.2]]),
        jnp.ones((1, 1)),
        jnp.asarray([0.1]),
    )
    conversion = phx.equations.compile_particle_conversion_problem(
        phx.equations.ParticleConversionProblemIR(
            "reactive-conversion", (conversion_material,)
        ),
        particles,
        (batch_plan,),
    )
    transfer = phx.discretization.ConservativeParticleGridTransferPlan(
        jnp.asarray([[0.0, 0.0]]), jnp.asarray([1.0]), 0.5, 1
    ).prepare(particles)
    coupling = phx.equations.ReactiveCFDDEMCouplingPlan(
        dem.dynamics,
        conversion.dynamics,
        phx.equations.ParticleContinuumExchangePlan(
            transfer,
            jnp.asarray([1.0]),
            jnp.asarray([[0.0]]),
            schema_id=schema.schema_id,
        ),
    )
    state = phx.solver.initialize_reactive_cfd_dem(
        coupling,
        dem_state,
        conversion.initialize_state((internal,)),
        (jnp.asarray([500.0]), jnp.asarray([[0.0]])),
    )
    boundary = phx.equations.ParticleTransportBoundary(
        jnp.asarray([300.0]),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
        jnp.zeros((1,)),
        jnp.zeros((1, 1)),
    )
    return coupling, state, boundary


def _sample(fluid_state):
    return phx.solver.ReactiveFluidFields(
        jnp.zeros((1, 2)),
        jnp.ones((1,)),
        jnp.ones((1,)),
        jnp.zeros((1, 2)),
        fluid_state[0],
        fluid_state[1],
    )


def _update(fluid, momentum, energy, species, step_size):
    del momentum, step_size
    return fluid[0] + energy, fluid[1] + species


def _case(mode, iterations):
    coupling, state, boundary = _problem()
    schedule = phx.solver.ReactiveParticleCouplingSchedulePlan(
        phx.solver.ParticleConversionSolverPlan(
            phx.solver.ParticleConversionBackend.STRUCTURED_NATIVE
        ),
        dem_substeps=1,
        mode=mode,
        maximum_iterations=iterations,
        coupling_tolerance=1.0e-8,
    )
    result = phx.solver.advance_reactive_cfd_dem_window(
        coupling,
        schedule,
        state,
        _sample,
        _update,
        (boundary,),
        jnp.zeros((0,)),
        jnp.asarray([0.001]),
        jnp.asarray(0.0),
        jnp.asarray(1.0e-5),
    )
    evaluation = result.evaluation
    passed = (
        result.successful
        & (jnp.linalg.norm(evaluation.momentum_residual) <= 1.0e-12)
        & (jnp.abs(evaluation.energy_residual) <= 1.0e-12)
        & (jnp.max(jnp.abs(evaluation.species_residual)) <= 1.0e-12)
    )
    return {
        "mode": mode.value,
        "iterations": iterations,
        "momentum_residual": float(jnp.linalg.norm(evaluation.momentum_residual)),
        "energy_residual": float(evaluation.energy_residual),
        "maximum_species_residual": float(jnp.max(jnp.abs(evaluation.species_residual))),
        "coupling_residual": float(evaluation.coupling_residual),
        "passed": bool(passed),
    }


def main():
    cases = [
        _case(phx.solver.ReactiveCouplingMode.STRANG_FROZEN_FLUID, 1),
        _case(phx.solver.ReactiveCouplingMode.ITERATED_STAGGERED, 2),
    ]
    print(
        json.dumps(
            {
                "campaign": "reactive-cfd-dem-atomic-balance",
                "passed": all(case["passed"] for case in cases),
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
