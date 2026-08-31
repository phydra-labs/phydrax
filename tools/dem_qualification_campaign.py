#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Reference restitution/timestep campaign for spherical DEM qualification."""

import json

import jax
import jax.numpy as jnp

import phydrax as phx


def _compiled(restitution):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[restitution]]),
        jnp.asarray([[0.0]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        maximum_overlap_fraction=0.2,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "restitution-campaign", materials, gravity=jnp.zeros((2,))
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )


def _case(restitution, step_size):
    compiled = _compiled(restitution)
    initial = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [1.02, 0.0]]),
        jnp.asarray([[0.5, 0.0], [-0.5, 0.0]]),
    )
    problem = phx.solver.FixedStepProblem(
        phx.solver.DEMFixedStepMethod(compiled.dynamics),
        initial,
        t0=0.0,
        t1=0.06,
        step_size=step_size,
        state_geometry=compiled.dynamics.state_geometry,
        discretization_bundle=compiled.discretization_bundle,
    )
    solution = phx.solver.solve_fixed_step(problem, save_every=problem.step_count)
    final = jax.tree.map(lambda value: value[-1], solution.states)
    measured = final.kinematics.velocity[1, 0] - final.kinematics.velocity[0, 0]
    diagnostics = compiled.diagnostics(0.06, final)
    profile = phx.discretization.DEMQualificationProfile(
        maximum_overlap_fraction=0.2,
        energy_balance_tolerance=1.0e-9,
    )
    qualification = phx.discretization.qualify_dem(diagnostics, profile)
    return {
        "target_restitution": restitution,
        "step_size": step_size,
        "measured_restitution": float(measured),
        "absolute_error": float(jnp.abs(measured - restitution)),
        "relative_energy_residual": float(
            diagnostics.energy.last_relative_energy_residual
        ),
        "contact_balance_loss": float(diagnostics.energy.cumulative_contact_balance_loss),
        "successful": bool(solution.successful),
        "qualified": bool(qualification.qualified),
        "method_id": compiled.dynamics.method.method_id,
        "bundle_id": compiled.discretization_bundle.bundle_id,
        "qualification_artifact_id": qualification.artifact_id,
    }


def main():
    cases = [
        _case(restitution, step_size)
        for restitution in (0.2, 0.5, 0.8, 0.95)
        for step_size in (4.0e-4, 2.0e-4, 1.0e-4)
    ]
    passed = all(value["successful"] for value in cases) and all(
        value["absolute_error"] <= 0.05 for value in cases if value["step_size"] == 1.0e-4
    )
    print(
        json.dumps(
            {
                "campaign": "linear-normal-restitution-refinement",
                "passed": passed,
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
