#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""One dense backward-Euler material-point step with implicit derivatives."""

import jax.numpy as jnp

import phydrax as phx


def run():
    grid = phx.discretization.TensorGridPlan(
        tuple(
            phx.discretization.UniformAxisSpec(8, periodic=True, endpoint=False)
            for _ in range(2)
        ),
        axis_names=("x", "y"),
    ).prepare(jnp.asarray([[0.0, 0.0], [1.0, 1.0]]))
    position = jnp.asarray([[0.27, 0.31], [0.43, 0.38], [0.36, 0.52]])
    volume = jnp.full((3,), 0.01)
    particles = phx.discretization.ParticleSetPlan(
        jnp.arange(3), volume, ambient_dimension=2
    ).prepare()
    splat = phx.discretization.ParticleGridSplatPlan(
        grid, assignment=phx.discretization.TensorBSplineSplatAssignment(2)
    ).prepare(particles)
    compiled = phx.equations.compile_material_point_problem(
        phx.equations.MaterialPointProblemIR(
            "implicit-example",
            phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(2),
        ),
        particles,
        splat,
        phx.discretization.ExplicitMPMMethodPlan(),
        phx.discretization.MPMParticleDomainPlan(
            jnp.asarray([[0.0, 0.0], [1.0, 1.0]]),
            periodic=(True, True),
            support_margin=0.0,
        ),
    )
    arguments = phx.equations.MaterialPointArguments(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0)
    )
    state = compiled.initialize_state(
        position,
        jnp.broadcast_to(jnp.asarray((0.03, -0.01)), position.shape),
        volume,
        arguments,
    )
    result = phx.solver.PreparedImplicitMPMDynamics(compiled.dynamics).step_detailed(
        state, 0.001, arguments
    )
    return {
        "successful": bool(result.successful),
        "residual_norm": float(result.diagnostics.residual_norm),
        "nonlinear_steps": int(result.diagnostics.nonlinear_steps),
        "linear_iterations": int(result.diagnostics.linear_iterations),
    }


if __name__ == "__main__":
    print(run())
