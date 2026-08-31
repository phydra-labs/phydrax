#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conservative unresolved Stokes CFD--DEM exchange."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=2
).prepare()
spheres = phx.discretization.RigidSphereSetPlan(jnp.asarray([0.1]), jnp.asarray([0]))
materials = phx.equations.DEMMaterialTable(
    jnp.asarray([1.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.9]]),
    jnp.asarray([[0.0]]),
)
method = phx.discretization.SoftSphereDEMMethodPlan(
    phx.discretization.DEMContactModelPlan(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e3)
    )
)
problem = phx.equations.DiscreteElementProblemIR(
    "one-particle-cfd-dem", materials, gravity=jnp.zeros((2,))
)
compiled = phx.equations.compile_discrete_element_problem(
    problem,
    particles,
    spheres,
    method,
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(0),
)
state = compiled.initialize_state(0.0, jnp.asarray([[0.0, 0.0]]), jnp.zeros((1, 2)))
transfer = phx.discretization.ConservativeParticleGridTransferPlan(
    jnp.asarray([[0.0, 0.0]]), jnp.asarray([1.0]), 0.5, 1
).prepare(particles)
coupling = phx.equations.UnresolvedCFDEMCouplingPlan(
    compiled.dynamics,
    transfer,
    phx.equations.StokesDragPlan(maximum_reynolds=1.0),
)
evaluation = phx.equations.evaluate_unresolved_cfd_dem(
    coupling,
    state,
    jnp.asarray([[0.01, 0.0]]),
    jnp.asarray([1.0]),
    jnp.asarray([10.0]),
    jnp.zeros((1, 2)),
    jnp.asarray([0.01]),
    jnp.asarray(1.0e-3),
)

print(f"successful={bool(evaluation.successful)}")
print(f"particle_force={evaluation.particle_force.tolist()}")
print(f"fluid_source={evaluation.fluid_momentum_source_rate.tolist()}")
print(f"momentum_residual={evaluation.momentum_residual.tolist()}")
