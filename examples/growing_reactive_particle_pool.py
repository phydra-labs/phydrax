#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Grow a reactive DEM pool transactionally when insertion exhausts free slots."""

import jax.numpy as jnp
import jax.random as jr

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([10]), jnp.ones((1,)), ambient_dimension=2
).prepare()
material = phx.equations.DEMMaterialTable(
    jnp.asarray([1.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.8]]),
    jnp.asarray([[0.2]]),
)
compiled = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "growing-pool", material, gravity=jnp.zeros((2,))
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
dem_state = compiled.initialize_state(0.0, jnp.asarray([[0.0, 0.0]]), jnp.zeros((1, 2)))
epoch = phx.discretization.initialize_particle_execution_epoch(
    compiled.dynamics, dem_state
)
batch = phx.discretization.ParticleInternalBatchPlan(
    jnp.asarray([0]),
    phx.discretization.RadialShellMeshPlan(
        phx.discretization.ParticleInternalGeometry.SPHERE, 1
    ),
    1,
).prepare(particles)
internal = phx.discretization.initialize_particle_internal_batch(
    batch,
    jnp.ones((1, 1)),
    jnp.ones((1, 1, 1)),
    jnp.asarray([[0.2]]),
    jnp.ones((1, 1)),
    jnp.asarray([0.1]),
)
template = phx.discretization.ReactiveParticleTemplatePlan(
    0.1,
    1.0,
    0,
    jnp.zeros((2,)),
    jnp.zeros((1,)),
    jnp.asarray([1.0]),
    jnp.asarray([[1.0]]),
    jnp.asarray([0.2]),
    jnp.asarray([1.0]),
)
result = phx.discretization.insert_reactive_particles_with_growth(
    phx.discretization.ParticleInsertionPlan(
        jnp.asarray([1.0, -0.5]), jnp.asarray([2.0, 0.5]), 1
    ),
    phx.discretization.ReactiveParticleTemplateDistributionPlan(
        (template,), jnp.asarray([1.0])
    ),
    epoch,
    batch,
    internal,
    jnp.asarray([1.0]),
    jr.key(0),
    jnp.asarray(0.0),
    phx.discretization.ParticleCapacityGrowthPolicy(
        minimum_increment=2, maximum_capacity=16
    ),
)
print(f"successful={bool(result.successful)}")
print(f"old_capacity={particles.capacity}")
print(f"new_capacity={result.epoch.dynamics.bodies.capacity}")
print(f"occupied={int(jnp.sum(result.epoch.ever_occupied))}")
print(f"mass_residual={float(result.transition.mass_residual):.6e}")
