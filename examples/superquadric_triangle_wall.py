#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolve a rotating superquadric against a triangle wall."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.ones((1,)), ambient_dimension=3
).prepare()
shapes = phx.discretization.SuperquadricSetPlan(
    jnp.asarray([[0.6, 0.4, 0.3]]),
    jnp.asarray([2.0]),
    jnp.asarray([2.0]),
    jnp.asarray([0]),
)
wall = phx.discretization.TriangleWallPlan(
    jnp.asarray([[-2.0, -2.0, 0.0], [2.0, -2.0, 0.0], [0.0, 2.0, 0.0]]),
    jnp.asarray([[0, 1, 2]]),
    jnp.asarray([0]),
)
material = phx.equations.DEMMaterialTable(
    jnp.asarray([2.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.8]]),
    jnp.asarray([[0.3]]),
)
dynamics = phx.discretization.SuperquadricDEMPlan(
    shapes,
    phx.discretization.SuperquadricContactPlan(iterations=24),
    phx.discretization.DEMContactModelPlan(
        phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
        tangential=phx.discretization.CundallStrackTangentialPlan(2.0e3),
    ),
    walls=(wall,),
    wall_geometry=phx.discretization.SuperquadricTriangleContactPlan(),
).prepare(
    particles,
    material,
    phx.discretization.DenseParticleNeighborhoodPlan(0),
)
state = dynamics.initialize_state(
    jnp.asarray([[0.0, 0.0, 0.25]]),
    jnp.zeros((1, 3)),
    jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
    jnp.asarray([[0.0, 0.0, 0.5]]),
)
result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-5))
wall_response = result.evaluation.walls[0]
print(f"successful={bool(result.successful)}")
print(f"gap={float(wall_response.geometry.geometry.gap[0]):.6e}")
print(f"feature_kind={int(wall_response.geometry.feature_kind[0])}")
print(f"witness_residual={float(wall_response.geometry.witness_residual[0]):.6e}")
reaction_balance = jnp.linalg.norm(
    jnp.sum(wall_response.particle_load.force, axis=0) + wall_response.reaction_force
)
print(f"reaction_balance={float(reaction_balance):.6e}")
