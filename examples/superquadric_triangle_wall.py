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
    contact_geometry=phx.discretization.SuperquadricContactPlan(iterations=24),
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
compiled = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "superquadric-wall", material, gravity=jnp.zeros((3,)), walls=(wall,)
    ),
    particles,
    shapes,
    phx.discretization.CompliantDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
            tangential=phx.discretization.CundallStrackTangentialPlan(2.0e3),
        ),
        wall_geometry=phx.discretization.SuperquadricTriangleContactPlan(),
    ),
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(0),
)
state = compiled.initialize_state(
    0.0,
    jnp.asarray([[0.0, 0.0, 0.25]]),
    jnp.zeros((1, 3)),
    jnp.asarray([[0.0, 0.0, 0.5]]),
    orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
)
result = compiled.dynamics.step_detailed(
    jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, jnp.asarray(1.0e-5), None
)
wall_response = result.evaluation.boundaries[0]
if not bool(result.successful):
    raise RuntimeError("Superquadric triangle-wall step failed")
if not isinstance(
    wall_response.triangle_witness, phx.discretization.SuperquadricWallContactResult
):
    raise RuntimeError("Triangle-wall response omitted its witness evidence")
print("successful=True")
print(f"gap={float(wall_response.geometry.gap[0]):.6e}")
print(f"feature_kind={int(wall_response.triangle_witness.feature_kind[0])}")
print(f"witness_residual={float(wall_response.geometry.residual[0]):.6e}")
reaction_balance = jnp.linalg.norm(
    jnp.sum(wall_response.particle_force, axis=0) + wall_response.reaction_force
)
print(f"reaction_balance={float(reaction_balance):.6e}")
