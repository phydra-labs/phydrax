#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Advance one certified collision between two convex superquadrics."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
).prepare()
shapes = phx.discretization.SuperquadricSetPlan(
    jnp.asarray([[0.6, 0.4, 0.3], [0.6, 0.4, 0.3]]),
    jnp.asarray([2.5, 2.5]),
    jnp.asarray([3.0, 3.0]),
    jnp.zeros((2,), dtype=jnp.int32),
)
materials = phx.equations.DEMMaterialTable(
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
).prepare(
    particles,
    materials,
    phx.discretization.DenseParticleNeighborhoodPlan(1),
)
state = dynamics.initialize_state(
    jnp.asarray([[0.0, 0.0, 0.0], [1.18, 0.0, 0.0]]),
    jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
    jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    jnp.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
)
result = dynamics.step(state, jnp.asarray(0.0), jnp.asarray(1.0e-5))
evaluation = result.evaluation
print(f"successful={bool(result.successful)}")
print(f"gap={float(evaluation.geometry.gap[0]):.6e}")
print(f"contact_residual={float(evaluation.geometry.residual[0]):.6e}")
print(f"normal_force={float(evaluation.contact.normal_force[0, 0]):.6e}")
print(
    f"force_balance={float(jnp.linalg.norm(jnp.sum(evaluation.load.force, axis=0))):.6e}"
)
