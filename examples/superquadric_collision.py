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
    contact_geometry=phx.discretization.SuperquadricContactPlan(iterations=24),
)
materials = phx.equations.DEMMaterialTable(
    jnp.asarray([2.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.8]]),
    jnp.asarray([[0.3]]),
)
compiled = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "superquadric-collision", materials, gravity=jnp.zeros((3,))
    ),
    particles,
    shapes,
    phx.discretization.CompliantDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4),
            tangential=phx.discretization.CundallStrackTangentialPlan(2.0e3),
        )
    ),
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
)
state = compiled.initialize_state(
    0.0,
    jnp.asarray([[0.0, 0.0, 0.0], [1.18, 0.0, 0.0]]),
    jnp.asarray([[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]),
    jnp.asarray([[0.0, 0.0, 0.5], [0.0, 0.0, -0.5]]),
    orientation=jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
)
result = compiled.dynamics.step_detailed(
    jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0.0), state, jnp.asarray(1.0e-5), None
)
evaluation = result.evaluation
if not bool(result.successful):
    raise RuntimeError("Superquadric collision step failed")
print("successful=True")
print(f"gap={float(evaluation.geometry.geometry.gap[0]):.6e}")
print(f"contact_residual={float(evaluation.geometry.geometry.residual[0]):.6e}")
print(f"normal_force={float(evaluation.particle_contact.normal_force[0, 0]):.6e}")
print(
    f"force_balance={float(jnp.linalg.norm(jnp.sum(evaluation.loads.total.force, axis=0))):.6e}"
)
