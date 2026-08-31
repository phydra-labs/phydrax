#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rigid two-sphere clump pose and component realization."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0]), jnp.asarray([1.0]), ambient_dimension=3
).prepare()
template = phx.discretization.SphereClumpTemplatePlan(
    jnp.asarray([[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]]),
    jnp.asarray([0.1, 0.1]),
    jnp.asarray([0.5, 0.5]),
    jnp.asarray([0, 0]),
)
clumps = phx.discretization.RigidSphereClumpSetPlan(
    (template,), jnp.asarray([0]), jnp.asarray([0])
).prepare(particles)
kinematics = clumps.bodies.kinematics(
    jnp.asarray([[0.0, 0.0, 0.0]]),
    jnp.zeros((1, 3)),
    jnp.asarray([[1.0, 0.0, 0.0, 0.0]]),
    jnp.asarray([[0.0, 0.0, 1.0]]),
)
components = clumps.component_kinematics(kinematics)

print(f"prepared_id={clumps.prepared_id}")
print(f"component_count={int(jnp.sum(components.valid))}")
print(f"component_positions={components.position.tolist()}")
