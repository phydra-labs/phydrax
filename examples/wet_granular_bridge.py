#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Resolve a wet two-sphere contact with capillarity and lubrication."""

import jax.numpy as jnp

import phydrax as phx


particles = phx.discretization.ParticleSetPlan(
    jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
).prepare()
materials = phx.equations.DEMMaterialTable(
    jnp.asarray([2.0e5]),
    jnp.asarray([0.25]),
    jnp.asarray([[0.8]]),
    jnp.asarray([[0.4]]),
    rolling_friction=jnp.asarray([[0.1]]),
)
cohesion = phx.discretization.CompositeDEMCohesionPlan(
    (
        phx.discretization.DMTContactCohesionPlan(0.05, 0.1),
        phx.discretization.LinearCapillaryBridgePlan(0.07, 0.0, 1.0e-9, 0.1),
        phx.discretization.NearContactLubricationPlan(1.0e-3, 0.1, 1.0e-5),
    )
)
contact = phx.discretization.DEMContactModelPlan(
    phx.discretization.HertzNormalContactPlan(),
    cohesion=cohesion,
    tangential=phx.discretization.CundallStrackTangentialPlan(2.0e3),
    rotational=phx.discretization.ElasticRollingTorsionalResistancePlan(
        100.0, 50.0, torsional_friction=0.05
    ),
)
compiled = phx.equations.compile_discrete_element_problem(
    phx.equations.DiscreteElementProblemIR(
        "wet-bridge", materials, gravity=jnp.zeros((3,))
    ),
    particles,
    phx.discretization.RigidSphereSetPlan(
        jnp.full((2,), 0.5), jnp.zeros((2,), dtype=jnp.int32)
    ),
    phx.discretization.SoftSphereDEMMethodPlan(contact),
    neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
)
state = compiled.initialize_state(
    0.0,
    jnp.asarray([[0.0, 0.0, 0.0], [0.99, 0.0, 0.0]]),
    jnp.asarray([[-0.02, 0.0, 0.0], [0.02, 0.0, 0.0]]),
    jnp.asarray([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]]),
)
result = None
for step in range(5):
    result = compiled.dynamics.step_detailed(
        jnp.asarray(step, dtype=jnp.int32),
        jnp.asarray(step * 1.0e-5),
        state,
        jnp.asarray(1.0e-5),
        None,
    )
    state = result.accepted_state

response = result.evaluation.particle_contact
print(f"successful={bool(result.successful)}")
print(f"interaction_range={compiled.dynamics.contact_model.interaction_range:.6e}")
print(f"normal_force={float(response.normal_force[0, 0]):.6e}")
print(f"bridge_active={bool(response.next_history.cohesion.components[1].active[0])}")
print(f"bridge_volume_residual={float(response.bridge_volume_residual[0]):.6e}")
