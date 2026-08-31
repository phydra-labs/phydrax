#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exercise rigid friction and sharp/diffuse fracture state contracts."""

import jax.numpy as jnp

import phydrax as phx


def run():
    contact = phx.discretization.RigidMPMContactPlan(
        phx.geometry.Circle((0.0, 0.0), 0.5).compile(),
        phx.discretization.SharpCoulombMPMFrictionPlan(0.25),
        contact_band=0.02,
    )
    contact_result = contact.apply(
        jnp.asarray([[0.49, 0.0]]),
        jnp.asarray([[-1.0, 0.6]]),
        jnp.asarray((2.0,)),
        0.0,
        0.01,
    )
    fracture = phx.discretization.MPMFieldPartitionFracturePlan(2)
    topology = fracture.update(
        jnp.asarray((0.1, 0.99)),
        jnp.asarray((-1.0, 1.0)),
        jnp.zeros((2,), dtype=jnp.int32),
        0,
    )
    material = phx.applications.solid_mechanics.PhaseFieldNeoHookeanMPMConstitutivePlan(2)
    parameters = phx.applications.solid_mechanics.MPMPhaseFieldParameters(
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(2.0, 8.0),
        1.0,
        0.1,
    )
    response = material.evaluate(
        jnp.asarray([[[1.08, 0.0], [0.0, 1.0]]]),
        jnp.asarray([[0.5, 0.0]]),
        jnp.asarray((1.0,)),
        parameters,
        0.0,
        0.01,
    )
    return {
        "contact": {
            "successful": bool(contact_result.successful),
            "post_velocity": contact_result.velocity[0].tolist(),
            "dissipation": float(contact_result.dissipation),
        },
        "sharp_topology": {
            "successful": bool(topology.successful),
            "generation": int(topology.topology_generation),
            "slots": topology.velocity_field_slots.tolist(),
        },
        "diffuse_material": {
            "successful": bool(response.successful[0]),
            "history": float(response.trial_state[0, 1]),
            "energy": float(response.reference_energy_density[0]),
        },
    }


if __name__ == "__main__":
    print(run())
