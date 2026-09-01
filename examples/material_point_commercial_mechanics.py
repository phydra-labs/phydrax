#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""General plane stress, non-associated plasticity, and K-way contact."""

import jax.numpy as jnp

import phydrax as phx


def run():
    plane = phx.applications.solid_mechanics.GeneralPlaneStressMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    )
    neo = phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(3.0, 11.0)
    deformation = jnp.asarray([[[1.1, 0.04], [0.01, 0.95]]])
    plane_response = plane.evaluate(
        deformation,
        plane.initialize_state((1,), jnp.float64),
        jnp.asarray((2.0,)),
        neo,
        0.0,
        0.01,
    )
    drucker_prager = phx.applications.solid_mechanics.DruckerPragerMPMConstitutivePlan()
    dp_response = drucker_prager.evaluate(
        jnp.asarray([[[1.0, 0.15, 0.0], [0.0, 0.94, 0.0], [0.0, 0.0, 1.06]]]),
        drucker_prager.initialize_state((1,), jnp.float64),
        jnp.asarray((1.0,)),
        phx.applications.solid_mechanics.DruckerPragerParameters(
            10.0, 30.0, 0.05, 0.5, 0.2, 1.0
        ),
        0.0,
        0.01,
    )
    mass = jnp.asarray([[1.0], [1.5], [2.0]])
    velocity = jnp.asarray([[[0.8, 0.3]], [[0.0, 0.0]], [[-0.6, -0.1]]])
    gradient = jnp.asarray([[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, -1.0]]])
    contact = phx.discretization.KWayMPMContactPlan(3, maximum_steps=40, tolerance=1e-8)
    contact_result = contact.solve(
        mass, velocity, contact.build_graph(mass, gradient), 0.01
    )
    return {
        "plane_stress": {
            "successful": bool(plane_response.successful[0]),
            "traction_residual": float(
                jnp.linalg.norm(plane_response.diagnostics["plane_stress_residual"][0])
            ),
        },
        "drucker_prager": {
            "successful": bool(dp_response.successful[0]),
            "yield_residual": float(dp_response.diagnostics["yield_residual"][0]),
            "dissipation": float(dp_response.dissipation_increment[0]),
        },
        "contact": {
            "successful": bool(contact_result.successful),
            "complementarity": float(contact_result.complementarity_residual),
            "action_reaction": float(contact_result.action_reaction_defect),
        },
    }


if __name__ == "__main__":
    print(run())
