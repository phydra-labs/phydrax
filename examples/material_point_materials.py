#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Evaluate plane-stress and finite-strain J2 MPM material contracts."""

import jax.numpy as jnp

import phydrax as phx


def run():
    neo_parameters = (
        phx.applications.solid_mechanics.NeoHookeanParameters.from_shear_bulk(3.0, 11.0)
    )
    plane = phx.applications.solid_mechanics.PlaneStressMPMConstitutivePlan(
        phx.applications.solid_mechanics.NeoHookeanMPMConstitutivePlan(3)
    )
    plane_history = plane.initialize_state((1,), jnp.float64)
    plane_response = plane.evaluate(
        jnp.asarray([[[1.12, 0.06], [0.02, 0.93]]]),
        plane_history,
        jnp.asarray((2.0,)),
        neo_parameters,
        0.0,
        0.01,
    )
    plastic = phx.applications.solid_mechanics.FiniteStrainJ2MPMConstitutivePlan()
    plastic_parameters = phx.applications.solid_mechanics.FiniteStrainJ2Parameters(
        10.0, 30.0, 0.15, 2.0
    )
    plastic_response = plastic.evaluate(
        jnp.asarray([[[1.0, 0.18, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]),
        plastic.initialize_state((1,), jnp.float64),
        jnp.asarray((1.0,)),
        plastic_parameters,
        0.0,
        0.01,
    )
    return {
        "plane_stress": {
            "successful": bool(plane_response.successful[0]),
            "P33_residual": float(plane_response.diagnostics["plane_stress_residual"][0]),
            "lambda3": float(plane_response.diagnostics["out_of_plane_stretch"][0]),
        },
        "plasticity": {
            "successful": bool(plastic_response.successful[0]),
            "branch": int(plastic_response.branch_code[0]),
            "plastic_multiplier": float(
                plastic_response.diagnostics["plastic_multiplier"][0]
            ),
            "dissipation": float(plastic_response.dissipation_increment[0]),
        },
    }


if __name__ == "__main__":
    print(run())
