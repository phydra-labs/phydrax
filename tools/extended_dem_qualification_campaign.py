#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Nondistributed cohesive-contact and superquadric DEM qualification campaign."""

import json

import jax.numpy as jnp

import phydrax as phx


def _material():
    return phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
        rolling_friction=jnp.asarray([[0.1]]),
    )


def _cohesive_case(center_distance):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    cohesion = phx.discretization.CompositeDEMCohesionPlan(
        (
            phx.discretization.DMTContactCohesionPlan(0.05, 0.1),
            phx.discretization.LinearCapillaryBridgePlan(0.07, 0.0, 1.0e-9, 0.1),
            phx.discretization.NearContactLubricationPlan(1.0e-3, 0.1, 1.0e-5),
        )
    )
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "cohesive-qualification", _material(), gravity=jnp.zeros((3,))
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(
            jnp.full((2,), 0.5), jnp.zeros((2,), dtype=jnp.int32)
        ),
        phx.discretization.SoftSphereDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.HertzNormalContactPlan(),
                cohesion=cohesion,
                rotational=phx.discretization.ElasticRollingTorsionalResistancePlan(
                    100.0, 50.0, torsional_friction=0.05
                ),
            )
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [center_distance, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 1.0, 0.0], [-1.0, -1.0, 0.0]]),
    )
    evaluation = compiled.dynamics.evaluate(
        jnp.asarray(0.0), state, jnp.asarray(1.0e-4), None
    )
    response = evaluation.particle_contact
    passed = (
        evaluation.successful
        & jnp.all(jnp.isfinite(response.pair_force))
        & (jnp.max(jnp.abs(response.bridge_volume_residual)) <= 1.0e-14)
        & jnp.all(response.rotational_dissipated_work >= 0.0)
    )
    return {
        "center_distance": center_distance,
        "normal_force": float(response.normal_force[0, 0]),
        "bridge_volume_residual": float(
            jnp.max(jnp.abs(response.bridge_volume_residual))
        ),
        "rotational_dissipation": float(jnp.sum(response.rotational_dissipated_work)),
        "passed": bool(passed),
    }


def _superquadric_case(semi_axes, epsilon1, epsilon2):
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=3
    ).prepare()
    axes = jnp.asarray([semi_axes, semi_axes])
    shapes = phx.discretization.SuperquadricSetPlan(
        axes,
        jnp.full((2,), epsilon1),
        jnp.full((2,), epsilon2),
        jnp.zeros((2,), dtype=jnp.int32),
    )
    dynamics = phx.discretization.SuperquadricDEMPlan(
        shapes,
        phx.discretization.SuperquadricContactPlan(iterations=24),
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
    ).prepare(particles, _material(), phx.discretization.DenseParticleNeighborhoodPlan(1))
    distance = 2.0 * semi_axes[0] - 0.02
    state = dynamics.initialize_state(
        jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
        jnp.asarray([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    evaluation = dynamics.evaluate(state, jnp.asarray(1.0e-5))
    passed = (
        evaluation.successful
        & jnp.all(evaluation.geometry.valid)
        & (jnp.max(evaluation.geometry.residual) <= 1.0e-8)
        & jnp.all(evaluation.geometry.effective_radius > 0.0)
    )
    return {
        "semi_axes": list(semi_axes),
        "epsilon1": epsilon1,
        "epsilon2": epsilon2,
        "gap": float(evaluation.geometry.gap[0]),
        "contact_residual": float(evaluation.geometry.residual[0]),
        "passed": bool(passed),
    }


def main():
    cohesive_cases = [_cohesive_case(distance) for distance in (0.95, 1.0, 1.05)]
    superquadric_cases = [
        _superquadric_case(axes, epsilon1, epsilon2)
        for axes, epsilon1, epsilon2 in (
            ((0.5, 0.5, 0.5), 2.0, 2.0),
            ((0.6, 0.4, 0.3), 2.0, 2.0),
            ((0.6, 0.4, 0.3), 2.5, 3.0),
        )
    ]
    cases = cohesive_cases + superquadric_cases
    print(
        json.dumps(
            {
                "campaign": "extended-dem-contact-and-shape-qualification",
                "passed": all(case["passed"] for case in cases),
                "cohesive_cases": cohesive_cases,
                "superquadric_cases": superquadric_cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
