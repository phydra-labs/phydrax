#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from scipy.special import beta

import phydrax as phx
from phydrax.discretization import DEMRuntimeState


def _hertz_problem() -> Any:
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]),
        jnp.asarray([1.0, 1.0]),
        ambient_dimension=3,
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    materials = phx.equations.DEMMaterialTable(
        jnp.asarray([2.0e6]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.9]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.CompliantDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.HertzNormalContactPlan(),
            tangential=phx.discretization.MindlinTangentialContactPlan(),
        ),
        maximum_overlap_fraction=0.25,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "hertz-contact", materials, gravity=jnp.zeros((3,))
    )
    return phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )


def test_hertz_force_matches_negative_distance_derivative_of_elastic_energy() -> None:
    compiled = _hertz_problem()

    def energy(distance: Any) -> Any:
        state = compiled.initialize_state(
            0.0,
            jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
            jnp.zeros((2, 3)),
        )
        return compiled.diagnostics(0.0, state).elastic_energy

    distance = jnp.asarray(0.9)
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        jnp.zeros((2, 3)),
    )
    force_on_right = state.loads.total.force[1, 0]
    energy_gradient = jax.grad(energy)(distance)

    assert force_on_right > 0.0
    assert jnp.allclose(force_on_right, -energy_gradient, rtol=2.0e-6)


def test_mindlin_tangential_force_respects_friction_cone() -> None:
    compiled = _hertz_problem()
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]]),
        jnp.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    detail = compiled.dynamics.step_detailed(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
        state,
        jnp.asarray(1.0e-5),
        None,
    )
    response = detail.evaluation.particle_contact

    assert detail.successful
    assert jnp.linalg.norm(response.tangential_force[0]) <= (
        0.4 * jnp.linalg.norm(response.normal_force[0]) + 1.0e-9
    )


def test_elastic_hertz_impact_restitution_and_duration_match_analytic_reference() -> None:
    """Energy-integral oracle: U=2kδ^(5/2)/5, reduced mass=1/2.

    Contact duration is (4/5)(δ_max/v0) B(2/5,1/2); the 800-step
    contact resolution bounds sampling error below the stated 0.5% tolerance.
    Hertz elastic dynamics: Sun et al., Results in Physics 30 (2021) 104870.
    """
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1], dtype=jnp.int64),
        jnp.ones((2,), dtype=jnp.float64),
        ambient_dimension=3,
    ).prepare()
    young = 2.0e6
    poisson = 0.25
    effective_modulus = young / (2.0 * (1.0 - poisson**2))
    stiffness = (4.0 / 3.0) * effective_modulus * np.sqrt(0.25)
    impact_speed = 1.0
    maximum_overlap = (5.0 * 0.5 * impact_speed**2 / (4.0 * stiffness)) ** 0.4
    analytic_duration = (
        (4.0 / 5.0)
        * maximum_overlap
        / impact_speed
        * beta(np.float64(0.4), np.float64(0.5))
    )
    step_size = jnp.asarray(analytic_duration / 800.0, dtype=jnp.float64)
    compiled = phx.equations.compile_discrete_element_problem(
        phx.equations.DiscreteElementProblemIR(
            "elastic-hertz-impact",
            phx.equations.DEMMaterialTable(
                jnp.asarray([young]),
                jnp.asarray([poisson]),
                jnp.asarray([[1.0]]),
                jnp.asarray([[0.0]]),
            ),
            gravity=jnp.zeros((3,)),
        ),
        particles,
        phx.discretization.RigidSphereSetPlan(
            jnp.full((2,), 0.5),
            jnp.zeros((2,), dtype=jnp.int32),
        ),
        phx.discretization.CompliantDEMMethodPlan(
            phx.discretization.DEMContactModelPlan(
                phx.discretization.HertzNormalContactPlan()
            ),
        ),
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    initial = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        jnp.asarray([[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]]),
    )

    def advance(
        state: DEMRuntimeState, index: Array
    ) -> tuple[DEMRuntimeState, tuple[Array, Array]]:
        detail = compiled.dynamics.step_detailed(
            index,
            index.astype(jnp.float64) * step_size,
            state,
            step_size,
            None,
        )
        return detail.accepted_state, (
            detail.successful,
            detail.evaluation.geometry.geometry.gap[0],
        )

    final, (successful, gaps) = jax.jit(
        lambda state: jax.lax.scan(advance, state, jnp.arange(1000, dtype=jnp.int32))
    )(initial)
    assert jnp.all(successful)
    restitution = final.kinematics.velocity[1, 0] - final.kinematics.velocity[0, 0]
    measured_duration = jnp.sum(gaps < 0.0) * step_size
    np.testing.assert_allclose(restitution / impact_speed, 1.0, rtol=1.0e-3)
    np.testing.assert_allclose(measured_duration, analytic_duration, rtol=5.0e-3)
