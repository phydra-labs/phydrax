#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

import phydrax as phx


def _diagnostics():
    particles = phx.discretization.ParticleSetPlan(
        jnp.asarray([0, 1]), jnp.ones((2,)), ambient_dimension=2
    ).prepare()
    spheres = phx.discretization.RigidSphereSetPlan(
        jnp.asarray([0.5, 0.5]), jnp.asarray([0, 0])
    )
    material = phx.equations.DEMMaterialTable(
        jnp.asarray([1.0e5]),
        jnp.asarray([0.25]),
        jnp.asarray([[0.8]]),
        jnp.asarray([[0.4]]),
    )
    method = phx.discretization.SoftSphereDEMMethodPlan(
        phx.discretization.DEMContactModelPlan(
            phx.discretization.LinearSpringDashpotNormalPlan(1.0e4)
        ),
        maximum_overlap_fraction=0.3,
    )
    problem = phx.equations.DiscreteElementProblemIR(
        "sensitivity", material, gravity=jnp.zeros((2,))
    )
    compiled = phx.equations.compile_discrete_element_problem(
        problem,
        particles,
        spheres,
        method,
        neighborhood=phx.discretization.DenseParticleNeighborhoodPlan(1),
    )
    state = compiled.initialize_state(
        0.0,
        jnp.asarray([[0.0, 0.0], [0.9, 0.0]]),
        jnp.zeros((2, 2)),
    )
    return compiled.diagnostics(0.0, state)


def test_batched_inverse_and_parameter_ensemble_require_valid_certificates():
    diagnostics = _diagnostics()
    policy = phx.discretization.DEMSensitivityPolicy(
        activation_margin=1.0e-12,
        no_tension_margin=1.0e-12,
        friction_margin=1.0e-12,
        frame_margin=1.0e-12,
        acceptance_margin=1.0e-12,
        neighborhood_margin=1.0e-12,
    )

    def forward(parameter, case):
        return parameter * case, diagnostics

    cases = jnp.asarray([1.0, 2.0, 3.0])
    observations = jnp.asarray([2.0, 4.0, 6.0])
    problem = phx.discretization.DEMInverseProblem(
        forward,
        observations,
        jnp.ones_like(observations, dtype=bool),
        policy,
        problem_id="linear-identification",
    )
    result = phx.discretization.evaluate_dem_inverse(problem, jnp.asarray(1.5), cases)
    assert result.usable
    assert result.gradient < 0.0
    assert result.qualification.rank == 1

    ensemble = phx.discretization.evaluate_dem_parameter_ensemble(
        problem, jnp.asarray([1.5, 2.0, 2.5]), cases
    )
    assert jnp.isclose(ensemble.successful_fraction, 1.0)
    assert ensemble.predictions.shape == (3, 3)


def test_hybrid_event_localization_and_saltation_are_transverse():
    plan = phx.solver.HybridEventPlan(
        lambda time, state, args: state[0],
        lambda time, state, args: -state,
        lambda time, state, args: jnp.asarray([1.0]),
        lambda time, state, args: jnp.asarray([-1.0]),
        event_kind=phx.discretization.DEMHybridEventKind.CONTACT_ONSET.value,
        plan_id="one-dimensional-impact",
    )
    result = phx.solver.localize_hybrid_event(
        plan,
        lambda time, args: jnp.asarray([time - 0.5]),
        jnp.asarray(0.0),
        jnp.asarray(1.0),
    )
    assert result.successful
    assert jnp.isclose(result.event_time, 0.5, atol=1.0e-10)
    assert not result.grazing
    assert jnp.all(jnp.isfinite(result.saltation_matrix))
