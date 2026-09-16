#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.applications.numerical_relativity import (
    AdSBoundaryScalarObservablePlan,
    AdSConformalBoundaryPlan,
    conformal_scalar_normal_mode,
    conformal_scalar_stress_energy,
    ConformalAdSScalarPlan,
    ConformalAdSScalarState,
    ConformalEinsteinState,
    ConformalEinsteinSystem,
    evaluate_ads_conformal_boundary,
    evaluate_conformal_einstein_zero_quantities,
    evaluate_generalized_wave_gauge,
    evaluate_holographic_stress_tensor,
    exact_ads_conformal_reference,
    extract_ads_boundary_scalar,
    GeneralizedWaveGaugePlan,
    HolographicStressTensorPlan,
    run_conformal_ads_scalar,
)


def test_exact_ads_satisfies_every_metric_conformal_zero_quantity():
    system = ConformalEinsteinSystem(
        -3.0,
        scalar_curvature_gauge=-12.0,
        residual_tolerance=1e-12,
    )
    state, derivatives = exact_ads_conformal_reference(system, (2, 2, 2))
    evidence = evaluate_conformal_einstein_zero_quantities(system, state, derivatives)
    assert bool(evidence.accepted)
    np.testing.assert_allclose(evidence.maximum_residual, 0.0, atol=1e-14)
    perturbed = ConformalEinsteinState(
        state.metric,
        1.01 * state.conformal_factor,
        state.friedrich_scalar,
        state.schouten,
        state.rescaled_weyl,
        state_id="perturbed-omega",
    )
    failed = evaluate_conformal_einstein_zero_quantities(system, perturbed, derivatives)
    assert not bool(failed.accepted)
    assert float(jnp.max(jnp.abs(failed.scalar_constraint))) > 0.0


def test_generalized_wave_gauge_transition_retains_source_and_curvature_residual():
    plan = GeneralizedWaveGaugePlan(
        (0.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        transition_time=1.0,
        transition_time_width=0.2,
        transition_radius=0.5,
        transition_radius_width=0.1,
        damping=0.1,
        scalar_curvature_gauge=-12.0,
    )
    radius = jnp.linspace(0.0, 1.0, 8)
    source = plan.source(2.0, radius)
    evidence = evaluate_generalized_wave_gauge(
        plan,
        -source,
        -12.0 * jnp.ones_like(radius),
        2.0,
        radius,
        tolerance=1e-12,
    )
    assert bool(evidence.accepted)
    np.testing.assert_allclose(evidence.maximum_wave_constraint, 0.0)


def test_timelike_boundary_checks_omega_normal_metric_radiation_and_corner():
    shape = (5, 5, 5)
    metric = jnp.broadcast_to(
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0, 1.0))).reshape(4, 4, 1, 1, 1),
        (4, 4) + shape,
    )
    x = jnp.linspace(0.0, 1.0, shape[0])[:, None, None]
    omega = jnp.broadcast_to(1.0 - x, shape)
    gradient = jnp.zeros((4,) + shape).at[1].set(-1.0)
    state = ConformalEinsteinState(
        metric,
        omega,
        jnp.zeros(shape),
        jnp.zeros((4, 4) + shape),
        jnp.zeros((4, 4, 4, 4) + shape),
        state_id="timelike-boundary-control",
    )
    plan = AdSConformalBoundaryPlan(
        0,
        "upper",
        jnp.diag(jnp.asarray((-1.0, 1.0, 1.0))),
        radiation_policy="reflecting",
        minimum_normal_gradient=0.9,
        minimum_normal_norm=0.9,
    )
    evidence = evaluate_ads_conformal_boundary(
        plan,
        state,
        gradient,
        jnp.zeros((5, 5)),
        conformal_rate=jnp.zeros(shape),
    )
    assert bool(evidence.accepted)
    np.testing.assert_allclose(evidence.conformal_factor_residual, 0.0)
    np.testing.assert_allclose(evidence.minimum_normal_norm, 1.0)


def test_conformal_scalar_normal_mode_and_reflecting_runtime():
    plan = ConformalAdSScalarPlan(
        65,
        time_step=0.002,
        potential=0.0,
        maximum_steps=200,
    )
    initial, frequency = conformal_scalar_normal_mode(plan, 1)
    run = run_conformal_ads_scalar(
        plan,
        initial,
        steps=100,
        energy_tolerance=2e-3,
    )
    expected = initial.field * jnp.cos(frequency * run.final_state.time)
    np.testing.assert_allclose(run.final_state.field, expected, rtol=2e-3, atol=2e-3)
    assert bool(run.evidence.accepted)
    np.testing.assert_allclose(run.evidence.maximum_boundary_residual, 0.0)


def test_conformal_scalar_stress_and_holographic_observables_are_audited():
    metric = jnp.diag(jnp.asarray((-1.0, 1.0, 1.0, 1.0)))
    stress = conformal_scalar_stress_energy(
        metric,
        metric,
        1.0,
        jnp.zeros((4,)),
        jnp.zeros((4, 4)),
        jnp.zeros((4, 4)),
    )
    assert bool(stress.tracefree)
    np.testing.assert_allclose(stress.stress_energy, 0.0)

    scalar_plan = ConformalAdSScalarPlan(33, time_step=0.005)
    defining = jnp.cos(scalar_plan.radial_points)
    field = 2.0 * defining + 3.0 * defining**2
    field = field.at[-1].set(0.0)
    scalar_state = ConformalAdSScalarState(scalar_plan, field, jnp.zeros_like(field))
    scalar_observable = extract_ads_boundary_scalar(
        AdSBoundaryScalarObservablePlan(
            1.0,
            2.0,
            fit_points=8,
            residual_tolerance=1e-10,
        ),
        scalar_plan,
        scalar_state,
    )
    np.testing.assert_allclose(scalar_observable.source_coefficient, 2.0, rtol=1e-10)
    np.testing.assert_allclose(scalar_observable.response_coefficient, 3.0, rtol=1e-10)
    assert bool(scalar_observable.accepted)

    boundary_metric = jnp.diag(jnp.asarray((-1.0, 1.0, 1.0)))
    coefficient = jnp.diag(jnp.asarray((2.0, 1.0, 1.0)))
    holographic = evaluate_holographic_stress_tensor(
        HolographicStressTensorPlan(3, 1.0, "declared-fg-control"),
        coefficient,
        jnp.zeros_like(coefficient),
        boundary_metric,
        jnp.zeros((3,)),
    )
    assert bool(holographic.accepted)
    np.testing.assert_allclose(holographic.trace, 0.0)
