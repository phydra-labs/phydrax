#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def test_error_norms_and_observed_order_use_physical_cell_measure():
    numerical = jnp.asarray([[1.1], [1.8], [3.2]])
    exact = jnp.asarray([[1.0], [2.0], [3.0]])
    volumes = jnp.asarray([0.2, 0.5, 0.3])
    norms = phx.equations.finite_volume_error_norms(
        numerical, exact, volumes
    )
    expected_l1 = 0.2 * 0.1 + 0.5 * 0.2 + 0.3 * 0.2
    np.testing.assert_allclose(norms.l1, expected_l1)
    np.testing.assert_allclose(norms.linf, 0.2)

    convergence = phx.equations.finite_volume_convergence_result(
        (16, 32, 64), jnp.asarray([4e-3, 1e-3, 2.5e-4]), 2.0
    )
    np.testing.assert_allclose(convergence.observed_orders, [2.0, 2.0])
    assert convergence.passed


def test_periodic_advection_case_exact_solution_advects_without_shape_loss():
    case = phx.equations.periodic_advection_verification_case(0.7)
    points = ((jnp.arange(16.0) + 0.5) / 16.0)[:, None]
    initial = case.initial_state(points, 0.0, None)
    exact_period = case.exact_state(points, 1.0 / 0.7, None)

    np.testing.assert_allclose(exact_period, initial, atol=2e-12)
    assert initial.shape == (16, 1)


def test_sod_case_and_viscous_reference_profiles_are_physical():
    case = phx.equations.sod_verification_case()
    points = ((jnp.arange(20.0) + 0.5) / 20.0)[:, None]
    state = case.initial_state(points, 0.0, None)
    primitive = case.system.conserved_to_primitive(state)
    assert jnp.all(case.system.admissible(state))
    np.testing.assert_allclose(primitive[0], [1.0, 0.0, 1.0])
    np.testing.assert_allclose(primitive[-1], [0.125, 0.0, 0.1])

    y = jnp.linspace(0.0, 1.0, 11)
    couette = phx.equations.couette_velocity_profile(y, 0.0, 2.0)
    poiseuille = phx.equations.poiseuille_velocity_profile(y, -2.0, 1.0)
    np.testing.assert_allclose(couette, 2.0 * y)
    np.testing.assert_allclose(
        poiseuille[jnp.asarray([0, -1])], 0.0, atol=0.0
    )


def test_severe_euler_reference_initial_states_are_admissible():
    points = ((jnp.arange(64.0) + 0.5) / 64.0)[:, None]
    for case in (
        phx.equations.lax_verification_case(),
        phx.equations.double_rarefaction_verification_case(),
        phx.equations.woodward_colella_verification_case(),
    ):
        state = case.initial_state(points, 0.0, None)
        assert state.shape == (64, 3)
        assert jnp.all(case.system.admissible(state))
        assert jnp.isfinite(case.final_time)
