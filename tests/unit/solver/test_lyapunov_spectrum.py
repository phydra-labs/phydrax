#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

from phydrax.solver import (
    DifferentialProblem,
    kaplan_yorke_dimension,
    LYAPUNOV_INSUFFICIENT_ACCUMULATION,
    LYAPUNOV_SINGULAR_TANGENT,
    lyapunov_spectrum_flow,
    lyapunov_spectrum_map,
    LYAPUNOV_SUCCESS,
)


def _linear_map(matrix):
    return lambda state, args: matrix @ state


def test_linear_map_full_leading_and_qr_cadence_agree():
    expected = jnp.asarray([0.2, -0.1, -0.5])
    matrix = jnp.diag(jnp.exp(expected))
    initial = jnp.asarray([0.7, -0.3, 0.2])

    every_step = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=80,
        qr_interval=1,
        accumulation_interval=10,
        system_id="diagonal-map",
    )
    sparse_qr = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=80,
        qr_interval=8,
        accumulation_interval=16,
        system_id="diagonal-map",
    )
    leading = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=80,
        leading_k=2,
        initial_basis=jnp.eye(3, 2),
        qr_interval=5,
        accumulation_interval=10,
        system_id="diagonal-map",
    )
    explicit_action = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=80,
        qr_interval=4,
        accumulation_interval=8,
        tangent_action=lambda state, vector, args: matrix @ vector,
        system_id="diagonal-map-action",
    )

    np.testing.assert_allclose(every_step.exponents, expected, atol=2e-13)
    np.testing.assert_allclose(sparse_qr.exponents, expected, atol=2e-13)
    np.testing.assert_allclose(leading.exponents, every_step.exponents[:2], atol=2e-13)
    np.testing.assert_allclose(explicit_action.exponents, expected, atol=2e-13)
    assert bool(every_step.valid)
    assert bool(every_step.kaplan_yorke_valid)
    assert not bool(leading.kaplan_yorke_valid)
    assert every_step.method_id == "periodic_qr"
    assert every_step.tangent_method == "jax_jvp"
    assert explicit_action.tangent_method == "user_jvp"
    assert every_step.finite_time_exponents.shape == (8, 3)
    assert bool(jnp.isfinite(every_step.convergence_drift))
    assert leading.approximation == "leading_k_finite_time_spectrum"


def test_truncated_default_basis_reaches_reversed_diagonal_leading_direction():
    matrix = jnp.diag(jnp.exp(jnp.asarray([-1.0, 1.0])))
    result = lyapunov_spectrum_map(
        _linear_map(matrix),
        jnp.asarray([0.4, -0.2]),
        num_steps=100,
        leading_k=1,
        qr_interval=5,
        accumulation_interval=20,
        system_id="reversed-diagonal-map",
    )

    np.testing.assert_allclose(result.exponents, jnp.asarray([1.0]), atol=1e-2)
    assert float(result.exponents[0]) > 0.0


def test_qr_cadence_invariance_for_a_rotated_tangent_basis():
    rotation = jnp.asarray([[0.8, -0.6], [0.6, 0.8]])
    matrix = rotation @ jnp.diag(jnp.exp(jnp.asarray([0.2, -0.3]))) @ rotation.T
    initial = jnp.asarray([0.2, 0.7])
    every_step = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=90,
        qr_interval=1,
        accumulation_interval=9,
        system_id="rotated-cadence-1",
    )
    every_ninth = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=90,
        qr_interval=9,
        accumulation_interval=18,
        system_id="rotated-cadence-9",
    )
    np.testing.assert_allclose(every_ninth.exponents, every_step.exponents, atol=3e-14)


def test_map_checkpoint_resume_matches_uninterrupted_accumulation():
    matrix = jnp.asarray([[1.03, 0.2], [-0.1, 0.94]])
    initial = jnp.asarray([0.4, -0.2])
    kwargs = dict(
        qr_interval=7,
        accumulation_interval=14,
        system_id="resume-map",
    )
    uninterrupted = lyapunov_spectrum_map(
        _linear_map(matrix), initial, num_steps=100, **kwargs
    )
    first = lyapunov_spectrum_map(_linear_map(matrix), initial, num_steps=40, **kwargs)
    resumed = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=60,
        checkpoint=first.checkpoint,
        **kwargs,
    )

    np.testing.assert_allclose(resumed.exponents, uninterrupted.exponents, atol=2e-13)
    np.testing.assert_allclose(resumed.final_state, uninterrupted.final_state, atol=2e-13)
    np.testing.assert_allclose(
        resumed.checkpoint.log_stretch,
        uninterrupted.checkpoint.log_stretch,
        atol=2e-13,
    )
    assert resumed.checkpoint.step_index == uninterrupted.checkpoint.step_index == 100


def test_pre_burn_checkpoint_remains_numerically_resumable():
    matrix = jnp.asarray([[1.08, 0.15], [-0.03, 0.92]])
    initial = jnp.asarray([0.4, -0.2])
    kwargs = dict(
        leading_k=1,
        qr_interval=5,
        burn_in=13,
        accumulation_interval=10,
        system_id="pre-burn-resume-map",
    )
    uninterrupted = lyapunov_spectrum_map(
        _linear_map(matrix), initial, num_steps=43, **kwargs
    )
    first = lyapunov_spectrum_map(_linear_map(matrix), initial, num_steps=7, **kwargs)
    resumed = lyapunov_spectrum_map(
        _linear_map(matrix),
        initial,
        num_steps=36,
        checkpoint=first.checkpoint,
        **kwargs,
    )

    assert not bool(first.valid)
    assert int(first.status) == LYAPUNOV_INSUFFICIENT_ACCUMULATION
    assert bool(first.checkpoint.valid)
    assert int(first.checkpoint.status) == LYAPUNOV_SUCCESS
    assert bool(resumed.valid)
    np.testing.assert_allclose(resumed.exponents, uninterrupted.exponents, atol=2e-12)
    np.testing.assert_allclose(
        resumed.finite_time_exponents,
        uninterrupted.finite_time_exponents,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        resumed.accumulation_times,
        uninterrupted.accumulation_times,
        atol=2e-12,
    )
    np.testing.assert_allclose(resumed.final_state, uninterrupted.final_state, atol=2e-12)
    np.testing.assert_allclose(
        resumed.checkpoint.log_stretch,
        uninterrupted.checkpoint.log_stretch,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        resumed.checkpoint.basis,
        uninterrupted.checkpoint.basis,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        resumed.checkpoint.accumulated_time,
        uninterrupted.checkpoint.accumulated_time,
        atol=2e-12,
    )
    assert resumed.checkpoint.step_index == uninterrupted.checkpoint.step_index
    assert bool(resumed.checkpoint.valid) == bool(uninterrupted.checkpoint.valid)
    assert int(resumed.checkpoint.status) == int(uninterrupted.checkpoint.status)
    assert (
        resumed.checkpoint.accumulated_intervals
        == uninterrupted.checkpoint.accumulated_intervals
    )


def test_linear_flow_and_flow_resume_match_analytic_spectrum():
    matrix = jnp.diag(jnp.asarray([0.35, -0.2, -0.8]))

    def drift(time, state, args):
        return matrix @ state

    whole_problem = DifferentialProblem(
        drift,
        jnp.asarray([0.6, -0.4, 0.2]),
        t0=0.0,
        t1=2.0,
    )
    whole = lyapunov_spectrum_flow(
        whole_problem,
        step_size=0.005,
        qr_interval=0.1,
        accumulation_interval=0.2,
        system_id="linear-flow",
    )
    explicit_action = lyapunov_spectrum_flow(
        whole_problem,
        step_size=0.005,
        qr_interval=0.1,
        accumulation_interval=0.2,
        tangent_action=lambda time, state, vector, args: matrix @ vector,
        system_id="linear-flow-action",
    )
    first_problem = DifferentialProblem(
        drift,
        whole_problem.initial_state,
        t0=0.0,
        t1=0.8,
    )
    first = lyapunov_spectrum_flow(
        first_problem,
        step_size=0.005,
        qr_interval=0.1,
        accumulation_interval=0.2,
        system_id="linear-flow",
    )
    second_problem = DifferentialProblem(
        drift,
        first.final_state,
        t0=0.8,
        t1=2.0,
    )
    resumed = lyapunov_spectrum_flow(
        second_problem,
        step_size=0.005,
        qr_interval=0.1,
        accumulation_interval=0.2,
        checkpoint=first.checkpoint,
        system_id="linear-flow",
    )

    np.testing.assert_allclose(whole.exponents, jnp.diag(matrix), atol=2e-9)
    np.testing.assert_allclose(explicit_action.exponents, jnp.diag(matrix), atol=2e-9)
    np.testing.assert_allclose(resumed.exponents, whole.exponents, atol=3e-13)
    np.testing.assert_allclose(resumed.final_state, whole.final_state, atol=3e-13)
    assert resumed.backend == "phydrax.solver"
    assert resumed.discretization_id == "fixed_rk4_dt=0.0050000000000000001"
    assert explicit_action.tangent_method == "user_jvp"


def test_post_burn_qr_and_reports_share_the_burn_relative_cadence():
    matrix = jnp.diag(jnp.asarray([1.03, 0.97]))
    map_result = lyapunov_spectrum_map(
        _linear_map(matrix),
        jnp.asarray([0.4, -0.2]),
        num_steps=10,
        qr_interval=4,
        burn_in=2,
        accumulation_interval=4,
        system_id="burn-relative-map",
    )

    generator = jnp.diag(jnp.asarray([0.3, -0.2]))

    def drift(time, state, args):
        return generator @ state

    flow_result = lyapunov_spectrum_flow(
        DifferentialProblem(
            drift,
            jnp.asarray([0.4, -0.2]),
            t0=0.0,
            t1=1.0,
        ),
        step_size=0.1,
        qr_interval=0.4,
        burn_in=0.2,
        accumulation_interval=0.4,
        system_id="burn-relative-flow",
    )

    np.testing.assert_allclose(map_result.accumulation_times, jnp.asarray([4.0, 8.0]))
    np.testing.assert_allclose(
        flow_result.accumulation_times,
        jnp.asarray([0.4, 0.8]),
        atol=1e-14,
    )


def test_lorenz_spectrum_has_literature_range_and_divergence_sum():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def lorenz(time, state, args):
        x, y, z = state
        return jnp.asarray([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    problem = DifferentialProblem(
        lorenz,
        jnp.asarray([1.0, 1.0, 1.0]),
        t0=0.0,
        t1=25.0,
    )
    result = lyapunov_spectrum_flow(
        problem,
        step_size=0.01,
        qr_interval=0.1,
        burn_in=5.0,
        accumulation_interval=1.0,
        system_id="lorenz-63",
    )

    assert bool(result.valid)
    assert 0.3 < float(result.exponents[0]) < 1.4
    assert -0.5 < float(result.exponents[1]) < 0.5
    assert -17.0 < float(result.exponents[2]) < -11.0
    np.testing.assert_allclose(
        jnp.sum(result.exponents), -(sigma + 1.0 + beta), atol=0.35
    )


def test_kaplan_yorke_dimension_known_spectrum():
    spectrum = jnp.asarray([0.9, 0.0, -14.4])
    np.testing.assert_allclose(
        kaplan_yorke_dimension(spectrum), 2.0 + 0.9 / 14.4, atol=1e-14
    )


def test_singular_tangent_is_recorded_as_invalid_without_repair():
    result = lyapunov_spectrum_map(
        _linear_map(jnp.zeros((2, 2))),
        jnp.ones(2),
        num_steps=2,
        system_id="singular-map",
    )
    assert not bool(result.valid)
    assert int(result.status) == LYAPUNOV_SINGULAR_TANGENT
    assert bool(jnp.all(jnp.isneginf(result.exponents)))
