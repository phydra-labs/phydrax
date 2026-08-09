#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import numpy as np

import phydrax as phx


def _map_evolution(matrix, *, system_id):
    state_shape = (int(matrix.shape[0]),)
    system = phx.dynamics.DiscreteSystem(
        lambda coordinate, state, args: matrix @ state,
        state_layout=phx.dynamics.StateLayout(state_shape),
        system_id=system_id,
    )
    return phx.dynamics.DiscreteEvolution(system)


def _iteration_grid(start, num_steps, *, grid_id):
    return phx.dynamics.IterationGrid(
        jnp.arange(start, start + num_steps + 1), iteration_id=grid_id
    )


def _flow_evolution(drift, *, state_dimension, system_id):
    system = phx.dynamics.ContinuousSystem(
        drift,
        state_layout=phx.dynamics.StateLayout((state_dimension,)),
        system_id=system_id,
    )
    return phx.solver.DiffraxEvolution(system, rtol=1.0e-9, atol=1.0e-11)


def _time_grid(start, end, num_steps, *, grid_id):
    return phx.dynamics.TimeGrid(jnp.linspace(start, end, num_steps + 1), time_id=grid_id)


def test_linear_map_full_leading_and_qr_cadence_agree():
    expected = jnp.asarray([0.2, -0.1, -0.5])
    matrix = jnp.diag(jnp.exp(expected))
    initial = jnp.asarray([0.7, -0.3, 0.2])
    evolution = _map_evolution(matrix, system_id="diagonal-map")
    grid = _iteration_grid(0, 80, grid_id="diagonal-map-grid")

    every_step = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        grid,
        qr_interval=1,
        accumulation_interval=10,
    )
    sparse_qr = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        grid,
        qr_interval=8,
        accumulation_interval=16,
    )
    leading = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        grid,
        leading_k=2,
        initial_basis=jnp.eye(3, 2),
        qr_interval=5,
        accumulation_interval=10,
    )

    np.testing.assert_allclose(every_step.exponents, expected, atol=2.0e-13)
    np.testing.assert_allclose(sparse_qr.exponents, expected, atol=2.0e-13)
    np.testing.assert_allclose(leading.exponents, every_step.exponents[:2], atol=2.0e-13)
    assert bool(every_step.valid)
    assert bool(every_step.kaplan_yorke_valid)
    assert not bool(leading.kaplan_yorke_valid)
    assert every_step.method_id == "periodic_qr"
    assert every_step.tangent_method == "jax-jvp:declared-transition"
    assert every_step.finite_time_exponents.shape == (8, 3)
    assert bool(jnp.isfinite(every_step.convergence_drift))
    assert leading.approximation == "leading_k_finite_time_spectrum"


def test_truncated_default_basis_reaches_reversed_diagonal_leading_direction():
    matrix = jnp.diag(jnp.exp(jnp.asarray([-1.0, 1.0])))
    result = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        _map_evolution(matrix, system_id="reversed-diagonal-map"),
        jnp.asarray([0.4, -0.2]),
        _iteration_grid(0, 100, grid_id="reversed-diagonal-grid"),
        leading_k=1,
        qr_interval=5,
        accumulation_interval=20,
    )

    np.testing.assert_allclose(result.exponents, jnp.asarray([1.0]), atol=1.0e-2)
    assert float(result.exponents[0]) > 0.0


def test_qr_cadence_invariance_for_a_rotated_tangent_basis():
    rotation = jnp.asarray([[0.8, -0.6], [0.6, 0.8]])
    matrix = rotation @ jnp.diag(jnp.exp(jnp.asarray([0.2, -0.3]))) @ rotation.T
    evolution = _map_evolution(matrix, system_id="rotated-map")
    grid = _iteration_grid(0, 90, grid_id="rotated-grid")
    every_step = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        jnp.asarray([0.2, 0.7]),
        grid,
        qr_interval=1,
        accumulation_interval=9,
    )
    every_ninth = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        jnp.asarray([0.2, 0.7]),
        grid,
        qr_interval=9,
        accumulation_interval=18,
    )

    np.testing.assert_allclose(every_ninth.exponents, every_step.exponents, atol=3.0e-14)


def test_map_checkpoint_resume_matches_uninterrupted_accumulation():
    matrix = jnp.asarray([[1.03, 0.2], [-0.1, 0.94]])
    initial = jnp.asarray([0.4, -0.2])
    evolution = _map_evolution(matrix, system_id="resume-map")
    kwargs = dict(qr_interval=7, accumulation_interval=14)
    uninterrupted = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _iteration_grid(0, 100, grid_id="resume-grid"),
        **kwargs,
    )
    first = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _iteration_grid(0, 40, grid_id="resume-grid"),
        **kwargs,
    )
    resumed = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        None,
        _iteration_grid(40, 60, grid_id="resume-grid"),
        checkpoint=first.checkpoint,
        **kwargs,
    )

    np.testing.assert_allclose(resumed.exponents, uninterrupted.exponents, atol=2.0e-13)
    np.testing.assert_allclose(
        resumed.final_state, uninterrupted.final_state, atol=2.0e-13
    )
    np.testing.assert_allclose(
        resumed.checkpoint.log_stretch,
        uninterrupted.checkpoint.log_stretch,
        atol=2.0e-13,
    )
    assert resumed.checkpoint.step_index == uninterrupted.checkpoint.step_index == 100


def test_pre_burn_checkpoint_remains_numerically_resumable():
    matrix = jnp.asarray([[1.08, 0.15], [-0.03, 0.92]])
    initial = jnp.asarray([0.4, -0.2])
    evolution = _map_evolution(matrix, system_id="pre-burn-resume-map")
    kwargs = dict(leading_k=1, qr_interval=5, burn_in=13, accumulation_interval=10)
    uninterrupted = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _iteration_grid(0, 43, grid_id="pre-burn-grid"),
        **kwargs,
    )
    first = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _iteration_grid(0, 7, grid_id="pre-burn-grid"),
        **kwargs,
    )
    resumed = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        None,
        _iteration_grid(7, 36, grid_id="pre-burn-grid"),
        checkpoint=first.checkpoint,
        **kwargs,
    )

    assert not bool(first.valid)
    assert int(first.status) == phx.dynamics.analysis.LYAPUNOV_INSUFFICIENT_ACCUMULATION
    assert bool(first.checkpoint.valid)
    assert int(first.checkpoint.status) == phx.dynamics.analysis.LYAPUNOV_SUCCESS
    assert bool(resumed.valid)
    np.testing.assert_allclose(resumed.exponents, uninterrupted.exponents, atol=2.0e-12)
    np.testing.assert_allclose(
        resumed.finite_time_exponents,
        uninterrupted.finite_time_exponents,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        resumed.accumulation_times,
        uninterrupted.accumulation_times,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        resumed.final_state, uninterrupted.final_state, atol=2.0e-12
    )
    np.testing.assert_allclose(
        resumed.checkpoint.log_stretch,
        uninterrupted.checkpoint.log_stretch,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        resumed.checkpoint.basis,
        uninterrupted.checkpoint.basis,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        resumed.checkpoint.accumulated_time,
        uninterrupted.checkpoint.accumulated_time,
        atol=2.0e-12,
    )
    assert resumed.checkpoint.step_index == uninterrupted.checkpoint.step_index
    assert (
        resumed.checkpoint.accumulated_intervals
        == uninterrupted.checkpoint.accumulated_intervals
    )


def test_linear_flow_and_resume_match_analytic_spectrum():
    matrix = jnp.diag(jnp.asarray([0.35, -0.2, -0.8]))

    def drift(time, state, args):
        return matrix @ state

    evolution = _flow_evolution(drift, state_dimension=3, system_id="linear-flow")
    initial = jnp.asarray([0.6, -0.4, 0.2])
    kwargs = dict(qr_interval=1, accumulation_interval=2)
    whole = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _time_grid(0.0, 2.0, 20, grid_id="linear-flow-grid"),
        **kwargs,
    )
    first = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        initial,
        _time_grid(0.0, 0.8, 8, grid_id="linear-flow-grid"),
        **kwargs,
    )
    resumed = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        evolution,
        None,
        _time_grid(0.8, 2.0, 12, grid_id="linear-flow-grid"),
        checkpoint=first.checkpoint,
        **kwargs,
    )

    np.testing.assert_allclose(whole.exponents, jnp.diag(matrix), atol=2.0e-8)
    np.testing.assert_allclose(resumed.exponents, whole.exponents, atol=2.0e-8)
    np.testing.assert_allclose(resumed.final_state, whole.final_state, atol=2.0e-8)
    assert resumed.backend == "backend:diffrax"
    assert resumed.tangent_method == "jax-jvp:numerical-differential-flow"


def test_post_burn_reports_use_physical_elapsed_time():
    matrix = jnp.diag(jnp.asarray([1.03, 0.97]))
    map_result = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        _map_evolution(matrix, system_id="burn-relative-map"),
        jnp.asarray([0.4, -0.2]),
        _iteration_grid(0, 10, grid_id="burn-relative-map-grid"),
        qr_interval=4,
        burn_in=2,
        accumulation_interval=4,
    )

    generator = jnp.diag(jnp.asarray([0.3, -0.2]))
    flow_result = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        _flow_evolution(
            lambda time, state, args: generator @ state,
            state_dimension=2,
            system_id="burn-relative-flow",
        ),
        jnp.asarray([0.4, -0.2]),
        _time_grid(0.0, 1.0, 10, grid_id="burn-relative-flow-grid"),
        qr_interval=4,
        burn_in=2,
        accumulation_interval=4,
    )

    np.testing.assert_allclose(map_result.accumulation_times, jnp.asarray([4.0, 8.0]))
    np.testing.assert_allclose(
        flow_result.accumulation_times, jnp.asarray([0.4, 0.8]), atol=1.0e-14
    )


def test_lorenz_spectrum_has_literature_range_and_divergence_sum():
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    def lorenz(time, state, args):
        x, y, z = state
        return jnp.asarray([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    result = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        _flow_evolution(lorenz, state_dimension=3, system_id="lorenz-63"),
        jnp.asarray([1.0, 1.0, 1.0]),
        _time_grid(0.0, 25.0, 250, grid_id="lorenz-grid"),
        qr_interval=1,
        burn_in=50,
        accumulation_interval=10,
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
        phx.dynamics.analysis.kaplan_yorke_dimension(spectrum),
        2.0 + 0.9 / 14.4,
        atol=1.0e-14,
    )


def test_singular_tangent_is_recorded_as_invalid_without_repair():
    result = phx.dynamics.analysis.finite_time_lyapunov_spectrum(
        _map_evolution(jnp.zeros((2, 2)), system_id="singular-map"),
        jnp.ones(2),
        _iteration_grid(0, 2, grid_id="singular-grid"),
    )

    assert not bool(result.valid)
    assert int(result.status) == phx.dynamics.analysis.LYAPUNOV_SINGULAR_TANGENT
    assert bool(jnp.all(jnp.isneginf(result.exponents)))
