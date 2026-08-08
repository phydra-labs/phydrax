#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


def _exponential_problem(rate=0.7, *, t0=0.0, t1=1.0, initial=None):
    if initial is None:
        initial = jnp.asarray([1.0])
    return phx.solver.DifferentialProblem(
        lambda time, state, parameter: parameter * state,
        initial,
        t0=t0,
        t1=t1,
        args=rate,
    )


def _terminal_error(order, update, steps):
    solution = phx.solver.solve_probabilistic_ode(
        _exponential_problem(),
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            order=order,
            update=update,
            num_steps=steps,
            smoothing=False,
            diffusion_calibration="none",
        ),
    )
    return jnp.abs(solution.means[0, 0] - jnp.exp(0.7))


@pytest.mark.parametrize("update", ["ek0", "ek1"])
def test_integrated_wiener_methods_converge_on_smooth_scalar_ode(update):
    coarse = _terminal_error(1, update, 8)
    fine = _terminal_error(1, update, 16)

    assert fine < coarse / 3.0
    assert fine < 2e-3


def test_higher_order_prior_converges_on_coupled_vector_ode():
    matrix = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: args @ state,
        jnp.asarray([1.0, 0.0]),
        t0=0.0,
        t1=1.0,
        args=matrix,
    )

    def error(steps):
        solution = phx.solver.solve_probabilistic_ode(
            problem,
            save_times=jnp.asarray([1.0]),
            method=phx.solver.ProbabilisticODEMethod(
                order=2,
                update="ek1",
                num_steps=steps,
                smoothing=False,
                diffusion_calibration="none",
            ),
        )
        expected = jnp.asarray([jnp.cos(1.0), -jnp.sin(1.0)])
        return jnp.linalg.norm(solution.means[0] - expected)

    assert error(16) < error(8) / 3.0
    assert error(16) < 2e-4


def test_work_precision_is_sane_against_canonical_diffrax_solution():
    problem = phx.solver.DifferentialProblem(
        lambda time, state, parameter: parameter * state,
        jnp.asarray([1.0]),
        t0=0.0,
        t1=2.0,
        args=-0.4,
    )
    times = jnp.linspace(0.0, 2.0, 17)
    probabilistic = phx.solver.solve_probabilistic_ode(
        problem,
        save_times=times,
        method=phx.solver.ProbabilisticODEMethod(num_steps=32),
    )
    canonical = phx.solver.solve_diffrax(problem, save_times=times)

    assert jnp.max(jnp.abs(probabilistic.means - canonical.states)) < 1e-3
    assert probabilistic.stats["num_steps"] == 32
    assert probabilistic.stats["num_drift_evaluations"] <= 64


def test_quasi_mle_calibration_tracks_residual_scale_and_preserves_sources():
    method = phx.solver.ProbabilisticODEMethod(
        num_steps=12,
        update="ek0",
        diffusion_calibration="quasi_mle",
    )
    smooth = phx.solver.solve_probabilistic_ode(
        _exponential_problem(rate=0.2),
        save_times=jnp.asarray([1.0]),
        method=method,
        initial_covariance=1e-4,
        process_covariance=2e-4,
        observation_covariance=3e-4,
        parameter_covariance=4e-4,
    )
    faster = phx.solver.solve_probabilistic_ode(
        _exponential_problem(rate=2.0),
        save_times=jnp.asarray([1.0]),
        method=method,
        initial_covariance=1e-4,
        process_covariance=2e-4,
        observation_covariance=3e-4,
        parameter_covariance=4e-4,
    )

    assert faster.diffusion_scale > smooth.diffusion_scale
    assert smooth.uncertainty_sources == (
        "numerical",
        "process",
        "observation",
        "initial_condition",
        "parameter",
    )
    for source in smooth.uncertainty_sources:
        assert source in smooth.source_covariances
        assert jnp.all(jnp.isfinite(smooth.source_covariances[source]))


def test_stiffness_and_step_exhaustion_have_explicit_status_codes():
    stiff = phx.solver.solve_probabilistic_ode(
        _exponential_problem(rate=-200.0),
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=8,
            stiffness_threshold=1.0,
        ),
    )
    exhausted = phx.solver.solve_probabilistic_ode(
        _exponential_problem(),
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(num_steps=2),
        step_size=0.1,
    )

    assert stiff.status in (
        phx.solver.PROBABILISTIC_ODE_STIFF,
        phx.solver.PROBABILISTIC_ODE_NONFINITE,
    )
    assert not stiff.successful
    assert exhausted.status == phx.solver.PROBABILISTIC_ODE_STEP_LIMIT_REACHED
    assert not exhausted.successful


def test_dense_and_block_diagonal_factors_agree_for_separable_vector_system():
    rates = jnp.asarray([0.4, -0.2, 0.1])
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: args * state,
        jnp.asarray([1.0, 2.0, -1.0]),
        t0=0.0,
        t1=1.0,
        args=rates,
    )
    common = dict(
        save_times=jnp.asarray([0.0, 0.4, 1.0]),
        observation_covariance=1e-7,
    )
    dense = phx.solver.solve_probabilistic_ode(
        problem,
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=24,
            factorization="dense",
            covariance_output="dense",
        ),
        **common,
    )
    block = phx.solver.solve_probabilistic_ode(
        problem,
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=24,
            factorization="block_diagonal",
            covariance_output="matrix_free",
        ),
        **common,
    )

    assert isinstance(dense.covariance_factor, phx.uq.GaussianFactor)
    assert isinstance(block.covariance_factor, phx.uq.GaussianFactor)
    assert jnp.allclose(dense.means, block.means, rtol=2e-6, atol=2e-7)
    assert jnp.allclose(
        jnp.diagonal(dense.covariances, axis1=-2, axis2=-1),
        jnp.diagonal(block.dense_covariance(), axis1=-2, axis2=-1),
        rtol=2e-5,
        atol=2e-9,
    )
    probe = jnp.arange(9.0).reshape((3, 3))
    assert jnp.allclose(
        block.covariance_matvec(probe),
        jnp.einsum("tij,tj->ti", block.dense_covariance(), probe),
    )


def test_residual_adaptation_redistributes_fixed_work_and_reaches_endpoint():
    solution = phx.solver.solve_probabilistic_ode(
        _exponential_problem(rate=2.0),
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=12,
            adaptive=True,
        ),
    )

    assert solution.status == phx.solver.PROBABILISTIC_ODE_SUCCESS
    assert solution.stats["pilot_used"]
    assert jnp.ptp(solution.step_sizes) > 0.0
    assert jnp.sum(solution.step_sizes) == 1.0


def test_adaptive_endpoint_roundoff_does_not_report_step_limit():
    solution = phx.solver.solve_probabilistic_ode(
        _exponential_problem(rate=-0.4, t1=2.0),
        save_times=jnp.asarray([2.0]),
        method=phx.solver.ProbabilisticODEMethod(
            order=2,
            update="ek1",
            num_steps=64,
            adaptive=True,
            diffusion_calibration="quasi_mle",
        ),
        initial_covariance=jnp.asarray([[1e-4]]),
        process_covariance=jnp.asarray([[2e-5]]),
        observation_covariance=jnp.asarray([[1e-6]]),
        parameter_covariance=jnp.asarray([[1e-4]]),
    )

    assert solution.status == phx.solver.PROBABILISTIC_ODE_SUCCESS
    assert solution.checkpoint.time == 2.0


def test_checkpoint_resume_replays_fixed_steps_deterministically():
    method = phx.solver.ProbabilisticODEMethod(num_steps=24)
    full = phx.solver.solve_probabilistic_ode(
        _exponential_problem(t1=1.0),
        save_times=jnp.asarray([1.0]),
        method=method,
        step_size=0.05,
    )
    first = phx.solver.solve_probabilistic_ode(
        _exponential_problem(t1=0.5),
        save_times=jnp.asarray([0.5]),
        method=method,
        step_size=0.05,
    )
    resumed_problem = _exponential_problem(
        t0=0.5,
        t1=1.0,
        initial=first.means[-1],
    )
    resumed = phx.solver.solve_probabilistic_ode(
        resumed_problem,
        save_times=jnp.asarray([1.0]),
        method=method,
        step_size=0.05,
        checkpoint=first.checkpoint,
    )

    assert jnp.array_equal(full.means, resumed.means)
    assert jnp.array_equal(full.standard_deviations, resumed.standard_deviations)
    assert jnp.array_equal(full.diffusion_scale, resumed.diffusion_scale)


def test_solver_is_jittable_and_differentiable_in_model_parameters():
    method = phx.solver.ProbabilisticODEMethod(num_steps=12)

    def terminal(rate):
        return phx.solver.solve_probabilistic_ode(
            _exponential_problem(rate),
            save_times=jnp.asarray([1.0]),
            method=method,
        ).means[0, 0]

    compiled = eqx.filter_jit(terminal)(jnp.asarray(0.7))
    derivative = jax.grad(terminal)(jnp.asarray(0.7))

    assert jnp.allclose(compiled, jnp.exp(0.7), rtol=2e-4, atol=2e-5)
    assert jnp.allclose(derivative, jnp.exp(0.7), rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("factorization", ["dense", "block_diagonal"])
@pytest.mark.parametrize(
    ("smoothing", "unit_variance"),
    [(False, 1.0 / 24.0), (True, 5.0 / 192.0)],
)
def test_midpoint_iwp_covariance_uses_filter_or_conditional_bridge(
    factorization, smoothing, unit_variance
):
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
    )
    solution = phx.solver.solve_probabilistic_ode(
        problem,
        save_times=jnp.asarray([0.5]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=1,
            smoothing=smoothing,
            factorization=factorization,
            diffusion_calibration="none",
            covariance_output="matrix_free",
        ),
        process_covariance=2.0,
    )

    numerical = solution.dense_covariance(source="numerical")[0, 0, 0]
    process = solution.dense_covariance(source="process")[0, 0, 0]
    components = sum(
        (
            solution.dense_covariance(source=source)
            for source in solution.uncertainty_sources
        ),
        jnp.zeros((1, 1, 1)),
    )

    assert jnp.allclose(numerical, unit_variance, rtol=2e-6, atol=2e-8)
    assert jnp.allclose(process, 2.0 * unit_variance, rtol=2e-6, atol=2e-8)
    assert jnp.allclose(solution.dense_covariance(), components)


@pytest.mark.parametrize("factorization", ["dense", "block_diagonal"])
def test_parameter_covariance_is_one_fixed_random_parameter(factorization):
    parameter_variance = 0.09
    times = jnp.asarray([0.5, 2.0])
    expected = parameter_variance * times**2
    problem = phx.solver.DifferentialProblem(
        lambda time, state, parameter: jnp.broadcast_to(parameter, state.shape),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=2.0,
        args=jnp.asarray(1.25),
    )

    for num_steps in (1, 4, 16):
        solution = phx.solver.solve_probabilistic_ode(
            problem,
            save_times=times,
            method=phx.solver.ProbabilisticODEMethod(
                num_steps=num_steps,
                factorization=factorization,
                diffusion_calibration="none",
                covariance_output="matrix_free",
            ),
            parameter_covariance=parameter_variance,
        )
        parameter = solution.dense_covariance(source="parameter")[:, 0, 0]

        assert jnp.allclose(parameter, expected, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize(
    "changed_method",
    [
        phx.solver.ProbabilisticODEMethod(
            num_steps=2,
            base_diffusion=0.75,
            covariance_regularization=0.1,
        ),
        phx.solver.ProbabilisticODEMethod(
            num_steps=2,
            base_diffusion=0.5,
            covariance_regularization=0.2,
        ),
    ],
)
def test_generated_method_id_rejects_resume_critical_mismatch(changed_method):
    first_problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=0.5,
    )
    original_method = phx.solver.ProbabilisticODEMethod(
        num_steps=2,
        base_diffusion=0.5,
        covariance_regularization=0.1,
    )
    first = phx.solver.solve_probabilistic_ode(
        first_problem,
        save_times=jnp.asarray([0.5]),
        method=original_method,
        step_size=0.25,
    )
    resumed_problem = phx.solver.DifferentialProblem(
        first_problem.drift,
        first.means[-1],
        t0=0.5,
        t1=1.0,
    )

    with pytest.raises(ValueError, match="method IDs"):
        phx.solver.solve_probabilistic_ode(
            resumed_problem,
            save_times=jnp.asarray([1.0]),
            method=changed_method,
            step_size=0.25,
            checkpoint=first.checkpoint,
        )


def test_dense_matrix_free_output_retains_no_covariance_matrix():
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: -state,
        jnp.asarray([1.0, -2.0]),
        t0=0.0,
        t1=1.0,
    )
    solution = phx.solver.solve_probabilistic_ode(
        problem,
        save_times=jnp.asarray([0.5, 1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=4,
            factorization="dense",
            covariance_output="matrix_free",
            diffusion_calibration="none",
        ),
    )
    probe = jnp.asarray([[1.0, -1.0], [0.5, 2.0]])

    assert solution.covariances is None
    assert solution.covariance_representation == "matrix_free"
    assert jnp.allclose(
        solution.covariance_matvec(probe),
        jnp.einsum("tij,tj->ti", solution.dense_covariance(), probe),
    )


@pytest.mark.parametrize(("update", "evaluations_per_step"), [("ek0", 1), ("ek1", 2)])
def test_adaptive_work_count_includes_pilot_and_production(update, evaluations_per_step):
    num_steps = 5
    solution = phx.solver.solve_probabilistic_ode(
        _exponential_problem(),
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=num_steps,
            update=update,
            adaptive=True,
        ),
    )

    assert solution.stats["num_drift_evaluations"] == (
        2 * evaluations_per_step * num_steps
    )


def test_block_dense_output_checks_guard_before_materialization():
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.zeros((3,)),
        t0=0.0,
        t1=1.0,
    )

    with pytest.raises(ValueError, match="max_dense_dimension"):
        phx.solver.solve_probabilistic_ode(
            problem,
            save_times=jnp.asarray([1.0]),
            method=phx.solver.ProbabilisticODEMethod(
                num_steps=1,
                factorization="block_diagonal",
                covariance_output="dense",
                max_dense_dimension=2,
            ),
        )


@pytest.mark.parametrize("factorization", ["dense", "block_diagonal"])
@pytest.mark.parametrize("observation_variance", [0.0, 3.0])
def test_quasi_likelihood_includes_innovation_normalization(
    factorization, observation_variance
):
    problem = phx.solver.DifferentialProblem(
        lambda time, state, args: jnp.zeros_like(state),
        jnp.asarray([0.0]),
        t0=0.0,
        t1=1.0,
    )
    solution = phx.solver.solve_probabilistic_ode(
        problem,
        save_times=jnp.asarray([1.0]),
        method=phx.solver.ProbabilisticODEMethod(
            num_steps=1,
            smoothing=False,
            factorization=factorization,
            diffusion_calibration="none",
        ),
        observation_covariance=observation_variance,
    )
    expected = -0.5 * (jnp.log(1.0 + observation_variance) + jnp.log(2.0 * jnp.pi))

    assert jnp.allclose(solution.log_quasi_likelihood, expected)
    assert solution.stats["normalized_chi_square"] == 0.0
