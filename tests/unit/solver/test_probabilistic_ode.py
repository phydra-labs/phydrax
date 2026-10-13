#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe
import pytest

import phydrax as phx
from phydrax.solver._probabilistic_ode import _block_filter


def _scaled_state(time: Any, state: Any, parameter: Any) -> Any:
    return parameter * state


def _linear_state(time: Any, state: Any, args: Any) -> Any:
    return args @ state


def _diagonal_state(time: Any, state: Any, args: Any) -> Any:
    return args * state


def _zero_drift(time: Any, state: Any, args: Any) -> Any:
    return jnp.zeros_like(state)


def _parameter_drift(time: Any, state: Any, parameter: Any) -> Any:
    return jnp.broadcast_to(parameter, state.shape)


def _negated_state(time: Any, state: Any, args: Any) -> Any:
    return -state


def _exponential_problem(
    rate: Any = 0.7, *, t0: Any = 0.0, t1: Any = 1.0, initial: Any = None
) -> Any:
    if initial is None:
        initial = jnp.asarray([1.0])
    return phx.solver.DifferentialProblem(
        _scaled_state,
        initial,
        t0=t0,
        t1=t1,
        args=rate,
    )


def _terminal_error(order: Any, update: Any, steps: Any) -> Any:
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
def test_integrated_wiener_methods_converge_on_smooth_scalar_ode(update: Any) -> None:
    coarse = _terminal_error(1, update, 8)
    fine = _terminal_error(1, update, 16)

    assert fine < coarse / 3.0
    assert fine < 2e-3


def test_higher_order_prior_converges_on_coupled_vector_ode() -> None:
    matrix = jnp.asarray([[0.0, 1.0], [-1.0, 0.0]])
    problem = phx.solver.DifferentialProblem(
        _linear_state,
        jnp.asarray([1.0, 0.0]),
        t0=0.0,
        t1=1.0,
        args=matrix,
    )

    def error(steps: Any) -> Any:
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


def test_work_precision_is_sane_against_canonical_diffrax_solution() -> None:
    problem = phx.solver.DifferentialProblem(
        _scaled_state,
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


def test_quasi_mle_calibration_tracks_residual_scale_and_preserves_sources() -> None:
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


def test_stiffness_and_step_exhaustion_have_explicit_status_codes() -> None:
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


def test_dense_and_block_diagonal_factors_agree_for_separable_vector_system() -> None:
    rates = jnp.asarray([0.4, -0.2, 0.1])
    problem = phx.solver.DifferentialProblem(
        _diagonal_state,
        jnp.asarray([1.0, 2.0, -1.0]),
        t0=0.0,
        t1=1.0,
        args=rates,
    )
    common: dict[str, Any] = dict(
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

    assert dense.covariances is not None
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
        oe.contract("tij,tj->ti", block.dense_covariance(), probe),
    )


def test_residual_adaptation_redistributes_fixed_work_and_reaches_endpoint() -> None:
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


def test_adaptive_endpoint_roundoff_does_not_report_step_limit() -> None:
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


def test_checkpoint_resume_replays_fixed_steps_deterministically() -> None:
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
    incompatible_problem = _exponential_problem(
        rate=0.9,
        t0=0.5,
        t1=1.0,
        initial=first.means[-1],
    )
    with pytest.raises(ValueError, match="does not belong"):
        phx.solver.solve_probabilistic_ode(
            incompatible_problem,
            save_times=jnp.asarray([1.0]),
            method=method,
            step_size=0.05,
            checkpoint=first.checkpoint,
        )
    assert jnp.array_equal(full.diffusion_scale, resumed.diffusion_scale)


def test_checkpoint_binds_declared_identity_of_opaque_drift() -> None:
    method = phx.solver.ProbabilisticODEMethod(num_steps=4)
    decay = lambda time, state, rate: -rate * state
    growth = lambda time, state, rate: rate * state

    def problem(drift: Any, *, t0: Any, t1: Any, initial: Any) -> Any:
        return phx.solver.DifferentialProblem(
            drift, initial, t0=t0, t1=t1, args=jnp.asarray(0.5)
        )

    with pytest.raises(TypeError, match="explicit semantic_id and numeric_id"):
        phx.solver.solve_probabilistic_ode(
            problem(decay, t0=0.0, t1=0.5, initial=jnp.asarray([1.0])),
            save_times=jnp.asarray([0.5]),
            method=method,
            step_size=0.125,
        )

    first = phx.solver.solve_probabilistic_ode(
        problem(decay, t0=0.0, t1=0.5, initial=jnp.asarray([1.0])),
        save_times=jnp.asarray([0.5]),
        method=method,
        step_size=0.125,
        drift_semantic_id="linear-rate-law",
        drift_numeric_id="decay",
    )
    resumed = problem(decay, t0=0.5, t1=1.0, initial=first.means[-1])
    phx.solver.solve_probabilistic_ode(
        resumed,
        save_times=jnp.asarray([1.0]),
        method=method,
        step_size=0.125,
        checkpoint=first.checkpoint,
        drift_semantic_id="linear-rate-law",
        drift_numeric_id="decay",
    )
    with pytest.raises(ValueError, match="does not belong"):
        phx.solver.solve_probabilistic_ode(
            problem(growth, t0=0.5, t1=1.0, initial=first.means[-1]),
            save_times=jnp.asarray([1.0]),
            method=method,
            step_size=0.125,
            checkpoint=first.checkpoint,
            drift_semantic_id="linear-rate-law",
            drift_numeric_id="growth",
        )


def test_solver_is_jittable_and_differentiable_in_model_parameters() -> None:
    method = phx.solver.ProbabilisticODEMethod(num_steps=12)

    def terminal(rate: Any) -> Any:
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
    factorization: Any, smoothing: Any, unit_variance: Any
) -> None:
    problem = phx.solver.DifferentialProblem(
        _zero_drift,
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
def test_parameter_covariance_is_one_fixed_random_parameter(factorization: Any) -> None:
    parameter_variance = 0.09
    times = jnp.asarray([0.5, 2.0])
    expected = parameter_variance * times**2
    problem = phx.solver.DifferentialProblem(
        _parameter_drift,
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
def test_generated_method_id_rejects_resume_critical_mismatch(
    changed_method: Any,
) -> None:
    first_problem = phx.solver.DifferentialProblem(
        _zero_drift,
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


def test_dense_matrix_free_output_retains_no_covariance_matrix() -> None:
    problem = phx.solver.DifferentialProblem(
        _negated_state,
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
        oe.contract("tij,tj->ti", solution.dense_covariance(), probe),
    )


@pytest.mark.parametrize(("update", "evaluations_per_step"), [("ek0", 1), ("ek1", 2)])
def test_adaptive_work_count_includes_pilot_and_production(
    update: Any, evaluations_per_step: Any
) -> None:
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


def test_block_dense_output_checks_guard_before_materialization() -> None:
    problem = phx.solver.DifferentialProblem(
        _zero_drift,
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
    factorization: Any, observation_variance: Any
) -> None:
    problem = phx.solver.DifferentialProblem(
        _zero_drift,
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


@pytest.mark.parametrize("parameter_size", [0, 2])
def test_block_filter_directional_drift_matches_numpy(parameter_size: int) -> None:
    rng = np.random.default_rng(718)
    n, h = 5, 0.07
    matrix = rng.normal(scale=0.2, size=(n, n))
    forcing = rng.normal(size=(n, parameter_size))
    parameters = rng.normal(size=parameter_size)
    mean = rng.normal(size=(2, n))
    sensitivity = rng.normal(size=(2, n, parameter_size))
    roots = rng.normal(scale=0.1, size=(5, n, 2, 2))
    sources = roots @ np.swapaxes(roots, -1, -2)
    observation = np.linspace(0.02, 0.04, n)
    process = np.linspace(0.01, 0.03, n)
    method = phx.solver.ProbabilisticODEMethod(
        order=1,
        update="ek1",
        factorization="block_diagonal",
        smoothing=False,
        diffusion_calibration="none",
    )

    def drift(time: jax.Array, state: jax.Array, args: jax.Array) -> jax.Array:
        del time
        return jnp.asarray(matrix) @ jnp.sin(state) + jnp.asarray(forcing) @ args

    problem = phx.solver.DifferentialProblem(
        drift, jnp.asarray(mean[0]), t0=0.0, t1=h, args=jnp.asarray(parameters)
    )
    transition = np.asarray([[1.0, h], [0.0, 1.0]])
    iwp = np.asarray([[h**3 / 3.0, h**2 / 2.0], [h**2 / 2.0, h]])
    predicted = transition @ mean
    predicted_sensitivity = np.einsum("ij,jdp->idp", transition, sensitivity)
    predicted_sources = np.einsum("ij,sdjk,lk->sdil", transition, sources, transition)
    predicted_sources[0] += method.base_diffusion * iwp
    predicted_sources[1] += process[:, None, None] * iwp
    jacobian = matrix * np.cos(predicted[0])[None, :]
    residual = predicted[1] - matrix @ np.sin(predicted[0]) - forcing @ parameters
    rows = np.column_stack((-np.diag(jacobian), np.ones(n)))
    covariance = predicted_sources.sum(axis=0)
    variance = (
        np.einsum("di,dij,dj->d", rows, covariance, rows)
        + observation
        + method.covariance_regularization
    )
    gain = np.einsum("dij,dj->di", covariance, rows) / variance[:, None]
    expected_mean = predicted - (gain * residual[:, None]).T
    residual_sensitivity = (
        predicted_sensitivity[1] - jacobian @ predicted_sensitivity[0] - forcing
    )
    expected_sensitivity = predicted_sensitivity - np.einsum(
        "dq,dp->qdp", gain, residual_sensitivity
    )
    joseph = np.eye(2)[None] - np.einsum("di,dj->dij", gain, rows)
    expected_sources = np.einsum("dij,sdjk,dlk->sdil", joseph, predicted_sources, joseph)
    outer = np.einsum("di,dj->dij", gain, gain)
    expected_sources[2] += observation[:, None, None] * outer
    expected_sources[0] += method.covariance_regularization * outer
    expected_sources = 0.5 * (expected_sources + np.swapaxes(expected_sources, -1, -2))

    def run(initial: jax.Array) -> jax.Array:
        result, _ = _block_filter(
            problem,
            method,
            initial,
            jnp.asarray(sources),
            jnp.asarray(sensitivity),
            jnp.asarray([h]),
            jnp.asarray(0.0),
            jnp.diag(jnp.asarray(observation)),
            jnp.diag(jnp.asarray(process)),
            jnp.asarray(parameters),
            lambda x: x,
            jnp.asarray(0.0),
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(0.0),
        )
        return result[0]

    result, _ = _block_filter(
        problem,
        method,
        jnp.asarray(mean),
        jnp.asarray(sources),
        jnp.asarray(sensitivity),
        jnp.asarray([h]),
        jnp.asarray(0.0),
        jnp.diag(jnp.asarray(observation)),
        jnp.diag(jnp.asarray(process)),
        jnp.asarray(parameters),
        lambda x: x,
        jnp.asarray(0.0),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0.0),
    )
    np.testing.assert_allclose(result[0], expected_mean, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result[1], expected_sources, atol=2e-12, rtol=2e-12)
    np.testing.assert_allclose(result[2], expected_sensitivity, atol=2e-12, rtol=2e-12)

    def dense_mean(initial: jax.Array) -> jax.Array:
        prediction = jnp.asarray(transition) @ initial
        drift_at_state = lambda x: drift(jnp.asarray(h), x, jnp.asarray(parameters))
        jac = jax.jacfwd(drift_at_state)(prediction[0])
        observation_rows = jnp.stack((-jnp.diag(jac), jnp.ones(n)), axis=1)
        innovation = (
            jnp.einsum(
                "di,dij,dj->d",
                observation_rows,
                jnp.asarray(covariance),
                observation_rows,
            )
            + jnp.asarray(observation)
            + method.covariance_regularization
        )
        kalman_gain = (
            jnp.einsum("dij,dj->di", jnp.asarray(covariance), observation_rows)
            / innovation[:, None]
        )
        return (
            prediction
            - (kalman_gain * (prediction[1] - drift_at_state(prediction[0]))[:, None]).T
        )

    np.testing.assert_allclose(
        jax.jacfwd(run)(jnp.asarray(mean)),
        jax.jacfwd(dense_mean)(jnp.asarray(mean)),
        atol=2e-11,
        rtol=2e-12,
    )
