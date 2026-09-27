import jax.random as jr

from tools.stochastic_convergence import (
    run_commutative_noise_benchmark,
    run_multilevel_monte_carlo_benchmark,
    run_multiplicative_reaction_diffusion_benchmark,
    run_rough_logode_convergence_benchmark,
    run_stochastic_advection_diffusion_benchmark,
    run_stochastic_heat_convergence_benchmark,
)


def test_stochastic_convergence_benchmarks_scenario_1() -> None:
    result = run_stochastic_heat_convergence_benchmark(
        jr.key(5),
        temporal_paths=128,
        moment_paths=1024,
    )

    assert result.passed
    assert result.temporal.sampling_is_subordinate()
    assert result.temporal.mean_square_stable()
    assert result.noise_truncation.levels[-1].strong_rms_error == 0.0
    result = run_stochastic_advection_diffusion_benchmark(jr.key(6))

    assert result.passed
    assert result.derivative_noise_variance_error < 0.08
    assert result.stratonovich_correction_error < 1e-10
    result = run_multiplicative_reaction_diffusion_benchmark(jr.key(7))

    assert result.passed
    assert result.temporal.regression_rate("strong") > 0.35
    assert result.temporal.regression_rate(observable="second_moment") > 0.5
    result = run_commutative_noise_benchmark()

    assert result.passed
    assert result.commutative_flow_order_error < result.noncommutative_flow_order_error


def test_stochastic_convergence_benchmarks_scenario_2() -> None:
    result = run_rough_logode_convergence_benchmark(jr.key(904), fine_steps=16)

    assert result.passed
    assert result.interval_counts == (2, 4, 8)
    assert result.terminal_errors[-1] < result.terminal_errors[0]
    assert result.general_linear_relative_error < 2e-6
    assert result.accepted_logode_steps > 0
    result = run_multilevel_monte_carlo_benchmark(
        jr.key(8),
        target_rmse=0.08,
        initial_samples=32,
    )

    assert result.passed
    assert result.estimate.diagnostics.sample_counts[0] > 32
    assert (
        result.estimate.diagnostics.correction_variance_norms[-1]
        < result.estimate.diagnostics.correction_variance_norms[0]
    )
