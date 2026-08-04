import jax.random as jr

from tools.stochastic_convergence import (
    run_commutative_noise_benchmark,
    run_multiplicative_reaction_diffusion_benchmark,
    run_stochastic_advection_diffusion_benchmark,
    run_stochastic_heat_convergence_benchmark,
)


def test_stochastic_heat_refinements_and_invariant_moments_pass():
    result = run_stochastic_heat_convergence_benchmark(
        jr.key(5),
        temporal_paths=128,
        moment_paths=1024,
    )

    assert result.passed
    assert result.temporal.sampling_is_subordinate()
    assert result.temporal.mean_square_stable()
    assert result.noise_truncation.levels[-1].strong_rms_error == 0.0


def test_stochastic_advection_diffusion_and_stratonovich_correction_pass():
    result = run_stochastic_advection_diffusion_benchmark(jr.key(6))

    assert result.passed
    assert result.derivative_noise_variance_error < 0.08
    assert result.stratonovich_correction_error < 1e-10


def test_multiplicative_reaction_diffusion_strong_and_weak_rates_pass():
    result = run_multiplicative_reaction_diffusion_benchmark(jr.key(7))

    assert result.passed
    assert result.temporal.regression_rate("strong") > 0.35
    assert result.temporal.regression_rate(observable="second_moment") > 0.5


def test_commutative_noise_benchmark_separates_levy_area_regimes():
    result = run_commutative_noise_benchmark()

    assert result.passed
    assert result.commutative_flow_order_error < result.noncommutative_flow_order_error
