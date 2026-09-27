import jax.random as jr

from tools.stochastic_non_brownian import (
    run_fractional_rough_reference_benchmark,
    run_levy_stable_reference_benchmark,
    run_memory_particle_reference_benchmark,
)


def test_non_brownian_reference_benchmarks_scenario_1() -> None:
    result = run_levy_stable_reference_benchmark(jr.key(100), num_paths=8192)

    assert result.passed
    assert result.complete_path_fraction == 1.0
    assert result.characteristic_function_max_error < 0.035
    result = run_fractional_rough_reference_benchmark(jr.key(101), num_paths=1024)

    assert result.passed
    assert result.covariance_relative_error < 0.12
    assert result.rough_linear_relative_rmse < 4e-3
    result = run_memory_particle_reference_benchmark(jr.key(102), num_paths=4096)

    assert result.passed
    assert result.volterra_variance_relative_error < 0.1
    assert result.particle_contraction_error < 1e-12
