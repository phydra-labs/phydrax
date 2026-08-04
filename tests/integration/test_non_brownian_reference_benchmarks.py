import jax.random as jr

from tools.stochastic_non_brownian import (
    run_fractional_rough_reference_benchmark,
    run_levy_stable_reference_benchmark,
    run_memory_particle_reference_benchmark,
)


def test_stable_levy_poisson_tail_and_characteristic_function_references_pass():
    result = run_levy_stable_reference_benchmark(jr.key(100), num_paths=8192)

    assert result.passed
    assert result.complete_path_fraction == 1.0
    assert result.characteristic_function_max_error < 0.035


def test_fractional_covariance_self_similarity_and_rough_equation_references_pass():
    result = run_fractional_rough_reference_benchmark(jr.key(101), num_paths=1024)

    assert result.passed
    assert result.covariance_relative_error < 0.12
    assert result.rough_linear_relative_rmse < 4e-3


def test_volterra_delay_and_interacting_particle_references_pass():
    result = run_memory_particle_reference_benchmark(jr.key(102), num_paths=4096)

    assert result.passed
    assert result.volterra_variance_relative_error < 0.1
    assert result.particle_contraction_error < 1e-12
