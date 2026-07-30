import jax.numpy as jnp
import jax.random as jr

from tools.operator_benchmarks import (
    allen_cahn_transition_data,
    run_allen_cahn_flow_benchmark,
    run_stochastic_heat_gaussian_benchmark,
    stochastic_heat_transition_data,
)


def test_stochastic_heat_transition_generator_replays_and_matches_linear_moments():
    first = stochastic_heat_transition_data(
        jr.key(0),
        grid_size=6,
        num_cases=3,
        num_realizations=64,
        duration=0.04,
        dt0=0.004,
        noise_rank=2,
    )
    replay = stochastic_heat_transition_data(
        jr.key(0),
        grid_size=6,
        num_cases=3,
        num_realizations=64,
        duration=0.04,
        dt0=0.004,
        noise_rank=2,
    )
    dataset = first.operator_dataset()
    empirical_mean = jnp.mean(first.final_states, axis=1)

    assert first.initial_states.shape == (3, 6)
    assert first.final_states.shape == (3, 64, 6)
    assert first.analytic_mean.shape == (3, 6)
    assert first.analytic_covariance.shape == (6, 6)
    assert jnp.array_equal(first.final_states, replay.final_states)
    assert dataset.size == 3 * 64
    assert dataset.batch.case_shape == (3 * 64,)
    assert all(
        record.identities["initial_state"] == f"state:{index // 64}"
        for index, record in enumerate(dataset.provenance)
    )
    assert jnp.sqrt(jnp.mean((empirical_mean - first.analytic_mean) ** 2)) < 0.03


def test_stochastic_heat_low_rank_gaussian_retention_gate():
    data = stochastic_heat_transition_data(
        jr.key(1),
        grid_size=6,
        num_cases=3,
        num_realizations=48,
        duration=0.04,
        dt0=0.004,
        noise_rank=2,
    )
    result = run_stochastic_heat_gaussian_benchmark(
        jr.key(2),
        data=data,
        evaluation_samples=48,
    )

    assert result.passed
    assert result.low_rank_energy_distance < result.diagonal_energy_distance
    assert result.low_rank_covariance_error < 1e-8
    assert result.location_rmse < 0.04
    assert result.fine_grid_finite


def test_allen_cahn_flowjax_benchmark_retains_distributional_gain_in_two_of_three_seeds():
    data = allen_cahn_transition_data(
        jr.key(3),
        grid_size=6,
        num_cases=4,
        num_realizations=16,
        duration=0.5,
        noise_scale=1.0,
        dt0=0.01,
        noise_rank=2,
    )
    result = run_allen_cahn_flow_benchmark(
        jr.key(4),
        data=data,
        seeds=(3, 4, 5),
        steps=300,
        batch_size=16,
        evaluation_samples=16,
    )

    assert len(result.trials) == 3
    assert all(trial.finite for trial in result.trials)
    assert all(trial.flow_final_nll < trial.flow_initial_nll for trial in result.trials)
    assert sum(trial.won for trial in result.trials) >= 2
    assert result.passed
