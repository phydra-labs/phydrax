import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def test_fractional_gaussian_realization_matches_finite_grid_covariance():
    process = phx.stochastic.FractionalGaussianProcess(
        0.75,
        0.8,
        process_id="persistent-fractional-gaussian",
    )
    grid = jnp.linspace(0.0, 1.0, 6)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(60),
        grid,
        sample_shape=(4096,),
    )
    samples = realization.values[..., 0]
    empirical = jnp.cov(samples, rowvar=False)
    expected = process.time_covariance(grid[:, None], grid[None, :]) * 0.8**2
    relative_error = jnp.linalg.norm(empirical - expected) / jnp.linalg.norm(expected)
    increments = realization.fractional_gaussian_noise[..., 0]
    empirical_adjacent = jnp.mean(increments[:, 0] * increments[:, 1])
    expected_adjacent = process.increment_covariance(grid[0], grid[1], grid[1], grid[2])[
        ..., 0, 0
    ]

    assert realization.values.shape == (4096, 6, 1)
    assert jnp.allclose(samples[:, 0], 0.0, rtol=0.0, atol=1e-12)
    assert relative_error < 0.04
    assert jnp.allclose(empirical_adjacent, expected_adjacent, rtol=0.12)


def test_fractional_paths_are_prefix_stable_across_sample_batch_growth():
    process = phx.stochastic.FractionalGaussianProcess(
        0.35,
        jnp.asarray([0.5, 1.2]),
        process_id="rough-two-component-fractional-gaussian",
    )
    grid = jnp.linspace(0.0, 1.0, 9)
    first = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(61),
        grid,
        sample_shape=(3,),
    )
    wider = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(61),
        grid,
        sample_shape=(7,),
    )

    assert first.coupling_id == wider.coupling_id
    assert first.realization_id != wider.realization_id
    assert jnp.array_equal(first.values, wider.values[:3])


def test_fractional_linear_queries_share_one_additive_global_interpolant():
    process = phx.stochastic.FractionalGaussianProcess(
        0.6,
        1.0,
        drift=0.2,
        process_id="interpolated-fractional-gaussian",
    )
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(62),
        jnp.asarray([0.0, 0.5, 1.0]),
        sample_shape=(4,),
    )
    pieces = realization.increments(
        jnp.asarray([0.0, 0.3]),
        jnp.asarray([0.3, 1.0]),
        interpolation="linear",
    )
    whole = realization.increments(
        jnp.asarray([0.0]),
        jnp.asarray([1.0]),
        interpolation="linear",
    )[:, 0]
    trajectory = realization.to_stochastic_trajectory(realization_axes=("path",))

    assert jnp.allclose(jnp.sum(pieces, axis=1), whole, rtol=0.0, atol=1e-12)
    assert trajectory.states.shape == (4, 3, 1)
    assert trajectory.realizations == (realization,)
    with pytest.raises(ValueError, match="grid node"):
        realization.evaluate(jnp.asarray([0.3]), interpolation="grid")


def test_hurst_half_recovers_brownian_covariance_contract():
    process = phx.stochastic.FractionalGaussianProcess(0.5, 1.0)
    left = jnp.asarray([[0.1], [0.7]])
    right = jnp.asarray([[0.2, 0.9]])

    assert jnp.allclose(
        process.time_covariance(left, right),
        jnp.minimum(left, right),
        rtol=0.0,
        atol=1e-12,
    )
