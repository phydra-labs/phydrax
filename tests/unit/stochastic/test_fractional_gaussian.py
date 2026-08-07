import hashlib
from typing import get_args

import equinox as eqx
import jax
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



def test_dense_method_preserves_seeded_values_and_identifiers():
    process = phx.stochastic.FractionalGaussianProcess(
        0.35,
        jnp.asarray([0.5, 1.2]),
        process_id="rough-two-component-fractional-gaussian",
    )
    grid = jnp.linspace(0.0, 1.0, 9)
    default = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(61),
        grid,
        sample_shape=(3,),
    )
    explicit = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(61),
        grid,
        sample_shape=(3,),
        method="dense",
    )
    automatic = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(61),
        grid,
        sample_shape=(3,),
        method="auto",
    )
    assert set(get_args(phx.stochastic.FractionalGaussianSamplingMethod)) == {
        "dense",
        "davies-harte",
        "auto",
    }
    values_digest = hashlib.sha256(
        bytes(jax.device_get(default.values))
    ).hexdigest()

    assert default.sampling_method == explicit.sampling_method == "dense"
    assert automatic.sampling_method == "dense"
    assert automatic.sampling_provenance == "auto:dense-small-grid"
    assert jnp.array_equal(default.values, explicit.values)
    assert jnp.array_equal(default.values, automatic.values)
    assert (
        default.realization_id
        == explicit.realization_id
        == automatic.realization_id
        == "2deb8edb0eececd5ec19ea467b854c04e199c203065f8831c212d922d0978f7f"
    )
    assert (
        default.coupling_id
        == explicit.coupling_id
        == automatic.coupling_id
        == "0c2e26058e824fcf5243448ae5fa48830f165bb204a2a625ba9c46fd6c5ab92b"
    )
    assert values_digest == (
        "9cf30c90eec6f2da47feff3c0fea917d48ee3c8a7059cf8a311ebbb66b87359c"
    )


def test_davies_harte_matches_covariance_autocovariance_and_hurst():
    hurst = 0.7
    process = phx.stochastic.FractionalGaussianProcess(hurst, 0.8)
    grid = jnp.linspace(0.0, 1.0, 65)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(63),
        grid,
        sample_shape=(4096,),
        method="davies-harte",
    )
    selected = jnp.asarray([0, 8, 16, 32, 64])
    samples = realization.values[:, selected, 0]
    empirical_covariance = jnp.cov(samples, rowvar=False)
    selected_times = grid[selected]
    exact_covariance = (
        process.time_covariance(
            selected_times[:, None],
            selected_times[None, :],
        )
        * 0.8**2
    )
    covariance_error = jnp.linalg.norm(
        empirical_covariance - exact_covariance
    ) / jnp.linalg.norm(exact_covariance)

    noise = realization.fractional_gaussian_noise[..., 0]
    empirical_autocovariance = jnp.asarray(
        [
            jnp.mean(noise * noise),
            jnp.mean(noise[:, :-1] * noise[:, 1:]),
            jnp.mean(noise[:, :-2] * noise[:, 2:]),
            jnp.mean(noise[:, :-3] * noise[:, 3:]),
        ]
    )
    exact_autocovariance = jnp.asarray(
        [
            process.increment_covariance(
                grid[0],
                grid[1],
                grid[lag],
                grid[lag + 1],
            )[0, 0]
            for lag in range(4)
        ]
    )
    autocovariance_error = jnp.max(
        jnp.abs(empirical_autocovariance - exact_autocovariance)
        / exact_autocovariance
    )

    values = realization.values[..., 0]
    variance_lag_one = jnp.mean((values[:, 1:] - values[:, :-1]) ** 2)
    variance_lag_eight = jnp.mean((values[:, 8:] - values[:, :-8]) ** 2)
    estimated_hurst = 0.5 * jnp.log(variance_lag_eight / variance_lag_one) / jnp.log(
        8.0
    )

    assert realization.sampling_method == "davies-harte"
    assert covariance_error < 0.055
    assert autocovariance_error < 0.055
    assert jnp.allclose(estimated_hurst, hurst, rtol=0.025)


def test_davies_harte_preserves_reference_drift_and_component_contracts():
    process = phx.stochastic.FractionalGaussianProcess(
        0.4,
        jnp.asarray([0.5, 1.2]),
        drift=jnp.asarray([0.3, -0.2]),
        reference_time=-1.0,
    )
    grid = jnp.linspace(-1.0, 1.0, 33)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(64),
        grid,
        sample_shape=(4096,),
        method="davies-harte",
    )
    terminal = realization.values[:, -1, :]
    empirical_mean = jnp.mean(terminal, axis=0)
    empirical_covariance = jnp.cov(terminal, rowvar=False)
    expected_mean = process.mean(grid[-1])
    expected_variance = (grid[-1] - process.reference_time) ** (
        2.0 * process.hurst
    ) * process.scale**2

    assert realization.values.shape == (4096, 33, 2)
    assert jnp.array_equal(realization.values[:, 0], jnp.zeros((4096, 2)))
    assert jnp.allclose(empirical_mean, expected_mean, rtol=0.0, atol=0.035)
    assert jnp.allclose(
        jnp.diag(empirical_covariance),
        expected_variance,
        rtol=0.04,
    )
    assert jnp.abs(empirical_covariance[0, 1]) < 0.035


def test_davies_harte_resolution_prefix_ids_and_scale_validation():
    process = phx.stochastic.FractionalGaussianProcess(
        0.65,
        jnp.asarray([0.4, 1.1]),
        process_id="davies-harte-resolution",
    )
    grid = jnp.linspace(0.0, 1.0, 257)
    explicit = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(65),
        grid,
        sample_shape=(3,),
        method="davies-harte",
    )
    automatic = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(65),
        grid,
        sample_shape=(3,),
        method="auto",
    )
    wider = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(65),
        grid,
        sample_shape=(7,),
        method="davies-harte",
    )
    dense = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(65),
        grid,
        sample_shape=(3,),
        method="dense",
    )
    tiny_grid = jnp.linspace(0.0, 1e-12, 257)
    tiny = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(65),
        tiny_grid,
        method="davies-harte",
    )

    assert explicit.sampling_method == automatic.sampling_method == "davies-harte"
    assert automatic.sampling_provenance == "auto:davies-harte"
    assert jnp.array_equal(explicit.values, automatic.values)
    assert explicit.realization_id == automatic.realization_id
    assert explicit.coupling_id == automatic.coupling_id == wider.coupling_id
    assert explicit.realization_id != dense.realization_id
    assert explicit.coupling_id != dense.coupling_id
    assert jnp.array_equal(explicit.values, wider.values[:3])
    assert tiny.sampling_method == "davies-harte"
    assert jnp.all(jnp.isfinite(tiny.values))


def test_davies_harte_jit_interpolation_trajectory_and_rough_consumers():
    process = phx.stochastic.FractionalGaussianProcess(0.6, 0.7, drift=0.1)
    grid = jnp.linspace(0.0, 1.0, 33)
    realization = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(66),
        grid,
        sample_shape=(2,),
        method="davies-harte",
    )
    compiled_values = eqx.filter_jit(lambda value: value.values)(realization)
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
    rough_path = phx.stochastic.GeometricRoughPath.from_fractional_gaussian(
        realization
    )

    assert jnp.allclose(compiled_values, realization.values, rtol=1e-14, atol=1e-14)
    assert jnp.allclose(jnp.sum(pieces, axis=1), whole, rtol=0.0, atol=1e-12)
    assert trajectory.realization_axes == ("path",)
    assert trajectory.state_axes == ("component",)
    assert trajectory.metadata["sampling_method"] == "davies-harte"
    assert trajectory.metadata["sampling_provenance"] == "explicit:davies-harte"
    assert rough_path.realization is realization
    assert rough_path.driver_id == realization.realization_id
    assert jnp.allclose(
        rough_path.terminal_signature[0],
        realization.values[:, -1] - realization.values[:, 0],
        rtol=0.0,
        atol=1e-12,
    )


def test_davies_harte_rejects_invalid_inputs_and_auto_records_fallbacks():
    process = phx.stochastic.FractionalGaussianProcess(0.7, 1.0)
    small_grid = jnp.linspace(0.0, 1.0, 9)
    nonuniform_grid = jnp.linspace(0.0, 1.0, 257).at[128].add(1e-4)

    with pytest.raises(ValueError, match="method must"):
        phx.stochastic.FractionalGaussianRealization(
            process,
            jr.key(67),
            small_grid,
            method="circulant",
        )
    with pytest.raises(ValueError, match="uniform grid"):
        phx.stochastic.FractionalGaussianRealization(
            process,
            jr.key(67),
            nonuniform_grid,
            method="davies-harte",
        )
    automatic_nonuniform = phx.stochastic.FractionalGaussianRealization(
        process,
        jr.key(67),
        nonuniform_grid,
        method="auto",
    )

    unstable_process = phx.stochastic.FractionalGaussianProcess(
        0.999999999999,
        1.0,
    )
    large_grid = jnp.linspace(0.0, 1.0, 257)
    with pytest.raises(ValueError, match="positive semidefinite"):
        phx.stochastic.FractionalGaussianRealization(
            unstable_process,
            jr.key(68),
            large_grid,
            method="davies-harte",
        )
    automatic_embedding = phx.stochastic.FractionalGaussianRealization(
        unstable_process,
        jr.key(68),
        large_grid,
        method="auto",
    )
    trajectory = automatic_embedding.to_stochastic_trajectory()

    assert automatic_nonuniform.sampling_method == "dense"
    assert (
        automatic_nonuniform.sampling_provenance
        == "auto:dense-nonuniform-grid"
    )
    assert automatic_embedding.sampling_method == "dense"
    assert (
        automatic_embedding.sampling_provenance
        == "auto:dense-invalid-embedding"
    )
    assert trajectory.metadata["sampling_method"] == "dense"
    assert (
        trajectory.metadata["sampling_provenance"]
        == "auto:dense-invalid-embedding"
    )