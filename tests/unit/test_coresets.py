#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import jax.random as jr
import pytest

import phydrax as phx


def _selection_weights(selection):
    return jnp.where(selection.mask, jnp.exp(selection.log_weights), 0.0)


def test_moment_recombination_preserves_weighted_feature_moments():
    points = jnp.linspace(-2.0, 2.0, 65)
    features = jnp.stack((points, points**2, jnp.sin(points)), axis=1)
    source_weights = jnp.linspace(1.0, 3.0, points.size)
    source_weights = source_weights / jnp.sum(source_weights)

    selection = phx.coresets.moment_recombine(
        features,
        log_weights=jnp.log(source_weights),
    )
    weights = _selection_weights(selection)

    assert bool(selection.diagnostics.valid)
    assert int(selection.diagnostics.active_points) <= features.shape[1] + 1
    assert jnp.all(weights >= 0.0)
    assert jnp.isclose(jnp.sum(weights), 1.0, atol=1e-12)
    assert jnp.allclose(
        weights @ features[selection.indices],
        source_weights @ features,
        atol=1e-10,
        rtol=1e-10,
    )


def test_moment_recombination_respects_mask_and_rank_deficiency():
    features = jnp.stack(
        (
            jnp.arange(12.0),
            jnp.ones((12,)),
            jnp.ones((12,)),
        ),
        axis=1,
    )
    mask = jnp.arange(12) % 2 == 0

    selection = phx.coresets.moment_recombine(features, mask=mask)

    assert bool(selection.diagnostics.valid)
    assert int(selection.diagnostics.active_points) <= 2
    assert jnp.all(mask[selection.indices[selection.mask]])
    assert float(selection.diagnostics.minimum_weight) >= 0.0


def test_weighted_mmd_matches_dense_kernel_evaluation():
    source = jnp.linspace(-1.0, 1.0, 17)[:, None]
    comparison = jnp.asarray([[-0.75], [0.0], [0.9]])
    source_weights = jnp.arange(1.0, 18.0)
    source_weights = source_weights / jnp.sum(source_weights)
    comparison_weights = jnp.asarray([0.2, 0.3, 0.5])
    kernel = phx.coresets.RadialKernel("matern32", length_scale=0.4)

    actual = phx.coresets.weighted_mmd(
        source,
        comparison,
        source_log_weights=jnp.log(source_weights),
        comparison_log_weights=jnp.log(comparison_weights),
        kernel=kernel,
        block_size=4,
    )
    source_gram = phx.coresets.kernel_matrix(kernel, source, source)
    comparison_gram = phx.coresets.kernel_matrix(kernel, comparison, comparison)
    cross_gram = phx.coresets.kernel_matrix(kernel, source, comparison)
    expected_squared = (
        source_weights @ source_gram @ source_weights
        + comparison_weights @ comparison_gram @ comparison_weights
        - 2.0 * source_weights @ cross_gram @ comparison_weights
    )

    assert jnp.isclose(actual, jnp.sqrt(jnp.maximum(expected_squared, 0.0)))


def test_weighted_mmd_rejects_nonpositive_block_size():
    with pytest.raises(ValueError, match="block_size"):
        phx.coresets.weighted_mmd(
            jnp.zeros((2, 1)),
            jnp.zeros((2, 1)),
            block_size=0,
        )


def test_kernel_herding_returns_unique_active_source_points():
    points = jnp.linspace(-3.0, 3.0, 41)[:, None]
    mask = jnp.arange(points.shape[0]) % 3 != 0
    selection = phx.coresets.kernel_herd(
        points,
        phx.coresets.KernelHerding(9, block_size=7),
        mask=mask,
    )
    active_indices = selection.indices[selection.mask]
    weights = _selection_weights(selection)

    assert bool(selection.diagnostics.valid)
    assert active_indices.shape == (9,)
    assert jnp.unique(active_indices).shape == active_indices.shape
    assert jnp.all(mask[active_indices])
    assert jnp.allclose(weights[selection.mask], jnp.full((9,), 1.0 / 9.0))
    assert jnp.isfinite(selection.diagnostics.mmd)


def test_randomized_pivoted_cholesky_is_keyed_and_reduces_trace():
    points = jnp.linspace(0.0, 1.0, 48)[:, None]
    method = phx.coresets.RandomizedPivotedCholesky(
        10,
        kernel=phx.coresets.RadialKernel(length_scale=0.15),
    )
    first = phx.coresets.randomized_pivoted_cholesky(
        points,
        method,
        key=jr.key(7),
    )
    repeated = phx.coresets.randomized_pivoted_cholesky(
        points,
        method,
        key=jr.key(7),
    )

    assert bool(first.diagnostics.valid)
    assert jnp.array_equal(first.indices, repeated.indices)
    assert jnp.unique(first.indices).shape == first.indices.shape
    assert first.diagnostics.residual_trace < first.diagnostics.initial_trace
    assert 0.0 < first.diagnostics.explained_trace_fraction <= 1.0


def test_randomized_pivoted_cholesky_rejects_oversized_selection():
    with pytest.raises(ValueError, match="more inducing points"):
        phx.coresets.randomized_pivoted_cholesky(
            jnp.ones((3, 1)),
            phx.coresets.RandomizedPivotedCholesky(4),
        )
