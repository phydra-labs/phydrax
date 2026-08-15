#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import cast

import jax
import jax.numpy as jnp

import phydrax as phx


def test_fourier_grid_sampler_is_owned_by_layer_namespace():
    assert "sample_fourier_grid" in phx.nn.layers.__all__
    assert "FourierEvaluationMethod" in phx.nn.layers.__all__
    assert "sample_fourier_grid" not in vars(phx.nn.models)
    assert "sample_fourier_grid" not in vars(phx.nn)


def test_fourier_grid_sampler_uses_normalized_periodic_coordinates():
    size = 9
    nodes = -1.0 + 2.0 * jnp.arange(size, dtype=float) / size
    values = jnp.stack(
        (
            jnp.cos(jnp.pi * nodes),
            jnp.sin(2.0 * jnp.pi * nodes),
        ),
        axis=-1,
    )
    query = jnp.asarray([[-0.73], [0.16], [1.27]])

    output, support = phx.nn.layers.sample_fourier_grid(
        values,
        query,
        spatial_ndim=1,
        return_support=True,
    )
    expected = jnp.stack(
        (
            jnp.cos(jnp.pi * query[:, 0]),
            jnp.sin(2.0 * jnp.pi * query[:, 0]),
        ),
        axis=-1,
    )

    assert output.shape == (3, 2)
    assert support.shape == (3,)
    assert jnp.all(support)
    assert jnp.allclose(output, expected, rtol=1e-12, atol=1e-12)


def test_fourier_grid_sampler_supports_physical_nodes_batches_and_nufft():
    batch_size = 2
    size = 8
    origin = 3.0
    period = 5.0
    nodes = origin + period * jnp.arange(size, dtype=float) / size
    base = jnp.stack(
        (
            jnp.cos(2.0 * jnp.pi * (nodes - origin) / period),
            jnp.sin(4.0 * jnp.pi * (nodes - origin) / period),
        ),
        axis=-1,
    )
    values = jnp.stack((base, 2.0 * base), axis=0)
    query = jnp.asarray(
        [
            [[3.2], [5.7], [7.8]],
            [[3.2], [5.7], [7.8]],
        ]
    )

    direct = phx.nn.layers.sample_fourier_grid(
        values,
        query,
        spatial_ndim=1,
        axis_nodes=(nodes,),
        periods=(period,),
    )
    approximate = phx.nn.layers.sample_fourier_grid(
        values,
        query,
        spatial_ndim=1,
        axis_nodes=(nodes,),
        periods=(period,),
        method="nufft",
        tolerance=1e-10,
        query_chunk_size=2,
    )
    direct = cast(jax.Array, direct)
    approximate = cast(jax.Array, approximate)

    assert direct.shape == (batch_size, 3, 2)
    assert jnp.allclose(approximate, direct, rtol=2e-8, atol=2e-8)
    assert jnp.allclose(direct[1], 2.0 * direct[0], rtol=1e-12, atol=1e-12)


def test_public_spectral_resample_evaluates_shifted_uniform_grid():
    source_size = 8
    target_size = 11
    offset = 0.125
    source = jnp.arange(source_size, dtype=float) / source_size
    values = jnp.stack(
        (
            jnp.cos(2.0 * jnp.pi * source),
            jnp.sin(4.0 * jnp.pi * source),
        ),
        axis=-1,
    )
    target = offset + jnp.arange(target_size, dtype=float) / target_size

    output = phx.nn.operator.architectures.spectral_resample(
        values,
        (target_size,),
        phase_offsets=(offset,),
    )
    expected = jnp.stack(
        (
            jnp.cos(2.0 * jnp.pi * target),
            jnp.sin(4.0 * jnp.pi * target),
        ),
        axis=-1,
    )

    assert output.shape == (target_size, 2)
    assert jnp.allclose(output, expected, rtol=1e-12, atol=1e-12)


def test_fourier_grid_sampler_is_jittable_and_differentiable_in_queries():
    nodes = -1.0 + 2.0 * jnp.arange(7, dtype=float) / 7.0
    values = jnp.cos(jnp.pi * nodes)[:, None]
    query = jnp.asarray([[-0.4], [0.3]])

    def total(points):
        sampled = cast(
            jax.Array,
            phx.nn.layers.sample_fourier_grid(
                values,
                points,
                spatial_ndim=1,
                method="nufft",
                tolerance=1e-10,
            ),
        )
        return jnp.sum(sampled)

    output = jax.jit(total)(query)
    gradient = jax.jit(jax.grad(total))(query)
    expected_gradient = -jnp.pi * jnp.sin(jnp.pi * query)

    assert jnp.isfinite(output)
    assert jnp.all(jnp.isfinite(gradient))
    assert jnp.allclose(gradient, expected_gradient, rtol=2e-7, atol=2e-7)
