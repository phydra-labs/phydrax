#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from phydrax._interpolation import fourier_interpolate


def test_direct_fourier_interpolation_uses_physical_axis_geometry_and_periodicity():
    count = 9
    origin = 2.5
    period = 3.0
    nodes = origin + period * jnp.arange(count, dtype=float) / count
    values = jnp.stack(
        (
            jnp.cos(4.0 * jnp.pi * (nodes - origin) / period),
            jnp.sin(2.0 * jnp.pi * (nodes - origin) / period),
        ),
        axis=-1,
    )
    query = jnp.asarray([[2.73], [4.91], [2.73 + 2.0 * period]])

    result = fourier_interpolate(
        values,
        query,
        spatial_ndim=1,
        axis_nodes=(nodes,),
        periods=(period,),
    )
    expected = jnp.stack(
        (
            jnp.cos(4.0 * jnp.pi * (query[:, 0] - origin) / period),
            jnp.sin(2.0 * jnp.pi * (query[:, 0] - origin) / period),
        ),
        axis=-1,
    )

    assert jnp.all(result.support)
    assert jnp.allclose(result.values, expected, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(result.values[0], result.values[2], rtol=1e-12, atol=1e-12)


def test_fourier_interpolation_preserves_batch_query_and_tensor_payload_axes():
    batch_size = 2
    source_shape = (5, 6)
    payload_shape = (2, 3)
    values = jax.random.normal(
        jax.random.key(14),
        (batch_size,) + source_shape + payload_shape,
    )
    queries = jax.random.uniform(jax.random.key(15), (batch_size, 3, 4, 2))

    eager = fourier_interpolate(
        values,
        queries,
        spatial_ndim=2,
        payload_ndim=2,
    )
    chunked = fourier_interpolate(
        values,
        queries,
        spatial_ndim=2,
        payload_ndim=2,
        query_chunk_size=5,
    )

    assert eager.values.shape == (batch_size, 3, 4) + payload_shape
    assert eager.support.shape == (batch_size, 3, 4)
    assert jnp.all(eager.support)
    assert jnp.allclose(chunked.values, eager.values, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("source_shape", ((7,), (8,), (7, 6), (6, 5), (4, 5, 6)))
def test_nufft_interpolation_matches_direct_for_broadband_complex_values(source_shape):
    dimensions = len(source_shape)
    values = jax.random.normal(
        jax.random.key(20 + dimensions),
        source_shape + (2,),
        dtype=jnp.complex128,
    )
    queries = jax.random.uniform(jax.random.key(30 + dimensions), (11, dimensions))

    direct = fourier_interpolate(
        values,
        queries,
        spatial_ndim=dimensions,
    )
    approximate = fourier_interpolate(
        values,
        queries,
        spatial_ndim=dimensions,
        method="nufft",
        tolerance=1e-10,
        query_chunk_size=4,
    )

    assert approximate.values.shape == direct.values.shape
    assert jnp.array_equal(approximate.support, direct.support)
    assert jnp.allclose(approximate.values, direct.values, rtol=2e-8, atol=2e-8)


def test_direct_and_nufft_reconstruct_real_values_at_even_source_nodes():
    source_shape = (8, 6)
    values = jax.random.normal(jax.random.key(42), source_shape + (3,))
    q0, q1 = jnp.meshgrid(
        jnp.arange(source_shape[0], dtype=float) / source_shape[0],
        jnp.arange(source_shape[1], dtype=float) / source_shape[1],
        indexing="ij",
    )
    queries = jnp.stack((q0, q1), axis=-1)

    direct = fourier_interpolate(values, queries, spatial_ndim=2).values
    approximate = fourier_interpolate(
        values,
        queries,
        spatial_ndim=2,
        method="nufft",
        tolerance=1e-10,
    ).values

    assert not jnp.iscomplexobj(direct)
    assert not jnp.iscomplexobj(approximate)
    assert jnp.allclose(direct, values, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(approximate, values, rtol=2e-8, atol=2e-8)


def test_fourier_point_evaluation_jit_and_coordinate_gradients_agree():
    source_size = 9
    nodes = jnp.arange(source_size, dtype=float) / source_size
    values = jnp.stack(
        (
            jnp.cos(4.0 * jnp.pi * nodes),
            jnp.sin(2.0 * jnp.pi * nodes),
        ),
        axis=-1,
    )
    query = jnp.asarray([[0.13], [0.41], [0.82]])

    def loss(points, method):
        kwargs: dict[str, Any] = (
            {} if method == "direct" else {"method": "nufft", "tolerance": 1e-10}
        )
        output = fourier_interpolate(
            values,
            points,
            spatial_ndim=1,
            **kwargs,
        ).values
        return jnp.sum(output**2)

    direct_value, direct_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=1)(
        query, "direct"
    )
    nufft_value, nufft_gradient = jax.jit(jax.value_and_grad(loss), static_argnums=1)(
        query, "nufft"
    )

    assert jnp.allclose(nufft_value, direct_value, rtol=2e-8, atol=2e-8)
    assert jnp.allclose(nufft_gradient, direct_gradient, rtol=2e-7, atol=2e-7)


def test_direct_fourier_interpolation_supports_scalar_queries_and_four_axes():
    values = jnp.ones((3, 3, 3, 3))
    query = jnp.asarray([0.12, 0.23, 0.34, 0.45])

    result = fourier_interpolate(
        values,
        query,
        spatial_ndim=4,
        payload_ndim=0,
    )

    assert result.values.shape == ()
    assert result.support.shape == ()
    assert bool(result.support)
    assert jnp.allclose(result.values, 1.0)


def test_fourier_interpolation_rejects_incompatible_backend_contracts():
    values = jnp.ones((5, 1))
    query = jnp.asarray([[0.2]])

    with pytest.raises(ValueError, match="does not accept a tolerance"):
        fourier_interpolate(
            values,
            query,
            spatial_ndim=1,
            tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="requires a tolerance"):
        fourier_interpolate(
            values,
            query,
            spatial_ndim=1,
            method="nufft",
        )
    with pytest.raises(ValueError, match="one to three"):
        fourier_interpolate(
            jnp.ones((2, 2, 2, 2, 1)),
            jnp.ones((1, 4)),
            spatial_ndim=4,
            method="nufft",
            tolerance=1e-6,
        )
    with pytest.raises(ValueError, match="query_chunk_size"):
        fourier_interpolate(
            values,
            query,
            spatial_ndim=1,
            query_chunk_size=0,
        )


def test_fourier_interpolation_rejects_nonuniform_periodic_nodes():
    with pytest.raises(eqx.EquinoxRuntimeError, match="uniformly spaced"):
        fourier_interpolate(
            jnp.ones((4, 1)),
            jnp.asarray([[0.1]]),
            spatial_ndim=1,
            axis_nodes=(jnp.asarray([0.0, 0.2, 0.5, 0.8]),),
            periods=(1.0,),
        )
