#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import pytest

from phydrax._interpolation import fourier_interpolate
from phydrax.signal import fourier_resample


@pytest.mark.parametrize("source_size,target_size", ((5, 8), (6, 9), (9, 6), (8, 5)))
def test_fourier_resample_preserves_representable_complex_modes(
    source_size,
    target_size,
):
    source_points = jnp.arange(source_size, dtype=float) / source_size
    target_points = jnp.arange(target_size, dtype=float) / target_size
    values = jnp.exp(2.0j * jnp.pi * source_points)

    output = fourier_resample(values, (target_size,))

    assert jnp.allclose(
        output,
        jnp.exp(2.0j * jnp.pi * target_points),
        rtol=1e-12,
        atol=1e-12,
    )


def test_even_nyquist_mode_splits_on_upsampling_and_merges_on_downsampling():
    source_size = 8
    fine_size = 13
    source = (-1.0) ** jnp.arange(source_size, dtype=float)

    fine = fourier_resample(source, (fine_size,))
    restored = fourier_resample(fine, (source_size,))
    expected_fine = jnp.cos(
        2.0 * jnp.pi * (source_size // 2) * jnp.arange(fine_size) / fine_size
    )

    assert jnp.allclose(fine, expected_fine, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(restored, source, rtol=1e-12, atol=1e-12)


def test_explicit_axes_preserve_batch_and_payload_axes():
    values = jnp.full((2, 5, 6, 3), 4.25)

    output = fourier_resample(values, (8, 9), axes=(1, 2))

    assert output.shape == (2, 8, 9, 3)
    assert jnp.allclose(output, 4.25)


def test_trailing_axis_default_and_explicit_middle_axis_are_unambiguous():
    x = jnp.arange(5, dtype=float) / 5.0
    values = jnp.stack(
        (
            jnp.cos(2.0 * jnp.pi * x),
            jnp.sin(2.0 * jnp.pi * x),
        ),
        axis=0,
    )
    target = jnp.arange(8, dtype=float) / 8.0

    output = fourier_resample(values, (8,), axes=(1,))
    trailing = fourier_resample(values, (8,))

    assert output.shape == (2, 8)
    assert trailing.shape == (2, 8)
    assert jnp.allclose(output, trailing)
    assert jnp.allclose(output[0], jnp.cos(2.0 * jnp.pi * target))
    assert jnp.allclose(output[1], jnp.sin(2.0 * jnp.pi * target))


def test_fourier_resampling_is_jittable_and_has_linear_gradients():
    target_size = 9

    @jax.jit
    def total(values):
        return jnp.sum(fourier_resample(values, (target_size,)))

    values = jnp.arange(6.0)
    output = total(values)
    gradient = jax.grad(total)(values)

    assert jnp.isfinite(output)
    assert jnp.allclose(gradient, target_size / values.size)


def test_shifted_resampling_matches_direct_multiaxis_evaluation():
    source_shape = (8, 6)
    target_shape = (9, 11)
    offsets = (0.137, -0.219)
    values = jax.random.normal(
        jax.random.key(12), source_shape + (2,), dtype=jnp.complex128
    )
    q0, q1 = jnp.meshgrid(
        offsets[0] + jnp.arange(target_shape[0], dtype=float) / target_shape[0],
        offsets[1] + jnp.arange(target_shape[1], dtype=float) / target_shape[1],
        indexing="ij",
    )
    queries = jnp.stack((q0, q1), axis=-1)

    shifted = fourier_resample(
        values,
        target_shape,
        axes=(0, 1),
        phase_offsets=offsets,
    )
    direct = fourier_interpolate(values, queries, spatial_ndim=2).values

    assert shifted.shape == target_shape + (2,)
    assert jnp.allclose(shifted, direct, rtol=1e-11, atol=1e-11)


def test_zero_phase_offsets_equal_ordinary_resampling():
    values = jax.random.normal(jax.random.key(13), (6, 8, 2))

    ordinary = fourier_resample(values, (9, 7), axes=(0, 1))
    shifted = fourier_resample(
        values,
        (9, 7),
        axes=(0, 1),
        phase_offsets=(0.0, 0.0),
    )

    assert jnp.allclose(shifted, ordinary, rtol=1e-12, atol=1e-12)
