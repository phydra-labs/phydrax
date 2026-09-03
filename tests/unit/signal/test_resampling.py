#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import signal as scipy_signal

from phydrax.signal import (
    kaiser_sinc_resampling_filter,
    resample_poly,
    upfirdn,
)


@pytest.mark.parametrize("up,down", ((1, 1), (2, 1), (1, 3), (3, 2), (4, 2)))
def test_raw_upfirdn_matches_scipy_without_ratio_reduction_or_tap_scaling(up, down):
    values = np.asarray((1.0, -0.5, 2.0, 0.25))
    taps = np.asarray((0.2, 0.5, -0.1, 0.3, 0.7))

    actual = upfirdn(values, taps, up=up, down=down)
    expected = scipy_signal.upfirdn(taps, values, up=up, down=down)

    assert actual.shape == expected.shape
    assert np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_upfirdn_preserves_middle_axis_and_complex_values():
    values = jnp.arange(2 * 5 * 3, dtype=float).reshape((2, 5, 3)).astype(complex)
    values = values + 0.25j
    taps = jnp.asarray((1.0 + 0.5j, -0.25j, 0.1))

    output = upfirdn(values, taps, up=3, down=2, axis=1)
    expected = scipy_signal.upfirdn(
        np.asarray(taps),
        np.asarray(values),
        up=3,
        down=2,
        axis=1,
    )

    assert output.shape == expected.shape
    assert np.allclose(output, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("up,down", ((3, 2), (5, 7), (4, 2), (147, 160)))
def test_finite_resample_poly_matches_scipy_default_alignment(up, down):
    values = np.linspace(-1.0, 1.0, 23)

    actual = resample_poly(values, up, down)
    expected = scipy_signal.resample_poly(values, up, down, window=("kaiser", 5.0))

    assert actual.shape == ((values.size * up + down - 1) // down,)
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)


def test_explicit_prototype_matches_scipy_and_rejects_even_centering():
    values = jnp.sin(2.0 * jnp.pi * 0.07 * jnp.arange(31))
    prototype = kaiser_sinc_resampling_filter(3, 2, half_width=6)

    actual = resample_poly(values, 3, 2, taps=prototype)
    expected = scipy_signal.resample_poly(
        np.asarray(values),
        3,
        2,
        window=np.asarray(prototype),
    )

    assert jnp.allclose(jnp.sum(prototype), 1.0, rtol=1e-12, atol=1e-12)
    assert np.allclose(actual, expected, rtol=1e-11, atol=1e-11)
    with pytest.raises(ValueError, match="odd tap count"):
        resample_poly(values, 3, 2, taps=jnp.ones((4,)))


def test_polyphase_paths_are_jittable_and_differentiable_in_values_and_taps():
    values = jnp.linspace(-1.0, 1.0, 17)
    taps = jnp.asarray((0.1, 0.3, 0.5, 0.3, 0.1))

    @jax.jit
    def loss(x, h):
        output = upfirdn(x, h, up=3, down=2)
        return jnp.sum(jnp.abs(output) ** 2)

    value_gradient, tap_gradient = jax.grad(loss, argnums=(0, 1))(values, taps)

    assert jnp.all(jnp.isfinite(value_gradient))
    assert jnp.all(jnp.isfinite(tap_gradient))
