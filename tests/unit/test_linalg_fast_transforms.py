#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import scipy.fft as scipy_fft

import phydrax as phx


@pytest.mark.parametrize("kind", ["dct", "dst"])
@pytest.mark.parametrize("transform_type", [1, 2, 3, 4])
def test_fast_trigonometric_transforms_match_scipy_and_invert(kind, transform_type):
    transform = phx.linalg.RealTrigonometricTransform(kind, transform_type, 17)
    values = jnp.linspace(-1.0, 2.0, 17)
    reference = scipy_fft.dct if kind == "dct" else scipy_fft.dst

    coefficients = eqx.filter_jit(transform.analyze)(values)
    reconstructed = eqx.filter_jit(transform.synthesize)(coefficients)

    np.testing.assert_allclose(
        coefficients,
        reference(np.asarray(values), type=transform_type, norm="ortho"),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(reconstructed, values, rtol=2e-12, atol=2e-12)


def test_tensor_fast_transform_handles_complex_intermediate_values_and_gradients():
    transform = phx.linalg.TensorLinearTransform(
        (
            phx.linalg.FFTLinearTransform(8),
            phx.linalg.RealTrigonometricTransform("dst", 4, 7),
        )
    )
    values = jnp.arange(56.0).reshape((8, 7)) / 11.0

    coefficients = eqx.filter_jit(transform.analyze)(values)
    reconstructed = eqx.filter_jit(transform.synthesize)(coefficients)
    gradient = jax.grad(
        lambda field: jnp.real(
            jnp.sum(jnp.conj(transform.analyze(field)) * transform.analyze(field))
        )
    )(values)

    np.testing.assert_allclose(jnp.real(reconstructed), values, rtol=2e-12, atol=2e-12)
    np.testing.assert_allclose(jnp.imag(reconstructed), 0.0, atol=2e-12)
    np.testing.assert_allclose(gradient, 2.0 * values, rtol=2e-11, atol=2e-11)
