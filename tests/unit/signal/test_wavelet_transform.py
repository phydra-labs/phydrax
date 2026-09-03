#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import pywt

from phydrax.signal import DiscreteWaveletTransform


_CASES = (
    ((31,), (0,), 3, "db2", "periodization"),
    ((2, 32, 3), (1,), 3, "db4", "symmetric"),
    (
        (2, 15, 16, 2),
        (1, 2),
        2,
        ("db2", "haar"),
        ("periodization", "symmetric"),
    ),
    (
        (10, 11, 12, 2),
        (0, 1, 2),
        1,
        ("haar", "db2", "sym4"),
        ("zero", "periodization", "symmetric"),
    ),
)


def _pywavelets_coefficients(values, transform):
    return pywt.wavedecn(
        np.asarray(values),
        wavelet=tuple(bank.name for bank in transform.filter_banks),
        mode=tuple(transform.boundaries),
        axes=transform.axes,
        level=transform.levels,
    )


@pytest.mark.parametrize("shape,axes,levels,wavelet,boundary", _CASES)
def test_discrete_wavelet_transform_matches_pywavelets_and_roundtrips(
    shape, axes, levels, wavelet, boundary
):
    values = np.random.default_rng(831).normal(size=shape)
    transform = DiscreteWaveletTransform(
        axes,
        levels=levels,
        wavelet=wavelet,
        boundary=boundary,
    )

    actual = transform.analysis(values)
    expected = _pywavelets_coefficients(values, transform)

    assert np.allclose(actual.scaling, expected[0], rtol=1e-12, atol=1e-12)
    for detail_level, expected_level in zip(actual.details, expected[1:], strict=True):
        for label, band in zip(transform.detail_labels, detail_level, strict=True):
            key = "".join("ad"[value] for value in label)
            assert np.allclose(band, expected_level[key], rtol=1e-12, atol=1e-12)
    assert np.allclose(transform.synthesis(actual), values, rtol=5e-12, atol=5e-12)


def test_discrete_wavelet_transform_is_jittable_vmappable_and_differentiable():
    transform = DiscreteWaveletTransform(
        (-2,), levels=3, wavelet="db2", boundary="periodization"
    )
    values = jnp.asarray(np.random.default_rng(77).normal(size=(4, 24, 3)))

    coefficients = eqx.filter_jit(lambda plan, field: plan.analysis(field))(
        transform, values
    )
    reconstructed = eqx.filter_jit(lambda plan, coeffs: plan.synthesis(coeffs))(
        transform, coefficients
    )
    mapped = jax.vmap(lambda field: transform.synthesis(transform.analysis(field)))(
        values
    )
    gradient = jax.grad(
        lambda field: jnp.sum(transform.synthesis(transform.analysis(field)) ** 2)
    )(values)

    assert jnp.allclose(reconstructed, values, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(mapped, values, rtol=1e-12, atol=1e-12)
    assert jnp.allclose(gradient, 2.0 * values, rtol=1e-11, atol=1e-11)


def test_discrete_wavelet_transform_is_shape_independent_and_plan_safe():
    transform = DiscreteWaveletTransform(
        (-2,), levels=2, wavelet="db2", boundary="periodization"
    )
    short = jnp.arange(17.0).reshape(17, 1)
    long = jnp.arange(29.0).reshape(29, 1)

    short_coefficients = transform.analysis(short)
    long_coefficients = transform.analysis(long)

    assert transform.synthesis(short_coefficients).shape == short.shape
    assert transform.synthesis(long_coefficients).shape == long.shape
    assert short_coefficients.transform_fingerprint == transform.fingerprint
    assert long_coefficients.transform_fingerprint == transform.fingerprint

    incompatible = DiscreteWaveletTransform(
        (-2,), levels=2, wavelet="haar", boundary="periodization"
    )
    with pytest.raises(ValueError, match="different transform"):
        incompatible.synthesis(short_coefficients)


def test_discrete_wavelet_transform_rejects_invalid_configuration_and_shapes():
    with pytest.raises(ValueError, match="at least one axis"):
        DiscreteWaveletTransform((), levels=1)
    with pytest.raises(ValueError, match="unique"):
        DiscreteWaveletTransform((0, 0), levels=1)
    with pytest.raises(ValueError, match="positive"):
        DiscreteWaveletTransform((0,), levels=0)
    with pytest.raises(ValueError, match="one value per transformed axis"):
        DiscreteWaveletTransform((0, 1), levels=1, wavelet=("haar",))
    with pytest.raises(ValueError, match="boundaries"):
        DiscreteWaveletTransform((0,), levels=1, boundary="reflect")

    transform = DiscreteWaveletTransform((0,), levels=3)
    with pytest.raises(ValueError, match="Too many wavelet levels"):
        transform.analysis(jnp.ones((4,)))
    with pytest.raises(ValueError, match="invalid"):
        DiscreteWaveletTransform((2,), levels=1).analysis(jnp.ones((4, 4)))
