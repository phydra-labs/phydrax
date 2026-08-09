#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._spectral import (
    AlpertMultiwaveletTransform,
    MultiresolutionCoefficients,
)


_CASES = (
    (1, 1, 7, "periodization"),
    (2, 2, 17, "symmetric"),
    (3, 3, 37, "zero"),
    (4, 2, 33, "periodization"),
)


@pytest.mark.parametrize("order,levels,num_points,boundary", _CASES)
def test_alpert_multiwavelet_roundtrips_divisible_and_padded_lengths(
    order, levels, num_points, boundary
):
    values = jnp.asarray(
        np.random.default_rng(912).normal(size=(2, num_points, 3))
    )
    transform = AlpertMultiwaveletTransform(
        order=order,
        levels=levels,
        boundary=boundary,
    )

    coefficients = transform.analysis(values)
    reconstructed = transform.synthesis(coefficients)

    assert isinstance(coefficients, MultiresolutionCoefficients)
    assert coefficients.levels == levels
    assert all(len(level) == 1 for level in coefficients.details)
    assert reconstructed.shape == values.shape
    assert jnp.allclose(reconstructed, values, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("order", (1, 2, 3, 4))
def test_alpert_multiwavelet_is_orthogonal_and_annihilates_constant_details(order):
    transform = AlpertMultiwaveletTransform(
        order=order,
        levels=3,
        boundary="periodization",
    )
    num_points = order * 2**4
    values = jnp.ones((num_points, 2))

    coefficients = transform.analysis(values)
    coefficient_energy = jnp.sum(coefficients.scaling**2) + sum(
        jnp.sum(band**2)
        for detail_level in coefficients.details
        for band in detail_level
    )

    assert jnp.allclose(transform.base_analysis @ transform.base_synthesis, jnp.eye(order))
    assert jnp.allclose(
        transform.level_analysis @ transform.level_synthesis,
        jnp.eye(2 * order),
        rtol=1e-12,
        atol=1e-12,
    )
    assert jnp.allclose(coefficient_energy, jnp.sum(values**2), rtol=1e-12, atol=1e-12)
    assert max(
        float(jnp.max(jnp.abs(level[0]))) for level in coefficients.details
    ) < 3e-14


def test_alpert_multiwavelet_is_shape_independent_jittable_and_differentiable():
    transform = AlpertMultiwaveletTransform(
        order=3,
        levels=2,
        boundary="periodization",
    )
    short = jnp.asarray(np.random.default_rng(2).normal(size=(17, 2)))
    long = jnp.asarray(np.random.default_rng(3).normal(size=(31, 2)))

    short_coefficients = eqx.filter_jit(
        lambda plan, values: plan.analysis(values)
    )(transform, short)
    short_reconstructed = eqx.filter_jit(
        lambda plan, coefficients: plan.synthesis(coefficients)
    )(transform, short_coefficients)
    mapped = jax.vmap(
        lambda values: transform.synthesis(transform.analysis(values))
    )(jnp.stack((short, -short)))
    gradient = jax.grad(
        lambda values: jnp.sum(transform.synthesis(transform.analysis(values)) ** 2)
    )(short)
    long_coefficients = transform.analysis(long)

    assert jnp.allclose(short_reconstructed, short, rtol=1e-11, atol=1e-11)
    assert jnp.allclose(mapped, jnp.stack((short, -short)), rtol=1e-11, atol=1e-11)
    assert jnp.allclose(gradient, 2.0 * short, rtol=1e-10, atol=1e-10)
    assert transform.synthesis(long_coefficients).shape == long.shape
    assert short_coefficients.transform_fingerprint == transform.fingerprint
    assert long_coefficients.transform_fingerprint == transform.fingerprint


def test_alpert_multiwavelet_rejects_invalid_configuration_and_coefficients():
    with pytest.raises(ValueError, match="positive"):
        AlpertMultiwaveletTransform(order=0)
    with pytest.raises(ValueError, match="positive"):
        AlpertMultiwaveletTransform(levels=0)
    with pytest.raises(ValueError, match="boundary"):
        AlpertMultiwaveletTransform(boundary="reflect")

    transform = AlpertMultiwaveletTransform(order=2, levels=2)
    with pytest.raises(ValueError, match="point and channel axes"):
        transform.analysis(jnp.ones((8,)))
    with pytest.raises(ValueError, match="at least two"):
        transform.analysis(jnp.ones((1, 2)))

    coefficients = transform.analysis(jnp.ones((9, 2)))
    incompatible = AlpertMultiwaveletTransform(order=3, levels=2)
    with pytest.raises(ValueError, match="different transform"):
        incompatible.synthesis(coefficients)
