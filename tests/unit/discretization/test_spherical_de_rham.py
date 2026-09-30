#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.discretization.spectral import SphericalDeRhamComplex, SphericalSpectralPlan


def _complex() -> SphericalDeRhamComplex:
    return SphericalDeRhamComplex(
        SphericalSpectralPlan(4, sampling="gl").prepare(radius=2.3)
    )


def test_spherical_spin_realization_matches_analytic_gradient_and_curl() -> None:
    realization = _complex()
    space = realization.space
    theta, _ = jnp.meshgrid(space.transform.theta, space.transform.phi, indexing="ij")
    scalar = jnp.cos(theta)
    coordinates = realization.from_physical(0, scalar)
    np.testing.assert_allclose(
        eqx.filter_jit(realization.to_physical)(0, coordinates), scalar, atol=2e-11
    )
    gradient = realization.exterior_derivative(0, coordinates)
    physical = realization.to_physical(1, gradient)
    np.testing.assert_allclose(physical[..., 0], 0.0, atol=2e-11)
    np.testing.assert_allclose(
        physical[..., 1], jnp.sin(theta) / space.radius, atol=2e-11
    )
    np.testing.assert_allclose(
        realization.from_physical(1, physical), gradient, atol=2e-11
    )
    np.testing.assert_allclose(
        realization.exterior_derivative(1, gradient), 0.0, atol=1e-13
    )
    rotated = realization.metric_star(1, gradient)
    density = realization.exterior_derivative(1, rotated)
    np.testing.assert_allclose(
        realization.to_physical(2, density), -2.0 / space.radius**2 * scalar, atol=2e-11
    )
    np.testing.assert_allclose(
        realization.to_physical(1, rotated),
        jnp.stack((-physical[..., 1], physical[..., 0]), axis=-1),
        atol=2e-11,
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_spherical_all_degree_metric_duality_and_betti_numbers(degree: int) -> None:
    realization = _complex()
    hilbert = realization.hilbert_complex()
    space = hilbert.space(degree)
    value = jnp.sin(0.2 * jnp.arange(space.size, dtype=jnp.float64))
    scalar_spectrum = (
        np.repeat(np.arange(4) * (np.arange(4) + 1), 2 * np.arange(4) + 1)
        / realization.space.radius**2
    )
    expected = np.repeat(scalar_spectrum[1:], 2) if degree == 1 else scalar_spectrum
    np.testing.assert_allclose(
        realization.hodge_laplacian(degree, value), expected * value, atol=2e-13
    )
    assert np.count_nonzero(expected == 0.0) == (1, 0, 1)[degree]
    assert realization.betti_numbers == (1, 0, 1)
    basis = realization.harmonic_basis(degree)
    np.testing.assert_allclose(
        realization.space.radius**2 * basis.T @ basis,
        np.eye((1, 0, 1)[degree]),
        atol=1e-13,
    )
    dual = realization.hodge_star(degree, value)
    np.testing.assert_allclose(dual, realization.space.radius**2 * value, atol=1e-13)
    np.testing.assert_allclose(
        realization.inverse_hodge_star(degree, dual), value, atol=1e-13
    )
    np.testing.assert_allclose(
        realization.metric_star(2 - degree, realization.metric_star(degree, value)),
        (-1) ** (degree * (2 - degree)) * value,
        atol=1e-13,
    )


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_spherical_decomposition_has_true_harmonic_sectors(degree: int) -> None:
    realization = _complex()
    space = realization.hilbert_complex().space(degree)
    value = jnp.cos(0.3 * jnp.arange(space.size, dtype=jnp.float64))
    result = eqx.filter_jit(realization.hodge_decomposition)(degree, value)
    assert bool(result.valid)
    np.testing.assert_allclose(
        result.exact + result.coexact + result.harmonic, value, atol=1e-13
    )
    np.testing.assert_allclose(space.inner(result.exact, result.coexact), 0.0, atol=1e-13)
    np.testing.assert_allclose(
        realization.hodge_laplacian(degree, result.harmonic), 0.0, atol=1e-13
    )
    if degree > 0:
        assert result.exact_potential is not None
        np.testing.assert_allclose(
            realization.exterior_derivative(degree - 1, result.exact_potential),
            result.exact,
            atol=1e-13,
        )
    if degree == 1:
        np.testing.assert_allclose(result.harmonic, 0.0, atol=0.0)


@pytest.mark.parametrize("degree", [1, 2])
def test_spherical_codifferential_is_positive_hilbert_adjoint(degree: int) -> None:
    realization = _complex()
    hilbert = realization.hilbert_complex()
    left = jnp.sin(jnp.arange(hilbert.space(degree - 1).size, dtype=jnp.float64))
    right = jnp.cos(jnp.arange(hilbert.space(degree).size, dtype=jnp.float64))
    derivative = realization.exterior_derivative(degree - 1, left)
    adjoint = realization.codifferential(degree, right)
    np.testing.assert_allclose(
        hilbert.space(degree).inner(derivative, right),
        hilbert.space(degree - 1).inner(left, adjoint),
        atol=1e-12,
    )
    with pytest.raises(ValueError, match="relative"):
        realization.hilbert_complex(boundary="relative")
