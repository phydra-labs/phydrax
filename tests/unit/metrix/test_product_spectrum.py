#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#


from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _two_point_basis(decomposition_id: Any = "two-point") -> Any:
    return phx.discretization.SpectralDecomposition(
        jnp.asarray([0.0, 2.0]),
        jnp.asarray([[1.0, 1.0], [1.0, -1.0]]),
        jnp.asarray([0.5, 0.5]),
        spectral_dimension=1.0,
        decomposition_id=decomposition_id,
    )


def test_product_spectrum_scenario_1() -> None:
    first = _two_point_basis("first")
    second = _two_point_basis("second")
    product = phx.metrix.product_laplacian_eigenbasis(
        (first, second),
        num_modes=None,
    )

    assert jnp.allclose(product.eigenvalues, jnp.asarray([0.0, 2.0, 2.0, 4.0]))
    assert jnp.allclose(product.probability_measure, 0.25)
    assert product.spectral_dimension == 2.0
    # ty: ignore[unresolved-attribute]
    assert product.report.exact
    assert jnp.allclose(
        product.eigenfunctions.T
        @ (product.probability_measure[:, None] * product.eigenfunctions),
        jnp.eye(4),
    )
    factors = (_two_point_basis("first"), _two_point_basis("second"))

    with pytest.raises(ValueError, match="degenerate product eigenspace"):
        phx.metrix.product_laplacian_eigenbasis(factors, num_modes=2)

    product = phx.metrix.product_laplacian_eigenbasis(factors, num_modes=3)
    assert jnp.allclose(product.eigenvalues, jnp.asarray([0.0, 2.0, 2.0]))
    # ty: ignore[unresolved-attribute]
    assert product.report.next_eigenvalue == pytest.approx(4.0)
    # ty: ignore[unresolved-attribute]
    assert not product.report.exact
    report = phx.discretization.LaplacianEigenbasisReport(
        method_id="test-truncation",
        source_id="truncated-factor",
        requested_modes=1,
        retained_modes=1,
        active_dimension=2,
        zero_mode_count=1,
        canonicalized_zero_count=0,
        exact=False,
        tail_certified=True,
        next_eigenvalue=5.0,
        boundary_gap=5.0,
        orthonormality_residual=0.0,
    )
    truncated = phx.discretization.SpectralDecomposition(
        jnp.asarray([0.0]),
        jnp.asarray([[1.0], [1.0]]),
        jnp.asarray([0.5, 0.5]),
        spectral_dimension=1.0,
        decomposition_id="truncated",
        report=report,
    )
    product = phx.metrix.product_laplacian_eigenbasis(
        (truncated, _two_point_basis()), num_modes=None
    )

    assert jnp.allclose(product.eigenvalues, jnp.asarray([0.0, 2.0]))
    # ty: ignore[unresolved-attribute]
    assert product.report.next_eigenvalue == pytest.approx(5.0)
    # ty: ignore[unresolved-attribute]
    assert not product.report.exact
    product = phx.metrix.product_laplacian_eigenbasis(
        (_two_point_basis("first"), _two_point_basis("second")),
        num_modes=3,
    )
    kernel = phx.kernels.SpectralFeatureKernel(
        product,
        phx.kernels.HeatSpectralMultiplier(0.3),
    )
    observations = jnp.tile(jnp.arange(4), 2)
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        observations,
        jnp.zeros((observations.size,)),
    )
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)

    assert isinstance(
        model.factor(state=state), phx.uq.FiniteFeatureGaussianProcessFactor
    )
    first = _two_point_basis("first")
    second = _two_point_basis("second")
    product_basis = phx.metrix.product_laplacian_eigenbasis(
        (first, second), num_modes=None
    )
    multiplier = phx.kernels.MaternSpectralMultiplier(0.8, 1.4)
    isotropic = phx.kernels.SpectralFeatureKernel(product_basis, multiplier)
    entities = jnp.arange(4)
    isotropic_matrix = isotropic.matrix(entities, entities)

    first_kernel = phx.kernels.SpectralFeatureKernel(first, multiplier)
    second_kernel = phx.kernels.SpectralFeatureKernel(second, multiplier)
    factor_entities = jnp.arange(2)
    separable = jnp.kron(
        first_kernel.matrix(factor_entities, factor_entities),
        second_kernel.matrix(factor_entities, factor_entities),
    )

    assert not jnp.allclose(isotropic_matrix, separable)
    assert np.min(np.linalg.eigvalsh(np.asarray(isotropic_matrix))) >= -1e-10
