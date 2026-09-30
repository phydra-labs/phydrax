#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from examples.hodge_laplace_mixed import square_complex
from phydrax.exterior import hodge_sector_spectra, HodgeSectorSpectra
from phydrax.kernels import (
    HeatSpectralMultiplier,
    HodgeSpectralKernel,
    MaternSpectralMultiplier,
)
from phydrax.linalg import codifferential
from phydrax.uq import (
    ExactGaussianProcessDiscrepancy,
    ExactGaussianProcessFactor,
    FiniteFeatureGaussianProcessFactor,
    GaussianProcessLikelihoodState,
)


pytestmark = [pytest.mark.strict_jax, pytest.mark.filterwarnings("error")]


def _kernel(spectra: HodgeSectorSpectra) -> HodgeSpectralKernel:
    return HodgeSpectralKernel(
        spectra,
        exact_multiplier=HeatSpectralMultiplier(0.2),
        coexact_multiplier=MaternSpectralMultiplier(0.6, 1.2),
        exact_amplitude=0.8,
        coexact_amplitude=0.7,
    )


def test_fe_hodge_sector_differential_constraints() -> None:
    realization = square_complex(2)
    hilbert = realization.hilbert_complex(boundary="absolute")
    spectra = hodge_sector_spectra(realization, 1)
    exact, coexact = spectra.exact, spectra.coexact
    assert exact is not None and coexact is not None
    assert spectra.total_rank == hilbert.space(1).size
    derivative = hilbert.differential(1)
    delta = codifferential(hilbert, 1)
    closed = jax.vmap(derivative.mv, in_axes=1, out_axes=1)(exact.eigenfunctions)
    coclosed = jax.vmap(delta.mv, in_axes=1, out_axes=1)(coexact.eigenfunctions)
    np.testing.assert_allclose(np.asarray(closed), 0.0, atol=1e-8)
    np.testing.assert_allclose(np.asarray(coclosed), 0.0, atol=1e-8)


def test_fe_hodge_covariance_is_positive_semidefinite() -> None:
    spectra = hodge_sector_spectra(square_complex(2), 1)
    kernel = _kernel(spectra)
    exact = spectra.exact
    assert exact is not None
    coefficient_ids = (
        exact.index_offset + jnp.arange(exact.num_points, dtype=jnp.float64)
    )[:, None]
    matrix = np.asarray(kernel.matrix(coefficient_ids, coefficient_ids))
    np.testing.assert_allclose(matrix, matrix.T, atol=1e-10)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    np.testing.assert_allclose(
        np.asarray(kernel.diagonal(coefficient_ids)), np.diag(matrix), atol=1e-10
    )


def test_fe_hodge_feature_likelihood_matches_dense_covariance() -> None:
    spectra = hodge_sector_spectra(square_complex(2), 1)
    exact = spectra.exact
    assert exact is not None
    coefficient_ids = jnp.tile(
        exact.index_offset + jnp.arange(exact.num_points, dtype=jnp.float64), 2
    )
    kernel = _kernel(spectra)
    model = ExactGaussianProcessDiscrepancy(
        coefficient_ids, jnp.zeros_like(coefficient_ids)
    )
    state = GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)
    factor = model.factor(state=state)
    dense = ExactGaussianProcessFactor(coefficient_ids, state=state)
    residual = jnp.sin(coefficient_ids)
    assert isinstance(factor, FiniteFeatureGaussianProcessFactor)
    np.testing.assert_allclose(
        np.asarray(factor.log_probability(residual)),
        np.asarray(dense.log_probability(residual)),
        atol=1e-8,
    )
