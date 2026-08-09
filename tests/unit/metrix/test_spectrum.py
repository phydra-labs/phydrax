#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest
from jaxtyping import ArrayLike

import phydrax as phx
from phydrax._trainable import partition_trainable


def _basis(
    *,
    eigenvalues: ArrayLike | None = None,
    eigenfunctions: ArrayLike | None = None,
    probability_measure: ArrayLike | None = None,
    active_mask: ArrayLike | None = None,
):
    return phx.metrix.DiscreteLaplacianEigenbasis(
        jnp.asarray([0.0, 2.0]) if eigenvalues is None else eigenvalues,
        (
            jnp.asarray([[1.0, 1.0], [1.0, -1.0]])
            if eigenfunctions is None
            else eigenfunctions
        ),
        (jnp.asarray([0.5, 0.5]) if probability_measure is None else probability_measure),
        spectral_dimension=1.0,
        basis_id="two-point",
        active_mask=active_mask,
    )


def test_discrete_laplacian_eigenbasis_validates_probability_orthonormality():
    basis = _basis()

    assert basis.mode_count == 2
    assert basis.entity_count == 2
    assert basis.zero_mode_count == 1
    assert basis.report.exact
    assert jnp.allclose(
        basis.eigenfunctions.T
        @ (basis.probability_measure[:, None] * basis.eigenfunctions),
        jnp.eye(2),
    )


def test_discrete_laplacian_eigenbasis_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="sorted"):
        _basis(eigenvalues=jnp.asarray([2.0, 0.0]))
    with pytest.raises(ValueError, match="materially negative"):
        _basis(eigenvalues=jnp.asarray([-0.1, 2.0]))
    with pytest.raises(ValueError, match="sum to one"):
        _basis(probability_measure=jnp.asarray([0.4, 0.4]))
    with pytest.raises(ValueError, match="orthonormal"):
        _basis(eigenfunctions=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="Inactive eigenfunction"):
        _basis(
            active_mask=jnp.asarray([True, False]),
            probability_measure=jnp.asarray([1.0, 0.0]),
        )


def test_probability_measure_tolerance_is_absolute():
    measure = jnp.asarray([0.5, 0.499995])
    functions = jnp.ones((2, 1)) / jnp.sqrt(jnp.sum(measure))
    with pytest.raises(ValueError, match="sum to one"):
        phx.metrix.DiscreteLaplacianEigenbasis(
            jnp.asarray([0.0]),
            functions,
            measure,
            spectral_dimension=1.0,
            basis_id="unnormalized-measure",
        )


def test_spectrum_reports_reject_invalid_or_inconsistent_provenance():
    with pytest.raises(ValueError, match="next_eigenvalue"):
        phx.metrix.LaplacianEigenbasisReport(
            method_id="test",
            source_id="test-source",
            requested_modes=2,
            retained_modes=2,
            active_dimension=2,
            zero_mode_count=1,
            canonicalized_zero_count=0,
            exact=True,
            tail_certified=True,
            next_eigenvalue=float("nan"),
            boundary_gap=float("inf"),
            orthonormality_residual=0.0,
        )

    inconsistent = phx.metrix.LaplacianEigenbasisReport(
        method_id="test",
        source_id="test-source",
        requested_modes=2,
        retained_modes=2,
        active_dimension=2,
        zero_mode_count=0,
        canonicalized_zero_count=0,
        exact=True,
        tail_certified=True,
        next_eigenvalue=float("inf"),
        boundary_gap=float("inf"),
        orthonormality_residual=0.0,
    )
    with pytest.raises(ValueError, match="zero_mode_count"):
        phx.metrix.DiscreteLaplacianEigenbasis(
            jnp.asarray([0.0, 2.0]),
            jnp.asarray([[1.0, 1.0], [1.0, -1.0]]),
            jnp.asarray([0.5, 0.5]),
            spectral_dimension=1.0,
            basis_id="inconsistent-report",
            report=inconsistent,
        )


def test_discrete_spectrum_is_fixed_while_multiplier_parameters_are_trainable():
    basis = _basis()
    kernel = phx.kernels.SpectralFeatureKernel(
        basis,
        phx.kernels.MaternSpectralMultiplier(0.7, 1.5),
    )

    trainable, fixed = partition_trainable(kernel)

    assert trainable.eigenbasis is None
    assert fixed.eigenbasis is basis
    assert trainable.multiplier.length_scale is not None
    assert trainable.multiplier.smoothness is not None
