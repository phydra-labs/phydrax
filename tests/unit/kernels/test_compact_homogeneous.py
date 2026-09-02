#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp
import pytest

import phydrax as phx


def _rotation(angle):
    return jnp.asarray(
        [[jnp.cos(angle), -jnp.sin(angle)], [jnp.sin(angle), jnp.cos(angle)]]
    )


def _so2_spectrum():
    modes = jnp.arange(5)

    def zonal(left, right):
        relative = left.T @ right
        angle = jnp.arctan2(relative[1, 0], relative[0, 0])
        return jnp.cos(modes * angle)

    return phx.kernels.PreparedCompactHomogeneousSpectrum(
        modes[:, None],
        modes.astype(float) ** 2,
        jnp.asarray([1.0, 2.0, 2.0, 2.0, 2.0]),
        zonal,
        space="so",
        tail_bound=1e-5,
        tail_certified=True,
        spectrum_id="so2-modes-0-4",
    )


def test_compact_heat_and_matern_kernels_have_psd_grams_and_tail_evidence():
    spectrum = _so2_spectrum()
    points = jnp.stack(tuple(_rotation(angle) for angle in (0.0, 0.3, 0.8, 1.4)))
    for kernel in (
        phx.kernels.CompactHomogeneousHeatKernel(spectrum, time=0.4),
        phx.kernels.CompactHomogeneousMaternKernel(
            spectrum,
            smoothness=1.5,
            inverse_length_squared=0.7,
            spectral_dimension=1.0,
        ),
    ):
        gram = kernel.matrix(points, points)
        assert jnp.allclose(gram, gram.T, atol=1e-6)
        assert jnp.min(jnp.linalg.eigvalsh(gram)) > -1e-5
        assert bool(kernel.evidence(gram).positive_definite_capability)
        assert kernel.evidence(gram).truncation_tail_bound == spectrum.tail_bound
        assert jnp.allclose(kernel.diagonal(points), 1.0, atol=1e-6)


def test_geodesic_exponential_exposes_branch_evidence_and_no_default_psd_claim():
    kernel = phx.kernels.GeodesicExponentialKernel(space="so", length_scale=0.8)
    left = _rotation(0.2)
    right = _rotation(0.7)
    evidence = kernel.distance_evidence(left, right)
    assert bool(evidence.valid)
    assert jnp.allclose(evidence.distance, 0.5, atol=1e-6)
    assert kernel.pairwise(left, right) < 1.0
    with pytest.raises(ValueError, match="positive-definiteness"):
        kernel.require_positive_definite()

    cut = kernel.distance_evidence(_rotation(0.0), _rotation(jnp.pi))
    assert not bool(cut.branch_valid)
    with pytest.raises(Exception):
        kernel.pairwise(_rotation(0.0), _rotation(jnp.pi))


@pytest.mark.parametrize("residual", [0.0, 5e-5, 1e-4])
def test_stiefel_log_accepts_finite_residual_at_or_below_tolerance(residual):
    point = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def stiefel_log(left, right):
        return right - left, jnp.asarray(residual)

    kernel = phx.kernels.GeodesicExponentialKernel(
        space="stiefel",
        length_scale=0.8,
        branch_tolerance=1e-4,
        stiefel_log=stiefel_log,
    )
    evidence = kernel.distance_evidence(point, point)
    assert bool(evidence.branch_valid)
    assert bool(evidence.valid)
    assert jnp.allclose(kernel.pairwise(point, point), 1.0)


@pytest.mark.parametrize("residual", [-1e-8, 1.01e-4, jnp.nan, jnp.inf])
def test_stiefel_log_rejects_invalid_or_nonfinite_residual(residual):
    point = jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])

    def stiefel_log(left, right):
        return right - left, jnp.asarray(residual)

    kernel = phx.kernels.GeodesicExponentialKernel(
        space="stiefel",
        length_scale=0.8,
        branch_tolerance=1e-4,
        stiefel_log=stiefel_log,
    )
    evidence = kernel.distance_evidence(point, point)
    assert not bool(evidence.branch_valid)
    assert not bool(evidence.valid)
    with pytest.raises(Exception):
        kernel.pairwise(point, point)


def test_uncertified_spectral_tail_fails_preparation():
    with pytest.raises(ValueError, match="certified truncation tail"):
        phx.kernels.PreparedCompactHomogeneousSpectrum(
            jnp.asarray([[0]]),
            jnp.asarray([0.0]),
            jnp.asarray([1.0]),
            lambda left, right: jnp.asarray([1.0]),
            space="grassmann",
            tail_bound=0.0,
            tail_certified=False,
            spectrum_id="uncertified",
        )
