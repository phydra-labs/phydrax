#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import phydrax as phx


def _hyperbolic_points():
    radii = jnp.asarray([0.0, 0.35, 0.7])
    return jnp.stack(
        (
            jnp.cosh(radii),
            jnp.sinh(radii),
            jnp.zeros_like(radii),
        ),
        axis=-1,
    )


def _spd_points():
    return jnp.asarray(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.5, 0.1], [0.1, 0.8]],
            [[0.7, -0.05], [-0.05, 1.3]],
        ]
    )


def _assert_psd(matrix, tolerance=1e-9):
    assert np.min(np.linalg.eigvalsh(np.asarray(matrix))) >= -tolerance


def test_hyperbolic_random_features_are_fixed_psd_and_differentiable():
    proposal = phx.kernels.hyperbolic_feature_proposal(
        jax.random.key(3), 2, 128, proposal_scale=1.0
    )
    points = _hyperbolic_points()

    def objective(length_scale, smoothness):
        kernel = phx.kernels.HyperbolicRandomFeatureKernel(
            proposal, length_scale, smoothness
        )
        return jnp.sum(kernel.matrix(points, points))

    kernel = phx.kernels.HyperbolicRandomFeatureKernel(proposal, 0.8, 1.4)
    first = kernel.matrix(points, points)
    second = kernel.matrix(points, points)
    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(0.8), jnp.asarray(1.4)
    )

    assert jnp.array_equal(first, second)
    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))
    _assert_psd(first)
    with pytest.raises(Exception, match="future unit sheet"):
        kernel.matrix(jnp.asarray([[1.0, 1.0, 0.0]]), points)
    with pytest.raises(ValueError, match="one hyperbolic point"):
        kernel.pairwise(points[:2], points[0])


def test_spd_random_features_are_fixed_psd_and_differentiable():
    proposal = phx.kernels.spd_feature_proposal(
        jax.random.key(5), 2, 128, proposal_scale=1.0
    )
    points = _spd_points()

    def objective(length_scale, smoothness):
        kernel = phx.kernels.SPDRandomFeatureKernel(proposal, length_scale, smoothness)
        return jnp.sum(kernel.matrix(points, points))

    kernel = phx.kernels.SPDRandomFeatureKernel(proposal, 0.9, 1.3)
    matrix = kernel.matrix(points, points)
    value, gradients = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        jnp.asarray(0.9), jnp.asarray(1.3)
    )

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(jnp.asarray(gradients)))
    assert jnp.allclose(matrix, matrix.T)
    _assert_psd(matrix)
    with pytest.raises(Exception, match="positive-definite"):
        kernel.matrix(jnp.asarray([[[1.0, 2.0], [2.0, 1.0]]]), points)
    with pytest.raises(ValueError, match="one SPD point"):
        kernel.pairwise(points[:2], points[0])


def test_spd_plane_wave_uses_affine_metric_log_eigenvalue_coordinates():
    frequency = jnp.asarray([[0.3, -0.1]])
    proposal = phx.kernels.NoncompactFeatureProposal(
        frequency,
        jnp.eye(2)[None, :, :],
        jnp.zeros((1,)),
        jnp.zeros((1,)),
        geometry_id="spd-SPD2",
        proposal_id="analytic-spd-coordinate-check",
        proposal_scale=1.0,
    )
    kernel = phx.kernels.SPDRandomFeatureKernel(proposal, 0.9, 1.3)
    log_diagonal = jnp.asarray([0.4, -0.2])
    points = jnp.stack((jnp.eye(2), jnp.diag(jnp.exp(log_diagonal))))
    features = kernel.features(points)
    rho = jnp.asarray([0.25, -0.25])
    expected_ratio = jnp.exp(-jnp.dot(rho, log_diagonal)) * jnp.cos(
        jnp.dot(frequency[0], log_diagonal)
    )

    assert jnp.allclose(features[1, 0] / features[0, 0], expected_ratio)


def test_importance_diagnostics_report_weight_degeneracy_and_uncertainty():
    hyperbolic = phx.kernels.HyperbolicRandomFeatureKernel(
        phx.kernels.hyperbolic_feature_proposal(jax.random.key(7), 2, 256),
        0.8,
        1.5,
    )
    spd = phx.kernels.SPDRandomFeatureKernel(
        phx.kernels.spd_feature_proposal(jax.random.key(9), 2, 256),
        0.9,
        1.2,
    )

    for kernel in (hyperbolic, spd):
        report = kernel.importance_diagnostics()
        assert jnp.allclose(jnp.sum(report.normalized_weights), 1.0)
        assert 1.0 <= report.effective_sample_size <= report.sample_count
        assert 0.0 < report.maximum_normalized_weight <= 1.0
        assert jnp.isfinite(report.normalizer_estimate)
        assert jnp.isfinite(report.monte_carlo_standard_error)
        assert report.proposal_id == kernel.proposal.proposal_id
        assert report.finite_importance_variance


def test_importance_diagnostics_are_stable_and_use_unbiased_standard_error():
    report = phx.kernels.ImportanceFeatureDiagnostics(
        jnp.log(jnp.asarray([1.0, 3.0])),
        "two-weights",
    )

    assert report.normalizer_estimate == pytest.approx(2.0)
    assert report.monte_carlo_standard_error == pytest.approx(1.0)
    assert report.effective_sample_size == pytest.approx(1.6)

    extreme_logs = jnp.full((128,), -jnp.inf).at[0].set(712.0)
    extreme = phx.kernels.ImportanceFeatureDiagnostics(extreme_logs, "extreme")
    expected = jnp.exp(jnp.asarray(712.0) - jnp.log(128.0))

    assert jnp.isfinite(extreme.normalizer_estimate)
    assert jnp.isfinite(extreme.monte_carlo_standard_error)
    assert jnp.allclose(extreme.normalizer_estimate, expected)
    assert jnp.allclose(extreme.monte_carlo_standard_error, expected)


def test_importance_diagnostics_expose_unavailable_or_infinite_variance():
    singleton = phx.kernels.ImportanceFeatureDiagnostics(jnp.asarray([0.0]), "one")
    infinite_variance = phx.kernels.ImportanceFeatureDiagnostics(
        jnp.log(jnp.asarray([1.0, 3.0])),
        "heavy-tail",
        finite_importance_variance=False,
    )

    assert jnp.isinf(singleton.monte_carlo_standard_error)
    assert not infinite_variance.finite_importance_variance
    assert jnp.isinf(infinite_variance.monte_carlo_standard_error)
    with pytest.raises(Exception, match="finite, nonzero total mass"):
        phx.kernels.ImportanceFeatureDiagnostics(
            jnp.asarray([-jnp.inf, -jnp.inf]),
            "zero-mass",
        )


def test_noncompact_matern_diagnostics_flag_infinite_cauchy_variance():
    proposal = phx.kernels.hyperbolic_feature_proposal(jax.random.key(10), 2, 32)
    report = phx.kernels.HyperbolicRandomFeatureKernel(
        proposal,
        0.8,
        0.25,
    ).importance_diagnostics()

    assert not report.finite_importance_variance
    assert jnp.isinf(report.monte_carlo_standard_error)


def test_resampling_is_explicit_and_fixed_proposal_prefixes_are_nested():
    key = jax.random.key(11)
    proposal = phx.kernels.hyperbolic_feature_proposal(key, 2, 128)
    repeated = phx.kernels.hyperbolic_feature_proposal(key, 2, 128)
    kernel = phx.kernels.HyperbolicRandomFeatureKernel(proposal, 0.7, 1.4)
    resampled = kernel.resample(jax.random.key(12))

    assert jnp.array_equal(proposal.frequencies, repeated.frequencies)
    assert jnp.array_equal(proposal.directions, repeated.directions)
    assert jnp.array_equal(proposal.phases, repeated.phases)
    assert proposal.proposal_id == repeated.proposal_id
    scaled = phx.kernels.hyperbolic_feature_proposal(key, 2, 128, proposal_scale=2.0)
    assert scaled.proposal_id != proposal.proposal_id
    assert proposal.prefix(64).prefix(32).proposal_id == proposal.prefix(32).proposal_id
    assert proposal.prefix(32).proposal_id.endswith("prefix=32")
    assert jnp.array_equal(proposal.prefix(32).frequencies, proposal.frequencies[:32])
    assert resampled.proposal.proposal_id != proposal.proposal_id


def test_fixed_noise_hyperbolic_features_converge_under_nested_rank_growth():
    proposal = phx.kernels.hyperbolic_feature_proposal(
        jax.random.key(21), 2, 2048, proposal_scale=1.0
    )
    points = _hyperbolic_points()

    def covariance(count):
        kernel = phx.kernels.HyperbolicRandomFeatureKernel(
            proposal.prefix(count), 0.8, 1.5
        )
        matrix = kernel.matrix(points, points)
        scale = jnp.mean(jnp.diag(matrix))
        return matrix / scale

    reference = covariance(2048)
    small_error = jnp.linalg.norm(covariance(64) - reference)
    large_error = jnp.linalg.norm(covariance(512) - reference)

    assert large_error < small_error


def test_noncompact_finite_features_reuse_weight_space_gp():
    proposal = phx.kernels.hyperbolic_feature_proposal(jax.random.key(31), 2, 16)
    kernel = phx.kernels.HyperbolicRandomFeatureKernel(proposal, 0.8, 1.5)
    points = jnp.tile(_hyperbolic_points(), (8, 1))
    model = phx.uq.ExactGaussianProcessDiscrepancy(
        points,
        jnp.zeros((points.shape[0],)),
    )
    state = phx.uq.GaussianProcessLikelihoodState(kernel=kernel, noise_scale=0.1)

    assert isinstance(
        model.factor(state=state), phx.uq.FiniteFeatureGaussianProcessFactor
    )
