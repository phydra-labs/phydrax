# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import jax
import jax.numpy as jnp
import numpy as np

from phydrax.metrix._bosonic_gaussian import BosonicGaussianState
from phydrax.metrix._gaussian_entanglement import (
    gaussian_entropy,
    gaussian_logarithmic_negativity,
    gaussian_ppt_margins,
    GaussianEntanglementPlan,
    GaussianSubsystemPlan,
)


jax.config.update("jax_enable_x64", True)


def _thermal_entropy(occupation):
    return (occupation + 1.0) * np.log(occupation + 1.0) - (
        0.0 if occupation == 0.0 else occupation * np.log(occupation)
    )


def _two_mode_squeezed(squeezing):
    cosine = np.cosh(2.0 * squeezing)
    sine = np.sinh(2.0 * squeezing)
    return 0.5 * np.asarray(
        (
            (cosine, 0.0, sine, 0.0),
            (0.0, cosine, 0.0, -sine),
            (sine, 0.0, cosine, 0.0),
            (0.0, -sine, 0.0, cosine),
        )
    )


def _thermal_loss(covariance, transmissivity, environment_occupation):
    root = np.sqrt(transmissivity)
    x = np.diag((1.0, 1.0, root, root))
    noise = np.zeros((4, 4))
    noise[2:, 2:] = (1.0 - transmissivity) * (environment_occupation + 0.5) * np.eye(2)
    return x @ covariance @ x.T + noise


def test_product_thermal_reduction_entropy_and_ppt_margin():
    occupations = (0.25, 1.5)
    covariance = np.diag(
        (
            occupations[0] + 0.5,
            occupations[0] + 0.5,
            occupations[1] + 0.5,
            occupations[1] + 0.5,
        )
    )
    state = BosonicGaussianState(np.zeros(4), covariance)
    subsystem = GaussianSubsystemPlan(2, (1,)).prepare().select(state)
    report = (
        GaussianEntanglementPlan(2, (0, 1), transposed_modes=(1,))
        .prepare()
        .evaluate(state)
    )

    np.testing.assert_allclose(subsystem.covariance, covariance[2:, 2:])
    np.testing.assert_allclose(
        report.entropy,
        sum(_thermal_entropy(value) for value in occupations),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(report.logarithmic_negativity, 0.0, atol=1e-12)
    assert bool(report.ppt)
    assert bool(report.valid)
    assert float(report.minimum_ppt_margin) >= 0.0


def test_two_mode_squeezed_state_has_exact_ppt_spectrum_and_log_negativity():
    squeezing = 0.73
    covariance = _two_mode_squeezed(squeezing)
    state = BosonicGaussianState(np.zeros(4), covariance)
    report = (
        GaussianEntanglementPlan(2, (0, 1), transposed_modes=(1,))
        .prepare()
        .evaluate(state)
    )
    reduced = GaussianSubsystemPlan(2, (0,)).prepare().select(state)
    entanglement_entropy = _thermal_entropy(np.sinh(squeezing) ** 2)

    np.testing.assert_allclose(report.entropy, 0.0, atol=2e-12)
    np.testing.assert_allclose(
        gaussian_entropy(reduced.covariance),
        entanglement_entropy,
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        report.partial_transpose_symplectic_eigenvalues,
        0.5 * np.exp(np.asarray((-2.0 * squeezing, 2.0 * squeezing))),
        rtol=2e-12,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        report.logarithmic_negativity, 2.0 * squeezing, rtol=2e-12, atol=2e-12
    )
    assert not bool(report.ppt)
    assert float(report.minimum_ppt_margin) < 0.0


def test_thermal_loss_separability_threshold_is_reported_without_clipping():
    covariance = _two_mode_squeezed(0.81)
    environment_occupation = 0.2
    threshold = environment_occupation / (environment_occupation + 1.0)
    below = _thermal_loss(covariance, threshold - 1e-3, environment_occupation)
    at = _thermal_loss(covariance, threshold, environment_occupation)
    above = _thermal_loss(covariance, threshold + 1e-3, environment_occupation)

    assert float(jnp.min(gaussian_ppt_margins(below, (1,)))) > 0.0
    np.testing.assert_allclose(jnp.min(gaussian_ppt_margins(at, (1,))), 0.0, atol=2e-12)
    assert float(jnp.min(gaussian_ppt_margins(above, (1,)))) < 0.0
    np.testing.assert_allclose(
        gaussian_logarithmic_negativity(below, (1,)), 0.0, atol=1e-12
    )
    assert float(gaussian_logarithmic_negativity(above, (1,))) > 0.0
    assert float(gaussian_entropy(above)) > 0.0
