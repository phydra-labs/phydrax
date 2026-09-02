#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from typing import Any

import coordax as cx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

import phydrax as phx


TERMINATION = phx.optim.OptimizationTermination(
    absolute_optimality=1e-8,
    relative_optimality=0.0,
    maximum_steps=100,
)


def _array(value):
    return value.data if isinstance(value, cx.Field) else value


def test_named_calibration_preserves_mass_support_ancestry_and_provenance():
    axis = "sample"
    coordinates = jnp.linspace(0.0, 1.0, 21)
    samples = cx.Field(coordinates, dims=(axis,))
    log_weights = cx.Field(jnp.linspace(-1.0, 1.0, 21), dims=(axis,))
    mask = cx.Field(jnp.arange(21) != 7, dims=(axis,))
    ancestry = cx.Field(jnp.arange(100, 121, dtype=jnp.int32), dims=(axis,))
    features = cx.Field(coordinates[:, None], dims=(axis, "moment"))
    target = phx.integration.weighted(
        samples,
        log_weights,
        normalized=False,
        target_mass=jnp.asarray(5.0),
        mask=mask,
        ancestry=ancestry,
        support_valid=jnp.asarray(True),
        sample_axes=axis,
        provenance="observed-ensemble",
    )
    source = phx.integration.materialize(target)

    calibrated = phx.integration.calibrate(
        source,
        phx.weighting.ExactMoments(jnp.array([0.6])),
        features=features,
        termination=TERMINATION,
    )
    estimate = phx.integration.reduce(lambda value: value, calibrated)

    assert jnp.isclose(calibrated.target.target_mass, 5.0)
    assert jnp.allclose(calibrated.batch.samples.data, coordinates)
    assert jnp.array_equal(calibrated.batch.mask.data, mask.data)
    assert calibrated.batch.log_weights.data[7] == -jnp.inf
    assert jnp.array_equal(calibrated.batch.ancestry_ids.data, ancestry.data)
    assert jnp.array_equal(calibrated.batch.support_valid, jnp.asarray(True))
    assert jnp.allclose(estimate.value.data, 3.0, atol=1e-8)
    assert estimate.error_estimate is None
    assert estimate.provenance.method == "calibrated"
    assert isinstance(
        estimate.diagnostics,
        phx.integration.TransformedIntegrationDiagnostics,
    )
    assert tuple(record.kind for record in calibrated.transformations) == ("calibration",)
    diagnostics = calibrated.transformations[0].diagnostics
    assert isinstance(diagnostics, phx.integration.MeasureCalibrationDiagnostics)
    assert diagnostics.calibration.successful
    assert jnp.allclose(
        diagnostics.calibration.achieved_moments,
        jnp.array([0.6]),
        atol=1e-8,
    )
    assert diagnostics.source_provenance == "observed-ensemble"


def test_raw_callable_features_calibrate_normalized_expectations():
    samples = jnp.linspace(-1.0, 1.0, 31)
    source = phx.integration.materialize(
        phx.integration.weighted(
            samples,
            -0.5 * samples**2,
            normalized=True,
        )
    )

    calibrated = phx.integration.calibrate(
        source,
        phx.weighting.ExactMoments(jnp.array([0.2, 0.5])),
        features=lambda values: jnp.stack((values, values**2), axis=1),
        termination=TERMINATION,
    )
    first = phx.integration.reduce(lambda values: values, calibrated)
    second = phx.integration.reduce(lambda values: values**2, calibrated)

    assert calibrated.target.target_mass is None
    assert jnp.allclose(_array(first.value), 0.2, atol=1e-8)
    assert jnp.allclose(_array(second.value), 0.5, atol=1e-8)


def test_calibration_then_recombination_preserves_calibrated_moments_and_history():
    samples = jnp.linspace(-1.0, 1.0, 65)
    features = jnp.stack((samples, samples**2), axis=1)
    source = phx.integration.materialize(
        phx.integration.weighted(
            samples,
            jnp.zeros_like(samples),
            normalized=False,
            target_mass=jnp.asarray(4.0),
            ancestry=jnp.arange(1000, 1065, dtype=jnp.int32),
        )
    )
    calibrated = phx.integration.calibrate(
        source,
        phx.weighting.ExactMoments(jnp.array([0.25, 0.45])),
        features=features,
        termination=TERMINATION,
    )

    compressed = phx.integration.compress(
        calibrated,
        phx.coresets.MomentRecombination(),
        features=features,
    )
    first = phx.integration.reduce(lambda values: values, compressed)
    second = phx.integration.reduce(lambda values: values**2, compressed)

    assert compressed.batch.num_samples <= 3
    assert jnp.isclose(compressed.target.target_mass, 4.0)
    assert jnp.allclose(_array(first.value), 1.0, atol=1e-8)
    assert jnp.allclose(_array(second.value), 1.8, atol=1e-8)
    assert tuple(record.kind for record in compressed.transformations) == (
        "calibration",
        "compression",
    )
    assert first.provenance.method == "compressed"
    assert tuple(record.kind for record in first.diagnostics.transformations) == (
        "calibration",
        "compression",
    )


def test_recombination_then_calibration_uses_the_compressed_prior_and_order():
    samples = jnp.linspace(0.0, 1.0, 41)
    source = phx.integration.materialize(
        phx.integration.weighted(samples, jnp.zeros_like(samples))
    )
    compressed = phx.integration.compress(
        source,
        phx.coresets.MomentRecombination(),
        features=jnp.stack((samples, samples**2), axis=1),
    )
    compressed_samples = jnp.asarray(compressed.batch.samples)
    compressed_logits = jnp.asarray(compressed.batch.log_weights)
    expected_weights = jax.nn.softmax(compressed_logits + 0.7 * compressed_samples)
    target_moment = jnp.sum(expected_weights * compressed_samples)

    calibrated = phx.integration.calibrate(
        compressed,
        phx.weighting.ExactMoments(jnp.reshape(target_moment, (1,))),
        features=compressed_samples,
        termination=TERMINATION,
    )
    estimate = phx.integration.reduce(lambda values: values, calibrated)

    assert calibrated.batch.num_samples == compressed.batch.num_samples
    assert jnp.allclose(_array(estimate.value), target_moment, atol=1e-8)
    assert tuple(record.kind for record in calibrated.transformations) == (
        "compression",
        "calibration",
    )
    assert estimate.provenance.method == "calibrated"


def test_soft_calibration_returns_a_finite_measure_for_unreachable_targets():
    samples = jnp.array([0.0, 1.0, 2.0])
    source = phx.integration.materialize(
        phx.integration.weighted(samples, jnp.log(jnp.array([0.2, 0.5, 0.3])))
    )

    calibrated = phx.integration.calibrate(
        source,
        phx.weighting.QuadraticMoments(
            jnp.array([-3.0]), covariance=jnp.asarray([[0.25]])
        ),
        features=samples,
        termination=TERMINATION,
    )
    result = calibrated.transformations[0].diagnostics.calibration

    assert result.successful
    assert jnp.all(jnp.isfinite(result.weights))
    assert 0.0 <= result.achieved_moments[0] <= 2.0
    assert jnp.isclose(jnp.sum(result.weights), 1.0)


def test_calibration_rejects_unpreserved_grouping_and_failed_exact_targets():
    samples = jnp.linspace(0.0, 1.0, 12)
    identifiers = jnp.arange(12, dtype=jnp.int32) // 2
    for identifier in ("stratum_ids", "pair_ids", "replicate_ids"):
        grouping: dict[str, Any] = {identifier: identifiers}
        grouped = phx.integration.materialize(
            phx.integration.weighted(
                samples,
                jnp.zeros_like(samples),
                **grouping,
            )
        )
        with pytest.raises(ValueError, match="transformed"):
            phx.integration.calibrate(
                grouped,
                phx.weighting.ExactMoments(jnp.array([0.5])),
                features=samples,
            )

    source = phx.integration.materialize(
        phx.integration.weighted(samples, jnp.zeros_like(samples))
    )
    with pytest.raises(eqx.EquinoxRuntimeError, match="did not converge"):
        phx.integration.calibrate(
            source,
            phx.weighting.ExactMoments(jnp.array([2.0])),
            features=samples,
        )
