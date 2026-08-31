#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax
import jax.numpy as jnp
import jax.random as jr
import pytest

from phydrax.velocimetry.imaging import ImageGeometry2D, ImagePair2D
from phydrax.velocimetry.piv import (
    accumulate_ensemble,
    ensemble_correlation,
    initialize_ensemble,
    piv,
    PIVPassPlan,
    PIVPlan,
    residual_disparity,
)


def _translated_pair():
    first = jr.normal(jr.key(12), (48, 48))
    second = jnp.zeros_like(first)
    second = second.at[2:, :-1].set(first[:-2, 1:])
    geometry = ImageGeometry2D(first.shape)
    return first, second, geometry, ImagePair2D(first, second, geometry, delta_t=0.01)


def _plan(*, retain_correlation=True, resource_limit_bytes=64 * 1024 * 1024):
    return PIVPlan(
        (PIVPassPlan(16, 8, 4),),
        correlation_mode="extended",
        minimum_valid_fraction=0.5,
        minimum_peak_ratio=0.0,
        minimum_correlation=-1.0,
        minimum_neighbors=0,
        replacement_iterations=0,
        retain_correlation=retain_correlation,
        chunk_size=3,
        resource_limit_bytes=resource_limit_bytes,
    )


def test_plan_prepare_run_and_one_shot_preserve_result_stages_and_provenance():
    first, second, geometry, pair = _translated_pair()
    plan = _plan()
    prepared = plan.prepare(geometry)
    result = jax.jit(lambda value: prepared.run(value))(pair)
    convenient = piv(first, second, plan, geometry=geometry, delta_t=0.01)

    assert prepared.report.window_counts == (25,)
    assert prepared.report.padded_window_counts == (27,)
    assert prepared.report.maximum_working_bytes <= plan.resource_limit_bytes
    assert prepared.report.requested_compute_dtype == "float32"
    assert prepared.report.resolved_compute_dtype == "float32"
    assert prepared.report.fft_complex_dtype == "complex64"
    assert result.raw.displacement_rc.shape == (5, 5, 2)
    assert str(result.raw.displacement_rc.dtype) == result.resolved_compute_dtype
    assert jnp.allclose(
        jnp.median(result.raw.displacement_rc[result.raw.valid], axis=0),
        jnp.asarray([2.0, -1.0]),
        atol=0.15,
    )
    assert jnp.array_equal(result.raw.displacement_rc, result.validated.displacement_rc)
    assert jnp.array_equal(
        result.validated.displacement_rc, result.replaced.displacement_rc
    )
    assert result.retention.retained
    assert result.retention.correlation.shape == (5, 5, 9, 9)
    assert result.prepared_id == convenient.prepared_id == prepared.prepared_id
    assert result.raw.provenance[:2] == (pair.pair_id, prepared.prepared_id)


def test_symmetric_multipass_deformation_adds_residual_to_previous_prediction():
    _, _, geometry, pair = _translated_pair()
    plan = PIVPlan(
        (
            PIVPassPlan(24, 12, 4),
            PIVPassPlan(16, 8, 2, deformation="symmetric"),
        ),
        minimum_valid_fraction=0.5,
        minimum_peak_ratio=0.0,
        minimum_correlation=-1.0,
        minimum_neighbors=0,
        replacement_iterations=0,
        chunk_size=4,
    )

    result = plan.prepare(geometry).run(pair)

    assert result.raw.displacement_rc.shape == (5, 5, 2)
    assert jnp.allclose(
        jnp.median(result.raw.displacement_rc[result.raw.valid], axis=0),
        jnp.asarray([2.0, -1.0]),
        atol=0.2,
    )
    assert result.raw.provenance[-1] == "symmetric"


def test_ensemble_accumulates_observed_lags_and_disparity_retains_support():
    _, _, geometry, pair = _translated_pair()
    prepared = _plan().prepare(geometry)
    result = prepared.run(pair)
    accumulator = initialize_ensemble(prepared)
    accumulator = accumulate_ensemble(accumulator, result)
    accumulator = accumulate_ensemble(accumulator, result)
    mean = ensemble_correlation(accumulator)
    diagnostics = residual_disparity(pair, result)

    expected = result.retention.correlation.reshape(mean.values.shape)
    assert accumulator.sample_count == 2
    assert jnp.allclose(
        mean.values[jnp.isfinite(mean.values)], expected[jnp.isfinite(expected)]
    )
    assert diagnostics.valid_fraction > 0.7
    assert jnp.isfinite(diagnostics.root_mean_square)
    assert diagnostics.source_field_id == result.replaced.field_id


def test_prepare_rejects_a_plan_that_exceeds_its_declared_resource_limit():
    geometry = ImageGeometry2D((48, 48))
    with pytest.raises(MemoryError, match="resource_limit_bytes"):
        _plan(resource_limit_bytes=1).prepare(geometry)
