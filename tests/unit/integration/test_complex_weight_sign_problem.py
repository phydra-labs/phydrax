#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

import jax.numpy as jnp

from phydrax.integration._complex_weight import (
    complex_weight_measure,
    ComplexWeightStatus,
    phase_quenched_reweight,
    PhaseQuenchedReweightingPlan,
    prepare_phase_quenched_reweighting,
)


def test_phase_quenched_reweighting_matches_exact_finite_complex_measure():
    samples = jnp.asarray([[0.0], [1.0], [2.0]])
    weights = jnp.asarray([1.0 + 1.0j, 1.0 - 1.0j, 2.0 + 0.0j])
    observable = jnp.asarray([0.0, 2.0, 1.0])
    measure = complex_weight_measure(samples, weights, source_id="three-point-exact")
    prepared = prepare_phase_quenched_reweighting(
        measure,
        PhaseQuenchedReweightingPlan(
            minimum_average_phase=0.1,
            minimum_effective_sample_size=2.0,
            maximum_samples=8,
        ),
    )
    result = phase_quenched_reweight(prepared, observable)
    expected = jnp.sum(weights * observable) / jnp.sum(weights)

    assert result.status == int(ComplexWeightStatus.SUCCESS)
    assert result.successful
    assert jnp.allclose(result.value, expected, rtol=1e-6, atol=1e-6)
    assert jnp.isfinite(result.uncertainty.standard_error)
    assert jnp.isfinite(result.uncertainty.denominator_standard_error)
    assert result.overlap.average_phase_magnitude > 0.5


def test_phase_quenched_reweighting_abstains_when_average_phase_vanishes():
    measure = complex_weight_measure(
        jnp.asarray([[0.0], [1.0], [2.0], [3.0]]),
        jnp.asarray([1.0 + 0.0j, -1.0 + 0.0j, 1.0 + 0.0j, -1.0 + 0.0j]),
        source_id="zero-overlap",
    )
    prepared = prepare_phase_quenched_reweighting(
        measure,
        PhaseQuenchedReweightingPlan(
            minimum_average_phase=0.05,
            minimum_effective_sample_size=2.0,
            maximum_samples=8,
        ),
    )
    result = phase_quenched_reweight(prepared, jnp.arange(4.0))

    assert prepared.diagnostics.status == int(ComplexWeightStatus.INSUFFICIENT_OVERLAP)
    assert result.status == int(ComplexWeightStatus.INSUFFICIENT_OVERLAP)
    assert not result.successful
    assert jnp.isnan(jnp.real(result.value))
