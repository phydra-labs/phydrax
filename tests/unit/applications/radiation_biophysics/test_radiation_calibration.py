#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#
"""Synthetic inference checks, explicitly not experimental qualification."""

import hashlib
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications import radiation_biophysics as rad
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import derived_unit, ELECTRONVOLT, JOULE, UnitDefinition


def reference(*, training=True, uncertainty=0.01):
    payload = b"hand-authored-known-Gaussian-sigma-radiation-regression"
    return ReferenceArtifactManifest(
        "synthetic-yield-law",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="LicenseRef-PHYDRA-synthetic-test",
        commercial_use_permitted=False,
        redistribution_permitted=False,
        training_use_permitted=training,
        export_permitted=False,
        export_classification="test-only",
        nondimensionalization={"yield_scale": 1.0},
        uncertainty=None
        if uncertainty is None
        else {"declared_synthetic_gaussian_sigma": uncertainty},
        lineage_ids=("synthetic-regression-not-experiment",),
    )


def dataset(prefix, pairs, *, oxygen_start=0.0, sigma=0.01):
    supports = tuple(
        rad.LesionExpectationSupport(
            tuple(d), tuple(i), 1.0, f"{prefix}-candidates-{row}"
        )
        for row, (d, i) in enumerate(pairs)
    )
    measured = tuple(
        sum(
            1 - 0.7**d * 0.4**i
            for d, i in zip(s.direct_multiplicity, s.indirect_multiplicity, strict=True)
        )
        for s in supports
    )
    return rad.RadiationCalibrationData(
        tuple(f"{prefix}-observation-{i}" for i in range(len(supports))),
        tuple(
            rad.RadiationCondition(
                f"{prefix}-condition-{i}", 1.0, oxygen_start + i, 0.0, 1e-9
            )
            for i in range(len(supports))
        ),
        supports,
        measured,
        (sigma,) * len(supports),
        derived_unit("Gy^-1", ((rad.GRAY, -1),)),
        "per-Gy",
        reference(uncertainty=sigma),
        "synthetic",
    )


def test_union_probability_is_not_sum_and_masked_support_is_jit_differentiable():
    direct = jnp.asarray([[1, 0, 99]])
    indirect = jnp.asarray([[2, 1, 99]])
    mask = jnp.asarray([[True, True, False]])
    logits = jnp.log(jnp.asarray([0.3, 0.6]) / jnp.asarray([0.7, 0.4]))
    fn = lambda theta: rad.expected_initial_lesion_yield(
        theta, direct, indirect, mask, jnp.asarray([1.0])
    )[0]
    assert float(jax.jit(fn)(logits)) == pytest.approx(0.888 + 0.6)
    gradient = jax.jit(jax.grad(fn))(logits)
    np.testing.assert_allclose(
        gradient, [0.3 * 0.7 * 0.4**2, 2 * 0.7 * 0.4 * 0.6 * 0.4 + 0.6 * 0.4], rtol=1e-6
    )


def test_native_probability_fit_predicts_independent_heldout_conditions():
    training = dataset("fit", [((1,), (0,)), ((0,), (1,)), ((1, 1), (0, 1))])
    heldout = dataset("validation", [((1,), (2,)), ((0, 1), (3, 2))], oxygen_start=10.0)
    result = rad.calibrate_radiation_lesions(
        training,
        heldout,
        initial_logits=jnp.zeros(2),
        prior_mean=jnp.zeros(2),
        prior_standard_deviation=jnp.full(2, 3.0),
    )
    np.testing.assert_allclose(
        result.heldout_predictions, heldout.observed_yields, atol=2e-3
    )
    assert result.likelihood_rank == 2
    assert np.all(np.asarray(result.heldout_parameter_variance) > 0)
    assert not result.scientifically_qualified
    assert any("chemical-G" in gate for gate in result.gates)
    assert any("experimental" in gate for gate in result.gates)
    with pytest.raises(ValueError, match="independent withheld"):
        rad.calibrate_radiation_lesions(
            training,
            training,
            initial_logits=jnp.zeros(2),
            prior_mean=jnp.zeros(2),
            prior_standard_deviation=jnp.ones(2),
        )


def test_prior_does_not_manufacture_likelihood_identifiability():
    training = dataset("fit", [((1,), (1,)), ((1,), (1,))])
    heldout = dataset("validation", [((1,), (1,))], oxygen_start=10.0)
    result = rad.calibrate_radiation_lesions(
        training,
        heldout,
        initial_logits=jnp.zeros(2),
        prior_mean=jnp.zeros(2),
        prior_standard_deviation=jnp.ones(2),
    )
    assert result.likelihood_rank == 1
    probabilities = np.asarray(result.probabilities)
    likelihood_gradient = (
        np.prod(1 - probabilities) * probabilities / training.standard_errors[0]
    )
    np.testing.assert_allclose(
        result.likelihood_singular_values,
        (np.sqrt(2) * np.linalg.norm(likelihood_gradient), 0.0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert any("prior-constrained" in gate for gate in result.gates)
    assert np.all(np.isfinite(np.asarray(result.posterior.covariance)))
    assert np.all(np.asarray(result.heldout_parameter_variance) > 0)
    assert np.all(np.isfinite(np.asarray(result.heldout_standardized_residuals)))


def test_unknown_uncertainty_and_training_rights_are_separate_gates():
    training = dataset("fit", [((1,), (0,)), ((0,), (1,))])
    heldout = dataset("validation", [((1,), (2,))], oxygen_start=10.0)
    with pytest.raises(ValueError, match="uncertainty"):
        replace(training, reference=reference(uncertainty=None))
    restricted = replace(training, reference=reference(training=False))
    with pytest.raises(PermissionError, match="training"):
        rad.calibrate_radiation_lesions(
            restricted,
            heldout,
            initial_logits=jnp.zeros(2),
            prior_mean=jnp.zeros(2),
            prior_standard_deviation=jnp.ones(2),
        )


def test_stage_evidence_compares_observables_against_declared_uncertainty():
    evidence = rad.RadiationStageEvidence(
        "chemical-G",
        ("1ps", "1ns"),
        (3.0, 2.0),
        (3.1, 2.2),
        (0.1, 0.1),
        derived_unit("eV^-1", ((ELECTRONVOLT, -1),)),
        reference(),
        "synthetic",
        2.0,
        ("synthetic-complete-reaction-ledger",),
    )
    assert evidence.standardized_rms == pytest.approx(2.5**0.5)
    assert evidence.accepted
    assert not replace(evidence, predicted=(10.0, 10.0)).accepted


def test_expected_lesion_yield_respects_measured_unit_scale():
    data = dataset("units", [((1,), (2,))])
    unit = UnitDefinition(
        "mGy^-1", data.yield_unit.dimension, data.yield_unit.reference_system_id, "1000"
    )
    scaled = replace(
        data,
        yield_unit=unit,
        observed_yields=tuple(value / 1000 for value in data.observed_yields),
        standard_errors=tuple(value / 1000 for value in data.standard_errors),
    )
    logits = jnp.log(jnp.asarray([0.3, 0.6]) / jnp.asarray([0.7, 0.4]))
    prediction = rad.expected_initial_lesion_yield(logits, *scaled.prepared_arrays())
    np.testing.assert_allclose(prediction, scaled.observed_yields, rtol=1e-6)
    with pytest.raises(ValueError, match="dimensions"):
        replace(data, yield_unit=JOULE)
