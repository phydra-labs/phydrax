#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

import phydrax as phx
from phydrax.applications.radiation_biophysics import circulating_blood as cb


def _two_state(*, forward=2.0, backward=3.0):
    model = cb.CirculatingBloodModel(
        (
            cb.BloodCompartment("central", 1.0),
            cb.BloodCompartment("peripheral", 1.0),
        ),
        (
            cb.BloodFlow("central", "peripheral", forward),
            cb.BloodFlow("peripheral", "central", backward),
        ),
    )
    return cb.prepare_circulating_blood_model(model)


def _quantity(reference=cb.ABSORBED_DOSE_RATE_REFERENCE):
    return cb.circulating_blood_dose_rate_quantity("blood_dose_rate", reference)


def _schedule(
    intervals,
    *,
    rates=(1.0, 2.0),
    uncertainty=(0.1, 0.2),
    reference=cb.ABSORBED_DOSE_RATE_REFERENCE,
):
    quantity = _quantity(reference)
    return cb.PiecewiseConstantDoseRateSchedule(
        tuple(
            cb.DoseRateInterval(
                start,
                end,
                quantity,
                np.asarray(rates, dtype="float64"),
                None if uncertainty is None else np.asarray(uncertainty, dtype="float64"),
            )
            for start, end in intervals
        )
    )


def test_two_state_transition_and_exact_occupation_match_analytic_solution():
    forward = 2.0
    backward = 3.0
    duration = 0.7
    prepared = _two_state(forward=forward, backward=backward)
    decay = np.exp(-(forward + backward) * duration)
    expected_transition = np.asarray(
        (
            (
                backward / (forward + backward) + forward / (forward + backward) * decay,
                forward / (forward + backward) * (1.0 - decay),
            ),
            (
                backward / (forward + backward) * (1.0 - decay),
                forward / (forward + backward) + backward / (forward + backward) * decay,
            ),
        )
    )
    np.testing.assert_allclose(
        prepared.generator.transition_matrix(duration),
        expected_transition,
        rtol=1e-6,
        atol=1e-7,
    )
    result = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, duration),), rates=(0.0, 0.0)),
        np.asarray((1.0, 0.0)),
        t0_s=0.0,
        t1_s=duration,
    )
    expected_peripheral_occupation = (
        forward / (forward + backward) * (duration - (1.0 - decay) / (forward + backward))
    )
    np.testing.assert_allclose(
        result.final_probabilities,
        expected_transition[0],
        rtol=1e-6,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        result.occupation_seconds,
        (duration - expected_peripheral_occupation, expected_peripheral_occupation),
        rtol=1e-6,
        atol=1e-7,
    )


def test_interval_splitting_boundaries_and_gaps_preserve_exact_reward():
    prepared = _two_state()
    initial = np.asarray((0.25, 0.75))
    whole = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 2.0),), rates=(1.5, 1.5)),
        initial,
        t0_s=0.0,
        t1_s=2.0,
    )
    split = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 0.7), (0.7, 2.0)), rates=(1.5, 1.5)),
        initial,
        t0_s=0.0,
        t1_s=2.0,
    )
    np.testing.assert_allclose(split.final_probabilities, whole.final_probabilities)
    np.testing.assert_allclose(split.occupation_seconds, whole.occupation_seconds)
    np.testing.assert_allclose(split.total_dose_gy, whole.total_dose_gy)

    gaps = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 0.5), (1.0, 1.5)), rates=(2.0, 2.0)),
        initial,
        t0_s=0.0,
        t1_s=2.0,
    )
    assert float(gaps.total_dose_gy) == pytest.approx(2.0, rel=1e-6)
    assert float(jnp.sum(gaps.occupation_seconds)) == pytest.approx(2.0, rel=1e-6)


def test_absorbing_state_is_exact_and_stationary_claim_is_refused():
    prepared = cb.prepare_circulating_blood_model(
        cb.CirculatingBloodModel(
            (
                cb.BloodCompartment("circulating", 2.0),
                cb.BloodCompartment("terminal", 1.0, absorbing=True),
            ),
            (cb.BloodFlow("circulating", "terminal", 1.0),),
        )
    )
    transition = np.asarray(prepared.generator.transition_matrix(2.0))
    np.testing.assert_allclose(transition[0], (np.exp(-1.0), 1.0 - np.exp(-1.0)))
    np.testing.assert_allclose(transition[1], (0.0, 1.0))
    with pytest.raises(ValueError, match="irreducible"):
        prepared.stationary_distribution()


def test_same_realization_replays_and_constant_dose_is_path_invariant():
    prepared = _two_state(forward=0.5, backward=0.25)
    schedule = _schedule(((0.0, 3.0),), rates=(0.4, 0.4))
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(2026),
        prepared.process.num_channels,
        support=(0.0, 3.0),
        max_events_per_channel=64,
        sample_shape=(64,),
        process_id=prepared.process.process_id,
    )
    first = cb.simulate_circulating_blood_dose(
        prepared,
        schedule,
        realization,
        "central",
        t0_s=0.0,
        t1_s=3.0,
    )
    replay = cb.simulate_circulating_blood_dose(
        prepared,
        schedule,
        realization,
        "central",
        t0_s=0.0,
        t1_s=3.0,
    )
    assert bool(jnp.all(first.successful))
    assert jnp.array_equal(first.solution.events.valid, replay.solution.events.valid)
    assert jnp.array_equal(
        first.solution.events.times,
        replay.solution.events.times,
        equal_nan=True,
    )
    assert jnp.array_equal(first.total_dose_gy, replay.total_dose_gy)
    np.testing.assert_allclose(first.total_dose_gy, np.full((64,), 1.2), atol=1e-6)
    np.testing.assert_allclose(
        np.sum(np.asarray(first.occupation_seconds), axis=-1),
        np.full((64,), 3.0),
        atol=1e-6,
    )


def test_preparation_and_history_capacity_fail_closed_with_evidence():
    with pytest.raises(ValueError, match="compartment capacity exceeded"):
        cb.prepare_circulating_blood_model(
            _two_state().model,
            maximum_compartments=1,
        )
    prepared = _two_state(forward=100.0, backward=100.0)
    realization = phx.stochastic.PoissonClockRealization(
        jr.key(17),
        prepared.process.num_channels,
        support=(0.0, 1.0),
        max_events_per_channel=2,
        sample_shape=(32,),
        process_id=prepared.process.process_id,
    )
    result = cb.simulate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 1.0),), rates=(1.0, 2.0)),
        realization,
        "central",
        t0_s=0.0,
        t1_s=1.0,
        max_events=1,
    )
    assert bool(jnp.all(result.capacity.capacity_exceeded))
    assert not bool(jnp.any(result.successful))
    assert bool(jnp.all(jnp.isnan(result.total_dose_gy)))


def test_dose_rate_meanings_and_unknown_uncertainty_are_not_collapsed():
    prepared = _two_state()
    known = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 1.0),), uncertainty=(0.1, 0.2)),
        (1.0, 0.0),
        t0_s=0.0,
        t1_s=1.0,
    )
    unknown = cb.integrate_circulating_blood_dose(
        prepared,
        _schedule(((0.0, 1.0),), uncertainty=None),
        (1.0, 0.0),
        t0_s=0.0,
        t1_s=1.0,
    )
    assert known.standard_uncertainty_gy is not None
    assert float(known.standard_uncertainty_gy) > 0.0
    assert unknown.standard_uncertainty_gy is None
    wrong_kind = phx.measurement.resolve_radiation_quantity(
        "absorbed_dose",
        phx.measurement.RadiationQuantityKind.ABSORBED_DOSE,
        phx.units.GRAY,
        support_association=cb.CIRCULATING_BLOOD_DOSE_RATE_SUPPORT,
        reference_configuration=cb.ABSORBED_DOSE_RATE_REFERENCE,
    )
    with pytest.raises(ValueError, match="Dose-rate quantity"):
        cb.DoseRateInterval(0.0, 1.0, wrong_kind, np.ones(2))
    with pytest.raises(ValueError, match="absorbed dose"):
        cb.circulating_blood_dose_rate_quantity("rate", "generic-blood-dose")


def _manifest(name):
    payload = name.encode()
    return phx.qualification.ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="public",
        nondimensionalization={"value": 1.0},
        uncertainty={"standard": 0.1},
        lineage_ids=("synthetic",),
    )


def _affine(*, offset=0.0):
    matrix = np.eye(4)
    matrix[0, 3] = offset
    return phx.imaging.ImageIndexAffine(
        matrix,
        "voxel-index",
        phx.SpatialCoordinateContract(
            phx.units.MILLIMETER,
            coordinate_system="cartesian-lps",
            reference_frame="synthetic-subject",
        ),
        phx.imaging.ImageAxisConvention.LPS,
    )


def _image(asset_id, values, spec, affine, *, uncertainty=None):
    return phx.imaging.MedicalImageAsset(
        asset_id,
        "synthetic",
        np.asarray(values),
        affine,
        spec,
        phx.imaging.DeidentificationEvidence(
            f"{asset_id}-deid", "subject-0", "synthetic", True, True, True
        ),
        (_manifest(f"{asset_id}-source"),),
        phx.measurement.DerivationRecord(
            phx.measurement.DataOrigin.SYNTHETIC,
            phx.measurement.DataStage.RECONSTRUCTED,
            transformation_id="circulating-blood-test",
        ),
        uncertainty=uncertainty,
    )


def _spatial_inputs(*, dose_affine=None, uncertainty=True):
    affine = _affine()
    dose_quantity = _quantity()
    dose_spec = phx.imaging.ImageFieldSpec(
        dose_quantity,
        phx.measurement.ValueLayout.scalar(),
        phx.measurement.SamplingSemantics(
            phx.measurement.SpatialSamplingKind.CELL_AVERAGE
        ),
    )
    values = np.asarray([1.0, 3.0]).reshape((2, 1, 1))
    sigma = (
        phx.measurement.IndependentStandardUncertainty(
            np.full(values.shape, 0.2), phx.units.GRAY_PER_SECOND
        )
        if uncertainty
        else None
    )
    dose = _image(
        "dose-rate",
        values,
        dose_spec,
        affine if dose_affine is None else dose_affine,
        uncertainty=sigma,
    )
    label_spec = phx.imaging.ImageFieldSpec.named(
        "segmentation",
        phx.units.ONE,
        phx.measurement.ValueKind.CATEGORICAL,
    )
    label_asset = _image(
        "labels",
        np.asarray([1, 2], dtype=np.int16).reshape((2, 1, 1)),
        label_spec,
        affine,
    )
    labels = phx.imaging.LabelVolume(
        label_asset,
        phx.imaging.LabelOntology(
            "blood-labels",
            "synthetic",
            "1",
            (
                phx.imaging.LabelDefinition(1, "central-label", "Central"),
                phx.imaging.LabelDefinition(2, "peripheral-label", "Peripheral"),
            ),
        ),
    )
    weights = {
        "central-label": {"central": 1.0, "peripheral": 0.0},
        "peripheral-label": {"central": 0.25, "peripheral": 0.75},
    }
    return dose, labels, weights


def test_spatial_compartment_mixture_requires_exact_affine_and_normalized_weights():
    dose, labels, weights = _spatial_inputs()
    prepared = cb.prepare_spatial_compartment_mixture(
        dose, labels, ("central", "peripheral"), weights
    )
    np.testing.assert_allclose(prepared.dose_rates_gy_per_s, (1.4, 3.0))
    assert prepared.standard_uncertainties_gy_per_s is not None
    assert prepared.interval(0.0, 1.0).quantity.compatible_with(dose.quantity)

    shifted, _, _ = _spatial_inputs(dose_affine=_affine(offset=0.1))
    with pytest.raises(ValueError, match="exact spatial affine"):
        cb.prepare_spatial_compartment_mixture(
            shifted, labels, ("central", "peripheral"), weights
        )
    invalid = {
        **weights,
        "central-label": {"central": 0.8, "peripheral": 0.3},
    }
    with pytest.raises(ValueError, match="normalized"):
        cb.prepare_spatial_compartment_mixture(
            dose, labels, ("central", "peripheral"), invalid
        )
    negative = {
        **weights,
        "central-label": {"central": 1.1, "peripheral": -0.1},
    }
    with pytest.raises(ValueError, match="nonnegative"):
        cb.prepare_spatial_compartment_mixture(
            dose, labels, ("central", "peripheral"), negative
        )


def test_spatial_mixture_preserves_unknown_uncertainty():
    dose, labels, weights = _spatial_inputs(uncertainty=False)
    prepared = cb.prepare_spatial_compartment_mixture(
        dose, labels, ("central", "peripheral"), weights
    )
    assert prepared.standard_uncertainties_gy_per_s is None
    assert prepared.interval(0.0, 1.0).standard_uncertainties_gy_per_s is None
