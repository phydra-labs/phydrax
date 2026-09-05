#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.linalg import expm

from phydrax.applications.systems_biology import (
    CountMeasurementPlan,
    TelegraphGeneExpressionPlan,
)
from phydrax.applications.systems_biology.single_cell import (
    CellIdentity,
    fit_stationary_counts,
    GeneIdentity,
    generate_transcripts,
    import_transcript_arrays,
    import_velocity_field,
    observe_transcripts,
    PiecewiseConstantRates,
    predict_transcript_velocity,
    predicted_count_moments,
    ScenarioExecutionError,
    ScenarioSegment,
    scheduled_transcript_mean,
    StationaryCountTarget,
    TranscriptCountAssay,
    TranscriptCounts,
    TranscriptScenario,
    transient_transcript_mean,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import derived_unit, MILLISECOND, SECOND


RATE_UNIT = derived_unit("per-second", ((SECOND, -1),))
RATES = np.asarray([2.0, 3.0, 12.0, 4.0, 1.5])


def _source(*, quantified=True, training=True):
    content = b"Independently specified synthetic telegraph/count law; exact input coefficients, not experimental data."
    return ReferenceArtifactManifest(
        "synthetic-transcript-law",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"time_seconds": 1.0, "transcript_count": 1.0},
        uncertainty={"synthetic_input_coefficient_error": 0.0} if quantified else None,
        lineage_ids=("independently-declared-analytic-fixture",),
    )


def _assay(capture=0.6, background=0.2):
    return TranscriptCountAssay.from_plans(
        CountMeasurementPlan(capture, background, observation_capacity=1024),
        CountMeasurementPlan(capture, background, observation_capacity=1024),
        _source(),
    )


def _scenario(*, cells=(7, 91), genes=(23,), branches=False, event_capacity=256):
    rates = np.broadcast_to(RATES, (2, len(genes), 5)).copy()
    rates[1, :, 2] = 20.0
    schedule = PiecewiseConstantRates((0.0, 0.125, 0.5), rates, rate_unit=RATE_UNIT)
    segments = [ScenarioSegment(101, schedule, (0.0, 0.2, 0.5))]
    if branches:
        child = PiecewiseConstantRates((0.5, 0.8), rates[1:], rate_unit=RATE_UNIT)
        segments.extend(
            (
                ScenarioSegment(103, child, (0.5, 0.8), 101),
                ScenarioSegment(107, child, (0.5, 0.8), 101),
            )
        )
    initial = np.zeros((len(cells), len(genes), 3))
    initial[..., 1:] = (10, 15)
    return TranscriptScenario(
        tuple(CellIdentity(i, f"cell-{i}") for i in cells),
        tuple(GeneIdentity(i, f"gene-{i}") for i in genes),
        tuple(segments),
        initial,
        max_paths=len(cells) * len(genes) * len(segments),
        max_events_per_interval=event_capacity,
    )


def _reference_mean(rates, initial, time):
    a, b, alpha, beta, gamma = rates
    matrix = np.asarray(
        [
            [-a - b, 0.0, 0.0, a],
            [alpha, -beta, 0.0, 0.0],
            [0.0, beta, -gamma, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    return (expm(time * matrix) @ np.r_[initial, 1.0])[:3]


def test_exact_moment_law_schedule_boundary_and_physical_units():
    model = TelegraphGeneExpressionPlan(*RATES).prepare()
    stationary = model.stationary_moments()
    expected = np.asarray([0.4, 1.2, 3.2])
    np.testing.assert_allclose(
        [stationary.promoter_mean, stationary.nascent_mean, stationary.mature_mean],
        expected,
    )
    np.testing.assert_allclose(
        transient_transcript_mean(RATES, expected, 0.7), expected, rtol=2e-5, atol=2e-6
    )
    rates = np.stack((RATES, RATES * np.asarray([1.0, 1.0, 5.0, 1.0, 1.0])))[:, None, :]
    schedule = PiecewiseConstantRates((0.0, 0.125, 1.0), rates, rate_unit=RATE_UNIT)
    predicted = scheduled_transcript_mean(schedule, jnp.zeros(3))
    middle = _reference_mean(RATES, np.zeros(3), 0.125)
    terminal = _reference_mean(rates[1, 0], middle, 0.875)
    np.testing.assert_allclose(
        predicted, np.stack((np.zeros(3), middle, terminal)), rtol=3e-5, atol=3e-6
    )
    milliseconds = PiecewiseConstantRates(
        (0.0, 125.0, 1000.0), rates, rate_unit=RATE_UNIT, time_unit=MILLISECOND
    )
    np.testing.assert_allclose(
        scheduled_transcript_mean(milliseconds, jnp.zeros(3)),
        predicted,
        rtol=3e-5,
        atol=3e-6,
    )
    assert (
        abs(float(predicted[-1, 1]) - _reference_mean(RATES, np.zeros(3), 1.0)[1]) > 1.0
    )
    repeated = schedule.repeat(2)
    assert repeated.boundaries == (0.0, 0.125, 1.0, 1.125, 2.0)
    with pytest.raises(ValueError, match="positive"):
        PiecewiseConstantRates((0.0, 1.0), np.zeros((1, 1, 5)), rate_unit=RATE_UNIT)


def test_exact_event_ledger_invariants_boundary_drift_and_workset_replay():
    scenario = _scenario(genes=(23, 24))
    full = generate_transcripts(scenario, jax.random.key(19))
    subset = generate_transcripts(
        scenario, jax.random.key(19), cell_ids=(91,), gene_ids=(24,)
    )
    selected = next(
        p for p in full.paths if p.cell.cell_id == 91 and p.gene.gene_id == 24
    )
    np.testing.assert_array_equal(selected.latent.values, subset.paths[0].latent.values)
    for path in full.paths:
        values = np.asarray(path.latent.values)
        assert np.all((values[:, 0] == 0) | (values[:, 0] == 1))
        assert np.all(values[:, 1:] >= 0) and np.all(values == np.floor(values))
        assert 0.125 in np.asarray(path.latent.support.coordinates)
        for index, solution in enumerate(path.intervals):
            events = solution.events
            times = np.asarray(events.times)[np.asarray(events.valid)]
            assert np.all(times >= path.segment.schedule.boundaries[index])
            assert np.all(times <= path.segment.schedule.boundaries[index + 1])
            pre = np.asarray(events.pre_states)[np.asarray(events.valid)]
            post = np.asarray(events.post_states)[np.asarray(events.valid)]
            np.testing.assert_array_equal(np.sum(pre[:, :2], axis=-1), np.ones(len(pre)))
            np.testing.assert_array_equal(
                np.sum(post[:, :2], axis=-1), np.ones(len(post))
            )
            assert np.all(post >= 0)
        np.testing.assert_allclose(
            path.conditional_drift.values, 4.0 * values[:, 1] - 1.5 * values[:, 2]
        )
    for first, replay in zip(selected.intervals, subset.paths[0].intervals, strict=True):
        np.testing.assert_array_equal(first.events.valid, replay.events.valid)
        np.testing.assert_array_equal(first.events.times, replay.events.times)
        np.testing.assert_array_equal(first.events.channels, replay.events.channels)


def test_scenario_forks_are_resets_not_division_or_lag_pairs():
    scenario = _scenario(cells=(7,), branches=True)
    experiment = generate_transcripts(scenario, jax.random.key(20))
    parent, first, second = experiment.paths
    np.testing.assert_array_equal(first.latent.values[0], parent.latent.values[-1])
    np.testing.assert_array_equal(second.latent.values[0], parent.latent.values[-1])
    joined = experiment.joined_series(7, 23)
    edges = np.asarray(joined.support.edge_valid)
    resets = (
        parent.latent.support.capacity - 1,
        parent.latent.support.capacity + first.latent.support.capacity - 1,
    )
    assert np.count_nonzero(~edges) == 2
    assert not edges[resets[0]] and not edges[resets[1]]
    assert np.all(np.diff(np.asarray(joined.support.coordinates))[edges] > 0)


def test_exhausted_event_capacity_cannot_seed_descendants():
    rates = RATES.copy()
    rates[2] = 1e6
    schedule = PiecewiseConstantRates(
        (0.0, 1.0), rates[None, None, :], rate_unit=RATE_UNIT
    )
    scenario = TranscriptScenario(
        (CellIdentity(7, "cell"),),
        (GeneIdentity(23, "gene"),),
        (ScenarioSegment(1, schedule, (0.0, 1.0)),),
        np.asarray([[[1.0, 0.0, 0.0]]]),
        max_paths=1,
        max_events_per_interval=1,
    )
    with pytest.raises(ScenarioExecutionError) as caught:
        generate_transcripts(scenario, jax.random.key(9))
    assert not bool(caught.value.solution.successful)


def test_observation_is_separate_and_identity_capture_is_exact():
    scenario = _scenario()
    experiment = generate_transcripts(scenario, jax.random.key(51))
    identity = _assay(1.0, 0.0)
    counts = observe_transcripts(
        experiment, identity, jax.random.key(51), gene_id=23, segment_id=101
    )
    np.testing.assert_array_equal(
        counts.counts, jnp.stack(tuple(p.latent.values[-1, 1:] for p in experiment.paths))
    )
    noisy_assay = _assay()
    first = observe_transcripts(
        experiment, noisy_assay, jax.random.key(22), gene_id=23, segment_id=101
    )
    second = observe_transcripts(
        experiment, noisy_assay, jax.random.key(23), gene_id=23, segment_id=101
    )
    assert not np.array_equal(first.counts, second.counts)
    subset_experiment = generate_transcripts(scenario, jax.random.key(51), cell_ids=(91,))
    subset = observe_transcripts(
        subset_experiment, noisy_assay, jax.random.key(22), gene_id=23, segment_id=101
    )
    np.testing.assert_array_equal(subset.counts[0], first.counts[1])
    assert not np.any(counts.to_series().support.edge_valid)
    with pytest.raises(ValueError, match="uncertainty"):
        TranscriptCountAssay(
            identity.unspliced, identity.spliced, _source(quantified=False)
        )


def _measured_target(assay, offset=0):
    # Fixed independently measured synthetic snapshots; no latent promoter or
    # privileged true velocity is available to the fitting API.
    observations = TranscriptCounts(
        GeneIdentity(23, "gene-23"),
        tuple(range(offset, offset + 12)),
        np.asarray(
            [
                [0, 1],
                [1, 2],
                [2, 3],
                [1, 5],
                [0, 4],
                [3, 2],
                [1, 4],
                [2, 6],
                [0, 2],
                [2, 5],
                [1, 3],
                [1, 4],
            ]
        ),
        coordinate_semantics="none",
        assay_id=assay.assay_id,
        source_id=f"synthetic-snapshots-{offset}",
        preprocessing_id="raw-counts",
    )
    return StationaryCountTarget.from_counts(
        observations,
        np.ones(5),
        equilibrium_evidence_id="declared-stationary-synthetic-profile",
    )


def test_measured_count_fit_predict_and_unidentifiable_clock():
    assay = _assay(1.0, 0.0)
    target = _measured_target(assay)
    initial = RATES.copy()
    initial[2] = 3.0
    fixed = {0: RATES[0], 1: RATES[1], 3: RATES[3], 4: RATES[4]}
    fit = fit_stationary_counts(
        target, assay, initial, fixed_rates=fixed, rate_calibration=_source()
    )
    assert bool(fit.result.successful)
    assert fit.identifiability.locally_identifiable
    start = predicted_count_moments(fit.model, initial, assay)
    assert float(
        jnp.sum((fit.predict_count_moments() - target.target.moments) ** 2)
    ) < float(jnp.sum((start - target.target.moments) ** 2))
    held_out = _measured_target(assay, 100)
    assert np.max(np.abs(fit.held_out_residuals(held_out))) < 3.0
    assert (
        fit.count_prediction_uq is not None and fit.free_log_rate_covariance is not None
    )
    velocity = predict_transcript_velocity(fit, held_out.observations)
    assert np.any(np.asarray(velocity.estimates) > 0) and np.any(
        np.asarray(velocity.estimates) < 0
    )
    with pytest.raises(ValueError, match="disjoint"):
        fit.held_out_residuals(target)
    unconstrained = fit_stationary_counts(target, assay, RATES)
    assert unconstrained.identifiability.rank < 5
    assert unconstrained.free_log_rate_covariance is None
    with pytest.raises(ValueError, match="clock"):
        predict_transcript_velocity(unconstrained, held_out.observations)
    np.testing.assert_allclose(
        predicted_count_moments(fit.model, RATES * 7.0, assay),
        predicted_count_moments(fit.model, RATES, assay),
        rtol=2e-6,
    )
    gradient = jax.jit(
        jax.grad(lambda rates: jnp.sum(predicted_count_moments(fit.model, rates, assay)))
    )(jnp.asarray(RATES))
    assert np.all(np.isfinite(gradient))


def test_import_preserves_missingness_pseudotime_estimator_and_rights():
    imported = import_transcript_arrays(
        [1.0, np.nan],
        [2.0, 3.0],
        gene=GeneIdentity(1, "reporter"),
        cell_ids=(9, 10),
        source=_source(quantified=False),
        assay_id="external-calibration",
        preprocessing_id="raw-layers",
        coordinate_semantics="pseudotime",
        coordinates=(0.8, 0.2),
        valid=np.asarray([[True, True], [False, True]]),
    )
    assert np.isnan(imported.raw_unspliced[1])
    assert not bool(imported.counts.valid[1, 0])
    assert not np.any(imported.counts.to_series().support.edge_valid)
    external = import_velocity_field(
        [[1.0, 2.0], [-1.0, 4.0]],
        imported.counts,
        source=_source(quantified=False),
        estimator_id="external-estimator-revision",
        preprocessing_id="moments-pca",
        representation_id="two-dimensional-embedding",
        uncertainty_id="unreported",
    )
    assert external.standard_errors is None
    assert external.observations.time_unit is None
    with pytest.raises(ValueError, match="physical-time"):
        TranscriptCounts(
            GeneIdentity(1, "reporter"),
            (9,),
            [[1, 2]],
            coordinate_semantics="pseudotime",
            coordinates=[0.5],
            time_unit=SECOND,
            assay_id="a",
            source_id="s",
            preprocessing_id="p",
        )
    with pytest.raises(ValueError):
        import_transcript_arrays(
            [1.5],
            [2],
            gene=GeneIdentity(1, "reporter"),
            cell_ids=(9,),
            source=_source(),
            assay_id="a",
            preprocessing_id="normalized",
            coordinate_semantics="none",
        )
    with pytest.raises(PermissionError):
        import_transcript_arrays(
            [1],
            [2],
            gene=GeneIdentity(1, "reporter"),
            cell_ids=(9,),
            source=_source(training=False),
            assay_id="a",
            preprocessing_id="raw",
            coordinate_semantics="none",
            training_use=True,
        )
