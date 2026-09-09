# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Observable pulse/chase contracts; fixtures are not biological validation."""

import hashlib

import numpy as np
import pytest

from phydrax.applications.systems_biology.single_cell import (
    assess_pulse_chase_prediction,
    GeneIdentity,
    import_sceu_seq_arrays,
    LabeledTranscriptAssay,
    LabeledTranscriptCounts,
    pulse_chase_identifiability,
    PulseChasePrediction,
    PulseChaseSchedule,
    sceu_seq_prerequisites,
    scheduled_labeled_transcript_mean,
)
from phydrax.interchange import AdapterStatus
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)
from phydrax.units import derived_unit, SECOND


def _reference(label: str, *, uncertainty=True) -> ReferenceArtifactManifest:
    payload = label.encode()
    return ReferenceArtifactManifest(
        label,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"time_s": 1.0},
        uncertainty={"standard_error": 0.1} if uncertainty else None,
        lineage_ids=("independent-test-fixture",),
    )


def _assay(*, calibration_covariance=True) -> LabeledTranscriptAssay:
    return LabeledTranscriptAssay(
        np.ones(4),
        np.zeros(4),
        np.eye(2),
        labeling_calibration=_reference("label-calibration"),
        count_calibration=_reference("count-calibration"),
        calibration_covariance=(np.eye(12) * 0.01 if calibration_covariance else None),
    )


def _schedule() -> PulseChaseSchedule:
    return PulseChaseSchedule(
        [0.0, 1.0, 2.0],
        np.asarray([[[8.0, 2.0, 1.0, 0.5]], [[8.0, 2.0, 1.0, 0.5]]]),
        [1.0, 0.0],
        rate_unit=derived_unit("s^-1", ((SECOND, -1),)),
    )


def _criterion() -> ScientificMetricCriterion:
    return ScientificMetricCriterion(
        "pulse-chase-culture-macro-standardized-rms",
        "at_most",
        None,
        2.0,
        "1",
        "independent_unit_macro",
    )


def _campaign() -> ScientificCampaign:
    criterion = _criterion()
    cases = (
        ScientificCase(
            "calibration-case",
            "cal-culture",
            "GENE7",
            "time-0",
            "cal-plate",
            "batch-cal",
            ("cal-source-parent",),
        ),
        ScientificCase(
            "locked-case",
            "locked-culture",
            "GENE7",
            "time-2",
            "locked-plate",
            "batch-locked",
            ("locked-source-parent",),
        ),
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("calibration-case",)),
            CampaignRole("locked_evaluation", ("locked-case",)),
        ),
        preprocessing_source_ids=("calibration-case",),
        criteria_ids=(criterion.criterion_id,),
    )


def _claim(
    *,
    frozen_criteria_ids: tuple[str, ...] | None = None,
    support_attributes: dict[str, str] | None = None,
) -> ScientificClaimProfile:
    capability = "single-cell.pulse-chase-rates"
    criterion = _criterion()
    campaign = _campaign()
    return ScientificClaimProfile(
        capability,
        SupportTuple(
            capability,
            {"cell-line": "RPE1"} if support_attributes is None else support_attributes,
        ),
        ("labeled-unlabeled-U-S",),
        ("declared-pulse-chase-schedule",),
        campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "locked-prediction",
            "external-transfer",
        ),
        (criterion,),
        "abstain-without-physical-time-or-identifiable-rate",
        ("model-parameters-change", "observation-law-change"),
        frozen_criteria_ids=(
            campaign.criteria_ids if frozen_criteria_ids is None else frozen_criteria_ids
        ),
    )


def _counts(
    assay: LabeledTranscriptAssay, prefix: str, time: float
) -> LabeledTranscriptCounts:
    return LabeledTranscriptCounts(
        GeneIdentity(7, "GENE7"),
        (1 if prefix == "cal" else 2,),
        [[1, 1, 1, 1]],
        culture_ids=(f"{prefix}-culture",),
        plate_ids=(f"{prefix}-plate",),
        times=[time],
        time_unit=SECOND,
        assay_id=assay.assay_id,
        source_id=f"{prefix}-source",
        preprocessing_id="raw-four-matrix",
        source_parent_ids=(f"{prefix}-source-parent",),
        preprocessing_parent_ids=(f"{prefix}-preprocessing-parent",),
    )


def _prediction(
    assay: LabeledTranscriptAssay,
    schedule: PulseChaseSchedule,
    calibration: LabeledTranscriptCounts,
    *,
    model_id: str = "pulse-chase-model",
) -> PulseChasePrediction:
    return PulseChasePrediction(
        np.ones((1, 4)),
        np.broadcast_to(np.eye(4), (1, 4, 4)),
        model_id=model_id,
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )


def _identifiability(
    assay: LabeledTranscriptAssay,
    schedule: PulseChaseSchedule,
    calibration: LabeledTranscriptCounts,
    *,
    model_id: str = "pulse-chase-model",
):
    return pulse_chase_identifiability(
        np.eye(4),
        ("synthesis", "splicing", "decay", "dilution"),
        model_id=model_id,
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )


def test_physical_schedule_keeps_four_channels_and_does_not_relabel_at_chase():
    per_second = derived_unit("s^-1", ((SECOND, -1),))
    schedule = PulseChaseSchedule(
        [0.0, 1.0, 2.0],
        np.asarray([[[8.0, 2.0, 1.0, 0.5]], [[8.0, 2.0, 1.0, 0.5]]]),
        [1.0, 0.0],
        rate_unit=per_second,
    )

    means = np.asarray(scheduled_labeled_transcript_mean(schedule, np.zeros(4)))

    assert means.shape == (3, 4)
    np.testing.assert_allclose(means[1, 2:], 0.0, atol=1e-7)
    assert means[2, 2] > 0.0
    assert means[2, 0] < means[1, 0]
    assert np.all(means >= 0.0)


def test_four_matrix_adapter_preserves_measured_zero_and_reports_missing_rights_inputs():
    complete = tuple(_reference(f"channel-{index}") for index in range(4))
    report = sceu_seq_prerequisites((complete[0], None, complete[2], complete[3]))
    assert report.missing == ("source-manifest:labeled-spliced",)
    assert not report.ready

    imported = import_sceu_seq_arrays(
        [[0], [2]],
        [[0], [3]],
        [[0], [5]],
        [[0], [7]],
        gene_ids=("GENE7",),
        cell_ids=(11, 12),
        culture_ids=("culture-a", "culture-b"),
        plate_ids=("plate-a", "plate-b"),
        times=[0.0, 2.0],
        time_unit=SECOND,
        manifests=complete,
        preprocessing_id="caller-raw-counts",
    )
    np.testing.assert_array_equal(imported.counts[0, 0], np.zeros(4))
    assert imported.report.status == AdapterStatus.LOSSLESS


def test_assessment_is_inconclusive_without_timing_and_fails_leaked_holdout():
    assay = _assay()
    schedule = _schedule()
    calibration = _counts(assay, "cal", 0.0)
    locked = _counts(assay, "locked", 2.0)
    prediction = _prediction(assay, schedule, calibration)
    identity = _identifiability(assay, schedule, calibration)
    inconclusive = assess_pulse_chase_prediction(
        assay,
        schedule,
        calibration,
        locked,
        prediction,
        identity,
        campaign=_campaign(),
        claim=_claim(),
        claimed_parameter_names=("synthesis",),
        timing_reference=None,
        maximum_standardized_rms=10.0,
    )
    leaked = LabeledTranscriptCounts(
        locked.gene,
        (3,),
        [[1, 1, 1, 1]],
        culture_ids=calibration.culture_ids,
        plate_ids=("locked-plate",),
        times=[2.0],
        time_unit=SECOND,
        assay_id=assay.assay_id,
        source_id="leaked-source",
        preprocessing_id="raw-four-matrix",
        source_parent_ids=("leaked-source-parent",),
        preprocessing_parent_ids=("leaked-preprocessing-parent",),
    )
    with pytest.raises(ValueError, match="campaign locked cases"):
        assess_pulse_chase_prediction(
            assay,
            schedule,
            calibration,
            leaked,
            prediction,
            identity,
            campaign=_campaign(),
            claim=_claim(),
            claimed_parameter_names=("synthesis",),
            timing_reference=_reference("timing"),
            maximum_standardized_rms=10.0,
        )

    assert inconclusive.status == "inconclusive"
    assert "physical-schedule-timing-calibration" in inconclusive.missing_prerequisites


def test_prediction_and_identifiability_are_exactly_fit_bound():
    assay = _assay()
    schedule = _schedule()
    calibration = _counts(assay, "cal", 0.0)
    locked = _counts(assay, "locked", 2.0)
    prediction = _prediction(assay, schedule, calibration)
    unrelated_identity = _identifiability(
        assay, schedule, calibration, model_id="unrelated-model"
    )

    with pytest.raises(TypeError, match="frozen PulseChasePrediction"):
        assess_pulse_chase_prediction(
            assay,
            schedule,
            calibration,
            locked,
            np.ones((1, 4)),
            unrelated_identity,
            campaign=_campaign(),
            claim=_claim(),
            claimed_parameter_names=("synthesis",),
            timing_reference=_reference("timing"),
            maximum_standardized_rms=10.0,
        )
    with pytest.raises(ValueError, match="does not belong"):
        assess_pulse_chase_prediction(
            assay,
            schedule,
            calibration,
            locked,
            prediction,
            unrelated_identity,
            campaign=_campaign(),
            claim=_claim(),
            claimed_parameter_names=("synthesis",),
            timing_reference=_reference("timing"),
            maximum_standardized_rms=10.0,
        )
    with pytest.raises(ValueError, match="frozen criteria"):
        assess_pulse_chase_prediction(
            assay,
            schedule,
            calibration,
            locked,
            prediction,
            _identifiability(assay, schedule, calibration),
            campaign=_campaign(),
            claim=_claim(
                frozen_criteria_ids=(
                    _criterion().criterion_id,
                    "unrelated-frozen-criterion",
                )
            ),
            claimed_parameter_names=("synthesis",),
            timing_reference=_reference("mismatched-criteria-timing"),
            maximum_standardized_rms=10.0,
        )
    with pytest.raises(ValueError, match="exact assessed scope"):
        assess_pulse_chase_prediction(
            assay,
            schedule,
            calibration,
            locked,
            prediction,
            _identifiability(assay, schedule, calibration),
            campaign=_campaign(),
            claim=_claim(support_attributes={"cell-line": "RPE1", "tissue": "retina"}),
            claimed_parameter_names=("synthesis",),
            timing_reference=_reference("broader-support-timing"),
            maximum_standardized_rms=10.0,
        )
    relabeled_locked = LabeledTranscriptCounts(
        calibration.gene,
        calibration.cell_ids,
        calibration.counts,
        culture_ids=("locked-culture",),
        plate_ids=("locked-plate",),
        times=[2.0],
        time_unit=SECOND,
        assay_id=assay.assay_id,
        source_id="relabel-source",
        preprocessing_id=calibration.preprocessing_id,
        source_parent_ids=("locked-source-parent",),
        preprocessing_parent_ids=("relabel-preprocessing-parent",),
    )
    relabeled = assess_pulse_chase_prediction(
        assay,
        schedule,
        calibration,
        relabeled_locked,
        prediction,
        _identifiability(assay, schedule, calibration),
        campaign=_campaign(),
        claim=_claim(),
        claimed_parameter_names=("synthesis",),
        timing_reference=_reference("relabel-timing"),
        maximum_standardized_rms=10.0,
    )
    assert relabeled.status == "failed"
    assert "cell-leakage" in relabeled.failed_checks


def test_missing_assay_covariance_blocks_readiness_and_latent_covariance_is_propagated():
    missing_assay = _assay(calibration_covariance=False)
    schedule = _schedule()
    calibration = _counts(missing_assay, "cal", 0.0)
    locked = _counts(missing_assay, "locked", 2.0)
    prediction = _prediction(missing_assay, schedule, calibration)
    identity = _identifiability(missing_assay, schedule, calibration)
    missing = assess_pulse_chase_prediction(
        missing_assay,
        schedule,
        calibration,
        locked,
        prediction,
        identity,
        campaign=_campaign(),
        claim=_claim(),
        claimed_parameter_names=("synthesis",),
        timing_reference=_reference("timing"),
        maximum_standardized_rms=10.0,
    )
    assert missing.status == "inconclusive"
    assert "assay-calibration-covariance" in missing.missing_prerequisites

    assay = LabeledTranscriptAssay(
        [0.5, 0.8, 0.6, 0.7],
        np.zeros(4),
        [[0.9, 0.2], [0.1, 0.8]],
        labeling_calibration=_reference("propagation-label-calibration"),
        count_calibration=_reference("propagation-count-calibration"),
        calibration_covariance=np.zeros((12, 12)),
    )
    calibration = _counts(assay, "cal", 0.0)
    locked = _counts(assay, "locked", 2.0)
    latent_covariance = np.asarray(
        [
            [
                [4.0, 1.0, 0.0, 0.0],
                [1.0, 9.0, 0.0, 0.0],
                [0.0, 0.0, 16.0, 2.0],
                [0.0, 0.0, 2.0, 25.0],
            ]
        ]
    )
    prediction = PulseChasePrediction(
        np.ones((1, 4)),
        latent_covariance,
        model_id="covariance-aware-model",
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )
    identity = _identifiability(
        assay, schedule, calibration, model_id="covariance-aware-model"
    )
    assessment = assess_pulse_chase_prediction(
        assay,
        schedule,
        calibration,
        locked,
        prediction,
        identity,
        campaign=_campaign(),
        claim=_claim(),
        claimed_parameter_names=("synthesis",),
        timing_reference=_reference("timing-propagation"),
        maximum_standardized_rms=10.0,
    )
    predicted, conditional_covariance = assay.conditional_moments(prediction.latent_means)
    probabilities = np.asarray(assay.observation_probabilities)
    total_covariance = np.asarray(conditional_covariance) + np.einsum(
        "ot,...tu,pu->...op",
        probabilities,
        latent_covariance,
        probabilities,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(total_covariance[0])
    expected_residuals = (
        eigenvectors.T @ (np.asarray(predicted)[0] - np.asarray(locked.counts)[0])
    ) / np.sqrt(eigenvalues)
    np.testing.assert_allclose(
        assessment.standardized_residuals[0], expected_residuals, rtol=1e-6
    )


def test_singular_joint_pulse_chase_covariance_is_inconclusive():
    assay = LabeledTranscriptAssay(
        np.ones(4),
        np.zeros(4),
        np.eye(2),
        labeling_calibration=_reference("singular-label-calibration"),
        count_calibration=_reference("singular-count-calibration"),
        calibration_covariance=np.zeros((12, 12)),
    )
    schedule = _schedule()
    calibration = _counts(assay, "cal", 0.0)
    locked = _counts(assay, "locked", 2.0)
    prediction = PulseChasePrediction(
        np.ones((1, 4)),
        np.zeros((1, 4, 4)),
        model_id="singular-covariance-model",
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )
    assessment = assess_pulse_chase_prediction(
        assay,
        schedule,
        calibration,
        locked,
        prediction,
        _identifiability(
            assay,
            schedule,
            calibration,
            model_id="singular-covariance-model",
        ),
        campaign=_campaign(),
        claim=_claim(),
        claimed_parameter_names=("synthesis",),
        timing_reference=_reference("singular-timing"),
        maximum_standardized_rms=10.0,
    )

    assert assessment.status == "inconclusive"
    assert any(
        item.startswith("singular-observation-covariance:")
        for item in assessment.missing_prerequisites
    )
