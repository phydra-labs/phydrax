# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Claim-bounded mechanics/electronics tests; data are independent fixtures."""

import hashlib

import numpy as np
import pytest

from benchmarks.nucleic_rigid import make_fixture
from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    fit_restricted_nucleotide_mechanics,
    NucleotideMechanicalResponseData,
)
from phydrax.applications.nucleic_acid_biophysics.electronics import (
    ChargeTransferObservationSeries,
    compare_electronic_models,
)
from phydrax.applications.nucleic_acid_biophysics.electronics._qualification import (
    ElectronicModelFit,
    ElectronicModelPrediction,
)
from phydrax.qualification import (
    CampaignRole,
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)
from phydrax.units import JOULE, ONE, SECOND


def _reference(label: str, *, training=True) -> ReferenceArtifactManifest:
    payload = label.encode()
    return ReferenceArtifactManifest(
        label,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=training,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"observable": 1.0},
        uncertainty={"standard_error": 0.1},
        lineage_ids=(f"lineage:{label}",),
    )


def _mechanical_data(prefix, independent_units, model, source, *, ill_conditioned=False):
    n = len(independent_units)
    sensitivities = np.zeros((n, 2, 3))
    if ill_conditioned:
        base = np.linspace(1.0, 2.0, n)[:, None]
        sensitivities[..., 0] = np.concatenate((base, 2.0 * base), axis=1)
        sensitivities[..., 1] = sensitivities[..., 0] * (1.0 + 1.0e-12)
    else:
        sensitivities[:, 0, 0] = np.linspace(1.0, 2.0, n)
        sensitivities[:, 1, 1] = np.linspace(2.0, 3.0, n)
    offsets = np.asarray([0.4, -0.2])
    observed = np.einsum("rop,p->ro", sensitivities[..., :2], offsets)
    return NucleotideMechanicalResponseData(
        tuple(f"{prefix}-case-{index}" for index in range(n)),
        tuple(independent_units),
        tuple(f"{prefix}-condition-{index}" for index in range(n)),
        source_row_ids=tuple(f"{prefix}-source-row-{index}" for index in range(n)),
        parent_case_ids=tuple(() for _ in range(n)),
        baseline_response=np.zeros((n, 2)),
        parameter_sensitivities=sensitivities,
        observed_response=observed,
        standard_errors=np.full((n, 2), 0.1),
        force_unit=JOULE,
        twist_unit=JOULE,
        source=source,
        model=model,
    )


def _fit_mechanics(calibration, locked, **overrides):
    campaign_cases = tuple(
        ScientificCase(
            case_id,
            unit_id,
            response.construct_id,
            condition_id,
            unit_id,
            f"batch:{case_id}",
            (response.source.manifest_id,),
        )
        for response in (calibration, locked)
        for case_id, unit_id, condition_id in zip(
            response.case_ids,
            response.independent_unit_ids,
            response.condition_ids,
            strict=True,
        )
    )
    campaign = ScientificCampaign(
        campaign_cases,
        (
            CampaignRole("calibration", calibration.case_ids),
            CampaignRole("locked_evaluation", locked.case_ids),
        ),
    )
    arguments = {
        "thermal_reference": _reference("thermal-evidence"),
        "structural_reference": _reference("structural-evidence"),
        "maximum_standardized_rms": 1.0,
        "relative_rank_tolerance": 1.0e-8,
        "campaign": campaign,
        "maximum_condition_number": 1.0e6,
    }
    arguments.update(overrides)
    return fit_restricted_nucleotide_mechanics(
        calibration,
        locked,
        ("stack", "twist", "excluded-volume"),
        (0, 1),
        **arguments,
    )


def test_restricted_mechanics_propagates_fit_covariance_to_locked_units():
    model, _ = make_fixture(4)
    source = _reference("mechanical-response")
    calibration = _mechanical_data("cal", ("prep-a", "prep-b"), model, source)
    locked = _mechanical_data("locked", ("prep-c", "prep-d"), model, source)

    assessment = _fit_mechanics(calibration, locked)

    assert assessment.status == "ready-for-claim-evaluation"
    assert assessment.fitted_parameter_names == ("stack", "twist")
    np.testing.assert_allclose(
        assessment.fitted_parameter_offsets, [0.4, -0.2], atol=1e-6
    )
    assert assessment.parameter_covariance is not None
    assert assessment.locked_prediction_standard_errors is not None
    assert np.all(np.asarray(assessment.locked_prediction_standard_errors) > 0.0)
    np.testing.assert_allclose(
        assessment.predicted_locked_response, locked.observed_response, atol=1e-6
    )
    criterion = ScientificMetricCriterion(
        "force-macro-standardized-rms",
        "at_most",
        None,
        2.0,
        "1",
        "independent_unit_macro",
    )
    wrong_support = dict(assessment.support_attributes)
    wrong_support["model_id"] = "forged-model"
    profile = ScientificClaimProfile(
        assessment.capability_name,
        SupportTuple(assessment.capability_name, wrong_support),
        assessment.observable_ids,
        assessment.condition_domain_ids,
        assessment.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
        (criterion,),
        "abstain",
        ("model-change",),
        frozen_criteria_ids=(criterion.criterion_id,),
    )
    with pytest.raises(ValueError, match="scope must exactly match"):
        assessment.evaluate_claim(
            profile,
            (),
            build_id="test-build",
            environment_id="test-environment",
            backend="cpu",
            topology="single-device",
            precision="float64",
            reduction="deterministic",
            replay_id="test-replay",
            raw_artifact_ids=("raw:claim",),
            reviewer_id="test-reviewer",
            issued_at=1,
            expires_at=100,
        )


def test_nearly_singular_mechanics_fit_is_inconclusive_despite_binary_rank():
    model, _ = make_fixture(4)
    source = _reference("ill-conditioned-response")
    calibration = _mechanical_data(
        "cal", ("prep-a", "prep-b"), model, source, ill_conditioned=True
    )
    locked = _mechanical_data(
        "locked", ("prep-c", "prep-d"), model, source, ill_conditioned=True
    )

    assessment = _fit_mechanics(calibration, locked)

    assert assessment.status == "inconclusive"
    assert "parameter-identifiability" in assessment.missing_prerequisites
    assert assessment.parameter_covariance is None
    assert assessment.locked_prediction_standard_errors is None


def test_mechanics_reports_unavailable_training_rights_as_inconclusive():
    model, _ = make_fixture(4)
    source = _reference("restricted-response", training=False)
    calibration = _mechanical_data("cal", ("prep-a", "prep-b"), model, source)
    locked = _mechanical_data("locked", ("prep-c", "prep-d"), model, source)

    assessment = _fit_mechanics(
        calibration,
        locked,
        thermal_reference=None,
        structural_reference=None,
    )

    assert assessment.status == "inconclusive"
    assert (
        "calibration-source-rights:training-use-not-permitted"
        in assessment.missing_prerequisites
    )
    assert "thermal-evidence" in assessment.missing_prerequisites


def _electronic_series(values=None):
    observed = np.asarray(
        [
            [[1.0, 0.0], [0.7, 0.3], [0.4, 0.6]],
            [[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]],
        ]
        if values is None
        else values
    )
    return ChargeTransferObservationSeries(
        ("series-a", "series-b"),
        ("preparation-a", "preparation-b"),
        ("buffer-a", "buffer-b"),
        sequence_id="AG",
        environment_id="declared-buffer-temperature",
        observable_kind="charge-transfer-population",
        times=[0.0, 1.0, 2.0],
        values=observed,
        standard_errors=np.full_like(observed, 0.1),
        valid=None,
        time_unit=SECOND,
        observable_unit=ONE,
        source=_reference("time-resolved-charge-transfer"),
    )


def _electronic_campaign(series):
    cases = [
        ScientificCase(
            "fit-case",
            "fit-unit",
            "AG",
            "fit-buffer",
            "fit-preparation",
            "fit-batch",
            (_reference("electronic-fit-source").manifest_id,),
        ),
        ScientificCase(
            "selection-case",
            "selection-unit",
            "AG",
            "selection-buffer",
            "selection-preparation",
            "selection-batch",
            (_reference("electronic-selection-source").manifest_id,),
        ),
    ]
    cases.extend(
        ScientificCase(
            case_id,
            independent_unit,
            series.sequence_id,
            condition_id,
            independent_unit,
            f"locked-batch-{index}",
            (series.source.manifest_id,),
        )
        for index, (case_id, independent_unit, condition_id) in enumerate(
            zip(
                series.series_ids,
                series.independent_unit_ids,
                series.condition_ids,
                strict=True,
            )
        )
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("fit-case",)),
            CampaignRole("model_selection", ("selection-case",)),
            CampaignRole("locked_evaluation", series.series_ids),
        ),
    )


def _electronic_prediction(series, campaign, values, *, model):
    parameters = np.asarray([0.25, -0.5])
    parameter_id = canonical_fingerprint(
        {
            "kind": "electronic-fitted-parameters",
            "values": array_tree_fingerprint(parameters),
        }
    )
    artifacts = (
        _reference("electronic-fit-source"),
        _reference("electronic-selection-source"),
    )
    code = _reference(f"{model}-prediction-code")
    fit_evidence = QualificationEvidence(
        "scientific",
        "passed",
        (
            campaign.campaign_id,
            model,
            parameter_id,
            code.manifest_id,
            *(artifact.manifest_id for artifact in artifacts),
        ),
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-device",
        precision="float64",
        reduction="deterministic",
        replay_id=f"fit-replay:{model}",
        criteria_ids=("fit-execution",),
        raw_artifact_ids=(f"raw:fit:{model}",),
        reviewer_id="test-reviewer",
        issued_at=1,
        expires_at=100,
        reason="fit execution passed",
    )
    fit = ElectronicModelFit(
        model,
        campaign,
        parameters,
        artifacts,
        code,
        fit_evidence,
    )
    return ElectronicModelPrediction(
        series,
        values,
        fit,
        observable_unit=ONE,
    )


def test_electronic_fit_rejects_forged_execution_identity():
    series = _electronic_series()
    campaign = _electronic_campaign(series)
    parameters = np.asarray([0.25, -0.5])
    artifacts = (
        _reference("electronic-fit-source"),
        _reference("electronic-selection-source"),
    )
    code = _reference("forged-prediction-code")
    forged = QualificationEvidence(
        "scientific",
        "passed",
        (
            campaign.campaign_id,
            "forged-model",
            code.manifest_id,
            *(artifact.manifest_id for artifact in artifacts),
        ),
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-device",
        precision="float64",
        reduction="deterministic",
        replay_id="forged-fit-replay",
        criteria_ids=("fit-execution",),
        raw_artifact_ids=("raw:forged-fit",),
        reviewer_id="test-reviewer",
        issued_at=1,
        expires_at=100,
        reason="forged fit identity",
    )
    with pytest.raises(ValueError, match="Fit execution evidence"):
        ElectronicModelFit(
            "forged-model",
            campaign,
            parameters,
            artifacts,
            code,
            forged,
        )


def _prediction_evidence(prediction):
    records = []
    for stage in (
        "parameter-identifiability",
        "predictive-calibration",
        "locked-prediction",
    ):
        subjects = (
            prediction.campaign_id,
            prediction.model_id,
            prediction.fit_id,
            *prediction.fit_integrity_ids,
        )
        if stage == "locked-prediction":
            subjects = (*subjects, prediction.prediction_id)
        records.append(
            QualificationEvidence(
                "scientific",
                "passed",
                subjects,
                build_id="test-build",
                environment_id="test-environment",
                backend="cpu",
                topology="single-device",
                precision="float64",
                reduction="deterministic",
                replay_id="test-replay",
                criteria_ids=(stage,),
                raw_artifact_ids=(f"raw:{stage}:{prediction.model_id}",),
                reviewer_id="test-reviewer",
                issued_at=1,
                expires_at=100,
                reason=f"{stage} passed",
            )
        )
    return tuple(records)


def _failed_locked_evidence(prediction):
    return QualificationEvidence(
        "scientific",
        "failed",
        (
            prediction.campaign_id,
            prediction.model_id,
            prediction.fit_id,
            *prediction.fit_integrity_ids,
            prediction.prediction_id,
        ),
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-device",
        precision="float64",
        reduction="deterministic",
        replay_id="failed-replay",
        criteria_ids=("locked-prediction",),
        raw_artifact_ids=(f"raw:failed:{prediction.model_id}",),
        reviewer_id="test-reviewer",
        issued_at=1,
        expires_at=100,
        reason="locked prediction failed",
    )


def test_electronic_comparison_requires_frozen_heldout_model_lineage():
    series = _electronic_series()
    observed = np.asarray(series.values)
    campaign = _electronic_campaign(series)
    quantum = observed + np.asarray([-0.01, 0.01])
    kinetic = observed + np.asarray([-0.2, 0.2])
    quantum_prediction = _electronic_prediction(
        series, campaign, quantum, model="quantum-model"
    )
    kinetic_prediction = _electronic_prediction(
        series, campaign, kinetic, model="kinetic-model"
    )

    ready = compare_electronic_models(
        series,
        quantum_prediction,
        kinetic_prediction,
        instrument_calibration=_reference("instrument-calibration"),
        environment_characterization=_reference("environment-characterization"),
        maximum_quantum_standardized_rms=1.0,
        minimum_quantum_improvement=1.0,
        prediction_evidence=(
            *_prediction_evidence(quantum_prediction),
            *_prediction_evidence(kinetic_prediction),
        ),
    )

    assert ready.status == "ready-for-claim-evaluation"
    assert ready.campaign_id == campaign.campaign_id
    assert ready.quantum_model_id == "quantum-model"
    assert ready.quantum_improvement > 1.0
    conflicting = compare_electronic_models(
        series,
        quantum_prediction,
        kinetic_prediction,
        instrument_calibration=_reference("instrument-calibration"),
        environment_characterization=_reference("environment-characterization"),
        maximum_quantum_standardized_rms=1.0,
        minimum_quantum_improvement=1.0,
        prediction_evidence=(
            *_prediction_evidence(quantum_prediction),
            *_prediction_evidence(kinetic_prediction),
            _failed_locked_evidence(quantum_prediction),
        ),
    )
    assert conflicting.status == "failed"
    assert "quantum:locked-prediction" in conflicting.failed_checks
    assert (
        _failed_locked_evidence(quantum_prediction).evidence_id
        in conflicting.qualification_evidence_ids
    )
    with pytest.raises(TypeError, match="ElectronicModelPrediction"):
        compare_electronic_models(
            series,
            quantum,
            kinetic,
            instrument_calibration=None,
            environment_characterization=None,
            maximum_quantum_standardized_rms=1.0,
            minimum_quantum_improvement=1.0,
            prediction_evidence=(),
        )
    wrong_criterion = ScientificMetricCriterion(
        "quantum-macro-standardized-rms",
        "at_most",
        None,
        2.0,
        "1",
        "independent_unit_macro",
    )
    weak_profile = ScientificClaimProfile(
        ready.capability_name,
        SupportTuple(ready.capability_name, dict(ready.support_attributes)),
        ready.observable_ids,
        ready.condition_domain_ids,
        campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
        (wrong_criterion,),
        "abstain",
        ("model-change",),
        frozen_criteria_ids=(wrong_criterion.criterion_id,),
    )
    with pytest.raises(ValueError, match="criteria must exactly match"):
        ready.evaluate_claim(
            weak_profile,
            (
                *_prediction_evidence(quantum_prediction),
                *_prediction_evidence(kinetic_prediction),
            ),
            build_id="test-build",
            environment_id="test-environment",
            backend="cpu",
            topology="single-device",
            precision="float64",
            reduction="deterministic",
            replay_id="test-replay",
            raw_artifact_ids=("raw:claim",),
            reviewer_id="test-reviewer",
            issued_at=1,
            expires_at=100,
        )
    wrong_support = dict(ready.support_attributes)
    wrong_support["sequence_id"] = "forged-sequence"
    scope_profile = ScientificClaimProfile(
        ready.capability_name,
        SupportTuple(ready.capability_name, wrong_support),
        ready.observable_ids,
        ready.condition_domain_ids,
        campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "parameter-identifiability",
            "predictive-calibration",
            "locked-prediction",
        ),
        (wrong_criterion,),
        "abstain",
        ("model-change",),
        frozen_criteria_ids=(wrong_criterion.criterion_id,),
    )
    with pytest.raises(ValueError, match="scope must exactly match"):
        ready.evaluate_claim(
            scope_profile,
            (
                *_prediction_evidence(quantum_prediction),
                *_prediction_evidence(kinetic_prediction),
            ),
            build_id="test-build",
            environment_id="test-environment",
            backend="cpu",
            topology="single-device",
            precision="float64",
            reduction="deterministic",
            replay_id="test-replay",
            raw_artifact_ids=("raw:claim",),
            reviewer_id="test-reviewer",
            issued_at=1,
            expires_at=100,
        )


def test_charge_transfer_populations_and_predictions_obey_probability_support():
    with pytest.raises(ValueError, match="sum to one"):
        _electronic_series(
            [
                [[1.0, 0.0], [0.8, 0.3], [0.4, 0.6]],
                [[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]],
            ]
        )
    series = _electronic_series()
    campaign = _electronic_campaign(series)
    observed = np.asarray(series.values)
    invalid = observed.copy()
    invalid[0, 1] = [1.1, -0.1]
    comparison = compare_electronic_models(
        series,
        _electronic_prediction(series, campaign, invalid, model="quantum-model"),
        _electronic_prediction(
            series,
            campaign,
            observed + np.asarray([-0.2, 0.2]),
            model="kinetic-model",
        ),
        instrument_calibration=_reference("instrument-calibration"),
        environment_characterization=_reference("environment-characterization"),
        maximum_quantum_standardized_rms=10.0,
        minimum_quantum_improvement=0.0,
        prediction_evidence=(
            *_prediction_evidence(
                _electronic_prediction(series, campaign, invalid, model="quantum-model")
            ),
            *_prediction_evidence(
                _electronic_prediction(
                    series,
                    campaign,
                    observed + np.asarray([-0.2, 0.2]),
                    model="kinetic-model",
                )
            ),
        ),
    )

    assert comparison.status == "failed"
    assert "quantum-population-outside-probability-support" in comparison.failed_checks
