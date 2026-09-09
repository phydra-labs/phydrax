# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Staged initial-lesion contracts; fixtures contain no transport or gel corpus."""

import hashlib

import numpy as np

from phydrax.applications import radiation_biophysics as rad
from phydrax.applications.radiation_biophysics.interchange import (
    NANOMETER,
    TimedRadiationHistoryProfile,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)
from phydrax.units import ELECTRONVOLT, SECOND


def _reference(
    label: str, *, commercial_use_permitted: bool = True
) -> ReferenceArtifactManifest:
    payload = label.encode()
    return ReferenceArtifactManifest(
        label,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=commercial_use_permitted,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"count": 1.0},
        uncertainty={"standard_error": 0.1},
        lineage_ids=("independent-test-fixture",),
    )


def _source() -> rad.RadiationSource:
    reference = _reference("external-history")
    return rad.RadiationSource(
        ScientificArtifactEnvelope(
            artifact_kind="external-radiation-history",
            content_digest=reference.checksum,
            producer="independent-test",
            producer_version="1",
            build_id="fixture",
            license_id=reference.license_id,
            resource_id="caller-array",
            status="complete",
        ),
        (reference,),
        "external-engine",
        "declared-revision",
        "declared-configuration",
        ("rng-lineage",),
        ("cross-section-table",),
        (),
        "world",
        NANOMETER,
        ELECTRONVOLT,
        SECOND,
        1.0,
        "chemical-model",
        "scavenging-model",
    )


def _profile(*, hit_count: int = 2) -> TimedRadiationHistoryProfile:
    source = _source()
    histories = tuple(
        rad.PrimaryHistoryKey(source.artifact.artifact_id, "run", str(index), "fraction")
        for index in range(3)
    )
    return TimedRadiationHistoryProfile(
        source,
        histories,
        physical_tuple_ids=("tuple-zero", "tuple-hit", "tuple-unreported"),
        physical_event_counts=[0, hit_count, 0],
        dose_gy=[0.0, 1.0, 0.0],
        dose_standard_errors_gy=[0.01, 0.05, 0.01],
        species_ids=("OH",),
        sample_times=[0.0, 1.0],
        species_counts=np.asarray([[[0, 0]], [[hit_count, 0]], [[99, 99]]]),
        species_valid=np.asarray([[[True, True]], [[True, True]], [[False, False]]]),
        time_unit=SECOND,
        dosimetry_reference=None,
        transport_reference=None,
        chemical_reference=None,
    )


def _criterion() -> ScientificMetricCriterion:
    return ScientificMetricCriterion(
        "gel-day-macro-standardized-rms",
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
            "gel-calibration-case",
            "day-cal",
            "plasmid",
            "tuple-zero",
            "preparation-calibration",
            "batch-cal",
            ("gel-calibration-source",),
        ),
        ScientificCase(
            "gel-locked-case",
            "day-locked",
            "plasmid",
            "tuple-hit",
            "preparation-locked",
            "batch-locked",
            ("gel-locked-source",),
        ),
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("gel-calibration-case",)),
            CampaignRole("locked_evaluation", ("gel-locked-case",)),
        ),
        criteria_ids=(criterion.criterion_id,),
    )


def _claim(
    *,
    frozen_criteria_ids: tuple[str, ...] | None = None,
    support_attributes: dict[str, str] | None = None,
) -> ScientificClaimProfile:
    capability = "radiation.initial-lesion-gel"
    criterion = _criterion()
    campaign = _campaign()
    return ScientificClaimProfile(
        capability,
        SupportTuple(
            capability,
            {"target": "plasmid"} if support_attributes is None else support_attributes,
        ),
        ("raw-plasmid-gel-bands",),
        ("declared-physical-tuples",),
        campaign.campaign_id,
        (
            "source-admission",
            "measurement-calibration",
            "numerical-validity",
            "chemical-validity",
            "locked-prediction",
            "external-transfer",
        ),
        (criterion,),
        "abstain-without-independent-stages",
        ("source-artifact-change", "observation-law-change"),
        frozen_criteria_ids=(
            campaign.criteria_ids if frozen_criteria_ids is None else frozen_criteria_ids
        ),
    )


def _gel(
    day: str,
    physical_tuple: str,
    label: str,
    *,
    lane_gain_uncertainty: bool = True,
    intensities=((10.0, 0.0, 0.0),),
    observation_covariance=None,
) -> rad.PlasmidGelObservations:
    return rad.PlasmidGelObservations(
        (f"lane-{label}",),
        (day,),
        (physical_tuple,),
        preparation_ids=(f"preparation-{label}",),
        lane_gain=[10.0],
        lane_gain_standard_errors=[2.0] if lane_gain_uncertainty else None,
        intensities=intensities,
        standard_errors=[[1.0, 1.0, 1.0]],
        observation_covariance=observation_covariance,
        source=_reference(f"gel-{label}"),
    )


def _form_prediction(
    profile: TimedRadiationHistoryProfile,
    observations: rad.PlasmidGelObservations,
    *,
    predictive_uncertainty: bool = True,
) -> rad.PlasmidFormPrediction:
    fit_tuple = next(
        item
        for item in profile.physical_tuple_ids
        if item not in observations.physical_tuple_ids
    )
    return rad.PlasmidFormPrediction(
        profile,
        observations.physical_tuple_ids,
        [[1.0, 0.0, 0.0]],
        campaign_id=_claim().campaign_id,
        model_id="radiation-form-model",
        fit_id="radiation-form-fit",
        prediction_source_artifact_id="radiation-prediction-artifact",
        fit_physical_tuple_ids=(fit_tuple,),
        fit_independent_unit_ids=("fit-day",),
        fit_preparation_ids=("fit-preparation",),
        form_fraction_covariance=(
            np.zeros((1, 3, 3)) if predictive_uncertainty else None
        ),
    )


def test_history_coverage_preserves_reported_zeros_without_inventing_missing_zeros():
    profile = _profile()
    coverage = profile.coverage(("OH",), (0.0, 1.0))

    assert profile.histories[0] in coverage.zero_physical_histories
    assert profile.histories[0] in coverage.zero_chemical_histories
    assert profile.histories[2] not in coverage.zero_chemical_histories
    assert coverage.missing_references == ("dosimetry", "transport", "chemical-G")
    assert not coverage.complete


def test_gel_observation_law_propagates_calibration_and_rejects_nonprobabilities():
    profile = _profile()
    assay = rad.PlasmidGelAssay(
        np.eye(3),
        np.zeros(3),
        _reference("gel-calibration"),
        calibration_covariance=np.eye(12) * 0.01,
    )
    observation_error_covariance = np.asarray(
        [[[1.0, 0.25, 0.0], [0.25, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    observations = _gel(
        "day-a",
        "tuple-zero",
        "locked",
        intensities=((8.0, 0.0, 0.0),),
        observation_covariance=observation_error_covariance,
    )
    evaluation = rad.evaluate_plasmid_gel(
        assay, observations, _form_prediction(profile, observations)
    )

    np.testing.assert_allclose(evaluation.predicted_intensities, [[10.0, 0.0, 0.0]])
    np.testing.assert_allclose(
        evaluation.combined_standard_errors[0, 0],
        np.sqrt(1.0 + 1.0 + 0.01 + 4.0),
        rtol=1e-6,
    )
    calibration_covariance = np.asarray(
        assay.calibration_intensity_covariance(
            [[1.0, 0.0, 0.0]],
            observations.lane_gain,
            observations.lane_gain_standard_errors,
        )
    )
    eigenvalues, eigenvectors = np.linalg.eigh(
        observation_error_covariance[0] + calibration_covariance[0]
    )
    expected_whitened = (
        eigenvectors.T
        @ (
            np.asarray(evaluation.predicted_intensities)[0]
            - np.asarray(observations.intensities)[0]
        )
    ) / np.sqrt(eigenvalues)
    np.testing.assert_allclose(
        evaluation.standardized_residuals[0], expected_whitened, rtol=1e-6
    )
    assert not evaluation.uncertainty_limitations
    singular_assay = rad.PlasmidGelAssay(
        np.eye(3),
        np.zeros(3),
        _reference("singular-gel-calibration"),
        calibration_covariance=np.zeros((12, 12)),
    )
    singular_observations = _gel(
        "day-singular",
        "tuple-zero",
        "singular",
        lane_gain_uncertainty=False,
        observation_covariance=np.ones((1, 3, 3)),
    )
    singular_evaluation = rad.evaluate_plasmid_gel(
        singular_assay,
        singular_observations,
        _form_prediction(profile, singular_observations),
    )
    assert not singular_evaluation.finite
    assert "singular-gel-observation-covariance:lane-0" in (
        singular_evaluation.uncertainty_limitations
    )
    try:
        rad.evaluate_plasmid_gel(assay, observations, [[1.0, 0.0, 0.0]])
    except TypeError as error:
        assert "profile-bound" in str(error)
    else:
        raise AssertionError("Anonymous plasmid form predictions were admitted.")
    leaked_prediction = rad.PlasmidFormPrediction(
        profile,
        observations.physical_tuple_ids,
        [[1.0, 0.0, 0.0]],
        campaign_id=_claim().campaign_id,
        model_id="leaked-model",
        fit_id="leaked-fit",
        prediction_source_artifact_id="leaked-prediction-artifact",
        fit_physical_tuple_ids=observations.physical_tuple_ids,
        fit_independent_unit_ids=("fit-day",),
        fit_preparation_ids=("different-preparation",),
        form_fraction_covariance=np.zeros((1, 3, 3)),
    )
    try:
        rad.evaluate_plasmid_gel(assay, observations, leaked_prediction)
    except ValueError as error:
        assert "overlap prediction fit tuples" in str(error)
    else:
        raise AssertionError("Locked history tuples were reused by the fitted model.")
    preparation_leaked_prediction = rad.PlasmidFormPrediction(
        profile,
        observations.physical_tuple_ids,
        [[1.0, 0.0, 0.0]],
        campaign_id=_claim().campaign_id,
        model_id="preparation-leaked-model",
        fit_id="preparation-leaked-fit",
        prediction_source_artifact_id="preparation-leaked-artifact",
        fit_physical_tuple_ids=("tuple-hit",),
        fit_independent_unit_ids=("fit-day",),
        fit_preparation_ids=observations.preparation_ids,
        form_fraction_covariance=np.zeros((1, 3, 3)),
    )
    try:
        rad.evaluate_plasmid_gel(assay, observations, preparation_leaked_prediction)
    except ValueError as error:
        assert "overlap prediction fit preparations" in str(error)
    else:
        raise AssertionError("Locked gel preparations were reused by the fitted model.")
    try:
        assay.expected_intensity([[0.8, 0.8, 0.0]], [1.0])
    except ValueError as error:
        assert "probability rows" in str(error)
    else:
        raise AssertionError("Non-normalized plasmid fractions were admitted.")


def test_staged_assessment_binds_history_and_blocks_missing_calibration_uncertainty():
    profile = _profile()
    assay = rad.PlasmidGelAssay(
        np.eye(3),
        np.zeros(3),
        _reference("gel-law"),
    )
    calibration = _gel("day-cal", "tuple-zero", "calibration")
    locked = _gel(
        "day-locked",
        "tuple-hit",
        "locked",
        lane_gain_uncertainty=False,
    )
    evaluation = rad.evaluate_plasmid_gel(
        assay,
        locked,
        _form_prediction(profile, locked, predictive_uncertainty=False),
    )
    try:
        rad.assess_radiation_initial_lesions(
            profile,
            calibration,
            locked,
            evaluation,
            (),
            required_species_ids=("OH",),
            required_sample_times=(0.0, 1.0),
            campaign=_campaign(),
            claim=_claim(
                frozen_criteria_ids=(
                    _criterion().criterion_id,
                    "unrelated-frozen-criterion",
                )
            ),
            maximum_day_macro_standardized_rms=2.0,
        )
    except ValueError as error:
        assert "frozen criteria" in str(error)
    else:
        raise AssertionError("Mismatched frozen campaign criteria were admitted.")
    try:
        rad.assess_radiation_initial_lesions(
            profile,
            calibration,
            locked,
            evaluation,
            (),
            required_species_ids=("OH",),
            required_sample_times=(0.0, 1.0),
            campaign=_campaign(),
            claim=_claim(support_attributes={"target": "plasmid", "matrix": "agarose"}),
            maximum_day_macro_standardized_rms=2.0,
        )
    except ValueError as error:
        assert "exact assessed scope" in str(error)
    else:
        raise AssertionError("Broader radiation support consumed gel evidence.")
    inconclusive = rad.assess_radiation_initial_lesions(
        profile,
        calibration,
        locked,
        evaluation,
        (),
        required_species_ids=("OH",),
        required_sample_times=(0.0, 1.0),
        campaign=_campaign(),
        claim=_claim(),
        maximum_day_macro_standardized_rms=2.0,
    )
    unrelated_profile = _profile(hit_count=3)
    try:
        rad.assess_radiation_initial_lesions(
            unrelated_profile,
            calibration,
            locked,
            evaluation,
            (),
            required_species_ids=("OH",),
            required_sample_times=(0.0, 1.0),
            campaign=_campaign(),
            claim=_claim(),
            maximum_day_macro_standardized_rms=2.0,
        )
    except ValueError as error:
        assert "history profile" in str(error)
    else:
        raise AssertionError("A gel evaluation was transferred across history profiles.")

    leaked = _gel("day-cal", "tuple-hit", "leaked")
    leaked_evaluation = rad.evaluate_plasmid_gel(
        assay, leaked, _form_prediction(profile, leaked)
    )
    failed = rad.assess_radiation_initial_lesions(
        profile,
        calibration,
        leaked,
        leaked_evaluation,
        (),
        required_species_ids=("OH",),
        required_sample_times=(0.0, 1.0),
        campaign=_campaign(),
        claim=_claim(),
        maximum_day_macro_standardized_rms=2.0,
    )

    assert inconclusive.status == "inconclusive"
    assert (
        "measurement-calibration:gel-response-background-calibration-covariance"
        in inconclusive.missing_prerequisites
    )
    assert (
        "measurement-calibration:gel-lane-gain-uncertainty"
        in inconclusive.missing_prerequisites
    )
    assert (
        "predictive-calibration:radiation-form-prediction-covariance"
        in inconclusive.missing_prerequisites
    )
    assert failed.status == "failed"
    assert "irradiation-day-leakage" in failed.failed_checks


def test_domain_stage_evidence_requires_profile_lineage_and_requested_use_rights():
    profile = _profile()
    assay = rad.PlasmidGelAssay(
        np.eye(3),
        np.zeros(3),
        _reference("bound-gel-law"),
        calibration_covariance=np.zeros((12, 12)),
    )
    calibration = _gel("day-cal", "tuple-zero", "bound-calibration")
    locked = _gel("day-locked", "tuple-hit", "bound-locked")
    evaluation = rad.evaluate_plasmid_gel(
        assay, locked, _form_prediction(profile, locked)
    )
    stages = tuple(
        rad.RadiationStageEvidence(
            stage,
            (f"{stage}-condition",),
            (1.0,),
            (1.0,),
            (0.1,),
            rad.GRAY,
            _reference(
                f"{stage}-evidence",
                commercial_use_permitted=stage != "dosimetry",
            ),
            "external-reference",
            2.0,
            (
                "unrelated-history-profile"
                if stage == "transport"
                else profile.profile_id,
            ),
        )
        for stage in (
            "dosimetry",
            "transport",
            "chemical-G",
            "target-reactions",
            "lesion-yields",
        )
    )
    assessment = rad.assess_radiation_initial_lesions(
        profile,
        calibration,
        locked,
        evaluation,
        stages,
        required_species_ids=("OH",),
        required_sample_times=(0.0, 1.0),
        campaign=_campaign(),
        claim=_claim(),
        maximum_day_macro_standardized_rms=2.0,
        commercial_use=True,
    )

    assert "history-profile-stage-binding:transport" in assessment.missing_prerequisites
    assert any(
        item.startswith("stage-rights:dosimetry:")
        for item in assessment.missing_prerequisites
    )
