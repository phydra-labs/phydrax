# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Protein-nucleic and cotranslation claim-input contracts on explicit fixtures."""

import hashlib

import numpy as np
import pytest

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.protein_folding import protein_nucleic
from phydrax.applications.protein_folding._construct import ProteinConstruct
from phydrax.applications.protein_folding.cotranslation import (
    assess_cotranslation_prediction,
    CotranslationObservationLaw,
    LengthResolvedCotranslationObservations,
)
from phydrax.applications.protein_folding.cotranslation._qualification import (
    CotranslationModelFit,
    CotranslationModelPrediction,
)
from phydrax.applications.protein_folding.protein_nucleic._qualification import (
    ProteinNucleicMechanicsPrediction,
    ProteinNucleicModelFit,
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
from phydrax.units import ANGSTROM, derived_unit, JOULE, METER, MILLISECOND, ONE, SECOND


_PER_SECOND = derived_unit("test-1/s", ((SECOND, -1),))


def _reference(label: str, *, uncertainty=True, training=True, lineage=()):
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
        uncertainty={"standard_error": 0.1} if uncertainty else None,
        lineage_ids=tuple(lineage) or (f"lineage:{label}",),
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
            *prediction.fit_integrity_ids,
            prediction.fit_id,
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


def _assert_scope_mismatch(assessment, evidence, metric_id):
    criterion = ScientificMetricCriterion(
        metric_id,
        "at_most",
        None,
        2.0,
        "1",
        "independent_unit_macro",
    )
    support = dict(assessment.support_attributes)
    support["model_id"] = "forged-model"
    profile = ScientificClaimProfile(
        assessment.capability_name,
        SupportTuple(assessment.capability_name, support),
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
            evidence,
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


def _model_fit(fit_type, model, campaign, parameter_kind, artifacts):
    parameters = np.asarray([0.25, -0.5])
    parameter_id = canonical_fingerprint(
        {
            "kind": parameter_kind,
            "values": array_tree_fingerprint(parameters),
        }
    )
    code = _reference(f"{model}-prediction-code")
    source_ids = tuple(
        sorted(
            {
                source_id
                for case in campaign.cases
                if case.case_id
                in {
                    *next(
                        role.case_ids
                        for role in campaign.roles
                        if role.name == "calibration"
                    ),
                    *next(
                        role.case_ids
                        for role in campaign.roles
                        if role.name == "model_selection"
                    ),
                }
                for source_id in case.source_manifest_ids
            }
        )
    )
    evidence = QualificationEvidence(
        "scientific",
        "passed",
        (
            campaign.campaign_id,
            model,
            parameter_id,
            code.manifest_id,
            *source_ids,
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
    return fit_type(model, campaign, parameters, artifacts, code, evidence)


def _campaign(observations, *, cotranslation=False, fit_sources=None):
    fit_artifacts = (
        (_reference("fit-source"), _reference("selection-source"))
        if fit_sources is None
        else tuple(fit_sources)
    )
    fit_source_ids = tuple(artifact.manifest_id for artifact in fit_artifacts)
    if cotranslation:
        locked_constructs = observations.construct_ids
    else:
        locked_constructs = (observations.complex_construct_id,) * len(
            observations.case_ids
        )
    cases = [
        ScientificCase(
            "fit-case",
            "fit-unit",
            "fit-construct",
            "fit-condition",
            "fit-preparation",
            "fit-batch",
            (fit_source_ids[0],),
        ),
        ScientificCase(
            "selection-case",
            "selection-unit",
            "selection-construct",
            "selection-condition",
            "selection-preparation",
            "selection-batch",
            (fit_source_ids[-1],),
        ),
    ]
    cases.extend(
        ScientificCase(
            case_id,
            unit_id,
            construct_id,
            condition_id,
            preparation_id,
            f"locked-batch-{index}",
            (observations.source.manifest_id,),
        )
        for index, (
            case_id,
            unit_id,
            construct_id,
            condition_id,
            preparation_id,
        ) in enumerate(
            zip(
                observations.case_ids,
                observations.independent_unit_ids,
                locked_constructs,
                observations.condition_ids,
                observations.preparation_ids,
                strict=True,
            )
        )
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("fit-case",)),
            CampaignRole("model_selection", ("selection-case",)),
            CampaignRole("locked_evaluation", observations.case_ids),
        ),
    )


def _mechanics_observations():
    return protein_nucleic.ProteinNucleicMechanicalObservations(
        ProteinConstruct(("protein",), ("AC",)),
        NucleicAcidConstruct(("dna",), ("AT",), ("DNA",), (False,)),
        ("case-a", "case-b"),
        ("preparation-a", "preparation-b"),
        ("prep-a", "prep-b"),
        ("buffer-a", "buffer-b"),
        observable_kind="contact-probability",
        values=[0.2, 0.8],
        standard_errors=[0.1, 0.1],
        unit=ONE,
        source=_reference("complex-mechanics"),
    )


def test_complex_mechanics_requires_frozen_heldout_prediction_lineage():
    observations = _mechanics_observations()
    campaign = _campaign(observations)
    fit = _model_fit(
        ProteinNucleicModelFit,
        "complex-model",
        campaign,
        "protein-nucleic-fitted-parameters",
        (_reference("fit-source"), _reference("selection-source")),
    )
    prediction = ProteinNucleicMechanicsPrediction(
        observations,
        [0.2, 0.8],
        [0.05, 0.05],
        fit,
        unit=ONE,
    )
    assessment = protein_nucleic.assess_protein_nucleic_mechanics(
        observations,
        prediction,
        mapping_reference=None,
        maximum_standardized_rms=1.0,
        prediction_evidence=_prediction_evidence(prediction),
    )

    assert assessment.status == "inconclusive"
    assert assessment.campaign_id == campaign.campaign_id
    assert assessment.missing_prerequisites == ("protein-nucleic-coordinate-mapping",)
    conflicting = protein_nucleic.assess_protein_nucleic_mechanics(
        observations,
        prediction,
        mapping_reference=None,
        maximum_standardized_rms=1.0,
        prediction_evidence=(
            *_prediction_evidence(prediction),
            _failed_locked_evidence(prediction),
        ),
    )
    assert conflicting.status == "failed"
    assert "locked-prediction" in conflicting.failed_checks
    assert (
        _failed_locked_evidence(prediction).evidence_id
        in conflicting.qualification_evidence_ids
    )
    with pytest.raises(TypeError, match="ProteinNucleicMechanicsPrediction"):
        protein_nucleic.assess_protein_nucleic_mechanics(
            observations,
            [0.2, 0.8],
            mapping_reference=None,
            maximum_standardized_rms=1.0,
            prediction_evidence=(),
        )
    _assert_scope_mismatch(
        assessment,
        _prediction_evidence(prediction),
        "complex-mechanics-macro-standardized-rms",
    )


def _affinity_inputs(**overrides):
    arguments = {
        "bound_free_energy": [-8.0],
        "unbound_protein_free_energy": [-2.0],
        "unbound_nucleic_free_energy": [-1.0],
        "standard_state_correction": [0.5],
        "bound_standard_error": [0.2],
        "unbound_protein_standard_error": [0.1],
        "unbound_nucleic_standard_error": [0.1],
        "standard_state_standard_error": [0.05],
        "component_covariance": None,
        "components_conditionally_independent": True,
        "observed_binding_free_energy": [-4.5],
        "observation_standard_error": [0.2],
        "energy_unit": JOULE,
        "bound_sampling_reference": _reference(
            "bound", lineage=("simulation-root", "bound-branch")
        ),
        "unbound_sampling_reference": _reference(
            "unbound", lineage=("simulation-root", "unbound-branch")
        ),
        "binding_measurement_reference": _reference(
            "binding-measurement", lineage=("measurement-root",)
        ),
        "shared_sampling_lineage_ids": ("simulation-root",),
    }
    arguments.update(overrides)
    bound = arguments["bound_sampling_reference"]
    unbound = arguments["unbound_sampling_reference"]
    measurement = arguments["binding_measurement_reference"]
    fit_sources = tuple(
        reference.manifest_id for reference in (bound, unbound) if reference is not None
    ) or ("missing-fit-source",)
    locked_sources = (
        (measurement.manifest_id,)
        if measurement is not None
        else ("missing-observation-source",)
    )
    campaign = ScientificCampaign(
        (
            ScientificCase(
                "affinity-fit",
                "affinity-fit-unit",
                "fit-complex",
                "fit-condition",
                "affinity-fit-preparation",
                "affinity-fit-batch",
                fit_sources,
            ),
            ScientificCase(
                "affinity-locked",
                "preparation-a",
                "locked-complex",
                "condition-a",
                "affinity-preparation",
                "affinity-locked-batch",
                locked_sources,
            ),
        ),
        (
            CampaignRole("calibration", ("affinity-fit",)),
            CampaignRole("locked_evaluation", ("affinity-locked",)),
        ),
    )
    fit = _model_fit(
        ProteinNucleicModelFit,
        "affinity-model",
        campaign,
        "protein-nucleic-fitted-parameters",
        tuple(reference for reference in (bound, unbound) if reference is not None),
    )
    return protein_nucleic.ProteinNucleicAffinityInputs(
        ("condition-a",),
        ("preparation-a",),
        ("one-molar",),
        ("affinity-locked",),
        ("affinity-preparation",),
        fit,
        **arguments,
    )


def test_affinity_propagates_correction_uncertainty_under_explicit_independence():
    inputs = _affinity_inputs()
    assessment = protein_nucleic.assess_protein_nucleic_affinity(
        inputs,
        maximum_standardized_rms=1.0,
        prediction_evidence=_prediction_evidence(inputs),
    )

    np.testing.assert_allclose(inputs.predicted_binding_free_energy, [-4.5])
    np.testing.assert_allclose(
        inputs.prediction_standard_error, [np.sqrt(0.2**2 + 0.1**2 + 0.1**2 + 0.05**2)]
    )
    assert assessment.status == "ready-for-claim-evaluation"
    _assert_scope_mismatch(
        assessment,
        _prediction_evidence(inputs),
        "binding-affinity-macro-standardized-rms",
    )


def test_affinity_rejects_reused_or_rights_denied_fit_artifacts():
    denied = _reference("denied", training=False, lineage=("denied-root",))
    with pytest.raises(PermissionError):
        _affinity_inputs(bound_sampling_reference=denied)

    reused = _reference("reused", lineage=("reused-root",))
    with pytest.raises(ValueError):
        _affinity_inputs(
            bound_sampling_reference=reused,
            unbound_sampling_reference=reused,
            binding_measurement_reference=reused,
            shared_sampling_lineage_ids=("reused-root",),
        )


def test_affinity_without_joint_covariance_or_independence_is_inconclusive():
    inputs = _affinity_inputs(
        component_covariance=None,
        components_conditionally_independent=False,
        standard_state_standard_error=None,
    )
    assessment = protein_nucleic.assess_protein_nucleic_affinity(
        inputs,
        maximum_standardized_rms=1.0,
        prediction_evidence=_prediction_evidence(inputs),
    )
    assert assessment.status == "inconclusive"
    assert (
        "affinity-component-joint-covariance-or-conditional-independence"
        in assessment.missing_prerequisites
    )


def _cotranslation_observations(
    timing_reference,
    *,
    time_unit=SECOND,
    dwell=(1.0, 2.0),
    dwell_error=(0.1, 0.1),
    values=(0.5, 1.0 / 65.0),
):
    return LengthResolvedCotranslationObservations(
        ("length-20", "length-30"),
        ("preparation-a", "preparation-b"),
        ("prep-a", "prep-b"),
        ("buffer-a", "buffer-b"),
        construct_ids=("construct-20", "construct-30"),
        nascent_lengths=[20, 30],
        measured_dwell_times=dwell,
        dwell_time_standard_errors=dwell_error,
        values=values,
        standard_errors=[0.05, 0.05],
        time_unit=time_unit,
        timing_semantics="measured-dwell-time",
        source=_reference("length-resolved-observations"),
        timing_reference=timing_reference,
    )


def _cotranslation_prediction(observations, values, errors, unit):
    campaign = _campaign(observations, cotranslation=True)
    fit = _model_fit(
        CotranslationModelFit,
        "frozen-cotranslation-model",
        campaign,
        "cotranslation-fitted-parameters",
        (_reference("fit-source"), _reference("selection-source")),
    )
    return CotranslationModelPrediction(
        observations,
        values,
        errors,
        fit,
        latent_unit=unit,
    )


def test_protein_fit_records_reject_forged_execution_identity():
    mechanics = _mechanics_observations()
    mechanics_campaign = _campaign(mechanics)
    cotranslation = _cotranslation_observations(_reference("forged-timing"))
    cotranslation_campaign = _campaign(cotranslation, cotranslation=True)
    artifacts = (_reference("fit-source"), _reference("selection-source"))
    parameters = np.asarray([0.25, -0.5])
    for fit_type, model, campaign in (
        (ProteinNucleicModelFit, "forged-protein-nucleic", mechanics_campaign),
        (CotranslationModelFit, "forged-cotranslation", cotranslation_campaign),
    ):
        code = _reference(f"{model}-code")
        forged = QualificationEvidence(
            "scientific",
            "passed",
            (
                campaign.campaign_id,
                model,
                code.manifest_id,
                *(artifact.manifest_id for artifact in artifacts),
            ),
            build_id="test-build",
            environment_id="test-environment",
            backend="cpu",
            topology="single-device",
            precision="float64",
            reduction="deterministic",
            replay_id=f"forged-replay:{model}",
            criteria_ids=("fit-execution",),
            raw_artifact_ids=(f"raw:forged:{model}",),
            reviewer_id="test-reviewer",
            issued_at=1,
            expires_at=100,
            reason="forged fit identity",
        )
        with pytest.raises(ValueError, match="Fit execution evidence"):
            fit_type(
                model,
                campaign,
                parameters,
                artifacts,
                code,
                forged,
            )


def test_cotranslation_converts_fret_length_units_and_requires_frozen_lineage():
    law = CotranslationObservationLaw(
        "length-resolved-fret",
        _reference("fret-law"),
        forster_radius=2.0,
        forster_radius_standard_error=0.05,
        length_unit=ANGSTROM,
    )
    observations = _cotranslation_observations(_reference("measured-timing"))
    prediction = _cotranslation_prediction(
        observations, [2.0e-10, 4.0e-10], [1.0e-12, 1.0e-12], METER
    )
    ready = assess_cotranslation_prediction(
        law,
        observations,
        prediction,
        maximum_standardized_rms=1.0,
        prediction_evidence=_prediction_evidence(prediction),
    )

    np.testing.assert_allclose(ready.predicted_observations, [0.5, 1.0 / 65.0])
    assert ready.status == "ready-for-claim-evaluation"
    conflicting = assess_cotranslation_prediction(
        law,
        observations,
        prediction,
        maximum_standardized_rms=1.0,
        prediction_evidence=(
            *_prediction_evidence(prediction),
            _failed_locked_evidence(prediction),
        ),
    )
    assert conflicting.status == "failed"
    assert "locked-prediction" in conflicting.failed_checks
    assert (
        _failed_locked_evidence(prediction).evidence_id
        in conflicting.qualification_evidence_ids
    )
    with pytest.raises(TypeError, match="CotranslationModelPrediction"):
        assess_cotranslation_prediction(
            law,
            observations,
            [2.0, 4.0],
            maximum_standardized_rms=1.0,
            prediction_evidence=(),
        )
    _assert_scope_mismatch(
        ready,
        _prediction_evidence(prediction),
        "cotranslation-macro-standardized-rms",
    )


def test_arrest_release_canonicalizes_time_and_propagates_dwell_uncertainty():
    target = 1.0 - np.exp(-1.0)
    observations = _cotranslation_observations(
        _reference("timing-ms"),
        time_unit=MILLISECOND,
        dwell=(1000.0, 1000.0),
        dwell_error=(200.0, 200.0),
        values=(target, target),
    )
    law = CotranslationObservationLaw(
        "calibrated-arrest-release", _reference("release-law")
    )
    prediction = _cotranslation_prediction(
        observations, [1.0, 1.0], [0.1, 0.1], _PER_SECOND
    )
    assessment = assess_cotranslation_prediction(
        law,
        observations,
        prediction,
        maximum_standardized_rms=1.0,
        prediction_evidence=_prediction_evidence(prediction),
    )

    np.testing.assert_allclose(observations.measured_dwell_times, [1.0, 1.0])
    np.testing.assert_allclose(observations.dwell_time_standard_errors, [0.2, 0.2])
    expected = np.exp(-1.0) * np.sqrt(0.1**2 + 0.2**2)
    np.testing.assert_allclose(
        assessment.predictive_standard_errors, [expected, expected]
    )
    assert assessment.status == "ready-for-claim-evaluation"
