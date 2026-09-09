import hashlib
import json
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax._fingerprint import canonical_fingerprint
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.interchange import (
    FluorescenceTimeTrace,
    PlateWellIdentity,
    StrandDisplacementCohort,
)
from phydrax.applications.nucleic_acid_biophysics.secondary_kinetics import (
    AssociationConvention,
    EffectiveDisplacementRateModel,
    fit_strand_displacement_model,
    MechanisticDisplacementRateModel,
    predict_locked_fluorescence,
    prepare_secondary_kinetics,
    PreparedEffectiveDisplacementInference,
    PreparedMechanisticDisplacementInference,
    qualify_strand_displacement_models,
    ReporterCalibration,
    ReporterObservationModel,
    SecondaryEnergyModel,
    SecondaryKineticParameterPlan,
    SecondaryRateLaw,
    StrandDisplacementModelFit,
    trace_log_probability,
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
from phydrax.uq import find_map, fit_laplace, ParameterSpace


_AVOGADRO = 6.02214076e23
_SOURCE_CONTENT = b"independent-strand-test-source"
_SOURCE_MANIFEST = ReferenceArtifactManifest(
    "independent-strand-test-source",
    checksum_algorithm="sha256",
    checksum=hashlib.sha256(_SOURCE_CONTENT).hexdigest(),
    size_bytes=len(_SOURCE_CONTENT),
    license_id="CC0-1.0",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=True,
    export_permitted=True,
    export_classification="unrestricted",
    nondimensionalization={"identity": 1.0},
    uncertainty={"reporter-intensity": 0.5},
    lineage_ids=("independent-test-source",),
)
_SOURCE_ID = _SOURCE_MANIFEST.manifest_id
_SOURCE_USE = {
    "commercial_use": False,
    "redistribution": False,
    "training_use": True,
    "export": False,
}
_RAW_TRACE_CRITERION = ScientificMetricCriterion(
    "raw-trace-log-score",
    "at_least",
    -1e9,
    None,
    "natural-log-unit-per-observation",
    "pooled",
)
_MECHANISTIC_IMPROVEMENT_CRITERION = ScientificMetricCriterion(
    "mechanistic-log-score-improvement",
    "at_least",
    1e-12,
    None,
    "natural-log-unit-per-observation",
    "independent_unit_macro",
)
_STRAND_CRITERION_IDS = (
    _RAW_TRACE_CRITERION.criterion_id,
    _MECHANISTIC_IMPROVEMENT_CRITERION.criterion_id,
)


def _campaign():
    cases = tuple(
        ScientificCase(
            case_id,
            family,
            canonical_fingerprint(("invader", "substrate")),
            "condition-1",
            preparation,
            plate,
            (_SOURCE_ID,),
        )
        for case_id, family, preparation, plate in (
            (
                "calibration-case",
                "family-calibration",
                "prep-calibration",
                "plate-calibration",
            ),
            ("fit-case", "family-fit", "prep-fit", "plate-fit"),
            ("locked-case", "family-locked", "prep-locked", "plate-locked"),
        )
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("calibration-case",)),
            CampaignRole("model_selection", ("fit-case",)),
            CampaignRole("locked_evaluation", ("locked-case",)),
        ),
        criteria_ids=_STRAND_CRITERION_IDS,
    )


def _trace(case_id, family, preparation, plate, intensity, *, saturated=None):
    values = np.asarray(intensity, dtype=float)
    mask = (
        np.zeros(values.shape, dtype=bool) if saturated is None else np.asarray(saturated)
    )
    return FluorescenceTimeTrace(
        case_id,
        PlateWellIdentity("experiment-1", plate, "A1", preparation, "replicate-1"),
        (0.0, 1.0, 2.0),
        values,
        mask,
        ("invader", "substrate"),
        (1e-7, 1e-7),
        temperature_kelvin=300.0,
        condition_id="condition-1",
        chemistry_direction="RNA>DNA",
        reporter_id="reporter-1",
        sequence_family_id=family,
        source_manifest_ids=(_SOURCE_ID,),
        injection_reference_seconds=0.0,
        saturation_threshold_intensity=20.0,
    )


def _observation(campaign, *, covariance=None):
    calibration = ReporterCalibration(
        "reporter-1",
        1e8,
        10.0,
        (0.5,),
        covariance,
        ("calibration-case",),
        campaign,
        source_manifests=(_SOURCE_MANIFEST,),
        requested_use=_SOURCE_USE,
    )
    return ReporterObservationModel(
        calibration,
        0.5,
        ("calibration-case",),
        campaign,
    )


def _fit_traces():
    return (
        _trace(
            "calibration-case",
            "family-calibration",
            "prep-calibration",
            "plate-calibration",
            (10.0, 11.0, 12.0),
        ),
        _trace(
            "fit-case",
            "family-fit",
            "prep-fit",
            "plate-fit",
            (10.0, 11.5, 13.0),
        ),
    )


def _fit_artifact(prepared, selection, campaign, *, with_uncertainty=True):
    problem = prepared.posterior_problem()
    if with_uncertainty:
        result = fit_laplace(
            problem,
            prepared.parameter_space.initial,
            damping=1e-12,
            stationarity_tolerance=None,
        )
    else:
        result = find_map(problem, gradient_tolerance=1e-5)
    return StrandDisplacementModelFit(
        ((prepared, result),),
        (selection,),
        campaign,
        model_selection_source_manifests=(_SOURCE_MANIFEST,),
        model_selection_requested_use=_SOURCE_USE,
    )


def _effective(
    campaign,
    observation,
    *,
    with_uncertainty=True,
    trace_requested_use=_SOURCE_USE,
):
    calibration, selection = _fit_traces()
    plan = SecondaryKineticParameterPlan(
        ("rate_constant_per_molar_second",),
        "RNA>DNA",
        300.0,
        "condition-1",
        (_SOURCE_MANIFEST,),
        ("effective-log-rate-prior",),
        requested_use=_SOURCE_USE,
    )
    center = 1e6
    space = ParameterSpace(
        {"rate_constant_per_molar_second": jnp.asarray(center)},
        log_prior=lambda values: (
            -0.5 * ((values["rate_constant_per_molar_second"] - center) / 1e5) ** 2
        ),
    )
    prepared = PreparedEffectiveDisplacementInference(
        (calibration,),
        observation,
        plan,
        space,
        ("invader", "substrate"),
        campaign,
        trace_source_manifests=(_SOURCE_MANIFEST,),
        trace_requested_use=trace_requested_use,
    )
    return EffectiveDisplacementRateModel(
        _fit_artifact(
            prepared,
            selection,
            campaign,
            with_uncertainty=with_uncertainty,
        )
    )


def _prepared_ctmc():
    content = json.dumps(
        {
            "profile": "pair_loop",
            "chemistry": "DNA-RNA",
            "pairing_rule": "watson_crick",
            "temperature": 300.0,
            "energy_convention": "dimensionless_molar_G_over_RT",
            "minimum_hairpin_unpaired": 0,
            "pair_energies": {"AT": -1.0},
            "stack_energies": {},
            "hairpin_energies": {},
            "bulge_energies": {},
            "internal_energies": {},
            "multibranch": [0.0, 0.0, 0.0],
            "association_initiation": 0.0,
        },
        sort_keys=True,
    ).encode()
    manifest = ReferenceArtifactManifest(
        "analytical-strand-displacement-test-model",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"temperature_kelvin": 300.0},
        uncertainty={"analytical_definition": 0.0},
        lineage_ids=("independent-analytical-definition",),
    )
    energy = SecondaryEnergyModel.from_bytes(
        content,
        manifest,
        requested_use={
            "commercial_use": True,
            "redistribution": False,
            "training_use": True,
            "export": False,
        },
    )
    concentration = 1e-7
    return prepare_secondary_kinetics(
        NucleicAcidConstruct(
            ("invader", "substrate"), ("A", "T"), ("RNA", "DNA"), (False, False)
        ),
        energy,
        AssociationConvention(
            mode="fixed_volume",
            standard_concentration=1000.0,
            volume=1.0 / (1000.0 * _AVOGADRO * concentration),
        ),
        SecondaryRateLaw("association_metropolis", 1.0, 1.0),
        temperature=300.0,
    )


def _mechanistic(campaign, observation, *, state_capacity, with_uncertainty=True):
    calibration, selection = _fit_traces()
    prepared = _prepared_ctmc()
    target = prepared.joined_target(("invader", "substrate"))
    product_ids = tuple(
        sorted(
            state.fingerprint()
            for state, selected in zip(
                prepared.states, np.asarray(target.mask), strict=True
            )
            if selected
        )
    )
    plan = SecondaryKineticParameterPlan(
        ("rate_scale",),
        "RNA>DNA",
        300.0,
        "condition-1",
        (_SOURCE_MANIFEST,),
        ("mechanistic-log-rate-scale-prior",),
        requested_use=_SOURCE_USE,
    )
    space = ParameterSpace(
        {"rate_scale": jnp.asarray(1.0)},
        log_prior=lambda values: -0.5 * ((values["rate_scale"] - 1.0) / 0.2) ** 2,
    )
    inference = PreparedMechanisticDisplacementInference(
        (calibration,),
        observation,
        plan,
        space,
        prepared,
        prepared.states[0],
        target,
        product_ids,
        ("invader", "substrate"),
        (1e-7, 1e-7),
        "RNA>DNA",
        "condition-1",
        campaign,
        trace_source_manifests=(_SOURCE_MANIFEST,),
        trace_requested_use=_SOURCE_USE,
        state_capacity=state_capacity,
        channel_capacity=prepared.process.num_channels,
    )
    return MechanisticDisplacementRateModel(
        _fit_artifact(
            inference,
            selection,
            campaign,
            with_uncertainty=with_uncertainty,
        )
    )


def _cohort(campaign, locked, cohort_id):
    return StrandDisplacementCohort(
        (*_fit_traces(), locked), campaign, (_SOURCE_ID,), cohort_id
    )


def _profile(campaign, *, mechanistic):
    criteria = [_RAW_TRACE_CRITERION]
    stages = [
        "source-admission",
        "measurement-calibration",
        "predictive-calibration",
        "locked-prediction",
    ]
    if mechanistic:
        criteria.append(_MECHANISTIC_IMPROVEMENT_CRITERION)
        stages.append("parameter-identifiability")
    capability = "nucleic.strand-displacement"
    return ScientificClaimProfile(
        capability,
        SupportTuple(capability, {"chemistry_direction": "RNA>DNA"}),
        ("raw-fluorescence-trace",),
        ("condition-1",),
        campaign.campaign_id,
        tuple(stages),
        tuple(criteria),
        "abstain-outside-declared-support",
        ("model-parameters", "source-artifact", "observation-law"),
        frozen_criteria_ids=campaign.criteria_ids,
    )


def _stage(campaign, criterion, *, model=None):
    subject_ids = (
        (campaign.campaign_id,)
        if model is None
        else (campaign.campaign_id, model.model_id, model.fit.fit_id)
    )
    return QualificationEvidence(
        "scientific",
        "passed",
        subject_ids,
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id="test-replay",
        criteria_ids=(criterion,),
        raw_artifact_ids=(
            (_SOURCE_ID,) if model is None else model.fit.source_manifest_ids
        ),
        reviewer_id="test-reviewer",
        issued_at=10,
        expires_at=20,
        reason="independent-test-evidence",
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
    )


def test_native_fit_rejects_trace_admission_without_training_rights():
    campaign = _campaign()
    observation = _observation(campaign)
    calibration, selection = _fit_traces()
    plan = SecondaryKineticParameterPlan(
        ("rate_constant_per_molar_second",),
        "RNA>DNA",
        300.0,
        "condition-1",
        (_SOURCE_MANIFEST,),
        ("effective-log-rate-prior",),
        requested_use=_SOURCE_USE,
    )
    prepared = PreparedEffectiveDisplacementInference(
        (calibration,),
        observation,
        plan,
        ParameterSpace(
            {"rate_constant_per_molar_second": jnp.asarray(1e6)},
            log_prior=lambda values: (
                -0.5 * ((values["rate_constant_per_molar_second"] - 1e6) / 1e5) ** 2
            ),
        ),
        ("invader", "substrate"),
        campaign,
        trace_source_manifests=(_SOURCE_MANIFEST,),
        trace_requested_use={**_SOURCE_USE, "training_use": False},
    )
    with pytest.raises(ValueError, match="did not authorize training use"):
        fit_strand_displacement_model(
            (prepared,),
            (selection,),
            campaign,
            model_selection_source_manifests=(_SOURCE_MANIFEST,),
            model_selection_requested_use=_SOURCE_USE,
        )


def test_reporter_calibration_isolated_and_full_trace_prediction_is_condition_bounded():
    campaign = _campaign()
    with pytest.raises(ValueError, match="calibration-role"):
        ReporterCalibration(
            "reporter-1",
            1e8,
            10.0,
            (),
            None,
            ("locked-case",),
            campaign,
            source_manifests=(_SOURCE_MANIFEST,),
            requested_use=_SOURCE_USE,
        )
    trace = _trace(
        "locked-case", "family-locked", "prep-locked", "plate-locked", (10, 12, 14)
    )
    observation = _observation(campaign)
    effective = _effective(campaign, observation)
    prediction = predict_locked_fluorescence(effective, observation, (trace,))
    assert prediction.case_ids == ("locked-case",)
    assert prediction.predictions[0].mean_intensity.shape == trace.time_seconds.shape
    assert trace_log_probability(trace, prediction.predictions[0]).shape == ()
    with pytest.raises(TypeError, match="StrandDisplacementModelFit"):
        EffectiveDisplacementRateModel(1e6)

    outside = FluorescenceTimeTrace(
        trace.case_id,
        trace.identity,
        trace.time_seconds,
        trace.intensity,
        trace.saturation_mask,
        trace.construct_ids,
        trace.initial_concentrations_molar,
        temperature_kelvin=trace.temperature_kelvin,
        condition_id=trace.condition_id,
        chemistry_direction="DNA>DNA",
        reporter_id=trace.reporter_id,
        sequence_family_id=trace.sequence_family_id,
        source_manifest_ids=trace.source_manifest_ids,
        injection_reference_seconds=trace.injection_reference_seconds,
        saturation_threshold_intensity=trace.saturation_threshold_intensity,
    )
    with pytest.raises(ValueError, match="chemistry-direction"):
        predict_locked_fluorescence(effective, observation, (outside,))


def test_trace_likelihood_uses_calibration_induced_time_correlation_and_censoring_rules():
    campaign = _campaign()
    trace = _trace(
        "locked-case", "family-locked", "prep-locked", "plate-locked", (10.0, 11.0, 12.0)
    )
    observation = _observation(
        campaign, covariance=jnp.diag(jnp.asarray([0.25, 1e10, 0.01]))
    )
    effective = _effective(campaign, observation)
    predicted = predict_locked_fluorescence(effective, observation, (trace,)).predictions[
        0
    ]
    reporter_only = observation.prediction(
        trace,
        effective.product_concentration(trace),
        forward_model_id=effective.model_id,
    )
    assert predicted.includes_model_uncertainty
    assert predicted.includes_reporter_uncertainty
    assert jnp.all(
        predicted.standard_deviation_intensity
        >= reporter_only.standard_deviation_intensity
    )
    assert jnp.any(
        predicted.standard_deviation_intensity
        > reporter_only.standard_deviation_intensity
    )
    covariance = (
        0.5**2 * np.eye(3)
        + np.asarray(predicted.epistemic_sensitivity)
        @ np.asarray(predicted.epistemic_parameter_covariance)
        @ np.asarray(predicted.epistemic_sensitivity).T
    )
    residual = np.asarray(trace.intensity - predicted.mean_intensity)
    sign, logdet = np.linalg.slogdet(covariance)
    expected = -0.5 * (
        3 * math.log(2 * math.pi)
        + logdet
        + residual @ np.linalg.solve(covariance, residual)
    )
    assert sign == 1
    assert float(trace_log_probability(trace, predicted)) == pytest.approx(
        expected, rel=2e-5
    )

    censored = _trace(
        "locked-case",
        "family-locked",
        "prep-locked",
        "plate-locked",
        (10.0, 11.0, np.nan),
        saturated=(False, False, True),
    )
    with pytest.raises(ValueError, match="censored samples"):
        trace_log_probability(
            censored,
            predict_locked_fluorescence(effective, observation, (censored,)).predictions[
                0
            ],
        )
    conditional_observation = _observation(campaign)
    conditional_model = _effective(
        campaign, conditional_observation, with_uncertainty=False
    )
    independent_prediction = predict_locked_fluorescence(
        conditional_model, conditional_observation, (censored,)
    ).predictions[0]
    assert jnp.isfinite(trace_log_probability(censored, independent_prediction))


def test_qualification_returns_inconclusive_for_declared_ctmc_capacity_without_leakage():
    campaign = _campaign()
    locked = _trace(
        "locked-case", "family-locked", "prep-locked", "plate-locked", (10, 12, 14)
    )
    observation = _observation(
        campaign,
        covariance=jnp.diag(jnp.asarray([0.25, 1e10, 0.01])),
    )
    effective = _effective(campaign, observation)
    mechanistic = _mechanistic(campaign, observation, state_capacity=1)
    cohort = _cohort(campaign, locked, "cohort-1")
    assert "enumerated-state-capacity-exceeded" in mechanistic.support_reasons(locked)
    result = qualify_strand_displacement_models(
        effective,
        mechanistic,
        observation,
        cohort,
        _profile(campaign, mechanistic=False),
        _profile(campaign, mechanistic=True),
        (
            _stage(campaign, "source-admission"),
            _stage(campaign, "measurement-calibration"),
            _stage(campaign, "predictive-calibration", model=effective),
            _stage(campaign, "predictive-calibration", model=mechanistic),
            _stage(campaign, "parameter-identifiability", model=mechanistic),
        ),
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id="test-replay",
        reviewer_id="test-reviewer",
        issued_at=10,
        expires_at=20,
    )
    assert result.effective.execution_evidence.outcome == "passed"
    assert result.mechanistic.execution_evidence.outcome == "inconclusive"
    assert "enumerated-state-capacity-exceeded" in result.mechanistic.metric_gaps

    base_profile = _profile(campaign, mechanistic=False)
    forged_profile = ScientificClaimProfile(
        base_profile.capability_name,
        base_profile.support,
        base_profile.observable_ids,
        base_profile.condition_domain_ids,
        base_profile.campaign_id,
        base_profile.required_stage_ids,
        base_profile.criteria,
        base_profile.abstention_policy_id,
        base_profile.invalidation_triggers,
        frozen_criteria_ids=(*campaign.criteria_ids, "caller-added-criterion"),
    )
    with pytest.raises(ValueError, match="frozen campaign criteria"):
        qualify_strand_displacement_models(
            effective,
            mechanistic,
            observation,
            cohort,
            forged_profile,
            _profile(campaign, mechanistic=True),
            (),
            build_id="test-build",
            environment_id="test-environment",
            backend="cpu",
            topology="single-process",
            precision="float64",
            reduction="deterministic",
            replay_id="test-replay",
            reviewer_id="test-reviewer",
            issued_at=10,
            expires_at=20,
        )


def test_unknown_reporter_covariance_and_missing_injection_are_kinetic_ineligible():
    campaign = _campaign()
    locked = _trace(
        "locked-case", "family-locked", "prep-locked", "plate-locked", (10, 12, 14)
    )
    observation = _observation(campaign)
    effective = _effective(campaign, observation)
    mechanistic = _mechanistic(campaign, observation, state_capacity=8)
    cohort = _cohort(campaign, locked, "cohort-uncertainty")
    stages = (
        _stage(campaign, "source-admission"),
        _stage(campaign, "measurement-calibration"),
        _stage(campaign, "predictive-calibration", model=effective),
        _stage(campaign, "predictive-calibration", model=mechanistic),
        _stage(campaign, "parameter-identifiability", model=mechanistic),
    )
    unknown = qualify_strand_displacement_models(
        effective,
        mechanistic,
        observation,
        cohort,
        _profile(campaign, mechanistic=False),
        _profile(campaign, mechanistic=True),
        stages,
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id="test-replay",
        reviewer_id="test-reviewer",
        issued_at=10,
        expires_at=20,
    )
    assert unknown.effective.execution_evidence.outcome == "inconclusive"
    assert set(unknown.effective.execution_evidence.criteria_ids) == {
        "locked-prediction",
        "predictive-calibration",
    }

    unreferenced = FluorescenceTimeTrace(
        locked.case_id,
        locked.identity,
        locked.time_seconds,
        locked.intensity,
        locked.saturation_mask,
        locked.construct_ids,
        locked.initial_concentrations_molar,
        temperature_kelvin=locked.temperature_kelvin,
        condition_id=locked.condition_id,
        chemistry_direction=locked.chemistry_direction,
        reporter_id=locked.reporter_id,
        sequence_family_id=locked.sequence_family_id,
        source_manifest_ids=locked.source_manifest_ids,
        injection_reference_seconds=None,
        saturation_threshold_intensity=locked.saturation_threshold_intensity,
    )
    calibrated_observation = _observation(
        campaign,
        covariance=jnp.diag(jnp.asarray([0.25, 1e10, 0.01])),
    )
    calibrated_effective = _effective(campaign, calibrated_observation)
    calibrated_mechanistic = _mechanistic(
        campaign, calibrated_observation, state_capacity=8
    )
    calibrated_stages = (
        _stage(campaign, "source-admission"),
        _stage(campaign, "measurement-calibration"),
        _stage(campaign, "predictive-calibration", model=calibrated_effective),
        _stage(campaign, "predictive-calibration", model=calibrated_mechanistic),
        _stage(
            campaign,
            "parameter-identifiability",
            model=calibrated_mechanistic,
        ),
    )
    no_reference = qualify_strand_displacement_models(
        calibrated_effective,
        calibrated_mechanistic,
        calibrated_observation,
        _cohort(campaign, unreferenced, "cohort-no-injection"),
        _profile(campaign, mechanistic=False),
        _profile(campaign, mechanistic=True),
        calibrated_stages,
        build_id="test-build",
        environment_id="test-environment",
        backend="cpu",
        topology="single-process",
        precision="float64",
        reduction="deterministic",
        replay_id="test-replay",
        reviewer_id="test-reviewer",
        issued_at=10,
        expires_at=20,
    )
    assert no_reference.effective.execution_evidence.outcome == "inconclusive"
    assert "injection-reference-unknown" in no_reference.effective.metric_gaps


def test_unequal_mass_action_is_finite_and_censored_likelihood_jits():
    campaign = _campaign()
    trace = FluorescenceTimeTrace(
        "locked-case",
        PlateWellIdentity(
            "experiment-1", "plate-locked", "A1", "prep-locked", "replicate-1"
        ),
        (0.0, 1e12, 2e12),
        (10.0, 11.0, np.nan),
        (False, False, True),
        ("invader", "substrate"),
        (2e-7, 1e-7),
        temperature_kelvin=300.0,
        condition_id="condition-1",
        chemistry_direction="RNA>DNA",
        reporter_id="reporter-1",
        sequence_family_id="family-locked",
        source_manifest_ids=(_SOURCE_ID,),
        injection_reference_seconds=0.0,
        saturation_threshold_intensity=20.0,
    )
    observation = _observation(campaign)
    effective = _effective(campaign, observation, with_uncertainty=False)
    product = effective.product_concentration(trace)
    assert jnp.all(jnp.isfinite(product))
    assert product[-1] == pytest.approx(1e-7)
    predicted = predict_locked_fluorescence(effective, observation, (trace,)).predictions[
        0
    ]
    assert jnp.isfinite(jax.jit(lambda: trace_log_probability(trace, predicted))())


def test_reporter_observation_refuses_calibration_from_another_campaign():
    campaign = _campaign()
    calibration = ReporterCalibration(
        "reporter-1",
        1e8,
        10.0,
        (0.5,),
        jnp.diag(jnp.asarray([0.25, 1e10, 0.01])),
        ("calibration-case",),
        campaign,
        source_manifests=(_SOURCE_MANIFEST,),
        requested_use=_SOURCE_USE,
    )
    changed_cases = tuple(
        ScientificCase(
            case.case_id,
            case.independent_unit_id,
            case.construct_id + "-changed",
            case.condition_id,
            case.preparation_id,
            case.batch_id,
            case.source_manifest_ids,
        )
        for case in campaign.cases
    )
    changed_campaign = ScientificCampaign(changed_cases, campaign.roles)
    with pytest.raises(ValueError, match="share one campaign"):
        ReporterObservationModel(
            calibration,
            0.5,
            ("calibration-case",),
            changed_campaign,
        )
