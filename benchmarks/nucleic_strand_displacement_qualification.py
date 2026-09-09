# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Leakage-controlled strand-displacement trace qualification workflow.

This executable uses an independently defined analytical two-state fixture to
exercise the complete reporter/effective/CTMC comparison. It deliberately marks
source admission, reporter calibration, and parameter identifiability evidence
inconclusive: the emitted scientific claims therefore cannot pass. Replace the
fixture with caller-admitted Zenodo 10090783 traces and reviewed evidence for a
real campaign; this module never downloads or fabricates experimental evidence.

Run from the worktree with:
python benchmarks/nucleic_strand_displacement_qualification.py
"""

from __future__ import annotations

import argparse
import hashlib
import json

import jax.numpy as jnp
import numpy as np
from _runtime import capture_environment

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
    prepare_secondary_kinetics,
    PreparedEffectiveDisplacementInference,
    PreparedMechanisticDisplacementInference,
    qualify_strand_displacement_models,
    ReporterCalibration,
    ReporterObservationModel,
    SecondaryEnergyModel,
    SecondaryKineticParameterPlan,
    SecondaryRateLaw,
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
from phydrax.uq import ExpBijector, ParameterSpace


_AVOGADRO = 6.02214076e23
_TRACE_SOURCE_CONTENT = b"independent-analytical-trace-definition"
_TRACE_SOURCE_MANIFEST = ReferenceArtifactManifest(
    "independent-analytical-trace-definition",
    checksum_algorithm="sha256",
    checksum=hashlib.sha256(_TRACE_SOURCE_CONTENT).hexdigest(),
    size_bytes=len(_TRACE_SOURCE_CONTENT),
    license_id="CC0-1.0",
    commercial_use_permitted=True,
    redistribution_permitted=True,
    training_use_permitted=True,
    export_permitted=True,
    export_classification="unrestricted",
    nondimensionalization={"identity": 1.0},
    uncertainty={"analytical_definition": 0.0},
    lineage_ids=("independent-analytical-trace-definition",),
)
_SOURCE_ID = _TRACE_SOURCE_MANIFEST.manifest_id
_SOURCE_USE = {
    "commercial_use": False,
    "redistribution": False,
    "training_use": True,
    "export": False,
}
_RAW_TRACE_CRITERION = ScientificMetricCriterion(
    "raw-trace-log-score",
    "at_least",
    -1e6,
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


def _energy_model():
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
        "independent-analytical-strand-displacement-ctmc",
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
        lineage_ids=("independently-defined-equations",),
    )
    return SecondaryEnergyModel.from_bytes(
        content,
        manifest,
        requested_use={
            "commercial_use": True,
            "redistribution": False,
            "training_use": True,
            "export": False,
        },
    )


def _campaign():
    definitions = (
        (
            "reporter-calibration",
            "family-calibration",
            "prep-calibration",
            "plate-calibration",
        ),
        ("rate-fit", "family-fit", "prep-fit", "plate-fit"),
        ("locked-a", "family-a", "prep-a", "plate-a"),
        ("locked-b", "family-b", "prep-b", "plate-b"),
    )
    cases = tuple(
        ScientificCase(
            case_id,
            family,
            canonical_fingerprint(("invader", "substrate")),
            "declared-condition",
            preparation,
            plate,
            (_SOURCE_ID,),
        )
        for case_id, family, preparation, plate in definitions
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", ("reporter-calibration",)),
            CampaignRole("model_selection", ("rate-fit",)),
            CampaignRole("locked_evaluation", ("locked-a", "locked-b")),
        ),
        criteria_ids=_STRAND_CRITERION_IDS,
    )


def _trace(case_id, family, preparation, plate, concentration, values):
    return FluorescenceTimeTrace(
        case_id,
        PlateWellIdentity(
            "analytical-experiment", plate, "A1", preparation, "replicate-1"
        ),
        np.linspace(0.0, 20.0, len(values)),
        values,
        np.zeros(len(values), dtype=bool),
        ("invader", "substrate"),
        (concentration, concentration),
        temperature_kelvin=300.0,
        condition_id="declared-condition",
        chemistry_direction="RNA>DNA",
        reporter_id="reporter-1",
        sequence_family_id=family,
        source_manifest_ids=(_SOURCE_ID,),
        injection_reference_seconds=0.0,
        saturation_threshold_intensity=1e9,
    )


def _claim(campaign, *, mechanistic):
    criteria = [_RAW_TRACE_CRITERION]
    stages = [
        "source-admission",
        "measurement-calibration",
        "predictive-calibration",
        "locked-prediction",
    ]
    if mechanistic:
        stages.append("parameter-identifiability")
        criteria.append(_MECHANISTIC_IMPROVEMENT_CRITERION)
    capability = "nucleic.strand-displacement"
    return ScientificClaimProfile(
        capability,
        SupportTuple(capability, {"chemistry_direction": "RNA>DNA"}),
        ("raw-fluorescence-trace",),
        ("declared-condition",),
        campaign.campaign_id,
        tuple(stages),
        tuple(criteria),
        "abstain-without-admitted-experimental-evidence",
        ("source-artifact", "observation-law", "model-parameters"),
        frozen_criteria_ids=campaign.criteria_ids,
    )


def _inconclusive_stage(campaign, criterion, *, model=None):
    subject_ids = (
        (campaign.campaign_id,)
        if model is None
        else (campaign.campaign_id, model.model_id, model.fit.fit_id)
    )
    return QualificationEvidence(
        "scientific",
        "inconclusive",
        subject_ids,
        build_id="analytical-benchmark-build",
        environment_id="local-benchmark-environment",
        backend="jax-default",
        topology="single-process",
        precision="runtime-default",
        reduction="deterministic",
        replay_id="analytical-benchmark-replay",
        criteria_ids=(criterion,),
        raw_artifact_ids=(
            (_SOURCE_ID,) if model is None else model.fit.source_manifest_ids
        ),
        reviewer_id="unreviewed-analytical-benchmark",
        issued_at=1,
        expires_at=2,
        reason="analytical-fixture-is-not-experimental-evidence",
    )


def run(*, state_capacity=8, channel_capacity=8):
    if state_capacity <= 0 or channel_capacity <= 0:
        raise ValueError("Declared CTMC capacities must be positive.")
    campaign = _campaign()
    concentration = 1e-7
    traces = (
        _trace(
            "reporter-calibration",
            "family-calibration",
            "prep-calibration",
            "plate-calibration",
            concentration,
            10.0 + np.asarray([0.0, 1.5, 2.8, 3.9, 4.8]),
        ),
        _trace(
            "rate-fit",
            "family-fit",
            "prep-fit",
            "plate-fit",
            concentration,
            10.0 + np.asarray([0.0, 1.6, 3.0, 4.1, 5.0]),
        ),
        _trace(
            "locked-a",
            "family-a",
            "prep-a",
            "plate-a",
            concentration,
            10.0 + np.asarray([0.0, 1.8, 3.3, 4.5, 5.5]),
        ),
        _trace(
            "locked-b",
            "family-b",
            "prep-b",
            "plate-b",
            concentration,
            10.0 + np.asarray([0.0, 1.3, 2.5, 3.5, 4.3]),
        ),
    )
    cohort = StrandDisplacementCohort(
        traces, campaign, (_SOURCE_ID,), "analytical-trace-cohort"
    )
    calibration = ReporterCalibration(
        "reporter-1",
        1e8,
        10.0,
        (0.5,),
        None,
        ("reporter-calibration",),
        campaign,
        source_manifests=(_TRACE_SOURCE_MANIFEST,),
        requested_use=_SOURCE_USE,
    )
    observation = ReporterObservationModel(
        calibration,
        0.75,
        ("reporter-calibration",),
        campaign,
    )
    effective_name = "rate_constant_per_molar_second"
    effective_plan = SecondaryKineticParameterPlan(
        (effective_name,),
        "RNA>DNA",
        300.0,
        "declared-condition",
        (_TRACE_SOURCE_MANIFEST,),
        ("analytical-effective-log-rate-prior",),
        requested_use=_SOURCE_USE,
    )
    effective_prepared = PreparedEffectiveDisplacementInference(
        (traces[0],),
        observation,
        effective_plan,
        ParameterSpace(
            {effective_name: jnp.asarray(jnp.log(1e6))},
            bijectors={effective_name: ExpBijector()},
            log_prior=lambda values: (
                -0.5 * ((jnp.log(values[effective_name]) - jnp.log(1e6)) / 1.0) ** 2
            ),
        ),
        ("invader", "substrate"),
        campaign,
        trace_source_manifests=(_TRACE_SOURCE_MANIFEST,),
        trace_requested_use=_SOURCE_USE,
    )
    effective = EffectiveDisplacementRateModel(
        fit_strand_displacement_model(
            (effective_prepared,),
            (traces[1],),
            campaign,
            model_selection_source_manifests=(_TRACE_SOURCE_MANIFEST,),
            model_selection_requested_use=_SOURCE_USE,
            gradient_tolerance=1e-5,
            laplace_damping=1e-8,
        )
    )
    prepared = prepare_secondary_kinetics(
        NucleicAcidConstruct(
            ("invader", "substrate"), ("A", "T"), ("RNA", "DNA"), (False, False)
        ),
        _energy_model(),
        AssociationConvention(
            mode="fixed_volume",
            standard_concentration=1000.0,
            volume=1.0 / (1000.0 * _AVOGADRO * concentration),
        ),
        SecondaryRateLaw("association_metropolis", 1.0, 1.0),
        temperature=300.0,
    )
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
    mechanistic_name = "rate_scale"
    mechanistic_plan = SecondaryKineticParameterPlan(
        (mechanistic_name,),
        "RNA>DNA",
        300.0,
        "declared-condition",
        (_TRACE_SOURCE_MANIFEST,),
        ("analytical-mechanistic-log-scale-prior",),
        requested_use=_SOURCE_USE,
    )
    mechanistic_prepared = PreparedMechanisticDisplacementInference(
        (traces[0],),
        observation,
        mechanistic_plan,
        ParameterSpace(
            {mechanistic_name: jnp.asarray(0.0)},
            bijectors={mechanistic_name: ExpBijector()},
            log_prior=lambda values: -0.5 * jnp.log(values[mechanistic_name]) ** 2,
        ),
        prepared,
        prepared.states[0],
        target,
        product_ids,
        ("invader", "substrate"),
        (concentration, concentration),
        "RNA>DNA",
        "declared-condition",
        campaign,
        trace_source_manifests=(_TRACE_SOURCE_MANIFEST,),
        trace_requested_use=_SOURCE_USE,
        state_capacity=state_capacity,
        channel_capacity=channel_capacity,
    )
    mechanistic = MechanisticDisplacementRateModel(
        fit_strand_displacement_model(
            (mechanistic_prepared,),
            (traces[1],),
            campaign,
            model_selection_source_manifests=(_TRACE_SOURCE_MANIFEST,),
            model_selection_requested_use=_SOURCE_USE,
            gradient_tolerance=1e-5,
            laplace_damping=1e-8,
        )
    )
    stage_evidence = (
        _inconclusive_stage(campaign, "source-admission"),
        _inconclusive_stage(campaign, "measurement-calibration"),
        _inconclusive_stage(campaign, "predictive-calibration", model=effective),
        _inconclusive_stage(campaign, "predictive-calibration", model=mechanistic),
        _inconclusive_stage(
            campaign,
            "parameter-identifiability",
            model=mechanistic,
        ),
    )
    result = qualify_strand_displacement_models(
        effective,
        mechanistic,
        observation,
        cohort,
        _claim(campaign, mechanistic=False),
        _claim(campaign, mechanistic=True),
        stage_evidence,
        build_id="analytical-benchmark-build",
        environment_id="local-benchmark-environment",
        backend="jax-default",
        topology="single-process",
        precision="runtime-default",
        reduction="deterministic",
        replay_id="analytical-benchmark-replay",
        reviewer_id="unreviewed-analytical-benchmark",
        issued_at=1,
        expires_at=2,
    )

    def evaluation(value):
        return {
            "model_id": value.model_id,
            "model_kind": value.model_kind,
            "execution_outcome": value.execution_evidence.outcome,
            "metric_values": dict(value.metric_values),
            "metric_units": dict(value.metric_units),
            "metric_aggregations": dict(value.metric_aggregations),
            "metric_gaps": list(value.metric_gaps),
            "family_scores": [
                {
                    "family_id": score.group_id,
                    "case_ids": list(score.case_ids),
                    "mean_log_score_per_observation": score.mean_log_score_per_observation,
                    "interval_coverage_95": score.interval_coverage_95,
                }
                for score in value.family_scores
            ],
            "preparation_scores": [
                {
                    "preparation_id": score.group_id,
                    "case_ids": list(score.case_ids),
                    "mean_log_score_per_observation": score.mean_log_score_per_observation,
                    "interval_coverage_95": score.interval_coverage_95,
                }
                for score in value.preparation_scores
            ],
            "full_trace_predictions": [
                {
                    "case_id": prediction.case_id,
                    "time_seconds": np.asarray(prediction.time_seconds).tolist(),
                    "mean_intensity": np.asarray(prediction.mean_intensity).tolist(),
                    "lower_95_intensity": np.asarray(
                        prediction.lower_95_intensity
                    ).tolist(),
                    "upper_95_intensity": np.asarray(
                        prediction.upper_95_intensity
                    ).tolist(),
                    "uncertainty_limitations": list(prediction.uncertainty_limitations),
                }
                for prediction in value.predictions
            ],
        }

    return {
        "scientific_scope": (
            "analytical workflow exercise only; experimental source admission, reporter "
            "calibration and parameter identifiability are intentionally inconclusive"
        ),
        "campaign_id": result.campaign_id,
        "comparison_id": result.comparison_id,
        "effective": evaluation(result.effective),
        "mechanistic": evaluation(result.mechanistic),
        "effective_claim_outcome": result.effective_claim_evidence.outcome,
        "mechanistic_claim_outcome": result.mechanistic_claim_evidence.outcome,
        "predictive_winner_model_id": result.predictive_winner_model_id,
        "environment": capture_environment().to_dict(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-capacity", type=int, default=8)
    parser.add_argument("--channel-capacity", type=int, default=8)
    arguments = parser.parse_args()
    print(
        json.dumps(
            run(
                state_capacity=arguments.state_capacity,
                channel_capacity=arguments.channel_capacity,
            ),
            indent=2,
        )
    )
