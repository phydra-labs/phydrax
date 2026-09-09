# Copyright © 2026 PHYDRA, Inc. All rights reserved.

import hashlib

import pytest

from phydrax.applications.protein_folding.interchange import (
    admit_megascale_processed_table,
    apply_mutation_code,
    MEGASCALE_DOI,
    MEGASCALE_RECORD_ID,
    MEGASCALE_RECORD_VERSION,
    parse_mutation_code,
    prepare_protein_stability_cohort,
    ProteinStabilityMeasurement,
)
from phydrax.applications.protein_folding.stability import (
    AminoAcidScalarDefinition,
    DoubleMutantCase,
    DoubleMutationFeatures,
    fit_global_substitution_baseline,
    fit_protein_feature_transform,
    fit_regularized_environment_model,
    fit_regularized_pair_interaction_model,
    prepare_double_mutant_campaign,
    protein_mutation_features,
    protein_stability_claim_profile,
    ProteinResidueEnvironment,
    ProteinStabilityModelSelectionRecord,
    ProteinStabilityThresholds,
    qualify_double_mutant_challenge,
    qualify_protein_stability,
)
from phydrax.qualification import (
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificMetricCriterion,
    SupportTuple,
)


_AA = "ACDEFGHIKLMNPQRSTVWY"
_FEATURE_USE = {
    "commercial_use": False,
    "redistribution": False,
    "training_use": True,
    "export": False,
}
_THRESHOLDS = ProteinStabilityThresholds(2.0, 0.8, 5.0, 0.0, 0.5)
_STABILITY_CRITERIA = (
    ScientificMetricCriterion(
        "protein-stability-family-macro-mae",
        "at_most",
        None,
        2.0,
        "kcal/mol",
        "independent_unit_macro",
    ),
    ScientificMetricCriterion(
        "protein-stability-interval-coverage",
        "between",
        0.8,
        1.0,
        "fraction",
        "independent_unit_macro",
    ),
    ScientificMetricCriterion(
        "protein-stability-mean-interval-width",
        "at_most",
        None,
        5.0,
        "kcal/mol",
        "independent_unit_macro",
    ),
    ScientificMetricCriterion(
        "protein-stability-baseline-benefit",
        "at_least",
        0.0,
        None,
        "kcal/mol",
        "independent_unit_macro",
    ),
    ScientificMetricCriterion(
        "protein-stability-prediction-coverage",
        "at_least",
        0.5,
        None,
        "fraction",
        "independent_unit_macro",
    ),
)
_DOUBLE_CRITERIA = (
    ScientificMetricCriterion(
        "protein-double-mutant-corrected-pair-macro-mae",
        "at_most",
        None,
        2.0,
        "kcal/mol",
        "independent_unit_macro",
    ),
    ScientificMetricCriterion(
        "protein-double-mutant-benefit-lower-95",
        "at_least",
        0.0,
        None,
        "kcal/mol",
        "independent_unit_macro",
    ),
)
_DOUBLE_CRITERION_IDS = tuple(criterion.criterion_id for criterion in _DOUBLE_CRITERIA)
_STABILITY_CRITERION_IDS = tuple(
    criterion.criterion_id for criterion in _STABILITY_CRITERIA
)


def _source(tmp_path, *, uncertainty_reported=True):
    payload = (
        b"name,aa_seq,mut_type,WT_name,WT_cluster,ddG_ML\n"
        b"family-a-A1C,CAAA,A1C,family-a,family-a-cluster,0.5\n"
        b"family-b-A1D,DAAA,A1D,family-b,family-b-cluster,-0.25\n"
        b"family-c-A1E,EAAA,A1E,family-c,family-c-cluster,0.1\n"
        b"family-d-A1F,FAAA,A1F,family-d,family-d-cluster,0.2\n"
    )
    path = tmp_path / "processed.csv"
    path.write_bytes(payload)
    manifest = ReferenceArtifactManifest(
        path.name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC-BY-4.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"stability-kcal-per-mol": 1.0},
        uncertainty={"standard-error-kcal-per-mol": 0.1},
        lineage_ids=(
            "independent-test-fixture",
            MEGASCALE_RECORD_ID,
            MEGASCALE_RECORD_VERSION,
            f"doi:{MEGASCALE_DOI}",
        ),
    )
    names = (
        "family-a-A1C",
        "family-b-A1D",
        "family-c-A1E",
        "family-d-A1F",
    )
    uncertainty_content = b"independent-standard-errors"
    uncertainty_manifest = ReferenceArtifactManifest(
        "independent-standard-errors",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(uncertainty_content).hexdigest(),
        size_bytes=len(uncertainty_content),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"kcal-per-mol": 1.0},
        uncertainty={"standard-error-kcal-per-mol": 0.1},
        lineage_ids=tuple(
            f"protein-stability-standard-error:{name}:{float(0.1).hex()}"
            for name in names
        ),
    )
    source = admit_megascale_processed_table(
        path,
        manifest,
        selected_names=names,
        wt_sequence_by_background={
            "family-a": "AAAA",
            "family-b": "AAAA",
            "family-c": "AAAA",
            "family-d": "AAAA",
        },
        censoring_by_name={name: "none" for name in names},
        quality_flags_by_name={name: ("independent-test-fixture",) for name in names},
        library_id="library",
        condition_id="condition",
        standard_error_by_name={
            name: (0.1 if uncertainty_reported else None) for name in names
        },
        standard_error_manifest=(uncertainty_manifest if uncertainty_reported else None),
        training_use=True,
    )
    return source


def _features(source, measurements=None):
    scalar = {residue: float(index) for index, residue in enumerate(_AA)}
    scalars = AminoAcidScalarDefinition(
        scalar,
        {residue: 0.0 for residue in _AA},
        {residue: -float(index) for index, residue in enumerate(_AA)},
        source_manifests=(source.source_manifest,),
        requested_use=_FEATURE_USE,
    )
    features = []
    selected = source.measurements if measurements is None else tuple(measurements)
    for measurement in selected:
        residue_position = parse_mutation_code(measurement.mutation_code)[0][1]
        environment = ProteinResidueEnvironment(
            residue_position,
            "H",
            0.25,
            4.0,
            {
                "aliphatic": 1.0,
                "aromatic": 0.0,
                "polar": 2.0,
                "positive": 0.0,
                "negative": 0.0,
            },
            -1.0,
            1.0,
            hypothesis_id=f"hypothesis-{measurement.background_id}",
            residue_mapping_id=f"mapping-{measurement.background_id}",
            source_manifests=(source.source_manifest,),
            requested_use=_FEATURE_USE,
        )
        features.append(
            protein_mutation_features(
                measurement,
                environment,
                scalars,
                feature_definition_id="transparent-v1",
                requested_use=_FEATURE_USE,
            )
        )
    return tuple(features)


def _double_measurement(source, measurement_id, family_id, mutation_code, value):
    template = source.measurements[0]
    uncertainty_manifest = template.uncertainty_source_manifest
    return ProteinStabilityMeasurement(
        measurement_id=measurement_id,
        domain_id=family_id,
        domain_family_id=family_id,
        background_id=f"background-{family_id}",
        sequence="AAAA",
        mutation_code=mutation_code,
        mutation_order=len(parse_mutation_code(mutation_code)),
        assay_channel=template.assay_channel,
        library_id="double-mutant-library",
        condition_id=template.condition_id,
        value_kcal_per_mol=value,
        standard_error_kcal_per_mol=0.1,
        censoring="none",
        quality_flags=("independent-test-fixture",),
        source_manifest_id=source.source_manifest.manifest_id,
        observable="delta_delta_g",
        sign_convention=template.sign_convention,
        source_mutant_sequence=apply_mutation_code("AAAA", mutation_code),
        shared_wt_id=f"shared-wt-{family_id}",
        source_fields=(("fixture-case-id", measurement_id),),
        inference_lineage=("independent-test-fixture",),
        uncertainty_source_manifest_id=uncertainty_manifest.manifest_id,
        source_manifest=source.source_manifest,
        uncertainty_source_manifest=uncertainty_manifest,
    )


def _double_case(source, case_id, family_id, distance, coupling):
    first = _double_measurement(source, f"{case_id}-A1C", family_id, "A1C", 0.25)
    second = _double_measurement(source, f"{case_id}-A2D", family_id, "A2D", -0.1)
    double = _double_measurement(
        source,
        case_id,
        family_id,
        "A1C:A2D",
        first.value_kcal_per_mol + second.value_kcal_per_mol + coupling,
    )
    first_feature, second_feature = _features(source, (first, second))
    pair_features = DoubleMutationFeatures(
        first_feature,
        second_feature,
        case_id=case_id,
        pair_unit_id=double.pair_unit_id,
        pair_distance=distance,
        pair_context_id="shared-double-mutant-context",
    )
    return (
        DoubleMutantCase(double, first, second, pair_features),
        first_feature,
        second_feature,
    )


def test_processed_stability_admission_preserves_grouped_locked_evaluation(tmp_path):
    source = _source(tmp_path)
    role_by_group = {
        source.measurements[0].independent_group_id: "calibration",
        source.measurements[1].independent_group_id: "calibration",
        source.measurements[2].independent_group_id: "model_selection",
        source.measurements[3].independent_group_id: "locked_evaluation",
    }
    cohort = prepare_protein_stability_cohort(
        source,
        role_by_group,
        preparation_id_by_measurement={
            item.measurement_id: item.background_id for item in source.measurements
        },
        batch_id_by_measurement={
            item.measurement_id: item.background_id for item in source.measurements
        },
        criteria_ids=_STABILITY_CRITERION_IDS,
    )
    features = _features(source)
    with pytest.raises(ValueError, match="exactly cover every eligible calibration"):
        fit_protein_feature_transform(features[1:], cohort)
    transform = fit_protein_feature_transform(features, cohort)
    baseline = fit_global_substitution_baseline(features, cohort, transform, ridge=1.0)
    environment = fit_regularized_environment_model(
        features,
        cohort,
        transform,
        ridge=1.0,
        family_effect_scale=1.0,
    )
    selection = ProteinStabilityModelSelectionRecord(
        cohort,
        features,
        (baseline, environment),
        candidate_hyperparameters={
            baseline.fit_id: {"ridge": 1.0},
            environment.fit_id: {"ridge": 1.0, "family_effect_scale": 1.0},
        },
    )
    evidence_inputs = dict(
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="two-independent-families",
        precision="float64",
        reduction="family-macro",
        replay_id="replay",
        criteria_ids=("parameter-identifiability",),
        raw_artifact_ids=(source.source_manifest.manifest_id,),
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=2,
        reason="native-posterior-identifiable",
    )
    unbound_identifiability = QualificationEvidence(
        "scientific",
        "passed",
        (cohort.campaign.campaign_id,),
        **evidence_inputs,
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
    )
    bound_identifiability = QualificationEvidence(
        "scientific",
        "passed",
        (
            cohort.campaign.campaign_id,
            selection.chosen_fit.predictor.model_id,
            selection.chosen_fit_id,
            selection.selection_id,
        ),
        **evidence_inputs,
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
    )
    result = qualify_protein_stability(
        cohort,
        features,
        baseline,
        selection,
        SupportTuple(
            "protein.mutation-stability-prediction",
            {"assay": "test", "scope": "two-independent-families"},
        ),
        _THRESHOLDS,
        stage_evidence=(unbound_identifiability, bound_identifiability),
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="two-independent-families",
        precision="float64",
        reduction="family-macro",
        replay_id="replay",
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=2,
    )

    assert [item.case_id for item in result.predictions] == ["family-d-A1F"]
    assert selection.model_selection_case_ids == ("family-c-A1E",)
    assert selection.chosen_fit_id in selection.candidate_fit_ids
    assert result.selected_model_fit_id == selection.chosen_fit_id
    assert result.model_selection_id == selection.selection_id
    assert result.evidence.outcome != "passed"
    assert result.metrics.evaluated_case_count == 1
    assert result.predictions[0].valid
    assert unbound_identifiability.evidence_id not in {
        item.evidence_id for item in result.stage_evidence
    }
    assert bound_identifiability.evidence_id in {
        item.evidence_id for item in result.stage_evidence
    }


def test_double_campaign_rejects_rekeyed_components_and_locked_family_leakage(
    tmp_path,
):
    source = _source(tmp_path)
    generated = (
        _double_case(source, "double-cal-a", "pair-cal-a", 4.0, 0.1),
        _double_case(source, "double-cal-b", "pair-cal-b", 8.0, -0.2),
        _double_case(source, "double-lock-a", "family-a-cluster", 5.0, 0.15),
        _double_case(source, "double-lock-b", "pair-lock-b", 9.0, -0.1),
    )
    cases = tuple(item[0] for item in generated)
    role_by_pair = {
        item.pair_features.pair_unit_id: (
            "calibration" if index < 2 else "locked_evaluation"
        )
        for index, item in enumerate(cases)
    }
    preparation = {
        item.double_measurement.measurement_id: item.double_measurement.background_id
        for item in cases
    }
    campaign = prepare_double_mutant_campaign(
        cases,
        role_by_pair,
        preparation_id_by_case=preparation,
        batch_id_by_case=preparation,
        criteria_ids=_DOUBLE_CRITERION_IDS,
    )
    assert set(_DOUBLE_CRITERION_IDS) == set(campaign.criteria_ids)

    single_roles = {
        source.measurements[0].independent_group_id: "calibration",
        source.measurements[1].independent_group_id: "calibration",
        source.measurements[2].independent_group_id: "model_selection",
        source.measurements[3].independent_group_id: "locked_evaluation",
    }
    single_preparation = {
        item.measurement_id: item.background_id for item in source.measurements
    }
    single_cohort = prepare_protein_stability_cohort(
        source,
        single_roles,
        preparation_id_by_measurement=single_preparation,
        batch_id_by_measurement=single_preparation,
        criteria_ids=_STABILITY_CRITERION_IDS,
    )
    single_features = _features(source)
    transform = fit_protein_feature_transform(single_features, single_cohort)
    single_fit = fit_global_substitution_baseline(
        single_features, single_cohort, transform, ridge=1.0
    )
    pair_fit = fit_regularized_pair_interaction_model(
        cases,
        campaign,
        single_model_id=single_fit.predictor.model_id,
        ridge=1.0,
    )
    assert pair_fit.successful

    locked = cases[2:]
    first = {
        item.double_measurement.measurement_id: generated[index + 2][1]
        for index, item in enumerate(locked)
    }
    second = {
        item.double_measurement.measurement_id: generated[index + 2][2]
        for index, item in enumerate(locked)
    }
    qualification = dict(
        maximum_corrected_pair_macro_mae_kcal_per_mol=2.0,
        minimum_benefit_lower_95_kcal_per_mol=0.0,
        build_id="build",
        environment_id="environment",
        backend="cpu",
        topology="held-out-pairs",
        precision="float64",
        reduction="pair-macro",
        replay_id="replay",
        reviewer_id="reviewer",
        issued_at=1,
        expires_at=2,
    )
    rekeyed = dict(first)
    left, right = tuple(rekeyed)
    rekeyed[left], rekeyed[right] = rekeyed[right], rekeyed[left]
    with pytest.raises(ValueError, match="exact ordered component"):
        qualify_double_mutant_challenge(
            cases,
            rekeyed,
            second,
            campaign,
            single_fit.predictor,
            pair_fit.predictor,
            {},
            SupportTuple(
                "protein.double-mutant-interaction-prediction",
                {"assay": "test"},
            ),
            **qualification,
        )
    with pytest.raises(ValueError, match="training families"):
        qualify_double_mutant_challenge(
            cases,
            first,
            second,
            campaign,
            single_fit.predictor,
            pair_fit.predictor,
            {},
            SupportTuple(
                "protein.double-mutant-interaction-prediction",
                {"assay": "test"},
            ),
            **qualification,
        )


def test_model_selection_rejects_relabelled_hyperparameters_and_cross_campaign_fit(
    tmp_path,
):
    source = _source(tmp_path)
    roles = {
        source.measurements[0].independent_group_id: "calibration",
        source.measurements[1].independent_group_id: "calibration",
        source.measurements[2].independent_group_id: "model_selection",
        source.measurements[3].independent_group_id: "locked_evaluation",
    }
    preparation = {
        item.measurement_id: item.background_id for item in source.measurements
    }
    cohort = prepare_protein_stability_cohort(
        source,
        roles,
        preparation_id_by_measurement=preparation,
        batch_id_by_measurement=preparation,
        criteria_ids=_STABILITY_CRITERION_IDS,
    )
    features = _features(source)
    transform = fit_protein_feature_transform(features, cohort)
    baseline = fit_global_substitution_baseline(features, cohort, transform, ridge=1.0)
    environment = fit_regularized_environment_model(
        features,
        cohort,
        transform,
        ridge=1.0,
        family_effect_scale=1.0,
    )
    candidates = (baseline, environment)
    hyperparameters = {
        baseline.fit_id: {"ridge": 1.0},
        environment.fit_id: {"ridge": 1.0, "family_effect_scale": 1.0},
    }
    with pytest.raises(ValueError, match="exactly match the fitted model"):
        ProteinStabilityModelSelectionRecord(
            cohort,
            features,
            candidates,
            candidate_hyperparameters={
                **hyperparameters,
                environment.fit_id: {
                    "ridge": 2.0,
                    "family_effect_scale": 1.0,
                },
            },
        )
    inflated_baseline = fit_global_substitution_baseline(
        features, cohort, transform, ridge=1000.0
    )
    baseline_selection = ProteinStabilityModelSelectionRecord(
        cohort,
        features,
        (baseline, inflated_baseline, environment),
        candidate_hyperparameters={
            **hyperparameters,
            inflated_baseline.fit_id: {"ridge": 1000.0},
        },
    )
    rejected_baseline = (
        inflated_baseline
        if baseline_selection.chosen_baseline_fit_id == baseline.fit_id
        else baseline
    )
    with pytest.raises(ValueError, match="strongest prespecified baseline"):
        qualify_protein_stability(
            cohort,
            features,
            rejected_baseline,
            baseline_selection,
            SupportTuple(
                "protein.mutation-stability-prediction",
                {"assay": "test"},
            ),
            _THRESHOLDS,
            build_id="build",
            environment_id="environment",
            backend="cpu",
            topology="independent-families",
            precision="float64",
            reduction="family-macro",
            replay_id="replay",
            reviewer_id="reviewer",
            issued_at=1,
            expires_at=2,
        )

    other_roles = {
        **roles,
        source.measurements[2].independent_group_id: "locked_evaluation",
        source.measurements[3].independent_group_id: "model_selection",
    }
    other = prepare_protein_stability_cohort(
        source,
        other_roles,
        preparation_id_by_measurement=preparation,
        batch_id_by_measurement=preparation,
        criteria_ids=_STABILITY_CRITERION_IDS,
    )
    with pytest.raises(ValueError, match="exact cohort"):
        ProteinStabilityModelSelectionRecord(
            other,
            features,
            candidates,
            candidate_hyperparameters=hyperparameters,
        )


def test_stability_requires_measurement_identifiability_and_feature_admission(tmp_path):
    source = _source(tmp_path, uncertainty_reported=False)
    cohort = prepare_protein_stability_cohort(
        source,
        {
            source.measurements[0].independent_group_id: "calibration",
            source.measurements[1].independent_group_id: "calibration",
            source.measurements[2].independent_group_id: "model_selection",
            source.measurements[3].independent_group_id: "locked_evaluation",
        },
        preparation_id_by_measurement={
            item.measurement_id: item.background_id for item in source.measurements
        },
        batch_id_by_measurement={
            item.measurement_id: item.background_id for item in source.measurements
        },
        criteria_ids=_STABILITY_CRITERION_IDS,
    )
    features = _features(source)
    transform = fit_protein_feature_transform(features, cohort)
    baseline = fit_global_substitution_baseline(features, cohort, transform, ridge=1.0)
    assert not baseline.successful
    assert "calibration-measurement-uncertainty-unquantified" in baseline.reasons

    profile = protein_stability_claim_profile(
        cohort,
        SupportTuple("protein.mutation-stability-prediction", {"assay": "test"}),
        _THRESHOLDS,
    )
    assert {
        "measurement-calibration",
        "parameter-identifiability",
    } <= set(profile.required_stage_ids)

    denied = ReferenceArtifactManifest(
        "rights-denied-feature-source",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(b"feature").hexdigest(),
        size_bytes=len(b"feature"),
        license_id="restricted",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=False,
        export_permitted=True,
        export_classification="restricted",
        nondimensionalization={"identity": 1.0},
        uncertainty=None,
        lineage_ids=("feature-source",),
    )
    scalar = {residue: float(index) for index, residue in enumerate(_AA)}
    with pytest.raises(PermissionError, match="training-use-not-permitted"):
        AminoAcidScalarDefinition(
            scalar,
            {residue: 0.0 for residue in _AA},
            {residue: -float(index) for index, residue in enumerate(_AA)},
            source_manifests=(denied,),
            requested_use=_FEATURE_USE,
        )
    non_training_use = {**_FEATURE_USE, "training_use": False}
    denied_scalars = AminoAcidScalarDefinition(
        scalar,
        {residue: 0.0 for residue in _AA},
        {residue: -float(index) for index, residue in enumerate(_AA)},
        source_manifests=(denied,),
        requested_use=non_training_use,
    )
    calibration = source.measurements[0]
    environment = ProteinResidueEnvironment(
        1,
        "H",
        0.25,
        4.0,
        {
            "aliphatic": 1.0,
            "aromatic": 0.0,
            "polar": 2.0,
            "positive": 0.0,
            "negative": 0.0,
        },
        -1.0,
        1.0,
        hypothesis_id="denied-training-hypothesis",
        residue_mapping_id="denied-training-mapping",
        source_manifests=(source.source_manifest,),
        requested_use=non_training_use,
    )
    denied_feature = protein_mutation_features(
        calibration,
        environment,
        denied_scalars,
        feature_definition_id="transparent-v1",
        requested_use=non_training_use,
    )
    with pytest.raises(PermissionError, match="training-use-not-permitted"):
        fit_protein_feature_transform((denied_feature, *features[1:]), cohort)


def test_mutation_admission_refuses_wrong_wild_type():
    with pytest.raises(ValueError, match="does not match"):
        apply_mutation_code("CAAA", "A1D")
