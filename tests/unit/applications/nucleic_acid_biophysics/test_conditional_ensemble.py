import hashlib

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from phydrax.applications.nucleic_acid_biophysics import NucleicAcidConstruct
from phydrax.applications.nucleic_acid_biophysics.interchange import (
    DanceMapFile,
    import_dance_map_files,
)
from phydrax.applications.nucleic_acid_biophysics.observations import (
    compare_ensemble_supports,
    ConditionalEnsembleQualificationWorkflow,
    ConditionPopulationModel,
    EnsembleDiagnosticPolicy,
    EnsembleMixtureAdvantageCriterion,
    EnsemblePosteriorUncertainty,
    EnsemblePredictiveScoreCriterion,
    FiniteStructuralEnsembleModel,
    group_posterior_predictive_log_scores,
    group_profile_log_scores,
    ModelLadderEvaluation,
    prepare_conditional_ensemble_campaign,
    prepare_conditional_mapping_ladder,
    StructuralEnsembleHypothesis,
)
from phydrax.applications.nucleic_acid_biophysics.secondary_kinetics import (
    SecondaryStructureState,
)
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificClaimProfile,
    ScientificMetricCriterion,
    SupportTuple,
)


def _source(payload: bytes, name: str) -> ReferenceArtifactManifest:
    return ReferenceArtifactManifest(
        name,
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="synthetic-test-data",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"mutation-indicator": 1.0},
        uncertainty=None,
        lineage_ids=(f"lineage:{name}",),
    )


def _parsed(rows, source_id="source"):
    return "\n".join(
        "\t".join(
            (
                "PAIRED",
                f"{source_id}:read-{index}",
                "0",
                "3",
                category,
                "-999",
                mapped,
                effective,
                mutation,
            )
        )
        for index, (category, mapped, effective, mutation) in enumerate(rows)
    ).encode()


def _record(tmp_path, construct, *, ordinal, condition, preparation, rows):
    payload = _parsed(rows, f"source-{ordinal}")
    path = tmp_path / f"source-{ordinal}.mut"
    path.write_bytes(payload)
    source = _source(payload, f"source-{ordinal}")
    return DanceMapFile(
        path,
        source,
        nucleotide_ids=tuple(f"r:{index}" for index in range(4)),
        construct_id=construct.fingerprint(),
        condition_id=condition,
        preparation_id=preparation,
        batch_id=f"batch-{ordinal}",
        replicate_id=f"replicate-{ordinal}",
        reagent_id="DMS",
        protocol_id="ShapeMapper-effective-depth",
        upstream_tool="ShapeMapper2",
        upstream_version="2.2",
        reference_sequence_id="synthetic-reference-v1",
    )


def _admission(tmp_path):
    construct = NucleicAcidConstruct(("r",), ("AAAA",), ("RNA",), (False,))
    rows = (
        ("INCLUDED", "1111", "1101", "0100"),
        ("INCLUDED", "1111", "1111", "0001"),
    )
    records = (
        _record(
            tmp_path,
            construct,
            ordinal=0,
            condition="apo",
            preparation="prep-0",
            rows=rows,
        ),
        _record(
            tmp_path,
            construct,
            ordinal=1,
            condition="apo",
            preparation="prep-1",
            rows=rows,
        ),
        _record(
            tmp_path,
            construct,
            ordinal=2,
            condition="ligand",
            preparation="prep-2",
            rows=rows,
        ),
        _record(
            tmp_path,
            construct,
            ordinal=3,
            condition="ligand",
            preparation="prep-3",
            rows=rows,
        ),
    )
    return construct, import_dance_map_files(records, requested_use={})


def test_dance_map_admission_retains_coverage_missingness_and_exclusions(tmp_path):
    construct = NucleicAcidConstruct(("r",), ("AAAA",), ("RNA",), (False,))
    record = _record(
        tmp_path,
        construct,
        ordinal=0,
        condition="apo",
        preparation="prep-0",
        rows=(
            ("INCLUDED", "1111", "1011", "0010"),
            ("LOW_MAPQ", "1111", "0000", "0000"),
        ),
    )
    admission = import_dance_map_files((record,), requested_use={})
    batch = admission.batch
    np.testing.assert_array_equal(batch.coverage[0], [1, 1, 1, 1])
    np.testing.assert_array_equal(batch.observed_mask[0], [True, False, True, True])
    np.testing.assert_array_equal(batch.mutation[0], [0, 0, 1, 0])
    np.testing.assert_array_equal(batch.analysis_mask, [True, False])
    assert admission.mapping_category_ids == ("INCLUDED", "LOW_MAPQ")
    assert admission.category_counts == (1, 1)
    assert int(batch.excluded_profile_count) == 1
    with pytest.raises(ValueError, match="manifest"):
        record_path = tmp_path / "source-0.mut"
        record_path.write_bytes(record_path.read_bytes() + b"\n")
        import_dance_map_files((record,), requested_use={})


def test_dance_map_admission_rejects_relabelled_duplicate_payload(tmp_path):
    construct = NucleicAcidConstruct(("r",), ("AAAA",), ("RNA",), (False,))
    rows = (("INCLUDED", "1111", "1111", "0001"),)
    first = _record(
        tmp_path,
        construct,
        ordinal=0,
        condition="apo",
        preparation="prep-0",
        rows=rows,
    )
    duplicate_path = tmp_path / "duplicate.mut"
    payload = (tmp_path / "source-0.mut").read_bytes()
    duplicate_path.write_bytes(payload)
    duplicate = DanceMapFile(
        duplicate_path,
        _source(payload, "renamed-source"),
        nucleotide_ids=first.nucleotide_ids,
        construct_id=first.construct_id,
        condition_id="ligand",
        preparation_id="renamed-preparation",
        batch_id="renamed-batch",
        replicate_id="renamed-replicate",
        reagent_id=first.reagent_id,
        protocol_id=first.protocol_id,
        upstream_tool=first.upstream_tool,
        upstream_version=first.upstream_version,
        reference_sequence_id=first.reference_sequence_id,
    )
    with pytest.raises(ValueError, match="Identical parsed-mutation payloads"):
        import_dance_map_files((first, duplicate), requested_use={})


def test_campaign_refuses_preparation_construct_leakage(tmp_path):
    construct = NucleicAcidConstruct(("r",), ("AAAA",), ("RNA",), (False,))
    rows = (("INCLUDED", "1111", "1111", "0001"),)
    first = _record(
        tmp_path,
        construct,
        ordinal=0,
        condition="apo",
        preparation="shared-preparation",
        rows=rows,
    )
    second = _record(
        tmp_path,
        construct,
        ordinal=1,
        condition="ligand",
        preparation="shared-preparation",
        rows=rows,
    )
    batch = import_dance_map_files((first, second), requested_use={}).batch
    with pytest.raises(ValueError, match="independent_unit_id"):
        prepare_conditional_ensemble_campaign(
            batch,
            (
                CampaignRole("calibration", (batch.case_ids[0],)),
                CampaignRole("locked_evaluation", (batch.case_ids[1],)),
            ),
        )


def test_mapping_ladder_is_condition_and_provenance_aware(tmp_path):
    _, admission = _admission(tmp_path)
    batch = admission.batch
    ladder = prepare_conditional_mapping_ladder(
        batch,
        np.array([[1.0, 0.0, 1.0, 0.0]]),
        nucleotide_features=np.eye(4)[None],
        nucleotide_feature_names=("site-0", "site-1", "site-2", "site-3"),
        local_context_features=np.array([[[0.0], [1.0], [1.0], [0.0]]]),
        local_context_feature_names=("internal",),
        condition_features=np.array([[0.0], [1.0]]),
        condition_feature_names=("ligand",),
        source_case_ids=(batch.case_ids[0],),
    )
    assert tuple(law.kind for law in ladder.laws) == (
        "binary-accessibility",
        "context",
        "hierarchical",
    )
    assert ladder.context.design.shape[-1] > ladder.baseline.design.shape[-1]
    assert ladder.hierarchical.design.shape[-1] > ladder.context.design.shape[-1]
    scores = ladder.baseline.per_profile_log_likelihood(jnp.zeros(2))
    assert scores.shape == (batch.profile_count,)
    assert bool(jnp.all(jnp.isfinite(scores)))
    calibration = batch.profile_mask_for_cases((batch.case_ids[0],))
    problem = ladder.baseline.posterior_problem(fit_profile_mask=calibration)
    expected = jnp.sum(jnp.where(calibration, scores, 0.0))
    np.testing.assert_allclose(problem.log_likelihood(jnp.zeros(2)), expected)


def _ensemble(tmp_path):
    construct, admission = _admission(tmp_path)
    batch = admission.batch
    keys = construct.nucleotide_keys
    states = (
        SecondaryStructureState(construct),
        SecondaryStructureState(construct, ((keys[0], keys[3]),)),
        SecondaryStructureState(construct, ((keys[0], keys[3]),)),
    )
    hypothesis = StructuralEnsembleHypothesis(
        ("open", "closed-a", "closed-b"),
        states,
        "binary-paired-accessibility-v1",
        ("independent-structure-panel",),
        (batch.case_ids[0],),
        (),
    )
    population = ConditionPopulationModel(
        batch.condition_ids,
        hypothesis.state_ids,
        kind="free-simplex",
    )
    response = np.array(
        [
            [-2.0, -2.0, -2.0, -2.0],
            [-4.0, -2.0, -2.0, -4.0],
            [-4.0, -2.0, -2.0, -4.0],
        ]
    )
    model = FiniteStructuralEnsembleModel(
        batch,
        hypothesis,
        population,
        np.zeros((batch.profile_count, 4)),
        response,
        mapping_law_id="preselected-context-law",
        observation_offset_source_case_ids=(batch.case_ids[0],),
        response_prior_scale=1.0,
        dirichlet_concentration=2.0,
    )
    problem = model.posterior_problem()
    parameters = problem.parameter_space.constrain(problem.initial_position)
    return batch, model, parameters


def test_finite_ensemble_exposes_equivalence_and_permutation_safe_summary(tmp_path):
    batch, model, parameters = _ensemble(tmp_path)
    policy = EnsembleDiagnosticPolicy(1e-10, 1e-6, 1e-8, 0.5)
    diagnostics = model.diagnostics(parameters, policy)
    assert bool(diagnostics.equivalent_state_pairs[1, 2])
    assert not bool(diagnostics.unique_state_interpretation_supported)
    failed = model.diagnostics(parameters, policy, optimization_converged=False)
    assert not bool(failed.numerical_valid)
    prediction = model.posterior_prediction(parameters)
    assert prediction.mutation_covariance.shape == (batch.profile_count, 4, 4)
    np.testing.assert_allclose(jnp.sum(prediction.state_population, axis=-1), 1.0)

    order = np.array([2, 0, 1])
    permuted_hypothesis = StructuralEnsembleHypothesis(
        tuple(model.hypothesis.state_ids[index] for index in order),
        tuple(model.hypothesis.structures[index] for index in order),
        model.hypothesis.feature_definition_id,
        model.hypothesis.source_ids,
        model.hypothesis.source_case_ids,
        model.hypothesis.parent_case_ids,
    )
    permuted_population = ConditionPopulationModel(
        batch.condition_ids,
        permuted_hypothesis.state_ids,
        kind="free-simplex",
    )
    permuted_model = FiniteStructuralEnsembleModel(
        batch,
        permuted_hypothesis,
        permuted_population,
        model.observation_offset,
        model.response_prior_mean[order],
        mapping_law_id=model.mapping_law_id,
        observation_offset_source_case_ids=model.observation_offset_source_case_ids,
        observation_offset_parent_case_ids=model.observation_offset_parent_case_ids,
        response_prior_scale=model.response_prior_scale,
    )
    permuted_parameters = {
        "response_logits": parameters["response_logits"][order],
        "population": parameters["population"][:, order],
    }
    original = model.permutation_invariant_summary(parameters)
    permuted = permuted_model.permutation_invariant_summary(permuted_parameters)
    np.testing.assert_allclose(
        original.sorted_condition_populations, permuted.sorted_condition_populations
    )
    np.testing.assert_allclose(
        original.sorted_state_mean_mutation, permuted.sorted_state_mean_mutation
    )
    np.testing.assert_allclose(
        original.sorted_pairwise_separation, permuted.sorted_pairwise_separation
    )
    np.testing.assert_allclose(
        original.posterior_predictive_mean, permuted.posterior_predictive_mean
    )


def _workflow_fixture(tmp_path, *, epistemic_uncertainty=True):
    batch, model, parameters = _ensemble(tmp_path)
    criteria = {
        "locked": EnsemblePredictiveScoreCriterion("locked-prediction", -1e9),
        "locked_fail": EnsemblePredictiveScoreCriterion("locked-prediction", 1e9),
        "calibration": EnsemblePredictiveScoreCriterion("predictive-calibration", -1e9),
        "advantage": EnsembleMixtureAdvantageCriterion(0.01),
        "advantage_fail": EnsembleMixtureAdvantageCriterion(1e9),
        "metric": ScientificMetricCriterion(
            "ensemble-held-out-score",
            "at_least",
            -1e9,
            None,
            "natural-log-unit-per-observed-event",
            "independent_unit_macro",
        ),
    }
    campaign = prepare_conditional_ensemble_campaign(
        batch,
        (
            CampaignRole("calibration", (batch.case_ids[0],)),
            CampaignRole("model_selection", (batch.case_ids[1],)),
            CampaignRole("interval_calibration", (batch.case_ids[2],)),
            CampaignRole("locked_evaluation", (batch.case_ids[3],)),
        ),
        preprocessing_source_ids=(batch.case_ids[0],),
        criteria_ids=tuple(item.criterion_id for item in criteria.values()),
    )
    finite_scores = model.per_profile_log_likelihood(parameters)
    levels = ("binary-accessibility", "context", "hierarchical", "finite-mixture")
    model_ids = ("model-0", "model-1", "model-2", model.model_id)
    policy = EnsembleDiagnosticPolicy(1e-10, 1e-6, 1e-8, 1.0)
    finite_fit = model.fit(
        policy=policy,
        requested_use={},
        fit_profile_mask=batch.profile_mask_for_cases((batch.case_ids[0],)),
        max_steps=1,
        gradient_tolerance=1e9,
    )
    finite_uncertainty = (
        EnsemblePosteriorUncertainty(model, finite_fit, campaign)
        if epistemic_uncertainty
        else None
    )
    evaluations = []
    for index, (model_id, level) in enumerate(zip(model_ids, levels, strict=True)):
        point_scores = finite_scores + (0.0 if index == 3 else -100.0 - index)
        if index == 3 and finite_uncertainty is not None:
            locked_score = group_posterior_predictive_log_scores(
                batch,
                campaign,
                "locked_evaluation",
                model,
                finite_uncertainty,
            )
            calibration_score = group_posterior_predictive_log_scores(
                batch,
                campaign,
                "interval_calibration",
                model,
                finite_uncertainty,
            )
        else:
            locked_score = group_profile_log_scores(
                batch,
                campaign,
                "locked_evaluation",
                point_scores,
                model_id=model_id,
            )
            calibration_score = group_profile_log_scores(
                batch,
                campaign,
                "interval_calibration",
                point_scores,
                model_id=model_id,
            )
        evaluations.append(
            ModelLadderEvaluation(
                model_id,
                level,
                group_profile_log_scores(
                    batch,
                    campaign,
                    "model_selection",
                    point_scores,
                    model_id=model_id,
                ),
                locked_score,
                predictive_calibration=calibration_score,
                derivation_source_case_ids=(batch.case_ids[0],),
                support_id="support-a" if level == "finite-mixture" else None,
                hypothesis_id=(
                    model.hypothesis.hypothesis_id if level == "finite-mixture" else None
                ),
                derivation_id=(
                    model.derivation_id if level == "finite-mixture" else model_id
                ),
                posterior_uncertainty=(
                    finite_uncertainty if level == "finite-mixture" else None
                ),
                execution_valid=True,
            )
        )
    evaluations = tuple(evaluations)
    workflow = ConditionalEnsembleQualificationWorkflow(
        batch, campaign, model_selection_tolerance=0.0
    )
    diagnostics = model.diagnostics(
        parameters,
        policy,
        profile_mask=workflow.profile_mask("calibration"),
    )
    locked_diagnostics = model.diagnostics(
        parameters,
        policy,
        profile_mask=workflow.profile_mask("locked_evaluation"),
    )
    return (
        workflow,
        evaluations,
        diagnostics,
        locked_diagnostics,
        criteria,
        model,
        finite_fit,
    )


def _support_comparison(workflow, evaluations, scores):
    finite = evaluations[3]
    return compare_ensemble_supports(
        ("support-a", "support-b"),
        scores,
        campaign_id=workflow.campaign.campaign_id,
        model_ids=(finite.model_id, "alternative-model"),
        hypothesis_ids=(finite.hypothesis_id, "alternative-hypothesis"),
        derivation_ids=(finite.derivation_id, "alternative-derivation"),
        score_ids=(finite.model_selection.score_id, "alternative-score"),
        equivalence_tolerance=0.01,
        support_valid=[True, True],
    )


def _evidence_metadata():
    return {
        "build_id": "build",
        "environment_id": "environment",
        "backend": "cpu",
        "topology": "single-process",
        "precision": "float64",
        "reduction": "deterministic",
        "replay_id": "replay",
        "reviewer_id": "reviewer",
        "issued_at": 1,
        "expires_at": 2,
    }


def test_grouped_workflow_keeps_equivalent_supports_inconclusive(tmp_path):
    workflow, evaluations, diagnostics, locked, criteria, _, _ = _workflow_fixture(
        tmp_path
    )
    comparison = _support_comparison(workflow, evaluations, [10.0, 10.0])
    assessment = workflow.assess(
        evaluations,
        support_comparison=comparison,
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=locked,
    )
    assert assessment.selected_model_id is None
    assert int(assessment.selected_index) == -1
    evidence = workflow.stage_evidence(
        "locked-prediction",
        assessment,
        predictive_criterion=criteria["locked"],
        mixture_advantage_criterion=criteria["advantage"],
        **_evidence_metadata(),
    )
    assert evidence.outcome == "inconclusive"
    with pytest.raises(TypeError, match="EnsemblePredictiveScoreCriterion"):
        workflow.stage_evidence(
            "locked-prediction",
            assessment,
            predictive_criterion=True,
            **_evidence_metadata(),
        )


def test_predictive_stages_require_frozen_threshold_model_score_and_advantage(tmp_path):
    workflow, evaluations, diagnostics, locked, criteria, model, finite_fit = (
        _workflow_fixture(tmp_path)
    )
    assessment = workflow.assess(
        evaluations,
        support_comparison=_support_comparison(workflow, evaluations, [11.0, 10.0]),
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=locked,
    )
    passed = workflow.stage_evidence(
        "locked-prediction",
        assessment,
        predictive_criterion=criteria["locked"],
        mixture_advantage_criterion=criteria["advantage"],
        **_evidence_metadata(),
    )
    assert passed.outcome == "passed"
    assert {
        workflow.campaign.campaign_id,
        assessment.selected_model_id,
        assessment.selected_locked_score_id,
        assessment.selected_epistemic_uncertainty_id,
        criteria["locked"].criterion_id,
        criteria["advantage"].criterion_id,
    }.issubset(passed.subject_ids)
    uncertainty = evaluations[3].posterior_uncertainty
    raw_draws = np.asarray(uncertainty.raw_parameter_draws)
    centered = raw_draws - np.mean(raw_draws, axis=0, keepdims=True)
    empirical_covariance = centered.T @ centered / raw_draws.shape[0]
    np.testing.assert_allclose(
        empirical_covariance,
        uncertainty.parameter_covariance,
        rtol=1e-10,
        atol=1e-10,
    )
    assert uncertainty.approximation_id == "laplace-spherical-radial-equal-weight-v1"
    object.__setattr__(finite_fit.optimization, "converged", False)
    with pytest.raises(ValueError, match="converged calibration fit"):
        EnsemblePosteriorUncertainty(
            model,
            finite_fit,
            workflow.campaign,
        )
    with pytest.raises(TypeError):
        EnsemblePosteriorUncertainty(
            model,
            finite_fit,
            workflow.campaign,
            np.ones((2, 1)),
        )

    missing_advantage = workflow.stage_evidence(
        "locked-prediction",
        assessment,
        predictive_criterion=criteria["locked"],
        **_evidence_metadata(),
    )
    assert missing_advantage.outcome == "inconclusive"
    failed_advantage = workflow.stage_evidence(
        "locked-prediction",
        assessment,
        predictive_criterion=criteria["locked"],
        mixture_advantage_criterion=criteria["advantage_fail"],
        **_evidence_metadata(),
    )
    assert failed_advantage.outcome == "failed"
    calibrated = workflow.stage_evidence(
        "predictive-calibration",
        assessment,
        predictive_criterion=criteria["calibration"],
        **_evidence_metadata(),
    )
    assert calibrated.outcome == "passed"
    with pytest.raises(ValueError, match="not frozen"):
        workflow.stage_evidence(
            "locked-prediction",
            assessment,
            predictive_criterion=EnsemblePredictiveScoreCriterion(
                "locked-prediction", -2e9
            ),
            mixture_advantage_criterion=criteria["advantage"],
            **_evidence_metadata(),
        )


def test_invalid_uncertain_or_locked_diagnostic_failure_cannot_pass(tmp_path):
    workflow, evaluations, diagnostics, locked, criteria, _, _ = _workflow_fixture(
        tmp_path
    )
    first = evaluations[0]
    invalid_evaluations = (
        ModelLadderEvaluation(
            first.model_id,
            first.level,
            first.model_selection,
            first.locked_evaluation,
            predictive_calibration=first.predictive_calibration,
            derivation_source_case_ids=first.derivation_source_case_ids,
            derivation_parent_case_ids=first.derivation_parent_case_ids,
            derivation_id=first.derivation_id,
            posterior_uncertainty=first.posterior_uncertainty,
            execution_valid=False,
        ),
        *evaluations[1:],
    )
    invalid = workflow.assess(
        invalid_evaluations,
        support_comparison=_support_comparison(
            workflow, invalid_evaluations, [11.0, 10.0]
        ),
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=locked,
    )
    evidence = workflow.stage_evidence(
        "locked-prediction",
        invalid,
        predictive_criterion=criteria["locked"],
        mixture_advantage_criterion=criteria["advantage"],
        **_evidence_metadata(),
    )
    assert evidence.outcome == "failed"

    uncertain = _workflow_fixture(tmp_path, epistemic_uncertainty=False)
    uncertain_assessment = uncertain[0].assess(
        uncertain[1],
        support_comparison=_support_comparison(uncertain[0], uncertain[1], [11.0, 10.0]),
        ensemble_diagnostics=uncertain[2],
        locked_ensemble_diagnostics=uncertain[3],
    )
    uncertain_evidence = uncertain[0].stage_evidence(
        "locked-prediction",
        uncertain_assessment,
        predictive_criterion=uncertain[4]["locked"],
        mixture_advantage_criterion=uncertain[4]["advantage"],
        **_evidence_metadata(),
    )
    assert uncertain_evidence.outcome == "inconclusive"

    violating_locked = eqx.tree_at(
        lambda item: (
            item.support_complete,
            item.residual_assumption_supported,
        ),
        locked,
        (jnp.asarray(False), jnp.asarray(False)),
    )
    violating_assessment = workflow.assess(
        evaluations,
        support_comparison=_support_comparison(workflow, evaluations, [11.0, 10.0]),
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=violating_locked,
    )
    violating_evidence = workflow.stage_evidence(
        "locked-prediction",
        violating_assessment,
        predictive_criterion=criteria["locked"],
        mixture_advantage_criterion=criteria["advantage"],
        **_evidence_metadata(),
    )
    assert violating_evidence.outcome == "failed"


def test_workflow_rejects_locked_derivation_and_unrelated_support_winner(tmp_path):
    workflow, evaluations, diagnostics, locked, _, _, _ = _workflow_fixture(tmp_path)
    finite = evaluations[3]
    leaked_finite = ModelLadderEvaluation(
        finite.model_id,
        finite.level,
        finite.model_selection,
        finite.locked_evaluation,
        predictive_calibration=finite.predictive_calibration,
        derivation_source_case_ids=(
            *finite.derivation_source_case_ids,
            workflow.campaign.roles[3].case_ids[0],
        ),
        support_id=finite.support_id,
        hypothesis_id=finite.hypothesis_id,
        derivation_id=finite.derivation_id,
        posterior_uncertainty=finite.posterior_uncertainty,
        execution_valid=True,
    )
    with pytest.raises(ValueError, match="preprocessing cases"):
        workflow.assess(
            (*evaluations[:3], leaked_finite),
            support_comparison=_support_comparison(workflow, evaluations, [11.0, 10.0]),
            ensemble_diagnostics=diagnostics,
            locked_ensemble_diagnostics=locked,
        )
    with pytest.raises(ValueError, match="Winning support"):
        workflow.assess(
            evaluations,
            support_comparison=_support_comparison(workflow, evaluations, [10.0, 11.0]),
            ensemble_diagnostics=diagnostics,
            locked_ensemble_diagnostics=locked,
        )


def test_synthetic_batch_cannot_mint_source_admission(tmp_path):
    workflow, evaluations, diagnostics, locked, _, _, _ = _workflow_fixture(tmp_path)
    assessment = workflow.assess(
        evaluations,
        support_comparison=_support_comparison(workflow, evaluations, [11.0, 10.0]),
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=locked,
    )
    evidence = workflow.stage_evidence(
        "source-admission",
        assessment,
        **_evidence_metadata(),
    )
    assert evidence.outcome == "inconclusive"


def test_claim_rejects_campaign_criteria_and_support_scope_mismatch(tmp_path):
    workflow, evaluations, diagnostics, locked, criteria, _, _ = _workflow_fixture(
        tmp_path
    )
    assessment = workflow.assess(
        evaluations,
        support_comparison=_support_comparison(workflow, evaluations, [11.0, 10.0]),
        ensemble_diagnostics=diagnostics,
        locked_ensemble_diagnostics=locked,
    )
    profile = ScientificClaimProfile(
        "rna-conditional-ensemble-inference",
        SupportTuple(
            "rna-conditional-ensemble-inference",
            {
                "construct": workflow.batch.construct_ids[0],
                "generalization": "held-out-condition-and-perturbation",
                "observable": "mapped-mutation-profile",
                "protocol": workflow.batch.protocol_ids[0],
                "state-support": assessment.hypothesis_ids[
                    int(assessment.selected_index)
                ],
            },
        ),
        ("mapped-mutation-profile",),
        workflow.batch.condition_ids,
        workflow.campaign.campaign_id,
        ("locked-prediction",),
        (criteria["metric"],),
        "abstain-outside-exact-campaign",
        ("campaign-change",),
        frozen_criteria_ids=(criteria["metric"].criterion_id,),
    )
    with pytest.raises(ValueError, match="exactly match the campaign"):
        workflow.evaluate_claim(
            profile,
            {"ensemble-held-out-score": 0.0},
            (),
            assessment=assessment,
            metric_units={
                "ensemble-held-out-score": "natural-log-unit-per-observed-event"
            },
            metric_aggregations={"ensemble-held-out-score": "independent_unit_macro"},
            **_evidence_metadata(),
        )
    weak_profile = ScientificClaimProfile(
        profile.capability_name,
        profile.support,
        profile.observable_ids,
        profile.condition_domain_ids,
        profile.campaign_id,
        ("locked-prediction",),
        profile.criteria,
        profile.abstention_policy_id,
        profile.invalidation_triggers,
        frozen_criteria_ids=workflow.campaign.criteria_ids,
    )
    with pytest.raises(ValueError, match="exact conditional-ensemble"):
        workflow.evaluate_claim(
            weak_profile,
            {"ensemble-held-out-score": 0.0},
            (),
            assessment=assessment,
            metric_units={
                "ensemble-held-out-score": "natural-log-unit-per-observed-event"
            },
            metric_aggregations={"ensemble-held-out-score": "independent_unit_macro"},
            **_evidence_metadata(),
        )
