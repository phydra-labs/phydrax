"""Caller-supplied DANCE-MaP conditional-ensemble qualification workflow.

The JSON campaign manifest supplies every local parsed-mutation path, immutable
source manifest, experimental grouping coordinate, fixed model feature, and
finite structural support.  This command never downloads, aligns, or invents
observations.  Model and support selection use only the model-selection role;
locked observations are scored only after fitting on calibration preparations.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import jax
import numpy as np

from benchmarks._runtime import capture_environment
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
    EnsemblePredictiveScoreCriterion,
    FiniteStructuralEnsembleModel,
    group_profile_log_scores,
    ModelLadderEvaluation,
    prepare_conditional_ensemble_campaign,
    prepare_conditional_mapping_ladder,
    StructuralEnsembleHypothesis,
)
from phydrax.applications.nucleic_acid_biophysics.secondary_kinetics import (
    SecondaryStructureState,
)
from phydrax.qualification import CampaignRole, ReferenceArtifactManifest


def _local_path(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _construct(record) -> NucleicAcidConstruct:
    return NucleicAcidConstruct(
        tuple(record["strand_ids"]),
        tuple(record["sequences"]),
        tuple(record["polymer_types"]),
        tuple(record["circular"]),
    )


def _support(
    record, construct: NucleicAcidConstruct, batch
) -> tuple[str, StructuralEnsembleHypothesis]:
    keys = construct.nucleotide_keys
    states = []
    state_ids = []
    for state in record["states"]:
        state_ids.append(state["state_id"])
        pairs = tuple(
            (keys[int(first)], keys[int(second)]) for first, second in state["pairs"]
        )
        states.append(SecondaryStructureState(construct, pairs))
    return record["support_id"], StructuralEnsembleHypothesis(
        tuple(state_ids),
        tuple(states),
        record["feature_definition_id"],
        tuple(record["source_ids"]),
        tuple(batch.case_ids[int(index)] for index in record["source_case_ordinals"]),
        tuple(
            batch.case_ids[int(index)] for index in record.get("parent_case_ordinals", ())
        ),
    )


def _json_value(value):
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, (jax.Array, np.ndarray)):
        return _json_value(np.asarray(value).tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def run(campaign_manifest: Path, *, max_steps: int, gradient_tolerance: float) -> dict:
    if max_steps < 0 or not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError(
            "Optimization controls must be nonnegative steps and positive tolerance."
        )
    manifest_path = campaign_manifest.expanduser().resolve()
    config = json.loads(manifest_path.read_text())
    base = manifest_path.parent
    construct = _construct(config["construct"])
    nucleotide_ids = tuple(config["nucleotide_ids"])
    requested_use = dict(config.get("requested_use", {}))

    files = []
    role_names = []
    preprocessing_ordinals = []
    for ordinal, record in enumerate(config["files"]):
        source = ReferenceArtifactManifest.from_record(record["source_manifest"])
        files.append(
            DanceMapFile(
                _local_path(base, record["path"]),
                source,
                nucleotide_ids=nucleotide_ids,
                construct_id=construct.fingerprint(),
                condition_id=record["condition_id"],
                preparation_id=record["preparation_id"],
                batch_id=record["batch_id"],
                replicate_id=record["replicate_id"],
                reagent_id=record["reagent_id"],
                protocol_id=record["protocol_id"],
                upstream_tool=record["upstream_tool"],
                upstream_version=record["upstream_version"],
                reference_sequence_id=record["reference_sequence_id"],
            )
        )
        role_names.append(record["role"])
        if bool(record.get("preprocessing_source", False)):
            preprocessing_ordinals.append(ordinal)
    admission = import_dance_map_files(
        tuple(files),
        included_mapping_categories=tuple(
            config.get("included_mapping_categories", ["INCLUDED"])
        ),
        requested_use=requested_use,
    )
    batch = admission.batch
    if len(batch.construct_ids) != 1:
        raise ValueError("This finite structural campaign requires one exact construct.")
    roles = tuple(
        CampaignRole(
            name,
            tuple(
                batch.case_ids[index]
                for index, value in enumerate(role_names)
                if value == name
            ),
        )
        for name in (
            "calibration",
            "model_selection",
            "interval_calibration",
            "locked_evaluation",
            "prospective",
        )
        if any(value == name for value in role_names)
    )
    predictive_criteria = tuple(
        EnsemblePredictiveScoreCriterion(stage_id, float(minimum))
        for stage_id, minimum in sorted(
            config.get("predictive_score_thresholds", {}).items()
        )
    )
    mixture_criterion = (
        None
        if config.get("minimum_locked_mixture_advantage") is None
        else EnsembleMixtureAdvantageCriterion(
            float(config["minimum_locked_mixture_advantage"])
        )
    )
    frozen_criterion_ids = tuple(
        dict.fromkeys(
            (
                *config.get("criteria_ids", ()),
                *(criterion.criterion_id for criterion in predictive_criteria),
                *(() if mixture_criterion is None else (mixture_criterion.criterion_id,)),
            )
        )
    )
    campaign = prepare_conditional_ensemble_campaign(
        batch,
        roles,
        preprocessing_source_ids=tuple(
            batch.case_ids[index] for index in preprocessing_ordinals
        ),
        criteria_ids=frozen_criterion_ids,
    )
    workflow = ConditionalEnsembleQualificationWorkflow(
        batch,
        campaign,
        model_selection_tolerance=float(config["model_selection_tolerance"]),
    )
    has_interval_calibration = any(
        role.name == "interval_calibration" for role in campaign.roles
    )
    calibration_mask = workflow.profile_mask("calibration")

    mapping = config["mapping"]
    ladder = prepare_conditional_mapping_ladder(
        batch,
        np.asarray(mapping["binary_accessibility"], dtype=float),
        nucleotide_features=np.asarray(mapping["nucleotide_features"], dtype=float),
        nucleotide_feature_names=tuple(mapping["nucleotide_feature_names"]),
        local_context_features=np.asarray(mapping["local_context_features"], dtype=float),
        local_context_feature_names=tuple(mapping["local_context_feature_names"]),
        condition_features=np.asarray(mapping["condition_features"], dtype=float),
        source_case_ids=tuple(
            batch.case_ids[int(index)] for index in mapping["source_case_ordinals"]
        ),
        parent_case_ids=tuple(
            batch.case_ids[int(index)]
            for index in mapping.get("parent_case_ordinals", ())
        ),
        condition_feature_names=tuple(mapping["condition_feature_names"]),
        shared_prior_scale=float(mapping.get("shared_prior_scale", 3.0)),
        hierarchical_prior_scale=float(mapping.get("hierarchical_prior_scale", 1.0)),
    )
    mapping_fits = tuple(
        law.fit(
            requested_use=requested_use,
            fit_profile_mask=calibration_mask,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
        )
        for law in ladder.laws
    )

    finite_law_kind = config["finite_mapping_law"]
    law_by_kind = {law.kind: law for law in ladder.laws}
    fit_by_kind = {
        law.kind: fit for law, fit in zip(ladder.laws, mapping_fits, strict=True)
    }
    if finite_law_kind not in law_by_kind:
        raise ValueError(
            "finite_mapping_law must name one predeclared mapping-ladder law."
        )
    finite_law = law_by_kind[finite_law_kind]
    finite_mapping_fit = fit_by_kind[finite_law_kind]
    accessibility_index = finite_law.parameter_names.index("binary-accessibility")
    parameters = finite_mapping_fit.optimization.parameters
    observation_offset = finite_law.logits(parameters) - (
        finite_law.design[:, :, accessibility_index] * parameters[accessibility_index]
    )

    population_record = config["population_model"]
    declared_conditions = tuple(population_record["condition_ids"])
    if declared_conditions != batch.condition_ids:
        raise ValueError(
            "Population condition_ids must exactly match admitted first-occurrence order."
        )
    policy = EnsembleDiagnosticPolicy(**config["diagnostic_policy"])
    support_ids = []
    support_fits = []
    support_models = []
    support_selection_scores = []
    support_calibration_scores = []
    support_valid = []
    for record in config["supports"]:
        support_id, hypothesis = _support(record, construct, batch)
        population = ConditionPopulationModel(
            declared_conditions,
            hypothesis.state_ids,
            kind=population_record["kind"],
            design=(
                None
                if population_record["kind"] == "free-simplex"
                else np.asarray(population_record["design"], dtype=float)
            ),
            feature_names=tuple(population_record.get("feature_names", ())),
        )
        accessibility = np.asarray(hypothesis.binary_accessibility())
        response_prior = accessibility * float(parameters[accessibility_index])
        model = FiniteStructuralEnsembleModel(
            batch,
            hypothesis,
            population,
            observation_offset,
            response_prior,
            mapping_law_id=finite_law.model_id,
            observation_offset_source_case_ids=tuple(
                batch.case_ids[int(index)]
                for index in mapping["observation_offset_source_case_ordinals"]
            ),
            observation_offset_parent_case_ids=tuple(
                batch.case_ids[int(index)]
                for index in mapping.get("observation_offset_parent_case_ordinals", ())
            ),
            response_prior_scale=float(record["response_prior_scale"]),
            population_prior_scale=float(record.get("population_prior_scale", 2.0)),
            dirichlet_concentration=float(record.get("dirichlet_concentration", 1.0)),
        )
        fit = model.fit(
            policy=policy,
            requested_use=requested_use,
            fit_profile_mask=calibration_mask,
            max_steps=max_steps,
            gradient_tolerance=gradient_tolerance,
        )
        selection_score = group_profile_log_scores(
            batch,
            campaign,
            "model_selection",
            fit.per_profile_log_score,
            model_id=model.model_id,
        )
        calibration_score = (
            group_profile_log_scores(
                batch,
                campaign,
                "interval_calibration",
                fit.per_profile_log_score,
                model_id=model.model_id,
            )
            if has_interval_calibration
            else None
        )
        support_ids.append(support_id)
        support_models.append(model)
        support_fits.append(fit)
        support_selection_scores.append(selection_score)
        support_calibration_scores.append(calibration_score)
        support_valid.append(
            bool(fit.optimization.converged)
            and bool(fit.diagnostics.numerical_valid)
            and bool(fit.diagnostics.support_complete)
        )
    comparison = compare_ensemble_supports(
        tuple(support_ids),
        [float(score.independent_unit_macro) for score in support_selection_scores],
        equivalence_tolerance=float(config["support_equivalence_tolerance"]),
        support_valid=np.asarray(support_valid, dtype=bool),
        campaign_id=campaign.campaign_id,
        model_ids=tuple(model.model_id for model in support_models),
        hypothesis_ids=tuple(model.hypothesis.hypothesis_id for model in support_models),
        derivation_ids=tuple(model.derivation_id for model in support_models),
        score_ids=tuple(score.score_id for score in support_selection_scores),
    )
    representative_index = int(comparison.best_support_index)
    if representative_index < 0:
        representative_index = 0
    representative_fit = support_fits[representative_index]
    representative_model = support_models[representative_index]

    evaluations = []
    for law, fit in zip(ladder.laws, mapping_fits, strict=True):
        evaluations.append(
            ModelLadderEvaluation(
                law.model_id,
                law.kind,
                group_profile_log_scores(
                    batch,
                    campaign,
                    "model_selection",
                    fit.per_profile_log_score,
                    model_id=law.model_id,
                ),
                group_profile_log_scores(
                    batch,
                    campaign,
                    "locked_evaluation",
                    fit.per_profile_log_score,
                    model_id=law.model_id,
                ),
                predictive_calibration=(
                    group_profile_log_scores(
                        batch,
                        campaign,
                        "interval_calibration",
                        fit.per_profile_log_score,
                        model_id=law.model_id,
                    )
                    if has_interval_calibration
                    else None
                ),
                derivation_source_case_ids=ladder.source_case_ids,
                derivation_parent_case_ids=ladder.parent_case_ids,
                derivation_id=law.model_id,
                execution_valid=fit.optimization.converged,
            )
        )
    evaluations.append(
        ModelLadderEvaluation(
            representative_model.model_id,
            "finite-mixture",
            support_selection_scores[representative_index],
            group_profile_log_scores(
                batch,
                campaign,
                "locked_evaluation",
                representative_fit.per_profile_log_score,
                model_id=representative_model.model_id,
            ),
            predictive_calibration=support_calibration_scores[representative_index],
            derivation_source_case_ids=tuple(
                sorted(
                    set(representative_model.hypothesis.source_case_ids)
                    | set(representative_model.observation_offset_source_case_ids)
                )
            ),
            derivation_parent_case_ids=tuple(
                sorted(
                    set(representative_model.hypothesis.parent_case_ids)
                    | set(representative_model.observation_offset_parent_case_ids)
                )
            ),
            support_id=support_ids[representative_index],
            hypothesis_id=representative_model.hypothesis.hypothesis_id,
            derivation_id=representative_model.derivation_id,
            execution_valid=representative_fit.optimization.converged,
        )
    )
    locked_diagnostics = representative_model.diagnostics(
        representative_fit.optimization.parameters,
        policy,
        profile_mask=workflow.profile_mask("locked_evaluation"),
        optimization_converged=representative_fit.optimization.converged,
    )
    assessment = workflow.assess(
        evaluations,
        support_comparison=comparison,
        ensemble_diagnostics=representative_fit.diagnostics,
        locked_ensemble_diagnostics=locked_diagnostics,
    )
    diagnostics = representative_fit.diagnostics
    result = {
        "environment": capture_environment().to_dict(),
        "campaign_manifest": str(manifest_path),
        "campaign_id": campaign.campaign_id,
        "source_manifest_ids": list(batch.source_ids),
        "profile_count": batch.profile_count,
        "excluded_profile_count": int(batch.excluded_profile_count),
        "observed_event_count": int(
            np.asarray(batch.observed_mask & batch.analysis_mask[:, None]).sum()
        ),
        "roles": {role.name: list(role.case_ids) for role in campaign.roles},
        "mapping_ladder": [
            {
                "kind": law.kind,
                "model_id": law.model_id,
                "optimizer_converged": fit.optimization.converged,
                "termination_reason": fit.optimization.termination_reason,
                "design_rank": int(fit.design_rank),
                "parameter_count": len(fit.parameter_names),
                "identifiable": bool(fit.identifiable),
                "maximum_absolute_residual_correlation": float(
                    fit.maximum_absolute_residual_correlation
                ),
            }
            for law, fit in zip(ladder.laws, mapping_fits, strict=True)
        ],
        "support_comparison": {
            "support_ids": support_ids,
            "model_selection_scores": [
                float(score.independent_unit_macro) for score in support_selection_scores
            ],
            "valid_support": np.asarray(comparison.valid_support),
            "equivalent_supports": np.asarray(comparison.equivalent_supports),
            "best_support_id": (
                None
                if int(comparison.best_support_index) < 0
                else support_ids[int(comparison.best_support_index)]
            ),
            "unique_best": bool(comparison.unique_best),
        },
        "representative_support_diagnostics": {
            "support_id": support_ids[representative_index],
            "selection_status": (
                "unique-best"
                if bool(comparison.unique_best)
                else "diagnostic representative only; support was not uniquely selected"
            ),
            "condition_ids": list(batch.condition_ids),
            "state_ids": list(representative_model.hypothesis.state_ids),
            "condition_populations": np.asarray(
                representative_model.condition_populations(
                    representative_fit.optimization.parameters
                )
            ),
            "sorted_condition_populations": np.asarray(
                representative_fit.summary.sorted_condition_populations
            ),
            "posterior_predictive_profile_mean": np.asarray(
                representative_fit.summary.posterior_predictive_mean
            ),
            "optimizer_converged": bool(diagnostics.optimization_converged),
            "local_rank": int(diagnostics.local_rank),
            "parameter_count": diagnostics.parameter_count,
            "singular_values": np.asarray(diagnostics.singular_values),
            "singular_directions": np.asarray(diagnostics.singular_directions),
            "pairwise_state_separation": np.asarray(
                diagnostics.pairwise_state_separation
            ),
            "equivalent_state_pairs": np.asarray(diagnostics.equivalent_state_pairs),
            "unsupported_states": np.asarray(diagnostics.unsupported_states),
            "maximum_absolute_residual_correlation": float(
                diagnostics.maximum_absolute_residual_correlation
            ),
            "residual_assumption_supported": bool(
                diagnostics.residual_assumption_supported
            ),
            "support_complete": bool(diagnostics.support_complete),
            "unique_state_interpretation_supported": bool(
                diagnostics.unique_state_interpretation_supported
            ),
        },
        "assessment": {
            "predictive_stage_criteria": [
                criterion.to_record() for criterion in predictive_criteria
            ],
            "mixture_advantage_criterion": (
                None if mixture_criterion is None else mixture_criterion.to_record()
            ),
            "model_ids": list(assessment.model_ids),
            "model_selection_scores": np.asarray(assessment.model_selection_scores),
            "predictive_calibration_scores": np.asarray(
                assessment.predictive_calibration_scores
            ),
            "locked_scores": np.asarray(assessment.locked_scores),
            "selection_candidates": np.asarray(assessment.selection_candidates),
            "selected_model_id": assessment.selected_model_id,
            "selected_predictive_calibration_score_id": (
                assessment.selected_predictive_calibration_score_id
            ),
            "selected_locked_score_id": assessment.selected_locked_score_id,
            "assessment_id": assessment.assessment_id,
            "model_selection_unique": bool(assessment.model_selection_unique),
            "mixture_advantage_on_locked_role": float(assessment.mixture_advantage),
            "execution_valid": bool(assessment.execution_valid),
            "unique_state_interpretation_supported": bool(
                assessment.unique_state_interpretation_supported
            ),
            "interpretation": (
                "unique finite-state interpretation supported"
                if bool(assessment.unique_state_interpretation_supported)
                else (
                    "inconclusive: equivalent, unsupported, residual-correlated, "
                    "or locally nonidentifiable support remains"
                )
            ),
        },
    }
    return _json_value(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign_manifest", type=Path)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--gradient-tolerance", type=float, default=1e-6)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(
        args.campaign_manifest,
        max_steps=args.max_steps,
        gradient_tolerance=args.gradient_tolerance,
    )
    encoded = json.dumps(result, indent=2, allow_nan=False)
    if args.output is None:
        print(encoded)
    else:
        args.output.write_text(encoded + "\n")


if __name__ == "__main__":
    main()
