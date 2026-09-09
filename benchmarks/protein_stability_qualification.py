# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Run the assay-bounded protein-stability campaign on caller-supplied data.

This executable never downloads data, invents structures, or substitutes synthetic
uncertainty. The processed table, exact artifact manifest, grouping declaration,
source-backed residue environments, scalar definitions, thresholds, and execution
identity are all caller inputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from phydrax.applications.protein_folding.interchange import (
    admit_megascale_processed_table,
    prepare_protein_stability_cohort,
)
from phydrax.applications.protein_folding.stability import (
    AminoAcidScalarDefinition,
    fit_global_substitution_baseline,
    fit_protein_feature_transform,
    fit_regularized_environment_model,
    protein_mutation_features,
    ProteinResidueEnvironment,
    ProteinStabilityModelSelectionRecord,
    ProteinStabilityThresholds,
    qualify_protein_stability,
)
from phydrax.qualification import ReferenceArtifactManifest, SupportTuple


def _load_object(path: Path, name: str) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{name} must contain one JSON object.")
    return value


def run_campaign(
    table_path: Path,
    manifest_path: Path,
    declaration_path: Path,
) -> dict[str, object]:
    manifest = ReferenceArtifactManifest.from_record(
        _load_object(manifest_path, "manifest")
    )
    declaration = _load_object(declaration_path, "declaration")
    source = admit_megascale_processed_table(
        table_path,
        manifest,
        selected_names=tuple(declaration["selected_names"]),
        wt_sequence_by_background=declaration["wt_sequence_by_background"],
        censoring_by_name=declaration["censoring_by_name"],
        quality_flags_by_name=declaration["quality_flags_by_name"],
        library_id=str(declaration["library_id"]),
        condition_id=str(declaration["condition_id"]),
        target_sign_convention=str(
            declaration.get("target_sign_convention", "positive-is-stabilizing")
        ),
        nominal_model_temperature_kelvin=float(
            declaration.get("nominal_model_temperature_kelvin", 298.0)
        ),
        experimental_temperature_kelvin=(
            None
            if declaration.get("experimental_temperature_kelvin") is None
            else float(declaration["experimental_temperature_kelvin"])
        ),
        standard_error_by_name=declaration.get("standard_error_by_name"),
        standard_error_manifest=(
            None
            if declaration.get("standard_error_manifest") is None
            else ReferenceArtifactManifest.from_record(
                declaration["standard_error_manifest"]
            )
        ),
        censoring_bounds_by_name=declaration.get("censoring_bounds_by_name"),
        training_use=True,
        commercial_use=bool(declaration.get("commercial_use", False)),
    )
    role_by_domain_family = declaration["role_by_domain_family"]
    source_families = {item.domain_family_id for item in source.measurements}
    if set(role_by_domain_family) != source_families:
        raise ValueError(
            "role_by_domain_family must cover admitted domain families exactly."
        )
    role_by_group = {
        item.independent_group_id: str(role_by_domain_family[item.domain_family_id])
        for item in source.measurements
    }
    cohort = prepare_protein_stability_cohort(
        source,
        role_by_group,
        preparation_id_by_measurement=declaration["preparation_id_by_measurement"],
        batch_id_by_measurement=declaration["batch_id_by_measurement"],
        preprocessing_source_ids=tuple(declaration.get("preprocessing_source_ids", ())),
        criteria_ids=tuple(declaration.get("criteria_ids", ())),
    )
    feature_manifest_records = declaration["feature_source_manifests"]
    feature_manifests = tuple(
        ReferenceArtifactManifest.from_record(record)
        for record in feature_manifest_records
    )
    feature_manifest_by_id = {value.manifest_id: value for value in feature_manifests}
    if len(feature_manifest_by_id) != len(feature_manifests):
        raise ValueError("Feature source manifests must have unique content identities.")
    feature_requested_use = declaration["feature_requested_use"]

    def admitted_feature_manifests(source_ids):
        ids = tuple(source_ids)
        if not ids or any(value not in feature_manifest_by_id for value in ids):
            raise ValueError(
                "Every declared feature source ID needs an admitted manifest."
            )
        return tuple(feature_manifest_by_id[value] for value in ids)

    scalar_record = declaration["amino_acid_scalars"]
    scalars = AminoAcidScalarDefinition(
        scalar_record["volume"],
        scalar_record["charge"],
        scalar_record["hydropathy"],
        source_manifests=admitted_feature_manifests(scalar_record["source_manifest_ids"]),
        requested_use=feature_requested_use,
    )
    environment_records = declaration["residue_environments_by_measurement"]
    single_measurements = tuple(
        item for item in cohort.measurements if item.mutation_order == 1
    )
    if set(environment_records) != {item.measurement_id for item in single_measurements}:
        raise ValueError(
            "Residue environments must cover admitted single mutants exactly."
        )
    features = []
    for measurement in single_measurements:
        record = environment_records[measurement.measurement_id]
        environment = ProteinResidueEnvironment(
            int(record["residue_position"]),
            str(record["secondary_structure"]),
            float(record["relative_solvent_exposure"]),
            float(record["neighborhood_density"]),
            record["contact_counts"],
            float(record["phi_radians"]),
            float(record["psi_radians"]),
            hypothesis_id=str(record["hypothesis_id"]),
            residue_mapping_id=str(record["residue_mapping_id"]),
            source_manifests=admitted_feature_manifests(record["source_manifest_ids"]),
            requested_use=feature_requested_use,
        )
        features.append(
            protein_mutation_features(
                measurement,
                environment,
                scalars,
                feature_definition_id=str(declaration["feature_definition_id"]),
                requested_use=feature_requested_use,
            )
        )
    features = tuple(features)
    transform = fit_protein_feature_transform(features, cohort)
    baseline_ridge = float(declaration["baseline_ridge"])
    environment_ridge = float(declaration["environment_ridge"])
    family_effect_scale = float(declaration["family_effect_scale"])
    baseline_fit = fit_global_substitution_baseline(
        features,
        cohort,
        transform,
        ridge=baseline_ridge,
    )
    environment_fit = fit_regularized_environment_model(
        features,
        cohort,
        transform,
        ridge=environment_ridge,
        family_effect_scale=family_effect_scale,
    )
    model_selection = ProteinStabilityModelSelectionRecord(
        cohort,
        features,
        (baseline_fit, environment_fit),
        candidate_hyperparameters={
            baseline_fit.fit_id: {"ridge": baseline_ridge},
            environment_fit.fit_id: {
                "ridge": environment_ridge,
                "family_effect_scale": family_effect_scale,
            },
        },
    )
    result = qualify_protein_stability(
        cohort,
        features,
        baseline_fit,
        model_selection,
        SupportTuple.from_record(declaration["support"]),
        ProteinStabilityThresholds(**declaration["thresholds"]),
        **declaration["execution"],
    )
    return {
        "scientific_scope": (
            "Empirical MegaScale processed delta-delta-G prediction for the exact "
            "declared natural-domain cohort and assay condition; not broad protein "
            "folding, force-field, or prospective qualification."
        ),
        "source_id": source.source_id,
        "cohort_id": cohort.cohort_id,
        "campaign_id": cohort.campaign.campaign_id,
        "transform_id": transform.transform_id,
        "baseline_fit": {
            "fit_id": baseline_fit.fit_id,
            "successful": baseline_fit.successful,
            "reasons": baseline_fit.reasons,
        },
        "selected_fit": {
            "fit_id": model_selection.chosen_fit_id,
            "successful": model_selection.chosen_fit.successful,
            "reasons": model_selection.chosen_fit.reasons,
        },
        "model_selection": {
            "selection_id": model_selection.selection_id,
            "case_ids": model_selection.model_selection_case_ids,
            "candidate_scores": model_selection.candidate_scores,
        },
        "qualification": result.evidence.to_record(),
        "metrics": asdict(result.metrics),
        "predictions": [asdict(item) for item in result.predictions],
        "result_id": result.result_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_campaign(args.table, args.manifest, args.declaration)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
