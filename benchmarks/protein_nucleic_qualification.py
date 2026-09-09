#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exercise frozen mechanics and gated affinity inputs without claiming affinity."""

import hashlib
import json

import numpy as np

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.applications.nucleic_acid_biophysics._construct import NucleicAcidConstruct
from phydrax.applications.protein_folding import protein_nucleic
from phydrax.applications.protein_folding._construct import ProteinConstruct
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
)
from phydrax.units import JOULE, ONE


def _reference(label: str) -> ReferenceArtifactManifest:
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
        nondimensionalization={"observable": 1.0},
        uncertainty={"declared_fixture_error": 0.1},
        lineage_ids=(f"benchmark:{label}",),
    )


def _fit(model, campaign, source):
    parameters = np.asarray([0.25, -0.5])
    parameter_id = canonical_fingerprint(
        {
            "kind": "protein-nucleic-fitted-parameters",
            "values": array_tree_fingerprint(parameters),
        }
    )
    code = _reference(f"{model}-prediction-code")
    evidence = QualificationEvidence(
        "scientific",
        "passed",
        (
            campaign.campaign_id,
            model,
            parameter_id,
            code.manifest_id,
            source.manifest_id,
        ),
        build_id="benchmark-build",
        environment_id="benchmark-environment",
        backend="cpu",
        topology="single-device",
        precision="float64",
        reduction="deterministic",
        replay_id=f"fit-replay:{model}",
        criteria_ids=("fit-execution",),
        raw_artifact_ids=(f"raw:fit:{model}",),
        reviewer_id="benchmark-reviewer",
        issued_at=1,
        expires_at=100,
        reason="fit execution passed",
        campaign_start_record_ids=(),
        campaign_observation_record_ids=(),
    )
    return ProteinNucleicModelFit(model, campaign, parameters, (source,), code, evidence)


def run() -> dict[str, object]:
    mechanics = protein_nucleic.ProteinNucleicMechanicalObservations(
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
        source=_reference("mechanics"),
    )
    mechanics_fit_source = _reference("fit-source")
    campaign = ScientificCampaign(
        (
            ScientificCase(
                "fit",
                "fit-unit",
                mechanics.complex_construct_id,
                "fit-buffer",
                "fit-prep",
                "fit-batch",
                (mechanics_fit_source.manifest_id,),
            ),
            ScientificCase(
                "case-a",
                "preparation-a",
                mechanics.complex_construct_id,
                "buffer-a",
                "prep-a",
                "locked-batch-a",
                (mechanics.source.manifest_id,),
            ),
            ScientificCase(
                "case-b",
                "preparation-b",
                mechanics.complex_construct_id,
                "buffer-b",
                "prep-b",
                "locked-batch-b",
                (mechanics.source.manifest_id,),
            ),
        ),
        (
            CampaignRole("calibration", ("fit",)),
            CampaignRole("locked_evaluation", mechanics.case_ids),
        ),
    )
    mechanics_fit = _fit("complex-model", campaign, mechanics_fit_source)
    prediction = ProteinNucleicMechanicsPrediction(
        mechanics,
        [0.2, 0.8],
        [0.05, 0.05],
        mechanics_fit,
        unit=ONE,
    )
    mechanics_result = protein_nucleic.assess_protein_nucleic_mechanics(
        mechanics,
        prediction,
        mapping_reference=None,
        maximum_standardized_rms=1.0,
        prediction_evidence=(),
    )
    affinity_fit_source = _reference("missing-fit-source")
    affinity_campaign = ScientificCampaign(
        (
            ScientificCase(
                "affinity-fit",
                "affinity-fit-unit",
                "fit-complex",
                "fit-condition",
                "affinity-fit-prep",
                "affinity-fit-batch",
                (affinity_fit_source.manifest_id,),
            ),
            ScientificCase(
                "affinity-locked",
                "affinity-locked-unit",
                "locked-complex",
                "condition-a",
                "affinity-locked-prep",
                "affinity-locked-batch",
                ("missing-observation-source",),
            ),
        ),
        (
            CampaignRole("calibration", ("affinity-fit",)),
            CampaignRole("locked_evaluation", ("affinity-locked",)),
        ),
    )
    affinity_fit = _fit("affinity-model", affinity_campaign, affinity_fit_source)
    affinity = protein_nucleic.ProteinNucleicAffinityInputs(
        ("condition-a",),
        ("affinity-locked-unit",),
        ("one-molar",),
        ("affinity-locked",),
        ("affinity-locked-prep",),
        affinity_fit,
        bound_free_energy=[-8.0],
        unbound_protein_free_energy=[-2.0],
        unbound_nucleic_free_energy=[-1.0],
        standard_state_correction=[0.5],
        bound_standard_error=[0.2],
        unbound_protein_standard_error=[0.1],
        unbound_nucleic_standard_error=[0.1],
        standard_state_standard_error=None,
        component_covariance=None,
        components_conditionally_independent=False,
        observed_binding_free_energy=[-4.5],
        observation_standard_error=[0.2],
        energy_unit=JOULE,
        bound_sampling_reference=None,
        unbound_sampling_reference=None,
        binding_measurement_reference=None,
    )
    affinity_result = protein_nucleic.assess_protein_nucleic_affinity(
        affinity,
        maximum_standardized_rms=1.0,
        prediction_evidence=(),
    )
    return {
        "mechanics_assessment_id": mechanics_result.assessment_id,
        "mechanics_campaign_id": mechanics_result.campaign_id,
        "mechanics_status": mechanics_result.status,
        "mechanics_missing_prerequisites": list(mechanics_result.missing_prerequisites),
        "affinity_assessment_id": affinity_result.assessment_id,
        "affinity_status": affinity_result.status,
        "affinity_missing_prerequisites": list(affinity_result.missing_prerequisites),
        "predicted_binding_free_energy": np.asarray(
            affinity.predicted_binding_free_energy
        ).tolist(),
        "scope": "Frozen mechanics and covariance-gated affinity arithmetic only; no calibrated claim.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
