#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exercise a frozen measured-time FRET prediction; no folding clock is inferred."""

import hashlib
import json

import numpy as np

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.applications.protein_folding.cotranslation import (
    assess_cotranslation_prediction,
    CotranslationObservationLaw,
    LengthResolvedCotranslationObservations,
)
from phydrax.applications.protein_folding.cotranslation._qualification import (
    CotranslationModelFit,
    CotranslationModelPrediction,
)
from phydrax.qualification import (
    CampaignRole,
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from phydrax.units import ANGSTROM, SECOND


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
            "kind": "cotranslation-fitted-parameters",
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
    return CotranslationModelFit(model, campaign, parameters, (source,), code, evidence)


def run() -> dict[str, object]:
    law = CotranslationObservationLaw(
        "length-resolved-fret",
        _reference("fret-calibration"),
        forster_radius=2.0,
        forster_radius_standard_error=0.05,
        length_unit=ANGSTROM,
    )
    observations = LengthResolvedCotranslationObservations(
        ("length-20", "length-30"),
        ("preparation-a", "preparation-b"),
        ("prep-a", "prep-b"),
        ("buffer-a", "buffer-b"),
        construct_ids=("construct-20", "construct-30"),
        nascent_lengths=[20, 30],
        measured_dwell_times=[1.0, 2.0],
        dwell_time_standard_errors=[0.1, 0.1],
        values=[0.5, 1.0 / 65.0],
        standard_errors=[0.05, 0.05],
        time_unit=SECOND,
        timing_semantics="measured-dwell-time",
        source=_reference("length-resolved-fret-fixture"),
        timing_reference=None,
    )
    fit_source = _reference("fit-source")
    campaign = ScientificCampaign(
        (
            ScientificCase(
                "fit",
                "fit-unit",
                "fit-construct",
                "fit-buffer",
                "fit-prep",
                "fit-batch",
                (fit_source.manifest_id,),
            ),
            ScientificCase(
                "length-20",
                "preparation-a",
                "construct-20",
                "buffer-a",
                "prep-a",
                "locked-batch-a",
                (observations.source.manifest_id,),
            ),
            ScientificCase(
                "length-30",
                "preparation-b",
                "construct-30",
                "buffer-b",
                "prep-b",
                "locked-batch-b",
                (observations.source.manifest_id,),
            ),
        ),
        (
            CampaignRole("calibration", ("fit",)),
            CampaignRole("locked_evaluation", observations.case_ids),
        ),
    )
    fit = _fit("frozen-model", campaign, fit_source)
    prediction = CotranslationModelPrediction(
        observations,
        [2.0, 4.0],
        [0.02, 0.02],
        fit,
        latent_unit=ANGSTROM,
    )
    assessment = assess_cotranslation_prediction(
        law,
        observations,
        prediction,
        maximum_standardized_rms=1.0,
        prediction_evidence=(),
    )
    return {
        "assessment_id": assessment.assessment_id,
        "campaign_id": assessment.campaign_id,
        "predicted_observations": np.asarray(assessment.predicted_observations).tolist(),
        "scientific_status": assessment.status,
        "missing_prerequisites": list(assessment.missing_prerequisites),
        "scope": (
            "Frozen calibrated observable prediction only; no ribosome-density "
            "dwell time or folded-fraction claim."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
