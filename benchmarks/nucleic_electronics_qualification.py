#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Compare frozen held-out transfer predictions; no damage claim is made."""

import hashlib
import json

import numpy as np

from phydrax._fingerprint import array_tree_fingerprint, canonical_fingerprint
from phydrax.applications.nucleic_acid_biophysics.electronics import (
    ChargeTransferObservationSeries,
    compare_electronic_models,
)
from phydrax.applications.nucleic_acid_biophysics.electronics._qualification import (
    ElectronicModelFit,
    ElectronicModelPrediction,
)
from phydrax.qualification import (
    CampaignRole,
    QualificationEvidence,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from phydrax.units import ONE, SECOND


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
        nondimensionalization={"signal": 1.0},
        uncertainty={"declared_fixture_error": 0.1},
        lineage_ids=(f"benchmark:{label}",),
    )


def _fit(model, campaign, source):
    parameters = np.asarray([0.25, -0.5])
    parameter_id = canonical_fingerprint(
        {
            "kind": "electronic-fitted-parameters",
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
    )
    return ElectronicModelFit(model, campaign, parameters, (source,), code, evidence)


def run() -> dict[str, object]:
    values = np.asarray([[[1.0, 0.0], [0.6, 0.4], [0.3, 0.7]]])
    observations = ChargeTransferObservationSeries(
        ("series-1",),
        ("preparation-1",),
        ("buffer-1",),
        sequence_id="AG",
        environment_id="declared-buffer-temperature",
        observable_kind="charge-transfer-population",
        times=[0.0, 1.0, 2.0],
        values=values,
        standard_errors=np.full_like(values, 0.1),
        valid=None,
        time_unit=SECOND,
        observable_unit=ONE,
        source=_reference("time-resolved-transfer-fixture"),
    )
    fit_source = _reference("fit-source")
    campaign = ScientificCampaign(
        (
            ScientificCase(
                "fit",
                "fit-unit",
                "AG",
                "fit-buffer",
                "fit-prep",
                "fit-batch",
                (fit_source.manifest_id,),
            ),
            ScientificCase(
                "series-1",
                "preparation-1",
                "AG",
                "buffer-1",
                "locked-prep",
                "locked-batch",
                (observations.source.manifest_id,),
            ),
        ),
        (
            CampaignRole("calibration", ("fit",)),
            CampaignRole("locked_evaluation", ("series-1",)),
        ),
    )
    quantum_fit = _fit("quantum", campaign, fit_source)
    kinetic_fit = _fit("kinetic", campaign, fit_source)
    quantum = ElectronicModelPrediction(
        observations,
        values + np.asarray([-0.01, 0.01]),
        quantum_fit,
        observable_unit=ONE,
    )
    kinetic = ElectronicModelPrediction(
        observations,
        values + np.asarray([-0.2, 0.2]),
        kinetic_fit,
        observable_unit=ONE,
    )
    comparison = compare_electronic_models(
        observations,
        quantum,
        kinetic,
        instrument_calibration=None,
        environment_characterization=None,
        maximum_quantum_standardized_rms=1.0,
        minimum_quantum_improvement=1.0,
        prediction_evidence=(),
    )
    return {
        "comparison_id": comparison.comparison_id,
        "campaign_id": comparison.campaign_id,
        "quantum_macro_standardized_rms": comparison.quantum_macro_standardized_rms,
        "kinetic_macro_standardized_rms": comparison.kinetic_macro_standardized_rms,
        "quantum_improvement": comparison.quantum_improvement,
        "scientific_status": comparison.status,
        "missing_prerequisites": list(comparison.missing_prerequisites),
        "scope": "One frozen held-out transfer series only; no coherence, lesion, or DNA-damage inference.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
