#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Execute the four-channel schedule contract; no experimental clock is claimed."""

import hashlib
import json

import numpy as np

from phydrax.applications.systems_biology.single_cell import (
    GeneIdentity,
    LabeledTranscriptAssay,
    LabeledTranscriptCounts,
    pulse_chase_identifiability,
    PulseChasePrediction,
    PulseChaseSchedule,
    sceu_seq_prerequisites,
    scheduled_labeled_transcript_mean,
)
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import derived_unit, SECOND


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
        nondimensionalization={"time_s": 1.0},
        uncertainty={"standard_error": 0.0},
        lineage_ids=("contract-benchmark-not-experiment",),
    )


def run() -> dict[str, object]:
    schedule = PulseChaseSchedule(
        [0.0, 1.0, 2.0],
        np.asarray([[[8.0, 2.0, 1.0, 0.5]], [[8.0, 2.0, 1.0, 0.5]]]),
        [1.0, 0.0],
        rate_unit=derived_unit("s^-1", ((SECOND, -1),)),
    )
    means = np.asarray(scheduled_labeled_transcript_mean(schedule, np.zeros(4)))
    prerequisites = sceu_seq_prerequisites((None, None, None, None))
    assay = LabeledTranscriptAssay(
        np.ones(4),
        np.zeros(4),
        np.eye(2),
        labeling_calibration=_reference("label-calibration-control"),
        count_calibration=_reference("count-calibration-control"),
        calibration_covariance=np.zeros((12, 12)),
    )
    calibration = LabeledTranscriptCounts(
        GeneIdentity(1, "CONTROL"),
        (1,),
        [[0, 0, 0, 0]],
        culture_ids=("fit-culture",),
        plate_ids=("fit-plate",),
        times=[0.0],
        time_unit=SECOND,
        assay_id=assay.assay_id,
        source_id="contract-control",
        preprocessing_id="raw-four-matrix",
        source_parent_ids=("contract-source-parent",),
        preprocessing_parent_ids=("contract-preprocessing-parent",),
    )
    prediction = PulseChasePrediction(
        means[-1:],
        np.zeros((1, 4, 4)),
        model_id="contract-pulse-chase-model",
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )
    identifiability = pulse_chase_identifiability(
        np.eye(4),
        ("synthesis", "splicing", "decay", "dilution"),
        model_id=prediction.model_id,
        schedule_id=schedule.schedule_id,
        assay_id=assay.assay_id,
        preprocessing_id=calibration.preprocessing_id,
        fit_observation_ids=(calibration.observation_id,),
        fit_culture_ids=calibration.culture_ids,
        fit_plate_ids=calibration.plate_ids,
    )
    if means.shape != (3, 4) or np.any(means < 0.0):
        raise AssertionError("Four-channel pulse/chase schedule contract failed.")
    return {
        "schedule_id": schedule.schedule_id,
        "prediction_id": prediction.prediction_id,
        "identifiability_id": identifiability.evidence_id,
        "fit_culture_ids": list(prediction.fit_culture_ids),
        "fit_plate_ids": list(prediction.fit_plate_ids),
        "channel_order": [
            "labeled-unspliced",
            "labeled-spliced",
            "unlabeled-unspliced",
            "unlabeled-spliced",
        ],
        "boundary_means": means.tolist(),
        "experimental_status": "inconclusive",
        "missing_prerequisites": list(prerequisites.missing),
        "scope": "Exact schedule mechanics only; no source corpus or universal biological clock.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
