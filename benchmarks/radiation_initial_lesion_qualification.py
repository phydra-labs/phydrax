#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exercise zero-preserving history and raw-gel contracts; no transport is run."""

import hashlib
import json

import numpy as np

from phydrax.applications import radiation_biophysics as rad
from phydrax.applications.radiation_biophysics.interchange import (
    NANOMETER,
    TimedRadiationHistoryProfile,
)
from phydrax.artifacts import ScientificArtifactEnvelope
from phydrax.qualification import ReferenceArtifactManifest
from phydrax.units import ELECTRONVOLT, SECOND


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
        nondimensionalization={"count": 1.0},
        uncertainty={"declared_fixture_error": 0.1},
        lineage_ids=("contract-benchmark-not-experiment",),
    )


def run() -> dict[str, object]:
    reference = _reference("external-history-fixture")
    artifact = ScientificArtifactEnvelope(
        artifact_kind="external-history-fixture",
        content_digest=reference.checksum,
        producer="PHYDRA-benchmark",
        producer_version="1",
        build_id="contract",
        license_id=reference.license_id,
        resource_id="in-memory",
        status="complete",
    )
    source = rad.RadiationSource(
        artifact,
        (reference,),
        "external-engine",
        "declared-revision",
        "declared-configuration",
        ("declared-rng-lineage",),
        ("declared-cross-section-table",),
        (),
        "world",
        NANOMETER,
        ELECTRONVOLT,
        SECOND,
        1.0,
        "declared-chemistry",
        "declared-scavenging",
    )
    histories = tuple(
        rad.PrimaryHistoryKey(artifact.artifact_id, "run", str(index), "fraction")
        for index in range(2)
    )
    profile = TimedRadiationHistoryProfile(
        source,
        histories,
        physical_tuple_ids=("zero-primary", "one-primary"),
        physical_event_counts=[0, 1],
        dose_gy=[0.0, 1.0],
        dose_standard_errors_gy=[0.01, 0.05],
        species_ids=("OH",),
        sample_times=[0.0, 1.0],
        species_counts=np.asarray([[[0, 0]], [[1, 0]]]),
        species_valid=np.ones((2, 1, 2), dtype=bool),
        time_unit=SECOND,
        dosimetry_reference=None,
        transport_reference=None,
        chemical_reference=None,
    )
    coverage = profile.coverage(("OH",), (0.0, 1.0))
    assay = rad.PlasmidGelAssay(np.eye(3), np.zeros(3), _reference("gel-calibration"))
    gel = rad.PlasmidGelObservations(
        ("lane-1",),
        ("day-1",),
        ("one-primary",),
        preparation_ids=("locked-preparation",),
        lane_gain=[10.0],
        intensities=[[10.0, 0.0, 0.0]],
        standard_errors=[[1.0, 1.0, 1.0]],
        source=_reference("raw-gel-fixture"),
    )
    prediction = rad.PlasmidFormPrediction(
        profile,
        gel.physical_tuple_ids,
        [[1.0, 0.0, 0.0]],
        campaign_id="held-out-irradiation-days",
        model_id="contract-radiation-form-model",
        fit_id="contract-radiation-form-fit",
        prediction_source_artifact_id="contract-prediction-artifact",
        fit_physical_tuple_ids=("zero-primary",),
        fit_independent_unit_ids=("fit-day",),
        fit_preparation_ids=("fit-preparation",),
    )
    evaluation = rad.evaluate_plasmid_gel(assay, gel, prediction)
    if histories[0] not in coverage.zero_physical_histories:
        raise AssertionError("Zero-primary history was not preserved.")
    return {
        "profile_id": profile.profile_id,
        "zero_primary_ids": [
            item.primary_id for item in coverage.zero_physical_histories
        ],
        "gel_day_macro_standardized_rms": evaluation.day_macro_standardized_rms,
        "gel_prediction_id": prediction.prediction_id,
        "gel_model_id": prediction.model_id,
        "gel_fit_id": prediction.fit_id,
        "measurement_uncertainty_limitations": list(evaluation.uncertainty_limitations),
        "experimental_status": "inconclusive",
        "missing_stage_references": list(coverage.missing_references),
        "scope": (
            "Adapter and gel observation mechanics only; no native transport, "
            "chemical-G, lesion, damage, or repair claim."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
