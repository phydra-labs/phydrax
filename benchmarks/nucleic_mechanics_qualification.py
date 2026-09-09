#!/usr/bin/env python3
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Exercise restricted force/twist fitting; synthetic rows are not calibration."""

import hashlib
import json

import numpy as np

from benchmarks.nucleic_rigid import make_fixture
from phydrax.applications.nucleic_acid_biophysics.coarse import (
    fit_restricted_nucleotide_mechanics,
    NucleotideMechanicalResponseData,
)
from phydrax.qualification import (
    CampaignRole,
    ReferenceArtifactManifest,
    ScientificCampaign,
    ScientificCase,
)
from phydrax.units import JOULE


def _reference() -> ReferenceArtifactManifest:
    payload = b"independent-mechanics-contract-fixture"
    return ReferenceArtifactManifest(
        "mechanics-contract-fixture",
        checksum_algorithm="sha256",
        checksum=hashlib.sha256(payload).hexdigest(),
        size_bytes=len(payload),
        license_id="CC0-1.0",
        commercial_use_permitted=True,
        redistribution_permitted=True,
        training_use_permitted=True,
        export_permitted=True,
        export_classification="unrestricted",
        nondimensionalization={"response": 1.0},
        uncertainty={"declared_fixture_error": 0.1},
        lineage_ids=("contract-benchmark-not-experiment",),
    )


def _data(prefix: str, units: tuple[str, ...], model) -> NucleotideMechanicalResponseData:
    count = len(units)
    sensitivity = np.zeros((count, 2, 3))
    sensitivity[:, 0, 0] = np.arange(1, count + 1)
    sensitivity[:, 1, 1] = np.arange(2, count + 2)
    observed = np.einsum("rop,p->ro", sensitivity[..., :2], np.asarray([0.4, -0.2]))
    return NucleotideMechanicalResponseData(
        tuple(f"{prefix}-{index}" for index in range(count)),
        units,
        tuple(f"condition-{prefix}-{index}" for index in range(count)),
        source_row_ids=tuple(f"{prefix}-source-row-{index}" for index in range(count)),
        parent_case_ids=tuple(() for _ in range(count)),
        baseline_response=np.zeros((count, 2)),
        parameter_sensitivities=sensitivity,
        observed_response=observed,
        standard_errors=np.full((count, 2), 0.1),
        force_unit=JOULE,
        twist_unit=JOULE,
        source=_reference(),
        model=model,
    )


def _campaign(
    calibration: NucleotideMechanicalResponseData,
    locked: NucleotideMechanicalResponseData,
) -> ScientificCampaign:
    cases = tuple(
        ScientificCase(
            case_id,
            unit_id,
            response.construct_id,
            condition_id,
            f"preparation-{case_id}",
            f"batch-{case_id}",
            (response.source.manifest_id,),
        )
        for response in (calibration, locked)
        for case_id, unit_id, condition_id in zip(
            response.case_ids,
            response.independent_unit_ids,
            response.condition_ids,
            strict=True,
        )
    )
    return ScientificCampaign(
        cases,
        (
            CampaignRole("calibration", calibration.case_ids),
            CampaignRole("locked_evaluation", locked.case_ids),
        ),
    )


def run() -> dict[str, object]:
    model, _ = make_fixture(4)
    calibration = _data("calibration", ("unit-a", "unit-b"), model)
    locked = _data("locked", ("unit-c", "unit-d"), model)
    assessment = fit_restricted_nucleotide_mechanics(
        calibration,
        locked,
        ("stack", "twist", "excluded-volume"),
        (0, 1),
        campaign=_campaign(calibration, locked),
        thermal_reference=None,
        structural_reference=None,
        maximum_standardized_rms=1.0,
        relative_rank_tolerance=1.0e-8,
        maximum_condition_number=1.0e6,
    )
    return {
        "assessment_id": assessment.assessment_id,
        "fitted_parameter_names": list(assessment.fitted_parameter_names),
        "fitted_offsets": np.asarray(assessment.fitted_parameter_offsets).tolist(),
        "condition_number": assessment.condition_number,
        "parameter_covariance": (
            None
            if assessment.parameter_covariance is None
            else np.asarray(assessment.parameter_covariance).tolist()
        ),
        "locked_force_macro_standardized_rms": assessment.force_macro_standardized_rms,
        "locked_twist_macro_standardized_rms": assessment.twist_macro_standardized_rms,
        "scientific_status": assessment.status,
        "missing_prerequisites": list(assessment.missing_prerequisites),
        "scope": "Restricted local linear contract only; no duplex-wide mechanics calibration.",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
