#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import numpy as np

from phydrax._array_archive import read_array_archive, write_array_archive
from phydrax._precision import (
    PrecisionEvidenceEnvelope,
    PrecisionRequest,
    PrecisionResolution,
)
from phydrax.operators.quantum import (
    ApproximationAxis,
    ApproximationQuantity,
    OpenSystemApproximationEvidence,
    OpenSystemPhysicalityEvidence,
)

from .contracts import (
    CampaignCapacityEvidence,
    CampaignPrecisionBundle,
    OpenSystemCampaignRecord,
    SemanticReplayEvidence,
    VerifiedOpenSystemCampaign,
)


_REQUIRED_FIELDS = {
    "problem_id",
    "plan_id",
    "backend",
    "runner_id",
    "code_fingerprint",
    "record",
}


def _scalar(value: Any, /) -> float:
    array = np.asarray(value)
    if array.shape != () or not np.isfinite(array):
        raise ValueError("Artifact evidence values must be finite scalars.")
    return float(array)


def _record_manifest(record: OpenSystemCampaignRecord, /) -> dict[str, Any]:
    approximation = record.approximation
    physicality = record.physicality
    replay = record.replay
    return {
        "campaign_id": record.campaign_id,
        "representation_id": record.representation_id,
        "execution_success": bool(record.execution_success),
        "approximation": {
            "representation_id": approximation.representation_id,
            "valid": bool(approximation.valid),
            "axes": [
                {
                    "name": axis.name,
                    "value": _scalar(axis.value),
                    "parent_value": (
                        None
                        if axis.parent_value is None
                        else _scalar(axis.parent_value)
                    ),
                    "units": axis.units,
                }
                for axis in approximation.axes
            ],
            "quantities": [
                {
                    "name": quantity.name,
                    "value": _scalar(quantity.value),
                    "threshold": _scalar(quantity.threshold),
                    "units": quantity.units,
                    "norm_id": quantity.norm_id,
                    "estimate_kind": quantity.estimate_kind,
                    "confidence": (
                        _scalar(quantity.confidence)
                        if np.isfinite(np.asarray(quantity.confidence))
                        else None
                    ),
                }
                for quantity in approximation.quantities
            ],
            "precision_policy_ids": list(approximation.precision_policy_ids),
        },
        "physicality": {
            "certified_properties": list(physicality.certified_properties),
            "status": physicality.status,
            "trace_residual": _optional_scalar(physicality.trace_residual),
            "hermiticity_residual": _optional_scalar(
                physicality.hermiticity_residual
            ),
            "positivity_margin": _optional_scalar(physicality.positivity_margin),
            "channel_cp_margin": _optional_scalar(physicality.channel_cp_margin),
            "trace_preservation_residual": _optional_scalar(
                physicality.trace_preservation_residual
            ),
            "closure_residual": _optional_scalar(physicality.closure_residual),
            "trace_tolerance": _scalar(physicality.trace_tolerance),
            "hermiticity_tolerance": _scalar(physicality.hermiticity_tolerance),
            "positivity_tolerance": _scalar(physicality.positivity_tolerance),
            "channel_cp_tolerance": _scalar(physicality.channel_cp_tolerance),
            "trace_preservation_tolerance": _scalar(
                physicality.trace_preservation_tolerance
            ),
            "closure_tolerance": _scalar(physicality.closure_tolerance),
        },
        "replay": {
            "variates_equal": bool(replay.variates_equal),
            "address_schema_equal": bool(replay.address_schema_equal),
            "event_time_difference": _scalar(replay.event_time_difference),
            "channel_disagreement_probability": _scalar(
                replay.channel_disagreement_probability
            ),
            "observable_difference": _scalar(replay.observable_difference),
            "event_time_tolerance": _scalar(replay.event_time_tolerance),
            "disagreement_tolerance": _scalar(replay.disagreement_tolerance),
            "observable_tolerance": _scalar(replay.observable_tolerance),
        },
        "capacity_evidence": [
            {
                "name": capacity.name,
                "used": capacity.used,
                "limit": capacity.limit,
                "saturated": bool(capacity.saturated),
            }
            for capacity in record.capacity_evidence
        ],
        "work": dict(record.work),
        "unsupported_claims": list(record.unsupported_claims),
        "artifact_names": list(record.artifact_names),
        "precision_request": record.precision.request.to_dict(),
        "precision_resolution": record.precision.resolution.to_dict(),
        "precision_evidence": record.precision.evidence.to_dict(),
    }


def _optional_scalar(value: Any, /) -> float | None:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError("Artifact physicality values must be scalar.")
    return float(array) if np.isfinite(array) else None


def _record_arrays(record: OpenSystemCampaignRecord, /) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(value)
        for name, value in zip(
            record.artifact_names, record.artifact_arrays, strict=True
        )
    }


def write_open_system_artifact(
    path: str | os.PathLike[str],
    record: OpenSystemCampaignRecord,
    /,
    *,
    problem_id: str,
    plan_id: str,
    backend: str,
    runner_id: str,
    code_fingerprint: str,
) -> Path:
    """Write one complete, unverified open-system campaign artifact."""
    if not isinstance(record, OpenSystemCampaignRecord):
        raise TypeError("record must be an OpenSystemCampaignRecord.")
    identifiers = tuple(
        str(value)
        for value in (problem_id, plan_id, backend, runner_id, code_fingerprint)
    )
    if any(not value for value in identifiers):
        raise ValueError("Artifact provenance identifiers must be non-empty.")
    manifest = {
        "problem_id": identifiers[0],
        "plan_id": identifiers[1],
        "backend": identifiers[2],
        "runner_id": identifiers[3],
        "code_fingerprint": identifiers[4],
        "record": _record_manifest(record),
    }
    return write_array_archive(
        path,
        manifest=manifest,
        arrays=_record_arrays(record),
    )


def _precision_bundle(payload: dict[str, Any], /) -> CampaignPrecisionBundle:
    request = PrecisionRequest.from_dict(payload["precision_request"])
    resolution = PrecisionResolution.from_dict(payload["precision_resolution"])
    evidence = PrecisionEvidenceEnvelope.from_dict(payload["precision_evidence"])
    if resolution.request_id != request.request_id:
        raise ValueError("Artifact precision resolution/request linkage failed.")
    if evidence.resolution_id != resolution.resolution_id:
        raise ValueError("Artifact precision evidence/resolution linkage failed.")
    bundle = CampaignPrecisionBundle(
        request.domain,
        resolution.provider,
        dict(resolution.effective),
        children=dict(evidence.children),
    )
    if (
        bundle.request.request_id != request.request_id
        or bundle.resolution.resolution_id != resolution.resolution_id
        or bundle.evidence.evidence_id != evidence.evidence_id
    ):
        raise ValueError("Artifact precision contracts do not reconstruct exactly.")
    return bundle


def _physicality(payload: dict[str, Any], precision: CampaignPrecisionBundle):
    optional = lambda name: np.nan if payload[name] is None else payload[name]
    evidence = OpenSystemPhysicalityEvidence(
        trace_residual=optional("trace_residual"),
        hermiticity_residual=optional("hermiticity_residual"),
        positivity_margin=optional("positivity_margin"),
        channel_cp_margin=optional("channel_cp_margin"),
        trace_preservation_residual=optional("trace_preservation_residual"),
        closure_residual=optional("closure_residual"),
        trace_tolerance=payload["trace_tolerance"],
        hermiticity_tolerance=payload["hermiticity_tolerance"],
        positivity_tolerance=payload["positivity_tolerance"],
        channel_cp_tolerance=payload["channel_cp_tolerance"],
        trace_preservation_tolerance=payload[
            "trace_preservation_tolerance"
        ],
        closure_tolerance=payload["closure_tolerance"],
        certified_properties=payload["certified_properties"],
        precision_evidence=precision.evidence,
    )
    if evidence.status != payload["status"]:
        raise ValueError("Artifact physicality status does not reproduce.")
    return evidence


def _record_from_manifest(
    payload: dict[str, Any], arrays: dict[str, np.ndarray], /
) -> OpenSystemCampaignRecord:
    expected_names = tuple(payload["artifact_names"])
    if tuple(sorted(arrays)) != tuple(sorted(expected_names)):
        raise ValueError("Artifact array inventory and campaign record differ.")
    precision = _precision_bundle(payload)
    approximation_payload = payload["approximation"]
    axes = tuple(
        ApproximationAxis(
            value["name"],
            value["value"],
            parent_value=value["parent_value"],
            units=value["units"],
        )
        for value in approximation_payload["axes"]
    )
    quantities = tuple(
        ApproximationQuantity(
            value["name"],
            value["value"],
            value["threshold"],
            units=value["units"],
            norm_id=value["norm_id"],
            estimate_kind=value["estimate_kind"],
            confidence=(
                np.nan if value["confidence"] is None else value["confidence"]
            ),
        )
        for value in approximation_payload["quantities"]
    )
    approximation = OpenSystemApproximationEvidence(
        approximation_payload["representation_id"],
        axes,
        quantities,
        execution_valid=payload["execution_success"],
        precision_evidence=precision.evidence,
        precision_policy_ids=approximation_payload["precision_policy_ids"],
    )
    if bool(approximation.valid) != bool(approximation_payload["valid"]):
        raise ValueError("Artifact approximation validity does not reproduce.")
    replay_payload = payload["replay"]
    replay = SemanticReplayEvidence(
        variates_equal=replay_payload["variates_equal"],
        address_schema_equal=replay_payload["address_schema_equal"],
        event_time_difference=replay_payload["event_time_difference"],
        channel_disagreement_probability=replay_payload[
            "channel_disagreement_probability"
        ],
        observable_difference=replay_payload["observable_difference"],
        event_time_tolerance=replay_payload["event_time_tolerance"],
        disagreement_tolerance=replay_payload["disagreement_tolerance"],
        observable_tolerance=replay_payload["observable_tolerance"],
    )
    capacities = tuple(
        CampaignCapacityEvidence(
            value["name"],
            value["used"],
            value["limit"],
            saturated=value["saturated"],
        )
        for value in payload["capacity_evidence"]
    )
    return OpenSystemCampaignRecord(
        payload["campaign_id"],
        payload["representation_id"],
        approximation,
        _physicality(payload["physicality"], precision),
        precision,
        replay,
        execution_success=payload["execution_success"],
        capacity_evidence=capacities,
        artifact_arrays=arrays,
        work=payload["work"],
        unsupported_claims=payload["unsupported_claims"],
    )


def read_open_system_artifact(
    path: str | os.PathLike[str],
    /,
    *,
    expected_campaign_id: str | None = None,
    expected_representation_id: str | None = None,
    expected_runner_id: str | None = None,
) -> tuple[OpenSystemCampaignRecord, dict[str, Any]]:
    manifest, arrays = read_array_archive(path)
    missing = _REQUIRED_FIELDS.difference(manifest)
    if missing:
        raise ValueError(f"Open-system artifact is missing fields: {sorted(missing)}")
    record = _record_from_manifest(manifest["record"], arrays)
    if expected_campaign_id is not None and record.campaign_id != expected_campaign_id:
        raise ValueError("Open-system campaign identity mismatch.")
    if (
        expected_representation_id is not None
        and record.representation_id != expected_representation_id
    ):
        raise ValueError("Open-system representation identity mismatch.")
    if expected_runner_id is not None and manifest["runner_id"] != expected_runner_id:
        raise ValueError("Open-system runner identity mismatch.")
    return record, manifest


def _records_equal(
    stored: OpenSystemCampaignRecord,
    reproduced: OpenSystemCampaignRecord,
    /,
) -> bool:
    if _record_manifest(stored) != _record_manifest(reproduced):
        return False
    stored_arrays = _record_arrays(stored)
    reproduced_arrays = _record_arrays(reproduced)
    return stored_arrays.keys() == reproduced_arrays.keys() and all(
        np.array_equal(
            stored_arrays[name],
            reproduced_arrays[name],
            equal_nan=True,
        )
        for name in stored_arrays
    )


def verify_open_system_artifact(
    path: str | os.PathLike[str],
    reproduced: OpenSystemCampaignRecord,
    /,
    *,
    expected_runner_id: str,
) -> VerifiedOpenSystemCampaign:
    """Verify integrity and exact independent campaign reproduction."""
    stored, _ = read_open_system_artifact(
        path,
        expected_campaign_id=reproduced.campaign_id,
        expected_representation_id=reproduced.representation_id,
        expected_runner_id=expected_runner_id,
    )
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return VerifiedOpenSystemCampaign(
        stored,
        digest,
        reproduction_verified=_records_equal(stored, reproduced),
    )


__all__ = [
    "read_open_system_artifact",
    "verify_open_system_artifact",
    "write_open_system_artifact",
]
