#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import equinox as eqx
import numpy as np

from .._fingerprint import canonical_fingerprint
from .._precision import PrecisionEvidenceEnvelope
from .._strict import StrictModule
from .._trainable import NonTrainableState
from ._finite_volume_case import FiniteVolumeCaseSpec
from ._finite_volume_runtime import FiniteVolumeRuntimeState


def _array_payload(value, /) -> bytes:
    stream = io.BytesIO()
    np.save(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def _checksum(payload: bytes, /) -> str:
    return hashlib.sha256(payload).hexdigest()


class FiniteVolumeCheckpointPlan(StrictModule, NonTrainableState):
    case: FiniteVolumeCaseSpec
    checkpoint_id: str = eqx.field(static=True)

    def __init__(self, case: FiniteVolumeCaseSpec, /):
        if not isinstance(case, FiniteVolumeCaseSpec):
            raise TypeError("case must be a FiniteVolumeCaseSpec.")
        self.case = case
        self.checkpoint_id = canonical_fingerprint(
            {
                "kind": "finite-volume-checkpoint-plan",
                "schema_version": 2,
                "case": case.case_id,
                "precision_policy_id": case.precision.policy_id,
                "precision_evidence_id": case.precision.evidence().evidence_id,
                "checkpoint_dtype": case.precision.checkpoint_dtype,
            }
        )


class FiniteVolumeCheckpoint(StrictModule):
    runtime_state: FiniteVolumeRuntimeState
    checkpoint_id: str = eqx.field(static=True)
    payload_id: str = eqx.field(static=True)
    precision_evidence: PrecisionEvidenceEnvelope


def write_finite_volume_checkpoint(
    path: str | Path,
    plan: FiniteVolumeCheckpointPlan,
    runtime_state: FiniteVolumeRuntimeState,
    /,
) -> FiniteVolumeCheckpoint:
    if not isinstance(plan, FiniteVolumeCheckpointPlan):
        raise TypeError("plan must be a FiniteVolumeCheckpointPlan.")
    if not isinstance(runtime_state, FiniteVolumeRuntimeState):
        raise TypeError("runtime_state must be a FiniteVolumeRuntimeState.")
    plan.case.precision.validate_state(runtime_state.conservative_state)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dtype = plan.case.precision.numpy_dtype("checkpoint")
    arrays = {
        "conservative_state": np.asarray(
            runtime_state.conservative_state, dtype=checkpoint_dtype
        ),
        "time": np.asarray(runtime_state.time),
        "accepted_step": np.asarray(runtime_state.accepted_step, dtype=np.int64),
        "step_size": np.asarray(runtime_state.step_size),
        "last_status": np.asarray(runtime_state.last_status, dtype=np.int32),
        "controller_state": np.asarray(runtime_state.controller_state),
        "integrator_state": np.asarray(runtime_state.integrator_state),
        "forcing_state": np.asarray(runtime_state.forcing_state),
        "random_state": np.asarray(runtime_state.random_state, dtype=np.uint32),
        "output_cursor": np.asarray(runtime_state.output_cursor, dtype=np.int32),
    }
    payloads = {name: _array_payload(value) for name, value in arrays.items()}
    manifest = {
        "schema_version": 2,
        "checkpoint_id": plan.checkpoint_id,
        "case": plan.case.to_dict(),
        "precision_evidence": plan.case.precision.evidence().to_dict(),
        "arrays": {
            name: {
                "file": f"arrays/{name}.npy",
                "sha256": _checksum(payload),
                "shape": list(arrays[name].shape),
                "dtype": str(arrays[name].dtype),
            }
            for name, payload in payloads.items()
        },
    }
    manifest_payload = json.dumps(
        manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    payload_id = _checksum(manifest_payload + b"".join(payloads.values()))
    manifest["payload_id"] = payload_id
    temporary = target.with_suffix(target.suffix + ".tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        for name, payload in payloads.items():
            archive.writestr(f"arrays/{name}.npy", payload)
    os.replace(temporary, target)
    return FiniteVolumeCheckpoint(
        runtime_state,
        plan.checkpoint_id,
        payload_id,
        plan.case.precision.evidence(),
    )


def read_finite_volume_checkpoint(
    path: str | Path,
    plan: FiniteVolumeCheckpointPlan,
    /,
) -> FiniteVolumeCheckpoint:
    if not isinstance(plan, FiniteVolumeCheckpointPlan):
        raise TypeError("plan must be a FiniteVolumeCheckpointPlan.")
    with zipfile.ZipFile(Path(path), "r") as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("schema_version") != 2:
            raise ValueError("Unsupported finite-volume checkpoint schema.")
        if manifest.get("checkpoint_id") != plan.checkpoint_id:
            raise ValueError("Finite-volume checkpoint is incompatible with this plan.")
        FiniteVolumeCaseSpec.validate_dict(manifest["case"])
        if manifest["case"]["case_id"] != plan.case.case_id:
            raise ValueError("Finite-volume checkpoint case identity changed.")
        precision_evidence = PrecisionEvidenceEnvelope.from_dict(
            manifest["precision_evidence"]
        )
        expected_precision = plan.case.precision.evidence()
        if precision_evidence.evidence_id != expected_precision.evidence_id:
            raise ValueError("Finite-volume checkpoint precision evidence changed.")
        arrays = {}
        payloads = []
        for name in (
            "conservative_state",
            "time",
            "accepted_step",
            "step_size",
            "last_status",
            "controller_state",
            "integrator_state",
            "forcing_state",
            "random_state",
            "output_cursor",
        ):
            metadata = manifest["arrays"][name]
            payload = archive.read(metadata["file"])
            if _checksum(payload) != metadata["sha256"]:
                raise ValueError(f"Finite-volume checkpoint array {name!r} is corrupt.")
            value = np.load(io.BytesIO(payload), allow_pickle=False)
            if (
                list(value.shape) != metadata["shape"]
                or str(value.dtype) != metadata["dtype"]
            ):
                raise ValueError(
                    f"Finite-volume checkpoint array {name!r} metadata changed."
                )
            arrays[name] = value
            payloads.append(payload)
    manifest_without_payload = dict(manifest)
    expected_payload_id = manifest_without_payload.pop("payload_id")
    manifest_payload = json.dumps(
        manifest_without_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    actual_payload_id = _checksum(manifest_payload + b"".join(payloads))
    if actual_payload_id != expected_payload_id:
        raise ValueError("Finite-volume checkpoint manifest or payload is corrupt.")
    runtime = FiniteVolumeRuntimeState(
        plan.case.precision.storage(arrays["conservative_state"]),
        plan.case.precision.decision(arrays["time"]),
        plan.case.precision.decision(arrays["step_size"]),
        accepted_step=arrays["accepted_step"],
        last_status=arrays["last_status"],
        controller_state=arrays["controller_state"],
        integrator_state=arrays["integrator_state"],
        forcing_state=arrays["forcing_state"],
        random_state=arrays["random_state"],
        output_cursor=arrays["output_cursor"],
    )
    return FiniteVolumeCheckpoint(
        runtime,
        plan.checkpoint_id,
        actual_payload_id,
        precision_evidence,
    )


__all__ = [
    "FiniteVolumeCheckpoint",
    "FiniteVolumeCheckpointPlan",
    "read_finite_volume_checkpoint",
    "write_finite_volume_checkpoint",
]
