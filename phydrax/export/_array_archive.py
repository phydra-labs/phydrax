#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np

from .._array_archive import (
    ArrayArchiveCorruptionError,
    ArrayArchiveLimits,
    DEFAULT_ARRAY_ARCHIVE_LIMITS,
    read_array_archive,
    write_array_archive,
)
from .._validation import canonical_identifier as _canonical_identifier


BEMArchiveRecordKind: TypeAlias = Literal["plan", "result"]
_BEM_ARCHIVE_FORMAT = "phydrax-bem-array-record"


@dataclass(frozen=True, slots=True)
class BEMArchiveLimits:
    """Hard pickle-free BEM archive member, manifest, and file limits."""

    max_file_bytes: int = 1_073_741_824
    max_manifest_bytes: int = 1_048_576
    max_arrays: int = 256
    max_array_bytes: int = 268_435_456
    max_total_array_bytes: int = 1_073_741_824

    def __post_init__(self) -> None:
        values = (
            self.max_file_bytes,
            self.max_manifest_bytes,
            self.max_arrays,
            self.max_array_bytes,
            self.max_total_array_bytes,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in values
        ):
            raise ValueError("BEM archive limits must be positive integers.")


@dataclass(frozen=True, slots=True)
class BEMArchiveDescriptor:
    """Scientific envelope persisted with every BEM plan or result record.

    The descriptor is deliberately explicit about dimensionality, PDE, geometry,
    formulation, provider, precision, resource/error evidence, and non-goals. It
    can only describe prepared-discrete evidence; continuum certification is
    always false.
    """

    ambient_dimension: int
    pde: str
    geometry: str
    formulation: str
    provider: str
    precision: str
    resource_evidence: tuple[str, ...]
    error_evidence: tuple[str, ...]
    non_goals: tuple[str, ...]
    continuum_certified: bool = False

    def __post_init__(self) -> None:
        values = (
            self.pde,
            self.geometry,
            self.formulation,
            self.provider,
            self.precision,
        )
        if self.ambient_dimension not in (2, 3):
            raise ValueError("BEM archive dimensionality must be 2 or 3.")
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in values
        ):
            raise ValueError(
                "BEM archive descriptor strings must be non-empty and stripped."
            )
        groups = (self.resource_evidence, self.error_evidence, self.non_goals)
        if any(not isinstance(group, tuple) or not group for group in groups):
            raise TypeError(
                "BEM archive evidence and non-goals must be non-empty tuples."
            )
        evidence = self.resource_evidence + self.error_evidence + self.non_goals
        if any(
            not isinstance(value, str) or not value or value != value.strip()
            for value in evidence
        ):
            raise ValueError(
                "BEM archive evidence and non-goals must be non-empty stripped strings."
            )
        if self.continuum_certified is not False:
            raise ValueError("BEM array records cannot claim continuum certification.")


@dataclass(frozen=True, slots=True)
class BEMPlanArchiveRecord:
    """Portable pickle-free prepared BEM plan arrays and exact envelope."""

    plan_id: str
    descriptor: BEMArchiveDescriptor
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        identifier = _canonical_identifier(self.plan_id, "plan_id")
        if not isinstance(self.descriptor, BEMArchiveDescriptor):
            raise TypeError("descriptor must be BEMArchiveDescriptor.")
        metadata = _normalized_metadata(self.metadata)
        arrays = _normalized_arrays(self.arrays, allow_nonfinite=False)
        object.__setattr__(self, "plan_id", identifier)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "arrays", MappingProxyType(arrays))


@dataclass(frozen=True, slots=True)
class BEMResultArchiveRecord:
    """Portable pickle-free BEM result arrays bound to one exact plan envelope."""

    result_id: str
    plan_id: str
    descriptor: BEMArchiveDescriptor
    metadata: Mapping[str, Any]
    arrays: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        result = _canonical_identifier(self.result_id, "result_id")
        plan = _canonical_identifier(self.plan_id, "plan_id")
        if not isinstance(self.descriptor, BEMArchiveDescriptor):
            raise TypeError("descriptor must be BEMArchiveDescriptor.")
        metadata = _normalized_metadata(self.metadata)
        arrays = _normalized_arrays(self.arrays, allow_nonfinite=True)
        object.__setattr__(self, "result_id", result)
        object.__setattr__(self, "plan_id", plan)
        object.__setattr__(self, "metadata", MappingProxyType(metadata))
        object.__setattr__(self, "arrays", MappingProxyType(arrays))


BEMArrayArchiveRecord: TypeAlias = BEMPlanArchiveRecord | BEMResultArchiveRecord


def _normalized_metadata(metadata: Mapping[str, Any], /) -> dict[str, Any]:
    if not isinstance(metadata, Mapping):
        raise TypeError("BEM archive metadata must be a mapping.")
    normalized = dict(metadata)
    if any(not isinstance(key, str) or not key for key in normalized):
        raise TypeError("BEM archive metadata keys must be non-empty strings.")
    try:
        payload = json.dumps(normalized, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "BEM archive metadata must contain finite JSON values."
        ) from error
    canonical = json.loads(payload)
    if not isinstance(canonical, dict):
        raise TypeError("BEM archive metadata must encode one JSON object.")
    return canonical


def _normalized_arrays(
    arrays: Mapping[str, Any],
    /,
    *,
    allow_nonfinite: bool,
) -> dict[str, np.ndarray]:
    if not isinstance(arrays, Mapping) or not arrays:
        raise TypeError("BEM archive arrays must be a non-empty mapping.")
    normalized: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or "\\" in name
            or any(part in ("", ".", "..") for part in name.split("/"))
        ):
            raise ValueError("BEM archive array names must be canonical logical paths.")
        array = np.asarray(value)
        if array.dtype.hasobject or array.dtype.kind not in "biufc":
            raise TypeError("BEM archive arrays must have numeric or boolean dtype.")
        if not allow_nonfinite and not np.all(np.isfinite(array)):
            raise ValueError("Prepared BEM plan arrays must contain only finite values.")
        if array.flags.writeable:
            array = array.copy()
            array.setflags(write=False)
        normalized[name] = array
    return normalized


def _descriptor_manifest(descriptor: BEMArchiveDescriptor, /) -> dict[str, Any]:
    return {
        "ambient_dimension": descriptor.ambient_dimension,
        "pde": descriptor.pde,
        "geometry": descriptor.geometry,
        "formulation": descriptor.formulation,
        "provider": descriptor.provider,
        "precision": descriptor.precision,
        "resource_evidence": list(descriptor.resource_evidence),
        "error_evidence": list(descriptor.error_evidence),
        "non_goals": list(descriptor.non_goals),
        "continuum_certified": False,
    }


def _descriptor_from_manifest(value: Any, /) -> BEMArchiveDescriptor:
    if not isinstance(value, dict) or set(value) != {
        "ambient_dimension",
        "pde",
        "geometry",
        "formulation",
        "provider",
        "precision",
        "resource_evidence",
        "error_evidence",
        "non_goals",
        "continuum_certified",
    }:
        raise ArrayArchiveCorruptionError("BEM archive descriptor fields are invalid.")
    if any(
        not isinstance(value[name], list)
        for name in ("resource_evidence", "error_evidence", "non_goals")
    ):
        raise ArrayArchiveCorruptionError("BEM archive evidence must be JSON arrays.")
    try:
        return BEMArchiveDescriptor(
            ambient_dimension=value["ambient_dimension"],
            pde=value["pde"],
            geometry=value["geometry"],
            formulation=value["formulation"],
            provider=value["provider"],
            precision=value["precision"],
            resource_evidence=tuple(value["resource_evidence"]),
            error_evidence=tuple(value["error_evidence"]),
            non_goals=tuple(value["non_goals"]),
            continuum_certified=value["continuum_certified"],
        )
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError("BEM archive descriptor is invalid.") from error


def _record_parts(
    record: BEMArrayArchiveRecord, /
) -> tuple[
    BEMArchiveRecordKind,
    str,
    str,
    BEMArchiveDescriptor,
    Mapping[str, Any],
    Mapping[str, np.ndarray],
]:
    if isinstance(record, BEMPlanArchiveRecord):
        return (
            "plan",
            record.plan_id,
            record.plan_id,
            record.descriptor,
            record.metadata,
            record.arrays,
        )
    if isinstance(record, BEMResultArchiveRecord):
        return (
            "result",
            record.result_id,
            record.plan_id,
            record.descriptor,
            record.metadata,
            record.arrays,
        )
    raise TypeError("record must be BEMPlanArchiveRecord or BEMResultArchiveRecord.")


def _payload_id(
    record_kind: BEMArchiveRecordKind,
    record_id: str,
    plan_id: str,
    descriptor: BEMArchiveDescriptor,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    /,
) -> str:
    digest = hashlib.sha256()
    header = {
        "record_kind": record_kind,
        "record_id": record_id,
        "plan_id": plan_id,
        "descriptor": _descriptor_manifest(descriptor),
        "metadata": dict(metadata),
    }
    digest.update(
        json.dumps(
            header,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    )
    for name in sorted(arrays):
        array = np.asarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(
            json.dumps(list(array.shape), separators=(",", ":")).encode("ascii")
        )
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _manifest_for_record(record: BEMArrayArchiveRecord, /) -> dict[str, Any]:
    kind, record_id, plan_id, descriptor, metadata, arrays = _record_parts(record)
    return {
        "format": _BEM_ARCHIVE_FORMAT,
        "record_kind": kind,
        "record_id": record_id,
        "plan_id": plan_id,
        "descriptor": _descriptor_manifest(descriptor),
        "metadata": dict(metadata),
        "payload_id": _payload_id(kind, record_id, plan_id, descriptor, metadata, arrays),
    }


def _array_archive_limits(limits: BEMArchiveLimits, /) -> ArrayArchiveLimits:
    return replace(
        DEFAULT_ARRAY_ARCHIVE_LIMITS,
        max_container_bytes=limits.max_file_bytes,
        max_aggregate_bytes=(limits.max_total_array_bytes + limits.max_manifest_bytes),
        max_member_bytes=limits.max_array_bytes,
        max_manifest_bytes=limits.max_manifest_bytes,
        max_members=limits.max_arrays + 1,
    )


def _bem_limit_message(message: str, /) -> str:
    mappings = (
        ("aggregate member byte limit", "max_total_array_bytes"),
        ("container byte limit", "max_file_bytes"),
        ("manifest byte limit", "max_manifest_bytes"),
        ("member byte limit", "max_array_bytes"),
        ("member count limit", "max_arrays"),
    )
    for phrase, name in mappings:
        if phrase in message:
            return f"BEM archive exceeds {name}."
    return message


def write_bem_array_archive(
    path: str | os.PathLike[str],
    record: BEMArrayArchiveRecord,
    /,
    *,
    limits: BEMArchiveLimits | None = None,
) -> Path:
    """Atomically write one bounded, checksum-protected, pickle-free BEM record."""
    policy = BEMArchiveLimits() if limits is None else limits
    if not isinstance(policy, BEMArchiveLimits):
        raise TypeError("limits must be BEMArchiveLimits or None.")
    manifest = _manifest_for_record(record)
    arrays = _record_parts(record)[5]
    try:
        return write_array_archive(
            path,
            manifest=manifest,
            arrays=arrays,
            limits=_array_archive_limits(policy),
        )
    except ValueError as error:
        raise ValueError(_bem_limit_message(str(error))) from error


def read_bem_array_archive(
    path: str | os.PathLike[str],
    /,
    *,
    limits: BEMArchiveLimits | None = None,
) -> BEMArrayArchiveRecord:
    """Read a bounded BEM plan/result record after size and checksum validation."""
    policy = BEMArchiveLimits() if limits is None else limits
    if not isinstance(policy, BEMArchiveLimits):
        raise TypeError("limits must be BEMArchiveLimits or None.")
    source = Path(path)
    try:
        manifest, arrays = read_array_archive(
            source,
            limits=_array_archive_limits(policy),
        )
    except ArrayArchiveCorruptionError as error:
        raise ArrayArchiveCorruptionError(_bem_limit_message(str(error))) from error
    expected_fields = {
        "format",
        "record_kind",
        "record_id",
        "plan_id",
        "descriptor",
        "metadata",
        "payload_id",
        "arrays",
    }
    if set(manifest) != expected_fields or manifest.get("format") != _BEM_ARCHIVE_FORMAT:
        raise ArrayArchiveCorruptionError("Archive is not a canonical BEM array record.")
    kind = manifest["record_kind"]
    if kind not in ("plan", "result"):
        raise ArrayArchiveCorruptionError("BEM archive record_kind is invalid.")
    try:
        record_id = _canonical_identifier(manifest["record_id"], "record_id")
        plan_id = _canonical_identifier(manifest["plan_id"], "plan_id")
        metadata = _normalized_metadata(manifest["metadata"])
    except (TypeError, ValueError) as error:
        raise ArrayArchiveCorruptionError(
            "BEM archive identities or metadata are invalid."
        ) from error
    descriptor = _descriptor_from_manifest(manifest["descriptor"])
    payload_id = _payload_id(
        kind,
        record_id,
        plan_id,
        descriptor,
        metadata,
        arrays,
    )
    if manifest["payload_id"] != payload_id:
        raise ArrayArchiveCorruptionError("BEM archive payload identity is inconsistent.")
    if kind == "plan":
        if record_id != plan_id:
            raise ArrayArchiveCorruptionError("BEM plan record identity is inconsistent.")
        return BEMPlanArchiveRecord(plan_id, descriptor, metadata, arrays)
    return BEMResultArchiveRecord(record_id, plan_id, descriptor, metadata, arrays)


def bem_archive_descriptor(envelope: object, /) -> BEMArchiveDescriptor:
    """Convert one declared BEM execution envelope without strengthening claims."""
    from ..operators.integral.layer_potential._fast_provider import BEMExecutionEnvelope

    if not isinstance(envelope, BEMExecutionEnvelope):
        raise TypeError("envelope must be BEMExecutionEnvelope.")
    return BEMArchiveDescriptor(
        ambient_dimension=envelope.ambient_dimension,
        pde=envelope.pde,
        geometry=envelope.geometry,
        formulation=envelope.formulation,
        provider=envelope.provider,
        precision=envelope.precision,
        resource_evidence=envelope.resource_evidence,
        error_evidence=envelope.error_evidence,
        non_goals=envelope.non_goals,
        continuum_certified=False,
    )


def laplace_dp0_plan_archive_record(prepared: object, /) -> BEMPlanArchiveRecord:
    """Capture the portable numeric state of one prepared 3D Laplace DP0 plan."""
    from ..operators.integral.layer_potential._fast_provider import (
        LaplaceDP0ExactNearProvider3D,
    )
    from ..operators.integral.layer_potential._galerkin3d import (
        _LaplaceDP0WeakOperator3D,
        LaplaceSingleLayerDP0Galerkin3D,
    )

    if not isinstance(prepared, LaplaceSingleLayerDP0Galerkin3D):
        raise TypeError("prepared must be LaplaceSingleLayerDP0Galerkin3D.")
    weak = prepared.weak_operator
    if not isinstance(weak, _LaplaceDP0WeakOperator3D):
        raise TypeError("Prepared BEM does not contain the supported blocked weak plan.")
    provider = LaplaceDP0ExactNearProvider3D(prepared, formulation="strong")
    pair_data = weak.pair_data
    report = prepared.assembly_report
    arrays: dict[str, Any] = {
        "geometry/vertices": prepared._binding.region.vertices,
        "geometry/faces": prepared._binding.region.faces,
        "geometry/face_areas": prepared.face_areas,
        "geometry/face_component_ids": prepared.face_component_ids,
        "panelization/chart_indices": prepared.panelization.chart_indices,
        "panelization/references": prepared.panelization.references,
        "panelization/panel_reference_vertices": (
            prepared.panelization.panel_reference_vertices
        ),
        "panelization/points": prepared.panelization.points,
        "panelization/normals": prepared.panelization.normals,
        "panelization/weights": prepared.panelization.weights,
        "panelization/panel_ids": prepared.panelization.panel_ids,
        "panelization/panel_reference_inverses": (
            prepared.panelization.panel_reference_inverses
        ),
        "pairs/exception_keys": pair_data.exception_keys,
        "pairs/targets": pair_data.targets,
        "pairs/sources": pair_data.sources,
        "pairs/classes": pair_data.classes,
        "pairs/values": pair_data.values,
        "quadrature/regular_points": pair_data.regular_points,
        "quadrature/regular_weights": pair_data.regular_weights,
        "evidence/maximum_errors": pair_data.maximum_errors,
        "evidence/pair_class_tolerances": pair_data.maximum_tolerances,
        "evidence/pair_class_supported": pair_data.supported,
        "evidence/evaluations": pair_data.evaluations,
        "evidence/finite": report.finite,
        "evidence/accuracy_supported": report.accuracy_supported,
    }
    if prepared.dense_oracle is not None:
        arrays["oracle/strong_matrix"] = prepared.dense_oracle.matrix
    metadata = {
        "binding_id": report.binding_id,
        "policy_id": report.policy_id,
        "kernel_id": report.kernel_id,
        "numeric_version": report.numeric_version,
        "face_count": report.face_count,
        "component_count": report.component_count,
        "pair_counts": list(report.pair_counts),
        "pair_class_names": list(report.pair_class_names),
        "pair_class_workspace_bytes": list(report.pair_class_workspace_bytes),
        "pair_class_resident_bytes": list(report.pair_class_resident_bytes),
        "exception_count": report.exception_count,
        "preparation_workspace_bytes": report.preparation_workspace_bytes,
        "resident_bytes": report.resident_bytes,
        "action_workspace_bytes_per_rhs": report.action_workspace_bytes_per_rhs,
        "panelization_id": prepared.panelization.panelization_id,
        "quadrature_order": prepared.panelization.quadrature_order,
        "nodes_per_panel": prepared.panelization.nodes_per_panel,
        "quadrature_rule_id": prepared.panelization.quadrature_rule_id,
        "dense_oracle_available": report.dense_oracle_available,
        "dense_oracle_bytes": report.dense_oracle_bytes,
        "materializable": report.materializable,
        "continuum_discretization_error_estimated": (
            report.continuum_discretization_error_estimated
        ),
        "report_id": report.report_id,
    }
    return BEMPlanArchiveRecord(
        report.report_id,
        bem_archive_descriptor(provider.envelope),
        metadata,
        arrays,
    )


def fused_bem_result_archive_record(
    result: object,
    plan_id: str,
    /,
) -> BEMResultArchiveRecord:
    """Capture fused values and per-column status, including failed NaN columns."""
    from ..operators.integral.layer_potential._fast_provider import (
        FusedBEMBlockActionResult,
    )

    if not isinstance(result, FusedBEMBlockActionResult):
        raise TypeError("result must be FusedBEMBlockActionResult.")
    plan = _canonical_identifier(plan_id, "plan_id")
    arrays = {
        "values": np.asarray(result.values),
        "column_status": np.asarray(result.column_status),
        "finite": np.asarray(result.finite),
    }
    content = hashlib.sha256()
    content.update(result.action_id.encode("utf-8"))
    content.update(str(bool(result.transpose)).encode("ascii"))
    for name in sorted(arrays):
        array = arrays[name]
        content.update(name.encode("utf-8"))
        content.update(array.dtype.str.encode("ascii"))
        content.update(array.tobytes(order="C"))
    result_id = f"fused-bem-result-{content.hexdigest()}"
    metadata = {
        "action_id": result.action_id,
        "transpose": result.transpose,
        "rhs_count": result.column_status.size,
        "status_semantics": {
            "0": "success",
            "1": "nonfinite_input",
            "2": "nonfinite_output",
        },
    }
    return BEMResultArchiveRecord(
        result_id,
        plan,
        bem_archive_descriptor(result.envelope),
        metadata,
        arrays,
    )


__all__ = [
    "BEMArchiveDescriptor",
    "BEMArchiveLimits",
    "BEMArrayArchiveRecord",
    "BEMPlanArchiveRecord",
    "BEMResultArchiveRecord",
    "bem_archive_descriptor",
    "fused_bem_result_archive_record",
    "laplace_dp0_plan_archive_record",
    "read_bem_array_archive",
    "write_bem_array_archive",
]
