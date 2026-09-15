#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
import tempfile
import time
import zipfile
from collections.abc import Sequence
from pathlib import Path

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, ArrayLike

from ..._fingerprint import canonical_fingerprint
from ..._strict import StrictModule
from ..._trainable import NonTrainableState
from ...artifacts import DifferentiationContract, ScientificArtifactEnvelope
from ...backends import (
    AbstractExternalBackend,
    BackendAvailability,
    BackendCapabilities,
)
from ...interchange._report import (
    AdapterCapability,
    AdapterReport,
    AdapterRequirement,
    AdapterStatus,
)
from ...qualification._reference import ReferenceArtifactManifest
from ...units import derived_unit, UnitDefinition
from ._linear_theory import CosmologyModelRequest
from ._products import (
    cosmology_product_content_id,
    CosmologyProductProvenance,
    MatterPowerDescriptor,
    MatterPowerTable,
)


_SCALE_FACTOR_UNIT = "dimensionless"
_NPZ_ARRAYS = frozenset(("metadata_json", "scale_factors", "wavenumbers", "power_values"))
_METADATA_FIELDS = frozenset(
    (
        "request_id",
        "scale",
        "descriptor",
        "scale_factor_unit",
        "wavenumber_unit",
        "power_unit",
        "support",
        "evaluation",
        "neutrino_semantics",
        "reference_manifest",
        "producer",
    )
)
_SUPPORT_FIELDS = frozenset(
    (
        "scale_factor_min",
        "scale_factor_max",
        "wavenumber_min",
        "wavenumber_max",
        "complete",
    )
)
_EVALUATION_FIELDS = frozenset(
    (
        "scale_factor_coordinates",
        "wavenumber_coordinates",
        "clamping_applied",
        "extrapolation_applied",
    )
)
_NEUTRINO_FIELDS = frozenset(("representation", "effective_neutrino_number", "species"))
_PRODUCER_FIELDS = frozenset(("name", "version", "build_id", "license_id"))
_DESCRIPTOR_FIELDS = frozenset(
    (
        "left_field",
        "right_field",
        "gauge",
        "normalization",
        "stage",
        "shot_noise",
        "spatial_dimension",
        "descriptor_id",
    )
)


def _power_unit(scale, descriptor: MatterPowerDescriptor, /) -> UnitDefinition:
    return derived_unit(
        f"{scale.length_unit.symbol}^{descriptor.spatial_dimension}",
        ((scale.length_unit, descriptor.spatial_dimension),),
    )


def _descriptor_record(descriptor: MatterPowerDescriptor, /) -> dict[str, object]:
    return {
        "left_field": descriptor.left_field,
        "right_field": descriptor.right_field,
        "gauge": descriptor.gauge,
        "normalization": descriptor.normalization,
        "stage": descriptor.stage,
        "shot_noise": descriptor.shot_noise,
        "spatial_dimension": descriptor.spatial_dimension,
        "descriptor_id": descriptor.descriptor_id,
    }


def _validated_nodes(values: ArrayLike, name: str, /) -> Array:
    host = np.asarray(values, dtype=np.float64)
    if (
        host.ndim != 1
        or host.size == 0
        or np.any(~np.isfinite(host))
        or np.any(host <= 0.0)
        or (host.size > 1 and np.any(np.diff(host) <= 0.0))
    ):
        raise ValueError(f"{name} must be finite, positive, and strictly increasing.")
    return jnp.asarray(host)


def _bounds(values: Array, /) -> tuple[float, float]:
    host = np.asarray(values)
    return float(host[0]), float(host[-1])


def _strict_bool(value: object, name: str, /) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean.")
    return value


def _positive_integer(value: object, name: str, /) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _mapping(value: object, name: str, /) -> dict[str, object]:
    if not isinstance(value, dict):
        raise MatterPowerProviderError(
            "malformed-output",
            f"Matter-power provider {name} must be a JSON object.",
            adapter_status=AdapterStatus.MALFORMED_SOURCE,
        )
    return value


def _exact_fields(
    value: dict[str, object], expected: frozenset[str], name: str, /
) -> None:
    if set(value) != expected:
        raise MatterPowerProviderError(
            "malformed-output",
            f"Matter-power provider {name} must contain exactly {sorted(expected)}.",
            adapter_status=AdapterStatus.MALFORMED_SOURCE,
        )


def _json_object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise MatterPowerProviderError(
                "malformed-output",
                f"Matter-power provider JSON contains duplicate key {key!r}.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        value[key] = item
    return value


def _reject_json_constant(value: str) -> None:
    raise MatterPowerProviderError(
        "malformed-output",
        f"Matter-power provider JSON contains non-finite constant {value!r}.",
        adapter_status=AdapterStatus.MALFORMED_SOURCE,
    )


def _decode_json_object(text: str, name: str, /) -> dict[str, object]:
    decoded = json.loads(
        text,
        object_pairs_hook=_json_object_pairs,
        parse_constant=_reject_json_constant,
    )
    return _mapping(decoded, name)


def _stream_digest(path: Path, algorithm: str, /) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


class MatterPowerEvaluationRequest(StrictModule, NonTrainableState):
    """Host-only physical request on one exact scale-factor/wavenumber grid."""

    cosmology: CosmologyModelRequest
    scale_factors: Array
    wavenumbers: Array
    descriptor: MatterPowerDescriptor
    scale_factor_unit: str = eqx.field(static=True)
    request_id: str = eqx.field(static=True)

    def __init__(
        self,
        cosmology: CosmologyModelRequest,
        scale_factors: ArrayLike,
        wavenumbers: ArrayLike,
        descriptor: MatterPowerDescriptor,
        /,
    ):
        if not isinstance(cosmology, CosmologyModelRequest):
            raise TypeError("cosmology must be CosmologyModelRequest.")
        if not isinstance(descriptor, MatterPowerDescriptor):
            raise TypeError("descriptor must be MatterPowerDescriptor.")
        if descriptor.gauge != cosmology.gauge:
            raise ValueError(
                "Matter-power descriptor gauge must match the cosmology request gauge."
            )
        scales = _validated_nodes(scale_factors, "scale_factors")
        wavenumber = _validated_nodes(wavenumbers, "wavenumbers")
        self.cosmology = cosmology
        self.scale_factors = scales
        self.wavenumbers = wavenumber
        self.descriptor = descriptor
        self.scale_factor_unit = _SCALE_FACTOR_UNIT
        self.request_id = canonical_fingerprint(
            {
                "kind": "matter-power-evaluation-request",
                "cosmology_request": cosmology.request_id,
                "scale_factors": np.asarray(scales).tolist(),
                "wavenumbers": np.asarray(wavenumber).tolist(),
                "descriptor": descriptor.descriptor_id,
                "scale_factor_unit": _SCALE_FACTOR_UNIT,
                "wavenumber_unit": cosmology.scale.wavenumber_unit.unit_id,
                "power_unit": _power_unit(cosmology.scale, descriptor).unit_id,
            }
        )

    def to_mapping(self, *, include_identity: bool = True) -> dict[str, object]:
        mapping: dict[str, object] = {
            "cosmology": self.cosmology.to_mapping(),
            "scale_factors": np.asarray(self.scale_factors).tolist(),
            "wavenumbers": np.asarray(self.wavenumbers).tolist(),
            "descriptor": _descriptor_record(self.descriptor),
            "scale_factor_unit": self.scale_factor_unit,
            "wavenumber_unit": self.cosmology.scale.wavenumber_unit.to_dict(),
            "power_unit": _power_unit(self.cosmology.scale, self.descriptor).to_dict(),
        }
        if include_identity:
            mapping["request_id"] = self.request_id
        return mapping


class EmulatorSupportEvidence(StrictModule, NonTrainableState):
    """Separate rectangular-domain and provider-completeness evidence."""

    requested_scale_factor_bounds: tuple[float, float] = eqx.field(static=True)
    requested_wavenumber_bounds: tuple[float, float] = eqx.field(static=True)
    provider_scale_factor_bounds: tuple[float, float] = eqx.field(static=True)
    provider_wavenumber_bounds: tuple[float, float] = eqx.field(static=True)
    scale_factor_range_covered: bool = eqx.field(static=True)
    wavenumber_range_covered: bool = eqx.field(static=True)
    rectangular_range_covered: bool = eqx.field(static=True)
    provider_support_complete: bool = eqx.field(static=True)
    complete: bool = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        requested_scale_factor_bounds: Sequence[float],
        requested_wavenumber_bounds: Sequence[float],
        provider_scale_factor_bounds: Sequence[float],
        provider_wavenumber_bounds: Sequence[float],
        /,
        *,
        provider_support_complete: bool,
    ):
        bounds = tuple(
            self._validated_bounds(value, name)
            for value, name in (
                (requested_scale_factor_bounds, "requested scale-factor bounds"),
                (requested_wavenumber_bounds, "requested wavenumber bounds"),
                (provider_scale_factor_bounds, "provider scale-factor bounds"),
                (provider_wavenumber_bounds, "provider wavenumber bounds"),
            )
        )
        complete = _strict_bool(provider_support_complete, "provider_support_complete")
        scale_covered = bounds[2][0] <= bounds[0][0] and bounds[0][1] <= bounds[2][1]
        wavenumber_covered = bounds[3][0] <= bounds[1][0] and bounds[1][1] <= bounds[3][1]
        rectangular = scale_covered and wavenumber_covered
        self.requested_scale_factor_bounds = bounds[0]
        self.requested_wavenumber_bounds = bounds[1]
        self.provider_scale_factor_bounds = bounds[2]
        self.provider_wavenumber_bounds = bounds[3]
        self.scale_factor_range_covered = scale_covered
        self.wavenumber_range_covered = wavenumber_covered
        self.rectangular_range_covered = rectangular
        self.provider_support_complete = complete
        self.complete = rectangular and complete
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "matter-power-emulator-support",
                "requested_scale_factor_bounds": list(bounds[0]),
                "requested_wavenumber_bounds": list(bounds[1]),
                "provider_scale_factor_bounds": list(bounds[2]),
                "provider_wavenumber_bounds": list(bounds[3]),
                "provider_support_complete": complete,
            }
        )

    @staticmethod
    def _validated_bounds(values: Sequence[float], name: str, /) -> tuple[float, float]:
        if isinstance(values, (str, bytes)):
            raise TypeError(f"{name} must contain exactly two numeric values.")
        normalized = tuple(float(value) for value in values)
        if (
            len(normalized) != 2
            or any(not np.isfinite(value) or value <= 0.0 for value in normalized)
            or normalized[0] > normalized[1]
        ):
            raise ValueError(f"{name} must be finite, positive, and ordered.")
        return normalized


class MatterPowerProcessEvidence(StrictModule, NonTrainableState):
    """Bounded process outputs and byte/time accounting for one successful call."""

    return_code: int = eqx.field(static=True)
    standard_output: str = eqx.field(static=True)
    standard_error: str = eqx.field(static=True)
    elapsed_seconds: float = eqx.field(static=True)
    request_bytes: int = eqx.field(static=True)
    result_bytes: int = eqx.field(static=True)
    result_uncompressed_bytes: int = eqx.field(static=True)
    standard_output_bytes: int = eqx.field(static=True)
    standard_error_bytes: int = eqx.field(static=True)
    evidence_id: str = eqx.field(static=True)

    def __init__(
        self,
        *,
        return_code: int,
        standard_output: str,
        standard_error: str,
        elapsed_seconds: float,
        request_bytes: int,
        result_bytes: int,
        result_uncompressed_bytes: int,
        standard_output_bytes: int,
        standard_error_bytes: int,
    ):
        code = int(return_code)
        elapsed = float(elapsed_seconds)
        counts = tuple(
            int(value)
            for value in (
                request_bytes,
                result_bytes,
                result_uncompressed_bytes,
                standard_output_bytes,
                standard_error_bytes,
            )
        )
        if (
            code != 0
            or not np.isfinite(elapsed)
            or elapsed < 0.0
            or any(value < 0 for value in counts)
        ):
            raise ValueError("Successful matter-power process evidence is invalid.")
        stdout = str(standard_output)
        stderr = str(standard_error)
        if (
            len(stdout.encode("utf-8")) != counts[3]
            or len(stderr.encode("utf-8")) != counts[4]
        ):
            raise ValueError("Matter-power process log byte accounting is inconsistent.")
        self.return_code = code
        self.standard_output = stdout
        self.standard_error = stderr
        self.elapsed_seconds = elapsed
        (
            self.request_bytes,
            self.result_bytes,
            self.result_uncompressed_bytes,
            self.standard_output_bytes,
            self.standard_error_bytes,
        ) = counts
        self.evidence_id = canonical_fingerprint(
            {
                "kind": "matter-power-process-evidence",
                "return_code": code,
                "standard_output": stdout,
                "standard_error": stderr,
                "elapsed_seconds": elapsed,
                "byte_counts": list(counts),
            }
        )


class ExternalMatterPowerResult(StrictModule, NonTrainableState):
    """Accepted external table with artifact, adapter, support, and process evidence."""

    table: MatterPowerTable
    artifact: ScientificArtifactEnvelope
    report: AdapterReport
    process: MatterPowerProcessEvidence
    support: EmulatorSupportEvidence
    reference_manifest: ReferenceArtifactManifest

    def __init__(
        self,
        table: MatterPowerTable,
        artifact: ScientificArtifactEnvelope,
        report: AdapterReport,
        process: MatterPowerProcessEvidence,
        support: EmulatorSupportEvidence,
        reference_manifest: ReferenceArtifactManifest,
        /,
    ):
        if not isinstance(table, MatterPowerTable):
            raise TypeError("table must be MatterPowerTable.")
        if not isinstance(artifact, ScientificArtifactEnvelope):
            raise TypeError("artifact must be ScientificArtifactEnvelope.")
        if not isinstance(report, AdapterReport):
            raise TypeError("report must be AdapterReport.")
        if not isinstance(process, MatterPowerProcessEvidence):
            raise TypeError("process must be MatterPowerProcessEvidence.")
        if not isinstance(support, EmulatorSupportEvidence):
            raise TypeError("support must be EmulatorSupportEvidence.")
        if not isinstance(reference_manifest, ReferenceArtifactManifest):
            raise TypeError("reference_manifest must be ReferenceArtifactManifest.")
        product_id = cosmology_product_content_id(table)
        if (
            artifact.status != "complete"
            or reference_manifest.manifest_id not in artifact.parent_artifact_ids
            or product_id not in artifact.parent_artifact_ids
            or not report.valid
            or report.target_id != product_id
            or not support.complete
            or process.return_code != 0
        ):
            raise ValueError("External matter-power result evidence is inconsistent.")
        self.table = table
        self.artifact = artifact
        self.report = report
        self.process = process
        self.support = support
        self.reference_manifest = reference_manifest


class MatterPowerProviderError(RuntimeError):
    """Fail-closed external-provider refusal with optional support evidence."""

    reason: str
    adapter_status: AdapterStatus
    support: EmulatorSupportEvidence | None

    def __init__(
        self,
        reason: str,
        message: str,
        /,
        *,
        adapter_status: AdapterStatus,
        support: EmulatorSupportEvidence | None = None,
    ):
        reason_ = str(reason).strip()
        message_ = str(message).strip()
        if not reason_ or not message_:
            raise ValueError("Matter-power provider errors require a reason and message.")
        self.reason = reason_
        self.adapter_status = AdapterStatus(adapter_status)
        self.support = support
        super().__init__(message_)


class SubprocessMatterPowerBackend(AbstractExternalBackend, NonTrainableState):
    """Bounded exact-grid JSON-request/NPZ-result matter-power provider."""

    application: str = eqx.field(static=True)
    arguments: tuple[str, ...] = eqx.field(static=True)
    reference_artifact_path: str = eqx.field(static=True)
    reference_manifest: ReferenceArtifactManifest
    timeout_seconds: float = eqx.field(static=True)
    maximum_request_bytes: int = eqx.field(static=True)
    maximum_result_bytes: int = eqx.field(static=True)
    maximum_log_bytes: int = eqx.field(static=True)
    backend_name: str = eqx.field(static=True)
    backend_version: str = eqx.field(static=True)
    build_id: str = eqx.field(static=True)
    numerical_policy_id: str = eqx.field(static=True)
    commercial_use: bool = eqx.field(static=True)
    redistribution: bool = eqx.field(static=True)
    training_use: bool = eqx.field(static=True)
    export: bool = eqx.field(static=True)

    def __init__(
        self,
        application: str,
        reference_artifact_path: str | os.PathLike[str],
        reference_manifest: ReferenceArtifactManifest,
        /,
        *,
        arguments: Sequence[str] = ("{request}", "{output}"),
        timeout_seconds: float = 600.0,
        maximum_request_bytes: int = 1_000_000,
        maximum_result_bytes: int = 64_000_000,
        maximum_log_bytes: int = 1_000_000,
        backend_name: str = "matter-power-subprocess",
        backend_version: str = "user-provided",
        build_id: str = "user-provided",
        numerical_policy_id: str = "external-matter-power-exact-grid",
        commercial_use: bool = False,
        redistribution: bool = False,
        training_use: bool = False,
        export: bool = False,
    ):
        executable = str(application).strip()
        artifact_path = str(Path(reference_artifact_path).expanduser().resolve())
        arguments_ = tuple(str(argument) for argument in arguments)
        timeout = float(timeout_seconds)
        name = str(backend_name).strip()
        version = str(backend_version).strip()
        build = str(build_id).strip()
        policy = str(numerical_policy_id).strip()
        rights = tuple(
            _strict_bool(value, field)
            for value, field in (
                (commercial_use, "commercial_use"),
                (redistribution, "redistribution"),
                (training_use, "training_use"),
                (export, "export"),
            )
        )
        if not isinstance(reference_manifest, ReferenceArtifactManifest):
            raise TypeError("reference_manifest must be ReferenceArtifactManifest.")
        if (
            not executable
            or not arguments_
            or not any("{request}" in argument for argument in arguments_)
            or not any("{output}" in argument for argument in arguments_)
            or not np.isfinite(timeout)
            or timeout <= 0.0
            or not name
            or not version
            or not build
            or not policy
        ):
            raise ValueError("Matter-power subprocess backend configuration is invalid.")
        resolved_executable = shutil.which(executable)
        executable_is_reference = (
            resolved_executable is not None
            and str(Path(resolved_executable).resolve()) == artifact_path
        )
        if not executable_is_reference and not any(
            "{reference_artifact}" in argument for argument in arguments_
        ):
            raise ValueError(
                "The governed reference artifact must be the executable or be bound "
                "through a {reference_artifact} argument."
            )
        self.application = executable
        self.arguments = arguments_
        self.reference_artifact_path = artifact_path
        self.reference_manifest = reference_manifest
        self.timeout_seconds = timeout
        self.maximum_request_bytes = _positive_integer(
            maximum_request_bytes, "maximum_request_bytes"
        )
        self.maximum_result_bytes = _positive_integer(
            maximum_result_bytes, "maximum_result_bytes"
        )
        self.maximum_log_bytes = _positive_integer(maximum_log_bytes, "maximum_log_bytes")
        self.backend_name = name
        self.backend_version = version
        self.build_id = build
        self.numerical_policy_id = policy
        (
            self.commercial_use,
            self.redistribution,
            self.training_use,
            self.export,
        ) = rights
        reference_manifest.require_rights(
            commercial_use=rights[0],
            redistribution=rights[1],
            training_use=rights[2],
            export=rights[3],
        )

    @property
    def name(self) -> str:
        return self.backend_name

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            backend=self.backend_name,
            problem_kinds=("cosmology-matter-power",),
            execution="host",
            host_only=True,
            supports_matrix_free=False,
            supports_assembled=False,
            coordinate_dtypes=("float32", "float64"),
            supports_plan_prepare_solve_refresh=False,
        )

    def availability(self, /) -> BackendAvailability:
        executable = shutil.which(self.application)
        artifact = Path(self.reference_artifact_path)
        available = executable is not None and artifact.is_file()
        if executable is None:
            reason = "executable not found on PATH"
        elif not artifact.is_file():
            reason = "governed reference artifact not found"
        else:
            reason = "executable and governed reference artifact resolved"
        return BackendAvailability(
            capabilities=self.capabilities,
            available=available,
            requirement=f"{self.application}; {self.reference_artifact_path}",
            reason=reason,
            versions=((self.backend_name, self.backend_version),),
        )

    def _verify_reference(self) -> None:
        path = Path(self.reference_artifact_path)
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise MatterPowerProviderError(
                "reference-artifact-mismatch",
                "Matter-power reference artifact must be a regular file.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        if (
            metadata.st_size != self.reference_manifest.size_bytes
            or _stream_digest(path, self.reference_manifest.checksum_algorithm)
            != self.reference_manifest.checksum
        ):
            raise MatterPowerProviderError(
                "reference-artifact-mismatch",
                "Matter-power reference artifact does not match its checksum manifest.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        self.reference_manifest.require_rights(
            commercial_use=self.commercial_use,
            redistribution=self.redistribution,
            training_use=self.training_use,
            export=self.export,
        )

    def _request_payload(self, request: MatterPowerEvaluationRequest, /) -> bytes:
        payload = {
            **request.to_mapping(),
            "provider_contract": {
                "reference_manifest": self.reference_manifest.to_record(),
                "requested_use": {
                    "commercial_use": self.commercial_use,
                    "redistribution": self.redistribution,
                    "training_use": self.training_use,
                    "export": self.export,
                },
                "producer": {
                    "name": self.backend_name,
                    "version": self.backend_version,
                    "build_id": self.build_id,
                    "license_id": self.reference_manifest.license_id,
                },
            },
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        if len(encoded) > self.maximum_request_bytes:
            raise MatterPowerProviderError(
                "request-size-limit",
                "Matter-power JSON request exceeds its byte limit.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        return encoded

    def _validate_archive(self, path: Path, /) -> tuple[int, int]:
        metadata = os.lstat(path)
        if not stat.S_ISREG(metadata.st_mode):
            raise MatterPowerProviderError(
                "malformed-output",
                "Matter-power provider result must be a regular NPZ file.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        result_bytes = metadata.st_size
        if result_bytes <= 0 or result_bytes > self.maximum_result_bytes:
            raise MatterPowerProviderError(
                "result-size-limit",
                "Matter-power provider NPZ exceeds its byte limit.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        if not zipfile.is_zipfile(path):
            raise MatterPowerProviderError(
                "malformed-output",
                "Matter-power provider result is not an NPZ archive.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            names = tuple(entry.filename for entry in entries)
            expected = frozenset(f"{name}.npy" for name in _NPZ_ARRAYS)
            if len(names) != len(set(names)) or frozenset(names) != expected:
                raise MatterPowerProviderError(
                    "malformed-output",
                    "Matter-power provider NPZ has missing, duplicate, or unknown arrays.",
                    adapter_status=AdapterStatus.MALFORMED_SOURCE,
                )
            if any(entry.is_dir() or entry.flag_bits & 0x1 for entry in entries):
                raise MatterPowerProviderError(
                    "malformed-output",
                    "Matter-power provider NPZ contains invalid archive members.",
                    adapter_status=AdapterStatus.MALFORMED_SOURCE,
                )
            uncompressed = sum(entry.file_size for entry in entries)
        if uncompressed > self.maximum_result_bytes:
            raise MatterPowerProviderError(
                "result-size-limit",
                "Matter-power provider uncompressed NPZ exceeds its byte limit.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        return result_bytes, uncompressed

    def _load_result(
        self, path: Path, request: MatterPowerEvaluationRequest, /
    ) -> tuple[np.ndarray, EmulatorSupportEvidence, str, int, int]:
        result_bytes, uncompressed_bytes = self._validate_archive(path)
        with np.load(path, allow_pickle=False, max_header_size=64_000) as arrays:
            metadata_array = np.asarray(arrays["metadata_json"])
            if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                raise MatterPowerProviderError(
                    "malformed-output",
                    "Matter-power provider metadata_json must be a scalar JSON string.",
                    adapter_status=AdapterStatus.MALFORMED_SOURCE,
                )
            metadata_text = str(metadata_array.item())
            scales = np.asarray(arrays["scale_factors"]).copy()
            wavenumbers = np.asarray(arrays["wavenumbers"]).copy()
            power = np.asarray(arrays["power_values"]).copy()
        metadata = _decode_json_object(metadata_text, "metadata")
        _exact_fields(metadata, _METADATA_FIELDS, "metadata")
        if metadata["request_id"] != request.request_id:
            raise MatterPowerProviderError(
                "request-mismatch",
                "Matter-power provider result request identity does not match.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        if metadata["scale"] != request.cosmology.scale.to_dict():
            raise MatterPowerProviderError(
                "scale-mismatch",
                "Matter-power provider result scale does not match its request.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        descriptor = _mapping(metadata["descriptor"], "descriptor")
        _exact_fields(descriptor, _DESCRIPTOR_FIELDS, "descriptor")
        if descriptor != _descriptor_record(request.descriptor):
            stage_or_field_changed = (
                descriptor.get("left_field") != request.descriptor.left_field
                or descriptor.get("right_field") != request.descriptor.right_field
                or descriptor.get("stage") != request.descriptor.stage
            )
            reason = (
                "stage-field-mismatch"
                if stage_or_field_changed
                else "descriptor-mismatch"
            )
            raise MatterPowerProviderError(
                reason,
                "Matter-power provider descriptor does not match its request.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            )
        if (
            metadata["scale_factor_unit"] != request.scale_factor_unit
            or metadata["wavenumber_unit"]
            != request.cosmology.scale.wavenumber_unit.to_dict()
            or metadata["power_unit"]
            != _power_unit(request.cosmology.scale, request.descriptor).to_dict()
        ):
            raise MatterPowerProviderError(
                "unit-mismatch",
                "Matter-power provider coordinate or power units do not match.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        producer = _mapping(metadata["producer"], "producer metadata")
        _exact_fields(producer, _PRODUCER_FIELDS, "producer metadata")
        expected_producer = {
            "name": self.backend_name,
            "version": self.backend_version,
            "build_id": self.build_id,
            "license_id": self.reference_manifest.license_id,
        }
        if producer != expected_producer:
            raise MatterPowerProviderError(
                "producer-mismatch",
                "Matter-power provider identity does not match its backend contract.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        manifest_record = _mapping(metadata["reference_manifest"], "reference manifest")
        if set(manifest_record) != set(self.reference_manifest.to_record()):
            raise MatterPowerProviderError(
                "missing-rights-manifest",
                "Matter-power provider result lacks its exact rights/checksum manifest.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            )
        returned_manifest = ReferenceArtifactManifest.from_record(manifest_record)
        if returned_manifest.manifest_id != self.reference_manifest.manifest_id:
            raise MatterPowerProviderError(
                "reference-manifest-mismatch",
                "Matter-power provider result changed its rights/checksum manifest.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        neutrinos = _mapping(metadata["neutrino_semantics"], "neutrino semantics")
        _exact_fields(neutrinos, _NEUTRINO_FIELDS, "neutrino semantics")
        expected_neutrinos = {
            "representation": "explicit-massive-neutrino-species",
            "effective_neutrino_number": request.cosmology.effective_neutrino_number,
            "species": [species.to_mapping() for species in request.cosmology.neutrinos],
        }
        if neutrinos != expected_neutrinos:
            raise MatterPowerProviderError(
                "unsupported-neutrino-semantics",
                "Matter-power provider does not preserve explicit neutrino semantics.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            )
        evaluation = _mapping(metadata["evaluation"], "evaluation evidence")
        _exact_fields(evaluation, _EVALUATION_FIELDS, "evaluation evidence")
        if (
            evaluation["scale_factor_coordinates"] != "exact-request-grid"
            or evaluation["wavenumber_coordinates"] != "exact-request-grid"
            or evaluation["clamping_applied"] is not False
            or evaluation["extrapolation_applied"] is not False
        ):
            raise MatterPowerProviderError(
                "hidden-domain-transformation",
                "Matter-power provider clamped, extrapolated, or remapped coordinates.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
            )
        support_record = _mapping(metadata["support"], "support evidence")
        _exact_fields(support_record, _SUPPORT_FIELDS, "support evidence")
        support = EmulatorSupportEvidence(
            _bounds(request.scale_factors),
            _bounds(request.wavenumbers),
            (
                support_record["scale_factor_min"],
                support_record["scale_factor_max"],
            ),
            (
                support_record["wavenumber_min"],
                support_record["wavenumber_max"],
            ),
            provider_support_complete=_strict_bool(
                support_record["complete"], "provider support complete"
            ),
        )
        if not support.rectangular_range_covered:
            raise MatterPowerProviderError(
                "outside-rectangular-support",
                "Matter-power request lies outside the provider rectangular range.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                support=support,
            )
        if not support.provider_support_complete:
            raise MatterPowerProviderError(
                "incomplete-provider-support",
                "Matter-power provider reports incomplete support inside its rectangle.",
                adapter_status=AdapterStatus.UNSUPPORTED_REQUIRED_SEMANTIC,
                support=support,
            )
        expected_scales = np.asarray(request.scale_factors)
        expected_wavenumbers = np.asarray(request.wavenumbers)
        expected_shape = (expected_scales.size, expected_wavenumbers.size)
        if (
            scales.dtype.kind != "f"
            or scales.shape != expected_scales.shape
            or not np.array_equal(scales, expected_scales)
        ):
            raise MatterPowerProviderError(
                "scale-factor-grid-mismatch",
                "Matter-power provider did not return the exact scale-factor grid.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        if (
            wavenumbers.dtype.kind != "f"
            or wavenumbers.shape != expected_wavenumbers.shape
            or not np.array_equal(wavenumbers, expected_wavenumbers)
        ):
            raise MatterPowerProviderError(
                "wavenumber-grid-mismatch",
                "Matter-power provider did not return the exact wavenumber grid.",
                adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
            )
        if (
            power.dtype.kind != "f"
            or power.shape != expected_shape
            or np.any(~np.isfinite(power))
            or (request.descriptor.is_auto and np.any(power < 0.0))
        ):
            raise MatterPowerProviderError(
                "malformed-power-values",
                "Matter-power provider returned invalid power values.",
                adapter_status=AdapterStatus.MALFORMED_SOURCE,
            )
        return (
            power,
            support,
            _stream_digest(path, "sha256"),
            result_bytes,
            uncompressed_bytes,
        )

    def _report(
        self,
        request: MatterPowerEvaluationRequest,
        table: MatterPowerTable,
        /,
    ) -> AdapterReport:
        semantics = (
            "matter-power:exact-coordinate-grids",
            f"matter-power:descriptor:{request.descriptor.descriptor_id}",
            f"matter-power:scale:{request.cosmology.scale.scale_id}",
            "matter-power:explicit-neutrino-species",
            "matter-power:complete-provider-support",
            f"artifact:rights-checksum-manifest:{self.reference_manifest.manifest_id}",
        )
        requirements = tuple(
            AdapterRequirement(
                semantic,
                rationale="Required for lossless external matter-power admission.",
            )
            for semantic in semantics
        )
        capabilities = tuple(
            AdapterCapability(semantic, detail="Verified by the provider protocol.")
            for semantic in semantics
        )
        return AdapterReport(
            AdapterStatus.LOSSLESS,
            "matter-power-provider-npz",
            "phydrax-matter-power-table",
            source_id=self.reference_manifest.manifest_id,
            target_id=cosmology_product_content_id(table),
            coordinate_mapping=(
                "scale_factors->scale_factors",
                "wavenumbers->wavenumbers",
            ),
            preserved_fields=(
                f"left_field:{request.descriptor.left_field}",
                f"right_field:{request.descriptor.right_field}",
                f"gauge:{request.descriptor.gauge}",
                f"stage:{request.descriptor.stage}",
                f"normalization:{request.descriptor.normalization}",
                f"shot_noise:{request.descriptor.shot_noise}",
                f"spatial_dimension:{request.descriptor.spatial_dimension}",
            ),
            stage="external-matter-power-admission",
            requirements=requirements,
            capabilities=capabilities,
        )

    def run(self, request: MatterPowerEvaluationRequest, /) -> ExternalMatterPowerResult:
        if not isinstance(request, MatterPowerEvaluationRequest):
            raise TypeError("request must be MatterPowerEvaluationRequest.")
        availability = self.availability()
        if not availability.available:
            raise RuntimeError(
                f"Matter-power backend {self.backend_name!r} is unavailable: "
                f"{availability.reason}."
            )
        self._verify_reference()
        request_payload = self._request_payload(request)
        started = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="phydrax-matter-power-") as directory:
            root = Path(directory)
            request_path = root / "request.json"
            result_path = root / "result.npz"
            stdout_path = root / "stdout.txt"
            stderr_path = root / "stderr.txt"
            request_path.write_bytes(request_payload)
            command = [
                self.application,
                *(
                    argument.replace("{request}", str(request_path))
                    .replace("{output}", str(result_path))
                    .replace("{reference_artifact}", self.reference_artifact_path)
                    for argument in self.arguments
                ),
            ]
            with (
                stdout_path.open("wb") as stdout_handle,
                stderr_path.open("wb") as stderr_handle,
            ):
                completed = subprocess.run(
                    command,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    timeout=self.timeout_seconds,
                    check=False,
                )
            stdout_bytes = stdout_path.stat().st_size
            stderr_bytes = stderr_path.stat().st_size
            if stdout_bytes + stderr_bytes > self.maximum_log_bytes:
                raise MatterPowerProviderError(
                    "process-log-size-limit",
                    "Matter-power provider process logs exceed their byte limit.",
                    adapter_status=AdapterStatus.MALFORMED_SOURCE,
                )
            stdout = stdout_path.read_bytes().decode("utf-8")
            stderr = stderr_path.read_bytes().decode("utf-8")
            if completed.returncode != 0:
                raise MatterPowerProviderError(
                    "process-failure",
                    f"Matter-power provider failed with code {completed.returncode}: "
                    f"{stderr.strip()}",
                    adapter_status=AdapterStatus.INCONSISTENT_SOURCE,
                )
            if not result_path.exists():
                raise MatterPowerProviderError(
                    "missing-output",
                    "Matter-power provider did not create its NPZ result.",
                    adapter_status=AdapterStatus.MALFORMED_SOURCE,
                )
            (
                power,
                support,
                content_digest,
                result_bytes,
                uncompressed_bytes,
            ) = self._load_result(result_path, request)
        provenance = CosmologyProductProvenance(
            producer=self.backend_name,
            producer_version=self.backend_version,
            model_form_id=request.cosmology.model_form_id,
            request_id=request.request_id,
            numerical_policy_id=self.numerical_policy_id,
            physics_policy_id=canonical_fingerprint(
                {
                    "kind": "external-matter-power-policy",
                    "descriptor": request.descriptor.descriptor_id,
                    "reference_manifest": self.reference_manifest.manifest_id,
                }
            ),
            scale_id=request.cosmology.scale.scale_id,
            source_kind="external",
            differentiation=DifferentiationContract.constant(),
            parent_product_ids=(self.reference_manifest.manifest_id,),
        )
        table = MatterPowerTable(
            request.scale_factors,
            request.wavenumbers,
            jnp.asarray(power),
            request.descriptor,
            request.cosmology.scale,
            provenance,
            request.cosmology.realization,
        )
        product_id = cosmology_product_content_id(table)
        artifact = ScientificArtifactEnvelope(
            artifact_kind="external-matter-power-table",
            content_digest=content_digest,
            producer=self.backend_name,
            producer_version=self.backend_version,
            build_id=self.build_id,
            license_id=self.reference_manifest.license_id,
            resource_id=self.numerical_policy_id,
            status="complete",
            parent_artifact_ids=(self.reference_manifest.manifest_id, product_id),
        )
        report = self._report(request, table)
        process = MatterPowerProcessEvidence(
            return_code=completed.returncode,
            standard_output=stdout,
            standard_error=stderr,
            elapsed_seconds=time.perf_counter() - started,
            request_bytes=len(request_payload),
            result_bytes=result_bytes,
            result_uncompressed_bytes=uncompressed_bytes,
            standard_output_bytes=stdout_bytes,
            standard_error_bytes=stderr_bytes,
        )
        return ExternalMatterPowerResult(
            table,
            artifact,
            report,
            process,
            support,
            self.reference_manifest,
        )


__all__ = [
    "EmulatorSupportEvidence",
    "ExternalMatterPowerResult",
    "MatterPowerEvaluationRequest",
    "MatterPowerProcessEvidence",
    "MatterPowerProviderError",
    "SubprocessMatterPowerBackend",
]
