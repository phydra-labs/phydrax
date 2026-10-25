#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Conversion of admitted fairchem UMA inference checkpoints into native models.

Source bytes are admitted through the canonical external-artifact policy
(exact SHA-256, size and license). Extraction runs only in a caller-pinned
interpreter through ``run_pinned_command`` (``-I -S``, empty environment),
executing the standard-library program ``_uma_worker.py``: it never imports
torch or fairchem and never unpickles; it interprets the checkpoint pickle
symbolically against an exact allowlist and returns a bounded pickle-free
array archive. The subprocess bounds time, logs and output files; it is not a
memory sandbox, so every container, record, storage and view is bounded by
`UMAConversionLimits` before it is read.

The host then admits the plain configuration exhaustively
(`parse_uma_source_config`) and reconstructs the native potential from the
exact EMA parameters (the source inference route, ``use_ema=True``). The
``module.`` EMA prefix and ``n_averaged`` count are consumed as EMA evidence;
the non-EMA training weights are not inference parameters, so their complete
name/dtype/shape inventory is checked against the EMA inventory and recorded,
never silently dropped. Gated weights and their terms remain the caller's
admission decision (`ExternalArtifactPolicy`); nothing is downloaded.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import MappingProxyType
from typing import Any, TYPE_CHECKING

import numpy as np

from ..._array_archive import ArrayArchiveLimits, read_array_archive
from ..._artifact_security import (
    AdmittedExternalArtifact,
    ExternalArtifactPolicy,
    read_admitted_artifact,
)
from ..._external_runtime import (
    ExternalExecutionPolicy,
    ExternalRuntimeError,
    PinnedExecutable,
    PinnedFileOutputs,
    PinnedFileRequest,
    run_pinned_command,
)
from ..._fingerprint import canonical_fingerprint
from ...artifacts import ArtifactManifest
from ...nn.atomistic._uma_so3 import UMAEdgeRollGauge
from ...typing import parse
from .._types import AtomisticPrecisionPolicy, AtomisticScaleContract


if TYPE_CHECKING:
    from ...nn.atomistic._uma import UMAModel
    from ...nn.atomistic._uma_source import UMASourceDeclaration
    from ...sparse import StreamedRelationPlan


_WORKER_NAME = "_uma_worker.py"
_REFUSED_EXIT = 3
_RESULT_FORMAT = "phydrax-uma-source-result"
_EMA_PREFIX = "module."
# Must equal the worker's Python-recursion ceiling for configuration depth.
_MAX_CONFIG_DEPTH = 512
_BOOTSTRAP = (
    "import runpy, sys; sys.argv = [sys.argv[1], *sys.argv[2:]]; "
    "runpy.run_path(sys.argv[0], run_name='__main__')"
)
_CHECKPOINT_FIELDS = frozenset(
    {"model_config", "tasks_config", "ema_n_averaged", "model_state_inventory"}
)


class UMASourceRefusedError(ValueError):
    """The isolated extraction refused a source outside the admitted format."""


@dataclass(frozen=True, slots=True)
class UMAExtractionRuntime:
    """Caller-pinned interpreter that runs the standard-library extraction program."""

    interpreter: PinnedExecutable

    def __post_init__(self) -> None:
        if not isinstance(self.interpreter, PinnedExecutable):
            raise TypeError("interpreter must be a PinnedExecutable.")


@dataclass(frozen=True, slots=True)
class UMASource:
    """One admitted fairchem ``MLIPInferenceCheckpoint`` (``torch.save`` container)."""

    artifact: AdmittedExternalArtifact
    manifest: ArtifactManifest
    policy: ExternalArtifactPolicy

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, AdmittedExternalArtifact):
            raise TypeError("artifact must be an AdmittedExternalArtifact.")
        if not isinstance(self.manifest, ArtifactManifest):
            raise TypeError("manifest must be an ArtifactManifest.")
        if not isinstance(self.policy, ExternalArtifactPolicy):
            raise TypeError("policy must be an ExternalArtifactPolicy.")


@dataclass(frozen=True, slots=True)
class UMAConversionLimits:
    """Explicit byte, member, element, metadata and time bounds of one extraction run.

    ``max_manifest_bytes`` also bounds the source ZIP central directory and the
    expanded JSON size of the source configuration; ``max_config_nodes`` and
    ``max_config_depth`` bound that configuration's expanded node count (every
    shared pickle reference is charged per occurrence) and nesting.
    """

    max_source_bytes: int = 4_294_967_296
    max_result_bytes: int = 4_294_967_296
    max_tensor_bytes: int = 1_073_741_824
    max_tensor_elements: int = 268_435_456
    max_members: int = 16_385
    max_manifest_bytes: int = 67_108_864
    max_config_nodes: int = 262_144
    max_config_depth: int = 64
    max_log_bytes: int = 16_777_216
    timeout_seconds: float = 1_800.0

    def __post_init__(self) -> None:
        integers = (
            self.max_source_bytes,
            self.max_result_bytes,
            self.max_tensor_bytes,
            self.max_tensor_elements,
            self.max_members,
            self.max_manifest_bytes,
            self.max_config_nodes,
            self.max_config_depth,
            self.max_log_bytes,
        )
        if any(type(value) is not int or value <= 0 for value in integers):
            raise ValueError("UMA conversion limits must be positive integers.")
        if self.max_config_depth > _MAX_CONFIG_DEPTH:
            raise ValueError(f"max_config_depth must not exceed {_MAX_CONFIG_DEPTH}.")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0:
            raise ValueError("timeout_seconds must be positive and finite.")

    def archive_limits(self) -> ArrayArchiveLimits:
        return ArrayArchiveLimits(
            max_container_bytes=self.max_result_bytes,
            max_aggregate_bytes=self.max_result_bytes,
            max_member_bytes=self.max_tensor_bytes,
            max_manifest_bytes=self.max_manifest_bytes,
            max_members=self.max_members,
            max_central_directory_bytes=self.max_manifest_bytes,
            max_axis_length=self.max_tensor_elements,
            max_array_elements=self.max_tensor_elements,
            max_total_array_elements=max(self.max_tensor_elements, self.max_result_bytes),
        )

    def worker_limits(self) -> dict[str, int]:
        archive = self.archive_limits()
        return {
            "max_source_bytes": self.max_source_bytes,
            "max_result_bytes": archive.max_container_bytes,
            "max_tensor_bytes": archive.max_member_bytes,
            "max_tensor_elements": archive.max_array_elements,
            "max_members": archive.max_members,
            "max_manifest_bytes": archive.max_manifest_bytes,
            "max_array_rank": archive.max_array_rank,
            "max_total_elements": archive.max_total_array_elements,
            "max_config_depth": self.max_config_depth,
            "max_config_nodes": self.max_config_nodes,
        }


@dataclass(frozen=True, slots=True)
class UMASourceExtraction:
    """Admitted plain source configuration, EMA tensors and evidence."""

    model_config: Mapping[str, Any]
    tasks_config: tuple[Mapping[str, Any], ...]
    configuration_arrays: Mapping[str, np.ndarray]
    tensors: Mapping[str, np.ndarray]
    ema_n_averaged: int
    training_inventory: tuple[tuple[str, str, tuple[int, ...]], ...]


@dataclass(frozen=True, slots=True)
class UMASourceProvenance:
    """Source, extraction, admission and native-realization identity of one UMA conversion.

    ``declaration`` is the admitted source configuration record; the selected
    edge-roll gauge, the native architecture fingerprint (model kind, gauge,
    realization and every admitted structural choice) and the native method
    are recorded separately so two realizations of one source never share a
    ``conversion_id``.
    """

    source_sha256: str
    source_byte_size: int
    source_license_id: str
    source_manifest_id: str
    source_admission_id: str
    source_commit: str
    model_id: str | None
    interpreter_sha256: str
    interpreter_version: str
    worker_sha256: str
    ema_n_averaged: int
    source_dtypes: tuple[str, ...]
    basis_residual: float
    excluded: Mapping[str, str]
    declaration: Mapping[str, Any]
    roll_gauge: str
    native_architecture_id: str
    native_method_id: str
    conversion_id: str

    def to_record(self) -> dict[str, Any]:
        record = {
            item.name: getattr(self, item.name) for item in dataclasses.fields(self)
        }
        record["excluded"] = dict(self.excluded)
        record["declaration"] = json.loads(json.dumps(self.declaration))
        record["source_dtypes"] = list(self.source_dtypes)
        return record

    @classmethod
    def from_record(cls, record: Mapping[str, Any], /) -> UMASourceProvenance:
        names = {item.name for item in dataclasses.fields(cls)}
        if not isinstance(record, Mapping) or set(record) != names:
            raise ValueError("UMA source provenance fields are not canonical.")
        excluded = MappingProxyType(dict(record["excluded"]))
        declaration = MappingProxyType(dict(record["declaration"]))
        source_dtypes = tuple(record["source_dtypes"])
        provenance = cls(
            source_sha256=record["source_sha256"],
            source_byte_size=record["source_byte_size"],
            source_license_id=record["source_license_id"],
            source_manifest_id=record["source_manifest_id"],
            source_admission_id=record["source_admission_id"],
            source_commit=record["source_commit"],
            model_id=record["model_id"],
            interpreter_sha256=record["interpreter_sha256"],
            interpreter_version=record["interpreter_version"],
            worker_sha256=record["worker_sha256"],
            ema_n_averaged=record["ema_n_averaged"],
            source_dtypes=source_dtypes,
            basis_residual=record["basis_residual"],
            excluded=excluded,
            declaration=declaration,
            roll_gauge=record["roll_gauge"],
            native_architecture_id=record["native_architecture_id"],
            native_method_id=record["native_method_id"],
            conversion_id=record["conversion_id"],
        )
        if provenance.conversion_id != _conversion_id(provenance):
            raise ValueError("UMA source provenance identity is not canonical.")
        return provenance


@dataclass(frozen=True, slots=True)
class UMACheckpointConversion:
    """A native UMA model (potential or property model), provenance and source tensors."""

    model: UMAModel
    provenance: UMASourceProvenance
    declaration: UMASourceDeclaration
    source_tensors: Mapping[str, np.ndarray]
    precision: AtomisticPrecisionPolicy | None
    streaming: StreamedRelationPlan | None
    node_block: int

    def reconstruct(self, replacements: Mapping[str, np.ndarray], /) -> UMAModel:
        """Rebuild through the same factory with existing source tensors replaced."""
        from ...nn.atomistic._uma_source import uma_model_from_source
        from ...units import ANGSTROM, ELECTRONVOLT

        tensors = dict(self.source_tensors)
        for name, value in replacements.items():
            original = tensors.get(name)
            replacement = np.asarray(value)
            if original is None or replacement.shape != original.shape:
                raise ValueError(
                    f"Replacement {name!r} is not an existing source tensor."
                )
            tensors[name] = replacement
        model, _ = uma_model_from_source(
            AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
            self.declaration,
            tensors,
            precision=self.precision,
            streaming=self.streaming,
            node_block=self.node_block,
        )
        return model


def _worker_bytes() -> bytes:
    return resources.files(__package__).joinpath(_WORKER_NAME).read_bytes()


def _conversion_id(provenance: UMASourceProvenance, /) -> str:
    record = provenance.to_record()
    record.pop("conversion_id")
    return canonical_fingerprint({"kind": "uma-source-conversion", **record})


def extract_uma_checkpoint(
    source: UMASource,
    /,
    *,
    runtime: UMAExtractionRuntime,
    limits: UMAConversionLimits = UMAConversionLimits(),
) -> UMASourceExtraction:
    """Run the isolated extraction and admit its result."""
    if not isinstance(source, UMASource):
        raise TypeError("source must be a UMASource.")
    if not isinstance(runtime, UMAExtractionRuntime):
        raise TypeError("runtime must be a UMAExtractionRuntime.")
    if not isinstance(limits, UMAConversionLimits):
        raise TypeError("limits must be UMAConversionLimits.")
    if source.manifest.byte_size > limits.max_source_bytes:
        raise ValueError("UMA source exceeds the declared source byte limit.")
    payload = read_admitted_artifact(
        source.artifact, source.manifest, policy=source.policy
    )
    request = {
        "operation": "extract",
        "source": {
            "sha256": source.artifact.sha256,
            "byte_size": source.artifact.byte_size,
        },
        "limits": limits.worker_limits(),
    }
    inputs = {
        "worker.py": _worker_bytes(),
        "request.json": json.dumps(request, allow_nan=False, sort_keys=True).encode(
            "utf-8"
        ),
        "source.bin": payload,
    }
    staged = sum(len(value) for value in inputs.values())
    with tempfile.TemporaryDirectory(prefix="phydrax-uma-extract-") as destination:
        try:
            result = run_pinned_command(
                runtime.interpreter,
                ["-I", "-S", "-c", _BOOTSTRAP, "worker.py", "request.json", "result.zip"],
                inputs=inputs,
                timeout=limits.timeout_seconds,
                max_output_bytes=staged + limits.max_log_bytes,
                execution_policy=ExternalExecutionPolicy(inherit_environment=False),
                artifacts=PinnedFileOutputs(
                    destination,
                    (PinnedFileRequest("result.zip", limits.max_result_bytes),),
                    limits.max_result_bytes,
                ),
            )
        except ExternalRuntimeError as error:
            failed = error.result
            if failed is not None and failed.returncode == _REFUSED_EXIT:
                lines = failed.stderr.decode("utf-8", "replace").strip().splitlines()
                refusal = next(
                    (line for line in reversed(lines) if line.startswith("REFUSED: ")),
                    "REFUSED: extraction refusal without a message",
                )
                raise UMASourceRefusedError(refusal.removeprefix("REFUSED: ")) from error
            raise
        location = Path(result.file_artifact("result.zip").location)
        manifest, arrays = read_array_archive(location, limits=limits.archive_limits())
    return _admitted_extraction(manifest, arrays, source)


def _admitted_extraction(
    manifest: Mapping[str, Any], arrays: Mapping[str, np.ndarray], source: UMASource, /
) -> UMASourceExtraction:
    if (
        set(manifest) != {"arrays", "format", "operation", "source", "checkpoint"}
        or manifest["format"] != _RESULT_FORMAT
        or manifest["operation"] != "extract"
        or manifest["source"]
        != {"sha256": source.artifact.sha256, "byte_size": source.artifact.byte_size}
    ):
        raise ValueError("The extraction result does not describe the admitted source.")
    checkpoint = manifest["checkpoint"]
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != _CHECKPOINT_FIELDS:
        raise ValueError("The extraction result checkpoint record is not canonical.")
    model_config = checkpoint["model_config"]
    tasks_config = checkpoint["tasks_config"]
    if not isinstance(model_config, Mapping) or not isinstance(tasks_config, list):
        raise ValueError("The extracted configuration has the wrong structure.")
    averaged = checkpoint["ema_n_averaged"]
    if type(averaged) is not int or averaged < 0:
        raise ValueError("The EMA averaging count must be a nonnegative integer.")
    configuration_arrays: dict[str, np.ndarray] = {}
    tensors: dict[str, np.ndarray] = {}
    for name, value in arrays.items():
        if name.startswith("config/"):
            configuration_arrays[name] = value
        elif name.startswith("ema/"):
            key = name.removeprefix("ema/")
            if not key.startswith(_EMA_PREFIX):
                raise ValueError(f"EMA parameter {key!r} lacks the AveragedModel prefix.")
            tensors[key.removeprefix(_EMA_PREFIX)] = value
        else:
            raise ValueError(f"Unexpected extraction array {name!r}.")
    inventory = []
    for entry in checkpoint["model_state_inventory"]:
        if (
            not isinstance(entry, list)
            or len(entry) != 3
            or not isinstance(entry[0], str)
            or not isinstance(entry[1], str)
            or not isinstance(entry[2], list)
        ):
            raise ValueError("The training-weight inventory is malformed.")
        inventory.append((entry[0], entry[1], tuple(int(extent) for extent in entry[2])))
    training = {name: (dtype, shape) for name, dtype, shape in inventory}
    expected = {
        name: (np.asarray(value).dtype.str, np.asarray(value).shape)
        for name, value in tensors.items()
    }
    if training != expected:
        raise ValueError(
            "The EMA parameter inventory differs from the training-weight inventory."
        )
    return UMASourceExtraction(
        model_config=MappingProxyType(dict(model_config)),
        tasks_config=tuple(MappingProxyType(dict(task)) for task in tasks_config),
        configuration_arrays=MappingProxyType(configuration_arrays),
        tensors=MappingProxyType(tensors),
        ema_n_averaged=averaged,
        training_inventory=tuple(sorted(inventory)),
    )


def convert_uma_checkpoint(
    source: UMASource,
    /,
    *,
    runtime: UMAExtractionRuntime,
    precision: AtomisticPrecisionPolicy | None = None,
    streaming: StreamedRelationPlan | None = None,
    node_block: int = 1024,
    roll_gauge: UMAEdgeRollGauge = "two-chart-nlerp-zero-roll",
    limits: UMAConversionLimits = UMAConversionLimits(),
) -> UMACheckpointConversion:
    """Convert one admitted UMA inference checkpoint into its native model kind.

    Source energies are eV and lengths Angstrom. Unadmitted configurations
    refuse with ``UMASourceConfigurationError``; unadmitted pickles with
    `UMASourceRefusedError`; inconsistent parameters with ``ValueError``.
    """
    from ...nn.atomistic._uma_source import (
        parse_uma_source_config,
        uma_model_from_source,
        UMA_SOURCE_COMMIT,
    )
    from ...units import ANGSTROM, ELECTRONVOLT

    gauge = parse(roll_gauge, UMAEdgeRollGauge, "roll_gauge")
    extraction = extract_uma_checkpoint(source, runtime=runtime, limits=limits)
    declaration = parse_uma_source_config(
        extraction.model_config,
        extraction.tasks_config,
        arrays=extraction.configuration_arrays,
        roll_gauge=gauge,
    )
    model, residual = uma_model_from_source(
        AtomisticScaleContract(ANGSTROM, ELECTRONVOLT),
        declaration,
        extraction.tensors,
        precision=precision,
        streaming=streaming,
        node_block=node_block,
    )
    excluded = {
        **declaration.excluded,
        "ema_state_dict.n_averaged": f"EMA averaging count {extraction.ema_n_averaged}",
        "model_state_dict": (
            "non-EMA training weights; the source inference route loads EMA weights "
            f"({len(extraction.training_inventory)} tensors, inventory-checked)"
        ),
    }
    unsigned = UMASourceProvenance(
        source_sha256=source.artifact.sha256,
        source_byte_size=source.artifact.byte_size,
        source_license_id=source.artifact.license_id,
        source_manifest_id=source.artifact.manifest_id,
        source_admission_id=source.artifact.admission_id,
        source_commit=UMA_SOURCE_COMMIT,
        model_id=declaration.model_id,
        interpreter_sha256=runtime.interpreter.sha256,
        interpreter_version=runtime.interpreter.version,
        worker_sha256=hashlib.sha256(_worker_bytes()).hexdigest(),
        ema_n_averaged=extraction.ema_n_averaged,
        source_dtypes=tuple(
            sorted({value.dtype.str for value in extraction.tensors.values()})
        ),
        basis_residual=residual,
        excluded=MappingProxyType(excluded),
        declaration=MappingProxyType(dict(declaration.record)),
        roll_gauge=gauge,
        native_architecture_id=model.architecture_id,
        native_method_id=model.method_id,
        conversion_id="pending",
    )
    provenance = dataclasses.replace(unsigned, conversion_id=_conversion_id(unsigned))
    return UMACheckpointConversion(
        model=model,
        provenance=provenance,
        declaration=declaration,
        source_tensors=extraction.tensors,
        precision=precision,
        streaming=streaming,
        node_block=node_block,
    )


__all__ = [
    "convert_uma_checkpoint",
    "extract_uma_checkpoint",
    "UMACheckpointConversion",
    "UMAConversionLimits",
    "UMAExtractionRuntime",
    "UMASource",
    "UMASourceExtraction",
    "UMASourceProvenance",
    "UMASourceRefusedError",
]
