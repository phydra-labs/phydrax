#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import equinox as eqx

from ...._document_resource import decode_json_resource
from ...._external_resource import read_bounded_resource, ResourceLimits
from ...._model import (
    operator_architecture_codec,
    operator_architecture_codec_for,
)
from ...._model._structure import (
    deserialize_model_leaf as _deserialize_leaf,
    model_from_structure_recipe as _from_structure_recipe,
    model_structure_recipe as _structure_recipe,
    serialize_model_leaf as _serialize_leaf,
)
from ...._publication import publish_resource_set
from ...._resource_set import read_bounded_resource_set, ResourceSetLimits
from ....privacy import PrivacyCertificate
from ..capabilities import OperatorTrainingEvidence
from ..data import OperatorBatch
from ..protocols import OperatorModel
from ..task import OperatorTask
from ._dtype import (
    OperatorDTypePolicy,
    OperatorPrecisionEvidence,
)
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline
from ._trained_operator import TrainedOperator


_OPERATOR_ARTIFACT_FORMAT = "phydrax-operator-artifact"


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorArtifactTrainingState:
    """Restored optional optimizer/loop state and its immutable metadata."""

    state: Any
    metadata: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorArtifactManifest:
    """Verified manifest for one native or externally backed trained operator."""

    format: str
    artifact_id: str
    task: Mapping[str, Any]
    task_fingerprint: str
    contract_fingerprint: str
    output_field_map: Mapping[str, str]
    fixed_query_fingerprints: Mapping[str, str]
    output_pipeline_fingerprint: str
    output_pipeline_recipe: Mapping[str, Any] | None
    execution_model_file: str
    execution_model_sha256: str
    execution_model_portable: bool
    execution_model_architecture_id: str
    execution_model_factory_id: str
    execution_model_recipe: Mapping[str, Any] | None
    normalization: Mapping[str, Any] | None
    dtype_policy: Mapping[str, str | None]
    precision_evidence: Mapping[str, str | None]
    training_evidence: Mapping[str, str]
    provenance: Mapping[str, Any]
    calibration: Mapping[str, Any]
    training_file: str | None
    training_sha256: str
    training_recipe: Mapping[str, Any] | None
    training_metadata: Mapping[str, Any]
    privacy_certificate: Mapping[str, Any] | None
    privacy_classification: Literal["restricted", "public"] | None
    external_manifest: Mapping[str, Any] | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "OperatorArtifactManifest":
        expected = {
            "format",
            "artifact_id",
            "task",
            "task_fingerprint",
            "contract_fingerprint",
            "output_field_map",
            "fixed_query_fingerprints",
            "output_pipeline_fingerprint",
            "output_pipeline_recipe",
            "execution_model_file",
            "execution_model_sha256",
            "execution_model_portable",
            "execution_model_architecture_id",
            "execution_model_factory_id",
            "execution_model_recipe",
            "normalization",
            "dtype_policy",
            "precision_evidence",
            "training_evidence",
            "provenance",
            "calibration",
            "training_file",
            "training_sha256",
            "training_recipe",
            "training_metadata",
            "privacy_certificate",
            "privacy_classification",
            "external_manifest",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing or unknown:
            raise ValueError(
                "Operator artifact manifest must use the current canonical fields; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        if value["format"] != _OPERATOR_ARTIFACT_FORMAT:
            raise ValueError("File is not a PhydraX operator artifact.")
        portable = bool(value["execution_model_portable"])
        architecture_id = str(value["execution_model_architecture_id"])
        factory_id = str(value["execution_model_factory_id"])
        if portable and not architecture_id:
            raise ValueError(
                "Portable operator artifacts require an architecture codec ID."
            )
        if not portable and not factory_id:
            raise ValueError(
                "Nonportable operator artifacts require an execution-model factory ID."
            )
        privacy_certificate = value["privacy_certificate"]
        privacy_classification = value["privacy_classification"]
        if privacy_classification not in (None, "restricted", "public"):
            raise ValueError("Operator artifact privacy classification is invalid.")
        if (privacy_certificate is None) != (privacy_classification is None):
            raise ValueError(
                "Operator artifact privacy certificate and classification disagree."
            )
        return cls(
            format=str(value["format"]),
            artifact_id=str(value["artifact_id"]),
            task=value["task"],
            task_fingerprint=str(value["task_fingerprint"]),
            contract_fingerprint=str(value["contract_fingerprint"]),
            output_field_map=value["output_field_map"],
            fixed_query_fingerprints=value["fixed_query_fingerprints"],
            output_pipeline_fingerprint=str(value["output_pipeline_fingerprint"]),
            output_pipeline_recipe=value["output_pipeline_recipe"],
            execution_model_file=str(value["execution_model_file"]),
            execution_model_sha256=str(value["execution_model_sha256"]),
            execution_model_portable=portable,
            execution_model_factory_id=factory_id,
            execution_model_recipe=value["execution_model_recipe"],
            normalization=value["normalization"],
            dtype_policy=value["dtype_policy"],
            precision_evidence=value["precision_evidence"],
            execution_model_architecture_id=architecture_id,
            training_evidence=value["training_evidence"],
            provenance=value["provenance"],
            calibration=value["calibration"],
            training_file=value["training_file"],
            training_sha256=str(value["training_sha256"]),
            training_recipe=value["training_recipe"],
            training_metadata=value["training_metadata"],
            privacy_certificate=value["privacy_certificate"],
            privacy_classification=privacy_classification,
            external_manifest=value["external_manifest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _operator_member(source: Path, name: str, /) -> bytes:
    return read_bounded_resource(
        name,
        trusted_root=source,
        limits=ResourceLimits(16 * 1024 * 1024 * 1024, 1, 1, 0, 0),
    ).data


def load_operator_artifact_manifest(path: str | Path, /) -> OperatorArtifactManifest:
    source = Path(path).expanduser().absolute()
    bundle = read_bounded_resource_set(
        source.name,
        trusted_root=source.parent,
        limits=ResourceSetLimits(
            32 * 1024 * 1024 * 1024,
            16 * 1024 * 1024 * 1024,
            3,
            1,
        ),
    )
    manifest_resource = read_bounded_resource(
        "manifest.json",
        trusted_root=source,
        limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
    )
    value = decode_json_resource(manifest_resource).value
    if not isinstance(value, Mapping):
        raise TypeError("Operator artifact manifest must contain an object.")
    manifest = OperatorArtifactManifest.from_dict(value)
    expected = {"manifest.json", manifest.execution_model_file}
    if manifest.training_file is not None:
        expected.add(manifest.training_file)
    if {member.relative_path for member in bundle.manifest.members} != expected:
        raise ValueError("Operator artifact bundle inventory changed.")
    execution_model = _operator_member(source, manifest.execution_model_file)
    if hashlib.sha256(execution_model).hexdigest() != manifest.execution_model_sha256:
        raise ValueError("Operator artifact execution-model checksum mismatch.")
    if manifest.training_file is not None:
        training = _operator_member(source, manifest.training_file)
        if hashlib.sha256(training).hexdigest() != manifest.training_sha256:
            raise ValueError("Operator artifact training-state checksum mismatch.")
    task = OperatorTask.from_dict(manifest.task)
    if task.fingerprint != manifest.task_fingerprint:
        raise ValueError("Operator artifact task fingerprint mismatch.")
    if manifest.privacy_classification == "public":
        if not isinstance(manifest.privacy_certificate, Mapping):
            raise ValueError("Public private artifact has no privacy certificate.")
        PrivacyCertificate.from_record(
            manifest.privacy_certificate
        ).require_public_release()
    return manifest


def save_operator_artifact(
    path: str | Path,
    trained: TrainedOperator,
    /,
    *,
    training_state: Any | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    portable: bool = True,
    execution_model_factory_id: str = "",
    public_release: bool = False,
) -> Path:
    """Atomically store one inference artifact with optional exact-resume state."""
    if not isinstance(trained, TrainedOperator):
        raise TypeError("save_operator_artifact requires a TrainedOperator.")
    if not isinstance(public_release, bool):
        raise TypeError("public_release must be a Boolean.")
    if trained.privacy_certificate is not None and training_state is not None:
        raise ValueError(
            "Private inference artifacts cannot contain restricted training state."
        )
    if trained.privacy_certificate is not None:
        evidence = trained.training_evidence
        if evidence.checkpoint_id or evidence.corpus_id:
            raise ValueError(
                "Private inference artifacts cannot publish training evidence IDs."
            )
        if trained.fixed_query_fingerprints:
            raise ValueError(
                "Private inference artifacts cannot publish fixed-query fingerprints."
            )
        if trained.provenance or trained.calibration or training_metadata:
            raise ValueError(
                "Private inference artifacts require empty provenance, calibration, and training metadata."
            )
        if public_release:
            trained.privacy_certificate.require_public_release()
    factory_id = str(execution_model_factory_id)
    architecture_id = ""
    execution_model_recipe: Mapping[str, Any] | None
    output_pipeline_recipe: Mapping[str, Any] | None
    if portable:
        architecture_codec = operator_architecture_codec_for(trained.execution_model)
        architecture_id = architecture_codec.architecture_id
        execution_model_recipe = (
            _structure_recipe(trained.execution_model, path="execution_model")
            if architecture_codec.encode is None
            else {
                "kind": "architecture_codec",
                "configuration": dict(architecture_codec.encode(trained.execution_model)),
            }
        )
        output_pipeline_recipe = (
            None
            if trained.output_pipeline is None
            else _structure_recipe(trained.output_pipeline, path="output_pipeline")
        )
    else:
        if not factory_id:
            raise ValueError(
                "Nonportable artifacts require an execution_model_factory_id."
            )
        execution_model_recipe = None
        output_pipeline_recipe = None
    training_recipe = (
        None
        if training_state is None
        else _structure_recipe(training_state, path="training_state")
    )
    destination = Path(path)
    model_buffer = io.BytesIO()
    eqx.tree_serialise_leaves(
        model_buffer,
        trained.execution_model,
        filter_spec=_serialize_leaf,
    )
    model_bytes = model_buffer.getvalue()
    model_checksum = hashlib.sha256(model_bytes).hexdigest()
    model_name = f"execution-model-{model_checksum[:16]}.eqx"

    training_name: str | None = None
    training_checksum = ""
    training_bytes: bytes | None = None
    if training_state is not None:
        training_buffer = io.BytesIO()
        eqx.tree_serialise_leaves(
            training_buffer,
            training_state,
            filter_spec=_serialize_leaf,
        )
        training_bytes = training_buffer.getvalue()
        training_checksum = hashlib.sha256(training_bytes).hexdigest()
        training_name = f"training-{training_checksum[:16]}.eqx"

    from ..adapters import ExternalOperatorAdapter

    external_manifest = None
    if isinstance(trained.execution_model, ExternalOperatorAdapter):
        external_manifest = trained.execution_model.manifest.to_dict()
    artifact_id = (
        trained.artifact_id
        or hashlib.sha256(
            f"{trained.execution_plan.fingerprint}:{model_checksum}".encode("utf-8")
        ).hexdigest()
    )
    evidence = trained.training_evidence
    manifest = OperatorArtifactManifest(
        format=_OPERATOR_ARTIFACT_FORMAT,
        artifact_id=artifact_id,
        task=trained.task.to_dict(),
        task_fingerprint=trained.task_fingerprint,
        contract_fingerprint=trained.contract_fingerprint,
        output_field_map=dict(trained.output_field_map),
        fixed_query_fingerprints=dict(trained.fixed_query_fingerprints),
        output_pipeline_fingerprint=(
            "" if trained.output_pipeline is None else trained.output_pipeline.fingerprint
        ),
        output_pipeline_recipe=output_pipeline_recipe,
        execution_model_file=model_name,
        execution_model_sha256=model_checksum,
        execution_model_portable=bool(portable),
        execution_model_architecture_id=architecture_id,
        execution_model_factory_id=factory_id,
        execution_model_recipe=execution_model_recipe,
        normalization=(
            None if trained.normalization is None else trained.normalization.to_dict()
        ),
        dtype_policy=trained.dtype_policy.to_dict(),
        precision_evidence=trained.precision_evidence.to_dict(),
        training_evidence={
            "regime": evidence.regime,
            "checkpoint_id": evidence.checkpoint_id,
            "corpus_id": evidence.corpus_id,
        },
        provenance=dict(trained.provenance),
        calibration=dict(trained.calibration),
        training_file=training_name,
        training_sha256=training_checksum,
        training_recipe=training_recipe,
        training_metadata=({} if training_metadata is None else dict(training_metadata)),
        privacy_certificate=(
            None
            if trained.privacy_certificate is None
            else trained.privacy_certificate.to_record()
        ),
        privacy_classification=(
            None
            if trained.privacy_certificate is None
            else ("public" if public_release else "restricted")
        ),
        external_manifest=external_manifest,
    )
    members = {
        model_name: model_bytes,
        "manifest.json": (
            json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True)
            + "\n"
        ).encode("utf-8"),
    }
    if training_name is not None and training_bytes is not None:
        members[training_name] = training_bytes
    publish_resource_set(
        destination,
        members,
        limits=ResourceSetLimits(
            max_total_bytes=32 * 1024 * 1024 * 1024,
            max_member_bytes=16 * 1024 * 1024 * 1024,
            max_members=3,
            max_depth=1,
        ),
        mode="atomic_replace",
    )
    return destination


def load_trained_operator(
    path: str | Path,
    /,
    *,
    execution_model_like: OperatorModel | None = None,
    output_pipeline_like: OperatorOutputPipeline | None = None,
) -> TrainedOperator:
    """Verify and restore a task-bound operator without templates when portable."""
    source = Path(path)
    manifest = load_operator_artifact_manifest(source)
    if manifest.execution_model_portable:
        if manifest.execution_model_recipe is None:
            raise ValueError("Portable operator artifact has no execution-model recipe.")
        architecture_codec = operator_architecture_codec(
            manifest.execution_model_architecture_id
        )
        if manifest.execution_model_recipe.get("kind") == "architecture_codec":
            if architecture_codec.decode is None:
                raise ValueError(
                    f"Operator architecture {architecture_codec.architecture_id!r} "
                    "does not define a configuration decoder."
                )
            model_template = architecture_codec.decode(
                manifest.execution_model_recipe["configuration"]
            )
        else:
            model_template = _from_structure_recipe(manifest.execution_model_recipe)
        if type(model_template) is not architecture_codec.model_type:
            raise TypeError(
                f"Operator architecture codec {architecture_codec.architecture_id!r} "
                f"restored {type(model_template).__name__}, expected "
                f"{architecture_codec.model_type.__name__}."
            )
        output_pipeline = (
            None
            if manifest.output_pipeline_recipe is None
            else _from_structure_recipe(manifest.output_pipeline_recipe)
        )
    else:
        if execution_model_like is None:
            raise ValueError(
                "Nonportable artifacts require an explicit execution_model_like."
            )
        model_template = execution_model_like
        if manifest.output_pipeline_fingerprint:
            if output_pipeline_like is None:
                raise ValueError(
                    "Nonportable artifacts with a physical output pipeline require output_pipeline_like."
                )
            output_pipeline = output_pipeline_like
        else:
            if output_pipeline_like is not None:
                raise ValueError("Artifact does not declare an output pipeline.")
            output_pipeline = None
    if not isinstance(model_template, OperatorModel):
        raise TypeError("Operator artifact recipe did not restore an execution model.")
    if output_pipeline is not None and not isinstance(
        output_pipeline, OperatorOutputPipeline
    ):
        raise TypeError("Operator artifact recipe did not restore an output pipeline.")
    actual_pipeline_fingerprint = (
        "" if output_pipeline is None else output_pipeline.fingerprint
    )
    if actual_pipeline_fingerprint != manifest.output_pipeline_fingerprint:
        raise ValueError("Operator artifact output-pipeline fingerprint mismatch.")
    execution_model = eqx.tree_deserialise_leaves(
        io.BytesIO(_operator_member(source, manifest.execution_model_file)),
        model_template,
        filter_spec=_deserialize_leaf,
    )
    task = OperatorTask.from_dict(manifest.task)
    normalization = (
        None
        if manifest.normalization is None
        else OperatorNormalizationPolicy.from_dict(manifest.normalization)
    )
    regime = manifest.training_evidence["regime"]
    if regime not in ("task_specific", "pretrained_system", "task_distribution"):
        raise ValueError(f"Unknown operator training regime {regime!r}.")
    evidence = OperatorTrainingEvidence(
        regime=regime,
        checkpoint_id=manifest.training_evidence.get("checkpoint_id", ""),
        corpus_id=manifest.training_evidence.get("corpus_id", ""),
    )
    dtype_policy = OperatorDTypePolicy.from_dict(dict(manifest.dtype_policy))
    precision_evidence = OperatorPrecisionEvidence.from_dict(manifest.precision_evidence)
    if dtype_policy.precision_evidence != precision_evidence:
        raise ValueError("Operator artifact precision evidence disagrees with policy.")
    privacy_certificate = (
        None
        if manifest.privacy_certificate is None
        else PrivacyCertificate.from_record(manifest.privacy_certificate)
    )
    trained = TrainedOperator(
        execution_model,
        task,
        training_evidence=evidence,
        output_field_map=manifest.output_field_map,
        fixed_query_fingerprints=manifest.fixed_query_fingerprints,
        output_pipeline=output_pipeline,
        normalization=normalization,
        dtype_policy=dtype_policy,
        artifact_id=manifest.artifact_id,
        privacy_certificate=privacy_certificate,
        provenance=dict(manifest.provenance),
        calibration=dict(manifest.calibration),
    )
    if trained.contract_fingerprint != manifest.contract_fingerprint:
        raise ValueError("Operator artifact instance-contract fingerprint mismatch.")
    if trained.precision_evidence != precision_evidence:
        raise ValueError("Operator artifact effective precision mismatch.")
    return trained


def _fixed_query_fingerprints(
    task: OperatorTask,
    batch: OperatorBatch | None,
    /,
) -> dict[str, str]:
    if task.problem.query_is_fixed is not True:
        return {}
    if batch is None:
        raise ValueError("Fixed-query tasks require fixed_query_batch.")
    task.validate_batch(batch)
    return {name: batch.query(name).geometry_fingerprint() for name in task.query_by_name}


def load_operator_training_state(
    path: str | Path,
    /,
    *,
    state_like: Any | None = None,
) -> OperatorArtifactTrainingState:
    """Restore optional exact-resume state from the unified artifact."""
    source = Path(path)
    manifest = load_operator_artifact_manifest(source)
    if manifest.training_file is None:
        raise ValueError("Operator artifact does not contain training state.")
    if manifest.training_recipe is not None:
        template = _from_structure_recipe(manifest.training_recipe)
    elif state_like is not None:
        template = state_like
    else:
        raise ValueError("Nonportable training state requires state_like.")
    state = eqx.tree_deserialise_leaves(
        io.BytesIO(_operator_member(source, manifest.training_file)),
        template,
        filter_spec=_deserialize_leaf,
    )
    return OperatorArtifactTrainingState(
        state=state,
        metadata=dict(manifest.training_metadata),
    )


def load_external_trained_operator(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    loader: Any,
    task: OperatorTask,
    training_evidence: OperatorTrainingEvidence,
    /,
    *,
    input_adapter: Any,
    output_adapter: Any,
    in_size: int | tuple[int, ...] | Literal["scalar"],
    out_size: int | tuple[int, ...] | Literal["scalar"],
    dtype_policy: OperatorDTypePolicy | None = None,
    output_field_map: Mapping[str, str] | None = None,
    fixed_query_batch: OperatorBatch | None = None,
    output_pipeline: OperatorOutputPipeline | None = None,
) -> TrainedOperator:
    """Verify an external checkpoint and place it behind the task-bound runtime."""
    from ..adapters import load_external_operator_adapter

    adapter = load_external_operator_adapter(
        manifest_path,
        checkpoint_path,
        loader,
        input_adapter=input_adapter,
        output_adapter=output_adapter,
        in_size=in_size,
        out_size=out_size,
    )
    manifest = adapter.manifest
    return TrainedOperator(
        adapter,
        task,
        training_evidence=training_evidence,
        output_field_map=output_field_map,
        fixed_query_fingerprints=_fixed_query_fingerprints(task, fixed_query_batch),
        output_pipeline=output_pipeline,
        dtype_policy=dtype_policy,
        provenance={
            "external_manifest": manifest.to_dict(),
            "source_uri": manifest.source_uri,
            "checkpoint_uri": manifest.checkpoint_uri,
            "revision": manifest.revision,
        },
    )


__all__ = [
    "OperatorArtifactManifest",
    "OperatorArtifactTrainingState",
    "load_external_trained_operator",
    "load_operator_artifact_manifest",
    "load_operator_training_state",
    "load_trained_operator",
    "save_operator_artifact",
]
