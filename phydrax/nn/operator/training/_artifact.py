#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import equinox as eqx

from ...._model import (
    operator_architecture_codec,
    operator_architecture_codec_for,
)
from ...._model._structure import (
    deserialise_model_leaf as _deserialise_leaf,
    model_from_structure_recipe as _from_structure_recipe,
    model_structure_recipe as _structure_recipe,
    serialise_model_leaf as _serialise_leaf,
)
from ..capabilities import OperatorTrainingEvidence
from ..data import OperatorBatch
from ..protocols import OperatorModel
from ..task import OperatorTask
from ._dtype import OperatorDTypePolicy
from ._normalization import OperatorNormalizationPolicy
from ._physics import OperatorOutputPipeline
from ._trained_operator import TrainedOperator


_OPERATOR_ARTIFACT_FORMAT = "phydrax-operator-artifact"
_OPERATOR_ARTIFACT_VERSION = 3


def _sha256(path: Path, /) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorArtifactTrainingState:
    """Restored optional optimizer/loop state and its immutable metadata."""

    state: Any
    metadata: Mapping[str, Any]


@dataclasses.dataclass(frozen=True, slots=True)
class OperatorArtifactManifest:
    """Verified manifest for one native or externally backed trained operator."""

    format: str
    version: int
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
    dtype_policy: Mapping[str, str]
    training_evidence: Mapping[str, str]
    provenance: Mapping[str, Any]
    calibration: Mapping[str, Any]
    training_file: str | None
    training_sha256: str
    training_recipe: Mapping[str, Any] | None
    training_metadata: Mapping[str, Any]
    external_manifest: Mapping[str, Any] | None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> "OperatorArtifactManifest":
        expected = {
            "format",
            "version",
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
            "training_evidence",
            "provenance",
            "calibration",
            "training_file",
            "training_sha256",
            "training_recipe",
            "training_metadata",
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
        if value["version"] != _OPERATOR_ARTIFACT_VERSION:
            raise ValueError(
                "Operator artifact version does not match the current runtime."
            )
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
        return cls(
            format=str(value["format"]),
            version=int(value["version"]),
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
            execution_model_architecture_id=architecture_id,
            training_evidence=value["training_evidence"],
            provenance=value["provenance"],
            calibration=value["calibration"],
            training_file=value["training_file"],
            training_sha256=str(value["training_sha256"]),
            training_recipe=value["training_recipe"],
            training_metadata=value["training_metadata"],
            external_manifest=value["external_manifest"],
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def load_operator_artifact_manifest(path: str | Path, /) -> OperatorArtifactManifest:
    source = Path(path)
    value = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest = OperatorArtifactManifest.from_dict(value)
    task = OperatorTask.from_dict(manifest.task)
    if task.fingerprint != manifest.task_fingerprint:
        raise ValueError("Operator artifact task fingerprint mismatch.")
    execution_model_path = source / manifest.execution_model_file
    if _sha256(execution_model_path) != manifest.execution_model_sha256:
        raise ValueError("Operator artifact execution-model checksum mismatch.")
    if manifest.training_file is not None:
        if _sha256(source / manifest.training_file) != manifest.training_sha256:
            raise ValueError("Operator artifact training-state checksum mismatch.")
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
) -> Path:
    """Atomically publish one inference artifact with optional exact-resume state."""
    if not isinstance(trained, TrainedOperator):
        raise TypeError("save_operator_artifact requires a TrainedOperator.")
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
    destination.mkdir(parents=True, exist_ok=True)

    temporary_model = destination / "execution-model.tmp.eqx"
    eqx.tree_serialise_leaves(
        temporary_model,
        trained.execution_model,
        filter_spec=_serialise_leaf,
    )
    model_checksum = _sha256(temporary_model)
    model_name = f"execution-model-{model_checksum[:16]}.eqx"
    os.replace(temporary_model, destination / model_name)

    training_name: str | None = None
    training_checksum = ""
    if training_state is not None:
        temporary_training = destination / "training.tmp.eqx"
        eqx.tree_serialise_leaves(
            temporary_training, training_state, filter_spec=_serialise_leaf
        )
        training_checksum = _sha256(temporary_training)
        training_name = f"training-{training_checksum[:16]}.eqx"
        os.replace(temporary_training, destination / training_name)

    from ..adapters import ExternalOperatorAdapter

    external_manifest = None
    if isinstance(trained.execution_model, ExternalOperatorAdapter):
        external_manifest = trained.execution_model.manifest.to_dict()
    artifact_id = (
        trained.artifact_id
        or hashlib.sha256(
            f"{trained.task_fingerprint}:{trained.contract_fingerprint}:"
            f"{model_checksum}".encode("utf-8")
        ).hexdigest()
    )
    evidence = trained.training_evidence
    manifest = OperatorArtifactManifest(
        format=_OPERATOR_ARTIFACT_FORMAT,
        version=_OPERATOR_ARTIFACT_VERSION,
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
        external_manifest=external_manifest,
    )
    temporary_manifest = destination / "manifest.tmp.json"
    temporary_manifest.write_text(
        json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_manifest, destination / "manifest.json")
    keep = {model_name, "manifest.json"}
    if training_name is not None:
        keep.add(training_name)
    for candidate in destination.iterdir():
        if candidate.is_file() and candidate.name not in keep:
            if candidate.name.startswith(("model-", "execution-model-", "training-")):
                candidate.unlink()
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
                    "Nonportable artifacts with a physical output pipeline require "
                    "output_pipeline_like."
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
        source / manifest.execution_model_file,
        model_template,
        filter_spec=_deserialise_leaf,
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
    trained = TrainedOperator(
        execution_model,
        task,
        training_evidence=evidence,
        output_field_map=manifest.output_field_map,
        fixed_query_fingerprints=manifest.fixed_query_fingerprints,
        output_pipeline=output_pipeline,
        normalization=normalization,
        dtype_policy=OperatorDTypePolicy.from_dict(dict(manifest.dtype_policy)),
        artifact_id=manifest.artifact_id,
        provenance=dict(manifest.provenance),
        calibration=dict(manifest.calibration),
    )
    if trained.contract_fingerprint != manifest.contract_fingerprint:
        raise ValueError("Operator artifact instance-contract fingerprint mismatch.")
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
        source / manifest.training_file,
        template,
        filter_spec=_deserialise_leaf,
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
