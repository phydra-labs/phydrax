#
#  Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import jax.numpy as jnp
from jaxtyping import Array

from ...._differentiation import DerivativeContract, DerivativeRoute
from ...._document_resource import decode_json_resource
from ...._external_resource import read_bounded_resource, ResourceLimits
from ...._external_runtime import _require_execution
from ...._frozendict import frozendict
from ...._host_io import open_regular_file
from ...._identity import (
    ArtifactBindingIdentity,
    ExecutableSignature,
    NumericRevision,
    SemanticProvenance,
)
from ...._model._array import value_derivative_contract
from ...._model._component import ExecutionCapabilities, ModelExecutionContract
from ...._publication import publish_bytes
from ...._trainable import NonTrainableState
from ..._keys import EvalKey
from ..data import OperatorBatch
from ..engine import AbstractOperatorModel


class OperatorCheckpointManifest(NonTrainableState):
    """Schema, preprocessing, provenance, and license contract."""

    architecture: str
    model_version: str
    source_uri: str
    checkpoint_uri: str
    revision: str
    input_schema: frozendict[str, Any]
    output_schema: frozendict[str, Any]
    preprocessing: frozendict[str, Any]
    normalization: frozendict[str, Any]
    dataset_provenance: tuple[str, ...]
    code_license: str
    weights_license: str
    checkpoint_sha256: str

    def __init__(
        self,
        *,
        architecture: str,
        model_version: str,
        source_uri: str,
        checkpoint_uri: str,
        revision: str,
        input_schema: Mapping[str, Any],
        output_schema: Mapping[str, Any],
        preprocessing: Mapping[str, Any],
        normalization: Mapping[str, Any],
        dataset_provenance: Sequence[str],
        code_license: str,
        weights_license: str,
        checkpoint_sha256: str,
    ):
        self.architecture = str(architecture)
        self.model_version = str(model_version)
        self.source_uri = str(source_uri)
        self.checkpoint_uri = str(checkpoint_uri)
        self.revision = str(revision)
        self.input_schema = frozendict(input_schema)
        self.output_schema = frozendict(output_schema)
        self.preprocessing = frozendict(preprocessing)
        self.normalization = frozendict(normalization)
        self.dataset_provenance = tuple(str(value) for value in dataset_provenance)
        self.code_license = str(code_license)
        self.weights_license = str(weights_license)
        self.checkpoint_sha256 = str(checkpoint_sha256).lower()
        required_strings = {
            "architecture": self.architecture,
            "model_version": self.model_version,
            "source_uri": self.source_uri,
            "checkpoint_uri": self.checkpoint_uri,
            "revision": self.revision,
            "code_license": self.code_license,
            "weights_license": self.weights_license,
        }
        empty = tuple(name for name, value in required_strings.items() if not value)
        if empty:
            raise ValueError(f"Operator manifest fields must be non-empty: {empty}.")
        if (
            not self.input_schema
            or not self.output_schema
            or not self.preprocessing
            or not self.normalization
            or not self.dataset_provenance
        ):
            raise ValueError(
                "Operator manifests require schemas, preprocessing, normalization, and dataset provenance."
            )
        if len(self.checkpoint_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.checkpoint_sha256
        ):
            raise ValueError(
                "checkpoint_sha256 must be a 64-character hexadecimal digest."
            )

    def to_dict(self, /) -> dict[str, Any]:
        return {
            "architecture": self.architecture,
            "model_version": self.model_version,
            "source_uri": self.source_uri,
            "checkpoint_uri": self.checkpoint_uri,
            "revision": self.revision,
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "preprocessing": dict(self.preprocessing),
            "normalization": dict(self.normalization),
            "dataset_provenance": list(self.dataset_provenance),
            "code_license": self.code_license,
            "weights_license": self.weights_license,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any], /) -> OperatorCheckpointManifest:
        required = {
            "architecture",
            "model_version",
            "source_uri",
            "checkpoint_uri",
            "revision",
            "input_schema",
            "output_schema",
            "preprocessing",
            "normalization",
            "dataset_provenance",
            "code_license",
            "weights_license",
            "checkpoint_sha256",
        }
        missing = required.difference(value)
        if missing:
            raise ValueError(f"Operator manifest is missing fields {sorted(missing)}.")
        unknown = set(value).difference(required)
        if unknown:
            raise ValueError(f"Operator manifest has unknown fields {sorted(unknown)}.")
        return cls(
            architecture=str(value["architecture"]),
            model_version=str(value["model_version"]),
            source_uri=str(value["source_uri"]),
            checkpoint_uri=str(value["checkpoint_uri"]),
            revision=str(value["revision"]),
            input_schema=value["input_schema"],
            output_schema=value["output_schema"],
            preprocessing=value["preprocessing"],
            normalization=value["normalization"],
            dataset_provenance=value["dataset_provenance"],
            code_license=str(value["code_license"]),
            weights_license=str(value["weights_license"]),
            checkpoint_sha256=str(value["checkpoint_sha256"]),
        )

    def binding_identity(self) -> ArtifactBindingIdentity:
        """Artifact binding identity of the checkpoint this manifest describes.

        The semantic provenance is the manifest content with the checkpoint as a
        named resource, the numeric revision is the checkpoint digest, and the
        executable signature records the declared architecture release.
        """
        semantic = SemanticProvenance(
            {"kind": "external-operator-checkpoint", **self.to_dict()},
            resource_ids={"checkpoint": self.checkpoint_sha256},
        )
        return ArtifactBindingIdentity(
            semantic,
            NumericRevision(semantic, {"checkpoint_sha256": self.checkpoint_sha256}),
            ExecutableSignature(
                algorithm_facts={
                    "architecture": self.architecture,
                    "model_version": self.model_version,
                    "revision": self.revision,
                }
            ),
        )


def save_operator_manifest(
    path: str | Path,
    manifest: OperatorCheckpointManifest,
    /,
) -> None:
    destination = Path(path)
    payload = (
        json.dumps(manifest.to_dict(), allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    publish_bytes(
        destination,
        payload,
        maximum_bytes=16 * 1024 * 1024,
        mode="atomic_replace",
    )


def load_operator_manifest(path: str | Path, /) -> OperatorCheckpointManifest:
    source = Path(path).expanduser().absolute()
    resource = read_bounded_resource(
        source.name,
        trusted_root=source.parent,
        limits=ResourceLimits(16 * 1024 * 1024, 64, 100_000, 100_000, 0),
    )
    payload = decode_json_resource(resource).value
    if not isinstance(payload, Mapping):
        raise TypeError("Operator manifest JSON must contain an object.")
    return OperatorCheckpointManifest.from_dict(payload)


def checkpoint_sha256(path: str | Path, /) -> str:
    digest = hashlib.sha256()
    with open_regular_file(path) as checkpoint:
        while chunk := checkpoint.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify_operator_checkpoint(
    path: str | Path,
    manifest: OperatorCheckpointManifest,
    /,
) -> bool:
    return checkpoint_sha256(path) == manifest.checkpoint_sha256


def _external_operator_contract(model):
    from ..catalog import operator_architecture_contract

    return operator_architecture_contract(
        model.manifest.architecture,
        configuration=(
            ("model_version", model.manifest.model_version),
            ("revision", model.manifest.revision),
        ),
    )


class ExternalOperatorAdapter(AbstractOperatorModel):
    """Schema-checked bridge from OperatorBatch to an externally loaded model.

    ``input_adapter`` owns normalization/tokenization, ``runner`` owns invocation,
    and ``output_adapter`` restores PhydraX channels/query layout. This keeps
    framework-specific tensor conventions out of the operator/domain runtime.

    The runner is an opaque fixed artifact: ``capabilities`` declare how it
    executes and ``binding`` is derived from ``manifest.binding_identity()``, the
    single authority identifying the loaded artifact. Every call is
    admitted against the capabilities before the input adapter or runner runs,
    so a host-only runner is refused under ``jit``, ``vmap``, ``grad``, ``jvp``,
    and ``vjp`` without being invoked. A host-only runner offers no derivative
    (``STOPPED`` route); a JAX-tier runner keeps the conservative undeclared
    ``DIRECT`` contract.
    """

    _operator_contract_builder = staticmethod(_external_operator_contract)

    runner: Callable[[Any, EvalKey], Any]
    input_adapter: Callable[[OperatorBatch, OperatorCheckpointManifest], Any]
    output_adapter: Callable[[Any, OperatorBatch, OperatorCheckpointManifest], Array]
    manifest: OperatorCheckpointManifest
    capabilities: ExecutionCapabilities
    binding: ArtifactBindingIdentity
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        *,
        runner: Callable[[Any, EvalKey], Any],
        input_adapter: Callable[[OperatorBatch, OperatorCheckpointManifest], Any],
        output_adapter: Callable[[Any, OperatorBatch, OperatorCheckpointManifest], Array],
        manifest: OperatorCheckpointManifest,
        capabilities: ExecutionCapabilities,
        in_size: int | tuple[int, ...] | Literal["scalar"],
        out_size: int | tuple[int, ...] | Literal["scalar"],
    ):
        if (
            not callable(runner)
            or not callable(input_adapter)
            or not callable(output_adapter)
        ):
            raise TypeError("runner, input_adapter, and output_adapter must be callable.")
        if not isinstance(manifest, OperatorCheckpointManifest):
            raise TypeError("manifest must be an OperatorCheckpointManifest.")
        if not isinstance(capabilities, ExecutionCapabilities):
            raise TypeError("capabilities must be ExecutionCapabilities.")
        self.runner = runner
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.manifest = manifest
        self.capabilities = capabilities
        self.binding = manifest.binding_identity()
        self.in_size = in_size
        self.out_size = out_size

    def model_execution_contract(self) -> ModelExecutionContract:
        """Return the declared capabilities with the runner's derivative route."""
        derivative = (
            DerivativeContract(route=DerivativeRoute.STOPPED)
            if self.capabilities.host_only
            else value_derivative_contract(None)
        )
        return self._execution_contract(derivative, execution=self.capabilities)

    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        _require_execution(self.capabilities, batch, key)
        payload = self.input_adapter(batch, self.manifest)
        raw_output = self.runner(payload, key)
        return jnp.asarray(self.output_adapter(raw_output, batch, self.manifest))

    def __call__(
        self,
        x: Array | tuple[Array, ...] | OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
        if not isinstance(x, OperatorBatch):
            raise TypeError("ExternalOperatorAdapter requires an OperatorBatch.")
        return self.__call_operator_batch__(x, key=key)


def load_external_operator_adapter(
    manifest_path: str | Path,
    checkpoint_path: str | Path,
    loader: Callable[
        [OperatorCheckpointManifest, Path],
        Callable[[Any, EvalKey], Any],
    ],
    /,
    *,
    input_adapter: Callable[[OperatorBatch, OperatorCheckpointManifest], Any],
    output_adapter: Callable[[Any, OperatorBatch, OperatorCheckpointManifest], Array],
    capabilities: ExecutionCapabilities,
    in_size: int | tuple[int, ...] | Literal["scalar"],
    out_size: int | tuple[int, ...] | Literal["scalar"],
) -> ExternalOperatorAdapter:
    """Verify a checkpoint before loading it behind the operator protocol.

    The adapter is bound to `binding_identity()` of the verified manifest and
    executes under the declared `capabilities`.
    """
    manifest = load_operator_manifest(manifest_path)
    checkpoint = Path(checkpoint_path)
    if not verify_operator_checkpoint(checkpoint, manifest):
        raise ValueError("External operator checkpoint checksum mismatch.")
    runner = loader(manifest, checkpoint)
    return ExternalOperatorAdapter(
        runner=runner,
        input_adapter=input_adapter,
        output_adapter=output_adapter,
        manifest=manifest,
        capabilities=capabilities,
        in_size=in_size,
        out_size=out_size,
    )


__all__ = [
    "ExternalOperatorAdapter",
    "OperatorCheckpointManifest",
    "checkpoint_sha256",
    "load_external_operator_adapter",
    "load_operator_manifest",
    "save_operator_manifest",
    "verify_operator_checkpoint",
]
