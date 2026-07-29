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

from ...._frozendict import frozendict
from ...._trainable import NonTrainableState
from ..core._base import _AbstractOperatorModel
from ..core._keys import EvalKey
from ..core._operator import OperatorBatch


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
                "Operator manifests require schemas, preprocessing, normalization, "
                "and dataset provenance."
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
            raise ValueError(
                f"Operator manifest has unknown fields {sorted(unknown)}."
            )
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


def save_operator_manifest(
    path: str | Path,
    manifest: OperatorCheckpointManifest,
    /,
) -> None:
    destination = Path(path)
    destination.write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_operator_manifest(path: str | Path, /) -> OperatorCheckpointManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Operator manifest JSON must contain an object.")
    return OperatorCheckpointManifest.from_dict(payload)


def checkpoint_sha256(path: str | Path, /) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint:
        for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_operator_checkpoint(
    path: str | Path,
    manifest: OperatorCheckpointManifest,
    /,
) -> bool:
    return checkpoint_sha256(path) == manifest.checkpoint_sha256


class ExternalOperatorAdapter(_AbstractOperatorModel):
    """Schema-checked bridge from OperatorBatch to an externally loaded model.

    ``input_adapter`` owns normalization/tokenization, ``runner`` owns invocation,
    and ``output_adapter`` restores PhydraX channels/query layout. This keeps
    framework-specific tensor conventions out of the operator/domain runtime.
    """

    runner: Callable[[Any, EvalKey], Any]
    input_adapter: Callable[[OperatorBatch, OperatorCheckpointManifest], Any]
    output_adapter: Callable[[Any, OperatorBatch, OperatorCheckpointManifest], Array]
    manifest: OperatorCheckpointManifest
    in_size: int | tuple[int, ...] | Literal["scalar"]
    out_size: int | tuple[int, ...] | Literal["scalar"]

    def __init__(
        self,
        *,
        runner: Callable[[Any, EvalKey], Any],
        input_adapter: Callable[[OperatorBatch, OperatorCheckpointManifest], Any],
        output_adapter: Callable[[Any, OperatorBatch, OperatorCheckpointManifest], Array],
        manifest: OperatorCheckpointManifest,
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
        self.runner = runner
        self.input_adapter = input_adapter
        self.output_adapter = output_adapter
        self.manifest = manifest
        self.in_size = in_size
        self.out_size = out_size


    def __call_operator_batch__(
        self,
        batch: OperatorBatch,
        /,
        *,
        key: EvalKey = None,
    ) -> Array:
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
    in_size: int | tuple[int, ...] | Literal["scalar"],
    out_size: int | tuple[int, ...] | Literal["scalar"],
) -> ExternalOperatorAdapter:
    """Verify a checkpoint before loading it behind the operator protocol."""
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
