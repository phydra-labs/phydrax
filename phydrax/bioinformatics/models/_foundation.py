#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array

from phydrax._fingerprint import (
    array_tree_fingerprint,
    array_tree_signature,
    canonical_fingerprint,
)
from phydrax._strict import StrictModule
from phydrax.nn.parameters import LowRankUpdate

from ..foundation import (
    BioinformaticsMethodContract,
    DifferentiationKind,
    ExecutionKind,
    MethodKind,
    OutputKind,
)
from ..sequence import AlphabetPlan


OverlapAssessment = Literal["unknown", "no-detected-overlap", "known-overlap"]
LicenseStatus = Literal["verified", "unknown", "restricted"]


class FoundationBindingStatus(IntEnum):
    """Successful binding and non-fatal provenance warning statuses."""

    SUCCESS = 0
    PRETRAINING_OVERLAP_UNKNOWN = 1


_BINDING_CONTRACT = BioinformaticsMethodContract(
    "foundation-model-artifact-binding",
    MethodKind.EXACT_MODEL,
    ExecutionKind.EXACT_DISCRETE,
    DifferentiationKind.NONE,
    OutputKind.STRUCTURED,
    conditioning_statement="Canonical SHA-256 identities are compared exactly on the host.",
    truncation_statement="No provenance fields are omitted or truncated.",
    capacity_semantics="Every identity named by the manifest is verified before binding.",
    assumptions=(
        "The caller supplies hashes computed from the artifact bytes being bound.",
    ),
    nondifferentiable_outputs=("identity evidence", "status"),
)


_HEX = frozenset("0123456789abcdef")


def _sha256(value: str, name: str) -> str:
    normalized = str(value).lower()
    if len(normalized) != 64 or any(character not in _HEX for character in normalized):
        raise ValueError(f"{name} must be a lowercase hexadecimal SHA-256 digest.")
    return normalized


def sha256_file(path: str | Path, /) -> str:
    """Hash artifact bytes on the host without admitting paths into model PyTrees."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_model_parameter_sha256(model: Any, /) -> str:
    """Return the canonical content hash of a native numeric model PyTree."""
    return str(array_tree_fingerprint(model)["sha256"])


def native_model_structure_fingerprint(model: Any, /) -> str:
    """Return the canonical array-path/shape/dtype identity of a native model."""
    return canonical_fingerprint(array_tree_signature(model))


class TokenizerProvenance(StrictModule):
    """Immutable tokenizer bytes, vocabulary, normalization, and alphabet identity."""

    tokenizer_id: str = eqx.field(static=True)
    tokenizer_sha256: str = eqx.field(static=True)
    alphabet_fingerprint: str = eqx.field(static=True)
    vocabulary_size: int = eqx.field(static=True)
    normalization: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        tokenizer_id: str,
        tokenizer_sha256: str,
        alphabet: AlphabetPlan,
        /,
        *,
        vocabulary_size: int | None = None,
        normalization: str = "identity",
    ):
        identifier = str(tokenizer_id).strip()
        if not identifier:
            raise ValueError("tokenizer_id must be non-empty.")
        if not isinstance(alphabet, AlphabetPlan):
            raise TypeError("alphabet must be an AlphabetPlan.")
        size = alphabet.size if vocabulary_size is None else int(vocabulary_size)
        if size != alphabet.size:
            raise ValueError(
                "Native biological tokenizers must cover the exact alphabet."
            )
        digest = _sha256(tokenizer_sha256, "tokenizer_sha256")
        normalization_ = str(normalization)
        payload = {
            "tokenizer_id": identifier,
            "tokenizer_sha256": digest,
            "alphabet_fingerprint": alphabet.fingerprint,
            "vocabulary_size": size,
            "normalization": normalization_,
        }
        self.tokenizer_id = identifier
        self.tokenizer_sha256 = digest
        self.alphabet_fingerprint = alphabet.fingerprint
        self.vocabulary_size = size
        self.normalization = normalization_
        self.fingerprint = canonical_fingerprint(payload)


class LicenseProvenance(StrictModule):
    """Auditable license identity and explicit native-use permissions."""

    spdx_id: str = eqx.field(static=True)
    license_sha256: str = eqx.field(static=True)
    status: LicenseStatus = eqx.field(static=True)
    inference_allowed: bool = eqx.field(static=True)
    adaptation_allowed: bool = eqx.field(static=True)
    redistribution_allowed: bool = eqx.field(static=True)
    attribution: str = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        spdx_id: str,
        license_sha256: str,
        /,
        *,
        status: LicenseStatus,
        inference_allowed: bool,
        adaptation_allowed: bool,
        redistribution_allowed: bool,
        attribution: str = "",
    ):
        if status not in ("verified", "unknown", "restricted"):
            raise ValueError("Unknown license status.")
        identifier = str(spdx_id).strip()
        if not identifier:
            raise ValueError("spdx_id must be non-empty.")
        digest = _sha256(license_sha256, "license_sha256")
        payload = {
            "spdx_id": identifier,
            "license_sha256": digest,
            "status": status,
            "inference_allowed": bool(inference_allowed),
            "adaptation_allowed": bool(adaptation_allowed),
            "redistribution_allowed": bool(redistribution_allowed),
            "attribution": str(attribution),
        }
        self.spdx_id = identifier
        self.license_sha256 = digest
        self.status = status
        self.inference_allowed = bool(inference_allowed)
        self.adaptation_allowed = bool(adaptation_allowed)
        self.redistribution_allowed = bool(redistribution_allowed)
        self.attribution = str(attribution)
        self.fingerprint = canonical_fingerprint(payload)


class PretrainingOverlapProvenance(StrictModule):
    """Tri-state homology-aware pretraining/evaluation overlap evidence."""

    assessment: OverlapAssessment = eqx.field(static=True)
    evaluation_split_id: str = eqx.field(static=True)
    homology_partition_id: str = eqx.field(static=True)
    search_method: str = eqx.field(static=True)
    identity_threshold: float = eqx.field(static=True)
    maximum_identity: float | None = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        assessment: OverlapAssessment,
        *,
        evaluation_split_id: str,
        homology_partition_id: str,
        search_method: str,
        identity_threshold: float,
        maximum_identity: float | None = None,
    ):
        if assessment not in ("unknown", "no-detected-overlap", "known-overlap"):
            raise ValueError("Unknown pretraining overlap assessment.")
        split_id = str(evaluation_split_id).strip()
        partition_id = str(homology_partition_id).strip()
        method = str(search_method).strip()
        threshold = float(identity_threshold)
        maximum = None if maximum_identity is None else float(maximum_identity)
        if not split_id or not partition_id:
            raise ValueError(
                "Evaluation split and homology partition identities are required."
            )
        if not math.isfinite(threshold) or threshold < 0.0 or threshold > 1.0:
            raise ValueError("identity_threshold must lie in [0, 1].")
        if assessment == "unknown":
            if maximum is not None:
                raise ValueError("Unknown overlap cannot claim a maximum identity.")
        else:
            if not method:
                raise ValueError("Known overlap assessments require a search method.")
            if (
                maximum is None
                or not math.isfinite(maximum)
                or maximum < 0.0
                or maximum > 1.0
            ):
                raise ValueError(
                    "Known overlap assessments require maximum_identity in [0, 1]."
                )
            detected = maximum >= threshold
            if detected != (assessment == "known-overlap"):
                raise ValueError(
                    "Overlap assessment contradicts maximum identity and threshold."
                )
        payload = {
            "assessment": assessment,
            "evaluation_split_id": split_id,
            "homology_partition_id": partition_id,
            "search_method": method,
            "identity_threshold": threshold,
            "maximum_identity": maximum,
        }
        self.assessment = assessment
        self.evaluation_split_id = split_id
        self.homology_partition_id = partition_id
        self.search_method = method
        self.identity_threshold = threshold
        self.maximum_identity = maximum
        self.fingerprint = canonical_fingerprint(payload)


class FoundationModelManifest(StrictModule):
    """Complete immutable identity and training provenance for one model artifact."""

    model_id: str = eqx.field(static=True)
    architecture_id: str = eqx.field(static=True)
    artifact_sha256: str = eqx.field(static=True)
    parameter_sha256: str = eqx.field(static=True)
    structure_fingerprint: str = eqx.field(static=True)
    base_model_sha256: str | None = eqx.field(static=True)
    tokenizer: TokenizerProvenance = eqx.field(static=True)
    license: LicenseProvenance = eqx.field(static=True)
    pretraining_overlap: PretrainingOverlapProvenance = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        model_id: str,
        architecture_id: str,
        artifact_sha256: str,
        parameter_sha256: str,
        structure_fingerprint: str,
        tokenizer: TokenizerProvenance,
        license: LicenseProvenance,
        pretraining_overlap: PretrainingOverlapProvenance,
        *,
        base_model_sha256: str | None = None,
    ):
        model_id_ = str(model_id).strip()
        architecture_id_ = str(architecture_id).strip()
        structure = str(structure_fingerprint).strip()
        if not model_id_ or not architecture_id_ or not structure:
            raise ValueError(
                "Model, architecture, and structure identities are required."
            )
        artifact = _sha256(artifact_sha256, "artifact_sha256")
        parameters = _sha256(parameter_sha256, "parameter_sha256")
        base = (
            None
            if base_model_sha256 is None
            else _sha256(base_model_sha256, "base_model_sha256")
        )
        payload = {
            "model_id": model_id_,
            "architecture_id": architecture_id_,
            "artifact_sha256": artifact,
            "parameter_sha256": parameters,
            "structure_fingerprint": structure,
            "base_model_sha256": base,
            "tokenizer_fingerprint": tokenizer.fingerprint,
            "license_fingerprint": license.fingerprint,
            "pretraining_overlap_fingerprint": pretraining_overlap.fingerprint,
        }
        self.model_id = model_id_
        self.architecture_id = architecture_id_
        self.artifact_sha256 = artifact
        self.parameter_sha256 = parameters
        self.structure_fingerprint = structure
        self.base_model_sha256 = base
        self.tokenizer = tokenizer
        self.license = license
        self.pretraining_overlap = pretraining_overlap
        self.fingerprint = canonical_fingerprint(payload)


class NativeArtifactBinding(StrictModule):
    """Verified static binding evidence attached to one pure native callable."""

    manifest: FoundationModelManifest = eqx.field(static=True)
    valid: Array
    status: Array
    evidence: Array
    method_contract: BioinformaticsMethodContract = eqx.field(static=True)

    def __init__(self, manifest: FoundationModelManifest):
        known_overlap = manifest.pretraining_overlap.assessment != "unknown"
        self.manifest = manifest
        self.valid = jnp.asarray(True)
        self.status = jnp.asarray(
            FoundationBindingStatus.SUCCESS
            if known_overlap
            else FoundationBindingStatus.PRETRAINING_OVERLAP_UNKNOWN,
            dtype=jnp.int32,
        )
        self.evidence = jnp.asarray((1, 1, int(known_overlap)), dtype=jnp.int32)
        self.method_contract = _BINDING_CONTRACT


def _call_native_model(model: Any, *args: Any, **kwargs: Any) -> Any:
    return model(*args, **kwargs)


class BoundNativeFoundationModel(StrictModule):
    """Export-ready pure native callable plus exact immutable artifact binding."""

    model: Any
    binding: NativeArtifactBinding
    adapter_fingerprint: str | None = eqx.field(static=True, default=None)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)

    def export_callable(self) -> Any:
        """Return a hash-safe pure callable whose model arrays remain PyTree leaves."""
        return jax.tree_util.Partial(_call_native_model, self.model)


def low_rank_adapter_parameter_sha256(adapted_model: Any, /) -> str:
    """Hash low-rank factor values and their exact structural binding sites."""

    nodes = jax.tree_util.tree_flatten_with_path(
        adapted_model,
        is_leaf=lambda value: isinstance(value, LowRankUpdate),
    )[0]
    updates = tuple(
        (jax.tree_util.keystr(path), value)
        for path, value in nodes
        if isinstance(value, LowRankUpdate)
    )
    if not updates:
        raise ValueError("Adapted model contains no LowRankUpdate leaves.")
    return canonical_fingerprint(
        {
            "kind": "low-rank-adapter-parameters",
            "updates": tuple(
                {
                    "path": path,
                    "factor_sha256": array_tree_fingerprint((update.left, update.right))[
                        "sha256"
                    ],
                    "alpha": update.alpha,
                    "scaling": update.scaling,
                }
                for path, update in updates
            ),
        }
    )


class LowRankAdapterProvenance(StrictModule):
    """Adapter artifact identity bound to one exact native base artifact."""

    adapter_id: str = eqx.field(static=True)
    adapter_sha256: str = eqx.field(static=True)
    adapter_parameter_sha256: str = eqx.field(static=True)
    base_artifact_sha256: str = eqx.field(static=True)
    base_parameter_sha256: str = eqx.field(static=True)
    rank: int = eqx.field(static=True)
    target_paths: tuple[str, ...] = eqx.field(static=True)
    fingerprint: str = eqx.field(static=True)

    def __init__(
        self,
        adapter_id: str,
        adapter_sha256: str,
        adapter_parameter_sha256: str,
        base_artifact_sha256: str,
        base_parameter_sha256: str,
        *,
        rank: int,
        target_paths: tuple[str, ...],
    ):
        identifier = str(adapter_id).strip()
        rank_ = int(rank)
        paths = tuple(str(path) for path in target_paths)
        if not identifier or rank_ <= 0 or not paths or any(not path for path in paths):
            raise ValueError(
                "Adapter identity, positive rank, and target paths are required."
            )
        if len(set(paths)) != len(paths):
            raise ValueError("Low-rank adapter target paths must be unique.")
        adapter_hash = _sha256(adapter_sha256, "adapter_sha256")
        adapter_parameters = _sha256(adapter_parameter_sha256, "adapter_parameter_sha256")
        base_artifact = _sha256(base_artifact_sha256, "base_artifact_sha256")
        base_parameters = _sha256(base_parameter_sha256, "base_parameter_sha256")
        payload = {
            "adapter_id": identifier,
            "adapter_sha256": adapter_hash,
            "adapter_parameter_sha256": adapter_parameters,
            "base_artifact_sha256": base_artifact,
            "base_parameter_sha256": base_parameters,
            "rank": rank_,
            "target_paths": paths,
        }
        self.adapter_id = identifier
        self.adapter_sha256 = adapter_hash
        self.adapter_parameter_sha256 = adapter_parameters
        self.base_artifact_sha256 = base_artifact
        self.base_parameter_sha256 = base_parameters
        self.rank = rank_
        self.target_paths = paths
        self.fingerprint = canonical_fingerprint(payload)


def bind_native_foundation_model(
    model: Any,
    manifest: FoundationModelManifest,
    /,
    *,
    artifact_sha256: str,
    tokenizer_fingerprint: str,
    alphabet_fingerprint: str,
    evaluation_split_id: str,
    homology_partition_id: str,
    base_model_sha256: str | None = None,
) -> BoundNativeFoundationModel:
    """Bind only after exact model/artifact/tokenizer/license/split verification."""
    if not isinstance(manifest, FoundationModelManifest):
        raise TypeError("manifest must be a FoundationModelManifest.")
    if _sha256(artifact_sha256, "artifact_sha256") != manifest.artifact_sha256:
        raise ValueError("Foundation artifact hash does not match its manifest.")
    if native_model_parameter_sha256(model) != manifest.parameter_sha256:
        raise ValueError("Native model parameter hash does not match its manifest.")
    if native_model_structure_fingerprint(model) != manifest.structure_fingerprint:
        raise ValueError("Native model structure does not match its manifest.")
    if str(tokenizer_fingerprint) != manifest.tokenizer.fingerprint:
        raise ValueError("Tokenizer identity does not match the foundation model.")
    if str(alphabet_fingerprint) != manifest.tokenizer.alphabet_fingerprint:
        raise ValueError("Alphabet identity does not match the foundation model.")
    if manifest.license.status != "verified" or not manifest.license.inference_allowed:
        raise PermissionError("Foundation model license is unknown or forbids inference.")
    overlap = manifest.pretraining_overlap
    if str(evaluation_split_id) != overlap.evaluation_split_id:
        raise ValueError("Evaluation split identity differs from overlap provenance.")
    if str(homology_partition_id) != overlap.homology_partition_id:
        raise ValueError("Homology partition identity differs from overlap provenance.")
    supplied_base = (
        None
        if base_model_sha256 is None
        else _sha256(base_model_sha256, "base_model_sha256")
    )
    if supplied_base != manifest.base_model_sha256:
        raise ValueError("Base model identity does not match the foundation manifest.")
    return BoundNativeFoundationModel(
        model=model, binding=NativeArtifactBinding(manifest)
    )


def bind_low_rank_foundation_adapter(
    adapted_model: Any,
    base: BoundNativeFoundationModel,
    provenance: LowRankAdapterProvenance,
    /,
    *,
    adapter_sha256: str,
) -> BoundNativeFoundationModel:
    """Bind an already-native low-rank model only to its exact verified base."""
    if not isinstance(base, BoundNativeFoundationModel):
        raise TypeError("base must be a BoundNativeFoundationModel.")
    if not isinstance(provenance, LowRankAdapterProvenance):
        raise TypeError("provenance must be LowRankAdapterProvenance.")
    if not base.binding.manifest.license.adaptation_allowed:
        raise PermissionError("Foundation model license forbids adaptation.")
    if _sha256(adapter_sha256, "adapter_sha256") != provenance.adapter_sha256:
        raise ValueError("Low-rank adapter artifact hash mismatch.")
    manifest = base.binding.manifest
    if manifest.artifact_sha256 != provenance.base_artifact_sha256:
        raise ValueError("Low-rank adapter base artifact mismatch.")
    if manifest.parameter_sha256 != provenance.base_parameter_sha256:
        raise ValueError("Low-rank adapter base parameter mismatch.")
    nodes = jax.tree_util.tree_flatten_with_path(
        adapted_model,
        is_leaf=lambda value: isinstance(value, LowRankUpdate),
    )[0]
    updates = tuple(
        (jax.tree_util.keystr(path), value)
        for path, value in nodes
        if isinstance(value, LowRankUpdate)
    )
    sites = tuple((path, update.rank) for path, update in updates)
    if tuple(path for path, _ in sites) != provenance.target_paths:
        raise ValueError("Low-rank adapter target paths do not match the adapted model.")
    if any(rank != provenance.rank for _, rank in sites):
        raise ValueError("Low-rank adapter rank does not match the adapted model.")
    if (
        low_rank_adapter_parameter_sha256(adapted_model)
        != provenance.adapter_parameter_sha256
    ):
        raise ValueError("Low-rank adapter parameter hash mismatch.")
    base_arrays = {
        jax.tree_util.keystr(path): value
        for path, value in jax.tree_util.tree_flatten_with_path(base.model)[0]
        if eqx.is_array(value)
    }
    for path, update in updates:
        if path not in base_arrays:
            raise ValueError(
                "Low-rank adapter target is absent from the bound base model."
            )
        if (
            array_tree_fingerprint(update.base)["sha256"]
            != array_tree_fingerprint(base_arrays[path])["sha256"]
        ):
            raise ValueError("Low-rank adapter embeds a different base parameter.")
    return BoundNativeFoundationModel(
        model=adapted_model,
        binding=base.binding,
        adapter_fingerprint=provenance.fingerprint,
    )


@dataclass(frozen=True, slots=True)
class ExternalFoundationRuntime:
    """Host-only external runtime; never a JAX or Equinox PyTree."""

    runtime: Any
    manifest: FoundationModelManifest

    def run_host(self, *args: Any, **kwargs: Any) -> Any:
        """Execute the explicitly host-only runtime without export claims."""
        return self.runtime(*args, **kwargs)


__all__ = [
    "BoundNativeFoundationModel",
    "ExternalFoundationRuntime",
    "FoundationBindingStatus",
    "FoundationModelManifest",
    "LicenseProvenance",
    "LowRankAdapterProvenance",
    "NativeArtifactBinding",
    "PretrainingOverlapProvenance",
    "TokenizerProvenance",
    "bind_low_rank_foundation_adapter",
    "bind_native_foundation_model",
    "native_model_parameter_sha256",
    "low_rank_adapter_parameter_sha256",
    "native_model_structure_fingerprint",
    "sha256_file",
]
