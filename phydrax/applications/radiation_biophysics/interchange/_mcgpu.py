#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned, artifact-import-only MCGPU RAW/config score profiles."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...._fingerprint import canonical_fingerprint
from ....interchange import (
    AdapterFormatProfile,
    AdapterLoss,
    AdapterReport,
    AdapterStatus,
    AdapterWaiver,
    BoundedResource,
)
from ....qualification import ReferenceArtifactManifest
from .._scores import (
    _identifier,
    _positive_integer,
    ExternalRadiationRunIdentity,
    ExternalRadiationScoreResult,
    radiation_score_content_id,
    RadiationEstimatorEvidence,
    RadiationScoreDefinition,
    require_import_rights,
    require_profile_semantics,
)


_BASE_SEMANTICS = (
    "build-identity",
    "calibration-identity",
    "configuration-identity",
    "grid-affine",
    "score-dtype",
    "score-normalization",
    "score-quantity",
    "score-shape",
    "seed-lineage",
    "table-identity",
    "units",
)


def _permutation(values: tuple[int, ...], rank: int, /) -> tuple[int, ...]:
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError("storage_to_score_axes must contain integers.")
    if len(result) != rank or set(result) != set(range(rank)):
        raise ValueError("storage_to_score_axes must be a complete axis permutation.")
    return result


def _decode_raw(resource: BoundedResource, profile: MCGPURawProfile, /) -> np.ndarray:
    dtype = np.dtype(profile.score.dtype)
    expected = int(np.prod(profile.storage_shape)) * dtype.itemsize
    if len(resource.data) != expected:
        raise ValueError("MCGPU RAW byte length differs from the exact profile.")
    storage = np.frombuffer(resource.data, dtype=dtype).reshape(profile.storage_shape)
    return np.transpose(storage, profile.storage_to_score_axes)


def _source_id(references: tuple[ReferenceArtifactManifest, ...], /) -> str:
    return canonical_fingerprint(
        {
            "kind": "external-radiation-artifact-set",
            "manifests": [reference.manifest_id for reference in references],
        }
    )


@dataclass(frozen=True, slots=True)
class MCGPURawProfile:
    """One exact MCGPU raw storage order tied to one governed configuration."""

    run_identity: ExternalRadiationRunIdentity
    score: RadiationScoreDefinition
    storage_shape: tuple[int, ...]
    storage_to_score_axes: tuple[int, ...]
    required_config_lines: tuple[str, ...]
    uncertainty_kind: str
    estimator: str
    history_count: int
    batch_count: int
    correlation_model: str
    uncertainty_raw: bool = False
    declared_losses: tuple[AdapterLoss, ...] = ()
    profile_id: str = field(init=False)
    available_semantics: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.run_identity, ExternalRadiationRunIdentity):
            raise TypeError("run_identity must be ExternalRadiationRunIdentity.")
        if self.run_identity.engine != "MCGPU":
            raise ValueError("MCGPU raw profiles require an MCGPU run identity.")
        if not isinstance(self.score, RadiationScoreDefinition):
            raise TypeError("score must be RadiationScoreDefinition.")
        storage_shape = tuple(
            _positive_integer(value, "storage shape entry")
            for value in self.storage_shape
        )
        if len(storage_shape) != len(self.score.shape):
            raise ValueError("MCGPU storage rank must match score rank.")
        permutation = _permutation(self.storage_to_score_axes, len(storage_shape))
        if tuple(storage_shape[index] for index in permutation) != self.score.shape:
            raise ValueError(
                "MCGPU storage shape/permutation do not produce score shape."
            )
        lines = tuple(
            _identifier(value, "required MCGPU config line")
            for value in self.required_config_lines
        )
        if not lines or len(set(lines)) != len(lines):
            raise ValueError("MCGPU profile requires unique exact configuration lines.")
        if not isinstance(self.uncertainty_raw, bool):
            raise TypeError("uncertainty_raw must be boolean.")
        uncertainty_kind = _identifier(self.uncertainty_kind, "uncertainty_kind")
        correlation = _identifier(self.correlation_model, "correlation_model")
        if self.uncertainty_raw != (uncertainty_kind != "unreported"):
            raise ValueError("MCGPU uncertainty RAW declaration and kind must agree.")
        if not self.uncertainty_raw and correlation != "unreported":
            raise ValueError(
                "Absent MCGPU uncertainty cannot claim correlation evidence."
            )
        if self.uncertainty_raw and correlation in (
            "unreported",
            "assumed-independent",
        ):
            raise ValueError("MCGPU uncertainty requires upstream correlation evidence.")
        histories = _positive_integer(self.history_count, "history_count")
        batches = _positive_integer(self.batch_count, "batch_count")
        losses = tuple(self.declared_losses)
        if any(not isinstance(loss, AdapterLoss) for loss in losses):
            raise TypeError("declared_losses must contain AdapterLoss values.")
        available = _BASE_SEMANTICS + ("estimator", "configuration-lines")
        if self.uncertainty_raw:
            available += ("uncertainty", "correlation")
        object.__setattr__(self, "storage_shape", storage_shape)
        object.__setattr__(self, "storage_to_score_axes", permutation)
        object.__setattr__(self, "required_config_lines", lines)
        object.__setattr__(self, "uncertainty_kind", uncertainty_kind)
        object.__setattr__(self, "estimator", _identifier(self.estimator, "estimator"))
        object.__setattr__(self, "history_count", histories)
        object.__setattr__(self, "batch_count", batches)
        object.__setattr__(self, "correlation_model", correlation)
        object.__setattr__(self, "declared_losses", losses)
        object.__setattr__(self, "available_semantics", tuple(sorted(set(available))))
        object.__setattr__(
            self,
            "profile_id",
            canonical_fingerprint(
                {
                    "kind": "mcgpu-raw-score-profile",
                    "run": self.run_identity.identity_id,
                    "score": self.score.definition_id,
                    "storage_shape": list(storage_shape),
                    "storage_to_score_axes": list(permutation),
                    "required_config_lines": list(lines),
                    "uncertainty_kind": uncertainty_kind,
                    "uncertainty_raw": self.uncertainty_raw,
                    "estimator": self.estimator,
                    "history_count": histories,
                    "batch_count": batches,
                    "correlation_model": correlation,
                    "declared_losses": [loss.loss_id for loss in losses],
                }
            ),
        )


def import_mcgpu_raw(
    raw_resource: BoundedResource,
    raw_reference: ReferenceArtifactManifest,
    config_resource: BoundedResource,
    config_reference: ReferenceArtifactManifest,
    profile: MCGPURawProfile,
    /,
    *,
    uncertainty_resource: BoundedResource | None = None,
    uncertainty_reference: ReferenceArtifactManifest | None = None,
    required_semantics: tuple[str, ...] = (),
    commercial_use: bool = False,
    redistribution: bool = False,
    training_use: bool = False,
    export: bool = False,
) -> ExternalRadiationScoreResult:
    """Import exact raw/config artifacts; never launch or wrap MCGPU."""

    if not isinstance(raw_resource, BoundedResource) or not isinstance(
        config_resource, BoundedResource
    ):
        raise TypeError("raw_resource and config_resource must be BoundedResource.")
    if not isinstance(raw_reference, ReferenceArtifactManifest) or not isinstance(
        config_reference, ReferenceArtifactManifest
    ):
        raise TypeError("RAW and config references must be ReferenceArtifactManifest.")
    if not isinstance(profile, MCGPURawProfile):
        raise TypeError("profile must be MCGPURawProfile.")
    if config_reference.manifest_id != profile.run_identity.configuration.manifest_id:
        raise ValueError("MCGPU config artifact is not the run's pinned configuration.")
    if (uncertainty_resource is None) != (uncertainty_reference is None):
        raise ValueError(
            "MCGPU uncertainty resource and reference must be supplied together."
        )
    if profile.uncertainty_raw != (uncertainty_resource is not None):
        raise ValueError("MCGPU uncertainty artifacts differ from the exact profile.")
    if uncertainty_resource is not None and not isinstance(
        uncertainty_resource, BoundedResource
    ):
        raise TypeError("uncertainty_resource must be BoundedResource.")
    if uncertainty_reference is not None and not isinstance(
        uncertainty_reference, ReferenceArtifactManifest
    ):
        raise TypeError("uncertainty_reference must be ReferenceArtifactManifest.")
    require_profile_semantics(required_semantics, profile.available_semantics)
    raw_reference.verify_bytes(raw_resource.data)
    config_reference.verify_bytes(config_resource.data)
    if uncertainty_resource is not None and uncertainty_reference is not None:
        uncertainty_reference.verify_bytes(uncertainty_resource.data)
    artifacts = (
        raw_reference,
        config_reference,
        *(() if uncertainty_reference is None else (uncertainty_reference,)),
    )
    require_import_rights(
        artifacts,
        profile.run_identity,
        commercial_use=commercial_use,
        redistribution=redistribution,
        training_use=training_use,
        export=export,
    )
    config_text = config_resource.data.decode("utf-8", errors="strict")
    config_lines = frozenset(
        line.strip() for line in config_text.splitlines() if line.strip()
    )
    missing_lines = tuple(
        line for line in profile.required_config_lines if line not in config_lines
    )
    if missing_lines:
        raise ValueError(f"MCGPU configuration omits pinned lines: {list(missing_lines)}")
    values = _decode_raw(raw_resource, profile)
    uncertainty = (
        None
        if uncertainty_resource is None
        else _decode_raw(uncertainty_resource, profile)
    )
    evidence_source = (
        raw_reference.manifest_id
        if uncertainty_reference is None
        else uncertainty_reference.manifest_id
    )
    evidence = RadiationEstimatorEvidence(
        profile.estimator,
        profile.history_count,
        profile.batch_count,
        profile.uncertainty_kind,
        uncertainty,
        profile.correlation_model,
        (evidence_source,),
    )
    score_id = radiation_score_content_id(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        artifacts,
        profile.profile_id,
    )
    status = (
        AdapterStatus.DECLARED_LOSS if profile.declared_losses else AdapterStatus.LOSSLESS
    )
    report = AdapterReport(
        status,
        "MCGPU-RAW+config",
        "ExternalRadiationScoreResult",
        source_id=_source_id(artifacts),
        target_id=score_id,
        source_profile=AdapterFormatProfile(
            "MCGPU-RAW+config",
            qualifiers={
                "external_profile_id": profile.profile_id,
                "provider_revision": profile.run_identity.engine_revision,
                "score_quantity": profile.score.quantity_kind.value,
            },
        ),
        coordinate_mapping=(
            f"storage-to-score axis permutation {profile.storage_to_score_axes}",
            f"grid affine {profile.score.grid_affine.affine_id}",
        ),
        preserved_fields=(
            "score values",
            "score dtype and byte order",
            "score normalization",
            "quantity and unit",
            "grid affine",
            "exact configuration bytes and required lines",
            "build/configuration/table/calibration/seed identities",
            "native estimator and correlation declaration",
        ),
        assumptions=(
            "caller-supplied profile metadata is authoritative for semantics absent from RAW storage",
            "artifact import does not qualify or execute MCGPU",
            "result scope is research-only",
        ),
        losses=profile.declared_losses,
        waivers=tuple(
            AdapterWaiver(
                loss,
                "The exact research-only profile explicitly retains this declared source loss.",
            )
            for loss in profile.declared_losses
            if loss.changes_interpretation
        ),
    )
    return ExternalRadiationScoreResult(
        values,
        profile.score,
        profile.run_identity,
        evidence,
        artifacts,
        profile.profile_id,
        report,
    )


__all__ = ["MCGPURawProfile", "import_mcgpu_raw"]
