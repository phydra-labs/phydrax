#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Literal, Mapping, TYPE_CHECKING, TypeAlias
from urllib.parse import urlparse

from .._execution_resources import ResourceRequest


if TYPE_CHECKING:
    from phydrax.execution import ExecutionPlan
    from phydrax.lifecycle import (
        AnalysisPlan,
        ResolvedRunSpec,
        RunRecord,
    )


JSONScalar: TypeAlias = str | int | float | bool | None
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]
ArtifactClassification: TypeAlias = Literal[
    "scientific", "cad", "checkpoint", "diagnostic", "support"
]


class RemoteServiceError(RuntimeError):
    """Base error for the REMOTE-01 service boundary."""


class AuthenticationError(RemoteServiceError):
    """The access token could not be authenticated."""


class AuthorizationError(RemoteServiceError):
    """The principal is not authorized for the requested tenant resource."""


class QuotaExceeded(RemoteServiceError):
    """A tenant resource or retained-storage quota would be exceeded."""


class ResourceNotFound(RemoteServiceError):
    """The tenant-scoped resource does not exist."""


class InvalidTransition(RemoteServiceError):
    """The requested job lifecycle transition is not valid."""


class ArtifactExpired(RemoteServiceError):
    """The retained artifact or signed grant has expired."""


class IntegrityError(RemoteServiceError):
    """Signed or content-addressed evidence failed integrity validation."""


class ProfileUnavailable(RemoteServiceError):
    """No execution provider is registered for the requested lifecycle backend."""


class CancellationRequested(RemoteServiceError):
    """Cooperative execution stopped at a service cancellation boundary."""


class JobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELING = "canceling"
    SUCCEEDED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

    @property
    def terminal(self) -> bool:
        return self in (self.SUCCEEDED, self.FAILED, self.CANCELED)


@dataclass(frozen=True, slots=True)
class TenantQuota:
    active_jobs: int
    cpu_cores: int
    memory_bytes: int
    gpu_count: int
    retained_artifact_bytes: int

    def __post_init__(self) -> None:
        values = (
            self.active_jobs,
            self.cpu_cores,
            self.memory_bytes,
            self.gpu_count,
            self.retained_artifact_bytes,
        )
        if any(value < 0 for value in values):
            raise ValueError("Tenant quota values must be nonnegative.")
        if self.active_jobs == 0 or self.cpu_cores == 0 or self.memory_bytes == 0:
            raise ValueError("Active-job, CPU, and memory quotas must be positive.")


@dataclass(frozen=True, slots=True)
class TenantUsage:
    active_jobs: int
    cpu_cores: int
    memory_bytes: int
    gpu_count: int
    retained_artifact_bytes: int


@dataclass(frozen=True, slots=True)
class ValidatedPrincipal:
    subject: str
    tenant_id: str
    issuer: str
    audience: str
    client_id: str
    token_id: str
    scopes: frozenset[str]
    issued_at: int
    expires_at: int

    def __post_init__(self) -> None:
        values = (
            self.subject,
            self.tenant_id,
            self.issuer,
            self.audience,
            self.client_id,
            self.token_id,
        )
        if any(not value.strip() for value in values):
            raise ValueError("Validated principal identifiers must be nonempty.")
        if not self.scopes or any(not value.strip() for value in self.scopes):
            raise ValueError("Validated principal scopes must be nonempty.")
        if self.expires_at <= self.issued_at:
            raise ValueError("Validated principal expiry must follow issuance.")


@dataclass(frozen=True, slots=True)
class SecretHandle:
    handle_id: str
    tenant_id: str
    created_at: int
    key_version: str

    def __post_init__(self) -> None:
        if not self.handle_id or not self.tenant_id or not self.key_version:
            raise ValueError("Secret handle metadata must be nonempty.")
        if self.created_at < 0:
            raise ValueError("Secret handle creation time must be nonnegative.")


@dataclass(frozen=True, slots=True)
class EncryptionMetadata:
    algorithm: str
    key_id: str
    encrypted_at_rest: bool
    transport_protocol: str
    key_rotated_at: int

    def __post_init__(self) -> None:
        if not self.algorithm or not self.key_id or not self.transport_protocol:
            raise ValueError("Encryption metadata values must be nonempty.")
        if self.key_rotated_at < 0:
            raise ValueError("key_rotated_at must be nonnegative.")


@dataclass(frozen=True, slots=True)
class ArtifactRights:
    scientific_artifact_id: str
    rights_id: str
    use_policy_id: str
    license_id: str
    source_uri: str
    attribution_id: str
    content_sha256: str
    byte_size: int
    classification: ArtifactClassification
    allow_redistribution: bool
    allow_export: bool
    redistribution_requested: bool
    export_requested: bool
    rights_binding_id: str = field(init=False)

    def __post_init__(self) -> None:
        identifiers = (
            self.scientific_artifact_id,
            self.rights_id,
            self.use_policy_id,
            self.license_id,
            self.source_uri,
            self.attribution_id,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identifiers):
            raise ValueError("Artifact rights identities must be nonempty strings.")
        source = urlparse(self.source_uri)
        if not source.scheme or not (source.netloc or source.path):
            raise ValueError("Artifact rights source_uri must be an absolute URI.")
        if len(self.content_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.content_sha256
        ):
            raise ValueError(
                "Artifact rights content_sha256 must be a lowercase SHA-256 digest."
            )
        if isinstance(self.byte_size, bool) or not isinstance(self.byte_size, int):
            raise TypeError("Artifact rights byte_size must be an integer.")
        if self.byte_size < 0:
            raise ValueError("Artifact rights byte_size must be nonnegative.")
        if self.classification not in {
            "scientific",
            "cad",
            "checkpoint",
            "diagnostic",
            "support",
        }:
            raise ValueError("Artifact rights classification is not accepted.")
        permissions = (
            self.allow_redistribution,
            self.allow_export,
            self.redistribution_requested,
            self.export_requested,
        )
        if any(type(value) is not bool for value in permissions):
            raise TypeError("Artifact rights permissions must be booleans.")
        payload = {
            "allow_export": self.allow_export,
            "allow_redistribution": self.allow_redistribution,
            "attribution_id": self.attribution_id,
            "byte_size": self.byte_size,
            "classification": self.classification,
            "content_sha256": self.content_sha256,
            "export_requested": self.export_requested,
            "license_id": self.license_id,
            "redistribution_requested": self.redistribution_requested,
            "rights_id": self.rights_id,
            "scientific_artifact_id": self.scientific_artifact_id,
            "source_uri": self.source_uri,
            "use_policy_id": self.use_policy_id,
        }
        object.__setattr__(
            self,
            "rights_binding_id",
            hashlib.sha256(
                json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        )

    @classmethod
    def unrestricted(
        cls,
        scientific_artifact_id: str,
        content_sha256: str,
        byte_size: int,
        /,
        *,
        rights_id: str,
        use_policy_id: str,
        license_id: str,
        source_uri: str,
        attribution_id: str,
        classification: ArtifactClassification = "scientific",
    ) -> ArtifactRights:
        return cls(
            scientific_artifact_id,
            rights_id,
            use_policy_id,
            license_id,
            source_uri,
            attribution_id,
            content_sha256,
            byte_size,
            classification,
            True,
            True,
            True,
            True,
        )

    def require_egress(self) -> None:
        if not (
            self.allow_redistribution
            and self.allow_export
            and self.redistribution_requested
            and self.export_requested
        ):
            raise AuthorizationError(
                "Artifact rights or requested-use policy denies redistribution or export."
            )


@dataclass(frozen=True, slots=True)
class CADArtifactMetadata:
    format: str
    destination_region: str
    export_classification: str = "unclassified"
    approval_id: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.format
            or not self.destination_region
            or not self.export_classification
        ):
            raise ValueError("CAD artifact metadata values must be nonempty.")
        if self.approval_id is not None and not self.approval_id.strip():
            raise ValueError("CAD approval_id must be nonempty when provided.")


@dataclass(frozen=True, slots=True)
class CADEgressPolicy:
    policy_id: str
    allow_download: bool
    allowed_formats: frozenset[str]
    allowed_destination_regions: frozenset[str]
    require_approval: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("CAD egress policy_id must be nonempty.")
        if self.allow_download and (
            not self.allowed_formats or not self.allowed_destination_regions
        ):
            raise ValueError("Enabled CAD egress requires formats and regions.")
        if any(not value for value in self.allowed_formats):
            raise ValueError("CAD egress formats must be nonempty.")
        if any(not value for value in self.allowed_destination_regions):
            raise ValueError("CAD egress regions must be nonempty.")

    @classmethod
    def deny_all(cls) -> CADEgressPolicy:
        return cls(
            policy_id="cad-egress-deny-all",
            allow_download=False,
            allowed_formats=frozenset(),
            allowed_destination_regions=frozenset(),
        )

    def authorize(self, metadata: CADArtifactMetadata, /) -> None:
        allowed = (
            self.allow_download
            and metadata.format in self.allowed_formats
            and metadata.destination_region in self.allowed_destination_regions
            and (not self.require_approval or metadata.approval_id is not None)
        )
        if not allowed:
            raise AuthorizationError("CAD artifact egress is denied by tenant policy.")


def _canonical_parameters(parameters: Mapping[str, Any]) -> Mapping[str, JSONValue]:
    encoded = json.dumps(
        dict(parameters),
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):
        raise TypeError("Job parameters must serialize to a JSON object.")
    return MappingProxyType(decoded)


@dataclass(frozen=True, slots=True)
class JobSubmission:
    analysis_plan: AnalysisPlan
    execution_plan: ExecutionPlan
    numeric_revision_id: str
    profile_id: str
    parameters: Mapping[str, JSONValue]
    resources: ResourceRequest
    secret_handles: tuple[SecretHandle, ...] = ()
    retention_seconds: int = 86_400
    request_id: str = ""
    resolved_run_spec: ResolvedRunSpec | None = None
    request_digest: str = field(init=False)

    def __post_init__(self) -> None:
        analysis_plan_id = self.analysis_plan.analysis_plan_id
        execution_plan_id = self.execution_plan.execution_plan_id
        if (
            not isinstance(analysis_plan_id, str)
            or not analysis_plan_id
            or not isinstance(execution_plan_id, str)
            or not execution_plan_id
        ):
            raise ValueError("Lifecycle plan identifiers must be nonempty strings.")
        if (
            not isinstance(self.numeric_revision_id, str)
            or not isinstance(self.profile_id, str)
            or not self.numeric_revision_id
            or not self.profile_id
        ):
            raise ValueError("Revision and profile identifiers must be nonempty strings.")
        if type(self.retention_seconds) is not int:
            raise TypeError("retention_seconds must be an integer.")
        if self.retention_seconds <= 0:
            raise ValueError("retention_seconds must be positive.")
        handles = tuple(self.secret_handles)
        if len({handle.handle_id for handle in handles}) != len(handles):
            raise ValueError("Secret handles must be unique within a submission.")
        resolved_run_spec = self.resolved_run_spec
        if resolved_run_spec is not None:
            if not hasattr(resolved_run_spec, "spec_id"):
                raise TypeError("resolved_run_spec must be a ResolvedRunSpec.")
            if self.profile_id not in resolved_run_spec.profile_ids:
                raise ValueError(
                    "Submission profile must be exactly bound by resolved_run_spec."
                )
        parameters = _canonical_parameters(self.parameters)
        request_id = self.request_id.strip()
        digest_payload = {
            "analysis_plan_id": analysis_plan_id,
            "analysis_plan_fingerprint": self.analysis_plan.plan_fingerprint,
            "execution_plan_id": execution_plan_id,
            "execution_plan_fingerprint": self.execution_plan.plan_fingerprint,
            "numeric_revision_id": self.numeric_revision_id,
            "profile_id": self.profile_id,
            "parameters": dict(parameters),
            "resources": self.resources.to_payload(),
            "secret_handles": [
                {
                    "created_at": handle.created_at,
                    "expires_at": getattr(handle, "expires_at", None),
                    "handle_id": handle.handle_id,
                    "key_version": handle.key_version,
                    "scopes": sorted(getattr(handle, "scopes", ())),
                    "tenant_id": handle.tenant_id,
                }
                for handle in handles
            ],
            "retention_seconds": self.retention_seconds,
            "resolved_run_spec_id": (
                None if resolved_run_spec is None else resolved_run_spec.spec_id
            ),
        }
        digest = hashlib.sha256(
            json.dumps(
                digest_payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        object.__setattr__(self, "parameters", parameters)
        object.__setattr__(self, "secret_handles", handles)
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "request_digest", digest)


@dataclass(frozen=True, slots=True)
class ProviderResult:
    result_ids: tuple[str, ...]
    diagnostic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        results = tuple(self.result_ids)
        diagnostics = tuple(self.diagnostic_ids)
        if any(not value for value in (*results, *diagnostics)):
            raise ValueError("Provider result identifiers must be nonempty.")
        if len(set(results)) != len(results) or len(set(diagnostics)) != len(diagnostics):
            raise ValueError("Provider result identifiers must be unique.")
        object.__setattr__(self, "result_ids", results)
        object.__setattr__(self, "diagnostic_ids", diagnostics)


@dataclass(frozen=True, slots=True)
class FailureEvidence:
    code: str
    exception_type: str
    message: str
    retryable: bool
    attempt: int
    diagnostic_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.code or not self.exception_type or not self.message:
            raise ValueError("Failure evidence values must be nonempty.")
        if type(self.retryable) is not bool:
            raise TypeError("Failure retryable must be a boolean.")
        if type(self.attempt) is not int or self.attempt <= 0:
            raise ValueError("Failure evidence attempt must be a positive integer.")
        diagnostics = tuple(self.diagnostic_ids)
        if len(set(diagnostics)) != len(diagnostics) or any(
            not isinstance(value, str) or not value for value in diagnostics
        ):
            raise ValueError("Failure diagnostic identifiers must be unique strings.")
        object.__setattr__(self, "diagnostic_ids", diagnostics)


@dataclass(frozen=True, slots=True)
class JobStatus:
    job_id: str
    tenant_id: str
    state: JobState
    attempt: int
    submitted_at: int
    started_at: int | None
    finished_at: int | None
    cancel_requested_at: int | None
    expires_at: int
    run_record: RunRecord
    prior_run_records: tuple[RunRecord, ...]
    checkpoint_ids: tuple[str, ...]
    recovered_checkpoint_id: str | None
    artifact_ids: tuple[str, ...]
    failure: FailureEvidence | None


@dataclass(frozen=True, slots=True)
class ArtifactDescriptor:
    artifact_id: str
    scientific_artifact_id: str
    job_id: str
    tenant_id: str
    content_sha256: str
    byte_size: int
    media_type: str
    classification: ArtifactClassification
    created_at: int
    expires_at: int
    storage_generation: str
    encryption: EncryptionMetadata
    rights: ArtifactRights
    cad: CADArtifactMetadata | None = None

    def __post_init__(self) -> None:
        values = (
            self.artifact_id,
            self.scientific_artifact_id,
            self.job_id,
            self.tenant_id,
            self.media_type,
            self.storage_generation,
        )
        if any(not value for value in values):
            raise ValueError("Artifact descriptor identifiers must be nonempty.")
        if len(self.content_sha256) != 64 or any(
            value not in "0123456789abcdef" for value in self.content_sha256
        ):
            raise ValueError(
                "Artifact content_sha256 must be a lowercase SHA-256 digest."
            )
        if self.byte_size < 0 or self.expires_at <= self.created_at:
            raise ValueError("Artifact size or retention interval is invalid.")
        if not isinstance(self.rights, ArtifactRights):
            raise TypeError("Artifact descriptor requires immutable artifact rights.")
        if (
            self.rights.scientific_artifact_id != self.scientific_artifact_id
            or self.rights.content_sha256 != self.content_sha256
            or self.rights.byte_size != self.byte_size
            or self.rights.classification != self.classification
        ):
            raise ValueError(
                "Artifact descriptor identity differs from its rights binding."
            )
        if self.classification == "cad" and self.cad is None:
            raise ValueError("CAD artifacts require CAD metadata.")
        if self.classification != "cad" and self.cad is not None:
            raise ValueError("CAD metadata is only valid for CAD artifacts.")


@dataclass(frozen=True, slots=True)
class SignedArtifactGrant:
    token: str
    artifact_id: str
    tenant_id: str
    expires_at: int
    rights: ArtifactRights

    def __post_init__(self) -> None:
        if (
            not self.token
            or not self.artifact_id
            or not self.tenant_id
            or isinstance(self.expires_at, bool)
            or not isinstance(self.expires_at, int)
        ):
            raise ValueError("Signed artifact grant fields are invalid.")
        if not isinstance(self.rights, ArtifactRights):
            raise TypeError("Signed artifact grants require immutable artifact rights.")


@dataclass(frozen=True, slots=True)
class FetchedArtifact:
    descriptor: ArtifactDescriptor
    content: bytes


@dataclass(frozen=True, slots=True)
class AuditRecord:
    sequence: int
    occurred_at: int
    event_id: str
    principal_id: str
    tenant_id: str
    action: str
    resource_type: str
    resource_id: str
    outcome: Literal["allowed", "denied", "failed"]
    reason: str
    request_id: str
    previous_digest: str
    record_digest: str


__all__ = [
    "ArtifactClassification",
    "ArtifactDescriptor",
    "ArtifactRights",
    "ArtifactExpired",
    "AuditRecord",
    "AuthenticationError",
    "AuthorizationError",
    "CADEgressPolicy",
    "CADArtifactMetadata",
    "CancellationRequested",
    "EncryptionMetadata",
    "FailureEvidence",
    "FetchedArtifact",
    "IntegrityError",
    "InvalidTransition",
    "JobState",
    "JobStatus",
    "JobSubmission",
    "ProfileUnavailable",
    "ProviderResult",
    "QuotaExceeded",
    "RemoteServiceError",
    "ResourceNotFound",
    "SecretHandle",
    "SignedArtifactGrant",
    "TenantQuota",
    "TenantUsage",
    "ValidatedPrincipal",
]
