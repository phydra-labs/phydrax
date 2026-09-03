#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Fail-closed in-process implementation of the REMOTE-01 service boundary."""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import threading
from dataclasses import dataclass, field, replace
from typing import Mapping, Protocol
from uuid import uuid4

from phydrax.lifecycle import CheckpointManifest, RunRecord

from ._auth import AccessTokenValidator, Clock, ResourceAuthorizer, SystemClock
from ._contracts import (
    ArtifactDescriptor,
    ArtifactExpired,
    AuditRecord,
    AuthorizationError,
    CADArtifactMetadata,
    CADEgressPolicy,
    CancellationRequested,
    EncryptionMetadata,
    FailureEvidence,
    FetchedArtifact,
    IntegrityError,
    InvalidTransition,
    JobState,
    JobStatus,
    JobSubmission,
    ProfileUnavailable,
    ProviderResult,
    QuotaExceeded,
    ResourceNotFound,
    SignedArtifactGrant,
    TenantQuota,
    TenantUsage,
    ValidatedPrincipal,
)


class ExecutionContext(Protocol):
    """Callbacks made available to an execution provider."""

    @property
    def job_id(self) -> str: ...

    def cancellation_point(self) -> None: ...

    def checkpoint(self, manifest: CheckpointManifest, /) -> str: ...


class ExecutionProvider(Protocol):
    """A synchronous provider invoked for a submitted execution profile."""

    def __call__(
        self, submission: JobSubmission, context: ExecutionContext, /
    ) -> ProviderResult: ...


@dataclass(slots=True)
class _Artifact:
    descriptor: ArtifactDescriptor
    content: bytes


@dataclass(slots=True)
class _Job:
    job_id: str
    tenant_id: str
    submission: JobSubmission
    state: JobState
    attempt: int
    submitted_at: int
    expires_at: int
    run_record: RunRecord
    prior_run_records: list[RunRecord] = field(default_factory=list)
    checkpoint_ids: list[str] = field(default_factory=list)
    recovered_checkpoint_id: str | None = None
    artifact_ids: list[str] = field(default_factory=list)
    started_at: int | None = None
    finished_at: int | None = None
    cancel_requested_at: int | None = None
    failure: FailureEvidence | None = None


class _ProviderContext:
    def __init__(self, service: InProcessReferenceService, job: _Job):
        self._service = service
        self._job = job

    @property
    def job_id(self) -> str:
        return self._job.job_id

    def cancellation_point(self) -> None:
        if self._job.state is JobState.CANCELLING:
            raise CancellationRequested("The job was cancelled.")

    def checkpoint(self, manifest: CheckpointManifest, /) -> str:
        return self._service._record_checkpoint(self._job, manifest)


class InProcessReferenceService:
    """Thread-safe, synchronous reference service with tenant-isolated state.

    Provider callbacks run only through :meth:`execute`; callers never choose a
    provider by untrusted profile metadata.  Every external operation validates a
    bearer token before looking up tenant resources.
    """

    def __init__(
        self,
        token_validator: AccessTokenValidator,
        authorizer: ResourceAuthorizer,
        tenant_quotas: Mapping[str, TenantQuota],
        /,
        *,
        clock: Clock | None = None,
        artifact_signing_secret: bytes | None = None,
        encryption: EncryptionMetadata | None = None,
        cad_egress_policies: Mapping[str, CADEgressPolicy] | None = None,
    ):
        if not tenant_quotas:
            raise ValueError("At least one tenant quota is required.")
        if any(
            not tenant_id or not isinstance(quota, TenantQuota)
            for tenant_id, quota in tenant_quotas.items()
        ):
            raise ValueError("Tenant quota configuration is invalid.")
        secret = (
            secrets.token_bytes(32)
            if artifact_signing_secret is None
            else artifact_signing_secret
        )
        if len(secret) < 32:
            raise ValueError("Artifact signing secret must contain at least 256 bits.")
        self._validator = token_validator
        self._authorizer = authorizer
        self._quotas = dict(tenant_quotas)
        self._clock = SystemClock() if clock is None else clock
        self._artifact_secret = bytes(secret)
        self._encryption = encryption or EncryptionMetadata(
            "AES-256-GCM", "reference", True, "TLS", 0
        )
        self._cad_policies = dict(cad_egress_policies or {})
        self._providers: dict[str, ExecutionProvider] = {}
        self._jobs: dict[str, _Job] = {}
        self._artifacts: dict[str, _Artifact] = {}
        self._checkpoints: dict[str, CheckpointManifest] = {}
        self._audit: list[AuditRecord] = []
        self._lock = threading.RLock()

    def register_provider(self, profile_id: str, provider: ExecutionProvider, /) -> None:
        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or not callable(provider)
        ):
            raise ValueError("Provider profile and callback must be valid.")
        with self._lock:
            if profile_id in self._providers:
                raise ValueError("A provider is already registered for this profile.")
            self._providers[profile_id] = provider

    def submit(self, token: str, submission: JobSubmission, /) -> JobStatus:
        principal = self._authenticate(token)
        self._authorize(principal, "service:submit", principal.tenant_id)
        if not isinstance(submission, JobSubmission):
            raise TypeError("submission must be a JobSubmission.")
        with self._lock:
            self._require_profile(submission.profile_id)
            if any(
                handle.tenant_id != principal.tenant_id
                for handle in submission.secret_handles
            ):
                self._deny(
                    principal,
                    "submit",
                    "job",
                    submission.request_id or "new",
                    "secret tenant mismatch",
                )
                raise AuthorizationError(
                    "Secret handles must belong to the submitting tenant."
                )
            self._reserve(principal.tenant_id, submission)
            now = self._clock.now()
            job_id = uuid4().hex
            job = _Job(
                job_id,
                principal.tenant_id,
                submission,
                JobState.QUEUED,
                1,
                now,
                now + submission.retention_seconds,
                self._run_record(job_id, submission, "queued"),
            )
            self._jobs[job_id] = job
            self._audit_event(
                principal,
                "submit",
                "job",
                job_id,
                "allowed",
                "queued",
                submission.request_id,
            )
            return self._status(job)

    def status(self, token: str, job_id: str, /) -> JobStatus:
        principal = self._authenticate(token)
        with self._lock:
            job = self._job(job_id)
            self._authorize(principal, "service:status", job.tenant_id)
            self._expire_job(job)
            self._audit_event(principal, "status", "job", job_id, "allowed", "read", "")
            return self._status(job)

    def cancel(self, token: str, job_id: str, /) -> JobStatus:
        principal = self._authenticate(token)
        with self._lock:
            job = self._job(job_id)
            self._authorize(principal, "service:cancel", job.tenant_id)
            self._expire_job(job)
            if job.state.terminal:
                raise InvalidTransition("A terminal job cannot be cancelled.")
            if job.state is JobState.QUEUED:
                job.state = JobState.CANCELLED
                job.finished_at = self._clock.now()
                job.run_record = self._run_record(
                    job_id,
                    job.submission,
                    "cancelled",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                )
            else:
                job.state = JobState.CANCELLING
                job.cancel_requested_at = self._clock.now()
            self._audit_event(
                principal, "cancel", "job", job_id, "allowed", job.state.value, ""
            )
            return self._status(job)

    def restart(self, token: str, job_id: str, /) -> JobStatus:
        principal = self._authenticate(token)
        with self._lock:
            job = self._job(job_id)
            self._authorize(principal, "service:restart", job.tenant_id)
            self._expire_job(job)
            if not job.state.terminal:
                raise InvalidTransition("Only a terminal job can be restarted.")
            self._reserve(job.tenant_id, job.submission)
            job.prior_run_records.append(job.run_record)
            job.attempt += 1
            job.state = JobState.QUEUED
            job.started_at = job.finished_at = job.cancel_requested_at = None
            job.failure = None
            job.recovered_checkpoint_id = (
                job.checkpoint_ids[-1] if job.checkpoint_ids else None
            )
            job.run_record = self._run_record(
                job_id, job.submission, "queued", job.recovered_checkpoint_id
            )
            self._audit_event(
                principal, "restart", "job", job_id, "allowed", "queued", ""
            )
            return self._status(job)

    def execute(self, token: str, job_id: str, /) -> JobStatus:
        """Execute one queued job synchronously using its registered provider."""
        principal = self._authenticate(token)
        with self._lock:
            job = self._job(job_id)
            self._authorize(principal, "service:execute", job.tenant_id)
            self._expire_job(job)
            if job.state is not JobState.QUEUED:
                raise InvalidTransition("Only a queued job can be executed.")
            provider = self._require_profile(job.submission.profile_id)
            job.state = JobState.RUNNING
            job.started_at = self._clock.now()
            job.run_record = self._run_record(
                job_id, job.submission, "running", job.recovered_checkpoint_id
            )
        try:
            result = provider(job.submission, _ProviderContext(self, job))
            if not isinstance(result, ProviderResult):
                raise IntegrityError("Provider must return a ProviderResult.")
            with self._lock:
                if job.state is JobState.CANCELLING:
                    raise CancellationRequested("The job was cancelled.")
                job.state = JobState.SUCCEEDED
                job.finished_at = self._clock.now()
                job.run_record = self._run_record(
                    job_id,
                    job.submission,
                    "completed",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                    result,
                )
                self._audit_event(
                    principal, "execute", "job", job_id, "allowed", "completed", ""
                )
        except CancellationRequested:
            with self._lock:
                job.state = JobState.CANCELLED
                job.finished_at = self._clock.now()
                job.run_record = self._run_record(
                    job_id,
                    job.submission,
                    "cancelled",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                )
                self._audit_event(
                    principal, "execute", "job", job_id, "allowed", "cancelled", ""
                )
        except Exception:
            with self._lock:
                job.state = JobState.FAILED
                job.finished_at = self._clock.now()
                job.failure = FailureEvidence(
                    "provider_failure",
                    "ProviderExecutionError",
                    "Provider execution failed.",
                    False,
                    job.attempt,
                )
                job.run_record = self._run_record(
                    job_id,
                    job.submission,
                    "failed",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                )
                self._audit_event(
                    principal, "execute", "job", job_id, "failed", "provider failure", ""
                )
        with self._lock:
            return self._status(job)

    def store_artifact(
        self,
        token: str,
        job_id: str,
        content: bytes,
        /,
        *,
        scientific_artifact_id: str,
        media_type: str,
        classification: str = "scientific",
        cad: CADArtifactMetadata | None = None,
    ) -> ArtifactDescriptor:
        principal = self._authenticate(token)
        if classification not in {
            "scientific",
            "cad",
            "checkpoint",
            "diagnostic",
            "support",
        }:
            raise IntegrityError("Artifact classification is not accepted.")
        if not isinstance(content, bytes) or not scientific_artifact_id or not media_type:
            raise IntegrityError("Artifact content and metadata are invalid.")
        with self._lock:
            job = self._job(job_id)
            self._authorize(principal, "service:artifact:write", job.tenant_id)
            self._expire_job(job)
            quota = self._quotas[job.tenant_id]
            if (
                self._usage(job.tenant_id).retained_artifact_bytes + len(content)
                > quota.retained_artifact_bytes
            ):
                raise QuotaExceeded("Tenant retained artifact quota would be exceeded.")
            now = self._clock.now()
            artifact_id = uuid4().hex
            descriptor = ArtifactDescriptor(
                artifact_id,
                scientific_artifact_id,
                job_id,
                job.tenant_id,
                hashlib.sha256(content).hexdigest(),
                len(content),
                media_type,
                classification,
                now,
                job.expires_at,
                uuid4().hex,
                self._encryption,
                cad,
            )
            self._artifacts[artifact_id] = _Artifact(descriptor, bytes(content))
            job.artifact_ids.append(artifact_id)
            self._audit_event(
                principal,
                "artifact.write",
                "artifact",
                artifact_id,
                "allowed",
                "stored",
                "",
            )
            return descriptor

    def grant_artifact(
        self, token: str, artifact_id: str, /, *, lifetime_seconds: int = 300
    ) -> SignedArtifactGrant:
        principal = self._authenticate(token)
        if lifetime_seconds <= 0:
            raise ValueError("Grant lifetime must be positive.")
        with self._lock:
            artifact = self._artifact(artifact_id)
            self._authorize(
                principal, "service:artifact:grant", artifact.descriptor.tenant_id
            )
            self._assert_artifact_live(artifact)
            if artifact.descriptor.classification == "cad":
                self._cad_policies.get(
                    artifact.descriptor.tenant_id, CADEgressPolicy.deny_all()
                ).authorize(artifact.descriptor.cad)  # type: ignore[arg-type]
            expires_at = min(
                self._clock.now() + lifetime_seconds, artifact.descriptor.expires_at
            )
            token_value = self._grant_token(
                artifact_id, artifact.descriptor.tenant_id, expires_at
            )
            self._audit_event(
                principal,
                "artifact.grant",
                "artifact",
                artifact_id,
                "allowed",
                "granted",
                "",
            )
            return SignedArtifactGrant(
                token_value, artifact_id, artifact.descriptor.tenant_id, expires_at
            )

    def fetch_artifact(
        self, token: str, grant: SignedArtifactGrant | str, /
    ) -> FetchedArtifact:
        principal = self._authenticate(token)
        value = grant.token if isinstance(grant, SignedArtifactGrant) else grant
        artifact_id, tenant_id, expires_at = self._verify_grant(value)
        with self._lock:
            artifact = self._artifact(artifact_id)
            self._authorize(principal, "service:artifact:fetch", tenant_id)
            if (
                artifact.descriptor.tenant_id != tenant_id
                or self._clock.now() >= expires_at
            ):
                raise ArtifactExpired("Artifact grant has expired.")
            self._assert_artifact_live(artifact)
            if not hmac.compare_digest(
                hashlib.sha256(artifact.content).hexdigest(),
                artifact.descriptor.content_sha256,
            ):
                raise IntegrityError(
                    "Artifact content digest does not match its descriptor."
                )
            self._audit_event(
                principal,
                "artifact.fetch",
                "artifact",
                artifact_id,
                "allowed",
                "fetched",
                "",
            )
            return FetchedArtifact(artifact.descriptor, artifact.content)

    def delete_expired(self) -> tuple[str, ...]:
        """Purge expired artifacts and terminal job records; return deleted IDs."""
        with self._lock:
            now = self._clock.now()
            deleted = []
            for artifact_id, artifact in tuple(self._artifacts.items()):
                if artifact.descriptor.expires_at <= now:
                    del self._artifacts[artifact_id]
                    deleted.append(artifact_id)
            for job_id, job in tuple(self._jobs.items()):
                self._expire_job(job)
                if job.state.terminal and job.expires_at <= now:
                    for checkpoint_id in job.checkpoint_ids:
                        del self._checkpoints[checkpoint_id]
                        deleted.append(checkpoint_id)
                    del self._jobs[job_id]
                    deleted.append(job_id)
            return tuple(deleted)

    def audit_records(self, token: str, tenant_id: str, /) -> tuple[AuditRecord, ...]:
        principal = self._authenticate(token)
        self._authorize(principal, "service:audit:read", tenant_id)
        with self._lock:
            return tuple(
                record for record in self._audit if record.tenant_id == tenant_id
            )

    def verify_audit_chain(self) -> None:
        previous = "0" * 64
        with self._lock:
            for sequence, record in enumerate(self._audit, 1):
                if record.sequence != sequence or record.previous_digest != previous:
                    raise IntegrityError("Audit chain ordering is invalid.")
                expected = self._audit_digest(record)
                if not hmac.compare_digest(record.record_digest, expected):
                    raise IntegrityError("Audit chain digest is invalid.")
                previous = record.record_digest

    def usage(self, token: str, /) -> TenantUsage:
        principal = self._authenticate(token)
        self._authorize(principal, "service:usage", principal.tenant_id)
        with self._lock:
            return self._usage(principal.tenant_id)

    def _authenticate(self, token: str) -> ValidatedPrincipal:
        try:
            return self._validator.validate(token)
        except Exception as error:
            # Authentication adapters are untrusted integration boundaries.
            from ._contracts import AuthenticationError

            if isinstance(error, AuthenticationError):
                raise
            raise AuthenticationError("Access token validation failed.") from error

    def _authorize(
        self, principal: ValidatedPrincipal, scope: str, tenant_id: str
    ) -> None:
        try:
            self._authorizer.authorize(principal, scope, tenant_id)
        except AuthorizationError:
            raise
        except Exception as error:
            raise AuthorizationError("Authorization policy evaluation failed.") from error

    def _require_profile(self, profile_id: str) -> ExecutionProvider:
        provider = self._providers.get(profile_id)
        if provider is None:
            raise ProfileUnavailable(
                "No provider is registered for this execution profile."
            )
        return provider

    def _job(self, job_id: str) -> _Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise ResourceNotFound("Job does not exist.")
        return job

    def _artifact(self, artifact_id: str) -> _Artifact:
        artifact = self._artifacts.get(artifact_id)
        if artifact is None:
            raise ResourceNotFound("Artifact does not exist.")
        return artifact

    def _reserve(self, tenant_id: str, submission: JobSubmission) -> None:
        usage = self._usage(tenant_id)
        quota = self._quotas[tenant_id]
        requested = submission.resources
        if (
            usage.active_jobs + 1 > quota.active_jobs
            or usage.cpu_cores + requested.cpu_cores > quota.cpu_cores
            or usage.memory_bytes + requested.memory_bytes > quota.memory_bytes
            or usage.gpu_count + requested.gpu_count > quota.gpu_count
        ):
            raise QuotaExceeded("Tenant active execution quota would be exceeded.")

    def _usage(self, tenant_id: str) -> TenantUsage:
        jobs = [
            job
            for job in self._jobs.values()
            if job.tenant_id == tenant_id and not job.state.terminal
        ]
        return TenantUsage(
            len(jobs),
            sum(job.submission.resources.cpu_cores for job in jobs),
            sum(job.submission.resources.memory_bytes for job in jobs),
            sum(job.submission.resources.gpu_count for job in jobs),
            sum(
                artifact.descriptor.byte_size
                for artifact in self._artifacts.values()
                if artifact.descriptor.tenant_id == tenant_id
            ),
        )

    def _record_checkpoint(self, job: _Job, manifest: CheckpointManifest) -> str:
        with self._lock:
            if job.state not in (JobState.RUNNING, JobState.CANCELLING):
                raise InvalidTransition(
                    "Checkpoints can only be recorded during execution."
                )
            submission = job.submission
            if (
                not manifest.complete
                or manifest.analysis_plan_id != submission.analysis_plan.analysis_plan_id
                or manifest.numeric_revision_id != submission.numeric_revision_id
                or manifest.execution_plan_id
                != submission.execution_plan.execution_plan_id
            ):
                raise IntegrityError("Checkpoint manifest does not match this execution.")
            if (
                manifest.parent_checkpoint_id
                and manifest.parent_checkpoint_id not in job.checkpoint_ids
            ):
                raise IntegrityError("Checkpoint parent is not owned by this job.")
            if manifest.checkpoint_id in self._checkpoints:
                raise IntegrityError("Checkpoint identifier already exists.")
            self._checkpoints[manifest.checkpoint_id] = manifest
            job.checkpoint_ids.append(manifest.checkpoint_id)
            return manifest.checkpoint_id

    def _expire_job(self, job: _Job) -> None:
        if self._clock.now() >= job.expires_at and not job.state.terminal:
            job.state = JobState.CANCELLED
            job.finished_at = self._clock.now()
            job.run_record = self._run_record(
                job.job_id,
                job.submission,
                "cancelled",
                job.checkpoint_ids[-1] if job.checkpoint_ids else None,
            )

    def _assert_artifact_live(self, artifact: _Artifact) -> None:
        if self._clock.now() >= artifact.descriptor.expires_at:
            raise ArtifactExpired("Artifact retention period has expired.")

    def _run_record(
        self,
        job_id: str,
        submission: JobSubmission,
        status: str,
        checkpoint_id: str | None = None,
        result: ProviderResult | None = None,
    ) -> RunRecord:
        return RunRecord(
            f"{job_id}:{uuid4().hex}",
            submission.analysis_plan.analysis_plan_id,
            submission.numeric_revision_id,
            submission.execution_plan.execution_plan_id,
            status,
            result_ids=() if result is None else result.result_ids,
            diagnostic_ids=() if result is None else result.diagnostic_ids,
            checkpoint_id=checkpoint_id,
        )

    def _status(self, job: _Job) -> JobStatus:
        return JobStatus(
            job.job_id,
            job.tenant_id,
            job.state,
            job.attempt,
            job.submitted_at,
            job.started_at,
            job.finished_at,
            job.cancel_requested_at,
            job.expires_at,
            job.run_record,
            tuple(job.prior_run_records),
            tuple(job.checkpoint_ids),
            job.recovered_checkpoint_id,
            tuple(job.artifact_ids),
            job.failure,
        )

    def _grant_token(self, artifact_id: str, tenant_id: str, expires_at: int) -> str:
        payload = json.dumps(
            {
                "artifact_id": artifact_id,
                "expires_at": expires_at,
                "tenant_id": tenant_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return (
            payload.hex()
            + "."
            + hmac.new(self._artifact_secret, payload, hashlib.sha256).hexdigest()
        )

    def _verify_grant(self, value: str) -> tuple[str, str, int]:
        try:
            encoded, signature = value.split(".", 1)
            payload = bytes.fromhex(encoded)
            expected = hmac.new(
                self._artifact_secret, payload, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise IntegrityError("Artifact grant signature is invalid.")
            decoded = json.loads(payload.decode("utf-8"))
            artifact_id, tenant_id, expires_at = (
                decoded["artifact_id"],
                decoded["tenant_id"],
                decoded["expires_at"],
            )
            if (
                not isinstance(artifact_id, str)
                or not isinstance(tenant_id, str)
                or isinstance(expires_at, bool)
                or not isinstance(expires_at, int)
            ):
                raise IntegrityError("Artifact grant payload is invalid.")
            return artifact_id, tenant_id, expires_at
        except (
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise IntegrityError("Artifact grant is malformed.") from error

    def _audit_event(
        self,
        principal: ValidatedPrincipal,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: str,
        reason: str,
        request_id: str,
    ) -> None:
        previous = self._audit[-1].record_digest if self._audit else "0" * 64
        record = AuditRecord(
            len(self._audit) + 1,
            self._clock.now(),
            uuid4().hex,
            principal.subject,
            principal.tenant_id,
            action,
            resource_type,
            resource_id,
            outcome,
            reason,
            request_id,
            previous,
            "",
        )
        self._audit.append(replace(record, record_digest=self._audit_digest(record)))

    def _deny(
        self,
        principal: ValidatedPrincipal,
        action: str,
        resource_type: str,
        resource_id: str,
        reason: str,
    ) -> None:
        self._audit_event(
            principal, action, resource_type, resource_id, "denied", reason, ""
        )

    @staticmethod
    def _audit_digest(record: AuditRecord) -> str:
        payload = {
            name: getattr(record, name)
            for name in (
                "sequence",
                "occurred_at",
                "event_id",
                "principal_id",
                "tenant_id",
                "action",
                "resource_type",
                "resource_id",
                "outcome",
                "reason",
                "request_id",
                "previous_digest",
            )
        }
        return hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()


__all__ = ["ExecutionContext", "ExecutionProvider", "InProcessReferenceService"]
