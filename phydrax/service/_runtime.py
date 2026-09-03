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
from typing import Mapping, Protocol, TYPE_CHECKING
from uuid import uuid4

from phydrax.lifecycle import CheckpointManifest, RunRecord
from phydrax.qualification._evidence import SupportDependency

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
from ._durability import DurableJobRecord, DurableServiceStore, OutboxMessage
from ._observability import SecretRedactor


if TYPE_CHECKING:
    pass


class SupportDependencyAdmitter(Protocol):
    """Fail-closed exact release-evidence admission boundary."""

    def require(self, dependency: SupportDependency, /, *, at_time: int) -> None: ...


class ReleaseIndexDependencyAdmitter:
    """Adapter from exact SupportDependency records to release-index admission."""

    def __init__(
        self,
        release_index: object,
        trust_policy: object,
        support_tuples: Mapping[str, object],
        /,
    ):
        if not support_tuples:
            raise ValueError("Dependency admission requires exact support tuples.")
        normalized: dict[str, object] = {}
        for tuple_id, support_tuple in support_tuples.items():
            if getattr(support_tuple, "support_tuple_id", None) != tuple_id:
                raise ValueError(
                    "Support tuple mapping key must equal its content-addressed ID."
                )
            normalized[tuple_id] = support_tuple
        self._release_index = release_index
        self._trust_policy = trust_policy
        self._support_tuples = normalized

    def require(self, dependency: SupportDependency, /, *, at_time: int) -> None:
        from phydrax.qualification._registry import require_profile

        support_tuple = self._support_tuples.get(dependency.support_tuple_id)
        if support_tuple is None:
            raise ProfileUnavailable(
                "Resolved support tuple is not present in the admission catalog."
            )
        try:
            admitted = require_profile(
                self._release_index,
                dependency.profile_id,
                support_tuple,
                self._trust_policy,
                at_time=at_time,
            )
        except Exception as error:
            raise ProfileUnavailable(
                "Resolved support dependency is not release-admissible."
            ) from error
        if (
            admitted.profile_id != dependency.profile_id
            or support_tuple.support_tuple_id != dependency.support_tuple_id
        ):
            raise ProfileUnavailable(
                "Release admission did not preserve the exact dependency identity."
            )


class ExecutionContext(Protocol):
    """Callbacks made available to an execution provider."""

    @property
    def job_id(self) -> str: ...

    def cancellation_point(self) -> None: ...
    def heartbeat(self) -> None: ...

    def checkpoint(self, manifest: CheckpointManifest, /) -> str: ...


class ExecutionProvider(Protocol):
    """A synchronous provider invoked for a submitted execution profile."""

    def __call__(
        self, submission: JobSubmission, context: ExecutionContext, /
    ) -> ProviderResult: ...


@dataclass(frozen=True, slots=True)
class ProviderBinding:
    provider: ExecutionProvider
    support_tuple_id: str


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


class _ExecutionSuperseded(RuntimeError):
    """Raised when an expired execution attempt has been durably replaced."""


class _ProviderContext:
    def __init__(
        self,
        service: InProcessReferenceService,
        job: _Job,
        attempt: int,
        durable_version: int | None,
    ):
        self._service = service
        self._job = job
        self._attempt = int(attempt)
        self._durable_version = durable_version

    @property
    def job_id(self) -> str:
        return self._job.job_id

    def cancellation_point(self) -> None:
        with self._service._lock:
            if self._job.attempt != self._attempt or self._job.state not in (
                JobState.RUNNING,
                JobState.CANCELLING,
            ):
                raise _ExecutionSuperseded(
                    "Execution attempt has been replaced or is no longer active."
                )
            if self._job.state is JobState.CANCELLING:
                raise CancellationRequested("The job was cancelled.")
            self._service._require_execution_fence(
                self._job, self._attempt, self._durable_version
            )

    def heartbeat(self) -> None:
        self._durable_version = self._service._heartbeat_execution(
            self._job, self._attempt, self._durable_version
        )

    def checkpoint(self, manifest: CheckpointManifest, /) -> str:
        self.heartbeat()
        return self._service._record_checkpoint(
            self._job, manifest, expected_attempt=self._attempt
        )


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
        dependency_admitter: SupportDependencyAdmitter | None = None,
        durable_store: DurableServiceStore | None = None,
        repository: object | None = None,
        scheduler: object | None = None,
        scheduler_id: str | None = None,
        auth_policy_id: str | None = None,
        execution_lease_seconds: int = 300,
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
        if execution_lease_seconds <= 0:
            raise ValueError("Execution lease duration must be positive.")
        repository_id = (
            None if repository is None else getattr(repository, "provider_id", None)
        )
        scheduler_provider_id = (
            None if scheduler is None else getattr(scheduler, "provider_id", None)
        )
        if (
            scheduler_id is not None
            and scheduler_provider_id is not None
            and scheduler_id != scheduler_provider_id
        ):
            raise ValueError("scheduler_id conflicts with the bound scheduler provider.")
        scheduler_id = scheduler_provider_id if scheduler_id is None else scheduler_id
        if auth_policy_id is None:
            auth_policy_id = getattr(
                token_validator,
                "policy_id",
                getattr(authorizer, "policy_id", None),
            )
        for value, name in (
            (repository_id, "repository provider_id"),
            (scheduler_id, "scheduler provider_id"),
            (auth_policy_id, "auth_policy_id"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be a nonempty string when configured.")
        self._validator = token_validator
        self._authorizer = authorizer
        self._quotas = dict(tenant_quotas)
        self._clock = SystemClock() if clock is None else clock
        self._artifact_secret = bytes(secret)
        self._encryption = encryption or EncryptionMetadata(
            "AES-256-GCM", "reference", True, "TLS", 0
        )
        self._cad_policies = dict(cad_egress_policies or {})
        self._dependency_admitter = dependency_admitter
        self._durable_store = durable_store
        self._repository_id = repository_id
        self._scheduler_id = scheduler_id
        self._auth_policy_id = auth_policy_id
        self._execution_lease_seconds = execution_lease_seconds
        self._providers: dict[str, ProviderBinding] = {}
        self._jobs: dict[tuple[str, str], _Job] = {}
        self._requests: dict[tuple[str, str], tuple[str, str]] = {}
        self._artifacts: dict[tuple[str, str], _Artifact] = {}
        self._checkpoints: dict[str, CheckpointManifest] = {}
        self._audit: list[AuditRecord] = []
        self._lock = threading.RLock()

    def register_provider(
        self,
        profile_id: str,
        provider: ExecutionProvider,
        /,
        *,
        support_tuple_id: str | None = None,
    ) -> None:
        tuple_id = (
            getattr(provider, "support_tuple_id", profile_id)
            if support_tuple_id is None
            else support_tuple_id
        )
        if (
            not isinstance(profile_id, str)
            or not profile_id.strip()
            or not isinstance(tuple_id, str)
            or not tuple_id.strip()
            or not callable(provider)
        ):
            raise ValueError(
                "Provider profile, tuple identity, and callback must be valid."
            )
        with self._lock:
            if profile_id in self._providers:
                raise ValueError("A provider is already registered for this profile.")
            self._providers[profile_id] = ProviderBinding(provider, tuple_id)

    def submit(self, token: str, submission: JobSubmission, /) -> JobStatus:
        principal = self._authenticate(token)
        self._authorize(principal, "service:submit", principal.tenant_id)
        if not isinstance(submission, JobSubmission):
            raise TypeError("submission must be a JobSubmission.")
        with self._lock:
            request_key = (principal.tenant_id, submission.request_id)
            if submission.request_id and request_key in self._requests:
                job_id, digest = self._requests[request_key]
                if digest != submission.request_digest:
                    raise IntegrityError(
                        "A request ID was reused for a different submission."
                    )
                return self._status(self._jobs[(principal.tenant_id, job_id)])
            binding = self._require_profile(submission.profile_id)
            self._admit_submission(submission, binding)
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
            self._persist_new_job(job)
            self._jobs[(job.tenant_id, job_id)] = job
            if submission.request_id:
                self._requests[request_key] = (job_id, submission.request_digest)
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
        self._authorize(principal, "service:status", principal.tenant_id)
        with self._lock:
            job = self._job(principal.tenant_id, job_id)
            self._expire_job(job)
            self._audit_event(principal, "status", "job", job_id, "allowed", "read", "")
            return self._status(job)

    def cancel(self, token: str, job_id: str, /) -> JobStatus:
        principal = self._authenticate(token)
        self._authorize(principal, "service:cancel", principal.tenant_id)
        with self._lock:
            job = self._job(principal.tenant_id, job_id)
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
            self._sync_durable_job(job)
            self._audit_event(
                principal, "cancel", "job", job_id, "allowed", job.state.value, ""
            )
            return self._status(job)

    def restart(self, token: str, job_id: str, /) -> JobStatus:
        principal = self._authenticate(token)
        self._authorize(principal, "service:restart", principal.tenant_id)
        with self._lock:
            job = self._job(principal.tenant_id, job_id)
            self._expire_job(job)
            if not job.state.terminal:
                raise InvalidTransition("Only a terminal job can be restarted.")
            binding = self._require_profile(job.submission.profile_id)
            self._admit_submission(job.submission, binding)
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
            self._sync_durable_job(job, enqueue=True, reserve=True)
            self._audit_event(
                principal, "restart", "job", job_id, "allowed", "queued", ""
            )
            return self._status(job)

    def execute(self, token: str, job_id: str, /) -> JobStatus:
        """Execute one queued job synchronously using a fenced renewable attempt."""

        principal = self._authenticate(token)
        self._authorize(principal, "service:execute", principal.tenant_id)
        with self._lock:
            job = self._job(principal.tenant_id, job_id)
            self._expire_job(job)
            if job.state is not JobState.QUEUED:
                raise InvalidTransition("Only a queued job can be executed.")
            binding = self._require_profile(job.submission.profile_id)
            self._admit_submission(job.submission, binding)
            job.state = JobState.RUNNING
            job.started_at = self._clock.now()
            job.run_record = self._run_record(
                job_id, job.submission, "running", job.recovered_checkpoint_id
            )
            execution_attempt = job.attempt
            durable_version = self._sync_durable_job(
                job,
                lease_expires_at=(job.started_at + self._execution_lease_seconds),
            )
            context = _ProviderContext(self, job, execution_attempt, durable_version)
        try:
            result = binding.provider(job.submission, context)
            if not isinstance(result, ProviderResult):
                raise IntegrityError("Provider must return a ProviderResult.")
            with self._lock:
                if job.state is JobState.CANCELLING:
                    raise CancellationRequested("The job was cancelled.")
                self._require_execution_fence(
                    job, execution_attempt, context._durable_version
                )
                finished_at = self._clock.now()
                run_record = self._run_record(
                    job_id,
                    job.submission,
                    "completed",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                    result,
                )
                committed_job = replace(
                    job,
                    state=JobState.SUCCEEDED,
                    finished_at=finished_at,
                    run_record=run_record,
                )
                self._sync_durable_job(
                    committed_job,
                    expected_attempt=execution_attempt,
                    expected_version=context._durable_version,
                )
                job.state = JobState.SUCCEEDED
                job.finished_at = finished_at
                job.run_record = run_record
                self._audit_event(
                    principal,
                    "execute",
                    "job",
                    job_id,
                    "allowed",
                    "completed",
                    "",
                )
        except _ExecutionSuperseded:
            pass
        except CancellationRequested:
            with self._lock:
                if job.attempt != execution_attempt:
                    return self._status(job)
                current_version = self._current_execution_version(job, execution_attempt)
                finished_at = self._clock.now()
                run_record = self._run_record(
                    job_id,
                    job.submission,
                    "cancelled",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                )
                committed_job = replace(
                    job,
                    state=JobState.CANCELLED,
                    finished_at=finished_at,
                    run_record=run_record,
                )
                self._sync_durable_job(
                    committed_job,
                    expected_attempt=execution_attempt,
                    expected_version=current_version,
                )
                job.state = JobState.CANCELLED
                job.finished_at = finished_at
                job.run_record = run_record
                self._audit_event(
                    principal,
                    "execute",
                    "job",
                    job_id,
                    "allowed",
                    "cancelled",
                    "",
                )
        except Exception as error:
            with self._lock:
                try:
                    self._require_execution_fence(
                        job, execution_attempt, context._durable_version
                    )
                except _ExecutionSuperseded:
                    return self._status(job)
                finished_at = self._clock.now()
                redacted_message = SecretRedactor().redact(
                    str(error) or type(error).__name__,
                    field_name="provider_error",
                )
                if not isinstance(redacted_message, str):
                    redacted_message = "<redacted>"
                failure = FailureEvidence(
                    "provider_failure",
                    type(error).__name__,
                    redacted_message,
                    False,
                    execution_attempt,
                )
                run_record = self._run_record(
                    job_id,
                    job.submission,
                    "failed",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                )
                committed_job = replace(
                    job,
                    state=JobState.FAILED,
                    finished_at=finished_at,
                    failure=failure,
                    run_record=run_record,
                )
                self._sync_durable_job(
                    committed_job,
                    expected_attempt=execution_attempt,
                    expected_version=context._durable_version,
                )
                job.state = JobState.FAILED
                job.finished_at = finished_at
                job.failure = failure
                job.run_record = run_record
                self._audit_event(
                    principal,
                    "execute",
                    "job",
                    job_id,
                    "failed",
                    "provider failure",
                    "",
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
        self._authorize(principal, "service:artifact:write", principal.tenant_id)
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
            job = self._job(principal.tenant_id, job_id)
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
            self._artifacts[(job.tenant_id, artifact_id)] = _Artifact(
                descriptor, bytes(content)
            )
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
        self._authorize(principal, "service:artifact:grant", principal.tenant_id)
        if lifetime_seconds <= 0:
            raise ValueError("Grant lifetime must be positive.")
        with self._lock:
            artifact = self._artifact(principal.tenant_id, artifact_id)
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
        self._authorize(principal, "service:artifact:fetch", tenant_id)
        with self._lock:
            artifact = self._artifact(tenant_id, artifact_id)
            if self._clock.now() >= expires_at:
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
            for artifact_key, artifact in tuple(self._artifacts.items()):
                if artifact.descriptor.expires_at <= now:
                    del self._artifacts[artifact_key]
                    deleted.append(artifact.descriptor.artifact_id)
            for job_key, job in tuple(self._jobs.items()):
                self._expire_job(job)
                if job.state.terminal and job.expires_at <= now:
                    for checkpoint_id in job.checkpoint_ids:
                        del self._checkpoints[checkpoint_id]
                        deleted.append(checkpoint_id)
                    del self._jobs[job_key]
                    if job.submission.request_id:
                        self._requests.pop(
                            (job.tenant_id, job.submission.request_id), None
                        )
                    deleted.append(job.job_id)
            return tuple(deleted)

    def audit_records(self, token: str, tenant_id: str, /) -> tuple[AuditRecord, ...]:
        principal = self._authenticate(token)
        self._authorize(principal, "service:audit:read", tenant_id)
        if self._durable_store is not None:
            return self._durable_store.audit_records(tenant_id)
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
        if self._durable_store is not None:
            self._durable_store.verify_audit_chain()

    def usage(self, token: str, /) -> TenantUsage:
        principal = self._authenticate(token)
        self._authorize(principal, "service:usage", principal.tenant_id)
        with self._lock:
            return self._usage(principal.tenant_id)

    def recover_stale_attempts(self) -> tuple[JobStatus, ...]:
        """Recover expired durable execution leases as new idempotent attempts."""
        if self._durable_store is None:
            return ()
        recovered = self._durable_store.recover_stale_attempts(self._clock.now())
        statuses: list[JobStatus] = []
        with self._lock:
            for record in recovered:
                job = self._jobs.get((record.tenant_id, record.job_id))
                if job is None:
                    continue
                job.prior_run_records.append(job.run_record)
                job.state = JobState.QUEUED
                job.attempt = record.attempt
                job.started_at = job.finished_at = job.cancel_requested_at = None
                job.failure = None
                job.recovered_checkpoint_id = (
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None
                )
                job.run_record = self._run_record(
                    job.job_id,
                    job.submission,
                    "queued",
                    job.recovered_checkpoint_id,
                )
                statuses.append(self._status(job))
        return tuple(statuses)

    def reconcile_quotas(self) -> Mapping[str, TenantUsage]:
        """Drop stale durable reservations and return reconciled tenant usage."""
        if self._durable_store is None:
            with self._lock:
                return {
                    tenant_id: self._usage(tenant_id)
                    for tenant_id in sorted(self._quotas)
                }
        with self._lock:
            active_by_tenant = {
                tenant_id: tuple(
                    job.job_id
                    for job in self._jobs.values()
                    if job.tenant_id == tenant_id and not job.state.terminal
                )
                for tenant_id in self._quotas
            }
        return {
            tenant_id: self._durable_store.reconcile_quota(
                tenant_id, active_by_tenant[tenant_id]
            )
            for tenant_id in sorted(active_by_tenant)
        }

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

    def _require_profile(self, profile_id: str) -> ProviderBinding:
        binding = self._providers.get(profile_id)
        if binding is None:
            raise ProfileUnavailable(
                "No provider is registered for this execution profile."
            )
        return binding

    def _job(self, tenant_id: str, job_id: str) -> _Job:
        job = self._jobs.get((tenant_id, job_id))
        if job is None:
            raise ResourceNotFound("Job does not exist.")
        return job

    def _artifact(self, tenant_id: str, artifact_id: str) -> _Artifact:
        artifact = self._artifacts.get((tenant_id, artifact_id))
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

    def _admit_submission(
        self, submission: JobSubmission, binding: ProviderBinding
    ) -> None:
        now = self._clock.now()
        for handle in submission.secret_handles:
            expires_at = getattr(handle, "expires_at", None)
            if expires_at is not None and now >= expires_at:
                raise AuthorizationError(
                    "A scoped secret handle expired before execution admission."
                )
        spec = submission.resolved_run_spec
        if spec is None:
            if self._dependency_admitter is not None:
                raise IntegrityError(
                    "Qualified service execution requires a ResolvedRunSpec."
                )
            return
        if self._dependency_admitter is None:
            raise ProfileUnavailable(
                "Resolved support dependencies require an admission provider."
            )
        if not spec.valid_from <= now <= spec.valid_until:
            raise ProfileUnavailable(
                "Resolved run specification is outside its validity window."
            )
        bindings = (
            (self._repository_id, spec.repository_id, "repository"),
            (self._scheduler_id, spec.scheduler_id, "scheduler"),
            (self._auth_policy_id, spec.auth_policy_id, "authentication policy"),
        )
        for configured, resolved, label in bindings:
            if configured is None or configured != resolved:
                raise ProfileUnavailable(
                    f"Resolved {label} identity does not match the service binding."
                )
        dependencies = tuple(spec.scientific_dependencies) + tuple(
            spec.deployment_dependencies
        )
        provider_dependencies = tuple(
            dependency
            for dependency in dependencies
            if dependency.profile_id == submission.profile_id
        )
        if len(provider_dependencies) != 1 or (
            provider_dependencies[0].support_tuple_id != binding.support_tuple_id
        ):
            raise ProfileUnavailable(
                "Execution provider does not match its exact resolved support tuple."
            )
        for dependency in dependencies:
            self._dependency_admitter.require(dependency, at_time=now)

    def _job_payload(self, job: _Job) -> dict[str, object]:
        spec = job.submission.resolved_run_spec
        binding = self._require_profile(job.submission.profile_id)
        return {
            "analysis_plan_id": job.submission.analysis_plan.analysis_plan_id,
            "auth_policy_id": None if spec is None else spec.auth_policy_id,
            "execution_plan_id": job.submission.execution_plan.execution_plan_id,
            "numeric_revision_id": job.submission.numeric_revision_id,
            "profile_id": job.submission.profile_id,
            "provider_tuple_id": binding.support_tuple_id,
            "repository_id": None if spec is None else spec.repository_id,
            "resolved_run_spec_id": None if spec is None else spec.spec_id,
            "secret_handle_ids": [
                handle.handle_id for handle in job.submission.secret_handles
            ],
            "scheduler_id": None if spec is None else spec.scheduler_id,
        }

    def _persist_new_job(self, job: _Job) -> None:
        if self._durable_store is None:
            return
        record = DurableJobRecord(
            job.job_id,
            job.tenant_id,
            job.submission.request_id,
            job.submission.request_digest,
            job.state,
            job.attempt,
            self._job_payload(job),
            job.submitted_at,
            job.submitted_at,
        )
        message = OutboxMessage(
            f"dispatch:{job.job_id}:{job.attempt}",
            job.tenant_id,
            "job.dispatch",
            f"{job.job_id}:{job.attempt}",
            {"attempt": job.attempt, "job_id": job.job_id},
            job.submitted_at,
            job.submitted_at,
        )
        with self._durable_store.transaction() as transaction:
            stored = transaction.insert_job(record)
            if stored.job_id != job.job_id:
                raise IntegrityError(
                    "Durable idempotency record is not present in this service instance."
                )
            transaction.reserve_quota(
                job.tenant_id,
                job.job_id,
                job.submission.resources,
                self._quotas[job.tenant_id],
            )
            transaction.enqueue(message)

    def _current_execution_version(self, job: _Job, attempt: int, /) -> int | None:
        if job.attempt != attempt or job.state not in (
            JobState.RUNNING,
            JobState.CANCELLING,
        ):
            raise _ExecutionSuperseded(
                "Execution attempt has been replaced or is no longer active."
            )
        if self._durable_store is None:
            return None
        with self._durable_store.transaction() as transaction:
            current = transaction.get_job(job.tenant_id, job.job_id)
            if (
                current is None
                or current.attempt != attempt
                or current.state not in (JobState.RUNNING, JobState.CANCELLING)
            ):
                raise _ExecutionSuperseded("Durable execution attempt was superseded.")
            return current.version

    def _require_execution_fence(
        self,
        job: _Job,
        attempt: int,
        durable_version: int | None,
        /,
    ) -> None:
        current_version = self._current_execution_version(job, attempt)
        if self._durable_store is None:
            return
        if durable_version is None or current_version != durable_version:
            raise _ExecutionSuperseded(
                "Durable execution attempt/version fence was superseded."
            )

    def _heartbeat_execution(
        self,
        job: _Job,
        attempt: int,
        durable_version: int | None,
        /,
    ) -> int | None:
        with self._lock:
            self._require_execution_fence(job, attempt, durable_version)
            if self._durable_store is None:
                return None
            if durable_version is None:
                raise _ExecutionSuperseded(
                    "Durable heartbeat is missing its version fence."
                )
            now = self._clock.now()
            with self._durable_store.transaction() as transaction:
                current = transaction.get_job(job.tenant_id, job.job_id)
                if (
                    current is None
                    or current.attempt != attempt
                    or current.version != durable_version
                    or current.state not in (JobState.RUNNING, JobState.CANCELLING)
                ):
                    raise _ExecutionSuperseded(
                        "Durable heartbeat lost its attempt/version fence."
                    )
                updated = replace(
                    current,
                    updated_at=now,
                    lease_expires_at=now + self._execution_lease_seconds,
                    version=current.version + 1,
                )
                transaction.update_job(updated, expected_version=current.version)
                return updated.version

    def _sync_durable_job(
        self,
        job: _Job,
        *,
        expected_attempt: int | None = None,
        expected_version: int | None = None,
        lease_expires_at: int | None = None,
        enqueue: bool = False,
        reserve: bool = False,
    ) -> int | None:
        if self._durable_store is None:
            return None
        now = self._clock.now()
        with self._durable_store.transaction() as transaction:
            current = transaction.get_job(job.tenant_id, job.job_id)
            if current is None:
                raise IntegrityError("Durable job disappeared during orchestration.")
            if expected_attempt is not None and current.attempt != expected_attempt:
                raise _ExecutionSuperseded("Durable job attempt changed before commit.")
            if expected_version is not None and current.version != expected_version:
                raise _ExecutionSuperseded("Durable job version changed before commit.")
            updated = replace(
                current,
                state=job.state,
                attempt=job.attempt,
                updated_at=now,
                lease_expires_at=lease_expires_at,
                version=current.version + 1,
            )
            transaction.update_job(updated, expected_version=current.version)
            durable = transaction.get_job(job.tenant_id, job.job_id)
            if durable is None:
                raise IntegrityError("Durable job disappeared after update.")
            if job.state.terminal:
                transaction.release_quota(job.tenant_id, job.job_id)
            if reserve:
                transaction.reserve_quota(
                    job.tenant_id,
                    job.job_id,
                    job.submission.resources,
                    self._quotas[job.tenant_id],
                )
            if enqueue:
                transaction.enqueue(
                    OutboxMessage(
                        f"dispatch:{job.job_id}:{job.attempt}",
                        job.tenant_id,
                        "job.dispatch",
                        f"{job.job_id}:{job.attempt}",
                        {"attempt": job.attempt, "job_id": job.job_id},
                        now,
                        now,
                    )
                )
            return durable.version

    def _record_checkpoint(
        self,
        job: _Job,
        manifest: CheckpointManifest,
        *,
        expected_attempt: int,
    ) -> str:
        with self._lock:
            if job.attempt != expected_attempt:
                raise _ExecutionSuperseded(
                    "Checkpoint belongs to a superseded execution attempt."
                )
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
            self._sync_durable_job(job)

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
        if self._durable_store is not None:
            with self._durable_store.transaction() as transaction:
                transaction.append_audit(record)

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


__all__ = [
    "ExecutionContext",
    "ExecutionProvider",
    "InProcessReferenceService",
    "ProviderBinding",
    "SupportDependencyAdmitter",
    "ReleaseIndexDependencyAdmitter",
]
