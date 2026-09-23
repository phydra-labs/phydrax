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
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol
from uuid import uuid4

from phydrax.execution import ExecutionPlan, ResourceRequest
from phydrax.lifecycle import (
    AnalysisPlan,
    CheckpointManifest,
    CheckpointShard,
    ResolvedRunSpec,
    RunRecord,
)
from phydrax.qualification._evidence import SupportDependency
from phydrax.qualification._registry import (
    ReleaseIndex,
    ReleaseTrustPolicy,
    SupportTuple,
)

from ..logging import emit
from ._auth import (
    _bounded_access_token,
    AccessTokenValidator,
    Clock,
    ResourceAuthorizer,
    SystemClock,
)
from ._contracts import (
    ArtifactDescriptor,
    ArtifactExpired,
    ArtifactRights,
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
    RemoteServiceError,
    ResourceNotFound,
    SecretHandle,
    SignedArtifactGrant,
    TenantQuota,
    TenantUsage,
    ValidatedPrincipal,
)
from ._durability import DurableJobRecord, DurableServiceStore, OutboxMessage
from ._security import ScopedSecretHandle


class SupportDependencyAdmitter(Protocol):
    """Fail-closed exact release-evidence admission boundary."""

    def require(self, dependency: SupportDependency, /, *, at_time: int) -> None: ...


class ReleaseIndexDependencyAdmitter:
    """Adapter from exact SupportDependency records to release-index admission."""

    def __init__(
        self,
        release_index: ReleaseIndex,
        trust_policy: ReleaseTrustPolicy,
        support_tuples: Mapping[str, SupportTuple],
        /,
    ):
        if not isinstance(release_index, ReleaseIndex):
            raise TypeError("Dependency admission requires a typed release index.")
        if not support_tuples:
            raise ValueError("Dependency admission requires exact support tuples.")
        normalized: dict[str, SupportTuple] = {}
        for tuple_id, support_tuple in support_tuples.items():
            if (
                not isinstance(support_tuple, SupportTuple)
                or support_tuple.support_tuple_id != tuple_id
            ):
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
    provider_tuple_id: str
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
                JobState.CANCELING,
            ):
                raise _ExecutionSuperseded(
                    "Execution attempt has been replaced or is no longer active."
                )
            if self._job.state is JobState.CANCELING:
                raise CancellationRequested("The job was canceled.")
            self._service._require_execution_fence(
                self._job, self._attempt, self._durable_version
            )

    def heartbeat(self) -> None:
        self._durable_version = self._service._heartbeat_execution(
            self._job, self._attempt, self._durable_version
        )

    def checkpoint(self, manifest: CheckpointManifest, /) -> str:
        self.heartbeat()
        checkpoint_id, durable_version = self._service._record_checkpoint(
            self._job,
            manifest,
            expected_attempt=self._attempt,
            expected_version=self._durable_version,
        )
        self._durable_version = durable_version
        return checkpoint_id


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
        self._repository = repository
        self._audit: list[AuditRecord] = []
        self._lock = threading.RLock()
        if self._durable_store is not None:
            self._rehydrate_durable_jobs()

    def _request_digest(
        self, submission: JobSubmission, provider_tuple_id: str, /
    ) -> str:
        payload = {
            "kind": "service-request",
            "submission_digest": submission.request_digest,
            "analysis_plan_fingerprint": submission.analysis_plan.plan_fingerprint,
            "execution_plan_fingerprint": submission.execution_plan.plan_fingerprint,
            "provider_tuple_id": provider_tuple_id,
            "repository_id": self._repository_id,
            "scheduler_id": self._scheduler_id,
            "auth_policy_id": self._auth_policy_id,
        }
        return hashlib.sha256(
            json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _job_payload(
        self,
        job: _Job,
        /,
        *,
        checkpoint_manifest: CheckpointManifest | None = None,
        artifact: _Artifact | None = None,
    ) -> dict[str, object]:
        return {
            "kind": "service-job",
            "submission": _submission_payload(job.submission),
            "provider_tuple_id": job.provider_tuple_id,
            "state": job.state.value,
            "attempt": job.attempt,
            "expires_at": job.expires_at,
            "run_record": _run_record_payload(job.run_record),
            "prior_run_records": [
                _run_record_payload(record) for record in job.prior_run_records
            ],
            "checkpoints": [
                _checkpoint_payload(
                    checkpoint_manifest
                    if checkpoint_manifest is not None
                    and checkpoint_manifest.checkpoint_id == checkpoint_id
                    else self._checkpoints[checkpoint_id]
                )
                for checkpoint_id in job.checkpoint_ids
            ],
            "recovered_checkpoint_id": job.recovered_checkpoint_id,
            "artifacts": [
                _artifact_descriptor_payload(
                    artifact.descriptor
                    if artifact is not None
                    and artifact.descriptor.artifact_id == artifact_id
                    else self._artifacts[(job.tenant_id, artifact_id)].descriptor
                )
                for artifact_id in job.artifact_ids
            ],
            "started_at": job.started_at,
            "finished_at": job.finished_at,
            "cancel_requested_at": job.cancel_requested_at,
            "failure": _failure_payload(job.failure),
        }

    def _job_from_durable(self, record: DurableJobRecord, /) -> _Job:
        payload = record.payload
        if payload.get("kind") != "service-job":
            raise IntegrityError("Durable job payload kind is invalid.")
        submission_payload = payload.get("submission")
        if not isinstance(submission_payload, Mapping):
            raise IntegrityError("Durable job submission is missing.")
        if (
            payload.get("state") != record.state.value
            or payload.get("attempt") != record.attempt
        ):
            raise IntegrityError("Durable job state revision is inconsistent.")
        submission = _submission_from_payload(submission_payload)
        provider_tuple_id = payload.get("provider_tuple_id")
        if not isinstance(provider_tuple_id, str) or not provider_tuple_id:
            raise IntegrityError("Durable provider support identity is invalid.")
        if record.request_digest != self._request_digest(submission, provider_tuple_id):
            raise IntegrityError("Durable service request identity is invalid.")
        if record.tenant_id not in self._quotas:
            raise IntegrityError("Durable job belongs to an unconfigured tenant.")
        checkpoints_payload = payload.get("checkpoints")
        prior_payload = payload.get("prior_run_records")
        artifacts_payload = payload.get("artifacts")
        if (
            not isinstance(checkpoints_payload, list)
            or not isinstance(prior_payload, list)
            or not isinstance(artifacts_payload, list)
        ):
            raise IntegrityError("Durable job collections are malformed.")
        checkpoints = tuple(
            _checkpoint_from_payload(value)
            for value in checkpoints_payload
            if isinstance(value, Mapping)
        )
        if len(checkpoints) != len(checkpoints_payload):
            raise IntegrityError("Durable checkpoint collection is malformed.")
        checkpoint_ids = [value.checkpoint_id for value in checkpoints]
        if len(set(checkpoint_ids)) != len(checkpoint_ids):
            raise IntegrityError("Durable checkpoint identities are duplicated.")
        for checkpoint in checkpoints:
            current = self._checkpoints.get(checkpoint.checkpoint_id)
            if current is not None and current.manifest_id != checkpoint.manifest_id:
                raise IntegrityError("Durable checkpoint identity conflicts across jobs.")
        for index, checkpoint in enumerate(checkpoints):
            parent = None if index == 0 else checkpoints[index - 1]
            if checkpoint.parent_checkpoint_id != (
                None if parent is None else parent.checkpoint_id
            ) or checkpoint.parent_manifest_id != (
                None if parent is None else parent.manifest_id
            ):
                raise IntegrityError("Durable checkpoint lineage is not a direct chain.")
        for checkpoint in checkpoints:
            self._checkpoints[checkpoint.checkpoint_id] = checkpoint
        prior = [
            _run_record_from_payload(value)
            for value in prior_payload
            if isinstance(value, Mapping)
        ]
        if len(prior) != len(prior_payload):
            raise IntegrityError("Durable prior run records are malformed.")
        raw_run = payload.get("run_record")
        if raw_run is None and record.state is not JobState.QUEUED:
            raise IntegrityError("Only a recovered queued job may omit its run record.")
        if raw_run is None:
            run_record = self._run_record(
                record.job_id,
                submission,
                record.state.value,
                checkpoint_ids[-1] if checkpoint_ids else None,
                attempt=record.attempt,
            )
        elif isinstance(raw_run, Mapping):
            run_record = _run_record_from_payload(raw_run)
        else:
            raise IntegrityError("Durable current run record is malformed.")
        expected_run_status = (
            "running" if record.state is JobState.CANCELING else record.state.value
        )
        if (
            run_record.status != expected_run_status
            or run_record.analysis_plan_id != submission.analysis_plan.analysis_plan_id
            or run_record.numeric_revision_id != submission.numeric_revision_id
            or run_record.execution_plan_id != submission.execution_plan.execution_plan_id
        ):
            raise IntegrityError("Durable run record contradicts the job state.")
        failure = _failure_from_payload(payload.get("failure"))
        if (record.state is JobState.FAILED) != (failure is not None):
            raise IntegrityError("Durable failure evidence contradicts the job state.")
        recovered_checkpoint_id = payload.get("recovered_checkpoint_id")
        if recovered_checkpoint_id is not None and (
            not isinstance(recovered_checkpoint_id, str)
            or recovered_checkpoint_id not in checkpoint_ids
        ):
            raise IntegrityError("Durable recovery checkpoint is not owned by the job.")
        expires_at = _required_nonnegative_int(payload.get("expires_at"), "expires_at")
        started_at = _optional_nonnegative_int(payload.get("started_at"), "started_at")
        finished_at = _optional_nonnegative_int(payload.get("finished_at"), "finished_at")
        cancel_requested_at = _optional_nonnegative_int(
            payload.get("cancel_requested_at"), "cancel_requested_at"
        )
        artifacts = tuple(
            self._restore_repository_artifact(_artifact_descriptor_from_payload(value))
            for value in artifacts_payload
            if isinstance(value, Mapping)
        )
        if len(artifacts) != len(artifacts_payload) or any(
            value.descriptor.tenant_id != record.tenant_id
            or value.descriptor.job_id != record.job_id
            for value in artifacts
        ):
            raise IntegrityError("Durable artifact collection is malformed.")
        artifact_ids = [value.descriptor.artifact_id for value in artifacts]
        if len(set(artifact_ids)) != len(artifact_ids):
            raise IntegrityError("Durable artifact identities are duplicated.")
        for value in artifacts:
            key = (value.descriptor.tenant_id, value.descriptor.artifact_id)
            current = self._artifacts.get(key)
            if current is not None and current.descriptor != value.descriptor:
                raise IntegrityError("Durable artifact identity conflicts across jobs.")
            self._artifacts[key] = value
        return _Job(
            record.job_id,
            record.tenant_id,
            submission,
            provider_tuple_id,
            record.state,
            record.attempt,
            record.submitted_at,
            expires_at,
            run_record,
            prior,
            checkpoint_ids,
            (None if recovered_checkpoint_id is None else recovered_checkpoint_id),
            artifact_ids,
            started_at,
            finished_at,
            cancel_requested_at,
            failure,
        )

    def _rehydrate_durable_jobs(self) -> None:
        assert self._durable_store is not None
        for record in self._durable_store.jobs():
            job = self._job_from_durable(record)
            key = (job.tenant_id, job.job_id)
            if key in self._jobs:
                raise IntegrityError("Durable job identity is duplicated.")
            self._jobs[key] = job
            if job.submission.request_id:
                request_key = (job.tenant_id, job.submission.request_id)
                request_identity = (job.job_id, record.request_digest)
                current = self._requests.get(request_key)
                if current is not None and current != request_identity:
                    raise IntegrityError("Durable request identity is duplicated.")
                self._requests[request_key] = request_identity

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
            if any(
                job.submission.profile_id == profile_id
                and job.provider_tuple_id != tuple_id
                for job in self._jobs.values()
            ):
                raise IntegrityError(
                    "Provider support identity differs from durable job state."
                )
            self._providers[profile_id] = ProviderBinding(provider, tuple_id)
        emit(
            "INFO",
            "service.provider.registered",
            "Execution provider registered",
            profile_id=profile_id,
            support_tuple_id=tuple_id,
        )

    def submit(self, token: str, submission: JobSubmission, /) -> JobStatus:
        principal = self._authenticate(token)
        self._authorize(principal, "service:submit", principal.tenant_id)
        if not isinstance(submission, JobSubmission):
            raise TypeError("submission must be a JobSubmission.")
        with self._lock:
            binding = self._require_profile(submission.profile_id)
            request_digest = self._request_digest(submission, binding.support_tuple_id)
            request_key = (principal.tenant_id, submission.request_id)
            if submission.request_id and request_key in self._requests:
                job_id, digest = self._requests[request_key]
                if digest != request_digest:
                    raise IntegrityError(
                        "A request ID was reused for a different submission."
                    )
                return self._status(self._jobs[(principal.tenant_id, job_id)])
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
                binding.support_tuple_id,
                JobState.QUEUED,
                1,
                now,
                now + submission.retention_seconds,
                self._run_record(job_id, submission, "queued", attempt=1),
            )
            audit_template = self._audit_template(
                principal,
                "submit",
                "job",
                job_id,
                "allowed",
                "queued",
                submission.request_id,
            )
            stored, audit_record = self._persist_new_job(job, audit_template)
            if stored is not None and stored.job_id != job.job_id:
                existing = self._jobs.get((stored.tenant_id, stored.job_id))
                if existing is None:
                    existing = self._job_from_durable(stored)
                    self._jobs[(existing.tenant_id, existing.job_id)] = existing
                if submission.request_id:
                    self._requests[request_key] = (
                        existing.job_id,
                        stored.request_digest,
                    )
                return self._status(existing)
            self._jobs[(job.tenant_id, job_id)] = job
            if submission.request_id:
                self._requests[request_key] = (job_id, request_digest)
            assert audit_record is not None
            self._audit.append(audit_record)
            status = self._status(job)
        emit(
            "INFO",
            "service.job.submitted",
            "Service job submitted",
            audit_event_id=audit_record.event_id,
            job_id=job_id,
            profile_id=submission.profile_id,
            run_record_id=status.run_record.record_id,
        )
        return status

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
                raise InvalidTransition("A terminal job cannot be canceled.")
            candidate = self._copy_job(job)
            if job.state is JobState.QUEUED:
                candidate.state = JobState.CANCELED
                candidate.finished_at = self._clock.now()
                candidate.run_record = self._run_record(
                    job_id,
                    job.submission,
                    "canceled",
                    job.checkpoint_ids[-1] if job.checkpoint_ids else None,
                    attempt=job.attempt,
                )
            else:
                candidate.state = JobState.CANCELING
                candidate.cancel_requested_at = self._clock.now()
            audit_record = self._commit_job_candidate(
                job,
                candidate,
                audit=self._audit_template(
                    principal,
                    "cancel",
                    "job",
                    job_id,
                    "allowed",
                    candidate.state.value,
                    "",
                ),
            )[1]
            assert audit_record is not None
            status = self._status(job)
        emit(
            "WARNING",
            "service.job.canceled",
            "Service job cancellation committed",
            audit_event_id=audit_record.event_id,
            job_id=job_id,
            state=status.state.value,
        )
        return status

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
            candidate = self._copy_job(job)
            candidate.prior_run_records.append(job.run_record)
            candidate.attempt += 1
            candidate.state = JobState.QUEUED
            candidate.started_at = None
            candidate.finished_at = None
            candidate.cancel_requested_at = None
            candidate.failure = None
            candidate.recovered_checkpoint_id = (
                candidate.checkpoint_ids[-1] if candidate.checkpoint_ids else None
            )
            candidate.run_record = self._run_record(
                job_id,
                candidate.submission,
                "queued",
                candidate.recovered_checkpoint_id,
                attempt=candidate.attempt,
            )
            audit_record = self._commit_job_candidate(
                job,
                candidate,
                enqueue=True,
                reserve=True,
                audit=self._audit_template(
                    principal,
                    "restart",
                    "job",
                    job_id,
                    "allowed",
                    "queued",
                    "",
                ),
            )[1]
            assert audit_record is not None
            status = self._status(job)
        emit(
            "INFO",
            "service.job.restarted",
            "Service job restarted",
            attempt=status.attempt,
            audit_event_id=audit_record.event_id,
            job_id=job_id,
            run_record_id=status.run_record.record_id,
        )
        return status

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
            candidate = self._copy_job(job)
            candidate.state = JobState.RUNNING
            candidate.started_at = self._clock.now()
            candidate.run_record = self._run_record(
                job_id,
                candidate.submission,
                "running",
                candidate.recovered_checkpoint_id,
                attempt=candidate.attempt,
            )
            execution_attempt = candidate.attempt
            durable_version = self._commit_job_candidate(
                job,
                candidate,
                lease_expires_at=(candidate.started_at + self._execution_lease_seconds),
            )[0]
            context = _ProviderContext(self, job, execution_attempt, durable_version)
        emit(
            "INFO",
            "service.job.started",
            "Service job execution started",
            attempt=execution_attempt,
            job_id=job_id,
            profile_id=job.submission.profile_id,
            run_record_id=job.run_record.record_id,
        )
        audit_record: AuditRecord | None = None
        try:
            result = binding.provider(job.submission, context)
            if not isinstance(result, ProviderResult):
                raise IntegrityError("Provider must return a ProviderResult.")
            with self._lock:
                if job.state is JobState.CANCELING:
                    raise CancellationRequested("The job was canceled.")
                self._require_execution_fence(
                    job, execution_attempt, context._durable_version
                )
                finished_at = self._clock.now()
                candidate = self._copy_job(job)
                candidate.state = JobState.SUCCEEDED
                candidate.finished_at = finished_at
                candidate.run_record = self._run_record(
                    job_id,
                    candidate.submission,
                    "completed",
                    candidate.checkpoint_ids[-1] if candidate.checkpoint_ids else None,
                    result,
                    attempt=candidate.attempt,
                )
                audit_record = self._commit_job_candidate(
                    job,
                    candidate,
                    expected_attempt=execution_attempt,
                    expected_version=context._durable_version,
                    audit=self._audit_template(
                        principal,
                        "execute",
                        "job",
                        job_id,
                        "allowed",
                        "completed",
                        "",
                    ),
                )[1]
        except _ExecutionSuperseded:
            pass
        except CancellationRequested:
            with self._lock:
                if job.attempt != execution_attempt:
                    return self._status(job)
                current_version = self._current_execution_version(job, execution_attempt)
                finished_at = self._clock.now()
                candidate = self._copy_job(job)
                candidate.state = JobState.CANCELED
                candidate.finished_at = finished_at
                candidate.run_record = self._run_record(
                    job_id,
                    candidate.submission,
                    "canceled",
                    candidate.checkpoint_ids[-1] if candidate.checkpoint_ids else None,
                    attempt=candidate.attempt,
                )
                audit_record = self._commit_job_candidate(
                    job,
                    candidate,
                    expected_attempt=execution_attempt,
                    expected_version=current_version,
                    audit=self._audit_template(
                        principal,
                        "execute",
                        "job",
                        job_id,
                        "allowed",
                        "canceled",
                        "",
                    ),
                )[1]
        except Exception as error:
            with self._lock:
                try:
                    self._require_execution_fence(
                        job, execution_attempt, context._durable_version
                    )
                except _ExecutionSuperseded:
                    return self._status(job)
                finished_at = self._clock.now()
                if isinstance(error, RemoteServiceError):
                    error_type = type(error)
                    message = str(error).strip() or error_type.__name__
                    failure = FailureEvidence(
                        error_type.__name__,
                        f"{error_type.__module__}.{error_type.__qualname__}",
                        message,
                        False,
                        execution_attempt,
                    )
                else:
                    failure = FailureEvidence(
                        "provider_failure",
                        "ProviderExecutionError",
                        "Provider execution failed.",
                        False,
                        execution_attempt,
                    )
                candidate = self._copy_job(job)
                candidate.state = JobState.FAILED
                candidate.finished_at = finished_at
                candidate.failure = failure
                candidate.run_record = self._run_record(
                    job_id,
                    candidate.submission,
                    "failed",
                    candidate.checkpoint_ids[-1] if candidate.checkpoint_ids else None,
                    diagnostic_ids=failure.diagnostic_ids,
                    attempt=candidate.attempt,
                )
                audit_record = self._commit_job_candidate(
                    job,
                    candidate,
                    expected_attempt=execution_attempt,
                    expected_version=context._durable_version,
                    audit=self._audit_template(
                        principal,
                        "execute",
                        "job",
                        job_id,
                        "failed",
                        failure.code,
                        "",
                    ),
                )[1]
        with self._lock:
            status = self._status(job)
        event_name = {
            JobState.CANCELED: "service.job.canceled",
            JobState.FAILED: "service.job.failed",
            JobState.SUCCEEDED: "service.job.completed",
        }.get(status.state, "service.job.updated")
        emit(
            "ERROR" if status.state is JobState.FAILED else "INFO",
            event_name,
            "Service job execution finished",
            attempt=status.attempt,
            audit_event_id=(None if audit_record is None else audit_record.event_id),
            job_id=job_id,
            run_record_id=status.run_record.record_id,
            state=status.state.value,
        )
        return status

    def store_artifact(
        self,
        token: str,
        job_id: str,
        content: bytes,
        /,
        *,
        scientific_artifact_id: str,
        media_type: str,
        rights: ArtifactRights,
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
        if (
            not isinstance(content, bytes)
            or not scientific_artifact_id
            or not media_type
            or not isinstance(rights, ArtifactRights)
        ):
            raise IntegrityError("Artifact content and metadata are invalid.")
        content_sha256 = hashlib.sha256(content).hexdigest()
        if (
            rights.scientific_artifact_id != scientific_artifact_id
            or rights.content_sha256 != content_sha256
            or rights.byte_size != len(content)
            or rights.classification != classification
        ):
            raise IntegrityError(
                "Artifact metadata differs from its immutable rights binding."
            )
        with self._lock:
            job = self._job(principal.tenant_id, job_id)
            self._expire_job(job)
            for stored in self._artifacts.values():
                descriptor = stored.descriptor
                if descriptor.tenant_id != job.tenant_id or not (
                    descriptor.scientific_artifact_id == scientific_artifact_id
                    or descriptor.content_sha256 == content_sha256
                    or descriptor.rights.rights_id == rights.rights_id
                ):
                    continue
                if (
                    descriptor.scientific_artifact_id != scientific_artifact_id
                    or descriptor.content_sha256 != content_sha256
                    or descriptor.classification != classification
                    or descriptor.rights != rights
                    or descriptor.cad != cad
                ):
                    raise IntegrityError(
                        "Artifact content or scientific identity is already bound to "
                        "different rights or classification."
                    )
            quota = self._quotas[job.tenant_id]
            if (
                self._usage(job.tenant_id).retained_artifact_bytes + len(content)
                > quota.retained_artifact_bytes
            ):
                raise QuotaExceeded("Tenant retained artifact quota would be exceeded.")
            now = self._clock.now()
            artifact_id = uuid4().hex
            stored_artifact = self._publish_repository_artifact(
                artifact_id=artifact_id,
                scientific_artifact_id=scientific_artifact_id,
                job=job,
                content=content,
                content_sha256=content_sha256,
                media_type=media_type,
                classification=classification,
                created_at=now,
                rights=rights,
                cad=cad,
            )
            descriptor = stored_artifact.descriptor
            candidate = self._copy_job(job)
            candidate.artifact_ids.append(artifact_id)
            audit_record = self._commit_job_candidate(
                job,
                candidate,
                artifact=stored_artifact,
                outbox_messages=(
                    OutboxMessage(
                        f"artifact:{artifact_id}",
                        job.tenant_id,
                        "artifact.published",
                        artifact_id,
                        {
                            "artifact_id": artifact_id,
                            "content_sha256": content_sha256,
                            "job_id": job.job_id,
                        },
                        now,
                        now,
                    ),
                ),
                audit=self._audit_template(
                    principal,
                    "artifact.write",
                    "artifact",
                    artifact_id,
                    "allowed",
                    "stored",
                    "",
                ),
            )[1]
            assert audit_record is not None
            self._artifacts[(job.tenant_id, artifact_id)] = stored_artifact
        emit(
            "INFO",
            "service.artifact.stored",
            "Service artifact stored",
            artifact_id=descriptor.artifact_id,
            audit_event_id=audit_record.event_id,
            byte_count=descriptor.byte_size,
            classification=descriptor.classification,
            job_id=job_id,
            scientific_artifact_id=descriptor.scientific_artifact_id,
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
            self._authorize_artifact_egress(artifact)
            expires_at = min(
                self._clock.now() + lifetime_seconds, artifact.descriptor.expires_at
            )
            token_value = self._grant_token(artifact.descriptor, expires_at)
            audit_record = self._audit_event(
                principal,
                "artifact.grant",
                "artifact",
                artifact_id,
                "allowed",
                "granted",
                "",
            )
            signed_grant = SignedArtifactGrant(
                token_value,
                artifact_id,
                artifact.descriptor.tenant_id,
                expires_at,
                artifact.descriptor.rights,
            )
        emit(
            "INFO",
            "service.artifact.granted",
            "Service artifact grant created",
            artifact_id=artifact_id,
            audit_event_id=audit_record.event_id,
            expires_at=expires_at,
        )
        return signed_grant

    def fetch_artifact(
        self, token: str, grant: SignedArtifactGrant | str, /
    ) -> FetchedArtifact:
        principal = self._authenticate(token)
        value = grant.token if isinstance(grant, SignedArtifactGrant) else grant
        (
            artifact_id,
            tenant_id,
            expires_at,
            content_sha256,
            classification,
            rights_binding_id,
        ) = self._verify_grant(value)
        if isinstance(grant, SignedArtifactGrant) and (
            grant.artifact_id != artifact_id
            or grant.tenant_id != tenant_id
            or grant.expires_at != expires_at
            or grant.rights.rights_binding_id != rights_binding_id
        ):
            raise IntegrityError("Artifact grant fields differ from its signed payload.")
        self._authorize(principal, "service:artifact:fetch", tenant_id)
        with self._lock:
            artifact = self._artifact(tenant_id, artifact_id)
            if self._clock.now() >= expires_at:
                raise ArtifactExpired("Artifact grant has expired.")
            self._assert_artifact_live(artifact)
            if (
                artifact.descriptor.content_sha256 != content_sha256
                or artifact.descriptor.classification != classification
                or artifact.descriptor.rights.rights_binding_id != rights_binding_id
            ):
                raise IntegrityError("Artifact descriptor differs from its signed grant.")
            self._authorize_artifact_egress(artifact)
            if not hmac.compare_digest(
                hashlib.sha256(artifact.content).hexdigest(),
                artifact.descriptor.content_sha256,
            ):
                raise IntegrityError(
                    "Artifact content digest does not match its descriptor."
                )
            audit_record = self._audit_event(
                principal,
                "artifact.fetch",
                "artifact",
                artifact_id,
                "allowed",
                "fetched",
                "",
            )
            fetched = FetchedArtifact(artifact.descriptor, artifact.content)
        emit(
            "DEBUG",
            "service.artifact.fetched",
            "Service artifact fetched",
            artifact_id=artifact_id,
            audit_event_id=audit_record.event_id,
            byte_count=fetched.descriptor.byte_size,
        )
        return fetched

    def delete_expired(self) -> tuple[str, ...]:
        """Purge expired artifacts and terminal job records; return deleted IDs."""
        with self._lock:
            now = self._clock.now()
            expired_artifacts = tuple(
                (key, artifact)
                for key, artifact in self._artifacts.items()
                if artifact.descriptor.expires_at <= now
            )
            for job in self._jobs.values():
                self._expire_job(job)
            expired_jobs = tuple(
                (key, job)
                for key, job in self._jobs.items()
                if job.state.terminal and job.expires_at <= now
            )
            if self._durable_store is not None and expired_jobs:
                with self._durable_store.transaction() as transaction:
                    for _key, job in expired_jobs:
                        current = transaction.get_job(job.tenant_id, job.job_id)
                        if current is None:
                            raise IntegrityError(
                                "Durable job disappeared before retention deletion."
                            )
                        transaction.delete_job(
                            job.tenant_id,
                            job.job_id,
                            expected_version=current.version,
                        )
            deleted: list[str] = []
            for artifact_key, artifact in expired_artifacts:
                del self._artifacts[artifact_key]
                deleted.append(artifact.descriptor.artifact_id)
            for job_key, job in expired_jobs:
                for checkpoint_id in job.checkpoint_ids:
                    del self._checkpoints[checkpoint_id]
                    deleted.append(checkpoint_id)
                del self._jobs[job_key]
                if job.submission.request_id:
                    self._requests.pop((job.tenant_id, job.submission.request_id), None)
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
        if self._durable_store is not None:
            self._durable_store.verify_audit_chain()
            return
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

    def recover_stale_attempts(self) -> tuple[JobStatus, ...]:
        """Recover expired durable execution leases as new idempotent attempts."""
        if self._durable_store is None:
            return ()
        recovered = self._durable_store.recover_stale_attempts(self._clock.now())
        statuses: list[JobStatus] = []
        with self._lock:
            for record in recovered:
                candidate = self._job_from_durable(record)
                job = self._jobs.get((record.tenant_id, record.job_id))
                if job is None:
                    self._jobs[(record.tenant_id, record.job_id)] = candidate
                    job = candidate
                else:
                    self._apply_job(job, candidate)
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
            bounded = _bounded_access_token(token)
            return self._validator.validate(bounded)
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

    def _publish_repository_artifact(
        self,
        *,
        artifact_id: str,
        scientific_artifact_id: str,
        job: _Job,
        content: bytes,
        content_sha256: str,
        media_type: str,
        classification: str,
        created_at: int,
        rights: ArtifactRights,
        cad: CADArtifactMetadata | None,
    ) -> _Artifact:
        if self._repository is None:
            if self._durable_store is not None:
                raise IntegrityError(
                    "Durable artifact publication requires the bound repository."
                )
            descriptor = ArtifactDescriptor(
                artifact_id,
                scientific_artifact_id,
                job.job_id,
                job.tenant_id,
                content_sha256,
                len(content),
                media_type,
                classification,  # type: ignore[arg-type]
                created_at,
                job.expires_at,
                uuid4().hex,
                self._encryption,
                rights,
                cad,
            )
            return _Artifact(descriptor, bytes(content))

        transaction = self._repository.begin(
            artifact_id,
            f"service:{job.tenant_id}:{job.job_id}",
            started_at=created_at,
        )
        descriptor = ArtifactDescriptor(
            artifact_id,
            scientific_artifact_id,
            job.job_id,
            job.tenant_id,
            content_sha256,
            len(content),
            media_type,
            classification,  # type: ignore[arg-type]
            created_at,
            job.expires_at,
            transaction.transaction_id,
            self._encryption,
            rights,
            cad,
        )
        chunks = []
        offset = 0
        index = 0
        while offset < len(content) or (not content and not chunks):
            end = min(offset + self._repository.maximum_chunk_bytes, len(content))
            chunks.append(
                self._repository.write_chunk(
                    transaction,
                    "content",
                    index,
                    offset,
                    content[offset:end],
                )
            )
            offset = end
            index += 1
        descriptor_json = json.dumps(
            _artifact_descriptor_payload(descriptor),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        repository_manifest = self._repository.commit(
            transaction,
            tuple(chunks),
            metadata={
                "kind": "service-artifact",
                "descriptor": descriptor_json,
                "descriptor_sha256": hashlib.sha256(
                    descriptor_json.encode("utf-8")
                ).hexdigest(),
            },
            committed_at=created_at,
        )
        if (
            repository_manifest.artifact_id != descriptor.artifact_id
            or repository_manifest.transaction_id != descriptor.storage_generation
            or repository_manifest.provider_id != self._repository_id
        ):
            raise IntegrityError("Repository artifact commit identity is invalid.")
        return _Artifact(descriptor, bytes(content))

    def _restore_repository_artifact(
        self, descriptor: ArtifactDescriptor, /
    ) -> _Artifact:
        if self._repository is None:
            raise IntegrityError(
                "Durable artifact restoration requires the bound repository."
            )
        repository_manifest = self._repository.get_manifest(descriptor.artifact_id)
        metadata = dict(repository_manifest.metadata)
        descriptor_json = json.dumps(
            _artifact_descriptor_payload(descriptor),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if (
            repository_manifest.provider_id != self._repository_id
            or repository_manifest.transaction_id != descriptor.storage_generation
            or metadata.get("kind") != "service-artifact"
            or metadata.get("descriptor") != descriptor_json
            or metadata.get("descriptor_sha256")
            != hashlib.sha256(descriptor_json.encode("utf-8")).hexdigest()
        ):
            raise IntegrityError("Repository artifact descriptor identity is invalid.")
        chunks = tuple(
            sorted(
                (
                    chunk
                    for chunk in repository_manifest.chunks
                    if chunk.logical_name == "content"
                ),
                key=lambda chunk: chunk.index,
            )
        )
        if len(chunks) != len(repository_manifest.chunks) or tuple(
            chunk.index for chunk in chunks
        ) != tuple(range(len(chunks))):
            raise IntegrityError("Repository artifact chunk inventory is invalid.")
        content = bytearray()
        for chunk in chunks:
            if chunk.offset != len(content):
                raise IntegrityError("Repository artifact chunk offsets are invalid.")
            content.extend(
                self._repository.read_chunk(
                    repository_manifest,
                    chunk,
                    maximum_plaintext_bytes=max(1, descriptor.byte_size),
                )
            )
            if len(content) > descriptor.byte_size:
                raise IntegrityError("Repository artifact exceeds its byte bound.")
        payload = bytes(content)
        if (
            len(payload) != descriptor.byte_size
            or hashlib.sha256(payload).hexdigest() != descriptor.content_sha256
        ):
            raise IntegrityError("Repository artifact content identity is invalid.")
        return _Artifact(descriptor, payload)

    def _reserve(self, tenant_id: str, submission: JobSubmission) -> None:
        usage = self._usage(tenant_id)
        quota = self._quotas[tenant_id]
        requested = submission.resources
        if (
            usage.active_jobs + 1 > quota.active_jobs
            or usage.cpu_cores + requested.cpu_cores > quota.cpu_cores
            or usage.memory_bytes + requested.memory_bytes > quota.memory_bytes
            or usage.gpu_count + requested.accelerator_count > quota.gpu_count
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
            sum(job.submission.resources.accelerator_count for job in jobs),
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

    def _persist_new_job(
        self, job: _Job, audit: AuditRecord, /
    ) -> tuple[DurableJobRecord | None, AuditRecord | None]:
        if self._durable_store is None:
            return None, self._commit_local_audit(audit)
        record = DurableJobRecord(
            job.job_id,
            job.tenant_id,
            job.submission.request_id,
            self._request_digest(job.submission, job.provider_tuple_id),
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
                return stored, None
            transaction.reserve_quota(
                job.tenant_id,
                job.job_id,
                job.submission.resources,
                self._quotas[job.tenant_id],
            )
            transaction.enqueue(message)
            committed_audit = transaction.append_audit(audit)
        return stored, committed_audit

    @staticmethod
    def _copy_job(job: _Job, /) -> _Job:
        return replace(
            job,
            prior_run_records=list(job.prior_run_records),
            checkpoint_ids=list(job.checkpoint_ids),
            artifact_ids=list(job.artifact_ids),
        )

    @staticmethod
    def _apply_job(job: _Job, candidate: _Job, /) -> None:
        for name in (
            "state",
            "attempt",
            "expires_at",
            "run_record",
            "recovered_checkpoint_id",
            "started_at",
            "finished_at",
            "cancel_requested_at",
            "failure",
        ):
            setattr(job, name, getattr(candidate, name))
        job.prior_run_records[:] = candidate.prior_run_records
        job.checkpoint_ids[:] = candidate.checkpoint_ids
        job.artifact_ids[:] = candidate.artifact_ids

    def _commit_job_candidate(
        self,
        job: _Job,
        candidate: _Job,
        /,
        *,
        expected_attempt: int | None = None,
        expected_version: int | None = None,
        lease_expires_at: int | None = None,
        enqueue: bool = False,
        reserve: bool = False,
        audit: AuditRecord | None = None,
        artifact: _Artifact | None = None,
        outbox_messages: tuple[OutboxMessage, ...] = (),
    ) -> tuple[int | None, AuditRecord | None]:
        version, committed_audit = self._sync_durable_job(
            candidate,
            expected_attempt=expected_attempt,
            expected_version=expected_version,
            lease_expires_at=lease_expires_at,
            enqueue=enqueue,
            reserve=reserve,
            audit=audit,
            artifact=artifact,
            outbox_messages=outbox_messages,
        )
        self._apply_job(job, candidate)
        if committed_audit is not None:
            self._audit.append(committed_audit)
        return version, committed_audit

    def _current_execution_version(self, job: _Job, attempt: int, /) -> int | None:
        if job.attempt != attempt or job.state not in (
            JobState.RUNNING,
            JobState.CANCELING,
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
                or current.state not in (JobState.RUNNING, JobState.CANCELING)
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
                    or current.state not in (JobState.RUNNING, JobState.CANCELING)
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
        audit: AuditRecord | None = None,
        checkpoint_manifest: CheckpointManifest | None = None,
        artifact: _Artifact | None = None,
        outbox_messages: tuple[OutboxMessage, ...] = (),
    ) -> tuple[int | None, AuditRecord | None]:
        if self._durable_store is None:
            committed = None if audit is None else self._commit_local_audit(audit)
            return None, committed
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
                request_digest=self._request_digest(
                    job.submission, job.provider_tuple_id
                ),
                state=job.state,
                attempt=job.attempt,
                payload=self._job_payload(
                    job,
                    checkpoint_manifest=checkpoint_manifest,
                    artifact=artifact,
                ),
                updated_at=now,
                lease_expires_at=lease_expires_at,
                version=current.version + 1,
            )
            transaction.update_job(updated, expected_version=current.version)
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
            for message in outbox_messages:
                transaction.enqueue(message)
            committed_audit = None if audit is None else transaction.append_audit(audit)
        return updated.version, committed_audit

    def _record_checkpoint(
        self,
        job: _Job,
        manifest: CheckpointManifest,
        *,
        expected_attempt: int,
        expected_version: int | None,
    ) -> tuple[str, int | None]:
        with self._lock:
            if not isinstance(manifest, CheckpointManifest):
                raise TypeError("manifest must be a CheckpointManifest.")
            if job.attempt != expected_attempt:
                raise _ExecutionSuperseded(
                    "Checkpoint belongs to a superseded execution attempt."
                )
            if job.state not in (JobState.RUNNING, JobState.CANCELING):
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
            existing = self._checkpoints.get(manifest.checkpoint_id)
            if existing is not None:
                if (
                    existing.manifest_id == manifest.manifest_id
                    and manifest.checkpoint_id in job.checkpoint_ids
                ):
                    return manifest.checkpoint_id, expected_version
                raise IntegrityError("Checkpoint identifier already exists.")
            if bool(job.checkpoint_ids) != (manifest.parent_checkpoint_id is not None):
                raise IntegrityError(
                    "Checkpoint parent presence must match the job checkpoint chain."
                )
            if job.checkpoint_ids:
                parent = self._checkpoints[job.checkpoint_ids[-1]]
                if (
                    manifest.parent_checkpoint_id != parent.checkpoint_id
                    or manifest.parent_manifest_id != parent.manifest_id
                ):
                    raise IntegrityError(
                        "Checkpoint parent must be the job's exact current manifest."
                    )
            self._verify_repository_checkpoint(manifest)
            candidate = self._copy_job(job)
            candidate.checkpoint_ids.append(manifest.checkpoint_id)
            durable_version, _ = self._sync_durable_job(
                candidate,
                expected_attempt=expected_attempt,
                expected_version=expected_version,
                lease_expires_at=self._clock.now() + self._execution_lease_seconds,
                checkpoint_manifest=manifest,
            )
            self._checkpoints[manifest.checkpoint_id] = manifest
            self._apply_job(job, candidate)
            return manifest.checkpoint_id, durable_version

    def _verify_repository_checkpoint(self, manifest: CheckpointManifest, /) -> None:
        if self._repository is None:
            raise IntegrityError(
                "Checkpoint publication requires the bound artifact repository."
            )
        repository_manifest = self._repository.get_manifest(manifest.checkpoint_id)
        metadata = dict(repository_manifest.metadata)
        if (
            repository_manifest.artifact_id != manifest.checkpoint_id
            or repository_manifest.provider_id != self._repository_id
            or metadata.get("checkpoint_manifest_id") != manifest.manifest_id
        ):
            raise IntegrityError(
                "Repository checkpoint does not bind the lifecycle manifest."
            )
        expected_names = {shard.shard_id for shard in manifest.shards}
        if {chunk.logical_name for chunk in repository_manifest.chunks} != expected_names:
            raise IntegrityError(
                "Repository checkpoint payload inventory differs from its manifest."
            )
        for shard in manifest.shards:
            chunks = tuple(
                sorted(
                    (
                        chunk
                        for chunk in repository_manifest.chunks
                        if chunk.logical_name == shard.shard_id
                    ),
                    key=lambda chunk: chunk.index,
                )
            )
            if tuple(chunk.index for chunk in chunks) != tuple(range(len(chunks))):
                raise IntegrityError(
                    "Repository checkpoint chunk indexes are not contiguous."
                )
            offset = 0
            digest = hashlib.sha256()
            for chunk in chunks:
                if chunk.offset != offset:
                    raise IntegrityError(
                        "Repository checkpoint chunk offsets are not contiguous."
                    )
                payload = self._repository.read_chunk(
                    repository_manifest,
                    chunk,
                    maximum_plaintext_bytes=shard.byte_count,
                )
                offset += len(payload)
                digest.update(payload)
            if offset != shard.byte_count or digest.hexdigest() != shard.payload_digest:
                raise IntegrityError("Repository checkpoint payload identity is invalid.")

    def _expire_job(self, job: _Job) -> None:
        if self._clock.now() >= job.expires_at and not job.state.terminal:
            candidate = self._copy_job(job)
            candidate.state = JobState.CANCELED
            candidate.finished_at = self._clock.now()
            candidate.run_record = self._run_record(
                candidate.job_id,
                candidate.submission,
                "canceled",
                candidate.checkpoint_ids[-1] if candidate.checkpoint_ids else None,
                attempt=candidate.attempt,
            )
            self._commit_job_candidate(job, candidate)

    def _assert_artifact_live(self, artifact: _Artifact) -> None:
        if self._clock.now() >= artifact.descriptor.expires_at:
            raise ArtifactExpired("Artifact retention period has expired.")

    def _authorize_artifact_egress(self, artifact: _Artifact) -> None:
        artifact.descriptor.rights.require_egress()
        if artifact.descriptor.classification == "cad":
            metadata = artifact.descriptor.cad
            if metadata is None:
                raise IntegrityError("CAD artifact is missing required egress metadata.")
            self._cad_policies.get(
                artifact.descriptor.tenant_id, CADEgressPolicy.deny_all()
            ).authorize(metadata)

    def _run_record(
        self,
        job_id: str,
        submission: JobSubmission,
        status: str,
        checkpoint_id: str | None = None,
        result: ProviderResult | None = None,
        *,
        diagnostic_ids: tuple[str, ...] = (),
        attempt: int,
    ) -> RunRecord:
        diagnostics = diagnostic_ids if result is None else result.diagnostic_ids
        return RunRecord(
            f"{job_id}:attempt-{attempt}:{status}",
            submission.analysis_plan.analysis_plan_id,
            submission.numeric_revision_id,
            submission.execution_plan.execution_plan_id,
            status,  # type: ignore[arg-type]
            result_ids=() if result is None else result.result_ids,
            diagnostic_ids=diagnostics,
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

    def _grant_token(self, descriptor: ArtifactDescriptor, expires_at: int) -> str:
        payload = json.dumps(
            {
                "artifact_id": descriptor.artifact_id,
                "classification": descriptor.classification,
                "content_sha256": descriptor.content_sha256,
                "expires_at": expires_at,
                "rights_binding_id": descriptor.rights.rights_binding_id,
                "tenant_id": descriptor.tenant_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return (
            payload.hex()
            + "."
            + hmac.new(self._artifact_secret, payload, hashlib.sha256).hexdigest()
        )

    def _verify_grant(self, value: str) -> tuple[str, str, int, str, str, str]:
        if type(value) is not str:
            raise IntegrityError("Artifact grant must be a string.")
        if len(value) > 4_096 or not value.isascii():
            raise IntegrityError("Artifact grant exceeds its encoded-byte limit.")
        try:
            encoded, signature = value.split(".", 1)
            if (
                not encoded
                or len(encoded) > 2_048
                or len(encoded) % 2
                or len(signature) != 64
            ):
                raise IntegrityError("Artifact grant encoding is invalid.")
            payload = bytes.fromhex(encoded)
            expected = hmac.new(
                self._artifact_secret, payload, hashlib.sha256
            ).hexdigest()
            if not hmac.compare_digest(signature, expected):
                raise IntegrityError("Artifact grant signature is invalid.")
            decoded = json.loads(payload.decode("utf-8"))
            if not isinstance(decoded, dict) or set(decoded) != {
                "artifact_id",
                "classification",
                "content_sha256",
                "expires_at",
                "rights_binding_id",
                "tenant_id",
            }:
                raise IntegrityError("Artifact grant payload is invalid.")
            artifact_id = decoded["artifact_id"]
            tenant_id = decoded["tenant_id"]
            expires_at = decoded["expires_at"]
            content_sha256 = decoded["content_sha256"]
            classification = decoded["classification"]
            rights_binding_id = decoded["rights_binding_id"]
            if (
                not isinstance(artifact_id, str)
                or not artifact_id
                or not isinstance(tenant_id, str)
                or not tenant_id
                or isinstance(expires_at, bool)
                or not isinstance(expires_at, int)
                or not isinstance(content_sha256, str)
                or len(content_sha256) != 64
                or not isinstance(classification, str)
                or classification
                not in {"scientific", "cad", "checkpoint", "diagnostic", "support"}
                or not isinstance(rights_binding_id, str)
                or len(rights_binding_id) != 64
            ):
                raise IntegrityError("Artifact grant payload is invalid.")
            return (
                artifact_id,
                tenant_id,
                expires_at,
                content_sha256,
                classification,
                rights_binding_id,
            )
        except (
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise IntegrityError("Artifact grant is malformed.") from error

    def _audit_template(
        self,
        principal: ValidatedPrincipal,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: Literal["allowed", "denied", "failed"],
        reason: str,
        request_id: str,
    ) -> AuditRecord:
        return AuditRecord(
            0,
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
            "",
            "",
        )

    def _commit_local_audit(self, record: AuditRecord, /) -> AuditRecord:
        previous = self._audit[-1].record_digest if self._audit else "0" * 64
        unsigned = replace(
            record,
            sequence=len(self._audit) + 1,
            previous_digest=previous,
            record_digest="",
        )
        return replace(unsigned, record_digest=self._audit_digest(unsigned))

    def _audit_event(
        self,
        principal: ValidatedPrincipal,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: Literal["allowed", "denied", "failed"],
        reason: str,
        request_id: str,
    ) -> AuditRecord:
        template = self._audit_template(
            principal,
            action,
            resource_type,
            resource_id,
            outcome,
            reason,
            request_id,
        )
        if self._durable_store is None:
            committed = self._commit_local_audit(template)
        else:
            with self._durable_store.transaction() as transaction:
                committed = transaction.append_audit(template)
        self._audit.append(committed)
        return committed

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


def _analysis_plan_record(plan: AnalysisPlan, /) -> dict[str, object]:
    return {
        "kind": "analysis-plan",
        "analysis_plan_id": plan.analysis_plan_id,
        "provider_plan_id": plan.provider_plan_id,
        "discretization_key": plan.discretization_key,
        "field_layout_ids": list(plan.field_layout_ids),
        "material_plan_id": plan.material_plan_id,
        "constraint_ids": list(plan.constraint_ids),
        "capability_ids": list(plan.capability_ids),
        "model_manifest_id": plan.model_manifest_id,
        "plan_fingerprint": plan.plan_fingerprint,
    }


def _analysis_plan_from_record(record: Mapping[str, object], /) -> AnalysisPlan:
    value = AnalysisPlan(
        str(record["analysis_plan_id"]),
        str(record["provider_plan_id"]),
        str(record["discretization_key"]),
        tuple(record["field_layout_ids"]),  # type: ignore[arg-type]
        material_plan_id=(
            None
            if record["material_plan_id"] is None
            else str(record["material_plan_id"])
        ),
        constraint_ids=tuple(record["constraint_ids"]),  # type: ignore[arg-type]
        capability_ids=tuple(record["capability_ids"]),  # type: ignore[arg-type]
        model_manifest_id=(
            None
            if record["model_manifest_id"] is None
            else str(record["model_manifest_id"])
        ),
    )
    if (
        record.get("kind") != "analysis-plan"
        or record.get("plan_fingerprint") != value.plan_fingerprint
    ):
        raise IntegrityError("Durable analysis plan identity is invalid.")
    return value


def _run_record_payload(record: RunRecord, /) -> dict[str, object]:
    return {
        "kind": "run-record",
        "run_id": record.run_id,
        "analysis_plan_id": record.analysis_plan_id,
        "numeric_revision_id": record.numeric_revision_id,
        "execution_plan_id": record.execution_plan_id,
        "status": record.status,
        "result_ids": list(record.result_ids),
        "diagnostic_ids": list(record.diagnostic_ids),
        "checkpoint_id": record.checkpoint_id,
        "record_id": record.record_id,
    }


def _run_record_from_payload(payload: Mapping[str, object], /) -> RunRecord:
    value = RunRecord(
        str(payload["run_id"]),
        str(payload["analysis_plan_id"]),
        str(payload["numeric_revision_id"]),
        str(payload["execution_plan_id"]),
        str(payload["status"]),  # type: ignore[arg-type]
        result_ids=tuple(payload["result_ids"]),  # type: ignore[arg-type]
        diagnostic_ids=tuple(payload["diagnostic_ids"]),  # type: ignore[arg-type]
        checkpoint_id=(
            None if payload["checkpoint_id"] is None else str(payload["checkpoint_id"])
        ),
    )
    if payload.get("kind") != "run-record" or payload.get("record_id") != value.record_id:
        raise IntegrityError("Durable run-record identity is invalid.")
    return value


def _checkpoint_payload(manifest: CheckpointManifest, /) -> dict[str, object]:
    return {
        "kind": "checkpoint-manifest",
        "checkpoint_id": manifest.checkpoint_id,
        "analysis_plan_id": manifest.analysis_plan_id,
        "numeric_revision_id": manifest.numeric_revision_id,
        "execution_plan_id": manifest.execution_plan_id,
        "shards": [
            {
                "shard_id": shard.shard_id,
                "payload_digest": shard.payload_digest,
                "byte_count": shard.byte_count,
                "layout_ids": list(shard.layout_ids),
                "metadata": [list(item) for item in shard.metadata],
                "shard_fingerprint": shard.shard_fingerprint,
            }
            for shard in manifest.shards
        ],
        "complete": manifest.complete,
        "parent_manifest_id": manifest.parent_manifest_id,
        "parent_checkpoint_id": manifest.parent_checkpoint_id,
        "diagnostic_ids": list(manifest.diagnostic_ids),
        "manifest_id": manifest.manifest_id,
    }


def _checkpoint_from_payload(payload: Mapping[str, object], /) -> CheckpointManifest:
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list):
        raise IntegrityError("Durable checkpoint shards must be a list.")
    shards: list[CheckpointShard] = []
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise IntegrityError("Durable checkpoint shard is malformed.")
        shard = CheckpointShard(
            str(raw["shard_id"]),
            str(raw["payload_digest"]),
            raw["byte_count"],  # type: ignore[arg-type]
            tuple(raw["layout_ids"]),  # type: ignore[arg-type]
            metadata=tuple(tuple(item) for item in raw["metadata"]),  # type: ignore[arg-type]
        )
        if raw.get("shard_fingerprint") != shard.shard_fingerprint:
            raise IntegrityError("Durable checkpoint shard identity is invalid.")
        shards.append(shard)
    complete = payload["complete"]
    if type(complete) is not bool:
        raise IntegrityError("Durable checkpoint completion flag is invalid.")
    value = CheckpointManifest(
        str(payload["checkpoint_id"]),
        str(payload["analysis_plan_id"]),
        str(payload["numeric_revision_id"]),
        str(payload["execution_plan_id"]),
        tuple(shards),
        complete=complete,
        parent_manifest_id=(
            None
            if payload["parent_manifest_id"] is None
            else str(payload["parent_manifest_id"])
        ),
        parent_checkpoint_id=(
            None
            if payload["parent_checkpoint_id"] is None
            else str(payload["parent_checkpoint_id"])
        ),
        diagnostic_ids=tuple(payload["diagnostic_ids"]),  # type: ignore[arg-type]
    )
    if (
        payload.get("kind") != "checkpoint-manifest"
        or payload.get("manifest_id") != value.manifest_id
    ):
        raise IntegrityError("Durable checkpoint manifest identity is invalid.")
    return value


def _secret_handle_payload(handle: SecretHandle, /) -> dict[str, object]:
    return {
        "handle_id": handle.handle_id,
        "tenant_id": handle.tenant_id,
        "created_at": handle.created_at,
        "key_version": handle.key_version,
        "scopes": sorted(getattr(handle, "scopes", ())),
        "expires_at": getattr(handle, "expires_at", None),
    }


def _secret_handle_from_payload(payload: Mapping[str, object], /) -> SecretHandle:
    scopes = payload["scopes"]
    expires_at = payload["expires_at"]
    if scopes or expires_at is not None:
        if not isinstance(scopes, list) or type(expires_at) is not int:
            raise IntegrityError("Durable scoped secret handle is malformed.")
        return ScopedSecretHandle(
            str(payload["handle_id"]),
            str(payload["tenant_id"]),
            frozenset(str(value) for value in scopes),
            payload["created_at"],  # type: ignore[arg-type]
            expires_at,
            str(payload["key_version"]),
        )
    return SecretHandle(
        str(payload["handle_id"]),
        str(payload["tenant_id"]),
        payload["created_at"],  # type: ignore[arg-type]
        str(payload["key_version"]),
    )


def _submission_payload(submission: JobSubmission, /) -> dict[str, object]:
    return {
        "kind": "job-submission",
        "analysis_plan": _analysis_plan_record(submission.analysis_plan),
        "execution_plan": submission.execution_plan.to_payload(),
        "numeric_revision_id": submission.numeric_revision_id,
        "profile_id": submission.profile_id,
        "parameters": dict(submission.parameters),
        "resources": submission.resources.to_payload(),
        "secret_handles": [
            _secret_handle_payload(handle) for handle in submission.secret_handles
        ],
        "retention_seconds": submission.retention_seconds,
        "request_id": submission.request_id,
        "resolved_run_spec": (
            None
            if submission.resolved_run_spec is None
            else submission.resolved_run_spec.to_record()
        ),
        "request_digest": submission.request_digest,
    }


def _artifact_rights_payload(rights: ArtifactRights, /) -> dict[str, object]:
    return {
        "scientific_artifact_id": rights.scientific_artifact_id,
        "rights_id": rights.rights_id,
        "use_policy_id": rights.use_policy_id,
        "license_id": rights.license_id,
        "source_uri": rights.source_uri,
        "attribution_id": rights.attribution_id,
        "content_sha256": rights.content_sha256,
        "byte_size": rights.byte_size,
        "classification": rights.classification,
        "allow_redistribution": rights.allow_redistribution,
        "allow_export": rights.allow_export,
        "redistribution_requested": rights.redistribution_requested,
        "export_requested": rights.export_requested,
        "rights_binding_id": rights.rights_binding_id,
    }


def _artifact_rights_from_payload(payload: Mapping[str, object], /) -> ArtifactRights:
    value = ArtifactRights(
        str(payload["scientific_artifact_id"]),
        str(payload["rights_id"]),
        str(payload["use_policy_id"]),
        str(payload["license_id"]),
        str(payload["source_uri"]),
        str(payload["attribution_id"]),
        str(payload["content_sha256"]),
        payload["byte_size"],  # type: ignore[arg-type]
        str(payload["classification"]),  # type: ignore[arg-type]
        payload["allow_redistribution"],  # type: ignore[arg-type]
        payload["allow_export"],  # type: ignore[arg-type]
        payload["redistribution_requested"],  # type: ignore[arg-type]
        payload["export_requested"],  # type: ignore[arg-type]
    )
    if payload.get("rights_binding_id") != value.rights_binding_id:
        raise IntegrityError("Durable artifact rights identity is invalid.")
    return value


def _artifact_descriptor_payload(descriptor: ArtifactDescriptor, /) -> dict[str, object]:
    return {
        "kind": "service-artifact-descriptor",
        "artifact_id": descriptor.artifact_id,
        "scientific_artifact_id": descriptor.scientific_artifact_id,
        "job_id": descriptor.job_id,
        "tenant_id": descriptor.tenant_id,
        "content_sha256": descriptor.content_sha256,
        "byte_size": descriptor.byte_size,
        "media_type": descriptor.media_type,
        "classification": descriptor.classification,
        "created_at": descriptor.created_at,
        "expires_at": descriptor.expires_at,
        "storage_generation": descriptor.storage_generation,
        "encryption": {
            "algorithm": descriptor.encryption.algorithm,
            "key_id": descriptor.encryption.key_id,
            "encrypted_at_rest": descriptor.encryption.encrypted_at_rest,
            "transport_protocol": descriptor.encryption.transport_protocol,
            "key_rotated_at": descriptor.encryption.key_rotated_at,
        },
        "rights": _artifact_rights_payload(descriptor.rights),
        "cad": (
            None
            if descriptor.cad is None
            else {
                "format": descriptor.cad.format,
                "destination_region": descriptor.cad.destination_region,
                "export_classification": descriptor.cad.export_classification,
                "approval_id": descriptor.cad.approval_id,
            }
        ),
    }


def _artifact_descriptor_from_payload(
    payload: Mapping[str, object], /
) -> ArtifactDescriptor:
    encryption = payload["encryption"]
    rights = payload["rights"]
    cad = payload["cad"]
    if (
        payload.get("kind") != "service-artifact-descriptor"
        or not isinstance(encryption, Mapping)
        or not isinstance(rights, Mapping)
        or (cad is not None and not isinstance(cad, Mapping))
    ):
        raise IntegrityError("Durable artifact descriptor is malformed.")
    if type(encryption["encrypted_at_rest"]) is not bool:
        raise IntegrityError("Durable artifact encryption flag is invalid.")
    return ArtifactDescriptor(
        str(payload["artifact_id"]),
        str(payload["scientific_artifact_id"]),
        str(payload["job_id"]),
        str(payload["tenant_id"]),
        str(payload["content_sha256"]),
        payload["byte_size"],  # type: ignore[arg-type]
        str(payload["media_type"]),
        str(payload["classification"]),  # type: ignore[arg-type]
        payload["created_at"],  # type: ignore[arg-type]
        payload["expires_at"],  # type: ignore[arg-type]
        str(payload["storage_generation"]),
        EncryptionMetadata(
            str(encryption["algorithm"]),
            str(encryption["key_id"]),
            encryption["encrypted_at_rest"],  # type: ignore[arg-type]
            str(encryption["transport_protocol"]),
            encryption["key_rotated_at"],  # type: ignore[arg-type]
        ),
        _artifact_rights_from_payload(rights),
        (
            None
            if cad is None
            else CADArtifactMetadata(
                str(cad["format"]),
                str(cad["destination_region"]),
                str(cad["export_classification"]),
                (None if cad["approval_id"] is None else str(cad["approval_id"])),
            )
        ),
    )


def _submission_from_payload(payload: Mapping[str, object], /) -> JobSubmission:
    if payload.get("kind") != "job-submission":
        raise IntegrityError("Durable job submission kind is invalid.")
    analysis = payload["analysis_plan"]
    execution = payload["execution_plan"]
    resources = payload["resources"]
    handles = payload["secret_handles"]
    resolved = payload["resolved_run_spec"]
    if (
        not isinstance(analysis, Mapping)
        or not isinstance(execution, Mapping)
        or not isinstance(resources, Mapping)
        or not isinstance(handles, list)
        or (resolved is not None and not isinstance(resolved, Mapping))
    ):
        raise IntegrityError("Durable job submission is malformed.")
    value = JobSubmission(
        _analysis_plan_from_record(analysis),
        ExecutionPlan.from_payload(execution),
        str(payload["numeric_revision_id"]),
        str(payload["profile_id"]),
        payload["parameters"],  # type: ignore[arg-type]
        ResourceRequest.from_payload(resources),
        tuple(
            _secret_handle_from_payload(handle)
            for handle in handles
            if isinstance(handle, Mapping)
        ),
        payload["retention_seconds"],  # type: ignore[arg-type]
        str(payload["request_id"]),
        (None if resolved is None else ResolvedRunSpec.from_record(resolved)),
    )
    if (
        len(value.secret_handles) != len(handles)
        or payload.get("request_digest") != value.request_digest
    ):
        raise IntegrityError("Durable job submission identity is invalid.")
    return value


def _failure_payload(value: FailureEvidence | None, /) -> dict[str, object] | None:
    if value is None:
        return None
    return {
        "code": value.code,
        "exception_type": value.exception_type,
        "message": value.message,
        "retryable": value.retryable,
        "attempt": value.attempt,
        "diagnostic_ids": list(value.diagnostic_ids),
    }


def _failure_from_payload(payload: object, /) -> FailureEvidence | None:
    if payload is None:
        return None
    if not isinstance(payload, Mapping) or type(payload["retryable"]) is not bool:
        raise IntegrityError("Durable failure evidence is malformed.")
    return FailureEvidence(
        str(payload["code"]),
        str(payload["exception_type"]),
        str(payload["message"]),
        payload["retryable"],
        payload["attempt"],  # type: ignore[arg-type]
        tuple(payload["diagnostic_ids"]),  # type: ignore[arg-type]
    )


def _required_nonnegative_int(value: object, name: str, /) -> int:
    if type(value) is not int or value < 0:
        raise IntegrityError(f"Durable {name} must be a nonnegative integer.")
    return value


def _optional_nonnegative_int(value: object, name: str, /) -> int | None:
    if value is None:
        return None
    return _required_nonnegative_int(value, name)


__all__ = [
    "ExecutionContext",
    "ExecutionProvider",
    "InProcessReferenceService",
    "ProviderBinding",
    "SupportDependencyAdmitter",
    "ReleaseIndexDependencyAdmitter",
]
