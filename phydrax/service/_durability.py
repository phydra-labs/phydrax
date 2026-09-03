#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Transactional persistence primitives for the commercial service boundary."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

from ._contracts import (
    AuditRecord,
    IntegrityError,
    JobState,
    QuotaExceeded,
    ResourceRequest,
    TenantQuota,
    TenantUsage,
)


_ZERO_DIGEST = "0" * 64


def _canonical_json(value: object, /) -> str:
    return json.dumps(
        value, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def _identifier(value: str, name: str, /) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
    ):
        raise ValueError(f"{name} must be a nonempty canonical identifier.")
    return value


_CREDENTIAL_FIELD = re.compile(
    r"(?:^|[_\-.])(?:authorization|credential|password|secret|private[_-]?key|"
    r"access[_-]?token|refresh[_-]?token)(?:$|[_\-.])",
    re.IGNORECASE,
)
_CREDENTIAL_VALUE = (
    re.compile(r"(?i)^bearer\s+\S+$"),
    re.compile(r"^-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)


def _assert_no_credentials(value: object, /, *, field_name: str = "") -> None:
    normalized_name = field_name.casefold().replace("-", "_").replace(".", "_")
    is_handle_reference = normalized_name.startswith(
        "secret_handle_"
    ) and normalized_name.endswith(("id", "ids"))
    if field_name and _CREDENTIAL_FIELD.search(field_name) and not is_handle_reference:
        raise ValueError("Durable service records cannot contain credential fields.")
    if isinstance(value, str) and any(
        pattern.search(value) for pattern in _CREDENTIAL_VALUE
    ):
        raise ValueError("Durable service records cannot contain credential values.")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_credentials(item, field_name=str(key))
    elif isinstance(value, (tuple, list)):
        for item in value:
            _assert_no_credentials(item, field_name=field_name)


@dataclass(frozen=True, slots=True)
class DurableJobRecord:
    """Minimal durable orchestration state; scientific payloads stay in repositories."""

    job_id: str
    tenant_id: str
    request_id: str
    request_digest: str
    state: JobState
    attempt: int
    payload: Mapping[str, object]
    submitted_at: int
    updated_at: int
    lease_expires_at: int | None = None
    scheduler_job_id: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        for value, name in ((self.job_id, "job_id"), (self.tenant_id, "tenant_id")):
            _identifier(value, name)
        if self.request_id:
            _identifier(self.request_id, "request_id")
        if len(self.request_digest) != 64 or any(
            c not in "0123456789abcdef" for c in self.request_digest
        ):
            raise ValueError("request_digest must be a lowercase SHA-256 digest.")
        if self.attempt <= 0 or self.version <= 0:
            raise ValueError("attempt and version must be positive.")
        if self.submitted_at < 0 or self.updated_at < self.submitted_at:
            raise ValueError("Job timestamps are invalid.")
        if self.lease_expires_at is not None and self.lease_expires_at < 0:
            raise ValueError("lease_expires_at must be nonnegative.")
        payload = json.loads(_canonical_json(dict(self.payload)))
        if not isinstance(payload, dict):
            raise TypeError("Job payload must be a JSON object.")
        object.__setattr__(self, "payload", MappingProxyType(payload))
        _assert_no_credentials(payload)


@dataclass(frozen=True, slots=True)
class OutboxMessage:
    message_id: str
    tenant_id: str
    topic: str
    idempotency_key: str
    payload: Mapping[str, object]
    created_at: int
    available_at: int
    attempts: int = 0
    lease_owner: str | None = None
    lease_expires_at: int | None = None
    delivered_at: int | None = None

    def __post_init__(self) -> None:
        for value, name in (
            (self.message_id, "message_id"),
            (self.tenant_id, "tenant_id"),
            (self.topic, "topic"),
            (self.idempotency_key, "idempotency_key"),
        ):
            _identifier(value, name)
        if min(self.created_at, self.available_at, self.attempts) < 0:
            raise ValueError("Outbox timestamps and attempts must be nonnegative.")
        payload = json.loads(_canonical_json(dict(self.payload)))
        if not isinstance(payload, dict):
            raise TypeError("Outbox payload must be a JSON object.")
        object.__setattr__(self, "payload", MappingProxyType(payload))

        _assert_no_credentials(payload)


class ServiceTransaction(Protocol):
    def insert_job(self, record: DurableJobRecord, /) -> DurableJobRecord: ...
    def get_job(self, tenant_id: str, job_id: str, /) -> DurableJobRecord | None: ...
    def get_job_by_request(
        self, tenant_id: str, request_id: str, /
    ) -> DurableJobRecord | None: ...
    def update_job(
        self, record: DurableJobRecord, /, *, expected_version: int
    ) -> DurableJobRecord: ...
    def enqueue(self, message: OutboxMessage, /) -> OutboxMessage: ...
    def append_audit(self, record: AuditRecord, /) -> AuditRecord: ...
    def reserve_quota(
        self,
        tenant_id: str,
        job_id: str,
        resources: ResourceRequest,
        quota: TenantQuota,
        /,
    ) -> None: ...
    def release_quota(self, tenant_id: str, job_id: str, /) -> None: ...


class DurableJobProvider(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[ServiceTransaction]: ...

    def recover_stale_attempts(self, now: int, /) -> tuple[DurableJobRecord, ...]: ...


class DurableAuditProvider(Protocol):
    def audit_records(self, tenant_id: str, /) -> tuple[AuditRecord, ...]: ...

    def verify_audit_chain(self) -> None: ...


class DurableOutboxProvider(Protocol):
    def claim_outbox(
        self,
        owner: str,
        now: int,
        /,
        *,
        lease_seconds: int = 60,
        limit: int = 100,
    ) -> tuple[OutboxMessage, ...]: ...

    def acknowledge_outbox(
        self, owner: str, message_id: str, delivered_at: int, /
    ) -> None: ...

    def release_outbox(
        self, owner: str, message_id: str, available_at: int, /
    ) -> None: ...


class DurableServiceStore(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[ServiceTransaction]: ...
    def claim_outbox(
        self, owner: str, now: int, /, *, lease_seconds: int = 60, limit: int = 100
    ) -> tuple[OutboxMessage, ...]: ...
    def acknowledge_outbox(
        self, owner: str, message_id: str, delivered_at: int, /
    ) -> None: ...
    def release_outbox(
        self, owner: str, message_id: str, available_at: int, /
    ) -> None: ...
    def recover_stale_attempts(self, now: int, /) -> tuple[DurableJobRecord, ...]: ...
    def quota_usage(self, tenant_id: str, /) -> TenantUsage: ...
    def reconcile_quota(
        self, tenant_id: str, active_job_ids: Sequence[str], /
    ) -> TenantUsage: ...
    def audit_records(self, tenant_id: str, /) -> tuple[AuditRecord, ...]: ...
    def verify_audit_chain(self) -> None: ...


class _SQLiteTransaction:
    def __init__(self, connection: sqlite3.Connection):
        self._connection = connection

    def insert_job(self, record: DurableJobRecord, /) -> DurableJobRecord:
        existing = (
            self.get_job_by_request(record.tenant_id, record.request_id)
            if record.request_id
            else None
        )
        if existing is not None:
            if existing.request_digest != record.request_digest:
                raise IntegrityError(
                    "An idempotency key was reused for a different submission."
                )
            return existing
        try:
            self._connection.execute(
                "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _job_values(record),
            )
        except sqlite3.IntegrityError as error:
            if record.request_id:
                existing = self.get_job_by_request(record.tenant_id, record.request_id)
                if (
                    existing is not None
                    and existing.request_digest == record.request_digest
                ):
                    return existing
            raise IntegrityError(
                "A durable job identity conflicts with an existing record."
            ) from error
        return record

    def get_job(self, tenant_id: str, job_id: str, /) -> DurableJobRecord | None:
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE tenant_id = ? AND job_id = ?", (tenant_id, job_id)
        ).fetchone()
        return None if row is None else _job_from_row(row)

    def get_job_by_request(
        self, tenant_id: str, request_id: str, /
    ) -> DurableJobRecord | None:
        if not request_id:
            return None
        row = self._connection.execute(
            "SELECT * FROM jobs WHERE tenant_id = ? AND request_id = ?",
            (tenant_id, request_id),
        ).fetchone()
        return None if row is None else _job_from_row(row)

    def update_job(
        self, record: DurableJobRecord, /, *, expected_version: int
    ) -> DurableJobRecord:
        if record.version != expected_version + 1:
            raise ValueError("Updated job version must advance exactly once.")
        cursor = self._connection.execute(
            """UPDATE jobs SET request_id=?, request_digest=?, state=?, attempt=?, payload_json=?,
               submitted_at=?, updated_at=?, lease_expires_at=?, scheduler_job_id=?, version=?
               WHERE tenant_id=? AND job_id=? AND version=?""",
            (
                record.request_id,
                record.request_digest,
                record.state.value,
                record.attempt,
                _canonical_json(dict(record.payload)),
                record.submitted_at,
                record.updated_at,
                record.lease_expires_at,
                record.scheduler_job_id,
                record.version,
                record.tenant_id,
                record.job_id,
                expected_version,
            ),
        )
        if cursor.rowcount != 1:
            raise IntegrityError("Durable job optimistic-concurrency check failed.")
        return record

    def enqueue(self, message: OutboxMessage, /) -> OutboxMessage:
        self._connection.execute(
            """INSERT INTO outbox VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(tenant_id, topic, idempotency_key) DO NOTHING""",
            (
                message.message_id,
                message.tenant_id,
                message.topic,
                message.idempotency_key,
                _canonical_json(dict(message.payload)),
                message.created_at,
                message.available_at,
                message.attempts,
                message.lease_owner,
                message.lease_expires_at,
                message.delivered_at,
            ),
        )
        row = self._connection.execute(
            "SELECT * FROM outbox WHERE tenant_id=? AND topic=? AND idempotency_key=?",
            (message.tenant_id, message.topic, message.idempotency_key),
        ).fetchone()
        assert row is not None
        existing = _outbox_from_row(row)
        if dict(existing.payload) != dict(message.payload):
            raise IntegrityError(
                "An outbox idempotency key was reused for a different payload."
            )
        return existing

    def append_audit(self, record: AuditRecord, /) -> AuditRecord:
        previous_row = self._connection.execute(
            "SELECT sequence, record_digest FROM audit ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        sequence = 1 if previous_row is None else int(previous_row[0]) + 1
        previous = _ZERO_DIGEST if previous_row is None else str(previous_row[1])
        unsigned = replace(
            record, sequence=sequence, previous_digest=previous, record_digest=""
        )
        signed = replace(unsigned, record_digest=audit_digest(unsigned))
        self._connection.execute(
            "INSERT INTO audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(getattr(signed, name) for name in _AUDIT_FIELDS),
        )
        return signed

    def reserve_quota(
        self,
        tenant_id: str,
        job_id: str,
        resources: ResourceRequest,
        quota: TenantQuota,
        /,
    ) -> None:
        existing = self._connection.execute(
            "SELECT cpu_cores, memory_bytes, gpu_count FROM quota_reservations WHERE tenant_id=? AND job_id=?",
            (tenant_id, job_id),
        ).fetchone()
        if existing is not None:
            if tuple(existing) != (
                resources.cpu_cores,
                resources.memory_bytes,
                resources.gpu_count,
            ):
                raise IntegrityError(
                    "A job quota reservation changed across an idempotent retry."
                )
            return
        usage = self._quota_usage(tenant_id)
        if (
            usage.active_jobs + 1 > quota.active_jobs
            or usage.cpu_cores + resources.cpu_cores > quota.cpu_cores
            or usage.memory_bytes + resources.memory_bytes > quota.memory_bytes
            or usage.gpu_count + resources.gpu_count > quota.gpu_count
        ):
            raise QuotaExceeded("Tenant active execution quota would be exceeded.")
        self._connection.execute(
            "INSERT INTO quota_reservations VALUES (?, ?, ?, ?, ?)",
            (
                tenant_id,
                job_id,
                resources.cpu_cores,
                resources.memory_bytes,
                resources.gpu_count,
            ),
        )

    def release_quota(self, tenant_id: str, job_id: str, /) -> None:
        self._connection.execute(
            "DELETE FROM quota_reservations WHERE tenant_id=? AND job_id=?",
            (tenant_id, job_id),
        )

    def _quota_usage(self, tenant_id: str) -> TenantUsage:
        row = self._connection.execute(
            """SELECT COUNT(*), COALESCE(SUM(cpu_cores),0), COALESCE(SUM(memory_bytes),0),
               COALESCE(SUM(gpu_count),0) FROM quota_reservations WHERE tenant_id=?""",
            (tenant_id,),
        ).fetchone()
        assert row is not None
        return TenantUsage(int(row[0]), int(row[1]), int(row[2]), int(row[3]), 0)


class SQLiteServiceStore:
    """SQLite reference store with atomic job, quota, audit, and outbox commits."""

    def __init__(self, path: str | Path = ":memory:", /):
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(path), isolation_level=None, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def _create_schema(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    request_digest TEXT NOT NULL,
                    state TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    submitted_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL,
                    lease_expires_at INTEGER,
                    scheduler_job_id TEXT,
                    version INTEGER NOT NULL
                );
                CREATE UNIQUE INDEX IF NOT EXISTS jobs_request_identity
                    ON jobs(tenant_id, request_id) WHERE request_id <> '';
                CREATE TABLE IF NOT EXISTS outbox (
                    message_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    topic TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    available_at INTEGER NOT NULL,
                    attempts INTEGER NOT NULL,
                    lease_owner TEXT,
                    lease_expires_at INTEGER,
                    delivered_at INTEGER,
                    UNIQUE(tenant_id, topic, idempotency_key)
                );
                CREATE TABLE IF NOT EXISTS audit (
                    sequence INTEGER PRIMARY KEY,
                    occurred_at INTEGER NOT NULL,
                    event_id TEXT NOT NULL UNIQUE,
                    principal_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT NOT NULL,
                    outcome TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    previous_digest TEXT NOT NULL,
                    record_digest TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS quota_reservations (
                    tenant_id TEXT NOT NULL,
                    job_id TEXT NOT NULL,
                    cpu_cores INTEGER NOT NULL,
                    memory_bytes INTEGER NOT NULL,
                    gpu_count INTEGER NOT NULL,
                    PRIMARY KEY(tenant_id, job_id)
                );
                CREATE TABLE IF NOT EXISTS scheduler_idempotency (
                    provider_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    scheduler_job_id TEXT NOT NULL,
                    PRIMARY KEY(provider_id, idempotency_key)
                );
                """
            )

    @contextmanager
    def transaction(self) -> Iterator[_SQLiteTransaction]:
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield _SQLiteTransaction(self._connection)
            except BaseException:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    def lookup(self, provider_id: str, key: str, /) -> str | None:
        _identifier(provider_id, "provider_id")
        _identifier(key, "idempotency key")
        with self._lock:
            row = self._connection.execute(
                "SELECT scheduler_job_id FROM scheduler_idempotency "
                "WHERE provider_id=? AND idempotency_key=?",
                (provider_id, key),
            ).fetchone()
        return None if row is None else str(row[0])

    def record(
        self,
        provider_id: str,
        key: str,
        scheduler_job_id: str,
        /,
    ) -> str:
        _identifier(provider_id, "provider_id")
        _identifier(key, "idempotency key")
        _identifier(scheduler_job_id, "scheduler_job_id")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                self._connection.execute(
                    "INSERT INTO scheduler_idempotency VALUES (?, ?, ?) "
                    "ON CONFLICT(provider_id, idempotency_key) DO NOTHING",
                    (provider_id, key, scheduler_job_id),
                )
                row = self._connection.execute(
                    "SELECT scheduler_job_id FROM scheduler_idempotency "
                    "WHERE provider_id=? AND idempotency_key=?",
                    (provider_id, key),
                ).fetchone()
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
        assert row is not None
        existing = str(row[0])
        if existing != scheduler_job_id:
            raise IntegrityError(
                "Scheduler idempotency key resolved to conflicting jobs."
            )
        return existing

    def claim_outbox(
        self,
        owner: str,
        now: int,
        /,
        *,
        lease_seconds: int = 60,
        limit: int = 100,
    ) -> tuple[OutboxMessage, ...]:
        _identifier(owner, "owner")
        if now < 0 or lease_seconds <= 0 or limit <= 0:
            raise ValueError("Outbox claim time, lease, and limit are invalid.")
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                rows = self._connection.execute(
                    """SELECT * FROM outbox WHERE delivered_at IS NULL AND available_at<=?
                       AND (lease_expires_at IS NULL OR lease_expires_at<=?)
                       ORDER BY created_at, message_id LIMIT ?""",
                    (now, now, limit),
                ).fetchall()
                identifiers = tuple(str(row["message_id"]) for row in rows)
                if identifiers:
                    placeholders = ",".join("?" for _ in identifiers)
                    self._connection.execute(
                        "UPDATE outbox SET lease_owner=?, lease_expires_at=?, "
                        "attempts=attempts+1 WHERE message_id IN "
                        f"({placeholders})",
                        (owner, now + lease_seconds, *identifiers),
                    )
                self._connection.commit()
            except BaseException:
                self._connection.rollback()
                raise
            if not identifiers:
                return ()
            placeholders = ",".join("?" for _ in identifiers)
            claimed = self._connection.execute(
                f"SELECT * FROM outbox WHERE message_id IN ({placeholders}) ORDER BY created_at, message_id",
                identifiers,
            ).fetchall()
            return tuple(_outbox_from_row(row) for row in claimed)

    def acknowledge_outbox(
        self, owner: str, message_id: str, delivered_at: int, /
    ) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE outbox SET delivered_at=?, lease_owner=NULL, lease_expires_at=NULL
                   WHERE message_id=? AND lease_owner=? AND delivered_at IS NULL""",
                (delivered_at, message_id, owner),
            )
            if cursor.rowcount != 1:
                raise IntegrityError(
                    "Outbox acknowledgement does not own an active lease."
                )

    def release_outbox(self, owner: str, message_id: str, available_at: int, /) -> None:
        with self._lock:
            cursor = self._connection.execute(
                """UPDATE outbox SET available_at=?, lease_owner=NULL, lease_expires_at=NULL
                   WHERE message_id=? AND lease_owner=? AND delivered_at IS NULL""",
                (available_at, message_id, owner),
            )
            if cursor.rowcount != 1:
                raise IntegrityError("Outbox release does not own an active lease.")

    def recover_stale_attempts(self, now: int, /) -> tuple[DurableJobRecord, ...]:
        recovered: list[DurableJobRecord] = []
        with self.transaction() as transaction:
            rows = self._connection.execute(
                "SELECT * FROM jobs WHERE state=? "
                "AND lease_expires_at IS NOT NULL "
                "AND lease_expires_at<=? ORDER BY job_id",
                (JobState.RUNNING.value, now),
            ).fetchall()
            for row in rows:
                stale = _job_from_row(row)
                updated = replace(
                    stale,
                    state=JobState.QUEUED,
                    attempt=stale.attempt + 1,
                    updated_at=now,
                    lease_expires_at=None,
                    scheduler_job_id=None,
                    version=stale.version + 1,
                )
                transaction.update_job(updated, expected_version=stale.version)
                transaction.enqueue(
                    OutboxMessage(
                        f"recover:{updated.job_id}:{updated.attempt}",
                        updated.tenant_id,
                        "job.dispatch",
                        f"{updated.job_id}:{updated.attempt}",
                        {"attempt": updated.attempt, "job_id": updated.job_id},
                        now,
                        now,
                    )
                )
                recovered.append(updated)
        return tuple(recovered)

    def quota_usage(self, tenant_id: str, /) -> TenantUsage:
        with self._lock:
            return _SQLiteTransaction(self._connection)._quota_usage(tenant_id)

    def reconcile_quota(
        self, tenant_id: str, active_job_ids: Sequence[str], /
    ) -> TenantUsage:
        active = tuple(
            dict.fromkeys(_identifier(value, "job_id") for value in active_job_ids)
        )
        with self.transaction():
            if active:
                placeholders = ",".join("?" for _ in active)
                self._connection.execute(
                    f"DELETE FROM quota_reservations WHERE tenant_id=? AND job_id NOT IN ({placeholders})",
                    (tenant_id, *active),
                )
            else:
                self._connection.execute(
                    "DELETE FROM quota_reservations WHERE tenant_id=?", (tenant_id,)
                )
        return self.quota_usage(tenant_id)

    def audit_records(self, tenant_id: str, /) -> tuple[AuditRecord, ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM audit WHERE tenant_id=? ORDER BY sequence", (tenant_id,)
            ).fetchall()
        return tuple(AuditRecord(*(row[name] for name in _AUDIT_FIELDS)) for row in rows)

    def verify_audit_chain(self) -> None:
        previous = _ZERO_DIGEST
        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM audit ORDER BY sequence"
            ).fetchall()
        for expected_sequence, row in enumerate(rows, 1):
            record = AuditRecord(*(row[name] for name in _AUDIT_FIELDS))
            if record.sequence != expected_sequence or record.previous_digest != previous:
                raise IntegrityError("Audit chain ordering is invalid.")
            if record.record_digest != audit_digest(record):
                raise IntegrityError("Audit chain digest is invalid.")
            previous = record.record_digest


class OutboxHandler(Protocol):
    def __call__(self, message: OutboxMessage, /) -> None: ...


@dataclass(frozen=True, slots=True)
class OutboxDispatchReport:
    delivered_message_ids: tuple[str, ...]
    failed_message_ids: tuple[str, ...]


class OutboxDispatcher:
    """Explicit pull dispatcher implementing replay-safe at-least-once delivery."""

    def __init__(
        self,
        store: DurableOutboxProvider,
        handlers: Mapping[str, OutboxHandler],
        /,
        *,
        retry_delay_seconds: int = 30,
    ):
        if not handlers or any(not topic for topic in handlers):
            raise ValueError("Outbox dispatcher requires nonempty topic handlers.")
        if retry_delay_seconds <= 0:
            raise ValueError("Outbox retry delay must be positive.")
        self._store = store
        self._handlers = dict(handlers)
        self._retry_delay = retry_delay_seconds

    def dispatch_once(
        self,
        owner: str,
        now: int,
        /,
        *,
        lease_seconds: int = 60,
        limit: int = 100,
    ) -> OutboxDispatchReport:
        delivered: list[str] = []
        failed: list[str] = []
        messages = self._store.claim_outbox(
            owner,
            now,
            lease_seconds=lease_seconds,
            limit=limit,
        )
        for message in messages:
            handler = self._handlers.get(message.topic)
            try:
                if handler is None:
                    raise LookupError("No handler is registered for the outbox topic.")
                handler(message)
            except Exception:
                self._store.release_outbox(
                    owner,
                    message.message_id,
                    now + self._retry_delay,
                )
                failed.append(message.message_id)
            else:
                self._store.acknowledge_outbox(
                    owner,
                    message.message_id,
                    now,
                )
                delivered.append(message.message_id)
        return OutboxDispatchReport(tuple(delivered), tuple(failed))


_AUDIT_FIELDS = (
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
    "record_digest",
)


def audit_digest(record: AuditRecord, /) -> str:
    payload = {name: getattr(record, name) for name in _AUDIT_FIELDS[:-1]}
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _job_values(record: DurableJobRecord) -> tuple[object, ...]:
    return (
        record.job_id,
        record.tenant_id,
        record.request_id,
        record.request_digest,
        record.state.value,
        record.attempt,
        _canonical_json(dict(record.payload)),
        record.submitted_at,
        record.updated_at,
        record.lease_expires_at,
        record.scheduler_job_id,
        record.version,
    )


def _job_from_row(row: sqlite3.Row) -> DurableJobRecord:
    return DurableJobRecord(
        str(row["job_id"]),
        str(row["tenant_id"]),
        str(row["request_id"]),
        str(row["request_digest"]),
        JobState(str(row["state"])),
        int(row["attempt"]),
        json.loads(str(row["payload_json"])),
        int(row["submitted_at"]),
        int(row["updated_at"]),
        None if row["lease_expires_at"] is None else int(row["lease_expires_at"]),
        None if row["scheduler_job_id"] is None else str(row["scheduler_job_id"]),
        int(row["version"]),
    )


def _outbox_from_row(row: sqlite3.Row) -> OutboxMessage:
    return OutboxMessage(
        str(row["message_id"]),
        str(row["tenant_id"]),
        str(row["topic"]),
        str(row["idempotency_key"]),
        json.loads(str(row["payload_json"])),
        int(row["created_at"]),
        int(row["available_at"]),
        int(row["attempts"]),
        None if row["lease_owner"] is None else str(row["lease_owner"]),
        None if row["lease_expires_at"] is None else int(row["lease_expires_at"]),
        None if row["delivered_at"] is None else int(row["delivered_at"]),
    )


# Explicit local-reference spelling retained beside the implementation name.
LocalTransactionalServiceStore = SQLiteServiceStore


__all__ = [
    "DurableJobRecord",
    "DurableAuditProvider",
    "DurableJobProvider",
    "DurableOutboxProvider",
    "DurableServiceStore",
    "LocalTransactionalServiceStore",
    "OutboxDispatcher",
    "OutboxDispatchReport",
    "OutboxHandler",
    "OutboxMessage",
    "SQLiteServiceStore",
    "ServiceTransaction",
    "audit_digest",
]
