#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from phydrax.execution import ExecutionPlan, ResourceRequest
from phydrax.lifecycle import AnalysisPlan, CheckpointManifest, CheckpointShard
from phydrax.lifecycle._provenance import (
    create_build_provenance,
    digest_paths,
    generate_spdx_sbom,
    InstalledPackage,
    spdx_json,
)
from phydrax.lifecycle._repository import (
    HPCFilesystemProfile,
    POSIXArtifactRepository,
    POSIXRepositoryPolicy,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.service._auth import (
    HMACOIDCTokenValidator,
    HMACSigningKey,
    OIDCConfiguration,
    ScopeTenantAuthorizer,
)
from phydrax.service._contracts import (
    ArtifactRights,
    AuditRecord,
    AuthenticationError,
    AuthorizationError,
    CADArtifactMetadata,
    IntegrityError,
    JobState,
    JobSubmission,
    ProfileUnavailable,
    ProviderResult,
    ResourceNotFound,
    TenantQuota,
    ValidatedPrincipal,
)
from phydrax.service._durability import (
    DurableJobRecord,
    OutboxDispatcher,
    OutboxMessage,
    SQLiteServiceStore,
)
from phydrax.service._identity import (
    HTTPSJWKSProvider,
    JSONWebKeySet,
    MTLSCertificatePolicy,
    OIDCJWKSTokenValidator,
    StaticJWKSProvider,
    WorkloadCertificate,
)
from phydrax.service._observability import (
    create_support_bundle,
    HostTelemetryCollector,
    HostTelemetryPolicy,
    HostTelemetrySnapshot,
    PrivacyClassification,
    SecretRedactor,
    SupportBundlePolicy,
    TelemetryDatum,
)
from phydrax.service._runtime import InProcessReferenceService
from phydrax.service._schedulers import (
    CommandResult,
    HTTPResponse,
    KubernetesJobSpec,
    KubernetesScheduler,
    SchedulerState,
    SlurmJobSpec,
    SlurmScheduler,
    SubprocessCommandExecutor,
)
from phydrax.service._security import (
    Ed25519Signer,
    Ed25519Verifier,
    KMSSigner,
    KMSVerifier,
    LocalSecretHandleBroker,
    SigningKeyTrustRecord,
    SigningTrustStore,
)


class _Clock:
    def __init__(self, now: int):
        self.value = now

    def now(self) -> int:
        return self.value


class _ServiceValidator:
    def validate(self, token: str, /) -> ValidatedPrincipal:
        return ValidatedPrincipal(
            "subject",
            token,
            "issuer",
            "audience",
            "client",
            f"token-{token}",
            frozenset(
                {
                    "service:artifact:fetch",
                    "service:artifact:grant",
                    "service:artifact:write",
                    "service:cancel",
                    "service:execute",
                    "service:restart",
                    "service:status",
                    "service:submit",
                    "service:usage",
                }
            ),
            0,
            100,
        )


def _service_submission(request_id: str) -> JobSubmission:
    return JobSubmission(
        AnalysisPlan(
            "analysis",
            "provider-plan",
            "discretization",
            ("layout",),
        ),
        ExecutionPlan("execution", "cpu", "float64", "direct"),
        "revision",
        "profile",
        {"problem": "durable"},
        ResourceRequest(1, 1024),
        request_id=request_id,
    )


class _Executor:
    def __init__(self, results: list[CommandResult]):
        self.results = results
        self.argv: list[tuple[str, ...]] = []
        self.stdin: list[bytes | None] = []

    def run(
        self, argv: tuple[str, ...], /, *, stdin: bytes | None = None
    ) -> CommandResult:
        self.stdin.append(stdin)
        self.argv.append(argv)
        return self.results.pop(0)


def test_transaction_rollback_outbox_idempotency_audit_and_quota_recovery():
    store = SQLiteServiceStore()
    quota = TenantQuota(2, 2, 4096, 0, 1024)
    resources = ResourceRequest(1, 1024)
    job = DurableJobRecord(
        "job", "tenant", "request", "0" * 64, JobState.QUEUED, 1, {}, 1, 1
    )
    message = OutboxMessage(
        "message", "tenant", "dispatch", "job:1", {"job_id": "job"}, 1, 1
    )

    with pytest.raises(RuntimeError, match="rollback"):
        with store.transaction() as transaction:
            transaction.insert_job(job)
            transaction.reserve_quota("tenant", "job", resources, quota)
            transaction.enqueue(message)
            raise RuntimeError("rollback")
    with store.transaction() as transaction:
        assert transaction.get_job("tenant", "job") is None
        transaction.insert_job(job)
        assert transaction.insert_job(job) == job
        transaction.reserve_quota("tenant", "job", resources, quota)
        transaction.enqueue(message)
        transaction.append_audit(
            AuditRecord(
                0,
                2,
                "event",
                "subject",
                "tenant",
                "submit",
                "job",
                "job",
                "allowed",
                "queued",
                "request",
                "",
                "",
            )
        )
        running = replace(
            job, state=JobState.RUNNING, updated_at=2, lease_expires_at=3, version=2
        )
        transaction.update_job(running, expected_version=1)

    claimed = store.claim_outbox("worker", 2)
    assert claimed[0].message_id == "message"
    assert store.claim_outbox("other-worker", 2) == ()
    recovered = store.recover_stale_attempts(3)
    assert recovered[0].state is JobState.QUEUED
    assert recovered[0].attempt == 2
    assert store.quota_usage("tenant").active_jobs == 1
    store.verify_audit_chain()
    assert store.reconcile_quota("tenant", ()).active_jobs == 0


def test_sqlite_store_rejects_symlink_database_path(tmp_path):
    target = tmp_path / "target.sqlite"
    target.touch()
    link = tmp_path / "service.sqlite"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        SQLiteServiceStore(link)


def test_sqlite_store_detects_final_path_swap_during_connect(tmp_path, monkeypatch):
    import phydrax.service._durability as durability

    path = tmp_path / "service.sqlite"
    target = tmp_path / "target.sqlite"
    target.touch()
    real_connect = durability.sqlite3.connect

    def swapping_connect(value, *args, **kwargs):
        path.rename(tmp_path / "admitted.sqlite")
        path.symlink_to(target)
        return real_connect(value, *args, **kwargs)

    monkeypatch.setattr(durability.sqlite3, "connect", swapping_connect)
    with pytest.raises(ValueError, match="changed during connection"):
        SQLiteServiceStore(path)


def test_sqlite_store_detects_parent_swap_during_connect(tmp_path, monkeypatch):
    import phydrax.service._durability as durability

    parent = tmp_path / "database"
    parent.mkdir()
    path = parent / "service.sqlite"
    target = tmp_path / "target.sqlite"
    target.touch()
    real_connect = durability.sqlite3.connect

    def swapping_connect(value, *args, **kwargs):
        parent.rename(tmp_path / "admitted-database")
        parent.mkdir()
        path.symlink_to(target)
        return real_connect(value, *args, **kwargs)

    monkeypatch.setattr(durability.sqlite3, "connect", swapping_connect)
    with pytest.raises(ValueError, match="changed during connection"):
        SQLiteServiceStore(path)


def test_sqlite_connection_remains_bound_through_swap_and_restore(tmp_path, monkeypatch):
    import phydrax.service._durability as durability

    path = tmp_path / "service.sqlite"
    target = tmp_path / "target.sqlite"
    target.touch()
    admitted = tmp_path / "admitted.sqlite"
    real_connect = durability.sqlite3.connect

    def swapping_connect(value, *args, **kwargs):
        path.rename(admitted)
        path.symlink_to(target)
        connection = real_connect(value, *args, **kwargs)
        path.unlink()
        admitted.rename(path)
        return connection

    monkeypatch.setattr(durability.sqlite3, "connect", swapping_connect)
    store = SQLiteServiceStore(path)
    store.close()
    assert path.stat().st_size > 0
    assert target.stat().st_size == 0


def test_outbox_dispatcher_releases_failures_for_durable_retry():
    store = SQLiteServiceStore()
    with store.transaction() as transaction:
        transaction.enqueue(
            OutboxMessage(
                "message",
                "tenant",
                "dispatch",
                "job:1",
                {"job_id": "job"},
                1,
                1,
            )
        )
    attempts = 0

    def handler(message: OutboxMessage, transaction) -> None:
        nonlocal attempts
        attempts += 1
        transaction.enqueue(
            OutboxMessage(
                "effect",
                message.tenant_id,
                "effect",
                message.idempotency_key,
                {"source_message_id": message.message_id},
                10,
                10,
            )
        )
        if attempts == 1:
            raise RuntimeError("transient")

    dispatcher = OutboxDispatcher(store, {"dispatch": handler})
    assert dispatcher.dispatch_once("worker", 10).failed_message_ids == ("message",)
    assert dispatcher.dispatch_once("worker", 39).delivered_message_ids == ()
    assert dispatcher.dispatch_once("worker", 40).delivered_message_ids == ("message",)
    effects = store.claim_outbox("effect-worker", 41)
    assert tuple(message.message_id for message in effects) == ("effect",)


def test_durable_request_id_rejects_conflicting_payload():
    store = SQLiteServiceStore()
    first = DurableJobRecord(
        "one", "tenant", "request", "0" * 64, JobState.QUEUED, 1, {}, 1, 1
    )
    conflict = DurableJobRecord(
        "two", "tenant", "request", "1" * 64, JobState.QUEUED, 1, {}, 1, 1
    )
    with pytest.raises(ValueError, match="credential"):
        DurableJobRecord(
            "secret-job",
            "tenant",
            "",
            "2" * 64,
            JobState.QUEUED,
            1,
            {"access_token": "must-not-persist"},
            1,
            1,
        )
    for field_name in ("token", "api_key", "session", "cookie", "secret_value"):
        with pytest.raises(ValueError, match="credential"):
            DurableJobRecord(
                f"secret-{field_name}",
                "tenant",
                "",
                "2" * 64,
                JobState.QUEUED,
                1,
                {field_name: "must-not-persist"},
                1,
                1,
            )
    with pytest.raises(ValueError, match="credential"):
        DurableJobRecord(
            "opaque-secret",
            "tenant",
            "",
            "2" * 64,
            JobState.QUEUED,
            1,
            {"value": "Q" * 44},
            1,
            1,
        )
    with store.transaction() as transaction:
        transaction.insert_job(first)
    with pytest.raises(IntegrityError, match="idempotency"):
        with store.transaction() as transaction:
            transaction.insert_job(conflict)


def test_scheduler_timeout_kills_descendants_that_ignore_sigterm(tmp_path):
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import signal,time;signal.signal(signal.SIGTERM, signal.SIG_IGN);time.sleep(60)"
    )
    parent_code = (
        "import subprocess,sys,time;"
        f"child=subprocess.Popen([sys.executable,'-c',{child_code!r}]);"
        f"open({str(child_pid_path)!r},'w').write(str(child.pid));"
        "time.sleep(60)"
    )
    executor = SubprocessCommandExecutor(
        timeout_seconds=0.5,
        maximum_response_bytes=1024,
    )
    with pytest.raises(TimeoutError, match="timeout"):
        executor.run((sys.executable, "-c", parent_code))
    child_pid = int(child_pid_path.read_text())
    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.01)
    else:
        pytest.fail("Scheduler timeout left a descendant process running.")


def test_scheduler_command_executor_bounds_response_before_decode():
    executor = SubprocessCommandExecutor(
        timeout_seconds=5.0,
        maximum_response_bytes=16,
    )
    with pytest.raises(IntegrityError, match="byte limit"):
        executor.run(
            (
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 17)",
            )
        )


def test_slurm_uses_argv_and_maps_machine_state():
    executor = _Executor(
        [
            CommandResult(0, "42;cluster\n", ""),
            CommandResult(
                0,
                json.dumps(
                    {
                        "jobs": [
                            {
                                "job_id": 42,
                                "job_state": "RUNNING",
                                "state_reason": "None",
                            }
                        ]
                    }
                ),
                "",
            ),
        ]
    )
    ledger = SQLiteServiceStore()
    scheduler = SlurmScheduler(executor, ledger=ledger)
    spec = SlurmJobSpec(
        "/safe/worker",
        ("argument with space", "; rm -rf /"),
        "request",
        "job-name",
        ResourceRequest(2, 4096),
    )
    duplicate = SlurmScheduler(executor, ledger=ledger)
    assert scheduler.submit(spec) == duplicate.submit(spec) == "42"
    assert executor.argv[0][-3:] == (
        "/safe/worker",
        "argument with space",
        "; rm -rf /",
    )
    assert scheduler.status("42").state is SchedulerState.RUNNING
    assert len(executor.argv) == 2


def test_slurm_ranked_job_uses_one_safe_srun_step() -> None:
    executor = _Executor([CommandResult(0, "43\n", "")])
    scheduler = SlurmScheduler(executor)
    spec = SlurmJobSpec(
        "/safe/worker",
        ("argument with space", "; rm -rf /"),
        "ranked-request",
        "ranked-job",
        ResourceRequest(
            4,
            8192,
            accelerator_count=2,
            host_count=2,
            process_count=2,
        ),
        ranked=True,
    )
    assert scheduler.submit(spec) == "43"
    assert "--nodes=2" in executor.argv[0]
    assert "--ntasks=2" in executor.argv[0]
    script = executor.stdin[0]
    assert script is not None
    decoded = script.decode("utf-8")
    assert decoded.startswith("#!/bin/sh\nset -eu\nexec srun ")
    assert "--ntasks=2" in decoded
    assert "'; rm -rf /'" in decoded


def test_slurm_idempotency_binds_the_complete_submission():
    executor = _Executor([CommandResult(0, "42\n", "")])
    ledger = SQLiteServiceStore()
    scheduler = SlurmScheduler(executor, ledger=ledger)
    original = SlurmJobSpec(
        "/safe/worker",
        ("argument",),
        "request",
        "first-name",
        ResourceRequest(1, 1024),
    )
    assert scheduler.submit(original) == "42"
    with pytest.raises(IntegrityError, match="different request"):
        scheduler.submit(replace(original, job_name="different-name"))
    assert len(executor.argv) == 1


class _KubernetesTransport:
    def __init__(self):
        self.requests: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.responses: list[HTTPResponse] = []

    def request(self, method, url, /, *, headers, body=None):
        self.requests.append((method, url, dict(headers), body))
        return self.responses.pop(0)


def test_kubernetes_rejects_oversized_transport_responses_before_json_decode():
    transport = _KubernetesTransport()
    transport.responses.append(HTTPResponse(200, {}, b"x" * 9))
    scheduler = KubernetesScheduler(
        "https://cluster.example",
        "credential",
        transport,
        maximum_response_bytes=8,
    )
    with pytest.raises(IntegrityError, match="byte limit"):
        scheduler.status("tenant", "job")


def test_kubernetes_idempotency_authentication_and_resource_version():
    spec = KubernetesJobSpec(
        "tenant",
        "image@sha256:digest",
        ("worker", "--safe"),
        "request",
        ResourceRequest(1, 1024),
    )
    name = "phydrax-" + hashlib.sha256(b"request").hexdigest()[:32]

    # Populate the create response with the digest generated in the outgoing body.
    class _CreateTransport(_KubernetesTransport):
        def request(self, method, url, /, *, headers, body=None):
            self.requests.append((method, url, dict(headers), body))
            if method == "GET":
                return HTTPResponse(404, {}, b"{}")
            assert body is not None
            decoded = json.loads(body)
            return HTTPResponse(201, {}, json.dumps(decoded).encode())

    create_transport = _CreateTransport()
    scheduler = KubernetesScheduler(
        "https://cluster.example", "credential", create_transport
    )
    assert scheduler.submit(spec) == name
    assert create_transport.requests[1][2]["Authorization"] == "Bearer credential"
    conflict_transport = _KubernetesTransport()
    conflict_transport.responses.append(HTTPResponse(409, {}, b'{"message":"changed"}'))
    conflict_scheduler = KubernetesScheduler(
        "https://cluster.example", "credential", conflict_transport
    )
    with pytest.raises(IntegrityError, match="resourceVersion"):
        conflict_scheduler.replace(spec, name, "7")


def test_kubernetes_ranked_job_creates_headless_rendezvous_service() -> None:
    class CreateTransport(_KubernetesTransport):
        def request(self, method, url, /, *, headers, body=None):
            self.requests.append((method, url, dict(headers), body))
            if method == "GET":
                return HTTPResponse(404, {}, b"{}")
            assert body is not None
            decoded = json.loads(body)
            return HTTPResponse(201, {}, json.dumps(decoded).encode())

    spec = KubernetesJobSpec(
        "tenant",
        "image@sha256:digest",
        ("worker", "--safe"),
        "ranked-request",
        ResourceRequest(
            4,
            4096,
            accelerator_count=2,
            host_count=2,
            process_count=2,
            accelerator_vendor="nvidia",
        ),
    )
    transport = CreateTransport()
    scheduler = KubernetesScheduler(
        "https://cluster.example",
        "credential",
        transport,
    )
    scheduler.submit(spec)
    created = [
        json.loads(body)
        for method, _, _, body in transport.requests
        if method == "POST" and body is not None
    ]
    service = next(value for value in created if value["kind"] == "Service")
    job = next(value for value in created if value["kind"] == "Job")
    assert service["spec"]["clusterIP"] == "None"
    assert service["spec"]["publishNotReadyAddresses"] is True
    assert job["spec"]["completionMode"] == "Indexed"
    assert job["spec"]["parallelism"] == 2
    container = job["spec"]["template"]["spec"]["containers"][0]
    environment = {item["name"]: item for item in container["env"]}
    assert environment["PHYDRAX_NUM_PROCESSES"]["value"] == "2"
    assert (
        "job-completion-index"
        in (environment["PHYDRAX_PROCESS_ID"]["valueFrom"]["fieldRef"]["fieldPath"])
    )
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode()


def test_jwks_ed25519_claims_and_key_rotation():
    crypto = pytest.importorskip("cryptography.hazmat.primitives.asymmetric.ed25519")
    serialization = pytest.importorskip("cryptography.hazmat.primitives.serialization")
    clock = _Clock(100)
    configuration = OIDCConfiguration("https://issuer.example", "phydrax", 0, 100)
    old = crypto.Ed25519PrivateKey.generate()
    new = crypto.Ed25519PrivateKey.generate()

    def jwk(key, kid):
        raw = key.public_key().public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )
        return {
            "kty": "OKP",
            "crv": "Ed25519",
            "alg": "EdDSA",
            "use": "sig",
            "kid": kid,
            "x": _b64(raw),
        }

    provider = StaticJWKSProvider(
        configuration.issuer, JSONWebKeySet((jwk(old, "old"),), 1, 200)
    )
    validator = OIDCJWKSTokenValidator(
        configuration, provider, clock=clock, accepted_algorithms=frozenset({"EdDSA"})
    )

    def issue(key, kid, **overrides):
        header = _b64(
            json.dumps(
                {"alg": "EdDSA", "kid": kid, "typ": "at+jwt"},
                separators=(",", ":"),
                sort_keys=True,
            ).encode()
        )
        claims = {
            "iss": configuration.issuer,
            "aud": "phydrax",
            "sub": "user",
            "tenant_id": "tenant",
            "client_id": "client",
            "jti": "token",
            "scope": "service:submit",
            "iat": 90,
            "nbf": 90,
            "exp": 110,
            **overrides,
        }
        payload = _b64(json.dumps(claims, separators=(",", ":"), sort_keys=True).encode())
        signing_input = f"{header}.{payload}".encode()
        return f"{header}.{payload}.{_b64(key.sign(signing_input))}"

    assert validator.validate(issue(old, "old")).tenant_id == "tenant"
    provider.rotate(JSONWebKeySet((jwk(new, "new"),), 50, 200, frozenset({"old"})))
    assert validator.validate(issue(new, "new")).token_id == "token"
    with pytest.raises(AuthenticationError):
        validator.validate(issue(old, "old"))
    with pytest.raises(AuthenticationError, match="audience"):
        validator.validate(issue(new, "new", aud="other"))


def test_mtls_san_expiry_and_issuer_policy():
    policy = MTLSCertificatePolicy("cluster.example", frozenset({"1" * 64}), 100, 0)
    certificate = WorkloadCertificate(
        ("spiffe://cluster.example/ns/tenant/sa/worker",), 10, 50, "0" * 64, "1" * 64
    )
    assert policy.validate(certificate, 20).spiffe_id.endswith("/worker")
    with pytest.raises(AuthenticationError, match="expired"):
        policy.validate(certificate, 50)
    wrong_san = replace(certificate, san_uris=("spiffe://other.example/worker",))
    with pytest.raises(AuthenticationError, match="SPIFFE"):
        policy.validate(wrong_san, 20)


def test_hmac_signing_key_repr_does_not_disclose_secret():
    secret = b"not-for-logs-" + b"x" * 32
    rendered = repr(HMACSigningKey("key", secret))
    assert secret.hex() not in rendered
    assert repr(secret) not in rendered


def test_ed25519_sign_verify_rotate_and_revoke():
    pytest.importorskip("cryptography")
    signer = Ed25519Signer("one", b"1" * 32)
    verifier = Ed25519Verifier("one", signer.public_key_bytes)
    envelope = signer.sign(b"payload", purpose="release", signed_at=10)
    trust = SigningTrustStore()
    trust.trust(SigningKeyTrustRecord("one", "Ed25519", 0, 100), verifier)
    trust.verify(b"payload", envelope, expected_purpose="release", at_time=20)
    trust.revoke("one", 21)
    with pytest.raises(IntegrityError, match="revoked"):
        trust.verify(
            b"payload",
            envelope,
            expected_purpose="release",
            at_time=22,
        )


def test_injected_kms_sign_and_verify():
    class KMS:
        def sign(self, key_id, algorithm, message, /):
            return hmac.new(b"kms", message, hashlib.sha256).digest()

        def verify(self, key_id, algorithm, message, signature, /):
            return hmac.compare_digest(signature, self.sign(key_id, algorithm, message))

    kms = KMS()
    envelope = KMSSigner("kms-key", "injected-test", kms).sign(
        b"payload", purpose="audit", signed_at=1
    )
    KMSVerifier("kms-key", "injected-test", kms).verify(
        b"payload", envelope, expected_purpose="audit"
    )
    with pytest.raises(IntegrityError, match="purpose"):
        KMSVerifier("kms-key", "injected-test", kms).verify(
            b"payload", envelope, expected_purpose="release"
        )


def test_short_lived_scoped_secrets_and_redaction():
    clock = _Clock(10)
    broker = LocalSecretHandleBroker(clock=clock, maximum_lifetime_seconds=60)
    handle = broker.issue(
        "tenant",
        b"do-not-log",
        frozenset({"provider:read"}),
        lifetime_seconds=10,
        key_version="one",
    )
    assert "do-not-log" not in repr(handle)
    assert broker.resolve(handle, "tenant", "provider:read") == b"do-not-log"
    with pytest.raises(AuthorizationError):
        broker.resolve(handle, "other", "provider:read")
    clock.value = 20
    with pytest.raises(AuthorizationError, match="expired"):
        broker.resolve(handle, "tenant", "provider:read")
    assert SecretRedactor().redact(
        {"authorization": "Bearer secret", "safe": "value"}
    ) == {
        "authorization": "<redacted>",
        "safe": "value",
    }


def test_support_bundle_is_allowlisted_redacted_and_privacy_bounded():
    telemetry = HostTelemetrySnapshot.create(
        (
            TelemetryDatum("safe", 1, "count", PrivacyClassification.INTERNAL, 10),
            TelemetryDatum(
                "hostname", "private", "string", PrivacyClassification.SENSITIVE, 10
            ),
        )
    )
    policy = SupportBundlePolicy(
        {
            "runtime": frozenset({"status", "token"}),
            "telemetry": frozenset({"safe", "hostname"}),
        }
    )
    bundle = create_support_bundle(
        {
            "runtime": {"status": "failed", "token": "Bearer secret", "unlisted": "no"},
            "unknown": {"value": 1},
        },
        policy,
        clock=_Clock(10),
        telemetry=telemetry,
    )
    assert bundle.sections["runtime"] == {"status": "failed", "token": "<redacted>"}
    assert bundle.sections["telemetry"] == {"safe": 1}
    assert (
        "unknown" not in bundle.sections and "unlisted" not in bundle.sections["runtime"]
    )


def test_provenance_hash_holds_ancestor_descriptors_during_replacement(
    tmp_path, monkeypatch
):
    import phydrax.lifecycle._provenance as provenance

    root = tmp_path / "root"
    source = root / "source"
    outside = tmp_path / "outside"
    source.mkdir(parents=True)
    outside.mkdir()
    (source / "input.txt").write_text("trusted")
    (outside / "input.txt").write_text("substituted")
    expected = digest_paths(root, ("source/input.txt",))
    real_open = provenance.open_regular_beneath

    @contextmanager
    def swapping_open(*args, **kwargs):
        with real_open(*args, **kwargs) as opened:
            source.rename(root / "held-source")
            source.symlink_to(outside, target_is_directory=True)
            yield opened

    monkeypatch.setattr(provenance, "open_regular_beneath", swapping_open)
    assert digest_paths(root, ("source/input.txt",)) == expected


def test_host_telemetry_is_explicit_structured_and_logged_by_identity(
    phydrax_events,
):
    snapshot = HostTelemetryCollector(
        clock=_Clock(10),
        policy=HostTelemetryPolicy(include_jax_runtime=False),
    ).collect()

    record = snapshot.to_record(PrivacyClassification.INTERNAL)
    observations = record["observations"]
    assert isinstance(observations, list)
    names = set()
    for value in observations:
        assert isinstance(value, dict)
        name = value.get("name")
        assert isinstance(name, str)
        names.add(name)
    assert "host.os.release" in names
    assert "host.os.kernel" in names
    assert "host.name" not in names
    assert json.loads(snapshot.to_json()) == record

    event = phydrax_events.records("runtime.environment.captured")[-1]
    assert event["fields"] == {
        "include_host_identity": False,
        "include_jax_devices": False,
        "observation_count": len(snapshot.observations),
        "snapshot_id": snapshot.snapshot_id,
    }
    serialized = phydrax_events.path.read_text(encoding="utf-8")
    assert snapshot.snapshot_id in serialized
    if host_name := platform.node():
        assert host_name not in serialized


def test_source_build_and_spdx_provenance_are_deterministic(tmp_path: Path):
    (tmp_path / "a.lock").write_text("a")
    (tmp_path / "source.py").write_text("source")
    first_digest = digest_paths(tmp_path, ("source.py", "a.lock"))
    assert first_digest == digest_paths(tmp_path, ("a.lock", "source.py"))
    packages = (InstalledPackage("B", "2"), InstalledPackage("a", "1", "MIT"))
    first = create_build_provenance(
        "phydrax",
        "1",
        source_digest=first_digest,
        lock_digest="0" * 64,
        repository_revision="revision",
        builder_id="builder",
        packages=packages,
    )
    second = create_build_provenance(
        "phydrax",
        "1",
        source_digest=first_digest,
        lock_digest="0" * 64,
        repository_revision="revision",
        builder_id="builder",
        packages=tuple(reversed(packages)),
    )
    assert first.to_json() == second.to_json()
    assert spdx_json(generate_spdx_sbom("phydrax", packages)) == spdx_json(
        generate_spdx_sbom("phydrax", tuple(reversed(packages)))
    )
    document = generate_spdx_sbom("phydrax", packages, provenance=first)
    annotations = document["annotations"]
    assert isinstance(annotations, list)
    annotation = annotations[0]
    assert isinstance(annotation, dict)
    comment = annotation["comment"]
    assert isinstance(comment, str)
    assert first.source_digest in comment
    assert first.lock_digest in comment


def test_provider_construction_has_no_network_or_telemetry_effects():
    transport = _KubernetesTransport()
    KubernetesScheduler("https://cluster.example", "credential", transport)
    HTTPSJWKSProvider(
        "https://issuer.example",
        "https://issuer.example/jwks",
        transport,
        clock=_Clock(10),
    )
    assert transport.requests == []


def test_runtime_admits_exact_support_before_allocation_and_bootstrap(
    phydrax_events,
):
    dependency = SupportDependency("profile", "provider-tuple")
    resolved = ResolvedRunSpec(
        (dependency,),
        (),
        release_index_id="release",
        profile_ids=("profile",),
        trust_policy_id="trust",
        valid_at=10,
        valid_from=1,
        valid_until=20,
        prepared_configuration_id="configuration",
        precision_policy_id="precision",
        resource_policy_id="resources",
        checkpoint_policy_id="checkpoint",
        output_policy_id="output",
        repository_id="repository",
        scheduler_id="scheduler",
        auth_policy_id="authentication",
    )

    class Validator:
        def validate(self, token: str, /) -> ValidatedPrincipal:
            return ValidatedPrincipal(
                "subject",
                token,
                "issuer",
                "audience",
                "client",
                f"token-{token}",
                frozenset(
                    {
                        "service:execute",
                        "service:status",
                        "service:submit",
                        "service:usage",
                    }
                ),
                0,
                100,
            )

    class Admitter:
        def __init__(self):
            self.calls: list[tuple[str, int]] = []

        def require(self, value: SupportDependency, /, *, at_time: int) -> None:
            self.calls.append((value.dependency_id, at_time))

    class Repository:
        provider_id = "repository"

    store = SQLiteServiceStore()
    admitter = Admitter()
    service = InProcessReferenceService(
        Validator(),
        ScopeTenantAuthorizer(),
        {
            "tenant": TenantQuota(1, 1, 1024, 0, 1024),
            "other": TenantQuota(1, 1, 1024, 0, 1024),
        },
        clock=_Clock(10),
        dependency_admitter=admitter,
        durable_store=store,
        repository=Repository(),
        scheduler_id="scheduler",
        auth_policy_id="authentication",
    )
    service.register_provider(
        "profile",
        lambda submission, context: ProviderResult(("result",)),
        support_tuple_id="provider-tuple",
    )
    submission = JobSubmission(
        AnalysisPlan(
            "analysis",
            "provider-plan",
            "discretization",
            ("layout",),
        ),
        ExecutionPlan("execution", "cpu", "float64", "direct"),
        "revision",
        "profile",
        {},
        ResourceRequest(1, 1024),
        request_id="request",
        resolved_run_spec=resolved,
    )
    queued = service.submit("tenant", submission)
    assert service.submit("tenant", submission).job_id == queued.job_id
    with pytest.raises(ResourceNotFound):
        service.status("other", queued.job_id)
    completed = service.execute("tenant", queued.job_id)

    assert completed.state is JobState.SUCCEEDED
    assert phydrax_events.records("service.provider.registered")
    assert phydrax_events.records("service.job.submitted")
    execution_event = phydrax_events.records("service.job.completed")[-1]
    assert execution_event["fields"]["run_record_id"] == completed.run_record.record_id
    serialized_events = phydrax_events.path.read_text(encoding="utf-8")
    assert "token-tenant" not in serialized_events
    assert '"principal_id"' not in serialized_events
    assert admitter.calls == [
        (dependency.dependency_id, 10),
        (dependency.dependency_id, 10),
    ]
    assert store.quota_usage("tenant").active_jobs == 0
    assert len(store.claim_outbox("dispatcher", 10)) == 1


def test_execution_heartbeat_and_attempt_fence_reject_stale_completion():
    class Validator:
        def validate(self, token: str, /) -> ValidatedPrincipal:
            return ValidatedPrincipal(
                "subject",
                token,
                "issuer",
                "audience",
                "client",
                f"token-{token}",
                frozenset({"service:execute", "service:status", "service:submit"}),
                0,
                100,
            )

    clock = _Clock(10)
    store = SQLiteServiceStore()
    started = threading.Event()
    release = threading.Event()
    calls = 0

    def provider(submission, context):
        nonlocal calls
        del submission
        calls += 1
        if calls == 1:
            started.set()
            assert release.wait(timeout=5.0)
        else:
            context.heartbeat()
        return ProviderResult((f"attempt-{calls}",))

    service = InProcessReferenceService(
        Validator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
        clock=clock,
        durable_store=store,
        execution_lease_seconds=1,
    )
    service.register_provider("profile", provider, support_tuple_id="provider-tuple")
    submission = JobSubmission(
        AnalysisPlan(
            "analysis",
            "provider-plan",
            "discretization",
            ("layout",),
        ),
        ExecutionPlan("execution", "cpu", "float64", "direct"),
        "revision",
        "profile",
        {},
        ResourceRequest(1, 1024),
        request_id="fenced-request",
    )
    queued = service.submit("tenant", submission)
    stale_results = []
    worker = threading.Thread(
        target=lambda: stale_results.append(service.execute("tenant", queued.job_id))
    )
    worker.start()
    assert started.wait(timeout=5.0)

    clock.value = 12
    recovered = service.recover_stale_attempts()
    assert recovered[0].attempt == 2
    assert recovered[0].state is JobState.QUEUED
    release.set()
    worker.join(timeout=5.0)
    assert not worker.is_alive()

    current = service.status("tenant", queued.job_id)
    assert current.attempt == 2
    assert current.state is JobState.QUEUED
    assert stale_results[0].attempt == 2
    assert stale_results[0].state is JobState.QUEUED

    completed = service.execute("tenant", queued.job_id)
    assert completed.attempt == 2
    assert completed.state is JobState.SUCCEEDED
    with store.transaction() as transaction:
        durable = transaction.get_job("tenant", queued.job_id)
    assert durable is not None
    assert durable.attempt == 2
    assert durable.state is JobState.SUCCEEDED


def test_durable_service_restart_rehydrates_results_and_idempotency(tmp_path):
    path = tmp_path / "service.sqlite"
    quota = {"tenant": TenantQuota(2, 2, 4096, 0, 1024)}
    first_store = SQLiteServiceStore(path)
    first = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        quota,
        clock=_Clock(10),
        durable_store=first_store,
    )
    first.register_provider(
        "profile",
        lambda submission, context: ProviderResult(("result",), ("provider-diagnostic",)),
        support_tuple_id="provider-tuple",
    )
    submission = _service_submission("restart-request")
    queued = first.submit("tenant", submission)
    completed = first.execute("tenant", queued.job_id)
    assert completed.run_record.result_ids == ("result",)
    first_store.close()

    second_store = SQLiteServiceStore(path)
    restarted = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        quota,
        clock=_Clock(10),
        durable_store=second_store,
    )
    restarted.register_provider(
        "profile",
        lambda submission, context: ProviderResult(("unexpected",)),
        support_tuple_id="provider-tuple",
    )

    restored = restarted.status("tenant", queued.job_id)
    assert restored.state is JobState.SUCCEEDED
    assert restored.run_record.result_ids == ("result",)
    assert restored.run_record.diagnostic_ids == ("provider-diagnostic",)
    assert restarted.submit("tenant", submission).job_id == queued.job_id


def test_provider_failure_preserves_exact_exception_status_and_message():
    service = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
        clock=_Clock(10),
    )

    def provider(submission, context):
        del submission, context
        raise ProfileUnavailable("provider refused exact problem identity")

    service.register_provider("profile", provider, support_tuple_id="provider-tuple")
    failed = service.execute(
        "tenant",
        service.submit("tenant", _service_submission("failed-request")).job_id,
    )

    assert failed.state is JobState.FAILED
    assert failed.failure is not None
    assert failed.failure.code == "ProfileUnavailable"
    assert (
        failed.failure.exception_type == "phydrax.service._contracts.ProfileUnavailable"
    )
    assert failed.failure.message == "provider refused exact problem identity"


def test_checkpoint_callback_requires_repository_commit_and_survives_restart(tmp_path):
    profile = HPCFilesystemProfile(
        "service-checkpoint",
        "local-posix",
        atomic_rename_same_filesystem=True,
        file_fsync=True,
        directory_fsync=True,
        advisory_locking=True,
        attempt_private_staging=True,
    )
    repository = POSIXArtifactRepository(
        tmp_path / "repository",
        POSIXRepositoryPolicy(
            profile,
            maximum_chunk_bytes=1024,
            maximum_metadata_bytes=64 * 1024,
        ),
    )
    payload = b"durable-checkpoint"
    checkpoint = CheckpointManifest(
        "checkpoint",
        "analysis",
        "revision",
        "execution",
        (
            CheckpointShard(
                "state",
                hashlib.sha256(payload).hexdigest(),
                len(payload),
            ),
        ),
        complete=True,
    )
    transaction = repository.begin("checkpoint", "provider", started_at=1)
    chunk = repository.write_chunk(transaction, "state", 0, 0, payload)
    repository.commit(
        transaction,
        (chunk,),
        metadata={"checkpoint_manifest_id": checkpoint.manifest_id},
        committed_at=1,
    )
    child_payload = b"durable-child-checkpoint"
    child = CheckpointManifest(
        "checkpoint-child",
        "analysis",
        "revision",
        "execution",
        (
            CheckpointShard(
                "state",
                hashlib.sha256(child_payload).hexdigest(),
                len(child_payload),
            ),
        ),
        complete=True,
        parent_checkpoint_id=checkpoint.checkpoint_id,
        parent_manifest_id=checkpoint.manifest_id,
    )
    child_transaction = repository.begin(child.checkpoint_id, "provider", started_at=2)
    child_chunk = repository.write_chunk(child_transaction, "state", 0, 0, child_payload)
    repository.commit(
        child_transaction,
        (child_chunk,),
        metadata={"checkpoint_manifest_id": child.manifest_id},
        committed_at=2,
    )
    second_root = CheckpointManifest(
        "checkpoint-second-root",
        "analysis",
        "revision",
        "execution",
        checkpoint.shards,
        complete=True,
    )
    store_path = tmp_path / "checkpoint-service.sqlite"
    artifact_signing_secret = b"a" * 32
    store = SQLiteServiceStore(store_path)
    service = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
        clock=_Clock(10),
        durable_store=store,
        repository=repository,
        artifact_signing_secret=artifact_signing_secret,
    )

    def provider(submission, context):
        del submission
        assert context.checkpoint(checkpoint) == checkpoint.checkpoint_id
        assert context.checkpoint(checkpoint) == checkpoint.checkpoint_id
        with pytest.raises(IntegrityError, match="parent presence"):
            context.checkpoint(second_root)
        assert context.checkpoint(child) == child.checkpoint_id
        return ProviderResult(("result",))

    service.register_provider("profile", provider, support_tuple_id="provider-tuple")
    completed = service.execute(
        "tenant",
        service.submit("tenant", _service_submission("checkpoint-request")).job_id,
    )
    assert completed.checkpoint_ids == ("checkpoint", "checkpoint-child")
    artifact_content = b"restart-artifact"
    artifact_rights = ArtifactRights.unrestricted(
        "scientific-restart-artifact",
        hashlib.sha256(artifact_content).hexdigest(),
        len(artifact_content),
        rights_id="rights",
        use_policy_id="use",
        license_id="license",
        source_uri="https://example.invalid/restart-artifact",
        attribution_id="attribution",
    )
    artifact_descriptor = service.store_artifact(
        "tenant",
        completed.job_id,
        artifact_content,
        scientific_artifact_id="scientific-restart-artifact",
        media_type="application/octet-stream",
        rights=artifact_rights,
    )
    store.close()

    restarted_store = SQLiteServiceStore(store_path)
    restarted = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
        clock=_Clock(10),
        durable_store=restarted_store,
        repository=repository,
        artifact_signing_secret=artifact_signing_secret,
    )
    restarted.register_provider(
        "profile",
        lambda submission, context: ProviderResult(("unexpected",)),
        support_tuple_id="provider-tuple",
    )
    restored = restarted.status("tenant", completed.job_id)
    assert restored.checkpoint_ids == ("checkpoint", "checkpoint-child")
    assert restored.artifact_ids == (artifact_descriptor.artifact_id,)
    restarted_grant = restarted.grant_artifact("tenant", artifact_descriptor.artifact_id)
    assert restarted.fetch_artifact("tenant", restarted_grant).content == artifact_content


def test_failed_durable_transition_leaves_memory_and_store_unchanged(monkeypatch):
    import phydrax.service._durability as durability

    store = SQLiteServiceStore()
    service = InProcessReferenceService(
        _ServiceValidator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
        clock=_Clock(10),
        durable_store=store,
    )
    service.register_provider(
        "profile",
        lambda submission, context: ProviderResult(("result",)),
        support_tuple_id="provider-tuple",
    )
    queued = service.submit("tenant", _service_submission("atomic-request"))
    with store.transaction() as transaction:
        before = transaction.get_job("tenant", queued.job_id)
    assert before is not None

    def fail_update(self, record, /, *, expected_version):
        del self, record, expected_version
        raise RuntimeError("commit failure")

    monkeypatch.setattr(durability._SQLiteTransaction, "update_job", fail_update)
    with pytest.raises(RuntimeError, match="commit failure"):
        service.cancel("tenant", queued.job_id)

    assert service.status("tenant", queued.job_id).state is JobState.QUEUED
    with store.transaction() as transaction:
        after = transaction.get_job("tenant", queued.job_id)
    assert after == before


def test_bearer_token_resource_limits_precede_validators_and_crypto():
    class IngressValidator:
        called = False

        def validate(self, token: str, /) -> ValidatedPrincipal:
            self.called = True
            raise AssertionError("oversize token reached the validator")

    ingress_validator = IngressValidator()
    ingress_service = InProcessReferenceService(
        ingress_validator,
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 1024)},
    )
    with pytest.raises(AuthenticationError, match="compact-token byte limit"):
        ingress_service.usage("a" * 16_385)
    assert ingress_validator.called is False

    class UnreachedJWKSProvider:
        def get(self, issuer: str, /, *, force_refresh: bool = False):
            raise AssertionError("rejected token reached JWKS lookup")

    configuration = OIDCConfiguration("https://issuer.example", "phydrax", 0, 100)
    validators = (
        HMACOIDCTokenValidator(
            configuration, (HMACSigningKey("key", b"k" * 32),), clock=_Clock(10)
        ),
        OIDCJWKSTokenValidator(
            configuration,
            UnreachedJWKSProvider(),
            clock=_Clock(10),
            accepted_algorithms=frozenset({"RS256"}),
        ),
    )
    header = _b64(
        json.dumps(
            {"alg": "RS256", "kid": "key", "typ": "at+jwt"},
            separators=(",", ":"),
        ).encode()
    )
    payloads = (
        (b'{"x":' + b"[" * 8 + b"0" + b"]" * 8 + b"}", "nesting"),
        (
            json.dumps(
                {f"k{index}": index for index in range(65)}, separators=(",", ":")
            ).encode(),
            "key-count",
        ),
        (
            json.dumps({"x": list(range(65))}, separators=(",", ":")).encode(),
            "array-item",
        ),
        (
            json.dumps({"x" * 129: 0}, separators=(",", ":")).encode(),
            "key-byte",
        ),
        (
            json.dumps({"x": "v" * 2_049}, separators=(",", ":")).encode(),
            "string-byte",
        ),
    )
    for validator in validators:
        with pytest.raises(AuthenticationError, match="decoded-byte"):
            validator.validate(f"{'A' * 1_368}.e30.c2ln")
        with pytest.raises(AuthenticationError, match="encoded-byte"):
            validator.validate(f"{header}.{'A' * 12_289}.c2ln")
        for payload, expected in payloads:
            with pytest.raises(AuthenticationError, match=expected):
                validator.validate(f"{header}.{_b64(payload)}.c2ln")


def test_artifact_rights_remain_bound_and_gate_grant_and_fetch():
    class Validator:
        def validate(self, token: str, /) -> ValidatedPrincipal:
            assert token == "token"
            return ValidatedPrincipal(
                "subject",
                "tenant",
                "issuer",
                "audience",
                "client",
                "token-id",
                frozenset(
                    {
                        "service:artifact:fetch",
                        "service:artifact:grant",
                        "service:artifact:write",
                        "service:submit",
                    }
                ),
                0,
                100,
            )

    service = InProcessReferenceService(
        Validator(),
        ScopeTenantAuthorizer(),
        {"tenant": TenantQuota(1, 1, 1024, 0, 16_384)},
        clock=_Clock(10),
        artifact_signing_secret=b"s" * 32,
    )
    with pytest.raises(IntegrityError, match="must be a string"):
        service.fetch_artifact("token", None)  # type: ignore[arg-type]
    with pytest.raises(IntegrityError, match="encoded-byte limit"):
        service.fetch_artifact("token", "00" * 3_000)
    service.register_provider(
        "profile", lambda submission, context: ProviderResult(("result",))
    )
    queued = service.submit(
        "token",
        JobSubmission(
            AnalysisPlan("analysis", "provider", "discretization", ("field",)),
            ExecutionPlan("execution", "cpu", "float64", "direct"),
            "revision",
            "profile",
            {},
            ResourceRequest(1, 1024),
        ),
    )

    denied_permissions = {
        "scientific": (True, False, True, True),
        "cad": (False, True, True, True),
        "checkpoint": (True, True, False, True),
        "diagnostic": (True, True, True, False),
        "support": (False, False, True, True),
    }
    restricted_descriptors = {}
    for classification in (
        "scientific",
        "cad",
        "checkpoint",
        "diagnostic",
        "support",
    ):
        scientific_artifact_id = f"restricted-{classification}"
        content = f"rights-bound-{classification}".encode()
        content_sha256 = hashlib.sha256(content).hexdigest()
        restricted = ArtifactRights(
            scientific_artifact_id,
            f"source-rights-{classification}",
            f"requested-use-{classification}",
            "LicenseRef-Restricted",
            f"https://source.example/{classification}",
            "source-attribution",
            content_sha256,
            len(content),
            classification,
            *denied_permissions[classification],
        )
        descriptor = service.store_artifact(
            "token",
            queued.job_id,
            content,
            scientific_artifact_id=scientific_artifact_id,
            media_type="application/octet-stream",
            classification=classification,
            rights=restricted,
            cad=(
                CADArtifactMetadata("step", "US", approval_id="approved")
                if classification == "cad"
                else None
            ),
        )
        restricted_descriptors[classification] = descriptor
        assert descriptor.rights == restricted
        with pytest.raises(AuthorizationError, match="redistribution or export"):
            service.grant_artifact("token", descriptor.artifact_id)
        preexisting_grant = service._grant_token(descriptor, 20)
        with pytest.raises(AuthorizationError, match="redistribution or export"):
            service.fetch_artifact("token", preexisting_grant)

    descriptor = restricted_descriptors["scientific"]
    restricted_content = b"rights-bound-scientific"
    restricted_digest = hashlib.sha256(restricted_content).hexdigest()

    unrestricted_content = b"unrestricted-support"
    unrestricted_digest = hashlib.sha256(unrestricted_content).hexdigest()
    unrestricted = ArtifactRights.unrestricted(
        "support",
        unrestricted_digest,
        len(unrestricted_content),
        rights_id="unrestricted-rights",
        use_policy_id="unrestricted-use",
        license_id="LicenseRef-Unrestricted",
        source_uri="phydrax://tenant/jobs/support",
        attribution_id="tenant",
        classification="support",
    )
    unrestricted_descriptor = service.store_artifact(
        "token",
        queued.job_id,
        unrestricted_content,
        scientific_artifact_id="support",
        media_type="application/octet-stream",
        classification="support",
        rights=unrestricted,
    )
    grant = service.grant_artifact("token", unrestricted_descriptor.artifact_id)
    fetched = service.fetch_artifact("token", grant)
    assert grant.rights == unrestricted
    assert fetched.descriptor.rights == unrestricted
    assert fetched.content == unrestricted_content

    stripped = ArtifactRights.unrestricted(
        "restricted-scientific",
        restricted_digest,
        len(restricted_content),
        rights_id="stripped-rights",
        use_policy_id="unrestricted-use",
        license_id="LicenseRef-Unrestricted",
        source_uri="phydrax://tenant/jobs/reclassified",
        attribution_id="tenant",
    )
    with pytest.raises(IntegrityError, match="different rights or classification"):
        service.store_artifact(
            "token",
            queued.job_id,
            restricted_content,
            scientific_artifact_id="restricted-scientific",
            media_type="application/octet-stream",
            rights=stripped,
        )

    reclassified = ArtifactRights.unrestricted(
        "reclassified",
        restricted_digest,
        len(restricted_content),
        rights_id="reclassified-rights",
        use_policy_id="unrestricted-use",
        license_id="LicenseRef-Unrestricted",
        source_uri="phydrax://tenant/jobs/reclassified",
        attribution_id="tenant",
        classification="support",
    )
    with pytest.raises(IntegrityError, match="different rights or classification"):
        service.store_artifact(
            "token",
            queued.job_id,
            restricted_content,
            scientific_artifact_id="reclassified",
            media_type="application/octet-stream",
            classification="support",
            rights=reclassified,
        )
