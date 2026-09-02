#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import replace
from pathlib import Path

import pytest

from phydrax.lifecycle import AnalysisPlan, ExecutionPlan
from phydrax.lifecycle._provenance import (
    create_build_provenance,
    digest_paths,
    generate_spdx_sbom,
    InstalledPackage,
    spdx_json,
)
from phydrax.lifecycle._resolved_run import ResolvedRunSpec
from phydrax.qualification._evidence import SupportDependency
from phydrax.service._auth import OIDCConfiguration, ScopeTenantAuthorizer
from phydrax.service._contracts import (
    AuditRecord,
    AuthenticationError,
    AuthorizationError,
    IntegrityError,
    JobState,
    JobSubmission,
    ProviderResult,
    ResourceNotFound,
    ResourceRequest,
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


class _Executor:
    def __init__(self, results: list[CommandResult]):
        self.results = results
        self.argv: list[tuple[str, ...]] = []

    def run(
        self, argv: tuple[str, ...], /, *, stdin: bytes | None = None
    ) -> CommandResult:
        assert stdin is None
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

    def handler(message: OutboxMessage) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("transient")

    dispatcher = OutboxDispatcher(store, {"dispatch": handler})
    assert dispatcher.dispatch_once("worker", 10).failed_message_ids == ("message",)
    assert dispatcher.dispatch_once("worker", 39).delivered_message_ids == ()
    assert dispatcher.dispatch_once("worker", 40).delivered_message_ids == ("message",)


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
    with store.transaction() as transaction:
        transaction.insert_job(first)
    with pytest.raises(IntegrityError, match="idempotency"):
        with store.transaction() as transaction:
            transaction.insert_job(conflict)


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


class _KubernetesTransport:
    def __init__(self):
        self.requests: list[tuple[str, str, dict[str, str], bytes | None]] = []
        self.responses: list[HTTPResponse] = []

    def request(self, method, url, /, *, headers, body=None):
        self.requests.append((method, url, dict(headers), body))
        return self.responses.pop(0)


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


def test_ed25519_sign_verify_rotate_and_revoke():
    pytest.importorskip("cryptography")
    signer = Ed25519Signer("one", b"1" * 32)
    verifier = Ed25519Verifier("one", signer.public_key_bytes)
    envelope = signer.sign(b"payload", purpose="release", signed_at=10)
    trust = SigningTrustStore()
    trust.trust(SigningKeyTrustRecord("one", "Ed25519", 0, 100), verifier)
    trust.verify(b"payload", envelope, at_time=20)
    trust.revoke("one", 21)
    with pytest.raises(IntegrityError, match="revoked"):
        trust.verify(b"payload", envelope, at_time=22)


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
    KMSVerifier("kms-key", "injected-test", kms).verify(b"payload", envelope)


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
    ) == {"authorization": "<redacted>", "safe": "value"}


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
    assert first.source_digest in document["annotations"][0]["comment"]
    assert first.lock_digest in document["annotations"][0]["comment"]


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


def test_runtime_admits_exact_support_before_allocation_and_bootstrap():
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
    assert admitter.calls == [
        (dependency.dependency_id, 10),
        (dependency.dependency_id, 10),
    ]
    assert store.quota_usage("tenant").active_jobs == 0
    assert len(store.claim_outbox("dispatcher", 10)) == 1
