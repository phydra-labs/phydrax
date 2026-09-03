#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

from phydrax.lifecycle import AnalysisPlan, ExecutionPlan
from phydrax.service import (
    InProcessReferenceService,
    JobSubmission,
    ResourceRequest,
    ScopeTenantAuthorizer,
    TenantQuota,
    ValidatedPrincipal,
)


class _Validator:
    def validate(self, token: str, /) -> ValidatedPrincipal:
        assert token == "token"
        return ValidatedPrincipal(
            "data-owner",
            "tenant",
            "issuer",
            "audience",
            "client",
            "token-id",
            frozenset({"service:submit", "service:execute"}),
            0,
            2**31,
        )


def test_provider_failure_telemetry_never_copies_untrusted_exception_text():
    service = InProcessReferenceService(
        _Validator(),
        ScopeTenantAuthorizer(),
        {
            "tenant": TenantQuota(
                active_jobs=1,
                cpu_cores=1,
                memory_bytes=4096,
                gpu_count=0,
                retained_artifact_bytes=4096,
            )
        },
    )

    def fail_with_sensitive_text(submission, context):
        del submission, context
        raise RuntimeError(
            "api-secret-value /private/patient/path patient-identifier-123"
        )

    service.register_provider("secure-provider", fail_with_sensitive_text)
    queued = service.submit(
        "token",
        JobSubmission(
            AnalysisPlan("analysis", "provider", "discretization", ("field",)),
            ExecutionPlan("execution", "cpu", "float64", "direct"),
            "numeric-revision",
            "secure-provider",
            {},
            ResourceRequest(cpu_cores=1, memory_bytes=1024),
        ),
    )
    failed = service.execute("token", queued.job_id)

    assert failed.failure is not None
    assert failed.failure.code == "provider_failure"
    assert failed.failure.exception_type == "ProviderExecutionError"
    assert failed.failure.message == "Provider execution failed."
    assert failed.failure.diagnostic_ids == ()
    telemetry = repr(failed.failure)
    assert "api-secret-value" not in telemetry
    assert "/private/patient/path" not in telemetry
    assert "patient-identifier-123" not in telemetry
