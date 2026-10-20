from typing import assert_type

from phydrax.backends import (
    assess_oneapi_environment,
    collect_oneapi_environment,
    oneapi_environment,
    OneapiDeviceFacts,
    OneapiEnvironmentFacts,
    OneapiEnvironmentIssue,
    OneapiEnvironmentReport,
    OneapiIssueCode,
    OneapiIssueSeverity,
)


def consumer_contract(facts: OneapiEnvironmentFacts) -> None:
    assert_type(collect_oneapi_environment(), OneapiEnvironmentFacts)
    assert_type(oneapi_environment(), OneapiEnvironmentReport)
    assert_type(oneapi_environment(initialize_backend=True), OneapiEnvironmentReport)
    report = assess_oneapi_environment(facts)
    assert_type(report, OneapiEnvironmentReport)
    assert_type(report.successful, bool)
    assert_type(report.facts, OneapiEnvironmentFacts)
    assert_type(report.issues, tuple[OneapiEnvironmentIssue, ...])
    assert_type(facts.versions, tuple[tuple[str, str], ...])
    assert_type(facts.selected_platforms, tuple[str, ...] | None)
    assert_type(facts.environment_overrides, tuple[tuple[str, str], ...])
    assert_type(facts.provider_libraries, tuple[tuple[str, str], ...])
    assert_type(facts.python_version, tuple[int, int, int] | None)
    assert_type(facts.python_free_threaded, bool | None)
    assert_type(facts.wsl, bool)
    assert_type(facts.device_probed, bool)
    assert_type(facts.runtime_platform, str | None)
    assert_type(facts.runtime_version, str | None)
    assert_type(facts.backend_error, str | None)
    assert_type(facts.devices, tuple[OneapiDeviceFacts, ...])
    for device in facts.devices:
        assert_type(device.platform, str)
        assert_type(device.client_platform, str)
        assert_type(device.runtime_version, str)
        assert_type(device.device_kind, str)
        assert_type(device.id, int)
        assert_type(device.process_index, int)
    for issue in report.issues:
        assert_type(issue.code, OneapiIssueCode)
        assert_type(issue.severity, OneapiIssueSeverity)
    assert_type(
        OneapiEnvironmentFacts(host_platform="linux", host_machine="x86_64"),
        OneapiEnvironmentFacts,
    )
    assert_type(
        OneapiEnvironmentIssue(
            code="backend-unavailable", severity="error", message="No devices"
        ),
        OneapiEnvironmentIssue,
    )
    OneapiEnvironmentIssue(code="unknown", severity="error", message="bad code")  # ty: ignore[invalid-argument-type]
    OneapiEnvironmentIssue(
        code="backend-unavailable",
        severity="fatal",  # ty: ignore[invalid-argument-type]
        message="bad severity",
    )
    oneapi_environment(initialize_backend="yes")  # ty: ignore[invalid-argument-type]
    assess_oneapi_environment(report)  # ty: ignore[invalid-argument-type]
