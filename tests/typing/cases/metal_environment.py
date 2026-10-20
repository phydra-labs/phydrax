from typing import assert_type

from phydrax.backends import (
    assess_metal_environment,
    metal_environment,
    MetalEnvironmentFacts,
    MetalEnvironmentIssue,
    MetalEnvironmentReport,
    MetalIssueCode,
    MetalIssueSeverity,
)


def consumer_contract(facts: MetalEnvironmentFacts) -> None:
    assert_type(metal_environment(), MetalEnvironmentReport)
    assert_type(metal_environment(initialize_backend=False), MetalEnvironmentReport)
    report = assess_metal_environment(facts)
    assert_type(report, MetalEnvironmentReport)
    assert_type(report.successful, bool)
    assert_type(report.facts, MetalEnvironmentFacts)
    assert_type(report.issues, tuple[MetalEnvironmentIssue, ...])
    assert_type(facts.device_probed, bool)
    assert_type(facts.runtime_platform, str | None)
    assert_type(facts.versions, tuple[tuple[str, str], ...])
    for issue in report.issues:
        assert_type(issue.code, MetalIssueCode)
        assert_type(issue.severity, MetalIssueSeverity)
    assert_type(
        MetalEnvironmentIssue(
            code="conflicting-plugins", severity="error", message="Remove jax-mps"
        ),
        MetalEnvironmentIssue,
    )
    assert_type(
        MetalEnvironmentFacts(
            host_platform="darwin", host_machine="arm64", macos_version="15.6"
        ),
        MetalEnvironmentFacts,
    )
    MetalEnvironmentIssue(code="unknown", severity="error", message="invalid code")  # ty: ignore[invalid-argument-type]
    MetalEnvironmentIssue(
        code="companion-missing",
        severity="fatal",  # ty: ignore[invalid-argument-type]
        message="invalid severity",
    )
    metal_environment(initialize_backend="yes")  # ty: ignore[invalid-argument-type]
    assess_metal_environment(report)  # ty: ignore[invalid-argument-type]
