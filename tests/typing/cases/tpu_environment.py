from typing import assert_type

from phydrax.backends import (
    assess_tpu_environment,
    collect_tpu_environment,
    tpu_environment,
    TpuDeviceFacts,
    TpuEnvironmentFacts,
    TpuEnvironmentIssue,
    TpuEnvironmentReport,
    TpuIssueCode,
    TpuIssueSeverity,
)


def consumer_contract(facts: TpuEnvironmentFacts) -> None:
    assert_type(collect_tpu_environment(), TpuEnvironmentFacts)
    assert_type(tpu_environment(), TpuEnvironmentReport)
    assert_type(tpu_environment(initialize_backend=True), TpuEnvironmentReport)
    report = assess_tpu_environment(facts)
    assert_type(report, TpuEnvironmentReport)
    assert_type(report.facts, TpuEnvironmentFacts)
    assert_type(report.issues, tuple[TpuEnvironmentIssue, ...])
    assert_type(report.successful, bool)
    assert_type(report.hardware_qualified, bool)
    assert_type(facts.python_version, tuple[int, int, int])
    assert_type(facts.python_free_threaded, bool)
    assert_type(facts.host_release, str)
    assert_type(facts.wsl, bool)
    assert_type(facts.versions, tuple[tuple[str, str], ...])
    assert_type(facts.selected_platforms, tuple[str, ...] | None)
    assert_type(facts.library_path, str | None)
    assert_type(facts.library_sha256, str | None)
    assert_type(facts.library_from_distribution, bool)
    assert_type(facts.library_override, str | None)
    assert_type(facts.enable_x64, bool)
    assert_type(facts.matmul_precision, str | None)
    assert_type(facts.distributed_initialized, bool)
    assert_type(facts.device_probed, bool)
    assert_type(facts.runtime_platform, str | None)
    assert_type(facts.runtime_version, str | None)
    assert_type(facts.devices, tuple[TpuDeviceFacts, ...])
    assert_type(facts.mapped_library_paths, tuple[str, ...])
    assert_type(facts.local_device_count, int | None)
    assert_type(facts.global_device_count, int | None)
    assert_type(facts.process_count, int | None)
    assert_type(facts.process_index, int | None)
    for device in facts.devices:
        assert_type(device.id, int)
        assert_type(device.process_index, int)
        assert_type(device.local_hardware_id, int | None)
        assert_type(device.platform, str)
        assert_type(device.device_kind, str)
    for issue in report.issues:
        assert_type(issue.code, TpuIssueCode)
        assert_type(issue.severity, TpuIssueSeverity)
    TpuEnvironmentIssue("unknown", "error", "Invalid selector")  # ty: ignore[invalid-argument-type]
    TpuEnvironmentIssue("unsupported-host", "fatal", "Invalid severity")  # ty: ignore[invalid-argument-type]
    tpu_environment(initialize_backend="yes")  # ty: ignore[invalid-argument-type]
    assess_tpu_environment(report)  # ty: ignore[invalid-argument-type]
