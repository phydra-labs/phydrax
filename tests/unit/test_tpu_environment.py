from __future__ import annotations

from dataclasses import replace

import pytest

from phydrax.backends import (
    assess_tpu_environment,
    TpuDeviceFacts,
    TpuEnvironmentFacts,
    TpuEnvironmentIssue,
    TpuEnvironmentReport,
)


def _facts() -> TpuEnvironmentFacts:
    """Synthetic host policy evidence, never a hardware qualification fixture."""
    return TpuEnvironmentFacts(
        host_platform="linux",
        host_machine="x86_64",
        host_release="6.8.0-native",
        wsl=False,
        python_implementation="CPython",
        python_version=(3, 13, 0),
        python_free_threaded=False,
        libc_name="glibc",
        libc_version="2.31",
        versions=(("jax", "0.12.0"), ("jaxlib", "0.12.0"), ("libtpu", "0.0.50")),
        selected_platforms=("tpu",),
        distribution_library_path="/installed/libtpu/libtpu.so",
        library_path="/installed/libtpu/libtpu.so",
        library_sha256="a" * 64,
        library_from_distribution=True,
        enable_x64=True,
        matmul_precision="highest",
    )


def _observed() -> TpuEnvironmentFacts:
    return replace(
        _facts(),
        device_probed=True,
        backends_initialized=True,
        devices=(TpuDeviceFacts(0, 0, "tpu", "TPU v5 lite", 0),),
        runtime_platform="tpu",
        runtime_version="observed upstream runtime identifier",
        mapped_library_paths=("/installed/libtpu/libtpu.so",),
        local_device_count=1,
        global_device_count=1,
        process_count=1,
        process_index=0,
    )


def test_host_success_is_not_hardware_qualification() -> None:
    report = assess_tpu_environment(_facts())
    assert report.successful
    assert not report.hardware_qualified
    assert not report.facts.device_probed
    assert report.facts.devices == ()
    assert report.facts.runtime_platform is None
    assert report.facts.local_device_count is None
    assert report.facts.global_device_count is None
    assert report.facts.process_count is None
    assert tuple(issue.code for issue in report.issues) == ("precision-limitations",)


@pytest.mark.parametrize(
    ("facts", "code"),
    [
        (
            replace(_facts(), host_platform="darwin", host_machine="arm64"),
            "unsupported-host",
        ),
        (replace(_facts(), host_platform="win32"), "unsupported-host"),
        (replace(_facts(), host_machine="aarch64"), "unsupported-host"),
        (
            replace(_facts(), host_release="5.15.167.4-microsoft-standard-WSL2"),
            "unsupported-host",
        ),
        (
            replace(_facts(), host_release="6.6.87.2-MICROSOFT-STANDARD-WSL2"),
            "unsupported-host",
        ),
        (replace(_facts(), host_release="6.6.0-WSL2"), "unsupported-host"),
        (replace(_facts(), wsl=True), "unsupported-host"),
        (replace(_facts(), python_implementation="PyPy"), "unsupported-host"),
        (replace(_facts(), python_version=(3, 12, 9)), "unsupported-host"),
        (replace(_facts(), python_version=(3, 14, 0)), "unsupported-host"),
        (replace(_facts(), python_version=(3, 15, 0)), "unsupported-host"),
        (replace(_facts(), python_free_threaded=True), "unsupported-host"),
        (replace(_facts(), libc_name="musl"), "unsupported-host"),
        (replace(_facts(), libc_version="2.30"), "unsupported-host"),
        (replace(_facts(), libc_version="unknown"), "unsupported-host"),
        (replace(_facts(), versions=()), "libtpu-missing"),
        (
            replace(
                _facts(),
                versions=(("jax", "0.12.0"), ("jaxlib", "0.11.2"), ("libtpu", "0.0.50")),
            ),
            "jax-version-mismatch",
        ),
        (
            replace(
                _facts(),
                versions=(("jax", "0.12.0"), ("jaxlib", "0.12.0"), ("libtpu", "0.0.49")),
            ),
            "libtpu-version-mismatch",
        ),
        (replace(_facts(), library_path=None), "library-invalid"),
        (replace(_facts(), library_sha256=None), "library-invalid"),
        (
            replace(_facts(), installation_error="Distribution RECORD digest mismatch."),
            "library-invalid",
        ),
        (
            replace(
                _facts(),
                library_override="/custom/libtpu.so",
                library_path="/custom/libtpu.so",
                library_from_distribution=False,
            ),
            "library-override-unattributed",
        ),
        (
            replace(
                _facts(), device_probed=True, backend_error="TPU initialization failed."
            ),
            "backend-unavailable",
        ),
    ],
)
def test_host_and_provider_refusals(facts: TpuEnvironmentFacts, code: str) -> None:
    report = assess_tpu_environment(facts)
    assert not report.successful
    assert any(
        issue.code == code and issue.severity == "error" for issue in report.issues
    )
    assert not report.hardware_qualified


def test_explicit_cpu_selection_does_not_rewrite_the_requested_platform() -> None:
    facts = replace(_facts(), selected_platforms=("cpu",))
    report = assess_tpu_environment(facts)
    assert report.successful
    assert report.facts is facts
    assert report.facts.selected_platforms == ("cpu",)
    assert "platform-not-selected" in tuple(issue.code for issue in report.issues)
    assert not report.facts.device_probed


def test_cpu_observed_under_explicit_tpu_request_is_a_failure() -> None:
    facts = replace(
        _observed(),
        runtime_platform="cpu",
        devices=(TpuDeviceFacts(0, 0, "cpu", "cpu", 0),),
    )
    report = assess_tpu_environment(facts)
    assert not report.successful
    assert "runtime-identity-mismatch" in tuple(issue.code for issue in report.issues)


@pytest.mark.parametrize(
    "facts",
    [
        replace(_observed(), runtime_version=None),
        replace(_observed(), mapped_library_paths=()),
        replace(_observed(), mapped_library_paths=("/different/libtpu.so",)),
        replace(_observed(), local_device_count=None),
        replace(_observed(), local_device_count=0),
        replace(_observed(), global_device_count=2),
        replace(_observed(), process_count=0),
        replace(_observed(), process_index=1),
        replace(_observed(), process_count=2, distributed_initialized=False),
        replace(_observed(), devices=(TpuDeviceFacts(0, 2, "tpu", "TPU v5 lite", 0),)),
    ],
)
def test_observed_runtime_requires_attributed_library_and_consistent_topology(
    facts: TpuEnvironmentFacts,
) -> None:
    report = assess_tpu_environment(facts)
    assert not report.successful
    assert any(issue.code == "runtime-identity-mismatch" for issue in report.issues)


def test_local_and_global_counts_remain_distinct_for_multiple_processes() -> None:
    facts = replace(
        _observed(),
        distributed_initialized=True,
        devices=(
            TpuDeviceFacts(0, 0, "tpu", "TPU v5 lite", 0),
            TpuDeviceFacts(1, 1, "tpu", "TPU v5 lite", 0),
        ),
        global_device_count=2,
        process_count=2,
    )
    report = assess_tpu_environment(facts)
    assert report.successful
    assert not report.hardware_qualified
    assert report.facts.local_device_count == 1
    assert report.facts.global_device_count == 2
    assert report.facts.process_count == 2


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("precision", [None, "default", "high", "highest"])
def test_effective_numerics_are_evidence_not_fp64_or_ieee_claims(
    enabled: bool,
    precision: str | None,
) -> None:
    report = assess_tpu_environment(
        replace(_facts(), enable_x64=enabled, matmul_precision=precision)
    )
    assert report.successful
    assert report.facts.enable_x64 is enabled
    assert report.facts.matmul_precision == precision
    issue = next(
        issue for issue in report.issues if issue.code == "precision-limitations"
    )
    assert issue.severity == "warning"
    assert "six BF16 passes" in issue.message
    assert not report.hardware_qualified


def test_versions_and_mapped_files_are_canonical() -> None:
    facts = _facts()
    assert replace(facts, versions=tuple(reversed(facts.versions))) == facts
    mapped = replace(
        _observed(), mapped_library_paths=("/b/libtpu.so", "/a/libtpu.so", "/b/libtpu.so")
    )
    assert mapped.mapped_library_paths == ("/a/libtpu.so", "/b/libtpu.so")
    with pytest.raises(ValueError, match="unique"):
        replace(facts, versions=(*facts.versions, ("jax", "0.12.0")))


def test_invalid_record_construction_is_refused() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        replace(_facts(), library_sha256="not-a-digest")
    with pytest.raises(ValueError, match="nonnegative"):
        TpuDeviceFacts(-1, 0, "tpu", "TPU")
    with pytest.raises(ValueError):
        TpuEnvironmentIssue("unknown", "error", "Invalid selector")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="facts"):
        TpuEnvironmentReport(_facts().devices, ())  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="facts"):
        assess_tpu_environment(())  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("patch", [0, 1, 99])
def test_standard_gil_cpython_313_patch_releases_are_admitted(patch: int) -> None:
    facts = replace(_facts(), python_version=(3, 13, patch), python_free_threaded=False)
    report = assess_tpu_environment(facts)
    assert report.successful
    assert not any(issue.code == "unsupported-host" for issue in report.issues)


def test_free_threaded_build_fact_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="booleans"):
        replace(_facts(), python_free_threaded=1)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("devices", (TpuDeviceFacts(0, 0, "tpu", "TPU", 0),)),
        ("runtime_platform", "tpu"),
        ("runtime_platform", ""),
        ("runtime_version", "observed runtime"),
        ("runtime_version", ""),
        ("mapped_library_paths", ("/installed/libtpu/libtpu.so",)),
        ("local_device_count", 0),
        ("global_device_count", 1),
        ("process_count", 1),
        ("process_index", 0),
        ("backend_error", "Initialization failed."),
        ("backend_error", ""),
    ],
)
def test_runtime_observations_cannot_be_marked_unprobed(
    field: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match="require device_probed=True"):
        replace(_facts(), **{field: value})


@pytest.mark.parametrize("backends_initialized", [False, True])
@pytest.mark.parametrize("distributed_initialized", [False, True])
def test_unprobed_facts_can_observe_existing_host_lifecycle_state(
    backends_initialized: bool,
    distributed_initialized: bool,
) -> None:
    facts = replace(
        _facts(),
        backends_initialized=backends_initialized,
        distributed_initialized=distributed_initialized,
    )
    report = assess_tpu_environment(facts)
    assert report.successful
    assert not report.facts.device_probed
    assert report.facts.backends_initialized is backends_initialized
    assert report.facts.distributed_initialized is distributed_initialized
    assert report.facts.devices == ()


def test_wsl_observation_flag_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="booleans"):
        replace(_facts(), wsl=1)
