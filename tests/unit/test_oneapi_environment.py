"""Synthetic evidence exercises pure policy; it is not Intel hardware evidence."""

from __future__ import annotations

from dataclasses import replace

import pytest

from phydrax.backends import (
    assess_oneapi_environment,
    OneapiDeviceFacts,
    OneapiEnvironmentFacts,
    OneapiEnvironmentIssue,
    OneapiEnvironmentReport,
)


def _facts() -> OneapiEnvironmentFacts:
    return OneapiEnvironmentFacts(
        host_platform="linux",
        host_machine="x86_64",
        python_implementation="CPython",
        python_version=(3, 13, 0),
        python_free_threaded=False,
        libc_name="glibc",
        libc_version="2.31",
        host_release="6.8.0",
        versions=(
            ("phydrax", "0.2.2"),
            ("jax", "0.12.0"),
            ("jaxlib", "0.11.2"),
            ("jax-oneapi-plugin", "0.11.2"),
            ("jax-oneapi-pjrt", "0.11.2"),
            ("dpcpp-cpp-rt", "2025.1.1"),
            ("intel-sycl-rt", "2025.1.1"),
            ("tbb", "2022.1.0"),
            ("intel-cmplr-lib-rt", "2025.1.1"),
            ("intel-cmplr-lib-ur", "2025.1.1"),
            ("intel-cmplr-lic-rt", "2025.1.1"),
            ("intel-opencl-rt", "2025.1.1"),
            ("intel-openmp", "2025.1.1"),
            ("impi-rt", "2021.15.0"),
            ("intel-pti", "0.12.0"),
            ("mkl", "2025.1.0"),
            ("onemkl-sycl-blas", "2025.1.0"),
            ("onemkl-sycl-dft", "2025.1.0"),
            ("onemkl-sycl-lapack", "2025.1.0"),
            ("onemkl-sycl-rng", "2025.1.0"),
            ("onemkl-sycl-sparse", "2025.1.0"),
            ("umf", "0.10.0"),
            ("tcmlib", "1.3.0"),
        ),
        selected_platforms=("oneapi",),
    )


def _version(
    facts: OneapiEnvironmentFacts, name: str, version: str | None
) -> OneapiEnvironmentFacts:
    versions = dict(facts.versions)
    if version is None:
        del versions[name]
    else:
        versions[name] = version
    return replace(facts, versions=tuple(versions.items()))


def _device(platform: str, client: str, runtime: str) -> OneapiDeviceFacts:
    return OneapiDeviceFacts(
        platform=platform,
        client_platform=client,
        runtime_version=runtime,
        device_kind="Synthetic policy fixture",
        id=0,
        process_index=0,
    )


def test_installation_assessment_is_not_hardware_qualification() -> None:
    report = assess_oneapi_environment(_facts())
    assert report.successful
    assert not report.facts.device_probed
    assert report.facts.devices == ()
    assert report.facts.runtime_platform is None
    assert report.facts.runtime_version is None
    assert tuple(issue.code for issue in report.issues) == ("mixed-frontend-stack",)
    assert "not Intel's documented" in report.issues[0].message


@pytest.mark.parametrize(
    "facts",
    [
        replace(_facts(), host_platform="darwin", host_machine="arm64"),
        replace(_facts(), host_platform="win32"),
        replace(_facts(), host_machine="aarch64"),
        replace(_facts(), wsl=True),
        replace(_facts(), host_release="6.6.0-microsoft-standard-WSL2"),
        replace(_facts(), python_implementation="PyPy"),
        replace(_facts(), python_version=(3, 11, 9)),
        replace(_facts(), python_version=(3, 12, 9)),
        replace(_facts(), python_version=(3, 14, 0)),
        replace(_facts(), python_version=(3, 15, 0)),
        replace(_facts(), python_free_threaded=True),
        replace(_facts(), libc_name="musl"),
        replace(_facts(), libc_version="2.26"),
    ],
    ids=(
        "mac",
        "windows",
        "arm",
        "wsl",
        "wsl-kernel",
        "pypy",
        "old-python",
        "python312",
        "python314",
        "python315",
        "free-threaded",
        "musl",
        "old-glibc",
    ),
)
def test_unsupported_hosts_refuse(facts: OneapiEnvironmentFacts) -> None:
    report = assess_oneapi_environment(facts)
    assert not report.successful
    assert any(issue.code == "unsupported-host" for issue in report.issues)
    assert not report.facts.device_probed


@pytest.mark.parametrize(
    ("name", "version", "code"),
    [
        ("jax-oneapi-plugin", None, "no-oneapi-plugin"),
        ("jax-oneapi-pjrt", None, "no-oneapi-plugin"),
        ("jax-oneapi-plugin", "0.11.1", "plugin-version-mismatch"),
        ("jax-oneapi-pjrt", "0.11.1", "plugin-version-mismatch"),
        ("jax", "0.11.2", "jax-version-mismatch"),
        ("jaxlib", "0.12.0", "jax-version-mismatch"),
        ("intel-sycl-rt", None, "runtime-dependency-mismatch"),
        ("mkl", "2025.2.0", "runtime-dependency-mismatch"),
    ],
)
def test_version_contract_is_exact(name: str, version: str | None, code: str) -> None:
    report = assess_oneapi_environment(_version(_facts(), name, version))
    assert not report.successful
    assert any(issue.code == code for issue in report.issues)


@pytest.mark.parametrize(
    "name",
    ("jax-cuda13-plugin", "phydrax-metal", "libtpu", "intel-extension-for-openxla"),
)
def test_competing_accelerator_distributions_refuse(name: str) -> None:
    facts = _version(_facts(), name, "1.0")
    report = assess_oneapi_environment(facts)
    assert not report.successful
    assert any(issue.code == "conflicting-plugins" for issue in report.issues)


def test_cpu_selection_is_preserved_without_claiming_a_broken_driver() -> None:
    report = assess_oneapi_environment(replace(_facts(), selected_platforms=("cpu",)))
    assert report.successful
    assert "platform-not-selected" in tuple(issue.code for issue in report.issues)
    assert not report.facts.device_probed
    assert report.facts.backend_error is None


def test_sdk_and_loader_overrides_are_reported_not_resolved_from_metadata() -> None:
    facts = replace(
        _facts(),
        environment_overrides=(("ONEAPI_ROOT", "/sdk"), ("LD_LIBRARY_PATH", "/sdk/lib")),
    )
    report = assess_oneapi_environment(facts)
    assert report.successful
    assert any(
        issue.code == "library-overrides" and issue.severity == "warning"
        for issue in report.issues
    )
    assert report.facts.environment_overrides == (
        ("LD_LIBRARY_PATH", "/sdk/lib"),
        ("ONEAPI_ROOT", "/sdk"),
    )
    assert report.facts.devices == ()


@pytest.mark.parametrize(
    ("platform", "client", "runtime", "accepted"),
    [
        ("oneapi", "oneapi", "SYCL 0.11.2", True),
        ("gpu", "oneapi", "SYCL 0.11.2", True),
        ("gpu", "SYCL", "provider runtime", True),
        ("gpu", "gpu", "SYCL 0.11.2", True),
        ("gpu", "cuda", "CUDA 13000", False),
        ("gpu", "gpu", "unknown SYCL", False),
        ("cpu", "cpu", "SYCL unrelated metadata", False),
        ("sycl", "SYCL", "SYCL 0.11.2", False),
    ],
)
def test_runtime_identity_policy_requires_provider_evidence(
    platform: str,
    client: str,
    runtime: str,
    accepted: bool,
) -> None:
    device = _device(platform, client, runtime)
    facts = replace(
        _facts(),
        device_probed=True,
        devices=(device,),
        runtime_platform=platform,
        runtime_version=runtime,
    )
    report = assess_oneapi_environment(facts)
    assert report.successful is accepted
    assert (
        "runtime-identity-mismatch" in tuple(issue.code for issue in report.issues)
    ) is not accepted


def test_probe_errors_and_empty_device_sets_are_not_successful() -> None:
    for facts in (
        replace(_facts(), device_probed=True, backend_error="No visible XPU devices"),
        replace(_facts(), device_probed=True),
    ):
        report = assess_oneapi_environment(facts)
        assert not report.successful
        assert any(issue.code == "backend-unavailable" for issue in report.issues)


def test_every_observed_device_must_have_oneapi_identity() -> None:
    device = _device("oneapi", "oneapi", "SYCL 0.11.2")
    cpu = _device("cpu", "cpu", "CPU")
    facts = replace(
        _facts(),
        device_probed=True,
        devices=(device, cpu),
        runtime_platform=device.platform,
        runtime_version=device.runtime_version,
    )
    assert not assess_oneapi_environment(facts).successful


def test_runtime_summary_cannot_disagree_with_observation() -> None:
    device = _device("oneapi", "oneapi", "SYCL 0.11.2")
    facts = replace(
        _facts(),
        device_probed=True,
        devices=(device,),
        runtime_platform="cpu",
        runtime_version=device.runtime_version,
    )
    assert not assess_oneapi_environment(facts).successful


def test_canonical_distribution_identity_and_unique_keys() -> None:
    facts = _facts()
    assert replace(facts, versions=tuple(reversed(facts.versions))) == facts
    with pytest.raises(ValueError, match="unique"):
        replace(facts, versions=(*facts.versions, ("jax", "0.11.2")))
    with pytest.raises(ValueError, match="device_probed"):
        replace(facts, runtime_platform="oneapi")
    with pytest.raises(TypeError, match="booleans"):
        replace(facts, device_probed=1)
    with pytest.raises(TypeError, match="non-empty tuple"):
        replace(facts, selected_platforms=())
    with pytest.raises(ValueError):
        OneapiEnvironmentIssue(code="unknown", severity="error", message="invalid")  # ty: ignore[invalid-argument-type]
    with pytest.raises(TypeError, match="facts"):
        OneapiEnvironmentReport(facts="invalid", issues=())  # ty: ignore[invalid-argument-type]


def test_unknown_host_requirements_are_not_invented() -> None:
    report = assess_oneapi_environment(
        replace(_facts(), python_version=None, libc_name="", libc_version="")
    )
    assert not report.successful
    assert any(issue.code == "host-requirements-unknown" for issue in report.issues)
    assert report.facts.python_version is None


@pytest.mark.parametrize(
    "facts",
    [
        replace(_facts(), python_implementation=""),
        replace(_facts(), python_version=None),
        replace(_facts(), python_free_threaded=None),
        replace(_facts(), libc_name=""),
        replace(_facts(), libc_version=""),
    ],
    ids=("implementation", "version", "gil-build", "libc", "libc-version"),
)
def test_missing_host_evidence_refuses_admission(facts: OneapiEnvironmentFacts) -> None:
    report = assess_oneapi_environment(facts)
    assert not report.successful
    assert any(
        issue.code == "host-requirements-unknown" and issue.severity == "error"
        for issue in report.issues
    )
    assert not report.facts.device_probed
