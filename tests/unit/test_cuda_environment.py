"""Consumer-visible CUDA environment rules and native dependency attribution."""

import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import pytest

from phydrax._cuda_environment import (
    _platform_version,
    assess_provider_dependencies,
    collect_cuda_environment,
    cuda_enabled,
    provider_dependency_closure,
)
from phydrax._elf_dependencies import (
    DependencyClosure,
    ResolvedDependency,
    UnresolvedDependency,
)
from phydrax._execution_device import device_platform
from phydrax.backends import (
    assess_cuda_environment,
    cuda_environment,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    CudaIssueCode,
    MappedCudaLibrary,
)


def _facts() -> CudaEnvironmentFacts:
    return CudaEnvironmentFacts(host_platform="linux", host_machine="x86_64")


def _only_issue(
    facts: CudaEnvironmentFacts, code: CudaIssueCode, severity: str = "error"
) -> CudaEnvironmentIssue:
    report = assess_cuda_environment(facts)
    assert len(report.issues) == 1
    issue = report.issues[0]
    assert issue.code == code
    assert issue.severity == severity
    assert report.successful == (severity == "warning")
    return issue


def test_cpu_install_is_successful() -> None:
    report = assess_cuda_environment(_facts())
    assert report.successful
    assert report.issues == ()


@pytest.mark.parametrize(
    "code",
    [
        "companion-missing",
        "companion-invalid",
        "companion-version-mismatch",
        "companion-major-mismatch",
        "native-dependency-missing",
    ],
)
def test_companion_issue_codes_preserve_actionable_evidence(code: CudaIssueCode) -> None:
    issue = CudaEnvironmentIssue(
        code=code,
        severity="error",
        message="Reinstall phydrax[cuda13].",
        paths=("/companion.so",),
    )
    assert issue.code == code
    assert issue.message == "Reinstall phydrax[cuda13]."
    assert issue.paths == ("/companion.so",)


def test_requested_cuda_requires_plugin() -> None:
    _only_issue(replace(_facts(), requested_major=13), "no-cuda-plugin")


@pytest.mark.parametrize(
    ("host_platform", "host_machine"),
    [("darwin", "arm64"), ("linux", "aarch64"), ("win32", "AMD64")],
)
def test_missing_plugin_warns_off_linux_x86_64(
    host_platform: str, host_machine: str
) -> None:
    issue = _only_issue(
        CudaEnvironmentFacts(
            host_platform=host_platform, host_machine=host_machine, requested_major=13
        ),
        "no-cuda-plugin",
        "warning",
    )
    assert host_platform in issue.message
    assert host_machine in issue.message


def test_duplicate_plugins_are_refused() -> None:
    _only_issue(replace(_facts(), plugin_majors=(12, 13)), "duplicate-cuda-plugins")


def test_marker_must_match_unique_plugin() -> None:
    _only_issue(
        replace(_facts(), requested_major=12, plugin_majors=(13,)),
        "plugin-major-mismatch",
    )


@pytest.mark.parametrize("major", [11, 12, 14])
@pytest.mark.parametrize("platforms", [None, ("cpu",), ("cuda",)])
def test_unsupported_plugin_is_refused_even_without_install_marker(
    major: int, platforms: tuple[str, ...] | None
) -> None:
    facts = replace(_facts(), plugin_majors=(major,), selected_platforms=platforms)
    _only_issue(facts, "plugin-major-mismatch")
    assert not cuda_enabled(facts)


def test_cuda13_install_marker_and_plugin_are_accepted() -> None:
    report = assess_cuda_environment(
        replace(_facts(), requested_major=13, plugin_majors=(13,))
    )
    assert report.successful
    assert report.issues == ()


@pytest.mark.parametrize("major", [11, 12, 14])
def test_unsupported_runtime_is_refused(major: int) -> None:
    _only_issue(
        replace(_facts(), plugin_majors=(13,), runtime_version=(major, 0)),
        "unsupported-cuda-major",
    )


def test_foreign_distributions_include_exact_uninstall_command() -> None:
    issue = _only_issue(
        replace(
            _facts(),
            foreign_nvidia_distributions=(
                "nvidia-cublas-cu12",
                "nvidia-cuda-runtime-cu12",
            ),
        ),
        "foreign-nvidia-distributions",
    )
    assert "pip uninstall nvidia-cublas-cu12 nvidia-cuda-runtime-cu12" in issue.message


def test_old_driver_is_refused() -> None:
    _only_issue(
        replace(_facts(), plugin_majors=(13,), driver_version=(12, 8)), "driver-too-old"
    )


def test_matching_driver_passes() -> None:
    assert assess_cuda_environment(
        replace(_facts(), plugin_majors=(13,), driver_version=(13, 2))
    ).successful


def test_explicit_cpu_platform_only_warns() -> None:
    facts = replace(
        _facts(), plugin_majors=(13,), selected_platforms=("cpu",), device_probed=True
    )
    _only_issue(facts, "cpu-platform-selected", "warning")
    assert not cuda_enabled(facts)


@pytest.mark.parametrize("platforms", [None, ("cuda",), ("gpu",), ("cpu", "cuda")])
def test_cuda_enabled_respects_cuda_or_gpu_selection(
    platforms: tuple[str, ...] | None,
) -> None:
    assert cuda_enabled(
        replace(_facts(), plugin_majors=(13,), selected_platforms=platforms)
    )


def test_static_facts_do_not_claim_backend_unavailable() -> None:
    report = assess_cuda_environment(replace(_facts(), plugin_majors=(13,)))
    assert report.successful
    assert report.issues == ()


def test_backend_unavailable_names_shadowing_and_initialization_error() -> None:
    paths = ("/opt/cuda/lib", "/usr/local/cuda/lib64")
    issue = _only_issue(
        replace(
            _facts(),
            plugin_majors=(13,),
            device_probed=True,
            backend_error="CUDA plugin failed",
            library_path_cuda_dirs=paths,
        ),
        "backend-unavailable",
    )
    assert issue.paths == paths
    assert all(path in issue.message for path in paths)
    assert "CUDA plugin failed" in issue.message


def test_library_path_shadowing_warns() -> None:
    issue = _only_issue(
        replace(
            _facts(),
            plugin_majors=(13,),
            device_probed=True,
            runtime_version=(13, 4),
            library_path_cuda_dirs=("/usr/local/cuda/lib64",),
        ),
        "library-path-shadowing",
        "warning",
    )
    assert issue.paths == ("/usr/local/cuda/lib64",)


def test_mapped_mixed_majors_are_refused_unknowns_are_not_counted() -> None:
    known = MappedCudaLibrary(
        soname="libcudart.so.13", path="/wheel/libcudart.so.13", major=13
    )
    unknown = MappedCudaLibrary(
        soname="libcusparse.so.12", path="/unknown/libcusparse.so.12", major=None
    )
    facts = replace(_facts(), mapped_libraries=(known, unknown))
    assert assess_cuda_environment(facts).issues == ()
    other = MappedCudaLibrary(
        soname="libcublas.so.12", path="/toolkit/libcublas.so.12", major=12
    )
    issue = _only_issue(
        replace(facts, mapped_libraries=(known, unknown, other)), "mixed-cuda-majors"
    )
    assert issue.paths == (known.path, other.path)


@pytest.mark.parametrize("major", [11, 12, 14])
def test_single_foreign_mapped_toolkit_is_refused(major: int) -> None:
    library = MappedCudaLibrary(
        soname=f"libcudart.so.{major}", path=f"/toolkit/libcudart.so.{major}", major=major
    )
    issue = _only_issue(
        replace(_facts(), mapped_libraries=(library,)), "unsupported-cuda-major"
    )
    assert issue.paths == (library.path,)


def test_assessment_issue_order_is_deterministic() -> None:
    facts = replace(
        _facts(),
        requested_major=12,
        plugin_majors=(13,),
        foreign_nvidia_distributions=("nvidia-cublas-cu12",),
        driver_version=(12, 8),
        device_probed=True,
        library_path_cuda_dirs=("/toolkit/lib",),
        mapped_libraries=(
            MappedCudaLibrary(soname="libcudart.so.12", path="/12", major=12),
            MappedCudaLibrary(soname="libcudart.so.13", path="/13", major=13),
        ),
    )
    assert tuple(issue.code for issue in assess_cuda_environment(facts).issues) == (
        "plugin-major-mismatch",
        "foreign-nvidia-distributions",
        "driver-too-old",
        "backend-unavailable",
        "mixed-cuda-majors",
    )


def _closure(
    *,
    resolved: tuple[ResolvedDependency, ...] = (),
    unresolved: tuple[UnresolvedDependency, ...] = (),
) -> DependencyClosure:
    return DependencyClosure(
        root="/provider/extension.so",
        machine="x86_64",
        resolved=resolved,
        unresolved=unresolved,
    )


def _dependency(soname: str, path: str, *, preloaded: bool = False) -> ResolvedDependency:
    return ResolvedDependency(
        soname=soname,
        path=path,
        requested_by="/provider/libsolver.so",
        preloaded=preloaded,
    )


@pytest.mark.parametrize(
    ("soname", "major"),
    [
        ("libcusolver.so.11", 12),
        ("libcusolver.so.12", 13),
        ("libcufft.so.11", 12),
        ("libcufft.so.12", 13),
    ],
)
def test_math_library_sonames_have_toolkit_specific_majors(
    soname: str, major: int
) -> None:
    closure = _closure(resolved=(_dependency(soname, f"/unknown/{soname}"),))
    issues = assess_provider_dependencies(closure, replace(_facts(), plugin_majors=(13,)))
    if major == 13:
        assert issues == ()
    else:
        assert len(issues) == 1
        assert issues[0].code == "provider-major-mismatch"


@pytest.mark.parametrize(
    ("path", "owners", "refused"),
    [
        (
            "/wheel/nvidia/cu13/lib/libcusparse.so.12",
            (("/wheel/nvidia/cu13/lib/libcusparse.so.12", 13),),
            False,
        ),
        ("/usr/local/cuda-12.8/lib64/libcusparse.so.12", (), True),
        ("/unknown/libcusparse.so.12", (), False),
    ],
)
def test_ambiguous_soname_uses_wheel_then_toolkit_path(
    path: str, owners: tuple[tuple[str, int], ...], refused: bool
) -> None:
    issues = assess_provider_dependencies(
        _closure(resolved=(_dependency("libcusparse.so.12", path),)),
        replace(_facts(), plugin_majors=(13,), wheel_library_majors=owners),
    )
    assert bool(issues) == refused


def test_provider_mismatch_names_soname_path_and_requester() -> None:
    dependency = _dependency("libcublas.so.12", "/cuda12/lib/libcublas.so.12")
    issues = assess_provider_dependencies(
        _closure(resolved=(dependency,)), replace(_facts(), plugin_majors=(13,))
    )
    assert len(issues) == 1
    issue = issues[0]
    assert issue.code == "provider-major-mismatch"
    assert issue.severity == "error"
    assert all(
        value in issue.message
        for value in (dependency.soname, dependency.path, dependency.requested_by)
    )
    assert issue.paths == (dependency.path,)


def test_unresolved_cuda_provider_dependency_is_refused() -> None:
    dependency = UnresolvedDependency(
        soname="libcudart.so.13", requested_by="/provider/ext.so", reason="not found"
    )
    issues = assess_provider_dependencies(
        _closure(unresolved=(dependency,)), replace(_facts(), plugin_majors=(13,))
    )
    assert len(issues) == 1
    assert issues[0].code == "provider-dependency-unresolved"
    assert issues[0].severity == "error"
    assert dependency.soname in issues[0].message


def test_non_cuda_dependencies_have_no_cuda_issue() -> None:
    closure = _closure(
        resolved=(
            _dependency("libcuda.so.1", "/driver/libcuda.so.1"),
            _dependency("libstdc++.so.6", "/system/libstdc++.so.6"),
        ),
        unresolved=(
            UnresolvedDependency(
                soname="libother.so.1",
                requested_by="/provider/ext.so",
                reason="not found",
            ),
        ),
    )
    assert (
        assess_provider_dependencies(closure, replace(_facts(), plugin_majors=(13,)))
        == ()
    )


def test_preloaded_ambiguous_cuda13_wheel_is_accepted() -> None:
    path = "/wheel/nvidia/cu13/lib/libcusparse.so.12"
    facts = replace(_facts(), plugin_majors=(13,), wheel_library_majors=((path, 13),))
    assert (
        assess_provider_dependencies(
            _closure(resolved=(_dependency("libcusparse.so.12", path, preloaded=True),)),
            facts,
        )
        == ()
    )


def test_disabled_cuda_does_not_assess_provider_dependencies() -> None:
    closure = _closure(
        resolved=(_dependency("libcudart.so.12", "/cuda12/libcudart.so.12"),)
    )
    assert (
        assess_provider_dependencies(
            closure, replace(_facts(), plugin_majors=(13,), selected_platforms=("cpu",))
        )
        == ()
    )


@pytest.mark.parametrize(
    ("text", "version"),
    [("cuda 13040", (13, 4)), ("cuda 12080", (12, 8)), ("cuda 13000", (13, 0))],
)
def test_platform_version_decoding(text: str, version: tuple[int, int]) -> None:
    assert _platform_version(text) == version


@pytest.mark.parametrize(
    "text",
    [
        "",
        "cuda",
        "cuda 13.4",
        "rocm 13040",
        "cuda 13041",
        "cuda -13040",
        "cuda 0",
        "cuda 13040 junk",
    ],
)
def test_malformed_platform_version_is_refused(text: str) -> None:
    with pytest.raises(ValueError, match="Malformed CUDA platform version"):
        _platform_version(text)


def test_closed_issue_selector_refuses_unknown_code() -> None:
    with pytest.raises(ValueError, match="code"):
        CudaEnvironmentIssue(code="unknown", severity="error", message="bad code")  # ty: ignore[invalid-argument-type]


def test_fact_constructors_refuse_invalid_metadata() -> None:
    with pytest.raises(ValueError, match="requested_major"):
        replace(_facts(), requested_major=-1)
    with pytest.raises(ValueError, match="runtime_version"):
        replace(_facts(), runtime_version=(13, -1))
    with pytest.raises(ValueError, match="host_platform"):
        replace(_facts(), host_platform="")


def test_static_collector_does_not_initialize_backends_in_fresh_process() -> None:
    root = Path(__file__).resolve().parents[2]
    script = """
import json
import phydrax.config
from phydrax._cuda_environment import collect_cuda_environment
before = phydrax.config.snapshot().backends_initialized
facts = collect_cuda_environment(initialize_backend=False)
print(json.dumps([before, phydrax.config.snapshot().backends_initialized, facts.device_probed, len(facts.mapped_libraries)]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root)},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == [False, False, False, 0]


def test_cuda13_unsuffixed_math_wheels_are_not_foreign() -> None:
    installed = {
        distribution.metadata["Name"].lower(): distribution
        for distribution in importlib.metadata.distributions()
    }
    if "jax-cuda13-plugin" not in installed:
        pytest.skip("CUDA 13 JAX plugin is not installed")
    facts = collect_cuda_environment(initialize_backend=False)
    for name in ("nvidia-cusparse", "nvidia-cusolver", "nvidia-cufft"):
        if name in installed:
            assert name not in facts.foreign_nvidia_distributions
            libraries = [
                os.path.realpath(str(installed[name].locate_file(file)))
                for file in installed[name].files or ()
                if ".so" in str(file)
            ]
            assert libraries
            assert all(
                (library, 13) in facts.wheel_library_majors for library in libraries
            )


@pytest.mark.parametrize("plugin_major", [None, 12, 13, 14])
@pytest.mark.parametrize("wheel_major", [12, 13, 14])
def test_collection_preserves_marker_and_attributes_foreign_wheels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plugin_major: int | None,
    wheel_major: int,
) -> None:
    wheel_name = (
        "nvidia-cusparse" if wheel_major == 13 else f"nvidia-cusparse-cu{wheel_major}"
    )
    names = [("phydrax-cuda-major", "13.0"), (wheel_name, "1.0")]
    if plugin_major is not None:
        names.append((f"jax-cuda{plugin_major}-plugin", "0.9.2"))
    distributions: list[importlib.metadata.Distribution] = []
    library = f"nvidia/cu{wheel_major}/lib/libcusparse.so.12"
    library_path = tmp_path / library
    library_path.parent.mkdir(parents=True)
    library_path.touch()
    for name, version in names:
        directory = tmp_path / f"{name}-{version}.dist-info"
        directory.mkdir()
        (directory / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
        if name == wheel_name:
            (directory / "RECORD").write_text(f"{library},,\n")
        distributions.append(importlib.metadata.PathDistribution(directory))
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: distributions)

    facts = collect_cuda_environment(initialize_backend=False)

    assert facts.requested_major == 13
    assert facts.plugin_majors == (() if plugin_major is None else (plugin_major,))
    assert facts.foreign_nvidia_distributions == (
        () if wheel_major == 13 else (wheel_name,)
    )
    assert facts.wheel_library_majors == (
        (str((tmp_path / library).resolve()), wheel_major),
    )


def test_provider_closure_host_helper_resolves_system_libraries() -> None:
    closure = provider_dependency_closure(
        sys.executable, facts=_facts(), jax_backend_preloaded=False
    )
    assert closure.root == sys.executable
    assert closure.unresolved == ()
    assert "libc.so.6" in {dependency.soname for dependency in closure.resolved}


def test_cuda_runtime_major_matches_plugin() -> None:
    if not any(device_platform(device) == "cuda" for device in jax.devices()):
        pytest.skip("No CUDA device")
    report = cuda_environment()
    assert report.facts.runtime_version is not None
    assert report.facts.plugin_majors == (13,)
    assert report.facts.runtime_version[0] == 13
