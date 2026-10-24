"""Consumer-visible CUDA environment rules and native dependency attribution."""

import importlib.metadata
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import NoReturn

import jax
import pytest

from phydrax._cuda_environment import (
    _platform_version,
    assess_provider_dependencies,
    collect_cuda_environment,
    competing_cuda_libraries,
    cuda_enabled,
    provider_dependency_closure,
)
from phydrax._execution_device import device_platform
from phydrax._native_dependencies import (
    DependencyClosure,
    resolve_native_dependencies,
    ResolvedDependency,
    UnresolvedDependency,
)
from phydrax.backends import (
    assess_cuda_environment,
    cuda_environment,
    CudaEnvironmentFacts,
    CudaEnvironmentIssue,
    CudaIssueCode,
    CudaPluginInstallation,
    MappedCudaLibrary,
)
from tests._support.elf import write_shared_object


def _plugins(majors: tuple[int, ...]) -> tuple[CudaPluginInstallation, ...]:
    return tuple(
        CudaPluginInstallation(
            distribution=f"jax-cuda{major}-plugin", version="0.12.0", cuda_major=major
        )
        for major in majors
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


@pytest.mark.parametrize(
    ("host", "machine"),
    [
        ("linux", "x86_64"),
        ("linux", "aarch64"),
        ("win32", "AMD64"),
        ("win32", "ARM64"),
    ],
    ids=["linux-x86", "linux-arm", "windows-x86", "windows-unsupported-architecture"],
)
def test_requested_cuda_requires_plugin(host: str, machine: str) -> None:
    _only_issue(
        CudaEnvironmentFacts(
            host_platform=host, host_machine=machine, requested_major=13
        ),
        "no-cuda-plugin",
    )


@pytest.mark.parametrize(
    ("host_platform", "host_machine"),
    [("darwin", "arm64"), ("linux", "ppc64le")],
)
def test_missing_plugin_warns_on_unsupported_hosts(
    host_platform: str, host_machine: str
) -> None:
    _only_issue(
        CudaEnvironmentFacts(
            host_platform=host_platform, host_machine=host_machine, requested_major=13
        ),
        "no-cuda-plugin",
        "warning",
    )


def test_same_major_plugin_owners_are_not_deduplicated() -> None:
    plugins = (
        *_plugins((13,)),
        CudaPluginInstallation(
            distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13
        ),
    )
    _only_issue(replace(_facts(), plugins=plugins), "duplicate-cuda-plugins")


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("installation_error", "plugin-invalid"),
        ("inspection_error", "native-inspection-unavailable"),
    ],
    ids=["corrupt-installation", "unavailable-inspection"],
)
def test_incomplete_cuda_evidence_cannot_succeed(field: str, code: CudaIssueCode) -> None:
    facts = replace(_facts(), **{field: "Required CUDA evidence is unavailable."})
    issue = _only_issue(facts, code)
    assert issue.message == "Required CUDA evidence is unavailable."


def test_duplicate_plugins_are_refused() -> None:
    _only_issue(replace(_facts(), plugins=_plugins((12, 13))), "duplicate-cuda-plugins")


def test_marker_must_match_unique_plugin() -> None:
    _only_issue(
        replace(_facts(), requested_major=12, plugins=_plugins((13,))),
        "plugin-major-mismatch",
    )


@pytest.mark.parametrize("major", [11, 12, 14])
@pytest.mark.parametrize("platforms", [None, ("cpu",), ("cuda",)])
def test_unsupported_plugin_is_refused_even_without_install_marker(
    major: int, platforms: tuple[str, ...] | None
) -> None:
    facts = replace(_facts(), plugins=_plugins((major,)), selected_platforms=platforms)
    _only_issue(facts, "plugin-major-mismatch")
    assert not cuda_enabled(facts)


def test_cuda13_install_marker_and_plugin_are_accepted() -> None:
    report = assess_cuda_environment(
        replace(_facts(), requested_major=13, plugins=_plugins((13,)))
    )
    assert report.successful
    assert report.issues == ()


@pytest.mark.parametrize("major", [11, 12, 14])
def test_unsupported_runtime_is_refused(major: int) -> None:
    _only_issue(
        replace(_facts(), plugins=_plugins((13,)), runtime_version=(major, 0)),
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
        replace(_facts(), plugins=_plugins((13,)), driver_version=(12, 8)),
        "driver-too-old",
    )


def test_matching_driver_passes() -> None:
    assert assess_cuda_environment(
        replace(_facts(), plugins=_plugins((13,)), driver_version=(13, 2))
    ).successful


def test_explicit_cpu_platform_only_warns() -> None:
    facts = replace(
        _facts(), plugins=_plugins((13,)), selected_platforms=("cpu",), device_probed=True
    )
    _only_issue(facts, "cpu-platform-selected", "warning")
    assert not cuda_enabled(facts)


@pytest.mark.parametrize("platforms", [None, ("cuda",), ("gpu",), ("cpu", "cuda")])
def test_cuda_enabled_respects_cuda_or_gpu_selection(
    platforms: tuple[str, ...] | None,
) -> None:
    assert cuda_enabled(
        replace(_facts(), plugins=_plugins((13,)), selected_platforms=platforms)
    )


def test_static_facts_do_not_claim_backend_unavailable() -> None:
    report = assess_cuda_environment(replace(_facts(), plugins=_plugins((13,))))
    assert report.successful
    assert report.issues == ()


def test_backend_unavailable_names_shadowing_and_initialization_error() -> None:
    paths = ("/opt/cuda/lib", "/usr/local/cuda/lib64")
    issue = _only_issue(
        replace(
            _facts(),
            plugins=_plugins((13,)),
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
            plugins=_plugins((13,)),
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
        plugins=_plugins((13,)),
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
        object_format="elf",
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
    issues = assess_provider_dependencies(
        closure, replace(_facts(), plugins=_plugins((13,)))
    )
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
        replace(_facts(), plugins=_plugins((13,)), wheel_library_majors=owners),
    )
    assert bool(issues) == refused


def test_provider_mismatch_names_soname_path_and_requester() -> None:
    dependency = _dependency("libcublas.so.12", "/cuda12/lib/libcublas.so.12")
    issues = assess_provider_dependencies(
        _closure(resolved=(dependency,)), replace(_facts(), plugins=_plugins((13,)))
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
        _closure(unresolved=(dependency,)), replace(_facts(), plugins=_plugins((13,)))
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
        assess_provider_dependencies(closure, replace(_facts(), plugins=_plugins((13,))))
        == ()
    )


def test_preloaded_ambiguous_cuda13_wheel_is_accepted() -> None:
    path = "/wheel/nvidia/cu13/lib/libcusparse.so.12"
    facts = replace(_facts(), plugins=_plugins((13,)), wheel_library_majors=((path, 13),))
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
            closure,
            replace(_facts(), plugins=_plugins((13,)), selected_platforms=("cpu",)),
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
    assert tuple(plugin.cuda_major for plugin in facts.plugins) == (
        () if plugin_major is None else (plugin_major,)
    )
    assert facts.foreign_nvidia_distributions == (
        () if wheel_major == 13 else (wheel_name,)
    )
    assert facts.wheel_library_majors == (
        (str((tmp_path / library).resolve()), wheel_major),
    )


@pytest.mark.skipif(
    sys.platform != "linux", reason="This contract owns glibc system resolution."
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
    assert tuple(plugin.cuda_major for plugin in report.facts.plugins) == (13,)
    if report.facts.host_platform == "win32":
        assert report.facts.compiled_cuda_major == 13
        assert report.facts.driver_compatibility_version is not None
        assert report.facts.driver_compatibility_version >= (13, 0)
        assert dict(report.facts.runtime_identity)["linkage"] == "driver-hybrid"
    else:
        assert report.facts.runtime_version[0] == 13


@pytest.mark.parametrize(
    "challenge", ["absent", "same-file", "alternate", "wrong-soname"]
)
def test_hostile_cuda_challenge_requires_an_alternate_matching_soname(
    tmp_path: Path, challenge: str
) -> None:
    soname = "libcusparse.so.12"
    installed = write_shared_object(
        tmp_path / "wheel" / "libcusparse.so.12.8", soname=soname
    )
    mapped = (MappedCudaLibrary(soname=installed.name, path=str(installed), major=13),)
    directory = tmp_path / "challenge"
    directory.mkdir()
    candidate = directory / soname
    if challenge == "same-file":
        candidate.hardlink_to(installed)
    elif challenge == "alternate":
        write_shared_object(candidate, soname=soname)
    elif challenge == "wrong-soname":
        write_shared_object(candidate, soname="libcusparse.so.99")
    if challenge == "wrong-soname":
        with pytest.raises(ValueError):
            competing_cuda_libraries(str(directory), mapped)
    else:
        expected = ((soname, str(candidate)),) if challenge == "alternate" else ()
        assert competing_cuda_libraries(str(directory), mapped) == expected


@pytest.mark.parametrize("name", ["cudart64_12.dll", "cublas64_12.dll"])
def test_windows_unowned_mapped_major_is_reported(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    import phydrax._cuda_environment as environment

    facts = replace(
        _facts(),
        host_platform="win32",
        host_machine="AMD64",
        plugins=(
            CudaPluginInstallation(
                distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13
            ),
        ),
    )
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")
    setattr(
        runtime,
        "runtime_identity",
        lambda: SimpleNamespace(
            _asdict=lambda: {"linkage": "driver-hybrid", "owner": "display-driver"}
        ),
    )
    setattr(
        runtime,
        "mapped_libraries",
        lambda: (
            SimpleNamespace(
                name="cublas64_13.dll", path="C:/cuda/cublas64_13.dll", cuda_major=13
            ),
            SimpleNamespace(name=name, path=f"C:/foreign/{name}", cuda_major=None),
        ),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(environment, "collect_cuda_environment", lambda **kwargs: facts)
    report = environment.cuda_environment()
    assert not report.successful
    issue = next(issue for issue in report.issues if issue.code == "mixed-cuda-majors")
    assert f"C:/foreign/{name}" in issue.paths


@pytest.mark.parametrize("compatibility", [(13, 1), (14, 0)])
def test_windows_driver_compatibility_is_not_compiled_runtime_major(
    compatibility: tuple[int, int],
) -> None:
    facts = CudaEnvironmentFacts(
        host_platform="win32",
        host_machine="AMD64",
        compiled_cuda_major=13,
        runtime_version=compatibility,
        driver_compatibility_version=compatibility,
        plugins=(
            CudaPluginInstallation(
                distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13
            ),
        ),
    )
    assert assess_cuda_environment(facts).successful
    _only_issue(replace(facts, driver_compatibility_version=(12, 9)), "driver-too-old")
    _only_issue(replace(facts, compiled_cuda_major=14), "unsupported-cuda-major")


def test_windows_unowned_same_major_userland_is_refused() -> None:
    library = MappedCudaLibrary(
        soname="cublas64_13.dll", path="C:/foreign/cublas64_13.dll", major=13
    )
    facts = replace(_facts(), host_platform="win32", mapped_libraries=(library,))
    assert _only_issue(facts, "unowned-cuda-library").paths == (library.path,)
    assert assess_cuda_environment(
        replace(facts, wheel_library_majors=(("c:\\FOREIGN\\cublas64_13.dll", 13),))
    ).successful


@pytest.mark.parametrize(
    "failure", ["missing hybrid", "foreign driver cohort", "inventory unavailable"]
)
def test_windows_runtime_identity_inspection_failure_is_not_clean(
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    import phydrax._cuda_environment as environment

    facts = CudaEnvironmentFacts(
        host_platform="win32",
        host_machine="AMD64",
        plugins=(
            CudaPluginInstallation(
                distribution="phydrax-cuda13-pjrt", version="0.2.2", cuda_major=13
            ),
        ),
    )

    def refused() -> NoReturn:
        raise RuntimeError(failure)

    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")
    setattr(runtime, "runtime_identity", refused)
    setattr(
        runtime,
        "mapped_libraries",
        lambda: (
            SimpleNamespace(
                name="cudart64_12.dll", path="C:/foreign/cudart64_12.dll", cuda_major=None
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(environment, "collect_cuda_environment", lambda **kwargs: facts)
    report = environment.cuda_environment()
    assert not report.successful
    assert report.facts.runtime_identity == ()
    assert report.issues[0].code == "native-inspection-unavailable"
    assert failure in report.issues[0].message
    assert {issue.code for issue in report.issues} == {
        "native-inspection-unavailable",
        "unsupported-cuda-major",
        "unowned-cuda-library",
    }


@pytest.mark.parametrize("exception", [RuntimeError, ImportError])
def test_windows_dependency_inspection_refusal_retains_cause(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, exception: type[Exception]
) -> None:
    from tests._support.pe import write_pe

    path = write_pe(tmp_path / "provider.dll")
    original = exception("Unowned resident CUDA DLL family member foreign/cudnn.dll")
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")

    def refused(path: str) -> NoReturn:
        raise original

    setattr(runtime, "dependency_closure", refused)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    with pytest.raises(ValueError, match="Unowned resident CUDA") as caught:
        resolve_native_dependencies(path, library_path=(), executable=None)
    assert caught.value.__cause__ is original


def test_windows_incomplete_inspection_keeps_all_issues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests._support.pe import write_pe

    path = write_pe(tmp_path / "provider.dll")
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")
    setattr(
        runtime,
        "dependency_closure",
        lambda path: SimpleNamespace(
            root=str(path),
            machine=0x8664,
            complete=False,
            issues=("unowned dependency missing.dll", "module inventory truncated"),
            edges=(
                SimpleNamespace(
                    name="missing.dll",
                    requester=str(path),
                    path=None,
                    kind="unresolved",
                ),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    with pytest.raises(ValueError, match="module inventory truncated"):
        resolve_native_dependencies(path, library_path=(), executable=None)


def test_windows_mapping_failure_retains_prior_inspection_issue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import phydrax._cuda_environment as environment

    facts = replace(
        _facts(),
        host_platform="win32",
        host_machine="AMD64",
        plugins=_plugins((13,)),
        inspection_error="driver inspection denied",
    )
    runtime = ModuleType("phydrax_cuda13_pjrt.runtime")

    def refused() -> NoReturn:
        raise RuntimeError("module inventory denied")

    setattr(runtime, "mapped_libraries", refused)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.setattr(environment, "collect_cuda_environment", lambda **kwargs: facts)
    report = environment.cuda_environment()
    assert not report.successful
    assert report.facts.inspection_error is not None
    assert "driver inspection denied" in report.facts.inspection_error
    assert "module inventory denied" in report.facts.inspection_error
