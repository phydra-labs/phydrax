"""Optional CUDA qualification refuses missing mathematics and failed evidence."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree

import pytest

from tools import _cuda_optional_qualification as qualification
from tools._cuda_optional_qualification import collect_profile_coverage, LibraryProfile


@pytest.fixture
def profile() -> LibraryProfile:
    return LibraryProfile(
        ("qualification.py",), ("forward", "reverse"), ("float32", "float64")
    )


@pytest.fixture
def windows_runtime_evidence() -> dict[str, object]:
    """Synthetic schema evidence only; no native execution is represented."""
    return {
        "compiled_cuda_major": 13,
        "driver_compatibility_version": [14, 0],
        "runtime_identity": {
            "linkage": "driver-hybrid",
            "owner": "display-driver",
            "runtime_path": "C:/Windows/System32/DriverStore/FileRepository/nvidia/nvcudart_hybrid64.dll",
            "runtime_sha256": "a" * 64,
            "driver_path": "C:/Windows/System32/DriverStore/FileRepository/nvidia/nvcuda64.dll",
            "driver_sha256": "b" * 64,
            "driver_dispatch_path": "C:/Windows/System32/nvcuda.dll",
            "driver_dispatch_sha256": "c" * 64,
        },
        "loader_provenance": {
            "manifest_sha256": "d" * 64,
            "runtime_linkage": {
                "kind": "driver-hybrid",
                "target_dll": "nvcudart_hybrid64.dll",
                "owner": "display-driver",
            },
            "loader_archives": {"cuda_link_library": {}, "cudart_link_library": {}},
            "toolkit_wheels": {"nvidia-cuda-runtime": "13.4.92"},
        },
        "component_dispatch": {
            "cutensor": {
                "path": "C:/fixture/phydrax_cuda_cutensor.dll",
                "sha256": "e" * 64,
                "machine": 0x8664,
                "imports": ["kernel32.dll"],
                "delay_imports": [],
                "closure_imports": ["kernel32.dll"],
                "dispatch_markers": [
                    "nvcudart_hybrid64.dll",
                    "cuGetProcAddress_v2",
                    "cuGetExportTable",
                ],
                "link_map_path": "C:/fixture/phydrax_cuda_cutensor.map",
                "link_map_sha256": "f" * 64,
                "runtime_linkage": {
                    "kind": "driver-hybrid",
                    "target_dll": "nvcudart_hybrid64.dll",
                    "owner": "display-driver",
                },
                "static_runtime_excluded": True,
                "build_hash": "fixture",
            }
        },
    }


@pytest.mark.parametrize(
    "missing",
    [
        None,
        "runtime_identity",
        "loader_provenance",
        "driver_compatibility_version",
        "compiled_cuda_major",
        "component_dispatch",
    ],
)
def test_windows_runtime_evidence_requires_observation_and_loader_provenance(
    windows_runtime_evidence: dict[str, object],
    missing: str | None,
) -> None:
    if missing is None:
        qualification.validate_windows_runtime_evidence(windows_runtime_evidence)
    else:
        windows_runtime_evidence.pop(missing)
        with pytest.raises(ValueError):
            qualification.validate_windows_runtime_evidence(windows_runtime_evidence)


@pytest.mark.parametrize("compatibility", [[12, 9], None, "13010"])
def test_windows_runtime_evidence_rejects_unsupported_or_unknown_driver(
    windows_runtime_evidence: dict[str, object],
    compatibility: object,
) -> None:
    windows_runtime_evidence["driver_compatibility_version"] = compatibility
    with pytest.raises(ValueError, match="compatible driver"):
        qualification.validate_windows_runtime_evidence(windows_runtime_evidence)


def _report(path: Path, names: tuple[str, ...], *, terminal: str | None = None) -> None:
    suite = ElementTree.Element("testsuite")
    for name in names:
        case = ElementTree.SubElement(
            suite, "testcase", name=name, classname="numerical_contract"
        )
        if terminal is not None and name == "reverse[float64]":
            ElementTree.SubElement(case, terminal)
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")


def test_complete_mathematical_precision_matrix_qualifies(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        ("forward[float32]", "reverse[float32]", "forward[float64]", "reverse[float64]"),
    )
    result = collect_profile_coverage(report, profile)
    assert result.passed
    assert result.missing == result.unsuccessful == result.duplicates == ()
    assert result.report_error is None


def test_missing_precision_refuses_an_otherwise_passing_report(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(report, ("forward[float32]", "reverse[float32]", "forward[float64]"))
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.missing == (("reverse", "float64"),)
    assert result.unsuccessful == ()


def test_missing_scenario_is_not_replaced_by_another_precision_case(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(report, ("forward[float32]", "forward[float64]", "unrelated[float64]"))
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.missing == (
        ("reverse", None),
        ("reverse", "float32"),
        ("reverse", "float64"),
    )


@pytest.mark.parametrize("terminal", ["skipped", "failure", "error"])
def test_required_scenario_must_actually_pass(
    tmp_path: Path, profile: LibraryProfile, terminal: str
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        ("forward[float32]", "reverse[float32]", "forward[float64]", "reverse[float64]"),
        terminal=terminal,
    )
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.unsuccessful == ("numerical_contract::reverse[float64]",)
    assert result.missing == ()


def test_duplicate_result_cannot_inflate_completed_coverage(
    tmp_path: Path, profile: LibraryProfile
) -> None:
    report = tmp_path / "cases.xml"
    _report(
        report,
        (
            "forward[float32]",
            "reverse[float32]",
            "forward[float64]",
            "reverse[float64]",
            "reverse[float64]",
        ),
    )
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.duplicates == ("numerical_contract::reverse[float64]",)


@pytest.mark.parametrize("contents", [None, "not XML", "<testsuite/>"])
def test_absent_or_empty_report_never_qualifies(
    tmp_path: Path, profile: LibraryProfile, contents: str | None
) -> None:
    report = tmp_path / "cases.xml"
    if contents is not None:
        report.write_text(contents, encoding="utf-8")
    result = collect_profile_coverage(report, profile)
    assert not result.passed
    assert result.cases == ()
    assert (result.report_error is not None) == (contents != "<testsuite/>")


@pytest.mark.parametrize("provider", ["cutensornet", "cuest"])
def test_windows_unavailable_provider_is_explicit_refusal_before_discovery(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    runner = Mock(side_effect=AssertionError("Unavailable provider must not run"))
    monkeypatch.setattr(qualification, "run", runner)
    evidence, errors = qualification.qualify_libraries(
        tmp_path / "Scripts/python.exe", tmp_path, (provider,), 1, {}, 60, str(tmp_path)
    )
    assert errors and evidence["refused"] == [provider]
    assert not evidence["passed"] and not evidence["hardware_qualified"]
    runner.assert_not_called()


def test_optional_hardware_requires_positive_gpu_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive"):
        qualification.qualify_libraries(
            tmp_path / "python", tmp_path, ("cutensor",), 0, {}, 60, str(tmp_path)
        )


def test_windows_hostile_bootstrap_retains_directory_handles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("PHYDRAX_CUDA_PROFILE_HOSTILE_PATH", str(tmp_path))
    handle = object()
    add = Mock(return_value=handle)
    monkeypatch.setattr(os, "add_dll_directory", add, raising=False)
    namespace = {}
    exec(qualification.HOSTILE_BOOTSTRAP, namespace)
    add.assert_called_once_with(str(tmp_path))
    assert namespace["_hostile_handles"] == [handle]


def test_cutensor_full_math_profile_runs_on_every_selected_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    windows_runtime_evidence: dict[str, object],
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    uuids = (
        "GPU-00000000-0000-0000-0000-000000000001",
        "GPU-00000000-0000-0000-0000-000000000002",
    )
    command = {
        "returncode": 0,
        "timed_out": False,
        "stdout": "CUDA_PROFILE_DEVICES "
        + json.dumps([{"platform": "gpu", "gpu_uuid": uuid} for uuid in uuids]),
    }
    discovery = Mock(return_value=command)
    run_profile = Mock(
        return_value=({"passed": True, "libraries": windows_runtime_evidence}, [])
    )
    monkeypatch.setattr(qualification, "run", discovery)
    monkeypatch.setattr(qualification, "_run_profile", run_profile)
    evidence, errors = qualification.qualify_libraries(
        tmp_path / "Scripts/python.exe", tmp_path, ("cutensor",), 2, {}, 60, str(tmp_path)
    )
    assert not errors and evidence["passed"]
    assert discovery.call_args.args[0][1] == "-I"
    local = [
        call
        for call in run_profile.call_args_list
        if call.args[7] in ("device-0", "device-1")
    ]
    assert len(local) == 2
    for index, call in enumerate(local):
        assert call.args[5] is qualification.LIBRARY_PROFILES["cutensor"]
        assert call.args[5].precisions == (
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
        assert call.args[6] == (uuids[index],)
    hostile = [
        call for call in run_profile.call_args_list if call.args[7].endswith("-hostile")
    ]
    assert len(hostile) == 2
    assert all("LD_LIBRARY_PATH" not in call.args[2] for call in hostile)
    assert run_profile.call_args_list[-1].args[7] == "native-host-slices"
