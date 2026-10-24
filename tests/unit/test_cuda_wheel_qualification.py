"""Hardware qualification cannot accept missing required native coverage."""

import base64
import importlib.metadata
import json
import os
import sys
import types
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import NoReturn
from unittest.mock import Mock
from xml.etree import ElementTree

import pytest

from tools import qualify_cuda_wheels as qualification


@pytest.fixture
def evidence(tmp_path: Path) -> qualification.Evidence:
    return {
        "timestamp": "",
        "completed_at": "",
        "major": 13,
        "required_gpus": 1,
        "command_timeout_seconds": qualification.DEFAULT_COMMAND_TIMEOUT,
        "libraries": [],
        "library_qualification": None,
        "artifact_directory": str(tmp_path),
        "selected_devices": [0],
        "coverage": {},
        "venv": str(tmp_path),
        "wheels": {},
        "install": [],
        "tests": None,
        "audits": {},
        "extras_gate": None,
        "errors": [],
        "successful": False,
        "mode": "hardware",
        "hardware_qualified": False,
        "work_root": str(tmp_path),
        "harness": {},
        "identity": {},
        "installation_checks": [],
    }


def _provider_report(path: Path, devices: list[int]) -> ElementTree.Element:
    suite = ElementTree.Element("testsuite")
    for name in (
        *qualification.DEVICE_COVERAGE_TESTS,
        "test_quda_refuses_multiple_processes",
        "test_quda_refuses_second_device",
    ):
        case = ElementTree.SubElement(suite, "testcase", name=name)
        if name in qualification.DEVICE_COVERAGE_TESTS:
            properties = ElementTree.SubElement(case, "properties")
            ElementTree.SubElement(
                properties,
                "property",
                name="cuda_native_devices",
                value=json.dumps(devices),
            )
    ElementTree.ElementTree(suite).write(path, encoding="utf-8")
    return suite


@pytest.mark.parametrize(
    ("required_gpus", "skip_reason", "refused"),
    [
        (1, "Same-plan cuDSS qualification requires two CUDA devices", False),
        (2, "Same-plan cuDSS qualification requires two CUDA devices", True),
        (2, "Non-default GPU AmgX qualification requires at least two GPUs", True),
        (2, "QUDA second-device refusal requires two CUDA devices", True),
        (4, "Same-plan cuDSS qualification requires two CUDA devices", True),
        (1, "Install phydrax[cuda13] to load the companion", True),
    ],
)
def test_required_native_coverage_cannot_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: qualification.Evidence,
    required_gpus: int,
    skip_reason: str,
    refused: bool,
) -> None:
    evidence["required_gpus"] = required_gpus
    evidence["selected_devices"] = list(range(required_gpus))

    def completed_tests(
        command: list[str],
        *,
        environment: dict[str, str] | None = None,
        cwd: Path = qualification.ROOT,
        timeout: float = qualification.DEFAULT_COMMAND_TIMEOUT,
    ) -> qualification.CommandEvidence:
        assert command[1:3] == ["-I", "-c"]
        assert cwd == tmp_path
        assert command[command.index("--rootdir") + 1] == str(tmp_path)
        assert str(qualification.ROOT / "pyproject.toml") not in command
        assert environment is not None and "PYTHONPATH" not in environment
        _provider_report(
            tmp_path / "native-provider-results.xml", list(range(required_gpus))
        )
        return {
            "command": command,
            "cwd": str(cwd),
            "timed_out": False,
            "returncode": 0,
            "stdout": f"SKIPPED [1] tests/provider.py:1: {skip_reason}\n",
            "stderr": "",
            "stdout_path": str(tmp_path / "command.stdout"),
            "stderr_path": str(tmp_path / "command.stderr"),
            "stdout_sha256": "0" * 64,
            "stderr_sha256": "0" * 64,
        }

    monkeypatch.setattr(qualification, "run", completed_tests)
    qualification.run_tests(tmp_path / "python", tmp_path, evidence)
    assert bool(evidence["errors"]) is refused


@pytest.mark.parametrize("required_gpus", [1, 2, 4, 8])
def test_qualification_requires_exact_per_case_device_coverage(
    tmp_path: Path, evidence: qualification.Evidence, required_gpus: int
) -> None:
    evidence["required_gpus"] = required_gpus
    evidence["selected_devices"] = [2 * index for index in range(required_gpus)]
    report = tmp_path / "native-provider-results.xml"
    _provider_report(report, evidence["selected_devices"])
    qualification.collect_test_coverage(report, evidence)
    assert not evidence["errors"]
    assert evidence["coverage"] == {
        name: evidence["selected_devices"] for name in qualification.DEVICE_COVERAGE_TESTS
    }


@pytest.mark.parametrize("name", qualification.DEVICE_COVERAGE_TESTS)
@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "skipped",
        "failure",
        "error",
        "no-property",
        "duplicate-property",
        "malformed",
        "only-two-devices",
        "duplicate-devices",
        "wrong-order",
        "bool-device",
    ],
)
def test_four_gpu_qualification_rejects_incomplete_hardware_cases(
    tmp_path: Path, evidence: qualification.Evidence, name: str, defect: str
) -> None:
    evidence["required_gpus"] = 4
    evidence["selected_devices"] = [0, 1, 2, 3]
    report = tmp_path / "native-provider-results.xml"
    suite = _provider_report(report, evidence["selected_devices"])
    case = next(case for case in suite if case.get("name") == name)
    properties = case.find("properties")
    assert properties is not None
    if defect == "missing":
        suite.remove(case)
    elif defect in ("skipped", "failure", "error"):
        ElementTree.SubElement(case, defect)
    elif defect == "no-property":
        case.remove(properties)
    elif defect == "duplicate-property":
        ElementTree.SubElement(
            properties, "property", name="cuda_native_devices", value="[0, 1, 2, 3]"
        )
    else:
        values = {
            "malformed": "{",
            "only-two-devices": "[0, 1]",
            "duplicate-devices": "[0, 1, 1, 3]",
            "wrong-order": "[3, 2, 1, 0]",
            "bool-device": "[false, 1, 2, 3]",
        }
        properties[0].set("value", values[defect])
    ElementTree.ElementTree(suite).write(report, encoding="utf-8")
    qualification.collect_test_coverage(report, evidence)
    assert evidence["errors"]
    assert name not in evidence["coverage"]


@pytest.mark.parametrize("required_gpus", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "name", ["test_quda_refuses_second_device", "test_quda_refuses_multiple_processes"]
)
def test_qualification_preserves_quda_refusal_boundaries(
    tmp_path: Path, evidence: qualification.Evidence, required_gpus: int, name: str
) -> None:
    evidence["required_gpus"] = required_gpus
    evidence["selected_devices"] = list(range(required_gpus))
    report = tmp_path / "native-provider-results.xml"
    suite = _provider_report(report, evidence["selected_devices"])
    case = next(case for case in suite if case.get("name") == name)
    ElementTree.SubElement(case, "skipped")
    ElementTree.ElementTree(suite).write(report, encoding="utf-8")
    qualification.collect_test_coverage(report, evidence)
    assert bool(evidence["errors"]) == (
        required_gpus >= 2 or name == "test_quda_refuses_multiple_processes"
    )


@pytest.mark.parametrize("content", [None, "<testsuite>", "<testsuite />"])
def test_qualification_rejects_missing_case_report(
    tmp_path: Path, evidence: qualification.Evidence, content: str | None
) -> None:
    report = tmp_path / "native-provider-results.xml"
    if content is not None:
        report.write_text(content)
    qualification.collect_test_coverage(report, evidence)
    assert evidence["errors"]


@pytest.mark.parametrize("hostile", [False, True], ids=["clean", "hostile"])
def test_audit_rejects_provider_refusal(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    hostile: bool,
) -> None:
    from phydrax import _cuda_native
    from phydrax._cuda_environment import CudaEnvironmentIssue

    paths = (qualification.DEFAULT_HOSTILE_LIBRARY_PATH,) if hostile else ()
    issue = CudaEnvironmentIssue(
        code="backend-unavailable",
        severity="error",
        message="Install phydrax[cuda13] in a working CUDA environment",
        paths=paths,
    )

    def unavailable(component: str) -> NoReturn:
        raise _cuda_native.CudaNativeUnavailableError(issue)

    monkeypatch.setattr(_cuda_native, "load_cuda_component", unavailable)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "1",
            qualification.DEFAULT_HOSTILE_LIBRARY_PATH if hostile else "",
            "hostile" if hostile else "clean",
        ],
    )
    if hostile:
        monkeypatch.setenv("LD_LIBRARY_PATH", qualification.DEFAULT_HOSTILE_LIBRARY_PATH)
    else:
        monkeypatch.delenv("LD_LIBRARY_PATH", raising=False)
    with pytest.raises(SystemExit) as error:
        exec(qualification.AUDIT, {})
    assert error.value.code == 1
    report = json.loads(capsys.readouterr().out.strip().removeprefix("AUDIT "))
    assert report == {
        "refused": {
            "code": "backend-unavailable",
            "paths": list(paths),
            "message": issue.message,
        }
    }


def test_selected_wheel_identity_overrides_same_named_cached_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = "phydrax-0.2.2-py3-none-any.whl"
    selected = tmp_path / name
    selected.write_bytes(b"selected artifact")
    wheelhouse = tmp_path / "cached"
    wheelhouse.mkdir()
    (wheelhouse / name).write_bytes(b"older cached artifact")
    existing_venv = tmp_path / "existing-venv"
    existing_venv.mkdir()
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    output = tmp_path / "evidence.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualify_cuda_wheels",
            "--python",
            sys.executable,
            "--work-root",
            str(tmp_path),
            "--hostile-library-path",
            str(hostile),
            "--venv",
            str(existing_venv),
            "--phydrax-wheel",
            str(selected),
            "--wheelhouse",
            str(wheelhouse),
            "--evidence",
            str(output),
        ],
    )
    assert qualification.main() == 1
    evidence = json.loads(output.read_text())
    assert evidence["wheels"][name] == sha256(b"selected artifact").hexdigest()


@pytest.mark.parametrize(
    "platform,relative", [("win32", "Scripts/python.exe"), ("linux", "bin/python")]
)
def test_environment_interpreter_is_platform_specific(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, relative: str
) -> None:
    monkeypatch.setattr(sys, "platform", platform)
    assert qualification.interpreter_path(tmp_path) == tmp_path / relative


def test_harness_is_copied_and_content_bound(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "copied"
    for package in ("tests", "tools", "benchmarks", "phydrax"):
        (source / package).mkdir(parents=True)
        (source / package / "__init__.py").write_bytes(f"# {package}\n".encode())
    (source / "pyproject.toml").write_bytes(b"[tool.pytest.ini_options]\n")
    monkeypatch.setattr(qualification, "ROOT", source)
    bound = qualification.copy_harness(destination)
    assert not (destination / "phydrax").exists()
    assert bound["tests/__init__.py"] == sha256(b"# tests\n").hexdigest()
    (source / "tests/__init__.py").write_text("# changed\n")
    assert (destination / "tests/__init__.py").read_text() == "# tests\n"
    assert not any(path.is_symlink() for path in destination.rglob("*"))


def test_isolated_installed_command_rejects_checkout_package(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    package = types.ModuleType("phydrax")
    package.__file__ = str(tmp_path / "checkout/phydrax/__init__.py")
    monkeypatch.setitem(sys.modules, "phydrax", package)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "venv"))
    command = qualification.installed_command(
        tmp_path / "venv/Scripts/python.exe", tmp_path / "harness", "tools.example", []
    )
    assert command[1:3] == ["-I", "-c"]
    monkeypatch.setattr(sys, "argv", ["-c", *command[4:]])
    with pytest.raises(RuntimeError, match="installed Phydrax bytes"):
        exec(command[3], {})


def test_windows_environment_uses_hostile_dll_search_not_elf_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/developer/toolkit")
    monkeypatch.setenv("CUDA_PATH", "/developer/cuda")
    monkeypatch.setenv("PYTHONPATH", "/checkout")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    result = qualification.clean_environment(
        library_path=str(tmp_path), work_root=tmp_path
    )
    assert not {"LD_LIBRARY_PATH", "CUDA_PATH", "PYTHONPATH"} & result.keys()
    assert result["PHYDRAX_CUDA_PROFILE_HOSTILE_PATH"] == str(tmp_path)
    assert result["PATH"].startswith(str(tmp_path) + os.pathsep)
    assert result["XLA_PYTHON_CLIENT_PREALLOCATE"] == "true"
    for key in (
        "TEMP",
        "TMP",
        "UV_CACHE_DIR",
        "QUDA_RESOURCE_PATH",
        "JAX_COMPILATION_CACHE_DIR",
        "CUDA_CACHE_PATH",
    ):
        assert Path(result[key]).is_relative_to(tmp_path)


def test_work_path_refuses_escape_and_reparse_before_descending(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="escapes"):
        qualification.validate_work_path(tmp_path / "../escape", tmp_path)
    original = Path.lstat
    link = tmp_path / "junction"
    visited = []

    def lstat(path: Path, *args: object, **kwargs: object) -> object:
        visited.append(path)
        if path == link:
            return types.SimpleNamespace(st_mode=0, st_file_attributes=0x400)
        return original(path)

    monkeypatch.setattr(Path, "lstat", lstat)
    with pytest.raises(ValueError, match="reparse"):
        qualification.validate_work_path(link / "outside/file", tmp_path)
    assert link / "outside" not in visited


def test_identity_probe_compares_installed_asset_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    asset = tmp_path / "owned.dll"
    asset.write_bytes(b"selected")
    selected = tmp_path / "selected.json"
    observed = tmp_path / "observed.json"
    selected.write_text(
        json.dumps(
            {
                "phydrax-cuda13-pjrt": {
                    "version": "0.2.2",
                    "files": {"owned.dll": sha256(b"selected").hexdigest()},
                }
            }
        )
    )
    distribution = types.SimpleNamespace(
        version="0.2.2", locate_file=lambda member: tmp_path / member
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda name: distribution)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(sys, "prefix", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["-c", str(selected), str(observed)])
    exec(qualification.IDENTITY, {})
    assert (
        capsys.readouterr().out.strip()
        == "IDENTITY " + sha256(observed.read_bytes()).hexdigest()
    )
    asset.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="differs from selected wheel"):
        exec(qualification.IDENTITY, {})


def test_selected_windows_identity_requires_both_binary_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    core = tmp_path / "phydrax-0.2.2-py3-none-any.whl"
    for name, wheel in (
        ("phydrax", core),
        ("phydrax-cuda13", tmp_path / "phydrax_cuda13-0.2.2-cp313-cp313-win_amd64.whl"),
        (
            "phydrax-cuda13-pjrt",
            tmp_path / "phydrax_cuda13_pjrt-0.2.2-cp313-cp313-win_amd64.whl",
        ),
    ):
        with zipfile.ZipFile(wheel, "w") as archive:
            archive.writestr(
                f"{name.replace('-', '_')}-0.2.2.dist-info/METADATA",
                f"Name: {name}\nVersion: 0.2.2\n",
            )
            archive.writestr(f"{name.replace('-', '_')}/asset.dll", name.encode())
    identity = qualification.selected_identity(core, tmp_path)
    assert set(identity) == {"phydrax", "phydrax-cuda13", "phydrax-cuda13-pjrt"}
    (tmp_path / "phydrax_cuda13_pjrt-0.2.2-cp313-cp313-win_amd64.whl").unlink()
    with pytest.raises(ValueError, match="phydrax_cuda13_pjrt"):
        qualification.selected_identity(core, tmp_path)


def test_installation_refusal_mode_never_runs_hardware(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    wheel = tmp_path / "core.whl"
    wheel.write_bytes(b"fixture")
    output = tmp_path / "result.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualifier",
            "--python",
            sys.executable,
            "--work-root",
            str(tmp_path),
            "--venv",
            str(tmp_path / "venv"),
            "--reuse-venv",
            "--phydrax-wheel",
            str(wheel),
            "--wheelhouse",
            str(tmp_path),
            "--evidence",
            str(output),
            "--mode",
            "installation-refusal",
        ],
    )
    monkeypatch.setattr(qualification, "install", lambda *args: True)
    identity = Mock()
    refusal = Mock()
    hardware = Mock(side_effect=AssertionError("No hardware in installation mode"))
    monkeypatch.setattr(qualification, "check_identity", identity)
    monkeypatch.setattr(qualification, "installation_refusal_checks", refusal)
    monkeypatch.setattr(qualification, "audit", hardware)
    monkeypatch.setattr(qualification, "run_tests", hardware)
    assert qualification.main() == 0
    identity.assert_called_once()
    refusal.assert_called_once()
    report = json.loads(output.read_text())
    assert report["successful"] and not report["hardware_qualified"]
    assert report["selected_devices"] == []


@pytest.mark.parametrize("mode", ["hardware", "installation-refusal"])
def test_zero_gpus_is_never_a_qualification_shortcut(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "qualifier",
            "--python",
            sys.executable,
            "--work-root",
            str(tmp_path),
            "--venv",
            str(tmp_path / "venv"),
            "--phydrax-wheel",
            str(tmp_path / "core.whl"),
            "--wheelhouse",
            str(tmp_path),
            "--evidence",
            str(tmp_path / "result.json"),
            "--mode",
            mode,
            "--required-gpus",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as error:
        qualification.main()
    assert error.value.code == 2


@pytest.mark.parametrize("outcome", ["passed", "timeout", "spawn-error"])
def test_installed_faults_restore_real_bytes_after_every_child_outcome(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: qualification.Evidence,
    outcome: str,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(b'{"valid": true}')
    library = tmp_path / "pjrt.dll"
    library.write_bytes(b"real fixture payload")
    evidence["identity"] = {
        "runtime": {
            "manifest_path": str(manifest),
            "assets": [[str(library), "digest", library.stat().st_size, "pjrt"]],
        },
    }
    originals = {manifest: manifest.read_bytes(), library: library.read_bytes()}
    for name in ("phydrax-cuda13", "phydrax-cuda13-pjrt"):
        member = f"{name}.dist-info/METADATA"
        path = tmp_path / member
        path.parent.mkdir()
        path.write_bytes(b"Name: fixture\nVersion: 0.2.2\n")
        originals[path] = path.read_bytes()
        evidence["identity"][name] = {
            "installed_root": str(tmp_path),
            "files": {member: "digest"},
            "version": "0.2.2",
        }
    calls = []

    def runner(command: list[str], **kwargs: object) -> dict[str, object]:
        calls.append(command)
        assert command[1:3] == ["-I", "-c"]
        assert any(
            not path.exists() or path.read_bytes() != data
            for path, data in originals.items()
        )
        if outcome == "spawn-error":
            raise OSError("fixture process creation failure")
        return {
            "returncode": 0 if outcome == "passed" else 1,
            "timed_out": outcome == "timeout",
            "stdout": "INSTALLATION_FAULT_REFUSED fixture" if outcome == "passed" else "",
        }

    monkeypatch.setattr(qualification, "run", runner)
    if outcome == "spawn-error":
        with pytest.raises(OSError, match="creation"):
            qualification.installed_fault_checks(tmp_path / "python", evidence)
    else:
        qualification.installed_fault_checks(tmp_path / "python", evidence)
        assert len(calls) == 8
        assert bool(evidence["errors"]) == (outcome == "timeout")
    assert all(path.read_bytes() == data for path, data in originals.items())
    assert not tuple(tmp_path.rglob("*.qualification-backup"))


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "static-runtime",
        "missing-members",
        "missing-symbol",
        "missing-hash",
        "missing-wheel",
        "missing-link-map",
    ],
)
def test_windows_loader_provenance_requires_canonical_vendor_archives(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str | None,
) -> None:
    # Manifest fixtures exercise schema admission, never GPU/provider execution.
    manifest = {
        "runtime_linkage": {
            "kind": "driver-hybrid",
            "target_dll": "nvcudart_hybrid64.dll",
            "owner": "display-driver",
        },
        "toolchain": {
            "cuda_link_library": {
                "path": "C:/fixture/cuda.lib",
                "sha256": "a" * 64,
                "kind": "driver-loader",
                "target_dll": "nvcuda.dll",
                "archive_members": ["cuda_driver_loader.obj"],
                "dispatch_symbols": [
                    "cudaDriverLoader_initializeLoader",
                    "cuGetProcAddress_v2",
                ],
            },
            "cudart_link_library": {
                "path": "C:/fixture/cudart.lib",
                "sha256": "b" * 64,
                "kind": "driver-hybrid",
                "target_dll": "nvcudart_hybrid64.dll",
                "archive_members": ["hybrid_runtime_loader.obj"],
                "dispatch_symbols": [
                    "initializeHybridRuntimeLoaderOnce",
                    "cuGetProcAddress_v2",
                    "cuGetExportTable",
                ],
            },
        },
        "toolkit_wheels": {"nvidia-cuda-runtime": "13.4.92"},
        "link_maps": {"cudss": "lib/phydrax_cuda_cudss.map"},
    }
    if fault == "static-runtime":
        manifest["runtime_linkage"]["kind"] = "static"
    elif fault == "missing-members":
        manifest["toolchain"]["cudart_link_library"]["archive_members"] = []
    elif fault == "missing-symbol":
        manifest["toolchain"]["cuda_link_library"]["dispatch_symbols"] = []
    elif fault == "missing-hash":
        manifest["toolchain"]["cudart_link_library"].pop("sha256")
    elif fault == "missing-wheel":
        manifest.pop("toolkit_wheels")
    elif fault == "missing-link-map":
        manifest.pop("link_maps")
    path = tmp_path / "phx_cuda_build.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    distribution = types.SimpleNamespace(
        files=[Path("phx_cuda_build.json")], locate_file=lambda member: tmp_path / member
    )
    monkeypatch.setattr(importlib.metadata, "distribution", lambda _: distribution)
    namespace = {}
    exec(qualification.WINDOWS_LOADER_PROVENANCE, namespace)
    if fault is not None:
        with pytest.raises(RuntimeError):
            namespace["windows_loader_provenance"]()
    else:
        observed = namespace["windows_loader_provenance"]()
        assert observed["manifest_sha256"] == sha256(path.read_bytes()).hexdigest()
        assert observed["runtime_linkage"] == manifest["runtime_linkage"]
        assert observed["loader_archives"] == manifest["toolchain"]
        assert "runtime_identity" not in observed


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing-runtime",
        "changed-loader",
        "changed-driver",
        "missing-dispatch",
        "changed-dispatch",
    ],
)
def test_windows_hostile_audit_retains_driver_cohort_and_loader_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    evidence: qualification.Evidence,
    fault: str | None,
) -> None:
    identity = {
        "linkage": "driver-hybrid",
        "owner": "display-driver",
        **{
            f"{name}_{field}": value
            for name in ("runtime", "driver", "driver_dispatch")
            for field, value in (("path", f"C:/fixture/{name}.dll"), ("sha256", "a" * 64))
        },
    }
    provenance = {
        "manifest_sha256": "b" * 64,
        "runtime_linkage": {
            "kind": "driver-hybrid",
            "target_dll": "nvcudart_hybrid64.dll",
            "owner": "display-driver",
        },
        "loader_archives": {"cuda_link_library": {}, "cudart_link_library": {}},
        "toolkit_wheels": {"nvidia-cuda-runtime": "13.4.92"},
    }
    dispatch = {
        "cudss": {
            "path": "C:/fixture/phydrax_cuda_cudss.dll",
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
            "link_map_path": "C:/fixture/phydrax_cuda_cudss.map",
            "link_map_sha256": "f" * 64,
            "runtime_linkage": provenance["runtime_linkage"],
            "static_runtime_excluded": True,
            "build_hash": "fixture",
        }
    }
    evidence["identity"]["loader_provenance"] = provenance
    evidence["audits"]["clean"] = {
        "report": {
            "runtime_identity": identity,
            "component_dispatch": dispatch,
        }
    }
    report = {
        "compiled_cuda_major": 13,
        "driver_compatibility_version": [14, 0],
        "runtime_identity": dict(identity),
        "loader_provenance": dict(provenance),
        "component_dispatch": json.loads(json.dumps(dispatch)),
    }
    if fault == "missing-runtime":
        report.pop("runtime_identity")
    elif fault == "changed-loader":
        report["loader_provenance"]["manifest_sha256"] = "c" * 64
    elif fault == "changed-driver":
        report["runtime_identity"]["driver_sha256"] = "d" * 64
    elif fault == "missing-dispatch":
        report.pop("component_dispatch")
    elif fault == "changed-dispatch":
        report["component_dispatch"]["cudss"]["sha256"] = "c" * 64
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(qualification, "clean_environment", lambda **kwargs: {})
    monkeypatch.setattr(
        qualification,
        "run",
        Mock(
            return_value={
                "returncode": 0,
                "timed_out": False,
                "stderr": "",
                "stdout": "AUDIT " + json.dumps(report),
            }
        ),
    )
    qualification.audit(tmp_path / "python.exe", "hostile", str(tmp_path), evidence)
    assert bool(evidence["errors"]) == (fault is not None)
    # Schema admission is not a hardware-qualification result.
    assert not evidence["hardware_qualified"]


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "second-static",
        "second-standalone",
        "changed-map",
        "changed-binary",
        "missing-record",
        "map-escape",
        "wrong-component-map",
        "no-components",
    ],
)
def test_installed_dispatch_inspects_every_component_and_authenticates_binary_and_map(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fault: str | None,
) -> None:
    from tests._support.pe import pe_image

    package = tmp_path / "phydrax_cuda13"
    libraries = package / "lib"
    libraries.mkdir(parents=True)
    linkage = {
        "kind": "driver-hybrid",
        "target_dll": "nvcudart_hybrid64.dll",
        "owner": "display-driver",
    }
    link_maps = {}
    components = []
    paths = []
    for name in ("cudss", "amgx"):
        path = libraries / f"phydrax_cuda_{name}.dll"
        imports = (
            ("cudart64_13.dll",)
            if fault == "second-standalone" and name == "amgx"
            else ()
        )
        data = bytearray(pe_image(imports=imports))
        markers = b"nvcudart_hybrid64.dll\0cuGetProcAddress_v2\0cuGetExportTable\0"
        data[0x1E00 : 0x1E00 + len(markers)] = markers
        path.write_bytes(data)
        mapping = "0001:00001000 initializeLoader 0000000180001000 cudart:cudart_static_loader.obj\n"
        if fault == "second-static" and name == "amgx":
            mapping += "0001:00002000 cudaMalloc 0000000180002000 cudart_static:cuda_runtime.obj\n"
        map_path = path.with_suffix(".map")
        map_path.write_text(mapping, encoding="utf-8")
        paths.extend((path, map_path))
        link_maps[name] = map_path.relative_to(package).as_posix()
        components.append(
            types.SimpleNamespace(name=name, path=str(path), build_hash="fixture")
        )
    if fault == "map-escape":
        link_maps["amgx"] = "../foreign.map"
    elif fault == "wrong-component-map":
        link_maps["amgx"] = link_maps["cudss"]
    manifest = package / "phx_cuda_build.json"
    manifest.write_text(
        json.dumps({"runtime_linkage": linkage, "link_maps": link_maps}), encoding="utf-8"
    )
    paths.append(manifest)
    records = []
    for path in paths:
        record = importlib.metadata.PackagePath(path.relative_to(tmp_path).as_posix())
        payload = path.read_bytes()
        record.hash = importlib.metadata.FileHash(
            "sha256="
            + base64.urlsafe_b64encode(bytes.fromhex(sha256(payload).hexdigest()))
            .rstrip(b"=")
            .decode()
        )
        record.size = len(payload)
        if fault != "missing-record" or path.name != "phydrax_cuda_amgx.map":
            records.append(record)
    if fault == "changed-map":
        with (libraries / "phydrax_cuda_amgx.map").open("a", encoding="utf-8") as stream:
            stream.write("\n")
    elif fault == "changed-binary":
        with (libraries / "phydrax_cuda_amgx.dll").open("ab") as stream:
            stream.write(b"changed")
    distribution = types.SimpleNamespace(
        files=records,
        locate_file=lambda member: tmp_path / member,
    )
    installed_distribution = importlib.metadata.distribution
    # Only replace companion ownership; importing the canonical PE parser also
    # initializes Phydrax dependencies, which need their real package metadata.
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            distribution if name == "phydrax-cuda13" else installed_distribution(name)
        ),
    )
    namespace = {}
    exec(qualification.WINDOWS_LOADER_PROVENANCE, namespace)
    provenance = {
        "manifest_path": str(manifest),
        "manifest_sha256": sha256(manifest.read_bytes()).hexdigest(),
        "runtime_linkage": linkage,
        "link_maps": link_maps,
    }
    if fault is not None:
        with pytest.raises((RuntimeError, ValueError)):
            namespace["windows_component_dispatch"](
                provenance, () if fault == "no-components" else components
            )
    else:
        proof = namespace["windows_component_dispatch"](provenance, components)
        assert set(proof) == {"cudss", "amgx"}
        for component in components:
            assert proof[component.name]["path"] == component.path
            assert (
                proof[component.name]["sha256"]
                == sha256(Path(component.path).read_bytes()).hexdigest()
            )
            assert proof[component.name]["static_runtime_excluded"] is True
        # Fixture PE bytes are never loaded and do not qualify native execution.
