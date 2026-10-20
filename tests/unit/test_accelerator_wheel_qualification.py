# Copyright © 2026 PHYDRA, Inc. All rights reserved.
"""Hardware-free evidence parser/refusal policy tests, never qualification."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tools._accelerator_process import (
    read_json_evidence,
    run,
    TEXT_LIMIT,
    write_json_evidence,
)
from tools.check_accelerator_extras import Accelerator
from tools.qualify_accelerator_wheels import (
    case_report_errors,
    clean_environment,
    host_errors,
    last_json,
    probe_record,
    REQUIRED_CASES,
    runtime_errors,
    wheel_files,
)


def _runtime(accelerator: Accelerator = Accelerator.TPU) -> dict[str, object]:
    return {
        "selected": accelerator.value,
        "platform": accelerator.value,
        "client_platform": accelerator.value,
        "platform_version": "upstream-runtime-test-parser-input",
        "device_kind": "parser-input-not-hardware-evidence",
        "qualified_device_id": 0,
        "execution_device_count": 1,
        "process_index": 0,
        "process_count": 1,
        "global_device_count": 8,
        "local_device_count": 8,
        "x64_enabled": True,
        "matmul_precision": "highest",
    }


def _xml(
    path: Path,
    names: tuple[str, ...] = REQUIRED_CASES,
    *,
    outcome: str | None = None,
    runtime: dict[str, object] | None = None,
    evidence: bool = True,
) -> None:
    suite = ElementTree.Element("testsuite")
    for name in names:
        node = ElementTree.SubElement(
            suite,
            "testcase",
            name=name,
            classname="tests.integration.test_accelerator_scientific_execution",
            time="0.1",
        )
        if outcome is not None:
            ElementTree.SubElement(node, outcome)
        if evidence:
            properties = ElementTree.SubElement(node, "properties")
            ElementTree.SubElement(
                properties,
                "property",
                name="accelerator_runtime",
                value=json.dumps(runtime or _runtime()),
            )
    ElementTree.ElementTree(suite).write(path)


def test_report_requires_exact_successful_case_set_and_runtime(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report)
    errors, cases = case_report_errors(report, _runtime())
    assert errors == [] and len(cases) == len(REQUIRED_CASES)


@pytest.mark.parametrize("outcome", ("skipped", "failure", "error"))
def test_skipped_or_failed_cases_never_qualify(tmp_path: Path, outcome: str) -> None:
    report = tmp_path / "report.xml"
    _xml(report, outcome=outcome)
    assert case_report_errors(report, _runtime())[0]


@pytest.mark.parametrize(
    "names",
    (
        REQUIRED_CASES[:-1],
        (*REQUIRED_CASES, REQUIRED_CASES[0]),
        (*REQUIRED_CASES, "unexpected"),
    ),
)
def test_missing_duplicate_or_extra_case_refuses(
    tmp_path: Path, names: tuple[str, ...]
) -> None:
    report = tmp_path / "report.xml"
    _xml(report, names)
    assert case_report_errors(report, _runtime())[0]


def test_missing_or_wrong_actual_placement_refuses(tmp_path: Path) -> None:
    report = tmp_path / "report.xml"
    _xml(report, evidence=False)
    assert case_report_errors(report, _runtime())[0]
    _xml(report, runtime={**_runtime(), "platform": "cpu"})
    assert case_report_errors(report, _runtime())[0]
    report.write_text("not xml")
    assert case_report_errors(report, _runtime())[0]
    assert case_report_errors(tmp_path / "absent", _runtime())[0]


@pytest.mark.parametrize("accelerator", tuple(Accelerator))
def test_cpu_another_gpu_and_absent_runtime_are_refused(accelerator: Accelerator) -> None:
    for platform in ("cpu", "cuda", "mps"):
        assert runtime_errors(
            {**_runtime(accelerator), "platform": platform, "client_platform": platform},
            accelerator,
        )
    assert runtime_errors({}, accelerator)
    assert not runtime_errors(_runtime(accelerator), accelerator)
    assert runtime_errors({**_runtime(accelerator), "x64_enabled": False}, accelerator)


def test_oneapi_generic_gpu_requires_upstream_oneapi_identity() -> None:
    assert not runtime_errors(
        {**_runtime(Accelerator.ONEAPI), "platform": "gpu"}, Accelerator.ONEAPI
    )
    assert runtime_errors(
        {**_runtime(Accelerator.ONEAPI), "platform": "gpu", "client_platform": "cuda"},
        Accelerator.ONEAPI,
    )


def test_tpu_highest_is_explicit_and_not_inferred_from_x64() -> None:
    assert runtime_errors({**_runtime(), "matmul_precision": "default"}, Accelerator.TPU)


def test_environment_excludes_provider_overrides(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "XLA_FLAGS",
        "JAX_PLATFORMS",
        "JAX_ENABLE_X64",
        "LIBTPU_INIT_ARGS",
        "ONEAPI_DEVICE_SELECTOR",
    ):
        monkeypatch.setenv(name, "hostile-inherited-value")
    environment = clean_environment(tmp_path, Accelerator.TPU)
    assert environment["JAX_PLATFORMS"] == "tpu"
    assert environment["JAX_ENABLE_X64"] == "1"
    assert (
        not {
            "PYTHONPATH",
            "LD_LIBRARY_PATH",
            "XLA_FLAGS",
            "LIBTPU_INIT_ARGS",
            "ONEAPI_DEVICE_SELECTOR",
        }
        & environment.keys()
    )
    assert host_errors("Darwin", "arm64") and host_errors("Linux", "aarch64")
    assert not host_errors("Linux", "x86_64")


def test_multi_device_host_qualifies_only_one_explicit_execution_device() -> None:
    assert runtime_errors(_runtime(), Accelerator.TPU) == []
    assert runtime_errors({**_runtime(), "execution_device_count": 8}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "process_count": 2}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "local_device_count": 0}, Accelerator.TPU)
    assert runtime_errors({**_runtime(), "qualified_device_id": None}, Accelerator.TPU)


def test_core_wheel_content_identity_changes_with_bytes(tmp_path: Path) -> None:
    wheel = tmp_path / "core.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", "first")
        archive.writestr("phydrax-0.2.2.dist-info/RECORD", "mutable installer record")
    first = wheel_files(wheel)
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phydrax/__init__.py", "second")
    assert first["phydrax/__init__.py"] != wheel_files(wheel)["phydrax/__init__.py"]
    assert not any(name.endswith("/RECORD") for name in first)


def test_failed_child_or_non_json_never_yields_runtime() -> None:
    assert last_json({"returncode": 1, "stdout": '{"status":"passed"}'}) is None
    assert last_json({"returncode": 0, "stdout": "no evidence"}) is None
    assert last_json({"returncode": 0, "stdout": "[]"}) is None
    assert last_json({"returncode": 0, "stdout": 'log\n{"observed":true}'}) == {
        "observed": True
    }


def _probe_payload() -> dict[str, object]:
    distributions: dict[str, object] = {}
    for name, version in {"phydrax": "0.2.2", **Accelerator.TPU.distributions}.items():
        files = {
            f"package/member-{index}.so": hashlib.sha256(str(index).encode()).hexdigest()
            for index in range(10_000 if name == "phydrax" else 1)
        }
        digest = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        distributions[name] = {
            "version": version,
            "files": files,
            "installed_sha256": digest,
        }
    return {
        "provenance": {
            "package": "/parser-input/venv/phydrax/__init__.py",
            "venv": "/parser-input/venv",
            "distributions": distributions,
        },
        "runtime": _runtime(),
        "diagnostics": {"facts": {"device_probed": True}, "issues": []},
    }


def test_large_authoritative_probe_sidecar_is_complete_and_hash_validated(
    tmp_path: Path,
) -> None:
    payload = _probe_payload()
    path = tmp_path / "probe.json"
    write_json_evidence(path, payload)
    assert path.stat().st_size > TEXT_LIMIT
    assert read_json_evidence(path) == payload
    assert probe_record(path, Accelerator.TPU) == payload
    decoded = json.loads(path.read_text())
    decoded["provenance"]["distributions"]["phydrax"]["files"][
        "package/member-9999.so"
    ] = "0" * 64
    write_json_evidence(path, decoded)
    assert probe_record(path, Accelerator.TPU) is None


def test_probe_schema_and_size_limits_refuse_instead_of_truncating(
    tmp_path: Path,
) -> None:
    path = tmp_path / "probe.json"
    with pytest.raises(ValueError, match="exceeds"):
        write_json_evidence(path, {"large": "x" * 100}, max_bytes=32)
    assert not path.exists()
    path.write_text('{"large":"' + "x" * 100 + '"}')
    with pytest.raises(ValueError, match="exceeds"):
        read_json_evidence(path, max_bytes=32)
    assert probe_record(path, Accelerator.TPU) is None
    write_json_evidence(path, {"provenance": []})
    assert probe_record(path, Accelerator.TPU) is None


def test_process_logs_are_bounded_without_losing_retained_output(tmp_path: Path) -> None:
    result = run(
        [sys.executable, "-c", f"print('x' * {TEXT_LIMIT * 3})"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == 0 and not result["timed_out"]
    full_log = Path(result["stdout_path"]).read_bytes()
    assert len(result["stdout"].encode()) <= TEXT_LIMIT
    assert len(full_log) > TEXT_LIMIT
    assert hashlib.sha256(full_log).hexdigest() == result["stdout_sha256"]


def test_process_timeout_is_not_a_success_or_hardware_refusal(tmp_path: Path) -> None:
    result = run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=0.1,
    )
    assert result["timed_out"] and result["returncode"] != 0


def test_process_group_cleanup_terminates_descendants_after_parent_exits(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "orphan-ran"
    child = f"import time; from pathlib import Path; time.sleep(0.5); Path({str(marker)!r}).write_text('orphan')"
    source = f"import subprocess,sys; child=subprocess.Popen([sys.executable,'-c',{child!r}]); print(child.pid,flush=True)"
    result = run(
        [sys.executable, "-c", source],
        cwd=tmp_path,
        environment=dict(os.environ),
        timeout=10,
    )
    assert result["returncode"] == 0 and result["stdout"].strip().isdigit()
    time.sleep(0.7)
    assert not marker.exists()
