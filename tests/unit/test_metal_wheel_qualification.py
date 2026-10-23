"""Metal wheel qualification cannot pass with missing, foreign or stale evidence."""

import copy
import json
import os
import shutil
import subprocess
import sys
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tools import qualify_metal_wheels as qualification
from tools.check_metal_extras import CompanionFacts, record_digest


SOURCE_ID = "b" * 64
COMPANION: CompanionFacts = {
    "version": "0.2.2",
    "jax_version": "0.12.0",
    "jaxlib_version": "0.12.0",
    "minimum_macos": "14.0",
    "build_id": "c" * 64,
    "source_id": SOURCE_ID,
    "tags": ["py3-none-macosx_14_0_arm64"],
}
RUNTIME: dict[str, object] = {
    "platform": "mps",
    "platform_version": f"phydrax-metal 0.2.2 {SOURCE_ID}",
    "device_kind": "Apple M1 Max",
    "device_id": 0,
    "x64_enabled": True,
}
PATH = "tests/integration/test_metal_fake.py"
DEVICE = qualification.RequiredFile("device", ("test_a", "test_b[x]", "test_b[y]"))
HOST = qualification.RequiredFile(
    "host", ("test_host", "test_child"), child_runtime_tests=("test_child",)
)


def _evidence(runtime: dict[str, object] | None = None) -> list[str]:
    return [] if runtime is None else [json.dumps(runtime, sort_keys=True)]


def _report(required: qualification.RequiredFile, venv: Path) -> dict[str, Any]:
    nodes = []
    for nodeid in required.nodeids(PATH):
        name = required.runtime_property(nodeid.removeprefix(f"{PATH}::"))
        nodes.append(
            {
                "nodeid": nodeid,
                "outcome": "passed",
                "message": "",
                "duration_seconds": 0.1,
                "xfail": False,
                "properties": [] if name is None else [[name, _evidence(RUNTIME)[0]]],
            }
        )
    return {
        "run": "run-1",
        "exit_status": 0,
        "collected": list(required.nodeids(PATH)),
        "failed_collectors": [],
        "nodes": nodes,
        "modules": {
            "phydrax": str(venv / "lib/python3.13/site-packages/phydrax/__init__.py"),
            "phydrax_metal": str(
                venv / "lib/python3.13/site-packages/phydrax_metal/__init__.py"
            ),
        },
        "checkout_modules": [],
        "sys_path": [],
    }


def _evaluate(
    report: dict[str, Any] | None,
    venv: Path,
    *,
    required: qualification.RequiredFile = DEVICE,
    returncode: int = 0,
) -> tuple[list[str], dict[str, str]]:
    return qualification.file_errors(
        PATH,
        required,
        report,
        returncode=returncode,
        run_id="run-1",
        expected_runtime=RUNTIME,
        venv=venv,
    )


def test_complete_device_and_host_files_pass(tmp_path: Path) -> None:
    for required in (DEVICE, HOST):
        errors, coverage = _evaluate(
            _report(required, tmp_path), tmp_path, required=required
        )
        assert errors == []
        assert coverage == dict.fromkeys(required.nodeids(PATH), "passed")


def _node(report: dict[str, Any], index: int = 0) -> dict[str, Any]:
    return report["nodes"][index]


def _set_runtime(**changes: object) -> Callable[[dict[str, Any]], None]:
    def mutate(report: dict[str, Any]) -> None:
        _node(report)["properties"] = [
            ["metal_runtime", _evidence({**RUNTIME, **changes})[0]]
        ]

    return mutate


_DEFECTS: dict[str, Callable[[dict[str, Any]], object]] = {
    "missing-node": lambda report: (report["nodes"].pop(), report["collected"].pop()),
    "skipped": lambda report: _node(report).update(
        outcome="skipped", message="Install phydrax[metal]"
    ),
    "failed": lambda report: _node(report).update(
        outcome="failed", message="AssertionError"
    ),
    "xpass": lambda report: _node(report).update(xfail=True),
    "duplicate-report": lambda report: report["nodes"].append(
        copy.deepcopy(_node(report))
    ),
    "duplicate-collection": lambda report: report["collected"].append(
        report["collected"][0]
    ),
    "unexpected-node": lambda report: (
        report["collected"].append(f"{PATH}::test_new"),
        report["nodes"].append(
            {**copy.deepcopy(_node(report)), "nodeid": f"{PATH}::test_new"}
        ),
    ),
    "no-runtime-evidence": lambda report: _node(report).update(properties=[]),
    "two-runtime-records": lambda report: _node(report)["properties"].append(
        _node(report)["properties"][0]
    ),
    "malformed-runtime": lambda report: _node(report).update(
        properties=[["metal_runtime", "{"]]
    ),
    "child-property-in-device-file": lambda report: _node(report).update(
        properties=[["metal_child_runtime", _evidence(RUNTIME)[0]]]
    ),
    "cpu-placement": _set_runtime(platform="cpu", platform_version="cpu"),
    "other-source-build": _set_runtime(
        platform_version=f"phydrax-metal 0.2.2 {'d' * 64}"
    ),
    "upstream-runtime": _set_runtime(platform_version="jax-mps 0.11.0"),
    "other-device": _set_runtime(device_id=1),
    "x64-off": _set_runtime(x64_enabled=False),
    "stale-run": lambda report: report.update(run="run-0"),
    "session-failure": lambda report: report.update(exit_status=1),
    "collection-error": lambda report: report.update(failed_collectors=[PATH]),
    "checkout-import": lambda report: report.update(checkout_modules=["phydrax"]),
    "source-phydrax": lambda report: report["modules"].update(
        phydrax=str(qualification.ROOT / "phydrax/__init__.py")
    ),
    "no-installed-phydrax": lambda report: report.update(modules={}),
}


@pytest.mark.parametrize("defect", sorted(_DEFECTS))
def test_incomplete_or_foreign_device_evidence_fails(tmp_path: Path, defect: str) -> None:
    report = _report(DEVICE, tmp_path)
    _DEFECTS[defect](report)
    errors, coverage = _evaluate(report, tmp_path)
    assert errors
    passed = [nodeid for nodeid, outcome in coverage.items() if outcome == "passed"]
    assert len(passed) < len(DEVICE.tests) or defect in {
        "duplicate-collection",
        "unexpected-node",
        "stale-run",
        "session-failure",
        "collection-error",
        "checkout-import",
        "source-phydrax",
        "no-installed-phydrax",
    }


@pytest.mark.parametrize(
    ("returncode", "report"),
    [(1, True), (0, False), (-1, True)],
    ids=["child-exit", "no-report", "timeout"],
)
def test_child_status_or_absent_report_fails(
    tmp_path: Path, returncode: int, report: bool
) -> None:
    errors, _ = _evaluate(
        _report(DEVICE, tmp_path) if report else None, tmp_path, returncode=returncode
    )
    assert errors


def test_host_child_runtime_evidence_is_required_only_where_declared(
    tmp_path: Path,
) -> None:
    report = _report(HOST, tmp_path)
    host, child = report["nodes"]
    assert host["properties"] == []
    child["properties"] = [
        ["metal_child_runtime", _evidence({**RUNTIME, "platform": "cpu"})[0]]
    ]
    errors, coverage = _evaluate(report, tmp_path, required=HOST)
    assert coverage[host["nodeid"]] == "passed"
    assert coverage[child["nodeid"]] == "unverified-runtime"
    assert any("metal_child_runtime" in error for error in errors)


def _identity_record() -> dict[str, Any]:
    return {
        "successful": True,
        "issues": [],
        "facts": {
            "versions": {
                "jax": "0.12.0",
                "jaxlib": "0.12.0",
                "phydrax": "0.2.2",
                "phydrax-metal": "0.2.2",
            },
            "competing_plugins": [],
            "selected_platforms": ["mps"],
            "build_id": COMPANION["build_id"],
            "source_id": SOURCE_ID,
            "companion_platform": "mps",
            "required_jax": "0.12.0",
            "required_jaxlib": "0.12.0",
            "minimum_macos": "14.0",
            "macos_version": "15.6",
            "device_probed": True,
            "runtime_platform": "mps",
            "runtime_version": RUNTIME["platform_version"],
            "device_kind": "Apple M1 Max",
            "backend_error": None,
        },
        "devices": [
            {
                "platform": "mps",
                "platform_version": RUNTIME["platform_version"],
                "device_kind": "Apple M1 Max",
                "device_id": 0,
            }
        ],
        "x64_enabled": True,
        "matmul_precision": "highest",
        "environment": {
            "phydrax_version": "0.2.2",
            "jaxlib_version": "0.12.0",
            "jax": {
                "version": "0.12.0",
                "backend": "mps",
                "x64_enabled": True,
                "devices": [{"platform": "mps", "kind": "Apple M1 Max"}],
            },
        },
    }


def test_matching_runtime_identity_defines_per_case_evidence() -> None:
    assert qualification.identity_errors(
        _identity_record(), companion=COMPANION, version="0.2.2"
    ) == ([], RUNTIME)


_IDENTITY_DEFECTS: dict[str, Callable[[dict[str, Any]], object]] = {
    "unprobed": lambda record: record["facts"].update(device_probed=False),
    "report-issue": lambda record: record.update(
        successful=False, issues=[["conflicting-plugins", "error", "jax-mps"]]
    ),
    "other-build": lambda record: record["facts"].update(build_id="d" * 64),
    "other-source-loaded": lambda record: (
        record["facts"].update(runtime_version="phydrax-metal 0.2.2 other"),
        record["devices"][0].update(platform_version="phydrax-metal 0.2.2 other"),
    ),
    "competing-plugin": lambda record: record["facts"].update(
        competing_plugins=["jax-mps"]
    ),
    "unselected-platform": lambda record: record["facts"].update(selected_platforms=None),
    "jax-mismatch": lambda record: record["facts"]["versions"].update(jax="0.12.1"),
    "two-devices": lambda record: record["devices"].append(
        {**record["devices"][0], "device_id": 1}
    ),
    "cpu-device": lambda record: record["devices"][0].update(platform="cpu"),
    "x64-off": lambda record: record.update(x64_enabled=False),
    "reduced-matmul": lambda record: record.update(matmul_precision=None),
    "cpu-benchmark-environment": lambda record: record["environment"]["jax"].update(
        backend="cpu"
    ),
    "mixed-benchmark-devices": lambda record: record["environment"]["jax"][
        "devices"
    ].append({"platform": "cpu", "kind": "cpu"}),
    "no-probe-output": lambda record: record.clear(),
}


@pytest.mark.parametrize("defect", sorted(_IDENTITY_DEFECTS))
def test_mismatched_runtime_identity_refuses_qualification(defect: str) -> None:
    record = _identity_record()
    _IDENTITY_DEFECTS[defect](record)
    errors, expected = qualification.identity_errors(
        record or None, companion=COMPANION, version="0.2.2"
    )
    assert errors and expected is None


def _host_record(venv: Path) -> dict[str, Any]:
    return {
        "versions": {
            "jax": "0.12.0",
            "jaxlib": "0.12.0",
            "phydrax": "0.2.2",
            "phydrax-metal": "0.2.2",
        },
        "modules": {
            "phydrax": str(venv / "lib/python3.13/site-packages/phydrax/__init__.py"),
            "phydrax_metal": str(
                venv / "lib/python3.13/site-packages/phydrax_metal/__init__.py"
            ),
        },
        "site_packages": [str(venv / "lib/python3.13/site-packages")],
        "backends_initialized": False,
        "python": [3, 13],
        "implementation": "cpython",
        "free_threaded": False,
    }


@pytest.mark.parametrize(
    "defect",
    [
        "none",
        "source-checkout",
        "backend-initialized",
        "python-3.12",
        "companion-0.2.1",
        "no-probe",
    ],
)
def test_installed_package_probe_requires_fresh_venv_wheels(
    tmp_path: Path, defect: str
) -> None:
    record: dict[str, Any] | None = _host_record(tmp_path)
    assert record is not None
    if defect == "source-checkout":
        record["modules"]["phydrax"] = str(qualification.ROOT / "phydrax/__init__.py")
    elif defect == "backend-initialized":
        record["backends_initialized"] = True
    elif defect == "python-3.12":
        record["python"] = [3, 12]
    elif defect == "companion-0.2.1":
        record["versions"]["phydrax-metal"] = "0.2.1"
    elif defect == "no-probe":
        record = None
    errors = qualification.host_probe_errors(
        record, version="0.2.2", jax_version="0.12.0", venv=tmp_path
    )
    assert bool(errors) is (defect != "none")


def _wheel(path: Path, members: dict[str, bytes]) -> Path:
    record = "".join(
        f"{name},{record_digest(data)},{len(data)}\n" for name, data in members.items()
    )
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)
        archive.writestr(
            "demo-1.0.dist-info/RECORD", record + "demo-1.0.dist-info/RECORD,,\n"
        )
    return path


@pytest.mark.parametrize("defect", ["none", "modified", "absent", "data-scheme-modified"])
def test_installed_files_must_match_wheel_bytes(tmp_path: Path, defect: str) -> None:
    wheel = _wheel(
        tmp_path / "demo-1.0-py3-none-any.whl",
        {
            "demo/__init__.py": b"VALUE = 1\n",
            "demo-1.0.data/purelib/demo_extra.py": b"EXTRA = 2\n",
            "demo-1.0.data/scripts/demo": b"#!/bin/sh\n",
            "demo-1.0.dist-info/METADATA": b"Name: demo\nVersion: 1.0\n",
        },
    )
    site = tmp_path / "site-packages"
    (site / "demo").mkdir(parents=True)
    (site / "demo-1.0.dist-info").mkdir()
    (site / "demo/__init__.py").write_bytes(b"VALUE = 1\n")
    (site / "demo_extra.py").write_bytes(b"EXTRA = 2\n")
    (site / "demo-1.0.dist-info/METADATA").write_bytes(b"Name: demo\nVersion: 1.0\n")
    if defect == "modified":
        (site / "demo/__init__.py").write_bytes(b"VALUE = 2\n")
    elif defect == "absent":
        (site / "demo/__init__.py").unlink()
    elif defect == "data-scheme-modified":
        (site / "demo_extra.py").write_bytes(b"EXTRA = 3\n")
    errors = qualification.installed_file_errors(wheel, [site])
    assert bool(errors) is (defect != "none")


def _example(stdout: str, returncode: int = 0) -> qualification.CommandEvidence:
    return {
        "command": ["python", "metal_scientific_execution.py"],
        "cwd": "/tmp",
        "environment": {},
        "returncode": returncode,
        "timed_out": False,
        "stdout": stdout,
        "stderr": "",
    }


_EXAMPLE = {
    "device": {
        key: RUNTIME[key] for key in ("platform", "platform_version", "device_kind")
    },
    "x64_enabled": True,
    "status": "passed",
    "final_loss": 0.01,
}


@pytest.mark.parametrize(
    ("stdout", "returncode"),
    [
        ("progress\n" + json.dumps({**_EXAMPLE, "status": "failed"}), 0),
        (
            json.dumps({**_EXAMPLE, "device": {**_EXAMPLE["device"], "platform": "cpu"}}),
            0,
        ),
        (json.dumps({**_EXAMPLE, "x64_enabled": False}), 0),
        (json.dumps(_EXAMPLE) + "\ntrailing text", 0),
        (json.dumps(_EXAMPLE), 1),
        ("", 0),
    ],
    ids=[
        "failed-status",
        "cpu-device",
        "x64-off",
        "not-last-line",
        "nonzero-exit",
        "silent",
    ],
)
def test_example_run_requires_actual_device_json(stdout: str, returncode: int) -> None:
    assert qualification.example_errors(
        _example("progress\n" + json.dumps(_EXAMPLE)), RUNTIME
    ) == (
        [],
        _EXAMPLE,
    )
    errors, _ = qualification.example_errors(_example(stdout, returncode), RUNTIME)
    assert errors


@pytest.mark.parametrize(
    ("changes", "refused"),
    [
        ({}, False),
        ({"platform": "linux", "machine": "x86_64"}, True),
        ({"machine": "x86_64"}, True),
        ({"macos_version": "13.7"}, True),
        ({"gpu": None}, True),
    ],
    ids=["m1-max", "linux", "intel-mac", "old-macos", "unidentified-gpu"],
)
def test_qualification_host_identity(changes: dict[str, object], refused: bool) -> None:
    facts: dict[str, object] = {
        "platform": "darwin",
        "machine": "arm64",
        "macos_version": "15.6",
        "gpu": {"model": "Apple M1 Max", "cores": "32"},
        **changes,
    }
    assert bool(qualification.host_errors(facts, COMPANION)) is refused


def test_stale_case_report_is_never_reused(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    suite.mkdir()
    required = qualification.RequiredFile("device", ("test_a",))
    stale = _report(required, tmp_path)
    stale["run"] = "run-1"
    (suite / qualification.CASE_REPORT).write_text(json.dumps(stale))
    if os.name != "posix":
        with pytest.raises(RuntimeError, match="requires a POSIX host"):
            qualification.run_file(
                Path(sys.executable),
                tmp_path,
                suite,
                PATH,
                required,
                run_id="run-1",
                expected_runtime=RUNTIME,
                timeout=60.0,
            )
        assert not (suite / qualification.CASE_REPORT).exists()
        return
    false = shutil.which("false")
    assert false is not None
    record, coverage = qualification.run_file(
        Path(false),
        tmp_path,
        suite,
        PATH,
        required,
        run_id="run-1",
        expected_runtime=RUNTIME,
        timeout=60.0,
    )
    assert record["report"] is None
    assert any("no case report" in error for error in record["errors"])
    assert coverage == {}


_CASES = """
import json
import sys

import pytest

RUNTIME = json.dumps(RUNTIME_VALUE, sort_keys=True)


def test_passes(record_property):
    record_property("metal_runtime", RUNTIME)


def test_skips():
    pytest.skip("Install phydrax[metal] to run Apple GPU cases.")


@pytest.mark.xfail(reason="known")
def test_xfails():
    raise AssertionError


@pytest.mark.xfail(reason="unexpected pass")
def test_xpasses(record_property):
    record_property("metal_runtime", RUNTIME)


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("teardown")


def test_teardown_fails(broken_teardown, record_property):
    record_property("metal_runtime", RUNTIME)


def test_without_evidence():
    pass


def test_imports_checkout(record_property):
    sys.path.insert(0, CHECKOUT)
    import leaky_checkout_module

    record_property("metal_runtime", RUNTIME)
"""


@pytest.mark.parametrize("symlinked", [False, True], ids=["physical", "symlink"])
def test_case_plugin_reduces_real_pytest_sessions(
    tmp_path: Path, symlinked: bool
) -> None:
    if os.name != "posix":
        with pytest.raises(RuntimeError, match="requires a POSIX host"):
            qualification.run([sys.executable, "-c", "pass"], cwd=tmp_path, timeout=300.0)
        return
    suite = tmp_path / "suite"
    checkout = tmp_path / "checkout"
    suite.mkdir()
    if symlinked:
        alias = tmp_path / "suite-alias"
        alias.symlink_to(suite, target_is_directory=True)
        suite = alias
    checkout.mkdir()
    (checkout / "leaky_checkout_module.py").write_text("")
    qualification.write_case_plugin(suite)
    (suite / "pyproject.toml").write_text("[tool.pytest.ini_options]\n")
    (suite / "test_cases.py").write_text(
        _CASES.replace("RUNTIME_VALUE", repr(RUNTIME)).replace(
            "CHECKOUT", repr(str(checkout))
        )
    )
    report_path = suite / qualification.CASE_REPORT
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTEST_ADDOPTS")
        and not key.startswith("PYTEST_XDIST")
    }
    environment.update(
        PHYDRAX_METAL_QUALIFICATION_RUN="run-1",
        PHYDRAX_METAL_CASE_REPORT=str(report_path),
        PHYDRAX_METAL_CHECKOUT=str(checkout),
    )
    result = qualification.run(
        qualification.pytest_command(Path(sys.executable), suite, "test_cases.py"),
        cwd=suite,
        environment=environment,
        timeout=300.0,
    )
    report = json.loads(report_path.read_text())
    tests = (
        "test_passes",
        "test_skips",
        "test_xfails",
        "test_xpasses",
        "test_teardown_fails",
        "test_without_evidence",
        "test_imports_checkout",
    )
    errors, coverage = qualification.file_errors(
        "test_cases.py",
        qualification.RequiredFile("device", tests),
        report,
        returncode=result["returncode"],
        run_id="run-1",
        expected_runtime=RUNTIME,
        venv=tmp_path / "venv",
    )
    assert result["returncode"] != 0
    assert report["checkout_modules"] == ["leaky_checkout_module"]
    assert coverage == {
        "test_cases.py::test_passes": "passed",
        "test_cases.py::test_skips": "skipped",
        "test_cases.py::test_xfails": "skipped",
        "test_cases.py::test_xpasses": "expected-failure",
        "test_cases.py::test_teardown_fails": "failed",
        "test_cases.py::test_without_evidence": "unverified-runtime",
        "test_cases.py::test_imports_checkout": "passed",
    }
    assert any(
        "test_skips skipped" in error and "phydrax[metal]" in error for error in errors
    )
    assert any(
        "test_teardown_fails failed: RuntimeError: teardown" in error for error in errors
    )
    assert any("modules loaded from the checkout" in error for error in errors)


def test_required_inventory_matches_collected_nodes() -> None:
    discovered = {
        path.relative_to(qualification.ROOT).as_posix()
        for path in (qualification.ROOT / "tests/integration").glob("test_metal_*.py")
    }
    assert discovered == set(qualification.REQUIRED_FILES)
    expected = [
        nodeid
        for path, required in qualification.REQUIRED_FILES.items()
        for nodeid in required.nodeids(path)
    ]
    assert len(expected) == len(set(expected))
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONPATH", "PYTEST_ADDOPTS")
        and not key.startswith("PYTEST_XDIST")
    }
    # Collection must be host-only; never let it reach an accelerator.
    environment["JAX_PLATFORMS"] = "cpu"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "no:warnings",
            *qualification.REQUIRED_FILES,
        ],
        cwd=qualification.ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    collected = [line for line in result.stdout.splitlines() if "::" in line]
    assert result.returncode == 0, result.stdout + result.stderr
    assert sorted(collected) == sorted(expected)
