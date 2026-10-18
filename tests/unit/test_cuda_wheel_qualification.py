"""Hardware qualification cannot accept missing required native coverage."""

import json
import sys
from pathlib import Path
from typing import NoReturn
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
    ) -> qualification.CommandEvidence:
        _provider_report(
            tmp_path / "native-provider-results.xml", list(range(required_gpus))
        )
        return {
            "command": command,
            "returncode": 0,
            "stdout": f"SKIPPED [1] tests/provider.py:1: {skip_reason}\n",
            "stderr": "",
        }

    monkeypatch.setattr(qualification, "run", completed_tests)
    qualification.run_tests(tmp_path / "python", tmp_path, evidence)
    assert bool(evidence["errors"]) is refused


@pytest.mark.parametrize("required_gpus", [1, 2, 4])
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


@pytest.mark.parametrize("required_gpus", [1, 2, 4])
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
    monkeypatch.setattr(sys, "argv", ["audit", "1"])
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
