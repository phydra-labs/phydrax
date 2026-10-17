"""Hardware qualification cannot accept missing required native coverage."""

from pathlib import Path

import pytest

from tools import qualify_cuda_wheels as qualification


@pytest.mark.parametrize(
    ("required_gpus", "skip_reason", "refused"),
    [
        (1, "Same-plan cuDSS qualification requires two CUDA devices", False),
        (2, "Same-plan cuDSS qualification requires two CUDA devices", True),
        (2, "Non-default GPU AmgX qualification requires at least two GPUs", True),
        (2, "QUDA second-device refusal requires two CUDA devices", True),
        (2, "Classical AMG PR #386 regression requires the CUDA 13 companion", False),
        (1, "Install phydrax[cuda13] to load the companion", True),
    ],
)
def test_required_native_coverage_cannot_skip(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    required_gpus: int,
    skip_reason: str,
    refused: bool,
) -> None:
    evidence: qualification.Evidence = {
        "timestamp": "",
        "completed_at": "",
        "major": 13,
        "required_gpus": required_gpus,
        "venv": str(tmp_path),
        "wheels": {},
        "install": [],
        "tests": None,
        "audits": {},
        "extras_gate": None,
        "errors": [],
        "successful": False,
    }

    def completed_tests(
        command: list[str],
        *,
        environment: dict[str, str] | None = None,
        cwd: Path = qualification.ROOT,
    ) -> qualification.CommandEvidence:
        return {
            "command": command,
            "returncode": 0,
            "stdout": f"SKIPPED [1] tests/provider.py:1: {skip_reason}\n",
            "stderr": "",
        }

    monkeypatch.setattr(qualification, "run", completed_tests)
    qualification.run_tests(tmp_path / "python", tmp_path, evidence)
    assert bool(evidence["errors"]) is refused
