#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exercise the changed-file contract gate without importing audited code."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools import _branch_scope, audit_contract_candidates


SOURCE = """def accepts(value: Operand) -> None:
    if not isinstance(value, Operand):
        raise TypeError("Plan.value must be an Operand; got another kind")
"""


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    package = tmp_path / "phydrax"
    package.mkdir()
    (package / "unchanged.py").write_text(SOURCE, encoding="utf-8")
    (package / "changed.py").write_text("", encoding="utf-8")
    policy = tmp_path / _branch_scope.POLICY
    policy.parent.mkdir(parents=True)
    policy.write_text("path\tprose_test\nphydrax/changed.py\t0\n", encoding="utf-8")
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "add", ".")
    _git(
        tmp_path,
        "-c",
        "user.name=Audit Test",
        "-c",
        "user.email=audit@example.test",
        "commit",
        "-qm",
        "baseline",
    )
    revision = _git(tmp_path, "rev-parse", "HEAD")
    monkeypatch.setattr(audit_contract_candidates, "ROOT", tmp_path)
    monkeypatch.setattr(audit_contract_candidates, "PACKAGE", package)
    return tmp_path, package, revision


def test_changed_gate_reports_unchecked_annotated_guards(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    (package / "changed.py").write_text(SOURCE, encoding="utf-8")
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert [
        (item.file, item.function, item.parameter, item.disposition) for item in findings
    ] == [("phydrax/changed.py", "accepts", "value", "undisposed-annotation-guard")]


def test_changed_gate_reports_dispatch_chains_once(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    (package / "added.py").write_text(
        "def dispatch(value: Operand) -> int:\n"
        "    if isinstance(value, First):\n        return 1\n"
        "    elif isinstance(value, Second):\n        return 2\n"
        "    elif isinstance(value, Third):\n        return 3\n"
        "    return 0\n",
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert [(item.file, item.function, item.disposition) for item in findings] == [
        ("phydrax/added.py", "dispatch", "isinstance-dispatch")
    ]


def test_changed_gate_reports_mixed_none_and_type_dispatch(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    (package / "added.py").write_text(
        "def dispatch(value: Operand | None) -> int:\n"
        "    if value is None:\n        return 0\n"
        "    elif isinstance(value, Operand):\n        return 1\n"
        "    return 2\n",
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert [(item.file, item.function, item.disposition) for item in findings] == [
        ("phydrax/added.py", "dispatch", "isinstance-dispatch")
    ]


@pytest.mark.parametrize(
    ("disposition", "reason", "refused"),
    [
        ("retained", "source-addressed-function", True),
        ("retained", "unchecked-method", True),
        ("retained", "static-only-annotation", True),
        ("retained", "", True),
        ("migrated", "checked-signature", True),
        ("retained", "Provider output is decoded here before use as Operand.", False),
    ],
    ids=[
        "blanket-source",
        "blanket-method",
        "blanket-static",
        "empty",
        "stale-migration",
        "specific-boundary",
    ],
)
def test_changed_gate_requires_a_specific_retention_reason(
    branch: tuple[Path, Path, str], disposition: str, reason: str, refused: bool
) -> None:
    root, package, revision = branch
    (package / "changed.py").write_text(SOURCE, encoding="utf-8")
    dispositions = root / "dispositions.tsv"
    dispositions.write_text(
        "path\tsymbol\tparameter\tannotation\tdisposition\treason\n"
        f"phydrax/changed.py\taccepts\tvalue\tOperand\t{disposition}\t{reason}\n",
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(
        package, revision, dispositions=dispositions
    )
    assert bool(findings) is refused


def test_changed_gate_checks_annotated_scalar_and_compound_guards(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    (package / "changed.py").write_text(
        "def accepts(value: int) -> None:\n"
        "    if not isinstance(value, int) or value < 0:\n"
        "        raise TypeError('Plan.value must be a nonnegative integer; got invalid input')\n",
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert [(item.parameter, item.annotation) for item in findings] == [("value", "int")]


def test_changed_gate_reports_guard_in_checked_code_without_importing_it(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    (package / "changed.py").write_text(
        "raise RuntimeError('must not import audited code')\n@checked\n" + SOURCE,
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert len(findings) == 1
    assert findings[0].disposition == "undisposed-annotation-guard"


def test_changed_gate_cli_emits_json_and_fails_on_findings(
    branch: tuple[Path, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    root, package, revision = branch
    (package / "changed.py").write_text(SOURCE, encoding="utf-8")
    output = root / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit_contract_candidates.py",
            "--signatures",
            "--changed-since",
            revision,
            "--json",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as refused:
        audit_contract_candidates.main()
    assert refused.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report[0]["file"] == "phydrax/changed.py"
    assert report[0]["disposition"] == "undisposed-annotation-guard"


def test_changed_gate_passes_when_changed_files_have_no_findings(
    branch: tuple[Path, Path, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, package, revision = branch
    (package / "changed.py").write_text(
        "def accepts(value: Operand) -> None:\n    return None\n", encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["audit_contract_candidates.py", "--signatures", "--changed-since", revision],
    )
    assert audit_contract_candidates.main() is None


def test_changed_gate_ignores_guards_on_unchanged_lines_of_changed_files(
    branch: tuple[Path, Path, str],
) -> None:
    _, package, revision = branch
    unchanged = package / "unchanged.py"
    unchanged.write_text(SOURCE + "\nVALUE = 1\n", encoding="utf-8")
    assert audit_contract_candidates.audit_changed_signatures(package, revision) == []
    unchanged.write_text(
        SOURCE + "\ndef added(item: Operand) -> None:\n"
        "    if not isinstance(item, Operand):\n"
        '        raise TypeError("Plan.item must be an Operand; got another kind")\n',
        encoding="utf-8",
    )
    findings = audit_contract_candidates.audit_changed_signatures(package, revision)
    assert [(item.function, item.parameter) for item in findings] == [("added", "item")]


def test_changed_gate_accepts_a_specific_dispatch_disposition(
    branch: tuple[Path, Path, str],
) -> None:
    root, package, revision = branch
    (package / "added.py").write_text(
        "def dispatch(value: Operand) -> int:\n"
        "    if isinstance(value, First):\n        return 1\n"
        "    elif isinstance(value, Second):\n        return 2\n"
        "    return 0\n",
        encoding="utf-8",
    )
    dispositions = root / "dispositions.tsv"
    dispositions.write_text(
        "path\tsymbol\tparameter\tannotation\tdisposition\treason\n"
        "phydrax/added.py\tdispatch\tvalue\tFirst\tretained\t"
        "Provider plugins register further Operand subclasses at runtime.\n",
        encoding="utf-8",
    )
    assert (
        audit_contract_candidates.audit_changed_signatures(
            package, revision, dispositions=dispositions
        )
        == []
    )
