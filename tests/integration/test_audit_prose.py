#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Exercise prose audit findings and command-line filtering."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools import _branch_scope, audit_contract_candidates, audit_prose


def _source(tmp_path: Path, source: str, /) -> Path:
    path = tmp_path / "module.py"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.parametrize(
    "word",
    ["actually", "genuine", "genuinely"],
    ids=["adverb", "adjective", "adverbial-adjective"],
)
def test_docstring_rejects_intensifiers(tmp_path: Path, word: str) -> None:
    path = _source(tmp_path, f'"""Return {word} useful evidence."""\n')
    findings, count = audit_prose.audit([path])
    assert count == 1
    assert [(item.kind, item.context, item.detail) for item in findings] == [
        ("intensifier", "docstring", word)
    ]


@pytest.mark.parametrize(
    "phrase",
    [
        "Use exact arithmetic and an exact predicate.",
        "Use canonical ordering for a canonical fingerprint.",
        "Compare actual consumption with the admitted upper bound.",
        "Retain source authority.",
        "Compare original coordinates with lifted coordinates.",
        "Round the exact source-owned P1 root to its binary64 carrier.",
    ],
    ids=[
        "precision",
        "identity",
        "resource-bound",
        "provenance",
        "derived-coordinates",
        "unrounded-value",
    ],
)
def test_domain_phrases_pass(tmp_path: Path, phrase: str) -> None:
    path = _source(tmp_path, f'"""{phrase}"""\n')
    assert audit_prose.audit([path])[0] == []


@pytest.mark.parametrize(
    "phrase",
    [
        "Return actual work.",
        "Keep original coordinates.",
        "Return exact results.",
        "Return exact roots.",
        "Return canonical evidence.",
        "Return actual original authoritative results.",
    ],
    ids=[
        "missing-bound",
        "missing-derived-object",
        "precision-claim",
        "unrounded-value-without-rounding-contrast",
        "identity-claim",
        "adjective-stack",
    ],
)
def test_unlicensed_domain_words_are_findings(tmp_path: Path, phrase: str) -> None:
    path = _source(tmp_path, f'"""{phrase}"""\n')
    findings, _ = audit_prose.audit([path])
    assert findings
    assert all(item.kind == "restricted-word" for item in findings)


@pytest.mark.parametrize(
    "message",
    [
        '"invalid input"',
        '"Plan.field must be positive"',
        '"field must be positive; got -1"',
    ],
    ids=["no-contract", "no-observed-value", "unqualified-field"],
)
def test_exception_message_format_violation(tmp_path: Path, message: str) -> None:
    path = _source(tmp_path, f"raise ValueError({message})\n")
    findings, _ = audit_prose.audit([path])
    assert [item.kind for item in findings] == ["message-format"]


@pytest.mark.parametrize(
    "message",
    [
        '"Plan.field must be positive; got -1"',
        'f"Plan.field must be positive; got {value}"',
        '"Plan.field must be positive; " + "got -1"',
        'f"{field} must be positive; got {value}"',
        '"Plan.field must be positive; got " + str(value)',
        'f"Plan.field must be positive; got {original}"',
    ],
    ids=[
        "literal",
        "formatted-observation",
        "concatenated",
        "dynamic-qualified-field",
        "dynamic-concatenation",
        "interpolation-not-prose",
    ],
)
def test_exception_messages_with_conditions_and_observations_pass(
    tmp_path: Path, message: str
) -> None:
    path = _source(tmp_path, f"raise ValueError({message})\n")
    assert audit_prose.audit([path])[0] == []


def test_exception_message_lexicon_is_scanned(tmp_path: Path) -> None:
    path = _source(
        tmp_path, 'raise ValueError("Plan.field must be genuinely positive; got -1")\n'
    )
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.context) for item in findings] == [
        ("intensifier", "exception")
    ]


def test_builtin_exception_constructor_is_scanned_without_a_raise(tmp_path: Path) -> None:
    path = _source(
        tmp_path,
        'error = StopIteration("Plan.field must be genuinely positive; got -1")\n',
    )
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.context) for item in findings] == [
        ("intensifier", "exception")
    ]


def test_declared_public_type_names_are_allowlisted(tmp_path: Path) -> None:
    path = _source(tmp_path, 'class Witness:\n    """Return a Witness."""\n')
    assert audit_prose.audit([path])[0] == []


def test_undeclared_type_name_is_not_allowlisted(tmp_path: Path) -> None:
    path = _source(tmp_path, '"""Return a Witness."""\n')
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.detail) for item in findings] == [
        ("restricted-word", "Witness")
    ]


def test_declared_fields_do_not_license_unqualified_prose(tmp_path: Path) -> None:
    path = _source(
        tmp_path,
        'class Plan:\n    """Retain Plan.authority and `authority`, not rhetorical authority."""\n'
        "    authority: str\n",
    )
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.detail) for item in findings] == [
        ("restricted-word", "authority")
    ]


def test_literal_in_dynamic_concatenation_is_scanned(tmp_path: Path) -> None:
    path = _source(
        tmp_path,
        'raise ValueError("Plan.field must be genuinely positive; got " + str(value))\n',
    )
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.context) for item in findings] == [
        ("intensifier", "exception")
    ]


def test_comment_scanning_excludes_ordinary_string_data(tmp_path: Path) -> None:
    path = _source(tmp_path, 'value = "genuinely"\n# Actually retain the value.\n')
    findings, _ = audit_prose.audit([path])
    assert [(item.kind, item.context, item.line, item.detail) for item in findings] == [
        ("intensifier", "comment", 2, "Actually")
    ]


def test_json_output_reports_findings_and_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _source(tmp_path, "# genuinely\n")
    assert audit_prose.main(["--json", str(path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["files_scanned"] == 1
    assert report["count"] == 1
    assert report["findings"] == [
        {
            "file": str(path),
            "line": 1,
            "column": 3,
            "kind": "intensifier",
            "context": "comment",
            "detail": "genuinely",
        }
    ]


def _policy(root: Path, entries: dict[str, bool], /) -> None:
    path = root / _branch_scope.POLICY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "path\tprose_test\n"
        + "".join(
            f"{name}\t{int(selected)}\n" for name, selected in sorted(entries.items())
        ),
        encoding="utf-8",
    )


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, text=True, capture_output=True
    ).stdout.strip()


def test_changed_since_filters_unchanged_files_and_keeps_worktree_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    unchanged = package / "unchanged.py"
    changed = package / "changed.py"
    unchanged.write_text("# genuinely\nclass Witness:\n    pass\n", encoding="utf-8")
    changed.write_text("# Before.\n", encoding="utf-8")
    _policy(tmp_path, {"phydrax/unchanged.py": False})
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
    changed.write_text('"""Return a Witness."""\n# actually\n', encoding="utf-8")
    (package / "added.py").write_text("# genuine\n", encoding="utf-8")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    assert audit_prose.main(["--json", "--changed-since", revision, str(package)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["files_scanned"] == 2
    assert [item["file"] for item in report["findings"]] == [
        "phydrax/added.py",
        "phydrax/changed.py",
    ]


def _commit_baseline(root: Path, /) -> str:
    _git(root, "init", "-q")
    _git(root, "add", ".")
    _git(
        root,
        "-c",
        "user.name=Audit Test",
        "-c",
        "user.email=audit@example.test",
        "commit",
        "-qm",
        "baseline",
    )
    return _git(root, "rev-parse", "HEAD")


def test_changed_since_scans_unowned_files_only_on_changed_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    consumer = package / "consumer.py"
    consumer.write_text("# genuinely legacy\nvalue = 1\n", encoding="utf-8")
    (package / "owned.py").write_text("value = 0\n", encoding="utf-8")
    _policy(tmp_path, {"phydrax/owned.py": False})
    revision = _commit_baseline(tmp_path)
    consumer.write_text("# genuinely legacy\nvalue = 2  # actually\n", encoding="utf-8")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    findings, count = audit_prose.audit([package], changed_since=revision)
    assert count == 1
    assert [(item.line, item.detail) for item in findings] == [(2, "actually")]


def test_changed_since_scans_policy_listed_files_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    owned = package / "owned.py"
    owned.write_text("# genuinely legacy\nvalue = 1\n", encoding="utf-8")
    _policy(tmp_path, {"phydrax/owned.py": False})
    revision = _commit_baseline(tmp_path)
    owned.write_text("# genuinely legacy\nvalue = 2\n", encoding="utf-8")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    findings, _ = audit_prose.audit([package], changed_since=revision)
    assert [(item.line, item.detail) for item in findings] == [(1, "genuinely")]


def test_default_changed_scope_scans_branch_authored_files_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _policy(
        tmp_path,
        {"tests/unit/geometry/test_geometry.py": True, "tools/owned_tool.py": False},
    )
    legacy = "# genuinely legacy\nvalue = 1\n"
    modified = [
        "tests/unit/geometry/test_geometry.py",
        "tests/unit/other/test_modified.py",
        "tools/meshing_design_source.py",
        "tools/native_moving_overset.py",
        "tools/owned_tool.py",
    ]
    added = [
        "examples/new_example.py",
        "tests/unit/other/test_added.py",
        "tests/unit/test_untracked.py",
        "tools/new_tool.py",
    ]
    for name in modified:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(legacy, encoding="utf-8")
    revision = _commit_baseline(tmp_path)
    for name in modified:
        (tmp_path / name).write_text(
            legacy.replace("1", "2  # actually"), encoding="utf-8"
        )
    for name in added:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(legacy, encoding="utf-8")
    _git(tmp_path, "add", "tests/unit/other/test_added.py")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    findings, count = audit_prose.audit(changed_since=revision)
    # A modified file outside the default selection stays out; an added file
    # outside phydrax/, tests/ and tools/ is not a gate path.
    assert count == 7
    assert [(item.file, item.line, item.detail) for item in findings] == [
        ("tests/unit/geometry/test_geometry.py", 1, "genuinely"),
        ("tests/unit/geometry/test_geometry.py", 2, "actually"),
        ("tests/unit/other/test_added.py", 1, "genuinely"),
        ("tests/unit/test_untracked.py", 1, "genuinely"),
        ("tools/meshing_design_source.py", 2, "actually"),
        ("tools/native_moving_overset.py", 2, "actually"),
        ("tools/new_tool.py", 1, "genuinely"),
        ("tools/owned_tool.py", 1, "genuinely"),
        ("tools/owned_tool.py", 2, "actually"),
    ]


@pytest.mark.parametrize(
    "setting",
    [
        None,
        ("diff.renames", "false"),
        ("diff.renames", "copies"),
        ("diff.mnemonicPrefix", "true"),
        ("diff.noprefix", "true"),
        ("diff.interHunkContext", "10"),
    ],
    ids=["default", "no-renames", "copies", "mnemonic", "no-prefix", "hunk-context"],
)
def test_changed_since_scopes_staged_rename_to_rewritten_lines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, setting: tuple[str, str] | None
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    padding = "".join(f"PAD_{index} = {index}\n" for index in range(8))
    (package / "legacy.py").write_text(
        "value = 1\n# genuinely legacy\nother = 1\n" + padding, encoding="utf-8"
    )
    (package / "owned.py").write_text("value = 0\n", encoding="utf-8")
    _policy(tmp_path, {"phydrax/owned.py": False})
    revision = _commit_baseline(tmp_path)
    _git(tmp_path, "mv", "phydrax/legacy.py", "phydrax/moved.py")
    (package / "moved.py").write_text(
        "value = 2\n# genuinely legacy\nother = 2  # actually\n" + padding,
        encoding="utf-8",
    )
    if setting is not None:
        _git(tmp_path, "config", *setting)
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    findings, count = audit_prose.audit(changed_since=revision)
    assert count == 1
    assert [(item.file, item.line, item.detail) for item in findings] == [
        ("phydrax/moved.py", 3, "actually")
    ]


@pytest.mark.parametrize("gate", ["prose", "contracts"])
def test_branch_gates_admit_policy_paths_deleted_since_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    (package / "removed.py").write_text("# genuinely legacy\n", encoding="utf-8")
    (package / "changed.py").write_text("VALUE = 1\n", encoding="utf-8")
    _policy(tmp_path, {"phydrax/removed.py": False})
    revision = _commit_baseline(tmp_path)
    (package / "removed.py").unlink()
    (package / "changed.py").write_text(
        "VALUE = 1\n\n\n"
        "def added(value: Operand) -> None:\n"
        "    # actually\n"
        "    if not isinstance(value, Operand):\n"
        '        raise TypeError("Operand.value must be an Operand; got None")\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    monkeypatch.setattr(audit_contract_candidates, "ROOT", tmp_path)
    if gate == "prose":
        findings, _ = audit_prose.audit([package], changed_since=revision)
        assert [(item.file, item.line, item.detail) for item in findings] == [
            ("phydrax/changed.py", 5, "actually")
        ]
    else:
        guards = audit_contract_candidates.audit_changed_signatures(package, revision)
        assert [(item.file, item.function, item.disposition) for item in guards] == [
            ("phydrax/changed.py", "added", "undisposed-annotation-guard")
        ]


@pytest.mark.parametrize("gate", ["prose", "contracts"])
def test_branch_diagnostics_preserve_whole_mixed_and_new_file_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    _policy(
        tmp_path,
        {"phydrax/whole.py": False, "phydrax/whole_unchanged.py": False},
    )
    source = (
        "def legacy(value: Operand) -> None:\n"
        "    # genuinely\n"
        "    if not isinstance(value, Operand):\n"
        '        raise TypeError("Operand.value must be an Operand; got None")\n'
    )
    for name in ["whole", "whole_unchanged", "mixed"]:
        (package / f"{name}.py").write_text(source, encoding="utf-8")
    revision = _commit_baseline(tmp_path)
    (package / "whole.py").write_text(source + "\nVALUE = 1\n", encoding="utf-8")
    added_source = source.replace("legacy", "added").replace("genuinely", "actually")
    (package / "mixed.py").write_text(source + "\n" + added_source, encoding="utf-8")
    (package / "tracked_new.py").write_text(source, encoding="utf-8")
    _git(tmp_path, "add", "phydrax/tracked_new.py")
    (package / "untracked_new.py").write_text(source, encoding="utf-8")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    monkeypatch.setattr(audit_contract_candidates, "ROOT", tmp_path)
    if gate == "prose":
        findings, count = audit_prose.audit([package], changed_since=revision)
        assert count == 4
        assert [(item.file, item.line, item.detail) for item in findings] == [
            ("phydrax/mixed.py", 7, "actually"),
            ("phydrax/tracked_new.py", 2, "genuinely"),
            ("phydrax/untracked_new.py", 2, "genuinely"),
            ("phydrax/whole.py", 2, "genuinely"),
        ]
    else:
        findings = audit_contract_candidates.audit_changed_signatures(package, revision)
        assert [
            (item.file, item.line, item.function, item.disposition) for item in findings
        ] == [
            ("phydrax/mixed.py", 8, "added", "undisposed-annotation-guard"),
            ("phydrax/tracked_new.py", 3, "legacy", "undisposed-annotation-guard"),
            ("phydrax/untracked_new.py", 3, "legacy", "undisposed-annotation-guard"),
            ("phydrax/whole.py", 3, "legacy", "undisposed-annotation-guard"),
        ]


@pytest.mark.parametrize("gate", ["prose", "contracts"])
@pytest.mark.parametrize(
    "policy",
    [
        None,
        "",
        "path\tprose_test\n",
        "wrong\tcolumns\nphydrax/owned.py\t0\n",
        "path\tprose_test\nphydrax/owned.py\n",
        "path\tprose_test\nphydrax/owned.py\t0\textra\n",
        "path\tprose_test\nphydrax/owned.py\t0\nphydrax/owned.py\t0\n",
        "path\tprose_test\nphydrax/z.py\t0\nphydrax/a.py\t0\n",
        "path\tprose_test\nphydrax/../outside.py\t0\n",
        "path\tprose_test\nphydrax/*.py\t0\n",
        "path\tprose_test\nphydrax/owned.py\tinvalid\n",
        "path\tprose_test\nphydrax/owned.py\t1\n",
        "path\tprose_test\nphydrax/absent.py\t0\n",
    ],
    ids=[
        "missing",
        "empty",
        "no-rows",
        "header",
        "missing-column",
        "extra-column",
        "duplicate",
        "unsorted",
        "outside",
        "glob",
        "flag",
        "non-test-selection",
        "absent",
    ],
)
def test_branch_gates_refuse_missing_or_invalid_scope_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: str, policy: str | None
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    module = package / "owned.py"
    module.write_text("# Before.\n", encoding="utf-8")
    revision = _commit_baseline(tmp_path)
    module.write_text("# actually\n", encoding="utf-8")
    if policy is not None:
        path = tmp_path / _branch_scope.POLICY
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(policy, encoding="utf-8")
    monkeypatch.setattr(audit_prose, "ROOT", tmp_path)
    monkeypatch.setattr(audit_contract_candidates, "ROOT", tmp_path)
    error = FileNotFoundError if policy is None else ValueError
    with pytest.raises(error, match="branch-scope.tsv"):
        if gate == "prose":
            audit_prose.audit([package], changed_since=revision)
        else:
            audit_contract_candidates.audit_changed_signatures(package, revision)
