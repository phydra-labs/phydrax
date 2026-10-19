#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Select the branch scope shared by the static audit gates.

A gate scans policy-listed, added and untracked files in full and other changed
files only on the lines added or rewritten since a revision, so text the branch
never touched stays with its owners. ``tools/branch-scope.tsv`` lists each
whole-file path once; ``prose_test=1`` also includes a test in the prose gate's
default scan. Missing or invalid policy data refuses the scan.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


POLICY = "tools/branch-scope.tsv"
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")


def _git(root: Path, /, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _python_files(root: Path, names: str, /) -> set[Path]:
    return {
        root / name
        for name in names.split("\0")
        if name.endswith(".py") and (root / name).is_file()
    }


def changed_files(root: Path, revision: str, /) -> set[Path]:
    """Return existing tracked changes and untracked Python files since a revision."""
    return _python_files(
        root,
        _git(root, "diff", "--name-only", "-z", "--diff-filter=ACMRT", revision, "--")
        + _git(root, "ls-files", "--others", "--exclude-standard", "-z"),
    )


def load_policy(root: Path, /) -> dict[str, bool]:
    """Read sorted whole-file paths and their default prose-test selection."""
    policy = root / POLICY
    rows = policy.read_text(encoding="utf-8").splitlines()
    if not rows or rows[0] != "path\tprose_test" or len(rows) == 1:
        raise ValueError(
            f"BranchScope.policy must contain a header and scope rows; got {policy}"
        )
    paths: dict[str, bool] = {}
    previous = ""
    for line, row in enumerate(rows[1:], start=2):
        fields = row.split("\t")
        if len(fields) != 2:
            raise ValueError(
                f"BranchScope.policy must contain two columns; got {policy}:{line}"
            )
        name, prose_test = fields
        if (
            re.fullmatch(
                r"(?:phydrax|tools|tests)/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py", name
            )
            is None
            or name <= previous
            or prose_test not in {"0", "1"}
            or (prose_test == "1" and not name.startswith("tests/"))
        ):
            raise ValueError(
                f"BranchScope.policy must list sorted unique Python paths and test flags; got {policy}:{line}"
            )
        paths[name] = prose_test == "1"
        previous = name
    return paths


def whole_files(root: Path, revision: str, /) -> set[Path]:
    """Return files scanned in full: policy-listed, added, or untracked since a revision."""
    policy = load_policy(root)
    files = _python_files(
        root,
        _git(root, "diff", "--name-only", "-z", "--diff-filter=A", revision, "--")
        + _git(root, "ls-files", "--others", "--exclude-standard", "-z"),
    )
    files.update(root / name for name in policy)
    return files


def changed_lines(root: Path, revision: str, /) -> dict[Path, frozenset[int]]:
    """Return the lines of each tracked file added or rewritten since a revision."""
    lines: dict[Path, set[int]] = {}
    current: Path | None = None
    diff = _git(root, "diff", "-U0", "--no-color", "--no-ext-diff", revision, "--")
    for row in diff.splitlines():
        if row.startswith("+++ "):
            current = root / row[6:] if row.startswith("+++ b/") else None
        elif current is not None and (hunk := _HUNK.match(row)) is not None:
            start = int(hunk[1])
            count = 1 if hunk[2] is None else int(hunk[2])
            lines.setdefault(current, set()).update(range(start, start + count))
    return {path: frozenset(rows) for path, rows in lines.items()}


def in_scope(
    path: Path,
    line: int,
    whole: set[Path],
    lines: dict[Path, frozenset[int]],
    /,
) -> bool:
    """Whether a finding at ``path:line`` belongs to the branch scope."""
    return path in whole or line in lines.get(path, frozenset())
