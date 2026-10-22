#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Select the branch scope shared by the static audit gates.

A gate scans policy-listed, added and untracked files in full and other changed
files only on the lines added or rewritten since a revision, so text the branch
never touched stays with its owners. A staged rename keeps the lines it moved
with their owners: git pairs it with its source at 50% similarity, whatever the
local diff configuration. ``tools/branch-scope.tsv`` lists each whole-file path
once; ``prose_test=1`` also includes a test in the prose gate's default scan.
Missing or invalid policy data refuses the scan, and so does a listed path that
neither exists nor was deleted since the revision.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path


POLICY = "tools/branch-scope.tsv"
ROOTS = ("phydrax", "tests", "tools")
_PATH = re.compile(rf"(?:{'|'.join(ROOTS)})/(?:[A-Za-z0-9_]+/)*[A-Za-z0-9_]+\.py")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
# Command-line options override configuration that would change which paths are
# renamed rather than added (diff.renames, diff.renameLimit) or which lines a hunk
# reports (prefixes, hunk merging, algorithm, external and textconv drivers).
_DIFF = (
    "diff",
    "--no-color",
    "--no-ext-diff",
    "--no-textconv",
    "--find-renames=50%",
    "-l0",
    "--diff-algorithm=myers",
    "--indent-heuristic",
    "--inter-hunk-context=0",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)


def _git(root: Path, /, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=root, check=True, capture_output=True, text=True
    ).stdout


def _python_files(root: Path, names: Iterable[str], /) -> set[Path]:
    return {
        root / name for name in names if name.endswith(".py") and (root / name).is_file()
    }


def _untracked(root: Path, /) -> list[str]:
    return _git(root, "ls-files", "--others", "--exclude-standard", "-z").split("\0")


def _changes(root: Path, revision: str, /) -> dict[str, str]:
    # One status letter per path; a rename source is recorded as deleted.
    fields = _git(root, *_DIFF, "--name-status", "-z", revision, "--").split("\0")
    changes: dict[str, str] = {}
    index = 0
    while index + 1 < len(fields):
        status = fields[index][0]
        if status == "R":
            changes[fields[index + 1]] = "D"
            index += 1
        changes[fields[index + 1]] = status
        index += 2
    return changes


def changed_files(root: Path, revision: str, /) -> set[Path]:
    """Return existing tracked changes and untracked Python files since a revision."""
    changes = _changes(root, revision)
    return _python_files(
        root,
        [
            *(name for name, status in changes.items() if status in "ACMRT"),
            *_untracked(root),
        ],
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
            _PATH.fullmatch(name) is None
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
    changes = _changes(root, revision)
    for line, name in enumerate(policy, start=2):
        if not (root / name).is_file() and changes.get(name) != "D":
            raise ValueError(
                f"BranchScope.policy must list existing or branch-deleted files; got {root / POLICY}:{line}"
            )
    return _python_files(
        root,
        [
            *policy,
            *(name for name, status in changes.items() if status == "A"),
            *_untracked(root),
        ],
    )


def changed_lines(root: Path, revision: str, /) -> dict[Path, frozenset[int]]:
    """Return the lines of each tracked file added or rewritten since a revision."""
    lines: dict[Path, set[int]] = {}
    current: Path | None = None
    diff = _git(root, *_DIFF, "-U0", revision, "--")
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
