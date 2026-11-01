#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools import audit_fraction_hotpaths as audit


_HEADER = "from fractions import Fraction\n"


@pytest.mark.parametrize(
    "body",
    [
        # Dense Gauss-Jordan on list-of-lists rows.
        """def solve(rows, size):
    for column in range(size):
        pivot = next(row for row in range(column, size) if rows[row][column])
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for row in range(size):
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
""",
        # In-place LU update of matrix entries.
        """def det(rows, size):
    for column in range(size):
        for row in range(column + 1, size):
            scale = rows[row][column] / rows[column][column]
            for inner in range(column + 1, size):
                rows[row][inner] -= scale * rows[column][inner]
""",
        # Sparse dictionary column reduction.
        """def rank(columns):
    basis = {}
    for source in columns:
        vector = dict(source)
        while vector:
            pivot = max(vector)
            owner = basis.get(pivot)
            if owner is None:
                for index in tuple(vector):
                    vector[index] /= vector[pivot]
                basis[pivot] = vector
                break
""",
    ],
)
def test_exact_elimination_loops_are_reported(body: str) -> None:
    findings = audit.scan_source(_HEADER + body, "phydrax/geometry/_example.py")
    assert [finding.rule for finding in findings] == ["fraction-elimination"]


@pytest.mark.parametrize(
    "body",
    [
        # Additive accumulation is not elimination.
        """def measures(rows, count):
    total = [Fraction(0)] * count
    for row in rows:
        for region in row:
            total[region] += row[region] * row[0] / 2
""",
        # De Casteljau subdivision is polynomial evaluation.
        """def split(values, t):
    for order in range(1, len(values)):
        for index in range(len(values) - 1, order - 1, -1):
            alpha = t / (index + 1)
            values[index] = tuple((1 - alpha) * a + alpha * b for a, b in values)
""",
        # Fixed three-component frames are the tiny-system case.
        """def solve3(rows):
    for column in range(3):
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for row in range(3):
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
""",
        # Lattice shifts subtract a scalar multiple, not a pivot row.
        """def lift(lifted, vertices, period):
    for vertex in vertices:
        for axis in range(2):
            turns = int(lifted[vertex][axis] / period)
            lifted[vertex][axis] -= turns * period
""",
    ],
)
def test_accumulation_evaluation_and_fixed_frames_are_not_reported(body: str) -> None:
    assert audit.scan_source(_HEADER + body, "phydrax/geometry/_example.py") == []


def test_exact_owner_and_non_rational_modules_are_not_scanned() -> None:
    body = """def solve(rows, size):
    for column in range(size):
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for row in range(size):
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
"""
    assert audit.scan_source(_HEADER + body, "phydrax/linalg/_exact.py") == []
    assert audit.scan_source(body, "phydrax/numerics/_float_solve.py") == []
    typed = body.replace("def solve(rows, size):", "def solve(rows: ExactScalar, size):")
    assert [
        finding.symbol
        for finding in audit.scan_source(typed, "phydrax/geometry/_typed.py")
    ] == ["solve"]


def test_exact_allowlist_rejects_stale_duplicate_and_unsorted_entries(
    tmp_path: Path,
) -> None:
    findings = [
        audit.Finding("phydrax/a.py", 8, "f", "fraction-elimination", "loops at lines 9")
    ]
    header = "path\tsymbol\trule\trationale\n"
    entry = "phydrax/a.py\tf\tfraction-elimination\tTiny fixed system.\n"
    allowlist = tmp_path / "allowlist.tsv"
    allowlist.write_text(header + entry, encoding="utf-8")
    assert audit.check_allowlist(findings, allowlist) == []
    allowlist.write_text(header + entry + entry, encoding="utf-8")
    assert any(
        "duplicate" in error for error in audit.check_allowlist(findings, allowlist)
    )
    allowlist.write_text(header + entry.replace("\tf\t", "\tg\t"), encoding="utf-8")
    errors = audit.check_allowlist(findings, allowlist)
    assert any("stale" in error for error in errors)
    assert any("unallowlisted" in error for error in errors)
    allowlist.write_text(header + entry.replace("a.py", "z.py") + entry, encoding="utf-8")
    assert any(
        "unsorted" in error for error in audit.check_allowlist(findings, allowlist)
    )
    allowlist.write_text(header + entry.replace("a.py", "*.py"), encoding="utf-8")
    assert any(
        "invalid exact" in error for error in audit.check_allowlist(findings, allowlist)
    )
    allowlist.write_text("path\tsymbol\trule\n", encoding="utf-8")
    assert any("header" in error for error in audit.check_allowlist(findings, allowlist))


def test_cli_json_is_stable_and_unallowlisted_findings_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    package = tmp_path / "phydrax"
    package.mkdir()
    (package / "solver.py").write_text(
        _HEADER
        + """def solve(rows, size):
    for column in range(size):
        rows[column] = [value / rows[column][column] for value in rows[column]]
        for row in range(size):
            factor = rows[row][column]
            rows[row] = [a - factor * b for a, b in zip(rows[row], rows[column])]
""",
        encoding="utf-8",
    )
    allowlist = tmp_path / "allowlist.tsv"
    allowlist.write_text("path\tsymbol\trule\trationale\n", encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_fraction_hotpaths",
            "--root",
            str(package),
            "--allowlist",
            str(allowlist),
            "--json",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as failure:
        audit.main()
    assert failure.value.code == 1
    assert "unallowlisted" in capsys.readouterr().out
    assert json.loads(report.read_text(encoding="utf-8")) == [
        {
            "file": "phydrax/solver.py",
            "line": 2,
            "symbol": "solve",
            "rule": "fraction-elimination",
            "detail": "loops at lines 3",
        }
    ]


def test_repository_fraction_elimination_has_exact_dispositions() -> None:
    findings = audit.audit(audit.PACKAGE)
    assert findings == sorted(findings)
    assert audit.check_allowlist(findings, audit.ALLOWLIST) == []
