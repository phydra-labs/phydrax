#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report exact-rational elimination loops outside the canonical exact linear-algebra owner.

The report is read-only and never rewrites code. Exact linear algebra (solve,
inverse, rank, determinant, nullspace) belongs to `phydrax.linalg._exact`, which
owns the fast substrates (modular screens, p-adic lifting, sparse elimination);
hand-written `Fraction` elimination elsewhere bypasses them. The audit scans
modules that import `fractions` and every function that names `Fraction` or
`ExactScalar`, and reports the `fraction-elimination` rule per enclosing
function: a loop whose body contains an inner loop or comprehension, a row
update (`a[i][j] -= ...`, `a[i, j] = ...`, `rows[r] = [x - f * y ...]`,
`row[k] /= ...` or `row[k] -= f * ...`), and a pivot (a `pivot` name or a
division). Loops over a literal dimension of at most four entries
(`range(3)`, a 3-tuple) are the fixed tiny-system case and are not reported.

Modules under `phydrax/linalg/_exact*` are the owner and are not scanned. Every
finding needs an exact `path, symbol, rule, rationale` row in
`docs/plans/fraction-hotpath-allowlist.tsv`; the check fails on unallowlisted,
stale, duplicate, unsorted, or malformed rows.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
ALLOWLIST = ROOT / "docs/plans/fraction-hotpath-allowlist.tsv"
OWNER_PREFIX = "phydrax/linalg/_exact"
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_COMPREHENSIONS = (ast.ListComp, ast.GeneratorExp, ast.SetComp, ast.DictComp)
_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    file: str
    line: int
    symbol: str
    rule: str
    detail: str

    @property
    def key(self) -> tuple[str, str, str]:
        return self.file, self.symbol, self.rule


def _local(node: ast.AST, /) -> Iterator[ast.AST]:
    """Nodes of one scope, without descending into nested functions or classes."""
    stack = [node]
    while stack:
        current = stack.pop()
        yield current
        stack.extend(
            child
            for child in ast.iter_child_nodes(current)
            if not isinstance(child, _SCOPES)
        )


def _body(loop: ast.For | ast.AsyncFor | ast.While, /) -> list[ast.AST]:
    return [node for statement in loop.body for node in _local(statement)]


def _matrix_entry(node: ast.AST, /) -> bool:
    return isinstance(node, ast.Subscript) and (
        isinstance(node.value, ast.Subscript) or isinstance(node.slice, ast.Tuple)
    )


def _has_operator(node: ast.AST, operators: tuple[type[ast.operator], ...], /) -> bool:
    return any(
        isinstance(child, ast.BinOp) and isinstance(child.op, operators)
        for child in _local(node)
    )


def _has_axpy(node: ast.AST, /) -> bool:
    """Whether `node` contains the elimination shape `x - f * y`."""
    return any(
        isinstance(child, ast.BinOp)
        and isinstance(child.op, ast.Sub)
        and isinstance(child.right, ast.BinOp)
        and isinstance(child.right.op, ast.Mult)
        for child in _local(node)
    )


def _row_update(node: ast.AST, /) -> bool:
    if isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Subscript):
        # `a[i][j] -= f * a[k][j]`, sparse `row[k] /= pivot` or
        # `row[k] -= f * other[k]`: the subtrahend reads a pivot-row entry.
        # Additive accumulation (`total[k] += x * y`) and scalar shifts
        # (`x[i][j] -= n * period`) are not elimination.
        if isinstance(node.op, ast.Div):
            return not _matrix_entry(node.target)
        return (
            isinstance(node.op, ast.Sub)
            and _has_operator(node.value, (ast.Mult,))
            and any(isinstance(child, ast.Subscript) for child in _local(node.value))
        )
    if not isinstance(node, ast.Assign):
        return False
    for target in node.targets:
        if not isinstance(target, ast.Subscript):
            continue
        if _matrix_entry(target):
            if _has_axpy(node.value):
                return True
            continue
        value = node.value
        if isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if isinstance(value, _COMPREHENSIONS) and _has_axpy(value):
            return True
    return False


def _pivot(node: ast.AST, /) -> bool:
    return (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)) or (
        isinstance(node, ast.Name) and "pivot" in node.id.lower()
    )


_FIXED_DIMENSION = 4


def _fixed_small(loop: ast.AST, /) -> bool:
    """Whether the loop runs over a literal dimension of at most four entries.

    Fixed-dimension 2-, 3- and 4-component work (affine frames, homogeneous
    transforms, 3x3 lattice bases) is the tiny-system case that stays in
    `Fraction` arithmetic by design.
    """

    def literal(iterable: ast.expr) -> bool:
        if isinstance(iterable, ast.Tuple | ast.List):
            return len(iterable.elts) <= _FIXED_DIMENSION
        return (
            isinstance(iterable, ast.Call)
            and isinstance(iterable.func, ast.Name)
            and iterable.func.id == "range"
            and all(
                isinstance(argument, ast.Constant)
                and isinstance(argument.value, int)
                and abs(argument.value) <= _FIXED_DIMENSION
                for argument in iterable.args
            )
        )

    if isinstance(loop, ast.For | ast.AsyncFor):
        return literal(loop.iter)
    if isinstance(loop, _COMPREHENSIONS):
        return all(literal(generator.iter) for generator in loop.generators)
    return False


def elimination_loops(function: ast.AST, /) -> list[int]:
    """Line numbers of loops in `function` with the elimination signature."""
    lines = []
    for loop in _local(function):
        if not isinstance(loop, _LOOPS) or loop is function or _fixed_small(loop):
            continue
        body = _body(loop)
        if (
            any(isinstance(node, _LOOPS + _COMPREHENSIONS) for node in body)
            and any(_row_update(node) for node in body)
            and any(_pivot(node) for node in body)
        ):
            lines.append(loop.lineno)
    return lines


def _imports_fractions(tree: ast.Module, /) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "fractions":
            return True
        if isinstance(node, ast.Import) and any(
            alias.name == "fractions" for alias in node.names
        ):
            return True
    return False


def _names_exact_scalar(function: ast.AST, /) -> bool:
    return any(
        isinstance(node, ast.Name) and node.id in ("Fraction", "ExactScalar")
        for node in ast.walk(function)
    )


def scan_source(source: str, relative: str, /) -> list[Finding]:
    """Findings of one module given its repository-relative path."""
    if relative.startswith(OWNER_PREFIX):
        return []
    tree = ast.parse(source, filename=relative)
    rational_module = _imports_fractions(tree)
    findings: list[Finding] = []

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                visit(child, (*scope, child.name))
                continue
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qualified = (*scope, child.name)
                if rational_module or _names_exact_scalar(child):
                    lines = elimination_loops(child)
                    if lines:
                        findings.append(
                            Finding(
                                relative,
                                child.lineno,
                                ".".join(qualified),
                                "fraction-elimination",
                                "loops at lines " + ", ".join(map(str, lines)),
                            )
                        )
                visit(child, qualified)
                continue
            visit(child, scope)

    visit(tree, ())
    return sorted(findings)


def audit(root: Path, /) -> list[Finding]:
    """Sorted findings under a package root (including nested modules)."""
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            findings.extend(
                scan_source(
                    path.read_text(encoding="utf-8"),
                    path.relative_to(root.parent).as_posix(),
                )
            )
    return sorted(findings)


def check_allowlist(findings: list[Finding], path: Path, /) -> list[str]:
    """Fail closed on malformed, duplicate, stale, or missing exact dispositions."""
    errors: list[str] = []
    keys = {finding.key for finding in findings}
    allowed: set[tuple[str, str, str]] = set()
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        if next(reader, None) != ["path", "symbol", "rule", "rationale"]:
            return [
                "allowlist header must be path, symbol, rule, rationale (tab-separated)"
            ]
        previous: tuple[str, str, str] | None = None
        for number, row in enumerate(reader, 2):
            if (
                len(row) != 4
                or any(
                    not field.strip() or any(character in field for character in "\r\n\t")
                    for field in row
                )
                or any(character in row[0] for character in "*?[")
            ):
                errors.append(f"allowlist row {number}: invalid exact entry")
                continue
            key = (row[0], row[1], row[2])
            if key in allowed:
                errors.append(f"duplicate allowlist entry: {key}")
            if key not in keys:
                errors.append(f"stale allowlist entry: {key}")
            if previous is not None and key < previous:
                errors.append(f"unsorted allowlist entry: {key}")
            previous = key
            allowed.add(key)
    errors.extend(f"unallowlisted finding: {key}" for key in sorted(keys - allowed))
    return sorted(errors)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--root", type=Path, default=PACKAGE)
    parser.add_argument("--allowlist", type=Path, default=ALLOWLIST)
    arguments = parser.parse_args()
    findings = audit(arguments.root)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps([asdict(finding) for finding in findings], indent=2) + "\n",
            encoding="utf-8",
        )
    for finding in findings:
        print(
            f"{finding.file}:{finding.line}: {finding.rule} [{finding.symbol}] {finding.detail}"
        )
    errors = check_allowlist(findings, arguments.allowlist)
    for error in errors:
        print(error)
    print(
        f"total: {len(findings)} {dict(sorted(Counter(finding.rule for finding in findings).items()))}; errors: {len(errors)}"
    )
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
