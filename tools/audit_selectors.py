#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report closed-selector duplication in first-party package code.

The report is read-only. For every module-level `Literal` alias it lists:

- tuples, lists, sets, or frozensets of string/int constants whose members equal
  the alias members (a duplicated option table);
- membership tests (`x in (...)`, `x not in (...)`) against an inline constant
  collection with exactly the alias members;
- `if`/`elif` chains of three or more equality tests of one name against alias
  members (candidates for an exhaustive `match`).

Selectors are matched by their member sets; a context-specific subset of an alias
is reported separately and is not a duplicate.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    file: str
    line: int
    kind: str
    alias: str
    detail: str


def _literal_members(node: ast.expr, /) -> frozenset[object] | None:
    if not isinstance(node, ast.Subscript):
        return None
    head = ast.unparse(node.value)
    if head not in ("Literal", "typing.Literal"):
        return None
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if not all(isinstance(element, ast.Constant) for element in elements):
        return None
    return frozenset(
        element.value for element in elements if isinstance(element, ast.Constant)
    )


def _constant_collection(node: ast.expr, /) -> frozenset[object] | None:
    if not isinstance(node, ast.Tuple | ast.List | ast.Set):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) in ("frozenset", "set", "tuple")
            and len(node.args) == 1
        ):
            return _constant_collection(node.args[0])
        return None
    if len(node.elts) < 2 or not all(isinstance(e, ast.Constant) for e in node.elts):
        return None
    return frozenset(e.value for e in node.elts if isinstance(e, ast.Constant))


def _aliases(tree: ast.Module, /) -> dict[frozenset[object], str]:
    aliases: dict[frozenset[object], str] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and value is not None:
            members = _literal_members(value)
            if members is not None and len(members) >= 2:
                aliases.setdefault(members, target.id)
    return aliases


def _equality_chain(node: ast.If, /) -> tuple[str, list[object]] | None:
    name: str | None = None
    values: list[object] = []
    current: ast.stmt | None = node
    while isinstance(current, ast.If):
        test = current.test
        if not (
            isinstance(test, ast.Compare)
            and len(test.ops) == 1
            and isinstance(test.ops[0], ast.Eq)
            and isinstance(test.comparators[0], ast.Constant)
        ):
            break
        subject = ast.unparse(test.left)
        if name is None:
            name = subject
        elif subject != name:
            break
        values.append(test.comparators[0].value)
        current = current.orelse[0] if len(current.orelse) == 1 else None
    if name is None or len(values) < 3:
        return None
    return name, values


def _findings(path: Path, /) -> Iterator[Finding]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases = _aliases(tree)
    if not aliases:
        return
    relative = path.relative_to(ROOT).as_posix()
    chained: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and id(node) not in chained:
            chain = _equality_chain(node)
            if chain is not None:
                subject, values = chain
                current: ast.stmt | None = node
                while isinstance(current, ast.If):
                    chained.add(id(current))
                    current = current.orelse[0] if len(current.orelse) == 1 else None
                for members, alias in aliases.items():
                    if set(values) <= members:
                        yield Finding(
                            relative, node.lineno, "equality-chain", alias, subject
                        )
                        break
        if isinstance(node, ast.Compare) and len(node.ops) == 1:
            if isinstance(node.ops[0], ast.In | ast.NotIn):
                members = _constant_collection(node.comparators[0])
                if members is not None and members in aliases:
                    yield Finding(
                        relative,
                        node.lineno,
                        "membership",
                        aliases[members],
                        ast.unparse(node.left),
                    )
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            members = _constant_collection(node.value)
            if members is not None and members in aliases:
                target = node.targets[0] if isinstance(node, ast.Assign) else node.target
                yield Finding(
                    relative,
                    node.lineno,
                    "option-table",
                    aliases[members],
                    ast.unparse(target),
                )


def audit(root: Path, /) -> list[Finding]:
    findings: list[Finding] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" not in path.parts:
            findings.extend(_findings(path))
    return sorted(findings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report closed-selector duplication.")
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()
    findings = audit(PACKAGE)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps([asdict(finding) for finding in findings], indent=2) + "\n",
            encoding="utf-8",
        )
    counts: dict[str, int] = {}
    for finding in findings:
        counts[finding.kind] = counts.get(finding.kind, 0) + 1
        print(
            f"{finding.file}:{finding.line}: {finding.kind} "
            f"[{finding.alias}] {finding.detail}"
        )
    print(f"total: {len(findings)} {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
