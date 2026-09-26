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
from collections.abc import Iterator, Mapping
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


def _aliases(tree: ast.Module, /) -> dict[str, frozenset[object]]:
    aliases: dict[str, frozenset[object]] = {}
    for node in tree.body:
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            target, value = node.target, node.value
        if isinstance(target, ast.Name) and value is not None:
            members = _literal_members(value)
            if members:
                aliases[target.id] = members
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


def _module_name(path: Path, /) -> str:
    parts = path.relative_to(ROOT).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _import_owner(path: Path, node: ast.ImportFrom, /) -> str:
    if node.level == 0:
        return node.module or ""
    current = _module_name(path)
    package = current if path.name == "__init__.py" else current.rsplit(".", 1)[0]
    parts = package.split(".")
    prefix = parts[: len(parts) - node.level + 1]
    return ".".join((*prefix, *((node.module,) if node.module else ())))


def _visible_aliases(
    path: Path,
    tree: ast.Module,
    registry: Mapping[str, Mapping[str, frozenset[object]]],
    /,
) -> dict[str, frozenset[object]]:
    visible = dict(registry[_module_name(path)])
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        available = registry.get(_import_owner(path, node), {})
        for alias in node.names:
            if alias.name in available:
                visible[alias.asname or alias.name] = available[alias.name]
    return visible


def _findings(
    path: Path,
    registry: Mapping[str, Mapping[str, frozenset[object]]],
    /,
) -> Iterator[Finding]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    aliases_by_name = _visible_aliases(path, tree, registry)
    if not aliases_by_name:
        return
    aliases = {
        members: name
        for name, members in sorted(aliases_by_name.items())
        if len(members) >= 2
    }
    relative = path.relative_to(ROOT).as_posix()
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    membership_tables = {
        comparison.comparators[0].id
        for comparison in ast.walk(tree)
        if isinstance(comparison, ast.Compare)
        and len(comparison.ops) == 1
        and isinstance(comparison.ops[0], ast.In | ast.NotIn)
        and isinstance(comparison.comparators[0], ast.Name)
    }
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
                    if set(values) == members:
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
            target = node.targets[0] if isinstance(node, ast.Assign) else node.target
            if not isinstance(target, ast.Name) or target.id not in membership_tables:
                continue
            members = _constant_collection(node.value)
            if members is not None and members in aliases:
                yield Finding(
                    relative,
                    node.lineno,
                    "option-table",
                    aliases[members],
                    target.id,
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "parse"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Name)
            and node.args[1].id in aliases_by_name
            and isinstance(parents.get(node), ast.Expr)
            and isinstance(node.args[0], ast.Name)
        ):
            yield Finding(
                relative,
                node.lineno,
                "discarded-parse",
                node.args[1].id,
                ast.unparse(node.args[0]),
            )


def audit(root: Path, /) -> list[Finding]:
    paths = tuple(
        path for path in sorted(root.rglob("*.py")) if "__pycache__" not in path.parts
    )
    registry = {
        _module_name(path): _aliases(
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        )
        for path in paths
    }
    findings: list[Finding] = []
    for path in paths:
        findings.extend(_findings(path, registry))
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
