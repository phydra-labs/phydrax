#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Deterministic, read-only AST candidates for avoidable memory growth."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
ALLOWLIST = ROOT / "docs/plans/memory-pattern-allowlist.tsv"
_REDUCTIONS = frozenset(
    {"sum", "mean", "max", "min", "prod", "any", "all", "argmin", "argmax", "top_k"}
)


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


def _name(node: ast.expr, aliases: dict[str, str], /) -> str:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        return f"{_name(node.value, aliases)}.{node.attr}"
    return ""


def _assigned(node: ast.AST, /) -> tuple[ast.expr | None, ast.expr | None]:
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
        return node.targets[0], node.value
    if isinstance(node, ast.AnnAssign):
        return node.target, node.value
    return None, None


def _expanded(node: ast.expr, bindings: dict[str, ast.expr], /) -> ast.expr:
    seen: set[str] = set()
    while isinstance(node, ast.Name) and node.id in bindings and node.id not in seen:
        seen.add(node.id)
        node = bindings[node.id]
    return node


def _eye(
    node: ast.expr, bindings: dict[str, ast.expr], aliases: dict[str, str], /
) -> bool:
    value = _expanded(node, bindings)
    return isinstance(value, ast.Call) and _name(value.func, aliases).rsplit(".", 1)[
        -1
    ] in {"eye", "identity"}


def _broadcast(node: ast.expr, /) -> tuple[str, tuple[int, ...]] | None:
    if not isinstance(node, ast.Subscript):
        return None
    items = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    positions = tuple(
        index
        for index, item in enumerate(items)
        if isinstance(item, ast.Constant) and item.value is None
    )
    if not positions:
        return None
    remaining = [
        ast.unparse(item)
        for item in items
        if not (isinstance(item, ast.Constant) and item.value is None)
    ]
    return ast.unparse(node.value) + "[" + ",".join(remaining) + "]", positions


def scan_source(source: str, path: str, /) -> list[Finding]:
    """Scan one source without importing it; paths are repository-relative POSIX paths."""
    tree = ast.parse(source, filename=path)
    aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                aliases[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                aliases[alias.asname or alias.name] = (
                    f"{node.module or ''}.{alias.name}".lstrip(".")
                )
    findings: list[Finding] = []
    payload_path = any(
        part in path
        for part in (
            "archive",
            "checkpoint",
            "resource",
            "runtime",
            "interchange",
            "_fingerprint",
            "_execution_workset",
        )
    )
    parents = {
        child: parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }
    owners: dict[ast.AST, ast.AST] = {}

    def owner(node: ast.AST) -> ast.AST:
        trail: list[ast.AST] = []
        resolved: ast.AST = tree
        while node in parents:
            if node in owners:
                resolved = owners[node]
                break
            trail.append(node)
            node = parents[node]
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                resolved = node
                break
        for child in trail:
            owners[child] = resolved
        return resolved

    loads: dict[tuple[ast.AST, str], list[ast.AST]] = {}
    scans: dict[tuple[ast.AST, str], int] = {}
    for child in ast.walk(tree):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
            loads.setdefault((owner(child), child.id), []).append(parents[child])
        scan_target, scan_value = _assigned(child)
        if (
            isinstance(scan_target, ast.Tuple)
            and len(scan_target.elts) == 2
            and isinstance(scan_target.elts[1], ast.Name)
            and isinstance(scan_value, ast.Call)
            and _name(scan_value.func, aliases) in {"jax.lax.scan", "lax.scan"}
        ):
            scans[owner(child), scan_target.elts[1].id] = scan_value.lineno

    def consumers(node: ast.AST) -> list[ast.AST]:
        parent = parents.get(node)
        target, value = _assigned(parent) if parent is not None else (None, None)
        if value is node and isinstance(target, ast.Name):
            return loads.get((owner(node), target.id), [])
        return [parent] if parent is not None else []

    def reduced(node: ast.AST, derivative: bool) -> bool:
        uses = consumers(node)
        if not uses:
            return False

        def accepted(use: ast.AST) -> bool:
            if (
                derivative
                and isinstance(use, ast.BinOp)
                and isinstance(use.op, ast.MatMult)
            ):
                return True
            if derivative and isinstance(use, ast.Subscript):
                items = (
                    use.slice.elts if isinstance(use.slice, ast.Tuple) else [use.slice]
                )
                return any(
                    isinstance(item, ast.Constant) and isinstance(item.value, int)
                    for item in items
                )
            if isinstance(use, ast.Attribute):
                use = parents.get(use, use)
            if isinstance(use, ast.Call):
                tail = _name(use.func, aliases).rsplit(".", 1)[-1]
                if derivative and tail in {"einsum", "contract"}:
                    return bool(
                        use.args
                        and isinstance(use.args[0], ast.Constant)
                        and use.args[0].value in {"ij,j->i", "ij,i->j", "ij,j", "ij,i"}
                    )
                return tail in (
                    {"dot", "vdot", "diag", "diagonal", "trace"}
                    if derivative
                    else _REDUCTIONS
                )
            return False

        return all(accepted(use) for use in uses)

    def add(node: ast.AST, scope: tuple[str, ...], rule: str, detail: str) -> None:
        if not isinstance(node, ast.expr | ast.stmt):
            raise ValueError(
                "Findings must be attached to a source expression or statement."
            )
        findings.append(
            Finding(path, node.lineno, ".".join(scope) or "<module>", rule, detail)
        )

    def visit(
        node: ast.AST, scope: tuple[str, ...], bindings: dict[str, ast.expr]
    ) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = (*scope, node.name)
            bindings = {}
        target, value = _assigned(node)
        if isinstance(target, ast.Name) and value is not None:
            bindings[target.id] = value
            array_dictionary = (
                isinstance(node, ast.AnnAssign)
                and (
                    isinstance(value, ast.Dict)
                    and not value.keys
                    or isinstance(value, ast.Call)
                    and not value.args
                    and not value.keywords
                )
                and any(
                    isinstance(part, ast.Name)
                    and "Array" in part.id
                    or isinstance(part, ast.Attribute)
                    and part.attr in {"ndarray", "Array"}
                    for part in ast.walk(node.annotation)
                )
            )
            if (
                not scope
                and ("cache" in target.id.lower() or array_dictionary)
                and (
                    isinstance(value, ast.Dict)
                    or isinstance(value, ast.Call)
                    and _name(value.func, aliases) in {"dict", "typing.Dict"}
                )
            ):
                add(
                    node,
                    (target.id,),
                    "array-cache",
                    "module-level dictionary cache requires payload/ownership review",
                )
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and len(scope) == 1:
            for decorator in node.decorator_list:
                head = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = _name(head, aliases)
                if name in {
                    "cache",
                    "functools.cache",
                    "lru_cache",
                    "functools.lru_cache",
                }:
                    add(decorator, scope, "array-cache", ast.unparse(decorator))
        if isinstance(node, ast.Call):
            name = _name(node.func, aliases)
            tail = name.rsplit(".", 1)[-1]
            if (
                tail == "solve"
                and (
                    any(_eye(arg, bindings, aliases) for arg in node.args[1:])
                    or any(
                        keyword.arg in {"b", "rhs"}
                        and _eye(keyword.value, bindings, aliases)
                        for keyword in node.keywords
                    )
                )
                or name
                in {
                    "numpy.linalg.inv",
                    "np.linalg.inv",
                    "jax.numpy.linalg.inv",
                    "jnp.linalg.inv",
                }
            ) and reduced(node, True):
                add(node, scope, "inverse-by-identity", ast.unparse(node.func))
            mapped = _expanded(node.func, bindings)
            if (
                isinstance(mapped, ast.Call)
                and _name(mapped.func, aliases).rsplit(".", 1)[-1] == "vmap"
                and any(_eye(arg, bindings, aliases) for arg in node.args)
                and reduced(node, True)
            ):
                add(node, scope, "vmap-identity", "mapped action over a dense identity")
            if name in {"jax.jacfwd", "jax.jacrev", "jax.hessian", "jax.jacobian"}:
                materialized = parents.get(node)
                if (
                    isinstance(materialized, ast.Call)
                    and materialized.func is node
                    and reduced(materialized, True)
                ):
                    add(node, scope, "dense-derivative", name)
            if tail == "tobytes" and payload_path:
                add(node, scope, "payload-tobytes", ast.unparse(node.func))
            if tail == "read_bytes" and payload_path:
                add(node, scope, "payload-read-bytes", ast.unparse(node.func))
            if tail == "getvalue" and payload_path:
                add(node, scope, "whole-buffer", ast.unparse(node.func))
            if (
                tail == "tolist"
                and payload_path
                and any(
                    word in ".".join(scope).lower()
                    for word in (
                        "identity",
                        "fingerprint",
                        "digest",
                        "payload",
                        "canonical",
                    )
                )
            ):
                add(node, scope, "identity-tolist", ast.unparse(node.func))
            if tail in _REDUCTIONS:
                operands = list(node.args)
                if isinstance(node.func, ast.Attribute):
                    operands.append(node.func.value)
                for operand in operands:
                    expanded = _expanded(operand, bindings)
                    if (
                        isinstance(expanded, ast.Subscript)
                        and isinstance(expanded.value, ast.Call)
                        and _name(expanded.value.func, aliases)
                        in {"jax.lax.scan", "lax.scan"}
                    ):
                        add(node, scope, "scan-reduce", "scan output immediately reduced")
                    if (
                        isinstance(operand, ast.Name)
                        and scans.get((owner(node), operand.id), node.lineno)
                        < node.lineno
                    ):
                        add(
                            node,
                            scope,
                            "scan-reduce",
                            f"scan output {operand.id} reduced",
                        )
                if payload_path and tail == "sum":
                    ancestor: ast.AST | None = parents.get(node)
                    while ancestor is not None and not isinstance(
                        ancestor, ast.FunctionDef | ast.AsyncFunctionDef
                    ):
                        if isinstance(ancestor, ast.For | ast.AsyncFor | ast.While):
                            if any(
                                isinstance(child, ast.Subscript)
                                and isinstance(child.slice, ast.Slice)
                                and child.slice.lower is None
                                and isinstance(child.slice.upper, ast.Name)
                                for argument in node.args
                                for child in ast.walk(argument)
                            ):
                                add(
                                    node,
                                    scope,
                                    "quadratic-prefix",
                                    "prefix sum recomputed inside a loop",
                                )
                            break
                        ancestor = parents.get(ancestor)
            if tail in {"nonzero", "argwhere", "csr_matrix", "coo_matrix", "csc_matrix"}:
                operands = list(node.args)
                if isinstance(node.func, ast.Attribute):
                    operands.append(node.func.value)
                for operand in operands:
                    expanded = _expanded(operand, bindings)
                    if (
                        isinstance(expanded, ast.Call)
                        and _name(expanded.func, aliases) in {"np.zeros", "numpy.zeros"}
                        and expanded.args
                        and isinstance(expanded.args[0], ast.Tuple)
                        and len(expanded.args[0].elts) == 2
                    ):
                        add(
                            node,
                            scope,
                            "dense-zero-sparse",
                            "dense two-axis zero allocation followed by sparse extraction",
                        )
                        break
        if isinstance(node, ast.BinOp | ast.Compare):
            operands = (
                [node.left, node.right]
                if isinstance(node, ast.BinOp)
                else [node.left, *node.comparators]
            )
            if len(operands) == 2:
                left, right = (_broadcast(operand) for operand in operands)
                if (
                    left is not None
                    and right is not None
                    and left[0] == right[0]
                    and left[1] != right[1]
                    and reduced(node, False)
                ):
                    add(node, scope, "pairwise-broadcast", ast.unparse(node))
        for child in ast.iter_child_nodes(node):
            visit(child, scope, bindings)

    visit(tree, (), {})
    return sorted(set(findings))


def audit(root: Path, /) -> list[Finding]:
    """Return sorted candidates under a package root (including nested modules)."""
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
