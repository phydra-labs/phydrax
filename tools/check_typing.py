#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned typing gate for first-party Phydrax sources.

The gate combines every ty diagnostic with annotation completeness: every
function, method, and nested helper annotates all of its parameters (except
`self`/`cls`) and its return type, as reported by the pinned Ruff `ANN` rules.

`report` summarizes every current diagnostic and always succeeds. `check` fails when
a file outside the quarantine emits any diagnostic, when a quarantined file emits a
diagnostic that is not recorded, when a recorded diagnostic has disappeared, or when
a first-party Python file carries a non-ty checker comment. `update-quarantine` only
removes resolved records. `touched` requires every first-party file changed since a
base revision to be clean. `waves` orders quarantined modules so that foundations
are cleaned before the modules that import them.

A diagnostic is identified by its file, rule, enclosing class or function, and
normalized message; identities are counted as a multiset, so one fixed diagnostic
cannot hide a different new diagnostic of the same rule.
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import re
import subprocess
import sys
import tokenize
import tomllib
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "phydrax"
QUARANTINE = Path("tools/typing_quarantine.json")
ANNOTATION_RULES = "ANN001,ANN002,ANN003,ANN201,ANN202,ANN204,ANN205,ANN206"
_CHECKER_COMMENT = re.compile(r"#\s*(?:(?:type|pyright|mypy|pytype)\s*:|pyre-)")
_SOURCE_LOCATION = re.compile(r"(\.pyi?):\d+(?::\d+)?")


@dataclass(frozen=True, order=True, slots=True)
class DiagnosticKey:
    """Line-independent identity of one typing diagnostic."""

    file: str
    rule: str
    symbol: str
    message: str

    def describe(self) -> str:
        return f"{self.file} [{self.rule}] {self.symbol}: {self.message}"


def pinned_version(root: Path, tool: str, /) -> str:
    """Return the exact version of `tool` pinned by the `qa` extra."""
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    for requirement in project["project"]["optional-dependencies"]["qa"]:
        name, separator, version = requirement.partition("==")
        if name.strip() == tool and separator:
            return version.strip()
    raise SystemExit(f"pyproject.toml must pin {tool} exactly in the qa extra.")


def run_tool(
    root: Path, tool: str, arguments: Iterable[str], /
) -> subprocess.CompletedProcess[str]:
    """Run a pinned tool installed in the current interpreter environment."""
    return subprocess.run(
        [sys.executable, "-m", tool, *arguments],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )


def verify_pinned(root: Path, tool: str, /) -> None:
    """Refuse to run any other version of `tool` than the pinned one."""
    result = run_tool(root, tool, ("--version",))
    if result.returncode != 0:
        raise SystemExit(f"{tool} is not available:\n{result.stdout}{result.stderr}")
    words = result.stdout.split()
    found = words[1] if words[:1] == [tool] and len(words) > 1 else ""
    pinned = pinned_version(root, tool)
    if found != pinned:
        raise SystemExit(f"{tool} {pinned} is pinned; found {result.stdout.strip()!r}.")


def _symbol_spans(path: Path, /) -> tuple[tuple[int, int, str], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    spans: list[tuple[int, int, str]] = []

    def visit(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                name = f"{prefix}.{child.name}" if prefix else child.name
                end = child.end_lineno if child.end_lineno is not None else child.lineno
                start = min(
                    (decorator.lineno for decorator in child.decorator_list),
                    default=child.lineno,
                )
                spans.append((start, end, name))
                visit(child, name)
            else:
                visit(child, prefix)

    visit(tree, "")
    return tuple(spans)


def _enclosing_symbol(spans: tuple[tuple[int, int, str], ...], line: int, /) -> str:
    enclosing = "<module>"
    width = sys.maxsize
    for start, end, name in spans:
        if start <= line <= end and end - start < width:
            enclosing, width = name, end - start
    return enclosing


def _normalized_message(message: str, rule: str, root: Path, /) -> str:
    text = message.removeprefix(f"{rule}: ")
    for prefix, marker in ((str(root), "<root>"), (sys.prefix, "<env>")):
        text = text.replace(prefix, marker)
    return _SOURCE_LOCATION.sub(r"\1", " ".join(text.split()))


class _Collector:
    """Accumulate diagnostic identities with per-file enclosing-symbol lookup."""

    def __init__(self, root: Path, /) -> None:
        self.root = root.resolve()
        self.found: Counter[DiagnosticKey] = Counter()
        self._spans: dict[str, tuple[tuple[int, int, str], ...]] = {}

    def add(self, path: str, line: int, rule: str, message: str, /) -> None:
        location = Path(path)
        absolute = location if location.is_absolute() else self.root / location
        file = absolute.resolve().relative_to(self.root).as_posix()
        if file not in self._spans:
            self._spans[file] = _symbol_spans(self.root / file)
        symbol = _enclosing_symbol(self._spans[file], line)
        normalized = _normalized_message(message, rule, self.root)
        self.found[DiagnosticKey(file, rule, symbol, normalized)] += 1


def _collect_ty(collector: _Collector, /) -> None:
    root = collector.root
    verify_pinned(root, "ty")
    result = run_tool(
        root,
        "ty",
        ("check", "--output-format", "gitlab", "--python", sys.prefix, PACKAGE),
    )
    if result.returncode not in (0, 1):
        raise SystemExit(
            f"ty failed with exit code {result.returncode}:\n{result.stderr}"
        )
    for record in json.loads(result.stdout) if result.stdout.strip() else []:
        location = record["location"]
        collector.add(
            location["path"],
            location["positions"]["begin"]["line"],
            record["check_name"],
            record["description"],
        )


def _collect_annotations(collector: _Collector, /) -> None:
    root = collector.root
    verify_pinned(root, "ruff")
    result = run_tool(
        root,
        "ruff",
        (
            "check",
            "--select",
            ANNOTATION_RULES,
            "--output-format",
            "json",
            "--no-cache",
            "--exit-zero",
            PACKAGE,
        ),
    )
    if result.returncode != 0:
        raise SystemExit(
            f"ruff failed with exit code {result.returncode}:\n{result.stderr}"
        )
    for record in json.loads(result.stdout):
        collector.add(
            record["filename"],
            record["location"]["row"],
            record["code"],
            record["message"],
        )


def diagnostics(root: Path, /) -> Counter[DiagnosticKey]:
    """Return the multiset of current first-party typing diagnostics."""
    collector = _Collector(root)
    _collect_ty(collector)
    _collect_annotations(collector)
    return collector.found


def load_quarantine(root: Path, /) -> Counter[DiagnosticKey]:
    path = root / QUARANTINE
    if not path.is_file():
        return Counter()
    records = json.loads(path.read_text(encoding="utf-8"))
    return Counter(
        {
            DiagnosticKey(
                record["file"], record["rule"], record["symbol"], record["message"]
            ): record["count"]
            for record in records
        }
    )


def canonical_records(entries: Mapping[DiagnosticKey, int], /) -> list[dict[str, object]]:
    """Return diagnostic records in deterministic canonical order."""
    return [
        {
            "count": count,
            "file": key.file,
            "message": key.message,
            "rule": key.rule,
            "symbol": key.symbol,
        }
        for key, count in sorted(entries.items())
        if count > 0
    ]


def write_quarantine(root: Path, entries: Mapping[DiagnosticKey, int], /) -> None:
    (root / QUARANTINE).write_text(
        json.dumps(canonical_records(entries), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _git_lines(root: Path, arguments: Iterable[str], /) -> tuple[str, ...]:
    result = subprocess.run(
        ["git", *arguments], cwd=root, capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        raise SystemExit(f"git {' '.join(arguments)} failed:\n{result.stderr}")
    return tuple(line for line in result.stdout.splitlines() if line)


def first_party_python_files(root: Path, /) -> tuple[str, ...]:
    """Return tracked and unignored untracked Python files of the repository."""
    tracked = _git_lines(root, ("ls-files", "--", "*.py"))
    untracked = _git_lines(
        root, ("ls-files", "--others", "--exclude-standard", "--", "*.py")
    )
    return tuple(sorted({*tracked, *untracked}))


def checker_comment_errors(root: Path, /) -> tuple[str, ...]:
    """Reject comments addressed to type checkers other than ty."""
    errors: list[str] = []
    for file in first_party_python_files(root):
        path = root / file
        if not path.is_file():
            continue
        source = path.read_text(encoding="utf-8")
        if _CHECKER_COMMENT.search(source) is None:
            continue
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT and _CHECKER_COMMENT.match(token.string):
                errors.append(
                    f"non-ty checker comment: {file}:{token.start[0]}: {token.string}"
                )
    return tuple(errors)


def check_errors(root: Path, /) -> tuple[str, ...]:
    """Return every gate violation for the current tree."""
    errors = list(checker_comment_errors(root))
    current = diagnostics(root)
    quarantine = load_quarantine(root)
    quarantined_files = {key.file for key in quarantine}
    for key, count in sorted(current.items()):
        excess = count - quarantine[key]
        if excess > 0:
            origin = "new" if key.file in quarantined_files else "unquarantined"
            errors.append(f"{origin} (x{excess}): {key.describe()}")
    for key, count in sorted(quarantine.items()):
        if current[key] < count:
            errors.append(f"resolved, run update-quarantine: {key.describe()}")
    return tuple(errors)


def update_quarantine(root: Path, /) -> tuple[str, ...]:
    """Remove resolved records; refuse to record any unquarantined diagnostic."""
    current = diagnostics(root)
    quarantine = load_quarantine(root)
    write_quarantine(
        root, {key: min(count, current[key]) for key, count in quarantine.items()}
    )
    return tuple(
        f"refusing to quarantine (x{count - quarantine[key]}): {key.describe()}"
        for key, count in sorted(current.items())
        if count > quarantine[key]
    )


def touched_errors(root: Path, base: str, /) -> tuple[str, ...]:
    """Require every first-party Python file changed since `base` to be clean."""
    merge_base = _git_lines(root, ("merge-base", base, "HEAD"))[0]
    changed = _git_lines(root, ("diff", "--name-only", "--diff-filter=ACMR", merge_base))
    untracked = _git_lines(root, ("ls-files", "--others", "--exclude-standard"))
    touched = {
        file
        for file in (*changed, *untracked)
        if file.startswith(f"{PACKAGE}/") and file.endswith(".py")
    }
    current = diagnostics(root)
    return tuple(
        f"touched file is not clean (x{count}): {key.describe()}"
        for key, count in sorted(current.items())
        if key.file in touched
    )


def _module_name(file: str, /) -> str:
    parts = file.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _imported_modules(root: Path, file: str, known: frozenset[str], /) -> frozenset[str]:
    module = _module_name(file)
    package = module if file.endswith("__init__.py") else module.rpartition(".")[0]
    tree = ast.parse((root / file).read_text(encoding="utf-8"), filename=file)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            candidates = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                anchor = ".".join(
                    package.split(".")[: len(package.split(".")) - node.level + 1]
                )
                base = f"{anchor}.{node.module}" if node.module else anchor
            else:
                base = node.module or ""
            candidates = [base, *(f"{base}.{alias.name}" for alias in node.names)]
        else:
            continue
        imported.update(name for name in candidates if name in known)
    imported.discard(module)
    return frozenset(imported)


def _strongly_connected(graph: Mapping[str, frozenset[str]], /) -> list[frozenset[str]]:
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[frozenset[str]] = []
    for start in sorted(graph):
        if start in index:
            continue
        work = [(start, iter(sorted(graph[start])))]
        index[start] = low[start] = len(index)
        stack.append(start)
        on_stack.add(start)
        while work:
            node, successors = work[-1]
            advanced = False
            for successor in successors:
                if successor not in index:
                    index[successor] = low[successor] = len(index)
                    stack.append(successor)
                    on_stack.add(successor)
                    work.append((successor, iter(sorted(graph[successor]))))
                    advanced = True
                    break
                if successor in on_stack:
                    low[node] = min(low[node], index[successor])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: set[str] = set()
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.add(member)
                    if member == node:
                        break
                components.append(frozenset(component))
    return components


def waves(root: Path, /) -> tuple[str, ...]:
    """Order quarantined modules so imported foundations come first."""
    quarantine = load_quarantine(root)
    counts = Counter[str]()
    for key, count in quarantine.items():
        counts[_module_name(key.file)] += count
    files = tuple(
        file for file in first_party_python_files(root) if file.startswith(f"{PACKAGE}/")
    )
    known = frozenset(_module_name(file) for file in files)
    imports = {_module_name(file): _imported_modules(root, file, known) for file in files}
    dirty = frozenset(counts)
    graph = {module: imports[module] & dirty for module in dirty}
    clean_importers = Counter[str](
        target
        for module, targets in imports.items()
        if module not in dirty
        for target in targets
        if target in dirty
    )
    components = _strongly_connected(graph)
    owner = {
        module: position
        for position, members in enumerate(components)
        for module in members
    }
    level: dict[int, int] = {}
    for position, members in enumerate(components):
        dependencies = {
            owner[target] for module in members for target in graph[module]
        } - {position}
        level[position] = 1 + max(
            (level[dependency] for dependency in dependencies), default=-1
        )
    lines: list[str] = []
    for wave in sorted(set(level.values())):
        members = sorted(
            (module for module in dirty if level[owner[module]] == wave),
            key=lambda module: (-clean_importers[module], module),
        )
        lines.append(f"wave {wave}: {len(members)} modules")
        lines.extend(
            f"  {module} diagnostics={counts[module]}"
            f" clean_importers={clean_importers[module]}"
            for module in members
        )
    return tuple(lines)


def report(root: Path, json_path: Path | None, /) -> tuple[str, ...]:
    """Summarize current diagnostics by rule, package, file, and enclosing symbol."""
    current = diagnostics(root)
    by_rule: Counter[str] = Counter()
    by_package: Counter[str] = Counter()
    by_file: Counter[str] = Counter()
    by_symbol: Counter[str] = Counter()
    for key, count in current.items():
        by_rule[key.rule] += count
        by_package[".".join(_module_name(key.file).split(".")[:2])] += count
        by_file[key.file] += count
        by_symbol[f"{key.file}::{key.symbol}"] += count
    summary = {
        "diagnostics": canonical_records(current),
        "files": dict(sorted(by_file.items())),
        "packages": dict(sorted(by_package.items())),
        "rules": dict(sorted(by_rule.items())),
        "symbols": dict(sorted(by_symbol.items())),
        "total": sum(current.values()),
    }
    if json_path is not None:
        json_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    lines = [f"total: {summary['total']} diagnostics in {len(by_file)} files"]
    lines.extend(f"rule {name}: {count}" for name, count in _ranked(by_rule))
    lines.extend(f"package {name}: {count}" for name, count in _ranked(by_package))
    return tuple(lines)


def _ranked(counter: Counter[str], /) -> list[tuple[str, int]]:
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Pinned ty gate for Phydrax.")
    commands = parser.add_subparsers(dest="command", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--json", type=Path, default=None)
    commands.add_parser("check")
    commands.add_parser("update-quarantine")
    touched_parser = commands.add_parser("touched")
    touched_parser.add_argument("--base", required=True)
    commands.add_parser("waves")
    arguments = parser.parse_args()
    match arguments.command:
        case "report":
            print("\n".join(report(ROOT, arguments.json)))
            return
        case "waves":
            print("\n".join(waves(ROOT)))
            return
        case "check":
            errors = check_errors(ROOT)
        case "update-quarantine":
            errors = update_quarantine(ROOT)
        case "touched":
            errors = touched_errors(ROOT, arguments.base)
        case command:
            raise SystemExit(f"unknown command {command!r}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
