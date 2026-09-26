#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Pinned typing gate for first-party Phydrax sources.

The gate combines every ty diagnostic with annotation completeness: every
function, method, and nested helper annotates all of its parameters (except
`self`/`cls`) and its return type, as reported by the pinned Ruff `ANN` rules.

`report` summarizes every current diagnostic and always succeeds. `check` fails
when any diagnostic exists or a first-party Python file carries a comment
addressed to a type checker other than ty.

A diagnostic is identified by its file, rule, enclosing class or function, and
normalized message.
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
    errors.extend(
        f"(x{count}) {key.describe()}" for key, count in sorted(diagnostics(root).items())
    )
    return tuple(errors)


def _module_name(file: str, /) -> str:
    parts = file.removesuffix(".py").split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


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
    parser = argparse.ArgumentParser(description="Pinned typing gate for Phydrax.")
    commands = parser.add_subparsers(dest="command", required=True)
    report_parser = commands.add_parser("report")
    report_parser.add_argument("--json", type=Path, default=None)
    commands.add_parser("check")
    arguments = parser.parse_args()
    match arguments.command:
        case "report":
            print("\n".join(report(ROOT, arguments.json)))
        case "check":
            errors = check_errors(ROOT)
            if errors:
                raise SystemExit("\n".join(errors))
        case command:
            raise SystemExit(f"unknown command {command!r}")


if __name__ == "__main__":
    main()
