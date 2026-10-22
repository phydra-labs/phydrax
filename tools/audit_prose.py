#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Report lexicon and error-message violations in the selected audit scope.

Docstrings, exception messages, and comments are scanned without importing the
package. By default, error formatting is checked in branch-changed files only;
explicit paths check all messages. With ``--changed-since REV``, the default
selection also takes every policy-listed, added and untracked Python file under
``phydrax/``, ``tests/`` and ``tools/``; those files are scanned whole and other
changed files only on the lines added or rewritten since REV. Findings are
ordered by file and location.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import re
import tokenize
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

from tools._branch_scope import (
    changed_files,
    changed_lines,
    in_scope,
    load_policy,
    ROOTS,
    whole_files,
)


ROOT = Path(__file__).resolve().parents[1]
BASE_REVISION = "d2dfcb941"
INTENSIFIERS = frozenset({"actually", "genuine", "genuinely"})
_BUILTIN_EXCEPTIONS = frozenset(
    name
    for name, value in vars(builtins).items()
    if isinstance(value, type) and issubclass(value, BaseException)
)
RESTRICTED = frozenset(
    {
        "actual",
        "authoritative",
        "owning",
        "original",
        "witness",
        "receipt",
        "premise",
        "theorem",
        "authority",
        "complete",
        "canonical",
        "exact",
    }
)


@dataclass(frozen=True, slots=True)
class AllowedPhrase:
    pattern: str
    reason: str
    context: str = ""
    declared_symbol: bool = False


# Context requirements prevent a domain noun from licensing an unrelated claim.
ALLOWLIST = (
    AllowedPhrase(
        r"\bactual (?:work|consumption|queries|bytes)\b",
        "Consumption contrasted with a resource admission bound.",
        r"\b(?:upper bounds?|admitted|reserved|reservation)\b",
    ),
    AllowedPhrase(r"\bsource authority\b", "Source provenance domain term."),
    AllowedPhrase(
        r"\b(?:Authority|Witness|Receipt)\b|`(?:authority|witness|receipt)`|"
        r"\b[A-Za-z_]\w*\.(?:authority|witness|receipt|complete|exact|canonical|"
        r"original|premise|theorem)\b",
        "Exact spelling of a declared public type or a field of a declared public type.",
        declared_symbol=True,
    ),
    AllowedPhrase(
        r"\b(?:[A-Z][\w'’-]*[ -]){1,4}theorem\b|\b(?:Theorem|Lemma|Proposition|Corollary) "
        r"\d+(?:\.\d+)*\b",
        "Named mathematical theorem or numbered citation locator.",
    ),
    AllowedPhrase(r"\bexact predicates?\b", "Geometric predicate precision."),
    AllowedPhrase(r"\bexact arithmetic\b", "Arithmetic representation."),
    AllowedPhrase(
        r"\bexact (?:rationals?|integers?|orientations?|signs?|division|zeros?|"
        r"determinants?|solves?|cancellation)\b",
        "Exactly computed mathematical quantity, not emphasis.",
    ),
    AllowedPhrase(
        r"\bexact[ -](?:(?:source[- ]owned|homogeneous|rational|algebraic|affine|"
        r"barycentric|coefficient|reference|quotient|squared|P1|source|vertex|child|"
        r"parent|coarse-simplex|polynomial|level-set|lattice|edge|length) )*"
        r"(?:roots?|positions?|coordinates?|corners?|points?|images?|rows?|banks?|"
        r"weights?|stencils?|actions?|maps?|charts?|sources?|geometry|lengths?|"
        r"barycentrics|sums?|translations?|squares?|endpoints?|subdivision|"
        r"restrictions?|vertices|trajectories)\b",
        "Unrounded mathematical value contrasted with its rounded binary64 carrier.",
        r"\b(?:binary64|round(?:ed|ing|s)?|round-to-nearest|RNE|carriers?|dyadic|"
        r"Fractions?|rational|inexact|bit length)\b",
    ),
    AllowedPhrase(r"\bcanonical ordering\b", "Deterministic record ordering."),
    AllowedPhrase(r"\bcanonical fingerprints?\b", "Deterministic identity encoding."),
    AllowedPhrase(
        r"\boriginal [a-z][a-z_]*\b",
        "Input contrasted with a named derived representation.",
        r"\b(?:lifted|transformed|mapped|derived|projected|refined|copied|wrapped|"
        r"compiled|converted|reconstructed|normalized|adapted|offset) [a-z][a-z_]*\b",
    ),
)
_WORD = re.compile(r"\b[A-Za-z]+\b")
# A qualified field path whose components may be interpolated (`Owner.{value}`).
_PATH_PART = r"(?:[A-Za-z_]\w*|\{[^{}]+\})"
_MESSAGE = re.compile(
    rf"^(?:{_PATH_PART}(?:\.{_PATH_PART})+|\{{[^{{}}]+\}}) "
    r"must \S[^\n]*; got \S[^\n]*$"
)


@dataclass(frozen=True, order=True, slots=True)
class Finding:
    file: str
    line: int
    column: int
    kind: str
    context: str
    detail: str


def scope_paths() -> list[Path]:
    """Return package, meshing tools, and policy-selected tests."""
    paths = [ROOT / "phydrax", ROOT / "tests/unit/meshing"]
    paths.extend(sorted((ROOT / "tools").glob("meshing_qualification*")))
    paths.extend(
        ROOT / "tools" / name
        for name in (
            "meshing_benchmarks.py",
            "_meshing_cases.py",
            "native_moving_overset.py",
            "meshing_design_source.py",
        )
    )
    paths.extend(
        ROOT / name for name, prose_test in load_policy(ROOT).items() if prose_test
    )
    return paths


def _files(paths: Sequence[Path], /) -> list[Path]:
    files: set[Path] = set()
    for path in paths:
        if path.is_dir():
            files.update(path.rglob("*.py"))
        elif path.is_file() and path.suffix == ".py":
            files.add(path)
    return sorted(path.resolve() for path in files if "__pycache__" not in path.parts)


def _public_names(files: Sequence[Path], /) -> frozenset[str]:
    # Public classes and their annotated fields; a restricted word is admitted only
    # in a declared-symbol spelling (capitalized type, backticks, or `owner.field`).
    names: set[str] = set()
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                names.add(node.name)
                for field in node.body:
                    if isinstance(field, ast.AnnAssign) and isinstance(
                        field.target, ast.Name
                    ):
                        names.add(field.target.id)
    return frozenset(names)


def _lexicon(
    text: str,
    file: str,
    line: int,
    column: int,
    context: str,
    public_names: frozenset[str],
    /,
) -> list[Finding]:
    allowed: list[tuple[int, int]] = []
    for phrase in ALLOWLIST:
        if not phrase.context or re.search(phrase.context, text, re.IGNORECASE):
            allowed.extend(
                match.span()
                for match in re.finditer(
                    phrase.pattern,
                    text,
                    0 if phrase.declared_symbol else re.IGNORECASE,
                )
                if not phrase.declared_symbol
                or _declared(match[0].strip("`"), public_names)
            )
    findings: list[Finding] = []
    for word in _WORD.finditer(text):
        value = word[0].lower()
        if value not in INTENSIFIERS and value not in RESTRICTED:
            continue
        # An all-uppercase token names an enum member or selector value (`EXACT`).
        if len(word[0]) > 1 and word[0].isupper():
            continue
        if value in RESTRICTED and any(
            start <= word.start() and word.end() <= end for start, end in allowed
        ):
            continue
        prefix = text[: word.start()]
        row = line + prefix.count("\n")
        col = len(prefix.rsplit("\n", 1)[-1]) + (column if "\n" not in prefix else 0)
        findings.append(
            Finding(
                file,
                row,
                col + 1,
                "intensifier" if value in INTENSIFIERS else "restricted-word",
                context,
                word[0],
            )
        )
    return findings


def _declared(name: str, public_names: frozenset[str], /) -> bool:
    # `Owner.field` is declared when its owner is public; a bare word must be public.
    # Declarations outside the audited files come from the package registry.
    parts = name.split(".")
    return any(
        parts[-1] in names or (len(parts) > 1 and parts[0] in names)
        for names in (public_names, _package_public_names())
    )


@cache
def _package_public_names() -> frozenset[str]:
    # Only an allowance source: a package file that does not parse contributes no
    # names, which can only add findings, never hide one.
    parsable = []
    for path in _files([ROOT / "phydrax"]):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        parsable.append(path)
    return _public_names(parsable)


def _message_text(expression: ast.expr, /) -> str | None:
    match expression:
        case ast.Constant(value=str() as value):
            return value
        case ast.JoinedStr(values=values):
            return "".join(
                value.value
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
                else "{value}"
                if isinstance(value, ast.FormattedValue)
                else ""
                for value in values
            )
        case ast.BinOp(left=left, op=ast.Add(), right=right):
            first, second = _message_text(left), _message_text(right)
            if first is None and second is None:
                return None
            return (first if first is not None else "{value}") + (
                second if second is not None else "{value}"
            )
        case ast.Call(func=ast.Attribute(value=value, attr="format")):
            return _message_text(value)
        case ast.BinOp(left=left, op=ast.Mod()):
            return _message_text(left)
        case _:
            return None


def _texts(tree: ast.Module, /) -> list[tuple[ast.expr, str]]:
    texts: dict[ast.expr, str] = {}
    raised = {node.exc for node in ast.walk(tree) if isinstance(node, ast.Raise)}
    exception_names = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef)
        and any(
            ast.unparse(base).endswith(("Error", "Exception", "Warning"))
            for base in node.bases
        )
    }
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                texts[first.value] = "docstring"
        if isinstance(node, ast.Call):
            name = ast.unparse(node.func).rsplit(".", 1)[-1]
            if (
                node in raised
                or name.endswith(("Error", "Exception", "Warning"))
                or name in exception_names
                or name in _BUILTIN_EXCEPTIONS
            ):
                for argument in (
                    *node.args,
                    *(
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg in ("message", "msg")
                    ),
                ):
                    text = _message_text(argument)
                    # A whitespace-free constant is a selector or field name
                    # (resource, status, check code), never a prose message.
                    if text is not None and any(char.isspace() for char in text):
                        # A raised factory call (not an exception type) assembles
                        # its message from parts; only the lexicon applies to them.
                        factory = not (
                            name.endswith(("Error", "Exception", "Warning"))
                            or name in exception_names
                            or name in _BUILTIN_EXCEPTIONS
                        )
                        texts[argument] = "exception-part" if factory else "exception"
    return list(texts.items())


_THEOREM_BLOCK = re.compile(
    r"^\s*Mesh certificate checks\s*\n\s*-+\s*\n.*?^\s*End mesh certificate checks\s*$",
    re.MULTILINE | re.DOTALL,
)


def _audit_file(
    path: Path, public_names: frozenset[str], messages: bool, /
) -> list[Finding]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    file = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    findings: list[Finding] = []
    for expression, context in _texts(tree):
        text = _message_text(expression)
        if text is None:
            continue
        if context == "docstring":
            # Certificate-register metadata encodes mathematical hypotheses;
            # blank nonspace characters without changing finding offsets.
            text = _THEOREM_BLOCK.sub(lambda block: re.sub(r"\S", " ", block[0]), text)
        findings.extend(
            _lexicon(
                text,
                file,
                expression.lineno,
                expression.col_offset,
                context,
                public_names,
            )
        )
        if (
            context == "exception"
            and messages
            and _MESSAGE.fullmatch(text.strip()) is None
        ):
            findings.append(
                Finding(
                    file,
                    expression.lineno,
                    expression.col_offset + 1,
                    "message-format",
                    context,
                    "Expected <qualified.field> must <condition>; got <observed summary>",
                )
            )
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            findings.extend(
                _lexicon(token.string, file, *token.start, "comment", public_names)
            )
    return findings


def audit(
    paths: Sequence[Path] | None = None, /, *, changed_since: str | None = None
) -> tuple[list[Finding], int]:
    """Return findings and scanned-file count for the selected scope."""
    files = _files(scope_paths() if paths is None else paths)
    whole: set[Path] = set()
    lines: dict[Path, frozenset[int]] = {}
    changed: set[Path] | None = None
    if changed_since is not None:
        changed = changed_files(ROOT, changed_since)
        whole = whole_files(ROOT, changed_since)
        lines = changed_lines(ROOT, changed_since)
        if paths is None:
            # Branch-authored files enter whole; other changed files stay
            # limited to the default selection.
            authored = [
                path for path in whole if path.relative_to(ROOT).parts[0] in ROOTS
            ]
            files = sorted({*files, *_files(authored)})
    public_names = _public_names(files)
    if changed is not None:
        files = [path for path in files if path in changed]
    message_files = (
        set(files)
        if paths is not None or changed_since is not None
        else changed_files(ROOT, BASE_REVISION)
    )
    findings = [
        finding
        for path in files
        for finding in _audit_file(path, public_names, path in message_files)
        if changed_since is None or in_scope(path, finding.line, whole, lines)
    ]
    return sorted(findings), len(files)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--json", action="store_true", help="emit findings and counts as JSON"
    )
    parser.add_argument(
        "--changed-since",
        metavar="REV",
        help="scan files changed since REV, including untracked files",
    )
    arguments = parser.parse_args(argv)
    findings, count = audit(
        arguments.paths or None, changed_since=arguments.changed_since
    )
    if arguments.json:
        print(
            json.dumps(
                {
                    "files_scanned": count,
                    "count": len(findings),
                    "findings": [asdict(item) for item in findings],
                },
                indent=2,
            )
        )
    else:
        for finding in findings:
            print(
                f"{finding.file}:{finding.line}:{finding.column}: {finding.kind} ({finding.context}): {finding.detail}"
            )
        print(f"total: {len(findings)} findings in {count} files")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
