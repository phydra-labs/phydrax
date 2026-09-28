#
# Copyright © 2026 PHYDRA, Inc. All rights reserved.
#

"""Rank strict modules that would benefit from structural contracts.

The report is read-only. It considers dataclass-based `StrictModule` classes that
have not opted in (``__strict_contract__ = True``) and scores the static facts a
contract could state:

- closed selectors: fields annotated with a `Literal` alias;
- static sizes: `int` static fields whose names count or size something;
- aligned arrays: two or more dynamic array fields;
- identifier tuples: `tuple[str, ...]` static fields;
- host arrays: `numpy.ndarray` fields.

Classes with a generated constructor are listed first because they currently run
no validation. Classes whose `__check_init__` or constructor uses
`equinox.error_if` hold data-dependent numerical predicates and are marked, since
those stay with their owner.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "phydrax"
_SIZE_WORDS = ("count", "size", "dimension", "rank", "length", "order", "degree")


@dataclass(frozen=True, slots=True)
class Candidate:
    file: str
    line: int
    name: str
    generated_constructor: bool
    selectors: int
    sizes: int
    arrays: int
    identifier_tuples: int
    host_arrays: int
    numerical_predicates: bool

    @property
    def score(self) -> int:
        # Metadata facts are cheap and exact to state; array extents need nominal
        # dimensions chosen by the owner, so aligned arrays count once.
        return (
            3 * self.selectors
            + 3 * self.sizes
            + 3 * self.identifier_tuples
            + self.host_arrays
            + (1 if self.arrays >= 2 else 0)
        )


def _literal_aliases(tree: ast.Module, /) -> set[str]:
    aliases: set[str] = set()
    for node in tree.body:
        value = (
            node.value
            if isinstance(node, ast.Assign | ast.AnnAssign | ast.TypeAlias)
            else None
        )
        target = (
            node.targets[0]
            if isinstance(node, ast.Assign) and len(node.targets) == 1
            else node.target
            if isinstance(node, ast.AnnAssign)
            else node.name
            if isinstance(node, ast.TypeAlias)
            else None
        )
        if (
            isinstance(target, ast.Name)
            and isinstance(value, ast.Subscript)
            and ast.unparse(value.value) in ("Literal", "typing.Literal")
        ):
            aliases.add(target.id)
    return aliases


def _is_strict_module(node: ast.ClassDef, /) -> bool:
    return any(ast.unparse(base).endswith("StrictModule") for base in node.bases)


def _static(field: ast.AnnAssign, /) -> bool:
    value = field.value
    return (
        isinstance(value, ast.Call)
        and ast.unparse(value.func) in ("eqx.field", "field", "equinox.field")
        and any(
            keyword.arg == "static"
            and isinstance(keyword.value, ast.Constant)
            and keyword.value.value is True
            for keyword in value.keywords
        )
    )


def _candidate(path: Path, node: ast.ClassDef, aliases: set[str], /) -> Candidate | None:
    body = node.body
    if any(
        isinstance(item, ast.Assign)
        and any(ast.unparse(target) == "__strict_contract__" for target in item.targets)
        for item in body
    ):
        return None
    fields = [
        item
        for item in body
        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
    ]
    if not fields:
        return None
    methods = {item.name: item for item in body if isinstance(item, ast.FunctionDef)}
    selectors = sizes = arrays = identifier_tuples = host_arrays = 0
    for field in fields:
        annotation = ast.unparse(field.annotation).replace('"', "")
        name = field.target.id if isinstance(field.target, ast.Name) else ""
        head = annotation.split("[", 1)[0]
        if head in aliases or head in ("Literal", "typing.Literal"):
            selectors += 1
        elif (
            annotation == "int"
            and _static(field)
            and any(word in name for word in _SIZE_WORDS)
        ):
            sizes += 1
        elif annotation in ("Array", "jax.Array") and not _static(field):
            arrays += 1
        elif annotation in ("tuple[str, ...]", "Tuple[str, ...]"):
            identifier_tuples += 1
        elif head in ("np.ndarray", "numpy.ndarray", "npt.NDArray"):
            host_arrays += 1
    numerical = any(
        isinstance(call, ast.Call) and ast.unparse(call.func).endswith("error_if")
        for method in methods.values()
        for call in ast.walk(method)
    )
    candidate = Candidate(
        path.relative_to(ROOT).as_posix(),
        node.lineno,
        node.name,
        "__init__" not in methods,
        selectors,
        sizes,
        arrays,
        identifier_tuples,
        host_arrays,
        numerical,
    )
    return candidate if candidate.score else None


def audit(root: Path, /) -> list[Candidate]:
    candidates: list[Candidate] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        aliases = _literal_aliases(tree)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and _is_strict_module(node):
                candidate = _candidate(path, node, aliases)
                if candidate is not None:
                    candidates.append(candidate)
    return sorted(
        candidates,
        key=lambda item: (
            not item.generated_constructor,
            -item.score,
            item.file,
            item.line,
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank structural contract candidates.")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=50)
    arguments = parser.parse_args()
    candidates = audit(PACKAGE)
    if arguments.json is not None:
        arguments.json.write_text(
            json.dumps(
                [asdict(item) | {"score": item.score} for item in candidates], indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    for item in candidates[: arguments.limit]:
        print(
            f"{item.file}:{item.line}: {item.name} score={item.score}"
            f" generated={item.generated_constructor} selectors={item.selectors}"
            f" sizes={item.sizes} arrays={item.arrays}"
            f" identifier_tuples={item.identifier_tuples} host_arrays={item.host_arrays}"
            f" numerical_predicates={item.numerical_predicates}"
        )
    print(f"total: {len(candidates)} candidates")


if __name__ == "__main__":
    main()
